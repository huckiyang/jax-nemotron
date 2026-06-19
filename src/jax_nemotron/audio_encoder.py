"""
HF-faithful Parakeet FastConformer audio encoder for Nemotron-3-Nano-Omni (Flax NNX).

This is a structurally-faithful JAX/NNX port of the HuggingFace ``ParakeetEncoder``
that ships inside the Nemotron-3-Nano-Omni checkpoint under the
``sound_encoder.encoder.*`` namespace. Every learnable submodule attribute name and
leaf is chosen so that the ``nnx.eval_shape`` flat slash-path tree lines up 1:1 with
the HF safetensors tensor names (the milestone-2 converter then only has to translate
the leaf-segment vocabulary — Linear ``kernel``<->``weight``, LayerNorm ``scale``<->
``weight`` — and apply per-leaf transposes; the *paths* already match).

------------------------------------------------------------------------------
Architecture (real config: hidden 1024, 8 heads x 128, 24 layers, d_ff 4096)
------------------------------------------------------------------------------

    waveform
      -> feature_extractor (log-mel via STORED featurizer.fb / window; NO pre-emphasis)
      -> subsampling   (2-D Conv2d depthwise/pointwise stack, 8x time downsample,
                        flatten freq, final Linear 4096 -> 1024)
      -> 24 x ConformerLayer (POST-norm macaron):
             x = x + 0.5 * FF1(norm_feed_forward1(x))            # macaron FF1
             x = x + SelfAttn(norm_self_att(x))                  # Transformer-XL rel-pos
             x = x + Conv(norm_conv(x))                          # GLU + depthwise + BN
             x = x + 0.5 * FF2(norm_feed_forward2(x))            # macaron FF2
             x = norm_out(x)                                     # final LayerNorm
      -> hidden states (B, T_out, 1024)

Each Conformer layer's submodules and leaves map to HF as:

    norm_feed_forward1.{weight,bias}   LayerNorm   (norm_feed_forward1.{scale,bias})
    feed_forward1.linear1  Linear 1024->4096       (feed_forward1.linear1.kernel)
    feed_forward1.linear2  Linear 4096->1024       (feed_forward1.linear2.kernel)
    norm_self_att.{weight,bias}        LayerNorm
    self_attn.q_proj / k_proj / v_proj / o_proj    Linear 1024->1024 (no bias)
    self_attn.relative_k_proj          Linear 1024->1024 (no bias)
    self_attn.bias_u / bias_v          Param (8,128)
    norm_conv.{weight,bias}            LayerNorm
    conv.pointwise_conv1   Conv1d 1024->2048 k1     (then GLU -> 1024)
    conv.depthwise_conv    Conv1d depthwise k=9 groups=1024
    conv.norm              BatchNorm1d (running_mean/var used at inference)
    conv.pointwise_conv2   Conv1d 1024->1024 k1
    norm_feed_forward2.{weight,bias}   LayerNorm
    feed_forward2.linear1 / linear2    Linear (same as FF1)
    norm_out.{weight,bias}             LayerNorm

------------------------------------------------------------------------------
Faithfulness notes
------------------------------------------------------------------------------
* POST-norm macaron (norm applied to the *input* of each sub-block; the residual
  adds the raw sub-block output), matching HF ParakeetEncoderLayer. The half-step
  macaron FF scaling of 0.5 is applied at runtime, NOT baked into weights.
* Transformer-XL relative positional self-attention: q/k/v/o linear projections,
  a separate ``relative_k_proj`` for the relative position keys, plus learned
  per-head ``bias_u`` / ``bias_v`` (8,128). We synthesize the relative sinusoidal
  position embedding in pure JAX (no stored table in the checkpoint).
* Conv module: pointwise_conv1 (1024->2048) + GLU -> 1024, depthwise k=9 groups=1024,
  BatchNorm1d (frozen running stats consumed at inference), pointwise_conv2.
* Subsampling is 2-D: the mel features are treated as a single-channel image
  (B, 1, T, F); three stride-2 stages downsample BOTH time and frequency by 8,
  then the 256 channels x (F/8) freq bins are flattened and a Linear maps
  256*(F/8)=4096 -> 1024.
* The mel filterbank (1,128,257) and STFT window (400,) are LOADED from the
  checkpoint as frozen ``nnx.Variable`` buffers, NOT synthesized. NO pre-emphasis.

Simplified from the full ASR model: encoder only (no CTC/TDT decode head), no
streaming/chunked attention, fixed (non-padded) sequence handling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx


# =============================================================================
# Config
# =============================================================================


@dataclass
class AudioEncoderConfig:
    """
    Configuration for the Parakeet FastConformer sound encoder.

    Field names match the HF ``sound_config`` (``parakeet``) semantics. The
    "tiny" omni preset shrinks the dims/layers (but keeps one of every submodule
    type) so a CPU forward stays cheap; the real "omni_30b" preset uses the
    values below.
    """

    # ---- spectrogram / feature extractor ----
    sample_rate: int = 16000        # sampling_rate
    n_mels: int = 128               # num_mel_bins -> featurizer.fb has n_mels rows
    n_fft: int = 512                # STFT size (rfft -> n_fft//2+1 = 257 bins)
    frame_length: int = 400         # window length (25 ms @ 16 kHz) -> featurizer.window
    hop_length: int = 160           # hop (10 ms @ 16 kHz)

    # ---- conformer body ----
    hidden_dim: int = 1024          # hidden_size
    num_heads: int = 8              # num_attention_heads
    head_dim: int = 128             # hidden_size / num_heads
    num_layers: int = 24            # num_hidden_layers
    ffn_dim: int = 4096             # intermediate_size (d_ff)
    conv_kernel_size: int = 9       # conv_kernel_size (depthwise) -- NOT 31
    norm_eps: float = 1e-5

    # ---- subsampling ----
    subsampling_conv_channels: int = 256   # subsampling_conv_channels
    subsampling_conv_kernel: int = 3       # subsampling_conv_kernel_size
    subsampling_factor: int = 8            # 3 stride-2 stages

    # Body output width (== hidden_dim); the sound projector reads this.
    proj_dim: int = 1024

    @property
    def n_freqs(self) -> int:
        """Number of rfft frequency bins = n_fft // 2 + 1 (257 for n_fft=512)."""
        return self.n_fft // 2 + 1

    @property
    def num_subsampling_stages(self) -> int:
        """Number of stride-2 stages: log2(subsampling_factor)."""
        return int(round(math.log2(self.subsampling_factor)))

    @property
    def subsampled_freq(self) -> int:
        """Frequency bins after the 2-D subsampling (n_mels >> num_stages)."""
        f = self.n_mels
        for _ in range(self.num_subsampling_stages):
            f = (f + 1) // 2  # stride-2 conv with same-padding halves (ceil)
        return f


# =============================================================================
# Feature extractor (log-mel via STORED filterbank + window, NO pre-emphasis)
# =============================================================================


class ParakeetFeatureExtractor(nnx.Module):
    """
    Raw waveform -> log-mel features using the checkpoint's STORED mel filterbank
    and STFT window (loaded as frozen buffers, not synthesized).

    HF buffers (under sound_encoder.encoder.feature_extractor.featurizer.*):
        fb     : (1, n_mels, n_freqs) = (1, 128, 257) mel filterbank
        window : (frame_length,)      = (400,) analysis window

    Pipeline (NO pre-emphasis):
        1. frame the signal (center=True style reflect padding) into windows of
           length frame_length, hop hop_length
        2. multiply by the stored window
        3. zero-pad to n_fft, rfft, |.|^2 power spectrum  -> (B, n_freqs, T)
        4. mel = fb @ power                                -> (B, n_mels, T)
        5. log(mel + eps), transposed to (B, T, n_mels)

    These ``fb`` / ``window`` leaves appear in the eval_shape tree so the converter
    can copy the checkpoint values straight in.
    """

    def __init__(self, config: AudioEncoderConfig, rngs: nnx.Rngs):
        del rngs
        self.frame_length = config.frame_length
        self.hop_length = config.hop_length
        self.n_fft = config.n_fft

        # Frozen buffers (nnx.Variable, NOT nnx.Param -> no gradients). Shapes
        # mirror the HF checkpoint exactly so the converter copies them raw.
        self.fb = nnx.Variable(
            jnp.zeros((1, config.n_mels, config.n_freqs), dtype=jnp.float32)
        )
        self.window = nnx.Variable(
            jnp.zeros((config.frame_length,), dtype=jnp.float32)
        )

    def __call__(self, waveform: jax.Array) -> jax.Array:
        """
        Args:
            waveform: (B, T) raw audio at sample_rate Hz
        Returns:
            (B, n_frames, n_mels) log-mel features
        """
        win = self.window.value          # (frame_length,)
        fb = self.fb.value               # (1, n_mels, n_freqs)

        # center=True reflect padding by n_fft//2 each side (HF/librosa default).
        pad = self.n_fft // 2
        wav = jnp.pad(waveform, ((0, 0), (pad, pad)), mode="reflect")

        T = wav.shape[-1]
        n_frames = 1 + (T - self.n_fft) // self.hop_length

        frame_idx = (
            jnp.arange(n_frames)[:, None] * self.hop_length
            + jnp.arange(self.frame_length)[None, :]
        )
        frames = wav[:, frame_idx]                      # (B, n_frames, frame_length)
        frames = frames * win[None, None, :]

        pad_len = self.n_fft - self.frame_length
        if pad_len > 0:
            frames = jnp.pad(frames, ((0, 0), (0, 0), (0, pad_len)))

        power = jnp.abs(jnp.fft.rfft(frames, n=self.n_fft)) ** 2
        # power: (B, n_frames, n_freqs)

        # mel = power @ fb^T   (fb is (1, n_mels, n_freqs))
        mel = jnp.einsum("bft,mt->bfm", power, fb[0])    # (B, n_frames, n_mels)
        log_mel = jnp.log(mel + 1e-6)
        return log_mel


# =============================================================================
# Subsampling (2-D Conv2d depthwise/pointwise stack + final Linear)
# =============================================================================


class ConvSubsampling(nnx.Module):
    """
    FastConformer ``dw_striding`` subsampling: 8x time downsampling via three
    stride-2 2-D convolution stages over the mel "image", then a Linear that
    folds the (channels x freq) feature map down to hidden_dim.

    HF stores only the parametrized layer indices in a single ``layers`` list,
    interleaved with (parameter-free) ReLU activations:

        layers.0 : Conv2d(1   -> C,  k3, s2)              [256,1,3,3]   (groups=1)
        layers.1 : ReLU                                    (no params)
        layers.2 : Conv2d(C   -> C,  k3, s2, groups=C)    [256,1,3,3]   depthwise
        layers.3 : Conv2d(C   -> C,  k1)                  [256,256,1,1] pointwise
        layers.4 : ReLU
        layers.5 : Conv2d(C   -> C,  k3, s2, groups=C)    [256,1,3,3]   depthwise
        layers.6 : Conv2d(C   -> C,  k1)                  [256,256,1,1] pointwise
        layers.7 : ReLU
        linear   : Linear(C * (F/8) -> hidden_dim)        [1024,4096]

    We mirror this exactly: ``self.layers`` is an ``nnx.List`` whose activation
    slots (1,4,7) hold an empty module (no leaves), so the eval_shape paths are
    ``subsampling/layers/{0,2,3,5,6}/...`` -- matching HF index-for-index.
    """

    def __init__(self, config: AudioEncoderConfig, rngs: nnx.Rngs):
        C = config.subsampling_conv_channels
        k = config.subsampling_conv_kernel
        pad = k // 2

        def conv2d(in_f, out_f, kernel, stride, groups):
            return nnx.Conv(
                in_features=in_f,
                out_features=out_f,
                kernel_size=kernel,
                strides=stride,
                padding=tuple((p, p) for p in pad_for(kernel)),
                feature_group_count=groups,
                use_bias=True,
                rngs=rngs,
            )

        def pad_for(kernel):
            return [kk // 2 for kk in kernel]

        # Stage layers, interleaved with empty placeholder modules for ReLU slots
        # so list indices line up with HF (0,2,3,5,6 carry params; 1,4,7 empty).
        relu = lambda: nnx.Module()  # empty module -> contributes no leaves
        self.layers = nnx.List([
            conv2d(1, C, (k, k), (2, 2), 1),          # 0: first conv (groups=1)
            relu(),                                     # 1
            conv2d(C, C, (k, k), (2, 2), C),          # 2: depthwise
            conv2d(C, C, (1, 1), (1, 1), 1),          # 3: pointwise
            relu(),                                     # 4
            conv2d(C, C, (k, k), (2, 2), C),          # 5: depthwise
            conv2d(C, C, (1, 1), (1, 1), 1),          # 6: pointwise
            relu(),                                     # 7
        ])

        # Final projection: (C * subsampled_freq) -> hidden_dim.
        self.linear = nnx.Linear(
            in_features=C * config.subsampled_freq,
            out_features=config.hidden_dim,
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: (B, T, F) log-mel features (F == n_mels)
        Returns:
            (B, T//8, hidden_dim)
        """
        # Treat mel as a single-channel image. nnx.Conv expects channels-last:
        # (B, T, F, 1).
        h = x[..., None]                                   # (B, T, F, 1)

        h = self.layers[0](h)
        h = jax.nn.relu(h)                                 # layers[1]
        h = self.layers[2](h)
        h = self.layers[3](h)
        h = jax.nn.relu(h)                                 # layers[4]
        h = self.layers[5](h)
        h = self.layers[6](h)
        h = jax.nn.relu(h)                                 # layers[7]
        # h: (B, T//8, F//8, C)

        B, Tp, Fp, C = h.shape
        # HF flattens (channels, freq) per time step; channels-major to match the
        # (C * F) linear input ordering -> (B, T', C * F').
        h = jnp.transpose(h, (0, 1, 3, 2)).reshape(B, Tp, C * Fp)
        return self.linear(h)                              # (B, T', hidden_dim)


# =============================================================================
# Macaron feed-forward (half-step)
# =============================================================================


class ConformerFeedForward(nnx.Module):
    """
    Macaron feed-forward: Linear(hidden->d_ff) -> SiLU -> Linear(d_ff->hidden).

    The 0.5 half-step scaling and the residual add live in the Conformer layer
    (applied at runtime), NOT here, matching HF ParakeetFeedForward + the layer's
    macaron residuals.

    Leaves (HF): feed_forwardN.linear1.weight, feed_forwardN.linear2.weight.
    """

    def __init__(self, config: AudioEncoderConfig, rngs: nnx.Rngs):
        D, F = config.hidden_dim, config.ffn_dim
        self.linear1 = nnx.Linear(D, F, use_bias=False, rngs=rngs)
        self.linear2 = nnx.Linear(F, D, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear2(jax.nn.silu(self.linear1(x)))


# =============================================================================
# Transformer-XL relative-position multi-head self-attention
# =============================================================================


def _rel_shift(x: jax.Array) -> jax.Array:
    """
    Transformer-XL relative-position shift.

    Input ``x`` has shape (B, H, T, 2T-1) of scores against relative positions
    [-(T-1) .. (T-1)]; produce (B, H, T, T) aligning each query to the correct
    relative offset. Implemented with the standard pad-then-reshape trick.
    """
    B, H, T, P = x.shape  # P = 2T - 1
    x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (1, 0)))      # (B,H,T,P+1)
    x = x.reshape(B, H, P + 1, T)
    x = x[:, :, 1:, :]                                     # drop first
    x = x.reshape(B, H, T, P)
    return x[:, :, :, :T]                                  # keep first T


class RelPositionMultiHeadAttention(nnx.Module):
    """
    Transformer-XL style relative-position multi-head self-attention (the
    FastConformer / Conformer attention).

    Projections (all Linear, no bias) + learned per-head content/position biases:
        q_proj, k_proj, v_proj, o_proj : (hidden -> hidden)
        relative_k_proj                : (hidden -> hidden) for relative pos keys
        bias_u, bias_v                 : (num_heads, head_dim)

    Score = ((q + bias_u) . k)  +  rel_shift((q + bias_v) . p)   then / sqrt(d).

    The relative positional embeddings ``p`` are synthesized in pure JAX (the
    standard sinusoidal table over offsets [-(T-1) .. (T-1)]) -- there is no
    stored position table in the checkpoint.
    """

    def __init__(self, config: AudioEncoderConfig, rngs: nnx.Rngs):
        D = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.q_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)
        self.relative_k_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)

        self.bias_u = nnx.Param(jnp.zeros((self.num_heads, self.head_dim)))
        self.bias_v = nnx.Param(jnp.zeros((self.num_heads, self.head_dim)))

    def _rel_pos_emb(self, T: int, dtype) -> jax.Array:
        """Sinusoidal relative position embedding over offsets [T-1 .. -(T-1)]."""
        D = self.num_heads * self.head_dim
        # offsets from +(T-1) down to -(T-1): length 2T-1 (Conformer convention).
        pos = jnp.arange(T - 1, -T, -1.0)                  # (2T-1,)
        div = jnp.exp(jnp.arange(0, D, 2) * (-math.log(10000.0) / D))
        ang = pos[:, None] * div[None, :]                  # (2T-1, D/2)
        pe = jnp.zeros((2 * T - 1, D))
        pe = pe.at[:, 0::2].set(jnp.sin(ang))
        pe = pe.at[:, 1::2].set(jnp.cos(ang))
        return pe.astype(dtype)                            # (2T-1, D)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, d)
        k = self.k_proj(x).reshape(B, T, H, d)
        v = self.v_proj(x).reshape(B, T, H, d)

        # Relative position keys.
        pe = self._rel_pos_emb(T, x.dtype)                 # (2T-1, D)
        p = self.relative_k_proj(pe).reshape(2 * T - 1, H, d)

        # (B, H, T, d)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        p = jnp.transpose(p, (1, 0, 2))                    # (H, 2T-1, d)

        bu = self.bias_u.value[None, :, None, :]           # (1, H, 1, d)
        bv = self.bias_v.value[None, :, None, :]

        # content score: (q + bias_u) . k
        ac = jnp.einsum("bhqd,bhkd->bhqk", q + bu, k)      # (B,H,T,T)
        # position score: (q + bias_v) . p  -> (B,H,T,2T-1) then rel-shift
        bd = jnp.einsum("bhqd,hpd->bhqp", q + bv, p)
        bd = _rel_shift(bd)                                # (B,H,T,T)

        scores = (ac + bd) / math.sqrt(d)
        attn = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.einsum("bhqk,bhkd->bhqd", attn, v)       # (B,H,T,d)
        ctx = jnp.transpose(ctx, (0, 2, 1, 3)).reshape(B, T, D)
        return self.o_proj(ctx)


# =============================================================================
# Conv module (pointwise + GLU, depthwise k=9, BatchNorm1d, pointwise)
# =============================================================================


class ConformerConvModule(nnx.Module):
    """
    Conformer convolution module:
        pointwise_conv1 : Conv1d(D -> 2D, k1)   then GLU -> D
        depthwise_conv  : Conv1d(D -> D, k=9, groups=D)
        norm            : BatchNorm1d (running stats used at inference)
        (SiLU)
        pointwise_conv2 : Conv1d(D -> D, k1)

    Conv weights are stored PyTorch-style (out, in/groups, k); we use nnx.Conv
    (channels-last, kernel (k, in/groups, out)) -- the converter applies the
    conv axis permutation. The BatchNorm exposes scale/bias/mean/var leaves that
    map to HF conv.norm.{weight,bias,running_mean,running_var}; HF's
    num_batches_tracked is intentionally NOT modeled (it is the sole skipped
    tensor).
    """

    def __init__(self, config: AudioEncoderConfig, rngs: nnx.Rngs):
        D = config.hidden_dim
        k = config.conv_kernel_size
        pad = k // 2

        self.pointwise_conv1 = nnx.Conv(
            in_features=D, out_features=2 * D, kernel_size=(1,),
            use_bias=False, rngs=rngs,
        )
        self.depthwise_conv = nnx.Conv(
            in_features=D, out_features=D, kernel_size=(k,),
            strides=(1,), padding=((pad, pad),),
            feature_group_count=D, use_bias=False, rngs=rngs,
        )
        # BatchNorm1d: inference uses the frozen running statistics.
        self.norm = nnx.BatchNorm(
            num_features=D, use_running_average=True,
            epsilon=1e-5, momentum=0.1, rngs=rngs,
        )
        self.pointwise_conv2 = nnx.Conv(
            in_features=D, out_features=D, kernel_size=(1,),
            use_bias=False, rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (B, T, D) channels-last.
        h = self.pointwise_conv1(x)                        # (B, T, 2D)
        a, b = jnp.split(h, 2, axis=-1)
        h = a * jax.nn.sigmoid(b)                           # GLU -> (B, T, D)
        h = self.depthwise_conv(h)                          # (B, T, D)
        h = self.norm(h, use_running_average=True)
        h = jax.nn.silu(h)
        h = self.pointwise_conv2(h)                         # (B, T, D)
        return h


# =============================================================================
# Conformer layer (POST-norm macaron)
# =============================================================================


class ConformerLayer(nnx.Module):
    """
    A single FastConformer layer in HF's POST-norm macaron arrangement:

        x = x + 0.5 * feed_forward1(norm_feed_forward1(x))
        x = x + self_attn(norm_self_att(x))
        x = x + conv(norm_conv(x))
        x = x + 0.5 * feed_forward2(norm_feed_forward2(x))
        x = norm_out(x)

    All five norms are LayerNorm with weight+bias (HF *.weight / *.bias ->
    nnx.LayerNorm *.scale / *.bias).
    """

    def __init__(self, config: AudioEncoderConfig, rngs: nnx.Rngs):
        D = config.hidden_dim
        eps = config.norm_eps

        self.norm_feed_forward1 = nnx.LayerNorm(D, epsilon=eps, rngs=rngs)
        self.feed_forward1 = ConformerFeedForward(config, rngs=rngs)

        self.norm_self_att = nnx.LayerNorm(D, epsilon=eps, rngs=rngs)
        self.self_attn = RelPositionMultiHeadAttention(config, rngs=rngs)

        self.norm_conv = nnx.LayerNorm(D, epsilon=eps, rngs=rngs)
        self.conv = ConformerConvModule(config, rngs=rngs)

        self.norm_feed_forward2 = nnx.LayerNorm(D, epsilon=eps, rngs=rngs)
        self.feed_forward2 = ConformerFeedForward(config, rngs=rngs)

        self.norm_out = nnx.LayerNorm(D, epsilon=eps, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x))
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        x = self.norm_out(x)
        return x


# =============================================================================
# Audio encoder (feature extractor -> subsampling -> conformer layers)
# =============================================================================


class AudioEncoder(nnx.Module):
    """
    HF-faithful Parakeet FastConformer encoder body.

    Submodule attribute names mirror the HF ``sound_encoder.encoder.*`` layout so
    that the eval_shape flat paths line up 1:1 with the checkpoint:

        feature_extractor.featurizer.{fb,window}
        subsampling.layers.{0,2,3,5,6}.* , subsampling.linear.*
        layers.{0..N-1}.*  (the ConformerLayer leaves above)

    Forward:  waveform (B,T) -> (B, T_out, hidden_dim).
    """

    def __init__(self, config: AudioEncoderConfig, rngs: nnx.Rngs):
        self.config = config
        self.feature_extractor = _FeatureExtractorWrapper(config, rngs=rngs)
        self.subsampling = ConvSubsampling(config, rngs=rngs)
        self.layers = nnx.List(
            [ConformerLayer(config, rngs=rngs) for _ in range(config.num_layers)]
        )

    def __call__(self, waveform: jax.Array) -> jax.Array:
        x = self.feature_extractor(waveform)       # (B, n_frames, n_mels)
        x = self.subsampling(x)                    # (B, T_out, hidden_dim)
        for layer in self.layers:
            x = layer(x)
        return x


class _FeatureExtractorWrapper(nnx.Module):
    """
    Thin wrapper giving the path segment ``feature_extractor.featurizer.*`` to
    match the HF key ``sound_encoder.encoder.feature_extractor.featurizer.{fb,
    window}``.
    """

    def __init__(self, config: AudioEncoderConfig, rngs: nnx.Rngs):
        self.featurizer = ParakeetFeatureExtractor(config, rngs=rngs)

    def __call__(self, waveform: jax.Array) -> jax.Array:
        return self.featurizer(waveform)


# =============================================================================
# HF -> Orbax NAME MAP for the sound encoder + projector
# =============================================================================
#
# The authoritative target-path -> {hf_name(s), transform} contract that the
# milestone-2 converter consumes for the sound namespace, mirroring
# ``nemotron_h.hf_name_map`` for the LLM backbone. The converter flattens OUR
# eval_shape tree to slash paths (e.g. "sound_encoder/layers/3/conv/norm/scale")
# and looks each one up here.
#
# Unlike the LLM (HF_PREFIX = "language_model."), the sound tensors sit at the HF
# TOP LEVEL: the encoder body under ``sound_encoder.encoder.*`` and the projector
# under ``sound_projection.*``. So the converter uses HF_PREFIX = "" for sound and
# the HF names below are FULL (absolute) safetensors keys.
#
# Transforms (see scripts/convert_hf_to_orbax.py for the pure implementations):
#   raw    : copy unchanged. LayerNorm/BatchNorm weight+bias, BN running_mean/var,
#            bias_u/bias_v, RMSNorm scale, featurizer.fb/window, conv biases.
#   T      : transpose a 2-D nn.Linear weight (out,in) -> (in,out). All Linear
#            weights: FF linear1/2, q/k/v/o/relative_k proj, subsampling.linear,
#            projection linear1/2.
#   conv   : PyTorch Conv1d (out_ch, in_ch/groups, k) -> nnx.Conv (k, in_ch/g, out).
#            pointwise_conv1/2 (k=1) and depthwise_conv (k=9, in/g==1).
#   conv2d : PyTorch Conv2d (out_ch, in_ch/groups, kH, kW) -> nnx.Conv
#            (kH, kW, in_ch/groups, out_ch), axes (2,3,1,0). subsampling conv layers.
#
# The HF BatchNorm buffer ``conv.norm.num_batches_tracked`` (an I64 counter, not a
# learnable param) is intentionally NOT modeled -> it is the sole sound tensor the
# converter expects to leave unconsumed.

HF_SOUND_PREFIX = ""  # sound tensors are top-level (sound_encoder.* / sound_projection.*)


def hf_sound_name_map(config) -> dict:
    """
    Build the target-path -> {hf, transform} map for the sound encoder + projector.

    ``config`` is the NemotronOmniConfig (it carries ``sound`` AudioEncoderConfig,
    ``sound_proj_in`` / ``sound_projector_hidden`` and the llm hidden/eps). Returns
    a dict keyed by the slash-joined eval_shape path of OUR sound encoder
    (prefix ``sound_encoder/``) and projector (prefix ``sound_projection/``). HF
    names are FULL safetensors keys (HF_SOUND_PREFIX == "").

    Covers EVERY leaf the faithful AudioEncoder + SoundProjector expose. The lone
    HF tensor NOT referenced here is ``...conv.norm.num_batches_tracked`` per layer
    (skipped by design).
    """
    sc = config.sound
    m: dict = {}

    # ---- feature extractor frozen buffers (raw) ----
    fe = "sound_encoder/feature_extractor/featurizer"
    hfe = "sound_encoder.encoder.feature_extractor.featurizer"
    m[f"{fe}/fb"] = {"hf": f"{hfe}.fb", "transform": "raw"}
    m[f"{fe}/window"] = {"hf": f"{hfe}.window", "transform": "raw"}

    # ---- subsampling (2-D conv stack + final linear) ----
    ss = "sound_encoder/subsampling"
    hss = "sound_encoder.encoder.subsampling"
    # Parametrized conv slots: 0 (first conv, groups=1), 2 & 5 (depthwise),
    # 3 & 6 (pointwise). All are Conv2d -> conv2d transform; biases raw.
    for j in (0, 2, 3, 5, 6):
        m[f"{ss}/layers/{j}/kernel"] = {
            "hf": f"{hss}.layers.{j}.weight",
            "transform": "conv2d",
        }
        m[f"{ss}/layers/{j}/bias"] = {
            "hf": f"{hss}.layers.{j}.bias",
            "transform": "raw",
        }
    # Final projection Linear (C*F/8 -> hidden): weight T, bias raw.
    m[f"{ss}/linear/kernel"] = {"hf": f"{hss}.linear.weight", "transform": "T"}
    m[f"{ss}/linear/bias"] = {"hf": f"{hss}.linear.bias", "transform": "raw"}

    # ---- conformer layers ----
    for i in range(sc.num_layers):
        lp = f"sound_encoder/layers/{i}"
        hp = f"sound_encoder.encoder.layers.{i}"

        # Five LayerNorms: nnx scale/bias <- HF weight/bias (raw).
        for norm in (
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "norm_feed_forward2",
            "norm_out",
        ):
            m[f"{lp}/{norm}/scale"] = {"hf": f"{hp}.{norm}.weight", "transform": "raw"}
            m[f"{lp}/{norm}/bias"] = {"hf": f"{hp}.{norm}.bias", "transform": "raw"}

        # Macaron feed-forwards (no bias): Linear weights -> T.
        for ff in ("feed_forward1", "feed_forward2"):
            m[f"{lp}/{ff}/linear1/kernel"] = {
                "hf": f"{hp}.{ff}.linear1.weight",
                "transform": "T",
            }
            m[f"{lp}/{ff}/linear2/kernel"] = {
                "hf": f"{hp}.{ff}.linear2.weight",
                "transform": "T",
            }

        # Transformer-XL relative-position self-attention.
        sa = f"{lp}/self_attn"
        hsa = f"{hp}.self_attn"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"):
            m[f"{sa}/{proj}/kernel"] = {
                "hf": f"{hsa}.{proj}.weight",
                "transform": "T",
            }
        m[f"{sa}/bias_u"] = {"hf": f"{hsa}.bias_u", "transform": "raw"}
        m[f"{sa}/bias_v"] = {"hf": f"{hsa}.bias_v", "transform": "raw"}

        # Conv module: two pointwise Conv1d + depthwise Conv1d + BatchNorm1d.
        cv = f"{lp}/conv"
        hcv = f"{hp}.conv"
        m[f"{cv}/pointwise_conv1/kernel"] = {
            "hf": f"{hcv}.pointwise_conv1.weight",
            "transform": "conv",
        }
        m[f"{cv}/depthwise_conv/kernel"] = {
            "hf": f"{hcv}.depthwise_conv.weight",
            "transform": "conv",
        }
        m[f"{cv}/pointwise_conv2/kernel"] = {
            "hf": f"{hcv}.pointwise_conv2.weight",
            "transform": "conv",
        }
        # BatchNorm1d: nnx scale/bias/mean/var <- HF weight/bias/running_mean/var.
        # (HF num_batches_tracked is the sole skipped sound tensor.)
        m[f"{cv}/norm/scale"] = {"hf": f"{hcv}.norm.weight", "transform": "raw"}
        m[f"{cv}/norm/bias"] = {"hf": f"{hcv}.norm.bias", "transform": "raw"}
        m[f"{cv}/norm/mean"] = {"hf": f"{hcv}.norm.running_mean", "transform": "raw"}
        m[f"{cv}/norm/var"] = {"hf": f"{hcv}.norm.running_var", "transform": "raw"}

    # ---- sound projector (top-level sound_projection.*) ----
    m["sound_projection/norm/scale"] = {
        "hf": "sound_projection.norm.weight",
        "transform": "raw",
    }
    m["sound_projection/linear1/kernel"] = {
        "hf": "sound_projection.linear1.weight",
        "transform": "T",
    }
    m["sound_projection/linear2/kernel"] = {
        "hf": "sound_projection.linear2.weight",
        "transform": "T",
    }

    return m
