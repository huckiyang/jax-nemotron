"""
Audio Encoder for Nemotron 3 Nano Omni in JAX/Flax NNX.

Simplified, educational Conformer-style audio encoder inspired by the
Parakeet-TDT-0.6B-v2 FastConformer encoder used in the paper:
  "Nemotron 3 Nano Omni: Efficient and Open Multimodal Intelligence"
  arXiv:2604.24954

Architecture overview (encoder-projector design):
  waveform → LogMelSpectrogram → ConvSubsampling → N × ConformerBlock
           → AudioProjector → [audio tokens for LLM]

Key design choices (matching the paper):
- Pure JAX log-mel spectrogram — no external audio libraries.
  The mel filterbank is stored as a frozen nnx.Variable (built once at init,
  never updated by gradients).
- 3 stride-2 conv layers: ~8× temporal downsampling → ~12.5 tokens/sec at 16kHz
- Conformer blocks with Macaron structure:
    ½ FFN → Multi-head Attention → Conv Module → ½ FFN → LayerNorm
- 2-layer MLP projector bridges encoder hidden dim to LLM d_model

Simplified from real Parakeet/FastConformer:
- No CTC or TDT decode head (encoder only)
- No streaming or chunked processing
- Standard LayerNorm instead of BatchNorm in the conv module
"""

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
    Configuration for the audio encoder and its MLP projector.

    Tiny defaults allow local experimentation. Scale hidden_dim and num_layers
    for paper-like quality (paper uses 0.6B-scale FastConformer).
    """

    sample_rate: int = 16000    # Expected waveform sample rate in Hz
    n_mels: int = 80            # Number of mel filterbank channels
    n_fft: int = 512            # FFT size (>= frame_length; next power of 2)
    frame_length: int = 400     # 25 ms window at 16 kHz
    hop_length: int = 160       # 10 ms hop at 16 kHz
    hidden_dim: int = 128       # Conformer hidden width
    num_heads: int = 4          # Attention heads
    head_dim: int = 32          # Per-head dimension (hidden_dim = num_heads × head_dim)
    num_layers: int = 2         # Number of conformer blocks
    ffn_dim: int = 256          # FFN inner width inside each conformer block
    conv_kernel_size: int = 31  # Depthwise conv kernel in the conv module
    proj_dim: int = 128         # Output dimension — must equal LLM d_model


# =============================================================================
# Mel Filterbank Builder (pure Python/NumPy, runs once at init)
# =============================================================================


def _build_mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> jax.Array:
    """
    Build a triangular mel filterbank matrix in NumPy (runs once at __init__).

    Each column of the output is a triangular mel filter spanning a band of
    the linear-frequency FFT spectrum. Multiplying a power spectrum by this
    matrix gives mel energies.

    The mel scale is a perceptual scale of pitch: equally spaced on the mel
    scale corresponds to exponentially spaced in Hz, matching human hearing.

    Conversion formulas:
      mel  = 2595 × log10(1 + hz / 700)
      hz   = 700 × (10^(mel / 2595) − 1)

    Args:
        n_mels:      Number of mel channels.
        n_fft:       FFT size.
        sample_rate: Audio sample rate in Hz.
        f_min:       Lowest frequency in Hz (default 0).
        f_max:       Highest frequency in Hz (default sample_rate / 2).
    Returns:
        filterbank: shape (n_fft // 2 + 1, n_mels) — maps FFT bins → mel bins
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    # n_mels + 2 linearly spaced mel-scale points (including edge anchors).
    mel_lo, mel_hi = hz_to_mel(f_min), hz_to_mel(f_max)
    mel_points = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Map Hz values to FFT bin indices (floor to nearest bin).
    n_freqs = n_fft // 2 + 1
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_freqs - 1)

    # Build triangular filters: each filter rises then falls between its
    # three anchor bins (lo, center, hi).
    filterbank = np.zeros((n_freqs, n_mels), dtype=np.float32)
    for m in range(n_mels):
        f_lo = bin_points[m]        # rising edge start
        f_mid = bin_points[m + 1]   # peak
        f_hi = bin_points[m + 2]    # falling edge end

        # Rising slope: linearly ramp from 0 at f_lo to 1 at f_mid.
        if f_mid > f_lo:
            for k in range(f_lo, f_mid):
                filterbank[k, m] = (k - f_lo) / (f_mid - f_lo)

        # Falling slope: linearly ramp from 1 at f_mid to 0 at f_hi.
        if f_hi > f_mid:
            for k in range(f_mid, f_hi):
                filterbank[k, m] = (f_hi - k) / (f_hi - f_mid)

    return jnp.array(filterbank)  # (n_freqs, n_mels)


# =============================================================================
# Log-Mel Spectrogram (pure JAX)
# =============================================================================


class LogMelSpectrogram(nnx.Module):
    """
    Converts a raw 16 kHz mono waveform to log-mel spectrogram features.

    Processing pipeline:
      1. Pre-emphasis: x[t] ← x[t] − 0.97·x[t−1]
         Amplifies high frequencies (speech intelligibility boost).
      2. Framing: split signal into overlapping short-time frames
         (25 ms window, 10 ms hop → frame_length=400, hop_length=160 at 16kHz).
      3. Hann window: smooth each frame to reduce spectral leakage at edges.
      4. Power spectrum: |rfft(frame)|²
      5. Mel filterbank: power_spectrum @ filterbank_matrix → mel energies
      6. Log-compress: log(mel + ε) to keep values numerically stable

    The mel filterbank is built once at __init__ and stored as a frozen
    nnx.Variable (base Variable, not nnx.Param → excluded from gradient
    updates). The Hann window is also frozen.

    Args:
        config: AudioEncoderConfig
    """

    def __init__(self, config: AudioEncoderConfig, rngs: nnx.Rngs):
        self.frame_length = config.frame_length
        self.hop_length = config.hop_length
        self.n_fft = config.n_fft

        # Build mel filterbank matrix once at init (pure NumPy, not JIT-traced).
        filterbank = _build_mel_filterbank(
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            sample_rate=config.sample_rate,
        )
        # Store as frozen nnx.Variable (not nnx.Param) — no gradients.
        self.filterbank = nnx.Variable(filterbank)    # (n_freqs, n_mels)

        # Hann window: smooth tapering reduces FFT spectral leakage.
        hann = jnp.array(np.hanning(config.frame_length), dtype=jnp.float32)
        self.hann_window = nnx.Variable(hann)         # (frame_length,)

    def __call__(self, waveform: jax.Array) -> jax.Array:
        """
        Args:
            waveform: (batch, T) float32 raw samples, expected at sample_rate Hz
        Returns:
            log_mel: (batch, n_frames, n_mels) log-mel spectrogram features
        """
        # 1) Pre-emphasis: boost high frequencies, suppress low-frequency bias.
        #    Boundary: keep the very first sample unchanged.
        waveform = jnp.concatenate(
            [waveform[:, :1], waveform[:, 1:] - 0.97 * waveform[:, :-1]],
            axis=-1,
        )  # (batch, T)

        # 2) Frame the signal into overlapping short-time windows.
        #    Build an index matrix: frame_idx[i, j] = start_of_frame_i + j
        T = waveform.shape[-1]
        n_frames = (T - self.frame_length) // self.hop_length + 1

        # (n_frames, frame_length) index matrix for advanced indexing.
        frame_idx = (
            jnp.arange(n_frames)[:, None] * self.hop_length
            + jnp.arange(self.frame_length)[None, :]
        )

        # Gather frames from every batch element simultaneously.
        frames = waveform[:, frame_idx]  # (batch, n_frames, frame_length)

        # 3) Apply Hann window to each frame.
        frames = frames * self.hann_window.value[None, None, :]

        # 4) Zero-pad frames to n_fft length if n_fft > frame_length.
        pad_len = self.n_fft - self.frame_length
        if pad_len > 0:
            frames = jnp.pad(frames, ((0, 0), (0, 0), (0, pad_len)))

        # Compute real FFT and take squared magnitude (power spectrum).
        spectrum = jnp.abs(jnp.fft.rfft(frames, n=self.n_fft)) ** 2
        # spectrum: (batch, n_frames, n_fft // 2 + 1)

        # 5) Apply mel filterbank: map linear-frequency bins → mel bins.
        mel = jnp.matmul(spectrum, self.filterbank.value)
        # mel: (batch, n_frames, n_mels)

        # 6) Log-compress: prevents large energy values from dominating.
        log_mel = jnp.log(mel + 1e-6)

        return log_mel  # (batch, n_frames, n_mels)


# =============================================================================
# Convolutional Subsampling (~8× temporal downsampling)
# =============================================================================


class ConvSubsampling(nnx.Module):
    """
    Three stride-2 1D convolutions providing ~8× temporal downsampling.

    The paper specifies: "three stride-2 convolutional subsampling layers,
    resulting in an overall ~8× temporal downsampling" → ~12.5 tokens/sec
    of audio at 16kHz (roughly 80 ms per audio token).

    Structure:
      Conv(stride=2) → SiLU → Conv(stride=2) → SiLU → Conv(stride=2)

    Input  shape: (batch, T,   n_mels)
    Output shape: (batch, T//8, hidden_dim)

    In Flax NNX, nnx.Conv with kernel_size=(3,) operates on 1D sequences
    with shape (batch, length, features). Setting strides=(2,) halves length.

    Args:
        in_features: Number of input mel channels (n_mels).
        hidden_dim:  Output dimension after subsampling.
    """

    def __init__(self, in_features: int, hidden_dim: int, rngs: nnx.Rngs):
        # First conv: n_mels → hidden_dim, stride 2
        self.conv1 = nnx.Conv(
            in_features=in_features,
            out_features=hidden_dim,
            kernel_size=(3,),
            strides=(2,),
            padding=((1, 1),),   # same-style padding for odd-length preservation
            use_bias=False,
            rngs=rngs,
        )
        # Second conv: hidden_dim → hidden_dim, stride 2
        self.conv2 = nnx.Conv(
            in_features=hidden_dim,
            out_features=hidden_dim,
            kernel_size=(3,),
            strides=(2,),
            padding=((1, 1),),
            use_bias=False,
            rngs=rngs,
        )
        # Third conv: hidden_dim → hidden_dim, stride 2
        self.conv3 = nnx.Conv(
            in_features=hidden_dim,
            out_features=hidden_dim,
            kernel_size=(3,),
            strides=(2,),
            padding=((1, 1),),
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: (batch, T, n_mels)
        Returns:
            (batch, T // 8, hidden_dim)
        """
        x = jax.nn.silu(self.conv1(x))  # (batch, T//2, hidden_dim)
        x = jax.nn.silu(self.conv2(x))  # (batch, T//4, hidden_dim)
        x = self.conv3(x)               # (batch, T//8, hidden_dim)
        return x


# =============================================================================
# Conformer Feed-Forward (half-step, Macaron structure)
# =============================================================================


class ConformerFeedForward(nnx.Module):
    """
    Half-step feed-forward module used at both ends of each Conformer block.

    The "Macaron" structure (named after the sandwich biscuit) places an FFN
    at both the beginning and end of the Conformer block, each contributing
    only half its output (scaled by 0.5). This was empirically shown to improve
    speech recognition over a single full-step FFN.

    Structure (added as residual):
      x = x + 0.5 × Linear(SiLU(Linear(RMSNorm(x))))

    Args:
        hidden_dim: Input/output dimension.
        ffn_dim:    Inner expansion dimension.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, rngs: nnx.Rngs):
        self.norm = nnx.RMSNorm(hidden_dim, rngs=rngs)
        self.fc1 = nnx.Linear(hidden_dim, ffn_dim, use_bias=False, rngs=rngs)
        self.fc2 = nnx.Linear(ffn_dim, hidden_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = jax.nn.silu(self.fc1(self.norm(x)))
        # Scale by 0.5: this is the "half-step" contribution in the Macaron design.
        return x + 0.5 * self.fc2(h)


# =============================================================================
# Conformer Conv Module
# =============================================================================


class ConformerConvModule(nnx.Module):
    """
    Lightweight depthwise conv module inside each Conformer block.

    Captures local temporal patterns in the audio sequence (phoneme duration,
    rhythm, spectral shape). The GLU gate controls how much information flows
    through, giving the model a learned "filter" over each time step.

    Structure:
      h = RMSNorm(x)
      h, gate = split(Linear(h, 2D), axis=-1)   # pointwise expand + split
      h = h × sigmoid(gate)                      # Gated Linear Unit (GLU)
      h = DepthwiseConv1D(h)                     # local temporal patterns
      h = SiLU(RMSNorm(h))                       # normalize + activate
      h = Linear(h, D)                           # pointwise project back
      return x + h                               # residual

    Args:
        hidden_dim:  Input/output dimension D.
        kernel_size: Depthwise conv kernel (31 in standard Conformer = ~193ms
                     receptive field at 12.5 tokens/sec).
    """

    def __init__(self, hidden_dim: int, kernel_size: int, rngs: nnx.Rngs):
        D = hidden_dim
        pad = kernel_size // 2  # symmetric padding for 'same' output length

        self.norm = nnx.RMSNorm(D, rngs=rngs)

        # Pointwise expand: D → 2D (one half is content, one half is the gate).
        self.pointwise_expand = nnx.Linear(D, 2 * D, use_bias=False, rngs=rngs)

        # Depthwise 1D conv: each channel has its own temporal kernel.
        # feature_group_count=D makes it fully depthwise (1 channel per group).
        self.depthwise = nnx.Conv(
            in_features=D,
            out_features=D,
            kernel_size=(kernel_size,),
            strides=(1,),
            padding=((pad, pad),),
            feature_group_count=D,
            use_bias=False,
            rngs=rngs,
        )

        self.norm_post = nnx.RMSNorm(D, rngs=rngs)

        # Pointwise project: D → D
        self.pointwise_project = nnx.Linear(D, D, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.norm(x)

        # Pointwise expansion and GLU gating.
        h = self.pointwise_expand(h)              # (batch, T, 2D)
        h, gate = jnp.split(h, 2, axis=-1)        # each (batch, T, D)
        h = h * jax.nn.sigmoid(gate)              # GLU: gate controls flow

        # Depthwise temporal convolution captures local acoustic patterns.
        h = self.depthwise(h)                     # (batch, T, D)

        # Post-conv normalization and activation.
        h = jax.nn.silu(self.norm_post(h))

        # Pointwise projection back to D.
        h = self.pointwise_project(h)

        return x + h  # residual


# =============================================================================
# Conformer Block
# =============================================================================


class ConformerBlock(nnx.Module):
    """
    Full Conformer block with the Macaron ½FFN–MHA–Conv–½FFN structure.

    The Conformer combines:
    - Feed-forward networks (capture global patterns)
    - Multi-head self-attention (attend to distant time steps)
    - Depthwise convolution (capture local temporal patterns)

    The Macaron structure with half-step FFNs at both ends empirically
    outperforms putting FFN only at one end for audio tasks.

    Block structure:
      x = ConformerFeedForward(x)           # ½ FFN (Macaron, first half)
      x = x + MHA(RMSNorm(x))              # multi-head self-attention
      x = ConformerConvModule(x)            # depthwise conv module
      x = ConformerFeedForward(x)           # ½ FFN (Macaron, second half)
      x = RMSNorm(x)                        # final normalization

    Args:
        config: AudioEncoderConfig
    """

    def __init__(self, config: AudioEncoderConfig, rngs: nnx.Rngs):
        D = config.hidden_dim

        # Two half-step FFNs for the Macaron structure.
        self.ffn1 = ConformerFeedForward(D, config.ffn_dim, rngs=rngs)
        self.ffn2 = ConformerFeedForward(D, config.ffn_dim, rngs=rngs)

        # Multi-head self-attention (non-causal — audio encoder is bidirectional).
        self.norm_attn = nnx.RMSNorm(D, rngs=rngs)
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.q_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)

        # Depthwise conv module for local temporal pattern extraction.
        self.conv_module = ConformerConvModule(D, config.conv_kernel_size, rngs=rngs)

        # Final layer norm after all sub-modules.
        self.final_norm = nnx.RMSNorm(D, rngs=rngs)

    def _attention(self, x: jax.Array) -> jax.Array:
        """
        Non-causal multi-head attention.
        Audio encoder is bidirectional: every frame can attend to all others.
        """
        batch, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = jnp.transpose(self.q_proj(x).reshape(batch, T, H, d), (0, 2, 1, 3))
        k = jnp.transpose(self.k_proj(x).reshape(batch, T, H, d), (0, 2, 1, 3))
        v = jnp.transpose(self.v_proj(x).reshape(batch, T, H, d), (0, 2, 1, 3))
        # Each: (batch, H, T, d)

        scale = 1.0 / math.sqrt(d)
        attn = jax.nn.softmax(
            jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale, axis=-1
        )
        context = jnp.einsum("bhqk,bhkd->bhqd", attn, v)  # (B, H, T, d)

        context = jnp.transpose(context, (0, 2, 1, 3)).reshape(batch, T, D)
        return self.out_proj(context)

    def __call__(self, x: jax.Array) -> jax.Array:
        # 1) First half-step FFN (Macaron opening).
        x = self.ffn1(x)

        # 2) Multi-head self-attention with pre-norm + residual.
        x = x + self._attention(self.norm_attn(x))

        # 3) Depthwise conv module (local temporal features).
        x = self.conv_module(x)

        # 4) Second half-step FFN (Macaron closing).
        x = self.ffn2(x)

        # 5) Final normalization.
        return self.final_norm(x)


# =============================================================================
# Audio Encoder
# =============================================================================


class AudioEncoder(nnx.Module):
    """
    Full Conformer-style audio encoder.

    Pipeline:
      waveform → LogMelSpectrogram → ConvSubsampling → [ConformerBlock × N]

    The output is a sequence of audio tokens in encoder hidden_dim space.
    AudioProjector (below) maps them to the LLM's d_model dimension.

    Args:
        config: AudioEncoderConfig
    """

    def __init__(self, config: AudioEncoderConfig, rngs: nnx.Rngs):
        self.log_mel = LogMelSpectrogram(config, rngs=rngs)
        self.subsampling = ConvSubsampling(config.n_mels, config.hidden_dim, rngs=rngs)
        self.blocks = nnx.List(
            [ConformerBlock(config, rngs=rngs) for _ in range(config.num_layers)]
        )

    def __call__(self, waveform: jax.Array) -> jax.Array:
        """
        Args:
            waveform: (batch, T) raw audio at sample_rate Hz
        Returns:
            tokens: (batch, T_out, hidden_dim)
                    T_out ≈ (T / hop_length) / 8
                    e.g. 1 second at 16kHz → 100 mel frames → ~12 audio tokens
        """
        # 1) Extract log-mel spectral features from the raw waveform.
        x = self.log_mel(waveform)      # (batch, n_frames, n_mels)

        # 2) 8× temporal downsampling via strided convolutions.
        x = self.subsampling(x)         # (batch, n_frames // 8, hidden_dim)

        # 3) Conformer blocks: attention + conv over the downsampled sequence.
        for block in self.blocks:
            x = block(x)

        return x  # (batch, T_out, hidden_dim)


# =============================================================================
# Audio Projector
# =============================================================================


class AudioProjector(nnx.Module):
    """
    2-layer MLP that adapts audio encoder tokens to the LLM's hidden dimension.

    Mirrors VisionProjector in design — a simple bridge from encoder space
    to LLM space. Trained first with the LLM frozen (stage 2 in paper),
    then jointly with the audio encoder in stage 3.

    Args:
        in_dim:  Audio encoder output dimension.
        out_dim: LLM hidden dimension (d_model).
    """

    def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs):
        hidden = (in_dim + out_dim) // 2
        self.fc1 = nnx.Linear(in_dim, hidden, use_bias=False, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, out_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: (batch, T, in_dim)
        Returns:
            (batch, T, out_dim)
        """
        return self.fc2(jax.nn.silu(self.fc1(x)))
