"""
Nemotron-3-Nano-Omni — HF-faithful omni (text + vision + sound) wrapper in Flax NNX.

This module wires the three modalities of NVIDIA Nemotron-3-Nano-Omni-30B-A3B into
one model whose forward pass matches the HuggingFace ``modeling.py`` fusion strategy:

    1. Embed the text token ids with the LLM embedding table. The token sequence
       ALREADY contains placeholder tokens at the positions where image / sound
       features belong:
           img_context_token_id   = 18   (HF default)
           sound_context_token_id = 27   (HF default)
       (The HF processor expands ``<image>`` / ``<audio>`` into runs of these
       placeholder ids before tokenization.)
    2. Encode pixels with the RADIO ViT vision encoder, project 1280 -> 20480 ->
       2688 (== LLM hidden) with a SquaredReLU MLP projector.
    3. Encode audio with the Parakeet FastConformer encoder, project 1024 -> mid ->
       2688 with a SiLU MLP projector.
    4. SPLICE the projected vision/sound tokens IN PLACE into the text-embedding
       tensor at the placeholder positions (masked scatter). This is HF's design:
       text-centric, in-place replacement — NOT prefix concatenation. See
       modeling.py line ~210: ``inputs_embeds[selected] = vit_embeds``.
    5. Run the fused embedding sequence through the Nemotron-H LLM backbone
       (embeddings already applied, so we call the layers/norm_f/lm_head directly).

------------------------------------------------------------------------------
Why in-place splice (not the reference repo's prefix concat)?
------------------------------------------------------------------------------
The educational reference (``reference/nemotron_omni.py``) concatenates
``[vis | aud | text]`` as a prefix. The REAL HF checkpoint instead reserves
placeholder token positions inside the text stream and overwrites their
embeddings with encoder features. We match HF so that, once real weights are
loaded (milestone-2), the token layout the LLM sees is identical to the
PyTorch model's. This is the structurally-faithful choice the task asks for.

------------------------------------------------------------------------------
HF namespace correspondence (for the milestone-2 converter)
------------------------------------------------------------------------------
This wrapper deliberately groups its submodules so the eventual HF->Orbax
converter has an obvious target for each HF top-level namespace:

    self.language_model   <- language_model.*        (nemotron_h.NemotronHModel)
    self.vision_model     <- vision_model.*          (RADIO ViT body)
    self.vision_projector <- mlp1.*                  (the 1280->20480->2688 MLP)
    self.sound_encoder    <- sound_encoder.*         (Parakeet FastConformer body)
    self.sound_projector  <- sound_projection.*      (the 1024->mid->2688 MLP)

The encoder BODIES are the vendored educational modules (vision_encoder.py /
audio_encoder.py). Their internal parameter names do NOT yet match the HF
``vision_model.radio_model.model.blocks.*`` / ``sound_encoder.encoder.layers.*``
layouts. Closing that gap is milestone-2 conversion work; the concrete
divergences are enumerated in the NAME-MAP TODOs at the bottom of this file.
The correctness bar HERE is: structurally sound + a tiny CPU forward over a
fused [vision|sound|text] sequence returns finite logits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from .config import NemotronHConfig
from .nemotron_h import NemotronHModel
from .vision_encoder import VisionEncoder, VisionEncoderConfig
from .audio_encoder import AudioEncoder, AudioEncoderConfig


# =============================================================================
# HF placeholder token ids (from config.json / modeling.py)
# =============================================================================

# These are the literal token ids the HF processor writes into the text stream
# to mark where image / sound features should be spliced in.
DEFAULT_IMG_CONTEXT_TOKEN_ID = 18
DEFAULT_SOUND_CONTEXT_TOKEN_ID = 27


# =============================================================================
# Squared-ReLU (relu2) — the vision projector activation, matching the LLM MoE
# =============================================================================


def _relu2(x: jax.Array) -> jax.Array:
    """relu(x)^2 — the squared-ReLU used by the HF vision projector (mlp1)."""
    r = jax.nn.relu(x)
    return r * r


# =============================================================================
# Projectors (modality encoder hidden -> LLM hidden)
# =============================================================================


class VisionProjector(nnx.Module):
    """
    Vision projector matching the HF ``mlp1`` adapter.

    HF ``mlp1`` is a 3-tensor stack (indices 0,1,3 in the checkpoint):
        mlp1.0 : RMSNorm-ish / Linear(vit_hidden*shuffle  -> projector_hidden)
        mlp1.1 : (activation index — SquaredReLU, no params)
        mlp1.3 : Linear(projector_hidden -> llm_hidden)
    We model the two learnable linears explicitly with a SquaredReLU between
    them (HF uses NewGELU/SquaredReLU in the projector; we follow the omni
    ground-truth note: "SquaredReLU between layers").

        in_dim  = vit_hidden_size  (real: 1280; after pixel-shuffle the vendored
                  encoder already projects back to vit_hidden, so in_dim == that)
        mid_dim = projector_hidden_size (real: 20480)
        out_dim = llm hidden_size (real: 2688)

    NOTE (milestone-2): the vendored ViT applies pixel-shuffle + projects back to
    vit_hidden internally, whereas HF folds the 2x2 pixel-shuffle (vit_hidden*4)
    into mlp1's first Linear. The converter must reconcile this; see TODOs.
    """

    def __init__(self, rngs: nnx.Rngs, in_dim: int, mid_dim: int, out_dim: int):
        self.fc1 = nnx.Linear(in_dim, mid_dim, use_bias=False, rngs=rngs)
        self.fc2 = nnx.Linear(mid_dim, out_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(_relu2(self.fc1(x)))


class SoundProjector(nnx.Module):
    """
    Sound projector matching the HF ``sound_projection`` adapter.

    HF ``sound_projection`` is a (linear1 -> norm -> linear2) stack:
        sound_projection.linear1.weight : Linear(sound_hidden -> mid)
        sound_projection.norm.weight    : RMSNorm(mid)
        sound_projection.linear2.weight : Linear(mid -> llm_hidden)
    Activation between is SiLU (omni ground-truth note: "SiLU between layers").

        in_dim  = sound hidden_size (real: 1024)
        mid_dim = projector mid width
        out_dim = llm hidden_size (real: 2688)
    """

    def __init__(self, rngs: nnx.Rngs, in_dim: int, mid_dim: int, out_dim: int, eps: float):
        self.linear1 = nnx.Linear(in_dim, mid_dim, use_bias=False, rngs=rngs)
        # Plain RMSNorm scale (HF sound_projection.norm.weight). nnx.RMSNorm is
        # plain scale * normed, matching the HF convention.
        self.norm = nnx.RMSNorm(mid_dim, epsilon=eps, rngs=rngs)
        self.linear2 = nnx.Linear(mid_dim, out_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = jax.nn.silu(self.linear1(x))
        h = self.norm(h)
        return self.linear2(h)


# =============================================================================
# Omni config (bundles llm + vision + sound sub-configs)
# =============================================================================


@dataclass
class NemotronOmniConfig:
    """
    Composite config for Nemotron-3-Nano-Omni.

    Bundles:
        llm:    NemotronHConfig     — the HF-faithful Mamba2/GQA/MoE backbone
        vision: VisionEncoderConfig — RADIO ViT body (vit_hidden 1280 in the real model)
        sound:  AudioEncoderConfig  — Parakeet FastConformer body (hidden 1024, 24 layers)

    Plus the two projector widths and the HF placeholder token ids.

    HARD CONSTRAINT (validated): the projector OUTPUTS must equal the LLM hidden
    width, so spliced multimodal tokens have the same dimensionality as text
    embeddings (2688 in the real model).
    """

    llm: NemotronHConfig = field(default_factory=NemotronHConfig)
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    sound: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)

    # Vision projector: vit_hidden -> projector_hidden -> llm_hidden.
    # Real model: 1280 -> 20480 -> 2688.
    vision_projector_hidden: int = 20480

    # Sound projector: sound_hidden -> sound_projector_hidden -> llm_hidden.
    sound_projector_hidden: int = 2688

    # HF placeholder token ids spliced into the text stream by the processor.
    img_context_token_id: int = DEFAULT_IMG_CONTEXT_TOKEN_ID
    sound_context_token_id: int = DEFAULT_SOUND_CONTEXT_TOKEN_ID

    @property
    def vision_proj_in(self) -> int:
        """Vision projector input width = the ViT body output width."""
        return self.vision.hidden_dim

    @property
    def sound_proj_in(self) -> int:
        """Sound projector input width = the Conformer body output width."""
        return self.sound.hidden_dim

    def validate(self) -> None:
        """Assert cross-modality alignment constraints with named messages."""
        # The LLM backbone validates its own internal shape constraints.
        self.llm.validate()

        # Projectors must output exactly the LLM hidden width.
        assert self.vision_projector_hidden > 0, "vision_projector_hidden must be > 0"
        assert self.sound_projector_hidden > 0, "sound_projector_hidden must be > 0"

        # The placeholder ids must be valid token ids.
        assert 0 <= self.img_context_token_id < self.llm.vocab_size, (
            f"img_context_token_id ({self.img_context_token_id}) out of range "
            f"[0, vocab_size={self.llm.vocab_size})"
        )
        assert 0 <= self.sound_context_token_id < self.llm.vocab_size, (
            f"sound_context_token_id ({self.sound_context_token_id}) out of range "
            f"[0, vocab_size={self.llm.vocab_size})"
        )
        assert self.img_context_token_id != self.sound_context_token_id, (
            "img_context_token_id and sound_context_token_id must differ"
        )

    # =========================================================================
    # Presets
    # =========================================================================

    @classmethod
    def from_preset(cls, preset: str = "omni_30b") -> "NemotronOmniConfig":
        """
        Build a composite omni config from a named preset.

        Presets:
          * "omni_30b": real model. LLM 30B preset; RADIO ViT vit_hidden=1280,
                        projector 20480; Parakeet hidden=1024, 24 layers.
          * "tiny":     CPU-runnable shrink. LLM "tiny" preset; tiny ViT and tiny
                        Conformer whose projector outputs == tiny LLM hidden.
        """
        key = preset.strip().lower()

        if key in ("omni_30b", "omni-30b", "omni", "30b", "real"):
            llm = NemotronHConfig.from_preset("omni_30b")
            # RADIO ViT body, real dims (vit_hidden 1280, 32 blocks, 512px/16 patch).
            vision = VisionEncoderConfig(
                image_size=512,
                patch_size=16,
                in_channels=3,
                hidden_dim=1280,        # vit_hidden_size
                num_heads=8,            # 1280 / 160
                head_dim=160,
                num_layers=32,
                mlp_dim=1280 * 4,
                proj_dim=1280,          # body output stays in vit_hidden; mlp1 projects to llm
                teachers=[],
            )
            # Parakeet FastConformer body, real dims (hidden 1024, 24 layers).
            sound = AudioEncoderConfig(
                sample_rate=16000,
                n_mels=128,             # num_mel_bins
                hidden_dim=1024,
                num_heads=8,
                head_dim=128,           # 1024 / 8
                num_layers=24,
                ffn_dim=1024 * 4,
                conv_kernel_size=31,
                proj_dim=1024,          # body output stays in sound_hidden
            )
            cfg = cls(
                llm=llm,
                vision=vision,
                sound=sound,
                vision_projector_hidden=20480,
                sound_projector_hidden=2688,
                img_context_token_id=DEFAULT_IMG_CONTEXT_TOKEN_ID,
                sound_context_token_id=DEFAULT_SOUND_CONTEXT_TOKEN_ID,
            )
            cfg.validate()
            return cfg

        if key in ("tiny", "test", "cpu"):
            llm = NemotronHConfig.from_preset("tiny")  # hidden 64, vocab 512
            # Tiny ViT: 32x32 image, 16px patch -> 2x2 patches -> pixel-shuffle 1 token.
            vision = VisionEncoderConfig(
                image_size=32,
                patch_size=16,
                in_channels=3,
                hidden_dim=32,
                num_heads=4,
                head_dim=8,
                num_layers=2,
                mlp_dim=64,
                proj_dim=32,            # body output in vit_hidden=32
                teachers=[],
            )
            # Tiny Conformer.
            sound = AudioEncoderConfig(
                sample_rate=16000,
                n_mels=16,
                n_fft=128,
                frame_length=64,
                hop_length=32,
                hidden_dim=32,
                num_heads=4,
                head_dim=8,
                num_layers=2,
                ffn_dim=64,
                conv_kernel_size=7,
                proj_dim=32,
            )
            cfg = cls(
                llm=llm,
                vision=vision,
                sound=sound,
                vision_projector_hidden=48,   # tiny projector mid width
                sound_projector_hidden=48,
                # Tiny LLM vocab is 512; keep placeholder ids inside it but
                # distinct from real ids would still be < 512 anyway.
                img_context_token_id=DEFAULT_IMG_CONTEXT_TOKEN_ID,
                sound_context_token_id=DEFAULT_SOUND_CONTEXT_TOKEN_ID,
            )
            cfg.validate()
            return cfg

        raise ValueError(
            f"Unknown preset {preset!r}. Supported: 'omni_30b', 'tiny'."
        )


# =============================================================================
# In-place splice helper
# =============================================================================


def _splice_modality(
    inputs_embeds: jax.Array,
    token_ids: jax.Array,
    context_token_id: int,
    modality_tokens: jax.Array,
) -> jax.Array:
    """
    Overwrite the embeddings at placeholder positions with encoder features.

    This reproduces the HF in-place scatter (modeling.py: ``inputs_embeds[mask]
    = encoder_embeds``) in a JAX-functional way that is safe under eval_shape /
    jit (no boolean-mask data-dependent indexing of *output* size).

    Strategy (order-preserving, fully static-shaped):
      * Build a boolean placeholder mask of shape (B, L).
      * For each batch row, the i-th True position is filled with the i-th token
        from ``modality_tokens`` for that row. We implement this by computing,
        per position, the running count of placeholders seen so far (an index
        into ``modality_tokens``), then gathering.
      * Positions that are not placeholders keep their text embedding.

    Args:
        inputs_embeds:   (B, L, D)  current (text) embeddings, possibly already
                         partially spliced by a previous modality.
        token_ids:       (B, L)     the original integer token ids (to find
                         placeholders; we test against ``context_token_id``).
        context_token_id: the placeholder id for this modality (18 or 27).
        modality_tokens: (B, M, D)  projected encoder features to splice in.
                         M must be >= the max number of placeholder positions in
                         any row; extra encoder tokens are simply unused.

    Returns:
        (B, L, D) with placeholder positions overwritten by encoder features.

    Shape contract: D of modality_tokens must equal D of inputs_embeds (the LLM
    hidden width); enforced by an assert.
    """
    B, L, D = inputs_embeds.shape
    assert modality_tokens.shape[0] == B, (
        f"batch mismatch: inputs_embeds B={B} vs modality_tokens "
        f"B={modality_tokens.shape[0]}"
    )
    assert modality_tokens.shape[-1] == D, (
        f"modality token width {modality_tokens.shape[-1]} != LLM hidden {D}"
    )
    M = modality_tokens.shape[1]

    mask = token_ids == context_token_id  # (B, L) bool

    # Running placeholder index per position: how many placeholders are at or
    # before this position (exclusive prefix sum), clamped to [0, M-1] so the
    # gather is always in-bounds even if a row has more placeholders than M.
    # cumsum-1 gives the 0-based index of the current placeholder.
    prefix = jnp.cumsum(mask.astype(jnp.int32), axis=-1) - 1  # (B, L)
    gather_idx = jnp.clip(prefix, 0, M - 1)  # (B, L)

    # Gather the encoder token assigned to each position (junk where not a
    # placeholder; masked out below). batched gather over axis 1.
    gathered = jnp.take_along_axis(
        modality_tokens, gather_idx[:, :, None], axis=1
    )  # (B, L, D)

    mask_e = mask[:, :, None]  # (B, L, 1)
    return jnp.where(mask_e, gathered, inputs_embeds)


# =============================================================================
# The omni model
# =============================================================================


class NemotronOmni(nnx.Module):
    """
    Nemotron-3-Nano-Omni: text + vision + sound, HF-faithful in-place fusion.

    Submodules (named toward HF top-level namespaces):
        language_model   : NemotronHModel        (language_model.*)
        vision_model      : VisionEncoder         (vision_model.* body)
        vision_projector  : VisionProjector       (mlp1.*)
        sound_encoder     : AudioEncoder          (sound_encoder.* body)
        sound_projector   : SoundProjector        (sound_projection.*)

    Forward (HF-faithful):
        1. text_embeds = language_model.embeddings(input_ids)
        2. if pixels:  vis = vision_projector(vision_model(pixels));
                       splice vis into text_embeds at img_context_token_id
        3. if audio:   aud = sound_projector(sound_encoder(audio));
                       splice aud into text_embeds at sound_context_token_id
        4. run language_model layers/norm_f/lm_head over the fused embeds
        5. return logits (B, L, vocab_size)

    The text sequence length L is preserved end to end (in-place splice, NOT
    prefix concat). The caller is responsible for having placed enough
    placeholder tokens (id 18 / 27) in input_ids to receive every encoder token
    it cares about.
    """

    def __init__(self, config: NemotronOmniConfig, rngs: nnx.Rngs):
        config.validate()
        self.config = config

        llm_hidden = config.llm.hidden_size

        # ---- Language model backbone (language_model.*) ----
        self.language_model = NemotronHModel(rngs=rngs, config=config.llm)

        # ---- Vision body + projector (vision_model.* / mlp1.*) ----
        self.vision_model = VisionEncoder(config.vision, rngs=rngs)
        self.vision_projector = VisionProjector(
            rngs=rngs,
            in_dim=config.vision_proj_in,
            mid_dim=config.vision_projector_hidden,
            out_dim=llm_hidden,
        )

        # ---- Sound body + projector (sound_encoder.* / sound_projection.*) ----
        self.sound_encoder = AudioEncoder(config.sound, rngs=rngs)
        self.sound_projector = SoundProjector(
            rngs=rngs,
            in_dim=config.sound_proj_in,
            mid_dim=config.sound_projector_hidden,
            out_dim=llm_hidden,
            eps=config.llm.norm_eps,
        )

    # ------------------------------------------------------------------ encode
    def encode_vision(self, pixel_values: jax.Array) -> jax.Array:
        """pixels (B,H,W,C) -> projected vision tokens (B, N_vis, llm_hidden)."""
        feats = self.vision_model(pixel_values)        # (B, N_vis, vit_hidden)
        return self.vision_projector(feats)            # (B, N_vis, llm_hidden)

    def encode_sound(self, waveform: jax.Array) -> jax.Array:
        """waveform (B,T) -> projected sound tokens (B, N_aud, llm_hidden)."""
        feats = self.sound_encoder(waveform)           # (B, N_aud, sound_hidden)
        return self.sound_projector(feats)             # (B, N_aud, llm_hidden)

    # ----------------------------------------------------------------- forward
    def __call__(
        self,
        input_ids: jax.Array,
        pixel_values: jax.Array | None = None,
        waveform: jax.Array | None = None,
    ) -> jax.Array:
        """
        Args:
            input_ids:    (B, L) int token ids. Must contain img_context_token_id
                          (18) where image features go, and sound_context_token_id
                          (27) where sound features go.
            pixel_values: (B, H, W, C) float image, or None to skip vision.
            waveform:     (B, T) float audio, or None to skip sound.
        Returns:
            logits: (B, L, vocab_size) — same L as input_ids (in-place fusion).
        """
        lm = self.language_model

        # 1. Text embeddings (placeholders included; overwritten below).
        embeds = lm.embeddings(input_ids)  # (B, L, llm_hidden)

        # 2. Vision: encode, project, splice in place at id 18.
        if pixel_values is not None:
            vis_tokens = self.encode_vision(pixel_values)
            embeds = _splice_modality(
                embeds, input_ids, self.config.img_context_token_id, vis_tokens
            )

        # 3. Sound: encode, project, splice in place at id 27.
        if waveform is not None:
            aud_tokens = self.encode_sound(waveform)
            embeds = _splice_modality(
                embeds, input_ids, self.config.sound_context_token_id, aud_tokens
            )

        # 4. Run the LLM backbone on the fused embeddings (skip re-embedding).
        x = embeds
        for layer in lm.layers:
            x = layer(x)
        x = lm.norm_f(x)

        # 5. LM head -> logits over the (length-preserved) fused sequence.
        return lm.lm_head(x)


# =============================================================================
# MILESTONE-2 NAME-MAP TODOs (vision / sound HF namespace gaps)
# =============================================================================
#
# The LLM backbone (language_model.*) is fully mapped in nemotron_h.hf_name_map.
# The vision/sound encoder BODIES here use the vendored educational modules, whose
# parameter names do NOT yet match the HF safetensors. The following gaps MUST be
# closed by the milestone-2 converter (or by renaming the encoder modules):
#
# VISION  (HF prefix: vision_model.radio_model.model.)
#   * Patch embedding: HF stores patch_generator.{embedder.weight, pos_embed,
#     cls_token.token, video_embedder.weight}. Our PatchEmbedding has a single
#     proj.kernel and uses CONDITIONAL positional encoding (depthwise conv), NOT a
#     learned pos_embed table. -> Need a learned-pos-embed ViT variant, plus a
#     cls_token, to load HF RADIO patch_generator.*.
#   * Blocks: HF blocks.N.{norm1,norm2}.{weight,bias} are LayerNorm (with bias);
#     our VisionTransformerBlock uses nnx.RMSNorm (no bias). -> swap to LayerNorm.
#   * Attention: HF blocks.N.attn.{qkv.{weight,bias}, proj.{weight,bias}} is a
#     FUSED qkv with bias; ours is separate q/k/v/out with no bias. -> fuse + add bias.
#   * MLP: HF blocks.N.mlp.{fc1,fc2}.{weight,bias} has bias; ours has none, and
#     uses SiLU where HF RADIO uses GELU. -> add bias, switch activation.
#   * input_conditioner.{norm_mean,norm_std}: pixel normalization stats not modeled
#     here at all. -> add as frozen buffers applied before patch embed.
#   * mlp1 projector: HF folds the 2x2 pixel-shuffle (vit_hidden*4 = 5120) into
#     mlp1.0's Linear input. Our VisionProjector takes vit_hidden (1280) because
#     the vendored encoder already projects pixel-shuffle back to vit_hidden.
#     -> reconcile: either feed the projector the *5120-wide* pre-projection
#        features, or fold HF mlp1.0 weight accordingly. mlp1 indices {0,1,3}
#        => {Linear, activation(no params), Linear}; index 2 is the activation slot.
#
# SOUND  (HF prefix: sound_encoder.encoder.)
#   * feature_extractor.featurizer.{fb,window}: HF stores a frozen mel filterbank +
#     window. Our LogMelSpectrogram builds these from scratch. -> load HF fb/window
#     into our frozen Variables (verify mel params: 128 bins, 16kHz).
#   * subsampling: HF has subsampling.linear.{weight,bias} + subsampling.layers.{0,2,3,5,6}
#     conv weights (only those indices). Our ConvSubsampling is 3 plain stride-2
#     convs (conv1/conv2/conv3) with no final linear. -> restructure to HF layout.
#   * Conformer layers.N: HF uses post-norm with explicit norm_self_att / norm_conv /
#     norm_feed_forward1 / norm_feed_forward2 / norm_out (LayerNorm with bias). Ours
#     uses pre-norm RMSNorm. -> swap norms + reorder to post-norm.
#   * Attention: HF self_attn has relative-position params (relative_k_proj, bias_u,
#     bias_v) for Transformer-XL-style relative attention; ours is plain absolute.
#     -> add relative-position attention.
#   * Conv module: HF conv.norm is BatchNorm (running_mean/var/num_batches_tracked);
#     ours is RMSNorm. depthwise_conv / pointwise_conv1 / pointwise_conv2 names also
#     differ. -> map to HF conv submodule names, handle BN running stats.
#   * sound_projection: HF (linear1 -> norm -> linear2). Our SoundProjector matches
#     this shape exactly (linear1/norm/linear2) — should map directly once mid width
#     is set from the real checkpoint. (Lowest-risk piece.)
#
# AUXILIARY
#   * mlp1.{0,1,3} (vision projector) and sound_projection.* live at the HF TOP
#     LEVEL (not inside vision_model / sound_encoder). Our wrapper mirrors that by
#     keeping vision_projector / sound_projector as siblings of vision_model /
#     sound_encoder. Converter targets: self.vision_projector.* <- mlp1.*,
#     self.sound_projector.* <- sound_projection.* .
# =============================================================================
