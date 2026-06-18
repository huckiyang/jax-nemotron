"""
Nemotron 3 Nano Omni — Full Multimodal Model in JAX/Flax NNX.

This module assembles the complete omni-modal model described in:
  "Nemotron 3 Nano Omni: Efficient and Open Multimodal Intelligence"
  arXiv:2604.24954

It wraps the existing Nemotron 3 Nano language model backbone (nemotron.py)
with a vision encoder (vision_encoder.py) and an audio encoder
(audio_encoder.py), connecting them via lightweight MLP projectors:

  ┌─────────────────────────────────────────────────────────────┐
  │  [image]  → VisionEncoder  → VisionProjector  ──────────┐  │
  │  [audio]  → AudioEncoder   → AudioProjector   ──────────┤  │
  │  [text]   → TokenEmbedding                    ──────────┤  │
  │                                                          ↓  │
  │               [ vis_tokens | aud_tokens | text_tokens ]     │
  │                              ↓                              │
  │         NemotronNano backbone (Mamba + Attention + MoE)     │
  │                              ↓                              │
  │                       LM head → logits                      │
  └─────────────────────────────────────────────────────────────┘

Design decisions:
- Token fusion order: [visual | audio | text] — modality tokens act as a
  "prefix" that the LLM can attend back to when generating text responses.
- We bypass NemotronNanoBlock.__call__ (which starts with embedding) and
  instead call the blocks directly after manually fusing the token sequences.
- All modality tokens are in d_model space — same width as text embeddings —
  so the LLM backbone sees a uniform sequence regardless of modality.
- Any modality is optional: pass None to skip it.

Paper's staged training curriculum (for reference, not implemented here):
  Stage 0: VisionProjector only (LLM + vision encoder frozen)
  Stage 1: LLM + VisionEncoder jointly
  Stage 2: AudioProjector only (everything else frozen)
  Stage 3: AudioEncoder + AudioProjector (LLM + vision frozen)
  Stage 4: All parameters, short context (16k)
  Stage 5: All parameters, long context (48k)
  Stage 6: All parameters, ultra-long context (256k)
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from nemotron import NemotronConfig, NemotronNanoBlock
from vision_encoder import VisionEncoder, VisionEncoderConfig, VisionProjector
from audio_encoder import AudioEncoder, AudioEncoderConfig, AudioProjector


# =============================================================================
# Config
# =============================================================================


@dataclass
class NemotronOmniConfig:
    """
    Composite configuration for the full Nemotron 3 Nano Omni model.

    Bundles three sub-configs:
      llm:    NemotronConfig       — the Mamba+Attention+MoE language model
      vision: VisionEncoderConfig  — ViT encoder + MLP projector
      audio:  AudioEncoderConfig   — Conformer encoder + MLP projector

    Constraint: vision.proj_dim and audio.proj_dim must both equal llm.d_model
    so that projected multimodal tokens have the same width as text embeddings.
    """

    llm: NemotronConfig = field(default_factory=NemotronConfig)
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    audio: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)

    def validate(self) -> None:
        """Check alignment constraints between sub-configs."""
        assert self.vision.proj_dim == self.llm.d_model, (
            f"vision.proj_dim ({self.vision.proj_dim}) must equal "
            f"llm.d_model ({self.llm.d_model}) so visual tokens fit the LLM"
        )
        assert self.audio.proj_dim == self.llm.d_model, (
            f"audio.proj_dim ({self.audio.proj_dim}) must equal "
            f"llm.d_model ({self.llm.d_model}) so audio tokens fit the LLM"
        )
        # Validate the LLM's own internal shape constraints.
        self.llm.validate()


# =============================================================================
# Nemotron 3 Nano Omni — Full Multimodal Model
# =============================================================================


class NemotronOmniModel(nnx.Module):
    """
    Nemotron 3 Nano Omni: encoder-projector-decoder multimodal model.

    Accepts any combination of text, image, and audio inputs. All modalities
    are optional — pass None to skip them. Text (input_ids) is required.

    Forward pass overview:
      1. Embed text input_ids         → text_tokens  (B, L,     d_model)
      2. Encode + project image       → vis_tokens   (B, N_vis, d_model)  [optional]
      3. Encode + project audio       → aud_tokens   (B, N_aud, d_model)  [optional]
      4. Concatenate in order:   [vis_tokens | aud_tokens | text_tokens]
      5. Run through all LLM backbone blocks
      6. LM head → logits over the full fused sequence

    Example (tiny config defaults):
      input_ids:     (1, 10)           → 10 text tokens
      pixel_values:  (1, 64, 64, 3)   → 4 visual tokens (after 4× pixel shuffle)
      waveform:      (1, 16000)        → ~12 audio tokens (after 8× downsampling)
      fused:         [4 + 12 + 10] = 26 tokens total
      logits:        (1, 26, vocab_size)

    Note: Logits are produced for every position (visual, audio, text alike).
    During training, only text positions are typically included in the LM loss
    (controlled by a loss mask in the training loop).

    Args:
        config: NemotronOmniConfig
        rngs:   Flax NNX random key container
    """

    def __init__(self, config: NemotronOmniConfig, rngs: nnx.Rngs):
        config.validate()
        self.config = config

        # ── Language model backbone ──────────────────────────────────────────
        # We store the full NemotronNanoBlock so we can access its embedding,
        # blocks, final_norm, and lm_head individually for the fused forward pass.
        self.llm = NemotronNanoBlock(config=config.llm, rngs=rngs)

        # ── Vision components ────────────────────────────────────────────────
        # Encoder: image → patch tokens in encoder hidden_dim space
        self.vision_encoder = VisionEncoder(config.vision, rngs=rngs)
        # Projector: encoder hidden_dim → LLM d_model
        self.vision_projector = VisionProjector(
            in_dim=config.vision.hidden_dim,
            out_dim=config.llm.d_model,
            rngs=rngs,
        )

        # ── Audio components ─────────────────────────────────────────────────
        # Encoder: waveform → conformer tokens in encoder hidden_dim space
        self.audio_encoder = AudioEncoder(config.audio, rngs=rngs)
        # Projector: encoder hidden_dim → LLM d_model
        self.audio_projector = AudioProjector(
            in_dim=config.audio.hidden_dim,
            out_dim=config.llm.d_model,
            rngs=rngs,
        )

    def encode_image(self, pixel_values: jax.Array) -> jax.Array:
        """
        Encode an image into LLM-compatible visual tokens.

        Runs the vision encoder (patch embed → CPE → ViT blocks → pixel shuffle)
        followed by the MLP projector to align dimensions with the LLM.

        Args:
            pixel_values: (batch, H, W, C) float32 image
        Returns:
            visual_tokens: (batch, N_vis, d_model) — ready to concat with text
        """
        # Encoder: image → patch tokens in encoder hidden space.
        enc = self.vision_encoder(pixel_values)     # (B, N_vis, vision_hidden)
        # Projector: align to LLM hidden dimension.
        return self.vision_projector(enc)           # (B, N_vis, d_model)

    def encode_audio(self, waveform: jax.Array) -> jax.Array:
        """
        Encode a raw audio waveform into LLM-compatible audio tokens.

        Runs the audio encoder (log-mel → conv subsampling → conformer blocks)
        followed by the MLP projector to align dimensions with the LLM.

        Args:
            waveform: (batch, T) float32 raw audio at config.audio.sample_rate Hz
        Returns:
            audio_tokens: (batch, N_aud, d_model) — ready to concat with text
        """
        # Encoder: waveform → conformer tokens in encoder hidden space.
        enc = self.audio_encoder(waveform)          # (B, N_aud, audio_hidden)
        # Projector: align to LLM hidden dimension.
        return self.audio_projector(enc)            # (B, N_aud, d_model)

    def __call__(
        self,
        input_ids: jax.Array,
        pixel_values: jax.Array | None = None,
        waveform: jax.Array | None = None,
    ) -> jax.Array:
        """
        Full multimodal forward pass.

        Fusion order: [visual tokens | audio tokens | text tokens]
        Multimodal tokens act as a "prefix" — the text tokens can attend back
        to the visual/audio context throughout every LLM layer.

        Args:
            input_ids:    (batch, L) integer token ids — always required
            pixel_values: (batch, H, W, C) float image — pass None to skip
            waveform:     (batch, T) float audio — pass None to skip
        Returns:
            logits: (batch, total_T, vocab_size)
                    total_T = N_vis (if image) + N_aud (if audio) + L
        """
        # ── Step 1: Embed text tokens ─────────────────────────────────────────
        # Use the LLM's token embedding table (same as a text-only forward pass).
        text_tokens = self.llm.embedding(input_ids)  # (B, L, d_model)

        # ── Step 2: Collect all token sequences ───────────────────────────────
        # We build a list and concatenate once — clean and modality-agnostic.
        token_parts = []

        if pixel_values is not None:
            # Visual prefix: encode image → project to d_model.
            vis_tokens = self.encode_image(pixel_values)  # (B, N_vis, d_model)
            token_parts.append(vis_tokens)

        if waveform is not None:
            # Audio prefix: encode waveform → project to d_model.
            aud_tokens = self.encode_audio(waveform)      # (B, N_aud, d_model)
            token_parts.append(aud_tokens)

        # Text tokens always go last so the LLM generates text as output.
        token_parts.append(text_tokens)

        # ── Step 3: Fuse all tokens into one flat sequence ────────────────────
        x = jnp.concatenate(token_parts, axis=1)  # (B, total_T, d_model)

        # ── Step 4: Run through all LLM backbone blocks ───────────────────────
        # We call the blocks directly here (bypassing llm.__call__ which would
        # re-embed input_ids) because we've already built the full embedding.
        for block in self.llm.blocks:
            x = block(x)

        # ── Step 5: Final normalization ───────────────────────────────────────
        x = self.llm.final_norm(x)

        # ── Step 6: LM head — project to vocabulary ───────────────────────────
        # Logits are produced for every position (visual, audio, and text).
        # Typically only text positions contribute to the language modelling loss.
        logits = self.llm.lm_head(x)  # (B, total_T, vocab_size)

        return logits
