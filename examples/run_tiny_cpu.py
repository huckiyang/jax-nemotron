#!/usr/bin/env python3
# =============================================================================
# run_tiny_cpu.py
#
# A single-file, laptop-CPU demo of the TINY NemotronOmni preset.
#
# What it does, end to end:
#   1. Builds the "tiny" NemotronOmni multimodal model (vision + audio + a tiny
#      hybrid Mamba/Attention/MoE language model) with RANDOM init weights --
#      no checkpoint required, no GPU required.
#   2. Prints the config and the parsed "hybrid pattern" (which mixer each LLM
#      layer uses: mamba / attention / moe).
#   3. Runs ONE fused forward pass that feeds an image, an audio clip, AND text
#      tokens through the model in a single call.
#   4. Prints the output logits shape and confirms they are finite.
#   5. Prints a clear "OK" line on success.
#
# Run it (from the repo root; CPU JAX lives in the project venv):
#   .venv/bin/python examples/run_tiny_cpu.py
# or, if you pip-installed the package:  python examples/run_tiny_cpu.py
#
# The weights are random, so the logits are meaningless numbers -- the point is
# that all the shapes line up and the whole multimodal pipeline executes.
# =============================================================================

# --- sys.path fix -----------------------------------------------------------
# The package lives under src/ and may not be pip-installed, so before we can
# `import jax_nemotron` we put that src/ dir on the import path. Every test and
# the converter does exactly this. The path is resolved relative to THIS file,
# so the script runs from any working directory on any machine (incl. Colab).
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# Public exports at the package top level are only NemotronOmni / its config
# (plus NemotronHConfig). That's all we need for the omni demo.
from jax_nemotron.nemotron_omni import NemotronOmni, NemotronOmniConfig


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Build the tiny config and the model.
    # -------------------------------------------------------------------------
    # `from_preset("tiny")` returns a fully-validated config sized for a CPU:
    # the LLM has hidden_size=64, vocab_size=512, chunk_size=4, and only 6
    # layers; the vision/audio encoders are correspondingly tiny. (The other
    # preset, "omni_30b", is the real 30B model -- do NOT instantiate that on a
    # laptop.) from_preset already calls .validate() for us.
    cfg = NemotronOmniConfig.from_preset("tiny")

    # Gotcha worth memorizing: the two model classes take their args in
    # DIFFERENT order. NemotronOmni is (config, rngs); NemotronHModel (the bare
    # LLM) is (rngs, config). To stay safe we always pass keywords. nnx.Rngs is
    # required -- it seeds the random weight init.
    model = NemotronOmni(config=cfg, rngs=nnx.Rngs(0))

    # -------------------------------------------------------------------------
    # 2. Print the config + parsed hybrid layer pattern.
    # -------------------------------------------------------------------------
    # The LLM is a "hybrid" stack: each layer is one of mamba / attention / moe.
    # The compact string `hybrid_override_pattern` encodes this (M=mamba,
    # *=attention, E=moe); `.parse_pattern()` expands it to a readable list.
    pattern_str = cfg.llm.hybrid_override_pattern
    pattern = cfg.llm.parse_pattern()

    print("=== TINY NemotronOmni config ===")
    print(f"  LLM hidden_size      : {cfg.llm.hidden_size}")
    print(f"  LLM vocab_size       : {cfg.llm.vocab_size}")
    print(f"  LLM num_hidden_layers: {cfg.llm.num_hidden_layers}")
    print(f"  LLM chunk_size       : {cfg.llm.chunk_size}  (seqlen must be a multiple of this)")
    print(f"  vision image_size    : {cfg.vision.image_size}  (square, channels-LAST)")
    print(f"  img_context_token_id : {cfg.img_context_token_id}")
    print(f"  sound_context_token_id: {cfg.sound_context_token_id}")
    print(f"  hybrid pattern string: {pattern_str!r}")
    print("  parsed layer mixers  :")
    for i, mixer in enumerate(pattern):
        print(f"    layer {i}: {mixer}")

    # -------------------------------------------------------------------------
    # 3. Build the three modality inputs.
    # -------------------------------------------------------------------------
    batch = 2
    key = jax.random.PRNGKey(0)
    k_img, k_aud = jax.random.split(key)

    # VISION: channels-LAST (B, H, W, C) -- NOT the PyTorch (B, C, H, W).
    # tiny image_size=32, 3 input channels.
    pixel_values = jax.random.normal(
        k_img, (batch, cfg.vision.image_size, cfg.vision.image_size, 3)
    )

    # AUDIO: a raw 1-D waveform per item, shape (B, T). It gets subsampled into
    # frames inside the encoder, so we need "enough" samples; 4096 is plenty for
    # the tiny model.
    waveform = jax.random.normal(k_aud, (batch, 4096))

    # The encoders emit a FIXED number of tokens for these inputs. We must
    # reserve exactly that many placeholder positions in input_ids, so measure
    # them first. (tiny -> 1 vision token, 16 sound tokens.)
    n_vis = model.encode_vision(pixel_values).shape[1]
    n_aud = model.encode_sound(waveform).shape[1]
    print()
    print(f"=== encoder token counts ===")
    print(f"  vision tokens (n_vis): {n_vis}")
    print(f"  sound  tokens (n_aud): {n_aud}")

    # -------------------------------------------------------------------------
    # 4. Assemble input_ids with placeholder tokens.
    # -------------------------------------------------------------------------
    # How the fused forward works: input_ids carries the special placeholder ids
    # 18 (image) and 27 (sound) at the positions where the encoded features
    # should be spliced in; everything else is ordinary text. The model embeds
    # the text ids, then OVERWRITES the placeholder positions in-place with the
    # vision/sound features. So L (sequence length) is preserved end to end.
    #
    # Constraint: L must be divisible by chunk_size (4 for tiny), or the Mamba
    # mixer raises ValueError. We lay out [vision | sound | text...] then round
    # the total length up to the next multiple of chunk_size.
    chunk = cfg.llm.chunk_size
    min_len = n_vis + n_aud + 1  # at least one real text token at the end
    L = ((min_len + chunk - 1) // chunk) * chunk  # round up to multiple of chunk

    # 5 is just "some ordinary text token" -- anything that is neither 18 nor 27.
    seq = np.full((batch, L), 5, dtype=np.int32)
    seq[:, 0:n_vis] = cfg.img_context_token_id              # 18  -> vision slots
    seq[:, n_vis:n_vis + n_aud] = cfg.sound_context_token_id  # 27  -> sound slots
    input_ids = jnp.asarray(seq)

    print()
    print(f"=== fused sequence layout ===")
    print(f"  total length L       : {L}  (divisible by chunk_size {chunk}: {L % chunk == 0})")
    print(f"  [0:{n_vis}] image slots, [{n_vis}:{n_vis + n_aud}] sound slots, rest text")

    # -------------------------------------------------------------------------
    # 5. ONE fused forward pass: vision + sound + text together.
    # -------------------------------------------------------------------------
    # pixel_values and waveform are optional keyword args; passing both runs the
    # full multimodal path. Returns logits of shape (B, L, vocab_size).
    logits = model(input_ids, pixel_values=pixel_values, waveform=waveform)

    print()
    print("=== forward output ===")
    print(f"  logits shape : {logits.shape}  (expected ({batch}, {L}, {cfg.llm.vocab_size}))")
    print(f"  logits dtype : {logits.dtype}  (float32 with random init)")

    # Sanity check: with random init the numbers are arbitrary, but they must be
    # finite (no NaN / inf), which tells us the whole pipeline ran cleanly.
    all_finite = bool(jnp.all(jnp.isfinite(logits)))
    print(f"  all finite   : {all_finite}")

    # Assert the contract so the demo fails loudly if anything regresses.
    assert logits.shape == (batch, L, cfg.llm.vocab_size), "unexpected logits shape"
    assert all_finite, "logits contain non-finite values"

    print()
    print("OK")


if __name__ == "__main__":
    main()
