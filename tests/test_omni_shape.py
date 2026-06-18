"""
Shape gate for the Nemotron-3-Nano-Omni wrapper (no pytest, runnable directly).

Run:
    .venv/bin/python tests/test_omni_shape.py

What this gate proves on CPU with random weights (NO real checkpoint):
  1. The "tiny" omni config validates and its projector outputs == LLM hidden.
  2. jax.eval_shape on NemotronOmni init produces a param tree WITHOUT allocation,
     and that tree contains the five expected submodule subtrees:
         language_model, vision_model, vision_projector, sound_encoder, sound_projector
  3. The tiny vision/sound encoders produce some tokens (we print the counts).
  4. A tiny random-param forward over a FUSED [vision|sound|text] sequence —
     where input_ids carry img_context_token_id(18) and sound_context_token_id(27)
     placeholders — returns finite logits of shape (B, L, vocab_size), with L
     preserved (in-place splice, not prefix concat).

Prints "OMNI SHAPE GATE PASSED" on success; raises with a clear message otherwise.

This is a STRUCTURE/SHAPE gate, not a value-correctness proof.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import jax
import jax.numpy as jnp
from flax import nnx

from jax_nemotron.nemotron_omni import (
    NemotronOmniConfig,
    NemotronOmni,
)


def _fail(msg: str):
    raise AssertionError(f"OMNI SHAPE GATE FAILED: {msg}")


def _flatten_state_paths(state) -> dict:
    """Flatten an nnx state pytree into {slash_path: leaf} via JAX keypaths."""
    leaves_with_paths = jax.tree_util.tree_leaves_with_path(
        state, is_leaf=lambda n: hasattr(n, "value")
    )
    out = {}
    for path, leaf in leaves_with_paths:
        parts = []
        for p in path:
            if isinstance(p, jax.tree_util.DictKey):
                parts.append(str(p.key))
            elif isinstance(p, jax.tree_util.GetAttrKey):
                parts.append(str(p.name))
            elif isinstance(p, jax.tree_util.SequenceKey):
                parts.append(str(p.idx))
            else:
                parts.append(str(p))
        out["/".join(parts)] = leaf
    return out


def main():
    # ----- 1. config -----
    cfg = NemotronOmniConfig.from_preset("tiny")
    cfg.validate()
    llm_hidden = cfg.llm.hidden_size
    print(f"[omni] tiny config: llm_hidden={llm_hidden}, "
          f"vision_hidden={cfg.vision.hidden_dim}, sound_hidden={cfg.sound.hidden_dim}, "
          f"img_tok={cfg.img_context_token_id}, snd_tok={cfg.sound_context_token_id}")

    # Projector outputs must equal LLM hidden (the hard alignment constraint).
    # We confirm structurally below via the forward; here check the config widths.
    if cfg.vision_proj_in <= 0 or cfg.sound_proj_in <= 0:
        _fail("projector input widths must be > 0")

    # ----- 2. eval_shape on init (no allocation) -----
    def init_fn():
        return NemotronOmni(config=cfg, rngs=nnx.Rngs(0))

    abstract_model = nnx.eval_shape(init_fn)
    _, abstract_state = nnx.split(abstract_model)
    paths = _flatten_state_paths(abstract_state)
    leaf_count = len(paths)
    print(f"[omni] eval_shape produced {leaf_count} param/variable leaves")
    if leaf_count == 0:
        _fail("eval_shape produced an empty param tree")

    pathset = set(paths.keys())

    def _require_prefix(prefix: str, what: str):
        if not any(p == prefix or p.startswith(prefix + "/") for p in pathset):
            sample = sorted(pathset)[:20]
            _fail(f"expected subtree {what!r} (prefix {prefix!r}) not found. "
                  f"sample paths: {sample}")

    _require_prefix("language_model", "LLM backbone")
    _require_prefix("vision_model", "RADIO ViT body")
    _require_prefix("vision_projector", "vision projector (mlp1)")
    _require_prefix("sound_encoder", "Parakeet body")
    _require_prefix("sound_projector", "sound projector (sound_projection)")

    # Projector output shapes: fc2/linear2 kernel out-dim must equal LLM hidden.
    def _shape(path):
        leaf = paths[path]
        val = getattr(leaf, "value", leaf)
        return tuple(val.shape)

    vproj_fc2 = next(p for p in pathset if p.startswith("vision_projector/fc2"))
    if _shape(vproj_fc2)[-1] != llm_hidden:
        _fail(f"vision_projector fc2 out-dim {_shape(vproj_fc2)} != llm hidden {llm_hidden}")
    sproj_l2 = next(p for p in pathset if p.startswith("sound_projector/linear2"))
    if _shape(sproj_l2)[-1] != llm_hidden:
        _fail(f"sound_projector linear2 out-dim {_shape(sproj_l2)} != llm hidden {llm_hidden}")
    print(f"[omni] structure OK; projector outputs == llm hidden ({llm_hidden})")

    # ----- 3. instantiate + measure encoder token counts -----
    model = NemotronOmni(config=cfg, rngs=nnx.Rngs(0))
    batch = 2
    key = jax.random.PRNGKey(0)
    k_img, k_aud = jax.random.split(key)

    # Tiny image: 32x32x3 -> (32/16)^2 = 4 patches -> pixel-shuffle 2x2 -> 1 token.
    pixel_values = jax.random.normal(k_img, (batch, cfg.vision.image_size,
                                             cfg.vision.image_size, 3))
    vis_tokens = model.encode_vision(pixel_values)
    n_vis = vis_tokens.shape[1]
    if vis_tokens.shape[-1] != llm_hidden:
        _fail(f"vision tokens width {vis_tokens.shape[-1]} != {llm_hidden}")
    print(f"[omni] vision encoder -> {n_vis} tokens of width {vis_tokens.shape[-1]}")

    # Tiny audio: enough samples for a few frames after 8x subsampling.
    n_samples = 4096
    waveform = jax.random.normal(k_aud, (batch, n_samples))
    aud_tokens = model.encode_sound(waveform)
    n_aud = aud_tokens.shape[1]
    if aud_tokens.shape[-1] != llm_hidden:
        _fail(f"sound tokens width {aud_tokens.shape[-1]} != {llm_hidden}")
    print(f"[omni] sound encoder -> {n_aud} tokens of width {aud_tokens.shape[-1]}")

    # ----- 4. fused forward with in-place placeholder splice -----
    # Build a token sequence of length L (divisible by mamba chunk_size) that
    # contains exactly n_vis image placeholders, n_aud sound placeholders, and
    # the rest ordinary text tokens. Layout: [img...][snd...][text...].
    chunk = cfg.llm.chunk_size
    # Total needed >= n_vis + n_aud + at least 1 text token, rounded up to chunk.
    min_len = n_vis + n_aud + 1
    L = ((min_len + chunk - 1) // chunk) * chunk
    print(f"[omni] fused seqlen L={L} (chunk_size={chunk}, "
          f"{n_vis} img + {n_aud} snd + {L - n_vis - n_aud} text)")

    img_id = cfg.img_context_token_id
    snd_id = cfg.sound_context_token_id
    # Ordinary text tokens: pick ids that are neither placeholder id.
    text_fill = 5
    assert text_fill not in (img_id, snd_id)

    import numpy as np
    seq = np.full((batch, L), text_fill, dtype=np.int32)
    seq[:, 0:n_vis] = img_id
    seq[:, n_vis:n_vis + n_aud] = snd_id
    input_ids = jnp.asarray(seq)

    logits = model(input_ids, pixel_values=pixel_values, waveform=waveform)
    expected = (batch, L, cfg.llm.vocab_size)
    if tuple(logits.shape) != expected:
        _fail(f"logits shape {tuple(logits.shape)} != {expected}")
    if not bool(jnp.all(jnp.isfinite(logits))):
        _fail("fused logits contain non-finite values (nan/inf)")
    print(f"[omni] fused forward OK: logits {tuple(logits.shape)}, finite, "
          f"dtype={logits.dtype}")

    # ----- 5. text-only forward still works (modalities optional) -----
    text_only_ids = jnp.asarray(np.full((batch, L), text_fill, dtype=np.int32))
    logits_txt = model(text_only_ids)
    if tuple(logits_txt.shape) != expected:
        _fail(f"text-only logits shape {tuple(logits_txt.shape)} != {expected}")
    if not bool(jnp.all(jnp.isfinite(logits_txt))):
        _fail("text-only logits contain non-finite values")
    print(f"[omni] text-only forward OK: logits {tuple(logits_txt.shape)}, finite")

    print("OMNI SHAPE GATE PASSED")


if __name__ == "__main__":
    main()
