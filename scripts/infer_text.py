#!/usr/bin/env python3
"""
Load the converted Nemotron-3-Nano-Omni Orbax checkpoint and generate text.

Self-contained, single-file reference for the LLM (text-only) path: build the
backbone's ABSTRACT param tree with ``nnx.eval_shape`` (no allocation — a 30B
tree is free), restore the Orbax checkpoint into it, load the tokenizer, and
greedy-decode a continuation. Nothing here imports the other scripts, so it
reads top-to-bottom as "this is how you load the checkpoint and run it."

(``scripts/generate.py`` is the fuller tool — temperature/top-k sampling, chat
template, multi-prompt batching. ``scripts/coherence_gate.py`` is the
correctness gate. This file is the minimal example.)

WHERE TO RUN
------------
On CPU (``JAX_PLATFORMS=cpu``): the ~60GB bf16 model does not fit one TPU/GPU
chip's HBM, but host RAM holds it and the MoE computes only ~3B active
params/token, so a short greedy decode finishes in minutes.

    JAX_PLATFORMS=cpu python scripts/infer_text.py \
        --orbax gs://bucket/nemotron-omni-30b-orbax \
        --tokenizer /path/to/hf/checkpoint_dir   (or an HF repo id) \
        --prompt "The capital of France is" \
        --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import os
import sys

# Put src/ on the path so `import jax_nemotron` works from any cwd (resolved
# relative to THIS file, so it runs anywhere — laptop, Colab, the bolt host).
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# =============================================================================
# Flat-path helpers — slash paths match exactly what the converter wrote
# =============================================================================


def _normalize_path(path: str) -> str:
    """Local dirs -> absolute (Orbax requires it); gs:// and other URIs pass
    through UNCHANGED (never abspath a gs:// path — it mangles to /cwd/gs:/...)."""
    return path if "://" in path else os.path.abspath(path)


def _slash(key_tuple) -> str:
    return "/".join(str(k) for k in key_tuple)


def _insert(tree: dict, slash_path: str, value) -> None:
    node = tree
    parts = slash_path.split("/")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


# =============================================================================
# Load: abstract backbone (eval_shape) <- restored Orbax params
# =============================================================================


def load_backbone(orbax_path: str, preset: str = "omni_30b"):
    """Restore the Orbax checkpoint into the NemotronH backbone. Returns (cfg, model).

    Mirrors the converter's writer exactly: it built the param tree from
    ``nnx.eval_shape`` and saved it wrapped under "params"; we rebuild the same
    abstract tree, restore into it, then scatter the arrays back onto the model."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import orbax.checkpoint as ocp

    from jax_nemotron.config import NemotronHConfig
    from jax_nemotron.nemotron_h import NemotronHModel

    cfg = NemotronHConfig.from_preset(preset)
    cfg.validate()

    # Abstract tree only — no arrays allocated until restore fills them in.
    abstract = nnx.eval_shape(lambda: NemotronHModel(rngs=nnx.Rngs(0), config=cfg))
    graphdef, abstract_state = nnx.split(abstract)

    # Restore target = {"params": <nested ShapeDtypeStruct tree>}, structurally
    # identical to what was written.
    target: dict = {}
    for key_tuple, var in abstract_state.flat_state():
        val = getattr(var, "value", var)
        _insert(target, _slash(key_tuple), jax.ShapeDtypeStruct(tuple(val.shape), val.dtype))

    dest = _normalize_path(orbax_path)
    print(f"[infer] restoring Orbax checkpoint from {dest!r} ...")
    restored = ocp.StandardCheckpointer().restore(dest, {"params": target})["params"]

    # Scatter each restored leaf back onto the concrete model state.
    flat = dict(abstract_state.flat_state())
    for key_tuple, var in flat.items():
        node = restored
        for seg in _slash(key_tuple).split("/"):
            node = node[seg]
        var.value = jnp.asarray(node)
    model = nnx.merge(graphdef, nnx.State.from_flat_path(flat))
    print(f"[infer] backbone restored (preset={preset!r})")
    return cfg, model


# =============================================================================
# Greedy decode
# =============================================================================


def greedy_generate(model, cfg, input_ids, max_new_tokens: int, eos_id=None):
    """Greedy (argmax) decode. Pads each step to a multiple of cfg.chunk_size
    (the Mamba mixer needs seqlen % chunk_size == 0) and reads the logits at the
    true last real position. Returns the continuation ids (prompt excluded)."""
    import jax.numpy as jnp
    import numpy as np

    chunk = cfg.chunk_size
    ids = [int(x) for x in np.asarray(input_ids).reshape(-1)]
    generated = []
    for _ in range(max_new_tokens):
        real_len = len(ids)
        padded_len = ((real_len + chunk - 1) // chunk) * chunk
        batch = jnp.asarray(np.asarray(ids + [0] * (padded_len - real_len),
                                       dtype=np.int32)[None, :])
        logits = model(batch)  # (1, L, vocab)
        next_id = int(np.argmax(np.asarray(logits[0, real_len - 1])))
        generated.append(next_id)
        ids.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break
    return generated


# =============================================================================
# Main
# =============================================================================


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orbax", required=True,
                   help="Orbax checkpoint path (local dir or gs://bucket/path).")
    p.add_argument("--tokenizer", required=True,
                   help="Tokenizer: a local dir (e.g. the HF checkpoint) or an HF repo id.")
    p.add_argument("--preset", default="omni_30b",
                   help="Model preset (omni_30b | tiny). Must match the checkpoint.")
    p.add_argument("--prompt", action="append", dest="prompts", default=None,
                   help="Prompt to continue. Repeatable for multiple prompts.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args(argv)
    prompts = args.prompts or ["The capital of France is"]

    from transformers import AutoTokenizer

    cfg, model = load_backbone(args.orbax, args.preset)
    print(f"[infer] loading tokenizer {args.tokenizer!r} ...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    for i, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="np")["input_ids"][0]
        gen = greedy_generate(model, cfg, ids, args.max_new_tokens, tok.eos_token_id)
        print("=" * 70)
        print(f"PROMPT [{i}]   : {prompt}")
        print(f"CONTINUATION : {tok.decode(gen, skip_special_tokens=True)}")
    print("=" * 70)
    print("[infer] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
