#!/usr/bin/env python3
"""
Text inference with the converted Nemotron-3-Nano-Omni LLM backbone.

Loads the Orbax checkpoint produced by ``convert_hf_to_orbax.py`` into the
Flax/NNX backbone, then generates a continuation for one or more prompts. This
is the user-facing companion to ``coherence_gate.py``: the gate just proves the
weights decode sanely; this script is the thing you actually run to get text.

WHERE TO RUN
------------
The ~60GB bf16 model does not fit one TPU/GPU chip's HBM, so by default this runs
on the HOST (CPU) where RAM holds the weights and the MoE only computes the ~3B
active params/token. Force it explicitly with ``JAX_PLATFORMS=cpu`` (the bolt
inference job does this). TPU-sharded inference is future work.

CLI
---
    JAX_PLATFORMS=cpu python scripts/generate.py \
        --orbax gs://bucket/nemotron-omni-30b-orbax \
        --tokenizer /path/to/hf/checkpoint_dir   (or an HF repo id) \
        --prompt "The capital of France is" \
        --prompt "Write a haiku about TPUs." \
        --max-new-tokens 64 --temperature 0.7 --top-k 40 --seed 0

``--temperature 0`` (the default) is greedy/argmax decoding; any value > 0 enables
temperature + optional top-k sampling. ``--chat`` wraps each prompt with the
tokenizer's chat template (if it has one) instead of plain completion.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make `import jax_nemotron...` AND `import coherence_gate` work whether run from
# the repo root or the scripts dir. coherence_gate lives next to this file and
# already implements the (fiddly) Orbax restore we reuse here.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO_ROOT, "src"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coherence_gate import restore_backbone, _normalize_path  # noqa: E402


# =============================================================================
# Decoding
# =============================================================================


def _next_token(logits_1d, temperature: float, top_k: int, rng):
    """Pick the next token id from a 1-D logit vector. temperature<=0 -> greedy
    argmax; otherwise temperature-scale, optional top-k truncation, then sample."""
    import numpy as np

    logits = np.asarray(logits_1d, dtype=np.float32)
    if temperature is None or temperature <= 0.0:
        return int(np.argmax(logits))

    logits = logits / float(temperature)
    if top_k and top_k > 0 and top_k < logits.shape[-1]:
        # keep only the top_k highest logits; mask the rest to -inf.
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    logits -= logits.max()  # stabilize
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(rng.choice(probs.shape[-1], p=probs))


def generate(model, cfg, input_ids, max_new_tokens: int, temperature: float,
             top_k: int, eos_id, rng):
    """Autoregressively decode up to max_new_tokens (stops early on eos_id).

    Pads each step's sequence up to a multiple of cfg.chunk_size (the Mamba mixer
    needs seqlen % chunk_size == 0) and reads the logits at the true last real
    position. Returns the generated continuation ids (excluding the prompt)."""
    import jax.numpy as jnp
    import numpy as np

    chunk = cfg.chunk_size
    ids = [int(x) for x in np.asarray(input_ids).reshape(-1)]
    generated = []
    pad_id = 0  # any in-vocab id; positions > real_len-1 are never read.

    for _ in range(max_new_tokens):
        real_len = len(ids)
        padded_len = ((real_len + chunk - 1) // chunk) * chunk
        padded = ids + [pad_id] * (padded_len - real_len)
        batch = jnp.asarray(np.asarray(padded, dtype=np.int32)[None, :])  # (1, L)

        logits = model(batch)  # (1, L, vocab)
        if not bool(jnp.all(jnp.isfinite(logits[0, real_len - 1]))):
            raise RuntimeError("forward produced non-finite logits — bad checkpoint?")
        next_id = _next_token(np.asarray(logits[0, real_len - 1]),
                              temperature, top_k, rng)
        generated.append(next_id)
        ids.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break

    return generated


# =============================================================================
# Main
# =============================================================================


def _encode(tok, prompt: str, chat: bool):
    """Encode a prompt to ids, optionally via the tokenizer's chat template."""
    if chat and getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
    return tok(prompt, return_tensors="np")["input_ids"][0]


def run(orbax_path, tokenizer, prompts, max_new_tokens, temperature, top_k,
        seed, chat, preset):
    import numpy as np
    from transformers import AutoTokenizer

    cfg, model, _ = restore_backbone(_normalize_path(orbax_path), preset=preset)
    print(f"[generate] loading tokenizer {tokenizer!r} ...")
    tok = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    rng = np.random.default_rng(seed)
    mode = "greedy" if temperature <= 0 else f"sample(T={temperature}, top_k={top_k})"
    print(f"[generate] {mode}, max_new_tokens={max_new_tokens}, {len(prompts)} prompt(s)\n")

    for i, prompt in enumerate(prompts):
        input_ids = _encode(tok, prompt, chat)
        gen = generate(model, cfg, input_ids, max_new_tokens, temperature,
                       top_k, tok.eos_token_id, rng)
        cont = tok.decode(gen, skip_special_tokens=True)
        print("=" * 70)
        print(f"PROMPT [{i}]   : {prompt}")
        print(f"CONTINUATION : {cont}")
    print("=" * 70)
    print("[generate] done")
    return 0


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orbax", required=True,
                   help="Orbax checkpoint path (local dir or gs://bucket/path).")
    p.add_argument("--tokenizer", required=True,
                   help="Tokenizer: a local dir (e.g. the HF checkpoint) or an HF repo id.")
    p.add_argument("--preset", default="omni_30b",
                   help="Model preset (omni_30b | tiny). Must match the checkpoint.")
    p.add_argument("--prompt", action="append", dest="prompts", default=None,
                   help="Prompt to continue. Repeatable for a batch of prompts.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy/argmax (default); >0 enables sampling.")
    p.add_argument("--top-k", type=int, default=0,
                   help="Top-k truncation when sampling (0 = disabled).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chat", action="store_true",
                   help="Wrap each prompt with the tokenizer's chat template.")
    args = p.parse_args(argv)
    if not args.prompts:
        args.prompts = ["The capital of France is"]
    return args


def main(argv=None):
    a = _parse_args(argv)
    return run(a.orbax, a.tokenizer, a.prompts, a.max_new_tokens,
               a.temperature, a.top_k, a.seed, a.chat, a.preset)


if __name__ == "__main__":
    raise SystemExit(main())
