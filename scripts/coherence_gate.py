"""
COHERENCE GATE — the decisive post-conversion correctness proof.

Run on a BIG-MEMORY / TPU host where the Orbax checkpoint lives (NOT the dev
sandbox; a 30B bf16 backbone needs ~60GB and a real accelerator):

    python scripts/coherence_gate.py \
        --orbax gs://bucket/nemotron-omni-30b-orbax \
        --preset omni_30b \
        --prompt "The capital of France is" \
        --max-new-tokens 32

------------------------------------------------------------------------------
WHY THIS GATE EXISTS — RIGHT SHAPES CAN STILL BE SCRAMBLED VALUES
------------------------------------------------------------------------------
The shape gate (tests/test_shape_gate.py), the name-map bijection check inside
the converter, and the converter's per-leaf ``assert arr.shape == target_shape``
together prove ONE thing: every target leaf got SOME source tensor of the
correct shape. That is necessary but NOT sufficient.

A conversion can pass every shape/structure check and still be WRONG in value:

  * A transpose applied to the wrong axis pair produces a correctly-shaped but
    transposed weight (e.g. a square matrix transposed: shape unchanged, values
    scrambled).
  * Two leaves of identical shape swapped with each other (q_proj <-> k_proj,
    two experts, gate vs up) pass every shape assert.
  * An RMSNorm copied with a spurious ``1 + weight`` (or missing it) is the right
    shape but the wrong magnitude.
  * A RoPE half-split-vs-interleaved mismatch leaves dimensions correct but
    rotates the wrong pairs.
  * bf16 truncation bugs, byte-order/endianness surprises in the safetensors
    read, or a stale shard handle returning the wrong tensor — all
    shape-preserving.

NONE of those are caught by shapes. The ONLY real proof that the HF -> Orbax
conversion preserved the model's learned function is to LOAD the converted
weights into our runtime and check the model still COMPUTES COHERENTLY: finite
logits, and a greedy decode that is not a single token repeated forever (the
classic signature of scrambled/zeroed weights). This gate does exactly that.

It restores the Orbax checkpoint with the SAME pattern the converter wrote it
(StandardCheckpointer, params wrapped under "params", restored into an abstract
target tree built from OUR model via nnx.eval_shape), then runs a real greedy
decode against the actual NVIDIA tokenizer and asserts coherence.

------------------------------------------------------------------------------
WHAT IT ASSERTS
------------------------------------------------------------------------------
  1. Every restored leaf is finite (no nan/inf snuck through the cast/write).
  2. The prompt forward produces finite logits.
  3. The greedy continuation is NOT degenerate (not the same token id repeated)
     — a hard, automatic signal that values are not scrambled to a constant.
  4. Prints the decoded text for a human to eyeball ("the capital of France is
     Paris" reads sane; word salad does not).

This gate intentionally restores ONLY the NemotronH language-model backbone
(milestone-1 converter scope). The Orbax checkpoint is the LLM backbone the
converter produces.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make `import jax_nemotron...` work whether run from repo root or scripts/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Default tokenizer = the Omni checkpoint this gate is built for. Override with
# --tokenizer (e.g. a local dir already on the host) to avoid a gated re-download.
TOKENIZER_NAME = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"


# =============================================================================
# Path helper (gs:// safe — mirrors the converter's _normalize_out)
# =============================================================================


def _normalize_path(path: str) -> str:
    """Local dirs -> absolute path (Orbax requires it); remote URIs (gs://, ...)
    pass through UNCHANGED. NEVER os.path.abspath a gs:// path — it mangles to
    '/cwd/gs:/...'. Identical convention to the converter."""
    if "://" in path:
        return path
    return os.path.abspath(path)


# =============================================================================
# Flat-path helpers (slash paths match the converter exactly)
# =============================================================================


def _slash(key_tuple) -> str:
    """Join an nnx flat_state tuple key into a slash path
    ('layers', 3, 'mixer', 'in_proj', 'kernel') -> 'layers/3/mixer/in_proj/kernel'."""
    return "/".join(str(k) for k in key_tuple)


def _insert(tree: dict, slash_path: str, value) -> None:
    """Insert value into a nested dict at slash_path (mirrors the converter's
    _insert so the restore-target structure matches the on-disk tree)."""
    parts = slash_path.split("/")
    node = tree
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


# =============================================================================
# Build abstract backbone + restore the Orbax checkpoint into it
# =============================================================================


def _build_abstract_backbone(preset: str):
    """Return (config, graphdef, abstract_state) for the NemotronH LLM backbone
    via nnx.eval_shape — NO real arrays allocated (a 30B abstract tree is free)."""
    from flax import nnx

    from jax_nemotron.config import NemotronHConfig
    from jax_nemotron.nemotron_h import NemotronHModel

    cfg = NemotronHConfig.from_preset(preset)
    cfg.validate()

    def init_fn():
        return NemotronHModel(rngs=nnx.Rngs(0), config=cfg)

    abstract_model = nnx.eval_shape(init_fn)
    graphdef, abstract_state = nnx.split(abstract_model)
    return cfg, graphdef, abstract_state


def _abstract_params_target(abstract_state) -> dict:
    """Build the {"params": <nested abstract tree>} restore TARGET that
    StandardCheckpointer.restore needs: a nested dict whose leaves are
    jax.ShapeDtypeStruct, structurally identical to the {"params": params} dict
    the converter wrote. This is the eval_shape abstract-target restore pattern."""
    import jax

    params: dict = {}
    for key_tuple, var in abstract_state.flat_state():
        val = getattr(var, "value", var)  # VariableState .value carries the SDS
        sds = jax.ShapeDtypeStruct(tuple(val.shape), val.dtype)
        _insert(params, _slash(key_tuple), sds)
    return {"params": params}


def _scatter_into_state(abstract_state, restored_params: dict):
    """Set each abstract leaf's value from the restored nested dict (looked up by
    slash path), returning a concrete nnx state ready for nnx.merge. A missing
    path raises KeyError, which is the structural check we want."""
    import jax.numpy as jnp
    from flax import nnx

    flat = dict(abstract_state.flat_state())
    for key_tuple, var in flat.items():
        node = restored_params
        for seg in _slash(key_tuple).split("/"):
            node = node[seg]
        var.value = jnp.asarray(node)
    return nnx.State.from_flat_path(flat)


def restore_backbone(orbax_path: str, preset: str):
    """Restore the Orbax checkpoint into our backbone and return the live model.

    Uses the SAME plumbing the converter wrote with: StandardCheckpointer, the
    param tree wrapped under "params", restored into the eval_shape abstract
    target tree."""
    import orbax.checkpoint as ocp
    from flax import nnx

    cfg, graphdef, abstract_state = _build_abstract_backbone(preset)
    target = _abstract_params_target(abstract_state)

    dest = _normalize_path(orbax_path)
    print(f"[coherence] restoring Orbax checkpoint from {dest!r} ...")
    ckpter = ocp.StandardCheckpointer()
    restored = ckpter.restore(dest, target)
    restored_params = restored["params"]

    restored_state = _scatter_into_state(abstract_state, restored_params)
    model = nnx.merge(graphdef, restored_state)
    print(f"[coherence] backbone restored (preset={preset!r})")
    return cfg, model, restored_params


# =============================================================================
# Coherence checks
# =============================================================================


def _assert_all_finite(restored_params: dict) -> int:
    """Walk the restored nested dict and assert every leaf is finite. Returns the
    leaf count. A non-finite leaf means the cast/write corrupted a value."""
    import jax.numpy as jnp
    import numpy as np

    n = 0

    def _walk(node, path):
        nonlocal n
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, path + "/" + str(k) if path else str(k))
        else:
            arr = np.asarray(jnp.asarray(node, dtype=jnp.float32))
            if not bool(np.all(np.isfinite(arr))):
                raise AssertionError(
                    f"COHERENCE GATE FAILED: restored leaf {path!r} has "
                    f"non-finite values (nan/inf)"
                )
            n += 1

    _walk(restored_params, "")
    return n


def greedy_decode(model, cfg, input_ids, max_new_tokens: int):
    """Greedy decode max_new_tokens from the backbone. Pads each step's sequence
    up to a multiple of cfg.chunk_size (the Mamba mixer requires seqlen %
    chunk_size == 0) and reads the logits at the true last real position.

    Returns the list of generated token ids (the continuation only)."""
    import jax.numpy as jnp
    import numpy as np

    chunk = cfg.chunk_size
    ids = list(int(x) for x in np.asarray(input_ids).reshape(-1))
    generated = []
    pad_id = 0  # any in-vocab id; only positions <= real_len-1 are read.

    for _ in range(max_new_tokens):
        real_len = len(ids)
        padded_len = ((real_len + chunk - 1) // chunk) * chunk
        padded = ids + [pad_id] * (padded_len - real_len)
        batch = jnp.asarray(np.asarray(padded, dtype=np.int32)[None, :])  # (1, L)

        logits = model(batch)  # (1, L, vocab)
        if not bool(jnp.all(jnp.isfinite(logits))):
            raise AssertionError(
                "COHERENCE GATE FAILED: forward produced non-finite logits"
            )
        last = np.asarray(logits[0, real_len - 1])  # logits at true last token
        next_id = int(np.argmax(last))
        generated.append(next_id)
        ids.append(next_id)

    return generated


def assert_not_degenerate(generated):
    """A scrambled/zeroed conversion almost always greedy-decodes a SINGLE token
    repeated forever (argmax of a near-constant logit vector). Reject that."""
    if len(generated) == 0:
        raise AssertionError("COHERENCE GATE FAILED: no tokens generated")
    if len(set(generated)) == 1:
        raise AssertionError(
            "COHERENCE GATE FAILED: degenerate output — the same token id "
            f"({generated[0]}) was produced {len(generated)} times. This is the "
            "classic signature of scrambled/zeroed weights (right shapes, wrong "
            "values)."
        )


# =============================================================================
# Main
# =============================================================================


def run(orbax_path: str, preset: str, prompt: str, max_new_tokens: int,
        tokenizer: str = TOKENIZER_NAME) -> int:
    from transformers import AutoTokenizer

    # 1. restore weights into our backbone (converter's save/restore pattern).
    cfg, model, restored_params = restore_backbone(orbax_path, preset)

    # 2. every restored leaf must be finite.
    n_leaves = _assert_all_finite(restored_params)
    print(f"[coherence] all {n_leaves} restored leaves are finite")

    # 3. tokenize the prompt with the REAL NVIDIA tokenizer.
    print(f"[coherence] loading tokenizer {tokenizer!r} ...")
    tok = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    enc = tok(prompt, return_tensors="np")
    input_ids = enc["input_ids"][0]
    print(f"[coherence] prompt {prompt!r} -> {len(input_ids)} tokens")

    # 4. greedy decode + coherence asserts.
    generated = greedy_decode(model, cfg, input_ids, max_new_tokens)
    assert_not_degenerate(generated)

    full_ids = list(int(x) for x in input_ids) + generated
    decoded = tok.decode(full_ids, skip_special_tokens=True)
    cont = tok.decode(generated, skip_special_tokens=True)

    print(f"[coherence] generated {len(generated)} tokens, "
          f"{len(set(generated))} distinct")
    print("---------------------------------------------------------------")
    print(f"PROMPT      : {prompt}")
    print(f"CONTINUATION: {cont}")
    print(f"FULL        : {decoded}")
    print("---------------------------------------------------------------")
    print("COHERENCE GATE PASSED")
    return 0


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Coherence gate: load the converted Orbax backbone and prove "
                    "it decodes coherently (the decisive value-correctness proof)."
    )
    p.add_argument("--orbax", required=True,
                   help="Orbax checkpoint path (local dir or gs://bucket/path).")
    p.add_argument("--preset", default="omni_30b",
                   help="Model preset (omni_30b | tiny). Default omni_30b.")
    p.add_argument("--prompt", default="The capital of France is",
                   help="Prompt to greedy-decode from.")
    p.add_argument("--max-new-tokens", type=int, default=32,
                   help="Number of tokens to greedily generate.")
    p.add_argument("--tokenizer", default=TOKENIZER_NAME,
                   help="Tokenizer to load (HF repo id or a local dir already on "
                        f"the host). Default {TOKENIZER_NAME!r}.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    return run(
        orbax_path=args.orbax,
        preset=args.preset,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        tokenizer=args.tokenizer,
    )


if __name__ == "__main__":
    raise SystemExit(main())
