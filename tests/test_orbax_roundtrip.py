"""
Orbax save/restore round-trip gate (no pytest, runnable directly on CPU).

Run from the repo root:
    .venv/bin/python tests/test_orbax_roundtrip.py

------------------------------------------------------------------------------
WHAT THIS GATE PROVES (and why it matters)
------------------------------------------------------------------------------
The converter (scripts/convert_hf_to_orbax.py) builds a plain nested dict of
leaves, wraps it under a top-level "params" key, and writes it with
``orbax.checkpoint.StandardCheckpointer().save(dest, {"params": params})``. The
coherence gate (scripts/coherence_gate.py) then RESTORES that checkpoint into an
abstract target tree built from OUR model via ``nnx.eval_shape`` and runs a real
forward.

This test PROVES the exact save/restore plumbing the coherence gate depends on,
on CPU with the tiny preset and random weights:

  1. Build the tiny NemotronH backbone and extract its concrete param tree.
  2. Save it with StandardCheckpointer (same {"params": ...} wrapping the
     converter uses) to a tempdir.
  3. Restore into an eval_shape abstract target tree (the same restore pattern
     the coherence gate uses): build {"params": <abstract tree>} from
     nnx.eval_shape and pass it as the restore target so Orbax knows the dtype /
     shape / structure of every leaf.
  4. Assert every restored leaf is allclose to the original leaf.
  5. Rebuild a live model from the restored leaves and assert the forward logits
     are IDENTICAL to the original model's logits on the same token ids.

If this passes, the save/restore contract is sound: a checkpoint written by the
converter can be loaded back into our model with bit-faithful values and an
identical forward. (It does NOT prove the HF->ours VALUE mapping is correct —
that is the coherence gate's job, on real weights.)

Prints "ORBAX ROUNDTRIP PASSED" on success; raises with a clear message
otherwise. This MUST pass on CPU.
"""

import os
import sys
import tempfile

# Make `import jax_nemotron...` work whether run from repo root or elsewhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import orbax.checkpoint as ocp

from jax_nemotron.config import NemotronHConfig
from jax_nemotron.nemotron_h import NemotronHModel


def _fail(msg: str):
    raise AssertionError(f"ORBAX ROUNDTRIP FAILED: {msg}")


# -----------------------------------------------------------------------------
# Flat-path helpers (slash paths match the converter / shape gate conventions)
# -----------------------------------------------------------------------------


def _slash(key_tuple) -> str:
    """Join an nnx flat_state tuple key (str / int segments) into a slash path,
    e.g. ('layers', 3, 'mixer', 'in_proj', 'kernel') -> 'layers/3/mixer/...'.

    This matches the slash paths produced by jax.tree_util keypaths in the
    converter's _flatten_state_paths and in build_target_tree."""
    return "/".join(str(k) for k in key_tuple)


def _flatten_to_slash_dict(state) -> dict:
    """Return {slash_path: leaf_array} from a CONCRETE nnx state.

    Reads each leaf via Variable.get_value() (the non-deprecated accessor the
    model code itself uses); flat_state() yields the Variable objects."""
    out = {}
    for key_tuple, var in state.flat_state():
        getter = getattr(var, "get_value", None)
        arr = getter() if callable(getter) else getattr(var, "value", var)
        out[_slash(key_tuple)] = np.asarray(arr)
    return out


def _insert(tree: dict, slash_path: str, value) -> None:
    """Insert value into nested dict at slash_path (mirrors the converter's
    _insert exactly, so the on-disk tree shape is identical)."""
    parts = slash_path.split("/")
    node = tree
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def _build_concrete_params_tree(state) -> dict:
    """Build the SAME nested dict the converter writes: a plain nested dict of
    arrays keyed by slash-path segments (integers kept as string segments)."""
    params: dict = {}
    for slash_path, arr in _flatten_to_slash_dict(state).items():
        _insert(params, slash_path, arr)
    return params


# -----------------------------------------------------------------------------
# Abstract target tree via eval_shape (the converter's build_target_tree side)
# -----------------------------------------------------------------------------


def _build_abstract_state(cfg):
    """Return (graphdef, abstract_state) from nnx.eval_shape — no allocation.
    The abstract_state's leaves are nnx Variables carrying ShapeDtypeStruct."""

    def init_fn():
        return NemotronHModel(rngs=nnx.Rngs(0), config=cfg)

    abstract_model = nnx.eval_shape(init_fn)
    graphdef, abstract_state = nnx.split(abstract_model)
    return graphdef, abstract_state


def _abstract_params_target(abstract_state) -> dict:
    """Build the {"params": <nested abstract tree>} RESTORE TARGET that Orbax's
    StandardCheckpointer.restore needs: nested dict whose leaves are
    jax.ShapeDtypeStruct, matching exactly the structure the converter saved.

    This is the same {"params": <abstract eval_shape tree>} restore target the
    coherence gate uses."""
    params: dict = {}
    for key_tuple, var in abstract_state.flat_state():
        val = getattr(var, "value", var)
        sds = jax.ShapeDtypeStruct(tuple(val.shape), val.dtype)
        _insert(params, _slash(key_tuple), sds)
    return params


def _scatter_restored_into_state(abstract_state, restored_params: dict):
    """Set each abstract leaf's value from the restored nested dict, returning a
    concrete nnx state ready for nnx.merge. Looks each leaf up by its slash
    path so the structure is checked implicitly (a missing path raises)."""
    flat = dict(abstract_state.flat_state())
    for key_tuple, var in flat.items():
        slash = _slash(key_tuple)
        node = restored_params
        for seg in slash.split("/"):
            node = node[seg]
        var.value = jnp.asarray(node)
    return nnx.State.from_flat_path(flat)


def main():
    cfg = NemotronHConfig.from_preset("tiny")
    cfg.validate()

    # ----- 1. build tiny backbone + extract concrete params -----
    model = NemotronHModel(rngs=nnx.Rngs(0), config=cfg)
    graphdef, state = nnx.split(model)
    orig_leaves = _flatten_to_slash_dict(state)
    params_tree = _build_concrete_params_tree(state)
    print(f"[roundtrip] tiny backbone: {len(orig_leaves)} leaves extracted")

    # Reference forward (random weights) on dummy ids; seqlen divisible by chunk.
    batch, seqlen = 2, 8
    if seqlen % cfg.chunk_size != 0:
        _fail(f"seqlen {seqlen} not divisible by chunk_size {cfg.chunk_size}")
    token_ids = jax.random.randint(
        jax.random.PRNGKey(0), (batch, seqlen), 0, cfg.vocab_size
    )
    orig_logits = model(token_ids)
    if not bool(jnp.all(jnp.isfinite(orig_logits))):
        _fail("original logits non-finite")
    print(f"[roundtrip] original forward logits {tuple(orig_logits.shape)}")

    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.abspath(os.path.join(tmp, "ckpt"))

        # ----- 2. SAVE exactly as the converter does -----
        ckpter = ocp.StandardCheckpointer()
        ckpter.save(dest, {"params": params_tree})
        ckpter.wait_until_finished()
        print(f"[roundtrip] saved Orbax checkpoint to {dest!r}")

        # ----- 3. RESTORE into the eval_shape abstract target tree -----
        abs_graphdef, abstract_state = _build_abstract_state(cfg)
        restore_target = {"params": _abstract_params_target(abstract_state)}
        # Fresh checkpointer to prove no in-process state is reused.
        restored = ocp.StandardCheckpointer().restore(dest, restore_target)
        restored_params = restored["params"]
        print("[roundtrip] restored checkpoint into eval_shape target tree")

    # ----- 4. assert every restored leaf == original leaf -----
    restored_flat: dict = {}

    def _walk(prefix, node):
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(prefix + [k] if prefix else [k], v)
        else:
            restored_flat["/".join(prefix)] = np.asarray(node)

    _walk([], restored_params)

    if set(restored_flat) != set(orig_leaves):
        only_orig = sorted(set(orig_leaves) - set(restored_flat))[:8]
        only_rest = sorted(set(restored_flat) - set(orig_leaves))[:8]
        _fail(
            f"leaf path sets differ. missing_from_restored={only_orig} "
            f"extra_in_restored={only_rest}"
        )
    for path, orig in orig_leaves.items():
        rest = restored_flat[path]
        if rest.shape != orig.shape:
            _fail(f"{path}: restored shape {rest.shape} != original {orig.shape}")
        if not np.allclose(rest, orig, rtol=1e-6, atol=1e-6):
            _fail(f"{path}: restored values not allclose to original")
    print(f"[roundtrip] all {len(orig_leaves)} leaves allclose to originals")

    # ----- 5. rebuild model from restored leaves; identical forward -----
    abs_graphdef2, abstract_state2 = _build_abstract_state(cfg)
    restored_state = _scatter_restored_into_state(abstract_state2, restored_params)
    restored_model = nnx.merge(abs_graphdef2, restored_state)

    restored_logits = restored_model(token_ids)
    if not bool(jnp.all(jnp.isfinite(restored_logits))):
        _fail("restored-model logits non-finite")
    if tuple(restored_logits.shape) != tuple(orig_logits.shape):
        _fail(
            f"restored logits shape {tuple(restored_logits.shape)} != "
            f"original {tuple(orig_logits.shape)}"
        )
    if not np.array_equal(np.asarray(restored_logits), np.asarray(orig_logits)):
        max_abs = float(np.max(np.abs(np.asarray(restored_logits) - np.asarray(orig_logits))))
        _fail(
            "restored-model logits are NOT bit-identical to the original "
            f"(max abs diff {max_abs:g}); save/restore is lossy"
        )
    print(
        f"[roundtrip] restored-model forward logits {tuple(restored_logits.shape)} "
        f"are bit-identical to the original"
    )

    print("ORBAX ROUNDTRIP PASSED")


if __name__ == "__main__":
    main()
