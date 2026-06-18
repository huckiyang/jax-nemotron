"""
Convert the HuggingFace Nemotron-3-Nano-Omni safetensors checkpoint into an
Orbax checkpoint loadable by this JAX/Flax(NNX) stack — LANGUAGE-MODEL backbone
(milestone 1).

------------------------------------------------------------------------------
WHAT THIS DOES (target-driven, per the hf-to-orbax-conversion playbook)
------------------------------------------------------------------------------
(a) Build the AUTHORITATIVE target param tree from OUR Flax model via
    ``jax.eval_shape`` (nnx.eval_shape) at the chosen preset — no allocation, so
    a 30B abstract tree is free. The flattened tree {slash_path -> shape} is the
    single source of truth for which leaves must be produced and at what shape.
(b) Open the safetensors shards LAZILY with ``safe_open(framework="numpy")``,
    indexed by ``model.safetensors.index.json``'s ``weight_map`` (tensor -> shard).
    Only the tensors needed for a given leaf are read; we never materialize the
    whole f32 tree.
(c) For each target leaf: look up the HF source key(s) via the NAME MAP that the
    model itself exports (``nemotron_h.hf_name_map``), apply the per-leaf
    reshape/transpose derived from the module layout (NOT guessed — see
    ``apply_transform`` docstring for the derivation of each one), ASSERT
    ``arr.shape == target_shape``, and cast to ``ml_dtypes.bfloat16`` (a 30B f32
    tree OOMs; bf16 is ~60GB).
(d) RMSNorm scales are copied RAW (Nemotron-H is PLAIN ``weight * normed``, NO
    ``1 + weight`` — verified in modeling_nemotron_h.py line 720). Per-layer
    scalars A_log / D / dt_bias / e_score_correction_bias are copied RAW too.
(e) RoPE q/k permutation: NOT applied. HF Nemotron-H attention uses half-split
    ``rotate_half`` (modeling_nemotron_h.py lines 855-883) and our attention
    (nemotron_h.py ``_rotate_half`` / ``_apply_rope``) uses the SAME half-split
    convention. A permutation is only needed when the two sides use DIFFERENT
    RoPE styles (half-split vs interleaved); they match, so q/k are copied with a
    plain transpose only.
(f) Write with ``orbax.checkpoint.StandardCheckpointer().save(out, params)`` then
    ``wait_until_finished()``, STRAIGHT to ``--out`` (which may be a ``gs://``
    path — we never ``os.path.abspath`` a gs:// path, which would mangle it to
    ``/cwd/gs:/...``). A local save + copy yields an "incomplete checkpoint".

A manifest of leaves written (and any unmapped/unconsumed) is printed at the end.

------------------------------------------------------------------------------
WHERE TO RUN THIS (NOT in the dev sandbox)
------------------------------------------------------------------------------
The 30B bf16 param tree is ~60GB in host RAM (and the converter holds it before
writing). It MUST run on a big-memory host where the *.safetensors shards live —
e.g. a large-RAM cloud TPU/GPU VM, NOT in a memory-limited
laptop/CI sandbox. In this public repo we only validate the PURE transform
functions and the name-map bijection on synthetic arrays (tests/) and on the
checkpoint index (no weights). The decisive correctness proof is the COHERENCE
gate: load the Orbax checkpoint in the real runtime on TPU and check the model
emits sensible text. Shapes can be right while values are scrambled.

------------------------------------------------------------------------------
CLI
------------------------------------------------------------------------------
    python scripts/convert_hf_to_orbax.py \
        --ckpt-dir /path/to/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
        --out      gs://bucket/nemotron-omni-30b-orbax   (or a local dir) \
        --preset   omni_30b \
        --dtype    bf16

Milestone-2 hooks (vision_model.*, sound_encoder.*, mlp1.*, sound_projection.*)
are clearly marked below; this script intentionally converts ONLY the LLM
backbone (``language_model.*``) for milestone 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable

import numpy as np
import ml_dtypes

# Make `import jax_nemotron...` work whether run from repo root or the scripts dir.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

BF16 = ml_dtypes.bfloat16


# =============================================================================
# Path / store helpers (gs:// safe)
# =============================================================================


def _is_remote(path: str) -> bool:
    """True for object-store URIs (gs://, s3://, ...). These must NOT be
    abspath'd — os.path.abspath('gs://b/k') mangles to '/cwd/gs:/b/k'."""
    return "://" in path


def _normalize_out(path: str) -> str:
    """Return the destination as Orbax/tensorstore expects it. Local dirs get an
    absolute path (Orbax requires absolute local paths); remote URIs pass through
    UNCHANGED."""
    if _is_remote(path):
        return path
    return os.path.abspath(path)


# =============================================================================
# Pure per-leaf transforms (UNIT-TESTED in tests/test_converter_units.py)
# =============================================================================
#
# Each transform takes the HF source array(s) (numpy, any dtype) and returns a
# numpy array whose shape MUST equal the target leaf shape. The casting to bf16
# happens AFTER the shape assert, in the converter loop, so these stay pure
# shape/layout ops and are trivially testable on synthetic arrays.
#
# Derivation of each transform (from the module layout, not guesses):
#   raw    : 1-D params (RMSNorm scale, dt_bias, A_log, D, conv bias,
#            e_score_correction_bias) and the (vocab, hidden) embedding/lm-head
#            tables that our nnx.Embed stores in the SAME (rows, cols) layout as
#            HF. Copied unchanged.
#   T      : PyTorch nn.Linear weight is (out_features, in_features); our
#            nnx.Linear kernel is (in_features, out_features). So transpose.
#            Applies to in_proj/out_proj (Mamba), q/k/v/o_proj (attn), gate, and
#            lm_head (HF lm_head.weight is (vocab, hidden); our lm_head.kernel is
#            (hidden, vocab) => transpose).
#   conv   : PyTorch Conv1d weight is (out_ch, in_ch/groups, kernel). For the
#            depthwise Mamba conv in_ch/groups == 1, giving (conv_dim, 1, k).
#            nnx.Conv kernel is (kernel, in_ch/groups, out_ch) == (k, 1, conv_dim)
#            (verified empirically against nnx.Conv). So transpose axes (0,1,2)->
#            (2,1,0).
#   stackT : MoE experts are stored per-expert in HF (experts.{i}.up_proj.weight,
#            each (moe_inter, hidden)) but PRE-STACKED in our tree
#            (routed_W1 : (E, hidden, moe_inter)). So transpose each source
#            (out,in)->(in,out) and stack along a new leading axis 0. Also used
#            for the single shared expert (N==1) so shared_W1/2 get a leading
#            axis of length 1, matching our (n_shared_experts, ...) storage.


def transform_raw(arrays: list[np.ndarray]) -> np.ndarray:
    """Copy a single source unchanged."""
    assert len(arrays) == 1, f"raw expects exactly one source, got {len(arrays)}"
    return arrays[0]


def transform_T(arrays: list[np.ndarray]) -> np.ndarray:
    """Transpose a single 2-D linear weight (out,in) -> (in,out)."""
    assert len(arrays) == 1, f"T expects exactly one source, got {len(arrays)}"
    a = arrays[0]
    assert a.ndim == 2, f"T expects a 2-D array, got shape {a.shape}"
    return np.ascontiguousarray(a.T)


def transform_conv(arrays: list[np.ndarray]) -> np.ndarray:
    """PyTorch Conv1d (out_ch, in_ch/groups, k) -> nnx.Conv (k, in_ch/groups, out_ch)."""
    assert len(arrays) == 1, f"conv expects exactly one source, got {len(arrays)}"
    a = arrays[0]
    assert a.ndim == 3, f"conv expects a 3-D array, got shape {a.shape}"
    return np.ascontiguousarray(np.transpose(a, (2, 1, 0)))


def transform_stackT(arrays: list[np.ndarray]) -> np.ndarray:
    """Transpose each 2-D source (out,in)->(in,out) and stack on a new axis 0,
    giving (N, in, out). N may be 1 (shared expert)."""
    assert len(arrays) >= 1, "stackT expects at least one source"
    s0 = arrays[0].shape
    for a in arrays:
        assert a.ndim == 2, f"stackT expects 2-D sources, got shape {a.shape}"
        assert a.shape == s0, f"stackT sources must share a shape; {a.shape} != {s0}"
    stacked = np.stack([a.T for a in arrays], axis=0)
    return np.ascontiguousarray(stacked)


_TRANSFORMS: dict[str, Callable[[list[np.ndarray]], np.ndarray]] = {
    "raw": transform_raw,
    "T": transform_T,
    "conv": transform_conv,
    "stackT": transform_stackT,
}


def apply_transform(transform: str, arrays: list[np.ndarray]) -> np.ndarray:
    """Dispatch to the named pure transform. Raises on an unknown name."""
    fn = _TRANSFORMS.get(transform)
    if fn is None:
        raise ValueError(f"unknown transform {transform!r}")
    return fn(arrays)


# =============================================================================
# Target tree (authoritative shapes from OUR model via eval_shape)
# =============================================================================


def _flatten_state_paths(state) -> dict:
    """Flatten an nnx state pytree into {slash_path: leaf} using JAX path APIs.
    The slash-joined paths match the keys produced by hf_name_map()."""
    import jax

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


def build_target_tree(preset: str):
    """Return (config, {slash_path -> tuple(shape)}) for the chosen preset,
    using nnx.eval_shape so NO real arrays are allocated."""
    import jax  # noqa: F401  (imported for tree_util in _flatten_state_paths)
    from flax import nnx

    from jax_nemotron.config import NemotronHConfig
    from jax_nemotron.nemotron_h import NemotronHModel

    cfg = NemotronHConfig.from_preset(preset)
    cfg.validate()

    def init_fn():
        return NemotronHModel(rngs=nnx.Rngs(0), config=cfg)

    abstract_model = nnx.eval_shape(init_fn)
    _, abstract_state = nnx.split(abstract_model)
    leaves = _flatten_state_paths(abstract_state)
    targets = {}
    for path, leaf in leaves.items():
        val = getattr(leaf, "value", leaf)
        targets[path] = tuple(val.shape)
    return cfg, targets


# =============================================================================
# Lazy safetensors reader (per-shard, numpy framework)
# =============================================================================


class ShardedSafetensors:
    """Lazily reads tensors from a sharded HF safetensors checkpoint using the
    index ``weight_map`` (tensor name -> shard file). One ``safe_open`` handle is
    cached per shard so each shard file is memory-mapped at most once; tensors
    are read on demand. Single-file checkpoints (model.safetensors, no index) are
    also supported."""

    def __init__(self, ckpt_dir: str):
        from safetensors import safe_open  # local import: optional dep

        self._safe_open = safe_open
        self.ckpt_dir = ckpt_dir
        index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
        single_path = os.path.join(ckpt_dir, "model.safetensors")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            self.weight_map = index["weight_map"]  # tensor -> shard filename
        elif os.path.exists(single_path):
            # Single-file checkpoint: every key maps to the one file.
            with self._safe_open(single_path, framework="numpy") as f:
                keys = list(f.keys())
            self.weight_map = {k: "model.safetensors" for k in keys}
        else:
            raise FileNotFoundError(
                f"No model.safetensors.index.json or model.safetensors in {ckpt_dir!r}"
            )
        self._handles: dict[str, object] = {}

    def keys(self):
        return self.weight_map.keys()

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def _handle(self, shard: str):
        h = self._handles.get(shard)
        if h is None:
            h = self._safe_open(os.path.join(self.ckpt_dir, shard), framework="numpy")
            self._handles[shard] = h
        return h

    def get(self, name: str) -> np.ndarray:
        """Read one tensor as a numpy array. safetensors returns bf16 as a
        ml_dtypes.bfloat16-backed array under framework='numpy'."""
        if name not in self.weight_map:
            raise KeyError(f"tensor {name!r} not found in checkpoint index")
        shard = self.weight_map[name]
        arr = self._handle(shard).get_tensor(name)
        return np.asarray(arr)


# =============================================================================
# Insert a leaf into a nested dict at a slash path
# =============================================================================


def _insert(tree: dict, slash_path: str, value) -> None:
    """Insert ``value`` into nested dict ``tree`` at ``slash_path`` (e.g.
    'layers/3/mixer/in_proj/kernel'). Integer segments stay strings here; the
    nesting mirrors the nnx state-dict the loader reconstructs."""
    parts = slash_path.split("/")
    node = tree
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


# =============================================================================
# Converter core
# =============================================================================


def convert(
    ckpt_dir: str,
    out: str,
    preset: str,
    dtype: str,
    dry_run: bool = False,
) -> dict:
    """Run the full target-driven conversion of the LLM backbone.

    Returns a manifest dict with counts and any unmapped/unconsumed lists.
    When ``dry_run`` is True, validates name-map membership + the bijection over
    language_model.* against the index metadata WITHOUT reading any tensor data
    (the index-only checkpoint has no shards), and does NOT write Orbax."""
    if dtype != "bf16":
        raise ValueError(
            f"--dtype {dtype!r} not supported; only 'bf16' (a 30B f32 tree OOMs)."
        )
    cast_dtype = BF16

    from jax_nemotron.nemotron_h import hf_name_map, HF_PREFIX

    print(f"[convert] preset={preset!r}  ckpt_dir={ckpt_dir!r}  out={out!r}")

    # (a) authoritative target tree from OUR model (no allocation).
    cfg, targets = build_target_tree(preset)
    print(f"[convert] target tree: {len(targets)} leaves (eval_shape, no alloc)")

    # name-map contract exported by the model.
    name_map = hf_name_map(cfg)

    # (b) lazy sharded reader.
    store = ShardedSafetensors(ckpt_dir)
    hf_all = set(store.keys())
    hf_llm = {k for k in hf_all if k.startswith(HF_PREFIX)}
    print(f"[convert] checkpoint tensors: {len(hf_all)} total, "
          f"{len(hf_llm)} under {HF_PREFIX!r}")

    # ----- milestone-2 HOOKS (NOT converted here) -----
    # vision_model.* / sound_encoder.* / mlp1.* / sound_projection.* are the
    # multimodal encoders + projectors. They are intentionally LEFT for
    # milestone 2; a future converter pass will build their target trees from
    # the vision/sound Flax modules and map them analogously. We only report
    # their presence so the manifest is honest about what was skipped.
    skipped_ns: dict[str, int] = {}
    for k in hf_all - hf_llm:
        ns = k.split(".")[0]
        skipped_ns[ns] = skipped_ns.get(ns, 0) + 1
    if skipped_ns:
        print(f"[convert] SKIPPED (milestone 2, multimodal): {skipped_ns}")

    # (c)-(e) build the param tree leaf-by-leaf.
    params: dict = {}
    written = []  # slash paths written
    unmapped_targets = []  # target leaves with no name-map entry
    consumed_hf: dict[str, str] = {}  # hf full key -> target that claimed it
    total_bytes = 0

    for tpath, tshape in targets.items():
        entry = name_map.get(tpath)
        if entry is None:
            # nnx may emit a trailing variable-kind segment on some leaves; the
            # name map keys to the un-trailed path. Try the trimmed form.
            trimmed = "/".join(tpath.split("/")[:-1])
            entry = name_map.get(trimmed)
        if entry is None:
            unmapped_targets.append(tpath)
            continue

        hf = entry["hf"]
        transform = entry["transform"]
        rel_names = [hf] if isinstance(hf, str) else list(hf)
        full_names = [HF_PREFIX + n for n in rel_names]

        # Read source array(s) lazily; record consumption / detect double-claims.
        arrays = []
        for fn in full_names:
            if fn not in store:
                raise KeyError(
                    f"target {tpath!r} needs HF tensor {fn!r}, absent from "
                    f"checkpoint index (shape expected from config: {tshape})"
                )
            if fn in consumed_hf:
                raise ValueError(
                    f"HF tensor {fn!r} double-claimed by {consumed_hf[fn]!r} and "
                    f"{tpath!r}"
                )
            consumed_hf[fn] = tpath
            # Dry-run validates name-map membership against the index only; it must
            # NOT read tensor data (the index-only checkpoint has no shards on disk).
            if not dry_run:
                arrays.append(store.get(fn))

        if not dry_run:
            # apply the documented reshape/transpose, then ASSERT shape, then cast.
            arr = apply_transform(transform, arrays)
            assert tuple(arr.shape) == tuple(tshape), (
                f"{tpath}: converted shape {tuple(arr.shape)} != target {tuple(tshape)} "
                f"(transform={transform}, sources={full_names})"
            )
            arr = arr.astype(cast_dtype)
            total_bytes += arr.nbytes
            _insert(params, tpath, arr)
        written.append(tpath)

    # Bijection bookkeeping over language_model.* (mirrors test_name_coverage).
    unconsumed_hf = sorted(hf_llm - set(consumed_hf.keys()))

    print(f"[convert] built {len(written)} leaves, ~{total_bytes / 1e9:.2f} GB bf16 "
          f"in RAM")
    if unmapped_targets:
        print(f"[convert] WARNING: {len(unmapped_targets)} target leaves UNMAPPED: "
              f"{unmapped_targets[:8]}")
    if unconsumed_hf:
        print(f"[convert] WARNING: {len(unconsumed_hf)} language_model.* HF tensors "
              f"UNCONSUMED: {unconsumed_hf[:8]}")

    # (f) write Orbax STRAIGHT to the destination (gs:// safe).
    if not dry_run:
        import orbax.checkpoint as ocp

        dest = _normalize_out(out)
        print(f"[convert] writing Orbax checkpoint to {dest!r} ...")
        ckpter = ocp.StandardCheckpointer()
        # Wrap under "params" so the runtime loads model.init's {"params": ...}
        # shape. (Loaders auto-classify the nested tree; see playbook.)
        ckpter.save(dest, {"params": params})
        ckpter.wait_until_finished()
        print(f"[convert] WROTE checkpoint to {dest!r}")
    else:
        print("[convert] DRY RUN: skipped Orbax write")

    manifest = {
        "preset": preset,
        "dtype": dtype,
        "n_target_leaves": len(targets),
        "n_written": len(written),
        "approx_bf16_gb": round(total_bytes / 1e9, 3),
        "n_hf_llm_tensors": len(hf_llm),
        "n_consumed_hf": len(consumed_hf),
        "unmapped_targets": unmapped_targets,
        "unconsumed_hf": unconsumed_hf,
        "skipped_multimodal_namespaces": skipped_ns,
    }
    return manifest


# =============================================================================
# CLI
# =============================================================================


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert HF Nemotron-3-Nano-Omni safetensors -> Orbax (LLM backbone)."
    )
    p.add_argument("--ckpt-dir", required=True,
                   help="HF checkpoint dir containing model.safetensors[.index.json].")
    p.add_argument("--out", required=True,
                   help="Destination Orbax dir (local path or gs://bucket/path).")
    p.add_argument("--preset", default="omni_30b",
                   help="Model preset (omni_30b | tiny). Default omni_30b.")
    p.add_argument("--dtype", default="bf16",
                   help="Cast dtype. Only 'bf16' supported (f32 30B tree OOMs).")
    p.add_argument("--dry-run", action="store_true",
                   help="Read/transform/assert every leaf but do NOT write Orbax.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    manifest = convert(
        ckpt_dir=args.ckpt_dir,
        out=args.out,
        preset=args.preset,
        dtype=args.dtype,
        dry_run=args.dry_run,
    )
    print("\n========== MANIFEST ==========")
    print(json.dumps(manifest, indent=2))
    # Non-zero exit if the contract is not a full bijection over the LLM backbone.
    if manifest["unmapped_targets"] or manifest["unconsumed_hf"]:
        print("[convert] FAILED: contract is not a full bijection over "
              "language_model.* (see WARNINGs above)")
        return 1
    print("[convert] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
