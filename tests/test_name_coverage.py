"""
Name/shape MAPPING COMPLETENESS GATE for the Nemotron-H LLM backbone.

Run from the repo root (or anywhere — sys.path is fixed up below):
    .venv/bin/python tests/test_name_coverage.py

What this gate proves WITHOUT any weights (shapes only, fits in RAM):

  1. Build the REAL "omni_30b" LLM backbone and ``jax.eval_shape`` its init to get
     the authoritative target param tree {path -> shape}. No allocation happens —
     eval_shape returns ShapeDtypeStructs only, so a 30B tree is free.

  2. Load model.safetensors.index.json's weight_map and take the
     ``language_model.*`` subset (the LLM backbone tensors; vision/sound are a
     later milestone and reported as uncovered TODO, not a failure).

  3. Use the target<->HF NAME MAP from nemotron_h.hf_name_map() (the converter's
     documented contract) and assert a BIJECTION over language_model.*:
       - every target leaf maps to exactly one or more EXISTING HF keys,
       - every HF language_model.* key is consumed by EXACTLY ONE target leaf
         (no HF tensor left on the floor, no HF tensor double-claimed).

  4. Shape check WITHOUT data: derive each HF tensor's expected shape from config
     dims (per the phase-1 inventory), apply the converter's planned transform
     (raw / transpose / conv-reshape / stack), and assert it yields the target
     leaf shape. Where a stacked source is used, assert the per-source product is
     consistent and the stacked product matches.

Prints "MAPPING GATE: <matched>/<targets> matched, <unconsumed> HF keys unconsumed".
passed iff a FULL bijection holds over language_model.*.

This is a STRUCTURE/NAME/SHAPE gate. It does NOT prove value correctness — that
requires the coherence gate (real forward on real weights) on the big-mem/TPU
host where the safetensors live.
"""

import json
import os
import sys

# Make `import jax_nemotron...` work whether run from repo root or elsewhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import jax
from flax import nnx

from jax_nemotron.config import (
    NemotronHConfig,
    MIXER_MAMBA,
    MIXER_ATTENTION,
    MIXER_MOE,
)
from jax_nemotron.nemotron_h import (
    NemotronHModel,
    hf_name_map,
    HF_PREFIX,
)

# The checkpoint index that lists every safetensors tensor name (no weights).
# Resolved from $NEMOTRON_CKPT_DIR if set, else a sibling clone of the HF repo
# next to this repo. Public users without the 62GB checkpoint -> graceful SKIP.
_CKPT_DIR = os.environ.get(
    "NEMOTRON_CKPT_DIR",
    os.path.join(
        os.path.dirname(_REPO_ROOT), "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    ),
)
_INDEX_JSON = os.path.join(_CKPT_DIR, "model.safetensors.index.json")


def _fail(msg: str):
    raise AssertionError(f"MAPPING GATE FAILED: {msg}")


# =============================================================================
# Flatten the eval_shape target tree to {slash_path -> ShapeDtypeStruct}
# =============================================================================


def _flatten_state_paths(state) -> dict:
    """
    Flatten an nnx state pytree into {slash_path: leaf} using JAX path APIs.

    nnx.eval_shape returns (graphdef, state); the state is a pytree of
    nnx.VariableState leaves whose `.value` carries the ShapeDtypeStruct. We use
    jax.tree_util keypaths to build stable slash-joined names that match the keys
    produced by hf_name_map() (e.g. "layers/3/mixer/in_proj/kernel").
    """
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


def _target_shape(leaf) -> tuple:
    val = getattr(leaf, "value", leaf)
    return tuple(val.shape)


# =============================================================================
# Expected HF tensor shapes derived from config (no weights loaded)
# =============================================================================


def _expected_hf_shapes(cfg: NemotronHConfig) -> dict:
    """
    Derive the expected PyTorch/safetensors shape for every language_model.*
    tensor purely from config dims (phase-1 inventory conventions). Returned keys
    are RELATIVE to HF_PREFIX (same convention as hf_name_map values).

    PyTorch conventions encoded here:
      * nn.Linear weight is (out_features, in_features).
      * Conv1d weight is (out_channels, in_channels/groups, kernel). Depthwise
        here => in_channels/groups == 1, so (conv_dim, 1, conv_kernel).
      * Embedding / lm_head are (rows, cols) tables.
      * RMSNorm / dt_bias / A_log / D / e_score_correction_bias are 1-D.
    """
    H = cfg.hidden_size
    exp: dict = {}

    # Top level.
    exp["backbone.embeddings.weight"] = (cfg.vocab_size, H)
    exp["backbone.norm_f.weight"] = (H,)
    exp["lm_head.weight"] = (cfg.vocab_size, H)

    d_inner = cfg.mamba_intermediate_size
    conv_dim = cfg.mamba_conv_dim
    in_proj_dim = cfg.mamba_in_proj_dim

    q_dim = cfg.attention_q_dim
    kv_dim = cfg.attention_kv_dim

    for i, mt in enumerate(cfg.parse_pattern()):
        hp = f"backbone.layers.{i}"
        exp[f"{hp}.norm.weight"] = (H,)

        if mt == MIXER_MAMBA:
            hm = f"{hp}.mixer"
            # in_proj: Linear(H -> in_proj_dim) => (in_proj_dim, H)
            exp[f"{hm}.in_proj.weight"] = (in_proj_dim, H)
            # conv1d: depthwise => (conv_dim, 1, conv_kernel)
            exp[f"{hm}.conv1d.weight"] = (conv_dim, 1, cfg.conv_kernel)
            exp[f"{hm}.conv1d.bias"] = (conv_dim,)
            exp[f"{hm}.dt_bias"] = (cfg.mamba_num_heads,)
            exp[f"{hm}.A_log"] = (cfg.mamba_num_heads,)
            exp[f"{hm}.D"] = (cfg.mamba_num_heads,)
            # gated RMSNorm over d_inner
            exp[f"{hm}.norm.weight"] = (d_inner,)
            # out_proj: Linear(d_inner -> H) => (H, d_inner)
            exp[f"{hm}.out_proj.weight"] = (H, d_inner)

        elif mt == MIXER_ATTENTION:
            hm = f"{hp}.mixer"
            exp[f"{hm}.q_proj.weight"] = (q_dim, H)
            exp[f"{hm}.k_proj.weight"] = (kv_dim, H)
            exp[f"{hm}.v_proj.weight"] = (kv_dim, H)
            exp[f"{hm}.o_proj.weight"] = (H, q_dim)

        elif mt == MIXER_MOE:
            hm = f"{hp}.mixer"
            # gate: Linear(H -> n_routed_experts) => (n_routed_experts, H)
            exp[f"{hm}.gate.weight"] = (cfg.n_routed_experts, H)
            exp[f"{hm}.gate.e_score_correction_bias"] = (cfg.n_routed_experts,)
            for j in range(cfg.n_routed_experts):
                # up_proj: Linear(H -> moe_inter) => (moe_inter, H)
                exp[f"{hm}.experts.{j}.up_proj.weight"] = (cfg.moe_intermediate_size, H)
                # down_proj: Linear(moe_inter -> H) => (H, moe_inter)
                exp[f"{hm}.experts.{j}.down_proj.weight"] = (H, cfg.moe_intermediate_size)
            if cfg.n_shared_experts > 0:
                si = cfg.moe_shared_expert_intermediate_size
                exp[f"{hm}.shared_experts.up_proj.weight"] = (si, H)
                exp[f"{hm}.shared_experts.down_proj.weight"] = (H, si)

    return exp


# =============================================================================
# Transform-aware shape check: does converting hf_shape(s) yield target shape?
# =============================================================================


def _transformed_shape(transform: str, hf_shapes: list) -> tuple:
    """
    Predict the JAX target shape after applying the documented converter
    transform to the given HF source shape(s). Mirrors the converter's planned
    reshape/transpose logic exactly so the gate proves the contract.

    transform:
      raw    : single source, copied unchanged.
      T      : single 2-D source, transposed (out,in) -> (in,out).
      conv   : single PyTorch Conv1d (out_ch, in_ch/groups, k) -> JAX
               nnx.Conv kernel (k, in_ch/groups, out_ch).
      stackT : N sources each 2-D (out,in); transpose each to (in,out) and stack
               along a new leading axis => (N, in, out).
    """
    if transform == "raw":
        assert len(hf_shapes) == 1, "raw expects exactly one source"
        return tuple(hf_shapes[0])

    if transform == "T":
        assert len(hf_shapes) == 1, "T expects exactly one source"
        s = hf_shapes[0]
        assert len(s) == 2, f"T expects 2-D source, got {s}"
        return (s[1], s[0])

    if transform == "conv":
        assert len(hf_shapes) == 1, "conv expects exactly one source"
        s = hf_shapes[0]
        assert len(s) == 3, f"conv expects 3-D source, got {s}"
        out_ch, in_per_group, k = s
        return (k, in_per_group, out_ch)

    if transform == "stackT":
        n = len(hf_shapes)
        # each (out, in) -> (in, out); stack on axis 0 => (n, in, out)
        s0 = hf_shapes[0]
        assert len(s0) == 2, f"stackT expects 2-D sources, got {s0}"
        for s in hf_shapes:
            assert tuple(s) == tuple(s0), (
                f"stackT sources must be identical shape; {s} != {s0}"
            )
        return (n, s0[1], s0[0])

    raise ValueError(f"unknown transform {transform!r}")


def main():
    # ----- 1. authoritative target tree from the REAL model -----
    cfg = NemotronHConfig.from_preset("omni_30b")
    cfg.validate()
    print(f"[gate] omni_30b config: {cfg.num_hidden_layers} layers, hidden="
          f"{cfg.hidden_size}, experts={cfg.n_routed_experts}")

    def init_fn():
        return NemotronHModel(rngs=nnx.Rngs(0), config=cfg)

    abstract_model = nnx.eval_shape(init_fn)
    _, abstract_state = nnx.split(abstract_model)
    targets = _flatten_state_paths(abstract_state)
    n_targets = len(targets)
    print(f"[gate] eval_shape produced {n_targets} target leaves (no allocation)")
    if n_targets == 0:
        _fail("eval_shape produced an empty target tree")

    # ----- 2. HF language_model.* keys from the index -----
    if not os.path.exists(_INDEX_JSON):
        print(f"[gate] SKIP: checkpoint index not found at {_INDEX_JSON}")
        print("[gate] set $NEMOTRON_CKPT_DIR to the HF checkpoint dir to run this "
              "gate (needs only model.safetensors.index.json, no weights).")
        return
    with open(_INDEX_JSON) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    hf_all = set(weight_map.keys())
    hf_llm = {k for k in hf_all if k.startswith(HF_PREFIX)}
    print(f"[gate] index has {len(hf_all)} tensors; "
          f"{len(hf_llm)} under {HF_PREFIX!r}")

    # Report (but do not fail on) the non-LLM namespaces — milestone 2.
    other_ns = {}
    for k in hf_all - hf_llm:
        other_ns.setdefault(k.split(".")[0], 0)
        other_ns[k.split(".")[0]] += 1
    print(f"[gate] non-LLM namespaces (TODO milestone 2, not a fail): {other_ns}")

    # ----- 3. build the name map and assert bijection over language_model.* -----
    name_map = hf_name_map(cfg)  # target_path -> {"hf": str|list, "transform": ...}

    # 3a. every target leaf must have a map entry. (nnx may emit a trailing
    # variable-kind segment for some leaves; normalize by trimming it as the
    # shape gate does.)
    target_to_hf = {}  # target_path -> list[str] of FULL hf keys
    unmapped_targets = []
    for tp in targets:
        entry = name_map.get(tp)
        if entry is None:
            trimmed = "/".join(tp.split("/")[:-1])
            entry = name_map.get(trimmed)
        if entry is None:
            unmapped_targets.append(tp)
            continue
        hf = entry["hf"]
        hf_list = [hf] if isinstance(hf, str) else list(hf)
        full = [HF_PREFIX + name for name in hf_list]
        target_to_hf[tp] = (full, entry["transform"])

    # 3b. also make sure the name_map doesn't reference target paths that don't
    # exist in the tree (stale entries) — over language_model coverage.
    target_pathset = set(targets.keys())
    stale_map_entries = []
    for tp in name_map:
        if tp in target_pathset:
            continue
        # allow the trimmed-form match in reverse: is there a target whose
        # trimmed path equals tp?
        if any("/".join(t.split("/")[:-1]) == tp for t in target_pathset):
            continue
        stale_map_entries.append(tp)

    # 3c. consume HF keys; detect missing sources and double-claims.
    consumed = {}  # hf_key -> target_path that claimed it
    missing_hf_sources = []  # (target, hf_key) the map points at but index lacks
    double_claimed = []  # (hf_key, prev_target, target)
    for tp, (full_list, _transform) in target_to_hf.items():
        for hk in full_list:
            if hk not in hf_llm:
                missing_hf_sources.append((tp, hk))
            if hk in consumed:
                double_claimed.append((hk, consumed[hk], tp))
            else:
                consumed[hk] = tp

    consumed_llm = set(consumed.keys()) & hf_llm
    unconsumed_hf = sorted(hf_llm - consumed_llm)

    matched = len(target_to_hf)

    print("\n[gate] ---- counts ----")
    print(f"[gate] targets (our leaves)          : {n_targets}")
    print(f"[gate] hf_llm_keys (language_model.*) : {len(hf_llm)}")
    print(f"[gate] matched (targets w/ hf source) : {matched}")
    print(f"[gate] unmapped_targets               : {len(unmapped_targets)}")
    print(f"[gate] unconsumed_hf                   : {len(unconsumed_hf)}")
    print(f"[gate] missing_hf_sources             : {len(missing_hf_sources)}")
    print(f"[gate] double_claimed_hf              : {len(double_claimed)}")
    print(f"[gate] stale_map_entries              : {len(stale_map_entries)}")

    if unmapped_targets:
        print(f"[gate] sample unmapped_targets: {sorted(unmapped_targets)[:12]}")
    if missing_hf_sources:
        print(f"[gate] sample missing_hf_sources: {missing_hf_sources[:12]}")
    if double_claimed:
        print(f"[gate] sample double_claimed: {double_claimed[:12]}")
    if unconsumed_hf:
        print(f"[gate] sample unconsumed_hf: {unconsumed_hf[:12]}")
    if stale_map_entries:
        print(f"[gate] sample stale_map_entries: {sorted(stale_map_entries)[:12]}")

    # ----- 4. shape check (transform-aware, no data) -----
    exp_hf = _expected_hf_shapes(cfg)
    shape_mismatches = []  # (target, detail)
    shape_unknown_hf = []  # hf key whose expected shape we couldn't derive
    for tp, (full_list, transform) in target_to_hf.items():
        # gather expected HF shapes (relative names)
        rel_names = [hk[len(HF_PREFIX):] for hk in full_list]
        try:
            hf_shapes = []
            for rn in rel_names:
                if rn not in exp_hf:
                    shape_unknown_hf.append(rn)
                    raise KeyError(rn)
                hf_shapes.append(exp_hf[rn])
        except KeyError:
            continue
        predicted = _transformed_shape(transform, hf_shapes)
        target_shape = _target_shape(targets[tp])
        if predicted != target_shape:
            shape_mismatches.append(
                (tp, f"transform={transform} sources={rel_names} "
                     f"hf_shapes={hf_shapes} -> predicted {predicted} "
                     f"!= target {target_shape}")
            )

    print("\n[gate] ---- shape check ----")
    print(f"[gate] shape_mismatches : {len(shape_mismatches)}")
    print(f"[gate] shape_unknown_hf : {len(shape_unknown_hf)}")
    if shape_mismatches:
        for tp, detail in shape_mismatches[:12]:
            print(f"[gate]   MISMATCH {tp}: {detail}")
    if shape_unknown_hf:
        print(f"[gate] sample shape_unknown_hf: {sorted(set(shape_unknown_hf))[:12]}")

    # ----- verdict: full bijection over language_model.* + shapes consistent ---
    problems = []
    if unmapped_targets:
        problems.append(f"{len(unmapped_targets)} target leaves have no HF source")
    if missing_hf_sources:
        problems.append(f"{len(missing_hf_sources)} mapped HF names absent from index")
    if double_claimed:
        problems.append(f"{len(double_claimed)} HF tensors double-claimed")
    if unconsumed_hf:
        problems.append(f"{len(unconsumed_hf)} language_model.* HF tensors unconsumed")
    if stale_map_entries:
        problems.append(f"{len(stale_map_entries)} stale name_map entries")
    if shape_mismatches:
        problems.append(f"{len(shape_mismatches)} shape mismatches")
    if shape_unknown_hf:
        problems.append(f"{len(set(shape_unknown_hf))} HF shapes underivable")

    print(
        f"\nMAPPING GATE: {matched}/{n_targets} matched, "
        f"{len(unconsumed_hf)} HF keys unconsumed"
    )

    if problems:
        _fail("; ".join(problems))

    print("MAPPING GATE PASSED")


if __name__ == "__main__":
    main()
