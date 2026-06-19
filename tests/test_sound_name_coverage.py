"""
Name/shape MAPPING COMPLETENESS GATE for the Nemotron-Omni SOUND path
(Parakeet FastConformer encoder + sound projector).

Run from the repo root (or anywhere — sys.path is fixed up below):
    .venv/bin/python tests/test_sound_name_coverage.py

What this gate proves WITHOUT any weights (shapes only, fits in RAM), mirroring
tests/test_name_coverage.py for the LLM backbone:

  1. Build the REAL "omni_30b" sound encoder + projector and ``jax.eval_shape``
     their init to get the authoritative target param tree {path -> shape}. No
     allocation happens — eval_shape returns ShapeDtypeStructs only.

  2. Load model.safetensors.index.json's weight_map and take the
     ``sound_encoder.*`` / ``sound_projection.*`` subset.

  3. Use the target<->HF NAME MAP from audio_encoder.hf_sound_name_map() and
     assert a FULL BIJECTION over the sound namespace:
       - every target leaf maps to exactly one EXISTING HF key,
       - every HF sound key is consumed by EXACTLY ONE target leaf, EXCEPT the
         per-layer ``conv.norm.num_batches_tracked`` buffers, which are the lone
         allowed unconsumed tensors (an I64 counter, not a param).

  4. Shape check WITHOUT data: derive each HF tensor's expected shape from config
     dims, apply the converter's planned transform (raw / T / conv / conv2d), and
     assert it yields the target leaf shape.

Prints "SOUND MAPPING GATE PASSED" on success.

This is a STRUCTURE/NAME/SHAPE gate. It does NOT prove value correctness — that
requires the coherence gate (real forward on real weights).
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

from jax_nemotron.audio_encoder import AudioEncoder, hf_sound_name_map, HF_SOUND_PREFIX
from jax_nemotron.nemotron_omni import NemotronOmniConfig, SoundProjector

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
    raise AssertionError(f"SOUND MAPPING GATE FAILED: {msg}")


# =============================================================================
# Flatten the eval_shape target tree to {slash_path -> ShapeDtypeStruct}
# =============================================================================


def _flatten_state_paths(state) -> dict:
    """Flatten an nnx state pytree into {slash_path: leaf} using JAX path APIs
    (matches the keys produced by hf_sound_name_map())."""
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


def _expected_hf_shapes(cfg: NemotronOmniConfig) -> dict:
    """
    Derive the expected PyTorch/safetensors shape for every sound_encoder.* /
    sound_projection.* tensor purely from config dims. Returned keys are FULL HF
    keys (HF_SOUND_PREFIX == "").

    PyTorch conventions encoded here:
      * nn.Linear weight is (out_features, in_features).
      * Conv1d weight is (out_channels, in_channels/groups, kernel).
      * Conv2d weight is (out_channels, in_channels/groups, kH, kW).
      * LayerNorm / BatchNorm weight,bias and BN running stats are 1-D (D,).
      * RMSNorm scale is 1-D.
    """
    sc = cfg.sound
    D = sc.hidden_dim            # 1024
    F = sc.ffn_dim               # 4096
    C = sc.subsampling_conv_channels  # 256
    k = sc.subsampling_conv_kernel    # 3
    cv_k = sc.conv_kernel_size   # 9 depthwise
    exp: dict = {}

    # ---- feature extractor frozen buffers ----
    fe = "sound_encoder.encoder.feature_extractor.featurizer"
    exp[f"{fe}.fb"] = (1, sc.n_mels, sc.n_freqs)
    exp[f"{fe}.window"] = (sc.frame_length,)

    # ---- subsampling ----
    ss = "sound_encoder.encoder.subsampling"
    # Conv2d weights. layers 0/2/5 are kxk (0 groups=1 in/g==1; 2,5 depthwise
    # in/g==1); layers 3/6 are pointwise 1x1 with in/g==C.
    exp[f"{ss}.layers.0.weight"] = (C, 1, k, k)
    exp[f"{ss}.layers.0.bias"] = (C,)
    exp[f"{ss}.layers.2.weight"] = (C, 1, k, k)
    exp[f"{ss}.layers.2.bias"] = (C,)
    exp[f"{ss}.layers.3.weight"] = (C, C, 1, 1)
    exp[f"{ss}.layers.3.bias"] = (C,)
    exp[f"{ss}.layers.5.weight"] = (C, 1, k, k)
    exp[f"{ss}.layers.5.bias"] = (C,)
    exp[f"{ss}.layers.6.weight"] = (C, C, 1, 1)
    exp[f"{ss}.layers.6.bias"] = (C,)
    # Final Linear (C * subsampled_freq -> hidden_dim).
    lin_in = C * sc.subsampled_freq
    exp[f"{ss}.linear.weight"] = (D, lin_in)
    exp[f"{ss}.linear.bias"] = (D,)

    # ---- conformer layers ----
    for i in range(sc.num_layers):
        hp = f"sound_encoder.encoder.layers.{i}"
        for norm in ("norm_feed_forward1", "norm_self_att", "norm_conv",
                     "norm_feed_forward2", "norm_out"):
            exp[f"{hp}.{norm}.weight"] = (D,)
            exp[f"{hp}.{norm}.bias"] = (D,)
        for ff in ("feed_forward1", "feed_forward2"):
            exp[f"{hp}.{ff}.linear1.weight"] = (F, D)   # Linear(D->F)
            exp[f"{hp}.{ff}.linear2.weight"] = (D, F)   # Linear(F->D)
        hsa = f"{hp}.self_attn"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"):
            exp[f"{hsa}.{proj}.weight"] = (D, D)
        exp[f"{hsa}.bias_u"] = (sc.num_heads, sc.head_dim)
        exp[f"{hsa}.bias_v"] = (sc.num_heads, sc.head_dim)
        hcv = f"{hp}.conv"
        # Conv1d: pointwise_conv1 (2D, D, 1), depthwise (D,1,k=9), pw2 (D,D,1).
        exp[f"{hcv}.pointwise_conv1.weight"] = (2 * D, D, 1)
        exp[f"{hcv}.depthwise_conv.weight"] = (D, 1, cv_k)
        exp[f"{hcv}.pointwise_conv2.weight"] = (D, D, 1)
        exp[f"{hcv}.norm.weight"] = (D,)
        exp[f"{hcv}.norm.bias"] = (D,)
        exp[f"{hcv}.norm.running_mean"] = (D,)
        exp[f"{hcv}.norm.running_var"] = (D,)

    # ---- sound projector ----
    in_dim = cfg.sound_proj_in            # 1024
    mid = cfg.sound_projector_hidden      # 4096
    out_dim = cfg.llm.hidden_size         # 2688
    exp["sound_projection.norm.weight"] = (in_dim,)
    exp["sound_projection.linear1.weight"] = (mid, in_dim)   # Linear(in->mid)
    exp["sound_projection.linear2.weight"] = (out_dim, mid)  # Linear(mid->out)

    return exp


# =============================================================================
# Transform-aware shape check
# =============================================================================


def _transformed_shape(transform: str, hf_shapes: list) -> tuple:
    """Predict the JAX target shape after the documented converter transform.
    Mirrors scripts/convert_hf_to_orbax.py exactly."""
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
        out_ch, in_per_group, kk = s
        return (kk, in_per_group, out_ch)
    if transform == "conv2d":
        assert len(hf_shapes) == 1, "conv2d expects exactly one source"
        s = hf_shapes[0]
        assert len(s) == 4, f"conv2d expects 4-D source, got {s}"
        out_ch, in_per_group, kh, kw = s
        return (kh, kw, in_per_group, out_ch)
    raise ValueError(f"unknown transform {transform!r}")


def main():
    # ----- 1. authoritative target tree from the REAL sound encoder+projector -----
    cfg = NemotronOmniConfig.from_preset("omni_30b")
    cfg.validate()
    sc = cfg.sound
    print(f"[gate] omni_30b sound config: {sc.num_layers} layers, hidden="
          f"{sc.hidden_dim}, heads={sc.num_heads}x{sc.head_dim}")

    def init_encoder():
        return AudioEncoder(sc, rngs=nnx.Rngs(0))

    def init_projector():
        return SoundProjector(
            rngs=nnx.Rngs(1),
            in_dim=cfg.sound_proj_in,
            mid_dim=cfg.sound_projector_hidden,
            out_dim=cfg.llm.hidden_size,
            eps=cfg.llm.norm_eps,
        )

    targets = {}
    for init_fn, prefix in ((init_encoder, "sound_encoder/"),
                            (init_projector, "sound_projection/")):
        abstract = nnx.eval_shape(init_fn)
        _, abstract_state = nnx.split(abstract)
        for path, leaf in _flatten_state_paths(abstract_state).items():
            targets[prefix + path] = leaf
    n_targets = len(targets)
    print(f"[gate] eval_shape produced {n_targets} sound target leaves (no allocation)")
    if n_targets == 0:
        _fail("eval_shape produced an empty sound target tree")

    # ----- 2. HF sound keys from the index -----
    if not os.path.exists(_INDEX_JSON):
        print(f"[gate] SKIP: checkpoint index not found at {_INDEX_JSON}")
        print("[gate] set $NEMOTRON_CKPT_DIR to the HF checkpoint dir to run this "
              "gate (needs only model.safetensors.index.json, no weights).")
        return
    with open(_INDEX_JSON) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    hf_all = set(weight_map.keys())
    hf_sound = {k for k in hf_all
                if k.startswith("sound_encoder.") or k.startswith("sound_projection.")}
    print(f"[gate] index has {len(hf_all)} tensors; {len(hf_sound)} sound tensors")

    # The lone allowed-unconsumed tensors: BatchNorm num_batches_tracked counters.
    nbt = {k for k in hf_sound if k.endswith("num_batches_tracked")}
    print(f"[gate] num_batches_tracked buffers (allowed unconsumed): {len(nbt)}")

    # ----- 3. name map + bijection -----
    name_map = hf_sound_name_map(cfg)

    target_to_hf = {}
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
        full = [HF_SOUND_PREFIX + name for name in hf_list]
        target_to_hf[tp] = (full, entry["transform"])

    # stale entries: name_map keys not present as a target (or trimmed target).
    target_pathset = set(targets.keys())
    stale_map_entries = []
    for tp in name_map:
        if tp in target_pathset:
            continue
        if any("/".join(t.split("/")[:-1]) == tp for t in target_pathset):
            continue
        stale_map_entries.append(tp)

    consumed = {}
    missing_hf_sources = []
    double_claimed = []
    for tp, (full_list, _t) in target_to_hf.items():
        for hk in full_list:
            if hk not in hf_sound:
                missing_hf_sources.append((tp, hk))
            if hk in consumed:
                double_claimed.append((hk, consumed[hk], tp))
            else:
                consumed[hk] = tp

    consumed_sound = set(consumed.keys()) & hf_sound
    unconsumed_hf = sorted(hf_sound - consumed_sound)
    # The ONLY allowed unconsumed tensors are the num_batches_tracked buffers.
    unexpected_unconsumed = sorted(set(unconsumed_hf) - nbt)
    missing_nbt = sorted(nbt - set(unconsumed_hf))  # any nbt we wrongly consumed

    matched = len(target_to_hf)

    print("\n[gate] ---- counts ----")
    print(f"[gate] targets (our leaves)          : {n_targets}")
    print(f"[gate] hf_sound_keys                  : {len(hf_sound)}")
    print(f"[gate] matched (targets w/ hf source) : {matched}")
    print(f"[gate] unmapped_targets               : {len(unmapped_targets)}")
    print(f"[gate] unconsumed_hf                   : {len(unconsumed_hf)}")
    print(f"[gate] unexpected_unconsumed          : {len(unexpected_unconsumed)}")
    print(f"[gate] missing_hf_sources             : {len(missing_hf_sources)}")
    print(f"[gate] double_claimed_hf              : {len(double_claimed)}")
    print(f"[gate] stale_map_entries              : {len(stale_map_entries)}")

    if unmapped_targets:
        print(f"[gate] sample unmapped_targets: {sorted(unmapped_targets)[:12]}")
    if missing_hf_sources:
        print(f"[gate] sample missing_hf_sources: {missing_hf_sources[:12]}")
    if double_claimed:
        print(f"[gate] sample double_claimed: {double_claimed[:12]}")
    if unexpected_unconsumed:
        print(f"[gate] sample unexpected_unconsumed: {unexpected_unconsumed[:12]}")
    if stale_map_entries:
        print(f"[gate] sample stale_map_entries: {sorted(stale_map_entries)[:12]}")

    # ----- 4. shape check (transform-aware, no data) -----
    exp_hf = _expected_hf_shapes(cfg)
    shape_mismatches = []
    shape_unknown_hf = []
    for tp, (full_list, transform) in target_to_hf.items():
        rel_names = [hk[len(HF_SOUND_PREFIX):] for hk in full_list]
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

    # ----- verdict: full bijection (minus num_batches_tracked) + shapes ok -----
    problems = []
    if unmapped_targets:
        problems.append(f"{len(unmapped_targets)} target leaves have no HF source")
    if missing_hf_sources:
        problems.append(f"{len(missing_hf_sources)} mapped HF names absent from index")
    if double_claimed:
        problems.append(f"{len(double_claimed)} HF tensors double-claimed")
    if unexpected_unconsumed:
        problems.append(f"{len(unexpected_unconsumed)} sound HF tensors unconsumed "
                        "beyond num_batches_tracked")
    if missing_nbt:
        problems.append(f"{len(missing_nbt)} num_batches_tracked tensors were consumed "
                        "(should be left unconsumed)")
    if stale_map_entries:
        problems.append(f"{len(stale_map_entries)} stale name_map entries")
    if shape_mismatches:
        problems.append(f"{len(shape_mismatches)} shape mismatches")
    if shape_unknown_hf:
        problems.append(f"{len(set(shape_unknown_hf))} HF shapes underivable")

    print(
        f"\nSOUND MAPPING GATE: {matched}/{n_targets} matched, "
        f"{len(unconsumed_hf)} HF keys unconsumed "
        f"({len(nbt)} expected = num_batches_tracked)"
    )

    if problems:
        _fail("; ".join(problems))

    print("SOUND MAPPING GATE PASSED")


if __name__ == "__main__":
    main()
