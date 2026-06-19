"""
Unit tests for the HF->Orbax converter's PURE reshape/transpose functions and
the NAME-MAP contract — no real weights needed (synthetic numpy arrays only).

Run with the project venv:
    .venv/bin/python \
        tests/test_converter_units.py

What this proves WITHOUT any checkpoint:
  1. Each pure transform (raw / T / conv / stackT) maps an HF-EXPECTED source
     shape to the correct target leaf shape, AND preserves values (so it is a
     genuine layout op, not a corruptor).
  2. The dtype cast to bf16 is value-preserving within bf16 precision.
  3. The NAME MAP (nemotron_h.hf_name_map) is internally consistent:
       - every target leaf produced by eval_shape on the 'tiny' preset has a
         map entry,
       - the transform each entry names, applied to a synthetic HF source built
         at the config-derived HF shape, yields exactly the target leaf shape
         (a full per-leaf round-trip on the tiny model).

Prints "CONVERTER UNIT TESTS PASSED" on success; raises AssertionError otherwise.
"""

import os
import sys

import numpy as np
import ml_dtypes

# Make `import jax_nemotron...` and `import scripts...` work from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.convert_hf_to_orbax import (
    transform_raw,
    transform_T,
    transform_conv,
    transform_conv2d,
    transform_stackT,
    apply_transform,
    BF16,
)


def _fail(msg: str):
    raise AssertionError(f"CONVERTER UNIT TEST FAILED: {msg}")


# =============================================================================
# 1. Pure transforms on synthetic arrays (shape + value preservation)
# =============================================================================


def test_transform_raw():
    a = np.arange(2688, dtype=np.float32)
    out = transform_raw([a])
    if out.shape != (2688,):
        _fail(f"raw shape {out.shape} != (2688,)")
    if not np.array_equal(out, a):
        _fail("raw must copy values unchanged")
    print("[unit] raw OK")


def test_transform_T():
    # HF Linear weight (out_features, in_features) -> JAX kernel (in, out).
    out_f, in_f = 4096, 2688  # mamba/attention-ish dims
    a = np.arange(out_f * in_f, dtype=np.float32).reshape(out_f, in_f)
    out = transform_T([a])
    if out.shape != (in_f, out_f):
        _fail(f"T shape {out.shape} != {(in_f, out_f)}")
    # value check: out[i, j] == a[j, i]
    if not np.array_equal(out, a.T):
        _fail("T must equal the matrix transpose")
    # lm_head case: HF (vocab, hidden) -> (hidden, vocab)
    v, h = 131072 // 1024, 2688  # shrink vocab for test speed; layout identical
    b = np.arange(v * h, dtype=np.float32).reshape(v, h)
    ob = transform_T([b])
    if ob.shape != (h, v):
        _fail(f"T lm_head shape {ob.shape} != {(h, v)}")
    print("[unit] T OK")


def test_transform_conv():
    # PyTorch depthwise Conv1d (out_ch, in/groups=1, k) -> nnx (k, 1, out_ch).
    conv_dim, k = 4736, 4
    a = np.arange(conv_dim * 1 * k, dtype=np.float32).reshape(conv_dim, 1, k)
    out = transform_conv([a])
    if out.shape != (k, 1, conv_dim):
        _fail(f"conv shape {out.shape} != {(k, 1, conv_dim)}")
    # value check: out[ki, 0, ci] == a[ci, 0, ki]
    if not np.array_equal(out, np.transpose(a, (2, 1, 0))):
        _fail("conv must be a (2,1,0) transpose")
    print("[unit] conv OK")


def test_transform_conv2d():
    # PyTorch Conv2d depthwise (out_ch, in/groups=1, kH, kW) -> nnx (kH, kW, 1, out_ch).
    out_ch, kh, kw = 256, 3, 3
    a = np.arange(out_ch * 1 * kh * kw, dtype=np.float32).reshape(out_ch, 1, kh, kw)
    out = transform_conv2d([a])
    if out.shape != (kh, kw, 1, out_ch):
        _fail(f"conv2d depthwise shape {out.shape} != {(kh, kw, 1, out_ch)}")
    if not np.array_equal(out, np.transpose(a, (2, 3, 1, 0))):
        _fail("conv2d must be a (2,3,1,0) transpose")
    # PyTorch Conv2d pointwise (out_ch, in_ch, 1, 1) -> nnx (1, 1, in_ch, out_ch).
    out_ch, in_ch = 256, 256
    b = np.arange(out_ch * in_ch, dtype=np.float32).reshape(out_ch, in_ch, 1, 1)
    ob = transform_conv2d([b])
    if ob.shape != (1, 1, in_ch, out_ch):
        _fail(f"conv2d pointwise shape {ob.shape} != {(1, 1, in_ch, out_ch)}")
    if not np.array_equal(ob, np.transpose(b, (2, 3, 1, 0))):
        _fail("conv2d pointwise must be a (2,3,1,0) transpose")
    print("[unit] conv2d OK")


def test_transform_stackT():
    # MoE routed experts: N sources each (moe_inter, hidden) -> (N, hidden, moe_inter).
    N, moe_inter, hidden = 8, 32, 64  # tiny dims
    srcs = [
        np.arange(moe_inter * hidden, dtype=np.float32).reshape(moe_inter, hidden) + e
        for e in range(N)
    ]
    out = transform_stackT(srcs)
    if out.shape != (N, hidden, moe_inter):
        _fail(f"stackT shape {out.shape} != {(N, hidden, moe_inter)}")
    # value check: each slab equals the transpose of its source.
    for e in range(N):
        if not np.array_equal(out[e], srcs[e].T):
            _fail(f"stackT slab {e} must equal source.T")
    # shared expert: N == 1 must still get a leading axis of length 1.
    s = np.arange(48 * 64, dtype=np.float32).reshape(48, 64)
    out1 = transform_stackT([s])
    if out1.shape != (1, 64, 48):
        _fail(f"stackT N=1 shape {out1.shape} != (1, 64, 48)")
    print("[unit] stackT OK")


def test_stackT_rejects_ragged():
    a = np.zeros((4, 5), dtype=np.float32)
    b = np.zeros((4, 6), dtype=np.float32)
    try:
        transform_stackT([a, b])
    except AssertionError:
        print("[unit] stackT ragged-rejection OK")
        return
    _fail("stackT must reject sources of differing shape")


def test_apply_transform_dispatch_and_unknown():
    a = np.zeros((3, 7), dtype=np.float32)
    if apply_transform("T", [a]).shape != (7, 3):
        _fail("apply_transform('T') dispatch wrong")
    try:
        apply_transform("nope", [a])
    except ValueError:
        print("[unit] apply_transform dispatch + unknown-name OK")
        return
    _fail("apply_transform must raise on unknown transform name")


def test_bf16_cast_value_preserving():
    # Casting AFTER the shape transform must preserve values within bf16 precision.
    a = np.array([1.0, 2.5, -3.25, 0.0, 1e3], dtype=np.float32)
    out = transform_raw([a]).astype(BF16)
    if out.dtype != ml_dtypes.bfloat16:
        _fail(f"cast dtype {out.dtype} != bfloat16")
    back = out.astype(np.float32)
    # these exact values are representable in bf16, so equality holds.
    if not np.array_equal(back, a):
        _fail(f"bf16 cast altered exactly-representable values: {back} != {a}")
    print("[unit] bf16 cast OK")


# =============================================================================
# 2. NAME-MAP round-trip on the 'tiny' preset (full per-leaf shape contract)
# =============================================================================


def _hf_shape_for(rel_name: str, cfg) -> tuple:
    """Derive the expected HF (PyTorch) shape for a language_model-relative
    tensor name from config dims. PyTorch conventions:
      Linear weight = (out_features, in_features); Conv1d = (out_ch, in/groups, k);
      embedding/lm_head = (rows, cols); 1-D scalars stay 1-D."""
    H = cfg.hidden_size
    if rel_name == "backbone.embeddings.weight":
        return (cfg.vocab_size, H)
    if rel_name == "backbone.norm_f.weight":
        return (H,)
    if rel_name == "lm_head.weight":
        return (cfg.vocab_size, H)

    # strip "backbone.layers.{i}." and dispatch on the suffix.
    assert rel_name.startswith("backbone.layers."), rel_name
    rest = rel_name[len("backbone.layers."):]
    _, suffix = rest.split(".", 1)  # drop the layer index

    if suffix == "norm.weight":
        return (H,)

    d_inner = cfg.mamba_intermediate_size
    conv_dim = cfg.mamba_conv_dim
    in_proj_dim = cfg.mamba_in_proj_dim
    q_dim = cfg.attention_q_dim
    kv_dim = cfg.attention_kv_dim
    si = cfg.moe_shared_expert_intermediate_size
    mi = cfg.moe_intermediate_size

    table = {
        # Mamba
        "mixer.in_proj.weight": (in_proj_dim, H),
        "mixer.conv1d.weight": (conv_dim, 1, cfg.conv_kernel),
        "mixer.conv1d.bias": (conv_dim,),
        "mixer.dt_bias": (cfg.mamba_num_heads,),
        "mixer.A_log": (cfg.mamba_num_heads,),
        "mixer.D": (cfg.mamba_num_heads,),
        "mixer.norm.weight": (d_inner,),
        "mixer.out_proj.weight": (H, d_inner),
        # Attention
        "mixer.q_proj.weight": (q_dim, H),
        "mixer.k_proj.weight": (kv_dim, H),
        "mixer.v_proj.weight": (kv_dim, H),
        "mixer.o_proj.weight": (H, q_dim),
        # MoE non-expert
        "mixer.gate.weight": (cfg.n_routed_experts, H),
        "mixer.gate.e_score_correction_bias": (cfg.n_routed_experts,),
        "mixer.shared_experts.up_proj.weight": (si, H),
        "mixer.shared_experts.down_proj.weight": (H, si),
    }
    if suffix in table:
        return table[suffix]
    # MoE routed experts: mixer.experts.{j}.up_proj.weight / down_proj.weight
    if suffix.startswith("mixer.experts."):
        if suffix.endswith("up_proj.weight"):
            return (mi, H)
        if suffix.endswith("down_proj.weight"):
            return (H, mi)
    raise KeyError(f"no HF shape rule for {rel_name!r} (suffix {suffix!r})")


def test_name_map_roundtrip():
    """For every target leaf of the tiny model, build a synthetic HF source at
    the config-derived shape, run the mapped transform, and assert it produces
    exactly the target leaf shape. This is the converter's contract, executed
    end-to-end on synthetic data (no weights)."""
    import jax  # noqa: F401
    from flax import nnx
    from jax_nemotron.config import NemotronHConfig
    from jax_nemotron.nemotron_h import NemotronHModel, hf_name_map, HF_PREFIX

    cfg = NemotronHConfig.from_preset("tiny")
    cfg.validate()

    def init_fn():
        return NemotronHModel(rngs=nnx.Rngs(0), config=cfg)

    abstract = nnx.eval_shape(init_fn)
    _, state = nnx.split(abstract)

    # Flatten the abstract state to {slash_path -> shape}.
    leaves = jax.tree_util.tree_leaves_with_path(
        state, is_leaf=lambda n: hasattr(n, "value")
    )
    targets = {}
    for path, leaf in leaves:
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
        val = getattr(leaf, "value", leaf)
        targets["/".join(parts)] = tuple(val.shape)

    name_map = hf_name_map(cfg)

    n_checked = 0
    unmapped = []
    for tpath, tshape in targets.items():
        entry = name_map.get(tpath)
        if entry is None:
            trimmed = "/".join(tpath.split("/")[:-1])
            entry = name_map.get(trimmed)
        if entry is None:
            unmapped.append(tpath)
            continue
        hf = entry["hf"]
        transform = entry["transform"]
        rel_names = [hf] if isinstance(hf, str) else list(hf)
        # Build synthetic HF sources at the derived HF shapes.
        arrays = []
        for rn in rel_names:
            shp = _hf_shape_for(rn, cfg)
            arrays.append(np.zeros(shp, dtype=np.float32))
        out = apply_transform(transform, arrays)
        if tuple(out.shape) != tuple(tshape):
            _fail(
                f"round-trip {tpath}: transform={transform} sources={rel_names} "
                f"-> {tuple(out.shape)} != target {tuple(tshape)}"
            )
        # also confirm the FULL hf key (with prefix) is well-formed.
        for rn in rel_names:
            assert (HF_PREFIX + rn).startswith("language_model."), rn
        n_checked += 1

    if unmapped:
        _fail(f"{len(unmapped)} target leaves have no name-map entry: {unmapped[:8]}")

    print(f"[unit] name-map round-trip OK: {n_checked} leaves shape-verified "
          f"(tiny preset, {cfg.num_hidden_layers} layers)")


def main():
    test_transform_raw()
    test_transform_T()
    test_transform_conv()
    test_transform_conv2d()
    test_transform_stackT()
    test_stackT_rejects_ragged()
    test_apply_transform_dispatch_and_unknown()
    test_bf16_cast_value_preserving()
    test_name_map_roundtrip()
    print("CONVERTER UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
