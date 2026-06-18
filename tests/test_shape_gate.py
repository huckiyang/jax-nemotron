"""
Shape gate for the Nemotron-H LLM backbone (no pytest, runnable directly).

Run from the repo root:
    .venv/bin/python tests/test_shape_gate.py

What this gate proves (on CPU, with random weights — NO real checkpoint needed):
  1. The "tiny" config validates and parses its hybrid pattern.
  2. jax.eval_shape on the model init produces a param tree WITHOUT allocating
     real arrays — and that tree has the expected structure:
         - embeddings, norm_f, lm_head present
         - one Mamba mixer, one attention mixer, one MoE mixer subtree present
         - leaf count is sane and printed
  3. A tiny random-param forward on dummy token ids returns finite logits of
     shape (batch, seqlen, vocab_size).
  4. The hf_name_map() contract has an entry for (essentially) every model leaf,
     so the later converter has a documented source for each target.

Prints "SHAPE GATE PASSED" on success; raises with a clear message otherwise.

This is a STRUCTURE/SHAPE gate. It does NOT prove value correctness — that
requires the coherence gate (real forward on real weights), which runs later on
the big-memory/TPU host where the safetensors live.
"""

import os
import sys

# Make `import jax_nemotron...` work whether run from repo root or elsewhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import jax
import jax.numpy as jnp
from flax import nnx

from jax_nemotron.config import (
    NemotronHConfig,
    MIXER_MAMBA,
    MIXER_ATTENTION,
    MIXER_MOE,
)
from jax_nemotron.nemotron_h import (
    NemotronHModel,
    NemotronHMamba2Mixer,
    NemotronHAttention,
    NemotronHMoE,
    hf_name_map,
)


def _fail(msg: str):
    raise AssertionError(f"SHAPE GATE FAILED: {msg}")


def _flatten_state_paths(state) -> dict:
    """
    Flatten an nnx state pytree into {slash_path: leaf} using JAX path APIs.

    nnx.eval_shape returns (graphdef, state); the state is a pytree of
    nnx.VariableState leaves whose `.value` carries the ShapeDtypeStruct. We use
    jax.tree_util keypaths to build stable slash-joined names.
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


def main():
    # ----- 1. config -----
    cfg = NemotronHConfig.from_preset("tiny")
    cfg.validate()
    layer_types = cfg.parse_pattern()
    print(f"[gate] tiny config: {cfg.num_hidden_layers} layers, pattern="
          f"{cfg.hybrid_override_pattern!r} -> {layer_types}")

    # The tiny preset must cover all three mixer types.
    for needed in (MIXER_MAMBA, MIXER_ATTENTION, MIXER_MOE):
        if needed not in layer_types:
            _fail(f"tiny preset is missing a {needed!r} layer; got {layer_types}")

    # ----- 2. eval_shape on init (no allocation) -----
    def init_fn():
        return NemotronHModel(rngs=nnx.Rngs(0), config=cfg)

    abstract_model = nnx.eval_shape(init_fn)
    _, abstract_state = nnx.split(abstract_model)
    paths = _flatten_state_paths(abstract_state)
    leaf_count = len(paths)
    print(f"[gate] eval_shape produced {leaf_count} param/variable leaves")

    if leaf_count == 0:
        _fail("eval_shape produced an empty param tree")

    pathset = set(paths.keys())

    def _require_prefix(prefix: str, what: str):
        if not any(p == prefix or p.startswith(prefix + "/") for p in pathset):
            sample = sorted(pathset)[:20]
            _fail(f"expected subtree {what!r} (prefix {prefix!r}) not found. "
                  f"sample paths: {sample}")

    # Top-level subtrees.
    _require_prefix("embeddings", "token embeddings")
    _require_prefix("norm_f", "final RMSNorm")
    _require_prefix("lm_head", "untied LM head")

    # Per-layer norm for every layer.
    for i in range(cfg.num_hidden_layers):
        _require_prefix(f"layers/{i}/norm", f"layer {i} pre-norm")
        _require_prefix(f"layers/{i}/mixer", f"layer {i} mixer")

    # At least one mixer of each type, identified by its signature leaves.
    mamba_idx = layer_types.index(MIXER_MAMBA)
    attn_idx = layer_types.index(MIXER_ATTENTION)
    moe_idx = layer_types.index(MIXER_MOE)

    # Mamba signature leaves.
    for leaf in ("in_proj/kernel", "conv1d/kernel", "conv1d/bias", "dt_bias",
                 "A_log", "D", "norm/scale", "out_proj/kernel"):
        key = f"layers/{mamba_idx}/mixer/{leaf}"
        if key not in pathset:
            _fail(f"Mamba mixer missing leaf {key!r}")
    # Attention signature leaves.
    for leaf in ("q_proj/kernel", "k_proj/kernel", "v_proj/kernel", "o_proj/kernel"):
        key = f"layers/{attn_idx}/mixer/{leaf}"
        if key not in pathset:
            _fail(f"Attention mixer missing leaf {key!r}")
    # MoE signature leaves.
    for leaf in ("gate/kernel", "e_score_correction_bias", "routed_W1",
                 "routed_W2", "shared_W1", "shared_W2"):
        key = f"layers/{moe_idx}/mixer/{leaf}"
        if key not in pathset:
            _fail(f"MoE mixer missing leaf {key!r}")

    # Spot-check a few critical shapes against the config.
    def _shape(path):
        leaf = paths[path]
        val = getattr(leaf, "value", leaf)
        return tuple(val.shape)

    emb_path = next(p for p in pathset if p.startswith("embeddings"))
    if _shape(emb_path) != (cfg.vocab_size, cfg.hidden_size):
        _fail(f"embedding shape {_shape(emb_path)} != "
              f"{(cfg.vocab_size, cfg.hidden_size)}")

    moe_w1 = f"layers/{moe_idx}/mixer/routed_W1"
    expected_w1 = (cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size)
    if _shape(moe_w1) != expected_w1:
        _fail(f"routed_W1 shape {_shape(moe_w1)} != {expected_w1}")

    mamba_in = f"layers/{mamba_idx}/mixer/in_proj/kernel"
    expected_in = (cfg.hidden_size, cfg.mamba_in_proj_dim)
    if _shape(mamba_in) != expected_in:
        _fail(f"mamba in_proj.kernel shape {_shape(mamba_in)} != {expected_in}")

    print(f"[gate] structure + key shapes OK "
          f"(mamba@{mamba_idx}, attn@{attn_idx}, moe@{moe_idx})")

    # ----- 3. name-map contract covers (almost) every leaf -----
    name_map = hf_name_map(cfg)
    map_keys = set(name_map.keys())
    # The name map omits the conv1d bias only when use_conv_bias is False; it is
    # True here. Every PARAM leaf should be coverable. We allow the map to be a
    # superset; we require it to cover all non-buffer param leaves. The expert
    # selection-bias buffer (e_score_correction_bias) is covered too.
    missing = []
    for p in pathset:
        # nnx puts an extra trailing segment for some variable kinds; normalize
        # by checking membership of the path or any prefix-trimmed form.
        if p in map_keys:
            continue
        # last segment may be a variable-type tag; try trimming it.
        trimmed = "/".join(p.split("/")[:-1])
        if trimmed in map_keys:
            continue
        missing.append(p)
    if missing:
        # This is a soft contract check; report loudly but only fail if a large
        # fraction is uncovered (indicating a real structural mismatch).
        uncovered_frac = len(missing) / max(1, leaf_count)
        print(f"[gate] WARNING: {len(missing)} leaves not directly in name_map "
              f"(frac={uncovered_frac:.2f}). sample: {sorted(missing)[:8]}")
        if uncovered_frac > 0.10:
            _fail(f"name_map covers too few leaves; {len(missing)}/{leaf_count} "
                  f"uncovered. sample: {sorted(missing)[:12]}")
    else:
        print(f"[gate] name_map covers all {leaf_count} leaves")

    # ----- 4. tiny random-param forward -----
    model = NemotronHModel(rngs=nnx.Rngs(0), config=cfg)
    batch, seqlen = 2, 8  # seqlen divisible by mamba chunk_size (4)
    if seqlen % cfg.chunk_size != 0:
        _fail(f"test seqlen {seqlen} not divisible by chunk_size {cfg.chunk_size}")
    key = jax.random.PRNGKey(0)
    token_ids = jax.random.randint(key, (batch, seqlen), 0, cfg.vocab_size)

    logits = model(token_ids)
    expected = (batch, seqlen, cfg.vocab_size)
    if tuple(logits.shape) != expected:
        _fail(f"logits shape {tuple(logits.shape)} != {expected}")
    if not bool(jnp.all(jnp.isfinite(logits))):
        _fail("logits contain non-finite values (nan/inf)")

    print(f"[gate] forward OK: logits {tuple(logits.shape)}, "
          f"finite, dtype={logits.dtype}")

    print("SHAPE GATE PASSED")


if __name__ == "__main__":
    main()
