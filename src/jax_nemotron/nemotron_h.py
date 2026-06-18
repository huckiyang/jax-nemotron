"""
Nemotron-H LLM backbone in Flax NNX — HF-faithful, conversion-ready.

This module assembles the language-model backbone of
NVIDIA Nemotron-3-Nano-Omni-30B-A3B (HF model_type ``nemotron_h``) so that the
parameter tree produced by ``jax.eval_shape(nnx.eval_shape...)`` maps cleanly,
name-for-name and shape-for-shape, onto the HuggingFace ``language_model.*``
safetensors tensors. The later HF->Orbax converter walks THIS tree, pulls each
leaf from the safetensors by the documented HF name, reshapes/transposes, asserts
the shape, casts to bf16, and writes Orbax. So the contract that matters is:

    OUR pytree path   <->   HF tensor name

documented exhaustively in the NAME-MAP table at the bottom of this file.

------------------------------------------------------------------------------
Architecture (one mixer per layer; flat stack)
------------------------------------------------------------------------------
    token_ids
      -> embeddings (nnx.Embed: vocab x hidden)            [backbone.embeddings.weight]
      -> for i in range(num_hidden_layers):
             h = h + mixer_i(norm_i(h))                    [backbone.layers.i.*]
         where mixer_i is Mamba2 / GQA-attention / MoE per parse_pattern()[i]
      -> final RMSNorm norm_f                              [backbone.norm_f.weight]
      -> lm_head (untied)                                  [lm_head.weight]

Each layer is pre-normed with ONE RMSNorm (``backbone.layers.i.norm.weight``)
and wrapped in a residual. This matches HF NemotronHBlock exactly (one norm, one
mixer, residual add).

------------------------------------------------------------------------------
HF-faithful conventions baked in (from Phase-1 ground truth)
------------------------------------------------------------------------------
  * RMSNorm: PLAIN  out = weight * (x / sqrt(mean(x^2) + eps)). NO (1+weight).
  * Mamba2 inner width d_inner = mamba_num_heads*mamba_head_dim (= 4096), NOT
    expand*hidden. in_proj splits [gate, xBC, dt]; xBC further [x, B, C].
    Activation SiLU. conv1d has bias; nothing else does. A = -exp(A_log).
  * Attention: GQA 32q/2kv, head_dim 128, NO bias, half-split RoPE theta=10000,
    full rotary (partial_rotary_factor=1.0), scale = head_dim**-0.5.
  * MoE: sigmoid gate + e_score_correction_bias for SELECTION; gate weights from
    the (unbiased) sigmoid scores, top-6, renormalized (norm_topk_prob), then
    scaled by routed_scaling_factor=2.5. relu2 (squared-ReLU) experts. 1 shared
    expert always-on. No bias anywhere in MoE.

We reuse the vendored SSD kernel (``ssd_minimal_discrete``/``segsum`` from
mamba_2.py) for the Mamba recurrence, but build the Mamba parameters here with
the correct HF inner width (the vendored Mamba2Block hardcodes
d_inner=expand*d_model, which is wrong for this checkpoint).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

# The vendored modules live alongside this file in the same package.
from .config import (
    NemotronHConfig,
    MIXER_MAMBA,
    MIXER_ATTENTION,
    MIXER_MOE,
)
from .mamba_2 import ssd_minimal_discrete


# =============================================================================
# RMSNorm (HF-faithful: plain weight * normalized, no offset)
# =============================================================================


class RMSNorm(nnx.Module):
    """
    Plain RMSNorm matching NemotronHRMSNorm exactly.

        var = mean(x^2, axis=-1)
        x_hat = x * rsqrt(var + eps)
        out = scale * x_hat            # PLAIN. No (1 + scale).

    The scale parameter is named ``scale`` in our tree; it maps to the HF
    ``*.weight`` of the corresponding norm. Initialized to ones so a fresh model
    is the identity scale (matches HF init).
    """

    def __init__(self, rngs: nnx.Rngs, num_features: int, eps: float):
        # rngs is accepted for a uniform constructor signature; ones-init needs none.
        del rngs
        self.eps = eps
        self.scale = nnx.Param(jnp.ones((num_features,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        # Compute in float32 for numerical stability, like HF (which upcasts).
        dtype = x.dtype
        xf = x.astype(jnp.float32)
        var = jnp.mean(xf * xf, axis=-1, keepdims=True)
        xf = xf * jax.lax.rsqrt(var + self.eps)
        out = self.scale.get_value() * xf.astype(dtype)
        return out


# =============================================================================
# Rotary position embedding (half-split / rotate_half), HF style
# =============================================================================


def _rotate_half(x: jax.Array) -> jax.Array:
    """
    HF rotate_half: split the last dim in two halves and rotate.
        x = [x1 ; x2]  ->  [-x2 ; x1]
    This is the HALF-SPLIT convention (not interleaved). Because the HF
    Nemotron-H attention uses this exact convention, and we use it here too, NO
    q/k permutation is needed during conversion.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def _build_rope_cos_sin(
    seqlen: int, rotary_dim: int, theta: float, dtype=jnp.float32
) -> tuple[jax.Array, jax.Array]:
    """
    Build (cos, sin) tables of shape (seqlen, rotary_dim) for half-split RoPE.

    The standard half-split layout duplicates each of the rotary_dim/2 base
    frequencies across the two halves: freqs = [f0..f_{d/2-1}, f0..f_{d/2-1}].
    This matches HF's ``emb = cat((freqs, freqs), dim=-1)``.
    """
    assert rotary_dim % 2 == 0, "rotary_dim must be even"
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    t = jnp.arange(seqlen, dtype=jnp.float32)
    freqs = jnp.einsum("t,f->tf", t, inv_freq)  # (seqlen, half)
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (seqlen, rotary_dim)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def _apply_rope(
    x: jax.Array, cos: jax.Array, sin: jax.Array, rotary_dim: int
) -> jax.Array:
    """
    Apply half-split RoPE to ``x`` of shape (batch, heads, seqlen, head_dim).

    Rotates the first ``rotary_dim`` channels; passes the rest through unchanged
    (no-op when rotary_dim == head_dim, i.e. partial_rotary_factor == 1.0).
    cos/sin are (seqlen, rotary_dim) and broadcast over batch/heads.
    """
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    cos_b = cos[None, None, :, :]  # (1, 1, seqlen, rotary_dim)
    sin_b = sin[None, None, :, :]
    x_rot = x_rot * cos_b + _rotate_half(x_rot) * sin_b
    if x_pass.shape[-1] == 0:
        return x_rot
    return jnp.concatenate([x_rot, x_pass], axis=-1)


# =============================================================================
# Mamba2 mixer (HF-faithful inner width)
# =============================================================================


class NemotronHMamba2Mixer(nnx.Module):
    """
    Mamba2 mixer matching HF NemotronHMamba2Mixer parameter layout.

    in_proj : Linear(hidden -> d_inner + conv_dim + nheads), no bias
              split into [gate(z): d_inner, xBC: conv_dim, dt: nheads]
    conv1d  : depthwise causal Conv1d over xBC (conv_dim channels), WITH bias
              xBC then splits into [x: d_inner, B: ngroups*state, C: ngroups*state]
    dt_bias : (nheads,)         added pre-softplus
    A_log   : (nheads,)         A = -exp(A_log); init log(1..nheads)
    D       : (nheads,)         skip connection; init ones
    norm    : gated RMSNorm over d_inner (we gate THEN norm, matching HF)
    out_proj: Linear(d_inner -> hidden), no bias

    Inner width d_inner = mamba_num_heads * mamba_head_dim (HF convention),
    decoupled from expand*hidden_size. We reuse the vendored chunked SSD kernel
    for the recurrence.

    HF pytree-name correspondence (this module is mounted at .mixer):
        in_proj.kernel   <- in_proj.weight     (transpose: (out,in)->(in,out))
        conv1d.kernel    <- conv1d.weight      (reshape PyTorch->JAX conv layout)
        conv1d.bias      <- conv1d.bias
        dt_bias          <- dt_bias
        A_log            <- A_log
        D                <- D
        norm.scale       <- norm.weight        (gated RMSNorm scale)
        out_proj.kernel  <- out_proj.weight    (transpose)
    """

    def __init__(self, rngs: nnx.Rngs, config: NemotronHConfig):
        self.hidden_size = config.hidden_size
        self.d_inner = config.mamba_intermediate_size  # = num_heads*head_dim
        self.nheads = config.mamba_num_heads
        self.headdim = config.mamba_head_dim
        self.ngroups = config.mamba_n_groups
        self.d_state = config.ssm_state_size
        self.d_conv = config.conv_kernel
        self.chunk_size = config.chunk_size
        self.conv_dim = config.mamba_conv_dim
        self.norm_eps = config.norm_eps

        # in_proj: hidden -> [gate, xBC, dt]. No bias (use_bias=False in HF).
        self.in_proj = nnx.Linear(
            self.hidden_size,
            config.mamba_in_proj_dim,
            use_bias=False,
            rngs=rngs,
        )

        # Depthwise causal conv over the conv_dim channels (x, B, C concatenated).
        # use_bias=True is the ONLY bias in the Mamba mixer.
        self.conv1d = nnx.Conv(
            in_features=self.conv_dim,
            out_features=self.conv_dim,
            kernel_size=(self.d_conv,),
            feature_group_count=self.conv_dim,  # depthwise
            use_bias=config.use_conv_bias,
            padding="VALID",  # we pad causally on the left ourselves
            rngs=rngs,
        )

        # Step-size bias (pre-softplus). HF inits to ones.
        self.dt_bias = nnx.Param(jnp.ones((self.nheads,)))

        # A_log init = log(1..nheads); A = -exp(A_log) (always negative -> decay).
        self.A_log = nnx.Param(
            jnp.log(jnp.arange(1, self.nheads + 1, dtype=jnp.float32))
        )

        # D skip connection, init ones.
        self.D = nnx.Param(jnp.ones((self.nheads,)))

        # Gated RMSNorm over d_inner. HF uses Zamba2RMSNormGated which gates
        # (y * silu(z)) THEN normalizes with a plain scale; we reproduce that.
        self.norm = RMSNorm(rngs, self.d_inner, config.norm_eps)

        self.out_proj = nnx.Linear(
            self.d_inner, self.hidden_size, use_bias=False, rngs=rngs
        )

    def __call__(self, u: jax.Array) -> jax.Array:
        batch, seqlen, _ = u.shape
        if seqlen % self.chunk_size != 0:
            raise ValueError(
                f"seqlen ({seqlen}) must be divisible by chunk_size "
                f"({self.chunk_size})"
            )

        # --- 1. in_proj and split [gate, xBC, dt] ---
        zxbcdt = self.in_proj(u)
        z, xBC, dt = jnp.split(
            zxbcdt,
            [self.d_inner, self.d_inner + self.conv_dim],
            axis=-1,
        )
        # z:   (B, T, d_inner)   gate
        # xBC: (B, T, conv_dim)
        # dt:  (B, T, nheads)

        # --- 2. causal depthwise conv + SiLU ---
        xBC_padded = jnp.pad(xBC, ((0, 0), (self.d_conv - 1, 0), (0, 0)))
        xBC = self.conv1d(xBC_padded)
        xBC = jax.nn.silu(xBC)

        x, B, C = jnp.split(
            xBC,
            [self.d_inner, self.d_inner + self.ngroups * self.d_state],
            axis=-1,
        )
        x = jnp.reshape(x, (batch, seqlen, self.nheads, self.headdim))
        B = jnp.reshape(B, (batch, seqlen, self.ngroups, self.d_state))
        C = jnp.reshape(C, (batch, seqlen, self.ngroups, self.d_state))

        # Broadcast grouped B,C to all heads (n_groups -> nheads).
        reps = self.nheads // self.ngroups
        B = jnp.repeat(B, reps, axis=2)
        C = jnp.repeat(C, reps, axis=2)

        # --- 3. discretize and run chunked SSD ---
        dt = jax.nn.softplus(dt + self.dt_bias.get_value())  # (B, T, nheads)
        A = -jnp.exp(self.A_log.get_value())  # (nheads,)
        A_discrete = A * dt  # (B, T, nheads)
        X = x * dt[..., None]  # (B, T, nheads, headdim)

        y = ssd_minimal_discrete(X, A_discrete, B, C, self.chunk_size)
        y = y + self.D.get_value()[None, None, :, None] * x
        y = jnp.reshape(y, (batch, seqlen, self.d_inner))

        # --- 4. gate (silu(z)), gated norm, out_proj ---
        y = y * jax.nn.silu(z)
        y = self.norm(y)
        return self.out_proj(y)


# =============================================================================
# Attention mixer (GQA + half-split RoPE), HF-faithful
# =============================================================================


class NemotronHAttention(nnx.Module):
    """
    Grouped-query causal self-attention with half-split RoPE.

    q_proj : Linear(hidden -> num_attention_heads*head_dim), no bias
    k_proj : Linear(hidden -> num_key_value_heads*head_dim), no bias
    v_proj : Linear(hidden -> num_key_value_heads*head_dim), no bias
    o_proj : Linear(num_attention_heads*head_dim -> hidden), no bias
    RoPE   : half-split, theta=10000, rotary_dim = head_dim*partial_rotary_factor
    scale  : head_dim ** -0.5
    NO q/k layernorm. NO attention bias.

    HF pytree-name correspondence (mounted at .mixer):
        q_proj.kernel   <- q_proj.weight  (transpose)
        k_proj.kernel   <- k_proj.weight  (transpose)
        v_proj.kernel   <- v_proj.weight  (transpose)
        o_proj.kernel   <- o_proj.weight  (transpose)
    """

    def __init__(self, rngs: nnx.Rngs, config: NemotronHConfig):
        self.hidden_size = config.hidden_size
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.rope_theta = config.rope_theta
        self.scale = self.head_dim ** -0.5
        self.kv_repeat = self.num_q_heads // self.num_kv_heads

        bias = config.attention_bias
        self.q_proj = nnx.Linear(
            self.hidden_size, config.attention_q_dim, use_bias=bias, rngs=rngs
        )
        self.k_proj = nnx.Linear(
            self.hidden_size, config.attention_kv_dim, use_bias=bias, rngs=rngs
        )
        self.v_proj = nnx.Linear(
            self.hidden_size, config.attention_kv_dim, use_bias=bias, rngs=rngs
        )
        self.o_proj = nnx.Linear(
            config.attention_q_dim, self.hidden_size, use_bias=bias, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        batch, seqlen, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = jnp.reshape(q, (batch, seqlen, self.num_q_heads, self.head_dim))
        k = jnp.reshape(k, (batch, seqlen, self.num_kv_heads, self.head_dim))
        v = jnp.reshape(v, (batch, seqlen, self.num_kv_heads, self.head_dim))

        # (batch, heads, seqlen, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # RoPE on Q and K (half-split). cos/sin: (seqlen, rotary_dim).
        cos, sin = _build_rope_cos_sin(
            seqlen, self.rotary_dim, self.rope_theta, dtype=q.dtype
        )
        q = _apply_rope(q, cos, sin, self.rotary_dim)
        k = _apply_rope(k, cos, sin, self.rotary_dim)

        # GQA: broadcast KV heads to match query heads.
        k = jnp.repeat(k, self.kv_repeat, axis=1)
        v = jnp.repeat(v, self.kv_repeat, axis=1)

        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        causal = jnp.tril(jnp.ones((seqlen, seqlen), dtype=bool))
        scores = jnp.where(causal[None, None, :, :], scores, -1e30)
        # Softmax in float32 (HF computes attn weights in fp32).
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)
        ctx = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

        ctx = jnp.transpose(ctx, (0, 2, 1, 3))
        ctx = jnp.reshape(ctx, (batch, seqlen, self.num_q_heads * self.head_dim))
        return self.o_proj(ctx)


# =============================================================================
# MoE / MLP mixer (sigmoid-gated top-k + shared expert + relu2), HF-faithful
# =============================================================================


def _relu2(x: jax.Array) -> jax.Array:
    """Squared-ReLU activation: relu(x)^2 (HF mlp_hidden_act='relu2')."""
    r = jax.nn.relu(x)
    return r * r


class NemotronHMoE(nnx.Module):
    """
    Sparse MoE mixer matching HF NemotronHMoE / NemotronHTopkRouter.

    Routing (route_tokens_to_experts in HF, lines 781-804):
        scores = sigmoid(router_logits)                       # independent, NOT softmax
        biased = scores + e_score_correction_bias             # SELECTION ONLY
        topk_indices = top_k(biased, num_experts_per_tok)
        topk_weights = scores.gather(topk_indices)            # UNBIASED scores for gating
        if norm_topk_prob: topk_weights /= sum(topk_weights)  # renormalize
        topk_weights *= routed_scaling_factor                 # 2.5
    Experts (relu2 FFN, no bias):
        up_proj   : Linear(hidden -> moe_intermediate_size)
        down_proj : Linear(moe_intermediate_size -> hidden)
    Shared expert (always-on, no gate):
        up_proj   : Linear(hidden -> moe_shared_expert_intermediate_size)
        down_proj : Linear(moe_shared_expert_intermediate_size -> hidden)

    Param storage: routed experts are PRE-STACKED so the converter gathers
    HF experts.{i}.{up,down}_proj.weight into axis 0. Shapes mirror the vendored
    SparseMoE so the converter logic is shared:
        routed_W1 : (n_routed_experts, hidden, moe_intermediate_size)
        routed_W2 : (n_routed_experts, moe_intermediate_size, hidden)
        shared_W1 : (n_shared_experts, hidden, shared_intermediate_size)
        shared_W2 : (n_shared_experts, shared_intermediate_size, hidden)

    HF pytree-name correspondence (mounted at .mixer):
        gate.kernel    <- gate.weight                  (transpose (E,H)->(H,E))
        e_score_correction_bias <- gate.e_score_correction_bias
        routed_W1[i]   <- experts.{i}.up_proj.weight   (transpose each (h,H)->(H,h))
        routed_W2[i]   <- experts.{i}.down_proj.weight (transpose each (H,h)->(h,H))
        shared_W1[0]   <- shared_experts.up_proj.weight   (transpose)
        shared_W2[0]   <- shared_experts.down_proj.weight (transpose)
    """

    def __init__(self, rngs: nnx.Rngs, config: NemotronHConfig):
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.n_shared_experts = config.n_shared_experts
        self.moe_inter = config.moe_intermediate_size
        self.shared_inter = config.moe_shared_expert_intermediate_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob

        # Router gate: hidden -> n_routed_experts logits. No bias.
        self.gate = nnx.Linear(
            self.hidden_size, self.n_routed_experts, use_bias=False, rngs=rngs
        )
        # Selection-bias buffer (zeros init). Stored as plain Variable: it is NOT
        # a gradient parameter (it is a load-balancing buffer in HF too).
        self.e_score_correction_bias = nnx.Variable(
            jnp.zeros((self.n_routed_experts,))
        )

        init = nnx.initializers.lecun_normal()
        self.routed_W1 = nnx.Param(
            init(rngs.params(), (self.n_routed_experts, self.hidden_size, self.moe_inter))
        )
        self.routed_W2 = nnx.Param(
            init(rngs.params(), (self.n_routed_experts, self.moe_inter, self.hidden_size))
        )
        if self.n_shared_experts > 0:
            self.shared_W1 = nnx.Param(
                init(rngs.params(), (self.n_shared_experts, self.hidden_size, self.shared_inter))
            )
            self.shared_W2 = nnx.Param(
                init(rngs.params(), (self.n_shared_experts, self.shared_inter, self.hidden_size))
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        batch, seqlen, d = x.shape
        n = batch * seqlen
        xf = jnp.reshape(x, (n, d))

        # --- routing ---
        logits = self.gate(xf)  # (n, E)
        scores = jax.nn.sigmoid(logits)  # independent per-expert scores
        biased = scores + self.e_score_correction_bias.get_value()  # selection only
        _, topk_idx = jax.lax.top_k(biased, self.top_k)  # (n, top_k)

        tok = jnp.arange(n)[:, None]
        topk_w = scores[tok, topk_idx]  # UNBIASED scores for gating, (n, top_k)
        if self.norm_topk_prob:
            topk_w = topk_w / (jnp.sum(topk_w, axis=-1, keepdims=True) + 1e-20)
        topk_w = topk_w * self.routed_scaling_factor

        # --- routed experts (gather selected weights, relu2 FFN) ---
        W1 = self.routed_W1.get_value()  # (E, H, h)
        W2 = self.routed_W2.get_value()  # (E, h, H)
        W1_sel = W1[topk_idx]  # (n, top_k, H, h)
        W2_sel = W2[topk_idx]  # (n, top_k, h, H)
        hdn = jnp.einsum("nd,nkdh->nkh", xf, W1_sel)
        hdn = _relu2(hdn)
        routed_out = jnp.einsum("nkh,nkhd->nkd", hdn, W2_sel)  # (n, top_k, H)
        routed_mix = jnp.sum(routed_out * topk_w[:, :, None], axis=1)  # (n, H)

        # --- shared expert(s), always-on ---
        if self.n_shared_experts > 0:
            sh = jnp.einsum("nd,edh->neh", xf, self.shared_W1.get_value())
            sh = _relu2(sh)
            sh = jnp.einsum("neh,ehd->ned", sh, self.shared_W2.get_value())
            shared_mix = jnp.sum(sh, axis=1)  # (n, H)
            yf = routed_mix + shared_mix
        else:
            yf = routed_mix

        return jnp.reshape(yf, (batch, seqlen, d))


# =============================================================================
# One backbone layer = pre-norm + single mixer + residual
# =============================================================================


class NemotronHLayer(nnx.Module):
    """
    A single backbone layer: ``h = h + mixer(norm(h))``.

    Exactly one mixer (Mamba2 / attention / MoE) chosen by the hybrid pattern,
    and exactly one pre-RMSNorm — matching HF NemotronHBlock.

    HF pytree-name correspondence (mounted at backbone.layers.{i}):
        norm.scale  <- norm.weight
        mixer.*     <- mixer.*   (see the per-mixer modules above)
    """

    def __init__(self, rngs: nnx.Rngs, config: NemotronHConfig, mixer_type: str):
        self.mixer_type = mixer_type
        self.norm = RMSNorm(rngs, config.hidden_size, config.norm_eps)

        if mixer_type == MIXER_MAMBA:
            self.mixer = NemotronHMamba2Mixer(rngs, config)
        elif mixer_type == MIXER_ATTENTION:
            self.mixer = NemotronHAttention(rngs, config)
        elif mixer_type == MIXER_MOE:
            self.mixer = NemotronHMoE(rngs, config)
        else:
            raise ValueError(f"Unknown mixer_type {mixer_type!r}")

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.mixer(self.norm(x))


# =============================================================================
# Full backbone
# =============================================================================


class NemotronHModel(nnx.Module):
    """
    The Nemotron-H LLM backbone: embeddings -> N layers -> norm_f -> lm_head.

    HF pytree-name correspondence (top level):
        embeddings.embedding <- backbone.embeddings.weight   (same (vocab, hidden))
        layers[i].*          <- backbone.layers.{i}.*
        norm_f.scale         <- backbone.norm_f.weight
        lm_head.kernel       <- lm_head.weight               (transpose (V,H)->(H,V))

    See NAME_MAP / hf_name_map() at the bottom of this file for the full,
    machine-usable contract the converter implements.
    """

    def __init__(self, rngs: nnx.Rngs, config: NemotronHConfig):
        config.validate()
        self.config = config

        self.embeddings = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            rngs=rngs,
        )

        layer_types = config.parse_pattern()
        self.layers = nnx.List(
            [NemotronHLayer(rngs, config, mt) for mt in layer_types]
        )

        self.norm_f = RMSNorm(rngs, config.hidden_size, config.norm_eps)

        # Untied LM head (tie_word_embeddings=False).
        self.lm_head = nnx.Linear(
            config.hidden_size, config.vocab_size, use_bias=False, rngs=rngs
        )

    def __call__(self, token_ids: jax.Array) -> jax.Array:
        """
        Args:
            token_ids: int array (batch, seqlen).
        Returns:
            logits: (batch, seqlen, vocab_size).
        """
        x = self.embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)


# =============================================================================
# NAME-MAP CONTRACT (target pytree path  <->  HF tensor name)
# =============================================================================
#
# The converter (later phase) builds this model's param tree via eval_shape and,
# for each leaf, looks up the HF tensor by the name below, applies the transform,
# asserts the resulting shape equals the target leaf shape, casts bf16, writes.
#
# HF tensors are prefixed ``language_model.`` (the omni wrapper). We drop that
# prefix in the table below and document it once here:
#   HF_PREFIX = "language_model."
#
# Transforms:
#   T        = transpose a 2D linear weight from PyTorch (out,in) to JAX (in,out)
#   conv     = reshape/transpose PyTorch Conv1d (out_ch, in_ch/groups, k) to
#              JAX nnx.Conv kernel (k, in_ch/groups, out_ch); depthwise here so
#              in_ch/groups == 1.
#   stackT   = gather experts.{i}.<proj>.weight over i into axis 0, each .T
#   raw      = copy as-is (no transpose), shape unchanged
#
# ------------------------------------------------------------------------------
# Top level
#   embeddings.embedding   <- backbone.embeddings.weight        raw   (vocab, hidden)
#   norm_f.scale           <- backbone.norm_f.weight            raw   (hidden,)
#   lm_head.kernel         <- lm_head.weight                    T     (V,H)->(H,V)
#
# Per layer i  (target prefix: layers[i].   HF prefix: backbone.layers.{i}.)
#   norm.scale             <- norm.weight                       raw   (hidden,)
#
#   -- if layer i is Mamba ('M'); target prefix layers[i].mixer. / HF mixer.
#   mixer.in_proj.kernel   <- mixer.in_proj.weight              T
#   mixer.conv1d.kernel    <- mixer.conv1d.weight               conv
#   mixer.conv1d.bias      <- mixer.conv1d.bias                 raw
#   mixer.dt_bias          <- mixer.dt_bias                     raw
#   mixer.A_log            <- mixer.A_log                        raw
#   mixer.D                <- mixer.D                            raw
#   mixer.norm.scale       <- mixer.norm.weight                 raw   (gated RMSNorm)
#   mixer.out_proj.kernel  <- mixer.out_proj.weight             T
#
#   -- if layer i is attention ('*')
#   mixer.q_proj.kernel    <- mixer.q_proj.weight               T
#   mixer.k_proj.kernel    <- mixer.k_proj.weight               T
#   mixer.v_proj.kernel    <- mixer.v_proj.weight               T
#   mixer.o_proj.kernel    <- mixer.o_proj.weight               T
#       (NO RoPE permutation: HF and we both use half-split rotate_half.)
#
#   -- if layer i is MoE ('E')
#   mixer.gate.kernel              <- mixer.gate.weight                       T
#   mixer.e_score_correction_bias  <- mixer.gate.e_score_correction_bias      raw
#   mixer.routed_W1                <- mixer.experts.{0..127}.up_proj.weight    stackT
#   mixer.routed_W2                <- mixer.experts.{0..127}.down_proj.weight  stackT
#   mixer.shared_W1                <- mixer.shared_experts.up_proj.weight      T -> [None]
#   mixer.shared_W2                <- mixer.shared_experts.down_proj.weight    T -> [None]
# ------------------------------------------------------------------------------

HF_PREFIX = "language_model."


def hf_name_map(config: NemotronHConfig) -> dict:
    """
    Build the machine-usable target-path -> {hf_name(s), transform} contract.

    Returns a dict keyed by a slash-joined target pytree path (the converter
    flattens the eval_shape tree to the same kind of path). Each value is a dict:
        {"hf": <str or list[str]>, "transform": <"raw"|"T"|"conv"|"stackT">}
    HF names here are RELATIVE to HF_PREFIX (prepend it for the safetensors key).

    This is documentation-as-code: the converter can either consume this dict
    directly or use it as the authoritative checklist.
    """
    m: dict = {}

    # Top level
    m["embeddings/embedding"] = {
        "hf": "backbone.embeddings.weight",
        "transform": "raw",
    }
    m["norm_f/scale"] = {"hf": "backbone.norm_f.weight", "transform": "raw"}
    m["lm_head/kernel"] = {"hf": "lm_head.weight", "transform": "T"}

    layer_types = config.parse_pattern()
    for i, mt in enumerate(layer_types):
        lp = f"layers/{i}"
        hp = f"backbone.layers.{i}"
        # Per-layer pre-norm.
        m[f"{lp}/norm/scale"] = {"hf": f"{hp}.norm.weight", "transform": "raw"}

        if mt == MIXER_MAMBA:
            mp, hm = f"{lp}/mixer", f"{hp}.mixer"
            m[f"{mp}/in_proj/kernel"] = {"hf": f"{hm}.in_proj.weight", "transform": "T"}
            m[f"{mp}/conv1d/kernel"] = {"hf": f"{hm}.conv1d.weight", "transform": "conv"}
            m[f"{mp}/conv1d/bias"] = {"hf": f"{hm}.conv1d.bias", "transform": "raw"}
            m[f"{mp}/dt_bias"] = {"hf": f"{hm}.dt_bias", "transform": "raw"}
            m[f"{mp}/A_log"] = {"hf": f"{hm}.A_log", "transform": "raw"}
            m[f"{mp}/D"] = {"hf": f"{hm}.D", "transform": "raw"}
            m[f"{mp}/norm/scale"] = {"hf": f"{hm}.norm.weight", "transform": "raw"}
            m[f"{mp}/out_proj/kernel"] = {"hf": f"{hm}.out_proj.weight", "transform": "T"}

        elif mt == MIXER_ATTENTION:
            mp, hm = f"{lp}/mixer", f"{hp}.mixer"
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                m[f"{mp}/{proj}/kernel"] = {"hf": f"{hm}.{proj}.weight", "transform": "T"}

        elif mt == MIXER_MOE:
            mp, hm = f"{lp}/mixer", f"{hp}.mixer"
            m[f"{mp}/gate/kernel"] = {"hf": f"{hm}.gate.weight", "transform": "T"}
            m[f"{mp}/e_score_correction_bias"] = {
                "hf": f"{hm}.gate.e_score_correction_bias",
                "transform": "raw",
            }
            m[f"{mp}/routed_W1"] = {
                "hf": [f"{hm}.experts.{j}.up_proj.weight" for j in range(config.n_routed_experts)],
                "transform": "stackT",
            }
            m[f"{mp}/routed_W2"] = {
                "hf": [f"{hm}.experts.{j}.down_proj.weight" for j in range(config.n_routed_experts)],
                "transform": "stackT",
            }
            if config.n_shared_experts > 0:
                m[f"{mp}/shared_W1"] = {
                    "hf": [f"{hm}.shared_experts.up_proj.weight"],
                    "transform": "stackT",
                }
                m[f"{mp}/shared_W2"] = {
                    "hf": [f"{hm}.shared_experts.down_proj.weight"],
                    "transform": "stackT",
                }

    return m
