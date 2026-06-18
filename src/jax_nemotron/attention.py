"""
Minimal Grouped-Query Attention (GQA) in JAX/Flax NNX.

This file implements a simple, educational attention block used by the
Nemotron 3 Nano hybrid model:
- Causal self-attention (decoder-only masking)
- Grouped-query attention (more query heads than KV heads)
- No positional embeddings
- No dropout
- No bias on linear projections

The implementation prioritizes readability over speed.
"""

import jax
import jax.numpy as jnp
from flax import nnx

# =============================================================================
# Attention Block
# =============================================================================


class GroupedQueryAttention(nnx.Module):
    """
    Minimal causal self-attention with grouped-query heads.

    Nemotron 3 Nano design choices:
    - Query heads and KV heads can be different (GQA)
    - No positional embeddings
    - No dropout
    - No bias on linear projections

    Args:
        d_model: Hidden/model dimension.
        num_query_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head channel dimension.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        assert self.num_query_heads % self.num_kv_heads == 0, (
            "num_query_heads must be divisible by num_kv_heads for GQA"
        )
        assert self.num_query_heads * self.head_dim == self.d_model, (
            "d_model must equal num_query_heads * head_dim"
        )

        # Project input tokens into Q, K, and V spaces.
        self.q_proj = nnx.Linear(
            self.d_model,
            self.num_query_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            self.d_model,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            self.d_model,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
        )

        # Project attention output back to model dimension.
        self.out_proj = nnx.Linear(
            self.num_query_heads * self.head_dim,
            self.d_model,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: Token hidden states, shape (batch, seqlen, d_model)
        Returns:
            Output hidden states, shape (batch, seqlen, d_model)
        """
        batch, seqlen, d_model = x.shape
        assert d_model == self.d_model, "Input d_model does not match module config"

        # 1) Compute Q, K, V projections.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2) Expose head dimensions.
        q = jnp.reshape(q, (batch, seqlen, self.num_query_heads, self.head_dim))
        k = jnp.reshape(k, (batch, seqlen, self.num_kv_heads, self.head_dim))
        v = jnp.reshape(v, (batch, seqlen, self.num_kv_heads, self.head_dim))

        # 3) Expand KV heads so each query head can attend to a matching KV head.
        kv_repeat = self.num_query_heads // self.num_kv_heads
        k = jnp.repeat(k, kv_repeat, axis=2)
        v = jnp.repeat(v, kv_repeat, axis=2)

        # 4) Move heads before sequence for standard attention math.
        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, heads, seqlen, head_dim)
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # 5) Dot-product attention scores.
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

        # 6) Causal mask: position i cannot see future position j > i.
        causal_mask = jnp.tril(jnp.ones((seqlen, seqlen), dtype=bool))
        scores = jnp.where(causal_mask[None, None, :, :], scores, -1e30)

        # 7) Softmax over keys, then weighted sum of values.
        attn = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

        # 8) Merge heads and project to model space.
        context = jnp.transpose(context, (0, 2, 1, 3))
        context = jnp.reshape(
            context, (batch, seqlen, self.num_query_heads * self.head_dim)
        )
        out = self.out_proj(context)
        return out
