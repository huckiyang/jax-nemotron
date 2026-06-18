"""
Minimal Nemotron-style Hybrid Model in JAX/Flax NNX.

Reference inspiration:
"Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid
 Mamba-Transformer Model for Agentic Reasoning"

What this minimal implementation keeps from the paper:
- Hybrid stack: Mamba mixer + Attention mixer (alternating pattern by default)
- MoE after each mixer block
- Sparse top-k MoE routing with shared experts
- Squared-ReLU experts
- RMSNorm + residual pre-norm structure
- No positional embeddings, no dropout, and bias-free linear layers

What is intentionally simplified:
- Tiny default dimensions for local experimentation
- Alternating layer pattern instead of large paper-scale block scheduling
- No distributed/expert-parallel optimization
"""

from dataclasses import dataclass, field

import jax
from flax import nnx

from attention import GroupedQueryAttention
from mamba_2 import Mamba2Block
from moe import SparseMoE

# =============================================================================
# Config
# =============================================================================


def _default_patterns() -> list[tuple[str, int]]:
    return [
        ("mamba_moe", 2),
        ("mamba_attention_moe", 1),
        ("mamba_moe", 2),
        ("mamba_attention_moe", 1),
        ("mamba_moe", 2),
        ("mamba_attention_moe", 1),
        ("mamba_moe", 1),
    ]


@dataclass
class NemotronConfig:
    """
    Config with tiny local defaults.

    Notes:
    - Defaults are intentionally small for easy local runs.
    - Paper-like behavior is preserved as configurable knobs.
    """

    # Token/model sizes
    vocab_size: int = 1000
    d_model: int = 128

    patterns: list[tuple[str, int]] = field(default_factory=_default_patterns)

    # Attention (GQA)
    num_attention_heads: int = 4
    num_kv_heads: int = 1
    attention_head_dim: int = 32

    # Mamba-2 settings
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64
    mamba_ngroups: int = 1
    mamba_chunk_size: int = 64

    # MoE settings
    num_experts: int = 4
    num_shared_experts: int = 1
    top_k: int = 2
    expert_hidden_dim: int = 256
    granularity_factor: int = 1
    scale_top_k_with_granularity: bool = True

    # Normalization and numerical stability
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_preset(cls, preset: str = "tiny") -> "NemotronConfig":
        """
        Builds a config from a named preset.

        Presets:
        - tiny: default local-friendly profile (fallback)
        - paper_close: larger profile that is closer to Nemotron-3-Nano style

        The paper_close preset increases attention heads, expert count, and
        uses top_k=6 routing with stronger granular MoE settings while still
        keeping this implementation simple.
        """
        key = preset.strip().lower()

        if key in ("tiny", "default"):
            return cls(
                patterns=_default_patterns(),
            )

        if key in ("paper_close", "paper-close", "paper"):
            return cls(
                # Bigger model than tiny defaults.
                d_model=2048,
                patterns=[
                    ("mamba_moe", 2),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe", 2),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe", 2),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe", 2),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe", 2),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe", 3),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe", 4),
                ],
                # Closer to paper-style GQA shape choices.
                num_attention_heads=32,
                num_kv_heads=2,
                attention_head_dim=64,
                # Closer to paper-style Mamba settings.
                mamba_d_state=128,
                mamba_d_conv=4,
                mamba_expand=2,
                mamba_headdim=64,
                mamba_ngroups=8,
                mamba_chunk_size=64,
                # Closer to paper-style MoE settings.
                num_experts=64,
                num_shared_experts=2,
                top_k=6,
                expert_hidden_dim=1856,
                granularity_factor=2,
                # Keep exactly 6 activated routed experts (paper behavior).
                scale_top_k_with_granularity=False,
                rms_norm_eps=1e-6,
            )

        raise ValueError(
            f"Unknown preset '{preset}'. Supported presets: tiny, paper_close"
        )

    def validate(self) -> None:
        """Checks shape constraints that must hold for this architecture."""
        assert len(self.patterns) > 0, "patterns cannot be empty"

        # Attention output must map cleanly back to d_model.
        assert self.num_attention_heads * self.attention_head_dim == self.d_model, (
            "d_model must equal num_attention_heads * attention_head_dim"
        )
        assert self.num_attention_heads % self.num_kv_heads == 0, (
            "num_attention_heads must be divisible by num_kv_heads"
        )

        # Mamba internal shape constraints.
        mamba_d_inner = self.mamba_expand * self.d_model
        assert mamba_d_inner % self.mamba_headdim == 0, (
            "(mamba_expand * d_model) must be divisible by mamba_headdim"
        )
        mamba_nheads = mamba_d_inner // self.mamba_headdim
        assert mamba_nheads % self.mamba_ngroups == 0, (
            "Mamba nheads must be divisible by mamba_ngroups"
        )

        # MoE routing constraints.
        assert self.top_k > 0, "top_k must be > 0"
        assert self.top_k <= self.num_experts, "top_k must be <= num_experts"
        assert self.granularity_factor > 0, "granularity_factor must be > 0"

        effective_num_routed_experts = self.num_experts * self.granularity_factor
        if self.scale_top_k_with_granularity:
            effective_top_k = self.top_k * self.granularity_factor
        else:
            effective_top_k = self.top_k
        assert effective_top_k <= effective_num_routed_experts, (
            "effective routed top-k must be <= effective routed experts"
        )


# =============================================================================
# Helper
# =============================================================================


def _build_mamba(config: NemotronConfig, rngs: nnx.Rngs) -> Mamba2Block:
    return Mamba2Block(
        d_model=config.d_model,
        d_state=config.mamba_d_state,
        d_conv=config.mamba_d_conv,
        expand=config.mamba_expand,
        headdim=config.mamba_headdim,
        ngroups=config.mamba_ngroups,
        chunk_size=config.mamba_chunk_size,
        rngs=rngs,
    )


def _build_moe(config: NemotronConfig, rngs: nnx.Rngs) -> SparseMoE:
    return SparseMoE(
        d_model=config.d_model,
        num_experts=config.num_experts,
        num_shared_experts=config.num_shared_experts,
        top_k=config.top_k,
        expert_hidden_dim=config.expert_hidden_dim,
        granularity_factor=config.granularity_factor,
        scale_top_k_with_granularity=config.scale_top_k_with_granularity,
        use_bias=False,
        rngs=rngs,
    )


# =============================================================================
# Mamba MoE Block
# =============================================================================


class MambaMoEBlock(nnx.Module):
    """
    Hybrid Mamba & MoE block.

    Block structure (pre-norm residual):
      x = x + Mamba(RMSNorm(x))
      x = x + MoE(RMSNorm(x))

    This keeps the architecture simple and easy to inspect.
    """

    def __init__(self, rngs: nnx.Rngs, config: NemotronConfig):
        self.norm_mamba = nnx.RMSNorm(config.d_model, rngs=rngs)
        self.norm_moe = nnx.RMSNorm(config.d_model, rngs=rngs)

        # Reuse the already-implemented Mamba-2 block
        self.mamba = _build_mamba(config=config, rngs=rngs)

        # MoE stage after every mixer layer.
        self.moe = _build_moe(config=config, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Mamba residual path.
        x = x + self.mamba(self.norm_mamba(x))

        # MoE residual path.
        return x + self.moe(self.norm_moe(x))


# =============================================================================
# Mamba Attention MoE Block
# =============================================================================


class MambaAttentionMoEBlock(nnx.Module):
    """
    Hybrid Mamba, Attention & MoE block.

    Block structure (pre-norm residual):
      x = x + Mamba(RMSNorm(x))
      x = x + Attention(RMSNorm(x))
      x = x + MoE(RMSNorm(x))

    This keeps the architecture simple and easy to inspect.
    """

    def __init__(self, rngs: nnx.Rngs, config: NemotronConfig):
        self.norm_mamba = nnx.RMSNorm(config.d_model, rngs=rngs)
        self.norm_attention = nnx.RMSNorm(config.d_model, rngs=rngs)
        self.norm_moe = nnx.RMSNorm(config.d_model, rngs=rngs)

        # Reuse the already-implemented Mamba-2 block as the mixer.
        self.mamba = _build_mamba(config=config, rngs=rngs)

        self.attention = GroupedQueryAttention(
            d_model=config.d_model,
            num_query_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.attention_head_dim,
            rngs=rngs,
        )

        # MoE stage after every mixer layer.
        self.moe = _build_moe(config=config, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Mamba residual path.
        x = x + self.mamba(self.norm_mamba(x))

        # Attention residual path.
        x = x + self.attention(self.norm_attention(x))

        # MoE residual path.
        return x + self.moe(self.norm_moe(x))


# =============================================================================
# Nemotron 3 Nano Block
# =============================================================================


class NemotronNanoBlock(nnx.Module):
    """
    Minimal Nemotron-style language model.

    Layout:
      token embedding -> N x NemotronBlock -> RMSNorm -> LM head

        RoPE is applied inside attention blocks; there is no separate learned
        positional embedding table.
    """

    def __init__(self, rngs: nnx.Rngs, config: NemotronConfig):
        config.validate()
        self.config = config

        self.embedding = nnx.Embed(config.vocab_size, config.d_model, rngs=rngs)
        block_factories = {
            "mamba_moe": MambaMoEBlock,
            "mamba_attention_moe": MambaAttentionMoEBlock,
        }

        blocks: list[MambaMoEBlock | MambaAttentionMoEBlock] = []
        for block_type, repeats in config.patterns:
            block_factory = block_factories.get(block_type)
            if block_factory is None:
                raise ValueError(f"Unknown block type '{block_type}'")
            for _ in range(repeats):
                blocks.append(block_factory(config=config, rngs=rngs))
        self.blocks = nnx.List(blocks)

        self.final_norm = nnx.RMSNorm(config.d_model, rngs=rngs)

        # Untied output head (separate from embeddings).
        self.lm_head = nnx.Linear(
            config.d_model,
            config.vocab_size,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, token_ids: jax.Array) -> jax.Array:
        x = self.embedding(token_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits
