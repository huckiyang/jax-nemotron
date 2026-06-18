"""
NemotronHConfig — the authoritative configuration for the Nemotron-H LLM backbone.

This file holds EVERY real dimension of NVIDIA Nemotron-3-Nano-Omni-30B-A3B's
language-model backbone (model_type ``nemotron_h``), taken verbatim from the
checkpoint's ``config.json`` -> ``llm_config``. It is the single source of truth
for shapes that the Flax/NNX model (``nemotron_h.py``) and the later HF->Orbax
converter both consume.

Design goals (Karpathy-style):
  * One concept per file. This file is config only — no model code.
  * Heavy docstrings, explicit shape relationships spelled out.
  * Two presets that share the SAME field names:
        - "omni_30b": the real 30B model (52 layers, hidden 2688, 128 experts).
        - "tiny":     a CPU-runnable shrink that still exercises one of each
                      mixer type (Mamba2 / attention / MoE) so the shape gate
                      catches structural bugs without a TPU.
  * A ``parse_pattern()`` that turns the HF ``hybrid_override_pattern`` string
    into an explicit per-layer list of mixer types.
  * A ``validate()`` with descriptive shape asserts (the #1 way silent bugs are
    caught before a multi-hour conversion run).

The hybrid backbone is a FLAT stack of ``num_hidden_layers`` layers. Each layer
is exactly ONE mixer, chosen per-position by ``hybrid_override_pattern``:
    'M' -> Mamba2 mixer
    '*' -> GQA attention mixer
    'E' -> MoE / MLP mixer
(This differs from the educational reference repo, which bundled mamba+moe per
block. We match HF: one mixer per layer.)
"""

from __future__ import annotations

from dataclasses import dataclass, field


# =============================================================================
# Mixer type tokens (the alphabet of hybrid_override_pattern)
# =============================================================================

MIXER_MAMBA = "mamba"  # 'M' in the HF pattern string
MIXER_ATTENTION = "attention"  # '*' in the HF pattern string
MIXER_MOE = "moe"  # 'E' in the HF pattern string

# HF single-character code -> our descriptive mixer-type token.
# NOTE: HF also defines '-' for a plain (dense) MLP, but the real
# Nemotron-3-Nano-Omni checkpoint contains only M / * / E, so '-' is mapped to
# the MoE/MLP family slot and would need a dense-MLP path if it ever appeared.
_PATTERN_CHAR_TO_MIXER = {
    "M": MIXER_MAMBA,
    "*": MIXER_ATTENTION,
    "E": MIXER_MOE,
}


# =============================================================================
# Config dataclass
# =============================================================================


@dataclass
class NemotronHConfig:
    """
    Configuration for the Nemotron-H LLM backbone.

    All defaults below are the REAL 30B values (i.e. an unmodified
    ``NemotronHConfig()`` is the production model). Use ``from_preset("tiny")``
    for a CPU-runnable shrink.

    Field grouping mirrors HF ``config.json`` -> ``llm_config`` so a reader can
    diff the two side by side.
    """

    # ----- Token / model sizes -------------------------------------------------
    vocab_size: int = 131072
    hidden_size: int = 2688  # d_model
    num_hidden_layers: int = 52

    # The per-position layer schedule. 'M'=Mamba2, '*'=attention, 'E'=MoE.
    # len(hybrid_override_pattern) MUST equal num_hidden_layers.
    hybrid_override_pattern: str = (
        "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    )

    # ----- Normalization -------------------------------------------------------
    # Nemotron-H RMSNorm is PLAIN: out = weight * (x / sqrt(mean(x^2)+eps)).
    # NO (1 + weight) offset. Verified in modeling_nemotron_h.py line 720.
    norm_eps: float = 1e-5

    # ----- Attention (GQA) mixer ('*') -----------------------------------------
    num_attention_heads: int = 32  # query heads
    num_key_value_heads: int = 2  # KV heads (GQA)
    head_dim: int = 128
    attention_bias: bool = False
    rope_theta: float = 10000.0
    # 1.0 => full rotary over the whole head_dim (no partial rotary).
    partial_rotary_factor: float = 1.0

    # ----- Mamba2 mixer ('M') --------------------------------------------------
    # CRITICAL: HF defines the Mamba inner width as
    #   mamba_intermediate_size = mamba_num_heads * mamba_head_dim
    # which for the real model is 64*64 = 4096. This is NOT expand*hidden_size
    # (= 2*2688 = 5376). The vendored Mamba2Block assumes d_inner=expand*d_model,
    # so our model wraps it with an explicit d_inner override (see nemotron_h.py).
    ssm_state_size: int = 128  # N (state dim per group)
    mamba_num_heads: int = 64  # H
    mamba_head_dim: int = 64  # P (per-head channel width)
    mamba_n_groups: int = 8  # B/C are shared across heads in this many groups
    conv_kernel: int = 4  # depthwise causal conv width
    mamba_expand: int = 2  # kept for reference; NOT used to size d_inner here
    chunk_size: int = 128  # SSD chunk length; MUST divide the test seq len
    use_conv_bias: bool = True  # the ONLY bias in the Mamba mixer

    # ----- MoE / MLP mixer ('E') -----------------------------------------------
    n_routed_experts: int = 128
    num_experts_per_tok: int = 6  # top-k
    n_shared_experts: int = 1
    moe_intermediate_size: int = 1856  # routed expert hidden dim
    moe_shared_expert_intermediate_size: int = 3712  # shared expert hidden dim
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    # MoE expert grouping (DeepSeek-style group-limited routing). The real model
    # uses a single group (n_group=1, topk_group=1) which is a plain global top-k.
    n_group: int = 1
    topk_group: int = 1

    # Plain dense-MLP hidden size (used only if a '-' layer ever appears; the
    # real checkpoint has none). Kept for completeness / future-proofing.
    intermediate_size: int = 1856

    # ----- Misc ----------------------------------------------------------------
    tie_word_embeddings: bool = False

    # =========================================================================
    # Derived shape helpers (computed, not stored)
    # =========================================================================

    @property
    def mamba_intermediate_size(self) -> int:
        """Mamba inner width d_inner = num_heads * head_dim (= 4096 for 30B)."""
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def mamba_conv_dim(self) -> int:
        """Depthwise-conv channel count: d_inner + 2 * n_groups * ssm_state_size."""
        return (
            self.mamba_intermediate_size
            + 2 * self.mamba_n_groups * self.ssm_state_size
        )

    @property
    def mamba_in_proj_dim(self) -> int:
        """
        Mamba in_proj output width.

        HF (non-cuda path) splits in_proj into [gate, xBC, dt]:
            gate (z) : mamba_intermediate_size
            xBC      : mamba_conv_dim
            dt       : mamba_num_heads
        so total = d_inner + conv_dim + num_heads.

        NOTE: the cuda_kernels_forward path additionally allows two leading MLP
        chunks (2*d_mlp), but for this checkpoint d_mlp=0, so there is no MLP
        branch and this width matches the stored in_proj.weight exactly.
        """
        return (
            self.mamba_intermediate_size
            + self.mamba_conv_dim
            + self.mamba_num_heads
        )

    @property
    def attention_q_dim(self) -> int:
        """Total query projection width = num_attention_heads * head_dim."""
        return self.num_attention_heads * self.head_dim

    @property
    def attention_kv_dim(self) -> int:
        """Total K (or V) projection width = num_key_value_heads * head_dim."""
        return self.num_key_value_heads * self.head_dim

    @property
    def rotary_dim(self) -> int:
        """Number of head channels that RoPE rotates (full head_dim when factor=1)."""
        return int(self.head_dim * self.partial_rotary_factor)

    # =========================================================================
    # Pattern parsing
    # =========================================================================

    def parse_pattern(self) -> list[str]:
        """
        Expand ``hybrid_override_pattern`` into a per-layer list of mixer types.

        Returns a list of length ``num_hidden_layers`` where each entry is one
        of ``MIXER_MAMBA`` / ``MIXER_ATTENTION`` / ``MIXER_MOE``.

        Raises:
            ValueError: if the pattern length disagrees with num_hidden_layers,
                        or if it contains an unknown character.
        """
        pattern = self.hybrid_override_pattern
        if len(pattern) != self.num_hidden_layers:
            raise ValueError(
                f"hybrid_override_pattern length {len(pattern)} != "
                f"num_hidden_layers {self.num_hidden_layers}. "
                f"pattern={pattern!r}"
            )
        layer_types: list[str] = []
        for i, ch in enumerate(pattern):
            mixer = _PATTERN_CHAR_TO_MIXER.get(ch)
            if mixer is None:
                raise ValueError(
                    f"Unknown mixer char {ch!r} at position {i} in "
                    f"hybrid_override_pattern. Known chars: "
                    f"{sorted(_PATTERN_CHAR_TO_MIXER)}"
                )
            layer_types.append(mixer)
        return layer_types

    def mixer_layer_indices(self, mixer: str) -> list[int]:
        """Return the layer indices whose mixer equals ``mixer`` (debug helper)."""
        return [i for i, m in enumerate(self.parse_pattern()) if m == mixer]

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self) -> None:
        """
        Assert every shape constraint the architecture relies on, with messages
        that name the offending fields. Cheap to call; call it in model __init__.
        """
        # --- Pattern <-> depth ---
        parsed = self.parse_pattern()  # also checks length + chars
        assert len(parsed) == self.num_hidden_layers, (
            "parse_pattern() length must equal num_hidden_layers"
        )

        # --- Attention (GQA) ---
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"num_attention_heads ({self.num_attention_heads}) must be divisible "
            f"by num_key_value_heads ({self.num_key_value_heads}) for GQA"
        )
        assert self.attention_q_dim > 0, "attention_q_dim must be > 0"
        # head_dim need NOT equal hidden_size/num_heads in Nemotron-H: q_proj maps
        # hidden_size -> num_attention_heads*head_dim, and o_proj maps back. We do
        # NOT assert num_attention_heads*head_dim == hidden_size (it is 4096 != 2688).
        assert 0.0 < self.partial_rotary_factor <= 1.0, (
            f"partial_rotary_factor must be in (0,1], got {self.partial_rotary_factor}"
        )
        assert self.rotary_dim % 2 == 0, (
            f"rotary_dim ({self.rotary_dim}) must be even for half-split RoPE"
        )

        # --- Mamba2 ---
        assert self.mamba_intermediate_size == self.mamba_num_heads * self.mamba_head_dim, (
            "mamba_intermediate_size must equal mamba_num_heads * mamba_head_dim"
        )
        assert self.mamba_num_heads % self.mamba_n_groups == 0, (
            f"mamba_num_heads ({self.mamba_num_heads}) must be divisible by "
            f"mamba_n_groups ({self.mamba_n_groups})"
        )
        assert self.conv_kernel >= 1, "conv_kernel must be >= 1"
        assert self.chunk_size >= 1, "chunk_size must be >= 1"

        # --- MoE ---
        assert self.n_routed_experts > 0, "n_routed_experts must be > 0"
        assert 0 < self.num_experts_per_tok <= self.n_routed_experts, (
            f"num_experts_per_tok ({self.num_experts_per_tok}) must be in "
            f"(0, n_routed_experts={self.n_routed_experts}]"
        )
        assert self.n_shared_experts >= 0, "n_shared_experts must be >= 0"
        assert self.moe_intermediate_size > 0, "moe_intermediate_size must be > 0"
        assert self.n_group >= 1 and self.topk_group >= 1, (
            "n_group and topk_group must be >= 1"
        )
        assert self.n_routed_experts % self.n_group == 0, (
            f"n_routed_experts ({self.n_routed_experts}) must be divisible by "
            f"n_group ({self.n_group})"
        )

        # --- General ---
        assert self.vocab_size > 0 and self.hidden_size > 0, (
            "vocab_size and hidden_size must be > 0"
        )

    # =========================================================================
    # Presets
    # =========================================================================

    @classmethod
    def from_preset(cls, preset: str = "omni_30b") -> "NemotronHConfig":
        """
        Build a config from a named preset.

        Presets (both share the SAME field names — only the values shrink):
          * "omni_30b": the real production model. 52 layers, hidden 2688,
                        128 routed experts, full hybrid schedule. Big-mem/TPU.
          * "tiny":     a CPU-runnable shrink. 6 layers covering at least one
                        Mamba ('M'), one attention ('*'), and one MoE ('E')
                        layer, with small hidden/heads/experts and a chunk_size
                        that divides the shape-gate test sequence length (8).
        """
        key = preset.strip().lower()

        if key in ("omni_30b", "omni-30b", "omni", "30b", "real"):
            # All defaults are already the real values.
            cfg = cls()
            cfg.validate()
            return cfg

        if key in ("tiny", "test", "cpu"):
            # A 6-layer pattern: M E M * E M.
            #   index 0: M (mamba)   1: E (moe)   2: M (mamba)
            #   index 3: * (attn)    4: E (moe)   5: M (mamba)
            # Covers all three mixer types. chunk_size=4 divides seq_len 8.
            cfg = cls(
                vocab_size=512,
                hidden_size=64,
                num_hidden_layers=6,
                hybrid_override_pattern="MEM*EM",
                norm_eps=1e-5,
                # Attention: 4 query heads, 2 kv heads (GQA 2:1), head_dim 16.
                # q_dim = 4*16 = 64; kv_dim = 2*16 = 32. head_dim != hidden/heads
                # on purpose (mirrors the real model's 32*128=4096 != 2688).
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                attention_bias=False,
                rope_theta=10000.0,
                partial_rotary_factor=1.0,
                # Mamba: 4 heads * 8 head_dim = 32 inner; 2 groups; state 16.
                ssm_state_size=16,
                mamba_num_heads=4,
                mamba_head_dim=8,
                mamba_n_groups=2,
                conv_kernel=4,
                mamba_expand=2,
                chunk_size=4,
                use_conv_bias=True,
                # MoE: 8 routed experts, top-2, 1 shared expert.
                n_routed_experts=8,
                num_experts_per_tok=2,
                n_shared_experts=1,
                moe_intermediate_size=32,
                moe_shared_expert_intermediate_size=48,
                routed_scaling_factor=2.5,
                norm_topk_prob=True,
                n_group=1,
                topk_group=1,
                intermediate_size=32,
                tie_word_embeddings=False,
            )
            cfg.validate()
            return cfg

        raise ValueError(
            f"Unknown preset {preset!r}. Supported: 'omni_30b', 'tiny'."
        )
