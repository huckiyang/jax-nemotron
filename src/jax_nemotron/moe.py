"""
Minimal Sparse Mixture-of-Experts (MoE) in JAX/Flax NNX.

Based on Nemotron 3 Nano (arXiv:2512.20848, §2.1).

Key design choices from the paper:

1. Granular routed experts (DeepSeekMoE style, Dai et al. 2024):
   Each "base" expert is split into finer-grained smaller experts.
   Total routed experts = num_experts * granularity_factor,
   each with hidden_dim = expert_hidden_dim / granularity_factor.
   This improves expert specialization without changing total parameter count.

2. Shared experts:
   Always-on FFN experts that run for every token, unconditionally.
   They provide a stable shared capacity outside of routing competition.
   Shared experts keep the full expert_hidden_dim.

3. Sigmoid gating (Nemotron-specific, unlike most MoE models that use softmax):
   Gate scores are produced independently per expert via sigmoid.
   This means experts do NOT compete with each other for probability mass.
   After top-k selection, selected scores are renormalized to sum to 1
   so that the combined output has a stable scale.

4. Squared-ReLU activation inside each expert FFN:
   relu(x)^2 — a stronger nonlinearity than plain ReLU.

5. Aux-loss-free load balancing (Wang et al. 2024, as used in Nemotron 3 Nano §2.4):
   Instead of an auxiliary gradient loss, each expert gets a learnable bias term.
   The bias is added to routing scores at selection time, nudging the router toward
   under-utilized experts — WITHOUT affecting the actual output gate weights.
   After each training step, biases are updated with a simple sign rule:
     if expert i got too many tokens → decrease its bias (harder to pick next time)
     if expert i got too few tokens  → increase its bias (easier to pick next time)
   The update rate is 1e-3 per the paper. This produces no extra gradient computation.

6. No bias on any linear layers (per paper).

The implementation is explicit and loop-based for readability, not performance.
"""

import jax
import jax.numpy as jnp
from flax import nnx

# =============================================================================
# Sparse MoE Block
# =============================================================================


class SparseMoE(nnx.Module):
    """
    Sparse MoE layer matching the Nemotron 3 Nano design (arXiv:2512.20848, §2.1).

    --- Granular experts (DeepSeekMoE style) ---
    Instead of a few large experts, we use many small, fine-grained experts.
    Given `num_experts` base experts and a `granularity_factor` g:
      - Actual routed expert count = num_experts * g
      - Each routed expert hidden dim = expert_hidden_dim / g
      - Top-k = top_k * g   (if scale_top_k_with_granularity=True)
    Total FLOPs per token stays the same, but with more diverse expert paths.
    In Nemotron 3 Nano: 128 total routable experts, 6 activated per token.

    --- Shared experts ---
    `num_shared_experts` always-on FFN experts run on every token.
    They are NOT subject to routing — they always contribute to the output.
    Their outputs are summed and added to the routed path output.
    Shared experts keep the full expert_hidden_dim (not reduced by granularity).
    In Nemotron 3 Nano: 2 shared experts.

    --- Sigmoid routing (Nemotron-specific) ---
    Router logits -> sigmoid -> top-k selection.
    With softmax (most MoEs): experts compete; picking one raises another's cost.
    With sigmoid: scores are independent; each expert is scored on its own merit.
    After top-k, selected scores are renormalized to sum to 1 for stable output scale.

    --- Aux-loss-free load balancing ---
    Without any balancing, the router collapses: it always picks a few favourite
    experts and ignores the rest. The fix used in Nemotron 3 Nano is NOT a loss
    term — instead, each expert gets a persistent bias scalar.

    At routing time: top-k uses (sigmoid_score + expert_bias) to decide which
    experts to activate. A higher bias makes an expert easier to pick.

    At output time: gate weights use the ORIGINAL sigmoid scores (without bias),
    so the learned expert magnitudes are preserved.

    After every training step (outside the gradient): biases are nudged with a
    simple sign update. Overloaded experts get a smaller bias; underloaded ones
    get a larger bias. Over time this pushes the router toward balance.

    Args:
        granularity_factor:
            1 = standard MoE (no granularity).
            >1 = each base expert is split into this many smaller experts.
        scale_top_k_with_granularity:
            True (default): effective top-k = top_k * granularity_factor.
            False: keep top-k fixed regardless of granularity.
        bias_update_rate:
            Step size for the expert bias update. 1e-3 per Nemotron 3 Nano §2.4.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        num_experts: int,
        num_shared_experts: int,
        top_k: int,
        expert_hidden_dim: int,
        use_bias: bool = False,
        granularity_factor: int = 1,
        scale_top_k_with_granularity: bool = True,
        bias_update_rate: float = 1e-3,  # Wang et al. 2024 / Nemotron 3 Nano §2.4
    ):
        self.d_model = d_model

        # Keep base values for reference.
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.expert_hidden_dim = expert_hidden_dim

        self.granularity_factor = granularity_factor
        self.scale_top_k_with_granularity = scale_top_k_with_granularity

        assert self.num_experts > 0, "num_experts must be > 0"
        assert self.top_k > 0, "top_k must be > 0"
        assert self.top_k <= self.num_experts, "top_k must be <= num_experts"
        assert self.num_shared_experts >= 0, "num_shared_experts must be >= 0"
        assert self.granularity_factor > 0, "granularity_factor must be > 0"

        # Total fine-grained routed experts after granular splitting.
        # e.g. 16 base experts * 8 granularity_factor = 128 (as in Nemotron 3 Nano).
        self.num_routed_experts = self.num_experts * self.granularity_factor

        # Scale top-k proportionally so the same fraction of capacity is activated.
        # e.g. top_k=1 with granularity=6 → select 6 out of 128 experts.
        if self.scale_top_k_with_granularity:
            self.routed_top_k = self.top_k * self.granularity_factor
        else:
            self.routed_top_k = self.top_k

        assert self.routed_top_k <= self.num_routed_experts, (
            "effective routed top-k must be <= num_routed_experts"
        )

        # Granular routed experts are narrower to keep total parameter count stable.
        # e.g. expert_hidden_dim=1856, granularity=8 → each expert hidden = 232.
        self.routed_expert_hidden_dim = max(
            1, self.expert_hidden_dim // self.granularity_factor
        )

        # Shared experts keep the full hidden dimension.
        # They're meant to model general token features, so they stay large.
        self.shared_expert_hidden_dim = self.expert_hidden_dim

        # Step size used when updating expert biases after each training step.
        self.bias_update_rate = bias_update_rate

        # Expert bias for aux-loss-free load balancing.
        # Shape: (num_routed_experts,), initialized to 0 for all experts equally.
        # NOT updated by the gradient optimizer — updated manually via update_expert_bias().
        # Stored as a plain nnx.Variable (not nnx.Param) so it can be filtered out
        # of gradient updates when extracting model parameters.
        self.expert_bias = nnx.Variable(jnp.zeros(self.num_routed_experts))

        # Stores the top-k expert indices from the most recent forward pass.
        # Shape at runtime: (num_tokens, routed_top_k). Placeholder shape here.
        # After each training step, the training loop reads this to call
        # update_expert_bias(self.last_topk_indices.value) — outside the gradient.
        self.last_topk_indices = nnx.Variable(
            jnp.zeros((1, self.routed_top_k), dtype=jnp.int32)
        )

        # Router: a single linear layer mapping each token to one logit per routed expert.
        # No bias per paper. Shared experts are NOT routed — they bypass this.
        self.router = nnx.Linear(
            self.d_model,
            self.num_routed_experts,
            use_bias=use_bias,
            rngs=rngs,
        )

        # All routed expert weights are stored pre-stacked so _collect_routed_outputs
        # can index directly — no jnp.stack() assembly on every forward pass.
        # routed_W1: (num_routed_experts, d_model, routed_expert_hidden_dim)
        # routed_W2: (num_routed_experts, routed_expert_hidden_dim, d_model)
        # No bias on any linear layers, per paper.
        init = nnx.initializers.lecun_normal()
        self.routed_W1 = nnx.Param(
            init(
                rngs.params(),
                (self.num_routed_experts, d_model, self.routed_expert_hidden_dim),
            )
        )
        self.routed_W2 = nnx.Param(
            init(
                rngs.params(),
                (self.num_routed_experts, self.routed_expert_hidden_dim, d_model),
            )
        )

        # Shared expert weights are also pre-stacked for the batched einsum in
        # _collect_shared_outputs. Shared experts keep the full hidden dimension.
        # shared_W1: (num_shared_experts, d_model, shared_expert_hidden_dim)
        # shared_W2: (num_shared_experts, shared_expert_hidden_dim, d_model)
        if self.num_shared_experts > 0:
            self.shared_W1 = nnx.Param(
                init(
                    rngs.params(),
                    (self.num_shared_experts, d_model, self.shared_expert_hidden_dim),
                )
            )
            self.shared_W2 = nnx.Param(
                init(
                    rngs.params(),
                    (self.num_shared_experts, self.shared_expert_hidden_dim, d_model),
                )
            )

    def _collect_routed_outputs(
        self,
        x_flat: jax.Array,
        topk_indices: jax.Array,
    ) -> jax.Array:
        """
        Run only the selected routed experts via pre-stacked weight gather + einsum.

        Weights are stored pre-stacked as (num_routed_experts, ...) tensors in
        __init__, so we index directly with topk_indices — no per-forward assembly.
        Only (num_tokens × routed_top_k) expert forward passes are computed;
        non-selected experts are never touched.

        Args:
            x_flat: (num_tokens, d_model)
            topk_indices: (num_tokens, routed_top_k)
        Returns:
            routed_outputs: (num_tokens, routed_top_k, d_model)
                            only the selected expert outputs, in top-k order.
        """
        # Weights are pre-stacked at init time — no assembly cost here.
        # W1: (num_routed_experts, d_model,      routed_expert_hidden_dim)
        # W2: (num_routed_experts, routed_expert_hidden_dim, d_model)
        W1 = self.routed_W1.get_value()
        W2 = self.routed_W2.get_value()

        # Gather only the weights of each token's top-k selected experts.
        # topk_indices: (num_tokens, routed_top_k)  →  index into axis-0 of W1/W2
        # W1_sel: (num_tokens, routed_top_k, d_model,      hidden_dim)
        # W2_sel: (num_tokens, routed_top_k, hidden_dim,   d_model)
        W1_sel = W1[topk_indices]
        W2_sel = W2[topk_indices]

        # fc1: apply each token's K selected up-projections simultaneously.
        # 't d, t k d h -> t k h'  means: for every token t and selected expert k,
        # dot x_flat[t] with W1_sel[t, k] to get a hidden vector of size h.
        h = jnp.einsum("td,tkdh->tkh", x_flat, W1_sel)

        # Squared-ReLU activation: relu(x)^2  (paper §2.1 / MoEExpert.__call__).
        h = jax.nn.relu(h)
        h = h * h

        # fc2: apply each token's K selected down-projections.
        # 't k h, t k h d -> t k d'  projects hidden vectors back to d_model.
        out = jnp.einsum("tkh,tkhd->tkd", h, W2_sel)
        # out: (num_tokens, routed_top_k, d_model)
        return out

    def _collect_shared_outputs(self, x_flat: jax.Array) -> jax.Array:
        """
        Run all shared experts on every token via a single batched einsum.

        Shared experts are always active — no routing decision is made.
        All experts are applied in parallel using pre-stacked weight matrices,
        producing one output per expert per token in a single fused operation.

        Args:
            x_flat: (num_tokens, d_model)
        Returns:
            shared_outputs: (num_tokens, num_shared_experts, d_model),
                            or (num_tokens, 0, d_model) if there are no shared experts.
        """
        if self.num_shared_experts == 0:
            return jnp.zeros((x_flat.shape[0], 0, self.d_model), dtype=x_flat.dtype)

        # All shared experts run in one batched einsum — no sequential loop.
        # 'td,edh->teh': for each of the E shared experts, project every token
        # from d_model up to shared_expert_hidden_dim simultaneously.
        h = jnp.einsum("td,edh->teh", x_flat, self.shared_W1.get_value())
        # Squared-ReLU: relu(x)^2 — same activation as the routed experts.
        h = jax.nn.relu(h)
        h = h * h
        # 'teh,ehd->ted': project all shared expert hidden states back to d_model.
        # out: (num_tokens, num_shared_experts, d_model)
        return jnp.einsum("teh,ehd->ted", h, self.shared_W2.get_value())

    def update_expert_bias(self, topk_indices: jax.Array) -> None:
        """
        Update the expert bias after each training step.

        This is the core of the aux-loss-free load balancing strategy
        (Wang et al. 2024), as used in Nemotron 3 Nano (§2.4).

        The idea is simple:
          - Count how many tokens each expert received in this step.
          - Compare to the ideal uniform count (tokens * top_k / num_experts).
          - Decrease the bias of overloaded experts so they win fewer top-k races.
          - Increase the bias of underloaded experts so they win more top-k races.

        The update uses sign() instead of the actual count difference, so the
        step size is always exactly +/- bias_update_rate regardless of how far
        off-balance the expert is. This keeps the bias values small and stable.

        IMPORTANT: Call this AFTER the optimizer step, outside the gradient tape.
        The expert_bias is NOT a gradient parameter — it must not be passed to
        the optimizer. Filter it out by type when building the optimizer state.

        Args:
            topk_indices: (num_tokens, routed_top_k) from the last forward pass.
                          These are the expert indices that were selected.
        """
        num_tokens = topk_indices.shape[0]

        # Count how many tokens were routed to each expert.
        # one_hot: (num_tokens, routed_top_k, num_routed_experts)
        # sum over (tokens, top_k slots) → (num_routed_experts,)
        actual_count = jnp.sum(
            jax.nn.one_hot(topk_indices, self.num_routed_experts),
            axis=(0, 1),
        )

        # The ideal count if all experts were equally loaded.
        expected_count = num_tokens * self.routed_top_k / self.num_routed_experts

        # sign(actual - expected):
        #   +1 means overloaded  → subtract from bias → harder to pick next time
        #   -1 means underloaded → add to bias        → easier to pick next time
        #    0 means perfect     → no change
        self.expert_bias.set_value(
            self.expert_bias.get_value()
            - self.bias_update_rate * jnp.sign(actual_count - expected_count)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through the sparse MoE layer.

        Routing overview (aux-loss-free approach):
            1. Router produces one sigmoid score per expert, per token.
            2. Expert bias is ADDED to scores for top-k selection only.
               This steers the router toward under-utilized experts.
            3. Top-k selection uses the biased scores to choose which experts run.
            4. Gate weights use the ORIGINAL (unbiased) sigmoid scores, renormalized.
               The bias is a routing hint, not a magnitude signal.
            5. Weighted sum of selected expert outputs forms the routed path.
            6. All shared experts run unconditionally; their outputs are summed in.

        To balance load: call update_expert_bias(topk_indices) after each training step.

        Args:
            x: (batch, seqlen, d_model)

        Returns:
            y: (batch, seqlen, d_model)
            jnp.zeros(()) (optional): only returned when return_aux_loss=True
        """
        batch, seqlen, d_model = x.shape
        assert d_model == self.d_model, "Input d_model does not match MoE config"

        # MoE routing is purely token-wise, so we flatten batch and sequence together.
        num_tokens = batch * seqlen
        x_flat = jnp.reshape(x, (num_tokens, d_model))  # (num_tokens, d_model)

        # ── Routed path ────────────────────────────────────────────────────────

        # Step 1: Compute one routing logit per expert for each token.
        routed_logits = self.router(x_flat)  # (num_tokens, num_routed_experts)

        # Step 2: Apply sigmoid to get independent gate scores.
        # Unlike softmax, sigmoid does NOT create a probability distribution.
        # Each expert's score is judged independently — scores do not compete.
        routed_scores = jax.nn.sigmoid(
            routed_logits
        )  # (num_tokens, num_routed_experts)

        # Step 3 (aux-loss-free): Add the expert bias to scores before top-k selection.
        # The bias is learned over time: underloaded experts accumulate a positive bias
        # (making them easier to pick), overloaded experts accumulate a negative bias
        # (making them harder to pick). This is the key load-balancing mechanism.
        # expert_bias shape: (num_routed_experts,) → broadcasts across all tokens.
        biased_scores = routed_scores + self.expert_bias.get_value()

        # Step 4: Select top-k experts using BIASED scores.
        # We use biased scores here so the selection reflects the desired load balance.
        # topk_indices: (num_tokens, routed_top_k) — which expert indices were chosen
        _, topk_indices = jax.lax.top_k(biased_scores, self.routed_top_k)

        # Save topk_indices so the training loop can call update_expert_bias()
        # AFTER the optimizer step, outside the gradient computation.
        self.last_topk_indices.set_value(topk_indices)

        # Step 5: Build gate weights using the ORIGINAL (unbiased) sigmoid scores.
        # The bias only determines WHO gets selected, not HOW MUCH they contribute.
        # Using original scores preserves the expert's learned signal magnitude.
        # Gather only the top-k scores — no need to build a full (num_tokens, num_routed_experts) tensor.
        token_ids = jnp.arange(num_tokens)[:, None]  # (num_tokens, 1)
        selected_scores = routed_scores[
            token_ids, topk_indices
        ]  # (num_tokens, routed_top_k)

        # Step 6: Renormalize so selected gate weights sum to 1 per token.
        # This keeps the output scale stable regardless of the absolute score values.
        selected_gates = selected_scores / (
            jnp.sum(selected_scores, axis=-1, keepdims=True) + 1e-6
        )  # (num_tokens, routed_top_k)

        # Step 7: Run only selected routed experts and compute the gated weighted sum.
        # routed_outputs: (num_tokens, routed_top_k, d_model) — already only top-k, no zeros padding.
        routed_outputs = self._collect_routed_outputs(x_flat, topk_indices)

        # selected_gates[:, :, None] broadcasts to (num_tokens, routed_top_k, d_model).
        routed_mix = jnp.sum(routed_outputs * selected_gates[:, :, None], axis=1)
        # routed_mix: (num_tokens, d_model)

        # ── Shared path ─────────────────────────────────────────────────────────

        if self.num_shared_experts > 0:
            # Shared experts run on every token with no gating or selection.
            shared_outputs = self._collect_shared_outputs(x_flat)
            # shared_outputs: (num_tokens, num_shared_experts, d_model)

            # Sum across shared experts: each expert adds its own contribution.
            shared_mix = jnp.sum(shared_outputs, axis=1)  # (num_tokens, d_model)

            # Final output = routed path + shared path.
            y_flat = routed_mix + shared_mix
        else:
            y_flat = routed_mix

        # Restore the original (batch, seqlen, d_model) shape.
        y = jnp.reshape(y_flat, (batch, seqlen, d_model))

        return y
