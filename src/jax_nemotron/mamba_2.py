"""
Minimal Mamba-2 Implementation in JAX/Flax NNX

Based on: "Transformers are SSMs: Generalized Models and Efficient Algorithms
           Through Structured State Space Duality" (Dao & Gu, 2024)

Reference code: https://github.com/state-spaces/mamba
  - mamba_ssm/modules/ssd_minimal.py  (Listing 1 from the paper)
  - mamba_ssm/modules/mamba2_simple.py (Full Mamba-2 block)

This implementation prioritizes simplicity and readability over performance.
"""

import jax
import jax.numpy as jnp
from flax import nnx

# =============================================================================
# Core SSD Algorithm (Listing 1 from the paper)
# =============================================================================


def segsum(x: jax.Array) -> jax.Array:
    """
    Stable segment sum calculation.

    Computes a lower-triangular matrix L where:
      L[i,j] = sum(x[j+1 : i+1])  for i >= j  (sum of x[j+1], x[j+2], ..., x[i])
      L[i,j] = -inf                for i < j   (causal mask → exp gives 0)
      L[i,i] = 0                                (diagonal → exp gives 1, no self-decay)

    This is used to build the decay matrix for the diagonal (intra-chunk) blocks
    of the structured state space model.

    In the SSM context, x contains log-space decay factors (A*dt values), so:
      exp(L[i,j]) = product of A[k]*dt[k] for k in (j, i]
    which gives the total decay from state at position j to output at position i.

    Args:
        x: Decay factors, shape (..., T)
    Returns:
        Lower-triangular segment sums, shape (..., T, T)
    """
    T = x.shape[-1]

    # Step 1: Broadcast x into a T x T matrix by repeating along a new axis.
    # Each row becomes a copy of x. Shape: (..., T, T)
    x = jnp.repeat(x[..., None], T, axis=-1)

    # Step 2: Zero out the upper triangle (excluding diagonal).
    # We only want to accumulate values strictly below the diagonal.
    # mask[i,j] = True if i > j (strict lower triangle)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
    x = jnp.where(mask, x, 0.0)

    # Step 3: Cumulative sum along rows (axis=-2).
    # After this, position (i, j) contains sum of x[k] for k in [j, i).
    x_segsum = jnp.cumsum(x, axis=-2)

    # Step 4: Mask out the upper triangle with -inf.
    # When we later do exp(segsum), -inf -> 0, ensuring causality.
    # We keep the diagonal (k=0) since exp(0) = 1 (no self-decay).
    mask_diag = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    x_segsum = jnp.where(mask_diag, x_segsum, -jnp.inf)

    return x_segsum


def ssd_minimal_discrete(
    X: jax.Array, A: jax.Array, B: jax.Array, C: jax.Array, block_len: int
) -> jax.Array:
    """
    Structured State Space Duality (SSD) algorithm — Listing 1 from the paper.

    This implements the SSM recurrence h[t] = A[t]*h[t-1] + B[t]*x[t] efficiently
    by chunking the sequence and factorizing the computation into:
      1. Intra-chunk outputs  (diagonal blocks  — within each chunk)
      2. Intra-chunk states   (right factor     — B terms)
      3. Inter-chunk states   (middle factor    — A terms, chunk-level recurrence)
      4. State-to-output      (left factor      — C terms)

    The key insight: the SSM output matrix has a semi-separable structure that can
    be decomposed into diagonal blocks (handled by step 1) and off-diagonal blocks
    (handled by steps 2-4 via low-rank factorization).

    Args:
        X: Input values (already multiplied by dt), shape (batch, length, n_heads, headdim)
        A: Discrete decay factors (already multiplied by dt), shape (batch, length, n_heads)
        B: Input-to-state matrix (keys in attention analogy), shape (batch, length, n_heads, d_state)
        C: State-to-output matrix (queries in attention analogy), shape (batch, length, n_heads, d_state)
        block_len: Chunk size Q for splitting the sequence

    Returns:
        Y: Output, shape (batch, length, n_heads, headdim)
    """
    batch, length, n_heads, headdim = X.shape
    d_state = B.shape[-1]

    assert length % block_len == 0, (
        f"Length {length} must be divisible by block_len {block_len}"
    )
    n_chunks = length // block_len

    # Reshape the sequence into chunks: (batch, n_chunks, block_len, ...)
    X = jnp.reshape(X, (batch, n_chunks, block_len, n_heads, headdim))
    A = jnp.reshape(A, (batch, n_chunks, block_len, n_heads))
    B = jnp.reshape(B, (batch, n_chunks, block_len, n_heads, d_state))
    C = jnp.reshape(C, (batch, n_chunks, block_len, n_heads, d_state))

    # Rearrange A to (batch, n_heads, n_chunks, block_len) for segment sums
    A = jnp.transpose(A, (0, 3, 1, 2))

    # Cumulative sum of A within each chunk — used to compute decays
    A_cumsum = jnp.cumsum(A, axis=-1)

    # ---- Step 1: Intra-chunk outputs (diagonal blocks) ----
    # Build the T×T causal decay matrix within each chunk using segsum.
    # L[i,j] = exp(sum of A[k] for k in [j,i)), i.e. the decay from j to i.
    # Then compute: Y_diag = C @ diag(L) @ B^T @ X  (within each chunk)
    L = jnp.exp(segsum(A))
    Y_diag = jnp.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # ---- Step 2: Intra-chunk states (right factor of off-diagonal blocks) ----
    # Compute the SSM state at the END of each chunk by accumulating inputs B*X
    # weighted by their decay to the chunk boundary.
    # decay_states[t] = exp(A_cumsum[-1] - A_cumsum[t]) = decay from t to end of chunk
    decay_states = jnp.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = jnp.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # ---- Step 3: Inter-chunk SSM recurrence (middle factor) ----
    # Propagate states across chunk boundaries using chunk-level decays.
    # Each chunk's output state decays into the next chunk's input state.
    # This is the "recurrence across chunks" part.
    initial_states = jnp.zeros_like(states[:, :1])
    states = jnp.concatenate([initial_states, states], axis=1)

    # Build chunk-level decay matrix using segsum on the total decay per chunk
    # A_cumsum[..., -1] = total decay within each chunk
    # Pad with a leading 0 to account for the initial state
    A_cumsum_last = A_cumsum[..., -1]
    A_cumsum_last_padded = jnp.pad(A_cumsum_last, ((0, 0), (0, 0), (1, 0)))
    decay_chunk = jnp.exp(segsum(A_cumsum_last_padded))

    # Apply chunk-level recurrence: propagate states through all previous chunks
    new_states = jnp.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states = new_states[:, :-1]  # Drop the last (it would be the "next" state)

    # ---- Step 4: State-to-output per chunk (left factor of off-diagonal blocks) ----
    # Convert each chunk's incoming state to per-element outputs.
    # state_decay_out[t] = exp(A_cumsum[t]) = decay from chunk start to position t
    state_decay_out = jnp.exp(A_cumsum)
    Y_off = jnp.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    # ---- Combine intra-chunk and inter-chunk outputs ----
    Y = Y_diag + Y_off
    Y = jnp.reshape(Y, (batch, length, n_heads, headdim))
    return Y


# =============================================================================
# Mamba-2 Block (based on mamba2_simple.py)
# =============================================================================


class Mamba2Block(nnx.Module):
    """
    A single Mamba-2 block.

    Architecture:
        input -> in_proj -> [z (gate), xBC, dt]
                              |
                        causal conv1d + SiLU
                              |
                         split [x, B, C]
                              |
                        SSD(x*dt, A*dt, B, C) + D*x
                              |
                         gate (z) + norm
                              |
                        out_proj -> output

    This mirrors the official Mamba2Simple module but uses Flax NNX instead
    of PyTorch, and calls ssd_minimal_discrete instead of CUDA/Triton kernels.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,  # Random number generator for parameter initialization
        d_model: int,  # Model dimension
        d_state: int = 64,  # SSM state dimension (N in the paper)
        d_conv: int = 4,  # Causal convolution kernel width
        expand: int = 2,  # Expansion factor for inner dimension
        headdim: int = 64,  # Dimension per head (P in the paper)
        ngroups: int = 1,  # Number of groups for B,C (like grouped-query attention)
        chunk_size: int = 64,  # Chunk size for SSD algorithm
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model  # D in the paper
        self.headdim = headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size

        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim  # H in the paper
        assert self.nheads % self.ngroups == 0, "nheads must be divisible by ngroups"

        # --- Learnable Parameters ---

        # 1. Input projection: projects d_model -> [z, x, B, C, dt] all at once
        #    z:  gating branch,         size = d_inner
        #    xBC: conv input branch,    size = d_inner + 2*ngroups*d_state
        #    dt: step size per head,    size = nheads
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nnx.Linear(self.d_model, d_in_proj, use_bias=False, rngs=rngs)

        # 2. Depthwise causal 1D convolution on [x, B, C]
        #    Each channel is convolved independently (feature_group_count = conv_dim)
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nnx.Conv(
            in_features=conv_dim,
            out_features=conv_dim,
            kernel_size=(d_conv,),
            feature_group_count=conv_dim,  # depthwise: each channel has its own filter
            use_bias=True,
            padding="VALID",  # We handle causal padding manually
            rngs=rngs,
        )

        # 3. dt (step size) bias — added before softplus activation
        self.dt_bias = nnx.Param(jnp.zeros((self.nheads,)))

        # 4. A parameter — log-parameterized so exp(A_log) > 0, then negated for decay
        #    Initialized as log(1), log(2), ..., log(nheads) following the paper
        A = jnp.arange(1, self.nheads + 1, dtype=jnp.float32)
        self.A_log = nnx.Param(jnp.log(A))

        # 5. D parameter — skip connection (like a residual from input to output)
        self.D = nnx.Param(jnp.ones((self.nheads,)))

        # 6. Output normalization + projection.
        #    The official Mamba-2 code uses RMSNormGated with norm_before_gate=False,
        #    meaning: apply the gate (y * silu(z)) first, then normalize.
        #    nnx.RMSNorm matches this: it has no bias term (unlike LayerNorm),
        #    which is consistent with the paper's bias-free design.
        self.norm = nnx.RMSNorm(self.d_inner, rngs=rngs)
        self.out_proj = nnx.Linear(
            self.d_inner, self.d_model, use_bias=False, rngs=rngs
        )

    def __call__(self, u: jax.Array) -> jax.Array:
        """
        Forward pass.

        Args:
            u: Input tensor of shape (batch, seqlen, d_model)
        Returns:
            Output tensor of shape (batch, seqlen, d_model)
        """
        batch, seqlen, _ = u.shape

        if seqlen % self.chunk_size != 0:
            raise ValueError(
                f"seqlen ({seqlen}) must be divisible by chunk_size ({self.chunk_size})"
            )
        # --- 1. Input Projection ---
        # Project input to get all branches in one matrix multiply
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)

        # Split into: z (gate), xBC (conv input), dt (step sizes)
        # Note: jnp.split takes split *indices*, not sizes (unlike torch.split)
        z, xBC, dt = jnp.split(
            zxbcdt,
            [self.d_inner, 2 * self.d_inner + 2 * self.ngroups * self.d_state],
            axis=-1,
        )
        # z:   (batch, seqlen, d_inner)     — gating signal
        # xBC: (batch, seqlen, conv_dim)    — will become x, B, C after conv
        # dt:  (batch, seqlen, nheads)      — per-head step sizes

        # Apply softplus to dt (ensures positive step sizes)
        dt = jax.nn.softplus(dt + self.dt_bias.get_value())  # (batch, seqlen, nheads)

        # --- 2. Causal 1D Convolution ---
        # Pad on the left for causal masking: output at time t only sees t-d_conv+1..t
        xBC_padded = jnp.pad(xBC, ((0, 0), (self.d_conv - 1, 0), (0, 0)))
        xBC = self.conv1d(xBC_padded)
        xBC = jax.nn.silu(xBC)  # SiLU/Swish activation

        # Split conv output into x (values), B (keys), C (queries)
        x, B, C = jnp.split(
            xBC, [self.d_inner, self.d_inner + self.ngroups * self.d_state], axis=-1
        )
        # x: (batch, seqlen, d_inner)
        # B: (batch, seqlen, ngroups * d_state)
        # C: (batch, seqlen, ngroups * d_state)

        # Reshape to expose head structure
        x = jnp.reshape(x, (batch, seqlen, self.nheads, self.headdim))
        B = jnp.reshape(B, (batch, seqlen, self.ngroups, self.d_state))
        C = jnp.reshape(C, (batch, seqlen, self.ngroups, self.d_state))

        # Expand B, C from ngroups to nheads (grouped-query attention analogy)
        # If ngroups=1, each group is shared across all heads
        B = jnp.repeat(
            B, self.nheads // self.ngroups, axis=2
        )  # (batch, seqlen, nheads, d_state)
        C = jnp.repeat(
            C, self.nheads // self.ngroups, axis=2
        )  # (batch, seqlen, nheads, d_state)

        # --- 3. SSD Core Computation ---
        # Continuous-time A: always negative (ensures decay/stability)
        A = -jnp.exp(self.A_log.get_value())  # (nheads,)

        # Discretize: multiply by step size dt
        # This converts continuous A to discrete A_bar = exp(A * dt)
        # (in log space, we just multiply and later exponentiate inside SSD)
        A_discrete = A * dt  # (batch, seqlen, nheads)
        X = x * dt[..., None]  # (batch, seqlen, nheads, headdim) — discretized input

        # Run the chunked SSD algorithm
        y = ssd_minimal_discrete(X, A_discrete, B, C, self.chunk_size)

        # Add D skip connection: D * x (direct input-to-output path)
        y = y + self.D.get_value()[None, None, :, None] * x

        # Flatten heads back to d_inner
        y = jnp.reshape(y, (batch, seqlen, self.d_inner))

        # --- 4. Gating, Normalization, and Output Projection ---
        # Gate: element-wise multiply with SiLU-activated z branch
        # (norm_before_gate=False in the official code: gate first, then norm)
        y = y * jax.nn.silu(z)
        y = self.norm(y)

        out = self.out_proj(y)
        return out
