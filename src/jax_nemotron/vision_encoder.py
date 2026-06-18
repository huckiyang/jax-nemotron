"""
Vision Encoder for Nemotron 3 Nano Omni in JAX/Flax NNX.

Simplified, educational ViT-style image encoder inspired by the C-RADIOv4-H
encoder used in the paper:
  "Nemotron 3 Nano Omni: Efficient and Open Multimodal Intelligence"
  arXiv:2604.24954

Architecture overview (encoder-projector design):
  image → PatchEmbedding → CPE → N × VisionTransformerBlock
        → PixelShuffle → VisionProjector → [visual tokens for LLM]

Key design choices (matching the paper):
- 16×16 patch embedding
- Conditional Positional Encoding (CPE) via 3×3 depthwise conv on the 2D
  patch grid (no positional embedding table — position is implicit in the
  conv's receptive field)
- Bidirectional (non-causal) attention — every patch attends to all others
- Pixel shuffle 4× spatial downsampling: merges 2×2 patch neighborhoods into
  the channel dim, then projects back — cuts token count by 4
- 2-layer MLP projector bridges encoder hidden dim to LLM d_model

Simplified from real C-RADIO:
- Multi-teacher distillation is now included (configure via VisionEncoderConfig.teachers)
"""

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx


# =============================================================================
# Config
# =============================================================================


@dataclass
class TeacherConfig:
    """
    Describes one teacher model used in multi-teacher distillation.

    C-RADIOv4 distils from three teachers simultaneously:
      - SigLIP2-g-384  (spatial_dim=1152, summary_dim=1152)  — text-image alignment
      - DINOv3-7B      (spatial_dim=1536, summary_dim=1536)  — dense semantic features
      - SAM3           (spatial_dim=1280, summary_dim=0)     — segmentation features
                                                               (no summary/CLS head)

    For local experiments, set spatial_dim / summary_dim to match your student's
    hidden_dim so the adapter heads stay small.

    Args:
        name:        Human-readable identifier, e.g. "siglip2", "dino_v3", "sam3".
        spatial_dim: Dimensionality of the teacher's dense patch feature output.
        summary_dim: Dimensionality of the teacher's global (CLS / summary) feature.
                     Use 0 if the teacher has no summary token (e.g. SAM3).
    """

    name: str           # e.g. "siglip2", "dino_v3", "sam3"
    spatial_dim: int    # teacher's dense spatial feature dimensionality
    summary_dim: int = 0  # teacher's CLS/summary feature dim; 0 = no summary loss


@dataclass
class VisionEncoderConfig:
    """
    Configuration for the vision encoder and its MLP projector.

    Tiny defaults allow local experimentation. Scale hidden_dim and num_layers
    for paper-like quality (paper uses ViT-H scale: hidden_dim=1280, 32 layers).
    """

    image_size: int = 64    # Input image side length (square images assumed)
    patch_size: int = 16    # Each patch covers patch_size × patch_size pixels
    in_channels: int = 3    # RGB images
    hidden_dim: int = 128   # Transformer hidden width
    num_heads: int = 4      # Attention heads (hidden_dim = num_heads × head_dim)
    head_dim: int = 32      # Dimension per attention head
    num_layers: int = 2     # Number of ViT transformer blocks
    mlp_dim: int = 256      # MLP inner width inside each transformer block
    proj_dim: int = 128     # Output dimension — must equal LLM d_model
    teachers: list[TeacherConfig] = field(default_factory=list)
    # ^ List of TeacherConfig entries. Leave empty ([]) for pure inference/LLM use.
    #   Populate to enable multi-teacher distillation during training; one entry
    #   per teacher causes one DistillationHead + a CLS token to be created.


# =============================================================================
# Patch Embedding
# =============================================================================


class PatchEmbedding(nnx.Module):
    """
    Splits an image into non-overlapping 16×16 patches and linearly embeds each.

    For a 64×64 image with patch_size=16:
      → (64/16) × (64/16) = 4×4 = 16 patches
      → Each patch is flattened: 16 × 16 × 3 = 768 values
      → A linear layer maps 768 → hidden_dim

    This is the standard ViT tokenization step — converting pixel grids into
    a sequence of patch vectors the transformer can process.

    Args:
        config: VisionEncoderConfig with patch_size, in_channels, hidden_dim.
    """

    def __init__(self, config: VisionEncoderConfig, rngs: nnx.Rngs):
        patch_flat_dim = config.patch_size * config.patch_size * config.in_channels
        self.proj = nnx.Linear(
            patch_flat_dim, config.hidden_dim, use_bias=False, rngs=rngs
        )
        self.patch_size = config.patch_size

    def __call__(self, x: jax.Array) -> tuple[jax.Array, int, int]:
        """
        Args:
            x: (batch, H, W, C) float image
        Returns:
            tokens:    (batch, h*w, hidden_dim) — flat sequence of patch embeddings
            h_patches: number of patch rows (H // patch_size)
            w_patches: number of patch cols (W // patch_size)
        """
        batch, H, W, C = x.shape
        p = self.patch_size
        h_patches, w_patches = H // p, W // p

        # Reshape image into a 2D grid of patches.
        # (B, H, W, C) → (B, h, p, w, p, C)
        x = x.reshape(batch, h_patches, p, w_patches, p, C)

        # Move patch pixel dims together: (B, h, w, p, p, C)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))

        # Flatten each patch into a vector: (B, h*w, p*p*C)
        x = x.reshape(batch, h_patches * w_patches, p * p * C)

        # Linear projection: each flattened patch → hidden_dim
        tokens = self.proj(x)  # (batch, N, hidden_dim)
        return tokens, h_patches, w_patches


# =============================================================================
# Conditional Positional Encoding (CPE)
# =============================================================================


class ConditionalPositionalEncoding(nnx.Module):
    """
    Conditional Positional Encoding via a 2D depthwise convolution.

    Why CPE instead of a learned table?
    A fixed positional embedding table maps each position index to a vector.
    CPE instead runs a small 3×3 depthwise conv over the 2D spatial token grid.
    Because the conv is local (sees each token's neighborhood), the network
    implicitly learns to encode where each token is — without any explicit
    position index. This naturally generalises to different image sizes.

    Implementation:
      1. Reshape flat sequence → 2D spatial grid: (B, N, D) → (B, H, W, D)
      2. 3×3 depthwise conv: each channel kernel sees its local neighborhood
      3. Add result as residual (preserves the original patch content)
      4. Reshape back to (B, N, D)

    Args:
        hidden_dim: Token channel width D.
    """

    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        # Depthwise: feature_group_count=D means in_features/group=1, so each
        # of the D channels has its own independent 3×3 kernel.
        self.dw_conv = nnx.Conv(
            in_features=hidden_dim,
            out_features=hidden_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),   # same padding — spatial size unchanged
            feature_group_count=hidden_dim,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, tokens: jax.Array, h: int, w: int) -> jax.Array:
        """
        Args:
            tokens: (batch, h*w, D)
            h, w:   spatial grid dimensions
        Returns:
            tokens with positional context added, same shape (batch, h*w, D)
        """
        batch, N, D = tokens.shape

        # Restore 2D spatial layout for the conv.
        grid = tokens.reshape(batch, h, w, D)

        # Depthwise conv over the spatial grid — encodes local position context.
        pos_delta = self.dw_conv(grid)  # (batch, h, w, D)

        # Residual addition: merge position hint with original patch content.
        grid = grid + pos_delta

        # Flatten back to sequence.
        return grid.reshape(batch, N, D)


# =============================================================================
# Multi-Teacher Distillation
# =============================================================================


def phi_s_normalize(features: jax.Array) -> jax.Array:
    """
    PHI-S normalization (arXiv:2410.01680 — "PHI-S: Distribution Balancing for
    Label-Free Multi-Teacher Distillation").

    Problem: different teachers produce features at very different magnitudes.
    Without normalization the teacher with the largest activations dominates the
    distillation loss and the student ignores the others.

    Fix: standardize each token's feature vector to zero mean and unit standard
    deviation across the channel dimension D. After normalization all teachers'
    features live on the same scale, so their loss terms contribute equally.

    Args:
        features: (B, N, D) — dense spatial features from one teacher
    Returns:
        (B, N, D) — each token normalized: mean=0, std=1 across D
    """
    mean = jnp.mean(features, axis=-1, keepdims=True)
    std = jnp.std(features, axis=-1, keepdims=True) + 1e-6  # avoid div-by-zero
    return (features - mean) / std


class DistillationHead(nnx.Module):
    """
    Per-teacher adapter: projects student features into one teacher's feature space.

    Why a separate head per teacher?
    SigLIP2, DINOv3, and SAM3 all have different output dimensionalities and
    different learned "vocabularies". A single linear projection can bridge two
    spaces — like a bilingual dictionary — so each head learns the translation
    specific to its teacher.

    Two projections (both are single linear layers, no activation):
      spatial_proj  — (B, N, D_student) → (B, N, D_teacher_spatial)
                      used with the spatial/dense MSE loss (after PHI-S)
      summary_proj  — (B, D_student)    → (B, D_teacher_summary)
                      used with the cosine summary loss
                      created only if teacher_config.summary_dim > 0

    Training pseudocode:
        student_pred = head(student_spatial, student_summary)
        teacher_target = run_frozen_teacher(images)          # detach gradients!
        loss += MSE( phi_s(student_pred.spatial),
                     phi_s(teacher_target.spatial) )

    Args:
        student_dim:    Hidden width of the student VisionEncoder (hidden_dim).
        teacher_config: TeacherConfig specifying name and target dimensions.
    """

    def __init__(self, student_dim: int, teacher_config: TeacherConfig, rngs: nnx.Rngs):
        self.name = teacher_config.name
        # Dense feature adapter: maps patch tokens to teacher's spatial feature space.
        self.spatial_proj = nnx.Linear(
            student_dim, teacher_config.spatial_dim, use_bias=False, rngs=rngs
        )
        # Global feature adapter: maps CLS token to teacher's summary space (optional).
        self.has_summary = teacher_config.summary_dim > 0
        if self.has_summary:
            self.summary_proj = nnx.Linear(
                student_dim, teacher_config.summary_dim, use_bias=False, rngs=rngs
            )

    def __call__(
        self,
        spatial_tokens: jax.Array,
        summary_token: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array | None]:
        """
        Args:
            spatial_tokens: (B, N, student_dim) — patch features from the student
            summary_token:  (B, student_dim)    — CLS token output; required when
                                                  has_summary=True
        Returns:
            spatial_pred:  (B, N, teacher_spatial_dim)
            summary_pred:  (B, teacher_summary_dim) or None
        """
        spatial_pred = self.spatial_proj(spatial_tokens)
        summary_pred = (
            self.summary_proj(summary_token)
            if (self.has_summary and summary_token is not None)
            else None
        )
        return spatial_pred, summary_pred


def compute_distillation_loss(
    teacher_preds: dict,
    teacher_targets: dict,
) -> dict:
    """
    Compute per-teacher distillation losses.

    Two loss components per teacher:

    1. Spatial loss — MSE in PHI-S normalized space:
           L_spatial = mean( (phi_s(student_pred) - phi_s(teacher_target))^2 )
       PHI-S ensures high-magnitude teachers (SAM3) don't swamp low-magnitude
       ones (SigLIP2).  Applied to dense per-patch feature maps.

    2. Summary loss — cosine distance:
           L_summary = mean( 1 - cos_similarity(student_pred, teacher_target) )
       Scale-invariant; used for teachers that produce a global (CLS) summary
       token such as SigLIP2 (text alignment) and DINOv3 (kNN classification).
       Value is 0 when perfectly aligned, 2 when completely opposite.

    Total training loss = sum over all teachers of (L_spatial + L_summary).

    Args:
        teacher_preds:   {name: {"spatial": (B,N,D_t), "summary": (B,D_s)|None}}
                         Student adapter predictions from DistillationHead.
        teacher_targets: {name: {"spatial": (B,N,D_t), "summary": (B,D_s)|None}}
                         Frozen teacher model outputs (must be detached from grad).
    Returns:
        {teacher_name: scalar JAX array} — one combined loss value per teacher
    """
    losses = {}
    for name, pred in teacher_preds.items():
        target = teacher_targets[name]
        loss = jnp.array(0.0)

        # --- Spatial distillation: MSE in PHI-S normalized space ---
        if pred["spatial"] is not None and target["spatial"] is not None:
            p_sp = phi_s_normalize(pred["spatial"])    # (B, N, D_t)
            t_sp = phi_s_normalize(target["spatial"])  # (B, N, D_t)
            loss = loss + jnp.mean((p_sp - t_sp) ** 2)

        # --- Summary distillation: cosine distance ---
        if pred["summary"] is not None and target["summary"] is not None:
            # L2-normalize to unit sphere, then cos_sim = dot product.
            p_su = pred["summary"] / (
                jnp.linalg.norm(pred["summary"], axis=-1, keepdims=True) + 1e-8
            )
            t_su = target["summary"] / (
                jnp.linalg.norm(target["summary"], axis=-1, keepdims=True) + 1e-8
            )
            cos_sim = jnp.sum(p_su * t_su, axis=-1)  # (B,)
            loss = loss + jnp.mean(1.0 - cos_sim)     # cosine distance, averaged

        losses[name] = loss
    return losses


# =============================================================================
# Vision Multi-Head Attention (bidirectional, no causal mask)
# =============================================================================


class VisionAttention(nnx.Module):
    """
    Standard multi-head self-attention for vision.

    Unlike text decoder attention, image patches have no causal ordering —
    patch 7 can freely attend to patch 2 or patch 15. So there is no
    lower-triangular causal mask here.

    Args:
        config: VisionEncoderConfig
    """

    def __init__(self, config: VisionEncoderConfig, rngs: nnx.Rngs):
        D = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.q_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: (batch, N, hidden_dim) — N patches
        Returns:
            (batch, N, hidden_dim)
        """
        batch, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        # Project to Q, K, V and expose head dimension.
        q = jnp.transpose(self.q_proj(x).reshape(batch, N, H, d), (0, 2, 1, 3))
        k = jnp.transpose(self.k_proj(x).reshape(batch, N, H, d), (0, 2, 1, 3))
        v = jnp.transpose(self.v_proj(x).reshape(batch, N, H, d), (0, 2, 1, 3))
        # Each: (batch, H, N, d)

        # Scaled dot-product attention — no causal mask for image patches.
        scale = 1.0 / math.sqrt(d)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale  # (B, H, N, N)
        attn = jax.nn.softmax(scores, axis=-1)

        # Weighted sum over values.
        context = jnp.einsum("bhqk,bhkd->bhqd", attn, v)  # (B, H, N, d)

        # Merge heads and project back to D.
        context = jnp.transpose(context, (0, 2, 1, 3)).reshape(batch, N, D)
        return self.out_proj(context)


# =============================================================================
# Vision Transformer Block
# =============================================================================


class VisionTransformerBlock(nnx.Module):
    """
    One ViT transformer block using pre-norm residuals.

    Structure:
      x = x + Attention(RMSNorm(x))    ← multi-head self-attention
      x = x + MLP(RMSNorm(x))          ← feed-forward network

    Pre-norm (normalize before the sublayer) gives more stable gradients
    than the original post-norm design. This matches the LLM backbone style.

    Args:
        config: VisionEncoderConfig
    """

    def __init__(self, config: VisionEncoderConfig, rngs: nnx.Rngs):
        D, M = config.hidden_dim, config.mlp_dim
        self.norm_attn = nnx.RMSNorm(D, rngs=rngs)
        self.norm_mlp = nnx.RMSNorm(D, rngs=rngs)
        self.attn = VisionAttention(config, rngs=rngs)

        # 2-layer MLP: expand to mlp_dim with SiLU, then contract back.
        self.mlp_fc1 = nnx.Linear(D, M, use_bias=False, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(M, D, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Attention path.
        x = x + self.attn(self.norm_attn(x))

        # MLP path: Linear → SiLU → Linear.
        h = self.mlp_fc2(jax.nn.silu(self.mlp_fc1(self.norm_mlp(x))))
        x = x + h

        return x


# =============================================================================
# Pixel Shuffle (4× spatial downsampling)
# =============================================================================


class PixelShuffle(nnx.Module):
    """
    Reduces spatial token count by 4× by merging 2×2 patch neighborhoods.

    Pixel shuffle (space-to-depth): fold a 2×2 block of spatial tokens into
    the channel dimension, doubling the channel width twice (×4 total).
    A linear layer then projects back to the original hidden_dim.

    Why? The paper applies "pixel shuffle with 4× downsampling" after the
    visual encoder to reduce the number of visual tokens fed to the LLM —
    keeping the sequence length manageable while preserving information.

    Process:
      (B, h×w, D)
        → reshape to spatial grid  (B, h, w, D)
        → fold 2×2 blocks          (B, h//2, w//2, 4D)
        → linear project           (B, h//2, w//2, D)
        → flatten                  (B, (h//2)*(w//2), D)

    Constraint: h and w must be even (patch grid must be divisible by 2).

    Args:
        hidden_dim: Token channel width D.
    """

    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        # Projects merged 4D back down to D.
        self.proj = nnx.Linear(4 * hidden_dim, hidden_dim, use_bias=False, rngs=rngs)

    def __call__(self, tokens: jax.Array, h: int, w: int) -> jax.Array:
        """
        Args:
            tokens: (batch, h*w, D)
            h, w:   spatial grid size — both must be even
        Returns:
            downsampled: (batch, (h//2)*(w//2), D)
        """
        batch, _, D = tokens.shape

        # Step 1: restore the 2D spatial layout.
        grid = tokens.reshape(batch, h, w, D)

        # Step 2: fold 2×2 spatial blocks into the channel dimension.
        # (B, h, w, D) → (B, h//2, 2, w//2, 2, D)
        grid = grid.reshape(batch, h // 2, 2, w // 2, 2, D)
        # → (B, h//2, w//2, 2, 2, D)
        grid = jnp.transpose(grid, (0, 1, 3, 2, 4, 5))
        # → (B, h//2, w//2, 4D)
        grid = grid.reshape(batch, h // 2, w // 2, 4 * D)

        # Step 3: project merged channels back down to D.
        grid = self.proj(grid)  # (B, h//2, w//2, D)

        # Step 4: flatten spatial dims back to a flat sequence.
        return grid.reshape(batch, (h // 2) * (w // 2), D)


# =============================================================================
# Vision Encoder
# =============================================================================


class VisionEncoder(nnx.Module):
    """
    Full ViT-style image encoder.

    Pipeline:
      image → PatchEmbedding → CPE → [VisionTransformerBlock × N] → PixelShuffle

    The output tokens are still in encoder hidden_dim space.
    VisionProjector (below) maps them to the LLM's d_model dimension.

    Args:
        config: VisionEncoderConfig
    """

    def __init__(self, config: VisionEncoderConfig, rngs: nnx.Rngs):
        self.config = config
        self.patch_embed = PatchEmbedding(config, rngs=rngs)
        self.cpe = ConditionalPositionalEncoding(config.hidden_dim, rngs=rngs)
        self.blocks = nnx.List(
            [VisionTransformerBlock(config, rngs=rngs) for _ in range(config.num_layers)]
        )
        self.pixel_shuffle = PixelShuffle(config.hidden_dim, rngs=rngs)
        self.norm = nnx.RMSNorm(config.hidden_dim, rngs=rngs)

        # --- Optional multi-teacher distillation components ---
        # These are only created when config.teachers is non-empty, so inference
        # and LLM integration (via __call__) are completely unaffected.
        if config.teachers:
            # Learnable CLS token: shape (1, 1, D), broadcast over batch in distill().
            # It is prepended to the patch sequence so every transformer layer can
            # read from it (and write to it via attention), producing a global image
            # summary analogous to the CLS token in BERT / ViT.
            self.cls_token = nnx.Param(jnp.zeros((1, 1, config.hidden_dim)))
            # One adapter head per teacher: linear projection into teacher's space.
            self.distillation_heads = nnx.List([
                DistillationHead(config.hidden_dim, t, rngs=rngs)
                for t in config.teachers
            ])

    def __call__(self, images: jax.Array) -> jax.Array:
        """
        Args:
            images: (batch, H, W, C) float image
        Returns:
            tokens: (batch, N_out, hidden_dim)
                    N_out = ((H // patch_size) // 2) * ((W // patch_size) // 2)
                    e.g. 64×64 image, patch 16 → 4×4 patches → 2×2 after shuffle = 4 tokens
        """
        # 1) Convert image pixels to patch token sequence.
        tokens, h, w = self.patch_embed(images)  # (B, h*w, D)

        # 2) Inject positional information via depthwise conv on the spatial grid.
        tokens = self.cpe(tokens, h, w)           # (B, h*w, D)

        # 3) Stack of ViT transformer blocks (bidirectional attention + MLP).
        for block in self.blocks:
            tokens = block(tokens)

        # 4) Normalize before downsampling (stabilises the pixel shuffle input).
        tokens = self.norm(tokens)

        # 5) Pixel shuffle: merge 2×2 neighborhoods → 4× fewer tokens.
        tokens = self.pixel_shuffle(tokens, h, w)  # (B, N_out, D)

        return tokens

    def distill(
        self, images: jax.Array
    ) -> tuple[dict, jax.Array, jax.Array]:
        """
        Forward pass for multi-teacher distillation training.

        This method is used ONLY during training — never for LLM inference.
        It runs the full encoder with a prepended CLS token, then applies
        each teacher's DistillationHead to produce predictions that are matched
        against frozen teacher outputs using compute_distillation_loss().

        Why a separate method (not inside __call__)?
        The LLM integration only needs pixel-shuffled patch tokens. Mixing the
        CLS token and distillation heads into __call__ would add dead weight at
        inference time and make both paths harder to read.

        Pipeline:
          1. PatchEmbedding + CPE  (same as __call__)
          2. Prepend CLS token → full_seq: (B, 1+N, D)
          3. All transformer blocks process full_seq (CLS ↔ patches via attention)
          4. RMSNorm on all positions
          5. Split: full_seq[:,0] = summary (global), full_seq[:,1:] = spatial (dense)
          6. Apply each DistillationHead → per-teacher student predictions

        Args:
            images: (B, H, W, C) float image
        Returns:
            teacher_preds:  {name: {"spatial": (B,N,D_t), "summary": (B,D_s)|None}}
                            Student adapter predictions; pass to compute_distillation_loss().
            spatial_tokens: (B, N, D) — raw patch features (before pixel shuffle)
            summary_token:  (B, D)    — CLS token output = global image summary
        """
        assert hasattr(self, "cls_token"), (
            "distill() requires config.teachers to be non-empty when building the encoder"
        )
        batch = images.shape[0]

        # Steps 1: patch embed + CPE (identical to __call__).
        tokens, h, w = self.patch_embed(images)  # (B, N, D)
        tokens = self.cpe(tokens, h, w)           # (B, N, D)

        # Step 2: prepend the learnable CLS token.
        # Broadcast (1, 1, D) → (B, 1, D), then concatenate at position 0.
        # Every subsequent attention layer can read from and write to this token.
        cls = jnp.broadcast_to(self.cls_token.value, (batch, 1, tokens.shape[-1]))
        full_seq = jnp.concatenate([cls, tokens], axis=1)  # (B, 1+N, D)

        # Step 3: transformer blocks — CLS and patch tokens attend to each other.
        for block in self.blocks:
            full_seq = block(full_seq)

        # Step 4: normalize all positions.
        full_seq = self.norm(full_seq)

        # Step 5: split into summary (CLS output) and spatial (patch outputs).
        summary_token = full_seq[:, 0, :]   # (B, D) — global image summary
        spatial_tokens = full_seq[:, 1:, :] # (B, N, D) — dense per-patch features

        # Step 6: project through each teacher's adapter head.
        teacher_preds = {}
        for head in self.distillation_heads:
            sp, su = head(spatial_tokens, summary_token)
            teacher_preds[head.name] = {"spatial": sp, "summary": su}

        return teacher_preds, spatial_tokens, summary_token


# =============================================================================
# Vision Projector
# =============================================================================


class VisionProjector(nnx.Module):
    """
    2-layer MLP that adapts visual encoder tokens to the LLM's hidden dimension.

    This is the "adapter" or "connector" between the vision encoder and the
    language model backbone. A simple 2-layer MLP with SiLU activation is
    what the paper uses (matching standard VLM practice).

    In the training curriculum, this projector is trained first (stage 0)
    with everything else frozen, so the LLM can initially "see" visual tokens
    without disrupting its text capabilities.

    Args:
        in_dim:  Vision encoder output dimension.
        out_dim: LLM hidden dimension (d_model).
    """

    def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs):
        hidden = (in_dim + out_dim) // 2  # smooth interpolation between dims
        self.fc1 = nnx.Linear(in_dim, hidden, use_bias=False, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, out_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: (batch, N, in_dim)
        Returns:
            (batch, N, out_dim)
        """
        return self.fc2(jax.nn.silu(self.fc1(x)))
