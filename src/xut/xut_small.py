"""
XUT-Small model factory for 256x256 image training

Configuration (from train_tpu_256.py):
- model_dim: 896
- context_dim: 768 (Embedding Gemma 300M)
- mlp_dim: 3072
- heads: 14
- depth: 4
- enc_blocks: 1
- dec_blocks: 2
"""

from flax import nnx
from .xut import XUDiT


def create_xut_small(
    patch_size: int = 2,
    input_dim: int = 4,       # VAE latent channels
    dim: int = 896,           # Model dimension
    ctx_dim: int = 768,       # Text embedding dimension (Gemma 300M)
    ctx_size: int = 77,       # Max context tokens
    heads: int = 14,
    dim_head: int = 64,
    mlp_dim: int = 3072,
    depth: int = 4,
    enc_blocks: int = 1,
    dec_blocks: int = 2,
    dec_ctx: bool = True,
    shared_adaln: bool = True,
    concat_ctx: bool = True,
    seed: int = 42,
) -> XUDiT:
    """Create XUT-Small model for 256x256 image generation

    Args:
        patch_size: Patch size for input (default 2 for 32x32 latent -> 16x16 patches)
        input_dim: Input channels (4 for VAE latent)
        dim: Model hidden dimension
        ctx_dim: Context (text embedding) dimension
        ctx_size: Maximum context sequence length
        heads: Number of attention heads
        dim_head: Dimension per head
        mlp_dim: MLP hidden dimension
        depth: Number of transformer blocks
        enc_blocks: Encoder blocks per stack
        dec_blocks: Decoder blocks per stack
        dec_ctx: Whether decoder attends to context
        shared_adaln: Use shared AdaLN parameters
        concat_ctx: Concatenate context to sequence
        seed: Random seed

    Returns:
        XUDiT model instance
    """
    rngs = nnx.Rngs(seed)

    model = XUDiT(
        patch_size=patch_size,
        input_dim=input_dim,
        dim=dim,
        ctx_dim=ctx_dim,
        ctx_size=ctx_size,
        heads=heads,
        dim_head=dim_head,
        mlp_dim=mlp_dim,
        depth=depth,
        enc_blocks=enc_blocks,
        dec_blocks=dec_blocks,
        dec_ctx=dec_ctx,
        class_cond=0,
        shared_adaln=shared_adaln,
        concat_ctx=concat_ctx,
        use_dyt=False,
        double_t=False,
        addon_info_embs_dim=None,
        tread_config=None,
        grad_ckpt=False,
        rngs=rngs,
    )

    return model


def count_parameters(model: XUDiT) -> int:
    """Count total trainable parameters"""
    import jax.numpy as jnp

    total = 0
    for path, value in nnx.iter_graph(model):
        if isinstance(value, nnx.Variable) and hasattr(value, 'value'):
            total += value.value.size
    return total


if __name__ == "__main__":
    # Test model creation
    model = create_xut_small()
    print(f"XUT-Small created")
    print(f"  Dimension: 896")
    print(f"  Context dim: 768")
    print(f"  Heads: 14")
    print(f"  Depth: 4")

    # Test forward pass
    import jax.numpy as jnp

    batch_size = 2
    height, width = 32, 32  # Latent size for 256x256 image
    channels = 4

    x = jnp.zeros((batch_size, height, width, channels))
    t = jnp.zeros((batch_size, 1))
    ctx = jnp.zeros((batch_size, 77, 768))

    output = model(x, t, ctx=ctx, deterministic=True)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  ✓ Forward pass successful")
