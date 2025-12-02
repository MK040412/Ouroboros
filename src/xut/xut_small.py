"""
XUT-Small 모델 (256x256 VAE latent용)
Configuration from train_tpu_256.py
"""

from flax import linen as nn
from .xut import XUDiT


def create_xut_small(
    patch_size: int = 2,
    input_dim: int = 4,
    dim: int = 896,
    ctx_dim: int = 768,
    ctx_size: int = 256,
    heads: int = 14,
    dim_head: int = 64,
    mlp_dim: int = 3072,
    depth: int = 4,
    enc_blocks: int = 1,
    dec_blocks: int = 2,
    shared_adaln: bool = True,
    grad_ckpt: bool = True,
) -> nn.Module:
    """
    XUT-Small 모델 생성
    
    Args:
        patch_size: Patch embedding size
        input_dim: VAE latent channels (4)
        dim: Model dimension (896)
        ctx_dim: Context embedding dimension (768 from Embedding Gemma 300M)
        ctx_size: Context sequence length
        heads: Number of attention heads (14)
        dim_head: Head dimension (64)
        mlp_dim: MLP hidden dimension (3072)
        depth: Number of transformer blocks (4)
        enc_blocks: Number of encoder blocks per depth
        dec_blocks: Number of decoder blocks per depth
        shared_adaln: Use shared AdaLN projections
        grad_ckpt: Use gradient checkpointing
    
    Returns:
        XUDiT model instance
    """
    return XUDiT(
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
        dec_ctx=False,
        class_cond=0,
        shared_adaln=shared_adaln,
        concat_ctx=True,
        use_dyt=False,
        double_t=False,
        addon_info_embs_dim=None,
        tread_config={
            "dropout_ratio": 0.5,  # TREAD selection rate
            "prev_trns_depth": 0,
            "post_trns_depth": 0,
        },
        grad_ckpt=grad_ckpt,
    )
