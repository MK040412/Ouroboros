from typing import Optional, Type

import jax
import jax.numpy as jnp
from flax import linen as nn

from .layers import SwiGLU
from .attention import SelfAttention, CrossAttention
from .norm import RMSNorm
from .adaln import AdaLN

class TransformerBlock(nn.Module):
    dim: int
    ctx_dim: Optional[int]
    heads: int
    dim_head: int
    mlp_dim: int
    pos_dim: int
    use_adaln: bool = False
    use_shared_adaln: bool = False
    ctx_from_self: bool = False
    norm_layer: Type[nn.Module] = RMSNorm

    def setup(self):
        self.sa = SelfAttention(self.dim, self.heads, self.dim_head, self.pos_dim)
        self.xa = CrossAttention(self.dim, self.ctx_dim, self.heads, self.dim_head, self.pos_dim) if self.ctx_dim is not None else None
        self.mlp = SwiGLU(self.dim, self.mlp_dim, self.dim)

        if self.use_adaln:
            self.sa_pre = AdaLN(self.dim, self.dim, norm_layer=self.norm_layer, shared=self.use_shared_adaln)
            self.xa_pre = AdaLN(self.dim, self.dim, norm_layer=self.norm_layer, shared=self.use_shared_adaln) if self.xa else None
            self.mlp_pre = AdaLN(self.dim, self.dim, norm_layer=self.norm_layer, shared=self.use_shared_adaln)
        else:
            self.sa_pre = self.norm_layer()
            self.xa_pre = self.norm_layer() if self.xa else None
            self.mlp_pre = self.norm_layer()

    def __call__(
        self,
        x,
        ctx=None,
        pos_map=None,
        ctx_pos_map=None,
        y=None,
        x_mask=None,
        ctx_mask=None,
        shared_adaln=None,
        deterministic: bool = True,
    ):
        # Self-Attention
        if self.use_adaln:
            sa_shared = shared_adaln[0] if shared_adaln is not None else None
            normed_x, gate = self.sa_pre(x, y, shared_adaln=sa_shared)
        else:
            normed_x = self.sa_pre(x)
            gate = jnp.ones_like(normed_x)
        gate = jnp.expand_dims(gate, axis=tuple(range(1, x.ndim - gate.ndim + 1)))
        x = x + gate * self.sa(
            normed_x,
            pos_map=pos_map,
            mask=x_mask,
            deterministic=deterministic
        )

        # Cross-Attention
        if self.xa is not None:
            if self.use_adaln:
                xa_shared = shared_adaln[1] if shared_adaln is not None else None
                normed_x, gate = self.xa_pre(x, y, shared_adaln=xa_shared)
            else:
                normed_x = self.xa_pre(x)
                gate = jnp.ones_like(normed_x)
            gate = jnp.expand_dims(gate, axis=tuple(range(1, x.ndim - gate.ndim + 1)))
            xa_ctx = x if self.ctx_from_self else ctx
            xa_mask = x_mask if self.ctx_from_self else ctx_mask
            x = x + gate * self.xa(
                normed_x, xa_ctx,
                pos_map=pos_map,
                ctx_pos_map=ctx_pos_map,
                mask=xa_mask,
                deterministic=deterministic
            )

        # MLP
        if self.use_adaln:
            mlp_shared = shared_adaln[2] if shared_adaln is not None else None
            normed_x, gate = self.mlp_pre(x, y, shared_adaln=mlp_shared)
        else:
            normed_x = self.mlp_pre(x)
            gate = jnp.ones_like(normed_x)
        gate = jnp.expand_dims(gate, axis=tuple(range(1, x.ndim - gate.ndim + 1)))
        x = x + gate * self.mlp(normed_x)
        return x

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    batch_size = 2
    seq_len = 16
    dim = 64
    y_dim = 64
    mlp_dim = 128
    heads = 8
    head_dim = 8
    pos_dim = 2

    x = jax.random.normal(key, (batch_size, seq_len, dim))
    ctx = jax.random.normal(key, (batch_size, seq_len, y_dim))

    print("Self-Attention Block")
    sa_block = TransformerBlock(
        dim=dim,
        ctx_dim=None, # w/o ctx
        heads=heads,
        dim_head=head_dim,
        mlp_dim=mlp_dim,
        pos_dim=pos_dim,
    )

    sa_variables = sa_block.init(key, x)
    sa_out = sa_block.apply(sa_variables, x)
    print(x.shape, "->", sa_out.shape)
    assert x.shape, sa_out.shape

    print("Cross-Attention Block")
    xa_block = TransformerBlock(
        dim=dim,
        ctx_dim=y_dim, # w/ ctx
        heads=heads,
        dim_head=head_dim,
        mlp_dim=mlp_dim,
        pos_dim=pos_dim,
    )

    xa_variables = xa_block.init(key, x, ctx)
    xa_out = xa_block.apply(xa_variables, x, ctx)
    print(x.shape, "->", xa_out.shape)
    assert x.shape, xa_out.shape

    print("AdaLN Block")
    adaln_block = TransformerBlock(
        dim=dim,
        ctx_dim=y_dim,
        heads=heads,
        dim_head=head_dim,
        mlp_dim=mlp_dim,
        pos_dim=pos_dim,
        use_adaln=True,
        use_shared_adaln=False,
    )

    adaln_variables = adaln_block.init(key, x, ctx, y=x)
    adaln_out = adaln_block.apply(adaln_variables, x, ctx, y=x)
    print(x.shape, "->", adaln_out.shape)
    assert x.shape, adaln_out.shape

    print("Shared AdaLN Block")
    shared_adaln_block = TransformerBlock(
        dim=dim,
        ctx_dim=y_dim,
        heads=heads,
        dim_head=head_dim,
        mlp_dim=mlp_dim,
        pos_dim=pos_dim,
        use_adaln=True,
        use_shared_adaln=True,
    )

    shared_adaln = (jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x))
    shared_variables = shared_adaln_block.init(key, x, ctx, y=x, shared_adaln=shared_adaln)
    out_shared = shared_adaln_block.apply(shared_variables, x, ctx, y=x, shared_adaln=shared_adaln)
    print(x.shape, "->", out_shared.shape)
    assert x.shape, out_shared.shape