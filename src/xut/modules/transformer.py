from typing import Optional, Type

import jax
import jax.numpy as jnp
from flax import nnx

from .layers import SwiGLU
from .attention import SelfAttention, CrossAttention
from .adaln import AdaLN

class TransformerBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        ctx_dim: Optional[int],
        heads: int,
        dim_head: int,
        mlp_dim: int,
        pos_dim: int,
        *,
        use_adaln: bool = False,
        use_shared_adaln: bool = False,
        ctx_from_self: bool = False,
        norm_layer: Type[nnx.Module] = nnx.RMSNorm,
        rngs: nnx.rnglib.Rngs,
    ):
        self.use_adaln = use_adaln
        self.ctx_from_self = ctx_from_self
        self.sa = SelfAttention(dim, heads, dim_head, pos_dim, rngs=rngs)
        self.xa = CrossAttention(dim, ctx_dim, heads, dim_head, pos_dim, rngs=rngs) if ctx_dim is not None else None
        self.mlp = SwiGLU(dim, mlp_dim, dim, rngs=rngs)

        if self.use_adaln:
            self.sa_pre = AdaLN(dim, dim, norm_layer=norm_layer, shared=use_shared_adaln, rngs=rngs)
            self.xa_pre = AdaLN(dim, dim, norm_layer=norm_layer, shared=use_shared_adaln, rngs=rngs) if self.xa else None
            self.mlp_pre = AdaLN(dim, dim, norm_layer=norm_layer, shared=use_shared_adaln, rngs=rngs)
        else:
            self.sa_pre = norm_layer(dim, rngs=rngs)
            self.xa_pre = norm_layer(dim, rngs=rngs) if self.xa else None
            self.mlp_pre = norm_layer(dim, rngs=rngs)

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
        if gate.ndim < x.ndim:
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
            if gate.ndim < x.ndim:
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
        if gate.ndim < x.ndim:
            gate = jnp.expand_dims(gate, axis=tuple(range(1, x.ndim - gate.ndim + 1)))
        x = x + gate * self.mlp(normed_x)
        return x

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
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

    print('Self-Attention Block')
    sa_block = TransformerBlock(
        dim=dim,
        ctx_dim=None,  # w/o ctx
        heads=heads,
        dim_head=head_dim,
        mlp_dim=mlp_dim,
        pos_dim=pos_dim,
        rngs=nnx.Rngs(key),
    )
    sa_out = sa_block(x, deterministic=True)
    print(x.shape, '->', sa_out.shape)
    assert sa_out.shape == x.shape

    print('Cross-Attention Block')
    xa_block = TransformerBlock(
        dim=dim,
        ctx_dim=y_dim,  # w/ ctx
        heads=heads,
        dim_head=head_dim,
        mlp_dim=mlp_dim,
        pos_dim=pos_dim,
        rngs=nnx.Rngs(key),
    )
    xa_out = xa_block(x, ctx=ctx, deterministic=True)
    print(x.shape, '->', xa_out.shape)
    assert xa_out.shape == x.shape

    print('AdaLN Block')
    adaln_block = TransformerBlock(
        dim=dim,
        ctx_dim=y_dim,
        heads=heads,
        dim_head=head_dim,
        mlp_dim=mlp_dim,
        pos_dim=pos_dim,
        use_adaln=True,
        use_shared_adaln=False,
        rngs=nnx.Rngs(key),
    )
    adaln_out = adaln_block(x, ctx=ctx, y=x, deterministic=True)
    print(x.shape, '->', adaln_out.shape)
    assert adaln_out.shape == x.shape

    print('Shared AdaLN Block')
    shared_adaln_block = TransformerBlock(
        dim=dim,
        ctx_dim=y_dim,
        heads=heads,
        dim_head=head_dim,
        mlp_dim=mlp_dim,
        pos_dim=pos_dim,
        use_adaln=True,
        use_shared_adaln=True,
        rngs=nnx.Rngs(key),
    )
    shared_adaln = (
        (jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)),  # sa
        (jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)),  # xa
        (jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)),  # mlp
    )
    out_shared = shared_adaln_block(
        x,
        ctx=ctx,
        y=x,
        shared_adaln=shared_adaln,
        deterministic=True,
    )
    print(x.shape, '->', out_shared.shape)
    assert out_shared.shape == x.shape