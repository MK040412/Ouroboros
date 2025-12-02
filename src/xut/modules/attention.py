import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

from .axial_rope import AxialRoPE

class SelfAttention(nn.Module):
    dim: int
    n_heads: int = 8
    head_dim: int = -1
    pos_dim: int = 2
    rope: Optional[nn.Module] = None  # AxialRoPE alternative

    def setup(self):
        self._head_dim = self.head_dim if self.head_dim > 0 else self.dim // self.n_heads
        self._n_heads = self.dim // self._head_dim
        self.qkv = nn.Dense(self.dim * 3, use_bias=False)
        self.out = nn.Dense(self.dim)
        self.rope_layer = (self.rope or AxialRoPE)(self._head_dim, self._n_heads, self.pos_dim)

    def __call__(self, x, pos_map=None, mask=None, deterministic=True):
        B, N, _ = x.shape
        H = self._n_heads
        D = self._head_dim

        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(B, N, H, D).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, H, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, H, D).transpose(0, 2, 1, 3)

        if pos_map is not None:
            q = self.rope_layer(q, pos_map)
            k = self.rope_layer(k, pos_map)
        
        bias = None
        if mask is not None:
            mask = jnp.asarray(mask, dtype=bool)
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            bias = jnp.where(mask, 0.0, -jnp.inf)
        
        attn = nn.dot_product_attention(
            q, k, v,
            bias=bias,
            deterministic=deterministic,
        )
        attn = attn.transpose(0, 2, 1, 3).reshape(B, N, H * self._head_dim)
        attn = self.out(attn)
        return attn

class CrossAttention(nn.Module):
    dim: int
    ctx_dim: int
    n_heads: int = 8
    head_dim: int = -1
    pos_dim: int = 2
    rope: Optional[nn.Module] = None

    def setup(self):
        self._head_dim = self.head_dim if self.head_dim > 0 else self.dim // self.n_heads
        self._n_heads = self.dim // self._head_dim
        self.q = nn.Dense(self.dim, use_bias=False)
        self.kv = nn.Dense(self.dim * 2, use_bias=False)
        self.out = nn.Dense(self.dim)
        self.rope_layer = (self.rope or AxialRoPE)(self._head_dim, self._n_heads, self.pos_dim)

    def __call__(self, x, ctx, pos_map=None, ctx_pos_map=None, mask=None, deterministic=True):
        B, N, _ = x.shape
        ctx_N = ctx.shape[1]
        H = self._n_heads
        D = self._head_dim

        q = self.q(x)
        kv = self.kv(ctx)
        k, v = jnp.split(kv, 2, axis=-1)

        q = q.reshape(B, N, H, D).transpose(0, 2, 1, 3)
        k = k.reshape(B, ctx_N, H, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, ctx_N, H, D).transpose(0, 2, 1, 3)

        if pos_map is not None:
            q = self.rope_layer(q, pos_map)
        if ctx_pos_map is not None:
            k = self.rope_layer(k, pos_map)
        
        bias = None
        if mask is not None:
            mask = jnp.asarray(mask, dtype=bool)
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            bias = jnp.where(mask, 0.0, -jnp.inf)
        
        attn = nn.dot_product_attention(
            q, k, v,
            bias=bias,
            deterministic=deterministic,
        )
        attn = attn.transpose(0, 2, 1, 3).reshape(B, N, H * self._head_dim)
        attn = self.out(attn)
        return attn

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    B, N, D = 2, 16, 64
    x = jax.random.normal(key, (B, N, D))
    ctx = jax.random.normal(key, (B, N, D))

    sa = SelfAttention(dim=D, n_heads=8)
    vars_sa = sa.init(key, x)
    out_sa = sa.apply(vars_sa, x)
    print("SelfAttention:", out_sa.shape)

    ca = CrossAttention(dim=D, ctx_dim=D, n_heads=8)
    vars_ca = ca.init(key, x, ctx)
    out_ca = ca.apply(vars_ca, x, ctx)
    print("CrossAttention:", out_ca.shape)