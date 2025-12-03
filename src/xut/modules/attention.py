from typing import Optional, Union, Callable
import jax
import jax.numpy as jnp
from flax import nnx

from .axial_rope import AxialRoPE

MaskType = Union[jnp.ndarray, None]

def _split_heads(x, n_heads):
    B, N, D = x.shape
    H = n_heads
    return x.reshape(B, N, H, D // H)

def _merge_heads(x):
    B, N, H, D = x.shape
    return x.reshape(B, N, H * D)

def _prepare_mask_bias(mask: MaskType, B: int, H: int, q_len: int, kv_len: int):
    if mask is None:
        return {'mask': None, 'bias': None}
    arr = jnp.asarray(mask)

    def _broadcast(arr):
        if arr.ndim == 2:
            if arr.shape == (B, kv_len):
                arr = arr[:, None, None, :]
            else:
                arr = arr[None, None, :, :]
        elif arr.ndim == 3:
            arr = arr[:, None, :, :]
        return jnp.broadcast_to(arr, (B, H, q_len, kv_len))

    if arr.dtype == jnp.bool_:
        return {'mask': _broadcast(arr), 'bias': None}
    return {'mask': None, 'bias': _broadcast(arr)}

class SelfAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        head_dim: int = -1,
        pos_dim: int = 2,
        *,
        rope_factory: Optional[Callable[[int, int, int], nnx.Module]] = None, # AxialRoPE alternative
        rngs: nnx.rnglib.Rngs,
    ):
        self.head_dim = head_dim if head_dim > 0 else dim // n_heads
        self.pos_dim = pos_dim
        self.n_heads = dim // self.head_dim
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=False, rngs=rngs)
        self.out = nnx.Linear(dim, dim, rngs=rngs)

        factory = rope_factory or (lambda d, h, p: AxialRoPE(d, h, p))
        self.rope = factory(self.head_dim, self.n_heads, self.pos_dim)

    def __call__(self, x, pos_map=None, mask=None, deterministic=True):
        b, n, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = _split_heads(q, self.n_heads)
        k = _split_heads(k, self.n_heads)
        v = _split_heads(v, self.n_heads)

        if pos_map is not None:
            # pos_map expected shape: (b, n, pos_dim)
            q = self.rope(q, pos_map)
            k = self.rope(k, pos_map)

        mask_bias = _prepare_mask_bias(mask, b, self.n_heads, n, n)
        attn = nnx.dot_product_attention(
            q, k, v,
            deterministic=deterministic,
            **mask_bias,
        )
        attn = _merge_heads(attn)
        return self.out(attn)

class CrossAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        ctx_dim: int,
        n_heads: int = 8,
        head_dim: int = -1,
        pos_dim: int = 2,
        *,
        rope_factory: Optional[Callable[[int, int, int], nnx.Module]] = None, # AxialRoPE alternative
        rngs: nnx.rnglib.Rngs,
    ):
        self.head_dim = head_dim if head_dim > 0 else dim // n_heads
        self.pos_dim = pos_dim
        self.n_heads = dim // self.head_dim
        self.q = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.kv = nnx.Linear(ctx_dim, dim * 2, use_bias=False, rngs=rngs)
        self.out = nnx.Linear(dim, dim, rngs=rngs)

        factory = rope_factory or (lambda d, h, p: AxialRoPE(d, h, p))
        self.rope = factory(self.head_dim, self.n_heads, self.pos_dim)

    def __call__(self, x, ctx, pos_map=None, ctx_pos_map=None, mask=None, deterministic=True):
        b, n, _ = x.shape
        ctx_n = ctx.shape[1]

        q = _split_heads(self.q(x), self.n_heads)
        k, v = jnp.split(self.kv(ctx), 2, axis=-1)
        k = _split_heads(k, self.n_heads)
        v = _split_heads(v, self.n_heads)

        if pos_map is not None:
            q = self.rope(q, pos_map)
        if ctx_pos_map is not None:
            k = self.rope(k, ctx_pos_map)

        mask_bias = _prepare_mask_bias(mask, b, self.n_heads, n, ctx_n)

        attn = nnx.dot_product_attention(
            q, k, v,
            deterministic=deterministic,
            **mask_bias,
        )
        attn = _merge_heads(attn)
        return self.out(attn)

if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    key, pkey_sa, pkey_ca, xkey, ckey = jax.random.split(key, 5)

    B, N, D = 2, 16, 64
    x = jax.random.normal(xkey, (B, N, D))
    ctx = jax.random.normal(ckey, (B, N, D))

    sa = SelfAttention(dim=D, n_heads=8, rngs=nnx.Rngs(pkey_sa))
    sa_out = sa(x)
    print(f'{sa_out.shape = }')

    ca = CrossAttention(dim=D, ctx_dim=D, n_heads=8, rngs=nnx.Rngs(pkey_ca))
    ca_out = ca(x, ctx)
    print(f'{ca_out.shape = }')
