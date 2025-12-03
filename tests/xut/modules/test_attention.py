import pytest
import jax
from jax import numpy as jnp
from flax import nnx

from xut.modules.attention import SelfAttention, CrossAttention

def test_self_attention_basic():
    key_params, key_data = jax.random.split(jax.random.PRNGKey(0))
    B, N, D = 2, 8, 32
    x = jax.random.normal(key_data, (B, N, D))
    sa = SelfAttention(dim=D, n_heads=4, rngs=nnx.Rngs(key_params))
    out = sa(x)
    assert out.shape == (B, N, D)
    assert jnp.all(jnp.isfinite(out))

def test_self_attention_with_mask():
    key_params, key_data = jax.random.split(jax.random.PRNGKey(1))
    B, N, D = 2, 6, 24
    x = jax.random.normal(key_data, (B, N, D))
    causal_mask = jnp.tril(jnp.ones((N, N), dtype=bool))  # (N, N) -> broadcast
    sa = SelfAttention(dim=D, n_heads=3, rngs=nnx.Rngs(key_params))
    out = sa(x, mask=causal_mask)
    assert out.shape == (B, N, D)
    assert jnp.all(jnp.isfinite(out))

def test_cross_attention_basic():
    key_params, key_x, key_ctx = jax.random.split(jax.random.PRNGKey(2), 3)
    B, N, D = 2, 7, 40
    x = jax.random.normal(key_x, (B, N, D))
    ctx = jax.random.normal(key_ctx, (B, N, D))
    ca = CrossAttention(dim=D, ctx_dim=D, n_heads=5, rngs=nnx.Rngs(key_params))
    out = ca(x, ctx)
    assert out.shape == (B, N, D)
    assert jnp.all(jnp.isfinite(out))

def test_cross_attention_with_pos_map():
    key_params, key_x, key_ctx = jax.random.split(jax.random.PRNGKey(3), 3)
    B, N, D = 2, 5, 32
    x = jax.random.normal(key_x, (B, N, D))
    ctx = jax.random.normal(key_ctx, (B, N, D))
    pos_map = jnp.linspace(-1.0, 1.0, N * 2).reshape(N, 2)
    pos_map = jnp.broadcast_to(pos_map, (B, N, 2))  # (B, N, pos_dim)
    ca = CrossAttention(dim=D, ctx_dim=D, n_heads=4, rngs=nnx.Rngs(key_params))
    out = ca(x, ctx, pos_map=pos_map, ctx_pos_map=pos_map)
    assert out.shape == (B, N, D)
    assert jnp.all(jnp.isfinite(out))

def test_self_attention_gradients():
    key_params, key_data = jax.random.split(jax.random.PRNGKey(4))
    B, N, D = 2, 4, 16
    x = jax.random.normal(key_data, (B, N, D))
    sa = SelfAttention(dim=D, n_heads=4, rngs=nnx.Rngs(key_params))

    def loss_fn(inp):
        return sa(inp).sum()

    grads = jax.grad(loss_fn)(x)
    assert grads.shape == x.shape
    assert jnp.all(jnp.isfinite(grads))

def test_cross_attention_gradients():
    key_params, key_x, key_ctx = jax.random.split(jax.random.PRNGKey(5), 3)
    B, N, D = 2, 4, 16
    x = jax.random.normal(key_x, (B, N, D))
    ctx = jax.random.normal(key_ctx, (B, N, D))
    ca = CrossAttention(dim=D, ctx_dim=D, n_heads=4, rngs=nnx.Rngs(key_params))

    def loss_fn(inp):
        return ca(inp, ctx).sum()

    grads = jax.grad(loss_fn)(x)
    assert grads.shape == x.shape
    assert jnp.all(jnp.isfinite(grads))