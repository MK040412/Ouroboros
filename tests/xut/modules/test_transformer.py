import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from xut.modules.transformer import TransformerBlock

def test_transformer_block_self_only():
    key_params, key_x = jax.random.split(jax.random.PRNGKey(0))
    B, N, D = 2, 5, 16
    x = jax.random.normal(key_x, (B, N, D))

    block = TransformerBlock(
        dim=D,
        ctx_dim=None,
        heads=4,
        dim_head=4,
        mlp_dim=32,
        pos_dim=2,
        rngs=nnx.Rngs(key_params),
    )

    out = block(x, deterministic=True)
    assert out.shape == x.shape
    assert jnp.isfinite(out).all()


def test_transformer_block_cross_with_adaln():
    key_params, key_x, key_ctx, key_y = jax.random.split(jax.random.PRNGKey(1), 4)
    B, N, D = 2, 6, 24
    x = jax.random.normal(key_x, (B, N, D))
    ctx = jax.random.normal(key_ctx, (B, N, D))
    y = jax.random.normal(key_y, (B, D))

    block = TransformerBlock(
        dim=D,
        ctx_dim=D,
        heads=3,
        dim_head=8,
        mlp_dim=48,
        pos_dim=2,
        use_adaln=True,
        rngs=nnx.Rngs(key_params),
    )

    out = block(x, ctx=ctx, y=y, deterministic=True)
    assert out.shape == x.shape
    assert jnp.isfinite(out).all()


def test_transformer_block_shared_adaln_ctx_from_self():
    key_params, key_x, key_y = jax.random.split(jax.random.PRNGKey(2), 3)
    B, N, D = 2, 4, 12
    x = jax.random.normal(key_x, (B, N, D))
    y = jax.random.normal(key_y, (B, D))

    # Shared AdaLN parameters (scale, shift, gate) for sa/xa/mlp
    scale = jnp.zeros((B, D))
    shift = jnp.zeros((B, D))
    gate = jnp.zeros((B, D))
    shared = (scale, shift, gate)

    block = TransformerBlock(
        dim=D,
        ctx_dim=D,
        heads=3,
        dim_head=4,
        mlp_dim=24,
        pos_dim=2,
        use_adaln=True,
        use_shared_adaln=True,
        ctx_from_self=True,
        rngs=nnx.Rngs(key_params),
    )

    out = block(x, y=y, shared_adaln=(shared, shared, shared), deterministic=True)
    assert out.shape == x.shape
    assert jnp.isfinite(out).all()
