import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from xut.xut import TBackBone, XUTBackBone, XUDiT


def test_tbackbone_forward():
    B, N, D = 2, 4, 16
    x = jax.random.normal(jax.random.PRNGKey(0), (B, N, D))
    y = jax.random.normal(jax.random.PRNGKey(1), (B, D))

    model = TBackBone(
        dim=D,
        ctx_dim=D,
        heads=4,
        dim_head=4,
        mlp_dim=32,
        depth=2,
        use_adaln=True,
        rngs=nnx.Rngs(0),
    )

    out = model(x, ctx=x, pos_map=None, y=y, deterministic=True)
    assert out.shape == x.shape
    assert jnp.isfinite(out).all()


def test_xutbackbone_forward_and_enc_out():
    B, N, D = 2, 5, 12
    x = jax.random.normal(jax.random.PRNGKey(2), (B, N, D))
    ctx = jax.random.normal(jax.random.PRNGKey(3), (B, N, D))
    y = jax.random.normal(jax.random.PRNGKey(4), (B, D))

    model = XUTBackBone(
        dim=D,
        ctx_dim=D,
        heads=3,
        dim_head=4,
        mlp_dim=24,
        depth=2,
        enc_blocks=1,
        dec_blocks=1,
        use_adaln=True,
        rngs=nnx.Rngs(1),
    )

    out = model(x, ctx=ctx, y=y, deterministic=True)
    assert out.shape == x.shape
    assert jnp.isfinite(out).all()

    out2, enc = model(x, ctx=ctx, y=y, return_enc_out=True, deterministic=True)
    assert out2.shape == x.shape
    assert enc.shape == x.shape
    assert jnp.isfinite(out2).all()
    assert jnp.isfinite(enc).all()


def test_xudit_forward_minimal():
    B, H, W, C = 1, 4, 4, 4
    x = jax.random.normal(jax.random.PRNGKey(5), (B, H, W, C))
    t = jnp.array([0.5], dtype=jnp.float32)

    model = XUDiT(
        patch_size=2,
        input_dim=C,
        dim=16,
        ctx_dim=8,
        heads=4,
        dim_head=4,
        mlp_dim=32,
        depth=1,
        enc_blocks=1,
        dec_blocks=1,
        concat_ctx=True,
        shared_adaln=True,
        rngs=nnx.Rngs(2),
    )

    out = model(x, t, ctx=None, pos_map=None, deterministic=True)
    assert out.shape == (B, C, H, W)
    assert jnp.isfinite(out).all()


def test_xudit_with_ctx_mask():
    B, H, W, C = 1, 4, 4, 4
    ctx_len = 3
    x = jax.random.normal(jax.random.PRNGKey(6), (B, H, W, C))
    t = jnp.array([0.2], dtype=jnp.float32)
    ctx = jax.random.normal(jax.random.PRNGKey(7), (B, ctx_len, 4))
    ctx_mask = jnp.array([[1, 1, 0]], dtype=jnp.int32)

    model = XUDiT(
        patch_size=2,
        input_dim=C,
        dim=16,
        ctx_dim=4,
        heads=2,
        dim_head=8,
        mlp_dim=32,
        depth=1,
        enc_blocks=1,
        dec_blocks=1,
        concat_ctx=False,
        rngs=nnx.Rngs(3),
    )

    out = model(x, t, ctx=ctx, pos_map=None, ctx_mask=ctx_mask, deterministic=True)
    assert out.shape == (B, C, H, W)
    assert jnp.isfinite(out).all()
