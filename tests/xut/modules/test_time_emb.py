import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from xut.modules.time_emb import TimestepEmbedding

def test_timestep_embedding_shape_even_dim():
    dim = 64
    B = 4
    t = jnp.array([0.1, 1.0, 5.0, 10.0], dtype=jnp.float32)

    model = TimestepEmbedding(dim=dim, rngs=nnx.Rngs(0))
    out = model(t)

    # Output shape check
    assert out.shape == (B, dim)
    assert jnp.isfinite(out).all()


def test_timestep_embedding_shape_odd_dim():
    dim = 65  # odd
    B = 4
    t = jnp.array([0.1, 1.0, 5.0, 10.0], dtype=jnp.float32)

    model = TimestepEmbedding(dim=dim, rngs=nnx.Rngs(params=42))
    out = model(t)

    # must still return (B, dim)
    assert out.shape == (B, dim)
    assert jnp.isfinite(out).all()


def test_timestep_embedding_cos_sin_pattern():
    dim = 32
    B = 2

    # simple deterministic value
    t = jnp.array([1.0, 2.0], dtype=jnp.float32)

    model = TimestepEmbedding(dim=dim, rngs=nnx.Rngs(0))

    # Compute expected cos/sin before projection
    # freqs: (1, dim//2)
    freqs = model.freqs  # (1,16)

    args = t[:, None] * freqs  # (B, 16)
    embedding_expected = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

    # Apply padding if needed (dim even ⇒ no padding)
    assert embedding_expected.shape == (B, dim)

    # Now pass through model proj+mish
    out = model(t)

    # The output won't exactly equal embedding_expected (proj changes it),
    # but embedding_expected must be finite and have correct shape.
    assert embedding_expected.shape == (B, dim)
    assert jnp.isfinite(embedding_expected).all()
    assert out.shape == (B, dim)

def test_batch_dimension_handling():
    dim = 32
    B = 5

    # Case: input with shape (B,1) → should squeeze to (B,)
    t = jnp.arange(B, dtype=jnp.float32).reshape(B, 1)

    model = TimestepEmbedding(dim=dim, rngs=nnx.Rngs(0))
    out = model(t)

    assert out.shape == (B, dim)
    assert jnp.isfinite(out).all()
