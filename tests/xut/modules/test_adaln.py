import jax
import jax.numpy as jnp
from flax import nnx

from xut.modules.adaln import AdaLN


def test_adaln_unshared_shapes_and_values():
    key_params, key_x, key_y = jax.random.split(jax.random.PRNGKey(0), 3)
    B, N, D = 2, 3, 8
    x = jax.random.normal(key_x, (B, N, D))
    y = jax.random.normal(key_y, (B, D))

    adaln = AdaLN(dim=D, y_dim=D, rngs=nnx.Rngs(key_params))
    out, gate = adaln(x, y)

    assert out.shape == x.shape
    assert gate.shape == (B, 1, D)
    assert jnp.isfinite(out).all()
    assert jnp.allclose(gate, 1.0, atol=1e-5)  # zero init -> gate ~1


def test_adaln_shared_applies_manual_state():
    key_params, key_x = jax.random.split(jax.random.PRNGKey(1))
    B, N, D = 1, 2, 4
    x = jax.random.normal(key_x, (B, N, D))

    adaln = AdaLN(dim=D, y_dim=D, shared=True, rngs=nnx.Rngs(key_params))

    # Manually craft shared AdaLN pieces
    scale = jnp.full((B, D), 0.5)
    shift = jnp.full((B, D), 0.1)
    gate_raw = jnp.full((B, D), 0.2)
    shared = (scale, shift, gate_raw)

    normed = adaln.norm(x)
    expected_out = normed * (scale[:, None, :] + 1.0) + shift[:, None, :]
    expected_gate = gate_raw[:, None, :] + 1.0

    out, gate = adaln(x, y=jnp.zeros((B, D)), shared_adaln=shared)

    assert out.shape == x.shape
    assert gate.shape == (B, 1, D)
    assert jnp.allclose(out, expected_out, atol=1e-5)
    assert jnp.allclose(gate, expected_gate, atol=1e-5)
