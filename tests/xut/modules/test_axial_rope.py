import pytest
import jax
import jax.numpy as jnp

from xut.modules.axial_rope import (
    rotate_half, apply_rotary_emb, make_axial_pos,
    freqs_pixel_log, AxialRoPE,
)

def test_rotate_half_basic():
    x = jnp.array([[1., 2., 3., 4.]])  # (..., 4)
    y = rotate_half(x)
    assert y.shape == x.shape
    # (1,2)->(-2,1), (3,4)->(-4,3)
    assert jnp.allclose(y, jnp.array([[-2., 1., -4., 3.]]))

def test_apply_rotary_emb_matches_manual():
    freqs = jnp.array([1., 2.])  # rot_dim=2
    t = jnp.array([[1., 2., 3., 4.]])  # last dim=4, rotate middle 2 dims
    out = apply_rotary_emb(freqs, t, start_index=1)
    # manual rotation on positions 1..2
    t_left, t_mid, t_right = t[..., :1], t[..., 1:3], t[..., 3:]
    manual = jnp.concatenate(
        [t_left,
         t_mid * jnp.cos(freqs) + rotate_half(t_mid) * jnp.sin(freqs),
         t_right],
        axis=-1,
    )
    assert jnp.allclose(out, manual)

def test_make_axial_pos_shape_and_range():
    pos = make_axial_pos(4, 6, pixel_aspect_ratio=1.0, align_corners=False)
    assert pos.shape == (24, 2)
    assert jnp.all(jnp.abs(pos) <= 1.0)

def test_axial_rope_shapes_and_norm_preserved():
    key = jax.random.PRNGKey(0)
    b, h, w = 2, 4, 4
    n_heads, head_dim = 8, 8
    x = jax.random.normal(key, (b, h * w, n_heads, head_dim))
    pos = make_axial_pos(h, w)  # (16,2)

    rope = AxialRoPE(dim=head_dim, n_heads=n_heads, pos_dim=2)
    y = rope(x, pos)
    assert y.shape == x.shape
    # RoPE는 회전이므로 norm 보존
    assert jnp.allclose(jnp.linalg.norm(y, axis=-1),
                        jnp.linalg.norm(x, axis=-1),
                        atol=1e-5)

def test_axial_rope_raises_on_wrong_pos_dim():
    rope = AxialRoPE(dim=8, n_heads=2, pos_dim=2)
    x = jnp.zeros((1, 4, 2, 4))
    pos_bad = jnp.zeros((4, 3))
    with pytest.raises(ValueError):
        _ = rope(x, pos_bad)

def test_jit_consistency():
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (1, 16, 8, 8))
    pos = make_axial_pos(4, 4)
    rope = AxialRoPE(dim=8, n_heads=8, pos_dim=2)

    f = jax.jit(lambda a, p: rope(a, p))
    y1 = rope(x, pos)
    y2 = f(x, pos)
    assert jnp.allclose(y1, y2)
