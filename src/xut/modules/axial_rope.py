from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax
from flax import nnx

from functools import partial

@jax.jit
def rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x = jnp.stack((-x2, x1), axis=-1)
    *shape, d, r = x.shape
    return x.reshape(*shape, d * r)

@partial(jax.jit, static_argnames=("start_index",))
def apply_rotary_emb(
    freqs: jnp.ndarray,
    t: jnp.ndarray,
    start_index: int = 0,
    scale: float = 1.0
) -> jnp.ndarray:
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    t_left  = lax.slice_in_dim(t, 0, start_index, axis=-1)
    t_mid   = lax.slice_in_dim(t, start_index, end_index, axis=-1)
    t_right = lax.slice_in_dim(t, end_index, t.shape[-1], axis=-1)

    freqs_cast = jnp.broadcast_to(freqs, t_mid.shape)
    t_rot = (t_mid * jnp.cos(freqs_cast) * scale) + (rotate_half(t_mid) * jnp.sin(freqs_cast) * scale)
    return jnp.concatenate([t_left, t_rot, t_right], axis=-1)

@partial(jax.jit, static_argnames=("num",))
def centers(start: float, stop: float, num: int, dtype=None):
    edges = jnp.linspace(start, stop, num + 1, dtype=dtype)
    return (edges[:-1] + edges[1:]) / 2

@jax.jit
def make_grid(h_pos, w_pos):
    grid = jnp.stack(jnp.meshgrid(h_pos, w_pos, indexing="ij"), axis=-1)
    return grid.reshape(-1, grid.shape[-1])

@jax.jit
def bounding_box(h: int, w: int, pixel_aspect_ratio: float = 1.0):
    w_adj = w
    h_adj = h * pixel_aspect_ratio
    ar_adj = w_adj / h_adj
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    return jnp.select(
        [ar_adj > 1, ar_adj < 1],
        [jnp.array([-1 / ar_adj, 1 / ar_adj, x_min, x_max]),
         jnp.array([y_min, y_max, -ar_adj, ar_adj])],
        jnp.array([y_min, y_max, x_min, x_max])
    ).astype(jnp.float32)
    # return jnp.array([y_min, y_max, x_min, x_max])

@partial(jax.jit, static_argnames=("h", "w", "align_corners",))
def make_axial_pos(
    h: int,
    w: int,
    pixel_aspect_ratio: float = 1.0,
    align_corners: bool = False,
    dtype = None,
):
    y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    if align_corners:
        h_pos = jnp.linspace(y_min, y_max, h, dtype=dtype)
        w_pos = jnp.linspace(x_min, x_max, w, dtype=dtype)
    else:
        h_pos = centers(jnp.asarray(y_min), jnp.asarray(y_max), h, dtype=dtype)
        w_pos = centers(jnp.asarray(x_min), jnp.asarray(x_max), w, dtype=dtype)
    return make_grid(h_pos, w_pos)

def freqs_pixel(max_freq: float = 10.0) -> Callable[[tuple], jnp.ndarray]:
    def init(shape: tuple) -> jnp.ndarray:
        freqs = jnp.linspace(1.0, max_freq / 2.0, shape[-1]) * jnp.pi
        return jnp.log(freqs).reshape((1,) * (len(shape) - 1) + (shape[-1],)).repeat(shape[0], 0)
    return init

def freqs_pixel_log(max_freq: float = 10.0) -> Callable[[tuple], jnp.ndarray]:
    def init(shape: tuple) -> jnp.ndarray:
        log_min = jnp.log(jnp.pi)
        log_max = jnp.log(max_freq * jnp.pi / 2.0)
        return jnp.linspace(log_min, log_max, shape[-1]).reshape((1,) * (len(shape) - 1) + (shape[-1],)).repeat(shape[0], 0)
    return init

class AxialRoPE(nnx.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        pos_dim: int = 2,
        start_index: int = 0,
        freqs_init: Callable[[tuple], jnp.ndarray] = freqs_pixel_log(max_freq=10.0),
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.pos_dim = pos_dim
        self.start_index = start_index
        
        per_axis = self.dim // (2 * self.pos_dim)
        log_freqs = freqs_init((self.n_heads, per_axis, 1))
        self.log_freqs = nnx.Param(log_freqs)

    def __call__(self, x: jnp.ndarray, pos: jnp.ndarray):
        if pos.shape[-1] != self.pos_dim:
            raise ValueError(f"Expected pos_dim={self.pos_dim}, got {pos.shape[-1]}")

        freqs_axes = jnp.exp(self.log_freqs.value)
        freqs_axes = freqs_axes.repeat(self.pos_dim, axis=-1)
        freqs = pos[..., None, None, :] * freqs_axes
        freqs = freqs.reshape(*freqs.shape[:-2], -1)
        freqs = jnp.repeat(freqs, 2, axis=-1)
        return apply_rotary_emb(freqs, x, self.start_index)

if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    batch_size, seq_len, dim = 2, 16, 64
    n_heads = 8
    pos_dim = 2

    x = jax.random.normal(key, (batch_size, seq_len, n_heads, dim // n_heads))

    h, w = 4, 4
    pos = make_axial_pos(h, w)

    model = AxialRoPE(dim=dim // n_heads, n_heads=n_heads, pos_dim=pos_dim, start_index=0)
    output = model(x, pos)
    print(f'{output.shape = }')
