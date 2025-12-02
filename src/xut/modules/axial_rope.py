from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn

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


def centers(start: float, stop: float, num: int, dtype=None) -> jnp.ndarray:
    edges = jnp.linspace(start, stop, num + 1, dtype=dtype)
    return (edges[:-1] + edges[1:]) / 2


def make_grid(h_pos, w_pos):
    grid = jnp.stack(jnp.meshgrid(h_pos, w_pos, indexing="ij"), axis=-1)
    return grid.reshape(-1, grid.shape[-1])


def bounding_box(h: int, w: int, pixel_aspect_ratio: float = 1.0) -> jnp.ndarray:
    w_adj = w
    h_adj = h * pixel_aspect_ratio
    ar_adj = w_adj / h_adj
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj

    return jnp.array([y_min, y_max, x_min, x_max])

def make_axial_pos(
    h: int,
    w: int,
    pixel_aspect_ratio: float = 1.0,
    align_corners: bool = False,
    dtype=None,
) -> jnp.ndarray:
    y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    if align_corners:
        h_pos = jnp.linspace(y_min, y_max, h, dtype=dtype)
        w_pos = jnp.linspace(x_min, x_max, w, dtype=dtype)
    else:
        h_pos = centers(jnp.asarray(y_min), jnp.asarray(y_max), h, dtype=dtype)
        w_pos = centers(jnp.asarray(x_min), jnp.asarray(x_max), w, dtype=dtype)
    return make_grid(h_pos, w_pos)

def make_cropped_pos(crop_h, crop_w, target_h, target_w):
    pos_map = make_axial_pos(target_h, target_w).reshape(
        target_h, target_w, -1
    )
    if target_h > target_w:
        pos_map = pos_map[crop_h : crop_h + target_w, :, :]
    elif target_h < target_w:
        pos_map = pos_map[:, crop_w : crop_w + target_h, :]
    return pos_map.reshape(-1, pos_map.shape[-1])

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

class AxialRoPE(nn.Module):
    dim: int
    n_heads: int
    pos_dim: int = 2
    start_index: int = 0
    freqs_init: Callable[[tuple], jnp.ndarray] = freqs_pixel_log(max_freq=10.0)

    def setup(self):
        per_axis = self.dim // (2 * self.pos_dim)
        log_freqs = self.freqs_init((self.n_heads, per_axis, 1))
        self.log_freqs = self.param("log_freqs",
            lambda *_: jnp.array(log_freqs),
            (self.n_heads, per_axis, 1),
        )

    def get_freqs(self, pos: jnp.ndarray) -> jnp.ndarray:
        if pos.shape[-1] != self.pos_dim:
            raise ValueError(f"Expected pos_dim={self.pos_dim}, got {pos.shape[-1]}")

        freqs_axes = jnp.exp(self.log_freqs).repeat(self.pos_dim, axis=-1)
        freqs = pos[..., None, None, :] * freqs_axes
        freqs = freqs.reshape(*freqs.shape[:-2], -1)
        freqs = jnp.repeat(freqs, 2, axis=-1)
        freqs = jnp.swapaxes(freqs, -3, -2)
        return freqs

    def __call__(self, x: jnp.ndarray, pos: jnp.ndarray) -> jnp.ndarray:
        freqs = self.get_freqs(pos)
        return apply_rotary_emb(freqs, x, self.start_index)

class AdditiveAxialRoPE(nn.Module):
    dim: int
    n_heads: int
    pos_dim: int = 2
    start_index: int = 0
    freqs_init: Callable[[tuple], jnp.ndarray] = freqs_pixel_log(max_freq=10.0)

    def setup(self):
        self.rope = AxialRoPE(
            self.dim,
            self.n_heads,
            self.pos_dim,
            self.start_index,
            self.freqs_init,
        )
        self.emb = self.param("emb",
            nn.initializers.normal(stddev=1.0 / jnp.sqrt(self.dim)),
            (self.dim,),
        )

    def __call__(self, x: jnp.ndarray, pos: jnp.ndarray) -> jnp.ndarray:
        pos_emb = jnp.zeros_like(x) + self.emb[-x.shape[-1]:]
        if x.ndim == 3:
            pos_emb = pos_emb[:, None, ...]
        rotated = self.rope(pos_emb, pos)
        return x + rotated.reshape(x.shape)

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    B, H, W, D = 2, 4, 4, 16
    x = jax.random.normal(key, (B, 1, H * W, D))
    pos = make_axial_pos(H, W)
    pos = pos.reshape(-1, pos.shape[-1])
    model = AxialRoPE(D, n_heads=1, pos_dim=2)
    variables = model.init(key, x, pos)
    output = model.apply(variables, x, pos)
    print(output.shape)