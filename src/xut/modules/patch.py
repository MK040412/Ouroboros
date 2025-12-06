from typing import Optional, Type

import jax
import jax.numpy as jnp
from jax import lax
from flax import nnx

class Identity(nnx.Module):
    def __call__(self, x):
        return x

class PatchEmbed(nnx.Module):
    def __init__(
        self,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 512,
        *,
        norm_layer: Optional[Type[nnx.Module]] = None,
        flatten: bool = True,
        use_bias: bool = True,
        rngs: nnx.rnglib.Rngs,
    ):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.flatten = flatten
        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding='VALID',
            use_bias=use_bias,
            rngs=rngs,
        )
        self.norm = Identity() if norm_layer is None else norm_layer(embed_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, pos_map: jnp.ndarray | None = None):
        # Accept both NHWC (preferred) and NCHW.
        if x.shape[-1] == self.in_channels:
            # Already NHWC
            b, h, w, _ = x.shape
        elif x.shape[1] == self.in_channels:
            # NCHW -> convert to NHWC
            b, _, h, w = x.shape
            x = jnp.transpose(x, (0, 2, 3, 1))
        else:
            raise ValueError(f"Expected channels={self.in_channels} on last or second axis, got shape {x.shape}")
        x = self.proj(x)                    # (B, H/P, W/P, D)
        new_h, new_w = x.shape[1:3]

        if pos_map is not None:
            pos_map = jax.image.resize(
                pos_map, (b, new_h, new_w, pos_map.shape[-1]), method="bilinear"
            )
            pos_map = pos_map.reshape(b, new_h * new_w, -1)

        if self.flatten:
            x = x.reshape(b, new_h * new_w, -1)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = jnp.transpose(x, (0, 3, 1, 2))  # NCHW

        return x, pos_map


class UnPatch(nnx.Module):
    def __init__(
        self,
        patch_size: int = 4,
        input_dim: int = 512,
        out_channels: int = 3,
        *,
        proj: bool = True,
        rngs: nnx.rnglib.Rngs,
    ):
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.out_channels = out_channels
        self.linear = (
            Identity()
            if not proj
            else nnx.Linear(
                self.input_dim,
                self.patch_size**2 * self.out_channels,
                rngs=rngs
            )
        )

    def __call__(
        self,
        x: jnp.ndarray,
        axis1: int | None = None,
        axis2: int | None = None,
        loss_mask: jnp.ndarray | None = None,
    ):
        b, n, _ = x.shape
        p = q = self.patch_size

        if axis1 is None and axis2 is None:
            w = h = int(jnp.sqrt(n).item())
            assert h * w == n
        else:
            h = axis1 // p if axis1 else n // (axis2 // p)
            w = axis2 // p if axis2 else n // h
            assert h * w == n

        x = self.linear(x)  # (B, N, p^2 * C)

        if loss_mask is not None:
            loss_mask = loss_mask[..., None]
            x = jnp.where(loss_mask, x, lax.stop_gradient(x))

        x = x.reshape(b, h, w, p, q, self.out_channels)
        # Output NHWC (B, H, W, C) to match input format
        x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, H, P, W, Q, C)
        x = x.reshape(b, h * p, w * q, self.out_channels)
        return x
