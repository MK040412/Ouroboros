from typing import Optional, Type

import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn

class Identity(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x

class PatchEmbed(nn.Module):
    patch_size: int = 4
    in_channels: int = 3
    embed_dim: int = 512
    norm_layer: Optional[Type[nn.Module]] = None
    flatten: bool = True
    use_bias: bool = True

    def setup(self):
        self.proj = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            use_bias=self.use_bias,
        )
        self.norm = Identity() if self.norm_layer is None else self.norm_layer()

    def __call__(self, x: jnp.ndarray, pos_map: jnp.ndarray | None = None):
        """
        NOTE! x: (B, H, W, C)
        """
        b, h, w, _ = x.shape
        x = self.proj(x) # (B, H/P, W/P, embed_dim)
        new_h, new_w = x.shape[1:3]

        if pos_map is not None:
            pos_map = jax.image.resize(
                pos_map, (b, new_h, new_w, pos_map.shape[-1]), method="bilinear"
            )
            pos_map = pos_map.reshape(b, new_h * new_w, -1)

        if self.flatten:
            x = x.reshape(b, new_h * new_w, -1)  # (B, N, D)

        x = self.norm(x)
        return x, pos_map

class UnPatch(nn.Module):
    patch_size: int = 4
    input_dim: int = 512
    out_channels: int = 3
    proj: bool = True

    def setup(self):
        self.linear = (
            nn.Dense(self.patch_size**2 * self.out_channels)
            if self.proj else nn.Identity()
        )

    def __call__(
        self,
        x: jnp.ndarray,
        axis1: int | None = None,
        axis2: int | None = None,
        loss_mask: jnp.ndarray | None = None,
    ):
        """
        NOTE! x: (B, H, W, C)
        """
        b, n, _ = x.shape
        p = q = self.patch_size

        if axis1 is None and axis2 is None:
            w = h = int(jnp.sqrt(n))
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
        x = x.transpose(0, 5, 1, 3, 2, 4)
        x = x.reshape(b, self.out_channels, h * p, w * q)
        return x

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 32, 32, 3))
    pos_map = jax.random.normal(key, (2, 32, 32, 2))

    pe = PatchEmbed(patch_size=4, in_channels=3, embed_dim=16)
    vars_pe = pe.init(key, x, pos_map)
    x_patched, pos_patched = pe.apply(vars_pe, x, pos_map)
    print(x_patched.shape)
    assert x_patched.shape == (2, 64, 16)

    up = UnPatch(patch_size=4, input_dim=16, out_channels=3)
    vars_up = up.init(key, x_patched)
    x_recon = up.apply(vars_up, x_patched)
    print(x_recon.shape)
    assert x_recon.shape == (2, 3, 32, 32)