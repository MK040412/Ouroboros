import pytest
import jax
from jax import numpy as jnp
from flax import nnx

from xut.modules.patch import PatchEmbed, UnPatch

def test_patchembed_flatten_true():
    B, C, H, W = 2, 3, 16, 16
    img = jax.random.normal(jax.random.key(0), (B, C, H, W))

    model = PatchEmbed(
        patch_size=4,
        in_channels=C,
        embed_dim=32,
        norm_layer=nnx.RMSNorm,
        flatten=True,
        rngs=nnx.Rngs(params=42),
    )

    x, pos_map = model(img)

    # Flatten → (B, N, D)
    assert x.ndim == 3
    B2, N, D = x.shape
    assert B2 == B
    assert N == (H // 4) * (W // 4)
    assert D == 32

    # pos_map None → None
    assert pos_map is None


def test_patchembed_flatten_false():
    B, C, H, W = 2, 3, 16, 16
    img = jax.random.normal(jax.random.key(1), (B, C, H, W))

    model = PatchEmbed(
        patch_size=4,
        in_channels=C,
        embed_dim=32,
        norm_layer=nnx.RMSNorm,
        flatten=False,
        rngs=nnx.Rngs(params=42),
    )

    x, pos_map = model(img)

    # flatten=False → (B, D, H/P, W/P)
    assert x.ndim == 4
    B2, D, Hp, Wp = x.shape
    assert B2 == B
    assert D == 32
    assert Hp == H // 4
    assert Wp == W // 4

    assert pos_map is None


def test_patchembed_with_pos_map():
    B, C, H, W = 2, 3, 16, 16
    img = jax.random.normal(jax.random.key(2), (B, C, H, W))
    pos_map = jax.random.normal(jax.random.key(3), (B, H, W, 8))  # arbitrary channel

    model = PatchEmbed(
        patch_size=4,
        in_channels=C,
        embed_dim=32,
        flatten=True,
        norm_layer=nnx.RMSNorm,
        rngs=nnx.Rngs(params=42),
    )

    x, pos_out = model(img, pos_map)

    assert pos_out.ndim == 3
    assert pos_out.shape[0] == B
    assert pos_out.shape[1] == (H // 4) * (W // 4)
    assert pos_out.shape[2] == 8


def test_unpatch_reconstruction():
    B, C, H, W = 2, 3, 16, 16
    img = jax.random.normal(jax.random.key(4), (B, C, H, W))

    patch = PatchEmbed(
        patch_size=4,
        in_channels=C,
        embed_dim=32,
        flatten=True,
        norm_layer=nnx.RMSNorm,
        rngs=nnx.Rngs(params=42),
    )

    unpatch = UnPatch(
        patch_size=4,
        input_dim=32,
        out_channels=C,
        proj=True,
        rngs=nnx.Rngs(params=42),
    )

    x_tokens, _ = patch(img)
    recon = unpatch(x_tokens)

    # Reconstruction shape must match original
    assert recon.shape == img.shape

    # Numbers won't match exactly (layer is learned) but dtype/finite check
    assert jnp.isfinite(recon).all()


def test_unpatch_no_proj():
    B, C, H, W = 2, 3, 16, 16
    img = jax.random.normal(jax.random.key(5), (B, C, H, W))

    patch = PatchEmbed(
        patch_size=4,
        in_channels=C,
        embed_dim=4 * 4 * C,  # identity unpatch possible
        flatten=True,
        norm_layer=nnx.RMSNorm,
        rngs=nnx.Rngs(params=42),
    )

    unpatch = UnPatch(
        patch_size=4,
        input_dim=4 * 4 * C,
        out_channels=C,
        proj=False,          # Identity
        rngs=nnx.Rngs(params=42),
    )

    x_tokens, _ = patch(img)
    recon = unpatch(x_tokens)

    # With correct embed_dim and proj=False, shape must match input
    assert recon.shape == img.shape