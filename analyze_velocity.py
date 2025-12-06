#!/usr/bin/env python3
"""
Analyze velocity distribution from training data.

Computes v = x_1 - x_0 where:
- x_0: VAE latent * scaling_factor
- x_1: standard Gaussian noise

This helps verify what the model should be predicting.
"""

import torch
import numpy as np
from pathlib import Path


def analyze_velocity(pt_file: str, num_samples: int = 100, vae_scale: float = 0.13025):
    """Analyze velocity distribution from PT file."""

    print("=" * 60)
    print("Velocity Distribution Analysis")
    print("=" * 60)

    # Load latents
    print(f"\nLoading {pt_file}...")
    pt_data = torch.load(pt_file, map_location="cpu")
    latents = pt_data['latents']  # Shape varies

    print(f"  Latents shape: {latents.shape}")
    print(f"  Latents dtype: {latents.dtype}")

    # Handle different shapes
    if len(latents.shape) == 5:  # (N, frames, C, H, W)
        print(f"  Detected video format, using first frame")
        latents = latents[:, 0]  # (N, C, H, W)

    # Use subset
    latents = latents[:num_samples].float()
    print(f"  Using {len(latents)} samples")

    # Raw latent stats (before scaling)
    print(f"\n--- Raw VAE Latents (x_0_raw) ---")
    print(f"  Range: [{latents.min():.4f}, {latents.max():.4f}]")
    print(f"  Mean: {latents.mean():.4f}")
    print(f"  Std: {latents.std():.4f}")

    # Scaled latent (what the model sees as x_0)
    x_0 = latents * vae_scale
    print(f"\n--- Scaled Latents (x_0 = raw * {vae_scale}) ---")
    print(f"  Range: [{x_0.min():.4f}, {x_0.max():.4f}]")
    print(f"  Mean: {x_0.mean():.4f}")
    print(f"  Std: {x_0.std():.4f}")

    # Noise (x_1)
    x_1 = torch.randn_like(x_0)
    print(f"\n--- Noise (x_1 ~ N(0,1)) ---")
    print(f"  Range: [{x_1.min():.4f}, {x_1.max():.4f}]")
    print(f"  Mean: {x_1.mean():.4f}")
    print(f"  Std: {x_1.std():.4f}")

    # Velocity (what model should predict)
    v = x_1 - x_0
    print(f"\n--- Velocity (v = x_1 - x_0) ---")
    print(f"  Range: [{v.min():.4f}, {v.max():.4f}]")
    print(f"  Mean: {v.mean():.4f}")
    print(f"  Std: {v.std():.4f}")

    # Per-channel stats
    print(f"\n--- Per-Channel Velocity Stats ---")
    for c in range(v.shape[1]):
        v_c = v[:, c]
        print(f"  Channel {c}: mean={v_c.mean():.4f}, std={v_c.std():.4f}, "
              f"range=[{v_c.min():.4f}, {v_c.max():.4f}]")

    # Interpolated samples at different t
    print(f"\n--- Interpolated Samples x_t = (1-t)*x_0 + t*x_1 ---")
    for t in [0.0, 0.2, 0.5, 0.8, 1.0]:
        x_t = (1 - t) * x_0 + t * x_1
        print(f"  t={t:.1f}: range=[{x_t.min():.4f}, {x_t.max():.4f}], "
              f"mean={x_t.mean():.4f}, std={x_t.std():.4f}")

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"  Model input x_t should be in range ~ [{x_0.min():.2f}, {x_0.max():.2f}] to [-3, 3]")
    print(f"  Model output v should be in range ~ [{v.min():.2f}, {v.max():.2f}]")
    print(f"  Model output v should have mean ~ {v.mean():.4f} (ideally 0)")
    print(f"  Model output v should have std ~ {v.std():.4f}")
    print("=" * 60)

    return {
        'x_0_range': (float(x_0.min()), float(x_0.max())),
        'x_0_mean': float(x_0.mean()),
        'x_0_std': float(x_0.std()),
        'v_range': (float(v.min()), float(v.max())),
        'v_mean': float(v.mean()),
        'v_std': float(v.std()),
    }


if __name__ == "__main__":
    import sys

    # Default PT file path
    pt_file = "data/000000-000009.pt"

    if len(sys.argv) > 1:
        pt_file = sys.argv[1]

    if not Path(pt_file).exists():
        print(f"PT file not found: {pt_file}")
        print("Usage: python analyze_velocity.py [path/to/file.pt]")
        sys.exit(1)

    analyze_velocity(pt_file)
