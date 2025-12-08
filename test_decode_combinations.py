#!/usr/bin/env python3
"""
Test all possible VAE decode combinations to find the correct one.

Tests:
1. VAE scaling: / 0.13025, * 0.13025, none
2. Transpose: NHWC->NCHW, none
3. Output normalization: (x+1)/2, x/2+0.5, clamp only
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# Patch jax.monitoring for older JAX versions
import jax
if not hasattr(jax, 'monitoring'):
    class _DummyMonitoring:
        @staticmethod
        def record_scalar(*args, **kwargs):
            pass
    jax.monitoring = _DummyMonitoring()
elif not hasattr(jax.monitoring, 'record_scalar'):
    jax.monitoring.record_scalar = lambda *args, **kwargs: None

import torch
from diffusers import AutoencoderKL
from PIL import Image

# Import from inference.py
sys.path.insert(0, str(Path(__file__).parent))
from inference import (
    setup_device,
    download_latest_checkpoint,
    load_checkpoint,
    create_model,
    restore_model,
    get_text_embedding,
    sample_rectified_flow,
)

SDXL_VAE_SCALE = 0.13025


def decode_with_options(
    latent: np.ndarray,
    output_path: str,
    scale_mode: str = "divide",  # "divide", "multiply", "none"
    transpose_mode: str = "nhwc_to_nchw",  # "nhwc_to_nchw", "none", "nchw_to_nhwc"
    output_mode: str = "standard",  # "standard", "direct", "abs"
):
    """Decode latent with different options."""

    # Force CPU to avoid OOM (JAX holds GPU memory)
    device = "cpu"
    vae = AutoencoderKL.from_pretrained("stabilityai/SDXL-VAE").to(device)
    vae.eval()

    # 1. Apply scaling
    if scale_mode == "divide":
        latent = latent / SDXL_VAE_SCALE
    elif scale_mode == "multiply":
        latent = latent * SDXL_VAE_SCALE
    # else: none - no scaling

    # 2. Apply transpose
    if transpose_mode == "nhwc_to_nchw":
        # (B, H, W, C) -> (B, C, H, W)
        latent = np.transpose(latent, (0, 3, 1, 2))
    elif transpose_mode == "nchw_to_nhwc":
        # (B, C, H, W) -> (B, H, W, C)
        latent = np.transpose(latent, (0, 2, 3, 1))
    # else: none - no transpose

    latent_tensor = torch.from_numpy(latent).float().to(device)

    print(f"    Latent shape for VAE: {latent_tensor.shape}")
    print(f"    Latent range: [{latent_tensor.min():.2f}, {latent_tensor.max():.2f}]")

    # Decode
    with torch.no_grad():
        try:
            decoded = vae.decode(latent_tensor).sample
        except Exception as e:
            print(f"    ERROR: {e}")
            return None

    # 3. Convert to image
    image = decoded[0].permute(1, 2, 0).cpu().numpy()

    if output_mode == "standard":
        # Standard: [-1, 1] -> [0, 255]
        image = np.clip((image + 1) / 2 * 255, 0, 255).astype(np.uint8)
    elif output_mode == "direct":
        # Direct clamp: assume already in [0, 1] or similar
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    elif output_mode == "abs":
        # Absolute value
        image = np.clip(np.abs(image) * 255, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(image)
    pil_image.save(output_path)
    print(f"    Saved: {output_path}")

    # Cleanup
    del vae, latent_tensor, decoded
    if device == "cuda":
        torch.cuda.empty_cache()

    return pil_image


def run_all_combinations(prompt: str = "wine bottle", num_steps: int = 100, seed: int = 42):
    """Run CFG sweep from 0 to 9 with 0.5 step."""
    import gc
    from datetime import datetime

    output_dir = Path("outputs/cfg_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    # CFG values: 0, 0.5, 1.0, ..., 9.0
    cfg_values = [x * 0.5 for x in range(19)]  # 0 to 9 in 0.5 steps

    print("=" * 60)
    print("CFG Sweep Test (0 to 9, step 0.5)")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print(f"CFG values: {cfg_values}")
    print(f"Total: {len(cfg_values)} images")

    # Setup
    device_type, _ = setup_device()
    checkpoint_path = download_latest_checkpoint()
    ckpt_data = load_checkpoint(checkpoint_path)
    config = ckpt_data.get('config', {})
    context_dim = config.get('context_dim', 640)

    # Get text embedding (once)
    print(f"\n[Embedding] '{prompt}'")
    text_embedding = get_text_embedding(prompt, embedding_dim=context_dim, device_type=device_type)

    # Create and restore model (once)
    print(f"\n[Model] Creating and restoring...")
    model = create_model(config)
    use_bfloat16 = device_type == 'gpu'
    restore_model(model, ckpt_data, use_bfloat16=use_bfloat16)

    # Sample for each CFG value
    print(f"\n{'=' * 60}")
    print(f"Sampling with {num_steps} steps, seed={seed}")
    print(f"{'=' * 60}")

    latents = {}
    for cfg in cfg_values:
        print(f"\n--- CFG {cfg:.1f} ---")
        latent = sample_rectified_flow(
            model=model,
            text_embedding=text_embedding,
            num_steps=num_steps,
            cfg_scale=cfg,
            seed=seed,
            device_type=device_type,
        )
        latents[cfg] = np.array(latent)  # Convert to numpy immediately

    # Release JAX model from GPU memory BEFORE loading PyTorch VAE
    print(f"\n[Memory] Releasing JAX model from GPU...")
    del model, ckpt_data, text_embedding
    jax.clear_caches()
    gc.collect()

    # Clear JAX backend buffers
    try:
        backend = jax.extend.backend.get_backend()
        for buf in backend.live_buffers():
            buf.delete()
    except Exception as e:
        print(f"  Warning: {e}")
    gc.collect()

    import time
    time.sleep(1)
    print("[Memory] JAX model released")

    # Decode all latents with VAE (CPU to avoid OOM)
    print(f"\n{'=' * 60}")
    print("Decoding with VAE (CPU)...")
    print(f"{'=' * 60}")

    device = "cpu"
    vae = AutoencoderKL.from_pretrained("stabilityai/SDXL-VAE").to(device)
    vae.eval()

    images = {}
    for cfg, latent in latents.items():
        print(f"  Decoding CFG {cfg:.1f}...")

        # Scale and transpose
        latent_scaled = latent / SDXL_VAE_SCALE
        latent_nchw = np.transpose(latent_scaled, (0, 3, 1, 2))
        latent_tensor = torch.from_numpy(latent_nchw).float().to(device)

        with torch.no_grad():
            decoded = vae.decode(latent_tensor).sample

        image = decoded[0].permute(1, 2, 0).cpu().numpy()
        image = np.clip((image + 1) / 2 * 255, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image)

        output_path = output_dir / f"cfg_{cfg:.1f}.png"
        pil_image.save(output_path)
        images[cfg] = pil_image
        print(f"    Saved: {output_path}")

    del vae
    gc.collect()

    # Create comparison grid
    print(f"\n[Grid] Creating comparison grid...")
    try:
        cols = 5
        rows = (len(images) + cols - 1) // cols
        first_img = list(images.values())[0]
        img_w, img_h = first_img.size
        label_height = 30

        grid = Image.new('RGB', (cols * img_w, rows * (img_h + label_height)), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(grid)

        for idx, (cfg, img) in enumerate(sorted(images.items())):
            row = idx // cols
            col = idx % cols
            x = col * img_w
            y = row * (img_h + label_height)
            draw.text((x + 10, y + 5), f"CFG {cfg:.1f}", fill='black')
            grid.paste(img, (x, y + label_height))

        grid_path = output_dir / "comparison_grid.png"
        grid.save(grid_path)
        print(f"  Grid saved: {grid_path}")
    except Exception as e:
        print(f"  Warning: Could not create grid: {e}")

    print(f"\n{'=' * 60}")
    print(f"Done! {len(images)} images saved to {output_dir}/")
    print(f"{'=' * 60}")


def test_with_pt_file(pt_file: str = None):
    """Test decode combinations with raw PT file latent (ground truth)."""

    output_dir = Path("outputs/decode_test_pt")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VAE Decode Test with PT File (Ground Truth)")
    print("=" * 60)

    if pt_file is None:
        # Try to find a PT file
        pt_files = list(Path(".").glob("**/*.pt"))
        if not pt_files:
            print("No PT file found. Please specify a PT file path.")
            return
        pt_file = str(pt_files[0])

    print(f"\n[Load] {pt_file}")
    pt_data = torch.load(pt_file, map_location="cpu")
    latents = pt_data['latents']

    print(f"  Shape: {latents.shape}")
    print(f"  Dtype: {latents.dtype}")
    print(f"  Range: [{latents.min():.4f}, {latents.max():.4f}]")

    # Take first sample
    if len(latents.shape) == 5:  # (N, frames, C, H, W)
        latent = latents[0, 0].float().numpy()  # (C, H, W)
    elif len(latents.shape) == 4:  # (N, C, H, W)
        latent = latents[0].float().numpy()  # (C, H, W)
    else:
        latent = latents[0].float().numpy()

    # PT file is NCHW format, add batch dim
    if latent.shape[0] == 4:  # (C, H, W)
        latent = latent[np.newaxis, ...]  # (1, C, H, W)

    print(f"  Sample shape: {latent.shape}")

    # Test combinations for PT file (which is NCHW)
    scale_modes = ["divide", "multiply", "none"]
    output_modes = ["standard", "direct"]

    print(f"\n{'=' * 60}")
    print("Testing decode combinations (PT file is NCHW)...")
    print(f"{'=' * 60}")

    for scale_mode in scale_modes:
        for output_mode in output_modes:
            combo_name = f"pt_scale_{scale_mode}_out_{output_mode}"
            output_path = output_dir / f"{combo_name}.png"

            print(f"\n[Test] {combo_name}")

            try:
                latent_copy = latent.copy()

                # Apply scaling
                if scale_mode == "divide":
                    latent_copy = latent_copy / SDXL_VAE_SCALE
                elif scale_mode == "multiply":
                    latent_copy = latent_copy * SDXL_VAE_SCALE

                # PT is already NCHW, no transpose needed
                decode_with_options(
                    latent_copy,
                    str(output_path),
                    scale_mode="none",  # Already applied above
                    transpose_mode="none",  # Already NCHW
                    output_mode=output_mode,
                )

            except Exception as e:
                print(f"    ERROR: {e}")

    print(f"\n[Done] Results saved to {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test VAE decode combinations')
    parser.add_argument('--pt-file', type=str, default=None,
                        help='Test with PT file (ground truth)')
    parser.add_argument('--prompt', '-p', type=str, default='wine bottle',
                        help='Text prompt for sampling')
    parser.add_argument('--steps', '-s', type=int, default=100,
                        help='Number of sampling steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    if args.pt_file:
        test_with_pt_file(args.pt_file)
    else:
        run_all_combinations(
            prompt=args.prompt,
            num_steps=args.steps,
            seed=args.seed,
        )
