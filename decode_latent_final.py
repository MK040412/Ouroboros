#!/usr/bin/env python3
"""
Decode latent from PT file using SDXL-VAE

CORRECT METHOD:
- Merge (3, 4, 32, 32) channels into (12, 32, 32)
- Use first 4 channels for SDXL VAE
- Produces real images

OPTION: Add one step of noise before decoding
"""

import torch
import numpy as np
from diffusers import AutoencoderKL
from PIL import Image
import os
from pathlib import Path


class DiffusionSchedule:
    """Noise schedule (alphas_cumprod based) - from train_tpu_256.py"""
    
    def __init__(self, beta_min: float = 0.0001, beta_max: float = 0.02, T: int = 1000):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self._cache_alphas()
    
    def _cache_alphas(self):
        # Linear schedule
        betas = np.linspace(self.beta_min, self.beta_max, self.T)
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        # sqrt_alphas_cumprod and sqrt(1 - alphas_cumprod)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
    
    def forward_diffusion(self, x_0: np.ndarray, noise: np.ndarray, timestep: int) -> np.ndarray:
        """
        Forward diffusion: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        
        Args:
            x_0: (4, 32, 32) - original VAE latent
            noise: (4, 32, 32) - noise
            timestep: int - timestep (0 ~ T-1)
        Returns:
            x_t: (4, 32, 32) - noisy latent
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timestep]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timestep]
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t

def decode_pt_latent(
    pt_file="000000-000009.pt",
    output_dir="decoded_output",
    sample_indices=None,
    max_samples=None,
    device=None,
    add_noise=False,
    noise_timestep=500
):
    """
    Decode PT latent to images using SDXL-VAE
    
    Args:
        add_noise: If True, add one step of diffusion noise before decoding
        noise_timestep: Which timestep to use for noise (0-999)
    """
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*70)
    print("PT Latent -> Image Decoding (SDXL-VAE)")
    if add_noise:
        print(f"With noise step (timestep={noise_timestep})")
    print("="*70)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load latent
    print(f"\n1. Loading {pt_file}...")
    pt_data = torch.load(pt_file, map_location="cpu")
    latents = pt_data['latents']  # (N, 3, 4, 32, 32)
    
    print(f"   Shape: {latents.shape}")
    print(f"   Type: {latents.dtype}")
    print(f"   Range: [{latents.min():.2f}, {latents.max():.2f}]")
    
    # Load VAE
    print(f"\n2. Loading SDXL-VAE...")
    vae = AutoencoderKL.from_pretrained("KMK040412/sdxl-vae-flax-msgpack")
    vae.eval().to(device)
    
    # Init diffusion schedule if adding noise
    schedule = None
    if add_noise:
        print(f"\n2b. Initializing diffusion schedule...")
        schedule = DiffusionSchedule()
    
    # Determine samples to decode
    if max_samples is None:
        max_samples = len(latents)
    
    if sample_indices is None:
        sample_indices = list(range(min(max_samples, len(latents))))
    
    output_suffix = "_with_noise" if add_noise else ""
    print(f"\n3. Decoding {len(sample_indices)} samples{output_suffix}...")
    
    for sample_idx in sample_indices:
        # (1, 3, 4, 32, 32)
        sample = latents[sample_idx:sample_idx+1].float()
        
        # Reshape: (1, 3, 4, 32, 32) -> (1, 12, 32, 32)
        b, c, frames, h, w = sample.shape
        sample = sample.reshape(b, c * frames, h, w)
        
        # Use first 4 channels for SDXL VAE: (1, 4, 32, 32)
        sample = sample[:, :4, :, :].to(device)
        
        # Add noise step if requested
        if add_noise:
            sample_np = sample[0].float().cpu().numpy()  # (4, 32, 32) as float32
            noise = np.random.randn(*sample_np.shape).astype(np.float32)
            noisy_sample = schedule.forward_diffusion(sample_np, noise, noise_timestep)
            sample = torch.from_numpy(noisy_sample[None, ...]).float().to(device)  # (1, 4, 32, 32)
        
        # Decode
        with torch.no_grad():
            decoded = vae.decode(sample).sample
        
        # Convert to image
        image = decoded[0].cpu().permute(1, 2, 0).numpy()
        image = np.clip((image + 1) / 2 * 255, 0, 255).astype(np.uint8)
        
        # Save
        out_path = f"{output_dir}/sample_{sample_idx:06d}{output_suffix}.png"
        Image.fromarray(image).save(out_path)
        print(f"   [{sample_idx:6d}] {out_path}")
    
    print(f"\n✓ Done! {len(sample_indices)} images saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    
    # Decode options
    add_noise = "--noise" in sys.argv
    noise_timestep = 500
    
    if "--noise-step" in sys.argv:
        idx = sys.argv.index("--noise-step")
        noise_timestep = int(sys.argv[idx + 1])
    
    # Clean output dir names
    output_dir_base = "decoded_output"
    output_dir = output_dir_base + ("_noisy" if add_noise else "")
    
    if add_noise:
        print(f"\n🔊 Adding noise with timestep={noise_timestep}")
    
    # Decode first 100 samples
    decode_pt_latent(
        max_samples=100,
        output_dir=output_dir,
        add_noise=add_noise,
        noise_timestep=noise_timestep
    )
