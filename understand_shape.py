import torch

pt_file = "000000-000009.pt"
pt_data = torch.load(pt_file, map_location="cpu")
latents = pt_data['latents']

print("="*60)
print("Understanding latent structure")
print("="*60)

print(f"\nShape: {latents.shape}")
print(f"  [0] = 119829 (samples)")
print(f"  [1] = 3 (???)")
print(f"  [2] = 4 (???)")
print(f"  [3] = 32 (spatial)")
print(f"  [4] = 32 (spatial)")

print(f"\nSDXL-VAE encodes images as:")
print(f"  Input: (B, 3, 512, 512) RGB image")
print(f"  Output: (B, 4, 64, 64) latent space")
print(f"  With scaling_factor ~0.13")

print(f"\nYour data:")
print(f"  Output: (B, 3, 4, 32, 32)")
print(f"  This is DIFFERENT from standard SDXL!")

print(f"\nHypotheses:")
print(f"  1. 3 channels, 4 frames (video): (B, C, T, H, W)")
print(f"     -> Process each of 4 frames separately")
print(f"  2. 4 channels split into 3+1: (B, C1+C2, T, H, W)")
print(f"     -> Need to understand what C1=3, C2=1 means")
print(f"  3. Already patched/processed: (B, ?, ?, 32, 32)")
print(f"     -> Different from raw VAE output")

# Check if frames are different
sample = latents[10]  # (3, 4, 32, 32)
print(f"\nSample 10 analysis:")
print(f"  Shape: {sample.shape}")

for frame_idx in range(4):
    frame = sample[:, frame_idx, :, :]  # (3, 32, 32)
    print(f"  Frame {frame_idx}: min={frame.min():.4f}, max={frame.max():.4f}, mean={frame.mean():.4f}")

print(f"\nDifferences between frames:")
for i in range(1, 4):
    diff = (latents[10, :, i, :, :] - latents[10, :, 0, :, :]).abs().mean()
    print(f"  Frame {i} vs Frame 0: mean diff = {diff:.6f}")
