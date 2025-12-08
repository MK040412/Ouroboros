#!/usr/bin/env python3
"""
COYO Dataset Debug Tool

Pre-training verification script that:
1. Downloads a PT file from GCS
2. Loads captions from parquet
3. Decodes VAE latents to images
4. Saves images with captions for visual inspection

Usage:
    python debug_coyo_data.py --num-samples 10 --output-dir /tmp/debug_coyo

    # Skip VAE decoding (faster, just check data structure)
    python debug_coyo_data.py --num-samples 10 --no-decode
"""

import os
import sys
import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def download_pt_file(gcs_path: str, local_path: str) -> bool:
    """Download PT file from GCS"""
    from google.cloud import storage

    # Parse gs://bucket/path
    parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1] if len(parts) > 1 else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List PT files
    blobs = list(bucket.list_blobs(prefix=blob_path))
    pt_files = [b for b in blobs if b.name.endswith('.pt')]

    if not pt_files:
        print(f"[Error] No PT files found in {gcs_path}")
        return False

    # Download first PT file
    blob = pt_files[0]
    print(f"[Download] {blob.name} ({blob.size / 1024 / 1024:.1f} MB)")
    blob.download_to_filename(local_path)
    return True


def load_pt_file(path: str) -> dict:
    """Load PT file and handle bfloat16"""
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)

    result = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.bfloat16:
                value = value.to(torch.float32)
            result[key] = value.numpy()
        else:
            result[key] = np.array(value) if hasattr(value, '__iter__') else value

    return result


def load_captions_from_gcs(parquet_gcs_path: str) -> Dict[int, str]:
    """Load captions from GCS parquet"""
    from google.cloud import storage
    import pyarrow.parquet as pq

    print(f"[Parquet] Loading captions from {parquet_gcs_path}")

    # Download parquet
    parts = parquet_gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1]

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.reload()  # Get metadata including size

    local_path = "/tmp/coyo_meta.parquet"
    size_mb = blob.size / 1024 / 1024 if blob.size else 0
    print(f"[Parquet] Downloading ({size_mb:.1f} MB)...")
    blob.download_to_filename(local_path)

    # Read parquet
    table = pq.read_table(local_path, columns=['key', 'caption_llava'])

    key_to_caption = {}
    for batch in table.to_batches():
        keys = batch['key'].to_pylist()
        captions = batch['caption_llava'].to_pylist()
        for k, c in zip(keys, captions):
            key_to_caption[int(k)] = c if c else ""

    print(f"[Parquet] Loaded {len(key_to_caption):,} captions")
    return key_to_caption


def decode_latent_to_image(latent: np.ndarray, vae) -> np.ndarray:
    """Decode VAE latent to RGB image

    Args:
        latent: Shape (3, 4, 32, 32) - 3 crops, 4 channels, 32x32
        vae: AutoencoderKL model

    Returns:
        RGB image as uint8 numpy array (256, 256, 3)
    """
    import torch

    # Use first crop only: (4, 32, 32)
    latent_single = latent[0]  # First crop

    # Add batch dimension: (1, 4, 32, 32)
    latent_tensor = torch.from_numpy(latent_single).unsqueeze(0).float()

    # Move to same device as VAE
    device = next(vae.parameters()).device
    latent_tensor = latent_tensor.to(device)

    # Decode
    with torch.no_grad():
        decoded = vae.decode(latent_tensor).sample  # (1, 3, 256, 256)

    # Convert to numpy image
    image = decoded[0].cpu().permute(1, 2, 0).numpy()  # (256, 256, 3)
    image = np.clip((image + 1) / 2 * 255, 0, 255).astype(np.uint8)

    return image


def compute_embedding_stats(embeddings: np.ndarray) -> dict:
    """Compute embedding statistics"""
    norms = np.linalg.norm(embeddings, axis=1)

    # Random cosine similarities
    n = min(100, len(embeddings))
    idx1 = np.random.choice(len(embeddings), n, replace=False)
    idx2 = np.random.choice(len(embeddings), n, replace=False)

    cos_sims = []
    for i, j in zip(idx1, idx2):
        if i != j:
            cos_sim = np.dot(embeddings[i], embeddings[j]) / (norms[i] * norms[j] + 1e-8)
            cos_sims.append(cos_sim)

    return {
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "cosine_sim_mean": float(np.mean(cos_sims)) if cos_sims else 0.0,
        "cosine_sim_std": float(np.std(cos_sims)) if cos_sims else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description='COYO Dataset Debug Tool')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default='/tmp/debug_coyo', help='Output directory')
    parser.add_argument('--no-decode', action='store_true', help='Skip VAE decoding')
    parser.add_argument('--gcs-pt-path', type=str,
                        default='gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/latents-3crop-gemma-3-270m/',
                        help='GCS path to PT files')
    parser.add_argument('--gcs-parquet-path', type=str,
                        default='gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/coyo11m-meta.parquet',
                        help='GCS path to parquet file')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COYO Dataset Debug Tool")
    print("=" * 60)
    print(f"PT path: {args.gcs_pt_path}")
    print(f"Parquet: {args.gcs_parquet_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {output_dir}")
    print(f"VAE decode: {not args.no_decode}")
    print("=" * 60)
    sys.stdout.flush()

    # Step 1: Download PT file
    print("\n[Step 1] Downloading PT file...")
    pt_local = "/tmp/debug_coyo_pt.pt"
    if not download_pt_file(args.gcs_pt_path, pt_local):
        return 1

    # Step 2: Load PT file
    print("\n[Step 2] Loading PT file...")
    data = load_pt_file(pt_local)

    print(f"\n[PT File Stats]")
    print(f"  Keys: {data['keys'].shape}")
    print(f"  Latents: {data['latents'].shape} {data['latents'].dtype}")
    print(f"  Embeddings: {data['embeddings'].shape} {data['embeddings'].dtype}")
    print(f"  Latent range: [{data['latents'].min():.2f}, {data['latents'].max():.2f}]")
    print(f"  Latent mean: {data['latents'].mean():.4f}, std: {data['latents'].std():.4f}")

    # Embedding stats
    emb_stats = compute_embedding_stats(data['embeddings'])
    print(f"\n[Embedding Stats]")
    print(f"  Norm - mean: {emb_stats['norm_mean']:.4f}, std: {emb_stats['norm_std']:.4f}")
    print(f"  Norm - min: {emb_stats['norm_min']:.4f}, max: {emb_stats['norm_max']:.4f}")
    print(f"  Cosine sim (random pairs) - mean: {emb_stats['cosine_sim_mean']:.4f}, std: {emb_stats['cosine_sim_std']:.4f}")
    sys.stdout.flush()

    # Step 3: Load captions
    print("\n[Step 3] Loading captions...")
    key_to_caption = load_captions_from_gcs(args.gcs_parquet_path)

    # Step 4: Load VAE if needed
    vae = None
    if not args.no_decode:
        print("\n[Step 4] Loading SDXL VAE...")
        import torch
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sdxl-vae",
            torch_dtype=torch.float32
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vae = vae.to(device).eval()
        print(f"  VAE loaded on {device}")

    # Step 5: Process samples
    print(f"\n[Step 5] Processing {args.num_samples} samples...")

    # Random sample indices
    num_total = len(data['keys'])
    sample_indices = np.random.choice(num_total, min(args.num_samples, num_total), replace=False)

    results = []
    for i, idx in enumerate(sample_indices):
        key = int(data['keys'][idx])
        caption = key_to_caption.get(key, "[Caption not found]")
        latent = data['latents'][idx]  # (3, 4, 32, 32)
        embedding = data['embeddings'][idx]  # (640,)

        print(f"\n  [{i}] Key: {key}")
        print(f"      Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}")
        print(f"      Latent stats: mean={latent.mean():.3f}, std={latent.std():.3f}")
        print(f"      Embedding[:5]: [{', '.join([f'{x:.3f}' for x in embedding[:5]])}]")
        print(f"      Embedding norm: {np.linalg.norm(embedding):.4f}")

        result = {
            "index": int(idx),
            "key": key,
            "caption": caption,
            "latent_mean": float(latent.mean()),
            "latent_std": float(latent.std()),
            "embedding_norm": float(np.linalg.norm(embedding)),
        }

        # Decode and save image
        if vae is not None:
            from PIL import Image

            image = decode_latent_to_image(latent, vae)
            image_path = output_dir / f"sample_{i:03d}_key{key}.png"
            Image.fromarray(image).save(image_path)
            result["image_path"] = str(image_path)
            print(f"      Image saved: {image_path}")

        results.append(result)
        sys.stdout.flush()

    # Save results JSON
    results_path = output_dir / "debug_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "pt_file_stats": {
                "num_samples": int(num_total),
                "latent_shape": list(data['latents'].shape),
                "embedding_shape": list(data['embeddings'].shape),
                "latent_range": [float(data['latents'].min()), float(data['latents'].max())],
            },
            "embedding_stats": emb_stats,
            "num_captions": len(key_to_caption),
            "samples": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] Results saved to {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("[Summary]")
    print("=" * 60)
    print(f"  PT file samples: {num_total:,}")
    print(f"  Parquet captions: {len(key_to_caption):,}")
    print(f"  Samples processed: {len(results)}")

    # Check caption match rate
    matched = sum(1 for r in results if r['caption'] != "[Caption not found]")
    print(f"  Caption match rate: {matched}/{len(results)} ({100*matched/len(results):.0f}%)")

    # Unique captions
    unique_captions = len(set(r['caption'] for r in results))
    print(f"  Unique captions: {unique_captions}/{len(results)}")

    if vae is not None:
        print(f"\n  Images saved to: {output_dir}/")
        print(f"  Open images to verify data quality!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
