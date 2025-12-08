#!/usr/bin/env python3
"""
Debug script to verify image-caption matching.

Loads 10 samples from both HuggingFace and GCS Parquet,
shows the image_id, caption, and saves images for visual verification.

Usage:
    python debug_caption_matching.py
"""

import os
import io
import numpy as np
from pathlib import Path
from PIL import Image


def extract_image_id_from_path(path: str) -> str:
    """Extract image_id from parquet path field.

    Parquet path: n03954731_53652_n03954731.JPEG
    image_id:     n03954731_53652
    """
    name = path.rsplit('.', 1)[0] if '.' in path else path
    image_id = name.rsplit('_', 1)[0]
    return image_id


def main():
    from datasets import load_dataset
    from google.cloud import storage
    import pyarrow.parquet as pq

    output_dir = Path("outputs/caption_debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Caption-Image Matching Debug")
    print("=" * 70)

    # Step 1: Load HuggingFace samples (streaming)
    print("\n[Step 1] Loading HuggingFace dataset (streaming)...")
    hf_ds = load_dataset(
        "visual-layer/imagenet-1k-vl-enriched",
        split="train",
        streaming=True
    )

    # Get first 10 samples from HuggingFace
    hf_samples = {}
    count = 0
    for sample in hf_ds:
        image_id = sample.get('image_id', '')
        if image_id:
            hf_samples[image_id] = {
                'caption': sample.get('caption_enriched', ''),
                'label': sample.get('label', -1),
                'image': sample.get('image', None)
            }
            count += 1
            if count >= 100:  # Get 100 to have enough matches
                break

    print(f"  Loaded {len(hf_samples)} HuggingFace samples")

    # Step 2: Load first parquet file from GCS
    print("\n[Step 2] Loading first GCS Parquet file...")
    client = storage.Client()
    bucket = client.bucket("rdy-tpu-data-2025")

    blob = bucket.blob("imagenet-1k/data/train-00000-of-00294.parquet")
    local_path = "/tmp/debug_parquet.parquet"
    blob.download_to_filename(local_path)

    table = pq.read_table(local_path)
    data = table.to_pydict()
    print(f"  Parquet rows: {len(data['image'])}")

    # Step 3: Match and display
    print("\n[Step 3] Matching samples...")
    print("=" * 70)

    matched = 0
    results = []

    for i, (img_data, label) in enumerate(zip(data['image'], data['label'])):
        if matched >= 10:
            break

        path = img_data.get('path', '')
        image_id = extract_image_id_from_path(path)
        image_bytes = img_data.get('bytes', None)

        if image_id in hf_samples:
            hf_data = hf_samples[image_id]

            result = {
                'index': i,
                'image_id': image_id,
                'parquet_path': path,
                'parquet_label': label,
                'hf_label': hf_data['label'],
                'hf_caption': hf_data['caption'],
            }
            results.append(result)

            # Save parquet image
            if image_bytes:
                img = Image.open(io.BytesIO(image_bytes))
                img_path = output_dir / f"{matched:02d}_{image_id}_parquet.jpg"
                img.save(img_path)
                result['parquet_image_path'] = str(img_path)

            # Save HuggingFace image
            hf_img = hf_data.get('image')
            if hf_img:
                hf_img_path = output_dir / f"{matched:02d}_{image_id}_hf.jpg"
                hf_img.save(hf_img_path)
                result['hf_image_path'] = str(hf_img_path)

            matched += 1

    # Display results
    print(f"\n{'=' * 70}")
    print(f"MATCHED SAMPLES: {matched}")
    print(f"{'=' * 70}\n")

    for r in results:
        print(f"[{r['index']}] image_id: {r['image_id']}")
        print(f"    Parquet path:  {r['parquet_path']}")
        print(f"    Parquet label: {r['parquet_label']}")
        print(f"    HF label:      {r['hf_label']}")
        print(f"    Caption:       {r['hf_caption'][:80]}..." if len(r['hf_caption']) > 80 else f"    Caption:       {r['hf_caption']}")
        print(f"    Label match:   {'✓ YES' if r['parquet_label'] == r['hf_label'] else '✗ NO'}")
        if 'parquet_image_path' in r:
            print(f"    Parquet image: {r['parquet_image_path']}")
        if 'hf_image_path' in r:
            print(f"    HF image:      {r['hf_image_path']}")
        print()

    # Summary
    print(f"{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total matched: {matched}")
    label_matches = sum(1 for r in results if r['parquet_label'] == r['hf_label'])
    print(f"  Label matches: {label_matches}/{matched}")
    print(f"\n  Images saved to: {output_dir}/")
    print(f"  Compare *_parquet.jpg vs *_hf.jpg to verify same image")

    # Cleanup
    os.remove(local_path)

    print(f"\n{'=' * 70}")
    print("Done! Check the images in outputs/caption_debug/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
