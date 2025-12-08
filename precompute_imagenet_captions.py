#!/usr/bin/env python3
"""
Pre-compute Gemma-3 embeddings for ImageNet enriched captions.

IMPORTANT: Captions are extracted in PARQUET FILE ORDER to match training data.
This ensures caption embedding index matches image index during training.

Workflow:
1. Load HuggingFace dataset → create image_id → caption mapping
2. Load GCS parquet files in sorted order (same as training)
3. For each image, extract path → image_id → caption
4. Compute embeddings in parquet order
5. Result: embeddings[i] = caption for parquet image i

Usage:
    python precompute_imagenet_captions.py --upload
"""

import os
import gc
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration"""
    # HuggingFace dataset (for captions)
    hf_dataset_name: str = "visual-layer/imagenet-1k-vl-enriched"
    hf_split: str = "train"

    # GCS Parquet (training data source - defines order)
    gcs_bucket: str = "rdy-tpu-data-2025"
    gcs_data_prefix: str = "imagenet-1k/data"
    parquet_pattern: str = "train-*.parquet"

    # Output
    output_dir: str = "data/imagenet_captions"
    embeddings_file: str = "imagenet_caption_embeddings.npy"
    captions_file: str = "imagenet_captions.json"
    gcs_output_prefix: str = "imagenet-1k"

    # Gemma-3 settings
    embedding_dim: int = 640
    batch_size: int = 64
    max_length: int = 128


def build_caption_mapping(config: Config) -> Dict[str, str]:
    """Build image_id → caption mapping from HuggingFace dataset."""
    from datasets import load_dataset

    print(f"\n[Step 1] Building caption mapping from HuggingFace...")
    print(f"  Dataset: {config.hf_dataset_name}")

    ds = load_dataset(config.hf_dataset_name, split=config.hf_split)
    print(f"  Loaded {len(ds):,} samples")
    print(f"  Columns: {ds.column_names}")

    caption_map = {}
    missing = 0

    for sample in tqdm(ds, desc="  Building mapping"):
        image_id = sample.get('image_id', '')
        caption = sample.get('caption_enriched', '')

        if not image_id:
            continue

        if not caption or caption.strip() == '':
            label = sample.get('label', 0)
            caption = f"class {label}"
            missing += 1

        caption_map[image_id] = caption.strip()

    print(f"  Mapping size: {len(caption_map):,}")
    print(f"  Missing captions (fallback to label): {missing:,}")

    return caption_map


def extract_image_id_from_path(path: str) -> str:
    """Extract image_id from parquet path field.

    Parquet path format: n03954731_53652_n03954731.JPEG
    HuggingFace image_id: n03954731_53652

    Extract by removing the last underscore and everything after.
    """
    # Remove extension first
    name = path.rsplit('.', 1)[0] if '.' in path else path
    # Split at last underscore to remove synset suffix
    image_id = name.rsplit('_', 1)[0]
    return image_id


def extract_captions_in_parquet_order(caption_map: Dict[str, str], config: Config) -> List[str]:
    """Extract captions in the exact order of parquet files (same as training)."""
    from google.cloud import storage
    import pyarrow.parquet as pq

    print(f"\n[Step 2] Extracting captions in parquet order...")
    print(f"  GCS: gs://{config.gcs_bucket}/{config.gcs_data_prefix}/")

    client = storage.Client()
    bucket = client.bucket(config.gcs_bucket)

    # List parquet files (sorted - same as training)
    blobs = list(bucket.list_blobs(prefix=f"{config.gcs_data_prefix}/train-"))
    parquet_files = sorted([b.name for b in blobs if b.name.endswith('.parquet')])
    print(f"  Found {len(parquet_files)} parquet files")

    captions_ordered = []
    matched = 0
    unmatched = 0

    for pq_file in tqdm(parquet_files, desc="  Processing parquet files"):
        # Download parquet file
        local_path = f"/tmp/{Path(pq_file).name}"
        blob = bucket.blob(pq_file)
        blob.download_to_filename(local_path)

        # Read parquet
        table = pq.read_table(local_path)
        data = table.to_pydict()

        # Extract captions in row order
        for img_data, label in zip(data['image'], data['label']):
            path = img_data.get('path', '')
            image_id = extract_image_id_from_path(path)

            if image_id in caption_map:
                caption = caption_map[image_id]
                matched += 1
            else:
                caption = f"class {label}"
                unmatched += 1

            captions_ordered.append(caption)

        # Cleanup
        os.remove(local_path)

    print(f"\n  Total captions: {len(captions_ordered):,}")
    print(f"  Matched: {matched:,} ({100*matched/len(captions_ordered):.1f}%)")
    print(f"  Unmatched (fallback to class): {unmatched:,}")

    return captions_ordered


def save_captions(captions: List[str], config: Config):
    """Save captions to JSON file."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    captions_path = output_dir / config.captions_file
    with open(captions_path, 'w', encoding='utf-8') as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    print(f"\n[Save] Captions saved to: {captions_path}")


def compute_gemma_embeddings(captions: List[str], config: Config) -> np.ndarray:
    """Compute Gemma-3 270M embeddings for all captions."""
    print(f"\n[Step 3] Computing Gemma-3 embeddings...")

    try:
        from gemma import gm
        import jax.numpy as jnp

        print("  Loading Gemma-3 270M...")
        model = gm.nn.Gemma3_270M()
        params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_270M_PT)
        tokenizer = gm.text.Gemma3Tokenizer()
        print("  Model loaded!")

        all_embeddings = []

        for i in tqdm(range(0, len(captions), config.batch_size), desc="  Encoding"):
            batch_captions = captions[i:i + config.batch_size]
            batch_embeddings = []

            for caption in batch_captions:
                tokens = tokenizer.encode(caption, add_bos=True)
                if len(tokens) > config.max_length:
                    tokens = tokens[:config.max_length]
                else:
                    tokens = tokens + [0] * (config.max_length - len(tokens))

                tokens_array = np.array([tokens], dtype=np.int32)

                out = model.apply(
                    {'params': params},
                    tokens=tokens_array,
                    return_last_only=False,
                    return_hidden_states=True,
                )
                last_hidden = out.hidden_states[-1]

                # Mean pooling
                mask = (tokens_array != 0).astype(np.float32)
                mask_expanded = mask[:, :, None]
                sum_embeddings = np.sum(np.array(last_hidden) * mask_expanded, axis=1)
                sum_mask = np.clip(mask.sum(axis=1, keepdims=True), a_min=1e-9, a_max=None)
                embedding = sum_embeddings / sum_mask

                # L2 normalize
                norm = np.linalg.norm(embedding, axis=1, keepdims=True)
                embedding = embedding / np.maximum(norm, 1e-8)

                batch_embeddings.append(embedding[0])

            all_embeddings.extend(batch_embeddings)

            if (i // config.batch_size) % 100 == 0:
                gc.collect()

        embeddings = np.array(all_embeddings, dtype=np.float32)
        print(f"  Embeddings shape: {embeddings.shape}")

        del model, params, tokenizer
        gc.collect()

        return embeddings

    except ImportError as e:
        print(f"  Warning: Gemma not available ({e})")
        print("  Using fallback: sentence-transformers")
        return compute_fallback_embeddings(captions, config)


def compute_fallback_embeddings(captions: List[str], config: Config) -> np.ndarray:
    """Fallback: Use sentence-transformers if Gemma not available."""
    from sentence_transformers import SentenceTransformer

    print("  Loading sentence-transformers model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(
        captions,
        batch_size=config.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    if embeddings.shape[1] != config.embedding_dim:
        print(f"  Warning: Embedding dim {embeddings.shape[1]} != target {config.embedding_dim}")
        if embeddings.shape[1] < config.embedding_dim:
            pad = np.zeros((embeddings.shape[0], config.embedding_dim - embeddings.shape[1]))
            embeddings = np.concatenate([embeddings, pad], axis=1)
        else:
            embeddings = embeddings[:, :config.embedding_dim]

    return embeddings.astype(np.float32)


def save_embeddings(embeddings: np.ndarray, config: Config):
    """Save embeddings to file."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / config.embeddings_file
    np.save(embeddings_path, embeddings)
    print(f"\n[Save] Embeddings saved to: {embeddings_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Size: {embeddings.nbytes / 1024 / 1024:.1f} MB")


def debug_embeddings(captions: List[str], embeddings: np.ndarray, sample_indices: List[int] = None):
    """Debug: Show sample captions and embeddings."""
    if sample_indices is None:
        n = len(captions)
        sample_indices = [0, 100, 1000, 10000, n // 2, n - 1]
        sample_indices = [i for i in sample_indices if i < n]

    print(f"\n{'=' * 60}")
    print("[Debug] Caption Embeddings Verification")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(captions):,}")
    print(f"Embedding shape: {embeddings.shape}")

    for idx in sample_indices:
        caption = captions[idx][:100] + "..." if len(captions[idx]) > 100 else captions[idx]
        emb = embeddings[idx]
        print(f"\n[{idx}] '{caption}'")
        print(f"  embedding[:5] = {emb[:5]}")
        print(f"  norm = {np.linalg.norm(emb):.4f}")

    # Diversity check
    print(f"\n[Diversity Check]")
    if len(sample_indices) >= 2:
        for i in range(min(3, len(sample_indices) - 1)):
            emb1 = embeddings[sample_indices[i]]
            emb2 = embeddings[sample_indices[i + 1]]
            cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            print(f"  Cosine sim [{sample_indices[i]}] vs [{sample_indices[i+1]}]: {cos_sim:.4f}")

    rng = np.random.default_rng(42)
    rand_pairs = rng.choice(len(embeddings), size=(10, 2), replace=False)
    sims = []
    for i, j in rand_pairs:
        e1, e2 = embeddings[i], embeddings[j]
        sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        sims.append(sim)
    mean_sim = np.mean(sims)
    print(f"  Mean cosine sim (10 random pairs): {mean_sim:.4f}")

    if mean_sim > 0.99:
        print(f"  [WARNING] Embeddings may be identical! Check encoding.")
    else:
        print(f"  [OK] Embeddings have proper diversity.")


def upload_to_gcs(config: Config):
    """Upload embeddings to GCS."""
    from google.cloud import storage

    print("\n[GCS] Uploading to GCS...")

    client = storage.Client()
    bucket = client.bucket(config.gcs_bucket)

    output_dir = Path(config.output_dir)

    files_to_upload = [
        config.embeddings_file,
        config.captions_file,
    ]

    for filename in files_to_upload:
        local_path = output_dir / filename
        if local_path.exists():
            gcs_path = f"{config.gcs_output_prefix}/{filename}"
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_path))
            print(f"  Uploaded: gs://{config.gcs_bucket}/{gcs_path}")
        else:
            print(f"  Skip (not found): {local_path}")


def create_verification_mapping(captions: List[str], config: Config):
    """Create a verification mapping file."""
    output_dir = Path(config.output_dir)

    mapping = {
        'num_samples': len(captions),
        'order': 'parquet_file_sorted_order',
        'sample_captions': {
            0: captions[0] if len(captions) > 0 else '',
            100: captions[100] if len(captions) > 100 else '',
            1000: captions[1000] if len(captions) > 1000 else '',
            10000: captions[10000] if len(captions) > 10000 else '',
        }
    }

    mapping_path = output_dir / "embedding_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"  Saved mapping to: {mapping_path}")


def main():
    parser = argparse.ArgumentParser(description='Pre-compute ImageNet caption embeddings (parquet order)')
    parser.add_argument('--upload', action='store_true', help='Upload to GCS after computation')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for embedding')
    parser.add_argument('--output-dir', type=str, default='data/imagenet_captions', help='Output directory')
    parser.add_argument('--no-debug', action='store_true', help='Skip debug verification')
    args = parser.parse_args()

    config = Config(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    print("=" * 60)
    print("ImageNet Caption Embedding Pre-computation")
    print("(Parquet Order - matches training data)")
    print("=" * 60)
    print(f"HuggingFace: {config.hf_dataset_name}")
    print(f"GCS Parquet: gs://{config.gcs_bucket}/{config.gcs_data_prefix}/")
    print(f"Output: {config.output_dir}")
    print("=" * 60)

    # Step 1: Build caption mapping from HuggingFace
    caption_map = build_caption_mapping(config)

    # Step 2: Extract captions in parquet order
    captions = extract_captions_in_parquet_order(caption_map, config)

    # Step 2.5: Save captions
    save_captions(captions, config)

    # Step 3: Compute embeddings
    embeddings = compute_gemma_embeddings(captions, config)

    # Step 3.5: Debug verification
    if not args.no_debug:
        debug_embeddings(captions, embeddings)

    # Step 4: Save embeddings
    save_embeddings(embeddings, config)

    # Step 5: Create verification mapping
    create_verification_mapping(captions, config)

    # Step 6: Upload to GCS (optional)
    if args.upload:
        upload_to_gcs(config)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nEmbeddings are in PARQUET ORDER (same as training).")
    print(f"embeddings[i] = caption embedding for parquet image i")
    print(f"\nTo use in training:")
    print(f"  1. Set use_captions=True in train_imagenet_tpu.py")
    print(f"  2. Embeddings path: gs://{config.gcs_bucket}/{config.gcs_output_prefix}/{config.embeddings_file}")


if __name__ == "__main__":
    main()
