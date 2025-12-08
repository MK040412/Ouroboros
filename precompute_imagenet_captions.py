#!/usr/bin/env python3
"""
ImageNet Enriched Caption Embeddings Pre-computation (TPU Optimized)

Computes Gemma-3 270M embeddings for BLIP2-generated captions from
visual-layer/imagenet-1k-vl-enriched dataset.

IMPORTANT: Captions are extracted in PARQUET FILE ORDER to match training data.
This ensures caption embedding index matches image index during training.

Usage:
    # Run on TPU worker 0
    TPU_WORKER_ID=0 python precompute_imagenet_captions.py

    # With upload to GCS
    TPU_WORKER_ID=0 python precompute_imagenet_captions.py --upload
"""

import os

# TPU 단일 호스트 모드 (분산 동기화 방지)
os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
os.environ["TPU_HOST_BOUNDS"] = "1,1,1"

import gc
import sys
import json
import subprocess
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm

# JAX 초기화
import jax
import jax.numpy as jnp

print(f"[JAX] Devices: {jax.devices()}")
print(f"[JAX] Backend: {jax.default_backend()}")
print(f"[JAX] Local device count: {jax.local_device_count()}")
sys.stdout.flush()

# Gemma
from gemma import gm


@dataclass
class Config:
    """Configuration"""
    # HuggingFace dataset (for captions)
    hf_dataset_name: str = "visual-layer/imagenet-1k-vl-enriched"
    hf_split: str = "train"

    # GCS Parquet (training data source - defines order)
    gcs_bucket: str = "rdy-tpu-data-2025"
    gcs_data_prefix: str = "imagenet-1k/data"

    # Output
    output_dir: str = "/tmp/imagenet_captions"
    embeddings_file: str = "imagenet_caption_embeddings.npy"
    captions_file: str = "imagenet_captions.json"
    gcs_output_prefix: str = "imagenet-1k"

    # Gemma-3 settings
    embedding_dim: int = 640
    max_length: int = 128
    batch_size: int = 64

    # Checkpoint (resume support)
    checkpoint_every: int = 50000  # Save every N captions
    checkpoint_file: str = "caption_checkpoint.npz"


class GemmaEmbedder:
    """Gemma-3 270M Embedder with TPU pmap optimization"""

    def __init__(self, max_length: int = 128):
        self.max_length = max_length
        self.embedding_dim = 640
        self.model = None
        self.tokenizer = None
        self.params = None
        self.num_devices = jax.local_device_count()
        self._encode_pmap = None

    def load(self):
        """Load model and create pmap function"""
        if self.model is not None:
            return

        print(f"[Gemma] Loading Gemma-3 270M ({self.num_devices} local devices)...")
        sys.stdout.flush()

        self.model = gm.nn.Gemma3_270M()
        self.params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_270M_PT)
        self.tokenizer = gm.text.Gemma3Tokenizer()

        print(f"[Gemma] Model loaded (embedding_dim={self.embedding_dim})")

        # Create pmap function
        model = self.model
        params = self.params

        def encode_fn(tokens):
            """Extract last hidden state and mean pool"""
            out = model.apply(
                {'params': params},
                tokens=tokens,
                return_last_only=False,
                return_hidden_states=True,
            )
            last_hidden = out.hidden_states[-1]

            # Attention mask (non-zero tokens)
            mask = (tokens != 0).astype(jnp.float32)
            mask_expanded = mask[:, :, None]

            # Mean pooling
            sum_embeddings = jnp.sum(last_hidden * mask_expanded, axis=1)
            sum_mask = jnp.clip(mask.sum(axis=1, keepdims=True), a_min=1e-9)
            embeddings = sum_embeddings / sum_mask

            # L2 normalize
            norms = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / jnp.maximum(norms, 1e-8)
            return embeddings

        self._encode_pmap = jax.pmap(encode_fn)
        print(f"[Gemma] pmap ready for {self.num_devices} devices")
        sys.stdout.flush()

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode batch of texts using pmap"""
        self.load()

        # Tokenize
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_bos=True)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                tokens = tokens + [0] * (self.max_length - len(tokens))
            all_tokens.append(tokens)

        tokens_array = np.array(all_tokens, dtype=np.int32)
        batch_size = tokens_array.shape[0]
        original_batch_size = batch_size

        # Pad to be divisible by num_devices
        if batch_size % self.num_devices != 0:
            pad_size = self.num_devices - (batch_size % self.num_devices)
            tokens_array = np.pad(tokens_array, ((0, pad_size), (0, 0)), mode='constant')
            batch_size = tokens_array.shape[0]

        per_device = batch_size // self.num_devices

        # Reshape for pmap: (num_devices, per_device_batch, seq_len)
        tokens_array = tokens_array.reshape(self.num_devices, per_device, -1)

        # Run on all devices in parallel
        embeddings = self._encode_pmap(tokens_array)

        # Reshape back
        embeddings = np.array(embeddings).reshape(-1, embeddings.shape[-1])

        # Remove padding
        embeddings = embeddings[:original_batch_size]

        return embeddings.astype(np.float32)


def extract_image_id_from_path(path: str) -> str:
    """Extract image_id from parquet path field.

    Parquet path: n03954731_53652_n03954731.JPEG
    image_id:     n03954731_53652
    """
    name = path.rsplit('.', 1)[0] if '.' in path else path
    image_id = name.rsplit('_', 1)[0]
    return image_id


def build_caption_mapping(config: Config) -> Dict[str, str]:
    """Build image_id → caption mapping from HuggingFace dataset (streaming)"""
    from datasets import load_dataset

    print(f"\n[Step 1] Building caption mapping from HuggingFace...")
    print(f"  Dataset: {config.hf_dataset_name}")
    print(f"  Using STREAMING mode (memory efficient)")
    sys.stdout.flush()

    ds = load_dataset(
        config.hf_dataset_name,
        split=config.hf_split,
        streaming=True
    )

    caption_map = {}
    missing = 0
    count = 0

    for sample in tqdm(ds, desc="  Building mapping", total=1281167):
        image_id = sample.get('image_id', '')
        caption = sample.get('caption_enriched', '')

        if not image_id:
            count += 1
            continue

        if not caption or caption.strip() == '':
            label = sample.get('label', 0)
            caption = f"class {label}"
            missing += 1

        caption_map[image_id] = caption.strip()
        count += 1

    print(f"  Processed: {count:,} samples")
    print(f"  Mapping size: {len(caption_map):,}")
    print(f"  Missing captions: {missing:,}")
    sys.stdout.flush()

    return caption_map


def extract_captions_in_parquet_order(caption_map: Dict[str, str], config: Config) -> List[str]:
    """Extract captions in parquet file order (same as training)"""
    from google.cloud import storage
    import pyarrow.parquet as pq

    print(f"\n[Step 2] Extracting captions in parquet order...")
    print(f"  GCS: gs://{config.gcs_bucket}/{config.gcs_data_prefix}/")
    sys.stdout.flush()

    client = storage.Client()
    bucket = client.bucket(config.gcs_bucket)

    # List parquet files (sorted - same as training)
    blobs = list(bucket.list_blobs(prefix=f"{config.gcs_data_prefix}/train-"))
    parquet_files = sorted([b.name for b in blobs if b.name.endswith('.parquet')])
    print(f"  Found {len(parquet_files)} parquet files")
    sys.stdout.flush()

    captions_ordered = []
    matched = 0
    unmatched = 0

    for pq_idx, pq_file in enumerate(tqdm(parquet_files, desc="  Processing parquet")):
        # Download parquet file
        local_path = f"/tmp/parquet_{pq_idx % 2}.parquet"
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

    print(f"\n  Total captions: {len(captions_ordered):,}")
    print(f"  Matched: {matched:,} ({100*matched/len(captions_ordered):.1f}%)")
    print(f"  Unmatched: {unmatched:,}")
    sys.stdout.flush()

    return captions_ordered


def compute_embeddings_with_checkpoint(
    captions: List[str],
    embedder: GemmaEmbedder,
    config: Config,
    start_idx: int = 0,
    existing_embeddings: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute embeddings with checkpoint support"""
    print(f"\n[Step 3] Computing Gemma-3 embeddings...")
    print(f"  Total captions: {len(captions):,}")
    print(f"  Starting from: {start_idx:,}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Checkpoint every: {config.checkpoint_every:,}")
    sys.stdout.flush()

    # Initialize or continue from existing
    if existing_embeddings is not None:
        all_embeddings = list(existing_embeddings)
    else:
        all_embeddings = []

    # Warmup
    if start_idx == 0:
        print("  Warming up pmap...")
        _ = embedder.encode_batch(["warmup"] * max(4, embedder.num_devices))
        print("  pmap ready!")
        sys.stdout.flush()

    # Process in batches
    checkpoint_dir = Path(config.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    num_batches = (len(captions) - start_idx + config.batch_size - 1) // config.batch_size

    for batch_idx in tqdm(range(num_batches), desc="  Encoding"):
        start = start_idx + batch_idx * config.batch_size
        end = min(start + config.batch_size, len(captions))
        batch_captions = captions[start:end]

        embeddings = embedder.encode_batch(batch_captions)
        all_embeddings.extend(embeddings)

        # Checkpoint
        current_count = start_idx + len(all_embeddings) - (len(existing_embeddings) if existing_embeddings is not None else 0)
        processed_total = start + len(batch_captions)

        if processed_total % config.checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / config.checkpoint_file
            np.savez(
                checkpoint_path,
                embeddings=np.array(all_embeddings, dtype=np.float32),
                processed=processed_total
            )
            print(f"\n  Checkpoint saved: {processed_total:,}/{len(captions):,}")
            sys.stdout.flush()

        # Memory cleanup
        if batch_idx % 100 == 0:
            gc.collect()

    result = np.array(all_embeddings, dtype=np.float32)
    print(f"  Final embeddings shape: {result.shape}")
    sys.stdout.flush()

    return result


def save_results(captions: List[str], embeddings: np.ndarray, config: Config):
    """Save captions and embeddings locally"""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save captions
    captions_path = output_dir / config.captions_file
    with open(captions_path, 'w', encoding='utf-8') as f:
        json.dump(captions, f, ensure_ascii=False)
    print(f"\n[Save] Captions: {captions_path}")

    # Save embeddings
    embeddings_path = output_dir / config.embeddings_file
    np.save(embeddings_path, embeddings)
    print(f"[Save] Embeddings: {embeddings_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Size: {embeddings.nbytes / 1024 / 1024:.1f} MB")
    sys.stdout.flush()


def upload_to_gcs(config: Config):
    """Upload to GCS using gsutil"""
    print(f"\n[Upload] Uploading to GCS...")
    sys.stdout.flush()

    output_dir = Path(config.output_dir)

    files = [config.embeddings_file, config.captions_file]

    for filename in files:
        local_path = output_dir / filename
        if local_path.exists():
            gcs_path = f"gs://{config.gcs_bucket}/{config.gcs_output_prefix}/{filename}"
            print(f"  {filename} -> {gcs_path}")
            sys.stdout.flush()

            result = subprocess.run(
                ["gsutil", "cp", str(local_path), gcs_path],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"    OK")
            else:
                print(f"    FAILED: {result.stderr}")
            sys.stdout.flush()


def load_checkpoint(config: Config):
    """Load checkpoint if exists"""
    checkpoint_path = Path(config.output_dir) / config.checkpoint_file
    if checkpoint_path.exists():
        data = np.load(checkpoint_path)
        embeddings = data['embeddings']
        processed = int(data['processed'])
        print(f"[Checkpoint] Loaded: {processed:,} processed, {len(embeddings):,} embeddings")
        return embeddings, processed
    return None, 0


def main():
    parser = argparse.ArgumentParser(description='Pre-compute ImageNet caption embeddings (TPU)')
    parser.add_argument('--upload', action='store_true', help='Upload to GCS')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    worker_id = int(os.environ.get('TPU_WORKER_ID', '0'))

    config = Config(batch_size=args.batch_size)

    print("=" * 60)
    print("ImageNet Caption Embedding Pre-computation (TPU)")
    print("=" * 60)
    print(f"Worker ID: {worker_id}")
    print(f"HuggingFace: {config.hf_dataset_name}")
    print(f"GCS Parquet: gs://{config.gcs_bucket}/{config.gcs_data_prefix}/")
    print(f"Output: {config.output_dir}")
    print(f"Batch size: {config.batch_size}")
    print("=" * 60)
    sys.stdout.flush()

    # Check for checkpoint
    existing_embeddings, start_idx = None, 0
    if args.resume:
        existing_embeddings, start_idx = load_checkpoint(config)

    # Step 1: Build caption mapping
    caption_map = build_caption_mapping(config)

    # Step 2: Extract captions in parquet order
    captions = extract_captions_in_parquet_order(caption_map, config)

    # Free memory
    del caption_map
    gc.collect()

    # Step 3: Compute embeddings
    embedder = GemmaEmbedder(max_length=config.max_length)
    embeddings = compute_embeddings_with_checkpoint(
        captions, embedder, config,
        start_idx=start_idx,
        existing_embeddings=existing_embeddings
    )

    # Step 4: Save
    save_results(captions, embeddings, config)

    # Step 5: Upload (optional)
    if args.upload:
        upload_to_gcs(config)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"Embeddings in PARQUET ORDER (matches training)")
    print(f"To use: set use_captions=True in train_imagenet_tpu.py")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
