"""
TPU-based Pre-compute Embeddings (Distributed)

BERT-base (768d) 모델을 TPU에서 실행하여 임베딩 계산
- 4개 Worker 분산 처리 (각 Worker가 다른 PT 파일 처리)
- Flax 네이티브 지원
- 각 Worker는 로컬 4개 TPU 칩 사용

Usage:
  # 각 Worker에서 실행 (TPU_WORKER_ID 환경변수로 구분)
  TPU_WORKER_ID=0 python precompute_embeddings_tpu.py
  TPU_WORKER_ID=1 python precompute_embeddings_tpu.py
  ...
"""

import os

# 각 Worker가 독립적으로 로컬 TPU만 사용하도록 설정
# (TPU pod 전체 동기화 방지)
os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
import gc
import time
import functools
import torch
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from google.cloud import storage
from dataclasses import dataclass
from tqdm import tqdm

# JAX TPU mode
import jax
import jax.numpy as jnp

# TPU 초기화 확인
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")


@dataclass
class TPUPrecomputeConfig:
    """Pre-compute 설정 (분산)"""
    gcs_bucket: str = "rdy-tpu-data-2025"
    input_prefix: str = "coyo11m-256px-ccrop-latent/latents-3crop/"
    output_prefix: str = "coyo11m-256px-ccrop-latent/latents-3crop-emb/"
    parquet_path: str = "gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/coyo11m-meta.parquet"

    # BERT-base has same 768d as Gemma 300M
    model_name: str = "bert-base-uncased"
    embedding_dim: int = 768
    max_length: int = 128  # Shorter = faster

    # TPU batch (larger batch for TPU)
    batch_size: int = 2048  # TPU can handle large batches

    local_cache: str = "/tmp/precompute_cache_tpu"

    # 분산 설정
    num_workers: int = 4  # TPU pod worker 수


class FlaxEmbeddingModel:
    """Flax BERT embedding model for TPU with pmap parallelization"""

    def __init__(self, model_name: str, max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.params = None
        self.num_devices = jax.local_device_count()
        self._encode_pmap = None

    def load(self):
        """Load model on TPU"""
        if self.model is not None:
            return

        from transformers import FlaxAutoModel, AutoTokenizer

        print(f"Loading {self.model_name} for TPU ({self.num_devices} devices)...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = FlaxAutoModel.from_pretrained(self.model_name)
        print(f"  Model loaded (dim={self.model.config.hidden_size})")

        # Create pmap function for parallel execution
        def encode_fn(input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Mean pooling
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask[:, :, None].astype(jnp.float32)
            sum_embeddings = jnp.sum(token_embeddings * mask_expanded, axis=1)
            sum_mask = jnp.clip(mask_expanded.sum(axis=1), a_min=1e-9)
            embeddings = sum_embeddings / sum_mask
            # L2 normalize
            norms = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / jnp.maximum(norms, 1e-8)
            return embeddings

        self._encode_pmap = jax.pmap(encode_fn)
        print(f"  pmap ready for {self.num_devices} devices")

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode batch of texts using all TPU devices"""
        self.load()

        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        batch_size = input_ids.shape[0]
        per_device = batch_size // self.num_devices

        # Pad to be divisible by num_devices
        if batch_size % self.num_devices != 0:
            pad_size = self.num_devices - (batch_size % self.num_devices)
            input_ids = np.pad(input_ids, ((0, pad_size), (0, 0)), mode='constant')
            attention_mask = np.pad(attention_mask, ((0, pad_size), (0, 0)), mode='constant')
            per_device = input_ids.shape[0] // self.num_devices

        # Reshape for pmap: (num_devices, per_device_batch, seq_len)
        input_ids = input_ids.reshape(self.num_devices, per_device, -1)
        attention_mask = attention_mask.reshape(self.num_devices, per_device, -1)

        # Run on all devices in parallel
        embeddings = self._encode_pmap(input_ids, attention_mask)

        # Reshape back: (num_devices, per_device_batch, dim) -> (total_batch, dim)
        embeddings = np.array(embeddings).reshape(-1, embeddings.shape[-1])

        # Remove padding
        if batch_size % self.num_devices != 0:
            embeddings = embeddings[:batch_size]

        return embeddings

    def encode_all(self, texts: List[str], batch_size: int = 2048) -> np.ndarray:
        """Encode all texts in batches using all TPU devices"""
        # Adjust batch size to be divisible by num_devices
        batch_size = (batch_size // self.num_devices) * self.num_devices
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="  TPU Encoding"):
            batch = texts[i:i + batch_size]
            embeddings = self.encode_batch(batch)
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)


class TPUPrecomputePipeline:
    """TPU Pre-compute Pipeline (Distributed)"""

    def __init__(self, config: TPUPrecomputeConfig, worker_id: int):
        self.config = config
        self.worker_id = worker_id
        self.client = storage.Client()
        self.bucket = self.client.bucket(config.gcs_bucket)

        # Worker별 캐시 디렉토리
        self.cache_dir = f"{config.local_cache}_worker_{worker_id}"
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self.captions_list: List[str] = []
        self.embedder = FlaxEmbeddingModel(config.model_name, config.max_length)

    def load_parquet_metadata(self):
        """Load parquet captions"""
        print(f"\n[Worker {self.worker_id}] Loading parquet metadata...")

        parquet_path = self.config.parquet_path

        if parquet_path.startswith("gs://"):
            parts = parquet_path[5:].split("/", 1)
            bucket_name = parts[0]
            blob_path = parts[1]

            local_parquet = os.path.join(self.cache_dir, "metadata.parquet")

            if not os.path.exists(local_parquet):
                print(f"  Downloading {parquet_path}...")
                bucket = self.client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                blob.download_to_filename(local_parquet)
            else:
                print(f"  Using cached parquet")

            parquet_path = local_parquet

        print(f"  Reading parquet...")
        table = pq.read_table(parquet_path, columns=['caption_llava'])
        self.captions_list = table['caption_llava'].to_pylist()
        print(f"  Loaded {len(self.captions_list):,} captions")

    def list_pt_files(self) -> List[str]:
        """List PT files assigned to this worker (round-robin)"""
        blobs = self.client.list_blobs(
            self.config.gcs_bucket,
            prefix=self.config.input_prefix
        )
        all_files = sorted([b.name for b in blobs if b.name.endswith('.pt')])

        # Round-robin 분배: Worker i는 파일 i, i+4, i+8, ... 처리
        my_files = [f for idx, f in enumerate(all_files)
                    if idx % self.config.num_workers == self.worker_id]
        return my_files

    def check_existing(self, pt_files: List[str]) -> List[str]:
        """Skip already processed files"""
        existing = set()
        blobs = self.client.list_blobs(
            self.config.gcs_bucket,
            prefix=self.config.output_prefix
        )
        for blob in blobs:
            if blob.name.endswith('.pt'):
                existing.add(os.path.basename(blob.name))

        remaining = []
        for pt_file in pt_files:
            if os.path.basename(pt_file) not in existing:
                remaining.append(pt_file)

        return remaining

    def process_single_file(self, blob_path: str) -> bool:
        """Process single PT file"""
        filename = os.path.basename(blob_path)
        local_input = os.path.join(self.cache_dir, f"input_{filename}")
        local_output = os.path.join(self.cache_dir, f"output_{filename}")

        try:
            # Download
            print(f"\n  [{filename}] Downloading...")
            blob = self.bucket.blob(blob_path)
            blob.download_to_filename(local_input)

            # Load PT
            print(f"  [{filename}] Loading PT...")
            data = torch.load(local_input, map_location='cpu')
            keys = data['keys'].numpy()
            num_samples = len(keys)
            print(f"  [{filename}] {num_samples:,} samples")

            # Get captions (keys are row indices)
            captions = []
            for key in keys:
                row_idx = int(key)
                if row_idx < len(self.captions_list):
                    caption = self.captions_list[row_idx] or ""
                else:
                    caption = ""
                captions.append(caption)

            # Compute embeddings on TPU
            print(f"  [{filename}] Computing embeddings on TPU...")
            embeddings = self.embedder.encode_all(captions, batch_size=self.config.batch_size)

            # Save as bfloat16
            embeddings_bf16 = torch.from_numpy(embeddings).to(torch.bfloat16)
            data['embeddings'] = embeddings_bf16

            print(f"  [{filename}] Saving...")
            torch.save(data, local_output)

            # Upload
            output_blob_path = self.config.output_prefix + filename
            print(f"  [{filename}] Uploading...")
            output_blob = self.bucket.blob(output_blob_path)
            output_blob.upload_from_filename(local_output)

            # Cleanup
            os.remove(local_input)
            os.remove(local_output)

            print(f"  [{filename}] Done!")
            return True

        except Exception as e:
            print(f"  [{filename}] Error: {e}")
            import traceback
            traceback.print_exc()
            for f in [local_input, local_output]:
                if os.path.exists(f):
                    os.remove(f)
            return False

    def run(self):
        """Run pipeline"""
        print("=" * 60)
        print(f"TPU Pre-compute Embeddings (Worker {self.worker_id}/{self.config.num_workers})")
        print(f"JAX backend: {jax.default_backend()}")
        print(f"TPU devices: {jax.device_count()}")
        print("=" * 60)

        self.load_parquet_metadata()

        print(f"\n[Worker {self.worker_id}] Listing PT files...")
        pt_files = self.list_pt_files()
        print(f"  Assigned: {len(pt_files)} files")

        print(f"\n[Worker {self.worker_id}] Checking existing...")
        remaining = self.check_existing(pt_files)
        print(f"  Remaining: {len(remaining)} files")

        if not remaining:
            print(f"\n[Worker {self.worker_id}] All files processed!")
            return

        print(f"\n[Worker {self.worker_id}] Processing {len(remaining)} files on TPU...")

        # Warmup JIT
        print("  Warming up JIT...")
        _ = self.embedder.encode_batch(["warmup text"] * 16)
        print("  JIT ready!")

        success, failed = 0, 0
        for i, blob_path in enumerate(remaining):
            print(f"\n--- Worker {self.worker_id}: File {i+1}/{len(remaining)} ---")
            if self.process_single_file(blob_path):
                success += 1
            else:
                failed += 1
            gc.collect()

        print("\n" + "=" * 60)
        print(f"[Worker {self.worker_id}] Done: {success} success, {failed} failed")
        print("=" * 60)


def main():
    worker_id = int(os.environ.get('TPU_WORKER_ID', '0'))
    config = TPUPrecomputeConfig()
    pipeline = TPUPrecomputePipeline(config, worker_id)
    pipeline.run()


if __name__ == "__main__":
    main()
