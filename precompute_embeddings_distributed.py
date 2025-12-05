"""
Distributed Pre-compute Embeddings (4-Worker Parallel)

각 TPU VM worker에서 독립적으로 실행
- Worker 0: 파일 0, 4, 8, 12, ...
- Worker 1: 파일 1, 5, 9, 13, ...
- Worker 2: 파일 2, 6, 10, 14, ...
- Worker 3: 파일 3, 7, 11, 15, ...

Usage:
  # Worker 0에서
  TPU_WORKER_ID=0 python precompute_embeddings_distributed.py

  # 또는 run_precompute_all.sh로 전체 실행
"""

import os
import gc
import torch
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from typing import List
from google.cloud import storage
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


@dataclass
class DistributedConfig:
    """분산 Pre-compute 설정"""
    gcs_bucket: str = "rdy-tpu-data-2025"
    input_prefix: str = "coyo11m-256px-ccrop-latent/latents-3crop/"
    output_prefix: str = "coyo11m-256px-ccrop-latent/latents-3crop-emb/"
    parquet_path: str = "gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/coyo11m-meta.parquet"

    model_name: str = "bert-base-uncased"
    embedding_dim: int = 768
    max_length: int = 128
    batch_size: int = 256

    local_cache: str = "/tmp/precompute_distributed"
    num_workers: int = 8
    num_threads: int = 28  # vCPU per worker (112 / 4)


class BertEmbedder:
    """PyTorch BERT Embedder"""

    def __init__(self, model_name: str, max_length: int, num_threads: int):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

        torch.set_num_threads(num_threads)
        print(f"  PyTorch threads: {num_threads}")

    def load(self):
        if self.model is not None:
            return

        print(f"  Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        print(f"  Model loaded (dim={self.model.config.hidden_size})")

    def encode_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        self.load()

        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )

            outputs = self.model(**inputs)

            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state

            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = (sum_embeddings / sum_mask).numpy()

            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-8)

            return embeddings.astype(np.float32)

    def encode_all(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="    Encoding", leave=False):
            batch = texts[i:i + batch_size]
            embeddings = self.encode_batch(batch)
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings, axis=0)


class DistributedPrecompute:
    """분산 Pre-compute 파이프라인"""

    def __init__(self, config: DistributedConfig, worker_id: int):
        self.config = config
        self.worker_id = worker_id
        self.client = storage.Client()
        self.bucket = self.client.bucket(config.gcs_bucket)

        # Worker별 캐시 디렉토리
        self.cache_dir = f"{config.local_cache}_worker_{worker_id}"
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self.captions_list: List[str] = []
        self.embedder = BertEmbedder(config.model_name, config.max_length, config.num_threads)

    def load_parquet(self):
        print(f"\n[Worker {self.worker_id}] Loading parquet...")

        parquet_path = self.config.parquet_path
        if parquet_path.startswith("gs://"):
            parts = parquet_path[5:].split("/", 1)
            local_parquet = os.path.join(self.cache_dir, "metadata.parquet")

            if not os.path.exists(local_parquet):
                print(f"  Downloading parquet...")
                bucket = self.client.bucket(parts[0])
                blob = bucket.blob(parts[1])
                blob.download_to_filename(local_parquet)
            else:
                print(f"  Using cached parquet")

            parquet_path = local_parquet

        table = pq.read_table(parquet_path, columns=['caption_llava'])
        self.captions_list = table['caption_llava'].to_pylist()
        print(f"  Loaded {len(self.captions_list):,} captions")

    def get_my_files(self) -> List[str]:
        """이 Worker가 처리할 파일 목록"""
        blobs = self.client.list_blobs(self.config.gcs_bucket, prefix=self.config.input_prefix)
        all_files = sorted([b.name for b in blobs if b.name.endswith('.pt')])

        # Round-robin 분배
        my_files = [f for i, f in enumerate(all_files) if i % self.config.num_workers == self.worker_id]
        return my_files

    def check_existing(self, files: List[str]) -> List[str]:
        """이미 처리된 파일 제외"""
        existing = set()
        blobs = self.client.list_blobs(self.config.gcs_bucket, prefix=self.config.output_prefix)
        for blob in blobs:
            if blob.name.endswith('.pt'):
                existing.add(os.path.basename(blob.name))

        return [f for f in files if os.path.basename(f) not in existing]

    def process_file(self, blob_path: str) -> bool:
        filename = os.path.basename(blob_path)
        local_input = os.path.join(self.cache_dir, f"input_{filename}")
        local_output = os.path.join(self.cache_dir, f"output_{filename}")

        try:
            # Download
            print(f"  [{filename}] Downloading...")
            blob = self.bucket.blob(blob_path)
            blob.download_to_filename(local_input)

            # Load
            print(f"  [{filename}] Loading...")
            data = torch.load(local_input, map_location='cpu')
            keys = data['keys'].numpy()
            print(f"  [{filename}] {len(keys):,} samples")

            # Get captions
            captions = []
            for key in keys:
                idx = int(key)
                caption = self.captions_list[idx] if idx < len(self.captions_list) else ""
                captions.append(caption or "")

            # Compute embeddings
            print(f"  [{filename}] Computing embeddings...")
            embeddings = self.embedder.encode_all(captions, self.config.batch_size)

            # Save
            data['embeddings'] = torch.from_numpy(embeddings).to(torch.bfloat16)
            torch.save(data, local_output)

            # Upload
            output_path = self.config.output_prefix + filename
            print(f"  [{filename}] Uploading...")
            self.bucket.blob(output_path).upload_from_filename(local_output)

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
        print("=" * 60)
        print(f"Distributed Pre-compute (Worker {self.worker_id}/{self.config.num_workers})")
        print("=" * 60)

        self.load_parquet()

        print(f"\n[Worker {self.worker_id}] Getting file list...")
        my_files = self.get_my_files()
        print(f"  Assigned: {len(my_files)} files")

        remaining = self.check_existing(my_files)
        print(f"  Remaining: {len(remaining)} files")

        if not remaining:
            print(f"\n[Worker {self.worker_id}] All done!")
            return

        success, failed = 0, 0
        for i, blob_path in enumerate(remaining):
            print(f"\n--- Worker {self.worker_id}: File {i+1}/{len(remaining)} ---")
            if self.process_file(blob_path):
                success += 1
            else:
                failed += 1
            gc.collect()

        print(f"\n[Worker {self.worker_id}] Completed: {success} success, {failed} failed")


def main():
    worker_id = int(os.environ.get('TPU_WORKER_ID', '0'))
    config = DistributedConfig()
    pipeline = DistributedPrecompute(config, worker_id)
    pipeline.run()


if __name__ == "__main__":
    main()
