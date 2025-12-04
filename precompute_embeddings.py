"""
Pre-compute embeddings for COYO-11M dataset

기존 PT 파일에 embeddings key를 추가하여 GCS에 재업로드
- 입력: gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/latents-3crop/*.pt
- 출력: gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/latents-3crop-emb/*.pt
"""

import os
import gc
import time
import torch
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from google.cloud import storage
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class PrecomputeConfig:
    """Pre-compute 설정"""
    gcs_bucket: str = "rdy-tpu-data-2025"
    input_prefix: str = "coyo11m-256px-ccrop-latent/latents-3crop/"
    output_prefix: str = "coyo11m-256px-ccrop-latent/latents-3crop-emb/"
    parquet_path: str = "gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/coyo11m-meta.parquet"

    model_name: str = "google/embeddinggemma-300m"
    embedding_dim: int = 768
    batch_size: int = 512  # 임베딩 배치 크기 (큰 배치 = 더 효율적)

    local_cache: str = "/tmp/precompute_cache"
    num_workers: int = 8  # 병렬 파일 처리 수


class EmbeddingComputer:
    """임베딩 계산기 (단일 프로세스용)"""

    def __init__(self, model_name: str, num_threads: int = 56):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.num_threads = num_threads  # 112 vCPU / 2 = 56

    def load_model(self):
        """모델 로드"""
        if self.model is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(self.num_threads)

        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        print(f"  Model loaded (using {self.num_threads} threads)")

    def encode_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """배치 인코딩"""
        self.load_model()

        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
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
        """전체 텍스트 인코딩 (배치 처리)"""
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="  Encoding", leave=False):
            batch = texts[i:i + batch_size]
            embeddings = self.encode_batch(batch)
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)


class PrecomputePipeline:
    """Pre-compute 파이프라인"""

    def __init__(self, config: PrecomputeConfig):
        self.config = config
        self.client = storage.Client()
        self.bucket = self.client.bucket(config.gcs_bucket)

        # 캐시 디렉토리
        Path(config.local_cache).mkdir(parents=True, exist_ok=True)

        # Parquet captions (row index로 접근)
        self.captions_list: List[str] = []

        # 임베딩 계산기
        self.embedder = EmbeddingComputer(config.model_name)

    def load_parquet_metadata(self):
        """Parquet에서 row_index -> caption 매핑 로드 (PT keys는 row index임)"""
        print("\n[1/4] Loading parquet metadata...")

        parquet_path = self.config.parquet_path

        # GCS에서 다운로드
        if parquet_path.startswith("gs://"):
            parts = parquet_path[5:].split("/", 1)
            bucket_name = parts[0]
            blob_path = parts[1]

            local_parquet = os.path.join(self.config.local_cache, "metadata.parquet")

            if not os.path.exists(local_parquet):
                print(f"  Downloading {parquet_path}...")
                bucket = self.client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                blob.download_to_filename(local_parquet)
                print(f"  Downloaded to {local_parquet}")
            else:
                print(f"  Using cached {local_parquet}")

            parquet_path = local_parquet

        # 로드 - caption_llava만 (row index로 접근)
        print(f"  Reading parquet (caption_llava only)...")
        table = pq.read_table(parquet_path, columns=['caption_llava'])

        # List로 변환 (row index로 접근 가능)
        self.captions_list = table['caption_llava'].to_pylist()
        print(f"  Loaded {len(self.captions_list):,} captions (by row index)")

    def list_pt_files(self) -> List[str]:
        """PT 파일 목록 조회"""
        blobs = self.client.list_blobs(
            self.config.gcs_bucket,
            prefix=self.config.input_prefix
        )
        pt_files = []
        for blob in blobs:
            if blob.name.endswith('.pt'):
                pt_files.append(blob.name)
        return sorted(pt_files)

    def check_existing(self, pt_files: List[str]) -> List[str]:
        """이미 처리된 파일 제외"""
        existing = set()
        blobs = self.client.list_blobs(
            self.config.gcs_bucket,
            prefix=self.config.output_prefix
        )
        for blob in blobs:
            if blob.name.endswith('.pt'):
                filename = os.path.basename(blob.name)
                existing.add(filename)

        remaining = []
        for pt_file in pt_files:
            filename = os.path.basename(pt_file)
            if filename not in existing:
                remaining.append(pt_file)
            else:
                print(f"  Skipping {filename} (already exists)")

        return remaining

    def process_single_file(self, blob_path: str) -> bool:
        """단일 PT 파일 처리"""
        filename = os.path.basename(blob_path)
        local_input = os.path.join(self.config.local_cache, f"input_{filename}")
        local_output = os.path.join(self.config.local_cache, f"output_{filename}")

        try:
            # 1. 다운로드
            print(f"\n  [{filename}] Downloading...")
            blob = self.bucket.blob(blob_path)
            blob.download_to_filename(local_input)

            # 2. 로드
            print(f"  [{filename}] Loading PT...")
            data = torch.load(local_input, map_location='cpu')
            keys = data['keys'].numpy()
            num_samples = len(keys)
            print(f"  [{filename}] {num_samples:,} samples")

            # 3. Caption 추출 (keys는 parquet row index)
            captions = []
            missing = 0
            for key in keys:
                row_idx = int(key)
                if row_idx < len(self.captions_list):
                    caption = self.captions_list[row_idx] or ""
                else:
                    caption = ""
                    missing += 1
                captions.append(caption)

            if missing > 0:
                print(f"  [{filename}] Warning: {missing} out-of-range indices")

            # 4. 임베딩 계산
            print(f"  [{filename}] Computing embeddings...")
            embeddings = self.embedder.encode_all(captions, batch_size=self.config.batch_size)

            # 5. bfloat16으로 변환하여 저장
            embeddings_bf16 = torch.from_numpy(embeddings).to(torch.bfloat16)
            data['embeddings'] = embeddings_bf16

            print(f"  [{filename}] Saving ({embeddings_bf16.shape})...")
            torch.save(data, local_output)

            # 6. 업로드
            output_blob_path = self.config.output_prefix + filename
            print(f"  [{filename}] Uploading to {output_blob_path}...")
            output_blob = self.bucket.blob(output_blob_path)
            output_blob.upload_from_filename(local_output)

            # 7. 정리
            os.remove(local_input)
            os.remove(local_output)

            print(f"  [{filename}] Done!")
            return True

        except Exception as e:
            print(f"  [{filename}] Error: {e}")
            import traceback
            traceback.print_exc()

            # 정리
            if os.path.exists(local_input):
                os.remove(local_input)
            if os.path.exists(local_output):
                os.remove(local_output)

            return False

    def run(self):
        """전체 파이프라인 실행"""
        print("="*60)
        print("Pre-compute Embeddings Pipeline")
        print("="*60)

        # 1. Parquet 메타데이터 로드
        self.load_parquet_metadata()

        # 2. PT 파일 목록
        print("\n[2/4] Listing PT files...")
        pt_files = self.list_pt_files()
        print(f"  Found {len(pt_files)} PT files")

        # 3. 이미 처리된 파일 제외
        print("\n[3/4] Checking existing files...")
        remaining = self.check_existing(pt_files)
        print(f"  Remaining: {len(remaining)} files")

        if not remaining:
            print("\n All files already processed!")
            return

        # 4. 순차 처리 (메모리 관리를 위해)
        print(f"\n[4/4] Processing {len(remaining)} files...")

        success = 0
        failed = 0

        for i, blob_path in enumerate(remaining):
            print(f"\n--- File {i+1}/{len(remaining)} ---")
            if self.process_single_file(blob_path):
                success += 1
            else:
                failed += 1

            # 메모리 정리
            gc.collect()

        print("\n" + "="*60)
        print(f"Completed: {success} success, {failed} failed")
        print("="*60)


def main():
    config = PrecomputeConfig()
    pipeline = PrecomputePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
