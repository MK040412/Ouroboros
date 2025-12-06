"""
ImageNet 1000 클래스 임베딩 사전 계산 (TPU)

Gemma-3 270M으로 ImageNet 클래스 이름을 임베딩하여 GCS에 저장
각 Worker가 독립적으로 로컬 TPU만 사용 (TPU pod 전체 동기화 방지)

Usage:
    # 각 Worker에서 실행 (TPU_WORKER_ID 환경변수로 구분)
    TPU_WORKER_ID=0 python precompute_imagenet_embeddings.py
"""

import os

# 각 Worker가 독립적으로 로컬 TPU만 사용하도록 설정
# (TPU pod 전체 동기화 방지 - orbax checkpoint에서 distributed 요구 안함)
os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
os.environ["TPU_HOST_BOUNDS"] = "1,1,1"

import gc
import sys
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from google.cloud import storage

# JAX TPU 초기화
import jax
import jax.numpy as jnp

# TPU 초기화 확인
print(f"[JAX] Devices: {jax.devices()}")
print(f"[JAX] Backend: {jax.default_backend()}")
print(f"[JAX] Local device count: {jax.local_device_count()}")
sys.stdout.flush()

# Gemma 라이브러리
from gemma import gm


@dataclass
class ImageNetEmbeddingConfig:
    """ImageNet 임베딩 계산 설정"""
    gcs_bucket: str = "rdy-tpu-data-2025"
    output_path: str = "imagenet-1k/imagenet_class_embeddings.npy"
    classes_path: str = "gs://rdy-tpu-data-2025/imagenet-1k/classes.py"

    # Gemma-3 270M 설정
    embedding_dim: int = 640
    max_length: int = 128
    batch_size: int = 50  # 작은 배치로 메모리 효율적 처리

    local_cache: str = "/tmp/imagenet_embeddings"

    # 분산 설정
    num_workers: int = 8


class GemmaEmbedder:
    """Gemma-3 270M 임베딩 계산기 (TPU with pmap)"""

    def __init__(self, max_length: int = 128):
        self.max_length = max_length
        self.embedding_dim = 640
        self.model = None
        self.tokenizer = None
        self.params = None
        self.num_devices = jax.local_device_count()
        self._encode_pmap = None

    def load(self):
        """모델 로드"""
        if self.model is not None:
            return

        print(f"[Gemma] Loading Gemma-3 270M ({self.num_devices} local devices)...")
        sys.stdout.flush()

        self.model = gm.nn.Gemma3_270M()
        self.params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_270M_PT)
        self.tokenizer = gm.text.Gemma3Tokenizer()

        print(f"[Gemma] Model loaded (embedding_dim={self.embedding_dim})")

        # pmap 함수 생성
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

            # Create attention mask (non-zero tokens)
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
        """배치 인코딩 (pmap 사용)"""
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

        # Reshape back: (num_devices, per_device_batch, dim) -> (total_batch, dim)
        embeddings = np.array(embeddings).reshape(-1, embeddings.shape[-1])

        # Remove padding
        embeddings = embeddings[:original_batch_size]

        return embeddings.astype(np.float32)


class ImageNetEmbeddingComputer:
    """ImageNet 클래스 임베딩 계산기"""

    def __init__(self, config: ImageNetEmbeddingConfig, worker_id: int):
        self.config = config
        self.worker_id = worker_id
        self.client = storage.Client()
        self.bucket = self.client.bucket(config.gcs_bucket)

        # Worker 0만 저장
        self.is_main = (worker_id == 0)

        # 캐시 디렉토리
        cache_dir = f"{config.local_cache}_worker_{worker_id}"
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir

        # 임베더
        self.embedder = GemmaEmbedder(max_length=config.max_length)

    def load_class_labels(self) -> Dict[int, str]:
        """GCS에서 classes.py 로드하여 클래스 라벨 추출"""
        print(f"\n[1/3] Loading class labels from {self.config.classes_path}...")
        sys.stdout.flush()

        classes_path = self.config.classes_path
        if classes_path.startswith("gs://"):
            parts = classes_path[5:].split("/", 1)
            bucket_name = parts[0]
            blob_path = parts[1]

            local_path = os.path.join(self.cache_dir, "classes.py")

            if not os.path.exists(local_path):
                print(f"  Downloading {classes_path}...")
                bucket = self.client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                blob.download_to_filename(local_path)
            else:
                print(f"  Using cached {local_path}")

            classes_path = local_path

        # classes.py 파싱
        labels = {}
        with open(classes_path, 'r') as f:
            content = f.read()

        # exec으로 IMAGENET_CLASSES 로드
        local_vars = {}
        exec(content, {}, local_vars)

        if 'IMAGENET_CLASSES' in local_vars:
            labels = local_vars['IMAGENET_CLASSES']
        else:
            # 다른 형식 시도
            for var_value in local_vars.values():
                if isinstance(var_value, dict) and len(var_value) == 1000:
                    labels = var_value
                    break

        if not labels:
            print("  Warning: Could not parse classes.py, using fallback")
            labels = {i: f"class_{i}" for i in range(1000)}

        print(f"  Loaded {len(labels)} class labels")
        sys.stdout.flush()
        return labels

    def compute_embeddings(self, class_labels: Dict[int, str]) -> np.ndarray:
        """클래스 임베딩 계산 (작은 배치로 순차 처리)"""
        print(f"\n[2/3] Computing embeddings (batch_size={self.config.batch_size})...")
        sys.stdout.flush()

        # Warmup JIT
        print("  Warming up pmap...")
        _ = self.embedder.encode_batch(["warmup text"] * max(4, self.embedder.num_devices))
        print("  pmap ready!")
        sys.stdout.flush()

        # 클래스 인덱스 순서대로 텍스트 리스트 생성
        class_texts = []
        for i in range(1000):
            label = class_labels.get(i, f"class_{i}")
            class_texts.append(label)

        # 배치 단위로 처리
        all_embeddings = []
        batch_size = self.config.batch_size
        num_batches = (len(class_texts) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(class_texts))
            batch_texts = class_texts[start:end]

            print(f"  Batch {batch_idx + 1}/{num_batches}: classes {start}-{end-1}")
            sys.stdout.flush()

            embeddings = self.embedder.encode_batch(batch_texts)
            all_embeddings.append(embeddings)

            # 메모리 정리
            gc.collect()

        # 전체 임베딩 합치기
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        print(f"  Computed embeddings: {all_embeddings.shape}")
        sys.stdout.flush()

        return all_embeddings

    def save_to_gcs(self, embeddings: np.ndarray):
        """GCS에 임베딩 저장 (Worker 0만) - gsutil 사용"""
        if not self.is_main:
            print(f"\n[3/3] Worker {self.worker_id}: Skipping save (not main worker)")
            sys.stdout.flush()
            return

        print(f"\n[3/3] Saving to GCS...")
        sys.stdout.flush()

        # 로컬에 저장
        local_path = os.path.join(self.cache_dir, "imagenet_class_embeddings.npy")
        np.save(local_path, embeddings)
        print(f"  Saved locally: {local_path}")

        # 파일 크기 확인
        file_size = os.path.getsize(local_path) / 1024
        print(f"  File size: {file_size:.1f} KB")

        # gsutil로 GCS 업로드
        gcs_path = f"gs://{self.config.gcs_bucket}/{self.config.output_path}"
        print(f"  Uploading with gsutil to: {gcs_path}")
        sys.stdout.flush()

        result = subprocess.run(
            ["gsutil", "cp", local_path, gcs_path],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"  Upload successful!")
        else:
            print(f"  Upload failed: {result.stderr}")

        sys.stdout.flush()

    def run(self):
        """전체 파이프라인 실행"""
        print("=" * 60)
        print(f"ImageNet Class Embeddings Pre-compute (Worker {self.worker_id})")
        print("=" * 60)
        print(f"  JAX backend: {jax.default_backend()}")
        print(f"  TPU devices: {jax.local_device_count()}")
        print(f"  GCS bucket: {self.config.gcs_bucket}")
        print(f"  Output: {self.config.output_path}")
        print(f"  Embedding dim: {self.config.embedding_dim}")
        print(f"  Batch size: {self.config.batch_size}")
        sys.stdout.flush()

        # 1. 클래스 라벨 로드
        class_labels = self.load_class_labels()

        # 2. 임베딩 계산
        embeddings = self.compute_embeddings(class_labels)

        # 3. GCS 저장 (Worker 0만)
        self.save_to_gcs(embeddings)

        print("\n" + "=" * 60)
        print(f"[Worker {self.worker_id}] Done!")
        print("=" * 60)
        sys.stdout.flush()


def main():
    worker_id = int(os.environ.get('TPU_WORKER_ID', '0'))
    config = ImageNetEmbeddingConfig()
    computer = ImageNetEmbeddingComputer(config, worker_id)
    computer.run()


if __name__ == "__main__":
    main()
