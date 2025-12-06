"""
ImageNet 1000 클래스 임베딩 사전 계산

Gemma-3 270M으로 ImageNet 클래스 이름을 임베딩하여 GCS에 저장
학습 시 VAE와 메모리 충돌 없이 사전 계산된 임베딩 사용

Usage:
    python precompute_imagenet_embeddings.py
"""

import os
import io
import gc
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from google.cloud import storage


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


class GemmaEmbedder:
    """Gemma-3 270M 임베딩 계산기"""

    def __init__(self, max_length: int = 128):
        self.max_length = max_length
        self.embedding_dim = 640
        self.model = None
        self.tokenizer = None
        self.params = None

    def load(self):
        """모델 로드"""
        if self.model is not None:
            return

        try:
            from gemma import gm

            print("[Gemma] Loading Gemma-3 270M...")
            self.model = gm.nn.Gemma3_270M()
            self.params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_270M_PT)
            self.tokenizer = gm.text.Gemma3Tokenizer()
            print(f"[Gemma] Model loaded (embedding_dim={self.embedding_dim})")
        except Exception as e:
            print(f"[Gemma] Failed to load: {e}")
            raise

    def encode_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """배치 인코딩"""
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

        # Forward pass
        out = self.model.apply(
            {'params': self.params},
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
        embeddings = sum_embeddings / sum_mask

        # L2 normalize
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)

        return embeddings.astype(np.float32)


class ImageNetEmbeddingComputer:
    """ImageNet 클래스 임베딩 계산기"""

    def __init__(self, config: ImageNetEmbeddingConfig):
        self.config = config
        self.client = storage.Client()
        self.bucket = self.client.bucket(config.gcs_bucket)

        # 캐시 디렉토리
        Path(config.local_cache).mkdir(parents=True, exist_ok=True)

        # 임베더
        self.embedder = GemmaEmbedder(max_length=config.max_length)

    def load_class_labels(self) -> Dict[int, str]:
        """GCS에서 classes.py 로드하여 클래스 라벨 추출"""
        print(f"\n[1/3] Loading class labels from {self.config.classes_path}...")

        classes_path = self.config.classes_path
        if classes_path.startswith("gs://"):
            parts = classes_path[5:].split("/", 1)
            bucket_name = parts[0]
            blob_path = parts[1]

            local_path = os.path.join(self.config.local_cache, "classes.py")

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
            for var_name, var_value in local_vars.items():
                if isinstance(var_value, dict) and len(var_value) == 1000:
                    labels = var_value
                    break

        if not labels:
            print("  Warning: Could not parse classes.py, using fallback")
            labels = {i: f"class_{i}" for i in range(1000)}

        print(f"  Loaded {len(labels)} class labels")
        return labels

    def compute_embeddings(self, class_labels: Dict[int, str]) -> np.ndarray:
        """클래스 임베딩 계산 (작은 배치로 순차 처리)"""
        print(f"\n[2/3] Computing embeddings (batch_size={self.config.batch_size})...")

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

            embeddings = self.embedder.encode_batch(batch_texts, normalize=True)
            all_embeddings.append(embeddings)

            # 메모리 정리
            gc.collect()

        # 전체 임베딩 합치기
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        print(f"  Computed embeddings: {all_embeddings.shape}")

        return all_embeddings

    def save_to_gcs(self, embeddings: np.ndarray):
        """GCS에 임베딩 저장"""
        print(f"\n[3/3] Saving to GCS...")

        # 로컬에 저장
        local_path = os.path.join(self.config.local_cache, "imagenet_class_embeddings.npy")
        np.save(local_path, embeddings)
        print(f"  Saved locally: {local_path}")

        # GCS 업로드
        blob = self.bucket.blob(self.config.output_path)
        blob.upload_from_filename(local_path)
        print(f"  Uploaded to: gs://{self.config.gcs_bucket}/{self.config.output_path}")

        # 파일 크기 확인
        file_size = os.path.getsize(local_path) / 1024
        print(f"  File size: {file_size:.1f} KB")

    def run(self):
        """전체 파이프라인 실행"""
        print("=" * 60)
        print("ImageNet Class Embeddings Pre-compute")
        print("=" * 60)
        print(f"  GCS bucket: {self.config.gcs_bucket}")
        print(f"  Output: {self.config.output_path}")
        print(f"  Embedding dim: {self.config.embedding_dim}")
        print(f"  Batch size: {self.config.batch_size}")

        # 1. 클래스 라벨 로드
        class_labels = self.load_class_labels()

        # 2. 임베딩 계산
        embeddings = self.compute_embeddings(class_labels)

        # 3. GCS 저장
        self.save_to_gcs(embeddings)

        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)


def main():
    config = ImageNetEmbeddingConfig()
    computer = ImageNetEmbeddingComputer(config)
    computer.run()


if __name__ == "__main__":
    main()
