"""
Text Embedding Provider for TPU Training

Caption에서 실시간으로 임베딩 계산
- CPU에서 HuggingFace 모델 사용
- 배치 단위 병렬 처리
- 캐싱으로 중복 계산 방지
"""

import numpy as np
import jax.numpy as jnp
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import threading


@dataclass
class EmbeddingConfig:
    """임베딩 설정"""
    model_name: str = "google/embeddinggemma-300m"
    embedding_dim: int = 768
    max_length: int = 512
    batch_size: int = 256


class CachedEmbeddingProvider:
    """CPU 기반 임베딩 + 캐싱"""

    def __init__(self, model_name: str = "google/embeddinggemma-300m",
                 embedding_dim: int = 768):
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        # 캐시 (LRU 스타일)
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = 50000  # 최대 캐시 엔트리

        # 모델 lazy loading
        self._model = None
        self._tokenizer = None
        self._model_lock = threading.Lock()
        self._device = "cpu"

    def _load_model(self):
        """모델 lazy loading (첫 사용 시)"""
        if self._model is not None:
            return

        with self._model_lock:
            if self._model is not None:
                return

            try:
                from transformers import AutoModel, AutoTokenizer
                import torch

                print(f"Loading embedding model: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.eval()

                # CPU에서 실행
                self._model = self._model.to(self._device)
                print(f"✓ Embedding model loaded on {self._device}")

            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
                print("Using random embeddings as fallback")
                self._model = None

    def _get_cached(self, text: str) -> Optional[np.ndarray]:
        """캐시에서 임베딩 조회"""
        with self.cache_lock:
            return self.cache.get(text)

    def _set_cached(self, text: str, embedding: np.ndarray):
        """캐시에 임베딩 저장"""
        with self.cache_lock:
            # LRU: 캐시가 가득 차면 절반 삭제
            if len(self.cache) >= self.max_cache_size:
                keys_to_remove = list(self.cache.keys())[:self.max_cache_size // 2]
                for k in keys_to_remove:
                    del self.cache[k]
            self.cache[text] = embedding

    def batch_encode(self, texts: List[str], batch_size: int = 256,
                     normalize: bool = True) -> np.ndarray:
        """배치 인코딩"""
        self._load_model()

        if self._model is None:
            # Fallback: 랜덤 임베딩
            embeddings = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-8)
            return embeddings

        try:
            import torch

            all_embeddings = []

            # 캐시 확인 및 미캐시 텍스트 분류
            cached_results = {}  # idx -> embedding
            uncached_texts = []  # (idx, text)

            for idx, text in enumerate(texts):
                cached = self._get_cached(text)
                if cached is not None:
                    cached_results[idx] = cached
                else:
                    uncached_texts.append((idx, text))

            # 미캐시 텍스트 배치 인코딩
            if uncached_texts:
                uncached_indices = [item[0] for item in uncached_texts]
                uncached_strs = [item[1] for item in uncached_texts]

                new_embeddings = []

                for i in range(0, len(uncached_strs), batch_size):
                    batch_texts = uncached_strs[i:i + batch_size]

                    with torch.no_grad():
                        inputs = self._tokenizer(
                            batch_texts,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512,
                            padding=True
                        )
                        inputs = {k: v.to(self._device) for k, v in inputs.items()}

                        outputs = self._model(**inputs)

                        # Mean pooling
                        attention_mask = inputs['attention_mask']
                        token_embeddings = outputs.last_hidden_state

                        # Masked mean
                        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                        batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

                        new_embeddings.append(batch_embeddings)

                # 새 임베딩 캐시에 저장
                new_embeddings = np.concatenate(new_embeddings, axis=0)
                for idx, (orig_idx, text) in enumerate(uncached_texts):
                    emb = new_embeddings[idx].astype(np.float32)
                    self._set_cached(text, emb)
                    cached_results[orig_idx] = emb

            # 결과 조합 (순서 유지)
            result = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
            for idx, emb in cached_results.items():
                result[idx] = emb

            if normalize:
                norms = np.linalg.norm(result, axis=1, keepdims=True)
                result = result / np.maximum(norms, 1e-8)

            return result

        except Exception as e:
            print(f"Batch encoding error: {e}")
            import traceback
            traceback.print_exc()

            # Fallback: 랜덤 임베딩
            embeddings = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-8)
            return embeddings


class DummyEmbeddingProvider:
    """테스트용 더미 임베딩 (랜덤)"""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

    def batch_encode(self, texts: List[str], batch_size: int = 256,
                     normalize: bool = True) -> np.ndarray:
        """랜덤 임베딩 반환"""
        embeddings = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
        return embeddings


def get_embedding_provider(model_name: str = "google/embeddinggemma-300m",
                           use_precomputed: bool = False,
                           embedding_dim: int = 768) -> Any:
    """임베딩 프로바이더 팩토리

    Args:
        model_name: HuggingFace 모델 이름
        use_precomputed: (무시됨 - 항상 실시간 계산)
        embedding_dim: 임베딩 차원

    Returns:
        EmbeddingProvider 인스턴스
    """
    print(f"Creating CachedEmbeddingProvider: {model_name} (dim={embedding_dim})")
    return CachedEmbeddingProvider(
        model_name=model_name,
        embedding_dim=embedding_dim
    )
