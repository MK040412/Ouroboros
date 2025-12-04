"""
Text Embedding Provider for TPU Training

Caption에서 실시간으로 임베딩 계산
- 멀티코어 CPU 병렬 처리 (112 vCPU 활용)
- 배치 단위 병렬 처리
- 캐싱으로 중복 계산 방지
"""

import numpy as np
import jax.numpy as jnp
import jax
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import threading
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp


@dataclass
class EmbeddingConfig:
    """임베딩 설정"""
    model_name: str = "google/embeddinggemma-300m"
    embedding_dim: int = 768
    max_length: int = 512
    batch_size: int = 256


# 글로벌 모델 (프로세스별 한 번만 로드)
_global_model = None
_global_tokenizer = None
_global_lock = threading.Lock()


def _init_worker_model(model_name: str):
    """워커 프로세스에서 모델 초기화"""
    global _global_model, _global_tokenizer

    if _global_model is not None:
        return

    with _global_lock:
        if _global_model is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            # CPU 스레드 수 설정
            num_threads = max(1, os.cpu_count() // 4)  # 워커당 스레드
            torch.set_num_threads(num_threads)

            _global_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _global_model = AutoModel.from_pretrained(model_name)
            _global_model.eval()

        except Exception as e:
            print(f"Worker model init failed: {e}")
            _global_model = None
            _global_tokenizer = None


def _encode_batch_worker(args: Tuple[List[str], str, int, bool]) -> np.ndarray:
    """단일 배치 인코딩 (워커 함수)"""
    texts, model_name, embedding_dim, normalize = args

    global _global_model, _global_tokenizer

    # 모델 초기화 (없으면)
    if _global_model is None:
        _init_worker_model(model_name)

    if _global_model is None:
        # Fallback: 랜덤 임베딩
        embeddings = np.random.randn(len(texts), embedding_dim).astype(np.float32)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
        return embeddings

    try:
        import torch

        with torch.no_grad():
            inputs = _global_tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            outputs = _global_model(**inputs)

            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state

            # Masked mean
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).numpy()

            if normalize:
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / np.maximum(norms, 1e-8)

            return batch_embeddings.astype(np.float32)

    except Exception as e:
        print(f"Batch encode error: {e}")
        embeddings = np.random.randn(len(texts), embedding_dim).astype(np.float32)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
        return embeddings


class CachedEmbeddingProvider:
    """멀티코어 CPU 병렬 임베딩 + 캐싱

    112 vCPU를 활용하여 병렬로 임베딩 계산
    """

    def __init__(self, model_name: str = "google/embeddinggemma-300m",
                 embedding_dim: int = 768,
                 num_workers: int = None):
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        # 워커 수 결정 (기본: CPU 코어의 절반, 최대 56)
        if num_workers is None:
            num_workers = min(56, max(1, os.cpu_count() // 2))
        self.num_workers = num_workers

        # 캐시 (LRU 스타일)
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = 100000  # 최대 캐시 엔트리

        # 모델 lazy loading (메인 프로세스용)
        self._model = None
        self._tokenizer = None
        self._model_lock = threading.Lock()

        # ThreadPool for parallel encoding
        self._executor = None

        print(f"CachedEmbeddingProvider initialized:")
        print(f"  Model: {model_name}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Parallel workers: {num_workers}")

    def _get_executor(self):
        """ThreadPoolExecutor 가져오기 (lazy init)"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        return self._executor

    def _load_model(self):
        """모델 lazy loading (메인 프로세스)"""
        if self._model is not None:
            return

        with self._model_lock:
            if self._model is not None:
                return

            try:
                from transformers import AutoModel, AutoTokenizer
                import torch

                # CPU 스레드 최적화
                torch.set_num_threads(self.num_workers)

                print(f"Loading embedding model: {self.model_name}")

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.eval()

                print(f"✓ Embedding model loaded on CPU ({self.num_workers} workers)")

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

    def _set_cached_batch(self, texts: List[str], embeddings: np.ndarray):
        """배치로 캐시에 저장"""
        with self.cache_lock:
            # LRU: 캐시가 가득 차면 절반 삭제
            if len(self.cache) + len(texts) >= self.max_cache_size:
                keys_to_remove = list(self.cache.keys())[:self.max_cache_size // 2]
                for k in keys_to_remove:
                    del self.cache[k]

            for text, emb in zip(texts, embeddings):
                self.cache[text] = emb

    def batch_encode(self, texts: List[str], batch_size: int = 64,
                     normalize: bool = True) -> np.ndarray:
        """멀티코어 병렬 배치 인코딩

        Args:
            texts: 인코딩할 텍스트 리스트
            batch_size: 워커당 배치 크기
            normalize: L2 정규화 여부

        Returns:
            (N, embedding_dim) 크기의 임베딩 배열
        """
        self._load_model()

        if self._model is None:
            # Fallback: 랜덤 임베딩
            embeddings = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-8)
            return embeddings

        # 캐시 확인 및 미캐시 텍스트 분류
        cached_results = {}  # idx -> embedding
        uncached_texts = []  # (idx, text)

        for idx, text in enumerate(texts):
            cached = self._get_cached(text)
            if cached is not None:
                cached_results[idx] = cached
            else:
                uncached_texts.append((idx, text))

        # 미캐시 텍스트가 있으면 병렬 인코딩
        if uncached_texts:
            uncached_indices = [item[0] for item in uncached_texts]
            uncached_strs = [item[1] for item in uncached_texts]

            # 배치로 분할
            batches = []
            for i in range(0, len(uncached_strs), batch_size):
                batch_texts = uncached_strs[i:i + batch_size]
                batches.append(batch_texts)

            # 병렬 인코딩 (ThreadPoolExecutor 사용)
            all_embeddings = []

            try:
                import torch

                # 단일 스레드에서 순차 처리 (모델이 이미 멀티스레드 활용)
                # PyTorch가 내부적으로 num_threads 만큼 병렬 처리
                for batch_texts in batches:
                    with torch.no_grad():
                        inputs = self._tokenizer(
                            batch_texts,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512,
                            padding=True
                        )

                        outputs = self._model(**inputs)

                        # Mean pooling
                        attention_mask = inputs['attention_mask']
                        token_embeddings = outputs.last_hidden_state

                        # Masked mean
                        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                        batch_embeddings = (sum_embeddings / sum_mask).numpy()

                        all_embeddings.append(batch_embeddings)

                # 결합
                new_embeddings = np.concatenate(all_embeddings, axis=0)

                if normalize:
                    norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
                    new_embeddings = new_embeddings / np.maximum(norms, 1e-8)

                new_embeddings = new_embeddings.astype(np.float32)

                # 캐시에 저장
                self._set_cached_batch(uncached_strs, new_embeddings)

                # 결과에 추가
                for i, orig_idx in enumerate(uncached_indices):
                    cached_results[orig_idx] = new_embeddings[i]

            except Exception as e:
                print(f"Batch encoding error: {e}")
                import traceback
                traceback.print_exc()

                # Fallback: 랜덤 임베딩
                for orig_idx, text in uncached_texts:
                    emb = np.random.randn(self.embedding_dim).astype(np.float32)
                    if normalize:
                        emb = emb / np.maximum(np.linalg.norm(emb), 1e-8)
                    cached_results[orig_idx] = emb

        # 결과 조합 (순서 유지)
        result = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for idx, emb in cached_results.items():
            result[idx] = emb

        return result

    def shutdown(self):
        """리소스 정리"""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


class ParallelEmbeddingProvider:
    """멀티프로세스 병렬 임베딩 (112 vCPU 최대 활용)

    각 프로세스가 독립적으로 모델을 로드하여 병렬 처리
    메모리 사용량이 높지만 최대 병렬성 확보
    """

    def __init__(self, model_name: str = "google/embeddinggemma-300m",
                 embedding_dim: int = 768,
                 num_workers: int = None):
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        # 워커 수 결정 (메모리 고려해서 제한)
        if num_workers is None:
            # 모델당 약 4GB 메모리 필요하므로 적절히 조절
            num_workers = min(8, max(1, os.cpu_count() // 14))
        self.num_workers = num_workers

        # 캐시
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = 100000

        print(f"ParallelEmbeddingProvider initialized:")
        print(f"  Model: {model_name}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Process workers: {num_workers}")

    def _get_cached(self, text: str) -> Optional[np.ndarray]:
        with self.cache_lock:
            return self.cache.get(text)

    def _set_cached_batch(self, texts: List[str], embeddings: np.ndarray):
        with self.cache_lock:
            if len(self.cache) + len(texts) >= self.max_cache_size:
                keys_to_remove = list(self.cache.keys())[:self.max_cache_size // 2]
                for k in keys_to_remove:
                    del self.cache[k]
            for text, emb in zip(texts, embeddings):
                self.cache[text] = emb

    def batch_encode(self, texts: List[str], batch_size: int = 128,
                     normalize: bool = True) -> np.ndarray:
        """멀티프로세스 병렬 인코딩"""

        # 캐시 확인
        cached_results = {}
        uncached_texts = []

        for idx, text in enumerate(texts):
            cached = self._get_cached(text)
            if cached is not None:
                cached_results[idx] = cached
            else:
                uncached_texts.append((idx, text))

        if uncached_texts:
            uncached_indices = [item[0] for item in uncached_texts]
            uncached_strs = [item[1] for item in uncached_texts]

            # 워커별로 배치 분할
            batches = []
            for i in range(0, len(uncached_strs), batch_size):
                batch = uncached_strs[i:i + batch_size]
                batches.append((batch, self.model_name, self.embedding_dim, normalize))

            # 멀티프로세스 실행
            try:
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    results = list(executor.map(_encode_batch_worker, batches))

                new_embeddings = np.concatenate(results, axis=0)

                # 캐시에 저장
                self._set_cached_batch(uncached_strs, new_embeddings)

                for i, orig_idx in enumerate(uncached_indices):
                    cached_results[orig_idx] = new_embeddings[i]

            except Exception as e:
                print(f"Parallel encoding error: {e}")
                for orig_idx, text in uncached_texts:
                    emb = np.random.randn(self.embedding_dim).astype(np.float32)
                    if normalize:
                        emb = emb / np.maximum(np.linalg.norm(emb), 1e-8)
                    cached_results[orig_idx] = emb

        result = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for idx, emb in cached_results.items():
            result[idx] = emb

        return result


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


class GemmaEmbeddingProvider:
    """Gemma-3 270M embedding provider using gemma library (TPU/CPU)

    precompute_embeddings_tpu.py의 GemmaEmbeddingModel을 기반으로 함
    transformers 대신 gemma 라이브러리 네이티브 사용
    """

    def __init__(self, max_length: int = 128):
        self.max_length = max_length
        self.embedding_dim = 640  # Gemma-3 270M hidden dimension
        self.model = None
        self.tokenizer = None
        self.params = None
        self._initialized = False

    def _initialize(self):
        """Lazy loading of model"""
        if self._initialized:
            return

        try:
            from gemma import gm

            print("Loading Gemma-3 270M via gemma library...")
            self.model = gm.nn.Gemma3_270M()
            self.params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_270M_PT)
            self.tokenizer = gm.text.Gemma3Tokenizer()
            self._initialized = True
            print(f"  ✓ Model loaded (embedding_dim={self.embedding_dim})")
        except Exception as e:
            print(f"  ✗ Failed to load Gemma model: {e}")
            self._initialized = False

    def batch_encode(self, texts: List[str], batch_size: int = 256,
                     normalize: bool = True) -> np.ndarray:
        """Encode texts to embeddings using Gemma-3 270M"""
        self._initialize()

        if not self._initialized or self.model is None:
            # Fallback: random embeddings
            print("Warning: Using random embeddings (Gemma model not available)")
            embeddings = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-8)
            return embeddings

        # Tokenize all texts
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


def get_embedding_provider(model_name: str = "gemma-3-270m",
                           use_precomputed: bool = True,
                           embedding_dim: int = 640,
                           num_workers: int = None) -> Any:
    """임베딩 프로바이더 팩토리

    Args:
        model_name: 모델 이름 (gemma-3-270m 또는 HuggingFace 모델)
        use_precomputed: True면 None 반환 (precomputed embeddings 사용)
        embedding_dim: 임베딩 차원
        num_workers: CPU 워커 수 (기본: 자동)

    Returns:
        EmbeddingProvider 인스턴스 또는 None (precomputed mode)
    """
    # Precomputed mode: embedding_provider 불필요
    if use_precomputed:
        print("Using precomputed embeddings (no embedding_provider needed)")
        return None

    # Gemma library provider (gemma-3-270m)
    if "gemma" in model_name.lower():
        print(f"Creating GemmaEmbeddingProvider:")
        print(f"  Model: {model_name}")
        print(f"  Embedding dim: 640 (Gemma-3 270M)")
        return GemmaEmbeddingProvider(max_length=128)

    # Legacy: transformers-based (fallback for other models)
    cpu_count = os.cpu_count() or 1
    if num_workers is None:
        num_workers = min(56, max(1, cpu_count // 2))

    print(f"Creating CachedEmbeddingProvider (transformers):")
    print(f"  Model: {model_name}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  CPU cores available: {cpu_count}")
    print(f"  Workers: {num_workers}")

    return CachedEmbeddingProvider(
        model_name=model_name,
        embedding_dim=embedding_dim,
        num_workers=num_workers
    )
