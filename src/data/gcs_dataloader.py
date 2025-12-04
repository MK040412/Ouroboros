"""
GCS DataLoader with Async Embedding Pipeline for TPU Training

최적화 전략:
1. PT 파일에서 latent 로드
2. Parquet에서 caption 로드 (key로 매핑)
3. 비동기 임베딩 계산 (CPU에서 미리 N개 배치 준비)
4. Double buffering으로 TPU 학습과 데이터 로딩 오버랩

GIL 최적화 (2024-12):
- ProcessPoolExecutor로 torch.load 병렬화 (GIL 우회)
- Lock-free 배치 버퍼 (atomic operations)
- JAX 변환을 메인 스레드에서 일괄 처리
"""

import os
import gc
import glob
import queue
import threading
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Iterator, Any, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from collections import deque
import time

import numpy as np
import jax.numpy as jnp
import torch
import pyarrow.parquet as pq
from google.cloud import storage


# ============================================
# GIL-free PT Loading (Module-level for pickle)
# ============================================
def _load_pt_file_worker(local_path: str) -> dict:
    """PT 파일 로드 (별도 프로세스에서 실행 - GIL 우회)

    이 함수는 ProcessPoolExecutor에서 호출되므로 GIL 영향 없음.
    """
    data = torch.load(local_path, map_location='cpu', weights_only=False)

    # Tensor -> NumPy 변환 (프로세스 내에서 완료)
    result = {
        'keys': data['keys'].numpy() if isinstance(data['keys'], torch.Tensor) else np.array(data['keys']),
        'latents': data['latents'].numpy() if isinstance(data['latents'], torch.Tensor) else data['latents'],
    }

    # Precomputed embeddings 로드 (bfloat16 -> float32)
    if 'embeddings' in data:
        embeddings = data['embeddings']
        if hasattr(embeddings, 'dtype') and embeddings.dtype == torch.bfloat16:
            embeddings = embeddings.float()
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy()
        result['embeddings'] = embeddings

    return result


@dataclass
class ParquetCache:
    """Parquet 메타데이터 캐시 (key -> caption 매핑)"""
    key_to_caption: Dict[int, str]


class GCSFileHandler:
    """GCS 파일 다운로드 핸들러 (google-cloud-storage 사용)"""

    def __init__(self, gcs_bucket: str, cache_dir: Optional[str] = None,
                 worker_id: Optional[int] = None, clean_old_caches: bool = True):
        self.gcs_bucket_url = gcs_bucket.rstrip('/')

        # 고정 캐시 디렉토리 사용 (worker별로 분리)
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            # Worker ID 결정 (JAX process index 또는 환경변수)
            if worker_id is None:
                worker_id = int(os.environ.get('JAX_PROCESS_INDEX', '0'))
            self.cache_dir = f"/tmp/gcs_cache_worker_{worker_id}"

        # 이전 임시 캐시 디렉토리 정리 (시작 시 1회)
        if clean_old_caches:
            self._cleanup_old_cache_dirs()

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Parse bucket name and prefix from gs:// URL
        # e.g., gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/
        if self.gcs_bucket_url.startswith("gs://"):
            parts = self.gcs_bucket_url[5:].rstrip('/').split('/', 1)
            self.bucket_name = parts[0]
            self.prefix = parts[1] + '/' if len(parts) > 1 else ''
        else:
            self.bucket_name = self.gcs_bucket_url
            self.prefix = ''

        # Initialize GCS client
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

    def _cleanup_old_cache_dirs(self):
        """이전 임시 캐시 디렉토리 정리 (gcs_cache_로 시작하는 랜덤 디렉토리들)"""
        try:
            import shutil
            for item in Path("/tmp").iterdir():
                # gcs_cache_로 시작하지만 gcs_cache_worker_가 아닌 디렉토리 삭제
                if item.is_dir() and item.name.startswith("gcs_cache_"):
                    if not item.name.startswith("gcs_cache_worker_"):
                        print(f"  Cleaning old cache dir: {item}")
                        shutil.rmtree(item, ignore_errors=True)
        except Exception as e:
            print(f"  Warning: Could not clean old caches: {e}")

    def download_file(self, gcs_path: str, local_path: Optional[str] = None) -> str:
        """GCS에서 파일 다운로드"""
        if local_path is None:
            filename = os.path.basename(gcs_path)
            local_path = os.path.join(self.cache_dir, filename)

        # 기존 파일이 있으면 크기 확인 (손상된 파일 감지)
        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            if file_size > 1024:  # 1KB 이상이면 유효한 파일로 간주
                return local_path
            else:
                # 손상된 파일 삭제
                print(f"  Removing corrupted file: {local_path} (size={file_size})")
                os.remove(local_path)

        # Parse blob name from gcs_path
        if gcs_path.startswith("gs://"):
            # gs://bucket/path/to/file -> path/to/file
            blob_name = gcs_path[5:].split('/', 1)[1] if '/' in gcs_path[5:] else ''
        else:
            blob_name = gcs_path

        # 임시 파일로 다운로드 후 이동 (원자적 작업)
        temp_path = local_path + ".tmp"
        try:
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(temp_path)

            # 다운로드 성공 확인
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1024:
                os.rename(temp_path, local_path)
            else:
                raise Exception(f"Downloaded file too small or missing: {temp_path}")

        except Exception as e:
            print(f"Warning: GCS download failed for {gcs_path}: {e}")
            # 실패 시 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise  # 에러 전파하여 호출자가 처리하도록

        return local_path

    def list_pt_files(self) -> List[str]:
        """GCS에서 PT 파일 목록 조회"""
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=self.prefix)
            pt_files = []
            for blob in blobs:
                if blob.name.endswith('.pt'):
                    pt_files.append(f"gs://{self.bucket_name}/{blob.name}")
            return sorted(pt_files)
        except Exception as e:
            print(f"Warning: GCS list failed: {e}")
            return []


class PTFileLoader:
    """PT 파일 로더 (latents + keys)"""

    def load(self, pt_path: str) -> dict:
        """PT 파일 로드

        Expected format:
        {
            'keys': tensor of shape (N,) - sample keys
            'latents': tensor of shape (N, 3, 4, 32, 32) - VAE latents (3 crops)
            'embeddings': tensor of shape (N, 640) - Precomputed Gemma-3 270M embeddings (bfloat16)
        }
        """
        data = torch.load(pt_path, map_location='cpu')

        result = {
            'keys': data['keys'].numpy() if isinstance(data['keys'], torch.Tensor) else np.array(data['keys']),
            'latents': data['latents'],  # Keep as torch tensor
        }

        # Precomputed embeddings 로드 (bfloat16 -> float32)
        if 'embeddings' in data:
            embeddings = data['embeddings']
            if embeddings.dtype == torch.bfloat16:
                embeddings = embeddings.float()
            result['embeddings'] = embeddings.numpy()

        return result


class PTFilePrefetchManager:
    """PT 파일 비동기 다운로드 관리자

    핵심 기능:
    1. 다음 N개 PT 파일을 백그라운드에서 미리 다운로드
    2. 다운로드 완료 순서대로 ready_queue에 추가
    3. 캐시 용량 관리 (오래된 파일 자동 삭제)
    """

    def __init__(self, gcs_handler: GCSFileHandler,
                 pt_files: List[str],
                 prefetch_ahead: int = 4,
                 max_workers: int = 4,
                 max_cache_files: int = 4):
        self.gcs_handler = gcs_handler
        self.pt_files = pt_files
        self.prefetch_ahead = prefetch_ahead
        self.max_cache_files = max_cache_files

        # 다운로드 완료 파일 큐 (idx, pt_path, local_path)
        self.ready_queue: queue.Queue[Tuple[int, str, str]] = queue.Queue(maxsize=prefetch_ahead + 1)

        # 다운로드 상태 추적
        self.in_progress: Dict[int, Future] = {}
        self.completed: Dict[int, str] = {}  # idx -> local_path
        self.lock = threading.Lock()

        # 다운로드 스레드풀
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # 프리페치 스케줄러 스레드
        self.stop_event = threading.Event()
        self.current_idx = 0
        self.next_deliver_idx = 0
        self.scheduler_thread: Optional[threading.Thread] = None

    def start(self, start_idx: int = 0):
        """프리페치 시작"""
        self.current_idx = start_idx
        self.next_deliver_idx = start_idx
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        self.scheduler_thread.start()

    def _scheduler_loop(self):
        """프리페치 스케줄링 루프"""
        while not self.stop_event.is_set():
            with self.lock:
                # 1. 완료된 다운로드 체크 및 ready_queue로 전달
                while self.next_deliver_idx in self.completed:
                    local_path = self.completed.pop(self.next_deliver_idx)
                    pt_path = self.pt_files[self.next_deliver_idx % len(self.pt_files)]
                    try:
                        self.ready_queue.put(
                            (self.next_deliver_idx, pt_path, local_path),
                            timeout=1.0
                        )
                        self.next_deliver_idx += 1
                    except queue.Full:
                        # 큐가 가득 차면 다시 저장하고 대기
                        self.completed[self.next_deliver_idx] = local_path
                        break

                # 2. 새 다운로드 시작 (prefetch_ahead 개수 유지)
                target_end = self.next_deliver_idx + self.prefetch_ahead

                for idx in range(self.next_deliver_idx, target_end):
                    if idx not in self.in_progress and idx not in self.completed:
                        future = self.executor.submit(
                            self._download_pt_file, idx
                        )
                        self.in_progress[idx] = future

                # 3. 완료된 Future 처리
                completed_indices = []
                for idx, future in self.in_progress.items():
                    if future.done():
                        try:
                            local_path = future.result()
                            self.completed[idx] = local_path
                            completed_indices.append(idx)
                        except Exception as e:
                            print(f"Download failed for idx {idx}: {e}")
                            completed_indices.append(idx)

                for idx in completed_indices:
                    del self.in_progress[idx]

            # 다음 체크까지 짧게 대기
            time.sleep(0.01)  # 10ms로 감소

    def _download_pt_file(self, idx: int) -> str:
        """단일 PT 파일 다운로드"""
        pt_path = self.pt_files[idx % len(self.pt_files)]
        filename = os.path.basename(pt_path)
        local_path = os.path.join(self.gcs_handler.cache_dir, filename)

        # 캐시 용량 관리 (다운로드 전)
        self._manage_cache_size()

        # 이미 존재하면 스킵
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
            return local_path

        # GCS에서 다운로드
        print(f"  [Prefetch] Downloading {filename}...")
        return self.gcs_handler.download_file(pt_path, local_path)

    def _get_protected_files(self) -> set:
        """현재 사용 중인 파일 목록 (삭제 보호)"""
        protected = set()

        with self.lock:
            # 1. 다운로드 진행 중인 파일들
            for idx in self.in_progress.keys():
                pt_path = self.pt_files[idx % len(self.pt_files)]
                protected.add(os.path.basename(pt_path))

            # 2. 다운로드 완료되어 대기 중인 파일들
            for idx in self.completed.keys():
                pt_path = self.pt_files[idx % len(self.pt_files)]
                protected.add(os.path.basename(pt_path))

            # 3. 현재 prefetch 범위의 파일들 (next_deliver_idx ~ +prefetch_ahead)
            for i in range(self.prefetch_ahead + 2):  # 여유분 추가
                idx = self.next_deliver_idx + i
                if idx < len(self.pt_files):
                    pt_path = self.pt_files[idx % len(self.pt_files)]
                    protected.add(os.path.basename(pt_path))

        return protected

    def _manage_cache_size(self):
        """캐시 용량 관리 (오래된 파일 삭제, 사용 중인 파일 보호)"""
        cached_files = glob.glob(
            os.path.join(self.gcs_handler.cache_dir, "*.pt")
        )

        if len(cached_files) >= self.max_cache_files:
            # 보호 대상 파일 목록
            protected = self._get_protected_files()

            # 가장 오래된 파일부터 삭제 (보호 대상 제외)
            for f in sorted(cached_files, key=os.path.getmtime):
                if len(cached_files) < self.max_cache_files:
                    break

                filename = os.path.basename(f)
                if filename in protected:
                    continue  # 사용 중인 파일은 건너뜀

                try:
                    os.remove(f)
                    cached_files.remove(f)
                    print(f"  [Cache] Removed old file: {filename}")
                except Exception:
                    pass

    def get_next(self, timeout: float = 300.0) -> Tuple[int, str, str]:
        """다음 PT 파일 반환 (다운로드 완료 대기)

        Returns:
            (idx, pt_path, local_path)
        """
        return self.ready_queue.get(timeout=timeout)

    def stop(self):
        """프리페치 중지"""
        self.stop_event.set()
        self.executor.shutdown(wait=False)
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=2.0)


class AsyncEmbeddingPipeline:
    """비동기 임베딩 파이프라인

    CPU에서 미리 N개 배치의 임베딩을 계산해두고
    TPU 학습과 오버랩
    """

    def __init__(self, embedding_provider: Any, num_prefetch: int = 4):
        self.embedding_provider = embedding_provider
        self.num_prefetch = num_prefetch

        # 입력 큐 (captions)
        self.input_queue = queue.Queue(maxsize=num_prefetch * 2)
        # 출력 큐 (embeddings)
        self.output_queue = queue.Queue(maxsize=num_prefetch)

        self.stop_event = threading.Event()

        # Worker thread
        self.worker = threading.Thread(target=self._encode_loop, daemon=True)
        self.worker.start()

    def _encode_loop(self):
        """임베딩 계산 루프"""
        while not self.stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=1.0)
                if item is None:
                    self.output_queue.put(None)
                    break

                batch_idx, captions = item

                # 임베딩 계산 (CPU)
                embeddings = self.embedding_provider.batch_encode(
                    captions, batch_size=256, normalize=True
                )

                self.output_queue.put((batch_idx, embeddings))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Embedding error: {e}")
                continue

    def submit(self, batch_idx: int, captions: List[str]):
        """임베딩 계산 요청"""
        self.input_queue.put((batch_idx, captions))

    def get_result(self, timeout: float = 30.0) -> Optional[Tuple[int, np.ndarray]]:
        """임베딩 결과 반환"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """파이프라인 중지"""
        self.stop_event.set()
        self.input_queue.put(None)
        self.worker.join(timeout=5.0)


class GCSDataLoaderSession:
    """GCS 기반 데이터로더 세션

    PT 파일 + Parquet caption
    - Precomputed mode: PT 파일의 embeddings 직접 사용 (embedding_provider=None 허용)
    - Legacy mode: 실시간 임베딩 계산 (embedding_provider 필요)
    """

    def __init__(self, batch_size: int, parquet_path: str,
                 embedding_provider: Any = None, gcs_bucket: str = "",
                 cache_dir: Optional[str] = None,
                 num_workers: int = 4,
                 prefetch_ahead: int = 4,
                 max_cache_files: int = 3):
        self.batch_size = batch_size
        self.embedding_provider = embedding_provider
        self.num_workers = num_workers
        self.prefetch_ahead = prefetch_ahead
        self.max_cache_files = max_cache_files

        # GCS 핸들러
        self.gcs_handler = GCSFileHandler(gcs_bucket, cache_dir)

        # PT 파일 목록
        self.pt_files = self.gcs_handler.list_pt_files()
        print(f"Found {len(self.pt_files)} PT files in {gcs_bucket}")

        # PT 로더
        self.pt_loader = PTFileLoader()

        # Parquet 메타데이터 로드 (key -> caption 매핑)
        self.parquet_cache = None
        self._load_parquet_metadata(parquet_path)

        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def _load_parquet_metadata(self, parquet_path: str):
        """Parquet 메타데이터 로드 (key -> caption_llava 매핑)"""
        try:
            print(f"Loading parquet metadata from {parquet_path}...")

            # GCS에서 다운로드 (필요 시)
            if parquet_path.startswith("gs://"):
                local_path = self.gcs_handler.download_file(parquet_path)
            else:
                local_path = parquet_path

            if os.path.exists(local_path):
                # key와 caption_llava 컬럼만 로드
                table = pq.read_table(local_path, columns=['key', 'caption_llava'])

                # key -> caption 딕셔너리 생성
                key_to_caption = {}
                for batch in table.to_batches():
                    keys = batch['key'].to_pylist()
                    captions = batch['caption_llava'].to_pylist()
                    for k, c in zip(keys, captions):
                        key_to_caption[int(k)] = c if c else ""

                self.parquet_cache = ParquetCache(key_to_caption=key_to_caption)
                print(f"✓ Loaded {len(key_to_caption)} key->caption mappings")
            else:
                print(f"Warning: Parquet file not found: {local_path}")
                self.parquet_cache = ParquetCache(key_to_caption={})

        except Exception as e:
            print(f"Warning: Could not load parquet metadata: {e}")
            import traceback
            traceback.print_exc()
            self.parquet_cache = ParquetCache(key_to_caption={})

    def _download_pt_file(self, gcs_path: str) -> str:
        """PT 파일 다운로드 (캐시 관리 포함)"""
        filename = os.path.basename(gcs_path)
        local_path = os.path.join(self.gcs_handler.cache_dir, filename)

        if os.path.exists(local_path):
            return local_path

        # 캐시 크기 관리
        cached_files = glob.glob(os.path.join(self.gcs_handler.cache_dir, "*.pt"))
        if len(cached_files) >= self.max_cache_files:
            # 가장 오래된 파일 삭제
            oldest = min(cached_files, key=os.path.getmtime)
            os.remove(oldest)
            print(f"  Removed cached file: {os.path.basename(oldest)}")

        # 다운로드 (google-cloud-storage 사용)
        print(f"  Downloading {filename}...")
        local_path = self.gcs_handler.download_file(gcs_path, local_path)
        return local_path

    def _load_pt_data(self, pt_path: str) -> dict:
        """PT 파일 로드"""
        local_path = self._download_pt_file(pt_path)
        data = self.pt_loader.load(local_path)
        return data

    def get_captions_for_keys(self, keys: np.ndarray) -> List[str]:
        """keys에 해당하는 captions 반환"""
        captions = []
        for key in keys:
            key_int = int(key)
            caption = self.parquet_cache.key_to_caption.get(key_int, "")
            captions.append(caption)
        return captions

    def get_epoch_loader(self, epoch: int, steps_per_epoch: int,
                         use_precomputed: bool = True,
                         num_download_workers: int = 4,
                         num_load_workers: int = 2) -> 'EpochDataLoader':
        """에포크용 데이터로더 생성"""
        return EpochDataLoader(
            session=self,
            epoch=epoch,
            steps_per_epoch=steps_per_epoch,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_ahead=self.prefetch_ahead,
            num_download_workers=num_download_workers,
            num_load_workers=num_load_workers,
            use_precomputed=use_precomputed
        )

    def shutdown(self):
        """세션 종료"""
        self.executor.shutdown(wait=True)
        gc.collect()


class EpochDataLoader:
    """에포크 단위 데이터로더 (3단계 비동기 파이프라인)

    Layer 1: PT 파일 프리페치 (PTFilePrefetchManager)
    Layer 2: PT 파일 로딩 (ProcessPoolExecutor - GIL 우회)
    Layer 3: 배치 샘플링 (ThreadPoolExecutor)

    GIL 최적화:
    - PT 로딩을 ProcessPoolExecutor로 실행 (torch.load GIL 우회)
    - Lock-free 배치 버퍼 (deque + atomic counter)
    - NumPy 상태로 전달, JAX 변환은 메인 스레드에서 일괄 처리
    """

    def __init__(self, session: GCSDataLoaderSession, epoch: int,
                 steps_per_epoch: int, batch_size: int,
                 num_workers: int = 8, prefetch_ahead: int = 4,
                 num_download_workers: int = 4,
                 num_load_workers: int = 2,
                 use_precomputed: bool = True):
        self.session = session
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_ahead = prefetch_ahead
        self.num_load_workers = num_load_workers
        self.use_precomputed = use_precomputed

        # === Layer 1: PT 파일 프리페치 ===
        self.prefetch_manager = PTFilePrefetchManager(
            gcs_handler=session.gcs_handler,
            pt_files=session.pt_files,
            prefetch_ahead=prefetch_ahead,
            max_workers=num_download_workers,
            max_cache_files=session.max_cache_files
        )

        # === Layer 2: PT 파일 로딩 (ProcessPoolExecutor - GIL 우회) ===
        # ProcessPoolExecutor는 별도 프로세스에서 torch.load 실행
        self.load_executor = ProcessPoolExecutor(max_workers=num_load_workers)
        self.loaded_pt_queue: queue.Queue[Tuple[int, dict]] = queue.Queue(maxsize=num_load_workers + 1)

        # === Layer 3: 배치 샘플링 ===
        self.batch_executor = ThreadPoolExecutor(max_workers=num_workers)
        self.batch_queue: queue.Queue[Optional[Tuple[np.ndarray, np.ndarray]]] = queue.Queue(
            maxsize=prefetch_ahead * 2
        )

        # === Lock-free 배치 버퍼 (GIL 최적화) ===
        # Dict 대신 고정 크기 리스트 + atomic counter 사용
        self._batch_slots = [None] * (steps_per_epoch + 1)  # Pre-allocated slots
        self._batch_ready = [False] * (steps_per_epoch + 1)  # Ready flags
        self.next_batch_idx = 0  # Atomic read/write in CPython

        # 레거시 호환용 (제거 예정)
        self.batch_buffer: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.buffer_lock = threading.Lock()

        # 제어
        self.stop_event = threading.Event()
        self.started = False

        # 비동기 임베딩 파이프라인 (precomputed mode에서는 불필요)
        if not use_precomputed:
            self.embedding_pipeline = AsyncEmbeddingPipeline(
                session.embedding_provider,
                num_prefetch=prefetch_ahead
            )
        else:
            self.embedding_pipeline = None

        # Latent 버퍼 (legacy mode용)
        self.latent_buffer: Dict[int, np.ndarray] = {}

        # 파이프라인 스레드들
        self.load_thread: Optional[threading.Thread] = None
        self.sample_thread: Optional[threading.Thread] = None
        self.combine_thread: Optional[threading.Thread] = None

    def _start_pipeline(self):
        """파이프라인 시작"""
        if self.started:
            return
        self.started = True

        # Layer 1: 프리페치 시작
        self.prefetch_manager.start(start_idx=0)

        # Layer 2: 로딩 스레드 시작
        self.load_thread = threading.Thread(target=self._load_loop, daemon=True)
        self.load_thread.start()

        # Layer 3: 샘플링 스레드 시작
        self.sample_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.sample_thread.start()

        # Legacy mode용 결합 스레드
        if not self.use_precomputed:
            self.combine_thread = threading.Thread(target=self._combine_batches, daemon=True)
            self.combine_thread.start()

    def _load_loop(self):
        """PT 파일 로딩 루프 (Layer 2) - ProcessPoolExecutor로 GIL 우회"""
        pt_files_needed = (self.steps_per_epoch // 50) + 2  # 필요한 PT 파일 수 추정
        files_submitted = 0
        files_completed = 0

        # 진행 중인 로딩 작업 (idx -> (future, local_path))
        pending_loads: Dict[int, Tuple[Future, str]] = {}

        while files_completed < pt_files_needed and not self.stop_event.is_set():
            # 1. 새 로딩 작업 시작 (ProcessPoolExecutor - GIL 우회)
            while len(pending_loads) < self.num_load_workers and files_submitted < pt_files_needed:
                try:
                    idx, pt_path, local_path = self.prefetch_manager.get_next(timeout=5.0)
                    # 모듈 레벨 함수 사용 (pickle 가능)
                    future = self.load_executor.submit(_load_pt_file_worker, local_path)
                    pending_loads[idx] = (future, local_path)
                    files_submitted += 1
                except queue.Empty:
                    break  # 프리페치 큐가 비었으면 대기

            # 2. 완료된 로딩 수집 및 큐에 전달
            completed_indices = []
            for idx, (future, local_path) in pending_loads.items():
                if future.done():
                    try:
                        pt_data = future.result(timeout=1.0)
                        self.loaded_pt_queue.put((idx, pt_data), timeout=30.0)
                        files_completed += 1
                        completed_indices.append(idx)
                    except Exception as e:
                        print(f"[Load] Error loading {local_path}: {e}")
                        completed_indices.append(idx)
                        files_completed += 1

            for idx in completed_indices:
                del pending_loads[idx]

            # CPU 과부하 방지 (polling interval)
            if not completed_indices and len(pending_loads) > 0:
                time.sleep(0.001)

        # 남은 작업 완료 대기
        for idx, (future, local_path) in pending_loads.items():
            try:
                pt_data = future.result(timeout=120.0)
                self.loaded_pt_queue.put((idx, pt_data), timeout=30.0)
            except Exception as e:
                print(f"[Load] Final error loading {local_path}: {e}")

        # 종료 신호
        try:
            self.loaded_pt_queue.put((-1, None), timeout=5.0)
        except queue.Full:
            pass

    def _sample_loop(self):
        """배치 샘플링 루프 (Layer 3) - Lock-free 최적화"""
        rng = np.random.RandomState(self.epoch * 1000)

        current_pt_data: Optional[dict] = None
        steps_on_current_pt = 0
        steps_per_pt = 50

        step = 0
        pending_futures: Dict[int, Future] = {}

        while step < self.steps_per_epoch and not self.stop_event.is_set():
            # 새 PT 데이터 필요한지 확인
            if current_pt_data is None or steps_on_current_pt >= steps_per_pt:
                try:
                    timeout = 300.0 if current_pt_data is None else 120.0
                    pt_idx, pt_data = self.loaded_pt_queue.get(timeout=timeout)
                    if pt_idx < 0:  # 종료 신호
                        break
                    current_pt_data = pt_data
                    steps_on_current_pt = 0
                    if step == 0:
                        print(f"  [Sample] First PT file loaded (GIL-free)")
                except queue.Empty:
                    print(f"[Sample] PT data queue timeout at step {step}")
                    continue

            # 병렬 배치 샘플링 작업 제출
            batch_to_submit = min(self.num_workers, self.steps_per_epoch - step)

            for i in range(batch_to_submit):
                step_idx = step + i
                if step_idx not in pending_futures and step_idx < self.steps_per_epoch:
                    seed = self.epoch * 10000 + step_idx
                    future = self.batch_executor.submit(
                        self._sample_single_batch,
                        current_pt_data,
                        step_idx,
                        seed
                    )
                    pending_futures[step_idx] = future

            # 완료된 작업 수집 (Lock-free: 슬롯에 직접 저장)
            completed_steps = []
            for step_idx in list(pending_futures.keys()):
                future = pending_futures[step_idx]
                if future.done():
                    try:
                        batch_data = future.result()
                        if batch_data is not None:
                            # Lock-free: atomic write to pre-allocated slot
                            self._batch_slots[step_idx] = batch_data
                            self._batch_ready[step_idx] = True
                        completed_steps.append(step_idx)
                    except Exception as e:
                        print(f"[Sample] Error at step {step_idx}: {e}")
                        completed_steps.append(step_idx)

            for step_idx in completed_steps:
                del pending_futures[step_idx]

            # 순서대로 batch_queue에 전달 (Lock-free: ready flag 체크)
            while self._batch_ready[self.next_batch_idx]:
                batch_data = self._batch_slots[self.next_batch_idx]
                self._batch_slots[self.next_batch_idx] = None  # Clear slot
                self._batch_ready[self.next_batch_idx] = False
                try:
                    put_timeout = 600 if self.next_batch_idx == 0 else 60
                    self.batch_queue.put(batch_data, timeout=put_timeout)
                    if self.next_batch_idx == 0:
                        print(f"  ✓ First batch ready (shape: {batch_data[0].shape})")
                    self.next_batch_idx += 1
                    step += 1
                    steps_on_current_pt += 1
                except queue.Full:
                    # 큐가 가득 차면 다시 슬롯에 저장
                    self._batch_slots[self.next_batch_idx] = batch_data
                    self._batch_ready[self.next_batch_idx] = True
                    break

            time.sleep(0.005)  # 5ms로 감소 (Lock-free라 오버헤드 적음)

        # 남은 pending futures 처리
        for step_idx, future in pending_futures.items():
            try:
                batch_data = future.result(timeout=5.0)
                if batch_data is not None:
                    self._batch_slots[step_idx] = batch_data
                    self._batch_ready[step_idx] = True
            except Exception:
                pass

        # 버퍼에 남은 배치 전달
        while self.next_batch_idx < self.steps_per_epoch and self._batch_ready[self.next_batch_idx]:
            batch_data = self._batch_slots[self.next_batch_idx]
            self._batch_slots[self.next_batch_idx] = None
            self._batch_ready[self.next_batch_idx] = False
            try:
                self.batch_queue.put(batch_data, timeout=5.0)
                self.next_batch_idx += 1
            except queue.Full:
                break

        # 종료 신호
        self.batch_queue.put(None)

    def _sample_single_batch(self, pt_data: dict, step_idx: int, seed: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """단일 배치 샘플링 (워커 함수) - GIL 최적화

        pt_data['latents']는 이미 numpy 배열 (_load_pt_file_worker에서 변환됨)
        """
        try:
            rng = np.random.RandomState(seed)

            num_samples = len(pt_data['keys'])
            indices = rng.randint(0, num_samples, size=self.batch_size)

            # Latents: 이미 numpy 배열 (GIL-free 로딩에서 변환됨)
            # Shape: (N, 3, 4, 32, 32) -> 랜덤 crop -> (B, 32, 32, 4) NHWC
            latents = pt_data['latents']

            # numpy 배열인지 확인 (ProcessPoolExecutor에서 이미 변환됨)
            if isinstance(latents, torch.Tensor):
                latents_np = latents.numpy()
            else:
                latents_np = latents  # 이미 numpy

            latents_subset = latents_np[indices]  # (B, 3, 4, 32, 32)

            crop_indices = rng.randint(0, 3, size=self.batch_size)
            latents_selected = latents_subset[np.arange(self.batch_size), crop_indices]  # (B, 4, 32, 32)
            latents_nhwc = np.transpose(latents_selected, (0, 2, 3, 1))  # (B, 32, 32, 4)

            # Embeddings (이미 numpy 배열)
            if self.use_precomputed and 'embeddings' in pt_data:
                embeddings = pt_data['embeddings'][indices]
            else:
                # Legacy mode: 실시간 계산 필요
                keys = pt_data['keys'][indices]
                captions = self.session.get_captions_for_keys(keys)
                # Lock-free: 슬롯에 직접 저장
                self._batch_slots[step_idx] = latents_nhwc
                self.latent_buffer[step_idx] = latents_nhwc
                self.embedding_pipeline.submit(step_idx, captions)
                return None  # combine_thread에서 처리

            return (latents_nhwc, embeddings)

        except Exception as e:
            print(f"[Sample] Batch error at {step_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _combine_batches(self):
        """Latent와 임베딩 결합 (Legacy mode용)"""
        completed = 0
        first_batch = True

        while completed < self.steps_per_epoch:
            if self.stop_event.is_set():
                break

            try:
                timeout = 300.0 if first_batch else 60.0
                result = self.embedding_pipeline.get_result(timeout=timeout)
                if result is None:
                    continue

                batch_idx, embeddings = result
                if batch_idx < 0:
                    break

                first_batch = False

                with self.buffer_lock:
                    if batch_idx in self.latent_buffer:
                        latents = self.latent_buffer.pop(batch_idx)
                    else:
                        print(f"  Warning: batch_idx {batch_idx} not in latent_buffer")
                        continue

                put_timeout = 600 if completed == 0 else 60
                self.batch_queue.put((latents, embeddings), timeout=put_timeout)
                completed += 1

                if completed == 1:
                    print(f"  ✓ First batch ready (embedding shape: {embeddings.shape})")

            except queue.Empty:
                print(f"Combine timeout at batch {completed}")
                break
            except Exception as e:
                print(f"Combine error: {e}")
                continue

        self.batch_queue.put(None)

    def get_batches(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """배치 이터레이터"""
        self._start_pipeline()

        batch_count = 0

        while batch_count < self.steps_per_epoch:
            try:
                timeout = 600 if batch_count == 0 else 120
                batch = self.batch_queue.get(timeout=timeout)

                if batch is None:
                    break

                latents, embeddings = batch
                yield jnp.array(latents), jnp.array(embeddings)
                batch_count += 1

            except queue.Empty:
                print(f"Batch queue timeout at step {batch_count} (waited {timeout}s)")
                break

    def stop(self):
        """로더 중지"""
        self.stop_event.set()
        self.prefetch_manager.stop()
        self.load_executor.shutdown(wait=False)
        self.batch_executor.shutdown(wait=False)
        if self.embedding_pipeline:
            self.embedding_pipeline.stop()
        if self.load_thread:
            self.load_thread.join(timeout=5.0)
        if self.sample_thread:
            self.sample_thread.join(timeout=5.0)
        if self.combine_thread:
            self.combine_thread.join(timeout=5.0)