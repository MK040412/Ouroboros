"""
GCS DataLoader with Async Embedding Pipeline for TPU Training

최적화 전략:
1. PT 파일에서 latent 로드
2. Parquet에서 caption 로드 (key로 매핑)
3. 비동기 임베딩 계산 (CPU에서 미리 N개 배치 준비)
4. Double buffering으로 TPU 학습과 데이터 로딩 오버랩

3단계 비동기 파이프라인:
- Layer 1: GCS Prefetch (ThreadPoolExecutor) - PT 파일 미리 다운로드
- Layer 2: PT Loading (ThreadPoolExecutor) - torch.load 병렬 실행
- Layer 3: Batch Sampling (ThreadPoolExecutor) - 랜덤 배치 샘플링
"""

import os
import gc
import glob
import queue
import threading
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Iterator, Any, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
from collections import deque
import time

# fork() 경고 억제 - torch.load 워커는 JAX 코드를 사용하지 않으므로 안전
warnings.filterwarnings("ignore", message="os.fork\\(\\) was called")

import numpy as np
import jax.numpy as jnp
import torch
import pyarrow.parquet as pq
from google.cloud import storage


# ============================================
# PT Loading with BFloat16 support
# ============================================

# BFloat16 역직렬화를 위한 커스텀 unpickler
import pickle
import io

class BFloat16Unpickler(pickle.Unpickler):
    """BFloat16 텐서를 float32로 자동 변환하는 Unpickler"""
    def find_class(self, module, name):
        # 기본 클래스 찾기
        return super().find_class(module, name)

def _torch_load_with_bf16_support(path: str):
    """BFloat16을 지원하는 torch.load 래퍼"""
    import sys

    # 방법 1: mmap=True로 시도 (메모리 효율적)
    try:
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except TypeError:
        # mmap 파라미터가 지원되지 않는 구버전
        pass
    except Exception as e:
        if "BFloat16" not in str(e):
            raise

    # 방법 2: 일반 로드
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        if "BFloat16" not in str(e):
            raise

    # 방법 3: pickle 직접 사용
    print(f"  [Load] Trying pickle fallback for {path}...")
    sys.stdout.flush()
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_pt_file_worker(local_path: str) -> dict:
    """PT 파일 로드 (별도 스레드에서 실행)

    torch.load는 C 확장이라 GIL 영향이 적음.
    BFloat16 텐서는 자동으로 float32로 변환.
    """
    import sys

    try:
        data = _torch_load_with_bf16_support(local_path)
    except Exception as e:
        print(f"  [Load] Failed to load {local_path}: {e}")
        sys.stdout.flush()
        raise

    # Tensor -> NumPy 변환 (bfloat16 -> float32 자동 변환)
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            # BFloat16은 numpy가 지원하지 않으므로 float32로 변환
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            return tensor.numpy()
        return np.array(tensor)

    result = {
        'keys': to_numpy(data['keys']),
        'latents': to_numpy(data['latents']),
    }

    # Precomputed embeddings 로드 (bfloat16 -> float32)
    if 'embeddings' in data:
        result['embeddings'] = to_numpy(data['embeddings'])

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

        # 파일별 다운로드 락 (동시 다운로드 방지)
        self._download_locks: Dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

        # 파일 참조 카운터 (사용 중인 파일 보호)
        self._file_refcount: Dict[str, int] = {}
        self._refcount_lock = threading.Lock()

    def acquire_file(self, filename: str):
        """파일 사용 시작 (refcount++)"""
        with self._refcount_lock:
            self._file_refcount[filename] = self._file_refcount.get(filename, 0) + 1

    def release_file(self, filename: str, delete_immediately: bool = True):
        """파일 사용 완료 (refcount--) 및 즉시 삭제

        Args:
            filename: 파일명
            delete_immediately: True이면 refcount가 0이 될 때 즉시 삭제
        """
        should_delete = False
        with self._refcount_lock:
            if filename in self._file_refcount:
                self._file_refcount[filename] -= 1
                if self._file_refcount[filename] <= 0:
                    del self._file_refcount[filename]
                    should_delete = delete_immediately

        # 락 밖에서 파일 삭제 (I/O 블록킹 최소화)
        if should_delete:
            file_path = os.path.join(self.cache_dir, filename)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"  [Cache] Deleted completed file: {filename}")
            except Exception as e:
                pass  # 삭제 실패해도 계속 진행

    def is_file_in_use(self, filename: str) -> bool:
        """파일이 사용 중인지 확인"""
        with self._refcount_lock:
            return self._file_refcount.get(filename, 0) > 0

    def _cleanup_old_cache_dirs(self):
        """이전 임시 캐시 디렉토리 및 고아 임시 파일 정리"""
        try:
            import shutil

            # 1. 랜덤 캐시 디렉토리 정리
            for item in Path("/tmp").iterdir():
                # gcs_cache_로 시작하지만 gcs_cache_worker_가 아닌 디렉토리 삭제
                if item.is_dir() and item.name.startswith("gcs_cache_"):
                    if not item.name.startswith("gcs_cache_worker_"):
                        print(f"  Cleaning old cache dir: {item}")
                        shutil.rmtree(item, ignore_errors=True)

            # 2. 현재 캐시 디렉토리의 고아 .tmp 파일 정리
            cache_path = Path(self.cache_dir)
            if cache_path.exists():
                for tmp_file in cache_path.glob("*.tmp*"):
                    try:
                        # 10분 이상 된 임시 파일만 삭제
                        if time.time() - tmp_file.stat().st_mtime > 600:
                            print(f"  Cleaning orphan temp file: {tmp_file.name}")
                            tmp_file.unlink()
                    except Exception:
                        pass

        except Exception as e:
            print(f"  Warning: Could not clean old caches: {e}")

    def _get_file_lock(self, filename: str) -> threading.Lock:
        """파일별 락 반환 (없으면 생성)"""
        with self._locks_lock:
            if filename not in self._download_locks:
                self._download_locks[filename] = threading.Lock()
            return self._download_locks[filename]

    def download_file(self, gcs_path: str, local_path: Optional[str] = None,
                      min_file_size: int = 1024 * 1024,  # 1MB 최소 크기 (PT 파일은 GB 단위)
                      max_retries: int = 3) -> str:
        """GCS에서 파일 다운로드 (스레드 세이프, 재시도 지원)"""
        import sys

        if local_path is None:
            filename = os.path.basename(gcs_path)
            local_path = os.path.join(self.cache_dir, filename)

        filename = os.path.basename(local_path)
        file_lock = self._get_file_lock(filename)

        # 파일별 락으로 동시 다운로드 방지
        with file_lock:
            # 기존 파일이 있으면 크기 확인 (손상된 파일 감지)
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                if file_size >= min_file_size:
                    return local_path
                else:
                    # 손상된 파일 삭제
                    print(f"  Removing corrupted file: {local_path} (size={file_size}, min={min_file_size})")
                    sys.stdout.flush()
                    try:
                        os.remove(local_path)
                    except OSError:
                        pass

            # Parse blob name from gcs_path
            if gcs_path.startswith("gs://"):
                blob_name = gcs_path[5:].split('/', 1)[1] if '/' in gcs_path[5:] else ''
            else:
                blob_name = gcs_path

            # 스레드별 고유 임시 파일 (PID + thread ID)
            thread_id = threading.current_thread().ident
            temp_path = f"{local_path}.tmp.{os.getpid()}_{thread_id}"

            last_error = None
            for attempt in range(max_retries):
                try:
                    # 이전 임시 파일 정리
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    # GCS에서 다운로드
                    blob = self.bucket.blob(blob_name)

                    # blob 존재 확인
                    if not blob.exists():
                        raise Exception(f"Blob does not exist: {blob_name}")

                    # 예상 파일 크기 확인
                    blob.reload()
                    expected_size = blob.size
                    if expected_size and expected_size < min_file_size:
                        print(f"  Warning: Blob {filename} is smaller than expected ({expected_size} bytes)")
                        sys.stdout.flush()

                    # 다운로드
                    blob.download_to_filename(temp_path)

                    # 다운로드 성공 확인
                    if not os.path.exists(temp_path):
                        raise Exception(f"Temp file not created: {temp_path}")

                    actual_size = os.path.getsize(temp_path)

                    # 예상 크기와 비교 (있으면)
                    if expected_size and actual_size != expected_size:
                        raise Exception(f"Size mismatch: expected {expected_size}, got {actual_size}")

                    # 최소 크기 확인
                    if actual_size < min_file_size:
                        raise Exception(f"Downloaded file too small: {actual_size} < {min_file_size}")

                    # 원자적 이동
                    os.rename(temp_path, local_path)
                    return local_path

                except Exception as e:
                    last_error = e
                    print(f"  Download attempt {attempt + 1}/{max_retries} failed for {filename}: {e}")
                    sys.stdout.flush()

                    # 임시 파일 정리
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass

                    # 마지막 시도가 아니면 잠시 대기 후 재시도
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s

            # 모든 재시도 실패
            raise Exception(f"GCS download failed after {max_retries} attempts for {gcs_path}: {last_error}")

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
        import sys
        print(f"  [Prefetch] Starting from idx={start_idx}, {len(self.pt_files)} PT files available")
        sys.stdout.flush()

        self.current_idx = start_idx
        self.next_deliver_idx = start_idx
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        self.scheduler_thread.start()
        print(f"  [Prefetch] Scheduler thread started")
        sys.stdout.flush()

    def _scheduler_loop(self):
        """프리페치 스케줄링 루프"""
        import sys
        print(f"  [Prefetch] Scheduler loop running...")
        sys.stdout.flush()
        loop_count = 0

        while not self.stop_event.is_set():
            loop_count += 1
            if loop_count == 1:
                print(f"  [Prefetch] First loop iteration...")
                sys.stdout.flush()
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
        import sys

        # PT 파일 최소 크기: 1MB (실제로는 GB 단위이지만 보수적으로)
        MIN_PT_FILE_SIZE = 1024 * 1024

        pt_path = self.pt_files[idx % len(self.pt_files)]
        filename = os.path.basename(pt_path)
        local_path = os.path.join(self.gcs_handler.cache_dir, filename)

        # 캐시 용량 관리 (다운로드 전)
        self._manage_cache_size()

        # 이미 존재하고 충분한 크기면 스킵
        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            if file_size >= MIN_PT_FILE_SIZE:
                print(f"  [Prefetch] {filename} already cached ({file_size // (1024*1024)}MB)")
                sys.stdout.flush()
                # 참조 획득 (Layer 2에서 사용될 때까지 삭제 방지)
                self.gcs_handler.acquire_file(filename)
                return local_path
            else:
                print(f"  [Prefetch] {filename} corrupted (only {file_size} bytes), re-downloading...")
                sys.stdout.flush()

        # GCS에서 다운로드 (download_file이 이제 락과 재시도 처리)
        print(f"  [Prefetch] Downloading {filename} from GCS...")
        sys.stdout.flush()
        result = self.gcs_handler.download_file(pt_path, local_path, min_file_size=MIN_PT_FILE_SIZE)
        final_size = os.path.getsize(result)
        print(f"  [Prefetch] Downloaded {filename} ({final_size // (1024*1024)}MB)")
        sys.stdout.flush()
        # 참조 획득 (Layer 2에서 사용될 때까지 삭제 방지)
        self.gcs_handler.acquire_file(filename)
        return result

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
        """캐시 용량 관리 (오래된 파일 삭제, 사용 중인 파일 보호)

        NOTE: 레이스 컨디션 방지를 위해 보수적으로 동작:
        - max_cache_files의 80%를 초과할 때만 삭제 시작
        - 삭제 시 최소 10%의 여유 공간 확보 목표
        """
        cached_files = glob.glob(
            os.path.join(self.gcs_handler.cache_dir, "*.pt")
        )

        # 80% 초과 시에만 삭제 시작 (여유 공간 확보)
        threshold = int(self.max_cache_files * 0.8)
        target = int(self.max_cache_files * 0.7)  # 70%까지 줄이는 것이 목표

        if len(cached_files) >= threshold:
            # 보호 대상 파일 목록
            protected = self._get_protected_files()

            # 가장 오래된 파일부터 삭제 (보호 대상 제외)
            for f in sorted(cached_files, key=os.path.getmtime):
                if len(cached_files) <= target:
                    break

                filename = os.path.basename(f)

                # 1. 참조 카운터 확인 (Layer 2에서 사용 중인 파일)
                if self.gcs_handler.is_file_in_use(filename):
                    continue

                # 2. 기존 보호 로직 (다운로드 중, 완료 대기 중, prefetch 범위)
                if filename in protected:
                    continue

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

    분산 학습 시 Worker별 PT 파일 샤딩:
    - 각 Worker는 자신에게 할당된 PT 파일만 다운로드/처리
    - Worker 0: idx % num_processes == 0 인 파일들
    - Worker 1: idx % num_processes == 1 인 파일들
    - 이를 통해 GCS 대역폭 및 디스크 사용량 최적화
    """

    def __init__(self, batch_size: int, parquet_path: str,
                 embedding_provider: Any = None, gcs_bucket: str = "",
                 cache_dir: Optional[str] = None,
                 num_workers: int = 4,
                 prefetch_ahead: int = 4,
                 max_cache_files: int = 3,
                 shard_pt_files: bool = True):
        self.batch_size = batch_size
        self.embedding_provider = embedding_provider
        self.num_workers = num_workers
        self.prefetch_ahead = prefetch_ahead
        self.max_cache_files = max_cache_files

        # GCS 핸들러
        self.gcs_handler = GCSFileHandler(gcs_bucket, cache_dir)

        # PT 파일 목록 (전체)
        all_pt_files = self.gcs_handler.list_pt_files()
        print(f"Found {len(all_pt_files)} PT files in {gcs_bucket}")

        # Worker별 PT 파일 샤딩 (분산 학습 최적화)
        if shard_pt_files:
            process_index = int(os.environ.get('JAX_PROCESS_INDEX', '0'))
            num_processes = int(os.environ.get('JAX_NUM_PROCESSES', '1'))

            if num_processes > 1:
                # 각 Worker가 담당할 PT 파일만 선택 (round-robin 분배)
                self.pt_files = [f for i, f in enumerate(all_pt_files) if i % num_processes == process_index]
                print(f"  Worker {process_index}/{num_processes}: Sharded to {len(self.pt_files)} PT files "
                      f"(indices {process_index}, {process_index + num_processes}, {process_index + 2*num_processes}, ...)")
            else:
                self.pt_files = all_pt_files
        else:
            self.pt_files = all_pt_files

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

    @property
    def total_samples(self) -> int:
        """전체 샘플 수 반환 (parquet 메타데이터 기준)"""
        if self.parquet_cache and self.parquet_cache.key_to_caption:
            return len(self.parquet_cache.key_to_caption)
        return 0

    def calculate_steps_per_epoch(self, global_batch_size: int) -> int:
        """에포크당 스텝 수 계산

        Args:
            global_batch_size: 전체 배치 크기 (모든 worker 합산)

        Returns:
            steps_per_epoch: 에포크당 스텝 수
        """
        total = self.total_samples
        if total == 0:
            raise ValueError("Cannot calculate steps: no samples found in parquet metadata")

        steps = total // global_batch_size
        print(f"  Calculated steps_per_epoch: {total:,} samples / {global_batch_size} batch = {steps:,} steps")
        return steps

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
    Layer 2: PT 파일 로딩 (ThreadPoolExecutor)
    Layer 3: 배치 샘플링 (ThreadPoolExecutor)

    최적화:
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

        # === Layer 2: PT 파일 로딩 (ThreadPoolExecutor) ===
        # Note: ProcessPoolExecutor + spawn은 TPU 환경에서 초기화 지연 문제 발생
        # ThreadPoolExecutor 사용 - torch.load는 C 확장이라 GIL 영향 적음
        self.load_executor = ThreadPoolExecutor(max_workers=num_load_workers)
        # PT 로드 결과 큐 (블록킹 방지를 위해 여유있게 설정)
        self.loaded_pt_queue: queue.Queue[Tuple[int, dict]] = queue.Queue(maxsize=prefetch_ahead + num_load_workers)

        # === Layer 3: 배치 샘플링 ===
        self.batch_executor = ThreadPoolExecutor(max_workers=num_workers)
        # 배치 큐 (TPU 학습이 느릴 때도 블록되지 않도록 여유있게 설정)
        self.batch_queue: queue.Queue[Optional[Tuple[np.ndarray, np.ndarray]]] = queue.Queue(
            maxsize=prefetch_ahead * 4
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
        import sys
        if self.started:
            return
        self.started = True

        print(f"  [Pipeline] Starting 3-layer async pipeline...")
        sys.stdout.flush()

        # Layer 1: 프리페치 시작
        print(f"  [Pipeline] Layer 1: Starting prefetch manager...")
        sys.stdout.flush()
        self.prefetch_manager.start(start_idx=0)

        # Layer 2: 로딩 스레드 시작
        print(f"  [Pipeline] Layer 2: Starting load thread...")
        sys.stdout.flush()
        self.load_thread = threading.Thread(target=self._load_loop, daemon=True)
        self.load_thread.start()

        # Layer 3: 샘플링 스레드 시작
        print(f"  [Pipeline] Layer 3: Starting sample thread...")
        sys.stdout.flush()
        self.sample_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.sample_thread.start()

        # Legacy mode용 결합 스레드
        if not self.use_precomputed:
            self.combine_thread = threading.Thread(target=self._combine_batches, daemon=True)
            self.combine_thread.start()

        print(f"  [Pipeline] All threads started, waiting for first batch...")
        sys.stdout.flush()

    def _load_loop(self):
        """PT 파일 로딩 루프 (Layer 2)"""
        import sys

        # 실제 PT 파일 개수 사용 (순환하지 않음)
        pt_files_available = len(self.session.pt_files)
        pt_files_needed = pt_files_available

        print(f"  [LoadLoop] Started, will load {pt_files_needed} PT files")
        sys.stdout.flush()
        files_submitted = 0
        files_completed = 0

        # 진행 중인 로딩 작업 (idx -> (future, local_path, filename))
        pending_loads: Dict[int, Tuple[Future, str, str]] = {}

        while files_completed < pt_files_needed and not self.stop_event.is_set():
            # 1. 새 로딩 작업 시작
            while len(pending_loads) < self.num_load_workers and files_submitted < pt_files_needed:
                try:
                    print(f"  [LoadLoop] Waiting for prefetch (submitted={files_submitted})...")
                    sys.stdout.flush()
                    idx, pt_path, local_path = self.prefetch_manager.get_next(timeout=5.0)
                    filename = os.path.basename(local_path)
                    print(f"  [LoadLoop] Got {filename}, submitting load...")
                    sys.stdout.flush()
                    # 모듈 레벨 함수 사용 (pickle 가능)
                    future = self.load_executor.submit(_load_pt_file_worker, local_path)
                    pending_loads[idx] = (future, local_path, filename)
                    files_submitted += 1
                except queue.Empty:
                    print(f"  [LoadLoop] Prefetch queue empty, waiting...")
                    sys.stdout.flush()
                    break  # 프리페치 큐가 비었으면 대기

            # 2. 완료된 로딩 수집 및 큐에 전달
            completed_indices = []
            for idx, (future, local_path, filename) in pending_loads.items():
                if future.done():
                    try:
                        pt_data = future.result(timeout=60.0)  # 3GB 로드에 충분한 시간
                        self.loaded_pt_queue.put((idx, pt_data), timeout=30.0)
                        files_completed += 1
                        completed_indices.append(idx)
                    except Exception as e:
                        error_type = type(e).__name__
                        error_msg = str(e) if str(e) else "(no message)"
                        print(f"[Load] Error loading {local_path}: {error_type}: {error_msg}")
                        import traceback
                        traceback.print_exc()
                        sys.stdout.flush()
                        completed_indices.append(idx)
                        files_completed += 1
                    finally:
                        # 로딩 완료 (성공/실패 무관) - 파일 참조 해제
                        self.session.gcs_handler.release_file(filename)

            for idx in completed_indices:
                del pending_loads[idx]

            # CPU 과부하 방지 (polling interval)
            if not completed_indices and len(pending_loads) > 0:
                time.sleep(0.001)

        # 남은 작업 완료 대기
        for idx, (future, local_path, filename) in pending_loads.items():
            try:
                pt_data = future.result(timeout=300.0)  # 3GB 로드에 충분한 시간
                self.loaded_pt_queue.put((idx, pt_data), timeout=30.0)
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e) if str(e) else "(no message)"
                print(f"[Load] Final error loading {local_path}: {error_type}: {error_msg}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
            finally:
                # 로딩 완료 - 파일 참조 해제
                self.session.gcs_handler.release_file(filename)

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
                    # PT 데이터 대기 timeout (3GB 파일 다운로드 시간 고려)
                    timeout = 600.0 if current_pt_data is None else 300.0
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

            # Precomputed mode: embeddings가 있는 샘플만 사용
            if self.use_precomputed:
                if 'embeddings' not in pt_data:
                    # PT 파일에 embeddings 키가 없으면 스킵
                    print(f"[Sample] Warning: PT file has no 'embeddings' key, skipping batch {step_idx}")
                    return None

                embeddings_array = pt_data['embeddings']

                # 유효한 샘플 인덱스 찾기 (NaN이 아닌 embeddings)
                # embeddings shape: (N, embed_dim)
                valid_mask = ~np.isnan(embeddings_array).any(axis=1)
                valid_indices = np.where(valid_mask)[0]

                if len(valid_indices) < self.batch_size:
                    print(f"[Sample] Warning: Only {len(valid_indices)} valid samples, need {self.batch_size}, skipping batch {step_idx}")
                    return None

                # 유효한 샘플 중에서만 랜덤 선택
                indices = rng.choice(valid_indices, size=self.batch_size, replace=False)
            else:
                # Legacy mode: 모든 샘플 사용 가능
                indices = rng.randint(0, num_samples, size=self.batch_size)

            # Latents: 이미 numpy 배열 (GIL-free 로딩에서 변환됨)
            # Shape: (N, 3, 4, 32, 32) -> 랜덤 crop -> (B, 32, 32, 4) NHWC
            latents = pt_data['latents']

            # numpy 배열인지 확인 (_load_pt_file_worker에서 이미 변환됨)
            if isinstance(latents, torch.Tensor):
                latents_np = latents.numpy()
            else:
                latents_np = latents  # 이미 numpy

            latents_subset = latents_np[indices]  # (B, 3, 4, 32, 32)

            crop_indices = rng.randint(0, 3, size=self.batch_size)
            latents_selected = latents_subset[np.arange(self.batch_size), crop_indices]  # (B, 4, 32, 32)
            latents_nhwc = np.transpose(latents_selected, (0, 2, 3, 1))  # (B, 32, 32, 4)

            # Embeddings
            if self.use_precomputed:
                # Precomputed mode: 이미 유효한 indices만 선택됨
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
                # 첫 배치는 600초, 이후 배치도 300초 (3GB PT 다운로드 시간 고려)
                timeout = 600 if batch_count == 0 else 300
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


# ============================================
# RAM Preload Data Loader (네트워크 병목 제거)
# ============================================

class RAMPreloadSession:
    """모든 PT 파일을 RAM에 프리로드하는 세션

    시작 시 1회 다운로드 → RAM에 상주 → 학습 중 네트워크 I/O 제로

    메모리 요구사항:
    - 92개 PT 파일 / 4 workers = 23개/worker
    - 23개 × 3GB = ~69GB/worker
    - 150GB RAM에서 ~81GB 여유
    """

    def __init__(self, batch_size: int, gcs_bucket: str,
                 parquet_path: str = None,
                 num_download_workers: int = 8,
                 shard_pt_files: bool = True):
        """
        Args:
            batch_size: 배치 크기 (per worker)
            gcs_bucket: GCS 버킷 URL (gs://...)
            parquet_path: Parquet 메타데이터 경로 (optional)
            num_download_workers: 병렬 다운로드 워커 수
            shard_pt_files: True면 worker별로 PT 파일 분배
        """
        import sys

        self.batch_size = batch_size
        self.num_download_workers = num_download_workers

        # Worker 정보
        self.process_index = int(os.environ.get('JAX_PROCESS_INDEX', '0'))
        self.num_processes = int(os.environ.get('JAX_NUM_PROCESSES', '1'))

        print(f"\n[RAMPreload] Initializing (Worker {self.process_index}/{self.num_processes})")
        sys.stdout.flush()

        # GCS 핸들러 (임시 캐시 디렉토리)
        self.gcs_handler = GCSFileHandler(
            gcs_bucket,
            cache_dir=f"/tmp/ram_preload_worker_{self.process_index}",
            clean_old_caches=True
        )

        # 전체 PT 파일 목록
        all_pt_files = self.gcs_handler.list_pt_files()
        print(f"[RAMPreload] Found {len(all_pt_files)} total PT files")
        sys.stdout.flush()

        # Worker별 PT 파일 샤딩
        if shard_pt_files and self.num_processes > 1:
            self.pt_files = [f for i, f in enumerate(all_pt_files)
                           if i % self.num_processes == self.process_index]
            print(f"[RAMPreload] Worker {self.process_index}: {len(self.pt_files)} PT files (sharded)")
        else:
            self.pt_files = all_pt_files

        # === RAM 데이터 저장소 ===
        # 모든 PT 파일의 데이터를 메모리에 저장
        self.ram_data: Dict[str, dict] = {}  # filename -> {keys, latents, embeddings}

        # 통합 인덱스 (빠른 랜덤 샘플링용)
        self.all_latents: Optional[np.ndarray] = None  # (N_total, 3, 4, 32, 32)
        self.all_embeddings: Optional[np.ndarray] = None  # (N_total, embed_dim)
        self.total_samples = 0

        # 프리로드 실행
        self._preload_all_pt_files()

    def _preload_all_pt_files(self):
        """모든 PT 파일을 RAM에 로드 (디스크 공간 최소화)

        전략: 1개씩 다운로드 → RAM 로드 → 즉시 삭제
        디스크 최대 사용량: ~3GB (1개 PT 파일)
        """
        import sys

        print(f"\n[RAMPreload] === Starting Preload ===")
        print(f"[RAMPreload] Files to load: {len(self.pt_files)}")
        print(f"[RAMPreload] Estimated RAM: {len(self.pt_files) * 3:.1f} GB")
        print(f"[RAMPreload] Strategy: Sequential download → RAM load → delete (max 3GB disk)")
        sys.stdout.flush()

        # 시작 전 캐시 디렉토리 정리 (이전 실행에서 남은 파일 삭제)
        self._cleanup_cache_directory()

        start_time = time.time()
        all_latents_list = []
        all_embeddings_list = []

        for i, pt_path in enumerate(self.pt_files):
            filename = os.path.basename(pt_path)
            file_start = time.time()

            try:
                # 1. 다운로드
                local_path = self._download_pt_file(pt_path)
                download_time = time.time() - file_start

                # 2. RAM에 로드
                load_start = time.time()
                data = _load_pt_file_worker(local_path)
                load_time = time.time() - load_start

                # 3. 즉시 디스크에서 삭제 (디스크 공간 확보)
                try:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                        # 삭제 확인
                        if os.path.exists(local_path):
                            print(f"  [RAMPreload] Warning: Failed to delete {filename}, file still exists")
                            sys.stdout.flush()
                except Exception as delete_err:
                    print(f"  [RAMPreload] Warning: Could not delete {filename}: {delete_err}")
                    sys.stdout.flush()
                    # 강제 삭제 재시도
                    try:
                        import shutil
                        if os.path.exists(local_path):
                            os.unlink(local_path)
                    except:
                        pass

                # 4. 유효한 샘플만 필터링 (NaN 임베딩 제외)
                if 'embeddings' in data:
                    valid_mask = ~np.isnan(data['embeddings']).any(axis=1)
                    num_valid = valid_mask.sum()

                    if num_valid > 0:
                        all_latents_list.append(data['latents'][valid_mask])
                        all_embeddings_list.append(data['embeddings'][valid_mask])
                        self.total_samples += num_valid

                # 진행 상황 출력
                total_time_file = time.time() - file_start
                mem_gb = self.total_samples * 3 * 4 * 32 * 32 * 4 / (1024**3)
                print(f"  [{i+1}/{len(self.pt_files)}] {filename}: "
                      f"{num_valid:,} samples, {download_time:.1f}s dl + {load_time:.1f}s load "
                      f"(total: {self.total_samples:,}, ~{mem_gb:.1f}GB)")
                sys.stdout.flush()

                # 메모리 정리 (중간중간)
                del data
                if (i + 1) % 10 == 0:
                    gc.collect()

            except Exception as e:
                print(f"  [{i+1}/{len(self.pt_files)}] ✗ {filename}: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()

        # 배열 통합 (메모리 효율적 방식: pre-allocate + copy + free)
        print(f"\n[RAMPreload] Consolidating arrays (memory-efficient)...")
        print(f"  Total samples to consolidate: {self.total_samples:,}")
        sys.stdout.flush()

        if all_latents_list and self.total_samples > 0:
            # 1. 첫 번째 배열에서 shape 정보 추출
            latent_shape = all_latents_list[0].shape[1:]  # (3, 4, 32, 32)
            embed_dim = all_embeddings_list[0].shape[1]   # 640

            # 2. 최종 배열 pre-allocate (원본 리스트 메모리 먼저 일부 해제)
            # 예상 메모리: latents ~130GB, embeddings ~7GB
            latent_bytes = self.total_samples * np.prod(latent_shape) * 4
            embed_bytes = self.total_samples * embed_dim * 4
            print(f"  Pre-allocating: latents ({self.total_samples}, {latent_shape}) = {latent_bytes/(1024**3):.1f}GB")
            print(f"  Pre-allocating: embeddings ({self.total_samples}, {embed_dim}) = {embed_bytes/(1024**3):.1f}GB")
            sys.stdout.flush()

            # GC 실행하여 가능한 메모리 확보
            gc.collect()

            self.all_latents = np.empty((self.total_samples, *latent_shape), dtype=np.float32)
            self.all_embeddings = np.empty((self.total_samples, embed_dim), dtype=np.float32)
            print(f"  Pre-allocation successful")
            sys.stdout.flush()

            # 3. 청크 단위로 복사하면서 원본 즉시 해제
            offset = 0
            for i, (lat_chunk, emb_chunk) in enumerate(zip(all_latents_list, all_embeddings_list)):
                chunk_size = lat_chunk.shape[0]
                self.all_latents[offset:offset + chunk_size] = lat_chunk
                self.all_embeddings[offset:offset + chunk_size] = emb_chunk
                offset += chunk_size

                # 원본 청크 즉시 해제
                all_latents_list[i] = None
                all_embeddings_list[i] = None

                # 5개 청크마다 GC 실행
                if (i + 1) % 5 == 0:
                    gc.collect()

            print(f"  Consolidation complete (offset={offset})")
            sys.stdout.flush()

            # 리스트 자체 해제
            del all_latents_list
            del all_embeddings_list
            gc.collect()

        total_time = time.time() - start_time
        latent_mem = self.all_latents.nbytes / (1024**3) if self.all_latents is not None else 0
        embed_mem = self.all_embeddings.nbytes / (1024**3) if self.all_embeddings is not None else 0

        print(f"\n[RAMPreload] === Preload Complete ===")
        print(f"  Total samples: {self.total_samples:,}")
        print(f"  Latents shape: {self.all_latents.shape if self.all_latents is not None else 'None'}")
        print(f"  Embeddings shape: {self.all_embeddings.shape if self.all_embeddings is not None else 'None'}")
        print(f"  Memory usage: {latent_mem:.2f}GB (latents) + {embed_mem:.2f}GB (embeddings) = {latent_mem + embed_mem:.2f}GB")
        print(f"  Total time: {total_time:.1f}s ({total_time/len(self.pt_files):.1f}s/file avg)")
        sys.stdout.flush()

        # 캐시 디렉토리 최종 정리 (혹시 남은 파일 삭제)
        self._cleanup_cache_directory()

    def _download_pt_file(self, gcs_path: str) -> str:
        """단일 PT 파일 다운로드"""
        filename = os.path.basename(gcs_path)
        local_path = os.path.join(self.gcs_handler.cache_dir, filename)
        return self.gcs_handler.download_file(gcs_path, local_path)

    def _cleanup_cache_directory(self):
        """캐시 디렉토리의 모든 PT 파일 및 임시 파일 삭제"""
        import sys
        import shutil

        cache_dir = self.gcs_handler.cache_dir
        if not os.path.exists(cache_dir):
            return

        deleted_count = 0
        deleted_size = 0

        try:
            for filename in os.listdir(cache_dir):
                filepath = os.path.join(cache_dir, filename)
                if os.path.isfile(filepath):
                    try:
                        file_size = os.path.getsize(filepath)
                        os.remove(filepath)
                        deleted_count += 1
                        deleted_size += file_size
                    except Exception as e:
                        print(f"  [RAMPreload] Warning: Could not delete {filename}: {e}")

            if deleted_count > 0:
                print(f"[RAMPreload] Cache cleanup: deleted {deleted_count} files ({deleted_size / (1024**3):.2f} GB)")
                sys.stdout.flush()
            else:
                print(f"[RAMPreload] Cache directory already clean")
                sys.stdout.flush()

        except Exception as e:
            print(f"[RAMPreload] Cache cleanup error: {e}")
            sys.stdout.flush()

    def calculate_steps_per_epoch(self, global_batch_size: int) -> int:
        """에포크당 스텝 수 계산"""
        if self.total_samples == 0:
            raise ValueError("No samples loaded!")

        # 분산 학습: 각 worker의 total_samples는 이미 샤딩됨
        # global 관점에서 전체 샘플 수 계산
        global_samples = self.total_samples * self.num_processes
        steps = global_samples // global_batch_size

        print(f"[RAMPreload] Steps calculation:")
        print(f"  Local samples: {self.total_samples:,}")
        print(f"  Global samples: {global_samples:,}")
        print(f"  Global batch size: {global_batch_size}")
        print(f"  Steps per epoch: {steps:,}")

        return steps

    def get_epoch_loader(self, epoch: int, steps_per_epoch: int,
                        num_workers: int = 8) -> 'RAMEpochDataLoader':
        """에포크용 데이터로더 생성"""
        return RAMEpochDataLoader(
            session=self,
            epoch=epoch,
            steps_per_epoch=steps_per_epoch,
            batch_size=self.batch_size,
            num_workers=num_workers
        )

    def shutdown(self):
        """세션 종료 및 메모리 해제"""
        self.all_latents = None
        self.all_embeddings = None
        self.ram_data.clear()
        gc.collect()
        print("[RAMPreload] Session shutdown, memory released")


class RAMEpochDataLoader:
    """RAM 프리로드 데이터에서 배치 샘플링

    네트워크 I/O 없음 - 순수 메모리 연산만
    """

    def __init__(self, session: RAMPreloadSession, epoch: int,
                 steps_per_epoch: int, batch_size: int,
                 num_workers: int = 8):
        self.session = session
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 배치 큐 (prefetch)
        self.batch_queue: queue.Queue = queue.Queue(maxsize=num_workers * 2)
        self.stop_event = threading.Event()

        # 샘플링 스레드풀
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.sampler_thread: Optional[threading.Thread] = None
        self.started = False

    def _start_sampling(self):
        """배치 샘플링 스레드 시작"""
        if self.started:
            return
        self.started = True

        self.sampler_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self.sampler_thread.start()

    def _sampling_loop(self):
        """배치 샘플링 루프 (순수 메모리 연산)"""
        import sys

        rng = np.random.RandomState(self.epoch * 10000)
        num_samples = self.session.total_samples

        pending_futures: Dict[int, Future] = {}
        next_step = 0
        completed_batches: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        print(f"[RAMLoader] Starting sampling loop (epoch {self.epoch}, {self.steps_per_epoch} steps)")
        sys.stdout.flush()

        while next_step < self.steps_per_epoch and not self.stop_event.is_set():
            # 새 작업 제출
            while len(pending_futures) < self.num_workers * 2:
                step_to_submit = next_step + len(pending_futures) + len(completed_batches)
                if step_to_submit >= self.steps_per_epoch:
                    break
                if step_to_submit not in pending_futures and step_to_submit not in completed_batches:
                    seed = self.epoch * 100000 + step_to_submit
                    future = self.executor.submit(
                        self._sample_batch, seed
                    )
                    pending_futures[step_to_submit] = future

            # 완료된 작업 수집
            done_steps = []
            for step_idx, future in pending_futures.items():
                if future.done():
                    try:
                        batch = future.result()
                        completed_batches[step_idx] = batch
                        done_steps.append(step_idx)
                    except Exception as e:
                        print(f"[RAMLoader] Sampling error at step {step_idx}: {e}")
                        done_steps.append(step_idx)

            for step_idx in done_steps:
                del pending_futures[step_idx]

            # 순서대로 큐에 전달
            while next_step in completed_batches:
                batch = completed_batches.pop(next_step)
                try:
                    self.batch_queue.put(batch, timeout=1.0)
                    if next_step == 0:
                        print(f"[RAMLoader] ✓ First batch ready (shape: {batch[0].shape})")
                        sys.stdout.flush()
                    next_step += 1
                except queue.Full:
                    completed_batches[next_step] = batch
                    break

            time.sleep(0.001)  # 1ms (메모리 연산이라 빠름)

        # 종료 신호
        self.batch_queue.put(None)

    def _sample_batch(self, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """단일 배치 샘플링 (순수 NumPy 연산)"""
        rng = np.random.RandomState(seed)
        num_samples = self.session.total_samples

        # 랜덤 인덱스
        indices = rng.randint(0, num_samples, size=self.batch_size)

        # Latents: (N, 3, 4, 32, 32) -> random crop -> (B, 32, 32, 4) NHWC
        latents = self.session.all_latents[indices]  # (B, 3, 4, 32, 32)
        crop_indices = rng.randint(0, 3, size=self.batch_size)
        latents_cropped = latents[np.arange(self.batch_size), crop_indices]  # (B, 4, 32, 32)
        latents_nhwc = np.transpose(latents_cropped, (0, 2, 3, 1))  # (B, 32, 32, 4)

        # Embeddings
        embeddings = self.session.all_embeddings[indices]  # (B, embed_dim)

        return (latents_nhwc, embeddings)

    def get_batches(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """배치 이터레이터"""
        self._start_sampling()

        batch_count = 0
        while batch_count < self.steps_per_epoch:
            try:
                batch = self.batch_queue.get(timeout=60)  # 메모리 연산이라 60초면 충분

                if batch is None:
                    break

                latents, embeddings = batch
                yield jnp.array(latents), jnp.array(embeddings)
                batch_count += 1

            except queue.Empty:
                print(f"[RAMLoader] Queue timeout at step {batch_count}")
                break

    def stop(self):
        """로더 중지"""
        self.stop_event.set()
        self.executor.shutdown(wait=False)
        if self.sampler_thread:
            self.sampler_thread.join(timeout=2.0)