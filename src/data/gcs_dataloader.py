"""
GCS DataLoader with Async Embedding Pipeline for TPU Training

최적화 전략:
1. PT 파일에서 latent 로드
2. Parquet에서 caption 로드 (key로 매핑)
3. 비동기 임베딩 계산 (CPU에서 미리 N개 배치 준비)
4. Double buffering으로 TPU 학습과 데이터 로딩 오버랩
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
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import jax.numpy as jnp
import torch
import pyarrow.parquet as pq
from google.cloud import storage


@dataclass
class ParquetCache:
    """Parquet 메타데이터 캐시 (key -> caption 매핑)"""
    key_to_caption: Dict[int, str]


class GCSFileHandler:
    """GCS 파일 다운로드 핸들러 (google-cloud-storage 사용)"""

    def __init__(self, gcs_bucket: str, cache_dir: Optional[str] = None):
        self.gcs_bucket_url = gcs_bucket.rstrip('/')
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="gcs_cache_")
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Parse bucket name and prefix from gs:// URL
        # e.g., gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/
        if gcs_bucket.startswith("gs://"):
            parts = gcs_bucket[5:].rstrip('/').split('/', 1)
            self.bucket_name = parts[0]
            self.prefix = parts[1] + '/' if len(parts) > 1 else ''
        else:
            self.bucket_name = gcs_bucket
            self.prefix = ''

        # Initialize GCS client
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

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
            'latents': tensor of shape (N, 4, 32, 32) - VAE latents
        }
        """
        data = torch.load(pt_path, map_location='cpu')

        result = {
            'keys': data['keys'].numpy() if isinstance(data['keys'], torch.Tensor) else np.array(data['keys']),
            'latents': data['latents'],  # Keep as torch tensor
        }

        return result


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

    PT 파일 + Parquet caption + 실시간 임베딩 계산
    """

    def __init__(self, batch_size: int, parquet_path: str,
                 embedding_provider: Any, gcs_bucket: str,
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

    def get_epoch_loader(self, epoch: int, steps_per_epoch: int) -> 'EpochDataLoader':
        """에포크용 데이터로더 생성"""
        return EpochDataLoader(
            session=self,
            epoch=epoch,
            steps_per_epoch=steps_per_epoch,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_ahead=self.prefetch_ahead
        )

    def shutdown(self):
        """세션 종료"""
        self.executor.shutdown(wait=True)
        gc.collect()


class EpochDataLoader:
    """에포크 단위 데이터로더 (비동기 임베딩 파이프라인 포함)"""

    def __init__(self, session: GCSDataLoaderSession, epoch: int,
                 steps_per_epoch: int, batch_size: int,
                 num_workers: int = 4, prefetch_ahead: int = 4):
        self.session = session
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_ahead = prefetch_ahead

        # 최종 배치 큐 (latents + embeddings)
        self.batch_queue = queue.Queue(maxsize=prefetch_ahead)
        self.stop_event = threading.Event()

        # 비동기 임베딩 파이프라인
        self.embedding_pipeline = AsyncEmbeddingPipeline(
            session.embedding_provider,
            num_prefetch=prefetch_ahead
        )

        # 데이터 준비 스레드
        self.data_thread = threading.Thread(target=self._prepare_batches, daemon=True)
        self.data_thread.start()

        # 임베딩 결합 스레드
        self.combine_thread = threading.Thread(target=self._combine_batches, daemon=True)
        self.combine_thread.start()

        # Latent 버퍼 (batch_idx -> latents)
        self.latent_buffer: Dict[int, np.ndarray] = {}
        self.buffer_lock = threading.Lock()

    def _prepare_batches(self):
        """배치 데이터 준비 (latent 로드 + caption 추출 + 임베딩 요청)"""
        rng = np.random.RandomState(self.epoch * 1000)

        pt_files = self.session.pt_files
        num_pt_files = len(pt_files)

        if num_pt_files == 0:
            print("No PT files found!")
            return

        current_pt_idx = 0
        current_data = None

        for step in range(self.steps_per_epoch):
            if self.stop_event.is_set():
                break

            try:
                # PT 파일 로드 (주기적으로 새 파일)
                if current_data is None or step % 50 == 0:
                    pt_path = pt_files[current_pt_idx % num_pt_files]
                    current_data = self.session._load_pt_data(pt_path)
                    current_pt_idx += 1

                # 배치 샘플링
                num_samples = len(current_data['keys'])
                indices = rng.randint(0, num_samples, size=self.batch_size)

                # Keys 추출
                keys = current_data['keys'][indices]

                # Latents: (B, 3, 4, 32, 32) -> 랜덤 crop 선택 -> (B, 4, 32, 32) -> (B, 32, 32, 4) NHWC
                latents = current_data['latents'][indices]  # (B, 3, 4, 32, 32)
                latents_np = latents.float().numpy()

                # 3개 crop 중 랜덤 선택 (각 샘플마다 다른 crop)
                batch_size = latents_np.shape[0]
                crop_indices = rng.randint(0, 3, size=batch_size)
                latents_selected = latents_np[np.arange(batch_size), crop_indices]  # (B, 4, 32, 32)

                # NCHW -> NHWC
                latents_nhwc = np.transpose(latents_selected, (0, 2, 3, 1))  # (B, 32, 32, 4)

                # Captions 가져오기
                captions = self.session.get_captions_for_keys(keys)

                # Latent 버퍼에 저장
                with self.buffer_lock:
                    self.latent_buffer[step] = latents_nhwc

                # 임베딩 계산 요청 (비동기)
                self.embedding_pipeline.submit(step, captions)

            except Exception as e:
                import traceback
                print(f"Data preparation error at step {step}: {e}")
                if step == 0:  # 첫 에러만 상세 출력
                    traceback.print_exc()
                    if current_data is not None:
                        print(f"  Debug - latents type: {type(current_data['latents'])}")
                        print(f"  Debug - latents shape: {current_data['latents'].shape}")
                        print(f"  Debug - indices shape: {indices.shape}")
                continue

        # 종료 신호
        self.embedding_pipeline.submit(-1, [])

    def _combine_batches(self):
        """Latent와 임베딩 결합"""
        completed = 0

        while completed < self.steps_per_epoch:
            if self.stop_event.is_set():
                break

            try:
                result = self.embedding_pipeline.get_result(timeout=60.0)
                if result is None:
                    continue

                batch_idx, embeddings = result
                if batch_idx < 0:  # 종료 신호
                    break

                # Latent 버퍼에서 가져오기
                with self.buffer_lock:
                    if batch_idx in self.latent_buffer:
                        latents = self.latent_buffer.pop(batch_idx)
                    else:
                        continue

                # 최종 배치 큐에 추가
                self.batch_queue.put((latents, embeddings), timeout=30)
                completed += 1

            except queue.Empty:
                print(f"Combine timeout at batch {completed}")
                break
            except Exception as e:
                print(f"Combine error: {e}")
                continue

        # 종료 신호
        self.batch_queue.put(None)

    def get_batches(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """배치 이터레이터"""
        batch_count = 0

        while batch_count < self.steps_per_epoch:
            try:
                batch = self.batch_queue.get(timeout=120)

                if batch is None:
                    break

                latents, embeddings = batch
                yield jnp.array(latents), jnp.array(embeddings)
                batch_count += 1

            except queue.Empty:
                print(f"Batch queue timeout at step {batch_count}")
                break

    def stop(self):
        """로더 중지"""
        self.stop_event.set()
        self.embedding_pipeline.stop()
        self.data_thread.join(timeout=5.0)
        self.combine_thread.join(timeout=5.0)