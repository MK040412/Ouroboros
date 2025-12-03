"""
GCS 기반 데이터로더 with 멀티프로세스 캐싱 (112 vCPU)
- GCS에서 PT 파일 자동 순회
- Parquet 메타데이터 캐싱
- 112 workers로 병렬 로딩 및 prefetch
"""

import os
import gc
import queue
import threading
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import pyarrow.parquet as pq
import jax
import jax.numpy as jnp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# GCS 파일 핸들러
# ============================================
class GCSFileHandler:
    """GCS 파일 관리"""
    
    def __init__(self, gcs_bucket: str = "gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/"):
        self.gcs_bucket = gcs_bucket
        self.latent_path = f"{gcs_bucket}latents-3crop/"
        self.metadata_path = gcs_bucket
        
        # gsutil 또는 gcloud storage 사용 여부 확인
        self._check_gcs_availability()
    
    def _check_gcs_availability(self):
        """GCS 접근 가능 여부 확인"""
        try:
            import subprocess
            result = subprocess.run(
                ["gcloud", "storage", "ls", self.gcs_bucket],
                capture_output=True, text=True, timeout=10
            )
            self.gcs_available = result.returncode == 0
            if self.gcs_available:
                logger.info("✓ GCS access verified")
            else:
                logger.warning(f"✗ GCS access failed: {result.stderr}")
        except Exception as e:
            self.gcs_available = False
            logger.warning(f"GCS check failed: {e}")
    
    def list_pt_files(self) -> List[str]:
        """GCS에서 PT 파일 목록 조회 (정렬됨)"""
        if not self.gcs_available:
            logger.warning("GCS not available, returning empty list")
            return []
        
        try:
            import subprocess
            result = subprocess.run(
                ["gcloud", "storage", "ls", f"{self.latent_path}*.pt"],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to list files: {result.stderr}")
                return []
            
            files = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            # 파일명으로 정렬 (000000-000009.pt, 000010-000019.pt, ...)
            files.sort()
            logger.info(f"Found {len(files)} PT files")
            return files
        except Exception as e:
            logger.error(f"Error listing PT files: {e}")
            return []
    
    def download_file(self, gcs_path: str, local_path: str) -> bool:
        """GCS 파일을 로컬에 다운로드"""
        if not self.gcs_available:
            logger.warning(f"GCS not available, cannot download {gcs_path}")
            return False
        
        try:
            import subprocess
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            result = subprocess.run(
                ["gcloud", "storage", "cp", gcs_path, local_path],
                capture_output=True, text=True, timeout=300  # 5분
            )
            
            if result.returncode == 0:
                logger.info(f"✓ Downloaded: {gcs_path}")
                return True
            else:
                logger.error(f"Download failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error downloading {gcs_path}: {e}")
            return False


# ============================================
# 파쿠엣 메타데이터 캐시
# ============================================
@dataclass
class ParquetCache:
    """Parquet 메타데이터 캐싱"""
    key_to_caption: Dict[int, str]
    key_to_url: Dict[int, str]
    key_to_caption_llava: Dict[int, str]
    all_keys: set
    
    @classmethod
    def load_from_parquet(cls, parquet_path: str) -> 'ParquetCache':
        """Parquet 파일에서 메타데이터 로드"""
        logger.info(f"Loading parquet metadata: {parquet_path}")
        
        try:
            # 필요한 컬럼만 로드
            table = pq.read_table(
                parquet_path,
                columns=['key', 'caption_llava', 'url']
            )
            
            # 메모리 효율을 위해 dict로 변환
            key_to_caption_llava = {}
            key_to_url = {}
            
            for batch in table.to_batches():
                keys = batch['key'].to_pylist()
                captions = batch['caption_llava'].to_pylist()
                urls = batch['url'].to_pylist()
                
                for k, c, u in zip(keys, captions, urls):
                    key_to_caption_llava[k] = c or ""
                    key_to_url[k] = u or ""
            
            logger.info(f"✓ Loaded {len(key_to_caption_llava)} metadata entries")
            
            return cls(
                key_to_caption=key_to_caption_llava,  # caption_llava를 기본으로 사용
                key_to_url=key_to_url,
                key_to_caption_llava=key_to_caption_llava,
                all_keys=set(key_to_caption_llava.keys())
            )
        except Exception as e:
            logger.error(f"Failed to load parquet: {e}")
            raise


# ============================================
# GCS 기반 데이터로더
# ============================================
class GCSCoyo11mDataLoader:
    """GCS의 PT 파일과 Parquet 메타데이터를 사용한 데이터로더
    
    PT 파일을 동적으로 로드/언로드하여 디스크 공간 최적화
    """
    
    def __init__(
        self,
        batch_size: int,
        parquet_cache: ParquetCache,
        embedding_provider=None,
        gcs_bucket: str = "gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/",
        cache_dir: Optional[str] = None,
        num_samples: Optional[int] = None,
        max_cache_files: int = 3  # 최대 3개 PT 파일까지만 메모리 유지
    ):
        self.batch_size = batch_size
        self.embedding_provider = embedding_provider
        self.parquet_cache = parquet_cache
        self.gcs_handler = GCSFileHandler(gcs_bucket)
        self.max_cache_files = max_cache_files
        
        # 로컬 캐시 디렉토리
        if cache_dir is None:
            cache_dir = tempfile.gettempdir()
        self.cache_dir = Path(cache_dir) / "gcs_pt_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_pt_file = None
        self.available_indices = []
        self.pt_keys = None
        self.latents_torch = None
        
        # 캐시 추적
        self.loaded_files = {}  # filename -> load_time
        self.num_samples = num_samples
        
        logger.info(f"Cache dir: {self.cache_dir} (max {max_cache_files} files at once)")
    
    def _cleanup_old_files(self):
        """오래된 PT 파일 삭제하여 디스크 공간 확보 (max_cache_files 유지)"""
        if len(self.loaded_files) >= self.max_cache_files:
            # 가장 오래 로드된 파일 삭제
            oldest_file = min(self.loaded_files.items(), key=lambda x: x[1])[0]
            old_path = self.cache_dir / oldest_file
            
            if old_path.exists():
                try:
                    file_size_mb = old_path.stat().st_size / (1024**2)
                    old_path.unlink()
                    del self.loaded_files[oldest_file]
                    logger.info(f"  🗑️ Cleaned cache: removed {oldest_file} ({file_size_mb:.1f}MB)")
                except Exception as e:
                    logger.warning(f"Failed to delete {oldest_file}: {e}")
    
    def load_pt_file(self, gcs_pt_path: str) -> bool:
        """GCS에서 PT 파일 로드 (자동 캐시 관리)"""
        try:
            # 파일명 추출
            pt_filename = gcs_pt_path.split("/")[-1]
            local_pt_path = self.cache_dir / pt_filename
            
            # 이미 로컬에 있으면 로드, 없으면 다운로드
            if not local_pt_path.exists():
                logger.info(f"Downloading PT file: {pt_filename}")
                if not self.gcs_handler.download_file(gcs_pt_path, str(local_pt_path)):
                    return False
            
            # 캐시 정리 (최대 N개 파일만 유지)
            self._cleanup_old_files()
            
            # PT 파일 로드
            logger.info(f"Loading PT file: {pt_filename}")
            pt_data = torch.load(str(local_pt_path), map_location="cpu")
            
            self.pt_keys = pt_data['keys'].numpy()
            self.latents_torch = pt_data['latents']
            self.current_pt_file = pt_filename
            
            # 로드 시간 기록
            import time
            self.loaded_files[pt_filename] = time.time()
            
            # 사용 가능한 샘플 찾기 (PT와 Parquet 모두에 있는 것)
            self._find_available_indices()
            
            file_size_mb = local_pt_path.stat().st_size / (1024**2)
            logger.info(f"✓ Loaded {pt_filename} ({file_size_mb:.1f}MB) with {len(self.available_indices)} samples")
            return True
        except Exception as e:
            logger.error(f"Error loading PT file {gcs_pt_path}: {e}")
            return False
    
    def _find_available_indices(self):
        """PT와 Parquet 모두에 있는 샘플 인덱스 찾기"""
        self.available_indices = []
        
        limit = self.num_samples if self.num_samples else len(self.pt_keys)
        for idx, key in enumerate(self.pt_keys[:limit]):
            if key in self.parquet_cache.all_keys:
                self.available_indices.append(idx)
    
    def get_batch(self, batch_idx: int, rng_key) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """배치 데이터 반환 (랜덤 샘플링)"""
        if not self.available_indices:
            raise ValueError("No available samples in current PT file")
        
        # 랜덤 인덱스 선택
        indices = jax.random.randint(rng_key, (self.batch_size,), 0, len(self.available_indices))
        indices_np = np.array(indices)
        selected_indices = [self.available_indices[i] for i in indices_np]
        
        # Latent 추출: (B, 4, 32, 32) NCHW → (B, 32, 32, 4) NHWC
        latents_subset = self.latents_torch[selected_indices]  # (B, 4, 32, 32)
        latents_np = latents_subset.float().numpy().astype(np.float32)
        batch_latents = jnp.array(np.transpose(latents_np, (0, 2, 3, 1)))  # (B, 32, 32, 4)
        
        # Caption 추출
        batch_captions = []
        for idx in selected_indices:
            key = int(self.pt_keys[idx])
            caption = self.parquet_cache.key_to_caption.get(key, "")
            batch_captions.append(caption)
        
        # 임베딩 계산
        batch_embeddings = self.embedding_provider.batch_encode(
            batch_captions, batch_size=512, normalize=True
        )
        
        return batch_latents, batch_embeddings


# ============================================
# 병렬 캐싱 매니저 (112 workers)
# ============================================
class ParallelCacheManager:
    """멀티프로세스 병렬 캐싱 매니저"""
    
    def __init__(self, num_workers: int = 112, cache_dir: Optional[str] = None):
        self.num_workers = num_workers
        self.cache_dir = Path(cache_dir or tempfile.gettempdir()) / "gcs_pt_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.download_tasks = {}
        self.cached_files = set()
        
        logger.info(f"Initialized ParallelCacheManager with {num_workers} workers")
    
    def prefetch_pt_files(self, pt_files: List[str], gcs_handler: GCSFileHandler):
        """PT 파일 목록을 병렬로 미리 다운로드"""
        logger.info(f"Starting parallel prefetch of {len(pt_files)} PT files")
        
        for gcs_path in pt_files:
            filename = gcs_path.split("/")[-1]
            local_path = self.cache_dir / filename
            
            # 이미 캐시되어 있으면 스킵
            if local_path.exists():
                self.cached_files.add(filename)
                continue
            
            # 다운로드 태스크 제출
            future = self.executor.submit(
                gcs_handler.download_file, gcs_path, str(local_path)
            )
            self.download_tasks[filename] = future
        
        logger.info(f"Submitted {len(self.download_tasks)} download tasks")
    
    def wait_for_file(self, filename: str, timeout: int = 600) -> bool:
        """특정 파일 다운로드 완료 대기 (최대 timeout초)"""
        if filename in self.cached_files:
            return True
        
        if filename not in self.download_tasks:
            logger.warning(f"File {filename} not in download queue")
            return False
        
        try:
            future = self.download_tasks[filename]
            result = future.result(timeout=timeout)
            if result:
                self.cached_files.add(filename)
            return result
        except Exception as e:
            logger.error(f"Error waiting for {filename}: {e}")
            return False
    
    def shutdown(self):
        """캐시 매니저 종료"""
        self.executor.shutdown(wait=True)
        logger.info("ParallelCacheManager shutdown complete")


# ============================================
# GCS Prefetch Data Loader (112 workers)
# ============================================
class GCSPrefetchDataLoader:
    """GCS 기반 Prefetch 데이터로딩 파이프라인"""
    
    def __init__(
        self,
        data_loader: GCSCoyo11mDataLoader,
        pt_files: List[str],
        steps_per_epoch: int,
        num_workers: int = 112,
        cache_manager: Optional[ParallelCacheManager] = None
    ):
        self.data_loader = data_loader
        self.pt_files = pt_files
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers
        self.cache_manager = cache_manager or ParallelCacheManager(num_workers)
        
        self.prefetch_queue = queue.Queue(maxsize=num_workers * 2)
        self.stop_event = threading.Event()
        
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.worker_threads = []
        
        # 백그라운드에서 PT 파일 prefetch 시작
        self._start_prefetch()
    
    def _start_prefetch(self):
        """Prefetch worker 시작"""
        for worker_id in range(self.num_workers):
            thread = threading.Thread(
                target=self._prefetch_worker,
                args=(worker_id,),
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
    
    def _prefetch_worker(self, worker_id: int):
        """각 worker가 담당 배치를 미리 로드"""
        rng_key = jax.random.PRNGKey(worker_id)
        current_pt_idx = 0
        current_batch_in_pt = 0
        
        while current_pt_idx < len(self.pt_files):
            if self.stop_event.is_set():
                break
            
            try:
                # PT 파일 로드
                pt_file = self.pt_files[current_pt_idx]
                pt_filename = pt_file.split("/")[-1]
                
                # 파일 다운로드 완료 대기
                if not self.cache_manager.wait_for_file(pt_filename):
                    logger.warning(f"Failed to load {pt_filename}, skipping")
                    current_pt_idx += 1
                    current_batch_in_pt = 0
                    continue
                
                # PT 파일 로드
                if current_batch_in_pt == 0:
                    if not self.data_loader.load_pt_file(pt_file):
                        current_pt_idx += 1
                        continue
                
                # 배치 생성
                if current_batch_in_pt < self.steps_per_epoch:
                    rng_key, subkey = jax.random.split(rng_key)
                    batch = self.data_loader.get_batch(current_batch_in_pt, subkey)
                    
                    global_batch_idx = current_pt_idx * self.steps_per_epoch + current_batch_in_pt
                    self.prefetch_queue.put((global_batch_idx, batch), timeout=30)
                    
                    current_batch_in_pt += self.num_workers
                else:
                    current_pt_idx += 1
                    current_batch_in_pt = 0
            
            except Exception as e:
                logger.error(f"Prefetch error (worker {worker_id}): {e}")
                break
        
        self.prefetch_queue.put(None)
    
    def get_batches(self):
        """Prefetch된 배치 반환"""
        batches_dict = {}
        next_batch_idx = 0
        none_count = 0
        
        total_batches = len(self.pt_files) * self.steps_per_epoch
        
        while next_batch_idx < total_batches:
            try:
                item = self.prefetch_queue.get(timeout=60)
                
                if item is None:
                    none_count += 1
                    if none_count >= self.num_workers:
                        break
                    continue
                
                batch_idx, batch = item
                batches_dict[batch_idx] = batch
                
                # 순차적으로 배치 반환
                while next_batch_idx in batches_dict:
                    yield batches_dict.pop(next_batch_idx)
                    next_batch_idx += 1
            
            except queue.Empty:
                logger.warning(f"Prefetch timeout at batch {next_batch_idx}")
                break
    
    def stop(self):
        """Prefetch 중지"""
        self.stop_event.set()
        self.executor.shutdown(wait=True)
        for thread in self.worker_threads:
            thread.join(timeout=5)
        self.cache_manager.shutdown()


# ============================================
# 세션 기반 데이터로더 (epoch마다 PT 파일 순회)
# ============================================
class GCSDataLoaderSession:
    """GCS 데이터를 이용한 세션 기반 로더
    
    디스크 공간 최적화를 위해 자동으로 오래된 PT 파일 삭제
    """
    
    def __init__(
        self,
        batch_size: int,
        parquet_path: str,
        embedding_provider=None,
        gcs_bucket: str = "gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/",
        cache_dir: Optional[str] = None,
        num_workers: int = 112,
        prefetch_ahead: int = 2,  # 몇 개의 PT 파일을 미리 다운로드할지
        max_cache_files: int = 3  # 최대 몇 개의 PT 파일을 동시에 보관할지 (디스크 절약)
    ):
        self.batch_size = batch_size
        self.embedding_provider = embedding_provider
        self.gcs_bucket = gcs_bucket
        self.cache_dir = cache_dir
        self.num_workers = num_workers
        self.prefetch_ahead = prefetch_ahead
        self.max_cache_files = max_cache_files
        
        # Parquet 메타데이터 캐싱
        logger.info("Loading parquet metadata cache...")
        self.parquet_cache = ParquetCache.load_from_parquet(parquet_path)
        
        # GCS 핸들러 및 캐시 매니저
        self.gcs_handler = GCSFileHandler(gcs_bucket)
        self.cache_manager = ParallelCacheManager(num_workers, cache_dir)
        
        # PT 파일 목록
        self.pt_files = self.gcs_handler.list_pt_files()
        if not self.pt_files:
            raise ValueError("No PT files found in GCS bucket")
        
        logger.info(f"✓ Session initialized")
        logger.info(f"  Total PT files: {len(self.pt_files)}")
        logger.info(f"  Max cache files: {max_cache_files} (disk space optimized)")
        logger.info(f"  Expected disk usage: ~{max_cache_files * 100}MB")
        
        # 첫 배치의 PT 파일들을 미리 다운로드
        self._prefetch_initial_files()
    
    def _prefetch_initial_files(self):
        """처음 prefetch_ahead개의 PT 파일 미리 다운로드"""
        files_to_prefetch = self.pt_files[:min(self.prefetch_ahead, len(self.pt_files))]
        self.cache_manager.prefetch_pt_files(files_to_prefetch, self.gcs_handler)
    
    def get_epoch_loader(self, epoch: int, steps_per_epoch: int) -> GCSPrefetchDataLoader:
        """에포크용 prefetch 로더 생성"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Creating epoch loader for epoch {epoch}")
        logger.info(f"{'='*70}")
        
        # 현재 에포크에 필요한 PT 파일들 미리 다운로드
        next_prefetch_start = min(self.prefetch_ahead, len(self.pt_files))
        next_prefetch_end = min(next_prefetch_start + self.prefetch_ahead, len(self.pt_files))
        
        if next_prefetch_start < len(self.pt_files):
            files_to_prefetch = self.pt_files[next_prefetch_start:next_prefetch_end]
            self.cache_manager.prefetch_pt_files(files_to_prefetch, self.gcs_handler)
        
        # 데이터로더 생성 (캐시 파일 개수 제한)
        data_loader = GCSCoyo11mDataLoader(
            batch_size=self.batch_size,
            parquet_cache=self.parquet_cache,
            embedding_provider=self.embedding_provider,
            gcs_bucket=self.gcs_bucket,
            cache_dir=self.cache_dir,
            max_cache_files=self.max_cache_files  # 디스크 공간 절약
        )
        
        # Prefetch 로더 생성
        return GCSPrefetchDataLoader(
            data_loader=data_loader,
            pt_files=self.pt_files,
            steps_per_epoch=steps_per_epoch,
            num_workers=self.num_workers,
            cache_manager=self.cache_manager
        )
    
    def shutdown(self):
        """세션 종료"""
        self.cache_manager.shutdown()
        logger.info("GCS DataLoader Session shutdown complete")
