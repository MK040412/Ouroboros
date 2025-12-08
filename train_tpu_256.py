"""
TPU v5e 32 Pod에서 256² 이미지로 XUT-Small 학습
- 8 workers (hosts), 각 worker에 4개 TPU 칩
- 총 32 TPU 코어
Batch Size: 2048 (한 번에 처리)
Dataset: KBlueLeaf/coyo11m-256px-ccrop-latent
"""

import os
import sys
import gc
import signal
import argparse
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import flax.linen as nn
from flax import nnx
import optax
import numpy as np
import torch
import pyarrow.parquet as pq
from dataclasses import dataclass
from typing import Tuple, Optional, List
import time
from pathlib import Path
import glob
import wandb
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
import tempfile
from datetime import datetime
from google.cloud import storage
from google.oauth2 import service_account
import pickle
import io

# GCS 서비스 계정 키 경로 (TPU VM 스코프 제한 우회용)
GCS_SA_KEY_PATH = Path.home() / "gcs-sa-key.json"


def get_gcs_client():
    """GCS 클라이언트 생성 (서비스 계정 키 사용)"""
    if GCS_SA_KEY_PATH.exists():
        credentials = service_account.Credentials.from_service_account_file(
            str(GCS_SA_KEY_PATH),
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return storage.Client(credentials=credentials)
    else:
        # 키 파일 없으면 기본 인증 사용
        return storage.Client()

from src.xut.xut_small import create_xut_small
from src.embeddings import get_embedding_provider
from src.data.gcs_dataloader import (
    GCSDataLoaderSession,
    GCSFileHandler,
    ParquetCache,
    RAMPreloadSession,  # RAM 프리로드 모드
)


# ============================================
# Configuration
# ============================================
@dataclass
class TrainingConfig256:
    """256² 스테이지 학습 설정"""
    # 배치 및 데이터
    global_batch_size: int = 1024      # OOM 방지: 2048 → 1024
    num_devices: int = 32              # TPU v5e pod size (또는 112 for TPU v5e 256)
    batch_size_per_device: int = 64    # OOM 방지: 128 → 64

    # dtype (TPU bfloat16 최적화)
    use_bfloat16: bool = True          # 메모리 절반 + TPU 최적화
    
    # 학습
    num_epochs: int = 20
    steps_per_epoch: Optional[int] = 10000  # 고정값 (10000 × 1024 = 10.24M images/epoch)
    learning_rate: float = 0.5          # muP base learning rate
    mup_base_dim: int = 1                # muP base dimension for scaling
    warmup_steps: int = 1000
    
    # 모델 (XUT-Small)
    model_dim: int = 896
    context_dim: int = 640              # Gemma-3 270M dimension (precomputed)
    mlp_dim: int = 3072
    heads: int = 14
    depth: int = 4
    enc_blocks: int = 1
    dec_blocks: int = 2
    
    # Text embedding
    embedding_model: str = "gemma-3-270m"  # 640d, precomputed
    
    # TREAD (Timestep-Random Encoder Architecture Design)
    tread_selection_rate: float = 0.5  # 기존 연구 설정값
    
    # GCS 설정 (precomputed embeddings 포함)
    gcs_bucket: str = "gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/latents-3crop-gemma-3-270m/"
    parquet_file: str = "gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/coyo11m-meta.parquet"  # 메타데이터 파일
    cache_dir: str = None  # 자동으로 /tmp 사용
    num_data_workers: int = 56  # 배치 샘플링 병렬 워커 (112 vCPU의 절반)
    prefetch_ahead: int = 10  # PT 파일 프리페치 개수 (네트워크 최대 활용)
    max_cache_files: int = 12  # 최대 캐시 PT 파일 (47GB 디스크, 3GB/파일, 11GB 여유)
    num_download_workers: int = 10  # GCS 다운로드 병렬 워커 (prefetch_ahead와 동일)
    num_load_workers: int = 2  # PT 파일 로딩 워커 (CPU 바운드, 2개면 충분)

    # RAM 프리로드 모드 (네트워크 병목 제거)
    use_ram_preload: bool = True  # True: 시작 시 모든 PT 파일을 RAM에 로드 (권장)
    ram_preload_download_workers: int = 8  # 병렬 다운로드 워커 수
    
    # TPU 설정
    use_pjit: bool = True
    use_gradient_checkpointing: bool = True

    # Wandb
    wandb_project: str = "xut-small-256"
    wandb_entity: str = None  # set to username

    # VAE scaling (SDXL-VAE standard)
    # Raw VAE latents need to be scaled for stable training
    # See: https://huggingface.co/stabilityai/sdxl-vae/commit/e6e6fee
    vae_scaling_factor: float = 0.13025

    # GCS Checkpoint 설정
    checkpoint_gcs_bucket: str = "rdy-tpu-data-2025"
    checkpoint_gcs_prefix: str = "checkpoints/xut-small-256"  # gs://bucket/prefix/run_YYYYMMDD_HHMMSS/
    checkpoint_keep_last_n: int = 3  # 최근 N개 체크포인트만 유지


# ============================================
# Checkpoint Management (Resume 지원)
# ============================================
def find_latest_checkpoint(config: TrainingConfig256) -> Optional[dict]:
    """GCS에서 가장 최근 체크포인트 찾기

    Returns:
        체크포인트 데이터 dict 또는 None
    """
    if jax.process_index() != 0:
        # Process 0만 탐색, 결과는 broadcast
        return None

    try:
        client = get_gcs_client()
        bucket = client.bucket(config.checkpoint_gcs_bucket)
        prefix = config.checkpoint_gcs_prefix + "/"

        print(f"\n[Checkpoint] Searching for checkpoints in gs://{config.checkpoint_gcs_bucket}/{prefix}")
        sys.stdout.flush()

        # 모든 체크포인트 파일 나열
        blobs = list(bucket.list_blobs(prefix=prefix))
        ckpt_blobs = [b for b in blobs if b.name.endswith('.ckpt')]

        if not ckpt_blobs:
            print(f"  No checkpoints found")
            return None

        # 가장 최근 파일 찾기 (updated 시간 기준)
        latest_blob = max(ckpt_blobs, key=lambda b: b.updated)
        print(f"  Found {len(ckpt_blobs)} checkpoints")
        print(f"  Latest: {latest_blob.name}")
        print(f"  Updated: {latest_blob.updated}")
        sys.stdout.flush()

        # 다운로드 및 로드
        ckpt_bytes = latest_blob.download_as_bytes()
        ckpt_data = pickle.loads(ckpt_bytes)

        print(f"  ✓ Loaded checkpoint: epoch={ckpt_data['epoch']}, step={ckpt_data['step']}, loss={ckpt_data['loss']:.6f}")
        sys.stdout.flush()

        return ckpt_data

    except Exception as e:
        print(f"  ✗ Failed to find/load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return None


def restore_from_checkpoint(model, optimizer, ckpt_data: dict, config: TrainingConfig256):
    """체크포인트에서 모델과 옵티마이저 상태 복원

    Returns:
        (start_epoch, start_step, run_id)
    """
    print(f"\n[Restore] Restoring from checkpoint...")
    sys.stdout.flush()

    try:
        # numpy -> JAX array 변환
        def to_jax(x):
            if isinstance(x, np.ndarray):
                # bfloat16 변환 (config에 따라)
                if config.use_bfloat16 and x.dtype == np.float32:
                    return jnp.array(x, dtype=jnp.bfloat16)
                return jnp.array(x)
            return x

        model_state_jax = jax.tree_util.tree_map(to_jax, ckpt_data['model_state'])
        optimizer_state_jax = jax.tree_util.tree_map(to_jax, ckpt_data['optimizer_state'])

        # 상태 복원
        nnx.update(model, model_state_jax)
        nnx.update(optimizer, optimizer_state_jax)

        start_epoch = ckpt_data['epoch']
        start_step = ckpt_data['step']
        run_id = ckpt_data.get('run_id', None)

        print(f"  ✓ Model and optimizer restored")
        print(f"  Resuming from epoch={start_epoch}, step={start_step}")
        if run_id:
            print(f"  Run ID: {run_id}")
        sys.stdout.flush()

        return start_epoch, start_step, run_id

    except Exception as e:
        print(f"  ✗ Failed to restore: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise


class GracefulKiller:
    """SIGTERM/SIGINT 핸들러 (Preemption 대비 긴급 체크포인트)"""

    def __init__(self):
        self.kill_now = False
        self.trainer = None
        self.current_epoch = 0
        self.current_step = 0
        self.current_loss = 0.0

        # 시그널 등록
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"

        # Worker 0만 메시지 출력
        if jax.process_index() == 0:
            print(f"\n{'='*60}")
            print(f"[SIGNAL] Received {sig_name} - Preemption detected!")
            print(f"{'='*60}")
            print(f"  Current: epoch={self.current_epoch}, step={self.current_step}")
            print(f"  Last checkpoint: epoch {self.current_epoch} start")
            print(f"  Resume with: ./run_tpu_distributed.sh")
            print(f"{'='*60}")
            sys.stdout.flush()

        self.kill_now = True

    def register_trainer(self, trainer, epoch, step, loss):
        """현재 학습 상태 등록 (주기적 호출)"""
        self.trainer = trainer
        self.current_epoch = epoch
        self.current_step = step
        self.current_loss = loss


# Global killer instance
_graceful_killer = GracefulKiller()


# ============================================
# Rectified Flow Schedule with Logit-Normal Sampling
# ============================================
class RectifiedFlowSchedule:
    """Rectified Flow 스케줄 (Logit-Normal timestep sampling)

    Linear interpolation: x_t = (1 - t) * x_0 + t * x_1
    여기서 x_1 = noise (standard Gaussian)

    Logit-Normal sampling (SD3 논문):
    - t = sigmoid(normal(mean, std))
    - 중간 timestep에 더 많은 샘플링 집중
    - mean=0, std=1이 기본값

    모델은 velocity v = x_1 - x_0 를 예측
    """

    def __init__(self, logit_mean: float = 0.0, logit_std: float = 1.0):
        """
        Args:
            logit_mean: logit-normal 분포의 mean (기본 0.0)
            logit_std: logit-normal 분포의 std (기본 1.0)
        """
        self.logit_mean = logit_mean
        self.logit_std = logit_std

    def sample_timesteps(self, key: jnp.ndarray, batch_size: int) -> jnp.ndarray:
        """Logit-normal 분포에서 timestep 샘플링

        Args:
            key: JAX random key
            batch_size: 배치 크기
        Returns:
            t: (B,) - 시간 [0, 1] 범위 (logit-normal 분포)
        """
        # Sample from normal distribution
        u = jax.random.normal(key, (batch_size,))
        # Apply logit-normal transform: t = sigmoid(mean + std * u)
        t = jax.nn.sigmoid(self.logit_mean + self.logit_std * u)
        return t

    def forward(self, x_0: jnp.ndarray, x_1: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Rectified Flow forward: x_t = (1 - t) * x_0 + t * x_1

        Args:
            x_0: (B, H, W, C) - 원본 VAE latent (clean)
            x_1: (B, H, W, C) - 노이즈 (target, standard Gaussian)
            t: (B,) - 시간 [0, 1] 범위
        Returns:
            x_t: (B, H, W, C) - interpolated sample
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, 1, 1)
        t = t[:, None, None, None]

        # dtype 맞추기
        t = t.astype(x_0.dtype)

        x_t = (1.0 - t) * x_0 + t * x_1
        return x_t

    def get_velocity(self, x_0: jnp.ndarray, x_1: jnp.ndarray) -> jnp.ndarray:
        """Velocity target: v = x_1 - x_0"""
        return x_1 - x_0


# ============================================
# Sharding Strategy (TPU Pod 32)
# ============================================
@dataclass
class ShardingRules:
    """TPU Pod 32 분산 전략

    TPU v5e-32 구성:
    - 8 workers (hosts), 각 4개 TPU 칩
    - 총 32 TPU 코어
    - Data parallelism across all devices
    """
    data_axis: str = "data"      # 데이터 병렬화 축

    def get_mesh(self):
        """TPU Pod 메시 생성 - 전체 device에 대해 data parallelism"""
        # jax.devices()는 분산 모드에서 모든 프로세스의 device를 반환
        all_devices = jax.devices()
        num_devices = len(all_devices)
        print(f"  Creating mesh with {num_devices} devices")

        # 1D mesh for pure data parallelism (가장 안정적)
        devices = mesh_utils.create_device_mesh((num_devices,))
        return Mesh(devices, (self.data_axis,))

    def named_sharding(self, partition_spec, mesh):
        """PartitionSpec을 NamedSharding으로 변환"""
        return NamedSharding(mesh, partition_spec)


def get_sharding_rules():
    """Sharding 규칙 반환"""
    return ShardingRules()


# ============================================
# Large Scale Data Loader with PT Files
# ============================================
class Coyo11mDataLoader:
    """PT 파일과 Parquet 메타데이터를 사용한 데이터로더"""
    
    def __init__(self, batch_size: int, pt_file: str, parquet_file: str, embedding_provider=None, num_samples: int = None):
        self.batch_size = batch_size
        self.embedding_provider = embedding_provider
        self.pt_file = pt_file
        self.parquet_file = parquet_file
        
        # PT 파일에서 latent 로드
        print(f"Loading latents from {pt_file}...")
        pt_data = torch.load(pt_file, map_location="cpu")
        self.pt_keys = pt_data['keys'].numpy()
        self.latents_torch = pt_data['latents']
        
        # Parquet에서 키 확인
        print(f"Loading metadata from {parquet_file}...")
        table = pq.read_table(parquet_file, columns=['key'])
        parquet_keys = set(table['key'].to_pylist())
        
        # 교집합 찾기 (PT와 Parquet 모두에 있는 샘플)
        self.available_indices = []
        self.captions = []
        
        limit = num_samples if num_samples else len(self.pt_keys)
        for idx, key in enumerate(self.pt_keys[:limit]):
            if key in parquet_keys:
                self.available_indices.append(idx)
        
        print(f"Found {len(self.available_indices)} samples with both latent and caption")
        
        if len(self.available_indices) == 0:
            raise ValueError("No matching samples found between PT and Parquet!")
        
        # Caption 로드 (메모리 효율 위해 필요한 것만)
        self._load_captions()
    
    def _load_captions(self):
        """필요한 샘플들의 caption만 로드 (배치 쿼리로 최적화)"""
        print("Loading captions...")
        available_keys = [int(self.pt_keys[idx]) for idx in self.available_indices]
        
        # 전체 Parquet 메타데이터 로드 (한 번만)
        full_table = pq.read_table(self.parquet_file, columns=['key', 'caption_llava'])
        key_to_caption = {row['key'].as_py(): row['caption_llava'].as_py() 
                          for row in full_table.to_batches()}
        
        for key in available_keys:
            caption = key_to_caption.get(key, "")
            self.captions.append(caption)
        
        print(f"✓ Loaded {len(self.captions)} captions")
    
    def get_batch(self, batch_idx: int, rng_key):
        """배치 데이터 반환 (랜덤 샘플링)"""
        # 랜덤 인덱스 선택
        indices = jax.random.randint(rng_key, (self.batch_size,), 0, len(self.available_indices))
        indices_np = np.array(indices)  # JAX array to numpy
        selected_indices = [self.available_indices[i] for i in indices_np]
        
        # Latent 추출: (B, 4, 32, 32) NCHW → (B, 32, 32, 4) NHWC
        latents_subset = self.latents_torch[selected_indices]  # (B, 4, 32, 32)
        latents_np = latents_subset.float().numpy().astype(np.float32)
        batch_latents = jnp.array(np.transpose(latents_np, (0, 2, 3, 1)))  # (B, 32, 32, 4)
        
        # Caption 추출
        batch_captions = [self.captions[i] for i in indices_np]
        
        # 임베딩 계산
        batch_embeddings = self.embedding_provider.batch_encode(
            batch_captions, batch_size=512, normalize=True
        )
        
        return batch_latents, batch_embeddings


# ============================================
# Prefetch Pipeline with Multi-Process (TPU 최적화)
# ============================================
class PrefetchDataLoader:
    """Multi-worker Prefetch 데이터로딩 파이프라인 (112 workers)"""
    
    def __init__(self, data_loader: Coyo11mDataLoader, 
                 steps_per_epoch: int, num_workers: int = 112):
        self.data_loader = data_loader
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers
        self.prefetch_queue = queue.Queue(maxsize=num_workers * 2)
        self.stop_event = threading.Event()
        
        # Multi-worker executor 시작
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.worker_threads = []
        
        # Worker 풀 시작
        for worker_id in range(num_workers):
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
        
        # 각 worker는 (worker_id, worker_id + num_workers, worker_id + 2*num_workers, ...)
        # 형태로 배치를 담당
        batch_idx = worker_id
        while batch_idx < self.steps_per_epoch:
            if self.stop_event.is_set():
                break
            try:
                rng_key, subkey = jax.random.split(rng_key)
                batch = self.data_loader.get_batch(batch_idx, subkey)
                self.prefetch_queue.put((batch_idx, batch), timeout=10)
            except Exception as e:
                print(f"Prefetch error at batch {batch_idx} (worker {worker_id}): {e}")
                break
            
            batch_idx += self.num_workers
        
        # 종료 신호
        self.prefetch_queue.put(None)
    
    def get_batches(self):
        """Prefetch된 배치 반환 (배치 인덱스 순서로 정렬)"""
        batches_dict = {}
        next_batch_idx = 0
        none_count = 0
        
        while next_batch_idx < self.steps_per_epoch:
            try:
                item = self.prefetch_queue.get(timeout=30)
                if item is None:
                    none_count += 1
                    if none_count >= self.num_workers:
                        # 모든 worker 종료
                        break
                    continue
                
                batch_idx, batch = item
                batches_dict[batch_idx] = batch
                
                # 순차적으로 배치 반환
                while next_batch_idx in batches_dict:
                    yield batches_dict.pop(next_batch_idx)
                    next_batch_idx += 1
            
            except queue.Empty:
                print(f"Prefetch timeout at batch {next_batch_idx}")
                break
    
    def stop(self):
        """Prefetch 중지"""
        self.stop_event.set()
        self.executor.shutdown(wait=True)
        for thread in self.worker_threads:
            thread.join(timeout=5)


# ============================================
# Training with pjit
# ============================================
def create_sharding_mesh(num_devices: int):
    """TPU 메시 생성"""
    # v5e 32 pod: 32개 TPU 칩 (8 workers × 4 chips)
    devices = mesh_utils.create_device_mesh((num_devices,))
    return Mesh(devices, axis_names=("batch",))


# ============================================
# JIT-compiled Train Step (Module-level for cache reuse)
# ============================================
@nnx.jit
def _train_step_jit(model, optimizer, x_t, t, velocity_target, text_emb):
    """JIT 컴파일된 학습 스텝 (Rectified Flow)

    Args:
        model: XUDiT 모델
        optimizer: nnx.Optimizer
        x_t: (B, H, W, C) - interpolated sample (NHWC)
        t: (B,) 또는 (B, 1) - 시간 [0, 1] 범위
        velocity_target: (B, H, W, C) - velocity target v = x_1 - x_0 (NHWC)
        text_emb: (B, 1, D) - text embeddings (CFG 드롭된 상태)
    """
    def loss_fn(model):
        # 모델 입력: NHWC, 출력: NCHW
        # 모델은 velocity를 예측
        # deterministic=False: TREAD 토큰 드롭아웃 활성화
        pred_v_nchw = model(x_t, t, ctx=text_emb, deterministic=False)
        # NCHW -> NHWC for loss computation
        pred_v = jnp.transpose(pred_v_nchw, (0, 2, 3, 1))
        # MSE loss: ||v_pred - v_target||^2
        return jnp.mean((pred_v - velocity_target) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


class TPUTrainer:
    """TPU 분산 학습기 (Sharding 적용)"""

    def __init__(self, model, optimizer, schedule, config: TrainingConfig256,
                 wandb_enabled: bool = False, run_id: str = None):
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.config = config
        self.wandb_enabled = wandb_enabled

        # Sharding 설정
        self.sharding_rules = get_sharding_rules()
        self.mesh = self.sharding_rules.get_mesh()

        print(f"\n[Sharding Setup]")
        print(f"  Mesh shape: {self.mesh.shape}")
        print(f"  Axes: {self.mesh.axis_names}")
        print(f"  Total devices: {self.mesh.shape['data']}")
        print(f"  Data parallelism: {self.mesh.shape['data']}-way")

        # Learning rate schedule
        self.lr_schedule = self._create_lr_schedule()

        # GCS Checkpoint 설정 (시간 기반 run ID)
        if run_id is None:
            run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.gcs_checkpoint_path = f"{config.checkpoint_gcs_prefix}/{run_id}"

        # GCS 클라이언트 초기화 (Process 0만 저장)
        self.gcs_client = None
        self.gcs_bucket = None
        if jax.process_index() == 0:
            try:
                self.gcs_client = get_gcs_client()
                self.gcs_bucket = self.gcs_client.bucket(config.checkpoint_gcs_bucket)
                print(f"\n[GCS Checkpoint Setup]")
                print(f"  Bucket: gs://{config.checkpoint_gcs_bucket}")
                print(f"  Path: {self.gcs_checkpoint_path}/")
                print(f"  Run ID: {run_id}")
            except Exception as e:
                print(f"  ⚠ GCS client init failed: {e}")
                print(f"  Checkpoints will be saved locally only")

        # 로컬 백업 디렉토리 (GCS 실패 시 사용)
        self.checkpoint_dir = Path("./checkpoints") / run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_lr_schedule(self):
        """Warmup + Cosine decay with muP scaling"""
        # muP scaled learning rate
        mup_lr = self.config.learning_rate * (self.config.mup_base_dim / self.config.model_dim)

        def lr_fn(step):
            if step < self.config.warmup_steps:
                return mup_lr * (step / self.config.warmup_steps)
            else:
                progress = (step - self.config.warmup_steps) / (
                    self.config.steps_per_epoch * self.config.num_epochs - self.config.warmup_steps
                )
                return mup_lr * 0.5 * (1 + jnp.cos(jnp.pi * progress))

        return lr_fn
    
    @staticmethod
    def loss_fn(model, x_t, timesteps, noise, text_emb):
        """노이즈 예측 손실 (epsilon/noise prediction)"""
        # 입력 형식 변환 (NCHW → NHWC)
        # x_t: (B, 4, 32, 32) NCHW → (B, 32, 32, 4) NHWC
        x_t_nhwc = jnp.transpose(x_t, (0, 2, 3, 1))
        
        # timesteps: (B,) 또는 (B, 1)
        if timesteps.ndim == 1:
            timesteps = timesteps[:, None]  # (B,) → (B, 1)
        
        # text_emb: (B, 640) → (B, 1, 640) for context
        ctx = text_emb[:, None, :]  # (B, 640) → (B, 1, 640)
        
        # 모델 호출: 입력은 NHWC, 출력은 (B, H, W, C) NHWC 형식
        # XUDiT는 NHWC 입력받고 (B, H, W, C) 출력
        pred_noise_nhwc = model(x_t_nhwc, timesteps, ctx=ctx, deterministic=False)
        
        # 출력 형식 확인: (B, 32, 32, 4) NHWC
        # NCHW로 변환: (B, 4, 32, 32)
        pred_noise = jnp.transpose(pred_noise_nhwc, (0, 3, 1, 2))
        
        # MSE loss: target은 noise (B, 4, 32, 32)
        loss = jnp.mean((pred_noise - noise) ** 2)
        return loss
    
    def train_step(self, x_t, t, velocity_target, text_emb, rng_key):
        """한 스텝 학습 (Rectified Flow, Sharded + JIT 컴파일)

        Args:
            x_t: (B, H, W, C) - interpolated sample
            t: (B,) - 시간 [0, 1] 범위
            velocity_target: (B, H, W, C) - velocity target v = x_1 - x_0
            text_emb: (B, D) - text embeddings
            rng_key: JAX random key
        """
        # CFG (Classifier-Free Guidance): text embedding 마스킹
        # tread_selection_rate 확률로 text embedding을 0으로 마스킹 → unconditional 학습
        # 참고: TREAD (token dropout)는 모델 내부에서 deterministic=False로 처리됨
        batch_size = t.shape[0]
        rng_key, subkey = jax.random.split(rng_key)
        cfg_mask = jax.random.uniform(subkey, (batch_size,)) < self.config.tread_selection_rate

        # text_emb: (batch, dim) → (batch, 1, dim) for sequence concatenation
        text_emb_3d = text_emb[:, None, :]

        # CFG 드롭: mask가 True인 샘플은 text embedding을 0으로
        text_emb_cond = jnp.where(cfg_mask[:, None, None], jnp.zeros_like(text_emb_3d), text_emb_3d)

        # 모듈 레벨 JIT 함수 호출
        # - t: 원본 timestep (마스킹 없음)
        # - text_emb_cond: CFG 드롭된 text embedding
        # - deterministic=False: TREAD 토큰 드롭아웃 활성화 (_train_step_jit 내부)
        loss = _train_step_jit(self.model, self.optimizer, x_t, t, velocity_target, text_emb_cond)
        return loss, rng_key
    
    def save_checkpoint(self, epoch: int, step: int, loss: float, is_step_checkpoint: bool = False):
        """체크포인트를 GCS에 저장 (Process 0만)

        저장 경로:
          - Epoch: gs://{bucket}/{prefix}/epoch_{epoch:03d}.ckpt
          - Step:  gs://{bucket}/{prefix}/step_{global_step:06d}.ckpt

        Args:
            epoch: 현재 에포크
            step: 현재 스텝 (에포크 내)
            loss: 현재 loss 값
            is_step_checkpoint: True면 step 체크포인트, False면 epoch 체크포인트
        """
        # Process 0만 저장 (분산 학습에서 중복 저장 방지)
        if jax.process_index() != 0:
            return

        if is_step_checkpoint:
            global_step = epoch * self.config.steps_per_epoch + step
            checkpoint_name = f"step_{global_step:06d}.ckpt"
        else:
            checkpoint_name = f"epoch_{epoch:03d}.ckpt"

        try:
            # 모델 상태 추출 (nnx.Module -> state dict)
            model_state = nnx.state(self.model)
            optimizer_state = nnx.state(self.optimizer)

            # JAX arrays -> numpy arrays (직렬화 가능하도록)
            def to_numpy(x):
                if hasattr(x, 'value'):
                    x = x.value
                if isinstance(x, jnp.ndarray):
                    # PRNGKey 타입은 특별 처리 필요
                    if hasattr(x, 'dtype') and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
                        return np.array(jax.random.key_data(x))
                    return np.array(x)
                return x

            model_state_np = jax.tree_util.tree_map(to_numpy, model_state)
            optimizer_state_np = jax.tree_util.tree_map(to_numpy, optimizer_state)

            checkpoint_data = {
                'epoch': epoch,
                'step': step,
                'loss': float(loss),
                'model_state': model_state_np,
                'optimizer_state': optimizer_state_np,
                'config': {
                    'model_dim': self.config.model_dim,
                    'context_dim': self.config.context_dim,
                    'depth': self.config.depth,
                    'learning_rate': self.config.learning_rate,
                },
                'run_id': self.run_id,
                'timestamp': datetime.now().isoformat(),
            }

            # Pickle로 직렬화
            buffer = io.BytesIO()
            pickle.dump(checkpoint_data, buffer)
            buffer.seek(0)
            checkpoint_bytes = buffer.getvalue()

            # GCS에 업로드
            if self.gcs_bucket is not None:
                gcs_path = f"{self.gcs_checkpoint_path}/{checkpoint_name}"
                blob = self.gcs_bucket.blob(gcs_path)
                blob.upload_from_string(checkpoint_bytes, content_type='application/octet-stream')
                print(f"  ✓ Checkpoint saved: gs://{self.config.checkpoint_gcs_bucket}/{gcs_path}")

                # 로테이션: step 체크포인트만 최근 N개 유지 (epoch 체크포인트는 보존)
                if is_step_checkpoint:
                    self._rotate_step_checkpoints()

                # Wandb에 GCS 경로 로깅
                if self.wandb_enabled:
                    wandb.log({
                        "checkpoint_gcs_path": f"gs://{self.config.checkpoint_gcs_bucket}/{gcs_path}",
                        "checkpoint_epoch": epoch,
                    })
            else:
                # GCS 사용 불가 시 로컬 저장
                local_path = self.checkpoint_dir / checkpoint_name
                with open(local_path, 'wb') as f:
                    f.write(checkpoint_bytes)
                print(f"  ✓ Checkpoint saved locally: {local_path}")

        except Exception as e:
            print(f"  ✗ Failed to save checkpoint: {e}")
            import traceback
            traceback.print_exc()

    def _rotate_step_checkpoints(self):
        """오래된 step 체크포인트 삭제 (최근 N개만 유지, epoch 체크포인트는 보존)"""
        if self.gcs_bucket is None:
            return

        try:
            # 체크포인트 목록 조회
            blobs = list(self.gcs_bucket.list_blobs(prefix=self.gcs_checkpoint_path + "/"))
            # step 체크포인트만 필터 (epoch 체크포인트 제외)
            step_ckpt_blobs = [b for b in blobs if b.name.endswith('.ckpt') and '/step_' in b.name]

            if len(step_ckpt_blobs) <= self.config.checkpoint_keep_last_n:
                return

            # 시간순 정렬 (오래된 것부터)
            step_ckpt_blobs.sort(key=lambda b: b.updated)

            # 삭제할 개수
            to_delete = len(step_ckpt_blobs) - self.config.checkpoint_keep_last_n
            for blob in step_ckpt_blobs[:to_delete]:
                blob.delete()
                print(f"  [Rotate] Deleted old step checkpoint: {blob.name}")

        except Exception as e:
            print(f"  [Rotate] Warning: Failed to rotate step checkpoints: {e}")
    
    def train_epoch(self, prefetch_loader: 'PrefetchDataLoader', epoch: int, start_step: int = 0):
        """에포크 학습 (Sharded prefetch 파이프라인)

        Args:
            prefetch_loader: 데이터 로더
            epoch: 현재 에포크 번호
            start_step: 시작 스텝 (resume 시 사용, 이전 스텝은 건너뜀)
        """
        losses = []
        rng_key = jax.random.PRNGKey(epoch)
        step = 0
        skipped_steps = 0
        last_log_time = time.time()  # 100 step 시간 측정용

        # Loss 로그 파일 (Worker 0만)
        loss_log_file = None
        if jax.process_index() == 0:
            loss_log_path = Path("/tmp/loss_log.csv")
            write_header = not loss_log_path.exists()
            loss_log_file = open(loss_log_path, "a")
            if write_header:
                loss_log_file.write("timestamp,epoch,step,global_step,loss,lr\n")

        # Sharding spec: 배치는 'data' 축, 나머지는 replica
        batch_sharding = self.sharding_rules.named_sharding(
            P(self.sharding_rules.data_axis, None, None, None),  # (B, 32, 32, 4)
            self.mesh
        )
        emb_sharding = self.sharding_rules.named_sharding(
            P(self.sharding_rules.data_axis, None),  # (B, D)
            self.mesh
        )

        # Local devices 정보
        local_devices = jax.local_devices()
        num_local_devices = len(local_devices)

        for batch_latents, batch_embeddings in prefetch_loader.get_batches():
            # Resume: start_step 이전 배치 건너뛰기
            if step < start_step:
                step += 1
                skipped_steps += 1
                if skipped_steps % 500 == 0:
                    print(f"  [Resume] Skipping steps... {skipped_steps}/{start_step}")
                    sys.stdout.flush()
                continue
            elif skipped_steps > 0 and step == start_step:
                print(f"  [Resume] Skipped {skipped_steps} steps, resuming from step {step}")
                sys.stdout.flush()

            local_batch_size = batch_latents.shape[0]
            per_device_batch = local_batch_size // num_local_devices

            # === 데이터 검증 (NaN/Inf 체크) ===
            if step < 10 or step % 500 == 0:
                # NumPy 배열일 때만 체크 (JAX array 변환 전)
                latent_np = np.asarray(batch_latents)
                emb_np = np.asarray(batch_embeddings)

                latent_nan = np.isnan(latent_np).sum()
                latent_inf = np.isinf(latent_np).sum()
                emb_nan = np.isnan(emb_np).sum()
                emb_inf = np.isinf(emb_np).sum()

                if latent_nan > 0 or latent_inf > 0 or emb_nan > 0 or emb_inf > 0:
                    print(f"[DATA WARNING] Step {step}: "
                          f"latent NaN={latent_nan}, Inf={latent_inf}, "
                          f"emb NaN={emb_nan}, Inf={emb_inf}")
                    print(f"  latent range: [{latent_np.min():.4f}, {latent_np.max():.4f}]")
                    print(f"  emb range: [{emb_np.min():.4f}, {emb_np.max():.4f}]")
                    sys.stdout.flush()
                elif step < 10:
                    print(f"[DATA OK] Step {step}: "
                          f"latent range=[{latent_np.min():.4f}, {latent_np.max():.4f}], "
                          f"emb range=[{emb_np.min():.4f}, {emb_np.max():.4f}]")
                    sys.stdout.flush()

            # Multi-host: 각 worker의 local 데이터를 global array로 변환
            # 각 local device에 데이터 조각 배치
            latent_arrays = [
                jax.device_put(
                    batch_latents[i*per_device_batch:(i+1)*per_device_batch],
                    d
                ) for i, d in enumerate(local_devices)
            ]
            emb_arrays = [
                jax.device_put(
                    batch_embeddings[i*per_device_batch:(i+1)*per_device_batch],
                    d
                ) for i, d in enumerate(local_devices)
            ]

            # Global array 생성 (모든 host의 데이터가 합쳐짐)
            # Shape: NHWC (batch, height, width, channels) - 모델 입력 형식
            global_batch_size = self.config.global_batch_size
            batch_latents = jax.make_array_from_single_device_arrays(
                (global_batch_size, 32, 32, 4),  # NHWC
                batch_sharding,
                latent_arrays
            )
            batch_embeddings = jax.make_array_from_single_device_arrays(
                (global_batch_size, batch_embeddings.shape[1]),
                emb_sharding,
                emb_arrays
            )

            batch_size = global_batch_size

            # Rectified Flow: t ~ Logit-Normal (SD3 스타일)
            rng_key, subkey = jax.random.split(rng_key)
            t = self.schedule.sample_timesteps(subkey, batch_size)

            # 노이즈 샘플링 (x_1 = noise, standard Gaussian)
            rng_key, subkey = jax.random.split(rng_key)
            compute_dtype = jnp.bfloat16 if self.config.use_bfloat16 else jnp.float32
            x_1 = jax.random.normal(subkey, batch_latents.shape, dtype=compute_dtype)

            # VAE scaling: 원본 latents에 scaling factor 적용
            # SDXL-VAE의 raw latent 출력을 ~N(0,1) 범위로 정규화
            x_0 = (batch_latents * self.config.vae_scaling_factor).astype(compute_dtype)
            batch_embeddings = batch_embeddings.astype(compute_dtype)

            # Rectified Flow forward: x_t = (1 - t) * x_0 + t * x_1
            x_t = self.schedule.forward(x_0, x_1, t)

            # Velocity target: v = x_1 - x_0
            velocity_target = self.schedule.get_velocity(x_0, x_1)

            # 학습 스텝 (Sharded execution)
            global_step = epoch * self.config.steps_per_epoch + step
            loss, rng_key = self.train_step(x_t, t, velocity_target, batch_embeddings, rng_key)

            loss_val = float(loss)

            # === Loss 검증 (NaN/Inf/폭발 체크) ===
            if np.isnan(loss_val) or np.isinf(loss_val):
                print(f"[LOSS ERROR] Step {step}: loss={loss_val} (NaN or Inf detected!)")
                print(f"  x_t range: [{float(x_t.min()):.4f}, {float(x_t.max()):.4f}]")
                print(f"  x_0 range: [{float(x_0.min()):.4f}, {float(x_0.max()):.4f}]")
                print(f"  x_1 range: [{float(x_1.min()):.4f}, {float(x_1.max()):.4f}]")
                print(f"  t range: [{float(t.min()):.4f}, {float(t.max()):.4f}]")
                sys.stdout.flush()
            elif loss_val > 1e6 and step > 100:  # warmup 이후 loss 폭발 감지
                print(f"[LOSS WARNING] Step {step}: loss={loss_val:.2e} (explosion detected!)")
                print(f"  x_t range: [{float(x_t.min()):.4f}, {float(x_t.max()):.4f}]")
                print(f"  velocity_target range: [{float(velocity_target.min()):.4f}, {float(velocity_target.max()):.4f}]")
                print(f"  learning_rate: {self.lr_schedule(global_step):.8f}")
                sys.stdout.flush()

            losses.append(loss_val)

            # SIGTERM 핸들러에 현재 상태 등록 (긴급 체크포인트용)
            _graceful_killer.register_trainer(self, epoch, step, loss_val)

            # Preemption 감지 시 즉시 종료
            if _graceful_killer.kill_now:
                print(f"\n[PREEMPT] Stopping training due to signal...")
                sys.stdout.flush()
                break

            # Wandb 로깅
            if self.wandb_enabled:
                wandb.log({
                    "loss": float(loss),
                    "learning_rate": self.lr_schedule(global_step),
                    "epoch": epoch + 1,
                    "step": step + 1,
                }, step=global_step)

            step += 1

            if step % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                current_time = time.time()
                elapsed = current_time - last_log_time
                steps_per_sec = 100 / elapsed if elapsed > 0 else 0
                last_log_time = current_time

                print(f"Epoch {epoch+1}/{self.config.num_epochs} "
                      f"Step {step}/{self.config.steps_per_epoch} "
                      f"Loss: {avg_loss:.6f} "
                      f"[{elapsed:.1f}s, {steps_per_sec:.2f} step/s]")

                # Worker 0: loss 로그 파일에 기록
                if loss_log_file is not None:
                    timestamp = datetime.now().isoformat()
                    lr = float(self.lr_schedule(global_step))
                    loss_log_file.write(f"{timestamp},{epoch+1},{step},{global_step},{avg_loss:.6f},{lr:.8f}\n")
                    loss_log_file.flush()

            if step >= self.config.steps_per_epoch:
                break

        # Loss 로그 파일 닫기
        if loss_log_file is not None:
            loss_log_file.close()

        epoch_avg_loss = np.mean(losses) if losses else 0.0
        return losses, epoch_avg_loss


# ============================================
# Main
# ============================================
def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='TPU v5e 32 Pod Training (256² XUT-Small)')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh training (ignore existing checkpoints)')
    return parser.parse_args()


def main():
    import logging
    import os as _os

    # 커맨드라인 인자 파싱
    args = parse_args()

    # Worker별 고유 로그 파일 (권한 문제 방지)
    worker_id = _os.environ.get('JAX_PROCESS_INDEX', '0')
    log_file = f'/tmp/train_worker_{worker_id}.log'

    # 파일 + 콘솔 로깅
    try:
        handlers = [
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    except PermissionError:
        # 파일 권한 오류 시 콘솔만 사용
        handlers = [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=handlers
    )
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("TPU v5e 32 Pod Training (256² XUT-Small)")
    print("="*60)
    sys.stdout.flush()
    logger.info("Main function started")
    
    # 멀티프로세스 초기화 (TPU Pod용) - optional
    process_index = 0
    process_count = 1
    
    # Distributed 환경 감지
    import os
    print(f"\n[Step 1] Environment Check")
    logger.info("[Step 1] Environment Check starting...")
    sys.stdout.flush()

    # 환경변수 확인 (jax.devices() 호출 전에 분산 초기화 필요)
    coordinator_addr = os.environ.get("JAX_COORDINATOR_ADDRESS")
    num_processes = os.environ.get("JAX_NUM_PROCESSES")
    process_idx = os.environ.get("JAX_PROCESS_INDEX")

    print(f"  JAX_COORDINATOR_ADDRESS: {coordinator_addr or 'Not set'}")
    print(f"  JAX_NUM_PROCESSES: {num_processes or 'Not set'}")
    print(f"  JAX_PROCESS_INDEX: {process_idx or 'Not set'}")
    sys.stdout.flush()

    use_distributed = coordinator_addr is not None and num_processes is not None

    if use_distributed:
        print(f"\n[Step 2] JAX Distributed Initialization")
        print(f"  Coordinator: {coordinator_addr}")
        print(f"  Num processes: {num_processes}")
        print(f"  This process index: {process_idx}")
        sys.stdout.flush()
        logger.info("Attempting JAX distributed init...")

        try:
            # 분산 초기화 (devices 조회 전에 반드시 수행)
            jax.distributed.initialize(
                coordinator_address=coordinator_addr,
                num_processes=int(num_processes),
                process_id=int(process_idx) if process_idx else None,
            )
            process_index = jax.process_index()
            process_count = jax.process_count()
            local_device_count = jax.local_device_count()
            print(f"  ✓ Distributed Setup Success")
            print(f"    Process: {process_index}/{process_count}")
            print(f"    Local devices: {local_device_count}")
            print(f"    Total devices: {jax.device_count()}")
            logger.info(f"Distributed init success: {process_index}/{process_count}")
        except Exception as e:
            print(f"  ✗ Distributed init failed: {e}")
            print(f"  Falling back to single-host mode")
            logger.error(f"Distributed init failed: {e}", exc_info=True)
            use_distributed = False
            process_index = 0
            process_count = 1
        sys.stdout.flush()
    else:
        print(f"\n[Step 2] Single-Host Mode (no JAX distributed)")
        logger.info("Using single-host mode")
        process_index = 0
        process_count = 1

    # 이제 devices 조회 가능
    try:
        dev_count = len(jax.devices())
        print(f"  Device count: {dev_count}")
        logger.info(f"Device count: {dev_count}")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"[Step 1] Error getting devices: {e}", exc_info=True)
        print(f"  ERROR: {e}")
        sys.stdout.flush()
        raise
    
    print(f"\n[Step 3] Creating TrainingConfig256...")
    sys.stdout.flush()
    config = TrainingConfig256()
    print(f"  ✓ Config created")
    
    print(f"\n[Step 4] Initializing Wandb (Process {process_index} only)...")
    sys.stdout.flush()
    
    # Wandb 초기화 (Process 0만)
    wandb_enabled = False
    if process_index == 0:
        try:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config={
                    "global_batch_size": config.global_batch_size,
                    "learning_rate": config.learning_rate,
                    "num_epochs": config.num_epochs,
                    "tread_selection_rate": config.tread_selection_rate,
                    "model_dim": config.model_dim,
                    "depth": config.depth,
                    "warmup_steps": config.warmup_steps,
                    "process_count": process_count,
                    "num_workers": 112,
                },
                name=f"xut-small-256-tpu-pod-32"
            )
            wandb_enabled = True
            print(f"  ✓ Wandb initialized")
        except Exception as e:
            print(f"  ⚠ Wandb init failed (non-critical): {e}")
    else:
        print(f"  (Skipped - not process 0)")
    sys.stdout.flush()
    
    print(f"\n[Step 5] Configuration Summary:")
    print(f"  TPU devices: {config.num_devices} cores")
    print(f"  CPU workers: 112 vCPUs (data loading + prefetch)")
    print(f"  Global batch size: {config.global_batch_size}")
    print(f"  Batch per device: {config.batch_size_per_device}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  TREAD selection rate: {config.tread_selection_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Steps per epoch: {config.steps_per_epoch}")
    sys.stdout.flush()
    
    # 디바이스 확인
    print("\n" + "="*60)
    print("[Step 6] Device Detection")
    print("="*60)
    sys.stdout.flush()
    
    devices = jax.devices()
    print(f"  Total devices: {len(devices)}")
    print(f"  Devices: {devices}")
    if devices:
        print(f"  Device type: {devices[0].device_kind}")
    else:
        print("  ⚠ Warning: No devices detected!")
    sys.stdout.flush()
    
    # Text embedding provider
    print("\n" + "="*60)
    print("[Step 7] Text Embedding Model Setup")
    print("="*60)
    print(f"  Model: {config.embedding_model}")
    print(f"  Dimension: {config.context_dim}")
    print(f"  Strategy: Pre-computed embeddings (from PT files)")
    sys.stdout.flush()

    # Pre-computed embeddings 사용 (PT 파일에 포함)
    # CPU 병목 제거! embedding_provider 불필요
    embedding_provider = get_embedding_provider(
        model_name=config.embedding_model,
        use_precomputed=True,  # PT 파일의 pre-computed embeddings 사용
        embedding_dim=config.context_dim
    )
    # embedding_provider is None in precomputed mode
    sys.stdout.flush()
    
    # GCS 데이터 설정
    print("\n" + "="*60)
    print("[Step 8] Data Setup")
    print("="*60)

    print(f"  GCS Bucket: {config.gcs_bucket}")
    print(f"  Mode: {'RAM Preload' if config.use_ram_preload else 'GCS Streaming'}")
    sys.stdout.flush()

    # 분산 학습: 각 worker는 local batch만 로드 (global / num_processes)
    local_batch_size = config.global_batch_size // jax.process_count()
    print(f"  Local batch size per worker: {local_batch_size} (global {config.global_batch_size} / {jax.process_count()} processes)")

    data_session = None  # 통합 세션 변수

    try:
        if config.use_ram_preload:
            # === RAM 프리로드 모드 (권장) ===
            print(f"\n  [8a] Initializing RAMPreloadSession...")
            print(f"    Download workers: {config.ram_preload_download_workers}")
            print(f"    Expected memory: ~69GB per worker (23 PT files × 3GB)")
            sys.stdout.flush()

            data_session = RAMPreloadSession(
                batch_size=local_batch_size,
                gcs_bucket=config.gcs_bucket,
                parquet_path=config.parquet_file,
                num_download_workers=config.ram_preload_download_workers,
                shard_pt_files=True
            )
            print(f"  ✓ RAM Preload session initialized")
            print(f"    PT files loaded: {len(data_session.pt_files)} (sharded)")
            print(f"    Total samples in RAM: {data_session.total_samples:,}")
        else:
            # === GCS 스트리밍 모드 (레거시) ===
            print(f"\n  [8a] Initializing GCSDataLoaderSession...")
            print(f"    CPU Workers: {config.num_data_workers} vCPUs")
            print(f"    Prefetch ahead: {config.prefetch_ahead} PT files")
            print(f"    Max cache files: {config.max_cache_files} (disk optimization)")
            sys.stdout.flush()

            data_session = GCSDataLoaderSession(
                batch_size=local_batch_size,
                parquet_path=config.parquet_file or f"{config.gcs_bucket}coyo11m-meta.parquet",
                embedding_provider=embedding_provider,
                gcs_bucket=config.gcs_bucket,
                cache_dir=config.cache_dir,
                num_workers=config.num_data_workers,
                prefetch_ahead=config.prefetch_ahead,
                max_cache_files=config.max_cache_files
            )
            print(f"  ✓ GCS session initialized")
            print(f"    PT files for this worker: {len(data_session.pt_files)} (sharded)")
            print(f"    Total samples (global): {data_session.total_samples:,}")
            if data_session.pt_files:
                print(f"    First PT file: {data_session.pt_files[0]}")
                print(f"    Last PT file: {data_session.pt_files[-1]}")

        # steps_per_epoch 자동 계산 (None이면)
        if config.steps_per_epoch is None:
            config.steps_per_epoch = data_session.calculate_steps_per_epoch(config.global_batch_size)
        else:
            print(f"  Using manual steps_per_epoch: {config.steps_per_epoch}")

        sys.stdout.flush()
    except Exception as e:
        print(f"  ✗ Failed to initialize data session: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return
    
    # 모델 초기화
    print("\n" + "="*60)
    print("[Step 9] Model Initialization")
    print("="*60)
    
    try:
        print(f"  Creating XUT-Small model...")
        sys.stdout.flush()
        model = create_xut_small(
            dim=config.model_dim,
            ctx_dim=config.context_dim,
            mlp_dim=config.mlp_dim,
            heads=config.heads,
            depth=config.depth,
            enc_blocks=config.enc_blocks,
            dec_blocks=config.dec_blocks,
        )

        # bfloat16 변환 (메모리 절약 + TPU 최적화)
        if config.use_bfloat16:
            print(f"  Converting model to bfloat16...")

            def to_bf16(x):
                if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating):
                    return x.astype(jnp.bfloat16)
                return x

            for path, value in nnx.iter_graph(model):
                if isinstance(value, nnx.Variable) and hasattr(value, 'value'):
                    value.value = to_bf16(value.value)
            print(f"  ✓ Model converted to bfloat16")

        print(f"  ✓ XUT-Small initialized")
        print(f"    Dimension: {config.model_dim}")
        print(f"    Context dim: {config.context_dim}")
        print(f"    Depth: {config.depth}")
        print(f"    dtype: {'bfloat16' if config.use_bfloat16 else 'float32'}")
        sys.stdout.flush()
    except Exception as e:
        print(f"  ✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return
    
    # 옵티마이저 (AdamW with weight decay + muP scaling)
    # muP: lr_scaled = base_lr * (base_dim / model_dim)
    print(f"\n[Step 10] Creating Optimizer...")
    sys.stdout.flush()

    try:
        # muP learning rate scaling
        mup_lr = config.learning_rate * (config.mup_base_dim / config.model_dim)
        print(f"  muP scaling: {config.learning_rate} * ({config.mup_base_dim}/{config.model_dim}) = {mup_lr:.6f}")

        optimizer = nnx.Optimizer(
            model,
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=mup_lr, weight_decay=1e-4)
            ),
            wrt=nnx.Param  # Flax 0.11.0+ requires wrt argument
        )
        print(f"  ✓ Optimizer created (AdamW + gradient clipping, lr={mup_lr:.6f})")
        sys.stdout.flush()
    except Exception as e:
        print(f"  ✗ Failed to create optimizer: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return

    # 체크포인트에서 Resume (--fresh가 아닐 때)
    start_epoch = 0
    start_step = 0
    resumed_run_id = None

    if not args.fresh:
        print("\n" + "="*60)
        print("[Step 10.5] Checking for existing checkpoints...")
        print("="*60)
        sys.stdout.flush()

        ckpt_data = find_latest_checkpoint(config)

        # 분산 학습: Process 0의 결과를 다른 프로세스에 broadcast
        if use_distributed:
            from jax.experimental.multihost_utils import broadcast_one_to_all
            # Process 0에서 찾은 체크포인트 데이터를 serialize해서 broadcast
            if process_index == 0 and ckpt_data is not None:
                ckpt_bytes = pickle.dumps(ckpt_data)
                ckpt_size = len(ckpt_bytes)
            else:
                ckpt_bytes = b''
                ckpt_size = 0

            # 크기 먼저 broadcast
            ckpt_size_arr = jnp.array([ckpt_size if process_index == 0 else 0])
            ckpt_size_arr = broadcast_one_to_all(ckpt_size_arr)
            ckpt_size = int(ckpt_size_arr[0])

            if ckpt_size > 0:
                # 체크포인트 데이터 broadcast
                if process_index == 0:
                    ckpt_np = np.frombuffer(ckpt_bytes, dtype=np.uint8)
                else:
                    ckpt_np = np.zeros(ckpt_size, dtype=np.uint8)

                ckpt_jax = jnp.array(ckpt_np)
                ckpt_jax = broadcast_one_to_all(ckpt_jax)
                ckpt_bytes = bytes(np.array(ckpt_jax))
                ckpt_data = pickle.loads(ckpt_bytes)

        if ckpt_data is not None:
            try:
                start_epoch, start_step, resumed_run_id = restore_from_checkpoint(
                    model, optimizer, ckpt_data, config
                )
                print(f"  ✓ Resumed from epoch={start_epoch}, step={start_step}")
            except Exception as e:
                print(f"  ✗ Resume failed, starting fresh: {e}")
                start_epoch = 0
                start_step = 0
        else:
            print(f"  No checkpoint found, starting fresh training")
    else:
        print("\n" + "="*60)
        print("[Step 10.5] --fresh flag set, starting new training")
        print("="*60)
    sys.stdout.flush()

    # 스케줄 (Rectified Flow)
    print(f"\n[Step 11] Creating Rectified Flow Schedule...")
    sys.stdout.flush()

    schedule = RectifiedFlowSchedule()
    print(f"  ✓ Rectified Flow schedule created")
    sys.stdout.flush()
    
    # 학습기
    print("\n" + "="*60)
    print("[Step 12] Initializing TPUTrainer...")
    print("="*60)
    sys.stdout.flush()

    trainer = TPUTrainer(
        model, optimizer, schedule, config,
        wandb_enabled=wandb_enabled,
        run_id=resumed_run_id  # 이전 run 재개 시 동일 run_id 사용
    )
    print(f"  ✓ TPUTrainer initialized")
    print(f"  Run ID: {trainer.run_id}")
    sys.stdout.flush()

    print("\n" + "="*70)
    print("[Step 13] Training Starting")
    print("="*70)
    print(f"  Start epoch: {start_epoch} (step: {start_step})")
    print(f"  Total epochs: {config.num_epochs}")
    print(f"  PT files per epoch: {len(data_session.pt_files)}")
    print(f"  Steps per epoch: {config.steps_per_epoch:,} ({data_session.total_samples:,} samples / {config.global_batch_size} batch)")
    print(f"  Global batch size: {config.global_batch_size}")
    print(f"  Checkpoint: every epoch")
    print(f"  Data mode: {'RAM Preload (network I/O free)' if config.use_ram_preload else 'GCS Streaming'}")
    sys.stdout.flush()

    total_start = time.time()

    # Epoch별로 모든 PT 파일 순회 (1 epoch = 모든 PT 파일 한 바퀴)
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\n{'='*70}")
        print(f"[Epoch {epoch+1}/{config.num_epochs}]")
        print(f"{'='*70}")
        sys.stdout.flush()
        
        epoch_start = time.time()
        epoch_losses = []
        
        # 에포크 로더 생성
        print(f"  [E{epoch+1}a] Creating epoch loader...")
        sys.stdout.flush()

        try:
            if config.use_ram_preload:
                # RAM 모드: 메모리에서 직접 샘플링 (네트워크 I/O 없음)
                epoch_loader = data_session.get_epoch_loader(
                    epoch=epoch,
                    steps_per_epoch=config.steps_per_epoch,
                    num_workers=config.num_data_workers
                )
            else:
                # GCS 스트리밍 모드
                epoch_loader = data_session.get_epoch_loader(
                    epoch=epoch,
                    steps_per_epoch=config.steps_per_epoch,
                    num_download_workers=config.num_download_workers,
                    num_load_workers=config.num_load_workers
                )
            print(f"  [E{epoch+1}b] ✓ Epoch loader ready")
            sys.stdout.flush()
        except Exception as e:
            print(f"  [E{epoch+1}b] ✗ Failed to create epoch loader: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            break
        
        # 학습 루프 시작 전 모든 worker 동기화
        print(f"  [E{epoch+1}c] Waiting for all workers to be ready...")
        sys.stdout.flush()

        # JAX barrier: 모든 worker가 이 지점에 도달할 때까지 대기
        from jax.experimental.multihost_utils import sync_global_devices
        sync_global_devices(f"epoch_{epoch}_start")

        print(f"  [E{epoch+1}c] All workers ready, starting training...")
        sys.stdout.flush()

        pt_files_processed = 0
        total_batches_processed = 0

        try:
            # Resume: 첫 epoch이고 start_step > 0이면 해당 step부터 시작
            epoch_start_step = start_step if epoch == start_epoch else 0
            losses, epoch_avg_loss = trainer.train_epoch(epoch_loader, epoch, epoch_start_step)
            epoch_losses.extend(losses)
            total_batches_processed += len(losses)

            # PT 파일 개수 계산 (대략적)
            if losses:
                pt_files_processed = len(data_session.pt_files)
        except Exception as e:
            print(f"  ✗ Training error: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
        finally:
            print(f"  [E{epoch+1}d] Stopping epoch loader...")
            sys.stdout.flush()
            epoch_loader.stop()
            gc.collect()
            print(f"  [E{epoch+1}e] ✓ Cleanup done")
            sys.stdout.flush()
        
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        
        print(f"\n{'='*70}")
        print(f"✓ EPOCH {epoch+1} completed in {epoch_time/3600:.1f}h")
        print(f"  Batches processed: {total_batches_processed}")
        print(f"  Average loss: {epoch_avg_loss:.6f}")
        if epoch_losses:
            print(f"  Min loss: {np.min(epoch_losses):.6f}")
            print(f"  Max loss: {np.max(epoch_losses):.6f}")
        print(f"{'='*70}")
        
        # 에포크 레벨 wandb 로깅 (Process 0만)
        if process_index == 0 and wandb_enabled:
            wandb.log({
                "epoch_avg_loss": epoch_avg_loss,
                "epoch_min_loss": np.min(epoch_losses) if epoch_losses else 0.0,
                "epoch_max_loss": np.max(epoch_losses) if epoch_losses else 0.0,
                "epoch_time_hours": epoch_time / 3600,
                "num_pt_files": len(data_session.pt_files),
                "batches_processed": total_batches_processed,
                "epoch": epoch + 1,
            }, step=epoch)

        # 10 에폭마다 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(epoch, config.steps_per_epoch, epoch_avg_loss)
            if process_index == 0 and wandb_enabled:
                wandb.log({"checkpoint_saved": epoch + 1})

        # Preemption 감지 시 루프 종료
        if _graceful_killer.kill_now:
            print(f"\n[PREEMPT] Training interrupted by signal, exiting...")
            sys.stdout.flush()
            break

    total_time = time.time() - total_start

    if _graceful_killer.kill_now:
        print("\n" + "="*60)
        print("⚠ Training interrupted (preemption)")
        print("="*60)
        print(f"  Last checkpoint saved. Resume with: ./run_tpu_distributed.sh")
    else:
        print("\n" + "="*60)
        print("✓ Training completed!")
        print("="*60)
    print(f"Total training time: {total_time/3600:.1f}h")

    # 데이터 세션 종료
    data_session.shutdown()
    
    # 최종 통계 (Process 0만)
    if process_index == 0 and wandb_enabled:
        wandb.log({
            "total_training_time_hours": total_time / 3600,
            "completed": True,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
