"""
TPU v5e 16 Pod에서 256² 이미지로 XUT-Small 학습
Batch Size: 2048 (한 번에 처리)
Dataset: KBlueLeaf/coyo11m-256px-ccrop-latent
"""

import os
import gc
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

from src.xut.xut_small import create_xut_small
from src.embeddings import get_embedding_provider
from src.data.gcs_dataloader import (
    GCSDataLoaderSession, 
    GCSFileHandler, 
    ParquetCache
)


# ============================================
# Configuration
# ============================================
@dataclass
class TrainingConfig256:
    """256² 스테이지 학습 설정"""
    # 배치 및 데이터
    global_batch_size: int = 2048      # 분할 안 함
    num_devices: int = 16              # TPU v5e pod size (또는 112 for TPU v5e 256)
    batch_size_per_device: int = 128   # 2048 / 16 = 128
    
    # 학습
    num_epochs: int = 20
    steps_per_epoch: int = 3750         # 7.624M / 2048
    learning_rate: float = 0.5          # muP base_dim=1
    warmup_steps: int = 1000
    
    # 모델 (XUT-Small)
    model_dim: int = 896
    context_dim: int = 768              # Embedding Gemma 300M dimension
    mlp_dim: int = 3072
    heads: int = 14
    depth: int = 4
    enc_blocks: int = 1
    dec_blocks: int = 2
    
    # 노이즈 스케줄
    beta_min: float = 0.0001
    beta_max: float = 0.02
    T: int = 1000
    
    # Text embedding
    embedding_model: str = "google/embeddinggemma-300m"  # 768d
    
    # TREAD (Timestep-Random Encoder Architecture Design)
    tread_selection_rate: float = 0.5  # 기존 연구 설정값
    
    # GCS 설정
    gcs_bucket: str = "gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/"
    parquet_file: str = None  # 자동으로 GCS에서 찾음
    cache_dir: str = None  # 자동으로 /tmp 사용
    num_data_workers: int = 8  # Prefetch workers (pre-computed embeddings이므로 적게 필요)
    prefetch_ahead: int = 4  # 미리 준비할 배치 수
    max_cache_files: int = 2  # 최대 동시 캐시 PT 파일 (worker당 ~5.4GB)
    
    # TPU 설정
    use_pjit: bool = True
    use_gradient_checkpointing: bool = True
    
    # Wandb
    wandb_project: str = "xut-small-256"
    wandb_entity: str = None  # set to username


# ============================================
# Diffusion Schedule (기존 HDM 코드와 동일)
# ============================================
class DiffusionSchedule:
    """노이즈 스케줄 (alphas_cumprod 기반)"""
    
    def __init__(self, beta_min: float = 0.0001, beta_max: float = 0.02, T: int = 1000):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self._cache_alphas()
    
    def _cache_alphas(self):
        # Linear schedule
        betas = jnp.linspace(self.beta_min, self.beta_max, self.T)
        self.alphas = 1.0 - betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
        # sqrt_alphas_cumprod and sqrt(1 - alphas_cumprod)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
    
    def forward_diffusion(self, x_0: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray) -> jnp.ndarray:
        """
        Forward diffusion: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        
        Args:
            x_0: (B, 4, 32, 32) - 원본 VAE latent
            noise: (B, 4, 32, 32) - 노이즈
            timesteps: (B,) - 각 샘플의 타임스텝
        Returns:
            x_t: (B, 4, 32, 32) - noisy latent
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps]  # (B,)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps]  # (B,)
        
        # Reshape for broadcasting: (B,) -> (B, 1, 1, 1)
        sqrt_alpha = sqrt_alpha[:, None, None, None]
        sqrt_one_minus_alpha = sqrt_one_minus_alpha[:, None, None, None]
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t


# ============================================
# Sharding Strategy (TPU Pod 16)
# ============================================
@dataclass
class ShardingRules:
    """TPU Pod 16 분산 전략

    TPU v5e-16 구성:
    - 4 workers (hosts), 각 4개 TPU 칩
    - 총 16 TPU 코어
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
    # v5e 16 pod: 16개 TPU 칩
    devices = mesh_utils.create_device_mesh((num_devices,))
    return Mesh(devices, axis_names=("batch",))


class TPUTrainer:
    """TPU 분산 학습기 (Sharding 적용)"""

    def __init__(self, model, optimizer, schedule, config: TrainingConfig256, wandb_enabled: bool = False):
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
        
        # Checkpoint 디렉토리
        self.checkpoint_dir = Path("./checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def _create_lr_schedule(self):
        """Warmup + Cosine decay"""
        def lr_fn(step):
            if step < self.config.warmup_steps:
                return self.config.learning_rate * (step / self.config.warmup_steps)
            else:
                progress = (step - self.config.warmup_steps) / (
                    self.config.steps_per_epoch * self.config.num_epochs - self.config.warmup_steps
                )
                return self.config.learning_rate * 0.5 * (1 + jnp.cos(jnp.pi * progress))
        
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
    
    def train_step(self, x_t, timesteps, noise, text_emb, step, rng_key):
        """한 스텝 학습 (Sharded + JIT 컴파일)

        배치는 'data' 축으로 분산
        모델 파라미터는 'model' 축으로 분산
        """
        # TREAD: Timestep-Random Encoder Architecture Design
        batch_size = timesteps.shape[0]
        rng_key, subkey = jax.random.split(rng_key)
        mask = jax.random.uniform(subkey, (batch_size,)) < self.config.tread_selection_rate
        t_cond = jnp.where(mask, jnp.zeros_like(timesteps), timesteps)

        # text_emb: (batch, dim) → (batch, 1, dim) for sequence concatenation
        # 모델은 ctx가 3D (batch, seq_len, dim) 형태를 기대함
        text_emb_3d = text_emb[:, None, :]

        # JIT된 내부 함수로 gradient 계산 및 업데이트
        # nnx.jit은 nnx.Module과 optimizer를 인자로 받아야 함
        @nnx.jit
        def _train_step_jit(model, optimizer, x_t, t_cond, noise, text_emb):
            def loss_fn(model):
                # 모델 입력: NHWC, 출력: NCHW
                pred_noise_nchw = model(x_t, t_cond, text_emb)
                # NCHW -> NHWC for loss computation (noise is NHWC)
                pred_noise = jnp.transpose(pred_noise_nchw, (0, 2, 3, 1))
                return jnp.mean((pred_noise - noise) ** 2)

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(grads)
            return loss

        loss = _train_step_jit(self.model, self.optimizer, x_t, t_cond, noise, text_emb_3d)
        return loss, rng_key
    
    def save_checkpoint(self, epoch: int, step: int, loss: float):
        """체크포인트 저장"""
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:03d}_step_{step:06d}.ckpt"
        try:
            # nnx를 사용하려면 모델이 nnx.Module이어야 함
            # nnx.save(self.model, str(checkpoint_path))
            # 임시로 간단한 메타데이터만 저장
            checkpoint_data = {
                'epoch': epoch,
                'step': step,
                'loss': loss,
            }
            import pickle
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"  ✗ Failed to save checkpoint: {e}")
    
    def train_epoch(self, prefetch_loader: 'PrefetchDataLoader', epoch: int):
        """에포크 학습 (Sharded prefetch 파이프라인)"""
        losses = []
        rng_key = jax.random.PRNGKey(epoch)
        step = 0

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
            local_batch_size = batch_latents.shape[0]
            per_device_batch = local_batch_size // num_local_devices

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
            
            # 타임스텝 샘플링
            rng_key, subkey = jax.random.split(rng_key)
            timesteps = jax.random.randint(subkey, (batch_size,), 0, self.config.T)
            
            # 노이즈 샘플링
            rng_key, subkey = jax.random.split(rng_key)
            noise = jax.random.normal(subkey, batch_latents.shape, dtype=jnp.float32)
            
            # Forward diffusion
            x_t = self.schedule.forward_diffusion(batch_latents, noise, timesteps)
            
            # 학습 스텝 (Sharded execution)
            global_step = epoch * self.config.steps_per_epoch + step
            loss, rng_key = self.train_step(x_t, timesteps, noise, batch_embeddings, 
                                            global_step, rng_key)
            
            losses.append(float(loss))

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
                print(f"Epoch {epoch+1}/{self.config.num_epochs} "
                      f"Step {step}/{self.config.steps_per_epoch} "
                      f"Loss: {avg_loss:.6f} [Sharded]")
            
            if step >= self.config.steps_per_epoch:
                break
        
        epoch_avg_loss = np.mean(losses) if losses else 0.0
        return losses, epoch_avg_loss


# ============================================
# Main
# ============================================
def main():
    import sys
    import logging
    import os as _os

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
    print("TPU v5e 16 Pod Training (256² XUT-Small)")
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
                name=f"xut-small-256-tpu-pod-16"
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

    try:
        print(f"  Initializing embedding provider...")
        sys.stdout.flush()
        # Pre-computed embeddings 사용 (PT 파일에 포함)
        # CPU 병목 제거!
        embedding_provider = get_embedding_provider(
            model_name=config.embedding_model,
            use_precomputed=True,  # PT 파일의 pre-computed embeddings 사용
            embedding_dim=config.context_dim
        )
        print(f"  ✓ Text embedding provider loaded (precomputed mode)")
        sys.stdout.flush()
    except Exception as e:
        print(f"  ✗ Failed to load embedding provider: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return
    
    # GCS 데이터 설정
    print("\n" + "="*60)
    print("[Step 8] GCS Data Setup")
    print("="*60)
    
    print(f"  GCS Bucket: {config.gcs_bucket}")
    print(f"  CPU Workers: {config.num_data_workers} vCPUs")
    print(f"  Prefetch ahead: {config.prefetch_ahead} PT files")
    print(f"  Max cache files: {config.max_cache_files} (disk optimization)")
    sys.stdout.flush()
    
    try:
        print(f"\n  [8a] Initializing GCSDataLoaderSession...")
        sys.stdout.flush()
        
        # GCS 데이터로더 세션 초기화
        # 분산 학습: 각 worker는 local batch만 로드 (global / num_processes)
        local_batch_size = config.global_batch_size // jax.process_count()
        print(f"    Local batch size per worker: {local_batch_size} (global {config.global_batch_size} / {jax.process_count()} processes)")

        gcs_session = GCSDataLoaderSession(
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
        print(f"    PT files found: {len(gcs_session.pt_files)}")
        if gcs_session.pt_files:
            print(f"    First PT file: {gcs_session.pt_files[0]}")
            print(f"    Last PT file: {gcs_session.pt_files[-1]}")
        sys.stdout.flush()
    except Exception as e:
        print(f"  ✗ Failed to initialize GCS session: {e}")
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
        model = create_xut_small()
        print(f"  ✓ XUT-Small initialized")
        print(f"    Dimension: 896")
        print(f"    Context dim: 640")
        print(f"    Depth: 4")
        print(f"    Parameters: ~237M (XUT part) + ~270M (Gemma)")
        sys.stdout.flush()
    except Exception as e:
        print(f"  ✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return
    
    # 옵티마이저 (AdamW with weight decay)
    # Note: learning_rate는 train_step에서 동적으로 설정됨
    print(f"\n[Step 10] Creating Optimizer...")
    sys.stdout.flush()
    
    try:
        optimizer = nnx.Optimizer(
            model,
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=config.learning_rate, weight_decay=1e-4)
            )
        )
        print(f"  ✓ Optimizer created (AdamW + gradient clipping, lr={config.learning_rate})")
        sys.stdout.flush()
    except Exception as e:
        print(f"  ✗ Failed to create optimizer: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return
    
    # 스케줄
    print(f"\n[Step 11] Creating Diffusion Schedule...")
    sys.stdout.flush()
    
    schedule = DiffusionSchedule(
        beta_min=config.beta_min,
        beta_max=config.beta_max,
        T=config.T
    )
    print(f"  ✓ Diffusion schedule created (T={config.T})")
    sys.stdout.flush()
    
    # 학습기
    print("\n" + "="*60)
    print("[Step 12] Initializing TPUTrainer...")
    print("="*60)
    sys.stdout.flush()
    
    trainer = TPUTrainer(model, optimizer, schedule, config, wandb_enabled=wandb_enabled)
    print(f"  ✓ TPUTrainer initialized")
    sys.stdout.flush()
    
    print("\n" + "="*70)
    print("[Step 13] Training Starting")
    print("="*70)
    print(f"  Total epochs: {config.num_epochs}")
    print(f"  PT files per epoch: {len(gcs_session.pt_files)}")
    print(f"  Steps per PT file: {config.steps_per_epoch}")
    print(f"  Global batch size: {config.global_batch_size}")
    sys.stdout.flush()
    
    total_start = time.time()
    
    # Epoch별로 모든 PT 파일 순회 (1 epoch = 모든 PT 파일 한 바퀴)
    for epoch in range(config.num_epochs):
        print(f"\n{'='*70}")
        print(f"[Epoch {epoch+1}/{config.num_epochs}]")
        print(f"{'='*70}")
        sys.stdout.flush()
        
        epoch_start = time.time()
        epoch_losses = []
        
        # GCS 에포크 로더 생성 (자동 prefetch + 병렬 다운로드)
        print(f"  [E{epoch+1}a] Creating epoch loader...")
        sys.stdout.flush()
        
        try:
            gcs_prefetch_loader = gcs_session.get_epoch_loader(
                epoch=epoch,
                steps_per_epoch=config.steps_per_epoch
            )
            print(f"  [E{epoch+1}b] ✓ Epoch loader ready")
            sys.stdout.flush()
        except Exception as e:
            print(f"  [E{epoch+1}b] ✗ Failed to create epoch loader: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            break
        
        # 학습 루프
        print(f"  [E{epoch+1}c] Starting training (pls wait, batches processing)...")
        sys.stdout.flush()
        
        pt_files_processed = 0
        total_batches_processed = 0
        
        try:
            losses, epoch_avg_loss = trainer.train_epoch(gcs_prefetch_loader, epoch)
            epoch_losses.extend(losses)
            total_batches_processed += len(losses)
            
            # PT 파일 개수 계산 (대략적)
            if losses:
                pt_files_processed = len(gcs_session.pt_files)
        except Exception as e:
            print(f"  ✗ Training error: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
        finally:
            print(f"  [E{epoch+1}d] Stopping prefetch loader...")
            sys.stdout.flush()
            gcs_prefetch_loader.stop()
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
                "num_pt_files": len(gcs_session.pt_files),
                "batches_processed": total_batches_processed,
                "epoch": epoch + 1,
            }, step=epoch)

        # 매 에포크 끝에 체크포인트 저장
        trainer.save_checkpoint(epoch, config.steps_per_epoch, epoch_avg_loss)
        if process_index == 0 and wandb_enabled:
            wandb.log({"checkpoint_saved": epoch + 1})
    
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print("✓ Training completed!")
    print("="*60)
    print(f"Total training time: {total_time/3600:.1f}h")
    
    # GCS 세션 종료
    gcs_session.shutdown()
    
    # 최종 통계 (Process 0만)
    if process_index == 0 and wandb_enabled:
        wandb.log({
            "total_training_time_hours": total_time / 3600,
            "completed": True,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
