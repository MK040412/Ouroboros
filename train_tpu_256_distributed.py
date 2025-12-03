"""
TPU v5e Multi-Host Training with Distributed Data Loading
Based on JAX official documentation:
- https://docs.jax.dev/en/latest/multi_process.html
- https://docs.jax.dev/en/latest/distributed_data_loading.html
- https://docs.jax.dev/en/latest/the-training-cookbook.html
"""

import os
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P
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


# ============================================
# Configuration
# ============================================
@dataclass
class TrainingConfig256:
    """256² 스테이지 학습 설정"""
    # 배치 및 데이터
    global_batch_size: int = 2048
    num_devices: int = 16
    batch_size_per_device: int = 128
    
    # 학습
    num_epochs: int = 20
    steps_per_epoch: int = 3750
    learning_rate: float = 0.5
    warmup_steps: int = 1000
    
    # 모델
    model_dim: int = 896
    context_dim: int = 768
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
    embedding_model: str = "google/embeddinggemma-300m"
    
    # TREAD
    tread_selection_rate: float = 0.5
    
    # TPU 설정
    use_pjit: bool = True
    use_gradient_checkpointing: bool = True
    
    # Wandb
    wandb_project: str = "xut-small-256"
    wandb_entity: str = None


# ============================================
# Diffusion Schedule
# ============================================
class DiffusionSchedule:
    """노이즈 스케줄 (alphas_cumprod 기반)"""
    
    def __init__(self, beta_min: float = 0.0001, beta_max: float = 0.02, T: int = 1000):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self._cache_alphas()
    
    def _cache_alphas(self):
        betas = jnp.linspace(self.beta_min, self.beta_max, self.T)
        self.alphas = 1.0 - betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
    
    def forward_diffusion(self, x_0: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray) -> jnp.ndarray:
        """Forward diffusion process"""
        sqrt_alphas = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        for _ in range(len(x_0.shape) - 1):
            sqrt_alphas = jnp.expand_dims(sqrt_alphas, axis=-1)
            sqrt_one_minus_alphas = jnp.expand_dims(sqrt_one_minus_alphas, axis=-1)
        
        return sqrt_alphas * x_0 + sqrt_one_minus_alphas * noise


# ============================================
# Distributed Data Loader (Option 3: Consolidated per-process)
# ============================================
class Coyo11mDataLoaderDistributed:
    """분산 학습용 데이터 로더 (각 프로세스가 다른 데이터 로드)"""
    
    def __init__(self, batch_size: int, embedding_provider=None):
        self.batch_size = batch_size
        self.embedding_provider = embedding_provider
        
        # 멀티프로세스 정보
        self.process_index = jax.process_index()
        self.process_count = jax.process_count()
        
        # 각 프로세스가 다른 시드로 데이터 셔플 (데이터 병렬화)
        seed = 42 + self.process_index
        
        # HF Streaming Dataset
        print(f"[Process {self.process_index}] Loading HF dataset...")
        from datasets import load_dataset
        self.dataset = load_dataset(
            "KBlueLeaf/coyo11m-256px-ccrop-latent",
            streaming=True,
            split="train"
        )
        self.dataset = self.dataset.shuffle(seed=seed)
        self.dataset_iter = iter(self.dataset)
    
    def get_batch(self):
        """배치 데이터 반환"""
        latents_list = []
        captions_list = []
        
        for _ in range(self.batch_size):
            try:
                sample = next(self.dataset_iter)
                latent = sample['latents']
                caption = sample.get('caption_llava', "")
                
                if isinstance(latent, torch.Tensor):
                    latent = latent.cpu().numpy()
                channels, frames, h, w = latent.shape
                latent = latent.reshape(channels * frames, h, w)[:4, :, :]
                latents_list.append(latent)
                captions_list.append(caption if caption else "")
                
            except StopIteration:
                # 데이터셋 끝나면 다시 시작
                self.dataset_iter = iter(self.dataset)
                sample = next(self.dataset_iter)
                latent = sample['latents']
                caption = sample.get('caption_llava', "")
                
                if isinstance(latent, torch.Tensor):
                    latent = latent.cpu().numpy()
                channels, frames, h, w = latent.shape
                latent = latent.reshape(channels * frames, h, w)[:4, :, :]
                latents_list.append(latent)
                captions_list.append(caption if caption else "")
                
            except Exception as e:
                print(f"[Process {self.process_index}] Error loading sample: {e}")
                latents_list.append(np.zeros((4, 32, 32), dtype=np.float32))
                captions_list.append("")
        
        batch_latents = np.stack(latents_list, axis=0)
        batch_latents = jnp.array(batch_latents, dtype=jnp.float32)
        
        # 각 프로세스가 임베딩 계산 (데이터 병렬화)
        batch_embeddings = self.embedding_provider.batch_encode(
            captions_list, batch_size=512, normalize=True
        )
        
        return batch_latents, batch_embeddings


# ============================================
# Training State & Distributed Setup
# ============================================
def create_distributed_mesh(num_devices: int):
    """분산 학습용 메시 생성"""
    devices = mesh_utils.create_device_mesh((num_devices,))
    return Mesh(devices, axis_names=("batch",))


# ============================================
# Main Training
# ============================================
def main():
    # ========== CRUCIAL: Multi-process initialization ==========
    jax.distributed.initialize()
    
    print("="*70)
    print("JAX Distributed Training - XUT-Small on TPU v5e")
    print("="*70)
    
    # 프로세스 정보
    process_index = jax.process_index()
    process_count = jax.process_count()
    local_device_count = jax.local_device_count()
    
    print(f"\n[Distributed Setup]")
    print(f"  Process: {process_index}/{process_count}")
    print(f"  Local devices: {local_device_count}")
    print(f"  Global devices: {len(jax.devices())}")
    
    config = TrainingConfig256()
    
    # Wandb init (Process 0 only)
    if process_index == 0:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config={
                "global_batch_size": config.global_batch_size,
                "learning_rate": config.learning_rate,
                "processes": process_count,
                "local_devices": local_device_count,
            },
            name=f"xut-small-256-tpu-v5e-distributed"
        )
    
    # 데이터 로더 (각 프로세스가 다른 데이터)
    print(f"\n[Data Loading]")
    embedding_provider = get_embedding_provider(config.embedding_model)
    data_loader = Coyo11mDataLoaderDistributed(
        batch_size=config.global_batch_size,
        embedding_provider=embedding_provider
    )
    
    # 모델 초기화
    print(f"\n[Model]")
    model = create_xut_small()
    print(f"  XUT-Small initialized")
    
    # 옵티마이저
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(weight_decay=1e-4)
        )
    )
    
    # 스케줄
    schedule = DiffusionSchedule(
        beta_min=config.beta_min,
        beta_max=config.beta_max,
        T=config.T
    )
    
    # 학습 루프
    print(f"\n[Training]")
    print(f"  Starting {config.num_epochs} epochs x {config.steps_per_epoch} steps/epoch")
    
    total_start = time.time()
    rng_key = jax.random.PRNGKey(0)
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        losses = []
        
        for step in range(config.steps_per_epoch):
            # 배치 로드
            batch_latents, batch_embeddings = data_loader.get_batch()
            
            # 타임스텝 샘플링
            rng_key, subkey = jax.random.split(rng_key)
            batch_size = batch_latents.shape[0]
            timesteps = jax.random.randint(subkey, (batch_size,), 0, config.T)
            
            # 노이즈 샘플링
            rng_key, subkey = jax.random.split(rng_key)
            noise = jax.random.normal(subkey, batch_latents.shape, dtype=jnp.float32)
            
            # Forward diffusion
            x_t = schedule.forward_diffusion(batch_latents, noise, timesteps)
            
            # 더미 손실 (실제 구현은 TPUTrainer.train_step 사용)
            loss = jnp.mean((x_t - batch_latents) ** 2)
            losses.append(float(loss))
            
            if (step + 1) % 100 == 0 and process_index == 0:
                print(f"Epoch {epoch+1}/{config.num_epochs} Step {step+1}/{config.steps_per_epoch} Loss: {np.mean(losses[-100:]):.6f}")
        
        if process_index == 0:
            epoch_time = time.time() - epoch_start
            print(f"✓ Epoch {epoch+1} completed in {epoch_time/60:.1f}m")
    
    if process_index == 0:
        total_time = time.time() - total_start
        print(f"\n✓ Training completed in {total_time/3600:.1f}h")
        wandb.finish()


if __name__ == "__main__":
    main()
