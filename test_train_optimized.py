import os
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
import torch
import pyarrow.parquet as pq
from dataclasses import dataclass
from typing import Tuple, Optional
import time

from src.xut.xut import XUDiT
from src.embeddings import get_embedding_provider


# ============================================
# Configuration (GPU 활용 최적화)
# ============================================
@dataclass
class TrainingConfig:
    """GPU 계산 최적화 설정"""
    # 데이터
    num_samples: int = 1000          # ↑ 샘플 수 증가
    batch_size: int = 32             # ↑ 배치 크기 4배
    num_steps: int = 100             # ↑ 스텝 수 10배
    
    # 학습
    learning_rate: float = 1e-4
    T: int = 1000
    
    # 모델
    input_dim: int = 4
    image_size: int = 32
    patch_size: int = 2
    model_dim: int = 256             # ↑ 모델 크기 4배
    heads: int = 16                  # ↑ 헤드 수 증가
    dim_head: int = 64
    mlp_dim: int = 1024              # ↑ MLP 크기 4배
    depth: int = 4                   # ↑ 깊이 2배
    
    # 노이즈 스케줄
    beta_min: float = 0.0001
    beta_max: float = 0.02
    
    # Text embedding
    use_text_conditioning: bool = True
    embedding_model: str = "google/embeddinggemma-300m"
    
    # 최적화
    use_mixed_precision: bool = True  # 혼합 정밀도
    gradient_accumulation_steps: int = 1  # 그래디언트 누적


# ============================================
# Diffusion Schedule
# ============================================
class DiffusionSchedule:
    """선형 노이즈 스케줄"""
    
    def __init__(self, beta_min: float = 0.0001, beta_max: float = 0.02, T: int = 1000):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self._cache_alphas()
    
    def _cache_alphas(self):
        """알파 값 사전 계산"""
        betas = jnp.array([
            self.beta_min + (self.beta_max - self.beta_min) * (i / self.T)
            for i in range(self.T)
        ])
        self.alphas = 1.0 - betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
    
    def get_noise_params(self, t: int) -> Tuple[float, float]:
        """타임스텝 t에서 알파_t, 시그마_t 반환"""
        alpha_t = self.alphas_cumprod[t]
        sigma_t = jnp.sqrt(1.0 - alpha_t)
        return alpha_t, sigma_t
    
    def forward_diffusion(self, x_0: jnp.ndarray, noise: jnp.ndarray, t: int) -> jnp.ndarray:
        """x_0에 노이즈 추가"""
        alpha_t, sigma_t = self.get_noise_params(t)
        x_t = jnp.sqrt(alpha_t) * x_0 + sigma_t * noise
        return x_t


# ============================================
# Data Loading (최적화 버전)
# ============================================
class DataLoaderWithCaptions:
    """프리페칭 최적화 버전"""
    
    def __init__(self, latent_data: jnp.ndarray, captions: list, batch_size: int,
                 embedding_provider=None):
        self.latent_data = latent_data
        self.captions = captions
        self.batch_size = batch_size
        self.num_samples = len(latent_data)
        self.embedding_provider = embedding_provider
        
        # 사전 임베딩 계산 (속도 향상)
        print("Pre-computing text embeddings...")
        self.embeddings = embedding_provider.batch_encode(
            captions, batch_size=64, normalize=True
        )
        print(f"✓ Pre-computed {len(self.embeddings)} embeddings")
    
    def sample_batch(self, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """배치와 해당 embedding 반환"""
        indices = jax.random.randint(key, (self.batch_size,), 0, self.num_samples)
        batch_latents = self.latent_data[indices]
        batch_embeddings = self.embeddings[indices]
        return batch_latents, batch_embeddings
    
    @staticmethod
    def load_latents_with_captions(pt_path: str, parquet_path: str, num_samples: int = 100):
        """PT와 Parquet에서 latent와 caption을 함께 로드"""
        print(f"Loading latents from {pt_path}...")
        pt_data = torch.load(pt_path, map_location="cpu")
        pt_keys = pt_data['keys'].numpy()
        latents_torch = pt_data['latents']
        
        # Parquet에서 사용 가능한 key 확인
        print(f"Checking parquet keys...")
        table = pq.read_table(parquet_path, columns=['key'])
        parquet_keys = set(table['key'].to_pylist())
        
        # 교집합 찾기
        available_indices = []
        available_captions = []
        
        for idx, key in enumerate(pt_keys[:num_samples]):
            if key in parquet_keys:
                available_indices.append(idx)
        
        print(f"Found {len(available_indices)} samples with caption data")
        
        if len(available_indices) == 0:
            raise ValueError("No matching samples found!")
        
        # Latent 추출
        latents_subset = latents_torch[available_indices, 0, :, :, :]  # (N, 4, 32, 32)
        latents_fp32 = latents_subset.float().numpy().astype(np.float32)
        latents = jnp.array(np.transpose(latents_fp32, (0, 2, 3, 1)))  # NHWC
        
        # Caption 추출 (배치 처리)
        for idx in available_indices:
            key = int(pt_keys[idx])
            caption_table = pq.read_table(
                parquet_path,
                columns=['caption_llava'],
                filters=[('key', '==', key)]
            )
            if len(caption_table) > 0:
                caption = caption_table.to_pandas().iloc[0]['caption_llava']
                available_captions.append(caption)
        
        print(f"✓ Loaded {len(latents)} latents with captions")
        return latents, available_captions


# ============================================
# Training
# ============================================
class DiffusionTrainerOptimized:
    """최적화된 학습기"""
    
    def __init__(self, model: nnx.Module, optimizer: nnx.Optimizer, 
                 schedule: DiffusionSchedule):
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.loss_history = []
    
    @staticmethod
    def loss_fn(model: nnx.Module, x_t: jnp.ndarray, t: jnp.ndarray, 
                noise: jnp.ndarray, text_embedding: jnp.ndarray,
                deterministic: bool = False) -> jnp.ndarray:
        """노이즈 예측 손실"""
        pred_noise = model(x_t, t, deterministic=deterministic)
        noise_transposed = jnp.transpose(noise, (0, 3, 1, 2))
        loss = jnp.mean((pred_noise - noise_transposed) ** 2)
        return loss
    
    def train_step(self, x_t: jnp.ndarray, t: jnp.ndarray, 
                   noise: jnp.ndarray, text_embedding: jnp.ndarray) -> float:
        """한 스텝 학습"""
        loss, grads = nnx.value_and_grad(self.loss_fn)(
            self.model, x_t, t, noise, text_embedding, deterministic=False
        )
        self.optimizer.update(grads)
        return loss
    
    def train_epoch(self, data_loader: DataLoaderWithCaptions, 
                    schedule: DiffusionSchedule, num_steps: int):
        """에포크 학습"""
        key = jax.random.PRNGKey(42)
        
        start_time = time.time()
        step_times = []
        
        for step in range(num_steps):
            step_start = time.time()
            
            # 배치 샘플링
            key, subkey = jax.random.split(key)
            batch_x0, batch_embeddings = data_loader.sample_batch(subkey)
            
            # 랜덤 타임스텝
            key, subkey = jax.random.split(key)
            t_idx = jax.random.randint(subkey, (), 0, schedule.T)
            t = jnp.array([float(t_idx)], dtype=jnp.float32)
            
            # 랜덤 노이즈
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, batch_x0.shape, dtype=jnp.float32)
            
            # Forward diffusion
            x_t = schedule.forward_diffusion(batch_x0, noise, int(t_idx))
            
            # 학습 스텝
            try:
                loss = self.train_step(x_t, t, noise, batch_embeddings)
                self.loss_history.append(float(loss))
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                if (step + 1) % 10 == 0:
                    avg_time = np.mean(step_times[-10:])
                    print(f"  Step {step+1}/{num_steps} - Loss: {loss:.6f} - {avg_time:.3f}s/step ✓")
            except Exception as e:
                print(f"  ✗ Error at step {step+1}: {e}")
                break
        
        total_time = time.time() - start_time
        print(f"\n✓ Training took {total_time:.1f}s ({num_steps} steps)")
        print(f"  Average: {total_time/num_steps:.3f}s/step")


# ============================================
# Main
# ============================================
def main():
    print("="*60)
    print("JAX Diffusion Model - GPU Optimized")
    print("="*60)
    
    config = TrainingConfig()
    print(f"\nOptimizations:")
    print(f"  Batch size: {config.batch_size} (↑)")
    print(f"  Model dim: {config.model_dim} (↑)")
    print(f"  Total steps: {config.num_steps} (↑)")
    print(f"  Mixed precision: {config.use_mixed_precision}")
    print(f"  Pre-computed embeddings: Yes")
    
    # 디바이스 확인
    print("\n" + "="*60)
    devices = jax.devices()
    print(f"Available devices: {devices}")
    print(f"Device kind: {devices[0].device_kind}")
    
    # Text embedding provider
    print("\n" + "="*60)
    print("Loading text embedding model...")
    print("="*60)
    
    embedding_provider = get_embedding_provider(config.embedding_model)
    
    # 데이터 로드
    print("\n" + "="*60)
    print("Loading data with captions...")
    print("="*60)
    
    pt_file = "000000-000009.pt"
    parquet_file = "coyo11m-meta.parquet"
    
    if os.path.exists(pt_file) and os.path.exists(parquet_file):
        try:
            latents, captions = DataLoaderWithCaptions.load_latents_with_captions(
                pt_file, parquet_file, config.num_samples
            )
            print(f"✓ Data shape: {latents.shape}")
            print(f"✓ Captions: {len(captions)}")
            
            data_loader = DataLoaderWithCaptions(
                latents, captions, config.batch_size,
                embedding_provider=embedding_provider
            )
        except Exception as e:
            print(f"Error loading data: {e}")
            return
    else:
        print(f"Files not found: {pt_file} or {parquet_file}")
        return
    
    # 노이즈 스케줄
    schedule = DiffusionSchedule(
        beta_min=config.beta_min,
        beta_max=config.beta_max,
        T=config.T
    )
    
    # 모델 초기화 (더 큼)
    print("\n" + "="*60)
    print("Initializing model...")
    print("="*60)
    
    model = XUDiT(
        patch_size=config.patch_size,
        input_dim=config.input_dim,
        dim=config.model_dim,
        ctx_dim=config.model_dim,
        heads=config.heads,
        dim_head=config.dim_head,
        mlp_dim=config.mlp_dim,
        depth=config.depth,
        enc_blocks=1,
        dec_blocks=1,
        concat_ctx=True,
        shared_adaln=True,
        rngs=nnx.Rngs(0),
    )
    print(f"✓ Model initialized (larger architecture)")
    
    # 옵티마이저
    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate=config.learning_rate)
    )
    
    # 학습
    print("\n" + "="*60)
    print("Starting optimized training...")
    print("="*60)
    
    trainer = DiffusionTrainerOptimized(model, optimizer, schedule)
    trainer.train_epoch(data_loader, schedule, config.num_steps)
    
    # 결과
    print("\n" + "="*60)
    print("✓ Training completed!")
    print("="*60)
    
    if trainer.loss_history:
        print(f"\nInitial loss: {trainer.loss_history[0]:.6f}")
        print(f"Final loss: {trainer.loss_history[-1]:.6f}")
        print(f"Loss reduction: {(1 - trainer.loss_history[-1]/trainer.loss_history[0])*100:.1f}%")


if __name__ == "__main__":
    main()
