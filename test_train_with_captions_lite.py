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

from src.xut.xut import XUDiT
from src.embeddings import get_embedding_provider


# ============================================
# Configuration (더 빠른 로드를 위해 경량 모델 사용)
# ============================================
@dataclass
class TrainingConfig:
    """LLaVA caption이 포함된 학습 설정"""
    # 데이터
    num_samples: int = 100
    batch_size: int = 8
    num_steps: int = 10
    
    # 학습
    learning_rate: float = 1e-4
    T: int = 1000  # 노이즈 스케줄 타임스텝
    
    # 모델
    input_dim: int = 4
    image_size: int = 32
    patch_size: int = 2
    model_dim: int = 64
    heads: int = 8
    dim_head: int = 64
    mlp_dim: int = 256
    depth: int = 2
    
    # 노이즈 스케줄
    beta_min: float = 0.0001
    beta_max: float = 0.02
    
    # Text embedding
    use_text_conditioning: bool = True
    embedding_model: str = "google/embeddinggemma-300m"


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
        """x_0에 노이즈 추가: x_t = √α_t · x_0 + σ_t · ε"""
        alpha_t, sigma_t = self.get_noise_params(t)
        x_t = jnp.sqrt(alpha_t) * x_0 + sigma_t * noise
        return x_t


# ============================================
# Data Loading with Captions
# ============================================
class DataLoaderWithCaptions:
    """Parquet 메타데이터와 함께 latent 데이터 로딩"""
    
    def __init__(self, latent_data: jnp.ndarray, captions: list, batch_size: int):
        self.latent_data = latent_data
        self.captions = captions
        self.batch_size = batch_size
        self.num_samples = len(latent_data)
        assert len(captions) == self.num_samples, "Caption 수가 데이터 수와 일치하지 않음"
    
    def sample_batch(self, key) -> Tuple[jnp.ndarray, list]:
        """배치와 해당 caption 반환"""
        indices = jax.random.randint(key, (self.batch_size,), 0, self.num_samples)
        batch_latents = self.latent_data[indices]
        batch_captions = [self.captions[i] for i in indices]
        return batch_latents, batch_captions
    
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
        
        # Caption 추출
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
# Training with Text Conditioning
# ============================================
class DiffusionTrainerWithText:
    """Text conditioning을 포함한 확산 모델 학습기"""
    
    def __init__(self, model: nnx.Module, optimizer: nnx.Optimizer, 
                 schedule: DiffusionSchedule, embedding_provider):
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.embedding_provider = embedding_provider
    
    @staticmethod
    def loss_fn(model: nnx.Module, x_t: jnp.ndarray, t: jnp.ndarray, 
                noise: jnp.ndarray, text_embedding: jnp.ndarray,
                deterministic: bool = False) -> jnp.ndarray:
        """
        Text conditioning이 있는 노이즈 예측 손실
        """
        pred_noise = model(x_t, t, deterministic=deterministic)  # (B, C, H, W)
        noise_transposed = jnp.transpose(noise, (0, 3, 1, 2))    # (B, H, W, C) -> (B, C, H, W)
        loss = jnp.mean((pred_noise - noise_transposed) ** 2)
        return loss
    
    def train_step(self, x_t: jnp.ndarray, t: jnp.ndarray, 
                   noise: jnp.ndarray, text_embedding: jnp.ndarray) -> float:
        """한 스텝 학습 (text embedding 포함)"""
        loss, grads = nnx.value_and_grad(self.loss_fn)(
            self.model, x_t, t, noise, text_embedding, deterministic=False
        )
        self.optimizer.update(grads)
        return loss
    
    def train_epoch(self, data_loader: DataLoaderWithCaptions, 
                    schedule: DiffusionSchedule, num_steps: int):
        """한 에포크 학습"""
        key = jax.random.PRNGKey(42)
        losses = []
        
        for step in range(num_steps):
            # 배치 샘플링
            key, subkey = jax.random.split(key)
            batch_x0, batch_captions = data_loader.sample_batch(subkey)
            
            # Text embedding
            text_embeddings = self.embedding_provider.encode(batch_captions)
            
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
                loss = self.train_step(x_t, t, noise, text_embeddings)
                losses.append(float(loss))
                print(f"  Step {step+1}/{num_steps} - Loss: {loss:.6f} ✓")
            except Exception as e:
                print(f"  ✗ Error at step {step+1}: {e}")
                break
        
        return losses


# ============================================
# Main
# ============================================
def main():
    print("="*60)
    print("JAX Diffusion Model with LLaVA Captions")
    print("="*60)
    
    # 설정
    config = TrainingConfig()
    print(f"\nConfig:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Steps: {config.num_steps}")
    print(f"  Text conditioning: {config.use_text_conditioning}")
    print(f"  Embedding model: {config.embedding_model}")
    
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
            print(f"\nSample captions (first 3):")
            for i, cap in enumerate(captions[:3]):
                print(f"  [{i+1}] {cap[:100]}...")
            
            data_loader = DataLoaderWithCaptions(latents, captions, config.batch_size)
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
    
    # 모델 초기화
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
    print(f"✓ Model initialized")
    
    # 옵티마이저
    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate=config.learning_rate)
    )
    
    # 학습
    print("\n" + "="*60)
    print("Starting training with text conditioning...")
    print("="*60)
    
    trainer = DiffusionTrainerWithText(model, optimizer, schedule, embedding_provider)
    losses = trainer.train_epoch(data_loader, schedule, config.num_steps)
    
    # 결과
    print("\n" + "="*60)
    print("✓ Training completed!")
    print("="*60)
    print(f"\nFinal losses: {losses}")
    if losses:
        print(f"Final loss: {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
