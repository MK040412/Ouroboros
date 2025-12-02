import os
# TPU 자동 감지: 설정하지 않으면 자동으로 사용 가능한 장치 사용
# os.environ['JAX_PLATFORMS'] = 'tpu'  # TPU 명시적 설정

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from src.xut.xut import XUDiT
from src.embeddings import get_embedding_provider


# ============================================
# Configuration
# ============================================
@dataclass
class TrainingConfig:
    """학습 설정 (TPU 최적화)"""
    # TPU 배치 크기는 8의 배수로 (TPU는 8개 코어)
    num_samples: int = 100
    batch_size: int = 8
    num_steps: int = 10
    learning_rate: float = 1e-4
    T: int = 1000  # 노이즈 스케줄 타임스텝
    
    # 모델 설정 (TPU 메모리 고려)
    input_dim: int = 4
    image_size: int = 32
    patch_size: int = 2
    model_dim: int = 64      # TPU는 더 큰 모델 지원
    heads: int = 8
    dim_head: int = 64
    mlp_dim: int = 256
    depth: int = 2
    
    # 노이즈 스케줄
    beta_min: float = 0.0001
    beta_max: float = 0.02
    
    # TPU 설정
    data_path: Optional[str] = None  # .npz 파일 경로 (없으면 더미 데이터)
    use_pmap: bool = False           # 다중 TPU 사용 시 True


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
# Data Loading
# ============================================
class DataLoader:
    """데이터 로딩 및 배치 샘플링 (TPU 최적화)"""
    
    def __init__(self, data: jnp.ndarray, batch_size: int):
        self.data = data
        self.batch_size = batch_size
        self.num_samples = data.shape[0]
        # TPU 호환: batch_size가 8의 배수 확인
        assert self.batch_size % 8 == 0, "TPU: batch_size must be multiple of 8"
    
    def sample_batch(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """랜덤 배치 샘플링"""
        indices = jax.random.randint(key, (self.batch_size,), 0, self.num_samples)
        return self.data[indices]
    
    @staticmethod
    def load_data_from_pt(pt_path: str, num_samples: int = 100) -> jnp.ndarray:
        """PT 파일에서 데이터 로드
        
        원본 형식: (119829, 3, 4, 32, 32) = (B, T, C, H, W)
        - 첫 num_samples개만
        - 첫 프레임만 선택 [:, 0, :, :, :]
        - (B, C, H, W) → (B, H, W, C) NHWC로 변환
        """
        try:
            import torch
            print(f"Loading from {pt_path}...")
            data_dict = torch.load(pt_path, map_location="cpu")
            latents_torch = data_dict['latents']
            print(f"  Original shape: {latents_torch.shape} = (B, T, C, H, W)")
            
            # [:num_samples, 0, :, :, :] -> (num_samples, 4, 32, 32)
            latents_subset = latents_torch[:num_samples, 0, :, :, :]
            print(f"  After subset: {latents_subset.shape} = (B, C, H, W)")
            
            # bfloat16 -> float32
            latents_fp32 = latents_subset.float().numpy().astype(np.float32)
            print(f"  After float32: {latents_fp32.dtype}")
            
            # (B, C, H, W) -> (B, H, W, C) NHWC
            latents = jnp.array(np.transpose(latents_fp32, (0, 2, 3, 1)))
            print(f"  Final shape: {latents.shape} (NHWC)")
            return latents
            
        except Exception as e:
            print(f"Error loading PT file: {e}")
            raise
    
    @staticmethod
    def load_data_from_npz(data_path: str, num_samples: int = 100) -> jnp.ndarray:
        """NPZ 파일에서 데이터 로드 (TPU 호환)"""
        data_dict = np.load(data_path)
        # 첫 num_samples개만 로드
        latents = jnp.array(data_dict['latents'][:num_samples], dtype=jnp.float32)
        print(f"Loaded data from {data_path}: {latents.shape}")
        return latents
    
    @staticmethod
    def load_dummy_data(num_samples: int, input_dim: int, image_size: int) -> jnp.ndarray:
        """더미 데이터 생성 (NHWC 형식)"""
        print(f"Generating dummy data: ({num_samples}, {image_size}, {image_size}, {input_dim})")
        return jax.random.normal(
            jax.random.PRNGKey(42),
            (num_samples, image_size, image_size, input_dim),
            dtype=jnp.float32
        )


# ============================================
# Loss & Training
# ============================================
class DiffusionTrainer:
    """확산 모델 학습기"""
    
    def __init__(self, model: nnx.Module, optimizer: nnx.Optimizer, schedule: DiffusionSchedule):
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
    
    @staticmethod
    def loss_fn(model: nnx.Module, x_t: jnp.ndarray, t: jnp.ndarray, 
                noise: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        """
        노이즈 예측 손실
        - x_t: (B, H, W, C) NHWC
        - noise: (B, H, W, C) NHWC
        - model output: (B, C, H, W) NCHW
        """
        pred_noise = model(x_t, t, deterministic=deterministic)  # (B, C, H, W)
        noise_transposed = jnp.transpose(noise, (0, 3, 1, 2))   # (B, H, W, C) -> (B, C, H, W)
        loss = jnp.mean((pred_noise - noise_transposed) ** 2)
        return loss
    
    def train_step(self, x_t: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray) -> float:
        """한 스텝 학습"""
        loss, grads = nnx.value_and_grad(self.loss_fn)(
            self.model, x_t, t, noise, deterministic=False
        )
        self.optimizer.update(grads)
        return loss
    
    def train_epoch(self, data_loader: DataLoader, schedule: DiffusionSchedule, num_steps: int):
        """한 에포크 학습"""
        key = jax.random.PRNGKey(42)
        losses = []
        
        for step in range(num_steps):
            # 배치 샘플링
            key, subkey = jax.random.split(key)
            batch_x0 = data_loader.sample_batch(subkey)
            
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
                loss = self.train_step(x_t, t, noise)
                losses.append(float(loss))
                print(f"  Step {step+1}/{num_steps} - Loss: {loss:.6f} ✓")
            except Exception as e:
                print(f"  ✗ Error at step {step+1}: {e}")
                break
        
        return losses


# ============================================
# Device Check
# ============================================
def check_device():
    """사용 가능한 디바이스 확인"""
    devices = jax.devices()
    print(f"Available devices: {devices}")
    if devices:
        print(f"Default device: {devices[0]}")
        if 'gpu' in str(devices[0]).lower():
            print("✓ GPU detected")
        elif 'tpu' in str(devices[0]).lower():
            print("✓ TPU detected")
        else:
            print("⚠ CPU only")


# ============================================
# Main
# ============================================
def main():
    print("="*60)
    print("JAX/Flax Diffusion Model Training")
    print("="*60)
    
    # 설정
    config = TrainingConfig()
    print(f"\nConfig:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Steps: {config.num_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # 디바이스 확인
    print("\n" + "="*60)
    check_device()
    
    # 데이터 로딩
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)
    
    # 데이터 로드 우선순위: PT > NPZ > 더미 데이터
    pt_file = "000000-000009.pt"
    if os.path.exists(pt_file):
        latents = DataLoader.load_data_from_pt(pt_file, config.num_samples)
    elif config.data_path and os.path.exists(config.data_path):
        latents = DataLoader.load_data_from_npz(config.data_path, config.num_samples)
    else:
        latents = DataLoader.load_dummy_data(
            config.num_samples,
            config.input_dim,
            config.image_size
        )
    print(f"✓ Data shape: {latents.shape} (NHWC)")
    
    data_loader = DataLoader(latents, config.batch_size)
    
    # 노이즈 스케줄
    schedule = DiffusionSchedule(
        beta_min=config.beta_min,
        beta_max=config.beta_max,
        T=config.T
    )
    
    # 모델 초기화
    print("\n" + "="*60)
    print("Initializing model (TPU optimized)...")
    print("="*60)
    
    model = XUDiT(
        patch_size=config.patch_size,
        input_dim=config.input_dim,
        dim=config.model_dim,
        ctx_dim=config.model_dim,  # TPU: ctx_dim = model_dim
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
    print(f"  Model dim: {config.model_dim}")
    print(f"  Heads: {config.heads}")
    print(f"  Depth: {config.depth}")
    
    # 옵티마이저
    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate=config.learning_rate)
    )
    
    # 학습
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    trainer = DiffusionTrainer(model, optimizer, schedule)
    losses = trainer.train_epoch(data_loader, schedule, config.num_steps)
    
    # 결과
    print("\n" + "="*60)
    print("✓ Training completed!")
    print("="*60)
    print(f"\nFinal losses: {losses}")
    print(f"Final loss: {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
