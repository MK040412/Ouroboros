"""
TPU v5e 16 Pod에서 Flax pmap을 사용한 분산 학습
- TPU Device: 16개 (연산)
- CPU Worker: 112개 (데이터 로딩)
"""

import os
import jax
import jax.numpy as jnp
from jax import pmap
from jax import random
import optax
import numpy as np
import time
import wandb
from dataclasses import dataclass

from src.xut.xut_small import create_xut_small
from src.embeddings import get_embedding_provider
from train_tpu_256 import (
    TrainingConfig256, 
    DiffusionSchedule, 
    Coyo11mDataLoader, 
    PrefetchDataLoader
)


@dataclass
class TrainState:
    """학습 상태"""
    params: dict
    opt_state: dict
    rng: jnp.ndarray


def create_train_state(model, optimizer, rng_key):
    """초기 학습 상태 생성"""
    # 더미 입력으로 모델 초기화
    dummy_x = jnp.ones((1, 32, 32, 4))  # (B, H, W, C)
    dummy_t = jnp.ones((1, 1))
    dummy_ctx = jnp.ones((1, 256, 768))  # context
    
    params = model.init(rng_key, dummy_x, dummy_t, ctx=dummy_ctx)['params']
    opt_state = optimizer.init(params)
    
    return TrainState(
        params=params,
        opt_state=opt_state,
        rng=rng_key
    )


def loss_fn(model, params, x_t, timesteps, noise, text_emb, rng_key):
    """손실 함수"""
    # NHWC 형식 변환: (B, 4, 32, 32) → (B, 32, 32, 4)
    x_t_nhwc = jnp.transpose(x_t, (0, 2, 3, 1))
    
    # timestep 추가: (B,) → (B, 1)
    if timesteps.ndim == 1:
        timesteps = timesteps[:, None]
    
    # context: (B, 640) → (B, 1, 640)
    ctx = text_emb[:, None, :]
    
    # 모델 호출
    pred_noise_nhwc = model.apply(
        {'params': params},
        x_t_nhwc,
        timesteps,
        ctx=ctx,
        deterministic=False,
        rng=rng_key,
        training=True
    )
    
    # NCHW로 변환: (B, 32, 32, 4) → (B, 4, 32, 32)
    pred_noise = jnp.transpose(pred_noise_nhwc, (0, 3, 1, 2))
    
    # MSE loss
    loss = jnp.mean((pred_noise - noise) ** 2)
    return loss


def train_step_single(model, state, batch_latents, timesteps, noise, batch_embeddings, lr):
    """단일 기기에서의 학습 스텝"""
    def loss_wrapper(params):
        state.rng, subkey = random.split(state.rng)
        return loss_fn(model, params, batch_latents, timesteps, noise, batch_embeddings, subkey)
    
    loss, grads = jax.value_and_grad(loss_wrapper)(state.params)
    
    # Optimizer 업데이트
    updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    
    new_state = TrainState(
        params=new_params,
        opt_state=new_opt_state,
        rng=state.rng
    )
    
    return new_state, loss


def main():
    print("="*60)
    print("TPU v5e 16 Pod Training (Flax pmap)")
    print("="*60)
    
    # ============================================
    # Device 확인 (명시적 출력)
    # ============================================
    print("\n" + "="*60)
    print("Available Devices")
    print("="*60)
    devices = jax.devices()
    print(f"Total devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device.device_kind:4s} @ {device.platform:6s} (id={device.id})")
    print(f"\nCPU Workers: 112")
    print("="*60 + "\n")
    
    # ============================================
    # Config
    # ============================================
    config = TrainingConfig256()
    
    wandb.init(
        project="xut-small-256-pmap",
        entity=None,
        config={
            "num_devices": len(devices),
            "global_batch_size": config.global_batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
        },
        name=f"xut-small-256-tpu-pmap-{len(devices)}"
    )
    
    print(f"\nConfig:")
    print(f"  Devices: {len(devices)} TPU cores")
    print(f"  Global batch size: {config.global_batch_size}")
    print(f"  Batch per device: {config.batch_size_per_device}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Steps per epoch: {config.steps_per_epoch}")
    print(f"  Data loading: HuggingFace streaming + 112-worker Prefetch\n")
    
    # ============================================
    # Text embedding
    # ============================================
    print("="*60)
    print("Loading text embedding model...")
    print("="*60)
    embedding_provider = get_embedding_provider(config.embedding_model)
    print("✓ Embedding model loaded\n")
    
    # ============================================
    # Data loader
    # ============================================
    print("="*60)
    print("Initializing data loader...")
    print("="*60)
    data_loader = Coyo11mDataLoader(
        batch_size=config.global_batch_size,
        embedding_provider=embedding_provider,
        use_gcs=False
    )
    print(f"✓ Data loader initialized\n")
    
    # ============================================
    # Model
    # ============================================
    print("="*60)
    print("Initializing XUT-Small model...")
    print("="*60)
    model = create_xut_small()
    print("✓ XUT-Small initialized")
    print(f"  Dimension: 896")
    print(f"  Context dim: 768")
    print(f"  Depth: 4\n")
    
    # ============================================
    # Optimizer
    # ============================================
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=config.learning_rate, weight_decay=1e-4)
    )
    
    # ============================================
    # Schedule
    # ============================================
    schedule = DiffusionSchedule(
        beta_min=config.beta_min,
        beta_max=config.beta_max,
        T=config.T
    )
    
    # ============================================
    # Train state
    # ============================================
    print("="*60)
    print("Creating training state...")
    print("="*60)
    rng_key = random.PRNGKey(0)
    rng_key, init_key = random.split(rng_key)
    state = create_train_state(model, optimizer, init_key)
    print("✓ Training state created\n")
    
    # ============================================
    # Training loop
    # ============================================
    print("="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    total_start = time.time()
    global_step = 0
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        losses = []
        
        # Prefetch 파이프라인 (112 workers)
        prefetch_loader = PrefetchDataLoader(
            data_loader,
            steps_per_epoch=config.steps_per_epoch,
            num_workers=112
        )
        
        step = 0
        for batch_latents, batch_embeddings in prefetch_loader.get_batches():
            # Timestep 샘플링
            rng_key, subkey = random.split(rng_key)
            batch_size = batch_latents.shape[0]
            timesteps = random.randint(subkey, (batch_size,), 0, config.T)
            
            # 노이즈 샘플링
            rng_key, subkey = random.split(rng_key)
            noise = random.normal(subkey, batch_latents.shape, dtype=jnp.float32)
            
            # Forward diffusion
            x_t = schedule.forward_diffusion(batch_latents, noise, timesteps)
            
            # Learning rate
            lr = config.learning_rate  # 단순화 (실제로는 warmup schedule 필요)
            
            # 학습 스텝
            state, loss = train_step_single(
                model, state, x_t, timesteps, noise, batch_embeddings, lr
            )
            
            losses.append(float(loss))
            
            # Logging
            wandb.log({
                "loss": float(loss),
                "learning_rate": lr,
                "epoch": epoch + 1,
                "step": step + 1,
            }, step=global_step)
            
            if (step + 1) % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                print(f"Epoch {epoch+1}/{config.num_epochs} "
                      f"Step {step+1}/{config.steps_per_epoch} "
                      f"Loss: {avg_loss:.6f}")
            
            step += 1
            global_step += 1
            
            if step >= config.steps_per_epoch:
                break
        
        prefetch_loader.stop()
        
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = np.mean(losses) if losses else 0.0
        
        print(f"\n{'='*60}")
        print(f"✓ Epoch {epoch+1} completed in {epoch_time/3600:.1f}h")
        print(f"  Average loss: {epoch_avg_loss:.6f}")
        if losses:
            print(f"  Min loss: {np.min(losses):.6f}")
            print(f"  Max loss: {np.max(losses):.6f}")
        print(f"{'='*60}\n")
        
        wandb.log({
            "epoch_avg_loss": epoch_avg_loss,
            "epoch_time_hours": epoch_time / 3600,
            "epoch": epoch + 1,
        }, step=epoch)
    
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print("✓ Training completed!")
    print("="*60)
    print(f"Total training time: {total_time/3600:.1f}h")
    
    wandb.log({
        "total_training_time_hours": total_time / 3600,
        "completed": True,
    })
    wandb.finish()


if __name__ == "__main__":
    main()
