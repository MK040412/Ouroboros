"""
TPU v5e 256 (112 cores)에서 256² 이미지로 XUT-Small 학습
Global Batch Size: 7168 (112 cores × 64 per device)
Dataset: HuggingFace streaming (KBlueLeaf/coyo11m-256px-ccrop-latent)
"""

# 기본 train_tpu_256.py를 import한 후 설정 수정
import sys
sys.path.insert(0, '/home/perelman/jax-hdm')

from train_tpu_256 import (
    TrainingConfig256, TPUTrainer, DiffusionSchedule, 
    Coyo11mDataLoader, PrefetchDataLoader, create_sharding_mesh,
    create_xut_small, get_embedding_provider
)
import jax
import optax
from flax import nnx
import time
import wandb
import numpy as np


def main():
    print("="*60)
    print("TPU v5e 256 Training (112 cores, 256² XUT-Small)")
    print("="*60)
    
    # 설정: 112 cores 최적화
    config = TrainingConfig256()
    config.num_devices = 112  # TPU v5e 256
    config.global_batch_size = 7168  # 112 × 64
    config.batch_size_per_device = 64
    config.context_dim = 768  # Embedding Gemma 300M
    config.num_epochs = 20
    config.steps_per_epoch = 1065  # 7.624M / 7168 ≈ 1065
    
    # Wandb 초기화
    wandb.init(
        project="xut-small-256-tpu-v5e-256",
        entity=None,
        config={
            "num_devices": config.num_devices,
            "global_batch_size": config.global_batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "tread_selection_rate": config.tread_selection_rate,
        },
        name=f"xut-small-256-tpu-v5e-256-112cores"
    )
    
    print(f"\nConfig (112 cores optimized):")
    print(f"  Devices: {config.num_devices} TPU cores (v5e 256)")
    print(f"  Global batch size: {config.global_batch_size}")
    print(f"  Batch per device: {config.batch_size_per_device}")
    print(f"  Steps per epoch: {config.steps_per_epoch}")
    print(f"  Data loading: HuggingFace streaming + Prefetch")
    
    # 디바이스 확인
    print("\n" + "="*60)
    devices = jax.devices()
    print(f"Available devices: {len(devices)}")
    print(f"Device type: {devices[0].device_kind if devices else 'None'}")
    
    # Text embedding provider
    print("\n" + "="*60)
    print("Loading text embedding model...")
    print("="*60)
    embedding_provider = get_embedding_provider(config.embedding_model)
    
    # 데이터로더 (HuggingFace streaming)
    print("\n" + "="*60)
    print("Initializing data loader (HF streaming)...")
    print("="*60)
    data_loader = Coyo11mDataLoader(
        batch_size=config.global_batch_size,
        embedding_provider=embedding_provider,
        use_gcs=False  # HuggingFace 스트리밍 사용
    )
    
    # 모델 초기화
    print("\n" + "="*60)
    print("Initializing XUT-Small model...")
    print("="*60)
    model = create_xut_small()
    print("✓ XUT-Small initialized")
    print(f"  Dimension: 896")
    print(f"  Context dim: 640")
    print(f"  Parameters: ~237M (XUT) + ~270M (Gemma)")
    
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
    
    # 학습기
    print("\n" + "="*60)
    print("Starting training on TPU v5e 256 (112 cores)...")
    print("="*60)
    
    trainer = TPUTrainer(model, optimizer, schedule, config)
    
    total_start = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Prefetch 파이프라인 (112 workers)
        prefetch_loader = PrefetchDataLoader(
            data_loader,
            steps_per_epoch=config.steps_per_epoch,
            num_workers=112  # 112개 worker로 병렬 로딩
        )
        
        losses, epoch_avg_loss = trainer.train_epoch(prefetch_loader, epoch)
        prefetch_loader.stop()
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n{'='*60}")
        print(f"✓ Epoch {epoch+1} completed in {epoch_time/3600:.1f}h")
        print(f"  Average loss: {epoch_avg_loss:.6f}")
        if losses:
            print(f"  Min loss: {np.min(losses):.6f}")
            print(f"  Max loss: {np.max(losses):.6f}")
        print(f"{'='*60}")
        
        wandb.log({
            "epoch_avg_loss": epoch_avg_loss,
            "epoch_time_hours": epoch_time / 3600,
            "epoch": epoch + 1,
        }, step=epoch)
        
        trainer.save_checkpoint(epoch, config.steps_per_epoch, epoch_avg_loss)
    
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
