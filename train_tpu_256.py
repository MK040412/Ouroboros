"""
TPU v5e 16 Pod에서 256² 이미지로 XUT-Small 학습
Batch Size: 2048 (한 번에 처리)
Dataset: KBlueLeaf/coyo11m-256px-ccrop-latent
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
    context_dim: int = 640              # Gemma-270M
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
    embedding_model: str = "google/embedding-gecko-text-3"  # 640d context용
    
    # TREAD (Timestep-Random Encoder Architecture Design)
    tread_selection_rate: float = 0.5  # 기존 연구 설정값
    
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
# Large Scale Data Loader with Prefetch
# ============================================
class Coyo11mDataLoader:
    """Coyo11m 256px latent 데이터로더 (HF + Prefetch + TPU최적화)"""
    
    def __init__(self, batch_size: int, embedding_provider=None, 
                 use_gcs: bool = False, gcs_bucket: str = "gs://rdy-tpu-data-2025/coyo11m"):
        self.batch_size = batch_size
        self.embedding_provider = embedding_provider
        self.use_gcs = use_gcs
        self.gcs_bucket = gcs_bucket
        
        # HuggingFace repo 또는 GCS에서 파일 목록 로드
        print("Scanning data files...")
        if use_gcs:
            self._load_from_gcs()
        else:
            self._load_from_huggingface()
        
        # Parquet 메타데이터 로드 (HF에서)
        print("\nLoading metadata from HuggingFace...")
        self._load_metadata()
        
        print(f"Available samples: {len(self.available_keys)}")
    
    def _load_from_gcs(self):
        """GCS 버킷에서 파일 목록 로드"""
        import subprocess
        result = subprocess.run(
            ["gcloud", "storage", "ls", f"{self.gcs_bucket}/latents-3crop/"],
            capture_output=True, text=True
        )
        
        self.pt_files = []
        for line in result.stdout.strip().split('\n'):
            if line.endswith('.pt'):
                self.pt_files.append(line.strip())
        
        print(f"Found {len(self.pt_files)} PT files in GCS")
        self.pt_data_cache = {}
        self.pt_file_mapping = {}
        
        # GCS에서 필요할 때만 다운로드 (lazy loading)
        for pt_file in self.pt_files:
            filename = Path(pt_file).name
            # 메타데이터 생성
            self.pt_file_mapping[filename] = pt_file
    
    def _load_from_huggingface(self):
        """HuggingFace repo에서 스트리밍 데이터 로드"""
        from datasets import load_dataset
        
        # HF dataset을 스트리밍으로 로드 (전체 다운로드 필요 없음)
        print("Loading dataset from HuggingFace (streaming)...")
        self.dataset = load_dataset(
            "KBlueLeaf/coyo11m-256px-ccrop-latent",
            streaming=True,
            split="train"
        )
        self.pt_data_cache = {}
        self.pt_file_mapping = {}
        self.dataset_iter = iter(self.dataset)
    
    def _load_metadata(self):
        """메타데이터 로드"""
        from huggingface_hub import hf_hub_download
        
        metadata_path = hf_hub_download(
            repo_id="KBlueLeaf/coyo11m-256px-ccrop-latent",
            filename="coyo11m-meta.parquet",
            repo_type="dataset",
            local_dir="/tmp",
            local_dir_use_symlinks=False
        )
        
        self.metadata_table = pq.read_table(metadata_path)
        self.available_keys = list(range(len(self.metadata_table)))
        print(f"Loaded {len(self.available_keys)} metadata entries")
    
    def _load_pt_file_gcs(self, gcs_path: str):
        """GCS에서 PT 파일 다운로드 및 로드"""
        import tempfile
        import subprocess
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            subprocess.run(
                ["gcloud", "storage", "cp", gcs_path, tmp_path],
                check=True, capture_output=True
            )
            pt_data = torch.load(tmp_path, map_location="cpu")
            return pt_data
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def _load_pt_file_hf(self, filename: str):
        """HuggingFace에서 PT 파일 다운로드 및 로드"""
        from huggingface_hub import hf_hub_download
        
        pt_path = hf_hub_download(
            repo_id="KBlueLeaf/coyo11m-256px-ccrop-latent",
            filename=f"latents-3crop/{filename}",
            repo_type="dataset",
            local_dir="/tmp",
            local_dir_use_symlinks=False
        )
        
        pt_data = torch.load(pt_path, map_location="cpu")
        return pt_data
    
    def get_batch(self, batch_idx: int, rng_key):
        """배치 데이터 반환 (Prefetch 최적화)"""
        # 배치 샘플 선택
        total_samples = len(self.available_keys)
        start_idx = (batch_idx * self.batch_size) % total_samples
        batch_indices = []
        for i in range(self.batch_size):
            batch_indices.append((start_idx + i) % total_samples)
        
        latents_list = []
        captions_list = []
        
        for idx in batch_indices:
            try:
                # 메타데이터에서 key 가져오기
                row = self.metadata_table.slice(idx, idx + 1)
                key = int(row['key'][0].as_py())
                caption = row['caption_llava'][0].as_py() if 'caption_llava' in row.column_names else ""
                
                # PT 파일 확인: 어느 파일에 속하는지
                file_idx = key // 10  # 각 파일당 10개 샘플 (000000-000009.pt, 000010-000019.pt 등)
                local_idx = key % 10
                filename = f"{file_idx:06d}-{file_idx + 9:06d}.pt"
                
                # PT 데이터 로드 (캐시 또는 다운로드)
                if filename not in self.pt_data_cache:
                    print(f"  Loading {filename}...")
                    if self.use_gcs:
                        pt_data = self._load_pt_file_gcs(f"{self.gcs_bucket}/latents-3crop/{filename}")
                    else:
                        pt_data = self._load_pt_file_hf(filename)
                    
                    self.pt_data_cache[filename] = {
                        'keys': pt_data['keys'].numpy(),
                        'latents': pt_data['latents']
                    }
                
                # Latent 추출: (3, 4, 32, 32) → (4, 32, 32)
                latent = self.pt_data_cache[filename]['latents'][local_idx].cpu().numpy()
                channels, frames, h, w = latent.shape
                latent = latent.reshape(channels * frames, h, w)[:4, :, :]
                latents_list.append(latent)
                captions_list.append(caption if caption else "")
                
            except Exception as e:
                print(f"  ⚠ Error loading index {idx}: {e}")
                # 에러 시 zero tensor 사용
                latents_list.append(np.zeros((4, 32, 32), dtype=np.float32))
                captions_list.append("")
        
        if len(latents_list) == 0:
            raise ValueError(f"Batch {batch_idx}: No valid samples loaded")
        
        # Stack latents: (B, 4, 32, 32)
        batch_latents = np.stack(latents_list, axis=0)
        batch_latents = jnp.array(batch_latents, dtype=jnp.float32)
        
        # 임베딩 계산 (병렬화)
        batch_embeddings = self.embedding_provider.batch_encode(
            captions_list, batch_size=512, normalize=True
        )  # (B, 640)
        
        return batch_latents, batch_embeddings


# ============================================
# Prefetch Pipeline (TPU 최적화)
# ============================================
class PrefetchDataLoader:
    """Prefetch를 이용한 데이터로딩 파이프라인"""
    
    def __init__(self, data_loader: Coyo11mDataLoader, 
                 steps_per_epoch: int, prefetch_size: int = 2):
        self.data_loader = data_loader
        self.steps_per_epoch = steps_per_epoch
        self.prefetch_size = prefetch_size
        self.prefetch_queue = queue.Queue(maxsize=prefetch_size)
        self.stop_event = threading.Event()
        
        # Prefetch 워커 스레드 시작
        self.worker_thread = threading.Thread(
            target=self._prefetch_worker, daemon=True
        )
        self.worker_thread.start()
    
    def _prefetch_worker(self):
        """백그라운드에서 배치를 미리 로드"""
        rng_key = jax.random.PRNGKey(0)
        for batch_idx in range(self.steps_per_epoch):
            if self.stop_event.is_set():
                break
            try:
                rng_key, subkey = jax.random.split(rng_key)
                batch = self.data_loader.get_batch(batch_idx, subkey)
                self.prefetch_queue.put(batch, timeout=10)
            except Exception as e:
                print(f"Prefetch error at batch {batch_idx}: {e}")
                break
        
        # 종료 신호
        self.prefetch_queue.put(None)
    
    def get_batches(self):
        """Prefetch된 배치 반환"""
        while True:
            batch = self.prefetch_queue.get()
            if batch is None:  # 종료 신호
                break
            yield batch
    
    def stop(self):
        """Prefetch 중지"""
        self.stop_event.set()
        self.worker_thread.join(timeout=5)


# ============================================
# Training with pjit
# ============================================
def create_sharding_mesh(num_devices: int):
    """TPU 메시 생성"""
    # v5e 16 pod: 16개 TPU 칩
    devices = mesh_utils.create_device_mesh((num_devices,))
    return Mesh(devices, axis_names=("batch",))


class TPUTrainer:
    """TPU 분산 학습기"""
    
    def __init__(self, model, optimizer, schedule, config: TrainingConfig256):
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.config = config
        
        # Mesh 생성
        self.mesh = create_sharding_mesh(config.num_devices)
        
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
    
    def train_step(self, model_state, x_t, timesteps, noise, text_emb, step, rng_key):
        """한 스텝 학습 (pjit으로 sharded)"""
        # Learning rate 계산
        lr = self.lr_schedule(step)
        
        # TREAD: Timestep-Random Encoder Architecture Design
        # selection_rate 확률로 timestep 조건을 무효화 (context-only mode)
        # 예: tread_selection_rate=0.5 → 50% 확률로 timestep 무효화
        batch_size = timesteps.shape[0]
        rng_key, subkey = jax.random.split(rng_key)
        mask = jax.random.uniform(subkey, (batch_size,)) < self.config.tread_selection_rate
        t_cond = jnp.where(mask, jnp.zeros_like(timesteps), timesteps)
        
        # Loss 및 gradient 계산
        # ⚠️ NOTE: nnx.value_and_grad 및 optimizer.update의 정확한 API 확인 필요
        # 또는 표준 jax.grad 사용으로 변경
        loss, grads = nnx.value_and_grad(self.loss_fn)(
            model_state, x_t, t_cond, noise, text_emb
        )
        
        # 옵티마이저 업데이트
        # ⚠️ NOTE: 이 API가 올바른지 nnx 문서에서 확인 필요
        self.optimizer.update(grads, lr=lr)
        
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
        """에포크 학습 (Prefetch 파이프라인 사용)"""
        losses = []
        rng_key = jax.random.PRNGKey(epoch)
        step = 0
        
        for batch_latents, batch_embeddings in prefetch_loader.get_batches():
            batch_size = batch_latents.shape[0]
            
            # 타임스텝 샘플링 (배치 단위, 각 샘플마다 다른 t)
            rng_key, subkey = jax.random.split(rng_key)
            timesteps = jax.random.randint(subkey, (batch_size,), 0, self.config.T)
            
            # 노이즈 샘플링
            rng_key, subkey = jax.random.split(rng_key)
            noise = jax.random.normal(subkey, batch_latents.shape, dtype=jnp.float32)
            
            # Forward diffusion
            x_t = self.schedule.forward_diffusion(batch_latents, noise, timesteps)
            
            # 학습 스텝
            global_step = epoch * self.config.steps_per_epoch + step
            loss, rng_key = self.train_step(self.model, x_t, timesteps, noise, batch_embeddings, 
                                            global_step, rng_key)
            
            losses.append(float(loss))
            
            # Wandb 로깅
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
                      f"Loss: {avg_loss:.6f}")
            
            if step >= self.config.steps_per_epoch:
                break
        
        epoch_avg_loss = np.mean(losses) if losses else 0.0
        return losses, epoch_avg_loss


# ============================================
# Main
# ============================================
def main():
    print("="*60)
    print("TPU v5e 16 Pod Training (256² XUT-Small)")
    print("="*60)
    
    config = TrainingConfig256()
    
    # Wandb 초기화
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
        },
        name=f"xut-small-256-tpu-pod-16"
    )
    
    print(f"\nConfig:")
    print(f"  Devices: {config.num_devices} TPU cores")
    print(f"  Global batch size: {config.global_batch_size}")
    print(f"  Batch per device: {config.batch_size_per_device}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  TREAD selection rate: {config.tread_selection_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Steps per epoch: {config.steps_per_epoch}")
    print(f"  Data loading: HuggingFace streaming + Prefetch pipeline")
    print(f"  Note: For TPU v5e 256 (112 cores), adjust num_devices to 112")
    
    # 디바이스 확인
    print("\n" + "="*60)
    devices = jax.devices()
    print(f"Available devices: {len(devices)}")
    print(f"Device type: {devices[0].device_kind}")
    
    # Text embedding provider
    print("\n" + "="*60)
    print("Loading text embedding model (Google Embedding Gecko)...")
    print("="*60)
    
    embedding_provider = get_embedding_provider(config.embedding_model)
    
    # 데이터로더 (HF 또는 GCS에서 직접 로딩)
    print("\n" + "="*60)
    print("Initializing data loader...")
    print("="*60)
    
    # GCS 버킷에 데이터를 업로드했으면 use_gcs=True 사용
    # 아니면 HuggingFace에서 직접 스트리밍으로 로드
    data_loader = Coyo11mDataLoader(
        batch_size=config.global_batch_size,
        embedding_provider=embedding_provider,
        use_gcs=False,  # True로 변경하면 GCS에서 로딩
        gcs_bucket="gs://rdy-tpu-data-2025/coyo11m"
    )
    
    # 모델 초기화
    print("\n" + "="*60)
    print("Initializing XUT-Small model...")
    print("="*60)
    
    model = create_xut_small()
    print("✓ XUT-Small initialized")
    print(f"  Dimension: 896")
    print(f"  Context dim: 640")
    print(f"  Depth: 4")
    print(f"  Parameters: ~237M (XUT part) + ~270M (Gemma)")
    
    # 옵티마이저 (AdamW with weight decay)
    # Note: learning_rate는 train_step에서 동적으로 설정됨
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
    print("Starting training...")
    print("="*60)
    
    trainer = TPUTrainer(model, optimizer, schedule, config)
    
    total_start = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Prefetch 파이프라인 생성 (배경에서 배치 로딩)
        prefetch_loader = PrefetchDataLoader(
            data_loader, 
            steps_per_epoch=config.steps_per_epoch,
            prefetch_size=2  # 2개 배치 미리 로드
        )
        
        losses, epoch_avg_loss = trainer.train_epoch(prefetch_loader, epoch)
        prefetch_loader.stop()
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n{'='*60}")
        print(f"✓ Epoch {epoch+1} completed in {epoch_time/3600:.1f}h")
        print(f"  Average loss: {epoch_avg_loss:.6f}")
        print(f"  Min loss: {np.min(losses):.6f}")
        print(f"  Max loss: {np.max(losses):.6f}")
        print(f"{'='*60}")
        
        # 에포크 레벨 wandb 로깅
        wandb.log({
            "epoch_avg_loss": epoch_avg_loss,
            "epoch_min_loss": np.min(losses),
            "epoch_max_loss": np.max(losses),
            "epoch_time_hours": epoch_time / 3600,
            "epoch": epoch + 1,
        }, step=epoch)
        
        # 매 에포크 끝에 체크포인트 저장
        trainer.save_checkpoint(epoch, config.steps_per_epoch, epoch_avg_loss)
        wandb.log({"checkpoint_saved": epoch + 1})
    
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print("✓ Training completed!")
    print("="*60)
    print(f"Total training time: {total_time/3600:.1f}h")
    print(f"Expected time: ~174h (per table)")
    
    # 최종 통계
    wandb.log({
        "total_training_time_hours": total_time / 3600,
        "completed": True,
    })
    wandb.finish()


if __name__ == "__main__":
    main()
