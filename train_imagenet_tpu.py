"""
TPU v5e 32 Pod에서 ImageNet으로 XUT-Small 학습

Features:
- ImageNet-1K dataset from local disk
- On-the-fly VAE encoding with FlaxAutoencoderKL (TPU optimized)
- Gemma-3 text embeddings from class names
- Rectified Flow with Logit-Normal timestep sampling

Usage:
    # Single worker test
    python train_imagenet_tpu.py --data-dir /path/to/imagenet/train

    # Distributed training (via gcp_run.sh)
    ./gcp_run.sh "cd ~/ouroboros && python train_imagenet_tpu.py"
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
from flax import nnx
import optax
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import time
from pathlib import Path
import wandb
from datetime import datetime
from google.cloud import storage
from google.oauth2 import service_account
import pickle
import io

# GCS 서비스 계정 키 경로
GCS_SA_KEY_PATH = Path.home() / "gcs-sa-key.json"


def get_gcs_client():
    """GCS 클라이언트 생성"""
    if GCS_SA_KEY_PATH.exists():
        credentials = service_account.Credentials.from_service_account_file(
            str(GCS_SA_KEY_PATH),
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return storage.Client(credentials=credentials)
    else:
        return storage.Client()


from src.xut.xut_small import create_xut_small
from src.data.imagenet_parquet_loader import ImageNetParquetRAMLoader, ImageNetParquetLoader


# ============================================
# Configuration
# ============================================
@dataclass
class TrainingConfigImageNet:
    """ImageNet 학습 설정"""
    # ImageNet 데이터 경로 (GCS Parquet)
    imagenet_gcs_path: str = "gs://rdy-tpu-data-2025/imagenet-1k/data/"
    use_ram_preload: bool = True  # Load all parquet to RAM at startup

    # 배치 및 데이터
    global_batch_size: int = 1024
    num_devices: int = 32              # TPU v5e pod size
    batch_size_per_device: int = 32    # Conservative for VAE encoding

    # dtype
    use_bfloat16: bool = True

    # 학습
    num_epochs: int = 100
    steps_per_epoch: Optional[int] = 1250  # ~1.28M / 1024 = 1250 steps
    learning_rate: float = 0.5
    mup_base_dim: int = 1
    warmup_steps: int = 1000

    # 모델 (XUT-Small)
    model_dim: int = 896
    context_dim: int = 640              # Gemma-3 dimension
    mlp_dim: int = 3072
    heads: int = 14
    depth: int = 4
    enc_blocks: int = 1
    dec_blocks: int = 2

    # TREAD
    tread_selection_rate: float = 0.5

    # VAE 설정
    vae_model_id: str = "KMK040412/sdxl-vae-flax-msgpack"
    vae_scaling_factor: float = 0.13025
    image_size: int = 256

    # Data loading
    num_data_workers: int = 8
    prefetch_batches: int = 4

    # TPU 설정
    use_pjit: bool = True
    use_gradient_checkpointing: bool = True

    # Wandb
    wandb_project: str = "xut-imagenet-256"
    wandb_entity: str = None

    # GCS Checkpoint
    checkpoint_gcs_bucket: str = "rdy-tpu-data-2025"
    checkpoint_gcs_prefix: str = "checkpoints/xut-imagenet-256"
    checkpoint_keep_last_n: int = 3


# ============================================
# Rectified Flow Schedule
# ============================================
class RectifiedFlowSchedule:
    """Rectified Flow with Logit-Normal timestep sampling"""

    def __init__(self, logit_mean: float = 0.0, logit_std: float = 1.0):
        self.logit_mean = logit_mean
        self.logit_std = logit_std

    def sample_timesteps(self, key: jnp.ndarray, batch_size: int) -> jnp.ndarray:
        """Sample timesteps from logit-normal distribution"""
        u = jax.random.normal(key, (batch_size,))
        t = jax.nn.sigmoid(self.logit_mean + self.logit_std * u)
        return t

    def forward(self, x_0: jnp.ndarray, x_1: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """x_t = (1 - t) * x_0 + t * x_1"""
        t = t[:, None, None, None].astype(x_0.dtype)
        return (1.0 - t) * x_0 + t * x_1

    def get_velocity(self, x_0: jnp.ndarray, x_1: jnp.ndarray) -> jnp.ndarray:
        """v = x_1 - x_0"""
        return x_1 - x_0


# ============================================
# Sharding
# ============================================
@dataclass
class ShardingRules:
    """TPU Pod sharding"""
    data_axis: str = "data"

    def get_mesh(self):
        all_devices = jax.devices()
        num_devices = len(all_devices)
        print(f"  Creating mesh with {num_devices} devices")
        devices = mesh_utils.create_device_mesh((num_devices,))
        return Mesh(devices, (self.data_axis,))

    def named_sharding(self, partition_spec, mesh):
        return NamedSharding(mesh, partition_spec)


# ============================================
# JIT-compiled Train Step
# ============================================
@nnx.jit
def _train_step_jit(model, optimizer, x_t, t, velocity_target, text_emb):
    """JIT compiled training step for Rectified Flow"""
    def loss_fn(model):
        pred_v_nchw = model(x_t, t, ctx=text_emb, deterministic=False)
        pred_v = jnp.transpose(pred_v_nchw, (0, 2, 3, 1))
        return jnp.mean((pred_v - velocity_target) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


# ============================================
# Signal Handler
# ============================================
class GracefulKiller:
    """SIGTERM/SIGINT handler"""

    def __init__(self):
        self.kill_now = False
        self.current_epoch = 0
        self.current_step = 0

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        if jax.process_index() == 0:
            print(f"\n[SIGNAL] Received {sig_name} - saving checkpoint...")
        self.kill_now = True

    def register_state(self, epoch, step):
        self.current_epoch = epoch
        self.current_step = step


_graceful_killer = GracefulKiller()


# ============================================
# Trainer
# ============================================
class ImageNetTPUTrainer:
    """TPU trainer for ImageNet"""

    def __init__(self, model, optimizer, schedule, config: TrainingConfigImageNet,
                 wandb_enabled: bool = False, run_id: str = None):
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.config = config
        self.wandb_enabled = wandb_enabled

        # Sharding
        self.sharding_rules = ShardingRules()
        self.mesh = self.sharding_rules.get_mesh()

        print(f"\n[Sharding]")
        print(f"  Mesh: {self.mesh.shape}")
        print(f"  Data parallelism: {self.mesh.shape['data']}-way")

        # Learning rate schedule
        self.lr_schedule = self._create_lr_schedule()

        # Checkpoint
        if run_id is None:
            run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.gcs_checkpoint_path = f"{config.checkpoint_gcs_prefix}/{run_id}"

        # GCS client
        self.gcs_client = None
        self.gcs_bucket = None
        if jax.process_index() == 0:
            try:
                self.gcs_client = get_gcs_client()
                self.gcs_bucket = self.gcs_client.bucket(config.checkpoint_gcs_bucket)
                print(f"[GCS] Checkpoint: gs://{config.checkpoint_gcs_bucket}/{self.gcs_checkpoint_path}/")
            except Exception as e:
                print(f"[GCS] Init failed: {e}")

        # Local backup
        self.checkpoint_dir = Path("./checkpoints") / run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_lr_schedule(self):
        """Warmup + Cosine decay with muP scaling"""
        mup_lr = self.config.learning_rate * (self.config.mup_base_dim / self.config.model_dim)

        def lr_fn(step):
            if step < self.config.warmup_steps:
                return mup_lr * (step / self.config.warmup_steps)
            else:
                total_steps = self.config.steps_per_epoch * self.config.num_epochs
                progress = (step - self.config.warmup_steps) / (total_steps - self.config.warmup_steps)
                return mup_lr * 0.5 * (1 + jnp.cos(jnp.pi * progress))

        return lr_fn

    def train_step(self, x_t, t, velocity_target, text_emb, rng_key):
        """Single training step"""
        batch_size = t.shape[0]
        rng_key, subkey = jax.random.split(rng_key)
        cfg_mask = jax.random.uniform(subkey, (batch_size,)) < self.config.tread_selection_rate

        text_emb_3d = text_emb[:, None, :]
        text_emb_cond = jnp.where(cfg_mask[:, None, None], jnp.zeros_like(text_emb_3d), text_emb_3d)

        loss = _train_step_jit(self.model, self.optimizer, x_t, t, velocity_target, text_emb_cond)
        return loss, rng_key

    def save_checkpoint(self, epoch: int, step: int, loss: float, is_step_checkpoint: bool = False):
        """Save checkpoint to GCS"""
        if jax.process_index() != 0:
            return

        if is_step_checkpoint:
            global_step = epoch * self.config.steps_per_epoch + step
            checkpoint_name = f"step_{global_step:06d}.ckpt"
        else:
            checkpoint_name = f"epoch_{epoch:03d}.ckpt"

        try:
            model_state = nnx.state(self.model)
            optimizer_state = nnx.state(self.optimizer)

            def to_numpy(x):
                if hasattr(x, 'value'):
                    x = x.value
                if isinstance(x, jnp.ndarray):
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
                'run_id': self.run_id,
                'timestamp': datetime.now().isoformat(),
            }

            buffer = io.BytesIO()
            pickle.dump(checkpoint_data, buffer)
            buffer.seek(0)
            checkpoint_bytes = buffer.getvalue()

            if self.gcs_bucket is not None:
                gcs_path = f"{self.gcs_checkpoint_path}/{checkpoint_name}"
                blob = self.gcs_bucket.blob(gcs_path)
                blob.upload_from_string(checkpoint_bytes)
                print(f"  ✓ Checkpoint: gs://{self.config.checkpoint_gcs_bucket}/{gcs_path}")
            else:
                local_path = self.checkpoint_dir / checkpoint_name
                with open(local_path, 'wb') as f:
                    f.write(checkpoint_bytes)
                print(f"  ✓ Checkpoint: {local_path}")

        except Exception as e:
            print(f"  ✗ Checkpoint failed: {e}")

    def train_epoch(self, epoch_loader, epoch: int, start_step: int = 0):
        """Train one epoch"""
        losses = []
        rng_key = jax.random.PRNGKey(epoch)
        step = 0
        last_log_time = time.time()

        # Loss log
        loss_log_file = None
        if jax.process_index() == 0:
            loss_log_path = Path("/tmp/imagenet_loss_log.csv")
            write_header = not loss_log_path.exists()
            loss_log_file = open(loss_log_path, "a")
            if write_header:
                loss_log_file.write("timestamp,epoch,step,global_step,loss,lr\n")

        # Sharding
        batch_sharding = self.sharding_rules.named_sharding(
            P(self.sharding_rules.data_axis, None, None, None),
            self.mesh
        )
        emb_sharding = self.sharding_rules.named_sharding(
            P(self.sharding_rules.data_axis, None),
            self.mesh
        )

        local_devices = jax.local_devices()
        num_local_devices = len(local_devices)

        for batch_latents, batch_embeddings in epoch_loader.get_batches():
            if step < start_step:
                step += 1
                continue

            local_batch_size = batch_latents.shape[0]
            per_device_batch = local_batch_size // num_local_devices

            # Device placement
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

            # Global arrays
            global_batch_size = self.config.global_batch_size
            # Latents are already NHWC from ImageNet loader
            batch_latents = jax.make_array_from_single_device_arrays(
                (global_batch_size, 32, 32, 4),
                batch_sharding,
                latent_arrays
            )
            batch_embeddings = jax.make_array_from_single_device_arrays(
                (global_batch_size, batch_embeddings.shape[1]),
                emb_sharding,
                emb_arrays
            )

            batch_size = global_batch_size

            # Sample timesteps
            rng_key, subkey = jax.random.split(rng_key)
            t = self.schedule.sample_timesteps(subkey, batch_size)

            # Sample noise
            rng_key, subkey = jax.random.split(rng_key)
            compute_dtype = jnp.bfloat16 if self.config.use_bfloat16 else jnp.float32
            x_1 = jax.random.normal(subkey, batch_latents.shape, dtype=compute_dtype)

            # x_0 = latents (already scaled by VAE loader)
            x_0 = batch_latents.astype(compute_dtype)
            batch_embeddings = batch_embeddings.astype(compute_dtype)

            # Forward diffusion
            x_t = self.schedule.forward(x_0, x_1, t)
            velocity_target = self.schedule.get_velocity(x_0, x_1)

            # Train step
            global_step = epoch * self.config.steps_per_epoch + step
            loss, rng_key = self.train_step(x_t, t, velocity_target, batch_embeddings, rng_key)

            loss_val = float(loss)
            losses.append(loss_val)

            _graceful_killer.register_state(epoch, step)

            if _graceful_killer.kill_now:
                break

            if self.wandb_enabled:
                wandb.log({
                    "loss": loss_val,
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
                      f"Loss: {avg_loss:.6f} [{elapsed:.1f}s, {steps_per_sec:.2f} step/s]")

                if loss_log_file is not None:
                    timestamp = datetime.now().isoformat()
                    lr = float(self.lr_schedule(global_step))
                    loss_log_file.write(f"{timestamp},{epoch+1},{step},{global_step},{avg_loss:.6f},{lr:.8f}\n")
                    loss_log_file.flush()

            if step % 1000 == 0 and step > 0:
                avg_loss_1k = np.mean(losses[-1000:]) if len(losses) >= 1000 else np.mean(losses)
                self.save_checkpoint(epoch, step, avg_loss_1k, is_step_checkpoint=True)

            if step >= self.config.steps_per_epoch:
                break

        if loss_log_file is not None:
            loss_log_file.close()

        epoch_avg_loss = np.mean(losses) if losses else 0.0
        return losses, epoch_avg_loss


# ============================================
# Main
# ============================================
def parse_args():
    parser = argparse.ArgumentParser(description='ImageNet TPU Training')
    parser.add_argument('--gcs-path', type=str, default='gs://rdy-tpu-data-2025/imagenet-1k/data/',
                        help='GCS path to ImageNet parquet files')
    parser.add_argument('--no-ram-preload', action='store_true',
                        help='Disable RAM preload (stream from GCS)')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh training')
    return parser.parse_args()


def main():
    args = parse_args()

    worker_id = os.environ.get('JAX_PROCESS_INDEX', '0')
    log_file = f'/tmp/imagenet_train_worker_{worker_id}.log'

    print("="*60)
    print("ImageNet TPU Training (XUT-Small)")
    print("="*60)
    sys.stdout.flush()

    # Distributed init
    coordinator_addr = os.environ.get("JAX_COORDINATOR_ADDRESS")
    num_processes = os.environ.get("JAX_NUM_PROCESSES")
    process_idx = os.environ.get("JAX_PROCESS_INDEX")

    use_distributed = coordinator_addr is not None and num_processes is not None

    if use_distributed:
        print(f"\n[Distributed] Coordinator: {coordinator_addr}")
        try:
            jax.distributed.initialize(
                coordinator_address=coordinator_addr,
                num_processes=int(num_processes),
                process_id=int(process_idx) if process_idx else None,
            )
            print(f"  ✓ Process {jax.process_index()}/{jax.process_count()}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            use_distributed = False

    # Config
    config = TrainingConfigImageNet()
    config.imagenet_gcs_path = args.gcs_path
    config.use_ram_preload = not args.no_ram_preload

    print(f"\n[Config]")
    print(f"  ImageNet GCS: {config.imagenet_gcs_path}")
    print(f"  RAM preload: {config.use_ram_preload}")
    print(f"  Global batch: {config.global_batch_size}")
    print(f"  Image size: {config.image_size}")
    print(f"  VAE: {config.vae_model_id}")

    # Wandb
    wandb_enabled = False
    if jax.process_index() == 0:
        try:
            wandb.init(
                project=config.wandb_project,
                config={
                    "dataset": "imagenet-1k",
                    "global_batch_size": config.global_batch_size,
                    "learning_rate": config.learning_rate,
                    "model_dim": config.model_dim,
                }
            )
            wandb_enabled = True
        except Exception as e:
            print(f"[Wandb] Init failed: {e}")

    # Data loader
    print(f"\n[Data] Loading ImageNet from GCS Parquet...")
    local_batch_size = config.global_batch_size // jax.process_count()

    LoaderClass = ImageNetParquetRAMLoader if config.use_ram_preload else ImageNetParquetLoader

    data_loader = LoaderClass(
        gcs_bucket=config.imagenet_gcs_path,
        batch_size=local_batch_size,
        image_size=config.image_size,
        vae_model_id=config.vae_model_id,
        embedding_dim=config.context_dim,
        num_workers=config.num_data_workers,
        shard_data=True,
        use_vae=True,
    )

    # Calculate steps if not set
    if config.steps_per_epoch is None:
        config.steps_per_epoch = data_loader.calculate_steps_per_epoch(config.global_batch_size)

    print(f"  Samples: {data_loader.total_samples:,}")
    print(f"  Steps/epoch: {config.steps_per_epoch}")

    # Model
    print(f"\n[Model] Creating XUT-Small...")
    model = create_xut_small(
        dim=config.model_dim,
        ctx_dim=config.context_dim,
        mlp_dim=config.mlp_dim,
        heads=config.heads,
        depth=config.depth,
        enc_blocks=config.enc_blocks,
        dec_blocks=config.dec_blocks,
    )

    if config.use_bfloat16:
        def to_bf16(x):
            if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(jnp.bfloat16)
            return x

        for path, value in nnx.iter_graph(model):
            if isinstance(value, nnx.Variable) and hasattr(value, 'value'):
                value.value = to_bf16(value.value)

    print(f"  ✓ Model created (dim={config.model_dim}, depth={config.depth})")

    # Optimizer
    mup_lr = config.learning_rate * (config.mup_base_dim / config.model_dim)
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=mup_lr, weight_decay=1e-4)
        ),
        wrt=nnx.Param
    )
    print(f"  ✓ Optimizer (lr={mup_lr:.6f})")

    # Schedule
    schedule = RectifiedFlowSchedule()

    # Trainer
    trainer = ImageNetTPUTrainer(
        model, optimizer, schedule, config,
        wandb_enabled=wandb_enabled,
    )
    print(f"  Run ID: {trainer.run_id}")

    # Training loop
    print("\n" + "="*60)
    print("[Training]")
    print("="*60)

    total_start = time.time()

    for epoch in range(config.num_epochs):
        print(f"\n[Epoch {epoch+1}/{config.num_epochs}]")

        epoch_start = time.time()

        epoch_loader = data_loader.get_epoch_loader(
            epoch=epoch,
            steps_per_epoch=config.steps_per_epoch,
            num_workers=config.num_data_workers,
        )

        # Sync workers
        from jax.experimental.multihost_utils import sync_global_devices
        sync_global_devices(f"epoch_{epoch}_start")

        try:
            losses, epoch_avg_loss = trainer.train_epoch(epoch_loader, epoch)
        finally:
            epoch_loader.stop()
            gc.collect()

        epoch_time = time.time() - epoch_start
        print(f"  ✓ Epoch {epoch+1} done in {epoch_time/60:.1f}m, avg loss: {epoch_avg_loss:.6f}")

        # Checkpoint
        trainer.save_checkpoint(epoch, config.steps_per_epoch, epoch_avg_loss)

        if _graceful_killer.kill_now:
            print("\n[SIGNAL] Training interrupted")
            break

    total_time = time.time() - total_start
    print(f"\n✓ Training completed in {total_time/3600:.1f}h")

    data_loader.shutdown()

    if jax.process_index() == 0 and wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
