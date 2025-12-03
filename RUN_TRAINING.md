# Running Training on TPU v5e Pod

## Critical: Use Python Unbuffered Mode

**ALWAYS run training with `-u` flag to disable Python output buffering:**

```bash
python -u train_tpu_256.py
```

This is essential because:
1. XLA/JAX core dumps need immediate output
2. Logging won't be lost if crash occurs
3. Debug messages appear in real-time

## Full Command

```bash
# Single-host mode (local GPU/TPU)
python -u train_tpu_256.py

# OR with explicit unbuffered + environment setup
python -u -B train_tpu_256.py
```

## Environment Setup (TPU Pod)

For TPU Pod distributed training, set these environment variables:

```bash
# Export for distributed training
export JAX_COORDINATOR_ADDRESS="0.0.0.0:1234"  # Coordinator address
export JAX_TASKS_PER_CHIP=1

# Optional: TPU settings
export XLA_FLAGS="--xla_gpu_concurrent_kernels=true"
export JAX_PLATFORMS=tpu

# Run training
python -u train_tpu_256.py
```

## Debugging

### If training hangs on initialization:

1. **Check Step 1 (Environment Check)**: Ensure device count is detected
   ```
   [Step 1] Environment Check
     Device count: 16  ← Should be > 0 for TPU
   ```

2. **Check if JAX distributed is being attempted**:
   ```
   [Step 2] JAX distributed detected in environment
     ⚠ WARNING: JAX_COORDINATOR_ADDRESS is set
     Attempting distributed initialization (30 second timeout)...
   ```
   
   If this hangs beyond 30 seconds, the distributed service is unavailable.

3. **Check Step 7 (Embedding Model Load)**:
   ```
   [Step 7] Text Embedding Model Setup
     Initializing embedding provider...
   ```
   
   This can take time on first load (downloads model).

4. **View Debug Log**:
   ```bash
   tail -f /tmp/train_debug.log
   ```

### If "Aborted (core dumped)" occurs:

This usually means:
1. **Out of memory** (model too large)
2. **Device unavailable** (no TPU/GPU)
3. **JAX distributed timeout** (coordinator not reachable)
4. **Embedding model download failed**

**Solution:**
```bash
# Run with more verbose output
python -u train_tpu_256.py 2>&1 | tee /tmp/train_output.log

# Check the log file
tail -100 /tmp/train_output.log
tail -100 /tmp/train_debug.log
```

## Expected Output Sequence

```
============================================================
TPU v5e 16 Pod Training (256² XUT-Small)
============================================================

[Step 1] Environment Check
  Device count: 16

[Step 2] Single-Host Mode (no JAX distributed)

[Step 3] Creating TrainingConfig256...
  ✓ Config created

[Step 4] Initializing Wandb (Process 0 only)...
  ✓ Wandb initialized

[Step 5] Configuration Summary:
  TPU devices: 16 cores
  CPU workers: 112 vCPUs
  Epochs: 20

[Step 6] Device Detection
  Total devices: 16
  Device type: tpu

[Step 7] Text Embedding Model Setup
  Loading: google/embeddinggemma-300m
  Initializing embedding provider...
  ✓ Text embedding provider loaded

[Step 8] GCS Data Setup
  [8a] Initializing GCSDataLoaderSession...
  ✓ GCS session initialized
    PT files found: 1000

[Step 9] Model Initialization
  Creating XUT-Small model...
  ✓ XUT-Small initialized

[Step 10] Creating Optimizer...
  ✓ Optimizer created

[Step 11] Creating Diffusion Schedule...
  ✓ Diffusion schedule created

[Step 12] Initializing TPUTrainer...
  ✓ TPUTrainer initialized

[Step 13] Training Starting
  [Epoch 1/20]
  [E1a] Creating epoch loader...
  [E1b] ✓ Epoch loader ready
  [E1c] Starting training (pls wait, batches processing)...
  Epoch 1/20 Step 100/3750 Loss: 2.345678 [Sharded]
  ...
```

## Configuration

Edit `train_tpu_256.py` to change settings:

```python
@dataclass
class TrainingConfig256:
    # Data
    global_batch_size: int = 2048
    num_epochs: int = 20
    steps_per_epoch: int = 3750
    
    # GCS
    gcs_bucket: str = "gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/"
    num_data_workers: int = 112
    prefetch_ahead: int = 3
    max_cache_files: int = 3  # Disk space optimization
    
    # Model
    model_dim: int = 896
    depth: int = 4
    
    # Learning
    learning_rate: float = 0.5
    warmup_steps: int = 1000
```

## Monitoring

### Real-time logs
```bash
tail -f /tmp/train_debug.log
```

### Weights & Biases
Training metrics logged to WandB project: `xut-small-256`

### Disk space
Check GCS cache usage:
```bash
du -sh /tmp/gcs_pt_cache/
# Should be ~300MB (3 PT files × 100MB)
```

## Stopping Training

Press `Ctrl+C` to stop gracefully:
```
^C
[E5d] Stopping prefetch loader...
[E5e] ✓ Cleanup done
✓ Training completed!
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Device count: 0` | TPU/GPU not detected. Check hardware. |
| `Embedding model load hangs` | Network issue. Check internet connectivity. |
| `DEADLINE_EXCEEDED` | JAX coordinator not responding. Check TPU Pod setup. |
| `Out of memory` | Reduce `global_batch_size` or `num_epochs`. |
| `GCS access failed` | Run `gcloud auth application-default login`. |

## Performance Expectations

- **Initialization**: ~2-3 minutes (embedding model download)
- **First epoch**: ~2-3 hours (data prefetch + first training pass)
- **Subsequent epochs**: ~1.5-2 hours
- **Total (20 epochs)**: ~40-50 hours on TPU v5e 16
