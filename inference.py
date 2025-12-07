#!/usr/bin/env python3
"""
Text-to-Image Inference Script for XUT-Small

Features:
- Downloads latest checkpoint from GCS if not available locally
- Automatic GPU detection with CPU fallback
- Interactive text prompt input
- Rectified Flow sampling (Euler method)
- SDXL-VAE decoding
"""

import os
import sys
import gc
import pickle
import argparse
from pathlib import Path
from datetime import datetime

# Patch jax.monitoring for older JAX versions (compatibility with orbax-checkpoint)
import jax
if not hasattr(jax, 'monitoring'):
    class _DummyMonitoring:
        @staticmethod
        def record_scalar(*args, **kwargs):
            pass
    jax.monitoring = _DummyMonitoring()
elif not hasattr(jax.monitoring, 'record_scalar'):
    jax.monitoring.record_scalar = lambda *args, **kwargs: None

import numpy as np


# ===========================================
# Device Detection & Setup
# ===========================================
def setup_device():
    """Detect and setup compute device (GPU/CPU)

    Returns:
        tuple: (device_type, jax_device) where device_type is 'gpu' or 'cpu'
    """
    import jax

    # Try GPU first
    try:
        devices = jax.devices('gpu')
        if devices:
            print(f"[Device] Using GPU: {devices[0]}")
            return 'gpu', devices[0]
    except RuntimeError:
        pass

    # Fallback to CPU
    print("[Device] GPU not available, using CPU (this will be slow)")
    jax.config.update("jax_platform_name", "cpu")
    devices = jax.devices('cpu')
    return 'cpu', devices[0]


# ===========================================
# Checkpoint Download & Loading
# ===========================================
def download_latest_checkpoint(
    local_dir: Path = Path("./checkpoint"),
    gcs_bucket: str = "rdy-tpu-data-2025",
    gcs_prefix: str = "checkpoints/xut-small-256"
) -> Path:
    """Download latest checkpoint from GCS if not available locally

    Args:
        local_dir: Local directory to store checkpoints
        gcs_bucket: GCS bucket name
        gcs_prefix: GCS prefix path

    Returns:
        Path to the checkpoint file
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing local checkpoints
    local_ckpts = list(local_dir.glob("*.ckpt"))
    if local_ckpts:
        # Use the most recent local checkpoint
        latest_local = max(local_ckpts, key=lambda p: p.stat().st_mtime)
        print(f"[Checkpoint] Using local checkpoint: {latest_local}")
        return latest_local

    # Download from GCS
    print(f"[Checkpoint] No local checkpoint found, downloading from GCS...")

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(gcs_bucket)

        # List all checkpoints
        prefix = gcs_prefix + "/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        ckpt_blobs = [b for b in blobs if b.name.endswith('.ckpt')]

        if not ckpt_blobs:
            raise FileNotFoundError(f"No checkpoints found in gs://{gcs_bucket}/{gcs_prefix}")

        # Find latest checkpoint
        latest_blob = max(ckpt_blobs, key=lambda b: b.updated)
        print(f"[Checkpoint] Found: {latest_blob.name}")
        print(f"[Checkpoint] Size: {latest_blob.size / 1024 / 1024:.1f} MB")

        # Download
        local_path = local_dir / Path(latest_blob.name).name
        print(f"[Checkpoint] Downloading to: {local_path}")

        latest_blob.download_to_filename(str(local_path))
        print(f"[Checkpoint] Download complete!")

        return local_path

    except ImportError:
        print("[ERROR] google-cloud-storage not installed. Run: pip install google-cloud-storage")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to download checkpoint: {e}")
        sys.exit(1)


class FlaxStateContainer:
    """Generic container for Flax NNX state objects with version compatibility"""

    def __init__(self):
        self.value = None
        self._state = None

    def __setstate__(self, state):
        # Store the raw state
        self._state = state
        # Try to extract the actual value from various Flax versions
        if isinstance(state, dict):
            # Try different key names used in different Flax versions
            self.value = (
                state.get('raw_value') or
                state.get('value') or
                state.get('_value') or
                state
            )
        else:
            self.value = state


class FlaxStateDict(dict):
    """Dict subclass that handles Flax State unpickling"""

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.update(state)
        elif isinstance(state, (list, tuple)):
            # State might be saved as (dict_items,) or similar
            for item in state:
                if isinstance(item, dict):
                    self.update(item)
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    self[item[0]] = item[1]


class FlaxCompatUnpickler(pickle.Unpickler):
    """Custom unpickler to handle Flax NNX version mismatches"""

    def find_class(self, module, name):
        # Handle ALL flax.nnx classes to avoid version mismatch issues
        # This catches Variable, Param, BatchStat, Cache, State, and any other classes
        if 'flax.nnx' in module or 'flax.linen' in module:
            # State is a dict-like container
            if name == 'State':
                return FlaxStateDict
            # All other classes get our compatible container
            return FlaxStateContainer

        return super().find_class(module, name)


def load_checkpoint(ckpt_path: Path) -> dict:
    """Load checkpoint data from file

    Args:
        ckpt_path: Path to checkpoint file

    Returns:
        Checkpoint data dictionary
    """
    print(f"[Checkpoint] Loading: {ckpt_path}")

    # Use custom unpickler to handle Flax version mismatches
    with open(ckpt_path, 'rb') as f:
        ckpt_data = FlaxCompatUnpickler(f).load()

    print(f"  Epoch: {ckpt_data.get('epoch', 'N/A')}")
    print(f"  Step: {ckpt_data.get('step', 'N/A')}")
    print(f"  Loss: {ckpt_data.get('loss', 'N/A'):.6f}")
    print(f"  Timestamp: {ckpt_data.get('timestamp', 'N/A')}")

    return ckpt_data


# ===========================================
# Model Creation & Restoration
# ===========================================
def create_model(config: dict):
    """Create XUT model with config from checkpoint

    Args:
        config: Model configuration dictionary

    Returns:
        XUDiT model instance
    """
    from flax import nnx
    from src.xut.xut_small import create_xut_small

    model = create_xut_small(
        dim=config.get('model_dim', 896),
        ctx_dim=config.get('context_dim', 640),
        mlp_dim=config.get('mlp_dim', 3072),
        heads=config.get('heads', 14),
        depth=config.get('depth', 4),
        enc_blocks=config.get('enc_blocks', 1),
        dec_blocks=config.get('dec_blocks', 2),
    )

    print(f"[Model] Created XUT-Small")
    print(f"  Dimension: {config.get('model_dim', 896)}")
    print(f"  Context dim: {config.get('context_dim', 640)}")
    print(f"  Depth: {config.get('depth', 4)}")

    return model


def restore_model(model, ckpt_data: dict, use_bfloat16: bool = False):
    """Restore model weights from checkpoint

    Args:
        model: XUDiT model instance
        ckpt_data: Checkpoint data dictionary
        use_bfloat16: Whether to use bfloat16 (for GPU)
    """
    import jax.numpy as jnp
    from flax import nnx

    print("[Model] Restoring weights from checkpoint...")

    model_state = ckpt_data['model_state']

    # Extract raw numpy arrays from FlaxStateContainer/FlaxStateDict structure
    def extract_raw_value(obj):
        """Recursively extract raw numpy array from checkpoint structure"""
        if isinstance(obj, FlaxStateContainer):
            return extract_raw_value(obj.value)
        elif isinstance(obj, dict):
            if '_raw_value' in obj:
                return obj['_raw_value']
            elif 'raw_value' in obj:
                return obj['raw_value']
            elif 'value' in obj:
                return extract_raw_value(obj['value'])
            else:
                # It's a nested dict, recurse into each key
                return {k: extract_raw_value(v) for k, v in obj.items()}
        elif hasattr(obj, 'shape'):  # numpy array or jax array
            return obj
        else:
            return obj

    # Convert to JAX arrays with proper dtype handling
    def to_jax(x):
        if isinstance(x, np.ndarray):
            arr = jnp.array(x)
            if use_bfloat16 and jnp.issubdtype(arr.dtype, jnp.floating):
                return arr.astype(jnp.bfloat16)
            return arr
        if hasattr(x, 'shape') and hasattr(jnp, 'ndarray'):
            if use_bfloat16 and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(jnp.bfloat16)
            return x
        return x

    # Get the mapping from checkpoint
    if hasattr(model_state, '__getitem__') and '_mapping' in model_state:
        ckpt_mapping = model_state['_mapping']
    elif hasattr(model_state, '_mapping'):
        ckpt_mapping = model_state._mapping
    else:
        ckpt_mapping = model_state

    # Extract raw values and convert to JAX
    ckpt_raw = extract_raw_value(ckpt_mapping)
    ckpt_jax = jax.tree_util.tree_map(to_jax, ckpt_raw)

    # Get current model state and update it
    current_state = nnx.state(model)

    # Recursive function to update model state from checkpoint
    def update_state(state_obj, ckpt_dict, path=""):
        """Recursively update state object from checkpoint dict"""
        if isinstance(ckpt_dict, dict):
            for key, value in ckpt_dict.items():
                # Handle integer keys (list indices)
                if isinstance(key, int):
                    if hasattr(state_obj, '__getitem__'):
                        try:
                            sub_state = state_obj[key]
                            new_path = f"{path}[{key}]"
                            update_state(sub_state, value, new_path)
                        except (KeyError, IndexError, TypeError):
                            pass
                elif hasattr(state_obj, str(key)):
                    sub_state = getattr(state_obj, str(key))
                    new_path = f"{path}.{key}" if path else str(key)
                    update_state(sub_state, value, new_path)
        elif hasattr(ckpt_dict, 'shape'):
            # This is an array - update the state value
            if hasattr(state_obj, 'value'):
                state_obj.value = ckpt_dict

    update_state(current_state, ckpt_jax)

    # Apply updated state back to model
    nnx.update(model, current_state)

    print("  Model weights restored!")


# ===========================================
# Text Embedding
# ===========================================
def get_text_embedding(text: str, embedding_dim: int = 640, device_type: str = 'gpu') -> np.ndarray:
    """Get text embedding using Gemma-3 270M on GPU, then release memory

    Args:
        text: Input text prompt
        embedding_dim: Expected embedding dimension
        device_type: 'gpu' or 'cpu'

    Returns:
        (1, D) numpy array of text embedding
    """
    try:
        from gemma import gm

        print(f"[Embedding] Loading Gemma-3 270M on {device_type.upper()}...")
        model = gm.nn.Gemma3_270M()
        params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_270M_PT)
        tokenizer = gm.text.Gemma3Tokenizer()

        # Tokenize
        max_length = 128
        tokens = tokenizer.encode(text, add_bos=True)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [0] * (max_length - len(tokens))

        tokens_array = np.array([tokens], dtype=np.int32)

        # Forward pass
        out = model.apply(
            {'params': params},
            tokens=tokens_array,
            return_last_only=False,
            return_hidden_states=True,
        )
        last_hidden = out.hidden_states[-1]

        # Mean pooling
        mask = (tokens_array != 0).astype(np.float32)
        mask_expanded = mask[:, :, None]
        sum_embeddings = np.sum(np.array(last_hidden) * mask_expanded, axis=1)
        sum_mask = np.clip(mask.sum(axis=1, keepdims=True), a_min=1e-9, a_max=None)
        embedding = sum_embeddings / sum_mask

        # L2 normalize
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / np.maximum(norm, 1e-8)

        print(f"  Embedding shape: {embedding.shape}")

        # Save embedding to file for debugging
        emb_file = "outputs/embedding_debug.txt"
        os.makedirs("outputs", exist_ok=True)
        with open(emb_file, 'w') as f:
            f.write(f"Text: {text}\n")
            f.write(f"Shape: {embedding.shape}\n")
            f.write(f"Dtype: {embedding.dtype}\n")
            f.write(f"Range: [{embedding.min():.6f}, {embedding.max():.6f}]\n")
            f.write(f"Mean: {embedding.mean():.6f}\n")
            f.write(f"Std: {embedding.std():.6f}\n")
            f.write(f"L2 norm: {np.linalg.norm(embedding):.6f}\n")
            f.write(f"\nFirst 20 values:\n{embedding[0, :20]}\n")
            f.write(f"\nFull embedding:\n{embedding[0].tolist()}\n")
        print(f"  Embedding saved to: {emb_file}")

        result = embedding.astype(np.float32)

        # Release Gemma model from GPU memory
        print("[Embedding] Releasing Gemma model from GPU memory...")
        del model, params, tokenizer, out, last_hidden
        jax.clear_caches()
        gc.collect()
        print("[Embedding] GPU memory released")

        return result

    except Exception as e:
        print(f"[Warning] Gemma failed: {e}")
        print("[Warning] Using random embedding as fallback")
        embedding = np.random.randn(1, embedding_dim).astype(np.float32)
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding / np.maximum(norm, 1e-8)


# ===========================================
# Rectified Flow Sampling
# ===========================================
def sample_rectified_flow(
    model,
    text_embedding: np.ndarray,
    num_steps: int = 50,
    cfg_scale: float = 0.0,
    latent_shape: tuple = (1, 32, 32, 4),
    seed: int = None,
    device_type: str = 'cpu'
):
    """Sample using Rectified Flow (Euler method)

    Rectified Flow: x_t = (1 - t) * x_0 + t * x_1
    Model predicts velocity v = x_1 - x_0
    Update: x_{t-dt} = x_t - dt * v_pred

    Args:
        model: XUDiT model
        text_embedding: (1, D) text embedding
        num_steps: Number of sampling steps
        cfg_scale: Classifier-free guidance scale (0 = conditional only, no CFG)
        latent_shape: Shape of latent (B, H, W, C)
        seed: Random seed
        device_type: 'gpu' or 'cpu'

    Returns:
        (B, H, W, C) sampled latent
    """
    import jax
    import jax.numpy as jnp

    if seed is None:
        seed = int(datetime.now().timestamp()) % (2**31)

    # CFG only needed when scale > 1
    use_cfg = cfg_scale > 1.0

    print(f"\n[Sampling] Rectified Flow with {num_steps} steps")
    print(f"  CFG scale: {cfg_scale}" + (" (conditional only)" if not use_cfg else ""))
    print(f"  Seed: {seed}")

    # Start from pure noise (t=1)
    key = jax.random.PRNGKey(seed)
    dtype = jnp.bfloat16 if device_type == 'gpu' else jnp.float32
    x = jax.random.normal(key, latent_shape, dtype=dtype)

    print(f"  Initial noise range: [{float(x.min()):.2f}, {float(x.max()):.2f}]")

    # Text embedding: (1, D) -> (1, 1, D)
    text_emb = jnp.array(text_embedding[:, None, :], dtype=dtype)

    # Null embedding for CFG (only if needed)
    if use_cfg:
        null_emb = jnp.zeros_like(text_emb)

    # Time steps from t=1 to t=0
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t = 1.0 - step * dt
        t_batch = jnp.array([[t]], dtype=dtype)

        # Conditional prediction
        v_cond = model(x, t_batch, ctx=text_emb, deterministic=True)

        if use_cfg:
            # Unconditional prediction (for CFG)
            v_uncond = model(x, t_batch, ctx=null_emb, deterministic=True)
            # CFG: v = v_uncond + cfg_scale * (v_cond - v_uncond)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            # No CFG - use conditional prediction directly
            v = v_cond

        # Transpose output from NCHW to NHWC if needed
        if v.shape[1] == 4 and v.shape[-1] != 4:  # NCHW format
            v = jnp.transpose(v, (0, 2, 3, 1))

        # Euler update: x_{t-dt} = x_t - dt * v
        x = x - dt * v

        if (step + 1) % 10 == 0 or step == 0:
            x_min, x_max = float(x.min()), float(x.max())
            v_min, v_max = float(v.min()), float(v.max())
            v_mean, v_std = float(v.mean()), float(v.std())
            print(f"  Step {step + 1}/{num_steps}, t={t:.3f}, x=[{x_min:.2f}, {x_max:.2f}], v=[{v_min:.2f}, {v_max:.2f}], v_mean={v_mean:.3f}")

    # Final stats
    x_np = np.array(x)
    print(f"\n  Final latent stats:")
    print(f"    Range: [{x_np.min():.2f}, {x_np.max():.2f}]")
    print(f"    Mean: {x_np.mean():.4f}, Std: {x_np.std():.4f}")

    return x_np


# ===========================================
# VAE Decoding
# ===========================================
def decode_latent_to_image(latent: np.ndarray, output_path: str = None, use_gpu: bool = True):
    """Decode VAE latent to image using SDXL-VAE on GPU

    Args:
        latent: (B, H, W, C) latent array (NHWC format)
        output_path: Path to save the image (optional)
        use_gpu: Whether to use GPU for decoding

    Returns:
        PIL Image
    """
    import torch
    from diffusers import AutoencoderKL
    from PIL import Image

    # Use GPU if available and requested
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"\n[Decode] Loading SDXL-VAE on {device.upper()}...")

    vae = AutoencoderKL.from_pretrained("stabilityai/SDXL-VAE").to(device)
    vae.eval()

    # Convert NHWC to NCHW: (B, H, W, C) -> (B, C, H, W)
    latent_nchw = np.transpose(latent, (0, 3, 1, 2))
    latent_tensor = torch.from_numpy(latent_nchw).float().to(device)

    print(f"  Latent shape: {latent_tensor.shape}")
    print(f"  Latent range: [{latent_tensor.min():.2f}, {latent_tensor.max():.2f}]")

    # Decode
    with torch.no_grad():
        decoded = vae.decode(latent_tensor).sample

    # Convert to image: (B, C, H, W) -> (H, W, C)
    image = decoded[0].permute(1, 2, 0).cpu().numpy()
    image = np.clip((image + 1) / 2 * 255, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(image)

    if output_path:
        pil_image.save(output_path)
        print(f"  Saved to: {output_path}")

    # Release VAE from GPU memory
    print("[Decode] Releasing VAE from GPU memory...")
    del vae, latent_tensor, decoded
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print("[Decode] GPU memory released")

    return pil_image


# ===========================================
# Main Inference Pipeline
# ===========================================
def run_inference(
    prompt: str,
    checkpoint_path: Path = None,
    output_dir: Path = Path("./outputs"),
    num_steps: int = 50,
    cfg_scale: float = 0.0,
    seed: int = None,
):
    """Run full inference pipeline with sequential GPU memory management

    Pipeline:
    1. Load Text Encoder (Gemma-3) -> Compute embedding -> Release from GPU
    2. Load XUT Model -> Sample latent -> Release from GPU
    3. Load VAE Decoder -> Decode image -> Release from GPU

    Args:
        prompt: Text prompt for image generation
        checkpoint_path: Path to checkpoint (downloads if None)
        output_dir: Output directory for generated images
        num_steps: Number of sampling steps
        cfg_scale: CFG guidance scale
        seed: Random seed
    """
    print("=" * 60)
    print("XUT-Small Text-to-Image Inference")
    print("(Sequential GPU Memory Management)")
    print("=" * 60)

    # Setup device
    device_type, _ = setup_device()

    # Get checkpoint path (don't load yet)
    if checkpoint_path is None:
        checkpoint_path = download_latest_checkpoint()

    # Load checkpoint data (CPU only, small)
    ckpt_data = load_checkpoint(checkpoint_path)
    config = ckpt_data.get('config', {})
    context_dim = config.get('context_dim', 640)

    # =============================================
    # Step 1: Text Embedding (Gemma-3 on GPU)
    # =============================================
    print(f"\n{'='*60}")
    print("[Step 1/3] Text Embedding")
    print(f"{'='*60}")
    print(f"[Prompt] \"{prompt}\"")

    text_embedding = get_text_embedding(
        prompt,
        embedding_dim=context_dim,
        device_type=device_type
    )
    # Gemma model is released inside get_text_embedding()

    # =============================================
    # Step 2: XUT Sampling (XUT Model on GPU)
    # =============================================
    print(f"\n{'='*60}")
    print("[Step 2/3] XUT Sampling")
    print(f"{'='*60}")

    # Create and restore model
    model = create_model(config)
    use_bfloat16 = device_type == 'gpu'
    restore_model(model, ckpt_data, use_bfloat16=use_bfloat16)

    # Sample
    latent = sample_rectified_flow(
        model=model,
        text_embedding=text_embedding,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        seed=seed,
        device_type=device_type,
    )

    # Release XUT model from GPU memory
    print("[XUT] Releasing XUT model from GPU memory...")
    del model, ckpt_data
    jax.clear_caches()
    gc.collect()
    print("[XUT] GPU memory released")

    # =============================================
    # Step 3: VAE Decoding (SDXL-VAE on GPU)
    # =============================================
    print(f"\n{'='*60}")
    print("[Step 3/3] VAE Decoding")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"generated_{timestamp}.png"

    # Inverse VAE scaling: model outputs normalized latents, VAE expects raw scale
    SDXL_VAE_SCALE = 0.13025
    latent = latent / SDXL_VAE_SCALE

    image = decode_latent_to_image(
        latent,
        str(output_path),
        use_gpu=(device_type == 'gpu')
    )
    # VAE is released inside decode_latent_to_image()

    print("\n" + "=" * 60)
    print(f"Image saved to: {output_path}")
    print("=" * 60)

    return image


def interactive_mode(
    checkpoint_path: Path = None,
    output_dir: Path = Path("./outputs"),
    num_steps: int = 50,
    cfg_scale: float = 0.0,
):
    """Interactive mode with sequential GPU memory management per image

    Each image generation follows the pipeline:
    1. Text Encoder -> GPU -> embedding -> release
    2. XUT Model -> GPU -> sample -> release
    3. VAE Decoder -> GPU -> decode -> release
    """
    print("=" * 60)
    print("XUT-Small Interactive Text-to-Image")
    print("(Sequential GPU Memory Management)")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop")
    print("Type 'seed:123' to set seed, 'steps:30' to change steps")
    print("=" * 60)

    # Setup device
    device_type, _ = setup_device()

    # Get checkpoint
    if checkpoint_path is None:
        checkpoint_path = download_latest_checkpoint()

    # Load checkpoint data (CPU only, small)
    ckpt_data = load_checkpoint(checkpoint_path)
    config = ckpt_data.get('config', {})
    context_dim = config.get('context_dim', 640)

    print("\n[Ready] Enter your prompts:\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    current_seed = None
    current_steps = num_steps

    while True:
        try:
            prompt = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        # Handle settings
        if prompt.lower().startswith('seed:'):
            try:
                current_seed = int(prompt.split(':')[1])
                print(f"  Seed set to: {current_seed}")
            except ValueError:
                print("  Invalid seed value")
            continue

        if prompt.lower().startswith('steps:'):
            try:
                current_steps = int(prompt.split(':')[1])
                print(f"  Steps set to: {current_steps}")
            except ValueError:
                print("  Invalid steps value")
            continue

        # Generate image with sequential GPU memory management
        try:
            print(f"\n[Generating] \"{prompt}\"")

            # Step 1: Text Embedding (Gemma on GPU, then release)
            print(f"\n[Step 1/3] Text Embedding")
            text_embedding = get_text_embedding(
                prompt,
                embedding_dim=context_dim,
                device_type=device_type
            )

            # Step 2: XUT Sampling (XUT on GPU, then release)
            print(f"\n[Step 2/3] XUT Sampling")
            model = create_model(config)
            use_bfloat16 = device_type == 'gpu'
            restore_model(model, ckpt_data, use_bfloat16=use_bfloat16)

            latent = sample_rectified_flow(
                model=model,
                text_embedding=text_embedding,
                num_steps=current_steps,
                cfg_scale=cfg_scale,
                seed=current_seed,
                device_type=device_type,
            )

            # Release XUT model
            print("[XUT] Releasing XUT model from GPU memory...")
            del model
            jax.clear_caches()
            gc.collect()

            # Step 3: VAE Decoding (VAE on GPU, then release)
            print(f"\n[Step 3/3] VAE Decoding")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"generated_{timestamp}.png"

            SDXL_VAE_SCALE = 0.13025
            latent = latent / SDXL_VAE_SCALE

            decode_latent_to_image(
                latent,
                str(output_path),
                use_gpu=(device_type == 'gpu')
            )

            print(f"\n[Done] Saved to: {output_path}")

        except Exception as e:
            print(f"\n[Error] {e}")
            import traceback
            traceback.print_exc()


# ===========================================
# CLI
# ===========================================
def main():
    parser = argparse.ArgumentParser(description='XUT-Small Text-to-Image Inference')

    parser.add_argument('--prompt', '-p', type=str, default=None,
                        help='Text prompt for image generation')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Path to checkpoint file (downloads from GCS if not specified)')
    parser.add_argument('--output-dir', '-o', type=str, default='./outputs',
                        help='Output directory for generated images')
    parser.add_argument('--steps', '-s', type=int, default=50,
                        help='Number of sampling steps (default: 50)')
    parser.add_argument('--cfg-scale', type=float, default=1.0,
                        help='CFG guidance scale (0 = conditional only, no CFG)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    output_dir = Path(args.output_dir)

    if args.interactive:
        interactive_mode(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            num_steps=args.steps,
            cfg_scale=args.cfg_scale,
        )
    elif args.prompt:
        run_inference(
            prompt=args.prompt,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            num_steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
        )
    else:
        # No prompt provided, ask for one
        prompt = input("Enter your prompt: ").strip()
        if prompt:
            run_inference(
                prompt=prompt,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                num_steps=args.steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
            )
        else:
            print("No prompt provided. Use --interactive for interactive mode.")
            parser.print_help()


if __name__ == "__main__":
    # Import JAX here to allow device setup
    import jax
    main()
