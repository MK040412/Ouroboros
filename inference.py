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
import pickle
import argparse
from pathlib import Path
from datetime import datetime

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


def load_checkpoint(ckpt_path: Path) -> dict:
    """Load checkpoint data from file

    Args:
        ckpt_path: Path to checkpoint file

    Returns:
        Checkpoint data dictionary
    """
    print(f"[Checkpoint] Loading: {ckpt_path}")

    with open(ckpt_path, 'rb') as f:
        ckpt_data = pickle.load(f)

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

    # Convert numpy to JAX arrays
    def to_jax(x):
        if isinstance(x, np.ndarray):
            arr = jnp.array(x)
            if use_bfloat16 and jnp.issubdtype(arr.dtype, jnp.floating):
                return arr.astype(jnp.bfloat16)
            return arr
        return x

    model_state_jax = jax.tree_util.tree_map(to_jax, model_state)
    nnx.update(model, model_state_jax)

    print("  Model weights restored!")


# ===========================================
# Text Embedding
# ===========================================
def get_text_embedding(text: str, embedding_dim: int = 640) -> np.ndarray:
    """Get text embedding using Gemma-3 270M

    Args:
        text: Input text prompt
        embedding_dim: Expected embedding dimension

    Returns:
        (1, D) numpy array of text embedding
    """
    try:
        from gemma import gm

        print(f"[Embedding] Loading Gemma-3 270M...")
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
        return embedding.astype(np.float32)

    except ImportError:
        print("[Warning] Gemma library not available, using random embedding")
        embedding = np.random.randn(1, embedding_dim).astype(np.float32)
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding / np.maximum(norm, 1e-8)


def get_text_embedding_transformers(text: str, model_name: str = "google/embeddinggemma-300m") -> np.ndarray:
    """Get text embedding using transformers (fallback method)

    Args:
        text: Input text prompt
        model_name: HuggingFace model name

    Returns:
        (1, D) numpy array of text embedding
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        print(f"[Embedding] Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        # Tokenize
        inputs = tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state

            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).numpy()

        # L2 normalize
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / np.maximum(norm, 1e-8)

        print(f"  Embedding shape: {embedding.shape}")
        return embedding.astype(np.float32)

    except ImportError:
        print("[Warning] transformers not available, using random embedding")
        embedding = np.random.randn(1, 768).astype(np.float32)
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding / np.maximum(norm, 1e-8)


# ===========================================
# Rectified Flow Sampling
# ===========================================
def sample_rectified_flow(
    model,
    text_embedding: np.ndarray,
    num_steps: int = 50,
    cfg_scale: float = 7.5,
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
        cfg_scale: Classifier-free guidance scale
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

    print(f"\n[Sampling] Rectified Flow with {num_steps} steps")
    print(f"  CFG scale: {cfg_scale}")
    print(f"  Seed: {seed}")

    # Start from pure noise (t=1)
    key = jax.random.PRNGKey(seed)
    dtype = jnp.bfloat16 if device_type == 'gpu' else jnp.float32
    x = jax.random.normal(key, latent_shape, dtype=dtype)

    # Text embedding: (1, D) -> (1, 1, D)
    text_emb = jnp.array(text_embedding[:, None, :], dtype=dtype)

    # Null embedding for CFG
    null_emb = jnp.zeros_like(text_emb)

    # Time steps from t=1 to t=0
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t = 1.0 - step * dt
        t_batch = jnp.array([[t]], dtype=dtype)

        # Conditional prediction
        v_cond = model(x, t_batch, ctx=text_emb, deterministic=True)

        # Unconditional prediction (for CFG)
        v_uncond = model(x, t_batch, ctx=null_emb, deterministic=True)

        # CFG: v = v_uncond + cfg_scale * (v_cond - v_uncond)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        # Transpose output from NCHW to NHWC if needed
        if v.shape[1] == 4 and v.shape[-1] != 4:  # NCHW format
            v = jnp.transpose(v, (0, 2, 3, 1))

        # Euler update: x_{t-dt} = x_t - dt * v
        x = x - dt * v

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Step {step + 1}/{num_steps}, t={t:.3f}")

    return np.array(x)


# ===========================================
# VAE Decoding
# ===========================================
def decode_latent_to_image(latent: np.ndarray, output_path: str = None):
    """Decode VAE latent to image using SDXL-VAE

    Args:
        latent: (B, H, W, C) latent array (NHWC format)
        output_path: Path to save the image (optional)

    Returns:
        PIL Image
    """
    import torch
    from diffusers import AutoencoderKL
    from PIL import Image

    print("\n[Decode] Loading SDXL-VAE...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    vae.eval().to(device)

    # Convert NHWC to NCHW: (B, H, W, C) -> (B, C, H, W)
    latent_nchw = np.transpose(latent, (0, 3, 1, 2))
    latent_tensor = torch.from_numpy(latent_nchw).float().to(device)

    print(f"  Latent shape: {latent_tensor.shape}")
    print(f"  Latent range: [{latent_tensor.min():.2f}, {latent_tensor.max():.2f}]")

    # Decode
    with torch.no_grad():
        decoded = vae.decode(latent_tensor).sample

    # Convert to image: (B, C, H, W) -> (H, W, C)
    image = decoded[0].cpu().permute(1, 2, 0).numpy()
    image = np.clip((image + 1) / 2 * 255, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(image)

    if output_path:
        pil_image.save(output_path)
        print(f"  Saved to: {output_path}")

    return pil_image


# ===========================================
# Main Inference Pipeline
# ===========================================
def run_inference(
    prompt: str,
    checkpoint_path: Path = None,
    output_dir: Path = Path("./outputs"),
    num_steps: int = 50,
    cfg_scale: float = 7.5,
    seed: int = None,
):
    """Run full inference pipeline

    Args:
        prompt: Text prompt for image generation
        checkpoint_path: Path to checkpoint (downloads if None)
        output_dir: Output directory for generated images
        num_steps: Number of sampling steps
        cfg_scale: CFG guidance scale
        seed: Random seed
    """
    import jax

    print("=" * 60)
    print("XUT-Small Text-to-Image Inference")
    print("=" * 60)

    # Setup device
    device_type, device = setup_device()

    # Get checkpoint
    if checkpoint_path is None:
        checkpoint_path = download_latest_checkpoint()

    # Load checkpoint
    ckpt_data = load_checkpoint(checkpoint_path)
    config = ckpt_data.get('config', {})

    # Create and restore model
    model = create_model(config)
    use_bfloat16 = device_type == 'gpu'
    restore_model(model, ckpt_data, use_bfloat16=use_bfloat16)

    # Get text embedding
    print(f"\n[Prompt] \"{prompt}\"")
    context_dim = config.get('context_dim', 640)
    text_embedding = get_text_embedding(prompt, embedding_dim=context_dim)

    # Sample
    latent = sample_rectified_flow(
        model=model,
        text_embedding=text_embedding,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        seed=seed,
        device_type=device_type,
    )

    # Decode to image
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"generated_{timestamp}.png"

    image = decode_latent_to_image(latent, str(output_path))

    print("\n" + "=" * 60)
    print(f"Image saved to: {output_path}")
    print("=" * 60)

    return image


def interactive_mode(
    checkpoint_path: Path = None,
    output_dir: Path = Path("./outputs"),
    num_steps: int = 50,
    cfg_scale: float = 7.5,
):
    """Interactive mode for continuous image generation"""
    import jax

    print("=" * 60)
    print("XUT-Small Interactive Text-to-Image")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop")
    print("Type 'seed:123' to set seed, 'steps:30' to change steps")
    print("=" * 60)

    # Setup device
    device_type, device = setup_device()

    # Get checkpoint
    if checkpoint_path is None:
        checkpoint_path = download_latest_checkpoint()

    # Load checkpoint
    ckpt_data = load_checkpoint(checkpoint_path)
    config = ckpt_data.get('config', {})

    # Create and restore model (once)
    model = create_model(config)
    use_bfloat16 = device_type == 'gpu'
    restore_model(model, ckpt_data, use_bfloat16=use_bfloat16)

    # Load embedding model once
    context_dim = config.get('context_dim', 640)
    print("\n[Ready] Model loaded. Enter your prompts:\n")

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

        # Generate image
        try:
            print(f"\n[Generating] \"{prompt}\"")

            # Get text embedding
            text_embedding = get_text_embedding(prompt, embedding_dim=context_dim)

            # Sample
            latent = sample_rectified_flow(
                model=model,
                text_embedding=text_embedding,
                num_steps=current_steps,
                cfg_scale=cfg_scale,
                seed=current_seed,
                device_type=device_type,
            )

            # Decode
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"generated_{timestamp}.png"
            image = decode_latent_to_image(latent, str(output_path))

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
    parser.add_argument('--cfg-scale', type=float, default=7.5,
                        help='CFG guidance scale (default: 7.5)')
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
