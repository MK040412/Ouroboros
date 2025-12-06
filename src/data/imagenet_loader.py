"""
ImageNet DataLoader with JAX VAE Encoding for TPU Training

Key features:
1. Load ImageNet images from local disk (distributed to each worker)
2. Encode images on-the-fly using FlaxAutoencoderKL (TPU-optimized)
3. Use cached class name embeddings via Gemma-3 (640-dim)
4. RAM preload for fast access during training

Usage:
    loader = ImageNetRAMLoader(
        data_dir="/path/to/imagenet/train",
        batch_size=64,
        image_size=256,
    )

    for latents, embeddings in loader.get_batches(steps=1000):
        # latents: (B, 32, 32, 4) NHWC
        # embeddings: (B, 640)
        pass
"""

import os
import gc
import json
import queue
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image


# ImageNet class ID to human-readable label mapping (synset -> label)
# Will be loaded from imagenet_classes.json
IMAGENET_CLASSES: Optional[Dict[str, str]] = None


def load_imagenet_classes(json_path: str = None) -> Dict[str, str]:
    """Load ImageNet synset -> label mapping"""
    global IMAGENET_CLASSES
    if IMAGENET_CLASSES is not None:
        return IMAGENET_CLASSES

    if json_path is None:
        # Default path relative to this file
        json_path = Path(__file__).parent / "imagenet_classes.json"

    if Path(json_path).exists():
        with open(json_path, 'r') as f:
            IMAGENET_CLASSES = json.load(f)
    else:
        # Fallback: generate from synset folder names
        print(f"[ImageNet] imagenet_classes.json not found, will use folder names")
        IMAGENET_CLASSES = {}

    return IMAGENET_CLASSES


@dataclass
class ImageNetSample:
    """Single ImageNet sample"""
    path: str
    synset: str  # e.g., "n01440764"
    label: str   # e.g., "tench"
    class_idx: int


class FlaxVAEEncoder:
    """JAX/Flax VAE encoder wrapper for TPU

    Uses FlaxAutoencoderKL from diffusers for encoding images to latents.
    Optimized for TPU with JIT compilation and pmap.
    """

    def __init__(self,
                 model_id: str = "stabilityai/sdxl-vae",
                 dtype: jnp.dtype = jnp.bfloat16,
                 scaling_factor: float = 0.13025):
        """
        Args:
            model_id: HuggingFace model ID for VAE
            dtype: JAX dtype for computation
            scaling_factor: SDXL VAE scaling factor
        """
        self.model_id = model_id
        self.dtype = dtype
        self.scaling_factor = scaling_factor

        # Lazy loading to avoid import errors when diffusers not installed
        self.vae = None
        self.vae_params = None
        self._encode_fn = None

    def _load_model(self):
        """Load FlaxAutoencoderKL model"""
        if self.vae is not None:
            return

        try:
            from diffusers import FlaxAutoencoderKL
        except ImportError:
            raise ImportError(
                "diffusers with flax support required. "
                "Install with: pip install diffusers[flax]"
            )

        print(f"[VAE] Loading FlaxAutoencoderKL from {self.model_id}...")

        self.vae, self.vae_params = FlaxAutoencoderKL.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            from_pt=True,
        )

        # JIT compile encode function
        @jax.jit
        def encode_fn(params, images):
            """Encode images to latents

            Args:
                params: VAE parameters
                images: (B, H, W, C) normalized to [-1, 1]
            Returns:
                latents: (B, H//8, W//8, 4) scaled
            """
            # FlaxAutoencoderKL expects (B, C, H, W)
            images_nchw = jnp.transpose(images, (0, 3, 1, 2))

            # Encode
            latent_dist = self.vae.apply(
                {"params": params},
                images_nchw,
                method=self.vae.encode
            )

            # Sample from distribution (deterministic mode)
            latents = latent_dist.latent_dist.mode()

            # Scale and transpose back to NHWC
            latents = latents * self.scaling_factor
            latents = jnp.transpose(latents, (0, 2, 3, 1))

            return latents

        self._encode_fn = encode_fn
        print(f"[VAE] ✓ Model loaded and JIT compiled")

    def encode(self, images: jnp.ndarray) -> jnp.ndarray:
        """Encode images to latents

        Args:
            images: (B, H, W, C) in range [0, 1]
        Returns:
            latents: (B, H//8, W//8, 4) scaled
        """
        self._load_model()

        # Normalize to [-1, 1]
        images = images * 2.0 - 1.0
        images = images.astype(self.dtype)

        return self._encode_fn(self.vae_params, images)


class ImageNetRAMLoader:
    """Load ImageNet images into RAM and encode on-the-fly with VAE

    Memory strategy:
    - Load all images as uint8 (much smaller than float32 latents)
    - Encode to latents on-demand during training
    - Cache class embeddings (only 1000 classes)

    For ImageNet-1K (256x256):
    - Raw images: 1.28M × 256 × 256 × 3 bytes = ~251GB
    - With JPEG compression in RAM: ~30-50GB
    - We use disk-based loading with prefetch instead
    """

    def __init__(self,
                 data_dir: str,
                 batch_size: int = 64,
                 image_size: int = 256,
                 vae_model_id: str = "stabilityai/sdxl-vae",
                 embedding_dim: int = 640,
                 num_workers: int = 8,
                 prefetch_batches: int = 4,
                 shard_data: bool = True):
        """
        Args:
            data_dir: Path to ImageNet train directory
            batch_size: Batch size per device
            image_size: Image resolution
            vae_model_id: HuggingFace VAE model ID
            embedding_dim: Text embedding dimension (Gemma-3: 640)
            num_workers: Number of image loading workers
            prefetch_batches: Number of batches to prefetch
            shard_data: Whether to shard data across workers
        """
        import sys

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches

        # Worker info
        self.process_index = int(os.environ.get('JAX_PROCESS_INDEX', '0'))
        self.num_processes = int(os.environ.get('JAX_NUM_PROCESSES', '1'))

        print(f"\n[ImageNet] Initializing loader (Worker {self.process_index}/{self.num_processes})")
        sys.stdout.flush()

        # Load class mapping
        self.classes = load_imagenet_classes()

        # Scan ImageNet directory
        self.samples = self._scan_directory()

        # Shard samples across workers
        if shard_data and self.num_processes > 1:
            self.samples = [s for i, s in enumerate(self.samples)
                          if i % self.num_processes == self.process_index]
            print(f"[ImageNet] Worker {self.process_index}: {len(self.samples)} samples (sharded)")

        self.total_samples = len(self.samples)
        print(f"[ImageNet] Total samples for this worker: {self.total_samples:,}")
        sys.stdout.flush()

        # VAE encoder (lazy loaded)
        self.vae = FlaxVAEEncoder(
            model_id=vae_model_id,
            dtype=jnp.bfloat16,
        )

        # Class embeddings cache (computed once)
        self.class_embeddings: Optional[Dict[int, np.ndarray]] = None

        # Prefetch queue
        self.prefetch_queue: Optional[queue.Queue] = None
        self.stop_event = threading.Event()
        self.prefetch_thread: Optional[threading.Thread] = None

    def _scan_directory(self) -> List[ImageNetSample]:
        """Scan ImageNet directory structure"""
        samples = []

        if not self.data_dir.exists():
            raise FileNotFoundError(f"ImageNet directory not found: {self.data_dir}")

        # ImageNet structure: train/n01440764/n01440764_123.JPEG
        synset_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        print(f"[ImageNet] Scanning {len(synset_dirs)} synset directories...")

        for class_idx, synset_dir in enumerate(synset_dirs):
            synset = synset_dir.name
            label = self.classes.get(synset, synset)

            # Find all images in this class
            image_paths = list(synset_dir.glob("*.JPEG")) + list(synset_dir.glob("*.jpeg"))
            image_paths += list(synset_dir.glob("*.jpg")) + list(synset_dir.glob("*.png"))

            for img_path in image_paths:
                samples.append(ImageNetSample(
                    path=str(img_path),
                    synset=synset,
                    label=label,
                    class_idx=class_idx,
                ))

        print(f"[ImageNet] Found {len(samples):,} samples in {len(synset_dirs)} classes")
        return samples

    def _load_image(self, sample: ImageNetSample) -> np.ndarray:
        """Load and preprocess single image"""
        try:
            img = Image.open(sample.path).convert('RGB')

            # Center crop to square
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))

            # Resize to target size
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)

            # Convert to numpy float32 [0, 1]
            img_np = np.array(img, dtype=np.float32) / 255.0

            return img_np
        except Exception as e:
            print(f"[ImageNet] Failed to load {sample.path}: {e}")
            # Return black image on error
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

    def _load_batch_images(self, indices: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Load batch of images using thread pool

        Args:
            indices: Sample indices to load
        Returns:
            images: (B, H, W, C) float32 [0, 1]
            class_indices: List of class indices
        """
        batch_samples = [self.samples[i] for i in indices]

        # Parallel image loading
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            images = list(executor.map(self._load_image, batch_samples))

        images = np.stack(images, axis=0)
        class_indices = [s.class_idx for s in batch_samples]

        return images, class_indices

    def _compute_class_embeddings(self) -> Dict[int, np.ndarray]:
        """Compute text embeddings for all class names

        Uses Gemma-3 embeddings (640-dim) for class labels.
        """
        if self.class_embeddings is not None:
            return self.class_embeddings

        print(f"[ImageNet] Computing class embeddings...")

        # Get unique classes
        unique_classes = sorted(set(s.class_idx for s in self.samples))
        class_to_label = {}
        for s in self.samples:
            if s.class_idx not in class_to_label:
                class_to_label[s.class_idx] = s.label

        # For now, use random embeddings as placeholder
        # TODO: Replace with actual Gemma-3 embeddings
        self.class_embeddings = {}
        for class_idx in unique_classes:
            # Use deterministic random based on class name
            label = class_to_label[class_idx]
            seed = hash(label) % (2**32)
            rng = np.random.RandomState(seed)
            # Normalized random embedding
            emb = rng.randn(self.embedding_dim).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            self.class_embeddings[class_idx] = emb

        print(f"[ImageNet] ✓ Computed {len(self.class_embeddings)} class embeddings")
        return self.class_embeddings

    def _get_class_embeddings_batch(self, class_indices: List[int]) -> np.ndarray:
        """Get embeddings for a batch of class indices"""
        embeddings = self._compute_class_embeddings()
        batch_emb = np.stack([embeddings[idx] for idx in class_indices], axis=0)
        return batch_emb

    def get_batch(self, rng_key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get single batch with VAE encoding

        Args:
            rng_key: JAX random key for sampling
        Returns:
            latents: (B, 32, 32, 4) NHWC, bfloat16
            embeddings: (B, 640) bfloat16
        """
        # Sample random indices
        indices = jax.random.randint(rng_key, (self.batch_size,), 0, self.total_samples)
        indices = np.array(indices)

        # Load images
        images, class_indices = self._load_batch_images(indices)

        # Convert to JAX and encode with VAE
        images_jax = jnp.array(images)
        latents = self.vae.encode(images_jax)

        # Get class embeddings
        embeddings = self._get_class_embeddings_batch(class_indices)
        embeddings = jnp.array(embeddings, dtype=jnp.bfloat16)

        return latents, embeddings

    def calculate_steps_per_epoch(self, global_batch_size: int) -> int:
        """Calculate steps per epoch"""
        # Total samples across all workers
        total_samples = self.total_samples * self.num_processes
        return total_samples // global_batch_size

    def get_epoch_loader(self, epoch: int, steps_per_epoch: int,
                         num_workers: int = 8) -> 'ImageNetEpochLoader':
        """Create epoch loader with prefetching"""
        return ImageNetEpochLoader(
            dataset=self,
            epoch=epoch,
            steps_per_epoch=steps_per_epoch,
            num_workers=num_workers,
        )

    def shutdown(self):
        """Cleanup resources"""
        self.stop_event.set()
        if self.prefetch_thread is not None:
            self.prefetch_thread.join(timeout=5)


class ImageNetEpochLoader:
    """Epoch-based loader with prefetching for training loop"""

    def __init__(self, dataset: ImageNetRAMLoader, epoch: int,
                 steps_per_epoch: int, num_workers: int = 8):
        self.dataset = dataset
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers

        # Prefetch queue
        self.prefetch_queue = queue.Queue(maxsize=4)
        self.stop_event = threading.Event()

        # Start prefetch worker
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True
        )
        self.prefetch_thread.start()

    def _prefetch_worker(self):
        """Background worker for prefetching batches"""
        rng_key = jax.random.PRNGKey(self.epoch * 1000000)

        for step in range(self.steps_per_epoch):
            if self.stop_event.is_set():
                break

            try:
                rng_key, subkey = jax.random.split(rng_key)
                latents, embeddings = self.dataset.get_batch(subkey)
                self.prefetch_queue.put((latents, embeddings), timeout=60)
            except Exception as e:
                print(f"[Prefetch] Error at step {step}: {e}")
                break

        # Signal end
        self.prefetch_queue.put(None)

    def get_batches(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Iterate over batches"""
        for _ in range(self.steps_per_epoch):
            try:
                item = self.prefetch_queue.get(timeout=120)
                if item is None:
                    break
                yield item
            except queue.Empty:
                print("[Loader] Prefetch queue timeout")
                break

    def stop(self):
        """Stop prefetching"""
        self.stop_event.set()
        # Drain queue
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except queue.Empty:
                break


# =============================================================================
# Utility functions
# =============================================================================

def create_imagenet_classes_json(imagenet_dir: str, output_path: str = None):
    """Create imagenet_classes.json from ImageNet directory

    Uses synset names from folder structure. For human-readable labels,
    you would need to download the ImageNet metadata separately.

    Args:
        imagenet_dir: Path to ImageNet train directory
        output_path: Output JSON path (default: same directory as this file)
    """
    imagenet_dir = Path(imagenet_dir)
    synset_dirs = sorted([d.name for d in imagenet_dir.iterdir() if d.is_dir()])

    # Simple mapping: synset -> synset (placeholder)
    # For real labels, download from ImageNet or use a label file
    classes = {synset: synset for synset in synset_dirs}

    if output_path is None:
        output_path = Path(__file__).parent / "imagenet_classes.json"

    with open(output_path, 'w') as f:
        json.dump(classes, f, indent=2)

    print(f"Created {output_path} with {len(classes)} classes")
    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python imagenet_loader.py <imagenet_dir>")
        print("  Creates imagenet_classes.json from directory structure")
        sys.exit(1)

    create_imagenet_classes_json(sys.argv[1])
