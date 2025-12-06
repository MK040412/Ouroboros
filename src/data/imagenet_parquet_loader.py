"""
ImageNet Parquet DataLoader with JAX VAE Encoding for TPU Training

Loads ImageNet from HuggingFace Parquet format:
- gs://rdy-tpu-data-2025/imagenet-1k/data/train-*.parquet
- Schema: {image: {bytes, path}, label: int64}

Key features:
1. Load Parquet files from GCS directly
2. Decode JPEG images on-the-fly
3. Encode with FlaxAutoencoderKL (TPU optimized)
4. Cache class name embeddings
"""

import os
import io
import queue
import threading
from typing import Dict, List, Optional, Tuple, Iterator
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
import pyarrow.parquet as pq
from google.cloud import storage

# ImageNet class labels (synset -> label mapping)
IMAGENET_LABELS: Optional[Dict[int, str]] = None


def load_imagenet_labels_from_gcs(gcs_path: str = "gs://rdy-tpu-data-2025/imagenet-1k/classes.py") -> Dict[int, str]:
    """Load ImageNet class labels from GCS classes.py file"""
    global IMAGENET_LABELS
    if IMAGENET_LABELS is not None:
        return IMAGENET_LABELS

    print(f"[ImageNet] Loading class labels from {gcs_path}...")

    try:
        # Parse GCS path
        if gcs_path.startswith("gs://"):
            parts = gcs_path[5:].split("/", 1)
            bucket_name = parts[0]
            blob_path = parts[1] if len(parts) > 1 else ""

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            content = blob.download_as_text()

            # Execute the classes.py to get IMAGENET2012_CLASSES
            local_dict = {}
            exec(content, local_dict)

            synset_to_label = local_dict.get('IMAGENET2012_CLASSES', {})

            # Convert synset -> idx mapping
            # Synsets are ordered, so index = position in ordered dict
            IMAGENET_LABELS = {}
            for idx, (synset, label) in enumerate(synset_to_label.items()):
                IMAGENET_LABELS[idx] = label

            print(f"[ImageNet] Loaded {len(IMAGENET_LABELS)} class labels")
        else:
            # Local file
            with open(gcs_path, 'r') as f:
                content = f.read()
            local_dict = {}
            exec(content, local_dict)
            synset_to_label = local_dict.get('IMAGENET2012_CLASSES', {})
            IMAGENET_LABELS = {idx: label for idx, (_, label) in enumerate(synset_to_label.items())}

    except Exception as e:
        print(f"[ImageNet] Warning: Could not load class labels: {e}")
        # Fallback: use class indices as labels
        IMAGENET_LABELS = {i: f"class_{i}" for i in range(1000)}

    return IMAGENET_LABELS


class FlaxVAEEncoder:
    """JAX/Flax VAE encoder wrapper for TPU"""

    def __init__(self,
                 model_id: str = "KMK040412/sdxl-vae-flax-msgpack",
                 dtype: jnp.dtype = jnp.bfloat16,
                 scaling_factor: float = 0.13025):
        self.model_id = model_id
        self.dtype = dtype
        self.scaling_factor = scaling_factor

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
        )

        @jax.jit
        def encode_fn(params, images):
            """Encode images to latents (B, H, W, C) -> (B, H//8, W//8, 4) NHWC"""
            images_nchw = jnp.transpose(images, (0, 3, 1, 2))
            latent_dist = self.vae.apply(
                {"params": params},
                images_nchw,
                method=self.vae.encode
            )
            latents = latent_dist.latent_dist.mode()
            # FlaxAutoencoderKL returns NHWC, keep it as is
            latents = latents * self.scaling_factor
            return latents

        self._encode_fn = encode_fn
        print(f"[VAE] Model loaded and JIT compiled")

    def encode(self, images: jnp.ndarray) -> jnp.ndarray:
        """Encode images to latents"""
        self._load_model()
        images = images * 2.0 - 1.0  # [0,1] -> [-1,1]
        images = images.astype(self.dtype)
        return self._encode_fn(self.vae_params, images)


class ImageNetParquetLoader:
    """Load ImageNet from HuggingFace Parquet format on GCS"""

    def __init__(self,
                 gcs_bucket: str = "gs://rdy-tpu-data-2025/imagenet-1k/data/",
                 batch_size: int = 64,
                 image_size: int = 256,
                 vae_model_id: str = "KMK040412/sdxl-vae-flax-msgpack",
                 embedding_dim: int = 640,
                 num_workers: int = 8,
                 shard_data: bool = True,
                 use_vae: bool = True):
        """
        Args:
            gcs_bucket: GCS path to parquet files (e.g., gs://bucket/imagenet-1k/data/)
            batch_size: Batch size per device
            image_size: Target image size (256 for XUT)
            vae_model_id: HuggingFace VAE model ID
            embedding_dim: Text embedding dimension
            num_workers: Number of image decoding workers
            shard_data: Whether to shard parquet files across workers
            use_vae: Whether to encode with VAE (set False for testing without VAE)
        """
        import sys

        self.gcs_bucket = gcs_bucket.rstrip('/')
        self.batch_size = batch_size
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.num_workers = num_workers
        self.use_vae = use_vae

        # Worker info
        self.process_index = int(os.environ.get('JAX_PROCESS_INDEX', '0'))
        self.num_processes = int(os.environ.get('JAX_NUM_PROCESSES', '1'))

        print(f"\n[ImageNet] Initializing Parquet loader (Worker {self.process_index}/{self.num_processes})")
        sys.stdout.flush()

        # Parse GCS path
        if self.gcs_bucket.startswith("gs://"):
            parts = self.gcs_bucket[5:].split("/", 1)
            self.bucket_name = parts[0]
            self.prefix = parts[1] + "/" if len(parts) > 1 else ""
        else:
            raise ValueError("gcs_bucket must start with gs://")

        # GCS client
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

        # List parquet files
        self.parquet_files = self._list_parquet_files()

        # Shard parquet files across workers
        if shard_data and self.num_processes > 1:
            self.parquet_files = [f for i, f in enumerate(self.parquet_files)
                                 if i % self.num_processes == self.process_index]
            print(f"[ImageNet] Worker {self.process_index}: {len(self.parquet_files)} parquet files (sharded)")

        # Count total samples
        self.total_samples = self._count_samples()
        print(f"[ImageNet] Total samples for this worker: {self.total_samples:,}")
        sys.stdout.flush()

        # Load class labels
        self.labels = load_imagenet_labels_from_gcs(
            f"gs://{self.bucket_name}/{self.prefix.rstrip('data/')}/classes.py"
        )

        # VAE encoder (lazy loaded)
        if use_vae:
            self.vae = FlaxVAEEncoder(model_id=vae_model_id, dtype=jnp.bfloat16)
        else:
            self.vae = None

        # Class embeddings cache
        self.class_embeddings: Optional[Dict[int, np.ndarray]] = None

        # Prefetch queue
        self.prefetch_queue: Optional[queue.Queue] = None
        self.stop_event = threading.Event()

    def _list_parquet_files(self) -> List[str]:
        """List all train parquet files from GCS"""
        blobs = self.bucket.list_blobs(prefix=self.prefix)
        files = []
        for blob in blobs:
            if blob.name.endswith('.parquet') and 'train-' in blob.name:
                files.append(f"gs://{self.bucket_name}/{blob.name}")
        files = sorted(files)
        print(f"[ImageNet] Found {len(files)} train parquet files")
        return files

    def _count_samples(self) -> int:
        """Count total samples across all parquet files (approximate)"""
        # Each parquet file has ~4358 samples (1.28M / 294 files)
        return len(self.parquet_files) * 4358

    def _load_parquet_file(self, gcs_path: str) -> pq.ParquetFile:
        """Load parquet file from GCS"""
        # Download to memory
        if gcs_path.startswith("gs://"):
            parts = gcs_path[5:].split("/", 1)
            blob_path = parts[1]
            blob = self.bucket.blob(blob_path)
            content = blob.download_as_bytes()
            return pq.ParquetFile(io.BytesIO(content))
        else:
            return pq.ParquetFile(gcs_path)

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode JPEG bytes to numpy array"""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # Center crop to square
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))

            # Resize to target size
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)

            # Convert to float32 [0, 1]
            return np.array(img, dtype=np.float32) / 255.0

        except Exception as e:
            print(f"[ImageNet] Image decode error: {e}")
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

    def _compute_class_embeddings(self) -> Dict[int, np.ndarray]:
        """Load precomputed class embeddings from GCS"""
        if self.class_embeddings is not None:
            return self.class_embeddings

        # GCS에서 사전 계산된 임베딩 다운로드
        embeddings_blob_path = "imagenet-1k/imagenet_class_embeddings.npy"
        embeddings_gcs_path = f"gs://{self.bucket_name}/{embeddings_blob_path}"
        print(f"[ImageNet] Loading precomputed embeddings from {embeddings_gcs_path}...")

        local_path = "/tmp/imagenet_class_embeddings.npy"

        try:
            blob = self.bucket.blob(embeddings_blob_path)
            blob.download_to_filename(local_path)
            embeddings = np.load(local_path)  # (1000, 640)

            # Dict로 변환
            self.class_embeddings = {}
            for class_idx in range(min(1000, embeddings.shape[0])):
                self.class_embeddings[class_idx] = embeddings[class_idx]

            # Update embedding_dim based on actual embeddings
            self.embedding_dim = embeddings.shape[1]

            print(f"[ImageNet] Loaded {len(self.class_embeddings)} class embeddings (dim={self.embedding_dim})")

        except Exception as e:
            print(f"[ImageNet] Failed to load precomputed embeddings: {e}")
            print("[ImageNet] Using random embeddings as fallback")

            # Fallback: 랜덤 임베딩
            self.class_embeddings = {}
            for class_idx in range(1000):
                emb = np.random.randn(self.embedding_dim).astype(np.float32)
                emb = emb / np.maximum(np.linalg.norm(emb), 1e-8)
                self.class_embeddings[class_idx] = emb

        return self.class_embeddings

    def _get_embeddings_batch(self, class_indices: List[int]) -> np.ndarray:
        """Get embeddings for a batch of class indices"""
        embeddings = self._compute_class_embeddings()
        return np.stack([embeddings[idx] for idx in class_indices], axis=0)

    def get_batch(self, rng_key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get single batch with VAE encoding

        Returns:
            latents: (B, 32, 32, 4) NHWC, bfloat16
            embeddings: (B, 640) bfloat16
        """
        # Randomly select a parquet file
        file_idx = int(jax.random.randint(rng_key, (), 0, len(self.parquet_files)))
        rng_key, subkey = jax.random.split(rng_key)

        parquet_path = self.parquet_files[file_idx]

        # Load parquet file
        pf = self._load_parquet_file(parquet_path)
        table = pf.read()
        num_rows = table.num_rows

        # Random sample indices
        indices = np.array(jax.random.randint(subkey, (self.batch_size,), 0, num_rows))

        # Get batch data
        batch_data = table.take(indices).to_pydict()

        # Decode images in parallel
        image_bytes_list = [img['bytes'] for img in batch_data['image']]
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            images = list(executor.map(self._decode_image, image_bytes_list))

        images = np.stack(images, axis=0)
        class_indices = batch_data['label']

        # Encode with VAE or return raw images
        if self.use_vae and self.vae is not None:
            images_jax = jnp.array(images)
            latents = self.vae.encode(images_jax)
        else:
            # Without VAE: return images directly (for testing)
            latents = jnp.array(images)

        # Get class embeddings
        embeddings = self._get_embeddings_batch(class_indices)
        embeddings = jnp.array(embeddings, dtype=jnp.bfloat16)

        return latents, embeddings

    def calculate_steps_per_epoch(self, global_batch_size: int) -> int:
        """Calculate steps per epoch"""
        total_samples = self.total_samples * self.num_processes
        return total_samples // global_batch_size

    def get_epoch_loader(self, epoch: int, steps_per_epoch: int,
                         num_workers: int = 8) -> 'ImageNetParquetEpochLoader':
        """Create epoch loader with prefetching"""
        return ImageNetParquetEpochLoader(
            dataset=self,
            epoch=epoch,
            steps_per_epoch=steps_per_epoch,
            num_workers=num_workers,
        )

    def shutdown(self):
        """Cleanup resources"""
        self.stop_event.set()


class ImageNetParquetEpochLoader:
    """Epoch-based loader with prefetching"""

    def __init__(self, dataset: ImageNetParquetLoader, epoch: int,
                 steps_per_epoch: int, num_workers: int = 8):
        self.dataset = dataset
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers

        self.prefetch_queue = queue.Queue(maxsize=4)
        self.stop_event = threading.Event()

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
                # Block until queue has space (check stop_event periodically)
                while not self.stop_event.is_set():
                    try:
                        self.prefetch_queue.put((latents, embeddings), timeout=10)
                        break
                    except queue.Full:
                        continue  # Retry until queue has space
            except Exception as e:
                print(f"[Prefetch] Error at step {step}: {e}")
                import traceback
                traceback.print_exc()
                break

        # Signal end of data
        while not self.stop_event.is_set():
            try:
                self.prefetch_queue.put(None, timeout=10)
                break
            except queue.Full:
                continue

    def get_batches(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Iterate over batches"""
        for _ in range(self.steps_per_epoch):
            try:
                item = self.prefetch_queue.get(timeout=180)
                if item is None:
                    break
                yield item
            except queue.Empty:
                print("[Loader] Prefetch queue timeout")
                break

    def stop(self):
        """Stop prefetching"""
        self.stop_event.set()
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except queue.Empty:
                break


# =============================================================================
# RAM Preload variant (load all parquet to RAM at startup)
# =============================================================================
class ImageNetParquetRAMLoader(ImageNetParquetLoader):
    """Load all ImageNet parquet files to RAM at startup

    Memory usage: ~140GB for full ImageNet (JPEG bytes)
    Use when: Workers have enough RAM and want zero GCS latency during training
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # RAM storage
        self.all_images: List[bytes] = []
        self.all_labels: List[int] = []

        # Preload all data
        self._preload_all()

    def _preload_all(self):
        """Load all parquet files to RAM"""
        import sys

        print(f"\n[RAMPreload] Loading {len(self.parquet_files)} parquet files to RAM...")
        sys.stdout.flush()

        start_time = time.time()

        for i, parquet_path in enumerate(self.parquet_files):
            if (i + 1) % 10 == 0:
                print(f"  Loading file {i+1}/{len(self.parquet_files)}...")
                sys.stdout.flush()

            pf = self._load_parquet_file(parquet_path)
            table = pf.read()
            data = table.to_pydict()

            for img, label in zip(data['image'], data['label']):
                self.all_images.append(img['bytes'])
                self.all_labels.append(label)

        elapsed = time.time() - start_time
        print(f"[RAMPreload] Loaded {len(self.all_images):,} samples in {elapsed:.1f}s")

        # Update total samples
        self.total_samples = len(self.all_images)

    def get_batch(self, rng_key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get batch from RAM"""
        # Random sample indices
        indices = np.array(jax.random.randint(rng_key, (self.batch_size,), 0, self.total_samples))

        # Get image bytes and labels
        image_bytes_list = [self.all_images[i] for i in indices]
        class_indices = [self.all_labels[i] for i in indices]

        # Decode images in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            images = list(executor.map(self._decode_image, image_bytes_list))

        images = np.stack(images, axis=0)

        # Encode with VAE
        if self.use_vae and self.vae is not None:
            images_jax = jnp.array(images)
            latents = self.vae.encode(images_jax)
        else:
            latents = jnp.array(images)

        # Get class embeddings
        embeddings = self._get_embeddings_batch(class_indices)
        embeddings = jnp.array(embeddings, dtype=jnp.bfloat16)

        return latents, embeddings


if __name__ == "__main__":
    # Test loading
    print("Testing ImageNet Parquet Loader...")

    loader = ImageNetParquetLoader(
        gcs_bucket="gs://rdy-tpu-data-2025/imagenet-1k/data/",
        batch_size=4,
        image_size=256,
        use_vae=False,  # Skip VAE for quick test
    )

    print(f"\nTotal samples: {loader.total_samples}")
    print(f"Parquet files: {len(loader.parquet_files)}")

    # Get one batch
    rng_key = jax.random.PRNGKey(0)
    images, embeddings = loader.get_batch(rng_key)

    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Embeddings: {embeddings.shape}")
