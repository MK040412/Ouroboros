#!/usr/bin/env python -u
"""
각 initialization 단계를 독립적으로 테스트
"""

import sys
import time

def test_step(step_num, description, timeout=30):
    """스텝 테스트 데코레이터"""
    def decorator(func):
        def wrapper():
            print(f"\n[Test Step {step_num}] {description}")
            print(f"  Timeout: {timeout}s")
            sys.stdout.flush()
            
            start = time.time()
            try:
                result = func()
                elapsed = time.time() - start
                print(f"  ✓ SUCCESS ({elapsed:.1f}s)")
                sys.stdout.flush()
                return True
            except Exception as e:
                elapsed = time.time() - start
                print(f"  ✗ FAILED ({elapsed:.1f}s)")
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                return False
        return wrapper
    return decorator


# ============ Test Steps ============

@test_step(1, "Import JAX", timeout=10)
def test_jax_import():
    import jax
    print(f"  JAX version: {jax.__version__}")
    devices = jax.devices()
    print(f"  Devices: {len(devices)}")
    return True


@test_step(2, "Import torch", timeout=10)
def test_torch_import():
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    return True


@test_step(3, "Import JAX NNX", timeout=10)
def test_nnx_import():
    from flax import nnx
    print(f"  Flax NNX loaded")
    return True


@test_step(4, "Import Parquet", timeout=10)
def test_parquet_import():
    import pyarrow.parquet as pq
    print(f"  PyArrow loaded")
    return True


@test_step(5, "Load embedding provider config", timeout=30)
def test_embedding_config():
    from src.embeddings import get_embedding_provider
    print(f"  get_embedding_provider imported")
    return True


@test_step(6, "Create XUT-Small model (no device)", timeout=60)
def test_model_creation():
    from src.xut.xut_small import create_xut_small
    print(f"  Creating model...")
    model = create_xut_small()
    print(f"  Model created: {type(model).__name__}")
    return True


@test_step(7, "Initialize embedding provider (first load, may take time)", timeout=120)
def test_embedding_provider():
    print(f"  This may download model (~1-2 GB), please wait...")
    from src.embeddings import get_embedding_provider
    
    start = time.time()
    provider = get_embedding_provider("google/embeddinggemma-300m")
    elapsed = time.time() - start
    
    print(f"  Provider loaded in {elapsed:.1f}s")
    
    # Test encode
    test_texts = ["test caption 1", "test caption 2"]
    embeddings = provider.batch_encode(test_texts, batch_size=2, normalize=True)
    print(f"  Test encode: {embeddings.shape}")
    
    return True


@test_step(8, "Load Parquet metadata from local file", timeout=60)
def test_parquet_load():
    from pathlib import Path
    from src.data.gcs_dataloader import ParquetCache
    
    parquet_path = Path("/home/perelman/jax-hdm/coyo11m-meta.parquet")
    if not parquet_path.exists():
        print(f"  WARNING: Parquet file not found: {parquet_path}")
        print(f"  Skipping parquet load test")
        return True
    
    print(f"  Loading parquet: {parquet_path}")
    cache = ParquetCache.load_from_parquet(str(parquet_path))
    print(f"  Loaded {len(cache.all_keys)} keys")
    return True


@test_step(9, "Load PT file from local", timeout=60)
def test_pt_load():
    from pathlib import Path
    import torch
    
    pt_path = Path("/home/perelman/jax-hdm/000000-000009.pt")
    if not pt_path.exists():
        print(f"  WARNING: PT file not found: {pt_path}")
        print(f"  Skipping PT load test")
        return True
    
    print(f"  Loading PT: {pt_path}")
    pt_data = torch.load(str(pt_path), map_location="cpu")
    print(f"  Keys in PT: {list(pt_data.keys())}")
    print(f"  Latents shape: {pt_data['latents'].shape}")
    print(f"  Keys array size: {pt_data['keys'].shape}")
    return True


@test_step(10, "Create GCSDataLoaderSession (with local files)", timeout=120)
def test_gcs_session():
    from pathlib import Path
    from src.data.gcs_dataloader import GCSDataLoaderSession
    from src.embeddings import get_embedding_provider
    import numpy as np
    
    parquet_path = Path("/home/perelman/jax-hdm/coyo11m-meta.parquet")
    
    if not parquet_path.exists():
        print(f"  WARNING: Parquet not found, skipping")
        return True
    
    print(f"  Creating session...")
    
    # Mock embedding provider
    class MockProvider:
        def batch_encode(self, texts, batch_size=512, normalize=True):
            return np.random.randn(len(texts), 640).astype(np.float32)
    
    try:
        session = GCSDataLoaderSession(
            batch_size=128,
            parquet_path=str(parquet_path),
            embedding_provider=MockProvider(),
            gcs_bucket="gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/",
            cache_dir="/tmp/test_cache",
            num_workers=4,  # Use fewer workers for testing
            prefetch_ahead=1,
            max_cache_files=1
        )
        
        print(f"  Session created")
        print(f"  PT files found: {len(session.pt_files)}")
        
        session.shutdown()
        print(f"  Session shutdown")
        
        return True
    except Exception as e:
        print(f"  Error creating session: {e}")
        raise


def main():
    print("="*70)
    print("JAX-HDM Training Initialization Test Suite")
    print("="*70)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()
    
    tests = [
        test_jax_import,
        test_torch_import,
        test_nnx_import,
        test_parquet_import,
        test_embedding_config,
        test_model_creation,
        test_embedding_provider,
        test_parquet_load,
        test_pt_load,
        test_gcs_session,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except KeyboardInterrupt:
            print("\n\n✗ Interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            results.append((test.__name__, False))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:40s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
