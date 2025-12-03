#!/usr/bin/env python3
"""
GCS DataLoader 테스트 스크립트
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gcs_handler():
    """GCS 핸들러 테스트"""
    print("\n" + "="*70)
    print("Testing GCSFileHandler")
    print("="*70)
    
    try:
        from src.data.gcs_dataloader import GCSFileHandler
        
        handler = GCSFileHandler()
        logger.info(f"GCS Available: {handler.gcs_available}")
        
        if handler.gcs_available:
            pt_files = handler.list_pt_files()
            logger.info(f"Found {len(pt_files)} PT files")
            if pt_files:
                logger.info(f"First 3 files: {pt_files[:3]}")
                logger.info(f"Last 3 files: {pt_files[-3:]}")
            return True
        else:
            logger.warning("GCS not available - this is expected if not on TPU")
            return True
    except Exception as e:
        logger.error(f"Error testing GCSFileHandler: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parquet_cache():
    """Parquet 캐시 테스트"""
    print("\n" + "="*70)
    print("Testing ParquetCache")
    print("="*70)
    
    try:
        from src.data.gcs_dataloader import ParquetCache
        
        # 로컬 parquet 파일 확인
        local_parquet = Path("/home/perelman/jax-hdm/coyo11m-meta.parquet")
        
        if not local_parquet.exists():
            logger.warning(f"Local parquet not found: {local_parquet}")
            logger.info("Skipping ParquetCache test")
            return True
        
        logger.info(f"Loading parquet from: {local_parquet}")
        cache = ParquetCache.load_from_parquet(str(local_parquet))
        
        logger.info(f"Cache size: {len(cache.all_keys)} keys")
        logger.info(f"Sample keys: {list(cache.all_keys)[:5]}")
        
        # 샘플 조회
        if cache.all_keys:
            sample_key = list(cache.all_keys)[0]
            caption = cache.key_to_caption.get(sample_key)
            logger.info(f"Sample caption for key {sample_key}: {caption[:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"Error testing ParquetCache: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gcs_coyo11m_dataloader():
    """GCS Coyo11m DataLoader 테스트"""
    print("\n" + "="*70)
    print("Testing GCSCoyo11mDataLoader")
    print("="*70)
    
    try:
        from src.data.gcs_dataloader import GCSCoyo11mDataLoader, ParquetCache
        import numpy as np
        
        # Parquet 캐시 로드
        local_parquet = Path("/home/perelman/jax-hdm/coyo11m-meta.parquet")
        if not local_parquet.exists():
            logger.warning(f"Local parquet not found: {local_parquet}")
            logger.info("Skipping GCSCoyo11mDataLoader test")
            return True
        
        logger.info("Loading parquet cache...")
        parquet_cache = ParquetCache.load_from_parquet(str(local_parquet))
        
        # Mock embedding provider
        class MockEmbeddingProvider:
            def batch_encode(self, texts, batch_size=512, normalize=True):
                return np.random.randn(len(texts), 640).astype(np.float32)
        
        # DataLoader 생성 (로컬 PT 파일 사용)
        local_pt = Path("/home/perelman/jax-hdm/000000-000009.pt")
        if not local_pt.exists():
            logger.warning(f"Local PT not found: {local_pt}")
            logger.info("Skipping GCSCoyo11mDataLoader test")
            return True
        
        logger.info(f"Loading PT from: {local_pt}")
        
        dataloader = GCSCoyo11mDataLoader(
            batch_size=128,
            parquet_cache=parquet_cache,
            embedding_provider=MockEmbeddingProvider(),
            gcs_bucket="gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/",
            cache_dir="/tmp/test_cache"
        )
        
        # PT 파일 로드
        success = dataloader.load_pt_file(str(local_pt))
        if not success:
            logger.error("Failed to load PT file")
            return False
        
        logger.info(f"PT file loaded successfully")
        logger.info(f"Available indices: {len(dataloader.available_indices)}")
        
        # 배치 테스트
        if dataloader.available_indices:
            import jax
            rng_key = jax.random.PRNGKey(0)
            batch_latents, batch_embeddings = dataloader.get_batch(0, rng_key)
            
            logger.info(f"Batch latents shape: {batch_latents.shape}")
            logger.info(f"Batch embeddings shape: {batch_embeddings.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing GCSCoyo11mDataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_cache_manager():
    """ParallelCacheManager 테스트"""
    print("\n" + "="*70)
    print("Testing ParallelCacheManager")
    print("="*70)
    
    try:
        from src.data.gcs_dataloader import ParallelCacheManager
        import tempfile
        
        cache_dir = tempfile.mkdtemp()
        logger.info(f"Cache directory: {cache_dir}")
        
        manager = ParallelCacheManager(num_workers=4, cache_dir=cache_dir)
        logger.info(f"Created ParallelCacheManager with 4 workers")
        
        manager.shutdown()
        logger.info("Manager shutdown successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error testing ParallelCacheManager: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gcs_dataloader_session():
    """GCSDataLoaderSession 테스트"""
    print("\n" + "="*70)
    print("Testing GCSDataLoaderSession")
    print("="*70)
    
    try:
        from src.data.gcs_dataloader import GCSDataLoaderSession
        import numpy as np
        
        # Mock embedding provider
        class MockEmbeddingProvider:
            def batch_encode(self, texts, batch_size=512, normalize=True):
                return np.random.randn(len(texts), 640).astype(np.float32)
        
        # Parquet 확인
        local_parquet = Path("/home/perelman/jax-hdm/coyo11m-meta.parquet")
        if not local_parquet.exists():
            logger.warning(f"Local parquet not found: {local_parquet}")
            logger.info("Skipping GCSDataLoaderSession test")
            return True
        
        logger.info("Initializing GCSDataLoaderSession...")
        
        session = GCSDataLoaderSession(
            batch_size=128,
            parquet_path=str(local_parquet),
            embedding_provider=MockEmbeddingProvider(),
            gcs_bucket="gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/",
            cache_dir="/tmp/test_session_cache",
            num_workers=4,  # 테스트용 4 workers
            prefetch_ahead=1
        )
        
        logger.info(f"Session initialized")
        logger.info(f"PT files: {len(session.pt_files)}")
        
        session.shutdown()
        logger.info("Session shutdown successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error testing GCSDataLoaderSession: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """모든 테스트 실행"""
    print("="*70)
    print("GCS DataLoader Test Suite")
    print("="*70)
    
    tests = [
        ("GCSFileHandler", test_gcs_handler),
        ("ParquetCache", test_parquet_cache),
        ("GCSCoyo11mDataLoader", test_gcs_coyo11m_dataloader),
        ("ParallelCacheManager", test_parallel_cache_manager),
        ("GCSDataLoaderSession", test_gcs_dataloader_session),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "✓ PASS" if result else "✗ FAIL"
        except Exception as e:
            logger.error(f"Unexpected error in {test_name}: {e}")
            results[test_name] = "✗ ERROR"
    
    # 결과 요약
    print("\n" + "="*70)
    print("Test Results Summary")
    print("="*70)
    for test_name, result in results.items():
        print(f"{test_name:30s} {result}")
    
    passed = sum(1 for r in results.values() if "PASS" in r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
