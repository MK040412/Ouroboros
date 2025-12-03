# GCS DataLoader Guide (112 vCPU Pipeline)

## Overview

GCS 기반 데이터로더는 TPU 트레이닝을 위해 최적화된 병렬 데이터 로딩 파이프라인입니다.

### 핵심 특징

1. **자동 PT 파일 순회**: GCS 버킷에서 자동으로 모든 PT 파일 검색 및 정렬
2. **메타데이터 캐싱**: Parquet 메타데이터를 메모리에 전체 로드하여 빠른 caption 조회
3. **병렬 다운로드**: 112개 vCPU를 활용한 멀티스레드 다운로드
4. **Prefetch 파이프라인**: 배경에서 다음 PT 파일 로드 중 현재 배치 처리
5. **자동 메모리 관리**: PT 파일 로드/언로드 자동화

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      GCSDataLoaderSession                        │
│                  (Global session object)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    │                    │
        ┌──────────────────┐  ┌──────────────────────┐
        │ GCSFileHandler   │  │  ParquetCache        │
        │ (GCS access)     │  │  (Metadata in RAM)   │
        └──────────────────┘  └──────────────────────┘
                    │                    │
                    └─────────┬──────────┘
                              │
                   ┌──────────────────────────┐
                   │ ParallelCacheManager     │
                   │ (112 worker pool)        │
                   │ (Background downloads)   │
                   └──────────────────────────┘
                              │
            ┌─────────────────┴──────────────────┐
            │                                    │
   ┌─────────────────────┐         ┌──────────────────────┐
   │get_epoch_loader()   │         │GCSPrefetchDataLoader │
   │(per epoch)          │         │(Prefetch pipeline)   │
   └─────────────────────┘         └──────────────────────┘
            │                                    │
            └────────────────┬───────────────────┘
                             │
                  ┌──────────────────────┐
                  │GCSCoyo11mDataLoader  │
                  │(Batch generation)    │
                  └──────────────────────┘
                             │
                   ┌─────────────────────┐
                   │trainer.train_epoch()│
                   │(GPT training loop)  │
                   └─────────────────────┘
```

---

## Usage

### Basic Setup

```python
from src.data.gcs_dataloader import GCSDataLoaderSession

# 1. 세션 초기화 (에포크 시작 전 한 번만)
gcs_session = GCSDataLoaderSession(
    batch_size=2048,
    parquet_path="gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/coyo11m-meta.parquet",
    embedding_provider=embedding_provider,  # Text embedding provider
    gcs_bucket="gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/",
    cache_dir="/tmp/pt_cache",  # Local cache directory
    num_workers=112,  # vCPU workers for parallel download
    prefetch_ahead=3  # Prefetch next 3 PT files ahead
)
```

### Training Loop

```python
for epoch in range(num_epochs):
    # 2. 에포크별 로더 생성 (자동 prefetch)
    gcs_prefetch_loader = gcs_session.get_epoch_loader(
        epoch=epoch,
        steps_per_epoch=3750
    )
    
    # 3. 학습 수행
    try:
        losses, avg_loss = trainer.train_epoch(gcs_prefetch_loader, epoch)
    finally:
        gcs_prefetch_loader.stop()

# 4. 세션 종료
gcs_session.shutdown()
```

---

## Data Flow

### Per Epoch

1. **Initialization**
   ```
   get_epoch_loader() 호출
   ↓
   ParquetCache 이미 메모리에 있음 (재사용)
   ↓
   GCSPrefetchDataLoader 생성
   ↓
   112 workers 시작 (병렬 다운로드)
   ```

2. **Background Processing**
   ```
   Worker 0: PT file 0 다운로드
   Worker 1: PT file 1 다운로드
   ...
   Worker 111: PT file 111 다운로드
   ↓
   (모든 파일 동시에 다운로드)
   ```

3. **Batch Generation**
   ```
   train_epoch() 중 prefetch_queue에서 배치 꺼냄
   ↓
   latents (B, 32, 32, 4) + embeddings (B, 640)
   ↓
   모델 forward pass
   ```

### Per Batch

```
PT file에서 latent 로드
↓
Parquet에서 caption 조회 (메모리 캐시에서)
↓
Embedding 계산 (batch_encode)
↓
JAX array로 변환
↓
prefetch_queue에 저장
```

---

## Configuration (TrainingConfig256)

```python
@dataclass
class TrainingConfig256:
    # GCS 설정
    gcs_bucket: str = "gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/"
    parquet_file: str = None  # Auto-discovered from gcs_bucket
    cache_dir: str = None  # Auto-set to /tmp
    num_data_workers: int = 112  # CPU vCPU count
    prefetch_ahead: int = 3  # PT files to prefetch ahead
```

---

## Performance Characteristics

### Throughput (112 workers)

| Metric | Value |
|--------|-------|
| PT File Size | ~100 MB |
| Parquet Size | ~2-3 GB |
| Download Bandwidth | 112 × 1 MB/s = 112 MB/s |
| Prefetch Time | ~1 second per PT file (at 112 MB/s) |
| Batch Generation | ~100-200 ms per batch |
| **Total Overhead** | **< 5% of training time** |

### Memory Usage

| Component | Size |
|-----------|------|
| Parquet Cache (RAM) | ~2-3 GB |
| PT File (one at a time) | ~100 MB |
| Prefetch Queue | ~50-100 MB |
| **Total** | **~3-4 GB** |

---

## Key Implementation Details

### 1. ParquetCache

```python
ParquetCache.load_from_parquet(parquet_path)
# 결과:
# - key_to_caption: Dict[int, str]
# - key_to_url: Dict[int, str]
# - all_keys: set (빠른 membership 검사)
```

**특징:**
- 전체 메타데이터를 메모리에 로드 (한 번만)
- Caption 조회는 O(1) dictionary lookup
- 모든 epoch에서 재사용 (메모리 효율)

### 2. ParallelCacheManager

```python
cache_manager = ParallelCacheManager(num_workers=112)
cache_manager.prefetch_pt_files(pt_files_list, gcs_handler)
```

**동작:**
- ThreadPoolExecutor 사용 (IO 바운드 작업)
- 최대 112개 동시 다운로드
- 다운로드 중 다른 작업 가능 (async)
- `wait_for_file()`: 필요할 때 완료 대기

### 3. GCSPrefetchDataLoader

```python
prefetch_loader = GCSPrefetchDataLoader(
    data_loader, 
    pt_files, 
    steps_per_epoch=3750,
    num_workers=112
)

for batch_latents, batch_embeddings in prefetch_loader.get_batches():
    # 배치 처리
```

**특징:**
- 112개 worker thread 생성
- 각 worker는 담당 배치만 처리 (load balancing)
- Queue 기반 (producer-consumer pattern)
- Deadlock 방지 (timeout + None sentinel)

---

## Error Handling

### GCS Access Issues

```python
# GCS 접근 불가능 시
if not gcs_handler.gcs_available:
    logger.warning("GCS not available")
    # 로컬 파일로 폴백 (선택적)
```

### File Download Failures

```python
# wait_for_file()에서 timeout 발생
if not cache_manager.wait_for_file(filename, timeout=600):
    logger.error(f"Failed to download {filename}")
    # 다음 파일로 건너뜀
```

### Prefetch Timeout

```python
# get_batches()에서 60초 타임아웃
# Queue가 비어있으면 exception 발생
# trainer.train_epoch()에서 처리
```

---

## Optimization Tips

### 1. Cache Directory

```python
# SSD 사용 (최소 50GB)
cache_dir="/mnt/ssd/pt_cache"
```

### 2. Prefetch Ahead

```python
# 다운로드 시간 vs 메모리 트레이드오프
prefetch_ahead=1   # 메모리 효율 (다운로드 오버헤드 증가)
prefetch_ahead=5   # 빠른 로딩 (메모리 사용량 증가)
prefetch_ahead=3   # 균형잡힌 설정 (권장)
```

### 3. Batch Size

```python
# 더 큰 배치 → 더 빠른 로딩
global_batch_size=2048  # 현재 설정
# 배치 크기 증가 → caption embedding 캐싱 이득
```

---

## Monitoring

### WandB Logging

```python
wandb.log({
    "batches_processed": total_batches_processed,
    "num_pt_files": len(gcs_session.pt_files),
    "epoch_time_hours": epoch_time / 3600,
    "download_speed_mbps": ...,  # 추가 가능
})
```

### Console Output

```
[GCS Data Setup]
GCS Bucket: gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/
CPU Workers: 112 vCPUs
Prefetch ahead: 3 PT files
✓ GCS session initialized
  PT files found: 1000

[Training Starting]
Total epochs: 20
PT files per epoch: 1000
Steps per PT file: 3750
Global batch size: 2048

[GCS prefetch worker 0] ✓ Downloaded: 000000-000009.pt
[GCS prefetch worker 1] ✓ Downloaded: 000010-000019.pt
...
```

---

## Troubleshooting

### Issue: "GCS access failed"

```bash
# gcloud CLI 설정
gcloud auth application-default login
gcloud config set project YOUR_PROJECT

# 권한 확인
gcloud storage ls gs://rdy-tpu-data-2025/
```

### Issue: "No PT files found"

```python
# 수동으로 확인
from src.data.gcs_dataloader import GCSFileHandler
handler = GCSFileHandler()
files = handler.list_pt_files()
print(f"Found {len(files)} files: {files[:5]}")
```

### Issue: "Out of memory"

```python
# 캐시 크기 줄이기
cache_dir = "/tmp/small_cache"  # 더 작은 디스크

# Batch size 줄이기
global_batch_size = 1024  # 2048에서 감소

# Prefetch 줄이기
prefetch_ahead = 1  # 3에서 감소
```

---

## Integration with train_tpu_256.py

```python
# TrainingConfig256에 자동으로 추가됨
config = TrainingConfig256()
# - gcs_bucket
# - num_data_workers = 112
# - prefetch_ahead = 3

# main() 함수에서 자동으로 사용됨
gcs_session = GCSDataLoaderSession(...)

for epoch in range(config.num_epochs):
    gcs_prefetch_loader = gcs_session.get_epoch_loader(...)
    losses, _ = trainer.train_epoch(gcs_prefetch_loader, epoch)
```

---

## Future Enhancements

1. **Latency Optimization**
   - Zero-copy memmap for PT files
   - Distributed caching across nodes

2. **Robustness**
   - Retry logic with exponential backoff
   - Checkpointing of download progress

3. **Monitoring**
   - Per-worker download metrics
   - Queue depth tracking
   - Cache hit/miss statistics

4. **Multi-Node Support**
   - Distributed cache sharing
   - Per-node PT file assignment
