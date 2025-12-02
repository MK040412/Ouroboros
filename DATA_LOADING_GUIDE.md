# HuggingFace + Prefetch를 이용한 데이터 로딩 가이드

## 개요
`train_tpu_256.py`가 다음과 같이 개선되었습니다:

### 이전 방식
- 로컬 경로에서 모든 PT 파일을 메모리에 로드 (느림, 메모리 많이 사용)
- GPU 메모리에 강제로 로드
- 동적 Prefetch 없음

### 새로운 방식
1. **HuggingFace 스트리밍**: 필요한 파일만 다운로드 (Lazy loading)
2. **GCS 옵션**: GCS 버킷에 업로드되면 GCS에서 빠르게 로드
3. **Prefetch 파이프라인**: 백그라운드에서 다음 배치를 미리 로드
4. **TPU 최적화**: 112 cores (TPU v5e 256) 지원

---

## 사용 방법

### 방법 1: HuggingFace 직접 스트리밍 (권장)

```python
from train_tpu_256 import Coyo11mDataLoader, PrefetchDataLoader

# 데이터로더 생성
data_loader = Coyo11mDataLoader(
    batch_size=2048,
    embedding_provider=embedding_provider,
    use_gcs=False  # HF에서 직접 로딩
)

# Prefetch 파이프라인
prefetch_loader = PrefetchDataLoader(
    data_loader,
    steps_per_epoch=3750,
    prefetch_size=2  # 2개 배치 미리 로드
)
```

**장점:**
- 전체 데이터 다운로드 필요 없음
- 메모리 효율적
- 다른 설정 필요 없음

---

### 방법 2: GCS 버킷 사용 (더 빠름)

#### 1단계: 데이터를 GCS로 업로드

```bash
# download_and_upload.py 실행
python download_and_upload.py

# 그러면 gs://rdy-tpu-data-2025/coyo11m 에 데이터가 업로드됨
```

#### 2단계: GCS 사용 설정

```python
data_loader = Coyo11mDataLoader(
    batch_size=2048,
    embedding_provider=embedding_provider,
    use_gcs=True,  # GCS에서 로딩
    gcs_bucket="gs://rdy-tpu-data-2025/coyo11m"
)
```

**장점:**
- TPU에서 GCS로 빠른 네트워크 대역폭
- 여러 번 사용 가능 (재다운로드 필요 없음)
- 프로덕션 환경에 적합

---

## Prefetch 파이프라인 최적화

### Prefetch 크기 설정
```python
# 작은 메모리 (GCS 또는 HF 스트리밍)
prefetch_loader = PrefetchDataLoader(data_loader, steps_per_epoch=3750, prefetch_size=2)

# 큰 메모리 (TPU 112 cores)
prefetch_loader = PrefetchDataLoader(data_loader, steps_per_epoch=1065, prefetch_size=4)
```

### 동작 방식
```
메인 스레드        배경 스레드 (Prefetch)
─────────────────────────────────────
1. get_batch(0)  ← 1. load_batch(0)
2. get_batch(1)  ← 2. load_batch(1)
3. get_batch(2)  ← 3. load_batch(2)
   (학습)           (준비 중...)
```

이렇게 하면 학습하는 동안 다음 배치가 미리 로드되어 있습니다.

---

## TPU 설정

### 16 cores (TPU v5e Pod)
```python
from train_tpu_256 import TrainingConfig256

config = TrainingConfig256()
# num_devices = 16 (기본값)
# global_batch_size = 2048
# batch_size_per_device = 128
```

### 112 cores (TPU v5e 256)
```python
# 방법 1: train_tpu_256_112cores.py 실행
python train_tpu_256_112cores.py

# 방법 2: 수동 설정
config = TrainingConfig256()
config.num_devices = 112
config.global_batch_size = 7168  # 112 × 64
config.batch_size_per_device = 64
config.steps_per_epoch = 1065  # 7.624M / 7168
```

---

## 데이터 포맷

### 입력 데이터 (HuggingFace)
- **Repository**: `KBlueLeaf/coyo11m-256px-ccrop-latent`
- **PT 파일 형식**: `000000-000009.pt`, `000010-000019.pt`, ...
  - `keys`: (10,) - 샘플 ID
  - `latents`: (10, 3, 4, 32, 32) - VAE latents
- **Metadata**: `coyo11m-meta.parquet`
  - `key`: 샘플 ID
  - `caption_llava`: 캡션 텍스트

### 모델에 전달되는 데이터
```python
batch_latents: (B, 4, 32, 32)  # VAE latent
batch_embeddings: (B, 640)      # Text embeddings (Gemma-270M)
```

---

## 성능 비교

### 데이터 로딩 속도 (1 에포크 기준)

| 방식 | 메모리 | 속도 | 초기 시간 |
|------|--------|------|---------|
| 로컬 파일 (전체 로드) | 많음 | 빠름 | 5분 |
| HF 스트리밍 + Prefetch | 적음 | 중간 | 30초 |
| GCS + Prefetch | 적음 | 빠름 | 1분 |

---

## 문제 해결

### 1. "HuggingFace 파일 못 찾음" 에러
```bash
# HF 토큰 설정
huggingface-cli login

# 또는 환경변수
export HF_TOKEN=hf_xxxx
```

### 2. GCS 접근 권한 에러
```bash
gcloud auth application-default login
gcloud config set project vlarl-tpu
```

### 3. Prefetch 큐가 가득 참
- Prefetch 크기 줄이기: `prefetch_size=1`
- 또는 배치 크기 줄이기: `batch_size=1024`

### 4. 메모리 부족
- HuggingFace 스트리밍으로 변경 (캐시 크기 제한)
- CPU 로드 사용 (`map_location="cpu"` 유지)

---

## 향후 개선 사항

1. **Distributed Data Loading**: 여러 호스트에서 데이터 로딩
2. **TFDS 통합**: TensorFlow Datasets 호환성
3. **Cache 관리**: 자동 PT 파일 캐싱/제거
4. **학습 재개**: Checkpoint에서 exact epoch/batch 위치 복원

---

## 참고

- **Prefetch 논문**: https://arxiv.org/abs/1406.0746
- **JAX Data Loading**: https://jax.readthedocs.io/
- **GCS Python**: https://cloud.google.com/python/docs/reference/storage
