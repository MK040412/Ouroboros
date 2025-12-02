# TPU Training Guide

## TPU 환경 설정

### 1. TPU VM 인스턴스 생성
```bash
# Google Cloud에서 TPU v4 또는 v5 인스턴스 생성
gcloud compute tpus tpu-vm create my-tpu \
  --zone=us-central1-a \
  --version=tpu-vm-tf-nightly
```

### 2. 의존성 설치
```bash
# JAX + TPU 지원
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Flax & Optax
pip install flax optax

# 다른 의존성
pip install numpy
```

### 3. 데이터 준비 (로컬)
```python
import torch
import numpy as np

# .pt 파일을 NPZ로 변환
pt_file = "000000-000009.pt"
data = torch.load(pt_file, map_location="cpu")
latents = data['latents'][:100, 0, :, :, :]  # (100, 4, 32, 32)

# float32로 변환
latents_fp32 = latents.float().numpy()

# NPZ로 저장
np.savez("data.npz", latents=latents_fp32)
```

### 4. Cloud Storage에 업로드
```bash
gsutil cp data.npz gs://your-bucket/data.npz
```

## TPU에서 학습 실행

### 방법 1: 로컬 TPU (자동 감지)
```python
# JAX가 자동으로 TPU 감지
# test_train.py 그냥 실행
python test_train.py
```

### 방법 2: 명시적 TPU 설정
```python
import os
os.environ['JAX_PLATFORMS'] = 'tpu'

# 이후 코드...
```

### 방법 3: Cloud Storage에서 데이터 로드
```python
from google.cloud import storage

config = TrainingConfig(
    num_samples=100,
    batch_size=8,  # TPU 최적화
    data_path="gs://your-bucket/data.npz"  # 또는 로컬 경로
)
```

## TPU 최적화 팁

### 1. 배치 크기
```python
# TPU는 8개 코어 -> 배치 크기는 8의 배수
batch_size = 8   # ✓ Good
batch_size = 16  # ✓ Good
batch_size = 7   # ✗ Bad
```

### 2. 모델 차원
```python
# TPU 최적화 차원
model_dim = 64   # 2의 배수 + 8의 배수 권장
model_dim = 128
model_dim = 256
```

### 3. Gradient Accumulation
```python
# 큰 배치 처리를 위해
def train_step_accumulated(model, x, steps=4):
    for i in range(steps):
        loss, grads = nnx.value_and_grad(loss_fn)(model, x[i])
        optimizer.update(grads)
    return loss
```

### 4. Mixed Precision (선택)
```python
import jax.experimental.pjit as pjit

# TPU에서 자동으로 bf16 사용
# 별도 설정 불필요
```

## 다중 TPU 학습

### pmap을 사용한 분산 학습
```python
def train_step_pmap(model_params, batch):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    return loss, grads

# TPU 8개 병렬 처리
train_step_pmap = jax.pmap(train_step_pmap)
```

## 성능 모니터링

### TPU 스트림 확인
```bash
# TPU 활용률 모니터링
curl http://localhost:8431
```

### JAX 컴파일 시간 확인
```python
import jax

jax.profiler.start_trace("gs://your-bucket/trace")

# 학습 코드...

jax.profiler.stop_trace()
```

## 트러블슈팅

### 1. TPU 감지 안 됨
```python
import jax
print(jax.devices())  # TPUDevice 확인
```

### 2. Out of Memory
```python
# 배치 크기 감소
config.batch_size = 4

# 모델 크기 감소
config.model_dim = 32
config.depth = 1
```

### 3. 데이터 I/O 병목
```python
# 미리 다운로드
gsutil -m cp gs://bucket/data.npz ./

# 로컬에서 로드
config.data_path = "data.npz"
```

## 최종 설정 예시

```python
config = TrainingConfig(
    # TPU 최적화
    num_samples=1000,
    batch_size=32,       # TPU 8코어 × 4
    num_steps=100,
    
    # 모델 (TPU 친화적)
    model_dim=256,
    heads=16,
    depth=4,
    mlp_dim=1024,
    
    # 데이터
    data_path="gs://your-bucket/data.npz",
    
    # 분산학습
    use_pmap=False,  # True면 다중 TPU
)
```

## 참고
- JAX TPU 가이드: https://jax.readthedocs.io/en/latest/
- Flax 분산학습: https://flax.readthedocs.io/en/latest/guides/distributed_training.html
