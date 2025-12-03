# Training Debug & Troubleshooting Guide

## 문제: "다 노는데?" (CPU 0% usage)

학습이 시작되지 않고 CPU가 사용되지 않는 경우 대응 방법입니다.

## Step 1: 각 initialization 단계 테스트

먼저 각 단계를 개별적으로 테스트하여 어디서 멈추는지 파악합니다:

```bash
python -u test_initialization_steps.py
```

이 스크립트는 다음을 테스트합니다:
1. JAX import
2. PyTorch import
3. Flax NNX import
4. PyArrow/Parquet import
5. Embedding provider 로드
6. XUT-Small 모델 생성
7. **Embedding provider 초기화** ← 가장 오래 걸림 (1-2 GB 다운로드)
8. Parquet 메타데이터 로드
9. PT 파일 로드
10. GCSDataLoaderSession 생성

### 각 단계별 예상 시간

```
Step 1-5: 즉시 (<1s)
Step 6: 1-2s (모델 생성)
Step 7: 60-120s (embedding model 다운로드, 첫 실행만)
Step 8-9: 30-60s (로컬 파일 로드)
Step 10: 60-120s (GCS 메타데이터 확인)
```

## Step 2: 진행 상황 모니터링

다른 터미널에서 실시간 모니터링:

```bash
# 디버그 로그 (실시간)
tail -f /tmp/train_debug.log

# CPU/메모리 상태 (실시간)
htop

# 디스크 사용량
watch -n 1 'du -sh /tmp/gcs_pt_cache/'

# 네트워크 (모델 다운로드 확인)
iftop  # or nethogs
```

## Step 3: 일반적인 문제 해결

### 문제 1: "Embedding provider" 단계에서 멈춤

**증상:**
```
[Test Step 7] Initialize embedding provider
  This may download model (~1-2 GB), please wait...
```
여기서 1-2분 이상 멈춤

**원인:**
- 네트워크 느림
- 모델 다운로드 중
- 디스크 공간 부족

**해결:**
```bash
# 다운로드 진행 확인
watch -n 1 'ls -lah ~/.cache/huggingface/'

# 디스크 확인
df -h /home/

# 네트워크 확인
ping huggingface.co
```

### 문제 2: "GCSDataLoaderSession" 단계에서 멈춤

**증상:**
```
[Test Step 10] Create GCSDataLoaderSession
  Creating session...
```
여기서 멈춤

**원인:**
- GCS 접근 권한 없음
- Parquet 파일이 로컬에 없음
- 네트워크 연결 끊김

**해결:**
```bash
# GCS 권한 확인
gcloud auth application-default login
gcloud storage ls gs://rdy-tpu-data-2025/coyo11m-256px-ccrop-latent/

# 로컬 파일 확인
ls -lh /home/perelman/jax-hdm/coyo11m-meta.parquet
ls -lh /home/perelman/jax-hdm/000000-000009.pt
```

### 문제 3: "JAX import" 단계에서 실패

**증상:**
```
[Test Step 1] Import JAX
  ✗ FAILED
  Error: ModuleNotFoundError
```

**원인:**
- JAX 미설치
- venv 활성화 안 됨

**해결:**
```bash
# venv 활성화
source /home/perelman/jax-hdm/.venv/bin/activate

# 재시도
python -u test_initialization_steps.py
```

## Step 4: 실제 학습 시작

initialization 테스트가 모두 성공하면:

```bash
python -u train_tpu_256.py
```

그래도 CPU가 0%인 경우:

```bash
# 로그 확인
tail -100 /tmp/train_debug.log

# 프로세스 상태 확인
ps aux | grep python

# 프로세스가 sleep 상태인지 확인
strace -p <PID>  # 특정 PID에 연결하여 시스템 콜 추적
```

## Step 5: 더 상세한 디버깅

### 옵션 1: 간단한 모델로 테스트

`train_tpu_256.py` 수정:
```python
config = TrainingConfig256(
    num_epochs=1,              # 1 epoch만
    steps_per_epoch=10,        # 10 steps만
    global_batch_size=128,     # 작은 배치
)
```

### 옵션 2: 데이터 로더만 테스트

```python
python -u -c "
from src.embeddings import get_embedding_provider
from src.data.gcs_dataloader import GCSDataLoaderSession

provider = get_embedding_provider('google/embeddinggemma-300m')
session = GCSDataLoaderSession(
    batch_size=128,
    parquet_path='coyo11m-meta.parquet',
    embedding_provider=provider,
    num_workers=4,
)
print('Session ready')
"
```

### 옵션 3: 모델만 테스트

```python
python -u -c "
import jax
from src.xut.xut_small import create_xut_small
import jax.numpy as jnp

model = create_xut_small()
x = jnp.zeros((1, 32, 32, 4))
t = jnp.array([0])
ctx = jnp.zeros((1, 1, 640))

print('Running forward pass...')
output = model(x, t, ctx=ctx, deterministic=True)
print(f'Output shape: {output.shape}')
"
```

## Step 6: 로그 파일 분석

학습이 멈춘 경우 로그 파일 확인:

```bash
# 마지막 100줄 확인
tail -100 /tmp/train_debug.log

# 에러만 필터링
grep -i error /tmp/train_debug.log
grep -i timeout /tmp/train_debug.log
grep -i fatal /tmp/train_debug.log

# 시간순으로 정렬된 전체 로그
sort /tmp/train_debug.log | less
```

## Step 7: 메모리/CPU 병목 확인

```bash
# Python 프로세스의 상세 정보
python -u train_tpu_256.py 2>&1 | strace -e trace=file,network,process

# 메모리 누수 확인
python -u -m memory_profiler train_tpu_256.py

# CPU 프로파일링
python -u -m cProfile -s cumtime train_tpu_256.py
```

## Checklist: Training 시작 전 확인사항

- [ ] `python -u` 명령으로 실행하는가?
- [ ] embedding provider 다운로드 완료? (`~/.cache/huggingface/`)
- [ ] GCS 권한 설정? (`gcloud auth application-default login`)
- [ ] 로컬 데이터 있는가? (`coyo11m-meta.parquet`, `000000-000009.pt`)
- [ ] 디스크 공간 충분한가? (`df -h /home/`)
- [ ] 메모리 충분한가? (htop에서 확인)
- [ ] 네트워크 연결 정상? (`ping huggingface.co`, `gcloud storage ls`)
- [ ] CPU 온도 정상? (`watch sensors` 또는 `nvidia-smi`)

## 문제 리포트 템플릿

버그 리포트할 때 다음 정보 포함:

```
문제:
- CPU 사용률: 0% / 높음 / 간헐적
- 멈춘 위치: test step 번호 또는 training step
- 에러 메시지: (있으면)

환경:
- OS: Ubuntu 22.04 / etc
- Python: 3.x
- JAX version: (python -c "import jax; print(jax.__version__)")
- GPU/TPU: CPU / TPU v5e / A100 / etc

로그:
- /tmp/train_debug.log (마지막 50줄)
- htop 스크린샷
- `test_initialization_steps.py` 결과

재현 방법:
- 실행 명령어
- 설정 변경사항 (있으면)
```

## 추가 자료

- [JAX Debugging Guide](https://jax.readthedocs.io/en/latest/)
- [Hugging Face Cache Issues](https://huggingface.co/docs/hub/security-tokens)
- [GCS Authentication](https://cloud.google.com/docs/authentication/provide-credentials-adc)
