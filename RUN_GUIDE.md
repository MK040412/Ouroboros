# train_tpu_256.py 실행 가이드

## 중요: Python 버퍼링 비활성화

반드시 `-u` 플래그를 사용해서 실행하세요. 그래야 로그가 실시간으로 출력됩니다.

```bash
python -u train_tpu_256.py
```

## 실행 환경별 명령어

### 1. 단일 호스트 (로컬 또는 단일 TPU 16)

```bash
python -u train_tpu_256.py
```

**자동 감지:**
- 분산 환경이 없으면 자동으로 single-host 모드 실행
- jax.devices() 확인하여 장치 감지

### 2. 멀티 호스트 TPU Pod (SLURM 환경)

```bash
srun python -u train_tpu_256.py
```

또는:

```bash
python -m jax.distributed.launch --nprocs=16 python -u train_tpu_256.py
```

## 로그 출력 단계

1. **[Environment Check]** - 분산 환경 감지
2. **[Device Detection]** - 이용 가능한 TPU 장치 확인
3. **[Text Embedding Model Setup]** - Text embedding 모델 로드
4. **[Data Files Setup]** - PT/Parquet 파일 확인
5. **Model Initialization** - XUT-Small 모델 생성
6. **[Training Starting]** - 학습 시작 정보
7. **EPOCH** - 각 epoch 진행 상황
8. **PT file** - 각 PT 파일당 손실값

## 멈추는 경우 대처법

### 1. jax.distributed.initialize() hang

**원인:** 멀티 호스트 분산 환경 미설정

**해결책:**
- 단일 호스트면 무시됨 (자동으로 fallback)
- SLURM 환경 아니면 그냥 실행하세요

### 2. PT 파일 로드 오래 걸림

**원인:** PT 파일이 크거나 느린 스토리지

**확인사항:**
```bash
ls -lh *.pt coyo11m-meta.parquet
```

로그에서 파일 크기 확인 가능:
```
✓ Found 3 PT files:
  [1] 000000-000009.pt (2.34GB)
  [2] 000010-000019.pt (2.34GB)
  ...
```

### 3. Text embedding 로드 오래 걸림

**원인:** 첫 실행 시 모델 다운로드

로그 확인:
```
[Text Embedding Model Setup]
Loading text embedding model...
  Model: google/embeddinggemma-300m
```

이 단계가 몇 분 걸릴 수 있습니다.

## 정상 실행 로그 예시

```
============================================================
TPU v5e 16 Pod Training (256² XUT-Small)
============================================================

[Environment Check]
  JAX_COORDINATOR_ADDRESS: Not set
  SLURM_PROCID: Not set
  use_distributed: False

[Single-Host Mode] Using default JAX setup

[Device Detection]
============================================================
Total devices: 16
Devices: [TpuDevice(id=0, process_index=0), ...]
Device type: tpu

[Text Embedding Model Setup]
============================================================
Loading text embedding model...
  Model: google/embeddinggemma-300m
✓ Text embedding provider loaded

[Data Files Setup]
============================================================
Scanning for PT files...
✓ Found 3 PT files:
  [1] 000000-000009.pt (2.34GB)
  [2] 000010-000019.pt (2.34GB)
  [3] 000020-000029.pt (2.34GB)
✓ Parquet file: coyo11m-meta.parquet (1.23GB)

[Training Starting]
======================================================================
Total epochs: 20
PT files per epoch: 3
Steps per PT file: 3750
Global batch size: 2048

======================================================================
EPOCH 1/20
======================================================================

  [1/20] PT 1/3: 000000-000009.pt
Loading latents from 000000-000009.pt...
Loading metadata from coyo11m-meta.parquet...
Found 10 samples with both latent and caption
Loading captions...
✓ Loaded 10 captions
Epoch 1/20 Step 100/3750 Loss: 0.123456
...
    ✓ PT 000000-000009.pt done in 15.2m - Loss: 0.098765
```

## 성능 팁

- **배치 크기:** 2048 (16 devices × 128)
- **Workers:** 112개 CPU에서 prefetch
- **기대 시간:** 에포크당 ~50시간 (3 PT files)

## 문제 보고

로그에서:
- ✗ 표시 = 오류
- ✓ 표시 = 성공
- [Tag] = 단계
