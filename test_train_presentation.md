---
marp: true
theme: default
paginate: true
---

# JAX/Flax Diffusion Model Training

## Overview
XUDiT 모델을 사용한 노이즈 예측 기반 확산 모델 학습

---

## Step 1: 데이터 로딩

### 원본 데이터 형식
```
(119829, 3, 4, 32, 32) = (B, T, C, H, W)
```

### 처리 과정
- **B**: 119829개 배치 (일부만 사용)
- **T**: 3개 타임스텝 (첫 프레임만 선택)
- **C**: 4채널 (latent 채널)
- **H, W**: 32x32 해상도

### 변환 결과
```
(10, 32, 32, 4) = (B, H, W, C) NHWC 형식
```

---

## Step 2: 모델 초기화

### XUDiT 구조
```python
model = XUDiT(
    patch_size=2,        # 2x2 패치로 분할
    input_dim=4,         # 입력 채널 (latent)
    dim=16,              # 내부 임베딩 차원
    heads=4,             # Attention 헤드
    depth=1,             # 트랜스포머 깊이
    enc_blocks=1,        # 인코더 블록
    dec_blocks=1,        # 디코더 블록
)
```

### 입출력 형식
- **입력**: (B, H, W, C) NHWC = (2, 32, 32, 4)
- **출력**: (B, C, H, W) NCHW = (2, 4, 32, 32)

---

## Step 3: Noise Schedule

### Linear Schedule 정의
```python
beta_t = beta_min + (beta_max - beta_min) * (t / T)

alpha_t = ∏(1 - beta_i)  # cumulative product
sigma_t = √(1 - alpha_t)  # noise level
```

### Forward Diffusion
원본 이미지에 노이즈 추가:
```
x_t = √α_t · x_0 + σ_t · ε
```
- **x_0**: 원본 이미지
- **ε**: 표준정규분포 노이즈
- **t**: 임의의 타임스텝

---

## Step 4: Loss Function

### Noise Prediction Loss
모델이 추가된 노이즈를 맞추도록 학습:

```python
# 입력 x_t에서 노이즈 예측
pred_noise = model(x_t, t)  # (B, C, H, W)

# 실제 노이즈와 비교
loss = MSE(pred_noise, noise)
```

### Shape 처리
- 모델 출력: (B, C, H, W)
- 노이즈: (B, H, W, C) → transpose → (B, C, H, W)

---

## Step 5: Optimizer 설정

### Adam Optimizer
```python
optimizer = nnx.Optimizer(
    model,
    optax.adam(learning_rate=1e-4)
)
```

---

## Step 6: Training Step

### 그래디언트 계산 및 업데이트
```python
def train_step(model, optimizer, x_t, t, noise):
    loss, grads = nnx.value_and_grad(loss_fn)(
        model, x_t, t, noise, deterministic=False
    )
    optimizer.update(grads)  # 가중치 업데이트
    return loss
```

---

## Step 7: Training Loop

### 반복 과정
```
for step in range(num_steps):
    1. 배치 샘플링
    2. 랜덤 타임스텝 선택 (0~1000)
    3. 랜덤 노이즈 생성
    4. Forward diffusion (노이즈 추가)
    5. Training step (손실 계산 및 업데이트)
```

---

## Step 8: 결과

### 학습 결과
```
Step 1/3 - Loss: 2.631680 ✓
Step 2/3 - Loss: 2.667943 ✓
Step 3/3 - Loss: 2.673234 ✓
```

### 특징
- CPU에서 정상 작동
- GPU/TPU 호환 가능
- Torch 의존성 제거 (JAX만 사용)

---

## Summary

### 핵심 구성 요소
1. **데이터**: (B, H, W, C) NHWC 형식
2. **모델**: XUDiT (패치 기반 트랜스포머)
3. **목표**: 노이즈 예측 학습
4. **최적화**: Adam + MSE Loss
5. **프레임워크**: JAX/Flax

### 확장 가능성
- 더 많은 데이터로 학습 가능
- GPU/TPU에서 가속화
- 조건부 생성 (conditional generation) 추가
