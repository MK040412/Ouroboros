import torch
import jax
import jax.numpy as jnp
from pathlib import Path

# 1. 파일 경로 설정
file_path = "/home/perelman/jax-hdm/000000-000009.pt"

# 2. PyTorch로 로드
data = torch.load(file_path, map_location="cpu")

# 3. 데이터 구조 확인
print("="*50)
print("Data Type:", type(data))

if isinstance(data, dict):
    print("Keys:", data.keys())
    print("\nKey Details:")
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}")
    
    # 메인 데이터 추출 (일반적인 경우)
    if 'latents' in data:
        sample_tensor = data['latents']
    elif 'image' in data:
        sample_tensor = data['image']
    else:
        # 첫 번째 텐서 값 사용
        sample_tensor = next((v for v in data.values() if isinstance(v, torch.Tensor)))
else:
    sample_tensor = data

print("\n" + "="*50)
print(f"Main Tensor Shape: {sample_tensor.shape}")
print(f"Main Tensor dtype: {sample_tensor.dtype}")
print(f"Main Tensor min/max: [{sample_tensor.min():.4f}, {sample_tensor.max():.4f}]")

# 4. JAX Array로 변환 테스트
print("\n" + "="*50)
try:
    # bfloat16은 JAX에서 직접 지원 안함 -> float32로 변환
    if sample_tensor.dtype == torch.bfloat16:
        print(f"bfloat16 감지 -> float32로 변환")
        sample_tensor_converted = sample_tensor.float()
    else:
        sample_tensor_converted = sample_tensor
    
    jax_array = jnp.array(sample_tensor_converted.numpy())
    print(f"JAX Array Shape: {jax_array.shape}")
    print(f"JAX Array dtype: {jax_array.dtype}")
    print(f"JAX Array min/max: [{jax_array.min():.4f}, {jax_array.max():.4f}]")
    print("\n✓ JAX 변환 성공")
except Exception as e:
    print(f"✗ JAX 변환 실패: {e}")

# 5. 결론
print("\n" + "="*50)
print("결론: 이 jax_array가 모델 입력 Shape과 맞다면 학습 가능합니다.")
