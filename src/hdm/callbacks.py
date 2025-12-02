"""JAX/Flax 학습 콜백 (PyTorch Lightning 대체)"""

import os
from typing import Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class CallbackConfig:
    """콜백 설정"""
    period: int = 100           # 몇 스텝마다 콜백 실행
    num: int = 4                # 생성할 샘플 수
    preview_num: int = 4        # 미리보기할 샘플 수
    batch_size: int = 4
    steps: int = 24             # 생성 스텝 (확산)
    save_dir: str = "./samples"


class SampleGenerationCallback:
    """주기적으로 샘플을 생성하고 저장하는 콜백"""
    
    def __init__(
        self,
        config: CallbackConfig,
        generation_fn: Callable[[nnx.Module, jnp.ndarray, CallbackConfig], Tuple[List[str], List[jnp.ndarray]]],
    ):
        """
        Args:
            config: 콜백 설정
            generation_fn: (model, batch, config) -> (captions, images) 함수
        """
        self.config = config
        self.generation_fn = generation_fn
        self.step = 0
    
    def on_train_step(
        self,
        model: nnx.Module,
        batch: Tuple[jnp.ndarray, List[str]],
        step: int,
        exp_id: Optional[str] = None,
    ):
        """학습 스텝 콜백
        
        Args:
            model: 학습 중인 모델
            batch: (images, captions) 배치
            step: 현재 스텝
            exp_id: 실험 ID
        """
        self.step = step
        
        if step % self.config.period != 0:
            return
        
        print(f"\n{'='*60}")
        print(f"[Step {step}] Generating samples...")
        print(f"{'='*60}")
        
        try:
            # 샘플 생성
            images, captions = self._generate_samples(model, batch)
            
            # 저장
            self._save_samples(images, captions, step, exp_id)
            
            print(f"✓ Generated {len(images)} samples")
            
        except Exception as e:
            print(f"✗ Error in sample generation: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_samples(
        self,
        model: nnx.Module,
        batch: Tuple[jnp.ndarray, List[str]],
    ) -> Tuple[List[jnp.ndarray], List[str]]:
        """샘플 생성
        
        Returns:
            (images, captions): 생성된 이미지와 캡션
        """
        _, captions = batch
        
        # 모델을 eval 모드로 (dropout 등 비활성화)
        # JAX는 deterministic=True로 처리
        
        # 샘플 생성 함수 호출
        images, captions_out = self.generation_fn(
            model,
            batch,
            self.config
        )
        
        return images, captions_out
    
    def _save_samples(
        self,
        images: List[jnp.ndarray],
        captions: List[str],
        step: int,
        exp_id: Optional[str] = None,
    ):
        """샘플 저장
        
        Args:
            images: 생성된 이미지 (NHWC 또는 NCHW)
            captions: 이미지 설명
            step: 현재 스텝
            exp_id: 실험 ID
        """
        exp_id = exp_id or "default"
        save_dir = os.path.join(self.config.save_dir, str(exp_id), str(step))
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving to {save_dir}...")
        
        for idx, (image, caption) in enumerate(zip(images[:self.config.preview_num], captions)):
            # 이미지 저장
            image_path = os.path.join(save_dir, f"{idx:04d}.png")
            self._save_image(image, image_path)
            
            # 캡션 저장
            caption_path = os.path.join(save_dir, f"{idx:04d}_caption.txt")
            with open(caption_path, 'w') as f:
                f.write(caption)
            
            print(f"  [{idx}] {caption[:50]}...")
    
    @staticmethod
    def _save_image(image: jnp.ndarray, path: str):
        """JAX 배열을 PNG로 저장
        
        Args:
            image: (H, W, C) 또는 (C, H, W) 배열, 값 범위 [0, 1] 또는 [-1, 1]
            path: 저장 경로
        """
        try:
            from PIL import Image
            import numpy as np
            
            # NCHW -> NHWC 변환
            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                if image.shape[0] < image.shape[-1]:  # CHW 형식
                    image = jnp.transpose(image, (1, 2, 0))
            
            # 값 범위 정규화 [0, 255]
            image_np = np.array(image)
            if image_np.min() < 0:  # [-1, 1]
                image_np = (image_np + 1) / 2
            else:  # [0, 1]
                pass
            
            image_np = (image_np * 255).astype(np.uint8)
            
            # 채널 수 확인
            if image_np.shape[-1] == 1:
                image_np = image_np.squeeze(-1)
            
            img = Image.fromarray(image_np)
            img.save(path)
            
        except Exception as e:
            print(f"Warning: Could not save image {path}: {e}")


# ============================================
# 사용 예시
# ============================================
def example_generation_fn(
    model: nnx.Module,
    batch: Tuple[jnp.ndarray, List[str]],
    config: CallbackConfig,
) -> Tuple[List[jnp.ndarray], List[str]]:
    """샘플 생성 함수 예시
    
    실제 구현에서는 확산 모델을 역방향으로 실행하여 이미지 생성
    """
    _, captions = batch
    
    # 더미 이미지 생성 (실제로는 모델로 생성)
    images = [
        jax.random.normal(jax.random.PRNGKey(i), (32, 32, 4))
        for i in range(config.num)
    ]
    
    # 첫 num개 캡션 반환
    captions_out = captions[:config.num]
    
    return images, captions_out


# 사용 예
if __name__ == "__main__":
    config = CallbackConfig(period=10, num=2, preview_num=2)
    callback = SampleGenerationCallback(config, example_generation_fn)
    
    # 더미 배치
    batch = (
        jnp.zeros((4, 32, 32, 4)),
        ["caption 1", "caption 2", "caption 3", "caption 4"]
    )
    
    # 콜백 테스트
    class DummyModel(nnx.Module):
        pass
    
    model = DummyModel()
    callback.on_train_step(model, batch, step=10, exp_id="test")
