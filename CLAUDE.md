# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ouroboros is a JAX-based implementation of a Home-made Diffusion Model (HDM) with a novel Transformer architecture called **XUT (Xpanded Universal Transformer)**. The project targets TPU infrastructure (Google Cloud TPU v5e) for training large-scale diffusion models.

## Build & Development Commands

```bash
# Install dependencies (using uv package manager)
uv sync

# Run all tests (CPU mode via conftest.py)
pytest

# Run specific test file
pytest tests/xut/test_xut.py -v

# Run specific test
pytest tests/xut/modules/test_attention.py::test_self_attention_basic -v

# Training (256px images on TPU)
python train_tpu_256.py
```

## Architecture

### Core Components

**`/src/xut/`** - JAX/Flax NNX implementation of XUT Transformer:
- `xut.py`: Main model classes - `TBackBone` (standard transformer), `XUTBackBone` (encoder-decoder with skip connections), `XUDiT` (full diffusion transformer)
- `modules/`: Building blocks - attention, AdaLN, axial RoPE, patch embedding, timestep embedding, SwiGLU MLP

**`/src/hdm/`** - PyTorch/HuggingFace Diffusers integration layer:
- `modules/xut.py`: `XUDiTConditionModel` wrapper bridging JAX to Diffusers
- `pipeline.py`: `HDMXUTPipeline` for inference with classifier-free guidance
- `trainer/`: Training loop with layer-wise LR scaling and SNR-weighted loss
- `data/`: Dataset loaders (Kohya format, GCS streaming)

### XUDiT Architecture Flow

1. **Patch Embedding**: 2D image → sequence via conv-based patching
2. **Timestep Embedding**: Sinusoidal embeddings with MLP projection
3. **XUTBackBone**: Encoder-decoder with skip connections
   - Encoder: Stacked transformer blocks with self-attention
   - Decoder: Cross-attention to encoder + self-attention
4. **AdaLN**: Adaptive layer normalization for time-conditioning (shared across layers)
5. **Axial RoPE**: 2D positional encoding with aspect ratio awareness
6. **Unpatch**: Sequence → 2D output image

### Key Design Patterns

- **Flax NNX**: Modern JAX module system - all models inherit from `nnx.Module`, use `nnx.Rngs` for reproducibility
- **TREAD**: Timestep-Random Encoder Architecture Design for dropout/regularization
- **Shared AdaLN**: Time-conditioning parameters shared across transformer blocks
- **Encoder-Decoder Skip Connections**: First decoder block cross-attends to encoder outputs

## Configuration

Training config in `train_tpu_256.py` (dataclass `TrainingConfig256`):
- Batch size: 2048 global, 128 per device
- Model: XUT-Small (dim=896, heads=14, depth=4)
- Data: GCS-based COYO-11M dataset with 112-worker prefetch

Test config in `tests/conftest.py`:
- Forces JAX CPU mode: `jax.config.update("jax_platform_name", "cpu")`

## Key Files

- `/src/xut/xut.py` - Core XUDiT architecture
- `/src/xut/modules/transformer.py` - Transformer building block
- `/train_tpu_256.py` - Training script and configuration
- `/src/hdm/modules/xut.py` - PyTorch/Diffusers integration

## Running Environment

- TPU v5e
- tpuv5litepod-16 environment
- 112 vCPU
- 150GB RAM per worker (free memory is less)