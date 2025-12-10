# Ouroboros: XUT Diffusion Transformer

A JAX/Flax implementation of **XUDiT (Cross UDiT)** - a novel encoder-decoder transformer architecture for image generation using Rectified Flow.

## Key Features

- **XUT Architecture**: Encoder-decoder transformer with skip connections (U-Net style)
- **Rectified Flow**: Linear interpolation diffusion with velocity prediction
- **Gemma-3 270M**: Lightweight text encoder (640-dim) with semantic clustering
- **TPU Optimized**: Designed for Google Cloud TPU v5e pods
- **JAX/Flax NNX**: Modern API with bfloat16 support

## Model Architecture

```
Input Latent (32×32×4)
        ↓
   Patch Embed + Timestep Embed
        ↓
┌─────────────────────┐
│   ENCODER STACK     │  ← Self-Attention + Cross-Attention
│   (1 block × depth) │     + AdaLN + SwiGLU MLP
└─────────────────────┘
        ↓ (skip connection)
┌─────────────────────┐
│   DECODER STACK     │  ← Cross-Attention (to encoder)
│   (2 block × depth) │     + Self-Attention + AdaLN
└─────────────────────┘
        ↓
    UnPatch → Output Velocity (32×32×4)
```

### XUT-Small Specifications

| Parameter | Value |
|-----------|-------|
| Model Dimension | 896 |
| MLP Dimension | 3,072 |
| Attention Heads | 14 |
| Depth | 4 |
| Text Embedding | 640 (Gemma-3) |
| Parameters | ~273M |

## Directory Structure

```
Ouroboros/
├── src/
│   ├── xut/                          # XUT Transformer (JAX/Flax NNX)
│   │   ├── xut.py                    # Main model: XUDiT, XUTBackBone
│   │   ├── xut_small.py              # XUT-Small factory
│   │   └── modules/
│   │       ├── attention.py          # Multi-head attention
│   │       ├── adaln.py              # Adaptive Layer Normalization
│   │       ├── axial_rope.py         # 2D Axial RoPE
│   │       ├── patch.py              # Patch embed/unpatch
│   │       ├── time_emb.py           # Timestep embedding
│   │       ├── transformer.py        # Transformer block
│   │       └── layers.py             # SwiGLU MLP, etc.
│   │
│   ├── hdm/                          # HuggingFace Diffusers integration
│   │   ├── modules/
│   │   │   ├── xut.py                # XUDiTConditionModel wrapper
│   │   │   └── text_encoders.py      # Text encoder wrappers
│   │   ├── pipeline.py               # HDMXUTPipeline
│   │   └── trainer/                  # Training utilities
│   │
│   ├── data/
│   │   ├── gcs_dataloader.py         # GCS streaming dataloader
│   │   ├── imagenet_loader.py        # ImageNet dataloader
│   │   └── imagenet_parquet_loader.py
│   │
│   └── embeddings.py                 # Text embedding providers
│
├── train_tpu_256.py                  # Main training script (TPU v5e-32)
├── inference.py                      # Inference script
├── precompute_imagenet_embeddings.py # Pre-compute Gemma embeddings
│
├── outputs/
│   ├── embedding_analysis/           # Embedding visualizations
│   │   ├── Detail.md                 # Full technical report
│   │   ├── model.txt                 # Model specifications
│   │   ├── embedding_comparison_*.png
│   │   ├── dog_cluster_comparison.png
│   │   └── ...
│   └── loss/                         # Training loss plots
│
├── data/
│   ├── loss.csv                      # Training logs
│   └── loss_imagenet.csv
│
└── tests/                            # Unit tests
    └── xut/
```

## Training

### Hardware Requirements
- Google Cloud TPU v5e-32 Pod (8 workers × 4 chips)
- 150GB RAM per worker
- GCS bucket for data storage

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Global Batch Size | 1,024 |
| Learning Rate | 0.5 (base) → 5.58e-4 (muP scaled) |
| Optimizer | AdamW (weight_decay=1e-4) |
| Precision | bfloat16 |
| Warmup Steps | 1,000 |
| Schedule | Cosine decay |

### Run Training

```bash
# Single host (testing)
python train_tpu_256.py

# Distributed (TPU Pod)
./run_tpu_distributed.sh
```

## Text Embedding: Gemma-3 270M

We use **Gemma-3 270M** for text encoding instead of larger models like CLIP or T5:

| Feature | Gemma-3 270M |
|---------|--------------|
| Parameters | 270M |
| Output Dimension | 640 |
| Pooling | Mean over last hidden |
| Library | Native JAX (gemma) |

### Why Gemma-3?

1. **Semantic Clustering**: Similar concepts cluster together
   - Dog breeds: 0.336 average similarity
   - vs Random ("class_0"): 0.000 similarity

2. **Lightweight**: 270M vs CLIP ViT-L (428M) or T5-XXL (11B)

3. **JAX Native**: Seamless TPU integration

### Embedding Quality Comparison

| Metric | Correct Embedding | Buggy ("class_0") |
|--------|-------------------|-------------------|
| Same-group similarity | 0.35 | 0.00 |
| Separation ratio | 1.31x | -1.85x |
| Dog breeds cohesion | 0.336 | 0.000 |

## Diffusion: Rectified Flow

We use **Rectified Flow** (SD3 style) instead of DDPM:

```python
# Forward process (linear interpolation)
x_t = (1 - t) * x_0 + t * x_1  # x_1 = noise

# Model predicts velocity
v = x_1 - x_0

# Timestep sampling (logit-normal)
t = sigmoid(normal(0, 1))
```

**Benefits**:
- Simpler than DDPM
- Fewer sampling steps
- Stable training

## Results

### Training Loss (ImageNet-1K)

| Steps | Loss |
|-------|------|
| 1K | ~0.95 |
| 10K | ~0.82 |
| 100K | ~0.78 |
| 188K | ~0.773 |

### Power Law Fitting
```
Loss(step) = 3.86 × step^(-0.527) + 0.762
```

## Related Papers

| Topic | Paper |
|-------|-------|
| Diffusion | [DDPM](https://arxiv.org/abs/2006.11239), [Rectified Flow](https://arxiv.org/abs/2209.03003) |
| Transformer | [DiT](https://arxiv.org/abs/2212.09748), [U-ViT](https://arxiv.org/abs/2209.12152) |
| Position | [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864) |
| Optimization | [μP](https://arxiv.org/abs/2203.03466), [AdamW](https://arxiv.org/abs/1711.05101) |
| Text Encoder | [Gemma](https://arxiv.org/abs/2403.08295), [CLIP](https://arxiv.org/abs/2103.00020) |
| Guidance | [CFG](https://arxiv.org/abs/2207.12598) |
| VAE | [LDM](https://arxiv.org/abs/2112.10752), [SDXL](https://arxiv.org/abs/2307.01952) |

## Documentation

- **[Detail.md](outputs/embedding_analysis/Detail.md)**: Full technical report with all analysis
- **[model.txt](outputs/embedding_analysis/model.txt)**: Model specifications

## License

MIT License

## Acknowledgments

- Google Cloud TPU Research Cloud
- JAX/Flax team
- HuggingFace Diffusers
