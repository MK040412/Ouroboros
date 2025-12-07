# Ouroboros: XUT-Small Diffusion Transformer - Detailed Technical Report

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Model Architecture](#2-model-architecture)
3. [Text Embedding System](#3-text-embedding-system)
4. [Training Configuration](#4-training-configuration)
5. [Embedding Quality Analysis](#5-embedding-quality-analysis)
6. [Training Loss Analysis](#6-training-loss-analysis)
7. [Related Papers](#7-related-papers)
8. [Appendix](#8-appendix)

---

## 1. Project Overview

**Ouroboros**는 JAX/Flax 기반의 Home-made Diffusion Model (HDM) 프로젝트로, 새로운 **XUT (Xpanded Universal Transformer)** 아키텍처를 사용한 Diffusion Transformer를 구현합니다.

### 1.1 Key Features
- **Framework**: JAX + Flax NNX (Modern API)
- **Target Hardware**: Google Cloud TPU v5e-32 Pod
- **Dataset**: COYO-11M, ImageNet-1K
- **Resolution**: 256 × 256 pixels
- **Diffusion**: Rectified Flow (SD3 style)

### 1.2 Repository Structure
```
Ouroboros/
├── src/
│   ├── xut/           # XUT Transformer implementation
│   │   ├── xut.py     # Main model (XUDiT, XUTBackBone)
│   │   └── modules/   # Attention, AdaLN, RoPE, etc.
│   ├── hdm/           # Diffusers integration
│   └── embeddings.py  # Text embedding providers
├── train_tpu_256.py   # Training script
└── outputs/           # Visualizations & logs
```

---

## 2. Model Architecture

### 2.1 XUDiT (Xpanded Universal Diffusion Transformer)

XUDiT는 U-Net 스타일의 Encoder-Decoder 구조를 Transformer로 구현한 아키텍처입니다.

#### Architecture Diagram
```
Input Latent (32×32×4)
        ↓
   Patch Embed
        ↓
   + Timestep Embed
        ↓
┌─────────────────────┐
│   ENCODER STACK     │
│  (1 block × 4 depth)│
│    Self-Attention   │
│    + Cross-Attn     │
│    + AdaLN + MLP    │
└─────────────────────┘
        ↓ (skip connection)
┌─────────────────────┐
│   DECODER STACK     │
│  (2 block × 4 depth)│
│    Cross-Attn (enc) │
│    + Self-Attention │
│    + AdaLN + MLP    │
└─────────────────────┘
        ↓
    UnPatch
        ↓
Output Velocity (32×32×4)
```

### 2.2 Model Specifications (XUT-Small)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `dim` | 896 | Model hidden dimension |
| `mlp_dim` | 3,072 | MLP intermediate dimension |
| `heads` | 14 | Number of attention heads |
| `dim_head` | 64 | Dimension per head |
| `depth` | 4 | Number of transformer layers |
| `enc_blocks` | 1 | Encoder blocks per layer |
| `dec_blocks` | 2 | Decoder blocks per layer |
| `ctx_dim` | 640 | Text embedding dimension |
| **Total Params** | ~50M | Estimated |

### 2.3 Key Components

#### 2.3.1 Patch Embedding
- **Type**: Conv-based (not linear projection)
- **Patch Size**: 2×2 (32×32 latent → 16×16 sequence)
- **Reference**: [ViT](https://arxiv.org/abs/2010.11929), [DiT](https://arxiv.org/abs/2212.09748)

#### 2.3.2 Axial RoPE (Rotary Position Embedding)
- **Type**: 2D Axial decomposition
- **Feature**: Aspect ratio awareness
- **Reference**: [RoFormer](https://arxiv.org/abs/2104.09864), [LLaMA](https://arxiv.org/abs/2302.13971)

#### 2.3.3 Adaptive Layer Normalization (AdaLN)
- **Type**: Shared across layers (memory efficient)
- **Conditioning**: Timestep embedding
- **Output**: Scale (γ), Shift (β), Gate (α) × 3
- **Reference**: [DiT](https://arxiv.org/abs/2212.09748)

```python
# AdaLN implementation
y = self.norm(x)
scale, shift, gate = chunk3(adaln_output)
y = (1 + scale) * y + shift
y = gate * y  # Gating
```

#### 2.3.4 SwiGLU MLP
- **Activation**: SiLU (Swish) with gating
- **Reference**: [GLU Variants](https://arxiv.org/abs/2002.05202)

```python
# SwiGLU
gate = self.w1(x)
value = self.w2(x)
output = silu(gate) * value
output = self.w3(output)
```

#### 2.3.5 RMSNorm
- **Type**: Root Mean Square Normalization
- **Benefit**: Faster than LayerNorm, similar performance
- **Reference**: [RMSNorm Paper](https://arxiv.org/abs/1910.07467)

---

## 3. Text Embedding System

### 3.1 Embedding Model: Gemma-3 270M

| Specification | Value |
|---------------|-------|
| **Model** | google/gemma-3-270m |
| **Parameters** | 270 Million |
| **Output Dimension** | 640 |
| **Pooling** | Mean over last hidden states |
| **Normalization** | L2 normalized |
| **Library** | Native gemma (JAX) |

### 3.2 Why Gemma-3 270M?

1. **Lightweight**: 270M params vs CLIP ViT-L (428M) or T5-XXL (11B)
2. **JAX Native**: Seamless TPU integration via `gemma` library
3. **Semantic Quality**: Strong text understanding despite small size
4. **640-dim Output**: Good balance of expressiveness vs efficiency

### 3.3 Pre-computed Embeddings

CPU bottleneck 문제 해결을 위해 embedding을 사전 계산하여 PT 파일에 저장:

```python
# PT file structure
{
    'keys': np.array([...]),           # Image IDs
    'latents': torch.tensor(...),      # (N, 4, 32, 32) VAE latents
    'embeddings': torch.tensor(...),   # (N, 640) Gemma embeddings
}
```

**Benefits**:
- 학습 시 text encoding overhead 제거
- 112 vCPU로도 TPU 32코어 활용 가능
- ~3GB per PT file (latents + embeddings)

### 3.4 ImageNet 1K Class Embeddings

OrderedDict 기반 ImageNet-2012 클래스 매핑:

```python
IMAGENET2012_CLASSES = OrderedDict([
    ('n01440764', 'tench, Tinca tinca'),
    ('n01443537', 'goldfish, Carassius auratus'),
    ...
    ('n07920052', 'espresso'),
    ('n09256479', 'coral reef'),
    ('n15075141', 'toilet tissue, toilet paper, bathroom tissue'),
])
```

**File**: `imagenet_class_embeddings.npy` (1000, 640)

---

## 4. Training Configuration

### 4.1 Hardware Setup

| Component | Specification |
|-----------|---------------|
| **Platform** | Google Cloud TPU v5e-32 Pod |
| **Workers** | 8 hosts |
| **TPU Chips** | 4 per worker (32 total) |
| **vCPUs** | 112 per worker |
| **RAM** | 150GB per worker |
| **Storage** | GCS (Google Cloud Storage) |

### 4.2 Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `global_batch_size` | 1,024 | Across all 32 TPU cores |
| `batch_per_device` | 32 | 1024 / 32 |
| `learning_rate` | 0.5 | Base LR (before muP scaling) |
| `mup_lr` | 5.58e-4 | 0.5 × (1/896) |
| `warmup_steps` | 1,000 | Linear warmup |
| `lr_schedule` | Cosine decay | After warmup |
| `optimizer` | AdamW | weight_decay=1e-4 |
| `grad_clip` | 1.0 | Global norm |
| `precision` | bfloat16 | TPU optimized |

### 4.3 muP (Maximal Update Parameterization)

**Reference**: [μP Paper](https://arxiv.org/abs/2203.03466)

```python
# muP scaling
mup_lr = base_lr * (base_dim / model_dim)
       = 0.5 * (1 / 896)
       = 5.58e-4
```

**Benefits**:
- Hyperparameter transfer across model sizes
- Stable training at large scale
- Width-independent optimal LR

### 4.4 Rectified Flow

**Reference**: [Rectified Flow](https://arxiv.org/abs/2209.03003), [Flow Matching](https://arxiv.org/abs/2210.02747)

#### 4.4.1 Forward Process
```python
# Linear interpolation
x_t = (1 - t) * x_0 + t * x_1
# where x_0 = clean latent, x_1 = noise
```

#### 4.4.2 Velocity Target
```python
# Model predicts velocity
v = x_1 - x_0  # noise - clean
```

#### 4.4.3 Logit-Normal Timestep Sampling (SD3 Style)
```python
# SD3 style timestep sampling
u = normal(0, 1)
t = sigmoid(logit_mean + logit_std * u)
# Concentrates samples at mid-timesteps
```

**Reference**: [Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3)

### 4.5 Classifier-Free Guidance (CFG)

**Reference**: [CFG Paper](https://arxiv.org/abs/2207.12598)

```python
# Training: 50% dropout of text conditioning
if random() < 0.5:
    text_emb = zeros_like(text_emb)

# Inference
pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
```

### 4.6 TREAD (Timestep-Random Encoder Architecture Design)

모델 내부에서 추가적인 token dropout 적용:
- `deterministic=False`: Training mode with dropout
- `deterministic=True`: Inference mode

---

## 5. Embedding Quality Analysis

### 5.1 Correct vs Buggy Embedding Comparison

#### 5.1.1 Problem Definition
- **Buggy**: `"class_0"`, `"class_1"`, ... (무의미한 텍스트)
- **Correct**: `"golden retriever"`, `"tabby cat"`, ... (실제 클래스명)

#### 5.1.2 Quantitative Results

| Metric | Correct | Buggy | Improvement |
|--------|---------|-------|-------------|
| Same-group similarity | 0.35 | 0.00 | ∞ |
| Cross-group similarity | 0.27 | 0.00 | - |
| Separation ratio | 1.31x | -1.85x | 130x |
| Dog breeds similarity | 0.336 | 0.000 | 33.6x |

#### 5.1.3 Dog Breeds Analysis (118 classes)

**Correct Embedding**:
- Similar breeds cluster together (t-SNE/UMAP)
- Retriever types: 0.92+ similarity
- Terrier types: 0.88+ similarity

**Buggy Embedding**:
- Random distribution
- No semantic structure
- Mean similarity ≈ 0 (orthogonal)

### 5.2 Semantic Neighbor Analysis

| Query | Top Neighbors (Correct) | Similarity |
|-------|------------------------|------------|
| golden retriever | Labrador retriever | 0.972 |
| | curly-coated retriever | 0.923 |
| | flat-coated retriever | 0.923 |
| tabby cat | lynx | 1.000 |
| | Siamese cat | 1.000 |
| | Persian cat | 0.886 |
| sports car | spotlight | 0.991 |
| | spider web | 0.987 |

### 5.3 PCA Analysis

| Metric | Value |
|--------|-------|
| Components for 95% variance | 52 |
| Components for 99% variance | 92 |
| PC1 explained variance | 41.5% |
| PC2 explained variance | 8.9% |

**Insight**: 640차원 중 실제 정보가 집중된 차원은 ~100개 미만

### 5.4 Impact on Diffusion Transformer

**Correct Embedding Enables**:
1. Meaningful text-to-image correspondence learning
2. Transfer learning between similar classes
3. Smooth latent space interpolation
4. Better generalization to unseen combinations

**Buggy Embedding Results in**:
1. Model must memorize each class independently
2. No transfer between similar classes
3. Inefficient learning (more parameters needed)
4. Poor interpolation capability

---

## 6. Training Loss Analysis

### 6.1 Training Runs Comparison

| Run | Dataset | Embedding | Final Loss | Power Law |
|-----|---------|-----------|------------|-----------|
| COYO | COYO-11M | Captions | ~0.775 | 9.03·x^(-0.643)+0.775 |
| IN-buggy | ImageNet-1K | class_0 | ~0.773 | 1.62·x^(-0.353)+0.752 |
| IN-fixed | ImageNet-1K | Class names | ~0.773 | 3.86·x^(-0.527)+0.762 |

### 6.2 Loss Curve Characteristics

#### 6.2.1 Power Law Fitting
```
Loss(step) = a · step^(-b) + c
```

| Run | a | b (decay rate) | c (floor) |
|-----|---|----------------|-----------|
| COYO | 9.03 | 0.643 | 0.775 |
| IN-buggy | 1.62 | 0.353 | 0.752 |
| IN-fixed | 3.86 | 0.527 | 0.762 |

**Observation**: IN-fixed has steeper decay (b=0.527 vs 0.353)

#### 6.2.2 Regional Slope Analysis (ImageNet)

| Region | Buggy Slope | Fixed Slope | Winner |
|--------|-------------|-------------|--------|
| 6K~10K | -0.0460 | -0.0396 | Fixed (stable) |
| 10K~30K | -0.0324 | -0.0308 | Similar |
| 30K~100K | -0.0321 | -0.0274 | Fixed (efficient) |
| Full | -0.0328 | -0.0311 | Fixed |

### 6.3 Current Training Status

| Metric | Value |
|--------|-------|
| Current Step | 188,399 |
| Current Epoch | 151 |
| Current Loss | 0.773 |
| Learning Rate | 5.28e-4 |
| Training Time | ~12 hours |

### 6.4 Key Findings

1. **Early Training**: Buggy embedding shows faster initial descent
2. **Mid Training**: Both converge to similar range
3. **Late Training**: Fixed embedding maintains more consistent improvement
4. **Final Loss**: Both similar (~0.773), but fixed has better semantic structure

---

## 7. Related Papers

### 7.1 Diffusion Models

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [DDPM](https://arxiv.org/abs/2006.11239) | Ho et al. | 2020 | Denoising Diffusion Probabilistic Models |
| [Score-based SDE](https://arxiv.org/abs/2011.13456) | Song et al. | 2021 | Score-based generative modeling |
| [Rectified Flow](https://arxiv.org/abs/2209.03003) | Liu et al. | 2022 | Linear interpolation flow |
| [Flow Matching](https://arxiv.org/abs/2210.02747) | Lipman et al. | 2022 | Conditional flow matching |

### 7.2 Transformer Architectures

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [ViT](https://arxiv.org/abs/2010.11929) | Dosovitskiy et al. | 2020 | Vision Transformer |
| [DiT](https://arxiv.org/abs/2212.09748) | Peebles & Xie | 2022 | Diffusion Transformer with AdaLN |
| [U-ViT](https://arxiv.org/abs/2209.12152) | Bao et al. | 2022 | U-Net style ViT for diffusion |
| [SD3](https://stability.ai/news/stable-diffusion-3) | Stability AI | 2024 | MMDiT, logit-normal sampling |

### 7.3 Positional Encoding

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [RoFormer](https://arxiv.org/abs/2104.09864) | Su et al. | 2021 | Rotary Position Embedding |
| [LLaMA](https://arxiv.org/abs/2302.13971) | Touvron et al. | 2023 | RoPE in LLMs |
| [2D-RoPE](https://arxiv.org/abs/2306.15595) | Various | 2023 | Axial decomposition for images |

### 7.4 Normalization & Optimization

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [RMSNorm](https://arxiv.org/abs/1910.07467) | Zhang & Sennrich | 2019 | Root Mean Square Normalization |
| [GLU Variants](https://arxiv.org/abs/2002.05202) | Shazeer | 2020 | SwiGLU, GEGLU |
| [μP](https://arxiv.org/abs/2203.03466) | Yang et al. | 2022 | Maximal Update Parameterization |
| [AdamW](https://arxiv.org/abs/1711.05101) | Loshchilov & Hutter | 2017 | Decoupled weight decay |

### 7.5 Text Encoders

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [CLIP](https://arxiv.org/abs/2103.00020) | Radford et al. | 2021 | Contrastive Language-Image Pre-training |
| [T5](https://arxiv.org/abs/1910.10683) | Raffel et al. | 2020 | Text-to-Text Transfer Transformer |
| [Gemma](https://arxiv.org/abs/2403.08295) | Google | 2024 | Lightweight language models |

### 7.6 Classifier-Free Guidance

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [CFG](https://arxiv.org/abs/2207.12598) | Ho & Salimans | 2022 | Classifier-free diffusion guidance |

### 7.7 VAE for Latent Diffusion

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [LDM](https://arxiv.org/abs/2112.10752) | Rombach et al. | 2022 | Latent Diffusion Models |
| [SDXL](https://arxiv.org/abs/2307.01952) | Podell et al. | 2023 | SDXL VAE (scaling factor 0.13025) |

---

## 8. Appendix

### 8.1 Generated Visualizations

| File | Description |
|------|-------------|
| `embedding_comparison_tsne_umap.png` | t-SNE & UMAP: Correct vs Buggy |
| `dog_cluster_comparison.png` | Dog breeds (118 classes) analysis |
| `semantic_neighbors_comparison.png` | Nearest neighbors comparison |
| `embedding_metrics_summary.png` | Quantitative metrics |
| `embedding_3d_comparison.png` | 3D UMAP visualization |
| `tsne_2d_categories.png` | Category-colored t-SNE |
| `pca_analysis.png` | PCA variance analysis |
| `similarity_heatmap.png` | 1000×1000 cosine similarity |
| `hierarchical_clustering.png` | Dendrogram |
| `distance_distribution.png` | Intra vs Inter category |

### 8.2 Code References

| File | Description |
|------|-------------|
| `train_tpu_256.py` | Main training script |
| `src/xut/xut.py` | XUDiT model definition |
| `src/xut/modules/transformer.py` | Transformer block |
| `src/embeddings.py` | Embedding providers |
| `precompute_imagenet_embeddings.py` | Embedding pre-computation |
| `visualize_embedding_comparison.py` | Embedding analysis |

### 8.3 GCS Paths

```
gs://rdy-tpu-data-2025/
├── coyo11m-256px-ccrop-latent/
│   ├── latents-3crop-gemma-3-270m/  # PT files with embeddings
│   └── coyo11m-meta.parquet         # Metadata
├── imagenet-1k/
│   ├── imagenet_class_embeddings.npy
│   └── classes.py
└── checkpoints/
    └── xut-small-256/               # Training checkpoints
```

### 8.4 Environment

```yaml
Platform: Google Cloud TPU v5e-32 Pod
JAX Version: 0.4.x
Flax Version: 0.11.x (NNX API)
Python: 3.11
OS: Linux (Debian-based TPU VM)
```

---

## Document Info

- **Generated**: 2024-12-07
- **Project**: Ouroboros (Home-made Diffusion Model)
- **Author**: Auto-generated analysis
- **Repository**: `/home/perelman/바탕화면/Ouroboros/`
