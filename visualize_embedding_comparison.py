"""
Gemma-3 270M Embedding Quality Comparison for Diffusion Transformer

이 스크립트는 다음을 비교합니다:
1. Buggy Mode: "class_0", "class_1", ... 같은 무의미한 텍스트로 임베딩
2. Correct Mode: "tench, Tinca tinca", "golden retriever" 같은 실제 클래스 이름

목적: Text Embedding의 품질이 Diffusion Transformer 학습에 미치는 영향 시각화
- 올바른 embedding: 의미적으로 유사한 클래스들이 클러스터링됨 (dog끼리, cat끼리)
- 잘못된 embedding: 랜덤하게 분포되어 모델이 의미적 관계를 학습할 수 없음
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from collections import OrderedDict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import umap

# 출력 디렉토리
OUTPUT_DIR = Path("outputs/embedding_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 스타일 설정
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10


def load_correct_embeddings():
    """올바른 임베딩 로드 (실제 클래스 이름 기반)"""
    emb_path = OUTPUT_DIR / "imagenet_class_embeddings.npy"
    embeddings = np.load(emb_path)

    classes_path = OUTPUT_DIR / "classes.py"
    with open(classes_path, 'r') as f:
        content = f.read()

    local_vars = {}
    exec(content, {}, local_vars)
    class_dict = local_vars['IMAGENET2012_CLASSES']
    class_names = list(class_dict.values())

    return embeddings, class_names


def create_buggy_embeddings(n_classes=1000, embedding_dim=640):
    """Buggy 임베딩 시뮬레이션 (class_0, class_1, ... 방식)

    실제로는 "class_0" 같은 무의미한 텍스트를 Gemma에 넣으면
    의미 없는 임베딩이 나옴. 여기서는 이를 시뮬레이션.

    방법: deterministic random embeddings (hash 기반)
    - 각 "class_i" 문자열의 hash로 seed 설정
    - 의미적 관계 없이 랜덤 분포
    """
    buggy_embeddings = []
    buggy_names = []

    for i in range(n_classes):
        # "class_0", "class_1", ... 같은 무의미한 이름
        name = f"class_{i}"
        buggy_names.append(name)

        # hash 기반 deterministic random embedding
        seed = hash(name) % (2**32)
        rng = np.random.RandomState(seed)
        emb = rng.randn(embedding_dim).astype(np.float32)
        emb = emb / np.linalg.norm(emb)  # L2 normalize
        buggy_embeddings.append(emb)

    return np.array(buggy_embeddings), buggy_names


def get_semantic_groups():
    """의미적 그룹 정의 (시각화용)"""
    groups = {
        'Dogs': list(range(151, 269)),      # 118 dog breeds
        'Cats': [281, 282, 283, 284, 285],  # cat varieties
        'Birds': list(range(7, 24)) + list(range(80, 100)),
        'Fish': list(range(0, 8)) + list(range(389, 398)),
        'Vehicles': list(range(404, 450)),
        'Food': list(range(924, 970)),
        'Primates': list(range(365, 385)),
        'Reptiles': list(range(25, 69)),
    }

    # 클래스별 그룹 할당
    class_to_group = {}
    for group_name, indices in groups.items():
        for idx in indices:
            if idx < 1000:
                class_to_group[idx] = group_name

    for i in range(1000):
        if i not in class_to_group:
            class_to_group[i] = 'Other'

    return groups, class_to_group


def compute_metrics(embeddings, class_to_group):
    """임베딩 품질 메트릭 계산"""
    # 코사인 유사도 행렬
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-8)
    sim_matrix = normalized @ normalized.T

    # 같은 그룹 vs 다른 그룹 유사도
    same_group_sims = []
    diff_group_sims = []

    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = sim_matrix[i, j]
            if class_to_group[i] == class_to_group[j] and class_to_group[i] != 'Other':
                same_group_sims.append(sim)
            elif class_to_group[i] != 'Other' and class_to_group[j] != 'Other':
                diff_group_sims.append(sim)

    metrics = {
        'mean_all': np.mean(sim_matrix[np.triu_indices(len(embeddings), k=1)]),
        'mean_same_group': np.mean(same_group_sims) if same_group_sims else 0,
        'mean_diff_group': np.mean(diff_group_sims) if diff_group_sims else 0,
        'separation_ratio': (np.mean(same_group_sims) / np.mean(diff_group_sims)) if diff_group_sims else 1,
    }

    return metrics, sim_matrix


def plot_comparison_tsne_umap(correct_emb, buggy_emb, class_names, class_to_group, groups):
    """t-SNE와 UMAP으로 두 임베딩 비교"""
    print("Creating t-SNE and UMAP comparison...")

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # 색상 설정
    group_names = list(groups.keys()) + ['Other']
    colors = plt.cm.tab10(np.linspace(0, 1, len(group_names)))
    color_map = {g: colors[i] for i, g in enumerate(group_names)}

    # t-SNE
    print("  Computing t-SNE for correct embeddings...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    correct_2d = tsne.fit_transform(correct_emb)

    print("  Computing t-SNE for buggy embeddings...")
    buggy_2d = tsne.fit_transform(buggy_emb)

    # UMAP
    print("  Computing UMAP for correct embeddings...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    correct_umap = reducer.fit_transform(correct_emb)

    print("  Computing UMAP for buggy embeddings...")
    buggy_umap = reducer.fit_transform(buggy_emb)

    # Plot 1: Correct t-SNE
    ax1 = fig.add_subplot(gs[0, 0])
    for group in group_names:
        mask = [class_to_group[i] == group for i in range(1000)]
        mask = np.array(mask)
        if mask.sum() > 0 and group != 'Other':
            ax1.scatter(correct_2d[mask, 0], correct_2d[mask, 1],
                       c=[color_map[group]], label=group, alpha=0.7, s=25)
    # Other는 회색으로 배경에
    other_mask = np.array([class_to_group[i] == 'Other' for i in range(1000)])
    ax1.scatter(correct_2d[other_mask, 0], correct_2d[other_mask, 1],
               c='lightgray', alpha=0.3, s=15, label='Other')

    ax1.set_title('Correct Embedding (t-SNE)\n"golden retriever", "tabby cat", ...', fontsize=12, fontweight='bold')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)

    # Plot 2: Buggy t-SNE
    ax2 = fig.add_subplot(gs[0, 1])
    for group in group_names:
        mask = [class_to_group[i] == group for i in range(1000)]
        mask = np.array(mask)
        if mask.sum() > 0 and group != 'Other':
            ax2.scatter(buggy_2d[mask, 0], buggy_2d[mask, 1],
                       c=[color_map[group]], label=group, alpha=0.7, s=25)
    ax2.scatter(buggy_2d[other_mask, 0], buggy_2d[other_mask, 1],
               c='lightgray', alpha=0.3, s=15)

    ax2.set_title('Buggy Embedding (t-SNE)\n"class_0", "class_1", ...', fontsize=12, fontweight='bold')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')

    # Plot 3: Correct UMAP
    ax3 = fig.add_subplot(gs[1, 0])
    for group in group_names:
        mask = [class_to_group[i] == group for i in range(1000)]
        mask = np.array(mask)
        if mask.sum() > 0 and group != 'Other':
            ax3.scatter(correct_umap[mask, 0], correct_umap[mask, 1],
                       c=[color_map[group]], label=group, alpha=0.7, s=25)
    ax3.scatter(correct_umap[other_mask, 0], correct_umap[other_mask, 1],
               c='lightgray', alpha=0.3, s=15)

    ax3.set_title('Correct Embedding (UMAP)\nSemantic clusters visible', fontsize=12, fontweight='bold')
    ax3.set_xlabel('UMAP 1')
    ax3.set_ylabel('UMAP 2')
    ax3.legend(loc='upper right', fontsize=8, ncol=2)

    # Plot 4: Buggy UMAP
    ax4 = fig.add_subplot(gs[1, 1])
    for group in group_names:
        mask = [class_to_group[i] == group for i in range(1000)]
        mask = np.array(mask)
        if mask.sum() > 0 and group != 'Other':
            ax4.scatter(buggy_umap[mask, 0], buggy_umap[mask, 1],
                       c=[color_map[group]], label=group, alpha=0.7, s=25)
    ax4.scatter(buggy_umap[other_mask, 0], buggy_umap[other_mask, 1],
               c='lightgray', alpha=0.3, s=15)

    ax4.set_title('Buggy Embedding (UMAP)\nNo semantic structure', fontsize=12, fontweight='bold')
    ax4.set_xlabel('UMAP 1')
    ax4.set_ylabel('UMAP 2')

    plt.suptitle('Embedding Quality Comparison for Diffusion Transformer\n' +
                 'Gemma-3 270M: Semantic Text vs Random Class IDs', fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(OUTPUT_DIR / 'embedding_comparison_tsne_umap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'embedding_comparison_tsne_umap.png'}")


def plot_dog_cluster_comparison(correct_emb, buggy_emb, class_names, class_to_group):
    """개 품종 클러스터 비교 (핵심 예시)"""
    print("Creating dog cluster comparison...")

    # 개 품종 인덱스 (151-268)
    dog_indices = list(range(151, 269))
    dog_names = [class_names[i].split(',')[0] for i in dog_indices]

    # 개 임베딩 추출
    correct_dogs = correct_emb[dog_indices]
    buggy_dogs = buggy_emb[dog_indices]

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=20, random_state=42, max_iter=1000)
    correct_dogs_2d = tsne.fit_transform(correct_dogs)
    buggy_dogs_2d = tsne.fit_transform(buggy_dogs)

    # 유사도 행렬
    correct_sim = (correct_dogs / np.linalg.norm(correct_dogs, axis=1, keepdims=True)) @ \
                  (correct_dogs / np.linalg.norm(correct_dogs, axis=1, keepdims=True)).T
    buggy_sim = (buggy_dogs / np.linalg.norm(buggy_dogs, axis=1, keepdims=True)) @ \
                (buggy_dogs / np.linalg.norm(buggy_dogs, axis=1, keepdims=True)).T

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Correct embeddings
    # t-SNE
    axes[0, 0].scatter(correct_dogs_2d[:, 0], correct_dogs_2d[:, 1],
                       c='forestgreen', alpha=0.7, s=40)
    # 일부 라벨 표시
    for i in range(0, len(dog_names), 10):
        axes[0, 0].annotate(dog_names[i][:12], (correct_dogs_2d[i, 0], correct_dogs_2d[i, 1]),
                           fontsize=7, alpha=0.8)
    axes[0, 0].set_title('Correct: Dog Breeds t-SNE\n(Similar breeds cluster together)',
                         fontsize=11, fontweight='bold', color='darkgreen')
    axes[0, 0].set_xlabel('t-SNE 1')
    axes[0, 0].set_ylabel('t-SNE 2')

    # Similarity heatmap
    im1 = axes[0, 1].imshow(correct_sim, cmap='RdYlBu_r', aspect='auto', vmin=0.3, vmax=1.0)
    axes[0, 1].set_title('Correct: Dog Breeds Similarity\n(High within-group similarity)',
                         fontsize=11, fontweight='bold', color='darkgreen')
    axes[0, 1].set_xlabel('Dog Breed Index')
    axes[0, 1].set_ylabel('Dog Breed Index')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    # Similarity distribution
    upper_tri = correct_sim[np.triu_indices(len(dog_indices), k=1)]
    axes[0, 2].hist(upper_tri, bins=50, color='forestgreen', alpha=0.7, edgecolor='white')
    axes[0, 2].axvline(np.mean(upper_tri), color='red', linestyle='--',
                       label=f'Mean: {np.mean(upper_tri):.3f}')
    axes[0, 2].set_title('Correct: Similarity Distribution\n(Concentrated at high values)',
                         fontsize=11, fontweight='bold', color='darkgreen')
    axes[0, 2].set_xlabel('Cosine Similarity')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    axes[0, 2].set_xlim(-0.5, 1.0)

    # Row 2: Buggy embeddings
    axes[1, 0].scatter(buggy_dogs_2d[:, 0], buggy_dogs_2d[:, 1],
                       c='crimson', alpha=0.7, s=40)
    for i in range(0, len(dog_names), 10):
        axes[1, 0].annotate(f'class_{151+i}', (buggy_dogs_2d[i, 0], buggy_dogs_2d[i, 1]),
                           fontsize=7, alpha=0.8)
    axes[1, 0].set_title('Buggy: "class_151" to "class_268" t-SNE\n(Random distribution)',
                         fontsize=11, fontweight='bold', color='darkred')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')

    im2 = axes[1, 1].imshow(buggy_sim, cmap='RdYlBu_r', aspect='auto', vmin=0.3, vmax=1.0)
    axes[1, 1].set_title('Buggy: Similarity Matrix\n(No structure, random noise)',
                         fontsize=11, fontweight='bold', color='darkred')
    axes[1, 1].set_xlabel('Class Index')
    axes[1, 1].set_ylabel('Class Index')
    plt.colorbar(im2, ax=axes[1, 1], shrink=0.8)

    upper_tri_buggy = buggy_sim[np.triu_indices(len(dog_indices), k=1)]
    axes[1, 2].hist(upper_tri_buggy, bins=50, color='crimson', alpha=0.7, edgecolor='white')
    axes[1, 2].axvline(np.mean(upper_tri_buggy), color='blue', linestyle='--',
                       label=f'Mean: {np.mean(upper_tri_buggy):.3f}')
    axes[1, 2].set_title('Buggy: Similarity Distribution\n(Centered at ~0, no semantic info)',
                         fontsize=11, fontweight='bold', color='darkred')
    axes[1, 2].set_xlabel('Cosine Similarity')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    axes[1, 2].set_xlim(-0.5, 1.0)

    plt.suptitle('Dog Breeds (118 classes): Semantic Embedding vs Random Embedding\n' +
                 'For Diffusion Transformer: Semantic embeddings enable learning visual-text relationships',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dog_cluster_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'dog_cluster_comparison.png'}")

    return np.mean(upper_tri), np.mean(upper_tri_buggy)


def plot_semantic_neighbors_comparison(correct_emb, buggy_emb, class_names):
    """의미적 이웃 비교"""
    print("Creating semantic neighbors comparison...")

    # 쿼리 클래스들
    queries = [
        ('golden retriever', 207),
        ('German shepherd', 235),
        ('tabby cat', 281),
        ('lion', 291),
        ('pizza', 963),
        ('sports car', 817),
    ]

    # 유사도 행렬
    correct_norm = correct_emb / np.linalg.norm(correct_emb, axis=1, keepdims=True)
    buggy_norm = buggy_emb / np.linalg.norm(buggy_emb, axis=1, keepdims=True)
    correct_sim = correct_norm @ correct_norm.T
    buggy_sim = buggy_norm @ buggy_norm.T

    fig, axes = plt.subplots(len(queries), 2, figsize=(16, 4 * len(queries)))

    for row, (query_name, query_idx) in enumerate(queries):
        # Correct neighbors
        correct_neighbors_idx = np.argsort(correct_sim[query_idx])[::-1][1:8]
        correct_neighbors_names = [class_names[i].split(',')[0][:20] for i in correct_neighbors_idx]
        correct_neighbors_sims = correct_sim[query_idx, correct_neighbors_idx]

        axes[row, 0].barh(range(len(correct_neighbors_names)), correct_neighbors_sims,
                          color='forestgreen', alpha=0.8)
        axes[row, 0].set_yticks(range(len(correct_neighbors_names)))
        axes[row, 0].set_yticklabels(correct_neighbors_names, fontsize=9)
        axes[row, 0].invert_yaxis()
        axes[row, 0].set_xlim(0, 1)
        axes[row, 0].set_xlabel('Cosine Similarity')
        axes[row, 0].set_title(f'Correct: Neighbors of "{query_name}"\n(Semantically related)',
                               fontsize=10, fontweight='bold', color='darkgreen')

        # Buggy neighbors
        buggy_neighbors_idx = np.argsort(buggy_sim[query_idx])[::-1][1:8]
        buggy_neighbors_names = [f'class_{i}' for i in buggy_neighbors_idx]
        buggy_neighbors_sims = buggy_sim[query_idx, buggy_neighbors_idx]

        axes[row, 1].barh(range(len(buggy_neighbors_names)), buggy_neighbors_sims,
                          color='crimson', alpha=0.8)
        axes[row, 1].set_yticks(range(len(buggy_neighbors_names)))
        axes[row, 1].set_yticklabels(buggy_neighbors_names, fontsize=9)
        axes[row, 1].invert_yaxis()
        axes[row, 1].set_xlim(0, 1)
        axes[row, 1].set_xlabel('Cosine Similarity')
        axes[row, 1].set_title(f'Buggy: Neighbors of "class_{query_idx}"\n(Random, no meaning)',
                               fontsize=10, fontweight='bold', color='darkred')

    plt.suptitle('Semantic Neighbors: Correct vs Buggy Embedding\n' +
                 'Diffusion Transformer benefits from semantic clustering for text-to-image generation',
                 fontsize=13, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'semantic_neighbors_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'semantic_neighbors_comparison.png'}")


def plot_metrics_summary(correct_metrics, buggy_metrics, dog_correct_sim, dog_buggy_sim):
    """메트릭 요약 비교"""
    print("Creating metrics summary...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. 전체 메트릭 비교
    metrics_names = ['Same Group\nSimilarity', 'Different Group\nSimilarity', 'Separation\nRatio']
    correct_values = [correct_metrics['mean_same_group'],
                      correct_metrics['mean_diff_group'],
                      correct_metrics['separation_ratio']]
    buggy_values = [buggy_metrics['mean_same_group'],
                    buggy_metrics['mean_diff_group'],
                    buggy_metrics['separation_ratio']]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = axes[0].bar(x - width/2, correct_values, width, label='Correct (Semantic)',
                        color='forestgreen', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, buggy_values, width, label='Buggy (class_i)',
                        color='crimson', alpha=0.8)

    axes[0].set_ylabel('Value')
    axes[0].set_title('Embedding Quality Metrics\n(Higher separation = better for DiT)',
                      fontsize=11, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # 값 표시
    for bar, val in zip(bars1, correct_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, buggy_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # 2. Dog breeds 비교
    dog_metrics = ['Dog Breeds\nAvg Similarity']
    dog_correct = [dog_correct_sim]
    dog_buggy = [dog_buggy_sim]

    x2 = np.arange(1)
    bars3 = axes[1].bar(x2 - width/2, dog_correct, width, label='Correct',
                        color='forestgreen', alpha=0.8)
    bars4 = axes[1].bar(x2 + width/2, dog_buggy, width, label='Buggy',
                        color='crimson', alpha=0.8)

    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Dog Breeds (118 classes)\nWithin-Group Similarity',
                      fontsize=11, fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(dog_metrics)
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[1].text(bars3[0].get_x() + bars3[0].get_width()/2, bars3[0].get_height() + 0.02,
                f'{dog_correct_sim:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[1].text(bars4[0].get_x() + bars4[0].get_width()/2, bars4[0].get_height() + 0.02,
                f'{dog_buggy_sim:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 3. 장점 요약 (텍스트)
    axes[2].axis('off')

    summary_text = """
    Why Semantic Embeddings Matter for Diffusion Transformer:

    Correct Embedding ("golden retriever", "tabby cat"):
    ✓ Similar concepts cluster together
    ✓ Dog breeds share embedding space region
    ✓ Model learns visual-semantic correspondence
    ✓ Better generalization to similar classes
    ✓ Enables smooth interpolation between classes

    Buggy Embedding ("class_0", "class_1"):
    ✗ Random distribution, no structure
    ✗ Model must memorize each class separately
    ✗ No transfer between similar classes
    ✗ Inefficient learning
    ✗ Poor interpolation capability

    Result: {:.1f}x better semantic separation
            {:.1f}x higher within-group similarity (dogs)
    """.format(
        correct_metrics['separation_ratio'] / max(buggy_metrics['separation_ratio'], 0.01),
        dog_correct_sim / max(dog_buggy_sim, 0.01)
    )

    axes[2].text(0.1, 0.9, summary_text, transform=axes[2].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[2].set_title('Summary: Benefits for Diffusion Transformer',
                      fontsize=11, fontweight='bold')

    plt.suptitle('Gemma-3 270M Embedding Quality: Impact on Diffusion Transformer Training',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'embedding_metrics_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'embedding_metrics_summary.png'}")


def plot_3d_comparison(correct_emb, buggy_emb, class_to_group, groups):
    """3D UMAP 비교 (수정된 버전)"""
    print("Creating 3D UMAP comparison...")

    fig = plt.figure(figsize=(18, 8))

    # UMAP 3D
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)

    print("  Computing UMAP 3D for correct embeddings...")
    correct_3d = reducer.fit_transform(correct_emb)

    print("  Computing UMAP 3D for buggy embeddings...")
    buggy_3d = reducer.fit_transform(buggy_emb)

    # 색상 설정
    group_names = list(groups.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(group_names)))

    # Correct 3D
    ax1 = fig.add_subplot(121, projection='3d')
    for i, group in enumerate(group_names):
        mask = np.array([class_to_group[j] == group for j in range(1000)])
        if mask.sum() > 0:
            ax1.scatter(correct_3d[mask, 0], correct_3d[mask, 1], correct_3d[mask, 2],
                       c=[colors[i]], label=group, alpha=0.7, s=20)

    ax1.set_title('Correct Embedding (UMAP 3D)\nSemantic Clusters', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=7)

    # Buggy 3D
    ax2 = fig.add_subplot(122, projection='3d')
    for i, group in enumerate(group_names):
        mask = np.array([class_to_group[j] == group for j in range(1000)])
        if mask.sum() > 0:
            ax2.scatter(buggy_3d[mask, 0], buggy_3d[mask, 1], buggy_3d[mask, 2],
                       c=[colors[i]], label=group, alpha=0.7, s=20)

    ax2.set_title('Buggy Embedding (UMAP 3D)\nRandom Distribution', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=7)

    plt.suptitle('3D Embedding Space Comparison\nGemma-3 270M for Diffusion Transformer',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'embedding_3d_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'embedding_3d_comparison.png'}")


def main():
    """메인 실행"""
    print("=" * 70)
    print("Embedding Quality Comparison: Correct vs Buggy")
    print("For Diffusion Transformer (XUDiT) Training")
    print("=" * 70)

    # 데이터 로드
    print("\n[1/6] Loading data...")
    correct_emb, class_names = load_correct_embeddings()
    buggy_emb, buggy_names = create_buggy_embeddings()
    groups, class_to_group = get_semantic_groups()

    print(f"  Correct embeddings: {correct_emb.shape}")
    print(f"  Buggy embeddings: {buggy_emb.shape}")
    print(f"  Semantic groups: {list(groups.keys())}")

    # 메트릭 계산
    print("\n[2/6] Computing metrics...")
    correct_metrics, _ = compute_metrics(correct_emb, class_to_group)
    buggy_metrics, _ = compute_metrics(buggy_emb, class_to_group)

    print(f"  Correct - Same group sim: {correct_metrics['mean_same_group']:.4f}")
    print(f"  Correct - Diff group sim: {correct_metrics['mean_diff_group']:.4f}")
    print(f"  Correct - Separation: {correct_metrics['separation_ratio']:.2f}x")
    print(f"  Buggy - Same group sim: {buggy_metrics['mean_same_group']:.4f}")
    print(f"  Buggy - Diff group sim: {buggy_metrics['mean_diff_group']:.4f}")
    print(f"  Buggy - Separation: {buggy_metrics['separation_ratio']:.2f}x")

    # 시각화
    print("\n[3/6] Creating t-SNE/UMAP comparison...")
    plot_comparison_tsne_umap(correct_emb, buggy_emb, class_names, class_to_group, groups)

    print("\n[4/6] Creating dog cluster comparison...")
    dog_correct_sim, dog_buggy_sim = plot_dog_cluster_comparison(
        correct_emb, buggy_emb, class_names, class_to_group)

    print("\n[5/6] Creating semantic neighbors comparison...")
    plot_semantic_neighbors_comparison(correct_emb, buggy_emb, class_names)

    print("\n[6/6] Creating metrics summary...")
    plot_metrics_summary(correct_metrics, buggy_metrics, dog_correct_sim, dog_buggy_sim)

    # 3D 비교
    print("\n[Bonus] Creating 3D comparison...")
    plot_3d_comparison(correct_emb, buggy_emb, class_to_group, groups)

    print("\n" + "=" * 70)
    print("All comparisons completed!")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print("=" * 70)

    # 최종 요약
    print("\n" + "=" * 70)
    print("SUMMARY: Why Semantic Embeddings Matter for Diffusion Transformer")
    print("=" * 70)
    print(f"""
    Semantic Separation Ratio:
      Correct: {correct_metrics['separation_ratio']:.2f}x
      Buggy:   {buggy_metrics['separation_ratio']:.2f}x
      Improvement: {correct_metrics['separation_ratio'] / max(buggy_metrics['separation_ratio'], 0.01):.1f}x better

    Dog Breeds Within-Group Similarity:
      Correct: {dog_correct_sim:.3f}
      Buggy:   {dog_buggy_sim:.3f}
      Improvement: {dog_correct_sim / max(dog_buggy_sim, 0.01):.1f}x higher cohesion

    → Semantic embeddings enable:
      1. Meaningful text-to-image correspondence learning
      2. Transfer learning between similar classes
      3. Smooth latent space interpolation
      4. Better generalization
    """)


if __name__ == "__main__":
    main()
