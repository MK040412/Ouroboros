"""
Gemma-3 270M ImageNet 1K Class Embedding 시각화

이 스크립트는 Gemma-3 270M 모델로 생성된 ImageNet 1K 클래스 텍스트 임베딩의
분포와 특성을 다양한 방법으로 시각화합니다.

시각화 항목:
1. t-SNE 2D/3D projection
2. UMAP 2D/3D projection
3. PCA 분석 및 variance explained
4. 클래스 간 유사도 히트맵
5. 계층적 클러스터링 덴드로그램
6. 의미적 이웃 분석 (semantic neighborhood)
7. Intra-class vs Inter-class 거리 분석
8. 특정 카테고리(동물, 사물, 음식 등) 별 클러스터링
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
import sys
from collections import OrderedDict

# 출력 디렉토리
OUTPUT_DIR = Path("outputs/embedding_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_embeddings_and_classes():
    """임베딩과 클래스 정보 로드"""
    print("Loading embeddings and class labels...")

    # 임베딩 로드
    emb_path = OUTPUT_DIR / "imagenet_class_embeddings.npy"
    embeddings = np.load(emb_path)
    print(f"  Embeddings shape: {embeddings.shape}")

    # classes.py 로드
    classes_path = OUTPUT_DIR / "classes.py"
    with open(classes_path, 'r') as f:
        content = f.read()

    local_vars = {}
    exec(content, {}, local_vars)
    class_dict = local_vars['IMAGENET2012_CLASSES']

    # OrderedDict에서 리스트로 변환
    class_names = list(class_dict.values())
    synset_ids = list(class_dict.keys())

    print(f"  Loaded {len(class_names)} class names")
    print(f"  Sample classes: {class_names[:5]}")

    return embeddings, class_names, synset_ids


def get_category_mapping():
    """ImageNet 클래스를 대분류 카테고리로 매핑"""
    # 주요 카테고리 정의 (synset 범위 기반)
    categories = {
        'Fish': list(range(0, 8)),  # tench ~ stingray
        'Birds': list(range(7, 24)) + list(range(80, 100)),
        'Amphibians/Reptiles': list(range(25, 69)),  # salamander ~ snake
        'Invertebrates': list(range(69, 80)) + list(range(300, 320)),
        'Dogs': list(range(151, 269)),  # Chihuahua ~ Mexican hairless
        'Wild Mammals': list(range(269, 300)) + list(range(120, 150)),
        'Primates': list(range(365, 385)),
        'Food': list(range(924, 970)),
        'Vehicles': list(range(404, 450)) + list(range(600, 650)),
        'Furniture/Household': list(range(450, 530)),
        'Musical Instruments': list(range(400, 420)),
        'Electronics': list(range(500, 550)),
        'Clothing': list(range(600, 640)),
        'Plants/Flowers': list(range(970, 1000)),
    }

    # 각 클래스에 카테고리 할당
    class_to_category = {}
    category_to_idx = {}

    for cat_idx, (cat_name, indices) in enumerate(categories.items()):
        category_to_idx[cat_name] = cat_idx
        for idx in indices:
            if idx < 1000:
                class_to_category[idx] = cat_name

    # 할당되지 않은 클래스는 'Other'
    for i in range(1000):
        if i not in class_to_category:
            class_to_category[i] = 'Other'

    if 'Other' not in category_to_idx:
        category_to_idx['Other'] = len(category_to_idx)

    return class_to_category, category_to_idx


def plot_tsne_2d(embeddings, class_names, class_to_category, category_to_idx):
    """t-SNE 2D 시각화"""
    print("\n[1/8] Creating t-SNE 2D visualization...")

    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 카테고리별 색상
    categories = list(category_to_idx.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
    color_map = {cat: colors[i] for i, cat in enumerate(categories)}

    fig, ax = plt.subplots(figsize=(16, 12))

    for cat in categories:
        mask = [class_to_category[i] == cat for i in range(len(class_names))]
        mask = np.array(mask)
        if mask.sum() > 0:
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color_map[cat]],
                label=cat,
                alpha=0.7,
                s=30
            )

    ax.set_title('Gemma-3 270M ImageNet 1K Class Embeddings (t-SNE 2D)', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tsne_2d_categories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'tsne_2d_categories.png'}")

    return embeddings_2d


def plot_umap_2d(embeddings, class_names, class_to_category, category_to_idx):
    """UMAP 2D 시각화"""
    print("\n[2/8] Creating UMAP 2D visualization...")

    try:
        import umap
    except ImportError:
        print("  UMAP not installed, skipping... (pip install umap-learn)")
        return None

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    categories = list(category_to_idx.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
    color_map = {cat: colors[i] for i, cat in enumerate(categories)}

    fig, ax = plt.subplots(figsize=(16, 12))

    for cat in categories:
        mask = [class_to_category[i] == cat for i in range(len(class_names))]
        mask = np.array(mask)
        if mask.sum() > 0:
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color_map[cat]],
                label=cat,
                alpha=0.7,
                s=30
            )

    ax.set_title('Gemma-3 270M ImageNet 1K Class Embeddings (UMAP 2D)', fontsize=14)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'umap_2d_categories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'umap_2d_categories.png'}")

    return embeddings_2d


def plot_pca_analysis(embeddings, class_names):
    """PCA 분석 및 시각화"""
    print("\n[3/8] Creating PCA analysis...")

    from sklearn.decomposition import PCA

    # Full PCA for variance analysis
    pca_full = PCA()
    pca_full.fit(embeddings)

    # Variance explained plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Cumulative variance explained
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    axes[0].plot(range(1, len(cumvar)+1), cumvar, 'b-', linewidth=2)
    axes[0].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    axes[0].axhline(y=0.99, color='g', linestyle='--', label='99% variance')

    # Find components for 95% and 99% variance
    n_95 = np.argmax(cumvar >= 0.95) + 1
    n_99 = np.argmax(cumvar >= 0.99) + 1
    axes[0].axvline(x=n_95, color='r', linestyle=':', alpha=0.5)
    axes[0].axvline(x=n_99, color='g', linestyle=':', alpha=0.5)

    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Cumulative Explained Variance')
    axes[0].set_title(f'PCA Cumulative Variance\n(95%: {n_95} comp, 99%: {n_99} comp)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Individual variance explained (first 50 components)
    axes[1].bar(range(1, 51), pca_full.explained_variance_ratio_[:50], color='steelblue')
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Explained Variance Ratio')
    axes[1].set_title('Variance Explained by Each Component (Top 50)')
    axes[1].grid(True, alpha=0.3)

    # 3. 2D PCA projection
    pca_2d = PCA(n_components=2)
    embeddings_2d = pca_2d.fit_transform(embeddings)

    axes[2].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=20, c='steelblue')
    axes[2].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
    axes[2].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
    axes[2].set_title('PCA 2D Projection of Embeddings')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pca_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'pca_analysis.png'}")
    print(f"  Components for 95% variance: {n_95}")
    print(f"  Components for 99% variance: {n_99}")

    return pca_full, n_95, n_99


def plot_similarity_heatmap(embeddings, class_names):
    """클래스 간 코사인 유사도 히트맵"""
    print("\n[4/8] Creating similarity heatmap...")

    # 코사인 유사도 계산
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-8)
    similarity_matrix = normalized @ normalized.T

    # 전체 히트맵 (1000x1000은 너무 크므로 축소 버전)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. 전체 히트맵 (축소)
    im1 = axes[0].imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-0.2, vmax=1.0)
    axes[0].set_title('Full Similarity Matrix (1000 Classes)', fontsize=12)
    axes[0].set_xlabel('Class Index')
    axes[0].set_ylabel('Class Index')
    plt.colorbar(im1, ax=axes[0], label='Cosine Similarity')

    # 2. 특정 구간 확대 (개 품종: 151-269)
    dogs_start, dogs_end = 151, 220
    dog_sim = similarity_matrix[dogs_start:dogs_end, dogs_start:dogs_end]
    dog_names = class_names[dogs_start:dogs_end]

    im2 = axes[1].imshow(dog_sim, cmap='RdYlBu_r', aspect='auto', vmin=0.3, vmax=1.0)
    axes[1].set_title('Dog Breeds Similarity (Classes 151-220)', fontsize=12)

    # 일부 라벨만 표시
    tick_positions = list(range(0, len(dog_names), 5))
    tick_labels = [dog_names[i].split(',')[0][:15] for i in tick_positions]
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)
    axes[1].set_yticks(tick_positions)
    axes[1].set_yticklabels(tick_labels, fontsize=7)
    plt.colorbar(im2, ax=axes[1], label='Cosine Similarity')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'similarity_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'similarity_heatmap.png'}")

    return similarity_matrix


def plot_hierarchical_clustering(embeddings, class_names):
    """계층적 클러스터링 덴드로그램"""
    print("\n[5/8] Creating hierarchical clustering dendrogram...")

    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    # 코사인 거리로 클러스터링
    distances = pdist(embeddings, metric='cosine')
    linkage_matrix = linkage(distances, method='ward')

    # 상위 50개 클러스터만 표시
    fig, ax = plt.subplots(figsize=(20, 10))

    dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        p=50,  # 마지막 50개 merge만 표시
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax
    )

    ax.set_title('Hierarchical Clustering of ImageNet 1K Classes (Top 50 Clusters)', fontsize=14)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Distance')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hierarchical_clustering.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'hierarchical_clustering.png'}")

    return linkage_matrix


def plot_semantic_neighbors(embeddings, class_names, query_classes=None):
    """의미적 이웃 분석"""
    print("\n[6/8] Creating semantic neighbor analysis...")

    if query_classes is None:
        # 다양한 카테고리에서 샘플링
        query_classes = [
            ('golden retriever', 207),
            ('tabby, tabby cat', 281),
            ('sports car, sport car', 817),
            ('pizza, pizza pie', 963),
            ('acoustic guitar', 402),
            ('lion, king of beasts, Panthera leo', 291),
            ('strawberry', 949),
            ('laptop, laptop computer', 620),
        ]

    # 코사인 유사도 계산
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-8)
    similarity_matrix = normalized @ normalized.T

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for ax_idx, (query_name, query_idx) in enumerate(query_classes[:8]):
        if ax_idx >= len(axes):
            break

        # Top-10 이웃 찾기
        similarities = similarity_matrix[query_idx]
        top_indices = np.argsort(similarities)[::-1][1:11]  # 자기 자신 제외
        top_names = [class_names[i].split(',')[0][:25] for i in top_indices]
        top_sims = similarities[top_indices]

        # 막대 그래프
        y_pos = range(len(top_names))
        bars = axes[ax_idx].barh(y_pos, top_sims, color='steelblue', alpha=0.8)
        axes[ax_idx].set_yticks(y_pos)
        axes[ax_idx].set_yticklabels(top_names, fontsize=8)
        axes[ax_idx].invert_yaxis()
        axes[ax_idx].set_xlabel('Cosine Similarity')
        axes[ax_idx].set_title(f'Neighbors of "{query_name.split(",")[0]}"', fontsize=10)
        axes[ax_idx].set_xlim(0, 1)

        # 값 표시
        for bar, sim in zip(bars, top_sims):
            axes[ax_idx].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{sim:.3f}', va='center', fontsize=7)

    plt.suptitle('Semantic Neighbors in Gemma-3 270M Embedding Space', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'semantic_neighbors.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'semantic_neighbors.png'}")


def plot_distance_distribution(embeddings, class_names, class_to_category):
    """클래스 간 거리 분포 분석"""
    print("\n[7/8] Creating distance distribution analysis...")

    # 코사인 유사도 계산
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-8)
    similarity_matrix = normalized @ normalized.T

    # 같은 카테고리 내 유사도 vs 다른 카테고리 유사도
    same_cat_sims = []
    diff_cat_sims = []

    for i in range(1000):
        for j in range(i+1, 1000):
            sim = similarity_matrix[i, j]
            if class_to_category[i] == class_to_category[j]:
                same_cat_sims.append(sim)
            else:
                diff_cat_sims.append(sim)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 전체 유사도 분포
    all_sims = similarity_matrix[np.triu_indices(1000, k=1)]
    axes[0].hist(all_sims, bins=100, color='steelblue', alpha=0.7, edgecolor='white')
    axes[0].axvline(np.mean(all_sims), color='red', linestyle='--', label=f'Mean: {np.mean(all_sims):.3f}')
    axes[0].axvline(np.median(all_sims), color='orange', linestyle='--', label=f'Median: {np.median(all_sims):.3f}')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('All Pairwise Similarities Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 같은 카테고리 vs 다른 카테고리
    axes[1].hist(same_cat_sims, bins=50, alpha=0.6, label=f'Same Category (n={len(same_cat_sims):,})', color='green')
    axes[1].hist(diff_cat_sims, bins=50, alpha=0.6, label=f'Different Category (n={len(diff_cat_sims):,})', color='red')
    axes[1].axvline(np.mean(same_cat_sims), color='darkgreen', linestyle='--')
    axes[1].axvline(np.mean(diff_cat_sims), color='darkred', linestyle='--')
    axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Intra-Category vs Inter-Category Similarities')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 카테고리별 평균 유사도
    categories = list(set(class_to_category.values()))
    cat_avg_sims = {}

    for cat in categories:
        cat_indices = [i for i in range(1000) if class_to_category[i] == cat]
        if len(cat_indices) > 1:
            cat_sims = []
            for i in range(len(cat_indices)):
                for j in range(i+1, len(cat_indices)):
                    cat_sims.append(similarity_matrix[cat_indices[i], cat_indices[j]])
            if cat_sims:
                cat_avg_sims[cat] = np.mean(cat_sims)

    # 정렬하여 표시
    sorted_cats = sorted(cat_avg_sims.items(), key=lambda x: x[1], reverse=True)
    cat_names = [c[0] for c in sorted_cats]
    cat_values = [c[1] for c in sorted_cats]

    bars = axes[2].barh(range(len(cat_names)), cat_values, color='steelblue', alpha=0.8)
    axes[2].set_yticks(range(len(cat_names)))
    axes[2].set_yticklabels(cat_names, fontsize=9)
    axes[2].invert_yaxis()
    axes[2].set_xlabel('Average Intra-Category Cosine Similarity')
    axes[2].set_title('Category Cohesion (Avg. Similarity within Category)')
    axes[2].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distance_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'distance_distribution.png'}")

    # 통계 출력
    print(f"\n  Statistics:")
    print(f"    All pairs mean similarity: {np.mean(all_sims):.4f}")
    print(f"    Same category mean: {np.mean(same_cat_sims):.4f}")
    print(f"    Different category mean: {np.mean(diff_cat_sims):.4f}")
    print(f"    Separation ratio: {np.mean(same_cat_sims) / np.mean(diff_cat_sims):.2f}x")


def plot_embedding_statistics(embeddings, class_names):
    """임베딩 통계 시각화"""
    print("\n[8/8] Creating embedding statistics visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. L2 norm 분포
    norms = np.linalg.norm(embeddings, axis=1)
    # L2 normalized embeddings는 norms가 거의 동일할 수 있음
    norm_range = norms.max() - norms.min()
    if norm_range < 1e-6:
        axes[0, 0].bar([np.mean(norms)], [len(norms)], width=0.01, color='steelblue', alpha=0.7)
        axes[0, 0].set_title(f'Embedding L2 Norm Distribution\n(All norms ≈ {np.mean(norms):.4f})')
    else:
        axes[0, 0].hist(norms, bins='auto', color='steelblue', alpha=0.7, edgecolor='white')
        axes[0, 0].axvline(np.mean(norms), color='red', linestyle='--', label=f'Mean: {np.mean(norms):.3f}')
        axes[0, 0].set_title('Embedding L2 Norm Distribution')
        axes[0, 0].legend()
    axes[0, 0].set_xlabel('L2 Norm')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 차원별 평균 및 표준편차
    dim_means = np.mean(embeddings, axis=0)
    dim_stds = np.std(embeddings, axis=0)

    axes[0, 1].plot(dim_means, 'b-', alpha=0.7, linewidth=0.5)
    axes[0, 1].fill_between(range(len(dim_means)), dim_means - dim_stds, dim_means + dim_stds, alpha=0.3)
    axes[0, 1].set_xlabel('Dimension')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_title('Mean ± Std per Embedding Dimension')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 차원별 분산 (활성화된 차원 확인)
    dim_vars = np.var(embeddings, axis=0)
    sorted_vars = np.sort(dim_vars)[::-1]

    axes[0, 2].bar(range(len(sorted_vars)), sorted_vars, color='steelblue', alpha=0.7)
    axes[0, 2].set_xlabel('Dimension (sorted by variance)')
    axes[0, 2].set_ylabel('Variance')
    axes[0, 2].set_title('Dimension Variance (Sorted)')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 가장 유사한 쌍들
    norms_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim_matrix = norms_normalized @ norms_normalized.T
    np.fill_diagonal(sim_matrix, -1)  # 자기 자신 제외

    # 상위 20개 유사 쌍
    flat_idx = np.argsort(sim_matrix.flatten())[::-1][:20]
    top_pairs = []
    for idx in flat_idx:
        i, j = idx // 1000, idx % 1000
        if i < j:  # 중복 제거
            top_pairs.append((i, j, sim_matrix[i, j]))

    pair_labels = [f"{class_names[p[0]].split(',')[0][:12]} - {class_names[p[1]].split(',')[0][:12]}"
                   for p in top_pairs[:10]]
    pair_sims = [p[2] for p in top_pairs[:10]]

    axes[1, 0].barh(range(len(pair_labels)), pair_sims, color='green', alpha=0.7)
    axes[1, 0].set_yticks(range(len(pair_labels)))
    axes[1, 0].set_yticklabels(pair_labels, fontsize=8)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel('Cosine Similarity')
    axes[1, 0].set_title('Most Similar Class Pairs')
    axes[1, 0].set_xlim(0, 1)

    # 5. 가장 다른 쌍들
    bottom_pairs = []
    for idx in np.argsort(sim_matrix.flatten())[:20]:
        i, j = idx // 1000, idx % 1000
        if i < j:
            bottom_pairs.append((i, j, sim_matrix[i, j]))

    pair_labels_bottom = [f"{class_names[p[0]].split(',')[0][:12]} - {class_names[p[1]].split(',')[0][:12]}"
                          for p in bottom_pairs[:10]]
    pair_sims_bottom = [p[2] for p in bottom_pairs[:10]]

    axes[1, 1].barh(range(len(pair_labels_bottom)), pair_sims_bottom, color='red', alpha=0.7)
    axes[1, 1].set_yticks(range(len(pair_labels_bottom)))
    axes[1, 1].set_yticklabels(pair_labels_bottom, fontsize=8)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel('Cosine Similarity')
    axes[1, 1].set_title('Most Dissimilar Class Pairs')

    # 6. 클러스터 수 추정 (실루엣 스코어)
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    k_range = range(5, 51, 5)
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, metric='cosine')
        silhouette_scores.append(score)

    axes[1, 2].plot(list(k_range), silhouette_scores, 'bo-', linewidth=2)
    best_k = list(k_range)[np.argmax(silhouette_scores)]
    axes[1, 2].axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
    axes[1, 2].set_xlabel('Number of Clusters (k)')
    axes[1, 2].set_ylabel('Silhouette Score')
    axes[1, 2].set_title('Optimal Cluster Number Estimation')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('Gemma-3 270M Embedding Space Statistics', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'embedding_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'embedding_statistics.png'}")


def plot_3d_visualization(embeddings, class_names, class_to_category, category_to_idx):
    """3D 시각화 (t-SNE & UMAP)"""
    print("\n[Bonus] Creating 3D visualizations...")

    from sklearn.manifold import TSNE

    # t-SNE 3D
    tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42, max_iter=1000)
    embeddings_3d = tsne_3d.fit_transform(embeddings)

    fig = plt.figure(figsize=(16, 7))

    # t-SNE 3D
    ax1 = fig.add_subplot(121, projection='3d')

    categories = list(category_to_idx.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))

    for cat_idx, cat in enumerate(categories):
        mask = [class_to_category[i] == cat for i in range(len(class_names))]
        mask = np.array(mask)
        if mask.sum() > 0:
            ax1.scatter(
                embeddings_3d[mask, 0],
                embeddings_3d[mask, 1],
                embeddings_3d[mask, 2],
                c=[colors[cat_idx]],
                label=cat if cat_idx < 10 else None,
                alpha=0.6,
                s=20
            )

    ax1.set_title('t-SNE 3D Projection')
    ax1.legend(loc='upper left', fontsize=7)

    # UMAP 3D (if available)
    try:
        import umap

        reducer_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
        embeddings_umap_3d = reducer_3d.fit_transform(embeddings)

        ax2 = fig.add_subplot(122, projection='3d')

        for cat_idx, cat in enumerate(categories):
            mask = [class_to_category[i] == cat for i in range(len(class_names))]
            mask = np.array(mask)
            if mask.sum() > 0:
                ax2.scatter(
                    embeddings_umap_3d[mask, 0],
                    embeddings_umap_3d[mask, 1],
                    embeddings_umap_3d[mask, 2],
                    c=[colors[cat_idx]],
                    label=cat if cat_idx < 10 else None,
                    alpha=0.6,
                    s=20
                )

        ax2.set_title('UMAP 3D Projection')
        ax2.legend(loc='upper left', fontsize=7)

    except ImportError:
        ax2 = fig.add_subplot(122)
        ax2.text(0.5, 0.5, 'UMAP not installed', ha='center', va='center')

    plt.suptitle('3D Embedding Space Visualizations', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'embedding_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'embedding_3d.png'}")


def plot_detailed_category_analysis(embeddings, class_names, class_to_category):
    """상세 카테고리 분석"""
    print("\n[Bonus] Creating detailed category analysis...")

    from sklearn.manifold import TSNE

    # 특정 카테고리만 추출하여 시각화
    focus_categories = {
        'Dogs': (151, 269),  # 개 품종
        'Birds': (80, 100),  # 새
        'Vehicles': (404, 480),  # 탈것
        'Food': (924, 970),  # 음식
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for ax_idx, (cat_name, (start, end)) in enumerate(focus_categories.items()):
        cat_embeddings = embeddings[start:end]
        cat_names = class_names[start:end]

        # t-SNE for this category
        if len(cat_embeddings) > 5:
            perplexity = min(30, len(cat_embeddings) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            cat_2d = tsne.fit_transform(cat_embeddings)

            axes[ax_idx].scatter(cat_2d[:, 0], cat_2d[:, 1], alpha=0.7, s=50, c='steelblue')

            # 일부 라벨 표시
            for i in range(0, len(cat_names), max(1, len(cat_names)//15)):
                short_name = cat_names[i].split(',')[0][:15]
                axes[ax_idx].annotate(short_name, (cat_2d[i, 0], cat_2d[i, 1]),
                                     fontsize=7, alpha=0.8)

            axes[ax_idx].set_title(f'{cat_name} (Classes {start}-{end})', fontsize=12)
            axes[ax_idx].set_xlabel('t-SNE 1')
            axes[ax_idx].set_ylabel('t-SNE 2')
        else:
            axes[ax_idx].text(0.5, 0.5, f'Not enough samples for {cat_name}',
                             ha='center', va='center')

    plt.suptitle('Detailed Category Analysis (t-SNE)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'category_detail.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'category_detail.png'}")


def create_summary_figure(embeddings, class_names, class_to_category, category_to_idx):
    """종합 요약 figure"""
    print("\n[Summary] Creating summary figure...")

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    fig = plt.figure(figsize=(20, 16))

    # 1. t-SNE 2D (왼쪽 상단, 큰 영역)
    ax1 = fig.add_subplot(2, 2, 1)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    categories = list(category_to_idx.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))

    for cat_idx, cat in enumerate(categories):
        mask = [class_to_category[i] == cat for i in range(len(class_names))]
        mask = np.array(mask)
        if mask.sum() > 0:
            ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[colors[cat_idx]], label=cat, alpha=0.7, s=15)

    ax1.set_title('t-SNE 2D Visualization by Category', fontsize=12)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)

    # 2. 유사도 히트맵 (오른쪽 상단)
    ax2 = fig.add_subplot(2, 2, 2)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-8)
    sim_matrix = normalized @ normalized.T

    im = ax2.imshow(sim_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-0.1, vmax=1.0)
    ax2.set_title('Cosine Similarity Matrix (1000 Classes)', fontsize=12)
    ax2.set_xlabel('Class Index')
    ax2.set_ylabel('Class Index')
    plt.colorbar(im, ax=ax2, shrink=0.8)

    # 3. PCA variance (왼쪽 하단)
    ax3 = fig.add_subplot(2, 2, 3)

    pca = PCA()
    pca.fit(embeddings)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    ax3.plot(range(1, len(cumvar)+1), cumvar, 'b-', linewidth=2)
    ax3.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95%')
    ax3.axhline(y=0.99, color='g', linestyle='--', alpha=0.7, label='99%')
    ax3.set_xlabel('Number of Components')
    ax3.set_ylabel('Cumulative Variance')
    ax3.set_title('PCA Cumulative Variance Explained', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 200)

    # 4. 거리 분포 (오른쪽 하단)
    ax4 = fig.add_subplot(2, 2, 4)

    same_cat_sims = []
    diff_cat_sims = []

    for i in range(0, 1000, 3):  # 샘플링으로 속도 향상
        for j in range(i+1, 1000, 3):
            sim = sim_matrix[i, j]
            if class_to_category[i] == class_to_category[j]:
                same_cat_sims.append(sim)
            else:
                diff_cat_sims.append(sim)

    ax4.hist(same_cat_sims, bins=40, alpha=0.6, label=f'Same Category', color='green', density=True)
    ax4.hist(diff_cat_sims, bins=40, alpha=0.6, label=f'Different Category', color='red', density=True)
    ax4.axvline(np.mean(same_cat_sims), color='darkgreen', linestyle='--', linewidth=2)
    ax4.axvline(np.mean(diff_cat_sims), color='darkred', linestyle='--', linewidth=2)
    ax4.set_xlabel('Cosine Similarity')
    ax4.set_ylabel('Density')
    ax4.set_title('Intra vs Inter Category Similarity', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Gemma-3 270M ImageNet 1K Embedding Space Analysis Summary', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'embedding_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'embedding_summary.png'}")


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("Gemma-3 270M ImageNet 1K Class Embedding Visualization")
    print("=" * 70)

    # 데이터 로드
    embeddings, class_names, synset_ids = load_embeddings_and_classes()
    class_to_category, category_to_idx = get_category_mapping()

    print(f"\nEmbedding dimension: {embeddings.shape[1]}")
    print(f"Number of classes: {embeddings.shape[0]}")
    print(f"Number of categories: {len(category_to_idx)}")

    # 시각화 생성
    plot_tsne_2d(embeddings, class_names, class_to_category, category_to_idx)
    plot_umap_2d(embeddings, class_names, class_to_category, category_to_idx)
    plot_pca_analysis(embeddings, class_names)
    plot_similarity_heatmap(embeddings, class_names)
    plot_hierarchical_clustering(embeddings, class_names)
    plot_semantic_neighbors(embeddings, class_names)
    plot_distance_distribution(embeddings, class_names, class_to_category)
    plot_embedding_statistics(embeddings, class_names)

    # 추가 시각화
    plot_3d_visualization(embeddings, class_names, class_to_category, category_to_idx)
    plot_detailed_category_analysis(embeddings, class_names, class_to_category)
    create_summary_figure(embeddings, class_names, class_to_category, category_to_idx)

    print("\n" + "=" * 70)
    print("All visualizations completed!")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
