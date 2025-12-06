#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ImageNet 클래스 임베딩 사전 계산 (Worker 0에서만 실행)
# 1000개 클래스만 처리하므로 분산 불필요
# =============================================================================

echo "=============================================="
echo "ImageNet Class Embeddings Pre-compute"
echo "=============================================="

# Worker 0에서 실행 중인지 확인
if [[ ! -f ~/.bashrc ]]; then
    echo "Error: This script should be run from a TPU VM"
    echo "Usage: gcloud compute tpus tpu-vm ssh ouroboros --zone=europe-west4-b --worker=0"
    echo "       Then run: ~/ouroboros/run_precompute_imagenet.sh"
    exit 1
fi

# 기존 프로세스 정리
pkill -9 -f precompute_imagenet 2>/dev/null || true

# Step 1: 코드 동기화
echo -e "\n[Step 1] Syncing code..."
cd ~/ouroboros
git pull origin main 2>/dev/null || true
echo "  Code synced"

# Step 2: Python 패키지 설치
echo -e "\n[Step 2] Installing Python packages..."
python3.11 -m pip install --user -q google-cloud-storage numpy 2>/dev/null || true
python3.11 -m pip install --user --upgrade gemma 2>&1 | tail -3
echo "  Packages installed"

# Step 3: 실행
echo -e "\n[Step 3] Running precompute..."
cd ~/ouroboros
TPU_WORKER_ID=0 python3.11 -u precompute_imagenet_embeddings.py

echo -e "\n=============================================="
echo "Done!"
echo "=============================================="
echo "Output: gs://rdy-tpu-data-2025/imagenet-1k/imagenet_class_embeddings.npy"
