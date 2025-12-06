#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ImageNet 클래스 임베딩 사전 계산
# 단일 워커에서 실행 (1000개 클래스만 처리하므로 분산 불필요)
# =============================================================================

ZONE="europe-west4-b"
INSTANCE="ouroboros"

echo "=============================================="
echo "ImageNet Class Embeddings Pre-compute"
echo "=============================================="

# Worker 0에서 실행 중인지 확인
if [[ ! -f ~/.bashrc ]]; then
    echo "Error: This script should be run from a TPU VM"
    exit 1
fi

# 현재 Worker ID 확인
CURRENT_WORKER=${TPU_WORKER_ID:-0}
echo "Current worker: $CURRENT_WORKER"

# 로컬 프로세스만 정리
pkill -9 -f precompute_imagenet_embeddings 2>/dev/null || true

# 모든 변경사항 커밋 및 푸시
echo -e "\n[Step 1] Syncing code changes..."
cd ~/ouroboros
COMMIT_MSG="chore: sync $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
git add -A
if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "$COMMIT_MSG" || true
    git push origin main || true
    echo "  Changes pushed"
else
    echo "  No changes to commit"
fi

# 코드 동기화
echo -e "\n[Step 2] Pulling latest code..."
cd ~/ouroboros && git pull origin main 2>/dev/null || true
echo "  Code synced"

# Python 패키지 설치
echo -e "\n[Step 3] Installing Python packages..."
python3.11 -m pip install --user -q google-cloud-storage numpy 2>&1 | tail -3
python3.11 -m pip install --user --upgrade gemma 2>&1 | tail -3
echo "  Packages installed"

# 실행
echo -e "\n[Step 4] Running precompute..."
cd ~/ouroboros
python3.11 -u precompute_imagenet_embeddings.py 2>&1 | tee /tmp/precompute_imagenet.log

echo -e "\n=============================================="
echo "Done!"
echo "=============================================="
echo ""
echo "Output saved to: gs://rdy-tpu-data-2025/imagenet-1k/imagenet_class_embeddings.npy"
echo "Log file: /tmp/precompute_imagenet.log"
