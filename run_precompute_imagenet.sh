#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ImageNet 클래스 임베딩 사전 계산 (TPU)
# 각 Worker가 독립적으로 로컬 TPU만 사용 (분산 동기화 없음)
# =============================================================================

ZONE="europe-west4-b"
INSTANCE="ouroboros"
NUM_WORKERS=8

echo "=============================================="
echo "ImageNet Class Embeddings Pre-compute"
echo "=============================================="

# 로컬에서 실행하는 경우
# Step 1: 코드 동기화
echo -e "\n[Step 1] Syncing code to git..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT_MSG="chore: sync $(date -u +'%Y-%m-%d %H:%M:%S UTC')"

git -C "$SCRIPT_DIR" add -A
if ! git -C "$SCRIPT_DIR" diff --cached --quiet 2>/dev/null; then
    git -C "$SCRIPT_DIR" commit -m "$COMMIT_MSG" || true
    git -C "$SCRIPT_DIR" push origin main || true
    echo "  Changes pushed"
else
    echo "  No changes to commit"
fi

REMOTE_URL="$(git -C "$SCRIPT_DIR" remote get-url origin)"
BRANCH="$(git -C "$SCRIPT_DIR" rev-parse --abbrev-ref HEAD)"

# Step 2: 모든 worker 정리
echo -e "\n[Step 2] Cleaning up all workers..."
gcloud compute tpus tpu-vm ssh "$INSTANCE" \
    --zone="$ZONE" \
    --worker="all" \
    --command="pkill -9 -f precompute_imagenet 2>/dev/null || true; echo 'Worker \$(hostname) cleaned'"
echo "  Cleanup done"

# Step 3: 코드 동기화 (모든 워커)
echo -e "\n[Step 3] Syncing code on all workers..."
read -r -d '' SYNC_CMD <<EOF || true
set -e
DEST_DIR=~/ouroboros
REMOTE_URL="$REMOTE_URL"

if [[ -d "\$DEST_DIR/.git" ]]; then
  cd "\$DEST_DIR"
  git fetch origin $BRANCH
  git checkout $BRANCH
  git pull --ff-only origin $BRANCH 2>/dev/null || git reset --hard origin/$BRANCH
else
  rm -rf "\$DEST_DIR"
  git clone "\$REMOTE_URL" "\$DEST_DIR"
  cd "\$DEST_DIR"
  git checkout $BRANCH
fi

# Python 패키지 설치
python3.11 -m pip install --user -q google-cloud-storage numpy 2>/dev/null || true
python3.11 -m pip install --user --upgrade gemma 2>/dev/null || true
echo "Worker \$(hostname) synced"
EOF

gcloud compute tpus tpu-vm ssh "$INSTANCE" \
    --zone="$ZONE" \
    --worker="all" \
    --command="bash -c '$SYNC_CMD'"
echo "  Code synced on all workers"

# Step 4: 모든 worker에서 precompute 시작
# 각 Worker가 독립적으로 로컬 TPU만 사용 (TPU_WORKER_ID로 구분)
echo -e "\n[Step 4] Starting precompute on all workers..."
read -r -d '' RUN_CMD <<EOF || true
set -e

# hostname에서 worker ID 추출 (형식: t1v-n-XXXXXXXX-w-N)
HOSTNAME=\$(hostname)
if [[ "\$HOSTNAME" =~ -w-([0-9]+)\$ ]]; then
  WORKER_ID="\${BASH_REMATCH[1]}"
else
  echo "Error: Cannot extract worker ID from hostname: \$HOSTNAME"
  exit 1
fi

# TPU_WORKER_ID 환경변수로 워커 구분 (JAX distributed 사용 안함)
export TPU_WORKER_ID=\$WORKER_ID
export PYTHONPATH=~/ouroboros/src:\$PYTHONPATH
export PYTHONUNBUFFERED=1

cd ~/ouroboros
echo "[Worker \$WORKER_ID] Starting precompute..."

# nohup으로 백그라운드 실행
nohup python3.11 -u precompute_imagenet_embeddings.py > /tmp/precompute_imagenet_worker_\${WORKER_ID}.log 2>&1 &
echo "[Worker \$WORKER_ID] Started with PID \$!"
EOF

gcloud compute tpus tpu-vm ssh "$INSTANCE" \
    --zone="$ZONE" \
    --worker="all" \
    --command="bash -c '$RUN_CMD'"

echo -e "\n=============================================="
echo "All workers started!"
echo "=============================================="
echo ""
echo "Monitor progress:"
echo "  ./gcp_run.sh --worker=0 'tail -f /tmp/precompute_imagenet_worker_0.log'"
echo "  ./gcp_run.sh --worker=1 'tail -f /tmp/precompute_imagenet_worker_1.log'"
echo ""
echo "Note: Only Worker 0 will save to GCS"
echo "Output: gs://rdy-tpu-data-2025/imagenet-1k/imagenet_class_embeddings.npy"
