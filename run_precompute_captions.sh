#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ImageNet Caption Embedding 전처리 스크립트
# =============================================================================
# visual-layer/imagenet-1k-vl-enriched 데이터셋에서 BLIP2 caption을 추출하고
# Gemma-3 임베딩을 계산하여 GCS에 업로드
#
# 사용법:
#   ./run_precompute_captions.sh              # 계산만 (로컬 저장)
#   ./run_precompute_captions.sh --upload     # 계산 + GCS 업로드
#   ./run_precompute_captions.sh --batch-size 128 --upload
# =============================================================================

# 커맨드라인 인자 파싱
UPLOAD_FLAG=""
BATCH_SIZE="64"

while [[ $# -gt 0 ]]; do
  case $1 in
    --upload)
      UPLOAD_FLAG="--upload"
      shift
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--upload] [--batch-size N]"
      exit 1
      ;;
  esac
done

ZONE="europe-west4-b"
INSTANCE="ouroboros"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PREFIX="[caption-precompute]"

log() {
  echo "${LOG_PREFIX} $(date '+%H:%M:%S') $*"
}

# =============================================================================
# 메인 실행
# =============================================================================
run_precompute() {
  log "Starting ImageNet Caption Embedding Precomputation"
  log "  Batch size: $BATCH_SIZE"
  log "  Upload: ${UPLOAD_FLAG:-disabled}"

  # 1. 코드 동기화
  log "Step 1: Syncing code to git..."

  COMMIT_MSG="chore: sync $(date -u +'%Y-%m-%d %H:%M:%S UTC')"

  git -C "$SCRIPT_DIR" add -A
  if ! git -C "$SCRIPT_DIR" diff --cached --quiet; then
    git -C "$SCRIPT_DIR" commit -m "$COMMIT_MSG"
  fi
  git -C "$SCRIPT_DIR" push origin main 2>/dev/null || true

  REMOTE_URL="$(git -C "$SCRIPT_DIR" remote get-url origin)"
  BRANCH="$(git -C "$SCRIPT_DIR" rev-parse --abbrev-ref HEAD)"

  # 2. TPU worker 0에서 이전 프로세스 정리 및 코드 동기화
  log "Step 2: Cleaning up and syncing code on TPU worker 0..."
  read -r -d '' SYNC_CMD <<EOF || true
set -e

# 이전 프로세스 정리
pkill -f precompute_imagenet_captions.py 2>/dev/null || true
echo "Previous process killed (if any)"

# HuggingFace 캐시 정리 (디스크 절약)
rm -rf ~/.cache/huggingface/datasets/visual-layer* 2>/dev/null || true
echo "HuggingFace cache cleared"

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

# Python 패키지 설치 (HuggingFace datasets, pyarrow, sentence-transformers)
python3.11 -m pip install --user -q datasets pyarrow sentence-transformers 2>/dev/null || true
echo "Code synced and packages installed"
EOF

  gcloud compute tpus tpu-vm ssh "$INSTANCE" \
    --zone="$ZONE" \
    --worker=0 \
    --command="bash -c '$SYNC_CMD'"
  log "  Code synced"

  # 3. Caption embedding 계산 실행
  log "Step 3: Starting caption embedding computation..."
  read -r -d '' RUN_CMD <<EOF || true
set -e
cd ~/ouroboros
export PYTHONPATH=~/ouroboros/src:\$PYTHONPATH
export PYTHONUNBUFFERED=1

echo "Starting precompute_imagenet_captions.py..."
echo "  Batch size: $BATCH_SIZE"
echo "  Upload: ${UPLOAD_FLAG:-disabled}"

# nohup으로 백그라운드 실행
nohup python3.11 -u precompute_imagenet_captions.py \
  --batch-size $BATCH_SIZE \
  $UPLOAD_FLAG \
  > /tmp/precompute_captions.log 2>&1 &

echo "Started with PID \$!"
echo "Log: /tmp/precompute_captions.log"
EOF

  gcloud compute tpus tpu-vm ssh "$INSTANCE" \
    --zone="$ZONE" \
    --worker=0 \
    --command="bash -c '$RUN_CMD'"

  log ""
  log "Caption embedding precomputation started!"
  log ""
  log "Monitor progress:"
  log "  ./gcp_run.sh 'tail -50 /tmp/precompute_captions.log'"
  log "  ./gcp_run.sh 'tail -f /tmp/precompute_captions.log'"
  log ""
  log "Output files (on TPU):"
  log "  ~/ouroboros/data/imagenet_captions/imagenet_caption_embeddings.npy"
  log "  ~/ouroboros/data/imagenet_captions/imagenet_captions.json"
  if [[ -n "$UPLOAD_FLAG" ]]; then
    log ""
    log "GCS upload destination:"
    log "  gs://rdy-tpu-data-2025/imagenet-1k/imagenet_caption_embeddings.npy"
    log "  gs://rdy-tpu-data-2025/imagenet-1k/imagenet_captions.json"
  fi
}

run_precompute
