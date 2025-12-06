#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 로컬 코드를 GitHub에 푸시하고 모든 TPU 워커에서 pull
# =============================================================================

ZONE="europe-west4-b"
INSTANCE="ouroboros"
NUM_WORKERS=8
LOG_PREFIX="[gcp_sync]"

log() {
  echo "${LOG_PREFIX} $*"
}

# 1. 로컬에서 GitHub에 푸시
log "Step 1: Pushing local changes to GitHub..."

COMMIT_MSG="chore: sync $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
git add -A
if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "$COMMIT_MSG" || true
    git push origin main || true
    log "  Changes pushed"
else
    log "  No changes to commit"
fi

# 2. 모든 워커에서 pull
log "Step 2: Pulling on all $NUM_WORKERS workers..."

for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
    log "  Worker $WORKER_ID: pulling..."
    gcloud compute tpus tpu-vm ssh "$INSTANCE" \
        --zone="$ZONE" \
        --worker="$WORKER_ID" \
        --command="cd ~/ouroboros && git pull origin main 2>/dev/null || true" &
done

wait
log "All workers synced!"
