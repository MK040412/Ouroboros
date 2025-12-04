#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 4개 Worker에서 Pre-compute 분산 실행
# =============================================================================

ZONE="europe-west4-b"
INSTANCE="ouroboros"
NUM_WORKERS=4

echo "=============================================="
echo "Distributed Pre-compute Embeddings"
echo "=============================================="

# Worker 0에서 실행 중인지 확인
if [[ ! -f ~/.bashrc ]]; then
    echo "Error: This script should be run from a TPU VM"
    exit 1
fi

# 현재 Worker ID 확인
CURRENT_WORKER=${TPU_WORKER_ID:-0}
echo "Current worker: $CURRENT_WORKER"

if [[ $CURRENT_WORKER -ne 0 ]]; then
    echo "Warning: This script is designed to run from Worker 0"
fi

# 로컬 프로세스만 정리
pkill -9 -f precompute_embeddings_tpu 2>/dev/null || true

# Precompute 관련 파일 커밋 및 푸시
echo -e "\n[Step 1] Committing precompute files..."
cd ~/ouroboros
git add precompute_embeddings*.py run_precompute_all.sh 2>/dev/null || true
if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "Update precompute embeddings scripts" || true
    git push origin main || true
    echo "  Changes pushed"
else
    echo "  No changes to commit"
fi

# 코드 동기화 (git pull on all workers)
echo -e "\n[Step 2] Syncing code on all workers..."
cd ~/ouroboros && git pull origin main 2>/dev/null || true
for WORKER_ID in $(seq 1 $((NUM_WORKERS - 1))); do
    gcloud compute tpus tpu-vm ssh "$INSTANCE" \
        --zone="$ZONE" \
        --worker="$WORKER_ID" \
        --command="cd ~/ouroboros && git pull origin main 2>/dev/null || true" &
done
wait
echo "  Code synced"

# Worker 0 시작
echo -e "\n[Step 3] Starting Worker 0..."
cd ~/ouroboros
nohup bash -c "TPU_WORKER_ID=0 python3 -u precompute_embeddings_tpu.py" \
    > /tmp/precompute_worker_0.log 2>&1 &
WORKER0_PID=$!
echo "  Worker 0 started (PID: $WORKER0_PID)"

# 잠시 대기 후 나머지 Worker들 시작
sleep 5

# 나머지 Worker들 시작
for WORKER_ID in $(seq 1 $((NUM_WORKERS - 1))); do
    echo "  Starting Worker $WORKER_ID..."
    gcloud compute tpus tpu-vm ssh "$INSTANCE" \
        --zone="$ZONE" \
        --worker="$WORKER_ID" \
        --command="cd ~/ouroboros && nohup bash -c 'TPU_WORKER_ID=$WORKER_ID python3 -u precompute_embeddings_tpu.py' > /tmp/precompute_worker_$WORKER_ID.log 2>&1 &" &
done
wait

echo -e "\n=============================================="
echo "All workers started!"
echo "=============================================="
echo ""
echo "Monitor progress:"
echo "  Worker 0: tail -f /tmp/precompute_worker_0.log"
echo "  Worker 1: gcloud compute tpus tpu-vm ssh $INSTANCE --zone=$ZONE --worker=1 --command='tail -f /tmp/precompute_worker_1.log'"
echo "  Worker 2: gcloud compute tpus tpu-vm ssh $INSTANCE --zone=$ZONE --worker=2 --command='tail -f /tmp/precompute_worker_2.log'"
echo "  Worker 3: gcloud compute tpus tpu-vm ssh $INSTANCE --zone=$ZONE --worker=3 --command='tail -f /tmp/precompute_worker_3.log'"
echo ""
echo "Expected time: ~5-8 hours (92 files / 4 workers × ~15 min/file)"

# Worker 0 로그 모니터링
tail -f /tmp/precompute_worker_0.log
