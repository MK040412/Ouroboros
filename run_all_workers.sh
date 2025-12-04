#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# TPU VM 내부에서 모든 Worker 실행 (Worker 0에서 실행)
# =============================================================================

ZONE="europe-west4-b"
INSTANCE="ouroboros"
NUM_WORKERS=4

# Coordinator IP (Worker 0 내부 IP)
COORDINATOR_IP=$(hostname -I | awk '{print $1}')
echo "[main] Coordinator IP: $COORDINATOR_IP"

# 먼저 모든 Worker에서 이전 프로세스 정리
echo "[main] Cleaning up previous processes..."
for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
  if [[ $WORKER_ID -eq 0 ]]; then
    pkill -9 -f train_tpu_256.py 2>/dev/null || true
    sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true
  else
    gcloud compute tpus tpu-vm ssh "$INSTANCE" \
      --zone="$ZONE" \
      --worker="$WORKER_ID" \
      --command="pkill -9 -f train_tpu_256.py 2>/dev/null; sudo rm -f /tmp/libtpu_lockfile 2>/dev/null; echo 'Worker $WORKER_ID cleaned'" &
  fi
done
wait
echo "[main] Cleanup done"

# Git pull (모든 Worker)
echo "[main] Pulling latest code on all workers..."
cd ~/ouroboros && git pull origin main
for WORKER_ID in $(seq 1 $((NUM_WORKERS - 1))); do
  gcloud compute tpus tpu-vm ssh "$INSTANCE" \
    --zone="$ZONE" \
    --worker="$WORKER_ID" \
    --command="cd ~/ouroboros && git pull origin main" &
done
wait
echo "[main] Git pull done"

# Worker 0 먼저 시작 (coordinator)
echo "[main] Starting Worker 0 (coordinator)..."
export TPU_WORKER_ID=0
export COORDINATOR_IP="$COORDINATOR_IP"
cd ~/ouroboros
nohup bash -c "source ~/.bashrc && TPU_WORKER_ID=0 COORDINATOR_IP=$COORDINATOR_IP ./run_tpu_distributed.sh --local" > /tmp/train_worker_0.log 2>&1 &
WORKER0_PID=$!
echo "[main] Worker 0 started (PID: $WORKER0_PID)"

# Coordinator가 리스닝 시작할 때까지 대기
sleep 5

# 나머지 Worker들 시작
for WORKER_ID in $(seq 1 $((NUM_WORKERS - 1))); do
  echo "[main] Starting Worker $WORKER_ID..."
  gcloud compute tpus tpu-vm ssh "$INSTANCE" \
    --zone="$ZONE" \
    --worker="$WORKER_ID" \
    --command="cd ~/ouroboros && source ~/.bashrc && nohup bash -c 'TPU_WORKER_ID=$WORKER_ID COORDINATOR_IP=$COORDINATOR_IP ./run_tpu_distributed.sh --local' > /tmp/train_worker_$WORKER_ID.log 2>&1 &" &
done
wait

echo "[main] All workers started!"
echo "[main] Monitor with: tail -f /tmp/train_worker_0.log"
echo "[main] Or check other workers:"
echo "  gcloud compute tpus tpu-vm ssh $INSTANCE --zone=$ZONE --worker=1 --command='tail -f /tmp/train_worker_1.log'"

# Worker 0 로그 따라가기
tail -f /tmp/train_worker_0.log
