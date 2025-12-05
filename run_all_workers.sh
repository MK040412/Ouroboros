#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# TPU VM 내부에서 모든 Worker 실행 (Worker 0에서 실행)
# =============================================================================
# TPU v5e-32 구성:
#   - 8개 worker (호스트), 각 worker에 4개 TPU 칩
#   - 총 32개 TPU 코어
#   - Worker 0이 coordinator 역할 (먼저 시작해야 함)
# =============================================================================

ZONE="europe-west4-b"
INSTANCE="ouroboros"
NUM_WORKERS=8
CHIPS_PER_WORKER=4
COORDINATOR_PORT=8476

# Coordinator IP (Worker 0 내부 IP)
COORDINATOR_IP=$(hostname -I | awk '{print $1}')
echo "[main] Coordinator IP: $COORDINATOR_IP"
echo "[main] NUM_WORKERS: $NUM_WORKERS"

# 먼저 모든 Worker에서 이전 프로세스 정리
echo "[main] Cleaning up previous processes on all workers..."
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

# Worker 0 (coordinator) 먼저 시작
echo "[main] Starting Worker 0 (coordinator) first..."
export TPU_WORKER_ID=0
export COORDINATOR_IP="$COORDINATOR_IP"
export JAX_COORDINATOR_ADDRESS="${COORDINATOR_IP}:${COORDINATOR_PORT}"
export JAX_COORDINATOR_PORT="${COORDINATOR_PORT}"
export JAX_NUM_PROCESSES="${NUM_WORKERS}"
export JAX_PROCESS_INDEX=0
# TPU topology는 JAX가 자동 감지하도록 함
export PYTHONPATH=~/ouroboros/src:$PYTHONPATH
export PYTHONUNBUFFERED=1

cd ~/ouroboros
nohup python3.11 -u train_tpu_256.py > /tmp/train_worker_0.log 2>&1 &
WORKER0_PID=$!
echo "[main] Worker 0 (coordinator) started with PID: $WORKER0_PID"

# Coordinator가 listening 상태가 될 때까지 대기
echo "[main] Waiting 30s for coordinator to start listening..."
sleep 30

# 나머지 Worker들 시작 (1-7)
echo "[main] Starting workers 1-7..."
for WORKER_ID in $(seq 1 $((NUM_WORKERS - 1))); do
  echo "[main] Starting Worker $WORKER_ID..."
  gcloud compute tpus tpu-vm ssh "$INSTANCE" \
    --zone="$ZONE" \
    --worker="$WORKER_ID" \
    --command="cd ~/ouroboros && \
      export TPU_WORKER_ID=$WORKER_ID && \
      export COORDINATOR_IP=$COORDINATOR_IP && \
      export JAX_COORDINATOR_ADDRESS=${COORDINATOR_IP}:${COORDINATOR_PORT} && \
      export JAX_COORDINATOR_PORT=${COORDINATOR_PORT} && \
      export JAX_NUM_PROCESSES=${NUM_WORKERS} && \
      export JAX_PROCESS_INDEX=${WORKER_ID} && \
      export PYTHONPATH=~/ouroboros/src:\$PYTHONPATH && \
      export PYTHONUNBUFFERED=1 && \
      nohup python3.11 -u train_tpu_256.py > /tmp/train_worker_$WORKER_ID.log 2>&1 &" &
done
wait

echo "[main] All $NUM_WORKERS workers started!"
echo ""
echo "[main] Monitor logs:"
echo "  Worker 0: tail -f /tmp/train_worker_0.log"
echo "  Worker N: gcloud compute tpus tpu-vm ssh $INSTANCE --zone=$ZONE --worker=N --command='tail -f /tmp/train_worker_N.log'"
echo ""
echo "[main] Kill all: pkill -9 -f train_tpu_256.py"
echo ""

# Worker 0 로그 따라가기
tail -f /tmp/train_worker_0.log
