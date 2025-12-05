#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# TPU v5e-16 분산 학습 스크립트 (Git 동기화 없음)
# =============================================================================
# Git sync 없이 현재 코드 그대로 실행 (수동으로 scp 등으로 동기화 후 사용)
#
# 사용법:
#   ./run_tpu_distributed_no_git.sh              # TPU VM에서 실행
#   ./run_tpu_distributed_no_git.sh --force      # 강제 실행
# =============================================================================

ZONE="europe-west4-b"
INSTANCE="ouroboros"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PREFIX="[tpu-no-git]"

# TPU Pod 설정
NUM_WORKERS=4
CHIPS_PER_WORKER=4
COORDINATOR_PORT=8476

log() {
  echo "${LOG_PREFIX} $(date '+%H:%M:%S') $*"
}

# =============================================================================
# 현재 TPU worker ID 감지
# =============================================================================
get_current_worker_id() {
  local hostname=$(hostname)
  if [[ "$hostname" =~ -w-([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "-1"
  fi
}

# =============================================================================
# 메인 실행
# =============================================================================
run_training() {
  log "Starting training WITHOUT git sync"
  log "Make sure all workers have the same code (use scp to sync manually)"

  CURRENT_WORKER_ID=$(get_current_worker_id)
  if [[ "$CURRENT_WORKER_ID" != "-1" ]]; then
    log "Running on TPU worker $CURRENT_WORKER_ID"
  fi

  # Coordinator IP 가져오기
  log "Step 1: Getting coordinator IP..."
  COORDINATOR_IP=$(gcloud compute tpus tpu-vm describe "$INSTANCE" \
    --zone="$ZONE" \
    --format="value(networkEndpoints[0].ipAddress)" 2>/dev/null || echo "")

  if [[ -z "$COORDINATOR_IP" ]]; then
    COORDINATOR_IP="10.164.0.2"
    log "Using default IP: $COORDINATOR_IP"
  else
    log "Coordinator IP: $COORDINATOR_IP"
  fi

  # 이전 프로세스 정리
  log "Step 2: Killing previous training processes..."
  gcloud compute tpus tpu-vm ssh "$INSTANCE" \
    --zone="$ZONE" \
    --worker=all \
    --command="pkill -9 -f 'python.*train_tpu_256' 2>/dev/null || true; sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true" \
    2>/dev/null || true

  sleep 2

  # 각 worker에서 실행
  log "Step 3: Launching training on all workers..."
  log "  Order: Workers 1,2,3 first, then Worker 0 (coordinator)"

  for WORKER_ID in 1 2 3 0; do
    log "Launching worker $WORKER_ID..."

    read -r -d '' WORKER_CMD <<EOF || true
set -e
export TPU_WORKER_ID=$WORKER_ID
export COORDINATOR_IP="$COORDINATOR_IP"
export JAX_COORDINATOR_ADDRESS="${COORDINATOR_IP}:${COORDINATOR_PORT}"
export JAX_COORDINATOR_PORT="${COORDINATOR_PORT}"
export JAX_NUM_PROCESSES=${NUM_WORKERS}
export JAX_PROCESS_INDEX=${WORKER_ID}
export JAX_LOCAL_DEVICE_COUNT=${CHIPS_PER_WORKER}

export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export TPU_HOST_BOUNDS="2,2,1"

export HF_TOKEN="\${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="\${HF_TOKEN:-}"

cd ~/ouroboros
export PYTHONPATH="\$PWD/src:\$PYTHONPATH"

# TPU lockfile 정리
sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true

echo "[Worker $WORKER_ID] Starting training (no git sync)..."
echo "[Worker $WORKER_ID] JAX_COORDINATOR_ADDRESS=\$JAX_COORDINATOR_ADDRESS"
echo "[Worker $WORKER_ID] JAX_PROCESS_INDEX=\$JAX_PROCESS_INDEX"

export PYTHONUNBUFFERED=1
nohup python3.11 -u train_tpu_256.py > /tmp/train_worker_${WORKER_ID}.log 2>&1 &
TRAIN_PID=\$!
echo "[Worker $WORKER_ID] Training started with PID \$TRAIN_PID"
wait \$TRAIN_PID
echo "[Worker $WORKER_ID] Training finished"
EOF

    if [[ "$WORKER_ID" == "$CURRENT_WORKER_ID" ]]; then
      log "Worker $WORKER_ID: Running locally"
      bash -lc "${WORKER_CMD}" &
    else
      log "Worker $WORKER_ID: Running via SSH"
      gcloud compute tpus tpu-vm ssh "$INSTANCE" \
        --zone="$ZONE" \
        --worker="$WORKER_ID" \
        --command="bash -lc '${WORKER_CMD}'" &
    fi

    # Coordinator 전에 대기 (짧게 - pip install 없으므로)
    if [[ $WORKER_ID -eq 3 ]]; then
      log "Waiting 10s for workers 1,2,3 before starting coordinator..."
      sleep 10
    fi
  done

  log "All workers launched. Waiting for completion..."
  wait
  log "Training completed"
}

# =============================================================================
# 메인
# =============================================================================
IS_ON_TPU_VM=false
if [[ "$(hostname)" == t1v-* ]] || [[ "$(hostname)" == tpu-* ]]; then
  IS_ON_TPU_VM=true
fi

if [[ "$IS_ON_TPU_VM" == true ]] && [[ "${1:-}" != "--force" ]]; then
  log "Running on TPU VM. Use --force to confirm."
  log "Usage: ./run_tpu_distributed_no_git.sh --force"
  exit 1
fi

run_training
