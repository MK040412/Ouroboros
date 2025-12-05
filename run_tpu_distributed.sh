#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# TPU v5e-32 (vtlitepod-32) 분산 학습 스크립트
# =============================================================================
# TPU v5e-32 구성:
#   - 8개 worker (호스트), 각 worker에 4개 TPU 칩
#   - 총 32개 TPU 코어
#   - worker 0이 coordinator 역할
#
# 사용법:
#   로컬에서 실행: ./run_tpu_distributed.sh
#   TPU worker에서 직접 실행: TPU_WORKER_ID=0 ./run_tpu_distributed.sh --local
# =============================================================================

ZONE="europe-west4-b"
INSTANCE="ouroboros"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PREFIX="[tpu-dist]"

# TPU Pod 설정
NUM_WORKERS=8
CHIPS_PER_WORKER=4
COORDINATOR_PORT=8476

log() {
  echo "${LOG_PREFIX} $(date '+%H:%M:%S') $*"
}

# =============================================================================
# 로컬 모드 (TPU worker에서 직접 실행)
# =============================================================================
run_local() {
  log "Running in LOCAL mode on TPU worker"

  # TPU_WORKER_ID는 gcloud ssh로 전달됨
  WORKER_ID="${TPU_WORKER_ID:-0}"

  # Worker 0의 내부 IP 가져오기 (coordinator)
  # TPU Pod 내부에서는 worker-0 호스트네임 또는 환경변수 사용
  if [[ -n "${TPU_WORKER_HOSTNAMES:-}" ]]; then
    # TPU Pod 환경에서 제공하는 호스트네임 리스트
    COORDINATOR_IP=$(echo "$TPU_WORKER_HOSTNAMES" | cut -d',' -f1)
  else
    # 수동 설정: worker-0 IP (TPU 내부 네트워크)
    COORDINATOR_IP="${COORDINATOR_IP:-10.164.0.2}"
  fi

  log "Worker ID: $WORKER_ID / $(($NUM_WORKERS - 1))"
  log "Coordinator: $COORDINATOR_IP:$COORDINATOR_PORT"

  # JAX 분산 환경변수 설정
  export JAX_COORDINATOR_ADDRESS="${COORDINATOR_IP}:${COORDINATOR_PORT}"
  export JAX_COORDINATOR_PORT="${COORDINATOR_PORT}"
  export JAX_NUM_PROCESSES="${NUM_WORKERS}"
  export JAX_PROCESS_INDEX="${WORKER_ID}"
  export JAX_LOCAL_DEVICE_COUNT="${CHIPS_PER_WORKER}"

  # TPU 관련 설정
  export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
  export TPU_HOST_BOUNDS="2,2,1"
  export TPU_VISIBLE_DEVICES="0,1,2,3"

  # XLA 최적화
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
  export TF_CPP_MIN_LOG_LEVEL=0

  log "Environment configured:"
  log "  JAX_COORDINATOR_ADDRESS=$JAX_COORDINATOR_ADDRESS"
  log "  JAX_NUM_PROCESSES=$JAX_NUM_PROCESSES"
  log "  JAX_PROCESS_INDEX=$JAX_PROCESS_INDEX"

  # 디렉토리 이동 및 실행
  cd "$SCRIPT_DIR"

  # venv 활성화 (있으면)
  if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    log "venv activated"
  fi

  log "Starting training..."
  # Python 출력 버퍼링 비활성화
  export PYTHONUNBUFFERED=1
  python3.11 -u train_tpu_256.py 2>&1 | tee "/tmp/train_worker_${WORKER_ID}.log"
}

# =============================================================================
# 현재 TPU worker ID 감지 (TPU VM 내부에서 실행 시)
# =============================================================================
get_current_worker_id() {
  local hostname=$(hostname)
  # 호스트네임 형식: t1v-n-XXXXXXXX-w-N (N이 worker ID)
  if [[ "$hostname" =~ -w-([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "-1"  # TPU VM이 아닌 경우
  fi
}

# =============================================================================
# 원격 모드 (로컬 머신에서 TPU에 명령 전송)
# =============================================================================
run_remote() {
  log "Running in REMOTE mode - dispatching to TPU workers"

  # 현재 TPU worker ID 감지 (TPU VM 내부에서 실행 시)
  CURRENT_WORKER_ID=$(get_current_worker_id)
  if [[ "$CURRENT_WORKER_ID" != "-1" ]]; then
    log "Detected running on TPU worker $CURRENT_WORKER_ID"
  fi

  # 1. 먼저 코드 동기화
  log "Step 1: Syncing code to TPU workers..."

  COMMIT_MSG="chore: sync $(date -u +'%Y-%m-%d %H:%M:%S UTC')"

  # Git push (변경사항 있으면)
  git -C "$SCRIPT_DIR" add -A
  if ! git -C "$SCRIPT_DIR" diff --cached --quiet; then
    git -C "$SCRIPT_DIR" commit -m "$COMMIT_MSG"
  fi
  git -C "$SCRIPT_DIR" push origin main 2>/dev/null || true

  REMOTE_URL="$(git -C "$SCRIPT_DIR" remote get-url origin)"
  BRANCH="$(git -C "$SCRIPT_DIR" rev-parse --abbrev-ref HEAD)"

  # 2. Worker 0의 내부 IP 가져오기
  log "Step 2: Getting coordinator IP..."
  COORDINATOR_IP=$(gcloud compute tpus tpu-vm describe "$INSTANCE" \
    --zone="$ZONE" \
    --format="value(networkEndpoints[0].ipAddress)" 2>/dev/null || echo "")

  if [[ -z "$COORDINATOR_IP" ]]; then
    log "Warning: Could not get coordinator IP automatically"
    log "Using default internal IP pattern"
    COORDINATOR_IP="10.164.0.2"
  fi
  log "Coordinator IP: $COORDINATOR_IP"

  # 3. 모든 worker 정리 및 코드 동기화 (병렬)
  log "Step 3: Cleaning up and syncing code on all workers..."
  for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
    gcloud compute tpus tpu-vm ssh "$INSTANCE" \
      --zone="$ZONE" \
      --worker="$WORKER_ID" \
      --command="pkill -9 -f 'python.*train_tpu' 2>/dev/null || true; sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true" &
  done
  wait
  log "  Cleanup done"

  # 4. 코드 동기화 (병렬)
  log "Step 4: Syncing code on all workers..."
  for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
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
python3.11 -m pip install --user -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 2>/dev/null || true
python3.11 -m pip install --user -q flax optax chex Pillow PyYAML wandb pyarrow torch transformers google-cloud-storage 2>/dev/null || true
EOF
    gcloud compute tpus tpu-vm ssh "$INSTANCE" \
      --zone="$ZONE" \
      --worker="$WORKER_ID" \
      --command="bash -c '$SYNC_CMD'" &
  done
  wait
  log "  Code synced on all workers"

  # 5. Worker 0 (coordinator) 먼저 시작
  log "Step 5: Starting Worker 0 (coordinator) first..."

  read -r -d '' WORKER0_CMD <<EOF || true
set -e
export TPU_WORKER_ID=0
export COORDINATOR_IP="$COORDINATOR_IP"
export JAX_COORDINATOR_ADDRESS="${COORDINATOR_IP}:${COORDINATOR_PORT}"
export JAX_COORDINATOR_PORT="${COORDINATOR_PORT}"
export JAX_NUM_PROCESSES=${NUM_WORKERS}
export JAX_PROCESS_INDEX=0
export JAX_LOCAL_DEVICE_COUNT=${CHIPS_PER_WORKER}
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export TPU_HOST_BOUNDS="2,2,1"
export PYTHONPATH=~/ouroboros/src:\$PYTHONPATH
export PYTHONUNBUFFERED=1

cd ~/ouroboros
echo "[Worker 0] Starting coordinator..."
echo "[Worker 0] JAX_COORDINATOR_ADDRESS=\$JAX_COORDINATOR_ADDRESS"

# nohup으로 백그라운드 실행
nohup python3.11 -u train_tpu_256.py > /tmp/train_worker_0.log 2>&1 &
echo "[Worker 0] Coordinator started with PID \$!"
EOF

  gcloud compute tpus tpu-vm ssh "$INSTANCE" \
    --zone="$ZONE" \
    --worker="0" \
    --command="bash -c '$WORKER0_CMD'"

  # Coordinator가 listening 상태가 될 때까지 대기
  log "  Waiting 30s for coordinator to start listening..."
  sleep 30

  # 6. Worker 1-7 시작 (병렬)
  log "Step 6: Starting workers 1-7..."
  for WORKER_ID in $(seq 1 $((NUM_WORKERS - 1))); do
    log "  Launching worker $WORKER_ID..."

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
export PYTHONPATH=~/ouroboros/src:\$PYTHONPATH
export PYTHONUNBUFFERED=1

cd ~/ouroboros
echo "[Worker $WORKER_ID] Starting training..."
echo "[Worker $WORKER_ID] JAX_COORDINATOR_ADDRESS=\$JAX_COORDINATOR_ADDRESS"

# nohup으로 백그라운드 실행
nohup python3.11 -u train_tpu_256.py > /tmp/train_worker_${WORKER_ID}.log 2>&1 &
echo "[Worker $WORKER_ID] Started with PID \$!"
EOF

    gcloud compute tpus tpu-vm ssh "$INSTANCE" \
      --zone="$ZONE" \
      --worker="$WORKER_ID" \
      --command="bash -c '$WORKER_CMD'" &
  done
  wait

  log "All workers launched!"
  log ""
  log "Monitor logs:"
  log "  ./gcp_show_log.sh --worker=0 -f"
  log "  ./gcp_show_log.sh --worker=1 -f"
  log ""
  log "Kill all: ./gcp_killall.sh"
}

# =============================================================================
# 메인
# =============================================================================
# TPU VM 내부에서 실행 중인지 감지 (t1v- 또는 tpu- 호스트네임)
IS_ON_TPU_VM=false
if [[ "$(hostname)" == t1v-* ]] || [[ "$(hostname)" == tpu-* ]]; then
  IS_ON_TPU_VM=true
fi

if [[ "${1:-}" == "--local" ]]; then
  run_local
elif [[ "$IS_ON_TPU_VM" == true ]]; then
  log "WARNING: Detected running on TPU VM ($(hostname))"
  log "You should run this script from your LOCAL machine, not from TPU VM."
  log "Or use: ./run_tpu_distributed.sh --local"
  log ""
  log "If you want to run remote mode from TPU VM, use: ./run_tpu_distributed.sh --force-remote"
  if [[ "${1:-}" == "--force-remote" ]]; then
    run_remote
  else
    exit 1
  fi
else
  run_remote
fi
