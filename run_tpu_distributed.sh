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
# 원격 모드 (로컬 머신에서 TPU에 명령 전송)
# =============================================================================
run_remote() {
  log "Running in REMOTE mode - dispatching to TPU workers"

  # 1. 먼저 코드 동기화
  log "Step 1: Syncing code to git..."

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

  # 3. 모든 worker 정리 (--worker=all 사용)
  log "Step 3: Cleaning up all workers..."
  gcloud compute tpus tpu-vm ssh "$INSTANCE" \
    --zone="$ZONE" \
    --worker="all" \
    --command="sudo pkill -9 python 2>/dev/null || true; sudo fuser -k /dev/vfio/0 2>/dev/null || true; sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true; echo 'Worker \$(hostname) cleaned'"
  log "  Cleanup done"

  # 4. 코드 동기화 (--worker=all 사용)
  log "Step 4: Syncing code on all workers..."
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

# Python 패키지 설치 (조용히)
python3.11 -m pip install --user -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 2>/dev/null || true
python3.11 -m pip install --user -q flax optax chex Pillow PyYAML wandb pyarrow torch transformers google-cloud-storage 2>/dev/null || true
echo "Worker \$(hostname) synced"
EOF
  gcloud compute tpus tpu-vm ssh "$INSTANCE" \
    --zone="$ZONE" \
    --worker="all" \
    --command="bash -c '$SYNC_CMD'"
  log "  Code synced on all workers"

  # 5. 모든 worker에서 학습 시작 (--worker=all 사용)
  # 각 worker가 자신의 hostname에서 worker ID를 추출하여 환경변수 설정
  log "Step 5: Starting training on all workers..."
  read -r -d '' TRAIN_CMD <<EOF || true
set -e

# hostname에서 worker ID 추출 (형식: t1v-n-XXXXXXXX-w-N)
HOSTNAME=\$(hostname)
if [[ "\$HOSTNAME" =~ -w-([0-9]+)\$ ]]; then
  WORKER_ID="\${BASH_REMATCH[1]}"
else
  echo "Error: Cannot extract worker ID from hostname: \$HOSTNAME"
  exit 1
fi

export TPU_WORKER_ID=\$WORKER_ID
export COORDINATOR_IP="$COORDINATOR_IP"
export JAX_COORDINATOR_ADDRESS="${COORDINATOR_IP}:${COORDINATOR_PORT}"
export JAX_COORDINATOR_PORT="${COORDINATOR_PORT}"
export JAX_NUM_PROCESSES=${NUM_WORKERS}
export JAX_PROCESS_INDEX=\$WORKER_ID
export JAX_LOCAL_DEVICE_COUNT=${CHIPS_PER_WORKER}

# TPU v5e-32 topology: 8 hosts × 4 chips = 32 chips
# TPU_CHIPS_PER_HOST_BOUNDS: 각 host의 chip 배열 (2×2×1 = 4 chips)
# TPU_HOST_BOUNDS: host 배열 (4×2×1 = 8 hosts)
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export TPU_HOST_BOUNDS="4,2,1"

export PYTHONPATH=~/ouroboros/src:\$PYTHONPATH
export PYTHONUNBUFFERED=1

cd ~/ouroboros
echo "[Worker \$WORKER_ID] Starting training..."
echo "[Worker \$WORKER_ID] JAX_COORDINATOR_ADDRESS=\$JAX_COORDINATOR_ADDRESS"
echo "[Worker \$WORKER_ID] JAX_PROCESS_INDEX=\$JAX_PROCESS_INDEX"

# nohup으로 백그라운드 실행
nohup python3.11 -u train_tpu_256.py > /tmp/train_worker_\${WORKER_ID}.log 2>&1 &
echo "[Worker \$WORKER_ID] Started with PID \$!"
EOF

  gcloud compute tpus tpu-vm ssh "$INSTANCE" \
    --zone="$ZONE" \
    --worker="all" \
    --command="bash -c '$TRAIN_CMD'"

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
run_remote
