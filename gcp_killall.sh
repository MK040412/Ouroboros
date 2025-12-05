#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# TPU v5e-32 전체 Worker Python 프로세스 종료
# =============================================================================
# 8개 worker (hosts), 각 worker에 4개 TPU 칩
# 총 32 TPU 코어
#
# 사용법:
#   ./gcp_killall.sh              # 모든 worker의 python 프로세스 종료
#   ./gcp_killall.sh --pattern "train_tpu"  # 특정 패턴만 종료
# =============================================================================

ZONE="europe-west4-b"
INSTANCE="ouroboros"
NUM_WORKERS=8
PATTERN="python"
LOG_PREFIX="[killall]"

log() {
  echo "${LOG_PREFIX} $(date '+%H:%M:%S') $*"
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pattern)
      PATTERN="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $(basename "$0") [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --pattern <name>  Kill processes matching pattern (default: python)"
      echo "  --help, -h        Show this help"
      exit 0
      ;;
    *)
      shift
      ;;
  esac
done

log "Killing '$PATTERN' processes on all $NUM_WORKERS workers..."

# 모든 Worker에서 프로세스 종료
for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
  log "Worker $WORKER_ID: killing '$PATTERN' processes..."
  gcloud compute tpus tpu-vm ssh "$INSTANCE" \
    --zone="$ZONE" \
    --worker="$WORKER_ID" \
    --command="pkill -9 -f '$PATTERN' 2>/dev/null && echo '  Killed' || echo '  No processes found'; sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true" &
done

# 모든 SSH 명령 완료 대기
wait

log "Done! All '$PATTERN' processes terminated on $NUM_WORKERS workers."
