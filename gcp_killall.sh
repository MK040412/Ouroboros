#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# TPU v5e-32 전체 Worker 정리 스크립트
# =============================================================================
# 8개 worker (hosts), 각 worker에 4개 TPU 칩
# 총 32 TPU 코어
#
# 사용법:
#   ./gcp_killall.sh              # 모든 python 프로세스 종료 + TPU/캐시 정리
#   ./gcp_killall.sh --pattern "train_tpu"  # 특정 패턴만 종료
#   ./gcp_killall.sh --full       # 전체 정리 (python + TPU lock + 캐시)
#   ./gcp_killall.sh --quick      # 빠른 정리 (python만)
# =============================================================================

ZONE="europe-west4-b"
INSTANCE="ouroboros"
PATTERN="python"
MODE="full"  # full, quick, pattern
LOG_PREFIX="[killall]"

log() {
  echo "${LOG_PREFIX} $(date '+%H:%M:%S') $*"
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pattern)
      PATTERN="$2"
      MODE="pattern"
      shift 2
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --quick)
      MODE="quick"
      shift
      ;;
    --help|-h)
      echo "Usage: $(basename "$0") [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --pattern <name>  Kill processes matching pattern only"
      echo "  --full            Full cleanup: python + TPU lock + GCS cache (default)"
      echo "  --quick           Quick cleanup: python processes only"
      echo "  --help, -h        Show this help"
      echo ""
      echo "Examples:"
      echo "  ./gcp_killall.sh                    # Full cleanup (recommended)"
      echo "  ./gcp_killall.sh --pattern train    # Kill only 'train' processes"
      echo "  ./gcp_killall.sh --quick            # Quick python kill only"
      exit 0
      ;;
    *)
      shift
      ;;
  esac
done

# 명령어 구성
case "$MODE" in
  "full")
    log "Full cleanup: python processes + TPU release + GCS cache..."
    KILL_CMD="sudo pkill -9 python 2>/dev/null || true; \
sudo fuser -k /dev/vfio/0 2>/dev/null || true; \
sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true; \
sudo rm -rf /tmp/gcs_cache_worker_* 2>/dev/null || true; \
sudo rm -rf /tmp/precompute_* 2>/dev/null || true; \
find ~/ouroboros -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true; \
echo 'TPU released and cache cleared'"
    ;;
  "quick")
    log "Quick cleanup: python processes only..."
    KILL_CMD="sudo pkill -9 python 2>/dev/null && echo 'Python killed' || echo 'No python processes'"
    ;;
  "pattern")
    log "Pattern cleanup: killing '$PATTERN' processes..."
    KILL_CMD="pkill -9 -f '$PATTERN' 2>/dev/null && echo 'Killed' || echo 'No processes found'"
    ;;
esac

# --worker=all 사용 (gcloud가 순차적으로 처리하므로 SSH 과부하 없음)
log "Executing on all workers..."
gcloud compute tpus tpu-vm ssh "$INSTANCE" \
  --zone="$ZONE" \
  --worker="all" \
  --command="$KILL_CMD"

log "Done!"
