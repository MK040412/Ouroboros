#!/usr/bin/env bash
set -euo pipefail

# Usage: gcp_retrieve.sh [--zone Z] [--worker W] [--instance I] <remote_path> [local_path]
# Example: gcp_retrieve.sh ~/loss.png ./outputs/loss.png
# Example: gcp_retrieve.sh --worker 0 /tmp/loss_log.csv ./loss_log.csv

# 1. 기본값 설정 (Hardcoded Defaults)
ZONE="europe-west4-b"
WORKER="0"  # 기본값: worker 0 (retrieve는 보통 특정 워커에서 가져옴)
INSTANCE="ouroboros"
LOG_PREFIX="[gcp_retrieve]"

log() {
  echo "${LOG_PREFIX} $*"
}

# 2. 인자 파싱 (옵션 처리)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --zone)
      ZONE="$2"
      shift 2
      ;;
    --worker)
      WORKER="$2"
      shift 2
      ;;
    --instance)
      INSTANCE="$2"
      shift 2
      ;;
    --) # '--'를 만나면 옵션 파싱 강제 종료
      shift
      break
      ;;
    -*) # 알 수 없는 옵션 처리
      echo "Error: Unknown option $1" >&2
      exit 1
      ;;
    *) # 옵션이 아닌 인자가 나오면 루프 종료
      break
      ;;
  esac
done

# 3. 인자 확인 (최소 1개: remote_path)
if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") [--zone Z] [--worker W] [--instance I] <remote_path> [local_path]" >&2
  echo "  local_path defaults to current directory with same filename" >&2
  echo "  worker defaults to 0 (unlike transfer which defaults to 'all')" >&2
  exit 1
fi

REMOTE_PATH=$1

# local_path 기본값: 현재 디렉토리에 같은 파일명
if [[ $# -ge 2 ]]; then
  LOCAL_PATH=$2
else
  LOCAL_PATH="./$(basename "$REMOTE_PATH")"
fi

# 로컬 디렉토리 생성 (필요시)
LOCAL_DIR=$(dirname "$LOCAL_PATH")
if [[ "$LOCAL_DIR" != "." && ! -d "$LOCAL_DIR" ]]; then
  mkdir -p "$LOCAL_DIR"
  log "Created local directory: $LOCAL_DIR"
fi

# Expand ~ to $HOME on remote (prevent local expansion)
if [[ "$REMOTE_PATH" == "~"* ]]; then
  # Get remote home directory and replace ~
  REMOTE_HOME=$(gcloud compute tpus tpu-vm ssh "$INSTANCE" --zone="$ZONE" --worker="$WORKER" --command="echo \$HOME" 2>/dev/null | tr -d '\r\n')
  REMOTE_PATH="${REMOTE_HOME}${REMOTE_PATH#"~"}"
fi

# Build full source spec (instance:path)
SRC="${INSTANCE}:${REMOTE_PATH}"

log "zone=$ZONE worker=$WORKER instance=$INSTANCE"
log "remote: $REMOTE_PATH -> local: $LOCAL_PATH"

# Retrieve file from TPU VM
gcloud compute tpus tpu-vm scp "$SRC" "$LOCAL_PATH" --zone="$ZONE" --worker="$WORKER"

log "completed"
