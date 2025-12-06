#!/usr/bin/env bash
set -euo pipefail

# Usage: gcp_transfer.sh [--zone Z] [--worker W] [--instance I] <local_path> [remote_path]
# Example: gcp_transfer.sh ./src/app.py ~/project/src/app.py
# Example: gcp_transfer.sh --worker 0 ./config.yaml ~/config.yaml

# 1. 기본값 설정 (Hardcoded Defaults)
ZONE="europe-west4-b"
WORKER="all"
INSTANCE="ouroboros"
LOG_PREFIX="[gcp_transfer]"

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

# 3. 인자 확인 (최소 1개: local_path)
if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") [--zone Z] [--worker W] [--instance I] <local_path> [remote_path]" >&2
  echo "  remote_path defaults to ~ (home directory)" >&2
  exit 1
fi

LOCAL_PATH=$1
REMOTE_PATH=${2:-~}

if [[ ! -e "$LOCAL_PATH" ]]; then
  echo "Local path does not exist: $LOCAL_PATH" >&2
  exit 1
fi

# Determine remote directory to create; default to $HOME when no directory is provided.
if [[ "$REMOTE_PATH" == */* ]]; then
  REMOTE_DIR=${REMOTE_PATH%/*}
else
  REMOTE_DIR="~"
fi

# Expand ~ on the remote by using $HOME in the mkdir command.
if [[ "$REMOTE_DIR" == "~"* ]]; then
  REMOTE_DIR_CMD="\$HOME${REMOTE_DIR#"~"}"
else
  REMOTE_DIR_CMD="$REMOTE_DIR"
fi

# Build full destination spec (instance:path)
DEST="${INSTANCE}:${REMOTE_PATH}"

log "zone=$ZONE worker=$WORKER instance=$INSTANCE"
log "local: $LOCAL_PATH -> remote: $REMOTE_PATH"

# Ensure the destination directory exists on the TPU VM.
gcloud compute tpus tpu-vm ssh "$INSTANCE" --zone="$ZONE" --worker="$WORKER" --command="mkdir -p \"$REMOTE_DIR_CMD\""

# Transfer file
gcloud compute tpus tpu-vm scp "$LOCAL_PATH" "$DEST" --zone="$ZONE" --worker="$WORKER"

log "completed"
