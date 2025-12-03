#!/usr/bin/env bash
set -euo pipefail

# Usage: gcp_transfer.sh <local_path> [remote_spec]
# remote_spec defaults to ouroboros:~
# Example (preserve relative path): gcp_transfer.sh ./src/app.py "ouroboros:~/project/src/app.py"

ZONE="europe-west4-b"
WORKER="all"
DEFAULT_DEST="ouroboros:~"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $(basename "$0") <local_path> [remote_spec]" >&2
  echo "  remote_spec defaults to ${DEFAULT_DEST}" >&2
  exit 1
fi

LOCAL_PATH=$1
DEST=${2:-$DEFAULT_DEST}

if [[ ! -e "$LOCAL_PATH" ]]; then
  echo "Local path does not exist: $LOCAL_PATH" >&2
  exit 1
fi

# remote_spec must include host and path (e.g., ouroboros:~/path)
if [[ "$DEST" != *:* ]]; then
  echo "remote_spec must include host and path (e.g., ouroboros:~/path)" >&2
  exit 1
fi

REMOTE_INSTANCE=${DEST%%:*}
REMOTE_PATH=${DEST#*:}

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

# Ensure the destination directory exists on the TPU VM.
gcloud compute tpus tpu-vm ssh "$REMOTE_INSTANCE" --zone="$ZONE" --worker="$WORKER" --command="mkdir -p \"$REMOTE_DIR_CMD\""

gcloud compute tpus tpu-vm scp "$LOCAL_PATH" "$DEST" --zone="$ZONE" --worker="$WORKER"
