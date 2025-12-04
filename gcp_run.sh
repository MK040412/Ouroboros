#!/usr/bin/env bash
set -euo pipefail

ZONE="europe-west4-b"
WORKER="all"
INSTANCE="ouroboros"
LOG_PREFIX="[gcp_run]"

log() {
  echo "${LOG_PREFIX} $*"
}

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") \"<command>\"" >&2
  exit 1
fi

COMMAND="$*"

log "zone=$ZONE worker=$WORKER instance=$INSTANCE"
log "command: $COMMAND"

gcloud compute tpus tpu-vm ssh "$INSTANCE" --zone="$ZONE" --worker="$WORKER" --command="$COMMAND"

log "completed"
