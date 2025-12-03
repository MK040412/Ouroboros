#!/usr/bin/env bash
set -euo pipefail

ZONE="europe-west4-b"
WORKER="all"
INSTANCE="ouroboros"

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") \"<command>\"" >&2
  exit 1
fi

COMMAND="$*"

gcloud compute tpus tpu-vm ssh "$INSTANCE" --zone="$ZONE" --worker="$WORKER" --command="$COMMAND"
