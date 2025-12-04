#!/usr/bin/env bash
set -euo pipefail

# Clone or update the repo on all TPU workers using gcp_run.sh.
# Usage: gcp_git_setup.sh [remote_dest_dir]
# Example: gcp_git_setup.sh ~/Ouroboros

REPO_URL="https://github.com/MK040412/Ouroboros.git"
DEST_DIR="${1:-~/ouroboros}"
LOG_PREFIX="[gcp_git_setup]"

log() {
  echo "${LOG_PREFIX} $*"
}

if [[ ! -x ./gcp_run.sh ]]; then
  echo "gcp_run.sh must be present and executable in the current directory." >&2
  exit 1
fi

read -r -d '' REMOTE_CMD <<EOF
set -e
DEST_DIR="$DEST_DIR"
REPO_URL="$REPO_URL"
echo "${LOG_PREFIX} running on \$(hostname) into \$DEST_DIR"

if ! command -v git >/dev/null 2>&1; then
  echo "${LOG_PREFIX} git is required on the TPU." >&2
  exit 1
fi

if [[ -d "\$DEST_DIR/.git" ]]; then
  echo "${LOG_PREFIX} existing repo found, pulling latest..."
  cd "\$DEST_DIR"
  git pull --ff-only
else
  echo "${LOG_PREFIX} cloning repo..."
  mkdir -p "\$DEST_DIR"
  git clone "\$REPO_URL" "\$DEST_DIR"
  cd "\$DEST_DIR"
fi

if [[ -f requirements.txt ]]; then
  echo "${LOG_PREFIX} creating venv and installing requirements..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  echo "${LOG_PREFIX} requirements installed."
else
  echo "${LOG_PREFIX} requirements.txt not found, skipping install."
fi
echo "${LOG_PREFIX} done on \$(hostname)."
EOF

# Quote the remote command safely for bash -lc.
ESCAPED_CMD=$(printf '%q' "$REMOTE_CMD")

log "running setup on all workers with destination: $DEST_DIR"
./gcp_run.sh "bash -lc $ESCAPED_CMD"
log "completed remote setup"
