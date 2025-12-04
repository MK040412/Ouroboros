#!/usr/bin/env bash
set -euo pipefail

# Push current repo, then clone/update on all TPU workers, install deps, and run training.
# Usage: gcp_sync_and_train.sh [remote_dest_dir] [commit_message]
#   remote_dest_dir defaults to ~/ouroboros on the TPU workers (one shared path for all)
#   commit_message defaults to "chore: sync <timestamp>"

LOG_PREFIX="[gcp_sync]"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${1:-~/ouroboros}"
COMMIT_MSG="${2:-"chore: sync $(date -u +'%Y-%m-%d %H:%M:%S UTC')"}"

log() {
  echo "${LOG_PREFIX} $*"
}

if [[ ! -x "$REPO_DIR/gcp_run.sh" ]]; then
  echo "gcp_run.sh must be present and executable in $REPO_DIR" >&2
  exit 1
fi

git -C "$REPO_DIR" config --global --add safe.directory "$REPO_DIR" >/dev/null

if ! git -C "$REPO_DIR" config user.name >/dev/null; then
  echo "Git user.name is not set. Run: git config --global user.name \"Your Name\"" >&2
  exit 1
fi

if ! git -C "$REPO_DIR" config user.email >/dev/null; then
  echo "Git user.email is not set. Run: git config --global user.email \"you@example.com\"" >&2
  exit 1
fi

REMOTE_URL="$(git -C "$REPO_DIR" remote get-url origin)"
BRANCH="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)"

log "repo=$REPO_DIR branch=$BRANCH remote=$REMOTE_URL"

git -C "$REPO_DIR" add -A
if git -C "$REPO_DIR" diff --cached --quiet; then
  log "no changes to commit"
else
  log "committing with message: $COMMIT_MSG"
  git -C "$REPO_DIR" commit -m "$COMMIT_MSG"
fi

log "pushing to origin/$BRANCH"
git -C "$REPO_DIR" push origin "$BRANCH"

read -r -d '' REMOTE_CMD <<EOF
set -e
LOG_PREFIX="[gcp_sync.remote]"
DEST_DIR=$DEST_DIR
REMOTE_URL="$REMOTE_URL"
BRANCH="$BRANCH"

log() { echo "\${LOG_PREFIX} \$*"; }

log "host=\$(hostname) dest=\$DEST_DIR"

if ! command -v git >/dev/null 2>&1; then
  log "git is required on the TPU" >&2
  exit 1
fi

if [[ -d "\$DEST_DIR/.git" ]]; then
  log "existing repo found, pulling latest"
  cd "\$DEST_DIR"
  git fetch origin "\$BRANCH"
  git checkout "\$BRANCH"
  git pull --ff-only origin "\$BRANCH"
else
  log "cloning repo"
  mkdir -p "\$DEST_DIR"
  git clone "\$REMOTE_URL" "\$DEST_DIR"
  cd "\$DEST_DIR"
  git checkout "\$BRANCH"
fi

if [[ -f requirements.txt ]]; then
  log "creating venv and installing requirements"
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  log "requirements installed"
else
  log "requirements.txt not found, skipping install"
fi

log "running train_tpu_256.py"
python3 train_tpu_256.py
log "training command finished"
EOF

ESCAPED_CMD=$(printf '%q' "$REMOTE_CMD")

log "running setup on all workers to $DEST_DIR"
"$REPO_DIR/gcp_run.sh" "bash -lc $ESCAPED_CMD"
log "remote setup/train completed"
