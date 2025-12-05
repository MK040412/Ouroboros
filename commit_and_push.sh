#!/usr/bin/env bash
set -euo pipefail

# Commit and push all changes in this repo to https://github.com/MK040412/Ouroboros

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/MK040412/Ouroboros"
COMMIT_MSG="${1:-"chore: sync $(date -u +'%Y-%m-%d %H:%M:%S UTC')"}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but not found." >&2
  exit 1
fi

# Avoid safe.directory errors when repo ownership differs.
git config --global --add safe.directory "$REPO_DIR"

# Ensure Git identity is configured.
if ! git -C "$REPO_DIR" config user.name >/dev/null; then
  echo "Git user.name is not set. Run: git config --global user.name \"Your Name\"" >&2
  exit 1
fi

if ! git -C "$REPO_DIR" config user.email >/dev/null; then
  echo "Git user.email is not set. Run: git config --global user.email \"you@example.com\"" >&2
  exit 1
fi

# Point origin at the requested repo.
if git -C "$REPO_DIR" remote get-url origin >/dev/null 2>&1; then
  git -C "$REPO_DIR" remote set-url origin "$REPO_URL"
else
  git -C "$REPO_DIR" remote add origin "$REPO_URL"
fi

BRANCH="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)"

git -C "$REPO_DIR" add -A

if git -C "$REPO_DIR" diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi
 
git -C "$REPO_DIR" commit -m "$COMMIT_MSG"

# If GitHub CLI is installed, ensure you're logged in.
if command -v gh >/dev/null 2>&1; then
  if ! gh auth status >/dev/null 2>&1; then
    echo "GitHub CLI not logged in. Run: gh auth login --web" >&2
    exit 1
  fi
fi

git -C "$REPO_DIR" push -u origin "$BRANCH"
echo "Pushed to $REPO_URL on branch $BRANCH"
