#!/usr/bin/env bash
# git_pull_all.sh
# Récupère et rebase la branche courante depuis le remote (par défaut: origin)

set -euo pipefail
IFS=$'\n\t'

usage() {
  echo "Usage: $0 [-r remote]"
  echo "  -r  remote (défaut: origin)"
  exit 1
}

REMOTE="origin"

while getopts ":r:" opt; do
  case ${opt} in
    r ) REMOTE="$OPTARG" ;;
    * ) usage ;;
  esac
done

# Show current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "Fetching from '$REMOTE'..."

git fetch "$REMOTE"

echo "Rebasing '$BRANCH' onto '$REMOTE/$BRANCH'..."

git rebase "$REMOTE/$BRANCH"

echo "Pull (fetch+rebase) terminé."