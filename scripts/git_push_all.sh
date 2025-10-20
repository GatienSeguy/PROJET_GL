#!/usr/bin/env bash
# git_push_all.sh
# Pousse la branche courante vers l'origine, avec ajout, commit et push.

set -euo pipefail
IFS=$'\n\t'

usage() {
  echo "Usage: $0 [-m 'commit message'] [-r remote]"
  echo "  -m  message de commit (défaut: 'update')"
  echo "  -r  remote (défaut: origin)"
  exit 1
}

MSG="update"
REMOTE="origin"

while getopts ":m:r:" opt; do
  case ${opt} in
    m ) MSG="$OPTARG" ;;
    r ) REMOTE="$OPTARG" ;;
    * ) usage ;;
  esac
done

# Add all changes
git add -A

# Only commit if there are staged changes
if git diff --cached --quiet; then
  echo "Aucun changement à committer."
else
  git commit -m "$MSG"
fi

# Show current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "Pushing branch '$BRANCH' to remote '$REMOTE'..."

git push "$REMOTE" "$BRANCH"

echo "Push terminé."