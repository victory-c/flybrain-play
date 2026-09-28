#!/usr/bin/env bash
# Commit and push what changed in this repo: code, sbatch scripts, results, Slurm logs, plus a fresh
# JOBS.md from sacct. Runs from cron every 30 min; safe to run by hand any time:
#   ./sync.sh ["commit message"]
set -euo pipefail
cd "$(dirname "$0")"
exec 9>.git/sync.lock
flock -n 9 || { echo "sync already running"; exit 0; }

python3 tools/jobs_ledger.py >/dev/null || echo "JOBS.md not updated (sacct unavailable?)"
git add -u
# new files: code, scripts, docs, logs and results only; stray downloads (images, archives, ...) stay local
git ls-files --others --exclude-standard -z | while IFS= read -r -d '' f; do
  case "$f" in
    *.py|*.sh|*.sbatch|*.md|*.json|*.html|*.js|*.ts|*.tsx|*.css|*.txt|*.yml|*.yaml|*.toml|*.npz|*.csv) git add -- "$f" ;;
    logs/*|fly_brain/results/*|fly_brain/dashboard/data/*|fly_brain/app/public/data/*) git add -- "$f" ;;
    *) echo "new file left local (not code or results): $f" ;;
  esac
done
# GitHub rejects files over 100 MB; leave those local and say so
git diff --cached --name-only --diff-filter=AM -z | while IFS= read -r -d '' f; do
  if [ "$(stat -c %s "$f")" -gt 95000000 ]; then
    echo "too big for GitHub, left local: $f"
    git reset -q -- "$f"
  fi
done
if git diff --cached --quiet; then
  echo "nothing to sync"
  exit 0
fi
files=$(git diff --cached --name-only | head -20)
git commit -q -m "${1:-Sync $(date '+%Y-%m-%d %H:%M')}" -m "$files"
git push -q origin main || { git pull -q --rebase origin main && git push -q origin main; }
echo "pushed $(git rev-parse --short HEAD): $(echo "$files" | wc -l) files"
