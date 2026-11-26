#!/usr/bin/env bash
# Render build step: copy the show's data (compact index + episodes.csv) from a private GitHub repo.
#
# The index holds transcript text, so it lives in a private repo, never in this public one.
# That repo's root is one show folder:  episodes.csv, show.json, index/<model>-512d.db
#
# Env (set in the Render dashboard):
#   DATA_REPO        owner/name of the private data repo, e.g. sharkybe/podcast_agent_data
#   DATA_REPO_TOKEN  a fine-grained GitHub token with read-only "Contents" access to that repo only
#   DATA_REPO_REF    optional branch or tag (default: the repo's default branch)
set -euo pipefail

dest="${PODCAST_DATA_DIR:-data}"
: "${DATA_REPO:?Set DATA_REPO (owner/name of the private data repo)}"
: "${DATA_REPO_TOKEN:?Set DATA_REPO_TOKEN (read-only token for the data repo)}"

rm -rf "$dest"
branch_args=()
if [ -n "${DATA_REPO_REF:-}" ]; then
  branch_args=(--branch "$DATA_REPO_REF")
fi
git clone --quiet --depth 1 "${branch_args[@]}" \
  "https://x-access-token:${DATA_REPO_TOKEN}@github.com/${DATA_REPO}.git" "$dest"
rm -rf "$dest/.git"   # drops the token from the clone's config too

test -f "$dest/episodes.csv" || { echo "ERROR: $dest/episodes.csv missing in $DATA_REPO" >&2; exit 1; }
ls "$dest"/index/*.db >/dev/null 2>&1 || { echo "ERROR: no index/*.db in $DATA_REPO" >&2; exit 1; }
echo "Data ready in $dest: $(ls "$dest"/index/*.db | xargs -n1 basename | tr '\n' ' ')"
