#!/bin/bash
# Package the code for upload to Sol when git cannot be used there:  bash hpc/pack.sh
# Only files tracked by git at HEAD go in (no data, no credentials, no untracked scratch files). The archive carries the commit id
# and a sha256 file, so the copy on Sol can be checked against the committed version.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain)" ]; then
  echo "warning: uncommitted changes are NOT in the archive (it holds HEAD only)" >&2
fi
rev=$(git rev-parse --short HEAD)
mkdir -p dist
out="dist/wxrisk-code-$rev.tar.gz"
git archive --format=tar.gz --prefix=nanoWxRisk/ -o "$out" HEAD
(cd dist && shasum -a 256 "wxrisk-code-$rev.tar.gz" > "wxrisk-code-$rev.tar.gz.sha256")
echo "$out ($(du -h "$out" | cut -f1)), commit $rev"
cat "$out.sha256"

# Folder form for Globus sync: the same tracked files unpacked into dist/stage/nanoWxRisk. Sync that folder (not the repository
# working tree, which also holds .git, __pycache__, dist/ and other files that do not belong on Sol).
rm -rf dist/stage && mkdir -p dist/stage
git archive --prefix=nanoWxRisk/ HEAD | tar -x -C dist/stage
echo "folder for sync: dist/stage/nanoWxRisk ($(find dist/stage -type f | wc -l | tr -d ' ') files)"
