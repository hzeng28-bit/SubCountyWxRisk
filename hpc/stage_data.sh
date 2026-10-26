#!/bin/bash
# Build dist/data_A with the files of hpc/transfer_manifest.txt (bundle A) under their relative paths, so one folder can be
# synced with Globus into $WXRISK_DATA on Sol:  bash hpc/stage_data.sh
# Hard links where the volume allows (no extra disk space), copies otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."
src="${WXRISK_DATA:-$HOME/wxrisk_data}"
rm -rf dist/data_A && mkdir -p dist/data_A
n=0
while read -r f; do
  case "$f" in ''|'#'*) continue;; esac
  mkdir -p "dist/data_A/$(dirname "$f")"
  ln "$src/$f" "dist/data_A/$f" 2>/dev/null || cp "$src/$f" "dist/data_A/$f"
  n=$((n+1))
done < hpc/transfer_manifest.txt
echo "dist/data_A: $n files, $(du -shL dist/data_A | cut -f1)"
