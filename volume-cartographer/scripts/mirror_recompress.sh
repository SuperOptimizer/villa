#!/usr/bin/env bash
# mirror_recompress.sh — recompress every zarr under
#   s3://vesuvius-challenge/<SCROLL>/volumes/<VOL>.zarr
# to compress3d, mirrored at
#   s3://philodemos/forrest/<SCROLL>/volumes/<VOL>.zarr
#
# Usage:
#   mirror_recompress.sh list                       # print all source zarrs
#   mirror_recompress.sh one  <SCROLL>/volumes/<VOL>.zarr
#   mirror_recompress.sh all  [--exclude REGEX]     # process everything
#
# Env:
#   VC_BIN       path to vc_zarr_recompress  [default: build/ci-release-gcc/bin/vc_zarr_recompress]
#   SRC_BUCKET   [default: vesuvius-challenge]
#   DST_PREFIX   [default: philodemos/forrest]
#   TARGET_RATIO [default: 50]
#   JOBS         outer workers [default: 16]
#   LOG_DIR      [default: /ephemeral/recompress-logs]

set -euo pipefail

VC_BIN="${VC_BIN:-/home/ubuntu/volume-cartographer/build/ci-release-gcc/bin/vc_zarr_recompress}"
SRC_BUCKET="${SRC_BUCKET:-vesuvius-challenge}"
DST_PREFIX="${DST_PREFIX:-philodemos/forrest}"
TARGET_RATIO="${TARGET_RATIO:-50}"
JOBS="${JOBS:-16}"
LOG_DIR="${LOG_DIR:-/ephemeral/recompress-logs}"

mkdir -p "$LOG_DIR"

# Enumerate every "*.zarr/" directory under <SCROLL>/volumes/ across all scrolls.
list_zarrs() {
  # Top-level "scroll" prefixes.
  aws s3 ls "s3://${SRC_BUCKET}/" | awk '/PRE /{print $2}' | while read -r scroll; do
    scroll="${scroll%/}"
    # Volumes under this scroll (skip scrolls with no volumes/ dir).
    aws s3 ls "s3://${SRC_BUCKET}/${scroll}/volumes/" 2>/dev/null \
      | awk '/PRE .*\.zarr\/$/{print $2}' | while read -r vol; do
        vol="${vol%/}"
        echo "${scroll}/volumes/${vol}"
      done
  done
}

recompress_one() {
  local rel="$1"   # e.g. PHercParis4/volumes/2026....zarr
  local src="s3://${SRC_BUCKET}/${rel}"
  local dst="s3://${DST_PREFIX}/${rel}"
  local logbase="${LOG_DIR}/$(echo "$rel" | tr '/' '_')"

  echo "[$(date -u +%H:%M:%S)] >>> ${rel}"
  echo "    src=${src}"
  echo "    dst=${dst}"
  "$VC_BIN" "$src" "$dst" \
      --target-ratio "$TARGET_RATIO" \
      --jobs "$JOBS" \
      --log "${logbase}.shards" \
      --stats-pct 2 \
      2>&1 | tee "${logbase}.out"
  echo "[$(date -u +%H:%M:%S)] <<< done ${rel}"
}

cmd="${1:-}"
case "$cmd" in
  list)
    list_zarrs
    ;;
  one)
    [[ -n "${2:-}" ]] || { echo "usage: $0 one <SCROLL>/volumes/<VOL>.zarr" >&2; exit 2; }
    recompress_one "$2"
    ;;
  all)
    shift
    exclude=""
    if [[ "${1:-}" == "--exclude" ]]; then exclude="$2"; fi
    list_zarrs | while read -r rel; do
      if [[ -n "$exclude" ]] && echo "$rel" | grep -qE "$exclude"; then
        echo "[skip] $rel"; continue
      fi
      recompress_one "$rel"
    done
    ;;
  *)
    echo "usage: $0 {list|one <rel>|all [--exclude REGEX]}" >&2
    exit 2
    ;;
esac
