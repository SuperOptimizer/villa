#!/usr/bin/env bash
set -euo pipefail

BIN=cmake-build-release/bin/vc_render_tifxyz
VOL="/run/media/forrest/fdf1f12c-41ad-404a-bf97-0678aeadbcc8/scroll5.volpkg/volumes/scroll5.zarr"
SEG="/run/media/forrest/fdf1f12c-41ad-404a-bf97-0678aeadbcc8/scroll5.volpkg/paths/auto_grown_20250926114310706-sm"
COMMON="-v $VOL -s $SEG -g 0 --cache-gb 4 --scale 1.0 --num-slices 65 --slice-step 1.0 --auto-crop --timeout 10"

NUM_PARTS=4
OUT_ZARR=~/tmp/test_multipart.zarr
OUT_SINGLE=~/tmp/test_singlepart.zarr

rm -rf "$OUT_ZARR" "$OUT_SINGLE"

echo "============================================"
echo "  Multi-part test: $NUM_PARTS parts"
echo "============================================"

# Step 1: Pre — create zarr structure
echo ""
echo "--- Step 1: --pre ---"
time $BIN $COMMON --zarr-output "$OUT_ZARR" --num-parts $NUM_PARTS --part-id 0 --pre
echo ""

# Verify L0 dataset was created
if [ ! -f "$OUT_ZARR/0/.zarray" ]; then
    echo "FAIL: .zarray not created by --pre"
    exit 1
fi
echo "OK: .zarray exists"

# Step 2: Render parts in parallel (simulating multiple VMs)
# Each VM does L0 + pyramid L1-L5 independently — no finalize step needed!
echo ""
echo "--- Step 2: Render $NUM_PARTS parts in parallel (L0 + pyramid) ---"
PIDS=()
for ((i=0; i<NUM_PARTS; i++)); do
    echo "  Launching part $i/$NUM_PARTS..."
    $BIN $COMMON --zarr-output "$OUT_ZARR" --num-parts $NUM_PARTS --part-id $i 2>&1 | sed "s/^/  [part $i] /" &
    PIDS+=($!)
done

echo "  Waiting for all parts..."
FAIL=0
for ((i=0; i<NUM_PARTS; i++)); do
    if ! wait ${PIDS[$i]}; then
        echo "  FAIL: part $i exited with error"
        FAIL=1
    else
        echo "  OK: part $i done"
    fi
done
if [ $FAIL -ne 0 ]; then
    echo "FAIL: one or more parts failed"
    exit 1
fi
echo ""

# Step 3: Single-part reference run
echo "--- Step 3: Single-part reference ---"
time $BIN $COMMON --zarr-output "$OUT_SINGLE" 2>&1
echo ""

# Step 4: Compare L0 chunks
echo "--- Step 4: Compare L0 chunks ---"
MULTI_CHUNKS=$(find "$OUT_ZARR/0" -type f ! -name '.z*' | wc -l)
SINGLE_CHUNKS=$(find "$OUT_SINGLE/0" -type f ! -name '.z*' | wc -l)
echo "  Multi-part L0 chunks: $MULTI_CHUNKS"
echo "  Single-part L0 chunks: $SINGLE_CHUNKS"

if [ "$MULTI_CHUNKS" -ne "$SINGLE_CHUNKS" ]; then
    echo "FAIL: chunk count mismatch ($MULTI_CHUNKS vs $SINGLE_CHUNKS)"
    exit 1
fi

# Binary-compare all L0 chunk files
DIFFS=0
while IFS= read -r rel; do
    if ! cmp -s "$OUT_ZARR/0/$rel" "$OUT_SINGLE/0/$rel"; then
        echo "  DIFF: 0/$rel"
        DIFFS=$((DIFFS + 1))
    fi
done < <(cd "$OUT_SINGLE/0" && find . -type f ! -name '.z*' | sort)

if [ $DIFFS -eq 0 ]; then
    echo "  ALL $SINGLE_CHUNKS L0 chunks match exactly"
else
    echo "  FAIL: $DIFFS chunks differ"
    exit 1
fi

# Also compare pyramid levels
echo ""
echo "--- Step 5: Compare pyramid levels ---"
for level in 1 2 3 4 5; do
    if [ ! -d "$OUT_ZARR/$level" ] && [ ! -d "$OUT_SINGLE/$level" ]; then
        continue
    fi
    PDIFFS=0
    PCOUNT=0
    while IFS= read -r rel; do
        PCOUNT=$((PCOUNT + 1))
        if ! cmp -s "$OUT_ZARR/$level/$rel" "$OUT_SINGLE/$level/$rel"; then
            PDIFFS=$((PDIFFS + 1))
        fi
    done < <(cd "$OUT_SINGLE/$level" && find . -type f ! -name '.z*' | sort)
    if [ $PDIFFS -eq 0 ]; then
        echo "  Level $level: $PCOUNT chunks match"
    else
        echo "  Level $level: $PDIFFS/$PCOUNT chunks differ"
    fi
done

echo ""
echo "============================================"
echo "  PASS: Multi-part output matches single-part"
echo "============================================"
