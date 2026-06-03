#!/usr/bin/env bash
#
# Re-profile Dijkstra2D and print self-time per source line.
#
# Pipeline: build the `profile_dijkstra` example with debug symbols -> generate a
# dSYM -> sample it with samply -> aggregate + symbolize with analyze_profile.py.
#
# Usage:
#   image_pathfinding/profiling/profile_dijkstra.sh [iters] [rate_hz]
#
# Defaults: 12000 iterations, 4000 Hz. Needs: cargo, samply, dsymutil, atos, python3.
# (macOS only — relies on dsymutil/atos. Install samply with `cargo install samply`.)

set -euo pipefail

ITERS="${1:-12000}"
RATE="${2:-4000}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BIN="$ROOT/target/release/examples/profile_dijkstra"
OUT="${TMPDIR:-/tmp}/dijkstra-profile"

for tool in cargo samply dsymutil atos python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: '$tool' not found on PATH" >&2
        [ "$tool" = samply ] && echo "       install it with: cargo install samply" >&2
        exit 1
    fi
done

mkdir -p "$OUT"
cd "$ROOT"

echo "==> building profile_dijkstra example with debug symbols"
CARGO_PROFILE_RELEASE_DEBUG=true \
    cargo build --release --example profile_dijkstra -p image_pathfinding

echo "==> generating dSYM"
dsymutil "$BIN" -o "$OUT/profile_dijkstra.dSYM"

echo "==> recording with samply ($ITERS iters @ ${RATE} Hz)"
samply record --save-only -o "$OUT/profile.json" --rate "$RATE" -- "$BIN" "$ITERS"

echo "==> analysis"
python3 "$HERE/analyze_profile.py" "$OUT/profile.json" "$OUT/profile_dijkstra.dSYM"

echo
echo "Raw profile saved to $OUT/profile.json"
echo "Open it interactively with:  samply load $OUT/profile.json"
