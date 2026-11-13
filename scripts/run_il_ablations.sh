#!/bin/bash
# Project 3 ablations: what matters for behaviour cloning on FetchPickAndPlace?
# Each run gets ~the same number of gradient steps (epochs scaled by 300/N demos).
# Results -> runs/il_ablations.csv  (name,demos,dart,chunk,success)
set -e
WS="$(cd "$(dirname "$0")/.." && pwd)"
source "$WS/env.sh" >/dev/null
THREADS=${THREADS:-2}
EVAL_EPS=${EVAL_EPS:-100}
CSV="$WS/runs/il_ablations.csv"
[ -f "$CSV" ] || echo "name,demos,dart,chunk,success" > "$CSV"

run() {  # name data demos chunk dart
  local name=$1 data=$2 demos=$3 chunk=$4 dart=$5
  if grep -q "^$name," "$CSV"; then echo "skip $name"; return; fi
  local epochs=$(( 300 * 300 / demos ))
  [ "$epochs" -gt 3600 ] && epochs=3600
  echo "== $name: $demos demos, chunk $chunk, dart $dart, $epochs epochs"
  python -m il_arm.train_bc "$data" --max-demos "$demos" --chunk "$chunk" --epochs "$epochs" \
      --out "$WS/runs/bc_$name" --threads "$THREADS" 2>&1 | grep -v Adroit | tail -1
  local s
  s=$(python -m il_arm.eval_policy "$WS/runs/bc_$name/best.pt" --episodes "$EVAL_EPS" 2>&1 | grep -v Adroit | tail -1 | awk '{print $3}')
  echo "$name,$demos,$dart,$chunk,$s" | tee -a "$CSV"
}

D="$WS/data/fetch_pick_300.hdf5"
N="$WS/data/fetch_pick_300_nodart.hdf5"
run main_300      "$D" 300 5 0.2
run nodart_300    "$N" 300 5 0.0
run chunk1_300    "$D" 300 1 0.2
run demos_100     "$D" 100 5 0.2
run demos_50      "$D" 50  5 0.2
run demos_25      "$D" 25  5 0.2
run nodart_50     "$N" 50  5 0.0
run nodart_100    "$N" 100 5 0.0
run nodart_25     "$N" 25  5 0.0
echo done
