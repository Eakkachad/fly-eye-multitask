#!/usr/bin/env bash
# Sequential local grid (PLAN A7). Resumable: skips runs whose summary.json says "completed".
# Each run: gpu.lock + hard timeout + stall watchdog (no file in the run dir updated for 45 min -> kill; M8 compile takes ~25 min).
set -u
cd "$(dirname "$0")/.."
export PATH=$HOME/flyproj/tools/zigcc:$PATH CC=$HOME/flyproj/tools/zigcc/cc CXX=$HOME/flyproj/tools/zigcc/c++ PYTHONUNBUFFERED=1
PY=$HOME/flyproj/.venv/bin/python; LOCK=$HOME/flyproj/.orchestra/gpu.lock
OUT=runs; LOG=$OUT/grid.log; mkdir -p $OUT
GRID=${GRID:-scripts/grid_v2.tsv}; N_ITERS=${N_ITERS:-30000}; LR=${LR:-5e-4}
grep -v '^#' "$GRID" | while IFS=$'\t' read -r name model seed frac extra th; do
  [ -z "$name" ] && continue
  if [ -f $OUT/$name/summary.json ] && grep -q '"completed"' $OUT/$name/summary.json; then echo "SKIP $name (done)" >> $LOG; continue; fi
  if [ -f "$HOME/flyproj/.orchestra/GRID_PAUSE" ]; then echo "PAUSED before $name $(date -Is)" | tee -a $LOG; exit 0; fi
  rm -rf $OUT/$name; mkdir -p $OUT/$name
  echo "START $name $(date -Is)" | tee -a $LOG
  flock $LOCK timeout --signal=KILL $((th*3600)) $PY train.py --model $model --seed $seed --data-fraction $frac \
     --n-iters $N_ITERS --lr $LR --val-every 1000 $extra --out-dir $OUT --name $name > $OUT/$name.stdout 2>&1 < /dev/null &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 120
    newest=$(find $OUT/$name $OUT/$name.stdout -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
    if [ -n "$newest" ] && [ $(( $(date +%s) - ${newest%.*} )) -gt 2700 ]; then
      echo "WATCHDOG kill $name (stalled 45 min) $(date -Is)" | tee -a $LOG; pkill -KILL -f "train.py .*--name $name( |$)"; kill -KILL $pid 2>/dev/null
    fi
  done
  wait $pid; rc=$?
  st=$(grep -o '"status": "[a-z_]*"' $OUT/$name/summary.json 2>/dev/null)
  echo "END $name rc=$rc $st $(date -Is)" | tee -a $LOG
done
if [ -f "$HOME/flyproj/.orchestra/GRID_PAUSE" ]; then echo "STOPPED (paused) $(date -Is)" | tee -a $LOG; else echo "GRID_DONE $(date -Is)" | tee -a $LOG; fi
