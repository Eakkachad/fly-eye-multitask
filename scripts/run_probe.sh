#!/usr/bin/env bash
# C014 learnability probe (PLAN.md A4). Sequential M1 then M4, GPU lock + hard timeout + stall watchdog.
set -u
cd "$(dirname "$0")/.."
PY=$HOME/flyproj/.venv/bin/python; export PYTHONUNBUFFERED=1
OUT=runs_probe; mkdir -p $OUT
run() {  # name model timeout_s maxmin
  local name=$1 model=$2 tmo=$3 maxmin=$4
  flock $HOME/flyproj/.orchestra/gpu.lock timeout --signal=KILL $tmo \
    $PY train.py --model $model --seed 0 --n-iters 10000 --lr 5e-4 --val-every 1000 \
      --max-minutes $maxmin --out-dir $OUT --name $name > $OUT/$name.log 2>&1 &
  local pid=$!
  # watchdog: kill if log not updated for 10 min (WSL GPU drop, D029)
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    local f=$OUT/$name.log
    if [ -f $f ] && [ $(( $(date +%s) - $(stat -c %Y $f) )) -gt 600 ]; then
      echo "WATCHDOG: $name stalled >10 min, killing" | tee -a $f; pkill -KILL -P $pid; kill -KILL $pid
    fi
  done
  wait $pid; echo "[$name] exit $?" | tee -a $OUT/probe.log
}
run probe_m1_s0 m1 3600 55
run probe_m4_s0 m4 2700 40
echo DONE | tee -a $OUT/probe.log
