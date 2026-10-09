#!/usr/bin/env bash
# Final evaluation of every completed grid run (PLAN A6/A7). TEST is evaluated ONCE: eval.py and floors.py refuse repeats.
#   scripts/eval_all.sh                  # SPLIT=test (default): the one-shot final evaluation
#   SPLIT=val scripts/eval_all.sh        # debugging on val (writes <run>/val*/, never test)
#   DRY_RUN=1 scripts/eval_all.sh        # print the commands only
#   DEVICE=cpu                           # optional, passed to eval.py (default cuda)
# For every run in the grid whose summary.json says "completed":
#   eval.py runs/<name>; m8: also --k 1..4 (any-time curve, A6.3); m6f/m7f: also --frontend-only (A6.2).
# Finally floors.py --split $SPLIT (--once on test). Runs not completed are listed and skipped.
set -u
cd "$(dirname "$0")/.."
export PATH=$HOME/flyproj/tools/zigcc:$PATH CC=$HOME/flyproj/tools/zigcc/cc CXX=$HOME/flyproj/tools/zigcc/c++ PYTHONUNBUFFERED=1
PY=$HOME/flyproj/.venv/bin/python; LOCK=$HOME/flyproj/.orchestra/gpu.lock
GRID=${GRID:-scripts/grid_v2.tsv}; SPLIT=${SPLIT:-test}; OUT=${OUT:-runs}; DRY_RUN=${DRY_RUN:-0}
DEVARG=""; [ -n "${DEVICE:-}" ] && DEVARG="--device $DEVICE"
LOG=$OUT/eval_all_$SPLIT.log
fails=0; missing=0
run() {  # run one command under the gpu lock with a hard timeout
  if [ "$DRY_RUN" = 1 ]; then echo "DRY: flock $LOCK timeout --signal=KILL 1800 $*"; return 0; fi
  echo "RUN $* $(date -Is)" | tee -a $LOG
  flock $LOCK timeout --signal=KILL 1800 "$@" >> $LOG 2>&1 < /dev/null
  rc=$?; [ $rc -ne 0 ] && { echo "FAIL rc=$rc: $*" | tee -a $LOG; fails=$((fails+1)); }
  return 0
}
while IFS=$'\t' read -r name model seed frac extra th; do
  [ -z "$name" ] && continue
  if ! { [ -f $OUT/$name/summary.json ] && grep -q '"completed"' $OUT/$name/summary.json; }; then
    echo "NOT COMPLETED, skipped: $name"; missing=$((missing+1)); continue
  fi
  run $PY eval.py $OUT/$name --split $SPLIT $DEVARG
  case $model in
    m8) for k in 1 2 3 4; do run $PY eval.py $OUT/$name --split $SPLIT --k $k $DEVARG; done ;;
    m6f|m7f) run $PY eval.py $OUT/$name --split $SPLIT --frontend-only $DEVARG ;;
  esac
done < <(grep -v '^#' "$GRID")
if [ "$SPLIT" = test ]; then run $PY floors.py --split test --once; else run $PY floors.py --split val; fi
echo "eval_all done: split=$SPLIT not_completed=$missing failures=$fails dry_run=$DRY_RUN"
[ "$missing" -gt 0 ] && echo "WARNING: $missing runs missing; finish them and re-run (finished evals refuse repeats and just report FAIL)"
exit 0
