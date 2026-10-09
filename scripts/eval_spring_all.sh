#!/usr/bin/env bash
# Phase-2 evaluation (PLAN A9): Spring test, evaluated ONCE per output dir (eval_spring.py / floors_spring.py refuse repeats).
#   scripts/eval_spring_all.sh               # everything below
#   DRY_RUN=1 scripts/eval_spring_all.sh     # print the commands only (touches nothing)
#   FASTFLY=0                                # do not pass --fastfly to flyvis models (default 1; same numerics, much faster)
#   DEVICE=cpu                               # optional, passed to the evals (default cuda)
# For every COMPLETED run of scripts/grid_v2.tsv (39, phase 1) and scripts/grid_p2.tsv (15, phase 2):
#   eval_spring.py --ckpt last (PRIMARY, -> spring_last/), --ckpt best (secondary, -> spring_best/);
#   m8/m9m arms: also last ckpt at K=1..4 (-> spring_last_k<K>/, no per_pixel.npz).
# Then Spring floors (floors_spring.py --once); then phase-2 arms on the OLD Sintel test with last.pt via eval.py
# into test_last/ (L2 "second look"; existing test/ outputs are never touched; K1 arm evaluated at K=1).
# DRY_RUN=1 prints the full plan for all grid rows regardless of completion.
# Finished outputs are skipped (resumable); failures are logged and counted.
set -u
cd "$(dirname "$0")/.."
export PATH=$HOME/flyproj/tools/zigcc:$PATH CC=$HOME/flyproj/tools/zigcc/cc CXX=$HOME/flyproj/tools/zigcc/c++ PYTHONUNBUFFERED=1
PY=$HOME/flyproj/.venv/bin/python; LOCK=$HOME/flyproj/.orchestra/gpu.lock
OUT=${OUT:-runs}; DRY_RUN=${DRY_RUN:-0}; FASTFLY=${FASTFLY:-1}; TMO=${TMO:-3600}
GRIDS=${GRIDS:-"scripts/grid_v2.tsv scripts/grid_p2.tsv"}
DEVARG=""; [ -n "${DEVICE:-}" ] && DEVARG="--device $DEVICE"
LOG=$OUT/eval_spring_all.log
fails=0; missing=0; skipped=0
run() {  # one command under the gpu lock (flock OUTSIDE) with a hard timeout (INSIDE)
  if [ "$DRY_RUN" = 1 ]; then echo "DRY: flock $LOCK timeout --signal=KILL $TMO $*"; return 0; fi
  echo "RUN $* $(date -Is)" | tee -a $LOG
  flock $LOCK timeout --signal=KILL $TMO "$@" >> $LOG 2>&1 < /dev/null
  rc=$?; [ $rc -ne 0 ] && { echo "FAIL rc=$rc: $*" | tee -a $LOG; fails=$((fails+1)); }
  return 0
}
todo() { [ "$DRY_RUN" = 1 ] && return 0; [ ! -f "$1/metrics.json" ] || { echo "SKIP (exists) $1"; skipped=$((skipped+1)); return 1; }; }
[ "$DRY_RUN" = 1 ] || mkdir -p $OUT
P2ARR=()
for grid in $GRIDS; do
  while IFS=$'\t' read -r name model seed frac extra th; do
    [ -z "$name" ] && continue
    if [ "$DRY_RUN" != 1 ] && ! { [ -f $OUT/$name/summary.json ] && grep -q '"completed"' $OUT/$name/summary.json; }; then
      echo "NOT COMPLETED, skipped: $name"; missing=$((missing+1)); continue
    fi
    FF=""; if [ "$FASTFLY" = 1 ] && echo "$extra" | grep -q -- '--fastfly'; then FF="--fastfly"; fi
    for ck in last best; do
      todo $OUT/$name/spring_$ck && run $PY eval_spring.py $OUT/$name --split spring --ckpt $ck $FF $DEVARG
    done
    case $model in
      m8|m9m|m9|m9s) for k in 1 2 3 4; do
        todo $OUT/$name/spring_last_k$k && run $PY eval_spring.py $OUT/$name --split spring --ckpt last --k $k --no-per-pixel $DEVARG
      done ;;
    esac
    case $name in p2_*) P2ARR+=("$name|$extra") ;; esac
  done < <(grep -v '^#' "$grid")
done
todo floors_out/spring && run $PY floors_spring.py --split spring --once
# L2 second look: phase-2 arms on the old Sintel test, last ckpt, new dir test_last/ (K = the arm's training k_max)
for e in "${P2ARR[@]}"; do
  name=${e%%|*}; extra=${e#*|}
  KARG=""; kk=$(echo "$extra" | grep -o -- '--k-max [0-9]*' | awk '{print $2}'); [ -n "$kk" ] && KARG="--k $kk"
  todo $OUT/$name/test_last && run $PY eval.py $OUT/$name --split test --ckpt last.pt --out-name test_last $KARG $DEVARG
done
echo "eval_spring_all done: not_completed=$missing skipped=$skipped failures=$fails dry_run=$DRY_RUN"
[ "$missing" -gt 0 ] && echo "WARNING: $missing runs not completed; finish them and re-run (finished outputs are skipped)"
exit 0
