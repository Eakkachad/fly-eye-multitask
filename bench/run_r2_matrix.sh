#!/usr/bin/env bash
# Round-2 speed matrix for m1 and m4 (sequential; each run holds the GPU lock).
# Usage: bench/run_r2_matrix.sh [extra profile_train.py args, e.g. --iters 30]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/flyproj/.venv/bin/python}"
LOCK="$HOME/flyproj/.orchestra/gpu.lock"
# no system gcc: Triton launcher / inductor use the zig shims (D043)
export PATH="$HOME/flyproj/tools/zigcc:$PATH" CC="$HOME/flyproj/tools/zigcc/cc" CXX="$HOME/flyproj/tools/zigcc/c++"
RES="$HERE/results"
EXTRA=("$@")
mkdir -p "$RES"

run() {  # run <name> <args...>
  local name="$1"; shift
  echo "== $name: $*"
  timeout --signal=KILL 900 flock "$LOCK" \
    "$PY" "$HERE/profile_train.py" "$@" "${EXTRA[@]}" --device cuda --out "$RES/$name.json" \
    || echo "!! $name exited non-zero (timeout/failure); continuing"
}

for m in m1 m4; do
  run "${m}_baseline"        --model "$m"
  run "${m}_tf32"            --model "$m" --tf32
  run "${m}_bf16"            --model "$m" --amp bf16
  run "${m}_compile_default" --model "$m" --compile default
  run "${m}_compile_ro"      --model "$m" --compile reduce-overhead
  run "${m}_bf16_compile"    --model "$m" --amp bf16 --compile default

  # batch-size sweep on the best variant so far (lowest median s/iter among the above)
  best="$("$PY" "$HERE/summarize_r2.py" --best-flags "$m" 2>/dev/null)"
  echo "== best flags for $m: ${best:-<none>}"
  for bs in 4 8 16; do
    # shellcheck disable=SC2086
    run "${m}_best_bs${bs}" --model "$m" $best --batch-size "$bs"
  done
done

"$PY" "$HERE/summarize_r2.py" > "$RES/summary_r2.md" && cat "$RES/summary_r2.md"
