#!/usr/bin/env bash
# Run on the router machine, after both backends pass checks and profiling.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
PYTHON="${PYTHON:-python3}"
TRACE="${TRACE:-traces/synthetic.jsonl}"
PROFILES="${PROFILES:-logs/profiles.json}"
RUN_DIR="${RUN_DIR:-logs/ec2_tpu_$(date +%Y%m%d_%H%M%S)}"
read -r -a policies <<< "${POLICIES:-LB FF CA RR PROFILE GAPSHIFT}"
read -r -a conc <<< "${CONCURRENCY:-4 8 16}"
common=(--trace "$TRACE" --sweep --policies "${policies[@]}"
  --concurrency_list "${conc[@]}" --reps "${REPS:-3}" --profiles "$PROFILES")
"$PYTHON" replay/replay.py "${common[@]}" --log_dir "$RUN_DIR/main" "$@"
if [[ ${WITH_ABLATIONS:-0} == 1 ]]; then
  for scale in 0 0.25 4; do
    "$PYTHON" replay/replay.py "${common[@]}" --gap_scale "$scale" --log_dir "$RUN_DIR/gap_$scale" "$@"
  done
  for margin in 0 0.2; do
    "$PYTHON" replay/replay.py "${common[@]}" --gap_margin_s "$margin" --log_dir "$RUN_DIR/margin_$margin" "$@"
  done
  "$PYTHON" replay/replay.py "${common[@]}" --max_prewarms 0 --log_dir "$RUN_DIR/no_prewarm" "$@"
  "$PYTHON" replay/replay.py "${common[@]}" --gap_mode oracle --log_dir "$RUN_DIR/oracle" "$@"
fi
shopt -s globstar nullglob
logs=("$RUN_DIR"/**/*_r*.jsonl)
"$PYTHON" analysis/analyze.py --json --aggregate "${logs[@]}" > "$RUN_DIR/summary.json"
echo "Saved experiments and summary to $RUN_DIR"
