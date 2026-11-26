#!/usr/bin/env bash
# Foreground server, suitable for tmux. Pass additional vLLM options as arguments.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
args=("$VENV/bin/vllm" serve "$MODEL" --host "$HOST" --port "$PORT"
  --dtype "$DTYPE" --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_MEM" --max-num-seqs "$MAX_NUM_SEQS"
  --enable-prefix-caching --enable-prompt-tokens-details)
if [[ -n ${MAX_NUM_BATCHED_TOKENS:-} ]]; then
  args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${args[@]}" "$@"
  printf '\n'
  exit 0
fi
[[ -x "$VENV/bin/vllm" ]] || { echo 'Run scripts/ec2/setup.sh first.' >&2; exit 1; }
"$VENV/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; check driver and installed vLLM/Torch wheels.')
print('Serving on', torch.cuda.get_device_name(0), flush=True)
PY
exec "${args[@]}" "$@"
