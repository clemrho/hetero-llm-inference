#!/usr/bin/env bash
# Run on the EC2 L40S instance, with a working NVIDIA driver. No Slurm/modules.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
[[ $(uname -s) == Linux ]] || { echo 'Run setup on the Linux EC2 instance.' >&2; exit 1; }
nvidia-smi
mkdir -p "$HAR_ROOT" "$REPO/logs/ec2_setup"
if ! command -v uv >/dev/null; then
  python3 -m pip install --user uv
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.12 "$VENV"
fi
# vLLM owns its matching Torch/CUDA wheels; never install it into the ASR environment.
uv pip install --python "$VENV/bin/python" --torch-backend=auto \
  "vllm${VLLM_VERSION:+==$VLLM_VERSION}" -r "$REPO/requirements.txt" ninja
"$VENV/bin/vllm" --help > "$REPO/logs/ec2_setup/vllm-help.txt"
if grep -q download-kernels "$REPO/logs/ec2_setup/vllm-help.txt"; then
  "$VENV/bin/vllm" download-kernels
fi
"$VENV/bin/python" - <<'PY'
import os
import torch, vllm
from huggingface_hub import snapshot_download
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Check the NVIDIA driver and the selected vLLM CUDA wheel.')
print('vLLM', vllm.__version__, 'Torch', torch.__version__, 'CUDA', torch.version.cuda)
print('GPU', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
print(snapshot_download(os.environ['MODEL']))
PY
uv pip freeze --python "$VENV/bin/python" > "$REPO/logs/ec2_setup/requirements.lock.txt"
nvidia-smi -q > "$REPO/logs/ec2_setup/nvidia-smi.txt"
echo "Setup complete. Start the backend: bash scripts/ec2/serve.sh"
