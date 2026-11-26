# Source this file from bash. A dedicated environment avoids changing other projects.
EC2_SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$EC2_SCRIPTS/../.." && pwd)
export HAR_ROOT="${HAR_ROOT:-$HOME/har-ec2}"
export VENV="${VENV:-$REPO/.venv-ec2}"
export HF_HOME="${HF_HOME:-$HAR_ROOT/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HAR_ROOT/uv_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HAR_ROOT/uv_python}"
export PATH="$VENV/bin:$HOME/.local/bin:$PATH"
export MODEL="${MODEL:-Qwen/Qwen3-4B}"
export MAX_LEN="${MAX_LEN:-16384}"
export DTYPE="${DTYPE:-bfloat16}"
export GPU_MEM="${GPU_MEM:-0.85}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8000}"
