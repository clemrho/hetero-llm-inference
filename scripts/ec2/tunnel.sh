#!/usr/bin/env bash
# Run on the router/client machine (e.g. Colab), NOT on EC2.
set -euo pipefail
: "${EC2_HOST:?Set EC2_HOST to the instance DNS or IP}"
: "${EC2_KEY:?Set EC2_KEY to the local private-key path}"
[[ -f "$EC2_KEY" ]] || { echo "Key file not found: $EC2_KEY" >&2; exit 1; }
exec ssh -N -T -i "$EC2_KEY" -p "${EC2_SSH_PORT:-22}" \
  -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L "127.0.0.1:${GPU_LOCAL_PORT:-8001}:127.0.0.1:${EC2_PORT:-8000}" \
  "${EC2_USER:-ubuntu}@${EC2_HOST}"
