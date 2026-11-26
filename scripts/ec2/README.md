# EC2 L40S + Colab TPU v5e

The current GPU backend is an EC2 L40S (for example, the G6e family). The router
still runs in Colab beside the TPU; Colab opens an outgoing SSH connection directly
to EC2. No PSC login node, Slurm allocation, or cluster modules are involved.

## 1. Start the GPU backend on EC2

Use a Linux instance with the NVIDIA driver already working (`nvidia-smi`). Clone
or copy the branch containing these changes, then run from the repository root:

```bash
bash scripts/ec2/setup.sh
bash scripts/ec2/serve.sh 2>&1 | tee logs/ec2_serve.log
```

Run the second command inside `tmux` so closing your terminal does not stop vLLM.
Wait for `/v1/models` to respond, then in a second terminal:

```bash
source scripts/ec2/env.sh
"$VENV/bin/python" scripts/check_backend.py --url http://127.0.0.1:8000 --model "$MODEL"
```

Defaults match the short-context experiments: `Qwen/Qwen3-4B`, bfloat16,
`MAX_LEN=16384`, `MAX_NUM_SEQS=8`, `GPU_MEM=0.85`, one GPU. Prefix caching and
per-request cached-token reporting are enabled. Startup may compile kernels.
The instance must have enough free GPU memory; stop other model servers first.

`setup.sh` creates `.venv-ec2`, separate from any speech/ASR environment. Caches
live in `$HOME/har-ec2` by default. Set `HAR_ROOT` to a sufficiently large EBS path
if needed. It installs a matching vLLM/Torch stack using uv's automatic Torch
backend selection, checks CUDA, downloads weights and available precompiled
kernels, and records resolved packages and driver details in `logs/ec2_setup/`.
To reproduce a tested version, set `VLLM_VERSION` when running setup.
Driver/wheel compatibility must still be checked on the actual instance.

Overrides are environment variables, and additional vLLM flags are arguments:

```bash
MODEL=Qwen/Qwen3-4B MAX_LEN=32768 MAX_NUM_SEQS=8 GPU_MEM=0.85 \
  bash scripts/ec2/serve.sh --max-num-batched-tokens 8192
DRY_RUN=1 bash scripts/ec2/serve.sh  # print the exact command without loading a model
```

Keep model, dtype and maximum context length consistent with the TPU. Record the
same configuration with every experiment. Historical PSC results are not EC2 results.

## 2. Connect Colab directly to EC2

Open `scripts/colab/tpu_ec2.ipynb`, choose a TPU v5e runtime, and set the repository
branch, model and context length. Its connection cell takes `EC2_HOST`,
`EC2_USER` (usually `ubuntu` for an Ubuntu AMI), and a private-key file available
inside Colab. Use a temporary runtime-local key path, not a shared Drive folder.

The EC2 security group must permit SSH from the client. The model API binds to
`127.0.0.1`, so there is no need to expose port 8000 publicly. The tunnel checks
and records the SSH host key on first connection. Verify the fingerprint through
your normal EC2 access path when establishing trust with a new host.

The same tunnel can be opened manually on the client machine:

```bash
EC2_HOST=your-instance-dns EC2_USER=ubuntu EC2_KEY=/path/to/key.pem \
  bash scripts/ec2/tunnel.sh
```

Keep that terminal open. Colab's `http://127.0.0.1:8001` then reaches EC2's
`http://127.0.0.1:8000`. The notebook also starts and closes its own tunnel process.

## 3. Profile from the router's location

On the Colab/router side, with both engines idle:

```bash
export MODEL=Qwen/Qwen3-4B
export TPU_URL=http://127.0.0.1:8000 GPU_URL=http://127.0.0.1:8001
python scripts/profile_backends.py --out logs/profiles.json \
  --lengths 1024 2048 4096 8192 16000 --reps 3
```

For `MAX_LEN=32768`, extend the profile through 32000 tokens. The largest prompt
plus profiling output must fit the served limit. Profiles must cover the trace's
smallest and largest prompts; replay rejects an uncovered range.

Profiles contain cold/warm streamed TTFT, actual warm cached-token counts,
decode seconds per token, raw samples, versions, model metadata and a persistent
connection probe. They include the network path, so do not subtract an old PSC RTT
or use an EC2-local profile for a Colab router. Warmup pairs for each length are
excluded from the measured samples. A cold hit or a missing warm hit is an error.

If startup logs show the engines' token capacities, pass `--gpu_cache_tokens N`
and `--tpu_cache_tokens N`. The router then uses a conservative LRU capacity
estimate. Without those flags, it uses only recent observed placements and TTL;
this is an estimate, not visibility into the engines' resident KV blocks.

## 4. Compare baselines, adaptive routing and GapShift

```bash
python traces/gen_synthetic.py --max_model_len 16384 --out logs/short.jsonl \
  --steps 3 5 --init_ctx 2048 8192 --per_combo 16 \
  --out_min 20 --out_max 60 --tool_min 0.2 --tool_max 0.8
TRACE=logs/short.jsonl PROFILES=logs/profiles.json REPS=3 \
  CONCURRENCY='8 16 32' bash scripts/run_experiments.sh
```

The default matrix compares `LB FF CA RR PROFILE GAPSHIFT`. Install and start the
existing official Production Stack router, then add `PS` to `POLICIES` if testing
that external baseline. It is not replaced by an in-house imitation.

Set `WITH_ABLATIONS=1` for gap scales, decision margins, disabled pre-warming and
an explicit oracle variant. This is a larger matrix; choose `REPS`, `CONCURRENCY`
and the trace size to match the available compute budget. Use `RUN_DIR` to choose
the output directory. Replaying directly also exposes every option:

```bash
python replay/replay.py --trace logs/short.jsonl --sweep \
  --policies CA PROFILE GAPSHIFT --concurrency_list 8 16 --reps 3 \
  --profiles logs/profiles.json --gap_mode estimated --gap_margin_s 0.05 \
  --max_prewarms 1 --prewarm_max_inflight 1 --log_dir logs/comparison
python analysis/analyze.py --json --aggregate logs/comparison/short/*_r*.jsonl
```

Run logs record foreground calls, pre-warms, decisions and workflow completion.
Metadata embeds profiles, tuning parameters and a source-code hash; engine `/metrics`, `/version`
and `/v1/models` are captured before and after each run. Resume skips a run only
if its completed metadata matches the requested settings. A partial or different
run requires `--overwrite` or a new log directory. Failed workflows return a
nonzero exit status and are excluded from aggregate timing means.

Analysis reports workflow mean/p50/p95, TTFT p50/p99, throughput, device splits,
cache reuse, failures, and pre-warm request/token overhead. Cancelled/failed
pre-warms may already have consumed accelerator work: their compute is explicitly
unmeasured, so token totals are lower bounds. Use engine metrics alongside these
client logs when evaluating interference or eviction. Small runs do not establish
tail latency or a speedup.

## Validation and remaining experiments

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

Tests exercise streamed protocol handling and the complete profile/replay/analysis
pipeline through an in-process HTTP transport. They do not measure accelerator
performance. Required hardware work remains: validate installation on EC2 and TPU,
collect fresh profiles, tune the queue model, repeat short/long context sweeps,
and replay measured HotpotQA traces when available. No EC2/GapShift speedup is
claimed by this implementation.

References: [AWS G6e](https://aws.amazon.com/ec2/instance-types/g6e/),
[vLLM GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/),
[vLLM serve options](https://docs.vllm.ai/en/stable/cli/serve/),
[automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/).
