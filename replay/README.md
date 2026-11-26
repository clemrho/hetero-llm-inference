# Replay (owner: Hannah)

`replay.py` reads a trace and drives `router.Router` exactly as the trace says:
call the LLM with the step's context, wait `tool_s`, call again with the longer context, and so on.
Every policy replays the same workflows with the same lengths and waits.

## Run

```bash
export TPU_URL=http://localhost:8000 GPU_URL=http://localhost:8001   # GPU via SSH tunnel

# one run
python replay/replay.py --trace traces/synthetic.jsonl --policy CA --concurrency 4 --rep 0

# the whole baseline matrix (2 reps); warm up both devices first so XLA compiles aren't timed
python replay/replay.py --trace traces/synthetic.jsonl --sweep \
    --policies LB FF CA --concurrency_list 1 4 8 16 --reps 2 \
    --warmup 512 1024 2048 4096 8000
```

Output in `logs/<trace name>/`:
- `CA_c4_r0.jsonl`: foreground `call` events, plus workflow completion events; GapShift adds `prewarm` and `gap_decision` events
- `CA_c4_r0.meta.json`: settings, salt, wall time
- `CA_c4_r0.metrics.json`: backend models, versions and raw engine metrics before/after the run
- `CA_c4_r0.errors.jsonl`: failed calls, only if any (each call is retried once; a workflow whose call
  still fails is dropped, and the run's summary line says so)

A run is skipped only when completed metadata matches the requested configuration.
An interrupted or different run requires `--overwrite` or a new output directory.
Failed workflows cause a nonzero exit status.

## Profile-based methods

Collect profiles from the router machine while both engines are idle, then run:

```bash
python scripts/profile_backends.py --out logs/profiles.json
python replay/replay.py --trace traces/synthetic.jsonl --sweep \
    --policies CA PROFILE GAPSHIFT --profiles logs/profiles.json \
    --concurrency_list 4 8 16 --reps 3
```

PROFILE adapts placement without pre-warming; GAPSHIFT can rebuild the previous
prompt during the tool wait. Default predictions use only previous observed
steps. `--gap_mode oracle` reveals actual future sizes/gaps as a labelled ablation.
`--gap_scale`, `--gap_margin_s` and `--max_prewarms 0` support controlled ablations.
The [EC2 guide](../scripts/ec2/README.md) and [method notes](../docs/gapshift.md)
describe the full comparison and model assumptions.

Analysis keeps pre-warm work separate from foreground latency and cache reuse.
Cancelled pre-warms have unknown partial engine work and are counted explicitly.

## Details worth knowing

- **Concurrency** = number of workflows in flight (closed loop). `--mode open` starts workflows at
  `arrival_s` instead.
- **Prompts are token-id lists**, so token counts are exact. Step k's prompt is a prefix of step k+1's.
  The part standing in for the model's own output is also synthetic, so the KV of what the model
  actually generated is never reused. That's 50–200 tokens out of thousands per step.
- **No cache leaks between runs**: every run uses a fresh random salt for its token ids (same lengths,
  different content), so you don't need to restart vLLM between policies.
- `--limit 4` runs only the first 4 workflows, for a quick check.

## Testing without a TPU or GPU

`mock_vllm.py` is a fake vLLM (streaming, prefix cache on 16-token blocks, `cached_tokens` in usage)
for checking the pipeline. Its numbers mean nothing.

```bash
python replay/mock_vllm.py --port 8001 --prefill_tps 20000 --tpot 0.004 &   # "GPU"
python replay/mock_vllm.py --port 8000 --prefill_tps 6000  --tpot 0.008 &   # "TPU"
python traces/gen_synthetic.py --tool_min 0.05 --tool_max 0.2 --per_combo 2 --out /tmp/fast.jsonl
python replay/replay.py --trace /tmp/fast.jsonl --sweep --policies LB FF CA RR --concurrency_list 1 4 --reps 1
python analysis/analyze.py logs/fast/*.jsonl
```
