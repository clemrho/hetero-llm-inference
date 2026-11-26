# hetero-llm-inference

Routing agent workflows across heterogeneous accelerators with vLLM.
CMU 11-868 LLM Systems · Yanze, Keyi, Hannah.

The current deployment is **Colab TPU v5e + AWS EC2 L40S**. Each device serves a
full model replica with prefix caching. Growing workflow prompts create a trade-off
between cache locality and load. **GapShift** uses tool-call gaps to rebuild a
known prefix on a better destination before the next call arrives.

- [EC2 setup and complete experiment commands](scripts/ec2/README.md)
- [Colab TPU + EC2 notebook](scripts/colab/tpu_ec2.ipynb)
- [GapShift design, cost model and ablations](docs/gapshift.md)
- [Midterm report](docs/midterm_report.md), [plan](docs/PLAN.md), [historical results](results/README.md)

The November 3 results and midterm report describe PSC experiments. The EC2
adaptation and GapShift implementation need fresh hardware measurements; no new
performance improvement is claimed here.

## Deployment

The router runs in Colab next to the TPU server on port 8000. Colab opens an SSH
tunnel from local port 8001 directly to the EC2 GPU server's loopback port 8000.
PSC's login node and Slurm jobs are no longer needed for this path.

1. On the EC2 instance, run `bash scripts/ec2/setup.sh`, then
   `bash scripts/ec2/serve.sh` inside tmux. Defaults: Qwen3-4B, bfloat16,
   16384 maximum sequence length, prefix caching, cached-token reporting.
2. Open `scripts/colab/tpu_ec2.ipynb` in Colab with a TPU v5e runtime. Set the
   branch containing these changes, the matching model/length, and EC2 connection
   details. Run the backend checks and collect profiles from Colab.
3. Replay the comparison matrix. Full commands and expected outputs are in the
   [EC2 guide](scripts/ec2/README.md).

The router uses `TPU_URL`, `GPU_URL` (`A100_URL` remains an alias), `PS_URL` and
`MODEL`. Server settings are environment variables; deployment code has no fixed
instance address, username or private-key path.

## Policies

| Policy | Rule |
|---|---|
| LB | Least in-flight requests; ties prefer `gpu` |
| FF | `gpu` until four requests are in flight, then spill to `tpu` |
| CA | First request by LB, then keep the workflow on that device |
| RR | Alternate devices using a global request counter |
| PS | Forward to the official vLLM Production Stack session router |
| PROFILE | Choose by measured queue, cached-prefill and decode cost; no pre-warming |
| GAPSHIFT | Keep a home device; pre-warm the known prefix during a tool gap when predicted savings exceed a margin |

`PROFILE` and `GAPSHIFT` require measured JSON profiles. GapShift bounds background
work and changes home only after a successful pre-warm. Unfinished pre-warms are
cancelled at tool return. Its default gap/size predictions use previous observed
steps; `--gap_mode oracle` is an explicit separate ablation.

## Layout

| Path | Purpose |
|---|---|
| `router/` | Baselines, GapShift, empirical cost model, shared streaming measurements |
| `scripts/ec2/` | Dedicated GPU environment, model server, direct SSH tunnel, run guide |
| `scripts/colab/tpu_ec2.ipynb` | TPU server, EC2 connection, profiles, repeated sweeps and ablations |
| `scripts/profile_backends.py` | Unloaded cold/warm TTFT and decode profiles on the actual client path |
| `scripts/run_experiments.sh` | Comparison matrix and optional ablations |
| `replay/` | Validated token-exact trace replay, metadata, engine metrics, mock backend |
| `analysis/analyze.py` | Workflow latency, TTFT, cache reuse, throughput, failures and pre-warm overhead |
| `tests/` | Protocol, routing, cancellation and profile-to-analysis integration tests |
| `baselines/prodstack/` | External official session-routing baseline |
| `traces/` | Synthetic traces and the common format for measured workflows |
| `scripts/psc/` | Historical PSC deployment, retained for reproduction |
| `results/`, `docs/` | Historical experiments, reports and current design notes |

## Local checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
DRY_RUN=1 bash scripts/ec2/serve.sh
```

Tests use an in-process HTTP transport and do not require accelerator access.
Hardware work still needed: validate both serving stacks, collect EC2/TPU profiles,
run repeated short/long context comparisons, calibrate queue estimates, and
collect/replay real QA-derived traces.
