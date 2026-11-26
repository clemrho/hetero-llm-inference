# GapShift implementation

This implements the policy proposed in Section 3.4 of `11868_report.pdf` and
Section 3.5 of the Markdown midterm report. Each backend hosts a full model
replica. A pre-warm recomputes a known prefix through a normal one-token
completion; it does not move KV tensors or separate prefill from decode.

## Decision and information boundary

`PROFILE` chooses each foreground request by predicted queue + cached-prefill +
decode cost. Its cache term comes from recent completed requests for that
workflow, with optional capacity bounds and a TTL. It moves only when the saving
exceeds the configured margin. It never sends pre-warm requests.

`GAPSHIFT` initially chooses the device with the lowest measured cost and keeps a
home device. At each tool gap it compares staying with rebuilding the previous
prompt on another device:

`max(0, estimated_rebuild_time - estimated_gap) + next_warm_cost + margin < next_stay_cost`

Rebuild and next-call costs include the client/network path and predicted queue,
prefill and decode. Queue reservations include background work. The destination
must also satisfy the pre-warm in-flight and optional capacity bounds.
The rebuild must be predicted to finish within both the estimated gap and the
pre-warm timeout; otherwise this bounded implementation stays on the home engine.

Only the previous prompt is sent during the gap. Tool observations and future
synthetic token IDs are not supplied. By default, gap duration starts from a
configurable prior and adapts to past observed gaps; prompt growth adapts to
previous completed steps; output length uses the preceding request. The actual
tool duration ends the wait but is not an input to the decision. `--gap_mode
oracle` explicitly reveals the next request's sizes and actual gap for a separate
labelled ablation. `--gap_scale` changes the simulated workload, not the estimated
prior; report both settings.

The new home is committed only after a successful streamed pre-warm response.
An unfinished pre-warm is cancelled when the tool returns, so it does not extend
the foreground critical path. A failed or timed-out pre-warm retains the old home.
Client cancellation does not prove that the engine performed no work. The global
pre-warm budget bounds issued requests; there is no unbounded background queue.

## Cost model and limits

`scripts/profile_backends.py` measures cold and warm TTFT at several total context
lengths, including actual cached lengths, and decode seconds per output token.
The model interpolates over total length, then between cold and warm anchors by
reusable-prefix fraction. This retains dependence on total context length while
remaining simple enough to inspect. It is an approximation between unloaded
samples, not an exact model of batched attention.

Queue time is estimated from projected finish times of the router's outstanding
requests. `queue_parallelism` in each device profile defaults to one and can be
calibrated independently. It is not automatically equated to vLLM's
`max_num_seqs`: batched requests share compute and do not run at unloaded speed.
Reservations are counted once, avoiding repeated addition of earlier queue time.

The controller does not observe exact cache residency. Optional token capacities
bound an LRU placement estimate; TTL limits stale reuse assumptions. A request
may still miss after a successful pre-warm. Raw engine metrics and actual
`cached_tokens` are recorded to study that failure mode. Predicted next sizes
outside the measured grid use the nearest endpoint; foreground trace sizes must
be covered by the grid. Network placement, model, software and serving settings
must match profiling. The profile JSON is embedded in every run's metadata.

## Controlled comparison

Compare CA, PROFILE, and GAPSHIFT on identical trace shapes and concurrency with
fresh prefix salts per run. Include LB/FF/RR and the official PS session baseline
where available. Repeat runs and report variation. Ablate gap scale, decision
margin, disabled pre-warming (`--max_prewarms 0`), and estimated versus oracle
information. Disabling pre-warms leaves GapShift's profile-based initial placement
and fixed home, which differs from CA's least-loaded initial placement.

Foreground cache hit rate and latency exclude pre-warm requests. Report pre-warm
attempts, successes, uncached prompt tokens, request time, subsequent hits, and
failed/cancelled attempts alongside foreground results. Unknown partial work is
explicitly marked; token counts do not establish GPU utilization. Workflow events
make completion times include retries and tool gaps and distinguish failed runs.
No benchmark results for this new implementation are included yet.
