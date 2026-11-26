"""
Replay agent traces through router.Router (owner: Hannah).

    python replay/replay.py --trace traces/synthetic.jsonl --policy CA --concurrency 4 --rep 0
    # -> logs/synthetic/CA_c4_r0.jsonl         one line per LLM call (written by Router)
    #    logs/synthetic/CA_c4_r0.meta.json     run settings + wall time
    #    logs/synthetic/CA_c4_r0.errors.jsonl  failed calls (only if any)

    python replay/replay.py --sweep --trace traces/synthetic.jsonl \
        --policies LB FF CA --concurrency_list 1 4 8 16 --reps 2   # whole matrix

How a workflow is replayed
    for each step:  router.call(prompt, max_tokens=output_tokens)  ->  sleep(tool_s)
    The prompt is a list of token ids (vLLM's /v1/completions accepts that), so token
    counts are exact and both devices see byte-identical input. Step k's prompt is
    the first prompt_tokens ids of one per-workflow random sequence, so every step is an
    exact prefix of the next one, just like a real agent's growing context.

Concurrency (closed loop, the default): N workers; each takes the next workflow from the
trace, runs it to the end, then takes another. --mode open instead starts every workflow
at its arrival_s (needs a trace generated with --rate).

Cache isolation between runs
    Every run uses a fresh random salt for the token ids. Lengths are identical across runs,
    content is not, so run B can't get free cache hits from prompts run A already sent.
    No server restart or cache reset needed between policies.
"""
import argparse
import asyncio
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import router.router as router_mod  # noqa: E402

TOKEN_LO, TOKEN_HI = 1000, 150000  # ordinary Qwen3 vocab ids, well below the special tokens (151643+)


def load_trace(path, limit=None):
    with open(path) as f:
        wfs = [json.loads(l) for l in f if l.strip()]
    if not wfs or len({w["wf_id"] for w in wfs}) != len(wfs):
        raise ValueError("trace must be nonempty with unique workflow IDs")
    for wf in wfs:
        st = wf["steps"]
        if not st:
            raise ValueError(f"{wf['wf_id']}: no steps")
        for s in st:
            if any(not isinstance(s[k], int) or s[k] < 1 for k in ("prompt_tokens", "output_tokens")):
                raise ValueError(f"{wf['wf_id']}: token counts must be positive integers")
            if not 0 <= float(s["tool_s"]) < float("inf"):
                raise ValueError(f"{wf['wf_id']}: invalid tool duration")
        for a, b in zip(st, st[1:]):
            if b["prompt_tokens"] < a["prompt_tokens"]:
                raise ValueError(f"{wf['wf_id']}: context shrinks")
    return wfs[:limit] if limit else wfs


def source_digest():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ("router", "replay", "analysis"):
        for path in sorted((root / folder).glob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def token_ids(salt, wf_id, n):
    seed = int.from_bytes(hashlib.sha256(f"{salt}|{wf_id}".encode()).digest()[:8], "little")
    rng = random.Random(seed)
    return [rng.randrange(TOKEN_LO, TOKEN_HI) for _ in range(n)]


async def call_with_retry(router, wf_id, step, prompt, max_tokens, errors, retries=1):
    for attempt in range(retries + 1):
        try:
            return await router.call(wf_id, step, prompt, max_tokens)
        except Exception as e:  # tunnel hiccup, server restart, ...
            errors.write(json.dumps({
                "wf_id": wf_id, "step": step, "attempt": attempt, "t": time.time(),
                "prompt_tokens": len(prompt), "error": f"{type(e).__name__}: {e}"[:500],
            }) + "\n")
            errors.flush()
            if attempt < retries:
                await asyncio.sleep(1.0)
    return None


async def run_workflow(router, wf, salt, errors, progress, gap_mode="estimated", gap_scale=1):
    st = wf["steps"]
    ids = token_ids(salt, wf["wf_id"], st[-1]["prompt_tokens"])
    started, status = time.time(), "failed"
    try:
        for k, s in enumerate(st):
            prompt = ids[:s["prompt_tokens"]]
            rec = await call_with_retry(router, wf["wf_id"], k, prompt, s["output_tokens"], errors)
            if rec is None:
                progress["failed"] += 1
                return
            if k + 1 < len(st):
                gap = s["tool_s"] * gap_scale
                oracle = {"gap_s": gap, "prompt_tokens": st[k + 1]["prompt_tokens"],
                          "output_tokens": st[k + 1]["output_tokens"]} if gap_mode == "oracle" else None
                await router.tool_gap(wf["wf_id"], k, prompt, gap, oracle=oracle)
        progress["done"] += 1
        status = "ok"
    finally:
        router.event(kind="workflow", wf_id=wf["wf_id"], status=status,
                     t_start=started, t_done=time.time(), expected_calls=len(st))
        router.finish_workflow(wf["wf_id"])


async def snapshot_backends():
    """Save raw engine metrics; missing /metrics is explicit, never fabricated."""
    result = {}
    async with httpx.AsyncClient(timeout=15) as client:
        for dev, url in router_mod.BACKENDS.items():
            info = {}
            for endpoint in ("/v1/models", "/version", "/metrics"):
                try:
                    response = await client.get(url.rstrip("/") + endpoint)
                    response.raise_for_status()
                    info[endpoint] = response.text if endpoint == "/metrics" else response.json()
                except Exception as error:
                    info[endpoint] = {"error": str(error)[:300]}
            result[dev] = info
    return result


async def warmup(lengths, max_tokens=8):
    """Hit every backend directly (not logged) so TPU/XLA compiles happen before timing."""
    async with httpx.AsyncClient(timeout=600) as c:
        for name, url in router_mod.BACKENDS.items():
            t0 = time.time()
            for n in lengths:
                for rep in range(2):
                    body = {"model": router_mod.MODEL, "prompt": token_ids(f"warmup{rep}", name, n),
                            "max_tokens": max_tokens, "ignore_eos": True, "temperature": 0}
                    r = await c.post(f"{url}/v1/completions", json=body)
                    r.raise_for_status()
            print(f"  warmup {name:>4} ({url}): {time.time() - t0:.1f}s for lengths {lengths}")


async def run_one(a, wfs, policy, conc, rep):
    out_dir = Path(a.log_dir) / Path(a.trace).stem
    name = f"{policy}_c{conc}_r{rep}"
    log_path = out_dir / f"{name}.jsonl"
    meta_path = out_dir / f"{name}.meta.json"
    config = {
        "trace_sha256": hashlib.sha256(Path(a.trace).read_bytes()).hexdigest(),
        "source_sha256": source_digest(), "timeout_s": a.timeout_s,
        "policy": policy, "concurrency": conc, "mode": a.mode, "gap_mode": a.gap_mode,
        "gap_scale": a.gap_scale, "gap_config": asdict(a.gap_config), "limit": a.limit,
        "profiles": a.profile_data, "backends": router_mod.BACKENDS,
        "model": router_mod.MODEL, "ff_threshold": a.ff_threshold,
    }
    if log_path.exists():
        if not a.overwrite:
            try:
                previous = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            except json.JSONDecodeError:
                previous = {}
            if (previous.get("done") == len(wfs) and previous.get("config") == config and
                    previous.get("log_sha256") == hashlib.sha256(log_path.read_bytes()).hexdigest()):
                print(f"skip {log_path} (matching completed run)")
                return True
            raise ValueError(f"{log_path} is incomplete or has different settings; use --overwrite or a new --log_dir")
        log_path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Invalidate an old completion marker before a replacement run can be interrupted.
    meta_path.unlink(missing_ok=True)

    salt = f"{name}-{time.time_ns()}"
    router = router_mod.Router(policy, str(log_path), ff_threshold=a.ff_threshold,
                               profiles=a.profile_data, gap_config=a.gap_config,
                               timeout_s=a.timeout_s)
    err_path = out_dir / f"{name}.errors.jsonl"
    errors = open(err_path, "w")
    progress = {"done": 0, "failed": 0}
    before = await snapshot_backends()
    t0 = time.time()

    async def ticker():
        while True:
            await asyncio.sleep(30)
            print(f"    {name}: {progress['done']}/{len(wfs)} done, {progress['failed']} failed, "
                  f"{time.time() - t0:.0f}s", flush=True)

    tick = asyncio.create_task(ticker())
    tasks = []
    try:
        if a.mode == "closed":
            queue = list(wfs)

            async def worker():
                while queue:
                    await run_workflow(router, queue.pop(0), salt, errors, progress, a.gap_mode, a.gap_scale)

            tasks = [asyncio.create_task(worker()) for _ in range(conc)]
            await asyncio.gather(*tasks)
        else:
            async def delayed(wf):
                await asyncio.sleep(max(0.0, wf["arrival_s"] - (time.time() - t0)))
                await run_workflow(router, wf, salt, errors, progress, a.gap_mode, a.gap_scale)

            tasks = [asyncio.create_task(delayed(wf)) for wf in wfs]
            await asyncio.gather(*tasks)
    finally:
        tick.cancel()
        for task in tasks:
            task.cancel()
        await asyncio.gather(tick, *tasks, return_exceptions=True)
        await router.close()
        errors.close()
    wall = time.time() - t0
    after = await snapshot_backends()
    (out_dir / f"{name}.metrics.json").write_text(json.dumps({"before": before, "after": after}, indent=2))
    if err_path.stat().st_size == 0:
        err_path.unlink()

    meta = {
        "trace": a.trace, "policy": policy, "concurrency": conc, "rep": rep, "mode": a.mode,
        "ff_threshold": a.ff_threshold, "salt": salt, "n_workflows": len(wfs),
        "done": progress["done"], "failed": progress["failed"], "wall_s": round(wall, 2),
        "t_start": t0, "backends": router_mod.BACKENDS, "model": router_mod.MODEL, "config": config,
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
    }
    temporary_meta = meta_path.with_suffix(".json.tmp")
    temporary_meta.write_text(json.dumps(meta, indent=2))
    temporary_meta.replace(meta_path)
    flag = "" if progress["failed"] == 0 else f"  !! {progress['failed']} workflows failed, see {err_path}"
    print(f"{name}: {progress['done']}/{len(wfs)} workflows in {wall:.1f}s -> {log_path}{flag}", flush=True)
    return progress["failed"] == 0


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", default="traces/synthetic.jsonl")
    ap.add_argument("--policy", type=str.upper, default="CA", choices=router_mod.POLICIES)
    ap.add_argument("--concurrency", type=int, default=4, help="concurrent workflows (closed loop)")
    ap.add_argument("--rep", type=int, default=0, help="repetition index, only used in the file name")
    ap.add_argument("--sweep", action="store_true", help="run policies x concurrency_list x reps")
    ap.add_argument("--policies", type=str.upper, choices=router_mod.POLICIES, nargs="+", default=None)
    ap.add_argument("--concurrency_list", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--mode", choices=["closed", "open"], default="closed")
    ap.add_argument("--ff_threshold", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N workflows (quick test)")
    ap.add_argument("--warmup", type=int, nargs="*", default=None,
                    help="warm up every backend first, e.g. --warmup 512 1024 2048 4096 8000")
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--profiles", help="measured JSON from scripts/profile_backends.py; required for PROFILE/GAPSHIFT")
    ap.add_argument("--gap_mode", choices=["estimated", "oracle"], default="estimated")
    ap.add_argument("--gap_scale", type=float, default=1, help="scale actual tool waits for ablations")
    ap.add_argument("--timeout_s", type=float, default=600)
    ap.add_argument("--gap_margin_s", type=float, default=0.05)
    ap.add_argument("--gap_estimate_s", type=float, default=1)
    ap.add_argument("--growth_tokens", type=int, default=512)
    ap.add_argument("--max_prewarms", type=int, default=1, help="0 disables pre-warming")
    ap.add_argument("--prewarm_max_inflight", type=int, default=1)
    ap.add_argument("--prewarm_timeout_s", type=float, default=5)
    ap.add_argument("--cache_ttl_s", type=float, default=30)
    a = ap.parse_args()
    if (min(a.concurrency_list + [a.concurrency, a.reps]) < 1 or
            not math.isfinite(a.gap_scale) or a.gap_scale < 0 or
            (a.limit is not None and a.limit < 1) or not math.isfinite(a.timeout_s) or a.timeout_s <= 0):
        ap.error("concurrency/reps must be positive and gap_scale nonnegative")
    a.gap_config = router_mod.GapConfig(margin_s=a.gap_margin_s, gap_estimate_s=a.gap_estimate_s,
        growth_tokens=a.growth_tokens, max_prewarms=a.max_prewarms,
        prewarm_max_inflight=a.prewarm_max_inflight, prewarm_timeout_s=a.prewarm_timeout_s,
        cache_ttl_s=a.cache_ttl_s)
    a.profile_data = json.loads(Path(a.profiles).read_text()) if a.profiles else None

    wfs = load_trace(a.trace, a.limit)
    calls = sum(len(w["steps"]) for w in wfs)
    print(f"trace {a.trace}: {len(wfs)} workflows, {calls} LLM calls; backends {router_mod.BACKENDS}")
    policies = (a.policies or ["LB", "FF", "CA"]) if a.sweep else [a.policy]
    if any(p in ("PROFILE", "GAPSHIFT") for p in policies):
        profile = router_mod.Profiles(a.profile_data, router_mod.MODEL, router_mod.BACKENDS)
        profile.validate_lengths([s["prompt_tokens"] for w in wfs for s in w["steps"]])
    info = await snapshot_backends()
    for dev, data in info.items():
        models = data["/v1/models"].get("data", [])
        served = next((m for m in models if m["id"] == router_mod.MODEL), None)
        if served is None:
            raise ValueError(f"{dev} does not serve {router_mod.MODEL}: {data['/v1/models']}")
        if served.get("max_model_len") and any(s["prompt_tokens"] + s["output_tokens"] > served["max_model_len"] for w in wfs for s in w["steps"]):
            raise ValueError(f"trace exceeds {dev} max_model_len={served['max_model_len']}")

    if a.warmup is not None:
        await warmup(a.warmup or [512, 1024, 2048, 4096])

    successful = True
    if a.sweep:
        # reps outermost, so an interrupted sweep still has one full pass of every config
        for rep in range(a.reps):
            for conc in a.concurrency_list:
                for p in policies:
                    successful = await run_one(a, wfs, p, conc, rep) and successful
    else:
        successful = await run_one(a, wfs, a.policy, a.concurrency, a.rep)
    return successful


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)
