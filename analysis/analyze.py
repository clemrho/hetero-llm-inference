"""Summarize legacy baseline logs and foreground/background GapShift events.

    python analysis/analyze.py logs/run/*_r*.jsonl
    python analysis/analyze.py --json --aggregate logs/run/*_r*.jsonl

Error and decision events are excluded from foreground latency/cache metrics.
Pre-warm work is reported separately, including unmeasured cancelled requests.
"""
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summarize(path):
    with open(path) as f:
        events = [json.loads(line) for line in f if line.strip()]
    foreground = [r for r in events if r.get("kind", "call") == "call"]
    recs = [r for r in foreground if r.get("status", "ok") == "ok"]
    if not recs:
        return None
    ttft = [r.get("ttft_s", r["t_first"] - r["t_arrive"]) for r in recs if r.get("t_first")]
    by_wf = defaultdict(list)
    for r in recs:
        by_wf[r["wf_id"]].append(r)
    workflows = [r for r in events if r.get("kind") == "workflow"]
    wf_time = [r["t_done"] - r["t_start"] for r in workflows if r["status"] == "ok"] if workflows else [
        max(x["t_done"] for x in rs) - min(x["t_arrive"] for x in rs) for rs in by_wf.values()]
    switched = sum(len({x["dev"] for x in rs}) > 1 for rs in by_wf.values())
    prompt = sum(r.get("prompt_tokens") or 0 for r in recs)
    cached = sum(r.get("cached_tokens") or 0 for r in recs)
    usage_complete = all(r.get("cached_tokens") is not None and r.get("prompt_tokens") is not None for r in recs)
    ideal = 0
    for rs in by_wf.values():
        rs = sorted(rs, key=lambda x: x["step"])
        ideal += sum(prev.get("prompt_tokens") or 0 for prev in rs[:-1])
    prewarm = [r for r in events if r.get("kind") == "prewarm"]
    warm_ok = [r for r in prewarm if r["status"] == "ok"]
    warm_prompt = sum(r.get("prompt_tokens") or 0 for r in warm_ok)
    warm_uncached = sum(r["prompt_tokens"] - r["cached_tokens"] for r in warm_ok
                        if r.get("prompt_tokens") is not None and r.get("cached_tokens") is not None)
    foreground_uncached = sum(r["prompt_tokens"] - r["cached_tokens"] for r in recs
                              if r.get("prompt_tokens") is not None and r.get("cached_tokens") is not None)
    failed_attempts = sum(r.get("status", "ok") != "ok" for r in foreground)
    # A cancellation may already have performed prefill in the engine. Its actual
    # work is unknown; do not present successful-request counts as total GPU work.
    unknown = sum(r["status"] != "ok" or r.get("cached_tokens") is None for r in prewarm)
    warmed_next = {(r["wf_id"], r["step"] + 1, r["dev"]) for r in warm_ok}
    span = max(r["t_done"] for r in foreground) - min(r["t_arrive"] for r in foreground)
    per_device = {}
    for dev in sorted({r["dev"] for r in recs}):
        group = [r for r in recs if r["dev"] == dev]
        per_device[dev] = {"calls": len(group), "ttft_p50_s": pct([
            r.get("ttft_s", r["t_first"] - r["t_arrive"]) for r in group if r.get("t_first")], 50)}
    result = {
        "file": str(path), "policy": recs[0]["policy"], "calls": len(recs), "makespan_s": span,
        "ttft_p50_s": pct(ttft, 50), "ttft_p99_s": pct(ttft, 99),
        "wf_mean_s": statistics.mean(wf_time) if wf_time else float("nan"),
        "wf_p50_s": pct(wf_time, 50), "wf_p95_s": pct(wf_time, 95),
        "cache_hit": cached / prompt if prompt and usage_complete else float("nan"),
        "cache_hit_max": ideal / prompt if prompt else float("nan"),
        "cache_usage_coverage": sum(r.get("cached_tokens") is not None for r in recs) / len(recs),
        "devices": dict(Counter(r["dev"] for r in recs)), "per_device": per_device,
        "wf_switched": f"{switched}/{len(by_wf)}", "wf_completed": len(wf_time),
        "wf_failed": sum(r["status"] != "ok" for r in workflows),
        "failed_call_attempts": failed_attempts,
        "calls_per_s": len(recs) / span if span else 0,
        "output_tokens_per_s": sum(r.get("output_tokens") or 0 for r in recs) / span if span else 0,
        "prewarm_attempts": len(prewarm), "prewarm_completed": len(warm_ok),
        "prewarm_unmeasured": unknown, "prewarm_prompt_tokens": warm_prompt,
        "prewarm_uncached_tokens": warm_uncached,
        "prewarm_request_s": sum(r.get("latency_s", 0) for r in warm_ok),
        "prewarm_followed_by_hit": sum((r["wf_id"], r["step"], r["dev"]) in warmed_next and
                                       (r.get("cached_tokens") or 0) > 0 for r in recs),
        "all_uncached_tokens_lower_bound": foreground_uncached + warm_uncached,
        "compute_accounting_complete": unknown == 0 and failed_attempts == 0 and usage_complete,
        "gap_decisions": dict(Counter(r["reason"] for r in events if r.get("kind") == "gap_decision")),
    }
    meta_path = Path(path).with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        result.update(concurrency=meta["concurrency"], rep=meta["rep"], run_failed=meta["failed"] > 0,
                      config=meta.get("config", {}))
    return result


def aggregate(summaries):
    groups = defaultdict(list)
    for s in summaries:
        # Keep trace, model, profile, gap mode/scale and tuning parameters separate.
        config = dict(s.get("config", {}))
        config.pop("policy", None)
        key = (s["policy"], s.get("concurrency"), json.dumps(config, sort_keys=True))
        groups[key].append(s)
    out = []
    for (policy, conc, _), group in groups.items():
        row = {"policy": policy, "concurrency": conc, "reps": len(group),
               "failed_runs": sum(s.get("run_failed", False) for s in group), "files": [s["file"] for s in group]}
        for key in ("makespan_s", "wf_mean_s", "wf_p95_s", "ttft_p50_s", "ttft_p99_s", "cache_hit"):
            values = [s[key] for s in group if math.isfinite(s[key]) and not s.get("run_failed", False)]
            row[key] = {"mean": statistics.mean(values) if values else None,
                        "stdev": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}
        out.append(row)
    return out


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    summaries = []
    for path in a.paths:
        if path.endswith((".errors.jsonl", ".meta.json", ".metrics.json")):
            continue
        s = summarize(path)
        if s is None:
            if not a.json:
                print(f"{path}: no successful foreground calls")
            continue
        summaries.append(s)
        if not a.json:
            print(f"{s['policy']:>8} {path}\n"
                  f"    calls={s['calls']} makespan={s['makespan_s']:.2f}s wf_mean={s['wf_mean_s']:.3f}s "
                  f"wf_p50/p95={s['wf_p50_s']:.3f}/{s['wf_p95_s']:.3f}s\n"
                  f"    ttft p50/p99={s['ttft_p50_s']:.3f}/{s['ttft_p99_s']:.3f}s "
                  f"cache_hit={s['cache_hit']:.1%} (reference {s['cache_hit_max']:.1%})\n"
                  f"    devices={s['devices']} wf_switched={s['wf_switched']} failed_wf={s['wf_failed']} "
                  f"failed_attempts={s['failed_call_attempts']}\n"
                  f"    prewarm={s['prewarm_completed']}/{s['prewarm_attempts']} "
                  f"extra_uncached_tokens={s['prewarm_uncached_tokens']} unmeasured_attempts={s['prewarm_unmeasured']}")
            if s["cache_hit"] > s["cache_hit_max"] + 0.01:
                print("    !! hit rate exceeds previous-prompt reference; check run isolation/retries")
    if a.json:
        print(json.dumps(json_safe({"runs": summaries, "aggregate": aggregate(summaries) if a.aggregate else []}), indent=2))
    elif a.aggregate:
        print(json.dumps(aggregate(summaries), indent=2))


if __name__ == "__main__":
    main()
