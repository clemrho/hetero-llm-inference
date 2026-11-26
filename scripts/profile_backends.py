"""Measure unloaded cold/warm TTFT and decode on the actual router-to-device path.

Run in Colab after opening the EC2 tunnel, with both engines idle. Fresh token
prefixes isolate every sample. Compilation warmups are excluded from the profile.
"""
import argparse
import asyncio
import json
import random
import statistics
import sys
import time
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from router.router import BACKENDS, MODEL  # noqa: E402
from router.transport import completion  # noqa: E402


def tokens(seed, count):
    rng = random.Random(seed)
    return [rng.randrange(1000, 150000) for _ in range(count)]


async def profile(a):
    data = {"version": 1, "model": MODEL, "created_at": time.time(),
            "notes": "Client-measured seconds, including network; unloaded engine required.",
            "backends": {}}
    async with httpx.AsyncClient(timeout=a.timeout_s) as client:
        for dev, url in BACKENDS.items():
            r = await client.get(f"{url}/v1/models")
            r.raise_for_status()
            models = r.json()
            model = next((m for m in models["data"] if m["id"] == MODEL), None)
            if model is None:
                raise ValueError(f"{url} does not serve {MODEL}")
            if model.get("max_model_len") and max(a.lengths) + a.output_tokens > model["max_model_len"]:
                raise ValueError(f"profile length + output exceeds {dev} max_model_len={model['max_model_len']}")
            info = {"url": url, "models": models, "points": [], "samples": [],
                    "queue_parallelism": 1, "cache_capacity_tokens": getattr(a, f"{dev}_cache_tokens")}
            for endpoint in ("/version", "/metrics"):
                response = await client.get(url + endpoint)
                info[endpoint] = response.text if response.is_success else {"http_status": response.status_code}
            # Prime the same keep-alive client that measures completions.
            probes = []
            for _ in range(5):
                start = time.perf_counter()
                probe = await client.get(f"{url}/v1/models")
                probe.raise_for_status()
                probes.append(time.perf_counter() - start)
            info["persistent_connection_probe_s"] = statistics.median(probes)
            for n in sorted(set(a.lengths)):
                rows = []
                for rep in range(a.warmup_reps + a.reps):
                    tag = f"profile-{uuid.uuid4().hex}"
                    prefix = tokens(tag, n - a.suffix_tokens)
                    cold = prefix + tokens(tag + "cold", a.suffix_tokens)
                    warm = prefix + tokens(tag + "warm", a.suffix_tokens)
                    c = await completion(client, url, MODEL, cold, a.output_tokens, tag)
                    w = await completion(client, url, MODEL, warm, a.output_tokens, tag)
                    if rep < a.warmup_reps:
                        continue
                    if c["prompt_tokens"] != n or w["prompt_tokens"] != n:
                        raise ValueError(f"{dev}: backend did not honor token-ID prompt length")
                    if c["output_tokens"] != a.output_tokens or w["output_tokens"] != a.output_tokens:
                        raise ValueError(f"{dev}: backend did not honor the requested decode length")
                    if c["cached_tokens"] != 0 or not w["cached_tokens"]:
                        raise ValueError(f"{dev}: cold must miss and warm must hit; got "
                                         f"{c['cached_tokens']}/{w['cached_tokens']}. Check APC/usage flags.")
                    row = {"prompt_tokens": n, "cold_ttft_s": c["ttft_s"], "warm_ttft_s": w["ttft_s"],
                           "cold_request_s": c["latency_s"], "warm_request_s": w["latency_s"],
                           "warm_cached_tokens": w["cached_tokens"],
                           "decode_s_per_token": statistics.mean((x["latency_s"] - x["ttft_s"]) /
                               max(1, x["output_tokens"] - 1) for x in (c, w))}
                    rows.append(row)
                    info["samples"].append({"rep": rep - a.warmup_reps, "cold": c, "warm": w})
                point = {key: statistics.median(row[key] for row in rows) for key in rows[0]}
                info["points"].append(point)
                print(f"{dev} n={n}: cold={point['cold_ttft_s']:.4f}s warm={point['warm_ttft_s']:.4f}s "
                      f"cached={point['warm_cached_tokens']:.0f} decode={point['decode_s_per_token']:.5f}s/token", flush=True)
            data["backends"][dev] = info
            # Checkpoint each completed backend, so a long TPU run is recoverable.
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(data, indent=2))
    print(f"saved measured profiles to {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="logs/profiles.json")
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16000])
    ap.add_argument("--suffix_tokens", type=int, default=32)
    ap.add_argument("--output_tokens", type=int, default=32)
    ap.add_argument("--warmup_reps", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout_s", type=float, default=1200)
    ap.add_argument("--gpu_cache_tokens", type=int, help="optional measured capacity from GPU startup log")
    ap.add_argument("--tpu_cache_tokens", type=int, help="optional measured capacity from TPU startup log")
    a = ap.parse_args()
    if min(a.lengths) <= a.suffix_tokens or a.suffix_tokens < 1 or a.output_tokens < 2 or a.reps < 1 or a.warmup_reps < 0:
        ap.error("lengths must exceed positive suffix_tokens; output_tokens >= 2, reps >= 1, warmup_reps >= 0")
    asyncio.run(profile(a))


if __name__ == "__main__":
    main()
