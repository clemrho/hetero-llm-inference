"""Client-side workflow router for TPU v5e + EC2 L40S replicas.

LB / FF / CA / RR / PS preserve the baseline rules. PROFILE routes by measured
queue + cached-prefill + decode cost. GAPSHIFT rebuilds known prefixes during
tool gaps. The public call(wf_id, step, prompt, max_tokens) interface is unchanged.
"""
import argparse
import asyncio
import contextlib
import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

try:
    from .profiles import Profiles
    from .transport import completion
except ImportError:  # python router/router.py, or replay's historical module import
    from profiles import Profiles
    from transport import completion

BACKENDS = {
    "gpu": os.environ.get("GPU_URL") or os.environ.get("A100_URL") or "http://localhost:8001",
    "tpu": os.environ.get("TPU_URL", "http://localhost:8000"),
}
PS_URL = os.environ.get("PS_URL", "http://localhost:8080")
FAST = "gpu"  # fixed baseline preference, not a measured device ranking
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-4B")
POLICIES = ("LB", "FF", "CA", "RR", "PS", "PROFILE", "GAPSHIFT")


@dataclass
class GapConfig:
    margin_s: float = 0.05
    gap_estimate_s: float = 1.0
    growth_tokens: int = 512
    output_tokens: int = 32
    max_prewarms: int = 1
    prewarm_max_inflight: int = 1
    prewarm_timeout_s: float = 5.0
    cache_ttl_s: float = 30.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.output_tokens < 1 or self.prewarm_timeout_s <= 0:
            raise ValueError("output_tokens and prewarm_timeout_s must be positive")


class Router:
    def __init__(self, policy, log_path, ff_threshold=4, *, profiles=None, gap_config=None,
                 backends=None, model=None, timeout_s=600, client=None):
        policy = policy.upper()
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy}")
        if ff_threshold < 1:
            raise ValueError("ff_threshold must be positive")
        self.policy, self.ff_threshold = policy, ff_threshold
        self.backends = dict(BACKENDS if backends is None else backends)
        self.model = model or MODEL
        self.config = gap_config or GapConfig()
        self.profiles = Profiles(profiles, self.model, self.backends) if policy in ("PROFILE", "GAPSHIFT") else None
        self.inflight = {k: 0 for k in self.backends}
        self.pending = {k: {} for k in self.backends}
        self.residency = {k: OrderedDict() for k in self.backends}
        self.home, self.previous_lengths, self.growth = {}, {}, {}
        self.last_output, self.gap_history = {}, {}
        self.rr = 0
        self.prewarms = set()
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "a")
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=30))

    def event(self, **record):
        self.log.write(json.dumps({"policy": self.policy, **record}) + "\n")
        self.log.flush()

    def _least_loaded(self):
        return min(self.inflight, key=lambda k: (self.inflight[k], k != FAST))

    def cached(self, dev, wf_id, n):
        entry = self.residency[dev].get(wf_id)
        if entry is None or time.monotonic() - entry[1] > self.config.cache_ttl_s:
            return 0
        return min(n, entry[0])

    def _remember(self, dev, wf_id, n):
        cache = self.residency[dev]
        cache[wf_id] = (n, time.monotonic())
        cache.move_to_end(wf_id)
        capacity = self.profiles.devices[dev].get("cache_capacity_tokens") if self.profiles else None
        if capacity is not None:
            while cache and sum(v[0] for v in cache.values()) > capacity:
                cache.popitem(last=False)

    def _queue(self, dev, after_s=0):
        now = time.monotonic() + after_s
        # Client-side approximation, not vLLM's internal queue. Pre-warms share the
        # reservations; queue_parallelism is an explicit calibration parameter.
        ends = sorted(self.pending[dev].values())
        lanes = int(self.profiles.devices[dev].get("queue_parallelism", 1))
        # Reservations already include preceding queue time. Summing them would
        # count that work repeatedly and grow the estimate exponentially.
        return max(0.0, ends[-lanes] - now) if len(ends) >= lanes else 0.0

    def _cost(self, dev, wf_id, n, output, *, after_s=0, cached=None):
        c = self.cached(dev, wf_id, n) if cached is None else cached
        entry = self.residency[dev].get(wf_id)
        if cached is None and entry and time.monotonic() + after_s - entry[1] > self.config.cache_ttl_s:
            c = 0
        return self._queue(dev, after_s) + self.profiles.service(dev, n, c, output)

    def pick(self, wf_id, n=None, output=None):
        if self.policy == "LB":
            return self._least_loaded()
        if self.policy == "FF":
            if self.inflight[FAST] < self.ff_threshold:
                return FAST
            others = [k for k in self.backends if k != FAST]
            return min(others, key=lambda k: self.inflight[k]) if others else FAST
        if self.policy == "RR":
            names = list(self.backends)
            self.rr += 1
            return names[(self.rr - 1) % len(names)]
        if self.policy == "CA":
            if wf_id not in self.home:
                self.home[wf_id] = self._least_loaded()
            return self.home[wf_id]
        if self.profiles:
            costs = {d: self._cost(d, wf_id, n, output) for d in self.backends}
            best = min(costs, key=lambda d: (costs[d], d != FAST))
            home = self.home.get(wf_id)
            if home and (self.policy == "GAPSHIFT" or costs[home] <= costs[best] + self.config.margin_s):
                return home
            return best
        return "ps"

    async def _request(self, dev, wf_id, step, prompt, max_tokens, kind, **extra):
        t_arrive = time.time()
        url = PS_URL if dev == "ps" else self.backends[dev]
        n = len(prompt) if isinstance(prompt, list) else None
        reservation = object()
        estimated_cache = self.cached(dev, wf_id, n) if self.profiles else None
        estimate = self._cost(dev, wf_id, n, max_tokens) if self.profiles else None
        if dev in self.inflight:
            self.inflight[dev] += 1
            if estimate is not None:
                self.pending[dev][reservation] = time.monotonic() + estimate
        base = dict(kind=kind, wf_id=wf_id, step=step, dev=dev, t_arrive=t_arrive,
                    prompt_tokens=n, requested_output_tokens=max_tokens,
                    estimated_cached_tokens=estimated_cache, estimated_cost_s=estimate, **extra)
        try:
            result = await completion(self.client, url, self.model, prompt, max_tokens, wf_id)
            rec = {**base, **result, "status": "ok"}
            if self.profiles:
                self._remember(dev, wf_id, result["prompt_tokens"] or n)
            self.event(**rec)
            return rec
        except BaseException as error:
            self.event(**base, t_done=time.time(), status="cancelled" if isinstance(error, asyncio.CancelledError) else "error",
                       error=f"{type(error).__name__}: {error}"[:500])
            raise
        finally:
            if dev in self.inflight:
                self.inflight[dev] -= 1
                self.pending[dev].pop(reservation, None)

    async def call(self, wf_id, step, prompt, max_tokens):
        if self.profiles and not isinstance(prompt, list):
            raise ValueError("profile-based policies require token-ID prompts with exact lengths")
        n = len(prompt)
        dev = self.pick(wf_id, n, max_tokens)
        rec = await self._request(dev, wf_id, step, prompt, max_tokens, "call")
        self.home[wf_id] = dev
        if wf_id in self.previous_lengths:
            growth = max(0, n - self.previous_lengths[wf_id])
            self.growth[wf_id] = 0.5 * growth + 0.5 * self.growth.get(wf_id, growth)
        self.previous_lengths[wf_id], self.last_output[wf_id] = n, max_tokens
        return rec

    def _gap_decision(self, wf_id, prompt, oracle):
        h = self.home[wf_id]
        gap = self.gap_history.get(wf_id, self.config.gap_estimate_s)
        n = len(prompt) + round(self.growth.get(wf_id, self.config.growth_tokens))
        out = self.last_output.get(wf_id, self.config.output_tokens)
        if oracle is not None:
            gap, n, out = oracle["gap_s"], oracle["prompt_tokens"], oracle["output_tokens"]
        stay = self._cost(h, wf_id, n, out, after_s=gap)
        candidates = []
        for dev in self.backends:
            if dev == h or self.inflight[dev] > self.config.prewarm_max_inflight:
                continue
            capacity = self.profiles.devices[dev].get("cache_capacity_tokens")
            if capacity is not None and len(prompt) > capacity:
                continue
            rebuild = self._cost(dev, wf_id, len(prompt), 1)
            warm = self._cost(dev, wf_id, n, out, after_s=max(gap, rebuild), cached=len(prompt))
            move = max(0.0, rebuild - gap) + warm
            candidates.append((move, dev, rebuild))
        if not candidates:
            return None, {"reason": "destination_busy_or_capacity", "stay_s": stay, "gap_estimate_s": gap}
        move, dev, rebuild = min(candidates)
        fits = gap > 0 and rebuild <= min(gap, self.config.prewarm_timeout_s)
        useful = fits and move + self.config.margin_s < stay
        return dev if useful else None, {
            "reason": "prewarm" if useful else ("gap_too_short" if not fits else "no_predicted_saving"), "home": h,
            "target": dev, "stay_s": stay, "move_s": move, "prewarm_estimate_s": rebuild,
            "gap_estimate_s": gap, "next_prompt_estimate": n, "next_output_estimate": out,
        }

    async def tool_gap(self, wf_id, step, prompt, duration_s, *, oracle=None):
        """Wait for a tool concurrently with pre-warm of ONLY the known prompt.

        Actual duration finishes the simulated tool; it never predicts a move unless
        oracle is provided. Unfinished pre-warms are cancelled when the tool returns.
        """
        started = time.monotonic()
        task = None
        if self.policy == "GAPSHIFT" and duration_s > 0:
            dev, decision = self._gap_decision(wf_id, prompt, oracle)
            if len(self.prewarms) >= self.config.max_prewarms:
                dev, decision["reason"] = None, "prewarm_budget"
            self.event(kind="gap_decision", wf_id=wf_id, step=step, oracle=oracle is not None, **decision)
            if dev is not None:
                async def prewarm():
                    try:
                        await asyncio.wait_for(self._request(dev, wf_id, step, prompt, 1, "prewarm"),
                                               timeout=self.config.prewarm_timeout_s)
                        self.home[wf_id] = dev  # commit only after a successful response
                    except Exception as error:
                        self.event(kind="prewarm_failure", wf_id=wf_id, step=step,
                                   dev=dev, error=f"{type(error).__name__}: {error}"[:500])
                task = asyncio.create_task(prewarm())
                self.prewarms.add(task)
        try:
            await asyncio.sleep(max(0, duration_s))
        finally:
            if task is not None:
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                self.prewarms.discard(task)
            observed = time.monotonic() - started
            self.gap_history[wf_id] = 0.5 * observed + 0.5 * self.gap_history.get(wf_id, observed)

    def finish_workflow(self, wf_id):
        for mapping in (self.home, self.previous_lengths, self.growth, self.last_output, self.gap_history):
            mapping.pop(wf_id, None)
        if self.profiles:
            for dev, cache in self.residency.items():
                if self.profiles.devices[dev].get("cache_capacity_tokens") is None:
                    cache.pop(wf_id, None)

    async def close(self):
        for task in self.prewarms:
            task.cancel()
        await asyncio.gather(*self.prewarms, return_exceptions=True)
        self.prewarms.clear()
        await self.client.aclose()
        self.log.close()


async def synthetic_workflow(router, wf_id, run_tag, steps=5, ctx=2000, add=500, tool_s=1.0, out=32):
    prompt = f"[{run_tag} {wf_id}] " + "hello " * ctx
    for s in range(steps):
        await router.call(wf_id, s, prompt, max_tokens=out)
        if s + 1 < steps:
            await router.tool_gap(wf_id, s, prompt, tool_s)
            prompt += f" [obs {s}] " + "obs " * add


async def main(policy, n_wf):
    if policy in ("PROFILE", "GAPSHIFT"):
        raise ValueError("Use replay/replay.py --profiles for token-exact profile-based experiments")
    router = Router(policy, f"logs/log_{policy}_{n_wf}.jsonl")
    t0 = time.time()
    try:
        await asyncio.gather(*(synthetic_workflow(router, f"wf{i}", str(time.time_ns())) for i in range(n_wf)))
        print(f"{policy} n_wf={n_wf}: {time.time() - t0:.1f}s total")
    finally:
        await router.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="LB", choices=POLICIES)
    ap.add_argument("--n_wf", type=int, default=4)
    a = ap.parse_args()
    asyncio.run(main(a.policy, a.n_wf))
