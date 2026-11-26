"""Deterministic protocol/policy tests; fake timings are never experiment results."""
import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.analyze import summarize  # noqa: E402
from router.router import GapConfig, Router  # noqa: E402
from router.profiles import Profiles  # noqa: E402
from router.transport import completion  # noqa: E402

MODEL = "test-model"
BACKENDS = {"gpu": "http://gpu", "tpu": "http://tpu"}


def profiles():
    return {"version": 1, "model": MODEL, "backends": {
        dev: {"points": [{"prompt_tokens": n, "cold_ttft_s": cold, "warm_ttft_s": warm,
                           "warm_cached_tokens": n - 16, "decode_s_per_token": 0.001}
                          for n in (64, 256)], "cache_capacity_tokens": 512}
        for dev, cold, warm in (("gpu", 0.02, 0.015), ("tpu", 0.04, 0.001))}}


class Stream(httpx.AsyncByteStream):
    def __init__(self, events, delay=0):
        self.events, self.delay = events, delay

    async def __aiter__(self):
        for event in self.events:
            await asyncio.sleep(self.delay)
            yield f"data: {json.dumps(event) if isinstance(event, dict) else event}\n\n".encode()


class FakeBackends:
    def __init__(self, tpu_delay=0, fail_tpu=False):
        self.requests, self.prefixes = [], {}
        self.tpu_delay, self.fail_tpu = tpu_delay, fail_tpu

    async def handle(self, request):
        body = json.loads(request.content)
        dev, prompt = request.url.host, body["prompt"]
        self.requests.append((dev, body))
        if dev == "tpu" and self.fail_tpu:
            return httpx.Response(503)
        cached = 0
        for old in self.prefixes.get(dev, []):
            common = 0
            for a, b in zip(prompt, old):
                if a != b:
                    break
                common += 1
            cached = max(cached, common // 16 * 16)
        self.prefixes.setdefault(dev, []).append(prompt)
        events = [{"choices": [{"text": ""}]}, {"choices": [{"text": "x"}]},
                  {"choices": [], "usage": {"prompt_tokens": len(prompt), "completion_tokens": body["max_tokens"],
                   "prompt_tokens_details": {"cached_tokens": cached}}}, "[DONE]"]
        return httpx.Response(200, stream=Stream(events, self.tpu_delay if dev == "tpu" else 0))


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "run.jsonl"
        self.routers = []

    async def asyncTearDown(self):
        for router in self.routers:
            await router.close()
        self.tmp.cleanup()

    def router(self, policy="GAPSHIFT", fake=None, **kwargs):
        fake = fake or FakeBackends()
        client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
        router = Router(policy, self.path, profiles=profiles(), backends=BACKENDS, model=MODEL,
                        client=client, **kwargs)
        self.routers.append(router)
        return router, fake

    def records(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    async def test_baseline_rules(self):
        for policy in ("LB", "FF", "CA", "RR", "PS"):
            router, _ = self.router(policy)
            self.assertEqual(router.pick("w"), "ps" if policy == "PS" else "gpu")
            router.inflight["gpu"] = 4
            if policy in ("LB", "FF", "RR"):
                self.assertEqual(router.pick("w"), "tpu")
            if policy == "CA":
                self.assertEqual(router.pick("w"), "gpu")

    async def test_profile_moves_without_prewarm(self):
        router, fake = self.router("PROFILE")
        await router.call("w", 0, list(range(64)), 2)
        router.pending["gpu"]["other"] = time.monotonic() + 1
        await router.call("w", 1, list(range(80)), 2)
        self.assertEqual([d for d, _ in fake.requests], ["gpu", "tpu"])
        self.assertFalse(any(r["kind"] == "prewarm" for r in self.records()))

    async def test_successful_prewarm_reuses_only_known_prefix(self):
        router, fake = self.router(gap_config=GapConfig(margin_s=0, growth_tokens=16))
        await router.call("w", 0, list(range(64)), 2)
        router.pending["gpu"]["other"] = time.monotonic() + 1
        await router.tool_gap("w", 0, list(range(64)), 0.02)
        self.assertEqual(router.home["w"], "tpu")
        self.assertEqual(fake.requests[1][1]["prompt"], list(range(64)))
        self.assertEqual(fake.requests[1][1]["max_tokens"], 1)
        result = await router.call("w", 1, list(range(80)), 2)
        self.assertEqual(result["cached_tokens"], 64)
        summary = summarize(self.path)
        self.assertEqual(summary["calls"], 2)
        self.assertEqual(summary["prewarm_completed"], 1)
        self.assertEqual(summary["prewarm_uncached_tokens"], 64)
        self.assertEqual(summary["prewarm_followed_by_hit"], 1)

    async def test_unfinished_prewarm_keeps_home_and_releases_reservation(self):
        router, _ = self.router(fake=FakeBackends(tpu_delay=0.1),
                                gap_config=GapConfig(margin_s=0, growth_tokens=16))
        await router.call("w", 0, list(range(64)), 2)
        router.pending["gpu"]["other"] = time.monotonic() + 1
        start = time.monotonic()
        await router.tool_gap("w", 0, list(range(64)), 0.01)
        self.assertLess(time.monotonic() - start, 0.1)
        self.assertEqual(router.home["w"], "gpu")
        self.assertEqual(router.inflight["tpu"], 0)
        self.assertFalse(router.pending["tpu"])
        self.assertEqual(summarize(self.path)["prewarm_unmeasured"], 1)

    async def test_failed_prewarm_does_not_fail_workflow(self):
        router, _ = self.router(fake=FakeBackends(fail_tpu=True),
                                gap_config=GapConfig(margin_s=0, growth_tokens=16))
        await router.call("w", 0, list(range(64)), 2)
        router.pending["gpu"]["other"] = time.monotonic() + 1
        await router.tool_gap("w", 0, list(range(64)), 0.01)
        self.assertEqual(router.home["w"], "gpu")
        self.assertTrue(any(r["kind"] == "prewarm_failure" for r in self.records()))
        await router.call("w", 1, list(range(80)), 2)

    async def test_prewarm_timeout_keeps_home(self):
        router, _ = self.router(fake=FakeBackends(tpu_delay=0.03), gap_config=GapConfig(
            margin_s=0, growth_tokens=16, prewarm_timeout_s=0.05))
        await router.call("w", 0, list(range(64)), 2)
        router.pending["gpu"]["other"] = time.monotonic() + 1
        await router.tool_gap("w", 0, list(range(64)), 0.08)
        self.assertEqual(router.home["w"], "gpu")
        self.assertEqual(router.inflight["tpu"], 0)
        self.assertTrue(any(r["kind"] == "prewarm_failure" and "TimeoutError" in r["error"]
                            for r in self.records()))

    async def test_foreground_error_is_logged_and_compute_is_incomplete(self):
        router, _ = self.router("LB", fake=FakeBackends(fail_tpu=True))
        await router.call("w", 0, list(range(64)), 2)
        router.inflight["gpu"] = 4
        with self.assertRaises(httpx.HTTPStatusError):
            await router.call("w", 1, list(range(80)), 2)
        self.assertEqual(router.inflight["tpu"], 0)
        s = summarize(self.path)
        self.assertEqual(s["failed_call_attempts"], 1)
        self.assertFalse(s["compute_accounting_complete"])

    async def test_global_prewarm_budget_and_no_prewarm_ablation(self):
        router, fake = self.router(gap_config=GapConfig(margin_s=0, growth_tokens=16))
        router.home.update(a="gpu", b="gpu")
        router.pending["gpu"]["other"] = time.monotonic() + 1
        await asyncio.gather(*(router.tool_gap(w, 0, list(range(64)), 0.01) for w in ("a", "b")))
        self.assertEqual(len(fake.requests), 1)
        router.config.max_prewarms = 0
        await router.tool_gap("b", 0, list(range(64)), 0.01)
        self.assertEqual(len(fake.requests), 1)

    async def test_no_oracle_leak_and_zero_gap(self):
        router, fake = self.router(gap_config=GapConfig(margin_s=0, growth_tokens=16, gap_estimate_s=0.7))
        router.home["w"] = "gpu"
        router.pending["gpu"]["other"] = time.monotonic() + 1
        _, decision = router._gap_decision("w", list(range(64)), None)
        self.assertEqual(decision["gap_estimate_s"], 0.7)
        self.assertEqual(decision["next_prompt_estimate"], 80)
        dev, decision = router._gap_decision("w", list(range(64)),
            {"gap_s": 0, "prompt_tokens": 160, "output_tokens": 5})
        self.assertIsNone(dev)
        await router.tool_gap("w", 0, list(range(64)), 0)
        self.assertFalse(fake.requests)

    async def test_short_predicted_gap_and_busy_destination_do_not_prewarm(self):
        router, fake = self.router(gap_config=GapConfig(margin_s=0, gap_estimate_s=0.001))
        router.home["w"] = "gpu"
        router.pending["gpu"]["other"] = time.monotonic() + 1
        dev, decision = router._gap_decision("w", list(range(64)), None)
        self.assertIsNone(dev)
        self.assertEqual(decision["reason"], "gap_too_short")
        router.inflight["tpu"] = 2
        dev, decision = router._gap_decision("w", list(range(64)), None)
        self.assertIsNone(dev)
        self.assertEqual(decision["reason"], "destination_busy_or_capacity")
        router.inflight["tpu"] = 0
        router.profiles.devices["tpu"]["cache_capacity_tokens"] = 32
        self.assertIsNone(router._gap_decision("w", list(range(64)), None)[0])

    async def test_capacity_and_ttl_are_conservative_estimates(self):
        router, _ = self.router()
        router._remember("gpu", "a", 256)
        router._remember("gpu", "b", 256)
        router._remember("gpu", "c", 256)
        self.assertEqual(router.cached("gpu", "a", 256), 0)
        router.config.cache_ttl_s = 0
        self.assertEqual(router.cached("gpu", "c", 256), 0)

    async def test_queue_reservations_do_not_double_count(self):
        router, _ = self.router()
        now = time.monotonic()
        router.pending["gpu"] = {"a": now + 1, "b": now + 2, "c": now + 3}
        self.assertAlmostEqual(router._queue("gpu"), 3, delta=0.05)
        self.assertAlmostEqual(router._queue("gpu", after_s=2), 1, delta=0.05)

    async def test_ttft_ignores_empty_choice_event(self):
        fake = FakeBackends(tpu_delay=0.01)
        client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
        async with client:
            rec = await completion(client, "http://tpu", MODEL, [1] * 64, 1, "w")
        self.assertGreater(rec["ttft_s"], 0.018)

    async def test_truncated_stream_is_failure(self):
        async def handler(request):
            return httpx.Response(200, stream=Stream([{"choices": [{"text": "x"}]}]))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                await completion(client, "http://gpu", MODEL, [1], 1, "w")

    async def test_profiles_match_model_and_lengths(self):
        with self.assertRaises(ValueError):
            Profiles(profiles(), "wrong-model", BACKENDS)
        profile = Profiles(profiles(), MODEL, BACKENDS)
        with self.assertRaisesRegex(ValueError, "Re-profile"):
            profile.validate_lengths([64, 512])
        self.assertLess(profile.service("gpu", 128, 100, 1), profile.service("gpu", 128, 0, 1))
        self.assertGreater(profile.service("gpu", 128, 0, 4), profile.service("gpu", 128, 0, 1))
        data = profiles()
        data["backends"]["gpu"]["url"] = "http://another-path"
        with self.assertRaisesRegex(ValueError, "different URL"):
            Profiles(data, MODEL, BACKENDS)


if __name__ == "__main__":
    unittest.main()
