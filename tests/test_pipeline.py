"""Profiler -> replay -> analysis integration, using the real streaming client.

MockTransport avoids network listeners, so the suite also runs in restricted
development environments. It exercises HTTP messages, not accelerator timing.
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from test_routing import BACKENDS, MODEL, FakeBackends
from analysis.analyze import aggregate, summarize
from replay.replay import load_trace, run_one
from router.profiles import Profiles
from router.router import GapConfig, POLICIES
from scripts.profile_backends import profile


class API(FakeBackends):
    async def handle(self, request):
        if request.method == "GET":
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": MODEL, "max_model_len": 1024}]})
            if request.url.path == "/version":
                return httpx.Response(200, json={"version": "mock"})
            return httpx.Response(200, text="vllm:prefix_cache_hits 0\n")
        return await super().handle(request)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_profile_and_all_policies_then_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fake = API()
            original_client = httpx.AsyncClient

            def client(*args, **kwargs):
                kwargs["transport"] = httpx.MockTransport(fake.handle)
                return original_client(*args, **kwargs)

            a = SimpleNamespace(out=str(tmp / "profiles.json"), lengths=[64, 128, 256],
                output_tokens=4, suffix_tokens=16, reps=2, warmup_reps=1, timeout_s=10,
                gpu_cache_tokens=512, tpu_cache_tokens=512)
            with patch("httpx.AsyncClient", client), patch("scripts.profile_backends.MODEL", MODEL), \
                    patch("scripts.profile_backends.BACKENDS", BACKENDS), \
                    patch("replay.replay.router_mod.MODEL", MODEL), patch("replay.replay.router_mod.BACKENDS", BACKENDS):
                await profile(a)
                measured = json.loads(Path(a.out).read_text())
                self.assertEqual(len(measured["backends"]["gpu"]["samples"]), 6)
                Profiles(measured, MODEL, BACKENDS).validate_lengths([64, 128, 256])
                trace = tmp / "trace.jsonl"
                trace.write_text("\n".join(json.dumps({"wf_id": f"w{i}", "steps": [
                    {"prompt_tokens": 64, "output_tokens": 2, "tool_s": 0.001},
                    {"prompt_tokens": 80, "output_tokens": 2, "tool_s": 0.001},
                    {"prompt_tokens": 96, "output_tokens": 2, "tool_s": 0}]
                }) for i in range(3)))
                wfs = load_trace(trace)
                settings = SimpleNamespace(trace=str(trace), log_dir=str(tmp / "logs"),
                    mode="closed", gap_mode="estimated", gap_scale=1, gap_config=GapConfig(),
                    limit=None, profile_data=measured, ff_threshold=4, timeout_s=10, overwrite=False)
                summaries = []
                for policy in POLICIES:
                    self.assertTrue(await run_one(settings, wfs, policy, 2, 0))
                    path = tmp / "logs" / "trace" / f"{policy}_c2_r0.jsonl"
                    s = summarize(path)
                    self.assertEqual(s["calls"], 9)
                    self.assertEqual(s["wf_completed"], 3)
                    self.assertEqual(s["failed_call_attempts"], 0)
                    self.assertTrue(path.with_suffix(".metrics.json").exists())
                    summaries.append(s)
                self.assertEqual(len(aggregate(summaries)), len(POLICIES))
                self.assertTrue(await run_one(settings, wfs, "CA", 2, 0))
                ca_log = tmp / "logs" / "trace" / "CA_c2_r0.jsonl"
                original_log = ca_log.read_text()
                ca_log.write_text(original_log + "\n")
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    await run_one(settings, wfs, "CA", 2, 0)
                ca_log.write_text(original_log)
                (tmp / "logs" / "trace" / "CA_c2_r0.meta.json").unlink()
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    await run_one(settings, wfs, "CA", 2, 0)

    async def test_invalid_trace_is_rejected_before_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            path.write_text(json.dumps({"wf_id": "bad", "steps": []}))
            with self.assertRaisesRegex(ValueError, "no steps"):
                load_trace(path)


if __name__ == "__main__":
    unittest.main()
