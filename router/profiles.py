"""Empirical latency model. All times include the measured client/network path."""
import bisect
import json
import math
from pathlib import Path


class Profiles:
    def __init__(self, data, model, backends):
        if isinstance(data, (str, Path)):
            data = json.loads(Path(data).read_text())
        if not data or data.get("version") != 1 or data.get("model") != model:
            raise ValueError("profiles must have version=1 and match the served MODEL")
        self.data = data
        self.devices = data["backends"]
        for dev in backends:
            device = self.devices[dev]
            if device.get("url") and device["url"].rstrip("/") != backends[dev].rstrip("/"):
                raise ValueError(f"{dev} profile was collected on a different URL; re-profile the current client path")
            points = device["points"]
            if not points:
                raise ValueError(f"empty profile for {dev}")
            points.sort(key=lambda p: p["prompt_tokens"])
            if len({p["prompt_tokens"] for p in points}) != len(points):
                raise ValueError(f"duplicate profile lengths for {dev}")
            for p in points:
                for key in ("prompt_tokens", "cold_ttft_s", "warm_ttft_s", "warm_cached_tokens",
                            "decode_s_per_token"):
                    if not math.isfinite(p[key]) or p[key] < 0:
                        raise ValueError(f"invalid {key} in {dev} profile")
                if p["prompt_tokens"] < 1 or not 0 < p["warm_cached_tokens"] < p["prompt_tokens"]:
                    raise ValueError(f"profile for {dev} has no usable measured warm prefix")
            lanes = device.get("queue_parallelism", 1)
            if not math.isfinite(lanes) or lanes < 1 or int(lanes) != lanes:
                raise ValueError("queue_parallelism must be a positive integer")
            capacity = device.get("cache_capacity_tokens")
            if capacity is not None and (not math.isfinite(capacity) or capacity < 1):
                raise ValueError("cache_capacity_tokens must be finite and positive")

    def _at(self, dev, n, key):
        points = self.devices[dev]["points"]
        lengths = [p["prompt_tokens"] for p in points]
        i = bisect.bisect_left(lengths, n)
        if i == 0:
            return points[0][key]
        if i == len(points):
            # Foreground inputs are validated against the measured range by replay.
            return points[-1][key]
        lo, hi = points[i - 1], points[i]
        weight = (n - lo["prompt_tokens"]) / (hi["prompt_tokens"] - lo["prompt_tokens"])
        return lo[key] * (1 - weight) + hi[key] * weight

    def service(self, dev, n, cached, output):
        cold = self._at(dev, n, "cold_ttft_s")
        warm = self._at(dev, n, "warm_ttft_s")
        anchor = self._at(dev, n, "warm_cached_tokens")
        # Interpolate between cold and the observed warm-cache anchor at this total
        # context length; new tokens still attend to the cached context.
        fraction = min(1.0, max(0.0, cached) / max(1.0, anchor))
        ttft = cold + fraction * (warm - cold)
        return max(0.0, ttft) + max(0, output - 1) * self._at(dev, n, "decode_s_per_token")

    def validate_lengths(self, lengths):
        for dev, device in self.devices.items():
            lo, hi = device["points"][0]["prompt_tokens"], device["points"][-1]["prompt_tokens"]
            if min(lengths) < lo or max(lengths) > hi:
                raise ValueError(f"{dev} profile covers {lo}..{hi} tokens; trace needs "
                                 f"{min(lengths)}..{max(lengths)}. Re-profile this range.")
