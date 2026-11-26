"""One streaming measurement path for profiling, foreground calls and pre-warms."""
import json
import time


async def completion(client, url, model, prompt, max_tokens, wf_id):
    start = time.perf_counter()
    rec = {"t_send": time.time(), "t_first": None}
    usage, finished = {}, False
    first_elapsed = None
    body = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "ignore_eos": True, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with client.stream("POST", f"{url.rstrip('/')}/v1/completions", json=body,
                             headers={"x-session-id": str(wf_id)}) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                finished = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(f"backend stream error: {event['error']}")
            # Role-only, empty-text, finish and usage events are not a first token.
            if first_elapsed is None and any(c.get("text") for c in event.get("choices", [])):
                first_elapsed = time.perf_counter() - start
                rec["t_first"] = time.time()
            if event.get("usage"):
                usage = event["usage"]
    if not finished or first_elapsed is None:
        raise RuntimeError("incomplete completion stream (missing token or [DONE])")
    rec.update(
        t_done=time.time(), ttft_s=first_elapsed, latency_s=time.perf_counter() - start,
        prompt_tokens=usage.get("prompt_tokens"),
        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        output_tokens=usage.get("completion_tokens"),
    )
    return rec
