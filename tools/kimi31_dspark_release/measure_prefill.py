"""Measure first-token latency for independent uncached 32K completions and verify server counters."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import time
from urllib.request import Request, urlopen

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("endpoint")
p.add_argument("request", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--prefix-ids", type=int, nargs=4, default=[5011, 5021, 5031, 5041])
p.add_argument("--runs", type=int, default=3)
args = p.parse_args()
args.output.mkdir()
template = json.loads(args.request.read_text())["prompt"]
if len(template) != 32768:
    raise ValueError("expected 32768 input token IDs")


def get(path):
    with urlopen(args.endpoint + path, timeout=15) as response:
        return response.read()


def counters():
    names = {
        "flare_replay_gpu_nanoseconds_total",
        "flare_prefill_tokens_total",
        "flare_generated_tokens_total",
        "flare_requests_completed_total",
        "flare_requests_admitted_total",
        "flare_cache_reused_tokens_total",
        "flare_active_requests",
        "flare_pending_requests",
        "flare_requests_failed_total",
        "flare_requests_cancelled_total",
    }
    values = {}
    for line in get("/metrics").decode().splitlines():
        if not line or line.startswith("#"):
            continue
        key, val = line.rsplit(" ", 1)
        if key in names or key.startswith("flare_graph_launches_total{"):
            values[key] = int(val)
    if not names <= values.keys():
        raise ValueError("missing request, cache or replay counters")
    return values


model = json.loads(get("/v1/models"))["data"][0]["id"]
runs = []
for i in range(args.runs):
    prompt = template.copy()
    prompt[:4] = [num + i for num in args.prefix_ids]
    payload = json.dumps(
        dict(
            model=model,
            prompt=prompt,
            max_tokens=1,
            temperature=0,
            seed=17,
            stream=True,
            stream_options={"include_usage": True},
        ),
        separators=(",", ":"),
    ).encode()
    before = counters()
    if before["flare_active_requests"] or before["flare_pending_requests"]:
        raise RuntimeError("another request is using the instance")
    request = Request(
        args.endpoint + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    start = time.monotonic_ns()
    first = None
    usage = None
    done = False
    token = ""
    frames = []
    with urlopen(request, timeout=300) as reply:
        for line in reply:
            now = time.monotonic_ns()
            if not line.startswith(b"data:"):
                continue
            data = line.split(b":", 1)[1].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            frames.append([now - start, event])
            if event.get("error"):
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("text"):
                    if first is None:
                        first = now
                    token += choice["text"]
    elapsed = (time.monotonic_ns() - start) / 1e9
    after = counters()
    delta = {k: after[k] - before[k] for k in before if after[k] != before[k]}
    raw = dict(
        model=model,
        request_sha256=hashlib.sha256(payload).hexdigest(),
        leading_ids=prompt[:4],
        frames=frames,
        before=before,
        after=after,
        delta=delta,
    )
    (args.output / f"run-{i + 1}-raw.json").write_text(json.dumps(raw, indent=2) + "\n")
    if (
        not done
        or not usage
        or first is None
        or usage["prompt_tokens"] != 32768
        or usage["completion_tokens"] != 1
        or usage["prompt_tokens_details"]["cached_tokens"] != 0
    ):
        raise ValueError(f"incomplete or cached stream {usage}")
    expected = dict(
        flare_prefill_tokens_total=32768,
        flare_generated_tokens_total=1,
        flare_cache_reused_tokens_total=0,
        flare_requests_admitted_total=1,
        flare_requests_completed_total=1,
        flare_requests_failed_total=0,
        flare_requests_cancelled_total=0,
    )
    if any(delta.get(k, 0) != v for k, v in expected.items()):
        raise ValueError(f"unexpected request counters {delta}")
    latency = (first - start) / 1e9
    runs.append(
        dict(
            input_tokens=32768,
            output_tokens=1,
            cached_tokens=0,
            output=token,
            ttft=latency,
            http_seconds=elapsed,
            tps=32768 / latency,
            gpu_seconds=delta["flare_replay_gpu_nanoseconds_total"] / 1e9,
            prefill_graphs=delta['flare_graph_launches_total{kind="prefill"}'],
        )
    )
    print(json.dumps(runs[-1]), flush=True)
summary = dict(
    endpoint=args.endpoint,
    model=model,
    runs=runs,
    median_tps=median(r["tps"] for r in runs),
    median_ttft=median(r["ttft"] for r in runs),
)
(args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
