"""Greedy continuations after uncached multi-chunk prefill. Save exact text and token usage."""

import argparse, json
from pathlib import Path
from urllib.request import urlopen, Request

p = argparse.ArgumentParser()
p.add_argument("base")
p.add_argument("request", type=Path)
p.add_argument("out", type=Path)
a = p.parse_args()
a.out.mkdir()
model = json.load(urlopen(a.base + "/v1/models", timeout=10))["data"][0]["id"]
base = json.loads(a.request.read_text())["prompt"]
for length, tail in [(1601, 0), (8192, 1), (32768, 2)]:
    prompt = base[:length].copy()
    prompt[:4] = [4101 + tail, 4202 + tail, 4333 + tail, 4404 + tail]
    body = dict(model=model, prompt=prompt, temperature=0, seed=710 + tail, max_tokens=32, logprobs=1)
    req = Request(
        a.base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=300) as reply:
        v = json.load(reply)
    if v["usage"]["prompt_tokens"] != length or v["usage"]["prompt_tokens_details"]["cached_tokens"] != 0:
        raise ValueError("continuation requires the full uncached prompt")
    (a.out / f"{length}.json").write_text(json.dumps(v, indent=2) + "\n")
    print(length, v["choices"][0]["text"][:70], flush=True)
