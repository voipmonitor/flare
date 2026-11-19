# SPDX-License-Identifier: Apache-2.0
"""Reconstruct a signed parent when a later snapshot reuses one of its paths."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file


@pytest.mark.parametrize("use_overrides", [False, True])
def test_parent_companion_precedence(tmp_path, monkeypatch, use_overrides):
    repo = Path(__file__).resolve().parents[2]
    tool = repo / "tools/kimi31_dspark_release"
    metadata = tool / "metadata"
    monkeypatch.syspath_prepend(str(metadata))
    import package_pb2 as wire

    snapshot = tmp_path / "snapshot"
    original = snapshot / "rebuild/prefill32k-v1"
    replaced = Path("deployments/rank/row-kernels.safetensors")
    weight = Path("weights/shared.safetensors")
    for path, contents in (
        (snapshot / replaced, b"serving program"),
        (original / replaced, b"original program"),
        (snapshot / weight, b"shared weight"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    locations = wire.LocationSet()
    for path in (replaced, weight):
        locations.locations.add(file=path.as_posix(), tensor="flare.payload")
    primary = original / "model.safetensors"
    save_file({"flare.locations": np.frombuffer(locations.SerializeToString(), np.uint8)}, primary)
    out = tmp_path / "parent"
    command = [
        sys.executable,
        str(tool / "parent_view.py"),
        str(snapshot),
        str(primary),
        hashlib.sha256(primary.read_bytes()).hexdigest(),
        str(out),
    ]
    if use_overrides:
        command.extend(["--companion-overrides", str(original)])
    subprocess.run(
        command, check=True, env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (metadata, repo)))}
    )
    assert (out / weight).samefile(snapshot / weight)
    assert (out / replaced).samefile((original if use_overrides else snapshot) / replaced)
    assert (out / "model.safetensors").samefile(primary)
