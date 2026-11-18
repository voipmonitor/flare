"""Export-stage recovery preserves immutable payloads after ordinary cleanup."""

import json
from pathlib import Path
import shutil

import pytest

from tools.kimi31_export_stage import restore_stage, save_stage


def test_stage_survives_cleanup_with_absolute_payload_references(tmp_path):
    stage = tmp_path / ".flare-export-owned"
    stage.mkdir()
    payload = stage / "graph.bin"
    payload.write_bytes(b"captured graph")
    rank = stage / "rank-0.json"
    rank.write_text(json.dumps({"payload": str(payload)}))
    checkpoint = tmp_path / "retained"
    peers = [{"owner": 0, "address": 4096, "nbytes": 8192, "readers": [0, 1]}]
    save_stage([rank], peers, checkpoint, cache=tmp_path / "cache", no_weights=True, form="directory")
    assert (checkpoint / "stage/graph.bin").stat().st_ino == payload.stat().st_ino
    shutil.rmtree(stage)
    ranks, record = restore_stage(checkpoint)
    assert ranks == [rank]
    assert record["peers"] == peers
    restored = Path(json.loads(rank.read_text())["payload"])
    assert restored.read_bytes() == b"captured graph"
    assert restored.stat().st_ino == (checkpoint / "stage/graph.bin").stat().st_ino


def test_restore_preserves_an_existing_stage(tmp_path):
    stage = tmp_path / ".flare-export-owned"
    stage.mkdir()
    rank = stage / "rank-0.json"
    rank.write_text("{}")
    checkpoint = tmp_path / "retained"
    save_stage([rank], [], checkpoint, cache=tmp_path / "cache", no_weights=False, form="directory")
    rank.write_text('{"live": true}')
    with pytest.raises(FileExistsError):
        restore_stage(checkpoint)
    assert rank.read_text() == '{"live": true}'


def test_rank_sources_must_share_one_stage(tmp_path):
    with pytest.raises(ValueError, match="one staging directory"):
        save_stage([tmp_path / "a/rank-0.json", tmp_path / "b/rank-1.json"], [],
                   tmp_path / "retained", cache=tmp_path / "cache", no_weights=False, form="directory")
