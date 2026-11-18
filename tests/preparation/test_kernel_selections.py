"""Fixed b12x choices reject misses and preserve exact program/configuration records."""

import json
from types import SimpleNamespace

import pytest
from b12x.preparation._cache import digest
from b12x.preparation.types import FrozenMapping

from ignition.preparation.kernel_selections import FixedSelectionCache


def table(tmp_path):
    identity = dict(schema_version=23, tuning_cache_version=1, namespace={"checkpoint": "example"},
                    compute_capability=[12, 0], sm_count=188)
    record = dict(assignment={"tile": 128}, config={"tile": 128}, coverage={},
                  programs=[["cute", "source-program-key"]])
    path = tmp_path / (digest(identity) + ".json")
    path.write_text(json.dumps(dict(identity=identity, records={"query-key": record})))
    return identity, record, path


def test_fixed_choices_need_no_compiled_artifacts(tmp_path):
    identity, record, path = table(tmp_path)
    before = path.read_bytes()
    cache = FixedSelectionCache(tmp_path, identity)
    assert cache.get("query-key").to_dict() == record
    cache.flush()
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_fixed_choices_reject_missing_programs_and_updates(tmp_path):
    identity, _, path = table(tmp_path)
    cache = FixedSelectionCache(tmp_path, identity)
    with pytest.raises(LookupError, match="automatic retuning is disabled"):
        cache.get("different-query-or-kernel-source")
    with pytest.raises(RuntimeError, match="cannot be updated"):
        cache.save("query-key")
    path.unlink()
    with pytest.raises(FileNotFoundError, match="workload/device"):
        FixedSelectionCache(tmp_path, identity)


def test_ranks_cannot_merge_different_fixed_choices(tmp_path):
    identity, _, _ = table(tmp_path)
    cache = FixedSelectionCache(tmp_path, identity)
    same = SimpleNamespace(identity=FrozenMapping(identity), records=FrozenMapping(cache.records))
    cache.reconcile((same, same))
    assert cache.get("query-key").to_dict() == cache.records["query-key"]
    different = SimpleNamespace(identity=FrozenMapping(identity), records=FrozenMapping({}))
    with pytest.raises(ValueError, match="identical fixed"):
        cache.reconcile((same, different))
