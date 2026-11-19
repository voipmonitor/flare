# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from types import SimpleNamespace
import sys

from ignition.recipes.kimi31.slot_tuning import slot_inputs


@pytest.mark.parametrize("available,supported,agreed,expected", (
    (False, True, True, False), (True, False, True, False),
    (True, True, False, False), (True, True, True, True)))
def test_all_ranks_require_the_optional_slot_provider(monkeypatch, available, supported, agreed, expected):
    from ignition.recipes.kimi31 import local_slots

    observed = []
    def agree(value):
        observed.append(value)
        return agreed

    context = SimpleNamespace(candidates=object(), coordinator=SimpleNamespace(same_on_every_rank=agree))
    monkeypatch.setattr(local_slots, "find_spec", lambda _: object() if available else None)
    monkeypatch.setitem(sys.modules, "b12x.comm.pcie.interleaved_topk",
                        SimpleNamespace(supports_channel=lambda _: supported))
    assert local_slots.supports_local_slots(context) == expected
    assert observed == [available and supported]


def test_compiler_metadata_keeps_only_the_recording_rank_handle():
    from ignition.preparation import forward_metadata
    from ignition.recipes.kimi31 import local_slots

    class OnlyOnRecorder:
        def __reduce__(self):
            raise AssertionError("compiler metadata reached a recording rank's provider")

    choice = local_slots.PreparedLocalSlots.__new__(local_slots.PreparedLocalSlots)
    choice.handle, choice.enabled = 173, True
    choice.geometry = local_slots.SlotGeometry(130, 130, 0)
    local_slots._PLANS[choice] = OnlyOnRecorder()
    worker = forward_metadata.loads(forward_metadata.dumps(choice))
    assert (worker.geometry, worker.handle, worker.enabled) == (choice.geometry, 173, True)
    assert worker not in local_slots._PLANS
    assert isinstance(choice.plan, OnlyOnRecorder)


@pytest.mark.parametrize("length", (0, 512, 2048, 8192, 65536, 262144))
@pytest.mark.parametrize("rank", (0, 7, 15))
def test_slot_trials_preserve_the_visible_local_indexer_prefix(rank, length):
    indices, scores, positions = slot_inputs(rank, 401, length)
    assert indices.shape == scores.shape == (4, 2048)
    assert indices.dtype == torch.int32 and scores.dtype == torch.float32
    for row, position in enumerate(positions.tolist()):
        visible = max(0, (position + 16 - rank) // 16)
        capacity = min(visible, 2048)
        assert (indices[row, capacity:] == -1).all()
        assert torch.isneginf(scores[row, capacity:]).all()
        valid = indices[row] >= 0
        assert (indices[row, valid] * 16 + rank <= position).all()
        assert torch.unique(indices[row, valid]).numel() == valid.sum()
    if length > 2048:
        assert torch.isnan(scores).any() and torch.isposinf(scores).any()
        assert (positions[-1] < 0).item()
