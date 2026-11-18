"""KDA overlap retains explicit scratch declarations and joins before recurrence."""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.kimi3 import kda, kda_overlap


@pytest.mark.parametrize('rows,mode,heads,overlap,capability,selected', [
    (8, 'verify', 6, True, (12, 0), True),
    (8, 'verify', 6, False, (12, 0), False),
    (8, 'prefill', 6, True, (12, 0), False),
    (1, 'decode', 6, True, (12, 0), False),
    (4, 'verify', 6, True, (12, 0), True),
    (7, 'verify', 6, True, (12, 0), True),
    (8, 'verify', 12, True, (12, 0), False),
    (8, 'verify', 6, True, (12, 1), False),
])
def test_overlap_declares_both_plans_only_for_the_measured_geometry(
        rows, mode, heads, overlap, capability, selected, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: capability)
    source = SimpleNamespace(shape=(rows, 7168), dtype=torch.bfloat16, device=torch.device('cuda:0'))
    projection, convolution, session = object(), object(), object()
    arena = SimpleNamespace(overlap=overlap)
    calls = []
    monkeypatch.setattr(kda_overlap, 'ConcurrentProjectionChain',
                        lambda projections, **kwargs: calls.append((projections, kwargs)) or object())
    result = kda_overlap.prepare_forget_overlap(source, projection, convolution, mode=mode,
        heads=heads, session=session, workspace=arena)
    assert (result is not None) is selected
    assert calls == ([((projection,), dict(parallel_plans=(convolution,), session=session, workspace=arena))]
                     if selected else [])


@pytest.mark.parametrize('overlap', [False, True])
def test_recurrence_waits_for_both_projection_and_convolution(overlap, monkeypatch):
    operation = kda.PreparedKDA.__new__(kda.PreparedKDA)
    events, pending = [], []
    operation.input_copy = None
    operation.forget = torch.zeros((8, 128), dtype=torch.bfloat16)
    operation.forget_source = torch.ones_like(operation.forget)
    operation.gated = torch.ones((8, 768), dtype=torch.bfloat16)
    operation.records = None
    operation.mode = 'decode'
    operation.options = {}
    operation.convolution = SimpleNamespace(out=object())
    operation.recurrence = object()
    operation.projections = dict(input=lambda x: events.append('input'),
        forget=lambda x: events.append('forget'), output=lambda x: events.append('output') or x)

    class Concurrent:
        def start(self, source):
            assert torch.equal(source, operation.forget_source)
            pending.append(True)
            events.append('start')

        def finish(self, source, dependency):
            assert source is operation.forget and dependency is operation.convolution.out
            assert pending.pop()
            events.append('join')

    operation.forget_overlap = Concurrent() if overlap else None
    monkeypatch.setattr(kda.short_conv, 'run', lambda binding: events.append('convolution'))

    def recurrence(binding, **kwargs):
        assert not pending
        events.append('recurrence')

    monkeypatch.setattr(kda.gdn_decode, 'run_kda', recurrence)
    assert operation(torch.ones(1)) is operation.gated
    assert events == (['input', 'start', 'convolution', 'join', 'recurrence', 'output'] if overlap else
                      ['input', 'forget', 'convolution', 'recurrence', 'output'])
