"""Qualification boundaries for prefix-bounded sparse candidate exchange."""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.kimi31.candidate_packing import use_prefix_exchange


@pytest.mark.parametrize('rows,expected', [(1, False), (2, False), (4, True), (7, True), (8, True), (16, False)])
def test_prefix_selection_keeps_other_target_capacities(rows, expected, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (12, 0))
    runtime = SimpleNamespace(_staging_ptrs=(), _threads_override=0, _block_limit_override=0)
    assert use_prefix_exchange(mode='verify', rows=rows, world=16, runtime=runtime,
                               session=None, device=torch.device('cuda:0')) is expected


@pytest.mark.parametrize('change', ['prefill', 'world', 'device', 'runtime_tuning', 'threads', 'blocks', 'channel'])
def test_prefix_selection_preserves_explicit_policy_and_other_geometries(change, monkeypatch):
    capability = (12, 1) if change == 'device' else (12, 0)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: capability)
    runtime = SimpleNamespace(_staging_ptrs=(), _threads_override=512 if change == 'threads' else 0,
                              _block_limit_override=16 if change == 'blocks' else 0)
    if change == 'channel':
        del runtime._staging_ptrs
    session = SimpleNamespace(collective_tuning='runtime' if change == 'runtime_tuning' else 'static')
    assert not use_prefix_exchange(mode='prefill' if change == 'prefill' else 'verify', rows=8,
        world=8 if change == 'world' else 16, runtime=runtime, session=session, device=torch.device('cuda:0'))


def test_prefix_selection_accepts_matching_explicit_launch_geometry(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (12, 0))
    runtime = SimpleNamespace(_staging_ptrs=(), _threads_override=128, _block_limit_override=8)
    assert use_prefix_exchange(mode='verify', rows=8, world=16, runtime=runtime,
                               session=None, device=torch.device('cuda:0'))


def test_fx_prefix_exchange_reads_bounds_and_preserves_query_join():
    from torch.fx.experimental.proxy_tensor import make_fx

    from ignition.components import collectives
    from ignition.components.prefix_candidate_gather import _gather

    def forward(source, gathered_query, prefixes, packed, output):
        collectives._start_all_gather(source, gathered_query, [], 1, 2)
        prefixes.fill_(0)
        packed.copy_(source)
        collectives._join_all_gather(source, gathered_query, packed, 2)
        _gather(packed, output, prefixes, [gathered_query], 3)
        prefixes.fill_(1)
        return output

    module = make_fx(forward, tracing_mode='fake')(*(torch.ones(4) for _ in range(5)))
    inputs = [node for node in module.graph.nodes if node.op == 'placeholder']
    exchange = next(node for node in module.graph.nodes
                    if node.target is torch.ops.flare.pcie_prefix_candidate_gather.default)
    prefix_write = exchange.args[2]
    assert prefix_write.target is torch.ops.aten.fill_.Scalar
    assert prefix_write.args[0] is inputs[2]
    assert exchange.args[3] == [inputs[1]]
    schema = torch.ops.flare.pcie_prefix_candidate_gather.default._schema
    assert schema.arguments[1].alias_info.is_write
    assert schema.arguments[2].alias_info is None
