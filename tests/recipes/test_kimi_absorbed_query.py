"""Absorbed-query fusion preserves both output representations and its shape boundary."""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.kimi3.absorbed_query import assemble_query, fuse_query_quantization


@pytest.mark.parametrize('rows,mode,heads,architecture,selected', [
    (8, 'verify', 6, (12, 0), True),
    (1, 'decode', 6, (12, 0), False),
    (4, 'verify', 6, (12, 0), False),
    (7, 'verify', 6, (12, 0), False),
    (16, 'verify', 6, (12, 0), False),
    (8, 'prefill', 6, (12, 0), False),
    (8, 'verify', 12, (12, 0), False),
    (8, 'verify', 6, (12, 1), False),
])
def test_preparation_selection(rows, mode, heads, architecture, selected, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: architecture)
    query = SimpleNamespace(shape=(rows, heads, 192), dtype=torch.bfloat16,
                            device=torch.device('cuda:0'), is_contiguous=lambda: True)
    assert fuse_query_quantization(query, mode=mode) is selected


@pytest.mark.parametrize('change', ['dtype', 'device', 'stride', 'width'])
def test_incompatible_query_storage_is_not_selected(change, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (12, 0))
    query = SimpleNamespace(shape=(8, 6, 128 if change == 'width' else 192),
        dtype=torch.float16 if change == 'dtype' else torch.bfloat16,
        device=torch.device('cpu' if change == 'device' else 'cuda:0'),
        is_contiguous=lambda: change != 'stride')
    assert not fuse_query_quantization(query, mode='verify')


def test_fx_operator_declares_both_output_mutations():
    from torch.fx.experimental.proxy_tensor import make_fx

    args = [torch.zeros(shape, dtype=dtype) for shape, dtype in (
        ((6, 8, 512), torch.bfloat16), ((8, 6, 192), torch.bfloat16),
        ((8, 6, 576), torch.bfloat16), ((8, 6, 576), torch.float8_e4m3fn))]
    graph = make_fx(assemble_query, tracing_mode='fake')(*args)
    operation = next(n for n in graph.graph.nodes if n.target is torch.ops.flare.kimi_absorbed_query.default)
    assert operation.args == tuple(n for n in graph.graph.nodes if n.op == 'placeholder')
    arguments = torch.ops.flare.kimi_absorbed_query.default._schema.arguments
    assert all(a.alias_info is None for a in arguments[:2])
    assert all(a.alias_info.is_write for a in arguments[2:])


@pytest.mark.parametrize('change', ['projected_shape', 'query_stride', 'output_shape', 'quantized_dtype'])
def test_invalid_buffers_rejected_before_launch(change):
    projected = torch.empty((6, 7 if change == 'projected_shape' else 8, 512), dtype=torch.bfloat16)
    query = torch.empty((8, 6, 384 if change == 'query_stride' else 192), dtype=torch.bfloat16)
    if change == 'query_stride':
        query = query[:, :, ::2]
    output = torch.empty((8, 6, 512 if change == 'output_shape' else 576), dtype=torch.bfloat16)
    quantized = torch.empty((8, 6, 576), dtype=torch.bfloat16 if change == 'quantized_dtype' else torch.float8_e4m3fn)
    with pytest.raises(ValueError, match='absorbed query requires'):
        assemble_query(projected, query, output, quantized)
