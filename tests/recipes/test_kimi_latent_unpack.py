"""Select latent-copy fusion only for the measured geometry."""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.kimi3.latent_unpack import select_unpack, unpack_latents


@pytest.mark.parametrize('rows,selected', [(1, False), (4, False), (7, False), (8, True), (16, False)])
def test_row_capacity_selection(rows, selected, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (12, 0))
    source = SimpleNamespace(shape=(rows, 7168), dtype=torch.bfloat16, device=torch.device('cuda:0'))
    result = select_unpack(source, mode='verify', query_width=1536, world=16, shard_width=136)
    assert (result is unpack_latents) is selected


@pytest.mark.parametrize('changed', ['prefill', 'query', 'world', 'padding', 'dtype', 'architecture', 'cpu'])
def test_other_projection_geometries_keep_existing_copies(changed, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (12, 1) if changed == 'architecture' else (12, 0))
    source = SimpleNamespace(shape=(8, 7168), dtype=torch.float16 if changed == 'dtype' else torch.bfloat16,
                             device=torch.device('cpu' if changed == 'cpu' else 'cuda:0'))
    assert select_unpack(source, mode='prefill' if changed == 'prefill' else 'verify',
        query_width=1024 if changed == 'query' else 1536,
        world=8 if changed == 'world' else 16, shard_width=132 if changed == 'padding' else 136) is None


def test_fx_copy_operator_exposes_bf16_output_and_tail_mutations():
    from torch.fx.experimental.proxy_tensor import make_fx

    def forward(gathered, output, latent):
        unpack_latents(gathered, output, latent[:, 512:])
        return output, latent

    args = [torch.zeros(shape, dtype=torch.bfloat16) for shape in ((8, 2176), (8, 2112), (8, 576))]
    graph = make_fx(forward, tracing_mode='fake')(*args)
    operation = next(node for node in graph.graph.nodes if node.target is torch.ops.flare.kimi_latent_unpack.default)
    inputs = [node for node in graph.graph.nodes if node.op == 'placeholder']
    assert operation.args[:2] == (inputs[0], inputs[1])
    tail = operation.args[2]
    assert tail.target is torch.ops.aten.slice.Tensor
    assert tail.args[1:4] == (1, 512, 9223372036854775807)
    schema = torch.ops.flare.kimi_latent_unpack.default._schema
    assert schema.arguments[0].alias_info is None
    assert schema.arguments[1].alias_info.is_write
    assert schema.arguments[2].alias_info.is_write
