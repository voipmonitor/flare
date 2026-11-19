# SPDX-License-Identifier: Apache-2.0
"""Map reduced attention and MLP outputs to the residual update that consumes them."""

import pytest
from ignition.recipes.kimi3.row_reductions import hidden_row_choices, RowGeometry


def test_layer_consumers_write_the_numbered_block_and_dspark_tap():
    class Catalog:
        def register(self, reduction, geometry):
            return geometry

    taps = (1, 13, 26, 40, 52)
    pairs = hidden_row_choices(Catalog(), [(None, None, None, None)] * 93, 12, taps, 8)
    assert len(pairs) == 93
    assert pairs[0][0] == RowGeometry(7168, sources=1, mode=2, blocks=8)
    assert pairs[0][1] == RowGeometry(7168, sources=1, mode=1, blocks=8, taps=5)
    assert pairs[11][1] == RowGeometry(7168, sources=1, write=1, mode=1, blocks=8)
    assert pairs[12][0] == RowGeometry(7168, sources=2, mode=2, blocks=8)
    assert pairs[-1][1] == RowGeometry(7168, sources=8, mode=1, blocks=8)
    assert sum(g.taps > 0 for _, g in pairs) == len(taps)


@pytest.mark.parametrize("geometry", [RowGeometry(3584), RowGeometry(7168, sources=1, mode=1, blocks=8, taps=5)])
def test_compiler_workers_receive_operands_and_handles_without_cuda_resources(geometry):
    from types import SimpleNamespace
    from cuda.bindings.driver import CUstream
    from ignition.preparation.forward_metadata import dumps, loads
    from ignition.recipes.kimi3.row_reductions import (
        RowReductionChoice,
        _BINDINGS,
        _PreparedTrial,
        _ReductionFixture,
    )
    import torch

    control = SimpleNamespace(handle=41, plan=None, stream=CUstream(0))
    choice = RowReductionChoice(SimpleNamespace(parts=[control]), geometry)
    _BINDINGS[choice].layout = SimpleNamespace(
        owner=SimpleNamespace(device=torch.device("cpu"), rank=0, stream=CUstream(0))
    )
    fixture = _ReductionFixture(choice, 1e-5)
    for fused in (True, False):
        traced = loads(dumps(_PreparedTrial(fixture, fused)))
        assert traced.choice.control.handle == 41
        assert traced.source.shape == (4, geometry.width)
        from torch.fx.experimental.proxy_tensor import make_fx

        graph = make_fx(traced, tracing_mode="fake", _allow_non_fake_inputs=True)()
        names = [str(node.target) for node in graph.graph.nodes if node.op == "call_function"]
        if fused:
            expected = (
                "flare.kimi3_reduce_norm_columns.default"
                if geometry.width == 3584
                else "flare.kimi3_reduce_residual_mix.default"
            )
            assert expected in names and "flare.pcie_all_reduce.default" not in names
        else:
            assert "flare.pcie_all_reduce.default" in names and "triton_kernel_wrapper_mutation" in names
