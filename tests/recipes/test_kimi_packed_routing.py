"""Packed routing selects a measured hardware/row band and preserves parallel scratch."""

from types import SimpleNamespace

import pytest

from ignition.recipes.kimi3.decoder import PreparedMLP
from ignition.recipes.kimi3.routing import packed_route_geometry


@pytest.mark.parametrize('rows,world,capability,sms,expected', [
    (1, 16, (12, 0), 188, False), (4, 16, (12, 0), 188, False),
    (6, 16, (12, 0), 188, False), (7, 16, (12, 0), 188, True),
    (8, 16, (12, 0), 188, True), (9, 16, (12, 0), 188, False),
    (8, 8, (12, 0), 188, False), (8, 16, (10, 3), 148, False),
    (8, 16, (12, 1), 40, False), (8, 16, (12, 0), 63, False),
])
def test_packed_transport_preserves_other_row_and_device_paths(rows, world, capability, sms, expected):
    assert packed_route_geometry(rows, world, capability, sms) is expected


@pytest.mark.parametrize('operator_bytes,route_bytes,expected', [(1024, 91392, 91392), (1048576, 91392, 1048576)])
def test_shared_expert_workspace_follows_the_larger_main_branch_requirement(operator_bytes, route_bytes, expected):
    def plan(size):
        return SimpleNamespace(memory_requirements=lambda: SimpleNamespace(scratch_nbytes=size))
    calls = []
    main, side = object(), object()
    operation = PreparedMLP.__new__(PreparedMLP)
    operation.parallel_plans = (plan(operator_bytes),)
    operation.parallel_scratch_nbytes = route_bytes
    operation.gate_up, operation.down = SimpleNamespace(plan=plan(512)), SimpleNamespace(plan=plan(256))
    operation.arena = SimpleNamespace(views=lambda *specs: calls.append(specs) or (main, side))
    operation.refresh()
    assert calls[0][0][0] == (expected,)
    assert calls[0][1][0] == (512,)
    assert operation.workspace is side
