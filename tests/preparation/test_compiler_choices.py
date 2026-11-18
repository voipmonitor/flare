"""Exact compiler choices must survive symbol renumbering and reject changed contracts."""

from types import SimpleNamespace

import pytest

from ignition.preparation.compiler_choices import choice_key, select_choice
from ignition.preparation.inductor_tuning import configuration_key


def test_fixed_algorithm_does_not_select_an_unrecorded_or_duplicate_candidate():
    key = choice_key("bmm", "shape/stride/dtype", "highest")
    first = SimpleNamespace(hash_key=lambda: "first")
    chosen = SimpleNamespace(hash_key=lambda: "chosen")
    assert select_choice([first, chosen], key, {key: "chosen"}) is chosen
    with pytest.raises(LookupError, match="missing"):
        select_choice([first, chosen], key, {})
    with pytest.raises(LookupError, match="not uniquely available"):
        select_choice([first], key, {key: "chosen"})
    with pytest.raises(LookupError, match="not uniquely available"):
        select_choice([chosen, chosen], key, {key: "chosen"})
    assert key != choice_key("bmm", "different stride", "highest")
    assert key != choice_key("bmm", "shape/stride/dtype", "high")


def test_kernel_configuration_ignores_symbol_and_ordinal_only():
    source = """device = DeviceProperties(index=3, cc=120)
metadata = {'kernel_name': 'triton_sum_7'}
def triton_sum_7(x):
    BLOCK = 128
    return x + BLOCK
"""
    key = configuration_key(source, 3, "triton_sum_7")
    renamed = source.replace("triton_sum_7", "triton_sum_91").replace("index=3", "index=5")
    assert key == configuration_key(renamed, 5, "triton_sum_91")
    for changed in (source.replace("128", "256"), source.replace("x + BLOCK", "x * BLOCK"),
                    source.replace("cc=120", "cc=100")):
        assert key != configuration_key(changed, 3, "triton_sum_7")


@pytest.mark.parametrize('device_index', [3, 5])
def test_registration_returns_source_without_changing_kernel_identity(device_index, monkeypatch):
    from torch._inductor.async_compile import CompiledTritonKernels
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    from ignition.preparation import inductor_tuning

    source = f"""device = DeviceProperties(index={device_index}, cc=120)
metadata = {{'kernel_name': 'triton_sum_7'}}
def triton_sum_7(x):
    return x + 128
"""
    cache_key = CompiledTritonKernels.key(source)
    entry = SimpleNamespace(cache_key=cache_key, kernel_name='triton_sum_7')
    graph = SimpleNamespace(device_idxs=(3,), cache_key='source-wrapper',
        source_code=f'triton_sum_7 = async_compile.triton("triton_sum_7", {source!r})',
        _triton_bundle=SimpleNamespace(static_autotuners=[entry]))
    tuner = CachingAutotuner.__new__(CachingAutotuner)
    tuner.inductor_meta = {'kernel_name': 'triton_sum_7'}
    monkeypatch.setattr(inductor_tuning, '_kernel_globals', lambda graph: {'triton_sum_7': tuner})
    bound = SimpleNamespace(backend=SimpleNamespace(artifact=SimpleNamespace(_flare_graph=graph)))
    selections = inductor_tuning.InductorSelections('cpu')
    if device_index != 3:
        with pytest.raises(ValueError, match='multiple CUDA devices'):
            selections.register(bound)
        return
    sources = selections.register(bound)
    assert sources == {'triton_sum_7': source}
    identity = CompiledTritonKernels.key(inductor_tuning._device_source(source, 3, 0))
    assert selections.configuration_keys == {identity: configuration_key(source, 3, 'triton_sum_7')}
    assert selections.forwards[id(bound)][2] == {cache_key: (identity, entry, tuner)}
