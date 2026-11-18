"""Nested fixed-selection scopes must preserve extracted source for their caller."""

import json
from types import SimpleNamespace

from ignition.preparation import inductor_tuning
from tools import kimi31_fixed_compiler


def test_nested_fixed_scopes_return_identical_kernel_source(tmp_path, monkeypatch):
    from torch._inductor.async_compile import CompiledTritonKernels
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    source = """device = DeviceProperties(index=3, cc=120)
metadata = {'kernel_name': 'triton_sum_7'}
def triton_sum_7(x):
    return x + 128
"""
    cache_key = CompiledTritonKernels.key(source)
    entry = SimpleNamespace(cache_key=cache_key, kernel_name="triton_sum_7")
    graph = SimpleNamespace(device_idxs=(3,), cache_key="source-wrapper",
        source_code=f'triton_sum_7 = async_compile.triton("triton_sum_7", {source!r})',
        _triton_bundle=SimpleNamespace(static_autotuners=[entry]))
    tuner = CachingAutotuner.__new__(CachingAutotuner)
    tuner.inductor_meta = {"kernel_name": "triton_sum_7"}
    tuner.configs = [SimpleNamespace(kwargs={"XBLOCK": 128}, num_warps=4)]
    monkeypatch.setattr(inductor_tuning, "_kernel_globals", lambda _: {"triton_sum_7": tuner})
    monkeypatch.setattr(kimi31_fixed_compiler, "_configuration_device", lambda _: {"test": "cpu"})
    installed = []
    monkeypatch.setattr(kimi31_fixed_compiler, "_install_configuration",
                        lambda kernel, state, text: installed.append((kernel, state, text)))
    (tmp_path / "algorithms.json").write_text(json.dumps({"schema": 1, "choices": {}}))
    (tmp_path / "inductor.json").write_text(json.dumps(
        {"schema": 2, "device": {"test": "cpu"}, "kernels": {}}))
    bound = SimpleNamespace(backend=SimpleNamespace(artifact=SimpleNamespace(_flare_graph=graph)))
    original_register = inductor_tuning.InductorSelections.register
    with kimi31_fixed_compiler.compiler_policy(tmp_path):
        with kimi31_fixed_compiler.compiler_policy(tmp_path):
            selections = inductor_tuning.InductorSelections("cpu")
            assert selections.register(bound) == {"triton_sum_7": source}
            assert bound.backend._tuning_frozen
    assert len(installed) == 2
    assert all(kernel is tuner and text == source for kernel, _, text in installed)
    assert inductor_tuning.InductorSelections.register is original_register
