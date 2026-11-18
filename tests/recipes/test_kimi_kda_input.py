"""KDA split-K copying binds after selection and exposes both output mutations."""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.kimi3 import kda_input


def tensor(shape, *, device='cuda:0', dtype=torch.bfloat16, contiguous=True):
    return SimpleNamespace(shape=shape, device=torch.device(device), dtype=dtype,
                           is_contiguous=lambda: contiguous)


@pytest.mark.parametrize('rows,mode,capability,selected', [
    (8, 'verify', (12, 0), True), (7, 'verify', (12, 0), True),
    (4, 'verify', (12, 0), True), (1, 'decode', (12, 0), False),
    (8, 'prefill', (12, 0), False), (8, 'verify', (12, 1), False),
])
def test_copy_geometry(rows, mode, capability, selected, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: capability)
    assert kda_input.copy_geometry(tensor((rows, 7168)), tensor((rows, 3208)), tensor((rows, 128)),
                                   mode=mode) is selected


@pytest.mark.parametrize('change', ['width', 'copy_width', 'dtype', 'stride', 'device'])
def test_other_storage_layouts_keep_the_existing_projection(change, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (12, 0))
    source = tensor((8, 7168))
    output = tensor((8, 3200 if change == 'width' else 3208))
    copied = tensor((8, 64 if change == 'copy_width' else 128),
                    dtype=torch.float16 if change == 'dtype' else torch.bfloat16,
                    contiguous=change != 'stride', device='cuda:1' if change == 'device' else 'cuda:0')
    assert not kda_input.copy_geometry(source, output, copied, mode='verify')


@pytest.mark.parametrize('mode,slices,enabled', [('a16', 4, True), ('a16', 1, False), ('quantized', 4, False)])
@pytest.mark.parametrize('rows', [4, 7, 8])
def test_binding_waits_for_the_selected_projection(mode, slices, enabled, rows, monkeypatch):
    from ignition.recipes.kimi3 import kda_input_kernel

    callbacks = []
    monkeypatch.setattr(kda_input, 'register_finalizer', lambda session, callback, **kw: callbacks.append(callback))
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    compiled = object()
    calls = []
    monkeypatch.setattr(kda_input_kernel, 'compile_reduce_copy', lambda device: calls.append(device) or compiled)
    projection = SimpleNamespace(sm103=False, encoded=None, bias=None, activation_scale=None,
                                 parts=(object(), object(), None), plan=SimpleNamespace())
    operation = kda_input.PreparedInputCopy(projection, tensor((rows, 128)), session=object(), workspace=None)
    assert not operation.enabled and calls == []
    state = SimpleNamespace(query=SimpleNamespace(recipe='mxfp8', source_activation='identity', num_tokens=rows,
                            in_features=7168, out_features=3208), config=SimpleNamespace(mode=mode, split_k=slices))
    projection.plan.prepared = SimpleNamespace(state=state)
    callbacks[0]()
    assert operation.enabled is enabled
    assert calls == ([0] if enabled else [])
    if enabled:
        assert operation.state is state and operation.program is compiled


def test_fake_projection_declares_output_copy_and_workspace_writes():
    from torch.fx.experimental.proxy_tensor import make_fx

    def forward(source, values, scales, output, forget, workspace):
        kda_input._project(source, values, scales, output, forget, workspace, 0)
        return output, forget

    args = [torch.zeros(shape, dtype=dtype) for shape, dtype in (
        ((8, 7168), torch.bfloat16), ((3208, 7168), torch.uint8), ((128,), torch.uint8),
        ((8, 3208), torch.bfloat16), ((8, 128), torch.bfloat16), ((410624,), torch.uint8))]
    graph = make_fx(forward, tracing_mode='fake')(*args)
    operation = next(n for n in graph.graph.nodes if n.target is torch.ops.flare.kimi_kda_input_copy.default)
    inputs = tuple(n for n in graph.graph.nodes if n.op == 'placeholder')
    assert operation.args == (*inputs, 0)
    schema = torch.ops.flare.kimi_kda_input_copy.default._schema
    assert all(a.alias_info is None for a in schema.arguments[:3])
    assert all(a.alias_info.is_write for a in schema.arguments[3:6])


def test_projection_passes_scale_bytes_to_the_compiler():
    from torch.fx.experimental.proxy_tensor import make_fx

    def forward(source, values, scales, output, forget, workspace):
        owner = object.__new__(kda_input.PreparedInputCopy)
        owner.projection = SimpleNamespace(parts=(values, scales, None), output=output, workspace=workspace)
        owner.forget, owner.handle = forget, 0
        owner(source)
        return output, forget

    args = [torch.empty(shape, dtype=dtype) for shape, dtype in (
        ((8, 7168), torch.bfloat16), ((3208, 7168), torch.float8_e4m3fn),
        ((128,), torch.float8_e8m0fnu), ((8, 3208), torch.bfloat16),
        ((8, 128), torch.bfloat16), ((410624,), torch.uint8))]
    graph = make_fx(forward, tracing_mode='fake')(*args)
    operation = next(n for n in graph.graph.nodes if n.target is torch.ops.flare.kimi_kda_input_copy.default)
    scale = operation.args[2]
    inputs = tuple(n for n in graph.graph.nodes if n.op == 'placeholder')
    assert scale.meta['val'].dtype == torch.uint8
    while scale.target is torch.ops.aten.alias.default:
        scale = scale.args[0]
    assert scale.target is torch.ops.aten.view.dtype
    assert scale.args == (inputs[2], torch.uint8)
