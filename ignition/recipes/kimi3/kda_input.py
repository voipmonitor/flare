# SPDX-License-Identifier: Apache-2.0

"""Compose KDA input projection with its rounded forget-input copy."""

from dataclasses import replace
import itertools
import weakref

import torch

from ignition.components.linear import QuantizedProjection
from ignition.preparation.tuning import register_finalizer

_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


def copy_geometry(source, output, forget, *, mode):
    """Select measured SM120 projection row capacities and the contiguous copy destination."""
    rows = source.shape[0]
    return (mode == 'verify' and rows in (4, 7, 8) and source.shape == (rows, 7168)
            and output.shape == (rows, 3208) and forget.shape == (rows, 128)
            and all(x.dtype == torch.bfloat16 and x.is_contiguous() and x.device == source.device
                    for x in (source, output, forget))
            and source.device.type == 'cuda'
            and torch.cuda.get_device_capability(source.device) == (12, 0))


def prepare_input_copy(projection, source, forget, *, mode, session, workspace):
    if (not isinstance(projection, QuantizedProjection)
            or not copy_geometry(source, projection.output, forget, mode=mode)):
        return None
    return PreparedInputCopy(projection, forget, session=session, workspace=workspace)


class PreparedInputCopy:
    def __init__(self, projection, forget, *, session, workspace):
        self.projection, self.forget = projection, forget
        self.state = self.program = None
        self.enabled = False
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self
        register_finalizer(session, self.bind, workspace=workspace)

    def bind(self):
        projection = self.projection
        self.enabled = False
        if (projection.sm103 or projection.encoded is not None or projection.bias is not None
                or projection.activation_scale is not None):
            return
        state = projection.plan.prepared.state
        query, config = state.query, state.config
        if (query.recipe != 'mxfp8' or query.source_activation != 'identity'
                or query.num_tokens not in (4, 7, 8) or query.in_features != 7168 or query.out_features != 3208
                or config.mode != 'a16' or config.split_k != 4
                or projection.parts[2] is not None):
            return
        from .kda_input_kernel import compile_reduce_copy

        self.program = compile_reduce_copy(torch.cuda.current_device())
        self.state = state
        self.enabled = True

    def run(self, source, values, scales, output, forget, workspace):
        import cutlass
        import cutlass.cute as cute
        from b12x._lib.utils.pointers import make_ptr

        # Functionalization may supply different output views. Bind the copy
        # destination from the actual operation arguments, never a hidden pointer.
        def reduce(partials, destination, rows, stream):
            self.program(partials, destination,
                make_ptr(cutlass.BFloat16, forget.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                rows, stream)

        state = replace(self.state, programs={**self.state.programs, 'reduce': reduce})
        state.run(source, values, scales, None, out=output, workspace=workspace)

    def __call__(self, source):
        projection = self.projection
        values, scales, _ = projection.parts
        if scales.dtype == torch.float8_e8m0fnu:
            scales = scales.view(torch.uint8)
        _project(source, values, scales, projection.output, self.forget, projection.workspace, self.handle)


@torch.library.custom_op('flare::kimi_kda_input_copy', mutates_args=('output', 'forget', 'workspace'))
def _project(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor, output: torch.Tensor,
             forget: torch.Tensor, workspace: torch.Tensor, handle: int) -> None:
    _OWNERS[handle].run(source, values, scales, output, forget, workspace)


@_project.register_fake
def _fake_project(source, values, scales, output, forget, workspace, handle):
    pass
