# SPDX-License-Identifier: Apache-2.0

"""Prepare the independent KDA forget projection beside short convolution."""

import torch

from ignition.components.projection_overlap import ConcurrentProjectionChain


def prepare_forget_overlap(source, projection, convolution, *, mode, heads, session, workspace):
    if (mode != 'verify' or source.shape[0] not in (4, 7, 8) or source.shape[1] != 7168 or heads != 6
            or source.dtype != torch.bfloat16 or source.device.type != 'cuda'
            or workspace is None or not workspace.overlap
            or torch.cuda.get_device_capability(source.device) != (12, 0)):
        return None
    return ConcurrentProjectionChain((projection,), parallel_plans=(convolution,),
                                     session=session, workspace=workspace)
