# SPDX-License-Identifier: Apache-2.0

"""Expose BF16 and FP8 query mutations to complete-forward compilation."""

import torch

from .absorbed_query import _assemble


@torch.library.custom_op('flare::kimi_absorbed_query', mutates_args=('output', 'quantized'))
def assemble(projected: torch.Tensor, query: torch.Tensor, output: torch.Tensor,
             quantized: torch.Tensor) -> None:
    rows, heads, _ = query.shape
    torch.library.wrap_triton(_assemble)[(rows * heads,)](
        projected, query, output, quantized.view(torch.uint8),
        ROWS=rows, HEADS=heads, BLOCK=1024, num_warps=4)


@assemble.register_fake
def _fake_assemble(projected, query, output, quantized):
    pass
