# SPDX-License-Identifier: Apache-2.0

"""Expose latent-copy mutations in their BF16 tensor views to whole-forward compilation."""

import torch

from .latent_unpack import _unpack


@torch.library.custom_op('flare::kimi_latent_unpack', mutates_args=('output', 'latent_tail'))
def unpack(gathered: torch.Tensor, output: torch.Tensor, latent_tail: torch.Tensor) -> None:
    torch.library.wrap_triton(_unpack)[(gathered.shape[0],)](
        gathered.view(torch.int32), output.view(torch.int32), latent_tail.view(torch.int32),
        SOURCE_WORDS=1088, TAIL_STRIDE=latent_tail.stride(0) // 2, BLOCK=2048, num_warps=4)


@unpack.register_fake
def _fake_unpack(gathered, output, latent_tail):
    pass
