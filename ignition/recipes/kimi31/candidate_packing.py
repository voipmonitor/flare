# SPDX-License-Identifier: Apache-2.0

"""Pack indexer candidates and derive DCP16 prefix bounds without changing score bits."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_with_prefixes(Indices, ScoreBits, Positions, Rank, Packed, Prefixes,
                       TOPK: tl.constexpr, PREFIX_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    column = tile * BLOCK + tl.arange(0, BLOCK)
    local = tl.load(Indices + row * TOPK + column, column < TOPK, other=-1)
    score = tl.load(ScoreBits + row * TOPK + column, column < TOPK, other=-8388608)
    owner = tl.load(Rank).to(tl.int64)
    identifier = tl.where(local >= 0, local.to(tl.int64) * 16 + owner, -1).to(tl.int32)
    tl.store(Packed + (row * TOPK + column) * 2, tl.where(local >= 0, score, -8388608), column < TOPK)
    tl.store(Packed + (row * TOPK + column) * 2 + 1, identifier, column < TOPK)
    if tile == 0:
        peer = tl.arange(0, 16)
        total = tl.maximum(tl.load(Positions + row) + 1, 0)
        records = tl.minimum(total // 16 + (total % 16 > peer), TOPK)
        words = ((records + 1) // 2 * 4).to(tl.int32)
        tl.store(Prefixes + row * PREFIX_STRIDE + peer, words)
        tl.store(Prefixes + row * PREFIX_STRIDE + 16, tl.min(words, 0) == TOPK * 2)
        tl.store(Prefixes + row * PREFIX_STRIDE + 17, tl.max(words, 0) // 4)


def pack_candidates_with_prefixes(indices, scores, positions, rank, packed, prefixes):
    """Pack top-2048 rows and replicated aligned bounds into caller-owned storage."""
    torch.library.wrap_triton(_pack_with_prefixes)[(indices.shape[0], 4)](
        indices, scores.view(torch.int32), positions, rank, packed.view(torch.int32), prefixes,
        TOPK=2048, PREFIX_STRIDE=prefixes.stride(0), BLOCK=512, num_warps=4)


def use_prefix_exchange(*, mode, rows, world, runtime, session, device):
    """Select the measured four/seven/eight-row TP16 PCIe preparation geometry."""
    return (mode == 'verify' and rows in (4, 7, 8) and world == 16
            and runtime is not None and hasattr(runtime, '_staging_ptrs')
            and getattr(session, 'collective_tuning', 'static') == 'static'
            and runtime._threads_override in (0, 128) and runtime._block_limit_override in (0, 8)
            and torch.cuda.get_device_capability(device) == (12, 0))
