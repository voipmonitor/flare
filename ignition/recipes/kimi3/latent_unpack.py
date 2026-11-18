# SPDX-License-Identifier: Apache-2.0

"""Unpack query/KV latent shards and copy the unnormalized KV tail without floating-point arithmetic."""

import torch
import triton
import triton.language as tl


@triton.jit
def _unpack(Source, Input, Tail, SOURCE_WORDS: tl.constexpr, TAIL_STRIDE: tl.constexpr,
            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    shard = SOURCE_WORDS // 16
    q_source = column // 48 * shard + column % 48
    kv_column = column - 768
    kv_source = kv_column // 18 * shard + 48 + kv_column % 18
    source_column = tl.where(column < 768, q_source, kv_source)
    words = tl.load(Source + row * SOURCE_WORDS + source_column, column < 1056, other=0)
    tl.store(Input + row * 1056 + column, words, column < 1056)
    tl.store(Tail + row * TAIL_STRIDE + column - 1024, words, (column >= 1024) & (column < 1056))


def unpack_latents(gathered, output, latent_tail):
    """Restore 1,536 query and 576 KV values, including the 64-value unnormalized tail."""
    rows = gathered.shape[0]
    if (gathered.shape != (rows, 2176) or output.shape != (rows, 2112)
            or latent_tail.shape != (rows, 64)
            or any(x.dtype != torch.bfloat16 or x.device != gathered.device
                   for x in (gathered, output, latent_tail))
            or not gathered.is_contiguous() or not output.is_contiguous()
            or latent_tail.stride(1) != 1 or latent_tail.stride(0) % 2):
        raise ValueError('latent unpack requires BF16 rows: gathered width 2176, input width 2112 and tail width 64')
    from .latent_unpack_op import unpack

    unpack(gathered, output, latent_tail)


def select_unpack(source, *, mode, query_width, world, shard_width):
    """Select the measured eight-row SM120 latent-copy geometry during preparation."""
    return (unpack_latents if mode == 'verify' and source.shape[0] == 8 and query_width == 1536
            and world == 16 and shard_width == 136 and source.dtype == torch.bfloat16
            and source.device.type == 'cuda'
            and torch.cuda.get_device_capability(source.device) == (12, 0) else None)
