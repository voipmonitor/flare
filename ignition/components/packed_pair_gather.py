# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gather two typed column partitions as unchanged bytes in shared operator scratch."""

import itertools
import weakref

import torch
import triton
import triton.language as tl
from b12x.preparation import CollectiveRequirement, PreparedCall

from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import declare, register_finalizer

_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


def packed_pair_geometry(first, second, out_first, out_second, *, world, channel_bytes):
    """Return word counts and the aligned scratch extent without inspecting tensor values."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0] or world < 2:
        raise ValueError('packed column gathering requires equal positive row counts and multiple ranks')
    rows = first.shape[0]
    if (rows <= 0 or first.dtype != torch.bfloat16 or second.dtype != torch.float32
            or out_first.dtype != first.dtype or out_second.dtype != second.dtype
            or out_first.shape != (rows, world * first.shape[1])
            or out_second.shape != (rows, world * second.shape[1])
            or any(value.device != first.device or not value.is_contiguous()
                   for value in (second, out_first, out_second)) or not first.is_contiguous()):
        raise ValueError('packed column gathering requires contiguous BF16/FP32 shards and matching outputs')
    first_bytes, second_bytes = first.shape[1] * 2, second.shape[1] * 4
    if min(first_bytes, second_bytes) <= 0 or first_bytes % 16 or second_bytes % 16:
        raise ValueError('packed column row widths must be positive multiples of sixteen bytes')
    if first_bytes + second_bytes != channel_bytes:
        raise ValueError('packed row bytes differ from the peer channel query dimension')
    spans = [(value.untyped_storage()._cdata, value.storage_offset() * value.element_size(), value.nbytes)
             for value in (first, second, out_first, out_second)]
    for index in (2, 3):
        storage, begin, size = spans[index]
        if any(storage == other and begin < offset + extent and offset < begin + size
               for other, offset, extent in spans[:index]):
            raise ValueError('packed column outputs overlap a live operand')
    local_bytes = rows * channel_bytes
    receive_offset = (local_bytes + 255) // 256 * 256
    return first_bytes // 4, second_bytes // 4, local_bytes, receive_offset, receive_offset + world * local_bytes


@triton.jit
def _pack_words(First, Second, Packed, FIRST: tl.constexpr, SECOND: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    first = tl.load(First + row * FIRST + column, column < FIRST, other=0)
    second = tl.load(Second + row * SECOND + column - FIRST,
                     (column >= FIRST) & (column < FIRST + SECOND), other=0)
    tl.store(Packed + row * (FIRST + SECOND) + column,
             tl.where(column < FIRST, first, second), column < FIRST + SECOND)


@triton.jit
def _unpack_words(Packed, First, Second, FIRST: tl.constexpr, SECOND: tl.constexpr,
                  WORLD: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0)
    row, rank = group // WORLD, group % WORLD
    column = tl.arange(0, BLOCK)
    word = tl.load(Packed + group * (FIRST + SECOND) + column, column < FIRST + SECOND, other=0)
    tl.store(First + row * (WORLD * FIRST) + rank * FIRST + column, word, column < FIRST)
    tl.store(Second + row * (WORLD * SECOND) + rank * SECOND + column - FIRST, word,
             (column >= FIRST) & (column < FIRST + SECOND))


class PreparedPackedPairGather:
    """Pack both fields into E4M3 transport lanes without converting any values.

    The E4M3 dtype describes one-byte transport elements; all packing uses Int32
    loads and stores, including FP32 NaN payloads and signed zero. Scratch is
    live through unpacking and may be reused only after the collective finishes.
    Concurrent work must use storage beyond ``scratch_nbytes``.
    """

    def __init__(self, first, second, out_first, out_second, *, runtime, workspace,
                 session, coordinator, name, config=None):
        from b12x.comm import pcie
        import torch.distributed as dist

        if not isinstance(workspace, ScratchArena):
            raise TypeError('packed column gathering requires a shared operator scratch arena')
        if workspace.device != first.device or runtime.device != first.device:
            raise ValueError('packed columns, scratch, and peer channel must belong to the same device')
        self.geometry = packed_pair_geometry(first, second, out_first, out_second,
            world=runtime.world_size, channel_bytes=runtime.query_head_dim)
        if runtime.heads_per_rank != 1 or first.shape[0] > runtime.max_batch_size:
            raise ValueError('packed column gathering requires one byte head per rank within channel capacity')
        self.first, self.second, self.out_first, self.out_second = first, second, out_first, out_second
        self.runtime, self.workspace = runtime, workspace
        self.rows, self.scratch_nbytes = first.shape[0], self.geometry[-1]
        self.refresh()
        call = dict(local_input=self.send, out=self.received)
        self.plan = pcie.plan(pcie.query_from_runtime(runtime, surface='DcpAllToAll.all_gather_heads', call=call),
                              runtime=runtime, override=config)
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(local_input=self.send, out=self.received), output=self.received,
            owners=(runtime, self.send, self.received)), collective=CollectiveRequirement(key=name, ranks=ranks)),),
            coordinator=coordinator)
        if not workspace.frozen:
            register_finalizer(session, self.refresh, workspace=workspace)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def refresh(self):
        self.scratch, = self.workspace.views(((self.scratch_nbytes,), torch.uint8))
        _, _, local, received, end = self.geometry
        for value in (self.first, self.second, self.out_first, self.out_second):
            if (value.data_ptr() < self.scratch.data_ptr() + end
                    and self.scratch.data_ptr() < value.data_ptr() + value.nbytes):
                raise ValueError('packed column scratch overlaps a live operand')
        width = self.runtime.query_head_dim
        self.send = self.scratch[:local].view(torch.float8_e4m3fn).view(self.rows, 1, width)
        self.received = self.scratch[received:end].view(torch.float8_e4m3fn).view(self.rows, self.runtime.world_size, width)

    def __call__(self):
        _packed_gather(self.first, self.second, self.out_first, self.out_second, self.scratch, self.handle)
        return self.out_first, self.out_second


@torch.library.custom_op('flare::pcie_packed_pair_gather', mutates_args=('out_first', 'out_second', 'scratch'))
def _packed_gather(first: torch.Tensor, second: torch.Tensor, out_first: torch.Tensor,
                   out_second: torch.Tensor, scratch: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    a, b, _, _, _ = owner.geometry
    block = triton.next_power_of_2(a + b)
    torch.library.wrap_triton(_pack_words)[(owner.rows,)](
        first.view(torch.int32), second.view(torch.int32), owner.send.view(torch.int32),
        FIRST=a, SECOND=b, BLOCK=block, num_warps=4)
    owner.runtime.all_gather_heads(owner.send, out=owner.received, plan=owner.plan)
    torch.library.wrap_triton(_unpack_words)[(owner.rows * owner.runtime.world_size,)](
        owner.received.view(torch.int32), out_first.view(torch.int32), out_second.view(torch.int32),
        FIRST=a, SECOND=b, WORLD=owner.runtime.world_size, BLOCK=block, num_warps=4)


@_packed_gather.register_fake
def _fake_packed_gather(first, second, out_first, out_second, scratch, handle):
    pass
