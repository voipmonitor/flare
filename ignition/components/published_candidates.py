# SPDX-License-Identifier: Apache-2.0

"""Select complete query rows by reading published candidates from their producing ranks."""

import torch

from ignition.components.candidate_rows import PreparedCandidateRows
from ignition.components.collectives import PreparedAllReduce, _all_reduce
from ignition.components.ordered_row_topk import PreparedOrderedRowTopK
from ignition.components.peer_record_gather import PreparedPeerRecordGather
from ignition.components.row_ownership import PreparedGatheredRows
from ignition.components.shared_pages import SharedPages


class PublishedCandidateStorage:
    """One reusable peer-visible score and ID array per rank in a context group."""

    def __init__(self, group, device, rows, topk, *, ipc):
        self.pages = SharedPages(group, device, rows * topk * 8, ipc=ipc)
        self.zero = torch.zeros((1, 8), device=device, dtype=torch.bfloat16)
        self.ready = torch.zeros_like(self.zero)

    def close(self):
        self.pages.close()


@torch.library.custom_op("flare::publish_candidate_rows", mutates_args=("stage", "ready"))
def _publish(packed: torch.Tensor, stage: torch.Tensor, zero: torch.Tensor, ready: torch.Tensor,
             dependencies: list[torch.Tensor], handle: int) -> None:
    # Every reader drains the preceding slice before any publisher reuses its
    # bank. The second group reduction publishes all local copies to their peers.
    _all_reduce(zero, ready, handle)
    stage[:packed.nbytes].copy_(packed.view(torch.uint8).view(-1))
    _all_reduce(zero, ready, handle)


@_publish.register_fake
def _fake_publish(packed, stage, zero, ready, dependencies, handle):
    pass


class PreparedPublishedTopK:
    """Read complete rank-major score/ID rows, then gather their exact selected IDs."""

    def __init__(self, packed, active, rank, output, *, storage, gather_runtime, reduction_runtime,
                 session, coordinator, name, buffers):
        rows, topk = output.shape
        world = gather_runtime.world_size
        owned = -(-rows // world)
        self.stage = storage.pages.tensor
        self.zero, self.ready = storage.zero, storage.ready
        if self.stage.numel() < packed.nbytes:
            raise ValueError("published candidate storage must hold every local score and ID")
        self.fence = PreparedAllReduce(self.zero, self.ready, runtime=reduction_runtime,
            session=session, coordinator=coordinator, name=name+".publication")
        device = output.device
        self.global_rows = torch.arange(owned, device=device) + rank * owned
        self.rows, self.topk, self.world, self.owned = rows, topk, world, owned
        self.sources = torch.arange(world, dtype=torch.int32, device=device).expand(owned, -1).contiguous()
        self.slots = self.global_rows.int()[:, None].expand(-1, world).contiguous()
        self.counts = torch.where(self.global_rows < rows, world, 0).int()
        raw = buffers.zeros((owned, world, topk * 8), torch.uint8)
        self.read = PreparedPeerRecordGather(storage.pages.addresses, self.sources, self.slots,
            self.counts, raw, peer_records=rows, session=session, name=name+".read")
        values = buffers.zeros((owned, world * topk), torch.float32)
        ids = buffers.zeros((owned, world * topk), torch.int32)
        lengths = buffers.zeros((owned,), torch.int32)
        self.candidates = PreparedCandidateRows(raw.view(torch.float32).view(owned, world * topk, 2),
            active[:owned], values, ids, lengths, topk=topk, buffers=buffers)
        selected = buffers.zeros((owned, topk), torch.int32)
        self.fold = PreparedOrderedRowTopK(values, lengths, ids,
            buffers.zeros((owned, topk), torch.float32), selected, session=session, name=name+".fold", buffers=buffers)
        self.replicate = PreparedGatheredRows(selected, output, runtime=gather_runtime,
            session=session, coordinator=coordinator, name=name+".ids", buffers=buffers)

        self.read.functional(session=session)
        self.candidates.functional()
        self.fold.functional(session=session)
        self.replicate.functional(session=session)

    def plans(self):
        return (("publication", self.fence.plan), ("transport", self.read.plan),
                ("selection", self.fold.plan), ("replication", self.replicate.gather.plan))

    def __call__(self, packed, active, *, dependencies):
        _publish(packed, self.stage, self.zero, self.ready, list(dependencies), self.fence.handle)
        raw = self.read(self.sources, self.slots, self.counts, dependencies=(self.stage, self.ready))
        mine = active[self.global_rows.clamp_max(self.rows - 1)] & (self.global_rows < self.rows)
        values, ids, lengths = self.candidates(raw.view(torch.float32).view(self.owned, self.world * self.topk, 2), mine)
        return self.replicate(self.fold(values, lengths, ids))
