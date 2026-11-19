# SPDX-License-Identifier: Apache-2.0
"""Map a stable global top-2048 directly to the records consumed by a DCP rank."""

from dataclasses import dataclass
from importlib.util import find_spec
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist
from b12x.preparation import CollectiveRequirement, PreparedCall, plan_from_handle

from ignition.preparation.tuning import declare


_PLANS = WeakKeyDictionary()


def supports_local_slots(context):
    available = find_spec("b12x.comm.pcie.interleaved_topk") is not None
    if available:
        from b12x.comm.pcie.interleaved_topk import supports_channel

        available = supports_channel(context.candidates)
    agreed = context.coordinator.same_on_every_rank(available)
    return available and agreed


@torch.library.custom_op("flare::kimi31_interleaved_slots", mutates_args=("owners", "slots", "counts"))
def _select(indices: torch.Tensor, scores: torch.Tensor, positions: torch.Tensor, tables: torch.Tensor,
            owners: torch.Tensor, slots: torch.Tensor, counts: torch.Tensor,
            dependencies: list[torch.Tensor], handle: int) -> None:
    plan_from_handle(handle).prepared.state.run(indices, scores, positions, tables, owners, slots, counts)


@_select.register_fake
def _fake(indices, scores, positions, tables, owners, slots, counts, dependencies, handle):
    pass


@dataclass(frozen=True)
class SlotGeometry:
    cache_pages: int
    table_columns: int
    table_stride: int


class PreparedLocalSlots:
    """Bind the same local score words and physical pages used by sparse attention."""

    def __init__(self, indices, scores, positions, tables, owners, slots, counts, *, cache_pages,
                 context, session, name):
        from b12x.comm.pcie import interleaved_topk

        self.geometry = SlotGeometry(cache_pages, tables.shape[1], tables.stride(0))
        plan = interleaved_topk.plan(context.candidates, cache_pages=cache_pages)
        # Compilation workers trace the functional operation by handle. The recording rank owns its IPC program.
        _PLANS[self] = plan
        self.handle = plan.handle
        self.enabled = False
        arguments = indices, scores, positions, tables, owners, slots, counts
        group = CollectiveRequirement(key=name, ranks=tuple(dist.get_process_group_ranks(
            context.candidates.exchange_group)))
        declare(session, (plan.request(name=name, collective=group, prepare_call=lambda state:
            PreparedCall(run=lambda: state.run(*arguments), output=slots, owners=arguments)),),
            coordinator=context.coordinator)

    @property
    def plan(self):
        return _PLANS[self]

    def __call__(self, indices, scores, positions, tables, *, dependencies=()):
        owners, slots = torch.empty_like(indices), torch.empty_like(indices)
        counts = torch.empty((indices.shape[0],), dtype=torch.int32, device=indices.device)
        _select(indices, scores, positions, tables, owners, slots, counts, list(dependencies), self.handle)
        return owners, slots, counts
