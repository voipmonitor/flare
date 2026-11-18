# SPDX-License-Identifier: Apache-2.0
"""Prepare owner-local attention slots from a stable global top-2048 cutoff."""

from importlib.util import find_spec

import torch
import torch.distributed as dist
from b12x.preparation import CollectiveRequirement, PreparedCall, plan_from_handle

from ignition.preparation.tuning import declare


def use_owner_selection(mode, rows, context, device, session):
    """Agree on one slot-selection protocol for the four-row SM120 verification graph."""
    if mode != "verify" or rows != 4 or context is None or context.coordinator is None:
        return False
    available = find_spec("b12x.comm.pcie.interleaved_topk") is not None
    if available:
        from b12x.comm.pcie.interleaved_topk import supports_channel

    runtime = context.candidates
    properties = torch.cuda.get_device_properties(device)
    eligible = (available and supports_channel(runtime)
                and getattr(session, "collective_tuning", "static") == "static"
                and runtime._threads_override in (0, 128) and runtime._block_limit_override in (0, 8)
                and (properties.major, properties.minor, properties.multi_processor_count) == (12, 0, 188))
    agreed = context.coordinator.same_on_every_rank(eligible)
    return eligible and agreed


@torch.library.custom_op("flare::kimi31_owner_selection", mutates_args=("owners", "slots", "counts"))
def _select(ids: torch.Tensor, scores: torch.Tensor, positions: torch.Tensor, tables: torch.Tensor,
            owners: torch.Tensor, slots: torch.Tensor, counts: torch.Tensor,
            dependencies: list[torch.Tensor], handle: int) -> None:
    plan_from_handle(handle).prepared.state.run(ids, scores, positions, tables, owners, slots, counts)


@_select.register_fake
def _fake_select(ids, scores, positions, tables, owners, slots, counts, dependencies, handle):
    pass


class PreparedOwnerSelection:
    def __init__(self, selector, owners, slots, counts, cache, context, session, name):
        from b12x.comm.pcie import interleaved_topk

        self.arguments = (selector.local, selector.scores, selector.positions, selector.tables, owners, slots, counts)
        self.dependencies = [cache]
        runtime = context.candidates
        self.plan = interleaved_topk.plan(runtime, cache_pages=cache.shape[0])
        group = CollectiveRequirement(key=name, ranks=tuple(dist.get_process_group_ranks(runtime.exchange_group)))
        declare(session, (self.plan.request(name=name, collective=group, prepare_call=lambda state:
            PreparedCall(run=lambda: state.run(*self.arguments), output=slots, owners=self.arguments)),),
            coordinator=context.coordinator)

    def __call__(self):
        _select(*self.arguments, self.dependencies, self.plan.handle)
