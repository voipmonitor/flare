# SPDX-License-Identifier: Apache-2.0

"""Gather padded score/ID rows using caller-produced, replicated prefix bounds."""

import itertools
import weakref

import torch
import torch.distributed as dist
from b12x.preparation import CollectiveRequirement, PreparedCall

from ignition.preparation.tuning import declare

_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


class PreparedPrefixCandidateGather:
    """Exact all-gather for canonical padding and identical bounds on every rank.

    ``prefixes`` has sixteen aligned word counts, an all-full flag and a maximum
    packet count per row. Its storage remains live until the collective ends.
    Input suffixes must repeat FP32 negative infinity and int32 minus one.
    Operations sharing the peer channel execute serially on every rank.
    """

    def __init__(self, source, output, prefixes, *, runtime, session, coordinator, name, dependencies=()):
        from b12x.comm import pcie
        from b12x.comm.pcie._prefix_gather_preparation import plan, validate_inputs

        self.runtime, self.output, self.prefixes = runtime, output, prefixes
        self.dependencies = list(dependencies)
        if source.ndim != 2 or output.ndim != 2 or prefixes.ndim != 2:
            raise ValueError('prefix candidate gathering requires data and metadata matrices')
        rows = source.shape[0]
        self.source_shape = (rows, runtime.heads_per_rank, runtime.query_head_dim)
        self.output_shape = (rows, runtime.total_heads, runtime.query_head_dim)
        if source.numel() != rows * runtime.heads_per_rank * runtime.query_head_dim:
            raise ValueError('prefix candidate input differs from the peer channel width')
        if output.numel() != rows * runtime.total_heads * runtime.query_head_dim:
            raise ValueError('prefix candidate output differs from the peer channel width')
        local_input, out = source.view(self.source_shape), output.view(self.output_shape)
        query = pcie.query_from_runtime(runtime, surface='DcpAllToAll.all_gather_heads',
                                        call=dict(local_input=local_input, out=out))
        self.plan = plan(query, runtime=runtime, prefix_stride=prefixes.shape[1])
        validate_inputs(self.plan.query, local_input, out, prefixes, source.device)
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(local_input=local_input, out=out, prefixes=prefixes),
            output=out, owners=(runtime, source, output, prefixes)),
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self, source):
        _gather(source, self.output, self.prefixes, self.dependencies, self.handle)
        return self.output


@torch.library.custom_op('flare::pcie_prefix_candidate_gather', mutates_args=('output',))
def _gather(source: torch.Tensor, output: torch.Tensor, prefixes: torch.Tensor,
            dependencies: list[torch.Tensor], handle: int) -> None:
    owner = _OWNERS[handle]
    owner.plan.prepared.state.run(local_input=source.view(owner.source_shape),
                                  out=output.view(owner.output_shape), prefixes=prefixes)


@_gather.register_fake
def _fake_gather(source, output, prefixes, dependencies, handle):
    pass
