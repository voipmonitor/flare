# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Packed TP16 routing followed by the unchanged prepared expert selector."""

import itertools
import weakref

import torch
from b12x.preparation import PreparedCall

from ignition.components.packed_pair_gather import PreparedPackedPairGather
from ignition.preparation.tuning import declare

_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


class PreparedPackedRouteGather:
    def __init__(self, first, second, correction, latent, logits, weights, ids, *,
                 runtime, workspace, session, coordinator, name, select_plan=None):
        from b12x.comm import pcie

        self.runtime, self.rows = runtime, first.shape[0]
        self.latent, self.logits, self.correction, self.weights, self.ids = latent, logits, correction, weights, ids
        self.gather = PreparedPackedPairGather(first, second, latent, logits, runtime=runtime, workspace=workspace,
            session=session, coordinator=coordinator, name=name + '/gather',
            config=pcie.CollectiveConfig(implementation='dcp_recursive_ready', threads=128,
                                         blocks=8, submode='barrier', active_blocks=8))
        self.plan, self.scratch_nbytes = self.gather.plan, self.gather.scratch_nbytes
        self.select_plan = select_plan
        if select_plan is None:
            arguments = dict(router_logits=logits, correction_bias=correction, output_weights=weights, output_ids=ids)
            self.select_plan = pcie.plan(pcie.query_from_runtime(runtime, surface='DcpAllToAll.kimi_topk16',
                                                                 call=arguments), runtime=runtime)
            declare(session, (self.select_plan.request(name=name + '/select', prepare_call=lambda state:
                PreparedCall(run=lambda: state.run(**arguments), output=weights, owners=(logits, correction, weights, ids))),))
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self):
        self.gather()
        _select(self.logits, self.correction, self.weights, self.ids, self.handle)
        return self.latent


@torch.library.custom_op('flare::kimi3_packed_route_select', mutates_args=('weights', 'ids'))
def _select(logits: torch.Tensor, correction: torch.Tensor, weights: torch.Tensor,
            ids: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.kimi_topk16(logits, correction, weights, ids, plan=owner.select_plan)


@_select.register_fake
def _fake_select(logits, correction, weights, ids, handle):
    pass
