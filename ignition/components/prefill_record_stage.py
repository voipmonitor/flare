# SPDX-License-Identifier: Apache-2.0

"""Stage an interleaved request's visible KV records for its prefill queries."""

from functools import partial

import torch
from b12x.comm import peer_record_gather
from b12x.preparation import PreparedCall

from ignition.preparation.tuning import benchmark_resource, declare


class PreparedPrefillRecordStage:
    """Reuse one selected-record slab across serial query slices of a single request."""

    def __init__(self, peers, cache, table, tokens, slots, counts, slab, *, session, name):
        rows, topk = tokens.shape
        query = peer_record_gather.InterleavedStageQuery(
            rows=rows, topk=topk, world_size=peers.numel(), page_size=cache.size(1),
            peer_pages=cache.size(0), table_columns=table.numel(),
            capacity=slab.size(0), record_bytes=slab.size(1))
        self.peers = peers
        self.plan = peer_record_gather.plan_interleaved_stage(query, device=cache.device)

        empty_end = torch.zeros((1,), dtype=torch.int64, device=cache.device)

        def prepare(state):
            return PreparedCall(run=lambda: state.run(peers, table, empty_end, tokens, slots, counts, slab, True),
                                owners=(peers, table, empty_end, tokens, slots, counts, slab))

        def fixture():
            # Only the physically backed first page of each owner's declaration cache is used.
            # The repeated page mappings still exercise local and peer reads in the serving layout.
            bench_table = torch.zeros_like(table)
            logical = torch.arange(tokens.numel(), dtype=torch.int32, device=tokens.device).view_as(tokens)
            physical = logical.div(query.world_size, rounding_mode="floor") % query.page_size
            live = torch.full_like(counts, query.topk)
            end = torch.zeros((1,), dtype=torch.int64, device=tokens.device)
            return bench_table, end, logical, physical, live

        def measure(state):
            table_, end, logical, physical, live = benchmark_resource(session, ("interleaved-stage", query), fixture)
            def run():
                state.run(peers, table_, end, logical, physical, live, slab, True)
            # Exercise both address layouts with the actual rank's peer bases. All addresses in
            # these measurements refer to page zero; no request data is overwritten.
            lengths = (min(query.capacity, query.world_size * query.page_size * query.table_columns, 32768),
                       query.capacity + 1)
            producers = tuple(partial(end.fill_, n) for n in lengths)
            return PreparedCall(run=run, produce=producers[0], benchmark_producers=producers,
                                owners=(peers, table_, end, logical, physical, live, slab))

        declare(session, (self.plan.request(name=name, prepare_call=prepare, benchmark_call=measure),))

    def __call__(self, table, end, tokens, slots, counts, slab, *, first):
        peer_record_gather.stage_interleaved_records(
            self.plan, self.peers, table, end, tokens, slots, counts, slab, first=first)
        return slab
