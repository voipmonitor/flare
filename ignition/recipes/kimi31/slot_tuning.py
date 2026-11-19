# SPDX-License-Identifier: Apache-2.0
"""Compare complete sparse record selection after the model has chosen its global-ID merge."""

from dataclasses import asdict
from functools import partial

import torch
from b12x.preparation import PreparedCall

from ignition.components.candidate_selection import PreparedCandidateSelection
from ignition.preparation.capture import capture_forwards
from ignition.preparation.composition_tuning import PreparedComposition, plan_identity, race_compositions
from ignition.preparation.execution import GraphSignature
from ignition.preparation.tuning import RecordedPreparationSession

from .dcp import TOPK, selection_slots
from .indexer import SharedBuffers
from .local_slots import PreparedLocalSlots


_WORKLOAD = ((521, 512), (331, 8192), (709, 65536), (811, 2048))


def slot_inputs(rank, seed, length):
    """Generate legal rank-local indexer output, including reordered winners and inactive rows."""
    positions = torch.tensor([length - 2, length - 1, length, -1], dtype=torch.int64)
    if length == 512:
        positions[-1] = length + 1
    generator = torch.Generator().manual_seed(seed + rank)
    indices = torch.full((4, TOPK), -1, dtype=torch.int32)
    scores = torch.full((4, TOPK), -torch.inf, dtype=torch.float32)
    for row, position in enumerate(positions.tolist()):
        local = max(0, (position + 16 - rank) // 16)
        count = min(local, TOPK)
        indices[row, :count] = torch.randperm(local, generator=generator, dtype=torch.int32)[:count]
        scores[row, :count] = torch.randint(-32, 33, (count,), generator=generator).float()
        if count:
            indices[row, 5:count:83] = -1
            scores[row, 5:count:83] = -torch.inf
            scores[row, 2:count:97] = -torch.inf
            scores[row, 3:count:101] = torch.inf
            scores[row, 7:count:211] = torch.nan
    return indices, scores, positions


class SlotForward:
    """Compare both routes at the same physical-slot output boundary."""

    def __init__(self, trial, direct):
        self.indices, self.scores, self.positions, self.tables = trial.inputs
        self.output = trial.output
        self.merge = trial.merge
        self.direct = trial.local if direct else None
        self.shards = trial.context.shards
        self.cache_pages = trial.geometry.cache_pages

    def __call__(self):
        if self.direct is None:
            packed = self.merge.pack(indices=self.indices, scores=self.scores)
            selected = self.merge.exchange(packed=packed, positions=self.positions)
            selection_slots(selected, self.positions, self.tables, *self.output, self.shards,
                            self.cache_pages, local=True)
        else:
            owners, slots, counts = self.output
            from .local_slots import _select

            _select(self.indices, self.scores, self.positions, self.tables, owners, slots,
                    counts, [], self.direct.handle)
        return self.output


class SlotTrial:
    def __init__(self, context, reduction, geometry, control, session, name):
        self.context, self.geometry, self.control = context, geometry, control
        device = context.shards.index.device
        indices = torch.full((4, TOPK), -1, dtype=torch.int32, device=device)
        scores = torch.full_like(indices, -torch.inf, dtype=torch.float32)
        positions = torch.full((4,), -1, dtype=torch.int64, device=device)
        tables = torch.empty_strided((4, geometry.table_columns), (geometry.table_stride, 1),
                                     device=device, dtype=torch.int32)
        columns = torch.arange(geometry.table_columns, device=device, dtype=torch.int32)
        pages = (columns * 7 + 3) % geometry.cache_pages
        pages[3::11] = -1
        pages[8::13] = geometry.cache_pages
        if geometry.table_stride == 0:
            tables[0].copy_(pages)
        else:
            tables.copy_(pages.expand_as(tables))
        self.inputs = indices, scores, positions, tables
        self.output = torch.empty_like(indices), torch.empty_like(indices), torch.empty_like(positions, dtype=torch.int32)
        self.values = {}
        self.merge = PreparedCandidateSelection(indices, scores, positions, context.shards.index,
            torch.empty_like(indices), world=context.shards.world, gather_runtime=context.candidates,
            owner_runtime=context.candidate_owners, reduction_runtime=reduction, session=session,
            coordinator=context.coordinator, name=name + ".global", buffers=SharedBuffers(self.values, device),
            local_counts=torch.zeros_like(positions, dtype=torch.int32) if control == "owner" else None,
            alternatives=(control,))
        self.merge.bind(control)
        self.merge.functional(session=session)
        self.local = PreparedLocalSlots(*self.inputs, *self.output, cache_pages=geometry.cache_pages,
                                        context=context, session=session, name=name + ".local")
        self.patterns = [tuple(t.to(device) for t in slot_inputs(context.shards.rank, seed, length))
                         for seed, length in _WORKLOAD]
        self.expected = []
        self.index = 0

    def produce(self, index):
        self.index = index
        for dest, src in zip(self.inputs[:3], self.patterns[index], strict=True):
            dest.copy_(src)

    def reset(self):
        for tensor in self.output:
            tensor.fill_(-131)

    def validate(self):
        for observed, expected in zip(self.output, self.expected[self.index], strict=True):
            if not torch.equal(observed, expected):
                raise AssertionError("direct sparse selection changed ordered owners, physical slots or live counts")

    def plans(self, kind):
        return self.merge.plans(self.control) if kind == "global" else (("local-slots", self.local.plan),)

    def resident_bytes(self):
        tensors = [*self.inputs, *self.output, *self.values.values()]
        tensors.extend(tensor for group in (*self.patterns, *self.expected) for tensor in group)
        return sum({tensor.untyped_storage()._cdata: tensor.untyped_storage().nbytes()
                    for tensor in tensors if isinstance(tensor, torch.Tensor)}.values())


def _measure(trial, bindings, session, namespace):
    coordinator = trial.context.coordinator
    device = trial.inputs[0].device
    for local, merge in bindings:
        if (plan_identity(local.plan) != plan_identity(trial.local.plan)
                or [(role, plan_identity(plan)) for role, plan in merge.plans(merge.implementation)]
                != [(role, plan_identity(plan)) for role, plan in trial.plans("global")]):
            raise ValueError("slot trial plans differ from the model's sparse selection")
    forwards = {kind: SlotForward(trial, kind == "local") for kind in ("global", "local")}
    signatures = {kind: GraphSignature("decode", 1, 4) for kind in forwards}
    trial.produce(0)
    graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device,
        stream=torch.cuda.Stream(device=device), pool=torch.cuda.graph_pool_handle(), coordinator=coordinator,
        report_catalog=False, cache_root=namespace, retain_forwards=True)
    try:
        for index in range(len(trial.patterns)):
            trial.produce(index)
            torch.cuda.synchronize(device)
            coordinator.barrier(f"slot reference {index}", coordinator.ranks)
            compiled["global"]()
            torch.cuda.synchronize(device)
            trial.expected.append(tuple(t.clone() for t in trial.output))
        producers = tuple(partial(trial.produce, index) for index in range(len(_WORKLOAD) - 1))
        candidates = {}
        for kind, forward in compiled.items():
            call = PreparedCall(run=forward, output=trial.output[1], produce=producers[0],
                benchmark_producers=producers, reset=trial.reset, validate=trial.validate, owners=(trial,),
                capture_safe=True, benchmark_repetitions=8)
            candidates[kind] = PreparedComposition(forward, trial.plans(kind), call)
        with session._tuning_clocks():
            result = race_compositions(candidates, operation="interleaved-sparse-selection-slots",
                geometry=dict(**asdict(trial.geometry), rows=4, world_size=16, topk=TOPK),
                numerics="ordered-fp32-keys-rank-major-ties-physical-slots", workload=dict(version=1,
                    patterns=_WORKLOAD[:-1]), default="global", session=session, coordinator=coordinator,
                cache_policy="warm", resident_bytes=trial.resident_bytes())
        trial.produce(len(_WORKLOAD) - 1)
        trial.reset()
        torch.cuda.synchronize(device)
        coordinator.barrier("slot selection holdout", coordinator.ranks)
        compiled[result["implementation"]]()
        torch.cuda.synchronize(device)
        trial.validate()
        for local, _ in bindings:
            local.enabled = result["implementation"] == "local"
        return dict(operation="interleaved-sparse-selection-slots", geometry=asdict(trial.geometry),
                    global_merge=trial.control, **result)
    finally:
        for graph in graphs.values():
            graph.reset()
        for forward in compiled.values():
            forward.retire_after_capture()


def tune_local_slots(registry, session):
    namespace = session.namespace.to_dict()
    checkpoint, schema, numerics = (namespace.pop(key) for key in ("checkpoint", "workload_schema", "numerical_policy"))
    with RecordedPreparationSession(device=registry.context.shards.index.device, checkpoint=checkpoint,
            workload_schema=schema, numerical_policy=numerics, namespace=namespace,
            coordinator=registry.context.coordinator, compile_workers=session.requested_compile_workers,
            autotune_effort=session.autotune_effort, collective_tuning="static",
            headroom_bytes=session.headroom_bytes, lock_gpu_clocks=session.lock_gpu_clocks) as trials:
        grouped = {}
        for geometry, bindings in registry.slot_groups.items():
            for local, merge in bindings:
                grouped.setdefault((geometry, merge.implementation), []).append((local, merge))
        fixtures = [(SlotTrial(registry.context, registry.reduction, geometry, control, trials, f"slot_trial.{index}"),
                     bindings) for index, ((geometry, control), bindings) in enumerate(grouped.items())]
        trials.tune(show_progress=False)
        return [_measure(trial, bindings, trials, registry.cache_root) for trial, bindings in fixtures]
