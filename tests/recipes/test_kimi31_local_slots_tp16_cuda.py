# SPDX-License-Identifier: Apache-2.0
"""Replay a complete four-row sparse-attention layer with both local record-selection routes."""

from copy import copy
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.recipes.test_kimi31_verify_tp16_cuda import (
    EPS, HEADS, HIDDEN, PAGE, RECORD, TOPK, WORLD, _features, _weights, _write_reference,
)


class ObservedIndexer:
    """Retain the actual local scores from this layer execution before any cross-rank fold."""

    def __init__(self, selector, device, columns):
        self.selector = selector
        self.ids = torch.full((4, TOPK), -1, dtype=torch.int32, device=device)
        self.scores = torch.full_like(self.ids, -torch.inf, dtype=torch.float32)
        self.selected = torch.full_like(self.ids, -91)
        self.positions = torch.full((4,), -1, dtype=torch.int64, device=device)
        self.tables = torch.full((4, columns), -1, dtype=torch.int32, device=device)

    def _record(self, indices, scores, positions, tables):
        self.ids.copy_(indices)
        self.scores.copy_(scores)
        self.positions.copy_(positions)
        self.tables.copy_(tables)

    def score_local(self, query, weights, positions, tables, *, shared_index):
        result = self.selector.score_local(query, weights, positions, tables, shared_index=shared_index)
        self._record(result[0], result[1], positions, tables)
        return result

    def prepare_local_result(self, query, weights, positions, tables, *, shared_index):
        result = self.selector.prepare_local_result(query, weights, positions, tables, shared_index=shared_index)
        self._record(result[1], result[2], positions, tables)
        return result

    def exchange_result(self, packed, positions, *, dependencies):
        selected, scores = self.selector.exchange_result(packed, positions, dependencies=dependencies)
        self.selected.copy_(selected)
        return selected, scores


class LayerForward:
    def __init__(self, operation, inputs, *, direct, observe=False):
        self.operation = copy(operation)
        self.operation.parts = {4: dict(operation.parts[4])}
        choice = copy(operation.parts[4]["local_slots"])
        choice.enabled = direct
        self.operation.parts[4]["local_slots"] = choice
        self.inputs = inputs
        self.output = torch.zeros_like(inputs.source)
        self.observed = None
        self.raw = None
        self.gathered = None
        self.gate = None
        if observe:
            device = inputs.source.device
            self.observed = (torch.empty((4, TOPK), dtype=torch.int32, device=device),
                torch.empty((4, TOPK), dtype=torch.int32, device=device),
                torch.empty(4, dtype=torch.int32, device=device),
                torch.empty((4, 96, 512), dtype=torch.bfloat16, device=device),
                torch.empty((4, 96), dtype=torch.float32, device=device))
            self.raw = ObservedIndexer(operation.parts[4]["selector"], device, inputs.tables.shape[1])
            self.operation.parts[4]["selector"] = self.raw
            self.gathered = torch.zeros((4, 96, RECORD), dtype=torch.float8_e4m3fn, device=device)
            self.gate = torch.zeros((4, HEADS * 128), dtype=inputs.source.dtype, device=device)

    def __call__(self):
        from ignition.components.paged_batch import Metadata
        from ignition.recipes.kimi31.indexer import SharedIndexMetadata

        data = self.inputs
        metadata = Metadata.pack(data.starts, data.lengths, data.states, data.tables, data.tokens,
            data.live, token_capacity=4, sample_count=8, max_query_tokens=8)
        shared = SharedIndexMetadata.create(metadata, self.operation.shards, 4)
        if self.observed is None:
            self.output.copy_(self.operation(data.source, metadata, shared_index=shared))
        else:
            result = self.operation._run(data.source, metadata, shared_index=shared, observe_selection=False)
            stage = result.last_stage
            for target, value in zip(self.observed, (stage.owners, stage.slots, stage.counts, stage.partial,
                                                    stage.lse), strict=True):
                target.copy_(value)
            self.gathered.copy_(stage.gathered.view_as(self.gathered))
            if self.operation.gated:
                self.gate.copy_(result.projected.gate.view_as(self.gate))
            self.output.copy_(result.output)
        return self.output


class FrozenSlotAttention:
    """Replay both selectors and their actual MLA consumer from the same scored candidate arrays."""

    def __init__(self, source, *, direct):
        self.raw, self.query, self.gate = source.raw, source.gathered, source.gate
        operation = source.operation
        self.operation = operation
        self.projected = SimpleNamespace(gate=self.gate)
        part = operation.parts[4]
        self.local = part["local_slots"] if direct else None
        self.merge = self.raw.selector.merge
        self.shards, self.cache = operation.shards, operation.cache
        self.attention, self.combine, self.starts = part["attention"], part["combine"], part["starts"]
        owners, slots, counts, partial, lse = source.observed
        self.output = (torch.empty_like(owners), torch.empty_like(slots), torch.empty_like(counts),
                       torch.empty_like(partial), torch.empty_like(lse),
                       torch.empty((4, HEADS, 512), dtype=torch.bfloat16, device=self.query.device),
                       torch.empty_like(source.output))

    def __call__(self):
        from ignition.recipes.kimi31.dcp import selection_slots
        data = self.raw
        if self.local is None:
            packed = self.merge.pack(indices=data.ids, scores=data.scores)
            selected = self.merge.exchange(packed=packed, positions=data.positions, dependencies=(self.cache,))
            owners, slots, counts = torch.empty_like(data.ids), torch.empty_like(data.ids), torch.empty_like(data.positions, dtype=torch.int32)
            selection_slots(selected, data.positions, data.tables, owners, slots, counts,
                            self.shards, self.cache.shape[0], local=True)
        else:
            owners, slots, counts = self.local(data.ids, data.scores, data.positions, data.tables,
                                             dependencies=(self.cache,))
        partial, lse = self.attention(self.query, page_table=slots, lengths=counts, starts=self.starts)
        attended = self.combine(partial, lse)
        hidden = self.operation.finish(attended, self.projected)
        for target, value in zip(self.output, (owners, slots, counts, partial, lse, attended, hidden), strict=True):
            target.copy_(value)
        return self.output


@torch.no_grad()
def exercise(rank, rendezvous, output, diagnose=False):
    from ignition import runtime_patches
    runtime_patches.apply()
    import torch.distributed as dist
    from b12x.comm.pcie.pcie_dcp_topk import PCIeDCPTopKOwnerExchange
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime, create_head_exchange
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.dcp import ContextShards
    from ignition.recipes.kimi31.attention import PreparedDSA
    from ignition.recipes.kimi31.dcp import AttentionConstants, ContextParallel
    from ignition.recipes.kimi31.recipe import RECIPE
    from ignition.recipes.kimi31.selection_tuning import CandidateSelectionTuning
    from ignition.preparation.scratch import ScratchArena

    for key, value in RECIPE.b12x_environment.items():
        os.environ.setdefault(key, value)
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD, timeout=timedelta(minutes=15))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    channels, graphs = [], {}
    try:
        with CoreStorage(rank=rank, world=WORLD, device=rank) as storage:
            group = dist.group.WORLD
            coordinator = CollectivePreparation(group)
            heads = create_head_exchange(group, device, max_rows=8, heads=WORLD * HEADS, head_dim=512,
                                         query_head_dim=RECORD, ipc=storage.peer_storage)
            channels.append(heads)
            candidates = create_head_exchange(group, device, max_rows=128, heads=WORLD, head_dim=8192,
                                              ipc=storage.peer_storage)
            channels.append(candidates)
            owner = PCIeDCPTopKOwnerExchange.from_process_group(process_group=group, device=device, max_rows=8,
                topk=TOPK, stream_affine=False, ipc=storage.peer_storage)
            channels.append(owner)
            reduction = create_all_reduce_runtime(group, device, max_tokens=1024, hidden_size=7168,
                                                  ipc=storage.peer_storage)
            channels.append(reduction)
            context = ContextParallel(ContextShards.on(rank, WORLD, device), heads, coordinator, {},
                AttentionConstants.allocate(device), candidates=candidates, candidate_owners=owner)
            registry = CandidateSelectionTuning(context, reduction, cache_root=output.parent / "kernels")
            context = replace(context, selection_tuning=registry)
            pages = 130
            table = ((torch.arange(pages, device=device) * 17 + 3) % pages).int().view(1, -1)
            inputs = SimpleNamespace(source=torch.zeros((4, HIDDEN), device=device, dtype=torch.bfloat16),
                starts=torch.tensor([512], device=device), lengths=torch.tensor([4], dtype=torch.int32, device=device),
                states=torch.ones(1, dtype=torch.int64, device=device), tables=table,
                tokens=torch.arange(8, device=device).view(1, -1), live=torch.ones(1, dtype=torch.int32, device=device))
            cache = torch.zeros((pages, PAGE, RECORD), device=device, dtype=torch.float8_e4m3fn)
            index = torch.zeros((pages * 2, 64 * 132), dtype=torch.uint8, device=device)
            # Four ranks may own a query's new record. Fill the older, interleaved history on every rank.
            local_positions = torch.arange(rank, 262144 + 5, WORLD, device=device)
            history = _features(0, local_positions)
            _write_reference(cache, index, table[0], rank, local_positions, history)
            inputs.source[:, :3] = _features(0, torch.arange(4, device=device) + 512)
            with RecordedPreparationSession(device=device, checkpoint="kimi31-dspark-sparse-verification",
                    workload_schema="four-interleaved-query-rows", numerical_policy="fp32-score-bits-bf16-mla",
                    coordinator=coordinator, compile_workers=24, autotune_effort="default",
                    collective_tuning="static") as session:
                arena = ScratchArena(device)
                session.register_scratch(arena)
                metadata = Metadata.allocate(4, 1, pages, device, samples=8, max_query_tokens=8)
                operation = PreparedDSA(_weights(rank, device), inputs.source, metadata, cache, index_cache=index,
                    eps=EPS, mode="verify", session=session, name="verify", workspace=arena, buffers={}, context=context)
                if rank == 0:
                    print("Preparing sparse attention and communication", flush=True)
                session.tune(show_progress=False)
                selections = registry.tune(session)
                assert len(registry.slot_groups) == 1
                selected = operation.parts[4]["local_slots"].enabled
                if rank == 0:
                    output.with_suffix(".choices.json").write_text(json.dumps(selections, indent=2) + "\n")
                    print("Completed ordinary slot selection", selections[-1]["implementation"], flush=True)
                roles = {name: LayerForward(operation, inputs, direct=direct, observe=observe)
                         for direct, name0 in ((False, "global"), (True, "local"))
                         for observe, name in ((False, name0), (True, name0+"-observed"))}
                roles.update({name: FrozenSlotAttention(roles["global-observed"], direct=direct)
                              for name, direct in (("same-global", False), ("same-local", True))})
                signatures = {name: GraphSignature("speculative", 1, 8, speculation_enabled=True,
                    packed_token_capacity=4, draft_tokens=7) for name in roles}
                graphs, _ = capture_forwards(roles, signatures, session=session, device=device,
                    stream=torch.cuda.Stream(device=device), pool=storage.graph_pool.id, coordinator=coordinator,
                    report_catalog=False, cache_root=output.parent / "kernels", warmup_storage=storage)
                cases = [(0, 0), (15, 1), (512, 4), (2046, 3), (2047, 4), (8192, 2),
                         (32768, 4), (65536, 4), (262144, 4)]
                if diagnose:
                    cases = [(512, 4), (2046, 3), (2047, 4), (32768, 4)]
                times = {}
                mismatches = []
                repeat_differences = []
                for start, length in cases:
                    inputs.starts.fill_(start)
                    inputs.lengths.fill_(length)
                    inputs.source[:, :3] = _features(0, torch.arange(4, device=device) + start, 1)
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    if rank == 7:
                        torch.cuda._sleep(500000)
                    graphs["global-observed"].replay()
                    graphs["local-observed"].replay()
                    torch.cuda.synchronize(device)
                    left, right = roles["global-observed"], roles["local-observed"]
                    failures = []
                    for index_field, (target, reference) in enumerate(zip(
                            (right.output, *right.observed), (left.output, *left.observed), strict=True)):
                        if not torch.equal(target.view(torch.uint8), reference.view(torch.uint8)):
                            failures.append(index_field)
                    if length and torch.equal(left.output[:length], torch.zeros_like(left.output[:length])):
                        failures.append("zero output")
                    rank_status = [None] * WORLD
                    dist.all_gather_object(rank_status, failures)
                    if any(rank_status):
                        repeat_differences.append(dict(start=start, length=length, fields_by_rank=rank_status))
                    samples = {}
                    failures = []
                    for repeat in range(2):
                        for role in ("global-observed", "local-observed"):
                            if repeat:
                                graphs[role].replay()
                                torch.cuda.synchronize(device)
                            layer = roles[role]
                            raw = layer.raw
                            values = dict(indices=raw.ids, words=raw.scores.view(torch.int32),
                                positions=raw.positions, tables=raw.tables, selected=raw.selected,
                                owners=layer.observed[0], slots=layer.observed[1], counts=layer.observed[2],
                                partial=layer.observed[3].view(torch.int16), lse=layer.observed[4].view(torch.int32),
                                output=layer.output.view(torch.int16))
                            samples.update({f"{role}.{repeat}.{key}": value.cpu().numpy().copy()
                                            for key, value in values.items()})
                        for role in ("same-global", "same-local"):
                            graphs[role].replay()
                            torch.cuda.synchronize(device)
                            names = ("owners", "slots", "counts", "partial", "lse", "attended", "output")
                            for field, value in zip(names, roles[role].output, strict=True):
                                if value.dtype == torch.bfloat16:
                                    value = value.view(torch.int16)
                                elif value.dtype == torch.float32:
                                    value = value.view(torch.int32)
                                samples[f"{role}.{repeat}.{field}"] = value.cpu().numpy().copy()
                        for field in names:
                            if not np.array_equal(samples[f"same-global.{repeat}.{field}"],
                                                  samples[f"same-local.{repeat}.{field}"]):
                                failures.append(f"same-producer {repeat} {field}")
                    if length and not np.any(samples["same-global.0.output"][:length]):
                        failures.append("zero hidden output for live rows")
                    np.savez_compressed(output.parent / f"inputs-{start}-{length}-rank{rank}.npz", **samples)
                    dist.all_gather_object(rank_status, failures)
                    if any(rank_status):
                        mismatches.append(dict(start=start, length=length, fields_by_rank=rank_status))
                    if rank == 0:
                        output.with_suffix(".checks.json").write_text(json.dumps(mismatches) + "\n")
                        print("Compared physical slots and complete attention", start, length, rank_status, flush=True)
                    if any(rank_status) or diagnose:
                        continue
                    if (start, length) not in ((512, 4), (32768, 4), (262144, 4)):
                        continue
                    for role in ("global", "local"):
                        for _ in range(3):
                            graphs[role].replay()
                    samples = {"global": [], "local": []}
                    for block in range(9):
                        for role in (("global","local") if block % 2 == 0 else ("local","global")):
                            torch.cuda.synchronize(device)
                            dist.barrier()
                            before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            before.record()
                            for _ in range(12):
                                graphs[role].replay()
                            after.record()
                            after.synchronize()
                            samples[role].append(before.elapsed_time(after) * 1000 / 12)
                    times[start] = samples
                    output.with_name(f"{output.stem}.rank{rank}.timings.json").write_text(json.dumps(times) + "\n")
                all_times = [None] * WORLD
                dist.all_gather_object(all_times, times)
                if rank == 0:
                    output.write_text(json.dumps(dict(selections=selections, ordinary_direct=selected,
                        cases=cases, exact_fields=["owners","slots","counts","partial","lse","attended","layer_output"],
                        mismatches=mismatches, independent_full_replays=repeat_differences,
                        rank_replay_us=all_times), indent=2) + "\n")
                for graph in graphs.values():
                    graph.reset()
                graphs.clear()
            for channel in reversed(channels):
                channel.close()
            channels.clear()
            assert not mismatches, mismatches
    finally:
        for graph in graphs.values():
            graph.reset()
        for channel in reversed(channels):
            channel.close()
        dist.destroy_process_group()


def check_recorded_slots(directory, *, pages):
    """Apply the stable rank-major cutoff and page mapping to each execution's actual FP32 score words."""
    for rank0 in sorted(directory.glob("inputs-*-rank0.npz")):
        paths = [rank0.with_name(rank0.name.replace("rank0", f"rank{rank}")) for rank in range(WORLD)]
        records = []
        for path in paths:
            with np.load(path) as data:
                records.append({key: data[key] for key in data if key.endswith(
                    (".indices", ".words", ".positions", ".tables", ".slots", ".counts"))})
        for role in ("global-observed", "local-observed"):
            for repeat in range(2):
                prefix = f"{role}.{repeat}."
                for row, position in enumerate(records[0][prefix + "positions"]):
                    indices = np.stack([record[prefix+"indices"][row] for record in records])
                    words = np.stack([record[prefix+"words"][row] for record in records]).view(np.uint32)
                    words[indices < 0] = np.uint32(0xff800000)
                    keys = np.where(words >> 31, ~words, words ^ np.uint32(1 << 31))
                    survivors = np.zeros(indices.size, dtype=bool)
                    survivors[np.argsort(~keys.ravel(), kind="stable")[:TOPK]] = True
                    for rank, record in enumerate(records):
                        local = indices[rank, survivors.reshape(WORLD, TOPK)[rank]].astype(np.int64)
                        table = record[prefix+"tables"][row]
                        local = local[(local >= 0) & (local * WORLD + rank <= position) & (local // PAGE < table.size)]
                        physical = table[local // PAGE]
                        kept = (physical >= 0) & (physical < pages)
                        mapped = physical[kept] * PAGE + local[kept] % PAGE
                        expected = np.full(TOPK, -1, np.int32)
                        expected[:mapped.size] = mapped
                        assert record[prefix+"counts"][row] == mapped.size, (rank0, rank, role, repeat, row)
                        assert np.array_equal(record[prefix+"slots"][row], expected), (rank0, rank, role, repeat, row)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1", reason="requires sixteen assigned CUDA devices")
def test_interleaved_slots_through_sparse_attention(tmp_path):
    assert torch.cuda.device_count() == WORLD
    torch.multiprocessing.start_processes(exercise, args=((tmp_path/"rendezvous").as_uri(), tmp_path/"results.json"),
                                          nprocs=WORLD, start_method="fork", join=True)
    assert len(list(tmp_path.glob("inputs-*-rank0.npz"))) == 9
    check_recorded_slots(tmp_path, pages=130)
