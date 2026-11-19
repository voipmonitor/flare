# SPDX-License-Identifier: Apache-2.0
"""Measure sparse prefill KV delivery over 16 peer-mapped GPUs without checkpoint weights."""

import argparse
import json
from datetime import timedelta
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist

from b12x.comm import peer_record_gather
from b12x.preparation import PreparationSession, PreparedCall
from ignition.components.shared_pages import SharedPages
from ignition.preparation.core_storage import CoreStorage


def measure_rank(rank, directory):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    world, rows, topk, chunks = 16, 128, 2048, 16
    page, pages, width = 128, 132, 576
    capacity = rows * topk
    dist.init_process_group("gloo", rank=rank, world_size=world,
        init_method=(directory / "gloo").as_uri(), timeout=timedelta(minutes=10))
    try:
        with CoreStorage(rank=rank, world=world, device=rank) as storage:
            shared = SharedPages(dist.group.WORLD, device, pages * page * width, ipc=storage.peer_storage)
            try:
                physical = torch.arange(pages * page, dtype=torch.int32, device=device)[:, None]
                channels = torch.arange(width, dtype=torch.int32, device=device)
                shared.tensor.view(pages * page, width).copy_((physical * 17 + channels * 7 + rank * 11) % 251)
                torch.cuda.synchronize(device)
                dist.barrier()
                table = torch.arange(pages - 1, -1, -1, dtype=torch.int32, device=device)
                ids = torch.empty((chunks, rows, topk), dtype=torch.int32, device=device)
                owners, slots = torch.empty_like(ids), torch.empty_like(ids)
                end = torch.zeros((1,), dtype=torch.int64, device=device)
                counts = torch.full((rows,), topk, device=device, dtype=torch.int32)
                staged = torch.empty((capacity, width), dtype=torch.uint8, device=device)
                gathered = torch.empty_like(staged).view(rows, topk, width)

                def populate(length):
                    generator = torch.Generator(device=device).manual_seed(length)
                    ids.copy_((torch.rand(ids.shape, generator=generator, device=device) * length).int())
                    local = ids // world
                    owners.copy_(ids % world)
                    slots.copy_(table[(local // page).long()] * page + local % page)
                    end.fill_(length)

                populate(32768)
                stage = peer_record_gather.plan_interleaved_stage(peer_record_gather.InterleavedStageQuery(
                    rows=rows, topk=topk, world_size=world, page_size=page, table_columns=pages,
                    peer_pages=pages, capacity=capacity, record_bytes=width), device=device)
                direct = peer_record_gather.plan(peer_record_gather.PeerRecordGatherQuery(
                    rows=rows, topk=topk, world_size=world, peer_records=pages * page,
                    record_bytes=width), device=device)

                def stage_call(state):
                    return PreparedCall(run=lambda: state.run(shared.addresses, table, end, ids[0],
                        slots[0], counts, staged, True), produce=lambda: end.fill_(32768), owners=(shared,))

                def direct_call(state):
                    return PreparedCall(run=lambda: state.run(shared.addresses, owners[0], slots[0],
                        counts, gathered), produce=lambda: end.fill_(32768), owners=(shared,))

                with PreparationSession(device=device, autotune=True, compile_workers=0,
                        cache_dir=directory / f"selections-{rank}") as session:
                    session.prepare((stage.request(name="stage", prepare_call=stage_call, benchmark_call=stage_call),
                                     direct.request(name="direct", prepare_call=direct_call, benchmark_call=direct_call)))
                    graphs = {}
                    for name in ("direct", "stage"):
                        graphs[name] = torch.cuda.CUDAGraph()
                        with session.capture(), torch.cuda.graph(graphs[name]):
                            for index in range(chunks):
                                if name == "direct":
                                    peer_record_gather.gather(direct, shared.addresses, owners[index],
                                                              slots[index], counts, gathered)
                                else:
                                    peer_record_gather.stage_interleaved_records(stage, shared.addresses, table, end,
                                        ids[index], slots[index], counts, staged, first=index == 0)

                    results = {}
                    for length in (32768, capacity + 1):
                        populate(length)
                        torch.cuda.synchronize(device)
                        dist.barrier()
                        for graph in graphs.values():
                            graph.replay()
                        # Check the last slice against the independently generated bytes on its physical owner.
                        sample = torch.arange(0, capacity, 53, dtype=torch.long, device=device)
                        offsets = ids[-1].flatten()[sample].long() if length <= capacity else sample
                        expected = ((slots[-1].flatten()[sample, None] * 17 + channels * 7
                                     + owners[-1].flatten()[sample, None] * 11) % 251).to(torch.uint8)
                        if not (torch.equal(gathered.view(capacity, width)[sample], expected)
                                and torch.equal(staged[offsets], expected)):
                            raise ValueError(f"selected KV bytes differ on rank {rank} at context {length}")
                        timings = {name: [] for name in graphs}
                        for repeat in range(8):
                            for name in (("direct", "stage") if repeat % 2 == 0 else ("stage", "direct")):
                                dist.barrier()
                                begin, finish = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                                begin.record()
                                graphs[name].replay()
                                finish.record()
                                finish.synchronize()
                                timings[name].append(begin.elapsed_time(finish))
                        results[str(length)] = timings
                    all_ranks = [None] * world
                    dist.all_gather_object(all_ranks, results)
                    if rank == 0:
                        times = {length: {name: median(max(r[length][name][i] for r in all_ranks)
                                  for i in range(8)) for name in graphs} for length in results}
                        (directory / "results.json").write_text(json.dumps(dict(
                            slowest_rank_median_ms=times, per_rank_ms=all_ranks), indent=2) + "\n")
                        print(times, flush=True)
                    for graph in graphs.values():
                        graph.reset()
            finally:
                shared.close()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    torch.multiprocessing.spawn(measure_rank, args=(directory,), nprocs=16, join=True)
