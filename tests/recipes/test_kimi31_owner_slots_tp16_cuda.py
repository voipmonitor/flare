# SPDX-License-Identifier: Apache-2.0
"""Exercise rank-local global top-k membership, channel reuse and capture on sixteen CUDA devices."""

import ctypes
from datetime import timedelta
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def candidate_case(positions, pattern, rank):
    """Compute the stable global selection independently using every padded input on the CPU."""
    positions = np.asarray(positions, dtype=np.int64)
    peer, item = np.ogrid[:16, :2048]
    ids = np.empty((4, 16, 2048), dtype=np.int32)
    bits = np.empty_like(ids, dtype=np.uint32)
    local_slots = np.full((4, 2048), -1, np.int32)
    owners = np.full_like(local_slots, -1)
    counts = np.zeros(4, np.int32)
    pages = np.tile(np.arange(512, dtype=np.int32), (4, 1))
    pages[:, 5::7] = -1
    pages[:, 8::11] = 512
    for row, position in enumerate(positions):
        available = np.maximum(0, (position + 16 - peer) // 16)
        count = np.minimum(2048, available)
        local = np.where(item < count, available - 1 - item, -1)
        rng = np.random.default_rng(3221 + row)
        scores = rng.normal(size=(16, 2048)).astype(np.float32).view(np.uint32)
        if pattern == "skew":
            scores = np.broadcast_to((peer * 2048 + item).astype(np.float32).view(np.uint32), scores.shape).copy()
        elif pattern == "ties":
            scores.fill(0xff800000)
            scores[:, ::9] = 0x7f800000
            scores[:, 4::15] = 0xffc00001
            scores[:, 9::19] = 0x7fc00001
        elif pattern == "padding":
            local[:, 1:] = -1
            scores.fill(0xff800000)
        elif pattern == "slow_reader":
            scores[15, ::3] = 0xff800000
        ids[row] = local
        bits[row] = scores
        canonical = np.where(local >= 0, scores, np.uint32(0xff800000))
        ordered = np.where(canonical >> 31 != 0, ~canonical, canonical ^ np.uint32(0x80000000))
        winners = np.argsort(~ordered.reshape(-1), kind="stable")[:2048]
        chosen = np.sort(winners[winners // 2048 == rank] % 2048)
        tokens = local[rank, chosen]
        tokens = tokens[(tokens >= 0) & (tokens * 16 + rank <= position)]
        physical = pages[row, tokens // 128]
        kept = (physical >= 0) & (physical < 512)
        n = counts[row] = np.count_nonzero(kept)
        local_slots[row, :n] = physical[kept] * 128 + tokens[kept] % 128
        owners[row, :n] = rank
    tensors = [ids[:, rank].copy(), bits[:, rank].copy().view(np.float32), positions, pages]
    return tuple(torch.from_numpy(x) for x in tensors), tuple(torch.from_numpy(x) for x in (owners, local_slots, counts))


def exercise(runtime, coordinator, device, *, broadcast_tables):
    import torch.distributed as dist
    from ignition.components.collectives import PreparedAllGather
    from ignition.preparation.capture_stream import capture_stream
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi31.owner_selection import PreparedOwnerSelection, use_owner_selection

    context = SimpleNamespace(candidates=runtime, coordinator=coordinator)
    rank = runtime.rank
    geometries = [
        ((-1, 0, 15, 511), "slow_reader"),
        ((2043, 2044, 2045, 2046), "padding"),
        ((2046, 2047, 2048, 2049), "ties"),
        ((2048, 2049, 2050, 2051), "finite"),
        ((3071, 4095, 6143, 8191), "skew"),
        ((32767, 65535, 131071, 1048575), "skew"),
        ((8191, 32767, 65535, 1048575), "ties"),
    ]
    cases = [candidate_case(pos, pattern, rank) for pos, pattern in geometries]
    host, expected = cases[0]
    ids, scores, positions, tables = (x.to(device) for x in host)
    if broadcast_tables:
        tables = tables[:1].expand_as(tables)
    args = (ids, scores, positions, tables)
    selector = SimpleNamespace(**dict(zip(("local", "scores", "positions", "tables"), args, strict=True)))
    outputs = tuple(torch.empty_like(t, device=device) for t in expected)
    inputs = [tuple(x.to(device) for x in data) for data, _ in cases]
    cache = torch.empty(512, device=device)
    source = torch.full((4, 8192), rank + 1, dtype=torch.bfloat16, device=device)
    gathered = torch.empty((4, 16 * 8192), dtype=source.dtype, device=device)
    graphs = []
    try:
        with RecordedPreparationSession(device=device, checkpoint="owner-local-attention",
                workload_schema="four-row-stable-topk", numerical_policy="exact", coordinator=coordinator,
                compile_workers=16, autotune_effort="off") as session:
            assert use_owner_selection("verify", 4, context, device, session)
            classify = PreparedOwnerSelection(selector, *outputs, cache, context, session, "select")
            ordinary = PreparedAllGather(source, gathered, runtime=runtime, session=session,
                                         coordinator=coordinator, name="gather")
            session.tune(show_progress=False)
            assert session.tuning_report["benchmarked_candidates"] == 0
            stream = capture_stream(device)
            stream.wait_stream(torch.cuda.current_stream())
            for operation in (classify, lambda: ordinary(source)):
                graph = torch.cuda.CUDAGraph()
                graphs.append(graph)
                with torch.cuda.graph(graph, stream=stream):
                    operation()
            # Seed both per-row protocol counters just below their wrap. The
            # empty mailboxes deliberately still contain generation zero.
            sequence = (ctypes.c_uint32 * 8)(*([0xffffff80 - 2] * 8))
            offset = (1024 + (63 * 16 + 15) * 32 + 4) * 4
            runtime._ipc.cudaMemcpyAsync(runtime._signal_ptrs[rank] + offset, ctypes.addressof(sequence),
                                         ctypes.sizeof(sequence), stream.cuda_stream)
            stream.synchronize()
            dist.barrier()
            # All devices queue these without host barriers. Rank 15 alone
            # has padding-score ties in the repeated short case and is delayed.
            schedule = [*range(len(cases)), *([0] * 132), 1, 3, 0, 6, 2, 5, 0]
            snapshots = []
            with torch.cuda.stream(stream):
                for step, case in enumerate(schedule):
                    if rank == 15 and step in (2, 10, 65, 126):
                        torch.cuda._sleep(1500000)
                    if step in (2, 5, 78, 140):
                        graphs[1].replay()
                    for target, src in zip(args[:3], inputs[case][:3], strict=True):
                        target.copy_(src)
                    if broadcast_tables:
                        tables[:1].copy_(inputs[case][3][:1])
                    else:
                        tables.copy_(inputs[case][3])
                    graphs[0].replay()
                    if step < len(cases) or step >= 135 or step % 15 == 0:
                        snapshots.append((case, tuple(x.clone() for x in outputs)))
            stream.synchronize()
            for case, result in snapshots:
                assert all(torch.equal(x.cpu(), y) for x, y in zip(result, cases[case][1], strict=True)), (rank, case)
            target = torch.arange(1, 17, dtype=source.dtype, device=device).view(1, 16, 1)
            assert torch.equal(gathered.view(4, 16, 8192), target.expand(4, 16, 8192))
            dist.barrier()
    finally:
        torch.cuda.synchronize(device)
        for graph in graphs:
            graph.reset()


def _rank(rank, rendezvous):
    import torch.distributed as dist
    from ignition.preparation.collectives import CollectivePreparation, create_head_exchange
    from ignition.preparation.package_storage import PackageStorage

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=16,
                            timeout=timedelta(minutes=5))
    try:
        with PackageStorage(rank=rank, world=16, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            for broadcast in (False, True):
                runtime = create_head_exchange(dist.group.WORLD, device, max_rows=8, heads=16,
                                               head_dim=8192, ipc=storage.peer_storage)
                try:
                    exercise(runtime, coordinator, device, broadcast_tables=broadcast)
                finally:
                    runtime.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                    reason="requires sixteen explicitly assigned SM120 GPUs")
def test_owner_slots_against_stable_cpu_selection_and_generation_wrap(tmp_path):
    import torch.multiprocessing as mp
    assert torch.cuda.device_count() == 16
    mp.spawn(_rank, args=((tmp_path / "interleaved-topk").as_uri(),), nprocs=16, join=True)
