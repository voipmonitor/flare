# SPDX-License-Identifier: Apache-2.0

"""Prepared candidate exchange against an independent CPU byte oracle on TP16."""

from datetime import timedelta
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_TP_CUDA_TESTS') != '1',
                              reason='requires sixteen explicitly assigned CUDA devices')


def exercise(runtime, coordinator, device):
    """Exercise the production components on either an isolated or caller-owned channel."""
    import torch.distributed as dist

    from ignition.components.prefix_candidate_gather import PreparedPrefixCandidateGather
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.dcp import ContextShards
    from ignition.recipes.kimi31.candidate_packing import pack_candidates_with_prefixes

    rank, reports = dist.get_rank(), []
    with RecordedPreparationSession(device=device, checkpoint='prefix-candidate-contract',
            workload_schema='prefix-candidate-contract', numerical_policy='exact',
            coordinator=coordinator, compile_workers=16, autotune_effort='default') as session:
        cases = []
        for rows in (1, 2, 4, 7, 8):
            indices = torch.full((rows, 2048), -1, dtype=torch.int32, device=device)
            scores = torch.empty((rows, 2048), dtype=torch.float32, device=device)
            positions = torch.full((rows,), -1, dtype=torch.int64, device=device)
            packed = torch.empty((rows, 2048, 2), dtype=torch.float32, device=device)
            gathered = torch.empty((rows, 32768, 2), dtype=torch.float32, device=device)
            prefixes = torch.zeros((rows, 64), dtype=torch.int32, device=device)
            shards = ContextShards.on(rank, 16, device)
            component = PreparedPrefixCandidateGather(packed.view(torch.bfloat16).view(rows, -1),
                gathered.view(torch.bfloat16).view(rows, -1), prefixes, runtime=runtime,
                session=session, coordinator=coordinator, name=f'prefix.contract.rows{rows}')
            cases.append((rows, indices, scores, positions, packed, gathered, prefixes, shards, component))
        session.tune(show_progress=False)
        assert session.tuning_report['benchmarked_candidates'] == 0
        for rows, indices, scores, positions, packed, gathered, prefixes, shards, component in cases:
            assert component.plan.selection.source == 'fixed'
            pack_candidates_with_prefixes(indices, scores, positions, shards.index, packed, prefixes)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    pack_candidates_with_prefixes(indices, scores, positions, shards.index, packed, prefixes)
                    component(packed.view(torch.bfloat16).view(rows, -1))
                for context in (-8, 0, 1, 15, 512, 32751, 32752, 32768, 1048576, -8):
                    pos = torch.arange(rows, dtype=torch.int64) + context - 1
                    total = (pos + 1).clamp(min=0)
                    length = (total // 16 + (total % 16 > rank)).clamp(max=2048)
                    column = torch.arange(2048, dtype=torch.int32)[None, :].expand(rows, -1)
                    local = torch.where((column < length[:, None]) & (column % 5 != 0), column, -1)
                    bits = torch.tensor([0, -(1 << 31), 1, -1, 2139095040, -8388608, 2143289345, 1065353216],
                                        dtype=torch.int32)[(column + rank) % 8]
                    local_expected = torch.stack((torch.where(local >= 0, bits, -8388608),
                                                   torch.where(local >= 0, local * 16 + rank, -1)), dim=-1)
                    peers = [None] * dist.get_world_size()
                    dist.all_gather_object(peers, local_expected)
                    expected = torch.stack(peers, dim=1).reshape(rows, 32768, 2)
                    indices.copy_(local)
                    scores.view(torch.int32).copy_(bits)
                    positions.copy_(pos)
                    packed.view(torch.int32).fill_(12345)
                    gathered.view(torch.int32).fill_(12345)
                    prefixes.fill_(-123)
                    allocations = torch.cuda.memory_stats(device)['allocation.all.allocated']
                    for _ in range(3):
                        graph.replay()
                    torch.cuda.synchronize(device)
                    stable = torch.cuda.memory_stats(device)['allocation.all.allocated'] == allocations
                    exact = (torch.equal(packed.view(torch.int32).cpu(), local_expected)
                             and torch.equal(gathered.view(torch.int32).cpu(), expected))
                    outcomes = [None] * dist.get_world_size()
                    dist.all_gather_object(outcomes, (stable, exact))
                    assert all(all(item) for item in outcomes), (rows, context, outcomes)
                    reports.append(dict(rows=rows, global_context=context, all_ranks_exact=True,
                                        allocation_stable=True, queued_replays=3))
            finally:
                torch.cuda.synchronize(device)
                graph.reset()
    return reports


def _rank(rank, rendezvous):
    import torch.distributed as dist

    from ignition.preparation.collectives import CollectivePreparation, create_head_exchange
    from ignition.preparation.package_storage import PackageStorage

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=16,
                            timeout=timedelta(minutes=5))
    runtime = None
    try:
        with PackageStorage(rank=rank, world=16, device=rank) as storage:
            runtime = create_head_exchange(dist.group.WORLD, device, max_rows=8, heads=16,
                                           head_dim=8192, ipc=storage.peer_storage)
            exercise(runtime, CollectivePreparation(dist.group.WORLD), device)
            runtime.close()
            runtime = None
    finally:
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


def test_candidate_prefixes_and_padding_match_cpu_oracle_on_tp16(tmp_path):
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() == 16, 'assign exactly sixteen CUDA devices'
    mp.spawn(_rank, args=((tmp_path / 'prefix-candidate-gather').as_uri(),), nprocs=16, join=True)
