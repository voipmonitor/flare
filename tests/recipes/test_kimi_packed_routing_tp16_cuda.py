"""The production routing constructor preserves gathered bits and exact expert selections on TP16."""

from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_TP_CUDA_TESTS') != '1',
                               reason='requires sixteen explicitly assigned CUDA devices')


class RouteForward:
    def __init__(self, operation):
        self.operation = operation

    def __call__(self):
        self.operation()
        self.operation.local_down.zero_()
        self.operation.local_router.zero_()


def _rank(rank, path):
    import torch.distributed as dist
    from b12x.comm import pcie
    from b12x.preparation import CollectiveRequirement, PreparedCall

    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.collectives import CollectivePreparation
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession, declare
    from ignition.recipes.kimi3.routing import PreparedRouteGather, create_route_runtime

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    root = Path(path)
    dist.init_process_group('gloo', init_method=(root / 'group').as_uri(), rank=rank, world_size=16,
                            timeout=timedelta(minutes=5))
    runtime = None
    reports = []
    try:
        with PackageStorage(rank=rank, world=16, device=rank) as storage:
            runtime = create_route_runtime(dist.group.WORLD, device, ipc=storage.peer_storage)
            coordinator = CollectivePreparation(dist.group.WORLD)
            for rows in (4, 7, 8):
                generator = torch.Generator(device=device).manual_seed(351 + rank)
                first = torch.randn((rows, 224), device=device, dtype=torch.bfloat16, generator=generator)
                second = torch.randn((rows, 56), device=device, generator=generator)
                latent = torch.empty((rows, 3584), device=device, dtype=torch.bfloat16)
                weights = torch.empty((rows, 16), device=device)
                ids = torch.empty((rows, 16), device=device, dtype=torch.int32)
                correction = torch.arange(896, dtype=torch.float32, device=device) / 89600
                reference_latent, reference_logits = torch.empty_like(latent), torch.empty((rows, 896), device=device)
                reference_weights, reference_ids = torch.empty_like(weights), torch.empty_like(ids)
                session = RecordedPreparationSession(device=device, checkpoint='packed-routing', workload_schema='tp16-pair',
                    numerical_policy='exact', coordinator=coordinator, compile_workers=16, autotune_effort='off')
                arena = ScratchArena(device)
                session.register_scratch(arena)
                graphs = {}
                try:
                    operation = PreparedRouteGather(first, second, correction, latent, weights, ids,
                        runtime=runtime, session=session, coordinator=coordinator, name='route', workspace=arena)
                    assert (operation.packed is not None) == (rows in (7, 8))
                    if rows == 4 and torch.cuda.get_device_properties(device).multi_processor_count == 188:
                        assert operation.plan.query.call["row_blocks"]
                    arguments = dict(local_first=first, local_second=second, out_first=reference_latent,
                                     out_second=reference_logits, threads=512)
                    reference = pcie.plan(pcie.query_from_runtime(runtime, surface='DcpAllToAll.all_gather_pair',
                                                                  call=arguments), runtime=runtime)
                    declare(session, (reference.request(name='reference', prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(**arguments)), collective=CollectiveRequirement(key='reference', ranks=tuple(range(16)))),),
                        coordinator=coordinator)
                    session.tune(show_progress=False)
                    forward = RouteForward(operation)
                    graphs, compiled = capture_forwards({'route': forward},
                        {'route': GraphSignature('speculative', 1, 8, packed_token_capacity=rows)},
                        session=session, device=device, stream=torch.cuda.Stream(device=device),
                        pool=torch.cuda.graph_pool_handle(), coordinator=coordinator,
                        cache_root=root / f'compiled-{rows}', retain_graphs=True, report_catalog=False)
                    session.freeze()
                    for iteration in range(4):
                        first.normal_(generator=generator)
                        second.normal_(generator=generator)
                        if iteration == 2:
                            first[-1].zero_()
                            second[-1].zero_()
                        seed = first.clone(), second.clone()
                        runtime.all_gather_pair(first, second, reference_latent, reference_logits, plan=reference, threads=512)
                        runtime.kimi_topk16(reference_logits, correction, reference_weights, reference_ids,
                                           plan=operation.select_plan)
                        torch.cuda.synchronize(device)
                        for mode in ('eager', 'compiled'):
                            first.copy_(seed[0])
                            second.copy_(seed[1])
                            arena.tensor.fill_(0xA5)
                            latent.fill_(float('nan'))
                            weights.fill_(float('nan'))
                            ids.fill_(-1)
                            if mode == 'eager':
                                forward()
                            else:
                                before = torch.cuda.memory_stats(device)
                                for _ in range(3):
                                    first.copy_(seed[0])
                                    second.copy_(seed[1])
                                    graphs['route'].replay()
                                torch.cuda.synchronize(device)
                                after = torch.cuda.memory_stats(device)
                                for key in ('allocation.all.allocated', 'allocation.all.freed', 'segment.all.allocated'):
                                    assert before[key] == after[key], (mode, key)
                            torch.cuda.synchronize(device)
                            exact = (torch.equal(latent.view(torch.uint8), reference_latent.view(torch.uint8))
                                     and torch.equal(operation.logits.view(torch.uint8), reference_logits.view(torch.uint8))
                                     and torch.equal(weights.view(torch.uint8), reference_weights.view(torch.uint8))
                                     and torch.equal(ids, reference_ids))
                            peers = [None] * 16
                            dist.all_gather_object(peers, exact)
                            assert all(peers), (rows, iteration, mode, peers)
                            assert not first.count_nonzero() and not second.count_nonzero()
                    if operation.packed is not None:
                        # Transport must preserve every FP32 bit, including values
                        # outside the finite router-selection domain.
                        pattern = torch.tensor([0, -2147483648, 0x7F800000, 0x7FC00001,
                                                0x7F800001, 0x00800000, 1, -1], device=device, dtype=torch.int32)
                        for value in (first, second):
                            words = value.view(torch.int32).flatten()
                            words.copy_(pattern.repeat((words.numel() + 7) // 8)[:words.numel()])
                        runtime.all_gather_pair(first, second, reference_latent, reference_logits,
                                                plan=reference, threads=512)
                        operation.packed.gather()
                        torch.cuda.synchronize(device)
                        assert torch.equal(latent.view(torch.uint8), reference_latent.view(torch.uint8))
                        assert torch.equal(operation.logits.view(torch.uint8), reference_logits.view(torch.uint8))
                    reports.append(dict(rows=rows, packed=operation.packed is not None,
                                        scratch_bytes=arena.nbytes, exact_gather_and_selection=True))
                finally:
                    for graph in graphs.values():
                        graph.reset()
                    session.close()
            if rank == 0:
                (root / 'checks.json').write_text(json.dumps(reports, indent=2) + '\n')
    finally:
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


def test_production_packed_routing_eager_compiled_poisoned_storage(tmp_path):
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() == 16
    mp.spawn(_rank, args=(str(tmp_path),), nprocs=16, join=True)
