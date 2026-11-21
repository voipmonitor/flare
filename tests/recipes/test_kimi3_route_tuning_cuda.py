# SPDX-License-Identifier: Apache-2.0
"""Exercise complete Kimi router tuning and warm reload over a real 16-rank peer group."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch


def _worker(rank, directory):
    import torch.distributed as dist
    from ignition.preparation.collectives import CollectivePreparation
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.routing import PreparedRouteGather, create_route_runtime

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=(Path(directory) / "group").as_uri(), rank=rank, world_size=16,
                            timeout=timedelta(minutes=5))
    channel = create_route_runtime(dist.group.WORLD, device)
    coordinator = CollectivePreparation(dist.group.WORLD)
    try:
        for load in range(2):
            with RecordedPreparationSession(device=device, checkpoint="kimi-expert-routing", workload_schema="kimi3",
                    numerical_policy="default", coordinator=coordinator, compile_workers=16) as session:
                session.cache_dir = Path(directory) / "cache"
                local = torch.empty((4, 224), device=device, dtype=torch.bfloat16)
                router = torch.empty((4, 56), device=device)
                bias = torch.zeros(896, device=device)
                route = PreparedRouteGather(local, router, bias, torch.empty((4, 3584), dtype=local.dtype, device=device),
                    torch.empty((4, 16), device=device), torch.empty((4, 16), dtype=torch.int32, device=device),
                    runtime=channel, session=session, coordinator=coordinator, name="route")
                session.tune()
                selected = route.plan.selection
                assert selected.source == ("cached" if load else "tuned")
                choices = [None] * 16
                dist.all_gather_object(choices, route.plan.contract.config_payload(selected.config).to_dict())
                assert all(c == choices[0] for c in choices)
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    latent, ids, weights = route(local, router)
                session.freeze()
                for step in range(3):
                    vals = torch.arange(4 * 3584, device=device).reshape(4, 3584).add(step).to(torch.int16)
                    logits = torch.sin(torch.arange(4 * 896, device=device).reshape(4, 896) * .017 + step)
                    if step == 1:
                        logits.zero_()
                    if step == 2:
                        logits[:, ::17] = float("nan")
                    local.copy_(vals.view(torch.bfloat16)[:, rank * 224:(rank + 1) * 224])
                    router.copy_(logits[:, rank * 56:(rank + 1) * 56])
                    graph.replay()
                    torch.cuda.synchronize()
                    scores = torch.nan_to_num(logits.sigmoid(), nan=0.)
                    chosen = torch.argsort(scores, descending=True, stable=True)[:, :16]
                    assert torch.equal(latent.view(torch.int16), vals)
                    assert torch.equal(ids, chosen.int())
                    expected = scores.gather(1, chosen)
                    torch.testing.assert_close(weights, expected / expected.sum(1, keepdim=True), rtol=1e-5, atol=1e-7)
                graph.reset()
    finally:
        torch.cuda.synchronize()
        channel.close()
        dist.destroy_process_group()


def test_complete_route_tunes_once_and_replays_on_all_ranks(tmp_path):
    if torch.cuda.device_count() < 16 or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires sixteen peer-accessible SM120 GPUs")
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path),), nprocs=16, join=True)
