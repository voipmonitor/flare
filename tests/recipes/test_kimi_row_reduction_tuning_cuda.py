# SPDX-License-Identifier: Apache-2.0
"""Check normal cold and cached selection of complete reduced Kimi activations on a TP16 peer group."""

from datetime import timedelta
from pathlib import Path
import os

import pytest
import torch


def _worker(rank, directory):
    import torch.distributed as dist
    from b12x.comm.pcie import AllReduce
    from ignition.components.collectives import PreparedChunkedAllReduce
    from ignition.preparation.collectives import CollectivePreparation
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.row_reductions import RowGeometry, RowReductionTuning
    from ignition.recipes.kimi31.recipe import RECIPE

    for name, value in RECIPE.b12x_environment.items():
        os.environ.setdefault(name, value)
    device = torch.device("cuda", rank)
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    root = Path(directory)
    dist.init_process_group(
        "gloo", init_method=(root / "group").as_uri(), rank=rank, world_size=16, timeout=timedelta(minutes=10)
    )
    coordinator = CollectivePreparation(dist.group.WORLD)
    owner = AllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD, device=device, single_channel=True, max_size=1 << 18
    )
    try:
        previous = None
        for load in range(2):
            with RecordedPreparationSession(
                device=device,
                checkpoint="kimi-row-fusion",
                workload_schema="kimi3",
                numerical_policy="default",
                coordinator=coordinator,
                compile_workers=16,
            ) as session:
                session.cache_dir = root / "cache"
                catalog = RowReductionTuning(coordinator, device, 1e-5, root / "forwards")
                specs = (
                    RowGeometry(3584),
                    RowGeometry(7168, sources=1, mode=2, blocks=8),
                    RowGeometry(7168, sources=6, mode=1, blocks=8, taps=5),
                )
                for n, spec in enumerate(specs):
                    source = torch.empty((4, spec.width), dtype=torch.bfloat16, device=device)
                    reduction = PreparedChunkedAllReduce(
                        source,
                        torch.empty_like(source),
                        runtime=owner,
                        session=session,
                        coordinator=coordinator,
                        name=f"layer-{n}.reduce",
                    ).functional(session=session)
                    catalog.register(reduction, spec)
                session.tune()
                results = catalog.tune(session)
                assert len(results) == len(specs)
                assert all(result["source"] == ("cached" if load else "tuned") for result in results)
                assert all(result.get("rejected_candidates", 0) == 0 for result in results)
                winning = [r["implementation"] for r in results]
                assert all(impl == "fused" for impl in winning), f"expected fused winners, got {winning}"
                ranks = [None] * 16
                dist.all_gather_object(ranks, winning)
                assert all(selection == winning for selection in ranks)
                if previous is not None:
                    assert winning == previous
                previous = winning
    finally:
        torch.cuda.synchronize()
        owner.close()
        dist.destroy_process_group()


def test_four_row_reduction_tuning_replays_and_restores_selections(tmp_path):
    if torch.cuda.device_count() < 16 or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires sixteen peer-accessible SM120 GPUs")
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path),), nprocs=16, join=True)
