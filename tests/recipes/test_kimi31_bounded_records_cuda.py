# SPDX-License-Identifier: Apache-2.0
"""Replay two compiled sparse prefill slices while their visible history changes."""

import os

import pytest
import torch

from b12x.preparation import PreparationSession
from ignition.components.prefill_record_stage import PreparedPrefillRecordStage
from ignition.recipes.kimi31.dcp import ContextShards, RECORD_BYTES, TOPK, selection_slots


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA device")
@pytest.mark.parametrize("world", [8, 16])
def test_compiled_prefill_consumes_original_selected_bytes(world, tmp_path):
    device = torch.device("cuda", 0)
    rows, page, pages = 2, 128, 6
    capacity = rows * TOPK
    generator = torch.Generator().manual_seed(172)
    host = torch.randint(0, 256, (world, pages, page, RECORD_BYTES), generator=generator, dtype=torch.uint8)
    cache = host.to(device)
    peers = torch.tensor([part.data_ptr() for part in cache], dtype=torch.int64, device=device)
    table = torch.tensor([2, 1, -1, 0, 4, 3], device=device, dtype=torch.int32)
    end = torch.zeros(1, device=device, dtype=torch.int64)
    positions = torch.empty((2 * rows,), device=device, dtype=torch.int64)
    ids = torch.empty((2 * rows, TOPK), device=device, dtype=torch.int32)
    shards = ContextShards.on(0, world, device)
    linear = torch.arange(capacity, device=device, dtype=torch.int32).view(rows, TOPK)
    declared = torch.empty((capacity, RECORD_BYTES), dtype=torch.uint8, device=device)

    with PreparationSession(device=device, compile_workers=0, cache_dir=tmp_path) as session:
        component = PreparedPrefillRecordStage(peers, cache[0], table, ids[:rows], linear, positions[:rows].int(),
                                               declared, session=session, name="prefill_records")

        def selected_bytes():
            records = torch.empty_like(declared)
            slices = []
            for start in (0, rows):
                tokens = torch.empty_like(linear)
                slots = torch.empty_like(tokens)
                counts = torch.empty((rows,), dtype=torch.int32, device=device)
                selection_slots(ids[start:start + rows], positions[start:start + rows], table.expand(rows, -1),
                                tokens, slots, counts, shards, pages, local=False, logical_ids=True)
                records = component(table, end, tokens, slots, counts, records, first=start == 0)
                page_ids = torch.where(end <= capacity, tokens.clamp_min(0), linear)
                valid = torch.arange(TOPK, device=device)[None, :] < counts[:, None]
                slices.append(torch.where(valid[:, :, None], records[page_ids.long()], 0))
            return torch.cat(slices)

        forward = torch.compile(selected_bytes, fullgraph=True)
        ids.fill_(-1)
        positions.fill_(-1)
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            output = forward()
        host_table = table.cpu().tolist()
        for length in (0, capacity, capacity + 1, 37, capacity + 225):
            p = [max(-1, length - 2), length - 1, length - 1, -1]
            pos = torch.tensor(p)
            chosen = torch.full((4, TOPK), -1, dtype=torch.int32)
            chosen[:, :16] = torch.randint(-2, max(1, length + 4), (4, 16), generator=generator)
            chosen[:, 16:20] = torch.tensor([0, 2, length + 1, -1])
            positions.copy_(pos)
            ids.copy_(chosen)
            end.fill_(length)
            graph.replay()
            observed = output.cpu()
            expected = torch.zeros_like(observed)
            for row, limit in enumerate(p):
                next_row = 0
                for token in chosen[row, :20].tolist():
                    if 0 <= token <= limit:
                        local = token // world
                        physical = host_table[local // page]
                        if 0 <= physical < pages:
                            expected[row, next_row].copy_(host[token % world, physical, local % page])
                            next_row += 1
            assert torch.equal(observed, expected), (world, length)
        graph.reset()
