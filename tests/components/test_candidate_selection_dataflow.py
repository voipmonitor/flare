# SPDX-License-Identifier: Apache-2.0

"""Candidate packets preserve score words and own their invocation storage."""

import pytest
import torch

from ignition.components.candidate_selection import PreparedCandidateSelection


@pytest.mark.parametrize("implementation", ["replicated", "owner"])
def test_invocation_packets_preserve_score_bits_and_global_ids(implementation):
    operation = PreparedCandidateSelection.__new__(PreparedCandidateSelection)
    operation.implementation = implementation
    operation.rows, operation.topk, operation.world = 2, 4, 16
    operation.rank = torch.tensor([3], dtype=torch.int64)
    scores = torch.tensor([[0, -2147483648, 2143289410, 1065353216],
                           [-8388608, 2139095040, -1082130432, 1056964608]], dtype=torch.int32).view(torch.float32)
    indices = torch.tensor([[0, 5, 9, -1], [3, 7, -1, 2]], dtype=torch.int32)
    packets = [operation.pack(indices=indices, scores=scores) for _ in range(2)]
    assert packets[0].untyped_storage()._cdata != packets[1].untyped_storage()._cdata
    live = indices >= 0
    expected_ids = torch.where(live, indices * 16 + 3, -1)
    expected_scores = torch.where(live, scores.view(torch.int32), -8388608)
    for packet in packets:
        if implementation == "owner":
            actual_scores, actual_ids = packet[:2].view(torch.int32), packet[2:].view(torch.int32)
        else:
            actual_scores, actual_ids = packet.view(torch.int32).unbind(-1)
        torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)
    packets[0].zero_()
    assert packets[1].count_nonzero()


@pytest.mark.parametrize("world,rows", [(8, 128), (16, 128), (16, 127)])
def test_gathered_owner_packets_restore_exact_rows(monkeypatch, world, rows):
    from types import SimpleNamespace
    from ignition.components import collectives
    from ignition.components.row_ownership import PreparedGatheredRows

    owned = -(-rows // world)
    topk = 2048
    # Include signed and noncanonical FP32 bit patterns; transport treats every int32 as bytes.
    words = torch.arange(world * owned * topk, dtype=torch.int32).reshape(world, owned, topk) ^ -8388608
    runtime = SimpleNamespace(heads_per_rank=1, query_head_dim=8192, max_batch_size=128)
    assert PreparedGatheredRows.fits(words[0], runtime)
    assert not PreparedGatheredRows.fits(words[0, :1], runtime)
    runtime.world_size = world
    rank, packets = world // 2, owned // 2

    class Gather:
        def __init__(self, source, output, **kwargs):
            assert source.shape == (packets, 8192)
            assert output.shape == (packets, world * 8192)

        def __call__(self, source):
            assert torch.equal(source.view(torch.int32), words[rank].reshape(packets, -1))
            return torch.cat([part.view(torch.bfloat16).reshape(packets, -1) for part in words], dim=1)

    monkeypatch.setattr(collectives, "PreparedAllGather", Gather)
    output = torch.empty((rows, topk), dtype=torch.int32)
    buffers = SimpleNamespace(zeros=lambda shape, dtype: torch.zeros(shape, dtype=dtype))
    operation = PreparedGatheredRows(words[rank], output, runtime=runtime, session=None,
                                     coordinator=None, name="owned IDs", buffers=buffers)
    expected = words.reshape(-1, topk)[:rows]
    assert torch.equal(operation(), expected)
    assert torch.equal(operation(words[rank]), expected)
