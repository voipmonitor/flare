# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Red Hat DSpark scratch ownership and checkpoint selection."""

import pytest
import torch


@pytest.mark.parametrize("replicated", [False, True])
def test_registering_packed_rounds_preserves_the_graph_catalog(monkeypatch, replicated):
    from types import SimpleNamespace
    from ignition.components import parallel_draft
    from ignition.components.capacity import parallel_speculative_shapes
    from ignition.preparation.execution import GraphSignature
    from ignition.recipes.kimi3.bootstrap import _register_speculative_graphs
    from ignition.recipes.kimi3.config import DSparkConfig
    from ignition.recipes.kimi3.dspark import DraftForwards

    logits = torch.arange(7)
    draft = DraftForwards(None, None, lambda gather_rows=None: logits[:gather_rows], None)
    owner = SimpleNamespace(forwards=draft, block_proposal_broadcast=replicated,
        weights=SimpleNamespace(config=SimpleNamespace(draft=DSparkConfig({}))))

    class Round:
        def __init__(self, owner, batch, width, name, *, target_token_capacity, forwards):
            self.draft = forwards.draft

        def prepare(self):
            self.logits = self.draft()

    monkeypatch.setattr(parallel_draft, "PreparedParallelDraftRound", Round)
    prefill = object()
    catalog = {"prefill2048": prefill}
    signatures = {}
    def register(name, operation, kind, rows, tokens, fields, *, enabled, packed_tokens):
        catalog[name] = operation
        signatures[name] = GraphSignature(kind, rows, tokens, fields, speculation_enabled=enabled,
                                          packed_token_capacity=packed_tokens)

    shapes = (*parallel_speculative_shapes(1, tuple(range(1, 9)), 32, 16), (2, 8, 6))
    _register_speculative_graphs(owner, shapes, SimpleNamespace(buffers=lambda: ("input_token_ids",)), register)
    assert catalog["prefill2048"] is prefill and owner.forwards is draft
    assert len(catalog) == 1 + len(shapes) and len(signatures) == len(shapes)
    for batch, width, packed in shapes:
        name = f"speculative{batch}x{width}-tokens{packed}"
        signature = signatures[name]
        assert (signature.batch_capacity, signature.token_capacity, signature.packed_token_capacity) == (batch, width, packed)
        assert signature.inputs == ("input_token_ids", "sampling_grammar_mask")
        count = packed - 1 if replicated and batch == 1 and 1 < packed < width else 7
        assert torch.equal(catalog[name].logits, logits[:count])


@pytest.mark.parametrize("recipe", ["kimi3", "kimi31"])
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_draft_result_preserves_every_block_position(recipe, batch):
    from importlib import import_module
    from types import SimpleNamespace

    prepared = import_module(f"ignition.recipes.{recipe}.dspark").PreparedDraft
    draft = prepared.__new__(prepared)
    block, hidden, vocabulary = 7, 16, 32
    embedding = torch.nn.Embedding(vocabulary, hidden)
    draft.embedding = lambda *, token_ids, num_rows: embedding(token_ids)
    draft.layers = []
    draft.norm = torch.nn.Identity()
    draft.head = torch.nn.Linear(hidden, vocabulary, bias=False)
    draft.block = block
    metadata = SimpleNamespace(token_ids=torch.zeros(batch * block, dtype=torch.int64),
        sample_indices=torch.arange(batch * block), live_counts=torch.tensor([batch * block]),
        lengths=torch.full((batch,), block))

    logits, selector_features = draft(metadata)
    assert logits.view(batch, block, vocabulary).shape == (batch, block, vocabulary)
    assert selector_features is None


def test_draft_scratch_export_ranges():
    import ctypes
    from dataclasses import fields, is_dataclass
    from types import SimpleNamespace
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.disposable import storage_ranges
    from ignition.recipes.kimi3.dspark import DraftScratch as DSparkScratch

    config = SimpleNamespace(hidden=128, vocabulary=512, draft=SimpleNamespace(
        heads=2, kv_heads=1, head_dim=64, window=128, text={"intermediate_size": 256}))
    scratch = DSparkScratch.allocate(config, TensorParallelShard(0, 1), 32, 8, "cpu")

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif is_dataclass(value):
            for field in fields(value):
                yield from tensors(getattr(value, field.name))
        elif isinstance(value, dict):
            for child in value.values():
                yield from tensors(child)

    owned = tuple(tensors(scratch))
    for tensor in owned:
        tensor.fill_(1)
    ranges = storage_ranges(scratch.scratch_tensors())
    assert len(ranges) == len({tensor.untyped_storage().data_ptr() for tensor in owned})
    assert ranges == storage_ranges(scratch.rows(8, 8).scratch_tensors())
    for address, size in ranges:
        ctypes.memset(address, 0, size)
    assert all(not tensor.count_nonzero() for tensor in owned)
