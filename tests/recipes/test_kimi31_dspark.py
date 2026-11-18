# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inferact DSpark checkpoint, context storage, and target-weight sharing."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from .test_kimi31 import config, raw_config


@pytest.fixture(scope="module")
def draft_snapshot():
    snapshot = Path(os.environ.get("FLARE_KIMI31_DSPARK_CHECKPOINT", str(Path.home() /
        ".cache/huggingface/hub/models--Inferact--Kimi-K3.1-Pro-DSpark-preview/snapshots/"
        "6a2ef20a05efe88aabbf9ecaa12d020c3b0edf94")))
    if not (snapshot / "model.safetensors").is_file():
        pytest.skip("Inferact Kimi K3.1 MLA DSpark checkpoint is not installed")
    return snapshot


@pytest.fixture(scope="module")
def draft_config(config, draft_snapshot):
    from ignition.recipes.kimi31.dspark_config import DSparkConfig

    return DSparkConfig.load(draft_snapshot, config)


@pytest.fixture(scope="module")
def qwen_raw_config():
    fixture = Path(__file__).parents[1] / "fixtures/kimi31/draft-qwen-config.json"
    return json.loads(fixture.read_text())


@pytest.fixture(scope="module")
def qwen_draft_config(config, qwen_raw_config):
    from ignition.recipes.kimi31.dspark_config import DSparkConfig

    return DSparkConfig.from_dict(qwen_raw_config, config.text)


@pytest.fixture(scope="module")
def qwen_draft_snapshot():
    snapshot = Path(os.environ.get("FLARE_KIMI31_QWEN_DSPARK_CHECKPOINT", str(Path.home() /
        ".cache/huggingface/hub/models--Inferact--Kimi-K3.1-Pro-DSpark/snapshots/"
        "e4e6e6261e9f8b6d5f249f9627eb6b109d762d5f")))
    if not (snapshot / "model.safetensors").is_file():
        pytest.skip("Inferact Kimi K3.1 Qwen DSpark checkpoint is not installed")
    return snapshot


def test_qwen_draft_geometry(qwen_draft_config):
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.kimi3.config import DSparkConfig

    assert isinstance(qwen_draft_config, DSparkConfig)
    assert qwen_draft_config.auxiliary_layers == (76, 80, 84, 88, 92)
    assert qwen_draft_config.auxiliary_stream == "attn_res"
    assert qwen_draft_config.window == 2048
    for rank in range(16):
        qwen_draft_config.validate_shard(TensorParallelShard(rank, 16))


def test_actual_qwen_checkpoint_headers(qwen_draft_snapshot):
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint

    manifest = inspect_indexed_checkpoint(qwen_draft_snapshot)
    assert len(manifest.tensors) == 62
    assert "embed_tokens.weight" not in manifest.tensors
    assert "lm_head.weight" not in manifest.tensors
    assert all(tensor.dtype == "BF16" for tensor in manifest.tensors.values())


def test_actual_checkpoint_geometry_and_headers(config, draft_config, draft_snapshot):
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.kimi31.dspark_weights import _validate_checkpoint

    _validate_checkpoint(inspect_indexed_checkpoint(draft_snapshot), draft_config)
    assert draft_config.auxiliary_layers == (76, 80, 84, 88, 92)
    assert draft_config.auxiliary_stream == "prefix"
    assert draft_config.window is None
    for rank in range(16):
        draft_config.validate_shard(TensorParallelShard(rank, 16))


@pytest.mark.parametrize("field,value", [
    ("full_attention_causal", False), ("sliding_window", 1024),
    ("sample_from_anchor", False), ("qk_rope_head_dim", 0),
    ("target_layer_ids", [75, 79, 83, 87, 92]),
    ("target_layer_ids", [75, 79, 83, 87, 87]),
    ("dflash_config", {"causal": False}),
])
def test_incompatible_draft_semantics_rejected(config, draft_config, field, value):
    from ignition.recipes.kimi31.dspark_config import DSparkConfig

    raw = deepcopy(draft_config.raw)
    raw[field] = value
    with pytest.raises(ValueError, match="Kimi K3.1 DSpark"):
        DSparkConfig.from_dict(raw, config.text)


def test_rotary_attention_factor_cannot_be_dropped(config, draft_config):
    from ignition.recipes.kimi31.dspark_config import DSparkConfig

    raw = deepcopy(draft_config.raw)
    del raw["rope_parameters"]["attn_factor"]
    with pytest.raises(ValueError, match="rotary"):
        DSparkConfig.from_dict(raw, config.text)


def test_redhat_draft_cannot_attach_to_k31(config, qwen_raw_config):
    from ignition.recipes.kimi31.dspark_config import DSparkConfig

    raw = deepcopy(qwen_raw_config)
    raw["speculators_config"]["verifier"]["architectures"] = ["KimiK3ForConditionalGeneration"]
    with pytest.raises(ValueError, match="KimiK31ForConditionalGeneration"):
        DSparkConfig.from_dict(raw, config.text)


def test_draft_pages_and_partial_snapshots_are_budgeted(config, draft_config):
    import torch
    from ignition.recipes.kimi3.pool import ContinuationPool, continuation_layout, page_bytes

    config = replace(config, draft=draft_config)
    layout = continuation_layout(config, 16, speculative_tokens=7)
    assert layout.row_bytes == 33_238_272
    assert page_bytes(config) == 2_912_256
    pool = ContinuationPool.for_model(config, 16, 2, 3, "meta", speculative_tokens=7)
    for layer in range(5):
        cache = pool.draft_attention(layer)
        assert cache.shape == (3, 128, 576) and cache.dtype == torch.bfloat16
        assert layout.region(f"draft{layer}.latent").nbytes == cache[0].nbytes
    assert set(dict(pool.partial_pages())) == set(pool.pages)


def test_qwen_draft_windows_and_partial_snapshots_are_budgeted(config, qwen_draft_config):
    from ignition.recipes.kimi3.pool import ContinuationPool, continuation_layout, page_bytes
    from ignition.recipes.kimi31.recipe import RECIPE

    config = replace(config, draft=qwen_draft_config)
    costs = RECIPE.costs(16, RECIPE.speculation)
    layout = continuation_layout(config, 16, speculative_tokens=7)
    assert layout.row_bytes == costs.state_row_bytes == 35_122_432
    assert page_bytes(config) == costs.page_bytes == 2_174_976
    pool = ContinuationPool.for_model(config, 16, 2, 3, "meta", speculative_tokens=7)
    for layer in range(5):
        for kind in ("key", "value"):
            region = layout.region(f"draft{layer}.{kind}")
            assert region.shape == (2048, 1, 64)
            assert region.nbytes == 262_144
            assert pool.reserved_region(region.name).shape == (2, 2048, 1, 64)
    assert not any(name.startswith("draft") for name in pool.pages)
    assert set(dict(pool.partial_pages())) == set(pool.pages)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")
def test_qwen_checkpoint_shares_target_vocabulary_and_matches_weight_budget(
        config, qwen_draft_config, qwen_draft_snapshot):
    import torch
    from dataclasses import fields, is_dataclass
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.kimi3.dspark_weights import DraftWeights
    from ignition.recipes.kimi31.recipe import RECIPE

    config = replace(config, draft=qwen_draft_config)
    device = torch.device("cuda:0")
    shard = TensorParallelShard(0, 16)
    hidden_rows = shard.partition(config.hidden, alignment=64)
    vocabulary = shard.partition(config.vocabulary, alignment=64)
    embedding = torch.empty((config.vocabulary, hidden_rows[1] - hidden_rows[0]),
                            dtype=torch.bfloat16, device=device)
    head = torch.empty((vocabulary[1] - vocabulary[0], config.hidden), dtype=torch.bfloat16, device=device)
    draft = DraftWeights.load(qwen_draft_snapshot, config, device, shard,
                              target_embedding=embedding, target_head=head)

    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.nbytes
        if is_dataclass(value) and not isinstance(value, type):
            return sum(tensor_bytes(getattr(value, field.name)) for field in fields(value))
        if isinstance(value, (tuple, list)):
            return sum(tensor_bytes(item) for item in value)
        if isinstance(value, dict):
            return sum(tensor_bytes(item) for item in value.values())
        return 0

    allocated = tensor_bytes(draft) - embedding.nbytes
    assert draft.embedding is embedding
    assert draft.head is None
    assert draft.context_kv.shape == (640, config.hidden)
    assert all(layer.qkv.shape == (512, config.hidden) for layer in draft.layers)
    assert allocated * 16 == RECIPE.checkpoints[0].draft_bytes


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")
def test_qwen_draft_quantization_preserves_target_vocabulary(
        config, qwen_draft_config, qwen_draft_snapshot):
    import torch
    from types import SimpleNamespace
    from ignition.components.markov_sampling import (PreparedRowScaledFP8MarkovProjection,
                                                     RowScaledFP8MarkovWeight)
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.linear import QuantizedProjection
    from ignition.components.vocabulary import PreparedVocabularyProjection
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.dspark import PreparedMarkov
    from ignition.recipes.kimi3.dspark_weights import DraftWeights
    from ignition.recipes.kimi3.weights import _prepare_draft_heads

    config = replace(config, draft=qwen_draft_config)
    device = torch.device("cuda:0")
    shard = TensorParallelShard(0, 16)
    hidden_rows = shard.partition(config.hidden, alignment=64)
    vocabulary = shard.partition(config.vocabulary, alignment=64)
    embedding = torch.empty((config.vocabulary, hidden_rows[1] - hidden_rows[0]),
                            dtype=torch.bfloat16, device=device)
    head = torch.randn((vocabulary[1] - vocabulary[0], config.hidden), dtype=torch.bfloat16, device=device) * .02
    head_pointer = head.data_ptr()
    draft = DraftWeights.load(qwen_draft_snapshot, config, device, shard, target_embedding=embedding,
                              target_head=head, online_mxfp8=True, replicated_markov=True)
    assert not isinstance(draft.projection, torch.Tensor)
    assert not isinstance(draft.context_kv, torch.Tensor)
    for layer in draft.layers:
        assert not isinstance(layer.qkv, torch.Tensor)
        assert not isinstance(layer.output, torch.Tensor)
        assert not isinstance(layer.mlp.gate_up, torch.Tensor)
        assert not isinstance(layer.mlp.down, torch.Tensor)
    markov_reference = draft.markov_head
    draft = _prepare_draft_heads(draft, head, draft_head_quantization="nvfp4",
                                 markov_head_quantization="fp8")
    assert draft.embedding is embedding
    assert draft.head is not None and not isinstance(draft.head, torch.Tensor)
    assert isinstance(draft.markov_head, RowScaledFP8MarkovWeight)
    assert draft.markov_head.out_features == config.vocabulary
    assert head.data_ptr() == head_pointer and head.dtype == torch.bfloat16

    markov_weights = SimpleNamespace(config=config, shard=shard, draft=draft)
    with RecordedPreparationSession(device=device, checkpoint=str(qwen_draft_snapshot),
            workload_schema="kimi-dspark-quantized-heads", numerical_policy="proposal-quantization",
            compile_workers=1) as session:
        tokens = torch.zeros((1,), dtype=torch.int64, device=device)
        markov = PreparedMarkov(markov_weights, tokens, runtime=None, coordinator=None, session=session, name="markov")
        assert isinstance(markov.projection, PreparedRowScaledFP8MarkovProjection)
        arena = ScratchArena(device)
        session.register_scratch(arena)
        hidden = torch.zeros((7, config.hidden), dtype=torch.bfloat16, device=device)
        local = torch.zeros((7, vocabulary[1] - vocabulary[0]), dtype=torch.bfloat16, device=device)
        logits = torch.zeros((7, draft.head.out_features), dtype=torch.bfloat16, device=device)
        projection = PreparedVocabularyProjection(hidden, draft.head, local, logits, runtime=None,
            coordinator=None, session=session, name="head", workspace=arena)
        assert isinstance(projection.projection, QuantizedProjection)
        session.tune(show_progress=False)
        actual = compile_forward(markov)()
        expected = draft.markov_embedding[tokens].float() @ markov_reference.float().T
        error = ((actual - expected).square().mean().sqrt()
                 / expected.square().mean().sqrt().clamp_min(1e-12))
        assert error < .02


def test_draft_weight_declaration_preserves_target_embedding_alias():
    import torch
    from dataclasses import dataclass
    from ignition.components.weights import MLPWeights
    from ignition.preparation.rebuild_manifest import declare_weights, weight_payload_declaration
    from ignition.recipes.kimi31.dspark_weights import DraftLayerWeights, DraftWeights

    def tensor():
        return torch.zeros((2, 2), dtype=torch.bfloat16)

    embedding = tensor()
    layer = DraftLayerWeights({"q_b": tensor()}, tensor(), tensor(), MLPWeights(tensor(), tensor()))
    draft = DraftWeights(tensor(), tensor(), tensor(), tensor(), embedding, None,
                        tensor(), tensor(), (layer,))
    @dataclass
    class Model:
        embedding: torch.Tensor
        draft: DraftWeights
        layers: tuple = ()

    model = Model(embedding, draft)
    declaration = declare_weights(model, components=("embedding", "draft"),
        aliases=(("weights.draft.embedding", "weights.embedding"),))
    views = {view["name"]: view for view in weight_payload_declaration(declaration)["views"]}
    assert views["weights.draft.embedding"]["alias_of"] == "weights.embedding"
    assert "weights.draft.layers[0].attention['q_b']" in views
    assert "weights.draft.layers[0].mlp.gate_up" in views
    assert "weights.draft.markov_embedding" in views
