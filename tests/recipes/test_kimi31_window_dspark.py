# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 grouped-query DSpark source weights and prepared cache geometry."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.kimi3.pool import continuation_layout, page_bytes
from ignition.recipes.kimi31.dspark_config import WINDOW_REPOSITORY, WINDOW_REVISION, window_draft
from ignition.recipes.kimi31.recipe import RECIPE
from .test_kimi31 import config, raw_config


@pytest.fixture(scope="module")
def snapshot():
    default = (Path.home() / ".cache/huggingface/hub" / ("models--" + WINDOW_REPOSITORY.replace("/", "--"))
               / "snapshots" / WINDOW_REVISION)
    path = Path(os.environ.get("FLARE_KIMI31_WINDOW_DSPARK", default))
    if not (path / "config.json").is_file():
        pytest.skip("Inferact Kimi K3.1 sliding-window DSpark configuration is not installed")
    return path


def test_windowed_draft_planner_and_state(config, snapshot):
    raw = json.loads((snapshot / "config.json").read_text())
    draft = window_draft(raw, config.text)
    assert draft.auxiliary_layers == (76, 80, 84, 88, 92)
    assert draft.auxiliary_stream == "attn_res"
    for rank in range(16):
        draft.validate_shard(TensorParallelShard(rank, 16))
    target = replace(config, draft=draft)
    costs = RECIPE.costs(16, replace(RECIPE.speculation, method="dspark"))
    assert page_bytes(target) == costs.page_bytes == 2_174_976
    layout = continuation_layout(target, 16, speculative_tokens=7)
    assert layout.row_bytes == costs.state_row_bytes == 35_122_432
    for layer in range(draft.layers):
        for kind in ("key", "value"):
            region = layout.region(f"draft{layer}.{kind}")
            assert region.shape == (2048, 1, 64) and region.dtype == torch.bfloat16
    for path, value in [(("full_attention_causal",), False), (("speculators_config", "verifier", "architectures"),
                         ["KimiK3ForConditionalGeneration"]), (("transformer_layer_config", "sliding_window"), 1024)]:
        invalid = deepcopy(raw)
        node = invalid
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        with pytest.raises(ValueError, match="Kimi K3.1 DSpark"):
            window_draft(invalid, config.text)


def test_recipe_selects_the_windowed_draft_and_records_its_precision(tmp_path, monkeypatch):
    from ignition.preparation.rebuild_manifest import from_plan
    from ignition.recipes import resolve
    from .test_declarations import fitted

    model = resolve("kimi-k3.1-pro")
    spec = replace(RECIPE.speculation, method="dspark", draft_head_quantization="nvfp4", markov_head_quantization="fp8")
    plan = fitted(model, 16, concurrency=1, speculation=spec, online_mxfp8=True, online_mxfp8_scope="all")
    calls = []
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda repository, **kw: calls.append((repository, kw["revision"])) or str(tmp_path / "draft"))
    monkeypatch.setattr("ignition.preparation.rank_group.prepare_group", lambda snapshot, config, prepare, **kwargs: config)
    prepared = RECIPE.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None)
    assert calls == [(WINDOW_REPOSITORY, WINDOW_REVISION)]
    assert prepared.speculative_method == "dspark"
    assert prepared.draft_head_quantization == "nvfp4" and prepared.markov_head_quantization == "fp8"
    assert prepared.snapshot_budget_bytes == plan.snapshots * 35_122_432
    settings = from_plan(model, plan, compiled_forwards=True)["speculation"]
    assert (settings["method"], settings["draft_head_quantization"], settings["markov_head_quantization"]) == (
        "dspark", "nvfp4", "fp8")


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA device")
def test_source_draft_shared_vocabulary_and_converted_weights(config, snapshot):
    from safetensors import safe_open
    from ignition.recipes.kimi3.dspark_weights import DraftWeights

    if not (snapshot / "model.safetensors").is_file():
        pytest.skip("Inferact Kimi K3.1 sliding-window DSpark weights are not installed")
    draft = window_draft(json.loads((snapshot / "config.json").read_text()), config.text)
    rank = TensorParallelShard(0, 16)
    embedding = torch.zeros(config.vocabulary, config.hidden // 16, dtype=torch.bfloat16, device="cuda")
    head = torch.ones(config.vocabulary // 16, config.hidden, dtype=torch.bfloat16, device="cuda")
    weights = DraftWeights.load(snapshot, replace(config, draft=draft), "cuda", rank, target_embedding=embedding,
        target_head=head, online_mxfp8=True, draft_head_quantization="nvfp4", markov_head_quantization="fp8")
    assert weights.embedding is embedding
    assert torch.all(head == 1) and head.dtype == torch.bfloat16
    assert (weights.head.out_features, weights.head.in_features) == tuple(head.shape)
    assert (weights.context_kv.out_features, weights.context_kv.in_features) == (640, config.hidden)
    assert all((layer.qkv.out_features, layer.qkv.in_features) == (512, config.hidden) for layer in weights.layers)
    assert weights.markov_head.values.shape == (config.vocabulary, draft.markov_rank)
    with safe_open(snapshot / "model.safetensors", framework="pt") as source:
        assert "embed_tokens.weight" not in source.keys() and "lm_head.weight" not in source.keys()
        rows = torch.stack([source.get_slice("markov_head.markov_w2.weight")[i] for i in (0, 131071, 163839)]).to(
            device=head.device, dtype=torch.float32)
    scale = rows.abs().amax(1).clamp_min(1e-12) / 448.
    assert torch.equal(weights.markov_head.scale[[0, 131071, 163839]], scale)
    assert torch.equal(weights.markov_head.values.view(torch.uint8)[[0, 131071, 163839]],
                       (rows / scale[:, None]).to(torch.float8_e4m3fn).view(torch.uint8))
