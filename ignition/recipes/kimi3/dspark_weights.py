# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sharded Kimi K3 DSpark draft: Qwen3 sliding-window layers, context projections and the Markov head."""

from dataclasses import dataclass

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.persistent import persistent_cat, persistent_copy
from ignition.preparation.rebuild_manifest import weight_container


@weight_container(('qkv', 'output', 'query_norm', 'key_norm', 'input_norm', 'post_attention_norm', 'mlp'))
@dataclass(frozen=True)
class DraftLayerWeights:
    qkv: torch.Tensor
    output: torch.Tensor
    query_norm: torch.Tensor
    key_norm: torch.Tensor
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights


@weight_container(('projection', 'hidden_norm', 'context_kv', 'norm', 'embedding', 'head', 'markov_embedding', 'markov_head', 'layers'))
@dataclass(frozen=True)
class DraftWeights:
    """One rank's draft weights.

    ``projection`` holds this rank's rows of the auxiliary-state combination, gathered across ranks;
    ``context_kv`` stacks every layer's key and value rows, so accepted target states reach all draft
    caches through one projection. The embedding is column-sharded like the target's. The output head
    is the target's when the checkpoint's head is identical, and the Markov transition keeps its
    embedding replicated. The FP8 Markov mode also replicates the transition projection to score the
    full vocabulary locally. The Kimi K3.1 draft shares vocabulary tensors omitted by its checkpoint.
    Its all-projection MXFP8 mode includes the per-layer QKV and accepted-context K/V projections.
    """

    projection: torch.Tensor
    hidden_norm: torch.Tensor
    context_kv: torch.Tensor
    norm: torch.Tensor
    embedding: torch.Tensor
    head: torch.Tensor | None
    markov_embedding: torch.Tensor
    markov_head: torch.Tensor
    layers: tuple[DraftLayerWeights, ...]

    @classmethod
    def load(cls, snapshot, config, device, shard, *, target_head, progress=None, online_mxfp8=False,
             target_embedding=None, draft_head_quantization=None, markov_head_quantization=None):
        from ignition.components.weights import quantize_bf16_linear_as_mxfp8 as mxfp8_rows

        stored = mxfp8_rows if online_mxfp8 else lambda weight: weight
        cat = torch.cat if online_mxfp8 else persistent_cat
        # The Kimi K3.1 all-projection conversion includes both uses of the draft K/V weights.
        convert_qkv = config.dsa is not None and online_mxfp8
        concat_qkv = torch.cat if convert_qkv else persistent_cat
        d = config.draft
        d.validate_shard(shard)
        c, hidden = d.text, config.hidden
        q_rows = shard.partition(d.heads * d.head_dim, alignment=d.head_dim)
        kv_rows = shard.partition(d.kv_heads * d.head_dim, alignment=d.head_dim)
        intermediate = shard.partition(c["intermediate_size"], alignment=32)
        vocabulary = shard.partition(config.vocabulary, alignment=64)
        manifest = inspect_indexed_checkpoint(snapshot)
        layers, context = [], []
        with DirectWeightLoader(manifest, device, progress=progress, progress_label="Load draft weights") as loader:
            def matrix(name, rows=None, columns=None, *, temporary=False):
                value = loader.load_tensor(name + ".weight", extent=(slice(*(rows or (None, None))),
                                                                     slice(*(columns or (None, None)))), temporary=temporary)
                if value.dtype != torch.bfloat16 or value.ndim != 2:
                    raise ValueError(f"Kimi K3 DSpark requires BF16 matrices: {name}")
                return value

            projection = matrix("fc", rows=shard.partition(hidden, alignment=64), temporary=online_mxfp8)
            if projection.shape[1] != len(d.auxiliary_layers) * hidden:
                raise ValueError("DSpark fc must combine one hidden state per auxiliary layer")
            projection = stored(projection)
            hidden_norm, norm = loader.load_tensors(("hidden_norm.weight", "norm.weight"))
            for index in range(d.layers):
                prefix = f"layers.{index}"
                key = matrix(prefix + ".self_attn.k_proj", rows=kv_rows, temporary=True)
                value = matrix(prefix + ".self_attn.v_proj", rows=kv_rows, temporary=True)
                qkv = concat_qkv((matrix(prefix + ".self_attn.q_proj", rows=q_rows, temporary=True), key, value))
                if convert_qkv:
                    qkv = mxfp8_rows(qkv)
                context.extend((key, value))
                output = stored(matrix(prefix + ".self_attn.o_proj", columns=q_rows, temporary=online_mxfp8))
                query_norm, key_norm, input_norm, post_norm = loader.load_tensors(tuple(prefix + name for name in
                    (".self_attn.q_norm.weight", ".self_attn.k_norm.weight", ".input_layernorm.weight",
                     ".post_attention_layernorm.weight")))
                mlp = MLPWeights(stored(cat([matrix(prefix + ".mlp." + name, rows=intermediate, temporary=True)
                                                   for name in ("gate_proj", "up_proj")])),
                                 stored(matrix(prefix + ".mlp.down_proj", columns=intermediate, temporary=online_mxfp8)))
                layers.append(DraftLayerWeights(qkv, output, query_norm, key_norm, input_norm, post_norm, mlp))
            hidden_rows = shard.partition(hidden, alignment=64)
            if target_embedding is None or "embed_tokens.weight" in manifest.tensors:
                embedding = matrix("embed_tokens", columns=hidden_rows)
            else:
                embedding = target_embedding
            if (embedding is None or embedding.dtype != torch.bfloat16
                    or embedding.shape != (config.vocabulary, hidden_rows[1] - hidden_rows[0])):
                raise ValueError("DSpark requires a BF16 embedding shard from the draft or target checkpoint")
            head = target_head
            if target_embedding is None or "lm_head.weight" in manifest.tensors:
                head = matrix("lm_head", rows=vocabulary, temporary=True)
            if (head is None or head.dtype != torch.bfloat16
                    or head.shape != (vocabulary[1] - vocabulary[0], hidden)):
                raise ValueError("DSpark requires a BF16 vocabulary head shard from the draft or target checkpoint")
            if draft_head_quantization is None:
                head = None if head is target_head or torch.equal(head, target_head) else persistent_copy(head)
            else:
                from ignition.components.vocabulary import mxfp8_draft_head, nvfp4_draft_head
                head = {"mxfp8": mxfp8_draft_head, "nvfp4": nvfp4_draft_head}[draft_head_quantization](head)
            markov_embedding = matrix("markov_head.markov_w1")
            replicate = markov_head_quantization == "fp8"
            markov_head = matrix("markov_head.markov_w2", rows=None if replicate else vocabulary, temporary=replicate)
            if markov_embedding.shape[1] != d.markov_rank or markov_head.shape[1] != d.markov_rank:
                raise ValueError("DSpark Markov projections disagree with markov_rank")
            if replicate:
                from ignition.components.markov_sampling import fp8_markov_head
                markov_head = fp8_markov_head(markov_head)
        context_kv = concat_qkv(context)
        if convert_qkv:
            context_kv = mxfp8_rows(context_kv)
        return cls(projection, hidden_norm, context_kv, norm, embedding, head, markov_embedding,
                   markov_head, tuple(layers))
