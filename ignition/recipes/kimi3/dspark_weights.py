# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sharded Kimi K3 DSpark draft: Qwen3 sliding-window layers, context projections and the Markov head."""

from dataclasses import dataclass

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint


@dataclass(frozen=True)
class DraftLayerWeights:
    qkv: torch.Tensor
    output: torch.Tensor
    query_norm: torch.Tensor
    key_norm: torch.Tensor
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights


@dataclass(frozen=True)
class DraftWeights:
    """One rank's draft weights.

    ``projection`` holds this rank's rows of the auxiliary-state combination, gathered across ranks;
    ``context_kv`` stacks every layer's key and value rows, so accepted target states reach all draft
    caches through one projection. The embedding is column-sharded like the target's. Checkpoint
    vocabulary tensors are optional because some drafts share both target tensors. The Markov transition
    keeps its embedding replicated; its projection is vocabulary-sharded unless `replicated_markov` selects
    the fused single-request sampler. A plan storing
    online MXFP8 holds the combination, QKV, context K/V, attention outputs and MLPs as MXFP8;
    embeddings, normalization and the Markov transition keep the checkpoint's BF16 unless separately converted.
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
    def load(cls, snapshot, config, device, shard, *, target_embedding=None, target_head, progress=None,
             online_mxfp8=False, replicated_markov=False):
        from ignition.components.weights import quantize_bf16_linear_as_mxfp8 as mxfp8_rows

        stored = mxfp8_rows if online_mxfp8 else lambda weight: weight
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
            def matrix(name, rows=None, columns=None):
                value = loader.load_tensor(name + ".weight", extent=(slice(*(rows or (None, None))),
                                                                     slice(*(columns or (None, None)))))
                if value.dtype != torch.bfloat16 or value.ndim != 2:
                    raise ValueError(f"Kimi K3 DSpark requires BF16 matrices: {name}")
                return value

            projection = matrix("fc", rows=shard.partition(hidden, alignment=64))
            if projection.shape[1] != len(d.auxiliary_layers) * hidden:
                raise ValueError("DSpark fc must combine one hidden state per auxiliary layer")
            projection = stored(projection)
            hidden_norm, norm = loader.load_tensors(("hidden_norm.weight", "norm.weight"))
            for index in range(d.layers):
                prefix = f"layers.{index}"
                key = matrix(prefix + ".self_attn.k_proj", rows=kv_rows)
                value = matrix(prefix + ".self_attn.v_proj", rows=kv_rows)
                qkv = stored(torch.cat((matrix(prefix + ".self_attn.q_proj", rows=q_rows), key, value)))
                context.extend((key, value))
                output = stored(matrix(prefix + ".self_attn.o_proj", columns=q_rows))
                query_norm, key_norm, input_norm, post_norm = loader.load_tensors(tuple(prefix + name for name in
                    (".self_attn.q_norm.weight", ".self_attn.k_norm.weight", ".input_layernorm.weight",
                     ".post_attention_layernorm.weight")))
                mlp = MLPWeights(stored(torch.cat([matrix(prefix + ".mlp." + name, rows=intermediate)
                                                   for name in ("gate_proj", "up_proj")])),
                                 stored(matrix(prefix + ".mlp.down_proj", columns=intermediate)))
                layers.append(DraftLayerWeights(qkv, output, query_norm, key_norm, input_norm, post_norm, mlp))
            hidden_rows = shard.partition(hidden, alignment=64)
            embedding = target_embedding
            if "embed_tokens.weight" in manifest.tensors:
                embedding = matrix("embed_tokens", columns=hidden_rows)
            if embedding is None or embedding.dtype != torch.bfloat16 \
                    or tuple(embedding.shape) != (config.vocabulary, hidden_rows[1] - hidden_rows[0]):
                raise ValueError("DSpark requires a compatible checkpoint or target embedding")
            head = None
            if "lm_head.weight" in manifest.tensors:
                head = matrix("lm_head", rows=vocabulary)
                head = None if torch.equal(head, target_head) else head
            if head is None:
                expected = (vocabulary[1] - vocabulary[0], hidden)
                if target_head is None or target_head.dtype != torch.bfloat16 \
                        or tuple(target_head.shape) != expected:
                    raise ValueError("DSpark requires a compatible target vocabulary head")
            markov_embedding = matrix("markov_head.markov_w1")
            markov_head = matrix("markov_head.markov_w2", rows=None if replicated_markov else vocabulary)
            if markov_embedding.shape[1] != d.markov_rank or markov_head.shape[1] != d.markov_rank:
                raise ValueError("DSpark Markov projections disagree with markov_rank")
        return cls(projection, hidden_norm, stored(torch.cat(context)), norm, embedding, head, markov_embedding,
                   markov_head, tuple(layers))
