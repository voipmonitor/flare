# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 attention residuals and latent MoE composed into complete forwards."""

from dataclasses import dataclass

import torch
import triton

from ignition.components.collectives import ConcurrentAllReduce, PreparedChunkedAllGather, PreparedChunkedAllReduce
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.moe import PreparedRouter, PreparedRoutedExperts, RouterScratch
from ignition.components.projection_overlap import ConcurrentProjectionChain
from ignition.components.column_pair_gather import PreparedColumnPairGather
from ignition.components.recurrent import DeltaMetadata, _commit_convolution
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.kda_speculation import KDARecords, PreparedKDAReplay
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.persistent import persistent_copy, persistent_zeros
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import register_finalizer

from .attention import PreparedMLA
from .kda import PreparedKDA
from .pointwise import rms_norm_columns, situ
from .projections import prepare_projection
from .residual import ADD, REPLACE, ResidualStream
from .routing import MAX_ROWS as ROUTE_ROWS, PreparedRouteGather


# A concurrent shared-expert MLP forks onto the auxiliary stream after its input is written and joins before its
# output is added; the explicit mutated outputs order both against the routed branch through compilation.
@torch.library.custom_op("flare::kimi3_start_shared_expert", mutates_args=("outputs", "workspace"))
def _start_shared_expert(source: torch.Tensor, weights: list[torch.Tensor], scales: list[torch.Tensor | None],
                         outputs: list[torch.Tensor], workspace: torch.Tensor, plans: list[int], auxiliary_stream: int) -> None:
    from ignition.components.sm103_projection import bf16_out, packed_out_workspace

    def project(value, index, output):
        if scales[index] is None:
            bf16_out(value, weights[index], output, plans[index])
        else:
            packed_out_workspace(value, weights[index], scales[index], None, None, output,
                                 workspace, "none", None, plans[index], "any")

    main = torch.cuda.current_stream(source.device)
    auxiliary = torch.cuda.ExternalStream(auxiliary_stream, device=source.device)
    auxiliary.wait_stream(main)
    gate_up, intermediate, projected = outputs
    with torch.cuda.stream(auxiliary):
        project(source, 0, gate_up)
        situ(gate_up, intermediate)
        project(intermediate, 1, projected)


@_start_shared_expert.register_fake
def _fake_start_shared_expert(source, weights, scales, outputs, workspace, plans, auxiliary_stream) -> None:
    pass


@torch.library.custom_op("flare::kimi3_join_shared_expert", mutates_args=("outputs", "workspace"))
def _join_shared_expert(source: torch.Tensor, routed: torch.Tensor, outputs: list[torch.Tensor],
                        workspace: torch.Tensor, auxiliary_stream: int) -> None:
    main = torch.cuda.current_stream(source.device)
    main.wait_stream(torch.cuda.ExternalStream(auxiliary_stream, device=source.device))


@_join_shared_expert.register_fake
def _fake_join_shared_expert(source, routed, outputs, workspace, auxiliary_stream) -> None:
    pass


@dataclass(frozen=True)
class DecoderStorage:
    hidden_columns: torch.Tensor
    latent_begin: torch.Tensor
    row_tuning: object = None
    dual_tc_flags: torch.Tensor | None = None

    @classmethod
    def allocate(cls, config, shard, device):
        n_a = config.text.get("routed_expert_hidden_size", 0) // shard.world_size
        n_b = config.text.get("num_experts", 0) // shard.world_size
        flags_n = max(64, (n_a + 15) // 16 + (n_b + 15) // 16)
        return cls(persistent_copy(torch.arange(*shard.partition(config.hidden), device=device)),
            persistent_copy(torch.tensor(shard.partition(config.text["routed_expert_hidden_size"])[:1],
                                         dtype=torch.int32, device=device)),
            dual_tc_flags=persistent_zeros((flags_n,), torch.uint32, device))


@dataclass(frozen=True)
class Scratch:
    activations: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, config, world_size, device, *, examples=None):
        def tensor(role, shape):
            return declaration_tensor(examples, role, shape, torch.bfloat16, device)
        return cls(tensor("kimi.hidden_input", (tokens, config.hidden)),
                   tensor("kimi.hidden", (rows, config.hidden)),
                   tensor("kimi.local_logits", (rows, config.vocabulary // world_size)),
                   tensor("kimi.logits", (rows, config.vocabulary)))

    def rows(self, tokens, rows):
        return type(self)(self.activations[:tokens], self.hidden[:rows], self.local_logits[:rows], self.logits[:rows])


class PreparedMLP:
    """SiTU projections with invocation-local intermediates and explicit side-stream joins."""

    def __init__(self, weights, source, *, session, name, workspace=None, parallel_plans=(), examples=None):
        rows, device = source.shape[0], source.device
        width = (weights["gate_up"].shape[0] if isinstance(weights["gate_up"], torch.Tensor)
                 else weights["gate_up"].out_features)
        examples = {} if examples is None else examples
        def tensor(role, columns):
            key = "kimi.hidden_output" if role == "output" else "kimi.mlp." + role
            return declaration_tensor(examples, key, (rows, columns), source.dtype, device)
        intermediate = tensor("intermediate", width // 2)
        self.gate_up = prepare_projection(source, weights["gate_up"], tensor("gate_up", width),
            session=session, name=name + ".gate_up", workspace=workspace)
        self.down = prepare_projection(intermediate, weights["down"], tensor("output", source.shape[1]),
            session=session, name=name + ".down", workspace=workspace)
        self.intermediate_width = width // 2
        self.concurrent = bool(parallel_plans) and workspace is not None and workspace.overlap
        if self.concurrent:
            self.stream = workspace.auxiliary_stream
            self.stream_handle = self.stream.cuda_stream
            self.weights, self.scales = [], []
            for projection in (self.gate_up, self.down):
                if isinstance(projection, UnquantizedProjection):
                    self.weights.append(projection.weight)
                    self.scales.append(None)
                else:
                    values, scales, _ = projection.parts
                    self.weights.append(values)
                    self.scales.append(scales.view(torch.uint8))
                for main in parallel_plans:
                    workspace.register_parallel(main, projection.plan)
            register_finalizer(session, self.refresh)
        self.gate_up.functional(session=session)
        self.down.functional(session=session)

    def refresh(self):
        self.workspace_nbytes = max(p.plan.memory_requirements().scratch_nbytes
                                   for p in (self.gate_up, self.down))

    def __call__(self, source):
        gate_up = self.gate_up(source)
        intermediate = torch.empty((source.shape[0], self.intermediate_width), dtype=source.dtype, device=source.device)
        situ(gate_up, intermediate)
        return self.down(intermediate)

    def start(self, source):
        outputs = [torch.empty((source.shape[0], width), dtype=source.dtype, device=source.device)
                   for width in (2 * self.intermediate_width, self.intermediate_width, source.shape[1])]
        workspace = torch.empty(self.workspace_nbytes, dtype=torch.uint8, device=source.device)
        _start_shared_expert(source, self.weights, self.scales, outputs, workspace,
            [self.gate_up.plan.handle, self.down.plan.handle], self.stream_handle)
        return source, outputs, workspace

    def finish(self, pending, routed):
        source, outputs, workspace = pending
        _join_shared_expert(source, routed, outputs, workspace, self.stream_handle)
        return outputs[-1]


class PreparedLatentMoE:
    def __init__(self, weights, source, config, shard, *, runtime, router_runtime, coordinator,
                 session, name, workspace, storage, route_runtime=None, prefill_route_runtime=None, examples=None):
        tokens, hidden = source.shape
        c, device = config.text, source.device
        latent = c["routed_expert_hidden_size"]
        examples = {} if examples is None else examples
        self.pipelined = (
            prefill_route_runtime is not None
            and tokens == 2048
            and workspace is not None
            and workspace.overlap
            and runtime is not None
        )
        work_tokens = tokens // 2 if self.pipelined else tokens
        work_source = source[:work_tokens]

        def buffer(role, width, dtype=torch.bfloat16):
            key = "kimi.hidden_output" if role == "output" else "kimi.moe." + role
            return declaration_tensor(examples, key, (work_tokens, width), dtype, device)

        self.hidden_columns, self.latent_begin = storage.hidden_columns, storage.latent_begin
        self.world, self.latent_width = shard.world_size, latent
        self.latent_norm, self.eps = weights["latent_norm"], c["rms_norm_eps"]
        latent_example, expert_output = buffer("latent", latent), buffer("expert_output", latent)
        up_input = buffer("up_input", latent // shard.world_size)
        self.up = prepare_projection(up_input, weights["latent_up"], buffer("output", hidden),
            session=session, name=name + ".latent_up", workspace=workspace).functional(session=session)
        self.expert_reduce = PreparedChunkedAllReduce(expert_output, buffer("latent_sum", latent),
            runtime=runtime, coordinator=coordinator, session=session,
            name=name + ".expert_reduce")
        self.row_norm = None
        if 1 <= work_tokens <= 64 and latent == 3584 and storage.row_tuning is not None:
            from .row_reductions import RowGeometry
            self.row_norm = storage.row_tuning.register(self.expert_reduce, RowGeometry(latent, rows=work_tokens))
        local_logits = buffer("local_router", weights["router"].shape[0], torch.float32)
        top_k = c["num_experts_per_token"]
        routes = RouterScratch(buffer("selected_logits", top_k, torch.float32),
            buffer("route_ids", top_k, torch.int32),
            buffer("route_weights", top_k, torch.float32).fill_(1 / top_k))
        self.route_gather = self.prefill_route_gather = None
        self.dual_down_router = None
        use_dual = False
        if route_runtime is not None and work_tokens <= ROUTE_ROWS:
            import importlib.util
            from ignition.components import sm103_projection
            w_down = weights["latent_down_rows"]
            w_router = weights["router"]
            use_dual = (
                1 <= work_tokens <= 32
                and hidden % 64 == 0
                and device.type == "cuda"
                and not sm103_projection.is_sm103(device)
                and work_source.dtype == torch.bfloat16
                and isinstance(w_down, torch.Tensor)
                and isinstance(w_router, torch.Tensor)
                and w_down.dtype == torch.bfloat16
                and w_router.dtype == torch.bfloat16
                and w_down.is_contiguous()
                and w_router.is_contiguous()
                and w_down.data_ptr() % 16 == 0
                and w_router.data_ptr() % 16 == 0
                and importlib.util.find_spec("b12x.gemm.bf16_gemv._dual_tc") is not None
            )
        if use_dual:
            from b12x.gemm.bf16_gemv._dual_tc import (
                compile_dual_tc_projection,
                dual_tc_flags_numel,
                dual_tc_scratch_numel,
                dual_tc_split_k,
            )
            w_down = weights["latent_down_rows"]
            w_router = weights["router"]
            record_numeric_tensor(w_down)
            record_numeric_tensor(w_router)
            n_a, n_b = int(w_down.shape[0]), int(w_router.shape[0])
            split_k = dual_tc_split_k(n_a, n_b, hidden, work_tokens)
            flags_needed = dual_tc_flags_numel(n_a, n_b)
            flags = storage.dual_tc_flags
            if flags is None or flags.numel() < flags_needed or flags.device != device:
                flags = persistent_zeros((flags_needed,), torch.uint32, device)
            # Precompile on the active CUDA context so native export captures the cubin.
            self._dual_program = compile_dual_tc_projection(
                int(device.index or 0), work_tokens, split_k, n_a, n_b, hidden
            )
            self.dual_down_router = (
                w_down,
                w_router,
                flags,
                n_a,
                n_b,
                work_tokens,
                split_k,
                dual_tc_scratch_numel(n_a, n_b, work_tokens, split_k),
            )
            self.router_projection = self.down = None
            down_output = buffer("local_down", latent // shard.world_size)
            self.route_gather = PreparedRouteGather(down_output, local_logits, weights["correction"],
                latent_example, routes.weights, routes.ids, runtime=route_runtime, session=session,
                coordinator=coordinator, name=name + ".route_gather")
        else:
            self.router_projection = UnquantizedProjection(work_source, weights["router"], local_logits,
                session=session, name=name + ".router").functional(session=session)
            if route_runtime is not None and work_tokens <= ROUTE_ROWS:
                down_output = buffer("local_down", latent // shard.world_size)
                self.down = prepare_projection(work_source, weights["latent_down_rows"], down_output,
                    session=session, name=name + ".latent_down_rows", workspace=workspace)
                self.route_gather = PreparedRouteGather(down_output, local_logits, weights["correction"],
                    latent_example, routes.weights, routes.ids, runtime=route_runtime, session=session,
                    coordinator=coordinator, name=name + ".route_gather")
            else:
                logits = buffer("router", c["num_experts"], torch.float32)
                if prefill_route_runtime is not None:
                    down_output = buffer("local_down", latent // shard.world_size)
                    self.down = prepare_projection(work_source, weights["latent_down_rows"], down_output,
                        session=session, name=name + ".latent_down_rows", workspace=workspace)
                    self.prefill_route_gather = PreparedColumnPairGather(down_output, local_logits, latent_example, logits,
                        runtime=prefill_route_runtime, session=session, coordinator=coordinator,
                        name=name + ".latent_router_gather",
                        buffers=examples.setdefault("kimi.moe.paired_columns", {}))
                else:
                    self.down = prepare_projection(buffer("source", hidden // shard.world_size), weights["latent_down"],
                        buffer("down", latent), session=session, name=name + ".latent_down", workspace=workspace)
                    self.down_reduce = PreparedChunkedAllReduce(self.down.output, latent_example, runtime=runtime,
                        coordinator=coordinator, session=session, name=name + ".down_reduce").functional(session=session)
                    self.local_experts = local_logits.shape[1]
                    self.router_width = (self.local_experts + 31) // 32 * 32
                    router_input = buffer("router_input", self.router_width, torch.float32)
                    gathered = buffer("gathered_router", self.router_width * shard.world_size, torch.float32)
                    self.router_gather = PreparedChunkedAllGather(router_input.view(torch.bfloat16), gathered.view(torch.bfloat16),
                        runtime=router_runtime, coordinator=coordinator, session=session,
                        name=name + ".router_gather").functional(session=session)
                self.router = PreparedRouter(logits, routes, score_func="sigmoid", correction_bias=weights["correction"],
                    renormalize=True, routed_scaling_factor=c["routed_scaling_factor"], session=session,
                    name=name + ".route").functional(session=session)
        self.experts = PreparedRoutedExperts(latent_example, weights["experts"], routes.ids, routes.weights,
            expert_output, workspace=workspace, session=session, name=name + ".experts").functional(session=session)
        shared_parallel = ()
        if self.route_gather is not None or (self.prefill_route_gather is not None and not self.pipelined):
            shared_parallel = ((self.experts.plan, self.up.plan) if self.dual_down_router is not None
                               else (self.router_projection.plan, self.down.plan, self.experts.plan, self.up.plan))
        self.shared = PreparedMLP(weights["shared"], work_source, session=session, name=name + ".shared", workspace=workspace,
            parallel_plans=shared_parallel, examples=examples)
        self.concurrent_expert_reduce = None
        self.all_plans = ()
        if self.pipelined and len(self.expert_reduce.parts) == 1:
            self.all_plans = (
                self.router_projection.plan,
                self.down.plan,
                self.router.plan,
                self.experts.plan,
                self.up.plan,
                self.shared.gate_up.plan,
                self.shared.down.plan,
            )
            self.prefill_route_gather.enable_concurrent(
                workspace=workspace, session=session, parallel_plans=self.all_plans)
            self.concurrent_expert_reduce = ConcurrentAllReduce(
                self.expert_reduce.parts[0], workspace=workspace, session=session,
                parallel_plans=self.all_plans)
        else:
            self.pipelined = False
        self.expert_reduce.functional(session=session)
        if self.concurrent_expert_reduce is not None:
            self.concurrent_expert_reduce.functional(session=session)
        if self.prefill_route_gather is not None:
            self.prefill_route_gather.functional(session=session)
        self.concurrent_down = (ConcurrentProjectionChain((self.down,), parallel_plans=(self.router_projection.plan,),
            session=session, workspace=workspace).functional(session=session)
            if self.dual_down_router is None and self.route_gather is not None and workspace is not None and workspace.overlap else None)
        if self.dual_down_router is None and self.concurrent_down is None:
            self.down.functional(session=session)

    def __call__(self, source, *, fuse_add=False):
        dual = getattr(self, "dual_down_router", None)
        if dual is not None:
            from b12x.gemm.bf16_gemv._dual_tc import bf16_dual_gemv_out
            w_down, w_router, flags, n_a, n_b, max_rows, split_k, scratch_numel = dual
            down = torch.empty((source.shape[0], n_a), dtype=torch.bfloat16, device=source.device)
            logits = torch.empty((source.shape[0], n_b), dtype=torch.float32, device=source.device)
            scratch = torch.empty((scratch_numel,), dtype=torch.float32, device=source.device)
            bf16_dual_gemv_out(source, w_down, w_router, down, logits, scratch, flags, max_rows, split_k)
        else:
            pending_down = None if self.concurrent_down is None else self.concurrent_down.start(source)
            logits = self.router_projection(source)
            if self.route_gather is not None:
                down = (self.down(source) if self.concurrent_down is None else
                        self.concurrent_down.finish(pending_down, logits))
            elif self.prefill_route_gather is not None:
                down = self.down(source)
        pending_shared = self.shared.start(source) if self.shared.concurrent else None
        if self.route_gather is not None:
            latent, ids, route_weights = self.route_gather(down, logits)
        else:
            if self.prefill_route_gather is not None:
                latent, logits = self.prefill_route_gather(down, logits)
            else:
                router_input = torch.zeros((source.shape[0], self.router_width), dtype=torch.float32, device=source.device)
                router_input[:, :self.local_experts].copy_(logits)
                gathered = self.router_gather(router_input.view(torch.bfloat16)).view(torch.float32)
                logits = gathered.view(source.shape[0], self.world, self.router_width)[:, :, :self.local_experts].reshape(source.shape[0], -1)
                latent = self.down_reduce(self.down(source.index_select(1, self.hidden_columns)))
            routes = self.router(logits)
            ids, route_weights = routes.ids, routes.weights
        partial = self.experts(latent, ids, route_weights)
        if self.row_norm is None:
            latent_sum = self.expert_reduce(partial)
            up_input = torch.empty((source.shape[0], self.latent_width // self.world), dtype=source.dtype, device=source.device)
            rms_norm_columns(latent_sum, self.latent_norm, up_input, self.eps, begin=self.latent_begin)
        else:
            up_input = self.row_norm.norm(partial, self.latent_norm, self.latent_begin, self.eps)
        output = self.up(up_input)
        shared = self.shared.finish(pending_shared, output) if self.shared.concurrent else self.shared(source)
        if fuse_add:
            return output, shared
        return output + shared


class PreparedDecoder:
    """Compose Kimi layers with local residuals and verification records."""

    def __init__(self, weights, scratch, metadata, pool, *, mode, runtime, vocabulary_runtime,
                 embedding_runtime, router_runtime, vision_state, coordinator, session, name,
                 workspace, buffers, kda_records=None, route_runtime=None, context=None,
                 latent_runtimes=(), scoring=None, prefill_route_runtime=None, examples=None):
        if mode not in ("prefill", "decode", "verify") or mode == "verify" and kda_records is None:
            raise ValueError("Kimi verification requires candidate recording storage")
        self.weights, self.scratch, self.pool = weights, scratch, pool
        self.mode = mode
        self.vision_state = vision_state if mode == "prefill" else None
        config, source = weights.config, scratch.activations
        tokens, hidden = source.shape
        self.tokens, self.hidden = tokens, hidden
        self.shared_index_type, self.index_shards = None, None
        if mode in ("decode", "verify") and context is not None and "dsa" in config.layers:
            from ignition.recipes.kimi31.indexer import SharedIndexMetadata

            self.shared_index_type, self.index_shards = SharedIndexMetadata, context.shards
        self.sequences = metadata.starts.numel()
        self.verification_columns = 8 if mode == "verify" else 1
        examples = {} if examples is None else examples
        local_embedding = declaration_tensor(examples, "embedding", (tokens, weights.embedding.shape[1]), source.dtype, source.device)
        self.embedding = PreparedShardedEmbedding(weights.embedding, metadata.token_ids, local_embedding, source,
            runtime=embedding_runtime, coordinator=coordinator, num_rows=metadata.live_counts[:1],
            session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        delta = DeltaMetadata.allocate(self.sequences, source.device, verification_columns=self.verification_columns)
        self.layers = []
        self.residual_blocks = -(-len(weights.layers) // config.text["attn_res_block_size"])
        self.auxiliary_layers = config.draft.auxiliary_layers if weights.draft is not None else ()
        self.auxiliary_stream = config.draft.auxiliary_stream if self.auxiliary_layers else None
        self.auxiliary = (None if not self.auxiliary_layers else declaration_tensor(examples, "auxiliary",
            (tokens, len(self.auxiliary_layers) * hidden), source.dtype, source.device))
        for index, w in enumerate(weights.layers):
            prefix, kind = f"{name}.layer{index}", config.layers[index]
            kwargs = dict(mode=mode, session=session, name=prefix + ".attention", workspace=workspace,
                          buffers=examples)
            if kind == "kda":
                attention = PreparedKDA(w["attention"], source, pool.recurrent(index), delta,
                    eps=config.text["rms_norm_eps"], records=kda_records.layer(index) if mode == "verify" else None,
                    **kwargs)
            else:
                attention_type, extra = PreparedMLA, {}
                if kind == "dsa":
                    from ignition.recipes.kimi31.attention import PreparedDSA
                    attention_type, extra = PreparedDSA, dict(index_cache=pool.indexer_pages(index))
                attention = attention_type(w["attention"], source, metadata, pool.attention(index),
                    eps=config.mla_norm_eps, context=context,
                    peers=None if context is None else context.peers[index], coordinator=coordinator,
                    latents=next((r for r in latent_runtimes if tokens <= r.max_batch_size),
                                 latent_runtimes[-1] if latent_runtimes else None), **extra, **kwargs)
            if "mlp" in w:
                mlp = PreparedMLP(w["mlp"], source, session=session, name=prefix + ".mlp", workspace=workspace, examples=examples)
            else:
                mlp = PreparedLatentMoE(w, source, config, weights.shard, runtime=runtime,
                    router_runtime=router_runtime, route_runtime=route_runtime if mode != "prefill" else None,
                    prefill_route_runtime=prefill_route_runtime if mode == "prefill" else None,
                    coordinator=coordinator, session=session, name=prefix + ".moe", workspace=workspace, storage=buffers, examples=examples)
            total = declaration_tensor(examples, "kimi.hidden_output", source.shape, source.dtype, source.device)
            reduce = PreparedChunkedAllReduce(source, total, runtime=runtime, coordinator=coordinator,
                session=session, name=prefix + ".reduce")
            concurrent_reduce = None
            if (mode == "prefill" and tokens == 2048 and isinstance(mlp, PreparedLatentMoE)
                    and mlp.pipelined and len(reduce.parts) == 2 and workspace is not None and workspace.overlap):
                concurrent_reduce = (
                    ConcurrentAllReduce(reduce.parts[0], workspace=workspace, session=session,
                        parallel_plans=mlp.all_plans),
                    ConcurrentAllReduce(reduce.parts[1], workspace=workspace, session=session,
                        parallel_plans=mlp.all_plans),
                )
            reduce.functional(session=session)
            if concurrent_reduce is not None:
                concurrent_reduce[0].functional(session=session)
                concurrent_reduce[1].functional(session=session)
            self.layers.append((w, attention, mlp, reduce, concurrent_reduce))
        self.row_mixes = None
        if 1 <= tokens <= 64 and hidden == 7168 and buffers.row_tuning is not None:
            from .row_reductions import hidden_row_choices
            self.row_mixes = hidden_row_choices(buffers.row_tuning,
                [(w, a, m, r) for w, a, m, r, _ in self.layers], config.text["attn_res_block_size"],
                self.auxiliary_layers if self.auxiliary_stream == "attn_res" else (), self.residual_blocks, rows=tokens)
        self.scoring = None if scoring is None else scoring.prepare(source, weights.head,
            vocabulary=config.text["vocab_size"], session=session, coordinator=coordinator, name=name + ".prompt")
        self.compact_head = scratch.hidden.shape[0] > tokens and scoring is None
        head_rows = tokens if self.compact_head else scratch.hidden.shape[0]
        self.head = PreparedVocabularyProjection(scratch.hidden[:head_rows], weights.head,
            scratch.local_logits[:head_rows], scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            row_indices=metadata.sample_indices if self.compact_head else None).functional(
            session=session, dynamic_indices=True)
        self.replay = None
        if mode == "verify":
            indices = kda_records.layers
            states = [pool.recurrent(i).recurrent[:kda_records.corrections.shape[1]] for i in indices]
            self.record_layers = indices
            self.record_shapes = (tuple(kda_records.corrections.shape), tuple(kda_records.keys_and_gates.shape))
            self.record_constants = (kda_records.A_log, kda_records.dt_bias)
            self.replay = PreparedKDAReplay(pool.backing, states, kda_records, self.layers[indices[0]][1].plan,
                session=session).functional()
            regions = [pool.layout.region(f"layer{i}.convolution") for i in indices]
            self.convolution_offsets = persistent_copy(torch.tensor([r.offset // 2 for r in regions],
                dtype=torch.int64, device=source.device))
            self.convolution_shape = regions[0].shape

    def release_outputs(self):
        self.scratch = self.auxiliary = None

    def commit(self, counts, verification, metadata):
        records, delta = verification
        self.replay(counts, records, delta)
        channels, columns = self.convolution_shape
        torch.library.wrap_triton(_commit_convolution)[
            (self.convolution_offsets.numel(), counts.numel(), triton.cdiv(channels, 128))](
            self.pool.backing.view(torch.bfloat16), self.convolution_offsets, metadata.state_rows, counts,
            self.pool.layout.row_bytes // 2, channels, columns, 128)

    def __call__(self, metadata):
        m, c = metadata, self.weights.config.text
        d = DeltaMetadata.allocate(self.sequences, m.token_ids.device, verification_columns=self.verification_columns)
        active = m.lengths > 0
        d.query_start_loc.copy_(m.query_start_loc)
        d.convolution_slots.copy_(torch.where(active, m.state_rows, -1))
        d.fresh.copy_(m.starts == 0)
        d.state_indices[:, 0].copy_(m.state_rows)
        d.initial_state_indices.copy_(torch.where(m.starts > 0, m.state_rows, 0))
        d.final_state_indices.copy_(m.state_rows)
        d.num_tokens.copy_(m.live_counts[:1])
        d.num_seqs.copy_(m.live_counts[1:])
        shared_index = (None if self.shared_index_type is None else
                        self.shared_index_type.create(m, self.index_shards, self.tokens))
        records = None
        if self.replay is not None:
            records = KDARecords(self.record_layers,
                torch.zeros(self.record_shapes[0], dtype=torch.float32, device=m.token_ids.device),
                torch.zeros(self.record_shapes[1], dtype=torch.bfloat16, device=m.token_ids.device), *self.record_constants)
        local = self.embedding.lookup(m.token_ids, num_rows=m.live_counts[:1])
        if self.vision_state is not None:
            self.vision_state.inject(m, local)
        size, eps = c["attn_res_block_size"], c["rms_norm_eps"]
        delta, mode = self.embedding.gather(local), REPLACE
        preceding, preceding_addend = None, None
        prefix = torch.zeros_like(delta)
        blocks = torch.zeros((self.tokens, self.residual_blocks, self.hidden), dtype=delta.dtype, device=delta.device)
        residual = ResidualStream(prefix, blocks, torch.empty_like(delta))
        auxiliary = (None if not self.auxiliary_layers else
            torch.empty((self.tokens, len(self.auxiliary_layers) * self.hidden), dtype=delta.dtype, device=delta.device))
        for index, (w, attention, mlp, reduce, concurrent_reduce) in enumerate(self.layers):
            committed, boundary = -(-index // size), index % size == 0
            captured = None
            if index in self.auxiliary_layers:
                column = self.auxiliary_layers.index(index) * self.hidden
                captured = auxiliary[:, column:column + self.hidden]
            hidden = residual.mix(delta, mode, sources=committed, write=index // size if boundary else -1,
                score=w["attention_res_score"], norm=w["input_layernorm"], eps=eps, output_eps=eps,
                mixture=captured if self.auxiliary_stream == "attn_res" else None, output=torch.empty_like(delta),
                reduction_choice=preceding, addend=preceding_addend)
            if captured is not None and self.auxiliary_stream == "prefix":
                captured.copy_(residual.prefix)
            if self.weights.config.layers[index] == "kda":
                attended = attention(hidden, d, records=None if records is None else records.layer(index))
            elif self.weights.config.layers[index] == "dsa":
                attended = attention(hidden, m, shared_index=shared_index)
            else:
                attended = attention(hidden, m)
            after_attention, preceding = (None, None) if self.row_mixes is None else self.row_mixes[index]
            if concurrent_reduce is not None:
                # 2-half (1024 + 1024) SM<->PCIe pipeline: overlaps attn_reduce(B), route_gather(A/B),
                # expert_reduce(A/B), and mlp_reduce(A) with independent main-stream compute (strictly 1 collective in flight).
                h = self.tokens // 2
                cr0, cr1 = concurrent_reduce
                att_mode = REPLACE if boundary else ADD
                att_sources = committed + boundary
                att_A = reduce.parts[0](attended[:h])
                p_att_B = cr1.start(attended[h:], dependencies=(att_A,))
                hidden_A = ResidualStream(residual.prefix[:h], residual.blocks[:h],
                    torch.empty((h, self.hidden), dtype=attended.dtype, device=attended.device)).mix(
                    att_A, att_mode, sources=att_sources, write=-1,
                    score=w["mlp_res_score"], norm=w["post_attention_layernorm"], eps=eps, output_eps=eps)
                logits_A = mlp.router_projection(hidden_A)
                down_A = mlp.down(hidden_A)
                shared_A = mlp.shared(hidden_A)
                att_B = cr1.finish(p_att_B, shared_A)

                p_rg_A = mlp.prefill_route_gather.start(down_A, logits_A, dependencies=(att_B,))
                hidden_B = ResidualStream(residual.prefix[h:], residual.blocks[h:],
                    torch.empty((h, self.hidden), dtype=attended.dtype, device=attended.device)).mix(
                    att_B, att_mode, sources=att_sources, write=-1,
                    score=w["mlp_res_score"], norm=w["post_attention_layernorm"], eps=eps, output_eps=eps)
                logits_B = mlp.router_projection(hidden_B)
                down_B = mlp.down(hidden_B)
                latent_A, full_logits_A = mlp.prefill_route_gather.finish(p_rg_A, down_B)

                p_rg_B = mlp.prefill_route_gather.start(down_B, logits_B, dependencies=(latent_A,))
                routes_A = mlp.router(full_logits_A)
                partial_A = mlp.experts(latent_A, routes_A.ids, routes_A.weights)
                latent_B, full_logits_B = mlp.prefill_route_gather.finish(p_rg_B, partial_A)

                p_er_A = mlp.concurrent_expert_reduce.start(partial_A, dependencies=(latent_B,))
                routes_B = mlp.router(full_logits_B)
                partial_B = mlp.experts(latent_B, routes_B.ids, routes_B.weights)
                latent_sum_A = mlp.concurrent_expert_reduce.finish(p_er_A, partial_B)

                p_er_B = mlp.concurrent_expert_reduce.start(partial_B, dependencies=(latent_sum_A,))
                up_in_A = torch.empty((h, mlp.latent_width // mlp.world), dtype=attended.dtype, device=attended.device)
                rms_norm_columns(latent_sum_A, mlp.latent_norm, up_in_A, mlp.eps, begin=mlp.latent_begin)
                out_A = mlp.up(up_in_A) + shared_A
                shared_B = mlp.shared(hidden_B)
                latent_sum_B = mlp.concurrent_expert_reduce.finish(p_er_B, shared_B)

                p_mr_A = cr0.start(out_A, dependencies=(latent_sum_B,))
                up_in_B = torch.empty((h, mlp.latent_width // mlp.world), dtype=attended.dtype, device=attended.device)
                rms_norm_columns(latent_sum_B, mlp.latent_norm, up_in_B, mlp.eps, begin=mlp.latent_begin)
                out_B = mlp.up(up_in_B) + shared_B
                delta_A = cr0.finish(p_mr_A, out_B)

                delta = torch.empty((self.tokens, self.hidden), dtype=attended.dtype, device=attended.device)
                delta[:h].copy_(delta_A)
                reduce.parts[1](out_B, delta[h:], dependencies=(delta_A,))
                preceding_addend = None
                mode = ADD
                continue
            if after_attention is None:
                attended = reduce(attended)
            hidden = residual.mix(attended, REPLACE if boundary else ADD, sources=committed + boundary, write=-1,
                score=w["mlp_res_score"], norm=w["post_attention_layernorm"], eps=eps, output_eps=eps,
                output=torch.empty_like(delta), reduction_choice=after_attention)
            if preceding is not None and preceding.fused and isinstance(mlp, PreparedLatentMoE):
                delta, preceding_addend = mlp(hidden, fuse_add=True)
            else:
                delta, preceding_addend = mlp(hidden), None
            mode = ADD
            if preceding is None:
                delta = reduce(delta)
        hidden = residual.mix(delta, mode, sources=-(-len(self.layers) // size), write=-1,
            score=self.weights.output_res_score, norm=self.weights.norm, eps=eps, output_eps=eps,
            output=torch.empty_like(delta), reduction_choice=preceding, addend=preceding_addend)
        if self.scoring is not None:
            self.scoring(hidden)
        if self.compact_head:
            logits = self.head(hidden, row_indices=m.sample_indices)
            sampled = hidden[m.sample_indices]
            return sampled, logits, auxiliary, (records, d)
        sampled = hidden[m.sample_indices]
        return sampled, self.head(sampled), auxiliary, (records, d)
