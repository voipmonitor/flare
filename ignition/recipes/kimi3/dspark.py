# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 DSpark: accepted-state context, sliding-window Qwen3 proposals and the Markov transition.

Accepted target positions project their auxiliary AttnRes states once into every draft layer's key
and value window. A proposal block holds the pending token followed by mask tokens; the five causal
sliding-window Qwen3 layers predict one draft per block position, and a low-rank transition from the
previously sampled token biases each position's logits while drafts are sampled left to right.
"""

from dataclasses import dataclass, replace
from functools import partial

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm

from ignition.components.collectives import PreparedAllGather, PreparedAllReduce, PreparedChunkedAllGather
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.speculation import SpeculativeRngState
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.persistent import persistent_copy, persistent_zeros

from .pointwise import rms_norm_columns


def canonical_norm(weight, eps):
    with torch.device("meta"):
        module = Qwen3RMSNorm(weight.numel(), eps=eps)
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module.eval()


class _FastRMSNorm(torch.nn.Module):
    """Single-launch RMSNorm matching Qwen3RMSNorm's exact BF16 rounding points."""

    def __init__(self, weight, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.eps = float(eps)
        self.fallback = canonical_norm(weight, eps)

    def forward(self, hidden_states):
        if (
            hidden_states.ndim == 2
            and hidden_states.is_contiguous()
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.shape[1] == self.weight.numel()
            and hidden_states.shape[1] <= 8192
        ):
            out = torch.empty_like(hidden_states)
            return rms_norm_columns(hidden_states, self.weight, out, self.eps)
        return self.fallback(hidden_states)


def inverse_frequencies(config, device):
    d = config.draft
    return 1.0 / (d.theta ** (torch.arange(0, d.head_dim, 2, dtype=torch.float32, device=device) / d.head_dim))


def rotate(values, positions, inverse):
    """Qwen3's full-head rotary embedding: FP32 angles, BF16 cosine and sine, rotate-half pairing."""
    frequencies = positions.clamp_min(0).float()[:, None] * inverse[None]
    angles = torch.cat((frequencies, frequencies), -1)
    cos, sin = angles.cos().to(values.dtype), angles.sin().to(values.dtype)
    left, right = values.chunk(2, -1)
    return values * cos[:, None] + torch.cat((-right, left), -1) * sin[:, None]


def stage_windows(history, values, output, metadata):
    """Linearize each request's window: committed history from its ring, then the block's own rows."""
    m = metadata
    rows, window = m.starts.numel(), history.shape[1]
    capacity = output.shape[0] * output.shape[1] // rows
    offsets = torch.arange(capacity, device=values.device)[None, :]
    prefix = m.starts.clamp(max=window)[:, None]
    positions = m.starts[:, None] - prefix + offsets
    previous = history[m.state_rows[:, None], positions.remainder(window)]
    indices = (m.query_start_loc[:-1, None] + offsets - prefix).clamp(0, values.shape[0] - 1)
    current = values[indices]
    valid = (m.lengths[:, None] > 0) & (offsets < prefix + m.lengths[:, None])
    output.view(rows, capacity, *values.shape[1:]).copy_(torch.where(valid[:, :, None, None],
        torch.where((offsets < prefix)[:, :, None, None], previous, current), 0))


def projection(source, weight, output, *, session, name, workspace):
    """A BF16 matrix projects as stored; online MXFP8 weights keep BF16 activations, like their BF16 inputs."""
    if isinstance(weight, torch.Tensor):
        return UnquantizedProjection(source, weight, output, session=session, name=name)
    return QuantizedProjection(source, weight, output, session=session, name=name, activation_mode="a16",
                               workspace=workspace)


def local_heads(config, shard):
    d = config.draft
    return d.heads // shard.world_size, d.kv_heads // shard.world_size


# b12x paged_decode supports native 64-wide heads directly (with split-KV), avoiding
# zero-padding to 128, halving KV staging traffic, and eliminating query rescaling & unpadding.
PADDED_HEAD = 64


@triton.jit
def _bf16_rot_pair(fa, fc, fb, fs):
    # Canonical eager BF16 rotation: round each product to BF16 before adding (no FMA contraction).
    return tl.inline_asm_elementwise(
        asm="""{
          .reg .b16 p1, p2;
          .reg .f32 m1, m2, total;
          mul.rn.f32 m1, $1, $2;
          cvt.rn.bf16.f32 p1, m1;
          cvt.f32.bf16 m1, p1;
          mul.rn.f32 m2, $3, $4;
          cvt.rn.bf16.f32 p2, m2;
          cvt.f32.bf16 m2, p2;
          add.rn.f32 total, m1, m2;
          cvt.rn.bf16.f32 $0, total;
        }""",
        constraints="=h,f,f,f,f",
        args=[fa, fc, fb, fs],
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    ).to(tl.bfloat16, bitcast=True)


@triton.jit
def _draft_qkv_prep(
    QKV,
    QNorm,
    KNorm,
    Pos,
    Inv,
    QueryOut,
    KeyOut,
    qkv_stride,
    q_out_stride,
    k_out_stride,
    eps,
    query_scale,
    Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    half: tl.constexpr = HEAD_DIM // 2
    j = tl.arange(0, half)
    pos = tl.maximum(tl.load(Pos + row), 0).to(tl.float32)
    inv = tl.load(Inv + j).to(tl.float32)
    angles = pos * inv
    cos_bf = libdevice.cos(angles).to(tl.bfloat16)
    sin_bf = libdevice.sin(angles).to(tl.bfloat16)
    c = cos_bf.to(tl.float32)
    s = sin_bf.to(tl.float32)
    if h < Q_HEADS:
        src = QKV + row * qkv_stride + h.to(tl.int64) * HEAD_DIM
        x_l = tl.load(src + j).to(tl.float32)
        x_r = tl.load(src + half + j).to(tl.float32)
        var = (tl.sum(x_l * x_l) + tl.sum(x_r * x_r)) * (1.0 / HEAD_DIM)
        rstd = libdevice.rsqrt(var + eps)
        w_l = tl.load(QNorm + j).to(tl.float32)
        w_r = tl.load(QNorm + half + j).to(tl.float32)
        n_l = (w_l * (x_l * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        n_r = (w_r * (x_r * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        rot_l = _bf16_rot_pair(n_l.to(tl.float32), c, -n_r.to(tl.float32), s)
        rot_r = _bf16_rot_pair(n_r.to(tl.float32), c, n_l.to(tl.float32), s)
        dst = QueryOut + row * q_out_stride + h.to(tl.int64) * PADDED_HEAD_DIM
        if PADDED_HEAD_DIM == HEAD_DIM:
            tl.store(dst + j, rot_l)
            tl.store(dst + half + j, rot_r)
        else:
            out_l = (rot_l.to(tl.float32) * query_scale).to(tl.bfloat16)
            out_r = (rot_r.to(tl.float32) * query_scale).to(tl.bfloat16)
            zeros = tl.zeros((half,), dtype=tl.bfloat16)
            tl.store(dst + j, out_l)
            tl.store(dst + half + j, out_r)
            tl.store(dst + HEAD_DIM + j, zeros)
            tl.store(dst + HEAD_DIM + half + j, zeros)
    else:
        kh = (h - Q_HEADS).to(tl.int64)
        src = QKV + row * qkv_stride + (Q_HEADS + kh) * HEAD_DIM
        x_l = tl.load(src + j).to(tl.float32)
        x_r = tl.load(src + half + j).to(tl.float32)
        var = (tl.sum(x_l * x_l) + tl.sum(x_r * x_r)) * (1.0 / HEAD_DIM)
        rstd = libdevice.rsqrt(var + eps)
        w_l = tl.load(KNorm + j).to(tl.float32)
        w_r = tl.load(KNorm + half + j).to(tl.float32)
        n_l = (w_l * (x_l * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        n_r = (w_r * (x_r * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        rot_l = _bf16_rot_pair(n_l.to(tl.float32), c, -n_r.to(tl.float32), s)
        rot_r = _bf16_rot_pair(n_r.to(tl.float32), c, n_l.to(tl.float32), s)
        dst = KeyOut + row * k_out_stride + kh * HEAD_DIM
        tl.store(dst + j, rot_l)
        tl.store(dst + half + j, rot_r)


@triton.jit
def _draft_stage_kv_windows(
    KeyHist,
    ValHist,
    CurKey,
    QKV,
    Starts,
    Lengths,
    StateRows,
    QStartLoc,
    StagedK,
    StagedV,
    LengthsOut,
    kh_row_stride,
    kh_pos_stride,
    kh_head_stride,
    vh_row_stride,
    vh_pos_stride,
    vh_head_stride,
    cur_k_stride,
    qkv_stride,
    qkv_v_offset,
    TOKENS: tl.constexpr,
    WINDOW: tl.constexpr,
    CAPACITY: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    off = tl.program_id(1).to(tl.int64)
    start = tl.load(Starts + r).to(tl.int64)
    length = tl.load(Lengths + r).to(tl.int64)
    prefix = tl.minimum(start, WINDOW)
    if off == 0:
        tl.store(LengthsOut + r, (prefix + length).to(tl.int32))
    valid = (length > 0) & (off < prefix + length)
    from_hist = off < prefix
    state_row = tl.load(StateRows + r).to(tl.int64)
    raw_pos = start - prefix + off
    hist_pos = ((raw_pos % WINDOW) + WINDOW) % WINDOW
    q_start = tl.load(QStartLoc + r).to(tl.int64)
    cur_idx = tl.minimum(tl.maximum(q_start + off - prefix, 0), TOKENS - 1)
    d = tl.arange(0, HEAD_DIM)
    zero = tl.zeros((HEAD_DIM,), dtype=tl.bfloat16)
    for kh in tl.static_range(KV_HEADS):
        dst_base = (r * CAPACITY + off) * (KV_HEADS * PADDED_HEAD_DIM) + kh * PADDED_HEAD_DIM
        if valid:
            if from_hist:
                k_val = tl.load(KeyHist + state_row * kh_row_stride + hist_pos * kh_pos_stride + kh * kh_head_stride + d)
                v_val = tl.load(ValHist + state_row * vh_row_stride + hist_pos * vh_pos_stride + kh * vh_head_stride + d)
            else:
                k_val = tl.load(CurKey + cur_idx * cur_k_stride + kh * HEAD_DIM + d)
                v_val = tl.load(QKV + cur_idx * qkv_stride + qkv_v_offset + kh * HEAD_DIM + d)
            tl.store(StagedK + dst_base + d, k_val)
            tl.store(StagedV + dst_base + d, v_val)
        else:
            tl.store(StagedK + dst_base + d, zero)
            tl.store(StagedV + dst_base + d, zero)
        if PADDED_HEAD_DIM > HEAD_DIM:
            tl.store(StagedK + dst_base + HEAD_DIM + d, zero)
            tl.store(StagedV + dst_base + HEAD_DIM + d, zero)


@triton.jit
def _draft_unpad_heads(
    Attended,
    Out,
    att_row_stride,
    out_row_stride,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, HEAD_DIM)
    val = tl.load(Attended + row * att_row_stride + h * PADDED_HEAD_DIM + d)
    tl.store(Out + row * out_row_stride + h * HEAD_DIM + d, val)


@triton.jit
def _draft_context_layer_write(
    KV,
    KNorm,
    Pos,
    Starts,
    Requests,
    StateRows,
    Counts,
    Inv,
    KeyCache,
    ValCache,
    kv_stride,
    layer_offset,
    kv_half_offset,
    k_row_stride,
    k_pos_stride,
    k_head_stride,
    v_row_stride,
    v_pos_stride,
    v_head_stride,
    eps,
    WINDOW: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    kh = tl.program_id(1).to(tl.int64)
    pos = tl.load(Pos + row).to(tl.int64)
    req = tl.load(Requests + row).to(tl.int64)
    start = tl.load(Starts + req).to(tl.int64)
    cnt = tl.load(Counts + req).to(tl.int64)
    offset = pos - start
    min_off = tl.maximum(cnt - WINDOW, 0)
    live = (pos >= 0) & (offset < cnt) & (offset >= min_off)
    if live:
        state_row = tl.load(StateRows + req).to(tl.int64)
        slot_pos = ((pos % WINDOW) + WINDOW) % WINDOW
        half: tl.constexpr = HEAD_DIM // 2
        j = tl.arange(0, half)
        k_src = KV + row * kv_stride + layer_offset + kh * HEAD_DIM
        x_l = tl.load(k_src + j).to(tl.float32)
        x_r = tl.load(k_src + half + j).to(tl.float32)
        var = (tl.sum(x_l * x_l) + tl.sum(x_r * x_r)) * (1.0 / HEAD_DIM)
        rstd = libdevice.rsqrt(var + eps)
        w_l = tl.load(KNorm + j).to(tl.float32)
        w_r = tl.load(KNorm + half + j).to(tl.float32)
        n_l = (w_l * (x_l * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        n_r = (w_r * (x_r * rstd).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        angles = pos.to(tl.float32) * tl.load(Inv + j).to(tl.float32)
        c = libdevice.cos(angles).to(tl.bfloat16).to(tl.float32)
        s = libdevice.sin(angles).to(tl.bfloat16).to(tl.float32)
        rot_l = _bf16_rot_pair(n_l.to(tl.float32), c, -n_r.to(tl.float32), s)
        rot_r = _bf16_rot_pair(n_r.to(tl.float32), c, n_l.to(tl.float32), s)
        k_dst = KeyCache + state_row * k_row_stride + slot_pos * k_pos_stride + kh * k_head_stride
        tl.store(k_dst + j, rot_l)
        tl.store(k_dst + half + j, rot_r)
        v_src = KV + row * kv_stride + layer_offset + kv_half_offset + kh * HEAD_DIM
        v_dst = ValCache + state_row * v_row_stride + slot_pos * v_pos_stride + kh * v_head_stride
        tl.store(v_dst + j, tl.load(v_src + j))
        tl.store(v_dst + half + j, tl.load(v_src + half + j))


class PreparedContext:
    """Project accepted target auxiliary states once into every draft layer's context keys and values."""

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace, runtime, coordinator, examples=None):
        config, draft = weights.config, weights.draft
        d, eps = config.draft, config.draft.text["rms_norm_eps"]
        self.window, self.eps = d.window, eps
        tokens, device = auxiliary.shape[0], auxiliary.device
        _, kv_heads = local_heads(config, weights.shard)
        hidden = config.hidden

        def buffer(role, *shape, dtype=auxiliary.dtype):
            return declaration_tensor(examples, "dspark.context." + role, (tokens, *shape), dtype, device)
        self.local = buffer("local", hidden // weights.shard.world_size)
        self.projected = buffer("projected", hidden)
        self.normalized = buffer("normalized", hidden)
        self.project = projection(auxiliary, draft.projection, self.local, session=session, name=name + ".project",
                                  workspace=workspace)
        self.gather = PreparedChunkedAllGather(self.local, self.projected, runtime=runtime, coordinator=coordinator,
                                        session=session, name=name + ".gather")
        self.hidden_norm = draft.hidden_norm
        self.kv = buffer("kv", 2 * len(draft.layers) * kv_heads * d.head_dim)
        self.project_kv = projection(self.normalized, draft.context_kv, self.kv,
                                     session=session, name=name + ".kv", workspace=workspace)
        self.key = buffer("key", kv_heads, d.head_dim)
        self.value = buffer("value", kv_heads, d.head_dim)
        self.slots = buffer("slots", dtype=torch.int64).fill_(-1)
        self.inverse = persistent_copy(inverse_frequencies(config, device))
        self.kv_heads, self.head_dim = kv_heads, d.head_dim
        self.fused_stages = []
        self.stages = []
        for index, layer in enumerate(draft.layers):
            k_cache = pool.reserved_region(f"draft{index}.key")
            v_cache = pool.reserved_region(f"draft{index}.value")
            if (d.head_dim == 64 and k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16
                    and k_cache.ndim == 4 and v_cache.ndim == 4
                    and k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1):
                self.fused_stages.append((layer.key_norm, k_cache, v_cache))
            else:
                writer = PreparedPagedKVWriter(self.key, self.value, k_cache, v_cache,
                    self.slots, session=session, name=f"{name}.{index}.write")
                self.stages.append((canonical_norm(layer.key_norm, eps), writer.functional()))
        self.use_fused_context = len(self.fused_stages) == len(draft.layers) and len(draft.layers) > 0
        if not self.use_fused_context and self.fused_stages:
            self.fused_stages = []
            self.stages = []
            for index, layer in enumerate(draft.layers):
                writer = PreparedPagedKVWriter(self.key, self.value, pool.reserved_region(f"draft{index}.key"),
                    pool.reserved_region(f"draft{index}.value"), self.slots, session=session, name=f"{name}.{index}.write")
                self.stages.append((canonical_norm(layer.key_norm, eps), writer.functional()))
        for operation in (self.project, self.gather, self.project_kv):
            operation.functional(session=session)
        self.key_shape = tuple(self.key.shape)
        self.local = self.projected = self.normalized = self.kv = self.key = self.value = self.slots = None

    def functional(self):
        return self

    def __call__(self, auxiliary, counts, metadata):
        m = metadata
        projected = self.gather(self.project(auxiliary))
        normalized = torch.empty_like(projected)
        rms_norm_columns(projected, self.hidden_norm, normalized, self.eps)
        kv = self.project_kv(normalized)
        if self.use_fused_context and kv.dtype == torch.bfloat16 and kv.is_contiguous():
            half = self.kv_heads * self.head_dim
            layer_width = 2 * half
            for index, (k_norm, k_cache, v_cache) in enumerate(self.fused_stages):
                torch.library.wrap_triton(_draft_context_layer_write)[(kv.shape[0], self.kv_heads)](
                    kv, k_norm, m.positions, m.starts, m.requests, m.state_rows, counts, self.inverse,
                    k_cache, v_cache, kv.stride(0), index * layer_width, half,
                    k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                    v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
                    float(self.eps), WINDOW=self.window, HEAD_DIM=self.head_dim, num_warps=1)
            return
        offset = m.positions - m.starts[m.requests]
        count = counts[m.requests]
        live = (m.positions >= 0) & (offset < count) & (offset >= (count - self.window).clamp_min(0))
        slots = torch.where(live, m.state_rows[m.requests] * self.window + m.positions % self.window, -1)
        chunks = kv.view(kv.shape[0], len(self.stages), 2, -1)
        for index, (norm, writer) in enumerate(self.stages):
            key = rotate(norm(chunks[:, index, 0].view(self.key_shape)), m.positions, self.inverse)
            value = chunks[:, index, 1].view(self.key_shape).contiguous()
            writer(key, value, slots=slots)

    def scratch_tensors(self):
        return ()


@dataclass(frozen=True)
class DraftScratch:
    local_embedding: torch.Tensor
    embedding: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    heads_output: torch.Tensor
    partial: torch.Tensor
    output: torch.Tensor
    window_key: torch.Tensor
    window_value: torch.Tensor
    mlp: MLPScratch
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, block, device):
        d = config.draft
        heads, kv_heads = local_heads(config, shard)
        rows = tokens // block
        pages = rows * ((d.window + block + 127) // 128)

        def zeros(*shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        hidden = config.hidden
        return cls(zeros(tokens, hidden // shard.world_size), zeros(tokens, hidden),
            zeros(tokens, (heads + 2 * kv_heads) * d.head_dim), zeros(tokens, heads, PADDED_HEAD),
            zeros(tokens, kv_heads, d.head_dim), zeros(tokens, kv_heads, d.head_dim),
            zeros(tokens, heads, PADDED_HEAD), zeros(tokens, heads * d.head_dim), zeros(tokens, hidden),
            zeros(tokens, hidden),
            zeros(pages, 128, kv_heads, PADDED_HEAD), zeros(pages, 128, kv_heads, PADDED_HEAD),
            MLPScratch.allocate(tokens, hidden, d.text["intermediate_size"] // shard.world_size, device),
            zeros(tokens, hidden), zeros(tokens, config.vocabulary // shard.world_size),
            zeros(tokens, config.vocabulary))

    def rows(self, tokens, samples):
        if samples != tokens:
            raise ValueError("DSpark projects every block position")
        return type(self)(*(getattr(self, field) if field.startswith("window_")
                           else self.mlp.rows(tokens) if field == "mlp"
                           else getattr(self, field)[:tokens] for field in self.__dataclass_fields__))

    def scratch_tensors(self):
        return (self,)


class DraftAttention(torch.nn.Module):
    def __init__(self, index, weights, config, scratch, metadata, pool, block, *, runtime, coordinator, session,
                 name, workspace):
        super().__init__()
        self.query_shape, self.key_shape = tuple(scratch.query.shape), tuple(scratch.key.shape)
        d, eps = config.draft, config.draft.text["rms_norm_eps"]
        self.qkv = projection(scratch.embedding, weights.qkv, scratch.qkv, session=session,
                              name=name + ".qkv", workspace=workspace).functional(session=session)
        self.q_norm, self.k_norm = canonical_norm(weights.query_norm, eps), canonical_norm(weights.key_norm, eps)
        self.inverse = persistent_copy(inverse_frequencies(config, scratch.embedding.device))
        self.key_cache, self.value_cache = (pool.reserved_region(f"draft{index}.{kind}") for kind in ("key", "value"))
        rows = metadata.lengths.numel()
        count = (d.window + block + 127) // 128
        self.key, self.value = scratch.window_key[:rows * count], scratch.window_value[:rows * count]
        self.pages = persistent_copy(torch.arange(rows * count, device=scratch.embedding.device, dtype=torch.int32).view(rows, count))
        self.lengths = torch.zeros_like(metadata.lengths)
        self.window = d.window
        # Each block position attends causally to the committed window and the block's earlier positions.
        self.attention = PreparedPagedAttention(scratch.query, self.key, self.value, self.pages, self.lengths,
            metadata.query_start_loc, scratch.attended, mode="verify", width=block, window=d.window - 1,
            causal=True, session=session, workspace=workspace, name=name + ".attention",
            caller_owned_placeholders=True).functional(session=session, k_cache=None, v_cache=None,
                                                       cache_seqlens=None, cu_seqlens_q=None)
        self.output = projection(scratch.heads_output, weights.output, scratch.partial, session=session,
                                 name=name + ".output", workspace=workspace).functional(session=session)
        self.head_dim, self.query_scale = d.head_dim, (PADDED_HEAD / d.head_dim) ** .5
        self.eps = eps
        self.q_norm_weight, self.k_norm_weight = weights.query_norm, weights.key_norm
        self.reduce = PreparedAllReduce(scratch.partial, scratch.output, runtime=runtime, coordinator=coordinator,
                                        session=session, name=name + ".reduce").functional(session=session)
        self.window_shape = tuple(self.key.shape)
        self.key = self.value = self.lengths = None

    def forward(self, hidden_states, *, metadata, reduce=True, **kwargs):
        m, width = metadata, self.head_dim
        q_heads, kv_heads = self.query_shape[1], self.key_shape[1]
        q_width, kv_width = q_heads * width, kv_heads * width
        qkv = self.qkv(hidden_states)
        if (width == 64 and PADDED_HEAD in (64, 128) and hidden_states.dtype == torch.bfloat16
                and qkv.is_contiguous() and self.key_cache.dtype == torch.bfloat16
                and self.value_cache.dtype == torch.bfloat16
                and self.key_cache.stride(-1) == 1 and self.value_cache.stride(-1) == 1):
            tokens = qkv.shape[0]
            rows = m.starts.numel()
            capacity = self.window_shape[0] * self.window_shape[1] // rows
            query = torch.empty(self.query_shape, dtype=hidden_states.dtype, device=hidden_states.device)
            key = torch.empty(self.key_shape, dtype=hidden_states.dtype, device=hidden_states.device)
            torch.library.wrap_triton(_draft_qkv_prep)[(tokens, q_heads + kv_heads)](
                qkv, self.q_norm_weight, self.k_norm_weight, m.positions, self.inverse, query, key,
                qkv.stride(0), query.stride(0), key.stride(0),
                float(self.eps), float(self.query_scale),
                Q_HEADS=q_heads, HEAD_DIM=width, PADDED_HEAD_DIM=PADDED_HEAD, num_warps=1)
            staged_key = torch.empty(self.window_shape, dtype=hidden_states.dtype, device=hidden_states.device)
            staged_value = torch.empty(self.window_shape, dtype=hidden_states.dtype, device=hidden_states.device)
            lengths = torch.empty((rows,), dtype=torch.int32, device=hidden_states.device)
            torch.library.wrap_triton(_draft_stage_kv_windows)[(rows, capacity)](
                self.key_cache, self.value_cache, key, qkv, m.starts, m.lengths, m.state_rows, m.query_start_loc,
                staged_key, staged_value, lengths,
                self.key_cache.stride(0), self.key_cache.stride(1), self.key_cache.stride(2),
                self.value_cache.stride(0), self.value_cache.stride(1), self.value_cache.stride(2),
                key.stride(0), qkv.stride(0), q_width + kv_width,
                TOKENS=tokens, WINDOW=self.window, CAPACITY=capacity, KV_HEADS=kv_heads,
                HEAD_DIM=width, PADDED_HEAD_DIM=PADDED_HEAD, num_warps=2)
            attended = self.attention(query, k_cache=staged_key, v_cache=staged_value,
                                      cache_seqlens=lengths, cu_seqlens_q=m.query_start_loc)
            if PADDED_HEAD == width:
                heads = attended.reshape(tokens, q_width)
            else:
                heads = torch.empty((tokens, q_width), dtype=attended.dtype, device=attended.device)
                torch.library.wrap_triton(_draft_unpad_heads)[(tokens, q_heads)](
                    attended, heads, attended.stride(0), heads.stride(0),
                    HEAD_DIM=width, PADDED_HEAD_DIM=PADDED_HEAD, num_warps=2)
            partial = self.output(heads)
            return (self.reduce(partial) if reduce else partial), None
        q, k, v = qkv.split((q_width, kv_width, kv_width), -1)
        query = torch.zeros(self.query_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        rotated_q = rotate(self.q_norm(q.view(-1, q_heads, width)), m.positions, self.inverse)
        query[..., :width].copy_(rotated_q if PADDED_HEAD == width else rotated_q * self.query_scale)
        key = rotate(self.k_norm(k.view(self.key_shape)), m.positions, self.inverse)
        value = v.view(self.key_shape)
        staged_key = torch.zeros(self.window_shape, dtype=key.dtype, device=key.device)
        staged_value = torch.zeros_like(staged_key)
        stage_windows(self.key_cache, key, staged_key[..., :width], m)
        stage_windows(self.value_cache, value, staged_value[..., :width], m)
        lengths = m.starts.clamp(max=self.window).int() + m.lengths
        attended = self.attention(query, k_cache=staged_key, v_cache=staged_value,
                                 cache_seqlens=lengths, cu_seqlens_q=m.query_start_loc)
        heads = attended[..., :width].contiguous().flatten(1)
        partial = self.output(heads)
        return (self.reduce(partial) if reduce else partial), None


class DraftMLP(torch.nn.Module):
    def __init__(self, weights, scratch, *, runtime, coordinator, session, name, workspace):
        super().__init__()
        s = scratch.mlp
        self.mlp = PreparedMLP(scratch.embedding, weights, s.gate_up, s.intermediate, s.output,
                               session=session, workspace=workspace, name=name, activation_mode="a16").functional(session=session)
        self.reduce = PreparedAllReduce(s.output, scratch.output, runtime=runtime, coordinator=coordinator,
                                        session=session, name=name + ".reduce").functional(session=session)

    def forward(self, hidden_states, *, reduce=True):
        partial = self.mlp(hidden_states)
        return self.reduce(partial) if reduce else partial


class PreparedDraft:
    """Project and gather draft logits for sampled proposals."""

    def __init__(self, weights, scratch, metadata, pool, *, block, runtime, vocabulary_runtime, embedding_runtime,
                 coordinator, session, name, workspace, gather_rows=None, row_tuning=None):
        if gather_rows is not None:
            scratch = replace(scratch, logits=scratch.logits[:gather_rows])
        self.logit_shape, self.local_logit_shape = tuple(scratch.logits.shape), tuple(scratch.local_logits.shape)
        self.logit_dtype, self.local_logit_dtype = scratch.logits.dtype, scratch.local_logits.dtype
        config, draft = weights.config, weights.draft
        eps = config.draft.text["rms_norm_eps"]
        self.eps = float(eps)
        self.embedding = PreparedShardedEmbedding(draft.embedding, metadata.token_ids, scratch.local_embedding,
            scratch.embedding, runtime=embedding_runtime, coordinator=coordinator,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        self.layers = []
        tokens = scratch.embedding.shape[0]
        use_row_tuning = row_tuning is not None and 1 <= tokens <= 32 and config.hidden == 7168
        self.row_choices = [] if use_row_tuning else None
        if use_row_tuning:
            from .row_reductions import RowGeometry
            draft_geom = RowGeometry(7168, rows=tokens, residual_norm=True, eps=self.eps)
        for index, layer_weights in enumerate(draft.layers):
            layer = Qwen3DecoderLayer.__new__(Qwen3DecoderLayer)
            torch.nn.Module.__init__(layer)
            layer.self_attn = DraftAttention(index, layer_weights, config, scratch, metadata, pool, block,
                runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.layer{index}.attention",
                workspace=workspace)
            layer.mlp = DraftMLP(layer_weights.mlp, scratch, runtime=runtime, coordinator=coordinator,
                                 session=session, name=f"{name}.layer{index}.mlp", workspace=workspace)
            layer.input_layernorm = _FastRMSNorm(layer_weights.input_norm, eps).eval()
            layer.post_attention_layernorm = _FastRMSNorm(layer_weights.post_attention_norm, eps).eval()
            self.layers.append(layer.eval())
            if use_row_tuning:
                self.row_choices.append((
                    row_tuning.register(layer.self_attn.reduce, draft_geom),
                    row_tuning.register(layer.mlp.reduce, draft_geom),
                ))
        self.norm = _FastRMSNorm(draft.norm, eps).eval()
        head = weights.head if draft.head is None else draft.head
        self.head = PreparedVocabularyProjection(scratch.hidden, head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            workspace=workspace).functional(session=session)

    def release_outputs(self):
        return self

    def forward(self, metadata):
        hidden = self.embedding(token_ids=metadata.token_ids, num_rows=metadata.live_counts[:1])
        if getattr(self, "row_choices", None) is not None:
            residual = hidden
            normed = self.layers[0].input_layernorm(residual)
            for index, (layer, (attn_choice, mlp_choice)) in enumerate(zip(self.layers, self.row_choices, strict=True)):
                if attn_choice is not None and attn_choice.fused:
                    partial, _ = layer.self_attn(normed, metadata=metadata, reduce=False)
                    residual, normed = attn_choice.residual_norm(
                        partial, residual, layer.post_attention_layernorm.weight, self.eps
                    )
                else:
                    attn_out, _ = layer.self_attn(normed, metadata=metadata)
                    residual = residual + attn_out
                    normed = layer.post_attention_layernorm(residual)
                next_norm = self.layers[index + 1].input_layernorm if index + 1 < len(self.layers) else self.norm
                if mlp_choice is not None and mlp_choice.fused:
                    partial = layer.mlp(normed, reduce=False)
                    residual, normed = mlp_choice.residual_norm(partial, residual, next_norm.weight, self.eps)
                else:
                    residual = residual + layer.mlp(normed)
                    normed = next_norm(residual)
            hidden = normed[metadata.sample_indices]
            return hidden, self.head(hidden)
        for layer in self.layers:
            hidden = layer(hidden, metadata=metadata)
        hidden = self.norm(hidden[metadata.sample_indices])
        return hidden, self.head(hidden)

    def __call__(self, metadata):
        return self.forward(metadata)[1], None


class PreparedMarkov:
    """Compute FP32 transition logits and gather them when the head is vocabulary-sharded."""

    def __init__(self, weights, tokens, *, runtime, coordinator, session, name, examples=None):
        from ignition.components.embedding import PreparedEmbedding
        from ignition.components.markov_sampling import PreparedRowScaledFP8MarkovProjection, RowScaledFP8MarkovWeight

        rows, device = tokens.numel(), tokens.device
        draft, vocabulary = weights.draft, weights.config.vocabulary
        columns = vocabulary if runtime is None else vocabulary // weights.shard.world_size
        self.embedding = declaration_tensor(examples, "dspark.markov.embedding",
            (rows, weights.config.draft.markov_rank), torch.bfloat16, device)
        self.local = declaration_tensor(examples, "dspark.markov.local",
            (rows, columns), torch.float32, device)
        self.embed = PreparedEmbedding(draft.markov_embedding, tokens, self.embedding, session=session,
                                       name=name + ".embed").functional(session=session, dynamic_inputs=True)
        projection_type = (PreparedRowScaledFP8MarkovProjection if isinstance(draft.markov_head, RowScaledFP8MarkovWeight)
                           else UnquantizedProjection)
        self.projection = projection_type(self.embedding, draft.markov_head, self.local,
            session=session, name=name + ".head").functional(session=session)
        self.gather = None
        if runtime is not None:
            logits = declaration_tensor(examples, "dspark.markov.logits", (rows, vocabulary), torch.float32, device)
            # BF16 views carry both halves of each FP32 score without conversion.
            self.gather = PreparedAllGather(self.local.view(torch.bfloat16), logits.view(torch.bfloat16),
                runtime=runtime, coordinator=coordinator, session=session, name=name + ".gather").functional(session=session)
        self.embedding = self.local = None

    def forward(self, tokens):
        local = self.projection(self.embed(token_ids=tokens))
        return local, local if self.gather is None else self.gather(local.view(torch.bfloat16)).view(torch.float32)

    def __call__(self, tokens):
        return self.forward(tokens)[1]


@dataclass(frozen=True)
class DraftForwards:
    decoder: object
    context: object
    draft: object
    markov: object


class SpeculativePreparation:
    """Shared draft storage and preparation inputs of every DSpark round, consumed by PreparedParallelDraftRound.

    ``collectives`` are the target decoder's; the rounds pass only the reduction, vocabulary gather and
    coordinator to their draft modules. Verification records every KDA layer's candidates (``KDARecords``),
    so each request keeps one committed recurrent row.
    """

    def __init__(self, weights, pool, requests, inputs, scratch, *, completion, frame, sequence, session, workspace,
                 collectives, decoder_buffers, draft_tokens, context_runtime, page_tokens, markov_runtime,
                 examples):
        from ignition.components.kda_speculation import KDARecords
        from .decoder import PreparedDecoder

        self.page_tokens = page_tokens
        self.weights, self.tables, self.pool = weights, None, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        coordinator = collectives["coordinator"]
        self.collectives = dict(runtime=collectives["runtime"], vocabulary_runtime=collectives["vocabulary_runtime"],
                                coordinator=coordinator)
        self.markov_runtime = markov_runtime
        self.block_proposal_broadcast = markov_runtime is None
        kda_layers = {i: w["attention"] for i, w in enumerate(weights.layers) if weights.config.layers[i] == "kda"}
        heads = weights.config.text["linear_attn_config"]["num_heads"] // weights.shard.world_size
        self.kda_records = KDARecords.allocate(kda_layers, requests.capacity + 1, heads, pool.backing.device)

        def decoder(weights, tables, scratch, metadata, pool, *, mode, **kwargs):
            return PreparedDecoder(weights, scratch, metadata, pool, mode=mode, buffers=decoder_buffers,
                kda_records=self.kda_records if mode == "verify" else None, examples=examples,
                **{**collectives, **kwargs})

        # The anchor predicts the first draft, so a block holds one position per draft.
        self.block_size, self.logit_offset = max(1, draft_tokens), 0
        context_type, draft_type, scratch_type = PreparedContext, PreparedDraft, DraftScratch
        if weights.config.draft.window is None:
            from ignition.recipes.kimi31.dspark import DraftScratch as scratch_type
            from ignition.recipes.kimi31.dspark import PreparedContext as context_type
            from ignition.recipes.kimi31.dspark import PreparedDraft as draft_type

            context_type = partial(context_type, context=collectives["context"])
            draft_type = partial(draft_type, context=collectives["context"],
                                 latents=collectives["latent_runtimes"][-1])
        draft_kwargs = dict(block=self.block_size, embedding_runtime=collectives["embedding_runtime"])
        if draft_type is PreparedDraft:
            draft_kwargs["row_tuning"] = decoder_buffers.row_tuning
        self.forwards = DraftForwards(decoder,
            partial(context_type, runtime=context_runtime, coordinator=coordinator, examples=examples),
            partial(draft_type, **draft_kwargs),
            partial(PreparedMarkov, examples=examples))
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = (persistent_zeros(requests.generation.shape, requests.generation.dtype,
            requests.generation.device) for _ in range(2))
        self.draft_scratch = scratch_type.allocate(weights.config, weights.shard,
            requests.capacity * self.block_size, self.block_size, pool.backing.device)

    def release_examples(self):
        self.scratch = self.draft_scratch = self.kda_records = self.forwards = None
