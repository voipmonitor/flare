# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""FP8 sparse selection with prepared quantization and caller-owned pages."""

import math

import torch
import triton
import triton.language as tl
from b12x.attention import dsa_indexer
from b12x.attention.dsa_indexer import glm_pool
from b12x.attention.dsa_indexer._glm_pool_kernels import _write_pool
from b12x.preparation import PreparedCall

from ignition.components.indexer import PreparedSparseIndexer
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.tuning import declare, register_finalizer


@triton.jit
def _write_keys(Keys, Cache, ScaleCache, Slots, stride, HADAMARD: tl.constexpr, POWER_OF_TWO_SCALE: tl.constexpr):
    row = tl.program_id(0)
    dim = tl.arange(0, 128)
    slot = tl.load(Slots + row)
    values = tl.load(Keys + row * 128 + dim)
    _write_pool(Cache, ScaleCache, slot, values, dim, dim < 128, slot >= 0,
                stride, 64, 128, 448.0, HADAMARD=HADAMARD, POWER_OF_TWO_SCALE=POWER_OF_TWO_SCALE)


@triton.jit
def _norm_and_write_keys(
    RawKeys,
    Weight,
    Bias,
    NormOut,
    Cache,
    ScaleCache,
    Slots,
    raw_stride,
    out_stride,
    cache_stride,
    eps,
    HADAMARD: tl.constexpr,
    POWER_OF_TWO_SCALE: tl.constexpr,
    WRITE_OUT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    dim = tl.arange(0, 128)
    slot = tl.load(Slots + row)
    key = tl.load(RawKeys + row * raw_stride + dim).to(tl.float32)
    mean = tl.sum(key, axis=0) * (1.0 / 128.0)
    centered = key - mean
    variance = tl.sum(centered * centered, axis=0) * (1.0 / 128.0)
    w = tl.load(Weight + dim).to(tl.float32)
    b = tl.load(Bias + dim).to(tl.float32)
    normed = (centered * tl.rsqrt(variance + eps) * w + b).to(tl.bfloat16)
    if WRITE_OUT:
        tl.store(NormOut + row * out_stride + dim, normed)
    _write_pool(
        Cache,
        ScaleCache,
        slot,
        normed,
        dim,
        dim < 128,
        slot >= 0,
        cache_stride,
        64,
        128,
        448.0,
        HADAMARD=HADAMARD,
        POWER_OF_TWO_SCALE=POWER_OF_TWO_SCALE,
    )


def norm_and_write_fp8_index_keys(raw_keys, weight, bias, eps, slots, cache, *, hadamard, power_of_two, out=None):
    """Fused FP32 LayerNorm(128) + BF16 round + FP8 index key cache write."""
    write_out = out is not None
    norm_out = raw_keys if out is None else out
    torch.library.wrap_triton(_norm_and_write_keys)[(raw_keys.shape[0],)](
        raw_keys,
        weight,
        bias,
        norm_out,
        cache.view(torch.float8_e4m3fn),
        cache.view(torch.float32),
        slots,
        raw_keys.stride(0),
        norm_out.stride(0),
        cache.stride(0),
        float(eps),
        HADAMARD=hadamard,
        POWER_OF_TWO_SCALE=power_of_two,
        WRITE_OUT=write_out,
        num_warps=4,
    )


@triton.jit
def _quantize_queries_and_weights_kernel(
    q,
    raw_weights,
    q_out,
    scale_out,
    weights_out,
    rows,
    pre_scale,
    weight_norm,
    HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_R: tl.constexpr,
    APPLY_PRE_SCALE: tl.constexpr,
):
    """Fused FP8 query quantization + BF16 head weight scaling (bitwise match to copy+mul+quant)."""
    pid = tl.program_id(0)
    row = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = row < rows
    dim = tl.arange(0, HEAD_DIM)
    x = tl.load(
        q + row[:, None] * HEAD_DIM + dim[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(absmax / FP8_MAX)))
    quantized = tl.minimum(tl.maximum(x / scale[:, None], -FP8_MAX), FP8_MAX)
    tl.store(
        q_out + row[:, None] * HEAD_DIM + dim[None, :],
        quantized,
        mask=row_mask[:, None],
    )
    tl.store(scale_out + row, scale, mask=row_mask)
    w = tl.load(raw_weights + row, mask=row_mask, other=0.0).to(tl.float32)
    if APPLY_PRE_SCALE:
        w = (w * pre_scale).to(tl.bfloat16).to(tl.float32)
    w = w * weight_norm
    w = w * scale
    w = w * (1.0 / 64.0)
    tl.store(weights_out + row, w, mask=row_mask)


def write_fp8_index_keys(keys, slots, cache, *, hadamard, power_of_two):
    """Write owner-local keys independently of the cache used for query selection."""
    torch.library.wrap_triton(_write_keys)[(keys.shape[0],)](
        keys, cache.view(torch.float8_e4m3fn), cache.view(torch.float32),
        slots, cache.stride(0), HADAMARD=hadamard, POWER_OF_TWO_SCALE=power_of_two, num_warps=4)


class PreparedFP8Indexer:
    """Quantize BF16 queries/keys and select from planar 64-record FP8 pages.

    Hadamard rotation and power-of-two key scales are enabled by default.
    Identity transforms and ordinary FP32 key scales can be declared explicitly.
    Query scales remain powers of two for either transform.

    Query scales are folded into FP32 head weights. B12x's prepared query
    operation includes normalization for 32 heads, so the input weight copy
    compensates for the declared head count before that operation runs.
    The default scales raw weights by ``1 / sqrt(heads)``; an explicit
    ``head_weight_scale`` replaces that multiplier. Already normalized gates
    use one, yielding ``weights * query_scale / sqrt(128)`` in FP32.
    ``prefill_records_per_call`` bounds each sequential candidate window and
    its score workspace; all declared cache records participate in selection.
    """

    def __init__(self, query, weights, cache, pages, lengths, active, output, *,
                 mode, session, name, workspace, buffers=None, prefill_records_per_call=None, topk=512, scores=None,
                 head_weight_scale=None, hadamard=True, key_power_of_two=True, examples=None, example_role=None):
        if type(hadamard) is not bool or type(key_power_of_two) is not bool:
            raise ValueError("FP8 Hadamard and key power-of-two declarations must be booleans")
        if head_weight_scale is not None and (type(head_weight_scale) not in (int, float)
                or not math.isfinite(head_weight_scale) or head_weight_scale <= 0):
            raise ValueError("FP8 head-weight scale must be a finite positive scalar")
        if prefill_records_per_call is not None and (type(prefill_records_per_call) is not int
                                                     or prefill_records_per_call <= 0):
            raise ValueError("FP8 prefill record capacity per call must be a positive integer")
        if type(topk) is not int or topk <= 0 or output.shape != (query.shape[0], topk):
            raise ValueError("FP8 selection output must match a positive top-k capacity")
        rows, heads, width = query.shape
        if width != 128 or query.dtype != torch.bfloat16 or cache.shape[1] != 64 * 132:
            raise ValueError("FP8 indexing requires BF16 128-channel queries and 64-record pages")
        self.query, self.weights, self.cache = query, weights, cache
        self.hadamard, self.key_power_of_two = hadamard, key_power_of_two
        if examples is not None and (not isinstance(example_role, str) or not example_role):
            raise ValueError("shared indexer declarations require an explicit operand role")

        def zeros(role, shape, dtype):
            if examples is not None:
                return declaration_tensor(examples, example_role + "." + role, shape, dtype, query.device)
            return (torch.zeros(shape, dtype=dtype, device=query.device) if buffers is None
                    else buffers.zeros(shape, dtype))
        self.quantized = zeros("quantized", query.shape, torch.float8_e4m3fn)
        self.scales = zeros("scales", (rows, heads), torch.float32)
        self.scaled_weights = zeros("scaled_weights", (rows, heads), torch.float32)
        self.weight_normalization = ((32 / heads) ** 0.5 if head_weight_scale is None
                                     else 32 ** 0.5 * head_weight_scale)
        self.query_outputs = (self.quantized.view(-1, 128), self.scales.view(-1),
                              self.scaled_weights.view(-1))
        self.query_arguments = (query.view(-1, 128), *self.query_outputs)
        self.query_plan = glm_pool.plan_queries(*self.query_arguments, hadamard=hadamard)
        query_arguments = self.query_arguments
        declare(session, (self.query_plan.request(name=name + ".query",
            prepare_call=lambda state: PreparedCall(run=lambda: state.quantize(*query_arguments))),))
        self.pages = pages
        self.packed_pages = (pages if mode == "prefill" or (pages.is_contiguous() and pages.dtype == torch.int32)
                             else zeros("packed_pages", pages.shape, torch.int32))
        caps = dsa_indexer.Caps(device=query.device, num_q_heads=heads, max_q_rows=rows,
            max_page_table_width=pages.shape[1], topk=topk, mode=mode,
            supertile_k=(min(prefill_records_per_call, pages.shape[1] * 64)
                         if mode == "prefill" and prefill_records_per_call is not None else 0))
        self.selection = PreparedSparseIndexer(self.quantized, self.scaled_weights, cache,
            self.packed_pages, lengths, active, output, caps=caps, session=session,
            name=name + ".select", workspace=workspace, scores=scores)

    def write_keys(self, keys, slots):
        write_fp8_index_keys(keys, slots, self.cache,
                            hadamard=self.hadamard, power_of_two=self.key_power_of_two)

    def functional(self, *, session, dynamic_cache=False):
        self.dynamic_cache = dynamic_cache
        self.query_layout = tuple(self.query.shape)
        self.copy_pages = self.packed_pages is not self.pages
        self.selection.functional(session=session)
        register_finalizer(session, self._release_examples)
        return self

    def _release_examples(self):
        self.query = self.weights = self.quantized = self.scales = self.scaled_weights = None
        self.query_arguments = self.query_outputs = self.pages = self.packed_pages = None
        if self.dynamic_cache:
            self.cache = None
            self.selection.inputs["index_k_cache"] = None

    def __call__(self, query=None, weights=None, pages=None, lengths=None, active=None, *, cache=None):
        output, _, _ = self._run(query, weights, pages, lengths, active, cache=cache)
        return output

    def _run(self, query=None, weights=None, pages=None, lengths=None, active=None, *, cache=None):
        output, quantized, _, scaled_weights = self._run_observed(query, weights, pages, lengths, active, cache=cache)
        return output, quantized, scaled_weights

    def _run_observed(self, query=None, weights=None, pages=None, lengths=None, active=None, *, cache=None,
                      raw_pre_scale=None):
        if self.query is None:
            rows, heads, _ = self.query_layout
            quantized = torch.empty(self.query_layout, dtype=torch.float8_e4m3fn, device=query.device)
            scales = torch.empty((rows, heads), dtype=torch.float32, device=query.device)
            scaled_weights = torch.empty((rows, heads), dtype=torch.float32, device=query.device)
            if (not self.hadamard and weights.dtype == torch.bfloat16 and weights.is_contiguous()
                    and query.is_contiguous() and query.dtype == torch.bfloat16):
                total_rows = rows * heads
                torch.library.wrap_triton(_quantize_queries_and_weights_kernel)[(triton.cdiv(total_rows, 4),)](
                    query.view(-1, 128),
                    weights.view(-1),
                    quantized.view(-1, 128),
                    scales.view(-1),
                    scaled_weights.view(-1),
                    total_rows,
                    float(1.0 if raw_pre_scale is None else raw_pre_scale),
                    float(self.weight_normalization),
                    HEAD_DIM=128,
                    FP8_MAX=448.0,
                    BLOCK_R=4,
                    APPLY_PRE_SCALE=raw_pre_scale is not None,
                    num_warps=1,
                )
            else:
                if raw_pre_scale is not None:
                    weights = weights * raw_pre_scale
                scaled_weights.copy_(weights)
                scaled_weights.mul_(self.weight_normalization)
                glm_pool.quantize_queries(query.view(-1, 128), quantized.view(-1, 128), scales.view(-1),
                                          scaled_weights.view(-1), plan=self.query_plan)
            output = self.selection(quantized, scaled_weights, pages.clone() if self.copy_pages else pages,
                                    lengths, active, cache=cache)
            return output, quantized, scales, scaled_weights
        query = self.query if query is None else query
        weights = self.weights if weights is None else weights
        self.scaled_weights.copy_(weights)
        self.scaled_weights.mul_(self.weight_normalization)
        glm_pool.quantize_queries(query.view(-1, 128), *self.query_outputs, plan=self.query_plan)
        if self.packed_pages is not self.pages:
            self.packed_pages.copy_(self.pages)
        return self.selection(), self.quantized, self.scales, self.scaled_weights
