# SPDX-License-Identifier: Apache-2.0

"""Assemble absorbed MLA queries and their saturated E4M3 representation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _assemble(Projected, Query, Output, Quantized, ROWS: tl.constexpr,
              HEADS: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0)
    row, head = group // HEADS, group % HEADS
    column = tl.arange(0, BLOCK)
    projected = tl.load(Projected + (head * ROWS + row) * 512 + column, column < 512, other=0)
    tail = tl.load(Query + group * 192 + column - 512 + 128,
                   (column >= 512) & (column < 576), other=0)
    value = tl.where(column < 512, projected, tail)
    tl.store(Output + group * 576 + column, value, column < 576)
    floating = value.to(tl.float32)
    quantized = tl.minimum(tl.maximum(floating, -448.), 448.).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    bits = value.to(tl.uint16, bitcast=True)
    invalid = ((bits & 0x7f80) == 0x7f80) & ((bits & 0x7f) != 0)
    # Torch's E4M3 conversion canonicalizes either-sign BF16 NaNs to positive 0x7f.
    quantized = tl.where(invalid, 0x7f, quantized.to(tl.uint32)).to(tl.uint8)
    tl.store(Quantized + group * 576 + column, quantized, column < 576)


def assemble_query(projected, query, output, quantized):
    """Preserve the BF16 query while writing its clamped FP8 copy into caller-owned storage."""
    rows, heads, width = query.shape
    if (heads != 6 or width != 192 or projected.shape != (heads, rows, 512)
            or output.shape != (rows, heads, 576) or quantized.shape != output.shape
            or quantized.dtype != torch.float8_e4m3fn or not quantized.is_contiguous()
            or quantized.device != query.device
            or any(x.dtype != torch.bfloat16 or not x.is_contiguous() or x.device != query.device
                   for x in (projected, query, output))):
        raise ValueError('absorbed query requires contiguous BF16 projected[6,rows,512], '
                         'query[rows,6,192], output[rows,6,576] and matching FP8 output')
    from .absorbed_query_op import assemble

    assemble(projected, query, output, quantized)


def fuse_query_quantization(query, *, mode):
    """Select the measured eight-row, six-head SM120 verification geometry."""
    return (mode == 'verify' and query.shape == (8, 6, 192) and query.dtype == torch.bfloat16
            and query.is_contiguous() and query.device.type == 'cuda'
            and torch.cuda.get_device_capability(query.device) == (12, 0))
