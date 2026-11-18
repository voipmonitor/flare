# SPDX-License-Identifier: Apache-2.0

"""Combine A16 split-K summation with a copy of the BF16 forget-input columns."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cuda.bindings import driver as cuda

from b12x._lib.program_cache import program_cache
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr


class ReduceCopy:
    def __init__(self, columns, slices, begin, width):
        self.columns, self.slices, self.begin, self.width = columns, slices, begin, width

    @cute.jit
    def __call__(self, partials: cute.Pointer, output: cute.Pointer, copied: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(partials, output, copied, rows).launch(
            grid=((self.columns + 255) // 256, rows, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, partials: cute.Pointer, output: cute.Pointer, copied: cute.Pointer, rows: Int32):
        column = Int64(cute.arch.block_idx()[0]) * 256 + cute.arch.thread_idx()[0]
        row = Int64(cute.arch.block_idx()[1])
        offset = row * self.columns + column
        size = Int64(rows) * self.columns
        if column < self.columns:
            value = cutlass.Float32(0)
            for part in cutlass.range_constexpr(self.slices):
                value += partials[Int64(part) * size + offset]
            rounded = cutlass.BFloat16(value)
            output[offset] = rounded
            if column >= self.begin and column < self.begin + self.width:
                copied[row * self.width + column - self.begin] = rounded


@program_cache(scope="preparation")
def compile_reduce_copy(device, columns=3208, slices=4, begin=2304, width=128):
    if not 0 <= begin < begin + width <= columns or slices < 1:
        raise ValueError('split-K copy requires a valid output column interval and positive slice count')
    key = columns, slices, begin, width
    return b12x_compile(ReduceCopy(*key),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        1, current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key('ignition.kimi.kda.split_k_copy', 1, key))


