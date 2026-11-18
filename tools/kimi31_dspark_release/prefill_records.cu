// SPDX-License-Identifier: Apache-2.0
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int world = 16;
constexpr int page_rows = 128;
constexpr int topk = 2048;
constexpr int slice_rows = 128;
constexpr int slab_records = topk * slice_rows;
constexpr int vectors_per_record = 576 / sizeof(uint4);

__device__ void copy_record(uint4 *out, const uint4 *source, int lane) {
    for (int v = lane; v < vectors_per_record; v += warpSize)
        out[v] = source ? source[v] : make_uint4(0, 0, 0, 0);
}
} // namespace

// Keep the selection order. The page table doubles as a compact vector of logical
// token IDs until a long-context gather replaces those IDs with slab offsets.
extern "C" __global__ void prefill_record_pages(
    const int *selected, const int64_t *positions, const int *table,
    int *pages, int *slots, int *counts, int columns, int table_stride, int cache_pages) {
    constexpr int threads = 256;
    constexpr int entries = topk / threads;
    using Scan = cub::BlockScan<int, threads>;
    __shared__ typename Scan::TempStorage scan;
    const int row = blockIdx.x;
    const int first = threadIdx.x * entries;
    const int offset = row * topk;
    const int64_t end = positions[row];
    int ids[entries], physical[entries];
    int valid = 0;
    #pragma unroll
    for (int i = 0; i < entries; ++i) {
        const int id = selected[offset + first + i];
        const int local = max(id, 0) / world;
        const int col = local / page_rows;
        int page = -1;
        if (id >= 0 && id <= end && col < columns)
            page = table[row * table_stride + col];
        ids[i] = id;
        physical[i] = page >= 0 && page < cache_pages ? page * page_rows + local % page_rows : -1;
        valid += physical[i] >= 0;
    }
    int target, count;
    Scan(scan).ExclusiveSum(valid, target, count);
    if (threadIdx.x == 0) counts[row] = count;
    #pragma unroll
    for (int i = 0; i < entries; ++i) {
        if (physical[i] >= 0) {
            pages[offset + target] = ids[i];
            slots[offset + target++] = physical[i];
        }
        if (first + i >= count) pages[offset + first + i] = 0;
    }
}

// A warp copies one latent record. The short-context path stages a logical row
// once per layer, in the same slab the captured attention kernel already reads.
extern "C" __global__ void prefill_record_cache(
    const uint4 *const *peer, int *pages, const int *slots, const int *counts, uint4 *slab,
    const int *table, const int64_t *positions, const int *query_length,
    int graph_rows, int first_slice, int columns, int cache_pages) {
    const int n = min(max(*query_length, 0), graph_rows);
    const int64_t end = n && positions[0] >= 0 ? positions[n-1] + 1 : 0;
    const int lane = threadIdx.x % warpSize;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int stride = gridDim.x * blockDim.x / warpSize;
    if (end <= slab_records) {
        if (!first_slice) return;
        for (int token = warp; token < end; token += stride) {
            const int local = token / world;
            const int col = local / page_rows;
            const int page = col < columns ? table[col] : -1;
            const uint4 *base = peer[token % world];
            const uint4 *source = nullptr;
            if (page >= 0 && page < cache_pages && base)
                source = base + (static_cast<int64_t>(page) * page_rows + local % page_rows) * vectors_per_record;
            copy_record(slab + token * vectors_per_record, source, lane);
        }
        return;
    }
    for (int record = warp; record < slab_records; record += stride) {
        const int row = record / topk;
        // Lane zero owns the in-place page index update. Broadcast the original
        // token before replacing it, including independent-thread-scheduling GPUs.
        int token = __shfl_sync(0xffffffffu, lane == 0 ? pages[record] : 0, 0);
        const uint4 *source = nullptr;
        if (record % topk < counts[row]) {
            const int slot = slots[record];
            const uint4 *base = peer[token % world];
            if (slot >= 0 && slot < static_cast<int64_t>(cache_pages) * page_rows && base)
                source = base + static_cast<int64_t>(slot) * vectors_per_record;
        }
        copy_record(slab + record * vectors_per_record, source, lane);
        if (lane == 0) pages[record] = record;
    }
}
