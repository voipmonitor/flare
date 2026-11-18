// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int world = 16;
constexpr int topk = 2048;
constexpr int rows = 128;
struct Peers { const uint4 *rows[world]; };
}

// This is the input order of the target's exact top-k fold. Preserve score
// bits (including signed zeros and NaNs) and the original rank's logical ID.
extern "C" __global__ void publish_candidates(const int *local_ids, const unsigned *score_bits,
                                               uint2 *staging, int rank) {
    for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<rows*topk; i+=gridDim.x*blockDim.x) {
        const int id=local_ids[i];
        staging[i]=id<0 ? make_uint2(0xff800000u,0xffffffffu)
                        : make_uint2(score_bits[i],static_cast<unsigned>(id)*world+rank);
    }
    __threadfence_system();
}

// The result all-gather treats two adjacent int32 ID rows as a single packet.
// Assign input rows in that packet order, so its output is already [128,2048].
extern "C" __global__ void read_candidate_rows(Peers peers, uint4 *gathered, unsigned char *active,
                                               const int64_t *positions, int rank) {
    constexpr int vectors_per_row=topk/2;
    const int local=blockIdx.y;
    const int row=(local/2*world+rank)*2+local%2;
    const int owner=blockIdx.x;
    if (owner==0 && threadIdx.x==0) active[local]=positions[row]>=0;
    const uint4 *src=peers.rows[owner]+row*vectors_per_row;
    uint4 *dst=gathered+(local*world+owner)*vectors_per_row;
    for(int i=threadIdx.x;i<vectors_per_row;i+=blockDim.x) dst[i]=src[i];
}
