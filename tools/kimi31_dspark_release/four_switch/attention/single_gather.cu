// SPDX-License-Identifier: Apache-2.0
#include "partial_exchange.cu"

// One CTA owns a row. All ranks either pull the owner's single published
// input or push it to its rank-major slot at each consumer. The same row
// barrier retires the old bank on every rank before alternating-bank reuse.
template<bool Push>
__device__ void gather_small(const uint4 *input,uint4 *output,PeerStages p,PeerSignals sig,
                             int rank,int rows,int vectors,int64_t stride) {
    int row=blockIdx.x,t=threadIdx.x,warp=t/32,lane=t%32;
    unsigned generation=*reinterpret_cast<volatile unsigned*>(sig.base[rank]+1025);
    int64_t offset=(generation&1u)*stride;
    if(row<rows){
        if(Push){
            for(int dst=warp;dst<16;dst+=blockDim.x/32) {
                uint4 *remote=reinterpret_cast<uint4*>(p.base[dst]+offset)+(row*16+rank)*vectors;
                for(int i=lane;i<vectors;i+=32)remote[i]=input[row*vectors+i];
            }
            __threadfence_system();
        }else{
            uint4 *stage=reinterpret_cast<uint4*>(p.base[rank]+offset)+row*vectors;
            for(int i=t;i<vectors;i+=blockDim.x)stage[i]=input[row*vectors+i];
        }
        __syncthreads();
        exchange::barrier(sig,rank);
        if(Push) {
            const uint4 *source=reinterpret_cast<const uint4*>(p.base[rank]+offset)+row*16*vectors;
            for(int i=t;i<vectors*16;i+=blockDim.x)output[row*16*vectors+i]=exchange::load_vector(source+i);
        }else{
            for(int j=warp;j<16;j+=blockDim.x/32){
                int peer=(rank+j)&15;const uint4 *src=reinterpret_cast<const uint4*>(p.base[peer]+offset)+row*vectors;
                uint4 *out=output+(row*16+peer)*vectors;
                for(int i=lane;i<vectors;i+=32)out[i]=peer==rank?input[row*vectors+i]:exchange::load_vector(src+i);
            }
        }
        __syncthreads();
    }
    if(t==0)exchange::advance(sig.base[rank],gridDim.x);
}

extern "C" __global__ void pull_gather(const uint4 *input,uint4 *output,PeerStages p,PeerSignals sig,
                                       int rank,int rows,int vectors,int64_t stride){
    gather_small<false>(input,output,p,sig,rank,rows,vectors,stride);
}
extern "C" __global__ void push_gather(const uint4 *input,uint4 *output,PeerStages p,PeerSignals sig,
                                       int rank,int rows,int vectors,int64_t stride){
    gather_small<true>(input,output,p,sig,rank,rows,vectors,stride);
}

extern "C" __global__ void gather_whole_rows(const uint4 *input,uint4 *output,PeerStages p,PeerSignals sig,
                                             int rank,int rows,int vectors,int64_t stride) {
    int t=threadIdx.x,warp=t/32,lane=t%32,total=rows*vectors;
    unsigned g=*reinterpret_cast<volatile unsigned*>(sig.base[rank]+1025);
    int64_t shift=(g&1u)*stride;
    uint4 *mine=reinterpret_cast<uint4*>(p.base[rank]+shift);
    for(int j=t;j<total;j+=blockDim.x)mine[j]=input[j];
    __syncthreads();
    exchange::barrier(sig,rank);
    int src=(warp%16+rank)&15;
    const uint4 *remote=reinterpret_cast<uint4*>(p.base[src]+shift);
    for(int j=warp/16*32+lane;j<total;j+=blockDim.x/16){
        int row=j/vectors,col=j-row*vectors;
        output[(row*16+src)*vectors+col]=src==rank?input[j]:exchange::load_vector(remote+j);
    }
    __syncthreads();
    if(!t)exchange::advance(sig.base[rank],1);
}

// Blocks own disjoint stripes of each warp packet across all rows. Two to
// four stripes can overlap PCIe reads while preserving one ready wave per stripe.
extern "C" __global__ void gather_packet_stripes(const uint4 *input,uint4 *output,PeerStages p,PeerSignals sig,
                                                 int rank,int rows,int vectors,int64_t stride) {
    int tid=threadIdx.x,lane=tid%32,warp=tid/32,total=rows*vectors;
    unsigned g=*reinterpret_cast<volatile unsigned*>(sig.base[rank]+1025);
    int64_t shift=(g&1u)*stride;
    int stride_vec=32*gridDim.x;
    uint4 *mine=reinterpret_cast<uint4*>(p.base[rank]+shift);
    for(int j=blockIdx.x*32+lane+warp*stride_vec;j<total;j+=(blockDim.x/32)*stride_vec)mine[j]=input[j];
    __syncthreads();
    exchange::barrier(sig,rank);
    int owner=(rank+warp%16)&15;const uint4 *remote=(const uint4*)(p.base[owner]+shift);
    for(int j=blockIdx.x*32+lane+warp/16*stride_vec;j<total;j+=stride_vec*(blockDim.x/512)){
        int row=j/vectors,col=j-row*vectors;
        output[(row*16+owner)*vectors+col]=rank==owner?input[j]:exchange::load_vector(remote+j);
    }
    __syncthreads();
    if(!tid)exchange::advance(sig.base[rank],gridDim.x);
}
