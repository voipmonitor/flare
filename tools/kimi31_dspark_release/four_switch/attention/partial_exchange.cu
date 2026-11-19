// SPDX-License-Identifier: Apache-2.0
#pragma once
// Route attention partials through peer-readable storage. Every rank keeps
// its six heads and uses the signed DCP implementation's source order.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>
struct PeerStages { unsigned char *base[16]; };
struct PeerSignals { unsigned *base[16]; };
namespace exchange {
constexpr int ranks=16, local_heads=6, heads=96, dims=512, max_rows=8;
constexpr unsigned payload_bytes=max_rows*heads*dims*sizeof(__nv_bfloat16);
constexpr unsigned lse_offset=payload_bytes;
__device__ __forceinline__ uint4 load_vector(const uint4 *p) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p));
    return v;
}
__device__ __forceinline__ float load_float(const float *p) {
    float v;asm volatile("ld.global.cg.f32 %0,[%1];":"=f"(v):"l"(p));return v;
}
__device__ __forceinline__ void advance(unsigned *self, unsigned blocks) {
    // The channel's graph epoch follows the last CTA, as in the provider.
    if(blocks==1 || atomicAdd(self+1026,1u)==blocks-1) {
        if(blocks>1) *reinterpret_cast<volatile unsigned*>(self+1026)=0;
        *reinterpret_cast<volatile unsigned*>(self+1025)=self[1025]+1;
    }
}
__device__ __forceinline__ void barrier(PeerSignals p,int rank) {
    unsigned tid=threadIdx.x,b=blockIdx.x;
    if(tid<ranks) {
        unsigned *self=p.base[rank]; unsigned value=self[b*ranks+tid]+1;
        self[b*ranks+tid]=value;
        unsigned bank=1024+((value%2)*64+b)*ranks*32;
        unsigned *send=p.base[tid]+bank+rank*32;
        unsigned *receive=self+bank+tid*32;
        asm volatile("fence.sc.sys; st.relaxed.sys.global.u32 [%0],%1;"::"l"(send),"r"(value):"memory");
        unsigned observed;
        do { asm volatile("ld.acquire.sys.global.u32 %0,[%1];":"=r"(observed):"l"(receive):"memory"); } while(observed!=value);
    }
    __syncthreads();
}
}
extern "C" __global__ void partials_ready(PeerSignals p,int rank) {
    exchange::barrier(p,rank);
    if(threadIdx.x==0)exchange::advance(p.base[rank],gridDim.x);
}
template<bool Pushed,bool Packed=false> __device__ void combine(PeerStages p,__nv_bfloat16* out,int rank) {
    using namespace exchange;
    int row=blockIdx.x, head=blockIdx.y, lane=threadIdx.x%32;
    int element=threadIdx.x*8;
    __shared__ float weights[ranks];
    if(threadIdx.x<32) {
        int owner=(rank+lane)&15;
        int offset=Pushed ? (owner*max_rows+row)*local_heads+head : row*heads+rank*local_heads+head;
        unsigned char *base=Pushed ? p.base[rank] : p.base[owner];
        float lane_lse=lane<16?load_float(Packed ? reinterpret_cast<float*>(base+1040*offset+1024) : reinterpret_cast<float*>(base+lse_offset)+offset):-CUDART_INF_F;
        if(!isfinite(lane_lse))lane_lse=-CUDART_INF_F;
        float high=-CUDART_INF_F;
        #pragma unroll
        for(int i=0;i<ranks;++i)high=fmaxf(high,__shfl_sync(0xffffffffu,lane_lse,i));
        if(!isfinite(high))high=0.0f;
        float val=isfinite(lane_lse)?expf(lane_lse-high):0.0f;
        float sum=0.0f;
        #pragma unroll
        for(int i=0;i<ranks;++i)sum+=__shfl_sync(0xffffffffu,val,i);
        if(lane<ranks)weights[lane]=val*(1.0f/fmaxf(sum,1e-10f));
    }
    __syncthreads();
    if(element>=dims)return;
    float accum[8]={};
    #pragma unroll
    for(int t=0;t<ranks;++t) {
        int owner=(rank+t)&15;float w=weights[t];
        if(w!=0.0f) {
            int target=Pushed ? (owner*max_rows+row)*local_heads+head : row*heads+rank*local_heads+head;
            unsigned char *base=Pushed ? p.base[rank] : p.base[owner];
            uint4 q=load_vector(reinterpret_cast<const uint4*>(base)+target*(Packed?65:dims/8)+threadIdx.x);
            const auto *vals=reinterpret_cast<const __nv_bfloat16*>(&q);
            #pragma unroll
            for(int i=0;i<8;++i)accum[i]=__fmaf_rn(w,__bfloat162float(vals[i]),accum[i]);
        }
    }
    uint4 q;
    auto *vals=reinterpret_cast<__nv_bfloat16*>(&q);
    #pragma unroll
    for(int i=0;i<8;++i)vals[i]=__float2bfloat16_rn(accum[i]);
    reinterpret_cast<uint4*>(out)[(row*local_heads+head)*(dims/8)+threadIdx.x]=q;
}
extern "C" __global__ void combine_pulled_partials(PeerStages p,__nv_bfloat16* output,int rank){combine<false>(p,output,rank);}
extern "C" __global__ void combine_pushed_partials(PeerStages p,__nv_bfloat16* output,int rank){combine<true>(p,output,rank);}

extern "C" __global__ void combine_bulk_partials(PeerStages p,__nv_bfloat16* output,int rank){combine<true,true>(p,output,rank);}

// Each warp reads one source into a distinct shared row before any multiply-
// accumulate chain begins. The numerical order stays cyclic on every output.
extern "C" __global__ void combine_cooperative(PeerStages peers,__nv_bfloat16 *out,int rank) {
    using namespace exchange;
    __shared__ uint4 data[ranks][dims/8];
    __shared__ float lse[ranks],w[ranks];
    int tid=threadIdx.x,lane=tid%32,warp=tid/32,row=blockIdx.x,head=blockIdx.y;
    int index=row*heads+rank*local_heads+head;
    for(int s=warp;s<ranks;s+=blockDim.x/32) {
        int owner=(rank+s)&15;
        const unsigned char *p=peers.base[owner];
        if(lane==0)lse[s]=load_float(reinterpret_cast<const float*>(p+lse_offset)+index);
        const uint4 *v=reinterpret_cast<const uint4*>(p)+(dims/8)*index;
        data[s][lane]=load_vector(v+lane);
        data[s][lane+32]=load_vector(v+lane+32);
    }
    __syncthreads();
    if(warp==0) {
        float x=lane<ranks && isfinite(lse[lane]) ? lse[lane] : -CUDART_INF_F;
        float hi=-CUDART_INF_F;
        #pragma unroll
        for(int j=0;j<ranks;++j)hi=fmaxf(hi,__shfl_sync(0xffffffffu,x,j));
        if(!isfinite(hi))hi=0;
        float y=isfinite(x)?expf(x-hi):0;
        float denom=0;
        #pragma unroll
        for(int j=0;j<ranks;++j)denom+=__shfl_sync(0xffffffffu,y,j);
        if(lane<ranks)w[lane]=y*(1.f/fmaxf(denom,1e-10f));
    }
    __syncthreads();
    __nv_bfloat162 *dst=reinterpret_cast<__nv_bfloat162*>(out)+(row*local_heads+head)*dims/2;
    for(int col=tid;col<dims/2;col+=blockDim.x){
        float2 v={0,0};
        #pragma unroll
        for(int s=0;s<ranks;++s)if(w[s]!=0){
            float2 x=__bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(data[s])[col]);
            v.x=__fmaf_rn(w[s],x.x,v.x);v.y=__fmaf_rn(w[s],x.y,v.y);
        }
        dst[col]=__floats2bfloat162_rn(v.x,v.y);
    }
}

extern "C" __global__ void combine_pulled_dma(PeerStages peers,__nv_bfloat16 *out,int rank) {
    using namespace exchange;
    constexpr int payload=local_heads*dims*2, lse_floats=8;
    constexpr int in_bytes=ranks*(payload+lse_floats*4);
    extern __shared__ __align__(128) unsigned char memory[];
    uint32_t sm=(uint32_t)__cvta_generic_to_shared(memory),bar=sm+in_bytes;
    int t=threadIdx.x,lane=t%32,row=blockIdx.x;
    if(t==0)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(bar):"memory");
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncthreads();
    if(t==0)asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar),"r"(in_bytes):"memory");
    if(t<ranks){
        int owner=(rank+t)&15,head=rank*local_heads;
        unsigned off=row*heads+head;
        const unsigned char *source=peers.base[owner];
        unsigned a=sm+t*payload,b=sm+ranks*payload+t*lse_floats*4;
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(a),"l"(source+off*dims*2),"r"(payload),"r"(bar):"memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(b),"l"(source+lse_offset+(off&~3u)*4),"r"(lse_floats*4),"r"(bar):"memory");
    }
    asm volatile("{ .reg .pred p; WAIT%=: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0; @!p bra.uni WAIT%=; }"::"r"(bar):"memory");
    __shared__ float weights[local_heads][ranks];
    int warp=t/32;
    if(warp<local_heads){
        const float *lse=reinterpret_cast<const float*>(memory+ranks*payload);
        float x=lane<ranks?lse[lane*lse_floats+(rank*local_heads%4)+warp]:-CUDART_INF_F;
        if(!isfinite(x))x=-CUDART_INF_F;float hi=-CUDART_INF_F;
        #pragma unroll
        for(int j=0;j<ranks;j++)hi=fmaxf(hi,__shfl_sync(0xffffffffu,x,j));
        if(!isfinite(hi))hi=0;
        float y=isfinite(x)?expf(x-hi):0,denom=0;
        #pragma unroll
        for(int j=0;j<ranks;j++)denom+=__shfl_sync(0xffffffffu,y,j);
        if(lane<ranks)weights[warp][lane]=y*(1.f/fmaxf(denom,1e-10f));
    }
    __syncthreads();
    for(int i=t;i<local_heads*dims/2;i+=blockDim.x){
        const int h=i/(dims/2);float2 a={0,0};
        #pragma unroll
        for(int s=0;s<ranks;s++) if(weights[h][s]!=0){
            auto z=__bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(memory+s*payload)[i]);
            a.x=__fmaf_rn(weights[h][s],z.x,a.x);a.y=__fmaf_rn(weights[h][s],z.y,a.y);
        }
        reinterpret_cast<__nv_bfloat162*>(out)[row*(local_heads*dims/2)+i]=__floats2bfloat162_rn(a.x,a.y);
    }
}
