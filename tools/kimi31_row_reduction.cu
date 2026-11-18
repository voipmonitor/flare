// SPDX-License-Identifier: Apache-2.0
// One CTA completes all three communication phases for one four-token row.
// The caller can then run that row's canonical residual logic immediately.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
using U=unsigned;
using B=unsigned char;
struct Args {
    B* local[4];
    B* across[4];
    uint4 const* input;
    uint4* delta;
    unsigned long long stage,part,final,capacity;
    U lane,island;
};
static_assert(sizeof(Args)==120);
template<class T> __device__ __forceinline__ T* at(T* const (&p)[4],U i){
    return i<2?(i==0?p[0]:p[1]):(i==2?p[2]:p[3]);
}
__device__ __forceinline__ float2 unpack(U bits) {
    return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&bits));
}
__device__ __forceinline__ U add_word(U a,U b,U c,U d) {
    float2 x=unpack(a),y=unpack(b),z=unpack(c),w=unpack(d);
    __nv_bfloat162 result=__floats2bfloat162_rn(((x.x+y.x)+z.x)+w.x,((x.y+y.y)+z.y)+w.y);
    return *reinterpret_cast<U*>(&result);
}
__device__ __forceinline__ uint4 add_v(uint4 a,uint4 b,uint4 c,uint4 d){
    return make_uint4(add_word(a.x,b.x,c.x,d.x),add_word(a.y,b.y,c.y,d.y),
                      add_word(a.z,b.z,c.z,d.z),add_word(a.w,b.w,c.w,d.w));
}
__device__ __forceinline__ uint4 read_v(uint4 const* p){
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");
    return v;
}
__device__ __forceinline__ void notify(B* dst,B* src,U region,U row,U myid,U peer,U epoch){
    U* out=reinterpret_cast<U*>(dst+region+(4*row+myid)*128);
    U* in=reinterpret_cast<U*>(src+region+(4*row+peer)*128);
    asm volatile("fence.sc.sys; st.relaxed.sys.global.u32 [%0],%1;"::"l"(out),"r"(epoch):"memory");
    U val;
    do {
        asm volatile("ld.acquire.sys.global.u32 %0,[%1];":"=r"(val):"l"(in):"memory");
        if(int(val-epoch)>=0)break;
        __nanosleep(24);
    } while(true);
}
template<bool Direct>
__device__ __forceinline__ void reduce(Args a){
    constexpr U S=224,RS=512,XS=RS+64*4*128,AG=XS+64*4*128;
    U row=blockIdx.x,t=threadIdx.x; B* slab=at(a.local,a.lane);
    U* counter=reinterpret_cast<U*>(slab);
    if(t==0){
        atomicAdd(counter+row,1);
        if(row<4)atomicAdd(counter+64+row,1);
    }
    if(row>=4)return;
    __syncthreads();
    U parity=counter[row]&1,epoch=counter[64+row];
    unsigned long long so=a.stage+parity*a.capacity*16,po=a.part+parity*a.capacity*4,fo=a.final+parity*a.capacity*16;
    for(U i=t;i<S;i+=blockDim.x){
        #pragma unroll
        for(U dest=0;dest<4;++dest)if(dest!=a.lane){
            uint4* inbox=reinterpret_cast<uint4*>(a.local[dest]+so);
            inbox[(4*a.lane+row)*S+i]=a.input[(4*row+dest)*S+i];
        }
    }
    __syncthreads();
    if(t<4 && t!=a.lane)notify(at(a.local,t),slab,RS,row,a.lane,t,epoch);
    __syncthreads();
    for(U i=t;i<S;i+=blockDim.x){
        uint4 v[4];
        #pragma unroll
        for(U src=0;src<4;++src) v[src]=src==a.lane?a.input[(4*row+a.lane)*S+i]:
              read_v(reinterpret_cast<uint4 const*>(slab+so)+(4*src+row)*S+i);
        reinterpret_cast<uint4*>(slab+po)[row*S+i]=add_v(v[0],v[1],v[2],v[3]);
    }
    __syncthreads();
    if(t<4 && t!=a.island)notify(at(a.across,t),slab,XS,row,a.island,t,epoch*2u);
    __syncthreads();
    for(U i=t;i<S;i+=blockDim.x){
        uint4 v[4];
        #pragma unroll
        for(U sw=0;sw<4;++sw)v[sw]=read_v(reinterpret_cast<uint4 const*>(a.across[sw]+po)+row*S+i);
        uint4 result=add_v(v[0],v[1],v[2],v[3]);U pos=(4*row+a.lane)*S+i;
        if constexpr(!Direct) a.delta[pos]=result;
        #pragma unroll
        for(U dest=0;dest<4;++dest)if(Direct || dest!=a.lane)reinterpret_cast<uint4*>(a.local[dest]+fo)[pos]=result;
    }
    __syncthreads();
    if(t<4 && t!=a.lane)notify(at(a.local,t),slab,AG,row,a.lane,t,epoch);
    __syncthreads();
    if constexpr(!Direct){
        for(U i=t;i<S;i+=blockDim.x){
            #pragma unroll
            for(U src=0;src<4;++src)if(src!=a.lane){
                U pos=(4*row+src)*S+i;a.delta[pos]=read_v(reinterpret_cast<uint4 const*>(slab+fo)+pos);
            }
        }
        __syncthreads();
    }
}
extern "C" __device__ __noinline__ void row_to_buffer(Args a){reduce<false>(a);}
extern "C" __device__ __noinline__ void row_to_mixture(Args a){reduce<true>(a);}
