// SPDX-License-Identifier: Apache-2.0
// Replay the signed candidate channel's three real gathers and check every word
// on sixteen GPUs while ranks enter successive slices at different times.
#include "shard_candidates.cu"
#include <cuda.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

static void checked(cudaError_t v){if(v!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(v));exit(1);}}
static void driver(CUresult v){if(v!=CUDA_SUCCESS){const char *s;cuGetErrorString(v,&s);fprintf(stderr,"%s\n",s);exit(1);}}
template<class T>T *device(size_t n){T *p;checked(cudaMalloc(&p,n*sizeof(T)));return p;}
constexpr int local_rows=rows/world;
constexpr unsigned stage_offset=4<<20;
constexpr uint64_t bank=0x2002000;
__host__ __device__ int id_at(int src,int row,int col,int step) {
    int v=row*2048+col;
    return (src+v+step)%17 ? (v*31+src+271*step)&65535 : -1;
}
__host__ __device__ unsigned bits_at(int src,int row,int col,int step) {
    switch ((row+col+src+step)%7) {
    case 0:return 0;case 1:return 0x80000000u;case 2:return 0x7f800000u;
    case 3:return 0x7fc01234u;case 4:return 0xff800000u;
    case 5:return 0xc0400000u;default:return 0x3f800000u;
    }
}
__global__ void seed_words(int *ids,unsigned *bits,int64_t *positions,int rank,int step) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*topk;i+=blockDim.x*gridDim.x) {
        ids[i]=id_at(rank,i/topk,i%topk,step);
        bits[i]=bits_at(rank,i/topk,i%topk,step);
    }
    if(!blockIdx.x&&threadIdx.x<rows)positions[threadIdx.x]=(threadIdx.x+step)%11 ? int64_t(threadIdx.x)+32768 : int64_t(-1);
}
__global__ void verify_received(const uint2 *pairs,const unsigned char *active,int *selected,
                                unsigned *error,int rank,int step) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<local_rows*world*topk;i+=blockDim.x*gridDim.x) {
        int local=i/(world*topk),src=i/topk%world,col=i%topk,row=(local/2*world+rank)*2+local%2;
        int id=id_at(src,row,col,step);
        uint2 got=pairs[i];
        bool mismatch=got.x!=(id<0?0xff800000u:bits_at(src,row,col,step)) ||
            got.y!=(id<0?0xffffffffu:unsigned(id*world+src)) || active[local]!=((row+step)%11!=0);
        if(mismatch)atomicAdd(error,1u);
        if(src==col%world)selected[local*topk+col]=got.y;
    }
}
__global__ void verify_result(const int *all,unsigned *error,int step) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*topk;i+=blockDim.x*gridDim.x) {
        int src=i%topk%world,id=id_at(src,i/topk,i%topk,step);
        if(all[i]!=(id<0?-1:id*world+src))atomicAdd(error,1u);
    }
}
__global__ void skew(unsigned clocks) {
    unsigned long long t=clock64();
    while(clock64()-t<clocks) __nanosleep(24);
}
struct Rank {
    CUdeviceptr region,stage;
    CUmemGenericAllocationHandle handle;
    CUmodule module; CUfunction ring;
    cudaStream_t stream;
    int *ids,*profile,*selected;
    unsigned *bits,*error;
    int64_t *positions;
    uint2 *received; unsigned char *active;
    unsigned blocks,threads,shared;
};
using Ranks=std::array<Rank,world>;
void gather(Ranks &ranks,int rank,void *source,void *output,int n) {
    Rank &r=ranks[rank];
    uint64_t stride=bank,offset=bank,counter=2*bank;
    void *args[39]={&source,&output,&r.profile};
    for(int i=0;i<world;++i){args[3+i]=&ranks[i].stage;args[19+i]=&ranks[i].region;}
    args[35]=&n;args[36]=&stride;args[37]=&offset;args[38]=&counter;
    driver(cuLaunchKernel(r.ring,r.blocks,1,1,r.threads,1,1,r.shared,r.stream,args,nullptr));
}
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    driver(cuInit(0));Ranks ranks;
    CUmemAccessDesc access[world]{};
    for(int i=0;i<world;i++){access[i].location={CU_MEM_LOCATION_TYPE_DEVICE,i};access[i].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;}
    Peers peer;
    for(int i=0;i<world;++i) {
        checked(cudaSetDevice(i));checked(cudaFree(nullptr));Rank &r=ranks[i];
        CUmemAllocationProp p{};p.type=CU_MEM_ALLOCATION_TYPE_PINNED;p.location=access[i].location;
        size_t granularity;driver(cuMemGetAllocationGranularity(&granularity,&p,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        size_t size=(0x41000+2*bank+granularity-1)/granularity*granularity;
        driver(cuMemCreate(&r.handle,size,&p,0));driver(cuMemAddressReserve(&r.region,size,0,0,0));
        driver(cuMemMap(r.region,size,0,r.handle,0));driver(cuMemSetAccess(r.region,size,access,world));
        r.stage=r.region+0x41000;peer.rows[i]=reinterpret_cast<uint4*>(r.stage+stage_offset);
        checked(cudaMemset(reinterpret_cast<void*>(r.region),0,size));
        r.ids=device<int>(rows*topk);r.bits=device<unsigned>(rows*topk);r.positions=device<int64_t>(rows);
        r.active=device<unsigned char>(local_rows);r.selected=device<int>(local_rows*topk);
        r.received=device<uint2>(local_rows*world*topk);r.error=device<unsigned>(1);r.profile=device<int>(1);
        checked(cudaMemset(r.error,0,4));int profile=8;checked(cudaMemcpy(r.profile,&profile,4,cudaMemcpyHostToDevice));
        checked(cudaStreamCreateWithFlags(&r.stream,cudaStreamNonBlocking));
        auto file=std::string(argv[1])+"/rank-"+std::to_string(i);
        driver(cuModuleLoad(&r.module,(file+".cubin").c_str()));
        std::string symbol;std::ifstream meta(file+".txt");meta>>symbol>>r.blocks>>r.threads>>r.shared;if(!meta)return 1;
        driver(cuModuleGetFunction(&r.ring,r.module,symbol.c_str()));checked(cudaDeviceSynchronize());
    }
    for(int step=0;step<8;++step)for(int i=0;i<world;++i) {
        int rank=(step%2?world-1-i:i);
        checked(cudaSetDevice(rank));Rank &r=ranks[rank];auto s=r.stream;
        skew<<<1,1,0,s>>>(((rank+step)%7)*30000);
        seed_words<<<512,256,0,s>>>(r.ids,r.bits,r.positions,rank,step);
        gather(ranks,rank,r.ids,r.received,1);
        publish_candidates<<<512,256,0,s>>>(r.ids,r.bits,reinterpret_cast<uint2*>(r.stage+stage_offset),rank);
        gather(ranks,rank,r.ids,r.received,1);
        read_candidate_rows<<<dim3(world,local_rows),256,0,s>>>(peer,reinterpret_cast<uint4*>(r.received),r.active,r.positions,rank);
        verify_received<<<512,256,0,s>>>(r.received,r.active,r.selected,r.error,rank,step);
        gather(ranks,rank,r.selected,r.received,local_rows/2);
        verify_result<<<512,256,0,s>>>(reinterpret_cast<int*>(r.received),r.error,step);
        checked(cudaPeekAtLastError());
    }
    for(int i=0;i<world;++i) {
        checked(cudaSetDevice(i));checked(cudaStreamSynchronize(ranks[i].stream));
        unsigned errors;checked(cudaMemcpy(&errors,ranks[i].error,4,cudaMemcpyDeviceToHost));
        if(errors){fprintf(stderr,"rank %d had %u incorrect candidate pairs or output IDs\n",i,errors);return 1;}
    }
    puts("TP16: 8 skewed iterations, 33,554,432 candidate pairs and 33,554,432 replicated IDs matched");
}
