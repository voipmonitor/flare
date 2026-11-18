// SPDX-License-Identifier: Apache-2.0
#include "shard_candidates.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>

static void checked(cudaError_t e) {
    if (e!=cudaSuccess) { fprintf(stderr,"%s\n",cudaGetErrorString(e)); exit(1); }
}
template<class T>T *buffer(size_t n) {
    T *p;checked(cudaMalloc(&p,n*sizeof(T)));return p;
}
static int local_id(int owner,int index) {
    return (index+owner)%17 ? (index*31+owner)%65536 : -1;
}
static unsigned score(int owner,int index) {
    const unsigned values[]={0,0x80000000u,0xff800000u,0x7f800000u,0x7f801234u,0xbf800000u,0x3f000000u};
    return values[(owner+index)%7];
}
int main() {
    constexpr int local_rows=rows/world;
    Peers peers;
    auto ids=buffer<int>(rows*topk);
    auto scores=buffer<unsigned>(rows*topk);
    auto pos=buffer<int64_t>(rows);
    auto active=buffer<unsigned char>(local_rows);
    auto gathered=buffer<uint2>(local_rows*world*topk);
    std::vector<int> ix(rows*topk);
    std::vector<unsigned> bits(rows*topk);
    std::vector<int64_t> positions(rows);
    for(int r=0;r<rows;r++)positions[r]=r%11 ? 34567+r : -1;
    checked(cudaMemcpy(pos,positions.data(),positions.size()*sizeof(int64_t),cudaMemcpyHostToDevice));
    for(int owner=0;owner<world;owner++) {
        uint2 *stage=buffer<uint2>(rows*topk);
        peers.rows[owner]=reinterpret_cast<uint4*>(stage);
        for(int i=0;i<rows*topk;i++) { ix[i]=local_id(owner,i);bits[i]=score(owner,i); }
        checked(cudaMemcpy(ids,ix.data(),ix.size()*sizeof(int),cudaMemcpyHostToDevice));
        checked(cudaMemcpy(scores,bits.data(),bits.size()*sizeof(unsigned),cudaMemcpyHostToDevice));
        publish_candidates<<<512,256>>>(ids,scores,stage,owner);
    }
    std::vector<uint2> pairs(local_rows*world*topk);
    std::vector<unsigned char> flags(local_rows);
    std::vector<int> coverage(rows);
    for(int rank=0;rank<world;rank++) {
        read_candidate_rows<<<dim3(world,local_rows),256>>>(peers,reinterpret_cast<uint4*>(gathered),active,pos,rank);
        checked(cudaMemcpy(pairs.data(),gathered,pairs.size()*sizeof(uint2),cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(flags.data(),active,flags.size(),cudaMemcpyDeviceToHost));
        for(int pair=0;pair<local_rows/2;pair++)for(int half=0;half<2;half++) {
            const int local=2*pair+half;
            // The original all-gather writes local source rows in rank order
            // for each of its four packets. Each packet holds two ID rows.
            const int row=(pair*world+rank)*2+half;
            if(flags[local]!=(positions[row]>=0))return 1;
            coverage[row]++;
            for(int owner=0;owner<world;owner++)for(int col=0;col<topk;col++){
                int i=row*topk+col,id=local_id(owner,i);
                uint2 v=pairs[(local*world+owner)*topk+col];
                if(v.x!=(id<0?0xff800000u:score(owner,i)) || v.y!=(id<0?0xffffffffu:unsigned(id*world+owner))) {
                    fprintf(stderr,"candidate mismatch row=%d owner=%d col=%d\n",row,owner,col);return 1;
                }
            }
        }
    }
    for(int count:coverage)if(count!=1)return 1;
    puts("All 4,194,304 score/ID pairs and 128 gathered row positions matched exactly");
    return 0;
}
