// SPDX-License-Identifier: Apache-2.0
// Compare the vectors the native attention page table addresses against the
// owning rank's KV bytes, including holes, padding and the slab boundary.
#include "prefill_records.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

static void checked(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); exit(1); }
}
template<class T> T *managed(size_t n) {
    T *p;
    checked(cudaMallocManaged(&p, n*sizeof(T)));
    return p;
}
int main() {
    constexpr int cols = 512, physical_pages = 3, graph_rows = 512;
    auto peer = managed<uint4*>(world);
    for (int owner=0; owner<world; ++owner) {
        peer[owner] = managed<uint4>(physical_pages * page_rows * vectors_per_record);
        auto *bytes = reinterpret_cast<unsigned char*>(peer[owner]);
        for(int i=0; i<physical_pages*page_rows*576; ++i)
            bytes[i] = (i*67 + (i >> 8)*7 + owner*31) & 255;
    }
    auto table=managed<int>(cols), selected=managed<int>(slab_records);
    auto pages=managed<int>(slab_records), slots=managed<int>(slab_records), counts=managed<int>(slice_rows);
    auto pos=managed<int64_t>(graph_rows);
    auto n=managed<int>(1);
    auto slab=managed<uint4>(size_t(slab_records)*vectors_per_record);
    for (int col=0; col<cols; ++col)
        table[col] = col%13 == 12 ? -1 : col%11 == 10 ? physical_pages : (col*7+2)%physical_pages;
    struct Case {int end, active;};
    for (Case c : {Case{0,0}, {73,73}, {32768,255}, {slab_records,256}, {slab_records+1,256}, {1048576,512}, {63,63}}) {
        *n=c.active;
        for(int i=0;i<graph_rows;++i) pos[i]=i<c.active ? c.end-c.active+i : -1;
        for(int s=0;s<2;++s) {
            for(int i=0;i<slab_records;++i) {
                unsigned x=unsigned(i)*2654435761u;
                selected[i] = i%17==0 ? -1 : i%19==0 ? c.end+7 : int(x%unsigned(std::max(c.end,1)));
            }
            prefill_record_pages<<<slice_rows,256>>>(selected,pos+s*slice_rows,table,pages,slots,counts,cols,0,physical_pages);
            prefill_record_cache<<<512,256>>>(peer,pages,slots,counts,slab,table,pos,n,graph_rows,s==0,cols,physical_pages);
            checked(cudaGetLastError());
            checked(cudaDeviceSynchronize());
            for(int row=0;row<slice_rows;++row) {
                int observed=0;
                for(int i=0;i<topk;++i) {
                    int id=selected[row*topk+i];
                    if(id<0 || id>pos[s*slice_rows+row]) continue;
                    int local=id/world, page=table[local/page_rows];
                    if(page<0 || page>=physical_pages) continue;
                    int at=row*topk+observed++;
                    int expected_page=c.end<=slab_records ? id : at;
                    if(pages[at]!=expected_page) { fprintf(stderr,"wrong page at %d, end %d\n",at,c.end); return 1; }
                    auto src=peer[id%world]+(page*page_rows+local%page_rows)*vectors_per_record;
                    auto dst=slab+size_t(pages[at])*vectors_per_record;
                    if(memcmp(src,dst,576)!=0) { fprintf(stderr,"wrong record %d end %d\n",id,c.end); return 1; }
                }
                if(counts[row]!=observed) return 1;
                // Dense MLA uses index zero for masked vector lanes.
                if(pages[row*topk]<0 || pages[row*topk]>=slab_records) return 1;
            }
        }
        printf("identical selected bytes at end=%d active=%d\n",c.end,c.active);
    }
    return 0;
}
