"""Dump rank-specific native CUDA images for fixed prepared gather and reduction choices."""

from argparse import ArgumentParser
import json
import hashlib
from pathlib import Path
from unittest.mock import patch
import subprocess
import struct

import cutlass.cute as cute
import b12x.comm.pcie._island_rs_cute as island
import b12x.comm.pcie._runtime_gather_cute as gather


from elf_text import cubin_code


def build(rank,spec,output,measure_abi=False,offline=None):
    key=spec["key"];d=output/f"rank-{rank}"/key;d.mkdir(parents=True,exist_ok=True)
    record=d/"image.json"
    if record.is_file():
        data=json.loads(record.read_text());image=d/data["file"]
        if hashlib.sha256(image.read_bytes()).hexdigest()==data["sha256"] and (not measure_abi or "argument_sizes" in data):return data
        raise ValueError("inconsistent compiled image")
    def emit(op,*args,**kwargs):
        return cute.compile(op,*args,options=f"--keep-cubin --gpu-arch=sm_120a --dump-dir={d.resolve()}",**({"no_jit_engine":True} if offline is not None else {}))
    if spec["type"]=="ring":
        with patch.object(gather,"compile_kernel",emit),patch.object(gather,"attach_programs",lambda fn,*deps:fn):
            launch=gather.compile_selected_gather(16,rank,spec["words"],spec["threads"],8,0,"ring")
    else:
        with patch.object(island,"b12x_compile",emit),patch.object(island,"attach_programs",lambda fn,*deps:fn):
            launch=island.get_island_rs_launcher(16,rank,rank,threads=spec["threads"],wait_nanosleep_cycles=24,
                local_push=spec["push"],cross_island_rsag=spec["scattered"])
    files=list(d.rglob("*.cubin"))
    if len(files)!=1:raise ValueError(f"{key} produced {len(files)} cubins")
    image=files[0].read_bytes();sym,digest=cubin_code(image)
    data=dict(rank=rank,file=str(files[0].relative_to(d)),symbol=sym,code_sha256=digest,
              sha256=hashlib.sha256(image).hexdigest(),**spec)
    if measure_abi:
        import torch
        from cuda.bindings import driver as cuda
        from ignition.bench.graph_patch import call, parameter_layout
        from ignition.preparation.capture_stream import capture_stream
        device=torch.device("cuda",0);stream=capture_stream(device)
        slab=torch.empty(96<<20,device=device,dtype=torch.uint8)
        inp=torch.empty(2<<20,device=device,dtype=torch.uint8);out=torch.empty(32<<20,device=device,dtype=torch.uint8)
        graph=torch.cuda.CUDAGraph(keep_graph=True)
        try:
            with torch.cuda.graph(graph,stream=stream):
                if spec["type"]=="ring":
                    launch(inp.data_ptr(),out.data_ptr(),slab.data_ptr(),[slab.data_ptr()]*16,[slab.data_ptr()]*16,1,1<<20,1<<20,2<<20)
                else:
                    launch([slab.data_ptr()]*16,inp.data_ptr(),out.data_ptr(),98816,29458944,36798976,917504,14336,64,stream=stream,active_blocks=4)
            raw=cuda.CUgraph(graph.raw_cuda_graph());_,cnt=call(cuda.cuGraphGetNodes,raw)
            nodes,_=call(cuda.cuGraphGetNodes,raw,cnt)
            if len(nodes)!=1:raise ValueError("single-kernel native replacement requires exactly one captured node")
            k=call(cuda.cuGraphKernelNodeGetParams,nodes[0]);data["shared"]=k.sharedMemBytes
            data["argument_sizes"]=[n for _,n in parameter_layout(k.func)]
            if k.blockDimX!=spec["threads"] or k.gridDimX!=(8 if spec["type"]=="ring" else 64):
                raise ValueError("captured launch geometry differs")
        finally:graph.reset()
    if offline is not None:
        ref=offline[str(rank)][key]
        if data["sha256"]!=ref["sha256"] or data["code_sha256"]!=ref["code_sha256"]:
            raise ValueError(f"cold build of {key} differs from the measured image on rank {rank}")
        for field in ("shared","argument_sizes"):
            if field in ref:data[field]=ref[field]
    record.write_text(json.dumps(data,indent=2)+"\n");return data


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__);parser.add_argument("spec",type=Path);parser.add_argument("output",type=Path)
    parser.add_argument("--rank",type=int,action="append",required=True)
    parser.add_argument("--capture-abi",action="store_true")
    parser.add_argument("--offline-abi",type=Path)
    args=parser.parse_args()
    reference=json.loads(args.offline_abi.read_text()) if args.offline_abi else None
    from contextlib import ExitStack
    from cuda.bindings import driver as cuda
    with ExitStack() as owned:
        if reference is not None:
            if args.capture_abi:raise ValueError("run GPU ABI capture or a verified offline rebuild")
            for module in (gather,island):
                owned.enter_context(patch.object(module,"current_cuda_stream",lambda:cuda.CUstream(0)))
        for rank in args.rank:
            for spec in json.loads(args.spec.read_text()):build(rank,spec,args.output,args.capture_abi,reference)
