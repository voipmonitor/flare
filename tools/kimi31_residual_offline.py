"""Recreate Kimi's fixed four-row residual programs from model configs without GPU access or weight loading."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource,compile,make_backend
from triton.runtime.jit import MockTensor,create_function_from_signature

from ignition.recipes.kimi3.residual import TILES,_mix


def main():
    model,draft,out=map(Path,sys.argv[1:4]);out.mkdir(exist_ok=True,parents=True)
    c=json.loads(model.read_text())["text_config"];taps=json.loads(draft.read_text())["aux_hidden_state_layer_ids"]
    h,L,sz,eps=c["hidden_size"],c["num_hidden_layers"],c["attn_res_block_size"],c["rms_norm_eps"]
    if (h,L,sz)!=(7168,93,12) or len(taps)!=5:raise ValueError("the fixed program catalog requires Kimi K3.1 and its five DSpark taps")
    target=GPUTarget("cuda",120,32);backend=make_backend(target)
    binder=create_function_from_signature(_mix.signature,_mix.params,backend)
    bf,fp=MockTensor(torch.bfloat16),MockTensor(torch.float32)
    blocks=(L+sz-1)//sz
    cases=[];compiled={}
    for site in range(1,2*L+1):
        layer=site//2;active=(layer+sz-1)//sz
        mode=1;write=-1;tap=False
        if layer<L:
            if site%2:
                active+=int(layer%sz==0)
                if layer%sz==0:mode=2
            else:
                if layer%sz==0:write=layer//sz
                tap=layer in taps
        tile,warps=TILES[active]
        key=(active,write,mode,tap)
        if key not in compiled:
            args=(bf,bf,bf,fp,bf,bf,bf,h,h,h*blocks,h,h,h*(len(taps) if tap else 1),eps,eps,h,active,write,mode,
                  tap,triton.next_power_of_2(h),tile)
            kwargs=dict(num_warps=warps,debug=False,instrumentation_mode="")
            bound,specialization,opts=binder(*args,**kwargs)
            opts,signature,constexprs,attrs=_mix._pack_args(backend,kwargs,bound,specialization,opts)
            kernel=compile(ASTSource(_mix,signature,constexprs,attrs),target=target,options=opts.__dict__)
            ptx=kernel.asm["ptx"].encode();image=kernel.asm["cubin"];digest=sha256(ptx).hexdigest()
            (out/(digest+".ptx")).write_bytes(ptx);(out/(digest+".cubin")).write_bytes(image)
            compiled[key]=dict(ptx=digest,cubin=sha256(image).hexdigest(),warps=warps,shared=kernel.metadata.shared)
        cases.append(dict(ordinal=site,**compiled[key]))
    (out/"sites.json").write_text(json.dumps(cases,indent=2)+"\n")
    print(f"{len(cases)} sites, {len(compiled)} compiled residual specializations")


if __name__=="__main__":main()
