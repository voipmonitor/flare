"""Build one CTA per row for TP16 reduction followed by the original residual arithmetic."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from kimi31_residual_stripes_build import device_body


def emit(args):
    source,dest,digest,mode,body=args
    ptx=(source/(digest+".ptx")).read_text()
    if sha256(ptx.encode()).hexdigest()!=digest:raise ValueError("residual PTX changed")
    ptx=ptx.replace(".version 9.3",".version 9.2").replace(".visible .entry _mix(",".visible .entry reduce_row_"+mode+"(")
    ptx,n=re.subn(r"\n\)\n(\.reqntid (?:128|256)\n\{)",r",\n\t.param .align 8 .b8 exchange_cfg[120]\n)\n\1",ptx,count=1)
    if n!=1:raise ValueError("unexpected residual launch dimensions")
    if mode=="direct":
        pattern=r"ld\.param\.[bu]64\s+(%rd[0-9]+),\s*\[_mix_param_1\];"
        regs=re.findall(pattern,ptx)
        if len(regs)!=1:raise ValueError("the canonical residual must load one delta base pointer")
        select="""
    {
        .reg .u64 base,off,stride,extra,index,addr;
        .reg .u32 lane,row,parity;
        .reg .pred p;
        ld.param.u32 lane,[exchange_cfg+112];
        ld.param.u64 base,[exchange_cfg];
        setp.eq.u32 p,lane,1;
        @p ld.param.u64 base,[exchange_cfg+8];
        setp.eq.u32 p,lane,2;
        @p ld.param.u64 base,[exchange_cfg+16];
        setp.eq.u32 p,lane,3;
        @p ld.param.u64 base,[exchange_cfg+24];
        mov.u32 row,%ctaid.x;
        mul.wide.u32 index,row,4;
        add.u64 addr,base,index;
        ld.relaxed.gpu.global.u32 parity,[addr];
        and.b32 parity,parity,1;
        ld.param.u64 off,[exchange_cfg+96];
        ld.param.u64 stride,[exchange_cfg+104];
        shl.b64 stride,stride,4;
        setp.ne.u32 p,parity,0;
        selp.u64 extra,stride,0,p;
        add.u64 addr,base,off;
        add.u64 REG,addr,extra;
    }
"""
        ptx=re.sub(pattern,select.replace("REG",regs[0]),ptx,count=1)
    start=re.search(r"\.reqntid (?:128|256)\n\{",ptx).end()
    check="""
    {
        .reg .u32 row;
        .reg .pred inactive;
        mov.u32 row,%ctaid.x;
        setp.ge.u32 inactive,row,4;
        @inactive ret;
    }
"""
    ptx=ptx[:start]+"\n"+body+check+ptx[start:]
    path=dest/(digest+"-"+mode+".ptx");path.write_text(ptx);image=path.with_suffix(".cubin")
    result=subprocess.run(["/usr/local/cuda-13.2/bin/ptxas","-O3","-arch=sm_120a",str(path),"-o",str(image)],capture_output=True,text=True)
    if result.returncode:raise ValueError(result.stderr[:1500])
    return digest,mode,dict(file=image.name,symbol="reduce_row_"+mode,sha256=sha256(image.read_bytes()).hexdigest())


def main():
    source,helper,dest=map(Path,sys.argv[1:4]);dest.mkdir(exist_ok=True,parents=True)
    code=helper.read_text();sites=json.loads((source/"sites.json").read_text());programs={}
    roles={r:device_body(code,f"row_to_{suffix}",120) for r,suffix in (("buffer","buffer"),("direct","mixture"))}
    jobs=[(source,dest,d,r,body) for d in sorted({i["ptx"] for i in sites}) for r,body in roles.items()]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for digest,role,item in pool.map(emit,jobs):programs.setdefault(digest,{})[role]=item
    (dest/"programs.json").write_text(json.dumps(dict(operations=sites,programs=programs,helper_sha256=sha256(helper.read_bytes()).hexdigest()),indent=2)+"\n")
    print(len(jobs),"row-reduction specializations")


if __name__=="__main__":main()
