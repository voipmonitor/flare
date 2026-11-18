"""Combine a fixed Triton residual program with a producer or receiver for each local row."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import json
import re
import subprocess
import sys


def device_body(text,name,size=96):
    match=re.search(r"\.visible \.func "+name+rf"\(\s*\.param \.align 8 \.b8 (\w+)\[{size}\]\s*\)\s*\{{(.*?)\n\}}",text,re.S)
    if match is None:raise ValueError(f"cannot find by-value exchange function {name}")
    argument,body=match.groups()
    if body.count("ret;")!=1 or ".local " in body:
        raise ValueError("the communication body must have one exit and use registers for its configuration")
    body=body.replace(argument,"exchange_cfg").replace("$L__BB","exchange_"+name+"_BB")
    return "{\n"+body.replace("ret;",f"bra.uni {name}_done;")+f"\n{name}_done:\n}}\n"


def emit(args):
    source,output,digest,mode,helpers=args
    base=(source/(digest+".ptx")).read_text()
    if sha256(base.encode()).hexdigest()!=digest: raise ValueError("the source residual PTX changed")
    text=base.replace(".version 9.3",".version 9.2")
    if not re.search(r"\.reqntid (128|256)\n\{",text) or text.count("ret;")!=1:
        raise ValueError("expected one exit and four or eight warps in the canonical residual kernel")
    name="residual_row_"+mode
    text=text.replace(".visible .entry _mix(",".visible .entry "+name+"(")
    text,n=re.subn(r"\n\)\n(\.reqntid (?:128|256)\n\{)",r",\n\t.param .align 8 .b8 exchange_cfg[104]\n)\n\1",text,count=1)
    if n!=1:raise ValueError("the canonical residual argument list differs")
    start=re.search(r"\.reqntid (?:128|256)\n\{",text).end()
    prologue="""
    {
      .reg .u32 row, owner;
      .reg .pred other;
      mov.u32 row,%ctaid.x;
      ld.param.u32 owner,[exchange_cfg+80];
      setp.ne.u32 other,row,owner;
      @other bra.uni EXCHANGE_ROWS;
    }
"""
    text=text[:start]+prologue+text[start:]
    # get the original single return, after the owned-row arithmetic
    text=text.replace("\tret;", "\nEXCHANGE_ROWS:\n"+helpers["exchange_stripes"]+"\n\tret;")
    out=output/(digest+"-"+mode+".cubin");ptx=out.with_suffix(".ptx");ptx.write_text(text)
    p=subprocess.run(["/usr/local/cuda-13.2/bin/ptxas","-O3","-arch=sm_120a",str(ptx),"-o",str(out)],capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr[:3000])
    return digest,mode,dict(file=out.name,sha256=sha256(out.read_bytes()).hexdigest(),symbol=name)


def main():
    source,epilogue,output=[Path(x) for x in sys.argv[1:4]];output.mkdir(exist_ok=True,parents=True)
    sites=json.loads((source/"sites.json").read_text());helpers={"exchange_stripes":device_body(epilogue.read_text(),"exchange_stripes",104)}
    tasks=[(source,output,d,m,helpers) for d in sorted({r["ptx"] for r in sites}) for m in ("stripes",)]
    programs={}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for digest,mode,record in pool.map(emit,tasks):programs.setdefault(digest,{})[mode]=record
    (output/"programs.json").write_text(json.dumps(dict(operations=sites,programs=programs,
        exchange_ptx=sha256(epilogue.read_bytes()).hexdigest()),indent=2)+"\n")
    print(len(tasks),"residual programs")


if __name__=="__main__":main()
