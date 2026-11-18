"""Bind verified FlashInfer images to eligible graphs of the signed TP16 Kimi package."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
import struct
import subprocess
from pathlib import Path
import package_pb2 as wire
import runtime_package_pb2 as rt
from tools.kimi31_collective_graph_audit import tensor,decode_frame
from tools.kimi31_package_collective_geometry import read_graph
from elf_text import cubin_code

CHUNK=4<<20

def resource_bytes(folder,primary,loc,res):
    use=res.stored;location=loc[use.payload_id]
    f=folder/location.file if location.file else primary
    raw=tensor(f,location.tensor,use.offset,use.nbytes)
    if list(res.chunk_sha256)!=[sha256(raw[i:i+res.chunk_nbytes]).digest() for i in range(0,len(raw),res.chunk_nbytes)]:
        raise ValueError("source executable resource checksum differs")
    return decode_frame(raw,res.decoded_nbytes)


def frame(raw):
    return subprocess.run(["zstd","-q","-3","-c"],input=raw,stdout=subprocess.PIPE,check=True).stdout


def append_span(res,raw,stored,offset):
    res.decoded_nbytes=len(raw);res.chunk_nbytes=CHUNK
    res.stored.offset=offset;res.stored.nbytes=len(stored)
    del res.chunk_sha256[:]
    res.chunk_sha256.extend(sha256(stored[i:i+CHUNK]).digest() for i in range(0,len(stored),CHUNK))


def stage(a):
    primary=a.source/"model.safetensors";manifest=tensor(primary,"flare.manifest")
    pkg=wire.ProtectedPackage.FromString(manifest)
    loc={i.payload_id:i for i in wire.LocationSet.FromString(tensor(primary,"flare.locations")).locations}
    ranks=[wire.RankBody.FromString(x.body) for x in pkg.ranks]
    if [r.rank for r in ranks]!=list(range(16)):raise ValueError("expected a single TP16 deployment")
    resid=max(x.resource_id for r in ranks for x in r.resources)+1
    a.out.mkdir(exist_ok=True,parents=True);files=[];reports=[]
    for r in ranks:
        num=r.rank;mods={m.name:m for m in r.modules};source_funcs=len(r.functions);funcs={};match={};modids={}
        sub=f"rank-{num}"
        old={};new={}
        for spec in json.loads(a.spec.read_text()):
            key=spec["key"]
            oldrec=json.loads((a.old/sub/key/"image.json").read_text());newrec=json.loads((a.new/sub/key/"image.json").read_text())
            old[oldrec["code_sha256"]]=key;new[key]=newrec
        image_bytes=bytearray();graph_bytes=bytearray();new_mod_resources=[];revised_graph_resources=[];report={}
        for g in r.graphs:
            res=r.resources[g.resource];nodes,edges=read_graph(a.source,primary,loc,res,g,rt);counts=Counter()
            for n in nodes:
                if not n.HasField("kernel"):continue
                k=n.kernel;orig=r.functions[k.function];name=orig.symbol
                ring="RingGather" in name
                if not ring:
                    if "IslandRSLaunch" not in name or not g.name.startswith("prefill"):continue
                    if len(k.arguments)!=24:raise ValueError("unexpected source island signature")
                    if int.from_bytes(k.arguments[22].data,"little")<1835008:continue
                if orig.module not in match:
                    oldres=r.resources[mods[orig.module].cubin]
                    _,digest=cubin_code(resource_bytes(a.source,primary,loc,oldres))
                    if digest not in old:raise ValueError(f"unsigned geometry guess refused for rank {num}, {orig.module} ({name})")
                    match[orig.module]=old[digest]
                key=match[orig.module];info=new[key]
                if (info["type"]=="ring")!=ring or info["threads"]!=k.block[0] or k.grid[0]!=(8 if ring else 64):
                    raise ValueError(f"source binary geometry differs at rank {num}, {g.name}, {key}")
                if not ring:
                    if not info["scattered"] or not info["push"]:raise ValueError("only the streamed island protocol is replaced")
                    for length in (8,4,4,4):k.arguments.add(data=bytes(length))
                if [len(v.data) for v in k.arguments]!=info["argument_sizes"]:raise ValueError(f"native parameter mismatch for {key}")
                if key not in modids:
                    image=(a.new/sub/key/info["file"]).read_bytes()
                    if sha256(image).hexdigest()!=info["sha256"]:raise ValueError("compiled program differs from the selected build")
                    stored=frame(image);module_name=f"pcie-{info['sha256']}"
                    nr=r.resources.add(resource_id=resid,kind=wire.EXECUTABLE_KIND_CUBIN);resid+=1
                    append_span(nr,image,stored,len(image_bytes));image_bytes.extend(stored)
                    modids[key]=module_name;new_mod_resources.append(len(r.resources)-1)
                    r.modules.add(name=module_name,cubin=len(r.resources)-1)
                fid=(k.function,key)
                if fid not in funcs:
                    f=r.functions.add();f.CopyFrom(orig);f.module=modids[key];f.symbol=info["symbol"]
                    del f.parameter_sizes[:];f.parameter_sizes.extend(info["argument_sizes"]);funcs[fid]=len(r.functions)-1
                k.function=funcs[fid];k.shared=info["shared"];counts[key]+=1
            if not counts:continue
            raw=bytearray()
            for v in [*nodes,*edges]:
                buf=v.SerializeToString(deterministic=True);raw.extend(struct.pack("<I",len(buf)));raw.extend(buf)
            stored=frame(bytes(raw));append_span(res,raw,stored,len(graph_bytes));graph_bytes.extend(stored);revised_graph_resources.append(g.resource)
            del g.functions[:];g.functions.extend(sorted({n.kernel.function for n in nodes if n.HasField("kernel")}))
            report[g.name]=dict(counts)
        if not report or not modids:raise ValueError("no matched graph programs")
        for label,data,indices in (("kernels",image_bytes,new_mod_resources),("graphs",graph_bytes,revised_graph_resources)):
            id=sha256(data).digest()
            r.payloads.add(id=id,nbytes=len(data))
            for ix in indices:r.resources[ix].stored.payload_id=id
            fname=f"{sub}-{label}.zst";(a.out/fname).write_bytes(data)
            files.append(dict(id=id.hex(),filename=fname,relative=f"deployments/{r.deployment_id}/{sub}/prepared-pcie-{label}.safetensors"))
        (a.out/f"{sub}.pb").write_bytes(r.SerializeToString(deterministic=True))
        reports.append(dict(rank=num,graphs=report,modules=modids,added_functions=len(r.functions)-source_funcs))
    (a.out/"resources.json").write_text(json.dumps(dict(source_manifest=sha256(manifest).hexdigest(),files=files),indent=2)+"\n")
    (a.out/"changes.json").write_text(json.dumps(reports,indent=2)+"\n")
    print("Verified old machine code and staged replacements on",len(ranks),"ranks",flush=True)

if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    for n in ("source","spec","old","new","out"):p.add_argument(n,type=Path)
    stage(p.parse_args())
