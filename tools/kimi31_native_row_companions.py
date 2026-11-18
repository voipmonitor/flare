"""Prepare rank and payload metadata for a separate native development package with fused P4 rows."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from tools.kimi31_collective_graph_audit import tensor


def zstd(data):
    return subprocess.run(["/usr/bin/zstd","-q","-3","-c"],input=data,stdout=subprocess.PIPE,check=True).stdout


def main():
    source,staged,programs,schema,out=map(Path,sys.argv[1:6]);sys.path.insert(0,str(schema.resolve()))
    import package_pb2 as wire
    manifest=tensor(source/"model.safetensors","flare.manifest");envelope=wire.ProtectedPackage.FromString(manifest)
    plan=json.loads((staged/"resource-plan.json").read_text())
    if sha256(manifest).hexdigest()!=plan["input_manifest_sha256"]:raise ValueError("native source manifest has changed")
    ranks=[wire.RankBody.FromString(r.body) for r in envelope.ranks]
    if len(ranks)!=16 or [r.rank for r in ranks]!=list(range(16)):raise ValueError("this build uses sixteen ordered ranks")
    resource_id=max(res.resource_id for r in ranks for res in r.resources)+1
    out.mkdir(parents=True,exist_ok=True)
    sample=json.loads((staged/"rank-0/change.json").read_text());frames={};raw=bytearray()
    for item in sample["modules"]:
        data=(programs/item["file"]).read_bytes()
        if sha256(data).hexdigest()!=item["sha256"]:raise ValueError("compiled cubin identity differs")
        cr=zstd(data);frames[item["name"]]=(len(raw),len(data),cr);raw.extend(cr)
    code_id=sha256(raw).digest();(out/"row-kernels.zst").write_bytes(raw)
    files=[dict(id=code_id.hex(),filename="row-kernels.zst",relative=f"deployments/{ranks[0].deployment_id}/row-kernels.safetensors")]
    for rank in ranks:
        r=rank.rank;folder=staged/f"rank-{r}";change=json.loads((folder/"change.json").read_text())
        if sha256(rank.SerializeToString(deterministic=True)).hexdigest()!=change["base_rank_sha256"] or change["modules"]!=sample["modules"]:
            raise ValueError(f"rank {r} source or module catalog differs")
        rank.payloads.add(id=code_id,nbytes=len(raw))
        for mod in sample["modules"]:
            off,plain,stored=frames[mod["name"]];n=len(rank.resources)
            res=rank.resources.add(resource_id=resource_id,kind=wire.EXECUTABLE_KIND_CUBIN,decoded_nbytes=plain,chunk_nbytes=4<<20)
            resource_id+=1;res.stored.payload_id=code_id;res.stored.offset=off;res.stored.nbytes=len(stored)
            res.chunk_sha256.extend(sha256(stored[i:i+(4<<20)]).digest() for i in range(0,len(stored),4<<20))
            rank.modules.add(name=mod["name"],cubin=n)
        for func in change["appended_functions"]:rank.functions.add().ParseFromString(bytes.fromhex(func))
        graph,=[g for g in rank.graphs if g.name==change["target"]]
        graph.ParseFromString(bytes.fromhex(change["graph_descriptor"]));res=rank.resources[graph.resource]
        data=(folder/"graph.bin").read_bytes()
        if sha256(data).hexdigest()!=change["raw_graph_sha256"]:raise ValueError("staged CUDA graph changed")
        stored=zstd(data);identity=sha256(stored).digest();name=f"rank-{r}-graph.zst"
        (out/name).write_bytes(stored);files.append(dict(id=identity.hex(),filename=name,
            relative=f"deployments/{rank.deployment_id}/rank-{r}/row-graph.safetensors"))
        rank.payloads.add(id=identity,nbytes=len(stored))
        res.stored.payload_id=identity;res.stored.offset=0;res.stored.nbytes=len(stored);res.decoded_nbytes=len(data)
        del res.chunk_sha256[:];res.chunk_sha256.extend(sha256(stored[i:i+res.chunk_nbytes]).digest() for i in range(0,len(stored),res.chunk_nbytes))
        (out/f"rank-{r}.pb").write_bytes(rank.SerializeToString(deterministic=True))
    (out/"resources.json").write_text(json.dumps(dict(source_manifest=sha256(manifest).hexdigest(),files=files),indent=2)+"\n")
    print(f"{len(ranks)} updated rank bodies and {len(files)} new executable companions")


if __name__=="__main__":main()
