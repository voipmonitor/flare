"""Stage new P4 graph resources from a Flare package using the same fixed row program as the resident."""

from argparse import ArgumentParser
from collections import deque
from hashlib import sha256
from pathlib import Path
import json
import struct
import sys

from tools.kimi31_collective_graph_audit import tensor
from tools.kimi31_package_collective_geometry import read_graph


def topology(nodes,edges):
    parents=[[] for _ in nodes];children=[[] for _ in nodes]
    for e in edges:
        parents[e.destination].append((e.source,e.data));children[e.source].append((e.destination,e.data))
    pending=[len(g) for g in parents];ready=deque(i for i,n in enumerate(pending) if not n);order=[]
    while ready:
        i=ready.popleft();order.append(i)
        for j,_ in children[i]:
            pending[j]-=1
            if pending[j]==0:ready.append(j)
    if len(order)!=len(nodes):raise ValueError("native graph contains a cycle")
    return parents,children,order


def pack_config(red,rank,runtime):
    sw,lane=divmod(rank,4);a=red.arguments
    if [len(b.data) for b in a]!=[8]*23+[4] or len(a)!=24:
        raise ValueError("expected the recorded four-island transport ABI")
    packed=bytearray(120);rel=[];offset=0
    for j in [*(sw*4+i for i in range(4)),*(i*4+lane for i in range(4)),*range(16,22)]:
        blob=a[j]
        if j<=17 and not any(blob.data) and not blob.relocations:raise ValueError("the required local or homologous peer pointer is absent")
        if j>=18 and blob.relocations:raise ValueError("a transport extent was recorded as a pointer")
        packed[offset:offset+8]=blob.data
        for old in blob.relocations:
            if old.offset!=0 or old.kind!=0:raise ValueError("the fused parameter requires a scalar device pointer")
            r=runtime.Relocation();r.CopyFrom(old);r.offset=offset;rel.append(r)
        offset+=8
    struct.pack_into("<II",packed,112,lane,sw)
    result=runtime.Blob(data=bytes(packed));result.relocations.extend(rel);return result


def stage_rank(directory,primary,locations,rank,programs,program_dir,runtime,package,out):
    graph,=[g for g in rank.graphs if g.name=="speculative1x8-tokens4"]
    nodes,edges=read_graph(directory,primary,locations,rank.resources[graph.resource],graph,runtime)
    parents,children,order=topology(nodes,edges)
    symbols=[rank.functions[n.kernel.function].symbol if n.HasField("kernel") else "" for n in nodes]
    mix=[i for i in order if symbols[i]=="_mix"]
    if len(mix)!=187:raise ValueError(f"rank {rank.rank} does not have 187 target mixtures")
    removed={};modules=[];functions=[];ids={};relocated=0
    for site,(idx,site_data) in enumerate(zip(mix[1:],programs["operations"],strict=True),1):
        (src,edge),=parents[idx]
        if (site!=site_data["ordinal"] or children[src]!=[(idx,bytes(8))] or edge!=bytes(8)
            or any(data!=bytes(8) for _,data in parents[src]) or "IslandRS" not in symbols[src]):
            raise ValueError(f"rank {rank.rank} site {site} violates the fused dependency contract")
        mk=nodes[idx].kernel;red=nodes[src].kernel;args=red.arguments
        parse=lambda n:int.from_bytes(args[n].data,"little")
        if (list(mk.grid)!=[4,1,1] or list(mk.block)!=[32*site_data["warps"],1,1]
            or list(red.grid)!=[64,1,1] or list(red.block)!=[1024,1,1] or (parse(22),parse(23))!=(28672,4)
            or args[17]!=mk.arguments[1] or not parse(18)<parse(19)<parse(20) or parse(21)<3584):
            raise ValueError(f"rank {rank.rank} site {site} launch or input alias differs")
        f=rank.functions[mk.function];info=programs["programs"][site_data["ptx"]]["direct"]
        key=(info["sha256"],f.SerializeToString(deterministic=True))
        if key not in ids:
            name="row-reduction-"+info["sha256"]
            if not any(m["name"]==name for m in modules):
                img=program_dir/info["file"]
                if sha256(img.read_bytes()).hexdigest()!=info["sha256"]:raise ValueError("compiled row image changed")
                modules.append(dict(name=name,file=img.name,sha256=info["sha256"]))
            nf=package.Function();nf.CopyFrom(f);nf.module=name;nf.symbol=info["symbol"];nf.parameter_sizes.append(120)
            ids[key]=len(rank.functions)+len(functions);functions.append(nf)
        cfg=pack_config(red,rank.rank,runtime);mk.arguments.append(cfg);relocated+=len(cfg.relocations)
        mk.function=ids[key];mk.grid[0]=64
        attrs=[x for x in mk.attributes if x.id==2]
        if len(attrs)!=1 or len(attrs[0].value.data)!=64 or attrs[0].value.relocations:raise ValueError("invalid cooperative attribute")
        attrs[0].value.data=struct.pack("<I",1)+bytes(60)
        removed[src]=idx
    remap={j:n for n,j in enumerate(j for j in range(len(nodes)) if j not in removed)}
    keep=[nodes[i] for i in range(len(nodes)) if i not in removed]
    new=[]
    for e in edges:
        if e.source in removed:continue
        dest=removed.get(e.destination,e.destination)
        new.append(runtime.Edge(source=remap[e.source],destination=remap[dest],data=e.data))
    topology(keep,new)
    if len(keep)!=len(nodes)-186 or len(new)!=len(edges)-186:raise ValueError("unexpected graph contraction")
    captured=sorted({n.kernel.function for n in keep if n.HasField("kernel")})
    ng=package.Graph();ng.CopyFrom(graph);ng.node_count=len(keep);ng.edge_count=len(new);del ng.functions[:];ng.functions.extend(captured)
    dst=out/f"rank-{rank.rank}";dst.mkdir(exist_ok=True,parents=True)
    blob=bytearray()
    for record in [*keep,*new]:
        data=record.SerializeToString(deterministic=True);blob+=struct.pack("<I",len(data))+data
    (dst/"graph.bin").write_bytes(blob)
    change=dict(rank=rank.rank,base_rank_sha256=sha256(rank.SerializeToString(deterministic=True)).hexdigest(),
                target=graph.name,source_nodes=len(nodes),nodes=len(keep),source_edges=len(edges),edges=len(new),
                raw_graph_sha256=sha256(blob).hexdigest(),unchanged_graphs=len(rank.graphs)-1,modules=modules,
                graph_descriptor=ng.SerializeToString(deterministic=True).hex(),
                appended_functions=[f.SerializeToString(deterministic=True).hex() for f in functions],
                embedded_pointer_relocations=relocated)
    (dst/"change.json").write_text(json.dumps(change,indent=2)+"\n")
    return {k:change[k] for k in ("rank","source_nodes","nodes","source_edges","edges","raw_graph_sha256","unchanged_graphs")}


def main():
    p=ArgumentParser(description=__doc__);p.add_argument("source",type=Path);p.add_argument("programs",type=Path)
    p.add_argument("schema",type=Path);p.add_argument("output",type=Path);a=p.parse_args();sys.path.insert(0,str(a.schema.resolve()))
    import package_pb2 as package
    import runtime_package_pb2 as runtime
    primary=a.source/"model.safetensors";body=tensor(primary,"flare.manifest");manifest=package.ProtectedPackage.FromString(body)
    locations={r.payload_id:r for r in package.LocationSet.FromString(tensor(primary,"flare.locations")).locations}
    if len(manifest.ranks)!=16:raise ValueError("the specialization requires the 16-rank native model")
    a.output.mkdir(exist_ok=True,parents=True)
    programs=json.loads((a.programs/"programs.json").read_text())
    ranks=[stage_rank(a.source,primary,locations,package.RankBody.FromString(item.body),programs,a.programs,
                      runtime,package,a.output) for item in manifest.ranks]
    (a.output/"resource-plan.json").write_text(json.dumps(dict(input_manifest_sha256=sha256(body).hexdigest(),
        compiled_program_manifest=sha256((a.programs/"programs.json").read_bytes()).hexdigest(),ranks=ranks),indent=2)+"\n")
    print(len(ranks),"rank graphs staged with 186 merged sites each")


if __name__=="__main__":main()
