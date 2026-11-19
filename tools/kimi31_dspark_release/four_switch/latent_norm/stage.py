"""Contract only the 92 exact four-row expert reductions and their immediately dependent canonical normalizers."""

import json, sys, struct
from pathlib import Path
from hashlib import sha256
import package_pb2 as p, runtime_package_pb2 as rt
from tools.kimi31_collective_graph_audit import tensor
from tools.kimi31_package_collective_geometry import read_graph
from tools.kimi31_native_row_resources import pack_config, topology
from stage_pcie import frame, append_span, resource_bytes
from elf_text import cubin_code

base, build, out = map(Path, sys.argv[1:])
out.mkdir()
f = base / "model.safetensors"
manifest = tensor(f, "flare.manifest")
pkg = p.ProtectedPackage.FromString(manifest)
loc = {x.payload_id: x for x in p.LocationSet.FromString(tensor(f, "flare.locations")).locations}
ranks = [p.RankBody.FromString(s.body) for s in pkg.ranks]
assert [r.rank for r in ranks] == list(range(16))
rid = 1 + max(x.resource_id for r in ranks for x in r.resources)
image = (build / "fused-buffer.cubin").read_bytes()
packed = frame(image)
cid = sha256(packed).digest()
label = "p4-fused-latent-norm"
module = label + "-" + sha256(image).hexdigest()
(out / "kernel.zst").write_bytes(packed)
files = [dict(id=cid.hex(), filename="kernel.zst", relative=f"deployments/gb202-tp16/{label}.safetensors")]
summary = []
for r in ranks:
    g = next(x for x in r.graphs if x.name == "speculative1x8-tokens4")
    nodes, edges = read_graph(base, f, loc, r.resources[g.resource], g, rt)
    parents, children, order = topology(nodes, edges)

    def symbol(n):
        return r.functions[n.kernel.function].symbol if n.HasField("kernel") else n.WhichOneof("operation")

    def imm(b):
        return int.from_bytes(b.data, "little")

    sites = [
        i
        for i in order
        if "ParallelPushIslandRS" in symbol(nodes[i]) and imm(nodes[i].kernel.arguments[22]) == 14336
    ]
    assert len(sites) == 92
    mods = {m.name: m for m in r.modules}

    def code(fn):
        return cubin_code(resource_bytes(base, f, loc, r.resources[mods[r.functions[fn].module].cubin]))[1]

    (source_r,) = {nodes[i].kernel.function for i in sites}
    assert code(source_r) == cubin_code((build / f"rank-{r.rank}.cubin").read_bytes())[1]
    idx = len(r.functions)
    nr = r.resources.add(resource_id=rid + r.rank, kind=p.EXECUTABLE_KIND_CUBIN)
    append_span(nr, image, packed, 0)
    nr.stored.payload_id = cid
    r.modules.add(name=module, cubin=len(r.resources) - 1)
    r.payloads.add(id=cid, nbytes=len(packed))
    fused = r.functions.add()
    skipped = {}
    change = []
    for i in sites:
        ((j, flags),) = children[i]
        assert flags == bytes(8) and parents[j] == [(i, bytes(8))]
        red = nodes[i].kernel
        norm = nodes[j].kernel
        assert symbol(nodes[j]) == "_rms_norm"
        assert list(red.grid) == [64, 1, 1] and list(red.block) == [512, 1, 1]
        assert list(norm.grid) == [4, 1, 1] and list(norm.block) == [256, 1, 1] and norm.shared == 8192
        assert [len(v.data) for v in red.arguments] == [8] * 23 + [4] and [
            imm(v) for v in red.arguments[18:]
        ] == [98816, 29458944, 36798976, 917504, 14336, 4]
        assert (
            norm.arguments[0] == red.arguments[17]
            and [len(v.data) for v in norm.arguments] == [8] * 4 + [4] * 3 + [8] * 2
            and list(map(imm, norm.arguments[4:6])) == [3584, 224]
        )
        if not change:
            assert code(norm.function) == "2180edc46e524f7280f287c004adbc08592243dd58bb34c300368d7f4692daca"
            original_norm = norm.function
            fused.CopyFrom(r.functions[norm.function])
            fused.symbol = "reduce_norm_buffer"
            fused.module = module
            fused.parameter_sizes.append(120)
        assert norm.function == original_norm
        config = pack_config(red, r.rank, rt)
        new = rt.KernelNode() if hasattr(rt, "KernelNode") else type(red)()
        new.CopyFrom(norm)
        new.grid[:] = red.grid
        new.function = idx
        new.arguments.append(config)
        nodes[i].kernel.CopyFrom(new)
        skipped[j] = i
        change.append(dict(reduction=i, normalization=j))
    keep = [i for i in range(len(nodes)) if i not in skipped]
    ix = {i: j for j, i in enumerate(keep)}
    kept = [nodes[i] for i in keep]
    newedges = []
    for edge in edges:
        if edge.destination in skipped:
            assert edge.source == skipped[edge.destination]
            continue
        v = rt.Edge()
        v.CopyFrom(edge)
        v.source = ix[skipped.get(edge.source, edge.source)]
        v.destination = ix[edge.destination]
        newedges.append(v)
    assert len(kept) == len(nodes) - 92 and len(newedges) == len(edges) - 92
    topology(kept, newedges)
    g.node_count = len(kept)
    g.edge_count = len(newedges)
    g.functions[:] = sorted({v.kernel.function for v in kept if v.HasField("kernel")})
    raw = bytearray()
    for v in [*kept, *newedges]:
        b = v.SerializeToString(deterministic=True)
        raw += struct.pack("<I", len(b)) + b
    z = frame(raw)
    did = sha256(z).digest()
    res = r.resources[g.resource]
    append_span(res, raw, z, 0)
    res.stored.payload_id = did
    filename = f"rank-{r.rank}-{label}.zst"
    (out / filename).write_bytes(z)
    r.payloads.add(id=did, nbytes=len(z))
    files.append(
        dict(
            id=did.hex(),
            filename=filename,
            relative=f"deployments/{r.deployment_id}/rank-{r.rank}/{label}-graph.safetensors",
        )
    )
    (out / f"rank-{r.rank}.pb").write_bytes(r.SerializeToString(deterministic=True))
    summary.append(
        dict(
            rank=r.rank,
            source_nodes=len(nodes),
            source_edges=len(edges),
            candidate_nodes=len(kept),
            candidate_edges=len(newedges),
            sites=change,
        )
    )
(out / "changes.json").write_text(json.dumps(summary, indent=2) + "\n")
(out / "resources.json").write_text(
    json.dumps(dict(source_manifest=sha256(manifest).hexdigest(), files=files), indent=2) + "\n"
)
print("contracted", len(sites), "pairs on each of", len(ranks), "ranks")
