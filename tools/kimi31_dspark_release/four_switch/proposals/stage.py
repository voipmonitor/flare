"""Bind the bounded Markov scan and draft-logit gather to single-request verification graphs."""

import json
from hashlib import sha256
from pathlib import Path
import struct
import sys

import package_pb2 as wire
import runtime_package_pb2 as runtime
from elf_text import cubin_code
from stage_pcie import append_span, frame, resource_bytes
from tools.kimi31_collective_graph_audit import tensor
from tools.kimi31_package_collective_geometry import read_graph
from tools.kimi31_native_row_resources import topology

base, kernels, dest = map(Path, sys.argv[1:])
dest.mkdir()
primary = base / "model.safetensors"
manifest = tensor(primary, "flare.manifest")
locations = {
    v.payload_id: v for v in wire.LocationSet.FromString(tensor(primary, "flare.locations")).locations
}
protected = wire.ProtectedPackage.FromString(manifest)
ranks = [wire.RankBody.FromString(s.body) for s in protected.ranks]
assert [r.rank for r in ranks] == list(range(16))
first_id = 1 + max(v.resource_id for r in ranks for v in r.resources)
oracle = cubin_code((kernels / "original.cubin").read_bytes())
info = json.loads((kernels / "programs.json").read_text())
images = {d: (kernels / f"prefix-{d}.cubin").read_bytes() for d in range(1, 7)}
frames = {d: frame(i) for d, i in images.items()}
packed = b"".join(frames.values())
kernel_id = sha256(packed).digest()
label = "packed-proposal-prefix"
(dest / "programs.zst").write_bytes(packed)
files = [
    dict(id=kernel_id.hex(), filename="programs.zst", relative=f"deployments/gb202-tp16/{label}.safetensors")
]
normal = bytes(8)


def integer(v):
    assert not v.relocations
    return int.from_bytes(v.data, "little")


for r in ranks:
    graphs = bytearray()
    touched = []
    parent_modules = {m.name: m for m in r.modules}
    position = 0
    r.payloads.add(id=kernel_id, nbytes=len(packed))
    for depth, image in images.items():
        (g,) = (g for g in r.graphs if g.name == f"speculative1x8-tokens{depth + 1}")
        nodes, edges = read_graph(base, primary, locations, r.resources[g.resource], g, runtime)
        parents, children, order = topology(nodes, edges)

        def symbol(i):
            node = nodes[i]
            return (
                r.functions[node.kernel.function].symbol
                if node.HasField("kernel")
                else node.WhichOneof("operation")
            )

        steps = [i for i in order if symbol(i) == "_markov_step"]
        assert len(steps) == 7 and all(list(nodes[i].kernel.grid) == [640, 1, 1] for i in steps)
        (ep,) = (i for i in order if symbol(i) == "_normalize" and list(nodes[i].kernel.grid) == [640, 1, 7])
        for i, j in zip(steps, steps[1:] + [ep], strict=True):
            assert children[i] == [(j, normal)] and parents[j] == [(i, normal)]
        n = nodes[ep].kernel
        old_fn = r.functions[n.function]
        old_image = resource_bytes(base, primary, locations, r.resources[parent_modules[old_fn.module].cubin])
        assert cubin_code(old_image) == oracle and list(n.block) == [128, 1, 1]
        for i in steps:
            k = nodes[i].kernel
            assert (
                k.arguments[6] == n.arguments[0]
                and k.arguments[7] == n.arguments[1]
                and k.arguments[8] == n.arguments[2]
                and k.arguments[9] == n.arguments[3]
                and k.arguments[5] == n.arguments[6]
                and integer(k.arguments[26]) == 163840
            )
        gather = steps[0]
        for name in ("copy", "triton_poi_fused", "RecursiveGather"):
            ((gather, flag),) = parents[gather]
            assert flag == normal and name in symbol(gather)
        gather_kernel = nodes[gather].kernel
        gather_args = gather_kernel.arguments
        assert (
            len(gather_args) == 39
            and gather_args[1] == nodes[steps[0]].kernel.arguments[0]
            and integer(gather_args[35]) == 7
            and [integer(v) for v in gather_args[36:]] == [0x294000, 0x294000, 0x528000]
            and list(gather_kernel.grid) == [8, 1, 1]
            and list(gather_kernel.block) == [128, 1, 1]
        )
        # The recursive gather addresses tile ownership with its runtime row count.
        # Preserve all 8 CTA epochs and both of the original staging banks.
        gather_args[35].data = depth.to_bytes(len(gather_args[35].data), "little")
        packed_frame = frames[depth]
        resource = r.resources.add(
            resource_id=first_id + 6 * r.rank + depth - 1, kind=wire.EXECUTABLE_KIND_CUBIN
        )
        append_span(resource, image, packed_frame, position)
        position += len(packed_frame)
        resource.stored.payload_id = kernel_id
        module = f"{label}-{depth}-{sha256(image).hexdigest()}"
        r.modules.add(name=module, cubin=len(r.resources) - 1)
        function = r.functions.add(
            module=module, symbol=old_fn.symbol, parameter_sizes=old_fn.parameter_sizes
        )
        assert cubin_code(image)[0] == function.symbol
        n.function = len(r.functions) - 1
        n.shared = info[str(depth)]["shared"]
        n.arguments[4].CopyFrom(nodes[steps[depth - 1]].kernel.arguments[12])
        n.arguments[5].CopyFrom(nodes[steps[depth - 1]].kernel.arguments[13])
        removed = set(steps[depth:])
        kept = [v for i, v in enumerate(nodes) if i not in removed]
        index = {old: new for new, old in enumerate(i for i in range(len(nodes)) if i not in removed)}
        revised = []
        for e in edges:
            if e.destination in removed:
                continue
            source = steps[depth - 1] if e.source in removed else e.source
            if e.source in removed:
                assert e.source == steps[-1] and e.destination == ep
            v = runtime.Edge()
            v.CopyFrom(e)
            v.source = index[source]
            v.destination = index[e.destination]
            revised.append(v)
        assert len(kept) == len(nodes) - (7 - depth) and len(revised) == len(edges) - (7 - depth)
        topology(kept, revised)
        g.node_count = len(kept)
        g.edge_count = len(revised)
        g.functions[:] = sorted({v.kernel.function for v in kept if v.HasField("kernel")})
        body = bytearray()
        for v in [*kept, *revised]:
            blob = v.SerializeToString(deterministic=True)
            body += struct.pack("<I", len(blob)) + blob
        compressed = frame(body)
        record = r.resources[g.resource]
        append_span(record, body, compressed, len(graphs))
        graphs += compressed
        touched.append(record)
    graph_id = sha256(graphs).digest()
    for rec in touched:
        rec.stored.payload_id = graph_id
    r.payloads.add(id=graph_id, nbytes=len(graphs))
    filename = f"rank-{r.rank}-graphs.zst"
    (dest / filename).write_bytes(graphs)
    files.append(
        dict(
            id=graph_id.hex(),
            filename=filename,
            relative=f"deployments/{r.deployment_id}/rank-{r.rank}/{label}-graphs.safetensors",
        )
    )
    (dest / f"rank-{r.rank}.pb").write_bytes(r.SerializeToString(deterministic=True))
    print("rank", r.rank, "updated", len(touched), "verification capacities", flush=True)
(dest / "resources.json").write_text(
    json.dumps(dict(source_manifest=sha256(manifest).hexdigest(), files=files), indent=2) + "\n"
)
