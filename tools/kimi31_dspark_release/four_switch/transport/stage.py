"""Bind the TP16 vocabulary exchange and expert selector to their signed verification graphs."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import struct

from elf_text import cubin_code
from stage_pcie import append_span, frame, resource_bytes
from tools.kimi31_collective_graph_audit import tensor
from tools.kimi31_native_row_resources import topology
from tools.kimi31_package_collective_geometry import read_graph
import package_pb2 as wire
import runtime_package_pb2 as runtime


def scalar(arg):
    if arg.relocations:
        raise ValueError("the sampled gather scalar has relocations")
    return int.from_bytes(arg.data, "little")


def encode(nodes, edges):
    records = []
    for v in [*nodes, *edges]:
        raw = v.SerializeToString(deterministic=True)
        records.extend([struct.pack("<I", len(raw)), raw])
    return b"".join(records)


def contract(nodes, edges, folded):
    kept = [node for i, node in enumerate(nodes) if i not in folded]
    index = {old: new for new, old in enumerate(i for i in range(len(nodes)) if i not in folded)}
    revised = []
    for edge in edges:
        if edge.destination in folded:
            assert edge.source == folded[edge.destination]
            continue
        v = runtime.Edge()
        v.CopyFrom(edge)
        v.source = index[folded.get(edge.source, edge.source)]
        v.destination = index[edge.destination]
        revised.append(v)
    topology(kept, revised)
    return kept, revised


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--vocabulary", type=Path, help="four-switch compiled and signed gather directories")
    p.add_argument("--router", type=Path, help="compiled and oracle router fixture directory")
    a = p.parse_args()
    assert a.vocabulary or a.router
    a.output.mkdir()
    primary = a.source / "model.safetensors"
    manifest = tensor(primary, "flare.manifest")
    locations = {
        v.payload_id: v for v in wire.LocationSet.FromString(tensor(primary, "flare.locations")).locations
    }
    ranks = [wire.RankBody.FromString(s.body) for s in wire.ProtectedPackage.FromString(manifest).ranks]
    assert [r.rank for r in ranks] == list(range(16))
    first_id = 1 + max(v.resource_id for r in ranks for v in r.resources)
    files = []
    changes = []
    for rank in ranks:
        r = rank.rank
        mods = {v.name: v for v in rank.modules}
        packed = bytearray()
        kernel_resources = []
        functions = {}

        def original(k):
            fn = rank.functions[k.function]
            return resource_bytes(a.source, primary, locations, rank.resources[mods[fn.module].cubin])

        def install(role, image, sizes):
            if role not in functions:
                comp = frame(image)
                symbol, _ = cubin_code(image)
                name = f"{role}-{sha256(image).hexdigest()}"
                resource = rank.resources.add(
                    resource_id=first_id + 2 * r + len(functions), kind=wire.EXECUTABLE_KIND_CUBIN
                )
                kernel_resources.append(len(rank.resources) - 1)
                append_span(resource, image, comp, len(packed))
                packed.extend(comp)
                rank.modules.add(name=name, cubin=kernel_resources[-1])
                functions[role] = len(rank.functions)
                rank.functions.add(module=name, symbol=symbol, parameter_sizes=sizes)
            return functions[role]

        if a.vocabulary:
            stem = f"gather-{r}-1280.cubin"
            vocab = (a.vocabulary / "cross-first" / stem).read_bytes()
            vocab_control = (a.vocabulary / "signed" / stem).read_bytes()
        if a.router:
            folder = a.router / f"rank-{r}"
            rows = [ln.split() for ln in (folder / "programs.txt").read_text().splitlines()]
            cfg = {line[1]: line for line in rows if line[0] == "4"}
            route = (folder / cfg["fused"][2]).read_bytes()
            shared = int(cfg["fused"][6])
            gather_control = (folder / cfg["base"][2]).read_bytes()
            top_control = (folder / cfg["select"][2]).read_bytes()
        graph_records = bytearray()
        target_resources = []
        counts = Counter()
        for g in rank.graphs:
            if g.name not in [f"speculative1x8-tokens{i}" for i in range(2, 9)]:
                continue
            width = int(g.name.rsplit("tokens", 1)[1])
            if not a.vocabulary and width != 4:
                continue
            nodes, edges = read_graph(a.source, primary, locations, rank.resources[g.resource], g, runtime)
            parents, children, order = topology(nodes, edges)

            def kind(i):
                n = nodes[i]
                return (
                    rank.functions[n.kernel.function].symbol
                    if n.HasField("kernel")
                    else n.WhichOneof("operation")
                )

            if a.vocabulary:
                start = next(i for i in order if kind(i) == "_markov_step")
                gather = start
                for prefix in ("copy", "triton_poi_fused", "RecursiveGather"):
                    ((gather, flag),) = parents[gather]
                    assert flag == bytes(8) and prefix in kind(gather)
                k = nodes[gather].kernel
                assert len(k.arguments) == 39 and k.arguments[1] == nodes[start].kernel.arguments[0]
                assert 1 <= scalar(k.arguments[35]) <= 7 and scalar(k.arguments[36]) == 0x294000
                assert (
                    original(k) == vocab_control
                    and list(k.grid) == [8, 1, 1]
                    and list(k.block) == [128, 1, 1]
                )
                k.function = install(
                    "cross-switch-vocabulary", vocab, list(rank.functions[k.function].parameter_sizes)
                )
                counts["vocabulary"] += 1
            folded = {}
            if width == 4 and a.router:
                sites = [i for i in order if "RowBlockAllGatherPair" in kind(i)]
                assert len(sites) == 92
                for pos, i in enumerate(sites):
                    ((j, flag),) = children[i]
                    assert flag == bytes(8) and parents[j] == [(i, flag)] and "WarpTopK16" in kind(j)
                    k = nodes[i].kernel
                    t = nodes[j].kernel
                    assert (
                        t.arguments[0] == k.arguments[3] and len(t.arguments) == 4 and len(k.arguments) == 37
                    )
                    assert (
                        list(k.grid) == list(t.grid) == [4, 1, 1]
                        and list(k.block) == [512, 1, 1]
                        and list(t.block) == [256, 1, 1]
                    )
                    if pos == 0:
                        pair = (k.function, t.function)
                        assert original(k) == gather_control and original(t) == top_control
                    else:
                        assert (k.function, t.function) == pair
                    f = install("gather-routed-experts", route, [8] * 39 + [4])
                    fused = runtime.Node()
                    fused.CopyFrom(nodes[i])
                    fused.kernel.function = f
                    fused.kernel.shared = shared
                    del fused.kernel.arguments[:]
                    fused.kernel.arguments.extend([*k.arguments[:4], *t.arguments[1:], *k.arguments[4:]])
                    nodes[i].CopyFrom(fused)
                    folded[j] = i
                    counts["router"] += 1
            if folded:
                nodes, edges = contract(nodes, edges, folded)
            g.node_count = len(nodes)
            g.edge_count = len(edges)
            g.functions[:] = sorted({n.kernel.function for n in nodes if n.HasField("kernel")})
            raw = encode(nodes, edges)
            compressed = frame(raw)
            append_span(rank.resources[g.resource], raw, compressed, len(graph_records))
            graph_records.extend(compressed)
            target_resources.append(g.resource)
        assert (not a.vocabulary or counts["vocabulary"] == 7) and (not a.router or counts["router"] == 92)
        for label, data, resources in [
            ("kernels", packed, kernel_resources),
            ("graphs", graph_records, target_resources),
        ]:
            digest = sha256(data).digest()
            rank.payloads.add(id=digest, nbytes=len(data))
            for i in resources:
                rank.resources[i].stored.payload_id = digest
            filename = f"rank-{r}-{label}.zst"
            (a.output / filename).write_bytes(data)
            rel = f"deployments/{rank.deployment_id}/rank-{r}/{label}-{digest.hex()[:20]}.safetensors"
            files.append(dict(id=digest.hex(), filename=filename, relative=rel))
        (a.output / f"rank-{r}.pb").write_bytes(rank.SerializeToString(deterministic=True))
        changes.append(dict(rank=r, **counts))
    (a.output / "resources.json").write_text(
        json.dumps(dict(source_manifest=sha256(manifest).hexdigest(), files=files), indent=2) + "\n"
    )
    print(changes)


if __name__ == "__main__":
    main()
