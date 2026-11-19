"""Replace signed four-row expert calls with an ordered FC1, SiTU and FC2 graph sequence."""

import argparse
import hashlib
import json
from pathlib import Path

from stage import BASE_PRIMARY, EXPERT_IMAGE, EXPERT_TEXT, digest, kernel_parameter_sizes, raw_graph, require
from stage_pcie import append_span, frame, resource_bytes
from elf_text import cubin_code
from tools.kimi31_collective_graph_audit import tensor
from tools.kimi31_package_collective_geometry import read_graph
import package_pb2 as capture
import runtime_package_pb2 as runtime


PHASES = {
    "fc1": "2ee66f19fac9506b783dc07c6af54835def4df89bfec9885b51829147bcc4b6e",
    "activation": "a92e1515526d8e54643ba50db7576a192209ba4cf6146117d3d88c40821c3127",
    "fc2": "02048b103d3efcbc6257194776e9580c20a5948d658faecd0356c5c62f78e730",
}
P4 = "speculative1x8-tokens4"
ZERO = bytes(8)


def load_core(folder):
    primary = folder / "model.safetensors"
    manifest = tensor(primary, "flare.manifest")
    ranks = [capture.RankBody.FromString(s.body) for s in capture.ProtectedPackage.FromString(manifest).ranks]
    require([r.rank for r in ranks] == list(range(16)), "expected 16 ordered ranks")
    locations = capture.LocationSet.FromString(tensor(primary, "flare.locations"))
    return (folder, primary, {v.payload_id: v for v in locations.locations}), manifest, ranks


def resolve(core, rank, symbol):
    (graph,) = (g for g in rank.graphs if g.name == P4)
    nodes, edges = read_graph(*core, rank.resources[graph.resource], graph, runtime)
    sites = [
        i
        for i, n in enumerate(nodes)
        if n.HasField("kernel") and rank.functions[n.kernel.function].symbol == symbol
    ]
    require(len(sites) == 92, f"rank {rank.rank} has {len(sites)} four-row expert sites")
    (original,) = {nodes[i].kernel.function for i in sites}
    fn = rank.functions[original]
    (module,) = (m for m in rank.modules if m.name == fn.module)
    image = resource_bytes(*core, rank.resources[module.cubin])
    require(
        digest(image).hex() == EXPERT_IMAGE and cubin_code(image) == (symbol, EXPERT_TEXT),
        "expert source image differs",
    )
    checks = []
    for i in sites:
        k = nodes[i].kernel
        require(
            list(k.grid) == [376, 1, 1] and list(k.block) == [128, 1, 1] and k.shared == 34816,
            "expert launch geometry differs",
        )
        require(
            [int.from_bytes(a.data, "little") for a in k.arguments[-3:]] == [896, 0, 4]
            and all(not a.relocations for a in k.arguments),
            "expected 896 signed experts and four active rows",
        )
        require(
            any(a.id == 2 and int.from_bytes(a.value.data[:4], "little") == 1 for a in k.attributes),
            "whole-grid cooperative admission is required for both projections",
        )
        normalized = runtime.Node()
        normalized.CopyFrom(nodes[i])
        normalized.kernel.function = 0
        checks.append(normalized.SerializeToString(deterministic=True))
    return graph, nodes, edges, sites, fn, checks


def append_phases(rank, images, template, first_id, payload):
    functions = []
    offset = 0
    for n, (phase, image, encoded) in enumerate(images):
        resource = rank.resources.add(resource_id=first_id + n, kind=capture.EXECUTABLE_KIND_CUBIN)
        append_span(resource, image, encoded, offset)
        resource.stored.payload_id = payload
        mod = f"p4-expert-{phase}-{digest(image).hex()}"
        rank.modules.add(name=mod, cubin=len(rank.resources) - 1)
        fn = rank.functions.add()
        fn.CopyFrom(template)
        fn.module = mod
        functions.append(len(rank.functions) - 1)
        offset += len(encoded)
    rank.payloads.add(id=payload, nbytes=offset)
    return functions


def expand(nodes, edges, sites, functions):
    selected = set(sites)
    starts, ends, inserted = {}, {}, []
    result = []
    for i, node in enumerate(nodes):
        starts[i] = len(result)
        if i in selected:
            for stage, fid in enumerate(functions):
                phase = runtime.Node()
                phase.CopyFrom(node)
                phase.kernel.function = fid
                if stage == 1:
                    phase.kernel.grid[:] = [12, 1, 1]
                    phase.kernel.shared = 0
                    del phase.kernel.attributes[:]
                result.append(phase)
                if stage:
                    inserted.append(
                        runtime.Edge(source=len(result) - 2, destination=len(result) - 1, data=ZERO)
                    )
        else:
            result.append(node)
        ends[i] = len(result) - 1

    revised = []
    for edge in edges:
        e = runtime.Edge()
        e.CopyFrom(edge)
        e.source = ends[edge.source]
        e.destination = starts[edge.destination]
        revised.append(e)
    # Contracting each three-node path restores every source node and edge,
    # including dependencies not ordered by their index in the record stream.
    require(
        all(
            (e.source, e.destination, e.data) == (ends[old.source], starts[old.destination], old.data)
            for e, old in zip(revised, edges, strict=True)
        ),
        "external graph dependency changed",
    )
    require(
        len(result) - len(nodes) == 184 and len(inserted) == 184,
        "expected two extra kernels and edges per routed layer",
    )
    return result, revised + inserted


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("control", "source", "programs", "output"):
        cli.add_argument(arg, type=Path)
    args = cli.parse_args()
    reference, _, baseline = load_core(args.control)
    with reference[1].open("rb") as fd:
        require(hashlib.file_digest(fd, "sha256").hexdigest() == BASE_PRIMARY, "unexpected numerical control")
    source, manifest, ranks = load_core(args.source)
    images = []
    abi, symbol = None, None
    for phase, expected in PHASES.items():
        path = args.programs / (phase + ".cubin")
        image = path.read_bytes()
        sym, _ = cubin_code(image)
        sizes = kernel_parameter_sizes(path)
        require(digest(image).hex() == expected, f"{phase} differs from the measured program")
        if abi is None:
            abi, symbol = sizes, sym
        require(abi == sizes and symbol == sym, "stage entry point or parameter layout differs")
        images.append((phase, image, frame(image)))
    encoded = b"".join(v[2] for v in images)
    code = digest(encoded)
    first_id = 1 + max(v.resource_id for r in ranks for v in r.resources)
    args.output.mkdir()
    (args.output / "expert-stages.zst").write_bytes(encoded)
    files = [
        dict(
            id=code.hex(),
            filename="expert-stages.zst",
            relative=f"deployments/gb202-tp16/expert-{code.hex()[:20]}.safetensors",
        )
    ]
    audits = []
    for ref, r in zip(baseline, ranks, strict=True):
        require(
            r.allocations == ref.allocations and r.initializers == ref.initializers,
            "signed source expert weights or allocations differ from numerical control",
        )
        graph, nodes, edges, sites, template, signatures = resolve(source, r, symbol)
        _, _, _, _, ref_template, ref_signatures = resolve(reference, ref, symbol)
        require(
            signatures == ref_signatures
            and template == ref_template
            and list(template.parameter_sizes) == abi,
            "expert graph bindings differ from the numerical control",
        )
        ids = append_phases(r, images, template, first_id + 3 * r.rank, code)
        next_nodes, next_edges = expand(nodes, edges, sites, ids)
        graph.node_count, graph.edge_count = len(next_nodes), len(next_edges)
        graph.functions[:] = sorted({n.kernel.function for n in next_nodes if n.HasField("kernel")})
        raw = raw_graph(next_nodes, next_edges)
        packed = frame(raw)
        graph_id = digest(packed)
        res = r.resources[graph.resource]
        append_span(res, raw, packed, 0)
        res.stored.payload_id = graph_id
        r.payloads.add(id=graph_id, nbytes=len(packed))
        name = f"rank-{r.rank}-split-expert-graph.zst"
        (args.output / name).write_bytes(packed)
        (args.output / f"rank-{r.rank}.pb").write_bytes(r.SerializeToString(deterministic=True))
        files.append(
            dict(
                id=graph_id.hex(),
                filename=name,
                relative=f"deployments/{r.deployment_id}/rank-{r.rank}/expert-{graph_id.hex()[:20]}.safetensors",
            )
        )
        audits.append(
            dict(
                rank=r.rank,
                functions=ids,
                original_sites=sites,
                graph_payload=graph_id.hex(),
                source_bindings_sha256=digest(b"".join(signatures)).hex(),
            )
        )
        print(f"rank {r.rank} verified all 92 signed inputs and graph boundaries", flush=True)
    (args.output / "resources.json").write_text(
        json.dumps(dict(source_manifest=digest(manifest).hex(), files=files), indent=2) + "\n"
    )
    (args.output / "expert-sites.json").write_text(json.dumps(audits, indent=2) + "\n")


if __name__ == "__main__":
    main()
