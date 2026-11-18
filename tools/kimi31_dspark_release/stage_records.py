"""Bind a paged record slab to the signed TP16 DSpark prefill graphs."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import re
import struct

import package_pb2 as wire
import runtime_package_pb2 as rt
from tools.kimi31_collective_graph_audit import tensor
from tools.kimi31_native_row_resources import topology
from tools.kimi31_package_collective_geometry import read_graph
from stage_pcie import frame, append_span

U32 = struct.Struct("<I")
U64 = struct.Struct("<Q")
SLICE = 128
TOPK = 2048
LAYER_COUNT = 24
RECORD_BYTES = 576


def require(value, message):
    if not value:
        raise ValueError(message)


def scalar(blob):
    require(not blob.relocations and len(blob.data) in (4, 8), "expected a fixed native scalar parameter")
    return int.from_bytes(blob.data, "little")


def bounds(rank, addr, nbytes):
    spans = [a for a in rank.allocations if a.address <= addr and addr + nbytes <= a.address + a.nbytes]
    require(len(spans) == 1, f"address {addr:#x} of extent {nbytes} lacks unique fixed backing")


def set_kernel(kernel, fid, args, blocks, threads):
    old = list(args)
    del kernel.arguments[:]
    kernel.arguments.extend(old)
    kernel.function = fid
    kernel.grid[:] = [blocks, 1, 1]
    kernel.block[:] = [threads, 1, 1]
    kernel.shared = 0


def patch(rank, directory, primary, locations, graph, rows, functions, query_length):
    nodes, edges = read_graph(directory, primary, locations, rank.resources[graph.resource], graph, rt)
    parents, children, order = topology(nodes, edges)

    def symbol(i):
        return rank.functions[nodes[i].kernel.function].symbol if nodes[i].HasField("kernel") else ""

    selected = [i for i in order if symbol(i) == "_selection_slots"]
    gathered = [i for i in order if symbol(i) == "_gather_records"]
    attended = [i for i in order if "DenseMlaForwardKernel" in symbol(i)]
    require(
        len(selected) == len(gathered) == len(attended) == LAYER_COUNT * (rows // SLICE),
        f"unexpected sparse attention sites in {graph.name}",
    )
    original_pages = None
    for sequence, (si, gi, ai) in enumerate(zip(selected, gathered, attended, strict=True)):
        sk, gk, ak = (nodes[i].kernel for i in (si, gi, ai))
        s, g, d = sk.arguments, gk.arguments, ak.arguments
        require([z for z, _ in parents[gi]] == [si], "record gather is not ordered after selection")
        (intermediate,) = [z for z, _ in parents[ai]]
        require(
            "StaticFp8Quant" in symbol(intermediate) and [z for z, _ in parents[intermediate]] == [gi],
            "attention does not follow the record gather and query quantizer",
        )
        require(
            list(sk.grid) == [SLICE, 1, 1]
            and list(gk.grid) == [SLICE * TOPK, 1, 1]
            and len(s) == 12
            and len(g) == 7
            and len(d) == 20
            and all(not a.relocations for a in [*s, *g, *d])
            and [len(a.data) for a in s[:10]] == [8] * 7 + [4] * 3
            and [len(a.data) for a in g[:5]] == [8] * 5,
            "unexpected source record ABI",
        )
        require(
            list(s[3:6]) == list(g[1:4])
            and scalar(s[7]) == 512
            and scalar(s[8]) == 0
            and 0 < scalar(s[9]) <= 0x7FFFFFFF // SLICE,
            "source slot geometry differs",
        )
        require(
            all(len(a.data) == 16 for a in d[1:4])
            and d[1].data[:8] == g[4].data
            and d[3].data[:8] == g[3].data
            and scalar(d[14]) == RECORD_BYTES
            and scalar(d[16]) == TOPK
            and scalar(d[17]) == SLICE,
            "attention descriptor differs from the selected-record layout",
        )
        page_pointer = U64.unpack_from(d[2].data)[0]
        if original_pages is None:
            original_pages = page_pointer
            bounds(rank, page_pointer, SLICE * TOPK * 4)
        require(original_pages == page_pointer, "attention page indices unexpectedly alias another table")
        part = sequence % (rows // SLICE)
        if part == 0:
            pos0, table, peer, slab = scalar(s[1]), scalar(s[2]), scalar(g[0]), scalar(g[4])
            bounds(rank, pos0, rows * 8)
            bounds(rank, slab, SLICE * TOPK * RECORD_BYTES)
            bounds(rank, scalar(s[3]), SLICE * TOPK * 4)
        require(
            (scalar(s[1]), scalar(s[2]), scalar(g[0]), scalar(g[4]))
            == (pos0 + part * SLICE * 8, table, peer, slab),
            "layer slice storage or position stride changed",
        )
        args_g = [
            *g[:5],
            s[2],
            rt.Blob(data=U64.pack(pos0)),
            rt.Blob(data=U64.pack(query_length)),
            rt.Blob(data=U32.pack(rows)),
            rt.Blob(data=U32.pack(int(part == 0))),
            s[7],
            s[9],
        ]
        args_s = [*s[:6], *s[7:10]]
        d[2].data = s[3].data + d[2].data[8:]
        set_kernel(gk, functions[1], args_g, 512, 256)
        set_kernel(sk, functions[0], args_s, SLICE, 256)
    graph.functions[:] = sorted({n.kernel.function for n in nodes if n.HasField("kernel")})
    result = bytearray()
    for value in [*nodes, *edges]:
        item = value.SerializeToString(deterministic=True)
        result += U32.pack(len(item)) + item
    return result, len(selected)


def stage_package(programs, label, patch_graph):
    parser = ArgumentParser(description=__doc__)
    for arg in ("source", "cubin", "out"):
        parser.add_argument(arg, type=Path)
    a = parser.parse_args()
    a.out.mkdir()
    f = a.source / "model.safetensors"
    raw_manifest = tensor(f, "flare.manifest")
    pkg = wire.ProtectedPackage.FromString(raw_manifest)
    loc = {x.payload_id: x for x in wire.LocationSet.FromString(tensor(f, "flare.locations")).locations}
    ranks = [wire.RankBody.FromString(x.body) for x in pkg.ranks]
    require([r.rank for r in ranks] == list(range(16)), "expected the sixteen-rank deployment")
    cubin = a.cubin.read_bytes()
    kernel_data = frame(cubin)
    image_hash = sha256(kernel_data).digest()
    asset = f"{label}-kernels.zst"
    (a.out / asset).write_bytes(kernel_data)
    files = [
        dict(
            id=image_hash.hex(),
            filename=asset,
            relative=f"deployments/gb202-tp16/{label}-kernels.safetensors",
        )
    ]
    module = label + "-" + sha256(cubin).hexdigest()
    resource_id = 1 + max(x.resource_id for r in ranks for x in r.resources)
    for rank in ranks:
        (b,) = (b for b in rank.execution.buffers if b.name == "query_lengths")
        (backing,) = (v for v in rank.allocations if v.name == b.reference.allocation)
        require(b.dtype == "int32" and list(b.shape) == [1], "only C1 query lengths are supported")
        query_length = backing.address + b.reference.offset
        resource = rank.resources.add(resource_id=resource_id + rank.rank, kind=wire.EXECUTABLE_KIND_CUBIN)
        append_span(resource, cubin, kernel_data, 0)
        resource.stored.payload_id = image_hash
        rank.payloads.add(id=image_hash, nbytes=len(kernel_data))
        rank.modules.add(name=module, cubin=len(rank.resources) - 1)
        functions = []
        for name, sizes in programs:
            functions.append(len(rank.functions))
            rank.functions.add(module=module, symbol=name, parameter_sizes=sizes)
        edits = []
        graphs = bytearray()
        for graph in rank.graphs:
            match = re.fullmatch(r"prefill(\d+)(?:-no-mtp)?", graph.name)
            if not match or int(match[1]) < 512:
                continue
            raw, count = patch_graph(rank, a.source, f, loc, graph, int(match[1]), functions, query_length)
            compressed = frame(raw)
            target = rank.resources[graph.resource]
            append_span(target, raw, compressed, len(graphs))
            graphs += compressed
            edits.append((graph.resource, graph.name, count))
        require(len(edits) == 8, "expected four prefill capacities with and without drafting")
        digest = sha256(graphs).digest()
        rank.payloads.add(id=digest, nbytes=len(graphs))
        for index, _, _ in edits:
            rank.resources[index].stored.payload_id = digest
        name = f"rank-{rank.rank}-{label}.zst"
        (a.out / name).write_bytes(graphs)
        files.append(
            dict(
                id=digest.hex(),
                filename=name,
                relative=f"deployments/{rank.deployment_id}/rank-{rank.rank}/{label}.safetensors",
            )
        )
        (a.out / f"rank-{rank.rank}.pb").write_bytes(rank.SerializeToString(deterministic=True))
        print(rank.rank, [(name, n) for _, name, n in edits], flush=True)
    (a.out / "resources.json").write_text(
        json.dumps(dict(source_manifest=sha256(raw_manifest).hexdigest(), files=files), indent=2) + "\n"
    )


if __name__ == "__main__":
    stage_package(
        (("prefill_record_pages", [8] * 6 + [4] * 3), ("prefill_record_cache", [8] * 8 + [4] * 4)),
        "prefill-records",
        patch,
    )
