"""Bind the measured DCP16 decode epilogue and single-field gather programs to a signed core."""

from pathlib import Path
from hashlib import sha256
from argparse import ArgumentParser
from collections import Counter
import json, struct, re
import package_pb2 as wire, runtime_package_pb2 as rt
from tools.kimi31_collective_graph_audit import tensor
from tools.kimi31_package_collective_geometry import read_graph
from tools.kimi31_native_row_resources import topology
from stage_pcie import frame, append_span, resource_bytes
from elf_text import cubin_code

MIB = 1 << 20
WORLD = 16
DENSE_CODE = "92c7ad0528913c88ad2b51865040a381a189b3977efec2c5beeb3b1ac73a52b8"


def check(x, why):
    if not x:
        raise ValueError(why)


def number(a):
    check(not a.relocations and len(a.data) in (4, 8), "non-scalar native immediate")
    return int.from_bytes(a.data, "little")


def scalar(n, size):
    return rt.Blob(data=n.to_bytes(size, "little"))


def peer(ns):
    return rt.Blob(data=struct.pack("<16Q", *ns))


def prefix(blob, new):
    check(not blob.relocations, "variable-address descriptor")
    blob.data = struct.pack("<Q", new) + blob.data[8:]


def emit(fid, grid, threads, args):
    n = rt.Node()
    n.kernel.function = fid
    n.kernel.grid.extend(grid)
    n.kernel.block.extend((threads, 1, 1))
    n.kernel.arguments.extend(args)
    return n


def run(a):
    a.out.mkdir()
    f = a.base / "model.safetensors"
    signed = tensor(f, "flare.manifest")
    m = wire.ProtectedPackage.FromString(signed)
    loc = {x.payload_id: x for x in wire.LocationSet.FromString(tensor(f, "flare.locations")).locations}
    ranks = [wire.RankBody.FromString(s.body) for s in m.ranks]
    check([r.rank for r in ranks] == list(range(16)), "expected TP16")
    idbase = 1 + max(o.resource_id for r in ranks for o in r.resources)
    image = a.image.read_bytes()
    packed = frame(image)
    digest = sha256(packed).digest()
    module = "decode-peer-records-" + sha256(image).hexdigest()
    path = "decode-partial-and-gather.zst"
    (a.out / path).write_bytes(packed)
    files = [
        dict(id=digest.hex(), filename=path, relative="deployments/gb202-tp16/" + module + ".safetensors")
    ]
    gcfg = json.loads(a.gathers.read_text())
    report = []
    for rr in ranks:
        rank = rr.rank
        mods = {v.name: v.cubin for v in rr.modules}
        shapes = {
            int(n): cubin_code((a.control / f"gather-{rank}-{n}.cubin").read_bytes())[1] for n in (17, 216)
        }
        source_sass = {}

        def symbol(k):
            return rr.functions[k.function].symbol

        def sass(k):
            mod = rr.functions[k.function].module
            if mod not in source_sass:
                source_sass[mod] = cubin_code(resource_bytes(a.base, f, loc, rr.resources[mods[mod]]))[1]
            return source_sass[mod]

        (p4,) = (g for g in rr.graphs if g.name == "speculative1x8-tokens4")
        ns, _ = read_graph(a.base, f, loc, rr.resources[p4.resource], p4, rt)
        local = [n.kernel for n in ns if n.HasField("kernel") and "local_candidate_slots" in symbol(n.kernel)]
        check(len(local) == 24, "expected signed four-row owner selection")
        ackdata = local[0].arguments[7].data
        roots = struct.unpack("<32Q", ackdata)[:16]
        check(
            all(
                k.arguments[7].data == ackdata
                and number(k.arguments[8]) >= 32 * MIB
                and number(k.arguments[9]) == rank
                for k in local
            ),
            "candidate channel identity is inconsistent",
        )
        (arena,) = [
            v
            for v in rr.allocations
            if v.address <= roots[rank] and roots[rank] + 32 * MIB <= v.address + v.nbytes
        ]
        # The untouched 128-row ring stores fifteen remote 16-KiB rows per
        # query. Other signed candidate transports consume smaller row counts.
        # Larger prefill graphs use the separately verified 4-MiB staging view.
        res = rr.resources.add(resource_id=idbase + rank, kind=wire.EXECUTABLE_KIND_CUBIN)
        append_span(res, image, packed, 0)
        res.stored.payload_id = digest
        rr.payloads.add(id=digest, nbytes=len(packed))
        rr.modules.add(name=module, cubin=len(rr.resources) - 1)
        functions = {}
        for name, sizes in [
            ("partials_ready", [128, 4]),
            ("combine_cooperative", [128, 8, 4]),
            *[(s, [8, 8, 128, 128, 4, 4, 4, 8]) for s in sorted({v["function"] for v in gcfg.values()})],
        ]:
            functions[name] = len(rr.functions)
            rr.functions.add(module=module, symbol=name, parameter_sizes=sizes)
        graphs = bytearray()
        changes = {}
        touched = []
        for g in rr.graphs:
            shape = re.fullmatch(r"speculative1x8-tokens([2-8])", g.name)
            if not shape:
                continue
            rows = int(shape[1])
            nodes, edges = read_graph(a.base, f, loc, rr.resources[g.resource], g, rt)
            parents, children, order = topology(nodes, edges)
            sites = [
                i
                for i in order
                if nodes[i].HasField("kernel") and "LseReduceScatter" in symbol(nodes[i].kernel)
            ]
            check(len(sites) == 24, "missing DCP LSE sites")
            replaced = Counter()
            if not a.only_gathers:
                for j, end in enumerate(sites):
                    cp, edge = parents[end][0]
                    src, edge2 = parents[cp][0]
                    check(
                        parents[end] == [(cp, bytes(8))]
                        and children[cp] == [(end, bytes(8))]
                        and parents[cp] == [(src, bytes(8))]
                        and children[src] == [(cp, bytes(8))],
                        "LSE chain has another consumer or an early dependency",
                    )
                    dense = nodes[src].kernel
                    old = nodes[end].kernel
                    da = dense.arguments
                    oa = old.arguments
                    check(
                        sass(dense) == DENSE_CODE
                        and list(dense.grid) == [rows, 12, 1]
                        and len(da) == 20
                        and da[5].data[:8] == oa[0].data
                        and len(da[5].data) == 40
                        and len(oa) == 45,
                        "producer layout differs from the matched native image",
                    )
                    copy = nodes[cp].copy
                    refs = {v.name: v.address for v in rr.allocations}
                    check(
                        copy.nbytes == rows * 96 * 4
                        and refs[copy.source.allocation] + copy.source.offset == number(da[6])
                        and refs[copy.destination.allocation] + copy.destination.offset == number(oa[1]),
                        "unexpected LSE scratch copy or layout",
                    )
                    stage = [b + 30 * MIB + (j % 2) * MIB for b in roots]
                    base = stage[rank]
                    prefix(da[5], base)
                    prefix(da[6], base + 786432)
                    prefix(da[7], base)
                    prefix(da[8], base + 786432)
                    output = rt.Blob.FromString(oa[2].SerializeToString())
                    signals = peer([number(x) for x in oa[19:35]])
                    nodes[cp].CopyFrom(
                        emit(functions["partials_ready"], list(old.grid), 32, [signals, scalar(rank, 4)])
                    )
                    nodes[end].CopyFrom(
                        emit(
                            functions["combine_cooperative"],
                            [rows, 6, 1],
                            256,
                            [peer(stage), output, scalar(rank, 4)],
                        )
                    )
                    replaced["LSE producer and combine"] += 1
            for i in order:
                if not nodes[i].HasField("kernel"):
                    continue
                k = nodes[i].kernel
                if "RecursiveGather" not in symbol(k):
                    continue
                args = k.arguments
                if len(args) != 39 or number(args[35]) != rows:
                    continue
                v = {0x8A00: 17, 0xD8C00: 216}.get(number(args[36]))
                if v is None or f"{v}/{rows}" not in gcfg:
                    continue
                if sass(k) != shapes[v]:
                    continue
                cfg = gcfg[f"{v}/{rows}"]
                w = int(cfg["threads"])
                blocks = int(cfg["blocks"])
                check(number(args[36]) >= 16 * rows * v * 16, "insufficient peer staging")
                new = emit(
                    functions[cfg["function"]],
                    [blocks, 1, 1],
                    w,
                    [
                        args[0],
                        args[1],
                        peer([number(x) for x in args[3:19]]),
                        peer([number(x) for x in args[19:35]]),
                        scalar(rank, 4),
                        args[35],
                        scalar(v, 4),
                        args[36],
                    ],
                )
                nodes[i].CopyFrom(new)
                replaced["single gather " + str(v)] += 1
            if not replaced:
                continue
            g.functions[:] = sorted({n.kernel.function for n in nodes if n.HasField("kernel")})
            raw = bytearray()
            for n in [*nodes, *edges]:
                b = n.SerializeToString(deterministic=True)
                raw += struct.pack("<I", len(b)) + b
            z = frame(raw)
            target = rr.resources[g.resource]
            append_span(target, raw, z, len(graphs))
            graphs += z
            touched.append(g.resource)
            changes[g.name] = dict(replaced)
        check(touched, f"no eligible decode programs on rank {rank}")
        gid = sha256(graphs).digest()
        rr.payloads.add(id=gid, nbytes=len(graphs))
        for i in touched:
            rr.resources[i].stored.payload_id = gid
        label = "rank-" + str(rank) + "-decode-partials.zst"
        (a.out / label).write_bytes(graphs)
        files.append(
            dict(
                id=gid.hex(),
                filename=label,
                relative=f"deployments/{rr.deployment_id}/rank-{rank}/decode-peer-records.safetensors",
            )
        )
        (a.out / f"rank-{rank}.pb").write_bytes(rr.SerializeToString(deterministic=True))
        report.append(changes)
        print(rank, changes, flush=True)
    (a.out / "resources.json").write_text(
        json.dumps(dict(source_manifest=sha256(signed).hexdigest(), files=files), indent=2) + "\n"
    )
    (a.out / "changes.json").write_text(json.dumps(report, indent=2) + "\n")


p = ArgumentParser()
p.add_argument("--only-gathers", action="store_true")
for k in ("base", "image", "control", "gathers", "out"):
    p.add_argument(k, type=Path)
run(p.parse_args())
