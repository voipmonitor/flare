"""Verify the 186 original reduction/mixture bindings and replace only the shared hidden-row code."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import package_pb2 as wire
import runtime_package_pb2 as rt
from stage_pcie import append_span, frame, resource_bytes
from tools.kimi31_collective_graph_audit import tensor
from tools.kimi31_native_row_resources import pack_config, topology
from tools.kimi31_package_collective_geometry import read_graph

PARENT = "fb9a083d29be4451a16febd1eb5f473b1543a9d603a29812a03a08f7b2eb20be"
CAPTURE = "91887e7a69bee716c2c469d03168731e2038f0159171d5dd0d99be0c4b415176"
COMPANION = "deployments/gb202-tp16/row-kernels.safetensors"


def require(cond, detail):
    if not cond:
        raise ValueError(detail)


def digest(data):
    return hashlib.sha256(data).digest()


def open_core(root, expected):
    primary = root / "model.safetensors"
    with primary.open("rb") as f:
        require(hashlib.file_digest(f, "sha256").hexdigest() == expected, f"unexpected primary {primary}")
    raw = tensor(primary, "flare.manifest")
    m = wire.ProtectedPackage.FromString(raw)
    loc = {v.payload_id: v for v in wire.LocationSet.FromString(tensor(primary, "flare.locations")).locations}
    ranks = [wire.RankBody.FromString(x.body) for x in m.ranks]
    require([r.rank for r in ranks] == list(range(16)), "expected 16 ordered ranks")
    return (root, primary, loc), raw, ranks


def p4(core, r):
    (graph,) = (g for g in r.graphs if g.name == "speculative1x8-tokens4")
    nodes, edges = read_graph(*core, r.resources[graph.resource], graph, rt)
    parents, children, order = topology(nodes, edges)
    return nodes, parents, children, order


def argument_int(k, i):
    a = k.arguments[i]
    require(not a.relocations and len(a.data) in (4, 8), "unexpected scalar capture")
    return int.from_bytes(a.data, "little")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("capture", "parent", "build", "output"):
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    ref, _, capture = open_core(a.capture, CAPTURE)
    core, manifest, ranks = open_core(a.parent, PARENT)
    control = json.loads((a.build / "control/programs.json").read_text())
    fixed = json.loads((a.build / "corrected/programs.json").read_text())
    require(
        control["operations"] == fixed["operations"] and len(control["operations"]) == 186,
        "canonical residual variants changed",
    )
    (old_payload,) = (v.payload_id for v in core[2].values() if v.file == COMPANION)
    a.output.mkdir()
    evidence = []
    common = None
    for r0, r in zip(capture, ranks, strict=True):
        nodes, parents, children, order = p4(ref, r0)
        targets, _, _, torder = p4(core, r)
        mixed = [
            i
            for i in order
            if nodes[i].HasField("kernel") and r0.functions[nodes[i].kernel.function].symbol == "_mix"
        ]
        hidden = [
            i
            for i in torder
            if targets[i].HasField("kernel")
            and r.functions[targets[i].kernel.function].symbol == "reduce_row_direct"
        ]
        require(len(mixed) == 187 and len(hidden) == 186, f"rank {r.rank} hidden mixture count differs")
        mods = {v.name: v for v in r.modules}
        changes = {}
        counts = Counter()
        bindings = []
        for site, (mi, ti, op) in enumerate(zip(mixed[1:], hidden, control["operations"], strict=True), 1):
            ((ri, edge),) = parents[mi]
            red, mix, new = (nodes[ri].kernel, nodes[mi].kernel, targets[ti].kernel)
            require(
                op["ordinal"] == site
                and children[ri] == [(mi, bytes(8))]
                and edge == bytes(8)
                and "IslandRS" in r0.functions[red.function].symbol
                and list(red.grid) == [64, 1, 1]
                and list(red.block) == [1024, 1, 1]
                and (argument_int(red, 22), argument_int(red, 23)) == (28672, 4)
                and red.arguments[17] == mix.arguments[1],
                f"rank {r.rank} site {site} original reduction differs",
            )
            require(
                list(new.arguments[:-1]) == list(mix.arguments)
                and new.arguments[-1] == pack_config(red, r.rank, rt)
                and list(new.grid) == [64, 1, 1]
                and list(new.block) == list(mix.block) == [32 * op["warps"], 1, 1]
                and new.shared == mix.shared,
                f"rank {r.rank} site {site} source/parent bindings differ",
            )
            fn = r.functions[new.function]
            oldfn = r0.functions[mix.function]
            require(
                list(fn.parameter_sizes) == [*oldfn.parameter_sizes, 120],
                "hidden kernel parameter layout differs",
            )
            index = mods[fn.module].cubin
            res = r.resources[index]
            old = control["programs"][op["ptx"]]["direct"]
            fresh = fixed["programs"][op["ptx"]]["direct"]
            require(
                fn.module == "row-reduction-" + old["sha256"]
                and res.kind == wire.EXECUTABLE_KIND_CUBIN
                and res.stored.payload_id == old_payload,
                f"rank {r.rank} site {site} module differs",
            )
            counts[index] += 1
            if index not in changes:
                image = resource_bytes(*core, res)
                require(
                    image == (a.build / "control" / old["file"]).read_bytes()
                    and digest(image).hex() == old["sha256"],
                    "cold control image differs from signed cubin",
                )
                replacement = (a.build / "corrected" / fresh["file"]).read_bytes()
                require(
                    digest(replacement).hex() == fresh["sha256"]
                    and replacement != image
                    and old["symbol"] == fresh["symbol"] == fn.symbol,
                    "compiled replacement/entry point differs",
                )
                changes[index] = (res.stored.offset, old["sha256"], fresh["sha256"], replacement)
            else:
                require(
                    changes[index][1:3] == (old["sha256"], fresh["sha256"]),
                    "one source module cannot name two residual specializations",
                )
            bindings.append((site, ri, mi, ti, index))
        require(
            len(changes) == 25
            and {i for i, v in enumerate(r.resources) if v.stored.payload_id == old_payload} == set(changes),
            "shared payload has unexpected consumers",
        )
        fids = {i for i, v in enumerate(r.functions) if mods[v.module].cubin in changes}
        require(
            all(v.symbol == "reduce_row_direct" for i, v in enumerate(r.functions) if i in fids),
            "shared image has another entry point",
        )
        for g in r.graphs:
            if g.name != "speculative1x8-tokens4":
                require(not set(g.functions) & fids, f"hidden images also used by graph {g.name}")
        packed = bytearray()
        updates = []
        for i, (offset, old, fresh, img) in sorted(changes.items(), key=lambda v: v[1][0]):
            data = frame(img)
            res = r.resources[i]
            require(res.chunk_nbytes == 4 << 20, "unexpected executable hash chunk geometry")
            append_span(res, img, data, len(packed))
            packed += data
            updates.append((i, old, fresh, counts[i]))
        if common is None:
            common = bytes(packed)
            (a.output / "hidden-row-kernels.zst").write_bytes(common)
        else:
            require(bytes(packed) == common, "TP16 hidden cubins must remain one shared payload")
        new_payload = digest(packed)
        for i in changes:
            r.resources[i].stored.payload_id = new_payload
        (catalog,) = (v for v in r.payloads if v.id == old_payload)
        catalog.id = new_payload
        catalog.nbytes = len(packed)
        (a.output / f"rank-{r.rank}.pb").write_bytes(r.SerializeToString(deterministic=True))
        evidence.append(
            dict(rank=r.rank, original_and_fused_sites=bindings, old_and_corrected_images=updates)
        )
        print(r.rank, len(hidden), "bindings and", len(changes), "full original cubins verified", flush=True)
    (a.output / "resources.json").write_text(
        json.dumps(
            dict(
                source_manifest=digest(manifest).hex(),
                old_payload=old_payload.hex(),
                files=[dict(id=digest(common).hex(), filename="hidden-row-kernels.zst", relative=COMPANION)],
            ),
            indent=2,
        )
        + "\n"
    )
    (a.output / "binding-audit.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
