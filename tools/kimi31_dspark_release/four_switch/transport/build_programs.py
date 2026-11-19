"""Compile the full-vocabulary exchange and routed expert selector against signed TP16 controls."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys
from stage_pcie import resource_bytes
from tools.kimi31_collective_graph_audit import tensor
from tools.kimi31_package_collective_geometry import read_graph
import package_pb2 as wire
import runtime_package_pb2 as rt

root = Path(__file__).resolve().parent
base, provider, build = map(Path, sys.argv[1:])
signed = build / "signed"
signed.mkdir()
router = build / "router"
router.mkdir()
primary = base / "model.safetensors"
m = wire.ProtectedPackage.FromString(tensor(primary, "flare.manifest"))
loc = {v.payload_id: v for v in wire.LocationSet.FromString(tensor(primary, "flare.locations")).locations}
for signature in m.ranks:
    r = wire.RankBody.FromString(signature.body)
    mods = {x.name: x for x in r.modules}
    (g,) = (v for v in r.graphs if v.name == "speculative1x8-tokens4")
    nodes, edges = read_graph(base, primary, loc, r.resources[g.resource], g, rt)
    fixture = router / f"rank-{r.rank}"
    fixture.mkdir()
    old = {}
    for n in nodes:
        if not n.HasField("kernel"):
            continue
        k = n.kernel
        fn = r.functions[k.function]
        if (
            "RecursiveGather" in fn.symbol
            and len(k.arguments) == 39
            and int.from_bytes(k.arguments[36].data, "little") == 0x294000
            and int.from_bytes(k.arguments[35].data, "little") == 3
        ):
            name = "vocabulary"
        elif "RowBlockAllGatherPair" in fn.symbol:
            name = "base"
        elif "WarpTopK16" in fn.symbol:
            name = "select"
        else:
            continue
        if name in old:
            assert old[name][0] == k.function
            continue
        img = resource_bytes(base, primary, loc, r.resources[mods[fn.module].cubin])
        old[name] = (k.function, k.grid[0], k.block[0], k.shared, fn.symbol)
        if name == "vocabulary":
            (signed / f"gather-{r.rank}-1280.cubin").write_bytes(img)
        else:
            (fixture / f"{name}.cubin").write_bytes(img)
    assert set(old) == {"vocabulary", "base", "select"}
    lines = [
        f"4 {name} {name}.cubin {old[name][4]} {old[name][1]} {old[name][2]} {old[name][3]}"
        for name in ("base", "select")
    ]
    (fixture / "programs.txt").write_text("\n".join(lines) + "\n")


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


run(root / "compose_router.py", provider, build / "compiled")


def compile_rank(rank):
    run(root / "build_gather.py", signed, build / "checked", rank, 1280, "recursive", "--control")
    run(root / "build_gather.py", signed, build / "cross-first", rank, 1280, "recursive")
    run(root / "compile_router.py", build / "compiled", rank)
    source = build / "compiled" / f"rank-{rank}"
    fixture = router / f"rank-{rank}"
    shutil.copyfile(source / "kernel.cubin", fixture / "fused.cubin")
    sym = (source / "symbol.txt").read_text().strip()
    with (fixture / "programs.txt").open("a") as f:
        f.write(f"4 fused fused.cubin {sym} 4 512 4608\n")


with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(compile_rank, range(16)))
