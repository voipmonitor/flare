"""Extract the checked sparse-candidate ring launch for a standalone TP16 transfer test."""

import argparse
import struct
from pathlib import Path
import package_pb2 as p
import runtime_package_pb2 as rt
from stage_pcie import resource_bytes
from tools.kimi31_collective_graph_audit import tensor
from tools.kimi31_package_collective_geometry import read_graph

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("package", type=Path)
parser.add_argument("out", type=Path)
a = parser.parse_args()
a.out.mkdir()
f = a.package / "model.safetensors"
m = p.ProtectedPackage.FromString(tensor(f, "flare.manifest"))
loc = {x.payload_id: x for x in p.LocationSet.FromString(tensor(f, "flare.locations")).locations}
for rank in m.ranks:
    r = p.RankBody.FromString(rank.body)
    (g,) = (g for g in r.graphs if g.name == "prefill2048")
    nodes, _ = read_graph(a.package, f, loc, r.resources[g.resource], g, rt)
    kernels = [n.kernel for n in nodes if n.HasField("kernel")]
    k = next(
        k
        for k in kernels
        if "RingGather" in r.functions[k.function].symbol
        and len(k.arguments) == 39
        and k.arguments[35].data == struct.pack("<I", 128)
        and int.from_bytes(k.arguments[36].data, "little") == 0x2002000
    )
    func = r.functions[k.function]
    (mod,) = (x for x in r.modules if x.name == func.module)
    (a.out / f"rank-{r.rank}.cubin").write_bytes(resource_bytes(a.package, f, loc, r.resources[mod.cubin]))
    (a.out / f"rank-{r.rank}.txt").write_text(f"{func.symbol}\n{k.grid[0]} {k.block[0]} {k.shared}\n")
