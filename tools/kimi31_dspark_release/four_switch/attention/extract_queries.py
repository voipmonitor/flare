"""Extract the Kimi latent and six-head quantized-query gathers from signed rank resources."""

from pathlib import Path
import sys
import package_pb2 as w
import runtime_package_pb2 as rt
from tools.kimi31_collective_graph_audit import tensor
from tools.kimi31_package_collective_geometry import read_graph
from stage_pcie import resource_bytes

d, out = map(Path, sys.argv[1:])
out.mkdir()
f = d / "model.safetensors"
m = w.ProtectedPackage.FromString(tensor(f, "flare.manifest"))
loc = {x.payload_id: x for x in w.LocationSet.FromString(tensor(f, "flare.locations")).locations}
for rank in m.ranks:
    r = w.RankBody.FromString(rank.body)
    (g,) = (v for v in r.graphs if v.name == "speculative1x8-tokens4")
    nodes, _ = read_graph(d, f, loc, r.resources[g.resource], g, rt)
    for vectors, stride in ((17, 0x8A00), (216, 0xD8C00)):
        matched = []
        for n in nodes:
            if not n.HasField("kernel"):
                continue
            k = n.kernel
            fn = r.functions[k.function]
            if (
                "RecursiveGather" in fn.symbol
                and len(k.arguments) == 39
                and int.from_bytes(k.arguments[36].data, "little") == stride
            ):
                assert (
                    int.from_bytes(k.arguments[35].data, "little") == 4
                    and list(k.grid) == [8, 1, 1]
                    and list(k.block) == [128, 1, 1]
                )
                matched.append(fn.module)
        assert len(matched) == 24 and len(set(matched)) == 1
        name = matched[0]
        fn = next(v for v in r.functions if v.module == name and "RecursiveGather" in v.symbol)
        mod = next(v for v in r.modules if v.name == name)
        image = resource_bytes(d, f, loc, r.resources[mod.cubin])
        path = out / f"gather-{r.rank}-{vectors}"
        path.with_suffix(".cubin").write_bytes(image)
        path.with_suffix(".txt").write_text(fn.symbol + "\n")
