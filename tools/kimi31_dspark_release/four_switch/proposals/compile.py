"""Build the SM120 proposal normalization for each packed verification capacity."""

import json
from pathlib import Path
import sys

import torch
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, compile, make_backend
from triton.runtime.jit import MockTensor, create_function_from_signature
from ignition.components.markov_sampling import _normalize
from elf_text import cubin_code
from stage_pcie import resource_bytes
from tools.kimi31_collective_graph_audit import tensor
from tools.kimi31_package_collective_geometry import read_graph
import package_pb2 as wire
import runtime_package_pb2 as runtime

base, out = map(Path, sys.argv[1:])
out.mkdir()
primary = base / "model.safetensors"
manifest = wire.ProtectedPackage.FromString(tensor(primary, "flare.manifest"))
locations = {
    v.payload_id: v for v in wire.LocationSet.FromString(tensor(primary, "flare.locations")).locations
}
rank = wire.RankBody.FromString(manifest.ranks[0].body)
graph = next(g for g in rank.graphs if g.name == "speculative1x8-tokens4")
nodes, _ = read_graph(base, primary, locations, rank.resources[graph.resource], graph, runtime)
(epilogue,) = (
    n.kernel
    for n in nodes
    if n.HasField("kernel")
    and rank.functions[n.kernel.function].symbol == "_normalize"
    and list(n.kernel.grid) == [640, 1, 7]
)
fn = rank.functions[epilogue.function]
module = next(m for m in rank.modules if m.name == fn.module)
oracle = resource_bytes(base, primary, locations, rank.resources[module.cubin])
expected = cubin_code(oracle)
(out / "original.cubin").write_bytes(oracle)
target = GPUTarget("cuda", 120, 32)
backend = make_backend(target)
binder = create_function_from_signature(_normalize.signature, _normalize.params, backend)
types = (
    torch.float32,
    torch.float32,
    torch.float32,
    torch.bool,
    torch.float32,
    torch.int64,
    torch.int64,
    torch.bool,
    torch.int32,
    torch.int64,
    torch.int32,
    torch.int64,
    torch.int32,
)


def build(active, capacity):
    kwargs = dict(
        V=163840,
        DEPTH=7,
        ACTIVE_DEPTH=active,
        BLOCKS=640,
        PAD=1024,
        CAPACITY=capacity,
        BATCH=1,
        RUNNING=1,
        BLOCK=256,
        num_warps=4,
        enable_fp_fusion=False,
        debug=False,
        instrumentation_mode="",
    )
    bound, spec, opts = binder(*(MockTensor(t) for t in types), **kwargs)
    opts, sig, consts, attrs = _normalize._pack_args(backend, kwargs, bound, spec, opts)
    result = compile(ASTSource(_normalize, sig, consts, attrs), target=target, options=opts.__dict__)
    return result.asm["cubin"], result.metadata.shared


# Match the complete calibrated normalizer before shortening its proposal prefix.
for capacity in (1, 4, 8, 16, 32):
    image, shared = build(7, capacity)
    observed = cubin_code(image)
    print("full-depth machine code", capacity, observed, expected, flush=True)
    if observed == expected and shared == epilogue.shared:
        break
else:
    raise ValueError("Offline normalization differs from the signed FP32 probability calculation")
info = {}
for depth in range(1, 7):
    image, shared = build(depth, capacity)
    (out / f"prefix-{depth}.cubin").write_bytes(image)
    info[depth] = {"shared": shared, "symbol": cubin_code(image)[0]}
(out / "programs.json").write_text(json.dumps(info, indent=2) + "\n")
