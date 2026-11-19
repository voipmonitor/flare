"""Compile the canonical 3,584-column RMSNorm for the captured P4 MoE input."""

import json, sys
from pathlib import Path
import torch
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, compile, make_backend
from triton.runtime.jit import MockTensor, create_function_from_signature
from ignition.recipes.kimi3.pointwise import _rms_norm
from tools.kimi31_collective_graph_audit import tensor
from tools.kimi31_package_collective_geometry import read_graph
from stage_pcie import resource_bytes
from elf_text import cubin_code
import package_pb2 as w, runtime_package_pb2 as rt

out, base = map(Path, sys.argv[1:])
primary = base / "model.safetensors"
m = w.ProtectedPackage.FromString(tensor(primary, "flare.manifest"))
loc = {x.payload_id: x for x in w.LocationSet.FromString(tensor(primary, "flare.locations")).locations}
target = GPUTarget("cuda", 120, 32)
backend = make_backend(target)
binder = create_function_from_signature(_rms_norm.signature, _rms_norm.params, backend)
rank = w.RankBody.FromString(m.ranks[0].body)
g = next(g for g in rank.graphs if g.name == "speculative1x8-tokens4")
nodes, edges = read_graph(base, primary, loc, rank.resources[g.resource], g, rt)
p = {e.destination: e.source for e in edges}
norms = [
    n.kernel
    for i, n in enumerate(nodes)
    if n.HasField("kernel")
    and rank.functions[n.kernel.function].symbol == "_rms_norm"
    and i in p
    and nodes[p[i]].HasField("kernel")
    and "IslandRS" in rank.functions[nodes[p[i]].kernel.function].symbol
]
assert len(norms) == 92 and len({n.function for n in norms}) == 1
func = rank.functions[norms[0].function]
mod = next(x for x in rank.modules if x.name == func.module)
image = resource_bytes(base, primary, loc, rank.resources[mod.cubin])
(out / "reference-norm.cubin").write_bytes(image)
name, want = cubin_code(image)
assert name == "_rms_norm"
bf = MockTensor(torch.bfloat16)
print("target", want, [len(x.data) for x in norms[0].arguments], norms[0].block[0], flush=True)
for dt in (torch.int64, torch.int32):
    args = (bf, bf, bf, MockTensor(dt), 3584, 224, 1e-6, 3584, 224, True, 4096)
    kwargs = dict(num_warps=8, debug=False, instrumentation_mode="")
    bound, spec, opts = binder(*args, **kwargs)
    opts, sig, constants, attrs = _rms_norm._pack_args(backend, kwargs, bound, spec, opts)
    program = compile(ASTSource(_rms_norm, sig, constants, attrs), target=target, options=opts.__dict__)
    text_digest = cubin_code(program.asm["cubin"])[1]
    print(dt, text_digest, flush=True)
    if text_digest == want:
        (out / "canonical.ptx").write_text(program.asm["ptx"])
        (out / "canonical.json").write_text(
            json.dumps(
                dict(
                    symbol=name,
                    code_sha256=want,
                    dtype=str(dt),
                    shared=norms[0].shared,
                    grid=list(norms[0].grid),
                    block=list(norms[0].block),
                ),
                indent=2,
            )
            + "\n"
        )
        break
else:
    raise ValueError("norm code does not match the signed release")
# Save the signed rank-specific reductions that feed the reference norm.
for signed in m.ranks:
    r = w.RankBody.FromString(signed.body)
    g = next(g for g in r.graphs if g.name == "speculative1x8-tokens4")
    ns, edges = read_graph(base, primary, loc, r.resources[g.resource], g, rt)
    matches = [
        n.kernel
        for n in ns
        if n.HasField("kernel")
        and "IslandRS" in r.functions[n.kernel.function].symbol
        and int.from_bytes(n.kernel.arguments[22].data, "little") == 14336
    ]
    assert len(matches) == 92 and len({k.function for k in matches}) == 1
    k = matches[0]
    f = r.functions[k.function]
    mo = next(v for v in r.modules if v.name == f.module)
    (out / f"rank-{r.rank}.cubin").write_bytes(resource_bytes(base, primary, loc, r.resources[mo.cubin]))
    vals = [int.from_bytes(v.data, "little") for v in k.arguments[18:]]
    (out / f"rank-{r.rank}.txt").write_text(
        f.symbol + "\n" + " ".join(str(q) for q in [k.grid[0], k.block[0], k.shared, *vals]) + "\n"
    )
