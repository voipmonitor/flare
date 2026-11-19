"""Compose the prepared four-row gather with the original Kimi selector's arithmetic."""

import argparse
from pathlib import Path
import textwrap

p = argparse.ArgumentParser()
p.add_argument("provider", type=Path)
p.add_argument("out", type=Path)
args = p.parse_args()
out = args.out.resolve()
out.mkdir(exist_ok=True)
comm = args.provider / "flashinfer/experimental/b12x/comm/pcie"
source = (comm / "_pair_rows_cute.py").read_text()
selector = (comm / "_kimi_topk_cute.py").read_text()
start = selector.index("        tidx, _, _ = cute.arch.thread_idx()")
end = selector.index("\n\n@program_cache", start)
body = selector[start:end]
before = body.index("        keys = cute.make_rmem_tensor")
sync = body.index("        cute.arch.sync_threads()")
# Only the lower 256 threads run the existing 8-warp sort. All 512 threads
# participate in its CTA-wide synchronization.
body = (
    body[:before]
    + "        if tidx < Int32(256):\n"
    + textwrap.indent(body[before:sync], "    ")
    + body[sync:]
)
helpers = """from b12x.comm.pcie._dcp_a2a_cute import _kimi_key_expert, _kimi_pack_key, _kimi_warp_max_key, _f32_ptr, _i32_ptr
from cutlass import Float32, Uint64

"""
src = source[: source.index("@program_cache")]
src = src.replace("from ._", "from b12x.comm.pcie._")
src = src.replace("class RowBlockAllGatherPair", helpers + "class GatherRouterTopK")
src = src.replace(
    "out_second: cute.Pointer, stages: tuple",
    "out_second: cute.Pointer, bias: cute.Pointer, weights: cute.Pointer, ids: cute.Pointer, stages: tuple",
)
src = src.replace(
    "self.kernel(first, second, out_first, out_second, stages, signals, stride)",
    "self.kernel(first, second, out_first, out_second, bias, weights, ids, stages, signals, stride)",
)
src = src.replace(
    "def kernel(self, first, second, out_first, out_second, staging, signals, stride):",
    "def kernel(self, first, second, out_first, out_second, bias, weights, ids, staging, signals, stride):",
)
src += "\n        self.select(cute.recast_ptr(out_second, dtype=Float32), bias, weights, ids)\n"
src += (
    "\n    @cute.jit\n    def select(self, router_logits, correction_bias, output_weights, output_ids):\n"
    + body
    + "\n"
)
(out / "fused_router.py").write_text(src)
