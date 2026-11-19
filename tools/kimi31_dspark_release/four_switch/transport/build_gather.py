"""Compile the signed PCIe exchange and a four-switch round order without a CUDA context."""

import argparse
from pathlib import Path
from unittest.mock import patch
import multiprocessing
from b12x._lib.compile_pool import _initialize_worker

_initialize_worker(
    0,
    (12, 0),
    "offline-sm120",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    188,
    101376,
    102400,
    multiprocessing.Array("i", [0, 0, 0]),
)
from cuda.bindings.driver import CUstream
import cutlass.cute as cute
from b12x.comm.pcie import _runtime_gather_cute as gather
from elf_text import cubin_code

p = argparse.ArgumentParser()
p.add_argument("signed", type=Path)
p.add_argument("out", type=Path)
p.add_argument("rank", type=int)
p.add_argument("packs", type=int)
p.add_argument("transport", choices=("recursive", "recursive_ready"))
p.add_argument("--control", action="store_true")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
files = a.out / f"compile-{a.rank}-{a.packs}"
files.mkdir()


def emit(op, *args, **kwargs):
    return cute.compile(
        op, *args, options=f"--keep-cubin --gpu-arch=sm_120a --dump-dir={files.resolve()}", no_jit_engine=True
    )


with (
    patch.object(gather, "compile_kernel", emit),
    patch.object(gather, "attach_programs", lambda fn, *deps: fn),
    patch.object(gather, "current_cuda_stream", lambda: CUstream(0)),
):
    gather.compile_selected_gather(
        16, a.rank, 4 * a.packs, 128, 8, 0, a.transport, cross_switch_first=not a.control
    )
(image,) = files.rglob("*.cubin")
data = image.read_bytes()
symbol, code = cubin_code(data)
stem = f"gather-{a.rank}-{a.packs}"
original = cubin_code((a.signed / (stem + ".cubin")).read_bytes())[1]
print(
    a.rank,
    a.packs,
    a.transport,
    "baseline" if a.control else "cross-first",
    code,
    "signed",
    original,
    flush=True,
)
if a.control and code != original:
    raise ValueError("The control image differs from the captured executable code")
(a.out / (stem + ".cubin")).write_bytes(data)
(a.out / (stem + ".txt")).write_text(symbol + "\n")
