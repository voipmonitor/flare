"""Compile a rank's row-preserving gather and selection for SM120."""

from pathlib import Path
import sys
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
from cutlass import Int32
from b12x.comm.pcie._dcp_a2a_cute import _u32_ptr, _u8_ptr, _i32_ptr, _f32_ptr
from elf_text import cubin_code

out = Path(sys.argv[1]).resolve()
rank = int(sys.argv[2])
sys.path.insert(0, str(out))
from fused_router import GatherRouterTopK

folder = out / f"rank-{rank}"
folder.mkdir()
word = _u32_ptr(16, align=4)
kernel = GatherRouterTopK(16, rank, 512, True)
cute.compile(
    kernel,
    word,
    word,
    word,
    word,
    _f32_ptr(16),
    _f32_ptr(16),
    _i32_ptr(16),
    (_u8_ptr(16),) * 16,
    (word,) * 16,
    Int32(1),
    CUstream(0),
    options=f"--keep-cubin --keep-ptx --gpu-arch=sm_120a --dump-dir={folder}",
    no_jit_engine=True,
)
(image,) = folder.glob("*.cubin")
sym, sha = cubin_code(image.read_bytes())
(folder / "kernel.cubin.tmp").write_bytes(image.read_bytes())
(folder / "kernel.cubin.tmp").rename(folder / "kernel.cubin")
(folder / "symbol.txt").write_text(sym + "\n")
print(rank, sha)
