"""Preserve the DWARF source entries in the signed CUDA images.

PTXAS stats paths named by a bare PTX .file directive. The signed images' DWARF
line tables carry the timestamp and size below. PTX supports those attributes
explicitly, so a clean build can assemble the same image without that filesystem.
"""

import re

ROOT = "/data/kimi31-dspark-final-mxfp4-20261115/flare/ignition/recipes/kimi3"
TRITON = "/data/kimi31-support/flare-main-k31-performance/.venv/lib/python3.12/site-packages/triton/language"
SOURCE_METADATA = {
    f"{ROOT}/pointwise.py": (1794701935, 4946),
    f"{ROOT}/residual.py": (1794701935, 6602),
    f"{TRITON}/standard.py": (1794128686, 17257),
}


def pin_debug_sources(ptx):
    present = set()

    def render(match):
        path = match.group(2)
        if path not in SOURCE_METADATA:
            raise ValueError(f"unrecognized PTX source file {path}")
        present.add(path)
        seconds, size = SOURCE_METADATA[path]
        return f"{match.group(1)}, {seconds}, {size}"

    rewritten = re.sub(r'(?m)^([ \t]*\.file[ \t]+\d+[ \t]+"([^"\n]+)")[ \t]*$', render, ptx)
    if not present:
        raise ValueError("canonical PTX lacks a recorded source location")
    return rewritten
