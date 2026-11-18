# SPDX-License-Identifier: Apache-2.0
"""Supply the captured DWARF source attributes to PTXAS in PTX .file directives."""
import re


_FILE = re.compile(r'(?m)^([ \t]*\.file[ \t]+[0-9]+[ \t]+"([^"\n]+)")[ \t]*$')


def annotate(ptx, files):
    """Keep the filename and index of each source and supply its timestamp and byte length."""
    def rewrite(match):
        path = match.group(2)
        if path not in files:
            raise ValueError(f"missing recorded DWARF attributes for PTX source {path!r}")
        timestamp, size = files[path]
        if any(type(value) is not int or not 0 <= value < 1 << 64 for value in (timestamp, size)):
            raise ValueError(f"invalid unsigned PTX source attributes for {path!r}")
        return f"{match.group(1)}, {timestamp}, {size}"

    value, count = _FILE.subn(rewrite, ptx)
    if not count:
        raise ValueError("expected bare .file directives in canonical residual PTX")
    return value
