"""Read a single embedded CUDA ELF from an offline CuTe host object."""

from pathlib import Path
import hashlib, json, struct, sys
from elf_text import cubin_code

obj = Path(sys.argv[1])
dst = Path(sys.argv[2])
data = obj.read_bytes()
start = data.index(b"\x7fELF", 4)
b = data[start:]
if b[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", b, 18)[0] != 190:
    raise ValueError("not an embedded little-endian CUDA ELF64")
ph, sh, flags, eh_size, ph_size, ph_count, sh_size, sh_count, si = struct.unpack_from("<QQIHHHHHH", b, 32)
ends = [eh_size, ph + ph_size * ph_count, sh + sh_size * sh_count]
for i in range(ph_count):
    fields = struct.unpack_from("<IIQQQQQQ", b, ph + i * ph_size)
    ends.append(fields[2] + fields[5])
for i in range(sh_count):
    name, kind, _, _, off, size, *_ = struct.unpack_from("<IIQQQQIIQQ", b, sh + i * sh_size)
    if kind != 8:
        ends.append(off + size)
end = max(ends)
assert end <= len(b)
image = b[:end]
name, code = cubin_code(image)
dst.write_bytes(image)
print(
    json.dumps(
        dict(symbol=name, sha256=hashlib.sha256(image).hexdigest(), text_sha256=code, size=end), indent=2
    )
)
