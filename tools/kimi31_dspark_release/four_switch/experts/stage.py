"""Identify the signed four-row expert and encode its native graph records."""

import hashlib
import re
import struct
import subprocess


BASE_PRIMARY = "42322b48aa8be819e97c35b50974b50bf5b16b0b22e283410519666857bcf40f"
EXPERT_IMAGE = "cba427c54e5f054736a4541e52ba356086af7351aa8432d8050d259e36c59351"
EXPERT_TEXT = "369e6ac6dae69c2999e58d77ca2cc1c8aa15f1431e35f72e9bd6117f36b69aa5"


def digest(value):
    return hashlib.sha256(value).digest()


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def kernel_parameter_sizes(image):
    dump = subprocess.check_output(["/usr/local/cuda-13.2/bin/cuobjdump", "-elf", str(image)], text=True)
    fields = re.findall(
        r"Ordinal\s*:\s*0x([0-9a-f]+)\s+Offset\s*:\s*0x[0-9a-f]+\s+Size\s*:\s*0x([0-9a-f]+)", dump
    )
    widths = {int(index, 16): int(width, 16) for index, width in fields}
    require(len(widths) == 30 and len(fields) == 30, "expected one 30-argument expert entry point")
    return [widths[i] for i in range(len(widths))]


def raw_graph(nodes, edges):
    data = bytearray()
    for record in [*nodes, *edges]:
        encoded = record.SerializeToString(deterministic=True)
        data.extend(struct.pack("<I", len(encoded)))
        data.extend(encoded)
    return data
