# SPDX-License-Identifier: Apache-2.0
"""Materialize exactly the signed parent companions from a release snapshot."""
import argparse
import errno
import hashlib
import os
from pathlib import Path
import shutil

import package_pb2 as wire
from tools.kimi31_collective_graph_audit import tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("primary", type=Path)
    parser.add_argument("primary_sha256")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    with args.primary.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != args.primary_sha256:
            raise ValueError("input primary differs from the pinned checkpoint")
    locations = wire.LocationSet.FromString(tensor(args.primary, "flare.locations"))
    files = {entry.file for entry in locations.locations if entry.file}
    args.out.mkdir()
    for name in sorted(files | {"model.safetensors"}):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ValueError(f"invalid signed companion path {name!r}")
        src = args.primary if name == "model.safetensors" else args.snapshot / relative
        dst = args.out / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = src.resolve(strict=True)
        try:
            os.link(src, dst)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copyfile(src, dst)
    print(f"Prepared the signed parent with {len(files)} companions", flush=True)


if __name__ == "__main__":
    main()
