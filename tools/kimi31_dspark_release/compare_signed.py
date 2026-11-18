"""Compare non-prefill graphs, allocator geometry and prepared weights on all ranks."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import package_pb2 as wire
from tools.kimi31_collective_graph_audit import tensor

p = ArgumentParser(description=__doc__)
for arg in ("control", "candidate", "output"):
    p.add_argument(arg, type=Path)
a = p.parse_args()


def read(d):
    return wire.ProtectedPackage.FromString(tensor(d / "model.safetensors", "flare.manifest"))


ctrl, cand = read(a.control), read(a.candidate)
rows = []
for a0, b0 in zip(ctrl.ranks, cand.ranks, strict=True):
    old = wire.RankBody.FromString(a0.body)
    new = wire.RankBody.FromString(b0.body)
    layout = (
        "allocations",
        "reservations",
        "peers",
        "execution",
        "initializers",
        "prepared_weights",
        "compressed_prepared_weights",
    )
    if any(getattr(old, k) != getattr(new, k) for k in layout):
        raise ValueError("storage or execution bindings changed")
    if list(new.modules[: len(old.modules)]) != list(old.modules) or list(
        new.functions[: len(old.functions)]
    ) != list(old.functions):
        raise ValueError("an existing executable definition changed")
    kept = []
    altered = []
    for left, right in zip(old.graphs, new.graphs, strict=True):
        if left.name != right.name:
            raise ValueError("changed graph catalog")
        if left == right and old.resources[left.resource] == new.resources[right.resource]:
            kept.append(left.name)
        else:
            altered.append(left.name)
    expected = {f"prefill{n}{suffix}" for n in (512, 1024, 1536, 2048) for suffix in ("", "-no-mtp")}
    if set(altered) != expected:
        raise ValueError(f"unexpected changed graphs {altered}")
    rows.append(dict(rank=old.rank, unchanged_graphs=kept, changed_graphs=altered))
a.output.write_text(
    json.dumps(
        dict(
            ranks=rows,
            allocations_identical=True,
            prepared_weights_identical=True,
            primary_sha256={
                k: sha256((v / "model.safetensors").read_bytes()).hexdigest()
                for k, v in (("control", a.control), ("candidate", a.candidate))
            },
        ),
        indent=2,
    )
    + "\n"
)
print(
    len(rows),
    "ranks;",
    len(rows[0]["unchanged_graphs"]),
    "unchanged graphs per rank;",
    len(rows[0]["changed_graphs"]),
    "changed prefill graphs per rank",
)
