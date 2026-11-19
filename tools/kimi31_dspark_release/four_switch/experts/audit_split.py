"""Read both signed packages and contract the FC1, SiTU, FC2 triplets back to the complete source graph."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from stage import require
from stage_pcie import resource_bytes
from stage_split import load_core, P4, PHASES
from tools.kimi31_package_collective_geometry import read_graph
import runtime_package_pb2 as rt


def encoded(value):
    return value.SerializeToString(deterministic=True)


def audit(before, after):
    cores, _, originals = load_core(before)
    result, _, derived = load_core(after)
    original_files = {
        p.relative_to(before) for p in before.rglob("*") if p.is_file() and p.name != "model.safetensors"
    }
    result_files = {
        p.relative_to(after) for p in after.rglob("*") if p.is_file() and p.name != "model.safetensors"
    }
    require(
        original_files <= result_files and all((before / p).samefile(after / p) for p in original_files),
        "original weight or deployment companion was copied or changed",
    )
    require(
        len(result_files - original_files) == 17,
        "expected one shared stage image bundle and 16 graph bundles",
    )
    counts = Counter()
    for a, b in zip(originals, derived, strict=True):
        restored = type(b)()
        restored.CopyFrom(b)
        for name, fixed in (("modules", 3), ("functions", 3), ("resources", 3), ("payloads", 2)):
            x, y = getattr(a, name), getattr(restored, name)
            require(len(y) == len(x) + fixed, f"unexpected {name} suffix")
            del y[len(x) :]
        (ga,) = [v for v in a.graphs if v.name == P4]
        (gb,) = [v for v in b.graphs if v.name == P4]
        (restored_graph,) = [v for v in restored.graphs if v.name == P4]
        restored_graph.CopyFrom(ga)
        restored.resources[ga.resource].CopyFrom(a.resources[ga.resource])
        require(restored == a, "rank contract, prefill catalog, storage or other graph changed")
        encoded_original, original_edges = read_graph(*cores, a.resources[ga.resource], ga, rt)
        encoded_derived, derived_edges = read_graph(*result, b.resources[gb.resource], gb, rt)
        phases = list(PHASES)
        original_fns = len(a.functions)
        for i, phase in enumerate(phases):
            fn = b.functions[original_fns + i]
            (mod,) = [m for m in b.modules if m.name == fn.module]
            image = resource_bytes(*result, b.resources[mod.cubin])
            require(
                hashlib.sha256(image).hexdigest() == PHASES[phase],
                f"incorrect signed {phase} image at rank {b.rank}",
            )

        first, last, triplet_edges = [], [], []
        idx = 0
        for source in encoded_original:
            first.append(idx)
            if (
                source.HasField("kernel")
                and "W4A16FusedMoeKernel" in a.functions[source.kernel.function].symbol
            ):
                for offset in range(3):
                    found = rt.Node()
                    found.CopyFrom(encoded_derived[idx + offset])
                    require(
                        found.kernel.function == original_fns + offset
                        and list(source.kernel.arguments) == list(found.kernel.arguments),
                        "stage function or any signed argument changed",
                    )
                    found.kernel.function = source.kernel.function
                    if offset == 1:
                        require(
                            list(found.kernel.grid) == [12, 1, 1]
                            and found.kernel.shared == 0
                            and not found.kernel.attributes,
                            "SiTU launch must have twelve ordinary CTAs and zero dynamic shared memory",
                        )
                        found.kernel.grid[:] = source.kernel.grid
                        found.kernel.shared = source.kernel.shared
                        found.kernel.attributes.extend(source.kernel.attributes)
                    require(found == source, "decoded kernel has an unexpected launch field")
                triplet_edges.extend([(idx, idx + 1, bytes(8)), (idx + 1, idx + 2, bytes(8))])
                idx += 3
                counts["expert_spans"] += 1
            else:
                require(source == encoded_derived[idx], "unrelated CUDA graph node changed")
                idx += 1
                counts["unchanged_p4_nodes"] += 1
            last.append(idx - 1)
        require(idx == len(encoded_derived), "trailing nodes after the last source kernel")
        outside = [(last[e.source], first[e.destination], e.data) for e in original_edges]
        actual = [(e.source, e.destination, e.data) for e in derived_edges]
        require(
            Counter(outside + triplet_edges) == Counter(actual),
            "external edge, dependency payload or stage order changed",
        )
        require(
            len(encoded_derived) == gb.node_count == ga.node_count + 184
            and len(derived_edges) == gb.edge_count == ga.edge_count + 184,
            "signed graph count disagrees with decoded records",
        )
        require(
            sorted({n.kernel.function for n in encoded_derived if n.HasField("kernel")})
            == list(gb.functions),
            "graph function catalog differs from decoded calls",
        )
        counts["original_p4_edges_preserved"] += len(outside)
        counts["new_ordering_edges"] += len(triplet_edges)
        counts["unchanged_other_graphs"] += len(a.graphs) - 1
    return dict(
        original=str(before),
        candidate=str(after),
        original_companions_hardlinked=len(original_files),
        added_companions=len(result_files - original_files),
        **dict(counts),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "candidate"):
        parser.add_argument(key, type=Path)
    a = parser.parse_args()
    print(json.dumps(audit(a.source, a.candidate), indent=2))
