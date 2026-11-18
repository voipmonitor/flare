"""Inspect row-gather, sparse-selection and TP16 reduction sites in the native verification graphs."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import struct
import sys

from tools.kimi31_collective_graph_audit import tensor, decode_frame


def read_graph(directory, primary, locations, resource, graph, runtime):
    stored = resource.stored
    location = locations[stored.payload_id]
    file = directory / location.file if location.file else primary
    if not file.resolve().is_relative_to(directory.resolve()):
        raise ValueError("graph resource path leaves the native package")
    data = tensor(file, location.tensor, stored.offset, stored.nbytes)
    extent = resource.chunk_nbytes
    if extent <= 0 or [sha256(data[i:i + extent]).digest() for i in range(0, len(data), extent)] != list(resource.chunk_sha256):
        raise ValueError("graph resource does not match its declared chunks")
    raw = decode_frame(data, resource.decoded_nbytes)
    cursor = 0
    def decode(kind):
        nonlocal cursor
        size, = struct.unpack_from("<I", raw, cursor)
        cursor += 4
        record = kind.FromString(raw[cursor:cursor + size])
        cursor += size
        return record
    nodes = [decode(runtime.Node) for _ in range(graph.node_count)]
    edges = [decode(runtime.Edge) for _ in range(graph.edge_count)]
    if cursor != len(raw):
        raise ValueError("trailing bytes after the recorded graph edges")
    return nodes, edges


def inspect(args):
    sys.path.insert(0, str(args.schema.resolve()))
    import package_pb2 as package
    import runtime_package_pb2 as runtime

    directory = args.package
    primary = directory / "model.safetensors"
    manifest = package.ProtectedPackage.FromString(tensor(primary, "flare.manifest"))
    locations = {r.payload_id: r for r in package.LocationSet.FromString(tensor(primary, "flare.locations")).locations}
    reports = []
    for signed in manifest.ranks:
        rank = package.RankBody.FromString(signed.body)
        for graph in rank.graphs:
            if graph.name not in {f"speculative1x8-tokens{n}" for n in (4, 7, 8)}:
                continue
            rows = int(graph.name.rsplit("tokens", 1)[1])
            nodes, edges = read_graph(directory, primary, locations, rank.resources[graph.resource], graph, runtime)
            names = [rank.functions[n.kernel.function].symbol if n.WhichOneof("operation") == "kernel" else "" for n in nodes]
            observed = Counter()
            ancestors = [0] * len(nodes)
            parents = [[] for _ in nodes]
            children = [[] for _ in nodes]
            for edge in edges:
                parents[edge.destination].append(edge.source)
                children[edge.source].append(edge.destination)
            pending = [len(x) for x in parents]
            ready = [i for i, count in enumerate(pending) if count == 0]
            count = 0
            while ready:
                i = ready.pop()
                count += 1
                for j in children[i]:
                    ancestors[j] |= ancestors[i] | 1 << i
                    pending[j] -= 1
                    if not pending[j]:
                        ready.append(j)
            if count != len(nodes):
                raise ValueError("verification graph has a cycle")
            for i, name in enumerate(names):
                if "RowBlockAllGatherPair" in name:
                    observed["row_gather"] += 1
                    if tuple(nodes[i].kernel.grid) != (4, 1, 1) or tuple(nodes[i].kernel.block) != (512, 1, 1):
                        raise ValueError("row gather has an unexpected recorded launch geometry")
                if "ParallelPushIslandRS" in name:
                    observed["parallel_flags"] += 1
                if "PrefixGather" in name:
                    observed["global_gather"] += 1
                if "local_candidate_slots" in name:
                    observed["local_selection"] += 1
                    # Follow immediate predecessors through copy/event nodes only. An
                    # earlier layer's indexer or query gather cannot satisfy this check.
                    predecessors = []
                    todo = list(parents[i])
                    visited = set()
                    while todo:
                        j = todo.pop()
                        if j in visited:
                            continue
                        visited.add(j)
                        if names[j] and not ("copy_kernel" in names[j] or "direct_copy" in names[j]):
                            predecessors.append(names[j])
                        else:
                            todo.extend(parents[j])
                    observed["has_indexer_predecessor"] += any("Indexer" in n for n in predecessors)
                    observed["has_query_predecessor"] += any("Gather" in n for n in predecessors)
            reports.append(dict(rank=rank.rank, graph=graph.name, rows=rows, nodes=len(nodes), edges=len(edges),
                                kernels=dict(observed)))
    if {(r["rank"], r["rows"]) for r in reports} != {(rank, n) for rank in range(16) for n in (4, 7, 8)}:
        raise ValueError("package does not contain all 48 rank/verification graphs")
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    for r in reports:
        rows, found = r["rows"], Counter(r["kernels"])
        expected = {"row_gather": 92 if rows == 4 else 0, "local_selection": 24 if rows == 4 else 0,
                    "global_gather": 0 if rows == 4 else 24, "parallel_flags": {4: 278, 7: 0, 8: 92}[rows]}
        if args.stock:
            expected = {"row_gather": 0, "local_selection": 0, "parallel_flags": 0, "global_gather": 24}
        if any(found[key] != value for key, value in expected.items()):
            raise ValueError(f"rank {r['rank']} graph {r['graph']}: expected {expected}, observed {dict(found)}")
    print("Verified collective kernel counts on sixteen ranks in three native verification graphs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stock", action="store_true")
    inspect(parser.parse_args())
