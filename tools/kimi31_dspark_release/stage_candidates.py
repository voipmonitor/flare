"""Distribute eight complete, ordered global selection rows to each of sixteen ranks."""

from struct import pack
from stage_records import (
    SLICE,
    TOPK,
    LAYER_COUNT,
    U32,
    U64,
    rt,
    read_graph,
    topology,
    require,
    scalar,
    bounds,
    stage_package,
)

WORLD = 16
LOCAL = SLICE // WORLD
# A four-row ring writes less than one MiB per bank. Use a disjoint region
# after a channel-wide barrier, and release it with the result all-gather.
STAGE_OFFSET = 4 << 20
PACKED_BYTES = SLICE * TOPK * 8


def uint(value):
    return rt.Blob(data=U32.pack(value))


def address(value):
    return rt.Blob(data=U64.pack(value))


def clone(node):
    return type(node).FromString(node.SerializeToString())


def launch(fid, grid, block, args):
    node = rt.Node()
    kernel = node.kernel
    kernel.function = fid
    kernel.grid.extend(grid)
    kernel.block.extend([block, 1, 1])
    kernel.arguments.extend(args)
    return node


def patch(rank, directory, primary, locations, graph, rows, functions, _query_length):
    nodes, edges = read_graph(directory, primary, locations, rank.resources[graph.resource], graph, rt)
    parents, children, order = topology(nodes, edges)

    def symbol(i):
        return (
            rank.functions[nodes[i].kernel.function].symbol
            if nodes[i].HasField("kernel")
            else nodes[i].WhichOneof("operation")
        )

    names = [symbol(i) for i in order]
    sites = [j for j, s in enumerate(names) if s == "prefill_record_pages"]
    require(len(sites) == LAYER_COUNT * (rows // SLICE), "unexpected record-page selection sites")
    for end in sites:
        chain = order[end - 12 : end + 1]
        for src, dst in zip(chain, chain[1:]):
            require(
                children[src] == [(dst, bytes(8))] and parents[dst] == [(src, bytes(8))],
                "candidate fold is not a single captured dependency chain",
            )
        (
            local,
            pack_scores,
            copy0,
            copy1,
            original_ring,
            active_copy,
            count,
            unpack,
            topk,
            cutoff,
            counts,
            emit,
            selection,
        ) = [clone(nodes[i]) for i in chain]
        original = names[end - 12 : end + 1]
        require(
            "DSATiledTopkKernel" in original[0]
            and original[1].startswith("triton_poi")
            and all("direct_copy" in name for name in original[2:4])
            and "RingGather" in original[4]
            and original[5:8] == ["copy", "_count", "_unpack"]
            and "DSATiledTopkKernel" in original[8]
            and original[9:] == ["_cutoff", "_counts", "_emit", "prefill_record_pages"],
            "candidate operations differ from the checked graph",
        )
        pa = pack_scores.kernel.arguments
        ring = original_ring.kernel.arguments
        sa = selection.kernel.arguments
        ta = topk.kernel.arguments
        la = local.kernel.arguments
        require(
            len(pa) == 8
            and la[4].data[:8] == pa[0].data
            and la[3].data[:8] == pa[1].data
            and scalar(pa[5]) == SLICE * TOPK
            and active_copy.copy.nbytes == SLICE,
            "local score or ID bindings changed",
        )
        require(
            len(ring) == 39
            and all(not b.relocations for b in ring)
            and [len(b.data) for b in ring] == [8] * 35 + [4] + [8] * 3
            and scalar(ring[35]) == SLICE
            and scalar(ring[36]) >= STAGE_OFFSET + PACKED_BYTES
            and scalar(ring[36]) < 1 << 31,
            "peer staging banks do not cover the per-rank candidate array",
        )
        require(
            count.kernel.arguments[0] == unpack.kernel.arguments[0] == ring[1]
            and count.kernel.arguments[1] == unpack.kernel.arguments[1]
            and sa[0] == emit.kernel.arguments[7]
            and scalar(ta[8]) == WORLD * TOPK
            and scalar(ta[9]) == SLICE
            and scalar(ta[15]) == TOPK,
            "global fold ABI differs from the native row layout",
        )
        peer = [scalar(b) + STAGE_OFFSET for b in ring[3:19]]
        bounds(rank, peer[rank.rank], PACKED_BYTES)
        for local_buffer, span in (
            (scalar(ring[1]), LOCAL * WORLD * TOPK * 8),
            (scalar(emit.kernel.arguments[7]), LOCAL * TOPK * 4),
        ):
            bounds(rank, local_buffer, span)

        def collective(n, source):
            op = clone(original_ring)
            op.kernel.arguments[0].CopyFrom(source)
            op.kernel.arguments[35].CopyFrom(uint(n))
            return op

        # Every rank retires prior channel users before reusing this bank. The
        # second ring publishes all local candidates. The last returns exact IDs.
        barrier = collective(1, pa[0])
        publish = launch(
            functions[0], [512, 1, 1], 256, [pa[0], pa[1], address(peer[rank.rank]), uint(rank.rank)]
        )
        read = launch(
            functions[1],
            [WORLD, LOCAL, 1],
            256,
            [rt.Blob(data=pack("<16Q", *peer)), ring[1], count.kernel.arguments[1], sa[1], uint(rank.rank)],
        )
        for node in (count, unpack, topk, cutoff, counts, emit):
            require(node.kernel.grid[0] == SLICE, "unexpected fold grid row dimension")
            node.kernel.grid[0] = LOCAL
        topk.kernel.arguments[9].CopyFrom(uint(LOCAL))
        agree = collective(LOCAL // 2, emit.kernel.arguments[7])
        selection.kernel.arguments[0].CopyFrom(ring[1])
        # The original top-k candidate kernel and the surrounding graph edges
        # remain in place. Each replacement has the same predecessor boundary.
        ops = [barrier, publish, barrier, read, count, unpack, topk, cutoff, counts, emit, agree, selection]
        for i, node in zip(chain[1:], ops, strict=True):
            nodes[i].CopyFrom(node)
    graph.functions[:] = sorted({n.kernel.function for n in nodes if n.HasField("kernel")})
    raw = bytearray()
    for op in [*nodes, *edges]:
        data = op.SerializeToString(deterministic=True)
        raw += U32.pack(len(data)) + data
    return raw, len(sites)


if __name__ == "__main__":
    stage_package(
        (("publish_candidates", [8] * 3 + [4]), ("read_candidate_rows", [8 * 16] + [8] * 3 + [4])),
        "query-row-candidates",
        patch,
    )
