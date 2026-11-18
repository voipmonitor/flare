# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Destroying pooled-stream graphs must preserve a retained GEMM's workspace."""

from contextlib import nullcontext
import gc
import os
from types import SimpleNamespace
import weakref

import pytest
import torch

from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature
from ignition.preparation.package_storage import PackageStorage


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@torch.no_grad()
def test_capture_keeps_workspace_through_unrelated_graph_destruction():
    torch.cuda.set_device(0)
    storage = PackageStorage(device=0)
    with storage:
        source = torch.ones((512, 7168), device="cuda", dtype=torch.bfloat16)
        weight = torch.ones((7168, 3208), device="cuda", dtype=torch.bfloat16)
        output = torch.empty((512, 3208), device="cuda", dtype=torch.bfloat16)
        temporary = torch.empty(1, device="cuda")
        supplied = torch.cuda.Stream()
        events = []

        def restore(name):
            # Callbacks may explicitly enqueue writes on the supplied stream.
            with torch.cuda.stream(supplied):
                source.fill_(1)
            events.append(name)

        options = dict(session=SimpleNamespace(capture=nullcontext), device="cuda:0",
                       stream=supplied, pool=storage.graph_pool.id, compile_decoder=False,
                       warmup_storage=storage, before_run=restore)
        graphs, _ = capture_forwards({"prefill": lambda: torch.mm(source, weight, out=output)},
                                    {"prefill": GraphSignature("prefill", 1, 512)}, **options)
        graph = graphs.pop("prefill")
        owner = weakref.ref(graph._ignition_capture_stream)
        assert owner().cuda_stream != supplied.cuda_stream
        assert events == ["prefill", "prefill"]
        try:
            # Revisit every pooled stream, including the caller's stream.
            # A workspace allocated outside the graph pool is actually unmapped
            # by PackageStorage when an unrelated reset drops its final owner.
            streams = [supplied, *(torch.cuda.Stream() for _ in range(64))]
            assert all(stream.cuda_stream != owner().cuda_stream for stream in streams)
            for index, stream in enumerate(streams):
                auxiliary = torch.cuda.CUDAGraph()
                with torch.cuda.graph(auxiliary, stream=stream):
                    temporary.fill_(0)
                auxiliary.replay()
                torch.cuda.synchronize()
                auxiliary.reset()
                torch.cuda.empty_cache()
                value = 1 if index % 2 else -1
                source.fill_(value)
                graph.replay()
                torch.cuda.synchronize()
                assert bool((output == value * 7168).all())
        finally:
            graph.reset()
        del graph
        gc.collect()
        assert owner() is None
