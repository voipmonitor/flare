# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Placeholders for process-local resources survive the pickling of compiled-forward guards."""

import pickle

import pytest
import torch
from cutlass.cute.runtime import from_dlpack

from ignition.preparation import forward_metadata
from ignition.preparation.forward_metadata import _Unavailable


def test_placeholder_round_trips_through_pickle():
    restored = pickle.loads(pickle.dumps(_Unavailable("RoCE proxy")))
    assert restored.description == "RoCE proxy"
    with pytest.raises(RuntimeError, match="RoCE proxy.post"):
        restored.post


def test_forward_metadata_replaces_cute_runtime_tensors_with_placeholders():
    # b12x bindings such as the TMEM KDA prefill binding keep CuTe DLPack descriptors of their scratch views.
    descriptor = from_dlpack(torch.zeros(4, 128, dtype=torch.bfloat16), assumed_align=16)
    restored = forward_metadata.loads(forward_metadata.dumps({"tmem_tensors": {"qd": descriptor}}))
    placeholder = restored["tmem_tensors"]["qd"]
    assert isinstance(placeholder, _Unavailable)
    with pytest.raises(RuntimeError, match="cutlass.cute.runtime._Tensor.shape"):
        placeholder.shape


def test_forward_metadata_keeps_transport_geometry_and_excludes_driver_stream():
    from cuda.bindings import driver
    from b12x.comm.pcie._island_copy import PacketTransport
    from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
    from b12x.comm.pcie.pcie_island_rs import PCIeIslandRSAllReduce
    from ignition.components.collectives import PreparedAllReduce

    # A previously prepared DMA candidate leaves its driver stream on the owning rank.
    # The compiler worker needs the handle of Flare's bound operation and its TP geometry.
    transport = object.__new__(PacketTransport)
    transport._handle = driver.CUstream(17)
    island = object.__new__(PCIeIslandRSAllReduce)
    island._packet_transport = transport
    runtime = object.__new__(PCIeAllReduce)
    runtime._island_rs = island
    runtime.world_size = 16
    operation = object.__new__(PreparedAllReduce)
    operation.runtime, operation.handle = runtime, 71

    worker = forward_metadata.loads(forward_metadata.dumps(operation))
    assert worker.handle == 71
    assert worker.runtime.world_size == 16
    stream = worker.runtime._island_rs._packet_transport._handle
    assert isinstance(stream, _Unavailable)
    with pytest.raises(RuntimeError, match=r"cuda\.bindings\.driver\.CUstream"):
        stream.value
