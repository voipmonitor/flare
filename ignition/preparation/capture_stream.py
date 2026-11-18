# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA streams owned by a captured graph catalog instead of PyTorch's stream pool."""

from weakref import finalize

import torch


def capture_stream(device):
    """Keep unrelated graph destruction from clearing a catalog's cuBLAS workspace."""
    from cuda.bindings import driver as cuda

    with torch.cuda.device(device):
        status, handle = cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING)
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cannot create graph capture stream: {status}")
        try:
            stream = torch.cuda.ExternalStream(int(handle), device=device)
        except BaseException:
            cuda.cuStreamDestroy(handle)
            raise
    finalize(stream, cuda.cuStreamDestroy, handle).atexit = False
    return stream
