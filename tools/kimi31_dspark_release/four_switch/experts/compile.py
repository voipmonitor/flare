"""Compile one fixed SM120 expert stage with the FlashInfer source used for its signed image."""

import json
import multiprocessing
from pathlib import Path
import sys
from unittest.mock import patch

from b12x._lib.compile_pool import _initialize_worker

_initialize_worker(
    0,
    (12, 0),
    "kimi-expert-p4",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    188,
    101376,
    102400,
    multiprocessing.Array("i", [0, 0, 0]),
)
from cuda.bindings import driver as cuda
from b12x.moe._shared.kernels.w4a16.kernel import fused_launch

stage = sys.argv[1]
if stage not in ("fc1", "activation", "fc2"):
    raise ValueError("expected an FC1, activation or FC2 expert stage")
params = json.loads(Path(__file__).with_name("expert.json").read_text())["source_arguments"]
params["force_tile_config"] = tuple(params["force_tile_config"])
with patch.object(fused_launch, "current_cuda_stream", lambda: cuda.CUstream(0)):
    fused_launch.compile_w4a16_fused_moe(**params, fixed_work_grid=376, expert_stage=stage)
