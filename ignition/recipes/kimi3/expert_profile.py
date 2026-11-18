# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit MXFP4 expert preparation for Kimi shards on 188-SM Blackwell GPUs."""

import torch
from b12x.moe.fused_moe import MoeDecodeConfig, WeightPlan


def weight_packing(*, experts, hidden, intermediate, world_size, device):
    if (experts, hidden, intermediate, world_size) != (896, 3584, 192, 16):
        return None
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor, properties.multi_processor_count) != (12, 0, 188):
        return None
    # These 64-aligned shards fit packed FC2 K64 tiles without padding. Packing
    # permutes FP4 codes in their existing allocation and leaves scales exact.
    return "mma_packed"


def execution_config(plan, rows, *, device):
    if not isinstance(plan, WeightPlan):
        return None
    geometry, activation = plan.geometry, plan.activation
    if ((geometry.num_experts, geometry.hidden_size, geometry.intermediate_size) != (896, 3584, 192)
            or plan.prepared_format.packing != "mma_packed"
            or activation.mode != "a16" or activation.nonlinearity != "situ"
            or activation.io_dtype != torch.bfloat16
            or plan.source.format != "fp4_e8m0_k32"
            or not (1 <= rows <= 8 or rows in (128, 256, 512, 1024, 1536, 2048))):
        return None
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor, properties.multi_processor_count) != (12, 0, 188):
        return None
    # Fixed choices keep preparation reproducible without another kernel race.
    # Separate scale expansion amortizes decoding across each expert's tiles.
    return MoeDecodeConfig(
        backend="w4a16", route_planner="internal", max_active_clusters=None,
        w4a16_route_mode="packed", csf_inline_scales=False,
        w4a16_tile_config=(128, 64, 64, 128), w4a16_block_size_m=16,
        w4a16_pipeline_stages=4 if rows <= 8 else 3,
    )
