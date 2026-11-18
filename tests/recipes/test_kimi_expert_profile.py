"""Fixed expert choices preserve Kimi's precision, geometry, and provider validation."""

from types import SimpleNamespace

import pytest
import torch
from b12x.moe import fused_moe as moe
from b12x.moe.fused_moe._tuning import MoeDecodeQuery, TUNING
from b12x.preparation import DeviceIdentity, FrozenMapping

from ignition.recipes.kimi3.expert_profile import execution_config, weight_packing


def weight_plan(packing, *, activation="situ", dtype=torch.bfloat16):
    return moe.plan_weights(
        source=moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
        activation=moe.ActivationSpec(mode="a16", nonlinearity=activation, io_dtype=dtype),
        geometry=moe.MoEGeometry(num_experts=896, hidden_size=3584, intermediate_size=192),
        constraints=moe.WeightPlanConstraints(required_packing=packing),
    )


@pytest.fixture
def blackwell(monkeypatch):
    properties = SimpleNamespace(major=12, minor=0, multi_processor_count=188)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: properties)
    return properties


@pytest.mark.parametrize("world,intermediate,expected", [(16, 192, "mma_packed"), (8, 384, None),
                                                        (6, 512, None), (16, 352, None)])
def test_profile_requires_measured_geometry(blackwell, world, intermediate, expected):
    assert weight_packing(experts=896, hidden=3584, intermediate=intermediate,
                          world_size=world, device="cuda:0") == expected


def test_packed_plan_keeps_numeric_format_and_shard_extent(blackwell):
    packing = weight_packing(experts=896, hidden=3584, intermediate=192, world_size=16, device="cuda:0")
    native, packed = weight_plan(None), weight_plan(packing)
    assert native.prepared_format.packing == "source_native"
    assert packed.prepared_format.packing == "mma_packed"
    assert native.geometry == packed.geometry
    assert native.activation == packed.activation
    assert native.prepared_format.weights == packed.prepared_format.weights
    assert native.prepared_format.scales == packed.prepared_format.scales
    assert packed._impl.reuses_source_storage


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8, 128, 256, 512, 1024, 1536, 2048])
def test_fixed_choices_are_legal_provider_overrides(blackwell, rows):
    config = execution_config(weight_plan("mma_packed"), rows, device="cuda:0")
    query = MoeDecodeQuery(
        quant_mode="w4a16", quant_modes=("w4a16",), source_format="fp4_e8m0_k32",
        activation="situ", io_dtype="bfloat16", num_experts=896, hidden_size=3584, intermediate_size=192,
        top_k=16, num_tokens=rows, routed_rows=rows * 16, route_num_experts=0, route_logits_dtype=None,
        apply_router_weight_on_input=False, collect_activation_amax=False, deterministic_output=False,
        swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None, w13_layout="w31", weight_layouts=("mma_packed",),
        w4a16_weight_layout="packed", w4a16_scale_format="e8m0_k32", w4a16_block_size_m=None,
        fast_math=True, numerical_recipe=None, controls=FrozenMapping(), csf_inline_scales=True, a4_prefill=False,
    )
    selected = TUNING.configure(query, device=DeviceIdentity("nvidia", (12, 0), 188, "SM120"), override=config)
    assert selected.pinned == config
    assert config.w4a16_pipeline_stages == (4 if rows <= 8 else 3)
    assert not config.csf_inline_scales


@pytest.mark.parametrize("change", ["native", "activation", "dtype", "device", "rows", "foreign"])
def test_profile_does_not_override_other_contracts(blackwell, change):
    plan = weight_plan("source_native" if change == "native" else "mma_packed",
                       activation="silu" if change == "activation" else "situ",
                       dtype=torch.float16 if change == "dtype" else torch.bfloat16)
    if change == "device":
        blackwell.multi_processor_count = 48
    assert execution_config(None if change == "foreign" else plan,
                            64 if change == "rows" else 8, device="cuda:0") is None
