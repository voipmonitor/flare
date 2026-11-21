# SPDX-License-Identifier: Apache-2.0
"""Prepare four-row PCIe reductions together with their Kimi normalization consumers."""

from dataclasses import dataclass
from functools import partial
from hashlib import sha256
import json
import itertools
import weakref

import torch
import triton
import triton.language as tl
from b12x.comm.pcie.island_row import prepared_rows, reduce_rows, select_row_protocol
from b12x.comm.pcie._island_rs_cute import (
    get_island_norm3584_launcher,
    get_island_resnorm7168_launcher,
)
from b12x.comm.pcie._island_v4_mix import get_island_mix7168_launcher
from b12x.preparation import PreparedCall

from ignition.components.collectives import _all_reduce
from ignition.preparation.capture import capture_forwards
from ignition.preparation.composition_tuning import PreparedComposition, plan_identity, race_compositions
from ignition.preparation.execution import GraphSignature
from .pointwise import _rms_norm, rms_norm_columns
from .residual import ADD, REPLACE, _mix, ResidualStream, TILES


@triton.jit
def _reduced_norm(
    Source,
    Full,
    Weight,
    Out,
    Begin,
    L0,
    L1,
    L2,
    L3,
    X0,
    X1,
    X2,
    X3,
    stage,
    part,
    final,
    capacity,
    eps,
    LANE: tl.constexpr,
    ISLAND: tl.constexpr,
    ROWS: tl.constexpr,
    PROTOCOL: tl.constexpr = 0,
):
    reduced = reduce_rows(
        Source, Full, L0, L1, L2, L3, X0, X1, X2, X3, stage, part, final, capacity,
        LANE, ISLAND, 3584, ROWS, False, Source, False, False, True, PROTOCOL,
    )
    if tl.program_id(0) < ROWS:
        _rms_norm(reduced, Weight, Out, Begin, 3584, 224, eps, 3584, 224, True, 4096, True)


@triton.jit
def _reduced_mix(
    Prefix,
    Source,
    Addend,
    Full,
    Blocks,
    Score,
    Norm,
    Out,
    Mixture,
    L0,
    L1,
    L2,
    L3,
    X0,
    X1,
    X2,
    X3,
    stage,
    part,
    final,
    capacity,
    prefix_stride,
    delta_stride,
    block_row_stride,
    block_stride,
    output_stride,
    mixture_stride,
    eps,
    output_eps,
    LANE: tl.constexpr,
    ISLAND: tl.constexpr,
    ROWS: tl.constexpr,
    SOURCES: tl.constexpr,
    WRITE: tl.constexpr,
    MODE: tl.constexpr,
    AUX: tl.constexpr,
    TILE: tl.constexpr,
    HAS_ADDEND: tl.constexpr = False,
    PROTOCOL: tl.constexpr = 0,
):
    reduced = reduce_rows(
        Source, Full, L0, L1, L2, L3, X0, X1, X2, X3, stage, part, final, capacity,
        LANE, ISLAND, 7168, ROWS, False, Addend, HAS_ADDEND, False, True, PROTOCOL,
    )
    if tl.program_id(0) < ROWS:
        _mix(
            Prefix,
            reduced,
            Blocks,
            Score,
            Norm,
            Out,
            Mixture,
            prefix_stride,
            7168,
            block_row_stride,
            block_stride,
            output_stride,
            mixture_stride,
            eps,
            output_eps,
            7168,
            SOURCES,
            WRITE,
            MODE,
            AUX,
            8192,
            TILE,
            True,
        )


@triton.jit
def _reduced_residual_norm(
    Source,
    Residual,
    Full,
    Weight,
    ResidualOut,
    NormOut,
    L0,
    L1,
    L2,
    L3,
    X0,
    X1,
    X2,
    X3,
    stage,
    part,
    final,
    capacity,
    res_stride,
    res_out_stride,
    norm_out_stride,
    eps,
    LANE: tl.constexpr,
    ISLAND: tl.constexpr,
    ROWS: tl.constexpr,
    PROTOCOL: tl.constexpr = 0,
):
    reduced = reduce_rows(
        Source, Full, L0, L1, L2, L3, X0, X1, X2, X3, stage, part, final, capacity,
        LANE, ISLAND, 7168, ROWS, False, Source, False, False, True, PROTOCOL,
    )
    if tl.program_id(0) < ROWS:
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, 8192)
        valid = cols < 7168
        red = tl.load(reduced + row * 7168 + cols, valid, other=0.0, cache_modifier=".cg").to(tl.float32)
        res = tl.load(Residual + row * res_stride + cols, valid, other=0.0).to(tl.float32)
        updated = (res + red).to(tl.bfloat16)
        tl.store(ResidualOut + row * res_out_stride + cols, updated, valid)
        val = updated.to(tl.float32)
        normed = (val * tl.rsqrt(tl.sum(val * val) / 7168.0 + eps)).to(tl.bfloat16)
        w = tl.load(Weight + cols, valid, other=0.0).to(tl.float32)
        tl.store(NormOut + row * norm_out_stride + cols, (w * normed.to(tl.float32)).to(tl.bfloat16), valid)


_HANDLES = itertools.count()
_OWNERS = weakref.WeakValueDictionary()
# CPU compilation workers trace custom operations by handle. CUDA IPC objects
# and mounted programs belong exclusively to the process that records the graph.
_BINDINGS = weakref.WeakKeyDictionary()


@dataclass
class _Binding:
    control: object
    layout: object = None


@torch.library.custom_op("flare::kimi3_reduce_norm_columns", mutates_args=("full", "out"))
def _norm_rows(
    source: torch.Tensor,
    full: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    begin: torch.Tensor,
    eps: float,
    handle: int,
) -> None:
    layout = _OWNERS[handle].layout
    owner = layout.owner
    if (
        layout.width == 3584
        and 1 <= layout.rows <= 32
        and out.shape == (layout.rows, 224)
        and source.is_contiguous()
        and out.is_contiguous()
        and weight.is_contiguous()
        and source.data_ptr() % 16 == 0
        and out.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and begin.dtype == torch.int32
    ):
        launcher = get_island_norm3584_launcher(
            owner.rank, layout.rows, owner.device.index or 0, blocks=owner.blocks
        )
        launcher(
            owner._slab_ptrs,
            source.data_ptr(),
            weight.data_ptr(),
            out.data_ptr(),
            begin.data_ptr(),
            owner.stage_offset,
            owner.part_offset,
            owner.final_offset,
            owner.quarter_capacity,
            float(eps),
            owner.blocks,
        )
        return
    protocol = select_row_protocol(layout.rows, layout.width)
    _reduced_norm[layout.grid](
        source,
        full,
        weight,
        out,
        begin,
        *layout.arguments(source, full),
        eps,
        *layout.rank_coordinates,
        layout.rows,
        protocol,
        num_warps=8,
    )


@_norm_rows.register_fake
def _fake_norm_rows(source, full, weight, out, begin, eps, handle):
    pass


@torch.library.custom_op(
    "flare::kimi3_reduce_residual_mix", mutates_args=("full", "prefix", "blocks", "out", "mixture")
)
def _mix_rows(
    source: torch.Tensor,
    addend: torch.Tensor | None,
    full: torch.Tensor,
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    score: torch.Tensor,
    norm: torch.Tensor,
    out: torch.Tensor,
    mixture: torch.Tensor | None,
    eps: float,
    output_eps: float,
    handle: int,
) -> None:
    target = _OWNERS[handle]
    layout, spec = target.layout, target.spec
    tap = out if mixture is None else mixture
    has_addend = addend is not None
    add_tensor = source if addend is None else addend
    if has_addend and (
        add_tensor.data_ptr() % 16
        or not add_tensor.is_contiguous()
        or add_tensor.dtype != torch.bfloat16
        or add_tensor.shape != source.shape
        or add_tensor.device != source.device
    ):
        raise ValueError("PCIe row reduction addend must match 16-byte aligned contiguous BF16 source rows")
    owner = layout.owner
    ab = 2 if layout.rows <= 3 else (4 if layout.rows <= 20 else 8)
    if (
        layout.width == 7168
        and 1 <= layout.rows <= 32
        and spec.mode in (ADD, REPLACE)
        and 0 <= spec.sources <= 8
        and owner.blocks >= ab + layout.rows
        and source.is_contiguous()
        and source.data_ptr() % 16 == 0
        and add_tensor.is_contiguous()
        and add_tensor.data_ptr() % 16 == 0
        and prefix.stride(1) == 1
        and prefix.stride(0) % 8 == 0
        and prefix.data_ptr() % 16 == 0
        and blocks.stride(2) == 1
        and blocks.stride(1) % 8 == 0
        and blocks.stride(0) % 8 == 0
        and blocks.data_ptr() % 16 == 0
        and score.is_contiguous()
        and score.dtype == torch.float32
        and score.data_ptr() % 16 == 0
        and norm.is_contiguous()
        and norm.data_ptr() % 16 == 0
        and out.stride(1) == 1
        and out.stride(0) % 8 == 0
        and out.data_ptr() % 16 == 0
        and tap.stride(1) == 1
        and tap.stride(0) % 8 == 0
        and tap.data_ptr() % 16 == 0
    ):
        launcher = get_island_mix7168_launcher(
            owner.rank,
            layout.rows,
            spec.sources,
            spec.write,
            spec.mode,
            mixture is not None,
            has_addend,
            owner.device.index or 0,
            blocks=owner.blocks,
        )
        launcher(
            owner._slab_ptrs,
            source.data_ptr(),
            add_tensor.data_ptr(),
            prefix.data_ptr(),
            blocks.data_ptr(),
            score.data_ptr(),
            norm.data_ptr(),
            out.data_ptr(),
            tap.data_ptr(),
            owner.stage_offset,
            owner.part_offset,
            owner.final_offset,
            owner.quarter_capacity,
            prefix.stride(0),
            blocks.stride(0),
            blocks.stride(1),
            out.stride(0),
            tap.stride(0),
            float(eps),
            float(output_eps),
            owner.blocks,
        )
        return
    tile, warps = TILES[min(spec.sources, len(TILES) - 1)]
    protocol = select_row_protocol(layout.rows, layout.width)
    _reduced_mix[layout.grid](
        prefix,
        source,
        add_tensor,
        full,
        blocks,
        score,
        norm,
        out,
        tap,
        *layout.arguments(source, full),
        prefix.stride(0),
        full.stride(0),
        blocks.stride(0),
        blocks.stride(1),
        out.stride(0),
        tap.stride(0),
        eps,
        output_eps,
        *layout.rank_coordinates,
        layout.rows,
        spec.sources,
        spec.write,
        spec.mode,
        mixture is not None,
        tile,
        has_addend,
        protocol,
        num_warps=warps,
    )


@_mix_rows.register_fake
def _fake_mix_rows(source, addend, full, prefix, blocks, score, norm, out, mixture, eps, output_eps, handle):
    pass


@torch.library.custom_op(
    "flare::kimi3_reduce_residual_norm", mutates_args=("full", "residual_out", "norm_out")
)
def _residual_norm_rows(
    source: torch.Tensor,
    residual: torch.Tensor,
    full: torch.Tensor,
    weight: torch.Tensor,
    residual_out: torch.Tensor,
    norm_out: torch.Tensor,
    eps: float,
    handle: int,
) -> None:
    layout = _OWNERS[handle].layout
    owner = layout.owner
    if (
        layout.width == 7168
        and 1 <= layout.rows <= 32
        and source.is_contiguous()
        and weight.is_contiguous()
        and residual.stride(1) == 1
        and residual_out.stride(1) == 1
        and norm_out.stride(1) == 1
        and residual.stride(0) % 8 == 0
        and residual_out.stride(0) % 8 == 0
        and norm_out.stride(0) % 8 == 0
        and source.data_ptr() % 16 == 0
        and residual.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and residual_out.data_ptr() % 16 == 0
        and norm_out.data_ptr() % 16 == 0
    ):
        launcher = get_island_resnorm7168_launcher(
            owner.rank, layout.rows, owner.device.index or 0, blocks=owner.blocks
        )
        launcher(
            owner._slab_ptrs,
            source.data_ptr(),
            residual.data_ptr(),
            weight.data_ptr(),
            residual_out.data_ptr(),
            norm_out.data_ptr(),
            owner.stage_offset,
            owner.part_offset,
            owner.final_offset,
            owner.quarter_capacity,
            residual.stride(0),
            residual_out.stride(0),
            norm_out.stride(0),
            float(eps),
            owner.blocks,
        )
        return
    protocol = select_row_protocol(layout.rows, layout.width)
    _reduced_residual_norm[layout.grid](
        source,
        residual,
        full,
        weight,
        residual_out,
        norm_out,
        *layout.arguments(source, full),
        residual.stride(0),
        residual_out.stride(0),
        norm_out.stride(0),
        eps,
        *layout.rank_coordinates,
        layout.rows,
        protocol,
        num_warps=8,
    )


@_residual_norm_rows.register_fake
def _fake_residual_norm_rows(source, residual, full, weight, residual_out, norm_out, eps, handle):
    pass


@dataclass(frozen=True)
class RowGeometry:
    width: int
    sources: int = 0
    write: int = -1
    mode: int = 0
    blocks: int = 0
    taps: int = 0
    rows: int = 4
    addend: bool = False
    residual_norm: bool = False
    eps: float | None = None

    @property
    def threads(self):
        if self.residual_norm:
            return 256
        return 256 if self.width == 3584 else TILES[min(self.sources, len(TILES) - 1)][1] * 32


@dataclass(frozen=True)
class _ControlInvocation:
    # Flare's AOT binder relocates .handle in each recording process.
    handle: int

    def __call__(self, source, output):
        _all_reduce(source, output, self.handle)


class RowReductionChoice:
    """The same functional operands feed the selected separate or row-fused implementation."""

    def __init__(self, reduction, geometry):
        control = reduction.parts[0] if hasattr(reduction, "parts") else reduction
        self.control = _ControlInvocation(control.handle)
        self.spec = geometry
        self.fused = False
        _BINDINGS[self] = _Binding(control)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    @property
    def layout(self):
        return _BINDINGS[self].layout

    @property
    def control_plan(self):
        return _BINDINGS[self].control.plan

    def sum(self, source, out=None):
        if out is None:
            out = torch.empty_like(source)
        self.control(source, out)
        return out

    def norm(self, source, weight, begin, eps):
        output = torch.empty((source.shape[0], weight.shape[0]), dtype=source.dtype, device=source.device)
        if self.fused:
            _norm_rows(source, torch.empty_like(source), weight, output, begin, float(eps), self.handle)
            return output
        return rms_norm_columns(self.sum(source), weight, output, eps, begin=begin)

    def residual_norm(self, source, residual, weight, eps):
        residual_out = torch.empty_like(residual)
        norm_out = torch.empty_like(residual)
        if self.fused:
            _residual_norm_rows(
                source, residual, torch.empty_like(source), weight, residual_out, norm_out, float(eps), self.handle
            )
            return residual_out, norm_out
        residual_out.copy_(residual + self.sum(source))
        rms_norm_columns(residual_out, weight, norm_out, eps)
        return residual_out, norm_out

    def fused_mix(self, residual, source, *, score, norm, eps, output_eps, mixture, output, addend=None):
        _mix_rows(
            source,
            addend,
            torch.empty_like(source),
            residual.prefix,
            residual.blocks,
            score,
            norm,
            output,
            mixture,
            float(eps),
            float(output_eps),
            self.handle,
        )
        return output


class _ReductionFixture:
    """Private trial state, including each observable residual update and consumer output."""

    def __init__(self, choice, eps):
        self.choice = choice
        spec, owner = choice.spec, choice.layout.owner
        self.eps = float(eps if spec.eps is None else spec.eps)
        device = owner.device
        rows = spec.rows
        self.source = torch.empty((rows, spec.width), dtype=torch.bfloat16, device=device)
        self.full = torch.empty_like(self.source)
        columns = 224 if spec.width == 3584 else spec.width
        self.out = torch.empty((rows, columns), dtype=torch.bfloat16, device=device)
        self.norm = torch.empty(columns, dtype=torch.bfloat16, device=device)
        self.begin = torch.tensor([owner.rank * 224], dtype=torch.int32, device=device)
        self.addend = torch.empty_like(self.source) if spec.addend else None
        self.state = [self.source, self.norm]
        if self.addend is not None:
            self.state.append(self.addend)
        self.outputs = [self.out]
        if spec.residual_norm:
            self.residual = torch.empty_like(self.source)
            self.residual_out = torch.empty_like(self.source)
            self.state.append(self.residual)
            self.outputs.append(self.residual_out)
        elif spec.width == 7168:
            self.prefix = torch.empty_like(self.source)
            self.blocks = torch.empty((rows, spec.blocks, 7168), dtype=torch.bfloat16, device=device)
            self.score = torch.empty(7168, dtype=torch.float32, device=device)
            self.state.extend((self.prefix, self.blocks, self.score))
            self.outputs.extend((self.prefix, self.blocks))
            self.tap = None
            if spec.taps:
                self.tap = torch.empty((rows, spec.taps * 7168), dtype=torch.bfloat16, device=device)[:, :7168]
                self.outputs.append(self.tap)
        self.patterns = []
        for index in range(4):
            generator = torch.Generator().manual_seed(307 + index)
            partials = torch.randn((16, rows, spec.width), generator=generator).to(torch.bfloat16)
            if index == 1:
                partials[:, :, ::3] = torch.tensor(-0.0, dtype=torch.bfloat16)
            if index == 2:
                partials[1::2] = -partials[::2]
            if index == 3:
                partials[::4].mul_(32)
            pattern = [partials[owner.rank].to(device)]
            for value in self.state[1:]:
                if value is self.addend:
                    add_partials = torch.randn((16, rows, spec.width), generator=generator).to(torch.bfloat16)
                    pattern.append(add_partials[owner.rank].to(device))
                else:
                    pattern.append(
                        torch.randn(value.shape, generator=generator).to(device=device, dtype=value.dtype)
                    )
            self.patterns.append(pattern)
        self.expected = []

    def produce(self, index):
        self.index = index
        for target, source in zip(self.state, self.patterns[index], strict=True):
            target.copy_(source)

    def reference(self):
        self.expected.append(tuple(v.clone() for v in self.outputs))

    def validate(self):
        for value, reference in zip(self.outputs, self.expected[self.index], strict=True):
            if not torch.equal(
                value.view(torch.int16) if value.dtype == torch.bfloat16 else value,
                reference.view(torch.int16) if reference.dtype == torch.bfloat16 else reference,
            ):
                raise AssertionError("four-row reduction changed a materialized Kimi activation")

    @property
    def nbytes(self):
        tensors = [*self.state, *self.outputs, self.begin]
        tensors.extend(v for pat in (*self.patterns, *self.expected) for v in pat)
        return sum({v.untyped_storage()._cdata: v.untyped_storage().nbytes() for v in tensors}.values())


class _PreparedTrial:
    """The complete two-kernel or fused invocation, without calibration-only tensors or IPC resources."""

    def __init__(self, fixture, fused):
        self.fused = fused
        self.choice, self.eps = fixture.choice, fixture.eps
        self.source, self.full, self.out = fixture.source, fixture.full, fixture.out
        self.addend = fixture.addend
        self.norm, self.begin = fixture.norm, fixture.begin
        if self.choice.spec.residual_norm:
            self.residual = fixture.residual
            self.residual_out = fixture.residual_out
        elif self.choice.spec.width == 7168:
            self.prefix, self.blocks, self.score, self.tap = (
                fixture.prefix,
                fixture.blocks,
                fixture.score,
                fixture.tap,
            )

    def __call__(self):
        choice, spec = self.choice, self.choice.spec
        if self.fused:
            if spec.residual_norm:
                _residual_norm_rows(
                    self.source,
                    self.residual,
                    self.full,
                    self.norm,
                    self.residual_out,
                    self.out,
                    self.eps,
                    choice.handle,
                )
            elif spec.width == 3584:
                _norm_rows(
                    self.source,
                    self.full,
                    self.norm,
                    self.out,
                    self.begin,
                    self.eps,
                    choice.handle,
                )
            else:
                _mix_rows(
                    self.source,
                    self.addend,
                    self.full,
                    self.prefix,
                    self.blocks,
                    self.score,
                    self.norm,
                    self.out,
                    self.tap,
                    self.eps,
                    self.eps,
                    choice.handle,
                )
        else:
            reduced_in = self.source if self.addend is None else (self.source + self.addend)
            choice.sum(reduced_in, self.full)
            if spec.residual_norm:
                self.residual_out.copy_(self.residual + self.full)
                rms_norm_columns(self.residual_out, self.norm, self.out, self.eps)
            elif spec.width == 3584:
                rms_norm_columns(self.full, self.norm, self.out, self.eps, begin=self.begin)
            else:
                ResidualStream(self.prefix, self.blocks, self.out).mix(
                    self.full,
                    spec.mode,
                    sources=spec.sources,
                    write=spec.write,
                    score=self.score,
                    norm=self.norm,
                    eps=self.eps,
                    output_eps=self.eps,
                    mixture=self.tap,
                    output=self.out,
                )
        return self.out


def hidden_row_choices(catalog, layers, block_size, auxiliary_layers, blocks, *, rows=4):
    """Bind the consumer of each attention and MLP reduction, including the final model norm."""
    pairs = []
    for index, (_, _, mlp, reduction) in enumerate(layers):
        committed = -(-index // block_size)
        boundary = index % block_size == 0
        after_attention = RowGeometry(
            7168, sources=committed + boundary, mode=REPLACE if boundary else ADD, blocks=blocks, rows=rows
        )
        following = index + 1
        after_mlp = RowGeometry(
            7168,
            sources=-(-following // block_size),
            write=following // block_size if following < len(layers) and following % block_size == 0 else -1,
            mode=ADD,
            blocks=blocks,
            taps=len(auxiliary_layers) if following in auxiliary_layers else 0,
            rows=rows,
            addend=hasattr(mlp, "shared"),
        )
        pairs.append((catalog.register(reduction, after_attention), catalog.register(reduction, after_mlp)))
    return tuple(pairs)


class RowReductionTuning:
    """Measure complete multi-row compositions after ordinary all-reduce plans are bound."""

    def __init__(self, coordinator, device, eps, cache_root):
        self.coordinator, self.device, self.eps, self.cache_root = coordinator, device, float(eps), cache_root
        self.choices = {}

    def register(self, reduction, geometry):
        if hasattr(reduction, "parts"):
            if len(reduction.parts) != 1:
                return None
        elif getattr(reduction, "single", True) or not hasattr(reduction, "plan"):
            return None
        target = RowReductionChoice(reduction, geometry)
        self.choices.setdefault(geometry, []).append(target)
        return target

    def groups(self):
        import torch.distributed as dist

        entries = [(geometry, target) for geometry, group in self.choices.items() for target in group]
        local = []
        for geometry, target in entries:
            plan = target.control_plan
            _BINDINGS[target].layout = prepared_rows(
                plan, rows=geometry.rows, width=geometry.width, threads=geometry.threads
            )
            key = (
                None
                if target.layout is None
                else sha256(json.dumps(plan_identity(plan), sort_keys=True).encode()).hexdigest()
            )
            local.append((geometry, key))
        ranks = [None] * len(self.coordinator.ranks)
        dist.all_gather_object(ranks, local, group=self.coordinator.group)
        geometry_order = [g for g, _ in entries]
        if any([g for g, _ in rank] != geometry_order for rank in ranks):
            raise ValueError("Kimi ranks declared different multi-row normalization geometries")
        groups = {}
        for i, (geometry, target) in enumerate(entries):
            # Identical control programs on every participant can share one measurement.
            key = tuple(rank[i][1] for rank in ranks)
            if None not in key:
                groups.setdefault((geometry, key), []).append(target)
        # For full-model catalogs, share one representative race per (rows, width, control_plans)
        # to avoid redundantly racing 18+ residual-block variants per token count.
        if len(groups) > 6:
            collapsed = {}
            for (geometry, controls), targets in groups.items():
                bucket = (geometry.rows, geometry.width, geometry.addend, geometry.residual_norm, geometry.eps, controls)
                if bucket not in collapsed:
                    collapsed[bucket] = [geometry, list(targets)]
                else:
                    # Prefer a median committed-block representative (sources ~ 4, ADD mode)
                    # so fusion races reflect the typical 93-layer residual workload rather than sources=1.
                    prev_g = collapsed[bucket][0]
                    if geometry.sources > 0 and geometry.mode == ADD:
                        if prev_g.sources == 0 or prev_g.mode != ADD or abs(geometry.sources - 4) < abs(prev_g.sources - 4):
                            collapsed[bucket][0] = geometry
                    collapsed[bucket][1].extend(targets)
            return [((geom, controls), targets) for (_, _, _, _, _, controls), (geom, targets) in collapsed.items()]
        return groups.items()

    def tune(self, session):
        if not session.autotune or session.autotune_effort == "off":
            return []
        report = []
        for (geometry, controls), targets_for_plan in self.groups():
            # ForwardCache keys compiled templates by name and graph signature.
            # Distinct reductions and epilogues need distinct persistent templates.
            identity = sha256(json.dumps((vars(geometry), controls), sort_keys=True).encode()).hexdigest()
            names = {kind: f"kimi-row-{geometry.rows}x{geometry.width}-{kind}-{identity}" for kind in ("separate", "fused")}
            representative = next(t for t in targets_for_plan if t.spec == geometry)
            fixture = _ReductionFixture(representative, self.eps)
            fixture.produce(0)
            forwards = {names[kind]: _PreparedTrial(fixture, kind == "fused") for kind in names}
            graphs, compiled = {}, {}
            try:
                graphs, compiled = capture_forwards(
                    forwards,
                    {n: GraphSignature("decode", 1, geometry.rows) for n in forwards},
                    session=session,
                    device=self.device,
                    stream=torch.cuda.Stream(device=self.device),
                    pool=torch.cuda.graph_pool_handle(),
                    coordinator=self.coordinator,
                    report_catalog=False,
                    cache_root=self.cache_root,
                    retain_forwards=True,
                )
                for i in range(len(fixture.patterns)):
                    fixture.produce(i)
                    torch.cuda.synchronize(self.device)
                    self.coordinator.barrier(f"Kimi row reference {i}", self.coordinator.ranks)
                    compiled[names["separate"]]()
                    fixture.reference()
                producers = tuple(partial(fixture.produce, i) for i in range(3))
                plan = representative.control_plan
                candidates = {}
                for kind, name in names.items():
                    func = compiled[name]
                    call = PreparedCall(
                        run=func,
                        output=fixture.out,
                        produce=producers[0],
                        benchmark_producers=producers,
                        validate=fixture.validate,
                        owners=(fixture,),
                        capture_safe=True,
                    )
                    candidates[kind] = PreparedComposition(func, (("reduction", plan),), call)
                function = (
                    _reduced_residual_norm
                    if geometry.residual_norm
                    else (_reduced_norm if geometry.width == 3584 else _reduced_mix)
                )
                op_name = (
                    "kimi-row-residual-norm-7168"
                    if geometry.residual_norm
                    else f"kimi-row-normalization-{geometry.width}"
                )
                direct_fo_tag = (
                    "sw4_grid_v9b"
                    if (geometry.residual_norm or geometry.width == 3584)
                    else "sw4_grid_v14_mix"
                )
                with session._tuning_clocks():
                    result = race_compositions(
                        candidates,
                        operation=op_name,
                        geometry=vars(geometry),
                        numerics=dict(epsilon=fixture.eps, kernel=function.cache_key, direct_fo=direct_fo_tag),
                        workload=dict(
                            version=4,
                            seeds=(307, 308, 309),
                            rows=geometry.rows,
                            addend=geometry.addend,
                            residual_norm=geometry.residual_norm,
                        ),
                        default="separate",
                        session=session,
                        coordinator=self.coordinator,
                        cache_policy="warm",
                        resident_bytes=fixture.nbytes,
                    )
                fixture.produce(3)
                torch.cuda.synchronize(self.device)
                self.coordinator.barrier("Kimi row holdout", self.coordinator.ranks)
                compiled[names[result["implementation"]]]()
                torch.cuda.synchronize(self.device)
                fixture.validate()
                for target in targets_for_plan:
                    target.fused = result["implementation"] == "fused"
                report.append(dict(geometry=vars(geometry), **result))
            finally:
                for graph in graphs.values():
                    graph.reset()
                for forward in compiled.values():
                    forward.retire_after_capture()
        return report
