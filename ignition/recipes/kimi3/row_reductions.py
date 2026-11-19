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
from b12x.comm.pcie.island_row import prepared_rows, reduce_rows
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
):
    reduce_rows(Source, Full, L0, L1, L2, L3, X0, X1, X2, X3, stage, part, final, capacity, LANE, ISLAND, 3584)
    if tl.program_id(0) < 4:
        _rms_norm(Full, Weight, Out, Begin, 3584, 224, eps, 3584, 224, True, 4096)


@triton.jit
def _reduced_mix(
    Prefix,
    Source,
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
    SOURCES: tl.constexpr,
    WRITE: tl.constexpr,
    MODE: tl.constexpr,
    AUX: tl.constexpr,
    TILE: tl.constexpr,
):
    reduce_rows(Source, Full, L0, L1, L2, L3, X0, X1, X2, X3, stage, part, final, capacity, LANE, ISLAND, 7168)
    if tl.program_id(0) < 4:
        _mix(
            Prefix,
            Full,
            Blocks,
            Score,
            Norm,
            Out,
            Mixture,
            prefix_stride,
            delta_stride,
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
        )


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
    _reduced_norm[layout.grid](
        source,
        full,
        weight,
        out,
        begin,
        *layout.arguments(source, full),
        eps,
        *layout.rank_coordinates,
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
    tile, warps = TILES[min(spec.sources, len(TILES) - 1)]
    _reduced_mix[layout.grid](
        prefix,
        source,
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
        spec.sources,
        spec.write,
        spec.mode,
        mixture is not None,
        tile,
        num_warps=warps,
    )


@_mix_rows.register_fake
def _fake_mix_rows(source, full, prefix, blocks, score, norm, out, mixture, eps, output_eps, handle):
    pass


@dataclass(frozen=True)
class RowGeometry:
    width: int
    sources: int = 0
    write: int = -1
    mode: int = 0
    blocks: int = 0
    taps: int = 0

    @property
    def threads(self):
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
        (control,) = reduction.parts
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

    def fused_mix(self, residual, source, *, score, norm, eps, output_eps, mixture, output):
        _mix_rows(
            source,
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
    """Private trial state, including each observable residual update and the complete reduced row."""

    def __init__(self, choice, eps):
        self.choice, self.eps = choice, eps
        spec, owner = choice.spec, choice.layout.owner
        device = owner.device
        self.source = torch.empty((4, spec.width), dtype=torch.bfloat16, device=device)
        self.full = torch.empty_like(self.source)
        columns = 224 if spec.width == 3584 else spec.width
        self.out = torch.empty((4, columns), dtype=torch.bfloat16, device=device)
        self.norm = torch.empty(columns, dtype=torch.bfloat16, device=device)
        self.begin = torch.tensor([owner.rank * 224], dtype=torch.int32, device=device)
        self.state = [self.source, self.norm]
        self.outputs = [self.full, self.out]
        if spec.width == 7168:
            self.prefix = torch.empty_like(self.source)
            self.blocks = torch.empty((4, spec.blocks, 7168), dtype=torch.bfloat16, device=device)
            self.score = torch.empty(7168, dtype=torch.float32, device=device)
            self.state.extend((self.prefix, self.blocks, self.score))
            self.outputs.extend((self.prefix, self.blocks))
            self.tap = None
            if spec.taps:
                self.tap = torch.empty((4, spec.taps * 7168), dtype=torch.bfloat16, device=device)[:, :7168]
                self.outputs.append(self.tap)
        self.patterns = []
        for index in range(4):
            generator = torch.Generator().manual_seed(307 + index)
            partials = torch.randn((16, 4, spec.width), generator=generator).to(torch.bfloat16)
            if index == 1:
                partials[:, :, ::3] = torch.tensor(-0.0, dtype=torch.bfloat16)
            if index == 2:
                partials[1::2] = -partials[::2]
            if index == 3:
                partials[::4].mul_(32)
            pattern = [partials[owner.rank].to(device)]
            pattern += [
                torch.randn(value.shape, generator=generator).to(device=device, dtype=value.dtype)
                for value in self.state[1:]
            ]
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
        self.norm, self.begin = fixture.norm, fixture.begin
        if self.choice.spec.width == 7168:
            self.prefix, self.blocks, self.score, self.tap = (
                fixture.prefix,
                fixture.blocks,
                fixture.score,
                fixture.tap,
            )

    def __call__(self):
        choice, spec = self.choice, self.choice.spec
        if self.fused:
            if spec.width == 3584:
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
            choice.sum(self.source, self.full)
            if spec.width == 3584:
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


def hidden_row_choices(catalog, layers, block_size, auxiliary_layers, blocks):
    """Bind the consumer of each attention and MLP reduction, including the final model norm."""
    pairs = []
    for index, (_, _, _, reduction) in enumerate(layers):
        committed = -(-index // block_size)
        boundary = index % block_size == 0
        after_attention = RowGeometry(
            7168, sources=committed + boundary, mode=REPLACE if boundary else ADD, blocks=blocks
        )
        following = index + 1
        after_mlp = RowGeometry(
            7168,
            sources=-(-following // block_size),
            write=following // block_size if following < len(layers) and following % block_size == 0 else -1,
            mode=ADD,
            blocks=blocks,
            taps=len(auxiliary_layers) if following in auxiliary_layers else 0,
        )
        pairs.append((catalog.register(reduction, after_attention), catalog.register(reduction, after_mlp)))
    return tuple(pairs)


class RowReductionTuning:
    """Measure complete four-row compositions after ordinary all-reduce plans are bound."""

    def __init__(self, coordinator, device, eps, cache_root):
        self.coordinator, self.device, self.eps, self.cache_root = coordinator, device, float(eps), cache_root
        self.choices = {}

    def register(self, reduction, geometry):
        if len(reduction.parts) != 1:
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
            _BINDINGS[target].layout = prepared_rows(plan, rows=4, width=geometry.width, threads=geometry.threads)
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
            raise ValueError("Kimi ranks declared different four-row normalization geometries")
        groups = {}
        for i, (geometry, target) in enumerate(entries):
            # Identical control programs on every participant can share one measurement.
            key = tuple(rank[i][1] for rank in ranks)
            if None not in key:
                groups.setdefault((geometry, key), []).append(target)
        return groups.items()

    def tune(self, session):
        if not session.autotune or session.autotune_effort == "off":
            return []
        report = []
        for (geometry, controls), targets_for_plan in self.groups():
            # ForwardCache keys compiled templates by name and graph signature.
            # Distinct reductions and epilogues need distinct persistent templates.
            identity = sha256(json.dumps((vars(geometry), controls), sort_keys=True).encode()).hexdigest()
            names = {kind: f"kimi-row-{geometry.width}-{kind}-{identity}" for kind in ("separate", "fused")}
            representative = targets_for_plan[0]
            fixture = _ReductionFixture(representative, self.eps)
            fixture.produce(0)
            forwards = {names[kind]: _PreparedTrial(fixture, kind == "fused") for kind in names}
            graphs, compiled = {}, {}
            try:
                graphs, compiled = capture_forwards(
                    forwards,
                    {n: GraphSignature("decode", 1, 4) for n in forwards},
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
                function = _reduced_norm if geometry.width == 3584 else _reduced_mix
                with session._tuning_clocks():
                    result = race_compositions(
                        candidates,
                        operation=f"kimi-row-normalization-{geometry.width}",
                        geometry=vars(geometry),
                        numerics=dict(epsilon=self.eps, kernel=function.cache_key),
                        workload=dict(version=1, seeds=(307, 308, 309), rows=4),
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
