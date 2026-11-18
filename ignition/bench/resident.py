# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Repeat diagnostic rounds and capture source-defined variants over resident weights."""

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from time import monotonic
from types import ModuleType

import torch


class ModelView:
    """Keep diagnostic graph selection separate from the prepared native bundle."""

    def __init__(self, model):
        self.base = model
        self.graphs = dict(model.graphs)
        self.benchmark_rounds = dict(getattr(model, "benchmark_rounds", None) or {})

    def __getattr__(self, name):
        return getattr(self.base, name)


@dataclass
class RoundVariant:
    """A complete round with the same input, completion, and persistent-state contract.

    Builders must keep the baseline objects and weight tensors unchanged. Retain
    extra prepared resources in ``owners``; their ``close`` methods run only after
    variant replay and graph destruction have drained.
    """

    forward: object
    owners: tuple = ()
    graph: object = None


def resolve_command(value):
    if not isinstance(value, dict):
        raise ValueError("resident command must be a JSON object")
    command = dict(value)
    operation = command.get("operation")
    allowed = {
        "measure": {"operation", "label"}, "baseline": {"operation"},
        "variant": {"operation"},
        "recapture": {"operation", "graph", "module", "module_sha256", "parameters"},
        "status": {"operation"}, "quit": {"operation"},
        "compare": {"operation"},
    }
    if operation not in allowed or set(command) - allowed[operation]:
        raise ValueError("unknown resident operation or command fields")
    if operation == "measure" and not re.fullmatch(r"[a-zA-Z0-9_-]+", command.get("label", "")):
        raise ValueError("measure requires an alphanumeric label (hyphens and underscores allowed)")
    if operation == "recapture":
        if not isinstance(command.get("graph"), str) or not command["graph"]:
            raise ValueError("recapture requires a graph name")
        if not isinstance(command.get("parameters", {}), dict):
            raise ValueError("variant parameters must be a JSON object")
        module = command.get("module")
        if module is not None:
            if not isinstance(module, str) or not all(part.isidentifier() for part in module.split(".")):
                raise ValueError("variant module must be an importable Python module name")
            spec = importlib.util.find_spec(module)
            if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
                raise ValueError("variant builder must have a Python source file")
            digest = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
            if command.get("module_sha256", digest) != digest:
                raise ValueError("variant source differs from the recorded module_sha256")
            command["module_sha256"] = digest
        elif command.get("parameters") or "module_sha256" in command:
            raise ValueError("variant parameters and source digest require a module")
    return command


def load_variant(command):
    """Execute exactly the recorded source; never reuse an unversioned live module."""
    checked = resolve_command(command)
    name, digest = checked["module"], checked["module_sha256"]
    previous = sys.modules.get(name)
    if previous is not None:
        if getattr(previous, "_resident_source_digest", None) != digest:
            raise ValueError("loaded variant differs from recorded source; use a distinct module name for each revision")
        return previous
    spec = importlib.util.find_spec(name)
    source = Path(spec.origin).read_bytes()
    if hashlib.sha256(source).hexdigest() != digest:
        raise ValueError("variant source changed before loading")
    module = ModuleType(name)
    module.__file__, module.__spec__ = spec.origin, spec
    module.__package__ = name.rpartition(".")[0]
    module._resident_source_digest = digest
    sys.modules[name] = module
    try:
        exec(compile(source, spec.origin, "exec"), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


class CapturedVariant:
    @torch.no_grad()
    def __init__(self, runner, command):
        from ignition.preparation.capture import capture_forwards
        from .executor import StateSnapshot

        started = monotonic()
        model, name = runner.model.base, command["graph"]
        if not model.benchmark_capture or name not in model.benchmark_rounds:
            raise ValueError(f"graph {name!r} has no retained round preparation")
        self.graphs, self.compiled, self.forwards, self.owners = {}, {}, {}, ()
        self.pool = None
        self.device = runner.device
        options = dict(model.benchmark_capture)
        bound = options.pop("compiled")
        snapshot = StateSnapshot(model.benchmark_state, runner.stream)
        forward = model.benchmark_rounds[name]
        replacement_graph = None
        try:
            if command.get("module") is not None:
                checked = resolve_command(command)
                module = load_variant(checked)
                variant = module.build(model, name, dict(checked.get("parameters", {})))
                if not isinstance(variant, RoundVariant):
                    raise TypeError("variant build must return RoundVariant")
                forward, self.owners = variant.forward, tuple(variant.owners)
                replacement_graph = variant.graph
                function, compile_decoder = forward, True
            else:
                function, compile_decoder = bound.get(name, forward), False
            if not callable(forward) or not hasattr(forward, "operations"):
                raise TypeError("variant must retain the complete speculative round interface")
            self.forwards = {name: forward}
            # Separate graph-local storage permits baseline/variant alternation.
            if replacement_graph is not None:
                self.graphs = {name: replacement_graph}
                compile_decoder = False
            else:
                self.pool = torch.cuda.graph_pool_handle()
                self.graphs, self.compiled = capture_forwards(
                    {name: function}, {name: model.signatures[name]}, device=runner.device,
                    stream=runner.stream, pool=self.pool, compile_decoder=compile_decoder,
                    before_run=lambda _: snapshot.restore(runner.stream), report_catalog=False,
                    retain_graphs=True, **options)
            self.report = dict(seconds=monotonic() - started, graph=name,
                               compiled_forward=compile_decoder,
                               graph_transform=replacement_graph is not None,
                               cache_hits={key: value.cache_hit for key, value in self.compiled.items()},
                               module=command.get("module"), module_sha256=command.get("module_sha256"),
                               parameters=command.get("parameters", {}))
        except BaseException:
            self.close()
            raise
        finally:
            snapshot.restore(runner.stream)
            runner.stream.synchronize()

    def close(self):
        import torch

        torch.cuda.synchronize(self.device)
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.compiled.clear()
        self.forwards.clear()
        self.pool = None
        for owner in reversed(self.owners):
            owner.close()
        self.owners = ()


def interact(prepared, tokenizer, *, scenarios, settings, directory, details):
    from .executor import BenchmarkExecutor
    from .runner import run_benchmark

    directory = Path(directory)
    executor = BenchmarkExecutor(prepared)
    active = {"operation": "baseline"}
    captured_command = None
    last = None
    failed = False

    def record(command):
        with (directory / "commands.jsonl").open("a") as file:
            file.write(json.dumps(command, sort_keys=True, allow_nan=False) + "\n")

    def publish(reply):
        with (directory / "events.jsonl").open("a") as file:
            file.write(json.dumps(reply, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(reply, allow_nan=False), flush=True)

    def execute(command):
        nonlocal active, last, captured_command
        operation = command["operation"]
        started = monotonic()
        if operation == "measure":
            output = directory / (command["label"] + ".json")
            with output.open("x"):
                pass

            def save(result):
                result.update(details)
                result["resident_variant"] = active
                result["wall_seconds"] = monotonic() - started
                output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

            last = run_benchmark(prepared, tokenizer, scenarios=scenarios, warmup=settings.warmup,
                                 repetitions=settings.repetitions, seeds=settings.seeds,
                                 progress=save, executor=executor)
            save(last)
            return dict(output=str(output), aggregate_ms=last["aggregate_ms"])
        if operation == "recapture":
            captured_command = None
            reports = executor._exchange(command)
            executor._exchange({"operation": "select_variant"})
            active = captured_command = command
            return dict(ranks=reports)
        if operation == "variant":
            if captured_command is None:
                raise RuntimeError("no successfully captured variant is available")
            executor._exchange({"operation": "select_variant"})
            active = captured_command
            return {}
        if operation == "baseline":
            executor._exchange(command)
            active = command
            return {}
        if operation == "compare":
            return compare(executor, tokenizer, scenarios, settings.seeds)
        if operation == "status":
            return dict(ranks=executor._exchange(command))
        return {}

    try:
        status = execute({"operation": "status"})
        (directory / "preparation.json").write_text(json.dumps({**details, **status}, indent=2) + "\n")
        from ignition.preparation.inductor_tuning import write_configurations

        write_configurations(executor.model.benchmark_capture["compiled"], executor.device,
                             directory / "inductor.json")
        first = {"operation": "measure", "label": "baseline"}
        result = execute(first)
        print(json.dumps(dict(event="resident_ready", pids=[rank["pid"] for rank in status["ranks"]],
                              **result)), flush=True)
        for line in sys.stdin:
            if not line.strip():
                continue
            started = monotonic()
            command = None
            try:
                command = resolve_command(json.loads(line))
                record(command)
                if command["operation"] == "quit":
                    break
                if failed and command["operation"] in ("measure", "compare"):
                    raise RuntimeError("a previous command failed; explicitly select baseline or recapture before measuring")
                result = execute(command)
                if command["operation"] in ("baseline", "recapture", "variant"):
                    failed = False
                reply = dict(event="resident_result", command=command,
                             wall_seconds=monotonic() - started, result=result)
            except Exception as error:
                # CUDA faults can also fail the drain; persist the triggering
                # operation before recovery or cleanup can raise another error.
                publish(dict(event="resident_error", command=command, input=line.strip(),
                             wall_seconds=monotonic() - started,
                             error=f"{type(error).__name__}: {error}"))
                # A returned rank error can be recoverable, such as a compiler
                # guard rejection. Reuse requires every rank to confirm its drain
                # and restore; a CUDA/context failure here terminates the session.
                executor._exchange(dict(operation="reset"))
                failed = True
                active = {"operation": "baseline"}
                continue
            publish(reply)
    finally:
        executor.close()
    return last


def compare(executor, tokenizer, scenarios, seeds):
    """Compare every rank's complete declared state from identical inputs and RNG."""
    from copy import deepcopy
    from ignition.direct.scheduler import Layout
    from .runner import _check_metadata, _check_request_state, _complete, _message, _prefill, _round

    model = executor.model
    layout = Layout.of(model)
    generation = int(model.benchmark_requests.generation.max().item())
    results = []
    for case in scenarios:
        for seed in seeds:
            executor._exchange(dict(operation="baseline"))
            generation += 1
            baseline = _prefill(model, executor, layout, case, tokenizer, seed, generation)
            executor.snapshot()
            try:
                for candidate in (False, True):
                    executor.restore()
                    if candidate:
                        executor._exchange(dict(operation="select_variant"))
                    scheduler = deepcopy(baseline, {id(layout): layout})
                    step = _round(scheduler, case)
                    elapsed = executor.run(_message(scheduler, step))
                    records = _complete(model, executor, scheduler, step, elapsed)
                    _check_request_state(model, step, records)
                    _check_metadata(model, step, case)
                    if candidate:
                        ranks = executor._exchange(dict(operation="compare_reference"))
                        results.append(dict(context=case.context_tokens, seed=seed,
                                            graph=step.graph, ranks=ranks))
                    else:
                        executor._exchange(dict(operation="save_reference"))
            finally:
                executor.restore()
                executor.release_snapshot()
                executor._exchange(dict(operation="release_reference"))
    return dict(equal=all(rank["equal"] for result in results for rank in result["ranks"]),
                comparison="bytewise complete declared state on every rank", cases=results)
