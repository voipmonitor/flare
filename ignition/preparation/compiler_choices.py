"""Record or replay Inductor algorithm choices before generated kernels exist."""

from contextlib import contextmanager
import ast
from functools import lru_cache
import json
from pathlib import Path
from threading import RLock

import torch

_LOCK = RLock()
FIXED_OPTIONS = dict(shape_padding=False, benchmark_epilogue_fusion=False,
                     pipeline_max_autotune_gemm=False, fx_graph_cache=False, fx_graph_remote_cache=False)


def choice_key(name, inputs, precision):
    return json.dumps([name, inputs, precision], separators=(",", ":"))


@lru_cache(maxsize=512)
def template_identity(path):
    from .inductor_tuning import configuration_key

    source = Path(path).read_text()
    tree = ast.parse(source)
    names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    devices = [keyword.value.value for node in ast.walk(tree)
               if isinstance(node, ast.Call) and ast.unparse(node.func) == "DeviceProperties"
               for keyword in node.keywords if keyword.arg == "index"]
    if len(names) != 1 or len(set(devices)) != 1:
        raise ValueError("fixed Triton templates require one kernel and one CUDA device")
    # Inductor's unrendered templates use a placeholder instead of the final symbol.
    source = source.replace("Placeholder.DESCRIPTIVE_NAME", names[0])
    return "triton:" + configuration_key(source, devices[0], names[0])


def algorithm_identity(choice):
    from torch._inductor.select_algorithm import TritonTemplateCaller

    if isinstance(choice, TritonTemplateCaller):
        return template_identity(choice.bmreq.module_path)
    return choice.hash_key()


def select_choice(choices, key, records):
    expected = records.get(key)
    if expected is None:
        raise LookupError(f"fixed compiler algorithm missing for {key}")
    matches = [choice for choice in choices if algorithm_identity(choice) == expected]
    if len(matches) != 1:
        raise LookupError(f"fixed compiler algorithm {expected} is not uniquely available for {key}")
    return matches[0]


@contextmanager
def algorithm_choices(directory=None, *, output=None):
    if directory is None and output is None:
        yield
        return
    from torch._inductor.select_algorithm import AlgorithmSelectorCache, create_inputs_key

    fixed = None
    if directory is not None:
        value = json.loads((Path(directory) / "algorithms.json").read_text())
        if value.get("schema") != 1:
            raise ValueError("fixed compiler algorithms require schema 1")
        fixed = value["choices"]
    records = {}
    with _LOCK:
        original = AlgorithmSelectorCache.__call__

        def choose(self, name, choices, input_nodes, layout, *args, **kwargs):
            key = choice_key(name, create_inputs_key(input_nodes), torch.get_float32_matmul_precision())
            if fixed is None:
                node, selected = original(self, name, choices, input_nodes, layout, *args, **kwargs)
                if selected is None:
                    raise RuntimeError("recorded compilation requires a completed algorithm choice")
            else:
                selected = select_choice(choices, key, fixed)
                from torch._inductor.select_algorithm import ExternKernelCaller, TritonTemplateCaller

                if not isinstance(selected, (ExternKernelCaller, TritonTemplateCaller)):
                    raise ValueError(f"fixed compilation does not implement {type(selected).__name__} workspace preparation")
                node = selected.output_node()
            selected_key = algorithm_identity(selected)
            if key in records and records[key] != selected_key:
                raise RuntimeError(f"one compiler query selected conflicting algorithms: {key}")
            records[key] = selected_key
            return node, selected

        AlgorithmSelectorCache.__call__ = choose
        try:
            yield
        finally:
            AlgorithmSelectorCache.__call__ = original
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(schema=1, choices=records), sort_keys=True, indent=2) + "\n")
