"""Resolve a complete compiler catalog without benchmark races, then record its choices.

Recorded matrix and generated-kernel choices take precedence. Unlisted matrix
queries use their unique ATen implementation; unlisted generated kernels use
their first generated launch configuration. Normal Flare selection recording
persists every resulting choice for subsequent strict source rebuilds.
"""

from contextlib import ExitStack, contextmanager
from functools import partial
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from ignition.preparation.compiler_choices import FIXED_OPTIONS, choice_key, select_choice
from ignition.preparation.inductor_tuning import (
    InductorSelections, _configuration_device, _install_configuration,
)


def compile_forward_fixed(seed_directory, *args, **kwargs):
    """Install the explicit policy inside Torch's Linux subprocess compiler."""
    from ignition.preparation import compile_pool

    compile_forward = getattr(compile_pool, "_flare_unwrapped_compile_forward", compile_pool._compile_forward)
    with compiler_policy(seed_directory):
        return compile_forward(*args, **kwargs)


@contextmanager
def compiler_policy(seed_directory):
    from torch._inductor.runtime import triton_heuristics
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from torch._inductor.select_algorithm import AlgorithmSelectorCache, ExternKernelCaller, create_inputs_key
    from ignition.preparation.compilation import _OPTIONS
    from ignition.preparation import compile_pool
    from ignition.preparation.aot import ForwardCache

    directory = Path(seed_directory)
    algorithms = json.loads((directory / "algorithms.json").read_text())
    configurations = json.loads((directory / "inductor.json").read_text())
    if algorithms.get("schema") != 1 or configurations.get("schema") != 2:
        raise ValueError("fixed compilation requires algorithm schema 1 and generated-kernel schema 2")
    original_register = InductorSelections.register
    original_compile = getattr(compile_pool, "_flare_unwrapped_compile_forward", compile_pool._compile_forward)
    original_cache = ForwardCache.__init__
    identity = hashlib.sha256(Path(__file__).read_bytes()
                              + (directory / "algorithms.json").read_bytes()
                              + (directory / "inductor.json").read_bytes()).hexdigest()

    def cache_init(self, *, namespace, **kwargs):
        # A policy and its seed table are part of the forward artifact identity.
        # Nested scopes preserve the innermost policy's explicit namespace.
        namespace = dict(namespace)
        namespace.setdefault("fixed_compiler_profile", identity)
        original_cache(self, namespace=namespace, **kwargs)

    def forbidden(*args, **kwargs):
        raise RuntimeError("fixed compiler profile attempted a benchmark")

    def first_configuration(configs, *args, **kwargs):
        if not configs:
            raise ValueError("fixed compiler profile requires a generated configuration")
        return configs[:1], None, dict(
            autotune_cache_state="only 1 config",
            only_config=triton_heuristics.triton_config_to_hashable(configs[0]),
        )

    def choose(self, name, choices, input_nodes, layout, *args, **kwargs):
        key = choice_key(name, create_inputs_key(input_nodes), torch.get_float32_matmul_precision())
        if key in algorithms["choices"]:
            selected = select_choice(choices, key, algorithms["choices"])
        else:
            external = [choice for choice in choices if isinstance(choice, ExternKernelCaller)]
            if len(external) != 1:
                raise ValueError(f"fixed matrix policy requires one ATen choice for {key}; found {len(external)}")
            selected, = external
        return selected.output_node(), selected

    def register(self, bound):
        if configurations["device"] != _configuration_device(self.device):
            raise ValueError("fixed compiler seed differs from the active device or toolchain")
        sources = original_register(self, bound)
        backend, graph, bindings = self.forwards[id(bound)]
        for identity, _, tuner in bindings.values():
            name = tuner.inductor_meta["kernel_name"]
            source = sources[name]
            key = self.configuration_keys[identity]
            selection = configurations["kernels"].get(key)
            if selection is None:
                available = tuner.configs or [result.config for result in tuner.compile_results]
                if not available:
                    raise RuntimeError(f"generated kernel {name} has no explicit launch configuration")
                selection = vars(available[0])
            _install_configuration(tuner, selection, source)
        backend._tuning_frozen = True
        return sources

    with ExitStack() as stack:
        stack.enter_context(patch.dict(_OPTIONS, FIXED_OPTIONS))
        stack.enter_context(patch.object(ForwardCache, "__init__", cache_init))
        stack.enter_context(patch.object(compile_pool, "_flare_unwrapped_compile_forward", original_compile, create=True))
        stack.enter_context(patch.object(compile_pool, "_compile_forward", partial(compile_forward_fixed, str(directory))))
        stack.enter_context(patch.object(AlgorithmSelectorCache, "__call__", choose))
        stack.enter_context(patch.object(AlgorithmSelectorCache, "autotune", forbidden))
        stack.enter_context(patch.object(CachingAutotuner, "bench", forbidden))
        stack.enter_context(patch.object(triton_heuristics, "check_autotune_cache", first_configuration))
        stack.enter_context(patch.object(triton_heuristics, "lookup_autotune_config", lambda *args, **kwargs: None))
        stack.enter_context(patch.object(InductorSelections, "register", register))
        yield
