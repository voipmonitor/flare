# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Identify compiled helpers by host architecture without invoking a process."""

from functools import lru_cache


@lru_cache
def platform_key() -> str:
    from platform import machine, system
    from struct import calcsize

    return ",".join([machine(), system(), f"{calcsize('P') * 8}bit"])


def replacements():
    from triton.runtime import build

    return ((build, "platform_key", platform_key),)


def initialize_b12x_worker(*args):
    from b12x._lib import compile_pool

    from . import apply_triton

    # A spawn entry module may install all patches before the pool invokes its
    # initializer. Resolve the preserved entrypoint in that case, rather than
    # calling this wrapper through the dependency's replaced module attribute.
    initializer = getattr(compile_pool, "_flare_unwrapped_initialize_worker", compile_pool._initialize_worker)
    if initializer is initialize_b12x_worker:
        raise RuntimeError("b12x compiler worker lacks its unwrapped initializer")
    apply_triton()
    initializer(*args)
