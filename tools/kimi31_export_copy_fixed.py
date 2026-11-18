"""Run a reproducible Kimi export with fixed compiler choices in every spawned rank."""

from pathlib import Path
import sys


if __name__ == "__main__" and "--config" in sys.argv:
    from ignition import planning
    from ignition.cli import environment
    from ignition.cli.config import Configuration
    from ignition.cli.preparation import Host
    from ignition.recipes.kimi31.recipe import RECIPE

    configuration = Configuration.load(Path(sys.argv[sys.argv.index("--config") + 1]))
    host = Host.discover()
    indices = planning.parse_gpus(sys.argv[sys.argv.index("--gpus") + 1]) if "--gpus" in sys.argv else tuple(range(16))
    environment.apply(configuration, tuple(gpu for gpu in host.gpus if gpu.index in indices), RECIPE.b12x_environment)

from ignition import runtime_patches  # noqa: E402

runtime_patches.apply()

from tools.kimi31_fixed_compiler import compiler_policy  # noqa: E402


# Multiprocessing imports this entry module in rank and compiler workers. Each
# process must install the same policy before it prepares any graph.
_policy = compiler_policy(Path(__file__).resolve().parent / "kimi31_dspark_release" / "selections" / "compiler")
_policy.__enter__()


if __name__ == "__main__":
    from ignition.cli import main

    try:
        raise SystemExit(main())
    finally:
        _policy.__exit__(None, None, None)
