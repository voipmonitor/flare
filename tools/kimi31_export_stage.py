"""Retain a local export stage so package assembly can resume without GPU preparation."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
from unittest.mock import patch


def save_stage(ranks, peers, destination, *, cache, no_weights, form):
    """Hardlink immutable staged payloads and retain the exact rank/peer descriptors."""
    ranks = [Path(rank).resolve() for rank in ranks]
    if not ranks or len({rank.parent for rank in ranks}) != 1:
        raise ValueError("export recovery requires rank sources in one staging directory")
    stage = ranks[0].parent
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(stage, destination / "stage", copy_function=os.link)
    record = dict(stage_path=str(stage), ranks=[rank.name for rank in ranks], peers=peers,
                  cache=str(cache), no_weights=no_weights, form=form)
    (destination / "assembly.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def restore_stage(checkpoint):
    """Restore the original owned stage path, preserving absolute payload references."""
    checkpoint = Path(checkpoint)
    record = json.loads((checkpoint / "assembly.json").read_text())
    stage = Path(record["stage_path"])
    # copytree rejects an existing destination; a recovery cannot overwrite a
    # live export or silently combine it with another set of rank descriptors.
    shutil.copytree(checkpoint / "stage", stage, copy_function=os.link)
    return [stage / name for name in record["ranks"]], record


@contextmanager
def retain_before_assembly(destination):
    from ignition.preparation import model_package
    from ignition.preparation.package_snapshot import _OPTIONS

    combine = model_package.combine

    def retained(path, ranks, peer_memory, **kwargs):
        settings = _OPTIONS.get()
        if settings is None:
            raise RuntimeError("export recovery requires configured package storage")
        save_stage(ranks, peer_memory, destination, cache=settings.cache,
                   no_weights=settings.no_weights, form=kwargs.get("form", "directory"))
        print(f"Retained package assembly stage: {destination}", flush=True)
        return combine(path, ranks, peer_memory, **kwargs)

    with patch.object(model_package, "combine", retained):
        yield


def main():
    import argparse
    from types import SimpleNamespace
    from ignition.cli.config import Configuration
    from ignition.cli.preparation import resolve_model
    from ignition.cli.serve import _publisher
    from ignition.preparation.package_file import combine
    from ignition.preparation.package_snapshot import export_options

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configuration = Configuration.load(args.config)
    model = resolve_model(configuration, args.model)
    publisher = _publisher(configuration, SimpleNamespace(official_source=None), model,
                           configuration.directory("hub_cache"))
    ranks, record = restore_stage(args.checkpoint)
    with export_options(record["cache"], no_weights=record["no_weights"]):
        result = combine(args.output, ranks, record["peers"], publisher=publisher, form=record["form"])
    print(json.dumps(vars(result), indent=2))


if __name__ == "__main__":
    main()
