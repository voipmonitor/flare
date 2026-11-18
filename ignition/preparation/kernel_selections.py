# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Require persisted b12x choices while allowing their programs to compile from source."""

from b12x.preparation._cache import SelectionCache


def record_selections(session, compiled, device, directory, coordinator):
    from pathlib import Path
    import shutil
    import torch.distributed as dist
    from .inductor_tuning import write_configurations

    root = coordinator is None or dist.get_rank() == coordinator.ranks[0]
    directory = Path(directory)
    if root:
        directory.mkdir(parents=True, exist_ok=False)
        cache = session._selection_cache()
        cache.flush()
        shutil.copyfile(cache.path, directory / cache.path.name)
    if coordinator is not None:
        coordinator.barrier("record kernel selections", coordinator.ranks)
    write_configurations(compiled, device, directory / "inductor.json", coordinator=coordinator)


class FixedSelectionCache(SelectionCache):
    """An immutable selection table: an unknown program/query cannot start a race.

    This deliberately differs from b12x ``cache_only``, which also requires
    compiled artifacts. Program compilation remains available when rebuilding
    from source with a complete selection table.
    """

    def __init__(self, root, identity):
        super().__init__(root, identity)
        if not self.path.is_file():
            raise FileNotFoundError(f"no fixed kernel selection table for this workload/device: {self.path}")

    def get(self, key):
        record = super().get(key)
        if record is None:
            raise LookupError(f"fixed kernel selection missing: table={self.path} query_program_key={key}; "
                              "automatic retuning is disabled")
        return record

    def save(self, *args, **kwargs):
        raise RuntimeError(f"fixed kernel selections cannot be updated: {self.path}")

    def reconcile(self, snapshots):
        snapshots = tuple(snapshots)
        if any(snapshot.records.to_dict() != self.records for snapshot in snapshots):
            raise ValueError("ranks must use identical fixed kernel selection tables")
        super().reconcile(snapshots)
