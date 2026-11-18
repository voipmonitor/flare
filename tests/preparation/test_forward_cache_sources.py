# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Forward caches track execution sources independently of diagnostic orchestration."""

from types import SimpleNamespace

import flashinfer

from ignition.preparation import aot


def test_diagnostic_edits_preserve_forward_cache_but_execution_edits_invalidate(tmp_path, monkeypatch):
    package = tmp_path / "ignition"
    preparation = package / "preparation"
    preparation.mkdir(parents=True)
    backend = preparation / "aot.py"
    backend.write_text("backend = 1\n")
    progress = package / "progress.py"
    progress.write_text("splash = 'sun'\n")
    entrypoint = package / "__main__.py"
    entrypoint.write_text("port = 8000\n")
    publication = preparation / "package_file.py"
    publication.write_text("threads = 1\n")
    benchmark = package / "bench" / "resident.py"
    benchmark.parent.mkdir()
    benchmark.write_text("event = 'resident_result'\n")
    recipe = package / "recipes" / "example.py"
    recipe.parent.mkdir()
    recipe.write_text("projection = 1\n")
    kernels = tmp_path / "flashinfer"
    kernels.mkdir()
    operator = kernels / "operator.py"
    operator.write_text("kernel = 1\n")
    monkeypatch.setattr(aot, "__file__", str(backend))
    monkeypatch.setattr(flashinfer, "__file__", str(kernels / "__init__.py"))
    monkeypatch.setattr(aot.torch.cuda, "get_device_properties", lambda device:
        SimpleNamespace(name="SM120", major=12, minor=0, multi_processor_count=96))

    def root():
        return aot.ForwardCache(namespace={"model": "example"}, device="cuda:0").root

    original = root()
    progress.write_text("splash = 'solar flare'\n")
    entrypoint.write_text("port = 8001\n")
    publication.write_text("threads = 16\n")
    benchmark.write_text("event = 'resident_error'\n")
    assert root() == original
    recipe.write_text("projection = 2\n")
    changed_recipe = root()
    assert changed_recipe != original
    backend.write_text("backend = 2\n")
    changed = root()
    assert changed != changed_recipe
    operator.write_text("kernel = 2\n")
    assert root() != changed
