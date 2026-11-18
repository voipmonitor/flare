# SPDX-License-Identifier: Apache-2.0
import pytest

from tools.kimi31_dspark_release.dwarf_files import annotate


def test_replay_supplies_source_attributes_without_changing_executable_ptx():
    source = '\t.file 7 "/abs/source.py"\n\t.loc 7 1 0\n\tadd.rn.f32 %f3, %f1, %f2;\n'
    expected = source.replace('"/abs/source.py"', '"/abs/source.py", 22, 49')
    assert annotate(source, {"/abs/source.py": [22, 49]}) == expected


def test_missing_or_invalid_source_metadata_is_rejected():
    source = '.file 1 "/abs/source.py"\n'
    with pytest.raises(ValueError, match="missing recorded"):
        annotate(source, {})
    with pytest.raises(ValueError, match="invalid unsigned"):
        annotate(source, {"/abs/source.py": [-1, 25]})
