#!/usr/bin/env bash
# Rebuild the signed SM120 DSpark checkpoint from its distributed PCIe parent.
set -euo pipefail
tool=$(cd "$(dirname "$0")" && pwd)
repo=$(realpath "$tool/../..")
snapshot=$(realpath "${1:?supply the complete HF snapshot}")
build=$(realpath -m "${2:?supply a new build directory}")
output=$(realpath -m "${3:?supply a new output directory}")
mkdir "$build"
test ! -e "$output"
PYTHONPATH="$tool/metadata:$repo" python "$tool/parent_view.py" "$snapshot" \
    "$snapshot/rebuild/pcie-parent/model.safetensors" 545ee0e8515ae02a1b3f70da286b8885bb8b46942ad6c4a35a2bf1384146b7f5 "$build/pcie-parent"
exec "$tool/prepare-prefill.sh" "$build/pcie-parent" "$build" "$output"
