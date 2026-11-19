#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")/.." && pwd)
flare=$(realpath "$1")
provider=$(realpath "$2")
work=$(realpath "$3")
output=$(realpath -m "$4")
publisher=$(realpath "$5")
python=$(command -v "${PYTHON:-python3}")
support="$flare/tools/kimi31_dspark_release"
export PYTHONPATH="$support:$support/metadata:$flare:$provider:$provider/flashinfer/experimental/b12x/_compat"
# The signed program keeps barrier generations across launches.
export B12X_W4A16_SMALL_M_HOST_BARRIER_RESET=0
mkdir "$work/expert-images"
for stage in fc1 activation fc2; do
    export B12X_COMPILE_CACHE_DIR="$work/expert-cache/$stage"
    "$python" "$recipe/experts/compile.py" "$stage"
    mapfile -t objects < <(find "$B12X_COMPILE_CACHE_DIR" -type f -name '*.o')
    if [[ ${#objects[@]} != 1 ]]; then
        echo "Expected one compiled object for $stage" >&2
        exit 1
    fi
    "$python" "$recipe/experts/extract_object.py" "${objects[0]}" "$work/expert-images/$stage.cubin"
done
"$python" "$recipe/experts/stage_split.py" "$work/cooperative-core" "$work/transport-core" "$work/expert-images" "$work/expert-inputs"
"$publisher" "$work/transport-core" "$work/expert-inputs" "$output" "$recipe/signatures" experts
"$python" "$recipe/experts/audit_split.py" "$work/transport-core" "$output" > "$work/expert-bindings.json"
