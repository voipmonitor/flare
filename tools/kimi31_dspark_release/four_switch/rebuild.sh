#!/usr/bin/env bash
# Reconstruct the signed TP16 package from the serving snapshot and its public capture inputs.
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
flare=$(git -C "$recipe" rev-parse --show-toplevel)
snapshot=$(realpath "${1:?supply the downloaded Kimi DSpark snapshot}")
provider=$(realpath "${2:?supply the tagged FlashInfer checkout}")
work=$(realpath -m "${3:?supply an unused build directory on the snapshot filesystem}")
output=$(realpath -m "${4:?supply an unused output directory on the snapshot filesystem}")
python=$(command -v "${PYTHON:-python3}")
export PYTHON="$python" CUDA_VISIBLE_DEVICES='' CUDA_HOME=/usr/local/cuda-13.2 OMP_NUM_THREADS=1
support="$flare/tools/kimi31_dspark_release"
export PYTHONPATH="$support:$support/metadata:$flare:$provider:$provider/flashinfer/experimental/b12x/_compat"

mkdir "$work"
mkdir "$work/capture-flare" "$work/expert-provider"
# These two generator revisions are ancestors of the tagged source branches.
git -C "$flare" archive 821493f0771692737aec5b3534ca2c203196c880 | tar -xf - -C "$work/capture-flare"
git -C "$provider" archive d6cf5c01f53e542c16c24310d51fff9b929c4d96 | tar -xf - -C "$work/expert-provider"
cargo build --release --locked --manifest-path "$recipe/assembler/Cargo.toml" --target-dir "$work/native-tools"
publisher="$work/native-tools/release/kimi-pcie-package"

original="$snapshot/model.safetensors"
overrides=()
if [[ -f "$snapshot/rebuild/prefill32k-v1/model.safetensors" ]]; then
    original="$snapshot/rebuild/prefill32k-v1/model.safetensors"
    overrides=(--companion-overrides "$snapshot/rebuild/prefill32k-v1")
fi
"$python" "$support/parent_view.py" "$snapshot" "$original" f5ba77f5d9500637163cf301ed085000bbe8415c64f63a86e3597b9b2aa0617a \
    "$work/prefill" "${overrides[@]}"
"$python" "$support/parent_view.py" "$snapshot" "$snapshot/rebuild/capture/model.safetensors" \
    91887e7a69bee716c2c469d03168731e2038f0159171d5dd0d99be0c4b415176 "$work/capture" "${overrides[@]}"
"$recipe/cooperative.sh" "$work/capture-flare" "$work" "$publisher"

TRITON_DISABLE_LINE_INFO=1 TRITON_CACHE_DIR="$work/proposal-cache" "$python" "$recipe/proposals/compile.py" \
    "$work/cooperative-core" "$work/proposal-kernels"
"$python" "$recipe/proposals/stage.py" "$work/cooperative-core" "$work/proposal-kernels" "$work/proposal-inputs"
"$publisher" "$work/cooperative-core" "$work/proposal-inputs" "$work/proposal-core" "$recipe/signatures" proposals

mkdir "$work/transport"
"$python" "$recipe/transport/build_programs.py" "$work/proposal-core" "$provider" "$work/transport"
"$python" "$recipe/transport/stage.py" "$work/proposal-core" "$work/transport-inputs" \
    --vocabulary "$work/transport" --router "$work/transport/router"
"$publisher" "$work/proposal-core" "$work/transport-inputs" "$work/transport-core" "$recipe/signatures" transport
"$recipe/experts/rebuild.sh" "$flare" "$work/expert-provider" "$work" "$output" "$publisher"
printf '%s  %s\n' 4069909f9b374894dfc050b6e600827ba6fdabcfc13f316c9e7ef3e6131cfc08 "$output/model.safetensors" | sha256sum -c -
