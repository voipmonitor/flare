#!/usr/bin/env bash
# Recreate decode fusion, prepared PCIe programs and both sparse prefill stages.
set -euo pipefail
tool=$(cd "$(dirname "$0")" && pwd)
repo=$(realpath "$tool/../..")
snapshot=$(realpath "${1:?supply the complete HF snapshot}")
build=$(realpath -m "${2:?supply a new build directory}")
output=$(realpath -m "${3:?supply a new output directory}")
original_provider=$(realpath "${4:?supply the FlashInfer capture checkout}")
release_provider=$(realpath "${5:?supply the FlashInfer release checkout}")
mkdir "$build"
test ! -e "$output"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 CUDA_HOME=/usr/local/cuda-13.2
export TRITON_CACHE_DIR="$build/triton"
export PYTHONPATH="$tool:$tool/metadata:$repo:$original_provider:$original_provider/flashinfer/experimental/b12x/_compat"
python "$tool/parent_view.py" "$snapshot" "$snapshot/rebuild/capture/model.safetensors" \
    91887e7a69bee716c2c469d03168731e2038f0159171d5dd0d99be0c4b415176 "$build/capture"
# Triton records absolute filenames in DWARF. Recreate both the recipe's
# and the Python environment's compile paths inside a private namespace.
captured_root=/data/kimi31-dspark-final-mxfp4-20261115
captured_venv=/data/kimi31-support/flare-main-k31-performance/.venv
venv=$(python -c 'import sys;print(sys.prefix)')
bwrap --ro-bind / / --proc /proc --dev /dev --tmpfs /tmp --tmpfs /data \
    --dir "$captured_root" --ro-bind "$repo" "$captured_root/flare" \
    --dir /data/kimi31-support --dir /data/kimi31-support/flare-main-k31-performance \
    --ro-bind "$venv" "$captured_venv" --bind "$build" /tmp/build \
    --setenv PYTHONPATH "$captured_root/flare" --setenv TRITON_CACHE_DIR /tmp/build/triton \
    --chdir "$captured_root" "$captured_venv/bin/python" flare/tools/kimi31_residual_offline.py \
    flare/tools/kimi31_dspark_release/choices/model-config.json \
    flare/tools/kimi31_dspark_release/choices/dspark-config.json /tmp/build/residual-source
"$CUDA_HOME/bin/nvcc" -std=c++17 -O3 -arch=sm_120a --frandom-seed=65633 -ccbin /usr/bin/g++-13 \
    --ptx -rdc=true "$repo/tools/kimi31_row_reduction.cu" -o "$build/reduce.ptx"
python "$repo/tools/kimi31_row_reduction_build.py" "$build/residual-source" "$build/reduce.ptx" "$build/residual-programs"
python -m tools.kimi31_native_row_resources "$build/capture" "$build/residual-programs" "$tool/metadata" "$build/residual-graphs"
python -m tools.kimi31_native_row_companions "$build/capture" "$build/residual-graphs" "$build/residual-programs" "$tool/metadata" "$build/residual-inputs"
cargo build --release --locked --manifest-path "$repo/tools/kimi31_row_package/Cargo.toml" --target-dir "$build/publisher"
"$build/publisher/release/kimi-row-package" "$build/capture" "$build/residual-inputs" "$build/fused-rows" "$tool" "$tool/signatures/fused-rows.pb"
ranks=()
for i in {0..15}; do ranks+=(--rank "$i"); done
for version in old new; do
    provider=$release_provider
    if [[ $version == old ]]; then provider=$original_provider; fi
    PYTHONPATH="$provider:$provider/flashinfer/experimental/b12x/_compat" B12X_COMPILE_CACHE_DIR="$build/provider-cache-$version" \
        python "$tool/compile_offline.py" "$tool/choices/native-specs.json" "$build/programs-$version" \
        "${ranks[@]}" --offline-abi "$tool/choices/pcie-$version-images.json"
done
python "$tool/stage_pcie.py" "$build/fused-rows" "$tool/choices/native-specs.json" "$build/programs-old" "$build/programs-new" "$build/pcie-inputs"
cargo build --release --locked --manifest-path "$repo/tools/kimi31_pcie_package/Cargo.toml" --target-dir "$build/publisher"
"$build/publisher/release/kimi-pcie-package" "$build/fused-rows" "$build/pcie-inputs" "$build/pcie-parent" "$tool" "$tool/signatures/pcie-parent.pb"
exec "$tool/prepare-prefill.sh" "$build/pcie-parent" "$build" "$output"
