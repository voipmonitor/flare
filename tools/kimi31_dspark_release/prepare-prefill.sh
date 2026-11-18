#!/usr/bin/env bash
# Internal offline reconstruction of the two sparse prefill execution stages.
set -euo pipefail
tool=$(cd "$(dirname "$0")" && pwd)
repo=$(realpath "$tool/../..")
parent=$(realpath "$1")
build=$(realpath "$2")
output=$(realpath -m "$3")
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
export PYTHONPATH="$tool:$tool/metadata:$repo"
nvcc=/usr/local/cuda-13.2/bin/nvcc
options=(-O3 -std=c++17 -arch=sm_120a --frandom-seed=913 -ccbin /usr/bin/g++-13 --cubin)
# NVCC hashes the source path into private CUB symbol names. Recreate the
# captured compilation path inside an isolated mount namespace.
source_root=/data/kimi31-dspark-final-mxfp4-20261115
source_dir="$source_root/frank1-prefill-recovery/source"
bwrap --ro-bind / / --proc /proc --dev /dev --tmpfs /tmp --tmpfs /data \
    --dir "$source_root" --dir "$source_root/frank1-prefill-recovery" \
    --ro-bind "$tool" "$source_dir" --bind "$build" /tmp/output --chdir / \
    /bin/bash -c 'src=$1; shift; exec "$@" < "$src"' _ "$source_dir/prefill_records.cu" \
    "$nvcc" "${options[@]}" -x cu /dev/stdin -o /tmp/output/records.cubin
"$nvcc" "${options[@]}" "$tool/shard_candidates.cu" -o "$build/candidates.cubin"
python "$tool/stage_records.py" "$parent" "$build/records.cubin" "$build/record-inputs"
cargo build --release --locked --manifest-path "$repo/tools/kimi31_pcie_package/Cargo.toml" --target-dir "$build/publisher"
publisher="$build/publisher/release/kimi-pcie-package"
"$publisher" "$parent" "$build/record-inputs" "$build/records-core" "$tool" "$tool/signatures/record-cache.pb"
python "$tool/stage_candidates.py" "$build/records-core" "$build/candidates.cubin" "$build/row-inputs"
"$publisher" "$build/records-core" "$build/row-inputs" "$output" "$tool" "$tool/signatures/query-rows.pb"
python "$tool/compare_signed.py" "$parent" "$output" "$build/retained-graphs.json"
(cd "$output" && sha256sum --check "$tool/prefill.sha256")
