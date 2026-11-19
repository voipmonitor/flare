#!/usr/bin/env bash
# Rebuild the sparse-attention exchange and the two reductions that consume peer data.
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
capture_flare=$(realpath "$1")
work=$(realpath "$2")
publisher=$(realpath "$3")
python=$(command -v "${PYTHON:-python3}")
venv=$("$python" -c 'import sys; print(sys.prefix)')
tools="$capture_flare/tools/kimi31_dspark_release"
export PYTHONPATH="$tools:$tools/metadata:$capture_flare"

# NVCC and Triton include these source paths in symbols and debug records.
# The mount namespaces reproduce those paths on any build host.
canonical=/data/kimi31-dspark-final-mxfp4-20261115
python_prefix=/data/kimi31-support/flare-main-k31-performance/.venv
"$python" "$recipe/attention/extract_queries.py" "$work/prefill" "$work/gather"
bwrap --ro-bind / / --proc /proc --dev /dev --tmpfs /tmp --tmpfs /data \
    --dir "$canonical" --dir "$canonical/release-and-decode" --dir "$canonical/release-and-decode/collectives" \
    --ro-bind "$recipe/attention" "$canonical/release-and-decode/collectives/src" \
    --bind "$work" /tmp/work --chdir "$canonical" \
    /usr/local/cuda-13.2/bin/nvcc -O3 -std=c++17 -arch=sm_120a --frandom-seed=21803 -ccbin /usr/bin/g++-13 \
    --cubin release-and-decode/collectives/src/single_gather.cu -o /tmp/work/attention.cubin
"$python" "$recipe/attention/stage_decode.py" "$work/prefill" "$work/attention.cubin" "$work/gather" \
    "$recipe/attention/gather-policy.json" "$work/attention-inputs"
"$publisher" "$work/prefill" "$work/attention-inputs" "$work/attention-core" "$recipe/signatures" attention

mkdir "$work/norm"
bwrap --ro-bind / / --proc /proc --dev /dev --tmpfs /tmp --tmpfs /data \
    --dir "$canonical" --ro-bind "$capture_flare" "$canonical/flare" \
    --dir /data/kimi31-support --dir /data/kimi31-support/flare-main-k31-performance \
    --ro-bind "$venv" "$python_prefix" --ro-bind "$recipe/latent_norm" /tmp/norm-src --bind "$work" /tmp/work \
    --setenv PYTHONPATH "$canonical/flare/tools/kimi31_dspark_release:$canonical/flare/tools/kimi31_dspark_release/metadata:$canonical/flare" \
    --setenv TRITON_CACHE_DIR /tmp/work/norm-triton --chdir "$canonical" \
    "$python_prefix/bin/python" /tmp/norm-src/offline.py /tmp/work/norm /tmp/work/attention-core
PYTHONPATH="$recipe/hidden_residual:$capture_flare/tools" "$python" "$recipe/latent_norm/fuse.py" "$work/norm" "$recipe/latent_norm/reduce.cu"
"$python" "$recipe/latent_norm/stage.py" "$work/attention-core" "$work/norm" "$work/norm-inputs"
"$publisher" "$work/attention-core" "$work/norm-inputs" "$work/norm-core" "$recipe/signatures" latent-norm

mkdir "$work/hidden"
bwrap --ro-bind / / --proc /proc --dev /dev --tmpfs /tmp --tmpfs /data \
    --dir "$canonical" --ro-bind "$capture_flare" "$canonical/flare" \
    --dir /data/kimi31-support --dir /data/kimi31-support/flare-main-k31-performance \
    --ro-bind "$venv" "$python_prefix" --bind "$work/hidden" /tmp/build --setenv PYTHONPATH "$canonical/flare" \
    --setenv TRITON_CACHE_DIR /tmp/build/triton --chdir "$canonical" "$python_prefix/bin/python" \
    flare/tools/kimi31_residual_offline.py flare/tools/kimi31_dspark_release/choices/model-config.json \
    flare/tools/kimi31_dspark_release/choices/dspark-config.json /tmp/build/residual-source
for kind in control corrected; do
    source_file="$capture_flare/tools/kimi31_row_reduction.cu"
    if [[ $kind == corrected ]]; then source_file="$recipe/hidden_residual/reduce.cu"; fi
    /usr/local/cuda-13.2/bin/nvcc -std=c++17 -O3 -arch=sm_120a --frandom-seed=65633 -ccbin /usr/bin/g++-13 \
        --ptx -rdc=true "$source_file" -o "$work/hidden/$kind.ptx"
    PYTHONPATH="$capture_flare/tools" "$python" "$recipe/hidden_residual/build_hidden_programs.py" \
        "$work/hidden/residual-source" "$work/hidden/$kind.ptx" "$work/hidden/$kind"
done
"$python" "$recipe/hidden_residual/stage.py" "$work/capture" "$work/norm-core" "$work/hidden" "$work/hidden-inputs"
"$publisher" "$work/norm-core" "$work/hidden-inputs" "$work/cooperative-core" "$recipe/signatures" hidden-residual
