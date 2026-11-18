#!/usr/bin/env bash
# Prepare the TP16/DCP16 Kimi K3.1 target and its seven-token DSpark draft.
set -euo pipefail
tools_dir=$(cd "$(dirname "$0")" && pwd)
flare_root=$(realpath "$tools_dir/../..")
target=$(realpath "${1:?supply the Kimi K3.1 Pro snapshot}")
draft=$(realpath "${2:?supply the Inferact Kimi K3.1 Pro DSpark snapshot}")
provider=$(realpath "${3:?supply the FlashInfer capture checkout}")
work=$(realpath -m "${4:?supply an empty capture directory on the model disk}")
shift 4
if [[ ! -f "$target/model.safetensors.index.json" || ! -f "$draft/model.safetensors" ]]; then
    echo "Expected a sharded target and a DSpark safetensors checkpoint." >&2
    exit 2
fi
mkdir "$work"
mkdir "$work/target"
for file in "$target"/*; do
    [[ -f "$file" ]] && ln -s "$file" "$work/target/"
done
ln -s "$draft" "$work/target/dspark"
python - "$work" <<'PY'
import json
import sys
from pathlib import Path

work = Path(sys.argv[1])
with (work / "preparation.toml").open("x") as config:
    config.write("[paths]\n")
    for name, folder in (("hub_cache", "hub"), ("kernel_cache", "kernels"), ("package_cache", "packages"),
                         ("export_staging", "staging"), ("prefix_cache", "prefix")):
        config.write(f"{name} = {json.dumps(str(work / folder))}\n")
    config.write('cuda = "/usr/local/cuda-13.2"\n[prefix_cache]\nmemory = "4G"\ndisk_limit = "0"\n[memory]\nreserve = "0M"\n')
PY
# The local FI record cache may write readiness data beside the fixed table.
cp -a "$tools_dir/selections/provider" "$work/selections"
export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13
export PYTHONPATH="$flare_root:$provider:$provider/flashinfer/experimental/b12x/_compat${PYTHONPATH:+:$PYTHONPATH}"
ulimit -n 1048576
ulimit -l unlimited
exec python "$flare_root/tools/kimi31_export_copy_fixed.py" export "$work/target" \
    --config "$work/preparation.toml" --gpus 0-15 --max-concurrency 1 --image-concurrency 1 \
    --prefill-chunk-size 2048 --mtp-draft-tokens 7 --speculative-token-capacity 32 \
    --speculative-graph-budget 16 --online-mxfp8 --online-mxfp8-scope all \
    --draft-head-quantization nvfp4 --markov-head-quantization fp8 --replicated-markov \
    --collective-tuning static --autotune-effort default --compile-workers 16 \
    --kernel-selections "$work/selections" --record-kernel-selections "$work/recorded" --verbose "$@"
