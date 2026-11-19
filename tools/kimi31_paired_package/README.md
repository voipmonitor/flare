# Replace a prepared verification graph over resident weights

The comparison CLI runs a signed TP16 control and accepts signed alternatives at drained request boundaries. Use this development workflow to change GPU kernels or graph dependencies while keeping the control’s weight bindings, allocations, peer mappings and KV state. Creating comparison packages requires development checkpoints and the matching development signing identity. Distribute the public verification profile with the signed inputs. The ordinary released Kimi checkpoint can be served with its published public profile.

## Prepare a reference bank and an executable view

Use two compatible signed Kimi DSpark TP16 development checkpoints in `control-core` and `candidate-core`. Put their matching `development-publisher.json` and your signing key `development-signing-key.pem` in a private `identity` directory. Run in a fresh working directory on their filesystem. The operations below prepare a shared reference bank and a four-row alternative.

```bash
set -e
git clone --depth 1 --branch kimi-k31-dspark-tp16-graph-comparison-v2 https://github.com/voipmonitor/flare.git flare-comparison
tools="$PWD/flare-comparison/tools/kimi31_paired_package"
cargo build --locked --release --manifest-path "$tools/Cargo.toml" --target-dir "$PWD/graph-tools"
cargo build --locked --release --manifest-path flare-comparison/flare-cli/Cargo.toml --target-dir "$PWD/graph-runtime"
packager="$PWD/graph-tools/release/kimi-paired-package"
cli="$PWD/graph-runtime/release/flare"
"$packager" --shared-reference-bank "$PWD/control-core" "$PWD/reference" "$PWD/identity" speculative1x8-tokens{2,3,4,5,6,7,8}
"$packager" --only "$PWD/control-core" "$PWD/candidate-core" "$PWD/paired" "$PWD/identity" speculative1x8-tokens4
core_id=$("$cli" inspect "$PWD/paired" --publisher-keys "$PWD/identity/development-publisher.json" --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["core"]["core_id"])')
python3 "$tools/thin_view.py" --packager "$packager" --reference "$PWD/reference" \
  --candidate "$PWD/paired" --core-id "$core_id" --publisher-dir "$PWD/identity" --output "$PWD/view" > view.json
```

The helper checks the signed operation set before staging and after reopening the result. The view contains the signed primary and the required executable companions. The model weights belong to the full reference. On a separate serving host, transfer the reference once, then transfer each view and its recorded core ID. A new executable view must use the same signed control. Keep the source commits, fixed compiler choices and checksums for each candidate.

## Run and select the alternative

The following starts a single-request server on sixteen available SM120 GPUs. `--benchmark-control` enables local graph transactions in the specified private directory. The ordinary CLI default has this option disabled. The shared reference bank initially reuses the permanent control handle for each idle alternative slot. After approval of your publisher in Flare’s trust prompt, requests can use either graph role.

```bash
mkdir -m 700 "$PWD/graph-control"
ulimit -Sn 65536
"$PWD/graph-runtime/release/flare" run "$PWD/reference" --offline \
  --publisher-keys "$PWD/identity/development-publisher.json" \
  --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --max-concurrency 1 --mtp-draft-tokens 7 --kv-cache-limit memory \
  --embedding-table-storage ram --host 127.0.0.1 --port 8000 \
  --served-model-name Kimi-K3.1-Pro-DSpark --benchmark-control "$PWD/graph-control"
```

In a second shell in the same working directory, select the verified candidate and send a request.

```bash
tools="$PWD/flare-comparison/tools/kimi31_paired_package"
core_id=$(python3 -c 'import json; print(json.load(open("view.json"))["core_id"])')
python3 "$tools/live_commands.py" --control "$PWD/graph-control" load "$core_id" "$PWD/view"
python3 "$tools/live_commands.py" --control "$PWD/graph-control" role candidate
curl -sS http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Kimi-K3.1-Pro-DSpark","messages":[{"role":"user","content":"What is 2 + 2?"}],"tool_choice":"none","max_tokens":64}'
python3 "$tools/live_commands.py" --control "$PWD/graph-control" status
python3 "$tools/live_commands.py" --control "$PWD/graph-control" role control
```

Import authenticates the source before changing the active candidate, drains all ranks, selects the control, then constructs the requested graphs. A signed control mismatch rejects the transaction before retiring a working candidate. A CUDA allocation failure during construction leaves the permanent control usable after confirmed cleanup. The returned receipt reports per-rank launch counters and free memory.

On sixteen 96 GB RTX PRO 6000 Blackwell Workstation GPUs at zero manual reserve, a whole four-row Kimi graph with six fresh kernels per rank loaded in 4.10 seconds at 2,304 KV pages and 116 snapshot slots. Installing seven alternatives at that capacity exhausted CUDA graph memory. The same process then completed a 128-token request on the control. A bank with independent handles for all seven shapes used 2,247 pages and 113 slots and loaded all seven in 7.41 seconds. The reference’s initial weight load took 140.1 seconds.
