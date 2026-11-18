# Rebuild the signed Kimi K3.1 DSpark checkpoint

From the repository root, run the following on a Linux x86-64 build host. Pass a complete local Hugging Face snapshot that contains the model primary, its weight and execution companions, and the two input primaries under `rebuild/`. The two output directories must be new. Keep the build and output on the same filesystem. Using the model filesystem for all three paths allows the publisher to hardlink the 1.4 TB of inherited payloads.

```bash
tools/kimi31_dspark_release/rebuild.sh "$PWD/../kimi-hf" "$PWD/../kimi-build" "$PWD/../kimi-rebuilt"
```

Install Python 3.12 with Protobuf 7.36.2, Rust 1.95 with Cargo, CUDA 13.2.86 at `/usr/local/cuda-13.2`, GCC 13, Protobuf compiler 3.21, Bubblewrap, Zstandard, OpenSSL development headers and the normal Flare native build dependencies. Cargo may fetch the crates pinned by the included lockfiles. The script hides CUDA devices. NVCC and the graph serializer operate on source code and signed allocation metadata. The first stage builds the selected-record cache and the second distributes complete index-selection rows. At each stage the publisher verifies the source and the reconstructed package. The signing witnesses contain the original public signatures and their body indices. Verification requires the computed bodies, publisher and package ID to match.

The release contains 1,637 files. Its PCIe parent accounts for 1,602 inherited companions. The script checks SHA-256 for the 34 regenerated execution companions and the signed primary, then verifies the graphs and allocation bindings for all sixteen ranks. The signed primary digest is `f5ba77f5d9500637163cf301ed085000bbe8415c64f63a86e3597b9b2aa0617a`. [Output hashes](prefill.sha256)

## Rebuild decode fusion and the PCIe transport programs as well

For the signed checkpoint tagged `kimi-k31-dspark-tp16-prefill32k-v1`, use the Flare build recipe tagged `kimi-k31-dspark-tp16-prefill32k-rebuild-v2`. This route starts at the signed Ignition capture. It additionally requires the [Flare preparation environment](../../docs/model-preparation.md), including PyTorch 2.14.1+cu132, Triton 3.8.0 and NVIDIA Cutlass DSL 4.8.1. Run the commands below from the directory containing the downloaded `kimi-hf` model folder. They place both provider revisions alongside Flare. Their fixed source and ABI selections let the compiler regenerate all per-rank programs from explicit geometry.

```bash
git clone --branch kimi-k31-dspark-tp16-prefill32k-rebuild-v2 https://github.com/voipmonitor/flare.git flare-replay
cd flare-replay
git clone --branch kimi-k31-dspark-tp16-capture-v1 https://github.com/voipmonitor/flashinfer.git ../flashinfer-capture
git clone --branch kimi-k31-dspark-tp16-prefill32k-v1 https://github.com/voipmonitor/flashinfer.git ../flashinfer-pcie
tools/kimi31_dspark_release/rebuild-from-capture.sh \
  "$PWD/../kimi-hf" "$PWD/../kimi-build-full" "$PWD/../kimi-rebuilt-full" \
  "$PWD/../flashinfer-capture" "$PWD/../flashinfer-pcie"
```

| Prepared artifact | Change | SHA-256 of `model.safetensors` |
|---|---|---|
| Signed Ignition capture | TP16/DCP16, C1, seven-token DSpark, 2,048-token prefill | `91887e7a69bee716c2c469d03168731e2038f0159171d5dd0d99be0c4b415176` |
| Four-row target verification | 186 residual reductions share their results with the following residual operation | `01a99e80706024f11a7052d62a518cb4b6dd911a1fbef9eb0a30dbb8da6d69de` |
| Prepared PCIe transports | Rank-specific ring and island kernels | `545ee0e8515ae02a1b3f70da286b8885bb8b46942ad6c4a35a2bf1384146b7f5` |
| Sparse record cache | Reuse 144 MiB for the live logical records of one layer | `f9577f7eab92132dd28957bb452c21a3dbb4b65eb74f409ced73171fd16b78b0` |
| Query-row selection | Each GPU folds eight complete candidate rows, then returns the selected IDs | `f5ba77f5d9500637163cf301ed085000bbe8415c64f63a86e3597b9b2aa0617a` |

The CUDA assembler normally looks up the timestamp and byte length of each source in a bare PTX `.file` directive when it builds the DWARF line table. A missing file yields zero for both fields. The recipe supplies the two recorded source entries explicitly so the hidden-row cubins match the signed originals when the capture machine's paths are absent. The executable PTX arithmetic and layout are preserved. [Recorded DWARF source attributes](choices/residual-dwarf-files.json)

The fixed GPU capture command, target and draft revisions, and selection tables are in the [capture instructions](CAPTURE.md). That preparation has hardware-dependent graph addresses and creates a fresh signed capture. The signed inputs supplied with the checkpoint preserve those addresses for exact CPU replay.

## Build the serving CLI

The qualified native executable uses the Flare runtime tag below. The preparation branch contains the Python and Rust graph-building tools for this checkpoint.

```bash
git clone --branch kimi-k31-dspark-tp16-prefill32k-runtime-v1 https://github.com/voipmonitor/flare.git ../flare-runtime
cargo build --locked --release --manifest-path ../flare-runtime/flare-cli/Cargo.toml
```

The qualified configuration is sixteen RTX PRO 6000 Blackwell Workstation GPUs, TP16/DCP16 and one active request. Three distinct uncached 32,768-token prompts with one requested output token measured a median of 2,632 input tokens/s on the native release. The client requested temperature zero with seed 17. The captured sampler fixes temperature at 1.0 and top-p at 0.95, and the runtime applies those captured values. The same machine and executable with the PCIe parent achieved 552 input tokens/s. The nine-request coding decode median was 130.84 versus 127.77 output tokens/s. Both retained 2,304 KV pages and 116 snapshots under the default memory policy. The exact release can also be selected with `--memory-limit=-0G`.

In a separate full-from-capture replay, both original debug source paths were absent during PTX assembly. All 1,637 model files matched, with 84 independently emitted files and 1,553 inherited companions.

The PCIe-parent rebuild was checked by comparing all 1,637 files with the qualified serving checkpoint. The 35 independently written files matched byte for byte and the remaining files shared the parent's inodes. Modifying a rank signature caused native verification to fail before the output directory was published.
