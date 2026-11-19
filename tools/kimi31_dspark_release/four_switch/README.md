# Rebuild the four-switch Kimi DSpark checkpoint

This recipe reproduces the native TP16/DCP16 package for one active Kimi K3.1-Pro request on sixteen SM120 GPUs. It builds on a Linux x86-64 CPU host using the captured graph metadata, prepared weights, fixed program selections and detached public signatures included with the release. The file verified at the end is the signed serving primary, SHA-256 `4069909f9b374894dfc050b6e600827ba6fdabcfc13f316c9e7ef3e6131cfc08`.

Install Git, Rust 1.95 with Cargo, CMake, the Protobuf compiler 3.21, GCC 13, Binutils, OpenSSL development headers, Bubblewrap, Zstandard and CUDA 13.2.86 at `/usr/local/cuda-13.2`. Activate a Python 3.12 preparation environment with PyTorch 2.14.1+cu132, Triton 3.8.0, NVIDIA CUTLASS DSL 4.8.1, Protobuf 7.36.2 and the Flare preparation dependencies. Use complete Git histories for both source checkouts. Keep the download, build directory and output on one filesystem. The download occupies 1.48 TB. Hardlinks let all intermediate packages share the prepared weights.

```bash
set -e
git clone --branch kimi-k31-dspark-tp16-prefill32k-v2 https://github.com/voipmonitor/flare.git flare-preparation
git clone --branch kimi-k31-dspark-tp16-prefill32k-v2 https://github.com/voipmonitor/flashinfer.git flashinfer-preparation
hf download local-inference-lab/kimi-k3.1-pro-dspark-flare \
  --revision kimi-k31-dspark-tp16-prefill32k-v2 --local-dir "$PWD/kimi-dspark"
flare-preparation/tools/kimi31_dspark_release/four_switch/rebuild.sh \
  "$PWD/kimi-dspark" "$PWD/flashinfer-preparation" "$PWD/kimi-build" "$PWD/kimi-rebuilt"
(cd kimi-rebuilt && sha256sum --quiet -c ../kimi-dspark/checksums.sha256)
```

Use fresh build and output directories. The script selects `python3` from the active environment. For a different preparation interpreter, prefix the build command with `PYTHON=/path/to/venv/bin/python`. Cargo can fetch its locked dependencies during the first build. For an offline build, populate the Cargo cache with `cargo fetch --locked --manifest-path flare-preparation/tools/kimi31_dspark_release/four_switch/assembler/Cargo.toml` while connected and then set `CARGO_NET_OFFLINE=true`. [Flare preparation environment](../../../docs/model-preparation.md)

The reconstruction uses two signed inputs in the HF snapshot. The Ignition capture supplies the original residual bindings. The signed 32K prefill package supplies the weights, executable graph catalog and prepared PCIe channels. Its primary and its original row-kernel companion are stored under `rebuild/prefill32k-v1/`. The same command can also start from the complete HF snapshot tagged `kimi-k31-dspark-tp16-prefill32k-v1`. [Capture from the upstream model weights](https://github.com/voipmonitor/flare/blob/kimi-k31-dspark-tp16-capture-v1/tools/kimi31_dspark_release/CAPTURE.md) · [Reconstruct the 32K prefill input from the original capture](https://huggingface.co/local-inference-lab/kimi-k3.1-pro-dspark-flare/blob/kimi-k31-dspark-tp16-prefill32k-v1/README.md)

Preparation first binds cooperative sparse attention and latent exchange, then fuses the expert reduction with normalization and preserves the original positive-zero residual arithmetic. Packed verification generates only the draft positions that fit in the target graph. For four adjacent tensor ranks per physical switch, the vocabulary gather crosses the two switch boundaries before the two within-switch exchanges. Four-row routing combines feature gathering with top-16 expert selection. The final expert program uses separate FC1, SiTU and FC2 graph nodes with the original weight bindings and lock contract.

The release branches contain the required compiler histories. The script extracts Flare revision `821493f0771692737aec5b3534ca2c203196c880` for the original captured programs and FlashInfer revision `d6cf5c01f53e542c16c24310d51fff9b929c4d96` for the three signed expert images. Later provider changes affect their machine code. The four-switch gather is compiled using the tagged provider checkout. Bubblewrap supplies the absolute source paths encoded in the recorded CUDA debug information. Those paths exist inside the build namespace. [Pinned expert compiler](https://github.com/voipmonitor/flashinfer/tree/kimi-k31-dspark-tp16-expert-build-v2)
