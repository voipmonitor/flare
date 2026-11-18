# Prepare the Kimi K3.1 Pro target and DSpark on SM120

Use sixteen RTX PRO 6000 Blackwell GPUs on a single host with the CUDA 13.2 toolchain, GCC 13 and Python 3.12. The target and draft use the original Hugging Face snapshots. Capture requires a writable model disk with room for the prepared target and its staging files. Set up the Flare preparation environment with the local Heliopause extension and its package metadata support. [Preparation environment](../../docs/model-preparation.md)

```bash
git clone --branch kimi-k31-dspark-tp16-capture-v1 https://github.com/voipmonitor/flare.git flare-capture
git clone --branch kimi-k31-dspark-tp16-capture-v1 https://github.com/voipmonitor/flashinfer.git flashinfer-capture
hf download moonshotai/Kimi-K3.1-Pro --revision 6d0a0202956796543364d666ab19c6ad1a494ee1 --local-dir snapshots/kimi
hf download Inferact/Kimi-K3.1-Pro-DSpark --revision e4e6e6261e9f8b6d5f249f9627eb6b109d762d5f --local-dir snapshots/dspark
cd flare-capture
uv sync --locked --extra prepare --extra test
PATH="$PWD/.venv/bin:$PATH" tools/kimi31_dspark_release/capture.sh ../snapshots/kimi ../snapshots/dspark ../flashinfer-capture ../kimi-capture
```

The capture uses TP16/DCP16, one active request and one image slot, 2,048-token prefill graphs and seven draft proposals. The target uses online MXFP8 projections and its checkpoint MXFP4 experts. The draft uses an NVFP4 head and replicated FP8 Markov head. The preparation entry point loads the fixed Inductor choices and the saved FlashInfer provider table. Native graph capture still requires real weight bindings on all ranks. A fresh capture creates its own package identity, signing authority and captured addresses. It is an operational reconstruction of the preparation stage. Exact release replay starts from the distributed signed capture metadata and its weight companions.
