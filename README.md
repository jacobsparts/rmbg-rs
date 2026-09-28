# rmbg-rs

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).

RMBG-2.0 (BiRefNet) background removal in one self-contained binary: feed it a
PNG and get back the subject on a transparent background. No Python, PyTorch,
ONNX Runtime, or CUDA toolkit needed.

```sh
rmbg --weights rmbg-2.0.safetensors -i input.png -o output.png
```

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels. The GPU is used when the CUDA driver can be brought up
  and the CPU engine otherwise; `--device cpu|gpu` overrides that choice.
* 1.9 MiB binary, statically linked except `libc`/`libm`/`libgcc_s`;
  `libcuda.so.1` is `dlopen`ed, so no driver is required on disk.
* Deterministic: the GPU and CPU paths agree to within one level of 255, and
  both agree with the original PyTorch model to within float32 accumulation
  (mean |Δalpha| ≈ 0.14 / 255).

Both backends agree with the upstream PyTorch implementation to within one level
of 255 (0.027% of pixels off by more).

## Download

The prebuilt binary is attached to the
[release](https://github.com/jacobsparts/rmbg-rs/releases).

| asset | what it is |
|---|---|
| `rmbg-linux-x86_64` | the engine: x86-64 Linux with glibc >= 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); the GPU path needs an NVIDIA driver |

```sh
chmod +x rmbg-linux-x86_64
./rmbg-linux-x86_64 --weights rmbg-2.0.safetensors -i input.png -o output.png
```

The examples below write the program as `rmbg`, which is the name
`cargo build --release` produces - rename the download to that, or keep the
full path.

> **Model licence:** RMBG-2.0 is licensed by [BRIA](https://bria.ai) for
> **non-commercial use only**. The checkpoint is not covered by this
> repository's MIT licence and is not distributed here. Read the
> [model card](https://huggingface.co/briaai/RMBG-2.0) before you use it.

## Getting the checkpoint

The 844 MiB checkpoint is gated on Hugging Face, so it cannot be redistributed.
`get-model.py` fetches it and verifies size and SHA-256; it is standard-library
only - no `huggingface_hub`, no second file:

```sh
./get-model.py                    # writes ./rmbg-2.0.safetensors
```

A Hugging Face account with access accepted on the
[model page](https://huggingface.co/briaai/RMBG-2.0), and a read token, are
required; the token is taken from `--token`, then `HF_TOKEN`, then
`~/.cache/huggingface/token`, and asked for interactively as a last resort. A
file that is already present and correct is left alone; the finished file is
renamed into place only once it verifies, so an interrupted run leaves nothing
behind.

## Usage

```sh
rmbg --weights rmbg-2.0.safetensors -i input.png -o output.png --device gpu
rmbg --weights rmbg-2.0.safetensors -i input.png -o output.png --device cpu
rmbg --weights rmbg-2.0.safetensors -i input.png -o alpha.png --alpha-only   # mask only
```

Naming the GPU (`--gpu` or `--device gpu`) is a demand: a driver that will not
load is an error rather than a fallback.

Input: PNG (RGB/RGBA/gray, 8- or 16-bit). Output: RGBA PNG, or a grayscale mask
with `--alpha-only`. Images are resized to 1024×1024 for the network and the
mask is resized back, matching the reference preprocessing.

| flag | meaning |
|---|---|
| `--weights` | the checkpoint as `get-model.py` writes it |
| `-i, --input` / `-o, --output` | PNG in / PNG (or mask) out |
| `--device` | `gpu` or `cpu` (default: `gpu` when the driver can be brought up, `cpu` otherwise) |
| `--gpu` / `--cpu` | shorthands for the two; `--gpu` refuses to fall back |
| `--alpha-only` | write the grayscale mask instead of the composited RGBA |
| `--cuda-selftest` | check every CUDA op against the CPU code |
| `--dump-dir` / `--dump-step` | dump intermediates for debugging |

## Licence and attribution

The Rust and CUDA code here is MIT licensed (see `LICENSE`). It is an
independent reimplementation of the BiRefNet / RMBG-2.0 architecture; the
architecture and weights are by [BRIA.AI](https://bria.ai) and the
shifted-window backbone derives from the MIT-licensed Swin Transformer
(© Microsoft Research). **The weights are not covered by this repository's
licence and are not distributed here**: the RMBG-2.0 checkpoint is free for
non-commercial use only - see the
[model card](https://huggingface.co/briaai/RMBG-2.0).
