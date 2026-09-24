# rmbg-rs

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).
The family also includes [locate-anything-rs](https://github.com/jacobsparts/locate-anything-rs),
[realesrgan-rs](https://github.com/jacobsparts/realesrgan-rs) and
[lama-inpaint-rs](https://github.com/jacobsparts/lama-inpaint-rs); they share the
[lightgpu toolkit](https://github.com/jacobsparts/lightgpu).
[pixeldeck](https://github.com/jacobsparts/pixeldeck) is a local web app for
cleaning up product photos that drives all of these engines.

A single-binary inference engine for [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
(BiRefNet) background removal, written in Rust. Feed it a PNG, get back the
subject on a transparent background. No Python, no PyTorch, no ONNX runtime,
no CUDA libraries to install — the binary is the whole runtime.

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels, selected at run time with `--device cpu|gpu`.
* 1.9 MiB binary, statically linked except `libc`/`libm`/`libgcc_s`; the CUDA
  kernels are embedded as two fatbins (only the ones this engine calls) and
  `libcuda.so.1` is `dlopen`ed, so the CPU path works on machines with no
  NVIDIA driver at all.
* Deterministic: the GPU and CPU paths agree to within the published one-level
  guarantee (measured max |Δalpha| = 1 of 255 on a single pixel of 1.35 M), and
  both agree with the original PyTorch model to within float32 accumulation
  (mean |Δalpha| ≈ 0.14 / 255).

## Download

Prebuilt binaries are attached to the GitHub releases:

| asset | contents | runs on |
|---|---|---|
| `rmbg-linux-x86_64` | CPU + CUDA | any x86-64 Linux with glibc ≥ 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); GPU path needs an NVIDIA driver |
| `rmbg-linux-x86_64-cpu-only` | CPU only | same, but nothing NVIDIA-related is ever touched |

> **Model licence:** RMBG-2.0 is licensed by [BRIA](https://bria.ai) for
> **non-commercial use only**, for research or evaluation. The checkpoint is
> not covered by this repository's MIT licence, and it is not distributed here
> or in the releases. Read the
> [model card](https://huggingface.co/briaai/RMBG-2.0) before you use it.

The checkpoint is not in the releases: the file is 844 MiB, and the upstream
repository is *gated*, so it cannot be redistributed here. `get-model.py`
fetches it and verifies its size and SHA-256 against the checkpoint this engine
was validated on. It is a single file, standard library only — no `curl`
needed, no `huggingface_hub`, and no second file to download:

```sh
./get-model.py                                   # from a checkout; writes ./rmbg-2.0.safetensors

python3 - <<'EOF'
import urllib.request
src = "https://raw.githubusercontent.com/jacobsparts/rmbg-rs/main/get-model.py"
exec(urllib.request.urlopen(src).read().decode())
EOF
```

**A Hugging Face account and an access token are required.** RMBG-2.0 is a
gated model: you need an account that has accepted the terms on the
[model page](https://huggingface.co/briaai/RMBG-2.0), and a read token from
<https://huggingface.co/settings/tokens>. The token is taken from `--token`,
then `HF_TOKEN`, then `HUGGING_FACE_HUB_TOKEN`, then
`~/.cache/huggingface/token`, and if none of those is set the tool asks for one
interactively. Nothing is echoed while you type it.

```sh
./get-model.py --token hf_xxxxxxxx
./get-model.py ~/models/rmbg.safetensors          # or set RMBG_MODEL_OUT
```

A file that is already present and correct is left alone rather than downloaded
again (`--force` overrides that). The finished file is renamed into place only
once it matches, so a re-uploaded upstream checkpoint fails the check instead of
being used silently, and a run that fails or is interrupted leaves nothing
behind — no partial file, and any existing checkpoint untouched. It needs Python
3 and nothing else.

## Usage

```sh
rmbg --weights rmbg-2.0.safetensors -i input.png -o output.png --device gpu
rmbg --weights rmbg-2.0.safetensors -i input.png -o output.png --device cpu
rmbg --weights rmbg-2.0.safetensors -i input.png -o alpha.png --alpha-only   # mask only
```

Input: PNG (RGB/RGBA/gray, 8- or 16-bit). Output: RGBA PNG (or a grayscale
mask with `--alpha-only`). Images are resized to 1024×1024 for the network and
the mask is resized back to the input size, matching the reference
`ViTFeatureExtractor` preprocessing.

Useful extras:

```sh
rmbg --cuda-selftest                     # check every CUDA op against the CPU code
rmbg ... --dump-dir ./steps --dump-step decoder_block1   # dump intermediates for debugging
```

`--weights` takes the raw checkpoint exactly as `get-model.py` writes it (the
upstream file is named `model.safetensors`; the saved name is up to you). No
index or preprocessing file is needed.

## Performance

Measured on a GTX 1080 with a 1500×900 input:

| backend | forward time | peak RSS |
|---|---|---|
| `--device gpu` | ~11 s | 1.1 GiB |
| `--device cpu` | ~91 s (all cores) | 4.3 GiB |

## Building from source

Rust (stable) and, for the GPU backend, the CUDA toolkit (`nvcc`):

```sh
cargo build --release                        # both backends (needs nvcc)
cargo build --release --no-default-features  # CPU-only, 1.1 MiB, no nvcc
```

Set `NVCC=/path/to/nvcc` if it is not on `PATH`. The GPU build compiles the
kernels this engine actually calls, in **two modules**:

* `cuda/swin.cu` — this project's own kernel family: the 21 vision-model ops
  (Swin window assembly, token shuffles, deformable convolution, resampling,
  NCHW channel ops and the token-layout attention). They used to sit in the
  shared toolkit, where they shared a file with an LLM/q8 kernel set they have
  almost nothing in common with.
* the shared [`lightgpu`](https://github.com/jacobsparts/lightgpu) toolkit's `cuda/kernels.cu` — the 12
  generic ops (elementwise, `layer_norm`, the convolution and linear family).

`build.rs` compiles each file to its own fatbin with its own `--entries` list
(`lightgpu_build::fatbin_modules`) for sm_61, sm_75 and sm_80 plus PTX
(`rmbg_toolkit.fatbin` 233,136 B + `rmbg_swin.fatbin` 390,184 B), and checks every
name against the file it is compiled from before `nvcc` runs - so a name in the
wrong file fails the build rather than the first forward pass. `src/cuda.rs`
loads both as separate modules: separate modules are separate namespaces, so
neither file can shadow a name in the other, and a name that is in neither
still fails eagerly at startup (every kernel is resolved at load). Selecting the
33 kernels this engine calls, rather than the toolkit's whole set, is what keeps
the embedded kernel bytes at 609 KB instead of 1.7 MB. The toolkit's build
script exports its source path (`DEP_LIGHTGPU_KERNELS_CU`), so no path needs
guessing. Nothing CUDA-related is needed to build or run the CPU-only binary.

### Supporting other GPUs

The release binary ships native code for the three most common
architectures, but the kernels use no architecture-specific intrinsics
(the only synchronization primitive is plain `__syncthreads`), so adding
support for another GPU is purely a build-time flag change — no source
edits. The `nvcc` invocation now lives in the toolkit's build helper
(`lightgpu-build`), whose `DEFAULT_ARCHES` list is what both of this crate's
fatbins are built from; set `LA_CUDA_ARCH` to override it for one build, e.g.
for Volta:

```sh
LA_CUDA_ARCH=compute_70,code=sm_70 cargo build --release
```

Three tiers, in increasing effort:

* **Volta (`sm_70`), Tesla P100 (`sm_60`), or native code for newer GPUs
  (`sm_86`/`sm_89`/`sm_90` instead of PTX JIT):** one flag, same CUDA 12.x
  toolkit. Verified: the full kernel set compiles cleanly for
  sm_52/60/70/86/90 with the same toolkit used for the release binary.
  Adding native SASS also *removes* the R550+ driver requirement for that
  card, since a cubin is loaded as-is and never recompiled.
* **Kepler / Maxwell (`sm_35`–`sm_52`):** CUDA 12 can no longer target
  these, so you additionally need the last toolkit that can: install
  CUDA 11.8 alongside and build with
  `NVCC=/path/to/cuda-11.8/bin/nvcc cargo build --release`. Note the
  cascading constraint: a cubin from an older toolkit also runs on *older
  drivers* than a CUDA 12 build would.
* **Anything older than Kepler:** not reachable — the kernels assume a
  modern SM (32-thread warps, enough shared memory per block).

## Runtime requirements

* **CPU path:** nothing but glibc ≥ 2.34. Baseline x86-64 (SSE2) — no AVX
  needed. Uses all cores via rayon.
* **GPU path:** NVIDIA driver (`libcuda.so.1` on the loader path) and a
  supported GPU. The fatbin carries native code for three architectures plus
  PTX for forward compatibility:

  | GPU generation | compute capability | example cards | how it runs | driver needed |
  |---|---|---|---|---|
  | Maxwell (sm_50/52) | 5.x | GTX 9xx | not supported | — |
  | Pascal | sm_60 | Tesla P100 | not supported | — |
  | Pascal | sm_61 | GTX 10xx, P4/P40 | native SASS | any driver |
  | Volta (sm_70/72) | 7.x | V100 | not supported | — |
  | Turing | sm_75 | RTX 20xx, T4 | native SASS | any driver |
  | Ampere | sm_80 | A100, A10 | native SASS | any driver |
  | Ampere (sm_86), Ada (sm_89), Hopper (sm_90), Blackwell (sm_100/120) | ≥ 8.6 | RTX 30xx/40xx/50xx, H100 | JIT from embedded PTX | driver R550+ (2024 or newer) |

  The CPU path in the same binary works on any of these — no GPU or driver
  required.
* **Data:** the 844 MiB checkpoint from the upstream model card, saved as
  `rmbg-2.0.safetensors`. That is the only file the binary needs besides the
  input image.
* No network, no temp files, no shell-outs, no config files.

## How it works

The whole network — Swin-Transformer backbone, the full BiRefNet decoder
including deformable convolutions (DCNv2) and the shifted-window attention
with its relative-position tables — is reimplemented from the original
PyTorch code in Rust. The CUDA kernels are split by ownership: the
vision-model family this engine needs (and no other engine does) lives here in
`cuda/swin.cu`, while the generic ops come from the shared `lightgpu` toolkit -
the toolkit is for what several projects share, not for whatever one of them
happens to need. `src/cuda.rs` is a thin layer that loads both modules,
allocates device buffers and marshals launch arguments.
Weights are the 754 tensors from the official checkpoint, `mmap`ed and used in
place. There are no neural-network frameworks anywhere in the dependency tree;
the only crates are `png`, `rayon`, `libc`, `lightgpu` and its build helper.

## Accuracy

Verified against the PyTorch reference on the official checkpoint:
end-to-end alpha outputs match the reference to within the float32
accumulation differences between backends (mean |Δ| 0.136 / 255, 0.027% of
pixels off by more than one level), and the CPU and GPU paths of this crate
agree with each other to within the published guarantee of at most one level:
measured over the 1500×900 test image (1,350,000 pixels), exactly one pixel
differs, by one level.
`--cuda-selftest` compares each of the 24 CUDA ops against its CPU
implementation (relative error ≤ 8e-7).

## License and attribution

The Rust and CUDA code in this repository is licensed under the MIT license;
see [LICENSE](LICENSE).

The code is an independent reimplementation of the BiRefNet / RMBG-2.0
architecture; the architecture and weights are by [BRIA.AI](https://bria.ai)
and the shifted-window transformer backbone derives from the MIT-licensed
Swin Transformer (© Microsoft Research). The **weights are not covered by
this repository's license and are not distributed here**: the RMBG-2.0
checkpoint is free for non-commercial use only, see the
[model card](https://huggingface.co/briaai/RMBG-2.0) for terms.
