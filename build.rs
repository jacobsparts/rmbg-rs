//! Compiles this engine's kernels into TWO MODULES.
//!
//! `cuda/swin.cu` is this project's own kernel family - the vision ops (Swin
//! window assembly, token shuffles, deformable convolution, resampling, NCHW
//! channel ops, token-layout attention) that no other engine calls. The rest
//! are generic ops taken from the shared `lightgpu` toolkit. Each file compiles
//! to its own fatbin with its own `--entries` list, and `src/cuda.rs` loads both
//! as separate modules, so neither file can shadow a name in the other.
//!
//! Both lists are checked against the file they are compiled from before nvcc
//! runs, so a typo or a name moved to the wrong file fails the build rather
//! than the first forward pass.

/// Generic ops that live in the shared toolkit.
const TOOLKIT_KERNELS: &[&str] = &[
    // elementwise / activation
    "lg_add",
    "lg_copy",
    "lg_relu",
    "lg_sigmoid",
    "lg_gelu_erf",
    // norm
    "lg_layer_norm",
    // convolution / linear
    "lg_conv1x1",
    "lg_conv_kxk",
    "lg_conv3x3s1p1",
    "lg_conv4x4s4",
    "lg_linear_1x1",
    "lg_linear",
];

/// This project's own kernel family, in `cuda/swin.cu`.
const SWIN_KERNELS: &[&str] = &[
    // elementwise / activation specific to the vision path
    "lg_double_sigmoid",
    "lg_channel_affine",
    "lg_mul_broadcast",
    // channel bookkeeping
    "lg_channel_copy",
    "lg_channel_mean",
    // layout / token bookkeeping
    "lg_nchw_to_tokens",
    "lg_tokens_to_nchw",
    "lg_pad_tokens",
    "lg_roll_tokens",
    "lg_add_crop",
    "lg_add_crop_tokens",
    "lg_patch_merge",
    "lg_tile_patches",
    "lg_window_gather",
    "lg_window_scatter",
    // pool / resize / convolution
    "lg_resize_bilinear",
    "lg_deform_conv",
    // attention
    "lg_repack_attn",
    "lg_attn_scores",
    "lg_softmax_rows",
    "lg_attn_apply",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/swin.cu");

    let toolkit = lightgpu_build::toolkit_kernels_cu()
        .expect("locate lightgpu's cuda/kernels.cu (set LA_GPU_DIR to override)");
    let toolkit = toolkit.to_string_lossy().into_owned();

    for k in TOOLKIT_KERNELS {
        assert!(
            lightgpu_build::known_kernel(k),
            "unknown kernel `{k}`: not defined by the toolkit - did it move into cuda/swin.cu?"
        );
    }
    let swin = std::fs::read_to_string("cuda/swin.cu").expect("read cuda/swin.cu");
    let defined = lightgpu_build::kernel_names_in(&swin);
    for k in SWIN_KERNELS {
        assert!(
            defined.iter().any(|d| d == k),
            "`{k}` is not defined in cuda/swin.cu (it has {})",
            defined.join(", ")
        );
    }

    lightgpu_build::fatbin_modules(&[
        lightgpu_build::Source {
            path: &toolkit,
            out_name: "rmbg_toolkit.fatbin",
            entries: Some(TOOLKIT_KERNELS),
        },
        lightgpu_build::Source {
            path: "cuda/swin.cu",
            out_name: "rmbg_swin.fatbin",
            entries: Some(SWIN_KERNELS),
        },
    ]);
}
