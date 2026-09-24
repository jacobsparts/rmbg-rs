//! CUDA implementation of the RMBG-2.0 forward graph.
//!
//! This mirrors `graph.rs` (the decoder, NCHW) and the token-layout backbone in
//! `forward.rs`, but keeps activations and weights in device memory. Weights are
//! uploaded once at load time - the 884 MB checkpoint is the dominant cost of
//! setting up - and every activation is a `DevBuf` reused through a small pool.
//!
//! Numerics: kernels reproduce the CPU accumulation orders (see cuda/kernels.cu
//! for the per-op notes) so that GPU and CPU agree to float rounding rather than
//! to a different reduction tree.
//!
//! Compiled only with `--features cuda`.

#![allow(dead_code)]

use crate::cuda::{Cuda, DevBuf};
use lightgpu::vm::{Args, Launch};
use crate::tensor::Tensor;
use crate::weights::Weights;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

const WINDOW: usize = 12;
const DEPTHS: [usize; 4] = [2, 2, 18, 2];
const HEADS: [usize; 4] = [6, 12, 24, 48];
const DIMS: [usize; 4] = [192, 384, 768, 1536];
const LN_EPS: f32 = 1e-5;

/// A device-resident weight buffer plus the shape information the kernels need.
#[derive(Clone)]
pub struct DW {
    pub buf: Rc<DevBuf>,
    pub shape: Vec<usize>,
}

impl DW {
    pub fn oc(&self) -> usize { self.shape[0] }
    pub fn ic(&self) -> usize { self.shape[1] }
}

/// Device-side equivalent of `tensor::Tensor` (NCHW f32).
#[derive(Clone)]
pub struct DTensor {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub buf: Rc<DevBuf>,
}

impl DTensor {
    pub fn hw(&self) -> usize { self.h * self.w }
    pub fn len(&self) -> usize { self.c * self.h * self.w }
}

pub struct CudaModel {
    pub cu: Cuda,
    /// Every weight stays resident for the lifetime of the model.
    weights: RefCell<Vec<Rc<DevBuf>>>,
    wmap: RefCell<HashMap<String, DW>>,
    /// Reusable scratch buffers, keyed by element count.
    pool: RefCell<HashMap<usize, Vec<Rc<DevBuf>>>>,
    pub upload_s: f64,
}

impl CudaModel {
    /// Upload the whole checkpoint and resolve every kernel.
    pub fn load(wts: &Weights) -> Result<CudaModel, String> {
        let t0 = std::time::Instant::now();
        let cu = Cuda::init(true)?;
        let m = CudaModel {
            cu,
            weights: RefCell::new(Vec::new()),
            wmap: RefCell::new(HashMap::new()),
            pool: RefCell::new(HashMap::new()),
            upload_s: 0.0,
        };
        let mut n = 0usize;
        let mut bytes = 0usize;
        for name in wts.names() {
            let src = wts.get(name)?;
            let info = &wts.tensors[name];
            let buf = Rc::new(DevBuf::from_host(src)?);
            bytes += src.len() * 4;
            n += 1;
            m.weights.borrow_mut().push(buf.clone());
            m.wmap.borrow_mut().insert(
                name.to_string(),
                DW { buf, shape: info.shape.clone() },
            );
        }
        // Resolve every kernel name eagerly so a missing kernel fails at load
        // time rather than mid-forward; the toolkit memoizes the lookup.
        for k in KERNEL_NAMES.iter() {
            m.module_of(k)?;
        }
        let upload_s = t0.elapsed().as_secs_f64();
        eprintln!("cuda: uploaded {} tensors ({:.1} MB) in {:.2}s", n, bytes as f64 / 1e6, upload_s);
        let mut mm = m;
        mm.upload_s = upload_s;
        Ok(mm)
    }

    pub fn w(&self, name: &str) -> Result<DW, String> {
        match self.wmap.borrow().get(name) {
            Some(d) => Ok(DW { buf: d.buf.clone(), shape: d.shape.clone() }),
            None => Err(format!("missing tensor {}", name)),
        }
    }

    /// Borrow a scratch buffer of at least `n` f32 (never freed during a run).
    pub fn scratch(&self, n: usize) -> Rc<DevBuf> {
        if n == 0 {
            return Rc::new(DevBuf::empty());
        }
        if let Some(v) = self.pool.borrow_mut().get_mut(&n) {
            if let Some(b) = v.pop() {
                return b;
            }
        }
        Rc::new(DevBuf::alloc(n).expect("cuda alloc"))
    }

    pub fn give_back(&self, b: Rc<DevBuf>) {
        if Rc::strong_count(&b) == 1 && b.len > 0 {
            self.pool.borrow_mut().entry(b.len).or_default().push(b);
        }
    }

    pub fn dt(&self, c: usize, h: usize, w: usize) -> DTensor {
        let buf = self.scratch(c * h * w);
        DTensor { c, h, w, buf }
    }

    pub fn sync(&self) -> Result<(), String> { self.cu.sync() }
}

/// Every kernel the graph may launch. Resolved eagerly so a typo fails at load
/// time rather than mid-forward.
pub static KERNEL_NAMES: &[&str] = &[
    "lg_add",
    "lg_copy",
    "lg_conv1x1",
    "lg_conv_kxk",
    "lg_conv3x3s1p1",
    "lg_conv4x4s4",
    "lg_layer_norm",
    "lg_channel_affine",
    "lg_gelu_erf",
    "lg_relu",
    "lg_sigmoid",
    "lg_resize_bilinear",
    "lg_deform_conv",
    "lg_double_sigmoid",
    "lg_channel_mean",
    "lg_channel_copy",
    "lg_linear_1x1",
    "lg_repack_attn",
    "lg_attn_scores",
    "lg_softmax_rows",
    "lg_attn_apply",
    "lg_window_gather",
    "lg_window_scatter",
    "lg_pad_tokens",
    "lg_roll_tokens",
    "lg_add_crop",
    "lg_patch_merge",
    "lg_tile_patches",
    "lg_mul_broadcast",
    "lg_nchw_to_tokens",
    "lg_tokens_to_nchw",
    "lg_linear",
    "lg_add_crop_tokens",
];

// ---------------------------------------------------------------------------
// Launch helpers for the NCHW ops. Each takes device tensors and returns fresh
// device tensors; the caller is responsible for giving buffers back.
// ---------------------------------------------------------------------------

const THREADS: u32 = 256;

fn grid_for(n: usize) -> u32 { ((n + THREADS as usize - 1) / THREADS as usize) as u32 }

impl CudaModel {
    /// The module a kernel lives in: this engine's own family first, the
    /// toolkit's second.
    ///
    /// The lookup is by CONTAINMENT, not by the compile-time lists, and that is
    /// deliberate for the two kernels that just moved: they are in the toolkit
    /// fatbin now, so `swin.has` is false for them, but a build of an older
    /// `swin.cu` that still defined them would resolve to that copy instead of
    /// launching a missing symbol. Both modules are probed once at startup
    /// (`KERNEL_NAMES`), so a genuinely absent name still fails there rather
    /// than in the middle of a forward pass.
    fn module_of(&self, name: &str) -> Result<&lightgpu::vm::Module, String> {
        if self.cu.swin.has(name) {
            Ok(&self.cu.swin)
        } else {
            Ok(&self.cu.module)
        }
    }

    /// Generic 1-D launch over `n` elements with an int64 count argument.
    fn launch_n(&self, name: &str, dst: &DTensor, src: &DTensor, n: usize) -> Result<(), String> {
        
        let mut aa = Args::new();
        aa.ptr(src.buf.ptr);
        aa.ptr(dst.buf.ptr);
        aa.i64(n as i64);
        aa.launch(self.module_of(name)?, name, Launch::new((grid_for(n), 1, 1), (THREADS, 1, 1)))
    }

    pub fn copy(&self, src: &DTensor) -> Result<DTensor, String> {
        let dst = self.dt(src.c, src.h, src.w);
        self.launch_n("lg_copy", &dst, src, src.len())?;
        Ok(dst)
    }

    pub fn relu(&self, src: &DTensor) -> Result<DTensor, String> {
        let dst = self.dt(src.c, src.h, src.w);
        self.launch_n("lg_relu", &dst, src, src.len())?;
        Ok(dst)
    }

    pub fn sigmoid(&self, src: &DTensor) -> Result<DTensor, String> {
        let dst = self.dt(src.c, src.h, src.w);
        self.launch_n("lg_sigmoid", &dst, src, src.len())?;
        Ok(dst)
    }

    pub fn gelu(&self, src: &DTensor) -> Result<DTensor, String> {
        let dst = self.dt(src.c, src.h, src.w);
        self.launch_n("lg_gelu_erf", &dst, src, src.len())?;
        Ok(dst)
    }

    /// 2 * sigmoid(x) in place (the deformable-conv modulator).
    pub fn double_sigmoid(&self, x: &DTensor) -> Result<DTensor, String> {
        // The toolkit's lg_double_sigmoid is OUT-OF-PLACE (x, y, n) with an int
        // count, unlike this engine's original 2-argument in-place kernel. The
        // cell below must take the result buffer and the int count in the same
        // positions.
        let dst = self.dt(x.c, x.h, x.w);
        
        let mut aa = Args::new();
        aa.ptr(x.buf.ptr);
        aa.ptr(dst.buf.ptr);
        aa.i32(x.len() as i32);
        aa.launch(self.module_of("lg_double_sigmoid")?, "lg_double_sigmoid", Launch::new((grid_for(x.len()), 1, 1), (THREADS, 1, 1)))?;
        Ok(dst)
    }

    /// Per-channel affine: out = in * scale[c] + shift[c] (folded BatchNorm).
    ///
    /// Calls `lg_channel_affine`, whose contract is exactly this op over NCHW.
    /// (An earlier version of this port tried `lg_row_affine`, which indexes its
    /// vector by the CONTIGUOUS dimension and therefore cannot express a
    /// per-channel affine - see the correction in the toolkit's op table.)
    pub fn affine(&self, src: &DTensor, scale: &DW, shift: &DW) -> Result<DTensor, String> {
        let dst = self.dt(src.c, src.h, src.w);
        
        let mut aa = Args::new();
        aa.ptr(src.buf.ptr);
        aa.ptr(dst.buf.ptr);
        aa.ptr(scale.buf.ptr);
        aa.ptr(shift.buf.ptr);
        aa.i32(src.c as i32);
        aa.i32(src.hw() as i32);
        aa.launch(self.module_of("lg_channel_affine")?, "lg_channel_affine", Launch::new((grid_for(src.len()), 1, 1), (THREADS, 1, 1)))?;
        Ok(dst)
    }

    /// Folded BatchNorm loaded from `prefix.{weight,bias,running_mean,running_var}`.
    /// The two derived vectors are tiny; they are computed on the host and
    /// uploaded, matching graph.rs::Affine::load bit-for-bit.
    pub fn affine_from_bn(&self, wts: &Weights, prefix: &str) -> Result<(DW, DW), String> {
        let w = wts.get(&format!("{}.weight", prefix))?;
        let b = wts.get(&format!("{}.bias", prefix))?;
        let mean = wts.get(&format!("{}.running_mean", prefix))?;
        let var = wts.get(&format!("{}.running_var", prefix))?;
        let n = w.len();
        let mut scale = vec![0.0f32; n];
        let mut shift = vec![0.0f32; n];
        for c in 0..n {
            let s = w[c] / (var[c] + 1e-5).sqrt();
            scale[c] = s;
            shift[c] = b[c] - mean[c] * s;
        }
        let sb = Rc::new(DevBuf::from_host(&scale)?);
        let hb = Rc::new(DevBuf::from_host(&shift)?);
        self.weights.borrow_mut().push(sb.clone());
        self.weights.borrow_mut().push(hb.clone());
        Ok((DW { buf: sb, shape: vec![n] }, DW { buf: hb, shape: vec![n] }))
    }

    /// k x k convolution with stride 1 and padding K/2 (or the 4x4/stride-4
    /// patch-embed variant when `k == 4` and `stride4`).
    pub fn conv(&self, src: &DTensor, w: &DW, bias: Option<&DW>, k: usize, stride4: bool) -> Result<DTensor, String> {
        let oc = w.oc();
        let ic = w.ic();
        let (oh, ow) = if stride4 { (src.h / 4, src.w / 4) } else { (src.h, src.w) };
        let dst = self.dt(oc, oh, ow);
        let name = if stride4 {
            "lg_conv4x4s4"
        } else if k == 1 {
            "lg_conv1x1"
        } else if k == 3 {
            "lg_conv3x3s1p1"
        } else {
            "lg_conv_kxk"
        };
        let bias_ptr = bias.map(|b| b.buf.ptr).unwrap_or(0);
        
        let total = oc * oh * ow;
        let mut aa = Args::new();
        aa.ptr(src.buf.ptr);
        aa.ptr(w.buf.ptr);
        aa.ptr(bias_ptr);
        aa.ptr(dst.buf.ptr);
        aa.i32(ic as i32);
        aa.i32(oc as i32);
        aa.i32(src.h as i32);
        aa.i32(src.w as i32);
        aa.i32(k as i32);
        aa.launch(self.module_of(name)?, name, Launch::new((grid_for(total), 1, 1), (THREADS, 1, 1)))?;
        Ok(dst)
    }

    /// 1x1 conv applied to a 1x1 spatial tensor (ASPP global-pool branch).
    pub fn linear_1x1(&self, src: &DTensor, w: &DW, bias: Option<&DW>) -> Result<DTensor, String> {
        let oc = w.oc();
        let ic = w.ic();
        let dst = self.dt(oc, 1, 1);
        let bias_ptr = bias.map(|b| b.buf.ptr).unwrap_or(0);
        
        let mut aa = Args::new();
        aa.ptr(src.buf.ptr);
        aa.ptr(w.buf.ptr);
        aa.ptr(bias_ptr);
        aa.ptr(dst.buf.ptr);
        aa.i32(ic as i32);
        aa.i32(oc as i32);
        aa.launch(self.module_of("lg_linear_1x1")?, "lg_linear_1x1", Launch::new((grid_for(oc), 1, 1), (THREADS, 1, 1)))?;
        Ok(dst)
    }

    /// Linear on the row-major token layout: (rows, ic) -> (rows, oc).
    pub fn linear(&self, x: &DTensor, w: &DW, b: Option<&DW>, rows: usize) -> Result<DTensor, String> {
        let ic = w.ic();
        let oc = w.oc();
        let dst = self.dt(oc, 1, rows);
        let bias_ptr = b.map(|d| d.buf.ptr).unwrap_or(0);
        
        let gx = ((oc + 15) / 16) as u32;
        let gy = ((rows + 15) / 16) as u32;
        let mut aa = Args::new();
        aa.ptr(x.buf.ptr);
        aa.ptr(w.buf.ptr);
        aa.ptr(bias_ptr);
        aa.ptr(dst.buf.ptr);
        aa.i32(rows as i32);
        aa.i32(ic as i32);
        aa.i32(oc as i32);
        aa.launch(self.module_of("lg_linear")?, "lg_linear", Launch::new((gx, gy, 1), (16, 16, 1)))?;
        Ok(dst)
    }

    /// Token-layout residual + crop: out = x + crop(buf), where x and out are
    /// (rows, C) row-major and buf is (C, 1, hp*wp) channel-major.
    pub fn add_crop_tokens(&self, x: &DTensor, buf: &DTensor, out: &DTensor, h: usize, w: usize, hp: usize, wp: usize) -> Result<(), String> {
        
        let mut aa = Args::new();
        aa.ptr(x.buf.ptr);
        aa.ptr(buf.buf.ptr);
        aa.ptr(out.buf.ptr);
        aa.i32(h as i32);
        aa.i32(w as i32);
        aa.i32(wp as i32);
        aa.i32(hp as i32);
        aa.i32(out.c as i32);
        aa.launch(self.module_of("lg_add_crop_tokens")?, "lg_add_crop_tokens", Launch::new((grid_for(h * w * out.c), 1, 1), (THREADS, 1, 1)))
    }

    /// LayerNorm over the channel axis of an NCHW tensor (rows = H*W).
    ///
    /// This is `lg_layer_norm`: rows are strided by C and the norm runs over the
    /// C contiguous values of each row. The pre-port kernel here was identical
    /// (ip = in + row*C, params (C, HW, eps)), so the mapping is unchanged - what
    /// changed is that the toolkit's kernel no longer keeps its reduction
    /// scratch in a fixed 256-entry shared array, which is what produced NaN
    /// whenever a caller launched it with blockDim > 256.
    pub fn layernorm(&self, src: &DTensor, w: &DW, b: &DW, eps: f32) -> Result<DTensor, String> {
        let dst = self.dt(src.c, src.h, src.w);
        let c = src.c;
        let hw = src.hw();
        // The kernel's halving tree is only correct for a power-of-two blockDim
        // (a non-power-of-two drops the tail lanes at the s=1 step).
        let threads = c.next_power_of_two().min(1024) as u32;
        // lg_layer_norm keeps its reduction scratch in fixed 1024-entry shared
        // arrays, so no dynamic shared memory is needed here.
        let shared = 0;
        
        let mut aa = Args::new();
        aa.ptr(src.buf.ptr);
        aa.ptr(w.buf.ptr);
        aa.ptr(b.buf.ptr);
        aa.ptr(dst.buf.ptr);
        aa.i32(c as i32);
        aa.i32(hw as i32);
        aa.f32(eps as f32);
        aa.launch(self.module_of("lg_layer_norm")?, "lg_layer_norm", Launch::new((hw as u32, 1, 1), (threads, 1, 1)).shared(shared))?;
        Ok(dst)
    }

    pub fn resize_bilinear(&self, src: &DTensor, oh: usize, ow: usize, align: bool) -> Result<DTensor, String> {
        let dst = self.dt(src.c, oh, ow);
        
        let mut aa = Args::new();
        aa.ptr(src.buf.ptr);
        aa.ptr(dst.buf.ptr);
        aa.i32(src.c as i32);
        aa.i32(src.h as i32);
        aa.i32(src.w as i32);
        aa.i32(oh as i32);
        aa.i32(ow as i32);
        aa.i32(align as i32);
        aa.launch(self.module_of("lg_resize_bilinear")?, "lg_resize_bilinear", Launch::new((grid_for(src.c * oh * ow), 1, 1), (THREADS, 1, 1)))?;
        Ok(dst)
    }

    /// Global average pool to (C,1,1).
    pub fn global_avg_pool(&self, src: &DTensor) -> Result<DTensor, String> {
        let dst = self.dt(src.c, 1, 1);
        let hw = src.hw();
        
        let mut aa = Args::new();
        aa.ptr(src.buf.ptr);
        aa.ptr(dst.buf.ptr);
        aa.i32(src.c as i32);
        aa.i32(hw as i32);
        // No `.shared(...)`: the toolkit's lg_channel_mean reserves its own
        // static 1024-slot scratch, sized for the largest legal blockDim, so a
        // caller never has to know what the reduction needs. This call used to
        // pass `threads * 4` bytes of dynamic shared memory, which the toolkit
        // version (correctly) ignores - passing it would not fail, it would just
        // be dead space, so the argument is gone rather than left in place.
        aa.launch(self.module_of("lg_channel_mean")?, "lg_channel_mean", Launch::new((src.c as u32, 1, 1), (THREADS, 1, 1)))?;
        Ok(dst)
    }

    /// Concatenate NCHW tensors along channels.
    pub fn cat(&self, parts: &[&DTensor]) -> Result<DTensor, String> {
        let c: usize = parts.iter().map(|t| t.c).sum();
        let h = parts[0].h;
        let w = parts[0].w;
        let dst = self.dt(c, h, w);
        let mut c0 = 0;
        for t in parts {
            
        let mut aa = Args::new();
        aa.ptr(t.buf.ptr);
        aa.ptr(dst.buf.ptr);
        aa.i32(t.c as i32);
        aa.i32(c0 as i32);
        aa.i32(t.hw() as i32);
        aa.launch(self.module_of("lg_channel_copy")?, "lg_channel_copy", Launch::new((grid_for(t.len()), 1, 1), (THREADS, 1, 1)))?;
            c0 += t.c;
        }
        Ok(dst)
    }

    /// Modulated deformable convolution, matching src/deform.rs.
    pub fn deform_conv(&self, x: &DTensor, offset: &DTensor, mask: &DTensor, w: &DW, k: usize, pad: usize) -> Result<DTensor, String> {
        let dst = self.dt(w.oc(), x.h, x.w);
        let hw = x.hw();
        
        let mut aa = Args::new();
        aa.ptr(x.buf.ptr);
        aa.ptr(offset.buf.ptr);
        aa.ptr(mask.buf.ptr);
        aa.ptr(w.buf.ptr);
        aa.ptr(dst.buf.ptr);
        aa.i32(x.c as i32);
        aa.i32(w.oc() as i32);
        aa.i32(x.h as i32);
        aa.i32(x.w as i32);
        aa.i32(k as i32);
        aa.i32(pad as i32);
        aa.launch(self.module_of("lg_deform_conv")?, "lg_deform_conv", Launch::new((grid_for(hw), w.oc() as u32, 1), (THREADS, 1, 1)))?;
        Ok(dst)
    }
}

impl DTensor {
    pub fn download(&self) -> Result<Vec<f32>, String> {
        let mut v = vec![0.0f32; self.len()];
        self.buf.download(&mut v)?;
        Ok(v)
    }
}

impl CudaModel {
    pub fn upload(&self, t: &Tensor) -> Result<DTensor, String> {
        let buf = Rc::new(DevBuf::from_host(&t.data)?);
        Ok(DTensor { c: t.c, h: t.h, w: t.w, buf })
    }
}

/// Compare one GPU op against its CPU counterpart.
fn cmp(name: &str, got: &[f32], want: &[f32], report: &mut Vec<String>) {
    if got.len() != want.len() {
        report.push(format!("{:<28} LEN {} vs {}", name, got.len(), want.len()));
        return;
    }
    let mut maxd = 0.0f32;
    let mut sumd = 0.0f64;
    let mut scale = 0.0f32;
    for (a, b) in got.iter().zip(want.iter()) {
        let d = (a - b).abs();
        if d > maxd { maxd = d; }
        sumd += d as f64;
        if b.abs() > scale { scale = b.abs(); }
    }
    let n = got.len().max(1) as f64;
    let rel = if scale > 0.0 { maxd / scale } else { maxd };
    report.push(format!(
        "{:<28} max|d| {:.3e}  rel {:.3e}  mean|d| {:.3e}  (n={}, scale {:.4})",
        name, maxd, rel, sumd / n, got.len(), scale
    ));
}

/// Deterministic pseudo-random f32 in [-1, 1] without pulling in a rand crate.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Rng { Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1)) }
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = ((self.0 >> 33) as u32) as f32 / (u32::MAX >> 1) as f32;
        v * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> { (0..n).map(|_| self.next()).collect() }
    fn tensor(&mut self, c: usize, h: usize, w: usize) -> Tensor {
        Tensor::from_vec(c, h, w, self.vec(c * h * w))
    }
}

/// Per-kernel CPU/GPU agreement check. Run with `--cuda-selftest`.
pub fn selftest() -> Result<(), String> {
    use crate::deform::deform_conv2d;
    use crate::tensor as T;

    let cu = Cuda::init(true)?;
    let m = CudaModel {
        cu,
        weights: RefCell::new(Vec::new()),
        wmap: RefCell::new(HashMap::new()),
        pool: RefCell::new(HashMap::new()),
        upload_s: 0.0,
    };
    for k in KERNEL_NAMES.iter() {
        m.module_of(k)?;
    }
    let mut r = Rng::new(12345);
    let mut report: Vec<String> = Vec::new();

    // --- elementwise -------------------------------------------------------
    let a = r.tensor(8, 16, 16);
    let da = m.upload(&a)?;
    cmp("copy", &m.copy(&da)?.download()?, &a.data, &mut report);

    let mut want = a.clone();
    T::relu_inplace(&mut want);
    cmp("relu", &m.relu(&da)?.download()?, &want.data, &mut report);

    let mut want = a.clone();
    T::sigmoid_inplace(&mut want);
    cmp("sigmoid", &m.sigmoid(&da)?.download()?, &want.data, &mut report);

    let mut want = a.clone();
    T::gelu_exact_inplace(&mut want);
    cmp("gelu", &m.gelu(&da)?.download()?, &want.data, &mut report);

    let scale = r.vec(8);
    let shift = r.vec(8);
    let ds = m.upload(&Tensor::from_vec(8, 1, 1, scale.clone()))?;
    let dh = m.upload(&Tensor::from_vec(8, 1, 1, shift.clone()))?;
    let dsw = DW { buf: ds.buf.clone(), shape: vec![8] };
    let dhw = DW { buf: dh.buf.clone(), shape: vec![8] };
    let mut want = a.clone();
    T::apply_affine_inplace(&mut want, &scale, &shift);
    cmp("affine", &m.affine(&da, &dsw, &dhw)?.download()?, &want.data, &mut report);

    // --- convolutions ------------------------------------------------------
    let x = r.tensor(16, 9, 7);
    let dx = m.upload(&x)?;
    let w11 = r.vec(32 * 16);
    let b11 = r.vec(32);
    let dw11 = m.upload(&Tensor::from_vec(32, 16, 1, w11.clone()))?;
    let db11 = m.upload(&Tensor::from_vec(32, 1, 1, b11.clone()))?;
    let want = T::conv1x1(&x, &w11, 32, 16, Some(&b11));
    let got = m.conv(&dx, &DW { buf: dw11.buf.clone(), shape: vec![32, 16] }, Some(&DW { buf: db11.buf.clone(), shape: vec![32] }), 1, false)?;
    cmp("conv1x1", &got.download()?, &want.data, &mut report);

    let w33 = r.vec(24 * 16 * 9);
    let b33 = r.vec(24);
    let dw33 = m.upload(&Tensor::from_vec(24, 16 * 3, 3, w33.clone()))?;
    let db33 = m.upload(&Tensor::from_vec(24, 1, 1, b33.clone()))?;
    let want = T::conv_kxk(&x, &w33, 24, 16, 3, Some(&b33));
    let got = m.conv(&dx, &DW { buf: dw33.buf.clone(), shape: vec![24, 16, 3, 3] }, Some(&DW { buf: db33.buf.clone(), shape: vec![24] }), 3, false)?;
    cmp("conv3x3", &got.download()?, &want.data, &mut report);

    let w77 = r.vec(8 * 16 * 49);
    let b77 = r.vec(8);
    let dw77 = m.upload(&Tensor::from_vec(8, 16 * 7, 7, w77.clone()))?;
    let db77 = m.upload(&Tensor::from_vec(8, 1, 1, b77.clone()))?;
    let want = T::conv_kxk(&x, &w77, 8, 16, 7, Some(&b77));
    let got = m.conv(&dx, &DW { buf: dw77.buf.clone(), shape: vec![8, 16, 7, 7] }, Some(&DW { buf: db77.buf.clone(), shape: vec![8] }), 7, false)?;
    cmp("conv7x7", &got.download()?, &want.data, &mut report);

    let x4 = r.tensor(3, 8, 8);
    let dx4 = m.upload(&x4)?;
    let w44 = r.vec(16 * 3 * 16);
    let b44 = r.vec(16);
    let dw44 = m.upload(&Tensor::from_vec(16, 3 * 4, 4, w44.clone()))?;
    let db44 = m.upload(&Tensor::from_vec(16, 1, 1, b44.clone()))?;
    let want = T::conv4x4_stride4(&x4, &w44, 16, 3, Some(&b44));
    let got = m.conv(&dx4, &DW { buf: dw44.buf.clone(), shape: vec![16, 3, 4, 4] }, Some(&DW { buf: db44.buf.clone(), shape: vec![16] }), 4, true)?;
    cmp("conv4x4s4", &got.download()?, &want.data, &mut report);

    // --- norms / pools -----------------------------------------------------
    let ln_w = r.vec(16);
    let ln_b = r.vec(16);
    let dlw = m.upload(&Tensor::from_vec(16, 1, 1, ln_w.clone()))?;
    let dlb = m.upload(&Tensor::from_vec(16, 1, 1, ln_b.clone()))?;
    let mut want = x.clone();
    T::layernorm_slice(&mut want.data, 16, &ln_w, &ln_b, LN_EPS);
    let got = m.layernorm(&dx, &DW { buf: dlw.buf.clone(), shape: vec![16] }, &DW { buf: dlb.buf.clone(), shape: vec![16] }, LN_EPS)?;
    cmp("layernorm C=16", &got.download()?, &want.data, &mut report);

    // odd channel count: the block reduction must still match
    let y = r.tensor(13, 5, 3);
    let dy = m.upload(&y)?;
    let ln2w = r.vec(13);
    let ln2b = r.vec(13);
    let d2w = m.upload(&Tensor::from_vec(13, 1, 1, ln2w.clone()))?;
    let d2b = m.upload(&Tensor::from_vec(13, 1, 1, ln2b.clone()))?;
    let mut want = y.clone();
    T::layernorm_slice(&mut want.data, 13, &ln2w, &ln2b, LN_EPS);
    let got = m.layernorm(&dy, &DW { buf: d2w.buf.clone(), shape: vec![13] }, &DW { buf: d2b.buf.clone(), shape: vec![13] }, LN_EPS)?;
    cmp("layernorm C=13", &got.download()?, &want.data, &mut report);

    let want = T::global_avg_pool(&x);
    let got = m.global_avg_pool(&dx)?;
    cmp("global_avg_pool", &got.download()?, &want, &mut report);

    // token-layout layernorm at the backbone's channel count: 192 is not a
    // power of two, which is exactly the case the block reduction must handle.
    let z = r.tensor(192, 1, 33);
    let dz = m.upload(&z)?;
    let z3w = r.vec(192);
    let z3b = r.vec(192);
    let d3w = m.upload(&Tensor::from_vec(192, 1, 1, z3w.clone()))?;
    let d3b = m.upload(&Tensor::from_vec(192, 1, 1, z3b.clone()))?;
    let mut want = z.clone();
    T::layernorm_slice(&mut want.data, 192, &z3w, &z3b, LN_EPS);
    let got = m.layernorm(&dz, &DW { buf: d3w.buf.clone(), shape: vec![192] }, &DW { buf: d3b.buf.clone(), shape: vec![192] }, LN_EPS)?;
    cmp("layernorm C=192", &got.download()?, &want.data, &mut report);

    // --- token-layout linear ----------------------------------------------
    // CudaLinear (qkv / proj / fc1 / fc2) all go through lg_linear, which is a
    // different kernel from the NCHW conv1x1 the selftest used to cover.
    for &(lr, lic, loc) in [(37usize, 24usize, 32usize), (37, 192, 100), (29, 768, 3072)].iter() {
        let x = r.tensor(lic, 1, lr);
        let wv = r.vec(loc * lic);
        let bv = r.vec(loc);
        let dx = m.upload(&x)?;
        let dw = m.upload(&Tensor::from_vec(loc, lic, 1, wv.clone()))?;
        let db = m.upload(&Tensor::from_vec(loc, 1, 1, bv.clone()))?;
        let mut want = vec![0.0f32; lr * loc];
        for rr in 0..lr {
            for o in 0..loc {
                let mut acc = bv[o];
                for c in 0..lic {
                    acc += x.data[rr * lic + c] * wv[o * lic + c];
                }
                want[rr * loc + o] = acc;
            }
        }
        let got = m.linear(
            &dx,
            &DW { buf: dw.buf.clone(), shape: vec![loc, lic] },
            Some(&DW { buf: db.buf.clone(), shape: vec![loc] }),
            lr,
        )?;
        cmp(&format!("linear {}x{}->{}", lr, lic, loc), &got.download()?, &want, &mut report);
    }

    // --- tile_patches (ipt input) -----------------------------------------
    // Non-square tiles catch the x/y decomposition order in lg_tile_patches.
    for &(tc, th, tw, tph, tpw) in [(2usize, 5usize, 7usize, 3usize, 2usize), (3, 32, 32, 12, 7)].iter() {
        let xt = r.tensor(tc, th, tw);
        let dxt = m.upload(&xt)?;
        let want = crate::engine::get_patches_batch(&xt, tph, tpw);
        let got = m.tile_patches(&dxt, tph, tpw)?;
        cmp(&format!("tile_patches {}x{}->{}x{}", th, tw, tph, tpw), &got.download()?, &want.data, &mut report);
    }

    // --- cat ---------------------------------------------------------------
    let p1 = r.tensor(3, 4, 5);
    let p2 = r.tensor(7, 4, 5);
    let dp1 = m.upload(&p1)?;
    let dp2 = m.upload(&p2)?;
    let want = T::cat_channels(&[&p1, &p2]);
    let got = m.cat(&[&dp1, &dp2])?;
    cmp("cat_channels", &got.download()?, &want.data, &mut report);

    // --- resize ------------------------------------------------------------
    let big = r.tensor(4, 21, 13);
    let dbig = m.upload(&big)?;
    let want = T::resize_bilinear(&big, 7, 9, false);
    let got = m.resize_bilinear(&dbig, 7, 9, false)?;
    cmp("resize (align=F)", &got.download()?, &want.data, &mut report);
    let want = T::resize_bilinear(&big, 7, 9, true);
    let got = m.resize_bilinear(&dbig, 7, 9, true)?;
    cmp("resize (align=T)", &got.download()?, &want.data, &mut report);

    // --- deformable conv ---------------------------------------------------
    for &(k, pad) in [(1usize, 0usize), (3, 1), (7, 3)].iter() {
        let ic = 8;
        let oc = 6;
        let xi = r.tensor(ic, 11, 9);
        let off = r.tensor(2 * k * k, 11, 9);
        let mut msk = r.tensor(k * k, 11, 9);
        for v in msk.data.iter_mut() { *v = (*v + 1.0) * 0.5; } // sigmoid-ish, non-zero
        let wv = r.vec(oc * ic * k * k);
        let dxi = m.upload(&xi)?;
        let doff = m.upload(&off)?;
        let dmsk = m.upload(&msk)?;
        let dwv = m.upload(&Tensor::from_vec(oc, ic * k, k, wv.clone()))?;
        let want = deform_conv2d(&xi, &off, &msk, &wv, oc, k, pad);
        let got = m.deform_conv(&dxi, &doff, &dmsk, &DW { buf: dwv.buf.clone(), shape: vec![oc, ic, k, k] }, k, pad)?;
        cmp(&format!("deform k={} pad={}", k, pad), &got.download()?, &want.data, &mut report);
    }

    for line in &report {
        println!("{}", line);
    }
    let bad = report.iter().filter(|l| !l.contains("max|d|")).count();
    println!("---- {} op(s) compared, {} malformed", report.len(), bad);
    Ok(())
}

// ---------------------------------------------------------------------------
// Full forward graph on the GPU.
// ---------------------------------------------------------------------------

/// Every weight the graph needs, hoisted so the forward loop does no host-side
/// allocation and no per-call weight lookup.
pub struct CudaWeights {
    pub proj_w: DW, pub proj_b: DW,
    pub pe_norm_w: DW, pub pe_norm_b: DW,
    pub norm_w: Vec<DW>, pub norm_b: Vec<DW>,
    pub stages: Vec<CudaSwinStage>,
    pub squeeze: CudaBasicDecBlk,
    pub dec: CudaDecoder,
}

pub struct CudaLinear { pub w: DW, pub b: Option<DW> }

pub struct CudaLayerNorm { pub w: DW, pub b: DW }

pub struct CudaBn { pub scale: DW, pub shift: DW }

pub struct CudaMlp { pub fc1: CudaLinear, pub fc2: CudaLinear }

pub struct CudaAttn {
    pub qkv: CudaLinear,
    pub proj: CudaLinear,
    pub bias_table: DW,
    pub index: DW,        // f32 copy of the i64 relative position index
    pub heads: usize,
    pub dim: usize,
    pub scale: f32,
}

pub struct CudaSwinBlock { pub shift: usize, pub hd: usize }

pub struct CudaSwinStage {
    pub dim: usize,
    pub heads: usize,
    pub blocks: Vec<CudaSwinBlock>,
    pub ln1_w: Vec<DW>, pub ln1_b: Vec<DW>,
    pub ln2_w: Vec<DW>, pub ln2_b: Vec<DW>,
    pub attn: Vec<CudaAttn>,
    pub mlp: Vec<CudaMlp>,
    pub ds_norm_w: Option<DW>, pub ds_norm_b: Option<DW>,
    pub ds_red: Option<CudaLinear>,
}

pub struct CudaDeformConv {
    pub offset_w: DW, pub offset_b: DW,
    pub modulator_w: DW, pub modulator_b: DW,
    pub regular_w: DW,
    pub k: usize, pub pad: usize, pub ic: usize, pub oc: usize,
}

pub struct CudaAsppBranch { pub conv: CudaDeformConv, pub bn: CudaBn }

pub struct CudaAspp {
    pub aspp1: CudaAsppBranch,
    pub deforms: Vec<CudaAsppBranch>,
    pub gap_conv_w: DW, pub gap_bn: CudaBn,
    pub conv1_w: DW, pub bn1: CudaBn,
}

pub struct CudaBasicDecBlk {
    pub conv_in_w: DW, pub conv_in_b: DW, pub bn_in: CudaBn,
    pub aspp: CudaAspp,
    pub conv_out_w: DW, pub conv_out_b: DW, pub bn_out: CudaBn,
    pub ic: usize, pub ic_mid: usize, pub oc: usize,
}

pub struct CudaSimpleConvs { pub c1w: DW, pub c1b: DW, pub cow: DW, pub cob: DW, pub ic: usize, pub oc: usize }

pub struct CudaGdt { pub w: DW, pub b: DW, pub bn: CudaBn, pub attn_w: DW, pub attn_b: DW, pub ic: usize }

pub struct CudaDecoder {
    pub ipt: Vec<CudaSimpleConvs>,
    pub blocks: Vec<CudaBasicDecBlk>,
    pub lat2: DW, pub lat2b: DW,
    pub lat3: DW, pub lat3b: DW,
    pub lat4: DW, pub lat4b: DW,
    pub ms: Vec<(DW, DW)>,
    pub conv_out1_w: DW, pub conv_out1_b: DW,
    pub gdt: Vec<CudaGdt>,
}

impl CudaLinear {
    fn load(m: &CudaModel, name: &str) -> Result<CudaLinear, String> {
        let w = m.w(name)?;
        let bname = format!("{}.bias", name.trim_end_matches(".weight"));
        let b = match m.w(&bname) { Ok(d) => Some(d), Err(_) => None };
        Ok(CudaLinear { w, b })
    }
    /// (rows, ic) -> (rows, oc). A 1x1 conv over an (ic, 1, rows) view is the
    /// same arithmetic, with the accumulation order of forward.rs::Linear.
    fn apply(&self, m: &CudaModel, x: &DTensor, rows: usize) -> Result<DTensor, String> {
        let ic = self.w.ic();
        debug_assert_eq!(x.len(), rows * ic);
        m.linear(x, &self.w, self.b.as_ref(), rows)
    }
}

impl CudaLayerNorm {
    fn load(m: &CudaModel, prefix: &str) -> Result<CudaLayerNorm, String> {
        Ok(CudaLayerNorm {
            w: m.w(&format!("{}.weight", prefix))?,
            b: m.w(&format!("{}.bias", prefix))?,
        })
    }
    /// LayerNorm over the channel axis of a (C, 1, rows) token buffer.
    fn apply(&self, m: &CudaModel, x: &DTensor, rows: usize) -> Result<DTensor, String> {
        let c = self.w.shape[0];
        let v = DTensor { c, h: 1, w: rows, buf: x.buf.clone() };
        m.layernorm(&v, &self.w, &self.b, LN_EPS)
    }
}

impl CudaMlp {
    fn load(m: &CudaModel, prefix: &str) -> Result<CudaMlp, String> {
        Ok(CudaMlp {
            fc1: CudaLinear::load(m, &format!("{}.fc1.weight", prefix))?,
            fc2: CudaLinear::load(m, &format!("{}.fc2.weight", prefix))?,
        })
    }
    fn apply(&self, m: &CudaModel, x: &DTensor, rows: usize) -> Result<DTensor, String> {
        let h = self.fc1.apply(m, x, rows)?;
        let h = m.gelu(&h)?;
        self.fc2.apply(m, &h, rows)
    }
}

impl CudaAttn {
    fn load(m: &CudaModel, prefix: &str, dim: usize, heads: usize) -> Result<CudaAttn, String> {
        Ok(CudaAttn {
            qkv: CudaLinear::load(m, &format!("{}.qkv.weight", prefix))?,
            proj: CudaLinear::load(m, &format!("{}.proj.weight", prefix))?,
            bias_table: m.w(&format!("{}.relative_position_bias_table", prefix))?,
            index: m.w(&format!("{}.relative_position_index", prefix))?,
            heads,
            dim,
            scale: ((dim / heads) as f32).powf(-0.5),
        })
    }
}

impl CudaSwinStage {
    fn load(m: &CudaModel, stage: usize) -> Result<CudaSwinStage, String> {
        let dim = DIMS[stage];
        let heads = HEADS[stage];
        let mut ln1_w = Vec::new();
        let mut ln1_b = Vec::new();
        let mut ln2_w = Vec::new();
        let mut ln2_b = Vec::new();
        let mut attn = Vec::new();
        let mut mlp = Vec::new();
        let mut blocks = Vec::new();
        for b in 0..DEPTHS[stage] {
            let p = format!("bb.layers.{}.blocks.{}", stage, b);
            ln1_w.push(m.w(&format!("{}.norm1.weight", p))?);
            ln1_b.push(m.w(&format!("{}.norm1.bias", p))?);
            ln2_w.push(m.w(&format!("{}.norm2.weight", p))?);
            ln2_b.push(m.w(&format!("{}.norm2.bias", p))?);
            attn.push(CudaAttn::load(m, &format!("{}.attn", p), dim, heads)?);
            mlp.push(CudaMlp::load(m, &format!("{}.mlp", p))?);
            blocks.push(CudaSwinBlock {
                shift: if b % 2 == 0 { 0 } else { WINDOW / 2 },
                hd: dim / heads,
            });
        }
        let (ds_norm_w, ds_norm_b, ds_red) = if stage < 3 {
            (
                Some(m.w(&format!("bb.layers.{}.downsample.norm.weight", stage))?),
                Some(m.w(&format!("bb.layers.{}.downsample.norm.bias", stage))?),
                Some(CudaLinear::load(m, &format!("bb.layers.{}.downsample.reduction.weight", stage))?),
            )
        } else {
            (None, None, None)
        };
        Ok(CudaSwinStage { dim, heads, blocks, ln1_w, ln1_b, ln2_w, ln2_b, attn, mlp, ds_norm_w, ds_norm_b, ds_red })
    }
}

impl CudaDeformConv {
    fn load(m: &CudaModel, prefix: &str, k: usize, pad: usize) -> Result<CudaDeformConv, String> {
        let ow = m.w(&format!("{}.offset_conv.weight", prefix))?;
        let rw = m.w(&format!("{}.regular_conv.weight", prefix))?;
        Ok(CudaDeformConv {
            offset_w: ow.clone(),
            offset_b: m.w(&format!("{}.offset_conv.bias", prefix))?,
            modulator_w: m.w(&format!("{}.modulator_conv.weight", prefix))?,
            modulator_b: m.w(&format!("{}.modulator_conv.bias", prefix))?,
            regular_w: rw.clone(),
            k,
            pad,
            ic: ow.ic(),
            oc: rw.oc(),
        })
    }

    fn apply(&self, m: &CudaModel, x: &DTensor) -> Result<DTensor, String> {
        let off = m.conv(x, &self.offset_w, Some(&self.offset_b), self.k, false)?;
        let mask = m.conv(x, &self.modulator_w, Some(&self.modulator_b), self.k, false)?;
        let mask = m.double_sigmoid(&mask)?;
        m.deform_conv(x, &off, &mask, &self.regular_w, self.k, self.pad)
    }
}

impl CudaAsppBranch {
    fn load(m: &CudaModel, wts: &Weights, prefix: &str, k: usize, pad: usize) -> Result<CudaAsppBranch, String> {
        let (scale, shift) = m.affine_from_bn(wts, &format!("{}.bn", prefix))?;
        Ok(CudaAsppBranch {
            conv: CudaDeformConv::load(m, &format!("{}.atrous_conv", prefix), k, pad)?,
            bn: CudaBn { scale, shift },
        })
    }
    fn apply(&self, m: &CudaModel, x: &DTensor) -> Result<DTensor, String> {
        let y = self.conv.apply(m, x)?;
        let y = m.affine(&y, &self.bn.scale, &self.bn.shift)?;
        m.relu(&y)
    }
}

impl CudaAspp {
    fn load(m: &CudaModel, wts: &Weights, prefix: &str) -> Result<CudaAspp, String> {
        let gb = m.affine_from_bn(wts, &format!("{}.global_avg_pool.2", prefix))?;
        let b1 = m.affine_from_bn(wts, &format!("{}.bn1", prefix))?;
        Ok(CudaAspp {
            aspp1: CudaAsppBranch::load(m, wts, &format!("{}.aspp1", prefix), 1, 0)?,
            deforms: vec![
                CudaAsppBranch::load(m, wts, &format!("{}.aspp_deforms.0", prefix), 1, 0)?,
                CudaAsppBranch::load(m, wts, &format!("{}.aspp_deforms.1", prefix), 3, 1)?,
                CudaAsppBranch::load(m, wts, &format!("{}.aspp_deforms.2", prefix), 7, 3)?,
            ],
            gap_conv_w: m.w(&format!("{}.global_avg_pool.1.weight", prefix))?,
            gap_bn: CudaBn { scale: gb.0, shift: gb.1 },
            conv1_w: m.w(&format!("{}.conv1.weight", prefix))?,
            bn1: CudaBn { scale: b1.0, shift: b1.1 },
        })
    }

    fn apply(&self, m: &CudaModel, x: &DTensor) -> Result<DTensor, String> {
        let x1 = self.aspp1.apply(m, x)?;
        let mut parts: Vec<DTensor> = vec![x1];
        for d in self.deforms.iter() {
            parts.push(d.apply(m, x)?);
        }
        let g = m.global_avg_pool(x)?;
        let g = m.linear_1x1(&g, &self.gap_conv_w, None)?;
        let g = m.affine(&g, &self.gap_bn.scale, &self.gap_bn.shift)?;
        let g = m.relu(&g)?;
        let g_up = m.resize_bilinear(&g, x.h, x.w, true)?;
        parts.push(g_up);
        let refs: Vec<&DTensor> = parts.iter().collect();
        let cat = m.cat(&refs)?;
        let y = m.conv(&cat, &self.conv1_w, None, 1, false)?;
        let y = m.affine(&y, &self.bn1.scale, &self.bn1.shift)?;
        m.relu(&y)
    }
}

impl CudaBasicDecBlk {
    fn load(m: &CudaModel, wts: &Weights, prefix: &str) -> Result<CudaBasicDecBlk, String> {
        let cin = m.w(&format!("{}.conv_in.weight", prefix))?;
        let cout = m.w(&format!("{}.conv_out.weight", prefix))?;
        let bni = m.affine_from_bn(wts, &format!("{}.bn_in", prefix))?;
        let bno = m.affine_from_bn(wts, &format!("{}.bn_out", prefix))?;
        Ok(CudaBasicDecBlk {
            conv_in_w: cin.clone(),
            conv_in_b: m.w(&format!("{}.conv_in.bias", prefix))?,
            bn_in: CudaBn { scale: bni.0, shift: bni.1 },
            aspp: CudaAspp::load(m, wts, &format!("{}.dec_att", prefix))?,
            conv_out_w: cout.clone(),
            conv_out_b: m.w(&format!("{}.conv_out.bias", prefix))?,
            bn_out: CudaBn { scale: bno.0, shift: bno.1 },
            ic: cin.ic(),
            ic_mid: cin.oc(),
            oc: cout.oc(),
        })
    }

    fn apply(&self, m: &CudaModel, x: &DTensor) -> Result<DTensor, String> {
        let y = m.conv(x, &self.conv_in_w, Some(&self.conv_in_b), 3, false)?;
        let y = m.affine(&y, &self.bn_in.scale, &self.bn_in.shift)?;
        let y = m.relu(&y)?;
        let y = self.aspp.apply(m, &y)?;
        let y = m.conv(&y, &self.conv_out_w, Some(&self.conv_out_b), 3, false)?;
        m.affine(&y, &self.bn_out.scale, &self.bn_out.shift)
    }
}

impl CudaSimpleConvs {
    fn load(m: &CudaModel, prefix: &str) -> Result<CudaSimpleConvs, String> {
        let c1 = m.w(&format!("{}.conv1.weight", prefix))?;
        let co = m.w(&format!("{}.conv_out.weight", prefix))?;
        Ok(CudaSimpleConvs {
            c1w: c1.clone(),
            c1b: m.w(&format!("{}.conv1.bias", prefix))?,
            cow: co.clone(),
            cob: m.w(&format!("{}.conv_out.bias", prefix))?,
            ic: c1.ic(),
            oc: co.oc(),
        })
    }
    fn apply(&self, m: &CudaModel, x: &DTensor) -> Result<DTensor, String> {
        let y = m.conv(x, &self.c1w, Some(&self.c1b), 3, false)?;
        m.conv(&y, &self.cow, Some(&self.cob), 3, false)
    }
}

impl CudaGdt {
    fn load(m: &CudaModel, wts: &Weights, stage: usize) -> Result<CudaGdt, String> {
        let p = format!("decoder.gdt_convs_{}", stage);
        let w = m.w(&format!("{}.0.weight", p))?;
        let (scale, shift) = m.affine_from_bn(wts, &format!("{}.1", p))?;
        Ok(CudaGdt {
            w: w.clone(),
            b: m.w(&format!("{}.0.bias", p))?,
            bn: CudaBn { scale, shift },
            attn_w: m.w(&format!("decoder.gdt_convs_attn_{}.0.weight", stage))?,
            attn_b: m.w(&format!("decoder.gdt_convs_attn_{}.0.bias", stage))?,
            ic: w.ic(),
        })
    }
    fn apply(&self, m: &CudaModel, p: &DTensor) -> Result<DTensor, String> {
        let g = m.conv(p, &self.w, Some(&self.b), 3, false)?;
        let g = m.affine(&g, &self.bn.scale, &self.bn.shift)?;
        let g = m.relu(&g)?;
        let a = m.conv(&g, &self.attn_w, Some(&self.attn_b), 1, false)?;
        m.sigmoid(&a)
    }
}

impl CudaDecoder {
    fn load(m: &CudaModel, wts: &Weights) -> Result<CudaDecoder, String> {
        let lat = |p: &str| -> Result<(DW, DW), String> {
            Ok((m.w(&format!("{}.conv.weight", p))?, m.w(&format!("{}.conv.bias", p))?))
        };
        let l2 = lat("decoder.lateral_block2")?;
        let l3 = lat("decoder.lateral_block3")?;
        let l4 = lat("decoder.lateral_block4")?;
        Ok(CudaDecoder {
            ipt: vec![
                CudaSimpleConvs::load(m, "decoder.ipt_blk1")?,
                CudaSimpleConvs::load(m, "decoder.ipt_blk2")?,
                CudaSimpleConvs::load(m, "decoder.ipt_blk3")?,
                CudaSimpleConvs::load(m, "decoder.ipt_blk4")?,
                CudaSimpleConvs::load(m, "decoder.ipt_blk5")?,
            ],
            blocks: vec![
                CudaBasicDecBlk::load(m, wts, "decoder.decoder_block1")?,
                CudaBasicDecBlk::load(m, wts, "decoder.decoder_block2")?,
                CudaBasicDecBlk::load(m, wts, "decoder.decoder_block3")?,
                CudaBasicDecBlk::load(m, wts, "decoder.decoder_block4")?,
            ],
            lat2: l2.0, lat2b: l2.1,
            lat3: l3.0, lat3b: l3.1,
            lat4: l4.0, lat4b: l4.1,
            ms: vec![
                (m.w("decoder.conv_ms_spvn_2.weight")?, m.w("decoder.conv_ms_spvn_2.bias")?),
                (m.w("decoder.conv_ms_spvn_3.weight")?, m.w("decoder.conv_ms_spvn_3.bias")?),
                (m.w("decoder.conv_ms_spvn_4.weight")?, m.w("decoder.conv_ms_spvn_4.bias")?),
            ],
            conv_out1_w: m.w("decoder.conv_out1.0.weight")?,
            conv_out1_b: m.w("decoder.conv_out1.0.bias")?,
            gdt: vec![
                CudaGdt::load(m, wts, 2)?,
                CudaGdt::load(m, wts, 3)?,
                CudaGdt::load(m, wts, 4)?,
            ],
        })
    }
}

impl CudaModel {
    /// Resolve every weight the graph needs into one struct.
    pub fn load_weights(&self, wts: &Weights) -> Result<CudaWeights, String> {
        let mut norm_w = Vec::new();
        let mut norm_b = Vec::new();
        for s in 0..4 {
            norm_w.push(self.w(&format!("bb.norm{}.weight", s))?);
            norm_b.push(self.w(&format!("bb.norm{}.bias", s))?);
        }
        Ok(CudaWeights {
            proj_w: self.w("bb.patch_embed.proj.weight")?,
            proj_b: self.w("bb.patch_embed.proj.bias")?,
            pe_norm_w: self.w("bb.patch_embed.norm.weight")?,
            pe_norm_b: self.w("bb.patch_embed.norm.bias")?,
            norm_w,
            norm_b,
            stages: vec![
                CudaSwinStage::load(self, 0)?,
                CudaSwinStage::load(self, 1)?,
                CudaSwinStage::load(self, 2)?,
                CudaSwinStage::load(self, 3)?,
            ],
            squeeze: CudaBasicDecBlk::load(self, wts, "squeeze_module.0")?,
            dec: CudaDecoder::load(self, wts)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Backbone window assembly helpers. These are pure index arithmetic on the
// host-friendly (C, 1, rows) token layout, so they run as trivial gather/
// scatter kernels rather than as reworked convolutions.
// ---------------------------------------------------------------------------

impl CudaModel {
    /// rows x C tokens -> windows: (n_win, N, C) where N = WIN*WIN, reading
    /// from `padded` (hp x wp x C) with the cyclic shift already applied.
    /// Mirrors the window-partition loop in forward.rs::SwinBlock::apply.
    pub fn window_from_padded(
        &self,
        padded: &DTensor,   // (C, 1, hp*wp)
        out: &DTensor,      // (C, 1, n_win*N)
        hp: usize,
        wp: usize,
        win: usize,
    ) -> Result<(), String> {
        
        let mut aa = Args::new();
        aa.ptr(padded.buf.ptr);
        aa.ptr(out.buf.ptr);
        aa.i32(hp as i32);
        aa.i32(wp as i32);
        aa.i32(win as i32);
        aa.i32(padded.c as i32);
        aa.launch(self.module_of("lg_window_gather")?, "lg_window_gather", Launch::new((grid_for(padded.len()), 1, 1), (THREADS, 1, 1)))
    }

    /// windows: (n_win, N, C) -> padded (hp x wp x C); inverse of the loop above.
    pub fn window_to_padded(
        &self,
        wins: &DTensor,
        padded: &DTensor,
        hp: usize,
        wp: usize,
        win: usize,
    ) -> Result<(), String> {
        
        let mut aa = Args::new();
        aa.ptr(wins.buf.ptr);
        aa.ptr(padded.buf.ptr);
        aa.i32(hp as i32);
        aa.i32(wp as i32);
        aa.i32(win as i32);
        aa.i32(padded.c as i32);
        aa.launch(self.module_of("lg_window_scatter")?, "lg_window_scatter", Launch::new((grid_for(wins.len()), 1, 1), (THREADS, 1, 1)))
    }
}

// ---------------------------------------------------------------------------
// Window-attention launch wrappers (token layout).
// ---------------------------------------------------------------------------

impl CudaModel {
    /// Full window attention for one block: qkv linear, repack, scores,
    /// softmax, apply, then the output projection. `rows` = B_*N tokens,
    /// `n_win` = B_ windows, `mask` = (n_win, N, N) additive or None.
    pub fn window_attention(
        &self,
        attn: &CudaAttn,
        x: &DTensor,
        rows: usize,
        heads: usize,
        n_win: usize,
        mask: Option<&DTensor>,
    ) -> Result<DTensor, String> {
        let dim = attn.dim;
        let n = rows / n_win;               // tokens per window
        let hd = dim / heads;
        let qkv = attn.qkv.apply(self, x, rows)?;   // (3C, 1, rows)

        // repack to window-major (n_win, 3, heads, N, hd)
        let packed = self.dt(3 * heads * n, n_win, hd);
        {
            
        let mut aa = Args::new();
        aa.ptr(qkv.buf.ptr);
        aa.ptr(packed.buf.ptr);
        aa.i32(rows as i32);
        aa.i32(dim as i32);
        aa.i32(heads as i32);
        aa.i32(hd as i32);
        aa.i32(n as i32);
        aa.launch(self.module_of("lg_repack_attn")?, "lg_repack_attn", Launch::new((grid_for(3 * dim * rows), 1, 1), (THREADS, 1, 1)))?;
        }

        // scores -> (n_win * heads, N, N)
        let scores = self.dt(n, n_win * heads, n);
        {
            
        let mut aa = Args::new();
        aa.ptr(packed.buf.ptr);
        aa.ptr(attn.bias_table.buf.ptr);
        aa.ptr(attn.index.buf.ptr);
        aa.ptr(mask.map(|m| m.buf.ptr).unwrap_or(0));
        aa.ptr(scores.buf.ptr);
        aa.i32(heads as i32);
        aa.i32(hd as i32);
        aa.i32(n as i32);
        aa.f32((hd as f32).powf(-0.5) as f32);
        aa.launch(self.module_of("lg_attn_scores")?, "lg_attn_scores", Launch::new((grid_for(heads * n * n), n_win as u32, 1), (THREADS, 1, 1)))?;
        }

        // softmax in place, one block per (win, head) row-set
        {
            
        let row_blocks = n_win * heads * n;
        let mut aa = Args::new();
        aa.ptr(scores.buf.ptr);
        aa.i32(n as i32);
        aa.i32(n as i32);
        aa.launch(self.module_of("lg_softmax_rows")?, "lg_softmax_rows", Launch::new((row_blocks as u32, 1, 1), (256, 1, 1)).shared(256 * 4))?;
        }

        // weighted sum over v -> (C, 1, rows)
        let out = self.dt(dim, 1, rows);
        {
            
        let mut aa = Args::new();
        aa.ptr(scores.buf.ptr);
        aa.ptr(packed.buf.ptr);
        aa.ptr(out.buf.ptr);
        aa.i32(heads as i32);
        aa.i32(dim as i32);
        aa.i32(hd as i32);
        aa.i32(n as i32);
        aa.i32(rows as i32);
        aa.launch(self.module_of("lg_attn_apply")?, "lg_attn_apply", Launch::new((grid_for(rows * dim), 1, 1), (THREADS, 1, 1)))?;
        }

        attn.proj.apply(self, &out, rows)
    }
}

// ---------------------------------------------------------------------------
// SwinBlock and SwinStage on the GPU.
// ---------------------------------------------------------------------------

impl CudaModel {
    /// x: (C, 1, rows) tokens for an (h, w) grid with rows = h*w.
    /// The window partition, cyclic shift and crop are index shuffles; the
    /// arithmetic (norm, attention, mlp) is all kernels.
    pub fn swin_block(
        &self,
        blk: &CudaSwinBlock,
        attn: &CudaAttn,
        ln1: &CudaLayerNorm,
        ln2: &CudaLayerNorm,
        mlp: &CudaMlp,
        x: &DTensor,
        h: usize,
        w: usize,
        mask: Option<&DTensor>,
        dbg: Option<(&crate::engine::Dump, &str)>,
    ) -> Result<DTensor, String> {
        let c = x.c;
        let rows = x.w;
        debug_assert_eq!(rows, h * w);

        // norm1
        let y = ln1.apply(self, x, rows)?;
        // padding: hp, wp are multiples of WINDOW
        let pad_r = (WINDOW - w % WINDOW) % WINDOW;
        let pad_b = (WINDOW - h % WINDOW) % WINDOW;
        let hp = h + pad_b;
        let wp = w + pad_r;

        let padded = self.dt(c, 1, hp * wp);
        {
            
        let mut aa = Args::new();
        aa.ptr(y.buf.ptr);
        aa.ptr(padded.buf.ptr);
        aa.i32(h as i32);
        aa.i32(w as i32);
        aa.i32(hp as i32);
        aa.i32(wp as i32);
        aa.i32(c as i32);
        aa.launch(self.module_of("lg_pad_tokens")?, "lg_pad_tokens", Launch::new((grid_for(c * hp * wp), 1, 1), (THREADS, 1, 1)))?;
        }

        // cyclic shift
        let shifted = if blk.shift > 0 {
            let s = self.dt(c, 1, hp * wp);
            
        let mut aa = Args::new();
        aa.ptr(padded.buf.ptr);
        aa.ptr(s.buf.ptr);
        aa.i32(hp as i32);
        aa.i32(wp as i32);
        aa.i32(blk.shift as i32);
        aa.i32(c as i32);
        aa.launch(self.module_of("lg_roll_tokens")?, "lg_roll_tokens", Launch::new((grid_for(c * hp * wp), 1, 1), (THREADS, 1, 1)))?;
            s
        } else {
            padded.clone()
        };

        // window partition -> attention -> merge
        let n_win = (hp / WINDOW) * (wp / WINDOW);
        let n = WINDOW * WINDOW;
        let wins = self.dt(c, 1, n_win * n);
        self.window_from_padded(&shifted, &wins, hp, wp, WINDOW)?;
        let attn_out = self.window_attention(attn, &wins, n_win * n, attn.heads, n_win, mask)?;
        if let Some((d, nm)) = dbg {
            self.dump_named(d, &format!("{}_norm1", nm), &y)?;
            self.dump_named(d, &format!("{}_pad", nm), &shifted)?;
            self.dump_named(d, &format!("{}_wins", nm), &wins)?;
            self.dump_named(d, &format!("{}_attnout", nm), &attn_out)?;
        }
        let merged = self.dt(c, 1, hp * wp);
        self.window_to_padded(&attn_out, &merged, hp, wp, WINDOW)?;
        if let Some((d, nm)) = dbg {
            self.dump_named(d, &format!("{}_merged", nm), &merged)?;
        }

        // inverse shift
        let unshifted = if blk.shift > 0 {
            let s = self.dt(c, 1, hp * wp);
            let back = (hp - blk.shift) % hp;
        
        let mut aa = Args::new();
        aa.ptr(merged.buf.ptr);
        aa.ptr(s.buf.ptr);
        aa.i32(hp as i32);
        aa.i32(wp as i32);
        aa.i32(back as i32);
        aa.i32(c as i32);
        aa.launch(self.module_of("lg_roll_tokens")?, "lg_roll_tokens", Launch::new((grid_for(c * hp * wp), 1, 1), (THREADS, 1, 1)))?;
            s
        } else {
            merged.clone()
        };

        // crop + residual: out = x + crop(unshifted)
        let out = self.dt(c, 1, rows);
        {
            
        let mut aa = Args::new();
        aa.ptr(x.buf.ptr);
        aa.ptr(unshifted.buf.ptr);
        aa.ptr(out.buf.ptr);
        aa.i32(h as i32);
        aa.i32(w as i32);
        aa.i32(wp as i32);
        aa.i32(hp as i32);
        aa.i32(c as i32);
        aa.launch(self.module_of("lg_add_crop_tokens")?, "lg_add_crop_tokens", Launch::new((grid_for(rows * c), 1, 1), (THREADS, 1, 1)))?;
        }

        // mlp residual: out += mlp(norm2(out))
        let z = ln2.apply(self, &out, rows)?;
        let mm = mlp.apply(self, &z, rows)?;
        self.add_inplace(&out, &mm)
    }

    pub fn add_inplace(&self, dst: &DTensor, src: &DTensor) -> Result<DTensor, String> {
        
        let n = dst.len();
        let mut aa = Args::new();
        aa.ptr(dst.buf.ptr);
        aa.ptr(src.buf.ptr);
        aa.ptr(dst.buf.ptr);
        aa.i32(dst.len() as i32);
        aa.launch(self.module_of("lg_add")?, "lg_add", Launch::new((grid_for(n), 1, 1), (THREADS, 1, 1)))?;
        Ok(DTensor { c: dst.c, h: dst.h, w: dst.w, buf: dst.buf.clone() })
    }
}

impl CudaModel {
    /// 2x2 patch merge: (C,1,h*w) -> (4C,1,oh*ow), taps in the order
    /// (y0,x0),(y1,x0),(y0,x1),(y1,x1) with zero padding outside the source.
    pub fn patch_merge(&self, src: &DTensor, h: usize, w: usize) -> Result<DTensor, String> {
        let c = src.c;
        let oh = (h + 1) / 2;
        let ow = (w + 1) / 2;
        let dst = self.dt(4 * c, 1, oh * ow);
        
        let mut aa = Args::new();
        aa.ptr(src.buf.ptr);
        aa.ptr(dst.buf.ptr);
        aa.i32(h as i32);
        aa.i32(w as i32);
        aa.i32(c as i32);
        aa.i32(oh as i32);
        aa.i32(ow as i32);
        aa.launch(self.module_of("lg_patch_merge")?, "lg_patch_merge", Launch::new((grid_for(4 * c * oh * ow), 1, 1), (THREADS, 1, 1)))?;
        Ok(dst)
    }

    /// get_patches_batch(x, ph, pw) on the GPU.
    pub fn tile_patches(&self, src: &DTensor, ph: usize, pw: usize) -> Result<DTensor, String> {
        let nrow = (src.h + ph - 1) / ph;
        let ncol = (src.w + pw - 1) / pw;
        let dst = self.dt(src.c * nrow * ncol, ph, pw);
        
        let mut aa = Args::new();
        aa.ptr(src.buf.ptr);
        aa.ptr(dst.buf.ptr);
        aa.i32(src.c as i32);
        aa.i32(src.h as i32);
        aa.i32(src.w as i32);
        aa.i32(ph as i32);
        aa.i32(pw as i32);
        aa.launch(self.module_of("lg_tile_patches")?, "lg_tile_patches", Launch::new((grid_for(dst.len()), 1, 1), (THREADS, 1, 1)))?;
        Ok(dst)
    }

    /// In-place NCHW multiply by a (1,H,W) attention map.
    pub fn mul_broadcast_channel(&self, x: &DTensor, a: &DTensor) -> Result<(), String> {
        
        let mut aa = Args::new();
        aa.ptr(x.buf.ptr);
        aa.ptr(a.buf.ptr);
        aa.i32(x.c as i32);
        aa.i32(x.hw() as i32);
        aa.launch(self.module_of("lg_mul_broadcast")?, "lg_mul_broadcast", Launch::new((grid_for(x.len()), 1, 1), (THREADS, 1, 1)))
    }
}

// ---------------------------------------------------------------------------
// Patch embed, Swin stage and the full backbone.
// ---------------------------------------------------------------------------

/// Build the additive SW-MSA attention mask for a padded (hp, wp) grid.
/// This is tiny (n_win*N*N floats) and is computed on the host, exactly as
/// forward.rs::build_mask does, so both backends shift by the same -100.0.
pub fn build_mask(hp: usize, wp: usize) -> Vec<f32> {
    let mut img = vec![0.0f32; hp * wp];
    let bounds = |start: isize, end: isize, len: usize| -> Vec<usize> {
        let l = len as isize;
        let s = if start < 0 { (l + start).max(0) } else { start.min(l) };
        let e = if end == isize::MAX { l } else if end < 0 { l + end } else { end.min(l) };
        if s >= e { vec![] } else { (s..e).map(|v| v as usize).collect() }
    };
    let hslices: [(isize, isize); 3] = [
        (0, -(WINDOW as isize)),
        (-(WINDOW as isize), -((WINDOW / 2) as isize)),
        (-((WINDOW / 2) as isize), isize::MAX),
    ];
    let mut cnt = 0.0f32;
    for &(h0, h1) in hslices.iter() {
        for &(w0, w1) in hslices.iter() {
            for y in bounds(h0, h1, hp) {
                for x in bounds(w0, w1, wp) {
                    img[y * wp + x] = cnt;
                }
            }
            cnt += 1.0;
        }
    }
    let nwh = hp / WINDOW;
    let nww = wp / WINDOW;
    let n_win = nwh * nww;
    let n = WINDOW * WINDOW;
    let mut windows = vec![0.0f32; n_win * n];
    for wy in 0..nwh {
        for wx in 0..nww {
            let widx = wy * nww + wx;
            for i in 0..WINDOW {
                for j in 0..WINDOW {
                    windows[widx * n + i * WINDOW + j] = img[(wy * WINDOW + i) * wp + wx * WINDOW + j];
                }
            }
        }
    }
    let mut mask = vec![0.0f32; n_win * n * n];
    for wid in 0..n_win {
        let wrow = &windows[wid * n..(wid + 1) * n];
        for i in 0..n {
            for j in 0..n {
                mask[wid * n * n + i * n + j] = if wrow[i] - wrow[j] != 0.0 { -100.0 } else { 0.0 };
            }
        }
    }
    mask
}

impl CudaModel {
    /// Tokens (C,1,rows) -> NCHW (C,h,w).
    pub fn tokens_to_nchw(&self, t: &DTensor, h: usize, w: usize) -> Result<DTensor, String> {
        let dst = self.dt(t.c, h, w);
        
        let mut aa = Args::new();
        aa.ptr(t.buf.ptr);
        aa.ptr(dst.buf.ptr);
        aa.i32(t.c as i32);
        aa.i32((h * w) as i32);
        aa.launch(self.module_of("lg_tokens_to_nchw")?, "lg_tokens_to_nchw", Launch::new((grid_for(t.len()), 1, 1), (THREADS, 1, 1)))?;
        Ok(dst)
    }

    /// NCHW (C,h,w) -> tokens (C,1,h*w).
    pub fn nchw_to_tokens(&self, t: &DTensor) -> Result<DTensor, String> {
        let dst = self.dt(t.c, 1, t.hw());
        
        let mut aa = Args::new();
        aa.ptr(t.buf.ptr);
        aa.ptr(dst.buf.ptr);
        aa.i32(t.c as i32);
        aa.i32(t.hw() as i32);
        aa.launch(self.module_of("lg_nchw_to_tokens")?, "lg_nchw_to_tokens", Launch::new((grid_for(t.len()), 1, 1), (THREADS, 1, 1)))?;
        Ok(dst)
    }

    /// Patch embed: conv4x4 stride 4 from the 3-channel input, flatten to
    /// tokens (192, 1, h*w) and LayerNorm over the channel axis.
    /// Returns (tokens, ch, cw): tokens are (192, 1, ch*cw) and LayerNorm-ed.
    pub fn patch_embed(&self, wts: &CudaWeights, input: &DTensor) -> Result<(DTensor, usize, usize), String> {
        let tok = self.conv(input, &wts.proj_w, Some(&wts.proj_b), 4, true)?;
        let (ch, cw) = (tok.h, tok.w);
        let rows = ch * cw;
        let t = self.nchw_to_tokens(&tok)?;
        debug_assert_eq!(t.len(), rows * 192);
        let n = self.layernorm(&t, &wts.pe_norm_w, &wts.pe_norm_b, LN_EPS)?;
        Ok((n, ch, cw))
    }

    /// SwinStage: run every block, apply the per-stage LayerNorm, and (stages
    /// 0..2) produce the 2x2 patch-merged downsample. Returns the stage output
    /// tokens plus the downsample tokens and grid.
    #[allow(clippy::type_complexity)]
    pub fn swin_stage(
        &self,
        st: &CudaSwinStage,
        wts: &CudaWeights,
        stage: usize,
        tokens: &DTensor,
        h: usize,
        w: usize,
        tag: &str,
        dump: &crate::engine::Dump,
    ) -> Result<(DTensor, Option<DTensor>, usize, usize), String> {
        let hp = h.div_ceil(WINDOW) * WINDOW;
        let wp = w.div_ceil(WINDOW) * WINDOW;
        let mask_host = build_mask(hp, wp);
        let mask = self.upload(&Tensor::from_vec(mask_host.len(), 1, 1, mask_host.clone()))?;
        let mut cur = DTensor { c: tokens.c, h: 1, w: tokens.w, buf: tokens.buf.clone() };
        for b in 0..st.blocks.len() {
            let blk = &st.blocks[b];
            let m = if blk.shift > 0 { Some(&mask) } else { None };
            let ln1 = CudaLayerNorm { w: st.ln1_w[b].clone(), b: st.ln1_b[b].clone() };
            let ln2 = CudaLayerNorm { w: st.ln2_w[b].clone(), b: st.ln2_b[b].clone() };
            let nm = format!("{}_{}_b{}", tag, stage, b);
            cur = self.swin_block(blk, &st.attn[b], &ln1, &ln2, &st.mlp[b], &cur, h, w, m, Some((dump, &nm)))?;
            self.dump_named(dump, &format!("{}_{}_blocks_{}", tag, stage, b), &cur)?;
        }
        // The stage feature map is the block output with the stage norm applied,
        // but the DOWNSAMPLE must use the raw block output: engine.rs applies
        // rmbg.norm_w[stage] to the returned tokens after apply_dbg, while the
        // downsample inside apply_dbg sees the un-normed `cur`.
        let out = self.layernorm(&cur, &wts.norm_w[stage], &wts.norm_b[stage], LN_EPS)?;
        if stage >= 3 {
            return Ok((out, None, 0, 0));
        }
        let cdim = cur.c;
        let oh = (h + 1) / 2;
        let ow = (w + 1) / 2;
        let cat = self.patch_merge(&cur, h, w)?;
        let cat = DTensor { c: 4 * cdim, h: 1, w: oh * ow, buf: cat.buf.clone() };
        let nw = st.ds_norm_w.as_ref().expect("downsample norm");
        let nb = st.ds_norm_b.as_ref().expect("downsample norm bias");
        let normed = self.layernorm(&cat, nw, nb, LN_EPS)?;
        let red = st.ds_red.as_ref().expect("downsample reduction");
        let down = red.apply(self, &normed, oh * ow)?;
        self.dump_named(dump, &format!("{}_{}_downsample", tag, stage), &down)?;
        Ok((out, Some(down), oh, ow))
    }

    /// Backbone once: patch embed + 4 stages, returning the four per-stage
    /// feature maps as NCHW tensors. `stage_fn` is called with each stage's
    /// output tokens (for the optional verification dumps).
    pub fn backbone_once(
        &self,
        wts: &CudaWeights,
        input: &DTensor,
        tag: &str,
        dump: &crate::engine::Dump,
    ) -> Result<Vec<DTensor>, String> {
        let (tokens, mut h, mut w) = self.patch_embed(wts, input)?;
        self.dump_named(dump, &format!("{}_patch_embed_norm", tag), &tokens)?;
        let mut outs: Vec<DTensor> = Vec::new();
        let mut cur = tokens;
        for s in 0..4 {
            let (out, down, dh, dw) = self.swin_stage(&wts.stages[s], wts, s, &cur, h, w, tag, dump)?;
            outs.push(self.tokens_to_nchw(&out, h, w)?);
            if s < 3 {
                cur = down.expect("downsample");
                h = dh;
                w = dw;
            }
        }
        Ok(outs)
    }
}

// ---------------------------------------------------------------------------
// Encoder, decoder and the public entry point.
// ---------------------------------------------------------------------------

impl CudaModel {
    /// Write a device tensor into the CPU-style verification dump directory
    /// (same 12-byte (c,h,w) header + f32 payload), so cmp_dump.py can compare
    /// the two backends directly.
    pub fn dump_named(&self, dump: &crate::engine::Dump, name: &str, t: &DTensor) -> Result<(), String> {
        if !dump.wants(name) {
            return Ok(());
        }
        let data = t.download()?;
        let tt = Tensor::from_vec(t.c, t.h, t.w, data);
        dump.write_tensor(name, &tt)
    }

    /// get_patches_batch + resize: the raw image tiled at (oh, ow) and then
    /// bilinearly resized (align_corners=True) to that grid.
    pub fn ipt_input(&self, x: &DTensor, oh: usize, ow: usize) -> Result<DTensor, String> {
        let tiled = self.tile_patches(x, oh, ow)?;
        self.resize_bilinear(&tiled, oh, ow, true)
    }

    /// BiRefNet.forward_enc: two backbone passes plus the lateral concatenation
    /// and the cxt stack.
    pub fn forward_enc(&self, w: &CudaWeights, input: &DTensor, dump: &crate::engine::Dump) -> Result<Vec<DTensor>, String> {
        let full = self.backbone_once(w, input, "full", dump)?;
        let half_in = self.resize_bilinear(input, input.h / 2, input.w / 2, true)?;
        let half = self.backbone_once(w, &half_in, "half", dump)?;
        let mut out: Vec<DTensor> = Vec::new();
        for s in 0..4 {
            let up = self.resize_bilinear(&half[s], full[s].h, full[s].w, true)?;
            out.push(self.cat(&[&full[s], &up])?);
        }
        for s in 0..4 {
            self.dump_named(dump, &format!("bb_stage{}", s), &out[s])?;
        }
        let (h4, w4) = (out[3].h, out[3].w);
        let mut parts: Vec<DTensor> = Vec::new();
        for s in 0..3 {
            parts.push(self.resize_bilinear(&out[s], h4, w4, true)?);
        }
        let mut refs: Vec<&DTensor> = parts.iter().collect();
        refs.push(&out[3]);
        let cxt = self.cat(&refs)?;
        self.dump_named(dump, "cxt_x4", &cxt)?;
        Ok(vec![
            DTensor { c: out[0].c, h: out[0].h, w: out[0].w, buf: out[0].buf.clone() },
            DTensor { c: out[1].c, h: out[1].h, w: out[1].w, buf: out[1].buf.clone() },
            DTensor { c: out[2].c, h: out[2].h, w: out[2].w, buf: out[2].buf.clone() },
            cxt,
        ])
    }

    /// BiRefNet's decoder, mirroring engine.rs::forward_dump.
    pub fn decoder(&self, w: &CudaWeights, enc: &[DTensor], x: &DTensor, dump: &crate::engine::Dump) -> Result<DTensor, String> {
        let d = &w.dec;
        let x4 = w.squeeze.apply(self, &enc[3])?;
        self.dump_named(dump, "squeeze_module", &x4)?;
        let i5_in = self.ipt_input(x, x4.h, x4.w)?;
        self.dump_named(dump, "ipt_in5", &i5_in)?;
        let i5 = d.ipt[4].apply(self, &i5_in)?;
        self.dump_named(dump, "ipt_blk5", &i5)?;
        let x4_cat = self.cat(&[&x4, &i5])?;
        let p4 = d.blocks[3].apply(self, &x4_cat)?;
        self.dump_named(dump, "decoder_block4", &p4)?;
        let _m4 = self.conv(&p4, &d.ms[2].0, Some(&d.ms[2].1), 1, false)?;
        self.dump_named(dump, "pred0", &_m4)?;
        let attn4 = d.gdt[2].apply(self, &p4)?;
        self.mul_broadcast_channel(&p4, &attn4)?;
        let p3_in = self.resize_bilinear(&p4, enc[2].h, enc[2].w, true)?;
        let lat4 = self.conv(&enc[2], &d.lat4, Some(&d.lat4b), 1, false)?;
        self.add_inplace(&p3_in, &lat4)?;

        let i4 = d.ipt[3].apply(self, &self.ipt_input(x, enc[2].h, enc[2].w)?)?;
        let p3_cat = self.cat(&[&p3_in, &i4])?;
        let p3 = d.blocks[2].apply(self, &p3_cat)?;
        self.dump_named(dump, "decoder_block3", &p3)?;
        let _m3 = self.conv(&p3, &d.ms[1].0, Some(&d.ms[1].1), 1, false)?;
        self.dump_named(dump, "pred1", &_m3)?;
        let attn3 = d.gdt[1].apply(self, &p3)?;
        self.mul_broadcast_channel(&p3, &attn3)?;
        let p2_in = self.resize_bilinear(&p3, enc[1].h, enc[1].w, true)?;
        let lat3 = self.conv(&enc[1], &d.lat3, Some(&d.lat3b), 1, false)?;
        self.add_inplace(&p2_in, &lat3)?;

        let i3 = d.ipt[2].apply(self, &self.ipt_input(x, enc[1].h, enc[1].w)?)?;
        let p2_cat = self.cat(&[&p2_in, &i3])?;
        let p2 = d.blocks[1].apply(self, &p2_cat)?;
        self.dump_named(dump, "decoder_block2", &p2)?;
        let _m2 = self.conv(&p2, &d.ms[0].0, Some(&d.ms[0].1), 1, false)?;
        self.dump_named(dump, "pred2", &_m2)?;
        let attn2 = d.gdt[0].apply(self, &p2)?;
        self.mul_broadcast_channel(&p2, &attn2)?;
        let p1_in = self.resize_bilinear(&p2, enc[0].h, enc[0].w, true)?;
        let lat2 = self.conv(&enc[0], &d.lat2, Some(&d.lat2b), 1, false)?;
        self.add_inplace(&p1_in, &lat2)?;

        let i2 = d.ipt[1].apply(self, &self.ipt_input(x, enc[0].h, enc[0].w)?)?;
        let p1_cat = self.cat(&[&p1_in, &i2])?;
        let p1 = d.blocks[0].apply(self, &p1_cat)?;
        self.dump_named(dump, "decoder_block1", &p1)?;
        let up = self.resize_bilinear(&p1, x.h, x.w, true)?;
        let i1 = d.ipt[0].apply(self, &self.ipt_input(x, x.h, x.w)?)?;
        let last = self.cat(&[&up, &i1])?;
        let out = self.conv(&last, &d.conv_out1_w, Some(&d.conv_out1_b), 1, false)?;
        self.dump_named(dump, "conv_out1", &out)?;
        self.dump_named(dump, "pred3", &out)?;
        Ok(out)
    }
}

/// The whole model on the GPU: device weights plus the loaded graph.
pub struct CudaNet {
    pub m: CudaModel,
    pub w: CudaWeights,
    pub load_s: f64,
}

impl CudaNet {
    pub fn load(wts: &Weights) -> Result<CudaNet, String> {
        let t0 = std::time::Instant::now();
        let m = CudaModel::load(wts)?;
        let w = m.load_weights(wts)?;
        let load_s = t0.elapsed().as_secs_f64();
        eprintln!("cuda: graph weights resolved in {:.2}s", load_s);
        Ok(CudaNet { m, w, load_s })
    }

    /// Logits for a (3,1024,1024) preprocessed input. `dump_dir`/`steps`
    /// behave exactly like the CPU path's --dump-dir/--dump-step.
    pub fn forward(&self, input: &Tensor, dump_dir: &Option<String>, steps: &[String]) -> Result<Tensor, String> {
        let dump = crate::engine::Dump { dir: dump_dir.clone(), steps: steps.to_vec() };
        let dx = self.m.upload(input)?;
        let enc = self.m.forward_enc(&self.w, &dx, &dump)?;
        let logits = self.m.decoder(&self.w, &enc, &dx, &dump)?;
        let data = logits.download()?;
        Ok(Tensor::from_vec(logits.c, logits.h, logits.w, data))
    }
}
