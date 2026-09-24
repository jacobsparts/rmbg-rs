
//! Tensor helpers: NCHW torch-compatible ops on f32 buffers.
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    /// NCHW, len == c*h*w
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(c: usize, h: usize, w: usize) -> Tensor {
        Tensor { c, h, w, data: vec![0.0; c * h * w] }
    }

    pub fn from_vec(c: usize, h: usize, w: usize, data: Vec<f32>) -> Tensor {
        assert_eq!(data.len(), c * h * w, "bad tensor size");
        Tensor { c, h, w, data }
    }

    #[inline]
    pub fn hw(&self) -> usize {
        self.h * self.w
    }

}

/// out[co][s] = sum_ci x[ci][s] * w[co][ci]
pub fn conv1x1(x: &Tensor, w: &[f32], oc: usize, ic: usize, bias: Option<&[f32]>) -> Tensor {
    let hw = x.hw();
    let mut out = Tensor::new(oc, x.h, x.w);
    out.data.par_chunks_mut(hw).enumerate().for_each(|(co, och)| {
        let wc = &w[co * ic..co * ic + ic];
        for (i, wv) in wc.iter().enumerate() {
            if *wv == 0.0 {
                continue;
            }
            let src = &x.data[i * hw..i * hw + hw];
            for (o, s) in och.iter_mut().zip(src) {
                *o += *wv * *s;
            }
        }
        if let Some(b) = bias {
            let bv = b[co];
            for o in och.iter_mut() {
                *o += bv;
            }
        }
    });
    out
}

/// kxk convolution, stride 1, given padding (dilation 1).
pub fn conv_kxk(
    x: &Tensor,
    w: &[f32],
    oc: usize,
    ic: usize,
    k: usize,
    bias: Option<&[f32]>,
) -> Tensor {
    let (h, wd) = (x.h, x.w);
    let hw = h * wd;
    let mut out = Tensor::new(oc, h, wd);
    out.data.par_chunks_mut(wd * h).enumerate().for_each(|(co, och)| {
        for y in 0..h {
            for ox in 0..wd {
                let mut acc = 0.0f32;
                for ky in 0..k {
                    let iy = y as isize + ky as isize - (k / 2) as isize;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for kx in 0..k {
                        let ix = ox as isize + kx as isize - (k / 2) as isize;
                        if ix < 0 || ix >= wd as isize {
                            continue;
                        }
                        for ci in 0..ic {
                            let wv = w[((co * ic + ci) * k + ky) * k + kx];
                            if wv == 0.0 {
                                continue;
                            }
                            acc += wv * x.data[ci * hw + iy as usize * wd + ix as usize];
                        }
                    }
                }
                if let Some(b) = bias {
                    acc += b[co];
                }
                och[y * wd + ox] = acc;
            }
        }
    });
    out
}

/// 4x4 stride-4 patch-embed convolution (no padding).
pub fn conv4x4_stride4(x: &Tensor, w: &[f32], oc: usize, ic: usize, bias: Option<&[f32]>) -> Tensor {
    let oh = x.h / 4;
    let ow = x.w / 4;
    let mut out = Tensor::new(oc, oh, ow);
    let hw = x.hw();
    out.data.par_chunks_mut(oh * ow).enumerate().for_each(|(co, och)| {
        for y in 0..oh {
            for xo in 0..ow {
                let mut acc = 0.0f32;
                for ci in 0..ic {
                    for ky in 0..4 {
                        for kx in 0..4 {
                            let v = x.data[ci * hw + (y * 4 + ky) * x.w + xo * 4 + kx];
                            acc += v * w[((co * ic + ci) * 4 + ky) * 4 + kx];
                        }
                    }
                }
                if let Some(b) = bias {
                    acc += b[co];
                }
                och[y * ow + xo] = acc;
            }
        }
    });
    out
}

/// LayerNorm over the last dim (channels) of a (n, c) row buffer.
pub fn layernorm_inplace(data: &mut [f32], n: usize, c: usize, w: &[f32], b: &[f32], eps: f32) {
    data.par_chunks_mut(c).take(n).for_each(|row| {
        let mean = row.iter().sum::<f32>() / c as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..c {
            row[i] = (row[i] - mean) * inv * w[i] + b[i];
        }
    });
}

pub fn layernorm_slice(data: &mut [f32], c: usize, w: &[f32], b: &[f32], eps: f32) {
    let n = data.len() / c;
    layernorm_inplace(data, n, c, w, b, eps);
}

pub fn apply_affine_inplace(x: &mut Tensor, scale: &[f32], shift: &[f32]) {
    let hw = x.hw();
    x.data.par_chunks_mut(hw).enumerate().for_each(|(c, ch)| {
        let s = scale[c];
        let b = shift[c];
        for v in ch.iter_mut() {
            *v = *v * s + b;
        }
    });
}

pub fn relu_inplace(x: &mut Tensor) {
    x.data.par_iter_mut().for_each(|v| {
        if *v < 0.0 {
            *v = 0.0;
        }
    });
}

pub fn sigmoid_inplace(x: &mut Tensor) {
    x.data.par_iter_mut().for_each(|v| {
        *v = 1.0 / (1.0 + (-*v).exp());
    });
}

/// tanh-approximation GELU (the default torch.nn.GELU approximate='none' is
/// the erf form; the reference uses nn.GELU() whose default is the exact erf
/// formulation, so this function implements that).
pub fn gelu_exact_inplace(x: &mut Tensor) {
    x.data.par_iter_mut().for_each(|v| {
        let t = *v;
        *v = 0.5 * t * (1.0 + erf(t / std::f32::consts::SQRT_2));
    });
}

/// Abramowitz & Stegun 7.1.26-based erf (max abs error ~1.5e-7).
pub fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Bilinear resize (NCHW single image), both torch align_corners modes.
pub fn resize_bilinear(x: &Tensor, oh: usize, ow: usize, align_corners: bool) -> Tensor {
    let mut out = Tensor::new(x.c, oh, ow);
    let (ih, iw) = (x.h, x.w);
    let (rh, rw): (f32, f32) = if align_corners {
        (
            if oh > 1 { (ih - 1) as f32 / (oh - 1) as f32 } else { 0.0 },
            if ow > 1 { (iw - 1) as f32 / (ow - 1) as f32 } else { 0.0 },
        )
    } else {
        (ih as f32 / oh as f32, iw as f32 / ow as f32)
    };
    let src_h: Vec<(usize, usize, f32)> = (0..oh)
        .map(|y| {
            let fy = if align_corners {
                y as f32 * rh
            } else {
                ((y as f32 + 0.5) * rh - 0.5).max(0.0)
            };
            let y0 = fy.floor();
            let y0i = (y0 as usize).min(ih - 1);
            let y1i = (y0i + 1).min(ih - 1);
            (y0i, y1i, fy - y0)
        })
        .collect();
    let src_w: Vec<(usize, usize, f32)> = (0..ow)
        .map(|xw| {
            let fx = if align_corners {
                xw as f32 * rw
            } else {
                ((xw as f32 + 0.5) * rw - 0.5).max(0.0)
            };
            let x0 = fx.floor();
            let x0i = (x0 as usize).min(iw - 1);
            let x1i = (x0i + 1).min(iw - 1);
            (x0i, x1i, fx - x0)
        })
        .collect();
    out.data.par_chunks_mut(oh * ow).enumerate().for_each(|(c, ch)| {
        let src = &x.data[c * ih * iw..(c + 1) * ih * iw];
        for y in 0..oh {
            let (y0, y1, wy) = src_h[y];
            for xw in 0..ow {
                let (x0, x1, wx) = src_w[xw];
                let v00 = src[y0 * iw + x0];
                let v01 = src[y0 * iw + x1];
                let v10 = src[y1 * iw + x0];
                let v11 = src[y1 * iw + x1];
                let top = v00 + (v01 - v00) * wx;
                let bot = v10 + (v11 - v10) * wx;
                ch[y * ow + xw] = top + (bot - top) * wy;
            }
        }
    });
    out
}

/// Adaptive average pool to 1x1: per-channel mean.
pub fn global_avg_pool(x: &Tensor) -> Vec<f32> {
    let hw = x.hw();
    x.data.par_chunks(hw).map(|ch| ch.iter().sum::<f32>() / hw as f32).collect()
}

/// Concatenate NCHW tensors along channels (they must share h,w).
pub fn cat_channels(parts: &[&Tensor]) -> Tensor {
    let c: usize = parts.iter().map(|t| t.c).sum();
    let h = parts[0].h;
    let w = parts[0].w;
    let mut out = Tensor::new(c, h, w);
    let hw = h * w;
    let mut co = 0;
    for t in parts {
        assert_eq!(t.h, h, "cat_channels height mismatch");
        assert_eq!(t.w, w, "cat_channels width mismatch");
        for ci in 0..t.c {
            out.data[(co + ci) * hw..(co + ci + 1) * hw]
                .copy_from_slice(&t.data[ci * hw..(ci + 1) * hw]);
        }
        co += t.c;
    }
    out
}

pub fn add_inplace(dst: &mut Tensor, src: &Tensor) {
    assert_eq!(dst.data.len(), src.data.len(), "add shape mismatch");
    dst.data.par_iter_mut().zip(src.data.par_iter()).for_each(|(d, s)| *d += *s);
}

/// Multiply a (C,H,W) tensor by a (1,H,W) attention map, broadcasting the
/// single channel over all C channels (BiRefNet's `p = p * attn`).
pub fn mul_broadcast_channel_inplace(dst: &mut Tensor, attn: &Tensor) {
    assert_eq!(attn.c, 1, "attention must have a single channel");
    let hw = dst.hw();
    assert_eq!(attn.hw(), hw, "attention spatial mismatch");
    dst.data.par_chunks_mut(hw).for_each(|ch| {
        for (v, a) in ch.iter_mut().zip(attn.data.iter()) {
            *v *= *a;
        }
    });
}

/// Write a token-layout activation (rows x c, no spatial dims) as a dump file
/// with a (c, 1, rows) header, matching the engine's Tensor dump format.
pub fn dump_tokens_public(dir: &str, name: &str, tokens: &[f32], c: usize) -> Result<(), String> {
    use std::io::Write;
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let path = format!("{}/{}.f32", dir, name);
    let mut f = std::fs::File::create(&path).map_err(|e| e.to_string())?;
    let hw = tokens.len() / c;
    f.write_all(&(c as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(1u32).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(&(hw as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for v in tokens {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&bytes).map_err(|e| e.to_string())?;
    eprintln!("dump {} {} tokens x {}", name, hw, c);
    Ok(())
}
