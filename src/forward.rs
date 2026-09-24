
//! Forward graph: RMBG-2.0 (BiRefNet with a swin_v1_l backbone).
use rayon::prelude::*;

use crate::tensor::*;
use crate::weights::Weights;

/// Optional per-block activation dumps used to bisect backbone drift.
#[derive(Clone)]
pub struct BlockDump {
    pub dir: String,
    pub names: Vec<String>,
}

const WINDOW: usize = 12;
const DEPTHS: [usize; 4] = [2, 2, 18, 2];
const HEADS: [usize; 4] = [6, 12, 24, 48];
pub const DIMS: [usize; 4] = [192, 384, 768, 1536];
const LN_EPS: f32 = 1e-5;

struct Linear {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    ic: usize,
    oc: usize,
}

impl Linear {
    fn load(wts: &Weights, name: &str) -> Result<Linear, String> {
        let shape = wts.shape(name)?;
        if shape.len() != 2 {
            return Err(format!("{}: expected 2-D linear weight", name));
        }
        let (oc, ic) = (shape[0], shape[1]);
        let w = wts.get(name)?.to_vec();
        let bname = format!("{}.bias", name.trim_end_matches(".weight"));
        let b = wts.get(&bname).ok().map(|s| s.to_vec());
        Ok(Linear { w, b, ic, oc })
    }

    /// x: (rows, ic) -> (rows, oc)
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let rows = x.len() / self.ic;
        let mut out = vec![0.0f32; rows * self.oc];
        out.par_chunks_mut(self.oc).enumerate().for_each(|(r, orow)| {
            let xr = &x[r * self.ic..(r + 1) * self.ic];
            for oc in 0..self.oc {
                let wr = &self.w[oc * self.ic..(oc + 1) * self.ic];
                let mut acc = 0.0f32;
                for i in 0..self.ic {
                    acc += wr[i] * xr[i];
                }
                if let Some(b) = &self.b {
                    acc += b[oc];
                }
                orow[oc] = acc;
            }
        });
        out
    }
}

struct LayerNorm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl LayerNorm {
    fn load(wts: &Weights, prefix: &str) -> Result<LayerNorm, String> {
        Ok(LayerNorm {
            w: wts.get(&format!("{}.weight", prefix))?.to_vec(),
            b: wts.get(&format!("{}.bias", prefix))?.to_vec(),
        })
    }

    fn apply(&self, x: &mut [f32]) {
        let c = self.w.len();
        layernorm_slice(x, c, &self.w, &self.b, LN_EPS);
    }
}


struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn load(wts: &Weights, prefix: &str) -> Result<Mlp, String> {
        Ok(Mlp {
            fc1: Linear::load(wts, &format!("{}.fc1.weight", prefix))?,
            fc2: Linear::load(wts, &format!("{}.fc2.weight", prefix))?,
        })
    }

    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut h = self.fc1.apply(x);
        let mut t = Tensor::new(1, 1, h.len());
        t.data.copy_from_slice(&h);
        gelu_exact_inplace(&mut t);
        h.copy_from_slice(&t.data);
        self.fc2.apply(&h)
    }
}

struct WindowAttention {
    qkv: Linear,
    proj: Linear,
    bias_table: Vec<f32>,
    index: Vec<usize>,
    heads: usize,
    dim: usize,
    scale: f32,
}

impl WindowAttention {
    fn load(wts: &Weights, prefix: &str, dim: usize, heads: usize) -> Result<WindowAttention, String> {
        let table = wts.get(&format!("{}.relative_position_bias_table", prefix))?;
        let idx = wts.get(&format!("{}.relative_position_index", prefix))?;
        Ok(WindowAttention {
            qkv: Linear::load(wts, &format!("{}.qkv.weight", prefix))?,
            proj: Linear::load(wts, &format!("{}.proj.weight", prefix))?,
            bias_table: table.to_vec(),
            index: idx.iter().map(|v| *v as usize).collect(),
            heads,
            dim,
            scale: ((dim / heads) as f32).powf(-0.5),
        })
    }

    /// x: (B_, N, C); mask: optional (nW, N, N) additive mask.
    fn apply(&self, x: &[f32], n_win: usize, mask: Option<&[f32]>) -> Vec<f32> {
        let n = WINDOW * WINDOW;
        let b_ = x.len() / (n * self.dim);
        let qkv = self.qkv.apply(x);
        let hd = self.dim / self.heads;
        let mut out = vec![0.0f32; x.len()];
        // per (window, head) attention
        let scale = self.scale;
        let per_head: Vec<Vec<f32>> = (0..b_ * self.heads)
            .into_par_iter()
            .map(|bh| {
                let b = bh / self.heads;
                let h = bh % self.heads;
                let base = b * n * 3 * self.dim;
                let q_base = base + h * hd;
                let k_base = base + self.dim + h * hd;
                let v_base = base + 2 * self.dim + h * hd;
                // scores (N, N)
                let mut scores = vec![0.0f32; n * n];
                for i in 0..n {
                    let qi = &qkv[q_base + i * 3 * self.dim..q_base + i * 3 * self.dim + hd];
                    for j in 0..n {
                        let kj = &qkv[k_base + j * 3 * self.dim..k_base + j * 3 * self.dim + hd];
                        let mut acc = 0.0f32;
                        for d in 0..hd {
                            acc += qi[d] * kj[d];
                        }
                        scores[i * n + j] = acc * scale;
                    }
                }
                // relative position bias
                let nh = self.heads;
                for i in 0..n {
                    for j in 0..n {
                        let idx = self.index[i * n + j];
                        scores[i * n + j] += self.bias_table[idx * nh + h];
                    }
                }
                if let Some(m) = mask {
                    let w = b % n_win;
                    for i in 0..n {
                        for j in 0..n {
                            scores[i * n + j] += m[w * n * n + i * n + j];
                        }
                    }
                }
                // softmax
                for i in 0..n {
                    let row = &mut scores[i * n..(i + 1) * n];
                    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for v in row.iter_mut() {
                        *v = (*v - mx).exp();
                        sum += *v;
                    }
                    let inv = 1.0 / sum;
                    for v in row.iter_mut() {
                        *v *= inv;
                    }
                }
                // weighted sum over v
                let mut o = vec![0.0f32; n * hd];
                for i in 0..n {
                    for d in 0..hd {
                        let mut acc = 0.0f32;
                        for j in 0..n {
                            acc += scores[i * n + j] * qkv[v_base + j * 3 * self.dim + d];
                        }
                        o[i * hd + d] = acc;
                    }
                }
                o
            })
            .collect();
        // reassemble into (B_, N, C)
        for bh in 0..b_ * self.heads {
            let b = bh / self.heads;
            let h = bh % self.heads;
            let o = &per_head[bh];
            for i in 0..n {
                for d in 0..hd {
                    out[(b * n + i) * self.dim + h * hd + d] = o[i * hd + d];
                }
            }
        }
        self.proj.apply(&out)
    }
}

struct SwinBlock {
    norm1: LayerNorm,
    attn: WindowAttention,
    norm2: LayerNorm,
    mlp: Mlp,
    shift: usize,
}

impl SwinBlock {
    fn load(wts: &Weights, prefix: &str, dim: usize, heads: usize, shift: usize) -> Result<SwinBlock, String> {
        Ok(SwinBlock {
            norm1: LayerNorm::load(wts, &format!("{}.norm1", prefix))?,
            attn: WindowAttention::load(wts, &format!("{}.attn", prefix), dim, heads)?,
            norm2: LayerNorm::load(wts, &format!("{}.norm2", prefix))?,
            mlp: Mlp::load(wts, &format!("{}.mlp", prefix))?,
            shift,
        })
    }

    /// x: (B, H*W, C) tokens; returns same layout.
    fn apply(&self, x: &[f32], h: usize, w: usize, mask: Option<&[f32]>) -> Vec<f32> {
        let c = self.norm1.w.len();
        let mut y = x.to_vec();
        self.norm1.apply(&mut y);
        // (B, H, W, C) with padding to window multiples
        let pad_r = (WINDOW - w % WINDOW) % WINDOW;
        let pad_b = (WINDOW - h % WINDOW) % WINDOW;
        let hp = h + pad_b;
        let wp = w + pad_r;
        let mut padded = vec![0.0f32; hp * wp * c];
        for yy in 0..h {
            for xx in 0..w {
                let dst = (yy * wp + xx) * c;
                let src = (yy * w + xx) * c;
                padded[dst..dst + c].copy_from_slice(&y[src..src + c]);
            }
        }
        // cyclic shift
        let shifted = if self.shift > 0 {
            let mut s = vec![0.0f32; hp * wp * c];
            for yy in 0..hp {
                let sy = (yy + self.shift) % hp;
                for xx in 0..wp {
                    let sx = (xx + self.shift) % wp;
                    s[(yy * wp + xx) * c..(yy * wp + xx) * c + c]
                        .copy_from_slice(&padded[(sy * wp + sx) * c..(sy * wp + sx) * c + c]);
                }
            }
            s
        } else {
            padded
        };
        // window partition: (nW, WINDOW*WINDOW, C)
        let nwh = hp / WINDOW;
        let nww = wp / WINDOW;
        let n_win = nwh * nww;
        let n = WINDOW * WINDOW;
        let mut windows = vec![0.0f32; n_win * n * c];
        for wy in 0..nwh {
            for wx in 0..nww {
                let widx = wy * nww + wx;
                for i in 0..WINDOW {
                    for j in 0..WINDOW {
                        let src = ((wy * WINDOW + i) * wp + wx * WINDOW + j) * c;
                        let dst = (widx * n + i * WINDOW + j) * c;
                        windows[dst..dst + c].copy_from_slice(&shifted[src..src + c]);
                    }
                }
            }
        }
        let attn_windows = self.attn.apply(&windows, n_win, mask);
        // reverse
        let mut merged = vec![0.0f32; hp * wp * c];
        for wy in 0..nwh {
            for wx in 0..nww {
                let widx = wy * nww + wx;
                for i in 0..WINDOW {
                    for j in 0..WINDOW {
                        let src = (widx * n + i * WINDOW + j) * c;
                        let dst = ((wy * WINDOW + i) * wp + wx * WINDOW + j) * c;
                        merged[dst..dst + c].copy_from_slice(&attn_windows[src..src + c]);
                    }
                }
            }
        }
        let unshifted = if self.shift > 0 {
            let mut s = vec![0.0f32; hp * wp * c];
            for yy in 0..hp {
                let sy = (yy + hp - self.shift) % hp;
                for xx in 0..wp {
                    let sx = (xx + wp - self.shift) % wp;
                    s[(yy * wp + xx) * c..(yy * wp + xx) * c + c]
                        .copy_from_slice(&merged[(sy * wp + sx) * c..(sy * wp + sx) * c + c]);
                }
            }
            s
        } else {
            merged
        };
        // crop + residual
        let mut out = x.to_vec();
        for yy in 0..h {
            for xx in 0..w {
                for k in 0..c {
                    out[(yy * w + xx) * c + k] += unshifted[(yy * wp + xx) * c + k];
                }
            }
        }
        // mlp branch
        let mut z = out.clone();
        self.norm2.apply(&mut z);
        let m = self.mlp.apply(&z);
        out.par_iter_mut().zip(m.par_iter()).for_each(|(o, v)| *o += *v);
        out
    }
}

pub struct SwinStage {
    blocks: Vec<SwinBlock>,
    downsample_norm: Option<LayerNorm>,
    downsample: Option<Linear>,
}

impl SwinStage {
    pub fn load(wts: &Weights, stage: usize) -> Result<SwinStage, String> {
        let dim = DIMS[stage];
        let heads = HEADS[stage];
        let mut blocks = Vec::with_capacity(DEPTHS[stage]);
        for b in 0..DEPTHS[stage] {
            let prefix = format!("bb.layers.{}.blocks.{}", stage, b);
            let shift = if b % 2 == 0 { 0 } else { WINDOW / 2 };
            blocks.push(SwinBlock::load(wts, &prefix, dim, heads, shift)?);
        }
        let (downsample_norm, downsample) = if stage < 3 {
            (
                Some(LayerNorm::load(wts, &format!("bb.layers.{}.downsample.norm", stage))?),
                Some(Linear::load(wts, &format!("bb.layers.{}.downsample.reduction.weight", stage))?),
            )
        } else {
            (None, None)
        };
        Ok(SwinStage { blocks, downsample_norm, downsample })
    }

    /// x: (B, H*W, C) -> (out_tokens (B,H*W,C), H, W, down_tokens (B,H/2*W/2,2C), H/2, W/2)
    pub fn apply_dbg(
        &mut self,
        x: &[f32],
        h: usize,
        w: usize,
        b: usize,
        stage: usize,
        dbg: Option<(&BlockDump, usize, &str, usize)>,
    ) -> Result<(Vec<f32>, usize, usize, Vec<f32>, usize, usize), String> {
        // attention mask for the shifted blocks
        let hp = h.div_ceil(WINDOW) * WINDOW;
        let wp = w.div_ceil(WINDOW) * WINDOW;
        let mask = build_mask(hp, wp);
        let mut cur = x.to_vec();
        for (bi, blk) in self.blocks.iter().enumerate() {
            let m = if blk.shift > 0 { Some(mask.as_slice()) } else { None };
            cur = blk.apply(&cur, h, w, m);
            if let Some((bd, hash, tag, cdim)) = dbg {
                if hash == 0 && bd.names.iter().any(|n| n == &format!("{}_{}_blocks_{}", tag, stage, bi)) {
                    crate::tensor::dump_tokens_public(&bd.dir, &format!("{}_{}_blocks_{}", tag, stage, bi), &cur, cdim)?;
                }
            }
        }
        let out_tokens = cur;
        let (dn, dh, dw) = match (&self.downsample_norm, &self.downsample) {
            (Some(norm), Some(red)) => {
                // x0..x3 gather, cat along channels, LayerNorm(4C), Linear(4C->2C)
                let c = DIMS[0] * 4; // not used; compute from tokens
                let _ = c;
                let cdim = out_tokens.len() / (b * h * w);
                let oh = h.div_ceil(2);
                let ow = w.div_ceil(2);
                let mut cat = vec![0.0f32; b * oh * ow * 4 * cdim];
                for bi in 0..b {
                    for y in 0..oh {
                        for xx in 0..ow {
                            let y0 = y * 2;
                            let x0 = xx * 2;
                            let y1 = y0 + 1;
                            let x1 = x0 + 1;
                            let src = |yy: usize, xx2: usize| -> &[f32] {
                                let base = (bi * h * w + yy * w + xx2) * cdim;
                                &out_tokens[base..base + cdim]
                            };
                            let zero = vec![0.0f32; cdim];
                            let g0 = if y0 < h && x0 < w { src(y0, x0) } else { &zero };
                            let g1 = if y1 < h && x0 < w { src(y1, x0) } else { &zero };
                            let g2 = if y0 < h && x1 < w { src(y0, x1) } else { &zero };
                            let g3 = if y1 < h && x1 < w { src(y1, x1) } else { &zero };
                            let dst = (bi * oh * ow + y * ow + xx) * 4 * cdim;
                            cat[dst..dst + cdim].copy_from_slice(g0);
                            cat[dst + cdim..dst + 2 * cdim].copy_from_slice(g1);
                            cat[dst + 2 * cdim..dst + 3 * cdim].copy_from_slice(g2);
                            cat[dst + 3 * cdim..dst + 4 * cdim].copy_from_slice(g3);
                        }
                    }
                }
                norm.apply(&mut cat);
                let down = red.apply(&cat);
                if let Some((bd, hash, tag, _)) = dbg {
                    if hash == 0 && bd.names.iter().any(|n| n == &format!("{}_{}_downsample", tag, stage)) {
                        crate::tensor::dump_tokens_public(&bd.dir, &format!("{}_{}_downsample", tag, stage), &down, 2 * cdim)?;
                    }
                }
                (down, oh, ow)
            }
            _ => (out_tokens.clone(), 0, 0),
        };
        Ok((out_tokens, h, w, dn, dh, dw))
    }
}

/// Build the SW-MSA additive mask for a (hp, wp) padded grid.
fn build_mask(hp: usize, wp: usize) -> Vec<f32> {
    let mut img = vec![0.0f32; hp * wp];
    let h_slices = [
        (0isize, -(WINDOW as isize)),
        (-(WINDOW as isize), -((WINDOW / 2) as isize)),
        (-((WINDOW / 2) as isize), isize::MAX),
    ];
    let w_slices = h_slices;
    let mut cnt = 0.0f32;
    for &(h0, h1) in h_slices.iter() {
        for &(w0, w1) in w_slices.iter() {
            let hs = slice_bounds(h0, h1, hp);
            let ws = slice_bounds(w0, w1, wp);
            for y in hs.clone() {
                for x in ws.clone() {
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
    mask.par_chunks_mut(n * n).enumerate().for_each(|(wid, m)| {
        let wrow = &windows[wid * n..(wid + 1) * n];
        for i in 0..n {
            for j in 0..n {
                m[i * n + j] = if wrow[i] - wrow[j] != 0.0 { -100.0 } else { 0.0 };
            }
        }
    });
    mask
}

fn slice_bounds(start: isize, end: isize, len: usize) -> Vec<usize> {
    let l = len as isize;
    let s = if start < 0 { (l + start).max(0) } else { start.min(l) };
    let e = if end == isize::MAX { l } else if end < 0 { l + end } else { end.min(l) };
    if s >= e {
        vec![]
    } else {
        (s..e).map(|v| v as usize).collect()
    }
}

