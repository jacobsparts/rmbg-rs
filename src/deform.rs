
//! torchvision-compatible modulated deformable convolution (DCNv2).
use rayon::prelude::*;

use crate::tensor::Tensor;

/// Bilinear sample of a (C,H,W) plane at (y, x) with zero padding outside.
#[inline]
fn bilinear(
    plane: &[f32],
    h: usize,
    w: usize,
    y: f32,
    x: f32,
) -> f32 {
    let y0f = y.floor();
    let x0f = x.floor();
    let y0 = y0f as isize;
    let x0 = x0f as isize;
    let ly = y - y0f;
    let lx = x - x0f;
    let mut acc = 0.0f32;
    for (dy, wy) in [(0isize, 1.0 - ly), (1, ly)] {
        let yy = y0 + dy;
        if yy < 0 || yy >= h as isize || wy == 0.0 {
            continue;
        }
        for (dx, wx) in [(0isize, 1.0 - lx), (1, lx)] {
            let xx = x0 + dx;
            if xx < 0 || xx >= w as isize || wx == 0.0 {
                continue;
            }
            acc += wy * wx * plane[yy as usize * w + xx as usize];
        }
    }
    acc
}

/// out[n, co, oy, ox] = bias[co] + sum_{ci,ky,kx} w[co,ci,ky,kx] * mask[n,ky,kx]
///                     * x[n, ci, oy + ky - pad + off_y, ox + kx - pad + off_x]
/// stride is assumed 1; padding is `pad` on all sides.
pub fn deform_conv2d(
    x: &Tensor,
    offset: &Tensor,
    mask: &Tensor,
    weight: &[f32],
    oc: usize,
    k: usize,
    pad: usize,
) -> Tensor {
    let ic = x.c;
    let (h, w) = (x.h, x.w);
    let hw = h * w;
    let khw = k * k;
    let mut out = Tensor::new(oc, h, w);
    out.data
        .par_chunks_mut(hw)
        .enumerate()
        .for_each(|(co, och)| {
            let wco = &weight[co * ic * khw..(co + 1) * ic * khw];
            let n_off = offset.c;
            debug_assert_eq!(n_off, 2 * khw);
            for oy in 0..h {
                for ox in 0..w {
                    let p = oy * w + ox;
                    let mut acc = 0.0f32;
                    for ky in 0..k {
                        for kx in 0..k {
                            let kk = ky * k + kx;
                            let off_y = offset.data[(2 * kk) * hw + p];
                            let off_x = offset.data[(2 * kk + 1) * hw + p];
                            let m = mask.data[kk * hw + p];
                            if m == 0.0 {
                                continue;
                            }
                            let sy = oy as f32 + ky as f32 - pad as f32 + off_y;
                            let sx = ox as f32 + kx as f32 - pad as f32 + off_x;
                            let a = m;
                            for ci in 0..ic {
                                let wv = wco[(ci * khw) + kk];
                                if wv == 0.0 {
                                    continue;
                                }
                                let plane = &x.data[ci * hw..(ci + 1) * hw];
                                let v = bilinear(plane, h, w, sy, sx);
                                acc += wv * a * v;
                            }
                        }
                    }
                    och[p] = acc;
                }
            }
        });
    out
}
