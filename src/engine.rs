
//! The rmbg-rs CPU implementation of the RMBG-2.0 forward graph.
use crate::graph::Decoder;
use crate::tensor::*;
use crate::weights::Weights;

pub struct Rmbg {
    pub stages: Vec<crate::forward::SwinStage>,
    pub pe_norm_w: Vec<f32>,
    pub pe_norm_b: Vec<f32>,
    pub proj_w: Vec<f32>,
    pub proj_b: Vec<f32>,
    pub norm_w: Vec<Vec<f32>>,
    pub norm_b: Vec<Vec<f32>>,
    pub squeeze: crate::graph::BasicDecBlk,
    pub decoder: Decoder,
}

impl Rmbg {
    pub fn load(wts: &Weights) -> Result<Rmbg, String> {
        let mut stages = Vec::with_capacity(4);
        for s in 0..4 {
            stages.push(crate::forward::SwinStage::load(wts, s)?);
        }
        let mut norm_w = Vec::new();
        let mut norm_b = Vec::new();
        for s in 0..4 {
            norm_w.push(wts.get(&format!("bb.norm{}.weight", s))?.to_vec());
            norm_b.push(wts.get(&format!("bb.norm{}.bias", s))?.to_vec());
        }
        Ok(Rmbg {
            stages,
            pe_norm_w: wts.get("bb.patch_embed.norm.weight")?.to_vec(),
            pe_norm_b: wts.get("bb.patch_embed.norm.bias")?.to_vec(),
            proj_w: wts.get_shaped("bb.patch_embed.proj.weight", &[192, 3, 4, 4])?.to_vec(),
            proj_b: wts.get_shaped("bb.patch_embed.proj.bias", &[192])?.to_vec(),
            norm_w,
            norm_b,
            squeeze: crate::graph::BasicDecBlk::load(wts, "squeeze_module.0")?,
            decoder: Decoder::load(wts)?,
        })
    }
}

/// Collector for intermediate activations (verification support).
pub struct Dump {
    pub dir: Option<String>,
    pub steps: Vec<String>,
}

impl Dump {
    /// Per-block dump names explicitly requested via --dump-step, e.g.
    /// "full_blocks_2_5" / "half_blocks_0_1" / "full_downsample_1".
    fn names_for_blocks(&self) -> Vec<String> {
        self.steps
            .iter()
            .filter(|s| s.starts_with("full_") || s.starts_with("half_"))
            .cloned()
            .collect()
    }

    /// Public accessors so the CUDA backend can write the same verification
    /// dumps the CPU path produces.
    #[allow(dead_code)] // used only by the cuda build
    pub fn wants(&self, name: &str) -> bool {
        self.want(name)
    }

    #[allow(dead_code)] // used only by the cuda build
    pub fn write_tensor(&self, name: &str, t: &Tensor) -> Result<(), String> {
        self.write(name, t)
    }

    fn want(&self, name: &str) -> bool {
        self.dir.is_some() && (self.steps.is_empty() || self.steps.iter().any(|s| s == name))
    }

    fn write(&self, name: &str, t: &Tensor) -> Result<(), String> {
        let dir = match &self.dir {
            Some(d) => d.clone(),
            None => return Ok(()),
        };
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        use std::io::Write;
        let path = format!("{}/{}.f32", dir, name);
        let mut f = std::fs::File::create(&path).map_err(|e| e.to_string())?;
        f.write_all(&(t.c as u32).to_le_bytes()).map_err(|e| e.to_string())?;
        f.write_all(&(t.h as u32).to_le_bytes()).map_err(|e| e.to_string())?;
        f.write_all(&(t.w as u32).to_le_bytes()).map_err(|e| e.to_string())?;
        let mut bytes = Vec::with_capacity(t.data.len() * 4);
        for v in &t.data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&bytes).map_err(|e| e.to_string())?;
        eprintln!("dump {} {}x{}x{}", name, t.c, t.h, t.w);
        Ok(())
    }
}

/// Run the backbone once and return the four per-stage feature maps.
fn backbone_once(rmbg: &mut Rmbg, x: &Tensor, tag: &str, dump: &Dump) -> Result<[Tensor; 4], String> {
    let blk_names: Vec<String> = dump.names_for_blocks();
    let tok = conv4x4_stride4(x, &rmbg.proj_w, 192, 3, Some(&rmbg.proj_b));
    let (ch, cw) = (tok.h, tok.w);
    let mut tokens = vec![0.0f32; ch * cw * 192];
    for y in 0..ch {
        for xx in 0..cw {
            for c in 0..192 {
                tokens[(y * cw + xx) * 192 + c] = tok.data[c * ch * cw + y * cw + xx];
            }
        }
    }
    layernorm_slice(&mut tokens, 192, &rmbg.pe_norm_w, &rmbg.pe_norm_b, 1e-5);
    if dump.want(&format!("{}_patch_embed_norm", tag)) {
        crate::tensor::dump_tokens_public(
            dump.dir.as_deref().unwrap_or("."),
            &format!("{}_patch_embed_norm", tag),
            &tokens,
            192,
        )?;
    }
    let (mut h, mut w) = (ch, cw);
    let mut outs: Vec<Tensor> = Vec::new();
    for s in 0..4 {
        let bd = crate::forward::BlockDump { dir: dump.dir.clone().unwrap_or_default(), names: blk_names.clone() };
        let (out_tokens, oh, ow, down, dwh, dww) =
            rmbg.stages[s].apply_dbg(&tokens, h, w, 1, s, Some((&bd, 0, tag, crate::forward::DIMS[s])))?;
        let mut t = out_tokens;
        let c = crate::forward::DIMS[s];
        layernorm_slice(&mut t, c, &rmbg.norm_w[s], &rmbg.norm_b[s], 1e-5);
        let mut img = Tensor::new(c, oh, ow);
        for y in 0..oh {
            for xx in 0..ow {
                for cc in 0..c {
                    img.data[cc * oh * ow + y * ow + xx] = t[(y * ow + xx) * c + cc];
                }
            }
        }
        outs.push(img);
        if s < 3 {
            tokens = down;
            h = dwh;
            w = dww;
        }
    }
    match <[Tensor; 4]>::try_from(outs) {
        Ok(v) => Ok(v),
        Err(_) => Err("stage count".to_string()),
    }
}

/// BiRefNet.forward_enc
fn forward_enc(rmbg: &mut Rmbg, x: &Tensor, dump: &Dump) -> Result<[Tensor; 4], String> {
    let full = backbone_once(rmbg, x, "full", dump)?;
    let half_in = resize_bilinear(x, x.h / 2, x.w / 2, true);
    if dump.want("half_input") {
        dump.write("half_input", &half_in)?;
    }
    if dump.want("norm_input") {
        dump.write("norm_input", x)?;
    }
    let half = backbone_once(rmbg, &half_in, "half", dump)?;
    let mut out: [Tensor; 4] = [
        full[0].clone(),
        full[1].clone(),
        full[2].clone(),
        full[3].clone(),
    ];
    for s in 0..4 {
        let up = resize_bilinear(&half[s], full[s].h, full[s].w, true);
        out[s] = cat_channels(&[&full[s], &up]);
        if dump.want(&format!("bb_stage{}", s)) {
            dump.write(&format!("bb_stage{}", s), &out[s])?;
        }
    }
    // cxt: cxt_num == 3, so x1, x2 and x3 (the *concatenated* lateral features,
    // 3072/1536/768 wide) are bilinearly upsampled to x4's grid and prepended
    // to x4 (3072 wide) in that order -> 2688 + 3072 = 5760 channels.
    let h4 = out[3].h;
    let w4 = out[3].w;
    let mut pre: Vec<Tensor> = Vec::new();
    for s in 0..3 {
        pre.push(resize_bilinear(&out[s], h4, w4, true));
    }
    let mut parts: Vec<&Tensor> = pre.iter().collect();
    parts.push(&out[3]);
    out[3] = cat_channels(&parts);
    if dump.want("cxt_x4") {
        dump.write("cxt_x4", &out[3])?;
    }
    Ok(out)
}

/// Reference Decoder.get_patches_batch(x, p) with split=True: split the image
/// into tiles of p's (H, W) - first by width, then by height - and concatenate
/// the tiles along the channel axis. The result has 3*(H/ph)*(W/pw) channels
/// and is then bilinearly resized (align_corners=True) to (ph, pw).
pub fn get_patches_batch(x: &Tensor, ph: usize, pw: usize) -> Tensor {
    let (h, w) = (x.h, x.w);
    let nrow = (h + ph - 1) / ph;
    let ncol = (w + pw - 1) / pw;
    let mut out = Tensor::new(x.c * nrow * ncol, ph, pw);
    // torch order: split by width into ncol columns, each column split by
    // height into nrow rows; tiles are concatenated along channels in the
    // order (cw, ch), and the image channels are the innermost dimension of
    // each tile, so out channel = (cw * nrow + ch) * x.c + cy.
    for cw in 0..ncol {
        for ch in 0..nrow {
            for cy in 0..x.c {
                let co = (cw * nrow + ch) * x.c + cy;
                for y in 0..ph {
                    let iy = ch * ph + y;
                    if iy >= h {
                        break;
                    }
                    for xx in 0..pw {
                        let ix = cw * pw + xx;
                        if ix >= w {
                            break;
                        }
                        out.data[co * ph * pw + y * pw + xx] = x.data[cy * h * w + iy * w + ix];
                    }
                }
            }
        }
    }
    out
}

/// ipt blocks consume the tiled image resized to the stage's feature grid.
fn ipt_input(x: &Tensor, oh: usize, ow: usize) -> Tensor {
    let tiled = get_patches_batch(x, oh, ow);
    resize_bilinear(&tiled, oh, ow, true)
}

/// BiRefNet.forward (eval): returns the 1024x1024 mask logits.
pub fn forward_dump(
    rmbg: &mut Rmbg,
    input: &Tensor,
    dump_dir: &Option<String>,
    steps: &[String],
) -> Result<Tensor, String> {
    let dump = Dump { dir: dump_dir.clone(), steps: steps.to_vec() };
    if dump.want("input") {
        dump.write("input", input)?;
    }
    let enc = forward_enc(rmbg, input, &dump)?;
    let x = input;
    let x4 = rmbg.squeeze.apply(&enc[3]);
    if dump.want("squeeze_module") {
        dump.write("squeeze_module", &x4)?;
    }

    // ---- decoder_block4 path -------------------------------------------
    // dec_ipt + split: get_patches_batch(x, x4) tiles the RAW IMAGE by x4's
    // patch size; for the square 1024 input and a 32x32 feature that is just
    // x resized to x4's grid, then interpolated to x4's grid (a no-op here).
    let i5 = rmbg.decoder.ipt[4].apply(&ipt_input(x, x4.h, x4.w));
    if dump.want("ipt_blk5") {
        dump.write("ipt_blk5", &i5)?;
    }
    let x4_cat = cat_channels(&[&x4, &i5]);
    let mut p4 = rmbg.decoder.blocks[3].apply(&x4_cat);
    if dump.want("decoder_block4") {
        dump.write("decoder_block4", &p4)?;
    }
    let m4 = conv1x1(&p4, &rmbg.decoder.ms[2].0, 1, rmbg.decoder.ms[2].0.len(), Some(&rmbg.decoder.ms[2].1));
    let attn4 = rmbg.decoder.gdt[2].apply(&p4);
    mul_broadcast_channel_inplace(&mut p4, &attn4);
    let mut p3_in = resize_bilinear(&p4, enc[2].h, enc[2].w, true);
    let lat4 = conv1x1(&enc[2], &rmbg.decoder.lat4, enc[2].c, enc[2].c, Some(&rmbg.decoder.lat4b));
    add_inplace(&mut p3_in, &lat4);

    // ---- decoder_block3 path -------------------------------------------
    // get_patches_batch(x, _p3) -> raw image tiled at _p3's size, i.e. x
    // resized to x3's 64x64 grid.
    let i4 = rmbg.decoder.ipt[3].apply(&ipt_input(x, enc[2].h, enc[2].w));
    if dump.want("ipt_blk4") {
        dump.write("ipt_blk4", &i4)?;
    }
    let p3_cat = cat_channels(&[&p3_in, &i4]);
    let mut p3 = rmbg.decoder.blocks[2].apply(&p3_cat);
    if dump.want("decoder_block3") {
        dump.write("decoder_block3", &p3)?;
    }
    let m3 = conv1x1(&p3, &rmbg.decoder.ms[1].0, 1, rmbg.decoder.ms[1].0.len(), Some(&rmbg.decoder.ms[1].1));
    let attn3 = rmbg.decoder.gdt[1].apply(&p3);
    mul_broadcast_channel_inplace(&mut p3, &attn3);
    let mut p2_in = resize_bilinear(&p3, enc[1].h, enc[1].w, true);
    let lat3 = conv1x1(&enc[1], &rmbg.decoder.lat3, enc[1].c, enc[1].c, Some(&rmbg.decoder.lat3b));
    add_inplace(&mut p2_in, &lat3);

    // ---- decoder_block2 path -------------------------------------------
    // get_patches_batch(x, _p2) -> raw image at x2's 128x128 grid.
    let i3 = rmbg.decoder.ipt[2].apply(&ipt_input(x, enc[1].h, enc[1].w));
    if dump.want("ipt_blk3") {
        dump.write("ipt_blk3", &i3)?;
    }
    let p2_cat = cat_channels(&[&p2_in, &i3]);
    let mut p2 = rmbg.decoder.blocks[1].apply(&p2_cat);
    if dump.want("decoder_block2") {
        dump.write("decoder_block2", &p2)?;
    }
    let m2 = conv1x1(&p2, &rmbg.decoder.ms[0].0, 1, rmbg.decoder.ms[0].0.len(), Some(&rmbg.decoder.ms[0].1));
    let attn2 = rmbg.decoder.gdt[0].apply(&p2);
    mul_broadcast_channel_inplace(&mut p2, &attn2);
    let mut p1_in = resize_bilinear(&p2, enc[0].h, enc[0].w, true);
    let lat2 = conv1x1(&enc[0], &rmbg.decoder.lat2, enc[0].c, enc[0].c, Some(&rmbg.decoder.lat2b));
    add_inplace(&mut p1_in, &lat2);

    // ---- decoder_block1 path -------------------------------------------
    // get_patches_batch(x, _p1) -> raw image at x1's 256x256 grid.
    let i2 = rmbg.decoder.ipt[1].apply(&ipt_input(x, enc[0].h, enc[0].w));
    if dump.want("ipt_blk2") {
        dump.write("ipt_blk2", &i2)?;
    }
    let p1_cat = cat_channels(&[&p1_in, &i2]);
    let p1 = rmbg.decoder.blocks[0].apply(&p1_cat);
    if dump.want("decoder_block1") {
        dump.write("decoder_block1", &p1)?;
    }
    let up = resize_bilinear(&p1, x.h, x.w, true);
    // get_patches_batch(x, _p1): the upsampled feature grid is the full
    // 1024x1024, so blk1 consumes the raw 3-channel image unchanged.
    let i1 = rmbg.decoder.ipt[0].apply(&ipt_input(x, x.h, x.w));
    if dump.want("ipt_blk1") {
        dump.write("ipt_blk1", &i1)?;
    }
    let p1_cat = cat_channels(&[&up, &i1]);
    let co_ic = p1_cat.c;
    let out = conv1x1(&p1_cat, &rmbg.decoder.conv_out1_w, 1, co_ic, Some(&rmbg.decoder.conv_out1_b));
    if dump.want("conv_out1") {
        dump.write("conv_out1", &out)?;
    }
    if dump.want("pred0") {
        dump.write("pred0", &m4)?;
    }
    if dump.want("pred1") {
        dump.write("pred1", &m3)?;
    }
    if dump.want("pred2") {
        dump.write("pred2", &m2)?;
    }
    if dump.want("pred3") {
        dump.write("pred3", &out)?;
    }
    let _ = (m2, m3, m4);
    Ok(out)
}
