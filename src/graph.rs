
//! Full RMBG-2.0 forward graph (backbone + decoder), CPU reference path.
use crate::deform::deform_conv2d;
use crate::tensor::*;
use crate::weights::Weights;

/// A folded BatchNorm.
#[derive(Clone)]
pub struct Affine {
    pub scale: Vec<f32>,
    pub shift: Vec<f32>,
}

impl Affine {
    pub fn load(wts: &Weights, prefix: &str) -> Result<Affine, String> {
        let w = wts.get(&format!("{}.weight", prefix))?;
        let b = wts.get(&format!("{}.bias", prefix))?;
        let mean = wts.get(&format!("{}.running_mean", prefix))?;
        let var = wts.get(&format!("{}.running_var", prefix))?;
        let mut scale = vec![0.0f32; w.len()];
        let mut shift = vec![0.0f32; w.len()];
        for c in 0..w.len() {
            let s = w[c] / (var[c] + 1e-5).sqrt();
            scale[c] = s;
            shift[c] = b[c] - mean[c] * s;
        }
        Ok(Affine { scale, shift })
    }

    pub fn apply(&self, x: &mut Tensor) {
        apply_affine_inplace(x, &self.scale, &self.shift);
    }
}

/// One DeformableConv2d (offset + modulator + regular conv).
pub struct DeformConv {
    pub offset_w: Vec<f32>,
    pub offset_b: Vec<f32>,
    pub modulator_w: Vec<f32>,
    pub modulator_b: Vec<f32>,
    pub regular_w: Vec<f32>,
    pub k: usize,
    pub pad: usize,
    pub ic: usize,
    pub oc: usize,
}

impl DeformConv {
    pub fn load(wts: &Weights, prefix: &str, k: usize, pad: usize) -> Result<DeformConv, String> {
        let ow = wts.get(&format!("{}.offset_conv.weight", prefix))?;
        let shape = wts.shape(&format!("{}.offset_conv.weight", prefix))?;
        let oc = wts.shape(&format!("{}.regular_conv.weight", prefix))?[0];
        let ic = shape[1];
        Ok(DeformConv {
            offset_w: ow.to_vec(),
            offset_b: wts.get(&format!("{}.offset_conv.bias", prefix))?.to_vec(),
            modulator_w: wts.get(&format!("{}.modulator_conv.weight", prefix))?.to_vec(),
            modulator_b: wts.get(&format!("{}.modulator_conv.bias", prefix))?.to_vec(),
            regular_w: wts.get(&format!("{}.regular_conv.weight", prefix))?.to_vec(),
            k,
            pad,
            ic,
            oc,
        })
    }

    pub fn apply(&self, x: &Tensor) -> Tensor {
        let offset = conv_kxk(x, &self.offset_w, 2 * self.k * self.k, self.ic, self.k, Some(&self.offset_b));
        let mut modulator = conv_kxk(x, &self.modulator_w, self.k * self.k, self.ic, self.k, Some(&self.modulator_b));
        // 2 * sigmoid(modulator)
        modulator.data.iter_mut().for_each(|v| {
            *v = 2.0 / (1.0 + (-*v).exp());
        });
        deform_conv2d(x, &offset, &modulator, &self.regular_w, self.oc, self.k, self.pad)
    }
}

/// _ASPPModuleDeformable: deform conv -> BN -> ReLU
pub struct AsppBranch {
    pub conv: DeformConv,
    pub bn: Affine,
}

impl AsppBranch {
    pub fn load(wts: &Weights, prefix: &str, k: usize, pad: usize) -> Result<AsppBranch, String> {
        Ok(AsppBranch {
            conv: DeformConv::load(wts, &format!("{}.atrous_conv", prefix), k, pad)?,
            bn: Affine::load(wts, &format!("{}.bn", prefix))?,
        })
    }

    pub fn apply(&self, x: &Tensor) -> Tensor {
        let mut y = self.conv.apply(x);
        self.bn.apply(&mut y);
        relu_inplace(&mut y);
        y
    }
}

/// ASPPDeformable: aspp1 (k=1) + three deform branches (k=1,3,7) + global pool,
/// concatenated 256 each, then conv1 1x1 -> BN -> ReLU.
pub struct AsppDeformable {
    pub aspp1: AsppBranch,
    pub deforms: Vec<AsppBranch>,
    pub gap_conv_w: Vec<f32>,
    pub gap_bn: Affine,
    pub conv1_w: Vec<f32>,
    pub bn1: Affine,
    pub oc: usize,
}

impl AsppDeformable {
    pub fn load(wts: &Weights, prefix: &str, _ic: usize, oc: usize) -> Result<AsppDeformable, String> {
        let deforms = vec![
            AsppBranch::load(wts, &format!("{}.aspp_deforms.0", prefix), 1, 0)?,
            AsppBranch::load(wts, &format!("{}.aspp_deforms.1", prefix), 3, 1)?,
            AsppBranch::load(wts, &format!("{}.aspp_deforms.2", prefix), 7, 3)?,
        ];
        Ok(AsppDeformable {
            aspp1: AsppBranch::load(wts, &format!("{}.aspp1", prefix), 1, 0)?,
            deforms,
            gap_conv_w: wts.get(&format!("{}.global_avg_pool.1.weight", prefix))?.to_vec(),
            gap_bn: Affine::load(wts, &format!("{}.global_avg_pool.2", prefix))?,
            conv1_w: wts.get(&format!("{}.conv1.weight", prefix))?.to_vec(),
            bn1: Affine::load(wts, &format!("{}.bn1", prefix))?,
            oc,
        })
    }

    pub fn apply(&self, x: &Tensor) -> Tensor {
        let x1 = self.aspp1.apply(x);
        let mut branches: Vec<Tensor> = vec![x1];
        for d in self.deforms.iter() {
            branches.push(d.apply(x));
        }
        // global average pool branch
        let pooled = global_avg_pool(x);
        let gap_ic = pooled.len();
        let gap_oc = self.gap_conv_w.len() / gap_ic;
        let mut g = Tensor::from_vec(gap_ic, 1, 1, pooled);
        g = conv1x1(&g, &self.gap_conv_w, gap_oc, gap_ic, None);
        self.gap_bn.apply(&mut g);
        relu_inplace(&mut g);
        let (h, w) = (x.h, x.w);
        let g_up = resize_bilinear(&g, h, w, true);
        branches.push(g_up);
        let refs: Vec<&Tensor> = branches.iter().collect();
        let cat = cat_channels(&refs);
        let mid_c: usize = branches.iter().map(|b| b.c).sum();
        let mut y = conv1x1(&cat, &self.conv1_w, self.oc, mid_c, None);
        self.bn1.apply(&mut y);
        relu_inplace(&mut y);
        y
    }
}

/// BasicDecBlk: conv_in -> BN -> ReLU -> ASPP -> conv_out -> BN
pub struct BasicDecBlk {
    pub conv_in_w: Vec<f32>,
    pub conv_in_b: Vec<f32>,
    pub bn_in: Affine,
    pub aspp: AsppDeformable,
    pub conv_out_w: Vec<f32>,
    pub conv_out_b: Vec<f32>,
    pub bn_out: Affine,
    pub ic: usize,
    pub ic_mid: usize,
    pub oc: usize,
}

impl BasicDecBlk {
    pub fn load(wts: &Weights, prefix: &str) -> Result<BasicDecBlk, String> {
        let cin = wts.shape(&format!("{}.conv_in.weight", prefix))?;
        let cout = wts.shape(&format!("{}.conv_out.weight", prefix))?;
        Ok(BasicDecBlk {
            conv_in_w: wts.get(&format!("{}.conv_in.weight", prefix))?.to_vec(),
            conv_in_b: wts.get(&format!("{}.conv_in.bias", prefix))?.to_vec(),
            bn_in: Affine::load(wts, &format!("{}.bn_in", prefix))?,
            aspp: AsppDeformable::load(wts, &format!("{}.dec_att", prefix), cin[1], cin[0])?,
            conv_out_w: wts.get(&format!("{}.conv_out.weight", prefix))?.to_vec(),
            conv_out_b: wts.get(&format!("{}.conv_out.bias", prefix))?.to_vec(),
            bn_out: Affine::load(wts, &format!("{}.bn_out", prefix))?,
            ic: cin[1],
            ic_mid: cin[0],
            oc: cout[0],
        })
    }

    pub fn apply(&self, x: &Tensor) -> Tensor {
        let mut y = conv_kxk(x, &self.conv_in_w, self.ic_mid, self.ic, 3, Some(&self.conv_in_b));
        self.bn_in.apply(&mut y);
        relu_inplace(&mut y);
        let mut y = self.aspp.apply(&y);
        y = conv_kxk(&y, &self.conv_out_w, self.oc, self.ic_mid, 3, Some(&self.conv_out_b));
        self.bn_out.apply(&mut y);
        y
    }
}

/// SimpleConvs used by the ipt blocks: conv1 -> conv_out (no activation).
pub struct SimpleConvs {
    pub c1w: Vec<f32>,
    pub c1b: Vec<f32>,
    pub cow: Vec<f32>,
    pub cob: Vec<f32>,
    pub ic: usize,
    pub oc: usize,
}

impl SimpleConvs {
    pub fn load(wts: &Weights, prefix: &str) -> Result<SimpleConvs, String> {
        Ok(SimpleConvs {
            c1w: wts.get(&format!("{}.conv1.weight", prefix))?.to_vec(),
            c1b: wts.get(&format!("{}.conv1.bias", prefix))?.to_vec(),
            cow: wts.get(&format!("{}.conv_out.weight", prefix))?.to_vec(),
            cob: wts.get(&format!("{}.conv_out.bias", prefix))?.to_vec(),
            ic: wts.shape(&format!("{}.conv1.weight", prefix))?[1],
            oc: wts.shape(&format!("{}.conv_out.weight", prefix))?[0],
        })
    }

    pub fn apply(&self, x: &Tensor) -> Tensor {
        let y = conv_kxk(x, &self.c1w, 64, self.ic, 3, Some(&self.c1b));
        conv_kxk(&y, &self.cow, self.oc, 64, 3, Some(&self.cob))
    }
}

pub struct GdtConv {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub bn: Affine,
    pub attn_w: Vec<f32>,
    pub attn_b: Vec<f32>,
    pub ic: usize,
}

impl GdtConv {
    pub fn load(wts: &Weights, stage: usize) -> Result<GdtConv, String> {
        let p = format!("decoder.gdt_convs_{}", stage);
        Ok(GdtConv {
            w: wts.get(&format!("{}.0.weight", p))?.to_vec(),
            b: wts.get(&format!("{}.0.bias", p))?.to_vec(),
            bn: Affine::load(wts, &format!("{}.1", p))?,
            attn_w: wts.get(&format!("decoder.gdt_convs_attn_{}.0.weight", stage))?.to_vec(),
            attn_b: wts.get(&format!("decoder.gdt_convs_attn_{}.0.bias", stage))?.to_vec(),
            ic: wts.shape(&format!("{}.0.weight", p))?[1],
        })
    }

    /// returns sigmoid(attn(relu(bn(conv(p)))))
    pub fn apply(&self, p: &Tensor) -> Tensor {
        let mut g = conv_kxk(p, &self.w, 16, self.ic, 3, Some(&self.b));
        self.bn.apply(&mut g);
        relu_inplace(&mut g);
        let mut a = conv1x1(&g, &self.attn_w, 1, 16, Some(&self.attn_b));
        sigmoid_inplace(&mut a);
        a
    }
}

pub struct Decoder {
    pub ipt: Vec<SimpleConvs>,
    pub blocks: Vec<BasicDecBlk>,
    pub lat2: Vec<f32>,
    pub lat2b: Vec<f32>,
    pub lat3: Vec<f32>,
    pub lat3b: Vec<f32>,
    pub lat4: Vec<f32>,
    pub lat4b: Vec<f32>,
    pub ms: Vec<(Vec<f32>, Vec<f32>)>,
    pub conv_out1_w: Vec<f32>,
    pub conv_out1_b: Vec<f32>,
    pub gdt: Vec<GdtConv>,
}

impl Decoder {
    pub fn load(wts: &Weights) -> Result<Decoder, String> {
        let lat = |p: &str| -> Result<(Vec<f32>, Vec<f32>), String> {
            Ok((
                wts.get(&format!("{}.conv.weight", p))?.to_vec(),
                wts.get(&format!("{}.conv.bias", p))?.to_vec(),
            ))
        };
        Ok(Decoder {
            ipt: vec![
                SimpleConvs::load(wts, "decoder.ipt_blk1")?,
                SimpleConvs::load(wts, "decoder.ipt_blk2")?,
                SimpleConvs::load(wts, "decoder.ipt_blk3")?,
                SimpleConvs::load(wts, "decoder.ipt_blk4")?,
                SimpleConvs::load(wts, "decoder.ipt_blk5")?,
            ],
            blocks: vec![
                BasicDecBlk::load(wts, "decoder.decoder_block1")?,
                BasicDecBlk::load(wts, "decoder.decoder_block2")?,
                BasicDecBlk::load(wts, "decoder.decoder_block3")?,
                BasicDecBlk::load(wts, "decoder.decoder_block4")?,
            ],
            lat2: lat("decoder.lateral_block2")?.0,
            lat2b: lat("decoder.lateral_block2")?.1,
            lat3: lat("decoder.lateral_block3")?.0,
            lat3b: lat("decoder.lateral_block3")?.1,
            lat4: lat("decoder.lateral_block4")?.0,
            lat4b: lat("decoder.lateral_block4")?.1,
            ms: vec![
                (wts.get("decoder.conv_ms_spvn_2.weight")?.to_vec(), wts.get("decoder.conv_ms_spvn_2.bias")?.to_vec()),
                (wts.get("decoder.conv_ms_spvn_3.weight")?.to_vec(), wts.get("decoder.conv_ms_spvn_3.bias")?.to_vec()),
                (wts.get("decoder.conv_ms_spvn_4.weight")?.to_vec(), wts.get("decoder.conv_ms_spvn_4.bias")?.to_vec()),
            ],
            conv_out1_w: wts.get("decoder.conv_out1.0.weight")?.to_vec(),
            conv_out1_b: wts.get("decoder.conv_out1.0.bias")?.to_vec(),
            gdt: vec![
                GdtConv::load(wts, 2)?,
                GdtConv::load(wts, 3)?,
                GdtConv::load(wts, 4)?,
            ],
        })
    }
}
