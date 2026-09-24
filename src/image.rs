
//! PNG IO: decode to NCHW f32 in [0,1], encode RGB/RGBA output.
use std::fs::File;
use std::io::BufWriter;

use crate::tensor::Tensor;

/// Decode a PNG (8/16-bit, gray/RGB/RGBA) into NCHW f32 RGB in [0,1].
pub fn load_rgb(path: &str) -> Result<Tensor, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let (w, h) = (info.width as usize, info.height as usize);
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => return Err(format!("unsupported png colour type {:?}", other)),
    };
    let bytes = &buf[..info.buffer_size()];
    let sample_scale = match info.bit_depth {
        png::BitDepth::Eight => 1.0 / 255.0,
        png::BitDepth::Sixteen => 1.0 / 65535.0,
        other => return Err(format!("unsupported png bit depth {:?}", other)),
    };
    let mut out = Tensor::new(3, h, w);
    let stride_bytes = channels * if info.bit_depth == png::BitDepth::Sixteen { 2 } else { 1 };
    let read_sample = |idx: usize| -> f32 {
        match info.bit_depth {
            png::BitDepth::Sixteen => {
                let b0 = bytes[idx * 2] as u16;
                let b1 = bytes[idx * 2 + 1] as u16;
                ((b0 << 8) | b1) as f32 * sample_scale
            }
            _ => bytes[idx] as f32 * sample_scale,
        }
    };
    for y in 0..h {
        let row = y * w * stride_bytes;
        for x in 0..w {
            let base = (row + x * stride_bytes) / if info.bit_depth == png::BitDepth::Sixteen { 2 } else { 1 };
            let (r, g, b) = match channels {
                1 => {
                    let v = read_sample(base);
                    (v, v, v)
                }
                2 => {
                    let v = read_sample(base);
                    (v, v, v)
                }
                3 => (read_sample(base), read_sample(base + 1), read_sample(base + 2)),
                _ => (read_sample(base), read_sample(base + 1), read_sample(base + 2)),
            };
            let hw = h * w;
            out.data[y * w + x] = r;
            out.data[hw + y * w + x] = g;
            out.data[2 * hw + y * w + x] = b;
        }
    }
    Ok(out)
}

/// Write an 8-bit grayscale PNG from a (1, H, W) tensor in [0,1].
pub fn save_gray(path: &str, x: &Tensor) -> Result<(), String> {
    let (h, w) = (x.h, x.w);
    let file = File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(BufWriter::new(file), w as u32, h as u32);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; w * h];
    for i in 0..w * h {
        let v = x.data[i].clamp(0.0, 1.0) * 255.0;
        buf[i] = v.round() as u8;
    }
    writer.write_image_data(&buf).map_err(|e| e.to_string())?;
    Ok(())
}

/// Write an 8-bit RGBA PNG from rgb (3,H,W) and alpha mask (1,H,W), both in [0,1].
pub fn save_rgba(path: &str, rgb: &Tensor, alpha: &Tensor) -> Result<(), String> {
    let (h, w) = (rgb.h, rgb.w);
    let file = File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(BufWriter::new(file), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| e.to_string())?;
    let hw = h * w;
    let mut buf = vec![0u8; hw * 4];
    for i in 0..hw {
        for c in 0..3 {
            buf[i * 4 + c] = (rgb.data[c * hw + i].clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        buf[i * 4 + 3] = (alpha.data[i].clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    writer.write_image_data(&buf).map_err(|e| e.to_string())?;
    Ok(())
}
