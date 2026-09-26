//! Mipmap generation: 2×2 box filter down to 1×1, averaging colour in linear
//! space for sRGB textures (averaging sRGB bytes directly darkens mips).

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MipLevel {
    pub width: u32,
    pub height: u32,
    /// Tightly packed RGBA8 rows.
    pub rgba: Vec<u8>,
}

fn to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn to_srgb(c: f32) -> u8 {
    let c = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (c.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Level 0 is `rgba` itself; each next level halves both sides (min 1) until
/// 1×1. `srgb` averages RGB in linear light; alpha is always linear.
pub fn mip_chain(rgba: Vec<u8>, width: u32, height: u32, srgb: bool) -> Vec<MipLevel> {
    assert_eq!(
        rgba.len(),
        (width * height * 4) as usize,
        "RGBA8 size mismatch"
    );
    let lut: Vec<f32> = (0..=255).map(to_linear).collect();
    let mut levels = vec![MipLevel {
        width,
        height,
        rgba,
    }];
    while let Some(prev) = levels.last().filter(|l| l.width > 1 || l.height > 1) {
        let (w, h) = ((prev.width / 2).max(1), (prev.height / 2).max(1));
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                // Clamp sources so odd or 1-wide levels reuse their last texel.
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let sx = (2 * x + dx).min(prev.width - 1);
                    let sy = (2 * y + dy).min(prev.height - 1);
                    let i = ((sy * prev.width + sx) * 4) as usize;
                    for (c, total) in sum.iter_mut().enumerate() {
                        let v = prev.rgba[i + c];
                        *total += if srgb && c < 3 {
                            lut[v as usize]
                        } else {
                            v as f32 / 255.0
                        };
                    }
                }
                for (c, total) in sum.iter().enumerate() {
                    let mean = total / 4.0;
                    rgba.push(if srgb && c < 3 {
                        to_srgb(mean)
                    } else {
                        (mean * 255.0).round() as u8
                    });
                }
            }
        }
        levels.push(MipLevel {
            width: w,
            height: h,
            rgba,
        });
    }
    levels
}
