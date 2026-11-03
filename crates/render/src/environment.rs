//! Image-based lighting from an equirectangular HDR sky, baked on the CPU
//! once per sky (pure maths, tested without a GPU):
//!
//! 1. The sun is pulled out of the image: texels far brighter than the
//!    rest are clamped, and the energy removed becomes a directional light
//!    (so it casts shadows, and the map has no hot spot to sparkle).
//! 2. The rest is resampled into a cube map, and each mip below the first
//!    is prefiltered for a GGX roughness (importance sampling with filtered
//!    lookups, Colbert & Křivánek), for glossy reflections.
//! 3. Diffuse light is projected to SH9 irradiance, like the procedural sky.

use glam::Vec3;

use crate::sky::project_irradiance;

/// Cube face size of the sharpest level.
pub const ENV_FACE: u32 = 256;
/// Levels, `ENV_FACE` down to 8: level `i` is for roughness `i / (ENV_MIPS - 1)`.
pub const ENV_MIPS: u32 = 6;
/// Texels brighter than this many times the median luminance are sun.
const SUN_CUTOFF: f32 = 50.0;
/// GGX samples per texel at a roughness: narrow lobes on big levels need
/// fewer (each lookup is already filtered to the sample's footprint).
fn prefilter_samples(roughness: f32) -> u32 {
    match roughness {
        r if r <= 0.2 => 16,
        r if r <= 0.4 => 32,
        _ => 64,
    }
}

/// A baked environment, ready for [`crate::Renderer::set_environment`].
#[derive(Debug, Clone, PartialEq)]
pub struct Environment {
    pub(crate) face: u32,
    /// Per level, six faces (+X, −X, +Y, −Y, +Z, −Z) of RGBA half floats.
    pub(crate) levels: Vec<Vec<[u16; 4]>>,
    /// Diffuse irradiance as SH9 (see `sky.rs`).
    pub irradiance_sh: [Vec3; 9],
    /// Unit direction toward the sun.
    pub sun_direction: Vec3,
    /// The sun's irradiance (radiance of the directional light); zero when
    /// the image has no compact sun to extract, e.g. one lost in haze.
    pub sun_color: Vec3,
}

impl Environment {
    /// Bakes an equirect (`width` = 2 × `height`, rows from straight up to
    /// straight down, linear RGB).
    pub fn bake(width: u32, height: u32, pixels: &[[f32; 3]]) -> Self {
        Self::bake_sized(width, height, pixels, ENV_FACE)
    }

    pub(crate) fn bake_sized(width: u32, height: u32, pixels: &[[f32; 3]], face: u32) -> Self {
        let mut equirect = Equirect::new(width, height, pixels);
        let sun = equirect.extract_sun();
        let pyramid = Pyramid::from_equirect(&equirect, face);
        let levels = (0..ENV_MIPS)
            .map(|level| {
                let size = (face >> level).max(1);
                let roughness = level as f32 / (ENV_MIPS - 1) as f32;
                let texels = if level == 0 {
                    pyramid.levels[0].clone()
                } else {
                    prefilter(&pyramid, size, roughness)
                };
                texels.iter().map(|c| half4(*c)).collect()
            })
            .collect();
        // Diffuse from a blurred level: the SH projection takes a few
        // thousand point samples.
        let blur = (face.trailing_zeros()).saturating_sub(5) as f32;
        let irradiance_sh = project_irradiance(|dir| pyramid.sample(dir, blur), 4096);
        Self {
            face,
            levels,
            irradiance_sh,
            sun_direction: sun.direction,
            sun_color: sun.color,
        }
    }
}

/// Direction for equirect coordinates `u`, `v` in [0, 1]: `v` 0 is straight
/// up, `u` 0.5 looks down −Z.
pub(crate) fn equirect_dir(u: f32, v: f32) -> Vec3 {
    let theta = v * std::f32::consts::PI;
    let phi = (u - 0.5) * std::f32::consts::TAU;
    Vec3::new(
        theta.sin() * phi.sin(),
        theta.cos(),
        -theta.sin() * phi.cos(),
    )
}

/// The inverse of [`equirect_dir`] for a unit direction.
pub(crate) fn equirect_uv(dir: Vec3) -> (f32, f32) {
    let u = dir.x.atan2(-dir.z) / std::f32::consts::TAU + 0.5;
    let v = dir.y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
    (u, v)
}

/// The direction through cube face `face` (+X, −X, +Y, −Y, +Z, −Z) at face
/// coordinates `s`, `t` in [−1, 1], `t` down the face's rows; the GPU's
/// cube map convention.
pub(crate) fn face_dir(face: usize, s: f32, t: f32) -> Vec3 {
    match face {
        0 => Vec3::new(1.0, -t, -s),
        1 => Vec3::new(-1.0, -t, s),
        2 => Vec3::new(s, 1.0, t),
        3 => Vec3::new(s, -1.0, -t),
        4 => Vec3::new(s, -t, 1.0),
        _ => Vec3::new(-s, -t, -1.0),
    }
    .normalize()
}

/// The face and face coordinates (`s`, `t` in [−1, 1]) a direction hits.
pub(crate) fn cube_coords(dir: Vec3) -> (usize, f32, f32) {
    let a = dir.abs();
    if a.x >= a.y && a.x >= a.z {
        if dir.x > 0.0 {
            (0, -dir.z / a.x, -dir.y / a.x)
        } else {
            (1, dir.z / a.x, -dir.y / a.x)
        }
    } else if a.y >= a.z {
        if dir.y > 0.0 {
            (2, dir.x / a.y, dir.z / a.y)
        } else {
            (3, dir.x / a.y, -dir.z / a.y)
        }
    } else if dir.z > 0.0 {
        (4, dir.x / a.z, -dir.y / a.z)
    } else {
        (5, -dir.x / a.z, -dir.y / a.z)
    }
}

fn luminance(c: Vec3) -> f32 {
    c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

struct Equirect {
    width: u32,
    height: u32,
    pixels: Vec<Vec3>,
}

struct Sun {
    direction: Vec3,
    color: Vec3,
}

impl Equirect {
    fn new(width: u32, height: u32, pixels: &[[f32; 3]]) -> Self {
        assert_eq!(pixels.len(), (width * height) as usize, "equirect size");
        Self {
            width,
            height,
            pixels: pixels.iter().map(|p| Vec3::from_array(*p)).collect(),
        }
    }

    fn texel_dir(&self, x: u32, y: u32) -> Vec3 {
        equirect_dir(
            (x as f32 + 0.5) / self.width as f32,
            (y as f32 + 0.5) / self.height as f32,
        )
    }

    /// Solid angle of a texel in row `y`.
    fn texel_solid_angle(&self, y: u32) -> f32 {
        let theta = (y as f32 + 0.5) / self.height as f32 * std::f32::consts::PI;
        (std::f32::consts::TAU / self.width as f32)
            * (std::f32::consts::PI / self.height as f32)
            * theta.sin()
    }

    /// Clamps texels brighter than `SUN_CUTOFF` × the median luminance and
    /// returns the energy removed as a sun (zero colour, toward the
    /// brightest texel, if nothing is that bright).
    fn extract_sun(&mut self) -> Sun {
        let mut lums: Vec<f32> = self.pixels.iter().map(|&p| luminance(p)).collect();
        let mid = lums.len() / 2;
        let median = *lums.select_nth_unstable_by(mid, f32::total_cmp).1;
        let cutoff = (median * SUN_CUTOFF).max(1e-3);
        let mut color = Vec3::ZERO;
        let mut toward = Vec3::ZERO;
        let (mut brightest, mut peak) = (Vec3::Y, -1.0);
        for y in 0..self.height {
            let solid_angle = self.texel_solid_angle(y);
            for x in 0..self.width {
                let dir = self.texel_dir(x, y);
                let p = &mut self.pixels[(y * self.width + x) as usize];
                let lum = luminance(*p);
                if lum > peak {
                    peak = lum;
                    brightest = dir;
                }
                if lum > cutoff {
                    let removed = *p * (1.0 - cutoff / lum);
                    *p -= removed;
                    color += removed * solid_angle;
                    toward += dir * luminance(removed) * solid_angle;
                }
            }
        }
        if luminance(color) > 1e-6 && toward.length() > 0.0 {
            Sun {
                direction: toward.normalize(),
                color,
            }
        } else {
            Sun {
                direction: brightest,
                color: Vec3::ZERO,
            }
        }
    }

    /// Bilinear radiance along unit `dir` (wrapping around the horizon).
    fn sample(&self, dir: Vec3) -> Vec3 {
        let (u, v) = equirect_uv(dir);
        let fx = u * self.width as f32 - 0.5;
        let fy = (v * self.height as f32 - 0.5).clamp(0.0, (self.height - 1) as f32);
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let at = |x: f32, y: f32| {
            let x = (x as i64).rem_euclid(self.width as i64) as u32;
            let y = (y as i64).clamp(0, self.height as i64 - 1) as u32;
            self.pixels[(y * self.width + x) as usize]
        };
        let top = at(x0, y0).lerp(at(x0 + 1.0, y0), tx);
        let bottom = at(x0, y0 + 1.0).lerp(at(x0 + 1.0, y0 + 1.0), tx);
        top.lerp(bottom, ty)
    }
}

/// A cube map with a full box-filtered mip chain, for filtered lookups.
struct Pyramid {
    face: u32,
    /// Per level, six faces of `size²` texels.
    levels: Vec<Vec<Vec3>>,
}

impl Pyramid {
    fn from_equirect(equirect: &Equirect, face: u32) -> Self {
        // 2×2 samples per texel: the equirect is denser near the poles.
        let mut base = Vec::with_capacity((6 * face * face) as usize);
        for f in 0..6 {
            for y in 0..face {
                for x in 0..face {
                    let mut sum = Vec3::ZERO;
                    for (dx, dy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                        let s = 2.0 * (x as f32 + dx) / face as f32 - 1.0;
                        let t = 2.0 * (y as f32 + dy) / face as f32 - 1.0;
                        sum += equirect.sample(face_dir(f, s, t));
                    }
                    base.push(sum * 0.25);
                }
            }
        }
        let mut levels = vec![base];
        let mut size = face;
        while size > 1 {
            let half = size / 2;
            let prev = levels.last().expect("a level");
            let mut next = Vec::with_capacity((6 * half * half) as usize);
            for f in 0..6 {
                let at = |x: u32, y: u32| prev[(f * size * size + y * size + x) as usize];
                for y in 0..half {
                    for x in 0..half {
                        next.push(
                            (at(2 * x, 2 * y)
                                + at(2 * x + 1, 2 * y)
                                + at(2 * x, 2 * y + 1)
                                + at(2 * x + 1, 2 * y + 1))
                                * 0.25,
                        );
                    }
                }
            }
            levels.push(next);
            size = half;
        }
        Self { face, levels }
    }

    fn sample_level(&self, dir: Vec3, level: usize) -> Vec3 {
        let size = (self.face >> level).max(1);
        let texels = &self.levels[level];
        let (f, s, t) = cube_coords(dir);
        let fx = ((s + 1.0) * 0.5 * size as f32 - 0.5).clamp(0.0, (size - 1) as f32);
        let fy = ((t + 1.0) * 0.5 * size as f32 - 0.5).clamp(0.0, (size - 1) as f32);
        let (x0, y0) = (fx.floor() as u32, fy.floor() as u32);
        let (x1, y1) = ((x0 + 1).min(size - 1), (y0 + 1).min(size - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let base = f as u32 * size * size;
        let at = |x: u32, y: u32| texels[(base + y * size + x) as usize];
        let top = at(x0, y0).lerp(at(x1, y0), tx);
        let bottom = at(x0, y1).lerp(at(x1, y1), tx);
        top.lerp(bottom, ty)
    }

    /// Trilinear radiance along unit `dir` at mip `lod`.
    fn sample(&self, dir: Vec3, lod: f32) -> Vec3 {
        let last = (self.levels.len() - 1) as f32;
        let lod = lod.clamp(0.0, last);
        let lo = lod.floor() as usize;
        let hi = (lo + 1).min(self.levels.len() - 1);
        self.sample_level(dir, lo)
            .lerp(self.sample_level(dir, hi), lod - lo as f32)
    }
}

/// Hammersley point `i` of `n`.
fn hammersley(i: u32, n: u32) -> (f32, f32) {
    (
        i as f32 / n as f32,
        i.reverse_bits() as f32 * 2.328_306_4e-10,
    )
}

/// One cube level (`size²` per face) of the environment as seen by a GGX
/// lobe of `roughness`, with N = V = R (the split-sum approximation).
fn prefilter(pyramid: &Pyramid, size: u32, roughness: f32) -> Vec<Vec3> {
    let a = roughness * roughness;
    let a2 = a * a;
    // Light directions in the lobe's frame (z = N) with their weights and
    // the pyramid level matching each sample's footprint.
    let source_texel = 4.0 * std::f32::consts::PI / (6.0 * (pyramid.face * pyramid.face) as f32);
    let count = prefilter_samples(roughness);
    let samples: Vec<(Vec3, f32, f32)> = (0..count)
        .filter_map(|i| {
            let (u, v) = hammersley(i, count);
            let phi = std::f32::consts::TAU * u;
            let cos_theta = ((1.0 - v) / (1.0 + (a2 - 1.0) * v)).sqrt();
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let h = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);
            let l = 2.0 * h.z * h - Vec3::Z;
            (l.z > 0.0).then(|| {
                let d = (cos_theta * cos_theta * (a2 - 1.0) + 1.0).powi(2);
                let distribution = a2 / (std::f32::consts::PI * d);
                // pdf of L with N = V: D · (N·H) / (4 (V·H)) = D / 4.
                let pdf = distribution * 0.25;
                let sample_angle = 1.0 / (count as f32 * pdf + 1e-4);
                let lod = (0.5 * (sample_angle / source_texel).log2() + 1.0).max(0.0);
                (l, l.z, lod)
            })
        })
        .collect();
    let mut out = Vec::with_capacity((6 * size * size) as usize);
    for f in 0..6 {
        for y in 0..size {
            for x in 0..size {
                let s = 2.0 * (x as f32 + 0.5) / size as f32 - 1.0;
                let t = 2.0 * (y as f32 + 0.5) / size as f32 - 1.0;
                let n = face_dir(f, s, t);
                let tangent = n.any_orthonormal_vector();
                let bitangent = n.cross(tangent);
                let mut sum = Vec3::ZERO;
                let mut weight = 0.0;
                for &(l, n_dot_l, lod) in &samples {
                    let dir = tangent * l.x + bitangent * l.y + n * l.z;
                    sum += pyramid.sample(dir, lod) * n_dot_l;
                    weight += n_dot_l;
                }
                out.push(sum / weight.max(1e-6));
            }
        }
    }
    out
}

/// An RGB colour as RGBA half floats (alpha 1).
fn half4(c: Vec3) -> [u16; 4] {
    [f16_bits(c.x), f16_bits(c.y), f16_bits(c.z), f16_bits(1.0)]
}

/// IEEE half-precision bits of a non-negative `v` (rounded to nearest;
/// infinity above half's range, 0 for NaN).
pub(crate) fn f16_bits(v: f32) -> u16 {
    if v.is_nan() || v <= 0.0 {
        return 0;
    }
    if v >= 65520.0 {
        return 0x7c00;
    }
    if v < 6.103_515_6e-5 {
        // Subnormal: multiples of 2^-24.
        return (v * 16_777_216.0).round() as u16;
    }
    let bits = v.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    let mut half = ((exponent as u32) << 10) | (mantissa >> 13);
    // Round to nearest; a carry into the exponent is still correct.
    if mantissa & 0x1000 != 0 {
        half += 1;
    }
    half as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sky::eval_sh;

    fn half_to_f32(h: u16) -> f32 {
        let exponent = (h >> 10) & 0x1f;
        let mantissa = (h & 0x3ff) as f32;
        match exponent {
            0 => mantissa * 2f32.powi(-24),
            31 => f32::INFINITY,
            e => (1.0 + mantissa / 1024.0) * 2f32.powi(e as i32 - 15),
        }
    }

    /// A deterministic spread of numbers in [0, 1).
    fn spread(i: usize, k: f32) -> f32 {
        (i as f32 * k).fract()
    }

    fn unit(i: usize) -> Vec3 {
        Vec3::new(
            spread(i, 0.754) - 0.5,
            spread(i, 0.569) - 0.5,
            spread(i, 0.321) - 0.5,
        )
        .normalize_or(Vec3::X)
    }

    #[test]
    fn half_floats_round_to_nearest() {
        for (v, bits) in [
            (1.0, 0x3c00),
            (0.5, 0x3800),
            (2.0, 0x4000),
            (65504.0, 0x7bff),
            (1e6, 0x7c00),
            (0.0, 0),
            (-3.0, 0),
            (f32::NAN, 0),
        ] {
            assert_eq!(f16_bits(v), bits, "{v}");
        }
        for i in 1..2000 {
            let v = (i as f32 * 0.137).powf(2.5) * 1e-3;
            let back = half_to_f32(f16_bits(v));
            assert!(
                (back - v).abs() <= v * 1e-3 + 6e-8,
                "{v} came back as {back}"
            );
        }
    }

    #[test]
    fn directions_survive_equirect_and_cube_round_trips() {
        for i in 0..500 {
            let dir = unit(i);
            let (u, v) = equirect_uv(dir);
            assert!((0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v));
            assert!(equirect_dir(u, v).abs_diff_eq(dir, 1e-4), "{dir}");
            let (face, s, t) = cube_coords(dir);
            assert!(s.abs() <= 1.0 + 1e-5 && t.abs() <= 1.0 + 1e-5);
            assert!(face_dir(face, s, t).abs_diff_eq(dir, 1e-4), "{dir}");
        }
        // The image's middle looks down −Z; its top is straight up.
        assert!(equirect_dir(0.5, 0.5).abs_diff_eq(Vec3::NEG_Z, 1e-6));
        assert!(equirect_dir(0.3, 0.0).abs_diff_eq(Vec3::Y, 1e-6));
        // Each face's centre is its axis.
        for (face, axis) in [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ]
        .into_iter()
        .enumerate()
        {
            assert!(face_dir(face, 0.0, 0.0).abs_diff_eq(axis, 1e-6));
        }
    }

    #[test]
    fn a_uniform_sky_stays_uniform_at_every_roughness() {
        let (w, h) = (64, 32);
        let sky = vec![[0.3, 0.6, 0.9]; (w * h) as usize];
        let env = Environment::bake_sized(w, h, &sky, 16);
        assert_eq!(env.sun_color, Vec3::ZERO, "no sun in a flat sky");
        assert_eq!(env.levels.len(), ENV_MIPS as usize);
        for (level, texels) in env.levels.iter().enumerate() {
            let size = (16 >> level).max(1) as usize;
            assert_eq!(texels.len(), 6 * size * size);
            for t in texels {
                let c = [t[0], t[1], t[2]].map(half_to_f32);
                for (got, want) in c.iter().zip([0.3, 0.6, 0.9]) {
                    assert!((got - want).abs() < 2e-3, "level {level}: {c:?}");
                }
            }
        }
        // Irradiance from a uniform sky is π times its radiance, any way up.
        for i in 0..20 {
            let e = eval_sh(&env.irradiance_sh, unit(i));
            assert!(
                e.abs_diff_eq(Vec3::new(0.3, 0.6, 0.9) * std::f32::consts::PI, 0.03),
                "{e}"
            );
        }
    }

    #[test]
    fn a_bright_spot_becomes_the_sun_and_keeps_its_energy() {
        let (w, h) = (128_u32, 64_u32);
        let mut sky = vec![[0.2, 0.3, 0.5]; (w * h) as usize];
        let (sx, sy) = (40_u32, 20_u32);
        let sun_dir = equirect_dir((sx as f32 + 0.5) / w as f32, (sy as f32 + 0.5) / h as f32);
        let mut equirect = Equirect::new(w, h, &sky);
        let solid_angle = equirect.texel_solid_angle(sy);
        sky[(sy * w + sx) as usize] = [5000.0, 4000.0, 3000.0];
        equirect = Equirect::new(w, h, &sky);
        let sun = equirect.extract_sun();
        assert!(
            sun.direction.abs_diff_eq(sun_dir, 1e-4),
            "{}",
            sun.direction
        );
        // Energy over the cutoff moved into the sun, not lost.
        let cutoff = luminance(Vec3::new(0.2, 0.3, 0.5)) * SUN_CUTOFF;
        let kept = equirect.pixels[(sy * w + sx) as usize];
        assert!((luminance(kept) - cutoff).abs() < 1e-2);
        let total = Vec3::new(5000.0, 4000.0, 3000.0) * solid_angle;
        assert!((sun.color + kept * solid_angle).abs_diff_eq(total, total.x * 1e-4));
    }

    #[test]
    fn the_bundled_skies_suns_are_found_or_left_to_the_caller() {
        let load = |bytes: &[u8]| impulse_assets::load_hdr(bytes).unwrap();
        let clouds = load(include_bytes!(
            "../../../assets/env/kloofendal_48d_partly_cloudy_puresky_1k.hdr"
        ));
        let env = Environment::bake_sized(clouds.width, clouds.height, &clouds.pixels, 16);
        let elevation = env.sun_direction.y.asin().to_degrees();
        assert!((elevation - 48.0).abs() < 1.5, "sun at {elevation}°");
        // About as strong as the procedural sky's sun (3), and warm-white.
        let sun = env.sun_color;
        assert!((2.0..10.0).contains(&luminance(sun)), "{sun}");
        assert!(sun.x >= sun.z, "{sun}");
        // Sky light still comes from above.
        let up = eval_sh(&env.irradiance_sh, Vec3::Y);
        let down = eval_sh(&env.irradiance_sh, Vec3::NEG_Y);
        assert!(luminance(up) > luminance(down), "{up} vs {down}");

        // The sunset's sun is a broad haze: nothing to extract, but the
        // direction points at it, low over the horizon.
        let sunset = load(include_bytes!(
            "../../../assets/env/belfast_sunset_puresky_1k.hdr"
        ));
        let env = Environment::bake_sized(sunset.width, sunset.height, &sunset.pixels, 16);
        assert_eq!(env.sun_color, Vec3::ZERO);
        let elevation = env.sun_direction.y.asin().to_degrees();
        assert!((0.0..6.0).contains(&elevation), "sun at {elevation}°");
    }

    #[test]
    fn rougher_levels_spread_a_bright_patch() {
        let (w, h) = (64_u32, 32_u32);
        // Dark, with a bright band just above the horizon (under the sun
        // cutoff, so it stays in the map).
        let sky: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let y = i / w;
                if (13..16).contains(&y) {
                    [1.0; 3]
                } else {
                    [0.05; 3]
                }
            })
            .collect();
        let env = Environment::bake_sized(w, h, &sky, 16);
        let up = |level: usize| {
            let size = (16 >> level).max(1) as usize;
            // The middle texel of the +Y face: straight up.
            let t = env.levels[level][2 * size * size + (size / 2) * size + size / 2];
            half_to_f32(t[0])
        };
        // Sharp levels see only the dark zenith; rough ones gather the band.
        assert!(up(0) < 0.1, "{}", up(0));
        assert!(up(5) > up(2), "{} vs {}", up(5), up(2));
    }
}
