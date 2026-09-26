//! The procedural sky: one function drawn as the background (with a sun disc)
//! and used as ambient light (without it: the sun already lights the scene
//! directly, so including its disc would count it twice).
//!
//! Diffuse ambient is the sky's irradiance, pre-integrated into 9 spherical
//! harmonic coefficients on the CPU; the shader evaluates them per normal.
//! `shaders/scene.wgsl` mirrors [`Sky::radiance`] for reflections and the
//! background — keep the two in step.

use glam::Vec3;

/// Linear HDR sky colours and the sun.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sky {
    pub zenith: Vec3,
    pub horizon: Vec3,
    pub ground: Vec3,
    /// Unit direction *toward* the sun.
    pub sun_direction: Vec3,
    /// Radiance of the sun's direct light (and, scaled, of its disc).
    pub sun_color: Vec3,
}

impl Default for Sky {
    fn default() -> Self {
        Self {
            // Sky light about a third of the sun's on a level surface, as
            // on a clear day; brighter reads flat and washed out.
            zenith: Vec3::new(0.12, 0.24, 0.50),
            horizon: Vec3::new(0.36, 0.40, 0.46),
            ground: Vec3::new(0.12, 0.11, 0.10),
            sun_direction: Vec3::new(0.4, 1.0, 0.3).normalize(),
            sun_color: Vec3::new(3.0, 2.85, 2.6),
        }
    }
}

impl Sky {
    /// Sky radiance seen along unit `dir`, without the sun disc. Mirrored in
    /// `scene.wgsl` (`sky_radiance`).
    pub fn radiance(&self, dir: Vec3) -> Vec3 {
        let y = dir.y;
        if y >= 0.0 {
            self.horizon.lerp(self.zenith, y.powf(0.5))
        } else {
            let t = (-y / 0.6).clamp(0.0, 1.0);
            self.horizon.lerp(self.ground, t * t * (3.0 - 2.0 * t))
        }
    }

    /// Irradiance coefficients: `Σ c[i] · Y_i(n)` is the cosine-weighted sky
    /// light arriving at a surface facing `n` (diffuse = albedo / π times it).
    pub fn irradiance_sh(&self) -> [Vec3; 9] {
        project_irradiance(|dir| self.radiance(dir), 4096)
    }
}

/// Real spherical harmonic basis, bands 0–2, at unit `n`.
pub fn sh_basis(n: Vec3) -> [f32; 9] {
    let (x, y, z) = (n.x, n.y, n.z);
    [
        0.282_095,
        0.488_603 * y,
        0.488_603 * z,
        0.488_603 * x,
        1.092_548 * x * y,
        1.092_548 * y * z,
        0.315_392 * (3.0 * z * z - 1.0),
        1.092_548 * x * z,
        0.546_274 * (x * x - y * y),
    ]
}

/// Evaluates irradiance coefficients from [`project_irradiance`] at `n`.
pub fn eval_sh(coeffs: &[Vec3; 9], n: Vec3) -> Vec3 {
    sh_basis(n).iter().zip(coeffs).map(|(y, c)| *c * *y).sum()
}

/// Projects a radiance function onto SH (uniform Fibonacci samples) and
/// convolves with the clamped cosine (Ramamoorthi & Hanrahan's A_l factors).
pub fn project_irradiance(radiance: impl Fn(Vec3) -> Vec3, samples: usize) -> [Vec3; 9] {
    let mut coeffs = [Vec3::ZERO; 9];
    for dir in sphere_samples(samples) {
        let l = radiance(dir);
        for (c, y) in coeffs.iter_mut().zip(sh_basis(dir)) {
            *c += l * y;
        }
    }
    let weight = 4.0 * std::f32::consts::PI / samples as f32;
    let band = |i: usize| match i {
        0 => std::f32::consts::PI,
        1..=3 => 2.0 * std::f32::consts::PI / 3.0,
        _ => std::f32::consts::PI / 4.0,
    };
    for (i, c) in coeffs.iter_mut().enumerate() {
        *c *= weight * band(i);
    }
    coeffs
}

/// `count` unit vectors spread evenly over the sphere (Fibonacci lattice).
fn sphere_samples(count: usize) -> impl Iterator<Item = Vec3> {
    let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    (0..count).map(move |i| {
        let y = 1.0 - 2.0 * (i as f32 + 0.5) / count as f32;
        let ring = (1.0 - y * y).sqrt();
        let phi = golden * i as f32;
        Vec3::new(ring * phi.cos(), y, ring * phi.sin())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cosine-weighted integral of `radiance` over the hemisphere around `n`,
    /// by brute force.
    fn irradiance_brute(radiance: impl Fn(Vec3) -> Vec3, n: Vec3) -> Vec3 {
        let count = 200_000;
        let weight = 4.0 * std::f32::consts::PI / count as f32;
        sphere_samples(count)
            .map(|d| radiance(d) * d.dot(n).max(0.0) * weight)
            .sum()
    }

    fn directions() -> Vec<Vec3> {
        let mut dirs: Vec<Vec3> = [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::NEG_X,
            Vec3::NEG_Y,
            Vec3::NEG_Z,
        ]
        .to_vec();
        dirs.extend(sphere_samples(20));
        dirs
    }

    #[test]
    fn a_uniform_sky_gives_pi_times_its_radiance_everywhere() {
        let coeffs = project_irradiance(|_| Vec3::splat(2.0), 4096);
        for n in directions() {
            let e = eval_sh(&coeffs, n);
            assert!(
                e.abs_diff_eq(Vec3::splat(2.0 * std::f32::consts::PI), 1e-3),
                "{n}: {e}"
            );
        }
    }

    #[test]
    fn sh_irradiance_matches_brute_force_for_the_gradient_sky() {
        let sky = Sky::default();
        let coeffs = sky.irradiance_sh();
        for n in directions() {
            let sh = eval_sh(&coeffs, n);
            let exact = irradiance_brute(|d| sky.radiance(d), n);
            let error = (sh - exact).abs() / exact.max(Vec3::splat(0.05));
            assert!(error.max_element() < 0.05, "{n}: SH {sh} vs {exact}");
        }
    }

    #[test]
    fn the_sky_is_brightest_overhead_and_darkest_below() {
        let sky = Sky::default();
        let e = |n| eval_sh(&sky.irradiance_sh(), n);
        assert!(e(Vec3::Y).element_sum() > e(Vec3::X).element_sum());
        assert!(e(Vec3::X).element_sum() > e(Vec3::NEG_Y).element_sum());
        assert_eq!(sky.radiance(Vec3::X), sky.horizon);
        assert_eq!(sky.radiance(Vec3::Y), sky.zenith);
        assert_eq!(sky.radiance(Vec3::NEG_Y), sky.ground);
    }
}
