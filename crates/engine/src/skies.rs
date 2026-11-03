//! The bundled HDR skies (Poly Haven, CC0; see `assets/env/LICENSES.md`)
//! and how a scene's [`SkyPreset`] reaches the renderer.

use std::collections::HashMap;

use glam::Vec3;
use impulse_render::{Environment, Gpu, Renderer};

use crate::scene_desc::SkyPreset;

const CLOUDS: &[u8] =
    include_bytes!("../../../assets/env/kloofendal_48d_partly_cloudy_puresky_1k.hdr");
const SUNSET: &[u8] = include_bytes!("../../../assets/env/belfast_sunset_puresky_1k.hdr");

/// The sunset's sun is lost in haze (nothing to extract): a warm one is
/// put toward the brightest haze, raised to at least `SUNSET_ELEVATION`
/// (the haze sits 2° up, where shadows would streak across the whole map).
const SUNSET_SUN: Vec3 = Vec3::new(1.6, 0.8, 0.35);
const SUNSET_ELEVATION: f32 = 12.0;

/// The HDR image behind a preset, if it isn't procedural.
fn hdri(preset: SkyPreset) -> Option<&'static [u8]> {
    match preset {
        SkyPreset::Clouds => Some(CLOUDS),
        SkyPreset::Sunset => Some(SUNSET),
        SkyPreset::Day | SkyPreset::Dusk | SkyPreset::Night => None,
    }
}

/// Decodes and bakes a preset's HDR sky (a fraction of a second).
pub fn bake(preset: SkyPreset) -> Result<Option<Environment>, String> {
    let Some(bytes) = hdri(preset) else {
        return Ok(None);
    };
    let start = web_time::Instant::now();
    let image = impulse_assets::load_hdr(bytes).map_err(|e| e.to_string())?;
    let mut env = Environment::bake(image.width, image.height, &image.pixels);
    log::info!(
        "baked the {} sky in {} ms",
        preset.label(),
        start.elapsed().as_millis()
    );
    if env.sun_color == Vec3::ZERO && preset == SkyPreset::Sunset {
        env.sun_color = SUNSET_SUN;
        env.sun_direction = raised(env.sun_direction, SUNSET_ELEVATION);
    }
    Ok(Some(env))
}

/// `dir` (unit) on the same bearing, at least `degrees` above the horizon.
fn raised(dir: Vec3, degrees: f32) -> Vec3 {
    let min = degrees.to_radians().sin();
    if dir.y >= min {
        return dir;
    }
    let flat = Vec3::new(dir.x, 0.0, dir.z).normalize_or(Vec3::X);
    flat * (1.0 - min * min).sqrt() + Vec3::Y * min
}

/// Keeps the renderer's sky in step with the scene's, baking each HDR sky
/// once.
#[derive(Default)]
pub struct Skies {
    applied: Option<SkyPreset>,
    baked: HashMap<SkyPreset, Environment>,
}

impl Skies {
    /// Hands `preset` to the renderer if it changed since the last call.
    /// A sky that fails to bake falls back to the procedural day (the
    /// error is returned for the status line).
    pub fn apply(
        &mut self,
        gpu: &Gpu,
        renderer: &mut Renderer,
        preset: SkyPreset,
    ) -> Result<(), String> {
        if self.applied == Some(preset) {
            return Ok(());
        }
        self.applied = Some(preset);
        if let std::collections::hash_map::Entry::Vacant(slot) = self.baked.entry(preset) {
            match bake(preset) {
                Ok(Some(env)) => {
                    slot.insert(env);
                }
                Ok(None) => {}
                Err(e) => {
                    renderer.set_environment(gpu, None);
                    renderer.set_sky(SkyPreset::Day.sky());
                    return Err(format!("{} sky: {e}", preset.label()));
                }
            }
        }
        match self.baked.get(&preset) {
            Some(env) => renderer.set_environment(gpu, Some(env)),
            None => {
                renderer.set_environment(gpu, None);
                renderer.set_sky(preset.sky());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_hdr_presets_bake_and_every_one_has_a_sun() {
        for preset in SkyPreset::ALL {
            let baked = bake(preset).unwrap();
            assert_eq!(baked.is_some(), hdri(preset).is_some(), "{preset:?}");
            if let Some(env) = baked {
                assert!(env.sun_color.max_element() > 0.5, "{preset:?}");
                assert!(env.sun_direction.y > 0.0, "{preset:?} sun below ground");
            }
        }
    }

    #[test]
    fn raising_keeps_the_bearing_and_only_lifts_low_suns() {
        let low = Vec3::new(3.0, 0.1, -4.0).normalize();
        let up = raised(low, 12.0);
        assert!((up.length() - 1.0).abs() < 1e-5);
        assert!((up.y.asin().to_degrees() - 12.0).abs() < 1e-3);
        assert!((up.x / up.z - low.x / low.z).abs() < 1e-5, "same bearing");
        let high = Vec3::new(0.3, 0.8, 0.2).normalize();
        assert_eq!(raised(high, 12.0), high);
    }
}
