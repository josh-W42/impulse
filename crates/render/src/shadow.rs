//! Cascaded shadow maps: which depths of the view each cascade covers, and
//! the light projection that covers them. Pure maths on the camera's
//! view-projection, so it is tested without a GPU.

use glam::camera::rh::{proj, view};
use glam::{Mat4, Vec3};

pub const CASCADES: usize = 4;
/// Per cascade; within WebGL2's guaranteed texture size.
pub const SHADOW_SIZE: u32 = 2048;
/// Practical split scheme: 1 is logarithmic, 0 uniform.
const SPLIT_LAMBDA: f32 = 0.75;
/// How far toward the sun beyond a cascade's slice casters still cast into it.
const CASTER_MARGIN: f32 = 100.0;
/// Radii round up to this, so float noise can't change a cascade's size.
const RADIUS_STEP: f32 = 1.0 / 16.0;

/// What the cascades cover.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShadowRange {
    /// Everything in view from the camera's near plane out to this view depth.
    Distance(f32),
    /// Only what is within `radius` of `focus`. Cascades end where one
    /// covering the whole ball is no bigger than the next slice, so the
    /// shadows are never blurrier than a single map over that ball.
    Focus { focus: Vec3, radius: f32 },
}

/// One frame's cascades.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cascades {
    /// How many cascades are in use (1..=CASCADES); the rest are unused.
    pub count: usize,
    /// World → each cascade's light clip space.
    pub view_proj: [Mat4; CASCADES],
    /// The view depth where cascade 0 starts.
    pub near: f32,
    /// The view depth where each cascade ends.
    pub far: [f32; CASCADES],
    /// The world size of one shadow texel in each cascade.
    pub texel: [f32; CASCADES],
    /// Unit camera forward; view depth is measured along it from the eye.
    pub forward: Vec3,
}

/// The split depths between `near` and `far`: `CASCADES + 1` of them,
/// a `lambda` blend of logarithmic and uniform spacing.
pub(crate) fn split_depths(near: f32, far: f32, lambda: f32) -> [f32; CASCADES + 1] {
    std::array::from_fn(|i| {
        let t = i as f32 / CASCADES as f32;
        let log = near * (far / near).powf(t);
        let uniform = near + (far - near) * t;
        lambda * log + (1.0 - lambda) * uniform
    })
}

/// The camera frustum as rays through the four corners of the screen.
struct Frustum {
    near: [Vec3; 4],
    far: [Vec3; 4],
    near_depth: f32,
    far_depth: f32,
    forward: Vec3,
}

impl Frustum {
    fn new(view_proj: Mat4, eye: Vec3) -> Self {
        let inverse = view_proj.inverse();
        let at = |x: f32, y: f32, z: f32| inverse.project_point3(Vec3::new(x, y, z));
        let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        let near = corners.map(|(x, y)| at(x, y, 0.0));
        let far = corners.map(|(x, y)| at(x, y, 1.0));
        let forward = (at(0.0, 0.0, 1.0) - at(0.0, 0.0, 0.0)).normalize();
        Self {
            near,
            far,
            near_depth: (near[0] - eye).dot(forward),
            far_depth: (far[0] - eye).dot(forward),
            forward,
        }
    }

    /// The four corners of the slice through the frustum at view depth `d`.
    fn slice(&self, d: f32) -> [Vec3; 4] {
        let t = (d - self.near_depth) / (self.far_depth - self.near_depth);
        std::array::from_fn(|i| self.near[i].lerp(self.far[i], t))
    }
}

/// The bounding sphere of `corners`: their centroid, and a radius rounded up
/// to `RADIUS_STEP`.
fn bounding_sphere(corners: &[Vec3]) -> (Vec3, f32) {
    let center = corners.iter().sum::<Vec3>() / corners.len() as f32;
    let radius = corners
        .iter()
        .map(|c| c.distance(center))
        .fold(0.0, f32::max);
    (center, round_radius(radius))
}

fn round_radius(radius: f32) -> f32 {
    (radius / RADIUS_STEP).ceil().max(1.0) * RADIUS_STEP
}

/// Covers a sphere with a light-space square, snapped to whole texels so it
/// doesn't shimmer as the camera moves. Returns the light's view-projection
/// and the world size of a texel.
fn fit(light_dir: Vec3, center: Vec3, radius: f32) -> (Mat4, f32) {
    let texel = 2.0 * radius / SHADOW_SIZE as f32;

    let up = if light_dir.y.abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let rotation = view::look_at_mat4(Vec3::ZERO, light_dir, up);
    let local = rotation.transform_point3(center);
    let snap = |v: f32| (v / texel).round() * texel;
    let center =
        rotation
            .inverse()
            .transform_point3(Vec3::new(snap(local.x), snap(local.y), local.z));

    let reach = radius + CASTER_MARGIN;
    let light_view = view::look_at_mat4(center - light_dir * reach, center, up);
    let r = radius;
    let projection = proj::directx::orthographic(-r, r, -r, r, 0.0, reach + radius);
    (projection * light_view, texel)
}

/// Fits the cascades for a camera (`view_proj`, `eye`) and a sun shining
/// along `light_dir` (unit).
pub(crate) fn cascades(
    view_proj: Mat4,
    eye: Vec3,
    light_dir: Vec3,
    range: ShadowRange,
) -> Cascades {
    let frustum = Frustum::new(view_proj, eye);
    let (lo, hi) = (frustum.near_depth, frustum.far_depth);
    let (near, far) = match range {
        ShadowRange::Distance(d) => (lo, d.clamp(lo, hi)),
        ShadowRange::Focus { focus, radius } => {
            let depth = (focus - eye).dot(frustum.forward);
            (
                (depth - radius).clamp(lo, hi),
                (depth + radius).clamp(lo, hi),
            )
        }
    };
    // A focus behind the camera, or a zero distance, still needs a slab.
    let far = far.max(near + 1.0);
    let splits = split_depths(near, far, SPLIT_LAMBDA);

    let ball = match range {
        ShadowRange::Focus { focus, radius } => Some((focus, round_radius(radius))),
        ShadowRange::Distance(_) => None,
    };

    let mut c = Cascades {
        count: CASCADES,
        view_proj: [Mat4::IDENTITY; CASCADES],
        near,
        far: std::array::from_fn(|i| splits[i + 1]),
        texel: [0.0; CASCADES],
        forward: frustum.forward,
    };
    for i in 0..CASCADES {
        let mut corners = [Vec3::ZERO; 8];
        corners[..4].copy_from_slice(&frustum.slice(splits[i]));
        corners[4..].copy_from_slice(&frustum.slice(splits[i + 1]));
        let (mut center, mut radius) = bounding_sphere(&corners);
        if let Some((focus, ball_radius)) = ball
            && ball_radius <= radius
        {
            // The ball is the smaller cover from here on: one last cascade.
            (center, radius) = (focus, ball_radius);
            c.count = i + 1;
            c.far[i] = far;
        }
        (c.view_proj[i], c.texel[i]) = fit(light_dir, center, radius);
        if c.count == i + 1 {
            break;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use glam::Vec4Swizzles;

    use super::*;

    fn sun() -> Vec3 {
        Vec3::new(-0.4, -1.0, -0.3).normalize()
    }

    fn camera(eye: Vec3, target: Vec3) -> Mat4 {
        proj::directx::perspective(1.0, 16.0 / 9.0, 0.1, 500.0)
            * view::look_at_mat4(eye, target, Vec3::Y)
    }

    /// Light clip coordinates of `p` in cascade `i`.
    fn light_clip(c: &Cascades, i: usize, p: Vec3) -> Vec3 {
        let v = c.view_proj[i] * p.extend(1.0);
        v.xyz() / v.w
    }

    fn inside(clip: Vec3) -> bool {
        clip.x.abs() <= 1.0 + 1e-4
            && clip.y.abs() <= 1.0 + 1e-4
            && (-1e-4..=1.0 + 1e-4).contains(&clip.z)
    }

    #[test]
    fn splits_run_from_near_to_far_and_increase() {
        let s = split_depths(0.1, 60.0, SPLIT_LAMBDA);
        assert!((s[0] - 0.1).abs() < 1e-6 && (s[CASCADES] - 60.0).abs() < 1e-4);
        assert!(s.windows(2).all(|w| w[0] < w[1]), "{s:?}");
    }

    #[test]
    fn lambda_blends_logarithmic_and_uniform_splits() {
        let uniform = split_depths(1.0, 81.0, 0.0);
        assert!((uniform[1] - 21.0).abs() < 1e-4, "{uniform:?}");
        let log = split_depths(1.0, 81.0, 1.0);
        assert!((log[1] - 3.0).abs() < 1e-4, "{log:?}");
    }

    #[test]
    fn every_slice_corner_and_the_casters_above_it_land_in_its_cascade() {
        let eye = Vec3::new(8.0, 6.0, 14.0);
        let view_proj = camera(eye, Vec3::new(0.0, 1.0, 0.0));
        let c = cascades(view_proj, eye, sun(), ShadowRange::Distance(60.0));
        let frustum = Frustum::new(view_proj, eye);
        assert_eq!(c.count, CASCADES);
        let mut start = c.near;
        for i in 0..CASCADES {
            for d in [start, c.far[i]] {
                for corner in frustum.slice(d) {
                    for p in [corner, corner - sun() * 50.0] {
                        let clip = light_clip(&c, i, p);
                        assert!(inside(clip), "cascade {i}: {p:?} → {clip:?}");
                    }
                }
            }
            start = c.far[i];
        }
    }

    #[test]
    fn turning_the_camera_keeps_cascade_sizes() {
        let eye = Vec3::new(0.0, 4.0, 10.0);
        let a = cascades(
            camera(eye, Vec3::ZERO),
            eye,
            sun(),
            ShadowRange::Distance(60.0),
        );
        let b = cascades(
            camera(eye, Vec3::new(9.0, 2.0, 3.0)),
            eye,
            sun(),
            ShadowRange::Distance(60.0),
        );
        assert_eq!(a.texel, b.texel);
    }

    #[test]
    fn moving_the_camera_moves_cascades_in_whole_texels() {
        let fixed = Vec3::new(1.0, 0.0, -2.0);
        let at = |x: f32| {
            let eye = Vec3::new(x, 5.0, 12.0);
            let target = Vec3::new(x, 0.0, 0.0);
            cascades(camera(eye, target), eye, sun(), ShadowRange::Distance(60.0))
        };
        let a = at(0.0);
        for x in [0.013, 0.37, 2.9] {
            let b = at(x);
            for i in 0..CASCADES {
                assert_eq!(a.texel[i], b.texel[i]);
                let shift = (light_clip(&b, i, fixed) - light_clip(&a, i, fixed)).truncate()
                    * SHADOW_SIZE as f32
                    / 2.0;
                let off_grid = (shift - shift.round()).abs().max_element();
                assert!(
                    off_grid < 0.02,
                    "cascade {i}, x {x}: shift {shift:?} texels"
                );
            }
        }
    }

    /// An RTS-style camera `back` metres from what it looks at.
    fn rts(focus: Vec3, back: f32) -> (Mat4, Vec3) {
        let eye = focus + Vec3::new(0.0, 0.75, 0.65).normalize() * back;
        (camera(eye, focus), eye)
    }

    #[test]
    fn focus_cascades_are_never_blurrier_than_one_map_over_the_ball() {
        let focus = Vec3::new(5.0, 0.0, 5.0);
        let single = 2.0 * 20.0 / SHADOW_SIZE as f32;
        for back in [3.0, 8.0, 15.0, 40.0, 120.0] {
            let (view_proj, eye) = rts(focus, back);
            let range = ShadowRange::Focus {
                focus,
                radius: 20.0,
            };
            let c = cascades(view_proj, eye, sun(), range);
            assert!((1..=CASCADES).contains(&c.count));
            for i in 0..c.count {
                assert!(c.texel[i] <= single + 1e-6, "back {back}: {:?}", c.texel);
            }
            let depth = (focus - eye).dot(c.forward);
            assert!((c.far[c.count - 1] - (depth + 20.0).min(500.0)).abs() < 1e-3);
        }
    }

    #[test]
    fn a_far_focus_needs_one_cascade_and_a_near_one_gets_sharper_ones() {
        let focus = Vec3::new(5.0, 0.0, 5.0);
        let range = ShadowRange::Focus {
            focus,
            radius: 20.0,
        };
        let (view_proj, eye) = rts(focus, 40.0);
        assert_eq!(cascades(view_proj, eye, sun(), range).count, 1);
        let (view_proj, eye) = rts(focus, 6.0);
        let close = cascades(view_proj, eye, sun(), range);
        assert!(close.count > 1, "{close:?}");
        assert!(close.texel[0] < close.texel[close.count - 1] * 0.5);
    }

    #[test]
    fn everything_within_the_focus_ball_lands_in_its_depths_cascade() {
        let focus = Vec3::new(-3.0, 0.0, 2.0);
        for back in [6.0, 40.0] {
            let (view_proj, eye) = rts(focus, back);
            let c = cascades(
                view_proj,
                eye,
                sun(),
                ShadowRange::Focus {
                    focus,
                    radius: 15.0,
                },
            );
            for x in -3..=3 {
                for z in -3..=3 {
                    let p = focus + Vec3::new(x as f32, 0.0, z as f32) * 5.0;
                    if p.distance(focus) > 15.0 {
                        continue;
                    }
                    let screen = view_proj.project_point3(p);
                    if screen.x.abs() > 1.0 || screen.y.abs() > 1.0 {
                        continue; // off screen
                    }
                    let depth = (p - eye).dot(c.forward);
                    if depth < c.near {
                        continue; // nearer than the range starts: not in view
                    }
                    let i = c.far[..c.count].iter().position(|&f| depth <= f).unwrap();
                    assert!(inside(light_clip(&c, i, p)), "back {back}: {p:?} in {i}");
                }
            }
        }
    }

    #[test]
    fn ranges_stay_within_the_camera_frustum() {
        let eye = Vec3::new(0.0, 2.0, 5.0);
        let view_proj = camera(eye, Vec3::ZERO);
        let c = cascades(view_proj, eye, sun(), ShadowRange::Distance(10_000.0));
        assert!((c.far[CASCADES - 1] - 500.0).abs() < 0.5, "{:?}", c.far);
        assert!((c.near - 0.1).abs() < 1e-3);
        // A focus behind the camera still yields a valid, increasing slab.
        let behind = cascades(
            view_proj,
            eye,
            sun(),
            ShadowRange::Focus {
                focus: eye + Vec3::new(0.0, 0.0, 30.0),
                radius: 5.0,
            },
        );
        assert!(behind.far[CASCADES - 1] > behind.near);
        assert!(behind.texel.iter().all(|t| t.is_finite() && *t > 0.0));
    }

    #[test]
    fn a_sun_straight_overhead_still_fits() {
        let eye = Vec3::new(0.0, 3.0, 8.0);
        let c = cascades(
            camera(eye, Vec3::ZERO),
            eye,
            Vec3::NEG_Y,
            ShadowRange::Distance(30.0),
        );
        assert!(c.view_proj.iter().all(|m| m.is_finite()));
        let clip = light_clip(&c, 0, Vec3::new(0.0, 0.0, 6.0));
        assert!(clip.is_finite());
    }
}
