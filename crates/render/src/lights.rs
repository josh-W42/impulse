//! Local lights: point and spot lights, binned on the CPU each frame into a
//! grid of view-space clusters (16 × 9 tiles × 24 depth slices) so a pixel
//! only shades the lights that can reach its cluster. Pure maths on the
//! camera's view-projection, tested without a GPU; `scene.wgsl` looks the
//! lists up (keep `cluster_of` in the tests in step with it).

use bytemuck::{Pod, Zeroable};
use glam::camera::rh::{proj, view};
use glam::{Mat4, Vec3};

/// Lights shaded per frame, after culling: 256 × 64 bytes is WebGL2's
/// guaranteed uniform binding size.
pub const MAX_LIGHTS: usize = 256;
pub(crate) const CLUSTERS_X: u32 = 16;
pub(crate) const CLUSTERS_Y: u32 = 9;
pub(crate) const CLUSTERS_Z: u32 = 24;
pub(crate) const CLUSTER_COUNT: usize = (CLUSTERS_X * CLUSTERS_Y * CLUSTERS_Z) as usize;
/// The light grid texture: a header per cluster, then the light indices.
pub(crate) const GRID_WIDTH: u32 = 1024;
pub(crate) const GRID_ROWS: u32 = 128;
const GRID_TEXELS: usize = (GRID_WIDTH * GRID_ROWS) as usize;
/// A header packs the offset of its cluster's indices in the low bits and
/// their count above.
const COUNT_SHIFT: u32 = 20;
/// Shadowed spot lights per frame, each a layer of the spot shadow map.
pub const MAX_SPOT_SHADOWS: usize = 4;
pub(crate) const SPOT_SHADOW_SIZE: u32 = 1024;
/// Spot cones wider than this (half angle) are clamped: a shadow frustum
/// can't open to 180°.
const MAX_SPOT_ANGLE: f32 = 80.0 * std::f32::consts::PI / 180.0;

/// A light at a point in the world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    pub position: Vec3,
    /// Linear RGB; multiplied by `intensity`.
    pub color: Vec3,
    /// Radiance at 1 m (inverse-square falloff beyond).
    pub intensity: f32,
    /// Where the light fades to nothing (metres); also its culling bound.
    pub range: f32,
    pub kind: LightKind,
    /// Spot lights only: cast shadows (the nearest few to the camera do).
    pub shadows: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightKind {
    /// Shines equally in every direction.
    Point,
    /// A cone along `direction` (unit): full strength inside `inner_angle`,
    /// fading to nothing at `outer_angle` (half angles, radians).
    Spot {
        direction: Vec3,
        inner_angle: f32,
        outer_angle: f32,
    },
}

impl Light {
    pub fn point(position: Vec3, color: Vec3, intensity: f32, range: f32) -> Self {
        Self {
            position,
            color,
            intensity,
            range,
            kind: LightKind::Point,
            shadows: false,
        }
    }

    /// The sphere around everything the light reaches: the range sphere for
    /// a point light, a tighter one around a spot's cone.
    pub(crate) fn bounds(&self) -> (Vec3, f32) {
        match self.kind {
            LightKind::Point => (self.position, self.range),
            LightKind::Spot {
                direction,
                outer_angle,
                ..
            } => cone_bounds(
                self.position,
                direction.normalize_or(Vec3::NEG_Y),
                outer_angle.clamp(0.0, MAX_SPOT_ANGLE),
                self.range,
            ),
        }
    }
}

/// The smallest sphere around a cone of half angle `angle` cut off by the
/// ball of radius `range` around its tip: through the rim circle for wide
/// cones, through the tip and rim for narrow ones.
fn cone_bounds(tip: Vec3, direction: Vec3, angle: f32, range: f32) -> (Vec3, f32) {
    if angle > std::f32::consts::FRAC_PI_4 {
        (tip + direction * range * angle.cos(), range * angle.sin())
    } else {
        let r = range / (2.0 * angle.cos());
        (tip + direction * r, r)
    }
}

/// A light as the shader reads it (see `Light` in scene.wgsl).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct LightRaw {
    /// xyz: position; w: range.
    pub position_range: [f32; 4],
    /// rgb: colour × intensity; w: spot shadow layer, or -1.
    pub color: [f32; 4],
    /// xyz: spot direction; w: cone scale (0 for point lights).
    pub direction: [f32; 4],
    /// x: cone offset (1 for point lights); y: a shadow texel's size per
    /// metre of distance.
    pub spot: [f32; 4],
}

impl LightRaw {
    fn new(light: &Light) -> Self {
        let (direction, scale, offset) = match light.kind {
            LightKind::Point => (Vec3::NEG_Y, 0.0, 1.0),
            LightKind::Spot {
                direction,
                inner_angle,
                outer_angle,
            } => {
                // glTF KHR_lights_punctual's cone falloff.
                let outer = outer_angle.clamp(1e-3, MAX_SPOT_ANGLE);
                let inner = inner_angle.clamp(0.0, outer);
                let scale = 1.0 / (inner.cos() - outer.cos()).max(1e-3);
                (
                    direction.normalize_or(Vec3::NEG_Y),
                    scale,
                    -outer.cos() * scale,
                )
            }
        };
        let c = light.color * light.intensity.max(0.0);
        Self {
            position_range: light.position.extend(light.range).to_array(),
            color: [c.x, c.y, c.z, -1.0],
            direction: direction.extend(scale).to_array(),
            spot: [offset, 0.0, 0.0, 0.0],
        }
    }
}

/// One frame's lights, ready to upload.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Clustered {
    /// At most [`MAX_LIGHTS`], nearest first.
    pub lights: Vec<LightRaw>,
    /// [`CLUSTER_COUNT`] headers, then the light indices they point at.
    pub grid: Vec<u32>,
    /// The view depths the slices span; `far` is 0 with no lights.
    pub near: f32,
    pub far: f32,
    /// Light view-projections of the shadowed spots, in layer order.
    pub spot_shadows: Vec<Mat4>,
    pub stats: LightStats,
}

/// What the last frame did with its lights.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LightStats {
    /// Lights handed to the renderer.
    pub submitted: usize,
    /// In view and shaded.
    pub visible: usize,
    /// In view but over [`MAX_LIGHTS`] (the farthest go).
    pub dropped: usize,
    /// Most lights any one cluster shades.
    pub max_per_cluster: usize,
    /// Cluster entries that didn't fit the grid texture (should stay 0).
    pub overflow: usize,
}

/// The camera as the clusters see it.
struct View {
    view_proj: Mat4,
    eye: Vec3,
    right: Vec3,
    up: Vec3,
    forward: Vec3,
    near: f32,
    far: f32,
}

impl View {
    fn new(view_proj: Mat4, eye: Vec3) -> Self {
        let inverse = view_proj.inverse();
        let at = |x: f32, y: f32, z: f32| inverse.project_point3(Vec3::new(x, y, z));
        let center = at(0.0, 0.0, 0.0);
        let forward = (at(0.0, 0.0, 1.0) - center).normalize();
        Self {
            view_proj,
            eye,
            right: (at(1.0, 0.0, 0.0) - center).normalize(),
            up: (at(0.0, 1.0, 0.0) - center).normalize(),
            forward,
            near: (center - eye).dot(forward),
            far: (at(0.0, 0.0, 1.0) - eye).dot(forward),
        }
    }

    fn depth(&self, p: Vec3) -> f32 {
        (p - self.eye).dot(self.forward)
    }

    /// Whether a sphere touches the view frustum (Gribb–Hartmann planes,
    /// clip depth 0..1).
    fn sees(&self, center: Vec3, radius: f32) -> bool {
        let m = self.view_proj;
        let (r0, r1, r2, r3) = (m.row(0), m.row(1), m.row(2), m.row(3));
        let p = center.extend(1.0);
        [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2]
            .iter()
            .all(|plane| plane.dot(p) >= -radius * plane.truncate().length())
    }
}

/// The depth slice of view depth `depth`: exponential from `near` to `far`.
pub(crate) fn slice(depth: f32, near: f32, far: f32) -> u32 {
    let t = (depth.max(near) / near).ln() / (far / near).ln();
    ((t * CLUSTERS_Z as f32).floor() as i64).clamp(0, CLUSTERS_Z as i64 - 1) as u32
}

/// The tile column and row of normalized device x, y (y up; rows count
/// down from the top, like pixels).
fn tile(ndc_x: f32, ndc_y: f32) -> (u32, u32) {
    let cell = |t: f32, n: u32| ((t * n as f32).floor() as i64).clamp(0, n as i64 - 1) as u32;
    (
        cell((ndc_x + 1.0) * 0.5, CLUSTERS_X),
        cell((1.0 - ndc_y) * 0.5, CLUSTERS_Y),
    )
}

pub(crate) fn cluster_index(x: u32, y: u32, z: u32) -> usize {
    (x + y * CLUSTERS_X + z * CLUSTERS_X * CLUSTERS_Y) as usize
}

/// Culls, caps and bins `lights` for a camera (`view_proj`, `eye`), and
/// picks and fits the shadowed spots.
pub(crate) fn cluster(lights: &[Light], view_proj: Mat4, eye: Vec3) -> Clustered {
    let view = View::new(view_proj, eye);
    let mut visible: Vec<(f32, &Light)> = lights
        .iter()
        .filter(|l| l.range > 0.0 && l.intensity > 0.0)
        .filter_map(|l| {
            let (center, radius) = l.bounds();
            view.sees(center, radius)
                .then(|| ((center.distance(eye) - radius).max(0.0), l))
        })
        .collect();
    visible.sort_by(|a, b| a.0.total_cmp(&b.0));
    let dropped = visible.len().saturating_sub(MAX_LIGHTS);
    visible.truncate(MAX_LIGHTS);

    let mut raws: Vec<LightRaw> = visible.iter().map(|(_, l)| LightRaw::new(l)).collect();
    let spot_shadows = fit_spot_shadows(&visible, &mut raws);

    let near = view.near;
    let far = visible
        .iter()
        .map(|(_, l)| {
            let (center, radius) = l.bounds();
            view.depth(center) + radius
        })
        .fold(0.0_f32, f32::max)
        .min(view.far);
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); CLUSTER_COUNT];
    if far > near {
        for (index, (_, light)) in visible.iter().enumerate() {
            let (center, radius) = light.bounds();
            bin(&view, near, far, center, radius, index as u32, &mut lists);
        }
    }

    let mut grid = vec![0_u32; CLUSTER_COUNT];
    let mut overflow = 0;
    for (c, list) in lists.iter().enumerate() {
        let offset = grid.len() - CLUSTER_COUNT;
        let room = GRID_TEXELS - grid.len();
        let count = list.len().min(room);
        overflow += list.len() - count;
        grid[c] = offset as u32 | (count as u32) << COUNT_SHIFT;
        grid.extend_from_slice(&list[..count]);
    }
    Clustered {
        stats: LightStats {
            submitted: lights.len(),
            visible: raws.len(),
            dropped,
            max_per_cluster: lists.iter().map(Vec::len).max().unwrap_or(0),
            overflow,
        },
        lights: raws,
        grid,
        near,
        far: if far > near { far } else { 0.0 },
        spot_shadows,
    }
}

/// Adds light `index` to every cluster its bounding sphere may touch: the
/// slices its depths span, and the tiles under its view-space bounding box
/// (the whole screen once it reaches the near plane).
fn bin(
    view: &View,
    near: f32,
    far: f32,
    center: Vec3,
    radius: f32,
    index: u32,
    lists: &mut [Vec<u32>],
) {
    let depth = view.depth(center);
    let (z0, z1) = (
        slice(depth - radius, near, far),
        slice((depth + radius).min(far), near, far),
    );
    let ((x0, x1), (y0, y1)) = if depth - radius <= near {
        ((0, CLUSTERS_X - 1), (0, CLUSTERS_Y - 1))
    } else {
        let x = view.right.dot(center - view.eye);
        let y = view.up.dot(center - view.eye);
        let (mut lo, mut hi) = ((u32::MAX, u32::MAX), (0, 0));
        for corner in 0..8 {
            let pick = |bit: u32| if corner & bit == 0 { -radius } else { radius };
            let p = view.eye
                + view.right * (x + pick(1))
                + view.up * (y + pick(2))
                + view.forward * (depth + pick(4));
            let ndc = view.view_proj.project_point3(p);
            let (x, y) = tile(ndc.x, ndc.y);
            lo = (lo.0.min(x), lo.1.min(y));
            hi = (hi.0.max(x), hi.1.max(y));
        }
        ((lo.0, hi.0), (lo.1, hi.1))
    };
    for z in z0..=z1 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                lists[cluster_index(x, y, z)].push(index);
            }
        }
    }
}

/// Gives the nearest shadow-casting spots (they come nearest first) a
/// shadow map layer each, and returns their light view-projections.
fn fit_spot_shadows(visible: &[(f32, &Light)], raws: &mut [LightRaw]) -> Vec<Mat4> {
    let mut matrices = Vec::new();
    for ((_, light), raw) in visible.iter().zip(raws.iter_mut()) {
        if matrices.len() == MAX_SPOT_SHADOWS || !light.shadows {
            continue;
        }
        let LightKind::Spot {
            direction,
            outer_angle,
            ..
        } = light.kind
        else {
            continue;
        };
        let (view_proj, texel) = spot_projection(
            light.position,
            direction.normalize_or(Vec3::NEG_Y),
            outer_angle.clamp(1e-3, MAX_SPOT_ANGLE),
            light.range,
        );
        raw.color[3] = matrices.len() as f32;
        raw.spot[1] = texel;
        matrices.push(view_proj);
    }
    matrices
}

/// A perspective light view-projection covering a spot's cone, and the size
/// of one shadow texel per metre of distance from the light.
fn spot_projection(position: Vec3, direction: Vec3, angle: f32, range: f32) -> (Mat4, f32) {
    // A little wider than the cone, so PCF at its edge stays inside the map.
    let fov = (2.0 * angle + 0.05).min(2.0 * MAX_SPOT_ANGLE + 0.05);
    let up = if direction.y.abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let near = (range * 0.01).max(0.02);
    let light_view = view::look_at_mat4(position, position + direction, up);
    let projection = proj::directx::perspective(fov, 1.0, near, range);
    let texel = 2.0 * (fov * 0.5).tan() / SPOT_SHADOW_SIZE as f32;
    (projection * light_view, texel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4Swizzles;

    /// A 16:9 camera at `eye` looking along -z (near 0.1, far 500), like
    /// the engine's.
    fn camera(eye: Vec3) -> Mat4 {
        let projection = proj::directx::perspective(60_f32.to_radians(), 16.0 / 9.0, 0.1, 500.0);
        projection * view::look_at_mat4(eye, eye + Vec3::NEG_Z, Vec3::Y)
    }

    /// The cluster the shader puts world point `p` in (mirrors `cluster_of`
    /// in scene.wgsl): its pixel's tile, its view depth's slice.
    fn cluster_of(clustered: &Clustered, view_proj: Mat4, eye: Vec3, p: Vec3) -> Option<usize> {
        let clip = view_proj * p.extend(1.0);
        let ndc = clip.xyz() / clip.w;
        let depth = View::new(view_proj, eye).depth(p);
        if clustered.far <= 0.0 || depth >= clustered.far {
            return None;
        }
        let (x, y) = tile(ndc.x, ndc.y);
        Some(cluster_index(
            x,
            y,
            slice(depth, clustered.near, clustered.far),
        ))
    }

    fn lights_in(clustered: &Clustered, cluster: usize) -> Vec<u32> {
        let header = clustered.grid[cluster];
        let offset = (header & ((1 << COUNT_SHIFT) - 1)) as usize + CLUSTER_COUNT;
        let count = (header >> COUNT_SHIFT) as usize;
        clustered.grid[offset..offset + count].to_vec()
    }

    /// A deterministic spread of numbers in [0, 1).
    fn spread(i: usize, k: f32) -> f32 {
        (i as f32 * k).fract()
    }

    #[test]
    fn slices_run_from_near_to_far_in_order() {
        let (near, far) = (0.1, 100.0);
        assert_eq!(slice(near, near, far), 0);
        assert_eq!(slice(0.01, near, far), 0);
        assert_eq!(slice(far * 0.9999, near, far), CLUSTERS_Z - 1);
        assert_eq!(slice(far * 10.0, near, far), CLUSTERS_Z - 1);
        let mut last = 0;
        for i in 0..1000 {
            let s = slice(near + i as f32 * 0.1, near, far);
            assert!(s >= last);
            last = s;
        }
    }

    #[test]
    fn cone_bounds_hold_the_whole_cone_and_beat_the_range_sphere() {
        for degrees in [5.0_f32, 20.0, 44.0, 46.0, 60.0, 80.0] {
            let angle = degrees.to_radians();
            let (tip, dir, range) = (Vec3::new(1.0, 2.0, 3.0), Vec3::NEG_Y, 4.0);
            let (center, radius) = cone_bounds(tip, dir, angle, range);
            assert!(radius < range, "{degrees}°: {radius}");
            let inside = |p: Vec3| p.distance(center) <= radius + 1e-4;
            assert!(inside(tip));
            // Points over the spherical cap and down the cone's edge.
            for i in 0..64 {
                let theta = angle * i as f32 / 63.0;
                let phi = std::f32::consts::TAU * spread(i, 0.618);
                let side = Vec3::new(phi.cos(), 0.0, phi.sin());
                let ray = dir * theta.cos() + side * theta.sin();
                for t in [0.25, 0.5, 1.0] {
                    assert!(inside(tip + ray * range * t), "{degrees}° θ {theta}");
                }
            }
        }
    }

    #[test]
    fn every_lit_point_finds_its_light_in_its_cluster() {
        let eye = Vec3::new(0.0, 2.0, 10.0);
        let view_proj = camera(eye);
        let lights: Vec<Light> = (0..40)
            .map(|i| {
                let position = Vec3::new(
                    spread(i, 0.37) * 16.0 - 8.0,
                    spread(i, 0.71) * 4.0,
                    spread(i, 0.53) * -30.0 + 9.0,
                );
                let mut light = Light::point(position, Vec3::ONE, 5.0, 1.0 + spread(i, 0.29) * 5.0);
                if i % 3 == 0 {
                    light.kind = LightKind::Spot {
                        direction: Vec3::new(spread(i, 0.11) - 0.5, -1.0, 0.2).normalize(),
                        inner_angle: 0.2,
                        outer_angle: 0.3 + spread(i, 0.43),
                    };
                }
                light
            })
            .collect();
        let clustered = cluster(&lights, view_proj, eye);
        assert_eq!(clustered.stats.overflow, 0);
        let mut checked = 0;
        for i in 0..20_000 {
            // Points within each light's range, kept if the camera sees them.
            let light = &lights[i % lights.len()];
            let dir = Vec3::new(
                spread(i, 0.754) - 0.5,
                spread(i, 0.569) - 0.5,
                spread(i, 0.321) - 0.5,
            )
            .normalize_or(Vec3::X);
            let p = light.position + dir * light.range * spread(i, 0.887) * 0.999;
            if let LightKind::Spot {
                direction,
                outer_angle,
                ..
            } = light.kind
                && dir.dot(direction) < outer_angle.cos()
            {
                continue;
            }
            let clip = view_proj * p.extend(1.0);
            let ndc = clip.xyz() / clip.w;
            if clip.w <= 0.1 || ndc.x.abs() > 1.0 || ndc.y.abs() > 1.0 || ndc.z > 1.0 {
                continue;
            }
            let index = clustered
                .lights
                .iter()
                .position(|raw| raw.position_range[..3] == light.position.to_array()[..])
                .expect("a light lighting a visible point is visible")
                as u32;
            let cluster = cluster_of(&clustered, view_proj, eye, p).expect("inside the slices");
            assert!(
                lights_in(&clustered, cluster).contains(&index),
                "point {p} of light {index} not in cluster {cluster}"
            );
            checked += 1;
        }
        assert!(checked > 5_000, "only {checked} points checked");
    }

    #[test]
    fn lights_out_of_view_are_culled() {
        let eye = Vec3::ZERO;
        let view_proj = camera(eye);
        let behind = Light::point(Vec3::new(0.0, 0.0, 5.0), Vec3::ONE, 1.0, 2.0);
        let beyond_far = Light::point(Vec3::new(0.0, 0.0, -600.0), Vec3::ONE, 1.0, 50.0);
        let off_to_the_side = Light::point(Vec3::new(50.0, 0.0, -5.0), Vec3::ONE, 1.0, 3.0);
        let dark = Light::point(Vec3::new(0.0, 0.0, -5.0), Vec3::ONE, 0.0, 3.0);
        let seen = Light::point(Vec3::new(0.0, 0.0, -5.0), Vec3::ONE, 1.0, 3.0);
        let clustered = cluster(
            &[behind, beyond_far, off_to_the_side, dark, seen],
            view_proj,
            eye,
        );
        assert_eq!(clustered.lights.len(), 1);
        assert_eq!(clustered.lights[0].position_range, [0.0, 0.0, -5.0, 3.0]);
        assert_eq!(clustered.stats.submitted, 5);
        // The slices end where the light's reach does.
        assert!((clustered.far - 8.0).abs() < 1e-3, "{}", clustered.far);
    }

    #[test]
    fn no_lights_means_empty_clusters() {
        let clustered = cluster(&[], camera(Vec3::ZERO), Vec3::ZERO);
        assert!(clustered.lights.is_empty());
        assert_eq!(clustered.far, 0.0);
        assert!(clustered.grid.iter().all(|&h| h == 0));
    }

    #[test]
    fn over_the_cap_the_farthest_lights_go() {
        let eye = Vec3::ZERO;
        let lights: Vec<Light> = (0..MAX_LIGHTS + 40)
            .map(|i| {
                Light::point(
                    Vec3::new(0.0, 0.0, -1.0 - i as f32 * 0.5),
                    Vec3::ONE,
                    1.0,
                    0.2,
                )
            })
            .collect();
        let clustered = cluster(&lights, camera(eye), eye);
        assert_eq!(clustered.lights.len(), MAX_LIGHTS);
        assert_eq!(clustered.stats.dropped, 40);
        let farthest = clustered
            .lights
            .iter()
            .map(|raw| -raw.position_range[2])
            .fold(0.0, f32::max);
        assert!(farthest <= 1.0 + (MAX_LIGHTS - 1) as f32 * 0.5 + 1e-3);
    }

    #[test]
    fn a_light_around_the_camera_covers_the_whole_screen() {
        let eye = Vec3::ZERO;
        let clustered = cluster(
            &[Light::point(Vec3::new(0.0, 0.0, 0.5), Vec3::ONE, 1.0, 3.0)],
            camera(eye),
            eye,
        );
        for (x, y) in [(0, 0), (CLUSTERS_X - 1, 0), (0, CLUSTERS_Y - 1), (7, 4)] {
            assert_eq!(lights_in(&clustered, cluster_index(x, y, 0)), [0]);
        }
    }

    #[test]
    fn the_nearest_shadowed_spots_get_shadow_layers() {
        let eye = Vec3::ZERO;
        let spot = |z: f32, shadows: bool| Light {
            kind: LightKind::Spot {
                direction: Vec3::NEG_Y,
                inner_angle: 0.3,
                outer_angle: 0.5,
            },
            shadows,
            ..Light::point(Vec3::new(0.0, 2.0, z), Vec3::ONE, 1.0, 4.0)
        };
        let mut lights: Vec<Light> = (0..6).map(|i| spot(-5.0 - i as f32 * 3.0, true)).collect();
        lights.push(spot(-3.0, false));
        lights.push(Light {
            shadows: true,
            ..Light::point(Vec3::new(0.0, 0.0, -2.0), Vec3::ONE, 1.0, 1.0)
        });
        let clustered = cluster(&lights, camera(eye), eye);
        assert_eq!(clustered.spot_shadows.len(), MAX_SPOT_SHADOWS);
        let mut layers: Vec<(f32, f32)> = clustered
            .lights
            .iter()
            .map(|raw| (raw.position_range[2], raw.color[3]))
            .collect();
        layers.sort_by(|a, b| b.0.total_cmp(&a.0));
        // The point light and the unshadowed spot get none, the four nearest
        // shadowed spots a layer each in order, the rest none.
        assert_eq!(
            layers,
            [
                (-2.0, -1.0),
                (-3.0, -1.0),
                (-5.0, 0.0),
                (-8.0, 1.0),
                (-11.0, 2.0),
                (-14.0, 3.0),
                (-17.0, -1.0),
                (-20.0, -1.0),
            ]
        );
        // The spot's own shadow frustum sees the floor point under it.
        let below = clustered.spot_shadows[0] * Vec3::new(0.0, 0.0, -5.0).extend(1.0);
        let ndc = below.xyz() / below.w;
        assert!(ndc.x.abs() < 1e-3 && ndc.y.abs() < 1e-3 && (0.0..1.0).contains(&ndc.z));
    }

    #[test]
    fn spot_cones_reach_the_shader_in_gltf_form() {
        let raw = LightRaw::new(&Light {
            kind: LightKind::Spot {
                direction: Vec3::new(0.0, -2.0, 0.0),
                inner_angle: 0.2,
                outer_angle: 0.4,
            },
            ..Light::point(Vec3::ZERO, Vec3::new(1.0, 0.5, 0.25), 4.0, 6.0)
        });
        assert_eq!(raw.color, [4.0, 2.0, 1.0, -1.0]);
        assert_eq!(&raw.direction[..3], &[0.0, -1.0, 0.0]);
        let falloff = |cos: f32| (cos * raw.direction[3] + raw.spot[0]).clamp(0.0, 1.0);
        assert!((falloff(0.2_f32.cos()) - 1.0).abs() < 1e-4);
        assert!(falloff(0.4_f32.cos()).abs() < 1e-4);
        let point = LightRaw::new(&Light::point(Vec3::ZERO, Vec3::ONE, 1.0, 1.0));
        assert_eq!((point.direction[3], point.spot[0]), (0.0, 1.0));
    }
}
