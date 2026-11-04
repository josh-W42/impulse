//! Which instances each pass draws (frustum culling) and at what detail
//! (LOD selection), on the CPU.

use glam::{Mat4, Vec3, Vec4};

use crate::draw_list::{Instance, InstanceRaw};
use crate::lights::MAX_SPOT_SHADOWS;
use crate::mesh::MeshData;
use crate::shadow::CASCADES;

/// A level of detail is used once its error spans at most this many pixels.
pub const LOD_PIXELS: f32 = 1.0;

/// The camera's pass (main and AO prepass), then each cascade, then each
/// shadowed spot.
pub(crate) const MAX_PASSES: usize = 1 + CASCADES + MAX_SPOT_SHADOWS;
pub(crate) const CAMERA_PASS: usize = 0;
pub(crate) const fn cascade_pass(i: usize) -> usize {
    1 + i
}
pub(crate) const fn spot_pass(i: usize) -> usize {
    1 + CASCADES + i
}

/// A bounding sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub center: Vec3,
    pub radius: f32,
}

impl Bounds {
    /// Around `positions`: their box's centre and the farthest of them from
    /// it (a point at the origin for none).
    pub fn of(positions: impl Iterator<Item = Vec3> + Clone) -> Self {
        let (lo, hi) = positions
            .clone()
            .fold((Vec3::MAX, Vec3::MIN), |(lo, hi), p| (lo.min(p), hi.max(p)));
        if lo.x > hi.x {
            return Self {
                center: Vec3::ZERO,
                radius: 0.0,
            };
        }
        let center = (lo + hi) * 0.5;
        let radius = positions.map(|p| p.distance(center)).fold(0.0, f32::max);
        Self { center, radius }
    }

    pub fn of_mesh(mesh: &MeshData) -> Self {
        Self::of(mesh.vertices.iter().map(|v| Vec3::from_array(v.position)))
    }

    /// The sphere around this one after `transform` (scaled by its largest
    /// axis, so it still covers under non-uniform scale).
    pub fn transformed(self, transform: Mat4) -> Self {
        Self {
            center: transform.transform_point3(self.center),
            radius: self.radius * max_scale(transform),
        }
    }
}

/// The longest of a transform's axes.
pub(crate) fn max_scale(transform: Mat4) -> f32 {
    transform
        .x_axis
        .truncate()
        .length()
        .max(transform.y_axis.truncate().length())
        .max(transform.z_axis.truncate().length())
}

/// A view volume as six inward-facing planes (xyz: unit normal, w: offset).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    /// The volume a view-projection with 0..1 clip depth sees
    /// (Gribb–Hartmann plane extraction).
    pub fn new(view_proj: Mat4) -> Self {
        let r = [0, 1, 2, 3].map(|i| view_proj.row(i));
        let planes = [
            r[3] + r[0],
            r[3] - r[0],
            r[3] + r[1],
            r[3] - r[1],
            r[2],
            r[3] - r[2],
        ]
        .map(|p| {
            let length = p.truncate().length();
            if length > 0.0 {
                p / length
            } else {
                // Degenerate: a plane everything is in front of.
                Vec4::W
            }
        });
        Self { planes }
    }

    /// A volume holding everything (culling off).
    pub fn everything() -> Self {
        Self {
            planes: [Vec4::W; 6],
        }
    }

    /// Whether any of `sphere` may be inside (conservative near corners).
    pub fn sees(&self, sphere: Bounds) -> bool {
        self.planes
            .iter()
            .all(|p| p.truncate().dot(sphere.center) + p.w >= -sphere.radius)
    }
}

/// How large things look from the camera, for choosing detail.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LodView {
    /// Row of the view-projection giving a point's view depth (clip w).
    depth: Vec4,
    /// Pixels one unit spans at depth 1.
    pixels_per_unit: f32,
}

impl LodView {
    /// For a perspective `view_proj` drawn `viewport_height` pixels high.
    pub fn new(view_proj: Mat4, viewport_height: u32) -> Self {
        // Row 1 is the projection's y scale times a unit row of the view.
        let y_scale = view_proj.row(1).truncate().length();
        Self {
            depth: view_proj.row(3),
            pixels_per_unit: y_scale * viewport_height as f32 * 0.5,
        }
    }

    /// Pixels `size` spans at the nearest depth of `sphere` (infinite when
    /// the sphere reaches the camera).
    fn pixels(&self, size: f32, sphere: Bounds) -> f32 {
        let depth = self.depth.dot(sphere.center.extend(1.0)) - sphere.radius;
        if depth <= 1e-4 {
            f32::INFINITY
        } else {
            size * self.pixels_per_unit / depth
        }
    }

    /// The coarsest level whose error (`errors[i]`, mesh units, ascending
    /// from the full mesh's 0) times `scale` spans at most [`LOD_PIXELS`]
    /// around `sphere` (world).
    pub fn select(&self, errors: &[f32], scale: f32, sphere: Bounds) -> usize {
        let mut level = 0;
        for (i, error) in errors.iter().enumerate().skip(1) {
            if self.pixels(error * scale, sphere) > LOD_PIXELS {
                break;
            }
            level = i;
        }
        level
    }
}

/// What a mesh needs for culling and LOD choice.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MeshCull {
    /// Around its full-detail vertices, in mesh space.
    pub bounds: Bounds,
    /// Per level, full detail first: error (mesh units) and triangles.
    pub errors: Vec<f32>,
    pub triangles: Vec<u32>,
}

/// What the last frame drew.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DrawStats {
    /// Static instances in the draw list (skinned ones aren't culled).
    pub submitted: usize,
    /// Of those, drawn by the camera.
    pub drawn: usize,
    /// Of those, outside the camera's view.
    pub culled: usize,
    /// Instances drawn into shadow maps, summed over cascades and spots.
    pub shadow_casters: usize,
    /// Triangles the camera drew (static instances).
    pub triangles: u64,
}

/// A frame's static instances sorted for drawing.
#[derive(Debug, Default)]
pub(crate) struct Buckets {
    /// Per mesh: its opaque instances, grouped by level, then by pass.
    pub instances: Vec<Vec<InstanceRaw>>,
    /// Per mesh, level and pass: (first, count) into `instances`.
    pub ranges: Vec<Vec<[(u32, u32); MAX_PASSES]>>,
    /// Blended instances the camera sees, farthest first: (mesh, level).
    pub blended: Vec<(usize, usize, InstanceRaw)>,
    pub stats: DrawStats,
}

/// Sorts `instances` into each pass that sees them (`passes[p]`; `None`:
/// the pass isn't drawn this frame) at the level `lod` picks (full detail
/// without one). `meshes[i]` is mesh i and whether its material blends;
/// instances of unknown meshes are dropped. Blended instances (blending
/// material, or alpha below 1) are drawn by the camera only.
pub(crate) fn bucket(
    instances: &[Instance],
    meshes: &[(&MeshCull, bool)],
    passes: &[Option<Frustum>; MAX_PASSES],
    lod: Option<&LodView>,
    eye: Vec3,
) -> Buckets {
    let mut lists: Vec<Vec<[Vec<InstanceRaw>; MAX_PASSES]>> = meshes
        .iter()
        .map(|(m, _)| (0..m.errors.len()).map(|_| Default::default()).collect())
        .collect();
    let mut blended = Vec::new();
    let mut stats = DrawStats::default();
    for instance in instances {
        let index = instance.mesh.0 as usize;
        let Some((mesh, blends)) = meshes.get(index) else {
            continue;
        };
        stats.submitted += 1;
        let scale = max_scale(instance.transform);
        let sphere = Bounds {
            center: instance.transform.transform_point3(mesh.bounds.center),
            radius: mesh.bounds.radius * scale,
        };
        let level = lod.map_or(0, |view| view.select(&mesh.errors, scale, sphere));
        let sees = |p: usize| passes[p].is_some_and(|f| f.sees(sphere));
        if !sees(CAMERA_PASS) && (*blends || instance.color[3] < 1.0) {
            stats.culled += 1;
            continue;
        }
        let raw = InstanceRaw::new(instance);
        if *blends || instance.color[3] < 1.0 {
            let distance = instance.transform.w_axis.truncate().distance_squared(eye);
            blended.push((distance, index, level, raw));
            stats.drawn += 1;
            stats.triangles += mesh.triangles[level] as u64;
            continue;
        }
        for (p, list) in lists[index][level].iter_mut().enumerate() {
            if sees(p) {
                list.push(raw);
                if p == CAMERA_PASS {
                    stats.drawn += 1;
                    stats.triangles += mesh.triangles[level] as u64;
                } else {
                    stats.shadow_casters += 1;
                }
            } else if p == CAMERA_PASS {
                stats.culled += 1;
            }
        }
    }
    blended.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut buckets = Buckets {
        blended: blended
            .into_iter()
            .map(|(_, mesh, level, raw)| (mesh, level, raw))
            .collect(),
        stats,
        ..Default::default()
    };
    for levels in lists {
        let mut flat = Vec::new();
        let ranges = levels
            .into_iter()
            .map(|passes| {
                passes.map(|list| {
                    let first = flat.len() as u32;
                    flat.extend(list);
                    (first, flat.len() as u32 - first)
                })
            })
            .collect();
        buckets.instances.push(flat);
        buckets.ranges.push(ranges);
    }
    buckets
}

#[cfg(test)]
mod tests {
    use glam::camera::rh::{proj, view};

    use super::*;
    use crate::draw_list::{DEFAULT_METALLIC, DEFAULT_ROUGHNESS};
    use crate::mesh::MeshId;

    fn camera() -> Mat4 {
        proj::directx::perspective(1.0, 16.0 / 9.0, 0.1, 500.0)
            * view::look_at_mat4(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y)
    }

    fn sphere(center: Vec3, radius: f32) -> Bounds {
        Bounds { center, radius }
    }

    #[test]
    fn spheres_just_outside_a_plane_are_culled_and_touching_ones_kept() {
        let f = Frustum::new(camera());
        // The near plane at 0.1, the far one at 500, sides at ±0.5 rad up/down.
        assert!(f.sees(sphere(Vec3::new(0.0, 0.0, -10.0), 0.1)));
        assert!(!f.sees(sphere(Vec3::new(0.0, 0.0, 1.0), 0.5)), "behind");
        assert!(!f.sees(sphere(Vec3::new(0.0, 0.0, 0.3), 0.39)));
        assert!(
            f.sees(sphere(Vec3::new(0.0, 0.0, 0.3), 0.41)),
            "reaches the near plane"
        );
        assert!(
            !f.sees(sphere(Vec3::new(0.0, 0.0, -501.0), 0.5)),
            "past far"
        );
        assert!(f.sees(sphere(Vec3::new(0.0, 0.0, -501.0), 1.01)));
        // Above the top plane at depth 10: the plane is at y = 10 tan 0.5.
        let top = 10.0 * 0.5f32.tan();
        // A sphere centred d above the plane along y is d cos 0.5 from it.
        let clearance = 0.5f32.cos();
        assert!(!f.sees(sphere(Vec3::new(0.0, top + 1.0, -10.0), clearance * 0.99)));
        assert!(f.sees(sphere(Vec3::new(0.0, top + 1.0, -10.0), clearance * 1.01)));
        assert!(Frustum::everything().sees(sphere(Vec3::splat(1e6), 0.0)));
    }

    #[test]
    fn everything_a_cascade_covers_survives_its_culling() {
        use crate::shadow::{ShadowRange, cascades};
        let eye = Vec3::new(3.0, 8.0, 12.0);
        let view_proj = proj::directx::perspective(1.0, 16.0 / 9.0, 0.1, 500.0)
            * view::look_at_mat4(eye, Vec3::ZERO, Vec3::Y);
        let sun = Vec3::new(-0.4, -1.0, -0.3).normalize();
        let c = cascades(view_proj, eye, sun, ShadowRange::Distance(60.0));
        let mut covered = 0;
        for i in 0..c.count {
            let f = Frustum::new(c.view_proj[i]);
            for x in -40..=40 {
                for y in -5..=20 {
                    for z in -40..=40 {
                        let p = Vec3::new(x as f32, y as f32, z as f32) * 1.5;
                        let clip = c.view_proj[i].project_point3(p);
                        let inside = clip.x.abs() <= 1.0
                            && clip.y.abs() <= 1.0
                            && (0.0..=1.0).contains(&clip.z);
                        if inside {
                            covered += 1;
                            assert!(f.sees(sphere(p, 0.0)), "cascade {i} drops {p}");
                        } else {
                            // Well outside: a small sphere there is culled.
                            let far = clip.x.abs() > 1.5 || clip.y.abs() > 1.5;
                            if far {
                                assert!(!f.sees(sphere(p, 0.01)), "cascade {i} keeps {p}");
                            }
                        }
                    }
                }
            }
        }
        assert!(covered > 1000, "the test samples the cascades ({covered})");
    }

    #[test]
    fn bounds_cover_their_points_and_scale_with_the_largest_axis() {
        let points = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(3.0, 2.0, 0.0)];
        let b = Bounds::of(points.into_iter());
        assert_eq!(b.center, Vec3::new(1.0, 1.0, 0.0));
        for p in points {
            assert!(p.distance(b.center) <= b.radius + 1e-6);
        }
        let t = Mat4::from_scale_rotation_translation(
            Vec3::new(1.0, 4.0, 2.0),
            glam::Quat::from_rotation_z(0.6),
            Vec3::new(5.0, 0.0, 0.0),
        );
        let moved = b.transformed(t);
        for p in points {
            assert!(t.transform_point3(p).distance(moved.center) <= moved.radius + 1e-5);
        }
        assert_eq!(Bounds::of(std::iter::empty()).radius, 0.0);
    }

    #[test]
    fn lod_coarsens_with_distance_and_switches_under_a_pixel() {
        let view = LodView::new(camera(), 1080);
        let errors = [0.0, 0.01, 0.05];
        let at = |z: f32| sphere(Vec3::new(0.0, 0.0, -z), 0.5);
        assert_eq!(view.select(&errors, 1.0, at(2.0)), 0);
        assert_eq!(view.select(&errors, 1.0, at(0.2)), 0, "camera inside");
        let mut last = 0;
        let mut z = 1.0;
        while z < 400.0 {
            let level = view.select(&errors, 1.0, at(z));
            assert!(level >= last, "never finer farther away");
            if level > 0 {
                assert!(view.pixels(errors[level], at(z)) <= LOD_PIXELS);
            }
            last = level;
            z *= 1.05;
        }
        assert_eq!(last, 2);
        // A bigger instance keeps detail longer: its error is bigger too.
        assert!(view.select(&errors, 10.0, at(60.0)) < view.select(&errors, 1.0, at(60.0)));
    }

    fn instance(mesh: u32, x: f32, z: f32, alpha: f32) -> Instance {
        Instance {
            mesh: MeshId(mesh),
            transform: Mat4::from_translation(Vec3::new(x, 0.0, z)),
            color: [1.0, 1.0, 1.0, alpha],
            roughness: DEFAULT_ROUGHNESS,
            metallic: DEFAULT_METALLIC,
            emissive: 0.0,
        }
    }

    fn unit_mesh(levels: usize) -> MeshCull {
        MeshCull {
            bounds: sphere(Vec3::ZERO, 0.5),
            errors: (0..levels).map(|l| l as f32 * 0.01).collect(),
            triangles: (0..levels).map(|l| 100 >> l).collect(),
        }
    }

    #[test]
    fn instances_go_to_each_pass_that_sees_them_grouped_by_level() {
        let mesh = unit_mesh(2);
        let glass = unit_mesh(1);
        let meshes = [(&mesh, false), (&glass, true)];
        let mut passes = [None; MAX_PASSES];
        passes[CAMERA_PASS] = Some(Frustum::new(camera()));
        // A "shadow" pass looking the other way.
        let behind = proj::directx::perspective(1.0, 1.0, 0.1, 500.0)
            * view::look_at_mat4(Vec3::ZERO, Vec3::Z, Vec3::Y);
        passes[cascade_pass(0)] = Some(Frustum::new(behind));
        let list = [
            instance(0, 0.0, -3.0, 1.0),   // in view, near: full detail
            instance(0, 0.0, 3.0, 1.0),    // behind: casts only
            instance(0, 0.0, -300.0, 1.0), // in view, far: coarse
            instance(1, 0.0, -5.0, 1.0),   // blends, in view
            instance(1, 0.0, 5.0, 1.0),    // blends, behind: dropped
            instance(0, 0.0, -9.0, 0.5),   // translucent: blends
            instance(7, 0.0, -3.0, 1.0),   // unknown mesh
        ];
        let view = LodView::new(camera(), 1080);
        let b = bucket(&list, &meshes, &passes, Some(&view), Vec3::ZERO);
        let zs = |mesh: usize, level: usize, pass: usize| {
            let (first, count) = b.ranges[mesh][level][pass];
            b.instances[mesh][first as usize..(first + count) as usize]
                .iter()
                .map(|r| r.model[3][2])
                .collect::<Vec<_>>()
        };
        assert_eq!(zs(0, 0, CAMERA_PASS), vec![-3.0]);
        assert_eq!(zs(0, 1, CAMERA_PASS), vec![-300.0]);
        assert_eq!(zs(0, 0, cascade_pass(0)), vec![3.0]);
        assert!(zs(0, 0, cascade_pass(1)).is_empty(), "unused pass");
        let blended: Vec<_> = b
            .blended
            .iter()
            .map(|(m, _, r)| (*m, r.model[3][2]))
            .collect();
        assert_eq!(blended, vec![(0, -9.0), (1, -5.0)], "far to near");
        assert_eq!(
            b.stats,
            DrawStats {
                submitted: 6,
                drawn: 4,
                culled: 2,
                shadow_casters: 1,
                triangles: 100 + 50 + 100 + 100,
            }
        );
    }

    #[test]
    fn blending_sorts_by_distance_from_the_eye_not_the_origin() {
        let mesh = unit_mesh(1);
        let mut passes = [None; MAX_PASSES];
        passes[CAMERA_PASS] = Some(Frustum::everything());
        let list = [instance(0, -5.0, 0.0, 0.5), instance(0, 5.0, 0.0, 0.5)];
        let eye = Vec3::new(6.0, 0.0, 0.0);
        let b = bucket(&list, &[(&mesh, false)], &passes, None, eye);
        let xs: Vec<_> = b.blended.iter().map(|(_, _, r)| r.model[3][0]).collect();
        assert_eq!(xs, vec![-5.0, 5.0]);
    }

    #[test]
    fn without_culling_or_lod_every_pass_draws_everything_at_full_detail() {
        let mesh = unit_mesh(3);
        let mut passes = [None; MAX_PASSES];
        passes[CAMERA_PASS] = Some(Frustum::everything());
        passes[spot_pass(0)] = Some(Frustum::everything());
        let list = [instance(0, 0.0, 3.0, 1.0), instance(0, 0.0, -300.0, 1.0)];
        let b = bucket(&list, &[(&mesh, false)], &passes, None, Vec3::ZERO);
        assert_eq!(b.ranges[0][0][CAMERA_PASS], (0, 2));
        assert_eq!(b.ranges[0][0][spot_pass(0)], (2, 2));
        assert!(b.ranges[0][1..].iter().flatten().all(|r| r.1 == 0));
        assert_eq!(
            (b.stats.drawn, b.stats.culled, b.stats.shadow_casters),
            (2, 0, 2)
        );
    }
}
