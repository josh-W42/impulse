//! Capsule contacts. A capsule is a segment inflated by a radius, so most pairs
//! reduce to sphere tests against the segment; capsule–box falls back to GJK/EPA
//! when neither end sphere touches (e.g. lying across an edge).

use glam::{Quat, Vec3};

use crate::box_box::closest_points_on_segments;
use crate::contact::{Geometry, sphere_box, sphere_plane, sphere_sphere};
use crate::gjk::{Convex, penetration};

/// A capsule posed in world space: segment along local Y through `center`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Capsule {
    pub center: Vec3,
    pub axis: Vec3,
    pub half_height: f32,
    pub radius: f32,
}

impl Capsule {
    pub(crate) fn new(center: Vec3, orientation: Quat, half_height: f32, radius: f32) -> Self {
        Self {
            center,
            axis: orientation * Vec3::Y,
            half_height,
            radius,
        }
    }

    pub(crate) fn ends(&self) -> [Vec3; 2] {
        let offset = self.axis * self.half_height;
        [self.center - offset, self.center + offset]
    }

    fn closest_point(&self, p: Vec3) -> Vec3 {
        let t = (p - self.center)
            .dot(self.axis)
            .clamp(-self.half_height, self.half_height);
        self.center + self.axis * t
    }

    fn convex(&self) -> Convex {
        let [a, b] = self.ends();
        Convex::Capsule {
            a,
            b,
            radius: self.radius,
        }
    }
}

/// Keeps at most two contacts: the deepest and the one farthest from it with the
/// same normal (a capsule's patch is a line), dropping near-duplicates.
fn keep_two(mut hits: Vec<Geometry>) -> Vec<Geometry> {
    let Some(first) = (0..hits.len()).max_by(|&i, &j| hits[i].1.total_cmp(&hits[j].1)) else {
        return hits;
    };
    let deepest = hits.swap_remove(first);
    // Same patch only: a slanted sliver where one end overhangs the other's cap
    // would otherwise win on distance and tilt the pair.
    let far = hits
        .into_iter()
        .filter(|h| h.0.dot(deepest.0) > 0.95 && h.2.distance(deepest.2) > 0.02)
        .max_by(|x, y| {
            x.2.distance_squared(deepest.2)
                .total_cmp(&y.2.distance_squared(deepest.2))
        });
    std::iter::once(deepest).chain(far).collect()
}

/// Capsule (A) resting on a plane (B): one contact per end sphere below it.
pub(crate) fn capsule_plane(
    capsule: Capsule,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Vec<Geometry> {
    capsule
        .ends()
        .into_iter()
        .filter_map(|end| sphere_plane(end, capsule.radius, plane_point, plane_normal))
        .collect()
}

/// Sphere (A) against a capsule (B). The normal points from the capsule to the sphere.
pub(crate) fn sphere_capsule(center: Vec3, radius: f32, capsule: Capsule) -> Option<Geometry> {
    sphere_sphere(
        center,
        radius,
        capsule.closest_point(center),
        capsule.radius,
    )
}

/// Capsule A against capsule B (normal B → A), up to two points so parallel
/// capsules lying on each other don't pivot.
pub(crate) fn capsule_capsule(a: Capsule, b: Capsule) -> Vec<Geometry> {
    let mut hits = Vec::new();
    let (pa, pb) = closest_points_on_segments(
        (a.center, a.axis, a.half_height),
        (b.center, b.axis, b.half_height),
    );
    hits.extend(sphere_sphere(pa, a.radius, pb, b.radius));
    for end in a.ends() {
        hits.extend(sphere_capsule(end, a.radius, b));
    }
    for end in b.ends() {
        if let Some((normal, depth, point)) = sphere_capsule(end, b.radius, a) {
            hits.push((-normal, depth, point)); // computed A→B; flip to B→A
        }
    }
    keep_two(hits)
}

/// Capsule (A) against a box (B): end spheres first (two points for a capsule
/// lying on a face), else one GJK/EPA point (e.g. lying across an edge).
pub(crate) fn capsule_box(
    capsule: Capsule,
    box_center: Vec3,
    orientation: Quat,
    half: Vec3,
) -> Vec<Geometry> {
    let ends: Vec<Geometry> = capsule
        .ends()
        .into_iter()
        .filter_map(|end| sphere_box(end, capsule.radius, box_center, orientation, half))
        .collect();
    if !ends.is_empty() {
        return keep_two(ends);
    }
    let shape = capsule.convex();
    let Some((normal, depth)) = penetration(&shape, &Convex::cuboid(box_center, orientation, half))
    else {
        return Vec::new();
    };
    // Deepest capsule point into the box, moved to the middle of the overlap.
    let point = shape.support(-normal) + normal * (depth * 0.5);
    vec![(normal, depth, point)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lying(center: Vec3) -> Capsule {
        // Axis along x.
        Capsule::new(
            center,
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            1.0,
            0.25,
        )
    }

    #[test]
    fn lying_capsule_touches_a_plane_at_both_ends() {
        let hits = capsule_plane(lying(Vec3::new(0.0, 0.2, 0.0)), Vec3::ZERO, Vec3::Y);
        assert_eq!(hits.len(), 2);
        assert!(
            hits.iter()
                .all(|h| (h.1 - 0.05).abs() < 1e-5 && h.0 == Vec3::Y)
        );
    }

    #[test]
    fn sphere_above_capsule_middle() {
        let (normal, depth, _) =
            sphere_capsule(Vec3::new(0.3, 0.6, 0.0), 0.4, lying(Vec3::ZERO)).unwrap();
        assert!(normal.abs_diff_eq(Vec3::Y, 1e-5));
        assert!((depth - 0.05).abs() < 1e-5);
    }

    #[test]
    fn parallel_capsules_lying_on_each_other_touch_at_two_points() {
        let hits = capsule_capsule(lying(Vec3::new(0.2, 0.45, 0.0)), lying(Vec3::ZERO));
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!(
            hits.iter()
                .all(|h| h.0.abs_diff_eq(Vec3::Y, 1e-4) && (h.1 - 0.05).abs() < 1e-4)
        );
    }

    #[test]
    fn crossed_capsules_touch_at_one_point() {
        let crossing = Capsule::new(
            Vec3::new(0.0, 0.45, 0.0),
            Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            1.0,
            0.25,
        );
        let hits = capsule_capsule(crossing, lying(Vec3::ZERO));
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert!(hits[0].0.abs_diff_eq(Vec3::Y, 1e-4));
    }

    #[test]
    fn capsule_across_a_box_edge_uses_gjk_epa() {
        // Box top at y = 0.5, capsule lying along z across the box's +x edge, far
        // enough out that neither end sphere touches the box.
        let across = Capsule::new(
            Vec3::new(0.6, 0.6, 0.0),
            Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            2.0,
            0.25,
        );
        let hits = capsule_box(across, Vec3::ZERO, Quat::IDENTITY, Vec3::splat(0.5));
        assert_eq!(hits.len(), 1, "{hits:?}");
        let (normal, depth, _) = hits[0];
        assert!(
            depth > 0.0 && normal.dot(Vec3::new(1.0, 1.0, 0.0).normalize()) > 0.9,
            "{normal}"
        );
    }
}
