//! Box–box contacts: separating-axis test (SAT) picks the contact normal, then
//! Sutherland–Hodgman clipping builds the manifold (face contacts) or the closest
//! points between two edges give a single contact (edge contacts).
//! References: Ericson, *Real-Time Collision Detection* §4.4 / §5.1.9; Gregorius,
//! "Robust Contact Creation for Physics Simulations" (GDC 2015).

use glam::{Mat3, Quat, Vec3};

use crate::contact::Geometry;

/// An oriented box in world space.
#[derive(Debug, Clone, Copy)]
struct Obb {
    center: Vec3,
    axes: Mat3,
    half: Vec3,
}

impl Obb {
    fn new(center: Vec3, orientation: Quat, half: Vec3) -> Self {
        Self {
            center,
            axes: Mat3::from_quat(orientation),
            half,
        }
    }

    fn axis(&self, i: usize) -> Vec3 {
        self.axes.col(i)
    }

    /// Half-width of the box's projection onto unit `direction`.
    fn radius_along(&self, direction: Vec3) -> f32 {
        (0..3)
            .map(|i| self.half[i] * self.axis(i).dot(direction).abs())
            .sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Feature {
    FaceA(usize),
    FaceB(usize),
    Edges(usize, usize),
}

/// A face axis wins unless an edge axis is clearly shallower: prevents flicker
/// between near-equal candidates and prefers the stabler face manifolds.
const EDGE_TOLERANCE_RELATIVE: f32 = 0.95;
const EDGE_TOLERANCE_ABSOLUTE: f32 = 0.01;

/// Contacts between box A and box B. Normals point from B toward A.
pub(crate) fn box_box(
    center_a: Vec3,
    orientation_a: Quat,
    half_a: Vec3,
    center_b: Vec3,
    orientation_b: Quat,
    half_b: Vec3,
) -> Vec<Geometry> {
    let a = Obb::new(center_a, orientation_a, half_a);
    let b = Obb::new(center_b, orientation_b, half_b);
    let offset = a.center - b.center;

    // Least-overlap axis among face axes, then among edge axes.
    let mut best_face: Option<(f32, Vec3, Feature)> = None;
    let mut best_edge: Option<(f32, Vec3, Feature)> = None;
    let candidates = (0..3)
        .map(|i| (a.axis(i), Feature::FaceA(i)))
        .chain((0..3).map(|i| (b.axis(i), Feature::FaceB(i))))
        .chain((0..9).map(|k| {
            let (i, j) = (k / 3, k % 3);
            (a.axis(i).cross(b.axis(j)), Feature::Edges(i, j))
        }));
    for (axis, feature) in candidates {
        let length = axis.length();
        if length < 1e-4 {
            continue; // parallel edges: no new axis (face axes cover it)
        }
        let axis = axis / length;
        let distance = offset.dot(axis);
        let overlap = a.radius_along(axis) + b.radius_along(axis) - distance.abs();
        if overlap < 0.0 {
            return Vec::new(); // separating axis found
        }
        // Orient from B toward A.
        let normal = if distance < 0.0 { -axis } else { axis };
        let best = if matches!(feature, Feature::Edges(..)) {
            &mut best_edge
        } else {
            &mut best_face
        };
        if best.is_none_or(|(o, _, _)| overlap < o) {
            *best = Some((overlap, normal, feature));
        }
    }
    let Some((face_overlap, face_normal, face)) = best_face else {
        return Vec::new();
    };
    let (depth, normal, feature) = match best_edge {
        Some((o, n, f)) if o < EDGE_TOLERANCE_RELATIVE * face_overlap - EDGE_TOLERANCE_ABSOLUTE => {
            (o, n, f)
        }
        _ => (face_overlap, face_normal, face),
    };

    match feature {
        // Reference face on A faces B, i.e. along -normal.
        Feature::FaceA(i) => reduce_to_four(clip_faces(&a, i, -normal, &b), normal)
            .into_iter()
            .map(|(point, d)| (normal, d, point))
            .collect(),
        Feature::FaceB(i) => reduce_to_four(clip_faces(&b, i, normal, &a), normal)
            .into_iter()
            .map(|(point, d)| (normal, d, point))
            .collect(),
        Feature::Edges(i, j) => {
            // Supporting edge of A furthest toward B, and of B toward A.
            let edge_a = support_edge(&a, i, -normal);
            let edge_b = support_edge(&b, j, normal);
            let (pa, pb) = closest_points_on_segments(edge_a, edge_b);
            vec![(normal, depth, (pa + pb) * 0.5)]
        }
    }
}

/// Midpoint and half-length of box edge along `axis` that is furthest along `direction`.
fn support_edge(obb: &Obb, axis: usize, direction: Vec3) -> (Vec3, Vec3, f32) {
    let mut point = obb.center;
    for k in (0..3).filter(|&k| k != axis) {
        let a = obb.axis(k);
        point += a * obb.half[k] * a.dot(direction).signum();
    }
    (point, obb.axis(axis), obb.half[axis])
}

/// Closest points between segments `center ± direction·half` (directions unit length).
pub(crate) fn closest_points_on_segments(
    s1: (Vec3, Vec3, f32),
    s2: (Vec3, Vec3, f32),
) -> (Vec3, Vec3) {
    let ((p1, d1, h1), (p2, d2, h2)) = (s1, s2);
    let r = p1 - p2;
    let b = d1.dot(d2);
    let (c, f) = (d1.dot(r), d2.dot(r));
    let denom = 1.0 - b * b;
    let mut s = if denom > 1e-6 {
        ((b * f - c) / denom).clamp(-h1, h1)
    } else {
        0.0
    };
    let t = (b * s + f).clamp(-h2, h2);
    s = (b * t - c).clamp(-h1, h1);
    (p1 + d1 * s, p2 + d2 * t)
}

/// Clips `incident`'s face most opposed to `face_normal` against the reference
/// face (axis `axis` of `reference`, outward normal `face_normal`). Returns points
/// below the reference face as `(midpoint of overlap, depth)`.
fn clip_faces(reference: &Obb, axis: usize, face_normal: Vec3, incident: &Obb) -> Vec<(Vec3, f32)> {
    let face_center = reference.center + face_normal * reference.half[axis];

    // Incident face: the one whose outward normal points most against face_normal.
    let (j, sign) = (0..3)
        .map(|j| (j, incident.axis(j).dot(face_normal)))
        .max_by(|x, y| x.1.abs().total_cmp(&y.1.abs()))
        .map(|(j, d)| (j, -d.signum()))
        .expect("three axes");
    let incident_center = incident.center + incident.axis(j) * incident.half[j] * sign;
    let (u, v) = match j {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let (eu, ev) = (
        incident.axis(u) * incident.half[u],
        incident.axis(v) * incident.half[v],
    );
    let mut polygon = vec![
        incident_center + eu + ev,
        incident_center - eu + ev,
        incident_center - eu - ev,
        incident_center + eu - ev,
    ];

    // Side planes of the reference face.
    for k in (0..3).filter(|&k| k != axis) {
        let side = reference.axis(k);
        let limit = reference.half[k];
        for normal in [side, -side] {
            let offset = normal.dot(face_center) + limit;
            polygon = clip_polygon(&polygon, normal, offset);
            if polygon.is_empty() {
                return Vec::new();
            }
        }
    }

    polygon
        .into_iter()
        .filter_map(|p| {
            let separation = (p - face_center).dot(face_normal);
            (separation < 0.0).then(|| (p - face_normal * (separation * 0.5), -separation))
        })
        .collect()
}

/// Keeps at most 4 points: the deepest, the one farthest from it, then the two
/// spanning the largest area on either side of that line. Clipping slightly twisted
/// boxes yields clusters of near-duplicate points; redundant points let the solver
/// shuffle impulse between them without converging (persistent stack jitter).
fn reduce_to_four(points: Vec<(Vec3, f32)>, normal: Vec3) -> Vec<(Vec3, f32)> {
    if points.len() <= 4 {
        return points;
    }
    let argmax = |key: &dyn Fn(&(Vec3, f32)) -> f32| {
        (0..points.len())
            .max_by(|&i, &j| key(&points[i]).total_cmp(&key(&points[j])))
            .expect("non-empty")
    };
    let first = argmax(&|p| p.1);
    let p0 = points[first].0;
    let second = argmax(&|p| p.0.distance_squared(p0));
    let p1 = points[second].0;
    let signed_area = |p: &(Vec3, f32)| normal.dot((p1 - p0).cross(p.0 - p0));
    let third = argmax(&|p| signed_area(p));
    let fourth = argmax(&|p| -signed_area(p));
    let mut keep = vec![first, second, third, fourth];
    keep.sort_unstable();
    keep.dedup();
    keep.into_iter().map(|i| points[i]).collect()
}

/// Sutherland–Hodgman: keeps the part of `polygon` where `normal·p <= offset`.
fn clip_polygon(polygon: &[Vec3], normal: Vec3, offset: f32) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(polygon.len() + 1);
    for (i, &current) in polygon.iter().enumerate() {
        let next = polygon[(i + 1) % polygon.len()];
        let (dc, dn) = (normal.dot(current) - offset, normal.dot(next) - offset);
        // One definition of "inside" (<= 0) for both checks: points exactly on the
        // plane (common for aligned boxes) must not also emit a zero-length crossing.
        let (current_inside, next_inside) = (dc <= 0.0, dn <= 0.0);
        if current_inside {
            out.push(current);
        }
        if current_inside != next_inside {
            out.push(current + (next - current) * (dc / (dc - dn)));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_PI_4;

    use super::*;

    const UNIT: Vec3 = Vec3::splat(0.5);

    fn contacts(ca: Vec3, qa: Quat, cb: Vec3, qb: Quat) -> Vec<Geometry> {
        box_box(ca, qa, UNIT, cb, qb, UNIT)
    }

    #[test]
    fn stacked_cubes_have_four_face_contacts() {
        let hits = contacts(
            Vec3::new(0.0, 0.9, 0.0),
            Quat::IDENTITY,
            Vec3::ZERO,
            Quat::IDENTITY,
        );
        assert_eq!(hits.len(), 4);
        for (normal, depth, point) in hits {
            assert!(
                normal.abs_diff_eq(Vec3::Y, 1e-5),
                "normal {normal} must point from B to A"
            );
            assert!((depth - 0.1).abs() < 1e-5);
            assert!((point.x.abs() - 0.5).abs() < 1e-5 && (point.z.abs() - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn swapping_boxes_flips_the_normal() {
        let hits = contacts(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(0.0, 0.9, 0.0),
            Quat::IDENTITY,
        );
        assert!(!hits.is_empty() && hits.iter().all(|h| h.0.abs_diff_eq(Vec3::NEG_Y, 1e-5)));
    }

    #[test]
    fn offset_stack_clips_to_the_overlap() {
        let hits = contacts(
            Vec3::new(0.6, 0.9, 0.0),
            Quat::IDENTITY,
            Vec3::ZERO,
            Quat::IDENTITY,
        );
        assert_eq!(hits.len(), 4);
        for (_, _, p) in hits {
            assert!(
                (0.1 - 1e-5..=0.5 + 1e-5).contains(&p.x),
                "point {p} outside the overlap"
            );
        }
    }

    #[test]
    fn rotated_stack_reduces_its_octagon_to_four_spread_points() {
        let hits = contacts(
            Vec3::new(0.0, 0.9, 0.0),
            Quat::from_rotation_y(FRAC_PI_4),
            Vec3::ZERO,
            Quat::IDENTITY,
        );
        assert_eq!(hits.len(), 4, "{hits:?}");
        assert!(hits.iter().all(|h| (h.1 - 0.1).abs() < 1e-4));
        // Well spread: the kept points span most of the ~1 m wide overlap.
        let widest = hits
            .iter()
            .flat_map(|a| hits.iter().map(move |b| a.2.distance(b.2)))
            .fold(0.0f32, f32::max);
        assert!(widest > 0.9, "points bunched together: {hits:?}");
    }

    #[test]
    fn slightly_twisted_stack_has_no_near_duplicate_points() {
        let hits = contacts(
            Vec3::new(0.0, 0.9, 0.0),
            Quat::from_rotation_y(0.003),
            Vec3::ZERO,
            Quat::IDENTITY,
        );
        assert!(hits.len() <= 4, "{} points", hits.len());
        for (i, a) in hits.iter().enumerate() {
            for b in &hits[i + 1..] {
                assert!(
                    a.2.distance(b.2) > 0.05,
                    "near-duplicates {:?} {:?}",
                    a.2,
                    b.2
                );
            }
        }
    }

    #[test]
    fn crossed_edges_give_one_edge_contact() {
        let a = Quat::from_rotation_x(FRAC_PI_4); // lowest edge along x
        let b = Quat::from_rotation_z(FRAC_PI_4); // highest edge along z
        let reach = 0.5 * 2f32.sqrt();
        let hits = contacts(Vec3::new(0.0, 2.0 * reach - 0.1, 0.0), a, Vec3::ZERO, b);
        assert_eq!(hits.len(), 1, "{hits:?}");
        let (normal, depth, point) = hits[0];
        assert!(normal.abs_diff_eq(Vec3::Y, 1e-4));
        assert!((depth - 0.1).abs() < 1e-4);
        assert!(point.x.abs() < 1e-4 && point.z.abs() < 1e-4);
    }

    #[test]
    fn separated_boxes_have_no_contacts() {
        assert!(
            contacts(
                Vec3::new(0.0, 1.1, 0.0),
                Quat::IDENTITY,
                Vec3::ZERO,
                Quat::IDENTITY
            )
            .is_empty()
        );
        // Only an edge–edge axis separates these: face axes all overlap.
        let a = Quat::from_rotation_x(FRAC_PI_4);
        let b = Quat::from_rotation_z(FRAC_PI_4);
        let reach = 0.5 * 2f32.sqrt();
        assert!(contacts(Vec3::new(0.0, 2.0 * reach + 0.05, 0.0), a, Vec3::ZERO, b).is_empty());
    }
}
