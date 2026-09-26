//! GJK (Gilbert–Johnson–Keerthi) intersection test and EPA (Expanding Polytope
//! Algorithm) penetration depth, over support functions of convex shapes.
//! References: Ericson §9.5; "GJK: Collision detection algorithm in 2D/3D"
//! (Casey Muratori's formulation of the simplex cases); van den Bergen (EPA).

use glam::{Mat3, Quat, Vec3};

/// A convex shape posed in world space, described by its support mapping.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Convex {
    // Used by the GJK tests; convex hulls (M6b) will pair with it too.
    #[cfg_attr(not(test), allow(dead_code))]
    Sphere { center: Vec3, radius: f32 },
    Box {
        center: Vec3,
        axes: Mat3,
        half: Vec3,
    },
    /// Segment from `a` to `b`, inflated by `radius`.
    Capsule { a: Vec3, b: Vec3, radius: f32 },
}

impl Convex {
    pub(crate) fn cuboid(center: Vec3, orientation: Quat, half: Vec3) -> Self {
        Convex::Box {
            center,
            axes: Mat3::from_quat(orientation),
            half,
        }
    }

    /// Farthest point of the shape along `direction` (need not be unit length).
    pub(crate) fn support(&self, direction: Vec3) -> Vec3 {
        let unit = direction.normalize_or(Vec3::X);
        match *self {
            Convex::Sphere { center, radius } => center + unit * radius,
            Convex::Box { center, axes, half } => {
                let local = axes.transpose() * direction;
                let corner = Vec3::new(
                    half.x.copysign(local.x),
                    half.y.copysign(local.y),
                    half.z.copysign(local.z),
                );
                center + axes * corner
            }
            Convex::Capsule { a, b, radius } => {
                let end = if (b - a).dot(direction) > 0.0 { b } else { a };
                end + unit * radius
            }
        }
    }
}

/// Support of the Minkowski difference A − B.
fn support(a: &Convex, b: &Convex, direction: Vec3) -> Vec3 {
    a.support(direction) - b.support(-direction)
}

const MAX_ITERATIONS: usize = 64;

/// If A and B overlap, returns a tetrahedron (in A − B space) containing the
/// origin, ready for EPA. `None` when separated.
pub(crate) fn gjk(a: &Convex, b: &Convex) -> Option<[Vec3; 4]> {
    let mut direction = Vec3::X;
    let mut simplex: Vec<Vec3> = vec![support(a, b, direction)];
    direction = -simplex[0];
    for _ in 0..MAX_ITERATIONS {
        if direction.length_squared() < 1e-12 {
            // Origin on the simplex boundary: touching; expand to a full tetrahedron.
            return complete_tetrahedron(a, b, &simplex);
        }
        let point = support(a, b, direction);
        if point.dot(direction) < 0.0 {
            return None; // the new point didn't pass the origin: separated
        }
        simplex.push(point);
        if let Some(tetra) = next_simplex(&mut simplex, &mut direction) {
            return Some(tetra);
        }
    }
    None
}

/// Reduces the simplex to the feature nearest the origin and picks the next
/// search direction. Returns the tetrahedron once it contains the origin.
fn next_simplex(simplex: &mut Vec<Vec3>, direction: &mut Vec3) -> Option<[Vec3; 4]> {
    match simplex.len() {
        2 => {
            let (b, a) = (simplex[0], simplex[1]);
            let (ab, ao) = (b - a, -a);
            if ab.dot(ao) > 0.0 {
                *direction = ab.cross(ao).cross(ab);
            } else {
                *simplex = vec![a];
                *direction = ao;
            }
            None
        }
        3 => {
            let (c, b, a) = (simplex[0], simplex[1], simplex[2]);
            let (ab, ac, ao) = (b - a, c - a, -a);
            let abc = ab.cross(ac);
            if abc.cross(ac).dot(ao) > 0.0 {
                if ac.dot(ao) > 0.0 {
                    *simplex = vec![c, a];
                    *direction = ac.cross(ao).cross(ac);
                } else {
                    *simplex = vec![b, a];
                    return next_simplex(simplex, direction);
                }
            } else if ab.cross(abc).dot(ao) > 0.0 {
                *simplex = vec![b, a];
                return next_simplex(simplex, direction);
            } else if abc.dot(ao) > 0.0 {
                *direction = abc;
            } else {
                *simplex = vec![b, c, a];
                *direction = -abc;
            }
            None
        }
        _ => {
            let (d, c, b, a) = (simplex[0], simplex[1], simplex[2], simplex[3]);
            let (ab, ac, ad, ao) = (b - a, c - a, d - a, -a);
            let (abc, acd, adb) = (ab.cross(ac), ac.cross(ad), ad.cross(ab));
            if abc.dot(ao) > 0.0 {
                *simplex = vec![c, b, a];
                return next_simplex(simplex, direction);
            }
            if acd.dot(ao) > 0.0 {
                *simplex = vec![d, c, a];
                return next_simplex(simplex, direction);
            }
            if adb.dot(ao) > 0.0 {
                *simplex = vec![b, d, a];
                return next_simplex(simplex, direction);
            }
            Some([d, c, b, a])
        }
    }
}

/// Grows a degenerate simplex (touching case) into a tetrahedron around the origin.
fn complete_tetrahedron(a: &Convex, b: &Convex, simplex: &[Vec3]) -> Option<[Vec3; 4]> {
    let mut points: Vec<Vec3> = simplex.to_vec();
    for axis in [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::NEG_X,
        Vec3::NEG_Y,
        Vec3::NEG_Z,
    ] {
        if points.len() == 4 {
            break;
        }
        let p = support(a, b, axis);
        if points.iter().all(|q| q.distance_squared(p) > 1e-10) {
            points.push(p);
        }
    }
    (points.len() == 4).then(|| [points[0], points[1], points[2], points[3]])
}

/// Penetration of overlapping A and B: `(normal from B to A, depth)`.
pub(crate) fn epa(a: &Convex, b: &Convex, tetra: [Vec3; 4]) -> Option<(Vec3, f32)> {
    let mut points = tetra.to_vec();
    // Faces as index triples, wound so normals point away from the interior.
    let mut faces: Vec<[usize; 3]> = vec![[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];
    let centroid = points.iter().copied().sum::<Vec3>() / 4.0;
    for f in &mut faces {
        let n = (points[f[1]] - points[f[0]]).cross(points[f[2]] - points[f[0]]);
        if n.dot(points[f[0]] - centroid) < 0.0 {
            f.swap(1, 2);
        }
    }

    for _ in 0..MAX_ITERATIONS {
        // Closest face to the origin.
        let (closest, normal, distance) = faces
            .iter()
            .enumerate()
            .filter_map(|(i, f)| {
                let n = (points[f[1]] - points[f[0]]).cross(points[f[2]] - points[f[0]]);
                let len = n.length();
                (len > 1e-12).then(|| {
                    let n = n / len;
                    (i, n, n.dot(points[f[0]]))
                })
            })
            .min_by(|x, y| x.2.total_cmp(&y.2))?;

        let p = support(a, b, normal);
        if p.dot(normal) - distance < 1e-4 {
            // Converged: the Minkowski difference goes no further this way. The
            // origin-to-surface normal points from A into B, so negate for B→A.
            let _ = closest;
            return Some((-normal, distance));
        }

        // Remove faces that see the new point; keep their boundary (horizon) edges.
        let new_index = points.len();
        points.push(p);
        let mut horizon: Vec<[usize; 2]> = Vec::new();
        faces.retain(|f| {
            let n = (points[f[1]] - points[f[0]]).cross(points[f[2]] - points[f[0]]);
            let visible = n.dot(p - points[f[0]]) > 0.0;
            if visible {
                for edge in [[f[0], f[1]], [f[1], f[2]], [f[2], f[0]]] {
                    // An edge shared by two visible faces appears reversed: interior.
                    if let Some(i) = horizon.iter().position(|e| *e == [edge[1], edge[0]]) {
                        horizon.swap_remove(i);
                    } else {
                        horizon.push(edge);
                    }
                }
            }
            !visible
        });
        faces.extend(horizon.into_iter().map(|[i, j]| [i, j, new_index]));
    }
    None
}

/// GJK + EPA: `(normal from B to A, depth)` if the shapes overlap.
pub(crate) fn penetration(a: &Convex, b: &Convex) -> Option<(Vec3, f32)> {
    epa(a, b, gjk(a, b)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box_box::box_box;

    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 8) as f32 / (1u32 << 24) as f32
        }
        fn rotation(&mut self) -> Quat {
            Quat::from_euler(
                glam::EulerRot::XYZ,
                self.next() * 6.3,
                self.next() * 6.3,
                self.next() * 6.3,
            )
        }
    }

    fn sphere(center: Vec3, radius: f32) -> Convex {
        Convex::Sphere { center, radius }
    }

    #[test]
    fn gjk_agrees_with_sphere_distance() {
        let a = sphere(Vec3::ZERO, 1.0);
        assert!(gjk(&a, &sphere(Vec3::new(1.9, 0.0, 0.0), 1.0)).is_some());
        assert!(gjk(&a, &sphere(Vec3::new(2.1, 0.0, 0.0), 1.0)).is_none());
        assert!(gjk(&a, &sphere(Vec3::new(1.0, 1.0, 1.0), 1.0)).is_some());
    }

    #[test]
    fn epa_matches_analytic_sphere_penetration() {
        let (normal, depth) = penetration(
            &sphere(Vec3::new(1.5, 0.0, 0.0), 1.0),
            &sphere(Vec3::ZERO, 1.0),
        )
        .unwrap();
        assert!(normal.abs_diff_eq(Vec3::X, 1e-2), "{normal}");
        assert!((depth - 0.5).abs() < 1e-2, "{depth}");
    }

    /// Independent cross-check: GJK/EPA vs the SAT box–box code on random poses.
    #[test]
    fn gjk_epa_agrees_with_sat_for_boxes() {
        let mut rng = Rng(3);
        let (mut overlapping, mut checked) = (0, 0);
        for _ in 0..300 {
            let (qa, qb) = (rng.rotation(), rng.rotation());
            let (ha, hb) = (
                Vec3::new(
                    0.3 + rng.next() * 0.5,
                    0.3 + rng.next() * 0.5,
                    0.3 + rng.next() * 0.5,
                ),
                Vec3::new(
                    0.3 + rng.next() * 0.5,
                    0.3 + rng.next() * 0.5,
                    0.3 + rng.next() * 0.5,
                ),
            );
            let pa = Vec3::new(rng.next(), rng.next(), rng.next()) * 2.4 - 1.2;
            let sat = box_box(pa, qa, ha, Vec3::ZERO, qb, hb);
            let gjk_hit = penetration(
                &Convex::cuboid(pa, qa, ha),
                &Convex::cuboid(Vec3::ZERO, qb, hb),
            );
            assert_eq!(
                !sat.is_empty(),
                gjk_hit.is_some(),
                "disagree on overlap at {pa}"
            );
            if let (Some((normal, depth)), Some(&(sat_normal, _, _))) = (gjk_hit, sat.first()) {
                overlapping += 1;
                // SAT prefers face axes within a tolerance, so compare only when it
                // reports its true minimum (its deepest point is the axis overlap).
                let sat_depth = sat.iter().map(|c| c.1).fold(0.0f32, f32::max);
                if normal.dot(sat_normal) > 0.99 {
                    checked += 1;
                    assert!(
                        (depth - sat_depth).abs() < 0.02,
                        "depth {depth} vs SAT {sat_depth}"
                    );
                }
            }
        }
        assert!(
            overlapping > 50 && checked > overlapping / 2,
            "{overlapping} overlapping, {checked} compared"
        );
    }

    #[test]
    fn capsule_support_reaches_its_ends() {
        let cap = Convex::Capsule {
            a: Vec3::ZERO,
            b: Vec3::Y * 2.0,
            radius: 0.5,
        };
        assert!(cap.support(Vec3::Y).abs_diff_eq(Vec3::Y * 2.5, 1e-6));
        assert!(
            cap.support(Vec3::NEG_Y)
                .abs_diff_eq(Vec3::NEG_Y * 0.5, 1e-6)
        );
    }
}
