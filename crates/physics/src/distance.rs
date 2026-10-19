//! GJK distance: the closest points of two separated convex shapes, for shape
//! casts and overlap queries. Where `gjk` only answers "do they overlap?", this
//! tracks the point of the Minkowski difference nearest the origin, together
//! with the support points on each shape that produce it (the witnesses).
//! References: Ericson §5.1.5 (closest point on a triangle) and §9.5.

use glam::Vec3;

use crate::gjk::Convex;

/// A shape with a support mapping, in world space.
pub(crate) trait Support {
    /// Farthest point along `direction` (need not be unit length).
    fn support(&self, direction: Vec3) -> Vec3;
}

impl Support for Convex<'_> {
    fn support(&self, direction: Vec3) -> Vec3 {
        Convex::support(self, direction)
    }
}

/// A shape moved by `offset`, without rebuilding it.
pub(crate) struct Translated<'a, S>(pub &'a S, pub Vec3);

impl<S: Support> Support for Translated<'_, S> {
    fn support(&self, direction: Vec3) -> Vec3 {
        self.0.support(direction) + self.1
    }
}

/// A vertex of the simplex: `w = a − b`, with its witnesses on A and B.
#[derive(Debug, Clone, Copy)]
struct Vertex {
    w: Vec3,
    a: Vec3,
    b: Vec3,
}

fn vertex(a: &impl Support, b: &impl Support, direction: Vec3) -> Vertex {
    let (pa, pb) = (a.support(direction), b.support(-direction));
    Vertex {
        w: pa - pb,
        a: pa,
        b: pb,
    }
}

const MAX_ITERATIONS: usize = 64;
/// Converged when the next support point gets the distance² no more than this
/// fraction closer.
const RELATIVE_TOLERANCE: f32 = 1e-6;
/// Distance below which the shapes count as touching (so overlapping).
const TOUCHING: f32 = 1e-6;

/// Closest points `(on A, on B)` of two convex shapes, or `None` if they
/// overlap or touch.
pub(crate) fn closest_points(a: &impl Support, b: &impl Support) -> Option<(Vec3, Vec3)> {
    let mut simplex = vec![vertex(a, b, Vec3::X)];
    let mut weights = vec![1.0];
    let mut best = f32::INFINITY;
    for _ in 0..MAX_ITERATIONS {
        let v = combine(&simplex, &weights, |p| p.w);
        let dist2 = v.length_squared();
        if dist2 < TOUCHING * TOUCHING {
            return None;
        }
        // Stop when the distance no longer shrinks (rounding), or when the
        // support point beyond the closest point brings nothing new.
        if dist2 >= best {
            break;
        }
        best = dist2;
        let next = vertex(a, b, -v);
        if dist2 - v.dot(next.w) <= RELATIVE_TOLERANCE * dist2
            || simplex.iter().any(|p| p.w.distance_squared(next.w) < 1e-12)
        {
            break;
        }
        simplex.push(next);
        let (reduced, w) = nearest_feature(&simplex)?;
        simplex = reduced;
        weights = w;
    }
    Some((
        combine(&simplex, &weights, |p| p.a),
        combine(&simplex, &weights, |p| p.b),
    ))
}

fn combine(simplex: &[Vertex], weights: &[f32], pick: impl Fn(&Vertex) -> Vec3) -> Vec3 {
    simplex.iter().zip(weights).map(|(p, &w)| pick(p) * w).sum()
}

/// The smallest sub-simplex holding the point nearest the origin, with that
/// point's barycentric weights. `None` if a tetrahedron contains the origin.
fn nearest_feature(simplex: &[Vertex]) -> Option<(Vec<Vertex>, Vec<f32>)> {
    Some(match *simplex {
        [a] => (vec![a], vec![1.0]),
        [a, b] => segment(a, b),
        [a, b, c] => triangle(a, b, c),
        [a, b, c, d] => return tetrahedron(a, b, c, d),
        _ => unreachable!("simplices have 1–4 vertices"),
    })
}

fn segment(a: Vertex, b: Vertex) -> (Vec<Vertex>, Vec<f32>) {
    let ab = b.w - a.w;
    let len2 = ab.length_squared();
    let t = if len2 > 0.0 {
        (-a.w.dot(ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    if t <= 0.0 {
        (vec![a], vec![1.0])
    } else if t >= 1.0 {
        (vec![b], vec![1.0])
    } else {
        (vec![a, b], vec![1.0 - t, t])
    }
}

/// Ericson's `ClosestPtPointTriangle` with the query point at the origin.
fn triangle(a: Vertex, b: Vertex, c: Vertex) -> (Vec<Vertex>, Vec<f32>) {
    let (ab, ac) = (b.w - a.w, c.w - a.w);
    let (d1, d2) = (ab.dot(-a.w), ac.dot(-a.w));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (vec![a], vec![1.0]);
    }
    let (d3, d4) = (ab.dot(-b.w), ac.dot(-b.w));
    if d3 >= 0.0 && d4 <= d3 {
        return (vec![b], vec![1.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        return (vec![a, b], vec![1.0 - t, t]);
    }
    let (d5, d6) = (ab.dot(-c.w), ac.dot(-c.w));
    if d6 >= 0.0 && d5 <= d6 {
        return (vec![c], vec![1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        return (vec![a, c], vec![1.0 - t, t]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (vec![b, c], vec![1.0 - t, t]);
    }
    let sum = va + vb + vc;
    if !(sum.is_finite() && sum > 0.0) {
        // Degenerate (collinear) triangle: the best of its edges.
        return [segment(a, b), segment(a, c), segment(b, c)]
            .into_iter()
            .min_by(|x, y| distance2(x).total_cmp(&distance2(y)))
            .expect("three candidates");
    }
    let (v, w) = (vb / sum, vc / sum);
    (vec![a, b, c], vec![1.0 - v - w, v, w])
}

/// The nearest point on the tetrahedron's surface, or `None` when the origin is
/// inside it (the shapes overlap). Outside, the nearest point of the solid is on
/// some face, so every face is checked: deciding which faces the origin is
/// outside of from signs alone breaks down for flat tetrahedra, which are
/// common (the Minkowski difference of two segments is a parallelogram).
fn tetrahedron(a: Vertex, b: Vertex, c: Vertex, d: Vertex) -> Option<(Vec<Vertex>, Vec<f32>)> {
    // Each face with the vertex opposite it.
    let faces = [(a, b, c, d), (a, c, d, b), (a, d, b, c), (b, d, c, a)];
    let size = [b.w, c.w, d.w]
        .iter()
        .map(|w| w.distance(a.w))
        .fold(0.0, f32::max);
    let volume = (b.w - a.w).cross(c.w - a.w).dot(d.w - a.w).abs();
    let flat = volume <= 1e-6 * size * size * size;
    let inside = !flat
        && faces.iter().all(|&(p, q, r, opposite)| {
            let normal = (q.w - p.w).cross(r.w - p.w);
            normal.dot(-p.w) * normal.dot(opposite.w - p.w) >= 0.0
        });
    if inside {
        return None;
    }
    faces
        .into_iter()
        .map(|(p, q, r, _)| triangle(p, q, r))
        .min_by(|x, y| distance2(x).total_cmp(&distance2(y)))
}

fn distance2((simplex, weights): &(Vec<Vertex>, Vec<f32>)) -> f32 {
    combine(simplex, weights, |p| p.w).length_squared()
}

#[cfg(test)]
mod tests {
    use glam::{Mat3, Quat};

    use super::*;

    /// Deterministic pseudo-random numbers in [-1, 1).
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }

        fn vec3(&mut self) -> Vec3 {
            Vec3::new(self.next(), self.next(), self.next())
        }
    }

    fn point(p: Vec3) -> Convex<'static> {
        Convex::Sphere {
            center: p,
            radius: 0.0,
        }
    }

    #[test]
    fn spheres() {
        let a = Convex::Sphere {
            center: Vec3::new(5.0, 0.0, 0.0),
            radius: 1.0,
        };
        let b = Convex::Sphere {
            center: Vec3::ZERO,
            radius: 2.0,
        };
        let (pa, pb) = closest_points(&a, &b).unwrap();
        assert!(pa.abs_diff_eq(Vec3::new(4.0, 0.0, 0.0), 1e-4), "{pa}");
        assert!(pb.abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 1e-4), "{pb}");
    }

    #[test]
    fn overlapping_and_touching_shapes_have_no_closest_points() {
        let unit = Convex::cuboid(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
        let inside = Convex::cuboid(Vec3::new(1.5, 0.3, 0.0), Quat::IDENTITY, Vec3::ONE);
        let touching = Convex::cuboid(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE);
        assert!(closest_points(&unit, &inside).is_none());
        assert!(closest_points(&unit, &touching).is_none());
        assert!(closest_points(&point(Vec3::new(0.2, 0.1, -0.4)), &unit).is_none());
    }

    /// A point against an oriented box, checked against clamping in the box's frame.
    #[test]
    fn point_to_box_matches_clamping() {
        let mut rng = Lcg(7);
        for _ in 0..500 {
            let half = rng.vec3().abs() + Vec3::splat(0.1);
            let orientation = Quat::from_scaled_axis(rng.vec3() * 3.0);
            let center = rng.vec3() * 2.0;
            let p = rng.vec3() * 6.0;
            let local = Mat3::from_quat(orientation).transpose() * (p - center);
            if local.abs().cmplt(half).all() {
                continue; // inside
            }
            let expected = center + orientation * local.clamp(-half, half);
            let cuboid = Convex::cuboid(center, orientation, half);
            let (on_point, on_box) = closest_points(&point(p), &cuboid).expect("separated");
            assert!(on_point.abs_diff_eq(p, 1e-4));
            assert!(
                on_box.distance(expected) < 2e-3,
                "box {on_box} expected {expected}"
            );
        }
    }

    /// Segment against segment (capsule cores), checked against dense sampling.
    #[test]
    fn segments_match_sampling() {
        let mut rng = Lcg(11);
        for _ in 0..200 {
            let (a0, a1, b0, b1) = (
                rng.vec3() * 3.0,
                rng.vec3() * 3.0,
                rng.vec3() * 3.0,
                rng.vec3() * 3.0,
            );
            let seg = |a, b| Convex::Capsule { a, b, radius: 0.0 };
            let sampled = (0..=200)
                .flat_map(|i| (0..=200).map(move |j| (i as f32 / 200.0, j as f32 / 200.0)))
                .map(|(s, t)| a0.lerp(a1, s).distance(b0.lerp(b1, t)))
                .fold(f32::INFINITY, f32::min);
            match closest_points(&seg(a0, a1), &seg(b0, b1)) {
                Some((pa, pb)) => {
                    let d = pa.distance(pb);
                    assert!(d <= sampled + 1e-4, "gjk {d} > sampled {sampled}");
                    assert!(d >= sampled - 0.03, "gjk {d} far below sampled {sampled}");
                }
                None => assert!(sampled < 0.03, "segments {sampled} apart reported touching"),
            }
        }
    }

    #[test]
    fn translation_moves_the_shape() {
        let unit = Convex::cuboid(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
        let p = point(Vec3::new(10.0, 0.0, 0.0));
        let (_, on_box) = closest_points(&p, &Translated(&unit, Vec3::new(5.0, 0.0, 0.0))).unwrap();
        assert!(on_box.abs_diff_eq(Vec3::new(6.0, 0.0, 0.0), 1e-4));
    }
}
