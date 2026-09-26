//! Convex collision shapes derived from a model's vertices (model space; the
//! caller applies scale). Models collide as convex shapes: there are no
//! triangle-mesh colliders.

use glam::Vec3;

use crate::ModelData;

/// Extreme vertices in `count` directions spread evenly over the sphere
/// (Fibonacci lattice), duplicates removed. Their convex hull approximates the
/// model's; every point is a real vertex, so the hull never exceeds the model.
pub fn extreme_points(model: &ModelData, count: usize) -> Vec<Vec3> {
    let vertices: Vec<Vec3> = model.positions().collect();
    let mut points: Vec<Vec3> = Vec::new();
    for dir in fibonacci_directions(count) {
        let Some(best) = vertices
            .iter()
            .copied()
            .max_by(|a, b| a.dot(dir).total_cmp(&b.dot(dir)))
        else {
            break;
        };
        if !points.contains(&best) {
            points.push(best);
        }
    }
    points
}

/// `count` unit vectors spread evenly over the sphere.
pub fn fibonacci_directions(count: usize) -> Vec<Vec3> {
    let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    (0..count)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / count as f32;
            let ring = (1.0 - y * y).sqrt();
            let phi = golden * i as f32;
            Vec3::new(ring * phi.cos(), y, ring * phi.sin())
        })
        .collect()
}

/// The bounds' center and half extents.
pub fn bounding_box(model: &ModelData) -> (Vec3, Vec3) {
    ((model.min + model.max) / 2.0, (model.max - model.min) / 2.0)
}

/// A sphere around the bounds' center enclosing every vertex.
pub fn bounding_sphere(model: &ModelData) -> (Vec3, f32) {
    let center = (model.min + model.max) / 2.0;
    let radius = model
        .positions()
        .map(|p| p.distance(center))
        .fold(0.0, f32::max);
    (center, radius)
}

/// A Y-aligned capsule around the bounds' center enclosing every vertex:
/// `(center, half_height, radius)`. The radius is the widest vertex from the
/// axis; the half height is the least that still covers every vertex.
pub fn bounding_capsule(model: &ModelData) -> (Vec3, f32, f32) {
    let center = (model.min + model.max) / 2.0;
    let offsets: Vec<(f32, f32)> = model
        .positions()
        .map(|p| {
            let d = p - center;
            (d.y.abs(), (d.x * d.x + d.z * d.z).sqrt())
        })
        .collect();
    let radius = offsets.iter().map(|&(_, r)| r).fold(0.0, f32::max);
    // A point `r` from the axis is covered up to `half_height + √(radius² − r²)`.
    let half_height = offsets
        .iter()
        .map(|&(along, r)| along - (radius * radius - r * r).max(0.0).sqrt())
        .fold(0.0, f32::max);
    (center, half_height, radius)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ModelMesh, ModelVertex};

    fn model(points: &[Vec3]) -> ModelData {
        let vertices = points
            .iter()
            .map(|p| ModelVertex {
                position: p.to_array(),
                normal: [0.0, 1.0, 0.0],
                uv: [0.0; 2],
                tangent: [1.0, 0.0, 0.0, 1.0],
            })
            .collect();
        let (min, max) = points.iter().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(lo, hi), p| (lo.min(*p), hi.max(*p)),
        );
        ModelData {
            meshes: vec![ModelMesh {
                vertices,
                indices: vec![0, 1, 2],
                material: None,
            }],
            materials: Vec::new(),
            textures: Vec::new(),
            min,
            max,
        }
    }

    /// A 2×4×6 box's corners, plus interior points and points on its faces.
    fn crate_points() -> Vec<Vec3> {
        let mut points: Vec<Vec3> = (0..8)
            .map(|i| {
                let s = |bit: usize, h: f32| if i & bit == 0 { -h } else { h };
                Vec3::new(s(1, 1.0) + 5.0, s(2, 2.0), s(4, 3.0))
            })
            .collect();
        points.extend([
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(5.5, 1.0, -2.0),
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(5.0, 2.0, 1.0),
        ]);
        points
    }

    #[test]
    fn extreme_points_are_exactly_the_corners() {
        let points = crate_points();
        let found = extreme_points(&model(&points), 64);
        assert_eq!(found.len(), 8, "{found:?}");
        assert!(found.iter().all(|p| points[..8].contains(p)));
    }

    #[test]
    fn extreme_points_support_matches_the_model_in_every_sampled_direction() {
        // An irregular blob: the sampled set must reach as far as the model.
        let points: Vec<Vec3> = fibonacci_directions(300)
            .into_iter()
            .enumerate()
            .map(|(i, d)| d * (1.0 + 0.3 * ((i * 7919 % 13) as f32 / 13.0)))
            .collect();
        let m = model(&points);
        let found = extreme_points(&m, 64);
        assert!(found.len() <= 64);
        for dir in fibonacci_directions(64) {
            let support = |set: &[Vec3]| set.iter().map(|p| p.dot(dir)).fold(f32::MIN, f32::max);
            assert_eq!(support(&found), support(&points));
        }
    }

    #[test]
    fn fitted_shapes_enclose_every_vertex() {
        let points = crate_points();
        let m = model(&points);
        let (center, half) = bounding_box(&m);
        assert_eq!(
            (center, half),
            (Vec3::new(5.0, 0.0, 0.0), Vec3::new(1.0, 2.0, 3.0))
        );

        let (center, radius) = bounding_sphere(&m);
        assert!(points.iter().all(|p| p.distance(center) <= radius + 1e-5));
        assert!(
            (radius - Vec3::new(1.0, 2.0, 3.0).length()).abs() < 1e-5,
            "tight"
        );

        let (center, half_height, radius) = bounding_capsule(&m);
        let inside = |p: Vec3| {
            let d = p - center;
            let along = d.y.clamp(-half_height, half_height);
            (d - Vec3::Y * along).length() <= radius + 1e-5
        };
        assert!(points.iter().all(|&p| inside(p)));
        assert!((radius - Vec3::new(1.0, 0.0, 3.0).length()).abs() < 1e-5);
        assert!(half_height <= 2.0, "no taller than the box ({half_height})");
    }
}
