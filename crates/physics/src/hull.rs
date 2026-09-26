//! Convex hulls: construction from a point cloud, polygon faces, and mass properties.
//!
//! Construction is deliberately brute force (O(n⁴), inputs capped at 64 points):
//! every point triple spanning a supporting plane yields a face plane; coplanar
//! planes merge, and each face's polygon is the 2D convex hull of its points
//! (dropping points on edges or inside the face). Hulls are built once at load
//! time, so robustness matters more than speed.

use glam::{Mat3, Vec2, Vec3};

pub const MAX_HULL_POINTS: usize = 64;

/// A planar face: `normal · p = offset`, vertices counter-clockwise about the normal.
#[derive(Debug, Clone, PartialEq)]
pub struct HullFace {
    pub normal: Vec3,
    pub offset: f32,
    pub vertices: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HullError {
    TooFewPoints,
    TooManyPoints,
    NonFinite,
    /// Coplanar or collinear points: no volume.
    Degenerate,
}

impl std::fmt::Display for HullError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewPoints => write!(f, "a hull needs at least 4 points"),
            Self::TooManyPoints => write!(f, "a hull takes at most {MAX_HULL_POINTS} points"),
            Self::NonFinite => write!(f, "hull points must be finite"),
            Self::Degenerate => write!(f, "hull points must not all lie in one plane"),
        }
    }
}

impl std::error::Error for HullError {}

/// A convex polyhedron centered on its center of mass.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvexHull {
    vertices: Vec<Vec3>,
    faces: Vec<HullFace>,
    volume: f32,
    /// Inertia tensor about the center of mass for mass 1.
    unit_inertia: Mat3,
    /// Where the center of mass was in the input points' frame.
    center: Vec3,
}

impl ConvexHull {
    /// Builds the hull of `points`, re-centered so its center of mass is the origin.
    pub fn new(points: &[Vec3]) -> Result<Self, HullError> {
        if points.len() < 4 {
            return Err(HullError::TooFewPoints);
        }
        if points.len() > MAX_HULL_POINTS {
            return Err(HullError::TooManyPoints);
        }
        if !points.iter().all(|p| p.is_finite()) {
            return Err(HullError::NonFinite);
        }
        let extent = points
            .iter()
            .fold(Vec3::ZERO, |m, p| m.max((*p - points[0]).abs()))
            .max_element();
        let eps = extent * 1e-5;
        if extent <= 0.0 {
            return Err(HullError::Degenerate);
        }

        let planes = supporting_planes(points, eps);
        let mut faces = Vec::new();
        for (normal, offset) in planes {
            let on_plane: Vec<usize> = (0..points.len())
                .filter(|&i| (normal.dot(points[i]) - offset).abs() <= eps)
                .collect();
            let polygon = convex_polygon(points, &on_plane, normal, eps);
            if polygon.len() >= 3 {
                faces.push(HullFace {
                    normal,
                    offset,
                    vertices: polygon,
                });
            }
        }
        if faces.len() < 4 {
            return Err(HullError::Degenerate);
        }

        // Keep only points that are corners of some face; remap indices.
        let mut remap = vec![usize::MAX; points.len()];
        let mut vertices = Vec::new();
        for face in &mut faces {
            for v in &mut face.vertices {
                if remap[*v] == usize::MAX {
                    remap[*v] = vertices.len();
                    vertices.push(points[*v]);
                }
                *v = remap[*v];
            }
        }

        let (volume, center, covariance) = mass_properties(&vertices, &faces);
        if volume <= eps * eps * eps {
            return Err(HullError::Degenerate);
        }
        for v in &mut vertices {
            *v -= center;
        }
        for face in &mut faces {
            face.offset -= face.normal.dot(center);
        }
        // Covariance about the center of mass, then inertia for unit mass.
        let about_center = covariance - outer(center, center) * volume;
        let trace = about_center.x_axis.x + about_center.y_axis.y + about_center.z_axis.z;
        let unit_inertia =
            (Mat3::from_diagonal(Vec3::splat(trace)) - about_center) * (1.0 / volume);
        Ok(Self {
            vertices,
            faces,
            volume,
            unit_inertia,
            center,
        })
    }

    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    pub fn faces(&self) -> &[HullFace] {
        &self.faces
    }

    /// The center of mass in the input points' frame: place the body here for the
    /// hull to occupy the input points' positions.
    pub fn center(&self) -> Vec3 {
        self.center
    }

    pub fn volume(&self) -> f32 {
        self.volume
    }

    /// Inertia tensor about the center of mass for a hull of mass 1.
    pub fn unit_inertia(&self) -> Mat3 {
        self.unit_inertia
    }
}

fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// Distinct planes through three points with every point on the inner side.
fn supporting_planes(points: &[Vec3], eps: f32) -> Vec<(Vec3, f32)> {
    let mut planes: Vec<(Vec3, f32)> = Vec::new();
    let n = points.len();
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let cross = (points[j] - points[i]).cross(points[k] - points[i]);
                let length = cross.length();
                if length <= eps * eps.max(1e-12).sqrt() {
                    continue; // collinear triple
                }
                let mut normal = cross / length;
                let mut offset = normal.dot(points[i]);
                let (mut above, mut below) = (false, false);
                for p in points {
                    let s = normal.dot(*p) - offset;
                    above |= s > eps;
                    below |= s < -eps;
                }
                if above && below {
                    continue; // points on both sides: not a supporting plane
                }
                if above {
                    normal = -normal;
                    offset = -offset;
                }
                let known = planes
                    .iter()
                    .any(|(m, d)| m.dot(normal) > 1.0 - 1e-6 && (d - offset).abs() <= eps);
                if !known {
                    planes.push((normal, offset));
                }
            }
        }
    }
    planes
}

/// The convex polygon (counter-clockwise about `normal`) of coplanar points,
/// via Andrew's monotone chain in the plane's 2D coordinates.
fn convex_polygon(points: &[Vec3], indices: &[usize], normal: Vec3, eps: f32) -> Vec<usize> {
    let u = normal.any_orthonormal_vector();
    let v = normal.cross(u);
    let mut pts: Vec<(Vec2, usize)> = indices
        .iter()
        .map(|&i| (Vec2::new(points[i].dot(u), points[i].dot(v)), i))
        .collect();
    pts.sort_by(|a, b| a.0.x.total_cmp(&b.0.x).then(a.0.y.total_cmp(&b.0.y)));
    let cross = |o: Vec2, a: Vec2, b: Vec2| (a - o).perp_dot(b - o);
    let area_eps = eps * eps.max(1e-12).sqrt();
    let mut hull: Vec<(Vec2, usize)> = Vec::new();
    for pass in [false, true] {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &(Vec2, usize)>> = if pass {
            Box::new(pts.iter().rev())
        } else {
            Box::new(pts.iter())
        };
        for &p in iter {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2].0, hull[hull.len() - 1].0, p.0) <= area_eps
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop(); // last point repeats as the next chain's first
    }
    // u × v = normal, so counter-clockwise in (u, v) is counter-clockwise about normal.
    hull.into_iter().map(|(_, i)| i).collect()
}

/// Volume, center of mass, and covariance ∫ x xᵀ dV (about the origin), summing
/// signed tetrahedra from the origin to each face-fan triangle.
fn mass_properties(vertices: &[Vec3], faces: &[HullFace]) -> (f32, Vec3, Mat3) {
    const K: Mat3 = Mat3::from_cols_array(&[2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]);
    let (mut volume, mut moment, mut covariance) = (0.0, Vec3::ZERO, Mat3::ZERO);
    for face in faces {
        let a = vertices[face.vertices[0]];
        for w in face.vertices[1..].windows(2) {
            let (b, c) = (vertices[w[0]], vertices[w[1]]);
            let det = a.dot(b.cross(c));
            volume += det / 6.0;
            moment += (a + b + c) * (det / 24.0);
            let m = Mat3::from_cols(a, b, c);
            covariance += m * K * m.transpose() * (det / 120.0);
        }
    }
    (volume, moment / volume, covariance)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_points(center: Vec3, half: f32) -> Vec<Vec3> {
        (0..8)
            .map(|i| {
                let s = |bit: usize| if i & bit == 0 { -half } else { half };
                center + Vec3::new(s(1), s(2), s(4))
            })
            .collect()
    }

    fn euler_characteristic(hull: &ConvexHull) -> i64 {
        let edges: usize = hull.faces().iter().map(|f| f.vertices.len()).sum::<usize>() / 2;
        hull.vertices().len() as i64 - edges as i64 + hull.faces().len() as i64
    }

    #[test]
    fn cube_hull_matches_box_formulas_and_is_recentered() {
        let hull = ConvexHull::new(&cube_points(Vec3::new(3.0, 4.0, 5.0), 0.5)).unwrap();
        assert_eq!(hull.faces().len(), 6);
        assert!(hull.faces().iter().all(|f| f.vertices.len() == 4));
        assert!((hull.volume() - 1.0).abs() < 1e-5);
        // Unit cube, mass 1: I = m/12·(1 + 1) = 1/6 on the diagonal.
        assert!(
            hull.unit_inertia()
                .abs_diff_eq(Mat3::from_diagonal(Vec3::splat(1.0 / 6.0)), 1e-5)
        );
        let centroid = hull.vertices().iter().copied().sum::<Vec3>() / 8.0;
        assert!(centroid.abs_diff_eq(Vec3::ZERO, 1e-5));
    }

    #[test]
    fn edge_and_face_interior_points_are_not_corners() {
        let mut points = cube_points(Vec3::ZERO, 1.0);
        points.extend([
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        ]);
        let hull = ConvexHull::new(&points).unwrap();
        assert_eq!(hull.vertices().len(), 8);
        assert!(hull.faces().iter().all(|f| f.vertices.len() == 4));
    }

    #[test]
    fn random_hulls_are_closed_convex_and_contain_their_points() {
        let mut seed = 11u32;
        let mut rand = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        for _ in 0..10 {
            let points: Vec<Vec3> = (0..40).map(|_| Vec3::new(rand(), rand(), rand())).collect();
            let hull = ConvexHull::new(&points).unwrap();
            assert_eq!(euler_characteristic(&hull), 2, "not a closed polyhedron");
            for face in hull.faces() {
                // Counter-clockwise winding agrees with the stored outward normal.
                let [a, b, c] = [0, 1, 2].map(|k| hull.vertices()[face.vertices[k]]);
                assert!(
                    (b - a).cross(c - a).dot(face.normal) > 0.0,
                    "clockwise face"
                );
                // Every input point lies inside every face (in the re-centered frame).
                for p in &points {
                    let local = *p - hull.center();
                    assert!(
                        face.normal.dot(local) <= face.offset + 1e-4,
                        "point outside the hull"
                    );
                }
            }
        }
    }

    #[test]
    fn tetrahedron_has_four_triangles_and_the_right_volume() {
        let points = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let hull = ConvexHull::new(&points).unwrap();
        assert_eq!(hull.faces().len(), 4);
        assert!((hull.volume() - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(
            ConvexHull::new(&[Vec3::ZERO; 3]),
            Err(HullError::TooFewPoints)
        );
        assert_eq!(
            ConvexHull::new(&[Vec3::ONE; 65]),
            Err(HullError::TooManyPoints)
        );
        let mut nan = cube_points(Vec3::ZERO, 1.0);
        nan[0].x = f32::NAN;
        assert_eq!(ConvexHull::new(&nan), Err(HullError::NonFinite));
        let flat: Vec<Vec3> = (0..10)
            .map(|i| Vec3::new(i as f32, (i * i) as f32 * 0.1, 0.0))
            .collect();
        assert_eq!(ConvexHull::new(&flat), Err(HullError::Degenerate));
    }
}
