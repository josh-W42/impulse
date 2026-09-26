//! Contacts involving convex hulls (and boxes paired with hulls): EPA supplies the
//! normal, face clipping (as in M3's box–box) builds a multi-point manifold.

use std::sync::OnceLock;

use glam::{Quat, Vec3};

use crate::box_box::{clip_polygon, reduce_to_four};
use crate::capsule::{Capsule, keep_two};
use crate::contact::Geometry;
use crate::gjk::{Convex, penetration};
use crate::hull::ConvexHull;

/// A convex polytope posed in world space: vertices and outward face planes
/// (`normal · x = offset`) with counter-clockwise vertex indices.
pub(crate) struct Polytope {
    pub vertices: Vec<Vec3>,
    pub faces: Vec<(Vec3, f32, Vec<usize>)>,
}

impl Polytope {
    pub(crate) fn hull(hull: &ConvexHull, position: Vec3, orientation: Quat) -> Self {
        Self {
            vertices: hull
                .vertices()
                .iter()
                .map(|v| position + orientation * *v)
                .collect(),
            faces: hull
                .faces()
                .iter()
                .map(|f| {
                    let normal = orientation * f.normal;
                    (normal, f.offset + normal.dot(position), f.vertices.clone())
                })
                .collect(),
        }
    }

    /// A box as a polytope: the unit cube's hull (built once) scaled by `half`.
    pub(crate) fn cuboid(center: Vec3, orientation: Quat, half: Vec3) -> Self {
        static UNIT: OnceLock<ConvexHull> = OnceLock::new();
        let unit = UNIT.get_or_init(|| {
            let corners: Vec<Vec3> = (0..8)
                .map(|i| {
                    let s = |bit: usize| if i & bit == 0 { -1.0 } else { 1.0 };
                    Vec3::new(s(1), s(2), s(4))
                })
                .collect();
            ConvexHull::new(&corners).expect("the unit cube is a valid hull")
        });
        let vertices: Vec<Vec3> = unit
            .vertices()
            .iter()
            .map(|v| center + orientation * (*v * half))
            .collect();
        let faces = unit
            .faces()
            .iter()
            .map(|f| {
                // Axis-aligned faces keep their normals under scaling.
                let normal = orientation * f.normal;
                (
                    normal,
                    normal.dot(vertices[f.vertices[0]]),
                    f.vertices.clone(),
                )
            })
            .collect();
        Self { vertices, faces }
    }

    fn convex(&self) -> Convex<'_> {
        Convex::Points {
            points: &self.vertices,
        }
    }

    /// The face whose normal points most along `direction`.
    fn best_face(&self, direction: Vec3) -> (usize, f32) {
        self.faces
            .iter()
            .enumerate()
            .map(|(i, f)| (i, f.0.dot(direction)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("polytopes have faces")
    }
}

/// Polytope (A) on a plane (B): each vertex below the plane, reduced to four.
pub(crate) fn polytope_plane(poly: &Polytope, plane_point: Vec3, normal: Vec3) -> Vec<Geometry> {
    let below: Vec<(Vec3, f32)> = poly
        .vertices
        .iter()
        .filter_map(|v| {
            let height = (*v - plane_point).dot(normal);
            (height < 0.0).then(|| (*v - normal * (height * 0.5), -height))
        })
        .collect();
    reduce_to_four(below, normal)
        .into_iter()
        .map(|(point, depth)| (normal, depth, point))
        .collect()
}

/// Polytope A against polytope B (normal B → A).
pub(crate) fn polytope_polytope(a: &Polytope, b: &Polytope) -> Vec<Geometry> {
    let Some((normal, depth)) = penetration(&a.convex(), &b.convex()) else {
        return Vec::new();
    };
    // Reference face: whichever is best aligned with the separation direction —
    // A's face toward B (along −normal) or B's face toward A (along +normal).
    let (face_a, align_a) = a.best_face(-normal);
    let (face_b, align_b) = b.best_face(normal);
    let (reference, ref_index, incident, reference_is_a) = if align_b >= align_a {
        (b, face_b, a, false)
    } else {
        (a, face_a, b, true)
    };
    let (ref_normal, ref_offset, ref_vertices) = &reference.faces[ref_index];
    let (incident_index, _) = incident.best_face(-*ref_normal);
    let mut polygon: Vec<Vec3> = incident.faces[incident_index]
        .2
        .iter()
        .map(|&i| incident.vertices[i])
        .collect();
    for k in 0..ref_vertices.len() {
        let p0 = reference.vertices[ref_vertices[k]];
        let p1 = reference.vertices[ref_vertices[(k + 1) % ref_vertices.len()]];
        // Outward side plane of a counter-clockwise edge: edge × face normal.
        let side = (p1 - p0).cross(*ref_normal).normalize_or_zero();
        polygon = clip_polygon(&polygon, side, side.dot(p0));
        if polygon.is_empty() {
            break;
        }
    }
    let points: Vec<(Vec3, f32)> = polygon
        .into_iter()
        .filter_map(|p| {
            let separation = ref_normal.dot(p) - ref_offset;
            (separation < 0.0).then(|| (p - *ref_normal * (separation * 0.5), -separation))
        })
        .collect();
    // Reference on B: its normal already points B → A. On A: flip it.
    let contact_normal = if reference_is_a {
        -*ref_normal
    } else {
        *ref_normal
    };
    if points.is_empty() {
        // Edge-to-edge (no face overlap after clipping): one point between the
        // deepest features.
        let point = (a.convex().support(-normal) + b.convex().support(normal)) * 0.5;
        return vec![(normal, depth, point)];
    }
    reduce_to_four(points, contact_normal)
        .into_iter()
        .map(|(point, d)| (contact_normal, d, point))
        .collect()
}

/// Sphere (A) against a polytope (B); the normal points from the polytope to the sphere.
pub(crate) fn sphere_polytope(center: Vec3, radius: f32, poly: &Polytope) -> Option<Geometry> {
    let (normal, depth) = penetration(&Convex::Sphere { center, radius }, &poly.convex())?;
    Some((normal, depth, center - normal * (radius - depth * 0.5)))
}

/// Capsule (A) against a polytope (B): end spheres first (two points when lying
/// on a face), else one GJK/EPA point.
pub(crate) fn capsule_polytope(capsule: Capsule, poly: &Polytope) -> Vec<Geometry> {
    let ends: Vec<Geometry> = capsule
        .ends()
        .into_iter()
        .filter_map(|end| sphere_polytope(end, capsule.radius, poly))
        .collect();
    if !ends.is_empty() {
        return keep_two(ends);
    }
    let shape = capsule.convex();
    let Some((normal, depth)) = penetration(&shape, &poly.convex()) else {
        return Vec::new();
    };
    vec![(
        normal,
        depth,
        shape.support(-normal) + normal * (depth * 0.5),
    )]
}
