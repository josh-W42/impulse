//! Unit primitive meshes. Instances scale them: cube and plane are size 1,
//! the sphere has radius 0.5. Triangles are counter-clockwise seen from outside.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// Texture coordinates (zero on built-in meshes, which are untextured).
    pub uv: [f32; 2],
    /// Tangent (xyz, along +U) and bitangent sign (w), for normal maps. Must
    /// be a unit vector perpendicular to `normal`.
    pub tangent: [f32; 4],
}

/// A unit tangent perpendicular to `normal` (for meshes without UV-derived
/// tangents; any perpendicular works with the flat default normal map).
pub fn perpendicular_tangent(normal: [f32; 3]) -> [f32; 4] {
    let n = Vec3::from_array(normal).normalize_or(Vec3::Y);
    let other = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let t = (other - n * n.dot(other)).normalize();
    [t.x, t.y, t.z, 1.0]
}

/// A vertex of a skinned mesh: a [`Vertex`] plus up to four joints (indices
/// into the instance's joint matrices) and their weights (summing to 1).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
    pub joints: [u16; 4],
    pub weights: [f32; 4],
}

#[derive(Debug, Clone, PartialEq)]
pub struct SkinnedMeshData {
    pub vertices: Vec<SkinnedVertex>,
    pub indices: Vec<u32>,
}

/// Identifies a skinned mesh added with `Renderer::add_skinned_mesh`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SkinnedMeshId(pub u32);

/// Identifies a mesh the renderer can draw: a built-in [`MeshKind`] or one
/// added at runtime with `Renderer::add_mesh`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshId(pub u32);

impl From<MeshKind> for MeshId {
    fn from(kind: MeshKind) -> Self {
        MeshId(kind.index() as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshKind {
    Cube,
    Sphere,
    Plane,
    /// Open tube (radius 0.5, height 1 along Y); capsules cap it with spheres.
    Cylinder,
}

impl MeshKind {
    pub const ALL: [MeshKind; 4] = [
        MeshKind::Cube,
        MeshKind::Sphere,
        MeshKind::Plane,
        MeshKind::Cylinder,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn data(self) -> MeshData {
        match self {
            MeshKind::Cube => cube(),
            MeshKind::Sphere => uv_sphere(32, 16),
            MeshKind::Plane => plane(),
            MeshKind::Cylinder => cylinder(32),
        }
    }

    /// Coarser levels of the round meshes, with measured errors (the flat
    /// ones have none).
    pub fn lods(self) -> Vec<MeshLod> {
        let coarse = match self {
            MeshKind::Sphere => vec![uv_sphere(16, 8), uv_sphere(8, 4)],
            MeshKind::Cylinder => vec![cylinder(16), cylinder(8)],
            MeshKind::Cube | MeshKind::Plane => Vec::new(),
        };
        let full = self.data();
        coarse
            .into_iter()
            .map(|data| MeshLod {
                error: lod_error(&full, &data),
                data,
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

/// Most levels of detail a mesh has, its full detail included.
pub const MAX_LODS: usize = 4;

/// A coarser version of a mesh, drawn where its error is under a pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshLod {
    pub data: MeshData,
    /// How far the full mesh strays from this one, in mesh units
    /// ([`lod_error`]).
    pub error: f32,
}

/// Full-detail vertices tested by [`lod_error`].
const ERROR_SAMPLES: usize = 1024;

/// How far `full`'s surface lies from `coarse`'s: the farthest of its
/// vertices (up to [`ERROR_SAMPLES`] of them, evenly strided) from the
/// nearest `coarse` triangle. Infinite when `coarse` has no triangles.
pub fn lod_error(full: &MeshData, coarse: &MeshData) -> f32 {
    let triangles: Vec<[Vec3; 3]> = coarse
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(|t| {
            let corner = |i: u32| {
                coarse
                    .vertices
                    .get(i as usize)
                    .map(|v| Vec3::from(v.position))
            };
            Some([corner(t[0])?, corner(t[1])?, corner(t[2])?])
        })
        .collect();
    if triangles.is_empty() {
        return f32::INFINITY;
    }
    let step = full.vertices.len().div_ceil(ERROR_SAMPLES).max(1);
    full.vertices
        .iter()
        .step_by(step)
        .map(|v| {
            let p = Vec3::from(v.position);
            triangles
                .iter()
                .map(|t| closest_on_triangle(p, t).distance_squared(p))
                .fold(f32::INFINITY, f32::min)
        })
        .fold(0.0, f32::max)
        .sqrt()
}

/// The point of triangle `t` nearest `p` (Ericson, Real-Time Collision
/// Detection, 5.1.5).
fn closest_on_triangle(p: Vec3, [a, b, c]: &[Vec3; 3]) -> Vec3 {
    let (ab, ac, ap) = (*b - *a, *c - *a, p - *a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return *a;
    }
    let bp = p - *b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return *b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return *a + ab * (d1 / (d1 - d3));
    }
    let cp = p - *c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return *c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return *a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return *b + (*c - *b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = va + vb + vc;
    if denom.abs() < f32::EPSILON {
        return *a; // degenerate: any corner
    }
    *a + ab * (vb / denom) + ac * (vc / denom)
}

fn vertex(position: Vec3, normal: Vec3) -> Vertex {
    Vertex {
        position: position.to_array(),
        normal: normal.to_array(),
        uv: [0.0; 2],
        tangent: perpendicular_tangent(normal.to_array()),
    }
}

/// Adds one quad; corners must be counter-clockwise seen from the `normal` side.
fn push_quad(mesh: &mut MeshData, corners: [Vec3; 4], normal: Vec3) {
    let base = mesh.vertices.len() as u32;
    mesh.vertices.extend(corners.map(|c| vertex(c, normal)));
    mesh.indices
        .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
}

pub fn cube() -> MeshData {
    let mut mesh = MeshData {
        vertices: Vec::with_capacity(24),
        indices: Vec::with_capacity(36),
    };
    for normal in [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ] {
        // Two axes spanning the face, chosen so u × v = normal (CCW from outside).
        let u = if normal.y.abs() > 0.5 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let v = normal.cross(u);
        let (u, v) = (v.cross(normal) * 0.5, v * 0.5);
        let c = normal * 0.5;
        push_quad(
            &mut mesh,
            [c - u - v, c + u - v, c + u + v, c - u + v],
            normal,
        );
    }
    mesh
}

/// Double-sided so the floor stays visible when the camera goes below it
/// (useful for inspecting bodies that sink into it).
pub fn plane() -> MeshData {
    let mut mesh = MeshData {
        vertices: Vec::with_capacity(8),
        indices: Vec::with_capacity(12),
    };
    let (x, z) = (Vec3::X * 0.5, Vec3::Z * 0.5);
    // Seen from +Y, going -x+z → +x+z → +x-z → -x-z is counter-clockwise.
    push_quad(&mut mesh, [-x + z, x + z, x - z, -x - z], Vec3::Y);
    // Same corners reversed: counter-clockwise seen from below.
    push_quad(&mut mesh, [-x - z, x - z, x + z, -x + z], Vec3::NEG_Y);
    mesh
}

/// Side wall only: the ends are hidden inside a capsule's end spheres.
pub fn cylinder(segments: u32) -> MeshData {
    assert!(segments >= 3);
    let mut mesh = MeshData {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    for segment in 0..=segments {
        let phi = std::f32::consts::TAU * segment as f32 / segments as f32;
        let normal = Vec3::new(phi.cos(), 0.0, -phi.sin());
        for y in [-0.5, 0.5] {
            mesh.vertices
                .push(vertex(normal * 0.5 + Vec3::Y * y, normal));
        }
    }
    for segment in 0..segments {
        let (bottom, top) = (segment * 2, segment * 2 + 1);
        let (next_bottom, next_top) = (bottom + 2, top + 2);
        mesh.indices
            .extend([bottom, next_bottom, next_top, bottom, next_top, top]);
    }
    mesh
}

pub fn uv_sphere(segments: u32, rings: u32) -> MeshData {
    assert!(segments >= 3 && rings >= 2);
    let mut mesh = MeshData {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    for ring in 0..=rings {
        let theta = std::f32::consts::PI * ring as f32 / rings as f32; // 0 at +Y pole
        for segment in 0..=segments {
            let phi = std::f32::consts::TAU * segment as f32 / segments as f32;
            let normal = Vec3::new(
                theta.sin() * phi.cos(),
                theta.cos(),
                -theta.sin() * phi.sin(),
            );
            mesh.vertices.push(vertex(normal * 0.5, normal));
        }
    }
    let stride = segments + 1;
    for ring in 0..rings {
        for segment in 0..segments {
            let a = ring * stride + segment;
            let (b, c, d) = (a + stride, a + stride + 1, a + 1);
            // Skip the degenerate triangle at each pole.
            if ring != 0 {
                mesh.indices.extend([a, b, d]);
            }
            if ring != rings - 1 {
                mesh.indices.extend([d, b, c]);
            }
        }
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_mesh(name: &str, mesh: &MeshData) {
        assert_eq!(mesh.indices.len() % 3, 0, "{name}: not triangles");
        for &i in &mesh.indices {
            assert!(
                (i as usize) < mesh.vertices.len(),
                "{name}: index {i} out of range"
            );
        }
        for v in &mesh.vertices {
            let n = Vec3::from(v.normal);
            assert!(
                (n.length() - 1.0).abs() < 1e-5,
                "{name}: normal {n} not unit"
            );
        }
        for tri in mesh.indices.chunks(3) {
            let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| mesh.vertices[i as usize]);
            let face = (Vec3::from(b.position) - Vec3::from(a.position))
                .cross(Vec3::from(c.position) - Vec3::from(a.position));
            let normal = Vec3::from(a.normal) + Vec3::from(b.normal) + Vec3::from(c.normal);
            assert!(
                face.dot(normal) > 0.0,
                "{name}: triangle {tri:?} is clockwise from outside"
            );
        }
    }

    #[test]
    fn every_mesh_is_valid_and_outward_facing() {
        for kind in MeshKind::ALL {
            check_mesh(&format!("{kind:?}"), &kind.data());
        }
    }

    #[test]
    fn cube_has_flat_faces_of_size_one() {
        let mesh = cube();
        assert_eq!((mesh.vertices.len(), mesh.indices.len()), (24, 36));
        for v in &mesh.vertices {
            assert!(v.position.iter().all(|c| (c.abs() - 0.5).abs() < 1e-6));
        }
    }

    #[test]
    fn plane_has_a_face_on_each_side() {
        let normals: Vec<[f32; 3]> = plane().vertices.iter().map(|v| v.normal).collect();
        assert!(normals.contains(&[0.0, 1.0, 0.0]) && normals.contains(&[0.0, -1.0, 0.0]));
    }

    #[test]
    fn closest_points_on_a_triangle() {
        let t = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let cases = [
            (Vec3::new(0.2, 0.2, 3.0), Vec3::new(0.2, 0.2, 0.0)), // face
            (Vec3::new(-1.0, -1.0, 0.0), Vec3::ZERO),             // corner a
            (Vec3::new(2.0, -0.1, 0.0), Vec3::X),                 // corner b
            (Vec3::new(0.5, -1.0, 1.0), Vec3::new(0.5, 0.0, 0.0)), // edge ab
            (Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.5, 0.5, 0.0)), // edge bc
            (Vec3::new(-2.0, 0.5, 0.0), Vec3::new(0.0, 0.5, 0.0)), // edge ca
        ];
        for (p, want) in cases {
            assert!(closest_on_triangle(p, &t).distance(want) < 1e-6, "{p}");
        }
    }

    #[test]
    fn built_in_lods_are_valid_coarser_and_measured_exactly() {
        assert!(MeshKind::Cube.lods().is_empty() && MeshKind::Plane.lods().is_empty());
        for kind in [MeshKind::Sphere, MeshKind::Cylinder] {
            let lods = kind.lods();
            assert!(lods.len() < MAX_LODS);
            let mut last = (kind.data().indices.len(), 0.0);
            for (i, lod) in lods.iter().enumerate() {
                check_mesh(&format!("{kind:?} lod {}", i + 1), &lod.data);
                assert!(lod.data.indices.len() < last.0 && lod.error > last.1);
                last = (lod.data.indices.len(), lod.error);
            }
        }
        assert!(lod_error(&MeshKind::Sphere.data(), &MeshKind::Sphere.data()) < 1e-6);
        // The cylinder's error is exact: the 32-gon has a vertex over the
        // middle of every 16-gon chord, a sagitta away from it.
        use std::f32::consts::PI;
        let chord = 0.5 * (1.0 - (PI / 16.0).cos());
        assert!((MeshKind::Cylinder.lods()[0].error - chord).abs() < 1e-4);
        // The sphere's at least a 16-gon's (its equator), and within the
        // sagitta of a face's diagonal (π/8 across at most).
        let sphere = MeshKind::Sphere.lods()[0].error;
        let diagonal = 0.5 * (1.0 - (PI / 8.0).cos());
        assert!(sphere >= chord - 1e-4 && sphere <= diagonal, "{sphere}");
        let empty = MeshData {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
        assert_eq!(lod_error(&cube(), &empty), f32::INFINITY);
    }

    #[test]
    fn sphere_vertices_lie_on_radius_half() {
        for v in uv_sphere(32, 16).vertices {
            assert!((Vec3::from(v.position).length() - 0.5).abs() < 1e-5);
        }
    }
}
