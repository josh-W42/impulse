//! Unit primitive meshes. Instances scale them: cube and plane are size 1,
//! the sphere has radius 0.5. Triangles are counter-clockwise seen from outside.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshKind {
    Cube,
    Sphere,
    Plane,
}

impl MeshKind {
    pub const ALL: [MeshKind; 3] = [MeshKind::Cube, MeshKind::Sphere, MeshKind::Plane];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn data(self) -> MeshData {
        match self {
            MeshKind::Cube => cube(),
            MeshKind::Sphere => uv_sphere(32, 16),
            MeshKind::Plane => plane(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

fn vertex(position: Vec3, normal: Vec3) -> Vertex {
    Vertex {
        position: position.to_array(),
        normal: normal.to_array(),
    }
}

/// Adds one quad; corners must be counter-clockwise seen from the `normal` side.
fn push_quad(mesh: &mut MeshData, corners: [Vec3; 4], normal: Vec3) {
    let base = mesh.vertices.len() as u16;
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

pub fn uv_sphere(segments: u16, rings: u16) -> MeshData {
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
    fn sphere_vertices_lie_on_radius_half() {
        for v in uv_sphere(32, 16).vertices {
            assert!((Vec3::from(v.position).length() - 0.5).abs() < 1e-5);
        }
    }
}
