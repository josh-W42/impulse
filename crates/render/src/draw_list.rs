//! The renderer's per-frame input: what to draw, rebuilt every frame.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};

use crate::mesh::MeshKind;

pub const MESH_KINDS: usize = MeshKind::ALL.len();

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Instance {
    pub mesh: MeshKind,
    pub transform: Mat4,
    /// Linear RGBA.
    pub color: [f32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DebugLine {
    pub a: Vec3,
    pub b: Vec3,
    pub color: [f32; 4],
}

#[derive(Debug, Clone, Default)]
pub struct DrawList {
    pub instances: Vec<Instance>,
    pub lines: Vec<DebugLine>,
}

impl DrawList {
    pub fn clear(&mut self) {
        self.instances.clear();
        self.lines.clear();
    }

    pub fn push_instance(&mut self, mesh: MeshKind, transform: Mat4, color: [f32; 4]) {
        self.instances.push(Instance {
            mesh,
            transform,
            color,
        });
    }

    pub fn push_line(&mut self, a: Vec3, b: Vec3, color: [f32; 4]) {
        self.lines.push(DebugLine { a, b, color });
    }
}

/// Per-instance vertex data as the shader reads it (128 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct InstanceRaw {
    pub model: [[f32; 4]; 4],
    /// Inverse-transpose of the model's 3x3 part, so normals stay perpendicular
    /// to surfaces under non-uniform scale. Columns padded to vec4.
    pub normal: [[f32; 4]; 3],
    pub color: [f32; 4],
}

impl InstanceRaw {
    pub fn new(instance: &Instance) -> Self {
        let n = Mat3::from_mat4(instance.transform).inverse().transpose();
        let pad = |v: Vec3| [v.x, v.y, v.z, 0.0];
        Self {
            model: instance.transform.to_cols_array_2d(),
            normal: [pad(n.x_axis), pad(n.y_axis), pad(n.z_axis)],
            color: instance.color,
        }
    }
}

/// Splits instances by mesh (indexed by [`MeshKind::index`]), keeping their order.
pub fn group_by_mesh(instances: &[Instance]) -> [Vec<InstanceRaw>; MESH_KINDS] {
    let mut groups: [Vec<InstanceRaw>; MESH_KINDS] = Default::default();
    for instance in instances {
        groups[instance.mesh.index()].push(InstanceRaw::new(instance));
    }
    groups
}

#[cfg(test)]
mod tests {
    use glam::Quat;

    use super::*;

    fn at(mesh: MeshKind, x: f32) -> Instance {
        Instance {
            mesh,
            transform: Mat4::from_translation(Vec3::X * x),
            color: [1.0; 4],
        }
    }

    #[test]
    fn groups_keep_counts_and_order() {
        let list = [
            at(MeshKind::Cube, 0.0),
            at(MeshKind::Sphere, 1.0),
            at(MeshKind::Cube, 2.0),
            at(MeshKind::Cube, 3.0),
        ];
        let groups = group_by_mesh(&list);
        let xs = |g: &Vec<InstanceRaw>| g.iter().map(|r| r.model[3][0]).collect::<Vec<_>>();
        assert_eq!(xs(&groups[MeshKind::Cube.index()]), vec![0.0, 2.0, 3.0]);
        assert_eq!(xs(&groups[MeshKind::Sphere.index()]), vec![1.0]);
        assert!(groups[MeshKind::Plane.index()].is_empty());
        assert!(group_by_mesh(&[]).iter().all(Vec::is_empty));
    }

    #[test]
    fn normals_stay_perpendicular_under_non_uniform_scale() {
        let transform = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 0.5, 1.0),
            Quat::from_rotation_z(0.7) * Quat::from_rotation_x(0.3),
            Vec3::new(4.0, 5.0, 6.0),
        );
        let raw = InstanceRaw::new(&Instance {
            mesh: MeshKind::Cube,
            transform,
            color: [1.0; 4],
        });
        let normal_matrix = Mat3::from_cols(
            Vec3::from_slice(&raw.normal[0][..3]),
            Vec3::from_slice(&raw.normal[1][..3]),
            Vec3::from_slice(&raw.normal[2][..3]),
        );
        // A surface with normal n and tangent t (n ⟂ t) must keep them perpendicular.
        let (n, t) = (Vec3::new(1.0, 1.0, 0.0), Vec3::new(1.0, -1.0, 0.0));
        let n_world = normal_matrix * n;
        let t_world = transform.transform_vector3(t);
        assert!(n_world.normalize().dot(t_world.normalize()).abs() < 1e-5);
        // The naive model matrix would get this wrong — make sure the test can tell.
        let naive = transform.transform_vector3(n);
        assert!(naive.normalize().dot(t_world.normalize()).abs() > 0.1);
    }
}
