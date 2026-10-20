//! The renderer's per-frame input: what to draw, rebuilt every frame.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};

#[cfg(test)]
use crate::mesh::MeshKind;
use crate::mesh::{MeshId, SkinnedMeshId};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Instance {
    pub mesh: MeshId,
    pub transform: Mat4,
    /// Linear RGBA (multiplies the material's base colour).
    pub color: [f32; 4],
    /// Multiply the material's roughness and metallic factors.
    pub roughness: f32,
    pub metallic: f32,
}

/// Roughness and metallic for built-in meshes drawn without their own values:
/// a slightly rough non-metal.
pub const DEFAULT_ROUGHNESS: f32 = 0.6;
pub const DEFAULT_METALLIC: f32 = 0.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DebugLine {
    pub a: Vec3,
    pub b: Vec3,
    pub color: [f32; 4],
}

/// Where an instance's joint matrices sit in [`DrawList::joints`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JointRange {
    pub offset: u32,
    pub count: u32,
}

/// A skinned mesh drawn with the joint matrices in `joints` (several meshes
/// of one model share a range).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkinnedInstance {
    pub mesh: SkinnedMeshId,
    pub transform: Mat4,
    pub color: [f32; 4],
    pub roughness: f32,
    pub metallic: f32,
    pub joints: JointRange,
}

#[derive(Debug, Clone, Default)]
pub struct DrawList {
    pub instances: Vec<Instance>,
    pub lines: Vec<DebugLine>,
    pub skinned: Vec<SkinnedInstance>,
    /// Skinning matrices for this frame's skinned instances.
    pub joints: Vec<Mat4>,
}

impl DrawList {
    pub fn clear(&mut self) {
        self.instances.clear();
        self.lines.clear();
        self.skinned.clear();
        self.joints.clear();
    }

    /// Adds one model's skinning matrices; pass the range to
    /// [`DrawList::push_skinned`] for each of its meshes.
    pub fn push_joints(&mut self, matrices: &[Mat4]) -> JointRange {
        let range = JointRange {
            offset: self.joints.len() as u32,
            count: matrices.len() as u32,
        };
        self.joints.extend_from_slice(matrices);
        range
    }

    /// A skinned mesh with its model's own surface (roughness and metallic
    /// multipliers of 1).
    pub fn push_skinned(
        &mut self,
        mesh: SkinnedMeshId,
        transform: Mat4,
        color: [f32; 4],
        joints: JointRange,
    ) {
        self.skinned.push(SkinnedInstance {
            mesh,
            transform,
            color,
            roughness: 1.0,
            metallic: 1.0,
            joints,
        });
    }

    /// An instance with the default surface ([`DEFAULT_ROUGHNESS`], [`DEFAULT_METALLIC`]).
    pub fn push_instance(&mut self, mesh: impl Into<MeshId>, transform: Mat4, color: [f32; 4]) {
        self.push_instance_pbr(mesh, transform, color, DEFAULT_ROUGHNESS, DEFAULT_METALLIC);
    }

    /// An instance whose roughness and metallic multiply its material's
    /// (1, 1 keeps a model's own material).
    pub fn push_instance_pbr(
        &mut self,
        mesh: impl Into<MeshId>,
        transform: Mat4,
        color: [f32; 4],
        roughness: f32,
        metallic: f32,
    ) {
        self.instances.push(Instance {
            mesh: mesh.into(),
            transform,
            color,
            roughness,
            metallic,
        });
    }

    pub fn push_line(&mut self, a: Vec3, b: Vec3, color: [f32; 4]) {
        self.lines.push(DebugLine { a, b, color });
    }
}

/// Per-instance vertex data as the shader reads it (144 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct InstanceRaw {
    pub model: [[f32; 4]; 4],
    /// Inverse-transpose of the model's 3x3 part, so normals stay perpendicular
    /// to surfaces under non-uniform scale. Columns padded to vec4.
    pub normal: [[f32; 4]; 3],
    pub color: [f32; 4],
    /// x: roughness, y: metallic.
    pub pbr: [f32; 4],
}

impl InstanceRaw {
    pub fn new(instance: &Instance) -> Self {
        let n = Mat3::from_mat4(instance.transform).inverse().transpose();
        let pad = |v: Vec3| [v.x, v.y, v.z, 0.0];
        Self {
            model: instance.transform.to_cols_array_2d(),
            normal: [pad(n.x_axis), pad(n.y_axis), pad(n.z_axis)],
            color: instance.color,
            pbr: [instance.roughness, instance.metallic, 0.0, 0.0],
        }
    }
}

/// Joint matrices reach the shader as an `Rgba32Float` texture this many
/// texels wide, four texels (the columns) per matrix, so a matrix never
/// straddles two rows.
pub const JOINT_TEXTURE_WIDTH: u32 = 1024;

/// Rows of joint texture for `matrices` matrices: at least one (the texture
/// is bound even when nothing is skinned), rounded up to a power of two so
/// the texture is rarely re-created.
pub fn joint_texture_rows(matrices: usize) -> u32 {
    let texels = matrices as u64 * 4;
    let rows = texels.div_ceil(JOINT_TEXTURE_WIDTH as u64).max(1);
    rows.next_power_of_two() as u32
}

/// The texture's contents: each matrix's columns in order, zero-padded to
/// `rows` full rows.
pub fn joint_texels(matrices: &[Mat4], rows: u32) -> Vec<[f32; 4]> {
    let mut texels: Vec<[f32; 4]> = matrices.iter().flat_map(|m| m.to_cols_array_2d()).collect();
    texels.resize((JOINT_TEXTURE_WIDTH * rows) as usize, [0.0; 4]);
    texels
}

/// Per-instance data of a skinned instance: [`InstanceRaw`] plus where its
/// joint matrices start (x; the rest is padding).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct SkinnedInstanceRaw {
    pub base: InstanceRaw,
    pub joints: [u32; 4],
}

/// Splits skinned instances by mesh (`mesh_count` groups), keeping their
/// order; unknown meshes are dropped.
pub fn group_skinned(
    instances: &[SkinnedInstance],
    mesh_count: usize,
) -> Vec<Vec<SkinnedInstanceRaw>> {
    let mut groups = vec![Vec::new(); mesh_count];
    for i in instances {
        if let Some(group) = groups.get_mut(i.mesh.0 as usize) {
            group.push(SkinnedInstanceRaw {
                base: InstanceRaw::new(&Instance {
                    mesh: MeshId(0),
                    transform: i.transform,
                    color: i.color,
                    roughness: i.roughness,
                    metallic: i.metallic,
                }),
                joints: [i.joints.offset, 0, 0, 0],
            });
        }
    }
    groups
}

/// Splits instances by mesh id (`mesh_count` groups), keeping their order.
/// Instances with unknown ids are dropped.
pub fn group_by_mesh(instances: &[Instance], mesh_count: usize) -> Vec<Vec<InstanceRaw>> {
    let mut groups = vec![Vec::new(); mesh_count];
    for instance in instances {
        if let Some(group) = groups.get_mut(instance.mesh.0 as usize) {
            group.push(InstanceRaw::new(instance));
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use glam::Quat;

    use super::*;

    fn at(mesh: MeshKind, x: f32) -> Instance {
        Instance {
            mesh: mesh.into(),
            transform: Mat4::from_translation(Vec3::X * x),
            roughness: DEFAULT_ROUGHNESS,
            metallic: DEFAULT_METALLIC,
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
        let groups = group_by_mesh(&list, MeshKind::ALL.len());
        let xs = |g: &Vec<InstanceRaw>| g.iter().map(|r| r.model[3][0]).collect::<Vec<_>>();
        assert_eq!(xs(&groups[MeshKind::Cube.index()]), vec![0.0, 2.0, 3.0]);
        assert_eq!(xs(&groups[MeshKind::Sphere.index()]), vec![1.0]);
        assert!(groups[MeshKind::Plane.index()].is_empty());
        assert!(group_by_mesh(&[], 4).iter().all(Vec::is_empty));
    }

    #[test]
    fn normals_stay_perpendicular_under_non_uniform_scale() {
        let transform = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 0.5, 1.0),
            Quat::from_rotation_z(0.7) * Quat::from_rotation_x(0.3),
            Vec3::new(4.0, 5.0, 6.0),
        );
        let raw = InstanceRaw::new(&Instance {
            mesh: MeshKind::Cube.into(),
            roughness: DEFAULT_ROUGHNESS,
            metallic: DEFAULT_METALLIC,
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
    fn translation(x: f32) -> Mat4 {
        Mat4::from_translation(Vec3::new(x, 0.0, 0.0))
    }

    #[test]
    fn push_joints_returns_consecutive_ranges() {
        let mut list = DrawList::default();
        let a = list.push_joints(&[translation(1.0), translation(2.0), translation(3.0)]);
        let b = list.push_joints(&[translation(4.0), translation(5.0)]);
        assert_eq!((a.offset, a.count, b.offset, b.count), (0, 3, 3, 2));
        assert_eq!(list.joints.len(), 5);
        list.push_skinned(SkinnedMeshId(0), Mat4::IDENTITY, [1.0; 4], b);
        assert_eq!(list.skinned[0].joints, b);
        assert_eq!(
            (list.skinned[0].roughness, list.skinned[0].metallic),
            (1.0, 1.0)
        );
        list.clear();
        assert!(list.joints.is_empty() && list.skinned.is_empty());
    }

    #[test]
    fn joint_rows() {
        // A frame with no skinned instances still gets a one-row texture.
        assert_eq!(joint_texture_rows(0), 1);
        assert_eq!(joint_texture_rows(1), 1);
        assert_eq!(joint_texture_rows(256), 1);
        assert_eq!(joint_texture_rows(257), 2);
        assert_eq!(joint_texture_rows(1000), 4);
    }

    #[test]
    fn joint_texels_are_columns_in_order() {
        let m = [translation(1.0), translation(2.0)];
        let texels = joint_texels(&m, 1);
        assert_eq!(texels.len(), JOINT_TEXTURE_WIDTH as usize);
        for (k, matrix) in m.iter().enumerate() {
            for c in 0..4 {
                assert_eq!(
                    texels[4 * k + c],
                    matrix.col(c).to_array(),
                    "matrix {k} column {c}"
                );
            }
        }
        assert!(texels[8..].iter().all(|t| *t == [0.0; 4]));
    }

    #[test]
    fn skinned_instances_group_by_mesh_with_their_offsets() {
        let mut list = DrawList::default();
        let joints = list.push_joints(&[Mat4::IDENTITY; 2]);
        let later = list.push_joints(&[Mat4::IDENTITY; 3]);
        list.push_skinned(SkinnedMeshId(1), Mat4::IDENTITY, [1.0; 4], joints);
        list.push_skinned(SkinnedMeshId(0), Mat4::IDENTITY, [1.0; 4], later);
        list.push_skinned(SkinnedMeshId(7), Mat4::IDENTITY, [1.0; 4], later);
        let groups = group_skinned(&list.skinned, 2);
        assert_eq!(groups[0].len(), 1);
        assert_eq!(groups[0][0].joints[0], 2);
        assert_eq!(groups[1][0].joints[0], 0);
        assert_eq!(
            groups.iter().map(Vec::len).sum::<usize>(),
            2,
            "unknown meshes dropped"
        );
    }
}
