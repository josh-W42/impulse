//! Skinned models: a mesh in bind space plus the skeleton that bends it.
//!
//! Unlike [`crate::load_glb`], nothing is baked: vertices stay where the
//! skin's inverse bind matrices expect them, and the skeleton keeps each
//! joint's own transform so animation clips can replace it.

use glam::{Mat4, Quat, Vec3};
use gltf::buffer::Source as BufferSource;
use gltf::mesh::Mode;

use crate::{Limits, MaterialData, ModelError, TextureData, open, read_materials, read_primitive};

/// A joint's transform relative to its parent: what clips animate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointTransform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for JointTransform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl JointTransform {
    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Joint {
    /// The node's name (`#<index>` when it has none). Clips find joints by it.
    pub name: String,
    /// Index into [`Skeleton::joints`]; always lower than this joint's own.
    pub parent: Option<usize>,
    /// The node's own transform in the file (the rest pose).
    pub rest: JointTransform,
    /// For a joint with no parent joint: the world transform of the nodes
    /// above it (an armature's rotation or unit scale). Identity otherwise.
    pub base: Mat4,
    pub inverse_bind: Mat4,
}

/// The joints of a skin, parents before children.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Skeleton {
    pub joints: Vec<Joint>,
}

impl Skeleton {
    pub fn find(&self, name: &str) -> Option<usize> {
        self.joints.iter().position(|j| j.name == name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
    /// Indices into [`Skeleton::joints`].
    pub joints: [u16; 4],
    /// Sum to 1.
    pub weights: [f32; 4],
}

/// One triangle list with one material, in bind space.
#[derive(Debug, Clone, PartialEq)]
pub struct SkinnedMesh {
    pub vertices: Vec<SkinnedVertex>,
    pub indices: Vec<u32>,
    /// Index into [`SkinnedModelData::materials`]; `None` is the glTF default material.
    pub material: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SkinnedModelData {
    pub meshes: Vec<SkinnedMesh>,
    pub materials: Vec<MaterialData>,
    pub textures: Vec<TextureData>,
    pub skeleton: Skeleton,
    /// Bounds of every vertex in bind space.
    pub min: Vec3,
    pub max: Vec3,
}

/// Loads a skinned `.glb` with the default [`Limits`].
pub fn load_skinned_glb(bytes: &[u8]) -> Result<SkinnedModelData, ModelError> {
    load_skinned_glb_with(bytes, &Limits::default())
}

/// Loads a `.glb` with exactly one skin.
///
/// - Skinned meshes keep their bind-space vertices; the transform of the node
///   holding them is ignored, as glTF specifies.
/// - A mesh without a skin that hangs under a joint (a helmet on the head)
///   becomes skinned to that joint with full weight.
/// - Other unskinned meshes are left out, as are non-joint nodes between two
///   joints (treated as identity).
pub fn load_skinned_glb_with(
    bytes: &[u8],
    limits: &Limits,
) -> Result<SkinnedModelData, ModelError> {
    let gltf = open(bytes, limits)?;
    let blob = gltf.blob.as_deref();
    let bin = |buffer: gltf::Buffer| match buffer.source() {
        BufferSource::Bin => blob,
        BufferSource::Uri(_) => None,
    };
    let skin = {
        let mut skins = gltf.skins();
        let skin = skins.next().ok_or(ModelError::NoSkin)?;
        if skins.next().is_some() {
            return Err(ModelError::TooManySkins);
        }
        skin
    };
    let skin_joints: Vec<gltf::Node> = skin.joints().collect();
    if skin_joints.len() > limits.max_joints {
        return Err(ModelError::TooManyJoints {
            max: limits.max_joints,
        });
    }
    let inverse_binds: Vec<Mat4> = match skin.reader(bin).read_inverse_bind_matrices() {
        Some(read) => read.map(|m| Mat4::from_cols_array_2d(&m)).collect(),
        None => vec![Mat4::IDENTITY; skin_joints.len()],
    };
    if inverse_binds.len() != skin_joints.len() {
        return Err(ModelError::BadGeometry(
            "inverse bind matrices don't match the joints".into(),
        ));
    }

    let nodes: Vec<gltf::Node> = gltf.nodes().collect();
    let mut parent_of = vec![None; nodes.len()];
    for node in &nodes {
        for child in node.children() {
            parent_of[child.index()] = Some(node.index());
        }
    }
    let local = |i: usize| Mat4::from_cols_array_2d(&nodes[i].transform().matrix());
    // Skin joint index of each node that is a joint.
    let mut skin_index = vec![None; nodes.len()];
    for (k, node) in skin_joints.iter().enumerate() {
        skin_index[node.index()] = Some(k);
    }
    // The nearest ancestor that is a joint, and the product of the local
    // transforms strictly between it and `node` (or from the scene root).
    let nearest_joint_above = |node: usize| -> (Option<usize>, Mat4) {
        let mut between = Mat4::IDENTITY;
        let mut at = parent_of[node];
        while let Some(i) = at {
            if skin_index[i].is_some() {
                return (Some(i), between);
            }
            between = local(i) * between;
            at = parent_of[i];
        }
        (None, between)
    };

    // Parent-first order: by the number of joints above, then skin order.
    let depth = |node: usize| {
        let mut depth = 0;
        let mut at = nearest_joint_above(node).0;
        while let Some(i) = at {
            depth += 1;
            at = nearest_joint_above(i).0;
        }
        depth
    };
    let mut order: Vec<usize> = (0..skin_joints.len()).collect();
    order.sort_by_key(|&k| (depth(skin_joints[k].index()), k));
    let mut new_index = vec![0u16; skin_joints.len()];
    for (new, &old) in order.iter().enumerate() {
        new_index[old] = new as u16;
    }
    let joints = order
        .iter()
        .map(|&old| {
            let node = &skin_joints[old];
            let (above, between) = nearest_joint_above(node.index());
            let (t, r, s) = node.transform().decomposed();
            Joint {
                name: node
                    .name()
                    .map_or_else(|| format!("#{}", node.index()), str::to_owned),
                parent: above.map(|i| new_index[skin_index[i].expect("a joint")] as usize),
                rest: JointTransform {
                    translation: Vec3::from_array(t),
                    rotation: Quat::from_array(r),
                    scale: Vec3::from_array(s),
                },
                base: if above.is_some() {
                    Mat4::IDENTITY
                } else {
                    between
                },
                inverse_bind: inverse_binds[old],
            }
        })
        .collect();
    let skeleton = Skeleton { joints };

    let (materials, textures) = read_materials(&gltf, blob, limits)?;
    let mut meshes = Vec::new();
    let mut vertex_count = 0;
    for node in &nodes {
        let Some(mesh) = node.mesh() else { continue };
        let skinned = node.skin().is_some();
        // Rigid meshes under a joint: where they sit in bind space, and the joint.
        let rigid = if skinned {
            None
        } else {
            match nearest_joint_above(node.index()) {
                (Some(joint), between) => {
                    let k = skin_index[joint].expect("a joint");
                    Some((
                        inverse_binds[k].inverse() * between * local(node.index()),
                        new_index[k],
                    ))
                }
                (None, _) => continue,
            }
        };
        for primitive in mesh.primitives() {
            if primitive.mode() != Mode::Triangles {
                continue;
            }
            let transform = rigid.map_or(Mat4::IDENTITY, |(m, _)| m);
            let base = read_primitive(&primitive, &bin, transform)?;
            let count = base.vertices.len();
            vertex_count += count;
            if vertex_count > limits.max_vertices {
                return Err(ModelError::TooManyVertices {
                    max: limits.max_vertices,
                });
            }
            let influences: Vec<Influence> = match rigid {
                Some((_, joint)) => vec![([joint, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]); count],
                None => read_influences(&primitive, &bin, &new_index, count)?,
            };
            let vertices = base
                .vertices
                .iter()
                .zip(influences)
                .map(|(v, (joints, weights))| SkinnedVertex {
                    position: v.position,
                    normal: v.normal,
                    uv: v.uv,
                    tangent: v.tangent,
                    joints,
                    weights,
                })
                .collect();
            meshes.push(SkinnedMesh {
                vertices,
                indices: base.indices,
                material: base.material,
            });
        }
    }
    if meshes.iter().all(|m| m.indices.is_empty()) {
        return Err(ModelError::NoTriangles);
    }
    let (min, max) = meshes
        .iter()
        .flat_map(|m| &m.vertices)
        .map(|v| Vec3::from_array(v.position))
        .fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(lo, hi), p| (lo.min(p), hi.max(p)),
        );
    Ok(SkinnedModelData {
        meshes,
        materials,
        textures,
        skeleton,
        min,
        max,
    })
}

/// A vertex's joints and their weights.
type Influence = ([u16; 4], [f32; 4]);

/// `JOINTS_0` remapped to skeleton order, and `WEIGHTS_0` normalized (all
/// zero becomes full weight on joint 0).
fn read_influences<'a, 's, F>(
    primitive: &'a gltf::Primitive<'a>,
    bin: &F,
    new_index: &[u16],
    count: usize,
) -> Result<Vec<Influence>, ModelError>
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    let reader = primitive.reader(bin.clone());
    let missing =
        || ModelError::BadGeometry("a skinned primitive has no JOINTS_0/WEIGHTS_0".into());
    let joints: Vec<[u16; 4]> = reader
        .read_joints(0)
        .ok_or_else(missing)?
        .into_u16()
        .collect();
    let weights: Vec<[f32; 4]> = reader
        .read_weights(0)
        .ok_or_else(missing)?
        .into_f32()
        .collect();
    if joints.len() != count || weights.len() != count {
        return Err(ModelError::BadGeometry("attribute counts differ".into()));
    }
    joints
        .into_iter()
        .zip(weights)
        .map(|(j, w)| {
            let sum: f32 = w.iter().sum();
            if !(sum > 0.0 && sum.is_finite()) {
                return Ok(([0; 4], [1.0, 0.0, 0.0, 0.0]));
            }
            let mut mapped = [0u16; 4];
            for (out, (&joint, &weight)) in mapped.iter_mut().zip(j.iter().zip(&w)) {
                *out = match new_index.get(joint as usize) {
                    Some(&new) => new,
                    // An unused slot may name any joint; only a weighted one must exist.
                    None if weight == 0.0 => 0,
                    None => {
                        return Err(ModelError::BadGeometry("joint index out of range".into()));
                    }
                };
            }
            Ok((mapped, w.map(|x| x / sum)))
        })
        .collect()
}
