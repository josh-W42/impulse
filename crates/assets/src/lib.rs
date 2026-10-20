//! Model loading for the impulse engine: `.glb` (binary glTF 2.0) in, plain
//! data out. No rendering or physics dependencies, so it is testable headless.
//!
//! A model comes out as one rigid mesh set in its own space: node transforms
//! are baked in, skins and animations are ignored (rest pose). Only
//! self-contained `.glb` files are accepted; external buffers and images can't
//! be followed from a browser file picker.

use std::fmt;
use std::io::Cursor;

use glam::{Mat3, Mat4, Vec3};
use gltf::buffer::Source as BufferSource;
use gltf::image::Source as ImageSource;
use gltf::mesh::Mode;

mod colliders;
mod mips;
mod skin;

pub use colliders::{
    bounding_box, bounding_capsule, bounding_sphere, extreme_points, fibonacci_directions,
};
pub use mips::{ColorSpace, MipLevel, mip_chain};
pub use skin::{
    Joint, JointTransform, Skeleton, SkinnedMesh, SkinnedModelData, SkinnedVertex,
    load_skinned_glb, load_skinned_glb_with,
};

/// Size and complexity caps for loading untrusted files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub max_bytes: usize,
    pub max_vertices: usize,
    /// Largest texture width or height (texels).
    pub max_texture_size: u32,
    /// Most joints in a skin.
    pub max_joints: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_bytes: 32 * 1024 * 1024,
            max_vertices: 500_000,
            max_texture_size: 4096,
            max_joints: 128,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// Tangent along +U (xyz) and bitangent sign (w), for normal maps.
    pub tangent: [f32; 4],
}

/// One triangle list with one material.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelMesh {
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u32>,
    /// Index into [`ModelData::materials`]; `None` is the glTF default material.
    pub material: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialData {
    /// Linear RGBA multiplier.
    pub base_color: [f32; 4],
    /// Indices into [`ModelData::textures`], all sampled with UV set 0.
    pub base_color_texture: Option<usize>,
    pub metallic: f32,
    pub roughness: f32,
    /// Roughness in G, metalness in B (glTF packing), linear.
    pub metallic_roughness_texture: Option<usize>,
    pub normal_texture: Option<usize>,
    pub normal_scale: f32,
    /// Ambient occlusion in R, linear.
    pub occlusion_texture: Option<usize>,
    pub occlusion_strength: f32,
    /// Linear RGB emitted light (multiplies the emissive texture).
    pub emissive: [f32; 3],
    pub emissive_texture: Option<usize>,
}

impl Default for MaterialData {
    /// glTF's default material.
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            base_color_texture: None,
            metallic: 1.0,
            roughness: 1.0,
            metallic_roughness_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            occlusion_texture: None,
            occlusion_strength: 1.0,
            emissive: [0.0; 3],
            emissive_texture: None,
        }
    }
}

/// An RGBA8 texture with its full mip chain (level 0 first, down to 1×1).
#[derive(Debug, Clone, PartialEq)]
pub struct TextureData {
    pub mips: Vec<MipLevel>,
    /// sRGB colour (sample as sRGB) or linear data.
    pub space: ColorSpace,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelData {
    pub meshes: Vec<ModelMesh>,
    pub materials: Vec<MaterialData>,
    pub textures: Vec<TextureData>,
    /// Axis-aligned bounds of every vertex, in model space.
    pub min: Vec3,
    pub max: Vec3,
}

impl ModelData {
    pub fn vertex_count(&self) -> usize {
        self.meshes.iter().map(|m| m.vertices.len()).sum()
    }

    /// Every vertex position, in model space.
    pub fn positions(&self) -> impl Iterator<Item = Vec3> + '_ {
        self.meshes
            .iter()
            .flat_map(|m| m.vertices.iter().map(|v| Vec3::from_array(v.position)))
    }
}

/// Why a model could not be loaded.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    TooLarge {
        bytes: usize,
        max: usize,
    },
    /// Not a valid binary glTF 2.0 file.
    Parse(String),
    /// Buffers or images live in other files (only self-contained `.glb` works).
    ExternalData,
    TooManyVertices {
        max: usize,
    },
    /// A texture failed to decode or exceeded the size limit.
    Texture(String),
    /// A triangle primitive has no positions or out-of-range indices.
    BadGeometry(String),
    /// Nothing drawable: no triangle primitives in the scene.
    NoTriangles,
    /// A skinned model was expected, but the file has no skin.
    NoSkin,
    /// More than one skin (only single-skeleton models are supported).
    TooManySkins,
    TooManyJoints {
        max: usize,
    },
    /// An animation's keyframes are malformed.
    Animation(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge { bytes, max } => write!(
                f,
                "file is {} MB; the limit is {} MB",
                bytes / (1024 * 1024),
                max / (1024 * 1024)
            ),
            Self::Parse(e) => write!(f, "not a valid .glb file: {e}"),
            Self::ExternalData => write!(
                f,
                "the model references external files; use a self-contained .glb"
            ),
            Self::TooManyVertices { max } => write!(f, "more than {max} vertices"),
            Self::Texture(e) => write!(f, "texture: {e}"),
            Self::BadGeometry(e) => write!(f, "geometry: {e}"),
            Self::NoTriangles => write!(f, "the model has no triangles to draw"),
            Self::NoSkin => write!(f, "the model has no skin (skeleton)"),
            Self::TooManySkins => write!(f, "the model has more than one skin"),
            Self::TooManyJoints { max } => write!(f, "more than {max} joints in the skin"),
            Self::Animation(e) => write!(f, "animation: {e}"),
        }
    }
}

impl std::error::Error for ModelError {}

/// Loads a `.glb` with the default [`Limits`].
pub fn load_glb(bytes: &[u8]) -> Result<ModelData, ModelError> {
    load_glb_with(bytes, &Limits::default())
}

pub fn load_glb_with(bytes: &[u8], limits: &Limits) -> Result<ModelData, ModelError> {
    let gltf = open(bytes, limits)?;
    let blob = gltf.blob.as_deref();
    let bin = |buffer: gltf::Buffer| match buffer.source() {
        BufferSource::Bin => blob,
        BufferSource::Uri(_) => None,
    };
    let (materials, textures) = read_materials(&gltf, blob, limits)?;

    // Geometry: walk the scene graph, baking each node's world transform.
    let roots: Vec<gltf::Node> = match gltf.default_scene().or_else(|| gltf.scenes().next()) {
        Some(scene) => scene.nodes().collect(),
        None => Vec::new(),
    };
    let mut meshes = Vec::new();
    let mut vertex_count = 0;
    let mut stack: Vec<(gltf::Node, Mat4)> =
        roots.into_iter().map(|n| (n, Mat4::IDENTITY)).collect();
    while let Some((node, parent)) = stack.pop() {
        let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        for child in node.children() {
            stack.push((child, world));
        }
        let Some(mesh) = node.mesh() else { continue };
        for primitive in mesh.primitives() {
            if primitive.mode() != Mode::Triangles {
                continue;
            }
            let built = read_primitive(&primitive, &bin, world)?;
            vertex_count += built.vertices.len();
            if vertex_count > limits.max_vertices {
                return Err(ModelError::TooManyVertices {
                    max: limits.max_vertices,
                });
            }
            meshes.push(built);
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
    Ok(ModelData {
        meshes,
        materials,
        textures,
        min,
        max,
    })
}

/// Parses a self-contained `.glb` within the size limit.
fn open(bytes: &[u8], limits: &Limits) -> Result<gltf::Gltf, ModelError> {
    if bytes.len() > limits.max_bytes {
        return Err(ModelError::TooLarge {
            bytes: bytes.len(),
            max: limits.max_bytes,
        });
    }
    let gltf = gltf::Gltf::from_slice(bytes).map_err(|e| ModelError::Parse(e.to_string()))?;
    if gltf
        .buffers()
        .any(|b| matches!(b.source(), BufferSource::Uri(_)))
        || gltf
            .images()
            .any(|i| matches!(i.source(), ImageSource::Uri { .. }))
    {
        return Err(ModelError::ExternalData);
    }
    Ok(gltf)
}

/// Every material, and the textures they use (each image and colour space
/// decoded once).
fn read_materials(
    gltf: &gltf::Gltf,
    blob: Option<&[u8]>,
    limits: &Limits,
) -> Result<(Vec<MaterialData>, Vec<TextureData>), ModelError> {
    let mut textures: Vec<TextureData> = Vec::new();
    let mut decoded: Vec<(usize, ColorSpace)> = Vec::new();
    let mut texture = |image: gltf::Image, space: ColorSpace| -> Result<usize, ModelError> {
        let key = (image.index(), space);
        if let Some(i) = decoded.iter().position(|k| *k == key) {
            return Ok(i);
        }
        let (rgba, width, height) = decode_image(&image, blob, limits)?;
        textures.push(TextureData {
            mips: mip_chain(rgba, width, height, space),
            space,
        });
        decoded.push(key);
        Ok(textures.len() - 1)
    };
    let mut materials = Vec::new();
    for material in gltf.materials() {
        let pbr = material.pbr_metallic_roughness();
        let mut data = MaterialData {
            base_color: pbr.base_color_factor(),
            metallic: pbr.metallic_factor(),
            roughness: pbr.roughness_factor(),
            emissive: material.emissive_factor(),
            ..MaterialData::default()
        };
        if let Some(info) = pbr.base_color_texture() {
            data.base_color_texture = Some(texture(info.texture().source(), ColorSpace::Srgb)?);
        }
        if let Some(info) = pbr.metallic_roughness_texture() {
            data.metallic_roughness_texture =
                Some(texture(info.texture().source(), ColorSpace::Linear)?);
        }
        if let Some(info) = material.normal_texture() {
            data.normal_texture = Some(texture(info.texture().source(), ColorSpace::Normal)?);
            data.normal_scale = info.scale();
        }
        if let Some(info) = material.occlusion_texture() {
            data.occlusion_texture = Some(texture(info.texture().source(), ColorSpace::Linear)?);
            data.occlusion_strength = info.strength();
        }
        if let Some(info) = material.emissive_texture() {
            data.emissive_texture = Some(texture(info.texture().source(), ColorSpace::Srgb)?);
        }
        materials.push(data);
    }
    Ok((materials, textures))
}

fn read_primitive<'a, 's, F>(
    primitive: &'a gltf::Primitive<'a>,
    bin: &F,
    world: Mat4,
) -> Result<ModelMesh, ModelError>
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    let reader = primitive.reader(bin.clone());
    let positions: Vec<Vec3> = reader
        .read_positions()
        .ok_or_else(|| ModelError::BadGeometry("a triangle primitive has no positions".into()))?
        .map(|p| world.transform_point3(Vec3::from_array(p)))
        .collect();
    if positions.iter().any(|p| !p.is_finite()) {
        return Err(ModelError::BadGeometry("non-finite positions".into()));
    }
    let mut indices: Vec<u32> = match reader.read_indices() {
        Some(read) => read.into_u32().collect(),
        None => (0..positions.len() as u32).collect(),
    };
    indices.truncate(indices.len() / 3 * 3);
    if indices.iter().any(|&i| i as usize >= positions.len()) {
        return Err(ModelError::BadGeometry("index out of range".into()));
    }
    // A mirroring transform flips triangles inside out; restore the winding.
    if world.determinant() < 0.0 {
        for tri in indices.as_chunks_mut::<3>().0 {
            tri.swap(1, 2);
        }
    }
    let normal_matrix = Mat3::from_mat4(world).inverse().transpose();
    let normals: Vec<Vec3> = match reader.read_normals() {
        Some(read) => read
            .map(|n| (normal_matrix * Vec3::from_array(n)).normalize_or(Vec3::Y))
            .collect(),
        None => smooth_normals(&positions, &indices),
    };
    let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
        Some(read) => read.into_f32().collect(),
        None => vec![[0.0; 2]; positions.len()],
    };
    if normals.len() != positions.len() || uvs.len() != positions.len() {
        return Err(ModelError::BadGeometry("attribute counts differ".into()));
    }
    let tangents: Vec<[f32; 4]> = match reader.read_tangents() {
        Some(read) => {
            let flip = if world.determinant() < 0.0 { -1.0 } else { 1.0 };
            read.zip(&normals)
                .map(|(t, n)| {
                    let dir = Mat3::from_mat4(world) * Vec3::new(t[0], t[1], t[2]);
                    let dir = (dir - *n * n.dot(dir)).normalize_or(any_perpendicular(*n));
                    let sign = if t[3] < 0.0 { -flip } else { flip };
                    [dir.x, dir.y, dir.z, sign]
                })
                .collect()
        }
        None => generate_tangents(&positions, &normals, &uvs, &indices),
    };
    if tangents.len() != positions.len() {
        return Err(ModelError::BadGeometry("attribute counts differ".into()));
    }
    let vertices = positions
        .iter()
        .zip(&normals)
        .zip(&uvs)
        .zip(&tangents)
        .map(|(((p, n), uv), t)| ModelVertex {
            position: p.to_array(),
            normal: n.to_array(),
            uv: *uv,
            tangent: *t,
        })
        .collect();
    Ok(ModelMesh {
        vertices,
        indices,
        material: primitive.material().index(),
    })
}

/// Some unit vector perpendicular to unit `n`.
fn any_perpendicular(n: Vec3) -> Vec3 {
    let other = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    n.cross(other).normalize()
}

/// Per-vertex tangents from UVs (a simplified MikkTSpace): each triangle's
/// ∂position/∂u and ∂position/∂v are summed per vertex, then the tangent is
/// made perpendicular to the normal; `w` is the bitangent's handedness.
/// Triangles with degenerate UVs contribute nothing; vertices with no usable
/// tangent get any perpendicular.
pub fn generate_tangents(
    positions: &[Vec3],
    normals: &[Vec3],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    let mut tan = vec![Vec3::ZERO; positions.len()];
    let mut bit = vec![Vec3::ZERO; positions.len()];
    for tri in indices.as_chunks::<3>().0 {
        let [a, b, c] = tri.map(|i| i as usize);
        let (e1, e2) = (positions[b] - positions[a], positions[c] - positions[a]);
        let (du1, dv1) = (uvs[b][0] - uvs[a][0], uvs[b][1] - uvs[a][1]);
        let (du2, dv2) = (uvs[c][0] - uvs[a][0], uvs[c][1] - uvs[a][1]);
        let det = du1 * dv2 - du2 * dv1;
        if det.abs() < 1e-12 {
            continue;
        }
        let t = (e1 * dv2 - e2 * dv1) / det;
        let bt = (e2 * du1 - e1 * du2) / det;
        for i in [a, b, c] {
            tan[i] += t;
            bit[i] += bt;
        }
    }
    normals
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let t = (tan[i] - n * n.dot(tan[i])).normalize_or(any_perpendicular(n));
            let w = if n.cross(t).dot(bit[i]) < 0.0 {
                -1.0
            } else {
                1.0
            };
            [t.x, t.y, t.z, w]
        })
        .collect()
}

/// Area-weighted vertex normals, for files that omit them.
fn smooth_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for tri in indices.as_chunks::<3>().0 {
        let [a, b, c] = tri.map(|i| i as usize);
        let face = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        for i in [a, b, c] {
            normals[i] += face;
        }
    }
    normals
        .into_iter()
        .map(|n| n.normalize_or(Vec3::Y))
        .collect()
}

/// Decodes an embedded PNG/JPEG to RGBA8, refusing oversized images before
/// allocating them.
fn decode_image(
    image: &gltf::Image,
    blob: Option<&[u8]>,
    limits: &Limits,
) -> Result<(Vec<u8>, u32, u32), ModelError> {
    let ImageSource::View { view, .. } = image.source() else {
        return Err(ModelError::ExternalData);
    };
    let data = blob
        .and_then(|b| b.get(view.offset()..view.offset() + view.length()))
        .ok_or_else(|| ModelError::Texture("image data is outside the file".into()))?;
    let mut reader = image::ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| ModelError::Texture(e.to_string()))?;
    let mut caps = image::Limits::default();
    caps.max_image_width = Some(limits.max_texture_size);
    caps.max_image_height = Some(limits.max_texture_size);
    reader.limits(caps);
    let decoded = reader.decode().map_err(|e| match e {
        image::ImageError::Limits(_) => {
            ModelError::Texture(format!("larger than {0}×{0}", limits.max_texture_size))
        }
        e => ModelError::Texture(e.to_string()),
    })?;
    let rgba = decoded.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok((rgba.into_raw(), width, height))
}

#[cfg(test)]
mod tests;
