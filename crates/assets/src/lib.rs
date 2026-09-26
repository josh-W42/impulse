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

pub use colliders::{
    bounding_box, bounding_capsule, bounding_sphere, extreme_points, fibonacci_directions,
};
pub use mips::{MipLevel, mip_chain};

/// Size and complexity caps for loading untrusted files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub max_bytes: usize,
    pub max_vertices: usize,
    /// Largest texture width or height (texels).
    pub max_texture_size: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_bytes: 32 * 1024 * 1024,
            max_vertices: 500_000,
            max_texture_size: 4096,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
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
    /// Index into [`ModelData::textures`] (sRGB), sampled with UV set 0.
    pub base_color_texture: Option<usize>,
}

/// An RGBA8 texture with its full mip chain (level 0 first, down to 1×1).
#[derive(Debug, Clone, PartialEq)]
pub struct TextureData {
    pub mips: Vec<MipLevel>,
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
        }
    }
}

impl std::error::Error for ModelError {}

/// Loads a `.glb` with the default [`Limits`].
pub fn load_glb(bytes: &[u8]) -> Result<ModelData, ModelError> {
    load_glb_with(bytes, &Limits::default())
}

pub fn load_glb_with(bytes: &[u8], limits: &Limits) -> Result<ModelData, ModelError> {
    if bytes.len() > limits.max_bytes {
        return Err(ModelError::TooLarge {
            bytes: bytes.len(),
            max: limits.max_bytes,
        });
    }
    let gltf = gltf::Gltf::from_slice(bytes).map_err(|e| ModelError::Parse(e.to_string()))?;
    let blob = gltf.blob.as_deref();
    if gltf
        .buffers()
        .any(|b| matches!(b.source(), BufferSource::Uri(_)))
        || gltf
            .images()
            .any(|i| matches!(i.source(), ImageSource::Uri { .. }))
    {
        return Err(ModelError::ExternalData);
    }
    let bin = |buffer: gltf::Buffer| match buffer.source() {
        BufferSource::Bin => blob,
        BufferSource::Uri(_) => None,
    };

    // Textures: decode each image used as a base colour, once.
    let mut textures = Vec::new();
    let mut texture_of_image = vec![None; gltf.images().len()];
    let mut materials = Vec::new();
    for material in gltf.materials() {
        let pbr = material.pbr_metallic_roughness();
        let base_color_texture = match pbr.base_color_texture() {
            Some(info) => {
                let image = info.texture().source();
                let slot = &mut texture_of_image[image.index()];
                if slot.is_none() {
                    let rgba = decode_image(&image, blob, limits)?;
                    textures.push(TextureData {
                        mips: mip_chain(rgba.0, rgba.1, rgba.2, true),
                    });
                    *slot = Some(textures.len() - 1);
                }
                *slot
            }
            None => None,
        };
        materials.push(MaterialData {
            base_color: pbr.base_color_factor(),
            base_color_texture,
        });
    }

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
    let vertices = positions
        .iter()
        .zip(&normals)
        .zip(&uvs)
        .map(|((p, n), uv)| ModelVertex {
            position: p.to_array(),
            normal: n.to_array(),
            uv: *uv,
        })
        .collect();
    Ok(ModelMesh {
        vertices,
        indices,
        material: primitive.material().index(),
    })
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
