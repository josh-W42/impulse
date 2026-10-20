//! Uploading loaded models to the renderer: their textures, materials and
//! meshes, static or skinned.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use impulse_assets::{
    ColorSpace, MaterialData, ModelData, SkinnedMesh, SkinnedModelData, TextureData,
};
use impulse_render::{
    Gpu, MaterialDesc, MaterialId, MeshData, MeshId, Renderer, SkinnedMeshData, SkinnedMeshId,
    SkinnedVertex, TextureId, Vertex,
};

/// Textures already uploaded, by content: identical pixels in the same
/// colour space (a kit's shared palette) upload once. Keep one for every
/// model a game uploads.
#[derive(Default)]
pub struct TextureCache {
    uploaded: HashMap<u64, TextureId>,
}

/// Uploads a model's textures, materials and meshes (one mesh per glTF
/// primitive, in order).
pub fn upload_model(
    renderer: &mut Renderer,
    gpu: &Gpu,
    data: &ModelData,
    cache: &mut TextureCache,
) -> Vec<MeshId> {
    let materials = upload_materials(renderer, gpu, &data.materials, &data.textures, cache);
    let mut default = None;
    data.meshes
        .iter()
        .map(|mesh| {
            let material = material_for(renderer, gpu, &materials, mesh.material, &mut default);
            let mesh_data = MeshData {
                vertices: mesh
                    .vertices
                    .iter()
                    .map(|v| Vertex {
                        position: v.position,
                        normal: v.normal,
                        uv: v.uv,
                        tangent: v.tangent,
                    })
                    .collect(),
                indices: mesh.indices.clone(),
            };
            renderer.add_mesh_with_material(gpu, &mesh_data, material)
        })
        .collect()
}

/// Uploads a skinned model's textures, materials and meshes (in order). Draw
/// them all with the same joint range.
pub fn upload_skinned_model(
    renderer: &mut Renderer,
    gpu: &Gpu,
    data: &SkinnedModelData,
    cache: &mut TextureCache,
) -> Vec<SkinnedMeshId> {
    let materials = upload_materials(renderer, gpu, &data.materials, &data.textures, cache);
    let mut default = None;
    data.meshes
        .iter()
        .map(|mesh| {
            let material = material_for(renderer, gpu, &materials, mesh.material, &mut default);
            renderer.add_skinned_mesh(gpu, &skinned_mesh_data(mesh), material)
        })
        .collect()
}

fn skinned_mesh_data(mesh: &SkinnedMesh) -> SkinnedMeshData {
    SkinnedMeshData {
        vertices: mesh
            .vertices
            .iter()
            .map(|v| SkinnedVertex {
                position: v.position,
                normal: v.normal,
                uv: v.uv,
                tangent: v.tangent,
                joints: v.joints,
                weights: v.weights,
            })
            .collect(),
        indices: mesh.indices.clone(),
    }
}

fn upload_materials(
    renderer: &mut Renderer,
    gpu: &Gpu,
    materials: &[MaterialData],
    textures: &[TextureData],
    cache: &mut TextureCache,
) -> Vec<MaterialId> {
    let mut texture = |renderer: &mut Renderer, index: Option<usize>| {
        let texture = textures.get(index?)?;
        Some(upload_texture(renderer, gpu, texture, cache))
    };
    materials
        .iter()
        .map(|m| {
            let desc = MaterialDesc {
                base_color: m.base_color,
                metallic: m.metallic,
                roughness: m.roughness,
                normal_scale: m.normal_scale,
                occlusion_strength: m.occlusion_strength,
                emissive: m.emissive,
                base_color_texture: texture(renderer, m.base_color_texture),
                metallic_roughness_texture: texture(renderer, m.metallic_roughness_texture),
                normal_texture: texture(renderer, m.normal_texture),
                occlusion_texture: texture(renderer, m.occlusion_texture),
                emissive_texture: texture(renderer, m.emissive_texture),
            };
            renderer.add_material(gpu, &desc)
        })
        .collect()
}

/// A primitive's material, or glTF's default material (made once per model).
fn material_for(
    renderer: &mut Renderer,
    gpu: &Gpu,
    materials: &[MaterialId],
    index: Option<usize>,
    default: &mut Option<MaterialId>,
) -> MaterialId {
    match index.and_then(|m| materials.get(m)) {
        Some(&id) => id,
        None => {
            *default.get_or_insert_with(|| renderer.add_material(gpu, &MaterialDesc::default()))
        }
    }
}

fn upload_texture(
    renderer: &mut Renderer,
    gpu: &Gpu,
    texture: &TextureData,
    cache: &mut TextureCache,
) -> TextureId {
    let base = &texture.mips[0];
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (base.width, base.height, &base.rgba, texture.space).hash(&mut hasher);
    *cache.uploaded.entry(hasher.finish()).or_insert_with(|| {
        let levels: Vec<&[u8]> = texture.mips.iter().map(|m| m.rgba.as_slice()).collect();
        let srgb = texture.space == ColorSpace::Srgb;
        renderer.add_texture(gpu, base.width, base.height, &levels, srgb)
    })
}

#[cfg(test)]
mod tests {
    use impulse_assets::{SkinnedMesh, SkinnedVertex as LoadedVertex};

    use super::*;

    #[test]
    fn skinned_vertices_convert_field_for_field() {
        let vertex = LoadedVertex {
            position: [1.0, 2.0, 3.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.25, 0.75],
            tangent: [1.0, 0.0, 0.0, -1.0],
            joints: [3, 1, 0, 0],
            weights: [0.75, 0.25, 0.0, 0.0],
        };
        let mesh = SkinnedMesh {
            vertices: vec![vertex],
            indices: vec![0, 0, 0],
            material: None,
        };
        let data = skinned_mesh_data(&mesh);
        assert_eq!(data.indices, [0, 0, 0]);
        let v = data.vertices[0];
        assert_eq!(
            (v.position, v.normal, v.uv, v.tangent, v.joints, v.weights),
            (
                vertex.position,
                vertex.normal,
                vertex.uv,
                vertex.tangent,
                vertex.joints,
                vertex.weights
            )
        );
    }
}
