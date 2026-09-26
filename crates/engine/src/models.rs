//! Models available to scenes: the bundled library (compiled in, so it loads
//! synchronously and identically everywhere) plus any imported this session.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use glam::{Mat4, Vec3};
use impulse_assets::{ColorSpace, ModelData, ModelError, TextureData, extreme_points, load_glb};
use impulse_physics::{Collider, World};
use impulse_render::{
    Gpu, MaterialDesc, MaterialId, MeshData, MeshId, Renderer, TextureId, Vertex,
};

use crate::scene_desc::{ModelCollider, ObjectError};

/// Directions sampled for a model's hull collider (the physics hull cap).
pub const HULL_DIRECTIONS: usize = impulse_physics::MAX_HULL_POINTS;

const BUNDLED: [(&str, &[u8]); 8] = [
    (
        "barrel",
        include_bytes!("../../../assets/models/barrel.glb"),
    ),
    ("box", include_bytes!("../../../assets/models/box.glb")),
    (
        "box-large",
        include_bytes!("../../../assets/models/box-large.glb"),
    ),
    (
        "bucket",
        include_bytes!("../../../assets/models/bucket.glb"),
    ),
    ("chest", include_bytes!("../../../assets/models/chest.glb")),
    (
        "rock-a",
        include_bytes!("../../../assets/models/rock-a.glb"),
    ),
    (
        "rock-b",
        include_bytes!("../../../assets/models/rock-b.glb"),
    ),
    (
        "tree-log",
        include_bytes!("../../../assets/models/tree-log.glb"),
    ),
];

/// Identifies a model in a [`ModelRegistry`]; stable for the session
/// (re-importing a name keeps its id).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelId(pub u32);

pub struct ModelAsset {
    pub data: ModelData,
    /// Extreme vertices for the hull collider (model space, unscaled).
    pub hull_points: Vec<Vec3>,
}

impl ModelAsset {
    fn new(data: ModelData) -> Self {
        let hull_points = extreme_points(&data, HULL_DIRECTIONS);
        Self { data, hull_points }
    }
}

#[derive(Clone)]
struct Entry {
    name: String,
    asset: Arc<ModelAsset>,
    bundled: bool,
}

#[derive(Default)]
pub struct ModelRegistry {
    entries: Vec<Entry>,
}

/// Why an import was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum ImportError {
    /// No usable name (see `scene_desc::file_stem`).
    BadName,
    /// Bundled models can't be replaced.
    NameTaken(String),
    Model(ModelError),
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadName => write!(f, "the file needs a name made of letters, digits, - or _"),
            Self::NameTaken(name) => write!(
                f,
                "a bundled model is already called '{name}'; rename the file"
            ),
            Self::Model(e) => write!(f, "{e}"),
        }
    }
}

impl ModelRegistry {
    /// The bundled library. Parsed once per process and shared (cheap to call).
    pub fn bundled() -> Self {
        static PARSED: OnceLock<Vec<Entry>> = OnceLock::new();
        let entries = PARSED.get_or_init(|| {
            BUNDLED
                .iter()
                .map(|(name, bytes)| Entry {
                    name: name.to_string(),
                    asset: Arc::new(ModelAsset::new(
                        load_glb(bytes).expect("bundled models are valid"),
                    )),
                    bundled: true,
                })
                .collect()
        });
        Self {
            entries: entries.clone(),
        }
    }

    /// Adds (or replaces) an imported model named after `file_name`'s stem;
    /// returns its id and name.
    pub fn import(
        &mut self,
        file_name: &str,
        bytes: &[u8],
    ) -> Result<(ModelId, String), ImportError> {
        let stem = file_name
            .rsplit_once('.')
            .map_or(file_name, |(stem, _)| stem);
        let name = crate::scene_desc::file_stem(stem).ok_or(ImportError::BadName)?;
        let asset = Arc::new(ModelAsset::new(
            load_glb(bytes).map_err(ImportError::Model)?,
        ));
        match self.entries.iter_mut().position(|e| e.name == name) {
            Some(i) if self.entries[i].bundled => Err(ImportError::NameTaken(name)),
            Some(i) => {
                self.entries[i].asset = asset;
                Ok((ModelId(i as u32), name))
            }
            None => {
                self.entries.push(Entry {
                    name: name.clone(),
                    asset,
                    bundled: false,
                });
                Ok((ModelId(self.entries.len() as u32 - 1), name))
            }
        }
    }

    pub fn id(&self, name: &str) -> Option<ModelId> {
        self.entries
            .iter()
            .position(|e| e.name == name)
            .map(|i| ModelId(i as u32))
    }

    pub fn get(&self, id: ModelId) -> Option<&ModelAsset> {
        self.entries.get(id.0 as usize).map(|e| &*e.asset)
    }

    /// `(id, name, bundled)` for every model, in id order.
    pub fn list(&self) -> impl Iterator<Item = (ModelId, &str, bool)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, e)| (ModelId(i as u32), e.name.as_str(), e.bundled))
    }
}

/// Scale that makes the model's largest dimension about `target` metres
/// (models arrive in any units), rounded to two decimals.
pub fn fitting_scale(asset: &ModelAsset, target: f32) -> f32 {
    let size = (asset.data.max - asset.data.min).max_element();
    if size <= 0.0 || !size.is_finite() {
        return 1.0;
    }
    ((target / size).clamp(0.01, 100.0) * 100.0).round() / 100.0
}

/// Render meshes per model (one per glTF primitive).
pub type ModelMeshes = HashMap<ModelId, Vec<MeshId>>;

/// Uploads every model in `registry` with its full material. Identical
/// textures (same pixels, same colour space; e.g. a kit's shared palette) are
/// uploaded once.
pub fn upload_models(renderer: &mut Renderer, gpu: &Gpu, registry: &ModelRegistry) -> ModelMeshes {
    let mut textures: HashMap<u64, TextureId> = HashMap::new();
    let mut out = ModelMeshes::new();
    for (id, _, _) in registry.list() {
        let data = &registry.get(id).expect("listed").data;
        let mut texture = |index: Option<usize>| {
            let texture = data.textures.get(index?)?;
            Some(upload_texture(renderer, gpu, texture, &mut textures))
        };
        let materials: Vec<MaterialDesc> = data
            .materials
            .iter()
            .map(|m| MaterialDesc {
                base_color: m.base_color,
                metallic: m.metallic,
                roughness: m.roughness,
                normal_scale: m.normal_scale,
                occlusion_strength: m.occlusion_strength,
                emissive: m.emissive,
                base_color_texture: texture(m.base_color_texture),
                metallic_roughness_texture: texture(m.metallic_roughness_texture),
                normal_texture: texture(m.normal_texture),
                occlusion_texture: texture(m.occlusion_texture),
                emissive_texture: texture(m.emissive_texture),
            })
            .collect();
        let material_ids: Vec<MaterialId> = materials
            .iter()
            .map(|m| renderer.add_material(gpu, m))
            .collect();
        // Primitives without a material get glTF's default material.
        let mut default = None;
        let meshes = data
            .meshes
            .iter()
            .map(|mesh| {
                let material = match mesh.material.and_then(|m| material_ids.get(m)) {
                    Some(&id) => id,
                    None => *default.get_or_insert_with(|| {
                        renderer.add_material(gpu, &MaterialDesc::default())
                    }),
                };
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
            .collect();
        out.insert(id, meshes);
    }
    out
}

fn upload_texture(
    renderer: &mut Renderer,
    gpu: &Gpu,
    texture: &TextureData,
    uploaded: &mut HashMap<u64, TextureId>,
) -> TextureId {
    let base = &texture.mips[0];
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (base.width, base.height, &base.rgba, texture.space).hash(&mut hasher);
    *uploaded.entry(hasher.finish()).or_insert_with(|| {
        let levels: Vec<&[u8]> = texture.mips.iter().map(|m| m.rgba.as_slice()).collect();
        let srgb = texture.space == ColorSpace::Srgb;
        renderer.add_texture(gpu, base.width, base.height, &levels, srgb)
    })
}

/// Where a model's mesh sits on its body: `transform` maps model space to
/// body space (scale, then shift so the collider's center is the body origin).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelVisual {
    pub model: ModelId,
    pub transform: Mat4,
}

/// The collider for a model at `scale`, and where its mesh goes on the body.
/// Hull colliders are added to `world`.
pub fn model_collider(
    id: ModelId,
    asset: &ModelAsset,
    scale: f32,
    kind: ModelCollider,
    world: &mut World,
) -> Result<(Collider, ModelVisual), ObjectError> {
    if !(scale.is_finite() && scale > 0.0) {
        return Err(ObjectError::InvalidScale);
    }
    let data = &asset.data;
    let bounds_center = (data.min + data.max) / 2.0 * scale;
    let (collider, center) = match kind {
        ModelCollider::Hull => {
            let points: Vec<Vec3> = asset.hull_points.iter().map(|p| *p * scale).collect();
            let hull = world.add_hull(&points).map_err(ObjectError::Hull)?;
            let center = world.hull(hull).expect("just added").center();
            (Collider::Hull { hull }, center)
        }
        ModelCollider::Box => {
            let (_, half) = impulse_assets::bounding_box(data);
            (
                Collider::Box {
                    half_extents: half * scale,
                },
                bounds_center,
            )
        }
        ModelCollider::Sphere => {
            let (_, radius) = impulse_assets::bounding_sphere(data);
            (
                Collider::Sphere {
                    radius: radius * scale,
                },
                bounds_center,
            )
        }
        ModelCollider::Capsule => {
            let (_, half_height, radius) = impulse_assets::bounding_capsule(data);
            (
                Collider::Capsule {
                    half_height: half_height * scale,
                    radius: radius * scale,
                },
                bounds_center,
            )
        }
    };
    let transform = Mat4::from_translation(-center) * Mat4::from_scale(Vec3::splat(scale));
    Ok((
        collider,
        ModelVisual {
            model: id,
            transform,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bundled_library_loads_with_hull_points() {
        let registry = ModelRegistry::bundled();
        assert_eq!(registry.list().count(), BUNDLED.len());
        for (id, name, bundled) in registry.list() {
            let asset = registry.get(id).unwrap();
            assert!(bundled);
            assert!(
                (4..=HULL_DIRECTIONS).contains(&asset.hull_points.len()),
                "{name}: {} hull points",
                asset.hull_points.len()
            );
            // Every hull builds in physics at a couple of scales.
            for scale in [0.5, 2.0] {
                let mut world = World::default();
                model_collider(id, asset, scale, ModelCollider::Hull, &mut world)
                    .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            }
        }
    }

    #[test]
    fn imports_are_named_by_file_stem_and_cannot_replace_bundled_models() {
        let mut registry = ModelRegistry::bundled();
        let bytes = include_bytes!("../../../assets/models/box.glb");
        let (id, name) = registry.import("My Crate.glb", bytes).unwrap();
        assert_eq!(name, "My_Crate");
        assert_eq!(registry.id("My_Crate"), Some(id));
        let (again, _) = registry.import("My Crate.glb", bytes).unwrap();
        assert_eq!(again, id, "re-importing keeps the id");
        assert_eq!(
            registry.import("barrel.glb", bytes).err(),
            Some(ImportError::NameTaken("barrel".into()))
        );
        assert!(matches!(
            registry.import("junk.glb", b"nope"),
            Err(ImportError::Model(_))
        ));
        assert_eq!(
            registry.import("...", bytes).err(),
            Some(ImportError::BadName)
        );
    }

    /// The mesh drawn at the body must sit exactly on its collider.
    #[test]
    fn meshes_line_up_with_their_colliders() {
        let registry = ModelRegistry::bundled();
        let id = registry.id("barrel").unwrap();
        let asset = registry.get(id).unwrap();
        let scale = 1.7;

        // Hull: every hull point is a mesh vertex, so the placed mesh must hit
        // each hull vertex exactly (in body space).
        let mut world = World::default();
        let (collider, visual) =
            model_collider(id, asset, scale, ModelCollider::Hull, &mut world).unwrap();
        let Collider::Hull { hull } = collider else {
            panic!("expected a hull");
        };
        let mesh: Vec<Vec3> = asset
            .data
            .positions()
            .map(|p| visual.transform.transform_point3(p))
            .collect();
        for v in world.hull(hull).unwrap().vertices() {
            let nearest = mesh.iter().map(|m| m.distance(*v)).fold(f32::MAX, f32::min);
            assert!(nearest < 1e-4, "hull vertex {v} is {nearest} from the mesh");
        }

        // Box: the placed mesh's bounds are the box.
        let (collider, visual) =
            model_collider(id, asset, scale, ModelCollider::Box, &mut world).unwrap();
        let Collider::Box { half_extents } = collider else {
            panic!("expected a box");
        };
        let placed: Vec<Vec3> = asset
            .data
            .positions()
            .map(|p| visual.transform.transform_point3(p))
            .collect();
        let lo = placed.iter().fold(Vec3::MAX, |a, p| a.min(*p));
        let hi = placed.iter().fold(Vec3::MIN, |a, p| a.max(*p));
        assert!(lo.abs_diff_eq(-half_extents, 1e-4) && hi.abs_diff_eq(half_extents, 1e-4));

        assert_eq!(
            model_collider(id, asset, 0.0, ModelCollider::Box, &mut world).err(),
            Some(ObjectError::InvalidScale)
        );
    }
}
