//! Models available to scenes: the bundled library (compiled in, so it loads
//! synchronously and identically everywhere) plus any imported this session.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use glam::{Mat4, Vec3};
use impulse_assets::{MaterialData, ModelData, ModelError, extreme_points, load_glb, mip_chain};
use impulse_physics::{Collider, World};
use impulse_render::{Gpu, MeshData, MeshId, Renderer, TextureId, Vertex};

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

/// Uploads every model in `registry`. Identical textures (e.g. a kit's shared
/// palette) are uploaded once. The material's base colour factor is baked into
/// its texture, or into a 1×1 texture for untextured materials.
pub fn upload_models(renderer: &mut Renderer, gpu: &Gpu, registry: &ModelRegistry) -> ModelMeshes {
    let mut textures: HashMap<u64, TextureId> = HashMap::new();
    let mut out = ModelMeshes::new();
    for (id, _, _) in registry.list() {
        let data = &registry.get(id).expect("listed").data;
        let meshes = data
            .meshes
            .iter()
            .map(|mesh| {
                let material = mesh.material.and_then(|m| data.materials.get(m));
                let texture = material_texture(renderer, gpu, data, material, &mut textures);
                let mesh_data = MeshData {
                    vertices: mesh
                        .vertices
                        .iter()
                        .map(|v| Vertex {
                            position: v.position,
                            normal: v.normal,
                            uv: v.uv,
                        })
                        .collect(),
                    indices: mesh.indices.clone(),
                };
                renderer.add_textured_mesh(gpu, &mesh_data, texture)
            })
            .collect();
        out.insert(id, meshes);
    }
    out
}

fn material_texture(
    renderer: &mut Renderer,
    gpu: &Gpu,
    data: &ModelData,
    material: Option<&MaterialData>,
    uploaded: &mut HashMap<u64, TextureId>,
) -> TextureId {
    let factor = material.map_or([1.0; 4], |m| m.base_color);
    let texture = material
        .and_then(|m| m.base_color_texture)
        .and_then(|t| data.textures.get(t));
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    factor.map(f32::to_bits).hash(&mut hasher);
    if let Some(texture) = texture {
        let base = &texture.mips[0];
        (base.width, base.height, &base.rgba).hash(&mut hasher);
    }
    let key = hasher.finish();
    if let Some(&id) = uploaded.get(&key) {
        return id;
    }
    let id = match texture {
        Some(texture) if factor == [1.0; 4] => {
            let levels: Vec<&[u8]> = texture.mips.iter().map(|m| m.rgba.as_slice()).collect();
            renderer.add_texture(gpu, texture.mips[0].width, texture.mips[0].height, &levels)
        }
        Some(texture) => {
            let base = &texture.mips[0];
            let tinted = tint(&base.rgba, factor);
            let mips = mip_chain(tinted, base.width, base.height, true);
            let levels: Vec<&[u8]> = mips.iter().map(|m| m.rgba.as_slice()).collect();
            renderer.add_texture(gpu, base.width, base.height, &levels)
        }
        None => renderer.add_texture(gpu, 1, 1, &[&tint(&[255; 4], factor)]),
    };
    uploaded.insert(key, id);
    id
}

/// Multiplies sRGB RGBA8 texels by a linear RGBA factor.
fn tint(rgba: &[u8], factor: [f32; 4]) -> Vec<u8> {
    let linear = |c: u8| {
        let c = c as f32 / 255.0;
        if c <= 0.040_45 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let srgb = |c: f32| {
        let c = if c <= 0.003_130_8 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (c.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    rgba.as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| {
            [
                srgb(linear(p[0]) * factor[0]),
                srgb(linear(p[1]) * factor[1]),
                srgb(linear(p[2]) * factor[2]),
                (p[3] as f32 * factor[3]).round().clamp(0.0, 255.0) as u8,
            ]
        })
        .collect()
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
    fn tint_multiplies_in_linear_light() {
        assert_eq!(
            tint(&[255, 255, 255, 255], [1.0, 0.5, 0.0, 1.0]),
            [255, 188, 0, 255]
        );
        assert_eq!(tint(&[188, 10, 0, 200], [1.0; 4]), [188, 10, 0, 200]);
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
