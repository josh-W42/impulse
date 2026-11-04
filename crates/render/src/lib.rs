//! GPU context and frame rendering for the impulse engine.
//!
//! Independent of windowing: the engine creates the surface from its window and
//! hands it over. Re-exports `wgpu` so every crate uses the same version.

pub use wgpu;

mod size;

pub use size::clamp_surface_size;

mod backend;

pub use backend::backend_name;

mod ao;
mod cull;
mod draw_list;
mod environment;
mod gpu;
mod lights;
mod mesh;
mod post;
mod renderer;
mod shaders;
mod shadow;
mod sky;

pub use ao::AoSettings;
pub use cull::{Bounds, DrawStats, LOD_PIXELS};
pub use draw_list::{
    DEFAULT_METALLIC, DEFAULT_ROUGHNESS, DebugLine, DrawList, Instance, InstanceRaw,
    JOINT_TEXTURE_WIDTH, JointRange, SkinnedInstance, SkinnedInstanceRaw, group_by_mesh,
    group_skinned, joint_texels, joint_texture_rows,
};
pub use environment::{ENV_FACE, ENV_MIPS, Environment};
pub use gpu::{Frame, FrameOutcome, Gpu, GpuInitError, SceneTarget};
pub use lights::{Light, LightKind, LightStats, MAX_LIGHTS, MAX_SPOT_SHADOWS};
pub use mesh::{
    MAX_LODS, MeshData, MeshId, MeshKind, MeshLod, SkinnedMeshData, SkinnedMeshId, SkinnedVertex,
    Vertex, cylinder, lod_error, perpendicular_tangent, uv_sphere,
};
pub use post::PostSettings;
pub use renderer::{
    AlphaMode, CameraData, CullSettings, FogSettings, MaterialDesc, MaterialId, Renderer,
    TextureId, UiFrame,
};
pub use shadow::ShadowRange;
pub use sky::{Sky, eval_sh, project_irradiance, sh_basis};
