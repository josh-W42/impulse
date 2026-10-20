//! GPU context and frame rendering for the impulse engine.
//!
//! Independent of windowing: the engine creates the surface from its window and
//! hands it over. Re-exports `wgpu` so every crate uses the same version.

pub use wgpu;

mod size;

pub use size::clamp_surface_size;

mod backend;

pub use backend::backend_name;

mod draw_list;
mod gpu;
mod mesh;
mod renderer;
mod shaders;
mod sky;

pub use draw_list::{
    DEFAULT_METALLIC, DEFAULT_ROUGHNESS, DebugLine, DrawList, Instance, InstanceRaw,
    JOINT_TEXTURE_WIDTH, JointRange, SkinnedInstance, SkinnedInstanceRaw, group_by_mesh,
    group_skinned, joint_texels, joint_texture_rows,
};
pub use gpu::{Frame, FrameOutcome, Gpu, GpuInitError};
pub use mesh::{
    MeshData, MeshId, MeshKind, SkinnedMeshData, SkinnedMeshId, SkinnedVertex, Vertex, cylinder,
    perpendicular_tangent, uv_sphere,
};
pub use renderer::{CameraData, MaterialDesc, MaterialId, Renderer, TextureId, UiFrame};
pub use sky::{Sky, eval_sh, project_irradiance, sh_basis};
