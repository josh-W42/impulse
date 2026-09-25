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
mod shaders;

pub use draw_list::{DebugLine, DrawList, Instance, InstanceRaw, group_by_mesh};
pub use gpu::{Frame, FrameOutcome, Gpu, GpuInitError};
pub use mesh::{MeshData, MeshKind, Vertex};
