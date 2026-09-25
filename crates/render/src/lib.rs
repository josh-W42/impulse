//! GPU context and frame rendering for the impulse engine.
//!
//! Independent of windowing: the engine creates the surface from its window and
//! hands it over. Re-exports `wgpu` so every crate uses the same version.

pub use wgpu;

mod size;

pub use size::clamp_surface_size;
