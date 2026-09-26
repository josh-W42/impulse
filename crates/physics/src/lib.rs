//! Rigid-body physics for the impulse engine.
//!
//! Architecture rule: this crate depends on `glam` only, never on rendering,
//! windowing, or engine crates, so it can be tested, benchmarked, and reused
//! headless. `scripts/check-physics-deps.sh` enforces this in CI.

mod body;
mod box_box;
mod broadphase;
mod capsule;
mod collider;
mod contact;
mod gjk;
mod hull;
mod polytope;
mod raycast;
mod solver;
mod union_find;
mod world;

pub use body::{BodyDesc, BodyError, BodyKind, RigidBody};
pub use broadphase::Aabb;
pub use collider::{Collider, HullId, Material};
pub use contact::{Contact, find_contacts};
pub use hull::{ConvexHull, HullError, HullFace, MAX_HULL_POINTS};
pub use raycast::RayHit;
pub use solver::SolverSettings;
pub use world::{BodyHandle, DEFAULT_GRAVITY, StepStats, World};
