//! Skeletal animation for impulse, headless: sample clips onto a skeleton's
//! pose, blend poses, and turn a pose into the matrices a skinned mesh is
//! drawn with. No rendering or GPU types, so the server or a test can use it.

mod clip;
mod pose;

pub use clip::BoundClip;
pub use pose::{Pose, blend, model_transforms, skinning_matrices};
