//! Engine glue for impulse: window, event loop, and GPU startup on desktop and
//! web, plus the app shell that runs a [`Game`] (the impulse sandbox is one).

// The URL parser is only called by the web build; desktop builds just compile it.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
mod backend;
mod platform;

mod app;

mod camera;
mod editor;
mod editor_ui;
mod egui_input;
mod game;
mod input;
mod joint_scene;
mod models;
mod pose;
mod running;
mod scene;
mod scene_desc;
mod shell;
mod timestep;
pub mod upload;

pub use app::{run, run_game};
pub use camera::{RtsCamera, ThirdPersonCamera, cursor_ray, perspective};
pub use game::{Context, Game};
pub use input::InputState;
pub use platform::DEFAULT_UI_SCALE;
pub use timestep::{FixedTimestep, StepPlan};
// One version of each shared crate for every game.
pub use {
    egui, glam, impulse_anim as anim, impulse_assets as assets, impulse_physics as physics,
    impulse_render as render, winit,
};
