//! Engine glue for impulse: window, event loop, and GPU startup on desktop and web.

// The URL parser is only called by the web build; desktop builds just compile it.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
mod backend;
mod platform;

mod app;

mod camera;
mod editor;
mod editor_ui;
mod egui_input;
mod input;
mod joint_scene;
mod models;
mod pose;
mod running;
mod scene;
mod scene_desc;
mod timestep;

pub use app::run;
