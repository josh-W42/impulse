//! Engine glue for impulse: window, event loop, and GPU startup on desktop and web.

// The URL parser is only called by the web build; desktop builds just compile it.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
mod backend;
mod platform;

mod app;

// Pure-logic modules for M1; `allow(dead_code)` is removed when they are wired in (plan task C1).
#[allow(dead_code)]
mod camera;
#[allow(dead_code)]
mod egui_input;
#[allow(dead_code)]
mod input;
#[allow(dead_code)]
mod pose;
#[allow(dead_code)]
mod timestep;

pub use app::run;
