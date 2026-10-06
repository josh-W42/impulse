//! The interface between the engine's app shell and a game built on it.

use impulse_render::{CameraData, DrawList, Gpu, Renderer};
use winit::event::{DeviceEvent, WindowEvent};
use winit::window::Window;

use crate::input::InputState;
use crate::platform::DEFAULT_UI_SCALE;

/// What a game gets each call. The shell owns all of it.
pub struct Context<'a> {
    pub gpu: &'a Gpu,
    pub renderer: &'a mut Renderer,
    pub window: &'a Window,
    pub input: &'a InputState,
    pub egui: &'a egui::Context,
    /// Seconds since the last frame, capped at 0.25.
    pub dt: f32,
    /// Seconds since the shell started.
    pub elapsed: f64,
}

impl Context<'_> {
    /// Width over height of the render target.
    pub fn aspect(&self) -> f32 {
        let (width, height) = self.gpu.size();
        width as f32 / height.max(1) as f32
    }

    /// The cursor in normalized device coordinates (x right, y up, both in
    /// [-1, 1]), once it has entered the window.
    pub fn cursor_ndc(&self) -> Option<glam::Vec2> {
        let cursor = self.input.cursor()?;
        let (width, height) = self.gpu.size();
        Some(glam::Vec2::new(
            2.0 * cursor.x / width.max(1) as f32 - 1.0,
            1.0 - 2.0 * cursor.y / height.max(1) as f32,
        ))
    }
}

/// A game run by [`crate::run_game`]. Each frame the shell calls [`Game::update`],
/// then [`Game::ui`] inside the egui pass, then [`Game::draw`].
pub trait Game {
    /// Once per frame, before the UI: input, simulation, uploads.
    fn update(&mut self, cx: &mut Context<'_>);

    /// Inside the egui pass.
    fn ui(&mut self, ui: &mut egui::Ui);

    /// After the UI: fill `list` (already cleared) and return the camera.
    fn draw(&mut self, cx: &mut Context<'_>, list: &mut DrawList) -> CameraData;

    /// Every window event, after egui and the [`InputState`] have seen it,
    /// except key presses egui is typing into. For gestures that must happen
    /// inside the event (browser pointer lock) or click-vs-drag logic.
    fn window_event(&mut self, _cx: &mut Context<'_>, _event: &WindowEvent) {}

    fn device_event(&mut self, _event: &DeviceEvent) {}

    /// UI size multiplier, applied next frame.
    fn ui_scale(&self) -> f32 {
        DEFAULT_UI_SCALE
    }
}
