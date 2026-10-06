//! Everything a game needs once the GPU is ready: the renderer, input, egui,
//! and the frame loop. The game itself sits behind [`Game`].

use std::sync::Arc;

use glam::Vec2;
use impulse_render::{DrawList, FrameOutcome, Gpu, Renderer, UiFrame, backend_name};
use web_time::Instant;
use winit::event::{DeviceEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::PhysicalKey;
use winit::window::Window;

use crate::egui_input::EguiInput;
use crate::game::{Context, Game};
use crate::input::InputState;

/// Builds the game once the shell is ready.
pub type MakeGame = Box<dyn FnOnce(&mut Context<'_>) -> Box<dyn Game>>;

pub enum FrameResult {
    Continue,
    Failed(String),
}

pub struct Shell {
    gpu: Gpu,
    renderer: Renderer,
    input: InputState,
    egui_ctx: egui::Context,
    egui_input: EguiInput,
    draw_list: DrawList,
    start: Instant,
    last_frame: Instant,
    game: Box<dyn Game>,
}

/// Borrows the shell's parts as a [`Context`], leaving the game free to borrow.
macro_rules! context {
    ($shell:expr, $window:expr, $dt:expr) => {
        Context {
            gpu: &$shell.gpu,
            renderer: &mut $shell.renderer,
            window: $window,
            input: &$shell.input,
            egui: &$shell.egui_ctx,
            dt: $dt,
            elapsed: $shell.start.elapsed().as_secs_f64(),
        }
    };
}

impl Shell {
    pub fn new(gpu: Gpu, window: &Window, make: MakeGame) -> Self {
        let mut renderer = Renderer::new(&gpu);
        let input = InputState::new();
        let egui_ctx = egui::Context::default();
        // UI size is the game's (see EguiInput::set_ui_scale); egui's own
        // Ctrl +/- zoom would desync pointer positions from the layout.
        egui_ctx.options_mut(|o| o.zoom_with_keyboard = false);
        let now = Instant::now();
        let game = make(&mut Context {
            gpu: &gpu,
            renderer: &mut renderer,
            window,
            input: &input,
            egui: &egui_ctx,
            dt: 0.0,
            elapsed: 0.0,
        });
        Self {
            egui_input: EguiInput::new(window.scale_factor() as f32),
            gpu,
            renderer,
            input,
            egui_ctx,
            draw_list: DrawList::default(),
            start: now,
            last_frame: now,
            game,
        }
    }

    pub fn backend(&self) -> &'static str {
        backend_name(self.gpu.backend())
    }

    /// Routes one window event to egui, the input state, the GPU, and the game.
    pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) {
        match event {
            WindowEvent::Resized(size) => self.gpu.resize(size.width, size.height),
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.egui_input.set_pixels_per_point(*scale_factor as f32);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.egui_input.on_cursor_moved((position.x, position.y));
                self.input
                    .on_cursor_moved(Vec2::new(position.x as f32, position.y as f32));
            }
            WindowEvent::CursorLeft { .. } => {
                self.egui_input.on_cursor_left();
                self.input.on_cursor_left();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state.is_pressed();
                let egui_button = match button {
                    MouseButton::Left => Some(egui::PointerButton::Primary),
                    MouseButton::Right => Some(egui::PointerButton::Secondary),
                    MouseButton::Middle => Some(egui::PointerButton::Middle),
                    _ => None,
                };
                if let Some(b) = egui_button {
                    self.egui_input.on_mouse_button(b, pressed);
                }
                self.input.on_mouse_button(*button, pressed);
            }
            WindowEvent::MouseWheel { delta, .. } => match *delta {
                MouseScrollDelta::LineDelta(x, y) => {
                    self.egui_input.on_wheel_lines(x, y);
                    self.input.on_wheel(y);
                }
                MouseScrollDelta::PixelDelta(p) => {
                    self.egui_input.on_wheel_pixels(p.x, p.y);
                    self.input.on_wheel(p.y as f32 / 50.0);
                }
            },
            WindowEvent::Touch(touch) => {
                let phase = match touch.phase {
                    winit::event::TouchPhase::Started => egui::TouchPhase::Start,
                    winit::event::TouchPhase::Moved => egui::TouchPhase::Move,
                    winit::event::TouchPhase::Ended => egui::TouchPhase::End,
                    winit::event::TouchPhase::Cancelled => egui::TouchPhase::Cancel,
                };
                let (x, y) = (touch.location.x, touch.location.y);
                self.egui_input.on_touch(touch.id, phase, (x, y));
                self.input
                    .on_touch(touch.id, touch.phase, Vec2::new(x as f32, y as f32));
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                let pressed = event.state.is_pressed();
                self.egui_input
                    .on_key(code, pressed, event.repeat, event.text.as_deref());
                // Typing into a panel field: the keys are the field's, not the game's.
                if pressed && self.egui_ctx.egui_wants_keyboard_input() {
                    return;
                }
                self.input.on_key(code, pressed);
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.egui_input.set_modifiers(modifiers.state());
            }
            WindowEvent::Focused(focused) => {
                self.egui_input.on_focus(*focused);
                if !focused {
                    self.input.release_all();
                }
            }
            _ => {}
        }
        self.game
            .window_event(&mut context!(self, window, 0.0), event);
    }

    pub fn on_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.input
                .on_raw_motion(Vec2::new(delta.0 as f32, delta.1 as f32));
        }
        self.game.device_event(event);
    }

    /// Runs one frame. Skipped frames leave input queued for the next one.
    pub fn frame(&mut self, window: &Arc<Window>) -> FrameResult {
        let frame = match self.gpu.acquire() {
            Ok(frame) => frame,
            Err(FrameOutcome::SurfaceLost) => {
                return match self.gpu.instance().create_surface(window.clone()) {
                    Ok(surface) => {
                        self.gpu.replace_surface(surface);
                        FrameResult::Continue
                    }
                    Err(e) => FrameResult::Failed(format!("Lost the rendering surface: {e}")),
                };
            }
            Err(_) => return FrameResult::Continue,
        };

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.25);
        self.last_frame = now;

        self.game.update(&mut context!(self, window, dt));

        let (width, height) = self.gpu.size();
        self.egui_input.set_ui_scale(self.game.ui_scale());
        let raw = self
            .egui_input
            .take([width, height], self.start.elapsed().as_secs_f64());
        let game = &mut self.game;
        let output = self.egui_ctx.run_ui(raw, |ui| game.ui(ui));

        self.draw_list.clear();
        let camera = self
            .game
            .draw(&mut context!(self, window, dt), &mut self.draw_list);

        let ui = UiFrame {
            primitives: self
                .egui_ctx
                .tessellate(output.shapes, output.pixels_per_point),
            textures_delta: output.textures_delta,
            pixels_per_point: output.pixels_per_point,
        };
        self.renderer.render(
            &self.gpu,
            &frame.view,
            &frame.ui_view,
            &camera,
            &self.draw_list,
            ui,
        );
        window.pre_present_notify();
        self.gpu.present(frame);
        self.input.end_frame();
        FrameResult::Continue
    }
}
