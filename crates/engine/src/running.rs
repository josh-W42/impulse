//! Everything that exists once the GPU is ready: simulation, cameras, input, UI, rendering.

use std::sync::Arc;

use glam::{Vec2, Vec3};
use impulse_physics::World;
use impulse_render::{CameraData, DrawList, FrameOutcome, Gpu, Renderer, UiFrame, backend_name};
use web_time::Instant;
use winit::event::{DeviceEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window};

use crate::camera::{CameraMode, CameraRig, OrbitCamera};
use crate::egui_input::EguiInput;
use crate::input::InputState;
use crate::scene::{self, SceneObject};
use crate::timestep::FixedTimestep;

const FIXED_STEP: f32 = 1.0 / 60.0;
const MAX_STEPS_PER_FRAME: u32 = 5;
/// Bodies below this height count as gone; once all are, the scene re-drops.
const RESET_HEIGHT: f32 = -30.0;
const FLY_SPEED: f32 = 8.0;

/// Panel-controlled settings plus one-shot requests the panel raises.
struct Settings {
    paused: bool,
    time_scale: f32,
    show_grid: bool,
    show_axes: bool,
    step_requested: bool,
    reset_requested: bool,
}

pub enum FrameResult {
    Continue,
    Failed(String),
}

pub struct Running {
    gpu: Gpu,
    renderer: Renderer,
    world: World,
    objects: Vec<SceneObject>,
    timestep: FixedTimestep,
    alpha: f32,
    camera: CameraRig,
    input: InputState,
    egui_ctx: egui::Context,
    egui_input: EguiInput,
    draw_list: DrawList,
    settings: Settings,
    backend: &'static str,
    start: Instant,
    last_frame: Instant,
    frame_ms: f32,
}

impl Running {
    pub fn new(gpu: Gpu, window: &Window) -> Self {
        let mut world = World::default();
        let objects = scene::demo_scene(&mut world);
        let now = Instant::now();
        Self {
            renderer: Renderer::new(&gpu),
            backend: backend_name(gpu.backend()),
            gpu,
            world,
            objects,
            timestep: FixedTimestep::new(FIXED_STEP, MAX_STEPS_PER_FRAME),
            alpha: 0.0,
            camera: CameraRig::new(OrbitCamera {
                target: Vec3::new(0.0, 2.0, 0.0),
                yaw: 0.6,
                pitch: 0.35,
                distance: 28.0,
            }),
            input: InputState::new(),
            egui_ctx: egui::Context::default(),
            egui_input: EguiInput::new(window.scale_factor() as f32),
            draw_list: DrawList::default(),
            settings: Settings {
                paused: false,
                time_scale: 1.0,
                show_grid: true,
                show_axes: false,
                step_requested: false,
                reset_requested: false,
            },
            start: now,
            last_frame: now,
            frame_ms: 0.0,
        }
    }

    pub fn backend(&self) -> &'static str {
        self.backend
    }

    /// Routes one window event to egui, the input state, and the GPU.
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
                // Toggle here, inside the key handler: browsers only grant pointer
                // lock during a user gesture.
                if pressed && !event.repeat {
                    let fly = self.camera.mode == CameraMode::Fly;
                    if code == KeyCode::KeyF || (fly && code == KeyCode::Escape) {
                        self.toggle_camera(window);
                    }
                }
                self.input.on_key(code, pressed);
            }
            WindowEvent::Focused(focused) => {
                self.egui_input.on_focus(*focused);
                if !focused {
                    self.input.release_all();
                    if self.camera.mode == CameraMode::Fly {
                        self.toggle_camera(window);
                    }
                }
            }
            _ => {}
        }
    }

    pub fn on_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event
            && self.camera.mode == CameraMode::Fly
        {
            self.input
                .on_raw_motion(Vec2::new(delta.0 as f32, delta.1 as f32));
        }
    }

    fn toggle_camera(&mut self, window: &Window) {
        self.camera.toggle();
        let fly = self.camera.mode == CameraMode::Fly;
        let grab = if fly {
            // Web and macOS support Locked; Windows and X11 only Confined.
            window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
        } else {
            window.set_cursor_grab(CursorGrabMode::None)
        };
        if let Err(e) = grab {
            log::warn!("cursor grab failed: {e}");
        }
        window.set_cursor_visible(!fly);
    }

    fn update_camera(&mut self, dt: f32, egui_has_pointer: bool) {
        let input = &self.input;
        match self.camera.mode {
            CameraMode::Orbit if !egui_has_pointer => {
                let orbit = &mut self.camera.orbit;
                let drag = input.cursor_delta();
                if input.button_held(MouseButton::Left) {
                    orbit.rotate(-drag.x * 0.005, drag.y * 0.005);
                }
                if input.button_held(MouseButton::Right) || input.button_held(MouseButton::Middle) {
                    orbit.pan(-drag.x * 0.0015, drag.y * 0.0015);
                }
                if input.wheel() != 0.0 {
                    orbit.zoom(0.9f32.powf(input.wheel()));
                }
                let touch = input.touch_rotate();
                orbit.rotate(-touch.x * 0.005, touch.y * 0.005);
                if input.touch_pinch() != 1.0 {
                    orbit.zoom(1.0 / input.touch_pinch());
                }
            }
            CameraMode::Orbit => {}
            CameraMode::Fly => {
                let fly = &mut self.camera.fly;
                let look = input.raw_delta();
                fly.look(look.x * 0.002, look.y * 0.002);
                let axis = |pos, neg| {
                    input.key_held(pos) as i32 as f32 - input.key_held(neg) as i32 as f32
                };
                let step = FLY_SPEED * dt;
                fly.translate(
                    axis(KeyCode::KeyW, KeyCode::KeyS) * step,
                    axis(KeyCode::KeyD, KeyCode::KeyA) * step,
                    axis(KeyCode::Space, KeyCode::ShiftLeft) * step,
                );
            }
        }
    }

    fn step_physics(&mut self, dt: f32) {
        let settings = &mut self.settings;
        if settings.reset_requested || scene::all_below(&self.world, &self.objects, RESET_HEIGHT) {
            settings.reset_requested = false;
            self.objects = scene::demo_scene(&mut self.world);
            self.timestep.reset();
            self.alpha = 0.0;
        }
        if settings.paused {
            if settings.step_requested {
                scene::snapshot(&self.world, &mut self.objects);
                self.world.step(self.timestep.step());
                self.alpha = 1.0;
            }
        } else {
            let plan = self.timestep.advance(dt * settings.time_scale);
            for _ in 0..plan.steps {
                scene::snapshot(&self.world, &mut self.objects);
                self.world.step(self.timestep.step());
            }
            self.alpha = plan.alpha;
        }
        settings.step_requested = false;
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
        self.frame_ms = self.frame_ms * 0.9 + dt * 1000.0 * 0.1;

        let egui_has_pointer =
            self.egui_ctx.egui_wants_pointer_input() || self.egui_ctx.egui_is_using_pointer();
        self.update_camera(dt, egui_has_pointer);
        self.step_physics(dt);
        scene::build_draw_list(
            &mut self.draw_list,
            &self.world,
            &self.objects,
            self.alpha,
            self.settings.show_grid,
            self.settings.show_axes,
        );

        let (width, height) = self.gpu.size();
        let raw = self
            .egui_input
            .take([width, height], self.start.elapsed().as_secs_f64());
        let (settings, stats) = (
            &mut self.settings,
            Stats {
                backend: self.backend,
                frame_ms: self.frame_ms,
                bodies: self.world.len(),
                mode: self.camera.mode,
            },
        );
        let output = self.egui_ctx.run_ui(raw, |ui| panel(ui, settings, &stats));
        let ui = UiFrame {
            primitives: self
                .egui_ctx
                .tessellate(output.shapes, output.pixels_per_point),
            textures_delta: output.textures_delta,
            pixels_per_point: output.pixels_per_point,
        };

        let aspect = width as f32 / height.max(1) as f32;
        let camera = CameraData {
            view_proj: CameraRig::projection(aspect) * self.camera.view(),
            eye: self.camera.eye(),
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

struct Stats {
    backend: &'static str,
    frame_ms: f32,
    bodies: usize,
    mode: CameraMode,
}

fn panel(ui: &mut egui::Ui, s: &mut Settings, stats: &Stats) {
    egui::Window::new("impulse")
        .default_pos([12.0, 12.0])
        .resizable(false)
        .show(ui.ctx(), |ui| {
            ui.label(format!("{} · {:.1} ms", stats.backend, stats.frame_ms));
            ui.label(format!("{} bodies", stats.bodies));
            ui.horizontal(|ui| {
                if ui
                    .button(if s.paused { "Resume" } else { "Pause" })
                    .clicked()
                {
                    s.paused = !s.paused;
                }
                if ui
                    .add_enabled(s.paused, egui::Button::new("Step"))
                    .clicked()
                {
                    s.step_requested = true;
                }
                if ui.button("Reset").clicked() {
                    s.reset_requested = true;
                }
            });
            ui.add(egui::Slider::new(&mut s.time_scale, 0.1..=2.0).text("time scale"));
            ui.checkbox(&mut s.show_grid, "grid");
            ui.checkbox(&mut s.show_axes, "body axes");
            ui.label(match stats.mode {
                CameraMode::Orbit => "Drag: orbit · right-drag: pan · wheel: zoom · F: fly",
                CameraMode::Fly => "WASD / Space / Shift: move · mouse: look · F or Esc: orbit",
            });
        });
}
