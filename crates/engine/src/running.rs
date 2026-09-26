//! Everything that exists once the GPU is ready: simulation, cameras, input, UI, rendering.

use std::sync::Arc;

use glam::{Vec2, Vec3};
use impulse_physics::{BodyHandle, SolverSettings, StepStats, World};
use impulse_render::{CameraData, DrawList, FrameOutcome, Gpu, Renderer, UiFrame, backend_name};
use web_time::Instant;
use winit::event::{DeviceEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{CursorGrabMode, Window};

use crate::camera::{self, CameraMode, CameraRig, OrbitCamera};
use crate::editor::Editor;
use crate::editor_ui::{self, InspectorAction, NewShape};
use crate::egui_input::EguiInput;
use crate::input::InputState;
use crate::models::{self, ModelMeshes, ModelRegistry};
use crate::platform;
use crate::scene::{self, Demo, SceneObject};
use crate::scene_desc;
use crate::timestep::FixedTimestep;

const FIXED_STEP: f32 = 1.0 / 60.0;
const MAX_STEPS_PER_FRAME: u32 = 5;
/// Bodies below this height count as gone; once all are, the scene re-drops.
const RESET_HEIGHT: f32 = -30.0;
const FLY_SPEED: f32 = 8.0;
/// Farthest an edit drag moves an object from where it started (m).
const MAX_EDIT_DRAG: f32 = 100.0;
/// Drag spring stiffness (1/s²) and damping (1/s, slightly above critical 2·√k).
const DRAG_STIFFNESS: f32 = 80.0;
const DRAG_DAMPING: f32 = 18.0;
/// Caps the drag acceleration (m/s²) so a fast flick can't explode the solver.
const DRAG_MAX_ACCEL: f32 = 200.0;

/// A body grabbed with the mouse, pulled toward the cursor by a spring.
struct Drag {
    body: BodyHandle,
    /// Grab point in the body's local frame.
    local_anchor: Vec3,
    /// Distance from the camera along the cursor ray at grab time.
    distance: f32,
    /// Where the cursor ray currently wants the anchor.
    target: Vec3,
}

/// Edit authors the scene (physics paused); Play simulates it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Edit,
    Play,
}

/// A selected object being dragged in edit mode, on a horizontal plane (or a
/// vertical one facing the camera).
struct EditDrag {
    object: usize,
    /// Where the cursor ray first hit the object.
    grab: Vec3,
    /// The object's authored position at that moment.
    start: Vec3,
    vertical: bool,
}

/// Panel requests carried out after the UI pass.
#[derive(Debug, Clone, PartialEq)]
enum Action {
    Play,
    Stop,
    Capture,
    Undo,
    Redo,
    Add(NewShape),
    AddModel(crate::models::ModelId),
    /// Desktop: import a `.glb` from the `models/` folder and add it.
    ImportFile(usize),
    LoadDemo(Demo),
    Inspector(InspectorAction),
    Save,
    /// Desktop: a scene from the `scenes/` folder.
    LoadSaved(usize),
    /// Browser: open the file picker.
    Open(platform::PickKind),
}

/// Panel-controlled settings plus one-shot requests the panel raises.
struct Settings {
    paused: bool,
    time_scale: f32,
    show_grid: bool,
    show_axes: bool,
    step_requested: bool,
    reset_requested: bool,
    /// Name to save the scene under.
    file_name: String,
    /// Desktop: saved scenes, refreshed when the load list opens.
    saved: Vec<String>,
    /// Desktop: `.glb` files in `models/`, refreshed when the import list opens.
    model_files: Vec<String>,
    show_colliders: bool,
    warm_starting: bool,
    iterations: usize,
    sleeping: bool,
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
    /// Physics work this frame (timings summed over its fixed steps).
    physics: StepStats,
    drag: Option<Drag>,
    hull_meshes: scene::HullMeshes,
    mode: Mode,
    /// The authored scene, selection, and undo history.
    editor: Editor,
    /// Editor revision the world was last built from.
    built_revision: u64,
    edit_drag: Option<EditDrag>,
    /// The demo the authored scene was loaded from, and the editor revision
    /// then (it stays highlighted until the scene is edited).
    demo: Option<(Demo, u64)>,
    /// Last error or status for the panel.
    message: Option<String>,
    /// Ctrl/Shift/etc., from the window (key events alone can miss them).
    modifiers: ModifiersState,
    /// Objects-list filter text.
    list_filter: String,
    /// Bundled and imported models, and their render meshes.
    models: ModelRegistry,
    model_meshes: ModelMeshes,
    /// The selection the objects list last showed (it follows viewport picks).
    listed: Option<usize>,
}

impl Running {
    pub fn new(gpu: Gpu, window: &Window) -> Self {
        let mut world = World::default();
        let initial = platform::scene_override()
            .and_then(|name| Demo::from_name(&name))
            .unwrap_or(Demo::Pit);
        let desc = scene::demo_desc(initial);
        let models = ModelRegistry::bundled();
        let objects =
            scene_desc::rebuild(&mut world, &desc, &models).expect("demo scenes are valid");
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
                file_name: initial.label().to_string(),
                saved: Vec::new(),
                model_files: Vec::new(),
                show_colliders: false,
                warm_starting: SolverSettings::default().warm_starting,
                iterations: SolverSettings::default().iterations,
                sleeping: SolverSettings::default().sleeping,
            },
            start: now,
            last_frame: now,
            frame_ms: 0.0,
            physics: StepStats::default(),
            drag: None,
            hull_meshes: scene::HullMeshes::new(),
            mode: Mode::Play,
            editor: Editor::new(desc),
            built_revision: 0,
            edit_drag: None,
            demo: Some((initial, 0)),
            message: None,
            modifiers: ModifiersState::empty(),
            list_filter: String::new(),
            models,
            model_meshes: ModelMeshes::new(),
            listed: None,
        }
        .with_hull_meshes()
    }

    fn with_hull_meshes(mut self) -> Self {
        self.upload_models();
        self
    }

    /// (Re-)uploads every model's meshes and textures, pinned so scene
    /// rebuilds keep them, then the current scene's hull meshes.
    fn upload_models(&mut self) {
        self.renderer.reset_custom_meshes();
        self.model_meshes = models::upload_models(&mut self.renderer, &self.gpu, &self.models);
        self.renderer.pin_custom_meshes();
        self.sync_hull_meshes();
    }

    /// Re-registers a render mesh for every hull shape in the (rebuilt) world.
    fn sync_hull_meshes(&mut self) {
        self.renderer.clear_custom_meshes();
        self.hull_meshes.clear();
        for id in scene::hulls_in_world(&self.world) {
            if let Some(hull) = self.world.hull(id) {
                let mesh = self.renderer.add_mesh(&self.gpu, &scene::hull_mesh(hull));
                self.hull_meshes.insert(id, mesh);
            }
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
                if *button == MouseButton::Left {
                    if pressed {
                        match self.mode {
                            Mode::Play => self.try_grab(),
                            Mode::Edit => self.try_select(),
                        }
                    } else {
                        self.drag = None; // release: the body keeps its velocity (throw)
                        if self.edit_drag.take().is_some() {
                            self.editor.commit();
                        }
                    }
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
                // Typing into a panel field: the keys are the field's, not the app's.
                if pressed && self.egui_ctx.egui_wants_keyboard_input() {
                    return;
                }
                // Toggle here, inside the key handler: browsers only grant pointer
                // lock during a user gesture.
                if pressed && !event.repeat {
                    let fly = self.camera.mode == CameraMode::Fly;
                    if code == KeyCode::KeyF || (fly && code == KeyCode::Escape) {
                        self.toggle_camera(window);
                    }
                }
                if pressed && self.mode == Mode::Edit {
                    self.edit_shortcut(code, event.repeat);
                }
                self.input.on_key(code, pressed);
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
                self.egui_input.set_modifiers(self.modifiers);
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

    fn view_proj(&self) -> glam::Mat4 {
        let (width, height) = self.gpu.size();
        CameraRig::projection(width as f32 / height.max(1) as f32) * self.camera.view()
    }

    /// World ray under the mouse cursor.
    fn cursor_ray(&self) -> Option<(Vec3, Vec3)> {
        let cursor = self.input.cursor()?;
        let (width, height) = self.gpu.size();
        let ndc = Vec2::new(
            2.0 * cursor.x / width.max(1) as f32 - 1.0,
            1.0 - 2.0 * cursor.y / height.max(1) as f32,
        );
        Some(camera::cursor_ray(self.view_proj(), ndc))
    }

    /// Left click in orbit mode: grab the dynamic body under the cursor, if any.
    fn try_grab(&mut self) {
        if self.camera.mode != CameraMode::Orbit || self.egui_ctx.egui_wants_pointer_input() {
            return;
        }
        let Some((origin, dir)) = self.cursor_ray() else {
            return;
        };
        let Some(hit) = self.world.raycast(origin, dir, 500.0) else {
            return;
        };
        let Some(body) = self
            .world
            .body(hit.body)
            .filter(|b| b.inverse_mass() != 0.0)
        else {
            return;
        };
        self.drag = Some(Drag {
            body: hit.body,
            local_anchor: body.orientation().inverse() * (hit.point - body.position()),
            distance: hit.t,
            target: hit.point,
        });
    }

    /// Left click in edit mode: select the object under the cursor and start
    /// dragging it (Shift: vertically); empty space clears the selection.
    fn try_select(&mut self) {
        if self.camera.mode != CameraMode::Orbit || self.egui_ctx.egui_wants_pointer_input() {
            return;
        }
        let Some((origin, dir)) = self.cursor_ray() else {
            return;
        };
        match editor_ui::object_at(&self.world, &self.objects, origin, dir) {
            Some((index, point)) => {
                self.editor.select(Some(index));
                let start = Vec3::from_array(self.editor.desc().objects[index].position);
                let vertical = self.modifiers.shift_key();
                self.editor.begin();
                self.edit_drag = Some(EditDrag {
                    object: index,
                    grab: point,
                    start,
                    vertical,
                });
            }
            None => self.editor.select(None),
        }
    }

    /// Moves the dragged object to follow the cursor on its drag plane.
    fn update_edit_drag(&mut self) {
        let Some(drag) = &self.edit_drag else {
            return;
        };
        let Some((origin, dir)) = self.cursor_ray() else {
            return;
        };
        let normal = if drag.vertical {
            let mut facing = self.camera.eye() - drag.grab;
            facing.y = 0.0;
            facing.try_normalize().unwrap_or(Vec3::Z)
        } else {
            Vec3::Y
        };
        let Some(hit) = editor_ui::ray_plane(origin, dir, drag.grab, normal) else {
            return;
        };
        let mut delta = (hit - drag.grab).clamp_length_max(MAX_EDIT_DRAG);
        if drag.vertical {
            delta = Vec3::new(0.0, delta.y, 0.0);
        }
        let position = (drag.start + delta).to_array();
        let index = drag.object;
        if self
            .editor
            .desc()
            .objects
            .get(index)
            .is_some_and(|o| o.position != position)
        {
            self.editor.modify(|d| d.objects[index].position = position);
        }
    }

    /// Delete/Backspace, Ctrl+D, Ctrl+Z, Ctrl+Y (or Ctrl+Shift+Z); Cmd on macOS.
    fn edit_shortcut(&mut self, code: KeyCode, repeat: bool) {
        let ctrl = self.modifiers.control_key() || self.modifiers.super_key();
        let shift = self.modifiers.shift_key();
        match code {
            KeyCode::Delete | KeyCode::Backspace if !repeat => {
                self.editor.delete_selected();
            }
            KeyCode::KeyD if ctrl && !repeat => {
                self.editor.duplicate_selected();
            }
            KeyCode::KeyZ if ctrl && shift => {
                self.editor.redo();
            }
            KeyCode::KeyZ if ctrl => {
                self.editor.undo();
            }
            KeyCode::KeyY if ctrl => {
                self.editor.redo();
            }
            _ => {}
        }
    }

    /// Rebuilds the world from the authored scene (keeping the old world and
    /// reporting why if the scene is invalid).
    fn rebuild_scene(&mut self) {
        match scene_desc::rebuild(&mut self.world, self.editor.desc(), &self.models) {
            Ok(objects) => {
                self.objects = objects;
                self.sync_hull_meshes();
                self.timestep.reset();
                self.alpha = 0.0;
                self.drag = None;
                self.message = None;
            }
            Err(e) => self.message = Some(e.to_string()),
        }
        self.built_revision = self.editor.revision();
    }

    fn apply(&mut self, action: Action) {
        match action {
            Action::Play => {
                self.editor.commit();
                self.edit_drag = None;
                self.mode = Mode::Play;
                self.settings.paused = false;
                self.rebuild_scene();
            }
            Action::Stop => {
                self.mode = Mode::Edit;
                self.rebuild_scene();
            }
            Action::Capture => {
                let poses: Vec<_> = self
                    .objects
                    .iter()
                    .map(|o| scene::current_pose(&self.world, o.handle))
                    .map(|p| p.map(|p| (p.position, p.orientation)))
                    .collect();
                let captured = self.editor.capture_poses(&poses);
                self.message = Some(format!(
                    "captured {captured} poses (jointed bodies keep their authored pose)"
                ));
            }
            Action::Undo => {
                self.editor.undo();
            }
            Action::Redo => {
                self.editor.redo();
            }
            Action::Add(shape) => {
                // On top of whatever is under the camera target (0.5 m shapes).
                let target = self.camera.orbit.target;
                let above = Vec3::new(target.x, 100.0, target.z);
                let surface = self
                    .world
                    .raycast(above, Vec3::NEG_Y, 200.0)
                    .map_or(0.0, |hit| hit.point.y);
                let at = Vec3::new(target.x, surface + 0.6, target.z);
                let index = self.editor.desc().objects.len();
                self.editor.add(editor_ui::new_object(shape, at, index));
            }
            Action::LoadDemo(demo) => {
                self.editor.replace(scene::demo_desc(demo));
                self.demo = Some((demo, self.editor.revision()));
                self.settings.file_name = demo.label().to_string();
                self.rebuild_scene();
            }
            Action::Save => {
                self.editor.commit();
                let Some(stem) = scene_desc::file_stem(&self.settings.file_name) else {
                    self.message = Some("give the scene a name (letters, digits, - or _)".into());
                    return;
                };
                let mut desc = self.editor.desc().clone();
                desc.name = stem.clone();
                self.message = Some(
                    match platform::save_scene(&stem, &scene_desc::to_ron(&desc)) {
                        Ok(place) => format!("saved {place}"),
                        Err(e) => e,
                    },
                );
            }
            Action::LoadSaved(index) => {
                let Some(stem) = self.settings.saved.get(index).cloned() else {
                    return;
                };
                match platform::read_scene(&stem) {
                    Ok(text) => self.load_scene_text(&stem, &text),
                    Err(e) => self.message = Some(e),
                }
            }
            Action::Open(kind) => platform::pick_file(kind),
            Action::AddModel(id) => {
                let target = self.camera.orbit.target;
                let above = Vec3::new(target.x, 100.0, target.z);
                let floor = self
                    .world
                    .raycast(above, Vec3::NEG_Y, 200.0)
                    .map_or(0.0, |hit| hit.point.y);
                let index = self.editor.desc().objects.len();
                if let Some(object) =
                    editor_ui::new_model_object(&self.models, id, target, floor, index)
                {
                    self.editor.add(object);
                }
            }
            Action::ImportFile(index) => {
                let Some(stem) = self.settings.model_files.get(index).cloned() else {
                    return;
                };
                match platform::read_model(&stem) {
                    Ok(bytes) => {
                        if let Some(id) = self.import_model(&format!("{stem}.glb"), &bytes) {
                            self.apply(Action::AddModel(id));
                        }
                    }
                    Err(e) => self.message = Some(e),
                }
            }
            Action::Inspector(InspectorAction::Duplicate) => {
                self.editor.duplicate_selected();
            }
            Action::Inspector(InspectorAction::Delete) => {
                self.editor.delete_selected();
            }
        }
    }

    /// Adds a model to the registry and uploads it; scenes using that name are
    /// rebuilt with the new mesh.
    fn import_model(&mut self, file_name: &str, bytes: &[u8]) -> Option<crate::models::ModelId> {
        match self.models.import(file_name, bytes) {
            Ok((id, name)) => {
                self.upload_models();
                let used = self.editor.desc().objects.iter().any(|o| {
                    matches!(&o.shape, Some(scene_desc::ShapeDesc::Model { name: n, .. }) if *n == name)
                });
                if used {
                    self.rebuild_scene();
                }
                self.message = Some(format!("imported '{name}'"));
                Some(id)
            }
            Err(e) => {
                self.message = Some(format!("{file_name}: {e}"));
                None
            }
        }
    }

    /// Replaces the authored scene with a scene file's contents, if it parses
    /// and builds; otherwise reports why and keeps the current scene.
    fn load_scene_text(&mut self, source: &str, text: &str) {
        let checked = scene_desc::from_ron(text)
            .and_then(|desc| scene_desc::build_world(&desc, &self.models).map(|_| desc));
        match checked {
            Ok(desc) => {
                self.settings.file_name = desc.name.clone();
                self.editor.replace(desc);
                self.demo = None;
                self.rebuild_scene();
                self.message = Some(format!("loaded {source}"));
            }
            Err(e) => self.message = Some(format!("{source}: {e}")),
        }
    }

    /// Spring-damper impulse pulling the grab point toward the cursor (gravity
    /// cancelled so the body doesn't sag). Called before every fixed step.
    fn apply_drag(&mut self, dt: f32) {
        let Some(drag) = &self.drag else {
            return;
        };
        let Some(body) = self.world.body(drag.body) else {
            self.drag = None; // removed (e.g. reset)
            return;
        };
        let r = body.orientation() * drag.local_anchor;
        let anchor = body.position() + r;
        let anchor_velocity = body.velocity() + body.angular_velocity().cross(r);
        let accel = (DRAG_STIFFNESS * (drag.target - anchor)
            - DRAG_DAMPING * anchor_velocity
            - self.world.gravity())
        .clamp_length_max(DRAG_MAX_ACCEL);
        let impulse = accel * dt / body.inverse_mass();
        let body = drag.body;
        self.world.apply_impulse_at(body, impulse, anchor);
    }

    fn update_camera(&mut self, dt: f32, egui_has_pointer: bool) {
        let dragging = self.drag.is_some() || self.edit_drag.is_some();
        let input = &self.input;
        match self.camera.mode {
            CameraMode::Orbit if !egui_has_pointer => {
                let orbit = &mut self.camera.orbit;
                let drag = input.cursor_delta();
                if input.button_held(MouseButton::Left) && !dragging {
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
        let solver = SolverSettings {
            iterations: self.settings.iterations,
            warm_starting: self.settings.warm_starting,
            sleeping: self.settings.sleeping,
        };
        if self.world.solver_settings() != solver {
            self.world.set_solver_settings(solver);
        }
        if self.mode == Mode::Edit {
            self.settings.reset_requested = false;
            self.settings.step_requested = false;
            if self.editor.revision() != self.built_revision {
                self.rebuild_scene();
            }
            return;
        }
        let fell =
            !self.objects.is_empty() && scene::all_below(&self.world, &self.objects, RESET_HEIGHT);
        if self.settings.reset_requested || fell {
            self.settings.reset_requested = false;
            self.rebuild_scene();
        }
        let settings = &mut self.settings;
        let steps = if settings.paused {
            if settings.step_requested {
                self.alpha = 1.0;
                1
            } else {
                0
            }
        } else {
            let plan = self.timestep.advance(dt * settings.time_scale);
            self.alpha = plan.alpha;
            plan.steps
        };
        settings.step_requested = false;

        let start = self.start;
        let mut clock = || start.elapsed().as_secs_f64() * 1000.0;
        let mut frame = StepStats::default();
        for _ in 0..steps {
            self.apply_drag(self.timestep.step());
            scene::snapshot(&self.world, &mut self.objects);
            let s = self.world.step_profiled(self.timestep.step(), &mut clock);
            frame = StepStats {
                broadphase_ms: frame.broadphase_ms + s.broadphase_ms,
                narrowphase_ms: frame.narrowphase_ms + s.narrowphase_ms,
                solve_ms: frame.solve_ms + s.solve_ms,
                integrate_ms: frame.integrate_ms + s.integrate_ms,
                ..s
            };
        }
        if steps > 0 {
            self.physics = frame;
        }
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
        self.update_edit_drag();
        match platform::take_picked() {
            Some(Ok(picked)) => match picked.kind {
                platform::PickKind::Scene => match String::from_utf8(picked.bytes) {
                    Ok(text) => self.load_scene_text(&picked.name, &text),
                    Err(_) => self.message = Some(format!("{} is not text", picked.name)),
                },
                platform::PickKind::Model => {
                    self.import_model(&picked.name, &picked.bytes);
                }
            },
            Some(Err(e)) => self.message = Some(e),
            None => {}
        }
        if let Some((origin, dir)) = self.cursor_ray()
            && let Some(drag) = &mut self.drag
        {
            drag.target = origin + dir * drag.distance;
        }
        self.step_physics(dt);
        scene::build_draw_list(
            &mut self.draw_list,
            &self.world,
            &self.objects,
            &self.hull_meshes,
            &self.model_meshes,
            scene::DrawOptions {
                arena: self.editor.desc().arena,
                selected: match self.mode {
                    Mode::Edit => self.editor.selected(),
                    Mode::Play => None,
                },
                show_colliders: self.settings.show_colliders,
                alpha: self.alpha,
                show_grid: self.settings.show_grid,
                show_axes: self.settings.show_axes,
            },
        );
        if let Some(drag) = &self.drag
            && let Some(body) = self.world.body(drag.body)
        {
            let anchor = body.position() + body.orientation() * drag.local_anchor;
            self.draw_list
                .push_line(anchor, drag.target, [1.0, 0.85, 0.2, 1.0]);
        }

        let (width, height) = self.gpu.size();
        let raw = self
            .egui_input
            .take([width, height], self.start.elapsed().as_secs_f64());
        let stats = Stats {
            backend: self.backend,
            frame_ms: self.frame_ms,
            bodies: self.world.len(),
            camera: self.camera.mode,
            physics: self.physics,
            mode: self.mode,
            demo: self
                .demo
                .filter(|&(_, revision)| revision == self.editor.revision())
                .map(|(demo, _)| demo),
            can_undo: self.editor.can_undo(),
            can_redo: self.editor.can_redo(),
            message: self.message.clone(),
            models: self
                .models
                .list()
                .map(|(id, name, _)| (id, name.to_string()))
                .collect(),
        };
        let mut actions = Vec::new();
        let follow = self.editor.selected() != self.listed;
        let (settings, editor, mode, filter) = (
            &mut self.settings,
            &mut self.editor,
            self.mode,
            &mut self.list_filter,
        );
        let output = self.egui_ctx.run_ui(raw, |ui| {
            panel(ui, settings, &stats, &mut actions);
            if mode == Mode::Edit {
                editor_ui::outliner(ui.ctx(), editor, filter, follow);
                let names: Vec<&str> = stats.models.iter().map(|(_, n)| n.as_str()).collect();
                if let Some(action) = editor_ui::inspector(ui.ctx(), editor, &names) {
                    actions.push(Action::Inspector(action));
                }
            }
        });
        self.listed = self.editor.selected();
        for action in actions {
            self.apply(action);
        }
        // A gesture ends when nothing is being dragged or typed into.
        if self.edit_drag.is_none()
            && !self.egui_ctx.egui_is_using_pointer()
            && self.egui_ctx.memory(|m| m.focused().is_none())
        {
            self.editor.commit();
        }
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
    camera: CameraMode,
    physics: StepStats,
    mode: Mode,
    demo: Option<Demo>,
    can_undo: bool,
    can_redo: bool,
    message: Option<String>,
    /// Every model's id and name, for the Model… picker.
    models: Vec<(crate::models::ModelId, String)>,
}

fn panel(ui: &mut egui::Ui, s: &mut Settings, stats: &Stats, actions: &mut Vec<Action>) {
    egui::Window::new("impulse")
        .default_pos([12.0, 12.0])
        .resizable(false)
        .show(ui.ctx(), |ui| {
            ui.horizontal(|ui| match stats.mode {
                Mode::Play => {
                    if ui.button("■ Edit").clicked() {
                        actions.push(Action::Stop);
                    }
                    if ui.button("Capture pose").clicked() {
                        actions.push(Action::Capture);
                    }
                }
                Mode::Edit => {
                    if ui.button("▶ Play").clicked() {
                        actions.push(Action::Play);
                    }
                    if ui
                        .add_enabled(stats.can_undo, egui::Button::new("Undo"))
                        .clicked()
                    {
                        actions.push(Action::Undo);
                    }
                    if ui
                        .add_enabled(stats.can_redo, egui::Button::new("Redo"))
                        .clicked()
                    {
                        actions.push(Action::Redo);
                    }
                }
            });
            if stats.mode == Mode::Edit {
                ui.horizontal(|ui| {
                    ui.label("add");
                    for shape in NewShape::ALL {
                        if ui.button(shape.label()).clicked() {
                            actions.push(Action::Add(shape));
                        }
                    }
                    // Loaded models, then (desktop) files in models/ not yet
                    // imported: picking one imports and adds it.
                    let list = egui::ComboBox::from_id_salt("add model")
                        .selected_text("Model…")
                        .show_ui(ui, |ui| {
                            for (id, name) in &stats.models {
                                if ui.selectable_label(false, name).clicked() {
                                    actions.push(Action::AddModel(*id));
                                }
                            }
                            if platform::HAS_SCENE_FOLDER {
                                ui.separator();
                                let loaded = |stem: &str| {
                                    scene_desc::file_stem(stem)
                                        .is_some_and(|n| stats.models.iter().any(|(_, m)| *m == n))
                                };
                                let mut any = false;
                                for (i, file) in s.model_files.iter().enumerate() {
                                    if loaded(file) {
                                        continue;
                                    }
                                    any = true;
                                    let label = format!("{file}.glb (models/)");
                                    if ui.selectable_label(false, label).clicked() {
                                        actions.push(Action::ImportFile(i));
                                    }
                                }
                                if !any {
                                    ui.label("more: put .glb files in models/");
                                }
                            }
                        });
                    if platform::HAS_SCENE_FOLDER && list.response.clicked() {
                        s.model_files = platform::model_files();
                    }
                    if !platform::HAS_SCENE_FOLDER && ui.button("Import…").clicked() {
                        actions.push(Action::Open(platform::PickKind::Model));
                    }
                });
            }
            ui.horizontal(|ui| {
                ui.label("scene");
                for demo in Demo::ALL {
                    if ui
                        .selectable_label(stats.demo == Some(demo), demo.label())
                        .clicked()
                    {
                        actions.push(Action::LoadDemo(demo));
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("file");
                ui.add(egui::TextEdit::singleline(&mut s.file_name).desired_width(110.0));
                if ui.button("Save").clicked() {
                    actions.push(Action::Save);
                }
                if platform::HAS_SCENE_FOLDER {
                    let list = egui::ComboBox::from_id_salt("load")
                        .selected_text("Load…")
                        .show_ui(ui, |ui| {
                            if s.saved.is_empty() {
                                ui.label("no saved scenes yet");
                            }
                            for (i, name) in s.saved.iter().enumerate() {
                                if ui.selectable_label(false, name).clicked() {
                                    actions.push(Action::LoadSaved(i));
                                }
                            }
                        });
                    if list.response.clicked() {
                        s.saved = platform::saved_scenes();
                    }
                } else if ui.button("Open…").clicked() {
                    actions.push(Action::Open(platform::PickKind::Scene));
                }
            });
            if let Some(message) = &stats.message {
                ui.colored_label(egui::Color32::from_rgb(240, 190, 90), message);
            }
            ui.separator();
            ui.label(format!("{} · {:.1} ms", stats.backend, stats.frame_ms));
            let p = &stats.physics;
            ui.label(format!(
                "{} bodies · {} awake · {} asleep",
                stats.bodies, p.awake, p.asleep
            ));
            ui.label(format!(
                "physics {:.2} ms: broad {:.2} · narrow {:.2} · solve {:.2} · integrate {:.2}",
                p.total_ms(),
                p.broadphase_ms,
                p.narrowphase_ms,
                p.solve_ms,
                p.integrate_ms
            ));
            ui.label(format!(
                "{} pairs · {} contacts ({} solved)",
                p.pairs, p.contacts, p.active_contacts
            ));
            ui.add_enabled_ui(stats.mode == Mode::Play, |ui| {
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
            });
            ui.separator();
            ui.checkbox(&mut s.warm_starting, "warm starting");
            ui.checkbox(&mut s.sleeping, "sleeping");
            ui.add(egui::Slider::new(&mut s.iterations, 1..=30).text("solver iterations"));
            ui.separator();
            ui.checkbox(&mut s.show_grid, "grid");
            ui.checkbox(&mut s.show_axes, "body axes");
            ui.checkbox(&mut s.show_colliders, "model colliders");
            ui.label(match (stats.camera, stats.mode) {
                (CameraMode::Orbit, Mode::Play) => {
                    "Drag a body: grab / throw · drag empty space: orbit · right-drag: pan · wheel: zoom · F: fly"
                },
                (CameraMode::Orbit, Mode::Edit) => {
                    "Click: select · drag it: move (Shift: up/down) · Del · Ctrl+D · Ctrl+Z / Ctrl+Y · drag empty space: orbit"
                },
                (CameraMode::Fly, _) => "WASD / Space / Shift: move · mouse: look · F or Esc: orbit",
            });
        });
}
