# impulse M1 — Renderer Minimum + Gravity: Design

**Date:** 2026-09-25
**Status:** Approved (brainstorming session)
**Parent design:** `2026-09-24-impulse-engine-design.md`

## Goal

A lit, shadowed 3D scene of ~30 boxes and spheres falling under gravity
(through the floor; collisions arrive in M2), viewed with an orbit or fly
camera, with an egui control panel. Runs natively and in the browser on
WebGPU and WebGL2.

## Decisions

- **Keep the WebGL2 fallback.** The whole engine targets the WebGL2 feature
  floor (no compute, no storage buffers, 16 KB uniform buffers). Cost is low
  because physics runs on the CPU by design; instance data goes in instanced
  vertex buffers. GPU-driven rendering is out of scope.
- **Camera: orbit by default, fly mode toggled with `F`.** Fly mode on the
  web needs pointer lock (unverified; spike first).
- **Renderer input: a per-frame draw list** (`{mesh, transform, color}` plus
  debug lines). The renderer keeps no scene state. Rejected: a retained
  render scene (duplicates physics state; wrong trade when everything moves)
  and an external ECS/render crate (undercuts "I built the engine").

## Components

### `physics` (adds `glam`)

- `World` with an arena of bodies behind generational handles
  (`BodyHandle { index, generation }`); stale handles return `None`.
- `RigidBody`: position, linear velocity, orientation (constant until M3),
  dynamic-with-mass or static.
- `World::step(dt)`: gravity via **semi-implicit Euler** (velocity, then
  position). No collisions.
- Creation validates inputs (finite values; mass > 0 unless static) and
  returns `Result`.

### `engine`

- **Fixed timestep**: accumulator runs `step(1/60)` up to 5 times per frame
  (spiral-of-death cap); leftover fraction `alpha` interpolates rendered
  transforms between previous and current physics states.
- **Camera**: orbit (drag rotate, wheel/pinch zoom, right/two-finger drag pan,
  clamped pitch and zoom) and fly (WASD + mouse look, pointer lock on web).
- **Input**: collected from winit events into a per-frame input state.
- **egui panel**: pause, single-step, time scale 0.1x–2x, reset scene,
  backend + frame time readout, debug toggles (grid, body axes).
- **Demo scene**: grid floor + ~30 boxes/spheres at staggered heights and
  colors; reset re-drops them.

### `render`

- Depth buffer, recreated on resize (retest native resize-flicker candidates
  here: present inside `Resized`, frame latency 1, DX12 comparison).
- Forward lit pass: Lambert diffuse + Blinn-Phong specular + hemisphere
  ambient.
- One directional light with a **shadow map** (depth pass, comparison
  sampler, PCF).
- Instanced draw per mesh (box, sphere, plane); debug-line pass; egui on top.

## Per-frame flow

input → camera update → accumulate time → `step` × N → interpolate →
build draw list → shadow pass → main pass → debug lines → egui.

## Testing

Unit tests assert contracts, not implementation:

- **physics**: free-fall velocity exact (`v = -g t`); position within
  `g·t·dt` of analytic and error roughly halves when `dt` halves; static
  bodies never move; invalid inputs rejected; removed handle → `None` and a
  reused slot does not revive the old handle; same inputs → bit-identical
  state (same platform).
- **engine**: accumulator step counts, 5-step cap, `alpha ∈ [0, 1)`;
  interpolation at `alpha` 0 / 0.5 / 1; orbit pitch and zoom clamps; fly
  camera moves along its facing direction.
- **render**: draw-list grouping by mesh preserves counts and in-group order.

Visual verification (manual, browser pane visible): screenshots on WebGPU and
`?backend=webgl` showing shadows, grid, egui; orbit/pan/zoom, fly toggle with
pointer lock, pause/step; buffer and depth sizes match the canvas after live
resize; native run to retest resize flicker; high-DPI check on a real device.

## Early spikes (verify before building on them)

1. winit 0.30 web pointer lock + raw mouse deltas.
2. egui-wgpu 0.36 under WebGL2 limits.
3. Comparison-sampler shadow maps on WebGL2.

## Out of scope

Collisions (M2), rotation dynamics (M3), PBR, post-processing, model loading,
GPU-driven rendering, benchmarks (M5).

## Research findings (2026-09-25, verified against source / compile tests)

Spikes — all resolved:
1. **Pointer lock (web):** winit 0.30.13 web supports `CursorGrabMode::Locked` (calls `requestPointerLock`) and emits `DeviceEvent::MouseMotion`. Runtime check still needed: browsers only grant lock inside a user gesture (keydown `F`). Native Windows supports `Confined`, not `Locked` → try Locked, fall back to Confined + hide cursor.
2. **egui:** egui 0.36.2 + egui-wgpu 0.36.2 compile for wasm32 and native (scratch test). **egui-winit 0.36.2 does NOT compile for wasm32** (its `DroppedFile` impl lacks the wasm-only `bytes_async`; default `clipboard` feature also breaks). Decision: drop egui-winit; write a small `egui_input` module translating winit events → `egui::RawInput` (CursorMoved→`PointerMoved(pos/ppp)`, MouseInput→`PointerButton`, MouseWheel→`MouseWheel{unit Line|Point}`, CursorLeft→`PointerGone`, Focused→`WindowFocused`, first touch→pointer events; set `screen_rect` in points, `time`, and `viewports[ROOT].native_pixels_per_point`). No keyboard/text/clipboard (panel has no text fields).
3. **Shadows on WebGL2:** wgpu-hal GLES always sets `DownlevelFlags::COMPARISON_SAMPLERS`; the official shadow example (Depth32Float + `sampler_comparison`) runs on WebGL.

Scope cut: touch = one-finger rotate + pinch zoom only (no two-finger pan).
Deps to add: `glam 0.33.10`, `bytemuck 1.25.2 (derive)`, `egui 0.36.2`, `egui-wgpu 0.36.2`, `web-time 1.1.0` (std `Instant` panics on wasm).

API facts (code from memory will get these wrong):
- **wgpu 30:** `DepthStencilState { depth_write_enabled: Some(true), depth_compare: Some(..), .. }`; `PipelineLayoutDescriptor { bind_group_layouts: &[Some(&bgl)], immediate_size: 0 }`; `RenderPipelineDescriptor { .., multiview_mask: None, cache: None }`; `VertexState/FragmentState { entry_point: Some("..."), compilation_options: Default::default(), .. }` and `VertexState::buffers: &[Option<VertexBufferLayout>]` (wrap each in `Some`); `SamplerDescriptor { mipmap_filter: wgpu::MipmapFilterMode::Nearest, compare: Some(LessEqual), .. }`; `RenderPass::forget_lifetime()` exists; shadow bias example `DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 }`. Reference: `examples/features/src/shadow/{mod.rs,shader.wgsl}` at tag v30.0.1.
- Shadow pass needs its own bind group layout (uniform only): the shadow texture can't be bound while it is the depth attachment. Use `textureSampleCompareLevel` (allowed in non-uniform control flow).
- **egui 0.36:** `ctx.run_ui(raw_input, |ui| ..)` (no `Context::run`); `egui::Window::new(..).show(ui.ctx(), ..)`; `textures_delta.set` is `(TextureId, SmallVec<[ImageDelta;1]>)` → call `update_texture` per delta; `ctx.egui_wants_pointer_input()` / `egui_is_using_pointer()` for input routing; `egui::ClippedPrimitive`/`TexturesDelta` re-exported at root; `ViewportInfo: Default`.
- **egui-wgpu 0.36:** `Renderer::new(&device, format, RendererOptions::default())`; `update_buffers(&device,&queue,&mut encoder,&jobs,&screen)` returns command buffers to submit first; `render(&mut RenderPass<'static>, &jobs, &ScreenDescriptor{size_in_pixels, pixels_per_point})`.
- **glam 0.33:** `Mat4::{look_at_rh, look_to_rh, perspective_rh, orthographic_rh}` are deprecated → `glam::camera::rh::view::{look_at_mat4, look_to_mat4}` and `glam::camera::rh::proj::directx::{perspective, orthographic}` (`directx` = wgpu's NDC: Z [0,1], Y up).
- **naga shader test:** naga is more lenient than Chrome's Tint on uniformity (accepted `textureSampleCompare` in non-uniform flow); browser checks remain necessary.
- WebGL2 budget: ≤16 vertex attributes (plan uses 10: pos, normal + instance mat4, normal-mat3, color), use u16 indices, uniforms ≤16 KB (globals ~176 B), shadow map 2048².

Plan sketch (lean plan still to write): Gpu refactor to `acquire()/present()`; render crate gets meshes (unit cube/sphere/plane with CCW-winding tests), `DrawList` + grouping (normal matrix = inverse-transpose, perpendicularity test), `Renderer` (lit+shadow+lines pipelines, lazy depth resize, growable instance buffers, egui pass); engine gets `FixedTimestep` (tests with dt=1/64 to avoid float ties), `Pose` interpolation, orbit/fly `CameraRig`, `InputState`, `egui_input`, demo scene (auto re-drop when all bodies below y=-30); physics `World`/`RigidBody`/`BodyHandle` + semi-implicit Euler. End with native resize-flicker retest and real high-DPI device check.

## Verification results (2026-09-25)

- Browser pane: WebGPU and `?backend=webgl` both render lit, shadowed scene + grid + egui panel at ~16.7 ms; canvas buffer == CSS size; no console errors. Shadows confirmed on WebGL2.
- Native (Windows/Vulkan, user-checked): visuals, orbit, fly (Confined cursor + raw motion), panel (pause/step/reset/time scale/toggles), auto re-drop all pass.
- **Deferred:** native live-resize edge flicker persists; user accepted for now (candidates in the M0 plan remain untested).
- Pointer lock could not be verified in the embedded browser pane (`WrongDocumentError`, a pane sandbox limit). Check in a real browser after deploy.
- Found during C3 (debug-build only, so the release web build hid them):
  - **egui 0.36 `TexturesDelta` panics in debug if dropped unapplied** → the renderer takes `UiFrame` by value and `drain()`s `set`/`free`.
  - egui warns on sRGB targets → surface configured with the non-sRGB format when supported; the scene renders through an sRGB view, egui through a surface-format `ui_view`.
- Floor plane made double-sided (user: seeing below the floor is useful for editing); orbit is not clamped.
- Release `.wasm` grew to ~9.8 MB raw with egui (measure the transfer size after deploy).
