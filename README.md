# impulse

A 3D game engine in Rust + wgpu whose deep spike is rigid-body physics.
Runs natively and in the browser (WebGPU, falling back to WebGL2).

Live demo: https://impulse-engine.web.app (scenes: `?scene=pit|tower|pyramid|rain|joints|props|materials`)

Design: `docs/plans/2026-09-24-impulse-engine-design.md` (one plan per milestone
in `docs/plans/`, each with its results). Performance: `BENCHMARKS.md`.

## Physics (written from scratch; depends only on `glam`)

- Two solvers, switchable live in the panel:
  - **soft step** (default, after Box2D v3): 4 substeps, soft (spring-damper)
    contacts, a relax pass each substep, restitution in its own pass. A 10-high
    box tower stands still (5.5 mm drift in 10 s)
  - **classic** sequential impulses: accumulated clamping, warm starting,
    Baumgarte correction, symmetric Gauss–Seidel, friction
- Shapes: sphere, box, capsule, convex hull, plane; box–box SAT with clipped
  manifolds, GJK + EPA for general convex pairs, manifolds reduced to 4 points
- Joints: ball-socket (3×3 block) with cone limits, hinge with angle limits,
  distance; soft (spring-regularized) rows so closed loops stay stable; joined
  bodies skip collision and share a sleep island
- Scale: dynamic AABB-tree broadphase, island sleeping, raycasts, per-phase profiler
- Rendering: physically based (glTF metallic-roughness with normal, occlusion,
  and emissive maps), a procedural sky as image-based light and background,
  Khronos PBR Neutral tonemapping, PCF shadows; WebGPU with a WebGL2 fallback
- Engine: fixed timestep with interpolation, instanced renderer, egui panel,
  mouse grab/throw

## Editor

Press **■ Edit** to author the scene; **▶ Play** simulates it and **■ Edit** returns
to exactly what you authored.

- Click to select; drag the selection to move it across the floor (Shift: up/down),
  drag elsewhere to orbit; the inspector
  edits shape, pose, body, material, colour, and starting velocity
- Add sphere/box/capsule, Ctrl+D duplicate, Del delete, Ctrl+Z / Ctrl+Y undo/redo
- **Hierarchy:** the objects list is a tree. Drag a row onto another to parent it
  (it stays put in the world), or onto "(top level)" to unparent; Add → Empty makes a
  folder. Moving, duplicating, or deleting a parent brings its children (and the joints
  among them) along. Parenting is for editing: at Play every body is independent.
- **UI size** (A−/A+ in the panel) scales the interface, 75–200%; the browser starts at 125%
- **New** starts a blank scene with a finite floor object (a static box you can
  resize, move, or delete; things fall off its edges). In Edit mode, **arena** adds
  built-in surroundings to any scene: none, an infinite floor, or a walled pit
- **Capture pose** (in Play) writes the simulated poses back as the starting scene
- Scenes are RON files: desktop saves to `scenes/`, the browser downloads and
  opens them. The demos are available as starting points.
- **Models** (glTF `.glb`): Add → Model… places a bundled CC0 prop
  (`assets/models/`, see its `LICENSES.md`). Your own: on desktop, `.glb` files in
  `models/` (next to where you run it) are listed in Model… too, and picking one
  imports and adds it; in the browser, use Import…. Each collides as a convex hull or a
  fitted box/sphere/capsule; the "model colliders" toggle shows them.

## Writing a game on impulse

`impulse-engine` runs any type that implements `Game`. The shell owns the
window, GPU, renderer, egui and input, and calls three methods each frame:

```rust
use impulse_engine::{Context, Game};
use impulse_engine::render::{CameraData, DrawList};

impl Game for MyGame {
    fn update(&mut self, cx: &mut Context<'_>) { /* input, simulation, uploads */ }
    fn ui(&mut self, ui: &mut impulse_engine::egui::Ui) { /* panels */ }
    fn draw(&mut self, cx: &mut Context<'_>, list: &mut DrawList) -> CameraData { /* … */ }
}

fn main() {
    impulse_engine::run_game("my game", |cx| Box::new(MyGame::new(cx)));
}
```

`Context` gives access to the `Gpu` and `Renderer` (for uploading meshes,
textures and materials), the `InputState`, the egui context, and the frame
time. The optional `window_event` hook is for gestures that must happen inside
an event, such as browser pointer lock. The engine re-exports `egui`, `glam`,
`winit` and the renderer (`impulse_engine::render`) so a game uses the same
versions. Also available: `RtsCamera` (strategy-game camera with default
controls), `perspective`, `cursor_ray`, `FixedTimestep`, and
`Renderer::set_shadow_focus` for scenes larger than the default 25 m shadow
radius. The smallest example: `cargo run -p impulse-engine --example spinning_cube`.
The sandbox (`crates/engine/src/running.rs`) is itself a `Game`.

## Prerequisites

- rustup (the pinned toolchain in `rust-toolchain.toml` installs automatically)
- wasm-bindgen CLI matching `Cargo.lock`:
  `cargo install --locked wasm-bindgen-cli --version "$(scripts/wasm-bindgen-version.sh)"`
- Python 3 (for serving the web build locally)

## Run

Desktop: `cargo run -p impulse-demo`

Browser:

    scripts/build-web.sh
    python -m http.server 8080 --directory web/dist

Open http://localhost:8080. Add `?backend=webgl` to force the WebGL2 fallback.

## Checks

    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo clippy --workspace --target wasm32-unknown-unknown -- -D warnings
    cargo test --workspace
    scripts/check-physics-deps.sh

## Layout

- `crates/physics`: rigid-body physics; depends on `glam` only (enforced)
- `crates/render`: wgpu renderer; no windowing
- `crates/engine`: app shell (`Game` trait), winit loop, platform glue, and the sandbox editor
- `crates/demo`: the demo binary
- `web/`: HTML shell for the browser build
