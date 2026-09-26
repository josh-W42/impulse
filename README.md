# impulse

A 3D game engine in Rust + wgpu whose deep spike is rigid-body physics.
Runs natively and in the browser (WebGPU, falling back to WebGL2).

Live demo: https://impulse-engine.web.app (scenes: `?scene=pit|tower|pyramid|rain|joints|props`)

Design: `docs/plans/2026-09-24-impulse-engine-design.md` (one plan per milestone
in `docs/plans/`, each with its results). Performance: `BENCHMARKS.md`.

## Physics (written from scratch; depends only on `glam`)

- Sequential-impulse solver: accumulated clamping, warm starting, Baumgarte
  correction, symmetric Gauss–Seidel, friction
- Shapes: sphere, box, capsule, convex hull, plane; box–box SAT with clipped
  manifolds, GJK + EPA for general convex pairs, manifolds reduced to 4 points
- Joints: ball-socket (3×3 block) with cone limits, hinge with angle limits,
  distance; soft (spring-regularized) rows so closed loops stay stable; joined
  bodies skip collision and share a sleep island
- Scale: dynamic AABB-tree broadphase, island sleeping, raycasts, per-phase profiler
- Engine: fixed timestep with interpolation, instanced renderer, egui panel,
  mouse grab/throw

## Editor

Press **■ Edit** to author the scene; **▶ Play** simulates it and **■ Edit** returns
to exactly what you authored.

- Click to select; drag to move it across the floor (Shift: up/down); the inspector
  edits shape, pose, body, material, colour, and starting velocity
- Add sphere/box/capsule, Ctrl+D duplicate, Del delete, Ctrl+Z / Ctrl+Y undo/redo
- **Capture pose** (in Play) writes the simulated poses back as the starting scene
- Scenes are RON files: desktop saves to `scenes/`, the browser downloads and
  opens them. The demos are available as starting points.
- **Models** (glTF `.glb`): Add → Model… places a bundled CC0 prop
  (`assets/models/`, see its `LICENSES.md`); import your own from `models/` on
  desktop or with Import… in the browser. Each collides as a convex hull or a
  fitted box/sphere/capsule; the "model colliders" toggle shows them.

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
- `crates/engine`: winit loop, platform glue
- `crates/demo`: the demo binary
- `web/`: HTML shell for the browser build
