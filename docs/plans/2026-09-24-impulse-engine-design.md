# impulse — Design

**Date:** 2026-09-24
**Status:** Approved (brainstorming session)

## Purpose

A 3D game engine built as a **portfolio / interview piece aimed at the game and
graphics industry**. The engine's deliberate depth ("spike") is **rigid-body
physics**; the renderer and engine layers exist to make the physics visible and
to demonstrate modern GPU-API fluency.

Success criteria:

- A live, clickable browser demo at every milestone (reviewers will not download
  binaries).
- A physics crate deep enough to discuss for 30+ minutes: stable stacking, warm
  starting, broadphase performance, GJK/EPA, joints.
- Numbers behind performance claims (benchmarks, per-phase profiler).

Timeline: open-ended; every milestone ships a deployed demo.

## Stack

**Rust + wgpu**, single codebase targeting:

- **Native** (Vulkan / DX12 / Metal via wgpu) — for development and profiling.
- **Browser** (`wasm32`, WebGPU with WebGL2 fallback) — for the portfolio demo.

Rationale: least toolchain friction for native + wasm, so effort goes into
physics rather than CMake/Emscripten. wgpu follows the WebGPU model (pipelines,
bind groups, command encoders), which maps directly onto Vulkan/D3D12/Metal
concepts graphics interviewers probe for.

Alternatives considered:

- **C++20 + WebGPU (Dawn / Emscripten)** — strongest industry signal, but heavy
  toolchain friction and a still-moving `webgpu.h` API. Rejected: user prefers
  not to use C++.
- **Rust physics + Three.js viewer** — fastest to demo, but forfeits the "own
  engine" claim (no own renderer or loop). Rejected.

## Architecture

Cargo workspace, strictly one-directional dependencies:

```
demo (bin)  ──►  engine  ──►  render  (wgpu)
                   │
                   └──────►  physics (glam only)
```

### `physics` — the spike

- Standalone library: depends on `glam` only. No wgpu, winit, or engine types.
- Owns `World`, rigid bodies, colliders, broadphase, narrowphase, solver.
- Bodies stored in an arena, referenced by **generational handles**
  (`BodyHandle { index, generation }`) — no pointers or `Rc<RefCell>`.
- Exposes debug data (contacts, AABBs, broadphase nodes) as plain line/point
  data, so it never depends on the renderer.
- Testable and benchmarkable headless.

### `render` — deliberately modest

- One forward lit pass + one directional shadow map.
- **Instanced drawing** (1,000 bodies = one draw call per mesh type).
- Debug-line pass.
- Primitive meshes: box, sphere, capsule, plane.
- No PBR, post-processing, or model loading unless later needed.

### `engine` — glue

- **Fixed-timestep loop**: physics at exactly 60 Hz via an accumulator; render
  interpolates between previous and current physics states.
- Input, orbit/fly camera, egui panels (pause, single-step, time scale, debug
  view toggles).
- Scene = plain list of `{ BodyHandle, mesh, material }`. **No general ECS**
  (not the spike; YAGNI).

### `demo` — binary

- Scene picker: box stack, pyramid, dominoes, 1,000-body pile, joint scenes.
- Builds native and wasm; `web/` holds the HTML shell.

### Per-frame data flow

input → accumulate time → `world.step(1/60)` × N → interpolate transforms →
upload instance buffer → draw scene → draw debug lines → draw egui.

## Milestones

Each milestone ends with a deployed live demo.

| # | Milestone | Proves | Demo |
|---|---|---|---|
| M0 | Toolchain spike: workspace; wgpu clears screen natively **and** in browser (WebGPU + WebGL2 fallback); deployed | Riskiest unknowns resolved first: compatible wgpu/winit/egui versions, wasm build, hosting | Colored canvas at a live URL |
| M1 | Renderer minimum (camera, instancing, lighting, shadow map, debug lines, egui) + gravity-only integration | Engine loop, fixed timestep, interpolation | Objects fall (through the floor) |
| M2 | Sphere/plane contacts; impulse resolution with restitution + friction | Core impulse physics | Balls bouncing in a pit |
| M3 | Rotation + boxes: inertia tensors, quaternion integration, SAT box–box with contact-manifold clipping | Real rigid-body dynamics | Tumbling boxes |
| M4 | **Stability**: sequential-impulse solver, accumulated-impulse clamping, warm starting (persistent contact IDs), sub-stepping / soft constraints, sleeping | Engine vs. toy | 10-high stack and pyramid that settle and stay |
| M5 | **Scale**: dynamic AABB tree broadphase, per-phase profiler panel, written benchmarks | Performance engineering with evidence | 1,000-body pile at 60 fps in a browser tab |
| M6 | Generic convex: GJK + EPA, capsules, convex hulls, raycasts, mouse-drag picking | Advanced collision algorithms | Throw arbitrary shapes |
| M7 | Joints: ball-socket, hinge, distance | Constraint generalization | Rope bridge, chain, simple ragdoll |

After M7 (engine usability, chosen 2026-09-26):

| # | Milestone | Status |
|---|---|---|
| M8 | RON scene format + edit/play editor, save/load (`2026-09-26-m8-scene-editor.md`); plus a flat objects list | Done |
| M9 | glTF models + textures; mesh-derived colliders (fitted primitives, or a hull from sampled extreme points) as a new object kind | Next |
| M10? | **Scene hierarchy** (possible): a tree in the objects list. Needs a design pass first: groups vs. real parenting (do children move with parents? world or local poses in the file?), a format version bump with migration, drag-to-reparent, and how joints and physics interact with nested transforms (physics bodies stay flat). | Backlog |

Stretch (only if wanted):

- Continuous collision detection (speculative contacts).
- Multithreaded island solving — wasm threads require cross-origin isolation
  (COOP/COEP headers) on hosting.
- Bit-identical native↔wasm determinism — plausible but **unverified**;
  transcendental float functions can differ per platform and may need a pure
  `libm`-style implementation. Verify before promising.

### References

- Erin Catto — GDC talks and Box2D solver blog posts (sequential impulses, soft step).
- Christer Ericson — *Real-Time Collision Detection* (SAT, GJK, BVH).
- Dirk Gregorius — GDC talks on contact manifolds.
- Ian Millington — *Game Physics Engine Development* (on-ramp).

## Testing

1. **Known-answer unit tests** — inertia tensors vs. textbook formulas; SAT/GJK
   against hand-built configurations; quaternion integration vs. analytic
   rotation.
2. **Physical-invariant tests** (behavior, not implementation) — momentum
   conservation in two-body collisions; kinetic-energy conservation (within
   tolerance) for elastic, gravity-free collisions; resting body stays at rest.
3. **Scene regression tests** — run each demo scene headless for N seconds and
   assert outcomes: top box of 10-stack moves < 1 cm; no NaN after 10k steps;
   identical inputs → bit-identical state hash on the same platform.
4. **Benchmarks** (M5+) recorded in `BENCHMARKS.md`, using the measurement
   discipline from the telemetry-viewer project.

Not tested automatically: rendered pixels (manual verification in browser).

## Error handling

- **Physics**: invalid inputs (non-finite mass, zero-size shapes) rejected with
  `Result` at creation, never panic mid-step. Stale handles → `None` via the
  generation check. NaN guards as `debug_assert`s inside `step`.
- **Engine / web**: fall back to WebGL2 when WebGPU is unavailable; if neither
  works, show a clear message instead of a blank canvas. Reconfigure the surface
  on loss/resize. Route wasm panics to the browser console.

## CI / Deployment

GitHub Actions (zephyr pattern): `cargo fmt --check`, `clippy`, tests, wasm
build, deploy to Firebase Hosting on `main`.

## Dependencies (verify before install; pin exact versions in M0)

`wgpu`, `winit`, `glam`, `egui` + `egui-wgpu` + `egui-winit`, `wasm-bindgen`,
`console_error_panic_hook`, `bytemuck`; optional `criterion` for benchmarks.
egui ↔ wgpu version compatibility is the most likely early snag.

## Out of scope (YAGNI)

Pixel-diff render tests, asset/model loader, audio, scripting, scene editor,
general-purpose ECS.
