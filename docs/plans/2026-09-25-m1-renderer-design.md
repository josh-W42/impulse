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
