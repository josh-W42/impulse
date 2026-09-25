# M3 — Rotation + Boxes: Design + Lean Plan

**Date:** 2026-09-26 · **Status:** approved · **Parent:** `2026-09-24-impulse-engine-design.md`
Shipped as two deployable halves: **M3a** (rotation, rolling, box–plane, sphere–box)
and **M3b** (box–box SAT + clipping).

## Design

- **Box–box: SAT + Sutherland–Hodgman clipping** (15 axes; least-penetration axis
  with a small bias toward face axes; clip incident face against reference face side
  planes → up to 4+ points). Rejected: GJK/EPA (single point, needs persistent
  manifolds — M6) and vertex-only tests (miss edge–edge, boxes snag).
- **Rotation state:** angular velocity + diagonal local inverse inertia from the
  collider (sphere `2/5·m·r²`; box with half-extents `h`: `Iₓ = m/3·(h_y²+h_z²)` etc.).
  World inverse inertia `R·I⁻¹·Rᵀ`. Orientation: `q += ½·(ω,0)·q·dt`, normalize.
  No gyroscopic term. Collider-less bodies don't rotate.
- **Solver:** contact velocity `v + ω×r`; effective mass
  `k = Σ 1/m + n·Σ (I⁻¹(r×n))×r`; impulses change `v` and `ω` → spheres roll,
  boxes tumble. Friction likewise along the tangent.
- **Positional correction:** linear, **per body pair using the deepest point**
  (multi-point manifolds would otherwise over-correct ×N).
- **Known limit until M4:** stacks creep/jitter (per-iteration clamping). M4 fixes it
  with accumulated clamping + warm starting.
- **M3a demo:** boxes + spheres in the pit (boxes may pass through each other until
  M3b). **M3b demo:** boxes collide with each other.

## M3a tasks (TDD; clippy native + wasm; commit each)

**R1. Rotation state + box collider** — `Collider::Box { half_extents }` (finite, > 0);
angular velocity (+ `with_angular_velocity`); inverse inertia from collider; world
inverse inertia; quaternion integration in `advance`.
Tests: sphere and box inertia match formulas; static/collider-less → no rotation;
torque-free spin about a principal axis turns `ω·t` radians and conserves ω;
orientation stays unit length over 10k steps.

**R2. Angular solver terms** — contact velocity at the point, effective mass, angular
impulses (normal + friction); correction per pair (deepest point).
Tests: ball given v₀ on a floor ends rolling without slipping at `≈ 5/7·v₀` with
`v = ω·r`; free off-center collision conserves linear **and** angular momentum;
off-center hit induces spin; all M2 contract tests still pass.

**R3. Box–plane** — every corner below the plane → a contact.
Tests: flat box → 4 contacts; corner-down box → 1; box dropped flat rests flat
(tilt < 1°, |v|,|ω| small); tilted box topples and settles on a face (some body axis
within 5° of vertical).

**R4. Sphere–box** — closest point in box space; center-inside uses the nearest face.
Tests: face, edge, and corner contacts (normal/depth); center inside → pushes out
through the nearest face.

**R5. Demo** — pit with boxes and spheres (render box scale = 2·half_extents,
orientation interpolated). Scene tests updated.

**R6. Verify + ship** — full backend matrix (pane visible) + native debug with user.

## M3b tasks (planned after M3a ships)

SAT axis selection (face–face, edge–edge, rotated cases, parallel-edge guard) →
reference/incident face clipping (point counts and positions) → box stacks tumble
and settle without passing through each other → demo + verify + ship.

## M3a results (2026-09-25)

- 88 workspace tests. New: inertia formulas, torque-free spin, unit quaternions,
  rolling without slipping (5/7·v₀, v = ω·r), angular-momentum conservation +
  friction-induced spin, box–plane (4 / 1 contacts), sphere–box (face/edge/corner/
  inside/rotated), box rests flat, tilted box topples onto a face, ball rests on a
  box / rolls off its edge. Mutation: removing angular impulses fails 2 tests.
- Two test premises corrected, not the physics: M2's "decelerates at μg" now holds
  only during slipping (ball rolls after ≈0.29 s); "ball rolls off a box it lands
  on" was wrong for a flat top — replaced by "rests on top" + "rolls off the edge".
- Verified: WebGPU + WebGL2 (pane visible, buffer = CSS, no errors) and native debug
  (user: spheres roll, boxes tumble/topple/settle, spheres rest on boxes).
- As designed: boxes pass through each other until M3b.
