# M2 — Sphere/Plane Collisions: Design + Lean Plan

**Date:** 2026-09-25 · **Status:** approved · **Parent:** `2026-09-24-impulse-engine-design.md`

## Design

- **Solver: sequential impulses** (Box2D lineage). Rejected: single-pass pairwise
  (jitters with multiple contacts, discarded by M4) and XPBD (different lineage,
  weaker M4 interview story).
- **Colliders live in `physics`:** `BodyDesc` gains `collider: Collider`
  (`Sphere { radius }` | `Plane { normal, offset }`, plane = static only) and
  `material: Material { restitution, friction }`. The engine derives render scale
  from the collider (removes the duplicated `Shape` in `scene.rs`).
- **Narrowphase:** O(n²) sphere–sphere and sphere–plane → `Contact { a, b, normal
  pointing from b to a, depth, point }`. Coincident sphere centers use +Y as the
  normal (no NaN). Broadphase is M5.
- **Step order:** `v += g·dt` → contacts → solve velocities (8 iterations) →
  `x += v·dt` → positional correction (re-detect; push apart 80% of depth beyond a
  1 cm slop, split by inverse mass).
- **Velocity solve:** normal impulse toward a target separating speed
  `-e·vₙ` when approaching faster than 0.5 m/s (restitution threshold, stops resting
  jitter), else 0; Coulomb friction clamped to `μ·jₙ`. **Per-iteration clamping**
  (jₙ ≥ 0 each pass) is deliberate for M2 — M4 switches to accumulated clamping +
  warm starting and shows why.
- **Materials combine:** restitution `max(a, b)`, friction `√(a·b)`.
- **Known limit until M3:** no rotation → friction makes spheres slide to a stop,
  not roll.
- **Demo:** ~30 spheres (varied radius and restitution) dropped into a pit: floor +
  4 wall planes (walls drawn as thin boxes). Reset re-drops.

## Tasks (TDD; clippy native + wasm; commit each)

**P1. Colliders + materials** — `physics/src/{body,collider}.rs`
Validation: sphere radius finite > 0; plane normal finite non-zero (normalized);
dynamic plane → `BodyError::InvalidCollider`; material values finite, ≥ 0.
Tests: each rejection; plane normal normalized; material defaults.

**P2. Narrowphase** — `physics/src/contact.rs`
`sphere_sphere`, `sphere_plane`, `find_contacts(&World) -> Vec<Contact>`.
Tests: overlapping pair → correct normal/depth/point; touching-but-separate → none;
coincident centers → finite +Y normal; sphere below plane surface → depth correct;
static–static pairs skipped.

**P3. Solver + step order** — `physics/src/{solver,world}.rs`
Contract tests: two-body collision conserves momentum; equal masses, head-on,
e = 1 → velocities swap; drop with e = 1 rebounds to ≥ 90% height, e = 0 → rests
(|v| small after 1 s); resting ball: never sinks more than slop + ε and speed stays
≈ 0 over 5 s; sliding ball (μ = 0.5, v₀ = 5 m/s) decelerates at μg (within 5%);
no NaN after 10 s of the demo scene; determinism (bit-identical) still holds.
Mutation-check at least one test (e.g. drop restitution → e=1 test must fail).

**P4. Demo pit** — `engine/src/scene.rs`, `running.rs`
Floor + 4 walls (static planes; walls drawn as thin cubes), 30 spheres (radius
0.3–0.7, restitution 0.1–0.9, small horizontal velocities), colors by restitution.
Test: scene builds 30 dynamic + 5 static bodies; all spheres start inside the pit.

**P5. Verify + ship** — tests, clippy ×2, physics guard; browser pane (visible):
WebGPU **and** WebGL2 screenshots; native debug run (user eyeballs: bounces,
settling, no tunneling through walls). Merge, push, confirm live on both backends.
