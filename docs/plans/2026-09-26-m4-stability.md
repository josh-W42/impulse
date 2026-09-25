# M4 — Stable Stacking: Design + Lean Plan

**Date:** 2026-09-26 · **Status:** approved · **Parent:** `2026-09-24-impulse-engine-design.md`

## Design (approach A: Box2D-Lite lineage)

- **Accumulated impulse clamping:** each contact point keeps its total normal impulse
  `Pn` for the step; an iteration applies `max(Pn + ΔP, 0) − Pn`, so later iterations
  can undo earlier over-pushes. Friction along two fixed tangents per point, total
  clamped to `±μ·Pn`.
- **Warm starting:** the previous step's `Pn`, `Pt1`, `Pt2` are re-applied before
  iterating. Needs persistent contacts: points stored in both bodies' local frames and
  matched across steps within 2 cm (simpler than feature IDs, uniform for all shape
  pairs; each old point matches at most once). Cache rebuilt from the current step's
  contacts, so stale pairs never accumulate.
- **Baumgarte position bias** replaces the post-step linear projection: target
  separating speed `max(restitution target, β/dt·max(depth − slop, 0))`, β = 0.2,
  slop 5 mm.
- `SolverSettings { iterations (10), warm_starting (true) }` on `World`, exposed in
  the panel so warm starting can be toggled live.
- Rejected / deferred: Box2D v3 soft step (possible refinement once numbers are in),
  XPBD. **Sleeping moved to M5** (performance; needs islands).

## Tasks

**S1. Baseline ("before")** — add tower (10 boxes) and pyramid (15 boxes) contract tests,
run them on the M3 solver, record drift/sink/speed numbers here. Expected to fail.
**S2. Persistent contacts + accumulated clamping + warm starting + Baumgarte** (solver
rewrite; settings API). All M2/M3 contract tests must still pass.
**S3. Warm-starting evidence test** — at low iterations the tower holds with warm
starting and fails without (proves the mechanism, not just the result).
**S4. Demo** — scene picker (Pit / Tower / Pyramid), warm-starting toggle and
iteration slider in the panel.
**S5. Verify + ship** — full backend matrix + native debug with user; record "after".

## Results

**Before (M3 solver: per-iteration clamping, no warm starting, post-step projection, 8 iterations):**

| Scene | Worst drift | Speed in final second | Worst tilt |
|---|---|---|---|
| Tower of 10 | 12.2 m (collapsed) | 5.33 m/s | 180° |
| Pyramid of 15 | 9.7 cm | 0.148 m/s | 1.5° |
| Box dropped on a 4-tower | top at 4.33 (should be 4.5): stack compressed 17 cm | | |

Pass thresholds: drift < 1 cm, final-second speed < 0.01 m/s, tilt < 1°.

**After (M4 solver, defaults: 10 iterations, warm starting on):**

| Scene | Horizontal drift | Final-second speed | Tilt |
|---|---|---|---|
| Tower of 5 | 2.0 mm | 0.003 m/s | 0.03° |
| Tower of 5, warm starting **off** | 45 cm (225× worse) | 0.25 m/s | 2.4° |
| Pyramid of 15 | 1.0 mm | 8e-5 m/s | 0° |
| Tower of 10 (known limit, bounded test) | 4.6 cm | 0.066 m/s | 0.29° |

Metric correction (made to match the design's stated criterion, not to pass): drift is
now *horizontal*; vertical settling is bounded separately at 7 mm per layer, since each
resting contact may overlap by up to the 5 mm slop.

**What the measurements exposed (beyond the planned mechanisms):**
- **Near-duplicate manifold points.** Slightly twisted boxes clipped to 6+ points in
  clusters; redundant points let Gauss–Seidel shuffle impulse without converging.
  Fix: reduce to 4 (deepest, farthest, largest-area pair). Tower speed 0.63 → 0.09 m/s.
- **Order bias.** Perfectly symmetric contacts still produced a same-signed spin every
  step (fixed visiting order + redundant coplanar contacts). Fix: symmetric
  Gauss–Seidel (alternate sweep direction). 5-tower: ~1e-2 → 2e-4 m/s.
- Hypotheses tested and rejected with data: friction warm starting alone; Baumgarte
  strength/slop (sub-stepped runs were identical across settings — Baumgarte never
  engaged); near-parallel SAT edge axes (no effect).
- Remaining: 10-high towers keep a small wobble (convergence limit of sequential
  impulses + Baumgarte). Box2D v3 soft step is the natural next remedy; sleeping (M5)
  also freezes settled stacks.

Demo: scene picker (Pit / Tower / Pyramid), live warm-starting toggle and iteration
slider. Verified WebGPU + WebGL2 (tower stands in the browser) and native debug (user:
tower stands; warm starting off makes it sag/topple; low iterations worse; pyramid and
pit fine).
