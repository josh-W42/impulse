# M13 — Gameplay Physics: Design + Results

**Date:** 2026-10-19 · **Status:** complete · **Driven by:** Ashen Crown
(`../ashen-crown/docs/plans/2026-10-19-design.md`), a tab-target online RPG
whose characters, Charge and Blink need these pieces.

## Why

Games need more from the physics than a simulation of falling things. A
character is a capsule moved by code, not a tumbling rigid body; a melee swing
or a teleport asks "what would this shape hit?"; an arrow ignores its shooter.
The world had only two body kinds (dynamic and static), one way to move a body
after creation (`apply_impulse_at`), no collision filtering, and raycasts as
its only query.

## What shipped

- **Kinematic bodies** (`BodyKind::Kinematic`, `BodyDesc::kinematic`). Moved
  only by their own velocity; infinite mass, so they push dynamic bodies and
  are never pushed back; no gravity; no contacts with static or other
  kinematic bodies. The solver already read every body's velocity and scaled
  impulses by inverse mass, so a body with zero inverse mass and a non-zero
  velocity pushes correctly in both solvers without solver changes.
  - `World::set_kinematic_target(handle, position, orientation, dt)` sets the
    velocities that reach the target in the next step (turning the short way
    round). The body sweeps through the step, pushing what is in its path,
    instead of teleporting into it.
  - Kinematic orientation integrates exactly (rotation by ω·dt). The
    first-order quaternion update used for dynamic bodies missed a 0.45 rad
    single-step target by more than a milliradian under the classic solver.
    Dynamic bodies are unchanged.
  - A moving kinematic body wakes the islands it touches and keeps them awake.
- **Moving bodies after creation:** `set_pose` (teleport any body; refits the
  broadphase at once and wakes whatever touched it at the old or new pose),
  `set_velocity`, `set_angular_velocity`. Each returns `false`, changing
  nothing, for unknown bodies or non-finite input, like `remove_joint`.
- **Collision filters** (`CollisionFilter { groups, mask }`, `with_filter`,
  `World::set_filter`). Two bodies collide only if each one's mask accepts the
  other's groups, so the rule is symmetric. Checked when building candidate
  pairs, planes included. Default: group 1, collides with everything.
- **Query filters** (`QueryFilter { mask, exclude }`): which groups a query
  sees, minus one body (typically the querying character).
  `raycast_filtered`; `raycast` is the unfiltered case.
- **Shape casts** (`World::shape_cast`): sweep a sphere, capsule, box or hull
  along a direction; returns the first hit's body, distance, point and normal.
  It starts with a swept-AABB broadphase query.
  - Shapes are treated as a core plus a radius (sphere: a point; capsule: a
    segment; box and hull: radius zero). A new **GJK distance** routine
    (`distance.rs`, closest points with witnesses, Ericson's
    closest-point-on-triangle) measures the gap between cores minus radii.
  - **Conservative advancement:** the separating plane is `gap` away, so the
    shape can travel `gap / closing speed` without touching. Repeat until the
    gap is within `CAST_TOLERANCE` (0.1 mm). Planes are solved analytically.
  - A shape that starts overlapping hits at distance zero, with the normal
    from the core separation, or from EPA when the cores themselves overlap.
- **Overlap queries** (`World::overlap`): every body a posed shape overlaps, in
  handle order. Touching doesn't count.

## Deviations from the plan

- The design doc listed sphere and capsule casts. Box and hull casts came free
  from the core-plus-radius approach, so they are included.
- Two bugs found by tests while building it:
  - The tetrahedron step of GJK distance decided which faces the origin was
    outside of from signs alone. That fails for flat simplices, and those are
    common: the Minkowski difference of two segments is a parallelogram. Found
    by the random segment–segment test, which reported 0.1 mm too much. It now
    checks every face whenever the origin isn't inside, with a scale-relative
    flatness test for the inside check.
  - Conservative advancement aimed each step at contact itself. Between flat
    features it lands on contact exactly, and rounding put the shapes 2e-7 m
    inside each other, so `overlap` disagreed with the cast. Found by the
    stepped-overlap test, for a hull cast into a capsule. Steps now aim at half
    the tolerance short.
- Not done: kinematic bodies in the editor and scene files (`BodyType` is still
  dynamic or static). The game creates its kinematic bodies in code; the editor
  can follow when a scene needs one.

## Verification

- fmt, clippy (native + wasm32, `-D warnings`), 281 workspace tests, 149
  classic-solver tests, physics dependency check.
- New tests: 13 kinematic (`tests/kinematic.rs`), 5 filter
  (`tests/filters.rs`), 9 query (`tests/queries.rs`), 6 unit tests (GJK
  distance and filter rules). The GJK distance tests compare 500 random
  point–box cases against clamping in the box's frame and 200 random
  segment–segment cases against dense sampling. The query tests compare
  every caster against every target shape (sphere, capsule, box, hull) with
  stepping `overlap` along the path in 1 mm steps.
- **Benchmark:** within this container's noise. Three alternating runs each of
  `main` and this branch (`cargo run --release -p impulse-physics --example
  bench`), 1,000 bodies, p95 per run:

  | solver | main | M13 |
  |---|---|---|
  | classic | 4.70 / 4.72 / 4.65 ms | 4.66 / 4.76 / 4.91 ms |
  | soft step | 6.53 / 6.69 / 7.44 ms | 6.64 / 6.68 / 6.42 ms |

  Classic averages about 2% slower at p95 (and its mean step, 1.50 against
  1.60 ms on average, about 6%), which may be the filter check in the pair
  loop or noise; the spread between runs of the same build is as large. A
  soft step 500-body p95 outlier on this branch (5.03 ms) came back at
  3.54 ms on the next run, against `main`'s 3.65 ms. The container is noisier
  than the machine in `BENCHMARKS.md`, so these compare branches; they don't
  replace that table. Worth rechecking on the development machine.
