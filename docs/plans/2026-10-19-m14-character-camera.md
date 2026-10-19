# M14 — Character Controller and Third-Person Camera: Design + Results

**Date:** 2026-10-19 · **Status:** complete · **Driven by:** Ashen Crown
(`../ashen-crown/docs/plans/2026-10-19-design.md`), whose players walk a zone
as capsules under an over-the-shoulder camera.

## Why

M13 gave the physics kinematic bodies and shape casts, but a game still had to
work out walking itself: what happens at a wall, a slope, a step, a ledge, the
top of a jump. The engine's cameras (orbit, fly, strategy) were not built to
follow a character.

## What shipped

- **`CharacterController`** (`impulse-physics`, so the game server runs it
  headless). A kinematic capsule plus the state that moving it needs (vertical
  speed, ground contact). `update(world, velocity, jump, dt)` plans one step
  against the world as it is, then calls `set_kinematic_target`; the next
  `World::step` carries the body there exactly.
  - **Collide and slide:** sweep the capsule, stop `skin` (2 cm) short of the
    first hit, slide what's left along the surfaces hit so far, up to four
    times. Two surfaces: along their crease; three: stop.
  - **Walls and slopes:** surfaces steeper than `max_slope` (45°) count as
    vertical walls while walking, so sliding along them can't climb them.
    Walkable slopes are slid up directly.
  - **Steps:** a walk blocked by a wall tries up by `step_height` (35 cm),
    across, and back down onto walkable ground, and keeps whichever got further.
  - **Ground:** falling lands on walkable ground (and stops there instead of
    sliding along it); walking down slopes and off small steps follows the
    ground within `snap_distance` (30 cm) instead of launching into the air.
  - **Jumps** with `jump_speed` (5 m/s, about 1.3 m) under the world's gravity;
    ceilings stop the rise.
  - **Edges:** a capsule meets a ledge first at its corner, where the contact
    normal points mostly sideways. What decides walkability there is the
    surface beyond the corner, read with a 4 cm ray just past the contact.
  - `CharacterDesc` sets the shape and limits; `query_mask` chooses what blocks
    the character (anything outside it is pushed by the kinematic capsule
    instead); `teleport` for spawning and respawning.
- **`ThirdPersonCamera`** (`impulse-engine`): orbits a target point (yaw,
  pitch, distance) with the same convention as the other cameras;
  `forward()`/`right()` give camera-relative movement directions on the ground.
  Default controls: right-drag or one-finger drag looks around, wheel or pinch
  zooms. `avoid_obstacles(world, filter, dt)` sweeps a 0.2 m sphere from the
  target toward the eye and pulls the eye in front of the first hit at once,
  then eases back out (rate 4/s) when the way clears.
- **`impulse_engine::physics`:** the engine re-exports `impulse-physics`, like
  `render` and `glam`, because `ThirdPersonCamera` takes physics types.

## Deviations from the plan

- Three behaviours failed their tests on the first run, all fixed:
  - The skin was measured along the sweep, not along the surface normal.
    Moving diagonally into a wall, stopping 2 cm short along the path is only
    1.4 cm from the wall, so the character crept closer each frame (6 mm over
    a test). It now backs off `skin / cos θ` along the path.
  - Step-up rejected every ledge: the corner contact's sideways normal looked
    like a wall. Fixed by the edge probe above.
  - With the probe, it then climbed a 0.6 m ledge (limit 0.35 m): the rounded
    bottom settled on the corner with the feet low enough to pass a rise-based
    check. The limit now applies to the height of the surface found.
- Not done: riding moving platforms (the character doesn't inherit a
  platform's velocity), and crouching. Neither is needed yet.

## Verification

- fmt, clippy (native + wasm32, `-D warnings`), 304 workspace tests, 168
  classic-solver tests, physics dependency check.
- New: `tests/character.rs` (16) and 3 controller unit tests; 4 camera tests.
  The character suite covers landing, exact walking speed, walls and sliding,
  inside corners (no drift over a second of pushing into one), 30° ramps up
  and down (grounded every frame going down), a 60° ramp refused, 0.3 m steps
  taken and 0.6 m refused, stairs, jump height (1.27 m expected, within
  10 cm), ceilings, query masks (walk through, push, or be blocked by
  crates), and teleport. A randomized test walks and jumps for 6,000 steps
  through 40 random rotated boxes and asserts every step that the capsule
  never overlaps static geometry.
- In Ashen Crown (browser, WebGL2): ran into a wall and stopped at the
  expected distance (z = −13.1, expected −13.13); a 55° ramp blocked; a jump
  left the ground with the camera following.
