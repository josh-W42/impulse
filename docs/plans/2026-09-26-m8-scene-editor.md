# M8 — Scene Format + Editor: Design + Lean Plan

**Date:** 2026-09-26 · **Status:** approved · **Parent:** `2026-09-24-impulse-engine-design.md`
**Follow-up:** M9 — glTF models, textures, mesh-derived colliders (a new object kind).

Goal: engine usability. Scenes become data; an in-engine editor authors them.

## Design

**Scene description** (engine crate, serde; the physics crate stays glam-only):

```rust
struct SceneDesc { version: u32, name: String, arena: Arena, objects: Vec<ObjectDesc>, joints: Vec<JointSpec> }
enum Arena { Pit { half: f32 }, Floor }
struct ObjectDesc {
    name: String,
    shape: Option<ShapeDesc>,          // None = invisible anchor
    position: [f32; 3], rotation_deg: [f32; 3],   // Euler XYZ degrees
    body: BodyType,                    // Dynamic { mass } | Static
    restitution: f32, friction: f32, color: [f32; 3],
    velocity: [f32; 3], angular_velocity: [f32; 3],
}
enum ShapeDesc { Sphere { radius }, Box { half_extents }, Capsule { half_height, radius }, Hull { points } }
struct JointSpec { a: usize, b: usize, kind: JointKindSpec, collide_connected: bool }
```

- `build_world(&SceneDesc) -> Result<(World, Vec<SceneObject>), SceneError>` is the
  only way scenes are built. The five demos become `SceneDesc` generators with
  unchanged behaviour.
- RON files with a `version` field. Vectors are plain arrays, so serde never
  enters the physics crate's dependency tree.
- Errors never panic. `SceneError` names the object or joint (e.g.
  `object 12 'crate': invalid mass`), wrapping `BodyError`/`JointError`, and the
  current scene is kept.
- New dependencies: `serde` (with the `derive` feature) and `ron`, both in the
  engine crate only.

**Editor** (Unity-style modes):

- **Edit** — physics paused; the world is rebuilt from the description after
  every edit, so the view always shows the authored scene.
- **Play** — simulates from the authored scene. **Stop** returns to it exactly.
- **Capture pose** — writes the simulated poses back into the description.
- **Pick** — clicking uses the existing raycast. The selection is tinted and
  its AABB drawn as lines.
- **Inspector** (egui) — name, shape and size, position and rotation,
  dynamic/static and mass, restitution and friction, colour, initial
  velocities. It also lists the object's joints, with delete.
- **Editing** — add sphere/box/capsule in front of the camera; drag on the
  floor plane (Shift = vertical); Ctrl+D duplicate; Del delete (removes the
  object's joints and reindexes the rest); Ctrl+Z/Ctrl+Y undo/redo using
  whole-description snapshots (100 deep).
- **Save/load** — native: a name field, and the `.ron` files in `scenes/`.
  Browser: download the file, or upload one through a file picker (web-sys
  features only). The demos stay available as templates, and `?scene=` still
  works.
- **Deferred** — gizmos, joint authoring, multi-select, hierarchy (backlog M10?), prefabs.
- **Added after review** — a flat objects list (Edit mode): select by name, filter,
  follows viewport picks; invisible anchors become selectable.

## Tasks (TDD; fmt + clippy ×2; commit each)

**S1. `scene_desc.rs`**: the types, RON read/write, `SceneError`, and
`build_world`. Tests: every shape, body, joint and arena builds; round trip;
malformed RON, bad shape or mass, and an out-of-range joint index each give a
specific error.
**S2. Port the demos to description generators**; `scene::build` goes through
`build_world`. A temporary check confirms new equals old (body count, poses,
joints); existing scene tests stay green.
**S3. `editor.rs` (pure logic)**: `EditorState { desc, selected, undo, redo }`
with add, duplicate, delete (reindexing joints), set, undo/redo and capture
pose. Unit tests.
**S4. Modes and UI wiring** in `running.rs`: Edit, Play and Stop; picking;
highlight; inspector; plane drag; shortcuts. Test that Stop restores the
authored scene bit-identically.
**S5. Save/load**: native `scenes/` and browser download/upload; errors shown
in the panel.
**S6. Verify** (native, WebGPU, WebGL2); record results here; update the README;
ship after desktop confirmation.

## Results

- 188 tests pass across the workspace. New tests cover:
  - scene descriptions: every shape, body and joint kind builds; RON round
    trip; hand-written files may omit defaults; specific errors, including
    oversized files; safe file names;
  - the editor: add, delete (joint renumbering), duplicate, gesture undo,
    history cap, capture pose, replace;
  - UI helpers: picking, ray–plane intersection, shape resizing;
  - input: key and text events;
  - native save, list and read;
  - Stop restoring the scene exactly.
- **Demos are unchanged by the port.** A before/after fingerprint of all five
  demos (2,238 lines of body, joint and colour state) matched structurally,
  with a largest numeric difference of 1.4e-7, the rounding from Euler
  degrees.
- **Bugs found along the way:**
  - **Rebuilt worlds replayed differently.** `World::clear()` reused slots in
    reverse, so a rebuilt scene got the opposite handle order, a different
    solve order, and a different simulation from a fresh world, alternating
    on each rebuild. Fixed in physics (slots are now reused lowest-first),
    with a regression test.
  - **Stale handles.** Swapping in a fresh `World` let old handles, such as
    the drag target, point at new bodies. `rebuild` now validates the
    description off to the side, then clears and refills in place.
  - **Idle rebuilds.** egui's colour button nudges the colour's low bits every
    frame, which marked the scene edited and rebuilt the world each frame. A
    test now runs the inspector idle over every demo and checks nothing
    changes.
  - **No keyboard input in egui.** The custom input layer never forwarded
    keys, text or modifiers, so no text field could work. Added, and app keys
    now stand down while a field has focus.
- **Changes from the approved design:**
  - Capture pose writes back only objects without joints. Joint limits and
    cones are defined at the authored pose and can't be re-expressed for a
    jointed body that has moved.
  - New shapes are placed on top of whatever is under the camera target,
    using a downward raycast.
- **Verified in the browser:** Edit/Play/Stop, selection highlight and
  inspector, adding and plane-dragging objects, and Delete, on both WebGPU and
  WebGL2.
- **Not verified by automation:**
  - Ctrl shortcuts: the automation tool sends Ctrl as a flag on the key,
    without the Ctrl key press itself.
  - Typing into fields: in the hidden, throttled pane, the focusing click
    and the keystrokes arrive in the same frame.
  - Browser download and upload.
  - These are covered by the manual desktop and browser check.
