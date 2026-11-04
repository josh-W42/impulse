# R7 — Culling and LOD: Plan + Results

**Date:** 2026-11-03 · **Status:** built, awaiting hand check · **Track:**
[render quality](2026-11-02-render-quality-track.md)

## Why

Every pass draws every instance. The four shadow cascades, the AO prepass
and the main pass each draw the whole scene, wherever the camera looks,
and a sphere 100 m away has as many triangles as one at your feet. A big
scene pays for all of it.

## Decisions

- **No new dependency** (chosen 2026-11-03). LOD meshes come from:
  - procedural coarser versions of the built-in sphere and cylinder;
  - authored glTF LODs, found by node name suffix (`_LOD1`, `_LOD2`, …).

  Imported models without authored LODs draw at full detail. The bundled
  props are already low-poly (26–412 triangles), so they need none.
- **No hysteresis.** The roadmap asked for it, but instances carry no
  identity from frame to frame, so there is nothing to keep the state on.
  Instead each LOD switches where its error is under a pixel on screen,
  so a switch changes the image by less than a pixel. Hiding that change
  is what hysteresis would be for.

## Design

- **Bounds.** Every mesh gets a local bounding sphere at upload: its box's
  centre, and the farthest vertex from it. An instance's world sphere is
  that centre transformed, with the radius times the transform's largest
  axis scale.
- **Frustum culling** (new `cull.rs`, on the CPU):
  - Planes come from a view-projection (Gribb–Hartmann, depth 0..1).
  - Each instance is tested against the camera, each active cascade and
    each shadowed spot, giving it a pass mask. A cascade's frustum is its
    whole light-space box, which already reaches back to catch casters
    outside the view.
  - Each mesh's instance buffer is laid out in ranges by LOD level and
    pass. A pass draws its own range, starting at an instance offset, as
    the blended pass already does on every backend.
  - Blended instances are culled against the camera only.
  - Skinned instances aren't culled: they have no animated bounds, and
    there are few of them.
- **LOD.**
  - A mesh holds up to 4 levels (`MeshLod { data, error }`) in one vertex
    and index buffer. Indices are offset on the CPU, because WebGL2 has no
    base vertex.
  - Selection: the coarsest level whose `error`, projected at the
    instance's distance and scale, is under `LOD_PIXELS` (1 px) at the
    current viewport height. Shadow passes use the camera's choice, so
    surfaces shadow themselves consistently.
  - Errors are measured, not authored: `lod_error(full, coarse)` is the
    farthest a full-detail vertex lies from the coarse surface. Up to 512
    sampled vertices are tested against every coarse triangle.
  - Built-in levels: the sphere goes 32×16 → 16×8 → 8×4 and the cylinder
    32 → 16 → 8 segments. The cube and plane have none.
  - glTF: a node named `<base>_LOD<n>` (any case, n ≥ 1) gives level n of
    the base node's primitives, matched by order, drawn with the base's
    material. LOD nodes that match nothing are dropped, never drawn as
    extra geometry. New field: `ModelMesh.lods`.
- **Renderer API (additive):**
  - `add_mesh_with_lods`;
  - `CullSettings { culling, lod }` with getter and setter, both on by
    default (for comparing on and off);
  - `draw_stats()`: instances submitted, drawn, culled, shadow casters
    and triangles.
- **Demo: Field.** A large floor with draw-only scattered props: rocks,
  logs, barrels, crates, spheres and pillars.
  - `SceneDesc.scatter: Option<Scatter { count, half_extent, seed }>`. It
    defaults to `None`, so old files still load.
  - The props are generated deterministically when the scene is built.
    They have no physics bodies and aren't in the object list.
  - 10,000 props by default. The Rendering panel shows the draw stats and
    the on/off toggles.

## Tasks

1. Render: bounds, frustum planes and sphere tests, pass masks, and LOD
   selection, with tests:
   - a sphere just outside each plane is culled, and one touching a plane
     is kept;
   - every instance a cascade would draw is kept for that cascade;
   - projected error picks coarser levels with distance and is under a
     pixel where it switches;
   - the measured error of the built-in sphere's levels matches the
     analytic sagitta.
2. Renderer: per-mesh instance ranges by LOD and pass, LOD levels in one
   buffer, `CullSettings`, `draw_stats`.
3. Assets: `_LOD<n>` nodes, with tests (matched, unmatched, wrong
   primitive count).
4. Engine: built-in LODs; uploading glTF LODs with measured errors; the
   scatter in `SceneDesc`; the Field demo; stats and toggles in the panel.
5. Checks: fmt, clippy (native + wasm32), tests, naga, and `cargo check`
   on solstice and Ashen Crown.
6. Verify on WebGPU, WebGL2 and native:
   - Field with culling and LOD on and off, comparing frame times;
   - no popping or holes visible when orbiting;
   - shadows unchanged at the edge of the view;
   - the other demos unchanged.

## Results

Built as designed, with these changes:

- **glTF LOD nodes with no base node are ordinary geometry**, not dropped.
  A file whose only mesh node happens to be called `rock_LOD1` still
  loads. Extra LOD primitives that match nothing on the base node are
  still dropped.
- **Each pass binds its range of the mesh's instance buffer** (a slice
  offset) instead of drawing from a first instance. It works the same on
  every backend.
- **`Renderer::replace_mesh` drops a mesh's levels of detail**, so it
  draws at full detail.

Field demo, default orbit camera, 1280×1100, WebGPU (RTX 3070 Ti):

| Props | Culling, LOD | Drawn by the camera | Shadow-map instances | Triangles (camera) | Frame |
| --- | --- | --- | --- | --- | --- |
| 10,000 | on, on | 2,823 of 10,006 | 1,111 | 363.7k | 16.7 ms |
| 10,000 | off, off | 10,006 | 30,018 | 3,029.2k | 16.7 ms |
| 50,000 (temporary build) | on, on | 14,202 of 50,006 | 5,491 | 1,819.3k | 16.7–33 ms (flips) |
| 50,000 | on, off | 14,202 | 5,491 | 4,304.3k | 33.3 ms |
| 50,000 | off, off | 50,006 | 150,018 | 15,186k | 33.4 ms, steady |

- At 10,000 props this GPU holds 60 fps either way. Culling and LOD cut
  the camera's triangles 8.3× and the shadow-map instances 27×.
- At 50,000 props, culling and LOD bring a steady 30 fps up to the edge
  of 60.
- Frame times are vsync-rounded, and the browser pane was hidden, which
  seems to cap animation frames near 30 fps: counted over 5 s, both
  settings gave 22–30 fps. Treat the frame column as indicative. The
  triangle and instance counts are exact.
- **No uncapped native benchmark.** During the 50,000-prop test the GPU
  sat at 100% and 63 °C with fan 1 dead, so a full-load run needs a
  cool-down routine like loom's `cool_bench.sh`.
- **CPU cost:** sorting 10,000 instances into the camera and 4 cascades,
  with LOD choice, takes 1.15 ms a frame (native release). That includes
  building each instance's GPU data, which the old path did too.

| Check | Result |
| --- | --- |
| Unit tests: cull | planes from a view-projection: spheres just past each plane culled, touching ones kept; every point inside each cascade's box survives its culling, and points well outside don't; bounds cover their points under rotation and non-uniform scale; LOD never gets finer with distance, and each switch is under a pixel; instances go to each pass that sees them, at their level; blended instances drawn by the camera only, far to near from the eye; stats; everything drawn with culling and LOD off |
| Unit tests: mesh | closest points on a triangle (face, corners, edges); built-in levels valid, coarser, with rising errors; the cylinder's measured error equals the 16-gon sagitta; the sphere's is within its bounds |
| Unit tests: assets | `_LOD<n>` parsing (any case, `_LOD0` base, rejects); levels join their base in level order; orphans stay geometry; bounds ignore LOD nodes; primitives matched by order, extras dropped |
| Unit tests: engine | scatters deterministic, inside the square and clear of the middle, on the ground, ~70% model props; plain shapes without models; a tiny field still fills; invalid scatters refused |
| clippy native + wasm32 (`-D warnings`), workspace tests (25 suites, 395) | clean / all green |
| WebGPU, Field | the image with culling and LOD on is identical to off, shadows included; no console errors |
| WebGL2, Field | same counts as WebGPU, 16.7 ms after warm-up |
| WebGPU, Glass and Lights | unchanged: blended glass sorted, spot shadows drawn |
| Native Vulkan debug and release, Field and Lights | no validation errors; the clouds bake takes 185 ms in release |
| solstice, Ashen Crown | `cargo check` clean |

- The web build grows from 14.78 to 14.85 MB of wasm.

Not done, or found along the way:
- **Skinned instances aren't culled** (no animated bounds).
- **No bundled model has authored LODs.** The glTF path is covered by
  unit tests only.
- **The GTAO streaks inside shadows (R3)** and the HDR skies' sky-coloured
  lower half (R6) are still open.
- **Frame-time cost of R1–R7 is still unmeasured with GPU timestamps.**
