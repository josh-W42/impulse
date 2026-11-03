# R4 — Transparency and Cut-outs: Plan + Results

**Date:** 2026-11-02 · **Status:** hand-checked and merged (PR #8) · **Track:**
[render quality](2026-11-02-render-quality-track.md)

## Why

Everything draws opaque. Instance alpha and glTF `alphaMode` are ignored,
so there is no glass, no foliage and no fences: a cut-out texture draws
as a solid quad and casts a solid shadow.

## Design

- **Material:** `MaterialDesc` gains `alpha_mode: AlphaMode { Opaque,
  Mask { cutoff }, Blend }` and `double_sided`. Both default to glTF's
  defaults (opaque, single-sided).
- **Assets:** `impulse-assets` reads `alphaMode`, `alphaCutoff` and
  `doubleSided`, and `engine/upload.rs` maps them.
- **Instance alpha:** an instance whose colour alpha is below 1 draws
  blended, whatever its material. That gives glass on built-in shapes.
- **Mask (cut-outs):**
  - **Under MSAA:** alpha-to-coverage, with the alpha sharpened to one
    pixel of falloff around the cutoff (`(a − cutoff) / fwidth(a) + 0.5`).
    Edges come out anti-aliased and there's no sorting.
  - **Without MSAA:** `discard`.
  - **Shadow and AO prepasses** alpha-test too, so cut-outs cast cut-out
    shadows and don't occlude as solid quads.
- **Double-sided:** no culling, and the normal flips on back faces
  (`front_facing`).
- **Blend (glass):**
  - after the opaque pass and the sky, before debug lines;
  - per instance, sorted back to front by distance to the eye;
  - depth test on, depth write off, standard alpha blending into the HDR
    target;
  - it ignores screen-space AO (that texel belongs to whatever is behind),
    casts no shadow and isn't in the AO prepass.
- **Pipelines:** every pipeline that draws meshes is built per alpha mode
  (opaque or mask, plus blend for lit) and per sidedness. Each material
  records its variant and the draws are grouped by it.
- **Skinned meshes:** stay opaque, single-sided. This is noted rather than
  extended; no skinned asset needs it yet.
- **Engine:**
  - `ObjectDesc.opacity` (serde default 1), with an inspector slider. No
    scene-version bump: missing fields already default.
  - A Glass demo: tinted glass boxes and spheres, plus a generated
    chain-link fence model (alpha-masked, double-sided). The fence comes
    from `scripts/make-fence-glb.py`, ours (CC0), and its thin wires are
    the alpha-to-coverage showcase.

## Tasks

1. Assets: read alpha mode, cutoff, double-sided (tests on a generated
   glb).
2. Renderer: `AlphaMode`, material variants, pipeline sets, mask and
   double-sided in `scene.wgsl`, shadow and prepass alpha tests, sorted
   blend pass (sorting unit-tested).
3. Engine: upload mapping, opacity field and inspector, fence model,
   Glass demo.
4. fmt, clippy (native + wasm), tests, `cargo check` solstice and Ashen
   Crown.
5. Verify WebGPU, WebGL2, native: glass sorting, fence edges with and
   without MSAA, cut-out shadows.

## Results

Built as designed. Additions made along the way:

- **Mip-aware cut-out alpha (Golus).** Mips average thin wires below the
  cutoff, so they would vanish with distance. The lit shader scales a
  mask's alpha up by 25% per mip level.
- **`IMPULSE_SCENE` env var on desktop**, matching the web build's
  `?scene=`, so native checks can open a specific demo.
- **Browser builds log uncaptured GPU errors**
  (`device.on_uncaptured_error`). Without it, an invalid pipeline in the
  browser blanks frames silently. I added it while chasing a blank Glass
  scene, which turned out to be the hidden browser pane reporting a 1×1
  canvas, not a renderer bug. It stays because it would have answered that
  in one step. Native keeps wgpu's panic.

| Check | Result |
| --- | --- |
| Unit tests | assets: alpha mode, cutoff (with glTF's 0.5 default), double-sided; renderer: translucent instances and blended meshes split off far to near, sorting by distance from the eye, unknown meshes dropped; engine: opacity round-trips, reaches the draw colour, defaults to opaque for old files |
| Shaders → WebGL2 GLSL (naga) | `fs_main` (front-facing, mask, fwidth/dpdx), `fs_prepass_mask`, `vs_shadow_mask` / `fs_shadow_mask`: ok |
| clippy native + wasm32 (`-D warnings`), workspace tests | clean / all green |
| WebGPU + WebGL2, Glass demo | overlapping tinted panes composite in depth order, opaque spheres show through, glass blocks and balls fall and stack, fence wire lattice anti-aliased (alpha-to-coverage) with solid posts, no console errors |
| WebGPU with MSAA off (temporary build) | fence still reads, with hard `discard` edges |
| Native Vulkan debug (Glass, Pit) and release (Glass); `skinned_character` | no validation errors |
| solstice, Ashen Crown | `cargo check` clean |

Not done:
- **Frame-time cost not measured** (R1–R4).
- **Skinned meshes stay opaque and single-sided.**
- **Translucent objects cast no shadow and take no screen-space AO** (by
  design). Coloured glass shadows would need a separate transmittance map.
- **Fence cut-out shadows not inspected closely.** The pass alpha-tests,
  but the demo's sun angle and camera made the lattice shadow hard to see
  in the browser pane.
