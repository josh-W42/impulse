# Render Quality Track — Roadmap

**Date:** 2026-11-02 · **Status:** in progress (R1–R6 merged) · **Goal:** raise the image quality
and the amount of detail impulse can draw, in milestones that each ship a
deployed demo, like every milestone before.

## Where the renderer stands

One forward pass straight into the sRGB swapchain (`renderer.rs::render`):

| Area | Today | Visible result |
| --- | --- | --- |
| Anti-aliasing | none (`sample_count: 1` everywhere) | jagged edges, shimmer on thin props |
| HDR | none; Khronos PBR Neutral tonemap runs inside `fs_main` | no bloom, no exposure control, bright sun clips |
| Shadows | one 2048² map over a ±25 m square, 3×3 PCF | soft-blurry up close, nothing beyond 25 m |
| Ambient occlusion | glTF occlusion texture only | objects float; contact points look flat |
| Lights | the sun only | no lamps, torches, spell glows |
| Environment | analytic 3-colour gradient sky as IBL | metals reflect a smooth gradient, not a world |
| Transparency | none (no blend state; glTF `alphaMode` ignored) | no foliage cut-outs, glass, particles |
| Geometry | every instance drawn every frame; fixed 32×16 spheres | no culling, no LOD, scene size bounded by draw cost |

## Constraints every milestone keeps

- **WebGL2 stays supported.** No compute shaders, no fragment storage
  buffers. Float render targets need `EXT_color_buffer_float`, so each
  milestone's first task checks the WebGL2 adapter and keeps a fallback path.
- **Public API is shared.** solstice and Ashen Crown call `Renderer`; new
  features default on with a switch, and any signature change is flagged.
- **Every milestone is checked on WebGPU, WebGL2 and native** (debug and
  release) before pushing, with before/after screenshots and frame-time cost
  added to `BENCHMARKS.md`.
- **No new dependencies** unless a milestone names one and it is approved.

## Milestones (in order)

### R1 — HDR pipeline, MSAA, bloom  *(foundation)*
- Render the scene into an `Rgba16Float` target with 4× MSAA; resolve, then
  a fullscreen post pass does exposure + tonemap into the swapchain.
- Bloom: downsample/upsample chain (Jimenez, CoD:AW style) in fragment shaders.
- Exposure and bloom controls in the panel; sky moves to the HDR target too.
- Fallback if WebGL2 can't render floats: today's in-shader tonemap path.
- **Why first:** every later effect (AO, lights, bloom on emissives) needs a
  linear HDR buffer, and MSAA is the single most visible fix.

### R2 — Cascaded shadow maps
- 4 cascades in one atlas (WebGL2-safe), practical split scheme,
  texel-snapped per cascade, blended at seams.
- Normal-offset bias; rotated-Poisson PCF.
- `set_shadow_focus` becomes the cascade-0 hint (kept for solstice).

### R3 — Ambient occlusion and fog
- Depth + normal prepass (non-MSAA, sampleable on WebGL2).
- Half-resolution GTAO-style AO with a bilateral blur, applied to ambient
  light only.
- Height/distance fog with the sky colour (aerial perspective lite).

### R4 — Transparency and cut-outs
- Read glTF `alphaMode` / `alphaCutoff` in `impulse-assets`.
- `MASK`: alpha test with alpha-to-coverage under MSAA (foliage).
- `BLEND`: a sorted back-to-front pass after opaque, before post.

### R5 — Local lights
- Point and spot lights (range, cone, colour, intensity) in a uniform buffer
  with a fixed cap (e.g. 64), CPU range-culled per frame.
- Editor: add/move lights, gizmo; scene format bump with migration.
- Optional: one shadowed spot light.

### R6 — Environment lighting
- Load an equirectangular HDR (`.hdr` RGBE: small enough to parse ourselves),
  convert to a cubemap, prefilter roughness mips, project to SH9 — all in
  render passes. Bundle one CC0 Poly Haven sky.
- The procedural sky stays as the default and as the fallback.

### R7 — Geometry scale: culling and LOD
- Per-instance bounding spheres; CPU frustum culling for scene and shadow
  passes (shadow cascades cull against their own frusta).
- `MeshId` gains LOD levels chosen by screen-space size, with hysteresis.
- LOD source: authored LOD meshes in the glTF (by node name suffix), or
  generated simplification — the latter means a dependency (`meshopt`),
  to be approved in R7's plan.
- Demo: a dense scene (e.g. 10k props) at locked 60 fps on WebGPU.

## Not in this track
Deferred shading, ray tracing, TAA/upscaling, virtualized geometry — large,
mostly WebGPU-only, and not needed for the scenes impulse's games draw.
