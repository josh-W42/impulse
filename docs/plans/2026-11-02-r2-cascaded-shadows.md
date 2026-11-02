# R2 — Cascaded Shadow Maps: Plan + Results

**Date:** 2026-11-02 · **Status:** built, awaiting hand check · **Track:**
[render quality](2026-11-02-render-quality-track.md)

## Why

One 2048² shadow map covers a fixed square (±25 m around the origin, or
around `set_shadow_focus`). Close up, a texel is 2.4 cm and shadows are
soft and blocky. Outside the square there are no shadows at all, and Ashen
Crown's world is much bigger than 50 m.

## Design

- **Cascades:** 4 cascades, each a 2048² layer of one `Depth32Float` 2D
  array texture. WebGL2 samples depth arrays with `sampler2DArrayShadow`.
  The array also avoids WebGL2's guaranteed 2048 size limit, which a
  4096² atlas would exceed. Memory: 64 MB.
- **Depth range the cascades cover:**
  - default (`set_shadow_distance(d)`, 60 m): from the camera's near plane
    to `d`;
  - `set_shadow_focus(focus, r)` (solstice): only the view depths within
    `r` of the focus. Same signature as today, with a narrower range and so
    sharper shadows.
- **Splits:** the "practical" scheme, λ = 0.75 between logarithmic and
  uniform.
- **Fit:** each cascade gets the bounding sphere of its frustum slice. The
  radius depends only on fov, aspect and the split depths, so it doesn't
  change as the camera turns. The light-space centre snaps to whole texels,
  so the cascade doesn't shimmer as the camera moves. The ortho depth range
  extends 100 m toward the sun so casters outside the view still cast.
- **All maths from `view_proj`.** It runs in a pure `shadow.rs`
  (unit-tested), so `CameraData` is unchanged.
- **Shadow passes:** one per cascade. Each binds a small caster uniform (its
  matrix) at group 3. The shadow vertex shaders read only that, not
  `Globals`.
- **Lit shader:**
  - picks the cascade by view depth;
  - offsets the position along the normal by about one texel of that
    cascade (normal-offset bias);
  - uses the existing 3×3 PCF;
  - blends into the next cascade over the last 10% of each one, and fades
    to unshadowed at the end of the last.
- **No rotated Poisson PCF.** Without TAA its per-pixel rotation shows as
  noise, so the roadmap item is dropped.
- **Debug view:** an optional tint per cascade.
- **Sandbox:** a shadow-distance slider and a "show cascades" toggle.

## Tasks

1. `shadow.rs`: splits, slice corners from `view_proj`, sphere fit, texel
   snap, focus/distance ranges. Tests: coverage of every slice corner,
   stability under camera rotation and translation, focus range.
2. Array texture, caster uniforms and bind groups, per-cascade shadow
   passes; `Globals` carries cascade matrices, splits and texel sizes.
3. `scene.wgsl`: cascade selection, normal offset, blend, fade, debug tint.
   naga WebGL2 check.
4. Renderer API `set_shadow_distance`, `set_show_cascades`; sandbox panel.
5. fmt, clippy (native + wasm), tests, `cargo check` solstice and Ashen
   Crown.
6. Verify WebGPU, WebGL2, native; screenshots with cascades tinted.

## Results

Built as designed, with one change found by the tests:

- **Focus mode caps cascades at the focus ball.** The first focus test
  showed whole-frustum spheres for a solstice-style camera, 40 m from its
  focus. They came out at 3–6.5 cm per texel, against today's 2 cm single
  map over a 20 m ball: a regression. A 16:9 slice is mostly area outside
  the ball. Now each cascade covers whichever is smaller, its slice or the
  ball. The first time the ball wins, that cascade covers every remaining
  depth and the rest are skipped (`Cascades::count`).
  - No cascade is ever blurrier than the old single map.
  - A zoomed-out RTS camera costs one shadow pass, as before.
  - Zoomed in, near cascades are much sharper.
- **The sandbox's orbit camera** uses focus mode around the orbit target
  with a 25 m radius: the old map's ±25 m, so the scene centre is never
  worse than before. Fly mode uses distance mode at twice the radius.
- **Ashen Crown** sets no range, so it gets the 60 m distance default.
  That gives shadows wherever the camera looks, where before they only
  existed within 25 m of the origin.

| Check | Result |
| --- | --- |
| `shadow.rs` tests (10) | splits; every slice corner and the casters 50 m above it land in their cascade; turning keeps sizes; moving shifts by whole texels; focus never blurrier than one map; far focus → 1 cascade, near focus → sharper ones; ball coverage; clamping; sun overhead |
| Shader → WebGL2 GLSL (naga) | ok (depth array + `sampler2DArrayShadow`) |
| clippy native + wasm32 (`-D warnings`), workspace tests | clean / all green |
| WebGPU, WebGL2 (browser pane, cascades tinted) | cascades and blends as expected, crisp contact shadows when zoomed in, no console errors |
| Native Vulkan debug + release; `skinned_character` example | no validation errors |
| solstice, Ashen Crown | `cargo check` clean |

Not done:
- **Frame-time cost not measured.** Up to 4 shadow passes instead of 1;
  there's no GPU timestamp tooling yet (as in R1).
- **Distance mode trades mid-range sharpness for reach.** At 60 m, the
  last cascade is about 6.8 cm per texel. A game with a natural focus
  should call `set_shadow_focus`.

