# R5 — Local Lights: Plan + Results

**Date:** 2026-11-02 · **Status:** hand-checked and merged (PR #9) · **Track:**
[render quality](2026-11-02-render-quality-track.md)

## Why

The sun is the only light. There are no lamps, torches or glowing
spells, and a dusk or night scene can't be lit at all.

## Design

- **Renderer API (additive):**
  - `Light { position, color, intensity, range, kind, shadows }` with
    `LightKind::{Point, Spot { direction, inner_angle, outer_angle }}`.
  - Lights go in the frame's `DrawList` (`lights`, `push_light`), like
    instances: immediate mode, rebuilt every frame.
  - `Instance.emissive` (a strength on the instance colour, through the
    unused `pbr.z`), so a built-in sphere can glow like the bulb it holds.
    It's a new public field (`push_instance_emissive`); downstream games use
    the push helpers, not struct literals.
  - `Renderer::set_sky` plus `Sky::dusk()` / `Sky::night()` presets, since
    local lights only read in a dim scene.
- **Shading:** the same GGX BRDF as the sun, factored into a function.
  - Inverse-square falloff with a smooth window to zero at `range` (Karis):
    `saturate(1 − (d/r)⁴)² / max(d², 0.01)`.
  - Spot cones as in glTF `KHR_lights_punctual`, so glTF lights can map
    straight in later.
- **Clustered forward (CPU-binned):** instead of every pixel looping over
  every light, the view is split into 16 × 9 tiles × 24 exponential depth
  slices (3,456 clusters). Each frame the CPU:
  1. frustum-culls light spheres, keeping at most 256 (nearest first);
  2. bins each light into the clusters its bounding box covers, using the
     tighter cone-bounding sphere for spots;
  3. uploads the lists to one `R32Uint` texture: a header per cluster
     (offset and count packed), then the light indices.
  The shader finds its cluster from the pixel and its view depth and loops
  only over that cluster's lights.
  - WebGL2-safe: textures and uniform buffers only, no storage buffers or
    compute. 256 lights × 64 bytes is exactly WebGL2's 16 KB uniform
    binding limit.
  - Depth slices run from the near plane to the farthest visible light's
    reach, so slices aren't wasted on empty distance.
- **Shadowed spot lights:** up to 4, chosen nearest the camera among those
  with `shadows` on.
  - A `Depth32Float` 2D array with 1024² layers, drawn with the existing
    shadow pipelines (one caster matrix per light).
  - 3×3 PCF, with a normal offset scaled to the texel size at the
    surface's distance.
  - Point-light shadows (cube maps, 6 passes each) are out of scope.
- **Debug:** a light-cluster heat map in the Rendering panel, plus a count
  of visible and binned lights.
- **Engine:**
  - `ObjectDesc.light: Option<LightDesc>`: a light rides on an object. An
    empty object makes a bare light, a sphere makes a lamp, and a dynamic
    body makes a light that moves. Spots point along the object's local −Y.
  - `ObjectDesc.emissive` (glow strength) and `SceneDesc.sky` (Day, Dusk,
    Night).
  - All new fields default, so there's no scene-version bump: old files
    load unchanged.
  - Editor:
    - "Add light" adds an empty with a point light.
    - The inspector gets a Light section and an emissive slider.
    - Lights draw as gizmos (a star, plus a range ring or a cone when
      selected) and can be picked by clicking the gizmo.
    - A sky picker sits with the scene settings.
  - A Lights demo at dusk:
    - a colonnade of coloured lamps;
    - two shadowed stage spots over a pile of blocks;
    - glowing balls that fall and roll, so the lights move;
    - a field of 64 small lanterns behind, so clustering has work to do.

## Tasks

1. Renderer:
   - `Light` types and the `DrawList` lights;
   - clustering in `lights.rs`, unit-tested: culling, slices, binning
     coverage, cone bounds, overflow;
   - the grid texture and lights uniform;
   - shading in `scene.wgsl`;
   - the heat map;
   - `set_sky` and the sky presets.
2. Spot shadows: an array texture, casters, passes, and the shader lookup.
3. Engine: light, emissive and sky fields; draw-list lights; editor UI,
   gizmos and picking; the Lights demo.
4. Checks: fmt, clippy (native + wasm), tests, and `cargo check` on
   solstice and Ashen Crown.
5. Verify on WebGPU, WebGL2 and native: lamps, spot shadows, moving
   lights, heat map, and no change to existing demos.

## Results

Built as designed. Changes and findings along the way:

- **Dusk preset tuned darker** after the first look. At the first values
  the low sun and orange haze outshone the lamps.
- **Frame time in the browser pane is noisy for the first seconds after a
  load.** Early readings of 120–150 ms on Lights sent me bisecting (light
  loop off, spot shadows off), but the numbers didn't reproduce. Once
  settled, Lights holds 16.7 ms (vsync) on WebGPU, like the other demos.
  CPU clustering of the demo's 82 lights takes about 0.2 ms natively
  (release, a throwaway timing test). GPU cost is still unmeasured; that
  needs timestamp queries.

| Check | Result |
| --- | --- |
| Unit tests: clustering | slices run from near to far in order; cone bounds hold the whole cone and beat the range sphere; every lit visible point finds its light in its own cluster (40 point and spot lights, about 5,000+ sampled points); out-of-view and dark lights culled, slices end at the farthest light's reach; no lights means empty clusters; over the cap, the farthest go; a light around the camera covers the whole screen; the nearest shadowed spots get layers; spot cones reach the shader in glTF form |
| Unit tests: engine | light, glow and sky round-trip, and old files load with none; spots shine along the object's local −Y; the Lights demo hands every light to the renderer at its body's position (2 shadowed spots, markers only on request); picking finds a bare light by its marker unless a body is in front |
| Shaders → WebGL2 GLSL (naga) | `fs_main` with the light loop, spot shadows and heat map: ok |
| clippy native + wasm32 (`-D warnings`), workspace tests (25 suites) | clean / all green |
| WebGPU, Lights demo | lamps colour the pillar faces; lanterns cast coloured pools; crossed stage-spot shadows under the block pile; glowing balls bloom and roll; heat map shows the 16×9 tiles and depth bands, hottest in the colonnade |
| WebGPU, Edit mode (temporary build) | a star at every light; clicking a spot's marker selects it, the inspector shows its Light section, and its cone draws |
| WebGL2, Lights demo | matches WebGPU (R32Uint grid and 16 KB light uniform both work); no console errors |
| WebGPU, Pit | unchanged in daylight |
| Native Vulkan debug (Lights, Glass) and release (Lights); `skinned_character` | no validation errors |
| solstice, Ashen Crown | `cargo check` clean |

Not done:
- **No point-light shadows.** Only spots cast shadows, at most 4, and
  each shadow pass draws every instance (culling is R7).
- **Clusters are binned conservatively.** Lights are binned by their
  screen-space box, so corner clusters list lights that can't reach them.
  The shader skips those after one distance test. A sphere–cluster test
  would tighten the lists if the heat map ever matters.
- **No glTF light import** (`KHR_lights_punctual`). The cone maths
  already matches it.
- **Local lights don't scatter in fog,** and translucent objects still
  cast no shadow.
- **Frame-time cost not measured with GPU timestamps** (R1–R5).
