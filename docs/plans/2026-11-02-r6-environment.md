# R6 — Environment Lighting: Plan + Results

**Date:** 2026-11-02 · **Status:** built, awaiting hand check · **Track:**
[render quality](2026-11-02-render-quality-track.md)

## Why

The sky is a three-colour gradient. Metals reflect a smooth blur instead
of a world, and ambient light has no direction beyond up versus down.

## Assets

Two 1K equirectangular HDRIs from Poly Haven, CC0. Both were downloaded
with approval on 2026-11-02 and match Poly Haven's published MD5s.

| File | Size | Look | Authors |
| --- | --- | --- | --- |
| `kloofendal_48d_partly_cloudy_puresky_1k.hdr` | 1.44 MB | midday, clouds, sun at 48° | Greg Zaal, Jarod Guest |
| `belfast_sunset_puresky_1k.hdr` | 1.19 MB | sunset, sun in haze at 2° | Dimitrios Savva, Greg Zaal, Jarod Guest |

Their luminance, measured before the design:
- **Kloofendal** has a compact sun: a peak of about 60,000, with 17
  texels over 200.
- **Belfast's** sun is a broad haze: a peak of 24.5, with about 3,800
  texels over 10.
- Both give about 3.4–4.8 of irradiance on an up-facing surface, close
  to the procedural day sky's ~3.7, so no rescaling is needed.

Bundling both adds about 2.6 MB to the web build.

## Design

- **Decoding:** `impulse-assets::load_hdr` uses the `image` crate's
  Radiance decoder. The crate is already a dependency; this turns on its
  `hdr` feature, which adds no crates. Size limits apply as for model
  textures.
- **Baking on the CPU, once per sky** (`impulse-render::Environment::bake`).
  The roadmap said render passes. CPU baking is testable without a GPU,
  identical on every backend, and needs no float render targets on
  WebGL2. The price is a one-off bake time when a scene switches sky.
  1. **Sun extraction.** Texels brighter than 50 × the median luminance
     are clamped there. Their excess energy becomes the sun's colour
     (irradiance), and their weighted centre its direction. The sun then
     lights and shadows the scene as before, and the map has no
     firefly-making hot spot. With no compact sun (Belfast), the sun is
     left at zero, pointing at the brightest texel, for the caller to
     fill.
  2. **Cube map:** 256² faces with 6 mips (256 → 8), filled from the
     equirect.
  3. **GGX prefiltering:** mip *i* is the environment as seen by
     roughness *i*/5. It uses importance sampling with filtered lookups
     into a box-filtered pyramid (Colbert & Křivánek), so 64 samples per
     texel come out smooth.
  4. **SH9 irradiance:** projected from a blurred mip with the existing
     `project_irradiance`.
  5. **Upload format:** `Rgba16Float`. It's filterable everywhere,
     WebGL2 included. The f32 → f16 conversion is ours.
- **Renderer:** `set_environment(Option<&Environment>)`. `None` returns
  to the procedural `Sky`, which stays the default and the fallback.
  - The cube is bound at group 0, with a 1×1 black cube while unused.
  - Specular reads the prefiltered mip for the surface's roughness.
  - Diffuse uses the environment's SH.
  - The background draws mip 0, plus the sun disc where the sun was
    extracted.
  - Fog takes its colour from a blurred mip along the view.
  - The sky pass moves to the scene bind group, which holds the cube.
- **Engine:**
  - `SkyPreset` gains **Clouds** (Kloofendal) and **Sunset** (Belfast).
    Sunset supplies a warm sun at the brightest point of its haze.
  - Bakes are cached per preset, so switching back is free.
  - The Materials demo, the PBR chart, moves to Clouds so its metals
    reflect the sky.
  - Licences go in `assets/env/LICENSES.md`.

## Tasks

1. Assets: `load_hdr`, with tests for both bundled files, a garbage
   file, and the size limit.
2. Render:
   - `Environment::bake` and the f16 conversion;
   - tests: direction ↔ texel round trips, a constant sky prefilters to
     itself at every roughness with irradiance π·L, a synthetic sun is
     found and its energy kept, and Kloofendal's sun is found near 48°.
3. Renderer: the cube texture, `set_environment`, the shader paths, and
   the sky pass on the scene bind group.
4. Engine: Clouds and Sunset presets, the bake cache, the Materials demo,
   licences and the README.
5. Checks: fmt, clippy (native + wasm), tests, and `cargo check` on
   solstice and Ashen Crown.
6. Verify on WebGPU, WebGL2 and native: Materials reflections, the
   Sunset scene, the procedural skies unchanged, and the bake time on web.

## Results

Built as designed, with these changes:

- **Decoding uses `image`'s HDR codec** (the `hdr` feature) instead of a
  hand-written reader.
- **Prefilter sample counts scale with roughness:** 16, 32, then 64. The
  first prefiltered level, a narrow lobe on 128² faces, took 226 of 372 ms
  at 64 samples, and filtered lookups keep it smooth at 16.
  - The bake now takes **about 220 ms in the browser** (logged at load),
    once per sky per session.
- **Sunset's supplied sun is raised to 12°.** At the haze's true 2°, every
  shadow streaked across the map.
- **Fixed an R3 bug.** With AO off (the panel toggle, or WebGL2 without
  float targets), the lit shader read its 1×1 white AO fallback at pixel
  coordinates. Every read out of bounds is undefined, and the whole frame
  turned to noise. The read is now clamped to the texture.

| Check | Result |
| --- | --- |
| Unit tests: assets | both bundled skies decode (1024×512, bright peaks); garbage, empty, square and over-limit files refused; values round-trip |
| Unit tests: render | half floats round to nearest; directions survive equirect and cube round trips (and match the GPU's face convention); a uniform sky stays uniform at every roughness with irradiance π·L; a bright spot becomes the sun with its energy kept; rough levels gather a bright band that sharp ones don't; Kloofendal's sun found at 48° with strength 2–10; the sunset's left to the caller, low over the horizon |
| Unit tests: engine | only HDR presets bake, and every one has a sun above the ground; raising keeps the bearing and only lifts low suns |
| Shaders → WebGL2 GLSL (naga) | `fs_main`, `fs_sky` with the cube map: ok |
| clippy native + wasm32 (`-D warnings`), workspace tests (25 suites, 384) | clean / all green |
| WebGPU, Materials (clouds) | the background is the photo; smooth metals reflect the clouds, rough ones a blur; looking along the extracted sun, the photo's sun sits under the drawn disc, so the cube orientation is right |
| WebGPU, Materials (sunset, temporary build) | warm, hazy, lit from the low sun |
| WebGPU, AO off (temporary build) | noise before the fix, clean after |
| WebGL2, Materials (clouds) | matches WebGPU |
| WebGPU, Pit (day) and Lights (dusk) | unchanged |
| Native Vulkan debug and release (Materials, Lights); `skinned_character` | no validation errors |
| solstice, Ashen Crown | `cargo check` clean |

Not done, or found along the way:
- **Faint streaks inside shadows with AO on.** They're present under the
  old procedural sky too, gone with AO off, and strongest when ambient
  light dominates. This is an R3 GTAO artifact to fix separately.
- **The pure-sky photos' lower half is a blurred sky, not ground.** Beyond
  the floor's edge the horizon looks like open air, and ambient light from
  below is sky-blue. Swapping the lower hemisphere for a ground colour
  would need a decision on how.
- **The web build grows from 12.1 to 14.8 MB of wasm** (the two skies).
- **Frame-time cost not measured with GPU timestamps** (R1–R6).
