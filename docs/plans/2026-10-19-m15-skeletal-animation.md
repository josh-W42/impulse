# M15 — Skeletal Animation: Design + Results

**Date:** 2026-10-19 · **Status:** complete · **Driven by:** Ashen Crown
(`../ashen-crown/docs/plans/2026-10-19-m15-animation-and-models.md`), which
replaces its capsules with rigged, animated low-poly characters.

## Why

`impulse-assets` baked every node transform into one rigid mesh and ignored
skins and animations, so a rigged character showed in its rest pose (usually
a T-pose). A game whose combat is read from who is casting, swinging or
falling needs bodies that move.

## What shipped

- **Skinned models** (`impulse-assets`, `load_skinned_glb`):
  - **Vertices:** stay in bind space. The skinned mesh node's own transform
    is ignored, as glTF specifies.
  - **Joints:** come parents first, with `JOINTS_0` remapped to that order.
    Each joint keeps its rest transform (what clips replace) and inverse bind
    matrix.
  - **Root joints:** carry the world transform of the non-joint nodes above
    them as `base`. Exported armatures often have one, such as the
    Quaternius wolf's −90° X and ×100 scale.
  - **Rigid meshes under a joint:** a helmet parented to the head becomes
    skinned to that joint with full weight.
  - **Weights:** normalized. All-zero weights go fully to joint 0.
  - **Limits:** one skin per file, at most 128 joints (`Limits::max_joints`).
- **Clips** (`load_clips`, and `SkinnedModelData::clips` for clips in the
  model's own file):
  - **Matching:** channels name their target node, so clips from one file
    drive a skeleton from another. Asset packs ship them separately, and
    KayKit's joint order differs between its files.
  - **Interpolation:** step and linear. Cubic splines keep each key's value.
  - Morph-weight channels are skipped.
- **`impulse-anim`**, a new headless crate (glam + impulse-assets only):
  - `BoundClip`: binds a clip to a skeleton by joint name, counts what didn't
    match, and samples at a time (clamped, shortest-arc rotation);
  - `Pose` and `blend`;
  - `model_transforms` and `skinning_matrices`;
  - `Animator`: plays looping or once, sets speed, crossfades between clips,
    and has `restart` (the same clip again) and `seek` (start part-way).
- **GPU skinning** (`impulse-render`):
  - `Renderer::add_skinned_mesh`;
  - `DrawList::push_joints` (one model's matrices) and `push_skinned` (each
    of its meshes, sharing the range);
  - `vs_skinned` and `vs_shadow_skinned` blend four joint matrices per
    vertex, in both the lit and the shadow pass.
- **Engine:**
  - `impulse_engine::upload::{upload_model, upload_skinned_model,
    TextureCache}` (the sandbox's model upload now goes through
    `upload_model`);
  - the re-exports `impulse_engine::assets` and `impulse_engine::anim`;
  - the example `skinned_character`, which plays the bundled KayKit
    mannequin's clips with crossfades.

## WebGL2 decided the shapes

- **No storage buffers in the vertex stage:**
  - joint matrices go in an `Rgba32Float` texture, 1024 texels wide, four
    texels (the columns) per matrix, read with `textureLoad`;
  - each skinned instance carries its first matrix's index;
  - the texture grows by powers of two in rows, up to the guaranteed 2048:
    about 500k matrices. Past that, a frame's skinned meshes aren't drawn and
    a warning is logged once.
- **16 vertex attributes at most:**
  - the existing static layout used 13;
  - skinned meshes add joints (`Uint16x4`) and weights, and the skinned
    instance adds the joint offset (`Uint32`): exactly 16.
  - A further per-vertex attribute needs packing first.
- **Normals:** take the blended skin matrix's 3×3 part (joints rotate,
  translate and scale uniformly), then the instance's normal matrix.

## Tests

- **`impulse-assets`:**
  - bind-space vertices;
  - parent-first joints and remapped indices;
  - rigid meshes under a joint;
  - `base`;
  - weight normalization;
  - skin errors;
  - clips by name, step and cubic interpolation, animation errors, clips in
    mesh-less files and in skinned files;
  - the bundled mannequin: 23 joints, 15 clips, every channel bound.
- **`impulse-anim`:** linear, step and clamped sampling; the shortest arc;
  unmatched channels; joints without channels; rest pose gives identity
  skinning; `base` applied once; blending; looping, one-shots, crossfades,
  `play` vs `restart`, speed and `seek`.
- **`impulse-render`:**
  - joint ranges;
  - texture rows (one even with nothing skinned);
  - texel layout;
  - grouping skinned instances;
  - `vs_skinned` and `vs_shadow_skinned` validated by naga and translated to
    WebGL2 GLSL.
- **Example:** `skinned_character` runs natively with no wgpu validation
  errors. The browser check is Ashen Crown's (below).

## Results

Checked in Ashen Crown's browser build, in a Chromium-based browser pane, by
screenshots:
- **WebGPU:** a skinned Knight standing, running and swinging, with two
  wolves.
- **WebGL2:** a Mage, plus eight different skinned models at once (24
  instances, alive, casting and dead), with weapons attached at hand joints
  through `model_transforms`.
- No validation errors on either backend.
- **Frame time:** at the game's camp, WebGL2 held 60 Hz (16.68 ms mean,
  16.80 ms p95) both with and without skinned models. vsync bounds both, so
  this shows no regression rather than the headroom.

What the game's checks changed here: nothing in the engine.
- The problems found were asset-side (prop orientation in an unarmed idle,
  model scale) and were solved in the game.
- One loader rule came from real files: KayKit's skeleton helmets are
  unskinned meshes under the head joint, so the loader skins them to it.

## Not done

- Blending clips per body part (upper body casting while legs run).
- Root motion, inverse kinematics, morph targets.
- Skinned models in the sandbox's scene editor.
