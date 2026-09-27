# Benchmarks

Measured 2026-09-25 on the development machine (Windows 11, Vulkan native).
Numbers are for comparison between versions on one machine, not absolute claims.

## Native physics step time

`cargo run --release -p impulse-physics --example bench` — N mixed spheres and boxes
dropped into a walled pit; every step of a 5 s simulation (300 steps at 60 Hz) timed.

| bodies | brute force (M4) mean / p95 | + AABB-tree broadphase mean / p95 | + island sleeping mean / p95 |
|---|---|---|---|
| 100 | 0.42 / 0.55 ms | 0.32 / 0.38 ms | 0.08 / 0.36 ms |
| 500 | 4.46 / 5.02 ms | 1.75 / 2.24 ms | 0.56 / 1.89 ms |
| 1000 | 14.10 / 14.99 ms | 3.15 / 3.75 ms | **1.24 / 3.43 ms** |

- Brute force scales ~O(n²) (10× bodies → 33× time); the broadphase is near-linear
  (10× → 10×) and 4.5× faster at 1,000 bodies.
- Sleeping cuts the **mean** a further 2.5× once piles settle; the **p95** is the
  falling phase, when every body is awake — that's the number to budget against.
- Contacts are bit-identical with and without the broadphase (equivalence test).

## Solver comparison (M11, native)

Measured 2026-09-26, same bench; it now runs under each solver. Soft step:
4 substeps, 30 Hz contacts, relax each substep (see `docs/plans/2026-09-27-m11-soft-step.md`).

| bodies | classic mean / p95 / max | soft step mean / p95 / max |
|---|---|---|
| 100 | 0.09 / 0.36 / 0.44 ms | 0.11 / 0.47 / 0.56 ms |
| 500 | 0.61 / 1.92 / 2.21 ms | 0.78 / 2.55 / 2.74 ms |
| 1000 | 1.27 / 3.46 / 3.78 ms | 2.43 / 4.64 / 5.32 ms |

- Soft step costs ~1.3× classic at p95 (the falling phase) and ~1.9× in the
  1,000-body mean: four substeps of one solve plus one relax (8 contact passes)
  against classic's 10 iterations, but contact geometry is recomputed each pass.
- Well inside a 16.7 ms frame; what it buys is stacking (10-high tower: 5.5 mm
  drift vs 7.8 cm, see the M11 doc).

## Browser frame time (Rain scene, 1,000 bodies)

`?scene=rain`, 1005 bodies incl. pit planes, 10 solver iterations, warm starting and
sleeping on. Method (as in the telemetry viewer): visible tab, frame intervals
collected in-page with `requestAnimationFrame` over 5 s windows, one read at the end
(no mid-run polling). Embedded Chromium browser pane, vsync-locked display.

| backend | falling (0–5 s) mean / p95 / max | settled (10–15 s) mean / p95 / max |
|---|---|---|
| WebGPU | 16.67 / 16.8 / 16.8 ms | 16.67 / 16.8 / 16.8 ms |
| WebGL2 | 16.67 / 16.8 / 16.8 ms | 16.67 / 16.8 / 16.9 ms |

Both hold a locked 60 fps. The in-app profiler (WebGPU, settled) read physics
2.9 ms/frame (broadphase 1.1 · narrowphase 0.4 · solve 1.3 · integrate 0.1),
7,380 candidate pairs, 2,522 contacts (1,405 solved), 560 awake / 440 asleep.

## Caveats

- Frame times are vsync-bound; they show the target is met, not the headroom. The
  profiler's physics time is the headroom indicator.
- One machine, one browser. A slower laptop or phone will differ; the wasm build is
  single-threaded.
