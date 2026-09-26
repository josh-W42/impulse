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
