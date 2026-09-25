# impulse

A 3D game engine in Rust + wgpu whose deep spike is rigid-body physics.
Runs natively and in the browser (WebGPU, falling back to WebGL2).

Design: `docs/plans/2026-09-24-impulse-engine-design.md`

## Prerequisites

- rustup (the pinned toolchain in `rust-toolchain.toml` installs automatically)
- wasm-bindgen CLI matching `Cargo.lock`:
  `cargo install --locked wasm-bindgen-cli --version "$(scripts/wasm-bindgen-version.sh)"`
- Python 3 (for serving the web build locally)

## Run

Desktop: `cargo run -p impulse-demo`

Browser:

    scripts/build-web.sh
    python -m http.server 8080 --directory web/dist

Open http://localhost:8080. Add `?backend=webgl` to force the WebGL2 fallback.

## Checks

    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo clippy --workspace --target wasm32-unknown-unknown -- -D warnings
    cargo test --workspace
    scripts/check-physics-deps.sh

## Layout

- `crates/physics`: rigid-body physics; depends on `glam` only (enforced)
- `crates/render`: wgpu renderer; no windowing
- `crates/engine`: winit loop, platform glue
- `crates/demo`: the demo binary
- `web/`: HTML shell for the browser build
