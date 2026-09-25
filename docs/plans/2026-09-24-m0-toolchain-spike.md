# M0 Toolchain Spike Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** A Cargo workspace whose demo clears the screen with wgpu natively and in the browser (WebGPU, with WebGL2 fallback), built and deployed to Firebase Hosting by GitHub Actions.

**Architecture:** Four crates, one-directional deps: `impulse-demo` (bin) → `impulse-engine` (winit loop, platform glue) → `impulse-render` (wgpu `Gpu`, no winit); `impulse-physics` is an empty lib guarded by a CI script to depend on `glam` only. Async GPU init runs via `pollster::block_on` natively and `wasm_bindgen_futures::spawn_local` on the web, delivered back to the event loop as a winit user event, following wgpu v30's own `examples/features/src/framework.rs`.

**Tech Stack:** Rust 1.98.1 (edition 2024), wgpu 30.0.1 (+`webgl` feature), winit 0.30.13, wasm-bindgen 0.2.128 + CLI, web-sys 0.3.105, GitHub Actions, Firebase Hosting.

Design doc: `docs/plans/2026-09-24-impulse-engine-design.md`.

---

## Verified facts this plan depends on (checked 2026-09-24)

- Latest compatible set: `wgpu 30.0.1`, `winit 0.30.13`, `egui/egui-wgpu/egui-winit 0.36.2` (egui-wgpu 0.36.2 requires `wgpu ^30.0`, `winit ^0.30.13`). egui is **not** added in M0; it lands in M1 at 0.36.x.
- wgpu 30 API (from `v30.0.1` tag source): `InstanceDescriptor::new_with_display_handle(Box<dyn WgpuHasDisplayHandle>)`, `.backends` field; `wgpu::util::new_instance_with_webgpu_detection(desc).await` drops `BROWSER_WEBGPU` when the browser lacks it; `request_adapter`/`request_device` return `Result`; `Surface::get_default_config` returns `Option`; `get_current_texture()` returns the exhaustive enum `CurrentSurfaceTexture { Success, Suboptimal, Occluded, Timeout, Outdated, Lost, Validation }`; presentation is `queue.present(surface_texture)`; `RenderPassDescriptor` has `multiview_mask`; `Backend` is exhaustive: `Noop, Vulkan, Metal, Dx12, Gl, BrowserWebGpu`.
- wgpu default features include `webgpu` but **not** `webgl`; the WebGL2 fallback needs `features = ["webgl"]`.
- On wasm the surface must be created **before** requesting the adapter (WebGL needs the canvas).
- wgpu MSRV 1.93; current stable 1.98.1.
- `console_log 1.1.0`: `init_with_level(log::Level) -> Result<(), SetLoggerError>`.
- The wasm-bindgen CLI version must exactly equal the `wasm-bindgen` crate version in `Cargo.lock`.

## Known constraint to carry into M1

Requesting `Limits::downlevel_webgl2_defaults()` keeps every backend on the WebGL2 feature floor: **no compute shaders and no storage buffers**. Instancing via per-instance vertex buffers works. M1 must decide whether to keep WebGL2 fallback (and that floor) or require WebGPU.

**Native live-resize artifacts (observed 2026-09-25, Windows + Vulkan):** while dragging a window edge, the newly exposed strip flashes black when growing and white when shrinking; dragging the left edge shows it on the right side. No crash, correct final image. Likely cause: presented frames lag the window size by the swapchain's frame latency. Revisit in M1 when the depth buffer also needs resize handling. Candidates to test (unverified): render+present synchronously inside `WindowEvent::Resized`; `desired_maximum_frame_latency = 1`; compare the DX12 backend (DXGI flip model).

**Browser findings from Task 10 (2026-09-25):**
- A *hidden* browser tab reports a 1x1 drawing buffer: hidden documents skip the rendering steps, so winit's `ResizeObserver` never fires. It corrects itself when the tab becomes visible. A suspected init/resize race was tested (fix applied, then reverted and re-tested visible) and does **not** exist. Always verify canvas size with the page visible.
- In the pane's mobile emulation (`devicePixelRatio=2`) the buffer matched CSS pixels (375x812), not physical pixels (750x1624). This may be an emulation artifact. **Verify on a real high-DPI device in M1** before changing anything; if real, the scene renders at half resolution.
- Without the `impulse-canvas` element, winit renders into a detached canvas: the console logs an error but the status still reads "Running on …". This only happens if our own `index.html` is broken, so it was left alone.
- The release `.wasm` is ~4.0 MB uncompressed (wgpu with both WebGPU and WebGL backends plus naga) but **801 KB transferred** from Firebase Hosting (compressed), plus 15 KB JS. Acceptable; no `wasm-opt`/LTO needed yet.

**M0 outcome (2026-09-25):** live at https://impulse-engine.web.app on WebGPU and forced WebGL2 (`?backend=webgl`). CI: 6m29s cold, 1m30s warm (rust-cache also keeps the wasm-bindgen CLI). The deploy secret is `FIREBASE_SERVICE_ACCOUNT_IMPULSE_ENGINE` (firebase-tools adds the project suffix); its generated PR-preview workflow was deleted because it deploys without building. GitHub notes `ubuntu-latest` moves to Ubuntu 26 from 2026-10-19; watch the first CI run after that date.

## Unverified points (confirm during execution; stop and report if wrong)

1. `rustup toolchain install` with no arguments installs the toolchain named in `rust-toolchain.toml` (rustup ≥ 1.28). Fallback: `rustup toolchain install 1.98.1 --component rustfmt clippy --target wasm32-unknown-unknown`.
2. wasm-bindgen runs a **bin** crate's `fn main` as the wasm start function (wgpu's `run-wasm` xtask builds bins this way). Verified for real in Task 10.
3. `ubuntu-latest` compiles winit 0.30 without extra apt packages (x11/wayland are dlopen'd). If CI fails on missing system libs, add an apt step.

## User actions (Claude must ask before doing any of these)

- Installing rustup (Task 0).
- Creating the GitHub repository and pushing (Task 12).
- Creating the Firebase project, service account, and repo secret/variable (Task 13).

---

### Task 0: Install the Rust toolchain

**Files:** none.

**Step 1: Install rustup** (ask the user first; MSVC Build Tools 2022 are already installed, which the MSVC Rust toolchain needs)

Run (PowerShell): `winget install --id Rustlang.Rustup -e`
Then open a **new** shell so `~/.cargo/bin` is on `PATH`.

**Step 2: Verify**

Run: `rustup --version` and `cargo --version`
Expected: both print versions. (The pinned 1.98.1 toolchain installs in Task 1.)

---

### Task 1: Workspace skeleton

**Files:**
- Create: `Cargo.toml`, `rust-toolchain.toml`, `.gitignore`
- Create: `crates/physics/Cargo.toml`, `crates/physics/src/lib.rs`
- Create: `crates/render/Cargo.toml`, `crates/render/src/lib.rs`
- Create: `crates/engine/Cargo.toml`, `crates/engine/src/lib.rs`
- Create: `crates/demo/Cargo.toml`, `crates/demo/src/main.rs`

**Step 1: Root files**

`Cargo.toml`:

```toml
[workspace]
resolver = "3"
members = ["crates/*"]

[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.98"
publish = false

[workspace.dependencies]
impulse-render = { path = "crates/render" }
impulse-engine = { path = "crates/engine" }

wgpu = { version = "30.0.1", features = ["webgl"] }
winit = "0.30.13"
log = "0.4.34"

# native only
env_logger = "0.11.11"
pollster = "0.4"

# wasm only
wasm-bindgen = "0.2.128"
wasm-bindgen-futures = "0.4.78"
web-sys = "0.3.105"
console_error_panic_hook = "0.1.7"
console_log = "1.1.0"
```

`rust-toolchain.toml`:

```toml
[toolchain]
channel = "1.98.1"
components = ["rustfmt", "clippy"]
targets = ["wasm32-unknown-unknown"]
```

`.gitignore`:

```
/target
/web/dist
/.firebase
firebase-debug.log
```

**Step 2: Crate stubs**

`crates/physics/Cargo.toml`:

```toml
[package]
name = "impulse-physics"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
publish.workspace = true

[dependencies]
```

`crates/physics/src/lib.rs`:

```rust
//! Rigid-body physics for the impulse engine.
//!
//! Architecture rule: this crate depends on `glam` only, never on rendering,
//! windowing, or engine crates, so it can be tested, benchmarked, and reused
//! headless. `scripts/check-physics-deps.sh` enforces this in CI.
```

`crates/render/Cargo.toml`:

```toml
[package]
name = "impulse-render"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
publish.workspace = true

[dependencies]
wgpu.workspace = true
log.workspace = true
```

`crates/render/src/lib.rs`:

```rust
//! GPU context and frame rendering for the impulse engine.
//!
//! Independent of windowing: the engine creates the surface from its window and
//! hands it over. Re-exports `wgpu` so every crate uses the same version.

pub use wgpu;
```

`crates/engine/Cargo.toml`:

```toml
[package]
name = "impulse-engine"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
publish.workspace = true

[dependencies]
impulse-render.workspace = true
winit.workspace = true
log.workspace = true

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
env_logger.workspace = true
pollster.workspace = true

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen.workspace = true
wasm-bindgen-futures.workspace = true
console_error_panic_hook.workspace = true
console_log.workspace = true
web-sys = { workspace = true, features = [
    "Document",
    "Element",
    "HtmlCanvasElement",
    "Location",
    "Node",
    "Window",
] }
```

`crates/engine/src/lib.rs`:

```rust
//! Engine glue for impulse: window, event loop, and GPU startup on desktop and web.
```

`crates/demo/Cargo.toml`:

```toml
[package]
name = "impulse-demo"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
publish.workspace = true

[dependencies]
impulse-engine.workspace = true
```

`crates/demo/src/main.rs` (temporary; replaced in Task 8):

```rust
fn main() {}
```

**Step 3: Build both targets**

Run: `cargo check --workspace` then `cargo check --workspace --target wasm32-unknown-unknown`
Expected: the first run installs toolchain 1.98.1 (see Unverified point 1), then both checks finish with no errors. `Cargo.lock` is created.

**Step 4: Commit**

```bash
git add Cargo.toml Cargo.lock rust-toolchain.toml .gitignore crates
git commit -m "chore: scaffold impulse cargo workspace"
```

---

### Task 2: Physics dependency guard

This script enforces the design's central rule. We test the guard by making it fail on purpose.

**Files:**
- Create: `scripts/check-physics-deps.sh`

**Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Enforces the core architecture rule: impulse-physics may depend on glam only.
set -euo pipefail
cd "$(dirname "$0")/.."

deps="$(cargo tree -p impulse-physics -e normal --depth 1 --prefix none \
  | tail -n +2 | awk '{print $1}' | sort -u)"
bad="$(printf '%s\n' "$deps" | grep -vxE 'glam|' || true)"

if [ -n "$bad" ]; then
  echo "impulse-physics must depend only on glam; found: $bad" >&2
  exit 1
fi
echo "impulse-physics dependencies OK: ${deps:-none}"
```

Run: `git add scripts/check-physics-deps.sh && git update-index --chmod=+x scripts/check-physics-deps.sh`

**Step 2: Prove it fails on a forbidden dependency**

Temporarily add `log.workspace = true` under `[dependencies]` in `crates/physics/Cargo.toml`.
Run: `bash scripts/check-physics-deps.sh`
Expected: exit 1, `impulse-physics must depend only on glam; found: log`

**Step 3: Revert and confirm it passes**

Remove the `log` line.
Run: `bash scripts/check-physics-deps.sh`
Expected: `impulse-physics dependencies OK: none`

**Step 4: Commit**

```bash
git add scripts/check-physics-deps.sh
git commit -m "chore: guard impulse-physics against non-glam dependencies"
```

---

### Task 3: `clamp_surface_size` (TDD)

**Files:**
- Create: `crates/render/src/size.rs`
- Modify: `crates/render/src/lib.rs`

**Step 1: Write the failing tests**

`crates/render/src/size.rs`:

```rust
/// Clamps a requested surface size into the range the device can present.
///
/// Surfaces must be at least 1x1 (a minimized window reports 0x0) and no larger
/// than the device's `max_texture_dimension_2d`.
pub fn clamp_surface_size(width: u32, height: u32, max_dimension: u32) -> (u32, u32) {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::clamp_surface_size;

    #[test]
    fn passes_through_valid_sizes() {
        assert_eq!(clamp_surface_size(1280, 720, 8192), (1280, 720));
    }

    #[test]
    fn zero_size_becomes_one_pixel() {
        assert_eq!(clamp_surface_size(0, 0, 8192), (1, 1));
    }

    #[test]
    fn oversized_is_limited_to_device_max() {
        assert_eq!(clamp_surface_size(10_000, 300, 4096), (4096, 300));
    }

    #[test]
    fn zero_device_limit_does_not_panic() {
        assert_eq!(clamp_surface_size(800, 600, 0), (1, 1));
    }
}
```

Append to `crates/render/src/lib.rs`:

```rust
mod size;

pub use size::clamp_surface_size;
```

**Step 2: Run and watch them fail**

Run: `cargo test -p impulse-render size`
Expected: 4 tests FAIL with `not yet implemented`.

**Step 3: Implement**

Replace `todo!()` with:

```rust
    let max = max_dimension.max(1);
    (width.clamp(1, max), height.clamp(1, max))
```

**Step 4: Run and watch them pass**

Run: `cargo test -p impulse-render size`
Expected: 4 passed.

**Step 5: Commit**

```bash
git add crates/render
git commit -m "feat(render): clamp surface size to device limits"
```

---

### Task 4: `backend_name` (TDD)

This is the label shown on the page and in the window title, and it is how we verify which backend actually ran.

**Files:**
- Create: `crates/render/src/backend.rs`
- Modify: `crates/render/src/lib.rs`

**Step 1: Write the failing tests**

`crates/render/src/backend.rs`:

```rust
/// Human-readable name of the graphics backend wgpu selected.
pub fn backend_name(backend: wgpu::Backend) -> &'static str {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::backend_name;

    #[test]
    fn names_native_backends() {
        assert_eq!(backend_name(wgpu::Backend::Vulkan), "Vulkan");
        assert_eq!(backend_name(wgpu::Backend::Dx12), "DirectX 12");
        assert_eq!(backend_name(wgpu::Backend::Metal), "Metal");
    }

    #[test]
    fn names_browser_webgpu() {
        assert_eq!(backend_name(wgpu::Backend::BrowserWebGpu), "WebGPU");
    }

    #[test]
    fn gl_is_opengl_on_desktop() {
        // Tests run natively; on wasm32 the same backend is reported as "WebGL2".
        assert_eq!(backend_name(wgpu::Backend::Gl), "OpenGL");
    }
}
```

Append to `crates/render/src/lib.rs`:

```rust
mod backend;

pub use backend::backend_name;
```

**Step 2: Run and watch them fail**

Run: `cargo test -p impulse-render backend`
Expected: 3 tests FAIL with `not yet implemented`.

**Step 3: Implement** (list every variant, no wildcard; `Backend` is exhaustive)

```rust
    match backend {
        wgpu::Backend::BrowserWebGpu => "WebGPU",
        wgpu::Backend::Gl if cfg!(target_arch = "wasm32") => "WebGL2",
        wgpu::Backend::Gl => "OpenGL",
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Dx12 => "DirectX 12",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Noop => "no-op",
    }
```

**Step 4: Run and watch them pass**

Run: `cargo test -p impulse-render`
Expected: 7 passed.

**Step 5: Commit**

```bash
git add crates/render
git commit -m "feat(render): human-readable backend names"
```

---

### Task 5: `Gpu`: device setup and clear-screen frame

GPU init can't be unit-tested without a device, so this task is verified by compiling it for both targets here and by running it in Tasks 8 and 10.

**Files:**
- Create: `crates/render/src/gpu.rs`
- Modify: `crates/render/src/lib.rs`

**Step 1: Write `gpu.rs`**

```rust
use std::fmt;

use crate::size::clamp_surface_size;

/// Why GPU initialization failed. Displayed to the user, so messages are plain language.
#[derive(Debug)]
pub enum GpuInitError {
    Surface(wgpu::CreateSurfaceError),
    Adapter(wgpu::RequestAdapterError),
    Device(wgpu::RequestDeviceError),
    SurfaceUnsupported,
}

impl fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Surface(e) => write!(f, "Could not create a rendering surface: {e}"),
            Self::Adapter(e) => write!(
                f,
                "No compatible GPU found (this demo needs WebGPU or WebGL2): {e}"
            ),
            Self::Device(e) => write!(f, "The GPU refused to create a device: {e}"),
            Self::SurfaceUnsupported => write!(f, "The GPU cannot present to this surface"),
        }
    }
}

impl std::error::Error for GpuInitError {}

/// Result of trying to draw one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameOutcome {
    /// The frame was drawn and presented.
    Presented,
    /// Nothing was drawn (occluded, timed out, or the surface was reconfigured); try next frame.
    Skipped,
    /// The surface is gone; create a new one and pass it to [`Gpu::replace_surface`].
    SurfaceLost,
}

/// Owns the wgpu device and queue and the surface being presented to.
pub struct Gpu {
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    view_format: wgpu::TextureFormat,
    backend: wgpu::Backend,
}

impl Gpu {
    /// Picks an adapter compatible with `surface`, creates a device, and configures the surface.
    ///
    /// Requests WebGL2-level limits so the same code runs on every backend,
    /// including the browser's WebGL2 fallback.
    pub async fn new(
        instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> Result<Self, GpuInitError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(GpuInitError::Adapter)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("impulse device"),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(GpuInitError::Device)?;

        let (width, height) =
            clamp_surface_size(width, height, device.limits().max_texture_dimension_2d);
        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or(GpuInitError::SurfaceUnsupported)?;
        // Render through an sRGB view so colors are gamma-correct even when the
        // surface format is not sRGB (browser WebGPU surfaces usually are not).
        let view_format = config.format.add_srgb_suffix();
        config.view_formats.push(view_format);
        surface.configure(&device, &config);

        Ok(Self {
            instance,
            surface,
            device,
            queue,
            config,
            view_format,
            backend: adapter.get_info().backend,
        })
    }

    pub fn backend(&self) -> wgpu::Backend {
        self.backend
    }

    /// The instance, needed to recreate the surface after [`FrameOutcome::SurfaceLost`].
    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    /// Reconfigures the surface for a new window size.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) =
            clamp_surface_size(width, height, self.device.limits().max_texture_dimension_2d);
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    /// Swaps in a freshly created surface after [`FrameOutcome::SurfaceLost`].
    pub fn replace_surface(&mut self, surface: wgpu::Surface<'static>) {
        self.surface = surface;
        self.surface.configure(&self.device, &self.config);
    }

    /// Clears the screen to `color` and presents it.
    pub fn render_clear(&mut self, color: wgpu::Color) -> FrameOutcome {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                drop(frame);
                self.surface.configure(&self.device, &self.config);
                return FrameOutcome::Skipped;
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return FrameOutcome::Skipped;
            }
            wgpu::CurrentSurfaceTexture::Occluded | wgpu::CurrentSurfaceTexture::Timeout => {
                return FrameOutcome::Skipped;
            }
            wgpu::CurrentSurfaceTexture::Lost => return FrameOutcome::SurfaceLost,
            wgpu::CurrentSurfaceTexture::Validation => {
                log::error!("validation error while acquiring the surface texture");
                return FrameOutcome::Skipped;
            }
        };

        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.view_format),
            ..Default::default()
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
        FrameOutcome::Presented
    }
}
```

Append to `crates/render/src/lib.rs`:

```rust
mod gpu;

pub use gpu::{FrameOutcome, Gpu, GpuInitError};
```

**Step 2: Compile both targets, lint clean**

Run: `cargo clippy -p impulse-render --all-targets -- -D warnings` then `cargo clippy -p impulse-render --target wasm32-unknown-unknown -- -D warnings`
Expected: no warnings or errors. If a wgpu signature differs from the one above, check it against the `v30.0.1` tag source; don't guess.

**Step 3: Commit**

```bash
git add crates/render
git commit -m "feat(render): Gpu with adapter/device setup and clear-screen frame"
```

---

### Task 6: `?backend=` URL override parsing (TDD)

This lets us force WebGL2 in a browser that supports WebGPU, so the fallback path can actually be tested.

**Files:**
- Create: `crates/engine/src/backend.rs`
- Modify: `crates/engine/src/lib.rs`

**Step 1: Write the failing tests**

`crates/engine/src/backend.rs`:

```rust
//! The `?backend=` URL override used to test the WebGL2 fallback in browsers that support WebGPU.

use impulse_render::wgpu;

/// A user-forced choice of browser graphics backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendOverride {
    WebGpu,
    WebGl,
}

impl BackendOverride {
    pub fn backends(self) -> wgpu::Backends {
        match self {
            Self::WebGpu => wgpu::Backends::BROWSER_WEBGPU,
            Self::WebGl => wgpu::Backends::GL,
        }
    }
}

/// Reads `backend=webgpu` or `backend=webgl` from a URL query string such as `?backend=webgl`.
/// Returns `None` when the parameter is absent or has any other value.
pub fn parse_backend_override(query: &str) -> Option<BackendOverride> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_query_has_no_override() {
        assert_eq!(parse_backend_override(""), None);
    }

    #[test]
    fn reads_webgl_and_webgpu() {
        assert_eq!(parse_backend_override("?backend=webgl"), Some(BackendOverride::WebGl));
        assert_eq!(parse_backend_override("?backend=webgpu"), Some(BackendOverride::WebGpu));
    }

    #[test]
    fn finds_backend_among_other_params() {
        assert_eq!(
            parse_backend_override("?debug=1&backend=webgl&x=2"),
            Some(BackendOverride::WebGl)
        );
    }

    #[test]
    fn ignores_unknown_or_missing_values() {
        assert_eq!(parse_backend_override("?backend=vulkan"), None);
        assert_eq!(parse_backend_override("?backend"), None);
        assert_eq!(parse_backend_override("?backend="), None);
    }

    #[test]
    fn maps_to_wgpu_backends() {
        assert_eq!(BackendOverride::WebGl.backends(), wgpu::Backends::GL);
        assert_eq!(BackendOverride::WebGpu.backends(), wgpu::Backends::BROWSER_WEBGPU);
    }
}
```

Replace `crates/engine/src/lib.rs` with:

```rust
//! Engine glue for impulse: window, event loop, and GPU startup on desktop and web.

// The URL parser is only called by the web build; desktop builds just compile it.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
mod backend;
```

**Step 2: Run and watch them fail**

Run: `cargo test -p impulse-engine`
Expected: 4 FAIL with `not yet implemented`; `maps_to_wgpu_backends` passes.

**Step 3: Implement**

Replace `todo!()` with:

```rust
    query
        .trim_start_matches('?')
        .split('&')
        .find_map(|pair| match pair.split_once('=') {
            Some(("backend", "webgpu")) => Some(BackendOverride::WebGpu),
            Some(("backend", "webgl")) => Some(BackendOverride::WebGl),
            _ => None,
        })
```

**Step 4: Run and watch them pass**

Run: `cargo test -p impulse-engine`
Expected: 5 passed.

**Step 5: Commit**

```bash
git add crates/engine
git commit -m "feat(engine): parse ?backend= override for fallback testing"
```

---

### Task 7: Platform glue (native and web)

**Files:**
- Create: `crates/engine/src/platform/mod.rs`, `crates/engine/src/platform/native.rs`, `crates/engine/src/platform/web.rs`

**Step 1: `platform/mod.rs`**

```rust
//! Everything that differs between desktop and browser, behind one set of function names.

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::*;

#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(target_arch = "wasm32")]
pub use web::*;
```

**Step 2: `platform/native.rs`**

```rust
//! Desktop platform glue.

use std::future::Future;

use winit::window::{Window, WindowAttributes};

use crate::backend::BackendOverride;

pub fn init_logging() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn,naga=warn"),
    )
    .init();
}

/// Runs `future` to completion on the current thread.
pub fn spawn(future: impl Future<Output = ()> + 'static) {
    pollster::block_on(future);
}

pub fn window_attributes(title: &str) -> WindowAttributes {
    Window::default_attributes().with_title(title)
}

/// Backend overrides come from the page URL, so desktop builds never have one.
pub fn backend_override() -> Option<BackendOverride> {
    None
}

pub fn report_status(message: &str) {
    log::info!("{message}");
}

pub fn report_error(message: &str) {
    log::error!("{message}");
}
```

**Step 3: `platform/web.rs`**

```rust
//! Browser platform glue: canvas lookup, async spawning, and on-page status text.

use std::future::Future;

use wasm_bindgen::JsCast;
use winit::platform::web::WindowAttributesExtWebSys;
use winit::window::{Window, WindowAttributes};

use crate::backend::{BackendOverride, parse_backend_override};

/// Id of the `<canvas>` in `web/index.html` that the engine renders into.
const CANVAS_ID: &str = "impulse-canvas";
/// Id of the on-page element that shows the active backend or a startup error.
const STATUS_ID: &str = "status";

pub fn init_logging() {
    console_error_panic_hook::set_once();
    // An error here only means a logger is already installed.
    let _ = console_log::init_with_level(log::Level::Info);
}

/// Runs `future` on the browser's event loop; it completes after this returns.
pub fn spawn(future: impl Future<Output = ()> + 'static) {
    wasm_bindgen_futures::spawn_local(future);
}

fn document() -> Option<web_sys::Document> {
    web_sys::window()?.document()
}

pub fn window_attributes(title: &str) -> WindowAttributes {
    let canvas = document()
        .and_then(|doc| doc.get_element_by_id(CANVAS_ID))
        .and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok());
    if canvas.is_none() {
        log::error!("no <canvas id=\"{CANVAS_ID}\"> in the page; nothing will be visible");
    }
    Window::default_attributes().with_title(title).with_canvas(canvas)
}

pub fn backend_override() -> Option<BackendOverride> {
    let search = web_sys::window()?.location().search().ok()?;
    parse_backend_override(&search)
}

pub fn report_status(message: &str) {
    log::info!("{message}");
    set_status(message, false);
}

pub fn report_error(message: &str) {
    log::error!("{message}");
    set_status(message, true);
}

fn set_status(message: &str, is_error: bool) {
    if let Some(el) = document().and_then(|doc| doc.get_element_by_id(STATUS_ID)) {
        el.set_text_content(Some(message));
        el.set_class_name(if is_error { "error" } else { "" });
    }
}
```

**Step 4: Register the module**

Append to `crates/engine/src/lib.rs`:

```rust
mod platform;
```

(Unused-function warnings are expected until Task 8, so run a plain `check` here rather than `clippy -D warnings`.)

Run: `cargo check -p impulse-engine` then `cargo check -p impulse-engine --target wasm32-unknown-unknown`
Expected: compiles on both (dead-code warnings allowed at this step).

**Step 5: Commit**

```bash
git add crates/engine
git commit -m "feat(engine): native and web platform glue"
```

---

### Task 8: App loop and demo binary; run natively

**Files:**
- Create: `crates/engine/src/app.rs`
- Modify: `crates/engine/src/lib.rs`, `crates/demo/src/main.rs`

**Step 1: `app.rs`**

```rust
//! The winit application: creates the window, starts GPU init, and draws frames.

use std::sync::Arc;

use impulse_render::{FrameOutcome, Gpu, GpuInitError, backend_name, wgpu};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy, OwnedDisplayHandle};
use winit::window::{Window, WindowId};

use crate::platform;

const CLEAR_COLOR: wgpu::Color = wgpu::Color { r: 0.02, g: 0.03, b: 0.06, a: 1.0 };

/// Delivered to the event loop when async GPU init finishes (immediately on
/// desktop; later on the web, where init can't block).
enum UserEvent {
    GpuReady(Result<Gpu, GpuInitError>),
}

enum State {
    Uninitialized,
    Loading,
    // Boxed: Gpu is large next to the unit variants (clippy::large_enum_variant).
    Running(Box<Gpu>),
    Failed,
}

struct App {
    title: &'static str,
    proxy: EventLoopProxy<UserEvent>,
    window: Option<Arc<Window>>,
    state: State,
}

/// Opens the window and runs the engine until it is closed.
pub fn run(title: &'static str) {
    platform::init_logging();
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .expect("failed to create the event loop");
    let app = App {
        title,
        proxy: event_loop.create_proxy(),
        window: None,
        state: State::Uninitialized,
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut app = app;
        event_loop.run_app(&mut app).expect("event loop failed");
    }
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }
}

async fn init_gpu(
    display: OwnedDisplayHandle,
    window: Arc<Window>,
) -> Result<Gpu, GpuInitError> {
    let mut desc = wgpu::InstanceDescriptor::new_with_display_handle(Box::new(display));
    if let Some(choice) = platform::backend_override() {
        desc.backends = choice.backends();
    }
    let instance = wgpu::util::new_instance_with_webgpu_detection(desc).await;
    // Created before the adapter: on the web, WebGL needs the canvas to pick an adapter.
    let surface = instance
        .create_surface(window.clone())
        .map_err(GpuInitError::Surface)?;
    let size = window.inner_size();
    Gpu::new(instance, surface, size.width, size.height).await
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !matches!(self.state, State::Uninitialized) {
            return;
        }
        self.state = State::Loading;

        let window = match event_loop.create_window(platform::window_attributes(self.title)) {
            Ok(window) => Arc::new(window),
            Err(e) => {
                platform::report_error(&format!("Could not create a window: {e}"));
                self.state = State::Failed;
                event_loop.exit();
                return;
            }
        };
        self.window = Some(window.clone());

        let display = event_loop.owned_display_handle();
        let proxy = self.proxy.clone();
        platform::spawn(async move {
            let result = init_gpu(display, window).await;
            // Fails only if the event loop already exited; nothing left to notify.
            let _ = proxy.send_event(UserEvent::GpuReady(result));
        });
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::GpuReady(Ok(gpu)) => {
                let name = backend_name(gpu.backend());
                platform::report_status(&format!("Running on {name}"));
                if let Some(window) = &self.window {
                    window.set_title(&format!("{} ({name})", self.title));
                    window.request_redraw();
                }
                self.state = State::Running(Box::new(gpu));
            }
            UserEvent::GpuReady(Err(e)) => {
                platform::report_error(&e.to_string());
                self.state = State::Failed;
                // On the web, keep the page (and its error message) alive.
                if !cfg!(target_arch = "wasm32") {
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let State::Running(gpu) = &mut self.state {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }
}

impl App {
    fn redraw(&mut self) {
        let (State::Running(gpu), Some(window)) = (&mut self.state, &self.window) else {
            return;
        };
        if gpu.render_clear(CLEAR_COLOR) == FrameOutcome::SurfaceLost {
            match gpu.instance().create_surface(window.clone()) {
                Ok(surface) => gpu.replace_surface(surface),
                Err(e) => {
                    platform::report_error(&format!("Lost the rendering surface: {e}"));
                    self.state = State::Failed;
                    return;
                }
            }
        }
        window.request_redraw();
    }
}
```

**Step 2: Wire up the crate and the binary**

Append to `crates/engine/src/lib.rs`:

```rust
mod app;

pub use app::run;
```

Replace `crates/demo/src/main.rs`:

```rust
//! impulse demo. Desktop: `cargo run -p impulse-demo`. Browser: `scripts/build-web.sh`.

fn main() {
    impulse_engine::run("impulse");
}
```

**Step 3: Format, then lint both targets**

Run: `cargo fmt --all`, then `cargo clippy --workspace --all-targets -- -D warnings` then `cargo clippy --workspace --target wasm32-unknown-unknown -- -D warnings`
Expected: clean.

**Step 4: Run natively**

Run: `cargo run -p impulse-demo`
Expected: a window titled `impulse (DirectX 12)` or `impulse (Vulkan)`, filled with a near-black navy color; the log shows `Running on …`. Resizing and minimizing don't crash, and closing the window exits with code 0. Ask the user to confirm the window looks right; Claude can only check the log.

**Step 5: Commit**

```bash
git add crates
git commit -m "feat(engine): winit app loop with async GPU init; demo binary"
```

---

### Task 9: Web shell and build script

**Files:**
- Create: `web/index.html`, `scripts/wasm-bindgen-version.sh`, `scripts/build-web.sh`, `.claude/launch.json`

**Step 1: `web/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>impulse</title>
    <style>
      html, body { margin: 0; height: 100%; background: #05080f; color: #cbd5e1;
                   font: 14px system-ui, sans-serif; overflow: hidden; }
      #impulse-canvas { display: block; width: 100vw; height: 100vh; }
      #status { position: fixed; left: 12px; bottom: 12px; padding: 4px 8px;
                background: rgba(0, 0, 0, 0.55); border-radius: 4px; }
      #status.error { color: #fca5a5; }
    </style>
  </head>
  <body>
    <canvas id="impulse-canvas"></canvas>
    <div id="status">Loading…</div>
    <script type="module">
      import init from "./impulse.js";
      init().catch((err) => {
        const status = document.getElementById("status");
        status.textContent = `Failed to start: ${err}`;
        status.className = "error";
      });
    </script>
  </body>
</html>
```

**Step 2: `scripts/wasm-bindgen-version.sh`** (shared by the build script and CI)

```bash
#!/usr/bin/env bash
# Prints the wasm-bindgen crate version locked in Cargo.lock.
# The wasm-bindgen CLI must match it exactly.
set -euo pipefail
cd "$(dirname "$0")/.."

version="$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/"/, "", $3); print $3; exit }' Cargo.lock)"
if [ -z "$version" ]; then
  echo "wasm-bindgen not found in Cargo.lock" >&2
  exit 1
fi
echo "$version"
```

**Step 3: `scripts/build-web.sh`**

```bash
#!/usr/bin/env bash
# Builds the browser demo into web/dist/ for local serving or Firebase Hosting.
set -euo pipefail
cd "$(dirname "$0")/.."

expected="$(scripts/wasm-bindgen-version.sh)"
if ! actual="$(wasm-bindgen --version 2>/dev/null)"; then
  echo "wasm-bindgen CLI not found. Install it with:" >&2
  echo "  cargo install --locked wasm-bindgen-cli --version $expected" >&2
  exit 1
fi
if [ "$actual" != "wasm-bindgen $expected" ]; then
  echo "wasm-bindgen CLI is '$actual' but Cargo.lock has $expected. Reinstall with:" >&2
  echo "  cargo install --locked wasm-bindgen-cli --version $expected --force" >&2
  exit 1
fi

cargo build --release -p impulse-demo --target wasm32-unknown-unknown

rm -rf web/dist
mkdir -p web/dist
wasm-bindgen --target web --no-typescript --out-dir web/dist --out-name impulse \
  target/wasm32-unknown-unknown/release/impulse-demo.wasm
cp web/index.html web/dist/
echo "Built web/dist/"
```

Run: `git add scripts && git update-index --chmod=+x scripts/wasm-bindgen-version.sh scripts/build-web.sh`

**Step 4: Test the version guard before installing the CLI**

Run: `bash scripts/build-web.sh`
Expected: exit 1 with `wasm-bindgen CLI not found. Install it with: cargo install --locked wasm-bindgen-cli --version 0.2.128` (or whatever version the lock file holds).

**Step 5: Install the matching CLI**

Run: `cargo install --locked wasm-bindgen-cli --version "$(bash scripts/wasm-bindgen-version.sh)"`
Expected: installs (takes a few minutes).

**Step 6: `.claude/launch.json`** (for the preview tool)

```json
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "impulse-web",
      "runtimeExecutable": "python",
      "runtimeArgs": ["-m", "http.server", "8080", "--directory", "web/dist"],
      "port": 8080
    }
  ]
}
```

**Step 7: Commit**

```bash
git add web/index.html scripts .claude/launch.json
git commit -m "build: web shell and wasm build script with CLI version guard"
```

---

### Task 10: Verify in the browser (WebGPU and forced WebGL2)

**Step 1: Build**

Run: `bash scripts/build-web.sh`
Expected: `Built web/dist/`; `web/dist` contains `index.html`, `impulse.js`, `impulse_bg.wasm`.

**Step 2: Serve and check WebGPU**

Start the `impulse-web` preview and open `http://localhost:8080/`.
Expected: the canvas fills with the navy clear color; `#status` reads `Running on WebGPU`; no console errors. (If the status reads `Running on WebGL2`, the browser pane lacks WebGPU and detection fell back correctly, which is also a pass; record which one happened.)
If the page stays on `Loading…` with no errors, wasm-bindgen did not run `main` (Unverified point 2): stop and report.

**Step 3: Check the forced fallback**

Navigate to `http://localhost:8080/?backend=webgl`.
Expected: `#status` reads `Running on WebGL2`, and the same color fills the canvas.

**Step 4: Check the error path**

Rename `#impulse-canvas` in `web/dist/index.html` only (not the source) and reload.
Expected: the console logs `no <canvas id="impulse-canvas">…`, and the page doesn't crash silently. Rebuild afterwards to restore `web/dist`.

**Step 5: Resize**

Resize the viewport (mobile preset, then desktop).
Expected: no console errors; the canvas still fills the viewport.

**Step 6: Screenshot** for proof, then reset the viewport to desktop.

---

### Task 11: README

**Files:**
- Create: `README.md`

```markdown
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
```

Commit:

```bash
git add README.md
git commit -m "docs: README with build and run instructions"
```

---

### Task 12: CI workflow and GitHub repo

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Workflow** (deploy step included, but it only runs once Task 13 has set up the secrets)

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:
permissions:
  contents: read
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - name: Install toolchain pinned in rust-toolchain.toml
        run: rustup toolchain install
      - uses: Swatinem/rust-cache@v2
      - run: cargo fmt --all --check
      - run: cargo clippy --workspace --all-targets -- -D warnings
      - run: cargo clippy --workspace --target wasm32-unknown-unknown -- -D warnings
      - run: cargo test --workspace
      - run: scripts/check-physics-deps.sh
      - name: Install wasm-bindgen CLI matching Cargo.lock
        run: cargo install --locked wasm-bindgen-cli --version "$(scripts/wasm-bindgen-version.sh)"
      - run: scripts/build-web.sh
      - name: Deploy to Firebase Hosting
        if: github.event_name == 'push' && github.ref == 'refs/heads/main' && vars.FIREBASE_PROJECT_ID != ''
        uses: FirebaseExtended/action-hosting-deploy@v0
        with:
          firebaseServiceAccount: ${{ secrets.FIREBASE_SERVICE_ACCOUNT }}
          projectId: ${{ vars.FIREBASE_PROJECT_ID }}
          channelId: live
```

Commit:

```bash
git add .github/workflows/ci.yml
git commit -m "ci: fmt, clippy (native + wasm), tests, physics dep guard, web build"
```

**Step 2: Create the repo and push** (ask the user first: repo name and public/private. zephyr is public under `josh-W42`)

```bash
gh repo create josh-W42/impulse --public --source . --push
```

**Step 3: Verify CI**

Run: `gh run list --limit 1` then `gh run watch <id>`
Expected: all steps green. The deploy step is skipped because `FIREBASE_PROJECT_ID` isn't set yet. If `rustup toolchain install` or the winit build fails, apply the fallbacks listed under Unverified points.

---

### Task 13: Firebase Hosting deploy

**Files:**
- Create: `firebase.json`, `.firebaserc`

**Step 1: User creates the Firebase project** (a console action; Claude does not do this). Suggested project id: `impulse-engine` (choose another if taken). Create a service account key with the Firebase Hosting Admin role (or run `npx firebase-tools init hosting:github`, which does this and sets the secret), then:

- Repo secret `FIREBASE_SERVICE_ACCOUNT` = the JSON key
- Repo variable `FIREBASE_PROJECT_ID` = the project id

**Step 2: `firebase.json`**

```json
{
  "hosting": {
    "public": "web/dist",
    "ignore": ["firebase.json", "**/.*"],
    "headers": [
      {
        "source": "**/*.@(html|js|wasm)",
        "headers": [{ "key": "Cache-Control", "value": "no-cache" }]
      }
    ]
  }
}
```

(`no-cache` because the output filenames aren't content-hashed. A cached `impulse.js` paired with a new `impulse_bg.wasm` would break. Revisit if load time matters.)

`.firebaserc` (substitute the real id):

```json
{ "projects": { "default": "impulse-engine" } }
```

**Step 3: Commit and push**

```bash
git add firebase.json .firebaserc
git commit -m "deploy: Firebase Hosting config"
git push
```

**Step 4: Verify the live deploy**

Run: `gh run watch` (latest run) and expect the deploy step to run and pass.
Open `https://<project-id>.web.app/` in the browser pane.
Expected: the navy canvas and `Running on WebGPU` (or `WebGL2`); `?backend=webgl` shows `Running on WebGL2`. Take a screenshot as proof.

**M0 is done when:** the live URL shows the clear color on WebGPU and on forced WebGL2, CI is green, and the physics-dependency guard runs in CI.
