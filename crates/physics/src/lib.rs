//! Rigid-body physics for the impulse engine.
//!
//! Architecture rule: this crate depends on `glam` only, never on rendering,
//! windowing, or engine crates, so it can be tested, benchmarked, and reused
//! headless. `scripts/check-physics-deps.sh` enforces this in CI.
