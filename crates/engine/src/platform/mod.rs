//! Everything that differs between desktop and browser, behind one set of function names.

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::*;

#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(target_arch = "wasm32")]
pub use web::*;

/// What a file picker was opened for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickKind {
    Scene,
    Model,
}

/// A file the browser picker delivered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Picked {
    pub kind: PickKind,
    pub name: String,
    pub bytes: Vec<u8>,
}

/// Largest file the pickers and folders will read (bytes).
pub const MAX_FILE_BYTES: usize = 32 * 1024 * 1024;
