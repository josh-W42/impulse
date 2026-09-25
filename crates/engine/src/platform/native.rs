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
