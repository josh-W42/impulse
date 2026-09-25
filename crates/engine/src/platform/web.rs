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
