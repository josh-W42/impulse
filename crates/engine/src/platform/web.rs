//! Browser platform glue: canvas lookup, async spawning, and on-page status text.

use std::cell::RefCell;
use std::future::Future;

use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};
use winit::platform::web::WindowAttributesExtWebSys;
use winit::window::{Window, WindowAttributes};

use crate::backend::{BackendOverride, parse_backend_override, query_value};

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
    Window::default_attributes()
        .with_title(title)
        .with_canvas(canvas)
}

pub fn backend_override() -> Option<BackendOverride> {
    let search = web_sys::window()?.location().search().ok()?;
    parse_backend_override(&search)
}

/// `?scene=tower` etc. opens a specific demo scene.
pub fn scene_override() -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    query_value(&search, "scene").map(str::to_owned)
}

/// Desktop lists saved scenes itself; the browser uses a file picker.
/// Starting UI size: browsers set body text around 16 px, larger than egui's
/// default, so the web build starts bigger.
pub const DEFAULT_UI_SCALE: f32 = 1.25;

pub const HAS_SCENE_FOLDER: bool = false;

thread_local! {
    /// A file the picker delivered, or why it couldn't be read.
    static PICKED: RefCell<Option<Result<super::Picked, String>>> = const { RefCell::new(None) };
}

/// Downloads `text` as `<stem>.ron`.
pub fn save_scene(stem: &str, text: &str) -> Result<String, String> {
    let js = |e: JsValue| format!("download failed: {e:?}");
    let doc = document().ok_or("no document")?;
    let options = web_sys::BlobPropertyBag::new();
    options.set_type("text/plain");
    let parts = js_sys::Array::of1(&JsValue::from_str(text));
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &options).map_err(js)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob).map_err(js)?;
    let link: web_sys::HtmlAnchorElement = doc
        .create_element("a")
        .map_err(js)?
        .dyn_into()
        .map_err(|_| "not an anchor")?;
    let file = format!("{stem}.ron");
    link.set_href(&url);
    link.set_download(&file);
    link.click();
    // Free the blob once the download has had time to start.
    let revoke = Closure::once_into_js(move || {
        let _ = web_sys::Url::revoke_object_url(&url);
    });
    if let Some(window) = web_sys::window() {
        let _ = window
            .set_timeout_with_callback_and_timeout_and_arguments_0(revoke.unchecked_ref(), 10_000);
    }
    Ok(file)
}

/// The browser has no scene folder to list.
pub fn saved_scenes() -> Vec<String> {
    Vec::new()
}

pub fn read_scene(_stem: &str) -> Result<String, String> {
    Err("use Open… to load a scene file in the browser".into())
}

/// The browser has no models folder to list.
pub fn model_files() -> Vec<String> {
    Vec::new()
}

pub fn read_model(_file_name: &str) -> Result<Vec<u8>, String> {
    Err("use Import… to load a model in the browser".into())
}

pub fn model_folder() -> String {
    String::new()
}

/// Opens the browser's file picker; the chosen file arrives later through
/// [`take_picked`].
pub fn pick_file(kind: super::PickKind) {
    let Some(doc) = document() else {
        return;
    };
    let Ok(input) = doc
        .create_element("input")
        .map(|e| e.unchecked_into::<web_sys::HtmlInputElement>())
    else {
        return;
    };
    input.set_type("file");
    input.set_accept(match kind {
        super::PickKind::Scene => ".ron,text/plain",
        super::PickKind::Model => ".glb,model/gltf-binary",
    });
    let picker = input.clone();
    let on_change = Closure::once_into_js(move || {
        let Some(file) = picker.files().and_then(|files| files.get(0)) else {
            return;
        };
        if file.size() > super::MAX_FILE_BYTES as f64 {
            deliver(Err(format!("{} is too large", file.name())));
            return;
        }
        wasm_bindgen_futures::spawn_local(async move {
            let buffer = wasm_bindgen_futures::JsFuture::from(file.array_buffer()).await;
            deliver(match buffer {
                Ok(buffer) => Ok(super::Picked {
                    kind,
                    name: file.name(),
                    bytes: js_sys::Uint8Array::new(&buffer).to_vec(),
                }),
                Err(_) => Err(format!("couldn't read {}", file.name())),
            });
        });
    });
    input.set_onchange(Some(on_change.unchecked_ref()));
    input.click();
}

fn deliver(result: Result<super::Picked, String>) {
    PICKED.with(|picked| *picked.borrow_mut() = Some(result));
}

/// The file the picker delivered since the last call, if any.
pub fn take_picked() -> Option<Result<super::Picked, String>> {
    PICKED.with(|picked| picked.borrow_mut().take())
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
