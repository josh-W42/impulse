//! Desktop platform glue.

use std::future::Future;
use std::path::Path;

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

/// Scenes are picked from the page URL, so desktop builds never have one.
pub fn scene_override() -> Option<String> {
    None
}

/// Desktop lists saved scenes itself; the browser uses a file picker.
pub const HAS_SCENE_FOLDER: bool = true;
/// Saved scenes live here, relative to the working directory.
const SCENE_DIR: &str = "scenes";
/// Models to import live here (`.glb`).
const MODEL_DIR: &str = "models";

/// Writes `scenes/<stem>.ron`; returns the path for the status line.
pub fn save_scene(stem: &str, text: &str) -> Result<String, String> {
    save_scene_in(Path::new(SCENE_DIR), stem, text)
}

/// Names of the `.ron` files in `scenes/`, sorted.
pub fn saved_scenes() -> Vec<String> {
    list_in(Path::new(SCENE_DIR), "ron")
}

/// Reads a file name from [`saved_scenes`].
pub fn read_scene(file_name: &str) -> Result<String, String> {
    let bytes = read_in(Path::new(SCENE_DIR), file_name)?;
    String::from_utf8(bytes).map_err(|_| format!("{file_name} is not text"))
}

/// Names of the `.glb` files in `models/`, sorted.
pub fn model_files() -> Vec<String> {
    list_in(Path::new(MODEL_DIR), "glb")
}

/// Reads a file name from [`model_files`].
pub fn read_model(file_name: &str) -> Result<Vec<u8>, String> {
    read_in(Path::new(MODEL_DIR), file_name)
}

/// Where `models/` resolves to (it is relative to the working directory).
pub fn model_folder() -> String {
    std::path::absolute(MODEL_DIR)
        .map_or_else(|_| MODEL_DIR.to_string(), |p| p.display().to_string())
}

/// The browser's file picker; nothing to do on desktop.
pub fn pick_file(_kind: super::PickKind) {}

pub fn take_picked() -> Option<Result<super::Picked, String>> {
    None
}

fn save_scene_in(dir: &Path, stem: &str, text: &str) -> Result<String, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("can't create {}: {e}", dir.display()))?;
    let path = dir.join(format!("{stem}.ron"));
    std::fs::write(&path, text).map_err(|e| format!("can't write {}: {e}", path.display()))?;
    Ok(path.display().to_string())
}

/// File names in `dir` with `extension`, ignoring case (`.GLB` counts).
fn list_in(dir: &Path, extension: &str) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            p.extension()
                .and_then(|x| x.to_str())
                .is_some_and(|x| x.eq_ignore_ascii_case(extension))
        })
        .filter_map(|p| p.file_name()?.to_str().map(str::to_owned))
        .collect();
    names.sort();
    names
}

/// Reads `file_name` (a bare name, as listed) from `dir`.
fn read_in(dir: &Path, file_name: &str) -> Result<Vec<u8>, String> {
    if Path::new(file_name).file_name().and_then(|n| n.to_str()) != Some(file_name) {
        return Err(format!("not a plain file name: {file_name}"));
    }
    let path = dir.join(file_name);
    let size = std::fs::metadata(&path)
        .map_err(|e| format!("can't read {}: {e}", path.display()))?
        .len();
    if size > super::MAX_FILE_BYTES as u64 {
        return Err(format!("{} is too large", path.display()));
    }
    std::fs::read(&path).map_err(|e| format!("can't read {}: {e}", path.display()))
}

pub fn report_status(message: &str) {
    log::info!("{message}");
}

pub fn report_error(message: &str) {
    log::error!("{message}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenes_save_list_and_read_back() {
        let dir = std::env::temp_dir().join(format!("impulse-scenes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            list_in(&dir, "ron").is_empty(),
            "missing folder lists nothing"
        );
        save_scene_in(&dir, "b_scene", "(b)").unwrap();
        save_scene_in(&dir, "a_scene", "(a)").unwrap();
        std::fs::write(dir.join("notes.txt"), "not a scene").unwrap();
        std::fs::write(dir.join("Shouty.RON"), "(s)").unwrap();
        assert_eq!(
            list_in(&dir, "ron"),
            ["Shouty.RON", "a_scene.ron", "b_scene.ron"],
            "extensions match ignoring case"
        );
        assert_eq!(list_in(&dir, "txt"), ["notes.txt"]);
        assert_eq!(read_in(&dir, "b_scene.ron").unwrap(), b"(b)");
        assert_eq!(read_in(&dir, "Shouty.RON").unwrap(), b"(s)");
        assert!(read_in(&dir, "missing.ron").is_err());
        assert!(
            read_in(&dir, "../b_scene.ron").is_err(),
            "names only, no paths"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
