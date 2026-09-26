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

/// Writes `scenes/<stem>.ron`; returns the path for the status line.
pub fn save_scene(stem: &str, text: &str) -> Result<String, String> {
    save_scene_in(Path::new(SCENE_DIR), stem, text)
}

/// Stems of the `.ron` files in `scenes/`, sorted.
pub fn saved_scenes() -> Vec<String> {
    saved_scenes_in(Path::new(SCENE_DIR))
}

pub fn read_scene(stem: &str) -> Result<String, String> {
    read_scene_in(Path::new(SCENE_DIR), stem)
}

/// The browser's file picker; nothing to do on desktop.
pub fn pick_scene_file() {}

pub fn take_picked_scene() -> Option<Result<(String, String), String>> {
    None
}

fn save_scene_in(dir: &Path, stem: &str, text: &str) -> Result<String, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("can't create {}: {e}", dir.display()))?;
    let path = dir.join(format!("{stem}.ron"));
    std::fs::write(&path, text).map_err(|e| format!("can't write {}: {e}", path.display()))?;
    Ok(path.display().to_string())
}

fn saved_scenes_in(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut stems: Vec<String> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "ron"))
        .filter_map(|p| p.file_stem()?.to_str().map(str::to_owned))
        .collect();
    stems.sort();
    stems
}

fn read_scene_in(dir: &Path, stem: &str) -> Result<String, String> {
    let path = dir.join(format!("{stem}.ron"));
    let size = std::fs::metadata(&path)
        .map_err(|e| format!("can't read {}: {e}", path.display()))?
        .len();
    if size > crate::scene_desc::MAX_SCENE_BYTES as u64 {
        return Err(format!("{} is too large", path.display()));
    }
    std::fs::read_to_string(&path).map_err(|e| format!("can't read {}: {e}", path.display()))
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
            saved_scenes_in(&dir).is_empty(),
            "missing folder lists nothing"
        );
        save_scene_in(&dir, "b_scene", "(b)").unwrap();
        save_scene_in(&dir, "a_scene", "(a)").unwrap();
        std::fs::write(dir.join("notes.txt"), "not a scene").unwrap();
        assert_eq!(saved_scenes_in(&dir), ["a_scene", "b_scene"]);
        assert_eq!(read_scene_in(&dir, "b_scene").unwrap(), "(b)");
        assert!(read_scene_in(&dir, "missing").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
