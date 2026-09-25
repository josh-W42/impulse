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
    query
        .trim_start_matches('?')
        .split('&')
        .find_map(|pair| match pair.split_once('=') {
            Some(("backend", "webgpu")) => Some(BackendOverride::WebGpu),
            Some(("backend", "webgl")) => Some(BackendOverride::WebGl),
            _ => None,
        })
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
        assert_eq!(
            parse_backend_override("?backend=webgl"),
            Some(BackendOverride::WebGl)
        );
        assert_eq!(
            parse_backend_override("?backend=webgpu"),
            Some(BackendOverride::WebGpu)
        );
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
        assert_eq!(
            BackendOverride::WebGpu.backends(),
            wgpu::Backends::BROWSER_WEBGPU
        );
    }
}
