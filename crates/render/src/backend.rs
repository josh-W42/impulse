/// Human-readable name of the graphics backend wgpu selected.
pub fn backend_name(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::BrowserWebGpu => "WebGPU",
        wgpu::Backend::Gl if cfg!(target_arch = "wasm32") => "WebGL2",
        wgpu::Backend::Gl => "OpenGL",
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Dx12 => "DirectX 12",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Noop => "no-op",
    }
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
