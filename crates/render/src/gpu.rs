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

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Current surface size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Format of the views returned by [`Gpu::acquire`] (sRGB); pipelines must target it.
    pub fn view_format(&self) -> wgpu::TextureFormat {
        self.view_format
    }

    /// Gets the next surface texture to draw into. `Err` means skip this frame
    /// ([`FrameOutcome::Skipped`]) or recreate the surface ([`FrameOutcome::SurfaceLost`]).
    pub fn acquire(&mut self) -> Result<Frame, FrameOutcome> {
        let texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                drop(texture);
                self.surface.configure(&self.device, &self.config);
                return Err(FrameOutcome::Skipped);
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Err(FrameOutcome::Skipped);
            }
            wgpu::CurrentSurfaceTexture::Occluded | wgpu::CurrentSurfaceTexture::Timeout => {
                return Err(FrameOutcome::Skipped);
            }
            wgpu::CurrentSurfaceTexture::Lost => return Err(FrameOutcome::SurfaceLost),
            wgpu::CurrentSurfaceTexture::Validation => {
                log::error!("validation error while acquiring the surface texture");
                return Err(FrameOutcome::Skipped);
            }
        };
        let view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.view_format),
            ..Default::default()
        });
        Ok(Frame { texture, view })
    }

    /// Shows a frame. Submit all work that draws into it first.
    pub fn present(&self, frame: Frame) {
        self.queue.present(frame.texture);
    }
}

/// A surface texture acquired for one frame, plus an sRGB view to draw into.
pub struct Frame {
    texture: wgpu::SurfaceTexture,
    pub view: wgpu::TextureView,
}
