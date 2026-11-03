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

/// The scene's colour target format, and whether it can be multisampled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneTarget {
    pub format: wgpu::TextureFormat,
    /// Linear HDR, tonemapped by the post pass. Otherwise display-ready sRGB:
    /// the scene shaders tonemap themselves and there is no bloom.
    pub hdr: bool,
    /// 4× MSAA with resolve, for both the colour and the depth format.
    pub msaa: bool,
    /// Ambient occlusion: needs `R32Float` render targets for view depth.
    pub ao: bool,
}

pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const SDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const VIEW_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// HDR needs half floats rendered, sampled with filtering (bloom) and blended
/// (bloom's upsample adds); WebGL2 needs `EXT_color_buffer_float` for that.
fn picks_scene_target(
    hdr: wgpu::TextureFormatFeatures,
    sdr: wgpu::TextureFormatFeatures,
    depth: wgpu::TextureFormatFeatures,
    view_depth: wgpu::TextureFormatFeatures,
) -> SceneTarget {
    use wgpu::TextureFormatFeatureFlags as F;
    use wgpu::TextureUsages as U;
    let renders = |f: &wgpu::TextureFormatFeatures| {
        f.allowed_usages
            .contains(U::RENDER_ATTACHMENT | U::TEXTURE_BINDING)
            && f.flags.contains(F::FILTERABLE | F::BLENDABLE)
    };
    let msaa = |f: &wgpu::TextureFormatFeatures| {
        f.flags.contains(F::MULTISAMPLE_X4 | F::MULTISAMPLE_RESOLVE)
    };
    let (format, colour) = if renders(&hdr) {
        (HDR_FORMAT, hdr)
    } else {
        (SDR_FORMAT, sdr)
    };
    SceneTarget {
        format,
        hdr: format == HDR_FORMAT,
        msaa: msaa(&colour) && depth.flags.contains(F::MULTISAMPLE_X4),
        ao: view_depth
            .allowed_usages
            .contains(U::RENDER_ATTACHMENT | U::TEXTURE_BINDING),
    }
}

/// Owns the wgpu device and queue and the surface being presented to.
pub struct Gpu {
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    view_format: wgpu::TextureFormat,
    scene_target: SceneTarget,
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
        // Browsers report validation errors only to an uncaptured-error
        // handler; without one a bad pipeline silently blanks every frame.
        // (Native keeps wgpu's default: panic with the message.)
        #[cfg(target_arch = "wasm32")]
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            log::error!("GPU error: {error}");
        }));

        let (width, height) =
            clamp_surface_size(width, height, device.limits().max_texture_dimension_2d);
        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or(GpuInitError::SurfaceUnsupported)?;
        // The scene renders through an sRGB view so its colors are gamma-correct.
        // Where views may differ from the surface format, the surface itself is
        // non-sRGB (egui prefers that). WebGL2 can't reinterpret formats, so there
        // the surface itself must be sRGB and every view uses that one format.
        let formats = surface.get_capabilities(&adapter).formats;
        let view_formats_supported = adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::SURFACE_VIEW_FORMATS);
        let preferred = if view_formats_supported {
            config.format.remove_srgb_suffix()
        } else {
            config.format.add_srgb_suffix()
        };
        if formats.contains(&preferred) {
            config.format = preferred;
        }
        let view_format = if view_formats_supported {
            config.format.add_srgb_suffix()
        } else {
            config.format
        };
        if view_format != config.format {
            config.view_formats.push(view_format);
        }
        surface.configure(&device, &config);

        let scene_target = picks_scene_target(
            adapter.get_texture_format_features(HDR_FORMAT),
            adapter.get_texture_format_features(SDR_FORMAT),
            adapter.get_texture_format_features(DEPTH_FORMAT),
            adapter.get_texture_format_features(VIEW_DEPTH_FORMAT),
        );
        log::info!("scene target: {scene_target:?}");

        Ok(Self {
            instance,
            surface,
            device,
            queue,
            config,
            view_format,
            scene_target,
            backend: adapter.get_info().backend,
        })
    }

    /// What the scene renders into before the post pass.
    pub fn scene_target(&self) -> SceneTarget {
        self.scene_target
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

    /// The surface's own format (often not sRGB); [`Frame::ui_view`] uses it.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
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
        let ui_view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.config.format),
            ..Default::default()
        });
        Ok(Frame {
            texture,
            view,
            ui_view,
        })
    }

    /// Shows a frame. Submit all work that draws into it first.
    pub fn present(&self, frame: Frame) {
        self.queue.present(frame.texture);
    }
}

/// A surface texture acquired for one frame, plus an sRGB view to draw into.
pub struct Frame {
    texture: wgpu::SurfaceTexture,
    /// sRGB view for the 3D scene (shaders output linear color).
    pub view: wgpu::TextureView,
    /// View in the surface's own format, for egui (which handles gamma itself).
    pub ui_view: wgpu::TextureView,
}

#[cfg(test)]
mod tests {
    use wgpu::TextureFormatFeatureFlags as F;
    use wgpu::TextureUsages as U;

    use super::*;

    fn features(usages: U, flags: F) -> wgpu::TextureFormatFeatures {
        wgpu::TextureFormatFeatures {
            allowed_usages: usages,
            flags,
        }
    }

    fn full() -> wgpu::TextureFormatFeatures {
        features(
            U::RENDER_ATTACHMENT | U::TEXTURE_BINDING,
            F::FILTERABLE | F::BLENDABLE | F::MULTISAMPLE_X4 | F::MULTISAMPLE_RESOLVE,
        )
    }

    fn depth_msaa() -> wgpu::TextureFormatFeatures {
        features(U::RENDER_ATTACHMENT, F::MULTISAMPLE_X4)
    }

    fn renderable() -> wgpu::TextureFormatFeatures {
        features(U::RENDER_ATTACHMENT | U::TEXTURE_BINDING, F::empty())
    }

    #[test]
    fn renderable_half_floats_give_hdr_with_msaa() {
        let target = picks_scene_target(full(), full(), depth_msaa(), renderable());
        assert_eq!(
            target,
            SceneTarget {
                format: HDR_FORMAT,
                hdr: true,
                msaa: true,
                ao: true,
            }
        );
    }

    #[test]
    fn ao_needs_renderable_float_view_depth() {
        // WebGL2 without EXT_color_buffer_float: R32Float only samples.
        let sample_only = features(U::TEXTURE_BINDING, F::empty());
        assert!(!picks_scene_target(full(), full(), depth_msaa(), sample_only).ao);
    }

    #[test]
    fn half_floats_that_only_sample_fall_back_to_srgb() {
        // WebGL2 without EXT_color_buffer_float: sampleable, not renderable.
        let sample_only = features(U::TEXTURE_BINDING, F::FILTERABLE);
        let target = picks_scene_target(sample_only, full(), depth_msaa(), renderable());
        assert_eq!(target.format, SDR_FORMAT);
        assert!(!target.hdr);
        assert!(target.msaa, "the sRGB target can still be multisampled");
    }

    #[test]
    fn half_floats_that_cannot_blend_fall_back_to_srgb() {
        let no_blend = features(
            U::RENDER_ATTACHMENT | U::TEXTURE_BINDING,
            F::FILTERABLE | F::MULTISAMPLE_X4 | F::MULTISAMPLE_RESOLVE,
        );
        assert!(!picks_scene_target(no_blend, full(), depth_msaa(), renderable()).hdr);
    }

    #[test]
    fn msaa_needs_resolve_and_a_multisampled_depth_format() {
        let no_resolve = features(
            U::RENDER_ATTACHMENT | U::TEXTURE_BINDING,
            F::FILTERABLE | F::BLENDABLE | F::MULTISAMPLE_X4,
        );
        assert!(!picks_scene_target(no_resolve, full(), depth_msaa(), renderable()).msaa);
        let single_depth = features(U::RENDER_ATTACHMENT, F::empty());
        let target = picks_scene_target(full(), full(), single_depth, renderable());
        assert!(target.hdr && !target.msaa);
    }
}
