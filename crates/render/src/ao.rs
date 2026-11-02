//! Ambient occlusion: the prepass targets (normals and linear view depth at
//! one sample per pixel), the GTAO and blur passes, and the AO texture the
//! lit shader reads at group 3.

use std::mem::size_of;
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::gpu::{DEPTH_FORMAT, Gpu, VIEW_DEPTH_FORMAT};
use crate::shaders::AO_WGSL;

pub const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// Screen-space ambient occlusion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AoSettings {
    pub enabled: bool,
    /// How far (m) geometry occludes.
    pub radius: f32,
    /// Exponent on the visibility: above 1 darkens.
    pub intensity: f32,
    /// Show the AO buffer alone, in grey (debug).
    pub show: bool,
}

impl Default for AoSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            radius: 1.0,
            intensity: 1.0,
            show: false,
        }
    }
}

/// The camera's view-space axes in world space, and how many pixels a
/// metre spans at view depth 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CameraBasis {
    pub right: Vec3,
    pub up: Vec3,
    /// Opposite the view direction (view space looks down -z).
    pub back: Vec3,
    pub pixels_per_metre: f32,
}

/// Recovers the camera's axes from its view-projection, for a viewport
/// `height` pixels tall.
pub(crate) fn camera_basis(view_proj: Mat4, eye: Vec3, height: u32) -> CameraBasis {
    let inverse = view_proj.inverse();
    let at = |x: f32, y: f32| inverse.project_point3(Vec3::new(x, y, 0.0));
    let center = at(0.0, 0.0);
    let top = at(0.0, 1.0);
    let forward = (inverse.project_point3(Vec3::new(0.0, 0.0, 1.0)) - center).normalize();
    let near = (center - eye).dot(forward);
    let tan_half_fov = top.distance(center) / near;
    CameraBasis {
        right: (at(1.0, 0.0) - center).normalize(),
        up: (top - center).normalize(),
        back: -forward,
        pixels_per_metre: height as f32 / (2.0 * tan_half_fov),
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AoUniforms {
    right: [f32; 4],
    up: [f32; 4],
    back: [f32; 4],
    /// x: radius; y: pixels per metre at depth 1; z: intensity.
    params: [f32; 4],
}

fn texture(
    device: &wgpu::Device,
    label: &str,
    (width, height): (u32, u32),
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// Everything sized to the surface.
struct Targets {
    size: (u32, u32),
    normals: wgpu::TextureView,
    view_depth: wgpu::TextureView,
    /// The prepass's own depth test (never sampled).
    depth: wgpu::TextureView,
    raw: wgpu::TextureView,
    ao: wgpu::TextureView,
    gtao: wgpu::BindGroup,
    blur: wgpu::BindGroup,
    /// The blurred AO, for the lit pass's group 3.
    lit: wgpu::BindGroup,
}

pub(crate) struct Ao {
    uniforms: wgpu::Buffer,
    gtao_bgl: wgpu::BindGroupLayout,
    blur_bgl: wgpu::BindGroupLayout,
    lit_bgl: wgpu::BindGroupLayout,
    gtao: wgpu::RenderPipeline,
    blur: wgpu::RenderPipeline,
    /// AO = 1 everywhere: the lit pass's group 3 while AO is off.
    white: wgpu::BindGroup,
    targets: Targets,
}

impl Ao {
    pub(crate) fn new(gpu: &Gpu) -> Self {
        let device = gpu.device();
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty,
            count: None,
        };
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size_of::<AoUniforms>() as u64),
        };
        let float = wgpu::BindingType::Texture {
            multisampled: false,
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
        };
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
        };
        let gtao_bgl = layout(
            "gtao",
            &[entry(0, uniform), entry(1, float), entry(2, float)],
        );
        let blur_bgl = layout(
            "ao blur",
            &[entry(0, uniform), entry(1, float), entry(3, float)],
        );
        let lit_bgl = create_lit_layout(device);

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ao uniforms"),
            size: size_of::<AoUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ao"),
            source: wgpu::ShaderSource::Wgsl(AO_WGSL.into()),
        });
        let pipeline = |label, bgl: &wgpu::BindGroupLayout, entry_point| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(label),
                        bind_group_layouts: &[Some(bgl)],
                        immediate_size: 0,
                    }),
                ),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    targets: &[Some(AO_FORMAT.into())],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let gtao = pipeline("gtao", &gtao_bgl, "fs_gtao");
        let blur = pipeline("ao blur", &blur_bgl, "fs_blur");

        let white_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("no ao"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: AO_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue().write_texture(
            white_texture.as_image_copy(),
            &[255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: None,
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let white = lit_bind_group(
            device,
            &lit_bgl,
            &white_texture.create_view(&Default::default()),
        );

        let targets = build_targets(
            device,
            gpu.size(),
            [&gtao_bgl, &blur_bgl, &lit_bgl],
            &uniforms,
        );
        Self {
            uniforms,
            gtao_bgl,
            blur_bgl,
            lit_bgl,
            gtao,
            blur,
            white,
            targets,
        }
    }

    /// Resizes the targets if the surface changed.
    pub(crate) fn prepare(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        if self.targets.size != size {
            self.targets = build_targets(
                device,
                size,
                [&self.gtao_bgl, &self.blur_bgl, &self.lit_bgl],
                &self.uniforms,
            );
        }
    }

    /// The layout of the lit pipelines' group 3.
    pub(crate) fn lit_layout(&self) -> &wgpu::BindGroupLayout {
        &self.lit_bgl
    }

    /// The lit pass's group 3: this frame's AO, or none.
    pub(crate) fn lit_bind_group(&self, enabled: bool) -> &wgpu::BindGroup {
        if enabled {
            &self.targets.lit
        } else {
            &self.white
        }
    }

    /// The prepass's attachments: normals and view depth (0 where nothing is
    /// drawn), and its own single-sample depth test.
    pub(crate) fn prepass_attachments(
        &self,
    ) -> (
        [Option<wgpu::RenderPassColorAttachment<'_>>; 2],
        wgpu::RenderPassDepthStencilAttachment<'_>,
    ) {
        let color = |view, clear| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        let up = wgpu::Color {
            r: 0.5,
            g: 1.0,
            b: 0.5,
            a: 1.0,
        };
        (
            [
                color(&self.targets.normals, up),
                color(&self.targets.view_depth, wgpu::Color::TRANSPARENT),
            ],
            wgpu::RenderPassDepthStencilAttachment {
                view: &self.targets.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            },
        )
    }

    /// Runs GTAO and the blur on what the prepass drew.
    pub(crate) fn run(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view_proj: Mat4,
        eye: Vec3,
        settings: &AoSettings,
    ) {
        let basis = camera_basis(view_proj, eye, self.targets.size.1);
        let uniforms = AoUniforms {
            right: basis.right.extend(0.0).to_array(),
            up: basis.up.extend(0.0).to_array(),
            back: basis.back.extend(0.0).to_array(),
            params: [
                settings.radius,
                basis.pixels_per_metre,
                settings.intensity,
                0.0,
            ],
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&uniforms));
        let t = &self.targets;
        for (label, view, pipeline, bind_group) in [
            ("gtao", &t.raw, &self.gtao, &t.gtao),
            ("ao blur", &t.ao, &self.blur, &t.blur),
        ] {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

fn create_lit_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lit ao"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        }],
    })
}

fn lit_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    ao: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lit ao"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::TextureView(ao),
        }],
    })
}

fn build_targets(
    device: &wgpu::Device,
    size: (u32, u32),
    [gtao_bgl, blur_bgl, lit_bgl]: [&wgpu::BindGroupLayout; 3],
    uniforms: &wgpu::Buffer,
) -> Targets {
    use wgpu::TextureUsages as U;
    let sampled = U::RENDER_ATTACHMENT | U::TEXTURE_BINDING;
    let normals = texture(device, "prepass normals", size, NORMAL_FORMAT, sampled);
    let view_depth = texture(
        device,
        "prepass view depth",
        size,
        VIEW_DEPTH_FORMAT,
        sampled,
    );
    let depth = texture(
        device,
        "prepass depth",
        size,
        DEPTH_FORMAT,
        U::RENDER_ATTACHMENT,
    );
    let raw = texture(device, "ao raw", size, AO_FORMAT, sampled);
    let ao = texture(device, "ao", size, AO_FORMAT, sampled);
    let bind_group = |label, layout, third: (u32, &wgpu::TextureView)| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view_depth),
                },
                wgpu::BindGroupEntry {
                    binding: third.0,
                    resource: wgpu::BindingResource::TextureView(third.1),
                },
            ],
        })
    };
    let gtao = bind_group("gtao", gtao_bgl, (2, &normals));
    let blur = bind_group("ao blur", blur_bgl, (3, &raw));
    let lit = lit_bind_group(device, lit_bgl, &ao);
    Targets {
        size,
        normals,
        view_depth,
        depth,
        raw,
        ao,
        gtao,
        blur,
        lit,
    }
}

#[cfg(test)]
mod tests {
    use glam::camera::rh::{proj, view};

    use super::*;

    #[test]
    fn recovers_the_camera_axes_and_pixel_scale() {
        let eye = Vec3::new(3.0, 4.0, 10.0);
        let fov = 1.0_f32;
        let view = view::look_at_mat4(eye, Vec3::new(3.0, 4.0, 0.0), Vec3::Y);
        let view_proj = proj::directx::perspective(fov, 16.0 / 9.0, 0.1, 500.0) * view;
        let b = camera_basis(view_proj, eye, 1080);
        assert!(b.right.abs_diff_eq(Vec3::X, 1e-4), "{:?}", b.right);
        assert!(b.up.abs_diff_eq(Vec3::Y, 1e-4), "{:?}", b.up);
        assert!(b.back.abs_diff_eq(Vec3::Z, 1e-4), "{:?}", b.back);
        let expected = 1080.0 / (2.0 * (fov / 2.0).tan());
        assert!(
            (b.pixels_per_metre - expected).abs() < 0.5,
            "{} vs {expected}",
            b.pixels_per_metre
        );
    }

    #[test]
    fn the_basis_matches_the_view_matrix_for_a_tilted_camera() {
        let eye = Vec3::new(-6.0, 9.0, 4.0);
        let view = view::look_at_mat4(eye, Vec3::new(1.0, 0.0, -2.0), Vec3::Y);
        let view_proj = proj::directx::perspective(0.8, 1.5, 0.1, 500.0) * view;
        let b = camera_basis(view_proj, eye, 720);
        // The view matrix's rows are the camera axes.
        let rows = view.transpose();
        assert!(b.right.abs_diff_eq(rows.x_axis.truncate(), 1e-4));
        assert!(b.up.abs_diff_eq(rows.y_axis.truncate(), 1e-4));
        assert!(b.back.abs_diff_eq(rows.z_axis.truncate(), 1e-4));
    }
}
