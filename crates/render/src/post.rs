//! The scene's colour and depth targets, and the post chain that turns them
//! into the displayed image: MSAA resolve, bloom, tonemap.

use crate::gpu::{DEPTH_FORMAT, Gpu, SceneTarget};
use crate::shaders::POST_WGSL;

/// Samples per pixel when MSAA is on.
pub const MSAA_SAMPLES: u32 = 4;
/// Bloom levels stop at this many halvings, or before a side drops below
/// `BLOOM_MIN_SIZE`; past that the blur adds little.
const BLOOM_MAX_LEVELS: usize = 6;
const BLOOM_MIN_SIZE: u32 = 8;

/// How the frame is post-processed. MSAA and bloom need adapter support
/// ([`SceneTarget`]); the renderer ignores them where it is missing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostSettings {
    pub msaa: bool,
    pub bloom: bool,
    /// How much of the blurred image is mixed in (0..=1).
    pub bloom_strength: f32,
    /// Scales scene radiance before tonemapping.
    pub exposure: f32,
}

impl Default for PostSettings {
    fn default() -> Self {
        Self {
            msaa: true,
            bloom: true,
            bloom_strength: 0.04,
            exposure: 1.0,
        }
    }
}

/// The sizes of the bloom levels for a `width` x `height` scene: each half
/// the last, rounded down.
pub(crate) fn bloom_sizes(width: u32, height: u32) -> Vec<(u32, u32)> {
    let mut sizes = Vec::new();
    let (mut w, mut h) = (width, height);
    while sizes.len() < BLOOM_MAX_LEVELS {
        w /= 2;
        h /= 2;
        if w.min(h) < BLOOM_MIN_SIZE {
            break;
        }
        sizes.push((w, h));
    }
    sizes
}

fn texture(
    device: &wgpu::Device,
    label: &str,
    (width, height): (u32, u32),
    format: wgpu::TextureFormat,
    samples: u32,
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
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// Everything sized to the surface; rebuilt when the size or sample count changes.
struct Targets {
    size: (u32, u32),
    samples: u32,
    /// Multisampled colour, resolved into `scene` (only when `samples > 1`).
    msaa: Option<wgpu::TextureView>,
    scene: wgpu::TextureView,
    depth: wgpu::TextureView,
    /// One texture per level, largest first; separate textures so no pass
    /// reads what it writes (WebGL2 rejects feedback loops).
    bloom: Vec<wgpu::TextureView>,
    /// `down[i]` reads the source of level i (the scene, then level i - 1).
    down: Vec<wgpu::BindGroup>,
    /// `up[i]` reads level i + 1.
    up: Vec<wgpu::BindGroup>,
    composite: wgpu::BindGroup,
}

/// What doesn't depend on the surface size.
struct Shared {
    target: SceneTarget,
    sample_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params: wgpu::Buffer,
    /// Bound as the bloom when there are no levels (a tiny surface).
    black: wgpu::TextureView,
    down_first: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
}

pub(crate) struct Post {
    shared: Shared,
    targets: Targets,
}

impl Post {
    pub(crate) fn new(gpu: &Gpu, samples: u32) -> Self {
        let device = gpu.device();
        let target = gpu.scene_target();
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let sampler_entry = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let sample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post sample"),
            entries: &[texture_entry(0), sampler_entry],
        });
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite"),
            entries: &[
                texture_entry(0),
                sampler_entry,
                texture_entry(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let black = texture(
            device,
            "no bloom",
            (1, 1),
            target.format,
            1,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post"),
            source: wgpu::ShaderSource::Wgsl(POST_WGSL.into()),
        });
        let pipeline = |label, bgl: &wgpu::BindGroupLayout, entry, format, blend| {
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
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let add = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let down_first = pipeline(
            "bloom downsample first",
            &sample_bgl,
            "fs_downsample_first",
            target.format,
            None,
        );
        let down = pipeline(
            "bloom downsample",
            &sample_bgl,
            "fs_downsample",
            target.format,
            None,
        );
        let up = pipeline(
            "bloom upsample",
            &sample_bgl,
            "fs_upsample",
            target.format,
            Some(add),
        );
        let composite = pipeline(
            "composite",
            &composite_bgl,
            "fs_composite",
            gpu.view_format(),
            None,
        );

        let shared = Shared {
            target,
            sample_bgl,
            composite_bgl,
            sampler,
            params,
            black,
            down_first,
            down,
            up,
            composite,
        };
        let targets = shared.build_targets(device, gpu.size(), samples);
        Self { shared, targets }
    }

    /// Resizes or re-samples the targets if the frame needs it.
    pub(crate) fn prepare(&mut self, device: &wgpu::Device, size: (u32, u32), samples: u32) {
        if self.targets.size != size || self.targets.samples != samples {
            self.targets = self.shared.build_targets(device, size, samples);
        }
    }

    pub(crate) fn target(&self) -> SceneTarget {
        self.shared.target
    }

    pub(crate) fn depth_view(&self) -> &wgpu::TextureView {
        &self.targets.depth
    }

    /// Where the scene pass draws: the multisampled target resolving into the
    /// scene texture, or the scene texture itself.
    pub(crate) fn color_attachment(&self) -> wgpu::RenderPassColorAttachment<'_> {
        let t = &self.targets;
        let (view, resolve_target, store) = match &t.msaa {
            Some(msaa) => (msaa, Some(&t.scene), wgpu::StoreOp::Discard),
            None => (&t.scene, None, wgpu::StoreOp::Store),
        };
        wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store,
            },
        }
    }

    /// Runs bloom (when on and supported) and composites into `output`.
    pub(crate) fn run(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        settings: &PostSettings,
    ) {
        let (s, t) = (&self.shared, &self.targets);
        let bloom = settings.bloom && s.target.hdr && !t.bloom.is_empty();
        let strength = if bloom {
            settings.bloom_strength.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let tonemap = if s.target.hdr { 1.0 } else { 0.0 };
        queue.write_buffer(
            &s.params,
            0,
            bytemuck::cast_slice(&[strength, tonemap, 0.0, 0.0]),
        );

        let pass = |encoder: &mut wgpu::CommandEncoder,
                    label,
                    view: &wgpu::TextureView,
                    load,
                    pipeline: &wgpu::RenderPipeline,
                    bind_group: &wgpu::BindGroup| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
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
        };
        if bloom {
            let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
            for (i, (view, bind_group)) in t.bloom.iter().zip(&t.down).enumerate() {
                let pipeline = if i == 0 { &s.down_first } else { &s.down };
                pass(encoder, "bloom down", view, clear, pipeline, bind_group);
            }
            for i in (0..t.up.len()).rev() {
                pass(
                    encoder,
                    "bloom up",
                    &t.bloom[i],
                    wgpu::LoadOp::Load,
                    &s.up,
                    &t.up[i],
                );
            }
        }
        pass(
            encoder,
            "composite",
            output,
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            &s.composite,
            &t.composite,
        );
    }
}

impl Shared {
    fn build_targets(&self, device: &wgpu::Device, size: (u32, u32), samples: u32) -> Targets {
        use wgpu::TextureUsages as U;
        let format = self.target.format;
        let msaa = (samples > 1).then(|| {
            texture(
                device,
                "scene msaa",
                size,
                format,
                samples,
                U::RENDER_ATTACHMENT,
            )
        });
        let scene = texture(
            device,
            "scene",
            size,
            format,
            1,
            U::RENDER_ATTACHMENT | U::TEXTURE_BINDING,
        );
        let depth = texture(
            device,
            "depth",
            size,
            DEPTH_FORMAT,
            samples,
            U::RENDER_ATTACHMENT,
        );
        let bloom: Vec<_> = if self.target.hdr {
            bloom_sizes(size.0, size.1)
                .into_iter()
                .map(|level| {
                    texture(
                        device,
                        "bloom",
                        level,
                        format,
                        1,
                        U::RENDER_ATTACHMENT | U::TEXTURE_BINDING,
                    )
                })
                .collect()
        } else {
            Vec::new()
        };
        let sample = |view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post sample"),
                layout: &self.sample_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        };
        let down = std::iter::once(&scene)
            .chain(&bloom)
            .take(bloom.len())
            .map(sample)
            .collect();
        let up = bloom.iter().skip(1).map(sample).collect();
        let composite = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        bloom.first().unwrap_or(&self.black),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params.as_entire_binding(),
                },
            ],
        });
        Targets {
            size,
            samples,
            msaa,
            scene,
            depth,
            bloom,
            down,
            up,
            composite,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bloom_levels_halve_until_too_small_or_six_deep() {
        assert_eq!(
            bloom_sizes(1920, 1080),
            vec![
                (960, 540),
                (480, 270),
                (240, 135),
                (120, 67),
                (60, 33),
                (30, 16)
            ]
        );
        assert_eq!(bloom_sizes(200, 100), vec![(100, 50), (50, 25), (25, 12)]);
    }

    #[test]
    fn tiny_surfaces_have_no_bloom_levels() {
        // A hidden browser canvas reports 1x1.
        assert!(bloom_sizes(1, 1).is_empty());
        assert!(bloom_sizes(15, 4000).is_empty());
    }
}
