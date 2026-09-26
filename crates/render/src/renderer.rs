//! Frame rendering: shadow map, lit instanced meshes, debug lines, egui.

use std::mem::size_of;
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use glam::camera::rh::{proj, view};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::draw_list::{DrawList, InstanceRaw, group_by_mesh};
use crate::gpu::Gpu;
use crate::mesh::{MeshData, MeshId, MeshKind, Vertex};
use crate::shaders::{LINES_WGSL, SCENE_WGSL};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Within WebGL2's guaranteed texture size.
const SHADOW_SIZE: u32 = 2048;
/// Linear-space sky color (the sRGB target encodes it).
const SKY: wgpu::Color = wgpu::Color {
    r: 0.18,
    g: 0.26,
    b: 0.38,
    a: 1.0,
};
const LIGHT_COLOR: [f32; 4] = [1.0, 0.96, 0.9, 1.0];
/// Half-width of the square the shadow map covers, centered on the origin.
const SHADOW_EXTENT: f32 = 25.0;

const MESH_ATTRS: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 10 => Float32x2];
const INSTANCE_ATTRS: [wgpu::VertexAttribute; 8] = wgpu::vertex_attr_array![
    2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4,
    6 => Float32x4, 7 => Float32x4, 8 => Float32x4, 9 => Float32x4
];
const LINE_ATTRS: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];

/// What the camera sees this frame.
#[derive(Debug, Clone, Copy)]
pub struct CameraData {
    pub view_proj: Mat4,
    pub eye: Vec3,
}

/// egui's output for this frame, already tessellated.
pub struct UiFrame {
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
    pub pixels_per_point: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GlobalsRaw {
    view_proj: [[f32; 4]; 4],
    light_view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    light_dir: [f32; 4],
    light_color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LineVertex {
    position: [f32; 3],
    color: [f32; 4],
}

struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    /// Index into `Renderer::textures`.
    texture: usize,
}

impl GpuMesh {
    fn new(device: &wgpu::Device, data: &MeshData) -> Self {
        Self {
            vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh vertices"),
                contents: bytemuck::cast_slice(&data.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh indices"),
                contents: bytemuck::cast_slice(&data.indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            index_count: data.indices.len() as u32,
            texture: 0,
        }
    }
}

const INITIAL_INSTANCE_BYTES: u64 = 64 * size_of::<InstanceRaw>() as u64;

/// A vertex buffer that grows to the next power of two when a frame needs more room.
struct GrowableBuffer {
    label: &'static str,
    buffer: wgpu::Buffer,
    capacity: u64,
}

impl GrowableBuffer {
    fn new(device: &wgpu::Device, label: &'static str, capacity: u64) -> Self {
        Self {
            label,
            buffer: Self::create(device, label, capacity),
            capacity,
        }
    }

    fn create(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let len = bytes.len() as u64;
        if len > self.capacity {
            self.capacity = len.next_power_of_two();
            self.buffer = Self::create(device, self.label, self.capacity);
        }
        queue.write_buffer(&self.buffer, 0, bytes);
    }
}

struct DepthTarget {
    view: wgpu::TextureView,
    size: (u32, u32),
}

impl DepthTarget {
    fn new(device: &wgpu::Device, size: (u32, u32)) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        Self {
            view: texture.create_view(&Default::default()),
            size,
        }
    }
}

pub struct Renderer {
    globals: wgpu::Buffer,
    /// Globals + shadow map + comparison sampler (lit pass).
    scene_bind_group: wgpu::BindGroup,
    /// Globals only: the shadow pass can't bind the texture it renders into.
    globals_bind_group: wgpu::BindGroup,
    lit_pipeline: wgpu::RenderPipeline,
    /// Layout and shared sampler for per-material bind groups (group 1).
    material_bgl: wgpu::BindGroupLayout,
    material_sampler: wgpu::Sampler,
    /// Material bind groups; 0 is a 1×1 white texture (untextured meshes).
    textures: Vec<wgpu::BindGroup>,
    /// Meshes/textures below these counts survive `clear_custom_meshes`.
    pinned_meshes: usize,
    pinned_textures: usize,
    shadow_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    shadow_view: wgpu::TextureView,
    /// Built-in meshes first (in `MeshKind` order), then runtime-added ones.
    meshes: Vec<GpuMesh>,
    instances: Vec<GrowableBuffer>,
    lines: GrowableBuffer,
    depth: DepthTarget,
    egui: egui_wgpu::Renderer,
}

impl Renderer {
    pub fn new(gpu: &Gpu) -> Self {
        let device = gpu.device();

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: size_of::<GlobalsRaw>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow map"),
            size: wgpu::Extent3d {
                width: SHADOW_SIZE,
                height: SHADOW_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&Default::default());
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let globals_entry = |visibility| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(size_of::<GlobalsRaw>() as u64),
            },
            count: None,
        };
        let globals_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[globals_entry(wgpu::ShaderStages::VERTEX)],
        });
        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene"),
            entries: &[
                globals_entry(wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &globals_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            }],
        });
        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene"),
            layout: &scene_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
            ],
        });

        let layout = |label, bgl: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            })
        };
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let material_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene"),
            bind_group_layouts: &[Some(&scene_bgl), Some(&material_bgl)],
            immediate_size: 0,
        });
        let globals_layout = layout("globals", &globals_bgl);

        let scene_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene"),
            source: wgpu::ShaderSource::Wgsl(SCENE_WGSL.into()),
        });
        let lines_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lines"),
            source: wgpu::ShaderSource::Wgsl(LINES_WGSL.into()),
        });

        let mesh_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &MESH_ATTRS,
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_ATTRS,
        };
        let solid = wgpu::PrimitiveState {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        };
        let depth = |write, compare, bias| {
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(write),
                depth_compare: Some(compare),
                stencil: wgpu::StencilState::default(),
                bias,
            })
        };

        let lit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lit"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(mesh_layout.clone()), Some(instance_layout.clone())],
            },
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(gpu.view_format().into())],
            }),
            primitive: solid,
            depth_stencil: depth(
                true,
                wgpu::CompareFunction::Less,
                wgpu::DepthBiasState::default(),
            ),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow"),
            layout: Some(&globals_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: Some("vs_shadow"),
                compilation_options: Default::default(),
                buffers: &[Some(mesh_layout), Some(instance_layout)],
            },
            fragment: None,
            primitive: solid,
            depth_stencil: depth(
                true,
                wgpu::CompareFunction::LessEqual,
                wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            ),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lines"),
            layout: Some(&globals_layout),
            vertex: wgpu::VertexState {
                module: &lines_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<LineVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &LINE_ATTRS,
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &lines_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(gpu.view_format().into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: depth(
                false,
                wgpu::CompareFunction::LessEqual,
                wgpu::DepthBiasState::default(),
            ),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let meshes: Vec<GpuMesh> = MeshKind::ALL
            .iter()
            .map(|k| GpuMesh::new(device, &k.data()))
            .collect();
        let instances = (0..meshes.len())
            .map(|_| GrowableBuffer::new(device, "instances", INITIAL_INSTANCE_BYTES))
            .collect();

        let white = texture_bind_group(
            gpu,
            &material_bgl,
            &material_sampler,
            1,
            1,
            &[&[255, 255, 255, 255]],
        );
        Self {
            globals,
            scene_bind_group,
            globals_bind_group,
            lit_pipeline,
            material_bgl,
            material_sampler,
            textures: vec![white],
            pinned_meshes: MeshKind::ALL.len(),
            pinned_textures: 1,
            shadow_pipeline,
            line_pipeline,
            shadow_view,
            meshes,
            instances,
            lines: GrowableBuffer::new(device, "lines", 256 * size_of::<LineVertex>() as u64),
            depth: DepthTarget::new(device, gpu.size()),
            egui: egui_wgpu::Renderer::new(
                device,
                gpu.surface_format(),
                egui_wgpu::RendererOptions::default(),
            ),
        }
    }

    /// Draws one frame and submits it; the caller presents. `target` is the sRGB
    /// view for the scene, `ui_target` the surface-format view for egui.
    pub fn render(
        &mut self,
        gpu: &Gpu,
        target: &wgpu::TextureView,
        ui_target: &wgpu::TextureView,
        camera: &CameraData,
        list: &DrawList,
        mut ui: UiFrame,
    ) {
        let (device, queue) = (gpu.device(), gpu.queue());
        if self.depth.size != gpu.size() {
            self.depth = DepthTarget::new(device, gpu.size());
        }

        let light_dir = Vec3::new(-0.4, -1.0, -0.3).normalize();
        let globals = GlobalsRaw {
            view_proj: camera.view_proj.to_cols_array_2d(),
            light_view_proj: light_view_proj(light_dir).to_cols_array_2d(),
            eye: camera.eye.extend(1.0).to_array(),
            light_dir: light_dir.extend(0.0).to_array(),
            light_color: LIGHT_COLOR,
        };
        queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));

        let groups = group_by_mesh(&list.instances, self.meshes.len());
        for (buffer, group) in self.instances.iter_mut().zip(&groups) {
            buffer.write(device, queue, bytemuck::cast_slice(group));
        }
        let line_vertices: Vec<LineVertex> = list
            .lines
            .iter()
            .flat_map(|l| {
                [l.a, l.b].map(|p| LineVertex {
                    position: p.to_array(),
                    color: l.color,
                })
            })
            .collect();
        self.lines
            .write(device, queue, bytemuck::cast_slice(&line_vertices));

        // Drain (not iterate): egui 0.36 panics in debug if a TexturesDelta is dropped unapplied.
        for (id, deltas) in ui.textures_delta.set.drain() {
            for delta in &deltas {
                self.egui.update_texture(device, queue, id, delta);
            }
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(0, &self.globals_bind_group, &[]);
            self.draw_meshes(&mut pass, &groups, false);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(SKY),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.lit_pipeline);
            pass.set_bind_group(0, &self.scene_bind_group, &[]);
            self.draw_meshes(&mut pass, &groups, true);

            if !line_vertices.is_empty() {
                let bytes = std::mem::size_of_val(line_vertices.as_slice()) as u64;
                pass.set_pipeline(&self.line_pipeline);
                pass.set_bind_group(0, &self.globals_bind_group, &[]);
                pass.set_vertex_buffer(0, self.lines.buffer.slice(..bytes));
                pass.draw(0..line_vertices.len() as u32, 0..1);
            }
        }

        let (width, height) = gpu.size();
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: ui.pixels_per_point,
        };
        let egui_commands =
            self.egui
                .update_buffers(device, queue, &mut encoder, &ui.primitives, &screen);
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: ui_target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.egui.render(&mut pass, &ui.primitives, &screen);
        }
        queue.submit(egui_commands.into_iter().chain([encoder.finish()]));

        for id in ui.textures_delta.free.drain() {
            self.egui.free_texture(&id);
        }
    }

    /// Uploads an RGBA8 sRGB texture with its mip chain (`mips[0]` is
    /// `width`×`height`, each next level half the size, down to 1×1).
    pub fn add_texture(&mut self, gpu: &Gpu, width: u32, height: u32, mips: &[&[u8]]) -> TextureId {
        let group = texture_bind_group(
            gpu,
            &self.material_bgl,
            &self.material_sampler,
            width,
            height,
            mips,
        );
        self.textures.push(group);
        TextureId((self.textures.len() - 1) as u32)
    }

    /// Uploads an untextured mesh for later instancing (e.g. a convex hull's shape).
    pub fn add_mesh(&mut self, gpu: &Gpu, data: &MeshData) -> MeshId {
        self.add_textured_mesh(gpu, data, TextureId::WHITE)
    }

    /// Uploads a mesh drawn with `texture` (from [`Renderer::add_texture`]).
    pub fn add_textured_mesh(&mut self, gpu: &Gpu, data: &MeshData, texture: TextureId) -> MeshId {
        let mut mesh = GpuMesh::new(gpu.device(), data);
        mesh.texture = if (texture.0 as usize) < self.textures.len() {
            texture.0 as usize
        } else {
            0
        };
        self.meshes.push(mesh);
        self.instances.push(GrowableBuffer::new(
            gpu.device(),
            "instances",
            INITIAL_INSTANCE_BYTES,
        ));
        MeshId((self.meshes.len() - 1) as u32)
    }

    /// Frees runtime-added meshes and textures added since the last
    /// [`Renderer::pin_custom_meshes`] (built-ins and pinned ones stay); their
    /// ids become invalid.
    pub fn clear_custom_meshes(&mut self) {
        self.meshes.truncate(self.pinned_meshes);
        self.instances.truncate(self.pinned_meshes);
        self.textures.truncate(self.pinned_textures);
    }

    /// Keeps everything uploaded so far (e.g. model meshes) through later
    /// `clear_custom_meshes` calls.
    pub fn pin_custom_meshes(&mut self) {
        self.pinned_meshes = self.meshes.len();
        self.pinned_textures = self.textures.len();
    }

    /// Frees every runtime-added mesh and texture, pinned or not.
    pub fn reset_custom_meshes(&mut self) {
        self.pinned_meshes = MeshKind::ALL.len();
        self.pinned_textures = 1;
        self.clear_custom_meshes();
    }

    /// `with_material`: bind each mesh's texture (the lit pass; the shadow pass
    /// has no material group).
    fn draw_meshes(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        groups: &[Vec<InstanceRaw>],
        with_material: bool,
    ) {
        for (index, group) in groups.iter().enumerate() {
            let count = group.len() as u32;
            if count == 0 {
                continue;
            }
            let mesh = &self.meshes[index];
            let bytes = count as u64 * size_of::<InstanceRaw>() as u64;
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_vertex_buffer(1, self.instances[index].buffer.slice(..bytes));
            if with_material {
                pass.set_bind_group(1, &self.textures[mesh.texture], &[]);
            }
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..count);
        }
    }
}

/// A texture handle from [`Renderer::add_texture`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(pub u32);

impl TextureId {
    /// The built-in 1×1 white texture.
    pub const WHITE: TextureId = TextureId(0);
}

/// Uploads an sRGB RGBA8 mip chain and binds it with the shared sampler.
fn texture_bind_group(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    width: u32,
    height: u32,
    mips: &[&[u8]],
) -> wgpu::BindGroup {
    let device = gpu.device();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("material"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: mips.len().max(1) as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, data) in mips.iter().enumerate() {
        let (w, h) = ((width >> level).max(1), (height >> level).max(1));
        gpu.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }
    let view = texture.create_view(&Default::default());
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("material"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Orthographic view of the scene from the directional light.
fn light_view_proj(light_dir: Vec3) -> Mat4 {
    let light_view = view::look_at_mat4(-light_dir * 40.0, Vec3::ZERO, Vec3::Y);
    let light_proj = proj::directx::orthographic(
        -SHADOW_EXTENT,
        SHADOW_EXTENT,
        -SHADOW_EXTENT,
        SHADOW_EXTENT,
        1.0,
        80.0,
    );
    light_proj * light_view
}
