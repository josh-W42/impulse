//! Frame rendering: shadow map, lit instanced meshes, debug lines, egui.

use std::mem::size_of;
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use glam::camera::rh::{proj, view};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::draw_list::{DrawList, InstanceRaw, MESH_KINDS, group_by_mesh};
use crate::gpu::Gpu;
use crate::mesh::{MeshKind, Vertex};
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

const MESH_ATTRS: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
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
}

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
    shadow_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    shadow_view: wgpu::TextureView,
    meshes: [GpuMesh; MESH_KINDS],
    instances: [GrowableBuffer; MESH_KINDS],
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
        let scene_layout = layout("scene", &scene_bgl);
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

        let meshes = MeshKind::ALL.map(|kind| {
            let data = kind.data();
            GpuMesh {
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
            }
        });
        let instance_bytes = 64 * size_of::<InstanceRaw>() as u64;
        let instances = [
            "cube instances",
            "sphere instances",
            "plane instances",
            "cylinder instances",
        ]
        .map(|label| GrowableBuffer::new(device, label, instance_bytes));

        Self {
            globals,
            scene_bind_group,
            globals_bind_group,
            lit_pipeline,
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

        let groups = group_by_mesh(&list.instances);
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
            self.draw_meshes(&mut pass, &groups);
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
            self.draw_meshes(&mut pass, &groups);

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

    fn draw_meshes(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        groups: &[Vec<InstanceRaw>; MESH_KINDS],
    ) {
        for kind in MeshKind::ALL {
            let count = groups[kind.index()].len() as u32;
            if count == 0 {
                continue;
            }
            let mesh = &self.meshes[kind.index()];
            let bytes = count as u64 * size_of::<InstanceRaw>() as u64;
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_vertex_buffer(1, self.instances[kind.index()].buffer.slice(..bytes));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..mesh.index_count, 0, 0..count);
        }
    }
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
