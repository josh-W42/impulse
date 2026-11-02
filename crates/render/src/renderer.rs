//! Frame rendering: shadow map, lit instanced meshes and debug lines into the
//! scene target, the post chain, egui.

use std::mem::size_of;
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::ao::{Ao, AoSettings, NORMAL_FORMAT};
use crate::draw_list::{
    DrawList, InstanceRaw, JOINT_TEXTURE_WIDTH, SkinnedInstanceRaw, group_by_mesh, group_skinned,
    joint_texels, joint_texture_rows,
};
use crate::gpu::{DEPTH_FORMAT, Gpu, VIEW_DEPTH_FORMAT};
use crate::mesh::{
    MeshData, MeshId, MeshKind, SkinnedMeshData, SkinnedMeshId, SkinnedVertex, Vertex,
};
use crate::post::{MSAA_SAMPLES, Post, PostSettings};
use crate::shaders::{LINES_WGSL, SCENE_WGSL};
use crate::shadow::{CASCADES, SHADOW_SIZE, ShadowRange, cascades};
use crate::sky::Sky;

/// How far shadows reach until a game sets a range
/// ([`Renderer::set_shadow_distance`], [`Renderer::set_shadow_focus`]).
const DEFAULT_SHADOW_DISTANCE: f32 = 60.0;
/// Focus radii outside this range waste or starve the shadow map.
const SHADOW_RADIUS_RANGE: (f32, f32) = (5.0, 100.0);
/// Shadow distances outside this range (the camera's far plane is 500 m).
const SHADOW_DISTANCE_RANGE: (f32, f32) = (5.0, 500.0);

const MESH_ATTRS: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 10 => Float32x2, 11 => Float32x4];
const INSTANCE_ATTRS: [wgpu::VertexAttribute; 9] = wgpu::vertex_attr_array![
    2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4,
    6 => Float32x4, 7 => Float32x4, 8 => Float32x4, 9 => Float32x4, 12 => Float32x4
];
/// Skinned meshes add joints and weights; with the skinned instance's joint
/// offset that makes 16 vertex attributes, WebGL2's limit.
const SKINNED_MESH_ATTRS: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
    0 => Float32x3, 1 => Float32x3, 10 => Float32x2, 11 => Float32x4,
    13 => Uint16x4, 14 => Float32x4
];
const SKINNED_INSTANCE_ATTRS: [wgpu::VertexAttribute; 10] = wgpu::vertex_attr_array![
    2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4,
    6 => Float32x4, 7 => Float32x4, 8 => Float32x4, 9 => Float32x4, 12 => Float32x4,
    15 => Uint32
];
/// WebGL2's guaranteed texture height bounds the joint texture: 2048 rows
/// of 256 matrices.
const MAX_JOINT_ROWS: u32 = 2048;
const LINE_ATTRS: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];

/// What the camera sees this frame.
#[derive(Debug, Clone, Copy)]
pub struct CameraData {
    pub view_proj: Mat4,
    pub eye: Vec3,
}

/// Exponential height fog, fading toward the horizon sky.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogSettings {
    pub enabled: bool,
    /// Extinction per metre at y = 0.
    pub density: f32,
    /// How fast the fog thins with height (per metre); 0 is uniform fog.
    pub height_falloff: f32,
}

impl Default for FogSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            density: 0.005,
            height_falloff: 0.1,
        }
    }
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
    inv_view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    /// xyz: direction sunlight travels; w: exposure.
    light_dir: [f32; 4],
    light_color: [f32; 4],
    sky_zenith: [f32; 4],
    sky_horizon: [f32; 4],
    sky_ground: [f32; 4],
    sh: [[f32; 4]; 9],
    /// x: 1 when the scene shaders tonemap (no HDR target).
    post: [f32; 4],
    cascades: [[[f32; 4]; 4]; CASCADES],
    /// xyz: camera forward; w: the view depth where cascade 0 starts.
    shadow_camera: [f32; 4],
    cascade_far: [f32; CASCADES],
    cascade_texel: [f32; CASCADES],
    /// x: cascades in use; y: 1 to tint each cascade.
    shadow_params: [f32; 4],
    /// x: fog density at y = 0 (0 = off); y: height falloff.
    fog: [f32; 4],
    /// x: 1 to show the AO buffer alone.
    ao: [f32; 4],
}

/// A material's factors as the shader reads them (see `Material` in scene.wgsl).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MaterialRaw {
    base_color: [f32; 4],
    emissive: [f32; 4],
    /// metallic, roughness, normal scale, occlusion strength
    params: [f32; 4],
}

/// A glTF metallic-roughness material. Textures are optional (defaults:
/// white, flat normal); factors default to glTF's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialDesc {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    /// Linear RGB.
    pub emissive: [f32; 3],
    /// sRGB.
    pub base_color_texture: Option<TextureId>,
    /// Linear: roughness in G, metallic in B.
    pub metallic_roughness_texture: Option<TextureId>,
    /// Linear, tangent space.
    pub normal_texture: Option<TextureId>,
    /// Linear, occlusion in R.
    pub occlusion_texture: Option<TextureId>,
    /// sRGB.
    pub emissive_texture: Option<TextureId>,
}

impl Default for MaterialDesc {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            metallic: 1.0,
            roughness: 1.0,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            emissive: [0.0; 3],
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
        }
    }
}

/// A material handle from [`Renderer::add_material`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u32);

impl MaterialId {
    /// White, factors 1: built-in meshes, which take roughness and metallic
    /// from their instances.
    pub const DEFAULT: MaterialId = MaterialId(0);
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
    /// Index into `Renderer::materials`.
    material: usize,
}

impl GpuMesh {
    fn new(device: &wgpu::Device, data: &MeshData) -> Self {
        Self::from_parts(device, bytemuck::cast_slice(&data.vertices), &data.indices)
    }

    fn from_parts(device: &wgpu::Device, vertices: &[u8], indices: &[u32]) -> Self {
        Self {
            vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh vertices"),
                contents: vertices,
                usage: wgpu::BufferUsages::VERTEX,
            }),
            indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh indices"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            index_count: indices.len() as u32,
            material: 0,
        }
    }
}

/// The joint matrices of every skinned instance this frame, as a float
/// texture the skinned vertex shaders read.
struct JointTexture {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    rows: u32,
}

impl JointTexture {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, rows: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("joint matrices"),
            size: wgpu::Extent3d {
                width: JOINT_TEXTURE_WIDTH,
                height: rows,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("joint matrices"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        Self {
            texture,
            bind_group,
            rows,
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

const SHADOW_BIAS: wgpu::DepthBiasState = wgpu::DepthBiasState {
    constant: 2,
    slope_scale: 2.0,
    clamp: 0.0,
};

fn mesh_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &MESH_ATTRS,
    }
}

fn instance_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<InstanceRaw>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &INSTANCE_ATTRS,
    }
}

fn skinned_mesh_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<SkinnedVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &SKINNED_MESH_ATTRS,
    }
}

fn skinned_instance_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<SkinnedInstanceRaw>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &SKINNED_INSTANCE_ATTRS,
    }
}

fn solid() -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: Some(wgpu::Face::Back),
        ..Default::default()
    }
}

fn depth_state(
    write: bool,
    compare: wgpu::CompareFunction,
    bias: wgpu::DepthBiasState,
) -> Option<wgpu::DepthStencilState> {
    Some(wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(write),
        depth_compare: Some(compare),
        stencil: wgpu::StencilState::default(),
        bias,
    })
}

/// The scene target's sample count for these settings on this adapter.
fn sample_count(gpu: &Gpu, settings: &PostSettings) -> u32 {
    if settings.msaa && gpu.scene_target().msaa {
        MSAA_SAMPLES
    } else {
        1
    }
}

/// What the colour pipelines are built from, kept to rebuild them when the
/// sample count changes.
struct ColorSource {
    scene_shader: wgpu::ShaderModule,
    lines_shader: wgpu::ShaderModule,
    scene_layout: wgpu::PipelineLayout,
    skinned_lit_layout: wgpu::PipelineLayout,
    globals_layout: wgpu::PipelineLayout,
}

/// The pipelines that draw into the scene target.
struct ColorPipelines {
    samples: u32,
    lit: wgpu::RenderPipeline,
    skinned_lit: wgpu::RenderPipeline,
    sky: wgpu::RenderPipeline,
    lines: wgpu::RenderPipeline,
}

impl ColorSource {
    fn pipelines(
        &self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        samples: u32,
    ) -> ColorPipelines {
        let multisample = wgpu::MultisampleState {
            count: samples,
            ..Default::default()
        };
        let pipeline = |label,
                        layout,
                        module,
                        (vs, fs),
                        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                        primitive,
                        depth_stencil| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                primitive,
                depth_stencil,
                multisample,
                multiview_mask: None,
                cache: None,
            })
        };
        let opaque = || {
            depth_state(
                true,
                wgpu::CompareFunction::Less,
                wgpu::DepthBiasState::default(),
            )
        };
        let overlay = || {
            depth_state(
                false,
                wgpu::CompareFunction::LessEqual,
                wgpu::DepthBiasState::default(),
            )
        };
        ColorPipelines {
            samples,
            lit: pipeline(
                "lit",
                &self.scene_layout,
                &self.scene_shader,
                ("vs_main", "fs_main"),
                &[Some(mesh_layout()), Some(instance_layout())],
                solid(),
                opaque(),
            ),
            skinned_lit: pipeline(
                "lit skinned",
                &self.skinned_lit_layout,
                &self.scene_shader,
                ("vs_skinned", "fs_main"),
                &[Some(skinned_mesh_layout()), Some(skinned_instance_layout())],
                solid(),
                opaque(),
            ),
            // At the far plane, after the scene: only uncovered pixels pass.
            sky: pipeline(
                "sky",
                &self.globals_layout,
                &self.scene_shader,
                ("vs_sky", "fs_sky"),
                &[],
                wgpu::PrimitiveState::default(),
                overlay(),
            ),
            lines: pipeline(
                "lines",
                &self.globals_layout,
                &self.lines_shader,
                ("vs_main", "fs_main"),
                &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<LineVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &LINE_ATTRS,
                })],
                wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                overlay(),
            ),
        }
    }
}

pub struct Renderer {
    globals: wgpu::Buffer,
    /// Globals + shadow map + comparison sampler (lit pass).
    scene_bind_group: wgpu::BindGroup,
    /// Globals only: the shadow pass can't bind the texture it renders into.
    globals_bind_group: wgpu::BindGroup,
    /// Shaders and layouts the colour pipelines are rebuilt from.
    color_source: ColorSource,
    /// Lit, skinned, sky and line pipelines at the scene target's sample count.
    color: ColorPipelines,
    post: Post,
    post_settings: PostSettings,
    sky: Sky,
    sky_sh: [[f32; 4]; 9],
    shadow_range: ShadowRange,
    show_cascades: bool,
    ao: Ao,
    ao_settings: AoSettings,
    fog_settings: FogSettings,
    /// Normals and depth for AO (lit vertex shaders, `fs_prepass`).
    prepass_pipeline: wgpu::RenderPipeline,
    skinned_prepass_pipeline: wgpu::RenderPipeline,
    /// Layout and shared sampler for per-material bind groups (group 1).
    material_bgl: wgpu::BindGroupLayout,
    material_sampler: wgpu::Sampler,
    /// Texture views; 0 white sRGB, 1 white linear, 2 flat normal.
    textures: Vec<wgpu::TextureView>,
    /// Material bind groups; 0 is [`MaterialId::DEFAULT`].
    materials: Vec<wgpu::BindGroup>,
    /// Meshes/textures/materials below these counts survive `clear_custom_meshes`.
    pinned_meshes: usize,
    pinned_textures: usize,
    pinned_materials: usize,
    shadow_pipeline: wgpu::RenderPipeline,
    /// One depth view per cascade (a layer of the shadow array) to render into.
    shadow_layers: Vec<wgpu::TextureView>,
    /// Per cascade: its light matrix, bound at group 3 in the shadow passes.
    casters: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    /// Built-in meshes first (in `MeshKind` order), then runtime-added ones.
    meshes: Vec<GpuMesh>,
    instances: Vec<GrowableBuffer>,
    skinned_shadow_pipeline: wgpu::RenderPipeline,
    joints_bgl: wgpu::BindGroupLayout,
    /// Bound at the shadow passes' unused groups.
    empty_bind_group: wgpu::BindGroup,
    joint_texture: JointTexture,
    /// Each with its material's bind group, held here so the material and
    /// its textures outlive `clear_custom_meshes`.
    skinned_meshes: Vec<(GpuMesh, wgpu::BindGroup)>,
    skinned_instances: Vec<GrowableBuffer>,
    /// Set once a frame has had more joint matrices than fit (logged once).
    joints_overflowed: bool,
    lines: GrowableBuffer,
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
                depth_or_array_layers: CASCADES as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let shadow_layers = (0..CASCADES as u32)
            .map(|layer| {
                shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow cascade"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
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
            entries: &[globals_entry(
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            )],
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
                        view_dimension: wgpu::TextureViewDimension::D2Array,
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
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<MaterialRaw>() as u64),
                    },
                    count: None,
                },
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
                texture_entry(4),
                texture_entry(5),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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
        let globals_layout = layout("globals", &globals_bgl);

        let scene_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene"),
            source: wgpu::ShaderSource::Wgsl(SCENE_WGSL.into()),
        });
        let lines_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lines"),
            source: wgpu::ShaderSource::Wgsl(LINES_WGSL.into()),
        });

        let joints_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("joint matrices"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });
        let empty_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("empty"),
            entries: &[],
        });
        let empty_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("empty"),
            layout: &empty_bgl,
            entries: &[],
        });
        let ao = Ao::new(gpu);
        let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene"),
            bind_group_layouts: &[
                Some(&scene_bgl),
                Some(&material_bgl),
                Some(&empty_bgl),
                Some(ao.lit_layout()),
            ],
            immediate_size: 0,
        });
        let skinned_lit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lit skinned"),
            bind_group_layouts: &[
                Some(&scene_bgl),
                Some(&material_bgl),
                Some(&joints_bgl),
                Some(ao.lit_layout()),
            ],
            immediate_size: 0,
        });
        let caster_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("caster"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size_of::<Mat4>() as u64),
                },
                count: None,
            }],
        });
        let casters = (0..CASCADES)
            .map(|_| {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("caster"),
                    size: size_of::<Mat4>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("caster"),
                    layout: &caster_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                });
                (buffer, bind_group)
            })
            .collect();
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("shadow"),
                    bind_group_layouts: &[
                        Some(&empty_bgl),
                        Some(&empty_bgl),
                        Some(&empty_bgl),
                        Some(&caster_bgl),
                    ],
                    immediate_size: 0,
                }),
            ),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: Some("vs_shadow"),
                compilation_options: Default::default(),
                buffers: &[Some(mesh_layout()), Some(instance_layout())],
            },
            fragment: None,
            primitive: solid(),
            depth_stencil: depth_state(true, wgpu::CompareFunction::LessEqual, SHADOW_BIAS),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let skinned_shadow_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shadow skinned"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("shadow skinned"),
                        bind_group_layouts: &[
                            Some(&empty_bgl),
                            Some(&empty_bgl),
                            Some(&joints_bgl),
                            Some(&caster_bgl),
                        ],
                        immediate_size: 0,
                    }),
                ),
                vertex: wgpu::VertexState {
                    module: &scene_shader,
                    entry_point: Some("vs_shadow_skinned"),
                    compilation_options: Default::default(),
                    buffers: &[Some(skinned_mesh_layout()), Some(skinned_instance_layout())],
                },
                fragment: None,
                primitive: solid(),
                depth_stencil: depth_state(true, wgpu::CompareFunction::LessEqual, SHADOW_BIAS),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        let joint_texture = JointTexture::new(device, &joints_bgl, 1);

        let prepass = |label, layout: &wgpu::PipelineLayout, vs, buffers: &[_]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &scene_shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &scene_shader,
                    entry_point: Some("fs_prepass"),
                    compilation_options: Default::default(),
                    targets: &[Some(NORMAL_FORMAT.into()), Some(VIEW_DEPTH_FORMAT.into())],
                }),
                primitive: solid(),
                depth_stencil: depth_state(
                    true,
                    wgpu::CompareFunction::Less,
                    wgpu::DepthBiasState::default(),
                ),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let prepass_pipeline = prepass(
            "prepass",
            &globals_layout,
            "vs_main",
            &[Some(mesh_layout()), Some(instance_layout())],
        );
        let skinned_prepass_pipeline = prepass(
            "prepass skinned",
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("prepass skinned"),
                bind_group_layouts: &[Some(&globals_bgl), Some(&empty_bgl), Some(&joints_bgl)],
                immediate_size: 0,
            }),
            "vs_skinned",
            &[Some(skinned_mesh_layout()), Some(skinned_instance_layout())],
        );

        let color_source = ColorSource {
            scene_shader,
            lines_shader,
            scene_layout,
            skinned_lit_layout,
            globals_layout,
        };
        let post_settings = PostSettings::default();
        let samples = sample_count(gpu, &post_settings);
        let color = color_source.pipelines(device, gpu.scene_target().format, samples);
        let post = Post::new(gpu, samples);

        let meshes: Vec<GpuMesh> = MeshKind::ALL
            .iter()
            .map(|k| GpuMesh::new(device, &k.data()))
            .collect();
        let instances = (0..meshes.len())
            .map(|_| GrowableBuffer::new(device, "instances", INITIAL_INSTANCE_BYTES))
            .collect();

        let textures = vec![
            upload_texture(gpu, 1, 1, &[&[255, 255, 255, 255]], true),
            upload_texture(gpu, 1, 1, &[&[255, 255, 255, 255]], false),
            upload_texture(gpu, 1, 1, &[&[128, 128, 255, 255]], false),
        ];
        let default_material = material_bind_group(
            gpu,
            &material_bgl,
            &material_sampler,
            &textures,
            &MaterialDesc::default(),
        );
        let sky = Sky::default();
        let sky_sh = sky.irradiance_sh().map(|c| [c.x, c.y, c.z, 0.0]);
        Self {
            color_source,
            color,
            post,
            post_settings,
            sky,
            sky_sh,
            shadow_range: ShadowRange::Distance(DEFAULT_SHADOW_DISTANCE),
            show_cascades: false,
            ao,
            ao_settings: AoSettings::default(),
            fog_settings: FogSettings::default(),
            prepass_pipeline,
            skinned_prepass_pipeline,
            globals,
            scene_bind_group,
            globals_bind_group,
            material_bgl,
            material_sampler,
            textures,
            materials: vec![default_material],
            pinned_meshes: MeshKind::ALL.len(),
            pinned_textures: 3,
            pinned_materials: 1,
            shadow_pipeline,
            shadow_layers,
            casters,
            meshes,
            instances,
            skinned_shadow_pipeline,
            joints_bgl,
            empty_bind_group,
            joint_texture,
            skinned_meshes: Vec::new(),
            skinned_instances: Vec::new(),
            joints_overflowed: false,
            lines: GrowableBuffer::new(device, "lines", 256 * size_of::<LineVertex>() as u64),
            egui: egui_wgpu::Renderer::new(
                device,
                gpu.surface_format(),
                egui_wgpu::RendererOptions::default(),
            ),
        }
    }

    /// Draws one frame and submits it; the caller presents. `target` is the sRGB
    /// view the post pass writes the scene into, `ui_target` the surface-format
    /// view for egui.
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
        let scene_target = gpu.scene_target();
        let samples = sample_count(gpu, &self.post_settings);
        self.post.prepare(device, gpu.size(), samples);
        self.ao.prepare(device, gpu.size());
        if self.color.samples != samples {
            self.color = self
                .color_source
                .pipelines(device, scene_target.format, samples);
        }

        let light_dir = -self.sky.sun_direction.normalize();
        let shadows = cascades(camera.view_proj, camera.eye, light_dir, self.shadow_range);
        for (i, (buffer, _)) in self.casters.iter().enumerate().take(shadows.count) {
            queue.write_buffer(
                buffer,
                0,
                bytemuck::bytes_of(&shadows.view_proj[i].to_cols_array_2d()),
            );
        }
        let rgba = |v: Vec3| v.extend(1.0).to_array();
        let globals = GlobalsRaw {
            view_proj: camera.view_proj.to_cols_array_2d(),
            inv_view_proj: camera.view_proj.inverse().to_cols_array_2d(),
            eye: camera.eye.extend(1.0).to_array(),
            light_dir: light_dir.extend(self.post_settings.exposure).to_array(),
            light_color: rgba(self.sky.sun_color),
            sky_zenith: rgba(self.sky.zenith),
            sky_horizon: rgba(self.sky.horizon),
            sky_ground: rgba(self.sky.ground),
            sh: self.sky_sh,
            post: [if scene_target.hdr { 0.0 } else { 1.0 }, 0.0, 0.0, 0.0],
            cascades: shadows.view_proj.map(|m| m.to_cols_array_2d()),
            shadow_camera: shadows.forward.extend(shadows.near).to_array(),
            cascade_far: shadows.far,
            cascade_texel: shadows.texel,
            shadow_params: [
                shadows.count as f32,
                if self.show_cascades { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
            fog: [
                if self.fog_settings.enabled {
                    self.fog_settings.density
                } else {
                    0.0
                },
                self.fog_settings.height_falloff,
                0.0,
                0.0,
            ],
            ao: [if self.ao_settings.show { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));

        let groups = group_by_mesh(&list.instances, self.meshes.len());
        for (buffer, group) in self.instances.iter_mut().zip(&groups) {
            buffer.write(device, queue, bytemuck::cast_slice(group));
        }
        let skinned_groups = self.prepare_skinned(gpu, list);
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
        for (layer, (_, caster)) in self
            .shadow_layers
            .iter()
            .zip(&self.casters)
            .take(shadows.count)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: layer,
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
            for group in 0..3 {
                pass.set_bind_group(group, &self.empty_bind_group, &[]);
            }
            pass.set_bind_group(3, caster, &[]);
            self.draw_meshes(&mut pass, &groups, false);
            if !skinned_groups.is_empty() {
                pass.set_pipeline(&self.skinned_shadow_pipeline);
                pass.set_bind_group(2, &self.joint_texture.bind_group, &[]);
                self.draw_skinned(&mut pass, &skinned_groups, false);
            }
        }
        let ao_on = self.ao_settings.enabled && scene_target.ao;
        if ao_on {
            let (color, depth) = self.ao.prepass_attachments();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("prepass"),
                color_attachments: &color,
                depth_stencil_attachment: Some(depth),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.prepass_pipeline);
            pass.set_bind_group(0, &self.globals_bind_group, &[]);
            self.draw_meshes(&mut pass, &groups, false);
            if !skinned_groups.is_empty() {
                pass.set_pipeline(&self.skinned_prepass_pipeline);
                pass.set_bind_group(1, &self.empty_bind_group, &[]);
                pass.set_bind_group(2, &self.joint_texture.bind_group, &[]);
                self.draw_skinned(&mut pass, &skinned_groups, false);
            }
        }
        if ao_on {
            self.ao.run(
                queue,
                &mut encoder,
                camera.view_proj,
                camera.eye,
                &self.ao_settings,
            );
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(self.post.color_attachment())],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.post.depth_view(),
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
            pass.set_pipeline(&self.color.lit);
            pass.set_bind_group(0, &self.scene_bind_group, &[]);
            pass.set_bind_group(2, &self.empty_bind_group, &[]);
            pass.set_bind_group(3, self.ao.lit_bind_group(ao_on), &[]);
            self.draw_meshes(&mut pass, &groups, true);
            if !skinned_groups.is_empty() {
                pass.set_pipeline(&self.color.skinned_lit);
                pass.set_bind_group(0, &self.scene_bind_group, &[]);
                pass.set_bind_group(2, &self.joint_texture.bind_group, &[]);
                self.draw_skinned(&mut pass, &skinned_groups, true);
            }

            pass.set_pipeline(&self.color.sky);
            pass.set_bind_group(0, &self.globals_bind_group, &[]);
            pass.draw(0..3, 0..1);

            if !line_vertices.is_empty() {
                let bytes = std::mem::size_of_val(line_vertices.as_slice()) as u64;
                pass.set_pipeline(&self.color.lines);
                pass.set_bind_group(0, &self.globals_bind_group, &[]);
                pass.set_vertex_buffer(0, self.lines.buffer.slice(..bytes));
                pass.draw(0..line_vertices.len() as u32, 0..1);
            }
        }
        self.post
            .run(queue, &mut encoder, target, &self.post_settings);

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

    /// Uploads an RGBA8 texture with its mip chain (`mips[0]` is
    /// `width`×`height`, each next level half the size, down to 1×1). `srgb`:
    /// colour (sampled as sRGB) rather than linear data.
    pub fn add_texture(
        &mut self,
        gpu: &Gpu,
        width: u32,
        height: u32,
        mips: &[&[u8]],
        srgb: bool,
    ) -> TextureId {
        self.textures
            .push(upload_texture(gpu, width, height, mips, srgb));
        TextureId((self.textures.len() - 1) as u32)
    }

    /// Creates a material from textures added with [`Renderer::add_texture`]
    /// (unknown ids fall back to the defaults).
    pub fn add_material(&mut self, gpu: &Gpu, desc: &MaterialDesc) -> MaterialId {
        let group = material_bind_group(
            gpu,
            &self.material_bgl,
            &self.material_sampler,
            &self.textures,
            desc,
        );
        self.materials.push(group);
        MaterialId((self.materials.len() - 1) as u32)
    }

    /// Shadows what is within `half_extent` metres of `focus` (clamped to
    /// 5..100); beyond that, shadows aren't guaranteed. Cascades nearer the
    /// camera than the focus are sharper; none is blurrier than one map over
    /// that circle. Change the extent in coarse steps: a continuously
    /// changing extent makes shadow edges shimmer.
    pub fn set_shadow_focus(&mut self, focus: Vec3, half_extent: f32) {
        let (lo, hi) = SHADOW_RADIUS_RANGE;
        self.shadow_range = ShadowRange::Focus {
            focus,
            radius: half_extent.clamp(lo, hi),
        };
    }

    /// Shadows everything in view out to `distance` metres from the camera
    /// (clamped to 5..500), in cascades that get blurrier with distance. The
    /// default, at 60 m.
    pub fn set_shadow_distance(&mut self, distance: f32) {
        let (lo, hi) = SHADOW_DISTANCE_RANGE;
        self.shadow_range = ShadowRange::Distance(distance.clamp(lo, hi));
    }

    /// Tints the scene by shadow cascade (red, green, blue, yellow) to show
    /// where each one reaches.
    pub fn set_show_cascades(&mut self, show: bool) {
        self.show_cascades = show;
    }

    /// The sky lighting the scene.
    pub fn ao_settings(&self) -> AoSettings {
        self.ao_settings
    }

    /// Takes effect next frame.
    pub fn set_ao_settings(&mut self, settings: AoSettings) {
        self.ao_settings = AoSettings {
            radius: settings.radius.clamp(0.05, 10.0),
            intensity: settings.intensity.clamp(0.1, 8.0),
            ..settings
        };
    }

    pub fn fog_settings(&self) -> FogSettings {
        self.fog_settings
    }

    /// Takes effect next frame.
    pub fn set_fog_settings(&mut self, settings: FogSettings) {
        self.fog_settings = FogSettings {
            density: settings.density.max(0.0),
            height_falloff: settings.height_falloff.max(0.0),
            ..settings
        };
    }

    pub fn post_settings(&self) -> PostSettings {
        self.post_settings
    }

    /// Takes effect next frame. MSAA and bloom are ignored where the adapter
    /// can't do them ([`Renderer::hdr_supported`], [`Renderer::msaa_supported`]).
    pub fn set_post_settings(&mut self, settings: PostSettings) {
        self.post_settings = PostSettings {
            bloom_strength: settings.bloom_strength.clamp(0.0, 1.0),
            exposure: settings.exposure.max(0.0),
            ..settings
        };
    }

    /// Whether the scene renders in HDR (and so whether bloom is available).
    pub fn hdr_supported(&self) -> bool {
        self.post.target().hdr
    }

    /// Whether ambient occlusion can run here (WebGL2 needs float targets).
    pub fn ao_supported(&self) -> bool {
        self.post.target().ao
    }

    pub fn msaa_supported(&self) -> bool {
        self.post.target().msaa
    }

    pub fn sky(&self) -> Sky {
        self.sky
    }

    /// Uploads a mesh with the default material (e.g. a convex hull's shape).
    pub fn add_mesh(&mut self, gpu: &Gpu, data: &MeshData) -> MeshId {
        self.add_mesh_with_material(gpu, data, MaterialId::DEFAULT)
    }

    /// Uploads a mesh drawn with `material` (from [`Renderer::add_material`]).
    pub fn add_mesh_with_material(
        &mut self,
        gpu: &Gpu,
        data: &MeshData,
        material: MaterialId,
    ) -> MeshId {
        let mut mesh = GpuMesh::new(gpu.device(), data);
        mesh.material = if (material.0 as usize) < self.materials.len() {
            material.0 as usize
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

    /// Uploads a skinned mesh drawn with `material`. Skinned meshes live for
    /// the renderer's lifetime, material included: `clear_custom_meshes` and
    /// `reset_custom_meshes` leave them alone (games upload them once).
    pub fn add_skinned_mesh(
        &mut self,
        gpu: &Gpu,
        data: &SkinnedMeshData,
        material: MaterialId,
    ) -> SkinnedMeshId {
        let mesh = GpuMesh::from_parts(
            gpu.device(),
            bytemuck::cast_slice(&data.vertices),
            &data.indices,
        );
        let group = self
            .materials
            .get(material.0 as usize)
            .unwrap_or(&self.materials[0])
            .clone();
        self.skinned_meshes.push((mesh, group));
        self.skinned_instances.push(GrowableBuffer::new(
            gpu.device(),
            "skinned instances",
            64 * size_of::<SkinnedInstanceRaw>() as u64,
        ));
        SkinnedMeshId((self.skinned_meshes.len() - 1) as u32)
    }

    /// Uploads this frame's joint matrices and skinned instances; returns the
    /// instances by mesh (empty when there is nothing to skin, or too much).
    fn prepare_skinned(&mut self, gpu: &Gpu, list: &DrawList) -> Vec<Vec<SkinnedInstanceRaw>> {
        if list.skinned.is_empty() {
            return Vec::new();
        }
        let rows = joint_texture_rows(list.joints.len());
        if rows > MAX_JOINT_ROWS {
            if !self.joints_overflowed {
                log::warn!(
                    "{} joint matrices in one frame; skinned meshes are not drawn",
                    list.joints.len()
                );
                self.joints_overflowed = true;
            }
            return Vec::new();
        }
        let (device, queue) = (gpu.device(), gpu.queue());
        if rows > self.joint_texture.rows {
            self.joint_texture = JointTexture::new(device, &self.joints_bgl, rows);
        }
        let texels = joint_texels(&list.joints, rows);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.joint_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(JOINT_TEXTURE_WIDTH * 16),
                rows_per_image: Some(rows),
            },
            wgpu::Extent3d {
                width: JOINT_TEXTURE_WIDTH,
                height: rows,
                depth_or_array_layers: 1,
            },
        );
        let groups = group_skinned(&list.skinned, self.skinned_meshes.len());
        for (buffer, group) in self.skinned_instances.iter_mut().zip(&groups) {
            buffer.write(device, queue, bytemuck::cast_slice(group));
        }
        groups
    }

    fn draw_skinned(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        groups: &[Vec<SkinnedInstanceRaw>],
        with_material: bool,
    ) {
        for (index, group) in groups.iter().enumerate() {
            let count = group.len() as u32;
            if count == 0 {
                continue;
            }
            let (mesh, material) = &self.skinned_meshes[index];
            let bytes = count as u64 * size_of::<SkinnedInstanceRaw>() as u64;
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_vertex_buffer(1, self.skinned_instances[index].buffer.slice(..bytes));
            if with_material {
                pass.set_bind_group(1, material, &[]);
            }
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..count);
        }
    }

    /// Rebuilds a runtime-added mesh (from [`Renderer::add_mesh`] or
    /// [`Renderer::add_mesh_with_material`]) from new data, keeping its id and
    /// material: for geometry that changes now and then, like a terrain chunk.
    /// Returns `false`, and changes nothing, for built-in or unknown ids.
    pub fn replace_mesh(&mut self, gpu: &Gpu, id: MeshId, data: &MeshData) -> bool {
        if !is_custom(id, self.meshes.len() as u32) {
            return false;
        }
        let mesh = &mut self.meshes[id.0 as usize];
        let material = mesh.material;
        *mesh = GpuMesh::new(gpu.device(), data);
        mesh.material = material;
        true
    }

    /// Frees runtime-added meshes and textures added since the last
    /// [`Renderer::pin_custom_meshes`] (built-ins and pinned ones stay); their
    /// ids become invalid.
    pub fn clear_custom_meshes(&mut self) {
        self.meshes.truncate(self.pinned_meshes);
        self.instances.truncate(self.pinned_meshes);
        self.textures.truncate(self.pinned_textures);
        self.materials.truncate(self.pinned_materials);
    }

    /// Keeps everything uploaded so far (e.g. model meshes) through later
    /// `clear_custom_meshes` calls.
    pub fn pin_custom_meshes(&mut self) {
        self.pinned_meshes = self.meshes.len();
        self.pinned_textures = self.textures.len();
        self.pinned_materials = self.materials.len();
    }

    /// Frees every runtime-added mesh and texture, pinned or not.
    pub fn reset_custom_meshes(&mut self) {
        self.pinned_meshes = MeshKind::ALL.len();
        self.pinned_textures = 3;
        self.pinned_materials = 1;
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
                pass.set_bind_group(1, &self.materials[mesh.material], &[]);
            }
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..count);
        }
    }
}

/// Whether `id` names a runtime-added mesh (not a built-in [`MeshKind`]) among
/// `mesh_count` meshes.
fn is_custom(id: MeshId, mesh_count: u32) -> bool {
    (MeshKind::ALL.len() as u32..mesh_count).contains(&id.0)
}

/// A texture handle from [`Renderer::add_texture`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(pub u32);

/// Uploads an RGBA8 mip chain (sRGB colour or linear data).
fn upload_texture(
    gpu: &Gpu,
    width: u32,
    height: u32,
    mips: &[&[u8]],
    srgb: bool,
) -> wgpu::TextureView {
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("material"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: mips.len().max(1) as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: if srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        },
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
    texture.create_view(&Default::default())
}

/// A material's factors buffer and textures (missing ones → defaults:
/// textures[0] white sRGB, [1] white linear, [2] flat normal).
fn material_bind_group(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    textures: &[wgpu::TextureView],
    desc: &MaterialDesc,
) -> wgpu::BindGroup {
    let raw = MaterialRaw {
        base_color: desc.base_color,
        emissive: [desc.emissive[0], desc.emissive[1], desc.emissive[2], 0.0],
        params: [
            desc.metallic,
            desc.roughness,
            desc.normal_scale,
            desc.occlusion_strength,
        ],
    };
    let factors = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material"),
            contents: bytemuck::bytes_of(&raw),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let pick = |id: Option<TextureId>, fallback: usize| {
        id.and_then(|t| textures.get(t.0 as usize))
            .unwrap_or(&textures[fallback])
    };
    let view = |binding, texture| wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(texture),
    };
    gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("material"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: factors.as_entire_binding(),
            },
            view(1, pick(desc.base_color_texture, 0)),
            view(2, pick(desc.metallic_roughness_texture, 1)),
            view(3, pick(desc.normal_texture, 2)),
            view(4, pick(desc.occlusion_texture, 1)),
            view(5, pick(desc.emissive_texture, 0)),
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Orthographic view from the directional light of the square `half_extent`
/// metres around `focus`.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_runtime_meshes_are_replaceable() {
        let builtins = MeshKind::ALL.len() as u32;
        for kind in MeshKind::ALL {
            assert!(!is_custom(kind.into(), builtins + 3), "{kind:?}");
        }
        assert!(is_custom(MeshId(builtins), builtins + 3));
        assert!(is_custom(MeshId(builtins + 2), builtins + 3));
        assert!(
            !is_custom(MeshId(builtins + 3), builtins + 3),
            "not added yet"
        );
    }
}
