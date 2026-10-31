//! wgpu renderer.
//!
//! Frame: shadow maps (one layer per beam) → forward pass into an HDR target in physical
//! units (nits) → exposure + tonemap into the output view.

mod bloom;
mod primitives;
mod timing;
mod volume;

pub use timing::PASSES as GPU_PASSES;
pub use volume::Haze;

/// Display transform from scene light to screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tonemapper {
    /// AgX: natural roll-off of bright saturated colours to white.
    #[default]
    AgX,
    /// ACES (Narkowicz fit): punchier, but skews hue of very bright saturated colours.
    Aces,
}

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use fixture::{BeamState, FixtureRig, BEAM_DIR_LOCAL, GDTF_TO_Y_UP};
use gdtf_loader::{Material, Mesh, ModelInfo};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const MAX_LIGHTS: usize = 4;
const SHADOW_SIZE: u32 = 2048;
const PROFILE_SAMPLES: usize = 256;
/// Gobo textures are resampled to this size (GDTF recommends 256×256).
pub const GOBO_SIZE: u32 = 256;
/// Wheel slot pitch relative to the aperture (slots sit slightly further apart than the
/// aperture is wide).
const SLOT_PITCH: f32 = 1.05;
/// Dynamic-offset stride for per-draw / per-light uniforms (WebGPU's minimum alignment).
const STRIDE: u64 = 256;
const NEAR: f32 = 0.02;
const FAR: f32 = 200.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightUniform {
    view_proj: [[f32; 4]; 4],
    apex: [f32; 4],
    lens: [f32; 4],
    dir: [f32; 4],
    color: [f32; 4],
    /// Per gobo wheel: layer of slot a, layer of slot b, wheel travel 0..1, rotation (rad).
    gobo: [[f32; 4]; 2],
    /// x = gobo count, y = projector-NDC → field-circle scale, z = 1/focus distance,
    /// w = lens aperture diameter (m).
    gobo_params: [f32; 4],
    /// Colour wheel split: rgb of slot a, w = 1 if active.
    split_a: [f32; 4],
    /// rgb of slot b, w = wheel travel.
    split_b: [f32; 4],
    /// x = tan(half field angle), y = slot pitch.
    optics: [f32; 4],
    /// Projector right / up unit vectors.
    basis_x: [f32; 4],
    basis_y: [f32; 4],
    profile: [f32; PROFILE_SAMPLES],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawUniform {
    model: [[f32; 4]; 4],
    normal: [[f32; 4]; 4],
    base_color: [f32; 4],
    params: [f32; 4],
    emissive: [f32; 4],
}

struct GpuPart {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    material: Material,
}

struct GpuMesh {
    parts: Vec<GpuPart>,
    /// Part that represents the emitting lens (primitive cylinder's bottom cap);
    /// `None` = every part (model files).
    lens_part: Option<usize>,
}

/// Camera parameters supplied by the app each frame.
#[derive(Debug, Clone, Copy)]
pub struct CameraParams {
    pub eye: Vec3,
    pub target: Vec3,
    pub fov_y_deg: f32,
    pub aspect: f32,
}

struct Targets {
    size: (u32, u32),
    depth: wgpu::TextureView,
    hdr: wgpu::TextureView,
    tonemap_bg: wgpu::BindGroup,
}

pub struct Renderer {
    forward: wgpu::RenderPipeline,
    shadow: wgpu::RenderPipeline,
    tonemap: wgpu::RenderPipeline,

    camera_buf: wgpu::Buffer,
    lights_buf: wgpu::Buffer,
    scene_bg: wgpu::BindGroup,

    shadow_cam_buf: wgpu::Buffer,
    shadow_cam_bg: wgpu::BindGroup,
    shadow_layers: Vec<wgpu::TextureView>,
    scene_layout: wgpu::BindGroupLayout,
    shadow_array: wgpu::TextureView,
    shadow_sampler: wgpu::Sampler,
    gobo_view: wgpu::TextureView,
    gobo_sampler: wgpu::Sampler,
    /// Media name → layer in the gobo texture array (layer 0 = open aperture).
    gobo_layers: HashMap<String, u32>,

    draw_layout: wgpu::BindGroupLayout,
    draw_buf: wgpu::Buffer,
    draw_bg: wgpu::BindGroup,
    draw_capacity: u64,

    tonemap_layout: wgpu::BindGroupLayout,
    tonemap_buf: wgpu::Buffer,
    targets: Option<Targets>,

    stage: GpuMesh,
    models: HashMap<String, GpuMesh>,
    timer: Option<timing::GpuTimer>,
    volume: volume::VolumePass,
    bloom: bloom::BloomPass,
    linear_sampler: wgpu::Sampler,
    /// Volumetric haze settings.
    pub haze: Haze,
    /// Glare: fraction of the image replaced by its blurred version (0 = off).
    pub bloom_strength: f32,
    pub tonemapper: Tonemapper,

    /// Where the fixture's base sits in the world (Y-up), e.g. hung from a truss.
    pub fixture_placement: Mat4,
    /// Camera exposure value at ISO 100. Higher = darker image.
    pub ev100: f32,
    /// Illuminance of the house/work lights, lux.
    pub house_lux: f32,
}

/// Where fixtures hang by default: from the pipe at trim height.
pub fn default_fixture_placement() -> Mat4 {
    Mat4::from_translation(primitives::TRIM_HEIGHT * Vec3::Y)
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages, dynamic: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

impl Renderer {
    /// `output_format` is the format of the view passed to [`Renderer::render`]; use an
    /// sRGB format so the display transfer curve is applied by the hardware.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, output_format: wgpu::TextureFormat) -> Self {
        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;

        // --- Bind group layouts ---
        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene"),
            entries: &[
                uniform_entry(0, vf, false),
                uniform_entry(1, wgpu::ShaderStages::FRAGMENT, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("draw"),
            entries: &[uniform_entry(0, vf, true)],
        });
        let shadow_cam_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow camera"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX, true)],
        });
        let tonemap_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                uniform_entry(1, wgpu::ShaderStages::FRAGMENT, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // --- Pipelines ---
        let forward_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward"),
            source: wgpu::ShaderSource::Wgsl([include_str!("common.wgsl"), include_str!("forward.wgsl")].concat().into()),
        });
        let shadow_sm = device.create_shader_module(wgpu::include_wgsl!("shadow.wgsl"));
        let tonemap_sm = device.create_shader_module(wgpu::include_wgsl!("tonemap.wgsl"));

        let forward_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("forward"),
            bind_group_layouts: &[Some(&scene_layout), Some(&draw_layout)],
            immediate_size: 0,
        });
        let forward = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("forward"),
            layout: Some(&forward_pl),
            vertex: wgpu::VertexState {
                module: &forward_sm,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(vertex_layout())],
            },
            // Fixture meshes are often not consistently wound; draw both sides.
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &forward_sm,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(HDR_FORMAT.into())],
            }),
            multiview_mask: None,
            cache: None,
        });

        let shadow_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow"),
            bind_group_layouts: &[Some(&shadow_cam_layout), Some(&draw_layout)],
            immediate_size: 0,
        });
        let shadow = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow"),
            layout: Some(&shadow_pl),
            vertex: wgpu::VertexState {
                module: &shadow_sm,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(vertex_layout())],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });

        let tonemap_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tonemap"),
            bind_group_layouts: &[Some(&tonemap_layout)],
            immediate_size: 0,
        });
        let tonemap = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tonemap"),
            layout: Some(&tonemap_pl),
            vertex: wgpu::VertexState {
                module: &tonemap_sm,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &tonemap_sm,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(output_format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });

        // --- Resources ---
        let camera_buf = uniform_buffer(device, "camera", std::mem::size_of::<CameraUniform>() as u64);
        let lights_buf = uniform_buffer(
            device,
            "lights",
            (std::mem::size_of::<LightUniform>() * MAX_LIGHTS) as u64,
        );
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow maps"),
            size: wgpu::Extent3d {
                width: SHADOW_SIZE,
                height: SHADOW_SIZE,
                depth_or_array_layers: MAX_LIGHTS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_array = shadow_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let shadow_layers = (0..MAX_LIGHTS as u32)
            .map(|l| {
                shadow_tex.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: l,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let (_, gobo_view) = create_gobo_texture(device, queue, &[]);
        let gobo_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("gobo"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let scene_bg = scene_bind_group(
            device,
            &scene_layout,
            &camera_buf,
            &lights_buf,
            &shadow_array,
            &shadow_sampler,
            &gobo_view,
            &gobo_sampler,
        );

        let shadow_cam_buf = uniform_buffer(device, "shadow cameras", STRIDE * MAX_LIGHTS as u64);
        let shadow_cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow cameras"),
            layout: &shadow_cam_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &shadow_cam_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(64),
                }),
            }],
        });

        let draw_capacity = 64;
        let (draw_buf, draw_bg) = Self::make_draw_buffer(device, &draw_layout, draw_capacity);
        let tonemap_buf = uniform_buffer(device, "tonemap", 16);
        let stage = upload_mesh(device, &primitives::stage(), None);
        let volume = volume::VolumePass::new(device, &scene_layout);

        Self {
            forward,
            shadow,
            tonemap,
            camera_buf,
            lights_buf,
            scene_bg,
            shadow_cam_buf,
            shadow_cam_bg,
            shadow_layers,
            scene_layout,
            shadow_array,
            shadow_sampler,
            gobo_view,
            gobo_sampler,
            gobo_layers: HashMap::new(),
            draw_layout,
            draw_buf,
            draw_bg,
            draw_capacity,
            tonemap_layout,
            tonemap_buf,
            targets: None,
            stage,
            models: HashMap::new(),
            timer: timing::GpuTimer::new(device, queue),
            volume,
            bloom: bloom::BloomPass::new(device),
            linear_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("linear"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            haze: Haze::default(),
            bloom_strength: 0.04,
            tonemapper: Tonemapper::default(),
            fixture_placement: default_fixture_placement(),
            ev100: 7.0,
            house_lux: 15.0,
        }
    }

    fn make_draw_buffer(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        capacity: u64,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buf = uniform_buffer(device, "draws", capacity * STRIDE);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draws"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<DrawUniform>() as u64),
                }),
            }],
        });
        (buf, bg)
    }

    /// Uploads every model of a fixture. Call again when loading another fixture.
    pub fn set_fixture(&mut self, device: &wgpu::Device, models: &HashMap<String, ModelInfo>) {
        self.models.clear();
        for (name, m) in models {
            let lens_part = m.mesh.is_none().then_some(1);
            let mesh = match &m.mesh {
                Some(mesh) => Some(mesh.clone()),
                None => primitives::primitive_mesh(m.primitive, m.size),
            };
            if let Some(mesh) = mesh {
                self.models.insert(name.clone(), upload_mesh(device, &mesh, lens_part));
            }
        }
    }

    /// Uploads gobo / wheel images as light-transmission maps (`GOBO_SIZE`², linear RGB),
    /// keyed by media name as referenced by [`fixture::GoboLook`].
    pub fn set_gobos(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, gobos: &[(String, Vec<[f32; 3]>)]) {
        let (layers, view) = create_gobo_texture(device, queue, gobos);
        self.gobo_layers = layers;
        self.gobo_view = view;
        self.scene_bg = scene_bind_group(
            device,
            &self.scene_layout,
            &self.camera_buf,
            &self.lights_buf,
            &self.shadow_array,
            &self.shadow_sampler,
            &self.gobo_view,
            &self.gobo_sampler,
        );
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        if self.targets.as_ref().is_some_and(|t| t.size == size) {
            return;
        }
        let tex = |label, format, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size.0,
                        height: size.1,
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
        };
        let depth = tex(
            "depth",
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        self.volume.resize(device, size, &depth);
        let hdr = tex(
            "hdr",
            HDR_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let bloom = self.bloom.resize(device, size, &hdr);
        let tonemap_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap"),
            layout: &self.tonemap_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.tonemap_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&bloom),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });
        self.targets = Some(Targets {
            size,
            depth,
            hdr,
            tonemap_bg,
        });
    }

    /// World position (Y-up) of a rig node given its fixture-space transform.
    pub fn world_position(&self, fixture_transform: Mat4) -> Vec3 {
        (self.fixture_placement * GDTF_TO_Y_UP * fixture_transform).w_axis.truncate()
    }

    /// Latest GPU time per pass in milliseconds (names in [`GPU_PASSES`]), if the device
    /// supports timestamp queries. Lags a frame or two behind.
    pub fn gpu_pass_times(&self) -> Option<[f32; GPU_PASSES.len()]> {
        self.timer.as_ref().map(|t| t.last)
    }

    /// Exposure multiplier from EV100 (standard photographic exposure, Lagarde & de Rousiers).
    pub fn exposure(&self) -> f32 {
        1.0 / (1.2 * 2f32.powf(self.ev100))
    }

    /// Renders one frame. `transforms` comes from [`FixtureRig::world_transforms`].
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        size: (u32, u32),
        camera: &CameraParams,
        rig: &FixtureRig,
        transforms: &[Mat4],
        beams: &[BeamState],
    ) {
        self.ensure_targets(device, size);
        if let Some(t) = &mut self.timer {
            t.begin_frame();
        }
        let root = self.fixture_placement * GDTF_TO_Y_UP;

        // --- Lights ---
        let mut lights = vec![LightUniform::zeroed(); MAX_LIGHTS];
        let mut shadow_cams = vec![0u8; (STRIDE as usize) * MAX_LIGHTS];
        // Beam node → (light index, lens luminance scale).
        let mut light_of_node: HashMap<usize, (usize, f32)> = HashMap::new();
        let lit = cluster_beams(rig, beams, transforms, root);
        for (i, c) in lit.iter().enumerate() {
            let (light, view_proj) = light_uniform(&c.beam, c.world, i, &self.gobo_layers);
            lights[i] = light;
            shadow_cams[i * STRIDE as usize..i * STRIDE as usize + 64]
                .copy_from_slice(bytemuck::bytes_of(&view_proj.to_cols_array_2d()));
            for &(node, scale) in &c.members {
                light_of_node.insert(node, (i, scale));
            }
        }
        queue.write_buffer(&self.lights_buf, 0, bytemuck::cast_slice(&lights));
        queue.write_buffer(&self.shadow_cam_buf, 0, &shadow_cams);

        // --- Camera ---
        let view = glam::camera::rh::view::look_at_mat4(camera.eye, camera.target, Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(
            camera.fov_y_deg.to_radians(),
            camera.aspect,
            NEAR,
            FAR,
        );
        let cam = CameraUniform {
            view_proj: (proj * view).to_cols_array_2d(),
            eye: camera.eye.extend(1.0).into(),
            params: [self.house_lux, lit.len() as f32, 0.0, 0.0],
        };
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cam));
        queue.write_buffer(
            &self.tonemap_buf,
            0,
            bytemuck::bytes_of(&[
                self.exposure(),
                self.bloom_strength.clamp(0.0, 1.0),
                if self.tonemapper == Tonemapper::AgX { 1.0 } else { 0.0 },
                self.bloom.mip_count() as f32,
            ]),
        );

        // --- Draw list ---
        let mut draws: Vec<(&GpuMesh, usize, DrawUniform)> = Vec::new();
        for (i, p) in self.stage.parts.iter().enumerate() {
            draws.push((&self.stage, i, draw_uniform(Mat4::IDENTITY, &p.material, None)));
        }
        for (idx, (node, world)) in rig.nodes.iter().zip(transforms).enumerate() {
            let Some(mesh) = node.model.as_ref().and_then(|m| self.models.get(m)) else {
                continue;
            };
            let mut model = root * *world;
            if matches!(node.kind, fixture::NodeKind::Beam(_)) {
                // The lens sits in the Beam geometry's plane, which is usually also the
                // housing's front face: nudge it 2 mm forward so it isn't buried in it.
                model *= Mat4::from_translation(BEAM_DIR_LOCAL * 0.002);
            }
            let light = light_of_node.get(&idx).copied();
            for (i, p) in mesh.parts.iter().enumerate() {
                let lens = light.filter(|_| mesh.lens_part.is_none_or(|l| l == i));
                draws.push((mesh, i, draw_uniform(model, &p.material, lens)));
            }
        }
        if draws.len() as u64 > self.draw_capacity {
            self.draw_capacity = (draws.len() as u64).next_power_of_two();
            (self.draw_buf, self.draw_bg) = Self::make_draw_buffer(device, &self.draw_layout, self.draw_capacity);
        }
        let mut bytes = vec![0u8; draws.len() * STRIDE as usize];
        for (i, (_, _, u)) in draws.iter().enumerate() {
            let o = i * STRIDE as usize;
            bytes[o..o + std::mem::size_of::<DrawUniform>()].copy_from_slice(bytemuck::bytes_of(u));
        }
        queue.write_buffer(&self.draw_buf, 0, &bytes);

        let draw_all = |pass: &mut wgpu::RenderPass, draw_bg: &wgpu::BindGroup| {
            for (i, (mesh, part, _)) in draws.iter().enumerate() {
                let p = &mesh.parts[*part];
                pass.set_bind_group(1, draw_bg, &[(i as u64 * STRIDE) as u32]);
                pass.set_vertex_buffer(0, p.vertices.slice(..));
                pass.set_index_buffer(p.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..p.index_count, 0, 0..1);
            }
        };

        let targets = self.targets.as_ref().unwrap();
        let mut enc = device.create_command_encoder(&Default::default());

        // --- Shadow maps ---
        for i in 0..lit.len() {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_layers[i],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: self.timer.as_ref().and_then(|t| t.writes(0, i == 0, i + 1 == lit.len())),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.shadow);
            pass.set_bind_group(0, &self.shadow_cam_bg, &[(i as u64 * STRIDE) as u32]);
            draw_all(&mut pass, &self.draw_bg);
        }

        // --- Forward (HDR) ---
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("forward"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.hdr,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        // Kept for the volume passes.
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: self.timer.as_ref().and_then(|t| t.writes(1, true, true)),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.forward);
            pass.set_bind_group(0, &self.scene_bg, &[]);
            draw_all(&mut pass, &self.draw_bg);
        }

        // --- Volumetric haze ---
        if self.haze.enabled {
            self.volume.encode(
                queue,
                &mut enc,
                &self.scene_bg,
                &targets.hdr,
                &volume::VolumeFrame {
                    view_proj: proj * view,
                    eye: camera.eye,
                    near: NEAR,
                    far: FAR,
                },
                &self.haze,
                self.timer.as_ref(),
                2,
            );
        } else {
            self.volume.reset_history();
        }

        // --- Bloom ---
        if self.bloom_strength > 0.0 {
            self.bloom.encode(queue, &mut enc, self.exposure(), self.timer.as_ref(), 3);
        }

        // --- Tonemap ---
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tonemap"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: self.timer.as_ref().and_then(|t| t.writes(4, true, true)),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.tonemap);
            pass.set_bind_group(0, &targets.tonemap_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        if let Some(t) = &self.timer {
            t.resolve(&mut enc);
        }
        queue.submit([enc.finish()]);
        if let Some(t) = &mut self.timer {
            t.after_submit();
        }
    }
}

/// A light as rendered: one beam, or several beams merged when there are more than
/// [`MAX_LIGHTS`] (multi-cell fixtures).
struct LightSource {
    beam: BeamState,
    /// World transform (Y-up) of the emitting surface's centre.
    world: Mat4,
    /// Member beam nodes and the factor that turns the merged light's lens luminance
    /// into that member's own.
    members: Vec<(usize, f32)>,
}

/// Groups beams into at most [`MAX_LIGHTS`] lights. With few beams each is its own light.
/// Otherwise beams are grouped by the nearest geometry they share (e.g. a pixel zone),
/// moving up the tree until few enough groups remain; a group becomes one light with the
/// summed intensity, centred on its flux-weighted centroid and wide enough to cover all
/// its cells, so near field and shadows stay plausible. Assumes the beams in a group are
/// alike (same optics), which holds for pixel rings and LED arrays.
fn cluster_beams(rig: &FixtureRig, beams: &[BeamState], transforms: &[Mat4], root: Mat4) -> Vec<LightSource> {
    let single = |b: &BeamState| LightSource {
        beam: b.clone(),
        world: root * transforms[b.node],
        members: vec![(b.node, 1.0)],
    };
    if beams.len() <= MAX_LIGHTS {
        return beams.iter().map(single).collect();
    }

    // Beams below each node.
    let mut below = vec![0usize; rig.nodes.len()];
    for b in beams {
        let mut n = Some(b.node);
        while let Some(i) = n {
            below[i] += 1;
            n = rig.nodes[i].parent;
        }
    }
    // Start at the nearest ancestor shared with another beam.
    let shared = |mut n: usize| {
        while let Some(p) = rig.nodes[n].parent {
            n = p;
            if below[n] > 1 {
                break;
            }
        }
        n
    };
    let mut keys: Vec<usize> = beams.iter().map(|b| shared(b.node)).collect();
    loop {
        let mut distinct = keys.clone();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() <= MAX_LIGHTS {
            break;
        }
        let before = keys.clone();
        keys = keys.iter().map(|&k| rig.nodes[k].parent.map_or(k, shared)).collect();
        if keys == before {
            break; // at the root and still too many (several top-level trees)
        }
    }

    let mut groups: Vec<(usize, Vec<&BeamState>)> = Vec::new();
    for (b, k) in beams.iter().zip(keys) {
        match groups.iter_mut().find(|(g, _)| *g == k) {
            Some((_, v)) => v.push(b),
            None => groups.push((k, vec![b])),
        }
    }
    if groups.len() > MAX_LIGHTS {
        log::warn!("{} light groups, only the first {MAX_LIGHTS} cast light", groups.len());
        groups.truncate(MAX_LIGHTS);
    }

    groups
        .into_iter()
        .map(|(_, members)| {
            if members.len() == 1 {
                return single(members[0]);
            }
            let flux: f32 = members.iter().map(|b| b.info.luminous_flux.max(1e-3)).sum();
            let centre = members
                .iter()
                .map(|b| (root * transforms[b.node]).w_axis.truncate() * b.info.luminous_flux.max(1e-3))
                .sum::<Vec3>()
                / flux;
            let radius = members
                .iter()
                .map(|b| (root * transforms[b.node]).w_axis.truncate().distance(centre) + b.info.beam_radius)
                .fold(0.0, f32::max);
            let mut beam = members[0].clone();
            beam.info.luminous_flux = flux;
            beam.info.beam_radius = radius;
            beam.peak_cd = members.iter().map(|b| b.peak_cd).sum();
            let mut world = root * transforms[members[0].node];
            world.w_axis = centre.extend(1.0);
            // Lens luminance: merged = I/(πR²); a member's own = Iᵢ/(πrᵢ²).
            let members = members
                .iter()
                .map(|b| {
                    let r = b.info.beam_radius.max(1e-3);
                    (b.node, (b.peak_cd / beam.peak_cd.max(1e-6)) * (radius * radius) / (r * r))
                })
                .collect();
            LightSource { beam, world, members }
        })
        .collect()
}

/// Builds the GPU light for a beam whose Beam geometry has world transform `world`.
/// Returns the uniform and the shadow projector's view-projection.
fn light_uniform(
    b: &BeamState,
    world: Mat4,
    shadow_layer: usize,
    gobo_layers: &HashMap<String, u32>,
) -> (LightUniform, Mat4) {
    let lens = world.w_axis.truncate();
    let dir = world.transform_vector3(BEAM_DIR_LOCAL).normalize();
    let max_angle = b.profile.max_angle().to_radians();
    // Light leaves a disc of `beam_radius` at the lens; treat it as coming from a virtual
    // apex behind the lens so the cone has the right width at the lens.
    let radius = b.info.beam_radius.max(0.001);
    let apex_offset = (radius / b.profile.half_field.to_radians().tan().max(1e-3)).min(3.0);
    let apex = lens - dir * apex_offset;

    let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let view = glam::camera::rh::view::look_to_mat4(apex, dir, up);
    let fov = (2.0 * max_angle + 2f32.to_radians()).min(170f32.to_radians());
    // Near plane just past the lens so the fixture's own head doesn't shadow its beam.
    let proj = glam::camera::rh::proj::directx::perspective(fov, 1.0, apex_offset + 0.02, 80.0);
    let view_proj = proj * view;

    let table = b.profile.table(PROFILE_SAMPLES);
    let mut profile = [0f32; PROFILE_SAMPLES];
    profile.copy_from_slice(&table);
    let cd = if b.emits_light() { b.intensity_cd() } else { 0.0 };
    let c = b.look.color;

    let layer = |m: &Option<String>| m.as_ref().and_then(|m| gobo_layers.get(m)).copied().unwrap_or(0) as f32;
    let mut gobo = [[0.0; 4]; 2];
    for (g, look) in gobo.iter_mut().zip(&b.look.gobos) {
        *g = [layer(&look.media_a), layer(&look.media_b), look.frac, look.rotation_deg.to_radians()];
    }
    let tan_half_field = b.profile.half_field.to_radians().tan().max(1e-4);
    let (split_a, split_b) = match &b.look.color_split {
        Some(sp) => (
            [sp.a[0], sp.a[1], sp.a[2], 1.0],
            [sp.b[0], sp.b[1], sp.b[2], sp.frac],
        ),
        None => ([1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]),
    };
    let light = LightUniform {
        view_proj: view_proj.to_cols_array_2d(),
        apex: apex.extend(max_angle).into(),
        lens: lens.extend(radius).into(),
        dir: dir.extend(shadow_layer as f32).into(),
        color: [c[0] * cd, c[1] * cd, c[2] * cd, 1.0],
        gobo,
        gobo_params: [
            b.look.gobos.len().min(2) as f32,
            (fov * 0.5).tan() / tan_half_field,
            1.0 / b.look.focus_distance.max(0.1),
            2.0 * radius,
        ],
        split_a,
        split_b,
        optics: [tan_half_field, SLOT_PITCH, 0.0, 0.0],
        // Same frame as glam's look_to_rh used for the shadow projector.
        basis_x: dir.cross(up).normalize().extend(0.0).into(),
        basis_y: dir.cross(up).normalize().cross(dir).extend(0.0).into(),
        profile,
    };
    (light, view_proj)
}

/// `lens`: (light index, luminance scale) if this part is the emitting lens of a beam.
fn draw_uniform(model: Mat4, m: &Material, lens: Option<(usize, f32)>) -> DrawUniform {
    DrawUniform {
        model: model.to_cols_array_2d(),
        normal: model.inverse().transpose().to_cols_array_2d(),
        base_color: m.base_color,
        params: [
            m.metallic,
            m.roughness,
            lens.map_or(0.0, |(l, _)| l as f32 + 1.0),
            lens.map_or(1.0, |(_, s)| s),
        ],
        emissive: [0.0; 4],
    }
}

fn upload_mesh(device: &wgpu::Device, mesh: &Mesh, lens_part: Option<usize>) -> GpuMesh {
    let parts = mesh
        .parts
        .iter()
        .filter(|p| !p.indices.is_empty())
        .map(|p| {
            let verts: Vec<Vertex> = p
                .positions
                .iter()
                .zip(&p.normals)
                .map(|(&pos, &normal)| Vertex { pos, normal })
                .collect();
            GpuPart {
                vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("vertices"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("indices"),
                    contents: bytemuck::cast_slice(&p.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
                index_count: p.indices.len() as u32,
                material: p.material,
            }
        })
        .collect();
    GpuMesh { parts, lens_part }
}

#[allow(clippy::too_many_arguments)]
fn scene_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera: &wgpu::Buffer,
    lights: &wgpu::Buffer,
    shadow_array: &wgpu::TextureView,
    shadow_sampler: &wgpu::Sampler,
    gobo_view: &wgpu::TextureView,
    gobo_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: lights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(shadow_array),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(shadow_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(gobo_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(gobo_sampler),
            },
        ],
    })
}

/// The open aperture: a clear disc inscribed in the square (the slot's hole), anti-aliased.
fn open_aperture() -> Vec<[f32; 3]> {
    aperture_disc(0.0)
}

/// Anti-aliased disc inscribed in the gobo square, shrunk by `inset` texels.
fn aperture_disc(inset: f32) -> Vec<[f32; 3]> {
    let n = GOBO_SIZE as usize;
    let mut px = Vec::with_capacity(n * n);
    for y in 0..n {
        for x in 0..n {
            let (dx, dy) = (x as f32 + 0.5 - n as f32 / 2.0, y as f32 + 0.5 - n as f32 / 2.0);
            let r = (dx * dx + dy * dy).sqrt();
            let t = (n as f32 / 2.0 - inset - r + 0.5).clamp(0.0, 1.0);
            px.push([t; 3]);
        }
    }
    px
}

fn linear_to_srgb8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

/// Builds the gobo texture array (layer 0 = open) with a full mip chain. Mips are
/// box-filtered in linear light and stored sRGB-encoded for precision in the darks;
/// they double as focus blur.
fn create_gobo_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gobos: &[(String, Vec<[f32; 3]>)],
) -> (HashMap<String, u32>, wgpu::TextureView) {
    let n = GOBO_SIZE;
    let mips = n.ilog2() + 1;
    // Gobos are masked slightly inside the open aperture: images often have clear (white)
    // surroundings starting right at the circle, which resampling bleeds inwards; at the
    // aperture's soft edge that would show up as a thin bright ring.
    let gobo_mask = aperture_disc(2.0);
    let mut layers: Vec<Vec<[f32; 3]>> = vec![open_aperture()];
    let mut map = HashMap::new();
    for (name, px) in gobos {
        if px.len() != (n * n) as usize {
            log::warn!("gobo '{name}' has {} pixels, expected {}", px.len(), n * n);
            continue;
        }
        map.insert(name.clone(), layers.len() as u32);
        // The gobo sits in the round field stop: mask it, so nothing outside the
        // inscribed circle passes (some files have opaque white corners there).
        layers.push(px.iter().zip(&gobo_mask).map(|(p, a)| p.map(|c| c * a[0])).collect());
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gobos"),
        size: wgpu::Extent3d {
            width: n,
            height: n,
            depth_or_array_layers: layers.len() as u32,
        },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (layer, base) in layers.iter().enumerate() {
        let mut level = base.clone();
        let mut size = n as usize;
        for mip in 0..mips {
            let bytes: Vec<u8> = level
                .iter()
                .flat_map(|c| [linear_to_srgb8(c[0]), linear_to_srgb8(c[1]), linear_to_srgb8(c[2]), 255])
                .collect();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: mip,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * size as u32),
                    rows_per_image: Some(size as u32),
                },
                wgpu::Extent3d {
                    width: size as u32,
                    height: size as u32,
                    depth_or_array_layers: 1,
                },
            );
            if size == 1 {
                break;
            }
            let half = size / 2;
            let mut next = vec![[0.0f32; 3]; half * half];
            for y in 0..half {
                for x in 0..half {
                    let mut acc = [0.0; 3];
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let p = level[(2 * y + dy) * size + 2 * x + dx];
                        for i in 0..3 {
                            acc[i] += p[i] * 0.25;
                        }
                    }
                    next[y * half + x] = acc;
                }
            }
            level = next;
            size = half;
        }
    }
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (map, view)
}
