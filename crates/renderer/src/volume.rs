//! Volumetric beams: half-resolution ray-march → temporal accumulation → depth-aware
//! upsample and composite into the HDR target.

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use crate::{timing::GpuTimer, HDR_FORMAT};

const VOLUME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Haze / volumetric settings.
#[derive(Debug, Clone, Copy)]
pub struct Haze {
    pub enabled: bool,
    /// Scattering coefficient σs in 1/m (extinction is taken equal: pure scattering).
    /// Light stage haze ≈ 0.01–0.03, thick haze ≈ 0.1+.
    pub density: f32,
    /// Henyey–Greenstein anisotropy: 0 = isotropic, ~0.7 = strongly forward (water haze).
    pub anisotropy: f32,
    /// Ray-march steps per beam per pixel.
    pub steps: u32,
    /// Blend with previous frames to remove the noise of a low step count.
    pub temporal: bool,
}

impl Default for Haze {
    fn default() -> Self {
        Self {
            enabled: true,
            density: 0.03,
            anisotropy: 0.6,
            steps: 32,
            temporal: true,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VolumeUniform {
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    haze: [f32; 4],
    sizes: [f32; 4],
    misc: [f32; 4],
    /// x = full-res pixels per volume pixel (2 = half resolution, 4 = quarter).
    extra: [f32; 4],
}

/// Camera data the volume needs for one frame.
pub struct VolumeFrame {
    pub view_proj: Mat4,
    pub eye: glam::Vec3,
    pub near: f32,
    pub far: f32,
}

struct Targets {
    scale: u32,
    full: (u32, u32),
    half: (u32, u32),
    raw: wgpu::TextureView,
    accum: [wgpu::TextureView; 2],
    march_bg: wgpu::BindGroup,
    /// Indexed by the accumulation buffer written this frame.
    taa_bg: [wgpu::BindGroup; 2],
    composite_bg: [wgpu::BindGroup; 2],
}

pub struct VolumePass {
    march: wgpu::RenderPipeline,
    taa: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    march_layout: wgpu::BindGroupLayout,
    taa_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    targets: Option<Targets>,
    frame: u32,
    prev_view_proj: Option<Mat4>,
}

fn shader(device: &wgpu::Device, label: &str, parts: &[&str]) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(parts.concat().into()),
    })
}

fn tex_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    layouts: &[Option<&wgpu::BindGroupLayout>],
    target: wgpu::ColorTargetState,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: layouts,
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_fullscreen"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(target)],
        }),
        multiview_mask: None,
        cache: None,
    })
}

impl VolumePass {
    pub fn new(device: &wgpu::Device, scene_layout: &wgpu::BindGroupLayout) -> Self {
        let common = include_str!("common.wgsl");
        let vcommon = include_str!("volume_common.wgsl");
        let depth = wgpu::TextureSampleType::Depth;
        let float = wgpu::TextureSampleType::Float { filterable: true };

        let march_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume march"),
            entries: &[uniform_entry(0), tex_entry(1, depth)],
        });
        let taa_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume taa"),
            entries: &[
                uniform_entry(0),
                tex_entry(1, depth),
                tex_entry(2, float),
                tex_entry(3, float),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume composite"),
            entries: &[uniform_entry(0), tex_entry(1, depth), tex_entry(2, float)],
        });

        let march = fullscreen_pipeline(
            device,
            "volume march",
            &shader(device, "volume march", &[common, vcommon, include_str!("volume.wgsl")]),
            &[Some(scene_layout), Some(&march_layout)],
            VOLUME_FORMAT.into(),
        );
        let taa = fullscreen_pipeline(
            device,
            "volume taa",
            &shader(device, "volume taa", &[vcommon, include_str!("volume_taa.wgsl")]),
            &[Some(&taa_layout)],
            VOLUME_FORMAT.into(),
        );
        let composite = fullscreen_pipeline(
            device,
            "volume composite",
            &shader(device, "volume composite", &[vcommon, include_str!("volume_composite.wgsl")]),
            &[Some(&composite_layout)],
            wgpu::ColorTargetState {
                format: HDR_FORMAT,
                // scene · transmittance + in-scattered light
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::SrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            },
        );

        Self {
            march,
            taa,
            composite,
            march_layout,
            taa_layout,
            composite_layout,
            uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("volume"),
                size: std::mem::size_of::<VolumeUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("volume history"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            targets: None,
            frame: 0,
            prev_view_proj: None,
        }
    }

    /// (Re)creates the half-resolution targets. `depth` is the full-resolution scene depth.
    /// Full-res pixels per volume pixel of the current targets.
    pub fn scale(&self) -> u32 {
        self.targets.as_ref().map_or(0, |t| t.scale)
    }

    pub fn resize(&mut self, device: &wgpu::Device, full: (u32, u32), depth: &wgpu::TextureView, scale: u32) {
        let scale = scale.clamp(1, 8);
        let half = (full.0.div_ceil(scale).max(1), full.1.div_ceil(scale).max(1));
        let tex = |label| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: half.0,
                        height: half.1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: VOLUME_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let raw = tex("volume raw");
        let accum = [tex("volume accum 0"), tex("volume accum 1")];
        let entry = |binding, resource| wgpu::BindGroupEntry { binding, resource };
        let march_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume march"),
            layout: &self.march_layout,
            entries: &[
                entry(0, self.uniform.as_entire_binding()),
                entry(1, wgpu::BindingResource::TextureView(depth)),
            ],
        });
        let taa_bg = [0, 1].map(|cur| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("volume taa"),
                layout: &self.taa_layout,
                entries: &[
                    entry(0, self.uniform.as_entire_binding()),
                    entry(1, wgpu::BindingResource::TextureView(depth)),
                    entry(2, wgpu::BindingResource::TextureView(&raw)),
                    entry(3, wgpu::BindingResource::TextureView(&accum[1 - cur])),
                    entry(4, wgpu::BindingResource::Sampler(&self.sampler)),
                ],
            })
        });
        let composite_bg = [0, 1].map(|cur| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("volume composite"),
                layout: &self.composite_layout,
                entries: &[
                    entry(0, self.uniform.as_entire_binding()),
                    entry(1, wgpu::BindingResource::TextureView(depth)),
                    entry(2, wgpu::BindingResource::TextureView(&accum[cur])),
                ],
            })
        });
        self.targets = Some(Targets {
            scale,
            full,
            half,
            raw,
            accum,
            march_bg,
            taa_bg,
            composite_bg,
        });
        self.prev_view_proj = None; // history is gone
    }

    /// Records the three volume passes. Must run after the forward pass (needs its depth)
    /// and before tonemapping.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        scene_bg: &wgpu::BindGroup,
        hdr: &wgpu::TextureView,
        frame: &VolumeFrame,
        haze: &Haze,
        timer: Option<&GpuTimer>,
        timer_slot: usize,
    ) {
        let Some(t) = &self.targets else { return };
        let cur = (self.frame % 2) as usize;
        let history_weight = match (haze.temporal, self.prev_view_proj) {
            (true, Some(_)) => 0.12,
            _ => 1.0,
        };
        let u = VolumeUniform {
            inv_view_proj: frame.view_proj.inverse().to_cols_array_2d(),
            prev_view_proj: self.prev_view_proj.unwrap_or(frame.view_proj).to_cols_array_2d(),
            eye: frame.eye.extend(1.0).into(),
            haze: [
                haze.density.max(0.0),
                haze.anisotropy.clamp(-0.95, 0.95),
                haze.steps.max(1) as f32,
                (self.frame % 1024) as f32,
            ],
            sizes: [t.full.0 as f32, t.full.1 as f32, t.half.0 as f32, t.half.1 as f32],
            misc: [frame.near, frame.far, history_weight, 60.0],
            extra: [t.scale as f32, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));

        let pass = begin_pass;
        let clear = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);
        {
            let mut p = pass(enc, "volume march", &t.raw, clear, timer.and_then(|t| t.writes(timer_slot, true, false)));
            p.set_pipeline(&self.march);
            p.set_bind_group(0, scene_bg, &[]);
            p.set_bind_group(1, &t.march_bg, &[]);
            p.draw(0..3, 0..1);
        }
        {
            let mut p = pass(enc, "volume taa", &t.accum[cur], clear, None);
            p.set_pipeline(&self.taa);
            p.set_bind_group(0, &t.taa_bg[cur], &[]);
            p.draw(0..3, 0..1);
        }
        {
            let mut p = pass(enc, "volume composite", hdr, wgpu::LoadOp::Load, timer.and_then(|t| t.writes(timer_slot, false, true)));
            p.set_pipeline(&self.composite);
            p.set_bind_group(0, &t.composite_bg[cur], &[]);
            p.draw(0..3, 0..1);
        }
        self.prev_view_proj = Some(frame.view_proj);
        self.frame = self.frame.wrapping_add(1);
    }

    /// Forget accumulated history (e.g. when haze is turned back on).
    pub fn reset_history(&mut self) {
        self.prev_view_proj = None;
    }
}

fn begin_pass<'e>(
    enc: &'e mut wgpu::CommandEncoder,
    label: &str,
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    ts: Option<wgpu::RenderPassTimestampWrites<'_>>,
) -> wgpu::RenderPass<'e> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
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
        timestamp_writes: ts,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}
