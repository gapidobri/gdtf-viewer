//! Bloom: half-resolution mip chain of the HDR image, downsampled then upsampled
//! additively. The result (mip 0) is mixed into the image in the tonemap pass.

use crate::{timing::GpuTimer, HDR_FORMAT};

const MAX_MIPS: u32 = 7;

struct Targets {
    /// One view per mip, for rendering into and sampling from.
    mips: Vec<wgpu::TextureView>,
    /// Bind group sampling the HDR image (first downsample).
    from_hdr: wgpu::BindGroup,
    /// Bind group sampling mip i.
    from_mip: Vec<wgpu::BindGroup>,
}

pub struct BloomPass {
    down_first: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    targets: Option<Targets>,
}

impl BloomPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("bloom.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        Self {
            down_first: pipeline("fs_down_first", None),
            down: pipeline("fs_down", None),
            up: pipeline("fs_up", Some(additive)),
            layout,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("bloom"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bloom"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            targets: None,
        }
    }

    fn bind(&self, device: &wgpu::Device, view: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform.as_entire_binding(),
                },
            ],
        })
    }

    /// (Re)creates the mip chain for a full-resolution HDR image of `size`.
    /// Returns the view to sample the bloom result from (mip 0).
    pub fn resize(&mut self, device: &wgpu::Device, size: (u32, u32), hdr: &wgpu::TextureView) -> wgpu::TextureView {
        let base = (size.0.div_ceil(2).max(1), size.1.div_ceil(2).max(1));
        // Down to roughly 8 px on the short side.
        let mips = (base.0.min(base.1).max(8) / 8).ilog2().clamp(1, MAX_MIPS - 1) + 1;
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bloom"),
            size: wgpu::Extent3d {
                width: base.0,
                height: base.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mip_views: Vec<wgpu::TextureView> = (0..mips)
            .map(|m| {
                tex.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: m,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let from_mip = mip_views.iter().map(|v| self.bind(device, v)).collect();
        let result = mip_views[0].clone();
        self.targets = Some(Targets {
            from_hdr: self.bind(device, hdr),
            from_mip,
            mips: mip_views,
        });
        result
    }

    pub fn mip_count(&self) -> u32 {
        self.targets.as_ref().map_or(1, |t| t.mips.len() as u32)
    }

    pub fn encode(
        &self,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        exposure: f32,
        timer: Option<&GpuTimer>,
        timer_slot: usize,
    ) {
        let Some(t) = &self.targets else { return };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&[exposure, 0.0, 0.0, 0.0]));
        let n = t.mips.len();
        let last_pass = 2 * n - 2; // n downsamples + (n − 1) upsamples, 0-based
        let mut pass_index = 0;
        let mut run = |pipeline: &wgpu::RenderPipeline,
                       bind: &wgpu::BindGroup,
                       target: &wgpu::TextureView,
                       load: wgpu::LoadOp<wgpu::Color>| {
            let ts = timer.and_then(|tm| tm.writes(timer_slot, pass_index == 0, pass_index == last_pass));
            let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
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
            });
            p.set_pipeline(pipeline);
            p.set_bind_group(0, bind, &[]);
            p.draw(0..3, 0..1);
            pass_index += 1;
        };
        let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
        run(&self.down_first, &t.from_hdr, &t.mips[0], clear);
        for m in 1..n {
            run(&self.down, &t.from_mip[m - 1], &t.mips[m], clear);
        }
        for m in (0..n - 1).rev() {
            run(&self.up, &t.from_mip[m + 1], &t.mips[m], wgpu::LoadOp::Load);
        }
    }
}
