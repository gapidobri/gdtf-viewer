//! Device/surface setup and headless screenshot rendering.

use std::sync::Arc;

use anyhow::{Context, Result};
use gdtf_loader::LoadedFixture;
use renderer::Renderer;
use winit::window::Window;

use crate::orbit::Orbit;
use crate::scene::Scene;

async fn device_for(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
) -> Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .context("no suitable GPU adapter")?;
    log::info!("GPU: {:?}", adapter.get_info());
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            // Per-pass GPU timing for the stats overlay, when available.
            required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            ..Default::default()
        })
        .await?;
    Ok((adapter, device, queue))
}

pub struct SurfaceGpu {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
}

impl SurfaceGpu {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let size = window.inner_size();
        let surface = instance.create_surface(window)?;
        let (adapter, device, queue) = device_for(&instance, Some(&surface)).await?;
        let caps = surface.get_capabilities(&adapter);
        // The surface itself is non-sRGB (egui blends in gamma space); the scene renders
        // through an sRGB view of the same texture so its linear output gets encoded.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0])
            .remove_srgb_suffix();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .context("surface not supported by adapter")?;
        config.format = format;
        config.view_formats = vec![format.add_srgb_suffix()];
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);
        Ok(Self {
            surface,
            device,
            queue,
            config,
        })
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w > 0 && h > 0 {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn acquire(&mut self) -> Option<wgpu::SurfaceTexture> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => Some(t),
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                self.surface.configure(&self.device, &self.config);
                Some(t)
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                None
            }
            _ => None,
        }
    }
}

/// Offscreen rendering target with the same arrangement as the window: the scene renders
/// through an sRGB view, egui onto the raw (non-sRGB) one.
struct Offscreen {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    scene_view: wgpu::TextureView,
    ctx: egui::Context,
    egui_renderer: egui_wgpu::Renderer,
    size: (u32, u32),
}

impl Offscreen {
    fn new(fx: &LoadedFixture, gobos: &[(String, Vec<[f32; 3]>)], size: (u32, u32)) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let (_adapter, device, queue) = pollster::block_on(device_for(&instance, None))?;
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut renderer = Renderer::new(&device, &queue, format.add_srgb_suffix());
        renderer.set_fixture(&device, &fx.models);
        renderer.set_gobos(&device, &queue, gobos);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[format.add_srgb_suffix()],
        });
        let view = tex.create_view(&Default::default());
        let scene_view = tex.create_view(&wgpu::TextureViewDescriptor {
            format: Some(format.add_srgb_suffix()),
            ..Default::default()
        });
        let egui_renderer = crate::ui_paint::new_renderer(&device, format);
        Ok(Self {
            device,
            queue,
            renderer,
            tex,
            view,
            scene_view,
            ctx: egui::Context::default(),
            egui_renderer,
            size,
        })
    }

    fn ui_input(&self) -> egui::RawInput {
        let (w, h) = self.size;
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32))),
            ..Default::default()
        }
    }

    /// Renders the scene, then paints `ui`.
    fn frame(
        &mut self,
        fx: &LoadedFixture,
        scene: &mut Scene,
        orbit: &Orbit,
        phases: &mut crate::stats::PhaseTimer,
        mut ui: egui::FullOutput,
    ) {
        let (w, h) = self.size;
        phases.lap(1);
        scene.update(1.0 / 60.0);
        phases.lap(2);
        let transforms = scene.rig.world_transforms(&scene.pose);
        self.renderer.ev100 = scene.view.ev100;
        self.renderer.house_lux = scene.view.house_lux;
        self.renderer.haze = scene.view.haze;
        self.renderer.bloom_strength = scene.view.bloom;
        self.renderer.tonemapper = scene.view.tonemapper;
        self.renderer.render(
            &self.device,
            &self.queue,
            &self.scene_view,
            (w, h),
            &orbit.camera(w as f32 / h as f32),
            &scene.rig,
            &transforms,
            &scene.beams,
        );
        phases.lap(3);
        let _ = fx;
        crate::ui_paint::paint(&self.ctx, &mut self.egui_renderer, &self.device, &self.queue, &self.view, (w, h), &mut ui);
        phases.lap(4);
    }

    fn wait(&self) -> Result<()> {
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;
        Ok(())
    }

    fn save_png(&self, out: &str) -> Result<()> {
        let (w, h) = self.size;
        let row = (w * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            self.tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(h),
                },
            },
            self.tex.size(),
        );
        self.queue.submit([enc.finish()]);
        buf.map_async(wgpu::MapMode::Read, .., |r| r.expect("map readback buffer"));
        self.wait()?;
        let data = buf.get_mapped_range(..)?;
        let mut img = image::RgbaImage::new(w, h);
        for y in 0..h {
            let src = &data[(y * row) as usize..(y * row + w * 4) as usize];
            img.as_flat_samples_mut().samples[(y * w * 4) as usize..((y + 1) * w * 4) as usize].copy_from_slice(src);
        }
        img.save(out)?;
        log::info!("wrote {out}");
        Ok(())
    }
}

/// Renders a single frame (scene + UI) offscreen and writes it as PNG.
pub fn screenshot(
    fx: &LoadedFixture,
    gobos: &[(String, Vec<[f32; 3]>)],
    scene: &mut Scene,
    orbit: &Orbit,
    size: (u32, u32),
    out: &str,
) -> Result<()> {
    let mut off = Offscreen::new(fx, gobos, size)?;
    // Run the UI a few times so layout (panel sizes, fonts) settles; keep every texture
    // upload (the font atlas arrives in the first pass).
    let mut output = off.ctx.run_ui(off.ui_input(), |ui| scene.ui(ui, fx, &mut false, &mut false));
    for _ in 0..2 {
        let mut next = off.ctx.run_ui(off.ui_input(), |ui| scene.ui(ui, fx, &mut false, &mut false));
        let mut deltas = std::mem::take(&mut output.textures_delta);
        deltas.append(std::mem::take(&mut next.textures_delta));
        next.textures_delta = deltas;
        output = next;
    }
    let mut phases = crate::stats::PhaseTimer::start();
    // A screenshot shows the state as given: no motion step.
    let dt_free = scene.simulate_motion;
    scene.simulate_motion = false;
    // Let temporal smoothing of the volumetrics converge first.
    if scene.view.haze.enabled {
        for _ in 0..24 {
            let empty = off.ctx.run_ui(off.ui_input(), |_| {});
            off.frame(fx, scene, orbit, &mut phases, empty);
        }
    }
    off.frame(fx, scene, orbit, &mut phases, output);
    scene.simulate_motion = dt_free;
    off.save_png(out)
}

/// Renders `frames` frames offscreen (pan sweeping, gobo spinning) and prints where the
/// time goes: CPU per phase, and GPU per pass from timestamp queries.
pub fn bench(
    fx: &LoadedFixture,
    gobos: &[(String, Vec<[f32; 3]>)],
    scene: &mut Scene,
    orbit: &Orbit,
    size: (u32, u32),
    frames: usize,
) -> Result<()> {
    use crate::stats::{PhaseTimer, CPU_PHASES};
    let mut off = Offscreen::new(fx, gobos, size)?;
    let mut cpu: Vec<[f32; CPU_PHASES.len()]> = Vec::new();
    let mut gpu: Vec<[f32; renderer::GPU_PASSES.len()]> = Vec::new();
    let mut wall = Vec::new();
    for i in 0..frames + 10 {
        // Keep things moving so nothing is cached between frames.
        let t = i as f32 / 60.0;
        scene.set_target(fixture::AxisAttr::Pan, 60.0 * (t * 0.5).sin(), true);
        let start = std::time::Instant::now();
        let mut phases = PhaseTimer::start();
        phases.lap(0);
        let output = off.ctx.run_ui(off.ui_input(), |ui| scene.ui(ui, fx, &mut false, &mut false));
        off.frame(fx, scene, orbit, &mut phases, output);
        off.wait()?; // CPU and GPU fully serialised: wall time = CPU + GPU
        phases.lap(5);
        if i >= 10 {
            cpu.push(phases.ms);
            wall.push(start.elapsed().as_secs_f32() * 1e3);
            if let Some(g) = off.renderer.gpu_pass_times() {
                gpu.push(g);
            }
        }
    }
    let stats = |v: &mut Vec<f32>| {
        v.sort_by(f32::total_cmp);
        let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
        (mean, v[v.len() / 2], v[(v.len() * 95 / 100).min(v.len() - 1)])
    };
    println!("{frames} frames at {}x{}", size.0, size.1);
    println!("{:<14}{:>8}{:>8}{:>8}", "ms", "mean", "p50", "p95");
    for (i, name) in CPU_PHASES.iter().enumerate() {
        let label = if *name == "present" { "gpu wait" } else { name };
        let (m, p50, p95) = stats(&mut cpu.iter().map(|c| c[i]).collect());
        println!("cpu {label:<10}{m:>8.2}{p50:>8.2}{p95:>8.2}");
    }
    for (i, name) in renderer::GPU_PASSES.iter().enumerate() {
        if gpu.is_empty() {
            break;
        }
        let (m, p50, p95) = stats(&mut gpu.iter().map(|g| g[i]).collect());
        println!("gpu {name:<10}{m:>8.2}{p50:>8.2}{p95:>8.2}");
    }
    let (m, p50, p95) = stats(&mut wall);
    println!("{:<14}{m:>8.2}{p50:>8.2}{p95:>8.2}", "frame (wall)");
    Ok(())
}
