//! GDTF fixture viewer.
//!
//! Usage:
//!   gdtf-viewer <fixture.gdtf> [options]
//!
//! Options:
//!   --screenshot out.png        render one frame headless and exit
//!   --bench N [--size WxH]      render N frames headless and print timings
//!   --continuous                redraw every frame and log frame stats every 2 s
//!   --haze DENSITY / --no-haze  haze scattering coefficient (1/m) or no volumetrics
//!   --bloom S / --tonemap agx|aces  glare strength (0 = off) / display transform
//!   --dmx ADDR=V,...            DMX control with these slot values (1-based within the fixture;
//!                               other slots: defaults + locate)
//!   --artnet U | --sacn U       listen for DMX on that universe; --address A (default 1)
//!   Physical flags (--pan, --wheel, --cmy, ...) switch to manual control.
//!   --view yaw,pitch,distance   initial camera (radians, radians, metres)
//!   --target x,y,z | lens       point the camera orbits (metres, Y-up), or the lens
//!   --off-axis DEG              with --target lens: look into the beam from DEG off axis
//!   --mode N                    DMX mode index (default 0)
//!   --pan DEG / --tilt DEG      initial position
//!   --dimmer 0..1               master dimmer
//!   --ev EV100 / --house LUX    exposure / house lights
//!   --wheel ATTR=POS            wheel position, e.g. Gobo2=3 or Color1=1.5 (0-based slots)
//!   --index ATTR=DEG            gobo index rotation, e.g. Gobo2=45
//!   --cmy C,M,Y                 CMY flags 0..1
//!   --cto K / --zoom DEG        CTO colour temperature / zoom field angle
//!   --focus-dist M              distance at which gobos are sharp
//!
//! Controls: left-drag orbit, right-drag pan, scroll zoom, R reset view.

mod gpu;
mod orbit;
mod net;
mod scene;
mod stats;
mod ui_paint;

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use fixture::AxisAttr;
use gdtf_loader::LoadedFixture;
use renderer::Renderer;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use orbit::Orbit;
use scene::Scene;

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn,naga=warn"),
    )
    .init();

    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .context("usage: gdtf-viewer <fixture.gdtf> [--screenshot out.png] [--view y,p,d] [--mode N] [--pan DEG] [--tilt DEG]")?;
    let mut screenshot = None;
    let mut bench = None;
    let mut continuous = false;
    let mut target_lens = false;
    let mut off_axis: Option<f32> = None;
    let mut haze: Option<f32> = None;
    let mut bloom: Option<f32> = None;
    let mut tonemap = None;
    let mut dmx_values: Vec<(usize, u8)> = Vec::new();
    let mut listen: Option<(net::Protocol, u16)> = None;
    let mut address: Option<u16> = None;
    let mut manual = false;
    let mut size = (2880, 1720); // 1440×860 at Retina scale
    let mut orbit = Orbit::default();
    let mut mode = 0;
    let mut position: Vec<(AxisAttr, f32)> = Vec::new();
    let (mut dimmer, mut ev, mut house) = (None, None, None);
    let mut optics_args: Vec<(String, String)> = Vec::new();
    while let Some(a) = args.next() {
        let mut value = |name: &str| args.next().with_context(|| format!("{name} needs a value"));
        match a.as_str() {
            "--screenshot" => screenshot = Some(value("--screenshot")?),
            "--bench" => bench = Some(value("--bench")?.parse()?),
            "--continuous" => continuous = true,
            "--haze" => haze = Some(value("--haze")?.parse()?),
            "--no-haze" => haze = Some(0.0),
            "--bloom" => bloom = Some(value("--bloom")?.parse()?),
            "--dmx" => {
                for kv in value("--dmx")?.split(',') {
                    let (a, v) = kv.split_once('=').context("--dmx ADDR=VALUE,...")?;
                    dmx_values.push((a.trim().parse()?, v.trim().parse()?));
                }
            }
            "--artnet" => listen = Some((net::Protocol::ArtNet, value("--artnet")?.parse()?)),
            "--sacn" => listen = Some((net::Protocol::Sacn, value("--sacn")?.parse()?)),
            "--address" => address = Some(value("--address")?.parse()?),
            "--tonemap" => {
                tonemap = Some(match value("--tonemap")?.as_str() {
                    "agx" => renderer::Tonemapper::AgX,
                    "aces" => renderer::Tonemapper::Aces,
                    other => anyhow::bail!("unknown tonemapper {other} (agx, aces)"),
                })
            }
            "--size" => {
                let v = value("--size")?;
                let (w, h) = v.split_once('x').context("--size WxH")?;
                size = (w.parse()?, h.parse()?);
            }
            "--view" => {
                let v: Vec<f32> = value("--view")?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
                anyhow::ensure!(v.len() == 3, "--view needs yaw,pitch,distance");
                (orbit.yaw, orbit.pitch, orbit.distance) = (v[0], v[1], v[2]);
            }
            "--target" => {
                let v = value("--target")?;
                if v == "lens" {
                    target_lens = true;
                    continue;
                }
                let v: Vec<f32> = v
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
                anyhow::ensure!(v.len() == 3, "--target needs x,y,z");
                orbit.target = glam::Vec3::new(v[0], v[1], v[2]);
            }
            "--off-axis" => off_axis = Some(value("--off-axis")?.parse()?),
            "--mode" => mode = value("--mode")?.parse()?,
            "--pan" => {
                manual = true;
                position.push((AxisAttr::Pan, value("--pan")?.parse()?))
            }
            "--tilt" => {
                manual = true;
                position.push((AxisAttr::Tilt, value("--tilt")?.parse()?))
            }
            "--dimmer" => {
                manual = true;
                dimmer = Some(value("--dimmer")?.parse()?)
            }
            "--ev" => ev = Some(value("--ev")?.parse()?),
            "--house" => house = Some(value("--house")?.parse()?),
            flag @ ("--wheel" | "--index" | "--cmy" | "--cto" | "--zoom" | "--focus-dist") => {
                manual = true;
                let v = value(flag)?;
                optics_args.push((flag.to_owned(), v));
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
    }

    let mut fx = LoadedFixture::open(&path).with_context(|| format!("loading {path}"))?;
    let gobos = load_gobos(&mut fx);
    let mut scene = Scene::new(&fx);
    anyhow::ensure!(mode < scene.modes.len().max(1), "mode {mode} out of range ({} modes)", scene.modes.len());
    scene.set_mode(&fx, mode);
    for (attr, deg) in position {
        if !scene.set_target(attr, deg, true) {
            log::warn!("this mode has no {attr:?} channel");
        }
    }

    if let Some(d) = dimmer {
        scene.dimmer = d;
    }
    if let Some(e) = ev {
        scene.view.ev100 = e;
    }
    if let Some(h) = house {
        scene.view.house_lux = h;
    }
    if manual {
        scene.control = scene::Control::Manual;
    }
    for (addr, v) in dmx_values {
        anyhow::ensure!(
            (1..=scene.patch.footprint).contains(&addr),
            "--dmx address {addr} outside the fixture's footprint 1..={}",
            scene.patch.footprint
        );
        scene.dmx.bytes[addr - 1] = v;
    }
    if let Some(a) = address {
        scene.net.address = a;
    }
    if let Some((protocol, universe)) = listen {
        scene.net.protocol = protocol;
        scene.net.universe = universe;
        scene.set_listening(true);
        if let Some(e) = &scene.net.error {
            anyhow::bail!("{e}");
        }
    }
    if let Some(b) = bloom {
        scene.view.bloom = b;
    }
    if let Some(t) = tonemap {
        scene.view.tonemapper = t;
    }
    if let Some(d) = haze {
        scene.view.haze.enabled = d > 0.0;
        if d > 0.0 {
            scene.view.haze.density = d;
        }
    }
    for (flag, v) in optics_args {
        apply_optics_arg(&mut scene, &flag, &v).with_context(|| format!("{flag} {v}"))?;
    }
    scene.update(0.0);
    if target_lens && let Some(b) = scene.beams.first() {
        let t = renderer::default_fixture_placement()
            * fixture::GDTF_TO_Y_UP
            * scene.rig.world_transforms(&scene.pose)[b.node];
        orbit.target = t.w_axis.truncate();
        // With --off-axis: put the eye in the beam, that many degrees off its axis.
        if let Some(deg) = off_axis {
            let dir = t.transform_vector3(fixture::BEAM_DIR_LOCAL).normalize();
            let side = dir.cross(glam::Vec3::Y).normalize_or(glam::Vec3::X);
            let eye_dir = glam::Quat::from_axis_angle(side, deg.to_radians()) * dir;
            orbit.yaw = eye_dir.x.atan2(eye_dir.z);
            orbit.pitch = eye_dir.y.asin();
        }
    }

    if let Some(out) = screenshot {
        return gpu::screenshot(&fx, &gobos, &mut scene, &orbit, (1440, 860), &out);
    }
    if let Some(frames) = bench {
        return gpu::bench(&fx, &gobos, &mut scene, &orbit, size, frames);
    }

    let event_loop = EventLoop::new()?;
    let mut app = App {
        fx,
        gobos,
        scene,
        state: None,
        orbit,
        drag: None,
        cursor: (0.0, 0.0),
        last_frame: Instant::now(),
        stats: stats::FrameStats::new(),
        settle_frames: 30,
    };
    app.stats.continuous = continuous;
    app.stats.log_every = continuous.then(|| std::time::Duration::from_secs(2));
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct WindowState {
    window: Arc<Window>,
    gpu: gpu::SurfaceGpu,
    renderer: Renderer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

struct App {
    fx: LoadedFixture,
    gobos: Vec<(String, Vec<[f32; 3]>)>,
    scene: Scene,
    state: Option<WindowState>,
    orbit: Orbit,
    drag: Option<MouseButton>,
    cursor: (f64, f64),
    last_frame: Instant,
    stats: stats::FrameStats,
    /// Frames to keep redrawing after the last change, so temporal smoothing settles.
    settle_frames: u32,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(&self.scene.title)
            .with_inner_size(winit::dpi::LogicalSize::new(1440, 860));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let gpu = pollster::block_on(gpu::SurfaceGpu::new(window.clone())).expect("init GPU");
        let mut renderer = Renderer::new(&gpu.device, &gpu.queue, gpu.config.format.add_srgb_suffix());
        renderer.set_fixture(&gpu.device, &self.fx.models);
        renderer.set_gobos(&gpu.device, &gpu.queue, &self.gobos);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = ui_paint::new_renderer(&gpu.device, gpu.config.format);

        window.request_redraw();
        self.last_frame = Instant::now();
        self.state = Some(WindowState {
            window,
            gpu,
            renderer,
            egui_ctx,
            egui_state,
            egui_renderer,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(st) = self.state.as_mut() else { return };

        let response = st.egui_state.on_window_event(&st.window, &event);
        if response.repaint {
            st.window.request_redraw();
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                st.gpu.resize(size.width, size.height);
                st.window.request_redraw();
            }
            WindowEvent::RedrawRequested => self.redraw(),
            // Everything below is camera input; egui gets first pick.
            _ if response.consumed => {
                if let WindowEvent::MouseInput { state: ElementState::Released, .. } = event {
                    self.drag = None;
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.drag = (state == ElementState::Pressed).then_some(button);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (dx, dy) = (position.x - self.cursor.0, position.y - self.cursor.1);
                self.cursor = (position.x, position.y);
                match self.drag {
                    Some(MouseButton::Left) => self.orbit.rotate(dx as f32, dy as f32),
                    Some(MouseButton::Right | MouseButton::Middle) => self.orbit.pan(dx as f32, dy as f32),
                    _ => return,
                }
                st.window.request_redraw();
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                self.orbit.zoom(d);
                st.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.logical_key {
                    Key::Character(c) if c.eq_ignore_ascii_case("r") => self.orbit = Orbit::default(),
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    _ => {}
                }
                st.window.request_redraw();
            }
            _ => {}
        }
    }
}

impl App {
    fn redraw(&mut self) {
        let Some(st) = self.state.as_mut() else { return };
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        let mut phases = stats::PhaseTimer::start();
        // Acquire first: egui's output must be painted once produced (its texture
        // deltas panic if dropped unapplied).
        let Some(frame) = st.gpu.acquire() else {
            st.window.request_redraw();
            return;
        };

        // UI before the scene, so slider changes show up in this frame.
        phases.lap(0);
        let raw_input = st.egui_state.take_egui_input(&st.window);
        let (scene, fx, frame_stats) = (&mut self.scene, &self.fx, &mut self.stats);
        let mut output = st.egui_ctx.run_ui(raw_input, |ui| {
            scene.ui(ui, fx, &mut frame_stats.visible, &mut frame_stats.continuous);
            frame_stats.ui(ui.ctx());
        });
        st.egui_state.handle_platform_output(&st.window, std::mem::take(&mut output.platform_output));
        phases.lap(1);
        let moving = self.scene.update(dt);
        phases.lap(2);
        let view = frame.texture.create_view(&Default::default());
        let scene_view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(st.gpu.config.format.add_srgb_suffix()),
            ..Default::default()
        });
        let size = (st.gpu.config.width, st.gpu.config.height);
        let transforms = self.scene.rig.world_transforms(&self.scene.pose);
        st.renderer.ev100 = self.scene.view.ev100;
        st.renderer.house_lux = self.scene.view.house_lux;
        st.renderer.haze = self.scene.view.haze;
        st.renderer.bloom_strength = self.scene.view.bloom;
        st.renderer.tonemapper = self.scene.view.tonemapper;
        st.renderer.render(
            &st.gpu.device,
            &st.gpu.queue,
            &scene_view,
            size,
            &self.orbit.camera(size.0 as f32 / size.1.max(1) as f32),
            &self.scene.rig,
            &transforms,
            &self.scene.beams,
        );
        phases.lap(3);

        // egui on top of the scene.
        ui_paint::paint(
            &st.egui_ctx,
            &mut st.egui_renderer,
            &st.gpu.device,
            &st.gpu.queue,
            &view,
            size,
            &mut output,
        );
        phases.lap(4);

        st.window.pre_present_notify();
        st.gpu.queue.present(frame);
        phases.lap(5);
        self.stats.record(&phases, st.renderer.gpu_pass_times().as_ref().map(|t| &t[..]));

        let repaint_now = output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .is_some_and(|v| v.repaint_delay.is_zero());
        // Volumetrics converge over a few frames (temporal smoothing): keep drawing briefly.
        let settling = self.scene.view.haze.enabled && self.settle_frames > 0;
        self.settle_frames = if moving || repaint_now { 30 } else { self.settle_frames.saturating_sub(1) };
        if moving || repaint_now || self.stats.continuous || settling {
            st.window.request_redraw();
        }
    }
}

/// Loads every wheel slot image of the fixture as a transmission map.
fn load_gobos(fx: &mut LoadedFixture) -> Vec<(String, Vec<[f32; 3]>)> {
    let mut names: Vec<String> = fx
        .wheels()
        .into_iter()
        .flat_map(|w| w.slots)
        .filter_map(|s| s.media_name)
        .collect();
    names.sort();
    names.dedup();
    let mut out = Vec::new();
    for n in names {
        match fx.gobo_transmission(&n, renderer::GOBO_SIZE) {
            Ok(px) => out.push((n, px)),
            Err(e) => log::warn!("{e}"),
        }
    }
    log::info!("loaded {} wheel images", out.len());
    out
}

fn apply_optics_arg(scene: &mut Scene, flag: &str, v: &str) -> Result<()> {
    let kv = |scene: &Scene| -> Result<(usize, f32)> {
        let (k, val) = v.split_once('=').context("expected ATTR=VALUE")?;
        let w = scene.wheel(k).with_context(|| format!("no wheel {k} in this mode"))?;
        Ok((w, val.parse()?))
    };
    match flag {
        "--wheel" => {
            let (w, p) = kv(scene)?;
            scene.optics_state.wheels[w].slot = p;
            return Ok(());
        }
        "--index" => {
            let (w, d) = kv(scene)?;
            scene.optics_state.wheels[w].index_deg = d;
            return Ok(());
        }
        _ => {}
    }
    let st = &mut scene.optics_state;
    match flag {
        "--cmy" => {
            for (l, x) in st.cmy_levels.iter_mut().zip(v.split(',')) {
                *l = x.parse()?;
            }
        }
        "--cto" => st.cto_k = Some(v.parse()?),
        "--zoom" => st.zoom_deg = Some(v.parse()?),
        "--focus-dist" => st.focus = fixture::focus_for_distance(v.parse()?),
        _ => unreachable!(),
    }
    Ok(())
}
