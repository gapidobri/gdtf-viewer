//! Fixture state driven by the UI: selected DMX mode, axis targets, motor motion and the
//! optical train (wheels, colour mixing, zoom, focus).

use fixture::{
    axis_controls, beams, focus_distance, AxisAttr, AxisControl, AxisMotion, BeamState, FixtureRig, Optics,
    OpticsState, Pose, WheelKind,
};
use fixture::control::{drive_beam, BeamDrive, NodeAttributes, Shutter};
use fixture::dmx::{DmxPatch, DmxValues, Evaluated};
use gdtf_loader::LoadedFixture;

use crate::net::{Protocol, Receiver};

/// Where the fixture's state comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    /// Physical sliders (degrees, levels, slots).
    Manual,
    /// DMX values through the mode's channel functions (faders or network input).
    Dmx,
}

/// Network DMX input settings.
pub struct NetInput {
    pub protocol: Protocol,
    pub universe: u16,
    /// 1-based DMX start address of the fixture.
    pub address: u16,
    pub receiver: Option<Receiver>,
    pub error: Option<String>,
}

pub struct Axis {
    pub control: AxisControl,
    /// Commanded angle (what the console sends).
    pub target: f32,
    /// Where the motor actually is.
    pub motion: AxisMotion,
}

pub struct Scene {
    pub title: String,
    pub modes: Vec<String>,
    pub mode: usize,
    pub rig: FixtureRig,
    pub pose: Pose,
    pub axes: Vec<Axis>,
    /// Follow targets at the speed the file specifies instead of snapping.
    pub simulate_motion: bool,
    pub beams: Vec<BeamState>,
    pub optics: Optics,
    pub optics_state: OpticsState,
    /// Master dimmer 0..1 applied to every beam in manual control.
    pub dimmer: f32,
    pub view: ViewSettings,

    pub control: Control,
    pub patch: DmxPatch,
    pub dmx: DmxValues,
    pub eval: Vec<Evaluated>,
    /// Per-beam optics in DMX control (zones and plates can differ).
    pub beam_optics: Vec<OpticsState>,
    pub drives: Vec<BeamDrive>,
    pub net: NetInput,
    /// Automatic quality (volumetric steps / resolution) to hold 60 fps.
    pub quality: crate::quality::AutoQuality,
    time: f32,
}

/// Camera/environment settings shown in the UI and copied to the renderer each frame.
#[derive(Debug, Clone, Copy)]
pub struct ViewSettings {
    pub ev100: f32,
    pub house_lux: f32,
    pub haze: renderer::Haze,
    pub bloom: f32,
    pub tonemapper: renderer::Tonemapper,
}

impl Default for ViewSettings {
    fn default() -> Self {
        Self {
            ev100: 7.0,
            house_lux: 15.0,
            haze: renderer::Haze::default(),
            bloom: 0.04,
            tonemapper: renderer::Tonemapper::default(),
        }
    }
}

impl Scene {
    pub fn new(fx: &LoadedFixture) -> Self {
        let ft = fx.fixture_type();
        let empty = Optics {
            source: [1.0; 3],
            native_cct: 6000.0,
            emitters: vec![],
            cmy: vec![],
            wheels: vec![],
            cto: None,
            zoom: None,
            has_focus: false,
            has_frost: false,
            spectral: Default::default(),
        };
        let mut s = Self {
            title: format!("{} {} — GDTF viewer", ft.manufacturer, ft.long_name),
            modes: ft
                .dmx_modes
                .iter()
                .map(|m| m.name.as_ref().map(|n| n.to_string()).unwrap_or_default())
                .collect(),
            mode: 0,
            rig: FixtureRig::default(),
            pose: Pose::default(),
            axes: Vec::new(),
            simulate_motion: true,
            beams: Vec::new(),
            optics_state: empty.default_state(25.0),
            optics: empty,
            dimmer: 1.0,
            view: ViewSettings::default(),
            control: Control::Dmx,
            patch: DmxPatch::default(),
            dmx: DmxValues {
                bytes: vec![],
                virtual_values: vec![],
            },
            eval: Vec::new(),
            beam_optics: Vec::new(),
            drives: Vec::new(),
            net: NetInput {
                protocol: Protocol::ArtNet,
                universe: 0,
                address: 1,
                receiver: None,
                error: None,
            },
            quality: Default::default(),
            time: 0.0,
        };
        s.set_mode(fx, 0);
        s
    }

    pub fn set_mode(&mut self, fx: &LoadedFixture, mode: usize) {
        let ft = fx.fixture_type();
        self.mode = mode;
        let dmx_mode = ft.dmx_modes.get(mode);
        let root = dmx_mode.and_then(|m| m.geometry.as_ref()).map(|g| g.to_string());
        self.rig = FixtureRig::from_loaded(fx, root.as_deref());
        self.pose = self.rig.rest_pose();
        self.beams = beams(&self.rig);
        self.axes = dmx_mode
            .map(|m| axis_controls(&self.rig, m))
            .unwrap_or_default()
            .into_iter()
            .map(|control| Axis {
                target: control.default_deg,
                motion: AxisMotion::at(control.default_deg),
                control,
            })
            .collect();
        if let Some(m) = dmx_mode {
            let beam = self.beams.first().map(|b| &b.info);
            self.optics = Optics::from_mode(ft, m, beam);
            self.optics_state = self.optics.default_state(beam.map_or(25.0, |b| b.field_angle));
            self.patch = DmxPatch::from_mode(m);
            self.dmx = self.patch.default_values();
            self.locate();
        }
        self.beam_optics = vec![self.optics_state.clone(); self.beams.len()];
        self.drives = vec![
            BeamDrive {
                dimmer: 1.0,
                shutter: Shutter::Open
            };
            self.beams.len()
        ];
        self.apply_pose();
    }

    /// Default DMX values, then dimmers full and shutters open: the "locate" a console
    /// does to find a fixture.
    pub fn locate(&mut self) {
        self.dmx = self.patch.default_values();
        for i in 0..self.patch.channels.len() {
            let c = &self.patch.channels[i];
            if c.functions.iter().any(|f| f.attribute == "Dimmer") {
                self.patch.set(&mut self.dmx, i, c.max());
            }
            if let Some(f) = c.functions.iter().find(|f| f.attribute == "Shutter1") {
                // Start of the first channel set named open, else keep the default.
                if let Some(set) = f.sets.iter().find(|s| {
                    let n = s.name.to_lowercase();
                    n.contains("open") && !n.contains("close")
                }) {
                    let v = set.dmx_from;
                    self.patch.set(&mut self.dmx, i, v);
                }
            }
        }
    }

    /// Starts or stops listening for network DMX with the current settings.
    pub fn set_listening(&mut self, on: bool) {
        self.net.receiver = None;
        self.net.error = None;
        if on {
            match Receiver::start(self.net.protocol, self.net.universe) {
                Ok(r) => self.net.receiver = Some(r),
                Err(e) => self.net.error = Some(format!("UDP {}: {e}", self.net.protocol.port())),
            }
        }
    }

    /// Sets an attribute's target by name ("Pan", "Tilt"); used by CLI flags.
    pub fn set_target(&mut self, attr: AxisAttr, deg: f32, snap: bool) -> bool {
        let mut found = false;
        for a in self.axes.iter_mut().filter(|a| a.control.attr == attr) {
            a.target = deg.clamp(a.control.min_deg(), a.control.max_deg());
            if snap {
                a.motion = AxisMotion::at(a.target);
            }
            found = true;
        }
        self.apply_pose();
        found
    }

    /// Index of the wheel selected by `attr` (e.g. "Gobo2").
    pub fn wheel(&self, attr: &str) -> Option<usize> {
        self.optics.wheels.iter().position(|w| w.attr.eq_ignore_ascii_case(attr))
    }

    /// Advances motion and re-evaluates the beams. Returns true while anything animates.
    pub fn update(&mut self, dt: f32) -> bool {
        let mut moving = false;
        for a in &mut self.axes {
            if self.simulate_motion {
                let c = &a.control;
                a.motion.step(a.target, dt, c.max_speed(), c.max_accel());
            } else {
                a.motion = AxisMotion::at(a.target);
            }
            moving |= a.motion.position != a.target;
        }
        self.apply_pose();
        self.time += dt;

        if self.control == Control::Dmx {
            return self.update_dmx(dt) || moving;
        }

        self.optics.update(&mut self.optics_state, dt);
        moving |= self
            .optics_state
            .wheels
            .iter()
            .any(|w| w.spin_dps != 0.0 || w.wheel_spin_dps != 0.0);
        let look = self.optics.evaluate(&self.optics_state);
        for b in &mut self.beams {
            b.dimmer = self.dimmer;
            b.set_look(look.clone());
        }
        moving
    }

    /// DMX control: network input → evaluation → axes and per-beam state. Returns true
    /// while something animates (network input, strobes, spinning wheels).
    fn update_dmx(&mut self, dt: f32) -> bool {
        let mut animating = false;
        if let Some(r) = &self.net.receiver {
            animating = true; // values can change at any time
            if let Some(frame) = r.latest() {
                let start = self.net.address.saturating_sub(1) as usize;
                for (k, b) in self.dmx.bytes.iter_mut().enumerate() {
                    *b = frame.get(start + k).copied().unwrap_or(0);
                }
            }
        }
        self.eval = self.patch.evaluate(&self.dmx);
        let attrs = NodeAttributes::from_dmx(&self.patch, &self.eval, &self.rig);

        for a in &mut self.axes {
            let name = match a.control.attr {
                AxisAttr::Pan => "Pan",
                AxisAttr::Tilt => "Tilt",
            };
            if let Some(v) = attrs.on_node(a.control.node, name) {
                a.target = v.physical.clamp(a.control.min_deg(), a.control.max_deg());
            }
        }

        for (i, b) in self.beams.iter_mut().enumerate() {
            let st = &mut self.beam_optics[i];
            let drive = drive_beam(&attrs, &self.rig, b.node, &self.optics, st);
            self.optics.update(st, dt);
            animating |= matches!(drive.shutter, Shutter::Strobe(_))
                || st.wheels.iter().any(|w| w.spin_dps != 0.0 || w.wheel_spin_dps != 0.0);
            b.set_look(self.optics.evaluate(st));
            b.dimmer = drive.output(self.time);
            self.drives[i] = drive;
        }
        animating
    }

    fn apply_pose(&mut self) {
        for a in &self.axes {
            a.control.apply(&mut self.pose, a.motion.position);
        }
    }

    pub fn ui(&mut self, root: &mut egui::Ui, fx: &LoadedFixture, show_stats: &mut bool, continuous: &mut bool) {
        egui::Panel::left("controls").exact_size(330.0).show(root, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading(&fx.fixture_type().long_name);
                ui.label(&fx.fixture_type().manufacturer);
                ui.separator();

                ui.spacing_mut().slider_width = 170.0;
                let mut mode = self.mode;
                ui.label("DMX mode");
                egui::ComboBox::from_id_salt("dmx_mode")
                    .selected_text(self.modes.get(mode).map_or("-", |s| s.as_str()))
                    .width(ui.available_width())
                    .show_ui(ui, |ui| {
                        for (i, m) in self.modes.iter().enumerate() {
                            ui.selectable_value(&mut mode, i, m);
                        }
                    });
                if mode != self.mode {
                    self.set_mode(fx, mode);
                }
                ui.horizontal(|ui| {
                    ui.label("Control");
                    ui.selectable_value(&mut self.control, Control::Dmx, "DMX")
                        .on_hover_text("Channel faders or Art-Net / sACN, through the mode's channel functions");
                    ui.selectable_value(&mut self.control, Control::Manual, "Manual")
                        .on_hover_text("Physical sliders: degrees, levels, wheel slots");
                });

                if self.control == Control::Dmx {
                    egui::CollapsingHeader::new("Network input").default_open(false).show(ui, |ui| self.net_ui(ui));
                    egui::CollapsingHeader::new("Light").default_open(false).show(ui, |ui| self.light_analysis_ui(ui));
                    egui::CollapsingHeader::new("DMX channels").default_open(true).show(ui, |ui| self.dmx_ui(ui));
                    egui::CollapsingHeader::new("Haze").default_open(false).show(ui, |ui| self.haze_ui(ui));
                    egui::CollapsingHeader::new("View").default_open(false).show(ui, |ui| self.view_ui(ui, show_stats, continuous));
                    return;
                }

                egui::CollapsingHeader::new("Position").default_open(true).show(ui, |ui| self.position_ui(ui));
                egui::CollapsingHeader::new("Intensity & beam").default_open(true).show(ui, |ui| self.beam_ui(ui));
                egui::CollapsingHeader::new("Colour").default_open(true).show(ui, |ui| self.colour_ui(ui));
                egui::CollapsingHeader::new("Gobos, prisms, frost").default_open(true).show(ui, |ui| self.gobo_ui(ui));
                egui::CollapsingHeader::new("Haze").default_open(true).show(ui, |ui| self.haze_ui(ui));
                egui::CollapsingHeader::new("View").default_open(true).show(ui, |ui| self.view_ui(ui, show_stats, continuous));
            });
        });
    }

    fn haze_ui(&mut self, ui: &mut egui::Ui) {
            let h = &mut self.view.haze;
            ui.checkbox(&mut h.enabled, "Volumetric beams");
            ui.add_enabled_ui(h.enabled, |ui| {
                ui.label("Density (scattering)");
                ui.add(
                    egui::Slider::new(&mut h.density, 0.001..=0.5)
                        .logarithmic(true)
                        .custom_formatter(|v, _| format!("{v:.3} /m")),
                )
                .on_hover_text("Light stage haze ≈ 0.01–0.03 /m, thick haze 0.1+");
                ui.label("Forward scattering (g)");
                ui.add(egui::Slider::new(&mut h.anisotropy, 0.0..=0.9).fixed_decimals(2))
                    .on_hover_text("How much brighter the beam looks when you look towards the fixture");
                ui.checkbox(&mut h.temporal, "Temporal smoothing");
            });
        
    }

    fn view_ui(&mut self, ui: &mut egui::Ui, show_stats: &mut bool, continuous: &mut bool) {
            ui.label("Exposure (EV100)");
            ui.add(egui::Slider::new(&mut self.view.ev100, 2.0..=16.0).step_by(0.1))
                .on_hover_text("Like a camera: higher = darker. ~7 for a dark stage.");
            ui.label("House lights");
            ui.add(egui::Slider::new(&mut self.view.house_lux, 0.0..=500.0).logarithmic(true).suffix(" lux"));
            ui.label("Glare (bloom)");
            ui.add(egui::Slider::new(&mut self.view.bloom, 0.0..=0.3).fixed_decimals(3))
                .on_hover_text("Light scattered in the camera/eye: makes very bright sources glow");
            ui.horizontal(|ui| {
                ui.label("Tonemap");
                ui.selectable_value(&mut self.view.tonemapper, renderer::Tonemapper::AgX, "AgX")
                    .on_hover_text("Natural roll-off of bright saturated light to white");
                ui.selectable_value(&mut self.view.tonemapper, renderer::Tonemapper::Aces, "ACES")
                    .on_hover_text("Punchier, but bright saturated colours shift hue");
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.quality.enabled, "Auto quality")
                    .on_hover_text("Adjust volumetric quality to hold 60 fps");
                let q = self.quality.current();
                ui.weak(format!("{} · {} steps · 1/{} res", q.name, q.steps, q.volume_scale));
            });
            if !self.quality.enabled {
                let n = crate::quality::LEVELS.len() - 1;
                ui.add(egui::Slider::new(&mut self.quality.level, 0..=n).text("quality level"));
            }
            ui.checkbox(show_stats, "Frame stats");
            ui.checkbox(continuous, "Redraw continuously")
                .on_hover_text("Normally the view only redraws on input or animation");
        
    }

    fn net_ui(&mut self, ui: &mut egui::Ui) {
        let mut listening = self.net.receiver.is_some();
        let before = (self.net.protocol, self.net.universe, listening);
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.net.protocol, Protocol::ArtNet, "Art-Net");
            ui.selectable_value(&mut self.net.protocol, Protocol::Sacn, "sACN");
        });
        ui.horizontal(|ui| {
            ui.label("Universe");
            let range = match self.net.protocol {
                Protocol::ArtNet => 0..=32767,
                Protocol::Sacn => 1..=63999,
            };
            ui.add(egui::DragValue::new(&mut self.net.universe).range(range));
            ui.label("Address");
            ui.add(egui::DragValue::new(&mut self.net.address).range(1..=512));
        });
        ui.checkbox(&mut listening, "Listen");
        if (self.net.protocol, self.net.universe, listening) != before {
            if self.net.protocol == Protocol::Sacn && self.net.universe == 0 {
                self.net.universe = 1;
            }
            self.set_listening(listening);
        }
        if let Some(e) = &self.net.error {
            ui.colored_label(egui::Color32::LIGHT_RED, e);
        }
        if let Some(r) = &self.net.receiver {
            let live = r.latest().is_some();
            ui.weak(format!(
                "{} · {} packets · fixture uses {}–{}",
                if live { "receiving" } else { "no data" },
                r.packets(),
                self.net.address,
                self.net.address as usize + self.patch.footprint.max(1) - 1
            ));
        }
    }

    fn dmx_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Locate").on_hover_text("Defaults, dimmer full, shutter open").clicked() {
                self.locate();
            }
            if ui.button("Defaults").clicked() {
                self.dmx = self.patch.default_values();
            }
        });
        if self.net.receiver.is_some() {
            ui.weak("Network input overrides the faders.");
        }
        ui.spacing_mut().slider_width = 150.0;
        for i in 0..self.patch.channels.len() {
            let c = &self.patch.channels[i];
            let addr = if c.is_virtual() {
                "virtual".to_owned()
            } else {
                c.offsets.iter().map(|o| (o + 1).to_string()).collect::<Vec<_>>().join("/")
            };
            let attr = c.name.strip_prefix(&format!("{}_", c.geometry)).unwrap_or(&c.name);
            ui.add_space(2.0);
            ui.label(format!("{addr}  {attr}  ({})", c.geometry));
            let mut v = self.patch.get(&self.dmx, i);
            let max = c.max();
            if ui.add(egui::Slider::new(&mut v, 0..=max)).changed() {
                self.patch.set(&mut self.dmx, i, v);
            }
            if let Some(e) = self.eval.get(i)
                && let Some(fi) = e.function
            {
                let f = &self.patch.channels[i].functions[fi];
                let set = e.set.map(|s| f.sets[s].name.as_str()).filter(|n| !n.is_empty());
                ui.weak(format!(
                    "{}{}  = {:.2}",
                    f.name,
                    set.map(|s| format!(" · {s}")).unwrap_or_default(),
                    e.physical
                ));
            }
        }
    }

    fn position_ui(&mut self, ui: &mut egui::Ui) {
        if self.axes.is_empty() {
            ui.label("This mode has no Pan/Tilt channels.");
        }
        for a in &mut self.axes {
            let c = &a.control;
            let name = match c.attr {
                AxisAttr::Pan => "Pan",
                AxisAttr::Tilt => "Tilt",
            };
            ui.label(format!("{name} ({})", c.geometry));
            ui.add(
                egui::Slider::new(&mut a.target, c.min_deg()..=c.max_deg())
                    .suffix("°")
                    .fixed_decimals(1)
                    .drag_value_speed(0.5),
            );
            let v = c.dmx_value(a.target);
            let bytes: Vec<String> = (0..c.bytes).rev().map(|i| ((v >> (8 * i)) & 0xff).to_string()).collect();
            ui.weak(format!(
                "DMX {} ({}-bit)   actual {:.1}°",
                bytes.join("/"),
                8 * c.bytes as u32,
                a.motion.position
            ));
            ui.add_space(4.0);
        }
        ui.horizontal(|ui| {
            if ui.button("Home").clicked() {
                for a in &mut self.axes {
                    a.target = a.control.default_deg;
                }
            }
            ui.checkbox(&mut self.simulate_motion, "Motor speed")
                .on_hover_text("Move at the speed given by the file's RealFade / RealAcceleration");
        });
    }

    fn beam_ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Dimmer");
        ui.add(egui::Slider::new(&mut self.dimmer, 0.0..=1.0).custom_formatter(|v, _| format!("{:.0}%", v * 100.0)));
        let st = &mut self.optics_state;
        if let (Some(r), Some(z)) = (self.optics.zoom, st.zoom_deg.as_mut()) {
            ui.label("Zoom (field angle)");
            ui.add(egui::Slider::new(z, r.min()..=r.max()).suffix("°").fixed_decimals(1));
        }
        if self.optics.has_focus {
            ui.label("Focus");
            ui.add(
                egui::Slider::new(&mut st.focus, 0.0..=1.0)
                    .custom_formatter(|v, _| format!("{:.1} m", focus_distance(v as f32))),
            );
        }
        // Identical beams (pixel rings, LED arrays) are listed once with a count.
        let mut groups: Vec<(&fixture::BeamState, usize)> = Vec::new();
        for b in &self.beams {
            let same = |g: &fixture::BeamState| {
                g.info.beam_type == b.info.beam_type
                    && g.info.luminous_flux == b.info.luminous_flux
                    && g.info.beam_angle == b.info.beam_angle
                    && g.info.field_angle == b.info.field_angle
            };
            match groups.iter_mut().find(|(g, _)| same(g)) {
                Some((_, n)) => *n += 1,
                None => groups.push((b, 1)),
            }
        }
        for (b, n) in groups {
            let count = if n > 1 { format!("{n} × ") } else { String::new() };
            ui.weak(format!(
                "{count}{:?}  {:.0} lm  {:.1}°/{:.1}°  {:.0} K",
                b.info.beam_type, b.info.luminous_flux, b.info.beam_angle, b.info.field_angle, b.info.color_temperature
            ));
            let total = if n > 1 { format!(" (all: {:.0} lux)", n as f32 * b.intensity_cd() / 25.0) } else { String::new() };
            ui.weak(format!(
                "now {:.1}° field, {:.0} cd peak, {:.0} lux at 5 m{total}",
                b.field_angle,
                b.intensity_cd(),
                b.intensity_cd() / 25.0
            ));
        }
    }

    fn colour_ui(&mut self, ui: &mut egui::Ui) {
        let (o, st) = (&self.optics, &mut self.optics_state);
        if !o.emitters.is_empty() {
            for (e, level) in o.emitters.iter().zip(&mut st.emitter_levels) {
                ui.add(egui::Slider::new(level, 0.0..=1.0).text(&e.name).fixed_decimals(2));
            }
            if ui.button("Calibrated white").clicked() {
                st.emitter_levels = o.white_levels();
            }
        }
        for (f, level) in o.cmy.iter().zip(&mut st.cmy_levels) {
            ui.add(egui::Slider::new(level, 0.0..=1.0).text(f.attr.trim_start_matches("ColorSub_")))
                .on_hover_text(&f.name);
        }
        if let (Some(r), Some(k)) = (o.cto, st.cto_k.as_mut()) {
            ui.add(egui::Slider::new(k, r.min()..=r.max()).suffix(" K").text("CTO"));
        }
        for (w, ws) in o.wheels.iter().zip(&mut st.wheels) {
            if w.kind == WheelKind::Color {
                wheel_ui(ui, w, ws);
            }
        }
        self.light_analysis_ui(ui);
    }

    /// Beam colour swatch, spectrum plot and CIE 1931 chromaticity diagram, for checking
    /// against a data sheet.
    fn light_analysis_ui(&self, ui: &mut egui::Ui) {
        use fixture::color;
        let Some(b) = self.beams.first() else { return };
        // Include the colour wheel (applied in the shader as a split between two slots).
        let c = match &b.look.color_split {
            Some(sp) => std::array::from_fn(|k| b.look.color[k] * (sp.a[k] + (sp.b[k] - sp.a[k]) * sp.frac)),
            None => b.look.color,
        };
        let to8 = |v: f32| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
        let swatch = |c: color::Rgb| {
            let s = color::working_to_srgb_linear(c);
            let m = s.iter().copied().fold(1e-6, f32::max);
            egui::Color32::from_rgb(to8(s[0] / m), to8(s[1] / m), to8(s[2] / m))
        };
        let xyz = match (&b.look.spectrum, &b.look.color_split) {
            (Some(s), None) => s.to_xyz(),
            _ => color::working_to_xyz(c),
        };
        let xy = color::xyz_to_xy(xyz);
        ui.horizontal(|ui| {
            ui.weak("Beam colour");
            let (rect, _) = ui.allocate_exact_size(egui::vec2(40.0, 14.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 2.0, swatch(c));
            ui.weak(format!("{:.0}% lum  xy {:.3}, {:.3}", 100.0 * color::luminance(c), xy[0], xy[1]));
        });
        let source = &self.optics.spectral.source_label;
        ui.weak(match (&b.look.spectrum, source.is_empty()) {
            (Some(_), false) => format!("Spectral: source {source}"),
            (Some(_), true) => "Spectral: measured emitters".to_owned(),
            (None, _) => "RGB approximation (no spectral data)".to_owned(),
        });

        let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width().min(280.0), 70.0), egui::Sense::hover());
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 2.0, egui::Color32::from_gray(18));
        if let Some(sp) = &b.look.spectrum {
            let max = sp.0.iter().copied().fold(1e-9, f32::max);
            let n = sp.0.len();
            for i in 0..n - 1 {
                let x = |k: usize| rect.left() + rect.width() * k as f32 / (n - 1) as f32;
                let y = |v: f32| rect.bottom() - rect.height() * (v / max) * 0.95;
                let mono = color::xyz_to_working(color::cie_xyz(color::wavelength(i)));
                p.line_segment(
                    [egui::pos2(x(i), y(sp.0[i])), egui::pos2(x(i + 1), y(sp.0[i + 1]))],
                    egui::Stroke::new(1.5, swatch(mono.map(|v| v.max(0.0)))),
                );
            }
            p.text(rect.left_top() + egui::vec2(3.0, 2.0), egui::Align2::LEFT_TOP, "380–780 nm", egui::FontId::proportional(9.0), egui::Color32::GRAY);
        }

        // Chromaticity diagram: spectral locus, Rec.709 and Rec.2020 gamuts, the beam.
        let side = 150.0;
        let (rect, _) = ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::hover());
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 2.0, egui::Color32::from_gray(18));
        let to = |x: f32, y: f32| egui::pos2(rect.left() + x / 0.8 * side, rect.bottom() - y / 0.9 * side);
        let locus: Vec<egui::Pos2> = (380..=700)
            .step_by(5)
            .map(|l| {
                let a = color::xyz_to_xy(color::cie_xyz(l as f32));
                to(a[0], a[1])
            })
            .collect();
        let mut closed = locus.clone();
        closed.push(locus[0]);
        p.add(egui::Shape::line(closed, egui::Stroke::new(1.0, egui::Color32::from_gray(140))));
        for (prims, col) in [
            ([(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)], egui::Color32::from_gray(90)),
            ([(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)], egui::Color32::from_rgb(70, 90, 140)),
        ] {
            let mut pts: Vec<egui::Pos2> = prims.iter().map(|&(x, y)| to(x, y)).collect();
            pts.push(pts[0]);
            p.add(egui::Shape::line(pts, egui::Stroke::new(1.0, col)));
        }
        p.circle_filled(to(xy[0], xy[1]), 3.5, swatch(c));
        p.circle_stroke(to(xy[0], xy[1]), 3.5, egui::Stroke::new(1.0, egui::Color32::WHITE));
        p.text(rect.right_bottom() - egui::vec2(3.0, 3.0), egui::Align2::RIGHT_BOTTOM, "709 · 2020", egui::FontId::proportional(9.0), egui::Color32::GRAY);
    }

    fn gobo_ui(&mut self, ui: &mut egui::Ui) {
        let (o, st) = (&self.optics, &mut self.optics_state);
        let mut any = false;
        for (w, ws) in o.wheels.iter().zip(&mut st.wheels) {
            if matches!(w.kind, WheelKind::Gobo | WheelKind::Prism) {
                wheel_ui(ui, w, ws);
                any = true;
            }
        }
        if o.has_frost {
            ui.add_space(4.0);
            ui.add(egui::Slider::new(&mut st.frost, 0.0..=1.0).text("frost"));
            any = true;
        }
        if !any {
            ui.label("No gobo or prism wheels in this mode.");
        }
    }
}

fn wheel_ui(ui: &mut egui::Ui, w: &fixture::WheelUse, ws: &mut fixture::WheelState) {
    let n = w.slots.len();
    if n == 0 {
        return;
    }
    ui.add_space(4.0);
    ui.strong(format!("{} ({})", w.wheel_name, w.attr));
    let current = (ws.slot.round() as usize) % n;
    egui::ComboBox::from_id_salt(&w.attr)
        .selected_text(&w.slots[current].name)
        .width(ui.available_width())
        .show_ui(ui, |ui| {
            for (i, s) in w.slots.iter().enumerate() {
                if ui.selectable_label(i == current, &s.name).clicked() {
                    ws.slot = i as f32;
                }
            }
        });
    ui.add(egui::Slider::new(&mut ws.slot, 0.0..=(n as f32 - 0.001)).text("position").fixed_decimals(2))
        .on_hover_text("Continuous wheel position: in between slots both are in the beam");
    if let Some(r) = w.index_range {
        ui.add(egui::Slider::new(&mut ws.index_deg, r.min()..=r.max()).text("index").suffix("°"));
    }
    if let Some(r) = w.spin_range {
        ui.add(egui::Slider::new(&mut ws.spin_dps, r.min()..=r.max()).text("rotate").suffix("°/s"));
    }
    if let Some(r) = w.wheel_spin_range {
        ui.add(egui::Slider::new(&mut ws.wheel_spin_dps, r.min()..=r.max()).text("wheel spin").suffix("°/s"));
    }
}
