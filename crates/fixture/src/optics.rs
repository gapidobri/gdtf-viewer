//! The optical train of a fixture, as described by a DMX mode: colour wheels, CMY flags,
//! additive emitters, CTO, gobo wheels, zoom and focus. [`Optics::evaluate`] turns the
//! current [`OpticsState`] into a [`BeamLook`] for the renderer.
//!
//! Colour is linear sRGB (Rec.709) here; spectral rendering is a later milestone.

use gdtf_loader::gdtf::dmx_mode::{ChannelFunction, DmxMode};
use gdtf_loader::gdtf::fixture_type::FixtureType;
use gdtf_loader::gdtf::physical_descriptions::EmitterOptic;
use gdtf_loader::gdtf::wheel::WheelSlotOptic;
use gdtf_loader::BeamInfo;

use crate::beam::{cct_to_linear_srgb, xyy_to_linear_srgb};

type Rgb = [f32; 3];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WheelKind {
    Color,
    Gobo,
    Prism,
    Animation,
    Other,
}

impl WheelKind {
    /// Classifies a wheel-selecting attribute such as `Gobo2`, `Color1` or `ColorMacro1`.
    /// Returns `None` for derived attributes (`Gobo1WheelSpin`, `Gobo2SelectShake`, ...).
    pub fn from_attribute(attr: &str) -> Option<Self> {
        let base = attr.trim_end_matches(|c: char| c.is_ascii_digit());
        if base.len() == attr.len() {
            return None; // must end in a number
        }
        Some(match base {
            "Gobo" => Self::Gobo,
            "Color" | "ColorMacro" => Self::Color,
            "Prism" => Self::Prism,
            "AnimationWheel" => Self::Animation,
            _ => Self::Other,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SlotOptics {
    pub name: String,
    /// Colour transmission relative to the fixture's white, 0..1 per channel.
    pub transmission: Rgb,
    /// Gobo / animation image (`wheels/<media>.png`).
    pub media: Option<String>,
    pub prism_facets: usize,
}

/// Physical range of a channel function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Range {
    pub from: f32,
    pub to: f32,
}

impl Range {
    pub fn min(&self) -> f32 {
        self.from.min(self.to)
    }
    pub fn max(&self) -> f32 {
        self.from.max(self.to)
    }
    fn of(f: &ChannelFunction) -> Self {
        Self {
            from: f.physical_from as f32,
            to: f.physical_to as f32,
        }
    }
}

/// A wheel as used by the DMX mode.
#[derive(Debug, Clone)]
pub struct WheelUse {
    /// Selecting attribute, e.g. `Gobo2`.
    pub attr: String,
    pub wheel_name: String,
    pub kind: WheelKind,
    /// Virtual colour wheel (colour macros on CMY/LED fixtures): no physical glass, so
    /// no split colours between slots.
    pub is_virtual: bool,
    pub slots: Vec<SlotOptics>,
    /// `<attr>Pos`: indexed gobo rotation, degrees.
    pub index_range: Option<Range>,
    /// `<attr>PosRotate`: continuous gobo rotation, degrees/s.
    pub spin_range: Option<Range>,
    /// `<attr>WheelSpin`: whole-wheel rotation, degrees/s.
    pub wheel_spin_range: Option<Range>,
}

#[derive(Debug, Clone)]
pub struct EmitterUse {
    pub attr: String,
    pub name: String,
    /// Colour at full level, weighted by its share of the total output (luminance).
    pub rgb: Rgb,
}

#[derive(Debug, Clone)]
pub struct FilterUse {
    pub attr: String,
    pub name: String,
    /// Transmission when fully inserted.
    pub transmission: Rgb,
}

/// Everything colour/gobo related that the selected DMX mode can control.
#[derive(Debug, Clone)]
pub struct Optics {
    /// The light source's own colour (luminance 1).
    pub source: Rgb,
    pub native_cct: f32,
    pub emitters: Vec<EmitterUse>,
    pub cmy: Vec<FilterUse>,
    pub wheels: Vec<WheelUse>,
    /// CTO target colour temperature range, kelvin.
    pub cto: Option<Range>,
    /// Zoom: field angle range, degrees.
    pub zoom: Option<Range>,
    pub has_focus: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WheelState {
    /// Continuous slot position, 0-based; 1.5 is halfway between the 2nd and 3rd slot.
    pub slot: f32,
    pub index_deg: f32,
    pub spin_dps: f32,
    pub wheel_spin_dps: f32,
    /// Accumulated continuous gobo rotation.
    pub spin_angle: f32,
}

impl WheelState {
    pub fn update(&mut self, dt: f32, n_slots: usize) {
        self.spin_angle = (self.spin_angle + self.spin_dps * dt).rem_euclid(360.0);
        if self.wheel_spin_dps != 0.0 && n_slots > 0 {
            let n = n_slots as f32;
            self.slot = (self.slot + self.wheel_spin_dps / 360.0 * n * dt).rem_euclid(n);
        }
    }

    pub fn rotation_deg(&self) -> f32 {
        self.index_deg + self.spin_angle
    }

    /// The two slots in the aperture and how far the wheel has moved from the first.
    pub fn slots(&self, n_slots: usize) -> (usize, usize, f32) {
        if n_slots == 0 {
            return (0, 0, 0.0);
        }
        let n = n_slots as f32;
        let p = self.slot.rem_euclid(n);
        let a = p.floor() as usize % n_slots;
        (a, (a + 1) % n_slots, p.fract())
    }
}

#[derive(Debug, Clone)]
pub struct OpticsState {
    pub emitter_levels: Vec<f32>,
    pub cmy_levels: Vec<f32>,
    pub wheels: Vec<WheelState>,
    pub cto_k: Option<f32>,
    pub zoom_deg: Option<f32>,
    /// 0 = near (1.5 m) .. 1 = far (40 m), linear in dioptres like a real focus lens.
    pub focus: f32,
}

/// Colour wheel half-and-half position in the beam.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorSplit {
    /// Transmission of the slot leaving the aperture.
    pub a: Rgb,
    /// Transmission of the slot entering.
    pub b: Rgb,
    pub frac: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GoboLook {
    pub media_a: Option<String>,
    pub media_b: Option<String>,
    pub frac: f32,
    pub rotation_deg: f32,
}

/// What the renderer needs to draw one beam.
#[derive(Debug, Clone, PartialEq)]
pub struct BeamLook {
    /// Uniform colour of the light (luminance ≤ 1 relative to full white).
    pub color: Rgb,
    pub color_split: Option<ColorSplit>,
    /// Up to two gobo wheels, nearest the lamp first.
    pub gobos: Vec<GoboLook>,
    /// Field angle (degrees) if zoom overrides the file's default.
    pub field_angle: Option<f32>,
    /// Distance at which gobos are sharp, metres (infinity for fixtures without focus).
    pub focus_distance: f32,
}

const FOCUS_NEAR: f32 = 1.5;
const FOCUS_FAR: f32 = 40.0;

pub fn focus_distance(focus: f32) -> f32 {
    let d = (1.0 - focus) / FOCUS_NEAR + focus / FOCUS_FAR;
    1.0 / d.max(1e-6)
}

pub fn focus_for_distance(d: f32) -> f32 {
    ((1.0 / FOCUS_NEAR - 1.0 / d) / (1.0 / FOCUS_NEAR - 1.0 / FOCUS_FAR)).clamp(0.0, 1.0)
}

fn mul(a: Rgb, b: Rgb) -> Rgb {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

fn lerp(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn luminance(c: Rgb) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Transmission of a colour (CIE xyY, Y in percent) relative to a white, clamped to 0..1.
fn transmission(x: f32, y: f32, big_y_percent: f32, white: Rgb) -> Rgb {
    let c = xyy_to_linear_srgb(x, y, big_y_percent / 100.0);
    [0, 1, 2].map(|i| (c[i] / white[i].max(1e-6)).clamp(0.0, 1.0))
}

impl Optics {
    pub fn from_mode(ft: &FixtureType, mode: &DmxMode, beam: Option<&BeamInfo>) -> Self {
        let native_cct = beam.map_or(6000.0, |b| b.color_temperature).max(1000.0);
        let source = cct_to_linear_srgb(native_cct);

        let functions: Vec<&ChannelFunction> = mode
            .dmx_channels
            .iter()
            .flat_map(|c| &c.logical_channels)
            .flat_map(|l| &l.channel_functions)
            .collect();
        let attr = |f: &ChannelFunction| f.attribute.to_string();
        let find = |name: &str| functions.iter().copied().find(|f| attr(f) == name);

        // Filters and wheel colours are measured as "light through the filter"; the
        // reference is the open slot of a colour wheel when there is one.
        let white_xy = ft
            .wheels
            .iter()
            .flat_map(|w| &w.slots)
            .find_map(|s| match (&s.optic, s.name.as_deref()) {
                (WheelSlotOptic::Color(c), Some(n)) if n.starts_with("Open") && c.z > 99.0 => {
                    Some((c.x as f32, c.y as f32))
                }
                _ => None,
            });
        let reference = white_xy.map_or(source, |(x, y)| xyy_to_linear_srgb(x, y, 1.0));

        // Additive emitters (ColorAdd_*), weighted by luminous intensity.
        let mut emitters = Vec::new();
        let mut weights = Vec::new();
        for f in &functions {
            let a = attr(f);
            if !a.starts_with("ColorAdd_") || emitters.iter().any(|e: &EmitterUse| e.attr == a) {
                continue;
            }
            let Some(e) = f.emitter(ft) else { continue };
            let EmitterOptic::Color { color, .. } = &e.optic else { continue };
            let rgb = xyy_to_linear_srgb(color.x as f32, color.y as f32, 1.0);
            let lum = luminance(rgb).max(1e-6);
            let w = e
                .measurements
                .iter()
                .find_map(|m| m.luminous_intensity)
                .map_or(color.z as f32, |v| v as f32);
            weights.push(w.max(0.0));
            emitters.push(EmitterUse {
                attr: a,
                name: e.name.as_deref().unwrap_or_default().to_owned(),
                rgb: rgb.map(|c| c / lum),
            });
        }
        let total: f32 = weights.iter().sum();
        for (e, w) in emitters.iter_mut().zip(&weights) {
            let share = if total > 0.0 { w / total } else { 1.0 / weights.len() as f32 };
            e.rgb = e.rgb.map(|c| c * share);
        }

        // Subtractive flags (ColorSub_*).
        let mut cmy = Vec::new();
        for f in &functions {
            let a = attr(f);
            if !a.starts_with("ColorSub_") || cmy.iter().any(|c: &FilterUse| c.attr == a) {
                continue;
            }
            let Some(filt) = f.filter(ft) else { continue };
            let c = &filt.color;
            cmy.push(FilterUse {
                attr: a,
                name: filt.name.as_deref().unwrap_or_default().to_owned(),
                transmission: transmission(c.x as f32, c.y as f32, c.z as f32, reference),
            });
        }

        // Wheels selected by Gobo1, Color1, Prism1, AnimationWheel1, ...
        let mut wheels = Vec::new();
        for f in &functions {
            let a = attr(f);
            let Some(kind) = WheelKind::from_attribute(&a) else { continue };
            if wheels.iter().any(|w: &WheelUse| w.attr == a) {
                continue;
            }
            let Some(w) = f.wheel(ft) else { continue };
            let slots = w
                .slots
                .iter()
                .map(|s| SlotOptics {
                    name: s.name.as_deref().unwrap_or_default().to_owned(),
                    transmission: match &s.optic {
                        WheelSlotOptic::Color(c) => transmission(c.x as f32, c.y as f32, c.z as f32, reference),
                        WheelSlotOptic::Filter(_) => s
                            .filter(ft)
                            .map_or([1.0; 3], |f| {
                                transmission(f.color.x as f32, f.color.y as f32, f.color.z as f32, reference)
                            }),
                    },
                    media: s.media_name.clone().filter(|m| !m.is_empty()),
                    prism_facets: s.facets.len(),
                })
                .collect();
            wheels.push(WheelUse {
                index_range: find(&format!("{a}Pos")).map(Range::of),
                spin_range: find(&format!("{a}PosRotate")).map(Range::of),
                wheel_spin_range: find(&format!("{a}WheelSpin")).map(Range::of),
                is_virtual: {
                    let name = w.name.as_deref().unwrap_or_default().to_lowercase();
                    a.starts_with("ColorMacro") || name.contains("virtual") || name.contains("macro")
                },
                attr: a,
                wheel_name: w.name.as_deref().unwrap_or_default().to_owned(),
                kind,
                slots,
            });
        }

        // CTO: physical values in kelvin (some files use 0..1; treat that as native..3200 K).
        // CTO (orange correction) or CTC (colour temperature control, e.g. on LED panels).
        let cto = find("CTO").or_else(|| find("CTC")).map(|f| {
            let r = Range::of(f);
            if r.max() > 1000.0 {
                r
            } else {
                Range {
                    from: native_cct,
                    to: 3200.0,
                }
            }
        });

        // Zoom: several functions may exist (e.g. beam/spot zoom modes); take the one that
        // covers the file's field angle, else the widest.
        let field = beam.map_or(25.0, |b| b.field_angle);
        let zooms: Vec<Range> = functions
            .iter()
            .filter(|f| attr(f) == "Zoom")
            .map(|f| Range::of(f))
            .collect();
        let zoom = zooms
            .iter()
            .copied()
            .find(|r| (r.min()..=r.max() + 0.01).contains(&field))
            .or_else(|| zooms.iter().copied().max_by(|a, b| a.max().total_cmp(&b.max())));

        Self {
            source,
            native_cct,
            emitters,
            cmy,
            wheels,
            cto,
            zoom,
            has_focus: find("Focus1").is_some(),
        }
    }

    pub fn default_state(&self, field_angle: f32) -> OpticsState {
        OpticsState {
            // A viewer should start with light: white on additive fixtures.
            emitter_levels: self.default_levels(),
            cmy_levels: vec![0.0; self.cmy.len()],
            wheels: vec![WheelState::default(); self.wheels.len()],
            cto_k: self.cto.map(|_| self.native_cct),
            zoom_deg: self.zoom.map(|r| field_angle.clamp(r.min(), r.max())),
            // Sharp at the default trim height.
            focus: focus_for_distance(5.0),
        }
    }

    /// Starting emitter levels: a dedicated white emitter (W, WW, CW) alone if there is
    /// one, which is how RGBW fixtures are usually run for white; otherwise
    /// [`Self::white_levels`].
    pub fn default_levels(&self) -> Vec<f32> {
        let is_white = |e: &EmitterUse| {
            let suffix = e.attr.trim_start_matches("ColorAdd_");
            matches!(suffix, "W" | "WW" | "CW") || e.name.to_lowercase().contains("white")
        };
        if self.emitters.iter().any(is_white) {
            self.emitters.iter().map(|e| if is_white(e) { 1.0 } else { 0.0 }).collect()
        } else {
            self.white_levels()
        }
    }

    /// Emitter levels that mix to the fixture's native white, as bright as possible.
    /// Solves with the three emitters closest to R, G and B; others stay off. Falls back
    /// to everything at full when that isn't possible.
    pub fn white_levels(&self) -> Vec<f32> {
        let full = vec![1.0; self.emitters.len()];
        if self.emitters.len() < 3 {
            return full;
        }
        // Pick the most red, green and blue emitters.
        let pick = |ch: usize| {
            (0..self.emitters.len()).max_by(|&a, &b| {
                let f = |i: usize| {
                    let c = self.emitters[i].rgb;
                    c[ch] / (c[0] + c[1] + c[2]).max(1e-6)
                };
                f(a).total_cmp(&f(b))
            })
        };
        let (Some(r), Some(g), Some(b)) = (pick(0), pick(1), pick(2)) else { return full };
        if r == g || g == b || r == b {
            return full;
        }
        let m = glam::Mat3::from_cols(
            self.emitters[r].rgb.into(),
            self.emitters[g].rgb.into(),
            self.emitters[b].rgb.into(),
        );
        if m.determinant().abs() < 1e-9 {
            return full;
        }
        let x = m.inverse() * glam::Vec3::from(self.source);
        if x.min_element() <= 0.0 {
            return full;
        }
        let x = x / x.max_element();
        let mut levels = vec![0.0; self.emitters.len()];
        levels[r] = x.x;
        levels[g] = x.y;
        levels[b] = x.z;
        levels
    }

    pub fn update(&self, state: &mut OpticsState, dt: f32) {
        for (w, s) in self.wheels.iter().zip(&mut state.wheels) {
            s.update(dt, w.slots.len());
        }
    }

    pub fn evaluate(&self, state: &OpticsState) -> BeamLook {
        let mut color = if self.emitters.is_empty() {
            self.source
        } else {
            self.emitters
                .iter()
                .zip(&state.emitter_levels)
                .fold([0.0; 3], |acc, (e, &l)| {
                    let l = l.clamp(0.0, 1.0);
                    [0, 1, 2].map(|i| acc[i] + e.rgb[i] * l)
                })
        };

        for (f, &level) in self.cmy.iter().zip(&state.cmy_levels) {
            // Partial insertion: the flag covers a fraction of the beam.
            color = mul(color, lerp([1.0; 3], f.transmission, level.clamp(0.0, 1.0)));
        }

        if let (Some(_), Some(k)) = (self.cto, state.cto_k)
            && (k - self.native_cct).abs() > 1.0
        {
            let to = cct_to_linear_srgb(k);
            let from = cct_to_linear_srgb(self.native_cct);
            let ratio = [0, 1, 2].map(|i| to[i] / from[i].max(1e-6));
            let m = ratio.iter().copied().fold(0.0, f32::max).max(1e-6);
            color = mul(color, ratio.map(|r| r / m));
        }

        let mut color_split = None;
        let mut gobos = Vec::new();
        for (w, s) in self.wheels.iter().zip(&state.wheels) {
            let (a, b, frac) = s.slots(w.slots.len());
            if w.slots.is_empty() {
                continue;
            }
            match w.kind {
                WheelKind::Color => {
                    let split = ColorSplit {
                        a: w.slots[a].transmission,
                        b: w.slots[b].transmission,
                        frac,
                    };
                    // A physical colour wheel splits the beam between slots; virtual
                    // wheels (macros) just blend.
                    if color_split.is_none() && !w.is_virtual {
                        color_split = Some(split);
                    } else {
                        color = mul(color, lerp(split.a, split.b, frac));
                    }
                }
                WheelKind::Gobo if gobos.len() < 2 => gobos.push(GoboLook {
                    media_a: w.slots[a].media.clone(),
                    media_b: w.slots[b].media.clone(),
                    frac,
                    rotation_deg: s.rotation_deg(),
                }),
                _ => {}
            }
        }

        BeamLook {
            color,
            color_split,
            gobos,
            field_angle: state.zoom_deg,
            // No focus channel: fixed optics with no focal plane (treated as infinity).
            focus_distance: if self.has_focus { focus_distance(state.focus) } else { f32::INFINITY },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_wheel_attributes() {
        assert_eq!(WheelKind::from_attribute("Gobo2"), Some(WheelKind::Gobo));
        assert_eq!(WheelKind::from_attribute("Color1"), Some(WheelKind::Color));
        assert_eq!(WheelKind::from_attribute("ColorMacro1"), Some(WheelKind::Color));
        assert_eq!(WheelKind::from_attribute("Prism1"), Some(WheelKind::Prism));
        assert_eq!(WheelKind::from_attribute("Gobo1WheelSpin"), None);
        assert_eq!(WheelKind::from_attribute("Gobo2Pos"), None);
    }

    #[test]
    fn wheel_slot_wraps() {
        let mut s = WheelState {
            slot: 9.25,
            ..Default::default()
        };
        assert_eq!(s.slots(10), (9, 0, 0.25));
        s.wheel_spin_dps = 36.0; // a tenth of a turn per second = one slot of ten
        s.update(1.0, 10);
        let (a, b, f) = s.slots(10);
        assert_eq!((a, b), (0, 1));
        assert!((f - 0.25).abs() < 1e-4);
    }

    fn optics(cmy: Vec<FilterUse>) -> Optics {
        Optics {
            source: [1.0; 3],
            native_cct: 6500.0,
            emitters: vec![],
            cmy,
            wheels: vec![],
            cto: Some(Range { from: 6500.0, to: 3200.0 }),
            zoom: None,
            has_focus: true,
        }
    }

    #[test]
    fn cmy_partial_insertion_is_linear_in_coverage() {
        let o = optics(vec![FilterUse {
            attr: "ColorSub_C".into(),
            name: "Cyan".into(),
            transmission: [0.0, 0.9, 1.0],
        }]);
        let mut st = o.default_state(20.0);
        st.cto_k = Some(6500.0);
        st.cmy_levels[0] = 0.5;
        let c = o.evaluate(&st).color;
        assert!((c[0] - 0.5).abs() < 1e-6 && (c[1] - 0.95).abs() < 1e-6 && c[2] == 1.0, "{c:?}");
    }

    #[test]
    fn cto_warms_and_never_amplifies() {
        let o = optics(vec![]);
        let mut st = o.default_state(20.0);
        st.cto_k = Some(3200.0);
        let c = o.evaluate(&st).color;
        assert!(c[0] > c[2], "{c:?}");
        assert!(c.iter().all(|&v| v <= 1.0 + 1e-6));
    }

    #[test]
    fn focus_round_trip() {
        for d in [2.0, 5.0, 12.0] {
            assert!((focus_distance(focus_for_distance(d)) - d).abs() < 1e-3);
        }
    }
}
