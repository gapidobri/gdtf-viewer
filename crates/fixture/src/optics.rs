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

use crate::beam::{cct_to_working, xyy_to_working};
use crate::color::{self, Spectrum};

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
    /// Prism facets (empty for non-prism slots).
    pub facets: Vec<Facet>,
}

/// One prism facet: maps a direction of the unaltered beam `u₀` (in units of its radius,
/// projector x/y) to `A·u₀ + t`, with a colour tint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Facet {
    /// Column-major 2×2: [a00, a10, a01, a11].
    pub a: [f32; 4],
    pub t: [f32; 2],
    pub tint: Rgb,
}

impl Facet {
    /// GDTF stores `{a,b,0}{c,d,0}{tx,ty,1}` for row vectors (`[x y 1]·M`), i.e.
    /// `x' = a·x + c·y + tx`, `y' = b·x + d·y + ty`.
    pub fn from_gdtf_rows(m: [[f32; 3]; 3], tint: Rgb) -> Self {
        Self {
            a: [m[0][0], m[0][1], m[1][0], m[1][1]],
            t: [m[2][0], m[2][1]],
            tint,
        }
    }

    pub fn apply(&self, u: [f32; 2]) -> [f32; 2] {
        [
            self.a[0] * u[0] + self.a[2] * u[1] + self.t[0],
            self.a[1] * u[0] + self.a[3] * u[1] + self.t[1],
        ]
    }

    pub fn det(&self) -> f32 {
        self.a[0] * self.a[3] - self.a[2] * self.a[1]
    }
}

/// The `gdtf` crate keeps `Rotation`'s values private; read them from its derived Debug
/// output, `Rotation([[a, b, c], [d, e, f], [g, h, i]])`, rows as in the file.
fn rotation_rows(r: &gdtf_loader::gdtf::values::Rotation) -> Option<[[f32; 3]; 3]> {
    let s = format!("{r:?}");
    let v: Vec<f32> = s
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.parse().ok())
        .collect();
    (v.len() == 9).then(|| [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]])
}

/// Prism in the beam: facets, rotated as a whole.
#[derive(Debug, Clone, PartialEq)]
pub struct PrismLook {
    pub facets: Vec<Facet>,
    pub rotation_deg: f32,
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
    /// Working space; colours outside it are clipped, see `xyz`.
    pub rgb: Rgb,
    /// Share of the total output at full level (all emitters full → 1).
    pub share: f32,
    /// CIE XYZ at full level (Y = share), exact even outside the working gamut.
    pub xyz: [f32; 3],
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
    pub has_frost: bool,
    pub spectral: SpectralOptics,
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
    /// 0..1.
    pub frost: f32,
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
    pub prism: Option<PrismLook>,
    /// Frost 0..1: softens and widens the beam, blurs gobos.
    pub frost: f32,
    /// The light's spectrum after measured filters (if the file has spectral data).
    pub spectrum: Option<Spectrum>,
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
    color::luminance(c)
}

/// The measurement with the highest `Physical` value (full output / full insertion).
fn measured(ms: &[gdtf_loader::gdtf::physical_descriptions::Measurement]) -> Option<Spectrum> {
    let m = ms
        .iter()
        .filter(|m| m.points.len() >= 2)
        .max_by(|a, b| a.physical.total_cmp(&b.physical))?;
    let pts: Vec<(f32, f32)> = m.points.iter().map(|p| (p.wave_length as f32, p.energy as f32)).collect();
    Spectrum::from_points(&pts)
}

/// Resolves measured spectra for the source, emitters, CMY flags and colour-wheel slots.
///
/// Filter measurements in real files are often the light *through* the filter (e.g. Robe:
/// a "White" filter that is the lamp, referenced by the beam's `EmitterSpectrum`), so when
/// the source spectrum comes from the file, transmissions are taken as filter / source.
fn build_spectral(
    ft: &FixtureType,
    beam: Option<&BeamInfo>,
    emitters: &[EmitterUse],
    cmy: &[FilterUse],
    wheels: &[WheelUse],
    native_cct: f32,
) -> SpectralOptics {
    let pd = &ft.physical_descriptions;
    let filter_spectrum = |name: &str| pd.filter(name).and_then(|f| measured(&f.measurements));
    let emitter_spectrum = |name: &str| pd.emitter(name).and_then(|e| measured(&e.measurements));

    // Source: the beam's EmitterSpectrum (an emitter, or in practice often a filter), else
    // a filter named like a reference white, else a blackbody at the beam's CCT.
    let named = beam.and_then(|b| b.emitter_spectrum.clone());
    let mut label = String::new();
    let mut from_file = named.as_deref().and_then(|n| {
        let s = emitter_spectrum(n).or_else(|| filter_spectrum(n));
        if s.is_some() {
            label = format!("measured \"{n}\"");
        }
        s
    });
    if from_file.is_none()
        && let Some(f) = pd.filters.iter().find(|f| {
            let n = f.name.as_deref().unwrap_or_default().to_lowercase();
            (n == "white" || n == "open") && measured(&f.measurements).is_some()
        })
    {
        from_file = measured(&f.measurements);
        label = format!("measured \"{}\"", f.name.as_deref().unwrap_or_default());
    }
    let source = match from_file {
        Some(s) => Some(s),
        None if emitters.is_empty() => {
            label = format!("blackbody {native_cct:.0} K");
            Some(Spectrum::blackbody(native_cct))
        }
        None => None,
    }
    .map(|s| s.scaled(1.0 / s.luminance().max(1e-12)));

    // Filter spectrum → transmission: relative to the raw measured source when there is
    // one (both are light measured the same way); otherwise the values are taken as a
    // transmission, scaled down if they exceed 1.
    let transmission_of = |name: &str| -> Option<Spectrum> {
        let f = filter_spectrum(name)?;
        Some(match &from_file {
            Some(raw_source) => f.ratio(raw_source, 0.0),
            None => {
                let peak = f.0.iter().copied().fold(0.0, f32::max);
                if peak > 1.05 { f.scaled(1.0 / peak) } else { f }
            }
        })
    };

    let emitter_spectra = emitters
        .iter()
        .map(|e| {
            let s = emitter_spectrum(&e.name)?;
            // Scale to this emitter's luminance share.
            Some(s.scaled(e.share / s.luminance().max(1e-12)))
        })
        .collect();
    let cmy_spectra = cmy.iter().map(|c| transmission_of(&c.name)).collect();

    // Wheel slots: a Filter link, or (common in practice) a filter whose name the slot's
    // name contains, e.g. slot "C01 (Dark Red)" ↔ filter "Dark red".
    let slots = wheels
        .iter()
        .map(|w| {
            let wheel = ft.wheels.iter().find(|x| x.name.as_deref() == Some(w.wheel_name.as_str()));
            w.slots
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    if w.kind != WheelKind::Color || w.is_virtual {
                        // Virtual wheels (macros) are CMY/LED mixes, not glass.
                        return None;
                    }
                    let linked = wheel
                        .and_then(|wh| wh.slots.get(i))
                        .and_then(|gs| match &gs.optic {
                            WheelSlotOptic::Filter(n) => Some(n.to_string()),
                            _ => None,
                        });
                    let name = linked.or_else(|| {
                        let slot = s.name.to_lowercase();
                        pd.filters
                            .iter()
                            .filter_map(|f| f.name.as_deref())
                            .filter(|n| n.len() >= 3 && slot.contains(&n.to_lowercase()) && measured_filter(pd, n))
                            .max_by_key(|n| n.len())
                            .map(str::to_owned)
                    })?;
                    let t = transmission_of(&name)?;
                    // Name matches are a heuristic: accept only if the measured filter's
                    // luminous transmission agrees with the slot's own Y (within 30%).
                    let slot_y = wheel.and_then(|wh| wh.slots.get(i)).and_then(|gs| match &gs.optic {
                        WheelSlotOptic::Color(c) => Some(c.z as f32 / 100.0),
                        WheelSlotOptic::Filter(_) => None,
                    });
                    if let (Some(src), Some(y)) = (&source, slot_y) {
                        let lt = src.mul(&t).luminance() / src.luminance().max(1e-12);
                        if y > 0.0 && (lt / y - 1.0).abs() > 0.3 && (lt - y).abs() > 0.01 {
                            log::debug!("slot '{}' vs filter '{name}': {lt:.3} vs {y:.3}, not used", s.name);
                            return None;
                        }
                    }
                    Some(t)
                })
                .collect()
        })
        .collect();

    SpectralOptics {
        source,
        emitters: emitter_spectra,
        cmy: cmy_spectra,
        slots,
        source_label: label,
    }
}

fn measured_filter(pd: &gdtf_loader::gdtf::physical_descriptions::PhysicalDescriptions, name: &str) -> bool {
    pd.filter(name).is_some_and(|f| measured(&f.measurements).is_some())
}

/// Measured spectral data, where the file has it (GDTF PhysicalDescriptions).
#[derive(Debug, Clone, Default)]
pub struct SpectralOptics {
    /// Unfiltered white of the light source, luminance 1 (subtractive fixtures).
    pub source: Option<Spectrum>,
    /// Per emitter (parallel to `Optics::emitters`), luminance = its share (all full → 1).
    pub emitters: Vec<Option<Spectrum>>,
    /// Transmission per CMY flag (parallel to `Optics::cmy`).
    pub cmy: Vec<Option<Spectrum>>,
    /// Transmission per wheel slot (parallel to `Optics::wheels[i].slots`).
    pub slots: Vec<Vec<Option<Spectrum>>>,
    /// Where the source spectrum came from, for the UI.
    pub source_label: String,
}

/// Transmission of a colour (CIE xyY, Y in percent) relative to a white, clamped to 0..1.
fn transmission(x: f32, y: f32, big_y_percent: f32, white: Rgb) -> Rgb {
    let c = xyy_to_working(x, y, big_y_percent / 100.0);
    [0, 1, 2].map(|i| (c[i] / white[i].max(1e-6)).clamp(0.0, 1.0))
}

impl Optics {
    pub fn from_mode(ft: &FixtureType, mode: &DmxMode, beam: Option<&BeamInfo>) -> Self {
        let native_cct = beam.map_or(6000.0, |b| b.color_temperature).max(1000.0);
        let source = cct_to_working(native_cct);

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
        let reference = white_xy.map_or(source, |(x, y)| xyy_to_working(x, y, 1.0));

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
            let rgb = xyy_to_working(color.x as f32, color.y as f32, 1.0);
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
                share: 1.0,
                xyz: color::xyy_to_xyz(color.x as f32, color.y as f32, 1.0),
            });
        }
        let total: f32 = weights.iter().sum();
        for (e, w) in emitters.iter_mut().zip(&weights) {
            let share = if total > 0.0 { w / total } else { 1.0 / weights.len() as f32 };
            e.rgb = e.rgb.map(|c| c * share);
            e.share = share;
            e.xyz = e.xyz.map(|c| c * share);
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
                    facets: s
                        .facets
                        .iter()
                        .filter_map(|f| {
                            let rows = rotation_rows(&f.rotation)?;
                            let tint = transmission(f.color.x as f32, f.color.y as f32, f.color.z as f32, reference);
                            Some(Facet::from_gdtf_rows(rows, tint))
                        })
                        .collect(),
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

        let spectral = build_spectral(ft, beam, &emitters, &cmy, &wheels, native_cct);
        Self {
            source,
            native_cct,
            emitters,
            cmy,
            wheels,
            cto,
            zoom,
            has_focus: find("Focus1").is_some(),
            has_frost: ["Frost1", "Frost2", "Frost3"].iter().any(|a| find(a).is_some()),
            spectral,
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
            frost: 0.0,
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
        // Solve in XYZ (exact even for LED colours outside the working gamut), from the
        // measured spectra where available.
        let xyz = |i: usize| -> glam::Vec3 {
            self.spectral
                .emitters
                .get(i)
                .and_then(Option::as_ref)
                .map_or(self.emitters[i].xyz, |s| s.to_xyz())
                .into()
        };
        let m = glam::Mat3::from_cols(xyz(r), xyz(g), xyz(b));
        if m.determinant().abs() < 1e-12 {
            return full;
        }
        let x = m.inverse() * glam::Vec3::from(color::working_to_xyz(self.source));
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
        let sp = &self.spectral;
        // Spectral path: the light as a spectrum where the file measured it. Factors
        // without measurements (CTO, xy-only filters) multiply in RGB afterwards.
        let mut spectrum: Option<Spectrum> = if self.emitters.is_empty() {
            sp.source
        } else if !sp.emitters.is_empty() && sp.emitters.iter().all(Option::is_some) {
            let mut s = Spectrum::default();
            for (e, &l) in sp.emitters.iter().zip(&state.emitter_levels) {
                s.add_scaled(e.as_ref().unwrap(), l.clamp(0.0, 1.0));
            }
            Some(s)
        } else {
            None
        };
        let mut tint = [1.0f32; 3];

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

        for (i, (f, &level)) in self.cmy.iter().zip(&state.cmy_levels).enumerate() {
            // Partial insertion: the flag covers a fraction of the beam, 1 − f + f·T.
            let level = level.clamp(0.0, 1.0);
            match (&mut spectrum, sp.cmy.get(i).and_then(Option::as_ref)) {
                (Some(s), Some(t)) => {
                    *s = s.mul(&Spectrum(t.0.map(|v| 1.0 - level + level * v)));
                }
                _ => tint = mul(tint, lerp([1.0; 3], f.transmission, level)),
            }
            color = mul(color, lerp([1.0; 3], f.transmission, level));
        }

        if let (Some(_), Some(k)) = (self.cto, state.cto_k)
            && (k - self.native_cct).abs() > 1.0
        {
            let to = cct_to_working(k);
            let from = cct_to_working(self.native_cct);
            let ratio = [0, 1, 2].map(|i| to[i] / from[i].max(1e-6));
            let m = ratio.iter().copied().fold(0.0, f32::max).max(1e-6);
            color = mul(color, ratio.map(|r| r / m));
            tint = mul(tint, ratio.map(|r| r / m));
        }

        let mut color_split = None;
        let mut gobos = Vec::new();
        let mut prism = None;
        for (w, s) in self.wheels.iter().zip(&state.wheels) {
            let (a, b, frac) = s.slots(w.slots.len());
            if w.slots.is_empty() {
                continue;
            }
            match w.kind {
                WheelKind::Color => {
                    let wi = self.wheels.iter().position(|x| std::ptr::eq(x, w)).unwrap_or(0);
                    let ts = |k: usize| sp.slots.get(wi).and_then(|v| v.get(k)).and_then(Option::as_ref);
                    // Spectral slots become RGB transmissions under the current source,
                    // T_rgb = RGB(S·T) / RGB(S) per channel.
                    let rgb_t = |k: usize| match (&spectrum, ts(k)) {
                        (Some(s), Some(t)) => {
                            let base = color::xyz_to_working(s.to_xyz());
                            let through = color::xyz_to_working(s.mul(t).to_xyz());
                            [0, 1, 2].map(|c| if base[c] > 1e-9 { (through[c] / base[c]).clamp(0.0, 1.5) } else { 0.0 })
                        }
                        _ => w.slots[k].transmission,
                    };
                    let split = ColorSplit {
                        a: rgb_t(a),
                        b: rgb_t(b),
                        frac,
                    };
                    // A physical colour wheel splits the beam between slots; virtual
                    // wheels (macros) just blend.
                    if color_split.is_none() && !w.is_virtual {
                        color_split = Some(split);
                    } else {
                        match (&mut spectrum, ts(a), ts(b)) {
                            (Some(s), Some(ta), Some(tb)) => {
                                *s = s.mul(&Spectrum(std::array::from_fn(|k| ta.0[k] + (tb.0[k] - ta.0[k]) * frac)));
                            }
                            _ => tint = mul(tint, lerp(split.a, split.b, frac)),
                        }
                        color = mul(color, lerp(split.a, split.b, frac));
                    }
                }
                WheelKind::Gobo if gobos.len() < 2 => gobos.push(GoboLook {
                    media_a: w.slots[a].media.clone(),
                    media_b: w.slots[b].media.clone(),
                    frac,
                    rotation_deg: s.rotation_deg(),
                }),
                // A prism swings in as a whole: the nearer slot is in the beam. Only the
                // first prism in the beam is used (stacked prisms multiply facets).
                WheelKind::Prism if prism.is_none() => {
                    let slot = &w.slots[if frac < 0.5 { a } else { b }];
                    if !slot.facets.is_empty() {
                        prism = Some(PrismLook {
                            facets: slot.facets.clone(),
                            rotation_deg: s.rotation_deg(),
                        });
                    }
                }
                _ => {}
            }
        }

        // With a spectrum, the colour is its integral (luminance relative to the unfiltered
        // source / all emitters full) times the RGB-only factors.
        if let Some(s) = &spectrum {
            color = mul(color::xyz_to_working(s.to_xyz()).map(|v| v.max(0.0)), tint);
        }

        BeamLook {
            color,
            color_split,
            spectrum,
            gobos,
            field_angle: state.zoom_deg,
            // No focus channel: fixed optics with no focal plane (treated as infinity).
            focus_distance: if self.has_focus { focus_distance(state.focus) } else { f32::INFINITY },
            prism,
            frost: if self.has_frost { state.frost.clamp(0.0, 1.0) } else { 0.0 },
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
            has_frost: false,
            spectral: SpectralOptics::default(),
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
    fn facet_from_gdtf_rows() {
        // MegaPointe 8F circular, first facet: translation in the third row.
        let f = Facet::from_gdtf_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.363822, 0.564914, 1.0]], [1.0; 3]);
        assert_eq!(f.apply([0.0, 0.0]), [1.363822, 0.564914]);
        // Cylindrical: stretches x.
        let c = Facet::from_gdtf_rows([[4.952, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [1.0; 3]);
        assert_eq!(c.apply([1.0, 1.0]), [4.952, 1.0]);
        assert!((c.det() - 4.952).abs() < 1e-6);
    }

    #[test]
    fn focus_round_trip() {
        for d in [2.0, 5.0, 12.0] {
            assert!((focus_distance(focus_for_distance(d)) - d).abs() < 1e-3);
        }
    }
}
