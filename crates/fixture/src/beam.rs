//! Photometric beam model: angular intensity profile, peak candela, colour.
//!
//! GDTF gives `BeamAngle` (full angle at 50% of peak intensity), `FieldAngle` (full angle
//! at 10%), `LuminousFlux` (lm) and `ColorTemperature`. We build an angular profile that
//! honours both angles and scale it so its integral over the sphere equals the flux.

use gdtf_loader::{BeamInfo, BeamShape};

use crate::optics::BeamLook;
use crate::{FixtureRig, NodeKind};

/// Widest angle from the axis a beam profile may reach, degrees.
pub const MAX_HALF_ANGLE: f32 = 85.0;

/// Relative intensity as a function of the angle from the beam axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamProfile {
    /// Half of BeamAngle, degrees (50% point).
    pub half_beam: f32,
    /// Half of FieldAngle, degrees (10% point).
    pub half_field: f32,
    /// Width of the fall-off to zero past the field edge, degrees.
    pub edge: f32,
}

impl BeamProfile {
    pub fn new(beam_angle: f32, field_angle: f32, shape: BeamShape) -> Self {
        let half_beam = (beam_angle * 0.5).clamp(0.1, MAX_HALF_ANGLE - 1.0);
        let half_field = (field_angle * 0.5).clamp(half_beam, MAX_HALF_ANGLE - 0.1);
        let edge = match shape {
            // Hard-edged optics: a crisp edge of about a degree.
            BeamShape::Spot | BeamShape::Rectangle => 0.5,
            // Soft optics fade out over a good part of the field.
            BeamShape::Soft | BeamShape::Glow => (half_field * 0.3).max(1.0),
        }
        // Keep the whole profile inside 85° of the axis (very wide floods/blinders):
        // projections and cone maths need less than a hemisphere.
        .min((MAX_HALF_ANGLE - half_field).max(0.1));
        Self {
            half_beam,
            half_field,
            edge,
        }
    }

    /// Angle (degrees from the axis) beyond which the intensity is zero.
    pub fn max_angle(&self) -> f32 {
        if self.is_hard() {
            self.half_beam + self.edge
        } else {
            self.half_field + self.edge
        }
    }

    /// Beam and field angle (nearly) equal: a flat top with a crisp edge.
    fn is_hard(&self) -> bool {
        self.half_field - self.half_beam < 0.5
    }

    /// Relative intensity 0..1 at `theta` degrees off-axis.
    pub fn eval(&self, theta: f32) -> f32 {
        let (a, b, e) = (self.half_beam, self.half_field, self.edge);
        if self.is_hard() {
            // 50% exactly at the beam edge, ~0 one edge-width later.
            return 1.0 - smoothstep(a - e, a + e, theta);
        }
        if theta <= a {
            1.0 - 0.5 * smoothstep(0.0, a, theta)
        } else if theta <= b {
            0.5 - 0.4 * smoothstep(a, b, theta)
        } else {
            0.1 * (1.0 - smoothstep(b, b + e, theta))
        }
    }

    /// Integral of the profile over the sphere, ∫ f(θ) dΩ (steradians).
    pub fn solid_angle(&self) -> f32 {
        let n = 2048;
        let max = self.max_angle().to_radians();
        let h = max / n as f32;
        (0..n)
            .map(|i| {
                let t = (i as f32 + 0.5) * h;
                self.eval(t.to_degrees()) * t.sin() * h
            })
            .sum::<f32>()
            * std::f32::consts::TAU
    }

    /// Samples the profile at `n` evenly spaced angles over `0..=max_angle()`.
    pub fn table(&self, n: usize) -> Vec<f32> {
        let max = self.max_angle();
        (0..n)
            .map(|i| self.eval(max * i as f32 / (n - 1) as f32))
            .collect()
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Working-space (linear Rec.2020) colour of a blackbody-ish white at `kelvin`, scaled to
/// luminance Y = 1. Uses the Kim et al. (2002) Planckian-locus fit, valid 1667–25000 K.
pub fn cct_to_working(kelvin: f32) -> [f32; 3] {
    let t = kelvin.clamp(1667.0, 25000.0) as f64;
    let (t2, t3) = (t * t, t * t * t);
    let x = if t <= 4000.0 {
        -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.179910
    } else {
        -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390
    };
    let (x2, x3) = (x * x, x * x * x);
    let y = if t <= 2222.0 {
        -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
    } else if t <= 4000.0 {
        -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
    } else {
        3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483
    };
    xyy_to_working(x as f32, y as f32, 1.0)
}

/// CIE xyY → working space (linear Rec.2020). The rare colours outside even Rec.2020
/// (spectral-locus lasers) are clipped.
pub fn xyy_to_working(x: f32, y: f32, big_y: f32) -> [f32; 3] {
    crate::color::xyz_to_working(crate::color::xyy_to_xyz(x, y, big_y)).map(|v| v.max(0.0))
}

/// Runtime state of one light-emitting Beam geometry.
#[derive(Debug, Clone)]
pub struct BeamState {
    /// Index into [`FixtureRig::nodes`].
    pub node: usize,
    pub info: BeamInfo,
    pub profile: BeamProfile,
    /// Peak (on-axis) intensity at full output, candela.
    pub peak_cd: f32,
    /// Colour, gobos and focus from the optical train ([`crate::Optics::evaluate`]).
    pub look: BeamLook,
    /// Current field angle (zoom), degrees, before frost.
    pub field_angle: f32,
    /// Current frost 0..1.
    pub frost: f32,
    /// 0..1.
    pub dimmer: f32,
}

impl BeamState {
    pub fn new(node: usize, info: &BeamInfo) -> Self {
        let cct = if info.color_temperature > 0.0 { info.color_temperature } else { 6000.0 };
        let mut b = Self {
            node,
            info: info.clone(),
            profile: BeamProfile::new(info.beam_angle, info.field_angle, info.shape),
            peak_cd: 0.0,
            look: BeamLook {
                color: cct_to_working(cct),
                color_split: None,
                gobos: Vec::new(),
                field_angle: None,
                focus_distance: 5.0,
                prism: None,
                frost: 0.0,
                spectrum: None,
            },
            field_angle: info.field_angle,
            frost: 0.0,
            dimmer: 1.0,
        };
        b.set_optics(info.field_angle, 0.0);
        b
    }

    /// Zoom: changes the field angle, scaling the beam angle with it. The flux stays the
    /// same, so a narrower beam gets a higher peak intensity.
    pub fn set_field_angle(&mut self, field: f32) {
        self.set_optics(field, self.look.frost);
    }

    /// Zoom and frost. Frost diffuses the beam: the beam angle grows towards the field,
    /// the field widens by up to 80% and the edge softens. Flux is kept, so the peak drops.
    fn set_optics(&mut self, field: f32, frost: f32) {
        let field = field.max(0.1);
        let frost = frost.clamp(0.0, 1.0);
        if self.peak_cd > 0.0 && (field - self.field_angle).abs() < 1e-4 && (frost - self.frost).abs() < 1e-4 {
            return;
        }
        let ratio = if self.info.field_angle > 0.0 { self.info.beam_angle / self.info.field_angle } else { 1.0 };
        self.field_angle = field;
        self.frost = frost;
        let beam = field * ratio;
        let frosted_field = field * (1.0 + 0.8 * frost);
        let frosted_beam = beam + (frosted_field * 0.85 - beam).max(0.0) * frost;
        let shape = if frost > 0.05 { BeamShape::Soft } else { self.info.shape };
        self.profile = BeamProfile::new(frosted_beam, frosted_field, shape);
        self.peak_cd = self.info.luminous_flux.max(0.0) / self.profile.solid_angle().max(1e-6);
    }

    /// Applies an evaluated look (colour, gobos, prism, zoom, focus, frost).
    pub fn set_look(&mut self, look: BeamLook) {
        let field = look.field_angle.unwrap_or(self.field_angle);
        self.set_optics(field, look.frost);
        self.look = look;
    }

    /// Whether this beam projects light (Glow / None beams only light their lens).
    pub fn emits_light(&self) -> bool {
        self.info.shape != BeamShape::Glow
    }

    /// Current on-axis intensity in candela.
    pub fn intensity_cd(&self) -> f32 {
        self.peak_cd * self.dimmer
    }
}

/// All Beam geometries of a rig.
pub fn beams(rig: &FixtureRig) -> Vec<BeamState> {
    rig.nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| match &n.kind {
            NodeKind::Beam(b) => Some(BeamState::new(i, b)),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn soft_profile_hits_50_and_10_percent() {
        // LEDBeam 150: beam 42.4°, field 60°, Wash.
        let p = BeamProfile::new(42.4, 60.0, BeamShape::Soft);
        assert_eq!(p.eval(0.0), 1.0);
        assert!(approx(p.eval(21.2), 0.5, 1e-4));
        assert!(approx(p.eval(30.0), 0.1, 1e-4));
        assert_eq!(p.eval(p.max_angle()), 0.0);
        let t = p.table(256);
        assert!(t.windows(2).all(|w| w[1] <= w[0] + 1e-6), "profile must not increase");
    }

    #[test]
    fn hard_spot_has_crisp_edge() {
        // MegaPointe: beam = field = 42°, Spot.
        let p = BeamProfile::new(42.0, 42.0, BeamShape::Spot);
        assert!(p.eval(20.0) > 0.99);
        assert!(approx(p.eval(21.0), 0.5, 1e-4));
        assert!(p.eval(21.4) < 0.11);
        assert_eq!(p.eval(21.6), 0.0);
    }

    #[test]
    fn peak_intensity_matches_flat_cone_estimate_for_hard_spot() {
        // For a flat-topped cone, I = Φ / (2π(1 − cos(θ/2))).
        let info = BeamInfo {
            beam_type: gdtf_loader::gdtf::geometry::BeamType::Spot,
            shape: BeamShape::Spot,
            beam_angle: 42.0,
            field_angle: 42.0,
            beam_radius: 0.09,
            luminous_flux: 20375.0,
            color_temperature: 6300.0,
            cri: 80,
            rectangle_ratio: 1.0,
            emitter_spectrum: None,
        };
        let b = BeamState::new(0, &info);
        let flat = 20375.0 / (std::f32::consts::TAU * (1.0 - 21f32.to_radians().cos()));
        assert!(approx(b.peak_cd / flat, 1.0, 0.02), "{} vs {}", b.peak_cd, flat);
    }

    #[test]
    fn very_wide_beams_stay_below_a_hemisphere() {
        // FOS F-6 GO plate: 123° beam, 160° field.
        let p = BeamProfile::new(123.0, 160.0, BeamShape::Soft);
        assert!(p.max_angle() <= MAX_HALF_ANGLE + 1e-4, "{}", p.max_angle());
        assert!((p.eval(61.5) - 0.5).abs() < 1e-4);
        assert!((p.eval(80.0) - 0.1).abs() < 1e-4);
    }

    #[test]
    fn cct_colours() {
        // A 6504 K blackbody is close to, but slightly magenta of, D65 (D65 lies just
        // above the Planckian locus), so allow a few percent.
        let d65 = cct_to_working(6504.0);
        assert!(d65.iter().all(|&c| approx(c, 1.0, 0.06)), "{d65:?}");
        assert!(d65[1] < d65[0] && d65[1] < d65[2]);
        let tungsten = cct_to_working(3200.0);
        assert!(tungsten[0] > tungsten[1] && tungsten[1] > tungsten[2]);
        // Luminance-normalised.
        let y = crate::color::luminance(tungsten);
        assert!(approx(y, 1.0, 0.01), "{y}");
    }
}
