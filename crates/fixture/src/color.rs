//! Colour science: spectra, CIE 1931 colour matching, and the renderer's working space.
//!
//! The renderer works in **linear Rec.2020** (wide gamut, D65): saturated LED and filter
//! colours stay positive there, and the tonemap pass maps to the display (Rec.709/sRGB).

pub type Rgb = [f32; 3];

/// Spectra are sampled at 380..=780 nm in 5 nm steps.
pub const SPECTRUM_START: f32 = 380.0;
pub const SPECTRUM_STEP: f32 = 5.0;
pub const SPECTRUM_LEN: usize = 81;

pub fn wavelength(i: usize) -> f32 {
    SPECTRUM_START + SPECTRUM_STEP * i as f32
}

/// CIE 1931 2° colour matching functions, multi-lobe Gaussian fit by Wyman, Sloan &
/// Shirley, "Simple Analytic Approximations to the CIE XYZ Color Matching Functions"
/// (JCGT 2013). Within about 1% of the tabulated functions.
pub fn cie_xyz(lambda: f32) -> [f32; 3] {
    let g = |mu: f32, s1: f32, s2: f32| {
        let t = (lambda - mu) / if lambda < mu { s1 } else { s2 };
        (-0.5 * t * t).exp()
    };
    [
        1.056 * g(599.8, 37.9, 31.0) + 0.362 * g(442.0, 16.0, 26.7) - 0.065 * g(501.1, 20.4, 26.2),
        0.821 * g(568.8, 46.9, 40.5) + 0.286 * g(530.9, 16.3, 31.1),
        1.217 * g(437.0, 11.8, 36.0) + 0.681 * g(459.0, 26.0, 13.8),
    ]
}

/// A spectral distribution (power, or transmission 0..1) on the fixed wavelength grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spectrum(pub [f32; SPECTRUM_LEN]);

impl Default for Spectrum {
    fn default() -> Self {
        Self([0.0; SPECTRUM_LEN])
    }
}

impl Spectrum {
    pub fn constant(v: f32) -> Self {
        Self([v; SPECTRUM_LEN])
    }

    /// Resamples measured `(wavelength nm, value)` points (any order/spacing) linearly;
    /// zero outside the measured range.
    pub fn from_points(points: &[(f32, f32)]) -> Option<Self> {
        let mut p: Vec<(f32, f32)> = points.iter().copied().filter(|(w, v)| w.is_finite() && v.is_finite()).collect();
        if p.len() < 2 {
            return None;
        }
        p.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut s = [0.0; SPECTRUM_LEN];
        for (i, out) in s.iter_mut().enumerate() {
            let w = wavelength(i);
            if w < p[0].0 || w > p[p.len() - 1].0 {
                continue;
            }
            let k = p.partition_point(|q| q.0 < w).clamp(1, p.len() - 1);
            let (a, b) = (p[k - 1], p[k]);
            let t = if b.0 > a.0 { (w - a.0) / (b.0 - a.0) } else { 0.0 };
            *out = a.1 + (b.1 - a.1) * t;
        }
        Some(Self(s))
    }

    /// Planck blackbody radiance at `kelvin` (arbitrary scale).
    pub fn blackbody(kelvin: f32) -> Self {
        let (c1, c2) = (3.741_771_8e-16_f64, 1.438_776_9e-2_f64);
        let t = kelvin.max(500.0) as f64;
        let mut s = [0.0; SPECTRUM_LEN];
        for (i, out) in s.iter_mut().enumerate() {
            let l = wavelength(i) as f64 * 1e-9;
            *out = (c1 / (l.powi(5) * ((c2 / (l * t)).exp() - 1.0)) * 1e-12) as f32;
        }
        Self(s)
    }

    pub fn mul(&self, o: &Spectrum) -> Spectrum {
        Self(std::array::from_fn(|i| self.0[i] * o.0[i]))
    }

    pub fn add_scaled(&mut self, o: &Spectrum, k: f32) {
        for (a, b) in self.0.iter_mut().zip(o.0) {
            *a += b * k;
        }
    }

    pub fn scaled(&self, k: f32) -> Spectrum {
        Self(self.0.map(|v| v * k))
    }

    /// Per-wavelength ratio `self / reference`, clamped to 0..1 (a transmission), where the
    /// reference has meaningful power; elsewhere `fallback`.
    pub fn ratio(&self, reference: &Spectrum, fallback: f32) -> Spectrum {
        let peak = reference.0.iter().copied().fold(0.0, f32::max);
        Self(std::array::from_fn(|i| {
            if reference.0[i] > peak * 1e-3 {
                (self.0[i] / reference.0[i]).clamp(0.0, 1.0)
            } else {
                fallback
            }
        }))
    }

    /// CIE XYZ (unnormalised: Σ S·cmf·Δλ).
    pub fn to_xyz(&self) -> [f32; 3] {
        let mut xyz = [0.0f32; 3];
        for (i, v) in self.0.iter().enumerate() {
            let c = cie_xyz(wavelength(i));
            for k in 0..3 {
                xyz[k] += v * c[k] * SPECTRUM_STEP;
            }
        }
        xyz
    }

    pub fn luminance(&self) -> f32 {
        self.to_xyz()[1]
    }
}

pub fn xyy_to_xyz(x: f32, y: f32, big_y: f32) -> [f32; 3] {
    if y <= 1e-6 {
        return [0.0; 3];
    }
    [x / y * big_y, big_y, (1.0 - x - y) / y * big_y]
}

pub fn xyz_to_xy(xyz: [f32; 3]) -> [f32; 2] {
    let s = xyz[0] + xyz[1] + xyz[2];
    if s <= 0.0 { [0.3127, 0.3290] } else { [xyz[0] / s, xyz[1] / s] }
}

fn mat(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2])
}

const XYZ_TO_REC2020: [[f32; 3]; 3] = [
    [1.716_651_2, -0.355_670_8, -0.253_366_3],
    [-0.666_684_4, 1.616_481_2, 0.015_768_5],
    [0.017_639_9, -0.042_770_6, 0.942_103_1],
];
const REC2020_TO_XYZ: [[f32; 3]; 3] = [
    [0.636_958, 0.144_616_9, 0.168_881],
    [0.262_700_2, 0.677_998_1, 0.059_301_7],
    [0.0, 0.028_072_7, 1.060_985_1],
];
pub const REC709_TO_REC2020: [[f32; 3]; 3] = [
    [0.627_404, 0.329_282, 0.043_313_6],
    [0.069_097, 0.919_540, 0.011_361_2],
    [0.016_391_6, 0.088_013_2, 0.895_595],
];
pub const REC2020_TO_REC709: [[f32; 3]; 3] = [
    [1.660_491, -0.587_641_1, -0.072_849_9],
    [-0.124_550_5, 1.132_899_9, -0.008_349_4],
    [-0.018_150_8, -0.100_578_9, 1.118_729_7],
];

/// XYZ → working space (linear Rec.2020).
pub fn xyz_to_working(xyz: [f32; 3]) -> Rgb {
    mat(&XYZ_TO_REC2020, xyz)
}

pub fn working_to_xyz(c: Rgb) -> [f32; 3] {
    mat(&REC2020_TO_XYZ, c)
}

/// Linear sRGB / Rec.709 (e.g. glTF materials, PNG gobos) → working space.
pub fn srgb_linear_to_working(c: Rgb) -> Rgb {
    mat(&REC709_TO_REC2020, c)
}

/// Working space → linear sRGB, for UI swatches (may be out of gamut: clipped).
pub fn working_to_srgb_linear(c: Rgb) -> Rgb {
    mat(&REC2020_TO_REC709, c).map(|v| v.max(0.0))
}

/// Relative luminance of a working-space colour.
pub fn luminance(c: Rgb) -> f32 {
    0.262_700_2 * c[0] + 0.677_998_1 * c[1] + 0.059_301_7 * c[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colour_matching_fit_is_sane() {
        // ȳ peaks at ~1 near 555 nm; equal-energy white is at x = y = 1/3.
        assert!((cie_xyz(555.0)[1] - 1.0).abs() < 0.02, "{:?}", cie_xyz(555.0));
        let xy = xyz_to_xy(Spectrum::constant(1.0).to_xyz());
        assert!((xy[0] - 1.0 / 3.0).abs() < 0.006 && (xy[1] - 1.0 / 3.0).abs() < 0.006, "{xy:?}");
    }

    #[test]
    fn blackbody_chromaticity() {
        // 2856 K (CIE illuminant A) is at about (0.4476, 0.4074).
        let xy = xyz_to_xy(Spectrum::blackbody(2856.0).to_xyz());
        assert!((xy[0] - 0.4476).abs() < 0.006 && (xy[1] - 0.4074).abs() < 0.006, "{xy:?}");
    }

    #[test]
    fn working_space_round_trips_and_keeps_white() {
        let c = srgb_linear_to_working([0.2, 0.5, 0.9]);
        let back = working_to_srgb_linear(c);
        assert!((back[0] - 0.2).abs() < 1e-3 && (back[1] - 0.5).abs() < 1e-3 && (back[2] - 0.9).abs() < 1e-3);
        let white = xyz_to_working(xyy_to_xyz(0.3127, 0.3290, 1.0));
        assert!(white.iter().all(|v| (v - 1.0).abs() < 2e-3), "{white:?}");
        assert!((luminance(white) - 1.0).abs() < 2e-3);
    }

    #[test]
    fn resampling() {
        let s = Spectrum::from_points(&[(400.0, 0.0), (500.0, 1.0)]).unwrap();
        assert_eq!(s.0[0], 0.0); // 380: outside → 0
        assert_eq!(s.0[4], 0.0); // 400
        assert!((s.0[14] - 0.5).abs() < 1e-6); // 450
        assert_eq!(s.0[24], 1.0); // 500
        assert_eq!(s.0[25], 0.0); // 505: outside
    }
}
