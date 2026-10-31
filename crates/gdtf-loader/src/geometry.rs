use gdtf::geometry::{BeamType, Geometry};
use gdtf::values::Matrix;
use glam::Mat4;

/// One node of the fixture's geometry tree.
#[derive(Debug, Clone)]
pub struct GeometryNode {
    pub name: String,
    pub model: Option<String>,
    /// Transform relative to the parent, GDTF space.
    pub local: Mat4,
    pub kind: GeometryKind,
    pub children: Vec<GeometryNode>,
}

#[derive(Debug, Clone)]
pub enum GeometryKind {
    /// Static part.
    Static,
    /// Rotating part, driven by Pan/Tilt (or other rotation) attributes.
    Axis,
    /// Light-emitting part.
    Beam(BeamInfo),
    FilterBeam,
    FilterColor,
    FilterGobo,
    FilterShaper,
    /// Instance of another top-level geometry (multi-cell fixtures).
    Reference { geometry: Option<String> },
    /// Anything we don't simulate yet (lasers, wiring, media servers, ...).
    Other(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeamShape {
    /// Cone, hard edge.
    Spot,
    /// Cone, soft edge (Wash / PC / Fresnel).
    Soft,
    /// Pyramid frustum, hard edge.
    Rectangle,
    /// No beam, emissive surface only.
    Glow,
}

#[derive(Debug, Clone)]
pub struct BeamInfo {
    pub beam_type: BeamType,
    pub shape: BeamShape,
    /// Degrees, full angle at 50% intensity.
    pub beam_angle: f32,
    /// Degrees, full angle at 10% intensity.
    pub field_angle: f32,
    /// Metres, radius of the light at the lens.
    pub beam_radius: f32,
    pub luminous_flux: f32,
    pub color_temperature: f32,
    pub cri: u32,
    pub rectangle_ratio: f32,
    pub emitter_spectrum: Option<String>,
}

impl GeometryNode {
    /// Depth-first visit with accumulated world transforms (GDTF space).
    pub fn walk(&self, parent: Mat4, f: &mut impl FnMut(&GeometryNode, Mat4, usize)) {
        self.walk_inner(parent, 0, f)
    }

    fn walk_inner(&self, parent: Mat4, depth: usize, f: &mut impl FnMut(&GeometryNode, Mat4, usize)) {
        let world = parent * self.local;
        f(self, world, depth);
        for c in &self.children {
            c.walk_inner(world, depth + 1, f);
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match &self.kind {
            GeometryKind::Static => "Geometry",
            GeometryKind::Axis => "Axis",
            GeometryKind::Beam(_) => "Beam",
            GeometryKind::FilterBeam => "FilterBeam",
            GeometryKind::FilterColor => "FilterColor",
            GeometryKind::FilterGobo => "FilterGobo",
            GeometryKind::FilterShaper => "FilterShaper",
            GeometryKind::Reference { .. } => "GeometryReference",
            GeometryKind::Other(n) => n,
        }
    }
}

/// The `gdtf` crate keeps `Matrix`'s fields private and offers no accessor, so read the
/// values back from its derived `Debug` output: `Matrix([[a, b, c, d], [...], ...])`.
/// GDTF matrices are row-major with the translation in the 4th column.
// TODO: replace once upstream exposes the values (worth a PR to cpdt/gdtf-rs).
fn matrix_to_mat4(m: &Matrix) -> Mat4 {
    parse_matrix_debug(&format!("{m:?}"))
}

fn parse_matrix_debug(s: &str) -> Mat4 {
    let vals: Vec<f32> = s
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.parse().ok())
        .collect();
    if vals.len() != 16 {
        log::warn!("could not read geometry matrix from {s}; using identity");
        return Mat4::IDENTITY;
    }
    let mut rows = [0f32; 16];
    rows.copy_from_slice(&vals);
    // from_cols_array reads column-major, so a row-major array needs a transpose.
    Mat4::from_cols_array(&rows).transpose()
}

macro_rules! common {
    ($g:expr) => {
        (
            $g.name.as_ref().map(|n| n.to_string()).unwrap_or_default(),
            $g.model.as_ref().map(|n| n.to_string()),
            matrix_to_mat4(&$g.position),
        )
    };
}

pub(crate) fn convert(g: &Geometry) -> GeometryNode {
    let (name, model, local, kind, children) = match g {
        Geometry::Generic(g) => {
            let (n, m, l) = common!(g);
            (n, m, l, GeometryKind::Static, &g.children[..])
        }
        Geometry::Axis(g) => {
            let (n, m, l) = common!(g);
            (n, m, l, GeometryKind::Axis, &g.children[..])
        }
        Geometry::FilterBeam(g) => {
            let (n, m, l) = common!(g);
            (n, m, l, GeometryKind::FilterBeam, &g.children[..])
        }
        Geometry::FilterColor(g) => {
            let (n, m, l) = common!(g);
            (n, m, l, GeometryKind::FilterColor, &g.children[..])
        }
        Geometry::FilterGobo(g) => {
            let (n, m, l) = common!(g);
            (n, m, l, GeometryKind::FilterGobo, &g.children[..])
        }
        Geometry::FilterShaper(g) => {
            let (n, m, l) = common!(g);
            (n, m, l, GeometryKind::FilterShaper, &g.children[..])
        }
        Geometry::Beam(g) => {
            let (n, m, l) = common!(g);
            let shape = match g.beam_type {
                BeamType::Spot => BeamShape::Spot,
                BeamType::Rectangle => BeamShape::Rectangle,
                BeamType::Wash | BeamType::Pc | BeamType::Fresnel => BeamShape::Soft,
                BeamType::Glow | BeamType::None => BeamShape::Glow,
            };
            let info = BeamInfo {
                beam_type: g.beam_type,
                shape,
                beam_angle: g.beam_angle as f32,
                field_angle: g.field_angle as f32,
                beam_radius: g.beam_radius as f32,
                luminous_flux: g.luminous_flux as f32,
                color_temperature: g.color_temperature as f32,
                cri: g.color_rendering_index,
                rectangle_ratio: g.rectangle_ratio as f32,
                emitter_spectrum: g.emitter_spectrum.as_ref().map(|n| n.to_string()),
            };
            (n, m, l, GeometryKind::Beam(info), &g.children[..])
        }
        Geometry::Reference(g) => {
            let (n, m, l) = common!(g);
            let geometry = g.geometry.as_ref().map(|n| n.to_string());
            (n, m, l, GeometryKind::Reference { geometry }, &[][..])
        }
        Geometry::MediaServerLayer(g) => other(common!(g), "MediaServerLayer", &g.children),
        Geometry::MediaServerCamera(g) => other(common!(g), "MediaServerCamera", &g.children),
        Geometry::MediaServerMaster(g) => other(common!(g), "MediaServerMaster", &g.children),
        Geometry::Display(g) => other(common!(g), "Display", &g.children),
        Geometry::Laser(g) => other(common!(g), "Laser", &g.children),
        Geometry::WiringObject(g) => other(common!(g), "WiringObject", &g.children),
        Geometry::Inventory(g) => other(common!(g), "Inventory", &g.children),
        Geometry::Structure(g) => other(common!(g), "Structure", &g.children),
        Geometry::Support(g) => other(common!(g), "Support", &g.children),
        Geometry::Magnet(g) => other(common!(g), "Magnet", &g.children),
    };
    GeometryNode {
        name,
        model,
        local,
        kind,
        children: children.iter().map(convert).collect(),
    }
}

fn other<'a>(
    (n, m, l): (String, Option<String>, Mat4),
    kind: &'static str,
    children: &'a [Geometry],
) -> (String, Option<String>, Mat4, GeometryKind, &'a [Geometry]) {
    (n, m, l, GeometryKind::Other(kind), children)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_round_trips() {
        assert_eq!(matrix_to_mat4(&Matrix::identity()), Mat4::IDENTITY);
    }

    #[test]
    fn translation_is_last_column() {
        // Same shape as the derived Debug output of gdtf's Matrix.
        let m = parse_matrix_debug(
            "Matrix([[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, -1e-2], [0.0, 0.0, 1.0, -0.284], [0.0, 0.0, 0.0, 1.0]])",
        );
        assert_eq!(m.w_axis.truncate(), glam::vec3(0.5, -0.01, -0.284));
        assert_eq!(m.x_axis.truncate(), glam::Vec3::X);
    }
}
