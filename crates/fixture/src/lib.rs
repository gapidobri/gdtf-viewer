//! Runtime fixture model: a flattened geometry tree with posable axes.
//!
//! No GPU code here, so it stays unit-testable. DMX → attribute evaluation will live
//! in this crate too (milestone 7); for now the pose is set directly.

mod axes;
mod beam;
pub mod control;
pub mod dmx;
mod optics;

pub use axes::{axis_controls, AxisAttr, AxisControl, AxisMotion};
pub use optics::{
    focus_distance, focus_for_distance, BeamLook, ColorSplit, EmitterUse, FilterUse, GoboLook, Optics, OpticsState, Range,
    SlotOptics, WheelKind, WheelState, WheelUse,
};
pub use beam::{beams, cct_to_linear_srgb, xyy_to_linear_srgb, BeamProfile, BeamState};

use gdtf_loader::{BeamInfo, GeometryKind, GeometryNode, LoadedFixture};
use glam::{Mat4, Vec3};

#[derive(Debug, Clone)]
pub enum NodeKind {
    Static,
    Axis,
    Beam(BeamInfo),
    /// Filters, references, wiring etc. Drawn if they have a model, otherwise ignored.
    Other,
}

#[derive(Debug, Clone)]
pub struct RigNode {
    pub name: String,
    pub parent: Option<usize>,
    /// Transform relative to the parent in GDTF space (Z-up).
    pub local: Mat4,
    pub model: Option<String>,
    pub kind: NodeKind,
}

/// The fixture's geometry, flattened in depth-first order (parents before children).
#[derive(Debug, Clone, Default)]
pub struct FixtureRig {
    pub nodes: Vec<RigNode>,
}

/// Rotation of each node, in degrees, about the axis the attribute drives.
/// Indexed like [`FixtureRig::nodes`]; only `Axis` nodes are affected.
#[derive(Debug, Clone, Default)]
pub struct Pose {
    pub rotations: Vec<AxisRotation>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AxisRotation {
    /// Pan: rotation about local Z.
    pub pan_deg: f32,
    /// Tilt: rotation about local X.
    pub tilt_deg: f32,
}

impl FixtureRig {
    /// Builds the rig from the top-level geometry named by `root` (a DMX mode's
    /// `Geometry`), falling back to the first top-level geometry. Other top-level
    /// geometries are templates for GeometryReference and are not drawn on their own;
    /// each reference is expanded into a copy of its template (multi-cell fixtures).
    pub fn from_loaded(fx: &LoadedFixture, root: Option<&str>) -> Self {
        let mut rig = Self::default();
        let g = root
            .and_then(|r| fx.geometry.iter().find(|g| g.name == r))
            .or_else(|| fx.geometry.first());
        if let Some(g) = g {
            rig.push(g, None, &fx.geometry, 0);
        }
        rig
    }

    fn push(&mut self, g: &GeometryNode, parent: Option<usize>, templates: &[GeometryNode], depth: usize) {
        if let GeometryKind::Reference { geometry } = &g.kind {
            // The reference places an instance of a top-level geometry. Keep the reference
            // node (DMX channels and breaks address it by name) without a model, and hang
            // the template's subtree under it; the template's own position is ignored.
            let idx = self.nodes.len();
            self.nodes.push(RigNode {
                name: g.name.clone(),
                parent,
                local: g.local,
                model: None,
                kind: NodeKind::Other,
            });
            let template = geometry.as_deref().and_then(|n| templates.iter().find(|t| t.name == n));
            match template {
                Some(t) if depth < 8 => {
                    let mut inst = t.clone();
                    inst.local = Mat4::IDENTITY;
                    self.push(&inst, Some(idx), templates, depth + 1);
                }
                Some(_) => log::warn!("GeometryReference '{}' nests too deep", g.name),
                None => log::warn!("GeometryReference '{}' points to unknown geometry {geometry:?}", g.name),
            }
            return;
        }
        let kind = match &g.kind {
            GeometryKind::Static => NodeKind::Static,
            GeometryKind::Axis => NodeKind::Axis,
            GeometryKind::Beam(b) => NodeKind::Beam(b.clone()),
            _ => NodeKind::Other,
        };
        let idx = self.nodes.len();
        self.nodes.push(RigNode {
            name: g.name.clone(),
            parent,
            local: g.local,
            model: g.model.clone(),
            kind,
        });
        for c in &g.children {
            self.push(c, Some(idx), templates, depth);
        }
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n.name == name)
    }

    pub fn rest_pose(&self) -> Pose {
        Pose {
            rotations: vec![AxisRotation::default(); self.nodes.len()],
        }
    }

    /// World transforms (GDTF space, relative to the fixture's base) for every node.
    pub fn world_transforms(&self, pose: &Pose) -> Vec<Mat4> {
        let mut out: Vec<Mat4> = Vec::with_capacity(self.nodes.len());
        for (i, n) in self.nodes.iter().enumerate() {
            let mut local = n.local;
            // Normally only Axis geometries move, but some files put Pan/Tilt channels on
            // plain geometries; apply whatever rotation the pose holds.
            if let Some(r) = pose.rotations.get(i).filter(|r| r.pan_deg != 0.0 || r.tilt_deg != 0.0) {
                local *= Mat4::from_rotation_z(r.pan_deg.to_radians())
                    * Mat4::from_rotation_x(r.tilt_deg.to_radians());
            }
            let parent = n.parent.map_or(Mat4::IDENTITY, |p| out[p]);
            out.push(parent * local);
        }
        out
    }
}

/// GDTF (right-handed, Z-up) → renderer (right-handed, Y-up): (x, y, z) → (x, z, −y).
pub const GDTF_TO_Y_UP: Mat4 = Mat4::from_cols(
    glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 0.0, -1.0, 0.0),
    glam::Vec4::new(0.0, 1.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
);

/// Beam emission direction in a Beam geometry's local space.
pub const BEAM_DIR_LOCAL: Vec3 = Vec3::NEG_Z;

#[cfg(test)]
mod tests {
    use super::*;

    fn node(name: &str, parent: Option<usize>, z: f32, kind: NodeKind) -> RigNode {
        RigNode {
            name: name.into(),
            parent,
            local: Mat4::from_translation(Vec3::new(0.0, 0.0, z)),
            model: None,
            kind,
        }
    }

    #[test]
    fn tilt_swings_beam_about_head_pivot() {
        let rig = FixtureRig {
            nodes: vec![
                node("Base", None, 0.0, NodeKind::Static),
                node("Yoke", Some(0), -0.1, NodeKind::Axis),
                node("Head", Some(1), -0.3, NodeKind::Axis),
                node("Beam", Some(2), -0.2, NodeKind::Other),
            ],
        };
        let mut pose = rig.rest_pose();
        let rest = rig.world_transforms(&pose);
        assert!((rest[3].w_axis.truncate() - Vec3::new(0.0, 0.0, -0.6)).length() < 1e-6);

        pose.rotations[2].tilt_deg = 90.0;
        let w = rig.world_transforms(&pose);
        // Head pivot stays put; beam swings 0.2 m around X: −Z → +Y.
        assert!((w[2].w_axis.truncate() - Vec3::new(0.0, 0.0, -0.4)).length() < 1e-6);
        assert!((w[3].w_axis.truncate() - Vec3::new(0.0, 0.2, -0.4)).length() < 1e-6);
    }

    #[test]
    fn z_up_maps_to_y_up() {
        assert_eq!(GDTF_TO_Y_UP.transform_vector3(Vec3::Z), Vec3::Y);
        assert_eq!(GDTF_TO_Y_UP.transform_vector3(Vec3::Y), Vec3::NEG_Z);
        // Still a proper rotation (no mirroring).
        assert!((GDTF_TO_Y_UP.determinant() - 1.0).abs() < 1e-6);
    }
}
