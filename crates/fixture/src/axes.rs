//! Pan/Tilt: which Axis geometry each attribute drives, its physical range from the DMX
//! mode, and a simple model of how fast the real motors get there.

use gdtf_loader::gdtf::dmx_mode::{ChannelFunction, DmxMode, LogicalChannel};

use crate::{FixtureRig, NodeKind, Pose};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisAttr {
    /// Rotation about the geometry's local Z.
    Pan,
    /// Rotation about the geometry's local X.
    Tilt,
}

impl AxisAttr {
    fn from_attribute(name: &str) -> Option<Self> {
        match name {
            "Pan" => Some(Self::Pan),
            "Tilt" => Some(Self::Tilt),
            _ => None,
        }
    }
}

/// One posable axis exposed by the selected DMX mode.
#[derive(Debug, Clone)]
pub struct AxisControl {
    pub attr: AxisAttr,
    /// Index into [`FixtureRig::nodes`].
    pub node: usize,
    pub geometry: String,
    /// DMX resolution of the channel in bytes (1 = 8-bit, 2 = 16-bit, ...).
    pub bytes: u8,
    /// Physical value (degrees) at the lowest and highest DMX value. May be inverted
    /// (from > to), as on the MegaPointe's tilt.
    pub physical_from: f32,
    pub physical_to: f32,
    /// Physical value at the channel's default DMX value.
    pub default_deg: f32,
    /// Seconds to travel the full range (`RealFade`), if the file says.
    pub full_range_time: Option<f32>,
    /// Seconds to reach full speed from rest (`RealAcceleration`), if the file says.
    pub accel_time: Option<f32>,
}

impl AxisControl {
    pub fn min_deg(&self) -> f32 {
        self.physical_from.min(self.physical_to)
    }

    pub fn max_deg(&self) -> f32 {
        self.physical_from.max(self.physical_to)
    }

    /// Maps a physical angle back to a normalised DMX fraction 0..1.
    pub fn dmx_fraction(&self, deg: f32) -> f32 {
        let span = self.physical_to - self.physical_from;
        if span.abs() < f32::EPSILON {
            0.0
        } else {
            ((deg - self.physical_from) / span).clamp(0.0, 1.0)
        }
    }

    /// DMX value for a physical angle at the channel's resolution.
    pub fn dmx_value(&self, deg: f32) -> u64 {
        let max = (1u64 << (8 * self.bytes as u32)) - 1;
        (self.dmx_fraction(deg) as f64 * max as f64).round() as u64
    }

    /// Maximum angular speed in °/s (infinite if the file gives no timing).
    pub fn max_speed(&self) -> f32 {
        match self.full_range_time {
            Some(t) if t > 0.0 => (self.max_deg() - self.min_deg()) / t,
            _ => f32::INFINITY,
        }
    }

    /// Angular acceleration in °/s² (infinite if the file gives no timing).
    pub fn max_accel(&self) -> f32 {
        match self.accel_time {
            Some(t) if t > 0.0 && self.max_speed().is_finite() => self.max_speed() / t,
            _ => f32::INFINITY,
        }
    }

    pub fn apply(&self, pose: &mut Pose, deg: f32) {
        let r = &mut pose.rotations[self.node];
        match self.attr {
            AxisAttr::Pan => r.pan_deg = deg,
            AxisAttr::Tilt => r.tilt_deg = deg,
        }
    }
}

/// Finds the Pan/Tilt channels of a DMX mode and maps them onto rig nodes.
pub fn axis_controls(rig: &FixtureRig, mode: &DmxMode) -> Vec<AxisControl> {
    let mut out = Vec::new();
    for ch in &mode.dmx_channels {
        let geometry = ch.geometry.to_string();
        let Some(node) = rig.find(&geometry) else {
            log::warn!("DMX channel on unknown geometry '{geometry}'");
            continue;
        };
        // Pan/Tilt should drive Axis geometries; accept others too (some files put them
        // on a plain Geometry), but never beams.
        if matches!(rig.nodes[node].kind, NodeKind::Beam(_)) {
            continue;
        }
        let bytes = ch.offset.as_ref().map_or(1, |o| o.len().max(1)) as u8;
        for lc in &ch.logical_channels {
            let Some(attr) = AxisAttr::from_attribute(&lc.attribute.to_string()) else {
                continue;
            };
            if out.iter().any(|c: &AxisControl| c.node == node && c.attr == attr) {
                continue;
            }
            if let Some(c) = control_from_logical(lc, attr, node, &geometry, bytes) {
                out.push(c);
            }
        }
    }
    out
}

fn control_from_logical(
    lc: &LogicalChannel,
    attr: AxisAttr,
    node: usize,
    geometry: &str,
    bytes: u8,
) -> Option<AxisControl> {
    // A logical channel can hold several functions (e.g. Pan and a continuous PanRotate);
    // use the plain angular one.
    let attr_name = lc.attribute.to_string();
    let funcs: Vec<&ChannelFunction> = lc
        .channel_functions
        .iter()
        .filter(|f| f.attribute.to_string() == attr_name)
        .collect();
    let f = funcs.first()?;
    let (from, to) = (f.physical_from as f32, f.physical_to as f32);

    // Default physical value: linear map of the default DMX value within this function's
    // DMX range (from its DMXFrom to the next function's DMXFrom, or the channel max).
    let max = ((1u128 << (8 * bytes as u32)) - 1) as f64;
    let start = f.dmx_from.to(bytes) as f64;
    let end = lc
        .channel_functions
        .iter()
        .map(|g| g.dmx_from.to(bytes) as f64)
        .filter(|&v| v > start)
        .fold(max + 1.0, f64::min)
        - 1.0;
    let t = if end > start {
        ((f.default.to(bytes) as f64 - start) / (end - start)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let default_deg = from + (to - from) * t as f32;

    let positive = |v: f64| (v > 0.0).then_some(v as f32);
    Some(AxisControl {
        attr,
        node,
        geometry: geometry.to_owned(),
        bytes,
        physical_from: from,
        physical_to: to,
        default_deg,
        full_range_time: positive(f.real_fade),
        accel_time: positive(f.real_acceleration),
    })
}

/// Motor model: moves towards a target with limited speed and acceleration
/// (trapezoidal velocity profile), like a real moving head following DMX.
#[derive(Debug, Clone, Copy, Default)]
pub struct AxisMotion {
    pub position: f32,
    pub velocity: f32,
}

impl AxisMotion {
    pub fn at(position: f32) -> Self {
        Self {
            position,
            velocity: 0.0,
        }
    }

    pub fn step(&mut self, target: f32, dt: f32, max_speed: f32, max_accel: f32) {
        if dt <= 0.0 {
            return;
        }
        if !max_speed.is_finite() {
            *self = Self::at(target);
            return;
        }
        let err = target - self.position;
        // Highest speed from which we can still stop at the target.
        let stop_speed = if max_accel.is_finite() {
            (2.0 * max_accel * err.abs()).sqrt()
        } else {
            f32::INFINITY
        };
        let desired = err.signum() * max_speed.min(stop_speed).min(err.abs() / dt);
        let dv = desired - self.velocity;
        let max_dv = max_accel * dt;
        self.velocity += dv.clamp(-max_dv, max_dv);
        self.position += self.velocity * dt;
        // Arrived: within what one braking step covers (or crossed the target).
        let rem = target - self.position;
        let settle = (self.velocity.abs() * dt).max(max_accel.min(1e4) * dt * dt);
        if rem.abs() <= settle || rem.signum() != err.signum() {
            *self = Self::at(target);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn control(from: f32, to: f32, fade: f32, accel: f32) -> AxisControl {
        AxisControl {
            attr: AxisAttr::Tilt,
            node: 0,
            geometry: "Head".into(),
            bytes: 2,
            physical_from: from,
            physical_to: to,
            default_deg: 0.0,
            full_range_time: Some(fade),
            accel_time: Some(accel),
        }
    }

    #[test]
    fn inverted_range() {
        let c = control(132.5, -132.5, 1.767, 0.549);
        assert_eq!((c.min_deg(), c.max_deg()), (-132.5, 132.5));
        assert_eq!(c.dmx_fraction(132.5), 0.0);
        assert_eq!(c.dmx_fraction(-132.5), 1.0);
        assert!((c.dmx_fraction(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn motion_reaches_target_in_about_real_fade() {
        // MegaPointe pan: 540° in 3.134 s, 0.549 s to full speed.
        let c = control(-270.0, 270.0, 3.134, 0.549);
        let mut m = AxisMotion::at(-270.0);
        let dt = 1.0 / 120.0;
        let mut t = 0.0;
        while m.position != 270.0 && t < 10.0 {
            m.step(270.0, dt, c.max_speed(), c.max_accel());
            assert!(m.velocity <= c.max_speed() + 1e-3);
            t += dt;
        }
        // Full-range move at max speed plus one acceleration time (accelerate + brake).
        assert!((t - (3.134 + 0.549)).abs() < 0.1, "took {t}s");
    }

    #[test]
    fn motion_without_timing_snaps() {
        let mut m = AxisMotion::at(0.0);
        m.step(42.0, 0.016, f32::INFINITY, f32::INFINITY);
        assert_eq!(m.position, 42.0);
    }
}
