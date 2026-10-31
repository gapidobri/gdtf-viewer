//! Applies evaluated DMX to the fixture: attributes land on the geometry their channel
//! names; each beam takes, per attribute, the value nearest to it up the geometry tree
//! (zone colour overrides head colour), and multiplies all dimmers on the way up
//! (master × zone).

use std::collections::HashMap;

use crate::dmx::{DmxPatch, Evaluated};
use crate::{FixtureRig, Optics, OpticsState};

#[derive(Debug, Clone)]
pub struct AttrValue {
    pub attribute: String,
    pub physical: f32,
    pub slot: Option<f32>,
    pub set_name: String,
}

/// Attribute values per rig node.
#[derive(Debug, Clone, Default)]
pub struct NodeAttributes {
    per_node: HashMap<usize, Vec<AttrValue>>,
}

impl NodeAttributes {
    pub fn from_dmx(patch: &DmxPatch, eval: &[Evaluated], rig: &FixtureRig) -> Self {
        let mut per_node: HashMap<usize, Vec<AttrValue>> = HashMap::new();
        for (c, e) in patch.channels.iter().zip(eval) {
            let Some(fi) = e.function else { continue };
            let f = &c.functions[fi];
            let set_name = e.set.map(|s| f.sets[s].name.clone()).unwrap_or_default();
            // Several rig nodes can share a name only inside instanced templates, which
            // DMX channels don't address directly; take every match.
            for (ni, n) in rig.nodes.iter().enumerate() {
                if n.name == c.geometry {
                    per_node.entry(ni).or_default().push(AttrValue {
                        attribute: f.attribute.clone(),
                        physical: e.physical,
                        slot: e.slot,
                        set_name: set_name.clone(),
                    });
                }
            }
        }
        Self { per_node }
    }

    /// Value of `attribute` set directly on `node`.
    pub fn on_node(&self, node: usize, attribute: &str) -> Option<&AttrValue> {
        self.per_node.get(&node)?.iter().find(|a| a.attribute == attribute)
    }

    /// Nearest value of `attribute` from `node` up to the root.
    pub fn nearest(&self, rig: &FixtureRig, node: usize, attribute: &str) -> Option<&AttrValue> {
        self.ancestors(rig, node).find_map(|n| self.on_node(n, attribute))
    }

    /// All values of `attribute` from `node` up to the root.
    pub fn all_up<'a>(&'a self, rig: &'a FixtureRig, node: usize, attribute: &'a str) -> impl Iterator<Item = &'a AttrValue> {
        self.ancestors(rig, node).filter_map(move |n| self.on_node(n, attribute))
    }

    fn ancestors<'a>(&self, rig: &'a FixtureRig, node: usize) -> impl Iterator<Item = usize> + 'a {
        std::iter::successors(Some(node), |&n| rig.nodes[n].parent)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shutter {
    Open,
    Closed,
    /// Strobing at this rate (Hz).
    Strobe(f32),
}

/// Per-beam result of applying DMX.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamDrive {
    /// Product of all dimmers above the beam, 0..1.
    pub dimmer: f32,
    pub shutter: Shutter,
}

impl BeamDrive {
    /// Output factor at time `t` (seconds). Strobes flash for a quarter of each period,
    /// close to the short flashes of real strobes while staying visible at 60 fps.
    pub fn output(&self, t: f32) -> f32 {
        let gate = match self.shutter {
            Shutter::Open => 1.0,
            Shutter::Closed => 0.0,
            Shutter::Strobe(hz) if hz > 0.0 => {
                if (t * hz).fract() < 0.25 {
                    1.0
                } else {
                    0.0
                }
            }
            Shutter::Strobe(_) => 1.0,
        };
        self.dimmer * gate
    }
}

/// Updates `state` (one beam's optics) from the attributes reaching `beam_node`, and
/// returns its dimmer and shutter. Attributes without a channel keep their current value.
pub fn drive_beam(attrs: &NodeAttributes, rig: &FixtureRig, beam_node: usize, optics: &Optics, state: &mut OpticsState) -> BeamDrive {
    let near = |a: &str| attrs.nearest(rig, beam_node, a);

    for (e, level) in optics.emitters.iter().zip(&mut state.emitter_levels) {
        if let Some(v) = near(&e.attr) {
            *level = v.physical.clamp(0.0, 1.0);
        }
    }
    for (f, level) in optics.cmy.iter().zip(&mut state.cmy_levels) {
        if let Some(v) = near(&f.attr) {
            *level = v.physical.clamp(0.0, 1.0);
        }
    }
    if let (Some(v), Some(k)) = (near("CTO").or_else(|| near("CTC")), state.cto_k.as_mut()) {
        // Files use kelvin; for 0..1 functions run from native to 3200 K like the slider.
        *k = if v.physical > 100.0 {
            v.physical
        } else {
            optics.native_cct + (3200.0 - optics.native_cct) * v.physical.clamp(0.0, 1.0)
        };
    }
    if let (Some(v), Some(z)) = (near("Zoom"), state.zoom_deg.as_mut()) {
        *z = v.physical;
    }
    if let Some(v) = near("Focus1") {
        state.focus = v.physical.clamp(0.0, 1.0);
    }
    for (w, ws) in optics.wheels.iter().zip(&mut state.wheels) {
        let spin = near(&format!("{}WheelSpin", w.attr));
        ws.wheel_spin_dps = spin.map_or(0.0, |v| v.physical);
        if spin.is_none()
            && let Some(slot) = near(&w.attr).and_then(|v| v.slot)
        {
            ws.slot = slot;
        }
        match (near(&format!("{}Pos", w.attr)), near(&format!("{}PosRotate", w.attr))) {
            (Some(index), _) => {
                ws.index_deg = index.physical;
                ws.spin_dps = 0.0;
                ws.spin_angle = 0.0;
            }
            (None, Some(rot)) => {
                ws.index_deg = 0.0;
                ws.spin_dps = rot.physical;
            }
            (None, None) => {}
        }
    }

    let dimmer = attrs
        .all_up(rig, beam_node, "Dimmer")
        .map(|v| v.physical.clamp(0.0, 1.0))
        .fold(1.0, |a, b| a * b);
    let shutter = shutter_state(attrs, rig, beam_node);
    BeamDrive { dimmer, shutter }
}

fn shutter_state(attrs: &NodeAttributes, rig: &FixtureRig, node: usize) -> Shutter {
    // Strobe functions share the Shutter1 channel; whichever is active is present.
    for a in ["Shutter1Strobe", "Shutter1StrobePulse", "Shutter1StrobePulseOpen", "Shutter1StrobePulseClose", "Shutter1StrobeRandom"] {
        if let Some(v) = attrs.nearest(rig, node, a) {
            return Shutter::Strobe(v.physical.max(0.0));
        }
    }
    match attrs.nearest(rig, node, "Shutter1") {
        // Open/closed is carried by the channel set name more reliably than the number.
        Some(v) => {
            let name = v.set_name.to_lowercase();
            if name.contains("closed") && !name.contains("open") {
                Shutter::Closed
            } else if name.contains("open") || v.physical >= 0.5 {
                Shutter::Open
            } else {
                Shutter::Closed
            }
        }
        None => Shutter::Open,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strobe_flashes_a_quarter_of_the_time() {
        let d = BeamDrive {
            dimmer: 0.8,
            shutter: Shutter::Strobe(10.0),
        };
        let on = (0..1000).filter(|i| d.output(*i as f32 / 1000.0) > 0.0).count();
        assert!((240..=260).contains(&on), "{on}");
        assert_eq!(d.output(0.0), 0.8);
    }
}
