//! DMX → physical attribute values for one DMX mode.
//!
//! A mode has DMX channels (1–4 bytes, coarse first), each with logical channels that hold
//! channel functions. A function covers a DMX range starting at its `DMXFrom` and maps it
//! linearly onto `PhysicalFrom..PhysicalTo`. Some functions only apply while another
//! channel (their mode master) is in a given range. Channel sets name sub-ranges and, for
//! wheels, select slots.

use gdtf_loader::gdtf::dmx_mode::{ChannelFunction, DmxMode, ModeMaster};

/// A DMX mode flattened for evaluation.
#[derive(Debug, Clone, Default)]
pub struct DmxPatch {
    pub channels: Vec<Channel>,
    /// Number of DMX slots the mode uses.
    pub footprint: usize,
}

#[derive(Debug, Clone)]
pub struct Channel {
    pub name: String,
    pub geometry: String,
    /// 0-based slot offsets within the fixture, coarse first. Empty for virtual channels
    /// (no DMX slots; set from the UI only).
    pub offsets: Vec<usize>,
    pub bytes: u8,
    /// Initial value (from the initial function's default).
    pub default: u64,
    pub functions: Vec<Function>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub attribute: String,
    /// In the channel's resolution.
    pub dmx_from: u64,
    pub physical_from: f32,
    pub physical_to: f32,
    pub wheel: Option<String>,
    pub master: Option<Master>,
    pub sets: Vec<Set>,
}

/// Mode master: the function applies only while `channel` (and, if given, that channel's
/// `function`) is active with a value in `from..=to` (in the master channel's resolution).
#[derive(Debug, Clone, Copy)]
pub struct Master {
    pub channel: usize,
    pub function: Option<usize>,
    pub from: u64,
    pub to: u64,
}

#[derive(Debug, Clone)]
pub struct Set {
    pub name: String,
    pub dmx_from: u64,
    /// 0-based wheel slot.
    pub slot: Option<u32>,
}

/// Result of evaluating one channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Evaluated {
    pub value: u64,
    pub function: Option<usize>,
    pub set: Option<usize>,
    pub physical: f32,
    /// Continuous wheel position (0-based slot) for wheel functions with slot sets.
    pub slot: Option<f32>,
}

/// DMX values for one fixture: `bytes` covers the footprint; virtual channels keep their
/// value separately.
#[derive(Debug, Clone)]
pub struct DmxValues {
    pub bytes: Vec<u8>,
    pub virtual_values: Vec<u64>,
}

impl Channel {
    pub fn max(&self) -> u64 {
        (1u64 << (8 * self.bytes as u32)) - 1
    }

    pub fn is_virtual(&self) -> bool {
        self.offsets.is_empty()
    }
}

impl DmxPatch {
    pub fn from_mode(mode: &DmxMode) -> Self {
        let channel_index = |ch: &gdtf_loader::gdtf::dmx_mode::DmxChannel| {
            mode.dmx_channels.iter().position(|c| std::ptr::eq(c, ch))
        };
        let mut channels = Vec::new();
        let mut footprint = 0;
        for ch in &mode.dmx_channels {
            let offsets: Vec<usize> = ch
                .offset
                .clone()
                .unwrap_or_default()
                .into_iter()
                .filter(|&o| o >= 1)
                .map(|o| o as usize - 1)
                .collect();
            footprint = footprint.max(offsets.iter().map(|o| o + 1).max().unwrap_or(0));
            let bytes = offsets.len().clamp(1, 4) as u8;
            let to_res = |v: &gdtf_loader::gdtf::values::DmxValue| v.to(bytes);

            let mut functions = Vec::new();
            for lc in &ch.logical_channels {
                for f in &lc.channel_functions {
                    let master = f.mode_master.as_ref().and_then(|m| {
                        let resolved = m.mode_master(mode)?;
                        let (mch, mfunc) = match resolved {
                            ModeMaster::DmxChannel(c) => (c, None),
                            ModeMaster::ChannelFunction(c, _, func) => (c, Some(func)),
                        };
                        let mi = channel_index(mch)?;
                        let mbytes = mch.offset.as_ref().map_or(1, |o| o.len().clamp(1, 4)) as u8;
                        let mfi = mfunc.and_then(|mf| {
                            mch.logical_channels
                                .iter()
                                .flat_map(|l| &l.channel_functions)
                                .position(|x| std::ptr::eq(x, mf))
                        });
                        Some(Master {
                            channel: mi,
                            function: mfi,
                            from: m.from.to(mbytes),
                            to: m.to.to(mbytes),
                        })
                    });
                    functions.push(function_from(f, lc.attribute.to_string(), bytes, master, &to_res));
                }
            }
            let default = ch
                .initial_function()
                .map(|(_, f)| f.default.to(bytes))
                .or_else(|| {
                    ch.logical_channels
                        .first()
                        .and_then(|l| l.channel_functions.first())
                        .map(|f| f.default.to(bytes))
                })
                .unwrap_or(0);
            channels.push(Channel {
                name: ch.name().to_string(),
                geometry: ch.geometry.to_string(),
                offsets,
                bytes,
                default,
                functions,
            });
        }
        Self { channels, footprint }
    }

    pub fn default_values(&self) -> DmxValues {
        let mut v = DmxValues {
            bytes: vec![0; self.footprint],
            virtual_values: vec![0; self.channels.len()],
        };
        for (i, c) in self.channels.iter().enumerate() {
            self.set(&mut v, i, c.default);
        }
        v
    }

    /// Reads channel `i`'s value at its resolution.
    pub fn get(&self, values: &DmxValues, i: usize) -> u64 {
        let c = &self.channels[i];
        if c.is_virtual() {
            return values.virtual_values[i];
        }
        c.offsets
            .iter()
            .fold(0u64, |acc, &o| (acc << 8) | *values.bytes.get(o).unwrap_or(&0) as u64)
    }

    pub fn set(&self, values: &mut DmxValues, i: usize, value: u64) {
        let c = &self.channels[i];
        let value = value.min(c.max());
        if c.is_virtual() {
            values.virtual_values[i] = value;
            return;
        }
        let n = c.offsets.len();
        for (k, &o) in c.offsets.iter().enumerate() {
            if let Some(b) = values.bytes.get_mut(o) {
                *b = (value >> (8 * (n - 1 - k))) as u8;
            }
        }
    }

    /// Evaluates every channel.
    pub fn evaluate(&self, values: &DmxValues) -> Vec<Evaluated> {
        let mut memo: Vec<Option<Option<usize>>> = vec![None; self.channels.len()];
        (0..self.channels.len())
            .map(|i| {
                let value = self.get(values, i);
                let function = self.active_function(values, i, &mut memo, 0);
                let (physical, set, slot) = match function {
                    Some(fi) => self.resolve(i, fi, value, values, &mut memo),
                    None => (0.0, None, None),
                };
                Evaluated {
                    value,
                    function,
                    set,
                    physical,
                    slot,
                }
            })
            .collect()
    }

    /// Functions of channel `i` whose mode master (if any) is satisfied, sorted by DMXFrom.
    fn candidates(&self, values: &DmxValues, i: usize, memo: &mut Vec<Option<Option<usize>>>, depth: usize) -> Vec<usize> {
        let c = &self.channels[i];
        let mut v: Vec<usize> = (0..c.functions.len())
            .filter(|&fi| match c.functions[fi].master {
                None => true,
                Some(m) => {
                    if depth > 8 || m.channel == i {
                        return false;
                    }
                    let mv = self.get(values, m.channel);
                    let in_range = (m.from..=m.to).contains(&mv);
                    let func_ok = m
                        .function
                        .is_none_or(|mf| self.active_function(values, m.channel, memo, depth + 1) == Some(mf));
                    in_range && func_ok
                }
            })
            .collect();
        v.sort_by_key(|&fi| c.functions[fi].dmx_from);
        v
    }

    fn active_function(&self, values: &DmxValues, i: usize, memo: &mut Vec<Option<Option<usize>>>, depth: usize) -> Option<usize> {
        if let Some(r) = memo[i] {
            return r;
        }
        let value = self.get(values, i);
        let cands = self.candidates(values, i, memo, depth);
        let f = cands
            .iter()
            .copied()
            .rfind(|&fi| self.channels[i].functions[fi].dmx_from <= value)
            .or_else(|| cands.first().copied());
        memo[i] = Some(f);
        f
    }

    /// Physical value, channel set and wheel slot for function `fi` at `value`.
    fn resolve(
        &self,
        i: usize,
        fi: usize,
        value: u64,
        values: &DmxValues,
        memo: &mut Vec<Option<Option<usize>>>,
    ) -> (f32, Option<usize>, Option<f32>) {
        let c = &self.channels[i];
        let f = &c.functions[fi];
        // The function's range ends where the next applicable function starts.
        let cands = self.candidates(values, i, memo, 0);
        let end = cands
            .iter()
            .map(|&k| c.functions[k].dmx_from)
            .filter(|&from| from > f.dmx_from)
            .min()
            .map_or(c.max(), |next| next - 1);
        let t = if end > f.dmx_from {
            (value.saturating_sub(f.dmx_from)) as f64 / (end - f.dmx_from) as f64
        } else {
            0.0
        };
        let physical = f.physical_from + (f.physical_to - f.physical_from) * t.clamp(0.0, 1.0) as f32;

        // Channel set: the last one starting at or below the value.
        let set = f.sets.iter().rposition(|s| s.dmx_from <= value);
        let slot = set.and_then(|si| {
            if let Some(s) = f.sets[si].slot {
                return Some(s as f32);
            }
            // A set without a slot between two slot sets (e.g. a split colour): position
            // between them in proportion to the value.
            let prev = f.sets[..si].iter().rev().find(|s| s.slot.is_some())?;
            let next = f.sets[si + 1..].iter().find(|s| s.slot.is_some())?;
            let span = next.dmx_from.saturating_sub(prev.dmx_from).max(1) as f32;
            let frac = (value.saturating_sub(prev.dmx_from)) as f32 / span;
            let (a, b) = (prev.slot? as f32, next.slot? as f32);
            Some(a + (b - a) * frac.clamp(0.0, 1.0))
        });
        (physical, set, slot)
    }
}

fn function_from(
    f: &ChannelFunction,
    attribute: String,
    _bytes: u8,
    master: Option<Master>,
    to_res: &dyn Fn(&gdtf_loader::gdtf::values::DmxValue) -> u64,
) -> Function {
    let mut sets: Vec<Set> = f
        .channel_sets
        .iter()
        .map(|s| Set {
            name: s.name.as_deref().unwrap_or_default().to_owned(),
            dmx_from: to_res(&s.dmx_from),
            slot: s.wheel_slot_index.filter(|&w| w >= 1).map(|w| w as u32 - 1),
        })
        .collect();
    sets.sort_by_key(|s| s.dmx_from);
    // The function's own attribute is per channel function (a logical channel can hold
    // functions of related attributes, e.g. Gobo2 + Gobo2SelectShake).
    let fattr = f.attribute.to_string();
    Function {
        name: f.name.as_deref().unwrap_or_default().to_owned(),
        attribute: if fattr.is_empty() { attribute } else { fattr },
        dmx_from: to_res(&f.dmx_from),
        physical_from: f.physical_from as f32,
        physical_to: f.physical_to as f32,
        wheel: f.wheel.as_ref().map(|w| w.to_string()),
        master,
        sets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn func(name: &str, attr: &str, from: u64, phys: (f32, f32), master: Option<Master>) -> Function {
        Function {
            name: name.into(),
            attribute: attr.into(),
            dmx_from: from,
            physical_from: phys.0,
            physical_to: phys.1,
            wheel: None,
            master,
            sets: vec![],
        }
    }

    fn channel(name: &str, offsets: Vec<usize>, functions: Vec<Function>) -> Channel {
        let bytes = offsets.len().max(1) as u8;
        Channel {
            name: name.into(),
            geometry: "Head".into(),
            offsets,
            bytes,
            default: 0,
            functions,
        }
    }

    /// Like the MegaPointe: channel 0 selects index (0–31) or rotate (32–59) mode for the
    /// 16-bit channel 1.
    fn megapointe_like() -> DmxPatch {
        let mut mode = func("Gobo2Index", "Gobo2", 0, (0.0, 1.0), None);
        mode.wheel = Some("RotatingGoboWheel".into());
        mode.sets = vec![
            Set { name: "Open".into(), dmx_from: 0, slot: Some(0) },
            Set { name: "Gobo 1".into(), dmx_from: 5, slot: Some(1) },
            Set { name: "Gobo 2".into(), dmx_from: 8, slot: Some(2) },
        ];
        let rot = func("Gobo2Rotation", "Gobo2", 32, (0.0, 1.0), None);
        let m = |from, to| Some(Master { channel: 0, function: None, from, to });
        DmxPatch {
            channels: vec![
                channel("Head_Gobo2", vec![0], vec![mode, rot]),
                channel(
                    "Head_Gobo2Pos",
                    vec![1, 2],
                    vec![
                        func("Gobo 2 Index", "Gobo2Pos", 0, (180.0, -180.0), m(0, 31)),
                        func("Gobo 2 Rotate", "Gobo2PosRotate", 0, (-900.0, 900.0), m(32, 59)),
                    ],
                ),
            ],
            footprint: 3,
        }
    }

    #[test]
    fn coarse_fine_round_trip() {
        let p = megapointe_like();
        let mut v = p.default_values();
        p.set(&mut v, 1, 0xABCD);
        assert_eq!(v.bytes, vec![0, 0xAB, 0xCD]);
        assert_eq!(p.get(&v, 1), 0xABCD);
    }

    #[test]
    fn mode_master_switches_function() {
        let p = megapointe_like();
        let mut v = p.default_values();
        p.set(&mut v, 1, 32768);
        p.set(&mut v, 0, 10); // index mode
        let e = p.evaluate(&v);
        assert_eq!(p.channels[1].functions[e[1].function.unwrap()].attribute, "Gobo2Pos");
        assert!(e[1].physical.abs() < 0.01, "{}", e[1].physical);
        p.set(&mut v, 0, 40); // rotate mode
        let e = p.evaluate(&v);
        assert_eq!(p.channels[1].functions[e[1].function.unwrap()].attribute, "Gobo2PosRotate");
        assert!(e[1].physical.abs() < 0.03, "{}", e[1].physical);
        p.set(&mut v, 1, 65535);
        assert!((p.evaluate(&v)[1].physical - 900.0).abs() < 1e-3);
    }

    #[test]
    fn function_range_ends_at_next_function() {
        let p = megapointe_like();
        let mut v = p.default_values();
        p.set(&mut v, 0, 31);
        let e = p.evaluate(&v)[0];
        assert_eq!(e.function, Some(0));
        assert!((e.physical - 1.0).abs() < 1e-6); // 31 is the end of 0..=31
        p.set(&mut v, 0, 32);
        assert_eq!(p.evaluate(&v)[0].function, Some(1));
    }

    #[test]
    fn channel_sets_select_wheel_slots() {
        let p = megapointe_like();
        let mut v = p.default_values();
        p.set(&mut v, 0, 9);
        let e = p.evaluate(&v)[0];
        assert_eq!(e.set, Some(2));
        assert_eq!(e.slot, Some(2.0));
    }
}
