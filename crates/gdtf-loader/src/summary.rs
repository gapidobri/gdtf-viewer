//! Lightweight, owned summaries of wheels and DMX modes (for CLI/UI listings).

use gdtf::dmx_mode::DmxBreak;
use gdtf::fixture_type::FixtureType;
use gdtf::wheel::WheelSlotOptic;

#[derive(Debug, Clone)]
pub struct WheelSummary {
    pub name: String,
    pub slots: Vec<SlotSummary>,
}

#[derive(Debug, Clone)]
pub struct SlotSummary {
    pub name: String,
    /// CIE 1931 xyY, if the slot has a colour.
    pub color: Option<[f32; 3]>,
    pub filter: Option<String>,
    pub media_name: Option<String>,
    pub prism_facets: usize,
}

#[derive(Debug, Clone)]
pub struct ModeSummary {
    pub name: String,
    pub geometry: Option<String>,
    /// Footprint in DMX slots (highest offset used).
    pub footprint: u32,
    pub channels: Vec<ChannelSummary>,
}

#[derive(Debug, Clone)]
pub struct ChannelSummary {
    /// 1-based DMX offsets, coarse first. Empty for virtual channels.
    pub offsets: Vec<i32>,
    pub dmx_break: Option<i32>,
    pub geometry: String,
    pub attributes: Vec<String>,
    pub function_count: usize,
}

pub(crate) fn wheels(ft: &FixtureType) -> Vec<WheelSummary> {
    ft.wheels
        .iter()
        .map(|w| WheelSummary {
            name: w.name.as_ref().map(|n| n.to_string()).unwrap_or_default(),
            slots: w
                .slots
                .iter()
                .map(|s| {
                    let (color, filter) = match &s.optic {
                        WheelSlotOptic::Color(c) => (Some([c.x as f32, c.y as f32, c.z as f32]), None),
                        WheelSlotOptic::Filter(n) => (None, Some(n.to_string())),
                    };
                    SlotSummary {
                        name: s.name.as_ref().map(|n| n.to_string()).unwrap_or_default(),
                        color,
                        filter,
                        media_name: s.media_name.clone().filter(|m| !m.is_empty()),
                        prism_facets: s.facets.len(),
                    }
                })
                .collect(),
        })
        .collect()
}

pub(crate) fn modes(ft: &FixtureType) -> Vec<ModeSummary> {
    ft.dmx_modes
        .iter()
        .map(|m| {
            let channels: Vec<ChannelSummary> = m
                .dmx_channels
                .iter()
                .map(|c| ChannelSummary {
                    offsets: c.offset.clone().unwrap_or_default(),
                    dmx_break: match c.dmx_break {
                        DmxBreak::Value(v) => Some(v),
                        DmxBreak::Overwrite => None,
                    },
                    geometry: c.geometry.to_string(),
                    attributes: c.logical_channels.iter().map(|l| l.attribute.to_string()).collect(),
                    function_count: c.logical_channels.iter().map(|l| l.channel_functions.len()).sum(),
                })
                .collect();
            let footprint = channels
                .iter()
                .flat_map(|c| c.offsets.iter())
                .copied()
                .max()
                .unwrap_or(0)
                .max(0) as u32;
            ModeSummary {
                name: m.name.as_ref().map(|n| n.to_string()).unwrap_or_default(),
                geometry: m.geometry.as_ref().map(|n| n.to_string()),
                footprint,
                channels,
            }
        })
        .collect()
}
