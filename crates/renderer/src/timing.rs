//! GPU pass timing with timestamp queries, read back without stalling: results arrive a
//! frame or two late, and frames are skipped while a readback is in flight.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub const PASSES: [&str; 5] = ["shadows", "forward", "volume", "bloom", "tonemap"];

pub struct GpuTimer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// A copy is pending or mapped; don't write new timestamps.
    in_flight: bool,
    ready: Arc<AtomicBool>,
    period_ns: f32,
    /// Latest per-pass durations, milliseconds.
    pub last: [f32; PASSES.len()],
    /// Whether this frame records timestamps.
    active: bool,
}

const COUNT: u32 = 2 * PASSES.len() as u32;

impl GpuTimer {
    /// `None` if the device lacks `Features::TIMESTAMP_QUERY`.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let size = 8 * COUNT as u64;
        Some(Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("pass timing"),
                ty: wgpu::QueryType::Timestamp,
                count: COUNT,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timing resolve"),
                size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timing readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            in_flight: false,
            ready: Arc::new(AtomicBool::new(false)),
            period_ns: queue.get_timestamp_period(),
            last: [0.0; PASSES.len()],
            active: false,
        })
    }

    /// Call at the start of a frame: collects finished results and decides whether this
    /// frame records.
    pub fn begin_frame(&mut self) {
        if self.in_flight && self.ready.swap(false, Ordering::AcqRel) {
            if let Ok(view) = self.readback.get_mapped_range(..) {
                let ticks: &[u64] = bytemuck::cast_slice(&view);
                for (i, out) in self.last.iter_mut().enumerate() {
                    let (a, b) = (ticks[2 * i], ticks[2 * i + 1]);
                    *out = b.saturating_sub(a) as f32 * self.period_ns / 1e6;
                }
            }
            self.readback.unmap();
            self.in_flight = false;
        }
        self.active = !self.in_flight;
    }

    /// Timestamp writes for pass `pass`; `first`/`last` let several render passes (one
    /// per shadow map) share one entry.
    pub fn writes(&self, pass: usize, first: bool, last: bool) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        // Passes in the middle of a group record nothing (wgpu requires at least one index).
        if !self.active || !(first || last) {
            return None;
        }
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: first.then_some(2 * pass as u32),
            end_of_pass_write_index: last.then_some(2 * pass as u32 + 1),
        })
    }

    /// Records the resolve + copy into `enc` (call after all passes).
    pub fn resolve(&self, enc: &mut wgpu::CommandEncoder) {
        if self.active {
            enc.resolve_query_set(&self.queries, 0..COUNT, &self.resolve, 0);
            enc.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, 8 * COUNT as u64);
        }
    }

    /// Call after submitting the frame's commands.
    pub fn after_submit(&mut self) {
        if !self.active {
            return;
        }
        self.in_flight = true;
        let ready = self.ready.clone();
        self.readback.map_async(wgpu::MapMode::Read, .., move |r| {
            if r.is_ok() {
                ready.store(true, Ordering::Release);
            }
        });
    }
}
