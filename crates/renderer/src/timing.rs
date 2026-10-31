//! GPU pass timing with timestamp queries, read back without stalling. A small ring of
//! readback buffers lets every frame be measured; results arrive a frame or two late.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub const PASSES: [&str; 5] = ["shadows", "forward", "volume", "bloom", "tonemap"];

/// Readbacks that can be in flight at once (frames of latency the GPU may run ahead).
const RING: usize = 3;

struct Readback {
    buffer: wgpu::Buffer,
    /// Mapped and ready to read.
    ready: Arc<AtomicBool>,
    /// Copy submitted (and mapping requested), not yet read.
    in_flight: bool,
    /// Submission order, to keep results in order.
    serial: u64,
}

pub struct GpuTimer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    ring: Vec<Readback>,
    period_ns: f32,
    /// Latest per-pass durations, milliseconds.
    pub last: [f32; PASSES.len()],
    /// GPU time of the whole frame: first pass start to last pass end, milliseconds.
    /// (Per-pass times overlap on tile-based GPUs; this span does not.)
    pub frame_ms: f32,
    /// Increments with every new measurement (lets callers count fresh samples).
    pub samples: u64,
    /// Ring slot this frame records into, if any is free.
    active: Option<usize>,
    next_serial: u64,
    last_serial_read: u64,
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
            ring: (0..RING)
                .map(|_| Readback {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("timing readback"),
                        size,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }),
                    ready: Arc::new(AtomicBool::new(false)),
                    in_flight: false,
                    serial: 0,
                })
                .collect(),
            period_ns: queue.get_timestamp_period(),
            last: [0.0; PASSES.len()],
            frame_ms: 0.0,
            samples: 0,
            active: None,
            next_serial: 1,
            last_serial_read: 0,
        })
    }

    /// Call at the start of a frame: collects finished results (polling the device so
    /// mapping callbacks run promptly) and picks a free readback slot for this frame.
    pub fn begin_frame(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);
        // Read finished slots oldest first, so the newest result ends up in `last`.
        let mut done: Vec<usize> = (0..RING)
            .filter(|&i| self.ring[i].in_flight && self.ring[i].ready.load(Ordering::Acquire))
            .collect();
        done.sort_by_key(|&i| self.ring[i].serial);
        for i in done {
            let rb = &mut self.ring[i];
            rb.ready.store(false, Ordering::Release);
            if rb.serial > self.last_serial_read
                && let Ok(view) = rb.buffer.get_mapped_range(..)
            {
                let ticks: &[u64] = bytemuck::cast_slice(&view);
                let (mut first, mut last_end) = (u64::MAX, 0u64);
                for (p, out) in self.last.iter_mut().enumerate() {
                    let (a, b) = (ticks[2 * p], ticks[2 * p + 1]);
                    *out = b.saturating_sub(a) as f32 * self.period_ns / 1e6;
                    if a > 0 && b > a {
                        first = first.min(a);
                        last_end = last_end.max(b);
                    }
                }
                if last_end > first {
                    self.frame_ms = (last_end - first) as f32 * self.period_ns / 1e6;
                    self.samples += 1;
                }
                self.last_serial_read = rb.serial;
            }
            rb.buffer.unmap();
            rb.in_flight = false;
        }
        self.active = (0..RING).find(|&i| !self.ring[i].in_flight);
    }

    /// Timestamp writes for pass `pass`; `first`/`last` let several render passes (one
    /// per shadow map) share one entry.
    pub fn writes(&self, pass: usize, first: bool, last: bool) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        // Passes in the middle of a group record nothing (wgpu requires at least one index).
        if self.active.is_none() || !(first || last) {
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
        if let Some(i) = self.active {
            enc.resolve_query_set(&self.queries, 0..COUNT, &self.resolve, 0);
            enc.copy_buffer_to_buffer(&self.resolve, 0, &self.ring[i].buffer, 0, 8 * COUNT as u64);
        }
    }

    /// Call after submitting the frame's commands.
    pub fn after_submit(&mut self) {
        let Some(i) = self.active.take() else { return };
        let rb = &mut self.ring[i];
        rb.in_flight = true;
        rb.serial = self.next_serial;
        self.next_serial += 1;
        let ready = rb.ready.clone();
        rb.buffer.map_async(wgpu::MapMode::Read, .., move |r| {
            if r.is_ok() {
                ready.store(true, Ordering::Release);
            }
        });
    }
}
