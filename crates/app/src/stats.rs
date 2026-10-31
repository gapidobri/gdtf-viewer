//! Frame statistics: presented FPS, CPU time per phase, GPU time per pass.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// CPU phases measured each frame, in order.
pub const CPU_PHASES: [&str; 6] = ["acquire", "ui", "update", "render", "egui", "present"];

#[derive(Default)]
pub struct FrameStats {
    /// Presentation times over the last second.
    presents: VecDeque<Instant>,
    /// Exponential moving averages, milliseconds.
    pub cpu: [f32; CPU_PHASES.len()],
    pub gpu: Option<Vec<(&'static str, f32)>>,
    pub visible: bool,
    /// Redraw every frame (otherwise only on input/animation), for measuring.
    pub continuous: bool,
    /// GPU time of the whole frame (first pass start to last pass end), ms.
    pub gpu_frame: Option<f32>,
    /// Current automatic quality level, if enabled.
    pub quality: Option<&'static str>,
    /// Log a summary at this interval (for measuring without looking at the overlay).
    pub log_every: Option<Duration>,
    last_log: Option<Instant>,
}

/// Times consecutive phases of one frame.
pub struct PhaseTimer {
    last: Instant,
    pub ms: [f32; CPU_PHASES.len()],
}

impl PhaseTimer {
    pub fn start() -> Self {
        Self {
            last: Instant::now(),
            ms: [0.0; CPU_PHASES.len()],
        }
    }

    /// Ends phase `i` (index into [`CPU_PHASES`]) and starts the next.
    pub fn lap(&mut self, i: usize) {
        let now = Instant::now();
        self.ms[i] += (now - self.last).as_secs_f32() * 1e3;
        self.last = now;
    }
}

impl FrameStats {
    pub fn new() -> Self {
        Self {
            visible: true,
            ..Default::default()
        }
    }

    pub fn record(&mut self, phases: &PhaseTimer, gpu: Option<&[f32]>) {
        let now = Instant::now();
        self.presents.push_back(now);
        while self.presents.front().is_some_and(|t| now - *t > Duration::from_secs(1)) {
            self.presents.pop_front();
        }
        for (avg, v) in self.cpu.iter_mut().zip(phases.ms) {
            *avg += (v - *avg) * 0.1;
        }
        if let Some(g) = gpu {
            self.gpu = Some(renderer::GPU_PASSES.iter().copied().zip(g.iter().copied()).collect());
        }
        if let Some(every) = self.log_every
            && self.last_log.is_none_or(|t| now - t >= every)
        {
            self.last_log = Some(now);
            log::info!(
                "{} fps | quality {:?} | gpu frame {:?} ms | cpu {:.2} ms {:?} | gpu {:?}",
                self.fps(),
                self.quality,
                self.gpu_frame.map(|v| (v * 100.0).round() / 100.0),
                self.cpu_total(),
                CPU_PHASES.iter().zip(self.cpu.map(|v| (v * 100.0).round() / 100.0)).collect::<Vec<_>>(),
                self.gpu.as_ref().map(|g| g.iter().map(|(n, v)| (*n, (v * 100.0).round() / 100.0)).collect::<Vec<_>>()),
            );
        }
    }

    /// Frames presented during the last second.
    pub fn fps(&self) -> usize {
        self.presents.len()
    }

    pub fn cpu_total(&self) -> f32 {
        // "acquire" and "present" are mostly waiting for the display, not work.
        self.cpu[1..5].iter().sum()
    }

    pub fn gpu_total(&self) -> Option<f32> {
        self.gpu.as_ref().map(|g| g.iter().map(|(_, v)| v).sum())
    }

    pub fn ui(&self, ctx: &egui::Context) {
        if !self.visible {
            return;
        }
        egui::Area::new(egui::Id::unique("frame_stats"))
            .anchor(egui::Align2::RIGHT_TOP, [-8.0, 8.0])
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.style_mut().override_font_id = Some(egui::FontId::monospace(11.0));
                    let work = self.cpu_total().max(self.gpu_frame.or(self.gpu_total()).unwrap_or(0.0));
                    ui.strong(format!("{:>3} fps", self.fps()));
                    ui.label(format!(
                        "frame cost {:.1} ms (≈{:.0} fps max)",
                        work,
                        1000.0 / work.max(0.01)
                    ));
                    ui.separator();
                    ui.label(format!("CPU {:>6.2} ms", self.cpu_total()));
                    for (name, ms) in CPU_PHASES.iter().zip(self.cpu) {
                        ui.weak(format!("  {name:<8}{ms:>6.2}"));
                    }
                    if let Some(q) = self.quality {
                        ui.label(format!("quality: {q} (auto)"));
                    }
                    match (&self.gpu, self.gpu_frame.or(self.gpu_total())) {
                        (Some(g), Some(total)) => {
                            ui.label(format!("GPU {total:>6.2} ms"));
                            for (name, ms) in g {
                                ui.weak(format!("  {name:<8}{ms:>6.2}"));
                            }
                        }
                        _ => {
                            ui.weak("GPU timing unavailable");
                        }
                    }
                });
            });
    }
}
