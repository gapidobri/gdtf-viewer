//! Automatic quality: keeps the GPU frame time inside the 60 fps budget by stepping the
//! volumetric settings (the expensive, scene-dependent part) up or down.

use std::time::{Duration, Instant};

/// One rung of the quality ladder.
#[derive(Debug, Clone, Copy)]
pub struct Level {
    pub name: &'static str,
    /// Ray-march steps per beam.
    pub steps: u32,
    /// Full-res pixels per volume pixel (2 = half resolution, 4 = quarter).
    pub volume_scale: u32,
}

pub const LEVELS: [Level; 6] = [
    Level { name: "lowest", steps: 8, volume_scale: 4 },
    Level { name: "low", steps: 12, volume_scale: 4 },
    Level { name: "medium", steps: 16, volume_scale: 2 },
    Level { name: "good", steps: 24, volume_scale: 2 },
    Level { name: "high", steps: 32, volume_scale: 2 },
    Level { name: "ultra", steps: 48, volume_scale: 2 },
];

/// GPU frame time above which quality steps down (60 Hz = 16.7 ms, with margin for the
/// compositor and frame-to-frame variation).
const TOO_SLOW_MS: f32 = 14.0;
/// Below this, step up (the next level costs up to ~1.5×).
const ROOM_MS: f32 = 8.5;
/// Where a step down aims (between ROOM_MS and TOO_SLOW_MS).
const TARGET_MS: f32 = 11.0;
/// Approximate cost ratio between neighbouring levels, for multi-level drops.
const LEVEL_COST_RATIO: f32 = 1.25;
/// Measurements still in flight from before a change (the GPU timer lags a few frames);
/// these are ignored after each change.
const STALE_SAMPLES: u32 = 3;
/// Fresh measurements needed before judging a level.
const MIN_SAMPLES: u32 = 4;
/// Minimum time at a level before stepping down (fast: dropped frames are visible) and
/// before stepping up (slower: avoids see-sawing).
const SETTLE_DOWN: Duration = Duration::from_millis(150);
const SETTLE_UP: Duration = Duration::from_millis(1000);
/// After a level proved too slow, don't try it again for this long (the scene may have
/// changed by then).
const BLOCK: Duration = Duration::from_secs(30);

pub struct AutoQuality {
    pub enabled: bool,
    pub level: usize,
    /// Smoothed GPU frame time, ms.
    pub gpu_ms: Option<f32>,
    /// Previous raw sample (with the current one: a spike-proof recent level).
    prev_ms: Option<f32>,
    samples: u32,
    last_change: Instant,
    blocked_until: [Option<Instant>; LEVELS.len()],
}

impl Default for AutoQuality {
    fn default() -> Self {
        Self {
            enabled: true,
            level: 4,
            gpu_ms: None,
            prev_ms: None,
            samples: 0,
            last_change: Instant::now(),
            blocked_until: [None; LEVELS.len()],
        }
    }
}

impl AutoQuality {
    pub fn current(&self) -> Level {
        LEVELS[self.level.min(LEVELS.len() - 1)]
    }

    /// Feeds one GPU frame time. Returns true if the level changed.
    pub fn update(&mut self, frame_ms: f32) -> bool {
        self.update_at(frame_ms, Instant::now())
    }

    /// Call when rendering paused (window hidden, idle, a hitch): the GPU clocks down
    /// while idle, so the next few measurements read slow. Start measuring afresh.
    pub fn restart_measuring(&mut self) {
        self.samples = 0;
        self.gpu_ms = None;
        self.prev_ms = None;
    }

    fn update_at(&mut self, frame_ms: f32, now: Instant) -> bool {
        self.samples += 1;
        if self.samples <= STALE_SAMPLES {
            return false; // still measuring frames rendered before the last change
        }
        let ema = self.gpu_ms.map_or(frame_ms, |e| e + (frame_ms - e) * 0.35);
        self.gpu_ms = Some(ema);
        // Two consecutive raw samples agreeing: reacts to a sudden change faster than
        // the average, while a single spike can't trigger it.
        let recent = self.prev_ms.map_or(frame_ms, |p| p.min(frame_ms));
        self.prev_ms = Some(frame_ms);
        if !self.enabled || self.samples < STALE_SAMPLES + MIN_SAMPLES {
            return false;
        }
        let since = now - self.last_change;
        // Too slow: two consecutive slow frames (a single spike can't trigger it); covers
        // both sudden changes and gradual creep.
        let too_slow = recent > TOO_SLOW_MS;
        let next = if too_slow && self.level > 0 && since >= SETTLE_DOWN {
            // Remember that this level is too heavy for this scene.
            self.blocked_until[self.level] = Some(now + BLOCK);
            // Far over budget: drop as many levels as the overload suggests, at once.
            let load = ema.max(recent);
            let steps = ((load / TARGET_MS).ln() / LEVEL_COST_RATIO.ln()).ceil().max(1.0) as usize;
            self.level.saturating_sub(steps)
        } else if ema < ROOM_MS
            && since >= SETTLE_UP
            && self.level + 1 < LEVELS.len()
            && self.blocked_until[self.level + 1].is_none_or(|t| now >= t)
        {
            self.level + 1
        } else {
            return false;
        };
        log::debug!("quality {} → {} ({ema:.1} ms)", LEVELS[self.level].name, LEVELS[next].name);
        self.level = next;
        self.last_change = now;
        self.samples = 0;
        self.gpu_ms = None;
        self.prev_ms = None;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(q: &mut AutoQuality, ms: impl Fn(usize) -> f32, seconds: f32, start: Instant) -> Instant {
        let mut t = start;
        for _ in 0..(seconds * 60.0) as usize {
            t += Duration::from_millis(16);
            let l = q.level;
            q.update_at(ms(l), t);
        }
        t
    }

    #[test]
    fn steps_down_until_within_budget_and_stays() {
        let mut q = AutoQuality::default();
        // Cost doubles per level from 4 ms at the lowest: 4, 8, 16, 32, 64, 128.
        let cost = |l: usize| 4.0 * 2f32.powi(l as i32);
        let t = run(&mut q, cost, 10.0, Instant::now());
        assert_eq!(q.level, 1, "should settle on the fastest level under 14 ms");
        // No oscillation: it doesn't keep trying level 2 every few frames.
        let before = q.level;
        run(&mut q, cost, 5.0, t);
        assert_eq!(q.level, before);
    }

    #[test]
    fn reacts_within_a_quarter_second() {
        let mut q = AutoQuality::default();
        let start = Instant::now();
        // Settle at "high" (level 4) with room to spare but blocked from going up.
        q.blocked_until[5] = Some(start + Duration::from_secs(60));
        let t = run(&mut q, |_| 9.0, 2.0, start);
        assert_eq!(q.level, 4);
        // The scene suddenly costs 3× as much (e.g. a prism comes in).
        let mut t2 = t;
        let mut frames = 0;
        while q.level == 4 && frames < 120 {
            t2 += Duration::from_millis(16);
            q.update_at(27.0, t2);
            frames += 1;
        }
        assert!(frames <= 15, "took {frames} frames");
        // It dropped several levels at once, not one.
        assert!(q.level <= 1, "level {}", q.level);
    }

    #[test]
    fn steps_up_when_there_is_room() {
        let mut q = AutoQuality {
            level: 0,
            ..Default::default()
        };
        run(&mut q, |_| 3.0, 10.0, Instant::now());
        assert_eq!(q.level, LEVELS.len() - 1);
    }

    #[test]
    fn disabled_keeps_level() {
        let mut q = AutoQuality {
            enabled: false,
            ..Default::default()
        };
        run(&mut q, |_| 40.0, 5.0, Instant::now());
        assert_eq!(q.level, 4);
    }
}
