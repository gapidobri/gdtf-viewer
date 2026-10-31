use glam::Vec3;
use renderer::CameraParams;

/// Orbit camera around a target point, Y-up.
#[derive(Debug, Clone, Copy)]
pub struct Orbit {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for Orbit {
    /// From the audience side, seeing the fixture on its pipe and the stage below.
    fn default() -> Self {
        Self {
            target: Vec3::new(0.0, 2.6, -1.0),
            distance: 10.0,
            yaw: 0.45,
            pitch: 0.12,
        }
    }
}

impl Orbit {
    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.5, 1.5);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let (right, up, _) = self.basis();
        let s = self.distance * 0.0015;
        self.target += (-right * dx + up * dy) * s;
    }

    pub fn zoom(&mut self, steps: f32) {
        self.distance = (self.distance * 0.9f32.powf(steps)).clamp(0.2, 60.0);
    }

    fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let fwd = -self.offset_dir();
        let right = fwd.cross(Vec3::Y).normalize();
        let up = right.cross(fwd);
        (right, up, fwd)
    }

    fn offset_dir(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(sy * cp, sp, cy * cp)
    }

    pub fn camera(&self, aspect: f32) -> CameraParams {
        CameraParams {
            eye: self.target + self.offset_dir() * self.distance,
            target: self.target,
            fov_y_deg: 45.0,
            aspect,
        }
    }
}
