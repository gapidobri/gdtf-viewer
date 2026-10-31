//! Procedural meshes: GDTF primitive models and the stage.

use gdtf_loader::gdtf::model::PrimitiveType;
use gdtf_loader::{Material, Mesh, SubMesh};
use glam::Vec3;

/// Box spanning `lo..hi`.
pub fn cuboid(lo: Vec3, hi: Vec3, material: Material) -> SubMesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let faces = [
        (Vec3::X, Vec3::Y, Vec3::Z),
        (Vec3::NEG_X, Vec3::Z, Vec3::Y),
        (Vec3::Y, Vec3::Z, Vec3::X),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z),
        (Vec3::Z, Vec3::X, Vec3::Y),
        (Vec3::NEG_Z, Vec3::Y, Vec3::X),
    ];
    let c = (lo + hi) * 0.5;
    let h = (hi - lo) * 0.5;
    for (n, u, v) in faces {
        let base = positions.len() as u32;
        for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            positions.push((c + (n + u * su + v * sv) * h).into());
            normals.push(n.into());
        }
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    SubMesh {
        positions,
        normals,
        indices,
        material,
    }
}

/// Cylinder along Z, centred, with caps. Returns (side+top, bottom cap) so the bottom
/// (the lens, facing −Z) can be shaded separately.
pub fn cylinder_z(radius: f32, height: f32, segments: u32) -> (SubMesh, SubMesh) {
    let mut side = SubMesh {
        positions: vec![],
        normals: vec![],
        indices: vec![],
        material: Material::default(),
    };
    let mut bottom = side.clone();
    let (z0, z1) = (-height * 0.5, height * 0.5);
    for i in 0..=segments {
        let a = i as f32 / segments as f32 * std::f32::consts::TAU;
        let (s, c) = a.sin_cos();
        let n = [c, s, 0.0];
        side.positions.push([c * radius, s * radius, z0]);
        side.positions.push([c * radius, s * radius, z1]);
        side.normals.extend([n, n]);
    }
    for i in 0..segments {
        let b = i * 2;
        side.indices.extend([b, b + 2, b + 1, b + 1, b + 2, b + 3]);
    }
    for (mesh, z, nz) in [(&mut bottom, z0, -1.0f32), (&mut side, z1, 1.0)] {
        let centre = mesh.positions.len() as u32;
        mesh.positions.push([0.0, 0.0, z]);
        mesh.normals.push([0.0, 0.0, nz]);
        for i in 0..=segments {
            let a = i as f32 / segments as f32 * std::f32::consts::TAU;
            let (s, c) = a.sin_cos();
            mesh.positions.push([c * radius, s * radius, z]);
            mesh.normals.push([0.0, 0.0, nz]);
        }
        for i in 0..segments {
            let (a, b) = (centre + 1 + i, centre + 2 + i);
            if nz > 0.0 {
                mesh.indices.extend([centre, a, b]);
            } else {
                mesh.indices.extend([centre, b, a]);
            }
        }
    }
    (side, bottom)
}

/// Mesh for a model that has no file, sized to its declared dimensions.
/// Returns `None` for primitives we don't draw (e.g. Pigtail, Undefined).
pub fn primitive_mesh(prim: PrimitiveType, size: Vec3) -> Option<Mesh> {
    let half = size * 0.5;
    let part = match prim {
        PrimitiveType::Cylinder => {
            let (side, cap) = cylinder_z(size.x.max(size.y) * 0.5, size.z.max(1e-3), 48);
            return Some(Mesh { parts: vec![side, cap] });
        }
        PrimitiveType::Pigtail | PrimitiveType::Undefined => return None,
        // Cube, Sphere, Base, Yoke, Head, Scanner, Conventional, and the 1.1 variants:
        // a box is a reasonable stand-in until we have proper shapes.
        _ => cuboid(-half, half, Material::default()),
    };
    Some(Mesh { parts: vec![part] })
}

/// Height of the pipe the fixture hangs from (top of the fixture's base), metres.
pub const TRIM_HEIGHT: f32 = 5.0;

/// Floor, back wall and a hanging pipe, in Y-up renderer space (metres).
pub fn stage() -> Mesh {
    let floor = Material {
        // Grey dance floor.
        base_color: [0.12, 0.12, 0.12, 1.0],
        metallic: 0.0,
        roughness: 0.8,
    };
    let wall = Material {
        // Light grey cyc, so projections read well.
        base_color: [0.55, 0.55, 0.55, 1.0],
        metallic: 0.0,
        roughness: 0.95,
    };
    let pipe = Material {
        base_color: [0.6, 0.6, 0.62, 1.0],
        metallic: 1.0,
        roughness: 0.35,
    };
    let prop = Material {
        base_color: [0.4, 0.4, 0.4, 1.0],
        metallic: 0.0,
        roughness: 0.7,
    };
    let r = 0.024; // 48 mm scaffold tube
    Mesh {
        parts: vec![
            cuboid(Vec3::new(-12.0, -0.05, -8.0), Vec3::new(12.0, 0.0, 12.0), floor),
            cuboid(Vec3::new(-12.0, 0.0, -8.1), Vec3::new(12.0, 10.0, -8.0), wall),
            // A 1 m cube, so beams have something to cast shadows with.
            cuboid(Vec3::new(1.0, 0.0, -2.5), Vec3::new(2.0, 1.0, -1.5), prop),
            cuboid(
                Vec3::new(-6.0, TRIM_HEIGHT, -r),
                Vec3::new(6.0, TRIM_HEIGHT + 2.0 * r, r),
                pipe,
            ),
        ],
    }
}
