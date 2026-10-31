use crate::LoadError;
use gdtf::{Model3Detail, Model3Format, ResourceMap};
use glam::{Mat4, Vec3};
use std::io::Read;

#[derive(Debug, Clone, Copy)]
pub struct Material {
    /// Linear RGBA.
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
}

impl Default for Material {
    /// Dark matte powder-coat, used when the model has no material.
    fn default() -> Self {
        Self {
            base_color: [0.02, 0.02, 0.022, 1.0],
            metallic: 0.0,
            roughness: 0.6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub material: Material,
}

/// Triangle mesh in the model's local GDTF space (Z-up, metres).
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    pub parts: Vec<SubMesh>,
}

impl Mesh {
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut it = self.parts.iter().flat_map(|p| p.positions.iter()).map(|&p| Vec3::from(p));
        let first = it.next()?;
        Some(it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p))))
    }

    pub fn vertex_count(&self) -> usize {
        self.parts.iter().map(|p| p.positions.len()).sum()
    }

    pub fn triangle_count(&self) -> usize {
        self.parts.iter().map(|p| p.indices.len() / 3).sum()
    }

    fn transform(&mut self, m: Mat4) {
        let n = m.inverse().transpose();
        for p in &mut self.parts {
            for v in &mut p.positions {
                *v = m.transform_point3(Vec3::from(*v)).into();
            }
            for v in &mut p.normals {
                *v = n.transform_vector3(Vec3::from(*v)).normalize_or_zero().into();
            }
        }
    }
}

/// glTF is Y-up; GDTF space is Z-up. GDTF's glTF models follow glTF's convention, so
/// rotate +90° about X: glTF (x, y, z) → GDTF (x, −z, y).
const GLTF_TO_GDTF: Mat4 = Mat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, -1.0, 0.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
]);

pub(crate) fn load_model(
    res: &mut ResourceMap,
    file: &str,
    model: &str,
    size: Vec3,
) -> Result<Option<Mesh>, LoadError> {
    let mut bytes = Vec::new();
    match res.read_model_mesh(file, Model3Format::Gltf, Model3Detail::Default) {
        Ok(mut r) => {
            r.read_to_end(&mut bytes)?;
        }
        Err(gdtf::GdtfError::ResourceNotFound) => {
            // TODO: minimal 3DS chunk parser (models/3ds/<file>.3ds).
            log::warn!("model '{model}': no glTF file for '{file}' (3DS not supported yet)");
            return Ok(None);
        }
        Err(e) => return Err(e.into()),
    }

    let (doc, buffers, _images) = gltf::import_slice(&bytes).map_err(|source| LoadError::Gltf {
        model: model.to_owned(),
        source,
    })?;

    let mut mesh = Mesh::default();
    let scene = doc.default_scene().or_else(|| doc.scenes().next());
    if let Some(scene) = scene {
        for node in scene.nodes() {
            collect(&node, Mat4::IDENTITY, &buffers, &mut mesh);
        }
    }
    mesh.transform(GLTF_TO_GDTF);
    fit_to_size(&mut mesh, model, size);
    Ok(Some(mesh))
}

fn collect(node: &gltf::Node, parent: Mat4, buffers: &[gltf::buffer::Data], out: &mut Mesh) {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(m) = node.mesh() {
        for prim in m.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = prim.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
            let Some(pos) = reader.read_positions() else { continue };
            let positions: Vec<[f32; 3]> = pos
                .map(|p| world.transform_point3(Vec3::from(p)).into())
                .collect();
            let indices: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            let normal_m = world.inverse().transpose();
            let normals: Vec<[f32; 3]> = match reader.read_normals() {
                Some(n) => n
                    .map(|n| normal_m.transform_vector3(Vec3::from(n)).normalize_or_zero().into())
                    .collect(),
                None => flat_normals(&positions, &indices),
            };
            let pbr = prim.material().pbr_metallic_roughness();
            let material = if prim.material().index().is_some() {
                Material {
                    base_color: pbr.base_color_factor(),
                    metallic: pbr.metallic_factor(),
                    roughness: pbr.roughness_factor(),
                }
            } else {
                Material::default()
            };
            out.parts.push(SubMesh {
                positions,
                normals,
                indices,
                material,
            });
        }
    }
    for c in node.children() {
        collect(&c, world, buffers, out);
    }
}

/// Area-weighted vertex normals for meshes that ship without them.
fn flat_normals(pos: &[[f32; 3]], idx: &[u32]) -> Vec<[f32; 3]> {
    let mut n = vec![Vec3::ZERO; pos.len()];
    for t in idx.as_chunks::<3>().0 {
        let [a, b, c] = [t[0], t[1], t[2]].map(|i| Vec3::from(pos[i as usize]));
        let f = (b - a).cross(c - a);
        for &i in t {
            n[i as usize] += f;
        }
    }
    n.into_iter().map(|v| v.normalize_or_zero().into()).collect()
}

/// GDTF models declare a nominal Length/Width/Height. Declared sizes are often rounded or
/// measured slightly differently, so only fix gross mismatches (wrong units: mm/cm/inch
/// exports), using a uniform scale to avoid distorting the mesh.
fn fit_to_size(mesh: &mut Mesh, model: &str, size: Vec3) {
    let Some((lo, hi)) = mesh.bounds() else { return };
    let ext = hi - lo;
    // Compare along the largest declared dimension, which is the most reliable.
    let axis = size.max_position();
    if size[axis] <= 1e-6 || ext[axis] <= 1e-9 {
        return;
    }
    let scale = size[axis] / ext[axis];
    if !(0.75..=1.333).contains(&scale) {
        log::info!("model '{model}': extents {ext:?} vs declared {size:?}, scaling by {scale}");
        mesh.transform(Mat4::from_scale(Vec3::splat(scale)));
    }
}
