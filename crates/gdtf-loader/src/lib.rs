//! Loads a `.gdtf` file into a convenient, renderer-friendly form.
//!
//! Wraps the [`gdtf`] crate (which stays close to the spec) and adds:
//! - a geometry tree with `glam` transforms ([`GeometryNode`]),
//! - triangle meshes for each model ([`Mesh`]), loaded from the glTF resources,
//! - wheel and DMX mode summaries.
//!
//! All coordinates stay in GDTF space: right-handed, Z-up, metres. The device hangs
//! from its base, pan axis is Z, tilt axis is X, and beams emit along −Z.

mod geometry;
mod mesh;
mod summary;

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

pub use gdtf;
pub use geometry::{BeamInfo, BeamShape, GeometryKind, GeometryNode};
pub use mesh::{Material, Mesh, SubMesh};
pub use summary::{ChannelSummary, ModeSummary, SlotSummary, WheelSummary};

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("gdtf error: {0}")]
    Gdtf(#[from] gdtf::GdtfError),
    #[error("gltf error in model '{model}': {source}")]
    Gltf { model: String, source: gltf::Error },
    #[error("image error in '{name}': {source}")]
    Image { name: String, source: image::ImageError },
    #[error("file contains no fixture types")]
    NoFixtureType,
}

/// Static description of a model referenced by geometries.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    /// Nominal size in metres: x = Length, y = Width, z = Height.
    pub size: glam::Vec3,
    pub primitive: gdtf::model::PrimitiveType,
    /// Mesh from the file, scaled to `size`. `None` for primitive-only models
    /// or when the file could not be loaded (see the log).
    pub mesh: Option<Mesh>,
}

/// A loaded fixture: the raw `gdtf` description plus derived data.
pub struct LoadedFixture {
    pub file: gdtf::GdtfFile,
    pub models: HashMap<String, ModelInfo>,
    pub geometry: Vec<GeometryNode>,
}

impl LoadedFixture {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let file = std::fs::File::open(path)?;
        let mut file = gdtf::GdtfFile::new(file)?;
        let ft = file
            .description
            .fixture_types
            .first()
            .ok_or(LoadError::NoFixtureType)?;

        let geometry = ft.geometries.iter().map(geometry::convert).collect();

        let mut models = HashMap::new();
        for m in &ft.models {
            let Some(name) = m.name.as_ref().map(|n| n.to_string()) else {
                continue;
            };
            let size = glam::vec3(m.length as f32, m.width as f32, m.height as f32);
            let mesh = match m.file.as_deref().filter(|f| !f.is_empty()) {
                Some(f) => match mesh::load_model(&mut file.resources, f, &name, size) {
                    Ok(mesh) => mesh,
                    Err(e) => {
                        log::warn!("{e}");
                        None
                    }
                },
                None => None,
            };
            models.insert(
                name.clone(),
                ModelInfo {
                    name,
                    size,
                    primitive: m.primitive_type,
                    mesh,
                },
            );
        }

        Ok(Self {
            file,
            models,
            geometry,
        })
    }

    pub fn fixture_type(&self) -> &gdtf::fixture_type::FixtureType {
        &self.file.description.fixture_types[0]
    }

    pub fn wheels(&self) -> Vec<WheelSummary> {
        summary::wheels(self.fixture_type())
    }

    pub fn modes(&self) -> Vec<ModeSummary> {
        summary::modes(self.fixture_type())
    }

    /// Loads a wheel slot image as a light-transmission map (GDTF Annex E): RGB is the
    /// sRGB-decoded colour × alpha (alpha 0 = gobo holder, black = opaque, white = clear),
    /// resampled to `size`×`size`. Returns linear RGB in 0..1, row-major.
    pub fn gobo_transmission(&mut self, media_name: &str, size: u32) -> Result<Vec<[f32; 3]>, LoadError> {
        let img = self.wheel_image(media_name)?;
        let img = if img.width() != size || img.height() != size {
            image::imageops::resize(&img, size, size, image::imageops::FilterType::Triangle)
        } else {
            img
        };
        let to_linear = |c: u8| {
            let c = c as f32 / 255.0;
            if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
        };
        Ok(img
            .pixels()
            .map(|p| {
                let a = p[3] as f32 / 255.0;
                [to_linear(p[0]) * a, to_linear(p[1]) * a, to_linear(p[2]) * a]
            })
            .collect())
    }

    /// Decodes a wheel slot image (`wheels/<media_name>.png`) to RGBA8.
    pub fn wheel_image(&mut self, media_name: &str) -> Result<image::RgbaImage, LoadError> {
        let mut bytes = Vec::new();
        self.file
            .resources
            .read_wheel_media(media_name)?
            .read_to_end(&mut bytes)?;
        image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
            .map(|i| i.to_rgba8())
            .map_err(|source| LoadError::Image {
                name: media_name.to_owned(),
                source,
            })
    }
}
