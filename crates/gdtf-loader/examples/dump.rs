//! Print a fixture's geometry tree, models, DMX modes and wheels.
//! Usage: cargo run -p gdtf-loader --example dump -- <file.gdtf> [--channels]
use gdtf_loader::{GeometryKind, LoadedFixture};
use glam::Mat4;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or_else(|| anyhow::anyhow!("usage: dump <file.gdtf> [--channels]"))?;
    let show_channels = args.any(|a| a == "--channels");
    let mut fx = LoadedFixture::open(&path)?;
    let ft = fx.fixture_type();
    println!("{} — {} ({})", ft.manufacturer, ft.long_name, ft.short_name);

    println!("\nModels:");
    let mut models: Vec<_> = fx.models.values().collect();
    models.sort_by(|a, b| a.name.cmp(&b.name));
    for m in models {
        let mesh = match &m.mesh {
            Some(mesh) => {
                let (lo, hi) = mesh.bounds().unwrap_or_default();
                format!("{} verts, {} tris, bounds {:.3?}..{:.3?}", mesh.vertex_count(), mesh.triangle_count(), lo, hi)
            }
            None => format!("primitive {:?}", m.primitive),
        };
        println!("  {:<10} size {:.3?}  {}", m.name, m.size, mesh);
    }

    println!("\nGeometry (world position, GDTF Z-up):");
    for root in &fx.geometry {
        root.walk(Mat4::IDENTITY, &mut |n, world, depth| {
            let t = world.w_axis.truncate();
            let extra = match &n.kind {
                GeometryKind::Beam(b) => format!(
                    "  {:?} beam {:.1}° field {:.1}° r={:.3}m {:.0}lm {:.0}K",
                    b.beam_type, b.beam_angle, b.field_angle, b.beam_radius, b.luminous_flux, b.color_temperature
                ),
                GeometryKind::Reference { geometry } => format!("  -> {geometry:?}"),
                _ => String::new(),
            };
            println!(
                "  {:indent$}{} \"{}\" model={:?} at ({:.3}, {:.3}, {:.3}){}",
                "", n.kind_name(), n.name, n.model.as_deref().unwrap_or("-"), t.x, t.y, t.z, extra,
                indent = depth * 2
            );
        });
    }

    println!("\nDMX modes:");
    for m in fx.modes() {
        println!("  \"{}\" — {} channels, footprint {}", m.name, m.channels.len(), m.footprint);
        if show_channels {
            for c in &m.channels {
                println!("      {:?} [{}] {}", c.offsets, c.geometry, c.attributes.join(", "));
            }
        }
    }

    println!("\nWheels:");
    let wheels = fx.wheels();
    for w in &wheels {
        println!("  {} ({} slots)", w.name, w.slots.len());
        for s in &w.slots {
            let mut desc = Vec::new();
            if let Some(c) = s.color { desc.push(format!("xyY {:.3},{:.3},{:.1}", c[0], c[1], c[2])); }
            if let Some(f) = &s.filter { desc.push(format!("filter {f}")); }
            if let Some(m) = &s.media_name { desc.push(format!("media {m}")); }
            if s.prism_facets > 0 { desc.push(format!("{} facets", s.prism_facets)); }
            println!("    {:<20} {}", s.name, desc.join("  "));
        }
    }

    // Check every wheel image decodes.
    let (mut ok, mut bad) = (0, 0);
    for s in wheels.iter().flat_map(|w| &w.slots) {
        if let Some(m) = &s.media_name {
            match fx.wheel_image(m) {
                Ok(_) => ok += 1,
                Err(e) => { bad += 1; eprintln!("  ! {e}"); }
            }
        }
    }
    println!("\nWheel images: {ok} decoded, {bad} failed");
    Ok(())
}
