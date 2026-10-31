//! Checks spectral data: filter transmissions vs the file's own values, and beam colours.
use fixture::{beams, color, FixtureRig, Optics};
use gdtf_loader::LoadedFixture;
fn main() {
    let fx = LoadedFixture::open(std::env::args().nth(1).unwrap()).unwrap();
    let ft = fx.fixture_type();
    let mode = &ft.dmx_modes[0];
    let rig = FixtureRig::from_loaded(&fx, mode.geometry.as_ref().map(|g| g.to_string()).as_deref());
    let b = beams(&rig);
    let o = Optics::from_mode(ft, mode, b.first().map(|b| &b.info));
    let sp = &o.spectral;
    println!("source: {} {:?}", sp.source_label, sp.source.map(|s| color::xyz_to_xy(s.to_xyz())));
    let check = |name: &str, t: &color::Spectrum| {
        let src = sp.source.unwrap_or(color::Spectrum::constant(1.0));
        let lt = src.mul(t).luminance() / src.luminance();
        let file = ft.physical_descriptions.filter(name).and_then(|f| f.measurements.iter().find_map(|m| m.transmission));
        println!("  {name:<28} luminous T {:6.2}%   file {:>8}", lt * 100.0, file.map_or("-".into(), |v| format!("{v:.2}%")));
    };
    for (c, t) in o.cmy.iter().zip(&sp.cmy) { if let Some(t) = t { check(&c.name, t) } }
    for (w, slots) in o.wheels.iter().zip(&sp.slots) {
        for (s, t) in w.slots.iter().zip(slots) {
            if let Some(t) = t {
                let src = sp.source.unwrap_or(color::Spectrum::constant(1.0));
                let lt = src.mul(t).luminance() / src.luminance();
                println!("  {:<28} luminous T {:6.2}%   (xyY Y in file {:.2}%)", s.name, lt * 100.0, 100.0 * color::luminance(s.transmission));
            }
        }
    }
    for (e, s) in o.emitters.iter().zip(&sp.emitters) {
        println!("  emitter {:<10} spectral {}  xy {:?}", e.name, s.is_some(), s.map(|s| color::xyz_to_xy(s.to_xyz())));
    }
    let st = o.default_state(25.0);
    let look = o.evaluate(&st);
    println!("default colour {:?} lum {:.3} spectral {}", look.color, color::luminance(look.color), look.spectrum.is_some());
}
