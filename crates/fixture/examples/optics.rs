//! Print the optical train a DMX mode exposes. Usage: optics <file.gdtf> [mode]
use fixture::{beams, FixtureRig, Optics};
use gdtf_loader::LoadedFixture;

fn main() {
    let mut args = std::env::args().skip(1);
    let fx = LoadedFixture::open(args.next().expect("file")).expect("load");
    let mode_idx: usize = args.next().map_or(0, |m| m.parse().unwrap());
    let ft = fx.fixture_type();
    let mode = &ft.dmx_modes[mode_idx];
    let root = mode.geometry.as_ref().map(|g| g.to_string());
    let rig = FixtureRig::from_loaded(&fx, root.as_deref());
    let b = beams(&rig);
    let o = Optics::from_mode(ft, mode, b.first().map(|b| &b.info));
    println!("source {:?} @ {} K", o.source, o.native_cct);
    for e in &o.emitters { println!("emitter {} {} rgb {:.3?}", e.attr, e.name, e.rgb); }
    for f in &o.cmy { println!("cmy {} {} T {:.3?}", f.attr, f.name, f.transmission); }
    for w in &o.wheels {
        println!("wheel {} ({:?}, {}) index {:?} spin {:?} wheelspin {:?}", w.attr, w.kind, w.wheel_name, w.index_range, w.spin_range, w.wheel_spin_range);
        for s in w.slots.iter().take(4) { println!("    {:<24} T {:.3?} media {:?} facets {}", s.name, s.transmission, s.media, s.prism_facets); }
    }
    println!("cto {:?} zoom {:?} focus {}", o.cto, o.zoom, o.has_focus);
    let st = o.default_state(b.first().map_or(25.0, |b| b.info.field_angle));
    println!("default look {:?}", o.evaluate(&st));
}
