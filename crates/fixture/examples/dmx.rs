//! Evaluate a DMX mode at its default values. Usage: dmx <file.gdtf> [mode] [chan=value ...]
use fixture::dmx::DmxPatch;
use gdtf_loader::LoadedFixture;

fn main() {
    let mut args = std::env::args().skip(1);
    let fx = LoadedFixture::open(args.next().expect("file")).expect("load");
    let mode_idx: usize = args.next().map_or(0, |m| m.parse().unwrap());
    let mode = &fx.fixture_type().dmx_modes[mode_idx];
    let p = DmxPatch::from_mode(mode);
    let mut v = p.default_values();
    for a in args {
        let (c, val) = a.split_once('=').unwrap();
        let i = p.channels.iter().position(|ch| ch.name == c).expect("channel");
        p.set(&mut v, i, val.parse().unwrap());
    }
    let masters: usize = p.channels.iter().flat_map(|c| &c.functions).filter(|f| f.master.is_some()).count();
    let expected = mode.dmx_channels.iter().flat_map(|c| &c.logical_channels).flat_map(|l| &l.channel_functions).filter(|f| f.mode_master.is_some()).count();
    println!("{} channels, footprint {}, mode masters resolved {masters}/{expected}", p.channels.len(), p.footprint);
    for (c, e) in p.channels.iter().zip(p.evaluate(&v)) {
        let f = e.function.map(|i| &c.functions[i]);
        println!(
            "{:>8} {:<26} {:>6} -> {:<28} {:>9.3} {}{}",
            format!("{:?}", c.offsets.iter().map(|o| o + 1).collect::<Vec<_>>()),
            c.name,
            e.value,
            f.map_or("-".into(), |f| format!("{} [{}]", f.name, f.attribute)),
            e.physical,
            e.set.and_then(|s| f.map(|f| f.sets[s].name.clone())).unwrap_or_default(),
            e.slot.map(|s| format!(" slot {s}")).unwrap_or_default(),
        );
    }
}
