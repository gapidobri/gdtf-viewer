//! DMX over the network: Art-Net (ArtDmx, UDP 6454) and sACN / E1.31 (UDP 5568,
//! multicast 239.255.hi.lo). A background thread keeps the latest frame of one universe.

use std::net::{Ipv4Addr, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    ArtNet,
    Sacn,
}

impl Protocol {
    pub fn port(self) -> u16 {
        match self {
            Protocol::ArtNet => 6454,
            Protocol::Sacn => 5568,
        }
    }
}

/// Parses an ArtDmx packet: (15-bit port address, DMX data).
pub fn parse_artnet(p: &[u8]) -> Option<(u16, &[u8])> {
    if p.len() < 18 || &p[..8] != b"Art-Net\0" {
        return None;
    }
    let opcode = u16::from_le_bytes([p[8], p[9]]);
    if opcode != 0x5000 {
        return None;
    }
    let universe = u16::from_le_bytes([p[14], p[15]]) & 0x7fff; // SubUni | Net << 8
    let len = u16::from_be_bytes([p[16], p[17]]) as usize;
    let data = p.get(18..18 + len.min(512))?;
    Some((universe, data))
}

/// Parses an E1.31 data packet with start code 0: (universe, DMX data).
pub fn parse_sacn(p: &[u8]) -> Option<(u16, &[u8])> {
    if p.len() < 126 || &p[4..16] != b"ASC-E1.17\0\0\0" {
        return None;
    }
    let root_vector = u32::from_be_bytes([p[18], p[19], p[20], p[21]]);
    let frame_vector = u32::from_be_bytes([p[40], p[41], p[42], p[43]]);
    if root_vector != 0x0000_0004 || frame_vector != 0x0000_0002 || p[117] != 0x02 {
        return None;
    }
    let universe = u16::from_be_bytes([p[113], p[114]]);
    let count = u16::from_be_bytes([p[123], p[124]]) as usize; // includes the start code
    if p[125] != 0 || count == 0 {
        return None; // not DMX level data
    }
    let data = p.get(126..125 + count.min(513))?;
    Some((universe, data))
}

#[derive(Debug, Default)]
pub struct Frame {
    pub data: Vec<u8>,
    pub received: Option<Instant>,
    pub packets: u64,
}

/// Listens for one universe on one protocol until dropped.
pub struct Receiver {
    pub frame: Arc<Mutex<Frame>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Receiver {
    pub fn start(protocol: Protocol, universe: u16) -> std::io::Result<Self> {
        let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, protocol.port()))?;
        if protocol == Protocol::Sacn {
            let group = Ipv4Addr::new(239, 255, (universe >> 8) as u8, universe as u8);
            socket.join_multicast_v4(&group, &Ipv4Addr::UNSPECIFIED)?;
        }
        socket.set_read_timeout(Some(Duration::from_millis(200)))?;
        let frame = Arc::new(Mutex::new(Frame::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let (f, s) = (frame.clone(), stop.clone());
        let thread = std::thread::Builder::new()
            .name(format!("{protocol:?} rx"))
            .spawn(move || {
                let mut buf = [0u8; 1500];
                while !s.load(Ordering::Relaxed) {
                    let Ok(n) = socket.recv(&mut buf) else { continue };
                    let parsed = match protocol {
                        Protocol::ArtNet => parse_artnet(&buf[..n]),
                        Protocol::Sacn => parse_sacn(&buf[..n]),
                    };
                    if let Some((u, data)) = parsed
                        && u == universe
                        && let Ok(mut fr) = f.lock()
                    {
                        fr.data.clear();
                        fr.data.extend_from_slice(data);
                        fr.data.resize(512, 0);
                        fr.received = Some(Instant::now());
                        fr.packets += 1;
                    }
                }
            })?;
        log::info!("listening for {protocol:?} universe {universe} on UDP {}", protocol.port());
        Ok(Self {
            frame,
            stop,
            thread: Some(thread),
        })
    }

    /// Latest data if a packet arrived within the last two seconds.
    pub fn latest(&self) -> Option<Vec<u8>> {
        let f = self.frame.lock().ok()?;
        f.received
            .filter(|t| t.elapsed() < Duration::from_secs(2))
            .map(|_| f.data.clone())
    }

    pub fn packets(&self) -> u64 {
        self.frame.lock().map_or(0, |f| f.packets)
    }
}

impl Drop for Receiver {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artdmx(net: u8, subuni: u8, data: &[u8]) -> Vec<u8> {
        let mut p = b"Art-Net\0".to_vec();
        p.extend_from_slice(&0x5000u16.to_le_bytes());
        p.extend_from_slice(&14u16.to_be_bytes());
        p.extend_from_slice(&[0, 0, subuni, net]);
        p.extend_from_slice(&(data.len() as u16).to_be_bytes());
        p.extend_from_slice(data);
        p
    }

    fn e131(universe: u16, data: &[u8]) -> Vec<u8> {
        let mut p = vec![0u8; 126];
        p[0..2].copy_from_slice(&0x0010u16.to_be_bytes());
        p[4..16].copy_from_slice(b"ASC-E1.17\0\0\0");
        p[18..22].copy_from_slice(&4u32.to_be_bytes());
        p[40..44].copy_from_slice(&2u32.to_be_bytes());
        p[113..115].copy_from_slice(&universe.to_be_bytes());
        p[117] = 0x02;
        p[123..125].copy_from_slice(&((data.len() + 1) as u16).to_be_bytes());
        p[125] = 0; // start code
        p.extend_from_slice(data);
        p
    }

    #[test]
    fn artnet_port_address() {
        let p = artdmx(1, 0x23, &[1, 2, 3]);
        let (u, d) = parse_artnet(&p).unwrap();
        assert_eq!(u, 0x123);
        assert_eq!(d, &[1, 2, 3]);
        assert!(parse_artnet(&p[..10]).is_none());
    }

    #[test]
    fn sacn_universe_and_data() {
        let p = e131(7, &[9, 8, 7, 6]);
        let (u, d) = parse_sacn(&p).unwrap();
        assert_eq!(u, 7);
        assert_eq!(d, &[9, 8, 7, 6]);
        let mut alt = p.clone();
        alt[125] = 0xDD; // per-address priority packet, not levels
        assert!(parse_sacn(&alt).is_none());
    }

    #[test]
    fn receives_artnet_over_loopback() {
        // Uses a high test port via the parser + a raw socket pair rather than 6454, which
        // may be taken by a console on this machine.
        let rx = UdpSocket::bind("127.0.0.1:0").unwrap();
        let tx = UdpSocket::bind("127.0.0.1:0").unwrap();
        tx.send_to(&artdmx(0, 1, &[255; 4]), rx.local_addr().unwrap()).unwrap();
        let mut buf = [0u8; 600];
        let n = rx.recv(&mut buf).unwrap();
        assert_eq!(parse_artnet(&buf[..n]), Some((1, &[255u8; 4][..])));
    }
}
