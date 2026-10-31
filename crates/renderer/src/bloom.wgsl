// Bloom (camera/eye glare): a mip chain of the HDR image, downsampled with a 13-tap
// filter and upsampled with a 3×3 tent, each upsample added onto the next level
// (Jimenez, "Next Generation Post Processing in Call of Duty: Advanced Warfare", 2014).

struct Params {
    // x = exposure (for the anti-firefly weighting), y = strength, z = mip count
    params: vec4<f32>,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> p: Params;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return o;
}

fn tap(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(src, src_sampler, uv, 0.0).rgb;
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Karis average of a 2×2 box: weights 1/(1+luma) (in exposed, display-ish units) stop a
// single extreme pixel (a lens seen edge-on) from flickering into a huge blob.
fn karis(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    let e = p.params.x;
    let wa = 1.0 / (1.0 + luma(a) * e);
    let wb = 1.0 / (1.0 + luma(b) * e);
    let wc = 1.0 / (1.0 + luma(c) * e);
    let wd = 1.0 / (1.0 + luma(d) * e);
    return (a * wa + b * wb + c * wc + d * wd) / (wa + wb + wc + wd);
}

fn downsample(uv: vec2<f32>, use_karis: bool) -> vec3<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src));
    let a = tap(uv + t * vec2<f32>(-2.0, -2.0));
    let b = tap(uv + t * vec2<f32>(0.0, -2.0));
    let c = tap(uv + t * vec2<f32>(2.0, -2.0));
    let d = tap(uv + t * vec2<f32>(-1.0, -1.0));
    let e = tap(uv + t * vec2<f32>(1.0, -1.0));
    let f = tap(uv + t * vec2<f32>(-2.0, 0.0));
    let g = tap(uv);
    let h = tap(uv + t * vec2<f32>(2.0, 0.0));
    let i = tap(uv + t * vec2<f32>(-1.0, 1.0));
    let j = tap(uv + t * vec2<f32>(1.0, 1.0));
    let k = tap(uv + t * vec2<f32>(-2.0, 2.0));
    let l = tap(uv + t * vec2<f32>(0.0, 2.0));
    let m = tap(uv + t * vec2<f32>(2.0, 2.0));
    if use_karis {
        return karis(d, e, i, j) * 0.5
            + karis(a, b, f, g) * 0.125 + karis(b, c, g, h) * 0.125
            + karis(f, g, k, l) * 0.125 + karis(g, h, l, m) * 0.125;
    }
    return (d + e + i + j) * 0.125
        + (a + c + k + m) * 0.03125
        + (b + f + h + l) * 0.0625
        + g * 0.125;
}

@fragment
fn fs_down_first(i: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(downsample(i.uv, true), 1.0);
}

@fragment
fn fs_down(i: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(downsample(i.uv, false), 1.0);
}

// 3×3 tent, added (blend One/One) onto the larger level.
@fragment
fn fs_up(i: VsOut) -> @location(0) vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src));
    var c = tap(i.uv) * 4.0;
    c += (tap(i.uv + vec2<f32>(-t.x, 0.0)) + tap(i.uv + vec2<f32>(t.x, 0.0))
        + tap(i.uv + vec2<f32>(0.0, -t.y)) + tap(i.uv + vec2<f32>(0.0, t.y))) * 2.0;
    c += tap(i.uv + vec2<f32>(-t.x, -t.y)) + tap(i.uv + vec2<f32>(t.x, -t.y))
        + tap(i.uv + vec2<f32>(-t.x, t.y)) + tap(i.uv + vec2<f32>(t.x, t.y));
    return vec4<f32>(c / 16.0, 1.0);
}
