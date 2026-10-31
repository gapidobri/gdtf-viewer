// HDR (nits) → display: mix in bloom, apply exposure (from EV100), tonemap (AgX or
// ACES), dither. Output is display-linear; the sRGB render target view encodes it.

struct Params {
    // x = exposure multiplier, y = bloom strength, z = tonemapper (0 ACES, 1 AgX),
    // w = bloom mip count (the bloom chain sums that many levels)
    params: vec4<f32>,
};

@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var<uniform> p: Params;
@group(0) @binding(2) var bloom: texture_2d<f32>;
@group(0) @binding(3) var bloom_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle.
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

// AgX (Troy Sobotka), polynomial fit by Benjamin Wrensch ("Minimal AgX"). Rolls very
// bright, saturated light (deep blue or red beams) towards white along a natural path
// instead of skewing its hue per channel the way ACES-fitted does.
fn agx_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn agx(c: vec3<f32>) -> vec3<f32> {
    let inset = mat3x3<f32>(
        vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
    );
    let outset = mat3x3<f32>(
        vec3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
    );
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var v = inset * max(c, vec3<f32>(1e-10));
    v = (clamp(log2(v), vec3<f32>(min_ev), vec3<f32>(max_ev)) - min_ev) / (max_ev - min_ev);
    v = outset * agx_contrast(v);
    // The curve's output is display-encoded (≈ gamma 2.2); return display-linear.
    return pow(clamp(v, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3<f32>(0.0031308));
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(hdr));
    let scene = textureLoad(hdr, vec2<i32>(pos.xy), 0).rgb;
    let glare = textureSampleLevel(bloom, bloom_sampler, pos.xy / size, 0.0).rgb / max(p.params.w, 1.0);
    let exposed = mix(scene, glare, p.params.y) * p.params.x;
    let c = select(aces(exposed), agx(exposed), p.params.z > 0.5);
    // Dither with ±1 LSB triangular noise in 8-bit sRGB output, so smooth dark
    // gradients (haze falloff, beam edges) don't band. The target view is sRGB, so add
    // the noise in encoded space and convert back.
    let n = fract(52.9829189 * fract(dot(pos.xy, vec2<f32>(0.06711056, 0.00583715))));
    let n2 = fract(52.9829189 * fract(dot(pos.xy + 17.0, vec2<f32>(0.06711056, 0.00583715))));
    let encoded = linear_to_srgb(c) + (n + n2 - 1.0) / 255.0;
    return vec4<f32>(srgb_to_linear(clamp(encoded, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
}
