// Shared by forward.wgsl and volume.wgsl (concatenated in front of them at load time):
// scene bindings (group 0), light data and the beam's angular/gobo/colour filtering.
// Units: lights in candela, illuminance in lux, radiance in cd/m² (nits).
const MAX_LIGHTS: u32 = 4u;
const PROFILE_SAMPLES: f32 = 256.0;
const GOBO_SIZE: f32 = 256.0;
const PI: f32 = 3.14159265;

struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    // x = house-light illuminance (lux), y = light count
    params: vec4<f32>,
};

struct Light {
    // Projector from the beam's virtual apex (for shadows).
    view_proj: mat4x4<f32>,
    // xyz = virtual apex behind the lens, w = max angle (radians) of the profile
    apex: vec4<f32>,
    // xyz = lens centre, w = lens radius (m)
    lens: vec4<f32>,
    // xyz = beam axis, w = shadow layer
    dir: vec4<f32>,
    // rgb = colour × on-axis intensity (cd), w = 1 if enabled
    color: vec4<f32>,
    // Per gobo wheel: layer of slot a, layer of slot b, wheel travel 0..1, rotation (rad)
    gobo: array<vec4<f32>, 2>,
    // x = gobo count, y = projector NDC → field circle scale, z = 1/focus distance (1/m),
    // w = lens aperture diameter (m)
    gobo_params: vec4<f32>,
    // Colour wheel split: rgb of slot a, w = 1 if active
    split_a: vec4<f32>,
    // rgb of slot b, w = wheel travel 0..1
    split_b: vec4<f32>,
    // x = tan(half field angle), y = slot pitch
    optics: vec4<f32>,
    // Projector frame: right and up unit vectors (gobo image x and y).
    basis_x: vec4<f32>,
    basis_y: vec4<f32>,
    // Relative intensity sampled evenly over 0..max angle, 4 per vec4.
    profile: array<vec4<f32>, 64>,
};

struct Lights {
    items: array<Light, MAX_LIGHTS>,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> lights: Lights;
@group(0) @binding(2) var shadow_maps: texture_depth_2d_array;
@group(0) @binding(3) var shadow_sampler: sampler_comparison;
@group(0) @binding(4) var gobo_maps: texture_2d_array<f32>;
@group(0) @binding(5) var gobo_sampler: sampler;

fn profile(i: u32, theta: f32) -> f32 {
    let l = &lights.items[i]; // pointer: copying the 1.3 KB struct spills registers
    let x = theta / l.apex.w * (PROFILE_SAMPLES - 1.0);
    if x >= PROFILE_SAMPLES - 1.0 {
        return 0.0;
    }
    let i0 = u32(floor(x));
    let i1 = i0 + 1u;
    let a = l.profile[i0 / 4u][i0 % 4u];
    let b = l.profile[i1 / 4u][i1 % 4u];
    return mix(a, b, fract(x));
}

fn shadow(i: u32, world: vec3<f32>, n: vec3<f32>) -> f32 {
    let l = &lights.items[i]; // pointer: copying the 1.3 KB struct spills registers
    // Normal offset scaled by distance keeps acne away without peter-panning up close.
    let d = distance(world, l.apex.xyz);
    let p = world + n * (0.0015 * d + 0.002);
    let clip = l.view_proj * vec4<f32>(p, 1.0);
    if clip.w <= 0.0 {
        return 0.0;
    }
    let ndc = clip.xyz / clip.w;
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z > 1.0 {
        return 0.0;
    }
    let layer = i32(l.dir.w);
    let texel = 1.0 / vec2<f32>(textureDimensions(shadow_maps));
    var sum = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            sum += textureSampleCompareLevel(
                shadow_maps, shadow_sampler, uv + vec2<f32>(f32(x), f32(y)) * texel, layer, ndc.z);
        }
    }
    return sum / 9.0;
}

fn inside01(uv: vec2<f32>) -> bool {
    return all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
}

fn rotate_about_centre(uv: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    let p = uv - 0.5;
    return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y) + 0.5;
}

// One gobo wheel. Between slots both neighbours are in the aperture: the wheel slides
// slot a out and slot b in (like a real wheel), rather than crossfading.
fn gobo_wheel(g: vec4<f32>, uv: vec2<f32>, lod: f32, pitch: f32) -> vec3<f32> {
    let ua = rotate_about_centre(uv - vec2<f32>(g.z * pitch, 0.0), -g.w);
    let ub = rotate_about_centre(uv + vec2<f32>((1.0 - g.z) * pitch, 0.0), -g.w);
    var t = vec3<f32>(0.0);
    if inside01(ua) {
        t += textureSampleLevel(gobo_maps, gobo_sampler, ua, i32(g.x), lod).rgb;
    }
    if inside01(ub) {
        t += textureSampleLevel(gobo_maps, gobo_sampler, ub, i32(g.y), lod).rgb;
    }
    return t;
}

// Light transmission through colour wheel and gobos towards a world position.
fn beam_filter(i: u32, world: vec3<f32>) -> vec3<f32> {
    let l = &lights.items[i]; // pointer: copying the 1.3 KB struct spills registers
    let count = u32(l.gobo_params.x);
    if count == 0u && l.split_a.w < 0.5 {
        return vec3<f32>(1.0);
    }
    // Where the gobo plane is seen from `world`. A point p receives light from every point
    // q of the lens; the ray q→p carries the gobo value at direction
    //     p/d + q·(1/d_focus − 1/d)        (tan units, d = distance along the axis),
    // so the gobo is seen blurred over a disc centred on the direction from the lens
    // *centre*, with radius r·|1/d_focus − 1/d|. Sharp at the focus distance, and near
    // the lens only the part of the gobo "visible" through the lens contributes.
    let axis = l.dir.xyz;
    let rel = world - l.lens.xyz;
    let ax = max(dot(rel, axis), 1e-4);
    let lat = rel - axis * ax;
    let tan_field = l.optics.x;
    let c = vec2<f32>(dot(lat, l.basis_x.xyz), dot(lat, l.basis_y.xyz)) / (ax * tan_field) * vec2<f32>(0.5, -0.5) + 0.5;
    let radius = l.lens.w * abs(l.gobo_params.z - 1.0 / ax) / tan_field * 0.5; // in gobo uv

    // The field stop (open aperture, layer 0) is the disc of radius ½ around the centre.
    // Pick where to sample and how blurred, so the lookup covers the part of the blur disc
    // that overlaps the field. A plain lookup at c would, for a blur larger than the gobo,
    // return the whole-gobo average everywhere and light up the full cone near the lens.
    let off = c - 0.5;
    let dc = length(off);
    var pt = c;
    var size = 2.0 * radius;
    if radius >= dc + 0.5 {
        pt = vec2<f32>(0.5); // the whole field is visible
        size = 1.0;
    } else if dc >= radius + 0.5 {
        return vec3<f32>(0.0); // no open direction reaches this point
    } else if dc + radius > 0.5 {
        let h = 0.5 + radius - dc; // how far the blur disc reaches into the field
        pt = 0.5 + off / dc * (0.5 - 0.5 * h);
        size = min(h, 2.0 * radius);
    }
    return gobo_colour_at(i, pt, log2(max(size * GOBO_SIZE, 1.0)));
}

// Colour wheel and gobo transmission at gobo-plane position `pt` (uv: 0..1 across the
// field, field stop = inscribed circle), blurred to mip `lod`. Relative to the open
// aperture: an open beam gives exactly 1.
fn gobo_colour_at(i: u32, pt: vec2<f32>, lod: f32) -> vec3<f32> {
    let l = &lights.items[i];
    let count = u32(l.gobo_params.x);
    var t = vec3<f32>(1.0);
    // Colour wheel near the lamp: far out of focus, so the split has a soft edge.
    if l.split_a.w > 0.5 {
        let edge = (l.split_b.w - 0.5) * l.optics.y + 0.5;
        t *= mix(l.split_b.rgb, l.split_a.rgb, smoothstep(edge - 0.12, edge + 0.12, pt.x));
    }
    if count > 0u {
        // Relative to the open aperture blurred the same way: an open slot gives exactly 1
        // (the beam profile already accounts for the aperture), a gobo gives the fraction
        // of the visible aperture that is clear.
        let open = textureSampleLevel(gobo_maps, gobo_sampler, pt, 0, lod).r;
        for (var g = 0u; g < count; g++) {
            // (Floor: where the aperture is all but closed, tiny edge leakage would be
            // amplified into a bright ring.)
            t *= min(gobo_wheel(l.gobo[g], pt, lod, l.optics.y) / max(open, 0.05), vec3<f32>(1.0));
        }
    }
    return t;
}

// Gobo-plane position (tan units, projector frame) for a world-space direction offset
// `lateral` perpendicular to the beam axis, per unit axial distance.
fn gobo_uv_from_tan(i: u32, lateral_tan: vec3<f32>) -> vec2<f32> {
    let l = &lights.items[i];
    let t = vec2<f32>(dot(lateral_tan, l.basis_x.xyz), dot(lateral_tan, l.basis_y.xyz));
    return t / l.optics.x * vec2<f32>(0.5, -0.5) + 0.5;
}

// Single-tap shadow test without normal offset (for points in the air).
fn shadow_point(i: u32, world: vec3<f32>) -> f32 {
    let l = &lights.items[i];
    let clip = l.view_proj * vec4<f32>(world, 1.0);
    if clip.w <= 0.0 {
        return 0.0;
    }
    let ndc = clip.xyz / clip.w;
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z > 1.0 {
        return 0.0;
    }
    return textureSampleCompareLevel(shadow_maps, shadow_sampler, uv, i32(l.dir.w), ndc.z);
}
