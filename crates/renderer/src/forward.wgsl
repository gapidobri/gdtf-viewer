// Forward pass into the HDR target. Everything is in physical units: lights in candela,
// illuminance in lux, output radiance in cd/m² (nits). Exposure and tonemapping happen
// in tonemap.wgsl.
// (common.wgsl is prepended.)

struct Draw {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
    base_color: vec4<f32>,
    // x = metallic, y = roughness, z = lens of light (index + 1, 0 = not a lens),
    // w = lens luminance scale
    params: vec4<f32>,
    emissive: vec4<f32>,
};


@group(1) @binding(0) var<uniform> draw: Draw;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var o: VsOut;
    let world = draw.model * vec4<f32>(v.pos, 1.0);
    o.world = world.xyz;
    o.normal = (draw.normal * vec4<f32>(v.normal, 0.0)).xyz;
    o.clip = cam.view_proj * world;
    return o;
}

fn d_ggx(n_h: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = n_h * n_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn v_smith(n_v: f32, n_l: f32, a: f32) -> f32 {
    let k = a * 0.5;
    return 0.25 / ((n_v * (1.0 - k) + k) * (n_l * (1.0 - k) + k));
}

// BRDF × cos: multiply by illuminance (lux) to get radiance (nits).
fn shade(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, albedo: vec3<f32>, metallic: f32, rough: f32) -> vec3<f32> {
    let h = normalize(v + l);
    let n_l = max(dot(n, l), 0.0);
    let n_v = max(dot(n, v), 1e-4);
    let n_h = max(dot(n, h), 0.0);
    let v_h = max(dot(v, h), 0.0);
    let a = max(rough * rough, 0.002);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let f = f0 + (1.0 - f0) * pow(1.0 - v_h, 5.0);
    let spec = d_ggx(n_h, a) * v_smith(n_v, n_l, a) * f;
    let diff = (1.0 - f) * (1.0 - metallic) * albedo / PI;
    return (diff + spec) * n_l;
}

@fragment
fn fs_main(i: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    var n = normalize(i.normal);
    if !front { n = -n; } // fixture meshes are often not consistently wound
    let v = normalize(cam.eye.xyz - i.world);
    let albedo = draw.base_color.rgb;
    let metallic = draw.params.x;
    let rough = draw.params.y;

    // House lights: soft light from above plus a frontal work light, both in lux.
    let house = cam.params.x;
    var c = albedo * (1.0 - metallic) / PI * house * (0.3 + 0.4 * (n.y * 0.5 + 0.5));
    c += shade(n, v, normalize(vec3<f32>(0.3, 0.8, 0.6)), albedo, metallic, rough) * house * 0.5;

    let count = u32(cam.params.y);
    for (var li = 0u; li < count; li++) {
        let l = &lights.items[li]; // pointer: copying the 1.3 KB struct spills registers
        if l.color.w < 0.5 {
            continue;
        }
        let to_p = i.world - l.apex.xyz;
        let light = beam_light(li, i.world);
        if max(light.r, max(light.g, light.b)) <= 0.0 {
            continue;
        }
        let ldir = normalize(l.lens.xyz - i.world);
        // Inverse square from the beam's virtual apex, not the lens: the beam's cross
        // section grows from the lens aperture, so near the lens the illuminance tends to
        // flux / aperture area instead of diverging.
        let lux = shadow(li, i.world, n) / max(dot(to_p, to_p), 1e-4);
        if lux <= 0.0 {
            continue;
        }
        c += shade(n, v, ldir, albedo, metallic, rough) * l.color.rgb * light * lux;
    }

    // Lens. Seen from the eye, lens point q shows the gobo plane at (tan units)
    //     eye/d + q·(1/d_focus − 1/d)
    // (eye and q lateral to the axis, d = the eye's distance along it): a magnified, out
    // of focus view of the gobo, sharpening as the eye nears the focus distance. Peak
    // luminance = intensity / lens area, shaped by the beam profile in that direction.
    let lens = u32(draw.params.z);
    if lens > 0u {
        let li = lens - 1u;
        let l = &lights.items[li]; // pointer: copying the 1.3 KB struct spills registers
        let axis = l.dir.xyz;
        let eye_rel = cam.eye.xyz - l.lens.xyz;
        let d = dot(eye_rel, axis);
        if d > 1e-3 {
            let eye_lat = eye_rel - axis * d;
            let q_rel = i.world - l.lens.xyz;
            let q_lat = q_rel - axis * dot(q_rel, axis);
            let g = eye_lat / d + q_lat * (l.gobo_params.z - 1.0 / d);
            let theta = atan(length(g));
            // A small floor: off-axis you still see the lit optics through the lens.
            let prof = max(profile(li, theta), 0.002);
            // Sharp: the viewer's pupil is tiny compared to the lens, so the gobo seen in
            // the lens is barely blurred (unlike the beam, which the whole lens forms).
            let pt = gobo_uv_from_tan(li, g);
            let filt = gobo_colour_at(li, clamp(pt, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
            // params.w: this lens's share when several beams render as one light.
            c += l.color.rgb * filt * prof / (PI * l.lens.w * l.lens.w) * draw.params.w;
        }
    }
    c += draw.emissive.rgb;

    return vec4<f32>(c, 1.0);
}
