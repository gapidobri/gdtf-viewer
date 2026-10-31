// Volumetric beams: per pixel (half resolution) and per beam, intersect the view ray with
// the beam's cone analytically and ray-march the segment in front of the scene surface.
// Each step uses the same profile, gobo/colour filter and shadow map as surface lighting.
// (common.wgsl and volume_common.wgsl are prepended.)

@group(1) @binding(0) var<uniform> vol: Volume;
@group(1) @binding(1) var scene_depth: texture_depth_2d;

// Interleaved gradient noise (Jimenez 2014): cheap, well distributed, TAA friendly.
fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4), 1.5));
}

// Ray parameter interval [t0, t1] where o + t·v is inside the forward cone (apex, axis,
// half-angle with cosine `cos_a`) and between `near_ax`..`far_ax` along the axis.
// Returns t0 >= t1 when there is no overlap.
fn cone_interval(o: vec3<f32>, v: vec3<f32>, apex: vec3<f32>, axis: vec3<f32>, cos_a: f32,
                 near_ax: f32, far_ax: f32) -> vec2<f32> {
    let co = o - apex;
    let dv = dot(v, axis);
    let dco = dot(co, axis);
    var t0 = 0.0;
    var t1 = 1e9;

    // Slab along the axis.
    if abs(dv) < 1e-6 {
        if dco < near_ax || dco > far_ax {
            return vec2<f32>(1.0, 0.0);
        }
    } else {
        let ta = (near_ax - dco) / dv;
        let tb = (far_ax - dco) / dv;
        t0 = max(t0, min(ta, tb));
        t1 = min(t1, max(ta, tb));
    }

    // Double cone: q(t) = (axial)² − |co + t v|² cos² ≥ 0. The slab keeps us on the
    // forward nappe, which is convex, so the result is a single interval.
    let cos2 = cos_a * cos_a;
    let a = dv * dv - cos2;
    let b = 2.0 * (dv * dco - dot(v, co) * cos2);
    let c = dco * dco - dot(co, co) * cos2;
    if abs(a) < 1e-7 {
        if abs(b) < 1e-9 {
            if c < 0.0 { return vec2<f32>(1.0, 0.0); }
        } else if b > 0.0 {
            t0 = max(t0, -c / b);
        } else {
            t1 = min(t1, -c / b);
        }
        return vec2<f32>(t0, t1);
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        if a < 0.0 { return vec2<f32>(1.0, 0.0); }
        return vec2<f32>(t0, t1);
    }
    let s = sqrt(disc);
    let r0 = min((-b - s) / (2.0 * a), (-b + s) / (2.0 * a));
    let r1 = max((-b - s) / (2.0 * a), (-b + s) / (2.0 * a));
    if a < 0.0 {
        // Inside between the roots.
        return vec2<f32>(max(t0, r0), min(t1, r1));
    }
    // Inside outside the roots: (-inf, r0] or [r1, inf); take the one within the slab.
    let lo = vec2<f32>(t0, min(t1, r0));
    let hi = vec2<f32>(max(t0, r1), t1);
    if hi.x < hi.y {
        return hi;
    }
    return lo;
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let half_px = floor(pos.xy);
    // Each half-res pixel uses the depth of its top-left full-res pixel (the composite
    // pass reads the same pixel for its depth-aware upsample).
    let full_px = half_px * vol.extra.x;
    let d = textureLoad(scene_depth, vec2<i32>(full_px), 0);
    let eye = vol.eye.xyz;
    let surface = world_at(vol.inv_view_proj, full_px + 0.5, vol.sizes.xy, d);
    let v = normalize(surface - eye);
    let t_surface = select(distance(surface, eye), vol.misc.w, d >= 1.0);

    let sigma = vol.haze.x;
    let g = vol.haze.y;
    let steps = vol.haze.z;
    let jitter = ign(half_px + 5.588238 * vol.haze.w);

    var radiance = vec3<f32>(0.0);
    let count = u32(cam.params.y);
    for (var li = 0u; li < count; li++) {
        let l = &lights.items[li];
        if l.color.w < 0.5 || dot(l.color.rgb, vec3<f32>(1.0)) <= 0.0 {
            continue;
        }
        let axis = l.dir.xyz;
        let apex = l.apex.xyz;
        let lens_ax = dot(l.lens.xyz - apex, axis);
        // The cone holding all of the beam's light, prism images included.
        let seg = cone_interval(eye, v, apex, axis, cos(l.prism_params.y), lens_ax, vol.misc.w);
        let t0 = seg.x;
        let t1 = min(seg.y, t_surface);
        if t0 >= t1 {
            continue;
        }
        // Steps follow the length of the segment: ~8 cm per step through gobo shafts
        // (sharp detail), ~25 cm through plain beams, capped by the quality setting.
        // Short crossings and wide floods get cheap.
        let has_detail = l.gobo_params.x > 0.5 || l.split_a.w > 0.5 || l.prism_params.x > 0.5;
        let step_len = select(0.25, 0.08, has_detail);
        // Plain beams are smooth along the ray: 3/8 of the quality setting is plenty once
        // temporal smoothing averages the jittered samples.
        let max_steps = select(max(4.0, ceil(steps * 0.375)), steps, has_detail);
        let n = clamp(ceil((t1 - t0) / step_len), 4.0, max_steps);
        let dt = (t1 - t0) / n;
        // Prisms with many facets: half of them per frame, alternating.
        let facet_stride = select(1u, 2u, l.prism_params.x > 8.5);
        let facet_first = u32(vol.haze.w) % 2u;
        var acc = vec3<f32>(0.0);
        for (var k = 0.0; k < n; k += 1.0) {
            let t = t0 + (k + jitter) * dt;
            let p = eye + v * t;
            let to_p = p - apex;
            let light = beam_light_facets(li, p, facet_stride, facet_first);
            if max(light.r, max(light.g, light.b)) <= 0.0 {
                continue;
            }
            let vis = shadow_point(li, p);
            if vis <= 0.0 {
                continue;
            }
            // Light travels lens → p, then scatters towards the eye (−v).
            let phase = henyey_greenstein(dot(normalize(p - l.lens.xyz), -v), g);
            // Inverse square from the virtual apex (see forward.wgsl): finite at the lens,
            // where it equals flux / aperture area.
            let lux = vis / max(dot(to_p, to_p), 1e-4);
            // Haze between the eye and p attenuates what reaches us.
            acc += light * (lux * phase * exp(-sigma * t));
        }
        radiance += l.color.rgb * acc * (sigma * dt);
    }
    return vec4<f32>(radiance, 1.0);
}
