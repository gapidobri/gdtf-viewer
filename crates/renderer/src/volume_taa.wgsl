// Temporal accumulation for the half-resolution volume: reproject last frame's result by
// scene depth, clamp it to the current 3×3 neighbourhood (limits ghosting when beams move)
// and blend. (volume_common.wgsl is prepended.)

@group(0) @binding(0) var<uniform> vol: Volume;
@group(0) @binding(1) var scene_depth: texture_depth_2d;
@group(0) @binding(2) var current: texture_2d<f32>;
@group(0) @binding(3) var history: texture_2d<f32>;
@group(0) @binding(4) var linear_sampler: sampler;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(pos.xy));
    let size = vec2<i32>(vol.sizes.zw);
    let cur = textureLoad(current, p, 0);
    if vol.misc.z >= 1.0 {
        return cur;
    }
    var lo = cur;
    var hi = cur;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let s = textureLoad(current, clamp(p + vec2<i32>(x, y), vec2<i32>(0), size - 1), 0);
            lo = min(lo, s);
            hi = max(hi, s);
        }
    }
    let full_px = vec2<f32>(p) * vol.extra.x;
    let d = textureLoad(scene_depth, vec2<i32>(full_px), 0);
    let world = world_at(vol.inv_view_proj, full_px + 0.5, vol.sizes.xy, d);
    let prev = vol.prev_view_proj * vec4<f32>(world, 1.0);
    let uv = prev.xy / prev.w * vec2<f32>(0.5, -0.5) + 0.5;
    if prev.w <= 0.0 || any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return cur;
    }
    let hist = clamp(textureSampleLevel(history, linear_sampler, uv, 0.0), lo, hi);
    return mix(hist, cur, vol.misc.z);
}
