// Adds the (half-resolution) volume to the HDR image with a depth-aware upsample, and
// attenuates the scene behind by the haze. Blend state: out = src.rgb + dst.rgb · src.a.
// (volume_common.wgsl is prepended.)

@group(0) @binding(0) var<uniform> vol: Volume;
@group(0) @binding(1) var scene_depth: texture_depth_2d;
@group(0) @binding(2) var volume_tex: texture_2d<f32>;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let near = vol.misc.x;
    let far = vol.misc.y;
    let full_px = floor(pos.xy);
    let d = textureLoad(scene_depth, vec2<i32>(full_px), 0);
    let z = linear_depth(d, near, far);

    // Bilinear taps on the half-res grid, re-weighted by depth similarity so beams don't
    // bleed across silhouettes.
    let half_size = vec2<i32>(vol.sizes.zw);
    let scale = vol.extra.x;
    let hp = (full_px + 0.5) / scale - 0.5;
    let base = floor(hp);
    let f = hp - base;
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let o = vec2<f32>(f32(i), f32(j));
            let h = clamp(vec2<i32>(base + o), vec2<i32>(0), half_size - 1);
            let wb = select(1.0 - f.x, f.x, i == 1) * select(1.0 - f.y, f.y, j == 1);
            let hz = linear_depth(textureLoad(scene_depth, h * i32(scale), 0), near, far);
            let wd = exp(-abs(hz - z) / (0.02 * z + 0.02));
            let w = wb * wd + 1e-5;
            sum += textureLoad(volume_tex, h, 0).rgb * w;
            wsum += w;
        }
    }
    let world = world_at(vol.inv_view_proj, full_px + 0.5, vol.sizes.xy, d);
    let t = select(distance(world, vol.eye.xyz), vol.misc.w, d >= 1.0);
    return vec4<f32>(sum / wsum, exp(-vol.haze.x * t));
}
