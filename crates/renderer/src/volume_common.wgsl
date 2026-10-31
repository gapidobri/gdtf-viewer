// Parameters shared by the volumetric passes (group 1 in volume.wgsl, group 0 elsewhere).
struct Volume {
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    // x = haze scattering coefficient σs (1/m), y = phase anisotropy g, z = steps, w = frame
    haze: vec4<f32>,
    // xy = full-resolution size, zw = half-resolution size
    sizes: vec4<f32>,
    // x = near, y = far, z = history blend weight (1 = no history), w = max march distance
    misc: vec4<f32>,
};

// World position of a full-resolution pixel at depth-buffer value `d`.
fn world_at(inv_vp: mat4x4<f32>, pixel: vec2<f32>, full: vec2<f32>, d: f32) -> vec3<f32> {
    let uv = pixel / full;
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d);
    let w = inv_vp * vec4<f32>(ndc, 1.0);
    return w.xyz / w.w;
}

fn linear_depth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
