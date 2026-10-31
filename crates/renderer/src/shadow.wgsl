// Depth-only pass from a beam's projector, one layer of the shadow map array per beam.

struct ShadowCamera {
    view_proj: mat4x4<f32>,
};

struct Draw {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
    base_color: vec4<f32>,
    params: vec4<f32>,
    emissive: vec4<f32>,
};

@group(0) @binding(0) var<uniform> shadow_cam: ShadowCamera;
@group(1) @binding(0) var<uniform> draw: Draw;

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return shadow_cam.view_proj * draw.model * vec4<f32>(pos, 1.0);
}
