struct Globals { view_proj: mat4x4<f32> };  // prefix of scene Globals; same buffer
@group(0) @binding(0) var<uniform> globals: Globals;

struct VertexOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32> };

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) color: vec4<f32>) -> VertexOut {
    var out: VertexOut;
    out.clip = globals.view_proj * vec4<f32>(position, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> { return in.color; }
