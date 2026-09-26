struct Globals {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    light_dir: vec4<f32>,   // direction light travels, normalized
    light_color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
// Per material: base colour (sRGB texture, so samples arrive linear).
@group(1) @binding(0) var base_texture: texture_2d<f32>;
@group(1) @binding(1) var base_sampler: sampler;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(10) uv: vec2<f32>,
};
struct InstanceIn {
    @location(2) model_0: vec4<f32>, @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>, @location(5) model_3: vec4<f32>,
    @location(6) normal_0: vec4<f32>, @location(7) normal_1: vec4<f32>,
    @location(8) normal_2: vec4<f32>, @location(9) color: vec4<f32>,
};
struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) light_clip: vec4<f32>,
    @location(4) uv: vec2<f32>,
};

fn model_matrix(i: InstanceIn) -> mat4x4<f32> {
    return mat4x4<f32>(i.model_0, i.model_1, i.model_2, i.model_3);
}

@vertex
fn vs_main(v: VertexIn, i: InstanceIn) -> VertexOut {
    let world = model_matrix(i) * vec4<f32>(v.position, 1.0);
    let normal_matrix = mat3x3<f32>(i.normal_0.xyz, i.normal_1.xyz, i.normal_2.xyz);
    var out: VertexOut;
    out.clip = globals.view_proj * world;
    out.world_pos = world.xyz;
    out.normal = normal_matrix * v.normal;
    out.color = i.color;
    out.light_clip = globals.light_view_proj * world;
    out.uv = v.uv;
    return out;
}

@vertex
fn vs_shadow(v: VertexIn, i: InstanceIn) -> @builtin(position) vec4<f32> {
    return globals.light_view_proj * model_matrix(i) * vec4<f32>(v.position, 1.0);
}

fn shadow_factor(light_clip: vec4<f32>) -> f32 {
    let ndc = light_clip.xyz / light_clip.w;
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z > 1.0) {
        return 1.0;
    }
    // 3x3 PCF on top of the hardware 2x2 comparison filter. The Level variant is
    // allowed in non-uniform control flow (no implicit derivatives).
    let texel = 1.0 / vec2<f32>(textureDimensions(shadow_map));
    var sum = 0.0;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + offset, ndc.z);
        }
    }
    return sum / 9.0;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Sampled first, in uniform control flow (implicit derivatives for mips).
    let base = textureSample(base_texture, base_sampler, in.uv) * in.color;
    let n = normalize(in.normal);
    let l = -globals.light_dir.xyz;
    let v = normalize(globals.eye.xyz - in.world_pos);
    let h = normalize(l + v);
    let shadow = shadow_factor(in.light_clip);
    let n_dot_l = max(dot(n, l), 0.0);
    let specular = pow(max(dot(n, h), 0.0), 32.0) * 0.25 * step(0.0001, n_dot_l);
    let ambient = mix(vec3<f32>(0.22, 0.20, 0.18), vec3<f32>(0.50, 0.60, 0.75), n.y * 0.5 + 0.5) * 0.35;
    let light = globals.light_color.rgb * shadow;
    let rgb = base.rgb * (ambient + light * n_dot_l) + light * specular;
    return vec4<f32>(rgb, base.a);
}
