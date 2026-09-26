// Physically based forward shading (glTF metallic-roughness), a procedural
// sky as image-based light, and in-shader tonemapping to the sRGB target.

struct Globals {
    view_proj: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    light_dir: vec4<f32>,   // xyz: direction sunlight travels; w: exposure
    light_color: vec4<f32>, // sun radiance
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    sky_ground: vec4<f32>,
    // Sky irradiance: Σ sh[i] · Y_i(n) (see `sky.rs`).
    sh: array<vec4<f32>, 9>,
};
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;

struct Material {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    // metallic, roughness, normal scale, occlusion strength
    params: vec4<f32>,
};
@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var base_texture: texture_2d<f32>;      // sRGB
@group(1) @binding(2) var mr_texture: texture_2d<f32>;        // G roughness, B metallic
@group(1) @binding(3) var normal_texture: texture_2d<f32>;    // tangent space
@group(1) @binding(4) var occlusion_texture: texture_2d<f32>; // R
@group(1) @binding(5) var emissive_texture: texture_2d<f32>;  // sRGB
@group(1) @binding(6) var material_sampler: sampler;

const PI: f32 = 3.14159265;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(10) uv: vec2<f32>,
    @location(11) tangent: vec4<f32>,
};
struct InstanceIn {
    @location(2) model_0: vec4<f32>, @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>, @location(5) model_3: vec4<f32>,
    @location(6) normal_0: vec4<f32>, @location(7) normal_1: vec4<f32>,
    @location(8) normal_2: vec4<f32>, @location(9) color: vec4<f32>,
    // x: roughness, y: metallic (multiply the material's factors)
    @location(12) pbr: vec4<f32>,
};
struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) light_clip: vec4<f32>,
    @location(4) uv: vec2<f32>,
    @location(5) tangent: vec4<f32>,
    @location(6) pbr: vec2<f32>,
};

fn model_matrix(i: InstanceIn) -> mat4x4<f32> {
    return mat4x4<f32>(i.model_0, i.model_1, i.model_2, i.model_3);
}

@vertex
fn vs_main(v: VertexIn, i: InstanceIn) -> VertexOut {
    let model = model_matrix(i);
    let world = model * vec4<f32>(v.position, 1.0);
    let normal_matrix = mat3x3<f32>(i.normal_0.xyz, i.normal_1.xyz, i.normal_2.xyz);
    var out: VertexOut;
    out.clip = globals.view_proj * world;
    out.world_pos = world.xyz;
    out.normal = normal_matrix * v.normal;
    out.color = i.color;
    out.light_clip = globals.light_view_proj * world;
    out.uv = v.uv;
    // Tangents follow the surface, so they take the model matrix itself.
    out.tangent = vec4<f32>((model * vec4<f32>(v.tangent.xyz, 0.0)).xyz, v.tangent.w);
    out.pbr = i.pbr.xy;
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

// Mirrors `Sky::radiance` in sky.rs (without the sun disc).
fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    let y = dir.y;
    if (y >= 0.0) {
        return mix(globals.sky_horizon.rgb, globals.sky_zenith.rgb, pow(y, 0.5));
    }
    let t = clamp(-y / 0.6, 0.0, 1.0);
    return mix(globals.sky_horizon.rgb, globals.sky_ground.rgb, t * t * (3.0 - 2.0 * t));
}

// Cosine-convolved sky light arriving at a surface facing `n`.
fn sky_irradiance(n: vec3<f32>) -> vec3<f32> {
    let s = globals.sh;
    var e = s[0].rgb * 0.282095;
    e += s[1].rgb * (0.488603 * n.y) + s[2].rgb * (0.488603 * n.z) + s[3].rgb * (0.488603 * n.x);
    e += s[4].rgb * (1.092548 * n.x * n.y) + s[5].rgb * (1.092548 * n.y * n.z);
    e += s[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0));
    e += s[7].rgb * (1.092548 * n.x * n.z) + s[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));
    return max(e, vec3<f32>(0.0));
}

// Khronos PBR Neutral: colours up to mid-brightness pass through unchanged;
// highlights are compressed toward white.
fn tonemap(color: vec3<f32>) -> vec3<f32> {
    let start = 0.8 - 0.04;
    let desaturation = 0.15;
    let x = min(color.r, min(color.g, color.b));
    let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
    var c = color - offset;
    let peak = max(c.r, max(c.g, c.b));
    if (peak < start) {
        return c;
    }
    let d = 1.0 - start;
    let new_peak = 1.0 - d * d / (peak + d - start);
    c = c * (new_peak / peak);
    let g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return mix(c, vec3<f32>(new_peak), g);
}

// Karis' analytic fit of the split-sum environment BRDF (scale, bias on F0).
fn env_brdf(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Every texture is sampled first, in uniform control flow (implicit
    // derivatives for mips).
    let base_sample = textureSample(base_texture, material_sampler, in.uv);
    let mr = textureSample(mr_texture, material_sampler, in.uv);
    let normal_sample = textureSample(normal_texture, material_sampler, in.uv).xyz;
    let occlusion_sample = textureSample(occlusion_texture, material_sampler, in.uv).r;
    let emissive_sample = textureSample(emissive_texture, material_sampler, in.uv).rgb;

    let base = base_sample * material.base_color * in.color;
    let metallic = clamp(material.params.x * mr.b * in.pbr.y, 0.0, 1.0);
    let roughness = clamp(material.params.y * mr.g * in.pbr.x, 0.045, 1.0);

    // Normal mapping in tangent space (flat default map: the vertex normal).
    let geometric = normalize(in.normal);
    var n = geometric;
    let t_raw = in.tangent.xyz - geometric * dot(geometric, in.tangent.xyz);
    if (dot(t_raw, t_raw) > 1e-8) {
        let t = normalize(t_raw);
        let b = cross(geometric, t) * select(1.0, -1.0, in.tangent.w < 0.0);
        var tn = normal_sample * 2.0 - 1.0;
        tn = vec3<f32>(tn.xy * material.params.z, tn.z);
        n = normalize(t * tn.x + b * tn.y + geometric * tn.z);
    }
    let occlusion = 1.0 + material.params.w * (occlusion_sample - 1.0);
    let emissive = emissive_sample * material.emissive.rgb;

    let v = normalize(globals.eye.xyz - in.world_pos);
    let n_dot_v = max(dot(n, v), 1e-4);
    let f0 = mix(vec3<f32>(0.04), base.rgb, metallic);
    let diffuse_color = base.rgb * (1.0 - metallic);

    // Sun: Cook-Torrance (GGX, height-correlated Smith, Schlick).
    let l = -globals.light_dir.xyz;
    let h = normalize(l + v);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    let distribution = a2 / (PI * denom * denom);
    let visibility = 0.5 / max(
        n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2)
            + n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2),
        1e-5,
    );
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);
    let specular = distribution * visibility * fresnel;
    let diffuse = (1.0 - fresnel) * diffuse_color / PI;
    let shadow = shadow_factor(in.light_clip);
    let direct = (diffuse + specular) * globals.light_color.rgb * n_dot_l * shadow;

    // Sky: SH irradiance for diffuse; the reflected sky, blurred toward the
    // irradiance as roughness grows, for specular.
    let irradiance = sky_irradiance(n);
    let reflected = reflect(-v, n);
    let blur = roughness * roughness;
    let prefiltered = mix(sky_radiance(reflected), irradiance / PI, blur);
    let ab = env_brdf(n_dot_v, roughness);
    let ambient = (diffuse_color * irradiance / PI + prefiltered * (f0 * ab.x + ab.y)) * occlusion;

    let color = (direct + ambient + emissive) * globals.light_dir.w;
    return vec4<f32>(tonemap(color), base.a);
}

// The sky behind everything: a fullscreen triangle at the far plane, drawn
// after the scene so only uncovered pixels shade.
struct SkyOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) index: u32) -> SkyOut {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    let ndc = corner * 2.0 - 1.0;
    var out: SkyOut;
    out.clip = vec4<f32>(ndc, 1.0, 1.0);
    out.ndc = ndc;
    return out;
}

@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
    let far = globals.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - globals.eye.xyz);
    var radiance = sky_radiance(dir);
    // The sun's disc (about 1.5° across) with a soft glow around it.
    let to_sun = -globals.light_dir.xyz;
    let c = dot(dir, to_sun);
    radiance += globals.light_color.rgb * (smoothstep(0.99965, 0.99985, c) * 12.0 + pow(max(c, 0.0), 400.0) * 0.4);
    return vec4<f32>(tonemap(radiance * globals.light_dir.w), 1.0);
}
