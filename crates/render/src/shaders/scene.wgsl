// Physically based forward shading (glTF metallic-roughness) and a procedural
// sky as image-based light, into the linear HDR scene target.

struct Globals {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    light_dir: vec4<f32>,   // xyz: direction sunlight travels; w: exposure
    light_color: vec4<f32>, // sun radiance
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    sky_ground: vec4<f32>,
    // Sky irradiance: Σ sh[i] · Y_i(n) (see `sky.rs`).
    sh: array<vec4<f32>, 9>,
    // x: 1 to tonemap in this shader (no HDR target), 0 to leave it to post
    post: vec4<f32>,
    // Sun shadows (see `shadow.rs`): world → each cascade's light clip space.
    cascades: array<mat4x4<f32>, 4>,
    // xyz: camera forward (view depth runs along it from the eye);
    // w: the view depth where cascade 0 starts
    shadow_camera: vec4<f32>,
    // The view depth where each cascade ends.
    cascade_far: vec4<f32>,
    // The world size of a shadow texel in each cascade.
    cascade_texel: vec4<f32>,
    // x: cascades in use; y: 1 to tint each cascade (debug)
    shadow_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var shadow_map: texture_depth_2d_array; // a layer per cascade
@group(0) @binding(2) var shadow_sampler: sampler_comparison;

// The shadow passes' only binding: the cascade being drawn.
struct Caster {
    view_proj: mat4x4<f32>,
};
@group(3) @binding(0) var<uniform> caster: Caster;

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
    @location(4) uv: vec2<f32>,
    @location(5) tangent: vec4<f32>,
    @location(6) pbr: vec2<f32>,
};

fn model_matrix(i: InstanceIn) -> mat4x4<f32> {
    return mat4x4<f32>(i.model_0, i.model_1, i.model_2, i.model_3);
}

// The lit pass's vertex output for a vertex already in the model's space.
fn shade_vertex(position: vec3<f32>, normal: vec3<f32>, uv: vec2<f32>, tangent: vec4<f32>, i: InstanceIn) -> VertexOut {
    let model = model_matrix(i);
    let world = model * vec4<f32>(position, 1.0);
    let normal_matrix = mat3x3<f32>(i.normal_0.xyz, i.normal_1.xyz, i.normal_2.xyz);
    var out: VertexOut;
    out.clip = globals.view_proj * world;
    out.world_pos = world.xyz;
    out.normal = normal_matrix * normal;
    out.color = i.color;
    out.uv = uv;
    // Tangents follow the surface, so they take the model matrix itself.
    out.tangent = vec4<f32>((model * vec4<f32>(tangent.xyz, 0.0)).xyz, tangent.w);
    out.pbr = i.pbr.xy;
    return out;
}

@vertex
fn vs_main(v: VertexIn, i: InstanceIn) -> VertexOut {
    return shade_vertex(v.position, v.normal, v.uv, v.tangent, i);
}

@vertex
fn vs_shadow(v: VertexIn, i: InstanceIn) -> @builtin(position) vec4<f32> {
    return caster.view_proj * model_matrix(i) * vec4<f32>(v.position, 1.0);
}

// Skinning. Joint matrices live in a float texture (WebGL2 has no storage
// buffers in the vertex stage): four texels, the columns, per matrix, rows
// JOINT_TEXTURE_WIDTH texels wide. Must match `draw_list::JOINT_TEXTURE_WIDTH`.
@group(2) @binding(0) var joint_matrices: texture_2d<f32>;
const JOINT_TEXTURE_WIDTH: u32 = 1024u;

struct SkinnedVertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(10) uv: vec2<f32>,
    @location(11) tangent: vec4<f32>,
    @location(13) joints: vec4<u32>,
    @location(14) weights: vec4<f32>,
};

fn joint_matrix(index: u32) -> mat4x4<f32> {
    let texel = index * 4u;
    let x = i32(texel % JOINT_TEXTURE_WIDTH);
    let y = i32(texel / JOINT_TEXTURE_WIDTH);
    return mat4x4<f32>(
        textureLoad(joint_matrices, vec2<i32>(x, y), 0),
        textureLoad(joint_matrices, vec2<i32>(x + 1, y), 0),
        textureLoad(joint_matrices, vec2<i32>(x + 2, y), 0),
        textureLoad(joint_matrices, vec2<i32>(x + 3, y), 0),
    );
}

// The weighted blend of a vertex's joint matrices; `first` is where the
// instance's matrices start.
fn skin_matrix(v: SkinnedVertexIn, first: u32) -> mat4x4<f32> {
    return joint_matrix(first + v.joints.x) * v.weights.x
        + joint_matrix(first + v.joints.y) * v.weights.y
        + joint_matrix(first + v.joints.z) * v.weights.z
        + joint_matrix(first + v.joints.w) * v.weights.w;
}

@vertex
fn vs_skinned(v: SkinnedVertexIn, i: InstanceIn, @location(15) first_joint: u32) -> VertexOut {
    let skin = skin_matrix(v, first_joint);
    // Joints only rotate, translate and scale uniformly, so their 3x3 part
    // carries normals as well as positions.
    let skin3 = mat3x3<f32>(skin[0].xyz, skin[1].xyz, skin[2].xyz);
    return shade_vertex(
        (skin * vec4<f32>(v.position, 1.0)).xyz,
        skin3 * v.normal,
        v.uv,
        vec4<f32>(skin3 * v.tangent.xyz, v.tangent.w),
        i,
    );
}

@vertex
fn vs_shadow_skinned(v: SkinnedVertexIn, i: InstanceIn, @location(15) first_joint: u32) -> @builtin(position) vec4<f32> {
    let skin = skin_matrix(v, first_joint);
    return caster.view_proj * model_matrix(i) * skin * vec4<f32>(v.position, 1.0);
}

// Sun visibility at `p` from cascade `i`: 3x3 PCF on top of the hardware 2x2
// comparison filter; 1 outside the cascade. The Level variant is allowed in
// non-uniform control flow (no implicit derivatives).
fn cascade_visibility(i: u32, p: vec3<f32>) -> f32 {
    let clip = globals.cascades[i] * vec4<f32>(p, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z > 1.0) {
        return 1.0;
    }
    let texel = 1.0 / vec2<f32>(textureDimensions(shadow_map));
    var sum = 0.0;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + offset, i32(i), ndc.z);
        }
    }
    return sum / 9.0;
}

// How far along the normal a position moves before its shadow lookup, in
// texels of the cascade: keeps lit surfaces from shadowing themselves.
const NORMAL_OFFSET: f32 = 1.5;
// Each cascade blends into the next over this last fraction of its depths;
// the last one fades to unshadowed.
const CASCADE_BLEND: f32 = 0.1;

fn offset_visibility(i: u32, world: vec3<f32>, normal: vec3<f32>) -> f32 {
    return cascade_visibility(i, world + normal * globals.cascade_texel[i] * NORMAL_OFFSET);
}

// x: sun visibility at `world` (unit geometric `normal`); y: the cascade it
// came from, fractional while blending into the next.
fn sun_visibility(world: vec3<f32>, normal: vec3<f32>) -> vec2<f32> {
    let count = u32(globals.shadow_params.x);
    let depth = dot(world - globals.eye.xyz, globals.shadow_camera.xyz);
    var start = globals.shadow_camera.w;
    for (var i = 0u; i < count; i += 1u) {
        let end = globals.cascade_far[i];
        if (depth <= end) {
            var visibility = offset_visibility(i, world, normal);
            let band = (end - start) * CASCADE_BLEND;
            let t = clamp((depth - (end - band)) / band, 0.0, 1.0);
            if (t > 0.0) {
                var next = 1.0;
                if (i + 1u < count) {
                    next = offset_visibility(i + 1u, world, normal);
                }
                visibility = mix(visibility, next, t);
            }
            return vec2<f32>(visibility, f32(i) + t);
        }
        start = end;
    }
    return vec2<f32>(1.0, f32(count));
}

// Debug tint for a (fractional) cascade index: red, green, blue, yellow,
// fading to white where the last cascade fades out.
fn cascade_tint(c: f32, count: f32) -> vec3<f32> {
    let palette = array<vec3<f32>, 4>(
        vec3<f32>(1.0, 0.35, 0.35),
        vec3<f32>(0.35, 1.0, 0.35),
        vec3<f32>(0.4, 0.55, 1.0),
        vec3<f32>(1.0, 0.9, 0.3),
    );
    let i = u32(clamp(floor(c), 0.0, 3.0));
    var next = vec3<f32>(1.0);
    if (f32(i) + 1.0 < count) {
        next = palette[i + 1u];
    }
    return mix(palette[i], next, fract(c));
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

// Linear HDR for the post pass, capped below half float's maximum (a sharp
// highlight would otherwise store infinity and bloom would smear it); or
// tonemapped here when the target can't hold HDR.
fn finish(color: vec3<f32>) -> vec3<f32> {
    if (globals.post.x > 0.5) {
        return tonemap(color);
    }
    return min(color, vec3<f32>(60000.0));
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
    let sun = sun_visibility(in.world_pos, geometric);
    let direct = (diffuse + specular) * globals.light_color.rgb * n_dot_l * sun.x;

    // Sky: SH irradiance for diffuse; the reflected sky, blurred toward the
    // irradiance as roughness grows, for specular.
    let irradiance = sky_irradiance(n);
    let reflected = reflect(-v, n);
    let blur = roughness * roughness;
    let prefiltered = mix(sky_radiance(reflected), irradiance / PI, blur);
    let ab = env_brdf(n_dot_v, roughness);
    let ambient = (diffuse_color * irradiance / PI + prefiltered * (f0 * ab.x + ab.y)) * occlusion;

    var color = (direct + ambient + emissive) * globals.light_dir.w;
    if (globals.shadow_params.y > 0.5 && sun.y < globals.shadow_params.x) {
        color *= cascade_tint(sun.y, globals.shadow_params.x);
    }
    return vec4<f32>(finish(color), base.a);
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
    return vec4<f32>(finish(radiance * globals.light_dir.w), 1.0);
}
