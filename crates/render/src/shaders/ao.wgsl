// Ground-truth ambient occlusion (Jimenez et al. 2016, "Practical Real-Time
// Strategies for Accurate Indirect Occlusion") from the prepass's linear
// depth and normals, then a depth-aware 4x4 blur that averages out its 4x4
// noise.

struct AoUniforms {
    // The camera's view-space axes in world space (view space looks down -z).
    right: vec4<f32>,
    up: vec4<f32>,
    back: vec4<f32>,
    // x: radius (m); y: pixels per metre at view depth 1; z: intensity (exponent)
    params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: AoUniforms;
// View depth along the camera's forward axis; 0 where nothing was drawn.
// A colour target, not the depth buffer: WebGL2 can't reliably texelFetch
// depth textures, and linear depth keeps full precision far away.
@group(0) @binding(1) var view_depth: texture_2d<f32>;
@group(0) @binding(2) var normal_texture: texture_2d<f32>; // world normal * 0.5 + 0.5
@group(0) @binding(3) var raw_ao: texture_2d<f32>;          // blur input

const PI: f32 = 3.14159265;
const SLICES: u32 = 2u;
const STEPS: u32 = 4u;
// Screen-space radius cap: wider costs cache misses for little change.
const MAX_RADIUS_PX: f32 = 128.0;
// Samples fade out over this outer fraction of the radius.
const FALLOFF: f32 = 0.4;

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

fn depth_at(px: vec2<i32>) -> f32 {
    let size = vec2<i32>(textureDimensions(view_depth));
    return textureLoad(view_depth, clamp(px, vec2<i32>(0), size - 1), 0).r;
}

// The view-space position of pixel `px` at view depth `z`.
fn view_position(px: vec2<i32>, z: f32) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(view_depth));
    let from_center = vec2<f32>(px) + 0.5 - size * 0.5;
    return vec3<f32>(from_center.x, -from_center.y, -u.params.y) * (z / u.params.y);
}

// A 4x4 Bayer index in [0, 1): spreads 16 rotations over every 4x4 block.
fn bayer4(px: vec2<i32>) -> f32 {
    let m = array<f32, 16>(0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
    let p = vec2<u32>(px) & vec2<u32>(3u);
    return m[p.y * 4u + p.x] / 16.0;
}

// The cosine-weighted visible arc between the normal's angle `n` and a
// horizon at angle `h`, both in the slice plane.
fn visible_arc(h: f32, n: f32) -> f32 {
    return 0.25 * (-cos(2.0 * h - n) + cos(n) + 2.0 * h * sin(n));
}

@fragment
fn fs_gtao(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(frag.xy);
    let z = depth_at(px);
    if (z <= 0.0) {
        return vec4<f32>(1.0); // sky
    }
    let p = view_position(px, z);
    let v = normalize(-p);
    let world_n = textureLoad(normal_texture, px, 0).xyz * 2.0 - 1.0;
    let n = normalize(vec3<f32>(
        dot(world_n, u.right.xyz),
        dot(world_n, u.up.xyz),
        dot(world_n, u.back.xyz),
    ));

    let radius = u.params.x;
    let radius_px = min(radius * u.params.y / z, MAX_RADIUS_PX);
    if (radius_px < 1.0) {
        return vec4<f32>(1.0); // too far away to resolve
    }
    let rotation = bayer4(px);
    let jitter = fract(rotation * 4.0) + 0.125;

    var visibility = 0.0;
    for (var s = 0u; s < SLICES; s += 1u) {
        let phi = PI * (f32(s) + rotation) / f32(SLICES);
        let dir = vec2<f32>(cos(phi), sin(phi)); // pixels, y down
        let dir3 = vec3<f32>(dir.x, -dir.y, 0.0); // view space, y up
        let ortho = dir3 - dot(dir3, v) * v;
        let axis = normalize(cross(dir3, v));
        let projected = n - axis * dot(n, axis);
        let projected_len = length(projected);
        let cos_n = clamp(dot(projected, v) / max(projected_len, 1e-4), 0.0, 1.0);
        let angle_n = sign(dot(ortho, projected)) * acos(cos_n);

        // Cosines of the highest horizon on each side: x along -dir, y along +dir.
        var horizon = vec2<f32>(-1.0);
        for (var j = 0u; j < STEPS; j += 1u) {
            let t = (f32(j) + jitter) / f32(STEPS);
            let step = max(t * t * radius_px, f32(j) + 1.0);
            let offset = vec2<i32>(round(dir * step));
            for (var side = 0; side < 2; side += 1) {
                let sp = px + offset * (side * 2 - 1);
                let sample_z = depth_at(sp);
                if (sample_z <= 0.0) {
                    continue; // sky: occludes nothing
                }
                let delta = view_position(sp, sample_z) - p;
                let len = length(delta);
                var c = dot(delta, v) / max(len, 1e-4);
                // Fade samples near the radius edge; ignore those past it.
                let weight = clamp((radius - len) / (radius * FALLOFF), 0.0, 1.0);
                c = mix(-1.0, c, weight);
                if (side == 0) {
                    horizon.x = max(horizon.x, c);
                } else {
                    horizon.y = max(horizon.y, c);
                }
            }
        }
        let h0 = angle_n + max(-acos(horizon.x) - angle_n, -PI / 2.0);
        let h1 = angle_n + min(acos(horizon.y) - angle_n, PI / 2.0);
        visibility += projected_len * (visible_arc(h0, angle_n) + visible_arc(h1, angle_n));
    }
    visibility /= f32(SLICES);
    let ao = pow(clamp(visibility, 0.0, 1.0), u.params.z);
    return vec4<f32>(ao, ao, ao, 1.0);
}

// Averages the 4x4 block the noise pattern repeats over, skipping texels
// across depth edges so dark halos don't bleed onto backgrounds.
@fragment
fn fs_blur(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(frag.xy);
    let z = depth_at(px);
    if (z <= 0.0) {
        return vec4<f32>(1.0);
    }
    let size = vec2<i32>(textureDimensions(view_depth));
    var sum = 0.0;
    var weights = 0.0;
    for (var y = -2; y < 2; y += 1) {
        for (var x = -2; x < 2; x += 1) {
            let sp = clamp(px + vec2<i32>(x, y), vec2<i32>(0), size - 1);
            let w = select(0.0, 1.0, abs(depth_at(sp) - z) < 0.05 * z);
            sum += textureLoad(raw_ao, sp, 0).r * w;
            weights += w;
        }
    }
    let ao = sum / max(weights, 1e-4);
    return vec4<f32>(ao, ao, ao, 1.0);
}
