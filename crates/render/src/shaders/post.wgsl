// The post chain: bloom (Jimenez's downsample/upsample, CoD: Advanced
// Warfare) and the composite that tonemaps into the swapchain.

struct FullscreenOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> FullscreenOut {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: FullscreenOut;
    out.clip = vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(corner.x, 1.0 - corner.y);
    return out;
}

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linear_clamp: sampler;

// 13 taps as five overlapping 2x2 boxes (centre weighted 0.5, corners 0.125
// each); no shimmer from a plain 2x2 box.
fn downsample_boxes(uv: vec2<f32>) -> array<vec3<f32>, 5> {
    let t = 1.0 / vec2<f32>(textureDimensions(source));
    let a = textureSample(source, linear_clamp, uv + t * vec2<f32>(-2.0, 2.0)).rgb;
    let b = textureSample(source, linear_clamp, uv + t * vec2<f32>(0.0, 2.0)).rgb;
    let c = textureSample(source, linear_clamp, uv + t * vec2<f32>(2.0, 2.0)).rgb;
    let d = textureSample(source, linear_clamp, uv + t * vec2<f32>(-2.0, 0.0)).rgb;
    let e = textureSample(source, linear_clamp, uv).rgb;
    let f = textureSample(source, linear_clamp, uv + t * vec2<f32>(2.0, 0.0)).rgb;
    let g = textureSample(source, linear_clamp, uv + t * vec2<f32>(-2.0, -2.0)).rgb;
    let h = textureSample(source, linear_clamp, uv + t * vec2<f32>(0.0, -2.0)).rgb;
    let i = textureSample(source, linear_clamp, uv + t * vec2<f32>(2.0, -2.0)).rgb;
    let j = textureSample(source, linear_clamp, uv + t * vec2<f32>(-1.0, 1.0)).rgb;
    let k = textureSample(source, linear_clamp, uv + t * vec2<f32>(1.0, 1.0)).rgb;
    let l = textureSample(source, linear_clamp, uv + t * vec2<f32>(-1.0, -1.0)).rgb;
    let m = textureSample(source, linear_clamp, uv + t * vec2<f32>(1.0, -1.0)).rgb;
    return array<vec3<f32>, 5>(
        (j + k + l + m) * 0.25,
        (a + b + d + e) * 0.25,
        (b + c + e + f) * 0.25,
        (d + e + g + h) * 0.25,
        (e + f + h + i) * 0.25,
    );
}

@fragment
fn fs_downsample(in: FullscreenOut) -> @location(0) vec4<f32> {
    let box = downsample_boxes(in.uv);
    let color = box[0] * 0.5 + (box[1] + box[2] + box[3] + box[4]) * 0.125;
    return vec4<f32>(color, 1.0);
}

fn karis_weight(c: vec3<f32>) -> f32 {
    let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    return 1.0 / (1.0 + luma);
}

// The first level from the scene: each box is weighted by 1 / (1 + luma)
// (Karis average) so a single very bright pixel can't flash a blob.
@fragment
fn fs_downsample_first(in: FullscreenOut) -> @location(0) vec4<f32> {
    let box = downsample_boxes(in.uv);
    var sum = vec3<f32>(0.0);
    var weights = 0.0;
    for (var i = 0; i < 5; i += 1) {
        let w = karis_weight(box[i]) * select(0.125, 0.5, i == 0);
        sum += box[i] * w;
        weights += w;
    }
    return vec4<f32>(sum / weights, 1.0);
}

// 3x3 tent around the source texel, blended additively into the larger level.
@fragment
fn fs_upsample(in: FullscreenOut) -> @location(0) vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(source));
    var sum = textureSample(source, linear_clamp, in.uv).rgb * 4.0;
    sum += textureSample(source, linear_clamp, in.uv + t * vec2<f32>(0.0, 1.0)).rgb * 2.0;
    sum += textureSample(source, linear_clamp, in.uv + t * vec2<f32>(0.0, -1.0)).rgb * 2.0;
    sum += textureSample(source, linear_clamp, in.uv + t * vec2<f32>(1.0, 0.0)).rgb * 2.0;
    sum += textureSample(source, linear_clamp, in.uv + t * vec2<f32>(-1.0, 0.0)).rgb * 2.0;
    sum += textureSample(source, linear_clamp, in.uv + t * vec2<f32>(1.0, 1.0)).rgb;
    sum += textureSample(source, linear_clamp, in.uv + t * vec2<f32>(-1.0, 1.0)).rgb;
    sum += textureSample(source, linear_clamp, in.uv + t * vec2<f32>(1.0, -1.0)).rgb;
    sum += textureSample(source, linear_clamp, in.uv + t * vec2<f32>(-1.0, -1.0)).rgb;
    return vec4<f32>(sum / 16.0, 1.0);
}

struct CompositeParams {
    // x: bloom strength, y: 1 to tonemap (HDR scene), 0 if already display-ready
    params: vec4<f32>,
};
@group(0) @binding(2) var bloom: texture_2d<f32>;
@group(0) @binding(3) var<uniform> composite: CompositeParams;

@fragment
fn fs_composite(in: FullscreenOut) -> @location(0) vec4<f32> {
    let scene = textureLoad(source, vec2<i32>(in.clip.xy), 0).rgb;
    let glow = textureSample(bloom, linear_clamp, in.uv).rgb;
    var color = mix(scene, glow, composite.params.x);
    if (composite.params.y > 0.5) {
        color = tonemap(color);
    }
    return vec4<f32>(color, 1.0);
}
