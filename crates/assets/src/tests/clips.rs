use glam::Quat;

use super::*;

/// A file of named nodes and one animation; no meshes. `samplers` and
/// `channels` are JSON arrays.
fn animation_file(glb: &Glb, nodes: &str, samplers: &str, channels: &str) -> Vec<u8> {
    glb.build(&format!(
        r#""nodes":[{nodes}],
        "animations":[{{"name":"Wave","samplers":{samplers},"channels":{channels}}}],
        "scenes":[{{"nodes":[0]}}],"scene":0"#
    ))
}

#[test]
fn clips_load_by_node_name() {
    let mut glb = Glb::default();
    let t = glb.floats(&[0.0, 1.0], "SCALAR", 1);
    let half = std::f32::consts::FRAC_1_SQRT_2;
    let q = glb.floats(&[0.0, 0.0, 0.0, 1.0, 0.0, half, 0.0, half], "VEC4", 4);
    let bytes = animation_file(
        &glb,
        r#"{"name":"root","children":[1]},{"name":"hips"}"#,
        &format!(r#"[{{"input":{t},"output":{q}}}]"#),
        r#"[{"sampler":0,"target":{"node":1,"path":"rotation"}}]"#,
    );
    let clips = load_clips(&bytes).unwrap();
    assert_eq!(clips.len(), 1);
    let clip = &clips[0];
    assert_eq!((clip.name.as_str(), clip.duration), ("Wave", 1.0));
    assert_eq!(clip.channels.len(), 1);
    let channel = &clip.channels[0];
    assert_eq!(channel.target, "hips");
    assert_eq!(channel.times, [0.0, 1.0]);
    assert_eq!(channel.interpolation, Interpolation::Linear);
    let Keyframes::Rotation(r) = &channel.keyframes else {
        panic!("{:?}", channel.keyframes)
    };
    assert!(r[1].abs_diff_eq(Quat::from_xyzw(0.0, half, 0.0, half), 1e-6));
}

#[test]
fn step_and_cubic_interpolation() {
    let mut glb = Glb::default();
    let t = glb.floats(&[0.0, 2.0], "SCALAR", 1);
    let step = glb.floats(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], "VEC3", 3);
    // Cubic: (in-tangent, value, out-tangent) per key.
    #[rustfmt::skip]
    let cubic = glb.floats(&[
        9.0, 9.0, 9.0,  1.0, 0.0, 0.0,  9.0, 9.0, 9.0,
        9.0, 9.0, 9.0,  3.0, 0.0, 0.0,  9.0, 9.0, 9.0,
    ], "VEC3", 3);
    let bytes = animation_file(
        &glb,
        r#"{"name":"a"},{"name":"b"}"#,
        &format!(
            r#"[{{"input":{t},"output":{step},"interpolation":"STEP"}},
                {{"input":{t},"output":{cubic},"interpolation":"CUBICSPLINE"}}]"#
        ),
        r#"[{"sampler":0,"target":{"node":0,"path":"scale"}},
            {"sampler":1,"target":{"node":1,"path":"translation"}}]"#,
    );
    let clip = &load_clips(&bytes).unwrap()[0];
    assert_eq!(clip.duration, 2.0);
    assert_eq!(clip.channels[0].interpolation, Interpolation::Step);
    assert!(matches!(&clip.channels[0].keyframes, Keyframes::Scale(s) if s.len() == 2));
    assert_eq!(clip.channels[1].interpolation, Interpolation::Linear);
    let Keyframes::Translation(values) = &clip.channels[1].keyframes else {
        panic!()
    };
    assert_eq!(values, &[Vec3::X, Vec3::new(3.0, 0.0, 0.0)]);
}

#[test]
fn animation_errors() {
    let mut glb = Glb::default();
    let t = glb.floats(&[1.0, 0.0], "SCALAR", 1);
    let v = glb.floats(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], "VEC3", 3);
    let bytes = animation_file(
        &glb,
        r#"{"name":"a"}"#,
        &format!(r#"[{{"input":{t},"output":{v}}}]"#),
        r#"[{"sampler":0,"target":{"node":0,"path":"translation"}}]"#,
    );
    assert!(matches!(load_clips(&bytes), Err(ModelError::Animation(_))));
}

#[test]
fn files_without_meshes_still_give_clips() {
    let mut glb = Glb::default();
    let t = glb.floats(&[0.0, 0.5], "SCALAR", 1);
    let v = glb.floats(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], "VEC3", 3);
    let bytes = animation_file(
        &glb,
        r#"{}"#,
        &format!(r#"[{{"input":{t},"output":{v}}}]"#),
        r#"[{"sampler":0,"target":{"node":0,"path":"translation"}}]"#,
    );
    assert!(matches!(load_glb(&bytes), Err(ModelError::NoTriangles)));
    let clips = load_clips(&bytes).unwrap();
    assert_eq!(clips[0].channels[0].target, "#0", "an unnamed node");
    assert_eq!(clips[0].duration, 0.5);
}
