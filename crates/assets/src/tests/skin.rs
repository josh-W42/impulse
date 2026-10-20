use super::*;

/// The unit triangle with `JOINTS_0` and `WEIGHTS_0` (one VEC4 each per vertex).
fn skinned_triangle(glb: &mut Glb, joints: &[u16; 12], weights: &[f32; 12]) -> String {
    let p = glb.floats(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], "VEC3", 3);
    let n = glb.floats(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], "VEC3", 3);
    let j = glb.u16s(joints, "VEC4", 4);
    let w = glb.floats(weights, "VEC4", 4);
    let i = glb.indices(&[0, 1, 2]);
    format!(
        r#"{{"attributes":{{"POSITION":{p},"NORMAL":{n},"JOINTS_0":{j},"WEIGHTS_0":{w}}},"indices":{i}}}"#
    )
}

/// Every vertex fully weighted to skin joint 0.
const ALL_ON_FIRST: ([u16; 12], [f32; 12]) = (
    [0; 12],
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
);

/// Column-major 4×4 matrices for an `inverseBindMatrices` accessor.
fn matrices(glb: &mut Glb, mats: &[Mat4]) -> usize {
    let data: Vec<f32> = mats.iter().flat_map(|m| m.to_cols_array()).collect();
    glb.floats(&data, "MAT4", 16)
}

fn positions(model: &SkinnedModelData) -> Vec<Vec3> {
    model
        .meshes
        .iter()
        .flat_map(|m| m.vertices.iter().map(|v| Vec3::from_array(v.position)))
        .collect()
}

#[test]
fn skinned_vertices_stay_in_bind_space() {
    let mut glb = Glb::default();
    let prim = skinned_triangle(&mut glb, &ALL_ON_FIRST.0, &ALL_ON_FIRST.1);
    let ib = matrices(&mut glb, &[Mat4::IDENTITY]);
    let bytes = glb.build(&format!(
        r#""meshes":[{{"primitives":[{prim}]}}],
        "nodes":[{{"mesh":0,"skin":0,"translation":[5,0,0]}},{{"name":"root"}}],
        "skins":[{{"joints":[1],"inverseBindMatrices":{ib}}}],
        "scenes":[{{"nodes":[0,1]}}],"scene":0"#
    ));
    let model = load_skinned_glb(&bytes).unwrap();
    assert_eq!(
        positions(&model),
        [Vec3::ZERO, Vec3::X, Vec3::Y],
        "the skinned mesh node's own transform is ignored"
    );
    assert_eq!(
        (model.min, model.max),
        (Vec3::ZERO, Vec3::new(1.0, 1.0, 0.0))
    );
    assert_eq!(model.skeleton.joints.len(), 1);
    assert_eq!(model.skeleton.find("root"), Some(0));
}

#[test]
fn joints_come_parent_first_and_vertex_indices_follow() {
    let mut glb = Glb::default();
    let prim = skinned_triangle(&mut glb, &ALL_ON_FIRST.0, &ALL_ON_FIRST.1);
    let ib = matrices(&mut glb, &[Mat4::IDENTITY, Mat4::IDENTITY]);
    let bytes = glb.build(&format!(
        r#""meshes":[{{"primitives":[{prim}]}}],
        "nodes":[{{"mesh":0,"skin":0}},{{"name":"root","children":[2]}},{{"name":"child","translation":[0,1,0]}}],
        "skins":[{{"joints":[2,1],"inverseBindMatrices":{ib}}}],
        "scenes":[{{"nodes":[0,1]}}],"scene":0"#
    ));
    let model = load_skinned_glb(&bytes).unwrap();
    let joints = &model.skeleton.joints;
    assert_eq!(joints[0].name, "root");
    assert_eq!(joints[0].parent, None);
    assert_eq!(joints[1].name, "child");
    assert_eq!(joints[1].parent, Some(0));
    assert_eq!(joints[1].rest.translation, Vec3::Y);
    // Skin joint 0 was `child`, which is now joint 1.
    assert!(model.meshes[0].vertices.iter().all(|v| v.joints[0] == 1));
}

#[test]
fn rigid_mesh_under_a_joint_is_skinned_to_it() {
    let mut glb = Glb::default();
    let body = skinned_triangle(&mut glb, &ALL_ON_FIRST.0, &ALL_ON_FIRST.1);
    let hat = triangle(&mut glb, true);
    let ib = matrices(
        &mut glb,
        &[Mat4::from_translation(Vec3::new(0.0, -2.0, 0.0))],
    );
    let bytes = glb.build(&format!(
        r#""meshes":[{{"primitives":[{body}]}},{{"primitives":[{hat}]}}],
        "nodes":[{{"mesh":0,"skin":0}},{{"name":"head","translation":[0,2,0],"children":[2]}},{{"mesh":1}}],
        "skins":[{{"joints":[1],"inverseBindMatrices":{ib}}}],
        "scenes":[{{"nodes":[0,1]}}],"scene":0"#
    ));
    let model = load_skinned_glb(&bytes).unwrap();
    assert_eq!(model.meshes.len(), 2);
    let hat = model
        .meshes
        .iter()
        .find(|m| m.vertices.iter().any(|v| v.position[1] > 1.5))
        .expect("the hat, moved into bind space");
    let p: Vec<Vec3> = hat
        .vertices
        .iter()
        .map(|v| Vec3::from_array(v.position))
        .collect();
    assert_eq!(
        p,
        [
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0)
        ]
    );
    assert!(
        hat.vertices
            .iter()
            .all(|v| v.joints == [0, 0, 0, 0] && v.weights == [1.0, 0.0, 0.0, 0.0])
    );
}

#[test]
fn non_joint_ancestors_become_the_base() {
    let mut glb = Glb::default();
    let prim = skinned_triangle(&mut glb, &ALL_ON_FIRST.0, &ALL_ON_FIRST.1);
    let ib = matrices(&mut glb, &[Mat4::IDENTITY]);
    let bytes = glb.build(&format!(
        r#""meshes":[{{"primitives":[{prim}]}}],
        "nodes":[{{"mesh":0,"skin":0}},{{"name":"Armature","scale":[100,100,100],"children":[2]}},{{"name":"root"}}],
        "skins":[{{"joints":[2],"inverseBindMatrices":{ib}}}],
        "scenes":[{{"nodes":[0,1]}}],"scene":0"#
    ));
    let model = load_skinned_glb(&bytes).unwrap();
    assert_eq!(
        model.skeleton.joints[0].base,
        Mat4::from_scale(Vec3::splat(100.0))
    );
}

#[test]
fn weights_are_normalized() {
    let mut glb = Glb::default();
    let weights = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let prim = skinned_triangle(&mut glb, &[0; 12], &weights);
    let ib = matrices(&mut glb, &[Mat4::IDENTITY]);
    let bytes = glb.build(&format!(
        r#""meshes":[{{"primitives":[{prim}]}}],
        "nodes":[{{"mesh":0,"skin":0}},{{"name":"root"}}],
        "skins":[{{"joints":[1],"inverseBindMatrices":{ib}}}],
        "scenes":[{{"nodes":[0,1]}}],"scene":0"#
    ));
    let model = load_skinned_glb(&bytes).unwrap();
    let v = &model.meshes[0].vertices;
    let third = 1.0 / 3.0;
    assert!(
        v[0].weights
            .iter()
            .zip([third, third, third, 0.0])
            .all(|(a, b)| (a - b).abs() < 1e-6),
        "{:?}",
        v[0].weights
    );
    assert_eq!(
        v[1].weights,
        [1.0, 0.0, 0.0, 0.0],
        "all-zero weights go to joint 0"
    );
    assert_eq!(v[2].weights, [1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn skin_errors() {
    // No skin at all.
    let mut glb = Glb::default();
    assert!(matches!(
        load_skinned_glb(&one_node(&mut glb, r#"{"mesh":0}"#)),
        Err(ModelError::NoSkin)
    ));

    // Two skins.
    let mut glb = Glb::default();
    let prim = skinned_triangle(&mut glb, &ALL_ON_FIRST.0, &ALL_ON_FIRST.1);
    let bytes = glb.build(&format!(
        r#""meshes":[{{"primitives":[{prim}]}}],
        "nodes":[{{"mesh":0,"skin":0}},{{"name":"a"}},{{"name":"b"}}],
        "skins":[{{"joints":[1]}},{{"joints":[2]}}],
        "scenes":[{{"nodes":[0,1,2]}}],"scene":0"#
    ));
    assert!(matches!(
        load_skinned_glb(&bytes),
        Err(ModelError::TooManySkins)
    ));

    // 129 joints, one more than the default limit.
    let mut glb = Glb::default();
    let prim = skinned_triangle(&mut glb, &ALL_ON_FIRST.0, &ALL_ON_FIRST.1);
    let joints: Vec<String> = (1..=129).map(|i| i.to_string()).collect();
    let nodes: Vec<String> = (1..=129).map(|i| format!(r#"{{"name":"j{i}"}}"#)).collect();
    let bytes = glb.build(&format!(
        r#""meshes":[{{"primitives":[{prim}]}}],
        "nodes":[{{"mesh":0,"skin":0}},{}],
        "skins":[{{"joints":[{}]}}],
        "scenes":[{{"nodes":[0]}}],"scene":0"#,
        nodes.join(","),
        joints.join(",")
    ));
    assert_eq!(
        load_skinned_glb(&bytes).err(),
        Some(ModelError::TooManyJoints { max: 128 })
    );
}
