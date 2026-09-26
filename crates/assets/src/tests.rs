use super::*;

const FLOAT: u32 = 5126;
const U16: u32 = 5123;

/// Builds small `.glb` files: binary data plus hand-written glTF JSON.
#[derive(Default)]
struct Glb {
    bin: Vec<u8>,
    views: Vec<String>,
    accessors: Vec<String>,
}

impl Glb {
    fn view(&mut self, bytes: &[u8]) -> usize {
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
        self.views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}}"#,
            self.bin.len(),
            bytes.len()
        ));
        self.bin.extend_from_slice(bytes);
        self.views.len() - 1
    }

    fn floats(&mut self, data: &[f32], ty: &str, width: usize) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let view = self.view(&bytes);
        let count = data.len() / width;
        // POSITION accessors must carry min/max.
        let bounds = if ty == "VEC3" {
            let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
            for p in data.chunks(3) {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            format!(r#","min":{lo:?},"max":{hi:?}"#)
        } else {
            String::new()
        };
        self.accessors.push(format!(
            r#"{{"bufferView":{view},"componentType":{FLOAT},"count":{count},"type":"{ty}"{bounds}}}"#
        ));
        self.accessors.len() - 1
    }

    fn indices(&mut self, data: &[u16]) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
        let view = self.view(&bytes);
        self.accessors.push(format!(
            r#"{{"bufferView":{view},"componentType":{U16},"count":{},"type":"SCALAR"}}"#,
            data.len()
        ));
        self.accessors.len() - 1
    }

    /// `rest` supplies the remaining top-level JSON members (meshes, nodes, ...).
    fn build(&self, rest: &str) -> Vec<u8> {
        self.build_with_buffer(rest, &format!(r#"{{"byteLength":{}}}"#, self.bin.len()))
    }

    fn build_with_buffer(&self, rest: &str, buffer: &str) -> Vec<u8> {
        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},"buffers":[{buffer}],"bufferViews":[{}],"accessors":[{}],{rest}}}"#,
            self.views.join(","),
            self.accessors.join(",")
        );
        let mut json = json.into_bytes();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let mut bin = self.bin.clone();
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let total = 12 + 8 + json.len() + 8 + bin.len();
        let mut out = Vec::new();
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(b"JSON");
        out.extend_from_slice(&json);
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(b"BIN\0");
        out.extend_from_slice(&bin);
        out
    }
}

/// A unit right triangle in the XY plane, counter-clockwise seen from +Z.
fn triangle(glb: &mut Glb, with_normals: bool) -> String {
    let p = glb.floats(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], "VEC3", 3);
    let uv = glb.floats(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], "VEC2", 2);
    let i = glb.indices(&[0, 1, 2]);
    let normals = if with_normals {
        let n = glb.floats(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], "VEC3", 3);
        format!(r#","NORMAL":{n}"#)
    } else {
        String::new()
    };
    format!(r#"{{"attributes":{{"POSITION":{p},"TEXCOORD_0":{uv}{normals}}},"indices":{i}}}"#)
}

fn one_node(glb: &mut Glb, node: &str) -> Vec<u8> {
    let prim = triangle(glb, true);
    glb.build(&format!(
        r#""meshes":[{{"primitives":[{prim}]}}],"nodes":[{node}],"scenes":[{{"nodes":[0]}}],"scene":0"#
    ))
}

fn positions(model: &ModelData) -> Vec<Vec3> {
    model.positions().collect()
}

#[test]
fn node_transforms_are_baked_into_vertices() {
    let mut glb = Glb::default();
    let model = load_glb(&one_node(
        &mut glb,
        r#"{"mesh":0,"translation":[1,2,3],"scale":[2,2,2]}"#,
    ))
    .unwrap();
    assert_eq!(
        positions(&model),
        [
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(3.0, 2.0, 3.0),
            Vec3::new(1.0, 4.0, 3.0)
        ]
    );
    let mesh = &model.meshes[0];
    assert_eq!(mesh.indices, [0, 1, 2]);
    assert!(mesh.vertices.iter().all(|v| v.normal == [0.0, 0.0, 1.0]));
    assert_eq!(mesh.vertices[1].uv, [1.0, 0.0]);
    assert_eq!(
        (model.min, model.max),
        (Vec3::new(1.0, 2.0, 3.0), Vec3::new(3.0, 4.0, 3.0))
    );
}

#[test]
fn parent_transforms_apply_to_children() {
    let mut glb = Glb::default();
    let prim = triangle(&mut glb, true);
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let bytes = glb.build(&format!(
        r#""meshes":[{{"primitives":[{prim}]}}],
        "nodes":[{{"rotation":[0,{s},0,{s}],"children":[1]}},{{"mesh":0,"translation":[1,0,0]}}],
        "scenes":[{{"nodes":[0]}}],"scene":0"#
    ));
    let model = load_glb(&bytes).unwrap();
    // A quarter turn about +Y takes +X to −Z and +Z to +X.
    assert!(positions(&model)[0].abs_diff_eq(Vec3::new(0.0, 0.0, -1.0), 1e-6));
    let n = Vec3::from_array(model.meshes[0].vertices[0].normal);
    assert!(n.abs_diff_eq(Vec3::X, 1e-6), "{n}");
}

#[test]
fn mirrored_nodes_keep_triangles_facing_their_normals() {
    let mut glb = Glb::default();
    let model = load_glb(&one_node(&mut glb, r#"{"mesh":0,"scale":[-1,1,1]}"#)).unwrap();
    let mesh = &model.meshes[0];
    let [a, b, c] =
        [0, 1, 2].map(|k| Vec3::from_array(mesh.vertices[mesh.indices[k] as usize].position));
    let facing = (b - a).cross(c - a);
    let normal = Vec3::from_array(mesh.vertices[0].normal);
    assert!(
        facing.dot(normal) > 0.0,
        "winding {facing} vs normal {normal}"
    );
}

#[test]
fn missing_normals_are_computed_from_the_winding() {
    let mut glb = Glb::default();
    let prim = triangle(&mut glb, false);
    let bytes = glb.build(&format!(
        r#""meshes":[{{"primitives":[{prim}]}}],"nodes":[{{"mesh":0}}],"scenes":[{{"nodes":[0]}}]"#
    ));
    let model = load_glb(&bytes).unwrap();
    assert!(
        model.meshes[0]
            .vertices
            .iter()
            .all(|v| v.normal == [0.0, 0.0, 1.0])
    );
}

fn png(width: u32, height: u32, pixel: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let img = image::RgbaImage::from_fn(width, height, |x, y| image::Rgba(pixel(x, y)));
    let mut out = Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

fn textured(glb: &mut Glb, image: &[u8]) -> Vec<u8> {
    let prim = triangle(glb, true).replace(r#","indices""#, r#","material":0,"indices""#);
    let view = glb.view(image);
    glb.build(&format!(
        r#""meshes":[{{"primitives":[{prim}]}}],"nodes":[{{"mesh":0}}],"scenes":[{{"nodes":[0]}}],
        "materials":[{{"pbrMetallicRoughness":{{"baseColorFactor":[0.5,1,1,1],"baseColorTexture":{{"index":0}}}}}}],
        "textures":[{{"source":0}}],"images":[{{"bufferView":{view},"mimeType":"image/png"}}]"#
    ))
}

#[test]
fn materials_and_textures_come_through_with_mips() {
    let image = png(4, 2, |x, y| [x as u8 * 60, y as u8 * 100, 7, 255]);
    let mut glb = Glb::default();
    let model = load_glb(&textured(&mut glb, &image)).unwrap();
    assert_eq!(model.meshes[0].material, Some(0));
    assert_eq!(
        model.materials,
        [MaterialData {
            base_color: [0.5, 1.0, 1.0, 1.0],
            base_color_texture: Some(0),
        }]
    );
    let mips = &model.textures[0].mips;
    let sizes: Vec<_> = mips.iter().map(|m| (m.width, m.height)).collect();
    assert_eq!(sizes, [(4, 2), (2, 1), (1, 1)]);
    assert_eq!(&mips[0].rgba[4..8], &[60, 0, 7, 255], "texel (1, 0)");
}

#[test]
fn srgb_mips_average_in_linear_light() {
    let levels = mip_chain(vec![0, 0, 0, 0, 255, 255, 255, 255], 2, 1, true);
    // Half black, half white is 0.5 linear: sRGB 188, not 128.
    assert_eq!(levels[1].rgba, [188, 188, 188, 128]);
    let linear = mip_chain(vec![0, 0, 0, 0, 255, 255, 255, 255], 2, 1, false);
    assert_eq!(linear[1].rgba, [128, 128, 128, 128]);
    let odd = mip_chain(vec![9; 3 * 5 * 4], 3, 5, false);
    let sizes: Vec<_> = odd.iter().map(|m| (m.width, m.height)).collect();
    assert_eq!(sizes, [(3, 5), (1, 2), (1, 1)]);
    assert!(odd.iter().all(|m| m.rgba.iter().all(|&c| c == 9)));
}

#[test]
fn bad_files_give_specific_errors() {
    assert!(matches!(
        load_glb(b"not a model"),
        Err(ModelError::Parse(_))
    ));

    let mut glb = Glb::default();
    let good = one_node(&mut glb, r#"{"mesh":0}"#);
    let tiny = Limits {
        max_bytes: 10,
        ..Limits::default()
    };
    assert!(matches!(
        load_glb_with(&good, &tiny),
        Err(ModelError::TooLarge { .. })
    ));
    let few = Limits {
        max_vertices: 2,
        ..Limits::default()
    };
    assert_eq!(
        load_glb_with(&good, &few),
        Err(ModelError::TooManyVertices { max: 2 })
    );

    let mut glb = Glb::default();
    let prim = triangle(&mut glb, true);
    let external = glb.build_with_buffer(
        &format!(r#""meshes":[{{"primitives":[{prim}]}}],"nodes":[{{"mesh":0}}],"scenes":[{{"nodes":[0]}}]"#),
        &format!(r#"{{"byteLength":{},"uri":"data.bin"}}"#, glb.bin.len()),
    );
    assert_eq!(load_glb(&external), Err(ModelError::ExternalData));

    let mut glb = Glb::default();
    let textured = textured(&mut glb, &png(4, 2, |_, _| [1, 2, 3, 255]));
    let small = Limits {
        max_texture_size: 2,
        ..Limits::default()
    };
    assert!(matches!(
        load_glb_with(&textured, &small),
        Err(ModelError::Texture(_))
    ));

    let mut glb = Glb::default();
    let points = triangle(&mut glb, true).replace(r#""indices""#, r#""mode":0,"indices""#);
    let bytes = glb.build(&format!(
        r#""meshes":[{{"primitives":[{points}]}}],"nodes":[{{"mesh":0}}],"scenes":[{{"nodes":[0]}}]"#
    ));
    assert_eq!(load_glb(&bytes), Err(ModelError::NoTriangles));

    let mut glb = Glb::default();
    let p = glb.floats(&[0.0; 9], "VEC3", 3);
    let i = glb.indices(&[0, 1, 7]);
    let bytes = glb.build(&format!(
        r#""meshes":[{{"primitives":[{{"attributes":{{"POSITION":{p}}},"indices":{i}}}]}}],"nodes":[{{"mesh":0}}],"scenes":[{{"nodes":[0]}}]"#
    ));
    assert!(matches!(load_glb(&bytes), Err(ModelError::BadGeometry(_))));
}

#[test]
fn the_bundled_models_load() {
    let models: [(&str, &[u8]); 8] = [
        (
            "barrel",
            include_bytes!("../../../assets/models/barrel.glb"),
        ),
        ("box", include_bytes!("../../../assets/models/box.glb")),
        (
            "box-large",
            include_bytes!("../../../assets/models/box-large.glb"),
        ),
        (
            "bucket",
            include_bytes!("../../../assets/models/bucket.glb"),
        ),
        ("chest", include_bytes!("../../../assets/models/chest.glb")),
        (
            "rock-a",
            include_bytes!("../../../assets/models/rock-a.glb"),
        ),
        (
            "rock-b",
            include_bytes!("../../../assets/models/rock-b.glb"),
        ),
        (
            "tree-log",
            include_bytes!("../../../assets/models/tree-log.glb"),
        ),
    ];
    for (name, bytes) in models {
        let model = load_glb(bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(model.vertex_count() > 0, "{name}");
        assert_eq!(model.textures.len(), 1, "{name}: one shared palette");
        assert_eq!(
            model.textures[0].mips.len(),
            10,
            "{name}: 512 → 1 is 10 levels"
        );
        let size = model.max - model.min;
        assert!(
            size.min_element() > 0.01 && size.max_element() < 10.0,
            "{name}: {size}"
        );
    }
}
