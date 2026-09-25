//! WGSL sources. Tests validate them with naga and translate every entry point
//! to GLSL ES 3.00 (what wgpu does for WebGL2), so shader errors fail `cargo test`
//! instead of surfacing at runtime in one browser.
//!
//! Limit: naga is more lenient than browser WGSL compilers (e.g. Chrome's Tint)
//! about uniformity rules — it accepted `textureSampleCompare` in non-uniform
//! control flow. Passing here doesn't replace checking both backends in a browser.

pub const SCENE_WGSL: &str = include_str!("shaders/scene.wgsl");
pub const LINES_WGSL: &str = include_str!("shaders/lines.wgsl");

#[cfg(test)]
mod tests {
    use naga::back::glsl;
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;

    fn check(name: &str, source: &str, entry_points: &[(&str, naga::ShaderStage)]) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
        let info = Validator::new(ValidationFlags::all(), Capabilities::default())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
        let options = glsl::Options {
            version: glsl::Version::Embedded {
                version: 300,
                is_webgl: true,
            },
            ..Default::default()
        };
        for &(entry_point, shader_stage) in entry_points {
            let pipeline = glsl::PipelineOptions {
                shader_stage,
                entry_point: entry_point.to_string(),
                multiview: None,
            };
            let mut out = String::new();
            glsl::Writer::new(
                &mut out,
                &module,
                &info,
                &options,
                &pipeline,
                Default::default(),
            )
            .and_then(|mut writer| writer.write())
            .unwrap_or_else(|e| panic!("{name}::{entry_point} → WebGL2 GLSL: {e}"));
        }
    }

    #[test]
    fn scene_shader_is_valid_for_webgl2() {
        use naga::ShaderStage::{Fragment, Vertex};
        check(
            "scene",
            SCENE_WGSL,
            &[
                ("vs_main", Vertex),
                ("vs_shadow", Vertex),
                ("fs_main", Fragment),
            ],
        );
    }

    #[test]
    fn lines_shader_is_valid_for_webgl2() {
        use naga::ShaderStage::{Fragment, Vertex};
        check(
            "lines",
            LINES_WGSL,
            &[("vs_main", Vertex), ("fs_main", Fragment)],
        );
    }
}
