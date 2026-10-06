//! The smallest impulse game: a spinning cube on a floor.
//! `cargo run -p impulse-engine --example spinning_cube`

use impulse_engine::glam::camera::rh::view::look_at_mat4;
use impulse_engine::glam::{Mat4, Quat, Vec3};
use impulse_engine::render::{CameraData, DrawList, MeshKind};
use impulse_engine::{Context, Game, perspective};

struct SpinningCube {
    angle: f32,
}

impl Game for SpinningCube {
    fn update(&mut self, cx: &mut Context<'_>) {
        self.angle += cx.dt;
    }

    fn ui(&mut self, ui: &mut impulse_engine::egui::Ui) {
        impulse_engine::egui::Window::new("spinning cube").show(ui.ctx(), |ui| {
            ui.label(format!("angle {:.1} rad", self.angle));
        });
    }

    fn draw(&mut self, cx: &mut Context<'_>, list: &mut DrawList) -> CameraData {
        let spin = Quat::from_rotation_y(self.angle) * Quat::from_rotation_x(0.4);
        list.push_instance(
            MeshKind::Cube,
            Mat4::from_rotation_translation(spin, Vec3::new(0.0, 1.0, 0.0)),
            [0.9, 0.4, 0.2, 1.0],
        );
        list.push_instance(
            MeshKind::Plane,
            Mat4::from_scale(Vec3::new(10.0, 1.0, 10.0)),
            [0.5, 0.5, 0.5, 1.0],
        );
        let eye = Vec3::new(3.0, 3.0, 5.0);
        let view = look_at_mat4(eye, Vec3::new(0.0, 0.8, 0.0), Vec3::Y);
        CameraData {
            view_proj: perspective(cx.aspect()) * view,
            eye,
        }
    }
}

fn main() {
    impulse_engine::run_game("spinning cube", |_| Box::new(SpinningCube { angle: 0.0 }));
}
