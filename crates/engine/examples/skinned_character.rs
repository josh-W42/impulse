//! A skinned, animated character: the bundled KayKit mannequin cycling
//! through its clips with crossfades. Space skips to the next clip.
//! `cargo run -p impulse-engine --example skinned_character`

use impulse_engine::anim::{Animator, BoundClip, Pose, model_transforms, skinning_matrices};
use impulse_engine::assets::{SkinnedModelData, load_skinned_glb};
use impulse_engine::glam::camera::rh::view::look_at_mat4;
use impulse_engine::glam::{Mat4, Vec3};
use impulse_engine::render::{CameraData, DrawList, MeshKind, SkinnedMeshId};
use impulse_engine::upload::{TextureCache, upload_skinned_model};
use impulse_engine::winit::event::{ElementState, WindowEvent};
use impulse_engine::winit::keyboard::{KeyCode, PhysicalKey};
use impulse_engine::{Context, Game, perspective};

const CLIPS: [&str; 6] = ["Idle_A", "Idle_B", "Interact", "Hit_A", "Throw", "Death_A"];
const SECONDS_PER_CLIP: f32 = 3.0;
const FADE: f32 = 0.25;

struct SkinnedCharacter {
    model: SkinnedModelData,
    /// The clips in `CLIPS` order, bound to the model's skeleton.
    clips: Vec<BoundClip>,
    meshes: Vec<SkinnedMeshId>,
    animator: Animator,
    playing: usize,
    clock: f32,
    skip: bool,
    scratch: Pose,
    pose: Pose,
    model_space: Vec<Mat4>,
    skin: Vec<Mat4>,
}

impl SkinnedCharacter {
    fn new() -> Self {
        let model = load_skinned_glb(include_bytes!(
            "../../../assets/models/mannequin-general.glb"
        ))
        .expect("the bundled mannequin loads");
        let clips = CLIPS
            .iter()
            .map(|name| {
                let clip = model
                    .clips
                    .iter()
                    .find(|c| c.name == *name)
                    .expect("bundled clip");
                BoundClip::bind(clip, &model.skeleton)
            })
            .collect();
        let pose = Pose::rest(&model.skeleton);
        Self {
            clips,
            meshes: Vec::new(),
            animator: Animator::new(0),
            playing: 0,
            clock: 0.0,
            skip: false,
            scratch: pose.clone(),
            pose,
            model_space: Vec::new(),
            skin: Vec::new(),
            model,
        }
    }

    fn next_clip(&mut self) {
        self.playing = (self.playing + 1) % CLIPS.len();
        self.clock = 0.0;
        // Death plays once and holds; everything else loops.
        let looping = CLIPS[self.playing] != "Death_A";
        self.animator.play(self.playing, looping, FADE);
    }
}

impl Game for SkinnedCharacter {
    fn update(&mut self, cx: &mut Context<'_>) {
        if self.meshes.is_empty() {
            self.meshes = upload_skinned_model(
                cx.renderer,
                cx.gpu,
                &self.model,
                &mut TextureCache::default(),
            );
        }
        self.clock += cx.dt;
        if self.clock > SECONDS_PER_CLIP || std::mem::take(&mut self.skip) {
            self.next_clip();
        }
        self.animator.advance(cx.dt, &self.clips);
    }

    fn window_event(&mut self, _cx: &mut Context<'_>, event: &WindowEvent) {
        if let WindowEvent::KeyboardInput { event, .. } = event
            && event.state == ElementState::Pressed
            && !event.repeat
            && event.physical_key == PhysicalKey::Code(KeyCode::Space)
        {
            self.skip = true;
        }
    }

    fn ui(&mut self, ui: &mut impulse_engine::egui::Ui) {
        impulse_engine::egui::Window::new("skinned character").show(ui.ctx(), |ui| {
            ui.label(format!("{} (Space: next)", CLIPS[self.playing]));
        });
    }

    fn draw(&mut self, cx: &mut Context<'_>, list: &mut DrawList) -> CameraData {
        let skeleton = &self.model.skeleton;
        self.animator
            .pose(&self.clips, skeleton, &mut self.scratch, &mut self.pose);
        model_transforms(skeleton, &self.pose, &mut self.model_space);
        skinning_matrices(skeleton, &self.model_space, &mut self.skin);
        let joints = list.push_joints(&self.skin);
        for &mesh in &self.meshes {
            list.push_skinned(mesh, Mat4::IDENTITY, [1.0; 4], joints);
        }
        list.push_instance(
            MeshKind::Plane,
            Mat4::from_scale(Vec3::new(10.0, 1.0, 10.0)),
            [0.5, 0.5, 0.5, 1.0],
        );
        let height = (self.model.max.y - self.model.min.y).max(0.5);
        let eye = Vec3::new(1.5, 1.2, 3.0) * height;
        let view = look_at_mat4(eye, Vec3::Y * height * 0.5, Vec3::Y);
        CameraData {
            view_proj: perspective(cx.aspect()) * view,
            eye,
        }
    }
}

fn main() {
    impulse_engine::run_game("skinned character", |_| Box::new(SkinnedCharacter::new()));
}
