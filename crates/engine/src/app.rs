//! The winit application: creates the window, starts GPU init, and draws frames.

use std::sync::Arc;

use impulse_render::{Gpu, GpuInitError, wgpu};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy, OwnedDisplayHandle};
use winit::window::{Window, WindowId};

use crate::game::{Context, Game};
use crate::platform;
use crate::running::Sandbox;
use crate::shell::{FrameResult, MakeGame, Shell};

/// Delivered to the event loop when async GPU init finishes (immediately on
/// desktop; later on the web, where init can't block).
enum UserEvent {
    GpuReady(Result<Gpu, GpuInitError>),
}

enum State {
    Uninitialized,
    Loading,
    // Boxed: Shell is large next to the unit variants (clippy::large_enum_variant).
    Running(Box<Shell>),
    Failed,
}

struct App {
    title: &'static str,
    proxy: EventLoopProxy<UserEvent>,
    window: Option<Arc<Window>>,
    state: State,
    /// Taken when the GPU is ready.
    make: Option<MakeGame>,
}

/// Opens the window and runs the impulse sandbox (the scene editor) until it
/// is closed.
pub fn run(title: &'static str) {
    run_game(title, |cx| Box::new(Sandbox::new(cx)));
}

/// Opens the window and runs `make`'s game until the window is closed. `make`
/// is called once the GPU is ready (later on the web, where init is async).
pub fn run_game(
    title: &'static str,
    make: impl FnOnce(&mut Context<'_>) -> Box<dyn Game> + 'static,
) {
    platform::init_logging();
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .expect("failed to create the event loop");
    let app = App {
        title,
        proxy: event_loop.create_proxy(),
        window: None,
        state: State::Uninitialized,
        make: Some(Box::new(make)),
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut app = app;
        event_loop.run_app(&mut app).expect("event loop failed");
    }
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }
}

async fn init_gpu(display: OwnedDisplayHandle, window: Arc<Window>) -> Result<Gpu, GpuInitError> {
    let mut desc = wgpu::InstanceDescriptor::new_with_display_handle(Box::new(display));
    if let Some(choice) = platform::backend_override() {
        desc.backends = choice.backends();
    }
    let instance = wgpu::util::new_instance_with_webgpu_detection(desc).await;
    // Created before the adapter: on the web, WebGL needs the canvas to pick an adapter.
    let surface = instance
        .create_surface(window.clone())
        .map_err(GpuInitError::Surface)?;
    let size = window.inner_size();
    Gpu::new(instance, surface, size.width, size.height).await
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !matches!(self.state, State::Uninitialized) {
            return;
        }
        self.state = State::Loading;

        let window = match event_loop.create_window(platform::window_attributes(self.title)) {
            Ok(window) => Arc::new(window),
            Err(e) => {
                platform::report_error(&format!("Could not create a window: {e}"));
                self.state = State::Failed;
                event_loop.exit();
                return;
            }
        };
        self.window = Some(window.clone());

        let display = event_loop.owned_display_handle();
        let proxy = self.proxy.clone();
        platform::spawn(async move {
            let result = init_gpu(display, window).await;
            // Fails only if the event loop already exited; nothing left to notify.
            let _ = proxy.send_event(UserEvent::GpuReady(result));
        });
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::GpuReady(Ok(gpu)) => {
                let (Some(window), Some(make)) = (&self.window, self.make.take()) else {
                    return;
                };
                let running = Shell::new(gpu, window, make);
                let name = running.backend();
                platform::report_status(&format!("Running on {name}"));
                window.set_title(&format!("{} ({name})", self.title));
                window.request_redraw();
                self.state = State::Running(Box::new(running));
            }
            UserEvent::GpuReady(Err(e)) => {
                platform::report_error(&e.to_string());
                self.state = State::Failed;
                // On the web, keep the page (and its error message) alive.
                if !cfg!(target_arch = "wasm32") {
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => self.redraw(),
            event => {
                if let (State::Running(running), Some(window)) = (&mut self.state, &self.window) {
                    running.on_window_event(window, &event);
                }
            }
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let State::Running(running) = &mut self.state {
            running.on_device_event(&event);
        }
    }
}

impl App {
    fn redraw(&mut self) {
        let (State::Running(running), Some(window)) = (&mut self.state, &self.window) else {
            return;
        };
        if let FrameResult::Failed(message) = running.frame(window) {
            platform::report_error(&message);
            self.state = State::Failed;
            return;
        }
        window.request_redraw();
    }
}
