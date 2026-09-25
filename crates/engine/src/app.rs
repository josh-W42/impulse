//! The winit application: creates the window, starts GPU init, and draws frames.

use std::sync::Arc;

use impulse_render::{FrameOutcome, Gpu, GpuInitError, backend_name, wgpu};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy, OwnedDisplayHandle};
use winit::window::{Window, WindowId};

use crate::platform;

const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.02,
    g: 0.03,
    b: 0.06,
    a: 1.0,
};

/// Delivered to the event loop when async GPU init finishes (immediately on
/// desktop; later on the web, where init can't block).
enum UserEvent {
    GpuReady(Result<Gpu, GpuInitError>),
}

enum State {
    Uninitialized,
    Loading,
    // Boxed: Gpu is large next to the unit variants (clippy::large_enum_variant).
    Running(Box<Gpu>),
    Failed,
}

struct App {
    title: &'static str,
    proxy: EventLoopProxy<UserEvent>,
    window: Option<Arc<Window>>,
    state: State,
}

/// Opens the window and runs the engine until it is closed.
pub fn run(title: &'static str) {
    platform::init_logging();
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .expect("failed to create the event loop");
    let app = App {
        title,
        proxy: event_loop.create_proxy(),
        window: None,
        state: State::Uninitialized,
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
                let name = backend_name(gpu.backend());
                platform::report_status(&format!("Running on {name}"));
                if let Some(window) = &self.window {
                    window.set_title(&format!("{} ({name})", self.title));
                    window.request_redraw();
                }
                self.state = State::Running(Box::new(gpu));
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
            WindowEvent::Resized(size) => {
                if let State::Running(gpu) = &mut self.state {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }
}

impl App {
    fn redraw(&mut self) {
        let (State::Running(gpu), Some(window)) = (&mut self.state, &self.window) else {
            return;
        };
        if gpu.render_clear(CLEAR_COLOR) == FrameOutcome::SurfaceLost {
            match gpu.instance().create_surface(window.clone()) {
                Ok(surface) => gpu.replace_surface(surface),
                Err(e) => {
                    platform::report_error(&format!("Lost the rendering surface: {e}"));
                    self.state = State::Failed;
                    return;
                }
            }
        }
        window.request_redraw();
    }
}
