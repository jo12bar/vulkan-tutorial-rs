use ash::{vk, Entry, Instance};
use color_eyre::{
    eyre::{eyre, Context},
    Result,
};
use std::ffi::CStr;
use tracing::{debug, info};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn main() -> Result<()> {
    setup_logging()?;

    let (event_loop, window) = build_window()?;

    debug!("Initializing app");
    let mut app = unsafe { App::create(&window)? };
    let mut destroying = false;

    debug!("Running event loop");
    event_loop.run(move |event, _, control_flow| {
        // Just continuously poll for events, never going to sleep (i.e. hot loop)
        *control_flow = ControlFlow::Poll;

        match event {
            // Render a frame if our Vulkan app is not being destroyed
            Event::MainEventsCleared if !destroying => unsafe { app.render(&window) }.unwrap(),

            // Destroy our Vulkan app if requested
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                debug!("Destroying app");

                destroying = true;
                *control_flow = ControlFlow::Exit;

                unsafe {
                    app.destroy();
                }

                info!("Goodbye.");
            }

            _ => {}
        }
    });
}

/// Create the window and event loop.
#[tracing::instrument(level = "DEBUG")]
fn build_window() -> Result<(EventLoop<()>, Window)> {
    debug!("Creating window of default size and event loop");

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Rusty Vulkan Tutorial")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    debug!(window_id = ?window.id(), "Window creation successful");

    Ok((event_loop, window))
}

/// Create a Vulkan instance from an entry point.
///
/// The window handle is required so that we can load the required extensions for
/// drawing to a window.
unsafe fn create_instance(window: &Window, entry: &Entry) -> Result<Instance> {
    let app_info = vk::ApplicationInfo::builder()
        .application_name(CStr::from_bytes_with_nul(b"Rusty Vulkan Tutorial\0")?)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(CStr::from_bytes_with_nul(b"Johann's Rust Special\0")?)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::make_api_version(0, 1, 0, 0));

    let extensions = ash_window::enumerate_required_extensions(&window)?;

    let instance_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_extension_names(extensions);

    Ok(entry.create_instance(&instance_info, None)?)
}

/// Our Vulkan app.
#[derive(Clone)]
struct App {
    entry: Entry,
    instance: Instance,
}

impl App {
    /// Creates the Vulkan app, binding it to a surface generated by some winit
    /// window handle.
    #[tracing::instrument(level = "DEBUG", name = "App::create", skip_all)]
    unsafe fn create(window: &Window) -> Result<Self> {
        debug!("Loading instance of Vulkan library");
        let entry = Entry::load()
            .map_err(|e| eyre!("{e}"))
            .wrap_err("Error loading Vulkan library")?;
        let instance = create_instance(window, &entry)?;

        Ok(Self { entry, instance })
    }

    /// Render a frame to the Vulkan app.
    #[tracing::instrument(level = "TRACE", name = "App::render", skip_all)]
    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        Ok(())
    }

    /// Destroys the Vulkan app. If this isn't called, then resources may be leaked.
    #[tracing::instrument(level = "DEBUG", name = "App::destroy", skip_all)]
    unsafe fn destroy(&mut self) {
        self.instance.destroy_instance(None);
    }
}

/// Vulkan handles and associates properties used by our Vulkan [`App`].
#[derive(Clone, Debug, Default)]
struct AppData {}

fn setup_logging() -> Result<()> {
    use tracing_subscriber::{prelude::*, EnvFilter};
    use tracing_tree::HierarchicalLayer;

    color_eyre::install()?;

    tracing_subscriber::registry()
        .with(HierarchicalLayer::new(4).with_bracketed_fields(true))
        .with(EnvFilter::from_default_env())
        .try_init()?;

    Ok(())
}
