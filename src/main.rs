use color_eyre::Result;
use tracing::{debug, info, warn};
use vk_tut::app::App;
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn main() -> Result<()> {
    setup_logging()?;

    let (event_loop, window) = build_window()?;

    info!("Initializing app");
    let mut app = unsafe { App::create(&window)? };
    let mut destroying = false;

    info!("Running event loop");
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
                info!("Destroying app");

                destroying = true;
                *control_flow = ControlFlow::Exit;

                unsafe {
                    app.wait_for_device_idle().unwrap();
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
