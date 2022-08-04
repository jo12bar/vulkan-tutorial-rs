use ash::{extensions::ext as vk_ext, vk, Entry, Instance};
use color_eyre::{
    eyre::{eyre, Context},
    Result,
};
use std::{collections::HashSet, ffi::CStr, os::raw::c_void};
use tracing::{debug, error, info, trace, warn};
use vk_tut::util::VkExtensionName;
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
#[tracing::instrument(level = "DEBUG", skip_all)]
unsafe fn create_instance(window: &Window, entry: &Entry, data: &mut AppData) -> Result<Instance> {
    type DebugSeverity = vk::DebugUtilsMessageSeverityFlagsEXT;
    type DebugMsgType = vk::DebugUtilsMessageTypeFlagsEXT;

    let validation_enabled = should_enable_validation_layers();

    let app_info = vk::ApplicationInfo::builder()
        .application_name(CStr::from_bytes_with_nul(b"Rusty Vulkan Tutorial\0")?)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(CStr::from_bytes_with_nul(b"Johann's Rust Special\0")?)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::make_api_version(0, 1, 0, 0));

    // check the available validation layers so we can make sure our required
    // validation layer is supported
    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| VkExtensionName::from(l.layer_name))
        .collect::<HashSet<_>>();

    if validation_enabled && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(eyre!(
            "Validation layer requested, but not supported by this platform"
        ));
    }

    let layers = if validation_enabled {
        debug!(layer = ?VALIDATION_LAYER, "Enabling validation layer");
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    // Load Vulkan extensions

    let mut extensions = Vec::from(ash_window::enumerate_required_extensions(&window)?);

    if validation_enabled {
        debug!(extension = ?vk_ext::DebugUtils::name(), "Enabling extension");
        extensions.push(vk_ext::DebugUtils::name().as_ptr());
    }

    // Create the Vulkan instance

    let mut instance_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            DebugSeverity::VERBOSE
                | DebugSeverity::INFO
                | DebugSeverity::WARNING
                | DebugSeverity::ERROR,
        )
        .message_type(DebugMsgType::GENERAL | DebugMsgType::VALIDATION | DebugMsgType::PERFORMANCE)
        .pfn_user_callback(Some(vk_debug_callback));

    if validation_enabled {
        instance_info = instance_info.push_next(&mut debug_info);
    }

    let instance = entry.create_instance(&instance_info, None)?;

    // Connect the validation layer callback function if enabled
    if validation_enabled {
        data.messenger = vk_ext::DebugUtils::new(entry, &instance)
            .create_debug_utils_messenger(&debug_info, None)?;
    }

    Ok(instance)
}

/// Our Vulkan app.
#[derive(Clone)]
struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
}

impl App {
    /// Creates the Vulkan app, binding it to a surface generated by some winit
    /// window handle.
    #[tracing::instrument(level = "DEBUG", name = "App::create", skip_all)]
    unsafe fn create(window: &Window) -> Result<Self> {
        let mut data = AppData::default();

        debug!("Loading instance of Vulkan library");
        let entry = Entry::load()
            .map_err(|e| eyre!("{e}"))
            .wrap_err("Error loading Vulkan library")?;
        let instance = create_instance(window, &entry, &mut data)?;

        Ok(Self {
            entry,
            instance,
            data,
        })
    }

    /// Render a frame to the Vulkan app.
    //#[tracing::instrument(level = "TRACE", name = "App::render", skip_all)]
    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        Ok(())
    }

    /// Destroys the Vulkan app. If this isn't called, then resources may be leaked.
    #[tracing::instrument(level = "DEBUG", name = "App::destroy", skip_all)]
    unsafe fn destroy(&mut self) {
        if should_enable_validation_layers() {
            vk_ext::DebugUtils::new(&self.entry, &self.instance)
                .destroy_debug_utils_messenger(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);
    }
}

/// Vulkan handles and associates properties used by our Vulkan [`App`].
#[derive(Clone, Debug, Default)]
struct AppData {
    /// For handling debug messages sent from Vulkan's validation layers.
    messenger: vk::DebugUtilsMessengerEXT,
}

/// Returns true if Vulkan validation layers should be enabled.
///
/// Will always return true in builds where `debug_assertions` is enabled.
/// Otherwise, will only return true if the environment variable
/// `ENABLE_VULKAN_VALIDATION_LAYERS` is set.
#[inline]
fn should_enable_validation_layers() -> bool {
    cfg!(debug_assertions) || std::env::var("ENABLE_VULKAN_VALIDATION_LAYERS").is_ok()
}

/// The default Vulkan validation layer bundle to be used if [`should_enable_validation_layers()`]
/// returns true.
const VALIDATION_LAYER: VkExtensionName =
    VkExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation\0");

/// A callback function that will be called whenever Vulkan has a validation layer message to output.
extern "system" fn vk_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message_id = unsafe { CStr::from_ptr(data.p_message_id_name) }.to_string_lossy();
    let message = unsafe { CStr::from_ptr(data.p_message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!({"type" = ?typ, id = %message_id}, "{}", message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!({"type" = ?typ, id = %message_id}, "{}", message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!({"type" = ?typ, id = %message_id}, "{}", message);
    } else {
        trace!({"type" = ?typ, id = %message_id}, "{}", message);
    }

    vk::FALSE
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
