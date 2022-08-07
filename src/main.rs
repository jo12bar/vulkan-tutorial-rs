use ash::{
    extensions::{ext as vk_ext, khr as vk_khr},
    vk, Device, Entry, Instance,
};
use color_eyre::{
    eyre::{eyre, Context},
    Result,
};
use lazy_static::lazy_static;
use std::{collections::HashSet, ffi::CStr, os::raw::c_void};
use thiserror::Error;
use tracing::{debug, error, info, trace, warn};
use vk_tut::util::{PhysicalDeviceName, VkExtensionName};
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

/// Our Vulkan app.
#[derive(Clone)]
struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,
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

        debug!("Creating render surface on main window");
        data.surface = ash_window::create_surface(&entry, &instance, window, None)?;

        debug!("Selecting render device");
        pick_physical_device(&entry, &instance, &mut data)?;
        let device = create_logical_device(&entry, &instance, &mut data)?;

        debug!("Creating swapchain");
        create_swapchain(window, &entry, &instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;

        debug!("Creating render pipeline");
        create_pipeline(&device, &mut data)?;

        Ok(Self {
            entry,
            instance,
            data,
            device,
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
        self.device
            .destroy_pipeline_layout(self.data.pipeline_layout, None);

        self.data
            .swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));

        vk_khr::Swapchain::new(&self.instance, &self.device)
            .destroy_swapchain(self.data.swapchain, None);

        self.device.destroy_device(None);

        vk_khr::Surface::new(&self.entry, &self.instance).destroy_surface(self.data.surface, None);

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
    surface: vk::SurfaceKHR,

    physical_device: vk::PhysicalDevice,

    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,

    pipeline_layout: vk::PipelineLayout,

    /// For handling debug messages sent from Vulkan's validation layers.
    messenger: vk::DebugUtilsMessengerEXT,
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

/// For when a physical device does not satisfy some requirement of the application.
#[derive(Debug, Error)]
pub enum PhysicalDeviceSuitabilityError {
    #[error("Physical device is unsuitable: {0}")]
    Unsuitable(&'static str),
    #[error("Physical device is missing required extensions: {0}")]
    MissingExtensions(String),
    #[error("Error while querying physical device suitability: {0}")]
    VkError(#[from] vk::Result),
}

lazy_static! {
    /// Device extensions that are absolutely _required_. Don't put optional
    /// extensions into this list!
    static ref REQUIRED_DEVICE_EXTENSIONS: Vec<VkExtensionName> = [
        vk_khr::Swapchain::name()
    ]
        .into_iter()
        .map(VkExtensionName::from_cstr)
        .collect();
}

/// Picks a physical device to use for rendering.
#[tracing::instrument(level = "DEBUG", skip_all)]
unsafe fn pick_physical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut AppData,
) -> Result<()> {
    let mut valid_devices = Vec::new();

    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);
        let device_name = PhysicalDeviceName::from(properties.device_name);

        match check_physical_device(entry, instance, data, physical_device) {
            Ok(score) => valid_devices.push((physical_device, score, device_name, properties)),
            Err(err) => {
                debug!(device_name = %device_name, reason = %err, "Skipping physical device")
            }
        }
    }

    if valid_devices.is_empty() {
        return Err(eyre!(
            "Failed to find suitable physical device for rendering."
        ));
    }

    // Choose the highest-scoring device
    valid_devices.sort_unstable_by(|(_, score_a, _, _), (_, score_b, _, _)| score_a.cmp(score_b));
    let (physical_device, _, device_name, properties) = valid_devices.last().unwrap();

    data.physical_device = *physical_device;
    info!(
        device_name = %device_name,
        device_id = properties.device_id,
        "Selected physical device for rendering"
    );

    Ok(())
}

/// Check if a physical device satisfies all the requirements of this application.
/// Returns a score based on its properties and available features.
unsafe fn check_physical_device(
    entry: &Entry,
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<u32, PhysicalDeviceSuitabilityError> {
    let mut score = 0;

    let properties = instance.get_physical_device_properties(physical_device);
    match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => score += 200,
        vk::PhysicalDeviceType::INTEGRATED_GPU => score += 100,
        _ => {
            return Err(PhysicalDeviceSuitabilityError::Unsuitable(
                "Only discrete and integrated GPUs are supported",
            ))
        }
    }

    let features = instance.get_physical_device_features(physical_device);
    if features.geometry_shader != vk::TRUE {
        return Err(PhysicalDeviceSuitabilityError::Unsuitable(
            "Missing geometry shader support",
        ));
    }

    // if the following function call doesn't panic, then the device supports
    // all the queue families needed for this app. we just discard the queue
    // family indices immediately though.
    QueueFamilyIndices::get(entry, instance, data, physical_device)?;

    // check if this device has all the extensions we care about. If this doesn't
    // panic, then we're all good.
    check_physical_device_extensions(instance, physical_device)?;

    // Check swapchain support. We only care if there's at least one supported
    // image format and one supported presentation mode, given our window surface.
    // Note that we can only query for swapchain support after checking if the
    // VK_KHR_swapchain extension is available - so this must be done after
    // checking for physical device extensions.
    let swapchain_support = SwapchainSupport::get(entry, instance, data, physical_device)?;
    if swapchain_support.formats.is_empty() || swapchain_support.present_modes.is_empty() {
        return Err(PhysicalDeviceSuitabilityError::Unsuitable(
            "Insufficient swapchain support.",
        ));
    }

    Ok(score)
}

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<(), PhysicalDeviceSuitabilityError> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device)?
        .into_iter()
        .map(|e| VkExtensionName::from(e.extension_name))
        .collect::<HashSet<_>>();

    let mut missing_extensions = Vec::new();

    for ext in REQUIRED_DEVICE_EXTENSIONS.iter() {
        if !extensions.contains(ext) {
            missing_extensions.push(*ext);
        }
    }

    if missing_extensions.is_empty() {
        Ok(())
    } else {
        Err(PhysicalDeviceSuitabilityError::MissingExtensions(
            missing_extensions
                .iter()
                .map(|e| e.to_string_lossy())
                .collect::<Vec<_>>()
                .join(", "),
        ))
    }
}

/// Create a logical device for rendering from a physical device.
#[tracing::instrument(level = "DEBUG", skip_all)]
unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut AppData,
) -> Result<Device> {
    let qf_indices = QueueFamilyIndices::get(entry, instance, data, data.physical_device)?;

    let mut unique_qf_indices = HashSet::new();
    unique_qf_indices.insert(qf_indices.graphics);
    unique_qf_indices.insert(qf_indices.present);

    // Setup command queues
    let queue_priorities = &[1.0];
    let queue_infos = unique_qf_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
                .build()
        })
        .collect::<Vec<_>>();

    // Setup validation layers (if needed)
    let layers = if should_enable_validation_layers() {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    // Set up device-specific features
    let features = vk::PhysicalDeviceFeatures::builder();

    // Convert our list of absolutely-required extensions to a seires of
    // null-terminated string pointers.
    let extension_names = REQUIRED_DEVICE_EXTENSIONS
        .iter()
        .map(|ext| ext.as_ptr())
        .collect::<Vec<_>>();

    // Fill in the device info and create the device
    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_features(&features)
        .enabled_extension_names(&extension_names);

    let device = instance.create_device(data.physical_device, &info, None)?;
    data.graphics_queue = device.get_device_queue(qf_indices.graphics, 0);
    data.present_queue = device.get_device_queue(qf_indices.present, 0);

    Ok(device)
}

/// Stores the indices of queue families to be used by this application.
#[derive(Copy, Clone, Debug)]
struct QueueFamilyIndices {
    graphics: u32,
    present: u32,
}

impl QueueFamilyIndices {
    unsafe fn get(
        entry: &Entry,
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, PhysicalDeviceSuitabilityError> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);

        let surface_ext = vk_khr::Surface::new(entry, instance);

        let mut graphics = None;
        let mut present = None;
        for (i, properties) in properties.iter().enumerate() {
            if properties.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                // Supports rendering graphics to an image
                graphics = Some(i as u32);
            }

            if surface_ext.get_physical_device_surface_support(
                physical_device,
                i as u32,
                data.surface,
            )? {
                // Supports rendering to a surface attached to some window
                present = Some(i as u32);
            }

            if graphics.is_some() && present.is_some() {
                break;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(PhysicalDeviceSuitabilityError::Unsuitable(
                "Missing required queue families",
            ))
        }
    }
}

/// Create the swapchain.
#[tracing::instrument(level = "DEBUG", skip_all)]
unsafe fn create_swapchain(
    window: &Window,
    entry: &Entry,
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(entry, instance, data, data.physical_device)?;
    let swapchain_support = SwapchainSupport::get(entry, instance, data, data.physical_device)?;

    let surface_format = swapchain_support.get_preferred_surface_format();
    let present_mode = swapchain_support.get_preferred_present_mode();
    let extent = swapchain_support.get_swapchain_extent(window);

    // Decide on the number of images to include in the swapchain. We choose
    // the minimum + 1 to decrease the chance of having to wait for the driver
    // when trying to render a frame.
    // Make sure to not exceed the max image count though. A reported max image
    // count of 0 means that there is no maximum.
    let mut image_count = swapchain_support.capabilities.min_image_count + 1;
    if swapchain_support.capabilities.max_image_count != 0
        && image_count > swapchain_support.capabilities.max_image_count
    {
        image_count = swapchain_support.capabilities.max_image_count;
    }

    // Specify how to transfer images between the presentation queue and the
    // graphics queue. If the GPU uses the same queue for both presentation and
    // graphics, then use VK_SHARING_MODE_EXCLUSIVE for max performance.
    // Otherwise, use VK_SHARING_MODE_CONCURRENT so we don't have to deal with
    // manually transferring images ourselves (this incurs a performance penalty).
    let mut queue_family_indices = Vec::new();
    let image_sharing_mode = if indices.graphics != indices.present {
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    debug!(
        image_format = ?surface_format.format,
        image_color_space = ?surface_format.color_space,
        image_extent = ?extent,
        ?present_mode,
        ?image_sharing_mode,
        "Selected swapchain creation properties"
    );

    // Build the swapchain creation info struct
    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1) // 1 layer per image; we aren't doing stereo 3D
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT) // write directly to images
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(swapchain_support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        // TODO(jo12bar): Handle swapchain recreation on, e.g., window resizing
        .old_swapchain(vk::SwapchainKHR::null());

    let swapchain_ext = vk_khr::Swapchain::new(instance, device);
    data.swapchain = swapchain_ext.create_swapchain(&info, None)?;
    data.swapchain_images = swapchain_ext.get_swapchain_images(data.swapchain)?;
    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    Ok(())
}

/// Create basic views to access parts of the swapchain images.
#[tracing::instrument(level = "DEBUG", skip_all)]
unsafe fn create_swapchain_image_views(device: &Device, data: &mut AppData) -> Result<()> {
    debug!(
        count = data.swapchain_images.len(),
        "Creating swapchain image views"
    );

    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|i| {
            // Just keep color channels as they are
            let components = vk::ComponentMapping::builder()
                .r(vk::ComponentSwizzle::IDENTITY)
                .g(vk::ComponentSwizzle::IDENTITY)
                .b(vk::ComponentSwizzle::IDENTITY)
                .a(vk::ComponentSwizzle::IDENTITY);

            // Use the images as color targets without any mipmapping or
            // multiple layers.
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            // Build the image view creation info struct
            let info = vk::ImageViewCreateInfo::builder()
                .image(*i)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(data.swapchain_format)
                .components(*components)
                .subresource_range(*subresource_range);

            device.create_image_view(&info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

/// Stores the capabilities of a swapchain tied to a physical device. This allows
/// for checking if a swapchain is suitable for this application.
#[derive(Clone, Debug)]
struct SwapchainSupport {
    /// Basic surface capabilities, such as min/max nnumber of images or min/max
    /// width and height of images.
    capabilities: vk::SurfaceCapabilitiesKHR,

    /// Surface formats, such as supported pixel formats or color spaces.
    formats: Vec<vk::SurfaceFormatKHR>,

    /// Available presentation modes
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    /// Get supported capabilites, formats, and present modes associated with
    /// swapchains created by a physical device.
    unsafe fn get(
        entry: &Entry,
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, PhysicalDeviceSuitabilityError> {
        let surface_ext = vk_khr::Surface::new(entry, instance);

        Ok(Self {
            capabilities: surface_ext
                .get_physical_device_surface_capabilities(physical_device, data.surface)?,
            formats: surface_ext
                .get_physical_device_surface_formats(physical_device, data.surface)?,
            present_modes: surface_ext
                .get_physical_device_surface_present_modes(physical_device, data.surface)?,
        })
    }

    /// Get the preferred color format to use.
    ///
    /// We prefer 8-bit BGRA format pixels in the sRGB color space. However, if
    /// this format can't be found then the GPU's first reported color format
    /// will be returned.
    fn get_preferred_surface_format(&self) -> vk::SurfaceFormatKHR {
        *self
            .formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or_else(|| &self.formats[0])
    }

    /// Get the preferred presentation mode.
    ///
    /// If supported, this function will select VK_PRESENT_MODE_MAILBOX_KHR.
    /// Otherwise it will select VK_PRESENT_MODE_FIFO_KHR, which is gauranteed
    /// to always be available.
    fn get_preferred_present_mode(&self) -> vk::PresentModeKHR {
        self.present_modes
            .iter()
            .find(|m| **m == vk::PresentModeKHR::MAILBOX)
            .copied()
            .unwrap_or(vk::PresentModeKHR::FIFO)
    }

    /// Gets the resolution of the swapchain images, given a window handle.
    fn get_swapchain_extent(&self, window: &Window) -> vk::Extent2D {
        // Vulkan tells us to match the resolution of the window by setting the
        // width and height in the current_extent member. However, some window
        // managers do allow us to differ here and this is indicated by setting
        // the width and height in current_extent to a special value: the
        // maximum value of u32. In that case we'll pick the resolution that
        // best matches the window within the min_image_extent and
        // max_image_extent bounds.
        if self.capabilities.current_extent.width != u32::MAX {
            self.capabilities.current_extent
        } else {
            let size = window.inner_size();
            vk::Extent2D {
                width: size.width.clamp(
                    self.capabilities.min_image_extent.width,
                    self.capabilities.max_image_extent.width,
                ),
                height: size.height.clamp(
                    self.capabilities.min_image_extent.height,
                    self.capabilities.max_image_extent.height,
                ),
            }
        }
    }
}

/// Create a render pipeline.
#[tracing::instrument(level = "DEBUG", skip_all)]
unsafe fn create_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    // Include our pre-compiled shaders.
    let vert = include_bytes!("../shaders/shader.vert.spv");
    let frag = include_bytes!("../shaders/shader.frag.spv");

    // Wrap the bytecode in shader modules
    let vert_shader_module = create_shader_module(device, &vert[..])?;
    let frag_shader_module = create_shader_module(device, &frag[..])?;

    // Create shader stages
    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_shader_module)
        .name(CStr::from_bytes_with_nul_unchecked(b"main\0"));
    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(CStr::from_bytes_with_nul_unchecked(b"main\0"));

    // FIXME(jo12bar): Set up vertex buffers and all that later. Right now the
    // shaders have hard-coded vertices.
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();

    // Vertices will be assembled into regular-old triangles.
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    // Take up the entire rendering surface for the viewport
    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    // Draw the ENTIRE viewport
    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(data.swapchain_extent);

    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(std::slice::from_ref(&viewport))
        .scissors(std::slice::from_ref(&scissor));

    // Configure the rasterizer
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(false);

    // For now, disable multisampling
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    // Use alpha blending
    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(std::slice::from_ref(&attachment))
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    // We want to be able to modify viewport size and line width dynamically
    // without having to completely re-create the pipeline.
    let dynamic_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::LINE_WIDTH];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(dynamic_states);

    // Setup the pipeline layout, including things like shader uniforms
    let layout_info = vk::PipelineLayoutCreateInfo::builder();

    data.pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

    // Destroy the shader modules
    device.destroy_shader_module(vert_shader_module, None);
    device.destroy_shader_module(frag_shader_module, None);

    Ok(())
}

/// Create a shader module from SPIR-V shader bytecode and a GPU.
unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    // Realign the bytecode to a u32 slice
    let bytecode = Vec::<u8>::from(bytecode);
    let (prefix, code, suffix) = bytecode.align_to::<u32>();
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err(eyre!(
            "Unable to create shader module due to improper alignment of shader bytecode"
        ));
    }

    let info = vk::ShaderModuleCreateInfo::builder().code(code);

    Ok(device.create_shader_module(&info, None)?)
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
