use crate::renderer::{
    commands::{create_command_buffers, create_command_pool},
    devices::{create_logical_device, pick_physical_device},
    extensions::Extensions,
    instance::create_instance,
    pipeline::{create_framebuffers, create_pipeline, create_render_pass},
    swapchain::{create_swapchain, create_swapchain_image_views},
    synchronization::{create_sync_objects, destroy_sync_objects},
    validation::should_enable_validation_layers,
};
use crate::MAX_FRAMES_IN_FLIGHT;
use ash::{
    extensions::{ext as vk_ext, khr as vk_khr},
    vk, Device, Entry, Instance,
};
use color_eyre::{
    eyre::{eyre, Context},
    Result,
};
use tracing::debug;
use winit::window::Window;

/// Our Vulkan app.
#[derive(Clone)]
pub struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,
    extensions: Extensions,

    /// The frame we're currently rendering
    frame: usize,
}

/// Vulkan handles and associated properties used by our Vulkan [`App`].
#[derive(Clone, Debug, Default)]
pub struct AppData {
    pub surface: vk::SurfaceKHR,

    pub physical_device: vk::PhysicalDevice,

    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,

    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,

    pub render_pass: vk::RenderPass,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,

    pub framebuffers: Vec<vk::Framebuffer>,

    pub command_pool: vk::CommandPool,
    /// Note that command buffers are automatically destroyed when the [`vk::CommandPool`]
    /// they're allocated from is destroyed.
    pub command_buffers: Vec<vk::CommandBuffer>,

    /// Use for signaling that an image has been acquired from the swapchain and
    /// is ready for rendering.
    pub image_available_semaphores: Vec<vk::Semaphore>,
    /// Use for signaling that rendering has finished and that presentation
    /// may begin.
    pub render_finished_semaphores: Vec<vk::Semaphore>,

    /// Use for pausing the CPU until the GPU has finished rendering once we've
    /// submitted at least [`MAX_FRAMES_IN_FLIGHT`] frames.
    pub in_flight_fences: Vec<vk::Fence>,

    /// Keeps track of CPU-GPU fences while swapchain images are being rendered.
    /// This prevents rendering to a swapchain image that is already *in flight*.
    pub images_in_flight: Vec<vk::Fence>,

    /// For handling debug messages sent from Vulkan's validation layers.
    pub messenger: vk::DebugUtilsMessengerEXT,
}

impl App {
    /// Creates the Vulkan app, binding it to a surface generated by some winit
    /// window handle.
    ///
    /// # Safety
    ///
    /// Extremely unsafe. Makes uncountable numbers of calls to the Vulkan API,
    /// all of which could cause crashes or memory corruption at any point.
    ///
    /// Fun.
    #[tracing::instrument(level = "DEBUG", name = "App::create", skip_all)]
    pub unsafe fn create(window: &Window) -> Result<Self> {
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
        create_render_pass(&device, &mut data)?;
        create_pipeline(&device, &mut data)?;

        debug!("Creating framebuffers");
        create_framebuffers(&device, &mut data)?;

        debug!("Creating command buffers");
        create_command_pool(&entry, &instance, &device, &mut data)?;
        create_command_buffers(&device, &mut data)?;

        create_sync_objects(&device, &mut data)?;

        // Cache links to extensions
        let extensions = Extensions {
            swapchain: vk_khr::Swapchain::new(&instance, &device),
        };

        Ok(Self {
            entry,
            instance,
            data,
            device,
            extensions,
            frame: 0,
        })
    }

    /// Render a frame to the Vulkan app.
    ///
    /// # Safety
    ///
    /// Extremely unsafe &mdash; but faster.
    //#[tracing::instrument(level = "TRACE", name = "App::render", skip_all)]
    pub unsafe fn render(&mut self, _window: &Window) -> Result<()> {
        // If we already have MAX_FRAMES_IN_FLIGHT frames busy being rendered,
        // wait for them all to finish rendering before we submit a new frame.
        self.device
            .wait_for_fences(&[self.data.in_flight_fences[self.frame]], true, u64::MAX)?;

        // Acquire an image from the swapchain.
        let (image_index, _is_suboptimal) = self.extensions.swapchain.acquire_next_image(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphores[self.frame],
            vk::Fence::null(),
        )?;

        // If this particular image is already in flight, wait for it to finish!
        if self.data.images_in_flight[image_index as usize] != vk::Fence::null() {
            self.device.wait_for_fences(
                &[self.data.images_in_flight[image_index as usize]],
                true,
                u64::MAX,
            )?;
        }

        self.data.images_in_flight[image_index as usize] = self.data.in_flight_fences[self.frame];

        // Submit command buffers to the queue for rendering.
        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index as usize]];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device
            .reset_fences(&[self.data.in_flight_fences[self.frame]])?;

        self.device.queue_submit(
            self.data.graphics_queue,
            &[*submit_info],
            self.data.in_flight_fences[self.frame],
        )?;

        // Submit the result back to the swapchain to have it eventually show up on screen
        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        self.extensions
            .swapchain
            .queue_present(self.data.present_queue, &present_info)?;

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    /// Wait for the app's GPU to stop processing. Use this before destroying
    /// the application and its resources to avoid an immediate crash.
    ///
    /// # Safety
    ///
    /// Makes a raw call to Vulkan, which executes arbitrary driver code.
    pub unsafe fn wait_for_device_idle(&self) -> Result<()> {
        self.device.device_wait_idle()?;
        Ok(())
    }

    /// Destroys the Vulkan app. If this isn't called, then resources may be leaked.
    #[tracing::instrument(level = "DEBUG", name = "App::destroy", skip_all)]
    pub unsafe fn destroy(&mut self) {
        destroy_sync_objects(&self.device, &self.data);

        self.device
            .destroy_command_pool(self.data.command_pool, None);

        self.data
            .framebuffers
            .iter()
            .for_each(|f| self.device.destroy_framebuffer(*f, None));

        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
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
