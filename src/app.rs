use crate::{
    model::load_model,
    mvp_matrix::MvpMat,
    renderer::{
        buffers::{
            create_index_buffer, create_vertex_buffer, destroy_index_buffer, destroy_vertex_buffer,
        },
        commands::{create_command_buffers, create_command_pool},
        depth_tests::create_depth_objects,
        devices::{create_logical_device, pick_physical_device},
        extensions::Extensions,
        instance::create_instance,
        multisampling::create_color_objects,
        pipeline::{create_framebuffers, create_pipeline, create_render_pass},
        swapchain::{create_swapchain, create_swapchain_image_views},
        synchronization::{create_sync_objects, destroy_sync_objects},
        texture::{create_texture_image, create_texture_image_view, create_texture_sampler},
        uniforms::{
            create_descriptor_pool, create_descriptor_set_layout, create_descriptor_sets,
            create_uniform_buffers, destroy_descriptor_pool, destroy_uniform_buffers,
        },
        validation::should_enable_validation_layers,
    },
    vertex::Vertex,
    MAX_FRAMES_IN_FLIGHT,
};

use std::mem::size_of;
use std::ptr;
use std::time::Instant;

use ash::{
    extensions::{ext as vk_ext, khr as vk_khr},
    vk, Device, Entry, Instance,
};
use color_eyre::{
    eyre::{eyre, Context},
    Result,
};
use nalgebra_glm as glm;
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

    /// Tracks if the window has been resized and the app needs to handle that.
    /// Set to true to signal to the app that the window has been resized.
    /// If it's set to false at some point, then the app has handled the resize
    /// event.
    resized: bool,

    /// Global model-view-projection matrix.
    mvp_mat: MvpMat,

    /// The time that the last frame was rendered at. Used for keeping basic
    /// animations temporally accurate, regardless of framerate.
    ///
    /// Because of Vulkan's asynchronus nature, this isn't the *actual* time the
    /// last frame was rendered at - but it's good enough for uniform buffers,
    /// especially when we make the CPU wait for the GPU to render
    /// MAX_FRAMES_IN_FLIGHT frames with memory fences.
    last_frame_time: Instant,
}

/// Vulkan handles and associated properties used by our Vulkan [`App`].
#[derive(Clone, Debug, Default)]
pub struct AppData {
    pub surface: vk::SurfaceKHR,

    pub physical_device: vk::PhysicalDevice,
    pub msaa_samples: vk::SampleCountFlags,

    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,

    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,

    pub render_pass: vk::RenderPass,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,

    pub framebuffers: Vec<vk::Framebuffer>,

    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_buffer_memory: vk::DeviceMemory,
    /// One uniform buffer per swapchain image, because we refer to it from
    /// each swapchain image's command buffer.
    pub uniform_buffers: Vec<vk::Buffer>,
    pub uniform_buffers_memory: Vec<vk::DeviceMemory>,
    pub descriptor_pool: vk::DescriptorPool,
    /// One descriptor set per swapchain image.
    pub descriptor_sets: Vec<vk::DescriptorSet>,

    pub color_image: vk::Image,
    pub color_image_memory: vk::DeviceMemory,
    pub color_image_view: vk::ImageView,

    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,

    pub texture_image: vk::Image,
    pub texture_image_memory: vk::DeviceMemory,
    pub texture_image_format: vk::Format,
    pub texture_image_view: vk::ImageView,
    pub texture_sampler: vk::Sampler,

    /// The count of mip-map levels for the model's textures.
    ///
    /// Calculate with something like:
    ///
    /// ```ignore
    /// app_data.mip_levels = (img_width.max(img_height) as f32).log2().floor() as u32 + 1;
    /// ```
    ///
    /// ...which calculates how many times the largest dimension can be divided by 2, while ensuring
    /// that at least one mip level (the original image) is generated.
    pub mip_levels: u32,

    pub command_pool: vk::CommandPool,
    /// Note that command buffers are automatically destroyed when the [`vk::CommandPool`]
    /// they're allocated from is destroyed. One per swapchain image.
    pub command_buffers: Vec<vk::CommandBuffer>,
    /// This command pool should only be used for very short-lived command buffers.
    /// That's why there's no place in this struct to store buffers allocated from
    /// it.
    pub transient_command_pool: vk::CommandPool,

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
        create_render_pass(&instance, &device, &mut data)?;
        create_descriptor_set_layout(&device, &mut data)?;
        create_pipeline(&device, &mut data)?;

        debug!("Creating command pools");
        create_command_pool(&entry, &instance, &device, &mut data)?;

        debug!("Creating multi-sampled color objects");
        create_color_objects(&instance, &device, &mut data)?;

        debug!("Creating depth-test objects");
        create_depth_objects(&instance, &device, &mut data)?;

        debug!("Creating framebuffers");
        create_framebuffers(&device, &mut data)?;

        debug!("Creating command, vertex, index, and uniform buffers, and loading textures");

        let (texture_image, texture_image_memory, texture_image_format, mip_levels) =
            create_texture_image(
                &instance,
                &device,
                &mut data,
                "./resources/viking-room/viking-room.png",
            )?;
        data.texture_image = texture_image;
        data.texture_image_memory = texture_image_memory;
        data.texture_image_format = texture_image_format;
        data.mip_levels = mip_levels;
        data.texture_image_view = create_texture_image_view(
            &device,
            data.texture_image,
            data.texture_image_format,
            data.mip_levels,
        )?;
        data.texture_sampler = create_texture_sampler(&device, &data)?;

        load_model(&mut data, "./resources/viking-room/viking-room.obj")?;
        create_vertex_buffer(&instance, &device, &mut data)?;
        create_index_buffer(&instance, &device, &mut data)?;

        create_uniform_buffers(&instance, &device, &mut data)?;
        create_descriptor_pool(&device, &mut data)?;
        create_descriptor_sets(&device, &mut data)?;

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
            resized: false,
            mvp_mat: MvpMat::default(),
            last_frame_time: Instant::now(),
        })
    }

    /// Trigger an app resize. Call this if the window manager has indicated that
    /// the window has resized. The app will then handle re-creating swapchains,
    /// graphics pipelines, and so on the next time [`App::render()`] is called.
    #[inline]
    pub fn trigger_resize(&mut self) {
        self.resized = true;
    }

    /// Re-creates the swapchain, which is required when (for example) the window
    /// is resized.
    ///
    /// Due to Vulkan's hierarchy of object dependencies, this also recreates the
    /// render passes, graphics pipeline, framebuffers, and command buffers.
    ///
    /// # Safety
    ///
    /// Lol you thought
    #[tracing::instrument(level = "DEBUG", name = "App::recreate_swapchain", skip_all)]
    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.device.device_wait_idle()?;
        self.destroy_swapchain();

        create_swapchain(
            window,
            &self.entry,
            &self.instance,
            &self.device,
            &mut self.data,
        )?;
        create_swapchain_image_views(&self.device, &mut self.data)?;
        create_render_pass(&self.instance, &self.device, &mut self.data)?;
        create_pipeline(&self.device, &mut self.data)?;
        create_color_objects(&self.instance, &self.device, &mut self.data)?;
        create_depth_objects(&self.instance, &self.device, &mut self.data)?;
        create_framebuffers(&self.device, &mut self.data)?;
        create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        create_descriptor_pool(&self.device, &mut self.data)?;
        create_descriptor_sets(&self.device, &mut self.data)?;
        create_command_buffers(&self.device, &mut self.data)?;
        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());

        self.resized = false;

        Ok(())
    }

    /// Render a frame to the Vulkan app.
    ///
    /// # Safety
    ///
    /// Extremely unsafe &mdash; but faster.
    //#[tracing::instrument(level = "TRACE", name = "App::render", skip_all)]
    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        // If we already have MAX_FRAMES_IN_FLIGHT frames busy being rendered,
        // wait for them all to finish rendering before we submit a new frame.
        self.device
            .wait_for_fences(&[self.data.in_flight_fences[self.frame]], true, u64::MAX)?;

        // Acquire an image from the swapchain.
        let result = self.extensions.swapchain.acquire_next_image(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphores[self.frame],
            vk::Fence::null(),
        );

        // If the swapchain is now incompatible with the rendering surface (e.g the
        // window was resized) then recreate it and just return early.
        let image_index = match result {
            Ok((image_index, _swapchain_is_suboptimal)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => {
                return Err(eyre!(
                    "Acquiring swapchain image failed due to Vulkan error: [{0:?}] {0}",
                    e
                ))
            }
        };

        // If this particular image is already in flight, wait for it to finish!
        if self.data.images_in_flight[image_index as usize] != vk::Fence::null() {
            self.device.wait_for_fences(
                &[self.data.images_in_flight[image_index as usize]],
                true,
                u64::MAX,
            )?;
        }

        self.data.images_in_flight[image_index as usize] = self.data.in_flight_fences[self.frame];

        self.update_uniform_buffers(image_index)?;

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

        let result = self
            .extensions
            .swapchain
            .queue_present(self.data.present_queue, &present_info);

        // Recreate the swapchain if needed
        let swapchain_must_be_recreated =
            result == Ok(true) || result == Err(vk::Result::ERROR_OUT_OF_DATE_KHR);

        if self.resized || swapchain_must_be_recreated {
            self.recreate_swapchain(window)?;
        } else if let Err(e) = result {
            // some other error happened while queing the image for presentation
            return Err(eyre!(
                "Failed to queue image in swapchain for presentation due to Vulkan error: [{0:?}] {0}",
                e
            ));
        }

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    /// Update all uniform buffers that need updating. Should be called right
    /// after we wait for the fence for the acquired swapchain image to be
    /// signalled in the render loop.
    fn update_uniform_buffers(&mut self, image_index: u32) -> Result<()> {
        // Figure out how much time has approximately passed since the last frame.
        let now = Instant::now();
        let delta_t = (now - self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        // Update model-view-projection matrix
        self.mvp_mat
            // Rotate the model 90 degrees per second about its z axis
            .model_rotate_z(delta_t * glm::radians(&glm::vec1(90.0))[0])
            // Look at model from above at an angle
            .look_at(
                &glm::vec3(2.0, 2.0, 2.0),
                &glm::vec3(0.0, 0.0, 0.0),
                &glm::vec3(0.0, 0.0, 1.0),
            )
            // Use a perspective projection with a 45-degree vertical FOV. Make sure
            // use the current swapchain extent so the aspect ratio is correct!
            .perspective(
                self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32,
                glm::radians(&glm::vec1(45.0))[0],
                0.1,
                10.0,
            );

        // Send model-view-projection matrix to the GPU
        unsafe {
            // scope the memory-map pointer for safety
            let memory = self.device.map_memory(
                self.data.uniform_buffers_memory[image_index as usize],
                0,
                size_of::<MvpMat>() as u64,
                vk::MemoryMapFlags::empty(),
            )?;
            ptr::copy_nonoverlapping(&self.mvp_mat, memory.cast(), 1);
            self.device
                .unmap_memory(self.data.uniform_buffers_memory[image_index as usize]);
        }

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
        self.destroy_swapchain();

        self.device.destroy_sampler(self.data.texture_sampler, None);
        self.device
            .destroy_image_view(self.data.texture_image_view, None);
        self.device.destroy_image(self.data.texture_image, None);
        self.device
            .free_memory(self.data.texture_image_memory, None);

        self.device
            .destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);

        destroy_vertex_buffer(&self.device, &self.data);
        destroy_index_buffer(&self.device, &self.data);
        destroy_sync_objects(&self.device, &self.data);

        self.device
            .destroy_command_pool(self.data.command_pool, None);
        self.device
            .destroy_command_pool(self.data.transient_command_pool, None);

        self.device.destroy_device(None);

        vk_khr::Surface::new(&self.entry, &self.instance).destroy_surface(self.data.surface, None);

        if should_enable_validation_layers() {
            vk_ext::DebugUtils::new(&self.entry, &self.instance)
                .destroy_debug_utils_messenger(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);
    }

    /// Destroy objects associated with the swapchain.
    ///
    /// # Safety
    ///
    /// Will destroy you in 1v1 Halo deathmatch
    #[tracing::instrument(level = "DEBUG", name = "App::destroy_swapchain", skip_all)]
    unsafe fn destroy_swapchain(&mut self) {
        self.device
            .destroy_image_view(self.data.color_image_view, None);
        self.device.free_memory(self.data.color_image_memory, None);
        self.device.destroy_image(self.data.color_image, None);

        self.device
            .destroy_image_view(self.data.depth_image_view, None);
        self.device.free_memory(self.data.depth_image_memory, None);
        self.device.destroy_image(self.data.depth_image, None);

        destroy_descriptor_pool(&self.device, &self.data);
        destroy_uniform_buffers(&self.device, &self.data);

        self.data
            .framebuffers
            .iter()
            .for_each(|f| self.device.destroy_framebuffer(*f, None));

        self.device
            .free_command_buffers(self.data.command_pool, &self.data.command_buffers);

        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device
            .destroy_pipeline_layout(self.data.pipeline_layout, None);

        self.device.destroy_render_pass(self.data.render_pass, None);

        self.data
            .swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));
        self.extensions
            .swapchain
            .destroy_swapchain(self.data.swapchain, None);
    }
}
