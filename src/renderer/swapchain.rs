//! Tools for creating, re-creating, and checking the capabilities of swapchains.

use super::devices::{PhysicalDeviceSuitabilityError, QueueFamilyIndices};
use crate::app::AppData;
use ash::{extensions::khr as vk_khr, vk, Device, Entry, Instance};
use color_eyre::Result;
use tracing::debug;
use winit::window::Window;

use super::texture::create_image_view;

/// Create the swapchain.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub(crate) unsafe fn create_swapchain(
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
pub(crate) unsafe fn create_swapchain_image_views(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    debug!(
        count = data.swapchain_images.len(),
        "Creating swapchain image views"
    );

    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|i| {
            create_image_view(
                device,
                *i,
                data.swapchain_format,
                vk::ImageAspectFlags::COLOR,
                1,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

/// Stores the capabilities of a swapchain tied to a physical device. This allows
/// for checking if a swapchain is suitable for this application.
#[derive(Clone, Debug)]
pub(crate) struct SwapchainSupport {
    /// Basic surface capabilities, such as min/max nnumber of images or min/max
    /// width and height of images.
    pub(crate) capabilities: vk::SurfaceCapabilitiesKHR,

    /// Surface formats, such as supported pixel formats or color spaces.
    pub(crate) formats: Vec<vk::SurfaceFormatKHR>,

    /// Available presentation modes
    pub(crate) present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    /// Get supported capabilites, formats, and present modes associated with
    /// swapchains created by a physical device.
    pub(crate) unsafe fn get(
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
