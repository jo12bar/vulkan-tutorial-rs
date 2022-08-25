//! For performing depth tests on the rendered image to figure out what objects
//! are farther away than others.

use ash::{vk, Device, Instance};
use color_eyre::{eyre::eyre, Result};

use crate::app::AppData;

use super::texture::{create_image, create_image_view, transition_image_layout};

/// Create depth and (some day) stencil buffers.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub unsafe fn create_depth_objects(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Get the best format
    let format = get_depth_format(instance, data)?;

    // Create the depth image
    let (depth_image, depth_image_memory) = create_image(
        instance,
        device,
        data,
        data.swapchain_extent.width,
        data.swapchain_extent.height,
        1,
        format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.depth_image = depth_image;
    data.depth_image_memory = depth_image_memory;

    // Create a view for the depth image
    data.depth_image_view = create_image_view(
        device,
        data.depth_image,
        format,
        vk::ImageAspectFlags::DEPTH,
        1,
    )?;

    // Transition the depth image to the optimal layout
    transition_image_layout(
        device,
        data,
        data.depth_image,
        format,
        1,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    )?;

    Ok(())
}

/// Select a buffer format with a depth component that supports usage as depth
/// attachment.
pub unsafe fn get_depth_format(instance: &Instance, data: &AppData) -> Result<vk::Format> {
    let candidates = &[
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];

    get_supported_format(
        instance,
        data,
        candidates,
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )
}

/// From a list of desired buffer formats, from most desireable to least desireable,
/// selects the first satisfying application requirements.
unsafe fn get_supported_format(
    instance: &Instance,
    data: &AppData,
    candidates: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Result<vk::Format> {
    candidates
        .iter()
        .copied()
        .find(|f| {
            let properties =
                instance.get_physical_device_format_properties(data.physical_device, *f);

            match tiling {
                vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
                _ => false,
            }
        })
        .ok_or_else(|| eyre!("Failed to find supported buffer format"))
}
