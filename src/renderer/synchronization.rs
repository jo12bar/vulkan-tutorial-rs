//! Vulkan synchronization utilities.

use crate::{app::AppData, MAX_FRAMES_IN_FLIGHT};
use ash::{vk, Device};
use color_eyre::Result;

/// Create Vulkan synchronization objects, such as semaphores.
pub(crate) unsafe fn create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        data.image_available_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);
        data.render_finished_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);

        data.in_flight_fences
            .push(device.create_fence(&fence_info, None)?);
    }

    data.images_in_flight = data
        .swapchain_images
        .iter()
        .map(|_| vk::Fence::null())
        .collect();

    Ok(())
}

/// Destroy Vulkan synchronization objects, such as semaphores.
pub(crate) unsafe fn destroy_sync_objects(device: &Device, data: &AppData) {
    data.render_finished_semaphores
        .iter()
        .for_each(|s| device.destroy_semaphore(*s, None));
    data.image_available_semaphores
        .iter()
        .for_each(|s| device.destroy_semaphore(*s, None));

    data.in_flight_fences
        .iter()
        .for_each(|f| device.destroy_fence(*f, None));
}
