//! Command buffer recording and allocating.

use super::devices::QueueFamilyIndices;
use crate::app::AppData;
use ash::{vk, Device, Entry, Instance};
use color_eyre::Result;

/// Create command pools, which manage the memory used to store command buffers.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub(crate) unsafe fn create_command_pools(
    entry: &Entry,
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Create a command pool specifically for "transient" command buffers,
    // which will be short-lived and will be reset or freed in a relatively
    // short timeframe. This can possibly enable memory allocation optimizations
    // by the implementation.
    data.transient_command_pool = create_transient_command_pool(entry, instance, device, data)?;

    // Create one command pool per swapchain image for use during rendering.
    let num_images = data.swapchain_images.len();
    for _ in 0..num_images {
        let command_pool = create_transient_command_pool(entry, instance, device, data)?;
        data.command_pools.push(command_pool);
    }

    Ok(())
}

/// Create a transient command pool for short-lived command buffers that can
/// be submitted to graphics queues.
unsafe fn create_transient_command_pool(
    entry: &Entry,
    instance: &Instance,
    device: &Device,
    data: &AppData,
) -> Result<vk::CommandPool> {
    let qf_indices = QueueFamilyIndices::get(entry, instance, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(qf_indices.graphics);

    Ok(device.create_command_pool(&info, None)?)
}

/// Create command buffers to use for rendering and such.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub(crate) unsafe fn create_command_buffers(data: &mut AppData) -> Result<()> {
    data.command_buffers = vec![vk::CommandBuffer::null(); data.framebuffers.len()];

    Ok(())
}

/// Begin recording commands into a command buffer meant for immediate execution
/// and deallocation.
pub unsafe fn begin_transient_commands(
    device: &Device,
    data: &AppData,
) -> Result<vk::CommandBuffer> {
    // Allocate a temporary command buffer
    let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.transient_command_pool)
        .command_buffer_count(1);
    let command_buffer = device.allocate_command_buffers(&info)?[0];

    // Start recording commands
    let info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device.begin_command_buffer(command_buffer, &info)?;

    Ok(command_buffer)
}

/// Stop recording a transient command buffer, submit it to the GPU for immediate
/// execution, wait for the GPU to catch up, and then deallocate the command
/// buffer.
pub unsafe fn end_transient_commands(
    device: &Device,
    data: &AppData,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    // End recording commands
    device.end_command_buffer(command_buffer)?;

    // Immediately execute the commands, and wait for completion
    let command_buffers = &[command_buffer];
    let info = vk::SubmitInfo::builder().command_buffers(command_buffers);

    device.queue_submit(data.graphics_queue, &[*info], vk::Fence::null())?;
    device.queue_wait_idle(data.graphics_queue)?;

    // Free the command buffer
    device.free_command_buffers(data.transient_command_pool, command_buffers);

    Ok(())
}
