//! Command buffer recording and allocating.

use super::devices::QueueFamilyIndices;
use crate::app::AppData;
use ash::{vk, Device, Entry, Instance};
use color_eyre::Result;
use tracing::debug;

/// Create a command pool, which manages the memory used to store command buffers.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub(crate) unsafe fn create_command_pool(
    entry: &Entry,
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let qf_indices = QueueFamilyIndices::get(entry, instance, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::empty())
        .queue_family_index(qf_indices.graphics);

    data.command_pool = device.create_command_pool(&info, None)?;

    Ok(())
}

/// Create command buffers to use for rendering and such.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub(crate) unsafe fn create_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(data.framebuffers.len() as u32);

    data.command_buffers = device.allocate_command_buffers(&allocate_info)?;

    // Begin recording the command buffer necessary for drawing a triangle.
    debug!("Recording drawing commands for rendering a single triangle.");
    for (i, command_buffer) in data.command_buffers.iter().enumerate() {
        // Begin the command buffer with no inheritance from past command buffers,
        // and no flags.
        let info = vk::CommandBufferBeginInfo::builder();
        device.begin_command_buffer(*command_buffer, &info)?;

        // Render to the entire available image
        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(data.swapchain_extent);

        // Clear to opaque black
        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let clear_values = &[color_clear_value];

        // Begin drawing
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(data.render_pass)
            .framebuffer(data.framebuffers[i])
            .render_area(*render_area)
            .clear_values(clear_values);
        device.cmd_begin_render_pass(*command_buffer, &info, vk::SubpassContents::INLINE);

        device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            data.pipeline,
        );
        device.cmd_draw(*command_buffer, 3, 1, 0, 0); // three vertices hardcoded in vertex shader

        // End drawing
        device.cmd_end_render_pass(*command_buffer);

        // End the command buffer
        device.end_command_buffer(*command_buffer)?;
    }

    Ok(())
}
