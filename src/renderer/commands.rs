//! Command buffer recording and allocating.

use super::devices::QueueFamilyIndices;
use crate::{
    app::AppData,
    mvp_matrix::{MvpMat, MvpMatPushConstants},
};
use ash::{vk, Device, Entry, Instance};
use color_eyre::Result;

/// Create a command pool, which manages the memory used to store command buffers.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub(crate) unsafe fn create_command_pool(
    entry: &Entry,
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let qf_indices = QueueFamilyIndices::get(entry, instance, data, data.physical_device)?;

    // Create a regular command pool for command buffers that might live a relatively long time.
    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::empty())
        .queue_family_index(qf_indices.graphics);

    data.command_pool = device.create_command_pool(&info, None)?;

    // Create a command pool specifically for "transient" command buffers,
    // which will be short-lived and will be reset or freed in a relatively
    // short timeframe. This can possibly enable memory allocation optimizations
    // by the implementation.
    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(qf_indices.graphics);

    data.transient_command_pool = device.create_command_pool(&info, None)?;

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

    let mvp_matrix = MvpMat::default();
    let mvp_matrix_pcs = mvp_matrix.as_push_constants();
    let model_mat = mvp_matrix_pcs.model;
    let (_, model_mat_bytes, _) = model_mat.as_slice().align_to::<u8>();

    let opacity = 0.25_f32;

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
        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        let clear_values = &[color_clear_value, depth_clear_value];

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
        device.cmd_bind_vertex_buffers(*command_buffer, 0, &[data.vertex_buffer], &[0]);
        device.cmd_bind_index_buffer(*command_buffer, data.index_buffer, 0, vk::IndexType::UINT32);
        device.cmd_bind_descriptor_sets(
            *command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            data.pipeline_layout,
            0,
            &[data.descriptor_sets[i]],
            &[],
        );
        device.cmd_push_constants(
            *command_buffer,
            data.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            model_mat_bytes,
        );
        device.cmd_push_constants(
            *command_buffer,
            data.pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            std::mem::size_of::<MvpMatPushConstants>() as u32,
            &opacity.to_ne_bytes()[..],
        );
        device.cmd_draw_indexed(*command_buffer, data.indices.len() as u32, 1, 0, 0, 0); // three vertices hardcoded in vertex shader

        // End drawing
        device.cmd_end_render_pass(*command_buffer);

        // End the command buffer
        device.end_command_buffer(*command_buffer)?;
    }

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
