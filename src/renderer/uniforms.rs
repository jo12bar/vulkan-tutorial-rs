//! Functions for binding uniform buffer objects to the GPU context.

use std::mem::size_of;

use ash::{vk, Device, Instance};
use color_eyre::Result;

use crate::{app::AppData, mvp_matrix::MvpMat};

use super::buffers::create_buffer;

/// Create descriptor set layouts, describing how shaders can access things like
/// uniform buffer objects. Call this before creating the pipeline - it needs
/// this info.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub unsafe fn create_descriptor_set_layout(device: &Device, data: &mut AppData) -> Result<()> {
    // Bind the model-view-projection matrix for the vertex shader
    let mvp_mat_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(std::slice::from_ref(&mvp_mat_binding));

    data.descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;

    Ok(())
}

/// Create as many uniform buffers as there are swapchain images for sending
/// uniform buffer objects to the GPU during rendering.
///
/// Uniform buffers must be re-created if the swapchain is re-created to ensure
/// that the number of buffers matches the number of swapchain images.
///
/// Clears out all pre-exisiting uniform buffers in the `data` struct, so make
/// sure to properly de-allocate them first using [`destroy_uniform_buffers()`].
#[tracing::instrument(level = "DEBUG", skip_all)]
pub unsafe fn create_uniform_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    data.uniform_buffers.clear();
    data.uniform_buffers_memory.clear();

    for _ in 0..data.swapchain_images.len() {
        // Create a buffer for the model-view-projection matrix for the vertex shader
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            device,
            data,
            size_of::<MvpMat>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        data.uniform_buffers.push(uniform_buffer);
        data.uniform_buffers_memory.push(uniform_buffer_memory);
    }

    Ok(())
}

/// Properly deallocate all uniform buffers created by [`create_uniform_buffers()`].
pub unsafe fn destroy_uniform_buffers(device: &Device, data: &AppData) {
    data.uniform_buffers
        .iter()
        .for_each(|b| device.destroy_buffer(*b, None));
    data.uniform_buffers_memory
        .iter()
        .for_each(|m| device.free_memory(*m, None));
}
