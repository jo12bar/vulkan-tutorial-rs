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

    // Bind a combined image sampler for the fragment shader
    let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let bindings = &[*mvp_mat_binding, *sampler_binding];
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

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

/// Create a memory pool to allocate descriptor sets from.
///
/// Dependent on the number of swapchain images created, so recreate this pool
/// if you recreate the swapchain. Make sure to deallocate the pre-exisiting
/// descriptor pool with [`destroy_descriptor_pool()`] first.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub unsafe fn create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
    let ubo_size = vk::DescriptorPoolSize::builder()
        .ty(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(data.swapchain_images.len() as u32);

    let sampler_size = vk::DescriptorPoolSize::builder()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(data.swapchain_images.len() as u32);

    let pool_sizes = &[*ubo_size, *sampler_size];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(data.swapchain_images.len() as u32);

    data.descriptor_pool = device.create_descriptor_pool(&info, None)?;

    Ok(())
}

/// Destroy the descriptor pool allocated by [`create_descriptor_pool()`].
pub unsafe fn destroy_descriptor_pool(device: &Device, data: &AppData) {
    device.destroy_descriptor_pool(data.descriptor_pool, None);
}

/// Create descriptor sets for sending to the GPU. Requires a descriptor pool
/// allocated by [`create_descriptor_pool()`].
///
/// Creates one descriptor set per swapchain image, all with the same layout.
/// Descriptor sets must be recreated if the swapchain is recreated.
///
/// Descriptor sets will be automatically freed when the descriptor pool is
/// freed with [`destroy_descriptor_pool()`].
#[tracing::instrument(level = "DEBUG", skip_all)]
pub unsafe fn create_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
    // Allocate the descriptor sets from the pool
    let layouts = vec![data.descriptor_set_layout; data.swapchain_images.len()];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts);

    data.descriptor_sets = device.allocate_descriptor_sets(&info)?;

    // Populate the descriptor sets
    for i in 0..data.swapchain_images.len() {
        // Define access to the model-view-projection matrix
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(data.uniform_buffers[i])
            .offset(0)
            .range(size_of::<MvpMat>() as u64);

        let mvp_mat_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&info));

        // Define access to the combined image sampler
        let info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(data.texture_image_view)
            .sampler(data.texture_sampler);

        let sampler_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&info));

        device.update_descriptor_sets(&[*mvp_mat_write, *sampler_write], &[] as _);
    }

    Ok(())
}
