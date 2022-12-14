//! Functions for dealing with vertex buffers, index buffers, and so on.

use std::{mem::size_of, ptr};

use ash::{vk, Device, Instance};
use color_eyre::Result;

use crate::{app::AppData, vertex::Vertex};

use super::{
    commands::{begin_transient_commands, end_transient_commands},
    memory::get_memory_type_index,
};

/// Create vertex buffers for use by the app.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub unsafe fn create_vertex_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Create a vertex buffer for our static set of vertices (in lieu of proper model loading).
    let size = (size_of::<Vertex>() * data.vertices.len()) as u64;

    // First copy the vertices to a host-visible staging buffer
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    {
        // keep the memory map pointer inside this scope to avoid use-after-free
        let memory =
            device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
        ptr::copy_nonoverlapping(data.vertices.as_ptr(), memory.cast(), data.vertices.len());
        device.unmap_memory(staging_buffer_memory);
    }

    // Copy the vertices from the staging buffer to the highest-performance memory
    // buffer the GPU will give us
    let (vertex_buffer, vertex_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.vertex_buffer = vertex_buffer;
    data.vertex_buffer_memory = vertex_buffer_memory;

    copy_buffer(device, data, staging_buffer, vertex_buffer, size)?;

    // remember to free the staging buffer
    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

/// Destroy the vertex buffer created in [`create_vertex_buffer()`].
pub unsafe fn destroy_vertex_buffer(device: &Device, data: &AppData) {
    device.destroy_buffer(data.vertex_buffer, None);
    device.free_memory(data.vertex_buffer_memory, None);
}

/// Create index buffers for use by the app.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub unsafe fn create_index_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Create an index buffer for our static set of vertex indices (in lieu of proper model loading)
    let size = (size_of::<u32>() * data.indices.len()) as u64;

    // First copy the indices to a host-visible staging buffer
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    {
        // keep the memory map pointer inside this scope to avoid use-after-free
        let memory =
            device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
        ptr::copy_nonoverlapping(data.indices.as_ptr(), memory.cast(), data.indices.len());
        device.unmap_memory(staging_buffer_memory);
    }

    // Copy the indices from the staging buffer to the highest-performance memory
    // buffer the GPU will give us
    let (index_buffer, index_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.index_buffer = index_buffer;
    data.index_buffer_memory = index_buffer_memory;

    copy_buffer(device, data, staging_buffer, index_buffer, size)?;

    // remember to free the staging buffer
    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

/// Destroy the index buffer created in [`create_index_buffer()`].
pub unsafe fn destroy_index_buffer(device: &Device, data: &AppData) {
    device.destroy_buffer(data.index_buffer, None);
    device.free_memory(data.index_buffer_memory, None);
}

/// Create some type of buffer.
pub unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = device.create_buffer(&buffer_info, None)?;

    let requirements = device.get_buffer_memory_requirements(buffer);

    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data.physical_device,
            properties,
            requirements,
        )?);

    let buffer_memory = device.allocate_memory(&memory_info, None)?;
    device.bind_buffer_memory(buffer, buffer_memory, 0)?;

    Ok((buffer, buffer_memory))
}

/// Copy data from one buffer to another.
///
/// - The source buffer must have been created with the [`vk::BufferUsageFlags::TRANSFER_SRC`]
///   flag.
/// - The destination buffer must have been created with the
///   [`vk::BufferUsageFlags::TRANSFER_DST`] flag.
pub unsafe fn copy_buffer(
    device: &Device,
    data: &AppData,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<()> {
    // Start recording commands
    let command_buffer = begin_transient_commands(device, data)?;

    // Perform the copy
    let regions = vk::BufferCopy::builder().size(size);
    device.cmd_copy_buffer(
        command_buffer,
        source,
        destination,
        std::slice::from_ref(&regions),
    );

    // End recording commands and execute
    end_transient_commands(device, data, command_buffer)?;

    Ok(())
}
