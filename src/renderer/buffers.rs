//! Functions for dealing with vertex buffers, index buffers, and so on.

use std::{mem::size_of, ptr};

use ash::{vk, Device, Instance};
use color_eyre::Result;

use crate::{
    app::AppData,
    vertex::{Vertex, VERTICES},
};

use super::memory::get_memory_type_index;

/// Create vertex buffers for use by the app.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub unsafe fn create_vertex_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Create a vertex buffer for our static set of vertices (in lieu of proper model loading).
    let buffer_info = vk::BufferCreateInfo::builder()
        .size((size_of::<Vertex>() * VERTICES.len()) as u64)
        .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
        // this buffer will only be used from the graphics queue
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    data.vertex_buffer = device.create_buffer(&buffer_info, None)?;

    // Allocate some memory for the vertices.
    let mem_requirements = device.get_buffer_memory_requirements(data.vertex_buffer);
    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data.physical_device,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            mem_requirements,
        )?);

    data.vertex_buffer_memory = device.allocate_memory(&memory_info, None)?;

    // Associate the memory with the buffer
    device.bind_buffer_memory(data.vertex_buffer, data.vertex_buffer_memory, 0)?;

    // Copy the vertices into the newly-allocated memory
    {
        let memory = device.map_memory(
            data.vertex_buffer_memory,
            0,
            buffer_info.size,
            vk::MemoryMapFlags::empty(),
        )?;
        ptr::copy_nonoverlapping(VERTICES.as_ptr(), memory.cast(), VERTICES.len());
        device.unmap_memory(data.vertex_buffer_memory);
    }

    Ok(())
}

/// Destroy the vertex buffer created in [`create_vertex_buffer()`].
pub unsafe fn destroy_vertex_buffer(device: &Device, data: &AppData) {
    device.destroy_buffer(data.vertex_buffer, None);
    device.free_memory(data.vertex_buffer_memory, None);
}
