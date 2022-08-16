//! Low-level functions for working with textures. This helps drive the material
//! system.

use std::{fmt::Debug, fs::File, path::Path, ptr};

use ash::{vk, Device, Instance};
use color_eyre::{eyre::eyre, Result};
use tracing::debug;

use crate::app::AppData;

use super::{
    buffers::create_buffer,
    commands::{begin_transient_commands, end_transient_commands},
    memory::get_memory_type_index,
};

/// Create a view into an image.
///
/// Remember to deallocate the image view before deallocating its image.
pub unsafe fn create_image_view(
    device: &Device,
    image: vk::Image,
    image_format: vk::Format,
) -> Result<vk::ImageView> {
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(image_format)
        .subresource_range(*subresource_range);

    Ok(device.create_image_view(&info, None)?)
}

/// Create a view into a texture image for use in a fragment shader.
///
/// Remember to deallocate the image view before deallocating its image.
#[inline]
#[tracing::instrument(level = "DEBUG", skip_all)]
pub unsafe fn create_texture_image_view(
    device: &Device,
    image: vk::Image,
    image_format: vk::Format,
) -> Result<vk::ImageView> {
    // Basically just a wrapper around create_image_view(), don't worry about it :P
    create_image_view(device, image, image_format)
}

/// Load a PNG image as a texture.
///
/// Returns a Vulkan handle to the created image object and a handle to the
/// device memory used to allocate it, as well as the Vulkan format of the
/// texture (for later reference).
///
/// # Notes
///
/// - All indexed images will be converted to RGB images.
/// - Any grayscale image with bitdepth less than 8-bit will be converted to 8-bit.
/// - All 16-bit images will be stripped to 8-bit.
///
/// # A note on colorspaces
///
/// This function assumes that all PNG images use the sRGB colorspace. While
/// this is commonly true, it isn't gauranteed - images may look weird if
/// they aren't encoded in the nonlinear sRGB format. This applies to grayscale
/// images too; it is assumed that the single grayscale format is encoded
/// nonlinearly as if it were an sRGB image.
#[tracing::instrument(level = "DEBUG", skip_all, fields(path = ?path))]
pub unsafe fn create_texture_image<P>(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    path: P,
) -> Result<(vk::Image, vk::DeviceMemory, vk::Format)>
where
    P: AsRef<Path> + Debug,
{
    // Open and read the image
    let file = File::open(&path)?;

    let mut decoder = png::Decoder::new(file);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);

    let mut reader = decoder.read_info()?;
    let mut pixels = vec![0; reader.output_buffer_size()];
    let size = reader.output_buffer_size() as u64;

    let img_info = reader.next_frame(&mut pixels)?;
    let vk_format = get_vulkan_image_format(img_info.color_type, img_info.bit_depth);

    debug!(
        ?path,
        width = img_info.width,
        height = img_info.height,
        size,
        line_size = img_info.line_size,
        color_type = ?img_info.color_type,
        bit_depth = ?img_info.bit_depth,
        vk_format = ?vk_format,
        "Successfully read image"
    );

    // Copy the image into a host-visible staging buffer
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    {
        // scope the mapped memory handle for safety
        let memory =
            device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
        ptr::copy_nonoverlapping(pixels.as_ptr(), memory.cast(), pixels.len());
        device.unmap_memory(staging_buffer_memory);
    }

    // Build the image object and allocate memory
    let (texture_image, texture_image_memory) = create_image(
        instance,
        device,
        data,
        img_info.width,
        img_info.height,
        vk_format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    // Prepare the image to be a copy destination
    transition_image_layout(
        device,
        data,
        texture_image,
        vk_format,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    )?;

    // Copy data from the staging buffer to the image object
    copy_buffer_to_image(
        device,
        data,
        staging_buffer,
        texture_image,
        img_info.width,
        img_info.height,
    )?;

    // Prepare the image for shader access
    transition_image_layout(
        device,
        data,
        texture_image,
        vk_format,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    )?;

    // Clean up the staging buffer
    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok((texture_image, texture_image_memory, vk_format))
}

#[allow(clippy::too_many_arguments)]
unsafe fn create_image(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    width: u32,
    height: u32,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    // Build the image object
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .samples(vk::SampleCountFlags::TYPE_1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let image = device.create_image(&info, None)?;

    // Allocate memory for the image
    let requirements = device.get_image_memory_requirements(image);

    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data.physical_device,
            properties,
            requirements,
        )?);

    let image_memory = device.allocate_memory(&info, None)?;
    device.bind_image_memory(image, image_memory, 0)?;

    Ok((image, image_memory))
}

/// Transition an image object from one layout to another.
///
/// Returns an error if an unimplemented combination of layout transitions is
/// requested.
unsafe fn transition_image_layout(
    device: &Device,
    data: &AppData,
    image: vk::Image,
    _format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) =
        match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),

            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),

            _ => return Err(eyre!("Unsupported image layout transition")),
        };

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(*subresource)
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask);

    // Start recording commands
    let command_buffer = begin_transient_commands(device, data)?;

    device.cmd_pipeline_barrier(
        command_buffer,
        src_stage_mask,
        dst_stage_mask,
        vk::DependencyFlags::empty(),
        &[] as _,
        &[] as _,
        &[*barrier],
    );

    // End recording commands & immediately execute
    end_transient_commands(device, data, command_buffer)?;

    Ok(())
}

/// Copy data from a staging buffer to an image object.
unsafe fn copy_buffer_to_image(
    device: &Device,
    data: &AppData,
    src_buffer: vk::Buffer,
    dst_image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let subresource = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);

    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(*subresource)
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    // Start recording commands
    let command_buffer = begin_transient_commands(device, data)?;

    device.cmd_copy_buffer_to_image(
        command_buffer,
        src_buffer,
        dst_image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[*region],
    );

    // End recording commands & immediately execute
    end_transient_commands(device, data, command_buffer)?;

    Ok(())
}

fn get_vulkan_image_format(color_type: png::ColorType, bit_depth: png::BitDepth) -> vk::Format {
    use png::{BitDepth, ColorType};
    use vk::Format;

    assert_ne!(
        color_type,
        ColorType::Indexed,
        "PNGs with indexed colors are unsupported by this function."
    );

    assert_ne!(
        bit_depth,
        BitDepth::Sixteen,
        "PNGs with 16-bit color are unsupported by this function."
    );

    match color_type {
        ColorType::Grayscale => match bit_depth {
            BitDepth::One | BitDepth::Two | BitDepth::Four | BitDepth::Eight => Format::R8_SRGB,
            BitDepth::Sixteen => unreachable!(),
        },

        ColorType::GrayscaleAlpha => match bit_depth {
            BitDepth::One | BitDepth::Two | BitDepth::Four | BitDepth::Eight => Format::R8G8_SRGB,
            BitDepth::Sixteen => unreachable!(),
        },

        ColorType::Rgb => match bit_depth {
            BitDepth::One | BitDepth::Two | BitDepth::Four | BitDepth::Eight => Format::R8G8B8_SRGB,
            BitDepth::Sixteen => unreachable!(),
        },

        ColorType::Rgba => match bit_depth {
            BitDepth::One | BitDepth::Two | BitDepth::Four | BitDepth::Eight => {
                Format::R8G8B8A8_SRGB
            }
            BitDepth::Sixteen => unreachable!(),
        },

        ColorType::Indexed => unreachable!(),
    }
}

/// Create a texture sampler for sampling texture images from fragment shaders.
pub unsafe fn create_texture_sampler(device: &Device) -> Result<vk::Sampler> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(0.0);

    Ok(device.create_sampler(&info, None)?)
}
