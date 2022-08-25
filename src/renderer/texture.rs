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
    image_aspects: vk::ImageAspectFlags,
    mip_levels: u32,
) -> Result<vk::ImageView> {
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(image_aspects)
        .base_mip_level(0)
        .level_count(mip_levels)
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
    mip_levels: u32,
) -> Result<vk::ImageView> {
    create_image_view(
        device,
        image,
        image_format,
        vk::ImageAspectFlags::COLOR,
        mip_levels,
    )
}

/// Load a PNG image as a texture.
///
/// Returns a Vulkan handle to the created image object and a handle to the
/// device memory used to allocate it, the Vulkan format of the
/// texture (for later reference), and finally the number of mip levels to
/// generate for the image.
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
) -> Result<(vk::Image, vk::DeviceMemory, vk::Format, u32)>
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

    // Calculate the number of mip levels for the image based on how many times
    // the largest dimension can be divded in two.
    let mip_levels = (img_info.width.max(img_info.height) as f32).log2().floor() as u32 + 1;

    debug!(
        ?path,
        width = img_info.width,
        height = img_info.height,
        size,
        line_size = img_info.line_size,
        color_type = ?img_info.color_type,
        bit_depth = ?img_info.bit_depth,
        vk_format = ?vk_format,
        mip_levels,
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
        mip_levels,
        vk::SampleCountFlags::TYPE_1,
        vk_format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    // Prepare the image to be a copy destination
    transition_image_layout(
        device,
        data,
        texture_image,
        vk_format,
        mip_levels,
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

    // Generate image mipmaps and transition image for fragment shader use
    generate_mipmaps(
        instance,
        device,
        data,
        texture_image,
        vk_format,
        img_info.width,
        img_info.height,
        mip_levels,
    )?;

    // Clean up the staging buffer
    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok((texture_image, texture_image_memory, vk_format, mip_levels))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn create_image(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    width: u32,
    height: u32,
    mip_levels: u32,
    samples: vk::SampleCountFlags,
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
        .mip_levels(mip_levels)
        .array_layers(1)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .samples(samples)
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
pub unsafe fn transition_image_layout(
    device: &Device,
    data: &AppData,
    image: vk::Image,
    format: vk::Format,
    mip_levels: u32,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) =
        match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            ),

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

    let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
        match format {
            vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT => {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            }
            vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
            _ => vk::ImageAspectFlags::COLOR,
        }
    } else {
        vk::ImageAspectFlags::COLOR
    };

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(aspect_mask)
        .base_mip_level(0)
        .level_count(mip_levels)
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

#[allow(clippy::too_many_arguments)]
pub unsafe fn generate_mipmaps(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> Result<()> {
    if !instance
        .get_physical_device_format_properties(data.physical_device, format)
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    {
        return Err(eyre!(
            "Image with format {:?} does not support linear blitting, so mipmaps cannot be generated",
            format
        ));
    }

    let cmd_buf = begin_transient_commands(device, data)?;

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);

    let mut barrier = vk::ImageMemoryBarrier::builder()
        .image(image)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(*subresource);

    let mut mip_width = width;
    let mut mip_height = height;

    for i in 1..mip_levels {
        // Transition level i - 1 from TRANSFER_DST_OPTIMAL to TRANSFER_SRC_OPTIMAL
        // so we can use it as a source for mip generation. Use a barrier to wait
        // for level i - 1 to be filled before executing the blit command for
        // layer i.
        barrier.subresource_range.base_mip_level = i - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

        device.cmd_pipeline_barrier(
            cmd_buf,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as _,
            &[] as _,
            &[*barrier],
        );

        // Specify the regions used in the blit operation.
        // The source level is i - 1, and the destination level is i.
        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i - 1)
            .base_array_layer(0)
            .layer_count(1);

        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i)
            .base_array_layer(0)
            .layer_count(1);

        let blit = vk::ImageBlit::builder()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width as i32,
                    y: mip_height as i32,
                    z: 1,
                },
            ])
            .src_subresource(*src_subresource)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: (if mip_width > 1 { mip_width / 2 } else { 1 }) as i32,
                    y: (if mip_height > 1 { mip_height / 2 } else { 1 }) as i32,
                    z: 1,
                },
            ])
            .dst_subresource(*dst_subresource);

        // Actually record the blit command
        device.cmd_blit_image(
            cmd_buf,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[*blit],
            vk::Filter::LINEAR,
        );

        // Transition level i - 1 to be optimal for fragment shader reading.
        // This transition wiats on the blit command to finish, and all sampling
        // operations will wait on this transition to finish.
        barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        device.cmd_pipeline_barrier(
            cmd_buf,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as _,
            &[] as _,
            &[*barrier],
        );

        // Calc next level's mip sizes, ensuring that the dimensions never become 0
        if mip_width > 1 {
            mip_width /= 2;
        }

        if mip_height > 1 {
            mip_height /= 2;
        }
    }

    // Remember to transition the very final mip level, after the loop ends,
    // to be optimal for fragment shader reading. Use a barrier to wait for
    // blitting to finish, and to prevent use in shaders until the transition
    // is done.
    barrier.subresource_range.base_mip_level = mip_levels - 1;
    barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

    device.cmd_pipeline_barrier(
        cmd_buf,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[] as _,
        &[] as _,
        &[*barrier],
    );

    end_transient_commands(device, data, cmd_buf)?;

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
pub unsafe fn create_texture_sampler(device: &Device, data: &AppData) -> Result<vk::Sampler> {
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
        .max_lod(data.mip_levels as f32);

    Ok(device.create_sampler(&info, None)?)
}
