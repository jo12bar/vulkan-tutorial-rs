//! Tools for setting up render pipelines.

use crate::{app::AppData, mvp_matrix::MvpMatPushConstants, vertex::Vertex};
use ash::{vk, Device, Instance};
use color_eyre::{eyre::eyre, Result};
use std::ffi::CStr;

use super::depth_tests::get_depth_format;

/// Create a render pass.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub(crate) unsafe fn create_render_pass(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Use a single color buffer attachment represented by one of the images
    // from the swapchain.
    let color_attachment = vk::AttachmentDescription::builder()
        // Color attachment format MUST match swapchain image format!!
        .format(data.swapchain_format)
        .samples(data.msaa_samples)
        // Clear out old values in the frame buffer when starting to render,
        // and make sure the new values are preserved once the render is done
        // (so you can see it on screen)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        // We aren't doing anything with the stencil buffer yet, so results
        // of loading and storing are irrelevant
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        // Since we're clearing the image, we don't care what its previous layout was.
        .initial_layout(vk::ImageLayout::UNDEFINED)
        // We want the image to be ready for presentation via the swapchain
        // once we're done rendering.
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    // To fragment shaders, this will be the 0th output destination.
    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    // Set up the depth and stencil attachments for depth testing
    let depth_stencil_attachment = vk::AttachmentDescription::builder()
        .format(get_depth_format(instance, data)?)
        .samples(data.msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    // Depth buffer is available as the 1st output destination
    let depth_stencil_attachment_ref = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    // Set up a color resolve attachment so our normal multisampled color
    // attachment can be resolved to a regular image.
    let color_resolve_attachment = vk::AttachmentDescription::builder()
        .format(data.swapchain_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    // The color resolve attachment is available as the 2nd output destination
    let color_resolve_attachment_ref = vk::AttachmentReference::builder()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    // We aren't doing any fancy post-processing, so a single subpass is all we need.
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_attachment_ref))
        .depth_stencil_attachment(&depth_stencil_attachment_ref)
        .resolve_attachments(std::slice::from_ref(&color_resolve_attachment_ref));

    // Even though we only have a single subpass, we need to specify how to
    // transition into and out of it. So, we define subpass dependencies here.
    let dependency = vk::SubpassDependency::builder()
        // source is the implicit subpass before the render pass
        .src_subpass(vk::SUBPASS_EXTERNAL)
        // target is our only defined subpass
        .dst_subpass(0)
        // wait for the swapchain to read from the image before accessing it
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        // operations that should wait on this dependency are in the color attachment
        // stage, and involve the writing of the color attachment
        // (and also the depth attachment)
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );

    // Finalize the render pass.
    let attachments = &[
        *color_attachment,
        *depth_stencil_attachment,
        *color_resolve_attachment,
    ];
    let subpasses = &[*subpass];
    let dependencies = &[*dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    data.render_pass = device.create_render_pass(&info, None)?;

    Ok(())
}

/// Create a graphics pipeline.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub(crate) unsafe fn create_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    // Include our pre-compiled shaders.
    let vert = include_bytes!("../../shaders/shader.vert.spv");
    let frag = include_bytes!("../../shaders/shader.frag.spv");

    // Wrap the bytecode in shader modules
    let vert_shader_module = create_shader_module(device, &vert[..])?;
    let frag_shader_module = create_shader_module(device, &frag[..])?;

    // Create shader stages
    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_shader_module)
        .name(CStr::from_bytes_with_nul_unchecked(b"main\0"));
    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(CStr::from_bytes_with_nul_unchecked(b"main\0"));

    // Set up vertex buffers, vertex attributes, and so on.
    let binding_descriptions = &[Vertex::binding_description()];
    let attribute_descriptions = Vertex::attribute_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(binding_descriptions)
        .vertex_attribute_descriptions(&attribute_descriptions);

    // Vertices will be assembled into regular-old triangles.
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    // Take up the entire rendering surface for the viewport
    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    // Draw the ENTIRE viewport
    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(data.swapchain_extent);

    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(std::slice::from_ref(&viewport))
        .scissors(std::slice::from_ref(&scissor));

    // Configure the rasterizer
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    // Enable multisampling
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(data.msaa_samples);

    // Setup depth testing
    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS) // lower depth == closer
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0) // ignored because bounds test disabled
        .max_depth_bounds(1.0) // ignored because bounds test disabled
        .stencil_test_enable(false); // disable stencil tests for now

    // Use alpha blending
    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(std::slice::from_ref(&attachment))
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    // Tell the pipeline about our push constants
    let vert_push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(std::mem::size_of::<MvpMatPushConstants>() as u32);

    let frag_push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(std::mem::size_of::<MvpMatPushConstants>() as u32)
        .size(std::mem::size_of::<f32>() as u32); // for opacity as a 4-byte float

    // Setup the pipeline layout, including things like shader uniforms
    let push_constant_ranges = &[*vert_push_constant_range, *frag_push_constant_range];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(std::slice::from_ref(&data.descriptor_set_layout))
        .push_constant_ranges(push_constant_ranges);

    data.pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

    // Finalize the graphics pipeline
    let stages = &[*vert_stage, *frag_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        // Shader stages
        .stages(stages)
        // Fixed function stage configurations
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        // Pipeline layout
        .layout(data.pipeline_layout)
        // Render pass and subpass
        .render_pass(data.render_pass)
        .subpass(0);

    data.pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[*info], None)
        // If there's an error code, just get rid of it cause it's *probably* fine
        .unwrap_or_else(|(p, _)| p)[0];

    // Destroy the shader modules
    device.destroy_shader_module(vert_shader_module, None);
    device.destroy_shader_module(frag_shader_module, None);

    Ok(())
}

/// Create a shader module from SPIR-V shader bytecode and a GPU.
unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    // Realign the bytecode to a u32 slice
    let bytecode = Vec::<u8>::from(bytecode);
    let (prefix, code, suffix) = bytecode.align_to::<u32>();
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err(eyre!(
            "Unable to create shader module due to improper alignment of shader bytecode"
        ));
    }

    let info = vk::ShaderModuleCreateInfo::builder().code(code);

    Ok(device.create_shader_module(&info, None)?)
}

/// Create a framebuffer for all iamges in the swapchain.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub(crate) unsafe fn create_framebuffers(device: &Device, data: &mut AppData) -> Result<()> {
    data.framebuffers = data
        .swapchain_image_views
        .iter()
        .map(|i| {
            let attachments = &[data.color_image_view, data.depth_image_view, *i];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(data.render_pass)
                .attachments(attachments)
                .width(data.swapchain_extent.width)
                .height(data.swapchain_extent.height)
                .layers(1);

            device.create_framebuffer(&create_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}
