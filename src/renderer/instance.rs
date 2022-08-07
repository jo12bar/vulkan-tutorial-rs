//! Functions for creating Vulkan instances.

use super::validation::{should_enable_validation_layers, vk_debug_callback, VALIDATION_LAYER};
use crate::{app::AppData, util::VkExtensionName};
use ash::{extensions::ext as vk_ext, vk, Entry, Instance};
use color_eyre::{eyre::eyre, Result};
use std::{collections::HashSet, ffi::CStr};
use tracing::debug;
use winit::window::Window;

/// Create a Vulkan instance from an entry point.
///
/// The window handle is required so that we can load the required extensions for
/// drawing to a window.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub(crate) unsafe fn create_instance(
    window: &Window,
    entry: &Entry,
    data: &mut AppData,
) -> Result<Instance> {
    type DebugSeverity = vk::DebugUtilsMessageSeverityFlagsEXT;
    type DebugMsgType = vk::DebugUtilsMessageTypeFlagsEXT;

    let validation_enabled = should_enable_validation_layers();

    let app_info = vk::ApplicationInfo::builder()
        .application_name(CStr::from_bytes_with_nul(b"Rusty Vulkan Tutorial\0")?)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(CStr::from_bytes_with_nul(b"Johann's Rust Special\0")?)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::make_api_version(0, 1, 0, 0));

    // check the available validation layers so we can make sure our required
    // validation layer is supported
    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| VkExtensionName::from(l.layer_name))
        .collect::<HashSet<_>>();

    if validation_enabled && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(eyre!(
            "Validation layer requested, but not supported by this platform"
        ));
    }

    let layers = if validation_enabled {
        debug!(layer = ?VALIDATION_LAYER, "Enabling validation layer");
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    // Load Vulkan extensions

    let mut extensions = Vec::from(ash_window::enumerate_required_extensions(&window)?);

    if validation_enabled {
        debug!(extension = ?vk_ext::DebugUtils::name(), "Enabling extension");
        extensions.push(vk_ext::DebugUtils::name().as_ptr());
    }

    // Create the Vulkan instance

    let mut instance_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            DebugSeverity::VERBOSE
                | DebugSeverity::INFO
                | DebugSeverity::WARNING
                | DebugSeverity::ERROR,
        )
        .message_type(DebugMsgType::GENERAL | DebugMsgType::VALIDATION | DebugMsgType::PERFORMANCE)
        .pfn_user_callback(Some(vk_debug_callback));

    if validation_enabled {
        instance_info = instance_info.push_next(&mut debug_info);
    }

    let instance = entry.create_instance(&instance_info, None)?;

    // Connect the validation layer callback function if enabled
    if validation_enabled {
        data.messenger = vk_ext::DebugUtils::new(entry, &instance)
            .create_debug_utils_messenger(&debug_info, None)?;
    }

    Ok(instance)
}
