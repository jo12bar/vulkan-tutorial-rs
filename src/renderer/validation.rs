//! Hooks connecting Vulkan's validation API to [`tracing`]

use crate::util::VkExtensionName;
use ash::vk;
use std::{ffi::CStr, os::raw::c_void};
use tracing::{debug, error, trace, warn};

/// Returns true if Vulkan validation layers should be enabled.
///
/// Will always return true in builds where `debug_assertions` is enabled.
/// Otherwise, will only return true if the environment variable
/// `ENABLE_VULKAN_VALIDATION_LAYERS` is set.
#[inline]
pub(crate) fn should_enable_validation_layers() -> bool {
    cfg!(debug_assertions) || std::env::var("ENABLE_VULKAN_VALIDATION_LAYERS").is_ok()
}

/// The default Vulkan validation layer bundle to be used if [`should_enable_validation_layers()`]
/// returns true.
pub(crate) const VALIDATION_LAYER: VkExtensionName =
    VkExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation\0");

/// A callback function that will be called whenever Vulkan has a validation layer message to output.
pub(crate) extern "system" fn vk_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = if data.is_null() {
        return vk::FALSE;
    } else {
        unsafe { *data }
    };
    let message_id = unsafe {
        if data.p_message_id_name.is_null() {
            CStr::from_bytes_with_nul_unchecked(b"<undefined id>\0")
        } else {
            CStr::from_ptr(data.p_message)
        }
    }
    .to_string_lossy();
    let message = unsafe {
        if data.p_message.is_null() {
            CStr::from_bytes_with_nul_unchecked(b"<undefined message>\0")
        } else {
            CStr::from_ptr(data.p_message)
        }
    }
    .to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!({"type" = ?typ, id = %message_id}, "{}", message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!({"type" = ?typ, id = %message_id}, "{}", message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!({"type" = ?typ, id = %message_id}, "{}", message);
    } else {
        trace!({"type" = ?typ, id = %message_id}, "{}", message);
    }

    vk::FALSE
}
