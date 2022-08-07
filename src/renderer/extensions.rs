//! Required and optional Vulkan extensions.

use crate::util::VkExtensionName;
use ash::extensions::khr as vk_khr;
use lazy_static::lazy_static;

lazy_static! {
    /// Device extensions that are absolutely _required_. Don't put optional
    /// extensions into this list!
    pub(crate) static ref REQUIRED_DEVICE_EXTENSIONS: Vec<VkExtensionName> = [
        vk_khr::Swapchain::name()
    ]
        .into_iter()
        .map(VkExtensionName::from_cstr)
        .collect();
}

/// [`ash`] dynamically links to extensions, on the fly. This can be detrimental
/// to performance if done repeatedly (e.g. in a render loop). This struct can
/// be used to "cache" the links to the extensions.
#[derive(Clone)]
pub(crate) struct Extensions {
    pub(crate) swapchain: vk_khr::Swapchain,
}
