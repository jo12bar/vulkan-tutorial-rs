//! Functions for checking GPUs for support, and selecting from among them.
//!
//! Also includes some queue family-related stuff.

use super::{
    extensions::REQUIRED_DEVICE_EXTENSIONS,
    swapchain::SwapchainSupport,
    validation::{should_enable_validation_layers, VALIDATION_LAYER},
};
use crate::{
    app::AppData,
    util::{PhysicalDeviceName, VkExtensionName},
};
use ash::{extensions::khr as vk_khr, vk, Device, Entry, Instance};
use color_eyre::{eyre::eyre, Result};
use std::collections::HashSet;
use thiserror::Error;
use tracing::{debug, info};

/// For when a physical device does not satisfy some requirement of the application.
#[derive(Debug, Error)]
pub(crate) enum PhysicalDeviceSuitabilityError {
    #[error("Physical device is unsuitable: {0}")]
    Unsuitable(&'static str),
    #[error("Physical device is missing required extensions: {0}")]
    MissingExtensions(String),
    #[error("Error while querying physical device suitability: {0}")]
    VkError(#[from] vk::Result),
}

/// Picks a physical device to use for rendering.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub(crate) unsafe fn pick_physical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut AppData,
) -> Result<()> {
    let mut valid_devices = Vec::new();

    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);
        let device_name = PhysicalDeviceName::from(properties.device_name);

        match check_physical_device(entry, instance, data, physical_device) {
            Ok(score) => valid_devices.push((physical_device, score, device_name, properties)),
            Err(err) => {
                debug!(device_name = %device_name, reason = %err, "Skipping physical device")
            }
        }
    }

    if valid_devices.is_empty() {
        return Err(eyre!(
            "Failed to find suitable physical device for rendering."
        ));
    }

    // Choose the highest-scoring device
    valid_devices.sort_unstable_by(|(_, score_a, _, _), (_, score_b, _, _)| score_a.cmp(score_b));
    let (physical_device, _, device_name, properties) = valid_devices.last().unwrap();

    data.physical_device = *physical_device;
    info!(
        device_name = %device_name,
        device_id = properties.device_id,
        "Selected physical device for rendering"
    );

    Ok(())
}

/// Check if a physical device satisfies all the requirements of this application.
/// Returns a score based on its properties and available features.
unsafe fn check_physical_device(
    entry: &Entry,
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<u32, PhysicalDeviceSuitabilityError> {
    let mut score = 0;

    let properties = instance.get_physical_device_properties(physical_device);
    match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => score += 200,
        vk::PhysicalDeviceType::INTEGRATED_GPU => score += 100,
        _ => {
            return Err(PhysicalDeviceSuitabilityError::Unsuitable(
                "Only discrete and integrated GPUs are supported",
            ))
        }
    }

    let features = instance.get_physical_device_features(physical_device);
    if features.geometry_shader != vk::TRUE {
        return Err(PhysicalDeviceSuitabilityError::Unsuitable(
            "Missing geometry shader support",
        ));
    }

    // if the following function call doesn't panic, then the device supports
    // all the queue families needed for this app. we just discard the queue
    // family indices immediately though.
    QueueFamilyIndices::get(entry, instance, data, physical_device)?;

    // check if this device has all the extensions we care about. If this doesn't
    // panic, then we're all good.
    check_physical_device_extensions(instance, physical_device)?;

    // Check swapchain support. We only care if there's at least one supported
    // image format and one supported presentation mode, given our window surface.
    // Note that we can only query for swapchain support after checking if the
    // VK_KHR_swapchain extension is available - so this must be done after
    // checking for physical device extensions.
    let swapchain_support = SwapchainSupport::get(entry, instance, data, physical_device)?;
    if swapchain_support.formats.is_empty() || swapchain_support.present_modes.is_empty() {
        return Err(PhysicalDeviceSuitabilityError::Unsuitable(
            "Insufficient swapchain support.",
        ));
    }

    Ok(score)
}

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<(), PhysicalDeviceSuitabilityError> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device)?
        .into_iter()
        .map(|e| VkExtensionName::from(e.extension_name))
        .collect::<HashSet<_>>();

    let mut missing_extensions = Vec::new();

    for ext in REQUIRED_DEVICE_EXTENSIONS.iter() {
        if !extensions.contains(ext) {
            missing_extensions.push(*ext);
        }
    }

    if missing_extensions.is_empty() {
        Ok(())
    } else {
        Err(PhysicalDeviceSuitabilityError::MissingExtensions(
            missing_extensions
                .iter()
                .map(|e| e.to_string_lossy())
                .collect::<Vec<_>>()
                .join(", "),
        ))
    }
}

/// Create a logical device for rendering from a physical device.
#[tracing::instrument(level = "DEBUG", skip_all)]
pub(crate) unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut AppData,
) -> Result<Device> {
    let qf_indices = QueueFamilyIndices::get(entry, instance, data, data.physical_device)?;

    let mut unique_qf_indices = HashSet::new();
    unique_qf_indices.insert(qf_indices.graphics);
    unique_qf_indices.insert(qf_indices.present);

    // Setup command queues
    let queue_priorities = &[1.0];
    let queue_infos = unique_qf_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
                .build()
        })
        .collect::<Vec<_>>();

    // Setup validation layers (if needed)
    let layers = if should_enable_validation_layers() {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    // Set up device-specific features
    let features = vk::PhysicalDeviceFeatures::builder();

    // Convert our list of absolutely-required extensions to a seires of
    // null-terminated string pointers.
    let extension_names = REQUIRED_DEVICE_EXTENSIONS
        .iter()
        .map(|ext| ext.as_ptr())
        .collect::<Vec<_>>();

    // Fill in the device info and create the device
    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_features(&features)
        .enabled_extension_names(&extension_names);

    let device = instance.create_device(data.physical_device, &info, None)?;
    data.graphics_queue = device.get_device_queue(qf_indices.graphics, 0);
    data.present_queue = device.get_device_queue(qf_indices.present, 0);

    Ok(device)
}

/// Stores the indices of queue families to be used by this application.
#[derive(Copy, Clone, Debug)]
pub(crate) struct QueueFamilyIndices {
    pub(crate) graphics: u32,
    pub(crate) present: u32,
}

impl QueueFamilyIndices {
    pub(crate) unsafe fn get(
        entry: &Entry,
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, PhysicalDeviceSuitabilityError> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);

        let surface_ext = vk_khr::Surface::new(entry, instance);

        let mut graphics = None;
        let mut present = None;
        for (i, properties) in properties.iter().enumerate() {
            if properties.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                // Supports rendering graphics to an image
                graphics = Some(i as u32);
            }

            if surface_ext.get_physical_device_surface_support(
                physical_device,
                i as u32,
                data.surface,
            )? {
                // Supports rendering to a surface attached to some window
                present = Some(i as u32);
            }

            if graphics.is_some() && present.is_some() {
                break;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(PhysicalDeviceSuitabilityError::Unsuitable(
                "Missing required queue families",
            ))
        }
    }
}
