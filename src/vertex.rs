//! Vertices to be passed on to the GPU in vertex buffers and such.

use std::mem::size_of;

use ash::vk;
use lazy_static::lazy_static;
use nalgebra_glm as glm;

/// A vertex and an associated color to be sent to the GPU.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub pos: glm::Vec3,
    pub color: glm::Vec3,
    pub tex_coord: glm::Vec2,
}

impl Vertex {
    /// Create a new vertex with an associated color.
    ///
    /// This is marked as constant, but will only actually be usable from
    /// constant contexts once [`nalgebra_glm`] supports compile-time constructors.
    pub const fn new(pos: glm::Vec3, color: glm::Vec3, tex_coord: glm::Vec2) -> Self {
        Self {
            pos,
            color,
            tex_coord,
        }
    }

    /// Return a descriptor telling Vulkan the number of bytes between data
    /// entries and whether to move to the next data entry after each vertex
    /// or after each instance.
    pub const fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    /// Return Vulkan attribute descriptions specifying how to access each
    /// part of a vertex.
    pub const fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let pos = vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0,
        };

        let color = vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: size_of::<glm::Vec3>() as u32,
        };

        let tex_coord = vk::VertexInputAttributeDescription {
            location: 2,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: 2 * size_of::<glm::Vec3>() as u32,
        };

        [pos, color, tex_coord]
    }
}

lazy_static! {
    /// Vertices to be sent to the GPU in lieu of proper model loading.
    pub static ref VERTICES: Vec<Vertex> = vec![
        Vertex::new(glm::vec3(-0.5, -0.5, 0.0), glm::vec3(1.0, 0.0, 0.0), glm::vec2(1.0, 0.0)),
        Vertex::new(glm::vec3(0.5, -0.5, 0.0), glm::vec3(0.0, 1.0, 0.0), glm::vec2(0.0, 0.0)),
        Vertex::new(glm::vec3(0.5, 0.5, 0.0), glm::vec3(0.0, 0.0, 1.0), glm::vec2(0.0, 1.0)),
        Vertex::new(glm::vec3(-0.5, 0.5, 0.0), glm::vec3(1.0, 1.0, 1.0), glm::vec2(1.0, 1.0)),
    ];
}

/// The order of indices in [`VERTICES`] to be drawn.
pub const INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];
