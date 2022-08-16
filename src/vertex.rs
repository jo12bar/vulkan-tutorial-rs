//! Vertices to be passed on to the GPU in vertex buffers and such.

use std::hash::{Hash, Hasher};
use std::mem::size_of;

use ash::vk;
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

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.color == other.color && self.tex_coord == other.tex_coord
    }
}

/// Note: This implementation of Eq is only valid if the vertices being compared
/// do not contain NaN in any of their data. For now, this is a safe assumption.
impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
    }
}
