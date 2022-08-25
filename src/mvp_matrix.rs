//! Implements a model-view-projection matrix, for (e.g.) 3D cameras.

use nalgebra_glm as glm;

/// A model-view-projection matrix, to be used for implementing things like
/// 3D cameras.
///
/// Despite the name, this is actually 3 seperate 4x4 matrices wrapped up inside
/// a struct.
#[derive(Clone, Copy, Debug)]
pub struct MvpMat {
    /// This should probably be sent as a push constant, _not_ in a uniform.
    pub model: glm::Mat4,
    pub view: glm::Mat4,
    pub projection: glm::Mat4,
}

impl MvpMat {
    pub fn new() -> Self {
        let mut this = Self {
            model: glm::identity(),
            view: glm::look_at(
                &glm::vec3(0.0, 0.0, 0.0),
                &glm::vec3(0.0, 0.0, 0.0),
                &glm::vec3(0.0, 0.0, 1.0),
            ),
            projection: glm::perspective(16.0 / 9.0, glm::radians(&glm::vec1(45.0))[0], 0.1, 10.0),
        };

        // Vulkan's Y axis is flipped compared to OpenGL, which GLM was originally
        // designed for. Compensate for this by flipping the y-axis's scaling factor
        // in the projection matrix.
        this.projection[(1, 1)] *= -1.0;

        this
    }

    /// Rotate the model by `a` radians about its z-axis.
    pub fn model_rotate_z(&mut self, a: f32) -> &mut Self {
        self.model = glm::rotate(&self.model, a, &glm::vec3(0.0, 0.0, 1.0));
        self
    }

    /// Position and orientate the camera.
    pub fn look_at(
        &mut self,
        eye: &glm::TVec3<f32>,
        center: &glm::TVec3<f32>,
        up: &glm::TVec3<f32>,
    ) -> &mut Self {
        self.view = glm::look_at(eye, center, up);
        self
    }

    /// Set the aspect ratio, vertical field-of-view, and far / near clip planes
    /// all at once.
    pub fn perspective(&mut self, aspect_ratio: f32, fovy: f32, near: f32, far: f32) -> &mut Self {
        self.projection = glm::perspective_rh_zo(aspect_ratio, fovy, near, far);

        // Vulkan's Y axis is flipped compared to OpenGL, which GLM was originally
        // designed for. Compensate for this by flipping the y-axis's scaling factor
        // in the projection matrix.
        self.projection[(1, 1)] *= -1.0;

        self
    }

    /// Copy view and projection into a struct ready for sending to the GPU
    /// as a uniform buffer object.
    pub const fn as_ubo(&self) -> MvpMatUBO {
        MvpMatUBO {
            view: self.view,
            projection: self.projection,
        }
    }

    /// Copy model matrix into a struct ready for sending to the GPU in a
    /// push constant.
    pub const fn as_push_constants(&self) -> MvpMatPushConstants {
        MvpMatPushConstants { model: self.model }
    }
}

impl Default for MvpMat {
    fn default() -> Self {
        Self::new()
    }
}

/// This is intended to be sent to the GPU within a uniform buffer
/// object, which is why it's `#[repr(C)]`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MvpMatUBO {
    pub view: glm::Mat4,
    pub projection: glm::Mat4,
}

/// This is intended to be sent to the GPU within a push constant,
/// which is why it's `#[repr(C)]`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MvpMatPushConstants {
    pub model: glm::Mat4,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mvp_mat_push_constants_has_correct_size() {
        assert_eq!(
            std::mem::size_of::<MvpMatPushConstants>(),
            64, // 16 × 4-byte floats
            "Size of overall struct is wrong"
        );

        let mvp_mat = MvpMat::default();
        let pcs = mvp_mat.as_push_constants();
        assert_eq!(
            std::mem::size_of_val(&pcs.model),
            64, // 16 × 4-byte floats
            "Size of model field is wrong"
        );
    }
}
