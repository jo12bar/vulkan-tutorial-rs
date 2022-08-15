//! Implements a model-view-projection matrix, for (e.g.) 3D cameras.

use nalgebra_glm as glm;

/// A model-view-projection matrix, to be used for implementing things like
/// 3D cameras.
///
/// This is intended to be sent to the GPU within a uniform buffer
/// object, which is why it's `#[repr(C)]`.
///
/// Despite the name, this is actually 3 seperate 4x4 matrices wrapped up inside
/// a struct.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MvpMat {
    pub model: glm::Mat4,
    pub view: glm::Mat4,
    pub projection: glm::Mat4,
}
