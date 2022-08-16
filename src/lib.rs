pub mod app;
pub(crate) mod model;
pub(crate) mod mvp_matrix;
pub(crate) mod renderer;
pub mod util;
pub(crate) mod vertex;

/// The maximum number of frames that the app is allowed to submit to the GPU
/// for rendering before we have to wait for the GPU to finish rendering a
/// frame.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
