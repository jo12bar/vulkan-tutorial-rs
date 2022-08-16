//! Tools for loading models.

use std::fmt::Debug;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use ahash::AHashMap;
use color_eyre::Result;
use nalgebra_glm as glm;
use tracing::debug;

use crate::app::AppData;
use crate::vertex::Vertex;

/// Load a model into the global AppData struct.
#[tracing::instrument(level = "DEBUG", skip_all, fields(path = ?path))]
pub fn load_model<P>(data: &mut AppData, path: P) -> Result<()>
where
    P: AsRef<Path> + Debug,
{
    let mut reader = BufReader::new(File::open(path)?);

    let (models, _) = tobj::load_obj_buf(
        &mut reader,
        &tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ..Default::default()
        },
        |_| Ok((vec![tobj::Material::default()], AHashMap::new())),
    )?;

    let mut unique_vertices = AHashMap::new();

    for model in &models {
        for index in &model.mesh.indices {
            let pos_offset = (3 * index) as usize;
            let tex_coord_offset = (2 * index) as usize;

            let vertex = Vertex {
                pos: glm::vec3(
                    model.mesh.positions[pos_offset],
                    model.mesh.positions[pos_offset + 1],
                    model.mesh.positions[pos_offset + 2],
                ),
                color: glm::vec3(1.0, 1.0, 1.0),
                tex_coord: glm::vec2(
                    model.mesh.texcoords[tex_coord_offset],
                    1.0 - model.mesh.texcoords[tex_coord_offset + 1],
                ),
            };

            if let Some(index) = unique_vertices.get(&vertex) {
                data.indices.push(*index as u32);
            } else {
                let index = data.vertices.len();
                unique_vertices.insert(vertex, index);
                data.vertices.push(vertex);
                data.indices.push(index as u32);
            }
        }
    }

    debug!(
        vertex_count = data.vertices.len(),
        index_count = data.indices.len(),
        "Successfully loaded model"
    );

    Ok(())
}
