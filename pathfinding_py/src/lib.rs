use image_pathfinding::{
    AStar2D, AStarTemporal, Dijkstra2D, DijkstraTemporal, Fringe2D, ImagePathfinder2D,
};
use numpy::ndarray::{Array2, Array3};
use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Find a path in a 2D heatmap using the specified algorithm.
///
/// # Arguments
/// * `array` - A 2D NumPy array with dtype uint8 (shape: height, width)
/// * `start` - Start position as (x, y) tuple
/// * `end` - End position as (x, y) tuple
/// * `algorithm` - Algorithm to use: "astar", "dijkstra", or "fringe"
///
/// # Returns
/// * `Optional[Tuple[List[Tuple[int, int]], int]]` - The path found and total cost, or None if no path was found
#[pyfunction]
fn find_path_2d(
    array: PyReadonlyArray2<u8>,
    start: (u32, u32),
    end: (u32, u32),
    algorithm: &str,
) -> PyResult<Option<(Vec<(u32, u32)>, u32)>> {
    // PyReadonlyArray2<u8> enforces 2D array with u8 dtype at the Python binding level.
    // This provides runtime validation from Python's perspective.
    // Convert to Array2<u8>
    let array_2d: Array2<u8> = array.as_array().to_owned();

    // Dispatch to appropriate algorithm
    let result = match algorithm.to_lowercase().as_str() {
        "astar" => AStar2D {}.find_path_in_heatmap(&array_2d, start, end),
        "dijkstra" => Dijkstra2D {}.find_path_in_heatmap(&array_2d, start, end),
        "fringe" => Fringe2D {}.find_path_in_heatmap(&array_2d, start, end),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown algorithm: {}. Supported algorithms: astar, dijkstra, fringe",
                algorithm
            )));
        }
    };

    Ok(result)
}

/// Find a route through a temporal volume using the specified algorithm.
///
/// # Arguments
/// * `array` - A 3D NumPy array with dtype uint8 (shape: time, height, width)
/// * `algorithm` - Algorithm to use: "astar" or "dijkstra"
/// * `reach` - Optional: Number of elements that can be skipped along each non-axis dimension (default: 1)
/// * `axis` - Optional: The axis along which the path must always move forward (default: 2 for time)
/// * `starts` - Optional: Start positions as list of (x, y, t) tuples. If None, uses all positions at axis=0
/// * `ends` - Optional: End positions as list of (x, y, t) tuples. If None, uses all positions at axis=-1
///
/// # Returns
/// * `Optional[Tuple[List[Tuple[int, int, int]], int]]` - The route found and total cost, or None if no route was found
#[pyfunction]
#[pyo3(signature = (array, algorithm, *, reach=None, axis=None, starts=None, ends=None))]
fn find_route_temporal(
    array: PyReadonlyArray3<u8>,
    algorithm: &str,
    reach: Option<usize>,
    axis: Option<usize>,
    starts: Option<Vec<(u32, u32, u32)>>,
    ends: Option<Vec<(u32, u32, u32)>>,
) -> PyResult<Option<(Vec<(u32, u32, u32)>, u32)>> {
    // PyReadonlyArray3<u8> enforces 3D array with u8 dtype at the Python binding level.
    // This provides runtime validation from Python's perspective.
    // Convert to Array3<u8>
    let array_3d: Array3<u8> = array.as_array().to_owned();

    // Dispatch to appropriate algorithm
    let result = match algorithm.to_lowercase().as_str() {
        "astar" => AStarTemporal {}.find_route_over_time(&array_3d, reach, axis, starts, ends),
        "dijkstra" => {
            DijkstraTemporal {}.find_route_over_time(&array_3d, reach, axis, starts, ends)
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown algorithm: {}. Supported algorithms: astar, dijkstra",
                algorithm
            )));
        }
    };

    Ok(result)
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn pathfinding_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_path_2d, m)?)?;
    m.add_function(wrap_pyfunction!(find_route_temporal, m)?)?;
    Ok(())
}
