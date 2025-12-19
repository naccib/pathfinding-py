use image_pathfinding::{
    AStar2D, AStarTemporal, Dijkstra2D, DijkstraTemporal, Fringe2D, ImagePathfinder2D,
};
use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Find a path in a 2D heatmap using the specified algorithm.
///
/// # Arguments
/// * `array` - A 2D NumPy array with dtype uint8 (shape: x, y) i.e. (width, height)
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
    // Arrays must be provided in (x, y) order, i.e. shape (width, height).
    // Use the array view directly - no transposing.
    let array_2d = array.as_array();

    let (width, height) = array_2d.dim();

    let width = width as u32;
    let height = height as u32;

    if start.0 >= width || start.1 >= height || end.0 >= width || end.1 >= height {
        return Err(PyValueError::new_err(format!(
            "Start or end position is out of bounds: start={:?}, end={:?}, shape={:?}",
            start,
            end,
            (width, height)
        )));
    }

    // Dispatch to appropriate algorithm
    let result = match algorithm.to_lowercase().as_str() {
        "astar" => AStar2D {}.find_path_in_heatmap(array_2d.view(), start, end),
        "dijkstra" => Dijkstra2D {}.find_path_in_heatmap(array_2d.view(), start, end),
        "fringe" => Fringe2D {}.find_path_in_heatmap(array_2d.view(), start, end),
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
/// * `array` - A 3D NumPy array with dtype uint8 (shape: x, y, t) i.e. (width, height, time)
/// * `algorithm` - Algorithm to use: "astar" or "dijkstra"
/// * `start` - Start position as (x, y, t) tuple
/// * `end` - End position as (x, y, t) tuple
/// * `reach` - Optional: Number of elements that can be skipped along each non-axis dimension (default: 1)
/// * `axis` - Optional: The axis along which the path must always move forward (default: 2 for time)
///
/// # Returns
/// * `Optional[Tuple[List[Tuple[int, int, int]], int]]` - The route found and total cost, or None if no route was found
#[pyfunction]
#[pyo3(signature = (array, algorithm, start, end, *, reach=None, axis=None))]
fn find_route_temporal(
    array: PyReadonlyArray3<u8>,
    algorithm: &str,
    start: (u32, u32, u32),
    end: (u32, u32, u32),
    reach: Option<usize>,
    axis: Option<usize>,
) -> PyResult<Option<(Vec<(u32, u32, u32)>, u32)>> {
    // PyReadonlyArray3<u8> enforces 3D array with u8 dtype at the Python binding level.
    // This provides runtime validation from Python's perspective.
    // Use the array view directly to avoid copying
    let array_3d = array.as_array();

    let (width, height, time) = array_3d.dim();
    let width = width as u32;
    let height = height as u32;
    let time = time as u32;

    if start.0 >= width
        || start.1 >= height
        || start.2 >= time
        || end.0 >= width
        || end.1 >= height
        || end.2 >= time
    {
        return Err(PyValueError::new_err(format!(
            "Start or end position is out of bounds: start={:?}, end={:?}, shape={:?}",
            start,
            end,
            (width, height, time)
        )));
    }

    // Convert single points to vectors for the underlying function
    let starts = Some(vec![start]);
    let ends = Some(vec![end]);

    // Dispatch to appropriate algorithm
    let result = match algorithm.to_lowercase().as_str() {
        "astar" => {
            AStarTemporal {}.find_route_over_time(array_3d.view(), reach, axis, starts, ends)
        }
        "dijkstra" => {
            DijkstraTemporal {}.find_route_over_time(array_3d.view(), reach, axis, starts, ends)
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
