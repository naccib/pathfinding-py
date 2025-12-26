use numpy::ndarray::{Array2, ArrayView2};
use pathfinding::prelude::{astar, dijkstra};

/// A position in the image.
pub type Pos2D = (u32, u32);

/// A position in the image with a cost.
pub type Pos2DWithCost = (Pos2D, u32);

// MARK: Helpers

/// Load a PNG image and convert it to a 2D ndarray (grayscale).
/// Returns an Array2<u8> with shape (width, height).
pub fn load_png_to_ndarray(path: &str) -> Array2<u8> {
    let img = image::open(path)
        .expect(&format!("Failed to open image at {}", path))
        .to_luma8();

    let (width, height) = img.dimensions();
    let mut array = Array2::zeros((width as usize, height as usize));

    for y in 0..height {
        for x in 0..width {
            array[[x as usize, y as usize]] = img.get_pixel(x, y)[0];
        }
    }

    array
}

/// Find the possible neighbours and their costs for a given pixel in a 2D ndarray.
/// Returns a vector of tuples, where each tuple contains a position and a cost.
///
/// # Arguments
/// * `array` - The 2D ndarray to find neighbours in.
/// * `pos` - The position to find neighbours for.
/// * `impassable` - An optional value that, if provided, will be used to filter out neighbours that have this value.
///
/// # Returns
/// * `Vec<Pos2DWithCost>` - A vector of tuples, where each tuple contains a position and a cost.
fn find_neighbours_with_cost(
    array: ArrayView2<u8>,
    pos: Pos2D,
    impassable: Option<u8>,
) -> Vec<Pos2DWithCost> {
    let mut neighbours = Vec::new();

    let (x, y) = pos;
    let (width, height) = array.dim();
    let height = height as u32;
    let width = width as u32;

    // Cardinal neighbors (up, down, left, right)
    if x > 0 {
        neighbours.push(((x - 1, y), array[[(x - 1) as usize, y as usize]] as u32));
    }
    if x < width - 1 {
        neighbours.push(((x + 1, y), array[[(x + 1) as usize, y as usize]] as u32));
    }
    if y > 0 {
        neighbours.push(((x, y - 1), array[[x as usize, (y - 1) as usize]] as u32));
    }
    if y < height - 1 {
        neighbours.push(((x, y + 1), array[[x as usize, (y + 1) as usize]] as u32));
    }

    // Diagonal neighbors
    if x > 0 && y > 0 {
        neighbours.push((
            (x - 1, y - 1),
            array[[(x - 1) as usize, (y - 1) as usize]] as u32,
        ));
    }
    if x < width - 1 && y > 0 {
        neighbours.push((
            (x + 1, y - 1),
            array[[(x + 1) as usize, (y - 1) as usize]] as u32,
        ));
    }
    if x > 0 && y < height - 1 {
        neighbours.push((
            (x - 1, y + 1),
            array[[(x - 1) as usize, (y + 1) as usize]] as u32,
        ));
    }
    if x < width - 1 && y < height - 1 {
        neighbours.push((
            (x + 1, y + 1),
            array[[(x + 1) as usize, (y + 1) as usize]] as u32,
        ));
    }

    if let Some(impassable) = impassable {
        neighbours.retain(|(_, cost)| *cost != impassable as u32);
    }

    neighbours
}

// MARK: Pathfinders

pub trait ImagePathfinder2D {
    /// Find a path in a heatmap in 2D space. The heatmap must be represented by a 2D ndarray.
    ///
    /// # Arguments
    ///
    /// * `array` - The heatmap as a 2D ndarray with shape (width, height).
    /// * `start_pos` - The start position (x, y).
    /// * `end_pos` - The end position (x, y).
    ///
    /// # Returns
    ///
    /// * `Option<(Vec<Pos2D>, u32)>` - The path found and the total cost, or `None` if no path was found.
    fn find_path_in_heatmap(
        &self,
        array: ArrayView2<u8>,
        start_pos: Pos2D,
        end_pos: Pos2D,
        impassable: Option<u8>,
    ) -> Option<(Vec<Pos2D>, u32)>;
}

// MARK: Dijkstra

/// A 2D pathfinder that uses Dijkstra's algorithm.
pub struct Dijkstra2D {}

impl ImagePathfinder2D for Dijkstra2D {
    fn find_path_in_heatmap(
        &self,
        array: ArrayView2<u8>,
        start_pos: Pos2D,
        end_pos: Pos2D,
        impassable: Option<u8>,
    ) -> Option<(Vec<Pos2D>, u32)> {
        let result = dijkstra(
            &start_pos,
            |&p| find_neighbours_with_cost(array, p, impassable),
            |&p| p == end_pos,
        );

        if let Some((path, costs)) = result {
            return Some((path, costs));
        }

        None
    }
}

// MARK: A*

pub struct AStar2D {}

impl AStar2D {
    fn manhattan_distance(&self, pos: Pos2D, end_pos: Pos2D) -> u32 {
        let (x1, y1) = pos;
        let (x2, y2) = end_pos;
        (x1.abs_diff(x2) + y1.abs_diff(y2)) as u32
    }
}

impl ImagePathfinder2D for AStar2D {
    fn find_path_in_heatmap(
        &self,
        array: ArrayView2<u8>,
        start_pos: Pos2D,
        end_pos: Pos2D,
        impassable: Option<u8>,
    ) -> Option<(Vec<Pos2D>, u32)> {
        let result = astar(
            &start_pos,
            |&p| find_neighbours_with_cost(array, p, impassable),
            // the minumum cost is the manhattan distance
            |&p| self.manhattan_distance(p, end_pos),
            |&p| p == end_pos,
        );

        if let Some((path, costs)) = result {
            return Some((path, costs));
        }

        None
    }
}

// MARK: Fringe

pub struct Fringe2D {}

impl Fringe2D {
    fn manhattan_distance(&self, pos: Pos2D, end_pos: Pos2D) -> u32 {
        let (x1, y1) = pos;
        let (x2, y2) = end_pos;
        (x1.abs_diff(x2) + y1.abs_diff(y2)) as u32
    }
}

impl ImagePathfinder2D for Fringe2D {
    fn find_path_in_heatmap(
        &self,
        array: ArrayView2<u8>,
        start_pos: Pos2D,
        end_pos: Pos2D,
        impassable: Option<u8>,
    ) -> Option<(Vec<Pos2D>, u32)> {
        let result = pathfinding::prelude::fringe(
            &start_pos,
            |&p| find_neighbours_with_cost(array, p, impassable),
            |&p| self.manhattan_distance(p, end_pos),
            |&p| p == end_pos,
        );

        if let Some((path, costs)) = result {
            return Some((path, costs));
        }

        None
    }
}
