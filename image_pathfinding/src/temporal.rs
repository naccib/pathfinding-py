use numpy::ndarray::Array3;
use pathfinding::prelude::{astar, dijkstra};

/// A position in the temporal volume (x, y, t).
pub type Pos3D = (u32, u32, u32);

/// A position in the temporal volume with a cost.
pub type Pos3DWithCost = (Pos3D, u32);

// MARK: Helpers

/// Load a list of grayscale images into a temporal volume (Time, Height, Width).
/// Note: Internally ndarray uses (Time, Height, Width) indexing, so [t, y, x].
pub fn load_images_to_volume(paths: &[String]) -> Array3<u8> {
    if paths.is_empty() {
        return Array3::zeros((0, 0, 0));
    }

    // Load first image to get dimensions
    let first_img = image::open(&paths[0])
        .expect("Failed to open first image")
        .to_luma8();
    let (width, height) = first_img.dimensions();
    let depth = paths.len(); // Time dimension

    let mut volume = Array3::zeros((depth, height as usize, width as usize));

    for (t, path) in paths.iter().enumerate() {
        let img = image::open(path)
            .unwrap_or_else(|_| panic!("Failed to open image at {}", path))
            .to_luma8();

        if img.dimensions() != (width, height) {
            panic!("All images must have the same dimensions");
        }

        // Copy pixels
        for y in 0..height {
            for x in 0..width {
                volume[[t, y as usize, x as usize]] = img.get_pixel(x, y)[0];
            }
        }
    }

    volume
}

/// Find neighbours with reach constraint: always move +1 along axis, can move within reach in other dimensions.
/// For temporal routing: axis=2 (time) is default, reach limits movement in x and y dimensions.
fn find_neighbours_with_reach(
    volume: &Array3<u8>,
    pos: Pos3D,
    axis: usize,
    reach: usize,
) -> Vec<Pos3DWithCost> {
    let (x, y, t) = pos;
    let (depth, height, width) = volume.dim(); // (t, y, x)

    let mut neighbours = Vec::new();

    // For temporal routing, axis should be 0 (x), 1 (y), or 2 (t)
    // Default axis=2 means we always move forward in time
    let axis = if axis >= 3 { 2 } else { axis };

    // Generate offsets for non-axis dimensions
    // For temporal routing with axis=2 (time), we generate offsets for x and y
    // Offset shape: (2*reach+1)^(ndim-1) = (2*reach+1)^2 for 3D
    let mut offsets = Vec::new();

    match axis {
        0 => {
            // Moving along x axis (always +1), generate offsets for y and t
            // But we always move +1 in x, so offsets are (1, dy, dt) where dt=1
            if x as usize >= width - 1 {
                return neighbours;
            }
            for dy in -(reach as i32)..=(reach as i32) {
                if (t as usize) < depth - 1 {
                    offsets.push((1i32, dy, 1u32));
                }
            }
        }
        1 => {
            // Moving along y axis (always +1), generate offsets for x and t
            // But we always move +1 in y, so offsets are (dx, 1, dt) where dt=1
            if y as usize >= height - 1 {
                return neighbours;
            }
            for dx in -(reach as i32)..=(reach as i32) {
                if (t as usize) < depth - 1 {
                    offsets.push((dx, 1i32, 1u32));
                }
            }
        }
        2 => {
            // Moving along t axis (time, always +1), generate offsets for x and y
            if t as usize >= depth - 1 {
                return neighbours;
            }
            for dx in -(reach as i32)..=(reach as i32) {
                for dy in -(reach as i32)..=(reach as i32) {
                    offsets.push((dx, dy, 1u32));
                }
            }
        }
        _ => return neighbours,
    }

    // Apply offsets
    for (dx, dy, dt) in offsets {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        let nt = t + dt;

        // Check bounds
        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 && nt < depth as u32 {
            let nx_u = nx as u32;
            let ny_u = ny as u32;

            // Cost is the value at the *destination* node
            let cost = volume[[nt as usize, ny_u as usize, nx_u as usize]] as u32;
            neighbours.push(((nx_u, ny_u, nt), cost));
        }
    }

    neighbours
}

/// Generate all positions at a specific axis index.
/// For temporal routing: if axis=2 and index=0, returns all (x, y, 0) positions.
fn generate_positions_at_axis_index(volume: &Array3<u8>, axis: usize, index: usize) -> Vec<Pos3D> {
    let (depth, height, width) = volume.dim();
    let mut positions = Vec::new();

    match axis {
        0 => {
            // All positions with x = index
            if index < width {
                for y in 0..height {
                    for t in 0..depth {
                        positions.push((index as u32, y as u32, t as u32));
                    }
                }
            }
        }
        1 => {
            // All positions with y = index
            if index < height {
                for x in 0..width {
                    for t in 0..depth {
                        positions.push((x as u32, index as u32, t as u32));
                    }
                }
            }
        }
        2 => {
            // All positions with t = index
            if index < depth {
                for x in 0..width {
                    for y in 0..height {
                        positions.push((x as u32, y as u32, index as u32));
                    }
                }
            }
        }
        _ => {}
    }

    positions
}

/// Generate default start positions (all positions at axis=0) or end positions (all positions at axis=-1).
fn generate_default_starts_ends(volume: &Array3<u8>, axis: usize, is_start: bool) -> Vec<Pos3D> {
    let (depth, height, width) = volume.dim();
    let axis = if axis >= 3 { 2 } else { axis };

    if is_start {
        generate_positions_at_axis_index(volume, axis, 0)
    } else {
        // For end positions, use the last index along the axis
        let end_index = match axis {
            0 => width - 1,
            1 => height - 1,
            2 => depth - 1,
            _ => 0,
        };
        generate_positions_at_axis_index(volume, axis, end_index)
    }
}

// MARK: Temporal Routers

// MARK: Dijkstra

pub struct DijkstraTemporal {}

impl DijkstraTemporal {
    /// Find the shortest route through a temporal volume from one side to another.
    ///
    /// # Arguments
    ///
    /// * `volume` - The temporal volume (Time, Height, Width)
    /// * `reach` - Number of elements that can be skipped along each non-axis dimension (default: 1)
    /// * `axis` - The axis along which the path must always move forward (default: 2 for time)
    /// * `starts` - Optional start positions. If None, uses all positions at axis=0
    /// * `ends` - Optional end positions. If None, uses all positions at axis=-1
    ///
    /// # Returns
    ///
    /// * `Option<(Vec<Pos3D>, u32)>` - The route found and the total cost, or None if no route was found
    pub fn find_route_over_time(
        &self,
        volume: &Array3<u8>,
        reach: Option<usize>,
        axis: Option<usize>,
        starts: Option<Vec<Pos3D>>,
        ends: Option<Vec<Pos3D>>,
    ) -> Option<(Vec<Pos3D>, u32)> {
        let reach = reach.unwrap_or(1);
        let axis = axis.unwrap_or(2); // Default to time axis

        let starts = starts.unwrap_or_else(|| generate_default_starts_ends(volume, axis, true));
        let ends = ends.unwrap_or_else(|| generate_default_starts_ends(volume, axis, false));

        if starts.is_empty() || ends.is_empty() {
            return None;
        }

        // Collect all end positions into a set for fast lookup
        let ends_set: std::collections::HashSet<Pos3D> = ends.iter().cloned().collect();

        // Run Dijkstra from each start position and find the minimum cost path to any end
        let mut best_path: Option<(Vec<Pos3D>, u32)> = None;
        let mut best_cost = u32::MAX;

        for &start in &starts {
            let result = dijkstra(
                &start,
                |&p| find_neighbours_with_reach(volume, p, axis, reach),
                |&p| ends_set.contains(&p),
            );

            if let Some((path, cost)) = result {
                if cost < best_cost {
                    best_cost = cost;
                    best_path = Some((path, cost));
                }
            }
        }

        best_path
    }
}

// MARK: A*

pub struct AStarTemporal {}

impl AStarTemporal {
    /// Minimum distance to any end position (for multi-end heuristic)
    fn min_distance_to_ends(&self, pos: Pos3D, ends: &[Pos3D], axis: usize) -> u32 {
        let (x, y, t) = pos;
        let mut min_dist = u32::MAX;

        for &(ex, ey, et) in ends {
            match axis {
                0 => {
                    // Moving along x axis
                    let spatial_dist = (y.abs_diff(ey) + t.abs_diff(et)) as u32;
                    if x <= ex {
                        min_dist = min_dist.min(spatial_dist);
                    }
                }
                1 => {
                    // Moving along y axis
                    let spatial_dist = (x.abs_diff(ex) + t.abs_diff(et)) as u32;
                    if y <= ey {
                        min_dist = min_dist.min(spatial_dist);
                    }
                }
                2 => {
                    // Moving along t axis (time)
                    let spatial_dist = (x.abs_diff(ex) + y.abs_diff(ey)) as u32;
                    if t <= et {
                        min_dist = min_dist.min(spatial_dist);
                    }
                }
                _ => {}
            }
        }

        min_dist
    }

    /// Find the shortest route through a temporal volume from one side to another.
    ///
    /// # Arguments
    ///
    /// * `volume` - The temporal volume (Time, Height, Width)
    /// * `reach` - Number of elements that can be skipped along each non-axis dimension (default: 1)
    /// * `axis` - The axis along which the path must always move forward (default: 2 for time)
    /// * `starts` - Optional start positions. If None, uses all positions at axis=0
    /// * `ends` - Optional end positions. If None, uses all positions at axis=-1
    ///
    /// # Returns
    ///
    /// * `Option<(Vec<Pos3D>, u32)>` - The route found and the total cost, or None if no route was found
    pub fn find_route_over_time(
        &self,
        volume: &Array3<u8>,
        reach: Option<usize>,
        axis: Option<usize>,
        starts: Option<Vec<Pos3D>>,
        ends: Option<Vec<Pos3D>>,
    ) -> Option<(Vec<Pos3D>, u32)> {
        let reach = reach.unwrap_or(1);
        let axis = axis.unwrap_or(2); // Default to time axis

        let starts = starts.unwrap_or_else(|| generate_default_starts_ends(volume, axis, true));
        let ends = ends.unwrap_or_else(|| generate_default_starts_ends(volume, axis, false));

        if starts.is_empty() || ends.is_empty() {
            return None;
        }

        // Collect all end positions into a set for fast lookup
        let ends_set: std::collections::HashSet<Pos3D> = ends.iter().cloned().collect();
        let ends_vec = ends;

        // Run A* from each start position and find the minimum cost path to any end
        let mut best_path: Option<(Vec<Pos3D>, u32)> = None;
        let mut best_cost = u32::MAX;

        for &start in &starts {
            let ends_vec_clone = ends_vec.clone();
            let result = astar(
                &start,
                |&p| find_neighbours_with_reach(volume, p, axis, reach),
                |&p| self.min_distance_to_ends(p, &ends_vec_clone, axis),
                |&p| ends_set.contains(&p),
            );

            if let Some((path, cost)) = result {
                if cost < best_cost {
                    best_cost = cost;
                    best_path = Some((path, cost));
                }
            }
        }

        best_path
    }
}
