use numpy::ndarray::{Array2, ArrayView2};
use pathfinding::prelude::astar;

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
            array[[x as usize, y as usize]] = img.get_pixel(x, y)[0].max(1);
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
        let (width, height) = array.dim();
        if width == 0 || height == 0 {
            return None;
        }

        let (sx, sy) = (start_pos.0 as usize, start_pos.1 as usize);
        let (ex, ey) = (end_pos.0 as usize, end_pos.1 as usize);
        if sx >= width || sy >= height || ex >= width || ey >= height {
            return None;
        }

        // Work on a row-major contiguous view, so element `(x, y)` lives at the flat
        // index `x * height + y`. `as_standard_layout` is a zero-copy borrow for the
        // standard-layout arrays `load_png_to_ndarray` produces, and only copies for an
        // exotic input layout (e.g. a transposed NumPy view).
        let standard = array.as_standard_layout();
        let grid = standard
            .as_slice()
            .expect("standard layout is always contiguous");

        let n = width * height;
        let start_idx = sx * height + sy;
        let end_idx = ex * height + ey;

        // Flat arrays keyed by node index avoid the per-expansion `Vec` allocation and
        // node hashing that a generic hashmap-based solver would incur.
        let mut dist = vec![u32::MAX; n];
        let mut came_from = vec![u32::MAX; n];

        // Dial's algorithm. Edge costs are pixel values bounded to `1..=255`, so every
        // node still on the frontier has a distance within 255 of the one being settled.
        // A circular array of `NUM_BUCKETS` buckets (indexed by `distance % NUM_BUCKETS`)
        // therefore holds the entire frontier at once, with each distinct distance landing
        // in its own bucket. That gives O(1) push and pop in place of the binary heap's
        // O(log n) — and the heap dominated the profile.
        const NUM_BUCKETS: usize = 256; // u8 weights => max edge cost 255 => window of 256
        let mut buckets: [Vec<u32>; NUM_BUCKETS] = std::array::from_fn(|_| Vec::new());
        let mut queued = 0usize; // entries across all buckets (including stale ones)

        let impassable = impassable.map(|v| v as u32);

        dist[start_idx] = 0;
        buckets[0].push(start_idx as u32);
        queued += 1;

        let mut cur = 0u32; // absolute distance level currently being scanned
        'search: while queued > 0 {
            // Advance to the next non-empty bucket. While `queued > 0` the window
            // invariant guarantees one exists within `NUM_BUCKETS` steps, and `cur` never
            // passes a non-empty bucket, so every queued distance is always `>= cur`.
            while buckets[cur as usize % NUM_BUCKETS].is_empty() {
                cur += 1;
            }

            let b = cur as usize % NUM_BUCKETS;
            // Every entry in this bucket has distance exactly `cur` (the window spans only
            // `NUM_BUCKETS` distinct distances, so no other level aliases to it).
            while let Some(u) = buckets[b].pop() {
                queued -= 1;
                let u = u as usize;
                // Stale entry: the node was re-inserted at a smaller distance and already
                // settled there, so this later copy is obsolete.
                if dist[u] != cur {
                    continue;
                }
                if u == end_idx {
                    break 'search;
                }

                let x = u / height;
                let y = u % height;
                let has_left = x > 0;
                let has_right = x + 1 < width;
                let has_up = y > 0;
                let has_down = y + 1 < height;

                // Relax the edge into a neighbour pixel; the edge cost is the destination
                // pixel's value, matching `find_neighbours_with_cost`.
                macro_rules! relax {
                    ($nidx:expr) => {{
                        let nidx = $nidx;
                        let cost = grid[nidx] as u32;
                        let passable = match impassable {
                            Some(im) => cost != im,
                            None => true,
                        };
                        if passable {
                            let nd = cur + cost;
                            if nd < dist[nidx] {
                                dist[nidx] = nd;
                                came_from[nidx] = u as u32;
                                buckets[nd as usize % NUM_BUCKETS].push(nidx as u32);
                                queued += 1;
                            }
                        }
                    }};
                }

                // Cardinal neighbours.
                if has_left {
                    relax!(u - height);
                }
                if has_right {
                    relax!(u + height);
                }
                if has_up {
                    relax!(u - 1);
                }
                if has_down {
                    relax!(u + 1);
                }
                // Diagonal neighbours.
                if has_left && has_up {
                    relax!(u - height - 1);
                }
                if has_right && has_up {
                    relax!(u + height - 1);
                }
                if has_left && has_down {
                    relax!(u - height + 1);
                }
                if has_right && has_down {
                    relax!(u + height + 1);
                }
            }

            // Bucket drained for this level; the next iteration scans `cur + 1` onward.
            cur += 1;
        }

        if dist[end_idx] == u32::MAX {
            return None;
        }

        // Reconstruct the path by walking the predecessor links from the goal.
        let mut path = Vec::new();
        let mut cur = end_idx;
        loop {
            path.push(((cur / height) as u32, (cur % height) as u32));
            if cur == start_idx {
                break;
            }
            cur = came_from[cur] as usize;
        }
        path.reverse();

        Some((path, dist[end_idx]))
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
