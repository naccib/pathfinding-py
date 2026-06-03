use numpy::ndarray::{Array2, ArrayView2};

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

        // A diagonal step costs ~1.5x a cardinal one. The cardinal cost is the destination
        // pixel value unchanged; the diagonal cost is `pixel + pixel / 2 = floor(1.5 * pixel)`,
        // computed with a shift. We deliberately do *not* scale every cost by 2 to make 1.5
        // exact (3/2): that would double every distance and, since Dial's has an O(maxDist)
        // bucket-scan term, measurably slow the search. So 1.5x is exact for even pixel values
        // and rounds down for odd ones.

        // Dial's algorithm. The max edge cost is the largest diagonal cost, `255 + 255 / 2 =
        // 382`, so every node still on the frontier has a distance within that of the one being
        // settled. A circular array of `NUM_BUCKETS` buckets (indexed by `distance % NUM_BUCKETS`)
        // therefore holds the entire frontier at once, each distinct distance landing in its own
        // bucket — O(1) push and pop in place of the binary heap's O(log n). `NUM_BUCKETS` is a
        // power of two larger than the max edge cost, so `% NUM_BUCKETS` compiles to a bitmask.
        const NUM_BUCKETS: usize = 512; // power of two > 382 (max diagonal edge cost)
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

                // Relax the edge into a neighbour pixel. A cardinal move costs the destination
                // pixel value; a diagonal move costs `pixel + pixel / 2` (~1.5x). `$diag` is a
                // bool literal, so the branch is resolved at compile time. The impassable test
                // uses the raw pixel value and so is independent of the move direction.
                macro_rules! relax {
                    ($nidx:expr, $diag:expr) => {{
                        let nidx = $nidx;
                        let pixel = grid[nidx] as u32;
                        let passable = match impassable {
                            Some(im) => pixel != im,
                            None => true,
                        };
                        if passable {
                            let cost = if $diag { pixel + (pixel >> 1) } else { pixel };
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
                    relax!(u - height, false);
                }
                if has_right {
                    relax!(u + height, false);
                }
                if has_up {
                    relax!(u - 1, false);
                }
                if has_down {
                    relax!(u + 1, false);
                }
                // Diagonal neighbours cost ~1.5x as much.
                if has_left && has_up {
                    relax!(u - height - 1, true);
                }
                if has_right && has_up {
                    relax!(u + height - 1, true);
                }
                if has_left && has_down {
                    relax!(u - height + 1, true);
                }
                if has_right && has_down {
                    relax!(u + height + 1, true);
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

/// A 2D pathfinder that uses A* with a Chebyshev heuristic.
pub struct AStar2D {}

impl ImagePathfinder2D for AStar2D {
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

        // Same contiguous, row-major view as Dijkstra2D: element `(x, y)` is at `x * height + y`.
        let standard = array.as_standard_layout();
        let grid = standard
            .as_slice()
            .expect("standard layout is always contiguous");

        let n = width * height;
        let start_idx = sx * height + sy;
        let end_idx = ex * height + ey;

        // `g[i]` is the best known cost from the start to node `i`; the frontier is ordered by
        // `f = g + h`. Flat arrays replace the generic solver's hash maps and per-expansion
        // `Vec`s, exactly as in Dijkstra2D.
        let mut g = vec![u32::MAX; n];
        let mut came_from = vec![u32::MAX; n];

        // Dial's bucket queue keyed on `f`. The largest edge cost is a diagonal `255 + 255 / 2 =
        // 382`, and the heuristic changes by at most 1 per move, so successive `f` values differ
        // by at most 383; a power-of-two window above that holds the whole frontier and makes
        // `% NUM_BUCKETS` a bitmask.
        const NUM_BUCKETS: usize = 512; // power of two > 382 + 1
        let mut buckets: [Vec<u32>; NUM_BUCKETS] = std::array::from_fn(|_| Vec::new());
        let mut queued = 0usize;

        let impassable = impassable.map(|v| v as u32);

        // Chebyshev distance to the goal. Each 8-connected move changes `max(|dx|, |dy|)` by at
        // most 1 and costs at least 1 (the minimum pixel value is 1, and a diagonal at pixel 1
        // also costs 1), so this never overestimates the remaining cost: it is admissible and
        // consistent. Consistency keeps `f` monotone, which is what lets the bucket queue work.
        // (The previous Manhattan heuristic overestimated diagonal travel and was not admissible,
        // so that A* was not guaranteed to return an optimal path.)
        macro_rules! heuristic {
            ($x:expr, $y:expr) => {
                ($x.max($y) + $x.min($y) >> 1) as u32
            };
        }

        g[start_idx] = 0;
        let mut cur = heuristic!(sx, sy); // f of the start node; `f` only grows from here
        buckets[cur as usize % NUM_BUCKETS].push(start_idx as u32);
        queued += 1;

        'search: while queued > 0 {
            while buckets[cur as usize % NUM_BUCKETS].is_empty() {
                cur += 1;
            }

            let b = cur as usize % NUM_BUCKETS;
            while let Some(u) = buckets[b].pop() {
                queued -= 1;
                let u = u as usize;
                let x = u / height;
                let y = u % height;
                // Stale or already-settled entry: its `f` no longer matches this level. With a
                // consistent heuristic a node's `g` is final the first time it is popped, so any
                // later (higher-`f`) duplicate is obsolete.
                if g[u] + heuristic!(x, y) != cur {
                    continue;
                }
                if u == end_idx {
                    break 'search;
                }

                let gu = g[u];
                let has_left = x > 0;
                let has_right = x + 1 < width;
                let has_up = y > 0;
                let has_down = y + 1 < height;

                // Relax the edge into a neighbour. A cardinal move costs the destination pixel
                // value; a diagonal move costs `pixel + pixel / 2` (~1.5x), matching Dijkstra2D.
                // The neighbour's coordinates are passed in so its heuristic needs no div/mod.
                macro_rules! relax {
                    ($nidx:expr, $nx:expr, $ny:expr, $diag:expr) => {{
                        let nidx = $nidx;
                        let pixel = grid[nidx] as u32;
                        let passable = match impassable {
                            Some(im) => pixel != im,
                            None => true,
                        };
                        if passable {
                            let step = if $diag { pixel + (pixel >> 1) } else { pixel };
                            let ng = gu + step;
                            if ng < g[nidx] {
                                g[nidx] = ng;
                                came_from[nidx] = u as u32;
                                let nf = ng + heuristic!($nx, $ny);
                                buckets[nf as usize % NUM_BUCKETS].push(nidx as u32);
                                queued += 1;
                            }
                        }
                    }};
                }

                // Cardinal neighbours.
                if has_left {
                    relax!(u - height, x - 1, y, false);
                }
                if has_right {
                    relax!(u + height, x + 1, y, false);
                }
                if has_up {
                    relax!(u - 1, x, y - 1, false);
                }
                if has_down {
                    relax!(u + 1, x, y + 1, false);
                }
                // Diagonal neighbours cost ~1.5x as much.
                if has_left && has_up {
                    relax!(u - height - 1, x - 1, y - 1, true);
                }
                if has_right && has_up {
                    relax!(u + height - 1, x + 1, y - 1, true);
                }
                if has_left && has_down {
                    relax!(u - height + 1, x - 1, y + 1, true);
                }
                if has_right && has_down {
                    relax!(u + height + 1, x + 1, y + 1, true);
                }
            }

            cur += 1;
        }

        if g[end_idx] == u32::MAX {
            return None;
        }

        // Reconstruct the path by walking the predecessor links from the goal.
        let mut path = Vec::new();
        let mut node = end_idx;
        loop {
            path.push(((node / height) as u32, (node % height) as u32));
            if node == start_idx {
                break;
            }
            node = came_from[node] as usize;
        }
        path.reverse();

        Some((path, g[end_idx]))
    }
}
