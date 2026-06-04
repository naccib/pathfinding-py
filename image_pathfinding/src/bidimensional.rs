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
    /// * `max_cost` - Optional cost budget. If provided, the search abandons as soon as it can
    ///   prove no path to the goal costs `<= max_cost`, returning `None`. Paths costing exactly
    ///   `max_cost` are still accepted.
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
        max_cost: Option<u32>,
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
        max_cost: Option<u32>,
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

            // Cost cutoff: Dial's settles nodes in non-decreasing distance order, so `cur` is a
            // lower bound on every node still on the frontier. Once it exceeds the budget, no
            // remaining node — the goal included — can be reached within budget, so stop.
            if let Some(cap) = max_cost {
                if cur > cap {
                    break 'search;
                }
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

        // `dist[end_idx]` is set during relaxation, before the goal is popped, so the early
        // `break` alone can leave a tentative over-budget cost here. Reject it explicitly: at the
        // point the cutoff fires every node with a final distance `<= cap` is already settled, so
        // `dist[end_idx]` is either its true optimum (`<= cap`) or genuinely exceeds the budget.
        let found = dist[end_idx];
        if found == u32::MAX || max_cost.is_some_and(|cap| found > cap) {
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
        max_cost: Option<u32>,
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

            // Cost cutoff: `cur` is the current `f = g + h` level and only grows. Because the
            // heuristic is admissible, `f` never overestimates the cost of a path through a node,
            // so once `cur > cap` no frontier node can reach the goal within budget — stop.
            if let Some(cap) = max_cost {
                if cur > cap {
                    break 'search;
                }
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

        // As in Dijkstra2D, `g[end_idx]` may hold a tentative over-budget cost when the cutoff
        // fires before the goal is popped, so reject `found > cap` explicitly rather than relying
        // on the early `break`. (This, like A*'s optimality itself, assumes an admissible
        // heuristic; see the `heuristic!` note above.)
        let found = g[end_idx];
        if found == u32::MAX || max_cost.is_some_and(|cap| found > cap) {
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

// MARK: Bidirectional Dijkstra

/// A 2D pathfinder that runs Dijkstra from both the start and the goal at once and meets
/// in the middle. Each search settles roughly the radius of an equivalent one-directional
/// Dijkstra, so the two together explore on the order of the area of two half-radius discs
/// instead of one full-radius disc — a large win when start and goal are far apart.
pub struct BiDijkstra2D {}

impl ImagePathfinder2D for BiDijkstra2D {
    fn find_path_in_heatmap(
        &self,
        array: ArrayView2<u8>,
        start_pos: Pos2D,
        end_pos: Pos2D,
        impassable: Option<u8>,
        max_cost: Option<u32>,
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

        // A zero-length path needs no search and would otherwise have to be special-cased
        // throughout the meeting logic below. The cap is inclusive and `max_cost >= 0`, so a
        // cost-0 path is always within budget.
        if start_idx == end_idx {
            return Some((vec![start_pos], 0));
        }

        // One set of flat arrays per direction. `dist_f`/`from_f` describe the forward search
        // rooted at the start; `dist_b`/`from_b` the backward search rooted at the goal, run on
        // the reverse graph.
        let mut dist_f = vec![u32::MAX; n];
        let mut dist_b = vec![u32::MAX; n];
        let mut from_f = vec![u32::MAX; n];
        let mut from_b = vec![u32::MAX; n];

        // Dial's bucket queue, one per direction, exactly as in Dijkstra2D. The max edge cost is
        // the largest diagonal cost `255 + 255 / 2 = 382`, so a 512-bucket circular window holds
        // each direction's whole frontier and `% NUM_BUCKETS` is a bitmask.
        const NUM_BUCKETS: usize = 512; // power of two > 382 (max diagonal edge cost)
        let mut buckets_f: [Vec<u32>; NUM_BUCKETS] = std::array::from_fn(|_| Vec::new());
        let mut buckets_b: [Vec<u32>; NUM_BUCKETS] = std::array::from_fn(|_| Vec::new());
        let mut queued_f = 0usize;
        let mut queued_b = 0usize;

        let impassable = impassable.map(|v| v as u32);

        dist_f[start_idx] = 0;
        buckets_f[0].push(start_idx as u32);
        queued_f += 1;

        dist_b[end_idx] = 0;
        buckets_b[0].push(end_idx as u32);
        queued_b += 1;

        // Best complete s->t path cost found so far and the node where its two halves meet.
        // `best == dist_f[meet] + dist_b[meet]` is maintained as an invariant (see `relax`).
        let mut best = u32::MAX;
        let mut meet = u32::MAX;

        // Absolute distance level each direction is currently scanning; equivalently the minimum
        // key on each frontier once advanced to its next non-empty bucket.
        let mut cur_f = 0u32;
        let mut cur_b = 0u32;

        // Settle one direction's current bucket level, relaxing every neighbour. The cost of an
        // edge differs by direction: a forward edge costs its *destination* pixel (`$current = false`),
        // while a backward edge is a reverse forward edge and so costs the *current* node's pixel
        // (`$current = true`). The branch on that bool literal is resolved at compile time. The
        // impassable test always looks at the neighbour being entered, matching Dijkstra2D's
        // "never step onto an impassable pixel" rule in both directions.
        macro_rules! expand_dir {
            (
                $cur:ident, $buckets:ident, $queued:ident,
                $dist:ident, $from:ident, $dist_other:ident,
                $current:literal
            ) => {{
                let level = $cur;
                let b = level as usize % NUM_BUCKETS;
                // Every entry in this bucket has distance exactly `level`; relaxation only ever
                // pushes to strictly greater levels, which (being within 382 < NUM_BUCKETS) alias
                // to different buckets, so this drain never pushes back into the bucket it pops.
                while let Some(u) = $buckets[b].pop() {
                    $queued -= 1;
                    let u = u as usize;
                    // Stale entry: re-inserted at a smaller distance and already settled there.
                    if $dist[u] != level {
                        continue;
                    }

                    let x = u / height;
                    let y = u % height;
                    let has_left = x > 0;
                    let has_right = x + 1 < width;
                    let has_up = y > 0;
                    let has_down = y + 1 < height;
                    let cur_pixel = grid[u] as u32;

                    macro_rules! relax {
                        ($nidx:expr, $diag:expr) => {{
                            let nidx = $nidx;
                            let npix = grid[nidx] as u32;
                            let passable = match impassable {
                                Some(im) => npix != im,
                                None => true,
                            };
                            if passable {
                                let pixel = if $current { cur_pixel } else { npix };
                                let cost = if $diag { pixel + (pixel >> 1) } else { pixel };
                                let nd = level + cost;
                                if nd < $dist[nidx] {
                                    $dist[nidx] = nd;
                                    $from[nidx] = u as u32;
                                    $buckets[nd as usize % NUM_BUCKETS].push(nidx as u32);
                                    $queued += 1;
                                    // Meeting check: if the opposite search has already reached
                                    // this node, `s -> nidx -> t` is a complete path. Recording it
                                    // on every improvement keeps `best == dist_f[meet] + dist_b[meet]`,
                                    // so the reconstructed path's cost always equals `best`.
                                    let other = $dist_other[nidx];
                                    if other != u32::MAX {
                                        let cand = nd + other;
                                        if cand < best {
                                            best = cand;
                                            meet = nidx as u32;
                                        }
                                    }
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

                $cur += 1;
            }};
        }

        loop {
            // If either search has drained its whole frontier it has settled its entire reachable
            // component, including the other search's root if that root is reachable. Because the
            // goal carries `dist_b = 0` (and the start `dist_f = 0`) from the outset, any complete
            // path is captured in `best` the moment one search reaches the other's root, so it is
            // safe to stop here: `best` is already optimal (or no path exists).
            if queued_f == 0 || queued_b == 0 {
                break;
            }

            // Advance each direction to its next non-empty bucket; while `queued > 0` the window
            // invariant guarantees one exists within `NUM_BUCKETS` steps.
            while buckets_f[cur_f as usize % NUM_BUCKETS].is_empty() {
                cur_f += 1;
            }
            while buckets_b[cur_b as usize % NUM_BUCKETS].is_empty() {
                cur_b += 1;
            }

            // Standard bidirectional stopping rule: `cur_f` and `cur_b` are the minimum keys on the
            // two frontiers, so `cur_f + cur_b` is a lower bound on every s->t path not yet found.
            // Once it reaches the best path already found, that path is optimal.
            if best != u32::MAX && cur_f + cur_b >= best {
                break;
            }

            // Cost cutoff: every s->t path still to be discovered costs at least `cur_f + cur_b`, so
            // once that exceeds the budget none can come in within it. Any in-budget path would
            // already be in `best` (it would have been found before this bound passed it), and the
            // final `best > cap` check below rejects an over-budget `best`.
            if let Some(cap) = max_cost {
                if cur_f + cur_b > cap {
                    break;
                }
            }

            // Expand whichever frontier is "behind" in distance. Balancing on the minimum key keeps
            // the two search radii comparable, which is what makes the meet-in-the-middle saving pay off.
            if cur_f <= cur_b {
                expand_dir!(cur_f, buckets_f, queued_f, dist_f, from_f, dist_b, false);
            } else {
                expand_dir!(cur_b, buckets_b, queued_b, dist_b, from_b, dist_f, true);
            }
        }

        if meet == u32::MAX {
            return None;
        }

        let meet = meet as usize;
        // `best` is maintained equal to this sum, but read it straight from the arrays so the
        // returned cost provably matches the path reconstructed from the predecessor links below.
        let cost = dist_f[meet] + dist_b[meet];
        if max_cost.is_some_and(|cap| cost > cap) {
            return None;
        }

        // Reconstruct start -> meet by walking forward predecessors, then append meet -> goal by
        // walking backward predecessors (which already point toward the goal).
        let mut path_idx = Vec::new();
        let mut node = meet;
        loop {
            path_idx.push(node);
            if node == start_idx {
                break;
            }
            node = from_f[node] as usize;
        }
        path_idx.reverse();

        let mut node = meet;
        while from_b[node] != u32::MAX {
            node = from_b[node] as usize;
            path_idx.push(node);
        }

        let path = path_idx
            .into_iter()
            .map(|i| ((i / height) as u32, (i % height) as u32))
            .collect();

        Some((path, cost))
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array2;

    /// Cost of a path under the same model the solvers use: a cardinal step costs the
    /// destination pixel, a diagonal step costs `pixel + pixel / 2`. Used to independently
    /// re-derive a returned path's cost from the grid, rather than trusting the solver's own sum.
    fn path_cost(grid: &Array2<u8>, path: &[Pos2D]) -> u32 {
        let mut total = 0u32;
        for w in path.windows(2) {
            let (ax, ay) = (w[0].0 as i64, w[0].1 as i64);
            let (bx, by) = (w[1].0, w[1].1);
            let pixel = grid[[bx as usize, by as usize]] as u32;
            let diag = (ax - bx as i64).abs() == 1 && (ay - by as i64).abs() == 1;
            total += if diag { pixel + (pixel >> 1) } else { pixel };
        }
        total
    }

    /// Deterministic xorshift so the property test is reproducible without a deps on `rand`.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u32) -> u32 {
            (self.next() % n as u64) as u32
        }
    }

    /// On a wide range of random grids, start/end pairs, and impassable settings, bidirectional
    /// Dijkstra must agree with the reference `Dijkstra2D` on the optimal cost, and the path it
    /// returns must be valid (contiguous, in-bounds, endpoints correct) and actually cost that much.
    #[test]
    fn bidijkstra_matches_dijkstra() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let reference = Dijkstra2D {};
        let bidir = BiDijkstra2D {};

        for _ in 0..400 {
            let width = 1 + rng.below(12) as usize;
            let height = 1 + rng.below(12) as usize;
            let mut grid = Array2::<u8>::zeros((width, height));
            for x in 0..width {
                for y in 0..height {
                    // Pixels in [1, 6]; small range keeps many ties, stressing the meeting logic.
                    grid[[x, y]] = 1 + (rng.below(6) as u8);
                }
            }

            // Half the trials carve out impassable cells (value 7) to exercise that path.
            let impassable = if rng.below(2) == 0 { Some(7u8) } else { None };
            if impassable.is_some() {
                for x in 0..width {
                    for y in 0..height {
                        if rng.below(4) == 0 {
                            grid[[x, y]] = 7;
                        }
                    }
                }
            }

            let start = (rng.below(width as u32), rng.below(height as u32));
            let end = (rng.below(width as u32), rng.below(height as u32));
            // The solvers never step onto an impassable pixel, so the endpoints must be passable
            // for a path to exist; clear them so both solvers see the same reachable graph.
            if impassable.is_some() {
                grid[[start.0 as usize, start.1 as usize]] = 1;
                grid[[end.0 as usize, end.1 as usize]] = 1;
            }

            let expected =
                reference.find_path_in_heatmap(grid.view(), start, end, impassable, None);
            let actual = bidir.find_path_in_heatmap(grid.view(), start, end, impassable, None);

            match (expected, actual) {
                (None, None) => {}
                (Some((_, ec)), Some((path, ac))) => {
                    assert_eq!(ec, ac, "cost mismatch on {width}x{height} {start:?}->{end:?}");
                    assert_eq!(path.first(), Some(&start), "path must start at start");
                    assert_eq!(path.last(), Some(&end), "path must end at end");
                    assert_eq!(
                        path_cost(&grid, &path),
                        ac,
                        "returned path does not actually cost what was reported"
                    );
                    // Each step must move to an 8-connected neighbour.
                    for w in path.windows(2) {
                        let dx = (w[0].0 as i64 - w[1].0 as i64).abs();
                        let dy = (w[0].1 as i64 - w[1].1 as i64).abs();
                        assert!(dx <= 1 && dy <= 1 && dx + dy > 0, "non-adjacent step in path");
                    }
                }
                (e, a) => panic!("existence mismatch: reference={:?} bidir={:?}", e.is_some(), a.is_some()),
            }
        }
    }

    /// The `max_cost` budget must behave exactly as it does for the other solvers: reject below the
    /// optimum, accept at and above it, and never change the optimal cost when generous.
    #[test]
    fn bidijkstra_respects_max_cost() {
        let mut grid = Array2::<u8>::from_elem((6, 6), 5u8);
        for i in 0..6 {
            grid[[i, i]] = 1;
        }
        let bidir = BiDijkstra2D {};
        let start = (0, 0);
        let end = (5, 5);

        let (_, optimal) = bidir
            .find_path_in_heatmap(grid.view(), start, end, None, None)
            .expect("a path exists");

        assert!(
            bidir
                .find_path_in_heatmap(grid.view(), start, end, None, Some(optimal - 1))
                .is_none(),
            "a budget below the optimum must yield no path"
        );
        assert_eq!(
            bidir
                .find_path_in_heatmap(grid.view(), start, end, None, Some(optimal))
                .map(|(_, c)| c),
            Some(optimal),
            "a budget exactly at the optimum must return the optimal path"
        );
        assert_eq!(
            bidir
                .find_path_in_heatmap(grid.view(), start, end, None, Some(optimal + 1000))
                .map(|(_, c)| c),
            Some(optimal),
            "a generous budget must not change the optimum"
        );
    }

    /// A start equal to the goal is a zero-length, zero-cost path.
    #[test]
    fn bidijkstra_start_equals_end() {
        let grid = Array2::<u8>::from_elem((4, 4), 3u8);
        let result = BiDijkstra2D {}.find_path_in_heatmap(grid.view(), (2, 2), (2, 2), None, None);
        assert_eq!(result, Some((vec![(2, 2)], 0)));
    }
}
