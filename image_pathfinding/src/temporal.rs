use numpy::ndarray::{Array3, ArrayView3};
use pathfinding::prelude::{astar, dijkstra};

/// A position in the temporal volume (x, y, t).
pub type Pos3D = (u32, u32, u32);

/// A position in the temporal volume with a cost.
pub type Pos3DWithCost = (Pos3D, u32);

/// Search node for the multi-start routers.
///
/// A single virtual [`Node::Source`] is connected to every start position with a
/// zero-cost edge. Running one search from `Source` then finds the optimal path
/// from *any* start to *any* end in a single pass, instead of running a separate
/// search per start and taking the minimum.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Node {
    /// Virtual super-source connected to all start positions with zero cost.
    Source,
    /// A real position in the volume.
    At(Pos3D),
}

// MARK: Helpers

/// Load a list of grayscale images into a temporal volume (Width, Height, Time).
/// Note: Internally ndarray uses (x, y, t) indexing, so [x, y, t].
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

    let mut volume = Array3::zeros((width as usize, height as usize, depth));

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
                volume[[x as usize, y as usize, t]] = img.get_pixel(x, y)[0].max(1);
            }
        }
    }

    volume
}

/// Find neighbours with reach constraint: always move +1 along axis, can move within reach in other dimensions.
/// For temporal routing: axis=2 (time) is default, reach limits movement in x and y dimensions.
fn find_neighbours_with_reach(
    volume: ArrayView3<u8>,
    pos: Pos3D,
    axis: usize,
    reach: usize,
) -> Vec<Pos3DWithCost> {
    let (x, y, t) = pos;
    let (width, height, depth) = volume.dim(); // (x, y, t)

    // For temporal routing, axis should be 0 (x), 1 (y), or 2 (t)
    // Default axis=2 means we always move forward in time
    let axis = if axis >= 3 { 2 } else { axis };
    let reach_i = reach as i32;
    let (width_i, height_i) = (width as i32, height as i32);

    // Every move steps +1 in time (dt = 1), so we can never leave the last
    // slice. Bail out before allocating if no forward step is possible.
    if t as usize >= depth - 1 {
        return Vec::new();
    }
    let nt = t + 1;
    let nt_idx = nt as usize;

    let mut neighbours = Vec::new();

    match axis {
        0 => {
            // Moving along x axis (always +1); vary y within reach.
            if x as usize >= width - 1 {
                return neighbours;
            }
            let nx = x + 1;
            let nx_idx = nx as usize;
            neighbours.reserve(2 * reach + 1);
            for dy in -reach_i..=reach_i {
                let ny = y as i32 + dy;
                if ny >= 0 && ny < height_i {
                    let cost = volume[[nx_idx, ny as usize, nt_idx]] as u32;
                    neighbours.push(((nx, ny as u32, nt), cost));
                }
            }
        }
        1 => {
            // Moving along y axis (always +1); vary x within reach.
            if y as usize >= height - 1 {
                return neighbours;
            }
            let ny = y + 1;
            let ny_idx = ny as usize;
            neighbours.reserve(2 * reach + 1);
            for dx in -reach_i..=reach_i {
                let nx = x as i32 + dx;
                if nx >= 0 && nx < width_i {
                    let cost = volume[[nx as usize, ny_idx, nt_idx]] as u32;
                    neighbours.push(((nx as u32, ny, nt), cost));
                }
            }
        }
        2 => {
            // Moving along t axis (time, always +1); vary x and y within reach.
            neighbours.reserve((2 * reach + 1) * (2 * reach + 1));
            for dx in -reach_i..=reach_i {
                let nx = x as i32 + dx;
                if nx < 0 || nx >= width_i {
                    continue;
                }
                let nx_idx = nx as usize;
                for dy in -reach_i..=reach_i {
                    let ny = y as i32 + dy;
                    if ny >= 0 && ny < height_i {
                        let cost = volume[[nx_idx, ny as usize, nt_idx]] as u32;
                        neighbours.push(((nx as u32, ny as u32, nt), cost));
                    }
                }
            }
        }
        _ => {}
    }

    neighbours
}

/// Generate all positions at a specific axis index.
/// For temporal routing: if axis=2 and index=0, returns all (x, y, 0) positions.
fn generate_positions_at_axis_index(
    volume: ArrayView3<u8>,
    axis: usize,
    index: usize,
) -> Vec<Pos3D> {
    let (width, height, depth) = volume.dim(); // (x, y, t)
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
fn generate_default_starts_ends(volume: ArrayView3<u8>, axis: usize, is_start: bool) -> Vec<Pos3D> {
    let (width, height, depth) = volume.dim(); // (x, y, t)
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
    /// * `volume` - The temporal volume (Width, Height, Time) i.e. (x, y, t)
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
        volume: ArrayView3<u8>,
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

        // Run a single Dijkstra from a virtual super-source connected to every
        // start with a zero-cost edge. This finds the optimal path from any start
        // to any end in one pass.
        let result = dijkstra(
            &Node::Source,
            |node| -> Vec<(Node, u32)> {
                match node {
                    Node::Source => starts.iter().map(|&s| (Node::At(s), 0)).collect(),
                    Node::At(p) => find_neighbours_with_reach(volume, *p, axis, reach)
                        .into_iter()
                        .map(|(pos, cost)| (Node::At(pos), cost))
                        .collect(),
                }
            },
            |node| matches!(node, Node::At(p) if ends_set.contains(p)),
        );

        result.map(|(path, cost)| (strip_source(path), cost))
    }
}

/// Strip the virtual [`Node::Source`] from a returned path and unwrap the real
/// positions.
fn strip_source(path: Vec<Node>) -> Vec<Pos3D> {
    path.into_iter()
        .filter_map(|node| match node {
            Node::At(p) => Some(p),
            Node::Source => None,
        })
        .collect()
}

// MARK: A*

pub struct AStarTemporal {}

impl AStarTemporal {
    /// Admissible (and consistent) heuristic: a lower bound on the cost to reach
    /// any end position.
    ///
    /// Every move advances exactly one step along `axis` and costs at least
    /// `min_cost` (the smallest pixel value in the volume, always >= 1). Reaching
    /// an end at axis-coordinate `e` from `pos` at axis-coordinate `c` therefore
    /// requires exactly `e - c` steps, so `(e - c) * min_cost` can never exceed
    /// the true cost. Spatial distance is intentionally ignored: it only ever
    /// raises the true cost, so omitting it keeps the estimate a valid lower
    /// bound (unlike the previous Manhattan estimate, which could overestimate
    /// when `reach > 1`).
    fn heuristic_to_ends(&self, pos: Pos3D, ends: &[Pos3D], axis: usize, min_cost: u32) -> u32 {
        let (x, y, t) = pos;
        let mut min_steps = u32::MAX;

        for &(ex, ey, et) in ends {
            let steps = match axis {
                0 if x <= ex => ex - x,
                1 if y <= ey => ey - y,
                2 if t <= et => et - t,
                _ => continue,
            };
            min_steps = min_steps.min(steps);
        }

        // No end is reachable from here (none lie ahead along the axis); 0 is a
        // valid lower bound and avoids overflow when added to the path cost.
        if min_steps == u32::MAX {
            0
        } else {
            min_steps * min_cost
        }
    }

    /// Find the shortest route through a temporal volume from one side to another.
    ///
    /// # Arguments
    ///
    /// * `volume` - The temporal volume (Width, Height, Time) i.e. (x, y, t)
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
        volume: ArrayView3<u8>,
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

        // Smallest pixel value in the volume (>= 1 after the load-time clamp); the
        // per-step lower bound used by the admissible heuristic.
        let min_cost = volume.iter().copied().min().unwrap_or(1).max(1) as u32;

        // Run a single A* from a virtual super-source connected to every start
        // with a zero-cost edge. The source's heuristic is 0 (trivially
        // admissible), so it never distorts the search.
        let result = astar(
            &Node::Source,
            |node| -> Vec<(Node, u32)> {
                match node {
                    Node::Source => starts.iter().map(|&s| (Node::At(s), 0)).collect(),
                    Node::At(p) => find_neighbours_with_reach(volume, *p, axis, reach)
                        .into_iter()
                        .map(|(pos, cost)| (Node::At(pos), cost))
                        .collect(),
                }
            },
            |node| match node {
                Node::Source => 0,
                Node::At(p) => self.heuristic_to_ends(*p, &ends_vec, axis, min_cost),
            },
            |node| matches!(node, Node::At(p) if ends_set.contains(p)),
        );

        result.map(|(path, cost)| (strip_source(path), cost))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random volume so tests are reproducible.
    fn make_volume(width: usize, height: usize, depth: usize) -> Array3<u8> {
        let mut v = Array3::zeros((width, height, depth));
        let mut state: u64 = 0x9e3779b97f4a7c15;
        for x in 0..width {
            for y in 0..height {
                for t in 0..depth {
                    // xorshift
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    v[[x, y, t]] = ((state % 255) as u8).max(1);
                }
            }
        }
        v
    }

    /// Reference: original approach — run Dijkstra from each start separately and
    /// take the minimum cost. This is ground truth for the optimal cost.
    fn reference_min_cost(
        volume: ArrayView3<u8>,
        reach: usize,
        axis: usize,
        starts: &[Pos3D],
        ends: &[Pos3D],
    ) -> Option<u32> {
        let ends_set: std::collections::HashSet<Pos3D> = ends.iter().cloned().collect();
        let mut best = None;
        for &start in starts {
            if let Some((_, cost)) = dijkstra(
                &start,
                |&p| find_neighbours_with_reach(volume, p, axis, reach),
                |&p| ends_set.contains(&p),
            ) {
                best = Some(best.map_or(cost, |b: u32| b.min(cost)));
            }
        }
        best
    }

    /// Validate that a path is internally consistent: contiguous, on valid
    /// neighbour edges, starts/ends in the right sets, and the reported cost is
    /// the sum of destination-node costs.
    fn assert_valid_path(
        volume: ArrayView3<u8>,
        reach: usize,
        axis: usize,
        starts: &[Pos3D],
        ends: &[Pos3D],
        path: &[Pos3D],
        cost: u32,
    ) {
        assert!(!path.is_empty(), "path should not be empty");
        assert!(starts.contains(&path[0]), "path must begin at a start");
        assert!(
            ends.contains(path.last().unwrap()),
            "path must end at an end"
        );

        let mut summed = 0u32;
        for win in path.windows(2) {
            let (from, to) = (win[0], win[1]);
            let neighbours = find_neighbours_with_reach(volume, from, axis, reach);
            let edge = neighbours
                .iter()
                .find(|(pos, _)| *pos == to)
                .expect("each step must be a valid neighbour edge");
            summed += edge.1;
        }
        assert_eq!(summed, cost, "reported cost must equal summed edge costs");
    }

    #[test]
    fn dijkstra_super_source_matches_reference() {
        let volume = make_volume(12, 10, 8);
        let starts = generate_default_starts_ends(volume.view(), 2, true);
        let ends = generate_default_starts_ends(volume.view(), 2, false);

        let (path, cost) = DijkstraTemporal {}
            .find_route_over_time(
                volume.view(),
                Some(2),
                Some(2),
                Some(starts.clone()),
                Some(ends.clone()),
            )
            .expect("a route should exist");

        let reference = reference_min_cost(volume.view(), 2, 2, &starts, &ends)
            .expect("reference route should exist");

        assert_eq!(cost, reference, "super-source Dijkstra must be optimal");
        assert_valid_path(volume.view(), 2, 2, &starts, &ends, &path, cost);
    }

    #[test]
    fn dijkstra_super_source_matches_reference_explicit_endpoints() {
        let volume = make_volume(15, 9, 10);
        let starts = vec![(0, 0, 0), (14, 8, 0), (7, 4, 0)];
        let ends = vec![(7, 4, 9), (0, 8, 9)];

        let (path, cost) = DijkstraTemporal {}
            .find_route_over_time(
                volume.view(),
                Some(2),
                Some(2),
                Some(starts.clone()),
                Some(ends.clone()),
            )
            .expect("a route should exist");

        let reference = reference_min_cost(volume.view(), 2, 2, &starts, &ends)
            .expect("reference route should exist");

        assert_eq!(cost, reference);
        assert_valid_path(volume.view(), 2, 2, &starts, &ends, &path, cost);
    }

    #[test]
    fn astar_matches_optimal_across_reach() {
        // The heuristic is now admissible, so A* must return the optimal cost
        // (equal to Dijkstra) for every reach — including reach > 1, which broke
        // the old Manhattan heuristic.
        let volume = make_volume(12, 10, 8);
        let starts = generate_default_starts_ends(volume.view(), 2, true);
        let ends = generate_default_starts_ends(volume.view(), 2, false);

        for reach in 1..=3 {
            let (path, cost) = AStarTemporal {}
                .find_route_over_time(
                    volume.view(),
                    Some(reach),
                    Some(2),
                    Some(starts.clone()),
                    Some(ends.clone()),
                )
                .expect("a route should exist");

            assert_valid_path(volume.view(), reach, 2, &starts, &ends, &path, cost);

            let reference = reference_min_cost(volume.view(), reach, 2, &starts, &ends).unwrap();
            assert_eq!(cost, reference, "A* must be optimal at reach={reach}");
        }
    }

    #[test]
    fn astar_matches_optimal_explicit_endpoints() {
        let volume = make_volume(15, 9, 10);
        let starts = vec![(0, 0, 0), (14, 8, 0), (7, 4, 0)];
        let ends = vec![(7, 4, 9), (0, 8, 9)];

        for reach in 1..=3 {
            let (path, cost) = AStarTemporal {}
                .find_route_over_time(
                    volume.view(),
                    Some(reach),
                    Some(2),
                    Some(starts.clone()),
                    Some(ends.clone()),
                )
                .expect("a route should exist");

            assert_valid_path(volume.view(), reach, 2, &starts, &ends, &path, cost);

            let reference = reference_min_cost(volume.view(), reach, 2, &starts, &ends).unwrap();
            assert_eq!(cost, reference, "A* must be optimal at reach={reach}");
        }
    }

    /// A/B perf probe on the real frame volume. Ignored by default (needs the
    /// asset frames and is slow). Run with:
    ///   cargo test -p image_pathfinding ab_super_source_vs_per_start -- --ignored --nocapture
    #[test]
    #[ignore]
    fn ab_super_source_vs_per_start() {
        use std::time::Instant;

        let dir = "../assets/black-on-white-lv-like-heatmap-rotating";
        let paths: Vec<String> = (0..120)
            .map(|i| format!("{}/frame_{:03}.png", dir, i))
            .filter(|p| std::path::Path::new(p).exists())
            .collect();
        assert!(!paths.is_empty(), "asset frames not found at {dir}");
        let volume = load_images_to_volume(&paths);
        let (w, h, d) = volume.dim();
        println!("volume = {w}x{h}x{d}");

        let starts = generate_default_starts_ends(volume.view(), 2, true);
        let ends = generate_default_starts_ends(volume.view(), 2, false);
        let n_starts = starts.len();

        // Correctness on real data: super-source over a small start subset must
        // equal the per-start minimum over that same subset.
        let subset: Vec<Pos3D> = starts.iter().take(12).copied().collect();
        let ss_subset_cost = DijkstraTemporal {}
            .find_route_over_time(volume.view(), Some(2), Some(2), Some(subset.clone()), None)
            .expect("route")
            .1;
        let ref_subset_cost = reference_min_cost(volume.view(), 2, 2, &subset, &ends).unwrap();
        assert_eq!(
            ss_subset_cost, ref_subset_cost,
            "super-source must equal per-start min on real data"
        );
        println!(
            "real-data correctness OK: super-source subset == per-start min == {ss_subset_cost}"
        );

        // NEW: single super-source search over all default starts.
        let t0 = Instant::now();
        let (_path, cost) = DijkstraTemporal {}
            .find_route_over_time(volume.view(), Some(2), Some(2), None, None)
            .expect("route");
        let new_time = t0.elapsed();
        println!("super-source (1 search, {n_starts} starts): {new_time:?}, cost={cost}");

        // OLD: per-start Dijkstra. Running all {n_starts} is infeasible, so time a
        // small sample and extrapolate.
        let sample = 25.min(n_starts);
        let ends_set: std::collections::HashSet<Pos3D> = ends.iter().cloned().collect();
        let t1 = Instant::now();
        for &start in starts.iter().take(sample) {
            let _ = dijkstra(
                &start,
                |&p| find_neighbours_with_reach(volume.view(), p, 2, 2),
                |&p| ends_set.contains(&p),
            );
        }
        let per_start = t1.elapsed() / sample as u32;
        let est_old = per_start * n_starts as u32;
        println!(
            "per-start avg (n={sample}): {per_start:?} -> estimated old total: {est_old:?}"
        );
        println!(
            "estimated speedup: {:.0}x",
            est_old.as_secs_f64() / new_time.as_secs_f64()
        );

        // A* vs Dijkstra on a single start -> single end (where the admissible
        // heuristic can actually prune). Coordinates from benches/simple.rs.
        let one_start = vec![(269u32, 172u32, 0u32)];
        let one_end = vec![(413u32, 260u32, (d as u32) - 1)];

        let runs = 20;
        let t2 = Instant::now();
        let mut dij_cost = 0;
        for _ in 0..runs {
            dij_cost = DijkstraTemporal {}
                .find_route_over_time(
                    volume.view(),
                    Some(2),
                    Some(2),
                    Some(one_start.clone()),
                    Some(one_end.clone()),
                )
                .expect("route")
                .1;
        }
        let dij_time = t2.elapsed() / runs;

        let t3 = Instant::now();
        let mut astar_cost = 0;
        for _ in 0..runs {
            astar_cost = AStarTemporal {}
                .find_route_over_time(
                    volume.view(),
                    Some(2),
                    Some(2),
                    Some(one_start.clone()),
                    Some(one_end.clone()),
                )
                .expect("route")
                .1;
        }
        let astar_time = t3.elapsed() / runs;

        println!(
            "single target: Dijkstra {dij_time:?} (cost={dij_cost}) vs A* {astar_time:?} (cost={astar_cost})"
        );
        assert_eq!(dij_cost, astar_cost, "A* must stay optimal on real data");
        println!(
            "A* speedup vs Dijkstra: {:.2}x",
            dij_time.as_secs_f64() / astar_time.as_secs_f64()
        );
    }

    #[test]
    fn empty_starts_returns_none() {
        let volume = make_volume(4, 4, 4);
        assert!(
            DijkstraTemporal {}
                .find_route_over_time(volume.view(), Some(1), Some(2), Some(vec![]), None)
                .is_none()
        );
    }
}
