//! Throwaway driver for line-level profiling of `Dijkstra2D` with cargo-flamegraph.
//! Run: CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --example profile_dijkstra

use image_pathfinding::{Dijkstra2D, ImagePathfinder2D, Pos2D, load_png_to_ndarray};
use std::hint::black_box;

const IMAGE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../assets/black-on-white-lv-like-heatmap.png"
);
const START: Pos2D = (269, 172);
const END: Pos2D = (470, 263);

fn main() {
    let array = load_png_to_ndarray(IMAGE);
    let solver = Dijkstra2D {};

    let iters: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);

    let mut acc = 0u64;
    for _ in 0..iters {
        let r = solver.find_path_in_heatmap(
            black_box(array.view()),
            black_box(START),
            black_box(END),
            None,
            None,
        );
        acc = acc.wrapping_add(r.unwrap().1 as u64);
    }
    println!("iters={iters} checksum={acc}");
}
