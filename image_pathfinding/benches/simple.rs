use criterion::{Criterion, criterion_group, criterion_main};
use image_pathfinding::{
    AStar2D, AStarTemporal, Dijkstra2D, DijkstraTemporal, Fringe2D, ImagePathfinder2D, Pos2D,
    load_images_to_volume, load_png_to_ndarray,
};
use std::hint::black_box;
use std::path::Path;

const IMAGE_PATH_2D: &str = "../assets/black-on-white-lv-like-heatmap.png";
const START_POS_2D: Pos2D = (269, 172);
const END_POS_2D: Pos2D = (470, 263);

const FRAMES_DIR_3D: &str = "../assets/black-on-white-lv-like-heatmap-rotating";
const START_POS_3D_X: u32 = 269;
const START_POS_3D_Y: u32 = 172;
const END_POS_3D_X: u32 = 413;
const END_POS_3D_Y: u32 = 260;
const REACH: usize = 2;
const AXIS: usize = 2;

fn criterion_benchmark(c: &mut Criterion) {
    // 2D Setup
    let array = load_png_to_ndarray(IMAGE_PATH_2D);

    let dji2d = Dijkstra2D {};
    let astar2d = AStar2D {};
    let fringe2d = Fringe2D {};

    c.bench_function("2D Dijkstra 600x600", |b| {
        b.iter(|| {
            dji2d.find_path_in_heatmap(
                black_box(array.view()),
                black_box(START_POS_2D),
                black_box(END_POS_2D),
                None,
            )
        })
    });

    c.bench_function("2D A* 600x600", |b| {
        b.iter(|| {
            astar2d.find_path_in_heatmap(
                black_box(array.view()),
                black_box(START_POS_2D),
                black_box(END_POS_2D),
                None,
            )
        })
    });

    c.bench_function("2D Fringe 600x600", |b| {
        b.iter(|| {
            fringe2d.find_path_in_heatmap(
                black_box(array.view()),
                black_box(START_POS_2D),
                black_box(END_POS_2D),
                None,
            )
        })
    });

    // 3D Setup
    // Check if output_frames exists, if not, skip 3D benchmarks or panic?
    // Let's assume they exist as per previous task.
    let mut paths = Vec::new();
    if Path::new(FRAMES_DIR_3D).exists() {
        for i in 0..120 {
            let p = format!("{}/frame_{:03}.png", FRAMES_DIR_3D, i);
            if Path::new(&p).exists() {
                paths.push(p);
            }
        }
    }

    if !paths.is_empty() {
        let volume = load_images_to_volume(&paths);

        let dijkstra_temporal = DijkstraTemporal {};
        let astar_temporal = AStarTemporal {};

        // Prepare start and end positions for temporal routing
        let starts = vec![(START_POS_3D_X, START_POS_3D_Y, 0)];
        let ends = vec![(END_POS_3D_X, END_POS_3D_Y, 119)]; // Last frame index

        c.bench_function("Temporal Dijkstra find_route_over_time (120 frames)", |b| {
            b.iter(|| {
                dijkstra_temporal.find_route_over_time(
                    black_box(volume.view()),
                    Some(REACH),
                    Some(AXIS),
                    Some(starts.clone()),
                    Some(ends.clone()),
                )
            })
        });

        c.bench_function("Temporal A* find_route_over_time (120 frames)", |b| {
            b.iter(|| {
                astar_temporal.find_route_over_time(
                    black_box(volume.view()),
                    Some(REACH),
                    Some(AXIS),
                    Some(starts.clone()),
                    Some(ends.clone()),
                )
            })
        });
    } else {
        println!(
            "Skipping temporal benchmarks: frames not found in {}",
            FRAMES_DIR_3D
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
