use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use image::{Rgb, RgbImage};
use image_pathfinding::{
    AStar2D, AStarTemporal, Dijkstra2D, DijkstraTemporal, Fringe2D, ImagePathfinder2D,
    load_images_to_volume, load_png_to_ndarray,
};
use std::fs;
use std::path::PathBuf;

/// Draw a filled circle on the image at the given position
fn draw_circle(img: &mut RgbImage, center_x: u32, center_y: u32, radius: u32, color: Rgb<u8>) {
    let width = img.width();
    let height = img.height();

    for dy in -(radius as i32)..=(radius as i32) {
        for dx in -(radius as i32)..=(radius as i32) {
            let distance_squared = (dx * dx + dy * dy) as f32;
            if distance_squared <= (radius * radius) as f32 {
                let x = center_x as i32 + dx;
                let y = center_y as i32 + dy;

                if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                    img.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Input images
    #[arg(required = true)]
    images: Vec<String>,

    /// Algorithm to use
    #[arg(long, value_enum, default_value_t = Algorithm::Astar)]
    algo: Algorithm,

    /// Start position (X Y). If not provided, uses all positions at axis=0
    #[arg(long, num_args = 2, value_names = ["X", "Y"])]
    start: Option<Vec<u32>>,

    /// End position (X Y). If not provided, uses all positions at axis=-1
    #[arg(long, num_args = 2, value_names = ["X", "Y"])]
    end: Option<Vec<u32>>,

    /// Reach parameter: number of elements that can be skipped along each non-axis dimension (default: 1)
    #[arg(long, default_value_t = 1)]
    reach: usize,

    /// Axis along which the path must always move forward (0=x, 1=y, 2=t/time, default: 2)
    #[arg(long, default_value_t = 2)]
    axis: usize,

    /// Output directory
    #[arg(long, default_value = "/tmp")]
    output_dir: PathBuf,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Algorithm {
    Astar,
    Dijkstra,
    Fringe,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Create output directory
    fs::create_dir_all(&cli.output_dir).context("Failed to create output directory")?;

    if cli.images.len() == 1 {
        // 2D Case - use ndarray for pathfinding
        println!("Running 2D pathfinding on {}", cli.images[0]);
        let img_path = &cli.images[0];
        let array = load_png_to_ndarray(img_path);

        // For 2D, we still need start/end positions
        let start_xy = if let Some(start) = &cli.start {
            (start[0], start[1])
        } else {
            anyhow::bail!("Start position is required for 2D pathfinding");
        };
        let end_xy = if let Some(end) = &cli.end {
            (end[0], end[1])
        } else {
            anyhow::bail!("End position is required for 2D pathfinding");
        };

        let path = match cli.algo {
            Algorithm::Dijkstra => Dijkstra2D {}.find_path_in_heatmap(&array, start_xy, end_xy),
            Algorithm::Astar => AStar2D {}.find_path_in_heatmap(&array, start_xy, end_xy),
            Algorithm::Fringe => Fringe2D {}.find_path_in_heatmap(&array, start_xy, end_xy),
        };

        if let Some((points, cost)) = path {
            println!("Path found with cost: {}", cost);
            let mut rgb_img = image::open(img_path)?.to_rgb8();
            let red = Rgb([255, 0, 0]);

            for (x, y) in points {
                // Draw a red circle with radius 3 at each path point
                draw_circle(&mut rgb_img, x, y, 3, red);
            }

            let file_name = std::path::Path::new(img_path)
                .file_name()
                .unwrap_or_default();
            let out_path = cli.output_dir.join(file_name);
            rgb_img
                .save(&out_path)
                .context("Failed to save output image")?;
            println!("Saved result to {:?}", out_path);
        } else {
            println!("No path found!");
        }
    } else {
        // Temporal Case - use find_route_over_time
        println!("Running temporal routing on {} frames", cli.images.len());
        println!("Reach: {}, Axis: {}", cli.reach, cli.axis);

        let volume = load_images_to_volume(&cli.images);
        println!("Volume shape: {:?}", volume.shape());

        // Prepare optional start/end positions
        let starts = cli.start.map(|start| {
            // For temporal routing, if start is provided, we need to determine the time coordinate
            // Since routing moves along an axis, if axis=2 (time), we use t=0
            // For other axes, we'd use the appropriate coordinate
            let t = if cli.axis == 2 { 0 } else { 0 }; // Default to t=0 for time axis
            vec![(start[0], start[1], t)]
        });

        let ends = cli.end.map(|end| {
            // For temporal routing, if end is provided, we need to determine the time coordinate
            // Since routing moves along an axis, if axis=2 (time), we use the last frame
            let (depth, _, _) = volume.dim();
            let t = if cli.axis == 2 {
                depth as u32 - 1
            } else {
                depth as u32 - 1
            };
            vec![(end[0], end[1], t)]
        });

        if let Some(ref starts) = starts {
            println!("Start positions: {:?}", starts);
        } else {
            println!("Using default start positions (all positions at axis=0)");
        }
        if let Some(ref ends) = ends {
            println!("End positions: {:?}", ends);
        } else {
            println!("Using default end positions (all positions at axis=-1)");
        }

        // Dispatch algorithm using find_route_over_time
        let path = match cli.algo {
            Algorithm::Dijkstra => DijkstraTemporal {}.find_route_over_time(
                &volume,
                Some(cli.reach),
                Some(cli.axis),
                starts,
                ends,
            ),
            Algorithm::Astar => AStarTemporal {}.find_route_over_time(
                &volume,
                Some(cli.reach),
                Some(cli.axis),
                starts,
                ends,
            ),
            Algorithm::Fringe => {
                anyhow::bail!(
                    "Fringe algorithm is not supported for temporal routing. Use Dijkstra or Astar instead."
                );
            }
        };

        if let Some((points, cost)) = path {
            let path_length = points.len();
            println!("Path found with cost: {}", cost);
            println!("Path length: {} points", path_length);

            // Group points by time t
            let mut points_by_time: std::collections::HashMap<u32, Vec<(u32, u32)>> =
                std::collections::HashMap::new();
            for (x, y, t) in &points {
                points_by_time.entry(*t).or_default().push((*x, *y));
            }

            // Write route to text file
            let route_file_path = cli.output_dir.join("route.txt");
            let mut route_file = std::fs::File::create(&route_file_path)
                .context("Failed to create route.txt file")?;
            use std::io::Write;

            // Write header
            writeln!(
                route_file,
                "# Route file: each line contains the frame number and x y coordinates for that frame"
            )?;
            writeln!(route_file, "# Format: frame_number x1 y1 x2 y2 ...")?;
            writeln!(route_file, "# Total cost: {}", cost)?;
            writeln!(route_file, "# Total path length: {} points", path_length)?;

            // Write route for each frame
            for (t, _img_path) in cli.images.iter().enumerate() {
                let t_u32 = t as u32;
                write!(route_file, "{}", t_u32)?;

                if let Some(pts) = points_by_time.get(&t_u32) {
                    for &(x, y) in pts {
                        write!(route_file, " {} {}", x, y)?;
                    }
                }
                writeln!(route_file)?;
            }
            route_file
                .flush()
                .context("Failed to flush route.txt file")?;
            println!("Saved route to {:?}", route_file_path);

            for (t, img_path) in cli.images.iter().enumerate() {
                let t_u32 = t as u32;
                let mut rgb_img = image::open(img_path)
                    .with_context(|| format!("Failed to open image at {}", img_path))?
                    .to_rgb8();

                if let Some(pts) = points_by_time.get(&t_u32) {
                    let red = Rgb([255, 0, 0]);
                    for &(x, y) in pts {
                        // Draw a red circle with radius 3 at each path point
                        draw_circle(&mut rgb_img, x, y, 3, red);
                    }
                }

                let file_name = std::path::Path::new(img_path)
                    .file_name()
                    .unwrap_or_default();
                let out_path = cli.output_dir.join(file_name);
                rgb_img
                    .save(&out_path)
                    .context("Failed to save output image")?;
            }
            println!("Saved {} frames to {:?}", cli.images.len(), cli.output_dir);
        } else {
            println!("No path found!");
        }
    }

    Ok(())
}
