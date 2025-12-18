# pathfinding-py

A high-performance pathfinding library implemented in Rust with Python bindings. This project provides efficient algorithms for finding optimal paths through 2D heatmaps and temporal volumes (3D sequences of images).

## Features
- **2D Pathfinding**: Find optimal paths through 2D heatmaps using pixel values as costs
- **Temporal Pathfinding**: Route through 3D volumes (time series of images) with constraints
- **Multiple Algorithms**: Support for A*, Dijkstra, and Fringe search algorithms
- **Python API**: Native Python bindings using PyO3 and maturin
- **Rust CLI**: Command-line tool for quick pathfinding on images
- **High Performance**: Optimized Rust implementation for fast pathfinding

## Project Structure

```
pathfinding-py/
├── image_pathfinding/     # Core Rust library with pathfinding algorithms
│   ├── src/
│   │   ├── bidimensional.rs  # 2D pathfinding implementations
│   │   └── temporal.rs        # Temporal (3D) pathfinding implementations
│   └── benches/              # Benchmark suite
├── pathfinding_cli/        # Rust CLI application
├── pathfinding_py/         # Python bindings (PyO3 + maturin)
└── tests/                   # Python integration tests
```

## Installation

### Prerequisites

- Rust (latest stable)
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (for Python package management)
- [maturin](https://github.com/PyO3/maturin) (for building Python extensions)
- [just](https://github.com/casey/just) (for running project commands)

### Building the Python Module

```bash
cd pathfinding_py
uvx maturin develop --release
```

This will build and install the `pathfinding_py` module into your current Python environment.

## Usage

### Python API

```python
import numpy as np
import pathfinding_py

# 2D Pathfinding
# Create a 2D heatmap (grayscale image) as a NumPy array
heatmap = np.array([
    [10, 50, 10],
    [10, 200, 10],
    [10, 50, 10]
], dtype=np.uint8)

# Find path from (0, 0) to (2, 2) using A* algorithm
result = pathfinding_py.find_path_2d(
    heatmap,
    start=(0, 0),
    end=(2, 2),
    algorithm="astar"
)

if result:
    path, cost = result
    print(f"Path found with cost: {cost}")
    print(f"Path: {path}")
else:
    print("No path found")

# Temporal Pathfinding
# Create a 3D volume (time, height, width)
volume = np.ones((10, 100, 100), dtype=np.uint8) * 100

# Find route through temporal volume
result = pathfinding_py.find_route_temporal(
    volume,
    algorithm="astar",
    reach=2,  # Allow skipping up to 2 pixels in spatial dimensions
    axis=2    # Move forward along time axis (default)
)

if result:
    route, cost = result
    print(f"Route found with cost: {cost}")
    print(f"Route length: {len(route)} points")
```

### Rust CLI

```bash
# 2D pathfinding on a single image
cargo run --release -p pathfinding_cli -- \
    --start 269 172 \
    --end 470 263 \
    --algo astar \
    --output-dir output \
    assets/black-on-white-lv-like-heatmap.png

# Temporal pathfinding on multiple frames
cargo run --release -p pathfinding_cli -- \
    --start 269 172 \
    --end 413 260 \
    --algo astar \
    --reach 2 \
    --axis 2 \
    assets/black-on-white-lv-like-heatmap-rotating/*.png
```

## Available Commands

The project uses `just` for task management. Run `just -l` to see all available commands:

- `just test` - Run 2D pathfinding on a single image
- `just test-python` - Run Python integration tests
- `just bench` - Benchmark the pathfinding algorithms
- `just video` - Run temporal pathfinding on rotating frames and create a video

## API Reference

### `find_path_2d(array, start, end, algorithm)`

Find a path in a 2D heatmap.

**Parameters:**
- `array`: 2D NumPy array with dtype `uint8` (shape: height, width)
- `start`: Start position as `(x, y)` tuple
- `end`: End position as `(x, y)` tuple
- `algorithm`: Algorithm to use: `"astar"`, `"dijkstra"`, or `"fringe"`

**Returns:**
- `Optional[Tuple[List[Tuple[int, int]], int]]`: The path found and total cost, or `None` if no path was found

### `find_route_temporal(array, algorithm, *, reach=None, axis=None, starts=None, ends=None)`

Find a route through a temporal volume.

**Parameters:**
- `array`: 3D NumPy array with dtype `uint8` (shape: time, height, width)
- `algorithm`: Algorithm to use: `"astar"` or `"dijkstra"`
- `reach` (optional): Number of elements that can be skipped along each non-axis dimension (default: 1)
- `axis` (optional): The axis along which the path must always move forward (default: 2 for time)
- `starts` (optional): Start positions as list of `(x, y, t)` tuples. If `None`, uses all positions at axis=0
- `ends` (optional): End positions as list of `(x, y, t)` tuples. If `None`, uses all positions at axis=-1

**Returns:**
- `Optional[Tuple[List[Tuple[int, int, int]], int]]`: The route found and total cost, or `None` if no route was found

## Examples

### Example: 2D Pathfinding

```python
import numpy as np
import pathfinding_py

# Create a simple maze-like heatmap
# Low values = easy to traverse, high values = obstacles
maze = np.ones((10, 10), dtype=np.uint8) * 200
maze[0, :] = 10  # Top row is clear
maze[:, 9] = 10  # Right column is clear
maze[9, :] = 10  # Bottom row is clear

# Find path from top-left to bottom-right
result = pathfinding_py.find_path_2d(maze, (0, 0), (9, 9), "astar")
if result:
    path, cost = result
    print(f"Found path with {len(path)} steps, cost: {cost}")
```

### Example: Temporal Pathfinding

```python
import numpy as np
import pathfinding_py

# Create a temporal volume representing a moving obstacle
volume = np.ones((5, 20, 20), dtype=np.uint8) * 100

# Create a path that moves diagonally through time
for t in range(5):
    x = t * 2
    y = t * 2
    volume[t, y, x] = 20  # Low cost path

# Find route through the temporal volume
result = pathfinding_py.find_route_temporal(
    volume,
    algorithm="astar",
    reach=1,
    axis=2  # Time axis
)

if result:
    route, cost = result
    print(f"Found route with {len(route)} points, cost: {cost}")
    # Verify route moves forward in time
    times = [pos[2] for pos in route]
    assert times == sorted(times), "Route must move forward in time"
```

## Development

### Running Tests

```bash
# Python integration tests
just test-python

# Rust benchmarks
just bench
```

### Building

```bash
# Build Rust crates
cargo build --release

# Build Python module
cd pathfinding_py
uvx maturin develop --release
```