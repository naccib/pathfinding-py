"""Integration tests for pathfinding_py module."""

import pathlib

import numpy as np
import pytest
from PIL import Image

try:
    import pathfinding_py
except ImportError:
    pytest.skip(
        "pathfinding_py module not available. Build it with 'maturin develop' first.",
        allow_module_level=True,
    )

# Get the project root directory (parent of tests/)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"


def test_find_path_2d_astar():
    """Test 2D pathfinding with A* algorithm."""
    # Create a simple 10x10 grid with low cost path from (0,0) to (9,9)
    # Use low values (0-50) for the path, higher values (200-255) for obstacles
    array = np.ones((10, 10), dtype=np.uint8) * 200

    # Create a clear diagonal path
    for i in range(10):
        array[i, i] = 10

    start = (0, 0)
    end = (9, 9)

    result = pathfinding_py.find_path_2d(array, start, end, "astar")

    assert result is not None, "Path should be found"
    path, cost = result
    assert len(path) > 0, "Path should contain at least one point"
    assert path[0] == start, "Path should start at the start position"
    assert path[-1] == end, "Path should end at the end position"
    assert cost > 0, "Cost should be positive"


def test_find_path_2d_dijkstra():
    """Test 2D pathfinding with Dijkstra algorithm."""
    # Create a simple 5x5 grid
    array = np.ones((5, 5), dtype=np.uint8) * 50

    # Create a clear path
    array[0, :] = 10  # Top row
    array[:, 4] = 10  # Right column

    start = (0, 0)
    end = (4, 4)

    result = pathfinding_py.find_path_2d(array, start, end, "dijkstra")

    assert result is not None, "Path should be found"
    path, cost = result
    assert len(path) > 0, "Path should contain at least one point"
    assert path[0] == start, "Path should start at the start position"
    assert path[-1] == end, "Path should end at the end position"


def test_find_path_2d_fringe():
    """Test 2D pathfinding with Fringe algorithm."""
    # Create a simple 8x8 grid
    array = np.ones((8, 8), dtype=np.uint8) * 100

    # Create a clear path
    for i in range(8):
        array[i, i] = 20

    start = (0, 0)
    end = (7, 7)

    result = pathfinding_py.find_path_2d(array, start, end, "fringe")

    assert result is not None, "Path should be found"
    path, cost = result
    assert len(path) > 0, "Path should contain at least one point"
    assert path[0] == start, "Path should start at the start position"
    assert path[-1] == end, "Path should end at the end position"


def test_find_path_2d_invalid_algorithm():
    """Test that invalid algorithm raises an error."""
    array = np.ones((5, 5), dtype=np.uint8) * 50

    with pytest.raises(Exception):  # Should raise ValueError or similar
        pathfinding_py.find_path_2d(array, (0, 0), (4, 4), "invalid_algo")


def test_find_route_temporal_astar():
    """Test temporal routing with A* algorithm."""
    # Create a simple 3D volume: 5 time frames, 10x10 spatial dimensions
    # Shape: (time, height, width)
    volume = np.ones((5, 10, 10), dtype=np.uint8) * 150

    # Create a clear path that moves forward in time, one step at a time
    # Each step moves within reach=1, so the path is: (0,0,0) -> (1,1,1) -> (2,2,2) -> (3,3,3) -> (4,4,4)
    for t in range(5):
        x = min(t, 9)
        y = min(t, 9)
        volume[t, y, x] = 20

    start = (0, 0, 0)
    end = (4, 4, 4)  # End is on the path and reachable with reach=1

    result = pathfinding_py.find_route_temporal(volume, "astar", start, end)

    assert result is not None, "Route should be found"
    path, cost = result
    assert len(path) > 0, "Path should contain at least one point"
    # Check that path moves forward in time
    times = [pos[2] for pos in path]
    assert times == sorted(times), "Path should move forward in time"
    assert cost > 0, "Cost should be positive"


def test_find_route_temporal_dijkstra():
    """Test temporal routing with Dijkstra algorithm."""
    # Create a simple 3D volume: 3 time frames, 5x5 spatial dimensions
    volume = np.ones((3, 5, 5), dtype=np.uint8) * 100

    # Create a clear path that moves one step at a time (reachable with reach=1)
    # Path: (0,0,0) -> (1,1,1) -> (2,2,2)
    for t in range(3):
        x = min(t, 4)
        y = min(t, 4)
        volume[t, y, x] = 30

    start = (0, 0, 0)
    end = (2, 2, 2)  # End is on the path and reachable with reach=1

    result = pathfinding_py.find_route_temporal(volume, "dijkstra", start, end)

    assert result is not None, "Route should be found"
    path, cost = result
    assert len(path) > 0, "Path should contain at least one point"
    # Check that path moves forward in time
    times = [pos[2] for pos in path]
    assert times == sorted(times), "Path should move forward in time"


def test_find_route_temporal_with_custom_starts_ends():
    """Test temporal routing with custom start and end positions."""
    volume = np.ones((4, 6, 6), dtype=np.uint8) * 80

    # Create a path from (1,1,0) to (4,4,3)
    for t in range(4):
        x = 1 + t
        y = 1 + t
        volume[t, y, x] = 15

    start = (1, 1, 0)
    end = (4, 4, 3)

    result = pathfinding_py.find_route_temporal(volume, "astar", start, end)

    assert result is not None, "Route should be found"
    path, cost = result
    assert len(path) > 0, "Path should contain at least one point"
    assert path[0] == start, "Path should start at the start position"
    assert path[-1] == end, "Path should end at the end position"


def test_find_route_temporal_with_reach():
    """Test temporal routing with custom reach parameter."""
    volume = np.ones((3, 8, 8), dtype=np.uint8) * 120

    # Create a path that requires reach > 1 and reaches the end
    for t in range(3):
        volume[t, t * 2, t * 2] = 25

    start = (0, 0, 0)
    end = (4, 4, 2)  # End is on the path (t=2, x=4, y=4)

    # Test with reach=2
    result = pathfinding_py.find_route_temporal(volume, "dijkstra", start, end, reach=2)

    assert result is not None, "Route should be found with reach=2"
    path, cost = result
    assert len(path) > 0, "Path should contain at least one point"


def test_find_route_temporal_invalid_algorithm():
    """Test that invalid algorithm raises an error."""
    volume = np.ones((3, 5, 5), dtype=np.uint8) * 50
    start = (0, 0, 0)
    end = (4, 4, 2)

    with pytest.raises(Exception):  # Should raise ValueError or similar
        pathfinding_py.find_route_temporal(volume, "invalid_algo", start, end)


def test_2d_pathfinding_on_real_image():
    """Test 2D pathfinding on the actual heatmap image (similar to justfile test command)."""
    image_path = ASSETS_DIR / "black-on-white-lv-like-heatmap.png"

    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")

    # Load image and convert to grayscale numpy array
    img = Image.open(image_path).convert("L")
    # Convert PIL image to numpy array
    # Note: PIL uses (width, height) but we need (height, width) for numpy
    array = np.array(img, dtype=np.uint8)

    # Verify array shape and dtype
    assert array.ndim == 2, "Image should be 2D"
    assert array.dtype == np.uint8, "Array should be uint8"

    # Use the same parameters as the justfile test command
    start = (269, 172)
    end = (470, 263)

    # Test with A* algorithm
    result = pathfinding_py.find_path_2d(array, start, end, "astar")

    assert result is not None, "Path should be found on real image"
    path, cost = result
    assert len(path) > 0, "Path should contain at least one point"
    assert path[0] == start, "Path should start at the start position"
    assert path[-1] == end, "Path should end at the end position"
    assert cost > 0, "Cost should be positive"

    # Verify path stays within image bounds
    height, width = array.shape
    for x, y in path:
        assert 0 <= x < width, f"Path point x={x} out of bounds [0, {width})"
        assert 0 <= y < height, f"Path point y={y} out of bounds [0, {height})"


def test_temporal_pathfinding_on_rotating_frames():
    """Test temporal pathfinding on rotating frame sequence (similar to justfile video command)."""
    frames_dir = ASSETS_DIR / "black-on-white-lv-like-heatmap-rotating"

    if not frames_dir.exists():
        pytest.skip(f"Frames directory not found: {frames_dir}")

    # Load all frame images
    frame_files = sorted(frames_dir.glob("frame_*.png"))

    if len(frame_files) == 0:
        pytest.skip(f"No frame images found in {frames_dir}")

    # Load first image to get dimensions
    first_img = Image.open(frame_files[0]).convert("L")
    width, height = first_img.size

    # Create 3D volume: (time, height, width)
    num_frames = len(frame_files)
    volume = np.zeros((num_frames, height, width), dtype=np.uint8)

    # Load all frames into the volume
    for t, frame_path in enumerate(frame_files):
        img = Image.open(frame_path).convert("L")
        # Convert PIL image to numpy array and ensure same dimensions
        frame_array = np.array(img, dtype=np.uint8)
        assert frame_array.shape == (height, width), (
            f"Frame {t} has wrong dimensions: {frame_array.shape} != ({height}, {width})"
        )
        volume[t] = frame_array

    start_pos = (269, 172)
    end_pos = (413, 260)
    reach = 2

    # Convert 2D start/end positions to 3D (add time dimension)
    # Start at time 0, end at last frame
    start = (start_pos[0], start_pos[1], 0)
    end = (end_pos[0], end_pos[1], num_frames - 1)

    # Test with A* algorithm
    result = pathfinding_py.find_route_temporal(
        volume,
        algorithm="astar",
        start=start,
        end=end,
        reach=reach,
        axis=2,  # Time axis
    )

    assert result is not None, "Route should be found through temporal volume"
    route, cost = result
    assert len(route) > 0, "Route should contain at least one point"

    # Verify route starts and ends at correct positions
    assert route[0] == start, f"Route should start at {start}, got {route[0]}"
    assert route[-1] == end, f"Route should end at {end}, got {route[-1]}"

    # Verify route moves forward in time
    times = [pos[2] for pos in route]
    assert times == sorted(times), "Route must move forward in time"
    assert times[0] == 0, "Route should start at time 0"
    assert times[-1] == num_frames - 1, f"Route should end at time {num_frames - 1}"

    # Verify all route points are within bounds
    for x, y, t in route:
        assert 0 <= x < width, f"Route point x={x} out of bounds [0, {width})"
        assert 0 <= y < height, f"Route point y={y} out of bounds [0, {height})"
        assert 0 <= t < num_frames, f"Route point t={t} out of bounds [0, {num_frames})"

    assert cost > 0, "Cost should be positive"
