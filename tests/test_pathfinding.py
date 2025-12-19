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


PROJECT_ROOT = pathlib.Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"


def test_find_path_2d_astar():
    """Test 2D pathfinding with A* algorithm."""

    array = np.ones((10, 10), dtype=np.uint8) * 200

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
    array = np.ones((5, 5), dtype=np.uint8) * 50

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
    array = np.ones((8, 8), dtype=np.uint8) * 100

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


def test_find_path_2d_invalid_start_out_of_bounds():
    """Test that out-of-bounds start position raises an error."""
    array = np.ones((5, 5), dtype=np.uint8) * 50

    with pytest.raises(Exception):  # Should raise ValueError
        pathfinding_py.find_path_2d(array, (10, 0), (4, 4), "astar")


def test_find_path_2d_invalid_end_out_of_bounds():
    """Test that out-of-bounds end position raises an error."""
    array = np.ones((5, 5), dtype=np.uint8) * 50

    with pytest.raises(Exception):  # Should raise ValueError
        pathfinding_py.find_path_2d(array, (0, 0), (10, 4), "astar")


def test_find_path_2d_invalid_both_out_of_bounds():
    """Test that out-of-bounds start and end positions raise an error."""
    array = np.ones((5, 5), dtype=np.uint8) * 50

    with pytest.raises(Exception):  # Should raise ValueError
        pathfinding_py.find_path_2d(array, (10, 10), (20, 20), "astar")


def test_find_route_temporal_astar():
    """Test temporal routing with A* algorithm."""
    volume = np.ones((10, 10, 5), dtype=np.uint8) * 150

    for t in range(5):
        x = min(t, 9)
        y = min(t, 9)
        volume[x, y, t] = 20

    start = (0, 0, 0)
    end = (4, 4, 4)

    result = pathfinding_py.find_route_temporal(volume, "astar", start, end)

    assert result is not None, "Route should be found"
    path, cost = result
    assert len(path) > 0, "Path should contain at least one point"
    times = [pos[2] for pos in path]
    assert times == sorted(times), "Path should move forward in time"
    assert cost > 0, "Cost should be positive"


def test_find_route_temporal_dijkstra():
    """Test temporal routing with Dijkstra algorithm."""
    volume = np.ones((5, 5, 3), dtype=np.uint8) * 100

    for t in range(3):
        x = min(t, 4)
        y = min(t, 4)
        volume[x, y, t] = 30

    start = (0, 0, 0)
    end = (2, 2, 2)

    result = pathfinding_py.find_route_temporal(volume, "dijkstra", start, end)

    assert result is not None, "Route should be found"
    path, cost = result
    assert len(path) > 0, "Path should contain at least one point"
    times = [pos[2] for pos in path]
    assert times == sorted(times), "Path should move forward in time"


def test_find_route_temporal_with_custom_starts_ends():
    """Test temporal routing with custom start and end positions."""
    volume = np.ones((6, 6, 4), dtype=np.uint8) * 80

    for t in range(4):
        x = 1 + t
        y = 1 + t
        volume[x, y, t] = 15

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
    volume = np.ones((8, 8, 3), dtype=np.uint8) * 120

    for t in range(3):
        x = t * 2
        y = t * 2
        volume[x, y, t] = 25

    start = (0, 0, 0)
    end = (4, 4, 2)

    # Test with reach=2
    result = pathfinding_py.find_route_temporal(volume, "dijkstra", start, end, reach=2)

    assert result is not None, "Route should be found with reach=2"
    path, cost = result
    assert len(path) > 0, "Path should contain at least one point"


def test_find_route_temporal_invalid_algorithm():
    """Test that invalid algorithm raises an error."""
    volume = np.ones((5, 5, 3), dtype=np.uint8) * 50
    start = (0, 0, 0)
    end = (4, 4, 2)

    with pytest.raises(Exception):  # Should raise ValueError or similar
        pathfinding_py.find_route_temporal(volume, "invalid_algo", start, end)


def test_find_route_temporal_invalid_start_x_out_of_bounds():
    """Test that out-of-bounds start x position raises an error."""
    volume = np.ones((5, 5, 3), dtype=np.uint8) * 50

    with pytest.raises(Exception):  # Should raise ValueError
        pathfinding_py.find_route_temporal(volume, "astar", (10, 0, 0), (4, 4, 2))


def test_find_route_temporal_invalid_start_y_out_of_bounds():
    """Test that out-of-bounds start y position raises an error."""
    volume = np.ones((5, 5, 3), dtype=np.uint8) * 50

    with pytest.raises(Exception):  # Should raise ValueError
        pathfinding_py.find_route_temporal(volume, "astar", (0, 10, 0), (4, 4, 2))


def test_find_route_temporal_invalid_start_t_out_of_bounds():
    """Test that out-of-bounds start t position raises an error."""
    volume = np.ones((5, 5, 3), dtype=np.uint8) * 50

    with pytest.raises(Exception):  # Should raise ValueError
        pathfinding_py.find_route_temporal(volume, "astar", (0, 0, 10), (4, 4, 2))


def test_find_route_temporal_invalid_end_x_out_of_bounds():
    """Test that out-of-bounds end x position raises an error."""
    volume = np.ones((5, 5, 3), dtype=np.uint8) * 50

    with pytest.raises(Exception):  # Should raise ValueError
        pathfinding_py.find_route_temporal(volume, "astar", (0, 0, 0), (10, 4, 2))


def test_find_route_temporal_invalid_end_y_out_of_bounds():
    """Test that out-of-bounds end y position raises an error."""
    volume = np.ones((5, 5, 3), dtype=np.uint8) * 50

    with pytest.raises(Exception):  # Should raise ValueError
        pathfinding_py.find_route_temporal(volume, "astar", (0, 0, 0), (4, 10, 2))


def test_find_route_temporal_invalid_end_t_out_of_bounds():
    """Test that out-of-bounds end t position raises an error."""
    volume = np.ones((5, 5, 3), dtype=np.uint8) * 50

    with pytest.raises(Exception):  # Should raise ValueError
        pathfinding_py.find_route_temporal(volume, "astar", (0, 0, 0), (4, 4, 10))


def test_find_route_temporal_invalid_both_out_of_bounds():
    """Test that out-of-bounds start and end positions raise an error."""
    volume = np.ones((5, 5, 3), dtype=np.uint8) * 50

    with pytest.raises(Exception):  # Should raise ValueError
        pathfinding_py.find_route_temporal(volume, "astar", (10, 10, 10), (20, 20, 20))


def test_2d_pathfinding_on_real_image():
    """Test 2D pathfinding on the actual heatmap image (similar to justfile test command)."""
    image_path = ASSETS_DIR / "black-on-white-lv-like-heatmap.png"

    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")

    img = Image.open(image_path).convert("L")
    array = np.array(img, dtype=np.uint8)

    assert array.ndim == 2, "Image should be 2D"
    assert array.dtype == np.uint8, "Array should be uint8"

    start = (269, 172)
    end = (470, 263)

    result = pathfinding_py.find_path_2d(array, start, end, "astar")

    assert result is not None, "Path should be found on real image"
    path, cost = result
    assert len(path) > 0, "Path should contain at least one point"
    assert path[0] == start, "Path should start at the start position"
    assert path[-1] == end, "Path should end at the end position"
    assert cost > 0, "Cost should be positive"

    height, width = array.shape
    for x, y in path:
        assert 0 <= x < width, f"Path point x={x} out of bounds [0, {width})"
        assert 0 <= y < height, f"Path point y={y} out of bounds [0, {height})"


def test_temporal_pathfinding_on_rotating_frames():
    """Test temporal pathfinding on rotating frame sequence (similar to justfile video command)."""
    frames_dir = ASSETS_DIR / "black-on-white-lv-like-heatmap-rotating"

    if not frames_dir.exists():
        pytest.skip(f"Frames directory not found: {frames_dir}")

    frame_files = sorted(frames_dir.glob("frame_*.png"))

    if len(frame_files) == 0:
        pytest.skip(f"No frame images found in {frames_dir}")

    first_img = Image.open(frame_files[0]).convert("L")
    width, height = first_img.size

    num_frames = len(frame_files)
    volume = np.zeros((width, height, num_frames), dtype=np.uint8)

    for t, frame_path in enumerate(frame_files):
        img = Image.open(frame_path).convert("L")
        frame_array = np.array(img, dtype=np.uint8)
        assert frame_array.shape == (height, width), (
            f"Frame {t} has wrong dimensions: {frame_array.shape} != ({width}, {height})"
        )
        volume[:, :, t] = frame_array

    start_pos = (269, 172)
    end_pos = (413, 260)
    reach = 2

    start = (start_pos[0], start_pos[1], 0)
    end = (end_pos[0], end_pos[1], num_frames - 1)

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

    assert route[0] == start, f"Route should start at {start}, got {route[0]}"
    assert route[-1] == end, f"Route should end at {end}, got {route[-1]}"

    times = [pos[2] for pos in route]
    assert times == sorted(times), "Route must move forward in time"
    assert times[0] == 0, "Route should start at time 0"
    assert times[-1] == num_frames - 1, f"Route should end at time {num_frames - 1}"

    for x, y, t in route:
        assert 0 <= x < width, f"Route point x={x} out of bounds [0, {width})"
        assert 0 <= y < height, f"Route point y={y} out of bounds [0, {height})"
        assert 0 <= t < num_frames, f"Route point t={t} out of bounds [0, {num_frames})"

    assert cost > 0, "Cost should be positive"
