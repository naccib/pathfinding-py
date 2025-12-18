# Benchmark the pathfinding algorithms
bench:
    cd image_pathfinding && cargo bench

# Run temporal pathfinding on rotating frames and create a video
video:
    cargo run --release -p pathfinding_cli -- --start 269 172 --end 413 260 --algo astar --reach 2 assets/black-on-white-lv-like-heatmap-rotating/*.png
    ffmpeg -framerate 30 -i '/tmp/frame_%03d.png' -c:v libx264 -pix_fmt yuv420p -crf 23 /tmp/pathfinding_output.mp4

# Run 2D pathfinding on a single image
test:
    cargo run --release -p pathfinding_cli -- --start 269 172 --end 470 263 --algo astar --output-dir output_test assets/black-on-white-lv-like-heatmap.png

# Run Python integration tests for pathfinding_py
test-python:
    # Ensure tests venv exists
    cd tests && uv sync
    # Build and install pathfinding_py module into tests venv
    # Set VIRTUAL_ENV so maturin uses the tests venv
    cd pathfinding_py && VIRTUAL_ENV=../tests/.venv uvx maturin develop --release
    # Run tests using uv
    cd tests && uv run pytest -v
