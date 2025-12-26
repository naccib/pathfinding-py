# Benchmark the pathfinding algorithms
bench:
    cd image_pathfinding && cargo bench

# Run temporal pathfinding on rotating frames and create a video
video:
    cargo run --release -p pathfinding_cli -- --start 269 172 --end 413 260 --algo astar --reach 2 assets/black-on-white-lv-like-heatmap-rotating/*.png
    ffmpeg -framerate 30 -i '/tmp/frame_%03d.png' -c:v libx264 -pix_fmt yuv420p -crf 23 /tmp/pathfinding_output.mp4

# Run 2D pathfinding on a single image
test:
    rm -rf /tmp/pathfinding-test/
    mkdir -p /tmp/pathfinding-test/

    cargo run -p pathfinding_cli -- --start 269 172 --end 470 263 --algo astar --output-dir /tmp/pathfinding-test assets/black-on-white-lv-like-heatmap.png
    
    cargo run -p pathfinding_cli -- \
        --start 186 85 --end 312 97 --impassable 18 --algo astar --output-dir /tmp/pathfinding-test \
        --filename "lv-with-impassable-barrier-and-impassable-equals-18.png" \
        assets/lv-with-impassable-barrier.png

    cargo run -p pathfinding_cli -- \
        --start 186 85 --end 312 97 --algo astar --output-dir /tmp/pathfinding-test \
        --filename "lv-with-impassable-barrier-no-impassable-value.png" \
        assets/lv-with-impassable-barrier.png

    open /tmp/pathfinding-test/ -a Finder


# Run Python integration tests for pathfinding_py
test-python:
    # Ensure tests venv exists
    cd tests && uv sync
    # Build and install pathfinding_py module into tests venv
    # Set VIRTUAL_ENV so maturin uses the tests venv
    cd pathfinding_py && VIRTUAL_ENV=../tests/.venv uvx maturin develop --release
    # Run tests using uv
    cd tests && uv run pytest -v
