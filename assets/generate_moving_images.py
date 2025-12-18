# /// script
# dependencies = [
#   "numpy",
#   "pillow",
#   "scipy",
# ]
# ///

import sys
import os
import argparse
import numpy as np
from PIL import Image
from scipy import ndimage

def generate_frames(input_path, output_folder, num_frames=120):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        img = Image.open(input_path).convert('L') # Convert to grayscale as requested "black and white"
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    img_arr = np.array(img)
    h, w = img_arr.shape

    # Coordinate grid
    y, x = np.mgrid[0:h, 0:w]
    
    # Center of rotation
    cy, cx = h / 2.0, w / 2.0

    print(f"Generating {num_frames} frames...")

    for t in range(num_frames):
        # Time normalized 0 to 1 (or slightly more to show movement)
        # Transformations are small between frames.
        
        # 1. Translation: Move towards right.
        # Target pixel (x, y) comes from Source (x - tv, y)
        trans_offset_x = 0.5 * t  # Move 0.5 pixel per frame right
        
        # 2. Rotation: Rotate CCW.
        # To find value at target, we rotate coordinates CW.
        angle_deg = 0.5 * t # 0.5 degree per frame
        angle_rad = np.deg2rad(angle_deg)
        
        # 3. Deformation: Non-linear.
        # Let's add a sine wave ripple that moves.
        # deform(x, y) -> adds offset to sampling coordinates
        freq = 0.05
        phase = 0.2 * t
        deform_amp = 2.0 * np.sin(t * 0.05)
        
        # We need to construct the source coordinates for each target pixel (y, x)
        
        # Start with identity coordinates
        coords_y = y - cy
        coords_x = x - cx
        
        # Apply Inverse Rotation (CW)
        # x_rot = x * cos(theta) + y * sin(theta)
        # y_rot = -x * sin(theta) + y * cos(theta)
        # (For CW rotation of the COORDINATE SYSTEM/sampling point, which corresponds to CCW object rotation)
        # Wait, strictly:
        # P_target = R * P_source
        # P_source = R_inv * P_target
        # R_inv for CCW(theta) is Rotation(-theta) = Rotation_CW(theta)
        # x_src = x_tgt * cos(-theta) - y_tgt * sin(-theta) = x_tgt * cos(theta) + y_tgt * sin(theta)
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        rot_x = coords_x * cos_a + coords_y * sin_a
        rot_y = -coords_x * sin_a + coords_y * cos_a
        
        # Restore center
        src_x = rot_x + cx
        src_y = rot_y + cy
        
        # Apply Inverse Translation (Move right -> sample from left)
        src_x = src_x - trans_offset_x
        
        # Apply Inverse Deformation (or just add distortion to source coords)
        # A simple ripple based on Y in the source domain
        src_x = src_x + deform_amp * np.sin(src_y * freq + phase)
        src_y = src_y + deform_amp * np.cos(src_x * freq + phase)

        # Map coordinates
        # map_coordinates expects (row_coords, col_coords) i.e. (y, x)
        # mode='nearest' or 'constant' or 'reflect'. 'nearest' creates a streak effect at edges, 'constant' fills bg.
        # Given it's a "black and white image ... heatmap", constant 0 (black) or 255 (white) might be good.
        # Let's assume white background if the image is black on white.
        # Check image content? The name is "black-on-white". So background is white (255).
        
        new_img = ndimage.map_coordinates(img_arr, [src_y, src_x], order=3, mode='constant', cval=255.0)
        
        frame_filename = os.path.join(output_folder, f"frame_{t:03d}.png")
        Image.fromarray(new_img.astype('uint8')).save(frame_filename)
        
        if t % 20 == 0:
            print(f"Saved {frame_filename}")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate moving images from a single input image.")
    parser.add_argument("input_image", help="Path to source image")
    parser.add_argument("output_dir", help="Directory to save output frames")
    
    args = parser.parse_args()
    
    generate_frames(args.input_image, args.output_dir)
