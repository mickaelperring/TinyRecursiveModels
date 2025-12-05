import os
import random
import argparse
from PIL import Image, ImageDraw
import numpy as np

def random_color():
    return tuple(np.random.randint(0, 256, size=3, dtype=int))

def generate_pair(size, min_shape_size, max_shape_size):
    # Colors
    bg_color = random_color()
    while True:
        fg_color = random_color()
        # Ensure distinct enough
        if sum(abs(c1 - c2) for c1, c2 in zip(bg_color, fg_color)) > 50:
            break

    # Shape size
    shape_size = random.randint(min_shape_size, max_shape_size)

    # Position (Centered)
    # Top-left corner for square/bounding box
    x0 = (size - shape_size) // 2
    y0 = (size - shape_size) // 2
    x1 = x0 + shape_size
    y1 = y0 + shape_size

    # Input: Square
    img_in = Image.new("RGB", (size, size), bg_color)
    draw_in = ImageDraw.Draw(img_in)
    draw_in.rectangle([x0, y0, x1 - 1, y1 - 1], fill=fg_color, outline=None)

    # Output: Circle, Colors Swapped
    img_out = Image.new("RGB", (size, size), fg_color) # Swapped BG
    draw_out = ImageDraw.Draw(img_out)
    # ellipse takes bounding box
    draw_out.ellipse([x0, y0, x1 - 1, y1 - 1], fill=bg_color, outline=None) # Swapped FG

    return img_in, img_out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--min-shape-size", type=int, default=4)
    parser.add_argument("--max-shape-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dirs = {
        "train_input": os.path.join(args.output_dir, "train_input"),
        "train_target": os.path.join(args.output_dir, "train_target"),
        "test_input": os.path.join(args.output_dir, "test_input"),
        "test_target": os.path.join(args.output_dir, "test_target") # Optional, for verification
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Generate Train
    print(f"Generating {args.num_train} training pairs...")
    for i in range(args.num_train):
        img_in, img_out = generate_pair(args.image_size, args.min_shape_size, args.max_shape_size)
        img_in.save(os.path.join(dirs["train_input"], f"{i}_input.png"))
        img_out.save(os.path.join(dirs["train_target"], f"{i}_output.png"))

    # Generate Test
    print(f"Generating {args.num_test} testing pairs...")
    for i in range(args.num_test):
        img_in, img_out = generate_pair(args.image_size, args.min_shape_size, args.max_shape_size)
        # Use different ID range for test
        idx = i + args.num_train
        img_in.save(os.path.join(dirs["test_input"], f"{idx}_input.png"))
        img_out.save(os.path.join(dirs["test_target"], f"{idx}_output.png"))

    print("Done.")

if __name__ == "__main__":
    main()
