import os
import glob
import json
import argparse
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
from dataset.common import PuzzleDatasetMetadata

def load_images(input_dir: str, target_dir: str) -> List[Tuple[str, str, str]]:
    """Finds pairs of images from input and target directories."""
    # Pattern: X_input.png in input_dir matches X_output.png in target_dir
    inputs = glob.glob(os.path.join(input_dir, "*_input.png"))
    pairs = []

    print(f"Scanning {input_dir} for *_input.png files...")
    for inp in inputs:
        # Extract base name X from X_input.png
        filename = os.path.basename(inp)
        if not filename.endswith("_input.png"):
            continue

        base_name = filename[:-10] # remove _input.png
        target_filename = base_name + "_output.png"
        target_path = os.path.join(target_dir, target_filename)

        if os.path.exists(target_path):
            pairs.append((base_name, inp, target_path))
        else:
            print(f"Warning: No matching output file found at {target_path} for input {inp}")

    return pairs

def process_image(path: str, size: int, palette: Dict[Tuple[int, int, int], int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.Resampling.NEAREST)
    arr = np.array(img)

    # Map to indices
    # To speed up, we can flatten
    arr_flat = arr.reshape(-1, 3)
    indices_flat = np.zeros(arr_flat.shape[0], dtype=np.int32)

    for i in range(arr_flat.shape[0]):
        color = tuple(arr_flat[i])
        if color in palette:
            indices_flat[i] = palette[color]
        else:
            # Fallback: Find closest color
            min_dist = float('inf')
            closest_id = 0
            for p_color, p_id in palette.items():
                dist = sum(abs(c1 - c2) for c1, c2 in zip(color, p_color))
                if dist < min_dist:
                    min_dist = dist
                    closest_id = p_id
            indices_flat[i] = closest_id

    return indices_flat

def main():
    parser = argparse.ArgumentParser(description="Build image dataset for TRM")
    parser.add_argument("--input-dir", required=True, help="Directory containing input images")
    parser.add_argument("--target-dir", required=True, help="Directory containing target images")
    parser.add_argument("--output-dir", required=True, help="Directory to save processed dataset")
    parser.add_argument("--image-size", type=int, default=30, help="Resize images to this size")
    parser.add_argument("--test-split-ratio", type=float, default=0.1, help="Ratio of data to use for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    np.random.seed(args.seed)
    pairs = load_images(args.input_dir, args.target_dir)
    print(f"Found {len(pairs)} image pairs.")

    if len(pairs) == 0:
        print("No pairs found. Exiting.")
        return

    # Build Palette
    print("Building palette...")
    unique_colors = set()
    for _, inp, out in pairs:
        for p in [inp, out]:
            img = Image.open(p).convert("RGB")
            img = img.resize((args.image_size, args.image_size), Image.Resampling.NEAREST)
            data = np.array(img)
            data = data.reshape(-1, 3)
            for i in range(data.shape[0]):
                # Convert to standard python ints
                unique_colors.add(tuple(map(int, data[i])))

    sorted_colors = sorted(list(unique_colors))
    # Map colors to 1..N (0 is PAD/Background)
    palette = {c: i+1 for i, c in enumerate(sorted_colors)}

    print(f"Palette size: {len(palette)}")

    # Save Palette
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "palette.json"), "w") as f:
        # Convert keys to string for JSON
        json_palette = {str(k): v for k, v in palette.items()}
        json.dump(json_palette, f)

    # Split
    np.random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - args.test_split_ratio))
    splits = {
        "train": pairs[:split_idx],
        "test": pairs[split_idx:]
    }

    # Identifiers logic
    current_id = 1
    identifiers_list = ["<blank>"]

    for split_name, split_pairs in splits.items():
        if len(split_pairs) == 0:
            continue

        print(f"Processing split: {split_name} ({len(split_pairs)} examples)")
        os.makedirs(os.path.join(args.output_dir, split_name), exist_ok=True)

        inputs_list = []
        labels_list = []
        puzzle_identifiers = []
        puzzle_indices = [0]
        group_indices = [0]

        example_count = 0
        puzzle_count = 0

        for name, inp_path, out_path in split_pairs:
            inp_seq = process_image(inp_path, args.image_size, palette)
            out_seq = process_image(out_path, args.image_size, palette)

            inputs_list.append(inp_seq)
            labels_list.append(out_seq)

            example_count += 1
            puzzle_count += 1

            puzzle_indices.append(example_count)
            puzzle_identifiers.append(current_id)
            group_indices.append(puzzle_count)

            identifiers_list.append(name)
            current_id += 1

        # Convert to numpy
        # Determine dtype based on vocab size
        dtype = np.uint8
        if len(palette) + 1 > 255:
            dtype = np.int32

        inputs_arr = np.stack(inputs_list, 0).astype(dtype)
        labels_arr = np.stack(labels_list, 0).astype(dtype)

        identifiers_arr = np.array(puzzle_identifiers, dtype=np.int32)
        puzzle_indices_arr = np.array(puzzle_indices, dtype=np.int32)
        group_indices_arr = np.array(group_indices, dtype=np.int32)

        # Save .npy
        np.save(os.path.join(args.output_dir, split_name, f"all__inputs.npy"), inputs_arr)
        np.save(os.path.join(args.output_dir, split_name, f"all__labels.npy"), labels_arr)
        np.save(os.path.join(args.output_dir, split_name, f"all__puzzle_identifiers.npy"), identifiers_arr)
        np.save(os.path.join(args.output_dir, split_name, f"all__puzzle_indices.npy"), puzzle_indices_arr)
        np.save(os.path.join(args.output_dir, split_name, f"all__group_indices.npy"), group_indices_arr)

        # Metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=args.image_size * args.image_size,
            vocab_size=len(palette) + 1,
            pad_id=0,
            ignore_label_id=None,
            blank_identifier_id=0,
            num_puzzle_identifiers=len(identifiers_list),
            total_groups=len(split_pairs),
            mean_puzzle_examples=1.0,
            total_puzzles=len(split_pairs),
            sets=["all"]
        )

        with open(os.path.join(args.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f) # Use model_dump for Pydantic V2

    # Save all identifiers
    with open(os.path.join(args.output_dir, "identifiers.json"), "w") as f:
        json.dump(identifiers_list, f)

    print("Done.")

if __name__ == "__main__":
    main()
