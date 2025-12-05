import os
import json
import torch
import numpy as np
from PIL import Image
from dataset.build_image_dataset import process_image
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
from dataset.common import PuzzleDatasetMetadata
import glob
import argparse
import ast

def load_palette(path):
    with open(path, "r") as f:
        p = json.load(f)
    # Convert keys back to tuples safely
    return {ast.literal_eval(k): v for k, v in p.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Path to checkpoint directory (containing all_config.yaml and step_X)")
    parser.add_argument("--step", type=str, required=True, help="Step number to load (e.g. 10000)")
    parser.add_argument("--input-dir", required=True, help="Directory with new images")
    parser.add_argument("--output-dir", required=True, help="Directory to save predictions")
    parser.add_argument("--palette-path", required=True, help="Path to palette.json generated during build")
    parser.add_argument("--image-size", type=int, default=30)
    args = parser.parse_args()

    # Load Palette
    palette = load_palette(args.palette_path)
    inv_palette = {v: k for k, v in palette.items()}
    inv_palette[0] = (0, 0, 0) # PAD is black

    # Load Config
    config_path = os.path.join(args.model_dir, "all_config.yaml")
    with open(config_path, "r") as f:
        import yaml
        full_conf = yaml.safe_load(f)

    # Attempt to locate metadata for vocab_size and seq_len
    # Try looking in the data path defined in the config
    data_path = full_conf['data_paths'][0]

    # Check if data_path is absolute or relative. If relative, try to resolve from current dir.
    if not os.path.exists(data_path):
        # Try relative to the config file location? Usually data paths are from repo root.
        pass

    metadata_path = os.path.join(data_path, "train", "dataset.json")
    if not os.path.exists(metadata_path):
         # Try without train subdir if it's a flat structure or different split
         metadata_path = os.path.join(data_path, "dataset.json")

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            meta = json.load(f)
            vocab_size = meta['vocab_size']
            seq_len = meta['seq_len']
            num_puzzle_identifiers = meta['num_puzzle_identifiers']
    else:
        print(f"Error: Could not find dataset metadata at {metadata_path}. Please ensure data path is accessible.")
        return

    arch_params = full_conf['arch']
    model_cfg_dict = arch_params.copy()
    # Remove keys not expected by TRM config if any (though pydantic allows extra)
    # Ensure name and loss are not passed if they cause issues, though TRM config structure seems flat in pydantic model
    # TinyRecursiveReasoningModel_ACTV1Config inherits from BaseModel.
    # We remove 'name' and 'loss' as they are used by pretrain.py factory, not the model config itself.
    model_cfg_dict.pop('name', None)
    model_cfg_dict.pop('loss', None)

    model_cfg_dict.update({
        "batch_size": 1,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "num_puzzle_identifiers": num_puzzle_identifiers,
        "causal": False
    })

    config = TinyRecursiveReasoningModel_ACTV1Config(**model_cfg_dict)
    model = TinyRecursiveReasoningModel_ACTV1(config.model_dump())

    # Load Weights
    checkpoint_file = os.path.join(args.model_dir, f"step_{args.step}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(checkpoint_file, map_location=device)

    # Clean keys
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("_orig_mod."):
            new_k = new_k[len("_orig_mod."):]
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]

        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    model.to(device)

    # Process Images
    images = glob.glob(os.path.join(args.input_dir, "*.png"))
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Found {len(images)} images to process.")

    for img_path in images:
        # Check if it is an output image (skip)
        if "_output.png" in img_path:
            continue

        print(f"Processing {img_path}...")

        puzzle_id = torch.tensor([0], dtype=torch.int32).to(device)

        inp_seq = process_image(img_path, args.image_size, palette)
        inp_tensor = torch.from_numpy(inp_seq).unsqueeze(0).to(device) # (1, SeqLen)

        batch = {
            "inputs": inp_tensor,
            "puzzle_identifiers": puzzle_id
        }

        with torch.no_grad():
            carry = model.initial_carry(batch)

            while True:
                carry, outputs = model(carry, batch)
                if carry.halted.all():
                    break

            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)

        pred_seq = preds[0].cpu().numpy()

        # Reconstruct Image
        h = w = args.image_size
        out_img = Image.new("RGB", (w, h))
        pixels = out_img.load()

        for i in range(len(pred_seq)):
            r = i // w
            c = i % w
            color_id = pred_seq[i]
            if color_id in inv_palette:
                pixels[c, r] = inv_palette[color_id]
            else:
                pixels[c, r] = (0, 0, 0)

        save_path = os.path.join(args.output_dir, os.path.basename(img_path).replace(".png", "_pred.png"))
        out_img.save(save_path)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
