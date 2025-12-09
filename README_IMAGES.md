# Image Training and Inference with TRM

This guide explains how to train the Tiny Recursion Model (TRM) on image-to-image tasks (e.g., transforming shapes) and run inference on new images.

## 1. Data Preparation

### Option A: Generate Synthetic Data (Squares to Circles)
To test the pipeline, you can generate a synthetic dataset where the task is to transform a square into a circle with swapped colors.

```bash
python -m dataset/generate_synthetic_shapes \
  --output-dir data/shapes_raw \
  --num-train 1000 \
  --num-test 100 \
  --image-size 30
```
This will create `train_input`, `train_target`, and `test_input` folders in `data/shapes_raw`.

### Option B: Use Your Own Images
Organize your images into two folders: one for inputs and one for targets (ground truth outputs).
*   Input images must end with `_input.png`.
*   Target images must end with `_output.png`.
*   The matching pair must share the same base name (e.g., `img1_input.png` matches `img1_output.png`).

## 2. Build the Dataset
Convert your images into the tokenized format required by TRM.

```bash
# Example for the synthetic shapes dataset
python -m dataset/build_image_dataset \
  --input-dir data/shapes_raw/train_input \
  --target-dir data/shapes_raw/train_target \
  --output-dir data/shapes_dataset \
  --image-size 30
```
*   `--input-dir`: Directory containing `*_input.png` images.
*   `--target-dir`: Directory containing `*_output.png` images.
*   `--output-dir`: Where to save the processed dataset.
*   `--image-size`: Resize images to this dimension (e.g., 30x30).

This will generate `.npy` files and a `palette.json` in `data/shapes_dataset`.

## 3. Train the Model

Launch the training using the provided configuration file (`config/cfg_shapes.yaml`). You can modify this file to adjust hyperparameters like `epochs` or `hidden_size`.

```bash
python -m pretrain --config-name cfg_shapes
```
*   Checkpoints will be saved in `checkpoints/shapes_project/shapes_run`.

## 4. Run Inference
To run the trained model on new images (which do not need to have matching targets):

```bash
python -m inference_image \
  --model-dir checkpoints/shapes_project/shapes_run \
  --step 225 \
  --input-dir data/shapes_raw/test_input \
  --output-dir shapes_inference_output \
  --palette-path data/shapes_dataset/palette.json \
  --image-size 30
```

*   `--model-dir`: Directory containing the checkpoint (e.g., `checkpoints/shapes_project/shapes_run`).
*   `--step`: The step number of the checkpoint to load (e.g., `225` or `latest`).
*   `--input-dir`: Directory containing new images (must end in `.png`).
*   `--output-dir`: Where to save the generated output images.
*   `--palette-path`: Path to the `palette.json` generated in step 2.

The predicted images will be saved in `shapes_inference_output` with the suffix `_pred.png`.
