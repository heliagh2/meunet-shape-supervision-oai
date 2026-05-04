# meU-Net Stage-1 — OAI Knee MRI

Descriptor-Only Supervision (Volume + Centroid)

This repository contains a minimal implementation of a 3D Memory-Efficient U-Net (MEUNet) training pipeline for the OAI Knee MRI dataset (nnU-Net formatted).

The code supports:

* Stage-1 baseline training (Dice + Cross-Entropy)
* Stage-1 descriptor-only training (volume + centroid penalties)
* Inference on full test volumes
* Dice evaluation on the held-out test set

The descriptor-only setup removes segmentation loss gradients and supervises the network using only global shape descriptors.

---

## Repository Structure

```
meunet_raw/
  config/        YAML experiment configurations
  data/          Dataset loader (STD/EXP patch sampling)
  models/        MEUNet3D architecture
  losses/        Dice+CE and quadratic descriptor penalties
  train/         Training entrypoints
  eval/          Inference + evaluation scripts
  slurm/         Example Snellius job scripts
  logs/          left empty
  results/       left empty

nnunet/
  nnUNet_preprocessed/
    Dataset701_OAIKnee/
      splits_final.json
```

I have only committed code and configuration files; no dataset files, checkpoints, predictions, or logs.

---

## Requirements

Tested on:

* Python ≥ 3.10
* PyTorch 2.1.2
* CUDA 12.1
* Snellius (SURF) A100/H100 GPUs

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Requirements

This repository assumes the dataset follows the nnU-Net raw data structure.

Expected directory layout:

```
nnUNet_raw/
  Dataset701_OAIKnee/
    imagesTr/
      XXXX_R_0000.nii.gz
    labelsTr/
      XXXX_R.nii.gz
    imagesTs/
      XXXX_R_0000.nii.gz
    labelsTs/
      XXXX_R.nii.gz
```

Additionally, a precomputed split file is required, I have added mine:

```
nnUNet_preprocessed/
  Dataset701_OAIKnee/
    splits_final.json
```

### Important Notes

* The code assumes case IDs follow the format `XXXX_R`.
* `imagesTr` and `labelsTr` must match exactly by stem.
* `imagesTs` and `labelsTs` must match exactly by stem.
* No subject overlap should exist between train/val/test.

The repository does not include the dataset.

---

## MEUNet Architecture (Stage-1)

The Stage-1 model uses a 3D Memory-Efficient U-Net with two pathways:

* **STD patches** (standard resolution)
* **EXP patches** (expanded field of view, reduced gradient memory)

The EXP pathway provides larger spatial context without storing full first-layer gradients, allowing larger effective receptive fields.

---

## Training Modes

### 1. Baseline (Dice + CE)

File:

```
meunet_raw/train/train_stage1.py
```

Config:

```
meunet_raw/config/oai_stage1_raw.yaml
```

Run:

```bash
cd meunet_raw
python -u train/train_stage1.py \
  --config config/oai_stage1_raw.yaml
```
OR run the job file:
```
sbatch slurm/train_stage1_raw.sbatch
```

This trains using standard pixel-wise segmentation supervision.

---

### 2. Descriptor-Only Supervision (Volume + Centroid)

File:

```
meunet_raw/train/train_stage1_desc_only.py
```

Best configuration:

```
meunet_raw/config/oai_stage1_desc_only_all_both_ESP100_minSTD2_maxEXP_600ep.yaml
```

Run:

```bash
cd meunet_raw
python -u train/train_stage1_desc_only.py \
  --config config/oai_stage1_desc_only_all_both_ESP100_minSTD2_maxEXP_600ep.yaml
```
OR run the job file:
```
sbatch slurm/meu-s1-desc-only-all-both-ESP100-minSTD2-maxEXP-600ep.sbatch
```

In this setup:

* `use_seg_loss: false`
* Volume penalty (quadratic)
* Centroid penalty (quadratic)
* Dice/CE computed only for monitoring
* Shape losses applied on both STD and EXP patches

This does not yet match the “one value per supervision” principle.

---

## Inference (Stage-1)

Script:

```
meunet_raw/eval/infer_stage1_oai.py
```

Example:

```bash
cd meunet_raw

python eval/infer_stage1_oai.py \
  --config config/oai_stage1_<experiment_name>.yaml \
  --stage1_ckpt /path/to/checkpoint_best.pt \
  --images_dir /path/to/nnUNet_raw/Dataset701_OAIKnee/imagesTs \
  --output_dir results/<experiment_name>/test_preds \
  --device cuda
```
Job file:
```
sbatch slurm/infer_<experiment_name>.sbatch
```

Outputs:

* One `.nii.gz` segmentation per test case
* Full resolution (384×384×160 for OAI)

---

## Evaluation (Dice on Test Set)

Script:

```
meunet_raw/eval/eval_dice_oai.py
```

Run:

```bash
cd meunet_raw

python eval/eval_dice_oai.py \
  --gt_dir /path/to/nnUNet_raw/Dataset701_OAIKnee/labelsTs \
  --pred_dir /path/to/results/<experiment_name>/test_preds \
  --out_csv /path/to/results/<experiment_name>/test_dice.csv
```
Job file:
```
sbatch slurm/eval_<experiment_name>.sbatch
```

Outputs:

* CSV with per-case and per-class Dice scores

---

## Slurm (Snellius)

Example job scripts are provided in:

```
meunet_raw/slurm/
```

These include:

* Training (baseline and descriptor-only)
* Inference
* Evaluation

Before running:

* Update paths inside `.sbatch` files
* Ensure correct partition (e.g., `gpu_a100`, `gpu_h100`, or CPU partition)
* Update appropriate virtual environment

---

## Important Configuration Notes

All YAML configs currently contain absolute paths.

Before running on your system:

* Update:

  * `images_dir`
  * `labels_dir`
  * `splits_file`
  * `workdir`
* Ensure checkpoint paths in inference match your training run.

---

## Reproducibility Notes

* Seed is set in config (`seed: 777`)
* I have conducted one of the experiments on 4 other seeds (111, 222, 333, 444), results were stable across seeds.
* Early stopping is based on validation Dice
* Mixed precision (`amp: true`) is enabled
* Patch sampling alternates STD and EXP
* Descriptor losses are quadratic penalties

For large patch sizes (e.g. expanded to 400³), H100 GPU is recommended. For patches below 300³, A100 is fine. 

---

## What is not included

This repository does **not** contain:

* OAI dataset
* Trained checkpoints
* Prediction files
* Log files

