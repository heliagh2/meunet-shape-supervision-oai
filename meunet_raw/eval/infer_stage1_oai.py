#!/usr/bin/env python

import os
import json
import argparse
from pathlib import Path
import sys

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

#ensure repo root is on PYTHONPATH so 'models' can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.meunet3d import MEUNet3D


def _strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def normalize_case_id(path_or_name: str) -> str:
    name = Path(path_or_name).name
    stem = _strip_nii_suffix(name)
    if stem.endswith("_0000"):
        stem = stem[:-5]
    if stem.endswith("_segmentation"):
        stem = stem[:-13]
    return stem


def load_config(cfg_path):
    cfg_path = Path(cfg_path)
    if cfg_path.suffix == ".json":
        return json.loads(cfg_path.read_text())
    else:
        import yaml
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)


def preprocess_volume(vol: np.ndarray):
    """
    Simple per-volume z-score normalization on non-zero voxels.
    Same as in infer_stage2_oai.py so the comparison is fair.
    """
    vol = vol.astype(np.float32)
    mask = vol > 0
    if mask.any():
        m = vol[mask].mean()
        s = vol[mask].std()
    else:
        m = vol.mean()
        s = vol.std()
    if s < 1e-8:
        s = 1.0
    vol = (vol - m) / s
    return vol


def load_or_create_test_split(cfg, images_dir: Path):
    splits_file = cfg.get("splits_file")
    if splits_file is None:
        return None

    splits_path = Path(splits_file)
    split_dir = splits_path.parent
    test_split_path = split_dir / "splits_test_final.json"

    img_paths = sorted(p for p in images_dir.glob("*.nii*"))
    image_ids = sorted({normalize_case_id(p.name) for p in img_paths})

    if test_split_path.exists():
        with open(test_split_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            test_ids = data.get("test") or data.get("test_ids") or data.get("test_keys")
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            split = data[int(cfg.get("split", 0))]
            test_ids = split.get("test") or split.get("test_ids") or split.get("test_keys")
        else:
            test_ids = data

        if test_ids is None:
            raise KeyError(f"Could not find test ids in {test_split_path}")

        test_ids = [normalize_case_id(str(case_id)) for case_id in test_ids]
        print(f"[INFO] Using existing test split from {test_split_path} ({len(test_ids)} ids)")
        return test_ids

    with open(splits_path, "r") as f:
        splits = json.load(f)

    split_idx = int(cfg.get("split", 0))
    if split_idx < 0 or split_idx >= len(splits):
        raise IndexError(
            f"split index {split_idx} out of range for splits_file with {len(splits)} entries"
        )

    split = splits[split_idx]
    train_ids = split.get("train") or split.get("train_ids") or split.get("train_keys") or []
    val_ids = split.get("val") or split.get("val_ids") or split.get("val_keys") or []
    trainval_ids = {normalize_case_id(str(case_id)) for case_id in list(train_ids) + list(val_ids)}

    test_ids = sorted(case_id for case_id in image_ids if case_id not in trainval_ids)

    with open(test_split_path, "w") as f:
        json.dump({"test": test_ids}, f, indent=2)

    print(
        f"[INFO] Created {test_split_path} with {len(test_ids)} ids "
        f"from {len(image_ids)} images not present in train/val split."
    )
    return test_ids


@torch.no_grad()
def predict_case(img_path, out_path, stage1, device, n_classes):
    img_nii = nib.load(str(img_path))
    img = img_nii.get_fdata()  # (Z,Y,X)
    img = preprocess_volume(img)

    # (1,1,D,H,W)
    img_t = torch.from_numpy(img[None, None, ...]).to(device)

    out = stage1(img_t, expanded=False)["logit1"]   # (1,C,D,H,W)
    pred = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int16)

    pred_nii = nib.Nifti1Image(pred, img_nii.affine, img_nii.header)
    nib.save(pred_nii, str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to oai_stage2.yaml (for arch + stage1_ckpt)")
    ap.add_argument("--stage1_ckpt", default=None, help="Override stage1 checkpoint path")
    ap.add_argument("--images_dir", required=True, help="Directory with test images (.nii.gz)")
    ap.add_argument("--output_dir", required=True, help="Where to save predicted segmentations")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    n_classes = cfg["n_classes"]
    ckpt_path = args.stage1_ckpt or cfg["stage1_ckpt"]

    #Stage-1 network
    print(f"[INFO] Loading stage-1 model from {ckpt_path}")
    stage1 = MEUNet3D(
        1,                         # in_channels for stage 1
        cfg["n_classes"],
        cfg["enc_channels"],
        cfg["dec_channels"],
        cfg["norm"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    stage1.load_state_dict(ckpt, strict=True)
    stage1.eval()

    #Inference over all test cases
    images_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)

    img_paths = sorted(p for p in images_dir.glob("*.nii*"))
    if not img_paths:
        raise RuntimeError(f"No NIfTI files found in {images_dir}")

    test_ids = load_or_create_test_split(cfg, images_dir)
    if test_ids is not None:
        test_set = set(test_ids)
        img_paths = [p for p in img_paths if normalize_case_id(p.name) in test_set]
        if not img_paths:
            raise RuntimeError(
                f"No test images left in {images_dir} after filtering by splits_test_final.json logic."
            )

    print(f"[INFO] Running stage-1 inference on {len(img_paths)} volumes from {images_dir}")
    for i, img_path in enumerate(img_paths, 1):
        # strip _0000 so filenames match labelsTs directly
        out_name = img_path.name.replace("_0000", "")
        out_path = out_dir / out_name
        print(f"[{i:03d}/{len(img_paths):03d}] {img_path.name} -> {out_name}")
        predict_case(img_path, out_path, stage1, device, n_classes)

    print("[INFO] Done stage-1 inference.")


if __name__ == "__main__":
    main()
