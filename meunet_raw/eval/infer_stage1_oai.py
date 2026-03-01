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
