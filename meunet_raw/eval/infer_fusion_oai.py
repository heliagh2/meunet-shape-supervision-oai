#!/usr/bin/env python
"""
Fuse the two MEUNet heads at inference.

The network runs ONCE per case; every fusion rule is computed from the same pair
of native-resolution probability maps and written to its own subdir of
--output_dir, so a single eval_dice_oai.py call compares all of them:

  std       logit1 alone (reference)
  exp       logit2 alone (reference)
  avg       mean of the two probability maps
  conf      voxel-wise: take the label of the more confident head (max softmax).
            Where the heads agree this changes nothing.
  conf_fov  conf inside the region the STD head was trained on (the fixed crop
            around fixed_center), EXP outside it
  poe       product of experts: argmax(log p_std + log p_exp). A label survives
            only where NEITHER head rules it out -- EXP vetoes STD's false
            positives far from a structure, STD sharpens EXP's coarse boundary.
  band      EXP decides the labels; STD may only relabel voxels within r of an
            EXP boundary, r = one EXP output voxel per axis (derived from the
            resampling, not tuned). Mass/topology from EXP, boundary from STD.
  wavg      per-class weighted mean, weights from --exp_weights (tune on --split
            val, never on test). Skipped unless --exp_weights is given.

Caveat for conf/conf_fov: the concentration term pushes both heads towards
confident outputs regardless of correctness, and EXP's trilinearly upsampled
logits are smoother at boundaries by construction -- so "more confident" is a
biased referee. fusion_stats.csv records how often each head wins.
"""

import os
import csv
import json
import math
import argparse
from pathlib import Path
import sys

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.meunet3d import MEUNet3D
from data.dataset_oai_raw_fixedpatch import _compute_patch_bounds
from eval.infer_stage1_oai import (
    load_config, preprocess_volume, load_or_create_test_split, normalize_case_id,
    exp_input_size, head_logits,
)

VARIANTS = ("std", "exp", "avg", "conf", "conf_fov", "poe", "band", "wavg")


def load_val_ids(cfg):
    with open(cfg["splits_file"], "r") as f:
        split = json.load(f)[int(cfg.get("split", 0))]
    val_ids = split.get("val") or split.get("val_ids") or split.get("val_keys") or []
    return [normalize_case_id(str(c)) for c in val_ids]


def resolve_fixed_center(cfg, ckpt_path, override):
    """The STD crop centre training used; None if it was per-case (GT-derived)."""
    p = Path(override) if override else Path(ckpt_path).parent / "fixed_center.json"
    if p.exists():
        return tuple(int(v) for v in json.loads(p.read_text())["fixed_center"])
    fc = cfg.get("fixed_center", None)
    if isinstance(fc, (list, tuple)):
        return tuple(int(round(float(v))) for v in fc)
    return None


def std_fov_mask(native, center, patch_size, device):
    """Boolean (D,H,W) mask of the STD training crop, as _sample_fixed_patch_pair cuts it."""
    m = torch.zeros(native, dtype=torch.bool, device=device)
    (z0, z1), (y0, y1), (x0, x1) = (_compute_patch_bounds(c, p, d)
                                    for c, p, d in zip(center, patch_size, native))
    m[z0:z1, y0:y1, x0:x1] = True
    return m


def band_radius(native, patch_size, expand_factor):
    """One EXP output voxel (logit2 = half the EXP input) in native voxels, per axis."""
    exp_out = [s // 2 for s in exp_input_size(native, patch_size, expand_factor)]
    return tuple(int(math.ceil(d / o)) for d, o in zip(native, exp_out))


def dilate_labels(lbl, n_classes, r):
    """(C,D,H,W) bool: voxels within r (per axis, box) of each class in lbl."""
    oh = F.one_hot(lbl.long(), n_classes).permute(3, 0, 1, 2)[None].half()
    k = tuple(2 * v + 1 for v in r)
    return F.max_pool3d(oh, kernel_size=k, stride=1, padding=r)[0].bool()


@torch.no_grad()
def fuse(logits, n_classes, fov=None, r=None, exp_w=None):
    p = {h: F.softmax(logits[h], dim=1)[0] for h in ("std", "exp")}  # (C,D,H,W)
    conf = {h: p[h].max(dim=0) for h in p}                              # (values, labels)
    lbl = {h: conf[h].indices for h in p}

    out = {"std": lbl["std"], "exp": lbl["exp"]}
    out["avg"] = (p["std"] + p["exp"]).argmax(dim=0)

    std_wins = conf["std"].values >= conf["exp"].values
    out["conf"] = torch.where(std_wins, lbl["std"], lbl["exp"])
    if fov is not None:
        out["conf_fov"] = torch.where(fov, out["conf"], lbl["exp"])

    logp = {h: F.log_softmax(logits[h], dim=1)[0] for h in p}
    out["poe"] = (logp["std"] + logp["exp"]).argmax(dim=0)

    if r is not None:
        allowed = dilate_labels(lbl["exp"], n_classes, r)                # (C,D,H,W)
        std_ok = allowed.gather(0, lbl["std"][None])[0]
        out["band"] = torch.where(std_ok, lbl["std"], lbl["exp"])

    if exp_w is not None:
        w = torch.as_tensor(exp_w, dtype=p["std"].dtype, device=p["std"].device).view(-1, 1, 1, 1)
        out["wavg"] = ((1 - w) * p["std"] + w * p["exp"]).argmax(dim=0)

    # diagnostics: where do the heads disagree, and who wins there under conf
    dis = lbl["std"] != lbl["exp"]
    fg = (lbl["std"] > 0) | (lbl["exp"] > 0)
    n_dis = int(dis.sum())
    stats = {
        "disagree_frac_fg": float(dis.sum()) / max(1, int(fg.sum())),
        "conf_std_wins_in_disagreement": float((dis & std_wins).sum()) / max(1, n_dis),
        "mean_conf_std_fg": float(conf["std"].values[fg].mean()) if fg.any() else float("nan"),
        "mean_conf_exp_fg": float(conf["exp"].values[fg].mean()) if fg.any() else float("nan"),
    }
    if fov is not None:
        stats["disagree_frac_outside_fov"] = float((dis & ~fov).sum()) / max(1, n_dis)
    if "band" in out:
        stats["band_std_kept_frac_in_disagreement"] = float((dis & (out["band"] == lbl["std"])).sum()) / max(1, n_dis)
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage1_ckpt", default=None)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split", choices=("test", "val"), default="test",
                    help="val: the training run's val cases -- use it to choose fusion/weights")
    ap.add_argument("--fixed_center_json", default=None,
                    help="default: <ckpt dir>/fixed_center.json (written by training)")
    ap.add_argument("--exp_weights", nargs="+", type=float, default=None,
                    help="per-class weight on EXP for 'wavg', one per class incl. background")
    ap.add_argument("--band_scale", type=float, default=1.0,
                    help="multiplies the derived band radius (1 = one EXP output voxel)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_classes = int(cfg["n_classes"])
    patch_size = tuple(int(v) for v in cfg["patch_size"])
    expand_factor = float(cfg.get("expand_factor", 1.25))
    ckpt_path = args.stage1_ckpt or cfg["stage1_ckpt"]

    if args.exp_weights is not None and len(args.exp_weights) != n_classes:
        raise ValueError(f"--exp_weights needs {n_classes} values, got {len(args.exp_weights)}")

    center = resolve_fixed_center(cfg, ckpt_path, args.fixed_center_json)
    if center is None:
        print("[WARN] no fixed_center.json and no fixed centre in config: the STD crop was "
              "per-case (GT-derived), so conf_fov is skipped.")
    else:
        print(f"[INFO] STD training crop centre: {center}")

    print(f"[INFO] Loading stage-1 model from {ckpt_path}")
    stage1 = MEUNet3D(1, n_classes, cfg["enc_channels"], cfg["dec_channels"], cfg["norm"]).to(device)
    stage1.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    stage1.eval()

    images_dir, out_dir = Path(args.images_dir), Path(args.output_dir)
    img_paths = sorted(images_dir.glob("*.nii*"))
    ids = load_val_ids(cfg) if args.split == "val" else load_or_create_test_split(cfg, images_dir)
    if ids is not None:
        keep = set(ids)
        img_paths = [p for p in img_paths if normalize_case_id(p.name) in keep]
    if not img_paths:
        raise RuntimeError(f"No {args.split} images found in {images_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "fusion_settings.json", "w") as f:
        json.dump({"ckpt": str(ckpt_path), "split": args.split, "fixed_center": center,
                   "patch_size": patch_size, "expand_factor": expand_factor,
                   "exp_weights": args.exp_weights, "band_scale": args.band_scale}, f, indent=2)

    stats_rows = []
    print(f"[INFO] Fusing both heads on {len(img_paths)} {args.split} volumes")
    for i, img_path in enumerate(img_paths, 1):
        img_nii = nib.load(str(img_path))
        img = preprocess_volume(img_nii.get_fdata())
        native = img.shape

        logits = head_logits(stage1, img, device, patch_size, expand_factor)
        fov = None if center is None else std_fov_mask(native, center, patch_size, device)
        r = tuple(max(1, int(round(v * args.band_scale)))
                  for v in band_radius(native, patch_size, expand_factor))
        preds, stats = fuse(logits, n_classes, fov=fov, r=r, exp_w=args.exp_weights)

        out_name = img_path.name.replace("_0000", "")
        for name, pred in preds.items():
            (out_dir / name).mkdir(exist_ok=True)
            nib.save(nib.Nifti1Image(pred.cpu().numpy().astype(np.int16), img_nii.affine, img_nii.header),
                     str(out_dir / name / out_name))

        stats_rows.append({"case": normalize_case_id(img_path.name), "band_radius": "x".join(map(str, r)), **stats})
        print(f"[{i:03d}/{len(img_paths):03d}] {out_name}  disagree(fg)={stats['disagree_frac_fg']:.3f}  "
              f"conf: std wins {stats['conf_std_wins_in_disagreement']:.0%} of disagreements")
        del logits, preds

    with open(out_dir / "fusion_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)
    written = [v for v in VARIANTS if (out_dir / v).is_dir()]
    print(f"[INFO] Done. Variants written to {out_dir}: {', '.join(written)}")
    print(f"[INFO] Evaluate all at once: eval_dice_oai.py --pred_dir {out_dir} ...")


if __name__ == "__main__":
    main()
