#!/usr/bin/env python

import os
import argparse
import glob
import numpy as np
import nibabel as nib
import csv


def dice_per_class(gt, pred, labels):
    scores = []
    for lab in labels:
        gt_bin = (gt == lab)
        pred_bin = (pred == lab)

        inter = (gt_bin & pred_bin).sum()
        denom = gt_bin.sum() + pred_bin.sum()

        if denom == 0:
            scores.append(np.nan)  # class not present in gt+pred
        else:
            scores.append(2.0 * inter / float(denom))
    return np.array(scores, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True, help="Directory with GT labels (.nii / .nii.gz)")
    ap.add_argument("--pred_dir", required=True, help="Directory with predicted labels (.nii / .nii.gz)")
    ap.add_argument("--out_csv", default=None, help="Optional CSV file to save per-case Dice")
    ap.add_argument("--labels", nargs="+", type=int, default=[1, 2, 3, 4], help="Foreground label IDs")
    args = ap.parse_args()

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.nii*")))
    if not gt_files:
        raise RuntimeError(f"No GT files found in {args.gt_dir}")

    print(f"[INFO] Found {len(gt_files)} GT files")
    print(f"[INFO] Using prediction dir: {args.pred_dir}")

    all_dice = []
    rows = []
    missing = 0
    skipped_shape = 0

    for gt_path in gt_files:
        name = os.path.basename(gt_path)
        pred_path = os.path.join(args.pred_dir, name)
        if not os.path.exists(pred_path):
            print(f"[WARN] Missing prediction for {name}, skipping")
            missing += 1
            continue

        gt_img = nib.load(gt_path)
        pred_img = nib.load(pred_path)

        gt = gt_img.get_fdata().astype(np.int16)
        pred = pred_img.get_fdata().astype(np.int16)

        if gt.shape != pred.shape:
            print(f"[WARN] Shape mismatch for {name}: gt {gt.shape}, pred {pred.shape}, skipping")
            skipped_shape += 1
            continue

        dice = dice_per_class(gt, pred, args.labels)
        all_dice.append(dice)

        row = {"case": name}
        for lab, val in zip(args.labels, dice):
            row[f"dice_c{lab}"] = float(val) if not np.isnan(val) else ""
        rows.append(row)

    if not all_dice:
        raise RuntimeError("No cases evaluated (all missing or mismatched shapes).")

    all_dice = np.stack(all_dice, axis=0)
    mean_dice = np.nanmean(all_dice, axis=0)
    std_dice = np.nanstd(all_dice, axis=0)

    print("\n=== Per-class Dice (mean ± std) ===")
    for lab, m, s in zip(args.labels, mean_dice, std_dice):
        print(f"class {lab}: {m:.3f} ± {s:.3f}")
    print("meanFGDice (avg over classes): {:.3f}".format(np.nanmean(mean_dice)))

    print(f"\n[INFO] Evaluated {len(all_dice)} cases")
    if missing:
        print(f"[INFO] Missing predictions for {missing} GT files")
    if skipped_shape:
        print(f"[INFO] Skipped {skipped_shape} cases due to shape mismatch")

    if args.out_csv:
        fieldnames = ["case"] + [f"dice_c{lab}" for lab in args.labels]
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[INFO] Saved per-case Dice to {args.out_csv}")


if __name__ == "__main__":
    main()
