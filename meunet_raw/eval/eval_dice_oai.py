#!/usr/bin/env python

import os
import argparse
import glob
import numpy as np
import nibabel as nib
import csv


def normalize_case_name(path_or_name):
    name = os.path.basename(path_or_name)
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if name.endswith("_0000"):
        name = name[:-5]
    if name.endswith("_segmentation"):
        name = name[:-13]
    return name


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


HEADS = ("std", "exp")


def evaluate_dir(gt_files, pred_dir, labels):
    """Per-case Dice for one prediction dir. Returns ({case: dice}, n_missing, n_skipped)."""
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii*")))
    pred_map = {normalize_case_name(path): path for path in pred_files}

    per_case = {}
    missing = 0
    skipped_shape = 0

    for gt_path in gt_files:
        name = os.path.basename(gt_path)
        case_id = normalize_case_name(name)
        pred_path = pred_map.get(case_id)
        if pred_path is None:
            print(f"[WARN] Missing prediction for {name} in {pred_dir}, skipping")
            missing += 1
            continue

        gt = nib.load(gt_path).get_fdata().astype(np.int16)
        pred = nib.load(pred_path).get_fdata().astype(np.int16)

        if gt.shape != pred.shape:
            print(f"[WARN] Shape mismatch for {name}: gt {gt.shape}, pred {pred.shape}, skipping")
            skipped_shape += 1
            continue

        per_case[case_id] = dice_per_class(gt, pred, labels)

    return per_case, missing, skipped_shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True, help="Directory with GT labels (.nii / .nii.gz)")
    ap.add_argument("--pred_dir", required=True,
                    help="infer_stage1_oai.py output_dir (with std/ and exp/ subdirs), "
                         "or a legacy flat directory of STD predictions")
    ap.add_argument("--out_csv", default=None, help="Optional CSV file to save per-case Dice")
    ap.add_argument("--labels", nargs="+", type=int, default=[1, 2, 3, 4], help="Foreground label IDs")
    ap.add_argument("--cases_from_pred", action="store_true",
                    help="evaluate only GT cases that have predictions (e.g. a val-split run "
                         "against the full labels dir) instead of warning on every other case")
    args = ap.parse_args()

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.nii*")))
    if not gt_files:
        raise RuntimeError(f"No GT files found in {args.gt_dir}")
    print(f"[INFO] Found {len(gt_files)} GT files")

    # Every subdir holding predictions is one head / fusion variant (std and exp
    # from infer_stage1_oai.py, plus avg/conf/... from infer_fusion_oai.py). A
    # flat dir predates that and only ever held STD (logit1) predictions.
    subdirs = sorted(d for d in os.listdir(args.pred_dir)
                     if glob.glob(os.path.join(args.pred_dir, d, "*.nii*")))
    subdirs = [h for h in HEADS if h in subdirs] + [d for d in subdirs if d not in HEADS]
    head_dirs = {h: os.path.join(args.pred_dir, h) for h in subdirs}
    if not all(h in head_dirs for h in HEADS):
        print(f"[WARN] {args.pred_dir} has no std/ and exp/ subdirs: treating it as legacy "
              f"STD-only predictions. Re-run infer_stage1_oai.py to get the EXP head too.")
        head_dirs = {"std": args.pred_dir}

    if args.cases_from_pred:
        # e.g. a val-split run: score only the cases that were predicted
        have = {normalize_case_name(p)
                for d in head_dirs.values() for p in glob.glob(os.path.join(d, "*.nii*"))}
        gt_files = [g for g in gt_files if normalize_case_name(g) in have]
        print(f"[INFO] --cases_from_pred: restricted to {len(gt_files)} predicted cases")

    results = {}
    for head, d in head_dirs.items():
        print(f"[INFO] {head}: using prediction dir {d}")
        per_case, missing, skipped_shape = evaluate_dir(gt_files, d, args.labels)
        if not per_case:
            raise RuntimeError(f"No cases evaluated for {head} (all missing or mismatched shapes).")
        results[head] = per_case

        all_dice = np.stack(list(per_case.values()), axis=0)
        mean_dice = np.nanmean(all_dice, axis=0)
        std_dice = np.nanstd(all_dice, axis=0)

        print(f"\n=== [{head}] Per-class Dice (mean ± std) ===")
        for lab, m, s in zip(args.labels, mean_dice, std_dice):
            print(f"class {lab}: {m:.3f} ± {s:.3f}")
        print("[{}] meanFGDice (avg over classes): {:.3f}".format(head, np.nanmean(mean_dice)))
        print(f"[INFO] [{head}] Evaluated {len(per_case)} cases")
        if missing:
            print(f"[INFO] [{head}] Missing predictions for {missing} GT files")
        if skipped_shape:
            print(f"[INFO] [{head}] Skipped {skipped_shape} cases due to shape mismatch")

    if len(results) > 1:
        print("\n=== Summary (meanFGDice) ===")
        for head, per_case in results.items():
            m = np.nanmean(np.stack(list(per_case.values()), axis=0), axis=0)
            print(f"{head}: {np.nanmean(m):.3f}  [" +
                  " ".join(f"c{lab}={v:.3f}" for lab, v in zip(args.labels, m)) + "]")

    if args.out_csv:
        # one row per case, one column block per head
        fieldnames = ["case"] + [f"{h}_dice_c{lab}" for h in results for lab in args.labels]
        cases = sorted(set().union(*(r.keys() for r in results.values())))
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for case in cases:
                row = {"case": case}
                for h, per_case in results.items():
                    dice = per_case.get(case)
                    for j, lab in enumerate(args.labels):
                        v = None if dice is None else dice[j]
                        row[f"{h}_dice_c{lab}"] = "" if v is None or np.isnan(v) else float(v)
                w.writerow(row)
        print(f"[INFO] Saved per-case Dice to {args.out_csv}")


if __name__ == "__main__":
    main()
