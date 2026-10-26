#!/usr/bin/env python
"""
Where do each class's GT voxels end up in the prediction?

For every GT class, the fraction of its voxels (pooled over all cases) that
received each predicted label, plus each predicted class's total volume
relative to its GT volume. A class with Dice 0 shows here as either "taken by
a neighbour" (another label dominates its row) or "lost to background".

  python eval/label_confusion.py --gt_dir <labels> --pred_dir <preds> \
      --cases_json <splits_test_final.json> [--classes 6] [--n_classes 16]
"""
import argparse
import glob
import json
import os

import nibabel as nib
import numpy as np

from eval_dice_oai import load_case_ids, normalize_case_name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--cases_json", default=None)
    ap.add_argument("--n_classes", type=int, default=16)
    ap.add_argument("--classes", nargs="*", type=int, default=None, help="rows to print (default: all fg)")
    ap.add_argument("--top", type=int, default=4, help="predicted labels shown per GT class")
    args = ap.parse_args()

    names = {}
    ds_json = os.path.join(os.path.dirname(os.path.normpath(args.gt_dir)), "dataset.json")
    if os.path.exists(ds_json):
        names = {int(k): v for k, v in json.load(open(ds_json)).get("labels", {}).items()}
    nm = lambda c: f"{c}:{names.get(c, '?')}"

    preds = {normalize_case_name(p): p for p in glob.glob(os.path.join(args.pred_dir, "*.nii*"))}
    gts = sorted(glob.glob(os.path.join(args.gt_dir, "*.nii*")))
    if args.cases_json:
        keep = load_case_ids(args.cases_json)
        gts = [g for g in gts if normalize_case_name(g) in keep]

    K = args.n_classes
    conf = np.zeros((K, K), dtype=np.int64)   # rows = GT, cols = pred
    n = 0
    for g in gts:
        cid = normalize_case_name(g)
        if cid not in preds:
            continue
        gt = np.asarray(nib.load(g).dataobj).astype(np.int64).ravel()
        pr = np.asarray(nib.load(preds[cid]).dataobj).astype(np.int64).ravel()
        if gt.shape != pr.shape:
            print(f"[WARN] shape mismatch {cid}, skipped")
            continue
        conf += np.bincount(gt * K + pr, minlength=K * K).reshape(K, K)
        n += 1
    print(f"{n} cases\n")

    rows = args.classes if args.classes else list(range(1, K))
    gt_tot = conf.sum(1)
    pred_tot = conf.sum(0)
    print(f"{'GT class':<22}{'pred/GT vol':>12}   where its GT voxels went")
    for c in rows:
        if gt_tot[c] == 0:
            continue
        frac = conf[c] / gt_tot[c]
        order = np.argsort(frac)[::-1][: args.top]
        where = "  ".join(f"{nm(int(k))} {100 * frac[k]:.0f}%" for k in order if frac[k] > 0.005)
        print(f"{nm(c):<22}{pred_tot[c] / gt_tot[c]:>12.2f}   {where}")

    print("\npredicted volume as a share of the image, for classes predicted much more than GT:")
    for c in range(1, K):
        if gt_tot[c] and pred_tot[c] / gt_tot[c] > 1.5:
            print(f"  {nm(c):<22} predicted {pred_tot[c] / gt_tot[c]:.1f}x its GT volume")


if __name__ == "__main__":
    main()
