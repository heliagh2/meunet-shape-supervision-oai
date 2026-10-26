#!/usr/bin/env python
"""
AMOS22 -> the fixed-grid layout the fixed-patch pipeline expects.

The fixed-patch pipeline assumes what OAI gives for free: every volume on the same
voxel grid, so that one shared patch centre frames the same anatomy in every case
and normalised patch coordinates mean the same thing across cases. AMOS has none
of that -- CT is 512x512x(67..369) at 0.45-1.07mm in-plane and 1.25-5mm slices,
LAS-oriented, MRI is different again -- so this script puts every case on ONE grid:

  1. reorient to RAS (nibabel closest canonical)
  2. resample to a fixed spacing (image trilinear, label nearest)
  3. crop/pad to a fixed shape, centred on the BODY bounding box. The body mask is
     an intensity threshold on the image, never the label, so the framing is
     identical for annotated and label-withheld cases.
  4. normalise intensities once, here:
       CT -> clip to [-175, 250] HU (abdominal soft-tissue window), map to [-1, 1]
       MR -> clip to the 0.5/99.5 percentiles of the body, z-score over the body
     Padding takes the value of air (-1 for CT, the minimum for MR) and label 0.
     Train with `intensity_norm: none` -- the loader must not re-normalise.

Outputs (out_dir):
  images/amos_XXXX.nii.gz   float32, canonical grid, affine kept consistent with
                            the original scanner space (overlays align in viewers)
  labels/amos_XXXX.nii.gz   uint8
  preprocess_meta.json      per-case original shape/spacing/orientation + crop
                            offsets, enough to map predictions back
  dataset.json              class names, grid, normalisation

Splits (written with --splits_dir, or alone with --splits_only):
  splits_final.json         [{"train": [...], "val": [...]}] from the official
                            imagesTr; val stratified by scanner manufacturer
  splits_test_final.json    {"test": [...]} = the official imagesVa, held out.
                            AMOS imagesTs has no public labels, so it is unused.

Usage (cluster):
  python data/preprocess_amos.py \
      --amos_root /projects/0/prjs1400/AMOS/amos22 \
      --meta_csv /projects/0/prjs1400/AMOS/labeled_data_meta_0000_0599.csv \
      --out_dir <writable dir> --splits_dir <repo>/nnunet/nnUNet_preprocessed/Dataset702_AMOS_CT
"""
import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np


CLASS_NAMES = {
    0: "background", 1: "spleen", 2: "right_kidney", 3: "left_kidney",
    4: "gallbladder", 5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta",
    9: "postcava", 10: "pancreas", 11: "right_adrenal", 12: "left_adrenal",
    13: "duodenum", 14: "bladder", 15: "prostate_uterus",
}

CT_WINDOW = (-175.0, 250.0)


def amos_id(stem: str) -> int:
    return int(stem.split("_")[1])


def modality_of(stem: str) -> str:
    # AMOS convention: ids < 500 are CT, >= 500 are MRI
    return "ct" if amos_id(stem) < 500 else "mr"


def list_cases(amos_root: Path, subdir: str, modality: str):
    stems = sorted(p.name[:-7] for p in (amos_root / subdir).glob("amos_*.nii.gz"))
    if modality != "all":
        stems = [s for s in stems if modality_of(s) == modality]
    return stems


# ---------------------------------------------------------------- splits

def read_meta(meta_csv):
    if meta_csv is None:
        return {}
    with open(meta_csv, newline="") as f:
        return {int(r["amos_id"]): r for r in csv.DictReader(f)}


def stratified_val(stems, meta, n_val, seed, key="Manufacturer"):
    """
    Pick n_val stems with each `key` group represented in proportion to its size
    (largest-remainder rounding). Deterministic in (sorted stems, seed).
    """
    groups = defaultdict(list)
    for s in sorted(stems):
        m = meta.get(amos_id(s))
        groups[(m or {}).get(key, "unknown").strip() or "unknown"].append(s)

    total = len(stems)
    quota = {g: n_val * len(v) / total for g, v in groups.items()}
    alloc = {g: int(np.floor(q)) for g, q in quota.items()}
    for g in sorted(quota, key=lambda g: quota[g] - alloc[g], reverse=True)[: n_val - sum(alloc.values())]:
        alloc[g] += 1

    rng = np.random.default_rng(int(seed))
    val = []
    for g in sorted(groups):
        members = groups[g]
        idx = rng.permutation(len(members))[: alloc[g]]
        val += [members[i] for i in idx]
    val = sorted(val)
    train = sorted(s for s in stems if s not in set(val))
    return train, val, {g: (len(groups[g]), alloc[g]) for g in sorted(groups)}


def write_splits(args):
    root = Path(args.amos_root)
    meta = read_meta(args.meta_csv)
    tr = list_cases(root, "imagesTr", args.modality)
    te = list_cases(root, "imagesVa", args.modality)
    train, val, alloc = stratified_val(tr, meta, args.n_val, args.split_seed)

    out = Path(args.splits_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "splits_final.json", "w") as f:
        json.dump([{"train": train, "val": val}], f, indent=2)
    with open(out / "splits_test_final.json", "w") as f:
        json.dump({"test": te}, f, indent=2)
    print(f"[splits] modality={args.modality}: train={len(train)} val={len(val)} test={len(te)} -> {out}")
    for g, (n, k) in alloc.items():
        print(f"[splits]   {g:<28s} {n:4d} in imagesTr -> {k} val")
    return train + val + te


# ---------------------------------------------------------- preprocessing

def _largest_component_bbox(mask):
    from scipy import ndimage

    mask = ndimage.binary_opening(mask, iterations=2)
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    keep = lab == (int(np.argmax(sizes)) + 1)
    idx = np.argwhere(keep)
    return idx.min(0), idx.max(0)


def _crop_pad(arr, start, shape, fill):
    out = np.full(shape, fill, dtype=arr.dtype)
    src, dst = [], []
    for a in range(3):
        s0 = start[a]
        lo, hi = max(s0, 0), min(s0 + shape[a], arr.shape[a])
        src.append(slice(lo, hi))
        dst.append(slice(lo - s0, hi - s0))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def _resample(arr, new_shape, mode):
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
    if mode == "linear":
        t = F.interpolate(t, size=new_shape, mode="trilinear", align_corners=False)
    else:
        t = F.interpolate(t, size=new_shape, mode="nearest-exact")
    return t[0, 0].numpy()


def preprocess_case(job):
    import torch

    torch.set_num_threads(1)
    stem, img_path, lbl_path, out_dir, spacing, shape = job
    spacing = np.asarray(spacing, dtype=np.float64)
    shape = tuple(int(v) for v in shape)
    mod = modality_of(stem)

    img_nii = nib.as_closest_canonical(nib.load(str(img_path)))
    lbl_nii = nib.as_closest_canonical(nib.load(str(lbl_path))) if lbl_path else None
    img = np.asarray(img_nii.dataobj, dtype=np.float32)
    old_sp = np.asarray(img_nii.header.get_zooms()[:3], dtype=np.float64)
    old_shape = np.asarray(img.shape)

    rs_shape = tuple(int(v) for v in np.maximum(np.round(old_shape * old_sp / spacing), 1))
    img_rs = _resample(img, rs_shape, "linear")

    # body mask -- image only, so annotated and withheld cases are framed alike
    if mod == "ct":
        body = img_rs > -500.0
    else:
        body = img_rs > 0.05 * np.percentile(img_rs, 99.0)
    bb = _largest_component_bbox(body)
    if bb is None:
        lo, hi = np.zeros(3, int), np.asarray(rs_shape) - 1
    else:
        lo, hi = bb
    centre = (lo + hi) // 2
    start = [int(centre[a] - shape[a] // 2) for a in range(3)]

    if mod == "ct":
        a, b = CT_WINDOW
        img_rs = (np.clip(img_rs, a, b) - (a + b) / 2) / ((b - a) / 2)
        fill = -1.0
    else:
        vals = img_rs[body] if body.any() else img_rs.ravel()
        p_lo, p_hi = np.percentile(vals, [0.5, 99.5])
        img_rs = np.clip(img_rs, p_lo, p_hi)
        vals = img_rs[body] if body.any() else img_rs.ravel()
        img_rs = (img_rs - vals.mean()) / max(vals.std(), 1e-8)
        fill = float(img_rs.min())
    img_out = _crop_pad(img_rs.astype(np.float32), start, shape, np.float32(fill))

    # affine: canonical voxel i_old = (j + start + 0.5) * f - 0.5 (align_corners=False)
    f = old_shape / np.asarray(rs_shape)
    M = np.eye(4)
    M[:3, :3] = np.diag(f)
    M[:3, 3] = (np.asarray(start) + 0.5) * f - 0.5
    affine = img_nii.affine @ M

    out_dir = Path(out_dir)
    hdr = nib.Nifti1Header()
    hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(img_out, affine, hdr), str(out_dir / "images" / f"{stem}.nii.gz"))

    counts = None
    if lbl_nii is not None:
        lbl = np.asarray(lbl_nii.dataobj).astype(np.uint8)
        lbl_rs = _resample(lbl, rs_shape, "nearest").round().astype(np.uint8)
        lbl_out = _crop_pad(lbl_rs, start, shape, np.uint8(0))
        hdr = nib.Nifti1Header()
        hdr.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(lbl_out, affine, hdr), str(out_dir / "labels" / f"{stem}.nii.gz"))
        # voxels lost to the crop, per class -- a too-small grid shows up here
        u0, c0 = np.unique(lbl_rs, return_counts=True)
        u1, c1 = np.unique(lbl_out, return_counts=True)
        before = dict(zip(u0.tolist(), c0.tolist()))
        after = dict(zip(u1.tolist(), c1.tolist()))
        counts = {int(c): [int(before[c]), int(after.get(c, 0))] for c in before if c > 0}

    return stem, {
        "modality": mod,
        "orig_shape": [int(v) for v in old_shape],
        "orig_spacing": [float(v) for v in old_sp],
        "orig_axcodes": "".join(nib.aff2axcodes(nib.load(str(img_path)).affine)),
        "resampled_shape": list(rs_shape),
        "crop_start": start,
        "body_bbox": [[int(v) for v in lo], [int(v) for v in hi]],
        "label_voxels_before_after_crop": counts,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amos_root", required=True, help="dir containing imagesTr/ labelsTr/ imagesVa/ labelsVa/")
    ap.add_argument("--meta_csv", default=None, help="labeled_data_meta_0000_0599.csv (for val stratification)")
    ap.add_argument("--out_dir", default=None, help="where the preprocessed volumes go")
    ap.add_argument("--splits_dir", default=None, help="where splits_final.json / splits_test_final.json go")
    ap.add_argument("--splits_only", action="store_true", help="write the splits and exit")
    ap.add_argument("--modality", default="ct", choices=["ct", "mr", "all"])
    # (x, y, z) after RAS reorientation: L-R, P-A, I-S
    ap.add_argument("--spacing", nargs=3, type=float, default=[1.5, 1.5, 3.0])
    ap.add_argument("--shape", nargs=3, type=int, default=[320, 256, 224],
                    help="canonical grid; keep each divisible by 8 (3 poolings) for whole-volume inference")
    ap.add_argument("--n_val", type=int, default=20)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="*", default=None, help="only these stems (debugging)")
    args = ap.parse_args()

    root = Path(args.amos_root)
    if args.splits_dir:
        write_splits(args)
    if args.splits_only:
        return
    if args.out_dir is None:
        ap.error("--out_dir is required unless --splits_only")

    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    jobs = []
    for sub in ("Tr", "Va"):
        for stem in list_cases(root, f"images{sub}", args.modality):
            if args.cases and stem not in args.cases:
                continue
            lbl = root / f"labels{sub}" / f"{stem}.nii.gz"
            jobs.append((stem, root / f"images{sub}" / f"{stem}.nii.gz",
                         lbl if lbl.exists() else None, str(out_dir), args.spacing, args.shape))
    print(f"[prep] {len(jobs)} cases -> {out_dir}  spacing={args.spacing} shape={args.shape}", flush=True)

    meta_path = out_dir / "preprocess_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (stem, m) in enumerate(ex.map(preprocess_case, jobs, chunksize=1), 1):
            meta[stem] = m
            lost = {c: b - a for c, (b, a) in (m["label_voxels_before_after_crop"] or {}).items() if b > a}
            if lost:
                print(f"[prep]   {stem}: crop removed label voxels {lost}", flush=True)
            if i % 25 == 0:
                print(f"[prep]   {i}/{len(jobs)}", flush=True)

    meta_path.write_text(json.dumps(meta, indent=1))
    (out_dir / "dataset.json").write_text(json.dumps({
        "name": "AMOS22", "modality": args.modality, "labels": CLASS_NAMES,
        "n_classes": len(CLASS_NAMES), "spacing_xyz_mm": args.spacing, "shape_xyz": args.shape,
        "orientation": "RAS", "ct_window_hu": CT_WINDOW,
        "intensity": "already normalised -- train with intensity_norm: none",
    }, indent=2))
    print(f"[prep] done. wrote {meta_path}")


if __name__ == "__main__":
    main()
