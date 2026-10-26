#!/usr/bin/env python
"""
Map AMOS predictions from the canonical training grid back to each scan's
ORIGINAL voxel grid, so Dice can be computed against the untouched AMOS labels
(the space AMOS results are normally reported in).

data/preprocess_amos.py writes every case with an affine that is consistent with
the original scanner space, and infer_stage1_oai.py copies that affine onto its
predictions. Mapping back is therefore a single nearest-neighbour resample of
the prediction onto the original image grid; voxels outside the canonical crop
become background.

  python eval/amos_to_native.py --pred_dir <workdir>/test_preds \
      --amos_root /projects/0/prjs1400/AMOS/amos22 --out_dir <workdir>/test_preds_native
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def find_original(amos_root: Path, stem: str) -> Path:
    for sub in ("imagesVa", "imagesTr", "imagesTs"):
        p = amos_root / sub / f"{stem}.nii.gz"
        if p.exists():
            return p
    raise FileNotFoundError(f"{stem}.nii.gz not found under {amos_root}/images{{Tr,Va,Ts}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True, help="predictions on the canonical grid (.nii.gz)")
    ap.add_argument("--amos_root", required=True, help="original AMOS dir (imagesTr/ imagesVa/ ...)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    amos_root, out_dir = Path(args.amos_root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = sorted(Path(args.pred_dir).glob("amos_*.nii*"))
    if not preds:
        raise RuntimeError(f"no amos_*.nii* predictions in {args.pred_dir}")

    for i, p in enumerate(preds, 1):
        stem = p.name.split(".")[0]
        orig = nib.load(str(find_original(amos_root, stem)))
        pred = nib.load(str(p))
        pred = nib.Nifti1Image(np.asarray(pred.dataobj).astype(np.uint8), pred.affine)
        nat = resample_from_to(pred, (orig.shape[:3], orig.affine), order=0, cval=0)
        hdr = orig.header.copy()
        hdr.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(np.asarray(nat.dataobj).astype(np.uint8), orig.affine, hdr),
                 str(out_dir / f"{stem}.nii.gz"))
        print(f"[{i:03d}/{len(preds):03d}] {stem}: {pred.shape} -> {orig.shape[:3]}", flush=True)


if __name__ == "__main__":
    main()
