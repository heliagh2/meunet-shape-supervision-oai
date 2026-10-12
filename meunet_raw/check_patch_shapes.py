#!/usr/bin/env python
"""
Sanity-check the STD/EXP patch geometry a config actually produces.

`_compute_patch_bounds` silently returns the FULL extent of any axis shorter
than patch_size, so the patches a config asks for are not necessarily the ones
it gets. Run this before trusting a descriptor tolerance or a bank entry.

    python check_patch_shapes.py config/A1_9_normalized.yaml \
        --images-dir <local images> --labels-dir <local labels>
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import nibabel as nib

try:
    import yaml
except ImportError:          # minimal fallback: only the two keys we need
    yaml = None


def _read_cfg(path):
    if yaml is not None:
        return yaml.safe_load(open(path))
    cfg, text = {}, open(path).read()
    m = re.search(r"^patch_size:\s*\[([^\]]+)\]", text, re.M)
    if m:
        cfg["patch_size"] = [int(v) for v in m.group(1).split(",")]
    m = re.search(r"^expand_factor:\s*([0-9.]+)", text, re.M)
    if m:
        cfg["expand_factor"] = float(m.group(1))
    m = re.search(r"^images_dir:\s*(\S+)", text, re.M)
    if m:
        cfg["images_dir"] = m.group(1)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = all cases")
    args = ap.parse_args()

    cfg = _read_cfg(args.config)
    images_dir = Path(args.images_dir or cfg["images_dir"])
    patch_size = [int(v) for v in cfg["patch_size"]]
    expand_factor = float(cfg.get("expand_factor", 1.25))

    files = sorted(images_dir.glob("*.nii*"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no NIfTI files under {images_dir}")

    shapes = Counter(nib.load(str(f)).shape for f in files)  # header only, no voxels
    print(f"{len(files)} cases under {images_dir}")
    for s, n in shapes.most_common():
        print(f"  volume {s}  x{n}")
    print(f"\npatch_size={patch_size}  expand_factor={expand_factor}\n")

    for vol, n in shapes.most_common():
        std, exp_crop, warn = [], [], []
        for ax, (full, want) in enumerate(zip(vol, patch_size)):
            std.append(min(full, want))
            if full <= want:
                warn.append(f"STD axis{ax} clipped to full extent ({full} < {want})")
            e = max(int(round(want * expand_factor)), want)
            exp_crop.append(min(full, e))
            if full <= e:
                warn.append(f"EXP axis{ax} saturated at full extent ({full} <= {e})")

        print(f"volume {vol}  (x{n})")
        print(f"  STD patch fed to net : {tuple(std)}")
        print(f"  EXP crop before resize: {tuple(exp_crop)} -> resampled to {tuple(patch_size)}")
        ratio = [round(c / s, 2) for c, s in zip(exp_crop, std)]
        print(f"  effective FoV gain EXP/STD per axis: {ratio}")
        # an axis saturates once round(patch*ef) >= full, i.e. ef >= full/patch;
        # EVERY axis is saturated (EXP == whole volume) at the largest such ratio
        sat = max(f / w for f, w in zip(vol, patch_size))
        print(f"  EXP == whole volume once expand_factor >= {sat:.2f}; "
              f"any larger value gives an identical EXP patch")
        for w in warn:
            print(f"  NOTE: {w}")
        print()


if __name__ == "__main__":
    main()
