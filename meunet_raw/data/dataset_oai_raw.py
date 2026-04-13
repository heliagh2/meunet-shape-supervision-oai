import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def _resolve_case_file(directory: Path, stem: str, suffixes) -> Path:
    for suffix in suffixes:
        for ext in (".nii.gz", ".nii"):
            candidate = directory / f"{stem}{suffix}{ext}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"Could not find file for stem '{stem}' in {directory}. "
        f"Tried suffixes: {list(suffixes)}"
    )


def zscore_nonzero(vol: np.ndarray) -> np.ndarray:
    """
    Per-volume z-score on non-zero voxels, like we used in inference.
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
    return (vol - m) / s


def _compute_patch_bounds(center: int, size: int, max_size: int):
    """
    Given a center index, patch size and full size, return [start, end)
    bounds that stay inside [0, max_size).
    """
    # if volume is smaller than patch -> just take the full axis
    if max_size <= size:
        return 0, max_size

    start = center - size // 2
    if start < 0:
        start = 0
    if start + size > max_size:
        start = max_size - size
    end = start + size
    return int(start), int(end)


def _sample_patch(
    img: np.ndarray,
    lbl: np.ndarray,
    patch_size,
    expand_factor,
    fg_sampling_prob,
    train: bool = True,
):
    """
    Sample a standard patch + an expanded patch from a 3D volume.

    img, lbl: (D, H, W)
    patch_size: (pz, py, px)
    expand_factor: e.g. 1.25 (None or <=1.0 -> no expansion)
    """
    assert img.shape == lbl.shape
    D, H, W = img.shape
    pz, py, px = patch_size

    # choose patch center
    if train and np.random.rand() < fg_sampling_prob and np.any(lbl > 0):
        # foreground-biased sampling
        fg_idx = np.argwhere(lbl > 0)  # (N,3)
        cz, cy, cx = fg_idx[np.random.randint(len(fg_idx))]
    else:
        # uniform sampling anywhere in the volume
        cz = np.random.randint(0, D)
        cy = np.random.randint(0, H)
        cx = np.random.randint(0, W)

    # standard patch bounds
    z0, z1 = _compute_patch_bounds(cz, pz, D)
    y0, y1 = _compute_patch_bounds(cy, py, H)
    x0, x1 = _compute_patch_bounds(cx, px, W)

    std_img = img[z0:z1, y0:y1, x0:x1]
    std_lbl = lbl[z0:z1, y0:y1, x0:x1]

    #expanded patch 
    if expand_factor is None or expand_factor <= 1.0:
        # fallback: no expansion, just copy std patch
        exp_img = std_img.copy()
        exp_lbl = std_lbl.copy()
    else:
        ez = int(round(pz * expand_factor))
        ey = int(round(py * expand_factor))
        ex = int(round(px * expand_factor))

        ez = max(ez, pz)
        ey = max(ey, py)
        ex = max(ex, px)

        z0e, z1e = _compute_patch_bounds(cz, ez, D)
        y0e, y1e = _compute_patch_bounds(cy, ey, H)
        x0e, x1e = _compute_patch_bounds(cx, ex, W)

        big_img = img[z0e:z1e, y0e:y1e, x0e:x1e]
        big_lbl = lbl[z0e:z1e, y0e:y1e, x0e:x1e]

        # interpolate down to patch_size
        big_img_t = torch.from_numpy(big_img[None, None]).float()  # (1,1,D,H,W)
        big_lbl_t = torch.from_numpy(big_lbl[None, None].astype(np.float32))

        exp_img_t = F.interpolate(
            big_img_t, size=(pz, py, px), mode="trilinear", align_corners=False
        )
        exp_lbl_t = F.interpolate(
            big_lbl_t, size=(pz, py, px), mode="nearest"
        )

        exp_img = exp_img_t[0, 0].numpy().astype(np.float32)
        exp_lbl = exp_lbl_t[0, 0].numpy().astype(np.int16)

    return std_img, std_lbl, exp_img, exp_lbl


class OAIPairedPatch(Dataset):
    """
    Raw-OAI dataset for MeUNet:

    - Reads NIfTI from either nnU-Net-style directories or flat image/label dirs
    - Applies per-volume z-score norm on non-zero voxels
    - Samples a standard patch and an expanded patch per __getitem__
    """

    def __init__(
        self,
        images_dir,
        labels_dir,
        stems,
        patch_size,
        expand_factor,
        fg_sampling_prob,
        train: bool = True,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.stems = list(stems)

        if not len(self.stems):
            raise RuntimeError(
                f"No stems provided to OAIPairedPatch (images_dir={self.images_dir})"
            )

        self.patch_size = tuple(int(x) for x in patch_size)
        if len(self.patch_size) != 3:
            raise ValueError(f"patch_size must have 3 elements, got {self.patch_size}")

        self.expand_factor = float(expand_factor) if expand_factor is not None else None
        self.fg_sampling_prob = float(fg_sampling_prob)
        self.train = bool(train)
        self.image_suffixes = ("_0000", "")
        self.label_suffixes = ("", "_segmentation")

    def __len__(self):
        return len(self.stems)

    def _load_case(self, stem: str):
        img_path = _resolve_case_file(self.images_dir, stem, self.image_suffixes)
        lbl_path = _resolve_case_file(self.labels_dir, stem, self.label_suffixes)

        img_nii = nib.load(str(img_path))
        lbl_nii = nib.load(str(lbl_path))

        # nibabel returns (X,Y,Z) but we treat it consistently as (D,H,W)
        img = img_nii.get_fdata().astype(np.float32)
        lbl = lbl_nii.get_fdata().astype(np.int16)

        img = zscore_nonzero(img)
        return img, lbl

    def __getitem__(self, idx):
        stem = self.stems[idx % len(self.stems)]
        img, lbl = self._load_case(stem)

        std_img, std_lbl, exp_img, exp_lbl = _sample_patch(
            img,
            lbl,
            self.patch_size,
            self.expand_factor,
            self.fg_sampling_prob,
            train=self.train,
        )

        std_img = std_img.astype(np.float32)
        exp_img = exp_img.astype(np.float32)
        std_lbl = std_lbl.astype(np.int16)
        exp_lbl = exp_lbl.astype(np.int16)

        # add channel dim for images
        std_img_t = torch.from_numpy(std_img[None]).float()   # (1,D,H,W)
        exp_img_t = torch.from_numpy(exp_img[None]).float()   # (1,D,H,W)
        std_lbl_t = torch.from_numpy(std_lbl).long()          # (D,H,W)
        exp_lbl_t = torch.from_numpy(exp_lbl).long()          # (D,H,W)

        return {
            "std_img": std_img_t,
            "std_lbl": std_lbl_t,
            "exp_img": exp_img_t,
            "exp_lbl": exp_lbl_t,
        }


def load_splits(cfg):
    """
    Read nnUNet's splits_final.json and return (train_stems, val_stems).

    Expected cfg entries:
      - splits_file: path to splits_final.json
      - split (optional): which split index to use (default 0)
    """
    splits_file = cfg.get("splits_file")
    if splits_file is None:
        raise RuntimeError("cfg must contain 'splits_file' pointing to nnUNet splits_final.json")

    splits_path = Path(splits_file)
    if not splits_path.exists():
        raise FileNotFoundError(f"splits_file not found: {splits_path}")

    with open(splits_path, "r") as f:
        splits = json.load(f)

    split_idx = int(cfg.get("split", 0))
    if split_idx < 0 or split_idx >= len(splits):
        raise IndexError(
            f"split index {split_idx} out of range for splits_file with {len(splits)} entries"
        )

    split = splits[split_idx]

    # nnUNet-style keys; we try a few variants just in case
    train_ids = split.get("train") or split.get("train_ids") or split.get("train_keys")
    val_ids   = split.get("val")   or split.get("val_ids")   or split.get("val_keys")

    if train_ids is None or val_ids is None:
        raise KeyError(
            f"Could not find 'train'/'val' keys in split entry: keys={list(split.keys())}"
        )

    train_ids = [_strip_nii_suffix(str(case_id)) for case_id in train_ids]
    val_ids = [_strip_nii_suffix(str(case_id)) for case_id in val_ids]
    return train_ids, val_ids
