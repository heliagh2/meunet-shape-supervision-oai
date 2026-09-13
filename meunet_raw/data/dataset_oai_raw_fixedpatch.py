import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IGNORE_INDEX = -1


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
    Per-volume z-score on non-zero voxels, like in inference.
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
    if max_size <= size:
        return 0, max_size

    start = center - size // 2
    if start < 0:
        start = 0
    if start + size > max_size:
        start = max_size - size
    end = start + size
    return int(start), int(end)


def _foreground_center_of_mass(lbl: np.ndarray):
    """
    Compute a deterministic foreground center for one case.
    If no foreground exists, fall back to geometric image center.
    """
    fg = np.argwhere(lbl > 0)  # (N, 3)
    if len(fg) == 0:
        D, H, W = lbl.shape
        return int(D // 2), int(H // 2), int(W // 2)

    center = fg.mean(axis=0)
    cz, cy, cx = np.round(center).astype(int)
    return int(cz), int(cy), int(cx)


def _sample_fixed_patch_pair(
    img: np.ndarray,
    lbl: np.ndarray,
    patch_size,
    expand_factor,
    center,
):
    """
    Extract one fixed standard patch + one fixed expanded patch from a 3D volume.

    img, lbl: (D, H, W)
    patch_size: (pz, py, px)
    center: fixed (cz, cy, cx) for this case
    """
    assert img.shape == lbl.shape
    D, H, W = img.shape
    pz, py, px = patch_size
    cz, cy, cx = center

    # standard patch
    z0, z1 = _compute_patch_bounds(cz, pz, D)
    y0, y1 = _compute_patch_bounds(cy, py, H)
    x0, x1 = _compute_patch_bounds(cx, px, W)

    std_img = img[z0:z1, y0:y1, x0:x1]
    std_lbl = lbl[z0:z1, y0:y1, x0:x1]

    # expanded patch
    if expand_factor is None or expand_factor <= 1.0:
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

        # Downsample expanded FoV back to patch_size, as in current pipeline
        big_img_t = torch.from_numpy(big_img[None, None]).float()  # (1,1,D,H,W)
        big_lbl_t = torch.from_numpy(big_lbl[None, None].astype(np.float32))

        exp_img_t = F.interpolate(
            big_img_t,
            size=(pz, py, px),
            mode="trilinear",
            align_corners=False,
        )
        exp_lbl_t = F.interpolate(
            big_lbl_t,
            size=(pz, py, px),
            mode="nearest",
        )

        exp_img = exp_img_t[0, 0].numpy().astype(np.float32)
        exp_lbl = exp_lbl_t[0, 0].numpy().astype(np.int16)

    return std_img, std_lbl, exp_img, exp_lbl


class OAIPairedPatch(Dataset):
    """
    Raw-OAI dataset for MeUNet with FIXED per-case sampling.

    For each case, we compute exactly one fixed center (foreground center of mass),
    and throughout the whole training we always use:
      - one fixed STD patch
      - one fixed EXP patch

    So each case contributes one fixed supervision sample for STD and one fixed
    supervision sample for EXP, reused across all epochs.
    """

    def __init__(
        self,
        images_dir,
        labels_dir,
        stems,
        patch_size,
        expand_factor,
        fg_sampling_prob,   # kept in signature for compatibility; unused now
        train: bool = True, # kept in signature for compatibility
        fixed_center=None,  # None -> legacy per-case GT foreground centre
        annotated_stems=None,  # None -> every case is annotated (legacy)
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
        self.fg_sampling_prob = float(fg_sampling_prob)  # not used, kept for compatibility
        self.train = bool(train)
        self.image_suffixes = ("_0000", "")
        self.label_suffixes = ("", "_segmentation")

        # None    -> legacy behaviour: one GT-derived centre per case
        # (z,y,x) -> the same image-derived centre for every case
        self.fixed_center = None
        if fixed_center is not None:
            if len(fixed_center) != 3:
                raise ValueError(f"fixed_center must have 3 elements, got {fixed_center}")
            self.fixed_center = tuple(int(round(float(v))) for v in fixed_center)

        # Cases outside this set are loaded WITHOUT their segmentation: the
        # label is replaced by ignore_index everywhere and has_gt is False, so
        # nothing downstream can accidentally consume a label the weak-annotation
        # ablation is pretending not to have.
        self.annotated_stems = None if annotated_stems is None else set(annotated_stems)

        # Precompute ONE fixed center per case
        self.fixed_centers = {}
        self._precompute_fixed_centers()

    def __len__(self):
        return len(self.stems)

    def _load_case(self, stem: str):
        img_path = _resolve_case_file(self.images_dir, stem, self.image_suffixes)
        lbl_path = _resolve_case_file(self.labels_dir, stem, self.label_suffixes)

        img_nii = nib.load(str(img_path))
        lbl_nii = nib.load(str(lbl_path))

        # Treat loaded arrays consistently as (D, H, W)
        img = img_nii.get_fdata().astype(np.float32)
        lbl = lbl_nii.get_fdata().astype(np.int16)

        img = zscore_nonzero(img)
        return img, lbl

    def _precompute_fixed_centers(self):
        if self.fixed_center is not None:
            # Shared image-derived centre: no label is read, so this also works
            # for cases whose segmentation is withheld (weak-annotation ablation).
            print(f"[OAIPairedPatch] Using shared fixed center {self.fixed_center} for all "
                  f"{len(self.stems)} cases (no GT centering).")
            for stem in self.stems:
                self.fixed_centers[stem] = self.fixed_center
            return

        print("[OAIPairedPatch] Precomputing one fixed STD/EXP center per case...")
        for stem in self.stems:
            _, lbl = self._load_case(stem)
            self.fixed_centers[stem] = _foreground_center_of_mass(lbl)
        print(f"[OAIPairedPatch] Done. Stored fixed centers for {len(self.fixed_centers)} cases.")

    def __getitem__(self, idx):
        stem = self.stems[idx % len(self.stems)]
        img, lbl = self._load_case(stem)

        center = self.fixed_centers[stem]

        std_img, std_lbl, exp_img, exp_lbl = _sample_fixed_patch_pair(
            img=img,
            lbl=lbl,
            patch_size=self.patch_size,
            expand_factor=self.expand_factor,
            center=center,
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

        has_gt = self.annotated_stems is None or stem in self.annotated_stems
        if not has_gt:
            std_lbl_t = torch.full_like(std_lbl_t, IGNORE_INDEX)
            exp_lbl_t = torch.full_like(exp_lbl_t, IGNORE_INDEX)

        return {
            "std_img": std_img_t,
            "std_lbl": std_lbl_t,
            "exp_img": exp_img_t,
            "exp_lbl": exp_lbl_t,
            "has_gt": torch.tensor(bool(has_gt)),
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

    # nnUNet-style keys
    train_ids = split.get("train") or split.get("train_ids") or split.get("train_keys")
    val_ids   = split.get("val")   or split.get("val_ids")   or split.get("val_keys")

    if train_ids is None or val_ids is None:
        raise KeyError(
            f"Could not find 'train'/'val' keys in split entry: keys={list(split.keys())}"
        )

    train_ids = [_strip_nii_suffix(str(case_id)) for case_id in train_ids]
    val_ids = [_strip_nii_suffix(str(case_id)) for case_id in val_ids]
    return train_ids, val_ids
def compute_average_center(images_dir, labels_dir, stems, image_suffixes=("_0000", ""),
                           label_suffixes=("", "_segmentation")):
    """
    Average GT foreground centre-of-mass over `stems`, in voxels (z, y, x).

    Used to resolve `fixed_center: auto`. Only the stems passed in are read, so
    when the weak-annotation ablation is active this must be called with the
    ANNOTATED subset only — the centre is then a statistic of the labels we are
    allowed to see, not of the ones we withhold.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    coms = []
    for stem in stems:
        lbl_path = _resolve_case_file(labels_dir, stem, label_suffixes)
        lbl = nib.load(str(lbl_path)).get_fdata().astype(np.int16)
        coms.append(_foreground_center_of_mass(lbl))

    if not coms:
        raise RuntimeError("compute_average_center called with no stems")

    center = np.mean(np.asarray(coms, dtype=np.float64), axis=0)
    return tuple(int(round(float(v))) for v in center)


def subsample_annotated(stems, fraction, seed):
    """
    Deterministically keep `fraction` of `stems` as the annotated subset.

    Returns (annotated, unannotated), both sorted. The permutation depends only
    on (sorted stems, seed), so a larger fraction is a strict superset of a
    smaller one for the same seed — the annotation ratios in the ablation are
    then nested rather than independent draws.
    """
    stems = sorted(stems)
    fraction = float(fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"annotated_fraction must be in (0, 1], got {fraction}")

    if fraction >= 1.0:
        return list(stems), []

    n_keep = max(1, int(round(len(stems) * fraction)))
    order = np.random.default_rng(int(seed)).permutation(len(stems))
    keep = {stems[i] for i in order[:n_keep]}

    annotated = [s for s in stems if s in keep]
    unannotated = [s for s in stems if s not in keep]
    return annotated, unannotated
