#!/usr/bin/env python
import sys
from pathlib import Path

# --- make repo root importable ---
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import json
import math
import os
import time
import random
import argparse

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from data.dataset_oai_raw_fixedpatch import (
    OAIPairedPatch,
    load_splits,
    compute_average_center,
    subsample_annotated,
)
from models.meunet3d import MEUNet3D
from losses.dice_ce import DiceCELoss
from losses.moment_invariants import (
    compute_2nd_moment_barrier,
    compute_3rd_moment_barrier,
    compute_moment_invariants_barrier,
)


# helpers

def lr_now(optimizer):
    for pg in optimizer.param_groups:
        return float(pg.get("lr", 0.0))
    return 0.0


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_int_list(x, default=None):
    """
    Accept int, str-int, list/tuple of ints; return list[int].
    """
    if x is None:
        return [] if default is None else list(default)
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


def resize_lbl_to_logits(lbl: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    lbl: (B,D,H,W) int (may contain -1 ignore)
    logits: (B,C,D',H',W')
    returns lbl resized to (B,D',H',W') using nearest neighbor
    """
    if tuple(lbl.shape[-3:]) == tuple(logits.shape[-3:]):
        return lbl
    lbl_f = lbl.unsqueeze(1).float()
    lbl_rs = F.interpolate(lbl_f, size=logits.shape[-3:], mode="nearest").squeeze(1)
    return lbl_rs.long()


def select_viz_slice(lbl, cartilage_classes, axis=0, min_pixels=1):
    """
    Pick the slice along `axis` that best shows the given (small) cartilage
    classes: maximize the smallest per-class pixel count among them, so both
    classes are visible rather than just the larger one. Falls back to their
    summed pixel count if no slice has >= min_pixels for every class.
    lbl: (D,H,W) int array.
    """
    lbl = np.asarray(lbl)
    n_slices = lbl.shape[axis]
    counts = np.zeros((n_slices, len(cartilage_classes)), dtype=np.int64)
    for i in range(n_slices):
        sl = np.take(lbl, i, axis=axis)
        for j, c in enumerate(cartilage_classes):
            counts[i, j] = int((sl == c).sum())
    min_counts = counts.min(axis=1)
    best = int(min_counts.argmax())
    if min_counts[best] < min_pixels:
        best = int(counts.sum(axis=1).argmax())
    return best


_VIZ_CLASS_COLORS = {
    1: (0.20, 0.45, 1.00),  # femur
    2: (0.20, 0.85, 0.30),  # femoral cartilage
    3: (1.00, 0.55, 0.10),  # tibia
    4: (1.00, 0.90, 0.10),  # tibial cartilage
}


def _draw_mask_overlay(ax, img, mask, class_names, alpha=0.45):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    for c, color in _VIZ_CLASS_COLORS.items():
        if c not in class_names:
            continue
        m = mask == c
        rgba[m, 0:3] = color
        rgba[m, 3] = alpha
    ax.imshow(rgba)
    ax.set_axis_off()


def build_viz_wandb_image(img_slice, gt_slice, pred_slice, class_names):
    """
    Builds a self-contained 3-panel PNG (scan | GT overlay | prediction
    overlay) with the overlay baked into the pixels, so it downloads/screenshots
    correctly (unlike wandb's client-side-rendered `masks=` overlay, which
    only bakes the underlying scan into the exported image).
    img_slice: (H,W) float array, min-max normalized for display.
    gt_slice, pred_slice: (H,W) int arrays of class indices.
    """
    img = np.asarray(img_slice, dtype=np.float32)
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    gt_slice = np.asarray(gt_slice)
    pred_slice = np.asarray(pred_slice)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), dpi=120)
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("scan", fontsize=9)
    axes[0].set_axis_off()
    _draw_mask_overlay(axes[1], img, gt_slice, class_names)
    axes[1].set_title("ground truth", fontsize=9)
    _draw_mask_overlay(axes[2], img, pred_slice, class_names)
    axes[2].set_title("prediction", fontsize=9)

    handles = [
        matplotlib.patches.Patch(color=color, label=class_names.get(c, str(c)))
        for c, color in _VIZ_CLASS_COLORS.items()
        if c in class_names
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    out = rgba[..., :3].copy()
    plt.close(fig)
    return wandb.Image(out)


def one_hot_labels(target, n_classes, ignore_index=-1):
    """
    target: (B,Z,Y,X) int with possible -1
    Returns:
      one_hot: (B,C,Z,Y,X) float
      valid_mask: (B,Z,Y,X) bool
    """
    mask = (target != ignore_index)
    t = target.clamp(min=0)
    oh = F.one_hot(t.long(), num_classes=n_classes).permute(0, 4, 1, 2, 3).float()
    oh = oh * mask.unsqueeze(1)
    return oh, mask


@torch.no_grad()
def soft_dice_per_class(logits, target, n_classes, ignore_index=-1, eps=1e-6):
    """
    Soft 'pseudo dice' for classes 1..n_classes-1 (exclude background).
    Returns list length (n_classes-1).
    """
    probs = F.softmax(logits.float(), dim=1)
    tgt_oh, valid_mask = one_hot_labels(target, n_classes, ignore_index)

    probs_fg = probs[:, 1:, ...]
    tgt_fg   = tgt_oh[:, 1:, ...]
    vm = valid_mask.unsqueeze(1)

    probs_fg = probs_fg * vm
    tgt_fg   = tgt_fg * vm

    dims = tuple(range(2, probs_fg.ndim))
    inter = (probs_fg * tgt_fg).sum(dim=dims)
    p_sum = probs_fg.sum(dim=dims)
    t_sum = tgt_fg.sum(dim=dims)

    dice = (2 * inter + eps) / (p_sum + t_sum + eps)   # (B, C-1)
    return dice.mean(dim=0).cpu().double().tolist()


DESC_SLICES = {"volume": slice(0, 1), "centroid": slice(1, 4), "spread": slice(4, 7)}


def mass_descriptors(mass, tot, eps=1e-6):
    """
    Descriptors of a nonnegative mass field, laid out as
    [frac, cz, cy, cx, sz, sy, sx] -- exactly the quantities the barriers
    constrain. Differentiable, so it serves both the val-time collapse
    diagnostic and the training-time distribution loss.

    mass: (B,D,H,W) nonnegative.  tot: (B,) voxel count the fraction is over.
    """
    device = mass.device
    _, D, H, W = mass.shape
    grids = [
        torch.linspace(0, 1, D, device=device).view(1, D, 1, 1),
        torch.linspace(0, 1, H, device=device).view(1, 1, H, 1),
        torch.linspace(0, 1, W, device=device).view(1, 1, 1, W),
    ]
    m = mass.sum(dim=(1, 2, 3)) + eps
    vals = [mass.sum(dim=(1, 2, 3)) / tot]
    cents = [(mass * g).sum(dim=(1, 2, 3)) / m for g in grids]
    vals += cents
    for g, c in zip(grids, cents):
        var = (mass * (g - c.view(-1, 1, 1, 1)) ** 2).sum(dim=(1, 2, 3)) / m
        vals.append(torch.sqrt(var + eps))
    return torch.stack(vals, dim=1)  # (B,7)


def predicted_descriptors(logits, classes, eps=1e-6):
    """Differentiable descriptors of the predicted softmax mass, (B, len(classes), 7).

    No label is touched — this is the path unannotated cases take.
    """
    probs = F.softmax(logits.float(), dim=1)
    B = probs.shape[0]
    tot = torch.full((B,), float(np.prod(probs.shape[-3:])), device=probs.device) + eps
    return torch.stack([mass_descriptors(probs[:, c], tot, eps) for c in classes], dim=1)


def descriptor_summary(logits, target, classes, n_classes, ignore_index=-1, eps=1e-6):
    """
    Per-sample descriptor values for the predicted and the GT mask, matching what
    the barriers constrain: volume fraction, normalised centroid and per-axis
    spread. Diagnostic only — never used for training.

    A bank-target barrier is minimised by predicting the population mean for every
    case, which satisfies every constraint while reproducing none of the real
    anatomical variation. Comparing the spread of predicted descriptors against
    the spread of GT descriptors over the val set makes that collapse visible:
    a ratio near 1 means the model reproduces population variation, a ratio near
    0 means it has collapsed onto the mean.

    Returns (pred, gt, present); pred/gt are (B, len(classes), 7) laid out as
    [frac, cz, cy, cx, sz, sy, sx] and present is (B, len(classes)).
    """
    device = logits.device
    probs = F.softmax(logits.float(), dim=1)
    valid = (target != ignore_index).float().unsqueeze(1)
    probs = probs * valid

    one_hot, _ = one_hot_labels(target, n_classes, ignore_index)
    one_hot = one_hot.to(device) * valid

    tot = valid.sum(dim=(2, 3, 4)).squeeze(1) + eps  # (B,)

    preds, gts, present = [], [], []
    for c in classes:
        preds.append(mass_descriptors(probs[:, c], tot, eps))
        gts.append(mass_descriptors(one_hot[:, c], tot, eps))
        present.append(one_hot[:, c].sum(dim=(1, 2, 3)) > 0)

    return torch.stack(preds, 1), torch.stack(gts, 1), torch.stack(present, 1)


def label_descriptors(lbl, classes, eps=1e-6):
    """
    Descriptor values of a single GT label patch, computed exactly as the
    barriers compute their `gt_*` terms: volume fraction over the whole patch,
    centroid and per-axis spread in normalised [0,1] patch coordinates, and mean
    Euclidean distance to the centroid.

    `lbl` must already be at the resolution the barrier sees (see
    build_descriptor_bank), because volume fraction is not resolution-invariant
    for thin structures under nearest-neighbour downsampling.
    """
    D, H, W = lbl.shape
    tot = float(lbl.size)
    grids = [np.linspace(0, 1, D), np.linspace(0, 1, H), np.linspace(0, 1, W)]

    out = {}
    for c in classes:
        idx = np.argwhere(lbl == c)
        if len(idx) == 0:
            out[int(c)] = None
            continue
        coords = [grids[a][idx[:, a]] for a in range(3)]
        cent = [float(cc.mean()) for cc in coords]
        spread = [float(np.sqrt(((cc - ce) ** 2).mean())) for cc, ce in zip(coords, cent)]
        dist = np.sqrt(sum((cc - ce) ** 2 for cc, ce in zip(coords, cent)))
        out[int(c)] = {
            "frac": len(idx) / tot,
            "cent": cent,
            "spread": spread,
            "avgdist": float(dist.mean()),
        }
    return out


def build_descriptor_bank(cfg, stems, fixed_center, classes, shapes, workdir=None, verbose=True):
    """
    Population mean of each descriptor, per patch type and per class, estimated
    from `stems` — which must be the ANNOTATED cases only. Using withheld cases
    here would leak exactly the labels the ablation pretends not to have.

    `shapes` maps "std"/"exp" to the spatial shape the corresponding barrier
    operates on. These differ: the STD branch is supervised through logit1 (full
    patch resolution) while the EXP branch goes through logit2, one decoder level
    up and therefore half resolution. Volume fractions computed at the wrong
    resolution are systematically biased for thin classes, so each patch type is
    measured at its own.

    Returns {"std": {class: {...}}, "exp": {...}}, with the population standard
    deviation kept alongside each mean for reporting.
    """
    ds = OAIPairedPatch(
        cfg["images_dir"], cfg["labels_dir"], stems,
        cfg["patch_size"], float(cfg.get("expand_factor", 1.25)),
        float(cfg.get("fg_sampling_prob", 0.5)), False,
        fixed_center=fixed_center,
    )

    acc = {p: {int(c): [] for c in classes} for p in ("std", "exp")}
    for i in range(len(ds)):
        item = ds[i]
        for patch in ("std", "exp"):
            lbl = item[f"{patch}_lbl"]
            want = tuple(shapes[patch])
            if tuple(lbl.shape[-3:]) != want:
                lbl = F.interpolate(
                    lbl[None, None].float(), size=want, mode="nearest"
                )[0, 0].long()
            d = label_descriptors(lbl.numpy(), classes)
            for c in classes:
                if d[int(c)] is not None:
                    acc[patch][int(c)].append(d[int(c)])
        if verbose and (i + 1) % 25 == 0:
            print(f"[bank]   {i + 1}/{len(ds)} cases", flush=True)

    bank = {}
    for patch in ("std", "exp"):
        bank[patch] = {}
        for c in classes:
            vals = acc[patch][int(c)]
            if not vals:
                raise RuntimeError(f"class {c} never present in {patch} patches — cannot build a bank entry")
            def ms(key):
                a = np.array([v[key] for v in vals], dtype=np.float64)
                return a.mean(axis=0).tolist(), a.std(axis=0).tolist()
            entry = {}
            for key in ("frac", "cent", "spread", "avgdist"):
                m, sd = ms(key)
                entry[key] = m
                entry[key + "_std"] = sd
            entry["n"] = len(vals)
            bank[patch][str(int(c))] = entry

    meta = {
        "n_stems": len(stems),
        "fixed_center": list(fixed_center) if fixed_center is not None else None,
        "shapes": {k: list(v) for k, v in shapes.items()},
        "classes": [int(c) for c in classes],
    }
    out = {"meta": meta, "bank": bank}
    if workdir is not None:
        with open(Path(workdir) / "descriptor_bank.json", "w") as f:
            json.dump(out, f, indent=2)
    return out


def bank_tensors(bank, patch, classes, device):
    """Pack one patch type's bank into the tensors the barriers expect."""
    b = bank["bank"][patch]
    return {
        "frac": {int(c): torch.tensor(b[str(int(c))]["frac"], dtype=torch.float32, device=device) for c in classes},
        "cent": {int(c): torch.tensor(b[str(int(c))]["cent"], dtype=torch.float32, device=device) for c in classes},
        "spread": {int(c): torch.tensor(b[str(int(c))]["spread"], dtype=torch.float32, device=device) for c in classes},
        "avgdist": {int(c): torch.tensor(b[str(int(c))]["avgdist"], dtype=torch.float32, device=device) for c in classes},
    }


class DescriptorMomentMatcher:
    """
    Expectation regularisation over the UNANNOTATED pool.

    A per-sample population target carries zero information about the individual
    scan, and its barrier is minimised by predicting mean anatomy for every case
    (see A1.3). This instead constrains the pool's aggregate:

        | mean_i(d_i) - bank_mean |  <= tol_mean
        | std_i(d_i)  - bank_std   |  <= tol_std

    The std constraint is the part that matters: collapsing onto the mean now
    VIOLATES a constraint instead of satisfying all of them.

    tol_mean is the standard error of a mean over n_eff samples, k * s/sqrt(n),
    so the aggregate is pinned much more tightly than any individual case could
    be -- the population mean is genuinely well known. tol_std is relative.

    batch_size is far too small to estimate a distribution, so each statistic
    blends the current batch (carrying gradient) with a detached EMA standing in
    for the rest of the pool:  x_hat = a * batch + (1 - a) * ema,  a = B/n_eff.
    The EMA is stale by construction; if that proves unstable the exact
    alternative is accumulating several batches before applying the term.
    """

    def __init__(self, bank, classes, device, n_eff=64.0, k_mean=2.0, rel_std_tol=0.25, eps=1e-8):
        self.classes = [int(c) for c in classes]
        self.n_eff = float(n_eff)
        self.eps = float(eps)
        self.state = {}
        self.target = {}
        self.active = {}
        for patch in ("std", "exp"):
            b = bank["bank"][patch]
            mu = torch.tensor([[b[str(c)]["frac"]] + list(b[str(c)]["cent"]) + list(b[str(c)]["spread"])
                               for c in self.classes], dtype=torch.float32, device=device)
            sd = torch.tensor([[b[str(c)]["frac_std"]] + list(b[str(c)]["cent_std"]) + list(b[str(c)]["spread_std"])
                               for c in self.classes], dtype=torch.float32, device=device)
            # A descriptor the annotated cases show no spread in carries no
            # distribution to match. Constraining it would demand sd == 0
            # exactly, which is unsatisfiable; disable those entries instead.
            degenerate = sd <= 1e-8
            self.active[patch] = ~degenerate
            sd = sd.clamp(min=self.eps)
            self.target[patch] = {
                "mu": mu,
                "sd": sd,
                "tol_mu": k_mean * sd / math.sqrt(self.n_eff),
                "tol_sd": rel_std_tol * sd,
            }
            if bool(degenerate.any()):
                print(f"[dist] warning: {int(degenerate.sum())} {patch} descriptor(s) have zero "
                      f"population spread — their distribution constraints are disabled.")
            # EMA seeded at the bank itself: before any evidence, assume the pool
            # already matches, so early batches move it rather than fight an
            # arbitrary initial value.
            self.state[patch] = {"mu": mu.clone(), "e2": (mu ** 2 + sd ** 2).clone()}

    def __call__(self, logits, patch, barrier, group_weights):
        """
        logits: (Bu,C,...) for the unannotated samples of this batch.
        group_weights: {"volume": w, "centroid": w, "spread": w} -- the same
        lambdas the per-sample barriers use, so the two paths stay commensurate.
        Returns (loss, stats).
        """
        pred = predicted_descriptors(logits, self.classes)      # (Bu, C, 7)
        Bu = pred.shape[0]
        a = min(1.0, Bu / self.n_eff)

        st, tg = self.state[patch], self.target[patch]
        batch_mu = pred.mean(dim=0)
        batch_e2 = (pred ** 2).mean(dim=0)

        mu_hat = a * batch_mu + (1.0 - a) * st["mu"]
        e2_hat = a * batch_e2 + (1.0 - a) * st["e2"]
        sd_hat = torch.sqrt((e2_hat - mu_hat ** 2).clamp(min=self.eps))

        with torch.no_grad():
            st["mu"] = (1.0 - a) * st["mu"] + a * batch_mu.detach()
            st["e2"] = (1.0 - a) * st["e2"] + a * batch_e2.detach()

        z_mu = torch.maximum(mu_hat - (tg["mu"] + tg["tol_mu"]), (tg["mu"] - tg["tol_mu"]) - mu_hat)
        z_sd = torch.maximum(sd_hat - (tg["sd"] + tg["tol_sd"]), (tg["sd"] - tg["tol_sd"]) - sd_hat)

        act = self.active[patch]
        loss = logits.new_tensor(0.0)
        for name, sl in DESC_SLICES.items():
            w = float(group_weights.get(name, 0.0))
            if w <= 0.0:
                continue
            a = act[:, sl]
            if not bool(a.any()):
                continue
            loss = loss + w * (barrier(z_mu[:, sl][a]) + barrier(z_sd[:, sl][a]))

        stats = {
            "z_mu": z_mu.detach(), "z_sd": z_sd.detach(), "active": act,
            "sd_ratio": (sd_hat / tg["sd"]).detach(),
            "mu_hat": mu_hat.detach(), "sd_hat": sd_hat.detach(),
            "n_unannotated": Bu,
        }
        return loss, stats

    def report(self, patch, classes):
        """Human-readable constraint status from the current EMA state."""
        st, tg = self.state[patch], self.target[patch]
        sd = torch.sqrt((st["e2"] - st["mu"] ** 2).clamp(min=self.eps))
        z_mu = torch.maximum(st["mu"] - (tg["mu"] + tg["tol_mu"]), (tg["mu"] - tg["tol_mu"]) - st["mu"])
        z_sd = torch.maximum(sd - (tg["sd"] + tg["tol_sd"]), (tg["sd"] - tg["tol_sd"]) - sd)
        act = self.active[patch]
        lines = []
        for i, c in enumerate(classes):
            for name, sl in DESC_SLICES.items():
                a = act[i, sl]
                if not bool(a.any()):
                    lines.append(f"      {patch} c{c} {name:<8} (disabled: no population spread)")
                    continue
                zm = float(z_mu[i, sl][a].max())
                zs = float(z_sd[i, sl][a].max())
                ok_m = "ok " if zm <= 0 else "VIOL"
                ok_s = "ok " if zs <= 0 else "VIOL"
                ratio = float((sd[i, sl][a] / tg["sd"][i, sl][a]).mean())
                lines.append(f"      {patch} c{c} {name:<8} mean {ok_m} z={zm:+.4f} | "
                             f"std {ok_s} z={zs:+.4f} | sd/bank={ratio:.2f}")
        return lines


def make_loader(cfg, stems, train: bool, sampler=None, fixed_center=None, annotated_stems=None):
    """
    DataLoader for the FIXED 1-STD + 1-EXP patch per case experiment.

    Each case contributes exactly one STD patch and one EXP patch which
    remain fixed during the whole training. Therefore we iterate over
    cases directly instead of using RandomSampler with replacement.
    """

    expand_factor = float(cfg.get("expand_factor", 1.25))
    fg_prob = float(cfg.get("fg_sampling_prob", 0.5))

    ds = OAIPairedPatch(
        cfg["images_dir"],
        cfg["labels_dir"],
        stems,
        cfg["patch_size"],
        expand_factor,
        fg_prob,
        train,
        fixed_center=fixed_center,
        annotated_stems=annotated_stems,
    )

    loader = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=(train and sampler is None),
        sampler=sampler,
        num_workers=int(cfg["num_workers"]),
        pin_memory=False,
        drop_last=train,
    )

    return loader


def detach_channels_for_seg_loss(logits: torch.Tensor, channel_indices) -> torch.Tensor:
    """
    Detach one or more logit channels so seg loss does NOT backprop through them.
    logits: (B,C,D,H,W)
    channel_indices: list[int]
    """
    if not channel_indices:
        return logits
    x = logits.clone()
    for c in channel_indices:
        x[:, c] = x[:, c].detach()
    return x


# log barrier

class LogBarrierLoss:
    """
    Log-barrier extension from constrained_cnn style:
      psi_t(z) =
          -log(-z)/t                          if z <= -1/t^2
          t*z - log(1/t^2)/t + 1/t           otherwise

    Assumes constraints are written as z <= 0.
    """

    def __init__(self, t: float):
        self.t = float(t)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        z_ = z.flatten()

        barrier_part = -torch.log(-z_) / self.t
        barrier_part = torch.where(torch.isfinite(barrier_part), barrier_part, torch.zeros_like(barrier_part))

        linear_part = self.t * z_ + (-np.log(1 / (self.t ** 2)) / self.t) + (1 / self.t)

        below_threshold = z_ <= (-1 / (self.t ** 2))
        res = torch.where(below_threshold, barrier_part, linear_part)

        res = torch.where(torch.isfinite(res), res, torch.zeros_like(res))
        return res.mean()


# descriptor constraints via log barrier

def _sample_tol(tol, B, device):
    """
    Tolerance as a per-sample (B,1) tensor.

    A batch can mix cases supervised by their own GT (tight band) with cases
    supervised by a population mean (which needs a band wide enough to cover the
    anatomical spread). A single scalar would hand the bank-supervised cases an
    infeasible constraint, so the caller may pass a per-sample vector instead.
    """
    if torch.is_tensor(tol):
        return tol.to(device=device, dtype=torch.float32).view(-1, 1)
    return torch.full((B, 1), float(tol), device=device, dtype=torch.float32)


def _bank_valid(valid, use_bank):
    """
    Voxel-validity mask, widened for samples supervised by the bank.

    Normally `valid` marks voxels whose label is usable. A bank-supervised case
    has no usable label at all, but its IMAGE is entirely real — the descriptor
    is compared against a population value over the whole patch. Without this the
    mask would be all-zero for such a case and the volume denominator would
    collapse to eps.
    """
    if use_bank is None:
        return valid
    ub = use_bank.view(-1, 1, 1, 1, 1).to(valid.dtype)
    return torch.clamp(valid + ub, max=1.0)


def compute_volume_barrier(
    logits,
    target,
    n_classes,
    volume_classes,
    barrier: LogBarrierLoss,
    volume_tolerance=0.10,
    ignore_index=-1,
    eps=1e-6,
    use_bank=None,
    bank_value=None,
    return_stats=False,
    log_space=False,
):
    """
    Enforce:
        lower <= pred_frac <= upper
    where
        lower = gt_frac * (1 - volume_tolerance)
        upper = gt_frac * (1 + volume_tolerance)

    Constraints are converted to:
        pred_frac - upper <= 0
        lower - pred_frac <= 0
    """
    device = logits.device
    probs = F.softmax(logits.float(), dim=1)

    valid = (target != ignore_index).float().unsqueeze(1)  # (B,1,D,H,W)
    valid = _bank_valid(valid, use_bank)
    probs = probs * valid

    one_hot, _ = one_hot_labels(target, n_classes, ignore_index)
    one_hot = one_hot.to(device) * valid

    # per-sample (batch dim preserved), matching centroid/avgdist/moment barriers —
    # pooling across the batch here would let one badly-sized case hide behind
    # others in the same step, and would make the constraint's precision depend
    # on batch size instead of being batch-size-invariant
    dims = (2, 3, 4)
    pred_mass = probs.sum(dim=dims)       # (B,C)
    gt_mass   = one_hot.sum(dim=dims)     # (B,C)
    tot_mass  = valid.sum(dim=dims) + eps # (B,1)

    pred_frac = pred_mass / tot_mass
    gt_frac   = gt_mass / tot_mass

    cls_idx = torch.as_tensor(volume_classes, device=device, dtype=torch.long)
    pred_sel = pred_frac[:, cls_idx]
    gt_sel   = gt_frac[:, cls_idx]

    if use_bank is not None:
        bank_sel = bank_value.to(device=device, dtype=gt_sel.dtype).view(1, -1)
        gt_sel = torch.where(use_bank.view(-1, 1), bank_sel.expand_as(gt_sel), gt_sel)

    vtol = _sample_tol(volume_tolerance, gt_sel.shape[0], device)

    if log_space:
        # Scale-free form. A RELATIVE band of `tol` becomes an ABSOLUTE band of
        # log(1+tol) in log space -- identical for every class. The linear form's
        # band is 0.10*gt_frac, which spans 115x across these classes (0.0002 for
        # tibial cartilage on the EXP patch up to 0.023 for femoral bone), so it
        # lands on either side of the barrier's dead-zone threshold 1/t^2 purely
        # according to how big the structure is. In that dead zone both sides of
        # the two-sided constraint sit in the linear extension and their
        # gradients cancel exactly, so the constraint contributes nothing no
        # matter how wrong it is. Here one t serves every class (live for any
        # t >= 1/sqrt(log(1+tol)) = 3.2 at tol=0.10).
        present = gt_sel > 0
        if not bool(present.any()):
            zero = logits.new_tensor(0.0)
            return (zero, logits.new_zeros(len(volume_classes))) if return_stats else zero
        log_pred = torch.log(pred_sel.clamp(min=eps))
        log_gt = torch.log(gt_sel.clamp(min=eps))
        band = torch.log1p(vtol)          # symmetric in log space -> multiplicative
        z_upper = (log_pred - (log_gt + band))[present]
        z_lower = ((log_gt - band) - log_pred)[present]
    else:
        lower = gt_sel * (1.0 - vtol)
        upper = gt_sel * (1.0 + vtol)

        z_upper = pred_sel - upper   # <= 0 wanted
        z_lower = lower - pred_sel   # <= 0 wanted

    loss = barrier(z_upper.reshape(-1)) + barrier(z_lower.reshape(-1))
    if not return_stats:
        return loss

    # Achieved error, RELATIVE, so it is directly comparable to volume_tolerance.
    # err/tol << 1 means the constraint is satisfied with slack to spare and the
    # tolerance is doing no work; err/tol > 1 means it is still binding.
    with torch.no_grad():
        present = gt_sel > 0
        rel = (pred_sel - gt_sel).abs() / gt_sel.clamp(min=eps)
        per_class = torch.where(
            present.any(dim=0),
            (rel * present).sum(dim=0) / present.sum(dim=0).clamp(min=1),
            torch.zeros_like(rel[0]),
        )
    return loss, per_class


def compute_centroid_barrier(
    logits,
    target,
    n_classes,
    centroid_class,
    barrier: LogBarrierLoss,
    centroid_tolerance=0.05,
    centroid_norm=True,
    ignore_index=-1,
    eps=1e-6,
    use_bank=None,
    bank_value=None,
    return_stats=False,
):
    """
    Enforce:
        gt_centroid - tol <= pred_centroid <= gt_centroid + tol

    Constraints:
        pred_centroid - (gt_centroid + tol) <= 0
        (gt_centroid - tol) - pred_centroid <= 0
    """
    device = logits.device
    probs = F.softmax(logits.float(), dim=1)
    valid = (target != ignore_index).float().unsqueeze(1)
    valid = _bank_valid(valid, use_bank)

    one_hot, _ = one_hot_labels(target, n_classes, ignore_index)
    one_hot = one_hot.to(device)

    pred_w  = probs[:, centroid_class:centroid_class+1] * valid
    gt_mask = one_hot[:, centroid_class:centroid_class+1] * valid

    B, _, D, H, W = pred_w.shape

    if centroid_norm:
        zz = torch.linspace(0, 1, D, device=device).view(1, 1, D, 1, 1)
        yy = torch.linspace(0, 1, H, device=device).view(1, 1, 1, H, 1)
        xx = torch.linspace(0, 1, W, device=device).view(1, 1, 1, 1, W)
    else:
        zz = torch.arange(D, device=device).float().view(1, 1, D, 1, 1)
        yy = torch.arange(H, device=device).float().view(1, 1, 1, H, 1)
        xx = torch.arange(W, device=device).float().view(1, 1, 1, 1, W)

    coords = torch.cat(
        [
            zz.expand(B, 1, D, H, W),
            yy.expand(B, 1, D, H, W),
            xx.expand(B, 1, D, H, W),
        ],
        dim=1,
    )  # (B,3,D,H,W)

    pred_sum = pred_w.sum(dim=(2, 3, 4), keepdim=True) + eps
    gt_sum   = gt_mask.sum(dim=(2, 3, 4), keepdim=True) + eps

    pred_centroid = (pred_w * coords).sum(dim=(2, 3, 4)) / pred_sum.squeeze(-1).squeeze(-1).squeeze(-1)
    gt_centroid   = (gt_mask * coords).sum(dim=(2, 3, 4)) / gt_sum.squeeze(-1).squeeze(-1).squeeze(-1)

    if use_bank is not None:
        bank_c = bank_value.to(device=device, dtype=gt_centroid.dtype).view(1, 3)
        gt_centroid = torch.where(use_bank.view(-1, 1), bank_c.expand_as(gt_centroid), gt_centroid)

    has_class = (gt_mask.sum(dim=(2, 3, 4)) > 0).squeeze(1)
    if use_bank is not None:
        # a bank-supervised sample is constrained even though its GT is unusable
        has_class = has_class | use_bank
    if not has_class.any():
        return (logits.new_tensor(0.0), logits.new_tensor(0.0)) if return_stats else logits.new_tensor(0.0)

    pred_c = pred_centroid[has_class]
    gt_c   = gt_centroid[has_class]
    ctol   = _sample_tol(centroid_tolerance, gt_centroid.shape[0], device)[has_class]

    upper = gt_c + ctol
    lower = gt_c - ctol

    z_upper = pred_c - upper     # <= 0 wanted
    z_lower = lower - pred_c     # <= 0 wanted

    loss = barrier(z_upper.reshape(-1)) + barrier(z_lower.reshape(-1))
    if not return_stats:
        return loss
    with torch.no_grad():
        err = (pred_c - gt_c).abs().mean()   # patch units, same as centroid_tolerance
    return loss, err


def compute_avgdist_barrier(
    logits,
    target,
    n_classes,
    distance_class,
    barrier: LogBarrierLoss,
    avgdist_tolerance=0.10,
    centroid_norm=True,
    ignore_index=-1,
    eps=1e-6,
    use_bank=None,
    bank_value=None,
):
    """
    Enforce:
        gt_avgdist - tol <= pred_avgdist <= gt_avgdist + tol

    where avgdist is the mean Euclidean distance to the class centroid.
    """
    device = logits.device
    probs = F.softmax(logits.float(), dim=1)
    valid = (target != ignore_index).float().unsqueeze(1)
    valid = _bank_valid(valid, use_bank)

    one_hot, _ = one_hot_labels(target, n_classes, ignore_index)
    one_hot = one_hot.to(device)

    pred_w = probs[:, distance_class:distance_class+1] * valid
    gt_mask = one_hot[:, distance_class:distance_class+1] * valid

    B, _, D, H, W = pred_w.shape

    if centroid_norm:
        zz = torch.linspace(0, 1, D, device=device).view(1, 1, D, 1, 1)
        yy = torch.linspace(0, 1, H, device=device).view(1, 1, 1, H, 1)
        xx = torch.linspace(0, 1, W, device=device).view(1, 1, 1, 1, W)
    else:
        zz = torch.arange(D, device=device).float().view(1, 1, D, 1, 1)
        yy = torch.arange(H, device=device).float().view(1, 1, 1, H, 1)
        xx = torch.arange(W, device=device).float().view(1, 1, 1, 1, W)

    coords = torch.cat(
        [
            zz.expand(B, 1, D, H, W),
            yy.expand(B, 1, D, H, W),
            xx.expand(B, 1, D, H, W),
        ],
        dim=1,
    )

    pred_sum = pred_w.sum(dim=(2, 3, 4), keepdim=True) + eps
    gt_sum = gt_mask.sum(dim=(2, 3, 4), keepdim=True) + eps

    pred_centroid = (pred_w * coords).sum(dim=(2, 3, 4), keepdim=True) / pred_sum
    gt_centroid = (gt_mask * coords).sum(dim=(2, 3, 4), keepdim=True) / gt_sum

    pred_dist = torch.linalg.norm(coords - pred_centroid, dim=1, keepdim=True)
    gt_dist = torch.linalg.norm(coords - gt_centroid, dim=1, keepdim=True)

    pred_avgdist = (pred_w * pred_dist).sum(dim=(2, 3, 4)) / pred_sum.squeeze(-1).squeeze(-1).squeeze(-1)
    gt_avgdist = (gt_mask * gt_dist).sum(dim=(2, 3, 4)) / gt_sum.squeeze(-1).squeeze(-1).squeeze(-1)

    if use_bank is not None:
        bank_d = bank_value.to(device=device, dtype=gt_avgdist.dtype).view(1, 1)
        gt_avgdist = torch.where(use_bank.view(-1, 1), bank_d.expand_as(gt_avgdist), gt_avgdist)

    has_class = (gt_mask.sum(dim=(2, 3, 4)) > 0).squeeze(1)
    if use_bank is not None:
        has_class = has_class | use_bank
    if not has_class.any():
        return logits.new_tensor(0.0)

    pred_d = pred_avgdist[has_class]
    gt_d = gt_avgdist[has_class]
    dtol = _sample_tol(avgdist_tolerance, gt_avgdist.shape[0], device)[has_class]

    upper = gt_d + dtol
    lower = gt_d - dtol

    z_upper = pred_d - upper
    z_lower = lower - pred_d

    return barrier(z_upper.reshape(-1)) + barrier(z_lower.reshape(-1))


def compute_avgdist_axis_barrier(
    logits,
    target,
    n_classes,
    distance_class,
    barrier: LogBarrierLoss,
    avgdist_axis_tolerance=0.05,
    centroid_norm=True,
    ignore_index=-1,
    eps=1e-6,
    return_stats=False,
    use_bank=None,
    bank_value=None,
):
    """
    Enforce:
        gt_axis_spread - tol <= pred_axis_spread <= gt_axis_spread + tol

    where axis_spread is computed per axis as sqrt(E[(coord - centroid_coord)^2]).
    """
    device = logits.device
    probs = F.softmax(logits.float(), dim=1)
    valid = (target != ignore_index).float().unsqueeze(1)
    valid = _bank_valid(valid, use_bank)

    one_hot, _ = one_hot_labels(target, n_classes, ignore_index)
    one_hot = one_hot.to(device)

    pred_w = probs[:, distance_class:distance_class+1] * valid
    gt_mask = one_hot[:, distance_class:distance_class+1] * valid

    B, _, D, H, W = pred_w.shape

    if centroid_norm:
        zz = torch.linspace(0, 1, D, device=device).view(1, 1, D, 1, 1)
        yy = torch.linspace(0, 1, H, device=device).view(1, 1, 1, H, 1)
        xx = torch.linspace(0, 1, W, device=device).view(1, 1, 1, 1, W)
    else:
        zz = torch.arange(D, device=device).float().view(1, 1, D, 1, 1)
        yy = torch.arange(H, device=device).float().view(1, 1, 1, H, 1)
        xx = torch.arange(W, device=device).float().view(1, 1, 1, 1, W)

    grids = [zz.expand(B, 1, D, H, W), yy.expand(B, 1, D, H, W), xx.expand(B, 1, D, H, W)]

    pred_sum = pred_w.sum(dim=(2, 3, 4), keepdim=True) + eps
    gt_sum = gt_mask.sum(dim=(2, 3, 4), keepdim=True) + eps

    pred_centroids = [(pred_w * axis_grid).sum(dim=(2, 3, 4), keepdim=True) / pred_sum for axis_grid in grids]
    gt_centroids = [(gt_mask * axis_grid).sum(dim=(2, 3, 4), keepdim=True) / gt_sum for axis_grid in grids]

    has_class = (gt_mask.sum(dim=(2, 3, 4)) > 0).squeeze(1)
    if use_bank is not None:
        has_class = has_class | use_bank
    if not has_class.any():
        if return_stats:
            return logits.new_tensor(0.0), logits.new_zeros(3)
        return logits.new_tensor(0.0)

    barrier_loss = logits.new_tensor(0.0)
    axis_stats = []
    for axis_i, (axis_grid, pred_c_axis, gt_c_axis) in enumerate(zip(grids, pred_centroids, gt_centroids)):
        pred_axis_var = (pred_w * (axis_grid - pred_c_axis) ** 2).sum(dim=(2, 3, 4)) / pred_sum.squeeze(-1).squeeze(-1).squeeze(-1)
        gt_axis_var = (gt_mask * (axis_grid - gt_c_axis) ** 2).sum(dim=(2, 3, 4)) / gt_sum.squeeze(-1).squeeze(-1).squeeze(-1)
        gt_s_full = torch.sqrt(gt_axis_var + eps)
        if use_bank is not None:
            bank_s = bank_value.to(device=device, dtype=gt_s_full.dtype).view(1, -1)[:, axis_i:axis_i + 1]
            gt_s_full = torch.where(use_bank.view(-1, 1), bank_s.expand_as(gt_s_full), gt_s_full)

        pred_s = torch.sqrt(pred_axis_var + eps)[has_class]
        gt_s = gt_s_full[has_class]
        atol = _sample_tol(avgdist_axis_tolerance, gt_s_full.shape[0], device)[has_class]

        upper = gt_s + atol
        lower = gt_s - atol

        z_upper = pred_s - upper
        z_lower = lower - pred_s
        barrier_loss = barrier_loss + barrier(z_upper.reshape(-1)) + barrier(z_lower.reshape(-1))

        if return_stats:
            axis_stats.append((pred_s - gt_s).abs().mean())

        del pred_axis_var
        del gt_axis_var
        del pred_s
        del gt_s
        del z_upper
        del z_lower

    if return_stats:
        return barrier_loss, torch.stack(axis_stats)
    return barrier_loss


# plotting

def plot_progress(workdir: Path):
    csv_path = workdir / "progress.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    if len(df) < 2:
        return

    if "meanFGDice" in df.columns:
        df["meanFGDice_ma"] = df["meanFGDice"].rolling(window=10, min_periods=1).mean()

    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(df["epoch"], df["loss_tr_total"], label="train total")
    if "loss_va_seg" in df.columns:
        ax1.plot(df["epoch"], df["loss_va_seg"], label="val seg")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")

    ax1b = ax1.twinx()
    if "meanFGDice" in df.columns:
        ax1b.plot(df["epoch"], df["meanFGDice"], linestyle="dotted", label="meanFGDice")
        ax1b.plot(df["epoch"], df["meanFGDice_ma"], label="meanFGDice (mov.avg.)")
        ax1b.set_ylabel("pseudo dice")
        ax1b.legend(loc="lower right")

    ax2 = fig.add_subplot(3, 1, 2)
    dice_cols = [c for c in df.columns if c.startswith("dice_c")]
    for c in dice_cols:
        ax2.plot(df["epoch"], df[c], label=c)
    ax2.set_ylabel("per-class pseudo dice")
    ax2.legend(loc="lower right", ncol=2)

    ax3 = fig.add_subplot(3, 1, 3)
    if "lr" in df.columns:
        ax3.plot(df["epoch"], df["lr"], label="lr")
    ax3.set_ylabel("lr")
    ax3.set_xlabel("epoch")
    ax3.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(workdir / "progress.png", dpi=150)
    plt.close(fig)


#main

def main(cfg_path: str):
    # DDP init — falls back gracefully to single-GPU when launched without torchrun
    use_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    is_main = (rank == 0)

    with open(cfg_path, "r") as f:
        cfg = json.load(f) if cfg_path.endswith(".json") else __import__("yaml").safe_load(f)

    workdir = Path(cfg["workdir"])
    if is_main:
        workdir.mkdir(parents=True, exist_ok=True)

    if use_ddp:
        dist.barrier()

    if is_main:
        with open(workdir / "config_snapshot.yaml", "w") as f:
            __import__("yaml").safe_dump(cfg, f)

    if is_main:
        print("=== Running LOG-BARRIER with FIXED 1 STD + 1 EXP patch experiment ===")

    # wandb (rank 0 only)
    use_wandb = bool(cfg.get("use_wandb", False))
    if use_wandb and not WANDB_AVAILABLE:
        if is_main:
            print("Warning: use_wandb=True but wandb is not installed — disabling.")
        use_wandb = False
    if use_wandb and is_main:
        wandb.init(
            project=cfg.get("wandb_project", "meunet-shape-supervision"),
            entity=cfg.get("wandb_entity", None),
            name=cfg.get("wandb_run_name", Path(cfg["workdir"]).name),
            config=cfg,
            dir=str(workdir),
        )

    # Each rank gets a different seed offset so augmentation is not identical
    seed_all(int(cfg.get("seed", 777)) + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main:
        print("Device:", device, f"| world_size={world_size}")

    # splits
    splits = load_splits(cfg)
    fold = int(cfg.get("fold", 0))
    split = splits[fold] if isinstance(splits, list) else splits

    if isinstance(split, dict):
        tr_stems = split["train"]
        va_stems = split["val"]
    elif isinstance(split, (list, tuple)) and len(split) >= 2:
        tr_stems, va_stems = split[0], split[1]
    else:
        raise ValueError(f"Unexpected split format: {type(split)} with value {str(split)[:200]}")
    if is_main:
        print(f"Fold {fold}: train={len(tr_stems)} val={len(va_stems)}")

    # ------------------------------------------------------------------
    # weak-annotation ablation (both toggles default to the legacy path)
    #
    #   annotated_fraction < 1.0 -> train on that fraction of the fold's train
    #     stems only. At this stage the withheld cases are simply dropped; the
    #     descriptor-bank arm that feeds them population targets comes later.
    #
    #   fixed_center -> use one shared image-independent patch centre for every
    #     case instead of each case's GT foreground centre-of-mass. Required for
    #     any arm involving unannotated cases (their label is unavailable, so a
    #     GT-derived centre cannot be computed), and it also removes the GT leak
    #     that GT centering introduces into the centroid descriptor. Validation
    #     uses the same centre so train/val patches are framed identically.
    # ------------------------------------------------------------------
    annotated_fraction = float(cfg.get("annotated_fraction", 1.0))
    annotation_seed = int(cfg.get("annotation_seed", cfg.get("seed", 777)))
    unannotated_stems = []
    if annotated_fraction < 1.0:
        tr_stems, unannotated_stems = subsample_annotated(
            tr_stems, annotated_fraction, annotation_seed
        )
        if is_main:
            print(
                f"[weak] annotated_fraction={annotated_fraction} seed={annotation_seed}: "
                f"train {len(tr_stems)} annotated / {len(unannotated_stems)} withheld"
            )
            with open(workdir / "annotation_split.json", "w") as f:
                json.dump(
                    {
                        "fold": fold,
                        "annotated_fraction": annotated_fraction,
                        "annotation_seed": annotation_seed,
                        "annotated": list(tr_stems),
                        "unannotated": list(unannotated_stems),
                    },
                    f,
                    indent=2,
                )

    # How the withheld cases are used (must be decided before the loaders exist):
    #   drop         -> not loaded at all (A1.x)
    #   bank         -> loaded, each given the population mean as its own target (A2)
    #   distribution -> loaded, constrained only in aggregate (A1.4)
    unannotated_mode = str(cfg.get("unannotated_mode", "drop")).lower()
    if unannotated_mode not in ("drop", "bank", "distribution"):
        raise ValueError(f"unannotated_mode must be drop|bank|distribution, got {unannotated_mode!r}")
    if unannotated_mode != "drop" and not unannotated_stems:
        raise ValueError(f"unannotated_mode={unannotated_mode} requires annotated_fraction < 1.0")

    annotated_stems = list(tr_stems)
    if unannotated_mode != "drop":
        tr_stems = sorted(list(tr_stems) + list(unannotated_stems))
        if is_main:
            print(f"[weak] unannotated_mode={unannotated_mode}: training on {len(tr_stems)} cases "
                  f"({len(annotated_stems)} with GT, {len(unannotated_stems)} label-withheld)")

    fixed_center = cfg.get("fixed_center", None)
    if isinstance(fixed_center, str):
        if fixed_center.lower() != "auto":
            raise ValueError(f"fixed_center must be a 3-element list or 'auto', got {fixed_center!r}")
        # Resolved from the ANNOTATED train stems only, then shared with val so
        # every patch in the run is framed by the same rule.
        fixed_center = compute_average_center(cfg["images_dir"], cfg["labels_dir"], annotated_stems)
        if is_main:
            print(f"[weak] fixed_center: auto -> {fixed_center} (from {len(annotated_stems)} annotated cases)")
    elif fixed_center is not None:
        fixed_center = tuple(int(round(float(v))) for v in fixed_center)
        if is_main:
            print(f"[weak] fixed_center: {fixed_center}")

    if is_main and fixed_center is not None:
        with open(workdir / "fixed_center.json", "w") as f:
            json.dump({"fixed_center": list(fixed_center)}, f, indent=2)

    expand_factor = float(cfg.get("expand_factor", 1.25))
    fg_prob = float(cfg.get("fg_sampling_prob", 0.5))

    if use_ddp:
        # Build datasets up front to pass to DistributedSampler
        from data.dataset_oai_raw_fixedpatch import OAIPairedPatch
        tr_ds = OAIPairedPatch(cfg["images_dir"], cfg["labels_dir"], tr_stems,
                               cfg["patch_size"], expand_factor, fg_prob, True,
                               fixed_center=fixed_center,
                               annotated_stems=(annotated_stems if unannotated_mode != "drop" else None))
        va_ds = OAIPairedPatch(cfg["images_dir"], cfg["labels_dir"], va_stems,
                               cfg["patch_size"], expand_factor, fg_prob, False,
                               fixed_center=fixed_center)
        train_sampler = DistributedSampler(tr_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler   = DistributedSampler(va_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        train_loader  = DataLoader(tr_ds, batch_size=int(cfg["batch_size"]), sampler=train_sampler,
                                   num_workers=int(cfg["num_workers"]), pin_memory=False, drop_last=True)
        val_loader    = DataLoader(va_ds, batch_size=int(cfg["batch_size"]), sampler=val_sampler,
                                   num_workers=int(cfg["num_workers"]), pin_memory=False, drop_last=False)
    else:
        train_sampler = None
        train_loader  = make_loader(cfg, tr_stems, train=True, fixed_center=fixed_center,
                                    annotated_stems=(annotated_stems if unannotated_mode != "drop" else None))
        val_loader    = make_loader(cfg, va_stems, train=False, fixed_center=fixed_center)

    # model
    model = MEUNet3D(
        cfg["in_channels"],
        cfg["n_classes"],
        cfg["enc_channels"],
        cfg["dec_channels"],
        cfg["norm"],
    ).to(device)

    if use_ddp:
        # Sync BN statistics across GPUs — critical with small per-rank batch sizes
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # MEUNet3D has two output heads (logit1/logit2); only one is used per step
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # loss for monitoring (and optionally training)
    crit = DiceCELoss(cfg["n_classes"])

    base_lr = float(cfg.get("lr", 9e-4))
    # how to scale LR for the larger effective (gradient-averaged) batch under DDP:
    #   "sqrt"   (default, backward compatible): base_lr * sqrt(world_size)
    #   "linear": base_lr * world_size  — matches the effective-batch growth exactly
    #   "none":   base_lr unchanged regardless of world_size
    lr_scaling = str(cfg.get("lr_scaling", "sqrt")).lower()
    if not use_ddp:
        effective_lr = base_lr
    elif lr_scaling == "linear":
        effective_lr = base_lr * world_size
    elif lr_scaling == "none":
        effective_lr = base_lr
    else:
        effective_lr = base_lr * (world_size ** 0.5)
    if is_main and use_ddp:
        print(f"LR scaled: {base_lr:.2e} -> {effective_lr:.2e} ({lr_scaling} rule, world_size={world_size})")

    opt = torch.optim.Adam(
        model.parameters(),
        lr=effective_lr,
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("amp", True))

    # descriptor config
    desc_classes = as_int_list(cfg.get("desc_only_classes", cfg.get("desc_only_class", [1])))
    volume_classes = as_int_list(cfg.get("volume_classes", desc_classes), default=desc_classes)
    centroid_classes = as_int_list(cfg.get("centroid_classes", cfg.get("centroid_class", desc_classes)), default=desc_classes)
    avgdist_classes = as_int_list(cfg.get("avgdist_classes", cfg.get("distance_classes", desc_classes)), default=desc_classes)
    avgdist_axis_classes = as_int_list(cfg.get("avgdist_axis_classes", cfg.get("distance_axis_classes", desc_classes)), default=desc_classes)
    moment2_classes = as_int_list(cfg.get("moment2_classes", desc_classes), default=desc_classes)
    moment3_classes = as_int_list(cfg.get("moment3_classes", desc_classes), default=desc_classes)
    moment_inv_classes = as_int_list(cfg.get("moment_inv_classes", desc_classes), default=desc_classes)

    shape_on = cfg.get("shape_on", "exp")  # "std" | "exp" | "both"

    lambda_volume = float(cfg.get("lambda_volume", 1.0))
    lambda_centroid = float(cfg.get("lambda_centroid", 1.0))
    lambda_avgdist = float(cfg.get("lambda_avgdist", cfg.get("lambda_distance", 0.0)))
    lambda_avgdist_axis = float(cfg.get("lambda_avgdist_axis", cfg.get("lambda_distance_axis", 0.0)))
    lambda_moment2 = float(cfg.get("lambda_moment2", 0.0))
    lambda_moment3 = float(cfg.get("lambda_moment3", 0.0))
    # separate weight for the off-diagonal (cross-covariance / mixed-skew) terms —
    # defaults to 0.0 so existing configs are unaffected unless explicitly opted in
    lambda_moment2_offdiag = float(cfg.get("lambda_moment2_offdiag", 0.0))
    lambda_moment3_offdiag = float(cfg.get("lambda_moment3_offdiag", 0.0))
    lambda_moment_inv_J1 = float(cfg.get("lambda_moment_inv_J1", 1.0))
    lambda_moment_inv_J2 = float(cfg.get("lambda_moment_inv_J2", 1.0))
    lambda_moment_inv_J3 = float(cfg.get("lambda_moment_inv_J3", 1.0))
    # linear warmup for moment3: ramp from 0 -> lambda_moment3 between these epochs
    moment3_warmup_start = int(cfg.get("moment3_warmup_start", 0))
    moment3_warmup_end   = int(cfg.get("moment3_warmup_end",   0))  # 0 = no warmup

    centroid_norm = bool(cfg.get("centroid_norm", True))

    # barrier config
    barrier_t = float(cfg.get("barrier_t", 5.0))
    # per-epoch tightening (Kervadec et al. 1904.04205 style): t <- min(t0 * mu^epoch, t_max).
    # Scoped to volume/centroid/avgdist/avgdist_axis only — the moment terms (esp.
    # off-diagonal) already sit deep in the barrier's linear-extension region at
    # small t given their tiny natural scale, so sharing a growing t with them
    # would just inflate their (constant, always-active) linear-branch gradient
    # rather than meaningfully tighten a log-region cutoff they never reach.
    barrier_t_mu = float(cfg.get("barrier_t_mu", 1.0))     # 1.0 = no growth (backward compatible)
    barrier_t_max = float(cfg.get("barrier_t_max", 100.0))
    volume_tolerance = float(cfg.get("volume_tolerance", 0.10))
    centroid_tolerance = float(cfg.get("centroid_tolerance", 0.05))
    avgdist_tolerance = float(cfg.get("avgdist_tolerance", 0.05))
    avgdist_axis_tolerance = float(cfg.get("avgdist_axis_tolerance", 0.05))
    moment2_tolerance = float(cfg.get("moment2_tolerance", 0.02))
    moment3_tolerance = float(cfg.get("moment3_tolerance", 0.01))
    # off-diagonal terms are raw covariance/mixed-skew (much smaller natural scale
    # than the sqrt'd diagonal terms) — default to the diag tolerance for
    # backward compatibility, but override per-experiment to actually engage them
    moment2_offdiag_tolerance = float(cfg.get("moment2_offdiag_tolerance", moment2_tolerance))
    moment3_offdiag_tolerance = float(cfg.get("moment3_offdiag_tolerance", moment3_tolerance))
    moment_inv_J1_tolerance = float(cfg.get("moment_inv_J1_tolerance", 0.01))
    moment_inv_J2_tolerance = float(cfg.get("moment_inv_J2_tolerance", 1e-3))
    moment_inv_J3_tolerance = float(cfg.get("moment_inv_J3_tolerance", 1e-4))
    moment2_sqrt_diagonal = bool(cfg.get("moment2_sqrt_diagonal", True))
    moment3_sqrt_diagonal = bool(cfg.get("moment3_sqrt_diagonal", True))
    # "gamma1" (default): weighted average, tractable gradients, no spreading artefacts
    # "fractional": fractional mass normalization, scale-invariant across resolutions
    _moment_norm = cfg.get("moment_normalization", "gamma1")
    moment2_gamma = 5/3 if _moment_norm == "fractional" else 1.0
    moment3_gamma = 2.0  if _moment_norm == "fractional" else 1.0
    moment_inv_gamma = 5/3 if _moment_norm == "fractional" else 1.0
    barrier = LogBarrierLoss(t=barrier_t)              # static — legacy fallback for moment terms
    barrier_shape = LogBarrierLoss(t=barrier_t)         # scheduled — volume/centroid/avgdist/avgdist_axis

    # ---- per-term barrier sharpness for the moment family -------------------
    # The log/linear switch of LogBarrierLoss sits at |z| = 1/t^2. When 1/t^2 > tol
    # the log branch is unreachable, and the upper/lower linear branches (slopes +t
    # and -t) cancel exactly, leaving a zero-gradient dead zone of half-width
    # (1/t^2 - tol) around the target. With the legacy shared t=5 that dead zone is
    # ~0.04 wide, which swallows every moment-scale quantity here (tolerances 5e-2
    # down to 1e-5), so those terms contributed no gradient at all.
    #
    # "auto" sizes each term's t from its own tolerance so 1/t^2 = margin_frac * tol,
    # keeping the switch a fixed fraction inside the band. Scoped to moment2/moment3/
    # moment_inv only: volume/centroid/avgdist keep barrier_shape at barrier_t so the
    # descriptor ablations stay comparable to earlier runs.
    moment_barrier_t_mode = str(cfg.get("moment_barrier_t_mode", "auto")).lower()
    moment_barrier_margin_frac = float(cfg.get("moment_barrier_margin_frac", 0.1))
    moment_barrier_t_max = float(cfg.get("moment_barrier_t_max", 1.0e4))

    def _moment_barrier(name, tol):
        """Barrier for one moment term; explicit override > formula > legacy static."""
        override = cfg.get(f"moment_barrier_t_{name}", None)
        if override is not None:
            return LogBarrierLoss(t=float(override))
        if moment_barrier_t_mode != "auto":
            return barrier
        if tol is None or tol <= 0.0:
            return barrier
        t_auto = min(1.0 / math.sqrt(moment_barrier_margin_frac * tol), moment_barrier_t_max)
        return LogBarrierLoss(t=t_auto)

    barrier_m2_diag    = _moment_barrier("m2_diag",    moment2_tolerance)
    barrier_m2_offdiag = _moment_barrier("m2_offdiag", moment2_offdiag_tolerance)
    barrier_m3_diag    = _moment_barrier("m3_diag",    moment3_tolerance)
    barrier_m3_offdiag = _moment_barrier("m3_offdiag", moment3_offdiag_tolerance)
    barrier_J1         = _moment_barrier("J1",         moment_inv_J1_tolerance)
    barrier_J2         = _moment_barrier("J2",         moment_inv_J2_tolerance)
    barrier_J3         = _moment_barrier("J3",         moment_inv_J3_tolerance)

    _moment_barrier_ts = {
        "m2_diag": barrier_m2_diag.t, "m2_offdiag": barrier_m2_offdiag.t,
        "m3_diag": barrier_m3_diag.t, "m3_offdiag": barrier_m3_offdiag.t,
        "J1": barrier_J1.t, "J2": barrier_J2.t, "J3": barrier_J3.t,
    }
    _moment_barrier_tols = {
        "m2_diag": moment2_tolerance, "m2_offdiag": moment2_offdiag_tolerance,
        "m3_diag": moment3_tolerance, "m3_offdiag": moment3_offdiag_tolerance,
        "J1": moment_inv_J1_tolerance, "J2": moment_inv_J2_tolerance,
        "J3": moment_inv_J3_tolerance,
    }

    if is_main:
        print(f"[barrier] shape terms (vol/cent/avgdist): t={barrier_t:.2f} "
              f"mu={barrier_t_mu:.3f} t_max={barrier_t_max:.1f} "
              f"({'scheduled' if barrier_t_mu > 1.0 else 'static'}) — unchanged")
        print(f"[barrier] moment terms: mode={moment_barrier_t_mode} "
              f"margin_frac={moment_barrier_margin_frac}")
        for _n, _t in _moment_barrier_ts.items():
            _tol = _moment_barrier_tols[_n]
            _dead = (1.0 / _t ** 2) - _tol
            print(f"           {_n:11s} tol={_tol:.2e}  t={_t:8.2f}  "
                  f"1/t^2={1.0/_t**2:.2e}  "
                  f"{'DEAD ZONE +/-%.2e' % _dead if _dead > 0 else 'ok (log region reachable)'}")

    # training mode switches
    use_seg_loss = bool(cfg.get("use_seg_loss", True))
    monitor_seg_loss = bool(cfg.get("monitor_seg_loss", True))
    detach_for_seg = bool(cfg.get("detach_desc_channels_for_seg", True))

    epochs = int(cfg.get("epochs", 300))
    log_every = int(cfg.get("log_every", 25))
    early_pat = int(cfg.get("early_stop_patience", 40))
    ckpt_every = int(cfg.get("checkpoint_every", 0))

    best_metric = -1.0

    # Matched-compute checkpoint. Arms differ in cases, so at the same EPOCH they
    # have taken different numbers of optimizer steps: 106/epoch at 107 cases vs
    # 356/epoch at 356. budget_epoch is the epoch at which this run has taken the
    # same number of steps as the arm it will be compared against; we keep the
    # best-so-far model up to that point and freeze it there, so the comparison is
    # best-vs-best under equal compute rather than final-vs-final.
    budget_epoch = int(cfg.get("budget_epoch", 0))
    budget_best_metric = -1.0
    budget_best_epoch = -1
    if is_main and budget_epoch > 0:
        print(f"[budget] extra checkpoint: best model among epochs <= {budget_epoch} "
              f"-> checkpoint_budget_best.pt")
    patience = 0

    csv_path = workdir / "progress.csv"
    rows = []
    t0_all = time.time()

    # --- fixed validation-sample visualization (wandb) ---
    # Tracks one val patient's segmentation over training so shape evolution
    # (esp. the small cartilage classes) can be inspected visually alongside
    # the scalar moment metrics. Rank 0 only: it independently indexes into
    # its own copy of the val split and runs a lone forward pass on the fixed
    # patch, so no cross-rank communication is needed — the val DistributedSampler
    # loop is untouched.
    viz_enable = bool(cfg.get("viz_enable", False))
    viz_every_n_epochs = int(cfg.get("viz_every_n_epochs", 10))
    viz_patient_stem = cfg.get("viz_patient_stem", None)  # None -> va_stems[0]
    viz_cartilage_classes = as_int_list(cfg.get("viz_cartilage_classes", moment2_classes), default=moment2_classes)
    viz_min_cartilage_pixels = int(cfg.get("viz_min_cartilage_pixels", 1))
    viz_slice_axis = int(cfg.get("viz_slice_axis", 0))
    viz_class_names = {int(k): v for k, v in cfg.get("viz_class_names", {
        0: "background", 1: "femur", 2: "femoral_cartilage", 3: "tibia", 4: "tibial_cartilage",
    }).items()}

    viz_ready = False
    viz_sample = None
    viz_slice_idx = None
    if viz_enable and is_main:
        if not (use_wandb and WANDB_AVAILABLE):
            print("Warning: viz_enable=True but wandb logging is not active — disabling viz.")
            viz_enable = False
        else:
            viz_stem = viz_patient_stem if viz_patient_stem is not None else va_stems[0]
            if viz_stem not in va_stems:
                print(f"Warning: viz_patient_stem={viz_stem!r} not found in val split — disabling viz.")
                viz_enable = False
            else:
                from data.dataset_oai_raw_fixedpatch import OAIPairedPatch as _OAIPairedPatchViz
                viz_ds = _OAIPairedPatchViz(
                    cfg["images_dir"], cfg["labels_dir"], [viz_stem],
                    cfg["patch_size"], expand_factor, fg_prob, False,
                    fixed_center=fixed_center,
                )
                viz_sample = viz_ds[0]
                viz_slice_idx = select_viz_slice(
                    viz_sample["std_lbl"].numpy(),
                    cartilage_classes=viz_cartilage_classes,
                    axis=viz_slice_axis,
                    min_pixels=viz_min_cartilage_pixels,
                )
                viz_ready = True
                print(f"[viz] Tracking patient={viz_stem} slice={viz_slice_idx} (axis={viz_slice_axis})")

    # ------------------------------------------------------------------
    # descriptor bank (population targets). bank_targets:
    #   none        -> every case uses its own GT (legacy; A0 / A1.0-A1.2)
    #   all         -> every case uses bank targets, GT discarded (A1.3)
    #   unannotated -> only withheld cases use the bank (A2)
    # ------------------------------------------------------------------
    bank_targets = str(cfg.get("bank_targets", "none")).lower()
    if bank_targets not in ("none", "all"):
        raise ValueError(f"bank_targets must be none|all, got {bank_targets!r} "
                         "(use unannotated_mode for how WITHHELD cases are supervised)")

    if unannotated_mode != "drop" and bank_targets != "none":
        raise ValueError("bank_targets and unannotated_mode are alternatives; set bank_targets: none")

    need_bank = (bank_targets != "none") or (unannotated_mode != "drop")
    lambda_dist = float(cfg.get("lambda_dist", 1.0))
    # Tolerances applied to cases whose target is the population mean rather
    # than their own GT: the band has to cover the anatomical spread, not just
    # prediction slack. Default to the GT values so behaviour is unchanged when
    # no population targets are in play.
    # Constrain log(volume) instead of volume, so the band is the same absolute
    # size for every class. Off by default: the linear form is what every earlier
    # run used.
    volume_log_space = bool(cfg.get("volume_log_space", False))
    if is_main and volume_log_space:
        _b = math.log1p(float(cfg.get("volume_tolerance", 0.10)))
        print(f"[volume] log-space constraint: band = log(1+tol) = {_b:.4f} for every class "
              f"(dead-zone threshold 1/t^2 = {1.0 / float(cfg.get('barrier_t', 5.0)) ** 2:.5f})")
    bank_volume_tolerance = float(cfg.get("bank_volume_tolerance", cfg.get("volume_tolerance", 0.10)))
    bank_centroid_tolerance = float(cfg.get("bank_centroid_tolerance", cfg.get("centroid_tolerance", 0.05)))
    bank_avgdist_tolerance = float(cfg.get("bank_avgdist_tolerance", cfg.get("avgdist_tolerance", 0.05)))
    bank_avgdist_axis_tolerance = float(cfg.get("bank_avgdist_axis_tolerance", cfg.get("avgdist_axis_tolerance", 0.05)))
    bank_classes = sorted(set(volume_classes) | set(centroid_classes)
                          | set(avgdist_classes) | set(avgdist_axis_classes))
    constraint_report_every = int(cfg.get("constraint_report_every", 5))

    bank = None
    bank_std = bank_exp = None
    matcher = None
    if need_bank:
        for name, lam in (("moment2", lambda_moment2), ("moment3", lambda_moment3),
                          ("moment_inv", lambda_moment_inv_J1 + lambda_moment_inv_J2 + lambda_moment_inv_J3)):
            if lam > 0.0:
                raise NotImplementedError(
                    f"population targets with lambda_{name}>0: the moment barriers still "
                    "derive their targets from GT and have no bank/distribution path."
                )

        # Resolutions the two branches are actually supervised at: STD goes
        # through logit1 (full patch), EXP through logit2 (one decoder level up).
        raw_model_for_shape = model.module if use_ddp else model
        with torch.no_grad():
            probe = torch.zeros(1, int(cfg["in_channels"]), *cfg["patch_size"], device=device)
            o = raw_model_for_shape(probe, expanded=False)
            std_shape = tuple(int(v) for v in o["logit1"].shape[-3:])
            exp_shape = tuple(int(v) for v in o["logit2"].shape[-3:])
            del probe, o
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shapes = {"std": std_shape, "exp": exp_shape}
        if is_main:
            print(f"[bank] barrier resolutions: std(logit1)={std_shape} exp(logit2)={exp_shape}")

        bank_src = cfg.get("descriptor_bank", "auto")
        if isinstance(bank_src, str) and bank_src.lower() == "auto":
            if is_main:
                print(f"[bank] building from {len(annotated_stems)} annotated cases, classes {bank_classes}...")
            bank = build_descriptor_bank(cfg, annotated_stems, fixed_center, bank_classes, shapes,
                                         workdir=workdir if is_main else None, verbose=is_main)
        else:
            with open(bank_src, "r") as f:
                bank = json.load(f)
            if is_main:
                print(f"[bank] loaded from {bank_src}")

        bank_std = bank_tensors(bank, "std", bank_classes, device)
        bank_exp = bank_tensors(bank, "exp", bank_classes, device)

        if unannotated_mode == "distribution":
            matcher = DescriptorMomentMatcher(
                bank, bank_classes, device,
                n_eff=float(cfg.get("dist_n_eff", 64.0)),
                k_mean=float(cfg.get("dist_k_mean", 2.0)),
                rel_std_tol=float(cfg.get("dist_rel_std_tol", 0.25)),
            )
            if is_main:
                tg = matcher.target["std"]
                print(f"[dist] moment matching on the unannotated pool: n_eff={matcher.n_eff:.0f} "
                      f"k_mean={float(cfg.get('dist_k_mean', 2.0))} rel_std_tol={float(cfg.get('dist_rel_std_tol', 0.25))}")
                for i, c in enumerate(bank_classes):
                    print(f"[dist]   std c{c} volume: target mu={float(tg['mu'][i,0]):.5f} "
                          f"+-{float(tg['tol_mu'][i,0]):.5f}  sd={float(tg['sd'][i,0]):.5f} "
                          f"+-{float(tg['tol_sd'][i,0]):.5f}")

        if is_main:
            print(f"[bank] per-sample targets applied to: {bank_targets}")
            for patch in ("std", "exp"):
                for c in bank_classes:
                    e = bank["bank"][patch][str(c)]
                    print(f"[bank]   {patch} c{c}: frac={e['frac']:.5f} (+-{e['frac_std']:.5f}) "
                          f"cent={[round(v,4) for v in e['cent']]} "
                          f"spread={[round(v,4) for v in e['spread']]}")

    # descriptor-collapse diagnostic (OFF unless the config asks for it): every
    # Nth validation, compare the spread of PREDICTED descriptors over the val
    # set against the spread of the GT ones. Set collapse_log_every_n_vals: 5
    # to enable; 0 or absent keeps the run byte-identical to earlier ones.
    collapse_every = int(cfg.get("collapse_log_every_n_vals", 0))
    collapse_classes = list(desc_classes)
    if is_main and collapse_every > 0:
        print(f"[collapse] logging predicted-vs-GT descriptor spread every "
              f"{collapse_every} validations for classes {collapse_classes}")

    for epoch in range(1, epochs + 1):
        # linear warmup for moment3 (same ramp applied to diag and off-diag weights)
        if moment3_warmup_end > moment3_warmup_start and epoch <= moment3_warmup_end:
            if epoch <= moment3_warmup_start:
                lambda_moment3_eff = 0.0
                lambda_moment3_offdiag_eff = 0.0
            else:
                progress = (epoch - moment3_warmup_start) / (moment3_warmup_end - moment3_warmup_start)
                lambda_moment3_eff = progress * lambda_moment3
                lambda_moment3_offdiag_eff = progress * lambda_moment3_offdiag
        else:
            lambda_moment3_eff = lambda_moment3
            lambda_moment3_offdiag_eff = lambda_moment3_offdiag

        # tighten the shape barrier (volume/centroid/avgdist/avgdist_axis) each epoch;
        # moment2/moment3/moment_inv keep using the static `barrier` at fixed barrier_t
        barrier_shape.t = min(barrier_t * (barrier_t_mu ** (epoch - 1)), barrier_t_max)

        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        loss_sum_total = 0.0
        loss_sum_dist = 0.0
        err_sum_vol = {int(c): 0.0 for c in volume_classes}
        err_sum_cent = {int(c): 0.0 for c in centroid_classes}
        n_dist = 0
        dist_last = {}
        loss_sum_seg_logged = 0.0
        loss_sum_seg_bw = 0.0
        loss_sum_vol = 0.0
        loss_sum_cent = 0.0
        loss_sum_avgdist = 0.0
        loss_sum_avgdist_axis = 0.0
        loss_sum_avgdist_axis_z = 0.0
        loss_sum_avgdist_axis_y = 0.0
        loss_sum_avgdist_axis_x = 0.0
        loss_sum_moment2 = 0.0
        loss_sum_moment3 = 0.0
        loss_sum_moment_inv = 0.0
        loss_sum_moment2_err = 0.0
        loss_sum_moment3_err = 0.0
        loss_sum_moment_inv_err = 0.0
        loss_sum_moment_inv_J1_err = 0.0
        loss_sum_moment_inv_J2_err = 0.0
        loss_sum_moment_inv_J3_err = 0.0
        loss_sum_moment2_err_per_class = {c: 0.0 for c in moment2_classes}
        loss_sum_moment3_err_per_class = {c: 0.0 for c in moment3_classes}
        loss_sum_moment_inv_err_per_class = {c: 0.0 for c in moment_inv_classes}
        n_it = 0

        for i, batch in enumerate(train_loader):

            # TRUE EXP-only mode
            expanded = True if shape_on == "exp" else (i % 2 == 0)

            # diag/off-diag moment breakdown printed on the first batch of every 10th
            # epoch (plus epoch 1), to spot-check without flooding the log
            moment_verbose = is_main and i == 0 and (epoch == 1 or epoch % 10 == 0)

            img = (batch["exp_img"] if expanded else batch["std_img"]).to(device, non_blocking=True)
            lbl = (batch["exp_lbl"] if expanded else batch["std_lbl"]).to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg.get("amp", True)):
                out = model(img, expanded=expanded)
                logits = out["logit2"] if expanded else out["logit1"]

                lbl_rs = resize_lbl_to_logits(lbl, logits)

                # Split the batch by supervision source. `has_gt` is all-True
                # unless withheld cases are being loaded.
                has_gt = batch.get("has_gt")
                has_gt = (torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
                          if has_gt is None else has_gt.to(logits.device).bool())
                patch_key = "exp" if expanded else "std"
                bk = bank_exp if expanded else bank_std

                if bank_targets == "all":
                    # every case takes a per-sample population target (A1.3)
                    use_bank = torch.ones_like(has_gt)
                    desc_logits, desc_lbl = logits, lbl_rs
                elif unannotated_mode == "bank":
                    # withheld cases take per-sample population targets (A2)
                    use_bank = ~has_gt
                    desc_logits, desc_lbl = logits, lbl_rs
                else:
                    # GT-only per-sample barriers; withheld cases (if any) are
                    # handled by the distribution term instead, so they must be
                    # excluded here -- their label is pure ignore and would
                    # otherwise drive every descriptor toward zero.
                    use_bank = None
                    if bool(has_gt.all()):
                        desc_logits, desc_lbl = logits, lbl_rs
                    else:
                        gt_idx = has_gt.nonzero(as_tuple=False).squeeze(1)
                        desc_logits, desc_lbl = logits[gt_idx], lbl_rs[gt_idx]

                # Seg loss (monitored and, if enabled, backward) must only ever
                # see cases that actually have a label: DiceCE on an all-ignore
                # target is NaN, not zero.
                if bool(has_gt.all()):
                    seg_logits, seg_lbl = logits, lbl_rs
                else:
                    _sidx = has_gt.nonzero(as_tuple=False).squeeze(1)
                    seg_logits, seg_lbl = logits[_sidx], lbl_rs[_sidx]

                # monitor-only seg loss
                if monitor_seg_loss and seg_logits.shape[0] > 0:
                    seg_loss_logged = crit(seg_logits, seg_lbl)
                else:
                    seg_loss_logged = logits.new_tensor(0.0)

                # seg loss used for backward
                if use_seg_loss and seg_logits.shape[0] > 0:
                    logits_for_bw = seg_logits
                    if detach_for_seg and desc_classes:
                        logits_for_bw = detach_channels_for_seg_loss(seg_logits, desc_classes)
                    seg_loss_bw = crit(logits_for_bw, seg_lbl)
                else:
                    seg_loss_bw = logits.new_tensor(0.0)

                apply_shape = (
                    (shape_on == "std" and (not expanded)) or
                    (shape_on == "exp" and expanded) or
                    (shape_on == "both")
                )

                vol_loss = logits.new_tensor(0.0)
                cent_loss = logits.new_tensor(0.0)
                avgdist_loss = logits.new_tensor(0.0)
                avgdist_axis_loss = logits.new_tensor(0.0)
                avgdist_axis_stats = logits.new_zeros(3)
                moment2_diag_loss = logits.new_tensor(0.0)
                moment2_offdiag_loss = logits.new_tensor(0.0)
                moment3_diag_loss = logits.new_tensor(0.0)
                moment3_offdiag_loss = logits.new_tensor(0.0)
                moment_inv_loss = logits.new_tensor(0.0)
                moment2_err = logits.new_tensor(0.0)
                moment3_err = logits.new_tensor(0.0)
                moment_inv_stats = logits.new_zeros(3)  # [err_J1, err_J2, err_J3]
                vol_errs = logits.new_zeros(len(volume_classes)) if volume_classes else logits.new_zeros(1)
                cent_errs_per_class = {int(c): logits.new_tensor(0.0) for c in centroid_classes}
                m2_errs_per_class = {c: logits.new_tensor(0.0) for c in moment2_classes}
                m3_errs_per_class = {c: logits.new_tensor(0.0) for c in moment3_classes}
                minv_errs_per_class = {c: logits.new_zeros(3) for c in moment_inv_classes}

                # GT-supervised samples keep the tight band; bank-supervised
                # ones get the wide band. Scalars when the batch is uniform.
                if use_bank is not None and bool(use_bank.any()) and not bool(use_bank.all()):
                    _ub = use_bank.float()
                    vol_tol_b = volume_tolerance + _ub * (bank_volume_tolerance - volume_tolerance)
                    cent_tol_b = centroid_tolerance + _ub * (bank_centroid_tolerance - centroid_tolerance)
                    avgd_tol_b = avgdist_tolerance + _ub * (bank_avgdist_tolerance - avgdist_tolerance)
                    axis_tol_b = avgdist_axis_tolerance + _ub * (bank_avgdist_axis_tolerance - avgdist_axis_tolerance)
                elif use_bank is not None and bool(use_bank.all()):
                    vol_tol_b, cent_tol_b = bank_volume_tolerance, bank_centroid_tolerance
                    avgd_tol_b, axis_tol_b = bank_avgdist_tolerance, bank_avgdist_axis_tolerance
                else:
                    vol_tol_b, cent_tol_b = volume_tolerance, centroid_tolerance
                    avgd_tol_b, axis_tol_b = avgdist_tolerance, avgdist_axis_tolerance

                if apply_shape and desc_logits.shape[0] > 0:
                    if lambda_volume > 0.0 and len(volume_classes) > 0:
                        vol_loss, vol_errs = compute_volume_barrier(
                            return_stats=True,
                            logits=desc_logits,
                            target=desc_lbl,
                            n_classes=cfg["n_classes"],
                            volume_classes=volume_classes,
                            barrier=barrier_shape,
                            volume_tolerance=vol_tol_b,
                            use_bank=use_bank,
                            bank_value=(torch.stack([bk["frac"][int(c)] for c in volume_classes])
                                        if use_bank is not None else None),
                            log_space=volume_log_space,
                        )

                    if lambda_centroid > 0.0 and len(centroid_classes) > 0:
                        cents = []
                        for c in centroid_classes:
                            _cl, _ce = compute_centroid_barrier(
                                    return_stats=True,
                                    logits=desc_logits,
                                    target=desc_lbl,
                                    n_classes=cfg["n_classes"],
                                    centroid_class=c,
                                    barrier=barrier_shape,
                                    centroid_tolerance=cent_tol_b,
                                    centroid_norm=centroid_norm,
                                    use_bank=use_bank,
                                    bank_value=(bk["cent"][int(c)] if use_bank is not None else None),
                                )
                            cents.append(_cl)
                            cent_errs_per_class[int(c)] = _ce
                        cent_loss = torch.stack(cents).mean() if len(cents) > 0 else logits.new_tensor(0.0)

                    if lambda_avgdist > 0.0 and len(avgdist_classes) > 0:
                        avgdists = []
                        for c in avgdist_classes:
                            avgdists.append(
                                compute_avgdist_barrier(
                                    logits=desc_logits,
                                    target=desc_lbl,
                                    n_classes=cfg["n_classes"],
                                    distance_class=c,
                                    barrier=barrier_shape,
                                    avgdist_tolerance=avgd_tol_b,
                                    centroid_norm=centroid_norm,
                                    use_bank=use_bank,
                                    bank_value=(bk["avgdist"][int(c)] if use_bank is not None else None),
                                )
                            )
                        avgdist_loss = torch.stack(avgdists).mean() if len(avgdists) > 0 else logits.new_tensor(0.0)

                    if lambda_avgdist_axis > 0.0 and len(avgdist_axis_classes) > 0:
                        avgdist_axes = []
                        avgdist_axis_stats_all = []
                        for c in avgdist_axis_classes:
                            axis_loss_c, axis_stats_c = compute_avgdist_axis_barrier(
                                    logits=desc_logits,
                                    target=desc_lbl,
                                    n_classes=cfg["n_classes"],
                                    distance_class=c,
                                    barrier=barrier_shape,
                                    avgdist_axis_tolerance=axis_tol_b,
                                    centroid_norm=centroid_norm,
                                    return_stats=True,
                                    use_bank=use_bank,
                                    bank_value=(bk["spread"][int(c)] if use_bank is not None else None),
                                )
                            avgdist_axes.append(axis_loss_c)
                            avgdist_axis_stats_all.append(axis_stats_c)
                        avgdist_axis_loss = torch.stack(avgdist_axes).mean() if len(avgdist_axes) > 0 else logits.new_tensor(0.0)
                        avgdist_axis_stats = (
                            torch.stack(avgdist_axis_stats_all).mean(dim=0)
                            if len(avgdist_axis_stats_all) > 0
                            else logits.new_zeros(3)
                        )

                    if (lambda_moment2 > 0.0 or lambda_moment2_offdiag > 0.0) and len(moment2_classes) > 0:
                        m2_diags, m2_offdiags, m2_errs_per_class = [], [], {}
                        for c in moment2_classes:
                            diag_c, offdiag_c, e_c = compute_2nd_moment_barrier(
                                logits=logits,
                                target=lbl_rs,
                                n_classes=cfg["n_classes"],
                                moment_class=c,
                                barrier=barrier_m2_diag,
                                barrier_offdiag=barrier_m2_offdiag,
                                moment_tolerance=moment2_tolerance,
                                offdiag_tolerance=moment2_offdiag_tolerance,
                                centroid_norm=centroid_norm,
                                return_stats=True,
                                gamma=moment2_gamma,
                                sqrt_diagonal=moment2_sqrt_diagonal,
                                verbose=moment_verbose,
                            )
                            m2_diags.append(diag_c)
                            m2_offdiags.append(offdiag_c)
                            m2_errs_per_class[c] = e_c
                        moment2_diag_loss = torch.stack(m2_diags).mean() if m2_diags else logits.new_tensor(0.0)
                        moment2_offdiag_loss = torch.stack(m2_offdiags).mean() if m2_offdiags else logits.new_tensor(0.0)
                        moment2_err  = torch.stack(list(m2_errs_per_class.values())).mean() if m2_errs_per_class else logits.new_tensor(0.0)

                    if len(moment3_classes) > 0:
                        m3_diags, m3_offdiags, m3_errs_per_class = [], [], {}
                        for c in moment3_classes:
                            diag_c, offdiag_c, e_c = compute_3rd_moment_barrier(
                                logits=logits,
                                target=lbl_rs,
                                n_classes=cfg["n_classes"],
                                moment_class=c,
                                barrier=barrier_m3_diag,
                                barrier_offdiag=barrier_m3_offdiag,
                                moment_tolerance=moment3_tolerance,
                                offdiag_tolerance=moment3_offdiag_tolerance,
                                centroid_norm=centroid_norm,
                                return_stats=True,
                                gamma=moment3_gamma,
                                sqrt_diagonal=moment3_sqrt_diagonal,
                                verbose=moment_verbose,
                            )
                            m3_diags.append(diag_c)
                            m3_offdiags.append(offdiag_c)
                            m3_errs_per_class[c] = e_c
                        moment3_diag_loss = torch.stack(m3_diags).mean() if m3_diags else logits.new_tensor(0.0)
                        moment3_offdiag_loss = torch.stack(m3_offdiags).mean() if m3_offdiags else logits.new_tensor(0.0)
                        moment3_err  = torch.stack(list(m3_errs_per_class.values())).mean() if m3_errs_per_class else logits.new_tensor(0.0)

                    pass

                dist_loss = logits.new_tensor(0.0)
                dist_stats = None
                if matcher is not None and apply_shape and not bool(has_gt.all()):
                    u_idx = (~has_gt).nonzero(as_tuple=False).squeeze(1)
                    if u_idx.numel() > 0:
                        dist_loss, dist_stats = matcher(
                            logits[u_idx], patch_key, barrier_shape,
                            {"volume": lambda_volume, "centroid": lambda_centroid,
                             "spread": lambda_avgdist_axis},
                        )

                if apply_shape and desc_logits.shape[0] > 0:
                    _any_inv_active = any(l > 0.0 for l in [lambda_moment_inv_J1, lambda_moment_inv_J2, lambda_moment_inv_J3])
                    if _any_inv_active and len(moment_inv_classes) > 0:
                        mis = []
                        for c in moment_inv_classes:
                            l_c, s_c = compute_moment_invariants_barrier(
                                logits=logits,
                                target=lbl_rs,
                                n_classes=cfg["n_classes"],
                                moment_class=c,
                                barrier=barrier,
                                barrier_J1=barrier_J1,
                                barrier_J2=barrier_J2,
                                barrier_J3=barrier_J3,
                                lambda_J1=lambda_moment_inv_J1,
                                lambda_J2=lambda_moment_inv_J2,
                                lambda_J3=lambda_moment_inv_J3,
                                tol_J1=moment_inv_J1_tolerance,
                                tol_J2=moment_inv_J2_tolerance,
                                tol_J3=moment_inv_J3_tolerance,
                                centroid_norm=centroid_norm,
                                return_stats=True,
                                gamma=moment_inv_gamma,
                                verbose=moment_verbose,
                            )
                            mis.append(l_c)
                            minv_errs_per_class[c] = s_c
                        moment_inv_loss = torch.stack(mis).mean() if mis else logits.new_tensor(0.0)
                        moment_inv_stats = (
                            torch.stack(list(minv_errs_per_class.values())).mean(dim=0)
                            if minv_errs_per_class else logits.new_zeros(3)
                        )

                total_loss = (
                    seg_loss_bw
                    + lambda_volume * vol_loss
                    + lambda_centroid * cent_loss
                    + lambda_avgdist * avgdist_loss
                    + lambda_avgdist_axis * avgdist_axis_loss
                    + lambda_moment2 * moment2_diag_loss
                    + lambda_moment2_offdiag * moment2_offdiag_loss
                    + lambda_moment3_eff * moment3_diag_loss
                    + lambda_moment3_offdiag_eff * moment3_offdiag_loss
                    + moment_inv_loss
                    + lambda_dist * dist_loss
                )

                if not total_loss.requires_grad:
                    continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.get("grad_clip", 5.0)))
            scaler.step(opt)
            scaler.update()

            total_loss_v = float(total_loss.detach())
            seg_loss_logged_v = float(seg_loss_logged.detach())
            seg_loss_bw_v = float(seg_loss_bw.detach())
            vol_loss_v = float(vol_loss.detach())
            cent_loss_v = float(cent_loss.detach())
            for _j, _c in enumerate(volume_classes):
                err_sum_vol[int(_c)] += float(vol_errs[_j].detach())
            for _c in centroid_classes:
                err_sum_cent[int(_c)] += float(cent_errs_per_class[int(_c)].detach())
            dist_loss_v = float(dist_loss.detach())
            if dist_stats is not None:
                loss_sum_dist += dist_loss_v
                n_dist += 1
                dist_last[patch_key] = dist_stats
            avgdist_loss_v = float(avgdist_loss.detach())
            avgdist_axis_loss_v = float(avgdist_axis_loss.detach())
            avgdist_axis_z_v = float(avgdist_axis_stats[0].detach())
            avgdist_axis_y_v = float(avgdist_axis_stats[1].detach())
            avgdist_axis_x_v = float(avgdist_axis_stats[2].detach())
            moment2_loss_v    = float((moment2_diag_loss + moment2_offdiag_loss).detach())
            moment3_loss_v    = float((moment3_diag_loss + moment3_offdiag_loss).detach())
            moment_inv_loss_v = float(moment_inv_loss.detach())
            moment2_err_v     = float(moment2_err.detach())
            moment3_err_v     = float(moment3_err.detach())
            moment_inv_stats_v = moment_inv_stats.detach().cpu().tolist()  # [J1, J2, J3]

            loss_sum_total += total_loss_v
            loss_sum_seg_logged += seg_loss_logged_v
            loss_sum_seg_bw += seg_loss_bw_v
            loss_sum_vol += vol_loss_v
            loss_sum_cent += cent_loss_v
            loss_sum_avgdist += avgdist_loss_v
            loss_sum_avgdist_axis += avgdist_axis_loss_v
            loss_sum_avgdist_axis_z += avgdist_axis_z_v
            loss_sum_avgdist_axis_y += avgdist_axis_y_v
            loss_sum_avgdist_axis_x += avgdist_axis_x_v
            loss_sum_moment2 += moment2_loss_v
            loss_sum_moment3 += moment3_loss_v
            loss_sum_moment_inv += moment_inv_loss_v
            loss_sum_moment2_err += moment2_err_v
            loss_sum_moment3_err += moment3_err_v
            loss_sum_moment_inv_err    += sum(moment_inv_stats_v) / max(1, len(moment_inv_stats_v))
            loss_sum_moment_inv_J1_err += moment_inv_stats_v[0]
            loss_sum_moment_inv_J2_err += moment_inv_stats_v[1]
            loss_sum_moment_inv_J3_err += moment_inv_stats_v[2]
            for c, e in m2_errs_per_class.items():
                loss_sum_moment2_err_per_class[c] += float(e.detach())
            for c, e in m3_errs_per_class.items():
                loss_sum_moment3_err_per_class[c] += float(e.detach())
            for c, s in minv_errs_per_class.items():
                loss_sum_moment_inv_err_per_class[c] += float(s.detach().mean())
            n_it += 1

            if is_main and i % log_every == 0:
                print(
                    f"[E{epoch:03d} i{i:04d} {'EXP' if expanded else 'STD'}] "
                    f"tot={total_loss_v:.4f} "
                    f"seg(log)={seg_loss_logged_v:.4f} "
                    f"seg(bw)={seg_loss_bw_v:.4f} "
                    f"vol={vol_loss_v:.4f} "
                    f"cent={cent_loss_v:.4f} "
                    f"avgdist={avgdist_loss_v:.4f} "
                    f"avgdist_axis={avgdist_axis_loss_v:.4f} "
                    f"m2={moment2_loss_v:.4f}(err={moment2_err_v:.2e}) "
                    f"m3={moment3_loss_v:.4f}(err={moment3_err_v:.2e}) "
                    f"minv={moment_inv_loss_v:.4f}(J1={moment_inv_stats_v[0]:.2e},J2={moment_inv_stats_v[1]:.2e},J3={moment_inv_stats_v[2]:.2e}) "
                    f"t_shape={barrier_shape.t:.2f} "
                    f"t_m3={barrier_m3_diag.t:.0f}/{barrier_m3_offdiag.t:.0f} "
                    f"t_J={barrier_J1.t:.0f}/{barrier_J2.t:.0f}/{barrier_J3.t:.0f} "
                    f"lr={lr_now(opt):.2e}"
                )

            # Drop large per-iteration tensors before the next forward to reduce memory
            del out
            del logits
            del lbl_rs
            del seg_loss_logged
            del seg_loss_bw
            del vol_loss
            del cent_loss
            del avgdist_loss
            del dist_loss
            del avgdist_axis_loss
            del avgdist_axis_stats
            del moment2_diag_loss
            del moment2_offdiag_loss
            del moment3_diag_loss
            del moment3_offdiag_loss
            del moment_inv_loss
            del moment2_err
            del moment3_err
            del moment_inv_stats
            del m2_errs_per_class
            del m3_errs_per_class
            del minv_errs_per_class
            del img
            del lbl
            del batch

        # All-reduce epoch train loss sums across ranks before computing averages
        if use_ddp:
            n_m2c = len(moment2_classes)
            n_m3c = len(moment3_classes)
            per_class_m2   = [loss_sum_moment2_err_per_class[c]   for c in moment2_classes]
            per_class_m3   = [loss_sum_moment3_err_per_class[c]   for c in moment3_classes]
            per_class_minv = [loss_sum_moment_inv_err_per_class[c] for c in moment_inv_classes]
            t = torch.tensor(
                [loss_sum_total, loss_sum_seg_logged, loss_sum_seg_bw, loss_sum_vol,
                 loss_sum_cent, loss_sum_avgdist, loss_sum_avgdist_axis,
                 loss_sum_avgdist_axis_z, loss_sum_avgdist_axis_y, loss_sum_avgdist_axis_x,
                 loss_sum_moment2, loss_sum_moment3, loss_sum_moment_inv,
                 loss_sum_moment2_err, loss_sum_moment3_err,
                 loss_sum_moment_inv_err, loss_sum_moment_inv_J1_err,
                 loss_sum_moment_inv_J2_err, loss_sum_moment_inv_J3_err,
                 float(n_it)]
                + per_class_m2 + per_class_m3 + per_class_minv,
                device=device, dtype=torch.float64,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            vals = t.cpu().tolist()
            (loss_sum_total, loss_sum_seg_logged, loss_sum_seg_bw, loss_sum_vol,
             loss_sum_cent, loss_sum_avgdist, loss_sum_avgdist_axis,
             loss_sum_avgdist_axis_z, loss_sum_avgdist_axis_y, loss_sum_avgdist_axis_x,
             loss_sum_moment2, loss_sum_moment3, loss_sum_moment_inv,
             loss_sum_moment2_err, loss_sum_moment3_err,
             loss_sum_moment_inv_err, loss_sum_moment_inv_J1_err,
             loss_sum_moment_inv_J2_err, loss_sum_moment_inv_J3_err,
             n_it_f) = vals[:20]
            n_it = int(n_it_f)
            for idx, c in enumerate(moment2_classes):
                loss_sum_moment2_err_per_class[c] = vals[20 + idx]
            for idx, c in enumerate(moment3_classes):
                loss_sum_moment3_err_per_class[c] = vals[20 + n_m2c + idx]
            for idx, c in enumerate(moment_inv_classes):
                loss_sum_moment_inv_err_per_class[c] = vals[20 + n_m2c + n_m3c + idx]

        # epoch averages
        loss_tr_total = loss_sum_total / max(1, n_it)
        loss_tr_seg_logged = loss_sum_seg_logged / max(1, n_it)
        loss_tr_seg_bw = loss_sum_seg_bw / max(1, n_it)
        loss_tr_vol = loss_sum_vol / max(1, n_it)
        loss_tr_cent = loss_sum_cent / max(1, n_it)
        loss_tr_avgdist = loss_sum_avgdist / max(1, n_it)
        loss_tr_avgdist_axis = loss_sum_avgdist_axis / max(1, n_it)
        loss_tr_avgdist_axis_z = loss_sum_avgdist_axis_z / max(1, n_it)
        loss_tr_avgdist_axis_y = loss_sum_avgdist_axis_y / max(1, n_it)
        loss_tr_avgdist_axis_x = loss_sum_avgdist_axis_x / max(1, n_it)
        loss_tr_moment2 = loss_sum_moment2 / max(1, n_it)
        loss_tr_moment3 = loss_sum_moment3 / max(1, n_it)
        loss_tr_moment_inv = loss_sum_moment_inv / max(1, n_it)
        loss_tr_moment2_err = loss_sum_moment2_err / max(1, n_it)
        loss_tr_moment3_err = loss_sum_moment3_err / max(1, n_it)
        loss_tr_moment_inv_err    = loss_sum_moment_inv_err    / max(1, n_it)
        loss_tr_moment_inv_J1_err = loss_sum_moment_inv_J1_err / max(1, n_it)
        loss_tr_moment_inv_J2_err = loss_sum_moment_inv_J2_err / max(1, n_it)
        loss_tr_moment_inv_J3_err = loss_sum_moment_inv_J3_err / max(1, n_it)
        err_tr_vol  = {c: err_sum_vol[c]  / max(1, n_it) for c in err_sum_vol}
        err_tr_cent = {c: err_sum_cent[c] / max(1, n_it) for c in err_sum_cent}
        loss_tr_moment2_err_per_class   = {c: loss_sum_moment2_err_per_class[c]   / max(1, n_it) for c in moment2_classes}
        loss_tr_moment3_err_per_class   = {c: loss_sum_moment3_err_per_class[c]   / max(1, n_it) for c in moment3_classes}
        loss_tr_moment_inv_err_per_class = {c: loss_sum_moment_inv_err_per_class[c] / max(1, n_it) for c in moment_inv_classes}

        # validation — all ranks participate, then metrics are all_reduced
        model.eval()
        loss_va_seg = 0.0
        dices_sum = None
        n_va = 0
        do_collapse = collapse_every > 0 and (epoch == 1 or epoch % collapse_every == 0)
        if do_collapse:
            n_cc = len(collapse_classes)
            col_n = torch.zeros(n_cc, dtype=torch.float64, device=device)
            col_sum = torch.zeros(2, n_cc, 7, dtype=torch.float64, device=device)
            col_sq = torch.zeros(2, n_cc, 7, dtype=torch.float64, device=device)
        with torch.no_grad():
            for batch in val_loader:
                x = batch["std_img"].to(device, non_blocking=True)
                y = batch["std_lbl"].to(device, non_blocking=True)
                out = model(x, expanded=False)
                logits = out["logit1"]
                y_rs = resize_lbl_to_logits(y, logits)

                if monitor_seg_loss:
                    loss_va_seg += float(crit(logits, y_rs))

                d = np.array(soft_dice_per_class(logits, y_rs, cfg["n_classes"]))
                dices_sum = d if dices_sum is None else (dices_sum + d)
                n_va += 1

                if do_collapse:
                    p_desc, g_desc, present = descriptor_summary(
                        logits, y_rs, collapse_classes, cfg["n_classes"]
                    )
                    w = present.double().unsqueeze(-1)  # (B,C,1) — skip absent classes
                    for j, arr in enumerate((p_desc, g_desc)):
                        a = arr.double()
                        col_sum[j] += (a * w).sum(dim=0)
                        col_sq[j] += (a * a * w).sum(dim=0)
                    col_n += present.double().sum(dim=0)

        if dices_sum is None:
            dices_sum = np.zeros(cfg["n_classes"] - 1)

        if use_ddp:
            va_t = torch.tensor(
                [loss_va_seg, float(n_va)] + dices_sum.tolist(),
                device=device, dtype=torch.float64,
            )
            dist.all_reduce(va_t, op=dist.ReduceOp.SUM)
            va_vals = va_t.cpu().tolist()
            loss_va_seg = va_vals[0]
            n_va_total = int(va_vals[1])
            dices_sum = np.array(va_vals[2:])
        else:
            n_va_total = n_va

        loss_va_seg = loss_va_seg / max(1, n_va_total) if monitor_seg_loss else 0.0
        dices = dices_sum / max(1, n_va_total)
        meanFGDice = float(dices.mean()) if dices is not None else 0.0

        # --- distribution-constraint status ---------------------------------
        # Logged as signed slack z: z <= 0 satisfied (magnitude = headroom),
        # z > 0 violated (magnitude = how far outside). Continuous, so a
        # constraint drifting toward its bound is visible before it crosses.
        dist_log = {}
        if matcher is not None and dist_last:
            n_sat = n_tot = 0
            for patch, st in dist_last.items():
                z_mu, z_sd, ratio, act = st["z_mu"], st["z_sd"], st["sd_ratio"], st["active"]
                for i, c in enumerate(bank_classes):
                    if i >= z_mu.shape[0]:
                        break
                    for gname, sl in DESC_SLICES.items():
                        a = act[i, sl]
                        if not bool(a.any()):
                            continue
                        dist_log[f"dist/{patch}_zmu_{gname}_c{c}"] = float(z_mu[i, sl][a].max())
                        dist_log[f"dist/{patch}_zsd_{gname}_c{c}"] = float(z_sd[i, sl][a].max())
                        dist_log[f"dist/{patch}_sdratio_{gname}_c{c}"] = float(ratio[i, sl][a].mean())
                n_sat += int((z_mu[act] <= 0).sum() + (z_sd[act] <= 0).sum())
                n_tot += int(2 * int(act.sum()))
            dist_log["dist/frac_satisfied"] = n_sat / max(1, n_tot)
            dist_log["dist/loss"] = loss_sum_dist / max(1, n_dist)

            if is_main and constraint_report_every > 0 and (epoch == 1 or epoch % constraint_report_every == 0):
                print(f"    [constraints] ep{epoch} satisfied "
                      f"{n_sat}/{n_tot} ({dist_log['dist/frac_satisfied']:.0%})  "
                      f"dist_loss={dist_log['dist/loss']:.4f}")
                for patch in sorted(dist_last):
                    for line in matcher.report(patch, bank_classes):
                        print(line)

        collapse_log = {}
        if do_collapse:
            if use_ddp:
                dist.all_reduce(col_n, op=dist.ReduceOp.SUM)
                dist.all_reduce(col_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(col_sq, op=dist.ReduceOp.SUM)
            n = col_n.clamp(min=1.0).view(1, -1, 1)
            mean = col_sum / n
            std = (col_sq / n - mean ** 2).clamp(min=0.0).sqrt()
            mean_np = mean.cpu().numpy()
            std_np = std.cpu().numpy()
            n_np = col_n.cpu().numpy()

            for i, c in enumerate(collapse_classes):
                if n_np[i] < 2:
                    continue
                # volume as RELATIVE spread, so it is directly comparable to the
                # population figures the bank tolerances were derived from;
                # centroid/spread are absolute, averaged over the three axes.
                pv = float(std_np[0, i, 0] / max(mean_np[0, i, 0], 1e-12))
                gv = float(std_np[1, i, 0] / max(mean_np[1, i, 0], 1e-12))
                pc = float(std_np[0, i, 1:4].mean())
                gc = float(std_np[1, i, 1:4].mean())
                ps = float(std_np[0, i, 4:7].mean())
                gs = float(std_np[1, i, 4:7].mean())
                collapse_log.update({
                    f"collapse/vol_relstd_pred_c{c}": pv,
                    f"collapse/vol_relstd_gt_c{c}": gv,
                    f"collapse/cent_std_pred_c{c}": pc,
                    f"collapse/cent_std_gt_c{c}": gc,
                    f"collapse/spread_std_pred_c{c}": ps,
                    f"collapse/spread_std_gt_c{c}": gs,
                })
                # A ratio is only meaningful when the GT actually varies across
                # the val set; emit nothing rather than a 0.0 that would read as
                # collapse when the denominator is simply degenerate.
                for key, num, den in (("vol_relstd", pv, gv),
                                      ("cent_std", pc, gc),
                                      ("spread_std", ps, gs)):
                    if den > 1e-8:
                        collapse_log[f"collapse/{key}_ratio_c{c}"] = num / den

            if is_main and collapse_log:
                parts = [
                    f"c{c} vol {collapse_log[f'collapse/vol_relstd_ratio_c{c}']:.2f}"
                    for c in collapse_classes
                    if f"collapse/vol_relstd_ratio_c{c}" in collapse_log
                ]
                print(f"[collapse] ep{epoch} pred/GT spread ratio (1.0 = no collapse): "
                      + "  ".join(parts))

        row = {
            "epoch": epoch,
            "loss_tr_total": loss_tr_total,
            "loss_tr_seg_logged": loss_tr_seg_logged,
            "loss_tr_seg_bw": loss_tr_seg_bw,
            "loss_tr_vol": loss_tr_vol,
            "loss_tr_cent": loss_tr_cent,
            "loss_tr_avgdist": loss_tr_avgdist,
            "loss_tr_avgdist_axis": loss_tr_avgdist_axis,
            "loss_tr_avgdist_axis_z": loss_tr_avgdist_axis_z,
            "loss_tr_avgdist_axis_y": loss_tr_avgdist_axis_y,
            "loss_tr_avgdist_axis_x": loss_tr_avgdist_axis_x,
            "loss_va_seg": loss_va_seg,
            "meanFGDice": meanFGDice,
            "lr": lr_now(opt),
            "use_seg_loss": int(use_seg_loss),
            "monitor_seg_loss": int(monitor_seg_loss),
            "barrier_t_shape": barrier_shape.t,
            "barrier_t_moment": barrier.t,
            "volume_tolerance": volume_tolerance,
            "centroid_tolerance": centroid_tolerance,
            "avgdist_tolerance": avgdist_tolerance,
            "avgdist_axis_tolerance": avgdist_axis_tolerance,
            "loss_tr_moment2": loss_tr_moment2,
            "loss_tr_moment3": loss_tr_moment3,
            "loss_tr_moment_inv": loss_tr_moment_inv,
            "loss_tr_moment2_err": loss_tr_moment2_err,
            "loss_tr_moment3_err": loss_tr_moment3_err,
            "loss_tr_moment_inv_err": loss_tr_moment_inv_err,
            "loss_tr_moment_inv_J1_err": loss_tr_moment_inv_J1_err,
            "loss_tr_moment_inv_J2_err": loss_tr_moment_inv_J2_err,
            "loss_tr_moment_inv_J3_err": loss_tr_moment_inv_J3_err,
        }
        for k, d in enumerate(dices.tolist(), start=1):
            row[f"dice_c{k}"] = float(d)
        row.update(collapse_log)
        row.update(dist_log)

        if is_main:
            rows.append(row)
            pd.DataFrame(rows).to_csv(csv_path, index=False)

        viz_log = {}
        if viz_enable and viz_ready and is_main and (epoch % viz_every_n_epochs == 0):
            raw_model = model.module if use_ddp else model
            raw_model.eval()
            with torch.no_grad():
                x = viz_sample["std_img"].unsqueeze(0).to(device)
                out = raw_model(x, expanded=False)
                # upsample the prediction back to the original patch resolution
                # (rather than downsizing GT) so viz_slice_idx, picked once
                # against std_lbl, stays valid across epochs.
                pred_full = F.interpolate(
                    out["logit1"].argmax(dim=1, keepdim=True).float(),
                    size=viz_sample["std_img"].shape[-3:],
                    mode="nearest",
                )[0, 0].long().cpu().numpy()
            img_slice = np.take(viz_sample["std_img"][0].numpy(), viz_slice_idx, axis=viz_slice_axis)
            gt_slice = np.take(viz_sample["std_lbl"].numpy(), viz_slice_idx, axis=viz_slice_axis)
            pred_slice = np.take(pred_full, viz_slice_idx, axis=viz_slice_axis)
            viz_log = {
                "viz/segmentation": build_viz_wandb_image(img_slice, gt_slice, pred_slice, viz_class_names)
            }

        if use_wandb and is_main:
            wandb.log({
                # total and seg monitoring
                "train/total":          loss_tr_total,
                "train/seg_logged":     loss_tr_seg_logged,
                "train/seg_bw":         loss_tr_seg_bw,
                # raw barrier values (unweighted) — what the barrier actually outputs
                "desc/vol_barrier":         loss_tr_vol,
                "desc/cent_barrier":        loss_tr_cent,
                "desc/avgdist_barrier":     loss_tr_avgdist,
                "desc/avgdist_axis_barrier":loss_tr_avgdist_axis,
                "desc/avgdist_axis_z":      loss_tr_avgdist_axis_z,
                "desc/avgdist_axis_y":      loss_tr_avgdist_axis_y,
                "desc/avgdist_axis_x":      loss_tr_avgdist_axis_x,
                "desc/m2_barrier":          loss_tr_moment2,
                "desc/m2_err":              loss_tr_moment2_err,
                "desc/m3_barrier":          loss_tr_moment3,
                "desc/m3_err":              loss_tr_moment3_err,
                "desc/minv_barrier":        loss_tr_moment_inv,
                "desc/minv_err":            loss_tr_moment_inv_err,
                "desc/minv_err_J1":         loss_tr_moment_inv_J1_err,
                "desc/minv_err_J2":         loss_tr_moment_inv_J2_err,
                "desc/minv_err_J3":         loss_tr_moment_inv_J3_err,
                # per-anatomy moment errors
                # achieved error vs the tolerance that is being asked for.
                # ratio << 1 -> the tolerance is loose and the term has stopped
                # constraining anything; ratio > 1 -> still binding.
                **{f"desc/vol_err_rel_c{c}": err_tr_vol[c] for c in err_tr_vol},
                **{f"desc/vol_err_over_tol_c{c}": err_tr_vol[c] / max(volume_tolerance, 1e-12) for c in err_tr_vol},
                **{f"desc/cent_err_c{c}": err_tr_cent[c] for c in err_tr_cent},
                **{f"desc/cent_err_over_tol_c{c}": err_tr_cent[c] / max(centroid_tolerance, 1e-12) for c in err_tr_cent},
                **{f"desc/m2_err_c{c}": loss_tr_moment2_err_per_class[c] for c in moment2_classes},
                **{f"desc/m3_err_c{c}": loss_tr_moment3_err_per_class[c] for c in moment3_classes},
                **{f"desc/minv_err_c{c}": loss_tr_moment_inv_err_per_class[c] for c in moment_inv_classes},
                # weighted contributions (lambda × barrier) — what drives the gradient
                "contrib/vol":          lambda_volume    * loss_tr_vol,
                "contrib/cent":         lambda_centroid  * loss_tr_cent,
                "contrib/avgdist":      lambda_avgdist   * loss_tr_avgdist,
                "contrib/avgdist_axis": lambda_avgdist_axis * loss_tr_avgdist_axis,
                "contrib/m2":           lambda_moment2   * loss_tr_moment2,
                "contrib/m3":           lambda_moment3_eff * loss_tr_moment3,
                "contrib/minv":         loss_tr_moment_inv,
                # validation
                "val/seg":              loss_va_seg,
                "val/meanFGDice":       meanFGDice,
                **{f"val/dice_c{k}": float(d)
                   for k, d in enumerate(dices.tolist(), start=1)},
                # schedule
                "schedule/lr":              lr_now(opt),
                "schedule/lambda_m3_eff":   lambda_moment3_eff,
                **{f"schedule/barrier_t_{_n}": _t for _n, _t in _moment_barrier_ts.items()},
                "schedule/barrier_t_shape": barrier_shape.t,
                "schedule/barrier_t_moment": barrier.t,
                **viz_log,
                **collapse_log,
                **dist_log,
            }, step=epoch)

        # periodic checkpoints (rank 0 only)
        model_state = model.module.state_dict() if use_ddp else model.state_dict()
        if is_main and ckpt_every > 0 and (epoch % ckpt_every == 0):
            torch.save(model_state, workdir / f"checkpoint_ep{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": opt.state_dict(),
                    "best_metric": best_metric,
                    "config": cfg,
                },
                workdir / f"checkpoint_ep{epoch:03d}_full.pt"
            )

        metric = meanFGDice
        if metric > best_metric:
            best_metric = metric
            patience = 0
            if is_main:
                torch.save(model_state, workdir / "checkpoint_best.pt")
        else:
            patience += 1

        # frozen once past budget_epoch; never interferes with early stopping
        if budget_epoch > 0 and epoch <= budget_epoch and metric > budget_best_metric:
            budget_best_metric = metric
            budget_best_epoch = epoch
            if is_main:
                torch.save(model_state, workdir / "checkpoint_budget_best.pt")
                with open(workdir / "checkpoint_budget_best.json", "w") as f:
                    json.dump({"budget_epoch": budget_epoch, "epoch": epoch,
                               "meanFGDice": float(metric)}, f, indent=2)
        if is_main and budget_epoch > 0 and epoch == budget_epoch:
            print(f"[budget] frozen at epoch {budget_epoch}: best was epoch "
                  f"{budget_best_epoch} with meanFGDice={budget_best_metric:.4f}")

        if is_main:
            plot_progress(workdir)

        if is_main:
            print(
                f"[E{epoch:03d}] "
                f"train tot={loss_tr_total:.4f} "
                f"seg(log)={loss_tr_seg_logged:.4f} "
                f"seg(bw)={loss_tr_seg_bw:.4f} "
                f"vol={loss_tr_vol:.4f} "
                f"cent={loss_tr_cent:.4f} | "
                f"avgdist={loss_tr_avgdist:.4f} "
                f"avgdist_axis={loss_tr_avgdist_axis:.4f} "
                f"(z={loss_tr_avgdist_axis_z:.4f}, y={loss_tr_avgdist_axis_y:.4f}, x={loss_tr_avgdist_axis_x:.4f}) | "
                # achieved error as a multiple of the tolerance being asked for:
                # <1 satisfied (and <<1 means the tolerance constrains nothing)
                f"err/tol vol=" + "/".join(f"{err_tr_vol[c] / max(volume_tolerance, 1e-12):.2f}" for c in sorted(err_tr_vol)) + " "
                f"cent=" + "/".join(f"{err_tr_cent[c] / max(centroid_tolerance, 1e-12):.2f}" for c in sorted(err_tr_cent)) + " "
                f"axis=" + "/".join(f"{v / max(avgdist_axis_tolerance, 1e-12):.2f}" for v in (loss_tr_avgdist_axis_z, loss_tr_avgdist_axis_y, loss_tr_avgdist_axis_x)) + " | "
                f"m2={loss_tr_moment2:.4f}(err={loss_tr_moment2_err:.2e}) "
                f"m3={loss_tr_moment3:.4f}(err={loss_tr_moment3_err:.2e}) "
                f"minv={loss_tr_moment_inv:.4f}(J1={loss_tr_moment_inv_J1_err:.2e},J2={loss_tr_moment_inv_J2_err:.2e},J3={loss_tr_moment_inv_J3_err:.2e}) | "
                f"val seg={loss_va_seg:.4f} "
                f"meanFGDice={meanFGDice:.4f} | "
                f"best={best_metric:.4f} "
                f"pat={patience}/{early_pat}"
            )

        if patience >= early_pat:
            if is_main:
                print("Early stopping.")
            break

    if is_main:
        model_state = model.module.state_dict() if use_ddp else model.state_dict()
        torch.save(model_state, workdir / "checkpoint_final.pt")
    dt = time.time() - t0_all
    if is_main:
        print(f"Done. Total time: {dt/3600:.2f}h. Workdir: {workdir}")

    if use_wandb and is_main:
        wandb.finish()

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()
    main(args.config)
