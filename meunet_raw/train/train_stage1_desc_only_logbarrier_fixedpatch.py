#!/usr/bin/env python
import sys
from pathlib import Path

# --- make repo root importable ---
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import json
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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from data.dataset_oai_raw_fixedpatch import OAIPairedPatch, load_splits
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


def make_loader(cfg, stems, train: bool, sampler=None):
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

def compute_volume_barrier(
    logits,
    target,
    n_classes,
    volume_classes,
    barrier: LogBarrierLoss,
    volume_tolerance=0.10,
    ignore_index=-1,
    eps=1e-6,
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
    probs = probs * valid

    one_hot, _ = one_hot_labels(target, n_classes, ignore_index)
    one_hot = one_hot.to(device) * valid

    dims = (0, 2, 3, 4)
    pred_mass = probs.sum(dim=dims)       # (C,)
    gt_mass   = one_hot.sum(dim=dims)     # (C,)
    tot_mass  = valid.sum(dim=dims) + eps

    pred_frac = pred_mass / tot_mass
    gt_frac   = gt_mass / tot_mass

    cls_idx = torch.as_tensor(volume_classes, device=device, dtype=torch.long)
    pred_sel = pred_frac[cls_idx]
    gt_sel   = gt_frac[cls_idx]

    lower = gt_sel * (1.0 - volume_tolerance)
    upper = gt_sel * (1.0 + volume_tolerance)

    z_upper = pred_sel - upper   # <= 0 wanted
    z_lower = lower - pred_sel   # <= 0 wanted

    return barrier(z_upper) + barrier(z_lower)


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

    has_class = (gt_mask.sum(dim=(2, 3, 4)) > 0).squeeze(1)
    if not has_class.any():
        return logits.new_tensor(0.0)

    pred_c = pred_centroid[has_class]
    gt_c   = gt_centroid[has_class]

    upper = gt_c + centroid_tolerance
    lower = gt_c - centroid_tolerance

    z_upper = pred_c - upper     # <= 0 wanted
    z_lower = lower - pred_c     # <= 0 wanted

    return barrier(z_upper.reshape(-1)) + barrier(z_lower.reshape(-1))


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
):
    """
    Enforce:
        gt_avgdist - tol <= pred_avgdist <= gt_avgdist + tol

    where avgdist is the mean Euclidean distance to the class centroid.
    """
    device = logits.device
    probs = F.softmax(logits.float(), dim=1)
    valid = (target != ignore_index).float().unsqueeze(1)

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

    has_class = (gt_mask.sum(dim=(2, 3, 4)) > 0).squeeze(1)
    if not has_class.any():
        return logits.new_tensor(0.0)

    pred_d = pred_avgdist[has_class]
    gt_d = gt_avgdist[has_class]

    upper = gt_d + avgdist_tolerance
    lower = gt_d - avgdist_tolerance

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
):
    """
    Enforce:
        gt_axis_spread - tol <= pred_axis_spread <= gt_axis_spread + tol

    where axis_spread is computed per axis as sqrt(E[(coord - centroid_coord)^2]).
    """
    device = logits.device
    probs = F.softmax(logits.float(), dim=1)
    valid = (target != ignore_index).float().unsqueeze(1)

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
    if not has_class.any():
        if return_stats:
            return logits.new_tensor(0.0), logits.new_zeros(3)
        return logits.new_tensor(0.0)

    barrier_loss = logits.new_tensor(0.0)
    axis_stats = []
    for axis_grid, pred_c_axis, gt_c_axis in zip(grids, pred_centroids, gt_centroids):
        pred_axis_var = (pred_w * (axis_grid - pred_c_axis) ** 2).sum(dim=(2, 3, 4)) / pred_sum.squeeze(-1).squeeze(-1).squeeze(-1)
        gt_axis_var = (gt_mask * (axis_grid - gt_c_axis) ** 2).sum(dim=(2, 3, 4)) / gt_sum.squeeze(-1).squeeze(-1).squeeze(-1)
        pred_s = torch.sqrt(pred_axis_var + eps)[has_class]
        gt_s = torch.sqrt(gt_axis_var + eps)[has_class]

        upper = gt_s + avgdist_axis_tolerance
        lower = gt_s - avgdist_axis_tolerance

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

    if use_ddp:
        # Build datasets up front to pass to DistributedSampler
        from data.dataset_oai_raw_fixedpatch import OAIPairedPatch
        expand_factor = float(cfg.get("expand_factor", 1.25))
        fg_prob = float(cfg.get("fg_sampling_prob", 0.5))
        tr_ds = OAIPairedPatch(cfg["images_dir"], cfg["labels_dir"], tr_stems,
                               cfg["patch_size"], expand_factor, fg_prob, True)
        va_ds = OAIPairedPatch(cfg["images_dir"], cfg["labels_dir"], va_stems,
                               cfg["patch_size"], expand_factor, fg_prob, False)
        train_sampler = DistributedSampler(tr_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler   = DistributedSampler(va_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        train_loader  = DataLoader(tr_ds, batch_size=int(cfg["batch_size"]), sampler=train_sampler,
                                   num_workers=int(cfg["num_workers"]), pin_memory=False, drop_last=True)
        val_loader    = DataLoader(va_ds, batch_size=int(cfg["batch_size"]), sampler=val_sampler,
                                   num_workers=int(cfg["num_workers"]), pin_memory=False, drop_last=False)
    else:
        train_sampler = None
        train_loader  = make_loader(cfg, tr_stems, train=True)
        val_loader    = make_loader(cfg, va_stems, train=False)

    # model
    model = MEUNet3D(
        cfg["in_channels"],
        cfg["n_classes"],
        cfg["enc_channels"],
        cfg["dec_channels"],
        cfg["norm"],
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    # loss for monitoring (and optionally training)
    crit = DiceCELoss(cfg["n_classes"])

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("lr", 9e-4)),
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
    lambda_moment_inv = float(cfg.get("lambda_moment_inv", 0.0))
    # linear warmup for moment3: ramp from 0 -> lambda_moment3 between these epochs
    moment3_warmup_start = int(cfg.get("moment3_warmup_start", 0))
    moment3_warmup_end   = int(cfg.get("moment3_warmup_end",   0))  # 0 = no warmup

    centroid_norm = bool(cfg.get("centroid_norm", True))

    # barrier config
    barrier_t = float(cfg.get("barrier_t", 5.0))
    volume_tolerance = float(cfg.get("volume_tolerance", 0.10))
    centroid_tolerance = float(cfg.get("centroid_tolerance", 0.05))
    avgdist_tolerance = float(cfg.get("avgdist_tolerance", 0.05))
    avgdist_axis_tolerance = float(cfg.get("avgdist_axis_tolerance", 0.05))
    moment2_tolerance = float(cfg.get("moment2_tolerance", 0.02))
    moment3_tolerance = float(cfg.get("moment3_tolerance", 0.01))
    moment_inv_tolerance = float(cfg.get("moment_inv_tolerance", 0.01))
    moment2_sqrt_diagonal = bool(cfg.get("moment2_sqrt_diagonal", True))
    moment3_sqrt_diagonal = bool(cfg.get("moment3_sqrt_diagonal", True))
    # "gamma1" (default): weighted average, tractable gradients, no spreading artefacts
    # "fractional": fractional mass normalization, scale-invariant across resolutions
    _moment_norm = cfg.get("moment_normalization", "gamma1")
    moment2_gamma = 5/3 if _moment_norm == "fractional" else 1.0
    moment3_gamma = 2.0  if _moment_norm == "fractional" else 1.0
    moment_inv_gamma = 5/3 if _moment_norm == "fractional" else 1.0
    barrier = LogBarrierLoss(t=barrier_t)

    # training mode switches
    use_seg_loss = bool(cfg.get("use_seg_loss", True))
    monitor_seg_loss = bool(cfg.get("monitor_seg_loss", True))
    detach_for_seg = bool(cfg.get("detach_desc_channels_for_seg", True))

    epochs = int(cfg.get("epochs", 300))
    log_every = int(cfg.get("log_every", 25))
    early_pat = int(cfg.get("early_stop_patience", 40))
    ckpt_every = int(cfg.get("checkpoint_every", 0))

    best_metric = -1.0
    patience = 0

    csv_path = workdir / "progress.csv"
    rows = []
    t0_all = time.time()

    for epoch in range(1, epochs + 1):
        # linear warmup for moment3
        if moment3_warmup_end > moment3_warmup_start and epoch <= moment3_warmup_end:
            if epoch <= moment3_warmup_start:
                lambda_moment3_eff = 0.0
            else:
                progress = (epoch - moment3_warmup_start) / (moment3_warmup_end - moment3_warmup_start)
                lambda_moment3_eff = progress * lambda_moment3
        else:
            lambda_moment3_eff = lambda_moment3

        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        loss_sum_total = 0.0
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
        loss_sum_moment2_err_per_class = {c: 0.0 for c in moment2_classes}
        loss_sum_moment3_err_per_class = {c: 0.0 for c in moment3_classes}
        n_it = 0

        for i, batch in enumerate(train_loader):

            # TRUE EXP-only mode
            expanded = True if shape_on == "exp" else (i % 2 == 0)

            img = (batch["exp_img"] if expanded else batch["std_img"]).to(device, non_blocking=True)
            lbl = (batch["exp_lbl"] if expanded else batch["std_lbl"]).to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg.get("amp", True)):
                out = model(img, expanded=expanded)
                logits = out["logit2"] if expanded else out["logit1"]

                lbl_rs = resize_lbl_to_logits(lbl, logits)

                # monitor-only seg loss
                if monitor_seg_loss:
                    seg_loss_logged = crit(logits, lbl_rs)
                else:
                    seg_loss_logged = logits.new_tensor(0.0)

                # seg loss used for backward
                if use_seg_loss:
                    logits_for_bw = logits
                    if detach_for_seg and desc_classes:
                        logits_for_bw = detach_channels_for_seg_loss(logits, desc_classes)
                    seg_loss_bw = crit(logits_for_bw, lbl_rs)
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
                moment2_loss = logits.new_tensor(0.0)
                moment3_loss = logits.new_tensor(0.0)
                moment_inv_loss = logits.new_tensor(0.0)
                moment2_err = logits.new_tensor(0.0)
                moment3_err = logits.new_tensor(0.0)
                m2_errs_per_class = {c: logits.new_tensor(0.0) for c in moment2_classes}
                m3_errs_per_class = {c: logits.new_tensor(0.0) for c in moment3_classes}

                if apply_shape:
                    if lambda_volume > 0.0 and len(volume_classes) > 0:
                        vol_loss = compute_volume_barrier(
                            logits=logits,
                            target=lbl_rs,
                            n_classes=cfg["n_classes"],
                            volume_classes=volume_classes,
                            barrier=barrier,
                            volume_tolerance=volume_tolerance,
                        )

                    if lambda_centroid > 0.0 and len(centroid_classes) > 0:
                        cents = []
                        for c in centroid_classes:
                            cents.append(
                                compute_centroid_barrier(
                                    logits=logits,
                                    target=lbl_rs,
                                    n_classes=cfg["n_classes"],
                                    centroid_class=c,
                                    barrier=barrier,
                                    centroid_tolerance=centroid_tolerance,
                                    centroid_norm=centroid_norm,
                                )
                            )
                        cent_loss = torch.stack(cents).mean() if len(cents) > 0 else logits.new_tensor(0.0)

                    if lambda_avgdist > 0.0 and len(avgdist_classes) > 0:
                        avgdists = []
                        for c in avgdist_classes:
                            avgdists.append(
                                compute_avgdist_barrier(
                                    logits=logits,
                                    target=lbl_rs,
                                    n_classes=cfg["n_classes"],
                                    distance_class=c,
                                    barrier=barrier,
                                    avgdist_tolerance=avgdist_tolerance,
                                    centroid_norm=centroid_norm,
                                )
                            )
                        avgdist_loss = torch.stack(avgdists).mean() if len(avgdists) > 0 else logits.new_tensor(0.0)

                    if lambda_avgdist_axis > 0.0 and len(avgdist_axis_classes) > 0:
                        avgdist_axes = []
                        avgdist_axis_stats_all = []
                        for c in avgdist_axis_classes:
                            axis_loss_c, axis_stats_c = compute_avgdist_axis_barrier(
                                    logits=logits,
                                    target=lbl_rs,
                                    n_classes=cfg["n_classes"],
                                    distance_class=c,
                                    barrier=barrier,
                                    avgdist_axis_tolerance=avgdist_axis_tolerance,
                                    centroid_norm=centroid_norm,
                                    return_stats=True,
                                )
                            avgdist_axes.append(axis_loss_c)
                            avgdist_axis_stats_all.append(axis_stats_c)
                        avgdist_axis_loss = torch.stack(avgdist_axes).mean() if len(avgdist_axes) > 0 else logits.new_tensor(0.0)
                        avgdist_axis_stats = (
                            torch.stack(avgdist_axis_stats_all).mean(dim=0)
                            if len(avgdist_axis_stats_all) > 0
                            else logits.new_zeros(3)
                        )

                    if lambda_moment2 > 0.0 and len(moment2_classes) > 0:
                        m2s, m2_errs_per_class = [], {}
                        for c in moment2_classes:
                            l_c, e_c = compute_2nd_moment_barrier(
                                logits=logits,
                                target=lbl_rs,
                                n_classes=cfg["n_classes"],
                                moment_class=c,
                                barrier=barrier,
                                moment_tolerance=moment2_tolerance,
                                centroid_norm=centroid_norm,
                                return_stats=True,
                                gamma=moment2_gamma,
                                sqrt_diagonal=moment2_sqrt_diagonal,
                            )
                            m2s.append(l_c)
                            m2_errs_per_class[c] = e_c
                        moment2_loss = torch.stack(m2s).mean() if m2s else logits.new_tensor(0.0)
                        moment2_err  = torch.stack(list(m2_errs_per_class.values())).mean() if m2_errs_per_class else logits.new_tensor(0.0)

                    if len(moment3_classes) > 0:
                        m3s, m3_errs_per_class = [], {}
                        for c in moment3_classes:
                            l_c, e_c = compute_3rd_moment_barrier(
                                logits=logits,
                                target=lbl_rs,
                                n_classes=cfg["n_classes"],
                                moment_class=c,
                                barrier=barrier,
                                moment_tolerance=moment3_tolerance,
                                centroid_norm=centroid_norm,
                                return_stats=True,
                                gamma=moment3_gamma,
                                sqrt_diagonal=moment3_sqrt_diagonal,
                            )
                            m3s.append(l_c)
                            m3_errs_per_class[c] = e_c
                        moment3_loss = torch.stack(m3s).mean() if m3s else logits.new_tensor(0.0)
                        moment3_err  = torch.stack(list(m3_errs_per_class.values())).mean() if m3_errs_per_class else logits.new_tensor(0.0)

                    if lambda_moment_inv > 0.0 and len(moment_inv_classes) > 0:
                        mis = []
                        for c in moment_inv_classes:
                            mis.append(
                                compute_moment_invariants_barrier(
                                    logits=logits,
                                    target=lbl_rs,
                                    n_classes=cfg["n_classes"],
                                    moment_class=c,
                                    barrier=barrier,
                                    inv_tolerance=moment_inv_tolerance,
                                    centroid_norm=centroid_norm,
                                    gamma=moment_inv_gamma,
                                )
                            )
                        moment_inv_loss = torch.stack(mis).mean() if mis else logits.new_tensor(0.0)

                total_loss = (
                    seg_loss_bw
                    + lambda_volume * vol_loss
                    + lambda_centroid * cent_loss
                    + lambda_avgdist * avgdist_loss
                    + lambda_avgdist_axis * avgdist_axis_loss
                    + lambda_moment2 * moment2_loss
                    + lambda_moment3_eff * moment3_loss
                    + lambda_moment_inv * moment_inv_loss
                )

                if not total_loss.requires_grad:
                    continue

            scaler.scale(total_loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss_v = float(total_loss.detach())
            seg_loss_logged_v = float(seg_loss_logged.detach())
            seg_loss_bw_v = float(seg_loss_bw.detach())
            vol_loss_v = float(vol_loss.detach())
            cent_loss_v = float(cent_loss.detach())
            avgdist_loss_v = float(avgdist_loss.detach())
            avgdist_axis_loss_v = float(avgdist_axis_loss.detach())
            avgdist_axis_z_v = float(avgdist_axis_stats[0].detach())
            avgdist_axis_y_v = float(avgdist_axis_stats[1].detach())
            avgdist_axis_x_v = float(avgdist_axis_stats[2].detach())
            moment2_loss_v    = float(moment2_loss.detach())
            moment3_loss_v    = float(moment3_loss.detach())
            moment_inv_loss_v = float(moment_inv_loss.detach())
            moment2_err_v     = float(moment2_err.detach())
            moment3_err_v     = float(moment3_err.detach())

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
            for c, e in m2_errs_per_class.items():
                loss_sum_moment2_err_per_class[c] += float(e.detach())
            for c, e in m3_errs_per_class.items():
                loss_sum_moment3_err_per_class[c] += float(e.detach())
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
                    f"minv={moment_inv_loss_v:.4f} "
                    f"t={barrier_t:.2f} "
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
            del avgdist_axis_loss
            del avgdist_axis_stats
            del moment2_loss
            del moment3_loss
            del moment_inv_loss
            del moment2_err
            del moment3_err
            del m2_errs_per_class
            del m3_errs_per_class
            del img
            del lbl
            del batch

        # All-reduce epoch train loss sums across ranks before computing averages
        if use_ddp:
            n_m2c = len(moment2_classes)
            per_class_m2 = [loss_sum_moment2_err_per_class[c] for c in moment2_classes]
            per_class_m3 = [loss_sum_moment3_err_per_class[c] for c in moment3_classes]
            t = torch.tensor(
                [loss_sum_total, loss_sum_seg_logged, loss_sum_seg_bw, loss_sum_vol,
                 loss_sum_cent, loss_sum_avgdist, loss_sum_avgdist_axis,
                 loss_sum_avgdist_axis_z, loss_sum_avgdist_axis_y, loss_sum_avgdist_axis_x,
                 loss_sum_moment2, loss_sum_moment3, loss_sum_moment_inv,
                 loss_sum_moment2_err, loss_sum_moment3_err, float(n_it)]
                + per_class_m2 + per_class_m3,
                device=device, dtype=torch.float64,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            vals = t.cpu().tolist()
            (loss_sum_total, loss_sum_seg_logged, loss_sum_seg_bw, loss_sum_vol,
             loss_sum_cent, loss_sum_avgdist, loss_sum_avgdist_axis,
             loss_sum_avgdist_axis_z, loss_sum_avgdist_axis_y, loss_sum_avgdist_axis_x,
             loss_sum_moment2, loss_sum_moment3, loss_sum_moment_inv,
             loss_sum_moment2_err, loss_sum_moment3_err, n_it_f) = vals[:16]
            n_it = int(n_it_f)
            for idx, c in enumerate(moment2_classes):
                loss_sum_moment2_err_per_class[c] = vals[16 + idx]
            for idx, c in enumerate(moment3_classes):
                loss_sum_moment3_err_per_class[c] = vals[16 + n_m2c + idx]

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
        loss_tr_moment2_err_per_class = {c: loss_sum_moment2_err_per_class[c] / max(1, n_it) for c in moment2_classes}
        loss_tr_moment3_err_per_class = {c: loss_sum_moment3_err_per_class[c] / max(1, n_it) for c in moment3_classes}

        # validation — all ranks participate, then metrics are all_reduced
        model.eval()
        loss_va_seg = 0.0
        dices_sum = None
        n_va = 0
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
            "barrier_t": barrier_t,
            "volume_tolerance": volume_tolerance,
            "centroid_tolerance": centroid_tolerance,
            "avgdist_tolerance": avgdist_tolerance,
            "avgdist_axis_tolerance": avgdist_axis_tolerance,
            "loss_tr_moment2": loss_tr_moment2,
            "loss_tr_moment3": loss_tr_moment3,
            "loss_tr_moment_inv": loss_tr_moment_inv,
            "loss_tr_moment2_err": loss_tr_moment2_err,
            "loss_tr_moment3_err": loss_tr_moment3_err,
        }
        for k, d in enumerate(dices.tolist(), start=1):
            row[f"dice_c{k}"] = float(d)

        if is_main:
            rows.append(row)
            pd.DataFrame(rows).to_csv(csv_path, index=False)

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
                # per-anatomy moment errors
                **{f"desc/m2_err_c{c}": loss_tr_moment2_err_per_class[c] for c in moment2_classes},
                **{f"desc/m3_err_c{c}": loss_tr_moment3_err_per_class[c] for c in moment3_classes},
                # weighted contributions (lambda × barrier) — what drives the gradient
                "contrib/vol":          lambda_volume    * loss_tr_vol,
                "contrib/cent":         lambda_centroid  * loss_tr_cent,
                "contrib/avgdist":      lambda_avgdist   * loss_tr_avgdist,
                "contrib/avgdist_axis": lambda_avgdist_axis * loss_tr_avgdist_axis,
                "contrib/m2":           lambda_moment2   * loss_tr_moment2,
                "contrib/m3":           lambda_moment3_eff * loss_tr_moment3,
                "contrib/minv":         lambda_moment_inv  * loss_tr_moment_inv,
                # validation
                "val/seg":              loss_va_seg,
                "val/meanFGDice":       meanFGDice,
                **{f"val/dice_c{k}": float(d)
                   for k, d in enumerate(dices.tolist(), start=1)},
                # schedule
                "schedule/lr":              lr_now(opt),
                "schedule/lambda_m3_eff":   lambda_moment3_eff,
                "schedule/barrier_t":       barrier_t,
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
                f"m2={loss_tr_moment2:.4f}(err={loss_tr_moment2_err:.2e}) "
                f"m3={loss_tr_moment3:.4f}(err={loss_tr_moment3_err:.2e}) "
                f"minv={loss_tr_moment_inv:.4f} | "
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
