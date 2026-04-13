#!/usr/bin/env python
import sys
from pathlib import Path

# make repo root importable
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.dataset_oai_raw import OAIPairedPatch, load_splits
from models.meunet3d import MEUNet3D
from losses.dice_ce import DiceCELoss
from losses.penalties import quadratic_penalty


#helpers

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


def make_loader(cfg, stems, train: bool):
    expand_factor = cfg.get("expand_factor", 1.25)
    fg_prob = cfg.get("fg_sampling_prob", 0.5)

    ds = OAIPairedPatch(
        cfg["images_dir"],
        cfg["labels_dir"],
        stems,
        cfg["patch_size"],
        expand_factor,
        fg_prob,
        train,
    )

    num = len(stems)
    if train and cfg.get("shape_on", "exp") == "exp" and not cfg.get("use_seg_loss", True):
        num *= 2  # double iterations/epoch for exp-only mode
    sam = RandomSampler(ds, replacement=True, num_samples=num) if train else None
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=(sam is None and train),
        sampler=sam,
        num_workers=cfg["num_workers"],
        pin_memory=False,
    )


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


def compute_volume_penalty(
    logits,
    target,
    n_classes,
    volume_classes,
    ignore_index=-1,
    eps=1e-6,
):
    """
    Quadratic penalty on global volume fractions for selected classes.
    logits: (B,C,D,H,W)
    target: (B,D,H,W) MUST match logits spatial size
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

    diff = pred_sel - gt_sel
    return quadratic_penalty(diff, torch.zeros_like(diff)).mean()


def compute_centroid_penalty(
    logits,
    target,
    n_classes,
    centroid_class,
    centroid_norm=True,
    ignore_index=-1,
    eps=1e-6,
):
    """
    Quadratic penalty on centroid (z,y,x) of a single class.
    logits: (B,C,D,H,W)
    target: (B,D,H,W) MUST match logits spatial size
    Returns scalar.
    """
    device = logits.device
    probs = F.softmax(logits.float(), dim=1)  # (B,C,D,H,W)
    valid = (target != ignore_index).float().unsqueeze(1)  # (B,1,D,H,W)

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

    diff = pred_centroid[has_class] - gt_centroid[has_class]
    return quadratic_penalty(diff.reshape(-1), torch.zeros_like(diff).reshape(-1)).mean()


def compute_avg_distance_to_centroid_penalty(
    logits,
    target,
    n_classes,
    distance_class,
    centroid_norm=True,
    ignore_index=-1,
    eps=1e-6,
):
    """
    Quadratic penalty on the mean Euclidean distance to the class centroid.
    logits: (B,C,D,H,W)
    target: (B,D,H,W) MUST match logits spatial size
    Returns scalar average to centroid
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
    )  # (B,3,D,H,W)

    pred_sum = pred_w.sum(dim=(2, 3, 4), keepdim=True) + eps
    gt_sum = gt_mask.sum(dim=(2, 3, 4), keepdim=True) + eps

    pred_centroid = (pred_w * coords).sum(dim=(2, 3, 4), keepdim=True) / pred_sum
    gt_centroid = (gt_mask * coords).sum(dim=(2, 3, 4), keepdim=True) / gt_sum

    pred_dist = torch.linalg.norm(coords - pred_centroid, dim=1, keepdim=True)
    gt_dist = torch.linalg.norm(coords - gt_centroid, dim=1, keepdim=True)

    pred_mean_dist = (pred_w * pred_dist).sum(dim=(2, 3, 4)) / pred_sum.squeeze(-1).squeeze(-1).squeeze(-1)
    gt_mean_dist = (gt_mask * gt_dist).sum(dim=(2, 3, 4)) / gt_sum.squeeze(-1).squeeze(-1).squeeze(-1)

    has_class = (gt_mask.sum(dim=(2, 3, 4)) > 0).squeeze(1)
    if not has_class.any():
        return logits.new_tensor(0.0)

    diff = pred_mean_dist[has_class] - gt_mean_dist[has_class]
    return quadratic_penalty(diff.reshape(-1), torch.zeros_like(diff).reshape(-1)).mean()


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


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = json.load(f) if cfg_path.endswith(".json") else __import__("yaml").safe_load(f)

    workdir = Path(cfg["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)

    with open(workdir / "config_snapshot.yaml", "w") as f:
        __import__("yaml").safe_dump(cfg, f)

    seed_all(int(cfg.get("seed", 777)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

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
    print(f"Fold {fold}: train={len(tr_stems)} val={len(va_stems)}")

    train_loader = make_loader(cfg, tr_stems, train=True)
    val_loader   = make_loader(cfg, va_stems, train=False)

    # model
    model = MEUNet3D(
        cfg["in_channels"],
        cfg["n_classes"],
        cfg["enc_channels"],
        cfg["dec_channels"],
        cfg["norm"],
    ).to(device)

    # loss for monitoring (and optionally training)
    crit = DiceCELoss(cfg["n_classes"])

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("lr", 9e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("amp", True))

    #descriptor config (lists supported)
    # Which classes are "descriptor-only" (for backwards compat: desc_only_class or desc_only_classes)
    desc_classes = as_int_list(cfg.get("desc_only_classes", cfg.get("desc_only_class", [1])))

    # volume loss can apply to subset; default = desc_classes
    volume_classes = as_int_list(cfg.get("volume_classes", desc_classes), default=desc_classes)

    # centroid loss can apply to subset; default = desc_classes
    centroid_classes = as_int_list(cfg.get("centroid_classes", cfg.get("centroid_class", desc_classes)), default=desc_classes)
    avgdist_classes = as_int_list(cfg.get("avgdist_classes", cfg.get("distance_classes", desc_classes)), default=desc_classes)

    shape_on = cfg.get("shape_on", "exp")  # "std" | "exp" | "both"
    lambda_volume = float(cfg.get("lambda_volume", 1.0))
    lambda_centroid = float(cfg.get("lambda_centroid", 1.0))
    lambda_avgdist = float(cfg.get("lambda_avgdist", cfg.get("lambda_distance", 0.0)))
    centroid_norm = bool(cfg.get("centroid_norm", True))

    # training mode switches
    use_seg_loss = bool(cfg.get("use_seg_loss", True))              # if False: no seg loss in total/backward
    monitor_seg_loss = bool(cfg.get("monitor_seg_loss", True))      # if False: skip computing seg loss entirely

    # if you still want seg loss but not for desc_classes: detach those channels for seg backward
    detach_for_seg = bool(cfg.get("detach_desc_channels_for_seg", True))

    epochs = int(cfg.get("epochs", 300))
    log_every = int(cfg.get("log_every", 25))
    early_pat = int(cfg.get("early_stop_patience", 40))

    best_metric = -1.0
    patience = 0

    csv_path = workdir / "progress.csv"
    rows = []
    t0_all = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum_total = 0.0
        loss_sum_seg_logged = 0.0
        loss_sum_seg_bw = 0.0
        loss_sum_vol = 0.0
        loss_sum_cent = 0.0
        loss_sum_avgdist = 0.0
        n_it = 0

        for i, batch in enumerate(train_loader):

            #TRUE EXP-only mode
            expanded = True if shape_on == "exp" else (i % 2 == 0)
            img = (batch["exp_img"] if expanded else batch["std_img"]).to(device, non_blocking=True)
            lbl = (batch["exp_lbl"] if expanded else batch["std_lbl"]).to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg.get("amp", True)):
                out = model(img, expanded=expanded)
                logits = out["logit2"] if expanded else out["logit1"]

                # match GT resolution to logits
                lbl_rs = resize_lbl_to_logits(lbl, logits)

                # monitor-only seg loss (always computed from full logits unless disabled)
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

                # shape losses: only on selected patch type
                apply_shape = (
                    (shape_on == "std" and (not expanded)) or
                    (shape_on == "exp" and expanded) or
                    (shape_on == "both")
                )

                vol_loss = logits.new_tensor(0.0)
                cent_loss = logits.new_tensor(0.0)
                avgdist_loss = logits.new_tensor(0.0)

                if apply_shape:
                    if lambda_volume > 0.0 and len(volume_classes) > 0:
                        vol_loss = compute_volume_penalty(
                            logits, lbl_rs, cfg["n_classes"], volume_classes
                        )
                    if lambda_centroid > 0.0 and len(centroid_classes) > 0:
                        # sum centroid penalties over classes (average for scale stability)
                        cents = []
                        for c in centroid_classes:
                            cents.append(
                                compute_centroid_penalty(
                                    logits, lbl_rs, cfg["n_classes"], c, centroid_norm=centroid_norm
                                )
                            )
                        cent_loss = torch.stack(cents).mean() if len(cents) > 0 else logits.new_tensor(0.0)
                    if lambda_avgdist > 0.0 and len(avgdist_classes) > 0:
                        avgdists = []
                        for c in avgdist_classes:
                            avgdists.append(
                                compute_avg_distance_to_centroid_penalty(
                                    logits, lbl_rs, cfg["n_classes"], c, centroid_norm=centroid_norm
                                )
                            )
                        avgdist_loss = torch.stack(avgdists).mean() if len(avgdists) > 0 else logits.new_tensor(0.0)

                total_loss = (
                    seg_loss_bw
                    + lambda_volume * vol_loss
                    + lambda_centroid * cent_loss
                    + lambda_avgdist * avgdist_loss
                )
                if not total_loss.requires_grad:
                    # This happens for STD iterations when shape_on="exp" and use_seg_loss=False
                    # No gradient signal -> skip backward/step safely
                    continue

            scaler.scale(total_loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_sum_total += float(total_loss)
            loss_sum_seg_logged += float(seg_loss_logged)
            loss_sum_seg_bw += float(seg_loss_bw)
            loss_sum_vol += float(vol_loss)
            loss_sum_cent += float(cent_loss)
            loss_sum_avgdist += float(avgdist_loss)
            n_it += 1

            if i % log_every == 0:
                print(
                    f"[E{epoch:03d} i{i:04d} {'EXP' if expanded else 'STD'}] "
                    f"tot={float(total_loss):.4f} "
                    f"seg(log)={float(seg_loss_logged):.4f} seg(bw)={float(seg_loss_bw):.4f} "
                    f"vol={float(vol_loss):.4f} cent={float(cent_loss):.4f} avgdist={float(avgdist_loss):.4f} "
                    f"lr={lr_now(opt):.2e}"
                )

        # epoch averages
        loss_tr_total = loss_sum_total / max(1, n_it)
        loss_tr_seg_logged = loss_sum_seg_logged / max(1, n_it)
        loss_tr_seg_bw = loss_sum_seg_bw / max(1, n_it)
        loss_tr_vol = loss_sum_vol / max(1, n_it)
        loss_tr_cent = loss_sum_cent / max(1, n_it)
        loss_tr_avgdist = loss_sum_avgdist / max(1, n_it)

        # validation (STD only; expanded=False), keep comparable metrics
        model.eval()
        loss_va_seg = 0.0
        dices = None
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
                dices = d if dices is None else (dices + d)

        loss_va_seg = loss_va_seg / max(1, len(val_loader)) if monitor_seg_loss else 0.0
        dices = dices / max(1, len(val_loader))
        meanFGDice = float(dices.mean()) if dices is not None else 0.0

        row = {
            "epoch": epoch,
            "loss_tr_total": loss_tr_total,
            "loss_tr_seg_logged": loss_tr_seg_logged,
            "loss_tr_seg_bw": loss_tr_seg_bw,
            "loss_tr_vol": loss_tr_vol,
            "loss_tr_cent": loss_tr_cent,
            "loss_tr_avgdist": loss_tr_avgdist,
            "loss_va_seg": loss_va_seg,
            "meanFGDice": meanFGDice,
            "lr": lr_now(opt),
            "use_seg_loss": int(use_seg_loss),
            "monitor_seg_loss": int(monitor_seg_loss),
        }
        for k, d in enumerate(dices.tolist(), start=1):
            row[f"dice_c{k}"] = float(d)

        rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        metric = meanFGDice
        if metric > best_metric:
            best_metric = metric
            patience = 0
            torch.save(model.state_dict(), workdir / "checkpoint_best.pt")
        else:
            patience += 1

        plot_progress(workdir)

        print(
            f"[E{epoch:03d}] "
            f"train tot={loss_tr_total:.4f} seg(log)={loss_tr_seg_logged:.4f} seg(bw)={loss_tr_seg_bw:.4f} "
            f"vol={loss_tr_vol:.4f} cent={loss_tr_cent:.4f} | "
            f"val seg={loss_va_seg:.4f} meanFGDice={meanFGDice:.4f} | best={best_metric:.4f} pat={patience}/{early_pat}"
        )

        if patience >= early_pat:
            print("Early stopping.")
            break

    torch.save(model.state_dict(), workdir / "checkpoint_final.pt")
    dt = time.time() - t0_all
    print(f"Done. Total time: {dt/3600:.2f}h. Workdir: {workdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()
    main(args.config)
