#!/usr/bin/env python

import os
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from data.dataset_oai_raw import OAIPairedPatch, load_splits
from models.meunet3d import MEUNet3D
from losses.dice_ce import DiceCELoss


# helpers

def _f(x, d):
    """Safe float conversion with default."""
    try:
        return float(x)
    except Exception:
        return d


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_now(optimizer):
    for pg in optimizer.param_groups:
        return float(pg.get("lr", 0.0))
    return 0.0


def one_hot_labels(target, n_classes, ignore_index=-1):
    """
    target: (B,Z,Y,X) int with possible -1
    Returns:
      one_hot: (B,C,Z,Y,X) float
      valid_mask: (B,Z,Y,X) bool
    """
    B, Z, Y, X = target.shape
    mask = (target != ignore_index)            # valid voxels
    t = target.clamp(min=0)                    # set -1 -> 0 before one_hot; will mask them out
    oh = F.one_hot(t.long(), num_classes=n_classes).permute(0, 4, 1, 2, 3).float()
    oh = oh * mask.unsqueeze(1)                # zero out ignored
    return oh, mask


@torch.no_grad()
def soft_dice_per_class(logits, target, n_classes, ignore_index=-1, eps=1e-6):
    """
    nnUNet-style 'Pseudo Dice': softmax probs; exclude background (class 0).

    Returns:
      list of per-class dice for classes 1..n_classes-1
    """
    probs = F.softmax(logits.float(), dim=1)                   # (B,C, ...)
    tgt_oh, valid_mask = one_hot_labels(target, n_classes, ignore_index)

    probs_fg = probs[:, 1:, ...]                               # drop bg
    tgt_fg   = tgt_oh[:, 1:, ...]
    vm = valid_mask.unsqueeze(1)

    probs_fg = probs_fg * vm
    tgt_fg   = tgt_fg * vm

    dims = tuple(range(2, probs_fg.ndim))
    inter = (probs_fg * tgt_fg).sum(dim=dims)
    p_sum = probs_fg.sum(dim=dims)
    t_sum = tgt_fg.sum(dim=dims)

    dice = (2 * inter + eps) / (p_sum + t_sum + eps)           # (B, C-1)
    return dice.mean(dim=0).cpu().double().tolist()


def make_loader(cfg, stems, train: bool):
    expand_factor = cfg.get("expand_factor", 1.25)        # default = 1.25
    fg_prob = cfg.get("fg_sampling_prob", 0.5)            # default = 0.5

    ds = OAIPairedPatch(
        cfg["images_dir"],
        cfg["labels_dir"],
        stems,
        cfg["patch_size"],
        expand_factor,
        fg_prob,
        train,
    )

    sam = RandomSampler(ds, replacement=True, num_samples=len(stems)) if train else None
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=(sam is None and train),
        sampler=sam,
        num_workers=cfg["num_workers"],
        pin_memory=False,
    )


def validate(model, loader, device, n_classes):
    model.eval()
    loss_ce_dice = DiceCELoss(n_classes)

    tot_loss, n_batches = 0.0, 0
    dices_acc = None

    with torch.no_grad():
        for batch in loader:
            x = batch["std_img"].to(device, non_blocking=True)
            y = batch["std_lbl"].to(device, non_blocking=True)

            out = model(x, expanded=False)["logit1"]
            loss = loss_ce_dice(out, y)

            tot_loss += float(loss)
            n_batches += 1

            d_list = soft_dice_per_class(out, y, n_classes)
            v = torch.tensor(d_list, dtype=torch.double)
            dices_acc = v if dices_acc is None else dices_acc + v

    if n_batches == 0:
        mean_loss = 0.0
        per_class = [0.0] * (n_classes - 1)
    else:
        mean_loss = tot_loss / n_batches
        per_class = (dices_acc / n_batches).tolist()

    mean_fg = float(np.mean(per_class)) if per_class else 0.0
    return mean_loss, mean_fg, per_class

def load_config(path: str):
    cfg_path = Path(path)
    if cfg_path.suffix == ".json":
        return json.loads(cfg_path.read_text())
    else:
        import yaml
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML/JSON config path")
    args = ap.parse_args()

    cfg = load_config(args.config)

    workdir = Path(cfg["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    csv_path = workdir / "progress.csv"

    # CSV header
    if not csv_path.exists():
        cols = ["epoch", "lr", "loss_tr", "loss_val", "meanFGDice", "epoch_time_s"] + \
               [f"dice_c{c}" for c in range(1, cfg["n_classes"])]
        with open(csv_path, "w") as f:
            f.write(",".join(cols) + "\n")

    seed_all(cfg.get("seed", 777))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data splits & loaders
    train_stems, val_stems = load_splits(cfg)
    print(f"[INFO] #train stems = {len(train_stems)}, #val stems = {len(val_stems)}")

    train_loader = make_loader(cfg, train_stems, train=True)
    val_loader   = make_loader(cfg, val_stems,   train=False)

    # model, optimizer, scheduler, loss
    model = MEUNet3D(
        cfg["in_channels"],
        cfg["n_classes"],
        cfg["enc_channels"],
        cfg["dec_channels"],
        cfg["norm"],
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=_f(cfg.get("lr", 1e-3), 1e-3),
        weight_decay=_f(cfg.get("weight_decay", 1e-5), 1e-5),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=cfg["epochs"],
        eta_min=_f(cfg.get("min_lr", 1e-6), 1e-6),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("amp", True))
    crit = DiceCELoss(cfg["n_classes"])

    log_every = int(cfg.get("log_every", 25))
    patience = int(cfg.get("early_stop_patience", 40))

    best, bad = -1.0, 0

    for epoch in range(1, cfg["epochs"] + 1):
        ep_t0 = time.time()
        model.train()
        loss_sum, n_it = 0.0, 0

        for i, batch in enumerate(train_loader):
            # Alternate EXP/STD like original MeUNet paper
            expanded = (i % 2 == 0)
            img = (batch["exp_img"] if expanded else batch["std_img"]).to(device, non_blocking=True)
            lbl = (batch["exp_lbl"] if expanded else batch["std_lbl"]).to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.get("amp", True)):
                out = model(img, expanded=expanded)
                logits = out["logit2"] if expanded else out["logit1"]
                loss = crit(logits, lbl)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_sum += float(loss)
            n_it += 1

            if i % log_every == 0:
                print(f"[ep {epoch:03d} it {i:05d}] "
                      f"{'EXP' if expanded else 'STD'} loss={float(loss):.4f}")

        # validation + logging
        loss_tr = loss_sum / max(1, n_it)
        loss_val, mean_fg, per_class = validate(model, val_loader, device, cfg["n_classes"])
        lr = lr_now(opt)
        ep_time = time.time() - ep_t0
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"{ts}: Epoch {epoch}")
        print(f"{ts}: Current learning rate: {lr:.5f}")
        print(f"{ts}: train_loss {loss_tr:.4f}")
        print(f"{ts}: val_loss {loss_val:.4f}")
        print(f"{ts}: Pseudo dice [{', '.join(f'{d:.4f}' for d in per_class)}]")
        print(f"{ts}: Epoch time: {ep_time:.2f} s")

        # append CSV row
        with open(csv_path, "a") as f:
            row = [epoch, lr, loss_tr, loss_val, mean_fg, ep_time] + per_class
            f.write(",".join(str(x) for x in row) + "\n")

        #plotting
        try:
            import pandas as pd
            import matplotlib.pyplot as plt

            df = pd.read_csv(csv_path)

            # moving average of meanFGDice (like dotted curve)
            if len(df) >= 3:
                df["meanFGDice_ma"] = df["meanFGDice"].rolling(window=20, min_periods=1).mean()
            else:
                df["meanFGDice_ma"] = df["meanFGDice"]

            fig = plt.figure(figsize=(8, 16))

            # (1) losses + mean dice + moving avg
            ax1 = fig.add_subplot(3, 1, 1)
            ax1.plot(df["epoch"], df["loss_tr"], label="loss_tr")
            ax1.plot(df["epoch"], df["loss_val"], label="loss_val")
            ax1.set_ylabel("loss")
            ax1.legend(loc="upper left")

            ax1b = ax1.twinx()
            ax1b.plot(df["epoch"], df["meanFGDice"], linestyle="dotted", label="meanFGDice")
            ax1b.plot(df["epoch"], df["meanFGDice_ma"], label="meanFGDice (mov.avg.)")
            ax1b.set_ylabel("pseudo dice")
            ax1b.legend(loc="lower right")

            # (2) per-class dice (foreground only)
            ax2 = fig.add_subplot(3, 1, 2)
            dice_cols = [c for c in df.columns if c.startswith("dice_c")]
            for c in dice_cols:
                ax2.plot(df["epoch"], df[c], label=c)
            ax2.set_ylabel("per-class pseudo dice")
            ax2.legend(loc="lower right", ncol=2)

            # (3) learning rate
            ax3 = fig.add_subplot(3, 1, 3)
            ax3.plot(df["epoch"], df["lr"], label="learning rate")
            ax3.set_ylabel("learning rate")
            ax3.set_xlabel("epoch")

            fig.tight_layout()
            fig.savefig(workdir / "progress.png", dpi=180)
            plt.close(fig)

        except Exception as e:
            print(f"[plot warning] {e}")

        # checkpointing & early stopping
        if mean_fg > best:
            best, bad = mean_fg, 0
            # main checkpoint used by inference scripts
            ckpt_main = workdir / "checkpoint_best.pt"
            torch.save(model.state_dict(), ckpt_main)

            # optional human-readable snapshot
            ckpt_named = workdir / f"best_{best:.4f}.pt"
            torch.save(model.state_dict(), ckpt_named)

            print(f"[checkpoint] new best meanFGDice={best:.4f} saved to {ckpt_main.name}")
        else:
            bad += 1
            if bad >= patience:
                print(f"[early-stop] best meanFGDice={best:.4f}")
                break

        scheduler.step()

    print(f"[done] best val meanFGDice = {best:.4f}")


if __name__ == "__main__":
    main()
