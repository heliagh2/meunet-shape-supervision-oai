#!/usr/bin/env python
"""
Descriptor statistics for a config, WITHOUT training -- a cheap pre-flight for a
new dataset or a new annotated_fraction.

Runs the same startup path as train_stage1_desc_only_logbarrier_fixedpatch.py:
split -> annotated subsample -> fixed_center (auto = mean GT COM of the annotated
cases only) -> barrier resolutions -> descriptor bank from the annotated cases ->
distribution-matcher targets. Only the bank's own functions are used, so what
this prints is what the run will be constrained by.

  --oracle  also reads the GT of the WITHHELD cases (diagnostic only, as in the
            trainer's distribution plot) and checks whether the TRUE withheld
            pool satisfies the constraints A2.1 imposes on it:
                |mean_pool - bank_mean| <= k * bank_sd / sqrt(n_eff)
                |std_pool  - bank_sd  | <= rel_std_tol * bank_sd
            A violation here means the 30% bank misestimates the population
            more than the tolerance allows, and the method is asked to push the
            pool AWAY from its real distribution.

Needs no GPU. Usage:
  python tools/descriptor_stats.py --config config/amos/AMOS_A2_1_distribution.yaml [--oracle] [--out stats.json]
"""
import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
sys.path.insert(0, str(THIS_DIR.parent / "train"))

import numpy as np
import torch
import yaml

from data.dataset_oai_raw_fixedpatch import (
    OAIPairedPatch, load_splits, compute_average_center, subsample_annotated,
)
from models.meunet3d import MEUNet3D
from train_stage1_desc_only_logbarrier_fixedpatch import (
    as_int_list, build_descriptor_bank, DescriptorMomentMatcher,
)

# axis 0/1/2 of the array (on the AMOS RAS grid that is x/y/z, on OAI it is not)
DESC_NAMES = ["frac", "cent0", "cent1", "cent2", "spread0", "spread1", "spread2"]


def barrier_shapes(cfg, std_in):
    """Spatial shapes of logit1 (STD) / logit2 (EXP), probed exactly as the
    trainer does. Output shapes do not depend on channel widths, so a 2-channel
    copy of the architecture gives the same answer at a fraction of the cost."""
    probe_model = MEUNet3D(int(cfg["in_channels"]), int(cfg["n_classes"]),
                           [2, 2, 2, 2], [2, 2, 2], cfg.get("norm", "batch")).eval()
    exp_in = tuple(int(v) for v in cfg["patch_size"])
    with torch.no_grad():
        std = probe_model(torch.zeros(1, int(cfg["in_channels"]), *std_in), expanded=False)["logit1"]
        exp = probe_model(torch.zeros(1, int(cfg["in_channels"]), *exp_in), expanded=True)["logit2"]
    return {"std": tuple(std.shape[-3:]), "exp": tuple(exp.shape[-3:])}


def pool_stats(samples, classes):
    """(C,7) mean and std of the per-case descriptors in a bank's `samples`."""
    mu, sd = [], []
    for c in classes:
        s = samples[str(int(c))]
        if not s["frac"]:
            mu.append([np.nan] * 7); sd.append([np.nan] * 7); continue
        a = np.concatenate([np.asarray(s["frac"])[:, None], np.asarray(s["cent"]),
                            np.asarray(s["spread"])], axis=1)
        mu.append(a.mean(0)); sd.append(a.std(0))
    return np.asarray(mu), np.asarray(sd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--oracle", action="store_true", help="also read withheld GT (diagnostic only)")
    ap.add_argument("--out", default=None, help="write everything to this JSON")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    names = {int(k): v for k, v in (cfg.get("viz_class_names") or {}).items()}
    nm = lambda c: names.get(int(c), f"c{c}")

    tr, va = load_splits(cfg)
    frac = float(cfg.get("annotated_fraction", 1.0))
    seed = int(cfg.get("annotation_seed", cfg.get("seed", 777)))
    ann, withheld = (subsample_annotated(tr, frac, seed) if frac < 1.0 else (list(tr), []))
    print(f"split: train={len(tr)} val={len(va)} | annotated={len(ann)} withheld={len(withheld)} "
          f"(fraction={frac}, seed={seed})")

    fc = cfg.get("fixed_center", None)
    if isinstance(fc, str) and fc.lower() == "auto":
        fc = compute_average_center(cfg["images_dir"], cfg["labels_dir"], ann)
        print(f"fixed_center: auto -> {fc}  (from the {len(ann)} annotated cases)")
    elif fc is not None:
        fc = tuple(int(round(float(v))) for v in fc)
        print(f"fixed_center: {fc}")

    norm = cfg.get("intensity_norm", "zscore_nonzero")
    ds0 = OAIPairedPatch(cfg["images_dir"], cfg["labels_dir"], ann[:1], cfg["patch_size"],
                         float(cfg.get("expand_factor", 1.25)), 0.5, False,
                         fixed_center=fc, intensity_norm=norm)
    item = ds0[0]
    std_in = tuple(item["std_img"].shape[-3:])
    img = item["std_img"].numpy()
    print(f"STD patch {std_in}, EXP patch {tuple(item['exp_img'].shape[-3:])}; "
          f"image intensities after intensity_norm={norm}: min {img.min():.2f} "
          f"mean {img.mean():.2f} max {img.max():.2f}")
    shapes = barrier_shapes(cfg, std_in)
    print(f"barrier resolutions: std(logit1)={shapes['std']} exp(logit2)={shapes['exp']}")

    classes = sorted(set(as_int_list(cfg.get("volume_classes", cfg.get("desc_only_classes", []))))
                     | set(as_int_list(cfg.get("centroid_classes", [])))
                     | set(as_int_list(cfg.get("avgdist_axis_classes", []))))
    print(f"\nbuilding the bank from {len(ann)} annotated cases, classes {classes} ...")
    bank = build_descriptor_bank(cfg, ann, fc, classes, shapes, workdir=None, verbose=True)

    report = {"config": args.config, "annotated": ann, "withheld": withheld,
              "fixed_center": list(fc) if fc is not None else None,
              "shapes": {k: list(v) for k, v in shapes.items()}, "bank": bank["bank"]}

    for patch in ("std", "exp"):
        print(f"\n=== {patch.upper()} bank (annotated) ===")
        print(f"{'class':<18}{'present':>9}{'frac mean':>12}{'frac sd':>10}{'cv':>6}"
              f"{'  cent (ax0,1,2)':<26}{'spread (ax0,1,2)':<22}")
        for c in classes:
            e = bank["bank"][patch][str(c)]
            cv = e["frac_std"] / max(e["frac"], 1e-12)
            print(f"{nm(c):<18}{e['n']:>4}/{len(ann):<4}{e['frac']:>12.5f}{e['frac_std']:>10.5f}{cv:>6.2f}  "
                  f"{str([round(v, 3) for v in e['cent']]):<26}{str([round(v, 3) for v in e['spread']]):<22}")
            if e["n"] < len(ann):
                print(f"{'':<18}  ^ absent from {len(ann) - e['n']} annotated {patch} patches: the bank "
                      f"averages present cases only, but the pool loss sees every case")

    if cfg.get("unannotated_mode", "drop") == "distribution":
        k = float(cfg.get("dist_k_mean", 2.0)); n_eff = float(cfg.get("dist_n_eff", 64.0))
        rel = float(cfg.get("dist_rel_std_tol", 0.25))
        m = DescriptorMomentMatcher(bank, classes, "cpu", n_eff=n_eff, k_mean=k, rel_std_tol=rel)
        print(f"\n=== distribution-matcher targets (k={k}, n_eff={n_eff:.0f}, rel_std_tol={rel}) ===")
        print("mean tolerance relative to the bank mean, per class (volume fraction):")
        for i, c in enumerate(classes):
            tg = m.target["std"]
            print(f"  {nm(c):<18} std: mu={float(tg['mu'][i, 0]):.5f} +-{float(tg['tol_mu'][i, 0]):.5f} "
                  f"({100 * float(tg['tol_mu'][i, 0] / tg['mu'][i, 0]):.1f}%)  "
                  f"sd={float(tg['sd'][i, 0]):.5f} +-{float(tg['tol_sd'][i, 0]):.5f}")

        if args.oracle and withheld:
            print(f"\n=== ORACLE: does the TRUE withheld pool ({len(withheld)} cases) satisfy them? ===")
            oracle = build_descriptor_bank(cfg, withheld, fc, classes, shapes, workdir=None, verbose=False)
            report["oracle_bank"] = oracle["bank"]
            report["oracle_check"] = {}
            for patch in ("std", "exp"):
                tg = m.target[patch]
                mu_o, sd_o = pool_stats(oracle["samples"][patch], classes)
                z_mu = np.abs(mu_o - tg["mu"].numpy()) / tg["tol_mu"].numpy()
                z_sd = np.abs(sd_o - tg["sd"].numpy()) / tg["tol_sd"].numpy()
                act = m.active[patch].numpy()
                print(f"  {patch.upper()}: |error| / tolerance  (> 1.00 = the real pool VIOLATES it)")
                print(f"  {'class':<18}" + "".join(f"{d:>13}" for d in DESC_NAMES))
                for i, c in enumerate(classes):
                    cells = [("  -  " if not act[i, j] else f"{z_mu[i, j]:.2f}/{z_sd[i, j]:.2f}")
                             for j in range(7)]
                    flag = "  <-" if np.nanmax(np.where(act[i], np.maximum(z_mu[i], z_sd[i]), 0)) > 1 else ""
                    print(f"  {nm(c):<18}" + "".join(f"{x:>13}" for x in cells) + flag)
                viol_mu = float(np.mean((z_mu > 1)[act])); viol_sd = float(np.mean((z_sd > 1)[act]))
                print(f"  {patch.upper()}: real pool violates {100 * viol_mu:.0f}% of mean constraints, "
                      f"{100 * viol_sd:.0f}% of std constraints   (cells are mean/std)")
                report["oracle_check"][patch] = {"z_mu": z_mu.tolist(), "z_sd": z_sd.tolist(),
                                                 "viol_mu": viol_mu, "viol_sd": viol_sd}

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
