#!/usr/bin/env python
"""
Paired comparison of runs on the SAME test scans, from the per-case Dice CSVs
written by eval_dice_oai.py (eval_amos.sbatch: test_<ckpt>[_exp]_dice_native.csv).

Each run is compared with the FIRST one (the reference). Per scan, the difference
Dice(run) - Dice(ref) is taken, so scan difficulty -- shared by both models --
cancels, and the test asks whether the run is better on most scans rather than
whether two noisy means differ. Reported, overall and per class:
  delta    difference of the means, computed as eval_dice_oai.py does
           (per-class mean over cases, then mean over classes)
  95% CI   paired bootstrap over scans (cases resampled together for both runs)
  +/-      scans where the run is better / worse
  p        Wilcoxon signed-rank on the per-scan differences; for the overall row
           the per-scan value is the mean over the classes present in that scan.
           Per-class p are Holm-corrected over the classes (p_holm).
This covers test-set sampling only. It says nothing about training randomness
(seed), which needs repeated runs.

Each run is LABEL=PATH or PATH, where PATH is a CSV or a run workdir (then
test_<ckpt>[_exp]_dice_native.csv inside it is used):

  python eval/paired_test.py \\
      results/AMOS_A1_13_t300 results/AMOS_A1_14_conc_all results/AMOS_A2_1c_tight_std
"""
import argparse
import csv
import os

import numpy as np
from scipy.stats import wilcoxon


def load(path):
    rows = list(csv.DictReader(open(path)))
    classes = [k for k in rows[0] if k.startswith("dice_c")]
    data = {r["case"]: [float(r[k]) if r[k] != "" else np.nan for k in classes] for r in rows}
    return [int(k[6:]) for k in classes], data


def eval_mean(a):
    """eval_dice_oai.py's meanFGDice: per-class nanmean over cases, then mean over classes.
    Works on (..., cases, classes)."""
    return np.nanmean(np.nanmean(a, axis=-2), axis=-1)


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = int(ok.sum())
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx])
    tmp = np.empty(m)
    tmp[idx] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out


def wilcoxon_p(d):
    d = d[~np.isnan(d)]
    if d.size < 2 or np.all(d == 0):
        return np.nan
    return float(wilcoxon(d).pvalue)


def resolve(spec, ckpt, head):
    label, _, path = spec.rpartition("=")
    if os.path.isdir(path):
        suffix = "" if head == "std" else f"_{head}"
        path = os.path.join(path, f"test_{ckpt}{suffix}_dice_native.csv")
    return (label or os.path.basename(os.path.dirname(path)) or path), path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="LABEL=PATH or PATH (CSV or workdir); the first is the reference")
    ap.add_argument("--ckpt", default="checkpoint_best", help="checkpoint tag when a workdir is given")
    ap.add_argument("--head", default="std", choices=("std", "exp"), help="head when a workdir is given")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--names", default=None, help="dataset.json for class names (optional)")
    args = ap.parse_args()

    if len(args.runs) < 2:
        ap.error("need a reference and at least one run to compare")
    names = {}
    if args.names and os.path.exists(args.names):
        import json
        names = {int(k): v for k, v in json.load(open(args.names)).get("labels", {}).items()}

    runs = [resolve(s, args.ckpt, args.head) for s in args.runs]
    loaded = [(label, *load(path)) for label, path in runs]
    classes = loaded[0][1]
    if any(c != classes for _, c, _ in loaded):
        raise ValueError("runs were evaluated on different class lists")
    cases = sorted(set.intersection(*(set(d) for _, _, d in loaded)))
    for label, _, d in loaded:
        if len(d) != len(cases):
            print(f"[WARN] {label}: {len(d) - len(cases)} cases not shared by every run, dropped")
    A = {label: np.array([d[c] for c in cases]) for label, _, d in loaded}   # (cases, classes)
    ref = loaded[0][0]
    rng = np.random.default_rng(args.seed)
    boot = rng.integers(0, len(cases), size=(args.n_boot, len(cases)))
    nm = lambda c: f"{c}:{names[c]}" if c in names else f"class {c}"

    print(f"{len(cases)} paired scans, reference = {ref} (head={args.head})\n")
    print(f"{'run':<34}{'mean':>7}{'delta':>8}{'95% CI':>18}{'+/-':>9}{'p':>9}")
    print(f"{ref:<34}{eval_mean(A[ref]):>7.3f}")
    for label, _, _ in loaded[1:]:
        X, R = A[label], A[ref]
        delta = eval_mean(X) - eval_mean(R)
        bd = eval_mean(X[boot]) - eval_mean(R[boot])
        lo, hi = np.percentile(bd, [2.5, 97.5])
        d_case = np.nanmean(X, axis=1) - np.nanmean(R, axis=1)
        flag = "  *" if lo > 0 or hi < 0 else ""
        print(f"{label:<34}{eval_mean(X):>7.3f}{delta:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]"
              f"{f'{(d_case > 0).sum()}/{(d_case < 0).sum()}':>9}{wilcoxon_p(d_case):>9.4f}{flag}")
    print("  * 95% CI excludes 0")

    for label, _, _ in loaded[1:]:
        X, R = A[label], A[ref]
        print(f"\n=== {label} vs {ref}: per class ===")
        print(f"  {'class':<24}{'ref':>7}{'run':>7}{'delta':>8}{'95% CI':>18}{'+/-':>9}{'p':>9}{'p_holm':>9}")
        D = X - R
        ps = [wilcoxon_p(D[:, j]) for j in range(len(classes))]
        ph = holm(ps)
        bx, br = np.nanmean(X[boot], axis=1), np.nanmean(R[boot], axis=1)     # (B, classes)
        for j, c in enumerate(classes):
            lo, hi = np.nanpercentile(bx[:, j] - br[:, j], [2.5, 97.5])
            d = D[:, j]
            mr, mx = np.nanmean(R[:, j]), np.nanmean(X[:, j])
            flag = "  *" if ph[j] < 0.05 else ""
            print(f"  {nm(c):<24}{mr:>7.3f}{mx:>7.3f}{mx - mr:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]"
                  f"{f'{(d > 0).sum()}/{(d < 0).sum()}':>9}{ps[j]:>9.4f}{ph[j]:>9.4f}{flag}")
        print("  * Holm-corrected p < 0.05")


if __name__ == "__main__":
    main()
