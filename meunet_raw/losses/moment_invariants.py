"""
Moment-based shape descriptor barriers for 3-D soft segmentation.

Moment hierarchy (building on existing centroid / avgdist barriers):
  0th order  ->  total mass (volume)        -- already in compute_volume_barrier
  1st order  ->  centroid                   -- already in compute_centroid_barrier
  2nd order  ->  covariance matrix          -- compute_2nd_moment_barrier
  rotation-invariant 2nd order              -- compute_moment_invariants_barrier
  3rd order  ->  skewness / asymmetry       -- compute_3rd_moment_barrier

Normalization is controlled by the `gamma` parameter in each function:

  gamma = 1.0  (default): weighted average  eta = mu / M000
      -- per-class size invariant, values O(1e-3), tractable gradients

  gamma = 5/3 or 2.0 (fractional mass): eta = mu / (f^gamma * vol), f = M000/vol
      -- additionally scale-invariant across resolutions, values O(1e-2)
      -- can cause "spreading" failure mode for large classes (see training notes)

Configure via `moment_normalization: "gamma1"` (default) or `"fractional"` in the
YAML config. The training script translates this to the correct gamma values.

sqrt_diagonal option (default True):
  For 2nd order diagonal (mu_200/020/002): regular sqrt → per-axis std dev,
    same gradient amplification as avgdist_axis (helpful for thin structures).
  For 3rd order pure cubic (mu_300/030/003): signed sqrt → preserves skewness
    direction while amplifying gradient when skewness is small.
  Off-diagonal / mixed terms are always kept raw (can be negative).

Centroid computation uses x_c = M_100/M_000 (unchanged, eq. 11-36).
"""

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ helpers --

def _one_hot(target, n_classes, ignore_index=-1):
    mask = target != ignore_index
    t = target.clamp(min=0)
    oh = F.one_hot(t.long(), num_classes=n_classes).permute(0, 4, 1, 2, 3).float()
    return oh * mask.unsqueeze(1), mask


def _coord_grids(B, D, H, W, device, centroid_norm):
    """Return (B, 3, D, H, W) coordinate grids in [0,1] or voxel indices."""
    if centroid_norm:
        zz = torch.linspace(0, 1, D, device=device).view(1, 1, D, 1, 1).expand(B, 1, D, H, W)
        yy = torch.linspace(0, 1, H, device=device).view(1, 1, 1, H, 1).expand(B, 1, D, H, W)
        xx = torch.linspace(0, 1, W, device=device).view(1, 1, 1, 1, W).expand(B, 1, D, H, W)
    else:
        zz = torch.arange(D, device=device).float().view(1, 1, D, 1, 1).expand(B, 1, D, H, W)
        yy = torch.arange(H, device=device).float().view(1, 1, 1, H, 1).expand(B, 1, D, H, W)
        xx = torch.arange(W, device=device).float().view(1, 1, 1, 1, W).expand(B, 1, D, H, W)
    return torch.cat([zz, yy, xx], dim=1).contiguous()


def _weighted_moment(w, term, w_sum, gamma=1.0):
    """Normalized central moment.

    gamma=1 (default): eta = mu / M000  (weighted average, per-class invariant)
    gamma>1:           eta = mu / (f^gamma * vol), f = M000/vol  (fractional mass)

    Returns (B, 1).
    """
    raw = (w * term).sum(dim=(2, 3, 4))
    m00 = w_sum.squeeze(-1).squeeze(-1).squeeze(-1)  # (B, 1)
    if gamma == 1.0:
        return raw / m00
    vol = float(w.shape[2] * w.shape[3] * w.shape[4])
    f = m00 / vol  # class fraction in (0, 1]
    return raw / (f ** gamma * vol)


def _barrier_pair(barrier, pred_m, gt_m, tol):
    """Apply log-barrier to enforce gt_m - tol <= pred_m <= gt_m + tol."""
    z_up = pred_m - (gt_m + tol)
    z_lo = (gt_m - tol) - pred_m
    return barrier(z_up.reshape(-1)) + barrier(z_lo.reshape(-1))


def _setup(logits, target, n_classes, moment_class, ignore_index, eps, centroid_norm):
    """Shared setup: probs, masks, coord grids, sums, centroids, has_class flag."""
    device = logits.device
    probs = F.softmax(logits.float(), dim=1)
    valid = (target != ignore_index).float().unsqueeze(1)

    oh, _ = _one_hot(target, n_classes, ignore_index)
    oh = oh.to(device)

    pred_w  = probs[:, moment_class:moment_class+1] * valid
    gt_mask = oh[:, moment_class:moment_class+1] * valid

    B, _, D, H, W = pred_w.shape
    coords = _coord_grids(B, D, H, W, device, centroid_norm)

    pred_sum = pred_w.sum(dim=(2, 3, 4), keepdim=True) + eps
    gt_sum   = gt_mask.sum(dim=(2, 3, 4), keepdim=True) + eps

    # centroids: (B, 3, 1, 1, 1)
    pred_c = (pred_w * coords).sum(dim=(2, 3, 4), keepdim=True) / pred_sum
    gt_c   = (gt_mask * coords).sum(dim=(2, 3, 4), keepdim=True) / gt_sum

    has_class = (gt_mask.sum(dim=(2, 3, 4)) > 0).squeeze(1)

    return pred_w, gt_mask, coords, pred_sum, gt_sum, pred_c, gt_c, has_class


# ------------------------------------------------- 2nd order central moments -

def compute_2nd_moment_barrier(
    logits,
    target,
    n_classes,
    moment_class,
    barrier,
    moment_tolerance=0.02,
    offdiag_tolerance=None,
    centroid_norm=True,
    ignore_index=-1,
    eps=1e-6,
    return_stats=False,
    gamma=1.0,
    sqrt_diagonal=True,
    verbose=False,
):
    """
    Log-barrier on the 6 normalized 2nd-order central moments.

    Diagonal (variance per axis) — mu_200, mu_020, mu_002:
      With sqrt_diagonal=True (default): square-rooted after normalization,
      giving per-axis standard deviations.  This amplifies gradient signal for
      thin structures (d(sqrt(eta))/d(eta) = 1/(2*sqrt(eta)) → large when small),
      matching the behaviour of avgdist_axis. Uses `moment_tolerance`.

    Off-diagonal (cross-covariances, can be negative) — mu_110, mu_101, mu_011:
      Always kept raw; no sqrt since covariances can be negative. Uses
      `offdiag_tolerance` (defaults to `moment_tolerance` if not given) — these
      terms are typically an order of magnitude smaller in natural scale, so a
      shared tolerance usually leaves them inert (always inside the band).

    gamma: normalization order (1.0 = weighted average, 5/3 = fractional mass).

    Returns (diag_loss, offdiag_loss) normally, or (diag_loss, offdiag_loss,
    mean_err) if return_stats=True — so diagonal and off-diagonal terms can be
    weighted by separate lambdas at the call site.
    """
    if offdiag_tolerance is None:
        offdiag_tolerance = moment_tolerance

    logits_ref = logits
    pred_w, gt_mask, coords, pred_sum, gt_sum, pred_c, gt_c, has_class = _setup(
        logits, target, n_classes, moment_class, ignore_index, eps, centroid_norm
    )

    if not has_class.any():
        z = logits_ref.new_tensor(0.0)
        if return_stats:
            return z, z, z
        return z, z

    dz_p = coords[:, 0:1] - pred_c[:, 0:1]
    dy_p = coords[:, 1:2] - pred_c[:, 1:2]
    dx_p = coords[:, 2:3] - pred_c[:, 2:3]
    dz_g = coords[:, 0:1] - gt_c[:, 0:1]
    dy_g = coords[:, 1:2] - gt_c[:, 1:2]
    dx_g = coords[:, 2:3] - gt_c[:, 2:3]

    # (is_diagonal, pred_term, gt_term)
    terms = [
        (True,  dz_p ** 2,   dz_g ** 2),    # mu_200  -> sqrt gives sigma_z
        (True,  dy_p ** 2,   dy_g ** 2),    # mu_020  -> sqrt gives sigma_y
        (True,  dx_p ** 2,   dx_g ** 2),    # mu_002  -> sqrt gives sigma_x
        (False, dz_p * dy_p, dz_g * dy_g),  # mu_110
        (False, dz_p * dx_p, dz_g * dx_g),  # mu_101
        (False, dy_p * dx_p, dy_g * dx_g),  # mu_011
    ]

    diag_loss = logits_ref.new_tensor(0.0)
    offdiag_loss = logits_ref.new_tensor(0.0)
    errors = [] if (return_stats or verbose) else None
    diag_errors = [] if verbose else None
    offdiag_errors = [] if verbose else None
    for is_diag, pt, gt in terms:
        pred_m = _weighted_moment(pred_w, pt, pred_sum, gamma=gamma).squeeze(1)[has_class]
        gt_m   = _weighted_moment(gt_mask, gt, gt_sum,  gamma=gamma).squeeze(1)[has_class]
        if sqrt_diagonal and is_diag:
            pred_m = torch.sqrt(pred_m.clamp(min=0) + eps)
            gt_m   = torch.sqrt(gt_m.clamp(min=0)   + eps)
        if return_stats or verbose:
            err = (pred_m - gt_m).abs().mean()
            errors.append(err)
            if verbose:
                (diag_errors if is_diag else offdiag_errors).append(float(err.detach()))
        tol = moment_tolerance if is_diag else offdiag_tolerance
        term_loss = _barrier_pair(barrier, pred_m, gt_m, tol)
        if is_diag:
            diag_loss = diag_loss + term_loss
        else:
            offdiag_loss = offdiag_loss + term_loss
        del pt, gt, pred_m, gt_m

    if verbose:
        d = sum(diag_errors) / len(diag_errors)
        o = sum(offdiag_errors) / len(offdiag_errors)
        print(f"        [moment2 class={moment_class}] diag(sigma)_err={d:.3e}  offdiag(cov)_err={o:.3e}")

    if return_stats:
        mean_err = torch.stack(errors).mean() if errors else logits_ref.new_tensor(0.0)
        return diag_loss, offdiag_loss, mean_err
    return diag_loss, offdiag_loss


# --------------------------------------- 2nd order rotation-invariant moments -

def compute_moment_invariants_barrier(
    logits,
    target,
    n_classes,
    moment_class,
    barrier,
    lambda_J1=1.0,
    lambda_J2=1.0,
    lambda_J3=1.0,
    tol_J1=0.01,
    tol_J2=1e-3,
    tol_J3=1e-4,
    centroid_norm=True,
    ignore_index=-1,
    eps=1e-6,
    return_stats=False,
    gamma=1.0,
):
    """
    Log-barrier on the three rotation-invariant scalars derived from the
    2nd-order covariance matrix (Sadjadi & Hall, 1980):

        J1 = trace(Sigma)                     -- total spread         O(sigma^2)
        J2 = sum of 2x2 principal minors      -- pairwise spread      O(sigma^4)
        J3 = det(Sigma)                        -- ellipsoid volume     O(sigma^6)

    Invariant to translation, rigid rotation, and (approximately) scale.
    Per-invariant lambdas and tolerances account for the large scale differences
    between J1/J2/J3 (sigma^2 vs sigma^4 vs sigma^6). Set lambda_J* = 0 to skip.

    gamma: normalization order (1.0 = weighted average, 5/3 = fractional mass).
    return_stats: if True, returns (loss, stats) where stats is a tensor [err_J1, err_J2, err_J3].
    """
    logits_ref = logits
    pred_w, gt_mask, coords, pred_sum, gt_sum, pred_c, gt_c, has_class = _setup(
        logits, target, n_classes, moment_class, ignore_index, eps, centroid_norm
    )

    if not has_class.any():
        if return_stats:
            return logits_ref.new_tensor(0.0), logits_ref.new_zeros(3)
        return logits_ref.new_tensor(0.0)

    def _cov(w, c, w_sum):
        dz = coords[:, 0:1] - c[:, 0:1]
        dy = coords[:, 1:2] - c[:, 1:2]
        dx = coords[:, 2:3] - c[:, 2:3]
        c200 = _weighted_moment(w, dz * dz, w_sum, gamma=gamma)
        c020 = _weighted_moment(w, dy * dy, w_sum, gamma=gamma)
        c002 = _weighted_moment(w, dx * dx, w_sum, gamma=gamma)
        c110 = _weighted_moment(w, dz * dy, w_sum, gamma=gamma)
        c101 = _weighted_moment(w, dz * dx, w_sum, gamma=gamma)
        c011 = _weighted_moment(w, dy * dx, w_sum, gamma=gamma)
        return c200, c020, c002, c110, c101, c011

    def _invariants(c200, c020, c002, c110, c101, c011):
        J1 = c200 + c020 + c002
        J2 = (c200 * c020 + c020 * c002 + c200 * c002
              - c110 ** 2 - c011 ** 2 - c101 ** 2)
        J3 = (c200 * (c020 * c002 - c011 ** 2)
              - c110 * (c110 * c002 - c011 * c101)
              + c101 * (c110 * c011 - c020 * c101))
        return J1.squeeze(1), J2.squeeze(1), J3.squeeze(1)

    pJ1, pJ2, pJ3 = _invariants(*_cov(pred_w, pred_c, pred_sum))
    gJ1, gJ2, gJ3 = _invariants(*_cov(gt_mask, gt_c, gt_sum))

    loss = logits_ref.new_tensor(0.0)
    inv_terms = [
        (lambda_J1, tol_J1, pJ1, gJ1),
        (lambda_J2, tol_J2, pJ2, gJ2),
        (lambda_J3, tol_J3, pJ3, gJ3),
    ]
    errors = []
    for lam, tol, pj, gj in inv_terms:
        pj_valid, gj_valid = pj[has_class], gj[has_class]
        if lam > 0.0:
            loss = loss + lam * _barrier_pair(barrier, pj_valid, gj_valid, tol)
        if return_stats:
            errors.append((pj_valid - gj_valid).abs().mean())

    if return_stats:
        stats = torch.stack(errors) if errors else logits_ref.new_zeros(3)
        return loss, stats
    return loss


# ------------------------------------------------- 3rd order central moments -

def compute_3rd_moment_barrier(
    logits,
    target,
    n_classes,
    moment_class,
    barrier,
    moment_tolerance=0.01,
    offdiag_tolerance=None,
    centroid_norm=True,
    ignore_index=-1,
    eps=1e-6,
    return_stats=False,
    gamma=1.0,
    sqrt_diagonal=True,
    verbose=False,
):
    """
    Log-barrier on all 10 normalized 3rd-order central moments.

    Pure cubic (per-axis skewness) — mu_300, mu_030, mu_003:
      With sqrt_diagonal=True (default): signed sqrt applied after normalization.
      sign(m) * sqrt(|m|) preserves the skewness direction while amplifying
      gradient signal when skewness magnitude is small (thin/symmetric structures).
      Uses `moment_tolerance`.

    Mixed and cross terms — mu_210, mu_201, mu_120, mu_021, mu_102, mu_012, mu_111:
      Always kept raw (can be positive or negative). Uses `offdiag_tolerance`
      (defaults to `moment_tolerance` if not given) — typically much smaller in
      natural scale than the pure-cubic terms, so a shared tolerance usually
      leaves them inert (always inside the band).

    gamma: normalization order (1.0 = weighted average, 2.0 = fractional mass).

    Returns (diag_loss, offdiag_loss) normally, or (diag_loss, offdiag_loss,
    mean_err) if return_stats=True.
    """
    if offdiag_tolerance is None:
        offdiag_tolerance = moment_tolerance

    logits_ref = logits
    pred_w, gt_mask, coords, pred_sum, gt_sum, pred_c, gt_c, has_class = _setup(
        logits, target, n_classes, moment_class, ignore_index, eps, centroid_norm
    )

    if not has_class.any():
        z = logits_ref.new_tensor(0.0)
        if return_stats:
            return z, z, z
        return z, z

    dz_p = coords[:, 0:1] - pred_c[:, 0:1]
    dy_p = coords[:, 1:2] - pred_c[:, 1:2]
    dx_p = coords[:, 2:3] - pred_c[:, 2:3]
    dz_g = coords[:, 0:1] - gt_c[:, 0:1]
    dy_g = coords[:, 1:2] - gt_c[:, 1:2]
    dx_g = coords[:, 2:3] - gt_c[:, 2:3]

    dz_p2 = dz_p ** 2;  dy_p2 = dy_p ** 2;  dx_p2 = dx_p ** 2
    dz_g2 = dz_g ** 2;  dy_g2 = dy_g ** 2;  dx_g2 = dx_g ** 2

    # (is_pure_cubic, pred_term, gt_term)
    terms = [
        (True,  dz_p2 * dz_p,        dz_g2 * dz_g),        # mu_300  -> signed sqrt
        (True,  dy_p2 * dy_p,        dy_g2 * dy_g),        # mu_030
        (True,  dx_p2 * dx_p,        dx_g2 * dx_g),        # mu_003
        (False, dz_p2 * dy_p,        dz_g2 * dy_g),        # mu_210
        (False, dz_p2 * dx_p,        dz_g2 * dx_g),        # mu_201
        (False, dz_p  * dy_p2,       dz_g  * dy_g2),       # mu_120
        (False, dy_p2 * dx_p,        dy_g2 * dx_g),        # mu_021
        (False, dz_p  * dx_p2,       dz_g  * dx_g2),       # mu_102
        (False, dy_p  * dx_p2,       dy_g  * dx_g2),       # mu_012
        (False, dz_p  * dy_p * dx_p, dz_g  * dy_g * dx_g), # mu_111
    ]

    diag_loss = logits_ref.new_tensor(0.0)
    offdiag_loss = logits_ref.new_tensor(0.0)
    errors = [] if (return_stats or verbose) else None
    diag_errors = [] if verbose else None
    offdiag_errors = [] if verbose else None
    for is_pure_cubic, pt, gt in terms:
        pred_m = _weighted_moment(pred_w, pt, pred_sum, gamma=gamma).squeeze(1)[has_class]
        gt_m   = _weighted_moment(gt_mask, gt, gt_sum,  gamma=gamma).squeeze(1)[has_class]
        if sqrt_diagonal and is_pure_cubic:
            # signed sqrt: preserves skewness direction, amplifies gradient for small |m|
            pred_m = torch.sign(pred_m) * torch.sqrt(pred_m.abs() + eps)
            gt_m   = torch.sign(gt_m)   * torch.sqrt(gt_m.abs()   + eps)
        if return_stats or verbose:
            err = (pred_m - gt_m).abs().mean()
            errors.append(err)
            if verbose:
                (diag_errors if is_pure_cubic else offdiag_errors).append(float(err.detach()))
        tol = moment_tolerance if is_pure_cubic else offdiag_tolerance
        term_loss = _barrier_pair(barrier, pred_m, gt_m, tol)
        if is_pure_cubic:
            diag_loss = diag_loss + term_loss
        else:
            offdiag_loss = offdiag_loss + term_loss
        del pt, gt, pred_m, gt_m

    del dz_p2, dy_p2, dx_p2, dz_g2, dy_g2, dx_g2

    if verbose:
        d = sum(diag_errors) / len(diag_errors)
        o = sum(offdiag_errors) / len(offdiag_errors)
        print(f"        [moment3 class={moment_class}] diag(skew)_err={d:.3e}  offdiag(mixed)_err={o:.3e}")

    if return_stats:
        mean_err = torch.stack(errors).mean() if errors else logits_ref.new_tensor(0.0)
        return diag_loss, offdiag_loss, mean_err
    return diag_loss, offdiag_loss
