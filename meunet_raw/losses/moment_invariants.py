"""
Moment-based shape descriptor barriers for 3-D soft segmentation.

Moment hierarchy (building on existing centroid / avgdist barriers):
  0th order  ->  total mass (volume)        -- already in compute_volume_barrier
  1st order  ->  centroid                   -- already in compute_centroid_barrier
  2nd order  ->  covariance matrix          -- compute_2nd_moment_barrier
  rotation-invariant 2nd order              -- compute_moment_invariants_barrier
  3rd order  ->  skewness / asymmetry       -- compute_3rd_moment_barrier

Normalization follows the standard normalized central moment definition]

    eta_pqr = mu_pqr / M000^gamma,   gamma = (p+q+r)/3 + 1

  2nd order (n=2): gamma = 5/3
  3rd order (n=3): gamma = 2

Centroid computation uses x_c = M_100/M_000 (unchanged).
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
    """eta_pqr = sum(w * term, spatial) / M000^gamma.  Returns (B, 1)."""
    raw = (w * term).sum(dim=(2, 3, 4))
    m00 = w_sum.squeeze(-1).squeeze(-1).squeeze(-1)
    return raw / (m00 ** gamma)


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
    centroid_norm=True,
    ignore_index=-1,
    eps=1e-6,
    return_stats=False,
):
    """
    Log-barrier on the 6 normalized 2nd-order central moments:

        eta_200, eta_020, eta_002   spread along z, y, x
        eta_110, eta_101, eta_011   cross-covariances

    Uses gamma = 5/3  (n=2, 3-D extension of eq. 11-37/11-38).
    These are the independent components of the scale-normalized covariance matrix.
    """
    logits_ref = logits  # keep for new_tensor
    pred_w, gt_mask, coords, pred_sum, gt_sum, pred_c, gt_c, has_class = _setup(
        logits, target, n_classes, moment_class, ignore_index, eps, centroid_norm
    )

    if not has_class.any():
        return logits_ref.new_tensor(0.0)

    dz_p = coords[:, 0:1] - pred_c[:, 0:1]
    dy_p = coords[:, 1:2] - pred_c[:, 1:2]
    dx_p = coords[:, 2:3] - pred_c[:, 2:3]
    dz_g = coords[:, 0:1] - gt_c[:, 0:1]
    dy_g = coords[:, 1:2] - gt_c[:, 1:2]
    dx_g = coords[:, 2:3] - gt_c[:, 2:3]

    terms = [
        (dz_p ** 2,   dz_g ** 2),    # mu_200
        (dy_p ** 2,   dy_g ** 2),    # mu_020
        (dx_p ** 2,   dx_g ** 2),    # mu_002
        (dz_p * dy_p, dz_g * dy_g),  # mu_110
        (dz_p * dx_p, dz_g * dx_g),  # mu_101
        (dy_p * dx_p, dy_g * dx_g),  # mu_011
    ]

    loss = logits_ref.new_tensor(0.0)
    errors = [] if return_stats else None
    for pt, gt in terms:
        pred_m = _weighted_moment(pred_w, pt, pred_sum, gamma=5/3).squeeze(1)[has_class]
        gt_m   = _weighted_moment(gt_mask, gt, gt_sum,  gamma=5/3).squeeze(1)[has_class]
        if return_stats:
            errors.append((pred_m - gt_m).abs().mean())
        loss = loss + _barrier_pair(barrier, pred_m, gt_m, moment_tolerance)
        del pt, gt, pred_m, gt_m

    if return_stats:
        mean_err = torch.stack(errors).mean() if errors else logits_ref.new_tensor(0.0)
        return loss, mean_err
    return loss


# --------------------------------------- 2nd order rotation-invariant moments -

def compute_moment_invariants_barrier(
    logits,
    target,
    n_classes,
    moment_class,
    barrier,
    inv_tolerance=0.01,
    centroid_norm=True,
    ignore_index=-1,
    eps=1e-6,
):
    """
    Log-barrier on the three rotation-invariant scalars derived from the
    2nd-order covariance matrix (Sadjadi & Hall, 1980):

        J1 = trace(Sigma)                     -- total spread
        J2 = sum of 2x2 principal minors      -- pairwise spread product
        J3 = det(Sigma)                        -- volume of the ellipsoid

    Invariant to translation, rigid rotation, and (approximately) scale
    when centroid_norm=True.
    """
    logits_ref = logits
    pred_w, gt_mask, coords, pred_sum, gt_sum, pred_c, gt_c, has_class = _setup(
        logits, target, n_classes, moment_class, ignore_index, eps, centroid_norm
    )

    if not has_class.any():
        return logits_ref.new_tensor(0.0)

    def _cov(w, c, w_sum):
        dz = coords[:, 0:1] - c[:, 0:1]
        dy = coords[:, 1:2] - c[:, 1:2]
        dx = coords[:, 2:3] - c[:, 2:3]
        c200 = _weighted_moment(w, dz * dz, w_sum, gamma=5/3)
        c020 = _weighted_moment(w, dy * dy, w_sum, gamma=5/3)
        c002 = _weighted_moment(w, dx * dx, w_sum, gamma=5/3)
        c110 = _weighted_moment(w, dz * dy, w_sum, gamma=5/3)
        c101 = _weighted_moment(w, dz * dx, w_sum, gamma=5/3)
        c011 = _weighted_moment(w, dy * dx, w_sum, gamma=5/3)
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
    for pj, gj in [(pJ1, gJ1), (pJ2, gJ2), (pJ3, gJ3)]:
        loss = loss + _barrier_pair(barrier, pj[has_class], gj[has_class], inv_tolerance)

    return loss


# ------------------------------------------------- 3rd order central moments -

def compute_3rd_moment_barrier(
    logits,
    target,
    n_classes,
    moment_class,
    barrier,
    moment_tolerance=0.01,
    centroid_norm=True,
    ignore_index=-1,
    eps=1e-6,
    return_stats=False,
):
    """
    Log-barrier on all 10 normalized 3rd-order central moments:

        eta_300, eta_030, eta_003               per-axis skewness
        eta_210, eta_201, eta_120, eta_021,
        eta_102, eta_012                        mixed cubic
        eta_111                                triple cross-term

    Uses gamma = 2  (n=3, 3-D extension of eq. 11-37/11-38).
    Captures asymmetry and handedness beyond the 2nd order ellipsoid.
    """
    logits_ref = logits
    pred_w, gt_mask, coords, pred_sum, gt_sum, pred_c, gt_c, has_class = _setup(
        logits, target, n_classes, moment_class, ignore_index, eps, centroid_norm
    )

    if not has_class.any():
        return logits_ref.new_tensor(0.0)

    dz_p = coords[:, 0:1] - pred_c[:, 0:1]
    dy_p = coords[:, 1:2] - pred_c[:, 1:2]
    dx_p = coords[:, 2:3] - pred_c[:, 2:3]
    dz_g = coords[:, 0:1] - gt_c[:, 0:1]
    dy_g = coords[:, 1:2] - gt_c[:, 1:2]
    dx_g = coords[:, 2:3] - gt_c[:, 2:3]

    # Reuse squared terms to halve the number of elementwise muls
    dz_p2 = dz_p ** 2;  dy_p2 = dy_p ** 2;  dx_p2 = dx_p ** 2
    dz_g2 = dz_g ** 2;  dy_g2 = dy_g ** 2;  dx_g2 = dx_g ** 2

    terms = [
        (dz_p2 * dz_p,        dz_g2 * dz_g),        # mu_300
        (dy_p2 * dy_p,        dy_g2 * dy_g),        # mu_030
        (dx_p2 * dx_p,        dx_g2 * dx_g),        # mu_003
        (dz_p2 * dy_p,        dz_g2 * dy_g),        # mu_210
        (dz_p2 * dx_p,        dz_g2 * dx_g),        # mu_201
        (dz_p  * dy_p2,       dz_g  * dy_g2),       # mu_120
        (dy_p2 * dx_p,        dy_g2 * dx_g),        # mu_021
        (dz_p  * dx_p2,       dz_g  * dx_g2),       # mu_102
        (dy_p  * dx_p2,       dy_g  * dx_g2),       # mu_012
        (dz_p  * dy_p * dx_p, dz_g  * dy_g * dx_g), # mu_111
    ]

    loss = logits_ref.new_tensor(0.0)
    errors = [] if return_stats else None
    for pt, gt in terms:
        pred_m = _weighted_moment(pred_w, pt, pred_sum, gamma=2.0).squeeze(1)[has_class]
        gt_m   = _weighted_moment(gt_mask, gt, gt_sum,  gamma=2.0).squeeze(1)[has_class]
        if return_stats:
            errors.append((pred_m - gt_m).abs().mean())
        loss = loss + _barrier_pair(barrier, pred_m, gt_m, moment_tolerance)
        del pt, gt, pred_m, gt_m

    del dz_p2, dy_p2, dx_p2, dz_g2, dy_g2, dx_g2
    if return_stats:
        mean_err = torch.stack(errors).mean() if errors else logits_ref.new_tensor(0.0)
        return loss, mean_err
    return loss
