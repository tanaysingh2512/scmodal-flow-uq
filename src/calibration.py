"""
Calibration curves and the post-hoc scalar correction.

The UA-Flow's learned variance head is informative but not well-calibrated
out of the box. A single scalar multiplier on the predicted std, fit on a
calibration split and evaluated on a held-out test split, closes most of
the gap -- on both datasets, despite the fitted scale factors themselves
differing by >4x (CITE-seq best_scale=0.520, TEA-seq best_scale_tea=0.120).
That the *method* generalizes across datasets even though the *scale
factor* does not is one of the project's key findings.
"""

import numpy as np
from scipy.stats import norm


def coverage_curve(mean, std, ground_truth, nominal_levels, scale: float = 1.0) -> np.ndarray:
    """Empirical coverage of analytic Gaussian intervals (mean +/- z*scale*std)
    at each nominal level."""
    coverages = []
    for level in nominal_levels:
        z = norm.ppf((1 + level) / 2)
        lower = mean - z * scale * std
        upper = mean + z * scale * std
        inside = (ground_truth >= lower) & (ground_truth <= upper)
        coverages.append(inside.mean())
    return np.array(coverages)


def fit_scalar_correction(
    mean, std, ground_truth, nominal_levels, seed: int = 42,
    scale_grid: np.ndarray | None = None,
):
    """Fits a single scalar multiplier on predicted std via a calibration/
    test split (50/50), minimizing mean absolute gap between nominal and
    actual coverage. Returns (best_scale, test_coverage) where
    test_coverage is evaluated on the held-out half only.

    NOTE: the original notebook used slightly different scale grids for the
    two datasets -- CITE-seq: np.linspace(0.1, 2.0, 191) (cell 154), TEA-seq:
    np.linspace(0.01, 2.0, 200) (cell 157). This function's default matches
    TEA-seq's; scripts/train_ua_flow.py passes CITE-seq's grid explicitly
    for that call. Don't assume the default alone is correct for both --
    check the caller (found during the reproducibility audit; see decisions.md).
    """
    if scale_grid is None:
        scale_grid = np.linspace(0.01, 2.0, 200)

    rng = np.random.RandomState(seed)
    n_cells = mean.shape[0]
    perm = rng.permutation(n_cells)
    calib_idx, test_idx = perm[: n_cells // 2], perm[n_cells // 2 :]

    best_scale, best_gap = 1.0, np.inf
    for s in scale_grid:
        cov = coverage_curve(mean[calib_idx], std[calib_idx], ground_truth[calib_idx], nominal_levels, scale=s)
        gap = np.mean(np.abs(np.array(nominal_levels) - cov))
        if gap < best_gap:
            best_gap, best_scale = gap, s

    test_coverage = coverage_curve(mean[test_idx], std[test_idx], ground_truth[test_idx], nominal_levels, scale=best_scale)
    return best_scale, test_coverage
