"""
Stretch Goal 2: train the heteroscedastic UA-Flow for CITE-seq and TEA-seq,
compare its learned variance against sample-spread, and fit the post-hoc
scalar calibration correction for each.

Requires: model_v2 (CITE-seq) and TEA-seq latents/ground truth already on disk.

Usage:
    python scripts/train_ua_flow.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from src.velocitynet import train_ua_flow, sample_ua_flow, learned_variance
from src.calibration import coverage_curve, fit_scalar_correction

NOMINAL_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CITE_DIR = "CITE-seq_PBMC_v2"
TEA_ROOT = "TEA-seq_PBMC"
TEA_DIR = os.path.join(TEA_ROOT, "TEA-seq_PBMC")


def run_citeseq(model_v2, z1_v2, z2_v2, feat_dim_v2, gt_cite_v2):
    y_all_cite = torch.from_numpy(model_v2.emb_B).float()
    z_all_cite = z2_v2.detach()

    ua_vnet_cite = train_ua_flow(z_all_cite, y_all_cite, feat_dim=feat_dim_v2, latent_dim=model_v2.n_latent)
    torch.save(ua_vnet_cite.state_dict(), os.path.join(CITE_DIR, "ua_flow_vnet.pth"))

    imputed_samples_ua = sample_ua_flow(ua_vnet_cite, z1_v2, feat_dim_v2, steps=100, K=50)
    imputed_mean_ua = imputed_samples_ua.mean(dim=1).cpu().numpy()
    imputed_std_ss = imputed_samples_ua.std(dim=1).cpu().numpy()
    imputed_std_learned = learned_variance(ua_vnet_cite, y_all_cite, z_all_cite).cpu().numpy()

    cov_ss = coverage_curve(imputed_mean_ua, imputed_std_ss, gt_cite_v2, NOMINAL_LEVELS)
    cov_learned = coverage_curve(imputed_mean_ua, imputed_std_learned, gt_cite_v2, NOMINAL_LEVELS)
    for nom, ss, learned in zip(NOMINAL_LEVELS, cov_ss, cov_learned):
        print(f"CITE-seq nominal={nom:.1f}  sample-spread={ss:.3f}  learned-variance={learned:.3f}")

    best_scale, test_coverage = fit_scalar_correction(
        imputed_mean_ua, imputed_std_learned, gt_cite_v2, NOMINAL_LEVELS,
        scale_grid=np.linspace(0.1, 2.0, 191),  # matches notebook cell 154 exactly -- CITE-seq's grid differs from TEA-seq's (see decisions.md)
    )
    print(f"CITE-seq best_scale = {best_scale:.3f}  (project record: 0.520)")

    return {
        "ua_vnet": ua_vnet_cite, "imputed_mean_ua": imputed_mean_ua,
        "imputed_std_learned": imputed_std_learned, "best_scale": best_scale,
        "test_coverage_recalibrated": test_coverage,
    }


def run_teaseq(z2_tea, gt_protein_tea):
    y_all_tea = torch.from_numpy(gt_protein_tea).float()
    z_all_tea = torch.from_numpy(z2_tea).float()

    ua_vnet_tea = train_ua_flow(z_all_tea, y_all_tea, feat_dim=y_all_tea.shape[1], latent_dim=z2_tea.shape[1])
    torch.save(ua_vnet_tea.state_dict(), os.path.join(TEA_ROOT, "ua_flow_vnet.pth"))

    imputed_samples_ua_tea = sample_ua_flow(ua_vnet_tea, z_all_tea, y_all_tea.shape[1], steps=100, K=50)
    imputed_mean_ua_tea = imputed_samples_ua_tea.mean(dim=1).cpu().numpy()
    imputed_std_learned_tea = learned_variance(ua_vnet_tea, y_all_tea, z_all_tea).cpu().numpy()

    best_scale_tea, test_coverage_tea = fit_scalar_correction(
        imputed_mean_ua_tea, imputed_std_learned_tea, gt_protein_tea, NOMINAL_LEVELS,
        scale_grid=np.linspace(0.01, 2.0, 200),  # matches notebook cell 157 exactly (also calibration.py's own default)
    )
    print(f"TEA-seq best_scale_tea = {best_scale_tea:.3f}  (project record: 0.120)")

    return {
        "ua_vnet": ua_vnet_tea, "imputed_mean_ua_tea": imputed_mean_ua_tea,
        "imputed_std_learned_tea": imputed_std_learned_tea, "best_scale_tea": best_scale_tea,
        "test_coverage_recalibrated_tea": test_coverage_tea,
    }


if __name__ == "__main__":
    raise SystemExit(
        "This script's run_citeseq()/run_teaseq() need live model/latent objects passed in "
        "from your driver -- wire them up (see train_scmodal.py + compute_uncertainty.py outputs) "
        "before running standalone."
    )
