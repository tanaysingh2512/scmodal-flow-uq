"""
Regenerates every figure referenced in the project write-up.

Two of these (SG4 seed-sweep bar chart, SG4 strength-sweep line chart) were
previously plotted from the OLD, uncorrected run_sg4_finetune numbers and
never regenerated after the switch to the baseline-corrected run_sg4_finetune_v2.
The corrected values are hardcoded below from the actual printed output of
the corrected run (see decisions.md, entry on baseline correction) -- replace
with a live results dict from finetune_sg4.py's output once you rerun it end
to end, rather than trusting these constants indefinitely.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

CITE_DIR = "CITE-seq_PBMC_v2"
os.makedirs(os.path.join("results", "figures"), exist_ok=True)
FIG_DIR = os.path.join("results", "figures")


def plot_sg4_seed_sweep():
    """Corrected (baseline-matched) 5-seed sweep, floor=0.2 vs. seed-matched
    floor=1.0 control. Values from the actual corrected run's output."""
    seed_results = {
        "low": [-0.37, -0.33, -0.35, -0.04, 0.10],
        "mid": [0.02, -0.11, -0.24, 0.02, 0.09],
        "high": [3.25, 2.17, 2.38, 2.69, 2.93],
    }
    tertiles = ["low", "mid", "high"]
    means = [np.mean(seed_results[t]) for t in tertiles]
    stds = [np.std(seed_results[t]) for t in tertiles]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(tertiles, means, yerr=stds, color=["#4C72B0", "#55A868", "#C44E52"], capsize=5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("% change in per-cell error\n(vs. seed-matched floor=1.0 control)")
    ax.set_title("SG4: down-weighting effect by uncertainty tertile\n(5-seed sweep, baseline-corrected)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sg4_seed_sweep_corrected.png"), dpi=150)
    plt.close(fig)


def plot_sg4_strength_sweep():
    """Corrected 4-point strength sweep (floor=1.0 is the control, delta=0)."""
    floors = [0.0, 0.25, 0.5, 1.0]
    high_tertile_change = [4.22, 2.77, 1.88, 0.0]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(floors, high_tertile_change, marker="o", linewidth=2, color="#C44E52", markersize=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="No effect")
    ax.set_xlabel("Down-weighting floor\n(1.0 = no down-weighting, control)")
    ax.set_ylabel("% change in per-cell error\n(high-uncertainty tertile)")
    ax.set_title("SG4: dose-response, baseline-corrected")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sg4_strength_sweep_corrected.png"), dpi=150)
    plt.close(fig)


def plot_nn_distance_scatter_both(epistemic_unc_cite, per_cell_error_cite_v2, epistemic_unc_tea, per_cell_error_tea):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    corr_cite = np.corrcoef(epistemic_unc_cite, per_cell_error_cite_v2)[0, 1]
    corr_tea = np.corrcoef(epistemic_unc_tea, per_cell_error_tea)[0, 1]
    axes[0].scatter(epistemic_unc_cite, per_cell_error_cite_v2, alpha=0.3, s=5)
    axes[0].set_xlabel("Epistemic uncertainty (NN-distance)")
    axes[0].set_ylabel("Per-cell error")
    axes[0].set_title(f"CITE-seq: r={corr_cite:.3f}")
    axes[1].scatter(epistemic_unc_tea, per_cell_error_tea, alpha=0.3, s=5)
    axes[1].set_xlabel("Epistemic uncertainty (NN-distance)")
    axes[1].set_ylabel("Per-cell error")
    axes[1].set_title(f"TEA-seq: r={corr_tea:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "nn_distance_scatter_both.png"), dpi=150)
    plt.close(fig)


def plot_ua_flow_recalibration_both(nominal_levels, actual_coverage_learned, test_coverage_recalibrated, best_scale,
                                     actual_coverage_learned_tea, test_coverage_recalibrated_tea, best_scale_tea):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(nominal_levels, actual_coverage_learned, "o-", color="red", label="Before calibration")
    axes[0].plot(nominal_levels, test_coverage_recalibrated, "o-", color="green", label=f"After calibration (x{best_scale:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")
    axes[0].set_xlabel("Nominal coverage"); axes[0].set_ylabel("Actual coverage"); axes[0].set_title("CITE-seq")
    axes[0].legend()

    axes[1].plot(nominal_levels, actual_coverage_learned_tea, "o-", color="red", label="Before calibration")
    axes[1].plot(nominal_levels, test_coverage_recalibrated_tea, "o-", color="green", label=f"After calibration (x{best_scale_tea:.3f})")
    axes[1].plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")
    axes[1].set_xlabel("Nominal coverage"); axes[1].set_ylabel("Actual coverage"); axes[1].set_title("TEA-seq")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ua_flow_recalibration_both.png"), dpi=150)
    plt.close(fig)


def plot_integration_uncertainty_scatter(integration_unc_cite, per_cell_error_cite_v2, integration_unc_tea, per_cell_error_tea):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    corr_cite = np.corrcoef(integration_unc_cite, per_cell_error_cite_v2)[0, 1]
    corr_tea = np.corrcoef(integration_unc_tea, per_cell_error_tea)[0, 1]
    axes[0].scatter(integration_unc_cite, per_cell_error_cite_v2, alpha=0.3, s=5)
    axes[0].set_xlabel("Integration uncertainty (weight-noise)"); axes[0].set_ylabel("Per-cell error")
    axes[0].set_title(f"CITE-seq: r={corr_cite:.3f}")
    axes[1].scatter(integration_unc_tea, per_cell_error_tea, alpha=0.3, s=5)
    axes[1].set_xlabel("Integration uncertainty (weight-noise)"); axes[1].set_ylabel("Per-cell error")
    axes[1].set_title(f"TEA-seq: r={corr_tea:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "integration_uncertainty_scatter_both.png"), dpi=150)
    plt.close(fig)


def plot_flow_vs_totalvi_calibration(nominal_levels_totalvi, actual_coverages_totalvi):
    """Flow's own calibration curve is hardcoded here from the recorded Phase 3
    result (0.475 corr run) since it comes from a different conda environment
    (scmodal) than totalVI (totalvi) -- kernels don't share variables across
    environments, so this was always a manual cross-environment stitch, not
    a live comparison. Replace with a live array if you have both saved to disk."""
    nominal_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    actual_coverages_flow = [0.069, 0.138, 0.208, 0.279, 0.353, 0.429, 0.511, 0.599, 0.702]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(nominal_levels, actual_coverages_flow, "o-", color="tab:blue", label="scMODAL + Flow")
    ax.plot(nominal_levels_totalvi, actual_coverages_totalvi, "s-", color="tab:orange", label="totalVI")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Actual coverage")
    ax.set_title("Calibration comparison: Flow (overconfident) vs.\ntotalVI (underconfident) -- opposite failure modes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "calibration_comparison_flow_totalvi.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # The two SG4 plots are self-contained (hardcoded corrected constants) and safe to run standalone:
    plot_sg4_seed_sweep()
    plot_sg4_strength_sweep()
    print(f"SG4 figures written to {FIG_DIR}/")
    print("Other figures (nn_distance_scatter_both, ua_flow_recalibration_both, "
          "integration_uncertainty_scatter_both) need live arrays passed in from "
          "compute_uncertainty.py / train_ua_flow.py -- call those functions directly.")
