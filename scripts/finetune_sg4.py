"""
Stretch Goal 4: down-weight uncertain cross-modal anchor links during
fine-tuning, and validate the result as a well-characterized null.

Three checks, all measured against a seed-matched floor=1.0 control
(fine-tuning itself moves accuracy by ~-0.66%, so the raw un-fine-tuned
checkpoint is not a fair baseline for the treatment runs):

  1. Seed sweep: floor=0.2 vs. matched control, 5 seeds.
  2. Strength sweep: floors [0.0, 0.25, 0.5] vs. the seed=0 control (floor=1.0
     is the control itself, delta=0 by definition).
  3. SG3-signal robustness check: same floor=0.2 down-weighting, but using
     the SG3 integration-uncertainty signal (r=0.12, weaker correlation
     with error) instead of the SG1 epistemic signal (r=0.59), to confirm
     the regression is driven by the down-weighting mechanism itself and
     not by the choice of uncertainty signal.

Requires: model_v2 checkpoint (train_scmodal.py) and epistemic_uncertainty.npy
+ integration_uncertainty.npy (compute_uncertainty.py) already on disk.

Usage:
    python scripts/finetune_sg4.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# scMODAL is a git submodule (see README), imported via current-directory
# resolution -- not pip-installed. This must be on sys.path before any
# `import scmodal` call below, regardless of the caller's cwd.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scMODAL"))

import numpy as np

from src.finetune import run_sg4_finetune

CITE_DIR = "CITE-seq_PBMC_v2"
CKPT_PATH = os.path.join(CITE_DIR, "ckpt.pth")


def main(model_v2, gt_protein_v2):
    epistemic_unc_cite = np.load(os.path.join(CITE_DIR, "epistemic_uncertainty.npy"))
    integration_unc_cite = np.load(os.path.join(CITE_DIR, "integration_uncertainty.npy"))

    results = {}

    # ---- Step 1: seed-matched sweep (floor=0.2 vs. seed-matched floor=1.0 control) ----
    seeds = [0, 1, 2, 3, 4]
    seed_matched_results = []
    control_errs = {}

    for s in seeds:
        print(f"--- seed {s}: floor=1.0 (control) ---")
        _, err_control = run_sg4_finetune(model_v2, gt_protein_v2, seed=s, unc=epistemic_unc_cite, floor=1.0, ckpt_path=CKPT_PATH)
        if s == 0:
            control_errs[0] = err_control

        print(f"--- seed {s}: floor=0.2 (treatment, vs. seed-matched control) ---")
        r, _ = run_sg4_finetune(model_v2, gt_protein_v2, seed=s, unc=epistemic_unc_cite, floor=0.2, baseline_err=err_control, ckpt_path=CKPT_PATH)
        print(r)
        seed_matched_results.append(r)

    print("\nSeed-matched summary (floor=0.2 vs. seed-matched floor=1.0 control):")
    for name in ["low", "mid", "high"]:
        vals = [r[name] for r in seed_matched_results]
        print(f"{name:5s}: mean={np.mean(vals):+.2f}%  std={np.std(vals):.2f}  values={[round(v, 2) for v in vals]}")
    results["seed_sweep"] = seed_matched_results

    # ---- Step 2: strength sweep vs. matched floor=1.0/seed=0 control ----
    floors_sweep = [0.0, 0.25, 0.5]
    strength_matched = []
    for f in floors_sweep:
        print(f"--- seed 0: floor={f} (vs. floor=1.0/seed=0 control) ---")
        r, _ = run_sg4_finetune(model_v2, gt_protein_v2, seed=0, unc=epistemic_unc_cite, floor=f, baseline_err=control_errs[0], ckpt_path=CKPT_PATH)
        print(r)
        strength_matched.append(r)
    results["strength_sweep"] = strength_matched

    # ---- Step 3: SG3-signal robustness check vs. matched control ----
    print("--- seed 0: SG3 signal, floor=0.2 (vs. floor=1.0/seed=0 control) ---")
    r_sg3, _ = run_sg4_finetune(model_v2, gt_protein_v2, seed=0, unc=integration_unc_cite, floor=0.2, baseline_err=control_errs[0], ckpt_path=CKPT_PATH)
    print(r_sg3)
    results["sg3_signal_check"] = r_sg3

    with open(os.path.join(CITE_DIR, "sg4_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {os.path.join(CITE_DIR, 'sg4_results.json')}")

    return results


if __name__ == "__main__":
    raise SystemExit(
        "This script's main() needs a live model_v2 + gt_protein_v2 passed in from "
        "train_scmodal.py's output -- wire them up before running standalone."
    )
