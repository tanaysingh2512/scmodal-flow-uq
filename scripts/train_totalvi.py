"""
totalVI baseline comparison, end to end. Must be run in the `totalvi` conda
environment (environment-totalvi.yml), never `scmodal`.

Reuses src/data_prep.py's CITE-seq raw loader (same underlying data, no
correspondence table needed -- totalVI trains jointly on all RNA + all
protein, no shared/unshared split).

Usage:
    conda activate totalvi
    python scripts/train_totalvi.py --resume     # skip training, reload saved model
    python scripts/train_totalvi.py               # ~35-90 min for 100 epochs on CPU
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.data_prep import load_citeseq_raw
from src.totalvi_baseline import (
    build_totalvi_anndata, train_totalvi, load_totalvi,
    posterior_predictive_samples, accuracy_count_scale, calibration_curve_totalvi,
)

NOMINAL_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/citeseq_pbmc")
    parser.add_argument("--model-dir", default="./totalvi_model")
    parser.add_argument("--out-dir", default="./totalvi_results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=100)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    adata_RNA, adata_ADT = load_citeseq_raw(args.data_dir)
    adata_totalvi = build_totalvi_anndata(adata_RNA, adata_ADT)

    if args.resume:
        vae = load_totalvi(adata_totalvi, args.model_dir)
    else:
        vae = train_totalvi(adata_totalvi, args.model_dir, max_epochs=args.max_epochs)

    gt_protein_raw = adata_ADT.to_df().values
    posterior_samples_full = posterior_predictive_samples(vae, adata_totalvi)
    np.save(os.path.join(args.out_dir, "totalvi_posterior_samples.npy"), posterior_samples_full)

    mean_counts, std_counts, corr_counts = accuracy_count_scale(posterior_samples_full, gt_protein_raw)
    np.save(os.path.join(args.out_dir, "totalvi_mean_counts.npy"), mean_counts)
    np.save(os.path.join(args.out_dir, "totalvi_std_counts.npy"), std_counts)
    np.save(os.path.join(args.out_dir, "totalvi_corr_counts.npy"), corr_counts)

    print(
        "\nCAVEAT (unresolved -- see src/totalvi_baseline.py docstring): this correlation is "
        "raw-count-scale over", gt_protein_raw.shape[1], "features. scMODAL/flow correlations are "
        "log-normalized-scaled-space over a different feature count. Not directly comparable without "
        "re-expressing totalVI predictions in scMODAL's feature space first."
    )

    actual_coverages = calibration_curve_totalvi(posterior_samples_full, gt_protein_raw, NOMINAL_LEVELS)
    np.save(os.path.join(args.out_dir, "totalvi_nominal_levels.npy"), np.array(NOMINAL_LEVELS))
    np.save(os.path.join(args.out_dir, "totalvi_actual_coverages.npy"), actual_coverages)


if __name__ == "__main__":
    main()
