"""
Train (or fast-resume) the scMODAL CITE-seq v2 backbone.

This now just wraps scripts/load_raw_data.py's CITE-seq path (which handles
raw loading + the reproducibility fix + training/resume in one call) and
adds the sanity-check comparison against the original ~0.485 baseline
correlation.

Usage:
    python scripts/train_scmodal.py --resume
    python scripts/train_scmodal.py                # full ~6hr retrain
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# scMODAL is a git submodule (see README), imported via current-directory
# resolution -- not pip-installed. This must be on sys.path before any
# `import scmodal` call below, regardless of the caller's cwd.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scMODAL"))

from scripts.load_raw_data import run_citeseq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="CITE-seq_PBMC_v2")
    parser.add_argument("--data-dir", default="./data/citeseq_pbmc")
    parser.add_argument("--resume", action="store_true", help="skip training, just reload an existing checkpoint")
    args = parser.parse_args()

    model_v2, adata_RNA, adata_ADT = run_citeseq(args.out_dir, args.data_dir, resume=args.resume)

    # Sanity check against the original ~0.485 baseline correlation.
    import torch
    import numpy as np

    with torch.no_grad():
        x_A_v2 = torch.from_numpy(model_v2.emb_A).float().to(model_v2.device)
        z1_v2 = model_v2.E_A(x_A_v2)
        deterministic_pred_v2 = model_v2.G_B(z1_v2).cpu().numpy()
    gt_protein_v2 = model_v2.emb_B
    det_corr_v2 = [
        np.corrcoef(deterministic_pred_v2[:, j], gt_protein_v2[:, j])[0, 1]
        for j in range(gt_protein_v2.shape[1])
    ]
    print("model_v2 deterministic mean correlation:", np.nanmean(det_corr_v2))
    print("Original Phase 0 baseline was ~0.485 -- should be close, not necessarily identical.")

    return model_v2


if __name__ == "__main__":
    main()
