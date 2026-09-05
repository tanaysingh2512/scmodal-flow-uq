"""
Stretch Goals 1 & 3: compute the two cheap, non-generative uncertainty
signals for CITE-seq and TEA-seq, and correlate each against per-cell
imputation error.

Requires: a trained/resumed model_v2 (train_scmodal.py) for CITE-seq, and
for TEA-seq: z_RNA_tea.npy, z_ADT_tea.npy, adt_scaled_gt.npy, rna_scaled.npy,
imputed_mean.npy, and ckpt.pth already on disk (see docs/flow.md).

Usage:
    python scripts/compute_uncertainty.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# scMODAL is a git submodule (see README), imported via current-directory
# resolution -- not pip-installed. This must be on sys.path before any
# `import scmodal` call below, regardless of the caller's cwd.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scMODAL"))

import numpy as np
import torch

from src.uncertainty import compute_epistemic_uncertainty, mc_noise_forward, integration_uncertainty_from_samples

CITE_DIR = "CITE-seq_PBMC_v2"
TEA_DIR = "TEA-seq_PBMC/TEA-seq_PBMC"
TEA_ROOT = "TEA-seq_PBMC"


def citeseq_uncertainty(model_v2):
    with torch.no_grad():
        x_A_v2 = torch.from_numpy(model_v2.emb_A).float().to(model_v2.device)
        x_B_v2 = torch.from_numpy(model_v2.emb_B).float().to(model_v2.device)
        z1_v2 = model_v2.E_A(x_A_v2)
        z2_v2 = model_v2.E_B(x_B_v2)
        deterministic_pred_v2 = model_v2.G_B(z1_v2).cpu().numpy()

    z1_cite = z1_v2.cpu().numpy()
    z2_cite = z2_v2.cpu().numpy()
    gt_protein_v2 = model_v2.emb_B

    # SG1: epistemic (NN-distance) uncertainty
    epistemic_unc_cite = compute_epistemic_uncertainty(z1_cite, z2_cite, k=10)
    np.save(os.path.join(CITE_DIR, "epistemic_uncertainty.npy"), epistemic_unc_cite)

    error_cite_v2 = np.abs(deterministic_pred_v2 - gt_protein_v2)
    per_cell_error_cite_v2 = error_cite_v2.mean(axis=1)
    corr_cite = np.corrcoef(per_cell_error_cite_v2, epistemic_unc_cite)[0, 1]
    print(f"CITE-seq: epistemic uncertainty vs. error correlation = {corr_cite:.3f}")

    # SG3: integration uncertainty (weight-noise MC; scMODAL's encoder has no dropout layers)
    z_samples = mc_noise_forward(model_v2.E_A, x_A_v2, n_samples=20, noise_scale=0.20)
    integration_unc_cite = integration_uncertainty_from_samples(z_samples)
    np.save(os.path.join(CITE_DIR, "integration_uncertainty.npy"), integration_unc_cite.numpy())

    corr_integration = np.corrcoef(integration_unc_cite.numpy(), per_cell_error_cite_v2)[0, 1]
    print(f"CITE-seq: integration uncertainty vs. error correlation = {corr_integration:.3f}  (expected ~0.12)")

    return per_cell_error_cite_v2, epistemic_unc_cite, integration_unc_cite.numpy()


def teaseq_uncertainty():
    z1_tea = np.load(os.path.join(TEA_ROOT, "z_RNA_tea.npy"))
    z2_tea = np.load(os.path.join(TEA_ROOT, "z_ADT_tea.npy"))
    gt_protein_tea = np.load(os.path.join(TEA_DIR, "adt_scaled_gt.npy"))
    imputed_mean_tea = np.load(os.path.join(TEA_DIR, "imputed_mean.npy"))

    # SG1
    epistemic_unc_tea = compute_epistemic_uncertainty(z1_tea, z2_tea, k=10)
    np.save(os.path.join(TEA_ROOT, "epistemic_uncertainty.npy"), epistemic_unc_tea)

    error_tea = np.abs(imputed_mean_tea - gt_protein_tea)
    per_cell_error_tea = error_tea.mean(axis=1)
    corr_tea = np.corrcoef(per_cell_error_tea, epistemic_unc_tea)[0, 1]
    print(f"TEA-seq: epistemic uncertainty vs. error correlation = {corr_tea:.3f}  (project record: 0.701)")

    # SG3: reload the tri-modal TEA-seq model to get its RNA encoder (index 1 = RNA).
    # Loading matches notebook cell 172 exactly: checkpoint keys are 'E_%d'/'G_%d' per
    # modality index, not a nested 'E_dict' key (that was a bug in an earlier draft of
    # this script -- caught during the reproducibility audit, see decisions.md).
    from scmodal.networks import encoder, generator
    import scmodal

    device = "cpu"
    n_latent = 20
    dims = [52, 1046, 1000]  # ADT, RNA, ATAC feature dims
    model = scmodal.model.Model(training_steps=20000, model_path=TEA_DIR)
    ckpt = torch.load(os.path.join(TEA_DIR, "ckpt.pth"), map_location=device)
    model.E_dict, model.G_dict = {}, {}
    for i, dim in enumerate(dims):
        model.E_dict[i] = encoder(dim, n_latent).to(device)
        model.G_dict[i] = generator(dim, n_latent).to(device)
        model.E_dict[i].load_state_dict(ckpt["E_%d" % i])
        model.G_dict[i].load_state_dict(ckpt["G_%d" % i])
        model.E_dict[i].eval()
        model.G_dict[i].eval()

    # CONFIRMED (not a guess -- see decisions.md audit entry): the original notebook's
    # cell 174 defines `x_tea_tensor = torch.tensor(adata2_X, dtype=torch.float32)`
    # directly, where `adata2_X` is the RNA-scaled array loaded via fast-resume
    # (cell 172: `adata2_X = np.load('.../rna_scaled.npy')`), confirmed shape (7437, 1046)
    # by the notebook's own printed output. This is exactly the array reproduced here.
    adata2_X = np.load(os.path.join(TEA_DIR, "rna_scaled.npy"))
    x_tea_tensor = torch.from_numpy(adata2_X).float()
    z_samples_tea = mc_noise_forward(model.E_dict[1], x_tea_tensor, n_samples=20, noise_scale=0.20)
    integration_unc_tea = integration_uncertainty_from_samples(z_samples_tea)
    np.save(os.path.join(TEA_DIR, "integration_uncertainty_tea.npy"), integration_unc_tea.numpy())

    corr_integration_tea = np.corrcoef(integration_unc_tea.numpy(), per_cell_error_tea)[0, 1]
    print(f"TEA-seq: integration uncertainty vs. error correlation = {corr_integration_tea:.3f}  (project record: 0.622)")

    return per_cell_error_tea, epistemic_unc_tea, integration_unc_tea.numpy()


if __name__ == "__main__":
    raise SystemExit(
        "This script needs a live model_v2 (from train_scmodal.py) passed into "
        "citeseq_uncertainty(); wire that up in your driver / __main__ before running standalone."
    )
