"""
totalVI baseline comparison. Runs in the separate `totalvi` conda
environment (scvi-tools), never mixed with `scmodal`'s environment.

IMPORTANT UNRESOLVED CAVEAT (flagged in the original notebook, still
unresolved here — do not use the 0.664 correlation figure in any final
write-up until this is fixed): totalVI's accuracy number is computed in
raw ADT count space over 228 proteins, while scMODAL/flow's correlations
are computed in log1p-normalized, z-scored space over 260 feature slots
(172 unique shared proteins + 56 unshared -- the correspondence table maps
some proteins to multiple genes, so "260" is not 260 distinct biological
targets). These are not directly comparable as-is, on two independent
axes (feature count/identity, and value scale). Separately, the 0.520
figure from `model_v2` (the reproducibility-fixed retrain) was never
actually compared against totalVI at all -- only the original checkpoint's
0.485/0.475 were. See decisions.md for the full evidence trail. Re-expressing
totalVI's predictions into scMODAL's exact feature space is a prerequisite
for any head-to-head accuracy claim, not just a nice-to-have.
"""

import numpy as np


def build_totalvi_anndata(adata_RNA, adata_ADT):
    """Wraps RNA counts + raw protein counts into the AnnData layout totalVI expects."""
    adata_totalvi = adata_RNA.copy()
    adata_totalvi.layers["counts"] = adata_RNA.X.copy()
    adata_totalvi.obsm["protein_expression"] = adata_ADT.to_df()
    print(adata_totalvi.shape, adata_totalvi.obsm["protein_expression"].shape)
    return adata_totalvi


def train_totalvi(adata_totalvi, out_dir: str = "./totalvi_model", max_epochs: int = 100):
    """Trains totalVI. ~23-25 s/epoch on CPU for this dataset. `reduce_lr_on_plateau=False`
    avoids a scvi-tools/PyTorch version incompatibility in the LR scheduler.

    Save happens immediately after training, before any posterior sampling --
    get_normalized_expression/posterior_predictive_sample on the full gene
    set can exhaust memory and kill the kernel, and there's no reason to
    risk losing a completed training run.
    """
    import scvi

    scvi.model.TOTALVI.setup_anndata(adata_totalvi, protein_expression_obsm_key="protein_expression", layer="counts")
    vae = scvi.model.TOTALVI(adata_totalvi, latent_distribution="normal")
    vae.train(max_epochs=max_epochs, early_stopping=True, reduce_lr_on_plateau=False)
    vae.save(out_dir, overwrite=True)
    print("saved to", out_dir)
    return vae


def load_totalvi(adata_totalvi, model_dir: str = "./totalvi_model"):
    """Reload after a kernel restart. scvi-tools' save format needs
    weights_only=False under newer PyTorch (which now defaults to True) --
    only do this for files you trained yourself."""
    import torch
    import scvi

    _original_load = torch.load
    torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})
    vae = scvi.model.TOTALVI.load(model_dir, adata=adata_totalvi)
    torch.load = _original_load
    return vae


def posterior_predictive_samples(vae, adata_totalvi, n_samples: int = 25, chunk_size: int = 2000) -> np.ndarray:
    """Draws posterior predictive protein-count samples in chunks (progress
    printed so a long run doesn't look like a hang). Returns raw count-scale
    samples, shape (cells, proteins, samples) -- NOT the same scale as
    get_normalized_expression, and not directly comparable to scMODAL's
    log-normalized scaled ground truth (see module docstring)."""
    n_cells = adata_totalvi.shape[0]
    all_samples = []
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        chunk = adata_totalvi[start:end].copy()
        samples = vae.posterior_predictive_sample(chunk, n_samples=n_samples, gene_list=[], protein_list=None)
        all_samples.append(samples)
        print(f"done {end}/{n_cells}")
    posterior_samples_full = np.concatenate(all_samples, axis=0)
    print(posterior_samples_full.shape)
    return posterior_samples_full


def accuracy_count_scale(posterior_samples_full: np.ndarray, gt_protein_raw: np.ndarray):
    """Mean/std across posterior samples, and per-protein correlation with
    raw-count ground truth. NOTE: this is the raw-count-scale figure with
    the cross-scale caveat above -- not a like-for-like comparison with
    scMODAL/flow's log-normalized-scale correlations without further work."""
    totalvi_mean_counts = posterior_samples_full.mean(axis=2)
    totalvi_std_counts = posterior_samples_full.std(axis=2)
    n_proteins = gt_protein_raw.shape[1]
    totalvi_corr_counts = [
        np.corrcoef(totalvi_mean_counts[:, j], gt_protein_raw[:, j])[0, 1] for j in range(n_proteins)
    ]
    print("totalVI mean corr (count scale, NOT directly comparable to scMODAL scale):", np.nanmean(totalvi_corr_counts))
    return totalvi_mean_counts, totalvi_std_counts, np.array(totalvi_corr_counts)


def calibration_curve_totalvi(posterior_samples_full: np.ndarray, gt_protein_raw: np.ndarray, nominal_levels):
    """Empirical coverage of percentile-based intervals from the posterior
    samples directly (no Gaussian assumption, unlike src/calibration.py's
    scalar-correction approach for the flow model)."""
    actual_coverages = []
    for level in nominal_levels:
        lo_pct = (1 - level) / 2 * 100
        hi_pct = (1 + level) / 2 * 100
        lower = np.percentile(posterior_samples_full, lo_pct, axis=2)
        upper = np.percentile(posterior_samples_full, hi_pct, axis=2)
        inside = (gt_protein_raw >= lower) & (gt_protein_raw <= upper)
        actual_coverages.append(inside.mean())
    for nom, act in zip(nominal_levels, actual_coverages):
        print(f"Nominal: {nom:.1f}  Actual: {act:.3f}")
    return np.array(actual_coverages)
