"""
Raw data loading and preprocessing for TEA-seq PBMC (tri-modal: RNA, ADT,
ATAC). Unlike CITE-seq, correspondence here comes from a static file
(substituted from the MaxFuse repo's `protein_gene_conversion.csv`, saved
locally as `protein_gene_conversion_new.csv`, with TCR naming fixes
applied), not a live API call -- so this dataset never had CITE-seq's
reload/reproducibility problem.

40 of 46 ADT markers match to RNA genes. The 6 that don't are expected, not
gaps: TCR-a/b and TCR-g/d are pan-TCR complexes that assemble from many
V/D/J segments (no single-gene HGNC symbol); TCR-Va24-Ja18 and TCR-Va7.2
are isoform-specific clones intentionally excluded from the source table;
IgG1-K-Isotype-Control is a negative control, not a biological target; IgD
(gene IGHD) is real but absent from this particular RNA reference.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc


def load_teaseq_raw(data_dir: str = "./TEA-seq_PBMC/data/tea-seq"):
    """Loads raw TEA-seq RNA/ATAC/ADT AnnData. Expected shapes:
    RNA (7437, 18352), ATAC (7437, 18352), ADT (7437, 46)."""
    data_dir = Path(data_dir)
    adata_RNA = sc.read_h5ad(data_dir / "RNA.h5ad")
    adata_ATAC = sc.read_h5ad(data_dir / "ATAC.h5ad")
    adata_ADT = sc.read_h5ad(data_dir / "ADT.h5ad")
    print(adata_RNA.shape, adata_ATAC.shape, adata_ADT.shape)
    return adata_RNA, adata_ATAC, adata_ADT


def build_teaseq_correspondence(adata_RNA, adata_ADT, conversion_csv: str = "./TEA-seq_PBMC/data/tea-seq/protein_gene_conversion_new.csv") -> np.ndarray:
    """Builds the RNA<->protein correspondence table from the static MaxFuse
    conversion file (with TCR naming fixes already applied in the CSV)."""
    correspondence = pd.read_csv(conversion_csv)
    correspondence["Protein name"] = correspondence["Protein name"].replace({
        "TCR.a.b": "TCR-a/b",
        "TCR.g.d": "TCR-g/d",
    })

    rna_protein_correspondence = []
    for i in range(correspondence.shape[0]):
        curr_protein_name, curr_rna_names = correspondence.iloc[i]
        if curr_protein_name not in adata_ADT.var_names:
            continue
        if curr_rna_names.find("Ignore") != -1:
            continue
        for r in curr_rna_names.split("/"):
            if r in adata_RNA.var_names:
                rna_protein_correspondence.append([r, curr_protein_name])

    rna_protein_correspondence = np.array(rna_protein_correspondence)
    matched = set(rna_protein_correspondence[:, 1])
    missing = set(adata_ADT.var_names) - matched
    print(f"{len(rna_protein_correspondence)} matched RNA-protein pairs")
    print(f"{len(matched)} of {adata_ADT.shape[1]} ADT markers matched")
    print("Unmatched (expected, see module docstring):", sorted(missing))
    return rna_protein_correspondence


def preprocess_teaseq(adata_RNA, adata_ADT, adata_ATAC, correspondence: np.ndarray, n_top_genes_rna: int = 1000, n_top_genes_atac: int = 1000):
    """Builds the three preprocessed modality matrices (adata1=ADT, adata2=RNA,
    adata3=ATAC -- note ADT is index 0 here, unlike CITE-seq where RNA is
    index 0) plus the shared PCA space used for the RNA<->ATAC MNN anchor.

    Returns (adata1, adata2, adata3, adata23_shared, RNA_shared).
    """
    RNA_shared = adata_RNA[:, correspondence[:, 0]].copy()
    ADT_shared = adata_ADT[:, correspondence[:, 1]].copy()
    RNA_shared.var["feature_name"] = RNA_shared.var.index.values
    ADT_shared.var["feature_name"] = ADT_shared.var.index.values
    RNA_shared.var_names_make_unique()
    ADT_shared.var_names_make_unique()

    RNA_unshared = adata_RNA[:, sorted(set(adata_RNA.var.index) - set(correspondence[:, 0]))].copy()
    ADT_unshared = adata_ADT[:, sorted(set(adata_ADT.var.index) - set(correspondence[:, 1]))].copy()

    sc.pp.highly_variable_genes(RNA_unshared, flavor="seurat_v3", n_top_genes=n_top_genes_rna)
    RNA_unshared = RNA_unshared[:, RNA_unshared.var.highly_variable].copy()
    RNA_unshared.var["feature_name"] = RNA_unshared.var.index.values
    ADT_unshared.var["feature_name"] = ADT_unshared.var.index.values

    RNA_counts = RNA_shared.X.sum(axis=1)
    target_sum = np.maximum(np.median(np.array(RNA_counts).copy()), 20)

    sc.pp.normalize_total(RNA_shared, target_sum=target_sum)
    sc.pp.log1p(RNA_shared)
    sc.pp.normalize_total(ADT_shared, target_sum=target_sum)
    sc.pp.log1p(ADT_shared)
    sc.pp.normalize_total(RNA_unshared)
    sc.pp.log1p(RNA_unshared)
    sc.pp.normalize_total(ADT_unshared)
    sc.pp.log1p(ADT_unshared)

    adata1 = ad.concat([ADT_shared, ADT_unshared], axis=1)   # index 0: ADT
    adata2 = ad.concat([RNA_shared, RNA_unshared], axis=1)   # index 1: RNA
    sc.pp.scale(adata1, max_value=10)
    sc.pp.scale(adata2, max_value=10)

    sc.pp.highly_variable_genes(adata_ATAC, flavor="seurat_v3", n_top_genes=n_top_genes_atac)
    adata_ATAC = adata_ATAC[:, adata_ATAC.var.highly_variable].copy()
    adata_ATAC.var["feature_name"] = adata_ATAC.var.index.values

    adata3 = adata_ATAC.copy()                               # index 2: ATAC
    sc.pp.normalize_total(adata3)
    sc.pp.log1p(adata3)
    sc.pp.scale(adata3, max_value=10)

    rna_atac_shared = sorted(list(adata2.var.index & adata3.var.index))
    adata23_shared = ad.concat([adata2[:, rna_atac_shared], adata3[:, rna_atac_shared]])
    sc.tl.pca(adata23_shared, n_comps=30)

    print(adata1.shape, adata2.shape, adata3.shape)  # expected: (7437, 52) (7437, 1046) (7437, 1000)
    return adata1, adata2, adata3, adata23_shared, RNA_shared


def train_teaseq_scmodal(adata1, adata2, adata3, adata23_shared, RNA_shared, out_dir: str = "./TEA-seq_PBMC"):
    """One-time (~13 hrs on CPU) tri-modal integration training. Only run
    this once -- use the fast-resume path in scripts/load_raw_data.py for
    every session after.

    PATH NOTE (resolved via audit, evidence below -- not a guess): scMODAL's
    tri-modal `integrate_datasets_feats` nests its own output one level
    deeper than `model_path` suggests -- notebook cell 70's own comment
    confirms this explicitly ("writes ckpt.pth into ./TEA-seq_PBMC/TEA-seq_PBMC/
    automatically"). The frozen reference cell that saves the companion
    ground-truth arrays (cell 72) uses top-level paths and was NEVER actually
    executed in the saved notebook (Section 2 is commented-out reference
    only). By contrast, cell 172 -- live, executed, with verified correct
    downstream results (correlations matching project records) -- loads
    those same arrays from the NESTED path. Since only cell 172 has proof
    of actually working, this function saves to the nested path to match
    it and to stay internally consistent with scripts/compute_uncertainty.py
    (which already expected the nested location). If a fresh --train run
    ever produces different behavior than this, that's evidence this
    resolution needs revisiting -- see decisions.md.

    Note the trailing insurance-save step (adt_scaled_gt.npy, rna_scaled.npy)
    must run in the same session immediately after training completes --
    same lesson as the VelocityNet checkpoint incident (see decisions.md).

    Also caches n_ADT/n_RNA/n_ATAC to cell_counts.json. TEA-seq is a paired
    tri-modal assay (same cells profiled simultaneously across all three
    modalities -- see docs/flow.md / decisions.md), so these three values are
    always equal to the shared cell count; caching them here means a later
    fast-resume + latent-extraction pass doesn't need the raw AnnData files
    on disk just to know how to split model.latent. This is bookkeeping only
    -- it changes nothing about the trained model or any existing result.
    """
    import json
    import scmodal

    os.makedirs(out_dir, exist_ok=True)
    model = scmodal.model.Model(training_steps=20000, model_path=out_dir)

    model.integrate_datasets_feats(
        input_feats=[adata1.X, adata2.X, adata3.X],
        paired_input_MNN=[
            [adata1.X[:, :RNA_shared.shape[1]], adata2.X[:, :RNA_shared.shape[1]]],  # ADT<->RNA
            [adata23_shared.obsm["X_pca"][: adata2.shape[0]], adata23_shared.obsm["X_pca"][adata2.shape[0]:]],  # RNA<->ATAC
        ],
    )
    # ckpt.pth lands nested at {out_dir}/TEA-seq_PBMC/ automatically (see docstring)
    nested_dir = os.path.join(out_dir, "TEA-seq_PBMC")
    os.makedirs(nested_dir, exist_ok=True)

    np.save(os.path.join(nested_dir, "adt_scaled_gt.npy"), adata1.X)
    np.save(os.path.join(nested_dir, "rna_scaled.npy"), adata2.X)
    print("Saved ground-truth matrices:", adata1.X.shape, adata2.X.shape)

    cell_counts = {"n_ADT": adata1.shape[0], "n_RNA": adata2.shape[0], "n_ATAC": adata3.shape[0]}
    with open(os.path.join(nested_dir, "cell_counts.json"), "w") as f:
        json.dump(cell_counts, f)
    print("Cached cell counts:", cell_counts)

    return model


def load_cell_counts(out_dir: str = "./TEA-seq_PBMC"):
    """Loads n_ADT/n_RNA/n_ATAC from cell_counts.json if it exists (written
    by train_teaseq_scmodal on the one-time training run, to the nested
    directory -- see that function's docstring for why). Returns None if
    the cache doesn't exist yet -- caller should fall back to loading raw
    AnnData for cell counts in that case (e.g. an older TEA-seq_PBMC/
    directory trained before this caching was added)."""
    import json

    cache_path = os.path.join(out_dir, "TEA-seq_PBMC", "cell_counts.json")
    if not os.path.exists(cache_path):
        return None
    with open(cache_path) as f:
        return json.load(f)


def resume_teaseq_scmodal(out_dir: str = "./TEA-seq_PBMC"):
    """Fast-resume: rebuilds encoder/decoder architecture and loads saved
    weights, no retraining. Use this in every session after the first."""
    import torch
    import scmodal
    from scmodal.networks import encoder, generator

    device = "cpu"
    n_latent = 20
    dims = [52, 1046, 1000]  # ADT, RNA, ATAC feature dims

    nested_dir = os.path.join(out_dir, "TEA-seq_PBMC")
    model = scmodal.model.Model(training_steps=20000, model_path=nested_dir)
    ckpt = torch.load(os.path.join(nested_dir, "ckpt.pth"), map_location=device)

    model.E_dict, model.G_dict = {}, {}
    for i, dim in enumerate(dims):
        model.E_dict[i] = encoder(dim, n_latent).to(device)
        model.G_dict[i] = generator(dim, n_latent).to(device)
        model.E_dict[i].load_state_dict(ckpt["E_%d" % i])
        model.G_dict[i].load_state_dict(ckpt["G_%d" % i])
        model.E_dict[i].eval()
        model.G_dict[i].eval()

    adata1_X = np.load(os.path.join(nested_dir, "adt_scaled_gt.npy"))
    adata2_X = np.load(os.path.join(nested_dir, "rna_scaled.npy"))
    print("Loaded -- ready to reuse E_dict/G_dict without retraining")
    return model, adata1_X, adata2_X


def extract_and_save_latents(model, n_ADT: int, n_RNA: int, n_ATAC: int, out_dir: str = "./TEA-seq_PBMC", adata1_X=None, adata2_X=None):
    """Extracts z_ADT/z_RNA/z_ATAC either from a freshly-trained model's
    `.latent` attribute, or (after a fast resume) via a forward pass through
    the loaded encoders. The fast-resume path can only recover z_ADT/z_RNA
    -- ATAC features aren't persisted to disk, so z_ATAC needs a fresh
    training run to reconstruct.
    """
    import torch

    if hasattr(model, "latent"):
        z_ADT_np = model.latent[:n_ADT]
        z_RNA_np = model.latent[n_ADT : n_ADT + n_RNA]
        z_ATAC_np = model.latent[n_ADT + n_RNA :]
    else:
        if adata1_X is None or adata2_X is None:
            raise ValueError("adata1_X/adata2_X required for the fast-resume forward-pass path")
        with torch.no_grad():
            z_ADT_np = model.E_dict[0](torch.from_numpy(adata1_X).float()).cpu().numpy()
            z_RNA_np = model.E_dict[1](torch.from_numpy(adata2_X).float()).cpu().numpy()
        z_ATAC_np = None

    np.save(os.path.join(out_dir, "z_ADT_tea.npy"), z_ADT_np)
    np.save(os.path.join(out_dir, "z_RNA_tea.npy"), z_RNA_np)
    if z_ATAC_np is not None:
        np.save(os.path.join(out_dir, "z_ATAC_tea.npy"), z_ATAC_np)
    print(z_ADT_np.shape, z_RNA_np.shape, z_ATAC_np.shape if z_ATAC_np is not None else "(not recomputed)")
    return z_ADT_np, z_RNA_np, z_ATAC_np
