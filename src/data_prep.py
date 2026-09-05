"""
Reproducible data preparation for CITE-seq PBMC.

Why this exists: the original pipeline resolved RNA gene -> ADT protein
correspondence via a live call to the mygene.info API, which can return
different matches across sessions/machines. That made the original
CITE-seq_PBMC/ckpt.pth checkpoint impossible to reload consistently.

The fix: resolve the correspondence table once, pin it to a CSV on disk,
and always load from that CSV afterwards.

See flow.md for where this sits in the overall pipeline, and decisions.md
for why this replaced the original live-API version.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc


def load_citeseq_raw(data_dir: str = "./data/citeseq_pbmc"):
    """Loads raw CITE-seq PBMC RNA + ADT AnnData, filtered to donor P1 with
    doublets removed. This is the actual entry point of the whole pipeline --
    everything else depends on adata_RNA/adata_ADT existing.

    data_dir must contain multi.h5ad and ADT.csv (see docs/flow.md for the
    expected source/layout -- not included in this repo).
    """
    data_dir = Path(data_dir)
    adata_RNA = sc.read_h5ad(data_dir / "multi.h5ad")
    adata_RNA.var.index = adata_RNA.var["_index"]
    adata_RNA.X = adata_RNA.raw.X.toarray()

    counts_ADT = pd.read_csv(data_dir / "ADT.csv").T
    adata_ADT = ad.AnnData(X=counts_ADT.values)
    adata_ADT.obs.index = counts_ADT.index
    adata_ADT.var.index = counts_ADT.columns
    adata_ADT.obs = adata_RNA.obs.loc[adata_ADT.obs.index]

    adata_RNA = adata_RNA[adata_RNA.obs.donor == "P1"]
    adata_ADT = adata_ADT[adata_RNA.obs.index]
    adata_RNA = adata_RNA[adata_RNA.obs["celltype.l2"].values != "Doublet"]
    adata_ADT = adata_ADT[adata_ADT.obs["celltype.l2"].values != "Doublet"]

    print("RNA shape:", adata_RNA.shape)
    print("ADT shape:", adata_ADT.shape)
    return adata_RNA, adata_ADT


def build_or_load_correspondence(adata_RNA, adata_ADT, out_dir: str) -> np.ndarray:
    """Load the pinned gene<->protein correspondence table if it exists,
    otherwise build it once via mygene.info and pin it to disk.

    Returns an (N, 2) array of [gene, protein] pairs.
    """
    corr_path = Path(out_dir) / "rna_protein_correspondence.csv"
    os.makedirs(out_dir, exist_ok=True)

    if corr_path.exists():
        corr_df = pd.read_csv(corr_path)
        correspondence = corr_df[["gene", "protein"]].values
        print(
            f"Loaded existing pinned table: {len(correspondence)} pairs, "
            f"{len(set(correspondence[:, 1]))} unique proteins"
        )
        return correspondence

    import mygene

    mg = mygene.MyGeneInfo()
    proteins = adata_ADT.var_names.tolist()
    result = mg.querymany(
        proteins, scopes="name,symbol,alias", fields="symbol", species="human", verbose=False
    )

    correspondence = []
    for entry in result:
        if "symbol" in entry and not entry.get("notfound"):
            protein = entry["query"]
            gene = entry["symbol"]
            if protein in adata_ADT.var_names and gene in adata_RNA.var_names:
                correspondence.append([gene, protein])
    correspondence = np.array(correspondence)

    pd.DataFrame(correspondence, columns=["gene", "protein"]).to_csv(corr_path, index=False)
    print(
        f"Built and PINNED new table: {len(correspondence)} pairs, "
        f"{len(set(correspondence[:, 1]))} unique proteins. "
        "This file will be reused on every future run -- mygene will not be queried again."
    )
    return correspondence


def build_shared_unshared(adata_RNA, adata_ADT, correspondence: np.ndarray, n_top_genes: int = 3000):
    """Split RNA/ADT into shared (paired) and unshared (HVG-selected) feature sets."""
    RNA_shared = adata_RNA[:, correspondence[:, 0]].copy()
    ADT_shared = adata_ADT[:, correspondence[:, 1]].copy()
    RNA_shared.var["feature_name"] = RNA_shared.var.index.values
    ADT_shared.var["feature_name"] = ADT_shared.var.index.values
    RNA_shared.var_names_make_unique()
    ADT_shared.var_names_make_unique()

    RNA_unshared = adata_RNA[:, sorted(set(adata_RNA.var.index) - set(correspondence[:, 0]))].copy()
    ADT_unshared = adata_ADT[:, sorted(set(adata_ADT.var.index) - set(correspondence[:, 1]))].copy()

    sc.pp.highly_variable_genes(RNA_unshared, flavor="seurat_v3", n_top_genes=n_top_genes)
    RNA_unshared = RNA_unshared[:, RNA_unshared.var.highly_variable].copy()  # deterministic given a pinned table

    RNA_unshared.var["feature_name"] = RNA_unshared.var.index.values
    ADT_unshared.var["feature_name"] = ADT_unshared.var.index.values

    return RNA_shared, ADT_shared, RNA_unshared, ADT_unshared


def normalize_and_concat(RNA_shared, ADT_shared, RNA_unshared, ADT_unshared):
    """Normalize each block, then concatenate shared + unshared per modality."""
    RNA_counts = RNA_shared.X.sum(axis=1)
    target_sum = np.maximum(np.median(RNA_counts.copy()), 20)

    sc.pp.normalize_total(RNA_shared, target_sum=target_sum)
    sc.pp.log1p(RNA_shared)
    sc.pp.normalize_total(ADT_shared, target_sum=target_sum)
    sc.pp.log1p(ADT_shared)
    sc.pp.normalize_total(RNA_unshared)
    sc.pp.log1p(RNA_unshared)
    sc.pp.normalize_total(ADT_unshared)
    sc.pp.log1p(ADT_unshared)

    adata1 = ad.concat([RNA_shared, RNA_unshared], axis=1)
    adata2 = ad.concat([ADT_shared, ADT_unshared], axis=1)
    sc.pp.scale(adata1, max_value=10)
    sc.pp.scale(adata2, max_value=10)

    return adata1, adata2
