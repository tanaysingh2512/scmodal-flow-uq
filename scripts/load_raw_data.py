"""
Entry point for raw data loading -- the actual start of the pipeline (see
docs/flow.md). Loads CITE-seq and/or TEA-seq raw AnnData, builds/loads the
correspondence table, and either trains scMODAL from scratch or fast-resumes
an existing checkpoint.

CITE-seq is a two-modality integration handled by src/data_prep.py +
scripts/train_scmodal.py (that script now has a live raw-loading path -- the
NotImplementedError stub from the first pass is gone).

TEA-seq is a three-modality (RNA/ADT/ATAC) integration with its own module,
src/teaseq_prep.py, since the correspondence, preprocessing, and training
calls are meaningfully different (static correspondence file instead of a
live API call, three input matrices instead of two, PCA-based RNA<->ATAC
anchor).

Usage:
    python scripts/load_raw_data.py --dataset citeseq --resume
    python scripts/load_raw_data.py --dataset teaseq --resume
    python scripts/load_raw_data.py --dataset teaseq --train   # ~13 hrs on CPU
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# scMODAL is a git submodule (see README), imported via current-directory
# resolution -- not pip-installed. This must be on sys.path before any
# `import scmodal` call below, regardless of the caller's cwd.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scMODAL"))


def run_citeseq(citeseq_dir: str, data_dir: str, resume: bool):
    import scmodal
    from src.data_prep import load_citeseq_raw, build_or_load_correspondence, build_shared_unshared, normalize_and_concat

    adata_RNA, adata_ADT = load_citeseq_raw(data_dir)
    correspondence = build_or_load_correspondence(adata_RNA, adata_ADT, citeseq_dir)
    RNA_shared, ADT_shared, RNA_unshared, ADT_unshared = build_shared_unshared(adata_RNA, adata_ADT, correspondence)
    adata1, adata2 = normalize_and_concat(RNA_shared, ADT_shared, RNA_unshared, ADT_unshared)

    model_v2 = scmodal.model.Model(model_path=citeseq_dir)
    model_v2.preprocess(adata1, adata2, shared_gene_num=RNA_shared.shape[1])
    if not resume:
        model_v2.train()  # ~6 hrs on CPU
    model_v2.eval()

    print("Checkpoint exists:", os.path.exists(os.path.join(citeseq_dir, "ckpt.pth")))
    return model_v2, adata_RNA, adata_ADT


def run_teaseq(teaseq_root: str, data_dir: str, resume: bool, train: bool):
    from src.teaseq_prep import (
        load_teaseq_raw, build_teaseq_correspondence, preprocess_teaseq,
        train_teaseq_scmodal, resume_teaseq_scmodal, extract_and_save_latents, load_cell_counts,
    )

    if resume:
        model, adata1_X, adata2_X = resume_teaseq_scmodal(teaseq_root)

        cell_counts = load_cell_counts(teaseq_root)
        if cell_counts is not None:
            n_ADT, n_RNA, n_ATAC = cell_counts["n_ADT"], cell_counts["n_RNA"], cell_counts["n_ATAC"]
            print(f"Loaded cached cell counts (no raw AnnData needed): {cell_counts}")
        else:
            # Fallback for a TEA-seq_PBMC/ directory trained before cell-count caching was
            # added -- needs the raw AnnData just to read off n_ADT/n_RNA/n_ATAC.
            print("No cell_counts.json cache found -- falling back to loading raw AnnData just for cell counts.")
            adata_RNA, adata_ATAC, adata_ADT = load_teaseq_raw(data_dir)
            n_ADT, n_RNA, n_ATAC = adata_ADT.shape[0], adata_RNA.shape[0], adata_ATAC.shape[0]

        extract_and_save_latents(
            model, n_ADT=n_ADT, n_RNA=n_RNA, n_ATAC=n_ATAC,
            out_dir=teaseq_root, adata1_X=adata1_X, adata2_X=adata2_X,
        )
        return model

    adata_RNA, adata_ATAC, adata_ADT = load_teaseq_raw(data_dir)
    correspondence = build_teaseq_correspondence(adata_RNA, adata_ADT, os.path.join(data_dir, "protein_gene_conversion_new.csv"))
    adata1, adata2, adata3, adata23_shared, RNA_shared = preprocess_teaseq(adata_RNA, adata_ADT, adata_ATAC, correspondence)

    if train:
        model = train_teaseq_scmodal(adata1, adata2, adata3, adata23_shared, RNA_shared, out_dir=teaseq_root)
        extract_and_save_latents(
            model, n_ADT=adata_ADT.shape[0], n_RNA=adata_RNA.shape[0], n_ATAC=adata_ATAC.shape[0], out_dir=teaseq_root,
        )
        return model

    print("Preprocessing complete. Pass --train to run the ~13hr integration, or --resume to load an existing checkpoint.")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["citeseq", "teaseq"], required=True)
    parser.add_argument("--citeseq-dir", default="CITE-seq_PBMC_v2")
    parser.add_argument("--teaseq-root", default="TEA-seq_PBMC")
    parser.add_argument("--citeseq-data-dir", default="./data/citeseq_pbmc")
    parser.add_argument("--teaseq-data-dir", default="./TEA-seq_PBMC/data/tea-seq")
    parser.add_argument("--resume", action="store_true", help="fast-resume an existing checkpoint instead of training")
    parser.add_argument("--train", action="store_true", help="(TEA-seq only) run the one-time ~13hr integration")
    args = parser.parse_args()

    if args.dataset == "citeseq":
        run_citeseq(args.citeseq_dir, args.citeseq_data_dir, args.resume)
    else:
        run_teaseq(args.teaseq_root, args.teaseq_data_dir, args.resume, args.train)


if __name__ == "__main__":
    main()
