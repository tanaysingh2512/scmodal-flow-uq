# Pipeline Flow

This is the exact order things run in, what each stage depends on, and what
it produces. If a step's inputs don't exist yet, run the steps above it first.

```
scripts/load_raw_data.py --dataset citeseq [--resume]
  - src/data_prep.py: load_citeseq_raw()               <- data/citeseq_pbmc/multi.h5ad, ADT.csv
  - src/data_prep.py: build_or_load_correspondence()   -> CITE-seq_PBMC_v2/rna_protein_correspondence.csv (tracked in git, see .gitignore)
  - src/data_prep.py: build_shared_unshared(), normalize_and_concat()
  - trains or fast-resumes scMODAL                     -> CITE-seq_PBMC_v2/ckpt.pth
  produces in-memory: model_v2, gt_protein_v2, deterministic_pred_v2 (point predictions)
  (scripts/train_scmodal.py wraps this same path + adds the ~0.485 baseline sanity check)

scripts/load_raw_data.py --dataset teaseq [--train | --resume]
  - src/teaseq_prep.py: load_teaseq_raw()              <- data/tea-seq/RNA.h5ad, ATAC.h5ad, ADT.h5ad
  - src/teaseq_prep.py: build_teaseq_correspondence()  <- data/tea-seq/protein_gene_conversion_new.csv (static, not live API; tracked in git, see .gitignore)
  - src/teaseq_prep.py: preprocess_teaseq()            -> adata1 (ADT), adata2 (RNA), adata3 (ATAC), adata23_shared (PCA anchor)
  - --train: train_teaseq_scmodal()  (~13 hrs on CPU)  -> TEA-seq_PBMC/TEA-seq_PBMC/{ckpt.pth, adt_scaled_gt.npy, rna_scaled.npy, cell_counts.json}
    (all four nested -- see decisions.md audit entry on the companion-file nesting fix)
  - --resume: resume_teaseq_scmodal()                  <- loads existing ckpt.pth, no retraining
  - extract_and_save_latents()                          -> TEA-seq_PBMC/z_ADT_tea.npy, z_RNA_tea.npy, (z_ATAC_tea.npy if fresh-trained)
        |
        v
scripts/compute_uncertainty.py
  - src/uncertainty.py: compute_epistemic_uncertainty()   (SG1, both datasets)
      -> CITE-seq_PBMC_v2/epistemic_uncertainty.npy
      -> TEA-seq_PBMC/epistemic_uncertainty.npy
  - src/uncertainty.py: mc_noise_forward()                (SG3, both datasets)
      -> CITE-seq_PBMC_v2/integration_uncertainty.npy
      -> TEA-seq_PBMC/TEA-seq_PBMC/integration_uncertainty_tea.npy
  produces in-memory: per_cell_error_cite_v2, per_cell_error_tea
        |
        v
scripts/train_ua_flow.py                                  (SG2, both datasets)
  - src/velocitynet.py: train_ua_flow(), sample_ua_flow(), learned_variance()
      -> CITE-seq_PBMC_v2/ua_flow_vnet.pth
      -> TEA-seq_PBMC/ua_flow_vnet.pth
  - src/calibration.py: fit_scalar_correction()  (post-hoc scalar fix)
      best_scale (CITE-seq, ~0.520) / best_scale_tea (TEA-seq, ~0.120)
        |
        v
scripts/finetune_sg4.py                                   (SG4, CITE-seq only)
  requires: model_v2 (from train_scmodal.py) + epistemic_uncertainty.npy,
            integration_uncertainty.npy (from compute_uncertainty.py)
  - src/finetune.py: run_sg4_finetune()
      1. seed sweep: floor=0.2 vs. seed-matched floor=1.0 control, seeds 0-4
      2. strength sweep: floors [0.0, 0.25, 0.5] vs. floor=1.0/seed=0 control
      3. SG3-signal robustness check: same floor=0.2, using integration_uncertainty
         instead of epistemic_uncertainty
      -> CITE-seq_PBMC_v2/sg4_results.json
        |
        v
scripts/make_figures.py
  regenerates every figure referenced in the write-up, including the two
  SG4 plots (which must use corrected numbers from finetune_sg4.py's output,
  not the old uncorrected run -- see decisions.md)
```

## totalVI baseline

Independent of the flow-matching pipeline above. Trained separately in the
`totalvi` conda environment (kept strictly apart from `scmodal` — do not
install both in one environment, see environment files).

```
scripts/train_totalvi.py [--resume]
  - src/data_prep.py: load_citeseq_raw()          <- same raw CITE-seq source as the main pipeline
  - src/totalvi_baseline.py: build_totalvi_anndata()
  - train_totalvi() (~35-90 min, 100 epochs, CPU) or load_totalvi() if --resume
      -> totalvi_model/
  - posterior_predictive_samples()                 -> totalvi_results/totalvi_posterior_samples.npy
  - accuracy_count_scale()                          -> totalvi_results/totalvi_{mean,std,corr}_counts.npy
      *** CAVEAT: raw-count-scale correlation, NOT directly comparable to
      scMODAL/flow's log-normalized-scaled-space correlations. Unresolved --
      see src/totalvi_baseline.py docstring and decisions.md. ***
  - calibration_curve_totalvi()                     -> totalvi_results/totalvi_{nominal_levels,actual_coverages}.npy
        |
        v
scripts/make_figures.py: plot_flow_vs_totalvi_calibration()
```

## Environments

- `environment-scmodal.yml` — used for every script above except totalVI.
- `environment-totalvi.yml` — used only for the totalVI baseline comparison.
Never install both into the same conda environment (scvi-tools dependency
conflicts silently break torch/pandas in the `scmodal` env).

## Open items (tracked here so they don't get silently forgotten)

1. ~~Raw data loading not migrated out of the notebook~~ — **resolved.**
   `scripts/load_raw_data.py` + `src/data_prep.py` (CITE-seq) and
   `src/teaseq_prep.py` (TEA-seq) now cover this. Note: the raw source files
   themselves (`data/citeseq_pbmc/multi.h5ad`, `data/tea-seq/*.h5ad`, etc.)
   are not included in this repo (too large / not meant for git) — you still
   need to place them at the expected paths yourself.
2. ~~totalVI baseline not migrated to a script~~ — **resolved.**
   `scripts/train_totalvi.py` + `src/totalvi_baseline.py`. **But**: the
   cross-scale accuracy caveat noted in the original notebook is still
   genuinely unresolved, not just a migration gap — see item #6 below.
3. ~~TEA-seq SG3 `x_tea_tensor` gap~~ — **resolved (confirmed by evidence, not guessed).**
   Notebook cell 174 literally defines `x_tea_tensor = torch.tensor(adata2_X, dtype=torch.float32)`,
   where `adata2_X` is loaded two cells earlier (cell 172) via
   `adata2_X = np.load('.../rna_scaled.npy')` (the fast-resume path), with
   shape (7437, 1046) confirmed by the notebook's own printed output in
   cell 173. `compute_uncertainty.py` now reproduces this exactly. Also
   found and fixed two real bugs while verifying this: (1) the checkpoint
   was being loaded with a wrong key (`ckpt["E_dict"][1]` instead of the
   correct `ckpt["E_%d" % i]` loop covering all three modalities' E/G
   networks), and (2) `adata2_X` itself was never actually loaded in the
   script (would have crashed with a `NameError`). Both fixed. See
   decisions.md for the full audit entry.
4. **SG1/SG2 combined plots** (`nn_distance_scatter_both.png`,
   `ua_flow_recalibration_both.png`) and the SG3 scatter plots were only
   ever shown via `plt.show()` in the working notebook, never confirmed
   saved to disk. `make_figures.py` now has `savefig` calls for all of
   these — run it end to end once the upstream scripts produce the needed
   arrays, and confirm the files actually land in `results/figures/`.
5. ~~TEA-seq fast-resume needs raw AnnData for cell counts~~ — **resolved.**
   `train_teaseq_scmodal()` now caches `n_ADT`/`n_RNA`/`n_ATAC` to
   `cell_counts.json` at training time (TEA-seq is a paired tri-modal assay
   — same cells profiled across all three modalities simultaneously, so
   these three values are always equal; confirmed via the notebook's own
   expected-shape comment: `(7437, ...)` for RNA, ATAC, and ADT alike).
   `load_raw_data.py`'s resume path reads this cache first and only falls
   back to loading raw AnnData if a `TEA-seq_PBMC/` directory predates this
   change. Pure bookkeeping — does not touch the trained checkpoint or any
   existing result.
6. **totalVI accuracy figure is on a different feature scale than scMODAL/flow
   — genuinely unresolved, not just a script-migration gap.** totalVI's 0.664
   correlation is computed in raw ADT count space over 228 features; scMODAL
   and the flow's correlations (0.485, 0.475 — the *original* checkpoint's
   numbers, the ones actually compared against totalVI) are computed in
   log-normalized *scaled* space over 260 features (172 unique shared
   proteins + 56 unshared, since the correspondence table maps some proteins
   to multiple genes). These three numbers are not a valid apples-to-apples
   comparison as currently computed. Separately: `model_v2`'s deterministic
   correlation (0.520) is a *different, later* checkpoint that was never
   compared against totalVI at all — don't substitute it into this
   comparison either. This was flagged as unresolved in the original
   notebook itself, before any of this restructuring — re-expressing
   totalVI's predictions in scMODAL's exact feature space is required before
   this comparison goes in any final write-up or paper. Do not cite
   "totalVI beats both on accuracy" as a settled finding until this is fixed.
   Full breakdown: docs/decisions.md, 2026-09-04 audit entry.
