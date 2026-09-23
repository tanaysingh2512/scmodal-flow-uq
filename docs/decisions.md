# Decision Log

Every entry explains a decision made while building/restructuring this repo,
and why. Newest entries at the top. This file is updated every time a
structural or scope decision changes — it's the fixed record, not the code
comments.

---

### 2026-09-05 — Notebook retained as historical record; curated docs anonymized separately

**Decision:** `notebooks/scMODAL_FINAL_AllPhases_StretchGoals.ipynb` is
retained as the historical experimental record and is **not** edited for
anonymization beyond the one earlier functional fix (the hardcoded
`/Users/<username>/scMODAL` path, replaced with a portable equivalent — see
the 2026-09-05 cleanliness-scan entry above). It may still contain
supervisor attribution in its cell content (e.g. mentions of "Prashant" in
markdown/comments), left as-is deliberately, since the notebook's job is to
preserve what actually happened, not to be a public-facing document.

By contrast, this project's **curated documentation** — `README.md`,
`docs/flow.md`, and this file — has been made anonymization-neutral:
personal first names removed and replaced with neutral wording ("the
project owner," "supervisor review," etc.), per explicit review and
approval. These are two different artifacts serving two different
purposes, and this distinction is intentional, not an oversight.

---

### 2026-09-05 — SG4 numerical reconciliation: 2.39% vs. 2.68%

**Two different high-uncertainty-tertile regression figures appeared across
this project's history: 2.39% and 2.68%. Both are real notebook outputs,
not typos.**

- **2.39%** — cell 184, the *original, uncorrected* `run_sg4_finetune`.
  5-seed high-tertile values: `[2.57, 3.04, 2.37, 2.0, 1.98]`. Measures
  each treatment run against the raw, un-fine-tuned `model_v2` checkpoint.
- **2.68%** — cell 190, the *corrected* `run_sg4_finetune_v2` (the only
  version migrated into `src/finetune.py`). 5-seed high-tertile values:
  `[3.25, 2.17, 2.38, 2.69, 2.93]`. Measures each treatment run against a
  seed-matched `floor=1.0` control instead.

Both use identical checkpoint/seeds/floor/steps/uncertainty signal — the
only difference is the reference baseline. Fine-tuning alone (`floor=1.0`,
no down-weighting) shifts the high tertile by −0.66% relative to the raw
checkpoint (cell 185). Comparing against the raw checkpoint therefore nets
some of that shift against the down-weighting harm, **understating** it.
**2.68% is the authoritative, corrected figure.** 2.39% is stale — an
earlier methodology, not a fabrication or a different metric. It was never
migrated into any script.

Neither number had actually been written into `README.md`, `flow.md`, or
`decisions.md` before this entry — `make_figures.py` already used the
correct 2.68% (sourced directly from cell 190's output), but nothing in
prose form stated it precisely. Found and fixed one real inconsistency
while resolving this: `src/finetune.py`'s docstring previously hedged with
"~+2.4-2.7%" (a vague range straddling both numbers) — corrected to state
2.68% precisely, with the exact per-seed values cited.

---

### 2026-09-05 — Correspondence CSVs committed as tracked reproducibility fixtures

**Decision:** `rna_protein_correspondence.csv` (~204 rows, gene/protein name
pairs) and `data/tea-seq/protein_gene_conversion_new.csv` (~46 rows, protein/
gene name pairs, substituted from MaxFuse) are now tracked via `.gitignore`
negation, while everything else in their surrounding directories
(checkpoints, raw `.h5ad` files) stays ignored. Explicitly confirmed by
the project owner after reviewing the size/sensitivity/necessity
characterization below, rather than decided unilaterally.

**Why:** both are small, non-sensitive, static mapping tables (no expression
values, no patient data — just public gene/protein nomenclature), and both
are required directly by the pipeline: `build_or_load_correspondence()`
falls back to a live, non-deterministic `mygene.info` call if the CITE-seq
file is absent (defeating the entire point of pinning it), and
`build_teaseq_correspondence()` has no fallback at all for the TEA-seq file.
Sweeping them into whole-directory ignores alongside multi-GB checkpoints
would have meant a fresh clone still couldn't reproduce the exact
correspondence tables without either the raw data or a fresh, non-
deterministic query.

**Note on these files themselves:** I have never had direct access to
either CSV in this sandbox — this characterization is inferred from the
code that reads/writes them and the notebook's own printed output (204
pairs confirmed via cell 121), not from directly inspecting file contents.
Confirm row counts/content match this description once you have the actual
files, before treating this as fully verified.

**`.gitignore` mechanics:** git will not descend into a directory ignored
via a bare trailing-slash pattern (e.g. `data/`) to apply a later negation
rule — this is a documented git limitation, not a bug. Both ignore blocks
were rewritten using the `dir/*` + `!dir/subpath` idiom instead, and the
resulting patterns were verified in a disposable scratch git repo (not
this project's) to confirm exactly the two fixture files get tracked and
nothing else does, before writing this entry.

---

### 2026-09-05 — Repository cleanliness/security scan

**Found and fixed:**
- The author's username leaked into the repo in two places, both fixed:
  (1) the `notebooks/scMODAL_FINAL_AllPhases_StretchGoals.ipynb` copy still
  contained the original absolute path (with the author's username baked
  into a `sys.path.insert(...)` call and an `os.chdir(...)` call) in its raw
  cell source (the `.py` migration had already fixed this, but the notebook
  copy sitting in `notebooks/` never was) — replaced with
  `os.path.expanduser('~/scMODAL')`, functionally identical, no username.
  (2) this very log's earlier entry documenting that fix had quoted the
  literal leaked path as evidence — redacted to describe it without
  reproducing it.
- `.gitignore` had three real gaps: missing `*.pt` (scvi-tools' actual
  checkpoint extension — distinct from `*.pth`, would NOT have been caught
  by the existing pattern), no coverage for `totalvi_model/`/`totalvi_results/`,
  and no coverage for `data/` (raw datasets). All three added.
- No API keys, tokens, credentials, `.env` files, or OS/editor cruft
  (`.DS_Store`, `Thumbs.db`, `*.swp`) found anywhere in the repo.
- No large binaries (`.h5ad`, `.npy`, `.pth`, `.pt`) are actually present in
  this repo as currently built — confirmed via direct filesystem search —
  so there was nothing to reclassify into LFS/release-assets/external-docs
  right now. The `.gitignore` fixes above are preventive, for when scripts
  are actually run and these files get created locally.

**Flagged, not fixed (policy question, not mine to decide unilaterally):**
- `rna_protein_correspondence.csv` (inside the now-ignored `CITE-seq_PBMC_v2/`)
  and `data/tea-seq/protein_gene_conversion_new.csv` (inside the now-ignored
  `data/`) are small, human-readable reproducibility fixtures — the whole
  point of pinning the CITE-seq correspondence table was so it doesn't
  depend on a live mygene.info call ever again. Sweeping them into
  whole-directory `.gitignore` rules alongside multi-GB checkpoints means a
  fresh clone still can't reproduce the exact correspondence without either
  the raw data or a fresh (non-deterministic) mygene query — arguably
  defeating the purpose of pinning them in the first place. Committing
  these two specific small files (via a `.gitignore` negation pattern,
  same technique already used for `results/figures/.gitkeep`) is a
  reasonable fix, but it's a scope/reproducibility-policy call about what
  ships in the repo, not a pure cleanliness bug — flagging for project
  review rather than deciding it here.

**Not yet verifiable (blocked on git not being initialized, per instruction):**
- Whether `git submodule add https://github.com/gefeiwang/scMODAL scMODAL`
  + `git submodule update --init` actually produces a working submodule for
  a fresh clone can't be tested until git exists in this repo — README's
  current instruction (`git submodule update --init`) is incomplete on its
  own (it requires `.gitmodules` to already exist, which only happens after
  `git submodule add` is run once during initial setup). This needs to
  happen as an explicit first-commit-time step, not something checkable now.

---

### 2026-09-05 — Environment audit: scipy was missing, matplotlib was unnecessary in totalvi env

**Method:** every `import`/`from` statement across `src/*.py` and
`scripts/*.py` was extracted and cross-checked against both `.yml` files
package-by-package.

**Findings:**
- `scipy` was **missing** from `environment-scmodal.yml` — `src/calibration.py`
  imports `scipy.stats.norm`, used by `scripts/train_ua_flow.py`. This is a
  real gap: a fresh `scmodal` env built from the old file would fail on
  `train_ua_flow.py` with an `ImportError`. **Fixed** — added.
- `matplotlib` was listed in `environment-totalvi.yml` but is **not actually
  imported anywhere** in `scripts/train_totalvi.py` or `src/totalvi_baseline.py`
  (plotting for totalVI results happens in `make_figures.py`, which runs in
  the `scmodal` env). **Removed** as unnecessary — noted in the file that
  it's harmless to add back for ad hoc exploration, just not required.
- `scanpy`/`anndata`/`pandas` in `environment-totalvi.yml` are needed only
  **transitively** (via `scripts/train_totalvi.py` reusing
  `src/data_prep.py`'s `load_citeseq_raw()`), not because
  `src/totalvi_baseline.py` itself uses them. Documented explicitly so a
  future reader doesn't assume scvi-tools alone would need them.
- `scikit-learn` and `mygene` are correctly **absent** from
  `environment-totalvi.yml` — neither is imported by anything the totalVI
  script path touches.
- `scmodal` (the package itself) and `scvi-tools` remain correctly
  segregated to their respective environments — confirms the project's
  existing separation is followed correctly in code, not just in principle.

**Both files remain starting-point templates, not verified `conda env
export` output** — exact patch versions for `scanpy`, `pandas`, etc. are
not pinned (only `torch==1.13.1`, `scanpy==1.9.8`, and `scvi-tools==1.1.6.post2`
carry versions, taken directly from project notes). Do not treat these as
guaranteeing byte-identical reproducibility on a fresh machine; regenerate
via `conda env export` from the actual working environments before that
matters.

**Fresh-environment expectation:**
- `environment-scmodal.yml`, once built, covers every pip-installable
  dependency for every script except `train_totalvi.py`. It does **not**
  include scMODAL itself — that's a git submodule, not a pip package (see
  `README.md`), and must be separately cloned (`git submodule update --init`)
  before `import scmodal` succeeds anywhere.
- `environment-totalvi.yml`, once built, covers every dependency
  `train_totalvi.py` needs directly. No submodule or extra manual step
  required for this one.

---

### 2026-09-04 — Reproducibility audit continued: TEA-seq companion-file nesting bug found and fixed

**What was found:** `src/teaseq_prep.py`'s `train_teaseq_scmodal()` and
`resume_teaseq_scmodal()` saved/loaded `adt_scaled_gt.npy` and
`rna_scaled.npy` at the top-level `TEA-seq_PBMC/` path. This was internally
inconsistent with `scripts/compute_uncertainty.py`, which already (correctly)
expected these files at the *nested* `TEA-seq_PBMC/TEA-seq_PBMC/` path.

**Evidence trail (why nested is correct, not a guess):**
- Cell 70's own comment states outright: "writes ckpt.pth into
  `./TEA-seq_PBMC/TEA-seq_PBMC/` automatically" — scMODAL's tri-modal
  `integrate_datasets_feats` nests its own output one level deeper than
  `model_path` suggests. This is a known behavior, not a hypothesis.
- Cell 72 (the "insurance save" of `adt_scaled_gt.npy`/`rna_scaled.npy`)
  uses **top-level** paths — but this cell is in Section 2, which is
  entirely commented-out/frozen reference and was **never executed** in
  this saved notebook.
- Cell 172 (Section 7, SG3) — **live, executed, with verified-correct
  downstream results** (cell 173's printed shape check, cell 177's
  integration-uncertainty correlation of 0.622 matching project records) —
  loads these same two files from the **nested** path and succeeds.
- Cell 78 (Section 2's own VelocityNet training, also frozen/never executed
  in this notebook) independently corroborates this: after `os.chdir` into
  `~/scMODAL/TEA-seq_PBMC/`, it loads `'./TEA-seq_PBMC/adt_scaled_gt.npy'`
  — relative to that already-changed cwd, this resolves to the same nested
  location. Two independent cells (172, 78) agree on nested; only the
  never-executed cell 72 suggests top-level.

**Resolution:** since only the nested-path version has actual proof of
working (a live execution with correct, verified results), and two
independent cells corroborate it, `train_teaseq_scmodal()` now saves
`adt_scaled_gt.npy`, `rna_scaled.npy`, and `cell_counts.json` to the nested
directory, and `resume_teaseq_scmodal()` loads from there. This is a
**bug fix in the migration**, not a change to any historical result — the
files being fixed are ground-truth arrays and a bookkeeping cache, not
model weights or experimental outputs. `z_ADT_tea.npy`/`z_RNA_tea.npy`
remain at the top-level path (unaffected) — cell 76 confirms these were
always saved relative to a cwd of `~/scMODAL/TEA-seq_PBMC/` without an
extra nested segment, consistent with `compute_uncertainty.py`'s existing
(unchanged) `TEA_ROOT` usage.

**If this project's actual on-disk `TEA-seq_PBMC/` directory doesn't match
this nested convention** (e.g. if the person manually relocated files at
some point in a way this notebook doesn't capture), this fix would need
revisiting — flagging this explicitly since it's inferred from notebook
execution evidence, not verified against the actual filesystem (which
isn't available in this environment). Check against the real directory
before trusting this if anything seems off when running `load_raw_data.py
--dataset teaseq --resume`.

---

### 2026-09-04 — Reproducibility audit: exact totalVI vs. scMODAL feature-space breakdown

**Purpose:** the totalVI-vs-scMODAL comparability caveat was already flagged
(see the 2026-09-03 totalVI migration entry below), but this audit pass
traced the exact dimensions and transformations on each side, from the
notebook's own cells and printed outputs, so the caveat is precise rather
than general. No modeling changes were made — this is documentation only,
per explicit instruction not to invent a correction.

**totalVI side** (cells 94-110, run in the `totalvi` conda environment):
- Ground truth: `gt_protein_raw = adata_ADT.to_df().values`, raw ADT
  protein counts, shape **(18034, 228)** — confirmed directly from Section
  1 cell 5's printed output (`ADT shape: (18034, 228)`), the same raw
  panel CITE-seq starts from before any scMODAL-specific processing.
- Prediction: `posterior_predictive_sample()`, which returns **raw
  count-scale** samples (explicitly not the same scale as
  `get_normalized_expression`, per the notebook's own comment) — no
  normalization, no log transform, no scaling.
- Accuracy: per-protein Pearson correlation between the posterior mean and
  raw counts, averaged over 228 proteins → **0.664**.

**scMODAL/flow side** (Section 4 Part B, cells 122-124):
- The correspondence table maps RNA genes to ADT proteins as *pairs*, not
  a 1:1 protein split. The pinned table has **204 pairs but only 172
  unique proteins** (confirmed by cell 121's printed output: "204 pairs,
  172 unique proteins") — meaning 32 proteins are paired with more than
  one gene, and `var_names_make_unique()` gives each pairing its own
  (suffixed) column rather than collapsing them.
- `ADT_shared` ends up with **204 columns** (one per pair, so some
  proteins' values appear more than once), `ADT_unshared` (proteins with
  no gene match) has **56 columns**. `172 + 56 = 228` — matches the raw
  panel size exactly, confirming the arithmetic; but `204 + 56 = 260` is
  the actual column count of the matrix scMODAL trains and evaluates on
  (confirmed by cell 124's printed output: `adata2 shape: (18034, 260)`).
  So scMODAL's "260 features" are 260 *slots* representing 228 unique
  proteins, not 260 distinct biological targets.
- Both `ADT_shared` and `ADT_unshared` are `sc.pp.normalize_total` +
  `sc.pp.log1p`'d, then the concatenated `adata2` is `sc.pp.scale`'d
  (z-scored, clipped at `max_value=10`). This is **log1p-normalized,
  z-scored, clipped** space — nothing like totalVI's raw counts.
- Accuracy: per-protein-slot Pearson correlation between the deterministic
  decoder's prediction and this scaled ground truth, averaged over 260
  slots. **Important checkpoint distinction, not just a scale distinction:**
  the notebook's own Section 3 markdown (cell 111) states the comparison as
  "0.664 vs. your flow's 0.475 and deterministic scMODAL's **0.485**" — both
  0.485 and 0.475 are from the *original* checkpoint (`CITE-seq_PBMC/`,
  Phase 0-3), the one totalVI was actually trained and compared against.
  The **0.520** figure belongs to `model_v2` (the reproducibility-fixed
  retrain from Section 4), a *different* checkpoint that didn't exist yet
  when totalVI was run — it was never compared against totalVI in the
  original notebook at all. Citing 0.520 alongside totalVI's 0.664 would be
  pairing numbers from a comparison that was never actually run, which is a
  second, independent problem layered on top of the feature-space mismatch.
  The historically accurate comparison, if reported at all, is
  0.664 (totalVI) vs. 0.485 (original deterministic) vs. 0.475 (original flow).

**Conclusion, unchanged from the notebook's own original caveat:** even
setting aside which checkpoint's numbers get cited, these correlations are
computed over different feature counts (228 vs. 260 slots representing 228
proteins) and different value scales (raw counts vs. log1p-normalized-and-
scaled). A valid head-to-head number would require deciding (a) how to
collapse scMODAL's 260 slots back to 228 unique proteins for duplicated
ones, and (b) how to map both onto a common scale (inverse-transform
scMODAL's scaled predictions back toward count space, or normalize
totalVI's counts to match scMODAL's pipeline). Neither exists anywhere in
this project as a validated procedure — inventing one now would be exactly
the kind of new modeling assumption this audit was told not to make. The
caveat stands: **do not report "totalVI beats scMODAL/flow on accuracy"
without this correction being done first and reviewed by the project
supervisor. And if any comparison number is cited at all in the meantime,
it must be the historically accurate pairing (0.664 vs. 0.485/0.475, both
original-checkpoint
numbers) — not 0.520, which was never actually compared against totalVI.**

**Also checked:** the calibration comparison (totalVI underconfident,
flow/scMODAL overconfident) does **not** have this problem — coverage is a
dimensionless percentage (fraction of intervals containing ground truth),
so it's valid to compare directly regardless of the underlying feature
scale or count. `README.md`'s headline-results section was corrected (see
below) since it previously stated "totalVI beats both on raw accuracy" as
settled fact without this caveat.

---

### 2026-09-03 — totalVI baseline migrated; cross-scale caveat surfaced, not resolved

**Decision:** `src/totalvi_baseline.py` + `scripts/train_totalvi.py` cover
the full totalVI pipeline: AnnData construction, training/resume, chunked
posterior predictive sampling, count-scale accuracy, and calibration curve.
Closes Open Item #2 in flow.md.

**Important — did not fix, only surfaced clearly:** the original notebook's
own header already flagged this as an unresolved open item: totalVI's 0.664
correlation figure is computed in raw ADT count space over 228 features,
while scMODAL/flow's correlations (0.485 / 0.475) are in log-normalized
*scaled* space over 260 features. These are not a valid head-to-head
comparison as currently computed. I did not attempt to fix this by
re-deriving a scale-matched totalVI number — that requires a real modeling
decision (how to re-express totalVI's raw-count posterior into scMODAL's
scaled feature space) that isn't mine to make unilaterally, consistent with
the project's existing pattern of escalating scope/framing decisions to
supervisor review rather than resolving them silently. Instead, the caveat
is now: in the module docstring, in the script's printed output, in
flow.md as Open Item #6, and repeated here — so it cannot be quietly
dropped or missed before submission. Flag this in supervisor review before
treating "totalVI wins on accuracy" as a citable finding.

**Also migrated:** the flow-vs-totalVI calibration comparison plot, added
to `make_figures.py` as `plot_flow_vs_totalvi_calibration()`. Its flow-side
calibration numbers are still hardcoded constants (the original notebook
stitched these together manually across two conda environments that don't
share kernel state — this was never a live comparison, just documented as
one now instead of silently repeating the manual copy-paste).

---

### 2026-09-03 — Raw data loading migrated; TEA-seq split into its own module

**Decision:** `src/data_prep.py` gained `load_citeseq_raw()`, and a new
module `src/teaseq_prep.py` covers TEA-seq's raw loading, static-file
correspondence, tri-modal preprocessing, training, fast-resume, and latent
extraction. `scripts/load_raw_data.py` is the new single entry point for
both; `scripts/train_scmodal.py` now delegates to it instead of raising
`NotImplementedError`. Closes Open Item #1 in flow.md.

**Why TEA-seq got its own module instead of extending `data_prep.py`:**
the two datasets' preprocessing genuinely diverge — TEA-seq is a
three-modality integration (RNA/ADT/ATAC) with a static correspondence file
and a PCA-based RNA<->ATAC anchor, versus CITE-seq's two-modality, live-API
(now pinned) correspondence. Cramming both into one file would mean a lot
of `if dataset == "teaseq"` branching inside otherwise-simple functions.
Separate modules keep each one readable on its own.

**Anonymization fix applied here:** the original Section 1 setup cell had
an absolute path with the author's home-directory username hardcoded into
both a `sys.path.insert(...)` call and an `os.chdir(...)` call.
`load_citeseq_raw()` takes a relative `data_dir` argument instead and does
not hardcode any path tied to a specific machine or person. Same treatment
applied to TEA-seq's loader. (Note for the audit trail: the same literal
string was also found, during the later cleanliness scan, inside the
`notebooks/` copy of the source notebook itself and inside this very log
entry -- both redacted; see the 2026-09-05 cleanliness-scan entry above.)

**New known gap surfaced by this pass:** TEA-seq's fast-resume path needs
raw AnnData shapes (`n_ADT`, `n_RNA`, `n_ATAC`) to split `model.latent`
correctly, which means loading raw files even when you only want to resume
from a checkpoint. Tracked as Open Item #5 in flow.md rather than quietly
worked around, since the "right" fix (cache the three integers during the
first `--train` run) is a small enough change that it shouldn't be done
silently inside this pass without being called out.

---

### 2026-09-03 — Only the corrected SG4 code is included, not the original

**Decision:** `src/finetune.py` contains only `run_sg4_finetune` (renamed from
the notebook's `run_sg4_finetune_v2`), the baseline-corrected version. The
notebook's original, uncorrected `run_sg4_finetune` (which measured deltas
against the raw un-fine-tuned `model_v2` checkpoint rather than a
seed-matched `floor=1.0` control) is **not** ported into this repo at all —
per explicit instruction, no previous/superseded versions are kept.

**Why it matters:** fine-tuning itself (with `floor=1.0`, i.e. no
down-weighting) shifts accuracy by about -0.66% relative to the raw
checkpoint. Without baseline correction, some of the apparent "harm" from
down-weighting was actually just the effect of fine-tuning at all.
Supervisor review requested this correction before treating SG4 as
publishable.

**Consequence:** `make_figures.py`'s two SG4 plots use the corrected numbers
(hardcoded from the corrected run's actual printed output, since the
notebook never regenerated those two plots after switching to the corrected
function — see Open Item #4 in flow.md). Anyone re-running `finetune_sg4.py`
end to end should replace those hardcoded constants with the live output.

---

### 2026-09-03 — SG3 uses weight-noise MC, not MC-dropout

**Decision:** `src/uncertainty.py`'s `mc_noise_forward` — small Gaussian
perturbations injected into the encoder's linear-layer weights
(`W_1, b_1, W_2, b_2`), run `n_samples` times with weights restored after
each pass — is the only SG3 method included.

**Why:** the original SG3 plan was MC-dropout. scMODAL's encoder
(`scmodal.networks.encoder`) contains zero `nn.Dropout` layers (confirmed by
enumerating `model_v2.E_A.modules()`), so MC-dropout would silently produce
no variation at all rather than erroring. Weight-noise perturbation was the
working alternative, with `noise_scale=0.20` chosen via a sweep over
`[0.01, 0.05, 0.1, 0.2, 0.3]`, picking the value that gave a reasonable
MC-std-to-latent-magnitude ratio (not so small it's dominated by float
noise, not so large it swamps real latent structure).

**Consequence:** the exploratory notebook cells that searched `globals()`
for candidate arrays and manually verified this (interactive debugging, not
real logic) were not ported — only the method that worked.

---

### 2026-09-03 — Data flagged as a gap: TEA-seq `x_tea_tensor`

**Decision:** documented rather than silently fixed. See flow.md Open
Item #3. `adata2_X` is used as a stand-in based on matching input
dimensionality, but this hasn't been confirmed against whatever actually
produced the original TEA-seq SG3 result.

**Why this is called out explicitly:** silently substituting a plausible
variable name for an undefined one is exactly the kind of thing that quietly
corrupts a result without anyone noticing. Flagging it here means it gets
checked before anyone treats the TEA-seq integration-uncertainty numbers as
trustworthy.

---

### 2026-09-03 — Original (base) VelocityNet included despite "final only" instruction

**Decision:** `src/velocitynet.py` includes both `VelocityNet` (the base
flow, from Section 1) and `HeteroscedasticVelocityNet` (UA-Flow, Section 6),
even though Section 1 is otherwise treated as frozen/reference-only and the
instruction was "final corrected versions only."

**Why:** the base `VelocityNet` isn't a superseded draft of anything — it's
the core flow-matching model the entire project's headline result rests on
(deterministic decoder = accurate point predictions; base flow = ~1
correlation point lower, in exchange for calibrated uncertainty). It's
foundational, not a previous iteration of SG2's UA-Flow. Treated as a
different case from the old, uncorrected SG4 function, which genuinely was
superseded by a later, corrected version.

---

### 2026-09-03 — Repo scaffold started; notebook split into `src/` + `scripts/`

**Decision:** moved from a single 193-cell notebook to a `src/` (reusable
logic: data prep, models, uncertainty, calibration, fine-tuning) +
`scripts/` (CLI entry points, one per pipeline stage) structure, matching
conference-submission norms (runnable code, not a frozen notebook — see
prior discussion on why frozen/commented sections don't fit a publication
context, especially one that may need double-blind anonymization).

**What's still notebook-only:** raw data loading (Sections 1-2) and the
totalVI baseline (Section 3) — see flow.md Open Items #1-2. These weren't
migrated in this pass because they weren't the "final corrected" stretch-goal
code this pass focused on; they're existing, working reference code that
still needs converting, tracked as follow-up work rather than silently
dropped.

**Anonymization note:** repo built anonymize-ready from the start per your
call that the submission venue isn't decided yet — no names, no
identifying paths, generic environment/config files. See prior conversation
for the anonymous.4open.science recommendation when it's time to submit.

---

### 2026-09-23 — TEA-seq correspondence-CSV path bug fixed on the actual working repo

**What was fixed:** `.gitignore`, `scripts/load_raw_data.py`, and
`src/teaseq_prep.py` were corrected to use `TEA-seq_PBMC/data/tea-seq/...`
instead of a top-level `data/tea-seq/...` for the TEA-seq raw `.h5ad` files
and the `protein_gene_conversion_new.csv` fixture. Root cause: the original
notebook's Section 2 changes working directory into `TEA-seq_PBMC/` before
loading these files with a relative path -- the same "nested one level
deeper" pattern already documented for TEA-seq checkpoints/`.npy` files,
just missed for this specific file during the original migration.

**Confirmed against the real filesystem, not assumed:**
`find ~/scMODAL -iname "protein_gene_conversion*"` and a follow-up `find`
for `RNA.h5ad`/`ATAC.h5ad`/`ADT.h5ad` both confirmed the nested location
directly.

**Process note, worth recording:** this fix was originally made and
verified in an ephemeral sandbox, but the corrected files were never
actually delivered to the real working repo (already transferred via git
bundle by that point) -- the sandbox later reset, and the fix was
effectively lost until re-diagnosed from `git check-ignore -v` output
against the live repo and reconstructed from scratch. Lesson: once a repo
has been handed off to its real, persistent location, further fixes need
to be applied and verified directly against that location, not just in a
disconnected working copy.
