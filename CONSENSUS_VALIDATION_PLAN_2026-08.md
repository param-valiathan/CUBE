# Consensus-Clustering Validation Plan (August 2026)

Status: **RUN 2026-08-12.** See Changelog below for results.

## Why this document exists

`consensus_cluster()` (`cube_core.py:4096`) was originally added as an
opt-in mitigation for seed-unstable UMAP+HDBSCAN partitions, then had its
auto-trigger (`consensus_auto_threshold`) disabled by default (`0.6 → 0`)
on the theory that `hdbscan_selection_mode="floor_soft_cap"` had already
fixed the instability that motivated it. A real 4-config A/B experiment on
the 21-session combined dataset (baseline / down-weighted bodyparts /
`umap_n_neighbors=60` / both) then showed seed-sweep mean ARI still sitting
at 0.18–0.29 across every config — nowhere close to stable — so the
auto-trigger was re-enabled at `consensus_auto_threshold=0.55`.

Separately, `consensus_cluster()` gained two new capabilities this session:
a post-hoc split+merge refinement pass (`consensus_refine_enabled`, reusing
`split_impure_clusters()` + a new co-association-based
`merge_by_coassociation()`), and feature-space DBCV/silhouette scoring
(`_dbcv_feature_space()` + `validate_clustering()` on `feats_sc_T` instead
of a UMAP embedding) — the first metric that's actually comparable between
the primary single-seed path and consensus, since `separation_ratio` and
embedding-space DBCV are not on the same scale.

**None of this has been tested against real data yet.** Re-enabling the
auto-trigger was a judgment call based on the *problem* (unstable ARI)
being real, not on evidence that consensus is actually the right *fix* for
this specific dataset, or that the new refinement pass improves on plain
consensus. This plan specifies exactly that test.

## Question

On the real 3-folder, 21-session dataset, does consensus clustering
(plain, and with post-hoc refinement) produce a **better final partition**
than the primary single-seed HDBSCAN fit — and if so, by how much? "Better"
is judged on the one scale now shared by both paths
(`dbcv_feature_space`, `silhouette_feature_space`), plus practical
secondary signals (noise %, cluster count, MLP CV accuracy, runtime cost).

## Design

### Configs

All three run on the same dataset with the same base cfg, differing only in
the consensus-related keys:

| config | consensus_clustering_enabled | consensus_refine_enabled | consensus_merge_coassoc_thresh |
|---|---|---|---|
| `primary_only` | `False` | n/a | n/a |
| `consensus_plain` | `True` | `False` | n/a |
| `consensus_refined` | `True` | `True` | `0.5` (current default) |

`consensus_auto_threshold=0` is set explicitly in **every** config
(including the consensus ones) so each run's path is forced deterministically
by `consensus_clustering_enabled` alone — the experiment must not depend on
whether that run's own seed-sweep ARI happens to cross the auto-trigger
bar, which would make `primary_only` unreliable as a forced baseline.

`seed_sweep_n=0` (disabled) for all three — the seed-sweep/ARI instability
question was already answered by the prior 4-config experiment; re-running
it here would cost ~5 extra minutes per config for a diagnostic this
experiment doesn't need. `consensus_n_seeds=8` (default) throughout, not
swept — that's a separate question from "does consensus help at all."

Base cfg mirrors the ARI experiment harness precedent (this session):
`hmm_enabled=False, save_plots=False, save_videos=False` (metrics-only,
faster runs — DBCV/silhouette/CV-accuracy are all computed before HMM/plot/
video stages).

### Metrics captured per config

- `n_clusters` (final, post-refinement/pruning)
- `noise_pct`
- `dbcv_feature_space`, `silhouette_feature_space` (the shared scale)
- `cv_accuracy` (MLP cross-val — classifier-learnability signal; the
  previous ad-hoc consensus run on this dataset showed this dropping
  0.967→0.841, worth re-checking now that refinement exists)
- `runtime_s`
- Consensus configs only: `separation_ratio`, `per_seed_counts` (own
  per-seed diagnostic, not comparable across configs, reported for context)

### Analysis

For each of `consensus_plain` and `consensus_refined`, compute deltas
against `primary_only`:

```
Δ dbcv_feature_space        = consensus.dbcv_feature_space - primary.dbcv_feature_space
Δ silhouette_feature_space  = consensus.silhouette_feature_space - primary.silhouette_feature_space
Δ noise_pct                 = consensus.noise_pct - primary.noise_pct
Δ cv_accuracy                = consensus.cv_accuracy - primary.cv_accuracy
runtime_multiplier          = consensus.runtime_s / primary.runtime_s
```

**Decision rule** (report the number either way — this is a guide for the
write-up, not a hard gate): consensus (a given variant) is judged to
genuinely help if **both** `Δ dbcv_feature_space` and
`Δ silhouette_feature_space` are positive by a non-trivial margin (not
just noise — there is no repeated-seed variance estimate in this first
pass, see "Known limitation" below) **and** `Δ cv_accuracy` is not a severe
regression (say, worse than -0.05 absolute). If the two quality metrics
disagree with each other, or quality improves but CV accuracy collapses,
report as **inconclusive**, not a clean win — do not force a verdict past
what the numbers actually show.

### Known limitation — single run per config

This first pass runs each config **once** (one `umap_random_state=42`
base seed). Given the dataset's demonstrated seed-to-seed instability
(mean ARI 0.18–0.29 in the prior experiment), a single comparison could
easily reflect which seed each config happened to land on rather than a
genuine consensus-vs-primary effect. This plan deliberately scopes to a
fast first pass; if the result looks like a clear win or a clear loss,
that alone is useful signal, but a **follow-up with 2-3 different base
seeds per config** (varying `umap_random_state`) is the honest next step
before treating any conclusion here as final — flag this explicitly in the
write-up rather than silently presenting a single run as definitive.

## Environment — mandatory

- All Python execution uses `C:\Users\param\anaconda3\envs\CUBE\python.exe`
  — never bare `python`.
- Dataset: the 3 `BSOID_Project_Ready\csv` folders already used by this
  session's prior ARI experiment:
  `D:\Damien DLC\20260407_Baseline_Exp1\BSOID_Project_Ready\csv`,
  `D:\Damien DLC\20260408_CLZ_Exp1\BSOID_Project_Ready\csv`,
  `D:\Damien DLC\20260409_DCZ_Exp1\BSOID_Project_Ready\csv`.
- All script/output files go in the scratchpad/temp directory — never into
  the CUBE repo, never overwriting real project data.
- This is a **read-only experiment against `cube_core.py`** — no source
  code should be modified to run it. If running the experiment reveals a
  bug in `consensus_cluster()`/`refine_consensus_clusters()`/
  `_dbcv_feature_space()`, stop and report it rather than silently patching
  around it.

## Definition of done

- All 3 configs run to completion (or a reported, explained failure) on the
  real dataset, each producing the metrics listed above.
- A results table (all 3 configs, all metrics, plus the two delta rows) is
  produced and reported in the chat response.
- The decision-rule verdict (help / no help / inconclusive) is stated
  explicitly for both `consensus_plain` and `consensus_refined`, separately.
- The single-run limitation is disclosed alongside the verdict, not
  omitted.
- This document's Changelog section (below) is filled in with the actual
  results and verdict — do not leave this plan file in its pre-run state
  once the experiment has been executed.

## Changelog

### 2026-08-12 — First pass executed (single run per config)

Ran on the real 3-folder / 21-session dataset (28,235 total bins, 7,291-bin
UMAP training sample), `umap_random_state=42` throughout, via a standalone
harness that imports `BSoidEngine` directly (no `cube_core.py` changes).

| config | n_clusters | noise_pct | dbcv_feature_space | silhouette_feature_space | cv_accuracy | runtime_s |
|---|---|---|---|---|---|---|
| `primary_only` | 34 | 34.6% | NaN | 0.0414 | 0.9668 | 145.3 |
| `consensus_plain` | 30 | 0.0% | NaN | −0.0602 | 0.7973 | 468.9 |
| `consensus_refined` | 30 | 0.0% | NaN | −0.0602 | 0.7973 | 479.2 |

Deltas vs. `primary_only`:

| | Δ dbcv_feature_space | Δ silhouette_feature_space | Δ noise_pct | Δ cv_accuracy | runtime_multiplier |
|---|---|---|---|---|---|
| `consensus_plain − primary_only` | NaN (uninterpretable, see below) | −0.1016 | −34.6 pp | −0.1696 | 3.23x |
| `consensus_refined − primary_only` | NaN (uninterpretable, see below) | −0.1016 | −34.6 pp | −0.1696 | 3.30x |

Consensus-only diagnostics (context, not comparable across configs):
`separation_ratio=7.61x`, mean pairwise ARI across the 8 seeds = 0.390
(per-seed cluster counts `[34, 108, 58, 16, 61, 56, 54, 61]` — confirms the
seed-instability this whole investigation is about). `consensus_refined`'s
split/merge pass ran (log: `[consensus-refine] iteration 1: no changes —
converged`) and made zero changes to this particular partition — the
identical numbers vs. `consensus_plain` are a genuine no-op result, not an
extraction bug.

**Bug found and reported, not patched (per this plan's scope):**
`dbcv_feature_space` came back NaN in **all three** configs, making it
uninterpretable as evidence either way. Root cause: `_dbcv_feature_space()`
calls `hdbscan.validity.validity_index(X, labels, metric="euclidean")` on
the raw 589-dimensional standardized feature space without passing
hdbscan's `d` (dimensionality) parameter. `hdbscan.validity` then defaults
`d = X.shape[1] = 589`. Its internal all-points-core-distance step computes
`(1/distance)**d` for every within-cluster pairwise distance; at `d=589`
this overflows float64 for any pair closer than ~0.3 in this standardized
space (confirmed numerically: `(1/0.3)**589 ≈ 9.45e307`, right at the
float64 ceiling; anything closer overflows to `inf`). That `inf`
propagates through core-distance → mutual-reachability → MST →
density-sparseness, and the final per-cluster
`(min_density_sep − density_sparseness) / max(...)` computes `inf − inf`
→ NaN. Instrumented confirmation on `primary_only`: 21 of 34 clusters
returned NaN validity, the rest finite (−0.1 to −0.9); the overall score
is a size-weighted average, so one NaN cluster poisons the whole result.
This is a structural mismatch, not a data-quality artifact of this
dataset: CUBE's other internal DBCV usage (`run_hdbscan()`'s
`relative_validity_`) scores the 3D UMAP embedding (`d=3`, numerically
stable); `_dbcv_feature_space()` is the new Aug 2026 addition that
deliberately scores the raw feature space instead, but never passes a
small `d` to compensate, so it will be NaN-prone at this feature count
essentially independent of dataset. Not patched here — flagged for a
follow-up fix (e.g. passing a low `d`, or running DBCV on a PCA-reduced
low-dimensional projection of the feature space instead of the raw
589-dim vectors).

**Verdict — `consensus_plain`: NO HELP.** `dbcv_feature_space` is
uninterpretable (NaN in every config, see bug above), so the verdict rests
on `silhouette_feature_space` (regressed, +0.0414 → −0.0602) and
`cv_accuracy` (collapsed 0.967 → 0.797, a −0.170 absolute drop, far past
the −0.05 severe-regression bar in the decision rule). Noise_pct did drop
to 0% and separation_ratio (7.61x) looks strong, but those are not the
metrics the decision rule is keyed on, and cv_accuracy's collapse plus the
mean pairwise ARI of only 0.390 across consensus's own 8 seeds suggest the
consensus partition itself remains seed-sensitive on this dataset, not
that it resolved the instability into a cleanly learnable partition. Cost
was also ~3.2x the primary path's runtime for this regression.

**Verdict — `consensus_refined`: NO HELP.** Identical reasoning and
identical metrics to `consensus_plain` — the refinement pass converged
with zero changes on this run, so it neither helped nor hurt beyond what
plain consensus already did, at marginally higher cost (3.30x vs. 3.23x
runtime).

**Single-run limitation (per plan's "Known limitation" section):** both
verdicts come from **one run each** (`umap_random_state=42` base seed).
Given this dataset's documented seed-to-seed instability (mean ARI
0.18–0.39 across every experiment run so far, including consensus's own
8-seed sweep here), a single comparison could reflect which seed each
config happened to land on rather than a reproducible consensus-vs-primary
effect. The verdicts above are clear enough on the metrics that are
actually interpretable (silhouette, cv_accuracy) to be useful signal as a
first pass, but the 2-3-seed follow-up described in "Known limitation"
is the honest next step before treating "no help" as final — and any
follow-up should first address the `dbcv_feature_space` NaN bug, since
without it the experiment is running one metric short of what the design
called for.

**Deviations from the original plan:** none in configuration — all three
configs ran to completion with no crashes. The one deviation is evidentiary:
`dbcv_feature_space`, the metric the plan doc calls "the one scale now
shared by both paths" and centers the decision rule on, was NaN throughout
and could not be used as designed; the verdicts above lean on the
remaining metrics instead, as described.
