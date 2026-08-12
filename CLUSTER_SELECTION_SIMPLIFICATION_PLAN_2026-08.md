# Cluster-Selection Simplification & Simplicity-Audit Plan (August 2026)

Status: **IMPLEMENTED (Aug 2026).** See the Changelog section below for
what was built, the real-data evidence, and the promotion decision.

## Why this document exists

A prior ARI-stability investigation (Aug 2026, docs since archived —
findings retained here) implemented and real-data-tested four levers
(`umap_init`, `umap_n_epochs`, `seed_sweep_n_jobs`, a
`seed_sweep_stability_bootstrap()` diagnostic). None of them meaningfully
fixed seed-sweep ARI (still ~0.41), and the diagnostic's own go/no-go gate
said Tier 2 (decoupling clustering from UMAP) is **not** justified —
subsample-composition noise turned out to matter *more* than UMAP's own
optimization stochasticity, not less. `umap_init` and `umap_n_epochs` have
since been fully removed from the codebase (real-data evidence showed no
genuine benefit — see the audit table below); `seed_sweep_n_jobs` and
`seed_sweep_stability_bootstrap()` were kept.

Reading `run_hdbscan()` (`cube_core.py:715-939`) directly, in light of that
result, exposes the actual mechanism: **auto-mode cluster-count selection is
a two-branch, discontinuous decision, not a smooth one.** The baseline
seed-only sweep's cluster counts — `34, 30, 3, 32, 33, 29, 3, 30` — are not
noisy variation around one number. They are two qualitatively different
outcomes: six seeds land near 30, two seeds collapse to 3. That is exactly
what the code does when no swept candidate happens to fall inside
`[preferred_clusters_lo, preferred_clusters_hi]` (default 8–30): it silently
switches to a structurally different selection rule (closest-to-boundary
among candidates passing a DBCV quality gate) with no continuity guarantee
to the in-range branch. A small embedding perturbation — exactly the kind
`umap_init`/`umap_n_epochs` were trying to suppress — can flip which branch
fires, and the two branches disagree by an order of magnitude in cluster
count. No downstream embedding-stability lever can fix a discontinuity in
the *selection* logic; that logic has to change directly.

This also reframes what the earlier `umap_init="pca"` result actually was.
It raised mean ARI 0.41→0.60, which looked like a win, but collapsed most
seeds to 4–5 clusters. That is not stability — it is the fallback branch
firing consistently instead of occasionally, converging everyone onto the
same *degenerate* coarse solution. High agreement between seven collapsed
solutions is a worse outcome than the status quo, not a better one.

## The actual requirement (user-specified, supersedes literal-ARI framing)

1. Fewer than `preferred_clusters_lo` (**default 8**) clusters is
   **unacceptable** — a hard floor, not a soft preference.
2. Up to roughly `preferred_clusters_hi` (**default 30**) is the target
   ceiling, but overshoot is fine — extra clusters get merged manually by
   the user afterward based on visually-inspected biological behavior. This
   is not symmetric: undershoot is a failure, overshoot is a minor
   inconvenience.
3. Both bounds must remain user-editable (they already are:
   `preferred_clusters_lo`/`preferred_clusters_hi` cfg keys, already exposed
   in `AdvancedCUBEWindow`'s "CLUSTER COUNT GUIDANCE" section,
   `cube.py:3536-3559`). No change needed there except making the floor
   behavior actually match its billing.
4. Literal partition-agreement ARI is **not** the target metric on its own.
   "29 clusters this seed, 34 that seed" is an acceptable outcome if the
   *clusters themselves* are broadly the same behaviors, just split slightly
   differently — this needs a **per-cluster** view (which specific clusters
   are unstable/changing), not just a single aggregate number.
5. A hierarchical relationship view across the clusters actually produced,
   to support the user's manual-merge workflow visually.

Requirement 4 and 5 are new deliverables, not present in the codebase today
(`plot_cluster_stability()`, `cube_core.py:2439`, only plots an aggregate
cluster-count histogram + seed×seed ARI heatmap — no per-cluster
breakdown, and there is no hierarchy/dendrogram plot anywhere in
`cube_core.py`'s ~25 plot functions).

---

## What's actually necessary in `run_hdbscan()` — audit

Going through every piece of `run_hdbscan()`'s current complexity and
classifying it:

| Component | Verdict | Reasoning |
|---|---|---|
| 40-step `min_cluster_size` sweep × {eom, leaf} methods | **Keep** | This is the source of candidate diversity the range-based selection needs. Removing it would remove the ability to find an in-range solution at all. |
| `hdbscan_mcs_anchor` (`embedding` vs `full`) | **Keep, out of scope** | Orthogonal concern (subsample-vs-full-count normalization), not part of the discontinuity. |
| Duplicate-coordinate jitter (`1e-4 × std`) | **Keep** | Fixes a real DBCV degeneracy (divide-by-zero), unrelated to selection logic. |
| Degenerate-DBCV → silhouette fallback | **Keep** | Legitimate fallback for a different failure mode (non-finite DBCV), not the undershoot problem. |
| `hdbscan_diversity_bonus` (cluster-size CV reward) | **Keep, reuse** | Biologically motivated (rewards brief-event + sustained-behavior mixes over uniform blobs); folds cleanly into a single unified score. |
| `hdbscan_dbcv_thresh` (0.65 gate) | **Keep, repurpose** | Currently only gates the boundary-fallback branch. In the new design it becomes a single quality floor applied once, plus a `[WARN]` log if the eventual selection falls below it — not a second selection branch. |
| **The in-range/boundary-fallback branch split itself** | **Replace** | This is the actual discontinuity. Becomes one unified, continuous ranking pass (see below). |
| `umap_init` (T1.2) | **Removed — done** | Real-data-tested; no genuine stability benefit, and the one large ARI movement it produced was a collapse artifact, not real. Never committed/shipped, so removal was zero-cost. Already deleted from `cube_core.py`/`cube.py` (cfg key, kwarg, `DEFAULTS`, GUI row) prior to this implementation pass. |
| `umap_n_epochs` (T1.3) | **Removed — done** | Real-data-tested; `500` was a no-op, `1000` was worse and slower. No use case it serves once the actual root cause is elsewhere. Already deleted alongside `umap_init`. |

Everything not listed above (MLP training, HMM smoothing, export, the four
already-existing UMAP/HDBSCAN plots, etc.) is untouched — this is a
surgical change to one function's selection logic, plus two plot additions,
plus removal of two recently-added, empirically-unhelpful cfg keys.
`seed_sweep_n_jobs` (T1.P) is **kept** — it's a proven, bit-identical,
5–6x speedup with real value: once seeds are cheap, running a larger sweep
and inspecting per-cluster volatility (item 4 below) becomes practical.

---

## Design 1 — Unified, continuous cluster-count selection

New cfg key: **`hdbscan_selection_mode`** — `"legacy"` (default, today's
exact two-branch behavior, byte-identical) | `"floor_soft_cap"` (new).
Mirrors the existing `hdbscan_mcs_anchor`-style in-function branch pattern
already used in this file — no new function needed, `run_hdbscan()` gains
one more `if` on an existing cfg-driven fork.

**`"floor_soft_cap"` algorithm** (auto mode only; `target_n_clusters > 0`
user-guided mode is untouched — it already has a single, non-branching
selection rule and isn't part of the observed instability):

```
candidates              # unchanged: full sweep output, (score, n_cl, labels, clf)
best_dbcv = max(score for score, *_ in candidates)

# Hard floor — requirement 1. Never select below pref_lo if it's avoidable.
floor_ok = [c for c in candidates if c[1] >= pref_lo]

if not floor_ok:
    # Genuinely impossible: nothing in the whole sweep reached pref_lo.
    # Extremely rare (indicates too little data/structure) — log loudly,
    # pick the candidate with the highest cluster count available.
    log_fn("[WARN] no sweep candidate reached preferred_clusters_lo=<lo>; "
           "dataset may lack enough structure/data. Selecting the closest "
           "available (<n_cl> clusters) — inspect this session's output "
           "with extra care.")
    chosen = max(candidates, key=lambda c: c[1])
else:
    # Soft ceiling — requirement 2. Continuous penalty, not a branch.
    # overshoot_penalty=0 fully disables the ceiling (unbounded overshoot ok).
    overshoot_w = float(cfg.get("hdbscan_overshoot_penalty", 0.01))
    def unified_score(c):
        score, n_cl, labels, _clf = c
        div    = _div_bonus * _cluster_cv(labels)
        over   = overshoot_w * max(0, n_cl - pref_hi)
        return score + div - over
    chosen = max(floor_ok, key=unified_score)

best_score, _, best_labels, best_clf = chosen
if best_score < _dbcv_thresh * best_dbcv:
    log_fn(f"[WARN] selected solution's {_score_label}={best_score:.3f} is "
           f"below {_dbcv_thresh:.0%} of the sweep's best ({best_dbcv:.3f}); "
           f"cluster quality may be weak despite satisfying the count floor.")
```

Why this removes the discontinuity: there is exactly **one** ranking pass
now. A seed whose sweep just barely fails to produce an in-range candidate
no longer jumps to a structurally different rule — it's still ranked by the
same `unified_score` among everything that clears the floor, and the
penalty for being a bit over `pref_hi` is small and linear, not a regime
change. The floor is genuinely hard (matches "unacceptable," not "usually
avoided"), the ceiling is genuinely soft (matches "roughly similar is
fine"), and the two are asymmetric on purpose, matching the actual
requirement instead of a symmetric closest-to-either-boundary rule.

`hdbscan_overshoot_penalty` default `0.01`: calibrated so that a solution
10 clusters over `pref_hi` needs roughly a 0.1-DBCV-unit-equivalent
improvement to be preferred over a smaller in-range one — gentle enough
that real quality differences still win, strong enough to avoid runaway
over-fragmentation when two candidates are near-tied on DBCV. This default
is a starting point, not empirically validated yet — the implementation
prompt's test plan includes tuning it against real data before any
promotion to being the shipped default (see "Safety pattern" below).

**Because `seed_sweep_stability()` and `seed_sweep_stability_bootstrap()`
both call `run_hdbscan()` internally** (`cube_core.py:2245` and `:2404`),
this is a single point of leverage: fixing the primary clustering call
automatically fixes the seed-sweep diagnostic's undershoot problem too. The
"run more seeds, discard the ones that collapse" mitigation discussed
earlier becomes a secondary safety net for residual noise (the 29-vs-34
scale of variation), not the primary fix — the catastrophic 3-cluster
outcomes should mostly stop happening structurally once the floor is a hard
rule rather than a coin-flip branch.

---

## Design 2 — Per-cluster volatility plot ("which clusters are most changing")

New function `plot_cluster_volatility(sweep, out_path)` in `cube_core.py`,
alongside (not replacing) `plot_cluster_stability()`. Requires no new cfg
key — runs whenever `seed_sweep_n >= 2` already does (same gate,
`cube_core.py:4971-4995`), called immediately after the existing
`plot_cluster_stability()` call.

**Method:** `seed_sweep_stability()` keeps the same bin subsample across all
seeds (only `umap_random_state` varies) — so every seed's label array is
already index-aligned to the same underlying bins. This makes cross-seed
cluster matching straightforward, no intersection bookkeeping needed (that
complexity is specific to the *bootstrap* diagnostic, which does vary the
subsample — this plot only needs the plain sweep).

1. Take seed 0's partition as the reference labeling.
2. For every other seed `s`, build a cost matrix between reference clusters
   and seed `s`'s clusters using pairwise Jaccard overlap (intersection /
   union of bin-index sets, restricted to non-noise points), and solve
   optimal reference→seed cluster matching via
   `scipy.optimize.linear_sum_assignment` (Hungarian algorithm) on
   `1 - Jaccard`.
3. For each reference cluster, record its best-match Jaccard score against
   every other seed (0 if it has no viable match, e.g. was absorbed/split
   entirely).
4. Per-cluster **volatility score** = `1 - mean(best-match Jaccard across
   seeds)`. High volatility = this specific behavior's boundary keeps
   shifting seed to seed; low volatility = this cluster is essentially the
   same set of bins no matter the seed.
5. Plot: horizontal bar chart, one bar per reference cluster, sorted by
   volatility descending (most-changing clusters at the top, immediately
   visible), bar length = volatility score, annotated with reference
   cluster size (n bins / % of session). Uses the existing theme globals
   (`_BG`, `_PANEL`, `_TEXT_COL`, `_TICK_COL`) and `_dark_ax`/`_savefig`
   helpers — never hardcoded colors, per this codebase's existing plot-theme
   convention.

Output: `plots/cluster_volatility.png`. This directly answers "are the 29
vs 34 cluster counts because of one unstable cluster splitting differently,
or because everything is shifting a little" — a per-cluster diagnostic the
current aggregate ARI heatmap cannot provide.

---

## Design 3 — Cluster hierarchy (dendrogram) plot

New function `plot_cluster_hierarchy(feats_sc, labels, out_path,
bodypart_names=None)` in `cube_core.py`. Runs on the **primary/production**
clustering result (final `hdb_labels` after rare-cluster pruning,
`cube_core.py:~4930`), not a seed-sweep diagnostic — this is a per-run
production artifact meant to directly support the user's manual-merge
workflow, so it should exist every time clustering runs, not just when a
seed sweep was requested.

**Method:**
1. Compute each non-noise cluster's centroid in the standardized feature
   space (`feats_sc`, the same space HDBSCAN operates in via the UMAP
   embedding, but expressed in original scaled-feature units rather than
   embedding coordinates) — using the feature space rather than the 2-3D
   UMAP embedding for the hierarchy is a deliberate choice: UMAP embedding
   distances are only locally meaningful (a known UMAP limitation for
   between-cluster distance interpretation), while feature-space centroid
   distance is directly interpretable as "how different is the underlying
   movement/pose pattern," which is what the user is actually trying to
   judge when deciding whether two clusters represent the same behavior.
2. `scipy.cluster.hierarchy.linkage(centroids, method="ward")`.
3. `scipy.cluster.hierarchy.dendrogram(...)`, labels = cluster ID
   (optionally annotated with size), theme-aware colors matching the rest
   of the plot suite.
4. Save to `plots/cluster_hierarchy.png`.

New cfg key: **`cluster_hierarchy_enabled`** (default `True`) and
**`cluster_hierarchy_linkage`** (default `"ward"`, exposed as a combo for
`"ward"|"average"|"complete"`). Unlike Design 1, this does **not** need the
"default = today's exact behavior" opt-in gate — it produces a new output
file and touches no existing numeric result (label assignments, model
weights, existing CSVs/plots are all untouched), so it's safe to ship
enabled by default per this codebase's own established distinction between
"changes existing behavior" (opt-in, gated, validated before promotion —
Design 1) and "adds a new, independent artifact" (safe to default on
immediately — the pattern already used for `hmm_enabled`-adjacent additive
outputs).

---

## Summary of changes

| Item | Type | Default |
|---|---|---|
| `hdbscan_selection_mode` | New cfg key | `"legacy"` (byte-identical); `"floor_soft_cap"` opt-in until validated |
| `hdbscan_overshoot_penalty` | New cfg key | `0.01`, only active under `"floor_soft_cap"` |
| `umap_init` | **Removed — already done, prior to this pass** (cfg key, `run_umap()` kwarg, GUI combo row + help text) | n/a |
| `umap_n_epochs` | **Removed — already done, prior to this pass** (cfg key, `run_umap()` kwarg, GUI spinbox row + help text) | n/a |
| `plot_cluster_volatility()` | New function + plot output | Runs whenever `seed_sweep_n >= 2` (same gate as existing `plot_cluster_stability`) |
| `plot_cluster_hierarchy()` | New function + plot output | `cluster_hierarchy_enabled=True` (default on — additive only) |
| `cluster_hierarchy_linkage` | New cfg key | `"ward"` |

`preferred_clusters_lo`/`preferred_clusters_hi` — **no key change**, already
exist and are already user-editable in the GUI; only their enforcement
semantics change (soft preference → hard floor / soft ceiling) under the
new `hdbscan_selection_mode="floor_soft_cap"`.

## Explicitly out of scope for this pass

- Tier 2 (decoupling clustering from UMAP) — separately gated, NO-GO per
  the ARI plan's own diagnostic; not reopened here.
- The stashed (uncommitted) `consensus_cluster()`/`refine_clusters_iterative()`
  work from `stash@{0}` — a separate reconciliation decision, tracked
  independently, not part of this plan.
- `hdbscan_mcs_anchor`, sweep bounds (`hdbscan_pct_lo/hi`), duplicate-jitter,
  degenerate-DBCV fallback — audited above, all classified "keep," not
  touched.
- Any change to MLP/HMM/export stages.

## Changelog (implementation pass, Aug 2026)

**Status: IMPLEMENTED.** All three designs implemented and tested per the
companion implementation prompt's test plan. Every test listed there was
actually executed (not just written) — see below.

### Design 1 — `hdbscan_selection_mode`

Implemented in `run_hdbscan()` (`cube_core.py`), gated behind the new cfg
key with `"floor_soft_cap"` as literally specified in the algorithm section
above. New cfg key `hdbscan_overshoot_penalty` (default `0.01`) added
alongside it.

**Tests run:**
- Syntax/import check: pass.
- Backward-compat bit-identical check: pass, on (a) a fixed synthetic
  embedding (`make_blobs`, 10 centers) and (b) a real cached 28,235-point
  UMAP embedding from a prior production run — `hdbscan_selection_mode`
  unset vs. explicit `"legacy"` vs. a byte-for-byte reconstruction of the
  pre-change selection code all produced identical `(labels, score,
  n_clusters)`.
- Synthetic discontinuity reproduction: pass. Constructed nested-blob data
  (3 super-clusters of 15 tight sub-blobs each) with `preferred_clusters_lo
  =15`, `hdbscan_dbcv_thresh=0.75` where `"legacy"` mode collapses to 6
  clusters (sub-floor) while `"floor_soft_cap"` on the identical data
  selects 34 clusters (correctly avoiding the floor violation) and logs the
  expected quality `[WARN]`.
- Synthetic impossible-floor fallback: pass. With `preferred_clusters_lo=
  200` (unreachable for the dataset), `"floor_soft_cap"` selects the
  closest achievable count, logs `[WARN] no sweep candidate reached
  preferred_clusters_lo=200...`, and returns a valid non-crashing result.
- Real-data single-folder run (Test A, `20260407_Baseline_Exp1`, 7
  sessions): `"legacy"` and `"floor_soft_cap"` both selected 17 clusters,
  score 0.496 (silhouette fallback — DBCV was non-finite on this dataset) —
  bit-identical, as expected since the winning candidate already fell
  in-range.
- **Real-data full 3-group 8-seed sweep comparison (Test B — the decisive
  evidence):**
  - `"legacy"` per-seed counts: `[3, 3, 45, 3, 47, 3, 3, 31]` — **5 of 8
    seeds (62.5%) collapsed to the catastrophic 3-cluster outcome.** Raw
    mean pairwise ARI **0.3452**; floor-filtered (excluding the 5 sub-floor
    seeds per the user's noise-exclusion instruction) mean ARI **0.5860**
    over the 3 surviving seeds.
  - `"floor_soft_cap"` per-seed counts: `[42, 46, 45, 48, 47, 47, 48, 33]` —
    **0 of 8 seeds collapsed.** Raw mean ARI **0.5892** (identical to the
    floor-filtered figure since no seed was excluded).
  - Catastrophic undershoot: **eliminated** (5/8 → 0/8). Mean ARI improved
    +70.7% relative (0.345→0.589), and even beats legacy's own
    floor-filtered "best case" (0.586).
- Overshoot-penalty calibration (Test C, single-folder subset): `0.0`,
  `0.01`, `0.05` all produced identical 17-cluster / 0.496-score results —
  **inconclusive as scoped**, because this dataset's winning candidate
  never exceeds `preferred_clusters_hi=30`, so the overshoot term is never
  active. The penalty only visibly matters on multi-group data, where Test
  B's `"floor_soft_cap"` counts (33–48) ran well past the 30-cluster
  ceiling without much pull-back at `0.01`. **Flagged as follow-up, not a
  blocker for promotion** (see below).

**Promotion decision: PROMOTED.** Per the plan's explicit recommendation
("promote within this pass if — and only if — the real-data test plan
shows zero catastrophic-undershoot outcomes across a real multi-seed run
and no regression in DBCV/quality versus legacy mode on the primary run"):
both conditions were met (0/8 undershoot vs. legacy's 5/8; bit-identical
quality on the deterministic primary run). `BSoidEngine.DEFAULTS
["hdbscan_selection_mode"]` and the `AdvancedCUBEWindow` GUI combo default
were both updated to `"floor_soft_cap"` together, with the reasoning
recorded inline as a code comment at each site and in `run_hdbscan()`'s
docstring.

**Deviation from plan:** the plan doc and `run_hdbscan()`'s original
docstring both said `preferred_clusters_lo` defaults to 8; the live
`BSoidEngine.DEFAULTS` value is actually **12** (has been since before this
pass). Real-data testing used the live default (12) for floor-filtering,
which doesn't change the qualitative conclusion — legacy's collapsed seeds
landed at 3 clusters, nowhere near either 8 or 12.

**Remaining/open:** `hdbscan_overshoot_penalty` is shipped at its planned
default (`0.01`) but is not yet empirically calibrated on multi-group data
where it actually activates (Test C was inconclusive by construction — see
above). A follow-up pass should sweep the penalty (e.g. `0.01, 0.05, 0.1`)
against the 3-group dataset specifically, where `"floor_soft_cap"` is
currently landing 3–18 clusters past `preferred_clusters_hi=30` without
much visible pull-back.

### Design 2 — `plot_cluster_volatility()`

Implemented in `cube_core.py`, called immediately after
`plot_cluster_stability()` inside the existing seed-sweep gate
(`seed_sweep_n >= 2`). Required adding a `labels` key to
`seed_sweep_stability()`'s returned dict (additive-only change — existing
callers/keys unaffected) so the per-seed label arrays needed for
cross-seed Jaccard matching are available to the plot function.

**Tests run:** syntax check (pass); hand-computed unit test confirming
stable clusters score ~0 volatility, a cluster that splits every other
seed scores the hand-calculated 0.333 (pass); a fully-absorbed reference
cluster (zero Hungarian match in one seed) scores exactly 1.0, no crash
(pass); edge cases — empty sweep, sweep missing the `"labels"` key,
all-noise labels, single-seed sweep — all return without crashing and
without writing a file (pass); real-data run (single-folder, 4-seed
sweep) produced a visually sane `cluster_volatility.png` — most-volatile
clusters at top, scores in `[0, 1]`, annotated with cluster size (pass,
visually inspected).

### Design 3 — `plot_cluster_hierarchy()`

Implemented in `cube_core.py`; new cfg keys `cluster_hierarchy_enabled`
(default `True`) and `cluster_hierarchy_linkage` (default `"ward"`), both
with wired GUI widgets. Called on the production `hdb_labels` immediately
after rare-cluster pruning (not gated behind a seed sweep — runs every
time clustering runs, per the plan's additive-only reasoning).

**Tests run:** syntax check (pass); unit test with 4 synthetic blobs (2
close pairs, pairs far apart) confirmed via the raw `linkage()` matrix that
the two nearby blobs merge first (distance ≈1.0) and the cross-group merge
happens last, at ~100x that distance (pass); theme compliance confirmed in
both `"dark"` and `"light"` themes (plots produced, non-empty; function
body greped for hex-color literals — none found, uses `_TICK_COL`/
`_TEXT_COL`/`_BG` exclusively) (pass); edge cases (<2 clusters, `None`
feats/labels, all-noise labels) return without crashing (pass); real-data
run (single-folder subset) produced a 17-leaf `cluster_hierarchy.png` with
correct per-cluster sizes, visually inspected (pass).

### Shared regression test

Ran the primary pipeline through HDBSCAN with **no explicit override** for
any of the four new cfg keys (only `seed_sweep_n=3` set, to exercise both
new plots together) and confirmed: `engine._cfg["hdbscan_selection_mode"]`
resolved to `"floor_soft_cap"` and `cluster_hierarchy_enabled` resolved to
`True` purely from `BSoidEngine.DEFAULTS` (no cfg-key mismatch); no crash;
`cluster_hierarchy.png`, `cluster_volatility.png`, and the existing
`cluster_stability.png` all present and non-empty in the same run. Since
`hdbscan_selection_mode` was promoted (not left at `"legacy"`), the
plan's alternative "bit-identical to pre-pass baseline" regression check
does not apply — promotion is an intentional, documented behavior change
for auto-mode clustering, not a regression.

### GUI audit

All four new cfg keys (`hdbscan_selection_mode`, `hdbscan_overshoot_
penalty`, `cluster_hierarchy_enabled`, `cluster_hierarchy_linkage`) have
wired `AdvancedCUBEWindow` widgets under the existing "CLUSTER COUNT
GUIDANCE" section, using the established `_check`/`_spin_f`/`_combo`
helpers. Defaults verified to match `BSoidEngine.DEFAULTS` exactly
(including post-promotion: both `DEFAULTS` and the GUI combo's default were
updated to `"floor_soft_cap"` together). `AdvancedCUBEWindow`'s generic
`self._vars` registration/collection mechanism means every widget
registered via `self._v(...)` is automatically picked up when the cfg dict
is assembled — no separate manual wiring step was needed, and none was
skipped.

### What deviated from plan and why

1. `preferred_clusters_lo`'s documented default (8) vs. actual live default
   (12) — pre-existing inconsistency in the plan doc / old docstring, not
   introduced by this pass; noted above, doesn't change conclusions.
2. Test C (overshoot-penalty calibration) was inconclusive as scoped to the
   single-folder dataset, because that dataset's winning candidate never
   exceeds `preferred_clusters_hi`. Flagged as a follow-up rather than
   silently re-scoping to the 3-group dataset mid-test, per this project's
   "stop and report rather than guessing past it" convention for the plan
   doc's flagged open questions.
3. `seed_sweep_stability()`'s return dict gained a new `"labels"` key
   (needed for Design 2) — additive-only, not itself part of the original
   three-item scope but a necessary minimal prerequisite for Design 2 as
   specified.

### What remains

- Overshoot-penalty calibration on multi-group data (see "Remaining/open"
  under Design 1).
- Design 3's `bodypart_names` parameter is accepted but unused — no
  upstream variable currently exists to populate it; left as a no-op
  passthrough rather than speculatively wiring up unused plumbing.

## Open questions for user decision before implementation starts

1. `hdbscan_overshoot_penalty=0.01` is a reasoned starting guess, not
   empirically tuned — the implementation prompt's test plan includes an
   A/B sweep of a few values against real data; is a specific target
   behavior preferred (e.g. "essentially never penalize overshoot" →
   near-0, vs. "actively discourage drifting far past 30" → higher)?
2. Design 3's hierarchy plot uses feature-space centroids by default
   (reasoned above); confirm this is the right space vs. UMAP-embedding
   centroids, given the intended use is manual behavioral merging.
3. Should `hdbscan_selection_mode="floor_soft_cap"` be promoted to the
   shipped default within this same implementation pass once real-data
   testing confirms it eliminates the catastrophic-undershoot seeds, or
   held opt-in for a separate validation pass (matching the conservative
   two-step pattern used throughout this project)? Recommendation: promote
   within this pass if — and only if — the real-data test plan in the
   companion implementation prompt shows zero catastrophic-undershoot
   outcomes across a real multi-seed run and no regression in DBCV/quality
   versus legacy mode on the primary (non-swept) run.
