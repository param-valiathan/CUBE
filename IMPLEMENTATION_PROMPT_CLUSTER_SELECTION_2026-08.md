# Implementation Prompt: Cluster-Selection Simplification & New Diagnostics

Copy everything below this line into a fresh session to execute the plan.

---

You are implementing a planned simplification of CUBE's HDBSCAN cluster-count
selection logic, plus two new plot outputs. CUBE is a B-SoiD-based
unsupervised behavior-clustering pipeline for pose-tracking data
(deer/animal behavior, single-camera, multi-experimental-group studies).
The full, already-reviewed design is written out in:

**`d:\CUBE\CLUSTER_SELECTION_SIMPLIFICATION_PLAN_2026-08.md`** — read this
document in full before writing any code. It is the spec: exact functions,
cfg keys, defaults, and the reasoning behind every choice (including which
existing complexity was audited and kept vs. removed). This prompt tells you
how to execute that document safely; it is not a substitute for reading it.

Background (for context only, no action needed — already done before this
prompt was written): a prior ARI-stability investigation found seed-sweep
mean ARI sitting at ~0.41 on the real 3-group dataset, tested three levers
(`umap_init`, `umap_n_epochs`, `seed_sweep_n_jobs`) via real-data A/B
comparison, found the first two gave no genuine benefit, and traced the
actual root cause to a discontinuous two-branch selection rule inside
`run_hdbscan()` — which is what this plan replaces. `umap_init` and
`umap_n_epochs` have **already been fully removed** from `cube_core.py` and
`cube.py` (cfg keys, `run_umap()` kwargs, `DEFAULTS` entries, GUI rows) —
confirm this with `grep -n "umap_init\|umap_n_epochs" cube_core.py cube.py`
before starting (should return nothing); if it returns matches, the removal
did not stick or was reverted — stop and report rather than assuming this
prompt's description of the starting state is accurate. `seed_sweep_n_jobs`
and the `seed_sweep_stability_bootstrap()` diagnostic were kept and remain
in the codebase. The investigation's own planning documents have since been
archived (not present in the repo) — nothing further needs reading from
them; everything you need is in the plan doc above.

## Scope — read carefully before starting

This pass implements exactly three things, each independently testable:

1. **Design 1** — new `hdbscan_selection_mode` cfg key (`"legacy"` default |
   `"floor_soft_cap"` new) replacing the two-branch in-range/fallback logic
   in `run_hdbscan()` (`cube_core.py:715-939` as of this plan's writing —
   **re-verify the actual current line numbers with Read before editing,
   they will have drifted**) with one continuous ranking pass. New cfg key
   `hdbscan_overshoot_penalty` (default `0.01`), active only under
   `"floor_soft_cap"`.
2. **Design 2** — new `plot_cluster_volatility()` function + call site.
3. **Design 3** — new `plot_cluster_hierarchy()` function + call site, plus
   `cluster_hierarchy_enabled` (default `True`) and
   `cluster_hierarchy_linkage` (default `"ward"`) cfg keys and their GUI
   rows.

(`umap_init`/`umap_n_epochs` removal, listed as a scope item in an earlier
draft of this prompt, is not part of this pass — it was already completed
beforehand; see the Background note above.)

**Explicitly out of scope, do not touch:**
- Tier 2 (decoupling clustering from UMAP) — separately gated, not
  reopened by this pass.
- The uncommitted `stash@{0}` work (`consensus_cluster()`,
  `refine_clusters_iterative()`, HMM emission-mode machinery, etc.) — a
  separate reconciliation the user has not yet decided on. Do not pop this
  stash, do not assume its contents exist, do not build on top of it. If a
  merge conflict or reference to stashed code shows up unexpectedly, stop
  and report rather than resolving it yourself.
- `hdbscan_mcs_anchor`, sweep bounds (`hdbscan_pct_lo`/`hdbscan_pct_hi`),
  the duplicate-coordinate jitter, the degenerate-DBCV→silhouette fallback
  — all audited in the plan doc and classified "keep as-is."
- MLP, HMM, export stages — untouched by this pass.

## Environment — mandatory, non-negotiable

- All Python execution uses `C:\Users\param\anaconda3\envs\CUBE\python.exe`
  — never bare `python`. Verify first:
  `"C:\Users\param\anaconda3\envs\CUBE\python.exe" -c "import umap, hdbscan, scipy; print('OK')"`.
  If this fails, stop and report.
- Files to edit: `d:\CUBE\cube_core.py` (all engine/plot logic) and
  `d:\CUBE\cube.py` (GUI — new cfg keys need Advanced Settings widgets, and
  removed cfg keys need their widgets removed too). Nothing else in the
  four-file architecture should need changes — if you find yourself wanting
  to touch `cube_analyser.py` or `cube_video_explorer.py`, stop and
  reconsider; this pass is scoped to the primary pipeline engine and its
  GUI only.
- Use your own scratchpad/temp directory for all test scripts, cached
  intermediate arrays, and throwaway output — never write test artifacts
  into the CUBE repo itself, and never overwrite the user's real project
  data (`D:\Damien DLC\*\BSOID_Project_Ready`, `D:\CUBE_Pipeline\*`).

## Backups — mandatory, before touching any code

1. Run `git status` in `d:\CUBE` and report it. As of this plan's writing
   the tracked-file working tree is clean (matches `origin/main`) with
   `stash@{0}` holding unrelated prior work and several untracked planning
   `.md` files present — **confirm this is still the state before
   proceeding**; if the tracked tree is dirty for a reason not explained
   above, stop and ask the user what to do with it, exactly as the prior
   ARI session's prompt required. Do not discard or commit on the user's
   behalf without asking.
2. Take a file-level backup independent of git: copy `cube_core.py` and
   `cube.py` to your scratchpad directory with a timestamp suffix (e.g.
   `cube_core.py.backup_<date>`) before the first edit of this pass. This
   is in addition to git, not instead of it.
3. Do **not** run `git commit` or `git push` until every test in this
   prompt's test plan has passed. Report what's ready to commit and wait
   for explicit instruction — standing project convention (`CLAUDE.md`).
4. If a change under test breaks the existing pipeline in a way you can't
   immediately fix, revert that specific change via git or your file
   backup before moving on — do not leave the working tree broken between
   test items.

## Non-negotiable safety principles

1. **Design 1 (the selection-logic change) must default to today's exact
   existing behavior** (`hdbscan_selection_mode="legacy"`) until real-data
   validation (test plan below) justifies promoting `"floor_soft_cap"` to
   the shipped default. A user who changes nothing must get bit-identical
   clustering output until this item is deliberately promoted. This is the
   same two-step pattern used for the Aug 2026 HMM work and the ARI Tier 1
   work — ship opt-in, validate, promote in a clearly-logged step.
2. **Design 2 and Design 3 (the two new plots) are additive-only and do
   not need the opt-in gate** — they produce new output files and touch no
   existing numeric result (labels, model weights, existing CSVs/plots are
   untouched). `cluster_hierarchy_enabled` defaults `True` for exactly this
   reason (per the plan doc's explicit reasoning) — do not over-apply
   safety-gate caution here; the plan doc already made this call, don't
   re-litigate it mid-implementation.
3. **Read before edit, always.** Use the Read tool on the exact
   function/region before every Edit call — do not trust this prompt's or
   the plan doc's line-number references, they drift.
4. **No large speculative rewrites.** Implement and test one item (Design
   1, then Design 2, then Design 3) before moving to the
   next — do not implement all three and test at the end.
5. **Every test must actually run**, executed via PowerShell with the CUBE
   python, with real captured output reported. Zero tolerance for "I wrote
   a test that would verify X" without running it.
6. **Real-data validation is required, not optional.** Use the real
   3-group, 21-session dataset at
   `D:\Damien DLC\20260407_Baseline_Exp1\BSOID_Project_Ready\csv` (+CLZ
   +DCZ) for full-scale tests, a single-folder subset for faster dev
   iteration — both patterns were validated and timed in the prior ARI-
   stability session (full 3-group sweep: ~15-20 min at
   `seed_sweep_n_jobs=1`, proportionally faster at higher `n_jobs`;
   single-folder full pipeline through export: ~4 min). Use the fast-path
   harness pattern (instantiate `BSoidEngine`, monkeypatch
   `cube_core.train_mlp` to raise a sentinel exception, catch it around
   `engine.run()`) for anything that only needs to exercise
   clustering/sweep stages.
7. **If a planned design detail turns out wrong or infeasible once you're
   actually in the code** (e.g. the exact overshoot-penalty calibration, or
   a data-space choice for the hierarchy plot), stop and report the
   discrepancy with your proposed resolution rather than silently
   improvising past it — the plan doc's "Open questions" section flags the
   two most likely spots this could happen (overshoot-penalty magnitude,
   hierarchy plot's feature space choice).

## GUI requirement — explicit

For every cfg key touched by this pass:
- **New keys** (`hdbscan_selection_mode`, `hdbscan_overshoot_penalty`,
  `cluster_hierarchy_enabled`, `cluster_hierarchy_linkage`) need a properly
  wired widget in `AdvancedCUBEWindow` (`cube.py`), following the
  established `_adv_row` + `_check`/`_spin_f`/`_spin_i`/`_combo` pattern —
  search `cube.py` for the "Cluster selection method" combo row
  (`hdbscan_method`, around `cube.py:3505` as of this writing — re-verify
  with Read/grep, it will have drifted) as your reference for a
  combo-with-help-text row, and the "CLUSTER COUNT GUIDANCE" section
  (`target_n_clusters`/`preferred_clusters_lo`, around `cube.py:3517-3534`)
  as your reference for the section these new keys extend.
- Confirm every new widget's default value matches `BSoidEngine.DEFAULTS`
  exactly. Double-check you're not introducing a cross-file key-name
  mismatch — this project has a demonstrated history of exactly that bug
  class (the `pca_pre_reduce`/`pca_n_components` naming confusion from the
  HMM work, and note `AdvancedCUBEWindow._BASELINE`'s fallback values are
  intentionally allowed to differ from `BSoidEngine.DEFAULTS` since
  `DEFAULTS = {**_BASELINE, **dict(BSoidEngine.DEFAULTS)}` always lets the
  engine win — don't "fix" `_BASELINE` values thinking they're bugs, they
  are deliberately just an import-failure fallback).
- Re-run a manual GUI trace (or `/code-review cube.py`) after adding/
  removing widgets to catch any unreachable-setting or dead-code-path bug
  before considering the item done.

## Test plan (execute in this order)

### Design 1 — `hdbscan_selection_mode`
1. Syntax/import check (`py_compile`) after editing `run_hdbscan()`.
2. Backward-compat: confirm `hdbscan_selection_mode` unset (or explicitly
   `"legacy"`) produces bit-identical `(labels, score, n_clusters)` output
   to pre-change code, on a fixed synthetic embedding and on one real
   cached embedding. (Note: unlike UMAP, HDBSCAN + this selection logic
   *is* deterministic given a fixed embedding and cfg — a true bit-identical
   check is valid here, unlike the UMAP-epoch/init checks in the prior ARI
   pass which needed a kwarg-level check instead due to UMAP's own
   run-to-run non-determinism.)
3. Unit test on synthetic data: construct a small synthetic embedding
   (e.g. `sklearn.datasets.make_blobs` with a deliberately awkward number
   of well-separated blobs, some tiny/brief clusters mixed with large ones)
   where you can hand-verify: (a) `"legacy"` mode can be made to produce a
   sub-floor result by picking blob counts/separations that starve the
   in-range bucket (reproduce the discontinuity deliberately), and (b)
   `"floor_soft_cap"` mode on the *same* synthetic data never selects below
   `pref_lo` clusters when at least one sweep candidate reaches it.
4. Unit test: construct synthetic data where **no** candidate in the whole
   sweep reaches `pref_lo` (e.g. a dataset with only 3-4 true clusters and
   `pref_lo=8`) — confirm `"floor_soft_cap"` hits the genuinely-impossible
   fallback path, logs the `[WARN]`, and still returns a valid result
   (doesn't crash).
5. Real-data run: run the primary pipeline (fast-path harness, single-
   folder subset) with `hdbscan_selection_mode="legacy"` vs.
   `"floor_soft_cap"`, same data/seed, report cluster count and
   score for both.
6. Real-data seed-sweep comparison (the actual test of whether this fixes
   the problem): run `seed_sweep_stability()` with `n_seeds=8` (or more,
   now that `seed_sweep_n_jobs` makes it cheap — consider 16-20 given the
   plan's point about outlier visibility) under both selection modes on
   the full 3-group dataset. Report the full cluster-count list and mean
   ARI for both. **This is the evidence that decides promotion — report it
   explicitly, don't bury it.** Specifically check: did the catastrophic
   low-cluster-count seeds disappear under `"floor_soft_cap"`?
7. `hdbscan_overshoot_penalty` calibration: with `"floor_soft_cap"` active,
   sweep 2-3 values (e.g. `0.0`, `0.01`, `0.05`) against the real single-
   folder subset, report resulting cluster count and DBCV for each — this
   is the evidence for the plan doc's open question 1.
8. Promotion decision: per the plan doc's recommendation, promote
   `hdbscan_selection_mode` default to `"floor_soft_cap"` **only if** step 6
   shows zero catastrophic-undershoot outcomes across the real multi-seed
   run and no DBCV/quality regression on the primary run from step 5. State
   the decision explicitly either way. If promoted, update
   `BSoidEngine.DEFAULTS` and the GUI widget's default together, and note
   the promotion in this plan doc's changelog (create one, matching the
   pattern used in the archived prior ARI-stability implementation plan).

### Design 2 — `plot_cluster_volatility()`
1. Syntax/import check.
2. Unit test on synthetic data: build a small synthetic multi-seed sweep
   result by hand (a `sweep` dict shaped like `seed_sweep_stability()`'s
   return, with known, hand-constructed label arrays across a few
   "seeds" where you control exactly which points move between which
   clusters) — confirm the computed per-cluster volatility scores match
   your hand calculation (e.g. a cluster whose membership is identical
   across all seeds should score ~0; a cluster that's a 50/50 split with
   another cluster every other seed should score high).
3. Confirm the plot function handles the edge cases: a seed where a
   reference cluster has zero match (fully absorbed elsewhere) → volatility
   1.0, not a crash; a `sweep` with only noise/no clusters → returns
   without crashing (matches existing `plot_cluster_stability`'s
   `if not sweep or "ari" not in sweep: return` guard style).
4. Real-data run: run the seed sweep on the full 3-group dataset (reuse the
   sweep result from Design 1's step 6 if still available/cached — no need
   to redo the 15-20 min sweep twice), generate `cluster_volatility.png`,
   and open/inspect it (or describe its content from the saved array data)
   to confirm it's visually sane — most-volatile clusters at the top,
   scores in `[0, 1]`.

### Design 3 — `plot_cluster_hierarchy()`
1. Syntax/import check.
2. Unit test on synthetic data: construct feature data with clear known
   groupings (e.g. 4 tight blobs where two are close together and two are
   far apart) and labels: confirm the dendrogram's linkage structure
   groups the two nearby blobs together before merging with the distant
   ones (a basic sanity check that centroid computation and `linkage()`
   wiring are correct, not a deep statistical test).
3. Confirm theme compliance: run with `plot_theme="dark"` and `"light"`,
   confirm `_BG`/`_PANEL`/`_TEXT_COL`/`_TICK_COL` globals are used (no
   hardcoded colors) — grep the new function for hex-color literals as a
   quick check, matching this codebase's existing convention.
4. Real-data run: full pipeline through export on the single-folder
   subset, confirm `cluster_hierarchy.png` is produced, non-empty, and
   openable, with a label per cluster matching the run's actual cluster
   count.

### Shared regression tests (run once, after all items individually pass)
1. Full pipeline run with `hdbscan_selection_mode` at whatever its final
   post-promotion-decision default is, `cluster_hierarchy_enabled=True`
   (its default) — confirm no crash, all expected output files present
   (existing ones unchanged in *kind*, new `cluster_hierarchy.png` and, if
   a seed sweep was requested, `cluster_volatility.png` present).
2. If `hdbscan_selection_mode` was **not** promoted (stayed `"legacy"`
   default): confirm a full real-data run with all-default cfg produces
   bit-identical clustering output to a pre-this-pass baseline run on the
   same fixed dataset/seed — the "did we actually preserve default
   behavior" guarantee this project enforces for every non-promoted
   opt-in change.

## Definition of done

- Design 1 implemented, gated behind `hdbscan_selection_mode`, tested per
  above, promotion decision made explicitly and justified with the real
  seed-sweep evidence from step 6.
- `umap_init`/`umap_n_epochs` confirmed still absent (zero references via
  grep) at the start of this pass — precondition check, not new work.
- Design 2 and Design 3 implemented and tested per above, both producing
  real, inspected output on real data.
- Every test in this prompt's test plan actually executed, with reported
  pass/fail results listed individually — not summarized as "tests
  passed."
- Every new/changed cfg key has a correctly-wired GUI widget, defaults
  verified to match `BSoidEngine.DEFAULTS` exactly, and a post-hoc GUI
  audit performed.
- A backup of `cube_core.py`/`cube.py` exists independent of git, taken
  before the first edit.
- Nothing committed or pushed to git without explicit, separate
  instruction.
- A short written summary (chat response, or an appended "Changelog"
  section in `CLUSTER_SELECTION_SIMPLIFICATION_PLAN_2026-08.md`) covering:
  what was implemented, the real seed-sweep before/after comparison
  (cluster counts + mean ARI, explicitly, both modes), the promotion
  decision and why, the `hdbscan_overshoot_penalty` calibration result,
  what the two new plots look like on real data, what deviated from plan
  and why, and what remains.

## If you get stuck

If the overshoot-penalty magnitude or the hierarchy plot's feature-space
choice (the plan doc's two flagged open questions) becomes a real blocker
— e.g. no tested value produces a clearly-better real-data result, or the
feature-space centroids produce a degenerate/uninterpretable dendrogram —
stop and report with your findings and a proposed resolution rather than
guessing past it, matching this project's established convention for
design questions left open deliberately.
