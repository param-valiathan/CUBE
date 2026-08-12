# Implementation Prompt: Consensus-Clustering Validation Experiment

Copy everything below this line into a fresh session to execute the plan.

---

You are running a validation experiment against CUBE's `consensus_cluster()`
clustering path in `d:\CUBE\cube_core.py`. CUBE is a B-SoiD-based
unsupervised behavior-clustering pipeline for pose-tracking data
(deer/animal behavior, single-camera, multi-experimental-group studies).
The full experiment design, rationale, and decision rule are written out in:

**`d:\CUBE\CONSENSUS_VALIDATION_PLAN_2026-08.md`** — read this document in
full before running anything. It is the spec: exact configs, metrics, and
the reasoning behind every choice. This prompt tells you how to execute
that document; it is not a substitute for reading it.

Background (for context only, no action needed — already done before this
prompt was written): `consensus_cluster()`'s auto-trigger
(`consensus_auto_threshold`) was just re-enabled at `0.55` after a prior
4-config real-data experiment showed seed-sweep ARI consistently unstable
(0.18–0.29) on this dataset even with the `hdbscan_selection_mode=
"floor_soft_cap"` fix already in place. Consensus also just gained a new
opt-in post-hoc refinement pass (`consensus_refine_enabled`,
`merge_by_coassociation()`) and feature-space DBCV/silhouette scoring
(`_dbcv_feature_space()`) — the first metric directly comparable between
consensus and the primary single-seed path. None of this has been
validated against real data — that's what this experiment does.

## Scope — read carefully before starting

This is a **read-only experiment**. You are not modifying `cube_core.py`,
`cube.py`, or any other source file. You are writing a standalone script
that imports `cube_core.BSoidEngine` and runs it 3 times with different
cfg overrides, exactly as specified in the plan doc's "Configs" table.

**Explicitly out of scope:**
- Any change to `cube_core.py`'s `DEFAULTS`, `consensus_cluster()`,
  `refine_consensus_clusters()`, `merge_by_coassociation()`, or
  `_dbcv_feature_space()`. If you find a bug in any of these while running
  the experiment, **stop and report it** — do not silently patch around it
  to make the experiment complete. The one exception: if the experiment
  cannot run *at all* due to a genuine crash (not a quality/logic
  question), report the crash with full traceback and ask before touching
  source.
- Repeating the earlier 4-config ARI-stability experiment (down-weighted
  bodyparts, `umap_n_neighbors=60`) — that already ran; this experiment is
  scoped purely to primary-vs-consensus comparison.
- The "2-3 seed" follow-up mentioned in the plan doc's "Known limitation"
  section — that is an explicit next step, not part of this pass, unless
  the user asks for it after seeing this pass's results.

## Environment — mandatory, non-negotiable

- All Python execution uses `C:\Users\param\anaconda3\envs\CUBE\python.exe`
  — never bare `python`. Verify first:
  `"C:\Users\param\anaconda3\envs\CUBE\python.exe" -c "import umap, hdbscan, scipy; print('OK')"`.
  If this fails, stop and report.
- Dataset (already prepped, do not re-run Step 2 prep):
  ```
  D:\Damien DLC\20260407_Baseline_Exp1\BSOID_Project_Ready\csv
  D:\Damien DLC\20260408_CLZ_Exp1\BSOID_Project_Ready\csv
  D:\Damien DLC\20260409_DCZ_Exp1\BSOID_Project_Ready\csv
  ```
- Write the experiment script and all run output to your scratchpad/temp
  directory — never into the CUBE repo, never touching
  `D:\Damien DLC\*\BSOID_Project_Ready` or `D:\CUBE_Pipeline\*` (real
  project data — read-only, csv folders are inputs only).

## Script to write

A single script, modeled on this session's prior ARI-experiment harness
(same repo, same dataset, same pattern — search for it in your scratchpad
if a copy still exists from this session; otherwise write fresh):

```python
import json, re, time, traceback
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, r"d:\CUBE")
import cube_core

CSV_DIRS = [
    r"D:\Damien DLC\20260407_Baseline_Exp1\BSOID_Project_Ready\csv",
    r"D:\Damien DLC\20260408_CLZ_Exp1\BSOID_Project_Ready\csv",
    r"D:\Damien DLC\20260409_DCZ_Exp1\BSOID_Project_Ready\csv",
]
OUT_ROOT = Path(r"<your scratchpad>/consensus_validation_out")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

BASE_CFG = dict(
    hmm_enabled=False, save_plots=False, save_videos=False,
    seed_sweep_n=0,                    # disabled -- see plan doc
    consensus_auto_threshold=0,        # force deterministic path per config
    consensus_n_seeds=8,
    umap_random_state=42,
)

CONFIGS = {
    "primary_only":     {"consensus_clustering_enabled": False},
    "consensus_plain":  {"consensus_clustering_enabled": True,
                          "consensus_refine_enabled": False},
    "consensus_refined": {"consensus_clustering_enabled": True,
                           "consensus_refine_enabled": True,
                           "consensus_merge_coassoc_thresh": 0.5},
}

# For each config: instantiate BSoidEngine(csv_folder=CSV_DIRS,
# video_folder=None, output_dir=<per-config timestamped dir under
# OUT_ROOT>, cfg={**BASE_CFG, **override}, logger=<capture to list + print>).
# Call engine.run(). Extract from the returned `results` dict / written
# validation_report.json / bsoid_run_summary.json:
#   - n_clusters            <- results["n_clusters"]
#   - cv_accuracy           <- results["summary"]["cv_accuracy"]
#   - runtime_s             <- wall-clock around engine.run()
# Extract from captured log lines (regex, same technique as the prior ARI
# harness) or from consensus_cluster()'s own quality dict if you can reach
# it (it isn't returned by run() directly -- log-line capture is the
# reliable path here, same as the prior harness used for DBCV):
#   - noise_pct
#   - dbcv_feature_space, silhouette_feature_space
#     (look for the "[feature-space] DBCV=... silhouette=..." log line --
#     note: for primary_only this line is logged once, post-refinement/
#     pruning; for the two consensus configs it's logged right after the
#     "Consensus: N clusters, ..." line, also post-refinement/pruning)
#   - separation_ratio, per_seed_counts (consensus configs only, from the
#     "Consensus: N clusters, ... separation_ratio=X.XXx ... per-seed
#     counts=[...]" log line)
# Write a summary.json with all 3 configs' extracted metrics after each
# config completes (not just at the end) so partial progress survives a
# crash on a later config.
```

Match the prior harness's robustness conventions: wrap each config's
`engine.run()` in try/except so one config's failure doesn't lose the
others; log which regex/extraction failed if a field comes back `None`
rather than silently reporting a wrong number.

## Execution

Run via the mandated interpreter, in the **background** (each config is a
full pipeline run — expect single-digit minutes for `primary_only`,
likely more for the two consensus configs given the 8-seed loop plus,
for `consensus_refined`, however much the split/merge pass adds — budget
generously, up to 20-30 min for `consensus_refined` if it hits many impure
clusters):
```
"C:\Users\param\anaconda3\envs\CUBE\python.exe" <script>.py
```

## Reporting — mandatory format

Produce a markdown table with one row per config and these columns:
`config | n_clusters | noise_pct | dbcv_feature_space |
silhouette_feature_space | cv_accuracy | runtime_s`.

Then two delta rows (`consensus_plain - primary_only`,
`consensus_refined - primary_only`) for the four numeric quality/cost
columns.

Then, explicitly, per the plan doc's decision rule: state **help / no
help / inconclusive** for `consensus_plain` and for `consensus_refined`
separately, with the one-line reasoning (which deltas drove the verdict).
State the single-run limitation alongside the verdict, every time — do not
let the table alone imply more confidence than a single run actually
supports.

Finally: **update `CONSENSUS_VALIDATION_PLAN_2026-08.md`'s Changelog
section** with the results table, verdicts, and any deviations from this
prompt's plan (e.g. a config that failed and how, a metric that couldn't
be extracted and why) — this is part of "done," not an optional extra.

## If you get stuck

If `consensus_refined` runs dramatically longer than expected (the plan
doc's cited historical worst case was 25+ minutes for the split pass alone
on one seed, before the mitigations described in `split_impure_clusters()`'s
own docstring), let it run rather than killing it early — but if it passes
30 minutes with no progress in the log, stop it, report exactly where it
was stuck, and ask before retrying with a smaller `hdbscan_split_max_candidates`
or similar mitigation — don't silently change consensus's own cfg defaults
to work around a real perf problem without flagging it.

If any of the three configs crashes outright (not just runs slow), report
the full traceback and the other two configs' results (don't let one
failure block reporting what did complete) — matching the "report what's
ready" convention used throughout this project's other experiment prompts.
