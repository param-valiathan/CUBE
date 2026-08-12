# Implementation Prompt: Clustering & HMM Improvement Plan

Copy everything below this line into a fresh session to execute the plan.

---

You are implementing a planned set of improvements to CUBE, a B-SoiD-based
unsupervised behavior-clustering pipeline for pose-tracking data (deer/animal
behavior, single-camera, multi-experimental-group studies). The full,
already-reviewed design is written out in
`d:\CUBE\CLUSTERING_HMM_IMPROVEMENT_PLAN_2026-08.md` — **read that file in
full before writing any code.** This prompt tells you how to execute it
safely, not what to build; the plan file is the spec.

## Environment — mandatory, non-negotiable

- All Python execution uses `C:\Users\param\anaconda3\envs\CUBE\python.exe`
  — never bare `python`, never system Python. Verify with
  `"C:\Users\param\anaconda3\envs\CUBE\python.exe" -c "import umap; print('OK')"`
  before doing anything else. If this fails, stop and report — do not work
  around it.
- The only files you should be editing are `d:\CUBE\cube_core.py` and, if
  Part A's log-line changes require it, `d:\CUBE\cube.py`. Nothing else in
  the four-file architecture (`cube_analyser.py`, `cube_video_explorer.py`)
  needs to change per the plan — if you find yourself wanting to touch
  those, stop and reconsider whether the plan actually requires it.
- Use your own scratchpad/temp directory for all test scripts, cached
  intermediate arrays, and throwaway output — never write test artifacts
  into the CUBE repo itself, and never overwrite the user's real project
  data (`D:\Damien DLC\*\BSOID_Project_Ready`, `D:\CUBE_Pipeline\*`) during
  testing. If a test needs real DLC data, either point read-only at the
  existing `D:\Damien DLC\...` raw folders/`BSOID_Project_Ready\csv`
  exports, or export a scratch copy elsewhere — never write into the live
  project tree.
- Do not run `git commit` or `git push` at any point, even after all tests
  pass. Report what's ready to commit and wait for explicit instruction.
  This matches the existing project convention in `CLAUDE.md`.

## Why this plan exists (condensed context — the plan file has full detail)

Real-data testing on a 3-group, 21-session, single-camera deer-behavior
dataset this session found two independent, unrelated sources of wasted
computation / missed quality:

1. **Consensus clustering** (an existing, already-implemented 8-seed
   co-association fallback for when the primary HDBSCAN partition is
   seed-unstable) triggers on effectively every real run of this dataset.
   But the pipeline still pays for a separate, purely diagnostic 6-seed
   stability sweep beforehand whose only job is deciding *whether* to
   trigger consensus — a decision that's moot once consensus is forced on.
   That's ~10 minutes of discarded computation per run.
2. **HMM temporal smoothing** (the post-hoc pass that cleans up single-frame
   label flicker in the final exported behavior labels) is structurally
   underpowered: it only ever sees the MLP's hard argmax labels, never its
   confidence, and it operates on frame-repeated labels rather than the
   underlying 100ms bins the clustering actually reasons over.

Both are real, verified findings from this session's investigation (grep
results, function reads, and reasoning are captured in the plan file) — you
are not re-deriving the rationale, you're executing an already-reviewed
design.

## Non-negotiable safety principles for this implementation

1. **Every new behavior must default to today's exact existing behavior.**
   Every new cfg key introduced by the plan (`consensus_clustering_enabled`
   already exists; `hmm_emission_mode`, `hmm_smoothing_level`,
   `hmm_transition_prior` are new) must default to the value that reproduces
   current output exactly. A user who changes nothing in their config must
   get bit-identical results before and after your changes. This is the
   single most important constraint in this entire task — violate it and
   the change is not acceptable regardless of how good the new mode is.
2. **Read before edit, always.** Use the Read tool on the exact function/
   region before every Edit call. Do not guess at line numbers or content
   from the plan document's line-number references — the file may have
   drifted since the plan was written; always re-verify against the live
   file.
3. **No large speculative rewrites.** Each of Part A, B.3, B.2, B.1 is a
   separably testable unit. Implement one, test it per its section in the
   plan, confirm it passes, *then* move to the next. Do not implement all
   four and test at the end — if something breaks, you need to know which
   change broke it.
4. **`predict_labels()`'s backward-compatibility test (plan section B.6,
   item 2) is the highest-stakes single test in this whole task.** It has
   two call sites in the entire codebase (`cube_core.py`, confirmed by grep
   during planning — re-confirm this yourself before editing, the codebase
   may have changed). One of them (`cube_core.py:~8499` at plan-writing
   time, an "apply a saved model to new data" utility) must be **fully
   traced and read** before you touch the function signature, to confirm
   the plan's assumption that it doesn't need the new return detail still
   holds. If that assumption is wrong, stop and report rather than forcing
   it to match.
5. **Every test must actually run, not just be written.** "I wrote a test
   that would verify X" is not sufficient — execute it via PowerShell with
   the CUBE python, capture and report the actual output/pass-fail.
6. **Long-running real-data tests should use the established fast-path
   harness**, not a full pipeline run through export: instantiate
   `BSoidEngine`, monkeypatch `cube_core.train_mlp` to raise a sentinel
   exception, catch it around `engine.run()`. This runs every pipeline
   stage through HDBSCAN/consensus/HMM-adjacent code paths without paying
   for the slow MLP-training-through-video-export tail, matching the
   pattern used throughout this session's own investigation. Example
   skeleton (adapt paths/cfg to what you're testing):

   ```python
   import sys, time, traceback
   sys.path.insert(0, r"D:\CUBE")
   import cube_core as cc

   class _StopEarly(Exception):
       pass

   def _stub_train_mlp(*a, **kw):
       raise _StopEarly()

   cc.train_mlp = _stub_train_mlp

   engine = cc.BSoidEngine(
       csv_folder=[...],       # existing BSOID_Project_Ready/csv folders, read-only
       video_folder=None,
       output_dir=r"<scratch dir>",
       fps=None,
       logger=print,
       cfg={...},               # the specific flag(s) under test
   )
   try:
       engine.run()
   except _StopEarly:
       pass
   ```

7. **If any planned design detail turns out to be wrong or infeasible once
   you're actually reading the code** (the plan document flags several
   places where this is possible — e.g. the `GaussianHMM` state-alignment
   logic in B.1 is explicitly unsolved in the plan, left for
   implementation-time design), stop and report the discrepancy with your
   proposed resolution rather than silently improvising past it.

## Implementation order (do not reorder or parallelize)

Follow the plan document's "Rollout order" section exactly:

1. **Part A** — consensus/seed-sweep dedup (`cube_core.py`, `consensus_cluster()`
   and `run()`'s sweep-trigger logic). Full test plan in the document's
   "A.4 Test plan", all 7 items.
2. **B.3** — per-cluster HMM transition priors (`train_hmm()` only, no
   signature changes elsewhere). Test plan: document section B.6, items 1,
   4, 7, 10 (the ones scoped to B.3).
3. **B.2** — bin-level HMM smoothing (requires the `predict_labels()`
   signature change). Test plan: B.6 items 1, 2, 3, 5, 7, 10.
4. **B.1** — soft-probability HMM emissions (`GaussianHMM` path, new
   alignment logic). Test plan: B.6 items 1, 2, 3, 6, 7, 10.
5. **Do not implement B.4** (the self-training/pseudo-labeling loop) — it
   is explicitly deferred in the plan, out of scope for this task.

After each numbered step: run its full test set, report pass/fail for each
test explicitly (don't summarize as "tests passed" — list each one), and
wait for acknowledgment before starting the next step **unless** you were
told at the start of this session to run the whole sequence autonomously —
if that instruction wasn't given, treat each step boundary as a checkpoint
to report back at.

## Definition of done, overall

- All four steps (A, B.3, B.2, B.1) implemented, each independently gated
  behind a cfg flag defaulting to current behavior.
- Every test in plan sections A.4 and B.6 has been actually executed (not
  just written) with reported results.
- At least one full real-data run (using the existing 3-folder deer dataset
  or an equivalent already-available scratch export) completed successfully
  with each new flag turned on individually, and one with all defaults
  (proving the no-op case still works).
- A short written summary (where and how — your choice: chat response,
  or an update appended to the plan document under a "Changelog" heading)
  covering: what was implemented, what deviated from the original plan and
  why, what the comparative-quality findings were (plan section B.6 item 9
  is explicitly informational, not pass/fail — report what you observed),
  and what remains for a possible future B.4 follow-up.
- Nothing committed or pushed to git.

## If you get stuck

If a design question in the plan's "Open questions for user decision" section
becomes blocking (e.g. you cannot proceed with B.1 without knowing whether
`GaussianHMM` as a new dependency-adjacent code path is acceptable), stop and
ask rather than guessing. These were left open deliberately.
