# Implementation Prompt 1 of 3: CUBE v6 — Kinematic Directedness

Copy the block below into a fresh Claude Code session (normal, non-plan-mode permissions) in the `d:\CUBE` working directory. This is prompt 1 of 3 in the CUBE v6 sequence — see `CUBE_v6_Implementation_Prompts_README.md` for the full sequence and shared safety contract. This prompt has no prerequisite prompt (it goes first).

---

## Prompt

Implement `d:\CUBE\Kinematic_Transition_v6_Implementation_Plan.md`, Steps 1 through 4 only (Step 5 is intentionally out of scope — it now lives in `CUBE_Analyser_Paradigm_Reporting_Plan.md`, implemented by a later prompt). Read the full plan document before starting, including its "Opt-in guarantee" section — treat that section as binding, not aspirational.

### Step 0 — Checkpoint

Before any other action: confirm `git status` is clean (stash or ask about anything uncommitted first, per `CLAUDE.md`'s destructive-operation guidance), then create two local git tags: `pre-v6-all` and `pre-v6-kinematics`, both pointing at the current commit. Do not push them. State both tag names at the top of the implementation report (see Deliverables) so a human knows exactly what to `git reset --hard` to if something goes wrong later in this prompt or in prompts 2/3.

### Ground rules (safety-critical — follow exactly)

1. **Verify the environment before touching anything.** Run `"C:\Users\param\anaconda3\envs\CUBE\python.exe" -c "import umap; print('OK')"`. If it fails, stop and report — do not use a different interpreter or attempt a workaround. Every command in this session uses this exact interpreter path.
2. **Re-verify every line-number/function-signature citation in the plan against actual current source before relying on it.** The plan was written against a snapshot of the codebase; confirm `attach_centroid_distance` (cited at `cube_core.py:2000-2035`), `labels_to_bouts()` (cited at `cube_core.py:3835-3856`), `compute_cluster_kinematics` (cited at `cube_core.py:3887-3944`), and `BSoidEngine.DEFAULTS` (cited at `cube_core.py:~7459-7841`) are still where the plan says before writing code that depends on their exact location or signature.
3. **This is purely additive, gated behind `kinematic_directedness_enabled` (default `False`) for every step except Step 1.** Step 1 (wiring `compute_cluster_kinematics`'s existing CSV into `cube_analyser.py`) is the sole exception explicitly called out in the plan as safe to run unconditionally — everything else must not execute, and must not change any existing output file, when the flag is at its default. Do not modify `attach_centroid_distance` itself — the new join utility (Step 2) is a separate sibling function.
4. **Work in phased checkpoints, one Step at a time (1 → 2 → 3 → 4).** After each step: implement it, verify it per that step's own "Verification" bullet in the plan, commit locally with a step-scoped message (e.g. `feat(v6-kinematics): step 2 — enrich_bouts_from_bin_source join utility`), before moving to the next step.
5. **The byte-identical regression check is mandatory, not optional**, and must be run at minimum once after Step 4 lands (the point where `kinematic_directedness_enabled` first gates real pipeline output): run the full pipeline twice on the same fixture — once on the pre-work state (check out the `pre-v6-kinematics` tag into a scratch location, or diff against output captured before starting), once after this prompt's changes with the flag left at its default `False` — and diff every output file byte-for-byte. They must be identical. If they are not, this is a blocking bug: stop, do not proceed to Step 3/4's remaining scope or to a later prompt, and report it clearly.
6. **If you discover a real, pre-existing bug unrelated to this plan** (e.g. in `attach_centroid_distance`, `labels_to_bouts()`, or `compute_cluster_kinematics` while reading them), do not fix it. Report it with a minimal reproduction in the implementation report and leave it untouched unless the user explicitly asks for a follow-up fix.
7. **Test-suite integration, if `tests/` exists**: check for a `tests/` directory and `Automated_Test_Suite_Plan.md`/`Test_Suite_Implementation_Report.md` first. If pytest infrastructure already exists, add real test cases for Step 2's join utility and Step 3's directedness computation (both are pure-function, easily fixture-able CUBE-authored logic — exactly the kind of thing that plan's two-tier philosophy calls for thorough coverage on) under the appropriate `tests/unit/` location, and run the fast subset (`-m "not slow"`) before and after your changes to confirm no regression. If `tests/` does not exist yet, do the plan's "synthetic test" verification steps manually/informally and say so explicitly in the report — do not block on building test infrastructure that's a separate, undelivered plan (`Automated_Test_Suite_Plan.md`).
8. **Never touch real user data or `CUBE_logs/`.** All fixtures for verification (synthetic trajectories, synthetic bout tables) must be constructed in-memory or under a scratch/test directory.
9. **Do not push to the git remote.** Commit locally only, using the VS-bundled git per `CLAUDE.md`'s git workflow section.
10. **If you hit a genuine ambiguity the plan doesn't resolve** — e.g., the exact aggregator-parameterization API shape for Step 2's dict-of-aggregators case, or exactly where in the Stage 8 export call sites to hook Step 3/4 — pause and ask rather than guessing silently and hoping it matches what prompt 2 will need.

### Required deliverables when finished (or when stopping at whatever step you reach)

1. Steps 1-4 implemented per the plan's scope and each step's own verification criteria, in order.
2. **A detailed implementation report**, exported as `d:\CUBE\Kinematics_v6_Implementation_Report.md`, containing:
   - The `pre-v6-kinematics`/`pre-v6-all` tag names (Step 0) at the top.
   - What was actually implemented per step, with file paths and function names for every new function/column/flag added.
   - Verification results per step, including the mandatory byte-identical regression check's outcome (pass/fail, and what was diffed).
   - Any bugs/inconsistencies found in existing code — reported, not fixed, per ground rule 6.
   - Any deviations from the plan and why (e.g., a cited line number had moved, a function signature differed from what the plan assumed).
   - Explicit confirmation of the interface shape for `enrich_bouts_from_bin_source()` (Step 2) and `compute_bout_directedness()` (Step 3) — prompt 2 depends on both by name and signature, so this report is the source of truth for what prompt 2 should expect to import.
   - Known gaps or anything deliberately left uncovered.
3. **Documentation updates per the plan's "Documentation follow-up" section**: update `README.md` (new sidecar CSV, new `kinematic_directedness_enabled` Advanced Setting) and `CUBE_GUIDE.md`, then run `"C:\Users\param\anaconda3\envs\CUBE\python.exe" d:\CUBE\md_to_docx.py`. Update `CLAUDE.md` directly (no docx regeneration needed for it) — add the new sidecar file to "Output directory layout" and mention `kinematic_directedness_enabled` near `DEFAULTS`.
4. **A final plain-text summary message** at the end of your work — completed steps, deferred/incomplete steps, bugs found, and the checkpoint tag name — not buried only in the report file.

---

## Notes for whoever runs this prompt

- Expect this to take multiple checkpoints given the phased-step requirement in ground rule 4.
- Prompt 2 (`CUBE_v6_Implementation_Prompt_2_Environmental_Context.md`) has a hard dependency on this prompt's Steps 2 and 3 landing correctly with the interface shape documented in the report — review this prompt's output and report before starting prompt 2.
