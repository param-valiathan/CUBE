# Implementation Prompt 2 of 3: CUBE v6 — Environmental Context & Object Interaction

Copy the block below into a fresh Claude Code session (normal, non-plan-mode permissions) in the `d:\CUBE` working directory. This is prompt 2 of 3 in the CUBE v6 sequence — see `CUBE_v6_Implementation_Prompts_README.md` for the full sequence and shared safety contract. **This prompt has a hard prerequisite: prompt 1 (`CUBE_v6_Implementation_Prompt_1_Kinematics.md`) must have already landed and been reviewed.**

---

## Prompt

Implement `d:\CUBE\Environmental_Context_v6_Implementation_Plan.md`, Steps 1 through 7 only (Step 8 is intentionally out of scope — it now lives in `CUBE_Analyser_Paradigm_Reporting_Plan.md`, implemented by prompt 3). Read the full plan document before starting, including its "Opt-in guarantee" section — treat that section as binding, not aspirational. This plan is larger and touches more of the pipeline than prompt 1 did (a new GUI window, a new parallel data list, a new post-processing pass, a new output file, bout-CSV enrichment) — budget for it to take longer and to need more checkpoints.

### Step 0 — Prerequisite check and checkpoint

Before any other action:
1. **Verify prompt 1's deliverables actually exist.** Read `d:\CUBE\Kinematics_v6_Implementation_Report.md` if present; regardless, directly confirm in current source that `enrich_bouts_from_bin_source()` and `compute_bout_directedness()` exist in `cube_core.py`, and that `kinematic_directedness_enabled` is a key in `BSoidEngine.DEFAULTS`. **If any of these are missing, stop immediately and report it — do not build this plan's Step 6/7 against a guessed interface, and do not implement prompt 1's missing pieces yourself as a substitute.** This plan's Step 6 calls `enrich_bouts_from_bin_source()` by name with a specific argument shape assumed from prompt 1's actual implementation, not from the original plan doc's proposal.
2. Confirm `git status` is clean (stash or ask about anything uncommitted first). Create local git tag `pre-v6-env-context` at the current commit (which should be at or after prompt 1's final commit). Do not push it. State this tag name, plus the `pre-v6-kinematics`/`pre-v6-all` tags from prompt 1, at the top of the implementation report.

### Ground rules (safety-critical — follow exactly)

1. **Verify the environment first**: `"C:\Users\param\anaconda3\envs\CUBE\python.exe" -c "import umap; print('OK')"`. Stop and report on failure — no workarounds, no different interpreter.
2. **Re-verify every line-number/function-signature citation against current source before relying on it** — including citations into prompt 1's newly-added code, which the original plan doc could not have cited precisely since it was written before prompt 1 existed. Pay particular attention to `cube_core.py:~8336-8362` (Stage 1/2 loop), `cube_core.py:8836/8841-8850` (`session_bin_ranges.json` writer, the cited precedent location for the new `session_env_context.json` writer), `cube.py:3531`/`4143` (`AdvancedCUBEWindow`/`BodyPartWeightWindow`), and `cube.py:4634` (`CropPreviewDialog`).
3. **Everything is gated behind `env_features_enabled` (default `False`), with a critical second gate**: even with the flag on, `compute_session_env_context()` must produce an effective no-op (empty summaries, no crash) if `env_arena_cfg` has no traced shapes for a session. Both conditions — flag off, and flag-on-but-empty — must be exercised in verification, not just the flag-off case.
4. **Purely additive.** No modification of prompt 1's Step 2 (`enrich_bouts_from_bin_source`) internals — Step 6 here is a pure caller of it with a different data source/aggregator set, per the plan's own framing ("no new join logic written here"). If the existing function's interface genuinely cannot serve this plan's multi-region/multi-object needs as built, stop and report the mismatch rather than modifying prompt 1's function to fit — that is a cross-prompt design conflict for a human to resolve, not something to silently patch around.
5. **Work in phased checkpoints, Step 1 → 7, in order** (Step 1 data model → Step 2 GUI → Step 3 core wiring → Step 4 post-processing → Step 5 output → Step 6 bout enrichment → Step 7 approach/avoid detector). Commit locally after each step with a step-scoped message. Given Step 2's size (the `EnvContextWindow` GUI is explicitly flagged in the plan as "the single largest implementation item"), it is acceptable to split Step 2 into multiple sub-checkpoints (e.g. shape primitive + canvas first, then paradigm-conditional toolbar, then Tab 2's per-video alignment) — but do not merge Step 2 with Step 1 or Step 3 into one commit.
6. **The byte-identical regression check is mandatory** after Step 5 (first point a new output file could appear) and again after Step 7 (full scope landed): full pipeline run on the same fixture with `env_features_enabled=False`, before/after diff, byte-identical required. Additionally verify the "flag on, no shapes traced" no-op case produces the documented near-no-op behavior (empty summaries in `session_env_context.json`, not a crash, not spurious bout-CSV columns).
7. **Verify the shared-sidecar interaction with prompt 1 explicitly.** Run the pipeline with both `kinematic_directedness_enabled=True` and `env_features_enabled=True` (with shapes traced) simultaneously on a fixture, and confirm `*_bout_lengths_hmm_enriched.csv` correctly contains both plans' columns without a naming collision or row mismatch — this is explicitly called out as a required test case in this plan's Step 6 verification and is the main cross-prompt integration risk in this step.
8. **Coordinate-space correctness is a named top risk in `CUBE_v6_Feasibility_Study.md`.** Do not skip or weaken Step 4's coordinate-space assertion (pose data and resolved shapes must agree on crop state, fail loudly if not) — this is a correctness requirement, not defensive boilerplate to trim for simplicity.
9. **If you discover a real, pre-existing bug** in code this plan touches (including in prompt 1's newly-landed code), report it, don't fix it, unless it blocks this plan's own execution — in which case ask before proceeding.
10. **Test-suite integration, if `tests/` exists**: add real pytest cases for Step 4's point-in-polygon/nearest-edge/region-entry-sequence/role-based-index logic (pure-function, fixture-able, exactly the kind of CUBE-authored logic the two-tier philosophy calls for thorough coverage on) — this is the highest-value new logic in this plan to cover. GUI code (Step 2) is harder to unit-test and can rely on the plan's manual-QA verification steps instead; note this explicitly as a known gap rather than skipping silently. If `tests/` doesn't exist yet, do all verification manually/informally and say so, per this plan's own "Note on test-suite sequencing" section.
11. **Never touch real user data or `CUBE_logs/`.**
12. **Do not push to the git remote.**
13. **If you hit a genuine ambiguity the plan doesn't resolve** — e.g., exact widget layout details for `EnvContextWindow` beyond what the plan's table specifies, or an edge case in role-resolution precedence between reference roles and per-video `role_overrides` — pause and ask.

### Required deliverables when finished (or when stopping at whatever step you reach)

1. Steps 1-7 implemented per the plan's scope and each step's own verification criteria, in order.
2. **A detailed implementation report**, exported as `d:\CUBE\Environmental_Context_v6_Implementation_Report.md`, containing:
   - All checkpoint tag names (this prompt's and the inherited ones from prompt 1) at the top.
   - What was actually implemented per step, with file paths/function/class names for every new addition, including the exact final schema of `env_arena_cfg` and `session_env_context.json` as actually built (the plan's schema is a target, not guaranteed byte-identical to the final implementation — document any deviation explicitly).
   - Verification results per step, including both mandatory byte-identical regression checks (post-Step-5, post-Step-7) and the dual-flag sidecar interaction test (ground rule 7).
   - Any bugs/inconsistencies found — reported, not fixed, per ground rule 9.
   - Any deviations from the plan and why.
   - Explicit confirmation of what `compute_session_env_context()` actually outputs (per-bin time series names, session-summary field names, which paradigm-specific derived metrics were actually implemented vs. skipped) — prompt 3 depends on this being accurate, since it builds the display layer directly on top of this output.
   - Known gaps or anything deliberately left uncovered.
3. **Documentation updates per the plan's "Documentation follow-up" section**: update `README.md`, `CUBE_GUIDE.md`, and `GROUP_PREDICTOR_REFERENCE.md` (only if any Step 8-adjacent Group Predictor content was touched — it should not have been, since Step 8 is out of scope for this prompt; if so, note why), then run `md_to_docx.py`. Update `CLAUDE.md` directly — add `session_env_context.json` and the shared enriched-bout sidecar to "Output directory layout," and mention `env_features_enabled`/`env_arena_cfg`/`env_interaction_threshold` near `DEFAULTS`.
4. **A final plain-text summary message** — completed steps, deferred/incomplete steps, bugs found, checkpoint tag names, and explicit confirmation of the dual-flag sidecar test result.

---

## Notes for whoever runs this prompt

- This is the largest of the three prompts. If it needs to span multiple sessions, stop at a Step boundary (never mid-step) and resume from `Environmental_Context_v6_Implementation_Report.md`'s "what was actually implemented" section in a fresh session.
- Prompt 3 (`CUBE_v6_Implementation_Prompt_3_Analyser_Reporting.md`) has a hard dependency on this prompt's Steps 4-7 landing with the output schema documented in the report — review this prompt's output before starting prompt 3.
- T-Maze and Morris Water Maze remain deferred per the plan's own scope decision — do not implement them even partially "while in the area."
