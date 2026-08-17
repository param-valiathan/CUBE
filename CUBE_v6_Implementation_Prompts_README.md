# CUBE v6 Implementation Prompts — How To Use

Three self-contained prompts implement CUBE v6 in the dependency order the design docs require. **Run them one at a time, in order, each in a fresh Claude Code session (normal, non-plan-mode permissions) in the `d:\CUBE` working directory.** Do not start prompt *N+1* until prompt *N*'s deliverables have been reviewed and accepted — each prompt's first ground rule is to verify the previous prompt's output actually exists before touching anything.

| Order | Prompt file | Implements | Depends on |
|---|---|---|---|
| 1 | `CUBE_v6_Implementation_Prompt_1_Kinematics.md` | `Kinematic_Transition_v6_Implementation_Plan.md`, Steps 1-4 | Nothing (first) |
| 2 | `CUBE_v6_Implementation_Prompt_2_Environmental_Context.md` | `Environmental_Context_v6_Implementation_Plan.md`, Steps 1-7 | Prompt 1 landed |
| 3 | `CUBE_v6_Implementation_Prompt_3_Analyser_Reporting.md` | `CUBE_Analyser_Paradigm_Reporting_Plan.md` (all sections) | Prompts 1 and 2 landed |

## Why this order, and why not one giant prompt

- The environmental-context plan reuses two things the kinematics plan builds (`enrich_bouts_from_bin_source()`, `compute_bout_directedness()`) — building it first would mean guessing at an interface that gets built properly one prompt later.
- The analyser-reporting plan reads output files (`session_env_context.json`, the enriched bout sidecar, `approach_events.csv`) that only exist once the first two prompts have actually run end-to-end — there is nothing for it to display before then.
- Each prompt is scoped to be reviewable as one unit of work: a real diff a human can read in one sitting, with its own regression check and its own report. Collapsing all three into one prompt would remove the checkpoint where a human looks at prompt *N*'s actual behavior before prompt *N+1* builds on top of it.

## Common safety contract across all three prompts

Every prompt below repeats these explicitly, but the shared logic is:

0. **Checkpoint the current working state before touching anything, every time.** Before Step 1 of *any* of the three prompts — including prompt 1 on a clean `main` — create a git tag marking the exact known-good commit to return to (e.g. `git tag pre-v6-kinematics`, `pre-v6-env-context`, `pre-v6-analyser-reporting`, one per prompt, plus a fallback `pre-v6-all` before prompt 1). This is the actual rollback mechanism, not just documentation of intent: if a prompt's regression check fails partway through and the cause isn't quickly obvious, the correct recovery is `git reset --hard <tag>` (after confirming with the user, per `CLAUDE.md`'s general rule on destructive git operations) or simply checking out the tag into a scratch worktree to compare against, not attempting to hand-unwind a partially-applied change. Push the tag nowhere — it's a local-only safety net, same as the "no push" rule below. State the exact tag name created at the start of the implementation report so a human reviewing the work later knows exactly what to roll back to and from where.
1. **Verify the `CUBE` conda environment first**, exactly as `CLAUDE.md` requires — never bare `python`/`pip`/`conda activate`.
2. **Verify the prior prompt's deliverables actually exist** (specific functions/flags/files named in each prompt) before writing anything — if a prerequisite is missing, stop and report rather than building around the gap or guessing at what it should look like.
3. **Everything shipped by these three plans is opt-in and off by default.** The enforcement mechanism is a mandatory byte-identical regression check (full pipeline run, flags at default, before/after diff) — this must pass before any prompt is considered done, not just before a plan doc is considered followed.
4. **Purely additive** — no modification of existing default-path behavior, ever, in any of the three prompts.
5. **Phased checkpoints matching each plan's own Step numbering** — implement one step, verify it, commit locally with a step-scoped message, then move to the next. Never one undifferentiated diff for a whole prompt.
6. **Report bugs found, don't fix them** — unless they block executing this specific plan, in which case ask first.
7. **Re-derive line numbers/signatures from current source** before relying on any plan doc's citations — the codebase may have moved since a doc was written, including by a prior prompt in this same sequence.
8. **No push to remote.** Commit locally only, using the VS-bundled git per `CLAUDE.md`.
9. **Ask, don't guess**, on any genuine ambiguity the plan doesn't resolve.
10. **Deliverables**: a dedicated implementation report per prompt, the documentation updates each plan's own "Documentation follow-up" section specifies (plus `md_to_docx.py` rerun), and a plain final summary message — not just a report file nobody reads.

## After all three land

Once all three prompts are complete and reviewed, `CLAUDE.md`'s "Architecture Overview" section should get a short new subsection describing the v6 opt-in features (kinematic directedness, environmental context, paradigm reporting) at the same level of detail as the existing "Planned 3D dual-camera extension" note — this is a documentation step for a human/future session to do once the real, landed behavior is known, not something to speculatively write now.
