# CUBE v6: Feasibility Study — Publication and Field Adoption

## Scope and purpose

`Competitive_Positioning_and_Publication_Pathway.md` assessed CUBE's standing against peer tools at an architectural level, before the two v6 implementation plans existed in their current, much more concrete form. Since then, `Kinematic_Transition_v6_Implementation_Plan.md` and `Environmental_Context_v6_Implementation_Plan.md` have grown substantially through iterative review — a paradigm-aware GUI, role-tagging, per-video overrides, six paradigm-specific derived-metric families, a new kinematic directedness/looping metric set. This study re-assesses feasibility against that mature, detailed scope specifically: is this actually buildable at reasonable risk, is it scientifically soundable, will labs actually use it, and does it meaningfully improve CUBE's publication case. It does not re-derive the competitive landscape (see the prior document for that) — it focuses on what changed once the plans got concrete.

---

## Part 1 — Technical feasibility

### What got bigger through review, and what that costs

The scope of both v6 plans roughly doubled through the review process in this conversation: the environmental-context plan alone went from "trace a rectangle" to a paradigm-selector screen, seven paradigm presets, a role-tagging system with paradigm-specific vocabularies, per-video role/correct-arm overrides, minimum-shape validation, and four families of paradigm-gated derived metrics (alternation %, discrimination index, sociability indices, MWM strategy classification with configurable thresholds). Each addition was individually well-justified — every one closed a real correctness or usability gap found by checking the plan against actual experimental paradigms — but the cumulative effect is that this is now a substantially larger engineering effort than the original architecture document scoped, concentrated almost entirely in one file (`cube.py`'s new `EnvContextWindow`) and one function (`cube_core.py`'s `compute_session_env_context()`).

**This is a real feasibility concern, not just a bigger number.** A single new GUI window now needs to correctly handle: paradigm switching without data loss, seven different toolbar/naming configurations, role dropdowns with paradigm-specific vocabularies, a two-tab structure with crop-aware frame sampling, translate-only rigid adjustment, a per-shape override escape hatch, per-video role and correct-arm overrides, minimum-shape validation warnings, and template import/export. Each piece is individually simple; the integration surface between all of them is where bugs will concentrate, and it is one continuous piece of new GUI code with no existing precedent in the codebase at this level of conditional complexity (the closest precedent, `CropPreviewDialog`, does exactly one thing).

### Most fragile points, ranked by silent-failure risk

The riskiest bugs in this plan are not crashes — they're **silent wrong output**, since a misconfigured pipeline still runs, still produces plausible-looking numbers, and nothing tells the user those numbers are wrong:

1. **Coordinate-space misalignment** (crop-configured video, shapes traced without accounting for it). If this fails, every distance/region computation for that session is wrong by a fixed offset, with no visible symptom in the output — it would look like a normal, slightly-odd result rather than an error. This is the single highest-priority thing to get an automated test around before shipping (see Part 4).
2. **Role misassignment** (wrong object tagged "novel," wrong region tagged "hub" instead of "arm"). Produces a plausible-looking but backwards discrimination index or alternation percentage. Nothing in the pipeline can detect this from the data alone — it's a pure user-input-trust problem, mitigated only by good default suggestions and the minimum-shape validation warnings already designed in, neither of which catches a role assigned to the *wrong* shape (only a role left *unassigned*).
3. **Per-video transform drift**. If a user nudges a video's shapes to "look right" on a single frame that happens to be a bad sample (e.g., a moment where the animal partially occludes the arena edge), the whole session's data shifts based on one visual judgment call with no cross-check.
4. **Rule-based classifier thresholds** (MWM strategy classification, Phase 3.5's approach/avoid detector) producing confident-looking labels on genuinely ambiguous behavior — a known, accepted limitation of any rule-based heuristic, but worth naming explicitly as a technical (not just scientific) risk, since a threshold that's slightly wrong for a given lab's rig/camera/species doesn't fail loudly, it just mislabels quietly.

### Engineering effort realism

Both plans' individual "Effort" ratings are locally reasonable, but summed across all steps this is a multi-week to multi-month single-developer effort at minimum, before any of the deferred v7 items. Given `CLAUDE.md`'s own note that CUBE currently has no automated test suite and this developer appears to be working largely solo, **the realistic risk is not "can this be built" but "can this be built correctly and stay correct as it grows."** The `Automated_Test_Suite_Plan.md` (still entirely unbuilt as of this study) becomes materially more urgent given how much new silently-fragile logic (coordinate resolution, role resolution, paradigm gating) v6 is adding — this was true before, but the case has gotten stronger, not weaker, through this review process.

**Technical feasibility verdict: buildable, but only safely if sequenced carefully.** Recommend building `Automated_Test_Suite_Plan.md`'s Phase T1 (bootstrap) and at minimum a coordinate-space/role-resolution test suite *before or alongside* `Environmental_Context_v6_Implementation_Plan.md`'s Step 4, not after — this is the one piece of technical sequencing advice this study adds beyond what's already in the plans.

---

## Part 2 — Scientific/analytical validity feasibility

This is the part most distinct from a pure engineering feasibility check, and the part most relevant to "for publication."

### What's mechanically correct vs. what's scientifically validated

Most of what v6 computes is **mechanically well-defined and essentially guaranteed correct if implemented as specified** — time-in-region, distance-to-object, spontaneous alternation percentage (a long-standing, precisely-defined formula), discrimination index (likewise standard). These are safe to publish as long as the implementation is tested (Part 1).

**Two pieces are categorically different: rule-based heuristic classifiers.** MWM strategy classification (directed_swim/wall_hugging/chaining) and Phase 3.5's approach/avoid sequence detector are both threshold-based interpretations of continuous kinematic signals, not standard, externally-validated formulas. This matters specifically for publication: a reviewer will ask "how do you know your chaining classifier actually identifies chaining, and not just any low-straightness, high-rotation swim path that happens not to be chaining?" The only real answer is a validation study — manually annotate a sample of real MWM sessions with expert-labeled strategies, compare against the classifier's output, report agreement (the same validation-module pattern eLife's `BehaviorDEPOT` used: precision/recall/F1 against held-out human annotation, cited in `Competitive_Positioning_and_Publication_Pathway.md`). **Without this, these two features are usable internally but not publication-ready as scientific claims** — they can be shipped and used, but any paper built around them needs this validation step regardless of how well-designed the rule logic is.

### Risk of over-claiming

There is a specific, named risk worth flagging: the temptation to describe CUBE v6 in a paper as "automatically classifies MWM search strategies" without the validation study above, because the feature genuinely does run and produce labels. Running and being validated are different claims, and conflating them is exactly the kind of thing that draws a sharp reviewer response. The mitigation is procedural, not technical: treat the rule-based classifiers as internally-useful, externally-unvalidated until a specific validation pass is done, and be explicit about that distinction in any manuscript.

### What this doesn't change

The core algorithmic critique from `Competitive_Positioning_and_Publication_Pathway.md` (CUBE's clustering is still frame/bin-HDBSCAN, the same family the field has already benchmarked against keypoint-MoSeq and found produces less temporally-realistic states) is completely unaffected by either v6 plan. Nothing here touches that; it remains the deepest, hardest-to-close gap, orthogonal to everything in this study.

---

## Part 3 — Field/adoption feasibility

### Genuine usability wins

The paradigm-selector design is a real, tangible improvement for lab adoption, not just a nicer UI. For a lab running Y-maze experiments, the difference between "here's a generic polygon tool, go define your own regions and figure out the analysis yourself" and "select Y-Maze, trace three arms with names pre-filled, get alternation % automatically" is the difference between a tool that requires a technical champion to operate and one a grad student can pick up directly. This is a real advantage specifically because CUBE already had the harder problem solved (unsupervised motif discovery) — bolting a well-designed paradigm layer on top converts that from "impressive but requires customization" into "usable out of the box for six common experiments," which is a meaningfully different adoption proposition.

The template import/export feature (reuse a traced arena across projects) directly targets the dominant real-world friction pattern (one physical rig, many experiments over time) rather than a hypothetical one.

### Genuine adoption risks

- **Complexity creep undermines the stated design goal.** The explicit ask throughout this review was to keep the GUI "as simplified as possible" — and each individual addition (role dropdowns, per-video overrides, correct-arm designation, minimum-shape warnings) was justified as necessary, but a first-time user opening `EnvContextWindow` for a Three-Chamber session now encounters: paradigm selection, two tabs, primary/Advanced tool distinction, role dropdowns on both regions and objects, and (if running phase 2) a per-video role-override screen. This is objectively more UI than "trace three boxes," even though every piece of it is there for a real reason. The risk isn't that any single addition was wrong — it's that the sum may not feel "as simplified as possible" to an actual first-time user, and there's no user-testing step anywhere in either plan to check this before it ships.
- **Competing with established, purpose-built commercial tools.** For a lab running exactly one of these six paradigms (a common case), commercial systems (e.g. ANY-maze, EthoVision) already do zone/object tracking, alternation %, and discrimination index natively, with support contracts and validated defaults. CUBE's differentiation is the *combination* with unsupervised motif discovery — but a lab that only wants the paradigm-standard metrics, without caring about discovering novel behavioral motifs, has no strong reason to switch. The realistic adoption path is labs that already use (or want) CUBE for its core clustering and now get paradigm-aware context "for free," not labs shopping for a best-in-class Y-maze tracker in isolation.
- **Documentation/training burden.** Six paradigm workflows, role semantics, the crop-coordinate-space requirement, per-video adjustment mechanics — this is a lot of new surface for `CUBE_GUIDE.md` to cover well. A user who misunderstands the role system silently gets wrong output (Part 1) rather than an error, which raises the stakes on documentation quality specifically here, more than for most other CUBE features.

**Field-adoption verdict: strong genuine value for existing/prospective CUBE users running one of these six paradigms; limited pull for users not already interested in unsupervised discovery; real risk that the final UI doesn't land as "simple" without an explicit usability-testing pass that isn't currently in either plan.**

---

## Part 4 — Publication feasibility (updated from the prior positioning document)

### What's improved since the last assessment

`Competitive_Positioning_and_Publication_Pathway.md` argued CUBE's strongest differentiation claim would be "unsupervised motif discovery with environmental/object context layered on top" — a real niche, but abstract at the time. The mature v6 plans make that claim concrete and demonstrable: six named, well-specified paradigm workflows with tailored, standard metrics is a much stronger, more specific "here's what this tool does that others don't" story than the original architecture document could support. This is a genuine improvement to the publication narrative, not just more engineering.

### What hasn't improved

Every publication blocker identified in the prior document is still exactly as blocking: no test suite, no packaging, no benchmark comparison against B-SOiD, no demonstrated biological finding. If anything, **the test-suite gap is now higher-stakes**, because v6 introduces more silently-fragile logic (Part 1) that a reviewer-facing correctness claim depends on. Shipping v6 without at least the coordinate-space/role-resolution test coverage from Part 1 and makes any resulting paper's correctness claims weaker, not stronger, regardless of how sophisticated the feature set looks in the manuscript's methods section.

### The validation-study requirement is now paradigm-specific, and cheaper than it looks

Because v6 targets six named, standard paradigms, the validation story for a methods paper doesn't need to be a single monolithic benchmark — it can be six small, focused validation vignettes (e.g., "in N=X Y-maze sessions, CUBE-computed alternation % agreed with manual scoring at r=..."), each individually modest in scope. This is actually an easier path to a credible validation section than the original architecture document implied, precisely because the paradigm-aware design gives natural, well-understood ground-truth comparisons to run (manual alternation scoring, manual discrimination-index scoring, expert MWM strategy labeling) rather than needing to invent a validation methodology from scratch.

**Publication feasibility verdict: the differentiation story is now genuinely stronger and more concrete than before, but the mechanical blockers (tests, packaging, benchmark comparison) are unchanged, and the rule-based classifiers specifically add a new validation requirement that didn't exist in the pre-v6 architecture. Net effect: better narrative, same amount of blocking infrastructure work, one new validation task.**

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Coordinate-space misalignment (crop vs. traced shapes) produces silently wrong distances | Medium | High — wrong science, no visible symptom | Automated test asserting the invariant (Part 1); build before/alongside Env plan Step 4 |
| Role misassignment (wrong object/region tagged) produces backwards derived metrics | Medium | High — wrong science, no visible symptom | Good default naming suggestions (already designed); clear documentation; consider a visual role-confirmation summary before finalizing a paradigm config |
| Rule-based classifiers (MWM strategy, approach/avoid) published without validation | Medium-High (temptation is real once the feature "works") | High — reviewer credibility risk | Explicit internal-use-only labeling until a validation study is run; treat validation as a hard prerequisite for any manuscript claim, not optional polish |
| GUI complexity undermines the "as simple as possible" goal | Medium | Medium — adoption friction, not correctness | A usability pass with 2-3 real users before wide release; not currently in either plan as an explicit step |
| Scope grew substantially through review; effort/timeline underestimated | High (already observed) | Medium — delayed delivery, not wrong output | Sequence ruthlessly: kinematics plan → environmental plan's Mode-A core (Steps 1-6, 8) → Phase 3.5/paradigm-specific metrics → defer Mode B/Option B further if timeline pressure appears |
| No test suite exists while fragile new logic is being added | High | High — compounds every other risk above | Treat `Automated_Test_Suite_Plan.md` Phase T1 + targeted coverage of Part 1's fragile points as a co-requisite of v6, not a separate later initiative |
| CUBE remains architecturally the same clustering family the field has already found weaker on state-duration realism | Certain (unchanged by v6) | Medium for publication, low for field use | Out of scope for v6; a distinct, larger future initiative if pursued at all |

## Advantages register

| Advantage | Beneficiary |
|---|---|
| Six concrete, tailored paradigm workflows with standard metrics computed automatically | Field adoption — converts a generic tool into an out-of-the-box solution for common experiments |
| Unsupervised discovery + paradigm-aware context is a real, underserved niche vs. every surveyed peer tool | Publication — the core differentiation claim, now demonstrable rather than abstract |
| Template reuse removes the dominant real friction point (same rig, many experiments) | Field adoption — ongoing time savings for repeat users |
| Role-tagging/per-video-override infrastructure is general (built once, reused across Novel Object, Three-Chamber, T-maze) | Engineering — the shared-mechanism discipline established throughout this review keeps the marginal cost of each paradigm low |
| The shared `enrich_bouts_from_bin_source()` utility (kinematics plan) benefits any future bout-level analysis feature, not just these two plans | Engineering — durable infrastructure investment beyond v6's immediate scope |
| Six small, paradigm-specific validation vignettes are individually cheaper than one monolithic benchmark | Publication — a more tractable path to a credible validation section |
| Position-dominance risk was designed out structurally (Mode A), not defaulted-off and hoped for | Scientific integrity — avoids the single most-cited failure mode in the literature this was built on |

---

## Overall verdict

**Feasible, with the sequencing caveat already stated in each part above being the difference between a genuine advance and a maintenance liability.** The engineering is buildable but has grown large enough that it needs the test-suite work pulled forward rather than left for later; the standard metrics (alternation, discrimination index) are publication-safe as specified; the rule-based classifiers (MWM strategy, approach/avoid) are field-usable now but need a dedicated validation pass before appearing as claims in a manuscript; the field-adoption case is genuinely strong for labs already interested in CUBE's core clustering, weaker as a standalone pitch to labs shopping only for paradigm-standard tracking; and the publication narrative is now concrete and differentiated in a way it wasn't before this round of design work, though none of the mechanical publication blockers (tests, packaging, benchmark comparison) have been reduced.

## Related documents

- `Competitive_Positioning_and_Publication_Pathway.md` — the original competitive/publication analysis this study updates.
- `Automated_Test_Suite_Plan.md` — now a co-requisite of v6, not a parallel-track nicety, per Part 1 and the risk register.
- `Kinematic_Transition_v6_Implementation_Plan.md`, `Environmental_Context_v6_Implementation_Plan.md` — the plans this study assesses.
