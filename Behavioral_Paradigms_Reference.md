# CUBE Behavioral Paradigms Reference

### Data requirements, staging, use cases, and limitations for the Analyser's Paradigm Results tab

This document is the scientific reference for CUBE's Paradigm Results tab (`cube_analyser.py`,
`ParadigmResultsPanel`) — what data and tracing each paradigm-specific view requires, at which pipeline
stage, what each graph answers, and where the analysis is known to be limited. It is a companion to
`README.md` (feature summary), `CUBE_GUIDE.md` (step-by-step usage), and `CUBE_ANALYSIS_METHODOLOGY.md`
(clustering methodology) — this document covers only the paradigm-reporting layer built on top of the
Environmental Context feature (`env_features_enabled`) and, for two graphs, the Kinematic Directedness
feature (`kinematic_directedness_enabled`). Implementation detail beyond what a user/reviewer needs is in
`Environmental_Context_v6_Implementation_Report.md`, `Kinematics_v6_Implementation_Report.md`, and
`Analyser_Paradigm_Reporting_v6_Implementation_Report.md`.

---

## 1. Prerequisites — what must exist before the tab shows anything

The Paradigm Results tab reads three files a pipeline run writes under `<output_dir>/`:
`model/session_env_context.json`, `bout_lengths/*_bout_lengths_hmm_enriched.csv`, and
`bout_lengths/*_approach_events.csv`. None of these are written unless the corresponding pipeline flags
were set **before that run**, and — critically — the arena/region/object tracing itself is a **project-wide,
pre-run configuration step**, not something done after the fact in the Analyser. A session run without
`env_features_enabled=True` cannot retroactively gain occupancy/region data by opening it in the Paradigm
Results tab; the pipeline must be re-run.

| Stage | Action | Governs |
|---|---|---|
| 1. Before Step 3 (Clustering Engine) | Main window → **Environments, Objects, Paradigms...** → tick `env_features_enabled` → **Configure Arena, Regions & Objects...** → `EnvContextWindow` | Master switch; without this, no environmental data is ever computed, regardless of what happens later |
| 2. Inside `EnvContextWindow`, Tab 1 | Choose one paradigm (project-wide — see §5.1) and trace a boundary + named regions/objects with roles, on **one reference video frame** | `env_arena_cfg["reference_shapes"]` — the shape set every session inherits by default |
| 3. Inside `EnvContextWindow`, Tab 2 (optional) | Per-video translate or independent override, if the arena moved or was repositioned between recordings | `env_arena_cfg["per_video"]` — per-session correction on top of the reference shapes |
| 4. Same **Environments, Objects, Paradigms...** window | Tick `kinematic_directedness_enabled` | Required for: total-distance-traveled (Open Field control bar), and *entirely* required for Approach/Avoid Events (needs `straightness_ratio`/`mean_speed_px_s` per bout) |
| 5. Run the pipeline (Step 3) | `BSoidEngine.run()` | Writes `session_env_context.json`, the enriched bout sidecar, and (if both flags 4a/4b are on and events are detected) `approach_events.csv` |
| 6. Analyser, Combined Analysis tab | Load the resulting `*_bout_lengths_hmm.csv` files as usual | Populates `get_animals_fn` — the Paradigm Results tab locates its sibling files from each animal's bout-CSV path |
| 7. Analyser, Paradigm Results tab | Click **Refresh from Loaded Sessions** | Loads and renders |

A session loaded without step 1 (or from a pre-v6 run) shows "no environmental context data for any loaded
session" for every sub-view — this is the documented, tested graceful-degradation path, not an error.

---

## 2. Shared components (every sub-view uses these)

### 2.1 Occupancy heatmap

A 2D histogram (`numpy.histogram2d`, 40×40 bins by default, `viridis` colormap) of the animal's centroid
position (mean of all tracked bodypart x/y coordinates), pooled across every currently-loaded session that
has environmental data. This is the position/spatial figure for every sub-view.

**Region/object outlines drawn on top are an approximation, not the traced shape.** `env_arena_cfg`'s
traced polygon vertices are never written to any per-run output file — they exist only in `cube.py`'s
session-local GUI state — so `cube_analyser.py` has no way to draw the literal traced boundary. The outline
shown is instead the **convex hull of the centroid positions actually recorded as belonging to that
region** (via `current_region`) or **as interacting with that object** (nose/paw within
`interaction_threshold_px`). This is a real, disclosed limitation: a convex hull is by definition a
superset of an irregular or concave true shape (e.g. an L-shaped chamber, or a Y-maze's non-convex arm
junction), and depends on the animal having actually visited enough of the true boundary to approximate its
extent. Labeled `"(approx.)"` in every plot title/annotation. **Do not use the drawn outline itself as a
precise spatial reference** — use it only as a rough visual anchor for where a region sits relative to the
density map.

### 2.2 Arena/region cluster-and-group cross-tab

Two grouped bar charts, built once and shown on every sub-view: **cluster-by-region** (% of time each
HDBSCAN/MLP/HMM behavioral cluster spends in each named region, weighted by bout duration) and
**group-by-region** (the same, split by experimental group instead of cluster). This answers "where in the
arena does behavior X happen" and "does the treatment group occupy different regions" directly, without
requiring the user to manually cross-reference the per-cluster and per-region summary numbers. Both are
derived from the enriched bout sidecar's `pct_time_in_region_<name>` columns (Environmental Context Step 6),
not recomputed from raw position — so their accuracy is bounded by that column's own accuracy, not an
independent measurement.

### 2.3 Time-course, ethogram, distribution, and paired plots

A second wave of shared components (added after the initial seven-sub-view build), all built from data
already present in `session_env_context.json` — no pipeline/schema change was needed for most of them, since
`compute_session_env_context()` already tracked nose-point distance and per-bin centroid position. Four new
`region_roles`/`object_roles`/`dist_to_region_boundary_nose`/`nose_outside_boundary`/`object_interaction_
bout_lengths_sec` keys were added additively (existing keys unchanged) to unlock the rest — see
`Environmental_Context_v6_Implementation_Report.md`'s addendum for the exact schema.

- **Time-course plots** (`build_timecourse_figure`): mean ± SEM line per experimental group, x-axis = % of
  trial elapsed (not absolute time, so sessions of different duration still align on one shared axis).
  Used for: Open Field center-zone %, EPM open-arm %, Discrimination Index / Sociability investigation
  distance to the role-tagged object, and CPP cumulative chamber crossings.
- **Region ethogram strip** (`build_region_ethogram_figure`): one horizontal row per session, colored by
  which region the animal occupied over time — the Y-Maze arm-entry sequence view, mirroring the existing
  behavior-ethogram's broken-barh visual language.
- **Bout-duration distribution** (`build_distribution_figure`): overlapping per-group histograms + a rug of
  individual bout durations underneath — each point is one investigation bout, not one animal. Used for
  Discrimination Index and Sociability's object-interaction bouts, and Open Field's per-animal mean-speed
  distribution.
- **Paired pre/post dumbbell** (`build_paired_dumbbell_figure`): one connecting line per matched subject
  (Repeated Measures design only) — CPP's pre → post preference-index shift, the standard CPP figure showing
  individual-animal change rather than only a summary delta.
- All bar charts in this tab (`_index_bar_fig` and the region/group cross-tab) additionally overlay each
  individual animal's own value as an unfilled ring on top of its group's bar — not just the mean ± SEM.

### 2.4 Statistical conventions

- **Between-group comparisons** (does the metric differ across experimental groups) reuse
  `run_cluster_statistics()` — Kruskal-Wallis omnibus + Dunn's/Mann-Whitney post-hoc for ≥3 groups, both
  Benjamini-Hochberg FDR-corrected — the identical machinery Unbiased Analytics uses for cluster metrics,
  adapted to a paradigm-level scalar index (alternation %, discrimination index, etc.) via a single-row
  wrapper (`_scalar_metric_compute_fn`) rather than a parallel implementation. Shown as pink/red brackets
  with `*`/`**`/`***` significance stars.
- **One-sample-vs-reference tests** (does this group's index differ from a fixed expectation — chance level
  for Discrimination Index, zero for CPP's delta score) use a paired-t-test or one-sample Wilcoxon
  signed-rank test (`run_one_sample_statistics`), independently BH-FDR corrected, shown as cyan
  `†`/`††`/`†††` dagger markers — a deliberately different visual convention from the star brackets above,
  so a reader cannot mistake "differs from chance" for "differs between groups." These are two different
  scientific questions and this tab never overlays them on the same symbol.
- **Repeated-measures/paired comparisons** (CPP pre- vs. post-test) reuse the exact same "Experimental
  Design" (Independent/Repeated Measures) and "Group by" controls Unbiased Analytics exposes — Wilcoxon
  signed-rank (2 levels) or Friedman's test (≥3), paired by Label 3/Animal ID. There is no
  CPP-specific "is this longitudinal" toggle; a pre/post design is expressed by setting "Group by" to
  whichever label column distinguishes the phases (e.g. Label 2 = "Pre"/"Post").
- Every between-group index gets a **mandatory activity/locomotor control metric** shown alongside it (see
  §3, per paradigm) — a literature-driven design choice, not incidental: a treatment that looks
  anxiolytic/preferring/discriminating can be, in reality, a treatment that changes general locomotion, and
  reporting the index alone without the control invites exactly that confound.

---

## 3. Per-paradigm reference

### 3.1 Generic (all paradigms, default view for Open Field / Custom)

**Tracing required:** none mandatory (works with a boundary alone), but any traced regions/objects populate
the cross-tabs and region-time control bar. Open Field's paradigm-specific extras additionally look for a
region whose name contains "center" (case-insensitive — see §5.2) and read `path_length_px` from the
enriched sidecar (requires `kinematic_directedness_enabled`).

**Graphs:** whole-arena occupancy heatmap; total-time-in-traced-regions bar (activity control); for
`open_field` specifically, total distance traveled, center-zone entry frequency, a center-zone %-time-course
across the trial (thigmotaxis dynamics, §2.3), and a mean-speed distribution across animals; cluster/group
cross-tabs.

**Use case:** the default view for any session that isn't one of the five named paradigms, or a mixed/pilot
cohort where no formal paradigm applies yet. Useful as a first-pass spatial sanity check before committing
to a specific paradigm's role tagging.

**Shortcomings:** with no role vocabulary for `custom`/`open_field` (`ENV_PARADIGM_ROLE_VOCAB` defines both
as `{"regions": None, "objects": None}`), there is no formal way to mark a "center zone" — the center-entry
control relies on the user having literally named a region containing "center," a naming-convention
heuristic that silently produces nothing if the convention isn't followed.

**Suggestions:** name a center region literally `"center"` or `"center_zone"` if using the Open Field
thigmotaxis control; enable `kinematic_directedness_enabled` alongside `env_features_enabled` if the
distance-traveled control matters for the study design — it is not retroactively computable otherwise.

### 3.2 Y-Maze Alternation

**Tracing required:** ≥3 regions tagged role `"arm"` (`ENV_PARADIGM_MIN_ROLES["y_maze"] = {"arm": 3}`); a
`"hub"`-tagged central region is supported and excluded from the alternation sequence.

**Graphs:** per-arm occupancy heatmap; spontaneous alternation % bar (between-group); total arm entries bar
(mandatory activity control, tested as a **separate** comparison from alternation %, per the field's
standard practice — an alternation difference confounded by an entries difference is a common review
objection); an arm-entry sequence strip (§2.3), one row per session, colored by which arm (or the hub, shown
in gray) the animal occupied over time; cluster/group cross-tabs.

**Use case:** spatial working-memory assessment — spontaneous alternation % is the field-standard
readout, computed from the region-entry sequence with the hub excluded (`_region_entry_sequence`,
`Environmental_Context_v6_Implementation_Report.md` §Step 4).

**Shortcomings:** alternation % and entry-sequence quality depend entirely on correct region role tagging
(arm vs. hub) — a mistagged hub counted as a fourth arm silently corrupts every alternation triad it
appears in. No ground-truth validation of the sequence extraction against hand-scored video exists in this
codebase.

**Suggestions:** verify the hub region is tagged `"hub"`, not left untagged or tagged `"arm"`, before
running the pipeline — this cannot be corrected after the fact in the Analyser, since role tags live in
`env_arena_cfg`, not in any per-run output.

### 3.3 Elevated Plus Maze (EPM)

**Tracing required:** ≥1 region tagged `"open_arm"` and ≥1 tagged `"closed_arm"`
(`ENV_PARADIGM_MIN_ROLES["elevated_plus_maze"]`); a `"center"`-tagged region is supported.

**Graphs:** open- vs. closed-arm occupancy heatmap; % time in open arms, % open-arm entries, and latency to
first open-arm entry (three separate bars); total arm entries bar (mandatory locomotor control — the single
most common EPM criticism in review is an anxiolytic-looking effect that is actually a locomotor confound);
an open-arm %-time-course across the trial (§2.3); a "head-dip / peering-out" bar — the mean fraction of
time the nose point fell outside the traced arena boundary while the body stayed on it (only shown when a
boundary shape was traced) — a **geometric proxy, not a validated head-dip or stretch-attend-posture
classifier**, using the new `nose_outside_boundary` per-bin key; cluster/group cross-tabs.

**Use case:** unconditioned anxiety-like behavior assay. All three primary metrics are reported
independently rather than folded into one index, since the literature does not treat them as
interchangeable (time, entries, and latency can dissociate).

**Shortcomings:** same role-tagging dependency as Y-Maze; additionally, the current occupancy heatmap
approximation (§2.1) is least informative for EPM specifically, since the true arm geometry (narrow,
elongated, meeting at a central platform) is exactly the kind of non-convex shape a convex hull represents
poorly. The head-dip/peering-out bar is a cheap geometric proxy (nose crossed the traced boundary edge,
body didn't) added because nose-point tracking was already available for objects but not applied to region
boundaries — it is **not** a validated stretch-attend-posture or head-dip behavioral classifier, and full
posture-based SAP detection remains explicitly out of scope (deferred, along with the Y-Maze novel-arm
variant).

**Suggestions:** trace open and closed arms as separate, non-overlapping region polygons even though they
physically share the central platform edge, so entry/exit counting is unambiguous; treat the drawn "(approx.)"
outline as directional guidance only, not a substitute for reviewing the raw traced arena in `EnvContextWindow`.

### 3.4 Discrimination Index (Novel Object Recognition)

**Tracing required:** ≥1 object tagged `"novel"` and ≥1 tagged `"familiar"`
(`ENV_PARADIGM_MIN_ROLES["novel_object"]`).

**Graphs:** investigation-density heatmap around each object; discrimination index bar, tested **both**
between-group and one-sample-against-chance (0.0 — the field-standard test, since a discrimination index of
0 is the null expectation of no preference); total exploration time across both objects (mandatory validity
control — low total exploration makes the index statistically unreliable regardless of its point estimate,
and is routinely reported alongside it in the literature); a nose-to-novel-object investigation-distance
time-course across the trial (§2.3, lower = closer investigation); an investigation bout-duration
distribution (each point is one bout, not one animal); a latency-to-first-contact bar; cluster/group
cross-tabs.

**Use case:** recognition-memory assay. The one-sample-vs-chance test is the primary scientific claim this
paradigm makes ("did this group show discrimination at all") — the between-group test is secondary
("did discrimination differ between groups").

**Shortcomings:** the discrimination index and total exploration time both derive from
nose/paw-to-object distance thresholding (`interaction_threshold_px`, auto-derived from one body length
unless set manually) — a single global threshold per session, not object-specific, and not validated
against hand-scored investigation bouts.

**Suggestions:** report total exploration time alongside the index in any write-up, not just the index
alone; if `interaction_threshold_px` is left on auto, verify the auto-derived value (visible in
`session_env_context.json`'s `summary.interaction_threshold_px`) is physically reasonable for the animal
and camera setup before trusting the index.

### 3.5 Sociability / Social Novelty (Three-Chamber Test)

**Tracing required:** for Phase 1 (sociability), objects tagged `"stranger"` and `"empty"`; for Phase 2
(social novelty), `"stranger"` and `"novel_stranger"` (`ENV_PARADIGM_MIN_ROLES["three_chamber"]` requires
`stranger`+`empty` at minimum for the derived metric to compute at all).

**Graphs:** per-chamber occupancy heatmap, one panel per phase (never pooled — pooling phase 1 and phase 2
together conflates two distinct hypotheses, a design choice made explicitly in the plan); Phase 1
sociability index and Phase 2 social-novelty index as two **independently** tested bars, each against chance
(0.0); chamber entry frequency (mandatory activity control); a nose-to-stranger investigation-distance
time-course across the trial (§2.3) **in place of** a chamber-occupancy time-course — this paradigm's role
vocabulary is on the traced objects (stranger/empty/novel_stranger), not the chamber regions, so a
role-filtered %-time-in-chamber-over-time isn't available the way EPM's open/closed-arm time-course is; an
investigation bout-duration distribution; a latency-to-first-contact-with-stranger bar; cluster/group
cross-tabs.

**Use case:** social-approach and social-novelty-preference assays, the two classically dissociated
sub-questions the three-chamber paradigm is designed to separate.

**Shortcomings:** the two-phase protocol requires per-video `role_overrides` (the same object cup is
`"empty"` in Phase 1 and may need re-tagging `"novel_stranger"` in Phase 2) — a manual step in
`EnvContextWindow` Tab 2 that is easy to skip, silently causing Phase 2's derived metric to compute against
the wrong role assignment rather than failing loudly.

**Suggestions:** double-check phase-appropriate role assignment per video explicitly before running the
pipeline, per session — this is flagged as a light validation caveat in the plan and is the single most
paradigm-specific tracing error to watch for in this tab.

### 3.6 Place Preference / CPP

**Tracing required:** ≥1 region tagged `"paired"` and ≥1 tagged `"unpaired"`
(`ENV_PARADIGM_MIN_ROLES["place_preference"]`); `"neutral"` is supported.

**Graphs:** occupancy heatmap (pre- vs. post-test side by side when a genuine pre-test phase exists — the
**primary** figure for this paradigm, not secondary, since the numeric index alone omits whether a
preference was uniform across the chamber or concentrated near the paired-side cue); preference index bar,
tested both one-sample-against-zero (delta/CPP-score) and between-group; when "Experimental Design" is set
to Repeated Measures with a genuine pre-test phase, an additional paired pre/post comparison
(`run_cluster_statistics(design="repeated")`) **and** a paired pre → post dumbbell plot (§2.3, one line per
matched subject — the standard CPP figure for individual-animal change); a cumulative chamber-crossing
time-course across the trial (activity control, shown for every design); cluster/group cross-tabs.

**Use case:** conditioned place preference/aversion — the paired pre/post design is the field-standard,
strongest form of this assay (baseline-corrected, within-subject); the between-group post-test-only
comparison is the fallback for designs without a genuine pre-test phase.

**Shortcomings:** a genuine pre-test phase requires the user to structure their loaded sessions/labels so
"Group by" resolves to a pre/post axis — sessions without a distinguishable pre-test phase silently degrade
to the absolute/delta bar and between-group comparison only, with the paired panel simply omitted (by
design, not a bug, but easy to miss if the omission isn't noticed).

**Suggestions:** use a dedicated label column (e.g. Label 2 = "Pre"/"Post") specifically for the phase axis,
distinct from the experimental-group label, so both the paired comparison and the between-group comparison
can be run from the same loaded session set without re-loading.

### 3.7 Approach/Avoid Events

**Tracing required:** any regions/objects (no paradigm restriction — applies to all six paradigms and
`custom`); **requires both `env_features_enabled` and `kinematic_directedness_enabled`.**

**Graphs:** trajectory plot — each detected event drawn as a start→end arrow, color-coded approach (green)
vs. avoid (red), against the approximate region/object outlines; event rate (events/minute, between-group);
target-distribution bar chart split by classification; cluster/group cross-tabs.

**Use case:** exploratory identification of directed approach/retreat bouts toward a traced region or
object, independent of which formal paradigm is in use — the only cross-paradigm sub-view in this tab.

**Shortcomings — stronger and more prominent than every other sub-view's caveat, by design:** this is a
rule-based heuristic (a directed-locomotion bout, defined by straightness ratio and relative speed,
immediately followed by an investigation-type bout for "approach"; a net-retreat distance change for
"avoid"), **not validated against expert human scoring**. The detector has no access to a genuine
investigation-state classifier and substitutes the following bout's own relative speed as a proxy. Absence
of `approach_events.csv` for a session can mean either "the flags were off" or "it ran and found nothing" —
these are not the same case, and the tab distinguishes them in its placeholder text.

**Suggestions:** treat this sub-view's output as a helpful starting point for manual review, not a
confirmed behavioral classification — explicitly, do not use it as the sole evidence for a claim in a
publication without independent verification against a human-scored subsample.

---

## 4. Cross-paradigm limitations

These apply regardless of which sub-view is in use, and are worth stating once rather than repeating per
paradigm:

1. **One paradigm per project.** `env_arena_cfg` is a single, project-wide configuration
   (`schema_version: 3`); a batch of recordings that genuinely mixes paradigms (e.g. some Y-Maze, some Open
   Field, in the same load folder) needs separate CUBE projects/output directories, one per paradigm — there
   is no per-session paradigm override.
2. **Occupancy heatmaps and region/object outlines are approximated from behavioral data, not measured from
   the traced shape directly** (§2.1) — a structural limitation of the current output schema, not a bug.
   Users who need the literal traced polygon should refer to `EnvContextWindow`'s own saved template (Save/
   Load Template, JSON), not this tab's rendered outline.
3. **Small-n statistics are typical and expected.** Rodent behavioral pilot cohorts routinely run 3-8
   animals per group; the between-group tests here (Kruskal-Wallis, Dunn's) and one-sample tests
   (t-test/Wilcoxon) are chosen for their applicability at low n, but statistical power at n<5 per group is
   genuinely limited regardless of the test — a non-significant result at this sample size should not be
   over-interpreted as evidence of no effect.
4. **Coordinate-space correctness between traced shapes and pose data is a structural guarantee** (identical
   crop-aware pixel space, enforced by `EnvContextWindow` always reading frames through the same crop
   rectangle DLC tracked on), not a runtime-checked one — see
   `Environmental_Context_v6_Implementation_Report.md`'s "Coordinate-space correctness" section for the full
   rationale. A user who manually alters video crop settings between tracing and running the pipeline is
   outside this guarantee.
5. **No object-movement tracking.** All traced objects are assumed static for the duration of a session
   (per-video transform/override handles a *repositioned-between-sessions* object, not one that moves
   *during* a session) — deferred (Option B, per `Environmental_Context_v6_Implementation_Plan.md`'s
   explicitly out-of-scope list).
6. **No multi-animal/social-interaction geometry beyond static object proxies.** The Three-Chamber
   paradigm's "stranger" role is a static object/cup, not a second tracked animal — CUBE has no
   dyadic/multi-animal pose tracking (see `Competitive_Positioning_and_Publication_Pathway.md` for how this
   compares to MARS's dedicated social-interaction focus).
7. **Nose-point tracking reaches region boundaries only through the new, narrow additions above** (EPM's
   head-dip proxy, `dist_to_region_boundary_nose`) — region/arm *occupancy* itself (which region "counts" as
   the animal's location, feeding alternation %, time-in-region, entries) remains centroid-based throughout,
   unchanged by this pass. Only object interaction (discrimination index, sociability index, exploration
   bout timing) was ever nose-based. Don't read the new EPM head-dip bar as evidence that region membership
   generally became nose-aware — it didn't.
8. **Three-Chamber has no chamber-occupancy time-course** (§3.5) — a direct consequence of its role
   vocabulary living on objects, not regions. This is a data-model constraint discovered while building the
   time-course plots, not an oversight; the nose-to-stranger investigation-distance time-course is the
   literature-relevant substitute this paradigm's data actually supports.

---

## 5. Reference tables

### 5.1 Paradigm → required roles → minimum count

| Paradigm | Regions/objects | Required role | Minimum count | Derived metric(s) gated on this |
|---|---|---|---|---|
| `y_maze` | regions | `arm` | 3 | `spontaneous_alternation_pct`, `n_arm_entries` |
| `elevated_plus_maze` | regions | `open_arm` + `closed_arm` | 1 + 1 | `pct_time_open_arms`, `pct_open_arm_entries`, `latency_to_first_open_arm_entry_sec` |
| `novel_object` | objects | `novel` + `familiar` | 1 + 1 | `discrimination_index` |
| `three_chamber` | objects | `stranger` + `empty` (Phase 1); `stranger` + `novel_stranger` (Phase 2) | 1 + 1 each | `sociability_index`, `social_novelty_index` |
| `place_preference` | regions | `paired` + `unpaired` | 1 + 1 | `preference_index` |
| `open_field`, `custom` | — | none | — | none (Generic view only) |

Below the minimum count, the derived metric is **silently omitted** from `session_env_context.json`'s
`derived` dict (not computed as a meaningless/zero value) — the corresponding sub-view bar simply doesn't
render for that session rather than showing a misleading number.

### 5.2 File → sub-view dependency

| Output file | Written when | Consumed by |
|---|---|---|
| `model/session_env_context.json` | `env_features_enabled=True`, any run | Occupancy heatmaps, all derived-metric bars, region-time controls |
| `*_bout_lengths_hmm_enriched.csv` | `env_features_enabled` and/or `kinematic_directedness_enabled` | Region/group cross-tabs (`pct_time_in_region_*`), distance-traveled control (`path_length_px`) |
| `*_approach_events.csv` | Both flags on, ≥1 event detected | Approach/Avoid Events sub-view only |

---

## 6. Recommended workflow

1. Decide the paradigm **before recording**, or at latest before running Step 3 — role tagging is
   configured once, project-wide, and cannot be added retroactively to an already-run session.
2. Trace the arena on a representative reference frame; tag roles precisely against §5.1's table, not by
   approximate name alone (a region literally named `"open_arm"` but tagged with no role, or the wrong role,
   silently fails the minimum-role gate).
3. Enable `kinematic_directedness_enabled` alongside `env_features_enabled` whenever Approach/Avoid Events
   or Open Field's distance-traveled control matter to the study design — this cannot be added after the
   pipeline has already run.
4. After running, load sessions in Combined Analysis as usual, then Refresh in Paradigm Results — the tab
   auto-selects the matching sub-view; browse others only to confirm the mismatch-placeholder behaves as
   expected, not to force a paradigm-specific view onto data it wasn't traced for.
5. Report the mandatory control-metric bar alongside every primary index in any downstream write-up — this
   tab computes and displays it specifically so it isn't left out.
6. Treat every occupancy-heatmap outline as approximate and every Approach/Avoid Events classification as
   heuristic (§4, item 2; §3.7) — neither substitutes for reviewing the traced arena/annotated video
   directly when precision matters.
