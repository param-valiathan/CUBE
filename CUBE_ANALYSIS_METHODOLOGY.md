# CUBE Analysis Methodology — How the Data Is Analysed, and How CUBE Differs From B-SOiD and the Field

This document explains, end to end, how CUBE turns raw DeepLabCut (DLC) pose-tracking output into
labelled behaviours, and positions CUBE's clustering pipeline — including the cluster-quality upgrades
added on top of the base B-SOiD architecture — against the wider unsupervised behaviour-classification
literature (B-SOiD, VAME, keypoint-MoSeq, MotionMapper, and the semi-supervised alternatives SimBA/
DeepEthogram). It is a companion to `README.md` (feature summary), `CUBE_GUIDE.md` (step-by-step usage), and
`Behavioral_Paradigms_Reference.md` (data requirements, use cases, and limitations for the paradigm-specific
reporting covered in Section 6) — this document is the "why it's built this way" reference.

---

## 1. How CUBE analyses data

CUBE is built on the B-SOiD (Hsu & Yttri, 2021) unsupervised pose-clustering paradigm: pose → engineered
kinematic features → nonlinear dimensionality reduction → density-based clustering → a fast supervised
classifier trained on the cluster labels, so clustering only has to run once and every subsequent video is
labelled by cheap inference. `BSoidEngine.run()` (`cube_core.py`) executes this as nine stages:

1. **DLC file discovery** — pairs pose files to their source videos by filename stem, prefix, or timestamp.
2. **Load & smooth** — normalises any DLC column layout, interpolates low-confidence frames, applies a
   centred boxcar smoothing window.
3. **Feature extraction** — multi-scale (50/100/200 ms) pairwise distances, velocities, smoothed
   accelerations, within-bin positional variance, temporal lag-drift, and body-axis angles, all computed
   per 100 ms bin. This is where CUBE's body-region weighting and adaptive visibility features are now
   injected (Section 2).
4. **UMAP** — nonlinear embedding of the feature matrix, with automatic PCA pre-reduction when the
   feature-to-sample ratio would otherwise degrade the neighbour graph.
5. **HDBSCAN sweep** — a `min_cluster_size` sweep (both `eom` and `leaf` selection methods) scored by DBCV
   (density-based cluster validity), picking the partition that balances validity against a preference for
   a biologically plausible number of clusters. This is where CUBE's split/merge refinement loop now runs
   (Section 2).
6. **MLP classifier** — a scikit-learn MLP is trained on the HDBSCAN-labelled bins, so clustering (slow,
   stochastic, run once) is decoupled from labelling (fast, deterministic, reused for every video).
7. **Inference** — the fitted scaler → (optional PCA) → MLP pipeline labels every session, bin by bin.
8. **HMM smoothing** — an `hmmlearn` categorical HMM (Baum-Welch + Viterbi) removes single-bin label flicker,
   producing the `_hmm` output variants used throughout the analyser.
9. **Export** — bout/frame/epoch CSVs, example video clips, labelled videos, and diagnostic plots.

This architecture is deliberately **not** a temporal deep-learning model: there is no autoencoder, no GPU
training loop, and a full multi-animal run typically completes in minutes. That speed/simplicity trade-off is
the central design choice this document evaluates against the field.

---

## 2. What CUBE adds on top of B-SOiD, and why

Frame/bin-wise embedding (B-SOiD's core design) has two well-documented failure modes, both raised
independently in the literature on unsupervised behaviour segmentation: a single continuous behaviour can
**fragment** across multiple clusters because each bin is embedded independently of its temporal neighbours,
and unreliable per-keypoint tracking (occlusion, animal turned away from camera) can get embedded as if it
were real postural variation, **contaminating** real clusters or splintering off spurious ones. CUBE addresses
both without adopting a different core architecture:

| Problem | Field precedent | What CUBE does |
|---|---|---|
| Behaviours split across clusters, or a cluster mixes unrelated behaviours | VAME's temporal embedding smooths instantaneous noise before clustering; HDBSCAN's own condensed tree already encodes multi-resolution structure that vanilla B-SOiD discards | An **iterative split/merge refinement pass**: clusters with low mean silhouette (impure/heterogeneous) get a local re-embed-and-recluster attempt, accepted only if a real stable split is found; clusters that separated at low condensed-tree persistence (near-duplicate over-splits) get merged, confirmed by embedding-centroid distance. Both off by default, both driven by diagnostics already computed for the new validity plot below, not a new architecture. |
| Some body regions (tail, back) matter less than others (mouth, limbs) for the behaviours of interest | keypoint-MoSeq's generative model learns per-keypoint uncertainty/relevance automatically | An **optional per-body-region feature weight** (GUI slider, grouped by region — Head/Mouth, Forelimbs, Hindlimbs, Trunk/Back, Neck, Tail), applied to pairwise-distance/velocity/acceleration/variance feature columns before scaling. Off by default (uniform weighting = today's exact behaviour). |
| Animal turned away from camera degrades face/forepaw tracking confidence, and those frames pollute or fragment real clusters | keypoint-MoSeq models per-keypoint measurement uncertainty generatively | **Adaptive visibility/occlusion features**: per-bin, per-region fraction of bodyparts below an *adaptive* (per-bodypart, per-session percentile-based, not a fixed global constant) confidence threshold, added as explicit feature columns. This lets HDBSCAN isolate low-confidence frames into their own identifiable cluster (flagged in `cluster_confidence.csv` and in the analyser's group editor) instead of scattering them through real behaviour clusters. **On by default** — the one new setting that changes default output, because it is a direct fix for a confirmed contamination pattern rather than a judgment call. |
| No way to see whether a cluster is internally consistent, only the 2D UMAP scatter | MotionMapper-style multi-resolution hierarchies; standard silhouette diagnostics in clustering generally | A new **`cluster_validity.png`** (silhouette diagram + HDBSCAN condensed-tree plot) generated every run, plus a matching "Cluster Validity" mode in the Step-5 analyser — both fed by data (the fitted `hdb_clf`, the embedding, the labels) that the pipeline already computes but previously discarded after label extraction. |
| Example clips shown for a cluster look inconsistent even when the cluster itself is fine | — (implementation detail, not a literature gap) | `create_example_clips` now selects clips nearest to the cluster's **feature-space centroid** in the UMAP embedding, instead of nearest to the cluster's *median bout duration* — two clips of "typical length" can sit at opposite ends of a cluster in feature space, which was a direct, fixable contributor to "inconsistent-looking" examples. |
| A cluster looks kinematically consistent (high silhouette) but is actually two spatially distinct behaviour contexts the feature space doesn't separate | — (CUBE-specific extension of this pipeline's own split mechanism, layered on Section 6's environmental-context data) | **Optional region-aware splitting** (Section 7): the same local re-embed-and-recluster mechanism as the split above, but candidate selection also considers normalized entropy of a cluster's traced-region membership, with a minority-fraction floor against noise-level contamination. Region data never enters the feature space — it only selects which clusters attempt a local re-cluster; HDBSCAN/DBCV still decide on kinematic terms alone. Off by default, requires traced regions (Section 6). |

Every mechanism above operates **after or alongside** the existing UMAP+HDBSCAN+MLP+HMM pipeline — none of it
replaces a stage or requires retraining a temporal model. That is a deliberate scope limit: it borrows the
*diagnosis* behind VAME's and keypoint-MoSeq's improvements (temporal fragmentation, keypoint uncertainty) and
applies the cheapest fix available inside B-SOiD's existing architecture, rather than re-implementing those
tools' full generative/sequence models. See `README.md`'s "How this relates to B-SOiD, VAME, and keypoint-MoSeq"
section for the shorter version of this argument.

---

## 3. Comparison to other methods in the field

| Method | Core approach | Temporal modelling | Per-keypoint importance / uncertainty | Occlusion / view-change robustness | Cluster validity diagnostics | Iterative refinement | Compute cost | Best fit |
|---|---|---|---|---|---|---|---|---|
| **B-SOiD** (Hsu & Yttri, 2021) — CUBE's base framework | Engineered kinematic features → UMAP → HDBSCAN → fast supervised classifier (random forest in the original paper) | None — bins embedded independently | None — all keypoints/features weighted equally | None — no explicit occlusion handling | None built in | None | Minutes, CPU-only | Fast, general-purpose unsupervised ethogram from any DLC pose set |
| **CUBE (this pipeline)** | B-SOiD architecture + optional post-hoc split/merge refinement, optional body-region feature weighting, adaptive visibility features, native validity diagnostics | Still bin-wise for embedding; refinement pass operates on the resulting partition, not a sequence model | **Optional**, manual, GUI-driven region weighting (not learned) | **Yes** — adaptive per-bodypart/per-session confidence features let low-confidence (turned-away) bins self-isolate into their own cluster | **Yes** — silhouette diagram + condensed-tree plot every run | **Yes** — silhouette-triggered split + condensed-tree-triggered merge, capped iterations, off by default | Minutes, CPU-only (unchanged from B-SOiD) | Same use case as B-SOiD, with mitigations for its two most common failure reports (fragmentation, view/occlusion contamination) and native quality plots for QC |
| **VAME** (Luxem et al., 2022) | RNN variational autoencoder learns a temporal latent embedding directly from pose sequences, then clusters the latent space (typically HMM or k-means) | **Learned**, explicit — the sequence model is the core mechanism, not a post-hoc fix | None inherent (uniform pose input to the RNN, unless features are pre-weighted) | Indirect — a temporal model can smooth through brief occlusion, but has no explicit uncertainty channel | Reconstruction loss / latent-space diagnostics, not silhouette-based | Not typically iterative — one trained embedding, one clustering pass | Requires GPU training (RNN autoencoder), hours per dataset/retrain | When temporal coherence matters more than iteration speed, or B-SOiD-style fragmentation is severe enough to need a different embedding, not just post-hoc merging |
| **keypoint-MoSeq** (Weinreb et al., 2024) | Generative state-space (ARHMM) model directly on keypoint coordinates, with explicit per-keypoint measurement-noise modelling | **Learned**, explicit — the switching-state model defines "syllables" as segments, not independent frames | **Learned automatically** — per-keypoint uncertainty is a first-class model parameter | **Yes, generatively** — the noise model is exactly designed for exactly this problem (unreliable keypoints) | Model-based (state usage, syllable duration distributions) | Model refitting, not a lightweight post-hoc pass | Model fitting (ARHMM) — CPU-feasible but slower and more setup than B-SOiD/CUBE | When per-keypoint reliability varies a lot (multi-camera, heavy occlusion) and a generative/statistical framework is preferred over the UMAP+HDBSCAN paradigm |
| **MotionMapper** (Berman et al., 2014) | Spectrogram/wavelet features of pose → t-SNE embedding → watershed segmentation of the resulting density map | None explicit, though wavelet features encode short-timescale dynamics | None | None explicit | Watershed/density-map structure is inherently multi-resolution | Multi-resolution is native to the method (adjustable watershed threshold), not an add-on | Minutes–hours depending on t-SNE settings, CPU | Continuous, high-frame-rate behaviours (e.g. Drosophila) where a dense, hierarchical behaviour map is the desired output |
| **SimBA / DeepEthogram** (semi-supervised alternatives) | Supervised/semi-supervised classifiers trained on human-annotated behaviour examples | Depends on features/architecture used | N/A — behaviours are defined by the annotator, not discovered | Limited by annotation coverage of occluded examples | Standard classifier metrics (precision/recall), not cluster validity | N/A — not a clustering approach | Requires labelled training data up front; inference is fast once trained | When specific, pre-defined behaviours (not open-ended discovery) are the goal, and annotated examples exist or are feasible to produce |

**Reading the table:** CUBE occupies the same speed/simplicity niche as B-SOiD (no GPU, no learned temporal
model, minutes not hours) while adding the specific diagnostics and mitigations that VAME and keypoint-MoSeq
were built to solve architecturally — fragmentation and keypoint-uncertainty contamination — as lightweight,
opt-in post-hoc passes rather than a different core method. If a dataset's problems turn out to be deeper than
these mitigations can fix (e.g. genuinely needs a learned temporal model to resolve very short, highly variable
behaviours), VAME or keypoint-MoSeq are the appropriate escalation — CUBE's design does not preclude exporting
the same pose data to either.

---

## 4. Practical implications

- **Every new mechanism is additive.** With `bodypart_weights={}`, `hdbscan_merge_thresh=0`, and
  `hdbscan_split_silhouette_thresh=None` (the defaults for everything except visibility features), CUBE's
  output is numerically unchanged from the pre-upgrade pipeline — this was verified during implementation
  by re-running the pipeline at default settings and confirming identical cluster counts/labels.
- **The one default that changes:** `visibility_features_enabled=True` adds occlusion-aware feature columns to
  every run out of the box, because it directly fixes a confirmed contamination pattern (turned-away frames
  polluting real clusters) rather than requiring a judgment call about which behaviours matter.
- **New outputs to inspect:** `cluster_validity.png` (is this run's clustering internally consistent?) and
  `cluster_confidence.csv` (which clusters, if any, are dominated by low-confidence tracking rather than real
  behaviour?) — both are new QC surfaces that did not exist before these changes, and both are worth checking
  on every new dataset, not just when something looks wrong.
- **Nothing here requires a GPU or a retraining cycle** — that remains CUBE's key practical advantage over
  VAME/keypoint-MoSeq for labs without dedicated compute or the time budget for iterative model refitting.

---

## 5. Hierarchy-informed reclustering (post-hoc, Analyser)

Everything above concerns the primary HDBSCAN clustering. A separate problem exists one layer up: the
Analyser's post-hoc reclustering (merging over-split HDBSCAN clusters into a smaller set of behaviours,
for statistics and write-up) originally judged merges on a single line of evidence — correlation of each
cluster's usage pattern across experimental groups, blended with transition-profile similarity. Two
clusters can be highly correlated on both measures for reasons unrelated to being the same movement
(shared arousal state, a session-length confound, coincidence in a small dataset), and nothing in that
blended distance alone could distinguish a real oversplit from a spurious correlation.

This is structurally the same problem WGCNA (Langfelder & Horvath) solved for gene-expression module
detection: `mergeCloseModules` only merges two modules if their similarity clears a threshold on a
*second, independently computed* criterion, not just the primary clustering signal. CUBE's analyser now
applies the same pattern — the primary clustering's own feature-space centroid hierarchy (already
computed and exported as `cluster_hierarchy_*.png`, persisted to `model/cluster_feature_centroids.npz`)
acts as a hard veto: a merge is only proposed if the two clusters are close in **both** response-pattern
space and pose space, not either alone. A greedy, self-stopping "Guided Merge" mode (a
`dynamicTreeCut`-style minimum-gap stopping rule) replaces "pick one k for everything" with "only take
merges that are individually justified," so the output favours the minimum number of merges actually
supported by the data rather than a global k the user has to guess from an elbow plot.

Full parameter reference, defaults, and worked use cases are in `CUBE_GUIDE.md`'s
[Recombination Guide](CUBE_GUIDE.md#6a-recombination-guide); the original design rationale is in
`RECLUSTERING_HIERARCHY_PLAN.md`.

---

## 6. Paradigm-specific reporting (v6 part 3, post-hoc, Analyser)

Everything above operates on clustering output alone — cluster identity, duration, frequency, transitions —
with no reference to the animal's position in the arena or its relationship to traced regions/objects.
CUBE's Environmental Context feature (`env_features_enabled`) closes that gap by computing per-bin
region/object membership and distance series alongside the existing pipeline (Section 2's table, this
document), and the Analyser's Paradigm Results tab (`ParadigmResultsPanel`, `cube_analyser.py`) turns that
series into six literature-standard behavioral-paradigm indices — spontaneous alternation (Y-Maze), open-arm
time/entries/latency (Elevated Plus Maze), discrimination index (Novel Object Recognition), sociability and
social-novelty indices (Three-Chamber Test), and preference index (Place/Conditioned Place Preference) —
plus a paradigm-agnostic rule-based approach/avoid event detector. A second wave of additions puts those same
indices in **motion**: %-of-trial time-course plots (not just a single trial-total number), investigation
bout-duration distributions, an arm-entry ethogram strip, and a paired pre/post dumbbell plot for CPP. Full
data requirements, per-paradigm use cases, and known limitations are in `Behavioral_Paradigms_Reference.md`;
this section covers the **methodological** choices, in the same register as the clustering methodology
above.

**Statistical design.** Two distinct questions recur across every paradigm and are deliberately kept
statistically and visually separate: "does this index differ between experimental groups" (Kruskal-Wallis +
Dunn's/Mann-Whitney post-hoc, BH-FDR corrected — identical machinery to Section 5's cluster-level tests,
reused via a thin adapter rather than re-implemented) and "does this index differ from a fixed reference
value" (chance level for Discrimination Index, zero for CPP's delta score — a one-sample t-test or
one-sample Wilcoxon signed-rank test, likewise BH-FDR corrected, and rendered with a visually distinct
marker convention). Folding both questions onto one significance symbol would let a reader mistake "group A
differs from group B" for "group A differs from chance," which are not interchangeable claims — this is the
same discipline the field expects of, e.g., a discrimination-index paper reporting both a one-sample test
against 50%/0 and a between-group test, not one or the other.

**The activity/locomotor confound is treated as a first-class requirement, not an afterthought.** A
literature review conducted while designing this feature found that every paradigm except Novel Object
pairs its primary index with a separate activity/exploration control metric in standard published usage,
because an index that looks anxiolytic, preferring, or discriminating can, in an underpowered or
confounded design, simply reflect a locomotor difference. CUBE's Paradigm Results tab makes this bar
mandatory (shown alongside the primary index automatically, not an optional toggle) rather than leaving it
to the user to remember.

**Spatial reporting is deliberately labeled as an approximation where it is one.** The occupancy-density
heatmap is a real, direct measurement (2D histogram of tracked centroid position). The region/object
*outlines* drawn on top of it are not — `env_arena_cfg`'s traced polygon vertices are never persisted to any
per-run output file (a genuine, disclosed architectural gap, not a rounding error), so the outline shown is
reconstructed as the convex hull of positions already known to belong to that region/object. This is stated
explicitly in-figure (`"(approx.)"`) rather than presented with the same visual authority as a directly
measured boundary — a distinction worth preserving in any downstream figure reuse.

**A new statistical primitive was added, not a new statistical framework.** `run_cluster_statistics()`
(Section 5's reused machinery) has no group-vs-fixed-constant mode; rather than generalizing that function
or building a parallel testing framework, one small, self-contained one-sample primitive
(`run_one_sample_statistics`) was added alongside it, sharing its BH-FDR convention. This mirrors the
scoping discipline already established in Section 5 — WGCNA's `mergeCloseModules` pattern was adopted
without adopting WGCNA's full module-detection framework, and the same "borrow the smallest correct
mechanism" approach applies here.

**Nose-point tracking was already computed for object interaction; the second wave extends it to region
boundaries, narrowly.** Object-interaction distance (discrimination index, sociability index, exploration
bout timing) was always based on the nose/paw point, not the whole-body centroid — the field-standard
definition of "investigation." Region/arm *occupancy itself* (which region "counts" as the animal's
location — alternation %, time-in-region, entries) remained, and remains, centroid-based; that did not
change. What was added is a single narrow extension: a nose-point distance to region boundaries
(`dist_to_region_boundary_nose`) and a nose-outside-traced-boundary flag (`nose_outside_boundary`), enabling
an EPM head-dip/peering-out proxy. This is explicitly labeled a geometric proxy, not a validated posture
classifier (no elongation/stretch-attend-posture detection was attempted) — the same "don't ship an
unvalidated behavioral claim as if it were confirmed" discipline already applied to the approach/avoid
detector above.

**Three-Chamber has no chamber-occupancy time-course, by data-model constraint, not oversight.** This
paradigm's role vocabulary (stranger/empty/novel_stranger) lives on the traced *objects*, not the chamber
*regions* — there is no role to filter a %-time-in-chamber-over-time by, unlike EPM's open/closed arms.
Investigation-distance-to-stranger-over-time is the literature-relevant substitute this data model actually
supports, used in its place.

Full parameter reference, per-paradigm tracing requirements, use cases, and shortcomings are in
`Behavioral_Paradigms_Reference.md`; implementation detail (schema, deviations, verification) is in
`Environmental_Context_v6_Implementation_Report.md`, `Kinematics_v6_Implementation_Report.md`, and
`Analyser_Paradigm_Reporting_v6_Implementation_Report.md`.

---

## 7. Region-aware cluster refinement (v1, opt-in, primary pipeline)

Section 2's iterative split/merge refinement pass (the fragmentation/impurity mitigation adopted from the
field's general silhouette/condensed-tree diagnostics) flags a candidate for splitting using exactly one
signal: how internally consistent a cluster looks in **kinematic feature space** (mean per-bin silhouette).
That signal is blind to a real, common failure mode Section 6's environmental-context work made newly
visible: a cluster can look kinematically consistent while actually being **two spatially distinct
behaviours** the feature space doesn't separate cleanly — e.g. a "grooming" cluster that is 85% Periphery
time and 15% Center time in an Open Field session may really be two different grooming *contexts* (a
defensive/anxious variant near the wall vs. a relaxed variant in the open), not one behaviour, even though
their kinematic signatures overlap enough that silhouette alone never flags the cluster as impure.

**Design constraint carried over unchanged from Section 6.** The same "structurally position-dominance-free"
commitment that keeps `env_features_enabled`'s region/object time series out of the UMAP/HDBSCAN feature
space applies here without exception: region membership never becomes a clustering feature. It only ever
**selects which existing clusters attempt a local re-embed-and-recluster**, exactly mirroring the mechanics
Section 2's silhouette-triggered split already uses — the local re-clustering itself still runs purely on
kinematic features, and HDBSCAN/DBCV still decide, on their own terms, whether a real stable split exists.
A cluster is never split just because its bins happen to sit in different regions; it is split only when
that spatial impurity crosses a threshold *and* a genuine kinematic sub-structure is found locally. This
keeps the same guarantee Section 6 established for the post-hoc analysis (position never substitutes for
movement as a behavioral signal) intact for the primary clustering itself.

**Impurity criterion: normalized Shannon entropy, not a hand-tuned position-overlap heuristic.** Each
candidate cluster's region-label distribution across its own bins is scored by normalized entropy (0 = every
bin in one region, 1 = maximally mixed across however many regions are represented) — the same
"cluster-level scalar, threshold-compared" shape `_mean_silhouette_per_cluster` already uses, so it composes
directly with the existing worst-first candidate-bounding machinery rather than requiring new selection
logic. A minority-fraction floor (default 15% of a cluster's region-labelled bins) sits alongside the
entropy threshold specifically to avoid splitting on noise-level contamination — e.g. a cluster that is 99%
Center / 1% Periphery (likely a single frame's tracking jitter near a region boundary) should not be forced
into two fragments over 1% of its bins.

**A one-shot convenience, not a target-seeking algorithm.** An earlier design idea — "lower the preferred
cluster-count range going in, and let region-aware splitting bring the count back up to a similar target" —
was evaluated and explicitly *not* built as a closed-loop retry mechanism. This codebase has no existing
precedent for iteratively re-running HDBSCAN toward a target count, and building one introduces real
oscillation/instability risk with no field precedent motivating it here. What shipped instead
(`region_split_pre_reduction_pct`) is a single, one-shot multiplier applied to the primary sweep's own
preferred-cluster-count range, with a clearly logged before/after cluster-count trace — an honestly-scoped
convenience the user can read and retune, not a promise that the final count lands anywhere specific.

**Consequence for how a region-split-derived cluster is later treated.** Because the split still produces
ordinary HDBSCAN-style cluster ids from a real local UMAP+HDBSCAN fit, every downstream stage — MLP
training, HMM smoothing, rare-cluster pruning, and (when active) the existing condensed-tree merge pass —
treats a region-split-derived cluster identically to a silhouette-split-derived or a directly-selected one.
No new downstream machinery was needed; this was verified rather than assumed (see the implementation
report's Phase 6 for the specific test coverage this claim rests on).

Full parameter reference and GUI location are in `CUBE_GUIDE.md`'s Environments/Objects/Paradigms section;
implementation detail (schema, phased equivalence-test results, deviations from the original plan) is in
`Region_Aware_Refinement_Implementation_Report.md`.

---

*Sources: B-SOiD (Hsu & Yttri, *eLife* 2021); VAME (Luxem et al., *Communications Biology* 2022); keypoint-MoSeq
(Weinreb et al., *Nature Methods* 2024); MotionMapper (Berman et al., *J. R. Soc. Interface* 2014); WGCNA /
`dynamicTreeCut` / `mergeCloseModules` (Langfelder & Horvath, *BMC Bioinformatics* 2008); cophenetic
correlation (Sokal & Rohlf, *Taxon* 1962); standard behavioral-paradigm metric/control conventions per
ConductScience's OFT/EPM/Y-maze/CPP protocol references, PMC reviews of novel-object-recognition and
three-chamber sociability methodology, and a Frontiers review of CPP quantification approaches (used to
confirm standard metric/control pairings, not cited as primary research claims). See also `README.md`'s "How
this relates to B-SOiD, VAME, and keypoint-MoSeq" section and `CUBE_GUIDE.md`'s Step 3 (Iterative
Split/Merge Refinement, Visibility & Confidence Features), Step 5 (Cluster Validity plot mode, Paradigm
Results tab), and Recombination Guide documentation for the corresponding user-facing settings.*
