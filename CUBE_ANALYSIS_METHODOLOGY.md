# CUBE Analysis Methodology — How the Data Is Analysed, and How CUBE Differs From B-SOiD and the Field

This document explains, end to end, how CUBE turns raw DeepLabCut (DLC) pose-tracking output into
labelled behaviours, and positions CUBE's clustering pipeline — including the cluster-quality upgrades
added on top of the base B-SOiD architecture — against the wider unsupervised behaviour-classification
literature (B-SOiD, VAME, keypoint-MoSeq, MotionMapper, and the semi-supervised alternatives SimBA/
DeepEthogram). It is a companion to `README.md` (feature summary) and `CUBE_GUIDE.md` (step-by-step usage) —
this document is the "why it's built this way" reference.

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

*Sources: B-SOiD (Hsu & Yttri, *eLife* 2021); VAME (Luxem et al., *Communications Biology* 2022); keypoint-MoSeq
(Weinreb et al., *Nature Methods* 2024); MotionMapper (Berman et al., *J. R. Soc. Interface* 2014). See also
`README.md`'s "How this relates to B-SOiD, VAME, and keypoint-MoSeq" section and `CUBE_GUIDE.md`'s Step 3
(Iterative Split/Merge Refinement, Visibility & Confidence Features) and Step 5 (Cluster Validity plot mode)
documentation for the corresponding user-facing settings.*
