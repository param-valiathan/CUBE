<p align="center">
  <img src="CUBE_logo.png" alt="CUBE Logo" width="240"/>
</p>

<h1 align="center">CUBE — Comprehensive Unsupervised Behavioral Explorer</h1>

<p align="center">
  Automated, end-to-end pipeline for discovering and quantifying animal behaviour from video — no manual labelling required.
</p>

<p align="center">
  <b>Current release: v5</b> &nbsp;·&nbsp; 4-camera 3D DeepLabCut + Anipose triangulation, fully integrated
</p>

---

## Overview

CUBE integrates **DeepLabCut (DLC) pose estimation** with unsupervised machine learning (UMAP + HDBSCAN + MLP classifier) to automatically segment, cluster, and analyse animal behavioural patterns from raw video — with no predefined behavioural categories.

The pipeline runs from raw video to full statistical group comparisons in six steps. It is built on the **B-SOiD** methodology (Hsu & Yttri 2021, *Nat. Commun.* 12:5188) and has been substantially extended with improved feature extraction, automatic cluster quality selection, HMM temporal smoothing, a comprehensive behavioural analyser, and full 3D multi-camera support.

**What CUBE does:**
- Step 0: 4-camera Bonsai recording synchronisation with fill-frame detection
- Step 1a (2D): Batch DLC inference (SuperAnimal quadruped model, Smart Adapt)
- Step 1b (3D): Per-camera DLC Zoo adaptation → aniposelib triangulation → 4-coord 3D H5
- Step 2: BSOID pre-processing — bodypart conservation, confidence filtering
- Step 3: Multi-scale feature extraction → UMAP → HDBSCAN sweep → MLP → HMM smoothing
- Step 4: Interactive cluster annotation via embedded video clips
- Step 5: Full statistical analysis: ethograms, group comparisons, transition dynamics, reclustering

---

## Documentation

| Document | Purpose |
|---|---|
| [CUBE_GUIDE.md](CUBE_GUIDE.md) | **Full user guide** — installation, step-by-step walkthrough, all features, troubleshooting |
| [GROUP_PREDICTOR_REFERENCE.md](GROUP_PREDICTOR_REFERENCE.md) | **Complete Group Predictor reference** — algorithms, figures, controls, interpretation, caveats |
| [TEST_README.md](TEST_README.md) | Test suite documentation |

---

## Quick Start

### Installation

```bat
conda create -n CUBE python=3.10
conda activate CUBE
pip install pillow opencv-python-headless scipy scikit-learn umap-learn customtkinter plotly hmmlearn deeplabcut aniposelib imageio-ffmpeg psutil
conda install -c conda-forge hdbscan
```

> **Note:** DeepLabCut must be installed in the `CUBE` environment — Step 1 DLC inference will fail otherwise. `aniposelib` and `imageio-ffmpeg` are required for 3D mode (Step 1b). Steps 2–5 do not require either.

### Launch

```bat
"C:\Users\<yourname>\anaconda3\envs\CUBE\python.exe" cube.py
```

Or run `install_shortcut.ps1` once to create a desktop shortcut.

### Standalone tools

```bat
"C:\...\python.exe" cube_analyser.py       # Behaviour Analyser (Step 5)
"C:\...\python.exe" cube_video_explorer.py  # Cluster Annotator (Step 4)
```

---

## The Six-Step Pipeline

| Step | File | Description |
|------|------|-------------|
| **0** | `bonsai_video_corrector.py` | Synchronise 4-camera Bonsai recordings; output corrected MP4s with fill-frame markers |
| **1** | `cube.py` + `cube_3d_dlc.py` | DLC inference — 2D SuperAnimal **or** 3D per-camera adaptation + aniposelib triangulation |
| **2** | `cube_core.py` | Load, normalise, filter DLC output; bodypart conservation; confidence report |
| **3** | `cube_core.py` | Feature extraction → UMAP → HDBSCAN sweep → MLP → HMM smoothing. Optional **Body-Region Weights** panel (Advanced Settings) lets you up/down-weight Head/Mouth, Forelimbs, Hindlimbs, Trunk/Back, Neck, or Tail before clustering |
| **4** | `cube_video_explorer.py` | Annotate clusters by watching example video clips |
| **5** | `cube_analyser.py` | Group statistics, ethograms, transition analyses, Group Predictor |

See [CUBE_GUIDE.md](CUBE_GUIDE.md) for full details on each step.

---

## How this relates to B-SOiD, VAME, and keypoint-MoSeq

CUBE's core pipeline is a B-SOiD-style architecture: frame/bin-wise pose features embedded with UMAP and
partitioned with HDBSCAN. That design is fast and needs no GPU training loop, but embedding each bin
independently of its neighbours has two well-known failure modes, which several cluster-quality features
target directly rather than through a pipeline rewrite:

- **B-SOiD** (the base framework CUBE already implements) is unsupervised and quick, but because each bin is
  embedded on its own, a single continuous behaviour can fragment across multiple clusters, and confidence
  artefacts (e.g. the animal turning away from camera) can get embedded as if they were real postural variation.
- **VAME** addresses fragmentation by learning a temporal embedding (an RNN autoencoder) instead of a frame-wise
  one, so instantaneous noise is smoothed by a sequence model before clustering happens at all. CUBE's
  condensed-tree cluster-merge pass and optional body-region feature weighting move in the same direction —
  reducing spurious fragmentation and down-weighting less-relevant dimensions — without adopting VAME's
  re-embedding architecture or its retraining cost.
- **keypoint-MoSeq** addresses per-keypoint tracking noise with a generative, uncertainty-aware model that
  automatically down-weights unreliable keypoints. CUBE's adaptive visibility/occlusion features are a much
  simpler analogue of the same idea: instead of modelling uncertainty generatively, low per-bodypart/per-region
  confidence becomes an explicit feature axis, so the existing HDBSCAN step can isolate those bins into their
  own cluster instead of scattering them through real behaviour clusters.
- HDBSCAN already builds a multi-resolution condensed tree internally on every run; CUBE now surfaces it (the
  `cluster_validity.png` silhouette + condensed-tree plot) and acts on it (the merge pass), instead of discarding
  it after label extraction — the same "use the hierarchy you already built" spirit behind MotionMapper-style
  multi-resolution behavioural analysis.
- **Net positioning:** these additions keep CUBE's core B-SOiD architecture (minutes, not hours, per run, no GPU
  training loop) while borrowing the *diagnosis* behind VAME's and keypoint-MoSeq's improvements and applying the
  cheapest available fix inside the existing pipeline for each one. All of it is additive and off-by-default
  except the visibility features (see Advanced Settings below) — a full temporal re-embedding (VAME-style) or
  generative uncertainty model (keypoint-MoSeq-style) would be the next escalation if these mitigations prove
  insufficient for a given dataset, not something attempted here.

---

## Architecture

CUBE is a five-file application:

| File | Role |
|------|------|
| `cube_core.py` | Pure analysis engine — all numerical logic, no GUI code |
| `cube.py` | Main six-step pipeline GUI (customtkinter). Lazy-loads the other modules at runtime |
| `cube_3d_dlc.py` | 3D pipeline module — session discovery, Zoo adaptation, fill masking, triangulation, quad video, skeleton video |
| `cube_analyser.py` | Standalone behavioural analyser (Step 5) |
| `cube_video_explorer.py` | Standalone cluster annotator (Step 4) |

---

## v5 — 3D Multi-Camera Pipeline

### Overview

v5 adds a dedicated **3D DLC + Anipose** step that integrates seamlessly with the existing pipeline. Selecting 3D mode replaces Step 1 (2D DLC) with a full 4-camera triangulation workflow. All downstream steps (2–5) are unchanged.

### How it works

```
Main panel: Video source folders (one per experimental group, containing Step 0 output)
       │  session["video_folders"] — same list used by all pipeline steps
       │
       ▼  cube_3d_dlc.py — Phase 1: Session discovery
       │  Reads dlc_3d_source_folders (re-run) or video_folders (first run)
       │  find_step0_sessions() — validates all 4 cams done, reads correction report
       │  session_src dict: maps session_id → source folder
       │  _out_root(sid): returns dlc_3d_output_folder or the session's source folder
       │
       ▼  Phase 2: Per-camera DLC Zoo adaptation + inference
       │  select_representative() — picks video with lowest fill rate for adaptation
       │  _adapt_zoo_for_camera() — GPU-accelerated Zoo adapt (one model per camera)
       │  _infer_camera_videos()  — batch inference with OOM-halving retry
       │  mask_fill_frames_in_h5() — zeros likelihood for fill frames
       │
       ▼  Phase 3: Triangulation (ThreadPoolExecutor, up to 4 parallel sessions)
       │  triangulate_cameras() — aniposelib CameraGroup.triangulate_ransac (default)
       │                          or CameraGroup.triangulate (RANSAC disabled)
       │  Combined likelihood: min/mean/max across cameras (user-selectable)
       │  Per-point confidence gate: zero likelihood < threshold before BSoid
       │  Output: {_out_root(sid)}/{session_id}_3d_filtered.h5
       │          (4-coord MultiIndex: x,y,z,likelihood; atomic write via .tmp.h5)
       │  make_quad_video(): probes each stream's resolution → normalises to common
       │          W×H → h264_nvenc (GPU) with libx264 -threads 0 (CPU) fallback
       │  Output: {session_id}_quad.mp4  (correctly-dimensioned 2×2 composite)
       │  Optional: {session_id}_skeleton3d.mp4  (3D skeleton QC animation)
       │
       ▼  Phase 4: video_folders handoff + BSOID prep
       │  Saves source folders → session["dlc_3d_source_folders"] (re-run guard)
       │  Replaces session["video_folders"] with session output dirs
       │  Carries group labels: session["video_groups"] updated per session dir
       │  run_bsoid_prep() — quad video explicitly preferred for example clips
              Writes BSOID_Project_Ready/ with z column preserved
```

### 3D H5 format

The output 3D H5 uses the standard DLC 3-level `(scorer, bodyparts, coords)` MultiIndex with **four coordinates per bodypart**: `['x', 'y', 'z', 'likelihood']`. This is auto-detected by `_h5_has_z()` in BSoidEngine, which also handles BSOID-ready CSV files.

```
(scorer='cube_3d', bodypart='nose', coord='x')     → 3D world X
(scorer='cube_3d', bodypart='nose', coord='y')     → 3D world Y
(scorer='cube_3d', bodypart='nose', coord='z')     → 3D world Z
(scorer='cube_3d', bodypart='nose', coord='likelihood') → min/mean/max across cams
```

### 3D feature extraction (BSoidEngine Step 3/7)

When a 3D H5 is detected (`_h5_has_z()` → True), BSoidEngine dispatches to `extract_features_3d()` instead of `extract_features_v2()`:

| Feature type | 2D (v2) | 3D (v3d) |
|---|---|---|
| Pairwise distances | `√(dx²+dy²)` | `√(dx²+dy²+dz²)` true Euclidean |
| Velocities | 2D magnitude | 3D magnitude per bodypart |
| Angular features | Spine landmark angles | — (world-coordinate distances already scale-invariant) |
| Temporal scales | 100 ms + 200 ms | 100 ms + 200 ms |
| Body normalisation | Optional (spine-relative) | Not needed (world coordinates) |

PCA, UMAP, HDBSCAN, MLP, and HMM are dimensionality-agnostic — they see a `(N_features, N_bins)` matrix regardless of 2D/3D mode.

### Triangulation options (ThreeDPanel settings)

| Setting | Default | Effect |
|---|---|---|
| Confidence aggregation | `min` | How to combine per-camera likelihoods: `min` (most conservative), `mean`, `max` |
| RANSAC triangulation | **On** | Use `CameraGroup.triangulate_ransac()` for robust rejection of occluded-camera outliers; falls back to standard triangulate if not available in installed aniposelib version |
| RANSAC threshold | 0.5 px | Reprojection error threshold for RANSAC inlier classification |
| Point confidence threshold | 0.0 (off) | After triangulation, zero-out likelihood for any bodypart×frame below this value; these slots are then excluded during BSoid bodypart conservation scoring |
| Pre-triangulation likelihood gate | **0.6** | Excludes any camera view whose DLC likelihood falls below this value entirely before aniposelib is called — a hard drop-gate rather than a soft down-weight. Addresses "spatial hysteresis" from transiently occluded camera views. Set to 0.0 to disable. |
| Temporal median filter | **3 frames** | Post-triangulation temporal median filter applied per bodypart per axis to remove single-frame 3D jitter that survives RANSAC spatial filtering. Kernel must be odd; set to 0 to disable. |

### Quad composite video

`make_quad_video()` produces a correctly-dimensioned composite even when cameras differ in resolution:

1. **Dimension probing** — `_probe_video_dimensions()` runs `ffmpeg -i` on each stream and parses the `WxH` from stderr.
2. **Normalisation** — the most-common width and height across all streams is chosen as the target; each stream gets a `scale=W:H` filter injected into the `filter_complex` before `xstack`/`hstack`.
3. **Efficient encoding** — tries `h264_nvenc` (NVENC GPU encoder, preset p4) first; on non-zero exit (NVENC absent) falls back to `libx264 -threads 0 -crf 23 -preset fast` which uses all available CPU cores.

The quad video is explicitly preferred by `run_bsoid_prep()` when copying a source video for example-clip generation, so the Video Explorer (Step 4) uses the multi-camera composite view.

### Per-session output layout

Output root per session = `dlc_3d_output_folder` if set, otherwise the session's source folder.

```
<source_folder or dlc_3d_output_folder>/
  <session_id>/
    <session_id>_3d_filtered.h5          ← 4-coord MultiIndex H5 (x,y,z,likelihood)
    <session_id>_quad.mp4                ← dimension-normalised 2×2 composite
                                            (cam0 TL, cam1 TR, cam2 BL, cam3 BR)
                                            GPU-encoded (NVENC) or CPU (libx264 -threads 0)
    <session_id>_skeleton3d.mp4          ← optional 3D skeleton QC animation
    BSOID_Project_Ready/
      h5/   <session_id>_3d.h5           ← conserved bodyparts, z column preserved
      csv/  <session_id>_3d.csv          ← same data as CSV (used by BSoidEngine)
      videos/ <session_id>_3d.mp4        ← quad composite copied for example-clip generation
      bodypart_confidence_report.csv
```

After Phase 4 completes, `session["video_folders"]` points to the session output dirs and `session["video_groups"]` maps each session dir to the experimental group of its source folder.  `session["dlc_3d_source_folders"]` preserves the original source folder list so re-running the 3D step rediscovers sessions without requiring the user to re-add folders.

### 2D backward compatibility

With `dlc_3d_enabled = False` (default), the pipeline runs identically to v4 (the 2D-only baseline). The `_h5_has_z()` check returns False for all standard 2D H5/CSV files, so `extract_features_v2()` is called as before. No 2D files are affected.

### ntfy notifications

| Phase | Event |
|---|---|
| Phase 2 | Each camera adaptation complete |
| Phase 2 | All DLC inference complete |
| Phase 3 | Triangulation complete |
| Phase 3 | Quad composite videos created |
| Phase 4 | BSOID pre-processing complete |

### Auto-BSOID chain

With `auto_bsoid = True`, completing the 3D DLC step automatically triggers Steps 2 → 3 in sequence (BSOID Prep → BSoidEngine), identical to how it works after 2D DLC. By Phase 4 `video_folders` has been replaced with session output dirs, `video_groups` carries the experimental group labels, and `dlc_3d_source_folders` is saved — so the chain finds all sessions and group assignments without user action.

---

## v5 Feature Summary

### Core Pipeline

**V2 multi-scale feature extraction**
- 100 ms and 200 ms window features concatenated — captures both fine and coarse motion scales.
- **Pairwise distances** between all tracked bodyparts (normalised by animal body size).
- **Velocities** per bodypart per bin.
- **Angular body-axis features** derived from spine landmarks (auto-skipped if landmarks not detected).
- **Within-bin positional variance** — distinguishes rapid oscillatory movements (tremor, shaking) from sustained postures, even at low between-bin velocity.
- **Temporal lag drift** (0.5 s and 1.0 s lag) — detects behaviour onset/offset independent of mean velocity.

**HDBSCAN sweep with DBCV guidance**
- Sweeps `min_cluster_size` across 40 steps; tries both `eom` and `leaf` methods.
- Selects the partition that maximises `relative_validity_` (DBCV) — no manual cluster count required.
- Rare clusters (< 0.2% of bins) are pruned before MLP training.
- Cluster count and anchor scale automatically adapt to recording length.
- **DBCV fix (Aug 2026):** `HDBSCAN(...)` is now constructed with `gen_min_span_tree=True`. Without it,
  `relative_validity_` (DBCV) is architecturally unavailable in the `hdbscan` library — it always raised
  `AttributeError`, silently caught, so every run unconditionally fell back to silhouette scoring even when the
  density graph wasn't actually degenerate. This was a pure library-usage bug, not tied to any particular
  dataset; fixing it makes DBCV a real, finite score again on every run.
- **`umap_n_neighbors` auto-formula floor raised 15 → 30 (Aug 2026):** a real short-session dataset (avg. ~333
  training bins/session) auto-set `n_neighbors` to 16 under the old floor and showed severe UMAP embedding
  seed-instability (6-seed sweep: cluster counts ranging 1–25, mean pairwise ARI 0.31 — small neighborhoods
  make UMAP's stochastic optimisation far more sensitive to its random seed). Forcing `n_neighbors=30` on the
  identical feature matrix raised mean pairwise ARI to 0.57; `n_neighbors=40` was worse (0.43), so 30 is a
  tested floor, not "higher is always better". Only validated on one dataset — see `consensus_clustering`
  below for a more general fix if `seed_sweep_stability`'s `mean_ari` is still low after this.
- **Consensus/co-association clustering (auto-triggered by default):** `consensus_clustering_enabled=True` (with
  `consensus_n_seeds`, default 8) forces on a partition built from agreement across several independent seeds —
  run UMAP+HDBSCAN `consensus_n_seeds` times, build a co-association matrix (fraction of seeds placing each pair
  of bins in the same non-noise cluster), then cluster that matrix once with Ward-linkage hierarchical
  clustering. This is the general fix for datasets where the embedding itself is seed-unstable (not just
  HDBSCAN's `min_cluster_size` selection) — it doesn't need to know *why* seeds disagree, only resolves the
  disagreement by construction. Even left at its default, it **auto-triggers** when the (also-default) seed-sweep
  diagnostic reports `mean_ari` below `consensus_auto_threshold` (default `0.5`) — a real `[CONSENSUS-AUTO]` log
  line always states when/why. Set `consensus_clustering_enabled=False` explicitly to opt out of auto-triggering
  entirely (always respected); set `consensus_auto_threshold=0` to disable auto-triggering while still allowing
  a manual `=True`. Costs roughly `consensus_n_seeds`× the primary UMAP+HDBSCAN runtime, which is why it isn't
  unconditionally on for every run. Validated result: 3–14× higher mean co-association within clusters than
  between, with balanced cluster sizes. `consensus_linkage` defaults to `"ward"` — `"average"` and `"complete"`
  were both tested and collapsed to a single giant cluster on real data (they chain through the noisy
  co-association matrix), so Ward isn't a casually swappable default. `consensus_max_memory_gb` (default `4.0`,
  `0`=unlimited) guards the `O(n_training_bins²)` co-association matrix (~200MB at 7,000 bins, ~1.6GB at 20,000,
  ~10GB at 50,000) — checked *before* the expensive per-seed loop runs, so an oversized dataset aborts
  immediately rather than burning that runtime first. When active (manual or auto-triggered), the split/merge
  refinement pass and the primary-embedding silhouette validation gate are both skipped for this partition (see
  `cube_core.py: consensus_cluster()` docstring for why) — check the logged `separation_ratio` instead.
- **Split-pass performance bounds (Aug 2026):** the split/merge refinement pass's split side used to run a full
  40-step × 2-method HDBSCAN sweep per impure-cluster candidate — fine for one or two, combinatorially expensive
  for many (confirmed on real data: one seed took 25+ min vs 1-2 min for others). `hdbscan_split_max_candidates`
  (default 10, worst-silhouette-first), `hdbscan_split_candidate_cutoff` (0=auto, a stricter silhouette bar
  applied first), `hdbscan_split_sweep_n_steps` (default 12 vs 40, single `eom` method), and
  `hdbscan_split_n_jobs` (default -1, thread pool — not separate processes, which triggered a real Windows
  crash in testing from a numba JIT-compilation race) together bound this without changing behavior for the
  common case (few impure clusters).
- **Adaptive parallel HDBSCAN sweep + system resource management (Aug 2026):** the primary `min_cluster_size`
  sweep (up to 40 steps × 2 methods) used to run as a plain sequential loop — most CPU cores sat idle for the
  majority of a run's wall-clock time. It dispatches via the same thread-pool pattern as `hdbscan_split_n_jobs`
  above (`hdbscan_sweep_n_jobs`), with selection unchanged (best-score pick over an unordered candidate list —
  verified byte-identical to the old sequential order on real data). **`hdbscan_sweep_n_jobs`'s default was
  changed from `-1` to `1` (sequential) later in Aug 2026** after a real Windows heap-corruption crash
  (`0xc0000374`) was captured at this exact call site — see the `hdbscan_sweep_n_jobs` entry in Advanced Settings
  below for the full root-cause writeup; parallel dispatch is still available by setting it back to `-1` or a
  pinned count. New
  **System Resources** Advanced Settings (`auto_resource_management`, default on) detect this machine's logical
  core count and RAM at run start (`psutil`) and size the sweep/split/seed-sweep thread pools to a configurable
  band — `system_resource_target_pct` (default `0.65`, the ideal 60–70% sustained-load range) hard-capped at
  `system_resource_cap_pct` (default `0.80`, never exceeded). The budget is re-checked before each heavy stage
  and shrinks (never grows past the cap) if RAM is already under pressure — logged as a `[SYSTEM]` line at run
  start and a `[MEMORY]` line at each stage boundary. `-1` on `hdbscan_sweep_n_jobs`/`hdbscan_split_n_jobs`/
  `seed_sweep_n_jobs`/`consensus_n_jobs` now means "auto-managed" rather than literal "all cores" when
  auto-management is on; any explicit value (`1` = sequential, or a pinned worker count) is honoured exactly as
  before. UMAP's embedding step stays forced single-threaded (`umap_n_jobs=1`) regardless — multi-threaded
  UMAP's nearest-neighbour search is non-deterministic, so this preserves exact embedding reproducibility across
  re-runs. Non-selected sweep candidates' HDBSCAN objects (condensed tree / MST) are now explicitly released
  right after selection rather than left for whatever scope the caller happens to hold them in, and
  `gc.collect()` runs at each major pipeline stage boundary (feature extraction, UMAP, HDBSCAN, MLP, per-session
  export). The publication benchmark metrics (`publication_metrics.json`) now report `peak_rss_gb` (real process
  memory via `psutil`, including numpy/BLAS temporaries) alongside the existing `peak_memory_gb` (Python-heap
  only, via `tracemalloc`).
- **Consensus co-association loop was accidentally sequential (fixed); thread-oversubscription bug in the seed
  sweeps (fixed, but does not solve overall wall-clock — see caveat):** `consensus_cluster()`'s per-seed
  UMAP+HDBSCAN loop (the "8-seed co-association" step `consensus_auto_threshold` can auto-trigger) had no
  parallel dispatch at all, unlike every other seed-loop in the file — a plain `for s in seeds:` regardless of
  cfg. It now dispatches via the same thread-pool pattern as `seed_sweep_n_jobs`, controlled by the new
  `consensus_n_jobs` (default `-1`, auto-managed). Separately, neither `seed_sweep_stability` nor
  `split_impure_clusters` capped numba's or BLAS's own internal thread pools inside each joblib worker thread —
  numba defaults to using *every* logical core for its own parallel regions independent of any `*_n_jobs`
  joblib setting, so N concurrent worker threads could each try to claim all cores for their own UMAP fit
  (verified `numba.get_num_threads() == 32` on a 16-core/32-thread box), causing real CPU oversubscription.
  Both are now capped per-worker (numba: thread-local, safe to set inside each worker independently; BLAS via
  `threadpoolctl`: process-global, so capped once around the whole `Parallel(...)` dispatch instead, to avoid
  workers racing on a shared limiter). **Caveat, confirmed by direct benchmarking (2,500 and 12,000 training
  bins):** fixing the oversubscription did not produce a net wall-clock win for `seed_sweep_stability` — parallel
  ran roughly tied with (not faster than) sequential at both scales. Profiling one seed's cost showed
  `run_hdbscan`'s 40-step sweep is dominated by Python-level orchestration (looping candidates, computing DBCV,
  building condensed trees), not pure Cython/numba compute — code that likely doesn't release the GIL, so
  running seeds concurrently on Python *threads* (required on Windows; process-based `loky` workers hit a real
  numba JIT-compilation race, see `hdbscan_split_n_jobs` above) can't get true multi-core throughput from that
  portion regardless of thread-pool sizing. The oversubscription fix is still correct and non-regressive (it was
  a genuine bug), but a real wall-clock fix for the seed sweep's overall runtime would require process-based
  parallelism with a JIT warm-up step to dodge the crash — not yet implemented.
- **Redundant seed-sweep elimination when consensus is forced on (Aug 2026):** when
  `consensus_clustering_enabled=True` explicitly, the standalone diagnostic seed sweep (whose only job is
  deciding *whether* to auto-trigger consensus) is skipped entirely — that decision is already moot once
  consensus is unconditionally on. `consensus_cluster()` now computes the same `seeds`/`counts`/`ari`/`mean_ari`
  stats for free from its own per-seed partitions (no extra UMAP/HDBSCAN calls), so `cluster_stability.png` and
  the `[VALID]` cluster-stability gate are populated exactly as before, just from a different (unrefined,
  clearly labelled as such in the log) source. Saves ~10 minutes per run on datasets where consensus is force-
  enabled. No effect when consensus is only auto-triggered (the sweep still has to run first there to make that
  decision) or left off.

**MLP classifier with cross-validation**
- Trains on HDBSCAN-labelled bins, generalises to all sessions including those not used for clustering.
- Cross-validated accuracy (standard and balanced) reported in the log.
- Optional class balancing, early stopping, and larger architecture (`256,128,64`) available in Advanced Settings.

**Post-hoc HMM smoothing**
- Two emission models, selected by `hmm_emission_mode`:
  - `"soft"` **(default since Aug 2026)** — `hmmlearn.GaussianHMM` fitted on the MLP's per-bin class-probability
    vectors (`predict_labels(..., return_proba=True)`) instead of collapsed hard labels. A frame the MLP was
    genuinely unsure about (a near-uniform probability row) contributes less confidently to the learned state
    sequence than one it was near-100% sure of — the old hard-label HMM smoothed both identically. State↔cluster
    alignment uses Hungarian assignment on each state's Gaussian mean (the analogue of the categorical path's
    emission-matrix alignment); alignment quality is logged the same way (`cube_emission_diag`).
  - `"categorical"` (pre-Aug-2026 default) — `hmmlearn.CategoricalHMM` fitted by Baum-Welch EM on hard MLP
    argmax labels.
- `hmm_smoothing_level` controls sequence resolution for the categorical path: `"bin"` **(default since Aug
  2026)** trains/decodes on the underlying 100ms-bin sequence (1/`win` the length of the old frame-repeated
  sequence) instead of `"frame"` (pre-Aug-2026 default, still available) — avoids diluting the learned
  transition matrix with trivial within-bin self-transitions, and is faster. Decoded bin-level states are
  expanded back to frame resolution the same way `predict_labels()` expands raw bin labels. Has no effect when
  `hmm_emission_mode="soft"`, which already operates at bin resolution unconditionally.
- `hmm_transition_prior` controls the transition-matrix Dirichlet prior fed to Baum-Welch, for **both** emission
  modes: `"per_cluster"` **(default since Aug 2026)** derives each cluster's own self-transition prior from its
  mean observed bout duration (`p_self = 1 − 1/mean_bout_frames`, clamped to `[0.5, 0.99]`) instead of one flat
  90%/10% self/spread prior for every state (`"global"`, pre-Aug-2026 default, still available) — avoids
  over-smoothing naturally brief behaviours under the same assumption used for naturally long ones.
- Viterbi decoding recovers the most probable state sequence, eliminating single-frame flickers.
- Produces `*_hmm` variants of all output CSVs. The Analyser prefers HMM files automatically.
- Diagnostic plots: bout-duration comparison, learned transition matrix (two panels), ethogram overlay, syntax network.
- Set `hmm_emission_mode="categorical"`, `hmm_smoothing_level="frame"`, `hmm_transition_prior="global"`
  explicitly to reproduce the pre-Aug-2026 HMM behavior exactly.

**Optional body-region feature weighting**
- `extract_features_v2`/`extract_features_3d` accept a per-bodypart weight multiplier (`bodypart_weights` cfg
  key, default `{}` = uniform/off — bit-identical output to today's pipeline). Weights scale pairwise-distance
  columns by `√(wᵢ·wⱼ)` and velocity/acceleration/within-bin-variance columns by `wᵢ` directly.
- Configured in the GUI via the **Body-Region Weights (optional)...** panel (Step 3 Advanced Settings): one
  slider per anatomical region, default `1.0` = uniform. Moving a slider off `1.0` is itself the "enable"
  signal for that region — there's no separate on/off checkbox; leaving every slider at `1.0` produces an
  empty weights dict (today's exact uniform behaviour).
- The last weights you `Apply` are remembered across projects in a small `body_region_weights.json` sidecar
  next to `cube.py` (same persistence pattern as the dark/light theme choice). A brand-new project's sliders
  pre-fill from this file instead of starting uniform every time; a project that already has its own explicit
  weights (e.g. loaded from a saved session) always keeps those instead. Nothing is committed to a project's
  settings until you click **Apply** in the editor.

**Automatic confidence-based bodypart weighting (on by default)**
- Complements the manual weighting above with a data-driven pass: after the existing hard-drop gate
  (`feature_bad_bp_thresh[_with_visibility]`, which removes only the most extreme bodyparts — default 0.70
  mean bad-frame fraction with visibility features on), any *surviving* bodypart that is still chronically
  unreliable gets its weight tapered down continuously instead of entering feature extraction at full weight.
  Bodyparts like this contribute flat-interpolated, near-identical feature vectors across many bins, which is
  what pushes HDBSCAN noise up and DBCV toward degenerate — the hard-drop gate alone only catches the worst
  cases, leaving a "moderately bad but not bad enough to drop" gap this fills.
- Linear taper on the **fraction of sessions a bodypart is genuinely bad in** — not a mean, and not a
  percentile-of-magnitude statistic. A session "counts" as affected when that bodypart's bad-frame fraction
  in it exceeds `auto_bp_weight_session_thresh` (default `0.3`). `auto_bp_weight_lo` (default `0.10`, i.e.
  ~2 sessions out of 21) and below → untouched (weight `1.0`); `auto_bp_weight_hi` (default `0.35`, i.e.
  ~7/21) and above → weight hits `auto_bp_weight_floor` (default `0.35`); linear in between.
- This statistic went through two iterations before landing here, both confirmed on real data: a **mean**
  dilutes a bodypart bad (42–67%) in only 5 of 21 sessions down to `0.157` (missed entirely — it never
  appeared in the `[AUTO-WEIGHT]` log despite being named the worst-tracked bodypart in several individual
  sessions). Switching to the **max** (100th percentile) across sessions fixed that case but over-corrected:
  with ~20 sessions, almost every bodypart has at least one session with a bad-frame spike — often driven by
  the animal turning away from camera, which naturally degrades jaw/forepaw tracking in whichever session has
  the most turned-away time — so max-based flagged 16 of 22 bodyparts in one real run (mostly jaws/paws) and
  measurably *increased* HDBSCAN noise (50.7% → 61.9%) and seed-to-seed instability (mean ARI 0.711 → 0.524)
  by flattening most of the feature space. **Counting sessions affected** solves both: a bodypart bad in only
  1–2 sessions (a fluke, including a turned-away-driven spike) stays under `auto_bp_weight_lo` untouched,
  while one bad in a recurring subset of sessions (5/21 = 24%, comfortably above `auto_bp_weight_lo`) still
  gets flagged regardless of how many other sessions are clean.
- The turned-away confound itself is also fixed at the source (see below), so a session's bad-frame fraction
  for face/forepaw bodyparts no longer counts frames that are already excluded from training anyway.
- An explicit entry in `bodypart_weights` (e.g. from the manual GUI panel above) always wins for that
  bodypart — this only auto-computes weights for bodyparts you haven't already customised.
- Controlled by `auto_bodypart_weighting` (default `True`). Set to `False` to revert to manual-only weighting
  (uniform if no manual weights are set). `compat_mode = "legacy_v2"` also disables it, for exact
  reproduction of runs from before this feature existed.
- Logged as a single `[AUTO-WEIGHT]` line per run (bodypart → computed weight, capped preview) — not spammed
  per-session.

**Turned-away-frame exclusion from bad-frame-fraction stats (on by default)**
- `turned_away_exclude_from_bad_frac` (default `True`): before either the FEAT-DROP hard-drop gate or the
  auto-weighting above sees a bodypart's per-session bad-frame fraction, frames already flagged
  turned-away-from-camera (see below) are excluded from that fraction's computation. Raw per-frame DLC
  likelihood naturally drops for face/forepaw bodyparts during turned-away moments — they're genuinely out
  of camera view — but those exact frames are already excluded from UMAP/HDBSCAN training separately, so
  counting them as "bad tracking" here double-penalises the bodypart for something that isn't chronic
  mistracking.
- `False` reverts to the raw (uncorrected) fraction. `compat_mode = "legacy_v2"` also disables it, since this
  changes what the pre-existing FEAT-DROP gate drops (not a new-this-session behaviour) and must stay
  opt-out for exact reproduction of older runs.

**Adaptive visibility / occlusion features (on by default)**
- New per-100 ms-bin feature block: mean bodypart-tracking likelihood, fraction of bodyparts below an
  adaptive per-bodypart/per-session confidence percentile, and per-body-region low-confidence fractions.
  Lets HDBSCAN isolate frames where the animal is turned away from camera (which selectively degrades
  face/forepaw tracking confidence) into their own cluster instead of polluting real behaviour clusters.
- `visibility_features_enabled` (default `True`) and `visibility_adaptive_pct` (default `10`) control this;
  set `visibility_features_enabled = False` for exact legacy reproducibility.
- Flagged clusters are written to a new `cluster_confidence.csv` (mean visibility, low-confidence fractions,
  a `low_confidence_flag` boolean) alongside the existing `cluster_kinematics.csv`.

**Turned-away-from-camera detection & dedicated "Turned Away" label (on by default)**
- This is the shipped, validated version of the visibility-feature mitigation above: rather than only hoping
  HDBSCAN naturally separates turned-away bins via the visibility features, CUBE now detects and (by default)
  removes them from training outright, and labels them explicitly instead of force-classifying them into a real
  behaviour cluster.
- Detection (`detect_turned_away_bins` in `cube_core.py`): a 100 ms bin is judged turned-away when **both**
  (a) the Head/Mouth region's `frac_low_conf` is ≥ `turned_away_conf_thresh` (default `0.30`) **and** (b) the
  `nose` keypoint's own bin-mean likelihood is below its adaptive per-bodypart/per-session threshold. Requiring
  the nose specifically (not just the Head/Mouth region average) was validated against real DLC data + video
  review to distinguish genuine turn-aways from a one-sided head turn or motion blur that degrades several head
  keypoints without the nose itself losing confidence. Sustained-window debouncing
  (`turned_away_min_window_s=0.4`, `turned_away_merge_gap_s=0.5`, engine-cfg only) drops single-bin jitter.
  Requires a `nose` bodypart; sessions without one skip detection with a warning (all-bins-False).
- `exclude_turned_away` (default `True`): flagged bins are excluded from UMAP/HDBSCAN training (same combined
  exclusion mask as the existing flat-held-bodypart exclusion, with the same too-few-good-bins safety fallback)
  and are given a reserved "Turned Away" cluster id (one above the highest real HDBSCAN cluster id) in
  `*_bout_lengths[_hmm].csv`, `*_frame_labels[_hmm].csv`, `*_epochs[_hmm].csv`, and `cluster_kinematics.csv`.
  Setting it to `False` is a true no-op escape hatch — detection still runs (for the video overlay and log line)
  but nothing is excluded or relabelled, reproducing pre-existing clustering output exactly.
- `turned_away_conf_thresh` (default `0.30`) is GUI-editable in Step 3 Advanced Settings (**FEATURE EXTRACTION**
  section). Raising it (e.g. to `0.45`) makes detection more conservative (fewer bins flagged).
- Labeled videos (`videos/labeled_videos/`) burn in an additional amber "TURNED AWAY" banner (bottom edge, full
  width) whenever a frame is flagged — independent of `exclude_turned_away`, always drawn when
  `save_labeled_video` is on, so detection stays visually auditable without a separate script. It never collides
  with the existing top-left `C{id}` cluster-label box.
- **Known limitation (documented, not a bug):** the empirical `transition_matrix.png`/`hmm_transition_matrix.png`
  plots and the HMM's own training data are built from the *raw* (unoverridden) label sequence, so they do not
  show "Turned Away" as a state — the dedicated label is applied only to the CSV/kinematics exports, downstream
  of HMM fitting (feeding the reserved id into HMM training would exceed the categories the `CategoricalHMM` was
  fit with). One consequence: a real behaviour bout interrupted mid-way by a turned-away window is split into two
  shorter bouts in the raw (non-overridden) transition-matrix view, but reported as bout-then-Turned-Away-then-
  bout in the CSVs — this can inflate a behaviour's bout *frequency* and deflate its mean *duration* for
  behaviours that happen to co-occur with turn-aways more than others. "% time in cluster X"-style metrics
  computed from these CSVs are over **total session time including Turned Away time** (not "real behaviour"
  time only) — every cluster's percentage is diluted by whatever fraction of the session was flagged
  turned-away (typically 8–15% in validation data), which is a denominator change worth knowing about when
  comparing against pre-feature runs or across animals/groups that may turn away at different rates.

**Per-cluster kinematic signatures in the Analyser (K1, always on)**
- `cluster_kinematics.csv` (mean centroid speed, body elongation, body-axis angular velocity per cluster —
  see Output Layout above) is now loaded automatically by `cube_analyser.py` and joined into the per-cluster
  metrics table alongside `total_duration`/`frequency`/`mean_bout`. The three new metric names
  (`mean_speed_px_s`, `mean_body_elongation_px`, `mean_angular_velocity_rad_s`) appear in the "Metric:"
  selector for the Top-N Bar, Volcano, and Heatmap plot modes. Runs predating this file simply show `NaN`
  for these metrics — no crash, no change to any existing metric. This is a pure data-loading addition with
  no gating flag (it reads a file that only exists once the pipeline has already produced it).

**Per-bout kinematic directedness (K2, opt-in — `kinematic_directedness_enabled`, default `False`)**
- When enabled, writes an additional sidecar CSV per session, `<stem>_bout_lengths_hmm_enriched.csv`
  (see Output Layout above): the canonical 3-column bout schema plus 5 new per-bout columns —
  `net_displacement_px`, `path_length_px`, `straightness_ratio` (net displacement ÷ path length; ~1.0 =
  straight-line movement, ~0 = meandering/returning near the start), `mean_speed_px_s`, and
  `heading_consistency` (mean resultant length of the bout's frame-to-frame heading vectors; 1.0 = every
  step points the same direction). Bouts shorter than 5 frames skip `straightness_ratio`/
  `heading_consistency` (`NaN`) rather than reporting a noisy value on too little data.
- The canonical `*_bout_lengths_hmm.csv` file is **never modified** by this flag, in either state — verified
  byte-identical whether the flag is on or off. When off (the default), no sidecar file is written at all.
- `predict_from_saved_model` also honors this flag from the saved model's own config, writing
  `<stem>_bout_lengths_enriched.csv` (no `_hmm` in the name, since that prediction path has no HMM-smoothed
  bout variant to pair with).
- Off by default and stays off unless explicitly enabled — this changes visible output schema, so (per the
  Kinematic_Transition_v6_Implementation_Plan.md's opt-in guarantee) it never auto-enables based on data
  characteristics the way `consensus_clustering_enabled`'s auto-threshold trigger does.

**Iterative split/merge cluster refinement (on by default)**
- After the HDBSCAN sweep, a bidirectional refinement loop (a) locally re-embeds and re-clusters
  low-mean-silhouette ("impure") clusters that mix distinct behaviours (`hdbscan_split_silhouette_thresh`,
  default `0.2`), and (b) merges sibling clusters that only barely separated in the HDBSCAN
  condensed tree — the classic signature of one behaviour over-split into near-duplicates
  (`hdbscan_merge_thresh`, default `0.08`). Split runs before merge, capped by
  `recluster_max_iterations` (default `2`), stopping early once an iteration makes no changes.
- Turned on by default after real-run validation showed the un-refined sweep over-fragmenting single
  behaviours into multiple clusters (e.g. "licking" split across 3 clusters, "sniffing" across 7) with
  seed-sensitive partitions (mean ARI ~0.59 across seeds). `0.2`/`0.08` are conservative starting points —
  well below/above the boundary of a healthy solution — chosen to catch clearly-impure or barely-separated
  clusters without over-merging genuinely distinct behaviours.
**Parameter reference — value ranges, tuning direction, and implications:**

| Parameter | Default | Valid range | Increasing it | Decreasing it | Implications |
|---|---|---|---|---|---|
| `hdbscan_split_silhouette_thresh` | `0.2` | `None` (off) or `-1.0`–`1.0`; useful range ≈ `0.0`–`0.4` | More clusters qualify as "impure" → more split candidates, more splits accepted | Fewer clusters qualify → only the most severely mixed clusters split; `None` disables the split pass entirely | A split is only *accepted* if the local re-cluster finds a genuinely stable sub-partition, so raising this is safer than it sounds — but too high (> 0.4) starts re-fragmenting naturally diffuse-but-real behaviours. Too low leaves genuinely mixed clusters unresolved. |
| `hdbscan_merge_thresh` | `0.08` | `0.0` (off) – `1.0`; useful range ≈ `0.0`–`0.2` | More sibling clusters qualify as "barely separated" → more consolidation, lower final cluster count | Fewer clusters qualify → more near-duplicate fragments survive; `0.0` disables the merge pass entirely | The main lever for undoing over-fragmentation. Centroid-distance confirmation guards against merging genuinely distinct behaviours, but values above ~`0.3` erode that safety margin. |
| `hdbscan_leaf_bonus` | `0.03` | `0.0`–`0.5` (only applied when `hdbscan_merge_thresh > 0`) | Biases `eom`/`leaf` tie-break toward `leaf` (finer, more fragmented base partition) | Reverts toward unbiased `eom`/`leaf` comparison | Only meaningful once merging is on. Set too high (> 0.15), it can force `leaf` even when `eom` was genuinely better. |
| `hdbscan_fine_bias` | `0.05` | `0.0`–`0.5` (only applied when `hdbscan_merge_thresh > 0` **and** DBCV is not degenerate for this run) | Pushes *initial* candidate selection toward the finer end of `[preferred_clusters_lo, preferred_clusters_hi]`, generating more clusters up front for merge to consolidate | Selection reverts toward pure DBCV + diversity-bonus ranking (tends toward coarser solutions) | Directly targets "not enough clusters to separate distinct behaviours" — pair with a non-zero `hdbscan_merge_thresh`. Too high (> 0.2) can dominate real DBCV quality differences. **Automatically disabled** when the run falls back to silhouette scoring (`[VALID-WARN] DBCV is non-finite...`) — biasing selection only makes sense when the underlying score is a trustworthy ranking signal; confirmed on real data that leaving it active under a degenerate-DBCV run made seed-sweep stability *worse* (mean ARI 0.55 vs 0.71, noise 55.2% vs 50.7%) rather than better. |
| `recluster_max_iterations` | `2` | `0`–`10` (int) | More split→merge cycles, letting each pass's output feed the other | Fewer cycles — risk of not converging; `0` disables the whole loop regardless of the thresholds above | Loop already stops early on zero-change convergence, so this mostly bounds worst-case runtime. `2` covers the common case (one split, one merge). |
| `hdbscan_split_max_subclusters` | `3` | `2`+ (int) | Allows a single impure cluster to accept a larger number of local sub-clusters in one split | Caps a split more tightly — fewer, coarser sub-clusters accepted | Hard ceiling on how many pieces one split pass can break a cluster into. Deliberately low: a split is meant to resolve the handful (typically 2–3) of genuinely distinct sub-behaviours mixed into one impure cluster, not fragment it extensively. The local re-cluster used for a split deliberately does NOT inherit `preferred_clusters_lo/hi`/`hdbscan_fine_bias` from the whole-session config (this value is used as its local `preferred_clusters_hi` instead) — without this, a single impure cluster could fragment into 20–30+ tiny pieces in one pass, since the whole-session preferred range (default up to `30`) and fine-bias nudge are calibrated for the full dataset, not for locally resolving one cluster. |
| `hdbscan_split_min_points` | `250` | `20`+ (int) | Allows smaller candidate clusters to attempt a local split | Requires a larger candidate cluster before a split is attempted at all | Below this many points, a local re-embedding is skipped entirely rather than attempted and likely rejected — sized at 5x UMAP's own PCA floor (50 components), since a sample/feature ratio under ~5 is the same curse-of-dimensionality regime `run_umap`'s auto-PCA trigger exists to avoid. With feature counts routinely exceeding 500–900 dims (more bodyparts → quadratically more pairwise-distance features), the old flat 20-point floor let splits run on 90–300 point subsets in practice — far too few to trust the resulting sub-clusters or local DBCV score. |

Related parameters that shape what the refinement pass operates on: `preferred_clusters_lo`/`preferred_clusters_hi`
(default `8`/`30`, the range `hdbscan_fine_bias` nudges within), `hdbscan_dbcv_thresh` (default `0.65`, the
quality floor used when no candidate falls in the preferred range), and `min_cluster_freq` (default `0.5`%,
rare-cluster pruning that runs *after* refinement and can prune a small split-off sub-cluster right back out).

Set `hdbscan_split_silhouette_thresh = None` and/or `hdbscan_merge_thresh = 0.0` (both GUI-editable in
Step 3 Advanced Settings) to disable either pass, or use `compat_mode = "legacy_v2"` to reproduce
pre-refinement runs exactly.
- `cluster_validity.png` (silhouette diagram + HDBSCAN condensed-tree plot, always generated when
  `save_plots` is on) is the direct diagnostic these settings are tuned against.
- `split_merge_refinement.png` (new): before/after diagnostic for the refinement pass itself — UMAP scatter
  coloured by cluster id before vs. after refinement, plus a before-cluster x after-cluster contingency
  heatmap (rows collapsing into one column = a merge; one row spread across several columns = a split).
  Only generated when refinement actually changed the labels (skipped if the pass is off, or converged with
  zero changes).

**Nearest-to-centroid example clips**
- `create_example_clips` now selects each cluster's example clips by proximity to the cluster's UMAP-embedding
  centroid instead of proximity to the cluster's median bout duration, whenever embedding data is available —
  two clips of "typical duration" can otherwise sit at opposite ends of a cluster in embedding space. Falls
  back to the old duration-based selection for standalone callers without embedding data.

**Built-in validation layer**
- Silhouette score, UMAP trustworthiness (vs. real feature space), MLP cross-validation accuracy.
- DLC quality gates (% interpolated frames per session).
- Faithfulness audit vs. B-SOiD reference parameters.
- Optional cluster-stability seed sweep: re-runs UMAP + HDBSCAN (+ the split/merge refinement pass, when
  enabled, via a shared `refine_clusters_iterative()` helper — so the reported stability reflects the same
  refined partition you actually get, not just the pre-refinement candidate) over N seeds, reports pairwise
  Adjusted Rand Index → `plots/cluster_stability.png`. Each seed's own DBCV (`relative_validity_` of its
  HDBSCAN selection) is logged individually and included in the sweep's return dict (`dbcv`) and the
  `cluster_stability` entry of the JSON validation report (`dbcv_scores`) — cluster-count/ARI stability and
  cluster tightness are different questions, so both are reported side by side rather than one standing in
  for the other. `seed_sweep_n` defaults to `6` (on by default) —
  stability is worth checking on every run rather than only when explicitly enabled; set to `0` to turn it
  off. If `mean_ari` comes back low (well under ~0.7), the UMAP embedding itself is likely seed-unstable for
  that dataset — see `consensus_clustering_enabled` above for the general fix, rather than tuning the HDBSCAN
  sweep, which doesn't touch the actual source of that instability.
- Optional PCA pre-reduction before UMAP (configurable via `pca_n_components`; default `"auto"`, triggers when sample/feature ratio < 5 and features > 50) prevents nearest-neighbour graph degradation in high-density 3D feature spaces.

**Cluster-stability (ARI) diagnostics — Aug 2026**
Investigation into seed-sweep partition instability (`seed_sweep_n` mean
ARI, ~0.41 on the 3-group validation dataset). Two embedding-side levers
(`umap_init="pca"`, `umap_n_epochs`) were prototyped, real-data-tested, and
then removed — neither gave a genuine stability improvement (`pca` init's
apparent ARI gain was a degenerate-convergence artifact: most seeds
collapsed to 4–5 clusters instead of the normal 3–34 range). The root cause
was instead traced to a discontinuous two-branch cluster-count selection
rule inside `run_hdbscan()`'s auto mode; see
`CLUSTER_SELECTION_SIMPLIFICATION_PLAN_2026-08.md` for the analysis and the
implemented fix, `hdbscan_selection_mode="floor_soft_cap"` (now the
default) — a real 3-group, 8-seed sweep test showed the old `"legacy"` rule
collapsing 5/8 seeds to a catastrophic 3-cluster outcome vs. 0/8 under the
new rule (mean ARI 0.345→0.589), with no quality regression on the
deterministic primary run.
- `hdbscan_selection_mode` (`"floor_soft_cap"` default | `"legacy"`) — auto-mode cluster-count selection rule for `run_hdbscan()`. `floor_soft_cap` never selects below `preferred_clusters_lo` if avoidable and only lightly penalises (`hdbscan_overshoot_penalty`) counts above `preferred_clusters_hi`, instead of the old rule's discontinuous fallback.
- `hdbscan_overshoot_penalty` (default `0.01`) — score penalty per cluster above `preferred_clusters_hi` under `floor_soft_cap`. `0` = no ceiling at all.
- `cluster_hierarchy_enabled` (default `True`) — saves `plots/cluster_hierarchy_dark.png` **and** `plots/cluster_hierarchy_light.png` (both theme variants always saved, regardless of the run's active `plot_theme`, so either can be pulled into external docs/talks without a re-run), a dendrogram of the final clustering's cluster centroids in feature space (`cluster_hierarchy_linkage`: `"ward"`/`"average"`/`"complete"`), to guide manual merging decisions. Redesigned Aug 2026: branch color is a single neutral tone throughout (topology alone shows which clusters are more similar — branch color was found to carry no real information and was actively misread as meaningful grouping), branches are drawn thicker, and each leaf gets a colored dot + `C{n} (n=size)` label in that cluster's permanent CUBE identity color (the same `_cmap()` mapping every other cluster plot uses) so a specific cluster can be spotted here and cross-referenced elsewhere at a glance.
- `seed_sweep_n_jobs` (default `1` = sequential | `-1` = auto-managed | `2`–cpu_count = pin an exact count) — dispatch of the cluster-stability seed sweep's per-seed UMAP+HDBSCAN(+refinement) fits. Changed from `-1` to `1` in Aug 2026 after a real run hit a Windows heap-corruption fault (`0xc0000374`, inside `ntdll.dll`'s own allocator — not reliably reproducible, so a safety margin rather than a pinned-down fix). Measured cost of going sequential: none — a real 6-seed sweep timed marginally *faster* sequential (490s) than parallel (503s), since the per-worker numba/BLAS single-threading already in effect had left little real parallel throughput to lose. `consensus_n_jobs` got the same change, same reasoning, same measured no-cost result (668s sequential vs 692s parallel for an 8-seed consensus run).
- `hdbscan_sweep_n_jobs` (default `1` = sequential | `-1` = auto-managed | `2`–cpu_count = pin an exact count) — dispatch of `run_hdbscan()`'s *primary* `min_cluster_size` sweep (not the seed-sweep or consensus call sites above). Changed from `-1` to `1` in Aug 2026 after `crash_diagnostics.log` captured this exact call site faulting with the identical `0xc0000374` heap-corruption signature as `seed_sweep_n_jobs`/`consensus_n_jobs`, despite the `_numba_single_thread`/`_blas_single_thread_for_dispatch` oversubscription guards already in place around each fit. This was the one remaining parallel-by-default sweep call site sharing the same hazard (joblib threading backend + numba JIT + HDBSCAN's Cython core contending under threads on Windows); it is now sequential for the same safety-margin reasoning as its two siblings.
- `seed_sweep_stability_bootstrap()` (`cube_core.py`) — an offline diagnostic function (not a GUI-exposed pipeline stage) that separates sampling-noise instability from UMAP-optimisation instability via bootstrap subsampling. Kept as a standalone tool for future validation passes.
- `plot_cluster_volatility()` — per-cluster volatility diagnostic (`plots/cluster_volatility.png`) that decomposes aggregate seed-sweep ARI into per-cluster stability via Hungarian-matched Jaccard overlap against a reference seed, so a low mean ARI can be traced to specific unstable clusters rather than treated as a single opaque number.

### Behavioural Analyser (cube_analyser)

**Combined Analysis tab**
- Per-animal group editor (up to 3 independent label columns — Label 3 doubles as **Animal ID**, supporting multi-factor and repeated-measures designs).
- **Group by:** dropdown independently selects which label defines the grouping axis in each analysis tab.
- Per-behaviour Kruskal-Wallis tests with Benjamini-Hochberg FDR correction across the full test family (4 metrics × k behaviour groups).
- Publication-ready multi-panel figures: ethogram, bar + dot plots with SEM and significance brackets.
- CSV export with raw p-values, FDR q-values, and significance annotations.

**Behaviour Statistics tab**
- **Two-part test design:** structural zeros (behaviour absent in a group) are separated from magnitude differences.
  - *Prevalence test* (Fisher's exact / Pearson χ²): is the behaviour present in both groups?
  - *Present-only Kruskal-Wallis*: among animals that expressed the behaviour, does the amount differ?
- `sig_driver` label per cluster: `magnitude`, `prevalence`, `both`, or `none`.
- Dunn's pairwise post-hoc (FDR-adjusted) for ≥3 group comparisons.

**Unbiased Analytics tab**
- **Experimental Design:** *Independent Groups* (default — different animals per group; Kruskal-Wallis omnibus + Dunn's/Mann-Whitney post-hoc, both FDR-pooled across every cluster/panel reported together) or *Repeated Measures* (same animals re-measured across the levels of "Group by", e.g. Baseline/Day 3/Day 7; paired via **Label 3 / Animal ID**). Repeated Measures uses Wilcoxon signed-rank (2 levels) or Friedman's test (≥3 levels) as the omnibus, with pairwise Wilcoxon signed-rank post-hoc — only animals with a matching Animal ID present at every level are included, and the status bar reports how many were matched/excluded. The structural-zero two-part decomposition (`sig_driver`) and the parametric ANOVA/η² columns are Independent Groups-only; Repeated Measures rows report `sig_driver = "n/a"`.
- Eleven plot modes:
| Mode | What it shows |
|---|---|
| Top-N Bar | Top N clusters ranked by group mean |
| Volcano | Effect size vs. significance for all clusters |
| Heatmap | Cluster × animal usage matrix (Ward-clustered) |
| Elbow/Silhouette | Within-cluster sum-of-squares and silhouette for Ward reclustering |
| Cluster Hierarchy | Re-rendered dendrogram of the primary clustering's pose-space (feature-centroid) hierarchy, loaded from `model/cluster_feature_centroids.npz` — the same hierarchy `cluster_hierarchy_*.png` shows, but theme-matched live and viewable without leaving the Analyser |
| Guided Merge | Hierarchy-informed automatic reclustering: greedily merges the closest pair of clusters by response-pattern/transition distance, but only when their pose-space distance also clears a user-set cap, and stops the moment no remaining merge is both close enough and clearly justified — "minimum necessary merges" instead of a global k. Includes a per-merge diagnostic table and the blended dendrogram's cophenetic correlation. See "Recombination Guide" in `CUBE_GUIDE.md` for full parameter/use-case guidance |
| Recombination | Ward-reclustered dendrogram (70% change-pattern + 30% transition cosine distance) at chosen k values, with an optional pose-distance cap (percentile slider) that vetoes any proposed merge whose constituent clusters look too different in pose space, splitting it back to singletons for display |
| Dist Matrix | Pairwise behavioural-distance matrix between animals |
| Transitions | Empirical cluster-to-cluster transition probability matrix |
| Cluster Stats | Per-cluster summary: mean bout length, frequency, total time |
| Cluster Validity | Silhouette diagram + 2D embedding scatter for the *primary* HDBSCAN clustering (loaded from the run's saved `umap_embedding.npy`/labels) — complements the other modes, which all operate on post-hoc reclustering of aggregated per-cluster behaviour stats rather than raw pose features |

Both Cluster Hierarchy and Guided Merge require `model/cluster_feature_centroids.npz`, saved automatically by Step 3 (Clustering Engine) alongside `cluster_hierarchy_*.png` — run folders produced before this feature was added need a Step 3 re-run before these two modes (and the Recombination pose-distance cap) become available; a status message in the UI explains this rather than failing silently. "Save Reclustered Groups" now has a Source toggle (Manual k-sweep / Guided Merge) so either partition can be pushed to the Group Editor or saved as a preset.

**Behavioural Explorer tab** — group-level dynamics:
- Diff Heatmap, Dwell Violin, Sankey diagram, Group Transition Networks, Energy Landscape.
- All flat 2-D UMAP scatters (Energy Landscape, UMAP comparison, UMAP groups) auto-select whichever pair of UMAP axes best separates the cluster labels (by silhouette score), instead of always plotting axes 1–2. Prevents cluster separation that lives on a 3rd UMAP axis (default `umap_n_components = 3`) from looking like a mixed blob in 2-D.
- Cluster/group labels on every Energy Landscape UMAP scatter are nudged toward a locally sparse spot and drawn above all points, so every label stays visible instead of being dropped when it would overlap a dense cluster of dots. A behaviour group spanning several disjoint original clusters is labelled at its largest constituent cluster rather than the mean of all its points, so the badge never lands in empty space between sub-clusters.
- Energy Landscape is now always a fixed 2-row × 3-column figure, pooled across all experimental groups/cohorts (previously one row per experimental group): row 1 ("All Clusters") labels every non-noise cluster in the UMAP scatter panel (no more size cutoff dropping small clusters), row 2 ("Behavioral Groups") shows user-defined behaviour groups instead, with a placeholder message when no groups are defined (or when groups are defined but none of their member cluster IDs exist in the current dataset). The heatmap/3D valley-star panels still cap at the top 5 highest-occupancy clusters (row 1) or groups (row 2).
- Row 2 ("Behavioral Groups") now computes its own energy surface rather than reusing row 1's. Each point's KDE weight is its **group's** total combined occupancy instead of its own individual cluster's, so every location belonging to a common group reads as low-energy (blue/common) even if that specific member cluster is individually rare — reflecting how much time the animal spends in the behavior overall, not just in that one pose-space variant of it. Points not belonging to any defined group are excluded (weight 0), same as noise.
- Fixed: a behaviour group's valley star (heatmap/3D panels) used to be positioned at the mean, or a bounding-box search, over *all* of its member clusters' points — for a group merging spatially disjoint clusters (e.g. the same behaviour expressed differently in pose-space) this could place the star in empty space between them, or have the 3D local-minimum search silently snap to whichever member is deepest and hide the rest. It's now anchored on the group's single highest-occupancy constituent cluster instead (matching the anchor logic already used for the scatter-panel group labels), while the group's ranking among the top-5 valleys still reflects its full combined occupancy across all members.
- Fixed: valley-star labels in the heatmap and 3D panels used a fixed offset direction, so nearby valleys' labels would overlap when nudged the same way. They now use the same outward spiral collision-avoidance search as the cluster/group scatter labels (trying increasing radii and 10 angles until a free spot is found), connected to their star with a leader line.
- The "Original Clusters vs. Behavioural Groups" comparison plot is now a 3×2 grid: row 1 plots original clusters and behavioural groups on the best-separating axis pair, row 2 plots the same two views on raw UMAP-1 vs UMAP-2, and row 3 adds raw UMAP-1 vs UMAP-3 (skipped if the embedding has only 2 components) so clusters/groups that only separate along the 3rd UMAP axis are no longer invisible. Unclustered/noise points in both "Original Clusters" panels use the same muted styling as unassigned points in the group panels, instead of an opaque palette color with its own label.
- Fixed: wide figures (notably Energy Landscape, ~3.4:1 aspect) no longer appear stretched vertically on screen. The plot panel only auto-fit width, leaving height pinned to the figure's original pixel size, which distorted the aspect ratio; height now tracks width to preserve the figure's native proportions, matching the exported image.
- Fixed: saved PNGs from every figure in this tab (Energy Landscape, UMAP comparison, UMAP groups, and the 3D-scatter/transition companion PNGs) could show solid black margins around the panels when the app was in light theme, instead of the theme's actual near-white background — the figure canvas's background was losing full opacity somewhere between creation and `savefig()`. Every figure now explicitly forces `fig.patch` to full opacity at creation, and every save call passes `transparent=False` explicitly.

**Group Predictor tab**
- Three parallel models (Frequency, Total Duration, Transition Probability) with LOO cross-validation.
- **Exhaustive optimal subset search:** evaluates every C(n, k) combination when ≤ 15,000 (parallel, CPU process pool). Falls back to greedy forward selection for larger spaces.
- **Shapley feature importances:** exact (≤ 8 clusters) or Monte Carlo (> 8) for fair, order-independent attribution.
- Conditional and nested permutation tests (Phipson & Smyth 2010 p-value correction).
- Cohen's κ, balanced accuracy, per-animal LOO probability strips, confusion matrices, ROC/AUC curves.
- Full documentation: [GROUP_PREDICTOR_REFERENCE.md](GROUP_PREDICTOR_REFERENCE.md).

---

## Robustness & Data Integrity

| Guard | Where | What it does |
|-------|-------|--------------|
| Triangulation shape validation | `cube_core.py: triangulate_cameras()` | Checks aniposelib return shape is `(N_frames, 3)` before writing to output array; logs WARN and zero-fills the affected bodypart on mismatch |
| H5/CSV frame count reconciliation | `cube_3d_dlc.py: mask_fill_frames_in_h5()` | Clamps or pads fill mask when H5 frame count diverges from CSV; now logs a WARN so data quality issues are visible |
| HDBSCAN legacy anchor deprecation | `cube_core.py: run_hdbscan()` | `hdbscan_mcs_anchor="full"` emits `DeprecationWarning`; use `"embedding"` (the default) for correctly-proportioned cluster granularity when UMAP subsamples |
| Permutation test stratification | `cube_analyser.py: GroupPredictor` | Uses `StratifiedShuffleSplit` instead of plain random shuffle so null-distribution folds preserve group proportions for imbalanced cohorts |
| Family FDR NaN filtering | `cube_analyser.py: family_wide_fdr()` | NaN p-values (from untestable clusters) are now excluded before BH pooling so they do not inflate the correction denominator and reduce power for valid tests |
| Plot-save failures no longer silent | `cube_core.py: _savefig()` | Previously caught save errors and only `print()`-ed them (invisible in the GUI log), then let the caller log a false "saved" success line — a plot could be missing on disk with the log claiming otherwise. Now propagates so the caller's own `try/except` logs an accurate `[WARN]` with the real traceback |
| Condensed-tree plot deferred-render crash | `cube_core.py: plot_cluster_validity()` | `hdbscan`'s `condensed_tree_.plot(colorbar=True)` can defer part of its rendering (e.g. colorbar layout) until the figure is actually drawn — a failure there previously surfaced only as a generic save error on the whole figure, well after the function's own graceful fallback path. Now forces rendering (`canvas.draw()`) inside the existing `try/except` so a failure here still gets the intended placeholder text. Also only requests cluster-boundary highlighting (`select_clusters=True`) when the passed labels still match `hdb_clf`'s own original selection (i.e. refinement/pruning hasn't since changed them) — otherwise it would visualise stale, pre-refinement cluster boundaries |
| Condensed-tree plot: upstream `hdbscan` Ellipse bug | `cube_core.py: plot_cluster_validity()` | Confirmed on real data: `hdbscan` 0.8.43's condensed-tree `select_clusters=True` path can raise `ValueError: setting an array element with a sequence` inside matplotlib's `Ellipse` transform — root-caused to `hdbscan/plots.py` falling back to a 1-element numpy array (`np.diff(np.percentile(...))`) instead of a scalar for a selection-ellipse height when a cluster has near-infinite persistence (an upstream library bug, not a CUBE one). Previously this always fell through to the "unavailable" placeholder. Now retries once with `select_clusters=False` (tree structure without cluster-boundary ovals) before giving up, and explicitly removes any colorbar axis a failed first attempt left orphaned on the figure so a successful retry never renders two overlapping colorbars |
| Near-duplicate embedding diagnostic | `cube_core.py: run_hdbscan()` | One `[DIAG]` log line (primary fit only, not per-seed/per-split) reports the fraction of near-duplicate embedding points before HDBSCAN when it exceeds 1% — the leading indicator of DBCV going degenerate / noise blowing up, typically caused by chronically low-confidence bodyparts producing flat-interpolated, near-identical feature vectors |
| Split-pass over-fragmentation | `cube_core.py: split_impure_clusters()` | The local re-cluster used to resolve one impure cluster was inheriting the whole-session `preferred_clusters_lo/hi` (default 8–30) and `hdbscan_fine_bias`, which pushed local selection toward the *top* of that session-scale range regardless of how few points were actually being re-clustered — observed splitting a single cluster into 29–30 tiny fragments in one pass on real data. Local re-clustering now uses its own small range (`preferred_clusters_hi` capped at `hdbscan_split_max_subclusters`, default `3`) with `hdbscan_fine_bias`/`hdbscan_leaf_bonus` disabled, plus a hard ceiling on accepted sub-cluster count as a second line of defence |
| Split attempted on too-small/high-dimensional subsets | `cube_core.py: split_impure_clusters()` | The minimum candidate-cluster size before attempting a local split was a flat 20 points, unrelated to feature dimensionality. On a real run with 915 features (28 bodyparts), splits were being attempted on 90–300 point subsets (`[PCA pre-UMAP]` log showing sample/feature ratios of 0.1–1.1) — far too few points to trust any resulting sub-clusters. New `hdbscan_split_min_points` (default `250`, ~5x UMAP's own PCA floor) skips a split attempt entirely below that size instead of running an unreliable one |
| Cluster colour aliasing beyond 20 clusters | `cube_core.py: _cmap()` | Cluster colours wrapped every 20 ids (`PALETTE[i % len(PALETTE)]`), so e.g. clusters 0, 20, 40, 60 all rendered in the exact same colour — made an 84-cluster `split_merge_refinement.png` look like it had far fewer distinct clusters than it did. Ids beyond the base palette now draw from a continuous colormap with golden-ratio hue spacing instead of wrapping, so every id gets a visually distinct colour; the first 20 ids are completely unaffected |
| Silent native crash during HDBSCAN sweep | `cube.py`, `cube_analyser.py` (module top) | On real runs, CUBE could hard-crash with no Python traceback and nothing past the startup line in the pipeline log — confirmed via Windows Event Viewer as `python.exe` faulting in `ucrtbase.dll`, exception `0xc0000409`. Root cause: MKL's `libiomp5md.dll` and scikit-learn's `VCOMP140.DLL` (two different OpenMP runtimes) both load into the same process — most likely to collide during the HDBSCAN sweep, which runs many candidate fits concurrently (`hdbscan_split_n_jobs`, default all cores). Intel's OpenMP runtime detects the second runtime and calls `abort()` (OMP Error #15) instead of tolerating it. The existing single-threaded BLAS/MKL env vars did not prevent this — only thread *count* was constrained, not the duplicate-runtime check. Fixed by also setting `KMP_DUPLICATE_LIB_OK=TRUE` before numpy is imported in both entry points |
| ~~Residual silent crash minutes after the HDBSCAN sweep~~ — theory retracted, see below | `cube.py`, `cube_analyser.py` (module top) | A real `0xc0000005` access-violation crash inside `python312.dll` (via Windows Event Viewer) was initially attributed to numba's `tbb` threading layer conflicting with the two OpenMP runtimes above, and "fixed" by forcing `NUMBA_THREADING_LAYER=workqueue`. **That diagnosis is now believed wrong and the change has been reverted.** The workqueue override was actively harmful: numba's own workqueue layer is explicitly documented as not thread-safe under concurrent access from multiple Python threads, and this codebase's own nested parallel dispatch (`seed_sweep_stability()`/`consensus_cluster()`/`split_impure_clusters()`, which all run numba-jitted UMAP/HDBSCAN work across several worker threads at once) does exactly that — reproduced directly: `Numba workqueue threading layer is terminating: Concurrent access has been detected.` The real cause of the original crash was very likely always the pynndescent bug documented below, unrelated to numba's threading layer entirely. Reverted to numba's own default (`tbb` on this machine), which its own docs recommend specifically *because* it is thread-safe under concurrent access — confirmed by reproducing the crash-triggering nested scenario successfully with only the pynndescent fix applied, no numba override at all |
| HDBSCAN primary sweep: numba + BLAS thread-pool oversubscription crash | `cube_core.py: run_hdbscan()` | The new `crash_diagnostics.log` faulthandler caught this one directly on a real run: `Fatal Python error: Aborted` (SIGABRT) with dozens of live threads, all sitting in `multiprocessing.pool.worker`. Root cause: the primary sweep's `_fit_one()` — dispatched across a joblib thread pool sized by `_sweep_n_jobs` — was missing BOTH oversubscription guards its three sibling functions (`split_impure_clusters()`, `seed_sweep_stability()`, `consensus_cluster()`) already apply: (1) `_numba_single_thread()` inside each worker, capping HDBSCAN's numba-jitted core-distance code to 1 thread per worker instead of letting it spin up its own full-width (all-CPU-core) pool on top of the outer joblib pool; and (2) `_blas_single_thread_for_dispatch()` wrapping the *entire* `Parallel(...)` dispatch call (not per-worker — threadpoolctl's limiter is process-global, so per-worker enter/exit would race), capping BLAS/MKL so PCA/distance ops inside each worker don't also each try to claim every core. Missing both meant total concurrency could reach roughly `_sweep_n_jobs x core_count` threads — severe enough to trigger a fatal abort. An audit of every other `Parallel(...)` call site in `cube_core.py`, `cube.py`, and `cube_analyser.py` turned up no other instance of this gap in cube_core.py's own dispatch sites (see the pynndescent row below for a gap that *was* missed, in a third-party library) — `cube_analyser.py`'s own parallel dispatch sites are pure sklearn/BLAS work already covered by its unconditional module-level `OMP_NUM_THREADS=1` etc., with no numba calls in that file at all |
| HDBSCAN primary sweep: heap-corruption crash survives the oversubscription guards (fixed by going sequential) | `cube_core.py: run_hdbscan()` DEFAULTS | Despite the `_numba_single_thread()`/`_blas_single_thread_for_dispatch()` guards above (row directly above this one), `crash_diagnostics.log` captured a *different* fatal signature at the exact same call site on a real run: `Windows fatal exception: code 0xc0000374` (heap corruption, inside the CRT allocator) with dozens of threads live in `hdbscan_.py: _tree_to_labels` under `multiprocessing.pool.worker`. This is the same crash class already seen (and worked around, not root-caused) for `seed_sweep_n_jobs`/`consensus_n_jobs` — see that entry in Advanced Settings — just not previously observed at the primary-sweep call site, whose default (`hdbscan_sweep_n_jobs=-1`) had been left parallel. Brought in line with its two siblings: default changed to `1` (sequential). Same caveat applies — not reliably reproducible on demand, so this is a safety margin rather than a pinned-down fix, and the guards above remain in place and still matter whenever a user pins `hdbscan_sweep_n_jobs` back to a parallel value |
| pynndescent (UMAP's nearest-neighbour engine): hardcoded `n_jobs=-1` causes nested thread-pool oversubscription crash | `cube_core.py: _patch_pynndescent_thread_safety()` | A *second* real SIGABRT crash, with the same dozens-of-`multiprocessing.pool.worker`-threads signature as the row above, survived even after that fix — its full faulthandler stack trace (this time complete, not truncated) pinpointed the actual call chain: `seed_sweep_stability()`'s per-seed worker → `run_umap()` → `pynndescent.rp_trees.rptree_leaf_array_parallel()`. Reading pynndescent 0.5.x's source confirmed the root cause: that one function — a post-processing step that runs unconditionally after every single UMAP fit, regardless of size — **hardcodes** `joblib.Parallel(n_jobs=-1, ...)` with no parameter to override it. CUBE's own `UMAP(n_jobs=1)` setting has zero effect on this specific step (confirmed by tracing the call chain: `NNDescent.__init__` → `rptree_leaf_array` → `rptree_leaf_array_parallel`, the last of which never receives or forwards `n_jobs` at all). Harmless when a UMAP fit runs standalone (the burst of up to all-CPU-core threads is brief and nothing else is contending for cores) — but `run_umap()` is also called from inside already-parallel outer joblib workers in `split_impure_clusters()`/`seed_sweep_stability()`/`consensus_cluster()`, and every one of those outer workers independently bursts its own full-width thread pool for this step *at the same time*, none of which is touched by either `_numba_single_thread()` (numba-specific) or `_blas_single_thread_for_dispatch()` (BLAS-specific) — this is joblib's own dispatch, a third, previously-unaudited mechanism. `joblib.parallel_config(n_jobs=1)` does **not** fix this — confirmed empirically that it only supplies a default for `Parallel(...)` calls that omit `n_jobs`, and pynndescent passes `n_jobs=-1` explicitly, which an outer context cannot override. Fixed by monkeypatching `rptree_leaf_array_parallel` at `cube_core.py` import time to always dispatch with `n_jobs=1` — verified bit-for-bit identical results, and confirmed via benchmark to cost no measurable wall-clock time (extracting each tree's leaf-index array from an already-built tree is cheap; this was never the bottleneck). Verified end-to-end: reproducing the exact crash-triggering nested scenario (8 concurrent outer workers, each calling `run_umap()`) completed cleanly with this fix, peaking at 14 threads instead of crashing |
| Crash diagnostics: forensics-proof logging | `cube.py`, `cube_analyser.py` (module top) | A third crash the same day left **no trace anywhere** — no WER report, no System-log entry, no Reliability Monitor entry, `python.exe` simply stopped existing — because it wasn't a catchable Python exception and (unlike the two rows above) apparently didn't even trip Windows' own crash reporter. `try/except` around pipeline stages can never see a fault like this. Added two safety nets installed at process start, before any heavy numeric/native code runs: (1) stdlib `faulthandler.enable(all_threads=True)`, which intercepts fatal native signals (SIGSEGV/access-violation, SIGABRT, SIGFPE, SIGILL) and prints exactly which Python thread and source line was executing at the moment of the fault — the only way to localise a crash inside a compiled extension (MKL/numba/HDBSCAN/OpenCV) to an actual pipeline stage; and (2) `sys.excepthook`/`threading.excepthook` overrides, which catch plain uncaught Python exceptions that would otherwise exit the process silently (e.g. raised in a background thread, whose default behaviour is to print to stderr and vanish with no GUI trace). Both write to a persistent, append-only `CUBE_logs/crash_diagnostics.log` that survives across runs and app restarts, so a crash that kills the process mid-write is still captured immediately beforehand. `cube_analyser.py`'s install is guarded against double-installing when loaded as a submodule of `cube.py` (the normal Step 5 launch path) |
| MLP cross-validation crash on tiny clusters | `cube_core.py: train_mlp()` | `cv_folds` (default 5) was clamped against the *number of classes* but not the *smallest class's member count*. A cluster surviving `min_cluster_freq` pruning with only 2-4 bins made `cross_val_score`'s internal `StratifiedKFold` raise, crashing the run. Now also clamps against the smallest class size, and falls back to a plain train-set score (no CV) when fewer than 2 folds are possible |
| HDBSCAN sweep divide-by-zero | `cube_core.py: run_hdbscan()` | Setting `hdbscan_sweep_n_steps` (or the auto-widened value) to `1` divided by `(n_steps - 1) = 0` when building the sweep's mcs percentage list — unguarded for the primary sweep (the local split-cluster sweep already caught this). `n_steps` is now floored at 2 |
| UMAP auto-PCA `n_components` exceeding sample count | `cube_core.py: run_umap()` | The auto-PCA pre-reduction step clamped its target dimensionality against `n_features - 1` but never against `n_samples`, so a small point subset (e.g. a local re-embedding inside `split_impure_clusters`) could request more PCA components than it had samples and raise `ValueError`. Now also clamped to `n_samples` |
| Low-confidence bins crashing/corrupting HMM smoothing | `cube_core.py: train_hmm()`, `decode_hmm()` | When `mlp_confidence_thresh > 0`, bins the MLP was unsure about are labelled `-1` ("unclassified"). These flowed unmodified into the categorical HMM, which is fit on exactly `n_clusters` valid symbols — `-1` either raised inside `hmmlearn` (silently caught, disabling HMM smoothing for the whole run with just a WARN) or risked corrupting decoding via negative array indexing. A new `_sanitize_labels_for_hmm()` helper now forward/backward-fills any out-of-range label with its nearest temporally-valid neighbour before training or decoding — consistent with the HMM's own self-persistence assumption, and applied on both the training sequences and every decode call |
| `smooth_boxcar` returned more rows than it was given on very short recordings | `cube_core.py: smooth_boxcar()` | Found by the new automated test suite (`tests/unit/test_core_pure.py`). `np.convolve(a, kernel, mode="same")` returns `max(len(a), len(kernel))` rows, not `len(a)`, whenever the smoothing window is wider than the recording (e.g. a 1-frame input with a 2-frame window) — silently desyncing frame counts for anything downstream. Fixed by clamping the window to the recording length before building the kernel |
| `smooth_boxcar` crashed on a 0-frame input | `cube_core.py: smooth_boxcar()` | Found by the same test suite. `np.convolve` raises `ValueError: a cannot be empty` on an empty array whenever the window is wider than 1 frame. Fixed with an explicit empty-input guard that returns immediately |
| HMM training (`train_hmm`/`train_hmm_soft`) was not reproducible across identical calls | `cube_core.py: train_hmm()`, `train_hmm_soft()` | Found by the same test suite. Neither function passed `random_state` to its underlying `hmmlearn` model (`CategoricalHMM`/`GaussianHMM`), so back-to-back fits on identical inputs could diverge by ~1e-5 in the learned transition/emission matrices — unlike `run_umap`/`train_mlp`, which were already seeded. Fixed by adding a `hmm_random_state` config key (default `42`) threaded into both HMM constructors, so a saved `bsoid_model.pkl`'s HMM component is now reproducible the same way its UMAP/MLP components already were |

---

## Output Layout

```
<output_dir>/
  bout_lengths/
    <stem>_bout_lengths.csv / _hmm.csv     ← MLP / HMM bouts (preferred: _hmm)
    <stem>_bout_lengths_hmm_enriched.csv   ← (if kinematic_directedness_enabled) canonical
                                              3 columns + net_displacement_px, path_length_px,
                                              straightness_ratio, mean_speed_px_s,
                                              heading_consistency; never written by default
    <stem>_frame_labels.csv / _hmm.csv     ← per-frame labels
    <stem>_epochs.csv / _hmm.csv
  model/
    bsoid_model.pkl                         ← scaler, UMAP, HDBSCAN, MLP, PCA, config
    umap_embedding.npy
    session_bin_ranges.json
    hmm_model.pkl
    feature_config.json                     ← feature_version: "v2" or "v3d"
  plots/
    umap_embedding.png
    umap_3d.html
    transition_matrix.png
    hmm_transition_matrix.png
    hmm_syntax_network.png
    state_space_projection.png
    cluster_stability.png                   ← (if seed_sweep_n > 0)
    cluster_volatility.png                  ← (if seed_sweep_n > 0) per-cluster ARI decomposition
    cluster_validity.png                    ← silhouette diagram + HDBSCAN condensed-tree plot
    cluster_hierarchy_dark.png               ← (if cluster_hierarchy_enabled) feature-space dendrogram, dark theme
    cluster_hierarchy_light.png              ← (if cluster_hierarchy_enabled) feature-space dendrogram, light theme
  cluster_kinematics.csv
  cluster_confidence.csv                    ← per-cluster visibility/confidence flags (low_confidence_flag)
  videos/
    example_clips/<cluster_id>/
    labeled_videos/
```

---

## File Reference

| File | Description |
|------|-------------|
| `cube.py` | Main pipeline GUI |
| `cube_core.py` | Analysis engine (`BSoidEngine`) |
| `cube_3d_dlc.py` | 3D DLC + Anipose pipeline module |
| `cube_analyser.py` | Behavioural analyser GUI |
| `cube_video_explorer.py` | Cluster annotation GUI |
| `CUBE.bat` | Windows launcher |
| `install_shortcut.ps1` | Desktop shortcut installer |
| `theme.txt` | UI theme preference (`dark` / `light`) |
| `CUBE_GUIDE.md` | Full user guide |
| `GROUP_PREDICTOR_REFERENCE.md` | Group Predictor complete reference |
| `TEST_README.md` | Test suite documentation |
| `test_group_predictor.py` | Headless Group Predictor backend test |
| `test_gp_timing_calibration.py` | Group Predictor timing/calibration test |
| `test_non_chance_transitions.py` | Transition model significance test |

---

## Dependencies

```
python=3.10
pillow
opencv-python-headless
scipy
scikit-learn
umap-learn
customtkinter
plotly
hmmlearn
hdbscan           (conda-forge)
deeplabcut        (for Step 1 DLC inference)
aniposelib        (for Step 1b 3D mode)
imageio-ffmpeg    (for quad composite + skeleton video)
```

All runtime dependencies are pre-installed in the `CUBE` conda environment. No build step required.

---

## Running Tests

CUBE has an automated `pytest` suite under `tests/` (unit + integration tests for `cube_core.py` and `cube_analyser.py`'s pure logic). Always invoke it through the mandatory `CUBE` conda environment's Python — never bare `pytest`/`python`.

Full suite (includes the slow end-to-end pipeline regression, ~30-60s):
```bat
"C:\Users\param\anaconda3\envs\CUBE\python.exe" -m pytest tests\ -v
```

Fast subset only (excludes the `slow`-marked end-to-end regression, seconds not minutes — use this for day-to-day iteration):
```bat
"C:\Users\param\anaconda3\envs\CUBE\python.exe" -m pytest tests\ -v -m "not slow"
```

`slow` marks the one full `BSoidEngine.run()` regression test (Phase T7); `integration` marks any test exercising multiple pipeline stages together (discovery+loading, config merging, algorithm-stage smoke tests, the full pipeline). Both markers are registered in `pyproject.toml`. See `Automated_Test_Suite_Plan.md` for the full test-suite design rationale and `Test_Suite_Implementation_Report.md` for what's implemented, coverage numbers, and known gaps.
