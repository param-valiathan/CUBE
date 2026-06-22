<p align="center">
  <img src="CUBE_logo.png" alt="CUBE Logo" width="240"/>
</p>

<h1 align="center">CUBE — Comprehensive Unsupervised Behavioral Explorer</h1>

<p align="center">
  Automated, end-to-end pipeline for discovering and quantifying animal behaviour from video — no manual labelling required.
</p>

<p align="center">
  <b>Current release: v4</b> &nbsp;·&nbsp; Upcoming: <a href="3D_Integration_Plan_v5.md">v5 — 3D dual-camera tracking</a>
</p>

---

## Overview

CUBE integrates **DeepLabCut (DLC) pose estimation** with unsupervised machine learning (UMAP + HDBSCAN + MLP classifier) to automatically segment, cluster, and analyse animal behavioural patterns from raw video — with no predefined behavioural categories.

The pipeline runs from raw video to full statistical group comparisons in five steps. It is built on the **B-SOiD** methodology (Hsu & Yttri 2021, *Nat. Commun.* 12:5188) and has been substantially extended with improved feature extraction, automatic cluster quality selection, HMM temporal smoothing, and a comprehensive behavioural analyser.

**What CUBE does:**
- Batch DLC inference (SuperAnimal quadruped model, with Smart Adapt for large datasets)
- Multi-scale feature extraction with body-size normalisation, angular features, within-bin variance, and temporal lag drift
- Automatic HDBSCAN cluster sweep with DBCV-guided selection
- MLP classifier that generalises discovered clusters to all sessions
- HMM wrapper that eliminates MLP state-flickering and enforces temporal consistency
- Interactive cluster annotation via embedded video clips
- Full statistical analysis: ethograms, group comparisons, transition dynamics, reclustering
- Multivariate group classification (Group Predictor) with LOO-CV and permutation testing
- Session autosave — resume after a crash from the last completed step

---

## Documentation

| Document | Purpose |
|---|---|
| [CUBE_GUIDE.md](CUBE_GUIDE.md) | **Full user guide** — installation, step-by-step walkthrough, all features, troubleshooting |
| [GROUP_PREDICTOR_REFERENCE.md](GROUP_PREDICTOR_REFERENCE.md) | **Complete Group Predictor reference** — algorithms, figures, controls, interpretation, caveats |
| [3D_Integration_Plan_v5.md](3D_Integration_Plan_v5.md) | v5 roadmap — dual-camera 3D tracking extension |
| [TEST_README.md](TEST_README.md) | Test suite documentation |

---

## Quick Start

### Installation

```bat
conda create -n CUBE python=3.10
conda activate CUBE
pip install pillow opencv-python-headless scipy scikit-learn umap-learn customtkinter plotly hmmlearn
conda install -c conda-forge hdbscan
```

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

## The Five-Step Pipeline

| Step | File | Description |
|------|------|-------------|
| **1** | `cube.py` | DLC inference — run SuperAnimal on raw videos or load existing H5/CSV files |
| **2** | `cube_core.py` | Load, normalise, and smooth DLC output |
| **3** | `cube_core.py` | Feature extraction → UMAP → HDBSCAN sweep → MLP → HMM smoothing |
| **4** | `cube_video_explorer.py` | Annotate clusters by watching example video clips |
| **5** | `cube_analyser.py` | Group statistics, ethograms, transition analyses, Group Predictor |

See [CUBE_GUIDE.md](CUBE_GUIDE.md) for full details on each step.

---

## Architecture

CUBE is a four-file application:

| File | Role |
|------|------|
| `cube_core.py` | Pure analysis engine — all numerical logic, no GUI code |
| `cube.py` | Main five-step pipeline GUI (customtkinter). Lazy-loads the other two at runtime |
| `cube_analyser.py` | Standalone behavioural analyser (Step 5) |
| `cube_video_explorer.py` | Standalone cluster annotator (Step 4) |

---

## v4 Feature Summary

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

**MLP classifier with cross-validation**
- Trains on HDBSCAN-labelled bins, generalises to all sessions including those not used for clustering.
- Cross-validated accuracy (standard and balanced) reported in the log.
- Optional class balancing, early stopping, and larger architecture (`256,128,64`) available in Advanced Settings.

**Post-hoc HMM smoothing**
- `hmmlearn.CategoricalHMM` fitted by Baum-Welch EM on the MLP output.
- Viterbi decoding recovers the most probable state sequence, eliminating single-frame flickers.
- Produces `*_hmm` variants of all output CSVs. The Analyser prefers HMM files automatically.
- Diagnostic plots: bout-duration comparison, learned transition matrix (two panels), ethogram overlay, syntax network.

**Built-in validation layer**
- Silhouette score, UMAP trustworthiness (vs. real feature space), MLP cross-validation accuracy.
- DLC quality gates (% interpolated frames per session).
- Faithfulness audit vs. B-SOiD reference parameters.
- Optional cluster-stability seed sweep: re-runs UMAP + HDBSCAN over N seeds, reports pairwise Adjusted Rand Index → `plots/cluster_stability.png`.

### Behavioural Analyser (cube_analyser)

**Combined Analysis tab**
- Per-animal group editor (up to 3 independent label columns — supports multi-factor designs).
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

**Unbiased Analytics tab** — eight plot modes:
| Mode | What it shows |
|---|---|
| Top-N Bar | Top N clusters ranked by group mean |
| Volcano | Effect size vs. significance for all clusters |
| Heatmap | Cluster × animal usage matrix (Ward-clustered) |
| Elbow/Silhouette | Within-cluster sum-of-squares and silhouette for Ward reclustering |
| Recombination | Ward-reclustered dendrogram (70% change-pattern + 30% transition cosine distance) |
| Dist Matrix | Pairwise behavioural-distance matrix between animals |
| Transitions | Empirical cluster-to-cluster transition probability matrix |
| Cluster Stats | Per-cluster summary: mean bout length, frequency, total time |

**Behavioural Explorer tab** — group-level dynamics:
- Diff Heatmap, Dwell Violin, Sankey diagram, Group Transition Networks, Energy Landscape.

**Group Predictor tab**
- Three parallel models (Frequency, Total Duration, Transition Probability) with LOO cross-validation.
- **Exhaustive optimal subset search:** evaluates every C(n, k) combination when ≤ 15,000 (parallel, CPU process pool). Falls back to greedy forward selection for larger spaces.
- **Shapley feature importances:** exact (≤ 8 clusters) or Monte Carlo (> 8) for fair, order-independent attribution.
- Conditional and nested permutation tests (Phipson & Smyth 2010 p-value correction).
- Cohen's κ, balanced accuracy, per-animal LOO probability strips, confusion matrices, ROC/AUC curves.
- Full documentation: [GROUP_PREDICTOR_REFERENCE.md](GROUP_PREDICTOR_REFERENCE.md).

### Other v4 Features

**Analysis v2.1 corrected defaults**
- `umap_min_dist` 0.0 → 0.1 (prevents DBCV non-finite values at HDBSCAN stage).
- HDBSCAN `min_cluster_size` anchored to clustered point count (not full bin count), so cluster granularity no longer shifts at the UMAP subsampling boundary.
- Angular features skipped when no spine landmarks match by keyword (`angular_fallback=False`).
- GUI Advanced Settings now derive from engine defaults — single source of truth, no GUI/engine mismatch.
- UMAP trustworthiness measured against real feature space (pre-PCA).

**3D UMAP visualisation**
- `umap_3d.html` — interactive Plotly figure with three pairwise projection panels.

**Transition matrix distinction**
- `transition_matrix.png` — empirical (every consecutive frame-pair, raw MLP labels).
- `hmm_transition_matrix.png` — model-learned A matrix from Baum-Welch; two panels (raw diagonal + off-diagonal renormalised to expose switching grammar).

**Compatibility mode**
- `compat_mode = "legacy_v2"` in Advanced Settings restores all pre-v2.1 numeric defaults exactly for reproduction of earlier results.

---

## Output Layout

```
<output_dir>/
  bout_lengths/
    <stem>_bout_lengths.csv / _hmm.csv     ← MLP / HMM bouts (preferred: _hmm)
    <stem>_frame_labels.csv / _hmm.csv     ← per-frame labels
    <stem>_epochs.csv / _hmm.csv
  model/
    bsoid_model.pkl                         ← scaler, UMAP, HDBSCAN, MLP, PCA, config
    umap_embedding.npy
    session_bin_ranges.json
    hmm_model.pkl
  plots/
    umap_embedding.png
    umap_3d.html
    transition_matrix.png
    hmm_transition_matrix.png
    hmm_syntax_network.png
    state_space_projection.png
    cluster_stability.png                   ← (if seed_sweep_n > 0)
  videos/
    example_clips/<cluster_id>/
    labeled_videos/
```

---

## Upcoming: v5

v5 will add **3D dual-camera DeepLabCut tracking**:
- Run 2D DLC on each camera independently
- Triangulate to 3D using aniposelib calibration
- Output a merged 3D H5 that flows into Steps 2–5 unchanged (via bodypart doubling with `_z` pseudo-keypoints)
- User-selectable toggle — 2D workflow fully preserved

See [3D_Integration_Plan_v5.md](3D_Integration_Plan_v5.md) for full implementation details.

---

## File Reference

| File | Description |
|------|-------------|
| `cube.py` | Main pipeline GUI |
| `cube_core.py` | Analysis engine (`BSoidEngine`) |
| `cube_analyser.py` | Behavioural analyser GUI |
| `cube_video_explorer.py` | Cluster annotation GUI |
| `CUBE.bat` | Windows launcher |
| `install_shortcut.ps1` | Desktop shortcut installer |
| `theme.txt` | UI theme preference (`dark` / `light`) |
| `CUBE_GUIDE.md` | Full user guide |
| `GROUP_PREDICTOR_REFERENCE.md` | Group Predictor complete reference |
| `3D_Integration_Plan_v5.md` | v5 3D extension roadmap |
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
hdbscan         (conda-forge)
deeplabcut      (for Step 1 only)
aniposelib      (for v5 3D mode, not yet implemented)
```

All runtime dependencies are pre-installed in the `CUBE` conda environment. No build step required.
