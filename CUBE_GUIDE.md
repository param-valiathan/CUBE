# CUBE User Guide
### Comprehensive Unsupervised Behavioral Explorer — v4

---

## Contents

1. [What is CUBE?](#1-what-is-cube)
2. [Installation and Setup](#2-installation-and-setup)
3. [Launching CUBE](#3-launching-cube)
4. [The Five-Step Pipeline](#4-the-five-step-pipeline)
   - [Step 1 — DLC Inference](#step-1--dlc-inference)
   - [Step 2 — Load & Smooth](#step-2--load--smooth)
   - [Step 3 — Feature Extraction, UMAP, and Clustering](#step-3--feature-extraction-umap-and-clustering)
   - [Step 4 — Cluster Annotation (Video Explorer)](#step-4--cluster-annotation-video-explorer)
   - [Step 5 — Behavioural Analysis (Analyser)](#step-5--behavioural-analysis-analyser)
5. [Understanding the Output Files](#5-understanding-the-output-files)
6. [The Behavioural Analyser (cube_analyser)](#6-the-behavioural-analyser-cube_analyser)
   - [Loading Data](#loading-data)
   - [Combined Analysis Tab](#combined-analysis-tab)
   - [Behaviour Statistics Tab](#behaviour-statistics-tab)
   - [Unbiased Analytics Tab](#unbiased-analytics-tab)
   - [Group Predictor Tab](#group-predictor-tab)
7. [HMM Smoothing](#7-hmm-smoothing)
8. [Advanced Settings](#8-advanced-settings)
9. [Session Autosave and Resume](#9-session-autosave-and-resume)
10. [Compatibility Mode](#10-compatibility-mode)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What is CUBE?

CUBE integrates **DeepLabCut (DLC) pose estimation** with an unsupervised machine learning pipeline to automatically discover and quantify animal behavioural patterns from video — no manual labelling required.

The core methodology is based on **B-SOiD** (Hsu & Yttri 2021, *Nat. Commun.* 12:5188), extended with:
- Multi-scale feature extraction (100/200 ms windows)
- Body-size normalisation and angular body-axis features
- HDBSCAN cluster sweep with DBCV-guided automatic selection
- Multinomial HMM wrapper for temporal smoothing (eliminates state flickering)
- Interactive video annotation and group-level statistical analysis
- Multivariate group classification (Group Predictor) with permutation testing

**What CUBE does not require:**
- Manual behavioural labelling
- Predefined behavioural categories
- Frame-by-frame annotation

**What you need to provide:**
- Video recordings of your animals
- (Optionally) a DLC SuperAnimal quadruped model — CUBE can run inference automatically

---

## 2. Installation and Setup

### Requirements

- Windows 10/11
- Anaconda or Miniconda
- CUDA-capable GPU (recommended for DLC inference, not required for analysis)

### Conda Environment

Create the CUBE environment once:

```bat
conda create -n CUBE python=3.10
conda activate CUBE
pip install pillow opencv-python-headless scipy scikit-learn umap-learn customtkinter plotly hmmlearn
conda install -c conda-forge hdbscan
```

DeepLabCut (for Step 1 inference only):
```bat
pip install deeplabcut[gui]
```

### Verify the Environment

```bat
"C:\Users\<yourname>\anaconda3\envs\CUBE\python.exe" -c "import umap; print('OK')"
```

### Desktop Shortcut (Windows)

Run `install_shortcut.ps1` once (right-click → Run with PowerShell). This creates a CUBE shortcut on your desktop that launches via `CUBE.bat` without needing to open a terminal.

---

## 3. Launching CUBE

Always launch using the full Python path or the CUBE.bat launcher:

```bat
"C:\Users\<yourname>\anaconda3\envs\CUBE\python.exe" cube.py
```

Or double-click the desktop shortcut created by `install_shortcut.ps1`.

You can also launch the standalone tools directly:

```bat
rem Behaviour Analyser (Step 5 only)
"C:\Users\<yourname>\anaconda3\envs\CUBE\python.exe" cube_analyser.py

rem Video Explorer / Cluster Annotator (Step 4 only)
"C:\Users\<yourname>\anaconda3\envs\CUBE\python.exe" cube_video_explorer.py
```

---

## 4. The Five-Step Pipeline

The main CUBE window presents five sequential steps. Each step must complete before the next can start. The pipeline saves progress automatically so you can close CUBE and resume later.

---

### Step 1 — DLC Inference

**Purpose:** Run DeepLabCut pose estimation on your videos to produce bodypart coordinate files (H5 or CSV).

**If you already have DLC output files (H5/CSV), skip to Step 2.**

**Inputs:**
- Video folder(s) containing `.mp4` / `.avi` / `.mov` files
- Output folder for results

**Options:**
- **SuperAnimal Quadruped** — uses DLC's pretrained SuperAnimal model (no training required; works for mice, rats, and other quadrupeds)
- **Smart Adapt mode** — adapts the model once on a representative video, then reuses the adapted weights for all remaining videos. Recommended for large batches where animals all look similar.

**What CUBE produces:** one `*_filtered.h5` file per video, placed in the output folder alongside the source videos. The filter step removes low-confidence keypoint detections (likelihood < threshold) by linear interpolation.

**File matching:** CUBE matches DLC H5/CSV files to videos by stem name, filename prefix, or `YYYYMMDD_HHMMSS` timestamp. If a file cannot be matched to a video, it is included in the analysis without video-linked features.

---

### Step 2 — Load & Smooth

**Purpose:** Load DLC output files, normalise low-likelihood frames, and apply temporal smoothing.

**What happens internally:**
1. Each H5/CSV file is normalised to a standard 3-level `(scorer, bodyparts, coords)` MultiIndex — any DLC format variant is handled automatically.
2. Frames where any tracked bodypart has likelihood below the threshold are interpolated from neighbours.
3. A **centred boxcar moving average** is applied across each coordinate series (default window = 3 frames). This reduces single-frame jitter without shifting timing.

**Key setting:** `likelihood_threshold` (default 0.6). Frames below this are interpolated. Raise it if bodypart tracking is noisy; lower it if too many frames are being interpolated.

---

### Step 3 — Feature Extraction, UMAP, and Clustering

This is the core analysis step. It takes ~5–30 minutes depending on data size.

**Feature Extraction**

Multi-scale (100 ms and 200 ms windows) features are computed for each time bin:
- **Pairwise distances** between all tracked bodyparts
- **Velocities** (displacement per bin) for each bodypart
- **Angular features** (body axis orientation from spine landmarks, if detected)
- **Within-bin variance** (captures postural stability vs. movement)
- **Body-size normalisation** (distances are normalised by animal size, making the model invariant to camera distance)

Output shape: `(n_features, n_bins)` where one bin = 100 ms.

**UMAP**

Reduces the high-dimensional feature space to 3 dimensions for visualisation and clustering. Key parameters:
- `n_neighbors` — adapts automatically to data size (controls local vs. global structure)
- `min_dist` — default 0.1; controls point packing in the embedding
- PCA pre-reduction is applied automatically when `n_features ≥ n_samples / 5`

**HDBSCAN Cluster Sweep**

Rather than requiring you to choose a cluster count, CUBE sweeps `min_cluster_size` across 40 steps and tries both `eom` and `leaf` methods. The partition is selected by **DBCV** (Density-Based Clustering Validation, `relative_validity_`), which measures cluster quality without requiring a predefined number of clusters.

Rare clusters (< 0.2% of all bins) are pruned before MLP training.

**MLP Classifier**

A scikit-learn `MLPClassifier` is trained on the HDBSCAN-labelled bins. The MLP learns to map feature vectors to cluster labels, then applies this to every session — including sessions not used for clustering. Cross-validated accuracy is reported in the log.

**HMM Smoothing**

A Multinomial Hidden Markov Model (HMM) is fitted to the MLP output using Baum-Welch EM. Viterbi decoding then re-labels every frame, enforcing temporal consistency. This eliminates rapid flickering between states that the MLP produces for ambiguous frames. See [Section 7](#7-hmm-smoothing) for details.

**Output plots generated:**
- `umap_embedding.png` — 2D UMAP scatter coloured by cluster
- `umap_3d.html` — interactive 3D Plotly visualisation
- `transition_matrix.png` — empirical transition matrix (MLP labels)
- `hmm_transition_matrix.png` — HMM learned transition matrix (two panels: raw A matrix and off-diagonal renormalised)
- `cluster_stability.png` — seed sweep stability (if `seed_sweep_n > 0`)

---

### Step 4 — Cluster Annotation (Video Explorer)

**Purpose:** Assign human-interpretable names to each HDBSCAN cluster by watching example video clips.

**Launch:** Click **Open Video Explorer** in Step 4, or run `cube_video_explorer.py` standalone.

**Workflow:**
1. The explorer loads example video clips for each cluster automatically (clips were saved during Step 3).
2. For each cluster, play 3–5 example clips and assign a name (e.g., "Rearing", "Grooming", "Locomotion").
3. Clusters can be grouped — e.g., assign multiple clusters to the same named behaviour category.
4. Annotations are saved and appear in the Analyser.

**Tips:**
- Watch at least 5–10 clips per cluster before assigning a name — behaviour within a cluster can be heterogeneous.
- If two clusters look identical, they may represent the same behaviour at different speeds or body angles. Assign them the same group name.
- Unlabelled clusters are displayed as "Cluster N" in the Analyser.

---

### Step 5 — Behavioural Analysis (Analyser)

**Launch:** Click **Open Analyser** in Step 5, or run `cube_analyser.py` standalone.

The analyser loads the output files from the pipeline and provides four tabs of analyses. See [Section 6](#6-the-behavioural-analyser-cube_analyser) for a full description.

---

## 5. Understanding the Output Files

After Step 3, all output is written to your chosen output folder:

```
<output_dir>/
  bout_lengths/
    <stem>_bout_lengths.csv          ← MLP bout table (preferred: _hmm version)
    <stem>_bout_lengths_hmm.csv      ← HMM-smoothed bouts (used by analyser by default)
    <stem>_frame_labels.csv          ← per-frame: frame, time_s, label
    <stem>_frame_labels_hmm.csv      ← HMM-smoothed per-frame labels
    <stem>_epochs.csv                ← epoch-level summary
    <stem>_epochs_hmm.csv
  model/
    bsoid_model.pkl                  ← scaler, UMAP, HDBSCAN, MLP, PCA, config
    umap_embedding.npy               ← (n_total_bins, n_components)
    session_bin_ranges.json          ← {stem: [start, end, video_path]}
    hmm_model.pkl                    ← HMM parameters
  plots/
    umap_embedding.png
    umap_3d.html
    transition_matrix.png
    hmm_transition_matrix.png
    hmm_syntax_network.png
    state_space_projection.png
    cluster_stability.png            ← (if seed sweep enabled)
  videos/
    example_clips/<cluster_id>/      ← short clips per cluster for annotation
    labeled_videos/                  ← full videos with per-frame cluster label overlaid
    umap_evolution/                  ← (if UMAP evolution export was run)
```

### Key file: bout_lengths CSV

The bout CSV is the primary input to the Analyser. It contains:

| Column | Description |
|--------|-------------|
| `label` | Cluster ID |
| `start_frame` / `end_frame` | Frame indices of the bout |
| `start_s` / `end_s` | Time in seconds |
| `duration_s` | Bout length in seconds |
| `fps` | Frames per second of the source video |

**HMM file priority:** The Analyser always loads `*_hmm` files when present, falling back to raw MLP files. To force the Analyser to use raw MLP labels, rename or delete the `_hmm` files.

---

## 6. The Behavioural Analyser (cube_analyser)

Launch with: `"C:\...\python.exe" cube_analyser.py` or via the **Open Analyser** button in CUBE Step 5.

### Loading Data

On startup, select the **output folder** from a previous pipeline run. The Analyser detects all `*_bout_lengths_hmm.csv` files in the `bout_lengths/` subdirectory and loads one animal per file.

**Important:** the Analyser expects the `bout_lengths/` folder structure created by the pipeline. If you are loading manually, place files in a folder matching this structure.

---

### Combined Analysis Tab

**Overview of all loaded animals.**

- Shows a summary table with one row per animal: session duration, total clusters observed, total bouts.
- **Assign experimental groups:** use the group editor to assign each animal to a named experimental condition (e.g., "Control", "Drug", "WT", "KO"). These assignments drive all downstream comparisons.
- **Behavioural fingerprint heatmap:** shows cluster usage (proportion of time) across all animals as a heatmap, useful for spotting inter-animal variability.
- **Ethogram:** horizontal timeline for each animal, with colour-coded bars showing when each cluster occurred.

---

### Behaviour Statistics Tab

**Group-level comparisons across clusters.**

For each cluster, CUBE reports:

1. **Prevalence test (structural zeros):** Tests whether the cluster occurs in both groups (presence/absence comparison). A cluster absent in one group represents a structural zero — a fundamentally different kind of effect than a magnitude difference.

2. **Kruskal-Wallis test (present-only):** Among animals that expressed the cluster, tests whether the amount (frequency or duration) differs between groups. This separates "does the behaviour exist in this group?" from "how much do they do it?".

3. **Dunn's pairwise post-hoc (FDR-adjusted):** For significant KW results with ≥3 groups, pairwise post-hoc comparisons with FDR correction indicate which specific groups differ.

4. **Family-wide FDR table:** Exported to `advanced_analyses/` folder.

**Plot types available:**
- Frequency (bouts/session) per cluster per group
- Duration (total seconds) per cluster per group
- Bout length distribution (violin/box)
- Ethogram overlay for selected clusters

---

### Unbiased Analytics Tab

Eight plot modes for exploratory analysis:

| Mode | What it shows |
|---|---|
| **Top-N Bar** | Top N most-frequent clusters ranked by group mean |
| **Volcano** | Effect size vs. significance scatter for all clusters |
| **Heatmap** | Cluster × animal usage matrix (Ward-clustered) |
| **Elbow/Silhouette** | Within-cluster sum-of-squares and silhouette scores for Ward reclustering |
| **Recombination** | Ward-reclustered cluster dendrogram with blended distance (70% change-pattern + 30% transition cosine) |
| **Dist Matrix** | Pairwise behavioural-distance matrix between animals |
| **Transitions** | Empirical cluster-to-cluster transition probability matrix |
| **Cluster Stats** | Per-cluster summary: mean bout length, frequency, total time |

**Reclustering** (`build_recluster_result`) groups the discovered HDBSCAN clusters into a smaller set using Ward linkage on a blended distance matrix: 70% change-pattern correlation + 30% cosine similarity of outgoing transition profiles.

---

### Group Predictor Tab

**Multivariate supervised classification to test whether the full behavioural profile discriminates experimental groups.**

This tab runs three parallel models (Frequency, Total Duration, Transition Probability) with Leave-One-Out cross-validation and a permutation test.

For full documentation of this tab — algorithms, figures, controls, and interpretation — see [GROUP_PREDICTOR_REFERENCE.md](GROUP_PREDICTOR_REFERENCE.md).

**Quick start:**
1. Assign experimental groups in the Combined Analysis tab first.
2. Click **Run Models** with default settings.
3. Check the permutation p-value (green = significant). Click **View** on the best model.

---

## 7. HMM Smoothing

The MLP classifier produces per-frame labels independently for each frame. In practice, brief ambiguous moments cause rapid flickering between clusters — a single frame of "grooming" in the middle of a "locomotion" bout. The HMM wrapper corrects this by modelling the temporal structure of behaviour.

### How It Works

A **Categorical Hidden Markov Model** (from `hmmlearn`) is fitted to the MLP output:
- **Baum-Welch EM** estimates the transition matrix A (probability of switching from cluster i to cluster j) and emission probabilities B.
- **Viterbi decoding** then finds the single most probable cluster sequence given the observed MLP predictions — enforcing temporal coherence.

The near-diagonal initialisation of the transition matrix (strong self-persistence prior) is calibrated to produce biologically plausible bout lengths.

### Two Sets of Outputs

Every output file is produced in both raw (MLP) and HMM-smoothed versions:
- `*_frame_labels.csv` and `*_frame_labels_hmm.csv`
- `*_bout_lengths.csv` and `*_bout_lengths_hmm.csv`

The Analyser uses HMM files by default. To use raw MLP labels, delete or rename the `_hmm` files.

### Transition Matrix Distinction

- **`transition_matrix.png`** — empirical, counts every consecutive frame-pair in the raw MLP output. Diagonal masked.
- **`hmm_transition_matrix.png`** — model-learned; shows the A matrix from Baum-Welch EM. Two panels: the raw A matrix and an off-diagonal renormalised panel that exposes the switching grammar independent of self-persistence.

---

## 8. Advanced Settings

Open **Advanced CUBE Settings** from the top toolbar. Settings are organised into sections:

### Feature Extraction
- `feature_scales_ms` — default `[100, 200]`; add `50` for fine-grained locomotion distinction
- `angular_features` — enable/disable angular body-axis features (auto-skipped if no spine landmarks detected)
- `likelihood_threshold` — bodypart confidence threshold; frames below are interpolated (default 0.6)

### UMAP
- `n_components` — default 3; UMAP embedding dimensions
- `n_neighbors` — adapts automatically, but can be overridden
- `min_dist` — default 0.1; lower values produce tighter clusters
- `umap_random_state` — seed for reproducibility (default 42)

### HDBSCAN
- `min_cluster_size` — minimum cluster size is swept automatically; this is the anchor for the sweep
- `min_cluster_freq` — minimum cluster frequency to retain (default 0.002 = 0.2% of bins)
- `preferred_clusters_lo` / `preferred_clusters_hi` — soft target range for cluster count

### MLP Classifier
- `mlp_hidden` — hidden layer sizes, e.g. `100,50` (default, legacy) or `256,128,64` (recommended for larger datasets)
- `mlp_max_iter` — maximum training iterations (default 1000)
- `mlp_alpha` — L2 regularisation strength (default 0.001)
- `mlp_class_weight` — `balanced` (recommended) or `none`
- `mlp_early_stopping` — reserve 10% as validation set to stop when loss plateaus

### HMM
- `hmm_n_iter` — Baum-Welch EM iterations (default 100)
- `hmm_n_components` — number of HMM states (default = number of HDBSCAN clusters)

### Seed Sweep
- `seed_sweep_n` — number of UMAP seeds to sweep for cluster stability measurement (0 = disabled, 5–10 recommended for reproducibility checks). Produces `plots/cluster_stability.png`.

### Compatibility Mode
- `compat_mode` — set to `legacy_v2` to reproduce pre-v2.1 numeric defaults exactly. See [Section 10](#10-compatibility-mode).

---

## 9. Session Autosave and Resume

CUBE automatically saves pipeline state after each completed step to a `.pipeline_session.json` file in the output directory. If CUBE crashes or is closed:

1. Re-open CUBE.
2. Select the same output folder.
3. CUBE detects the saved session and offers to resume from the last completed step.

The autosave contains: output paths, parameter settings, step completion flags, and DLC model state. It does **not** contain the numerical data (that is re-loaded from the output files when resuming).

---

## 10. Compatibility Mode

CUBE v4 introduced corrected defaults (previously v2.1). To reproduce results from an earlier run exactly:

1. Open Advanced CUBE Settings.
2. Set **Compatibility mode** to `legacy_v2`.

This reverts:
- `umap_min_dist` from 0.1 back to 0.0
- `hdbscan_mcs_anchor` from `"embedding"` back to `"full"`
- `angular_fallback` from `False` back to `True`

Only keys that were not explicitly set by the user in the current session are reverted — explicit overrides always win. The analysis version is stamped in `feature_config.json` and `validation_report.json` so each output folder records which defaults were used.

---

## 11. Troubleshooting

### "No clusters found" after Step 3

1. Check the log — HDBSCAN may have returned all noise points. Try lowering `min_cluster_size` anchor, or increasing `n_neighbors`.
2. Ensure DLC tracking quality is good: open a few H5 files and check that likelihood values are mostly > 0.6.
3. Try lowering the `likelihood_threshold` if too many frames are being interpolated (check the "% interpolated" log line).

### Very low MLP cross-validation accuracy

1. Check that `min_cluster_freq` is not too low — tiny clusters (< 1% of frames) are hard to classify.
2. Try `mlp_hidden = "256,128,64"` (larger network) in Advanced Settings.
3. Enable `mlp_class_weight = "balanced"` to handle imbalanced cluster sizes.
4. Check for degenerate UMAP — if `umap_3d.html` shows all points in one clump, try lowering `min_dist` or increasing `n_neighbors`.

### HMM smoothing produces implausibly long bouts

The HMM has a strong self-persistence prior by design. If bouts are too long:
1. Look at `hmm_transition_matrix.png` — the diagonal should be high (0.95–0.99) but not 1.0.
2. Try reducing `hmm_n_iter` (fewer EM iterations leave the transition matrix closer to prior).
3. Compare `*_bout_lengths.csv` vs `*_bout_lengths_hmm.csv` — if the HMM version looks wrong, the Analyser can be forced to use raw MLP labels by deleting `_hmm` files.

### DLC inference fails in Step 1

1. Verify CUDA drivers if using GPU. Run DLC outside CUBE first to confirm it works.
2. Try **Smart Adapt mode** disabled — it sometimes fails if the adaptation video does not contain enough diverse frames.
3. Check that video files are readable by OpenCV: open one in any video player, and try `cv2.VideoCapture(path).isOpened()` in Python.

### The Analyser shows no animals / empty tables

1. Confirm the selected folder contains a `bout_lengths/` subfolder.
2. Confirm `*_bout_lengths_hmm.csv` (or `*_bout_lengths.csv`) files are present in that subfolder.
3. Check file naming: files must end in `_bout_lengths.csv` or `_bout_lengths_hmm.csv`.

### Group Predictor shows near-chance accuracy for all models

See [GROUP_PREDICTOR_REFERENCE.md — Section 16](GROUP_PREDICTOR_REFERENCE.md#16-tips-and-user-guidance) for detailed guidance. Common causes:
- Groups are genuinely behaviourally similar in this recording context.
- n is too small (< 6 per group is very challenging).
- Group assignments are incorrect.
- The wrong feature source is being used — try Custom mode with a targeted cluster subset.

### tkinter "invalid command name" warnings on close

These are harmless. They occur when the application closes and some scheduled GUI update callbacks fire after the widgets they reference have been destroyed. They do not affect analysis results or saved files.

---

*For the full Group Predictor documentation, see [GROUP_PREDICTOR_REFERENCE.md](GROUP_PREDICTOR_REFERENCE.md).*  
*For the v5 3D dual-camera extension plan, see [3D_Integration_Plan_v5.md](3D_Integration_Plan_v5.md).*
