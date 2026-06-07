<p align="center">
  <img src="Cube_logo.PNG" alt="CUBE Logo" width="220"/>
</p>

<h1 align="center">CUBE — Comprehensive Unsupervised Behavioral Explorer</h1>

<p align="center">
  An automated, end-to-end pipeline for discovering and quantifying animal behavior from video — no manual labeling required.
</p>

---

## Overview

CUBE integrates **DeepLabCut pose estimation** with unsupervised machine learning (UMAP + HDBSCAN + MLP classifier) to automatically segment, cluster, and analyze animal behavioral patterns from raw video recordings. The full pipeline runs from raw video to statistical reports without requiring any predefined behavioral categories.

**Key features:**
- Batch DLC inference using the SuperAnimal quadruped model
- Smart Adapt mode: adapts the model once on a representative video, then reuses weights for all videos
- V2 multi-scale feature extraction with body-size normalization and angular body-axis features
- Automatic HDBSCAN cluster sweep with DBCV-guided selection
- Post-hoc Multinomial HMM wrapper that eliminates MLP state-flickering and enforces temporal consistency (see [HMM Smoothing](#hmm-smoothing) below)
- Built-in validation layer (silhouette, UMAP trustworthiness, CV accuracy, DLC quality gates)
- Interactive video annotation and behavioral analysis with group statistics and ethograms
- Session autosave — resume after a crash from the last completed step

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `cube.py` | **DLC Inference** — Run DeepLabCut SuperAnimal on raw videos |
| 2 | `cube_core.py` | **Pre-processing** — Filter bodyparts, export H5/CSV |
| 3 | `cube_core.py` | **Clustering** — UMAP → HDBSCAN → MLP classifier |
| 4 | `cube_video_explorer.py` | **Video Annotation** — Label clusters via example clips |
| 5 | `cube_analyser.py` | **Behaviour Analysis** — Ethograms, statistics, group comparisons |

---

## HMM Smoothing

### Why it exists

B-SOiD's MLP classifier scores each 100 ms time bin **independently**, with no memory of adjacent frames. Minor tracker jitter or overlapping UMAP cluster boundaries can produce single-frame flips between unrelated behaviors (e.g., Walk → Rear → Walk over 3 consecutive frames). These "state flickers" inflate transition counts, shorten apparent bout durations, and obscure real behavioral sequences.

CUBE wraps the raw MLP output with a **post-hoc Multinomial (Categorical) Hidden Markov Model** trained on the same label sequences via Baum-Welch EM. The model treats the raw cluster assignments as noisy observations of a hidden state trajectory and recovers the most probable sequence via Viterbi decoding. No upstream step (DLC, features, UMAP, HDBSCAN, MLP) is modified.

This approach mirrors the temporal smoothing used in published behavioral HMM systems (Wiltschko et al. 2015 *Neuron*; Luxem et al. 2022 *Commun. Biol.*) while keeping the B-SOiD cluster labels and workflow intact.

### What it produces

After a successful run with HMM enabled, the following additional files appear alongside the standard outputs:

| File | Location | Description |
|------|----------|-------------|
| `<session>_bout_lengths_hmm.csv` | `bout_lengths/` | Run-length encoded bouts in B-SOiD format — **this is what the Analyser uses** |
| `<session>_frame_labels_hmm.csv` | `bout_lengths/` | Per-frame HMM state IDs with timestamps |
| `<session>_epochs_hmm.csv` | `bout_lengths/` | Epoch table (filtered by min/max bout duration) |
| `<session>_epoch_stats_hmm.csv` | `bout_lengths/` | Per-cluster summary statistics on HMM epochs |
| `hmm_model.pkl` | `model/` | Serialised fitted `CategoricalHMM` for reuse or inspection |
| `hmm_duration_comparison.png` | `plots/` | Log-scale bout duration histograms — raw vs. HMM |
| `hmm_transition_matrix.png` | `plots/` | Heatmap of the learned transition matrix A[i→j] |
| `hmm_ethogram_<session>.png` | `plots/` | Dual-row raster: raw B-SOiD on top, Viterbi below |
| `hmm_syntax_network.png` | `plots/` | Directed behavioral grammar graph (node size ∝ stationary occupancy) |

> **Analyser integration**: when `_hmm` files are present, the Analyser automatically prefers them over raw `_bout_lengths.csv` files. To revert to raw labels, delete or rename the `_hmm` files.

### HMM parameters (Advanced Settings → HMM Smoothing)

| Parameter | Default | Recommended range | Notes |
|-----------|---------|-------------------|-------|
| **Enable HMM smoothing** | On | — | Disable only if hmmlearn is not installed or you want pure MLP output |
| **HMM states** | 0 (auto) | 0 or leave blank | `0` sets n_states = n_clusters (smoothing-only mode, recommended). Set a smaller value, e.g. `5`, to discover behavioral macro-states — note that macro-state IDs will not match the original cluster mapping |
| **Baum-Welch iterations** | 100 | 50–200 | 100 iterations is sufficient for convergence on typical behavioral data (10–30 clusters, 10⁵–10⁶ frames). Increase to 200 only if the log-likelihood has not plateaued — check the pipeline log for `[HMM] trained in X s` |
| **Min edge prob (syntax graph)** | 0.05 | 0.02–0.20 | Edges below this transition probability are hidden in the syntax network graph. Lower = more connections shown, higher = only dominant transitions visible |

### Interpreting the diagnostic plots

**`hmm_duration_comparison.png`** — The 1-frame spike (at `1/fps` on the x-axis) in the left panel (raw) should be absent or much smaller in the right panel (HMM). If the right panel still shows a spike at 1 frame, the HMM is not smoothing effectively — try increasing Baum-Welch iterations or checking that the cluster count is not too high relative to the data length.

**`hmm_transition_matrix.png`** — Strong diagonal (dark blue cells top-left to bottom-right) means behavioral states are self-persistent: animals tend to stay in the same state. Off-diagonal entries reveal which state transitions are most common. Very uniform rows (all values ≈ 1/n) indicate the HMM has insufficient data to learn a transition structure.

**`hmm_ethogram_<session>.png`** — The two rows should look similar but with the bottom (HMM) row visibly "cleaner" — shorter isolated single-frame blocks should consolidate into longer uniform runs. If the two rows look identical the HMM has not changed anything (often because n_iter is too low or data is very short).

**`hmm_syntax_network.png`** — Node size encodes how much time animals spend in that state (stationary probability). Large nodes are dominant behavioral states. Thick arrows between nodes represent high-probability transitions. Isolated nodes with no outgoing arrows above `min_prob` are transient states visited briefly.

### Smoothing-only mode vs. macro-state mode

| Mode | HMM states setting | Effect | Analyser compatible? |
|------|-------------------|--------|----------------------|
| **Smoothing-only** (default) | 0 / blank | n_states = n_clusters; each HMM state corresponds to one B-SOiD cluster. State IDs are aligned to cluster IDs via the Hungarian algorithm. | Yes — cluster mapping from Video Explorer is preserved |
| **Macro-state discovery** | e.g. `5` | n_states < n_clusters; HMM learns higher-order groupings of clusters. State IDs are arbitrary (0..n_states−1). | Partial — you must re-annotate macro-states in Video Explorer before running the Analyser |

### Performance expectations

- **Training time**: < 5 s for a 1-hour session at 30 fps (~108 000 frames, 15 clusters). Training is done once on all sessions combined.
- **Viterbi decoding**: < 1 s per session.
- **Memory**: negligible — the CategoricalHMM parameter count is O(n_states²) regardless of data length.

---

## Setting Up the Anaconda Environment

### 1. Install Anaconda or Miniconda

Download from [https://www.anaconda.com/download](https://www.anaconda.com/download) and follow the installer instructions.

### 2. Create a new environment

Open **Anaconda Prompt** and run:

```bash
conda create -n CUBE python=3.10 -y
conda activate CUBE
```

### 3. Install PyTorch (with GPU support)

Check your CUDA version first (`nvidia-smi` in the terminal), then install the matching PyTorch build. For CUDA 11.8:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

For CPU-only (no GPU):

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### 4. Install DeepLabCut

```bash
pip install "deeplabcut[pytorch]"
```

### 5. Install CUBE dependencies

```bash
pip install pillow opencv-python-headless scipy scikit-learn umap-learn customtkinter ruamel.yaml h5py hmmlearn>=0.3.2 networkx
```

```bash
conda install -c conda-forge hdbscan -y
```

### 6. Verify the installation

```bash
python -c "import deeplabcut; import umap; import hdbscan; import customtkinter; print('All packages OK')"
```

---

## Required Packages Summary

| Package | Source | Purpose |
|---------|--------|---------|
| `deeplabcut[pytorch]` | pip | Pose estimation (Step 1) |
| `torch`, `torchvision`, `torchaudio` | conda (pytorch channel) | GPU inference backend |
| `numpy` | pip/conda | Numerical computing |
| `pandas` | pip/conda | Data manipulation |
| `matplotlib` | pip/conda | Plotting and figure export |
| `scipy` | pip | Signal filtering (H5 post-processing) |
| `scikit-learn` | pip | MLP classifier, validation metrics |
| `umap-learn` | pip | Dimensionality reduction (Step 3) |
| `hdbscan` | conda-forge | Density-based clustering (Step 3) |
| `opencv-python-headless` | pip | Video reading and resizing |
| `pillow` | pip | Image handling in GUI |
| `customtkinter` | pip | Analyser GUI (Step 5) |
| `ruamel.yaml` | pip | DLC config injection (Smart Adapt) |
| `h5py` | pip | HDF5 pose file I/O |
| `hmmlearn` | pip | Post-hoc HMM smoothing (Step 3) |
| `networkx` | pip | Behavioral syntax network graph (Step 3) |

---

## Running CUBE

1. Activate the environment:
   ```bash
   conda activate CUBE
   ```

2. Navigate to the CUBE folder and launch:
   ```bash
   python cube.py
   ```

3. In the GUI:
   - Add your video source folders
   - Set your output root directory
   - Configure DLC and clustering settings as needed
   - Run steps 1–5 in order (or use Auto-run to chain them)

Sessions are automatically saved after each step. To resume after a crash, click **Load** and select the `autosave.pipeline_session.json` file in your output folder.

---

## File Structure

```
CUBE 3/
├── cube.py                  # Main launcher and GUI
├── cube_core.py             # Core analysis engine (V2 features, UMAP, HDBSCAN, MLP)
├── cube_analyser.py         # Behaviour analysis and statistics (Step 5)
├── cube_video_explorer.py   # Video annotation tool (Step 4)
├── theme.txt                # UI theme setting ("dark" or "light")
└── CUBE_logs/               # Pipeline log files
```

---

## Outputs

After a full run, results are saved to your chosen output root:

**Core outputs**
- `BSOID_Project_Ready/` — filtered H5 and CSV pose files per session
- `bout_lengths/<session>_bout_lengths.csv` — raw MLP per-frame labels in B-SOiD format
- `bout_lengths/<session>_frame_labels.csv` — per-frame label array with timestamps
- `bout_lengths/<session>_epochs.csv` — epoch table (start/end time per bout)
- `umap_embedding.png` — UMAP scatter plot coloured by cluster
- `ethogram_<session>.png` — behavioural raster plot per session
- `validation_dashboard.png` — pass/warn/block quality gates at a glance
- `validation_report.json` — machine-readable validation summary
- `model/` — saved UMAP, scaler, MLP, and HDBSCAN models
- `example_clips/cluster_NN/` — representative video clips per cluster

**HMM outputs** (when HMM smoothing is enabled — see [HMM Smoothing](#hmm-smoothing))
- `bout_lengths/<session>_bout_lengths_hmm.csv` — HMM-smoothed labels (**used by Analyser**)
- `bout_lengths/<session>_frame_labels_hmm.csv` — per-frame HMM state IDs with timestamps
- `bout_lengths/<session>_epochs_hmm.csv` — epoch table from HMM labels
- `bout_lengths/<session>_epoch_stats_hmm.csv` — per-cluster duration statistics
- `model/hmm_model.pkl` — serialised fitted `CategoricalHMM`
- `plots/hmm_duration_comparison.png` — bout duration histograms before vs. after
- `plots/hmm_transition_matrix.png` — learned transition matrix heatmap
- `plots/hmm_ethogram_<session>.png` — dual-row raster (raw top, Viterbi bottom)
- `plots/hmm_syntax_network.png` — directed behavioral grammar graph

---

## Troubleshooting

### General

| Issue | Fix |
|-------|-----|
| `DeepLabCut not found` | Activate the CUBE conda environment before launching: `conda activate CUBE` |
| `cube_core.py not found` | All four `.py` files must be in the same folder |
| `umap-learn / hdbscan missing` | `pip install umap-learn` and `conda install -c conda-forge hdbscan` |
| `customtkinter missing` | `pip install customtkinter` |
| H5 MultiIndex error | Steps 2 and 3 handle this automatically — no action needed |
| CUDA out of memory | Reduce batch size in Advanced DLC Parameters, or enable Smart Adapt mode |
| Windows MAX_PATH errors | Enable long path support: Group Policy → `Enable Win32 long paths` |
| Analyser shows wrong cluster count | Check that you have loaded the correct output folder. If HMM files exist but n_states ≠ n_clusters, re-run with HMM states = 0 (auto) |

### HMM Smoothing

| Issue | Fix |
|-------|-----|
| `hmmlearn missing` | `pip install "hmmlearn>=0.3.2"` — HMM smoothing is skipped silently until installed |
| `networkx missing` | `pip install networkx` — only the syntax network plot is affected; all other HMM outputs still appear |
| HMM training very slow (> 30 s) | Reduce Baum-Welch iterations to 50. Check that n_states is not set to a very large number — leave at 0 (auto) for normal use |
| `_hmm.csv` files not appearing | Check the pipeline log for `[WARN] HMM smoothing failed` and read the traceback. Most common cause: `hmmlearn` not installed or a session with fewer frames than n_states |
| Analyser loads raw files instead of HMM | Ensure the output folder contains `_bout_lengths_hmm.csv` files (they are created in `bout_lengths/`). Delete any stray `_bout_lengths.csv` that are not alongside their `_hmm` counterparts, or re-run Step 3 |
| HMM has no effect on flickering | Increase Baum-Welch iterations to 200 or reduce the cluster count (high cluster counts with little data give the HMM insufficient signal). Also confirm that session length is > 1000 frames |
| Syntax network graph is empty | All transition probabilities are below `min_prob`. Lower the Min edge prob setting to 0.02 or check that n_states > 1 |
| State IDs in `_hmm` files differ from cluster IDs | This can happen in macro-state mode (n_states < n_clusters). In smoothing-only mode (n_states = 0/auto) state alignment is automatic. If you observe misalignment in smoothing mode, re-run Step 3 — alignment uses the Hungarian algorithm and is deterministic |
| `scipy` not found (alignment skipped) | `pip install scipy` — scipy is also required by `hdbscan` so it should already be present in the CUBE environment |

---

## Disclaimer

> Parts of this codebase were developed with the assistance of AI tools, including large language models used for code generation, debugging, and documentation. All outputs have been reviewed, tested, and validated by the authors. Users should independently verify results for their specific use cases.

---

## License

This project is for research use. Please cite DeepLabCut and B-SOiD if you use this pipeline in published work.
