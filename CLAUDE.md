# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment — MANDATORY

**All code execution, tests, and corrections MUST use the `CUBE` conda environment.**  
Never run any CUBE Python file with the system Python or any other interpreter.

**Anaconda is installed at `C:\Users\param\anaconda3`.**

**The CUBE conda environment Python executable is at `C:\Users\param\anaconda3\envs\CUBE\python.exe`.**

**ALWAYS invoke Python using this full path — never use `python`, `conda activate`, or any other form. Claude terminals are not Anaconda terminals and `conda` / `python` will not resolve correctly.**

```bat
"C:\Users\param\anaconda3\envs\CUBE\python.exe" <script>
```

Verify the environment is correct before running anything:
```bat
"C:\Users\param\anaconda3\envs\CUBE\python.exe" -c "import umap; print('OK')"
```

If this fails, report the error — do not attempt workarounds or fall back to system Python.

---

## Running the Application

```bat
rem Always use the full Python path — conda is not on the PATH in Claude terminals
"C:\Users\param\anaconda3\envs\CUBE\python.exe" cube.py

rem Launch the standalone behaviour analyser (Step 5 only)
"C:\Users\param\anaconda3\envs\CUBE\python.exe" cube_analyser.py

rem Launch the standalone cluster annotator (Step 4 only)
"C:\Users\param\anaconda3\envs\CUBE\python.exe" cube_video_explorer.py
```

No build step. Dependencies are already installed in the `CUBE` environment:
```bash
pip install pillow opencv-python-headless scipy scikit-learn umap-learn customtkinter plotly hmmlearn
conda install -c conda-forge hdbscan
```

`environment.yml` (repo root) captures this same dependency list (plus `pytest`/`pytest-cov`) for reproducibility and CI — `conda env create -f environment.yml` recreates the `CUBE` environment from scratch.

### Automated tests

An automated `pytest` suite exists under `tests/` (`tests/unit/`, `tests/integration/`, `tests/conftest.py`), purely additive to `cube_core.py`/`cube_analyser.py` — synthetic fixtures only, never real user data or `CUBE_logs/`. Always run it through the mandatory `CUBE` env Python path:

```bat
rem Full suite (includes the slow end-to-end BSoidEngine.run() regression)
"C:\Users\param\anaconda3\envs\CUBE\python.exe" -m pytest tests\ -v

rem Fast subset only (excludes the slow-marked regression -- use for iteration)
"C:\Users\param\anaconda3\envs\CUBE\python.exe" -m pytest tests\ -v -m "not slow"
```

See `Automated_Test_Suite_Plan.md` for the full design rationale (testing philosophy, fixture strategy, phase-by-phase scope) and `Test_Suite_Implementation_Report.md` for what's implemented, pass/fail counts, coverage numbers, and known gaps/deviations. Verification beyond the automated suite is still done by running the pipeline against real DLC output files where appropriate (e.g. exploratory GUI checks).

---

## Architecture Overview

CUBE is a four-file GUI application built on tkinter/customtkinter that wraps the B-SOiD unsupervised pose-based behaviour classification pipeline.

### File responsibilities

| File | Role |
|---|---|
| `cube_core.py` | Pure analysis engine — **no GUI code**. All numerical logic lives here. |
| `cube.py` | Main 5-step pipeline GUI. Imports from `cube_core.py`; lazy-loads `cube_analyser.py` and `cube_video_explorer.py` at runtime via `importlib`. |
| `cube_analyser.py` | Step 5 standalone GUI (customtkinter). Statistical analysis, reclustering, group comparisons, transition dynamics. Has its own data-loading layer and matplotlib figure builders. |
| `cube_video_explorer.py` | Step 4 standalone GUI (tkinter). Cluster annotation via embedded video playback. No dependency on `cube_core.py`. |

`cube_analyser.py` and `cube_video_explorer.py` are loaded by `cube.py` as sibling modules via `importlib.util.spec_from_file_location`. They can also run standalone.

### The core pipeline (`BSoidEngine` in `cube_core.py`)

The entire analysis is one class, `BSoidEngine` (line ~3955). Its `run()` method executes these stages in sequence:

1. **DLC file discovery** — `find_dlc_files` / `pair_files` match H5/CSV files to videos by stem, prefix, or `YYYYMMDD_HHMMSS` timestamp.
2. **Load & smooth** — `load_dlc_file` normalises any DLC MultiIndex to 3-level `(scorer, bodyparts, coords)`, interpolates low-likelihood frames, then `smooth_boxcar` applies a centred moving average.
3. **Feature extraction** — `extract_features_v2` produces fps-adaptive multi-scale features: 100 ms + 200 ms bins always present; 50 ms fine-scale bin added at ≥ 60 fps (at 30 fps 50 ms = 1.5 frames — DLC jitter, not real movement). Within-bin positional variance is always computed and captures sub-100 ms tremor, flinching, and oscillatory motion. Features: pairwise distances + velocity + smoothed acceleration + optional body-normalised angular features. Output shape: `(n_features, n_bins)` where one bin = 100 ms. For 3D pose data, `extract_features_3d` is used instead (100 ms + 200 ms bins, temporal lag drift at 5- and 10-bin offsets, no within-bin variance).
4. **UMAP** — `run_umap` with adaptive `n_neighbors`. Auto-triggers PCA pre-reduction when `n_features >= n_samples / 5`.
5. **HDBSCAN sweep** — `run_hdbscan` sweeps `min_cluster_size` across 40 steps, tries both `eom` and `leaf` methods, and selects by DBCV (`relative_validity_`). Rare clusters (<0.2% of bins) are pruned before MLP training.
6. **MLP classifier** — `train_mlp` trains a scikit-learn `MLPClassifier` on HDBSCAN-labelled bins. Cross-validated accuracy is recorded.
7. **Inference** — `predict_labels` applies the fitted scaler→(optional PCA)→MLP to every session, expanding 100 ms bins back to per-frame labels.
8. **HMM smoothing** — `train_hmm` + `decode_hmm` wrap MLP output with `hmmlearn.CategoricalHMM` (Baum-Welch + Viterbi). Produces `*_hmm` variants of all output CSVs.
9. **Export** — bout CSVs, frame-label CSVs, epoch CSVs, example clips, labeled videos, plots. Model saved as `model/bsoid_model.pkl`.

### Output directory layout

```
<output_dir>/
  bout_lengths/
    <stem>_bout_lengths.csv          # raw MLP bouts
    <stem>_bout_lengths_hmm.csv      # HMM-smoothed bouts  (preferred by analyser)
    <stem>_bout_lengths_hmm_enriched.csv  # (if kinematic_directedness_enabled and/or
                                           #  env_features_enabled, both off by default;
                                           #  a SHARED sidecar -- canonical 3 cols +
                                           #  whichever flag(s) are on contribute columns)
                                           # kinematics: net_displacement_px, path_length_px,
                                           #   straightness_ratio, mean_speed_px_s, heading_consistency
                                           # env context: pct_time_in_region_<name>,
                                           #   pct_time_near_object_<name>,
                                           #   mean_heading_angle_to_object_<name>
    <stem>_approach_events.csv       # (if BOTH kinematic_directedness_enabled AND
                                      #  env_features_enabled with traced shapes; only
                                      #  written when >=1 event detected)
    <stem>_frame_labels.csv          # per-frame: frame,time_s,label
    <stem>_frame_labels_hmm.csv
    <stem>_epochs.csv / _epochs_hmm.csv
  model/
    bsoid_model.pkl                  # scaler, umap, hdb, mlp, pca, config, etc.
    umap_embedding.npy               # (n_total_bins, n_components)
    session_bin_ranges.json          # {stem: [start, end, video_path]}
    session_env_context.json         # (if env_features_enabled; not written at all otherwise)
                                      # {stem: {"per_bin": {...}, "summary": {...},
                                      #         "derived": {...}, "paradigm": str, "bin_ms": 100.0}}
    hmm_model.pkl
  plots/
    umap_embedding.png
    umap_3d.html                     # interactive Plotly
    transition_matrix.png            # empirical (from raw frame labels)
    hmm_transition_matrix.png        # HMM learned A matrix (two panels)
    hmm_syntax_network.png
    state_space_projection.png
    ...
  videos/
    example_clips/<cluster_id>/
    labeled_videos/
    umap_evolution/
```

### HMM file priority

`cube_analyser.py`'s `_prefer_hmm()` (line ~181) always loads `*_hmm` files when present, falling back to raw. To force raw labels, delete or rename the `_hmm` files.

### Transition matrix distinction

- **`transition_matrix.png`** (`plot_transition_matrix`, line 2774): empirical — counts every consecutive frame-pair `(label_t → label_t+1)` in the raw MLP output, then row-normalises. Diagonal is masked (dominated by self-persistence). Single heatmap.
- **`hmm_transition_matrix.png`** (`plot_hmm_transition_matrix`, line 1161): model-learned — shows `hmm_model.transmat_` (the `A` matrix from Baum-Welch EM). Includes a second panel with the diagonal zeroed and rows renormalised to expose the switching grammar. The diagonal is partly prior-driven by the near-diagonal emission initialisation.

### `cube_analyser.py` data flow

Loads `*_bout_lengths[_hmm].csv` from a user-selected folder → builds per-animal `{"df": DataFrame, "name": str, "fps": int}` dicts → feeds into stat/recluster functions. Reclustering (`build_recluster_result`) uses Ward linkage on a blended distance matrix: 70% change-pattern correlation + 30% cosine similarity of outgoing transition profiles. The `UnbiasedAnalyticsPanel` generates eight plot modes (Top-N Bar, Volcano, Heatmap, Elbow/Silhouette, Recombination, Dist Matrix, Transitions, Cluster Stats).

### Plot theme system

`cube_core.py` has four mutable module-level globals (`_BG`, `_PANEL`, `_TEXT_COL`, `_TICK_COL`) set by `_apply_plot_theme("dark"|"light")` at the start of each run. All plot functions read these globals — **never hardcode colour constants in new plot functions**; always use `_BG`, `_PANEL`, `_TEXT_COL`, `_TICK_COL`.

### Compatibility / reproducibility

`BSoidEngine.DEFAULTS["compat_mode"] = "current"` (v2.1). Set `"legacy_v2"` to restore pre-2.1 numeric defaults (different `umap_min_dist`, `hdbscan_mcs_anchor`, `angular_fallback`). Only keys the caller did not explicitly pass are reverted — explicit overrides always win.

### Kinematic directedness (v6 K1/K2, opt-in)

`BSoidEngine.DEFAULTS["kinematic_directedness_enabled"] = False`. When `True`, writes the `*_bout_lengths_hmm_enriched.csv` sidecar above per session (via `compute_bout_directedness()` in `cube_core.py`); the canonical `*_bout_lengths_hmm.csv` is never modified in either state. `enrich_bouts_from_bin_source()` (also in `cube_core.py`, a generalization of `attach_centroid_distance`) is the shared per-bin-to-bout join utility this and the environmental-context work build on. `cube_analyser.py` also unconditionally (no flag) joins `cluster_kinematics.csv`'s per-cluster kinematic signatures into `compute_per_cluster_metrics()`'s output, surfaced in the Top-N Bar/Volcano/Heatmap "Metric:" selector. See `Kinematic_Transition_v6_Implementation_Plan.md` and `Kinematics_v6_Implementation_Report.md` for full detail.

### Environmental context & object interaction (v6 part 2, opt-in)

`BSoidEngine.DEFAULTS["env_features_enabled"] = False` (master switch). `env_arena_cfg = None` holds one project-wide traced-arena config (`{"schema_version": 3, "paradigm": ..., "reference_shapes": {...}, "per_video": {...}}` — see `Environmental_Context_v6_Implementation_Plan.md` Step 1 for the full schema and `ENV_PARADIGMS`/`ENV_PARADIGM_ROLE_VOCAB`/`ENV_PARADIGM_MIN_ROLES` in `cube_core.py` for the paradigm/role vocabulary). `env_interaction_threshold = None` auto-derives one body length per session when unset. With the flag off (default), none of this runs — no GUI invitation, no `session_env_context.json`, no bout-CSV columns; output is byte-identical to a pre-this-feature run. Traced in `cube.py`'s `EnvContextWindow` (opened from `AdvancedCUBEWindow`'s "Environmental Context..." button); `resolve_env_shapes()` resolves one session's effective shape list (reference + per-video transform/override + role_overrides); `compute_session_env_context()` computes per-bin region/object time series, session summaries, and paradigm-gated derived metrics (Y-Maze alternation %, EPM anxiety metrics, Novel Object discrimination index, Three-Chamber sociability/social-novelty indices, Place Preference index). `detect_approach_avoid_events()` (Step 7) is a rule-based sequence-plus-tag detector requiring BOTH `env_features_enabled` and `kinematic_directedness_enabled`. Coordinate-space correctness between traced shapes and pose data is a **structural** guarantee (`EnvContextWindow` reads video frames through the same `dlc_crop_x/y/w/h` rectangle DLC tracked on), not a runtime assertion — see `Environmental_Context_v6_Implementation_Report.md` for why. See `Environmental_Context_v6_Implementation_Plan.md` and `Environmental_Context_v6_Implementation_Report.md` for full detail.

### Planned 3D dual-camera extension

`3D_Integration_Plan_v5.md` documents a planned `triangulate_camera_pair()` function in `cube_core.py` and a `DualCamera3DWindow` + `_run_dlc_3d_step()` in `cube.py`. The chosen H5 format doubles bodyparts (`bp` + `bp_z` pseudo-keypoints) so Steps 2–5 require zero changes. This is not yet implemented.

---

## Git and GitHub — Version Control Workflow

### Git executable

**Do NOT use conda git for pushing.** Use the Visual Studio bundled git, which has proper Windows Credential Manager integration:

```bat
set GIT="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer\Git\cmd\git.exe"
```

The conda git at `C:\Users\param\anaconda3\Library\bin\git.exe` works for local operations (status, diff, log) but fails on push due to credential helper conflicts.

### Commit and push workflow

Run these after completing a session with meaningful changes:

```bat
set GIT="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer\Git\cmd\git.exe"
cd d:\CUBE

%GIT% add -A
%GIT% commit -m "short description of what changed"
%GIT% push origin main
```

Or in PowerShell:

```powershell
$git = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer\Git\cmd\git.exe"
Set-Location d:\CUBE
& $git add -A
& $git commit -m "short description"
& $git push origin main
```

### Authentication

Credentials are stored in **Windows Credential Manager** (set up June 2026). No token is needed in the URL. If a push is rejected with 401/403:
1. Open Windows Credential Manager → Windows Credentials
2. Find the entry for `git:https://github.com`
3. Edit it and paste a new PAT as the password

GitHub user: `param-valiathan`  
Commit email: `288124827+param-valiathan@users.noreply.github.com` (noreply — required by GitHub email privacy settings)  
Remote: `https://github.com/param-valiathan/CUBE.git`

### When to commit

Commit after:
- Any feature addition or bug fix to the four source files
- Documentation updates (README.md, CUBE_GUIDE.md, GROUP_PREDICTOR_REFERENCE.md)
- Running `md_to_docx.py` is **not** needed before committing — `.docx` files are gitignored

Use clear commit messages: `"feat: add X"`, `"fix: Y was broken"`, `"docs: update README for Z"`.

---

## Documentation Maintenance — MANDATORY

CUBE's documentation lives in three markdown files that must stay current. After **any** significant change to the pipeline or analyser, follow these rules before closing the session.

### Files to keep updated

| File | What it covers | Update when... |
|---|---|---|
| `README.md` | v5 feature summary, architecture, quick start, output layout | New feature, changed behaviour, new output file, new Advanced Setting, bug fix that changes user-visible output |
| `CUBE_GUIDE.md` | Full user guide — all steps, all settings, troubleshooting | Any step workflow changes, new/removed settings, new error scenarios |
| `GROUP_PREDICTOR_REFERENCE.md` | Complete Group Predictor reference | Any change to `cube_analyser.py` Group Predictor tab logic, new controls, new figures, changed algorithm |

### What counts as a significant change

Include in `README.md` under the relevant feature section:
- New features added to any of the four source files
- Changes to default parameter values
- New or renamed output files or plots
- Changes to the HMM pipeline, MLP classifier, HDBSCAN sweep, or UMAP stage
- New tabs or controls in the Analyser or Group Predictor
- Bug fixes that alter what users see or get as output

Do **not** add entries for:
- Internal refactors with no user-visible effect
- Code style / comment-only changes
- Performance improvements with identical output

### Regenerate Word documents after every edit

After updating any markdown documentation file, run:

```bat
"C:\Users\param\anaconda3\envs\CUBE\python.exe" d:\CUBE\md_to_docx.py
```

This regenerates `README.docx`, `CUBE_GUIDE.docx`, and `GROUP_PREDICTOR_REFERENCE.docx` from the markdown sources. The script auto-installs `python-docx` on first run. Always run it — do not skip even for minor wording changes.
