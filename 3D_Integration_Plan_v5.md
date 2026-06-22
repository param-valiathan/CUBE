# Plan: 3D Dual-Camera DeepLabCut Integration into CUBE v5

## Context

CUBE currently runs 2D DLC SuperAnimal pose estimation on single-camera videos. The user has a two-perpendicular-camera rig where each recording session produces a parent video folder (e.g. `10C_NPY_1_0000_Videos`) containing two subfolders — one per camera — named with the pattern `pseudo_resized_Cam<ID>_Series<N>`. Both cameras are hardware-synchronized (identical frame counts). The user already holds an aniposelib calibration TOML for the rig.

Adding 3D mode: run 2D DLC on each camera independently → triangulate to 3D using aniposelib → output a merged 3D H5 → flow into existing Steps 2–5 unchanged.

This is a user-selectable toggle; the existing 2D workflow must be fully preserved.

---

## What the User Must Provide (One-Time)

| Input | Format | How obtained |
|---|---|---|
| Camera calibration file | aniposelib `calibration.toml` | Already have it; DLC's `calibrate_cameras()` or aniposelib board calibration |
| Camera subfolder prefix | string (default `pseudo_resized_`) | Auto-detected; user can override |

No synchronization work needed — cameras are hardware-synced.

---

## Architecture Decision: 3D H5 Format

The 3D H5 must pass through existing Steps 2–5 with **zero changes** to `run_bsoid_prep`, `load_dlc_file`, `_normalise_dlc_df`, `extract_features_v2`, and the analyser.

**Chosen approach — bodypart doubling with `_z` pseudo-keypoints:**

For each original bodypart `bp`, write two entries to the output H5:
- `bp` → x = 3D-X, y = 3D-Y, likelihood = min(cam1_ll, cam2_ll)
- `bp_z` → x = 3D-Z, y = 0.0, likelihood = min(cam1_ll, cam2_ll)

This produces a standard 3-level `(scorer, bodyparts, coords)` MultiIndex H5. All downstream code sees twice as many bodyparts, computes XY pairwise distances (now containing Z info via the `bp_z` pseudo-keypoints), and runs without modification. Angular features still use `nose`, `tailbase`, etc. (the non-`_z` names), so spine curvature detection is unaffected.

---

## Files to Modify

| File | Change |
|---|---|
| `cube_core.py` | Add `triangulate_camera_pair()` function |
| `cube.py` | Add 5 SessionState fields, `DualCamera3DWindow` class, button wire-up, `_run_dlc_3d_step()` function, mode branch in `_run_dlc_step` |

`cube_video_explorer.py` and `cube_analyser.py` — **no changes needed.**

---

## Implementation Steps

### 1. `cube_core.py` — Add `triangulate_camera_pair()`

Place after `filter_dlc_h5()` (around line 2525).

```python
def triangulate_camera_pair(h5_cam1, h5_cam2, calib_toml, out_path, log_fn=print):
```

Internals:
1. Guard: `try: from aniposelib.cameras import CameraGroup except ImportError: raise ImportError("aniposelib not found — install via: pip install aniposelib")`
2. Load calibration: `cgroup = CameraGroup.load(str(calib_toml))`
3. Load and normalise both H5 files with the existing `_normalise_dlc_df()` helper
4. Build `pts2d` array of shape `(2, N_frames, N_bp, 2)` and `scores` array `(2, N_frames, N_bp)` from x/y/likelihood columns
5. Triangulate: `pts3d = cgroup.triangulate(pts2d, scores=scores, progress=False)` → shape `(N_frames, N_bp, 3)`
6. Compute `combined_ll = np.minimum(scores[0], scores[1])` (conservative)
7. Build output DataFrame with doubled bodyparts (`bp` and `bp_z`) using standard 3-level MultiIndex
8. Save: `df_out.to_hdf(str(out_path), key="df_with_missing", mode="w", format="fixed")`
9. Add `triangulate_camera_pair` to the module namespace so `cube.py` can import it

### 2. `cube.py` — SessionState Fields

Add to `SessionState.DEFAULTS` dict (around line 185):

```python
mode_3d_enabled    = False,
mode_3d_calib_toml = "",
mode_3d_cam_prefix = "pseudo_resized_",
mode_3d_cam1_name  = "",   # auto-detected, display only
mode_3d_cam2_name  = "",   # auto-detected, display only
```

### 3. `cube.py` — Import

Add `triangulate_camera_pair` to the `from cube_core import (...)` line.

### 4. `cube.py` — `DualCamera3DWindow` class

New `tk.Toplevel` subclass placed after `AdvancedCUBEWindow` (~line 3210), following the same `_load()` / `_apply()` pattern as `AdvancedDLCWindow`.

Layout (~80 lines):
- **Row 1:** `CTkCheckBox` "Enable 3D Dual-Camera Mode" → `session["mode_3d_enabled"]`
- **Row 2:** Label + `CTkEntry` (path) + `CTkButton("Browse…")` → `filedialog.askopenfilename(filetypes=[("TOML","*.toml")])` → `session["mode_3d_calib_toml"]`
- **Row 3:** Label "Camera subfolder prefix:" + `CTkEntry` (default `pseudo_resized_`) → `session["mode_3d_cam_prefix"]`
- **Row 4:** `CTkButton("Detect Camera Folders")` → runs `_detect_camera_folders(session, video_folders)` and displays found cam1/cam2 subfolder names in read-only labels
- **Row 5:** Dim info label: *"Cameras matched in sorted subfolder order. Calibrate once with DLC's calibrate_cameras() or aniposelib."*
- **Bottom:** `CTkButton("Apply & Close")`

### 5. `cube.py` — `_detect_camera_folders()` helper

Free function (not a method):

```python
def _detect_camera_folders(session, video_folders):
    prefix = session.get("mode_3d_cam_prefix", "pseudo_resized_")
    cam_dirs = []
    for folder in video_folders:
        subs = sorted([d for d in Path(folder).iterdir()
                       if d.is_dir() and d.name.startswith(prefix)])
        cam_dirs.extend(subs)
    # Return first two found
    return cam_dirs[0] if len(cam_dirs) > 0 else None, \
           cam_dirs[1] if len(cam_dirs) > 1 else None
```

### 6. `cube.py` — Button wire-up

In `_build_top_pane()` near line 3435, add alongside the existing advanced buttons:

```python
CTkButton(adv_row, text="3D Dual-Camera…",
          command=self._open_3d_settings).pack(side="left", padx=4)
```

Add `_open_3d_settings` method: `DualCamera3DWindow(self, self._session)`.

### 7. `cube.py` — Mode branch in `_run_dlc_step`

At the top of `_run_dlc_step` (line ~1017), before existing mode checks:

```python
if bool(session.get("mode_3d_enabled", False)):
    _run_dlc_3d_step(session, settings, logger, pb, after_fn)
    return
```

### 8. `cube.py` — `_run_dlc_3d_step()` function

New function placed after `_run_dlc_zoo_per_video` (~line 2260). Structure:

**Phase 1 — Validate:**
- Check `mode_3d_calib_toml` is a real file; raise clear error if not
- Check `aniposelib` is importable; raise with install instructions if not

**Phase 2 — Discover camera subfolders:**
For each folder in `session["video_folders"]`:
- Find subdirs whose name starts with `mode_3d_cam_prefix`, sort them
- Require exactly 2; raise `ValueError` with folder name if not found
- Store as `(cam1_dir, cam2_dir)` pairs

**Phase 3 — 2D DLC inference per camera:**
For each `(cam1_dir, cam2_dir)`:
- Inline the core 80-line video inference block from `_run_dlc_step` (resize video → `dlc.video_inference_superanimal(...)` → rename H5 → `filter_dlc_h5(...)`) applied first to each video in `cam1_dir`, then each in `cam2_dir`
- Call `cleanup_video_byproducts(results_subdir, logger)` — pass only the `_results` subdirectory (never the camera source subfolder itself)

**Phase 4 — Match and triangulate:**
For each `(cam1_dir, cam2_dir)` pair:
- Collect `*_filtered.h5` files from `cam1_dir` and `cam2_dir`, sort by stem
- Assert equal counts (they are synced)
- For each matched pair: call `triangulate_camera_pair(h5_cam1, h5_cam2, calib_toml, parent_folder / f"{shared_stem}_3d_filtered.h5", log_fn=logger.info)`

**Phase 5 — Prep (optional):**
- If `session.get("dlc_run_prep", True)`: call `run_bsoid_prep(session, ...)` on the parent video folder (which now contains the `_3d_filtered.h5` files)

---

## Cleanup Safety Note

`cleanup_video_byproducts(dest, logger)` is only ever called with a `_results`-suffixed path. Camera source subfolders (`pseudo_resized_Cam*`) live at the parent video folder root and are never passed to this function. **No change needed** — add a one-line comment at the call site in `_run_dlc_3d_step` confirming this invariant.

---

## Downstream Data Flow (Steps 2–5, unchanged)

```
parent_folder/
  <stem>_3d_filtered.h5   ← standard 3-level MultiIndex, 2×N_bp bodyparts
       ↓
run_bsoid_prep()          ← unchanged: detects *_filtered.h5, normalises, exports
       ↓
BSOID_Project_Ready/h5/   ← conserved bodyparts (bp + bp_z per original bp)
       ↓
BSoidEngine / extract_features_v2()  ← unchanged: sees 2×N_bp keypoints,
                                        computes XY pairwise distances + velocities
                                        Z info encoded via bp_z pseudo-keypoints
       ↓
UMAP → HDBSCAN → MLP → HMM   ← unchanged
       ↓
Steps 4–5 (annotation, analysis)  ← unchanged
```

---

## Verification

1. **Unit test `triangulate_camera_pair`:** Use two synthetic H5 files with random 2D coords and a mock calibration TOML. Confirm output has 3-level MultiIndex, `bp_z` bodyparts present, no NaN-dominated columns.
2. **`_is_bsoid_ready_h5` check:** Confirm `<stem>_3d_filtered.h5` is accepted (stem ends with `_filtered`, no `BSOID_` prefix).
3. **`run_bsoid_prep` smoke test:** Point at a folder with one 3D H5. Confirm `BSOID_Project_Ready/h5/` is created with one file.
4. **`BSoidEngine` smoke test:** Load the BSOID-ready H5, run `extract_features_v2`. Confirm features are produced without error.
5. **Cleanup safety test:** Verify camera source subfolders are not deleted after a full Step 1 run.
6. **2D regression test:** Disable 3D mode, run a standard 2D video folder. Confirm full pipeline unchanged.
7. **End-to-end real data test:** Provide real calibration TOML + two matched camera subfolders. Inspect 3D H5 for plausible coordinates, run Steps 1–3, confirm cluster labels are produced.
