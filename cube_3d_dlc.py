"""
cube_3d_dlc.py — 3D DLC + Anipose pipeline for CUBE

Orchestrates per-camera DeepLabCut Zoo adaptation, fill-frame masking,
aniposelib triangulation, quad-composite video export, and optional 3D
skeleton visualization for Step 0 synchronised recordings (4 cameras).

Called from cube.py via _launch_3d_dlc → _run_step("dlc_3d", _run_3d_dlc_step).
"""

from __future__ import annotations

import functools
import gc
import json
import os
import shutil
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from cube_core import (SUPERANIMAL_QUADRUPED_CHAINS, SUPERANIMAL_QUADRUPED_EXCLUDED_BODYPARTS,
                        infer_skeleton as _infer_skeleton_core)


# ─────────────────────────────────────────────────────────────────────────────
#  Session discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_step0_sessions(root: Path) -> list[dict]:
    """
    Discover completed Step 0 sessions under *root* (recursive).

    Returns a list of session dicts sorted by session_id:
        {
          "session_id": str,
          "cameras":    {cam_label: Path(mp4)},
          "timestamps": {cam_label: Path(csv)},
          "report":     dict,            # parsed JSON; {} if missing
          "fps":        float,           # from report or 30.0 default
        }

    Only sessions where every camera in the report has status=="done" are
    returned.  Sessions with a missing report are included only if at least
    2 camera MP4s are found (partial-completion guard is then skipped).
    """
    from cube_core import parse_doric_filename

    sessions: dict[str, dict] = {}

    for mp4 in sorted(root.rglob("*_corrected.mp4")):
        # Skip calibration recordings/videos (picked up by Phase 1 auto-calibration, not animal DLC)
        try:
            rel_parts = mp4.relative_to(root).parts[:-1]
        except ValueError:
            rel_parts = ()
        if any("calibration" in p.lower() for p in rel_parts):
            continue

        parsed = parse_doric_filename(mp4)
        if parsed is None:
            continue
        cam_label, session_id = parsed["cam_label"], parsed["session_id"]
        if session_id not in sessions:
            sessions[session_id] = {
                "session_id": session_id,
                "cameras": {},
                "timestamps": {},
                "report": {},
                "fps": 30.0,
            }
        sessions[session_id]["cameras"][cam_label] = mp4

        ts_csv = mp4.parent / f"{cam_label}_{session_id}_corrected_timestamps.csv"
        if ts_csv.exists():
            sessions[session_id]["timestamps"][cam_label] = ts_csv

    for sid, sess in sessions.items():
        report_candidates = list(root.rglob(f"{sid}_correction_report.json"))
        if report_candidates:
            try:
                sess["report"] = json.loads(report_candidates[0].read_text())
                sess["fps"] = float(sess["report"].get("fps", 30.0))
            except Exception:
                pass

    complete = []
    for sess in sessions.values():
        report = sess["report"]
        if report and "cameras" in report:
            if not all(c["status"] == "done" for c in report["cameras"]):
                continue
        elif len(sess["cameras"]) < 2:
            continue
        complete.append(sess)

    complete.sort(key=lambda s: s["session_id"])
    return complete


def should_use_homography_fallback(sessions: list[dict], fallback_enabled: bool) -> bool:
    """
    Decide whether _run_3d_dlc_step should fall back to 2D floor-homography
    fusion instead of aborting when no calibration.toml can be resolved.

    True only when *fallback_enabled* (the dlc_3d_homography_fallback_enabled
    checkbox) is on AND every discovered session has <=3 cameras — a
    4+-camera rig always needs a real calibration, no fallback, matching
    today's existing abort behavior exactly. See
    C:\\Users\\param\\.claude\\plans\\woolly-cuddling-goose.md.
    """
    max_cams = max((len(s["cameras"]) for s in sessions), default=0)
    return bool(fallback_enabled) and max_cams <= 3


def present_camera_labels(sessions: list[dict]) -> list[str]:
    """
    Camera labels ACTUALLY present across *sessions* (union of each
    session's own "cameras" dict keys), sorted.

    Deliberately distinct from the run's configured `dlc_3d_cam_labels`,
    which defaults to `["cam0","cam1","cam2","cam3"]` regardless of how
    many cameras a given rig really has (correct for the calibrated 4-
    camera path, where every configured label is expected to exist).
    Used to scope the homography-fallback path's calibration-pair
    prompting (ensure_camera_pairs_calibrated) and per-session fusion so a
    genuine 2-camera rig is never asked to calibrate phantom pairs like
    cam2-cam3 that have no video at all.
    """
    return sorted({cam for s in sessions for cam in s["cameras"].keys()})


# ─────────────────────────────────────────────────────────────────────────────
#  Auto-run Step 0 for raw (uncorrected) recordings
# ─────────────────────────────────────────────────────────────────────────────

# Step 0 (bonsai_video_corrector.py) is a separate standalone tkinter app,
# not a library this module can call into directly — see its own module
# docstring. Hardcoded like the other cross-file paths this environment
# assumes (D:\CUBE, D:\3D DLC); guarded with an existence check everywhere
# it's used rather than assumed.
_STEP0_SCRIPT = Path(r"D:\3D DLC\bonsai_video_corrector.py")


def _load_step0_module():
    r"""
    Imports bonsai_video_corrector.py by file path (it lives outside this
    package, in D:\3D DLC, not on sys.path by default) so its pure functions
    (scan_folder) can be reused without launching its GUI — the module-level
    code only builds constants/dataclasses; VideoPrep().mainloop() is gated
    behind `if __name__ == "__main__":`, so importing it here is safe.
    Returns None (logged by the caller) if the script isn't present.
    """
    if not _STEP0_SCRIPT.exists():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bonsai_video_corrector", str(_STEP0_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    # Must be registered in sys.modules BEFORE exec_module() runs — the
    # module's own @dataclass-decorated classes (SessionInfo, CamInfo) look
    # up sys.modules[cls.__module__] internally during class creation, which
    # is empty/absent otherwise and crashes with an unrelated-looking
    # AttributeError deep inside the dataclasses stdlib module.
    sys.modules["bonsai_video_corrector"] = mod
    spec.loader.exec_module(mod)
    return mod


def _folders_needing_step0(source_folders: list[Path], logger) -> list[Path]:
    """
    Returns the subset of source_folders that contain at least one raw,
    valid Bonsai/Doric recording session (per Step 0's own scan_folder) with
    no corresponding *_corrected.mp4 output yet (per this module's own
    find_step0_sessions) — i.e. folders that genuinely need Step 0 run
    before Phase 1 can find anything. A folder already fully corrected, or
    with no raw recordings at all (e.g. externally-prepared videos), is
    never included.
    """
    mod = _load_step0_module()
    if mod is None:
        logger.warn(f"  Step 0 script not found at {_STEP0_SCRIPT} — "
                     f"cannot check for unprocessed raw recordings")
        return []
    needing = []
    for folder in source_folders:
        try:
            # Exclude calibration recordings — find_step0_sessions() excludes
            # them too (session_id containing "calibration"), since they're
            # never expected to produce a *_corrected.mp4: Step 0 copies them
            # verbatim into _calibration/{session_id}/ instead, done during
            # the scan step itself (copy_calibration_sessions(), wired into
            # _on_scan) rather than the Run/correct step this function checks
            # for. Without this exclusion, a folder whose only remaining raw
            # recording is its calibration one would be wrongly flagged as
            # still needing Step 0 forever.
            raw_sessions = [s for s in mod.scan_folder(folder)
                             if s.valid and "calibration" not in s.session_id.lower()]
        except Exception as e:
            logger.warn(f"  Step 0 raw-recording scan failed for {folder}: {e}")
            continue
        if not raw_sessions:
            continue
        corrected_ids = {s["session_id"] for s in find_step0_sessions(folder)}
        raw_ids = {s.session_id for s in raw_sessions}
        if raw_ids - corrected_ids:
            needing.append(folder)
    return needing


def _autorun_step0(folders: list[Path], logger) -> None:
    """
    Launches bonsai_video_corrector.py as its own separate process/window,
    one folder at a time — pre-filling and auto-scanning that folder (via
    the CLI-arg support added to its entry point for exactly this use) so
    the user only has to review the discovered session list and click Run
    themselves. This deliberately keeps the pre-encode manual review
    checkpoint (catching a bad camera/session grouping before a real,
    possibly hours-long re-encode) — only the "open the app, browse to the
    folder, click scan" friction is removed, not the review itself.

    Blocks until each window is closed before moving to the next folder —
    safe to block here since this runs in _run_3d_dlc_step's own background
    orchestrator thread, never the Tk main thread (same convention Step 0
    itself uses for its own per-camera worker threads).
    """
    if not _STEP0_SCRIPT.exists():
        raise FileNotFoundError(
            f"dlc_3d_autorun_step0 is on but Step 0 script not found at "
            f"{_STEP0_SCRIPT} — run Step 0 manually first, or disable "
            f"dlc_3d_autorun_step0.")
    for folder in folders:
        logger.info(f"  [auto-step0] Launching Step 0 for: {folder}")
        proc = subprocess.Popen([sys.executable, str(_STEP0_SCRIPT), str(folder)])
        proc.wait()
        logger.info(f"  [auto-step0] Step 0 window closed for: {folder}")


def select_representative(session_dicts: list[dict], cam_label: str) -> "Path | None":
    """
    Return the corrected MP4 for *cam_label* that has the lowest fill rate
    (most genuine frames → best for Zoo adaptation).
    """
    candidates = []
    for sess in session_dicts:
        mp4 = sess["cameras"].get(cam_label)
        if mp4 is None:
            continue
        report = sess["report"]
        fill_rate = 1.0
        if report and "cameras" in report:
            for c in report["cameras"]:
                if c["cam"] == cam_label and c.get("n_output_frames", 0) > 0:
                    fill_rate = c.get("fills_inserted", 0) / c["n_output_frames"]
                    break
        candidates.append((fill_rate, mp4))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# ─────────────────────────────────────────────────────────────────────────────
#  Fill-frame masking
# ─────────────────────────────────────────────────────────────────────────────

def mask_fill_frames_in_h5(h5_path: Path, timestamps_csv: Path,
                            out_path: Path, log_fn=print) -> None:
    """
    Zero-out the likelihood of fill frames in a DLC H5 so that
    triangulation discards them when choosing the best camera.

    Atomic: writes to a .tmp.h5 then renames (safe if h5_path == out_path).
    """
    import pandas as pd

    data = np.genfromtxt(str(timestamps_csv), delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    fill_mask = data[:, 3].astype(bool)

    df = pd.read_hdf(str(h5_path))
    scorer = df.columns.get_level_values(0)[0]
    bodyparts = df[scorer].columns.get_level_values(0).unique().tolist()

    # Guard: reconcile frame counts if DLC inference dropped or added frames
    n_h5  = len(df)
    n_csv = len(fill_mask)
    if n_h5 != n_csv:
        log_fn(f"  [WARN] mask_fill_frames: H5 has {n_h5} frames but CSV has {n_csv} "
               f"timestamps — reconciling (delta={n_h5 - n_csv:+d})")
        if n_csv > n_h5:
            fill_mask = fill_mask[:n_h5]
        else:
            fill_mask = np.pad(fill_mask, (0, n_h5 - n_csv), constant_values=False)

    for bp in bodyparts:
        try:
            df.loc[fill_mask, (scorer, bp, "likelihood")] = 0.0
        except Exception:
            pass

    tmp = out_path.with_suffix(".tmp.h5")
    df.to_hdf(str(tmp), key="df_with_missing", mode="w", format="fixed")
    tmp.replace(out_path)


PERCAM_HEAD_CAGE_PAIRS = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "right_eye"),
    ("left_eye", "left_earbase"), ("right_eye", "right_earbase"),
    ("left_earbase", "right_earbase"),
]


def _percam_foreground_mask(gray, dilate_px: int = 12):
    """
    Otsu-threshold silhouette mask for one grayscale video frame -- direct
    per-frame intensity thresholding, NOT background subtraction. A
    median-frame / MOG2 background model was tried first and rejected: the
    animal re-occupies similar positions often enough across a session
    that the background model itself becomes partly mouse-shaped, turning
    the diff mask into a hollow ring around the body edges rather than a
    solid blob (53% false-flag rate on a real torso-interior point under
    that approach; see CLAUDE.md's "Per-Camera 2D Pose Cleaning" section).
    Returns a uint8 0/255 mask of the animal's silhouette + a dilation
    margin, or an all-zero mask if no foreground component is found.
    """
    import cv2
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = (lbl == largest).astype(np.uint8) * 255
    if dilate_px > 0:
        keep = cv2.dilate(keep, np.ones((dilate_px, dilate_px), np.uint8))
    return keep


def percam_pose_cleaning_gate(
        h5_path: Path, video_path: Path, out_path: Path,
        ll_gate: float = 0.4,
        silhouette_gate: bool = True,
        bone_length_gate: bool = True,
        bone_len_factor: float = 3.0,
        sil_margin_dilate: int = 12,
        max_interp_gap_s: float = 5.0 / 30.0,
        fps: float = 30.0,
        jitter_reject_factor: float = 3.0,
        head_cage_pairs=None,
        log_fn=print) -> dict:
    """
    Per-camera 2D pose cleaning applied BEFORE triangulation -- called from
    _run_3d_dlc_step's Phase 2, immediately after mask_fill_frames_in_h5().
    Real-data-validated on F:\\TMZ7 (2026-09-16); see CLAUDE.md's
    "Per-Camera 2D Pose Cleaning" section for full methodology, validation
    numbers, and the removed motion-freeze detector's rejection rationale
    (do not reintroduce a freeze/occlusion detector here without new
    evidence addressing the paw stance-phase false-positive mechanism
    documented there). No-op (file untouched) if both gates are off.

    Three passes, in order:
      1. Silhouette containment gate (silhouette_gate): a confidently-
         tracked point (likelihood >= ll_gate) that falls outside the
         animal's own Otsu-thresholded silhouette (+ sil_margin_dilate px
         margin) that frame is flagged.
      2. Scoped head-cage relative bone-length gate (bone_length_gate):
         restricted to head_cage_pairs (near-rigid skull landmarks only --
         never limb segments, which genuinely foreshorten during
         locomotion). Flags the lower-confidence endpoint of any pair
         whose instantaneous 2D distance exceeds bone_len_factor times
         this session's own median distance for that pair (learned only
         from confidently-tracked frames).
      3. Short-gap linear interpolation + jitter-ratio guardrail: any
         point flagged by 1-2 above, or already below ll_gate, gets
         linearly interpolated across gaps <= max_interp_gap_s seconds
         (converted to a frame count via `fps` internally, so the real
         time window stays identical at 60/120fps, not just 30fps);
         accepted only if the interpolated segment's own local jitter is
         within jitter_reject_factor x that bodypart's real (genuinely-
         tracked) jitter, else reverted to a gap (NaN position,
         likelihood 0). An accepted fill is given likelihood == ll_gate
         exactly -- a recovered-but-unconfident detection, just enough to
         clear the triangulation gate, matching this codebase's existing
         "a temporally-interpolated position is never reported as a fully
         confident detection" convention (dlc_3d_bone_constraints,
         _apply_nose_multipoint_prior).

    Idempotent / resume-safe: the FIRST call for a given h5_path writes an
    untouched copy to '{h5_path.stem}.pre_percam_cleaning_backup.h5'
    alongside it and cleans from that backup; every SUBSEQUENT call (e.g.
    a resumed run) re-derives from that same backup rather than the
    previous call's own output, so repeated invocation can never stack
    cleaning passes on top of each other.

    Returns a stats dict: {'flagged', 'silhouette', 'bone_length',
    'interpolated', 'reverted_to_nan', 'total_slots'}.
    """
    import cv2
    import pandas as pd

    stats = dict(flagged=0, silhouette=0, bone_length=0,
                 interpolated=0, reverted_to_nan=0, total_slots=0)
    if not silhouette_gate and not bone_length_gate:
        return stats

    head_cage_pairs = head_cage_pairs or PERCAM_HEAD_CAGE_PAIRS

    backup_path = h5_path.with_name(h5_path.stem + ".pre_percam_cleaning_backup.h5")
    if backup_path.exists():
        src_path = backup_path
    else:
        shutil.copy2(str(h5_path), str(backup_path))
        src_path = backup_path

    df = pd.read_hdf(str(src_path))
    scorer = df.columns.get_level_values(0)[0]
    bps = df[scorer].columns.get_level_values(0).unique().tolist()
    n = len(df)
    X = np.stack([df[(scorer, bp, "x")].values.astype(float) for bp in bps], axis=1)
    Y = np.stack([df[(scorer, bp, "y")].values.astype(float) for bp in bps], axis=1)
    L = np.stack([df[(scorer, bp, "likelihood")].values.astype(float) for bp in bps], axis=1)
    bi = {b: i for i, b in enumerate(bps)}
    stats["total_slots"] = int(n * len(bps))
    # Time-based (not a raw frame count) so the real interpolation window
    # this was validated against (~0.167s = 5 frames @ 30fps) stays
    # identical at any camera fps (60/120fps rigs included).
    max_interp_gap = max(1, int(round(max_interp_gap_s * fps)))

    sil_flags = np.zeros((n, len(bps)), dtype=bool)
    if silhouette_gate:
        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        while True:
            ok, fr = cap.read()
            if not ok or idx >= n:
                break
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            mask = _percam_foreground_mask(gray, dilate_px=sil_margin_dilate)
            h, w = mask.shape
            for b_i in range(len(bps)):
                if L[idx, b_i] < ll_gate:
                    continue
                x, y = X[idx, b_i], Y[idx, b_i]
                if np.isnan(x) or np.isnan(y):
                    continue
                xi, yi = int(round(x)), int(round(y))
                if xi < 0 or yi < 0 or xi >= w or yi >= h or mask[yi, xi] == 0:
                    sil_flags[idx, b_i] = True
            idx += 1
        cap.release()
        if 0 < idx < n:
            log_fn(f"  [WARN] percam cleaning: video had {idx} frames but H5 has {n} "
                   f"-- silhouette gate only checked the first {idx}")

    bone_flags = np.zeros((n, len(bps)), dtype=bool)
    if bone_length_gate:
        for a, b in head_cage_pairs:
            if a not in bi or b not in bi:
                continue
            ia, ib = bi[a], bi[b]
            both_ok = (L[:, ia] >= ll_gate) & (L[:, ib] >= ll_gate)
            dist = np.sqrt((X[:, ia] - X[:, ib]) ** 2 + (Y[:, ia] - Y[:, ib]) ** 2)
            ref = np.nanmedian(dist[both_ok]) if both_ok.sum() > 20 else np.nan
            if np.isnan(ref) or ref <= 0:
                continue
            bad = both_ok & (dist > bone_len_factor * ref)
            lower_is_a = L[:, ia] <= L[:, ib]
            bone_flags[bad & lower_is_a, ia] = True
            bone_flags[bad & ~lower_is_a, ib] = True

    combined_flag = sil_flags | bone_flags
    stats["silhouette"] = int(sil_flags.sum())
    stats["bone_length"] = int(bone_flags.sum())
    stats["flagged"] = int(combined_flag.sum())

    # Gap-fill candidates: newly-flagged points OR already-below-ll_gate.
    need_fill = combined_flag | (L < ll_gate)
    Xc, Yc, Lc = X.copy(), Y.copy(), L.copy()
    Xc[need_fill] = np.nan
    Yc[need_fill] = np.nan

    n_interp = 0
    n_reverted = 0
    for b_i in range(len(bps)):
        col_bad = need_fill[:, b_i]
        if not col_bad.any():
            continue
        good_idx = np.where(~col_bad)[0]
        if len(good_idx) < 2:
            n_reverted += int(col_bad.sum())
            Lc[col_bad, b_i] = 0.0
            continue
        # Real jitter reference, from genuinely-tracked consecutive frames.
        gd = np.diff(good_idx)
        consec = good_idx[:-1][gd == 1]
        if len(consec):
            ref_jit = np.median(np.sqrt(
                (X[consec + 1, b_i] - X[consec, b_i]) ** 2 +
                (Y[consec + 1, b_i] - Y[consec, b_i]) ** 2))
        else:
            ref_jit = np.nan

        i = 0
        while i < n:
            if not col_bad[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and col_bad[j + 1]:
                j += 1
            gap_len = j - i + 1
            prev_i, next_i = i - 1, j + 1
            can_interp = (gap_len <= max_interp_gap and prev_i >= 0 and next_i < n
                          and not col_bad[prev_i] and not col_bad[next_i])
            if can_interp:
                t = np.arange(i, j + 1)
                w = (t - prev_i) / float(next_i - prev_i)
                fx = (1 - w) * X[prev_i, b_i] + w * X[next_i, b_i]
                fy = (1 - w) * Y[prev_i, b_i] + w * Y[next_i, b_i]
                accept = True
                if not np.isnan(ref_jit) and ref_jit > 0:
                    seg_x = np.concatenate([[X[prev_i, b_i]], fx, [X[next_i, b_i]]])
                    seg_y = np.concatenate([[Y[prev_i, b_i]], fy, [Y[next_i, b_i]]])
                    seg_jit = np.median(np.sqrt(np.diff(seg_x) ** 2 + np.diff(seg_y) ** 2))
                    if seg_jit > jitter_reject_factor * ref_jit:
                        accept = False
                if accept:
                    Xc[i:j + 1, b_i] = fx
                    Yc[i:j + 1, b_i] = fy
                    Lc[i:j + 1, b_i] = ll_gate
                    n_interp += gap_len
                else:
                    Lc[i:j + 1, b_i] = 0.0
                    n_reverted += gap_len
            else:
                Lc[i:j + 1, b_i] = 0.0
                n_reverted += gap_len
            i = j + 1

    stats["interpolated"] = int(n_interp)
    stats["reverted_to_nan"] = int(n_reverted)

    for b_i, bp in enumerate(bps):
        df[(scorer, bp, "x")] = Xc[:, b_i]
        df[(scorer, bp, "y")] = Yc[:, b_i]
        df[(scorer, bp, "likelihood")] = Lc[:, b_i]

    tmp2 = out_path.with_suffix(".tmp.h5")
    df.to_hdf(str(tmp2), key="df_with_missing", mode="w", format="fixed")
    tmp2.replace(out_path)

    log_fn(f"  Per-camera pose cleaning ({video_path.stem}): flagged={stats['flagged']} "
           f"(silhouette={stats['silhouette']}, bone={stats['bone_length']}) "
           f"interp={stats['interpolated']} reverted_to_nan={stats['reverted_to_nan']} "
           f"/ {stats['total_slots']} slots")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  Quad composite video
# ─────────────────────────────────────────────────────────────────────────────

def _probe_video_dimensions(ffmpeg_exe: str, video_path: Path) -> tuple | None:
    """Return (width, height) by parsing FFmpeg's -i stderr output."""
    import re
    try:
        r = subprocess.run(
            [ffmpeg_exe, "-i", str(video_path)],
            capture_output=True, timeout=15)
        text = r.stderr.decode(errors="replace")
        m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", text)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return None


def _crop_camera_video(mp4_path: Path, rect: dict, ffmpeg_exe: str, logger,
                        out_dir: Path | None = None) -> Path:
    """
    Crop *mp4_path* to *rect* = {"x","y","w","h"} (pixels, in the source
    video's own frame), writing {stem}_cropped.mp4 into *out_dir* (this
    run's "CUBE 1b - 3D DLC/_prep/" folder -- the crop pipeline itself is
    Step 1b's own processing, not Step 0's, so its output belongs under
    Step 1b's folder rather than alongside the source video). Defaults to
    mp4_path.parent (the old alongside-the-source-video behavior) when
    out_dir isn't given, for any caller that hasn't been updated.

    Skip-if-exists.  Runs before adaptation/inference (so DLC tracks a
    tighter, animal-filling frame) and the resulting cropped video also
    becomes the camera's video for the overlay/quad/skeleton outputs;
    _triangulate_session() adds the crop offset back before triangulation
    so 2D coordinates line up with calibration.toml's full-frame space.
    """
    out_dir = Path(out_dir) if out_dir is not None else mp4_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{mp4_path.stem}_cropped.mp4"
    req_rect = {"x": int(rect["x"]), "y": int(rect["y"]),
                "w": int(rect["w"]), "h": int(rect["h"])}

    # Sidecar JSON recording the exact rect used, so a future run can
    # recover dlc_3d_crop_rects from disk (see _recover_crop_rects_from_prep)
    # even if the in-memory/persisted session that originally set it is
    # ever lost (e.g. a crash followed by a fresh session) -- crop x/y is
    # NOT otherwise derivable from the cropped video file itself (only w/h
    # are; the offset within the original frame isn't encoded anywhere
    # else on disk). It also doubles as the skip-if-exists cache key: the
    # output filename doesn't encode the rect at all, so without comparing
    # against a recorded rect, a changed crop while an old {stem}_cropped.mp4
    # from a *different* rect already sits at that same deterministic path
    # would silently keep reusing the stale video forever.
    crop_meta_path = out_dir / f"{out_path.stem}.crop.json"
    prev_rect = None
    if crop_meta_path.exists():
        try:
            prev_rect = json.loads(crop_meta_path.read_text())
        except Exception:
            prev_rect = None

    if out_path.exists() and prev_rect == req_rect:
        return out_path
    if out_path.exists():
        # Either a genuine rect change, or a pre-sidecar cropped video (no
        # recorded rect to trust) -- either way, re-crop rather than guess.
        # Cropping is a cheap ffmpeg pass (seconds) against the still-intact
        # raw source video, unlike the DLC adaptation/inference downstream
        # of it, so there's no real cost to regenerating whenever the cache
        # is anything less than a confirmed exact match.
        logger.info(f"  [{mp4_path.name}] Cached {out_path.name} doesn't match "
                    f"the requested crop rect (or predates rect tracking) — "
                    f"re-cropping from source")

    # Atomic: ffmpeg writes to a .tmp sibling, renamed onto out_path only on
    # success -- a crash/kill mid-encode would otherwise leave a truncated
    # file at out_path itself, which the skip-if-exists check above would
    # then treat as finished forever.
    # tmp marker goes BEFORE the real suffix (foo.tmp.mp4, not foo.mp4.tmp) --
    # ffmpeg infers output container from the last extension with no
    # explicit -f override, and ".tmp" is not a recognized container.
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    x, y, w, h = int(rect["x"]), int(rect["y"]), int(rect["w"]), int(rect["h"])
    # Real retry across encoders (mirrors make_quad_video's pattern), not just
    # a one-shot pick based on the cached _nvenc_available() probe: crop now
    # runs several jobs concurrently (see the ThreadPoolExecutor call site),
    # so a transient NVENC concurrent-session-limit failure is a real
    # possibility here even when _nvenc_available() reported true. Without a
    # real fallback attempt, that failure would raise and the caller would
    # silently use the UNCROPPED video for DLC tracking instead -- unlike
    # overlay's fallback (a cosmetic QC-video loss), losing the crop here
    # defeats the actual purpose of cropping (tighter DLC tracking).
    encoders = [("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]),
                ("libx264",    ["-c:v", "libx264", "-crf", "18", "-preset", "fast"])]
    if not _nvenc_available(ffmpeg_exe):
        encoders = encoders[1:]
    last_stderr = b""
    for enc_name, enc_args in encoders:
        cmd = [ffmpeg_exe, "-y", "-i", str(mp4_path),
               "-vf", f"crop={w}:{h}:{x}:{y}",
               *enc_args,
               str(tmp_path)]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            tmp_path.replace(out_path)
            try:
                crop_meta_path.write_text(json.dumps(req_rect))
            except Exception:
                pass
            logger.info(f"  Cropped video written [{enc_name}]: "
                        f"{out_path.name} ({w}x{h} @ {x},{y})")
            return out_path
        last_stderr = result.stderr
        if enc_name == "h264_nvenc":
            logger.info(f"  [{mp4_path.name}] NVENC crop failed, retrying with libx264")
    tmp_path.unlink(missing_ok=True)
    raise RuntimeError(
        f"Crop failed for {mp4_path.name}:\n{last_stderr.decode(errors='replace')}")


def _recover_crop_rects_from_prep(step1_prep: Path, logger) -> dict:
    """
    Reconstruct dlc_3d_crop_rects from the *.crop.json sidecars
    _crop_camera_video() writes next to each {stem}_cropped.mp4 in this
    run's _prep folder. Used when session["dlc_3d_crop_rects"] is empty
    (e.g. a fresh session after a crash lost the original in-memory/
    persisted setting) but a prior run of this same project already
    cropped videos on disk -- without this, an empty crop_rects silently
    falls back to the raw uncropped video, which then can't match any of
    the existing *_cropped_*.h5 outputs a resume would otherwise reuse.
    cam_label is read from each sidecar's own filename prefix (videos are
    always named "{cam_label}_..."), not stored separately. One rect per
    camera is enough -- all sessions for a given camera share the same
    rect, so the first sidecar found for a camera wins.
    """
    recovered: dict = {}
    if not step1_prep.exists():
        return recovered
    for meta_path in sorted(step1_prep.glob("*.crop.json")):
        cam_label = meta_path.name.split("_", 1)[0]
        if cam_label in recovered:
            continue
        try:
            rect = json.loads(meta_path.read_text())
            if int(rect.get("w", 0)) > 0 and int(rect.get("h", 0)) > 0:
                recovered[cam_label] = rect
        except Exception as e:
            logger.warn(f"  Failed to read crop sidecar {meta_path.name}: {e}")
    if recovered:
        logger.info(f"  Recovered per-camera crop rect(s) from disk: "
                    f"{', '.join(sorted(recovered))}")
    return recovered


def discover_crop_rects_from_source_folders(source_folders: list, logger=None) -> dict:
    """
    Recursively scans every folder in *source_folders* for *.crop.json
    sidecars written by any PRIOR _crop_camera_video() run anywhere under
    that folder tree -- not just one specific run's own "CUBE 1b - 3D
    DLC/_prep" folder, unlike _recover_crop_rects_from_prep() (which is
    scoped to the CURRENT run root and only ever runs deep inside a live
    Step 1b pass). Intended to be called the moment a folder is added to
    the main panel (cube.py's _folders_changed()), so a returning user's
    crop presets are auto-restored immediately -- before 3D DLC Settings
    is even opened -- rather than silently lost whenever a fresh run root
    is created against the same raw recordings (e.g. re-running Step 0),
    since that fresh run root's own _prep folder starts out empty.

    Root-caused on F:\\TMZ7 (2026-09-17): a new "CUBE results <ts>" run
    root nested under an old one meant dlc_3d_crop_rects was empty and
    _recover_crop_rects_from_prep() found nothing (new run's _prep was
    empty), even though the OLD run's own "CUBE 1b - 3D DLC/_prep" folder
    -- reachable under the very folder the user re-added to the main
    panel -- still had every crop.json sidecar from before. A live 3D DLC
    run then silently tracked on full, uncropped frames with no warning
    until the Zoo-adaptation log's video filename was inspected by hand.

    One rect per camera; the first sidecar found for a camera wins (same
    convention as _recover_crop_rects_from_prep()). Returns {} (a no-op)
    if none of the folders contain any crop.json sidecar anywhere.
    """
    recovered: dict = {}
    for folder in source_folders:
        folder = Path(folder)
        if not folder.exists():
            continue
        for meta_path in sorted(folder.rglob("*.crop.json")):
            cam_label = meta_path.name.split("_", 1)[0]
            if cam_label in recovered:
                continue
            try:
                rect = json.loads(meta_path.read_text())
                if int(rect.get("w", 0)) > 0 and int(rect.get("h", 0)) > 0:
                    recovered[cam_label] = {
                        "x": int(rect.get("x", 0)), "y": int(rect.get("y", 0)),
                        "w": int(rect.get("w", 0)), "h": int(rect.get("h", 0)),
                    }
            except Exception as e:
                if logger:
                    logger.warn(f"  Failed to read crop sidecar {meta_path.name}: {e}")
    if recovered and logger:
        logger.info(f"  Auto-discovered per-camera crop rect(s) from folder history: "
                    f"{', '.join(sorted(recovered))}")
    return recovered


def make_quad_video(cam_mp4s: dict, out_path: Path,
                    ffmpeg_exe: str, logger) -> None:
    """
    Create a 2×2 composite MP4 from up to four camera streams.

    All input streams are scaled to a common resolution (most frequent W×H
    across cameras) before compositing, so mismatched sensors produce a
    correctly-dimensioned output.  Encoding prefers NVENC (GPU) and falls
    back to multi-threaded libx264 (CPU).

    cam_mp4s: {cam_label: Path}  (cam0 TL, cam1 TR, cam2 BL, cam3 BR)
    """
    sorted_cams = sorted(cam_mp4s.items())
    n = len(sorted_cams)
    if n < 2:
        raise ValueError(f"Need at least 2 cameras for quad video, got {n}")

    # ── Probe each stream's dimensions ───────────────────────────────────────
    dims: list[tuple[int, int] | None] = [
        _probe_video_dimensions(ffmpeg_exe, p) for _, p in sorted_cams
    ]
    valid_dims = [d for d in dims if d is not None]

    if valid_dims:
        from collections import Counter
        target_w = Counter(w for w, _ in valid_dims).most_common(1)[0][0]
        target_h = Counter(h for _, h in valid_dims).most_common(1)[0][0]
        logger.info(f"  Quad video: normalising streams to {target_w}x{target_h}")
    else:
        target_w = target_h = None

    # ── Build filter_complex ──────────────────────────────────────────────────
    inputs: list[str] = []
    for _, p in sorted_cams:
        inputs += ["-i", str(p)]

    # Label each pane with its camera name (e.g. "cam0") so a viewer can
    # tell which physical camera they're looking at without cross-
    # referencing the fixed TL/TR/BL/BR layout convention in this
    # function's own docstring. Needs an explicit fontfile on Windows —
    # ffmpeg's drawtext can't reliably find a system font via fontconfig
    # here — so this degrades to no labels (not a failure) if neither
    # common Windows font is present.
    _font_candidates = [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"]
    _font_path = next((f for f in _font_candidates if Path(f).exists()), None)
    _font_escaped = _font_path.replace("\\", "/").replace(":", "\\:") if _font_path else None

    if target_w and target_h:
        # force_original_aspect_ratio=decrease + pad (letterbox/pillarbox)
        # instead of a plain stretch scale: a per-camera crop (dlc_3d_crop_rects)
        # can leave one camera's aspect ratio different from the others', and a
        # bare scale=W:H would visibly squash/stretch that stream to fit —
        # cosmetic only (the actual 3D triangulation uses raw 2D coordinates,
        # never this composite), but distracting in a QC video meant to show
        # what each camera actually saw.
        scale_part = "".join(
            f"[{i}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
            + (f",drawtext=fontfile='{_font_escaped}':text='{sorted_cams[i][0]}':"
               f"x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6"
               if _font_escaped else "")
            + f"[s{i}];" for i in range(n))
        ins = "".join(f"[s{i}]" for i in range(n))
    elif _font_escaped:
        scale_part = "".join(
            f"[{i}:v]drawtext=fontfile='{_font_escaped}':text='{sorted_cams[i][0]}':"
            f"x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6[s{i}];"
            for i in range(n))
        ins = "".join(f"[s{i}]" for i in range(n))
    else:
        scale_part = ""
        ins = "".join(f"[{i}:v]" for i in range(n))

    if n == 4:
        xstack = f"{ins}xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]"
    elif n == 2:
        xstack = f"{ins}hstack=inputs=2[v]"
    else:
        xstack = f"{ins}xstack=inputs=3:layout=0_0|w0_0|0_h0[v]"

    filter_graph = scale_part + xstack

    base_cmd = [ffmpeg_exe, "-y"] + inputs + [
        "-filter_complex", filter_graph,
        "-map", "[v]",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]

    # ── Try GPU (NVENC) then fall back to multi-threaded CPU ──────────────────
    # Atomic: encode to a .tmp sibling, renamed onto out_path only on success
    # -- a crash/kill mid-encode (this is the video-export step that took
    # down the whole machine on 2026-08-28, see D:\3D DLC\CLAUDE.md) would
    # otherwise leave a truncated/corrupt file directly at out_path, which
    # every call site's `if not quad_path.exists()` skip check then treats
    # as a finished quad video forever.
    # tmp marker before the real suffix -- see _crop_camera_video's note;
    # ffmpeg needs the last extension to be a recognized container.
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    encoders = [
        ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]),
        ("libx264",    ["-c:v", "libx264", "-threads", "0", "-crf", "23", "-preset", "fast"]),
    ]
    for enc_name, enc_args in encoders:
        cmd = base_cmd + enc_args + [str(tmp_path)]
        logger.info(f"  Creating quad video [{enc_name}]: {out_path.name}")
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            tmp_path.replace(out_path)
            logger.info(f"  Quad video written: {out_path.name}")
            return
        if enc_name == "h264_nvenc":
            logger.info("  NVENC unavailable, retrying with libx264")
        else:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"FFmpeg quad composite failed:\n{result.stderr.decode()}")


# ─────────────────────────────────────────────────────────────────────────────
#  3D skeleton visualization video
# ─────────────────────────────────────────────────────────────────────────────

# SUPERANIMAL_QUADRUPED_EXCLUDED_BODYPARTS is now defined in cube_core.py
# (Two-Camera fusion Phase 4 extraction, 2026-09-07 -- its own validating
# comment notes this was confirmed chronically unreliable on BOTH the CUBE 3D
# rig and an independent single-camera 2D rig, making it directly applicable
# to the 2-camera fusion plan's skeleton/bodypart selection too, not just this
# 3D triangulation pipeline) and imported above. Kept as a module-level name
# here unchanged so every existing reference below still resolves.
#
# Bodyparts confirmed (D:\3D DLC — 2D-vs-3D confidence investigation, both the
# CUBE 3D rig and an independent single-camera 2D rig agree these are chronically
# unreliable) to correspond to no learnable feature on a mouse under the
# SuperAnimal-Quadruped schema (antler/mouth-corner/belly/body-middle/tail-tip
# points a mouse doesn't have distinct anatomy for). Excluded from triangulation,
# H5 output, overlays, skeleton video, and the legend — not merely hidden.
# neck_end and back_base added 2026-08-27: real-data investigation found both
# at ~0.000 median combined likelihood in both test sessions (back_base's rare
# surviving points were themselves 532px-median-error outliers on just 2 of
# ~11,000 frames pooled) — same chronic anatomy/schema-mismatch pattern as the
# bodyparts above, not fixable by triangulation method or camera count. See
# `D:\3D DLC\ERROR_SOURCES_AND_FIXES.md` item 3 and CLAUDE.md Crash Protection.
#
# back_middle added 2026-08-27 (same day, separate follow-up): initially kept
# as the sole "extra" interior spine point despite a weak 341px median
# reprojection error, on the assumption that SOME real tracking beat none.
# Real per-camera likelihood data proved otherwise: only 1.44% of frames
# (55/3818) have even 2 of 4 cameras clearing ll_gate=0.6 for back_middle —
# essentially the same dead tier as back_base (0.03%) and neck_end (0%), not
# a meaningfully better "least-bad option" as originally assumed. The other
# torso candidates (belly_bottom, body_middle_left/right) are equally dead on
# this rig (0% of frames, any camera) confirming there is no reliable 4th
# interior spine point anywhere in this schema for this rig — the backbone's
# real ceiling is exactly the 3 points below. A generic two-link geometric
# position prior (_two_link_expected_position / _apply_two_link_spine_prior)
# was built and validated to partially compensate for a case like this (a
# weak point flanked by two well-tracked neighbours) — median outcome was
# reasonable (comparable to real bodyparts' natural jitter) but with a real
# tail-risk of large spurious jumps near full two-rod extension (a mouse's
# spine curvature isn't well modeled by fixed chord lengths across its full
# range of motion) — see CLAUDE.md's Crash Protection entry for the numbers.
# Given back_middle's real tracking rate doesn't clear a materially higher
# bar than the already-excluded points, it was excluded rather than kept
# purely as a vehicle for that prior. The prior was instead retargeted at 6
# other candidates with a materially better real feasibility tier (14-56% vs.
# back_middle's 1.4%) — validated on real data and now the default targets in
# SPINE_GEOMETRIC_PRIOR_TARGETS below (session key
# `dlc_3d_spine_geometric_prior`, default on).

# Backbone anchor design (revised 2026-08-27): the spine has exactly 3
# limb/tail attachment points and NO extra interior point — neck_base (front
# legs), back_end (hind legs), tail_base (tail), connected directly
# (neck_base -> back_end) since no reliable point exists between them (see
# the back_middle exclusion note above). neck_base replaces the excluded
# neck_end as the front-leg anchor: it is not just present but the
# best-tracked point in the whole skeleton (22px median reprojection error
# vs. neck_end's 449px). Hind legs stay anchored at back_end (back/tail
# junction = hip region), unchanged from the original front/rear separation
# fix (front and hind legs must not share one anchor — that made all four
# legs visually fan out from the same point).
# SUPERANIMAL_QUADRUPED_CHAINS is now defined in cube_core.py (Two-Camera
# fusion Phase 4 extraction, 2026-09-07) and imported above as
# SUPERANIMAL_QUADRUPED_CHAINS -- kept as a module-level name here
# unchanged so every existing reference below still resolves.

# Reduced "body outline" point set + explicit edges for make_skeleton_video's
# render_bodyparts/render_edges — a recognizable quadruped silhouette (nose
# + 2 ears as a face triangle, a 3-point backbone, and 4 limbs with a real
# knee bend — 2 points each: knee + paw, not a single straight body-to-paw
# line — instead of every SuperAnimal-Quadruped keypoint). Explicit edges,
# not a chain-collapse of SUPERANIMAL_QUADRUPED_CHAINS: nose->eye->earbase
# is two separate chain entries there, so dropping the eye would break the
# connection rather than collapse through it — no single-list-collapse
# gives a real nose-earbase-earbase triangle. Front limbs hang from
# neck_base, hind limbs from back_end, matching the forelimb/hindlimb
# anchor fix documented for SUPERANIMAL_QUADRUPED_CHAINS above (not the
# anatomically-wrong back_base neck_end/back_base pairing). The thigh joint
# stays excluded (3 points/limb was not requested) — knee+paw is the
# minimum that shows a bend rather than a rigid stick.
MOUSE_OUTLINE_BODYPARTS = [
    # Both eyes are MANDATORY, not confidence-gated extras: measured on
    # real data (2026-08-28), left_eye/right_eye are tracked far more
    # reliably than the earbases (91-93% vs. 11-22% of frames clearing
    # ll_gate=0.3 on this rig) AND they sit laterally/below the nose-ear
    # line, breaking the otherwise near-planar nose+2-earbase triangle —
    # without them a head turn just skews a flat shape in-plane instead
    # of giving a real depth/rotation cue ("the head looks too 2D" —
    # confirmed as a real perceptual problem in a rendered video).
    "nose", "left_eye", "right_eye", "left_earbase", "right_earbase",
    "neck_base", "back_end",
    "front_left_knee", "front_right_knee", "back_left_knee", "back_right_knee",
    "front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw",
]
MOUSE_OUTLINE_HEAD_EDGES = [
    # A 3D "cage" (nose + both eyes + both ears), not a single flat
    # triangle: nose-eye and eye-eye edges give the depth/rotation cue
    # even when both earbases are untracked (the common case on this
    # rig), since eyes are tracked almost as reliably as nose itself.
    # eye-earbase edges connect the two rings into one cage whenever ears
    # ARE available.
    ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "right_eye"),
    ("left_eye", "left_earbase"), ("right_eye", "right_earbase"),
    ("left_earbase", "right_earbase"),
    # A single line from a SYNTHETIC "_head_anchor" point to a SYNTHETIC
    # "_body_anchor" point, not a static real-bodypart-to-real-bodypart
    # edge on either end. Two static ear-neck edges visually detach the
    # head whenever BOTH earbases dip below the visibility gate at once
    # (earbases aren't this rig's most reliably-tracked points) —
    # confirmed as the cause of a real "head comes off" artifact. A
    # single static nose-neck_base edge instead always stays connected
    # (nose tends to track better) but looks unnatural, cutting straight
    # through the middle of the face triangle rather than flowing from
    # the base of the skull. make_skeleton_video() computes "_head_anchor"
    # per frame with priority ear-midpoint > single ear > eye-midpoint >
    # single eye > nose, and "_body_anchor" with priority neck_base >
    # back_end — so the connection is anatomically correct whenever
    # possible (ear-to-neck) AND stays drawn as long as ANY head point
    # and ANY spine point are tracked, not one specific pair of them.
    ("_head_anchor", "_body_anchor"),
]


def _choose_head_tip(LL: "np.ndarray", bps: list, ll_gate: float) -> str:
    """
    Picks ONE of nose/lower_jaw as the head outline's single "pointy end,"
    based on which is empirically better tracked for THIS specific
    animal/session (fraction of frames clearing ll_gate) — not both.
    Drawing both created a real, confusing artifact on a rendered video
    (2026-08-28): two near-duplicate points at the front of the head made
    the cage's shape hard to read as a clean triangle/pentagon. Defaults
    to "nose" when both are equally tracked or lower_jaw isn't present at
    all — nose is this rig's best-tracked point overall and the
    anatomically conventional choice; lower_jaw only wins when the data
    genuinely supports it for that specific session.
    """
    if "lower_jaw" not in bps:
        return "nose"
    if "nose" not in bps:
        return "lower_jaw"
    nose_frac = float((LL[:, bps.index("nose")] >= ll_gate).mean())
    jaw_frac = float((LL[:, bps.index("lower_jaw")] >= ll_gate).mean())
    return "lower_jaw" if jaw_frac > nose_frac else "nose"


def _head_cage_edges(tip_bp: "str | None") -> list:
    """
    A clean fan/pentagon head outline around a SINGLE tip point (see
    _choose_head_tip — nose OR lower_jaw, never both): tip connects to
    each eye, eyes connect to each other and to their own-side earbase,
    earbases connect to each other. At most 5 vertices (tip + 2 eyes + 2
    earbases), 6 cage edges, no crossing lines — plus the tip-independent
    synthetic "_head_anchor"-to-"_body_anchor" connecting edge (see
    MOUSE_OUTLINE_HEAD_EDGES's own docstring comment for why that's
    synthetic rather than a static real-bodypart edge). tip_bp=None (no
    nose AND no lower_jaw tracked at all — should be rare, ~92%+ combined
    coverage on real data) omits the cage entirely, keeping only the
    anchor edge.
    """
    anchor_edge = [("_head_anchor", "_body_anchor")]
    if tip_bp is None:
        return anchor_edge
    return [
        (tip_bp, "left_eye"), (tip_bp, "right_eye"), ("left_eye", "right_eye"),
        ("left_eye", "left_earbase"), ("right_eye", "right_earbase"),
        ("left_earbase", "right_earbase"),
    ] + anchor_edge


MOUSE_OUTLINE_PAW_BODYPARTS = {
    "front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw",
}


def default_outline_smooth_map(reduced_bps: list) -> dict:
    """
    Default bodypart-specific render_smooth_frames map for the outline
    skeleton video: paws get none (they move fast during real strides —
    over-smoothing would blur genuine motion, not just jitter), nose/eyes/
    ears get a light touch, everything else (spine hubs, knees/thighs,
    the chronically jitterier points) gets a stronger median window. This
    split, and the median-not-mean choice in make_skeleton_video's
    render_smooth_frames itself, were validated on real data (2026-08-28)
    when the outline style became the pipeline's default skeleton-video
    output — see the render_smooth_frames docstring for why median beats
    a moving average here (robust to the occasional large single-camera
    misdetection jump, rather than smearing it across more frames).
    """
    m = {}
    for bp in reduced_bps:
        if bp in MOUSE_OUTLINE_PAW_BODYPARTS:
            m[bp] = 2
        elif bp in ("nose", "left_eye", "right_eye", "left_earbase", "right_earbase"):
            m[bp] = 5
        else:
            m[bp] = 8
    return m


def select_confident_outline_bodyparts(h5_3d: Path, ll_gate: float = 0.6,
                                        min_confident_frac: float = 0.5,
                                        base_bodyparts: "list | None" = None,
                                        df: "object | None" = None) -> list:
    """
    MOUSE_OUTLINE_BODYPARTS (or base_bodyparts, if given) is a FLOOR, not a
    ceiling — this adds any other bodypart in the H5 whose fraction of
    frames clearing ll_gate is >= min_confident_frac, so a session with
    unusually good tracking on e.g. an eye or a thigh shows it, while a
    session that doesn't still gets the guaranteed minimum outline. Returns
    bodyparts in the H5's own column order (not alphabetical/insertion
    order) so build_outline_skeleton_edges' chain-collapse sees them in
    real anatomical chain order.

    df: optional pre-loaded pd.read_hdf(h5_3d) DataFrame — pass this when
    the caller has already read the H5 for another purpose (e.g. to get
    the full bodypart list for build_outline_skeleton_edges' all_bps),
    so this function doesn't re-parse the same HDF5 file a second time.
    When given, h5_3d itself is not read (only used as a path if df is None).
    """
    base = list(base_bodyparts) if base_bodyparts is not None else list(MOUSE_OUTLINE_BODYPARTS)
    if df is None:
        import pandas as pd
        df = pd.read_hdf(str(h5_3d))
    scorer = df.columns.get_level_values(0)[0]
    bps = df[scorer].columns.get_level_values(0).unique().tolist()
    LL = df[scorer].xs("likelihood", level="coords", axis=1).values

    # Single head-tip choice (nose OR lower_jaw, see _choose_head_tip) —
    # swap "nose" in the base set for the winner if it isn't nose.
    tip = _choose_head_tip(LL, bps, ll_gate)
    if tip != "nose" and "nose" in base:
        base = [tip if bp == "nose" else bp for bp in base]

    base_set = set(base)
    # Excluded from the extras pool regardless of their own confidence:
    # (1) "*_earend" — a leaf point whose ONLY possible skeleton
    # connection is its earbase, and earbases are among this rig's least
    # reliable points (11-22% of frames vs. 60%+ for earend itself), so
    # adding it renders it visibly floating, disconnected from the rest
    # of the head, whenever its earbase isn't also tracked. (2) the
    # LOSING head-tip candidate ("upper_jaw" too, plus whichever of
    # nose/lower_jaw _choose_head_tip did NOT pick) — adding it back as
    # an "extra" would reintroduce the exact two-near-duplicate-points-
    # at-the-front clutter _choose_head_tip exists to avoid. Both
    # confirmed as real artifacts on a rendered video (2026-08-28). (3)
    # any "*_thai" (thigh) point — the base outline already connects a leg
    # as hub->knee->paw directly (skipping the thigh joint entirely, see
    # MOUSE_OUTLINE_BODYPARTS); letting a single confidently-tracked thigh
    # sneak back in as an "extra" reintroduces the intermediate joint on
    # ONE side only (real data: e.g. front_right_thai confident enough to
    # add while front_left_thai isn't), producing a visibly LOPSIDED
    # skeleton -- one leg with an extra segment, the other without.
    # Confirmed as a real artifact on rendered TMZ7 data (2026-09-09),
    # reported directly as "use the knee if possible, not the thigh."
    losing_tip = "lower_jaw" if tip == "nose" else "nose"
    always_excluded = {losing_tip, "upper_jaw", "tail_base", "tailbase"}
    extra = {bp for i, bp in enumerate(bps)
             if bp not in base_set and bp not in always_excluded
             and not bp.lower().endswith("earend")
             and not bp.lower().endswith("thai")
             and float((LL[:, i] >= ll_gate).mean()) >= min_confident_frac}
    return [bp for bp in bps if bp in base_set or bp in extra]


# Moved to cube_core.py (as collapse_skeleton_graph) 2026-09-10 so
# _compute_cluster_posture_data's cluster-trajectory plots can reuse the
# exact same correct collapse this module's build_outline_skeleton_edges()
# already relies on. Re-exported here under the original private name so
# every existing call site in this module keeps working unchanged.
from cube_core import collapse_skeleton_graph as _collapse_skeleton_graph


def build_outline_skeleton_edges(reduced_bps: list, all_bps: "list | None" = None,
                                  logger=None) -> list:
    """
    Combines a tip-parametrized head cage (_head_cage_edges — always
    governs the nose-or-lower-jaw/eye/ear/anchor region; see
    _choose_head_tip for why the tip is picked dynamically per session
    rather than hardcoded to "nose") with a graph-based collapse
    (_collapse_skeleton_graph) of the FULL anatomical skeleton (derived
    from every bodypart in the H5, not just reduced_bps) for everything
    else — spine/limb/tail, plus any extra high-confidence bodypart from
    select_confident_outline_bodyparts. This correctly reconnects e.g.
    back_end to back_left_knee straight through a dropped back_left_thai,
    instead of leaving it floating (see _collapse_skeleton_graph's
    docstring).

    all_bps: full bodypart list to build the underlying graph from before
    collapsing; defaults to reduced_bps itself (no collapsing possible)
    if not given — callers with the full H5 bodypart list available
    should always pass it.

    The graph collapse alone still isn't enough for the head: the tip is
    directly adjacent to neck_base in the real anatomical graph (not
    behind any excluded node), so collapsing can't remove that edge — the
    explicit head cage always overrides anything generic touching the
    head/neck nodes, replacing the straight-through-the-face-triangle
    line with a natural ear-to-neck attachment.
    """
    bp_set = set(reduced_bps)
    tip = "nose" if "nose" in bp_set else ("lower_jaw" if "lower_jaw" in bp_set else None)
    # "_head_anchor"/"_body_anchor" are synthetic points make_skeleton_video()
    # computes at render time — they never appear in the H5's real
    # bodypart list, so they must be allowed through this filter
    # explicitly rather than requiring literal membership in bp_set.
    head_filter_set = bp_set | {"_head_anchor", "_body_anchor"}
    head_edges = [(a, b) for a, b in _head_cage_edges(tip)
                  if a in head_filter_set and b in head_filter_set]

    full_bps = list(all_bps) if all_bps is not None else list(reduced_bps)
    full_edges = _infer_skeleton(full_bps, logger=logger)
    generic_edges = _collapse_skeleton_graph(full_edges, bp_set)

    head_nodes = {"nose", "lower_jaw", "left_eye", "right_eye",
                  "left_earbase", "right_earbase", "neck_base"}
    generic_edges = [(a, b) for a, b in generic_edges
                      if not (a in head_nodes and b in head_nodes)]
    seen = set()
    edges = []
    for a, b in head_edges + generic_edges:
        key = frozenset((a, b))
        if key not in seen:
            seen.add(key)
            edges.append((a, b))
    return edges


# Reusable, schema-agnostic targets for the two-link geometric prior (see
# _apply_two_link_spine_prior below). Each entry is (mid_bp, end_a_bp,
# end_b_bp); _run_3d_dlc_step applies _apply_two_link_spine_prior once per
# tuple here when dlc_3d_spine_geometric_prior is on.
#
# Validated on real data (2026-08-27, F:\3D CAM TEST\TRY 3, session
# 439158_3_48C_0000) after back_middle's own exclusion above: each candidate
# was first checked for real per-camera feasibility (>=2 cameras clearing
# ll_gate — 14-56% for these, vs. back_middle's 1.4%, i.e. a materially
# different tier), then run end-to-end through this exact function on the
# real triangulated+bone-constrained output. All 6 improved substantially
# over the raw zero-fill-collapse baseline and landed in the same range as
# real, well-tracked bodyparts' own natural frame-to-frame jitter (for
# context, neck_base/back_end/nose/tail_base: median 0.46-1.4mm, mean
# 1.3-2.3mm, p95 5.7-8.0mm):
#   right_earbase  (A=right_eye, C=right_earend):  median 0.16, mean 1.81, p95 5.6mm
#   left_earbase   (A=left_eye,  C=left_earend):   median 0.00, mean 0.90, p95 2.9mm
#   back_left_thai (A=back_end,  C=back_left_knee): median 0.46, mean 6.09, p95 16.7mm
#   back_right_thai(A=back_end,  C=back_right_knee):median 0.48, mean 4.50, p95 12.3mm
#   front_left_knee (A=front_left_thai, C=front_left_paw):  median 0.00, mean 3.68, p95 8.1mm
#   front_right_knee(A=front_right_thai,C=front_right_paw): median 0.00, mean 3.75, p95 8.4mm
# (back_left_thai/back_right_thai run somewhat higher on mean/p95 than the
# others — still same order of magnitude as natural jitter, not a red flag,
# but the weakest of the six.) This validation only used ONE session; see
# CLAUDE.md's Crash Protection entry for full methodology, including the two
# bugs found and fixed during this validation (degenerate-frame direction
# disambiguation; momentary-bad-geometry carry-forward) — both covered by
# regression tests in test_spine_geometric_prior.py.
SPINE_GEOMETRIC_PRIOR_TARGETS: list[tuple[str, str, str]] = [
    ("right_earbase", "right_eye", "right_earend"),
    ("left_earbase", "left_eye", "left_earend"),
    ("back_left_thai", "back_end", "back_left_knee"),
    ("back_right_thai", "back_end", "back_right_knee"),
    ("front_left_knee", "front_left_thai", "front_left_paw"),
    ("front_right_knee", "front_right_thai", "front_right_paw"),
]


def _infer_skeleton(bps: list[str], logger=None) -> list[tuple[str, str]]:
    """
    Build skeleton connections from DLC-style body-part names.
    Falls back to sequential nearest-neighbour if names are not recognised.

    Thin wrapper around cube_core.infer_skeleton() (Two-Camera fusion Phase 4
    extraction, 2026-09-07) -- identical logic, kept as a same-named function
    here so every existing caller in this file is unaffected.
    """
    return _infer_skeleton_core(bps, logger=logger)


@functools.lru_cache(maxsize=1)
def _nvenc_available(ffmpeg_exe: str) -> bool:
    """One-time, cached probe for a working h264_nvenc encoder."""
    try:
        probe = subprocess.run(
            [ffmpeg_exe, "-y", "-f", "lavfi", "-i", "color=black:s=64x64:d=0.1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=15)
        return probe.returncode == 0
    except Exception:
        return False


def _drain_stream(stream, sink: list) -> None:
    """Continuously read a subprocess pipe so it never blocks the writer."""
    try:
        sink.append(stream.read())
    except Exception:
        pass


def _interpolate_short_gaps(XYZ: np.ndarray, valid: np.ndarray, max_gap: int):
    """
    Linearly interpolate each bodypart's x/y/z across runs of invalid
    (sub-gate) frames no longer than max_gap, so a brief dropout shows a
    smooth in-between position instead of vanishing or snapping to a
    noisy raw triangulation.  Gaps longer than max_gap, and gaps touching
    either end of the recording, are left unfilled (limit_area="inside").

    Returns (XYZ_filled, filled_mask) — filled_mask is True wherever a
    bodypart is either originally valid or was interpolated.
    """
    import pandas as pd
    n_bp = XYZ.shape[1]
    XYZ_filled = XYZ.copy()
    filled_mask = valid.copy()
    for b in range(n_bp):
        col = XYZ[:, b, :].copy()
        col[~valid[:, b]] = np.nan
        s = pd.DataFrame(col).interpolate(
            method="linear", limit=max_gap, limit_area="inside")
        filled_mask[:, b] = valid[:, b] | s.notna().all(axis=1).values
        XYZ_filled[:, b, :] = s.values
    nan_left = np.isnan(XYZ_filled)
    XYZ_filled[nan_left] = XYZ[nan_left]
    return XYZ_filled, filled_mask


HEAD_RIGID_BODYPART_CANDIDATES = [
    "nose", "upper_jaw", "lower_jaw", "left_eye", "right_eye",
    "left_earbase", "right_earbase", "left_earend", "right_earend",
]


def _mds_canonical_shape(cand_idx: np.ndarray, XYZ_DRAW: np.ndarray,
                          VISIBLE: np.ndarray, min_co_visible: int = 20) -> "np.ndarray | None":
    """
    Builds a canonical rigid-head shape from ROBUST (median) pairwise
    distances across ALL frames, via classical multidimensional scaling
    (MDS) — not a single reference frame's raw geometry.

    A single-reference-frame canonical shape (the original approach) has
    no way to know whether its one sample was a good one: if that frame
    happened to have noisy triangulation on any one point (e.g. nose and
    left_eye landing unusually close together), that flaw gets baked in
    and repeats IDENTICALLY in every other frame's rendered shape for the
    rest of the video. Confirmed as the real cause of a user-reported
    artifact (2026-08-28): one head connection consistently looked wrong
    "throughout the video," not intermittently, which only makes sense if
    every frame was inheriting the same single bad sample rather than
    each frame having independently-insufficient data.

    Uses the median (not mean) observed distance between each pair of
    candidates across every frame where BOTH are visible — the same
    "learn it from the session's own real motion, use a robust statistic"
    pattern already established elsewhere in this codebase (e.g. the
    two-link spine geometric prior's rod-length learning). Classical MDS
    (double-centered squared-distance matrix, top-3 eigenvectors) then
    reconstructs a 3D point configuration whose pairwise distances match
    those robust medians as closely as possible — a genuine consensus
    shape, not any single frame's.

    Returns None (caller should fall back to the single-reference-frame
    approach) if any pair of candidates never has >=min_co_visible frames
    where both are tracked — MDS needs a complete distance matrix.
    """
    n = len(cand_idx)
    D = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            both = VISIBLE[:, cand_idx[i]] & VISIBLE[:, cand_idx[j]]
            if int(both.sum()) < min_co_visible:
                continue
            d = np.linalg.norm(
                XYZ_DRAW[both][:, cand_idx[i], :] - XYZ_DRAW[both][:, cand_idx[j], :], axis=1)
            D[i, j] = D[j, i] = float(np.median(d))
    np.fill_diagonal(D, 0.0)
    if np.isnan(D).any():
        return None

    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1][:3]
    eigvals = np.clip(eigvals[order], 0, None)
    return eigvecs[:, order] * np.sqrt(eigvals)  # (n, 3)


def _rigidify_head_points(bps: list, XYZ_DRAW: np.ndarray, VISIBLE: np.ndarray,
                           render_mask: np.ndarray, logger=None) -> None:
    """
    Enforces a rigid head shape for display: a mouse's skull does not
    change shape frame to frame, only its position/orientation does —
    but independent per-bodypart triangulation and (for the outline
    style) independent per-bodypart median smoothing can otherwise make
    the rendered head cage visibly warp between frames. Confirmed as a
    real, confusing artifact on a rendered video ("the shape of the head
    is impossible to discern... the head cannot alter shape"). Mutates
    XYZ_DRAW IN PLACE for whichever HEAD_RIGID_BODYPART_CANDIDATES are
    both present in bps and currently drawn (render_mask'd) — never
    touches the underlying H5 or any other consumer of it, purely a
    display correction, same convention as render_smooth_frames.

    Method: pick a canonical rest shape from the single real frame with
    the most head-candidate points simultaneously visible (an actual
    observed geometry, not an average that could itself be distorted).
    For every frame with >=3 visible head-candidate points (the minimum
    needed to determine a unique 3D rotation), solve the best-fit rigid
    rotation+translation (Kabsch algorithm via SVD) mapping the
    canonical shape's corresponding points onto that frame's observed
    points.

    That transform is then applied to draw ALL head candidates that
    frame, not just whichever ones happened to be independently tracked
    — VISIBLE is also set True for the whole candidate set on any frame
    the pose could be solved. Earbases in particular are tracked only
    11-22% of the time on this rig, so requiring the specific eye+earbase
    pairs the head cage's edges need to ALL be independently visible at
    once meant a full, recognizable head almost never actually got drawn
    — confirmed as a real "the shape is unrecognizable, this is a video-
    wide issue" artifact from inspecting real rendered frames (2026-08-28).
    Treating the head as a known rigid object and using whatever 3+
    points ARE currently trackable to localize+orient the WHOLE shape
    (a standard technique for rigid marker-cluster tracking) makes a
    complete, clean cage appear on far more frames — any 3 of 5 points,
    not one specific combination of them.

    Frames with fewer than 3 visible candidates are left entirely
    unmodified (no inference possible, VISIBLE unchanged) — graceful
    degradation, not a hard failure. Skipped entirely if fewer than 3
    candidates are ever drawn at all, or if no single frame ever has 3
    simultaneously visible.
    """
    idx_map = {bp: i for i, bp in enumerate(bps)}
    candidates = [bp for bp in HEAD_RIGID_BODYPART_CANDIDATES
                  if bp in idx_map and render_mask[idx_map[bp]]]
    if len(candidates) < 3:
        return
    cand_idx = np.array([idx_map[bp] for bp in candidates])
    cand_vis = VISIBLE[:, cand_idx]  # (N_frames, n_cand)

    n_visible_per_frame = cand_vis.sum(axis=1)

    mds_shape = _mds_canonical_shape(cand_idx, XYZ_DRAW, VISIBLE)
    if mds_shape is not None:
        canonical_centered = mds_shape - mds_shape.mean(axis=0)
    else:
        # Fallback: not enough co-visible frames for every candidate pair
        # to build a full MDS distance matrix — use the single frame with
        # the most simultaneously-visible candidates instead. Less robust
        # (see _mds_canonical_shape's docstring) but still functional.
        ref_frame = int(np.argmax(n_visible_per_frame))
        if n_visible_per_frame[ref_frame] < 3:
            if logger is not None:
                logger.warn("  Skeleton video: no frame has >=3 simultaneously "
                            "visible head points — skipping rigid head-shape constraint")
            return
        canonical = XYZ_DRAW[ref_frame][cand_idx, :].copy()  # (n_cand, 3)
        canonical_centered = canonical - canonical.mean(axis=0)

    n_frames_ = XYZ_DRAW.shape[0]
    for f in range(n_frames_):
        vis_mask = cand_vis[f]
        if vis_mask.sum() < 3:
            continue
        obs = XYZ_DRAW[f][cand_idx[vis_mask], :]
        obs_center = obs.mean(axis=0)
        obs_centered = obs - obs_center
        canon_sub = canonical_centered[vis_mask]
        H = canon_sub.T @ obs_centered
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        rigid_pts = (R @ canonical_centered.T).T + obs_center  # (n_cand, 3) — ALL candidates
        XYZ_DRAW[f, cand_idx, :] = rigid_pts
        VISIBLE[f, cand_idx] = True


def make_skeleton_video(h5_3d: Path, out_path: Path, fps: float,
                         ffmpeg_exe: str, logger,
                         subsample: int = 1, ll_gate: float = 0.0,
                         theme: str = "light",
                         visibility_hold_frames: int = 5,
                         interp_max_gap_frames: int = 3,
                         render_bodyparts: "list | None" = None,
                         render_edges: "list | None" = None,
                         render_smooth_frames: "int | dict" = 0,
                         force_libx264: bool = False) -> None:
    """
    Render a 3D skeleton animation from a triangulated 3D H5 file and pipe
    frames to FFmpeg.  Uses matplotlib Agg backend (no display needed).

    subsample: render every Nth frame (default 1 → full output fps).
    ll_gate: likelihood below this hides a point/line segment (gate only,
    not a hue driver — reuses dlc_3d_ll_threshold when set, else 0.3;
    raised from the historical 0.1 default after real data showed 0.1 let
    through barely-above-gate points that were still wrong triangulations,
    not just noisy-but-correct ones).
    theme: "dark"/"light" — same engine_cfg.plot_theme convention every
    other cube_core.py diagnostic plot uses (via _apply_plot_theme()).
    visibility_hold_frames: a bodypart that clears ll_gate at least once
    within +/-N frames stays drawn across that whole window, instead of
    blinking on/off every frame a noisy likelihood dips just under the
    gate (measured on real data: ~93% of frames changed their visible
    bodypart set frame-to-frame at the default gate, which is what reads
    as jitter/"not a mouse" rather than a smooth animal shape).
    interp_max_gap_frames: runs of sub-gate frames up to this length get a
    linearly interpolated position (and count as visible) instead of
    either vanishing or falling back to their own noisy raw triangulation
    for the hold window above — position accuracy, not just cosmetic
    continuity. Longer gaps are left to visibility_hold_frames alone.
    render_bodyparts: optional list of bodypart names to draw (points +
    skeleton segments whose both endpoints are in the list); None (default)
    draws every bodypart in the H5, unchanged from prior behavior. Meant
    for a reduced "body outline" set (e.g. nose/ears/spine/tail/4 paws)
    that reads as a recognizable mouse shape without every SuperAnimal
    keypoint cluttering the view — the underlying H5 and every other
    consumer of it are untouched, this only restricts what THIS video draws.
    render_smooth_frames: half-width (in frames) of a display-only MEDIAN
    filter (`scipy.ndimage.median_filter`, window = 2*render_smooth_frames+1)
    applied to the drawn position purely to damp visual jitter in the
    rendered animation; 0 (default) = off, unchanged from prior behavior.
    A median filter, not a moving average: this pipeline's remaining
    jitter is less small continuous noise and more the occasional large
    single-camera misdetection jump (the "confidently wrong" outlier
    problem documented throughout this codebase's real-data investigations)
    — averaging blends a wrong position into its correct neighbors and
    smears the error across more frames, where a median can drop a single
    bad frame's position entirely. Accepts either an int (applied
    uniformly to every bodypart) or a {bodypart_name: half_width} dict for
    per-bodypart smoothing strength — a bodypart with genuinely fast real
    motion (a paw mid-stride) wants less/no smoothing than a chronically
    jittery weak one; a bodypart missing from the dict gets no smoothing.
    Applied AFTER gap-filling, to the array actually drawn — never written
    back to the H5 or seen by any other consumer, so it cannot bias
    reprojection error, jitter QC, or BSoid features; it only changes how
    this one video looks.
    render_edges: optional explicit list of (bp_a, bp_b) name pairs to use
    as the drawn skeleton instead of the anatomical-chain derivation —
    needed because a chain-collapse (dropping an intermediate joint and
    reconnecting straight through) only works within a single
    SUPERANIMAL_QUADRUPED_CHAINS sub-list; nose->eye->earbase spans two
    separate sub-lists, so no collapse gives a real nose-earbase-earbase
    face triangle. See MOUSE_OUTLINE_BODYPARTS/build_outline_skeleton_edges()
    for the intended pairing. Ignored if render_bodyparts is None.
    force_libx264: skip the _nvenc_available() probe entirely and encode
    with libx264 — used by make_skeleton_video_isolated()'s retry so a
    persistent NVENC issue gets a materially different attempt on retry
    instead of repeating the same failure twice.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import pandas as pd
    import cube_core

    cube_core._apply_plot_theme(theme)
    bg_col, text_col, tick_col = (
        cube_core._BG, cube_core._TEXT_COL, cube_core._TICK_COL)
    # Floor grid gets its own theme colour (distinct from tick_col, which is
    # a pale, low-alpha colour in dark theme and barely visible against the
    # near-black background) -- light grey in light theme, a clearly
    # visible accent colour in dark theme. Shared with the cluster-
    # trajectory plots' own floor grid (cube_core._draw_trajectory_panel).
    floor_col = cube_core._FLOOR_COL

    df = pd.read_hdf(str(h5_3d))
    scorer = df.columns.get_level_values(0)[0]
    bps = df[scorer].columns.get_level_values(0).unique().tolist()

    try:
        X  = df[scorer].xs("x",          level="coords", axis=1).values
        Y  = df[scorer].xs("y",          level="coords", axis=1).values
        Z  = df[scorer].xs("z",          level="coords", axis=1).values
        LL = df[scorer].xs("likelihood", level="coords", axis=1).values
    except KeyError as e:
        raise ValueError(f"H5 file lacks required coord column: {e}")

    XYZ = np.stack([X, Y, Z], axis=-1)   # (N_frames, N_bp, 3)
    SKELETON = _infer_skeleton(bps, logger=logger)

    from cube_core import bodypart_region_colors
    bp_colors = bodypart_region_colors(bps)
    ll_gate = float(ll_gate) if ll_gate else 0.4

    # Uses the actual configured ll_gate (not a separate hardcoded 0.3)
    # so the axis bounds are derived from the same points that will
    # actually be gated visible during rendering below — a caller who
    # raises ll_gate (e.g. dlc_3d_ll_threshold=0.6) would otherwise have
    # the view bounds silently shaped by points beneath that gate that
    # never actually get drawn.
    mask = LL >= ll_gate
    valid_pts = XYZ[mask.any(axis=1)].reshape(-1, 3)
    lo = np.nanpercentile(valid_pts, 2.5, axis=0)
    hi = np.nanpercentile(valid_pts, 97.5, axis=0)
    padding = (hi - lo) * 0.1 + 1e-6
    lo -= padding
    hi += padding
    # box_aspect is finalized below, once floor_z is known and lo[2] has
    # been aligned to it — this initial value is intentionally unused
    # past that point.

    # Floor-plane reference: without one, the 3D view gives no visual cue
    # for "the ground is here" the way the 2D camera views do (their
    # physical box floor is directly visible) — a mouse standing on all
    # four paws can look like it's floating in an abstract point cloud.
    # Estimated as the DENSITY MODE of tracked paw Z across the whole
    # session, not a percentile cutoff: the floor height is a real
    # physical constant, and mice essentially always have at least one
    # foot planted on it (confirmed on real data, 2026-08-28: 95%+ of
    # frames with any visible paw have one within 10mm of this estimate)
    # — so stance-phase paw positions cluster tightly at the true floor
    # height, while swing-phase positions spread out over a wider range
    # above it. A low percentile (the original approach) is an arbitrary
    # cutoff biased toward whatever noise sits below that stance cluster;
    # the histogram mode finds where the density actually concentrates,
    # which is the physically meaningful floor estimate — confirmed on
    # real data to differ from the 5th-percentile estimate by ~4mm, with
    # a clear, genuinely unimodal peak (not a sharp outlier spike) at the
    # mode. Matched by "paw" appearing in the bodypart name so this works
    # for both the reduced outline set and the full 24-bodypart video.
    # Falls back to the overall lower Z bound if no paw-named bodypart is
    # present, or to the plain low-percentile estimate if there are too
    # few valid samples for a histogram mode to be meaningful.
    #
    # Prefers HIND paws specifically when both are present: a mouse's hind
    # feet almost never leave the ground (weight-bearing stance is their
    # default state; only the front paws routinely lift for reaching/
    # rearing), so hind-paw Z is a cleaner, less-diluted floor reference
    # than pooling front+hind together, which mixes in a meaningful amount
    # of genuinely-elevated front-paw data.
    paw_idx = [i for i, bp in enumerate(bps) if "paw" in bp.lower()]
    hind_paw_idx = [i for i in paw_idx if bps[i].lower().startswith("back")]
    if hind_paw_idx:
        paw_idx = hind_paw_idx
    if paw_idx:
        paw_z = Z[:, paw_idx]
        paw_valid = LL[:, paw_idx] >= ll_gate
        # Shared with _compute_cluster_posture_data() (cube_core.py) so the
        # skeleton video and the cluster-trajectory plots agree on exactly
        # the same real-world floor height.
        _est = cube_core.estimate_floor_z_from_paws(paw_z[paw_valid])
        floor_z = float(lo[2]) if _est is None else _est
    else:
        floor_z = float(lo[2])

    # Align the z-axis lower bound to the floor plane, not the generic
    # per-axis percentile above: lo[2] was computed from ALL bodyparts'
    # Z pooled together (head, spine, ears — mostly elevated well above
    # paw height), so it typically sits above floor_z, leaving empty dead
    # space below the floor plane instead of the floor sitting flush at
    # the bottom of the visible box the way a real cage floor would.
    # Pinned to EXACTLY floor_z, no padding below it: a mouse's paws
    # don't go underground, so showing negative-height empty space below
    # the floor is wasted view real estate — confirmed as unnecessary by
    # the user directly. A rare below-floor triangulation-noise outlier
    # (if lo[2] was already lower than floor_z from the generic
    # percentile) is now visually clipped at the floor instead of
    # expanding the view to accommodate it — an intentional trade in
    # favor of a tight, legible ground-referenced view over showing every
    # possible outlier.
    lo[2] = floor_z
    box_aspect = np.maximum(hi - lo, 1e-6)

    # Shift the whole scene so the floor sits exactly at Z=0 — makes
    # "grounded" legible straight from the axis itself (the "0" tick IS
    # the floor), which is safer than a filled translucent floor plane
    # (mplot3d's Poly3DCollection, tried earlier, reliably crashed this
    # render loop natively — see the BLAS-thread-contention Crash
    # Protection entry) and was the user's own preferred alternative.
    # Applied to the raw XYZ (not XYZ_DRAW, which doesn't exist yet at
    # this point) so gap-filling/smoothing/rigidification/anchor-
    # synthesis below all naturally inherit the shifted frame — a
    # uniform translation commutes through every one of those (each is
    # either linear/averaging or a robust statistic on differences).
    XYZ[:, :, 2] -= floor_z
    lo[2] -= floor_z
    hi[2] -= floor_z
    floor_z = 0.0

    raw_valid = LL >= ll_gate
    if interp_max_gap_frames > 0:
        XYZ_DRAW, gap_filled = _interpolate_short_gaps(
            XYZ, raw_valid, interp_max_gap_frames)
    else:
        XYZ_DRAW, gap_filled = XYZ, raw_valid

    if visibility_hold_frames > 0:
        from scipy.ndimage import maximum_filter1d
        VISIBLE = maximum_filter1d(
            gap_filled.astype(np.uint8),
            size=2 * visibility_hold_frames + 1, axis=0, mode="nearest") > 0
    else:
        VISIBLE = gap_filled

    # Display-only smoothing: damps rendered jitter without touching the
    # underlying data (XYZ, the H5, or anything else derived from it) —
    # VISIBLE above is computed from the raw/gap-filled data, not this.
    # Median, not mean: robust to the occasional large single-camera
    # misdetection jump (a mean filter would just smear that error across
    # more frames instead of removing it). Per-bodypart strength via a
    # {bodypart: half_width} dict, or one half_width applied to every
    # bodypart via a plain int.
    if render_smooth_frames:
        from scipy.ndimage import median_filter
        smooth_map = (render_smooth_frames if isinstance(render_smooth_frames, dict)
                      else {bp: render_smooth_frames for bp in bps})
        for bi, bp in enumerate(bps):
            half = smooth_map.get(bp, 0)
            if half and half > 0:
                k = 2 * int(half) + 1
                XYZ_DRAW[:, bi, :] = median_filter(
                    XYZ_DRAW[:, bi, :], size=(k, 1), mode="nearest")

    # Dual hind-limb floor grounding: ensure both hind limbs never leave
    # the floor simultaneously for longer than 0.2 seconds.
    # In quadrupeds, normal locomotion maintains ground contact with at least
    # one hind foot. A flight phase where BOTH hind paws are airborne only
    # occurs during rare vertical jumps and lasts < 0.15s. Simultaneous airborne
    # excursions lasting > 0.2s represent triangulation vertical drift.
    # For every contiguous run exceeding 0.2s where both hind paws are elevated
    # (> 2.0mm above the shifted floor Z=0), we ground the lower hind limb to
    # the floor and shift associated limb joints by Delta Z = Z_lower, strictly
    # preserving anatomical bone lengths without limb distortion.
    _hl_paw_bps = ["back_left_paw", "left_hindpaw", "left_hindleg3", "back_left_foot"]
    _hr_paw_bps = ["back_right_paw", "right_hindpaw", "right_hindleg3", "back_right_foot"]
    _hl_knee_bps = ["back_left_knee", "left_hindleg2", "back_left_thai"]
    _hr_knee_bps = ["back_right_knee", "right_hindleg2", "back_right_thai"]

    _idx_hl_paw = next((i for i, b in enumerate(bps) if b in _hl_paw_bps or (b.lower().startswith("back_l") and "paw" in b.lower())), None)
    _idx_hr_paw = next((i for i, b in enumerate(bps) if b in _hr_paw_bps or (b.lower().startswith("back_r") and "paw" in b.lower())), None)
    _idx_hl_knee = next((i for i, b in enumerate(bps) if b in _hl_knee_bps or (b.lower().startswith("back_l") and "knee" in b.lower())), None)
    _idx_hr_knee = next((i for i, b in enumerate(bps) if b in _hr_knee_bps or (b.lower().startswith("back_r") and "knee" in b.lower())), None)

    if _idx_hl_paw is not None and _idx_hr_paw is not None:
        _dual_lift_margin = 2.0  # mm above floor
        _max_dual_lift_frames = max(1, int(round(0.2 * fps)))
        _z_hl = XYZ_DRAW[:, _idx_hl_paw, 2]
        _z_hr = XYZ_DRAW[:, _idx_hr_paw, 2]
        _both_lifted = (VISIBLE[:, _idx_hl_paw] & VISIBLE[:, _idx_hr_paw] &
                        (_z_hl > _dual_lift_margin) & (_z_hr > _dual_lift_margin))
        _edges = np.diff(np.r_[0, _both_lifted.astype(np.int8), 0])
        _starts = np.flatnonzero(_edges == 1)
        _ends = np.flatnonzero(_edges == -1)
        for _s, _e in zip(_starts, _ends):
            if (_e - _s) <= _max_dual_lift_frames:
                continue
            for _f in range(_s, _e):
                _dz = min(XYZ_DRAW[_f, _idx_hl_paw, 2], XYZ_DRAW[_f, _idx_hr_paw, 2])
                if _dz > 0:
                    XYZ_DRAW[_f, _idx_hl_paw, 2] -= _dz
                    XYZ_DRAW[_f, _idx_hr_paw, 2] -= _dz
                    if _idx_hl_knee is not None:
                        XYZ_DRAW[_f, _idx_hl_knee, 2] -= _dz
                    if _idx_hr_knee is not None:
                        XYZ_DRAW[_f, _idx_hr_knee, 2] -= _dz

    # Reduced "body outline" set: restrict which bodyparts get drawn, without
    # touching XYZ/VISIBLE themselves (indexing by the full bps list stays
    # valid throughout — only SKELETON and render_mask change).
    #
    # The skeleton chain is re-derived from the REDUCED bodypart list rather
    # than post-filtering the full-skeleton edges: SUPERANIMAL_QUADRUPED_CHAINS
    # connects a leg as neck_base -> thigh -> knee -> paw, so simply dropping
    # edges whose endpoints aren't both in the reduced set would strand every
    # limb (no edge survives with an intermediate joint removed) — the whole
    # leg would vanish instead of collapsing into one line from body to paw.
    # Passing the reduced list straight into _infer_skeleton() lets it do the
    # same "connect consecutive PRESENT points in the anatomical chain" logic
    # it already does for missing bodyparts, so an omitted thigh/knee just
    # collapses into a direct neck_base-to-paw segment — still anatomically
    # ordered (nose->head->spine->tail, legs off the two spine hubs), just
    # with fewer intermediate vertices, which is exactly a body outline.
    if render_bodyparts is not None:
        render_set = set(render_bodyparts)
        if render_edges is not None:
            # "_head_anchor"/"_body_anchor" are synthetic points synthesized
            # below, not real H5 bodyparts — allow them through this filter
            # explicitly rather than requiring literal membership in bps.
            bp_set = set(bps) | {"_head_anchor", "_body_anchor"}
            SKELETON = [(a, b) for a, b in render_edges if a in bp_set and b in bp_set]
        else:
            reduced_bps = [bp for bp in bps if bp in render_set]
            SKELETON = _infer_skeleton(reduced_bps, logger=logger)
        render_mask = np.array([bp in render_set for bp in bps], dtype=bool)
    else:
        render_mask = np.ones(len(bps), dtype=bool)

    # Rigid head-shape constraint — must run BEFORE the anchor synthesis
    # below, so "_head_anchor" (computed from these same points' XYZ_DRAW
    # values) inherits the shape-corrected positions too, not the raw
    # independently-noisy ones.
    _rigidify_head_points(bps, XYZ_DRAW, VISIBLE, render_mask, logger=logger)

    # Synthetic priority-fallback anchor points: "_head_anchor" (see
    # MOUSE_OUTLINE_HEAD_EDGES) picks the best available head-region
    # point per frame — ear midpoint > eye midpoint > single
    # ear/eye/nose, in priority order — and "_body_anchor" similarly
    # falls back from neck_base to back_end. This makes the head-to-body
    # connection survive as long as ANY point on each side is tracked,
    # not one specific hardcoded pair, while still preferring the
    # anatomically natural attachment (ear-to-neck) whenever it's
    # available. Never scattered as their own dots (render_mask stays
    # False for both) — they exist only to give a connecting edge a
    # stable, dynamically-chosen endpoint.
    def _synthesize_anchor(name, tiers):
        """
        tiers: ordered list where each entry is either a bodypart name
        (str, single-point fallback) or a 2-tuple of names (averaged when
        BOTH are visible). Processed STRICTLY in list order — each tier
        only fills frames not already resolved by an earlier (higher-
        priority) tier — so the priority is exactly the order given, not
        "all pairs before any single" (an earlier version processed every
        pair first regardless of tier position, so e.g. the eye-midpoint
        pair could preempt a single tracked ear even though a single ear
        is supposed to outrank the eye pair — fixed by this strict
        per-tier ordering).
        """
        nonlocal bps, XYZ_DRAW, VISIBLE, render_mask, bp_colors
        all_names = [n for t in tiers for n in (t if isinstance(t, tuple) else (t,))]
        sources = [bp for bp in dict.fromkeys(all_names) if bp in bps]
        if not sources:
            return False
        src_idx = {bp: bps.index(bp) for bp in sources}
        n_frames_ = XYZ_DRAW.shape[0]
        anchor_xyz = np.zeros((n_frames_, 3), dtype=np.float64)
        anchor_vis = np.zeros(n_frames_, dtype=bool)
        for t in tiers:
            if isinstance(t, tuple):
                pa, pb = t
                if pa in src_idx and pb in src_idx:
                    ia, ib = src_idx[pa], src_idx[pb]
                    both = VISIBLE[:, ia] & VISIBLE[:, ib] & ~anchor_vis
                    anchor_xyz[both] = (XYZ_DRAW[both, ia, :] + XYZ_DRAW[both, ib, :]) / 2.0
                    anchor_vis |= both
            else:
                if t in src_idx:
                    i = src_idx[t]
                    need = VISIBLE[:, i] & ~anchor_vis
                    anchor_xyz[need] = XYZ_DRAW[need, i, :]
                    anchor_vis |= VISIBLE[:, i]
        bps = bps + [name]
        XYZ_DRAW = np.concatenate([XYZ_DRAW, anchor_xyz[:, None, :]], axis=1)
        VISIBLE = np.concatenate([VISIBLE, anchor_vis[:, None]], axis=1)
        render_mask = np.concatenate([render_mask, [False]])
        bp_colors = dict(bp_colors)
        bp_colors[name] = bp_colors.get("neck_base", (0.5, 0.5, 0.5))
        return True

    if any("_head_anchor" in (a, b) for a, b in SKELETON):
        if not _synthesize_anchor("_head_anchor", tiers=[
                ("left_earbase", "right_earbase"),
                "left_earbase", "right_earbase",
                ("left_eye", "right_eye"),
                "left_eye", "right_eye",
                "nose", "lower_jaw",
        ]):
            SKELETON = [(a, b) for a, b in SKELETON if "_head_anchor" not in (a, b)]

    if any("_body_anchor" in (a, b) for a, b in SKELETON):
        if not _synthesize_anchor("_body_anchor", tiers=["neck_base", "back_end"]):
            SKELETON = [(a, b) for a, b in SKELETON if "_body_anchor" not in (a, b)]

    # Per-frame limb-hub-to-paw bridging: if a mid-joint (thigh/knee, or a
    # mouse-schema forelegN/hindlegN segment) drops below ll_gate for a
    # handful of frames while the body hub and the paw both stay tracked,
    # draw a direct hub-to-paw line for those frames instead of leaving the
    # limb broken into disconnected dangling segments.
    #
    # The mid-joint sequence is derived from whatever path ACTUALLY exists
    # between hub and paw in the current SKELETON graph, not assumed to be
    # the full anatomical chain — the outline-style render (the CUBE
    # production default) already statically drops the thigh joint from
    # MOUSE_OUTLINE_BODYPARTS, collapsing "neck_base -> knee" into a single
    # edge with no thigh node in the graph at all, so requiring a fixed
    # hub-thigh-knee-paw chain to already exist (an earlier version of this
    # fix) never matched and the bridge never fired in the video CUBE
    # actually ships. Walking the graph instead correctly finds mid_idxs=[knee] in
    # outline mode and mid_idxs=[thigh, knee] in the unrestricted full
    # video, so the same bridging logic below works for both.
    _adj: dict = {}
    for a, b in SKELETON:
        _adj.setdefault(a, set()).add(b)
        _adj.setdefault(b, set()).add(a)

    def _skeleton_path(start: str, goal: str) -> "list[str] | None":
        if start not in _adj or start == goal:
            return None
        from collections import deque
        q = deque([[start]])
        seen = {start}
        while q:
            path = q.popleft()
            for nxt in _adj.get(path[-1], ()):
                if nxt in seen:
                    continue
                if nxt == goal:
                    return path + [nxt]
                seen.add(nxt)
                q.append(path + [nxt])
        return None

    LIMB_ENDPOINTS = [
        ("neck_base", "front_left_paw"), ("neck_base", "front_right_paw"),
        ("back_end", "back_left_paw"), ("back_end", "back_right_paw"),
        ("neck", "left_foreleg3"), ("neck", "right_foreleg3"),
        ("tailbase", "left_hindleg3"), ("tailbase", "right_hindleg3"),
    ]
    limb_bridge_specs = []
    for hub_bp, paw_bp in LIMB_ENDPOINTS:
        if hub_bp not in bps or paw_bp not in bps:
            continue
        path = _skeleton_path(hub_bp, paw_bp)
        if not path or len(path) < 3:
            continue  # no path, or hub/paw already directly connected (nothing to bridge)
        mid_bps = path[1:-1]
        if not all(bp in bps for bp in mid_bps):
            continue  # a synthetic anchor (e.g. "_body_anchor") sits on this path — skip
        idxs = [bps.index(bp) for bp in mid_bps]
        limb_bridge_specs.append(
            (bps.index(hub_bp), np.array(idxs), bps.index(paw_bp), hub_bp, paw_bp))

    out_fps = fps / subsample
    # Bodypart-color legend always sits alongside the 3D view (a narrow
    # side panel, not a separate PNG) so a viewer can identify which
    # colored dot is which bodypart without cross-referencing
    # plot_bodypart_legend()'s separate image file. Only lists whatever
    # render_mask actually draws — the full 24-bodypart list for a normal
    # video, just the outline set for render_bodyparts modes — so the
    # legend never lists a color that never appears on screen.
    # Legend column keeps roughly the same absolute width as before (enough
    # for the longest bodypart name at this font size) while the overall
    # figure grows wider and the width ratio shifts further toward the 3D
    # panel — previously the legend's share (1.4 of 6.4 width units, ~22%)
    # visibly cramped the 3D plot into a small corner of the frame (reported
    # on real rendered output). The 3D panel now gets a materially larger
    # share of a materially wider canvas instead of just a larger fraction
    # of the same canvas, so the legend doesn't lose readable width either.
    fig = plt.figure(figsize=(9.0, 6), dpi=120, facecolor=bg_col)
    gs = fig.add_gridspec(1, 2, width_ratios=[7, 1.4], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0], projection="3d", facecolor=bg_col)
    ax.view_init(elev=25, azim=45)

    legend_ax = fig.add_subplot(gs[0, 1], facecolor=bg_col)
    legend_ax.axis("off")
    legend_bps = [bp for bp, keep in zip(bps, render_mask) if keep]
    for i, bp in enumerate(legend_bps):
        y = 1.0 - (i + 0.5) / max(len(legend_bps), 1)
        legend_ax.scatter([0.08], [y], s=40, c=[bp_colors[bp]], transform=legend_ax.transAxes)
        legend_ax.text(0.22, y, bp.replace("_", " "), transform=legend_ax.transAxes,
                        fontsize=7, color=text_col, va="center", ha="left")

    # Draw once up front so the encoder's frame size matches what
    # tostring_rgb() will actually emit (backend/DPI rounding can make this
    # differ from the nominal figsize*dpi, which used to desync the raw
    # video stream and crash FFmpeg on frame 1).
    fig.canvas.draw()
    canvas_w, canvas_h = fig.canvas.get_width_height()

    use_gpu = (not force_libx264) and _nvenc_available(ffmpeg_exe)
    enc_args = (["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "6M"] if use_gpu
                else ["-c:v", "libx264", "-crf", "20"])

    # Atomic: ffmpeg writes to a .tmp sibling, renamed onto out_path only on
    # success -- a crash/kill mid-render (this render loop's own native-crash
    # history is the reason make_skeleton_video_isolated()/retry exist) would
    # otherwise leave a truncated/corrupt file directly at out_path, which
    # every call site's `if not skel_path.exists()` skip check then treats
    # as a finished skeleton video forever.
    # tmp marker before the real suffix -- see _crop_camera_video's note.
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    cmd = [ffmpeg_exe, "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{canvas_w}x{canvas_h}", "-r", str(out_fps), "-i", "pipe:0",
           *enc_args, "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(tmp_path)]
    logger.info(f"  Creating skeleton video [{'h264_nvenc' if use_gpu else 'libx264'}]: {out_path.name}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # Drain stderr continuously in the background: FFmpeg can block writing
    # to its own stderr pipe if nothing reads it, which in turn blocks our
    # stdin writes below and deadlocks this thread forever.
    stderr_sink: list = []
    stderr_thread = threading.Thread(
        target=_drain_stream, args=(proc.stderr, stderr_sink), daemon=True)
    stderr_thread.start()

    # Cap BLAS (OpenBLAS/MKL, via threadpoolctl) to 1 thread for the whole
    # render loop below, same rationale/pattern as cube_core.py's
    # _blas_single_thread_for_dispatch() (used around joblib.Parallel
    # dispatches elsewhere in this codebase for the identical crash
    # category: intermittent, "not reliably reproducible on demand"
    # native crashes from BLAS/numba/joblib thread contention on Windows).
    # This function isn't a Parallel dispatch, so that context manager
    # doesn't apply directly, but the same underlying hazard applies: a
    # ~3800-frame render loop runs for minutes inside CUBE's long-lived
    # orchestrator process, alongside whatever else that process or its
    # other concurrent sessions are doing, and Windows Event Log evidence
    # (2026-08-28) showed multiple real crashes faulting in matplotlib's
    # _backend_agg.pyd at the SAME exact code offset across unrelated
    # runs spanning a full day — the signature of state corruption from
    # concurrent thread contention, not a bug in any specific draw call
    # this function makes. Using threadpoolctl's non-context-manager API
    # (enter here, restore in the existing `finally:` below) rather than
    # wrapping the whole try/except/finally in a `with` — avoids
    # reindenting the entire render loop for an equivalent effect.
    _blas_limiter = None
    try:
        from threadpoolctl import threadpool_limits
        _blas_limiter = threadpool_limits(limits=1)
    except ImportError:
        pass

    try:
        n_total = len(XYZ)
        for k in range(0, n_total, subsample):
            pts = XYZ_DRAW[k]
            # Edge visibility uses the real per-frame tracking state,
            # unmasked by render_mask: SKELETON already only contains
            # edges between bodyparts in the render set, so render_mask
            # is redundant there for a real bodypart, and actively wrong
            # for the synthetic "_head_anchor" point, whose render_mask
            # entry is deliberately False (see its construction above) so
            # it never gets its own scatter dot — ANDing that into the
            # edge check would make the head-to-neck line never draw at
            # all, regardless of whether the anchor is actually tracked
            # that frame. Confirmed as a real bug on real data (2026-08-28):
            # the head-to-neck connection never appeared once, not just
            # intermittently, until this was split into two separate masks.
            visible = VISIBLE[k]

            ax.cla()
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            # tuple(), not the raw array: Axes3D.set_box_aspect() mutates
            # (rescales) whatever array it's given IN PLACE -- confirmed
            # directly. Harmless here today only because lo/hi (used for
            # set_xlim/ylim/zlim above) are fixed, independently-computed
            # arrays never re-derived from box_aspect after this point, and
            # the mutation preserves the aspect RATIO (matplotlib only
            # cares about relative proportions) -- but passing a tuple
            # costs nothing and removes the latent risk entirely, same fix
            # applied to cube_core.py's cluster-trajectory panels after it
            # caused a real bug there (a shared, reused box_aspect array
            # got corrupted after the first of several panels that render
            # from it, silently blowing up every subsequent panel's axis
            # range).
            ax.set_box_aspect(tuple(box_aspect))
            ax.set_facecolor(bg_col)
            ax.tick_params(colors=tick_col)
            ax.xaxis.label.set_color(tick_col)
            ax.yaxis.label.set_color(tick_col)
            ax.zaxis.label.set_color(tick_col)
            ax.set_zlabel("height above floor (0 = floor)", fontsize=7)
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.set_facecolor(bg_col)
                pane.set_edgecolor(tick_col)
                pane.set_alpha(1.0)

            # Floor plane: the single biggest fix for "the mouse looks like
            # it's floating" — a ground reference at floor_z the way the
            # 2D camera views' physical box floor gives one. Drawn as a
            # plain line grid via ax.plot, NOT a filled mplot3d
            # Poly3DCollection: a Poly3DCollection has to be recreated
            # every single frame here (ax.cla() wipes it, same as every
            # other artist), and doing that ~3800 times/video reliably
            # crashed the render (STATUS_ACCESS_VIOLATION /
            # STATUS_ILLEGAL_INSTRUCTION — confirmed reproducible 3/3
            # real attempts once this loop is actually run end-to-end,
            # not a random transient glitch like some earlier unrelated
            # crashes this session). Matplotlib's mplot3d Poly3DCollection
            # has documented instability in exactly this repeated-
            # create/clear-per-frame animation pattern (internal
            # do_3d_projection/_facecolors2d caching). ax.plot has been
            # called thousands of times in this exact loop across this
            # session's testing with zero crashes, so it's the safe
            # primitive — this matters because this render loop is now
            # the CUBE production default (dlc_3d_export_skeleton_video),
            # not just a one-off comparison script.
            _floor_x = [lo[0], hi[0], hi[0], lo[0], lo[0]]
            _floor_y = [lo[1], lo[1], hi[1], hi[1], lo[1]]
            ax.plot(_floor_x, _floor_y, [floor_z] * 5, color=floor_col, linewidth=1.4, alpha=0.75)
            for _t in (1 / 3, 2 / 3):
                _gx = lo[0] + (hi[0] - lo[0]) * _t
                ax.plot([_gx, _gx], [lo[1], hi[1]], [floor_z, floor_z],
                        color=floor_col, linewidth=0.6, alpha=0.4)
                _gy = lo[1] + (hi[1] - lo[1]) * _t
                ax.plot([lo[0], hi[0]], [_gy, _gy], [floor_z, floor_z],
                        color=floor_col, linewidth=0.6, alpha=0.4)

            for bp_a, bp_b in SKELETON:
                if bp_a not in bps or bp_b not in bps:
                    continue
                ia, ib = bps.index(bp_a), bps.index(bp_b)
                if not visible[ia] or not visible[ib]:
                    continue
                ax.plot([pts[ia, 0], pts[ib, 0]],
                        [pts[ia, 1], pts[ib, 1]],
                        [pts[ia, 2], pts[ib, 2]],
                        color=cube_core.outline_edge_color(bp_a, bp_b, bp_colors.get),
                        linewidth=2)

            # Limb-hub-to-paw bridge (see limb_bridge_specs above): fires only on
            # frames where a mid-joint (thigh/knee, or a mouse-schema
            # forelegN/hindlegN segment) is untracked THIS frame while the
            # hub and paw are both still tracked -- a real gap in the
            # middle of an otherwise-drawn limb, not a permanently-omitted
            # bodypart (those are already handled by the static outline
            # chain-collapse, which needs no per-frame bridge). Drawn as
            # a separate direct line rather than replacing the normal
            # chain segments above, so a limb with only ONE mid-joint
            # missing (e.g. knee out, thigh still tracked) keeps its
            # hub-thigh segment AND gets this hub-paw bridge, instead of
            # losing the thigh's own position information.
            for hub_idx, mid_idxs, paw_idx, hub_bp, paw_bp in limb_bridge_specs:
                if not (visible[hub_idx] and visible[paw_idx]):
                    continue
                if mid_idxs.size and visible[mid_idxs].all():
                    continue
                ax.plot([pts[hub_idx, 0], pts[paw_idx, 0]],
                        [pts[hub_idx, 1], pts[paw_idx, 1]],
                        [pts[hub_idx, 2], pts[paw_idx, 2]],
                        color=cube_core.outline_edge_color(hub_bp, paw_bp, bp_colors.get),
                        linewidth=2)

            scatter_visible = visible & render_mask
            if scatter_visible.any():
                colors = [bp_colors[bp] for bp, v in zip(bps, scatter_visible) if v]
                ax.scatter(pts[scatter_visible, 0], pts[scatter_visible, 1], zs=pts[scatter_visible, 2],
                           s=30, c=colors, depthshade=True)

            ax.set_title(f"{k / fps:.2f}s / {n_total / fps:.2f}s", color=text_col, fontsize=9)

            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            if buf.nbytes != canvas_w * canvas_h * 3:
                raise RuntimeError(
                    f"Rendered frame size {buf.nbytes}B does not match "
                    f"expected {canvas_w}x{canvas_h}x3 — matplotlib canvas "
                    f"size changed mid-render")
            try:
                proc.stdin.write(buf.tobytes())
            except (BrokenPipeError, OSError) as e:
                stderr_thread.join(timeout=5)
                err = stderr_sink[0].decode(errors="replace") if stderr_sink else str(e)
                raise RuntimeError(f"FFmpeg exited early while writing skeleton video:\n{err}") from e

        proc.stdin.close()
        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            logger.warn("  Skeleton video FFmpeg did not exit after final flush; killing it")
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
    except Exception:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        if _blas_limiter is not None:
            _blas_limiter.restore_original_limits()
        plt.close(fig)

    stderr_thread.join(timeout=5)
    if proc.returncode != 0:
        err = stderr_sink[0].decode(errors="replace") if stderr_sink else ""
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Skeleton video FFmpeg failed:\n{err}")
    tmp_path.replace(out_path)
    logger.info(f"  Skeleton video written: {out_path.name}")


def _skeleton_video_subprocess_worker(h5_3d, out_path, fps, ffmpeg_exe, kwargs, conn) -> None:
    """
    Module-level (not nested — spawn-based multiprocessing needs the
    target to be importable by dotted path, which a locally-defined
    closure isn't) worker that runs the real make_skeleton_video() inside
    an isolated child process. Never receives the caller's real logger
    (it may hold GUI/queue state that isn't safely picklable across a
    process boundary via spawn, which is what Windows always uses) — logs
    to plain stdout instead, which the parent captures and relays.
    """
    class _PrintLogger:
        def info(self, m): print(f"[INFO] {m}", flush=True)
        def warn(self, m): print(f"[WARN] {m}", flush=True)
        def error(self, m): print(f"[ERROR] {m}", flush=True)
        def success(self, m): print(f"[OK] {m}", flush=True)

    try:
        make_skeleton_video(h5_3d, out_path, fps, ffmpeg_exe, _PrintLogger(), **kwargs)
        conn.send(("ok", None))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        conn.close()


def make_skeleton_video_isolated(h5_3d: Path, out_path: Path, fps: float,
                                  ffmpeg_exe: str, logger, timeout: float = 1800.0,
                                  max_attempts: int = 2, **kwargs) -> bool:
    """
    Runs make_skeleton_video() in a separate OS process (multiprocessing.
    Process, not a thread) so a native crash there can only kill that
    isolated process, never CUBE's own long-lived orchestrator process
    running everything else for the current session.

    Why this exists: real-data investigation (2026-08-28, see
    D:\\3D DLC\\CLAUDE.md's BLAS-thread-contention Crash Protection entry)
    found this render loop crashing natively and unpredictably — at
    different times faulting inside matplotlib's Agg backend, numpy's own
    core array module, and opencv, with no single deterministic bug
    identified despite genuine investigation (a BLAS-thread-count cap was
    added as a real, worthwhile mitigation for one contributing factor,
    but did not eliminate the crashes on its own; system resource
    headroom was confirmed ample — 98.9GB free of 127.7GB RAM, 320GB free
    disk — ruling out memory/disk pressure as the direct cause; the true
    root cause remains unconfirmed). A native crash is NOT a catchable
    Python exception — the ordinary try/except already wrapping the
    make_skeleton_video() call at the production call site
    (_run_3d_dlc_step's `export_skel` block) does nothing against it,
    which is the actual gap this function closes.

    max_attempts: retries the whole isolated render this many times
    before giving up (default 2 = one retry). Observed empirically all
    session: a crash here has consistently NOT repeated on an immediate
    retry (every crash this session succeeded within 1-2 attempts), the
    signature of a transient/timing-sensitive fault rather than a
    deterministic one — so a single automatic retry resolves the large
    majority of real cases without ever surfacing a skipped video.

    Returns True on success, False if every attempt failed (crash,
    timeout, or a caught exception inside the child) — logs a WARN either
    way rather than raising, matching this codebase's established "log
    and continue, don't abort the whole session's run over one
    QC/diagnostic output" convention (see e.g. the overlay/quad-video
    per-camera failure handling in _run_3d_dlc_step).
    """
    import multiprocessing as mp

    for attempt in range(1, max(1, max_attempts) + 1):
        # On retry, force libx264 instead of repeating the same NVENC attempt
        # that may have just failed -- a persistent NVENC issue then gets a
        # materially different attempt instead of the same failure twice.
        attempt_kwargs = dict(kwargs)
        if attempt > 1:
            attempt_kwargs["force_libx264"] = True

        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_skeleton_video_subprocess_worker,
            args=(h5_3d, out_path, fps, ffmpeg_exe, attempt_kwargs, child_conn))
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            logger.warn(f"  Skeleton video render for {out_path.name} timed out "
                        f"after {timeout:.0f}s (attempt {attempt}/{max_attempts}); "
                        f"killing isolated process")
            proc.terminate()
            proc.join(timeout=10)
            continue

        if proc.exitcode != 0:
            logger.warn(f"  Skeleton video render for {out_path.name} crashed in an "
                        f"isolated process (exit code {proc.exitcode}, attempt "
                        f"{attempt}/{max_attempts}). See D:\\3D DLC\\CLAUDE.md "
                        f"BLAS-thread-contention Crash Protection entry.")
            continue

        if parent_conn.poll():
            status, err = parent_conn.recv()
            if status == "error":
                logger.warn(f"  Skeleton video render for {out_path.name} failed "
                            f"(attempt {attempt}/{max_attempts}): {err}")
                continue

        if attempt > 1:
            logger.info(f"  Skeleton video render for {out_path.name} succeeded "
                        f"on retry (attempt {attempt}/{max_attempts})")
        return True

    logger.warn(f"  Skeleton video render for {out_path.name} failed after "
                f"{max_attempts} attempt(s) — skipping, pipeline continues.")
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Bodypart legend
# ─────────────────────────────────────────────────────────────────────────────

def plot_bodypart_legend(bps: list[str], out_path: Path, theme: str = "light") -> None:
    """
    Render a swatch legend (colored dot + bodypart name, head->tail order)
    matching bodypart_rainbow_colors(bps), so overlay/skeleton colors can be
    visually cross-referenced by name.  Skip-if-exists.

    theme: "dark"/"light" — same engine_cfg.plot_theme convention every
    other cube_core.py diagnostic plot uses (via _apply_plot_theme()).
    """
    if out_path.exists():
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cube_core
    from cube_core import bodypart_region_colors

    cube_core._apply_plot_theme(theme)
    bg_col, text_col = cube_core._BG, cube_core._TEXT_COL

    bp_colors = bodypart_region_colors(bps)
    n = len(bps)
    fig, ax = plt.subplots(figsize=(3.5, max(2.0, n * 0.28)), dpi=120,
                            facecolor=bg_col)
    ax.set_facecolor(bg_col)
    for i, bp in enumerate(bps):
        y = n - 1 - i
        ax.scatter([0], [y], color=bp_colors[bp], s=60)
        ax.text(0.3, y, bp, va="center", ha="left", color=text_col, fontsize=8)
    ax.set_xlim(-0.2, 3.5)
    ax.set_ylim(-1, n)
    ax.axis("off")
    plt.tight_layout()
    # cube_core._savefig writes to a .tmp sibling and renames onto out_path
    # only on success -- a crash/kill mid-save would otherwise leave a
    # truncated file at out_path, which the skip-if-exists check above
    # would then treat as finished forever.
    cube_core._savefig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  2D keypoint overlay (drawn before quad compositing)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_turned_away_frame_mask(h5_path: Path, fps: float, logger,
                                    likelihood_thresh: float = 0.3,
                                    visibility_adaptive_pct: float = 10.0,
                                    turned_away_conf_thresh: float = 0.30,
                                    turned_away_min_window_s: float = 0.4,
                                    turned_away_merge_gap_s: float = 0.5
                                    ) -> "np.ndarray | None":
    """
    Per-frame boolean "turned away from this camera" mask for one camera's
    own H5, for the quad-video overlay banner (2026-09-14, TMZ7 cam0
    investigation). Reuses cube_core.detect_turned_away_bins -- the SAME
    algorithm and SAME default thresholds BSoidEngine.run() already uses
    for its 2D-pipeline turned-away exclusion -- rather than a parallel
    heuristic, so the banner reflects the pipeline's actual judgment, not
    a separately-tuned approximation. Returns None (not an all-False mask)
    on any failure, so the caller can skip drawing the banner entirely
    rather than falsely claiming "never turned away".
    """
    try:
        import pandas as pd
        from cube_core import detect_turned_away_bins
        df = pd.read_hdf(str(h5_path))
        scorer = df.columns.get_level_values(0)[0]
        bps = df[scorer].columns.get_level_values(0).unique().tolist()
        ll = df[scorer].xs("likelihood", level="coords", axis=1).values
        bin_mask = detect_turned_away_bins(
            ll, bps, fps, likelihood_thresh, visibility_adaptive_pct,
            turned_away_conf_thresh, turned_away_min_window_s,
            turned_away_merge_gap_s)
        if bin_mask.size == 0:
            return None
        win = max(1, int(round(fps / 10)))
        frame_mask = np.repeat(bin_mask, win)
        n = len(ll)
        if len(frame_mask) < n:
            frame_mask = np.concatenate([frame_mask, np.zeros(n - len(frame_mask), dtype=bool)])
        else:
            frame_mask = frame_mask[:n]
        return frame_mask
    except Exception as e:
        logger.warn(f"  [{h5_path.name}] turned-away mask computation failed, "
                    f"skipping banner: {e}")
        return None


def overlay_keypoints_on_video(mp4_path: Path, h5_path: Path, out_path: Path,
                                bp_colors: dict, ll_gate: "float | None" = None, logger=None,
                                ffmpeg_exe: str = None, dot_radius: int = 3,
                                line_width: int = 1,
                                skeleton: "list[tuple[str, str]] | None" = None,
                                turned_away_mask: "np.ndarray | None" = None) -> None:
    """
    Draw DLC-detected keypoints (small filled dots, optional thin skeleton
    lines) onto *mp4_path* using *h5_path*'s per-frame predictions.

    Small dots (default 3px radius) so the overlay doesn't obscure the
    animal in the raw camera view — this is deliberately much smaller than
    the 3D skeleton video's scatter markers.  The background footage is
    converted to grayscale so the color-coded keypoints/skeleton stand out
    clearly.  Skip-if-exists.

    ll_gate (2026-09-18, D:\\3D DLC\\CLAUDE.md, direct user request):
    defaults to cube_core.DISPLAY_LL_THRESHOLD (0.4) -- a fixed, universal
    display confidence gate used identically for every CUBE run, 2D and
    3D alike. Previously every call site passed its own session-derived
    value (dlc_pcutoff for 2D; dlc_3d_ll_threshold for 3D, which defaults
    to 0.0 -- already flagged elsewhere in this codebase as a real,
    separate "shows near-zero-confidence junk" bug). Pass an explicit
    value only to deliberately override the universal default for one
    call; production call sites should not do this.

    Per-bodypart visibility floor (same request): any bodypart whose
    likelihood clears ll_gate on fewer than cube_core.DISPLAY_BP_MIN_
    SHOWN_FRAC (0.5) of this video's frames is hidden ENTIRELY for the
    whole video -- never drawn on any frame, and excluded from every
    skeleton edge that touches it -- rather than flickering in and out.
    Logged once (bodypart names + their shown-fraction) so a reviewer can
    see which points were dropped and why.

    turned_away_mask (2026-09-14, TMZ7 cam0 investigation): an optional
    per-frame boolean array (see _compute_turned_away_frame_mask) -- when
    given and True for a frame, draws a solid red "TURNED AWAY" banner
    across the bottom of that frame for THIS camera. Lets a manual reviewer
    of the quad video immediately see which camera(s) the pipeline itself
    judged to have lost a facing-camera view that frame, rather than having
    to infer it by eye (this is exactly the ambiguity a real reviewer ran
    into: "the animal is turned away at 1:16 -- was this detected?").
    """
    import cv2
    import pandas as pd
    from cube_core import outline_edge_color, DISPLAY_LL_THRESHOLD, DISPLAY_BP_MIN_SHOWN_FRAC

    if out_path.exists():
        return

    ll_gate = float(ll_gate) if ll_gate else DISPLAY_LL_THRESHOLD

    df = pd.read_hdf(str(h5_path))
    scorer = df.columns.get_level_values(0)[0]
    bps = df[scorer].columns.get_level_values(0).unique().tolist()
    X  = df[scorer].xs("x",          level="coords", axis=1).values
    Y  = df[scorer].xs("y",          level="coords", axis=1).values
    LL = df[scorer].xs("likelihood", level="coords", axis=1).values

    shown_frac = np.mean(LL >= ll_gate, axis=0)
    bp_enabled = shown_frac >= DISPLAY_BP_MIN_SHOWN_FRAC
    dropped = [f"{bps[i]} ({shown_frac[i]*100:.0f}%)"
               for i in range(len(bps)) if not bp_enabled[i] and bps[i] in bp_colors]
    if dropped and logger is not None:
        logger.info(f"  [{mp4_path.name}] hiding {len(dropped)} unreliable bodypart(s) "
                    f"(<{DISPLAY_BP_MIN_SHOWN_FRAC*100:.0f}% shown @{ll_gate}): "
                    + ", ".join(dropped))

    bgr_colors = {
        bp: (round(c[2] * 255), round(c[1] * 255), round(c[0] * 255))
        for bi, (bp, c) in enumerate(bp_colors.items()) if bp in bps and bp_enabled[bps.index(bp)]
    }

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for overlay: {mp4_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(X))
    cap.release()

    # Real retry across encoders (mirrors _crop_camera_video's pattern), not
    # just a one-shot pick based on the cached _nvenc_available() probe: this
    # camera-loop step now runs multiple overlays concurrently, so a
    # transient NVENC concurrent-session-limit failure is a real possibility
    # even when _nvenc_available() reported true. Re-runs the full
    # decode/draw/pipe loop per attempt -- unlike crop's one-shot ffmpeg
    # call, but the decode/draw cost here is bounded and this is not the
    # crash-prone matplotlib path skeleton uses.
    encoders = [("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]),
                ("libx264",    ["-c:v", "libx264", "-threads", "0", "-crf", "20", "-preset", "fast"])]
    if not _nvenc_available(ffmpeg_exe):
        encoders = encoders[1:]

    # Atomic: ffmpeg writes to a .tmp sibling, renamed onto out_path only on
    # success -- a crash/kill mid-encode would otherwise leave a
    # truncated/corrupt file directly at out_path, which the skip-if-exists
    # check at the top of this function would then treat as finished forever.
    # tmp marker before the real suffix -- see _crop_camera_video's note.
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    last_err: Exception | None = None

    for enc_name, enc_args in encoders:
        cap = cv2.VideoCapture(str(mp4_path))
        cmd = [ffmpeg_exe, "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
               *enc_args, "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(tmp_path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        stderr_sink: list = []
        stderr_thread = threading.Thread(
            target=_drain_stream, args=(proc.stderr, stderr_sink), daemon=True)
        stderr_thread.start()

        try:
            try:
                for k in range(n_frames):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    # Grayscale background so the color-coded keypoint overlay
                    # stands out clearly against the animal/enclosure instead
                    # of competing with real footage colors.
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    ll = LL[k]
                    gate = ll >= ll_gate

                    if skeleton:
                        for bp_a, bp_b in skeleton:
                            if bp_a not in bps or bp_b not in bps:
                                continue
                            if bp_a not in bgr_colors or bp_b not in bgr_colors:
                                continue
                            ia, ib = bps.index(bp_a), bps.index(bp_b)
                            if not (gate[ia] and gate[ib]):
                                continue
                            pa = (int(X[k, ia]), int(Y[k, ia]))
                            pb = (int(X[k, ib]), int(Y[k, ib]))
                            edge_bgr = outline_edge_color(bp_a, bp_b, bgr_colors.get)
                            cv2.line(frame, pa, pb, edge_bgr, line_width)

                    for bi, bp in enumerate(bps):
                        if bp not in bgr_colors or not gate[bi]:
                            continue
                        pt = (int(X[k, bi]), int(Y[k, bi]))
                        cv2.circle(frame, pt, dot_radius, bgr_colors[bp], -1)

                    if turned_away_mask is not None and k < len(turned_away_mask) and turned_away_mask[k]:
                        bar_h = max(24, h // 18)
                        cv2.rectangle(frame, (0, h - bar_h), (w, h), (0, 0, 200), -1)
                        cv2.putText(frame, "TURNED AWAY", (8, h - bar_h // 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, bar_h / 40.0,
                                    (255, 255, 255), 2, cv2.LINE_AA)

                    try:
                        proc.stdin.write(frame.tobytes())
                    except (BrokenPipeError, OSError) as e:
                        stderr_thread.join(timeout=5)
                        err = stderr_sink[0].decode(errors="replace") if stderr_sink else str(e)
                        raise RuntimeError(
                            f"FFmpeg exited early while writing overlay video:\n{err}") from e

                proc.stdin.close()
                proc.wait(timeout=300)
            finally:
                cap.release()

            stderr_thread.join(timeout=5)
            if proc.returncode != 0:
                err = stderr_sink[0].decode(errors="replace") if stderr_sink else ""
                raise RuntimeError(f"Overlay video FFmpeg failed:\n{err}")
            tmp_path.replace(out_path)
            logger.info(f"  Overlay video written [{enc_name}]: {out_path.name}")
            return
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            last_err = e
            if enc_name == "h264_nvenc":
                logger.info(f"  [{mp4_path.name}] NVENC overlay failed, "
                            "retrying with libx264")

    raise RuntimeError(f"Overlay failed for {mp4_path.name}: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
#  Per-camera Zoo adaptation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_batch_size() -> int:
    """Return a conservative GPU batch size (85 % of free VRAM)."""
    try:
        import torch
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
            usable  = free_gb * 0.85
            return 32 if usable >= 10 else (16 if usable >= 5 else 8)
    except Exception:
        pass
    return 8


def _compute_scale_list(video_path: Path, _adv: dict) -> list:
    """
    Mirrors 2D's (cube.py) scale_list Auto/Manual logic exactly:
    Manual uses the shared dlc_scale_min/max/step advanced-cfg values;
    Auto opens *video_path* and derives a range centred on 35% of the
    video's shorter side, recomputed per video.
    """
    scale_mode = str(_adv.get("dlc_scale_mode", "Auto"))
    if scale_mode == "Manual":
        scale_min  = int(_adv.get("dlc_scale_min",  100))
        scale_max  = int(_adv.get("dlc_scale_max",  600))
        scale_step = int(_adv.get("dlc_scale_step", 50))
        return list(range(scale_min, scale_max + scale_step, scale_step))
    try:
        import cv2 as _cv2
        cap = _cv2.VideoCapture(str(video_path))
        short = min(int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
    except Exception:
        short = 720
    centre = max(150, int(short * 0.35))
    return list(range(max(100, centre - 150), min(1200, centre + 200), 50))


def _adapt_zoo_for_camera(rep_video: Path, cam_label: str,
                           adapt_work: Path, logger,
                           detector_epochs: int, pose_epochs: int,
                           sa_name: str, model_name: str, detector_name: str,
                           pcutoff: float, bbox_thr_adapt: float,
                           max_ind: int, inf_batch: int, det_batch: int,
                           pseudo_thr: float, transfer: bool, _adv: dict):
    """
    Run DLC Zoo video adaptation on *rep_video*.
    Returns (adapted_pose_ckpt, adapted_detector_ckpt, snapshot_base, adapt_project_dir)
    where the first available pair should be used for batch inference.
    """
    import deeplabcut as dlc

    cam_adapt_dir = adapt_work / f"adapt_{cam_label}"
    cam_adapt_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  [{cam_label}] Zoo adaptation on: {rep_video.name}")

    adapt_config_path  = None
    adapt_project_dir  = None
    adapted_pose_ckpt  = None
    adapted_det_ckpt   = None
    snapshot_base      = None

    try:
        if hasattr(dlc, "create_video_adaptation_project"):
            kw = dict(
                videos            = [str(rep_video)],
                working_directory = str(cam_adapt_dir),
                superanimal_name  = sa_name,
                num_epochs        = pose_epochs,
                batch_size        = inf_batch,
            )
            try:
                import inspect as _insp
                sig = _insp.signature(dlc.create_video_adaptation_project)
                if "display_iters" in sig.parameters:
                    kw["display_iters"] = 100
                elif "displayiters" in sig.parameters:
                    kw["displayiters"] = 100
            except Exception:
                pass

            result = dlc.create_video_adaptation_project(**kw)
            if result and Path(str(result)).is_file():
                adapt_config_path = str(result)
                adapt_project_dir = Path(result).parent

        if adapt_config_path is None:
            # Fallback: video_inference_superanimal with video_adapt=True
            rep_dest = cam_adapt_dir / f"{rep_video.stem}_adapt_results"
            rep_dest.mkdir(parents=True, exist_ok=True)
            rep_inf = rep_dest / rep_video.name
            if not rep_inf.exists():
                shutil.copy2(str(rep_video), str(rep_inf))

            sa_kw = dict(
                superanimal_name     = sa_name,
                model_name           = model_name,
                detector_name        = detector_name,
                scale_list           = _compute_scale_list(rep_video, _adv),
                pcutoff              = pcutoff,
                bbox_threshold       = bbox_thr_adapt,
                max_individuals      = max_ind,
                batch_size           = inf_batch,
                detector_batch_size  = det_batch,
                create_labeled_video = False,
                video_adapt          = True,
                pseudo_threshold     = pseudo_thr,
                detector_epochs      = detector_epochs,
                pose_epochs          = pose_epochs,
                device               = "auto",
            )
            try:
                import inspect as _insp
                sig = _insp.signature(dlc.video_inference_superanimal).parameters
                if "superanimal_transfer_learning" in sig:
                    sa_kw["superanimal_transfer_learning"] = transfer
                if "video_adapt_batch_size" in sig:
                    sa_kw["video_adapt_batch_size"] = inf_batch
            except Exception:
                pass

            dlc.video_inference_superanimal([str(rep_inf)], **sa_kw)
            adapt_project_dir = cam_adapt_dir

        # Find adapted checkpoints
        adapted_pose_ckpt, adapted_det_ckpt = _find_adapted_pt_checkpoints(
            adapt_project_dir, model_name, detector_name, logger)
        if adapted_pose_ckpt is None:
            snapshot_base = _find_highest_snapshot_path(adapt_project_dir, logger)

    except Exception:
        logger.error(
            f"  [{cam_label}] Adaptation failed:\n{traceback.format_exc()}")

    return adapted_pose_ckpt, adapted_det_ckpt, snapshot_base, adapt_project_dir


def _find_adapted_pt_checkpoints(search_root: Path, model_name: str,
                                  detector_name: str, logger
                                  ) -> tuple[str | None, str | None]:
    """Locate DLC 3.x PyTorch .pt checkpoints in adaptation workspace."""
    pose_ckpt = None
    det_ckpt  = None
    if search_root is None or not search_root.exists():
        return None, None
    all_pts = sorted(search_root.rglob("*.pt"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    for pt in all_pts:
        nm = pt.name.lower()
        if "pose" in nm or model_name.lower() in nm:
            if pose_ckpt is None:
                pose_ckpt = str(pt)
        elif "detect" in nm or detector_name.lower() in nm:
            if det_ckpt is None:
                det_ckpt = str(pt)
        if pose_ckpt and det_ckpt:
            break
    if pose_ckpt:
        logger.info(f"  Found adapted pose checkpoint: {Path(pose_ckpt).name}")
    if det_ckpt:
        logger.info(f"  Found adapted detector checkpoint: {Path(det_ckpt).name}")
    return pose_ckpt, det_ckpt


def _find_highest_snapshot_path(search_root: Path, logger) -> str | None:
    """Locate the highest-numbered TF snapshot-N.index file."""
    if search_root is None or not search_root.exists():
        return None
    import re as _re
    best_num  = -1
    best_path = None
    for p in search_root.rglob("snapshot-*.index"):
        m = _re.search(r"snapshot-(\d+)\.index$", p.name)
        if m:
            n = int(m.group(1))
            if n > best_num:
                best_num  = n
                best_path = str(p.with_suffix(""))   # strip .index
    if best_path:
        logger.info(f"  Found TF snapshot: snapshot-{best_num}")
    return best_path


def _autodetect_cached_adaptation(adapt_work: Path, cam_label: str,
                                   model_name: str, detector_name: str,
                                   logger) -> dict | None:
    """
    Detect an already-completed Zoo adaptation for *cam_label* directly on
    disk, at its deterministic location adapt_work/adapt_{cam_label}/ —
    independent of session["dlc_3d_models"], which only survives when the
    in-memory pipeline session carries over from the run that produced it.
    A fresh session (e.g. after an app restart or a crash mid-run) starts
    with that cache empty even though real checkpoints already sit on disk,
    which previously made Step 1b silently redo the whole DLC-Zoo
    adaptation from scratch. Returns the same dict shape used for
    session["dlc_3d_models"][cam_label], or None if nothing usable is
    found (caller falls through to running adaptation as normal).
    """
    cam_adapt_dir = adapt_work / f"adapt_{cam_label}"
    if not cam_adapt_dir.exists():
        return None
    pose_ckpt, det_ckpt = _find_adapted_pt_checkpoints(
        cam_adapt_dir, model_name, detector_name, logger)
    snap_base = None
    if pose_ckpt is None:
        snap_base = _find_highest_snapshot_path(cam_adapt_dir, logger)
    if pose_ckpt is None and snap_base is None:
        return None
    return {
        "pose_ckpt": pose_ckpt,
        "det_ckpt":  det_ckpt,
        "snap_base": snap_base,
        "adapt_dir": str(cam_adapt_dir),
    }


def _infer_camera_videos(dlc, videos: list[Path], cam_label: str,
                          adapted_pose_ckpt: str | None,
                          adapted_det_ckpt: str | None,
                          _snapshot_base: str | None,
                          inference_config: str | None,
                          sa_name: str, model_name: str, detector_name: str,
                          pcutoff: float, bbox_thr: float, max_ind: int,
                          filter_types: list,
                          fps: float,
                          logger, inf_batch: int, det_batch: int,
                          transfer: bool, _adv: dict,
                          dest_root: Path | None = None) -> None:
    """
    Run batch inference for *cam_label* on all *videos*, applying OOM-retry.
    Writes {stem}_filtered.h5 into dest_root/{stem}_results/ when dest_root
    is given (this run's "CUBE 1b - 3D DLC/_prep/" folder) -- unconditional,
    regardless of whether *videos* were cropped, so a per-camera H5 lands
    under Step 1b's own folder even when cropping is disabled for that
    camera. Defaults to writing alongside each input video (the old
    behavior) when dest_root isn't given.
    """
    from cube_core import filter_dlc_h5, dlc_filter_extra_kwargs

    current_batch = inf_batch
    for idx, vpath in enumerate(videos, 1):
        vname       = vpath.name
        base_noext  = vpath.stem
        dest_folder = ((Path(dest_root) if dest_root is not None else vpath.parent)
                        / f"{base_noext}_results")
        dest_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"  [{cam_label}] [{idx}/{len(videos)}] {vname}")

        # Skip-if-exists: a video whose *_filtered.h5 already sits in
        # dest_folder (from a prior run, or a prior attempt this same run
        # that crashed later) doesn't need re-inference — re-running here
        # was previously unconditional and silently redid DLC inference
        # for every video on every Step 1b launch, discarding real prior
        # GPU work whenever the in-memory session state didn't carry over
        # (e.g. after an app restart). Only a real, non-empty H5 counts;
        # a zero-byte or unreadable leftover from an interrupted run is
        # still re-inferred.
        existing_h5 = dest_folder / f"{base_noext}_filtered.h5"
        if existing_h5.exists() and existing_h5.stat().st_size > 0:
            logger.info(f"  [{cam_label}] [{idx}/{len(videos)}] "
                        f"Skipping (already exists): {existing_h5.name}")
            continue

        h5_before = set(dest_folder.glob("*.h5"))
        succeeded = False
        while not succeeded:
            try:
                if adapted_pose_ckpt:
                    sa_kw = dict(
                        superanimal_name               = sa_name,
                        model_name                     = model_name,
                        detector_name                  = detector_name,
                        scale_list                     = _compute_scale_list(vpath, _adv),
                        pcutoff                        = pcutoff,
                        bbox_threshold                 = bbox_thr,
                        max_individuals                = max_ind,
                        batch_size                     = current_batch,
                        # bounded by current_batch so a prior OOM-driven
                        # reduction (see the retry loop below) is respected
                        # even when dlc_det_batch was set to a larger value
                        detector_batch_size            = min(det_batch, current_batch),
                        create_labeled_video           = False,
                        video_adapt                    = False,
                        dest_folder                    = str(dest_folder),
                        device                         = "auto",
                        customized_pose_checkpoint     = adapted_pose_ckpt,
                    )
                    if adapted_det_ckpt:
                        sa_kw["customized_detector_checkpoint"] = adapted_det_ckpt
                    try:
                        import inspect as _insp2
                        sig2 = _insp2.signature(dlc.video_inference_superanimal).parameters
                        if "superanimal_transfer_learning" in sig2:
                            sa_kw["superanimal_transfer_learning"] = transfer
                    except Exception:
                        pass
                    dlc.video_inference_superanimal([str(vpath)], **sa_kw)
                elif inference_config:
                    dlc.analyze_videos(
                        inference_config,
                        [str(vpath)],
                        save_as_csv    = True,
                        destfolder     = str(dest_folder),
                        batchsize      = current_batch,
                        allow_growth   = True,
                        robust_nframes = True,
                    )
                else:
                    sa_kw = dict(
                        superanimal_name     = sa_name,
                        model_name           = model_name,
                        detector_name        = detector_name,
                        scale_list           = _compute_scale_list(vpath, _adv),
                        pcutoff              = pcutoff,
                        bbox_threshold       = bbox_thr,
                        max_individuals      = max_ind,
                        batch_size           = current_batch,
                        detector_batch_size  = min(det_batch, current_batch),
                        create_labeled_video = False,
                        video_adapt          = False,
                        dest_folder          = str(dest_folder),
                        device               = "auto",
                    )
                    try:
                        import inspect as _insp3
                        sig3 = _insp3.signature(dlc.video_inference_superanimal).parameters
                        if "superanimal_transfer_learning" in sig3:
                            sa_kw["superanimal_transfer_learning"] = transfer
                    except Exception:
                        pass
                    dlc.video_inference_superanimal([str(vpath)], **sa_kw)
                succeeded = True
            except RuntimeError as oom_exc:
                exc_l = str(oom_exc).lower()
                if "cuda" in exc_l and ("memory" in exc_l or "oom" in exc_l
                                         or "alloc" in exc_l):
                    if current_batch <= 1:
                        logger.error(
                            f"  [{cam_label}] OOM at batchsize=1 — skipping {vname}")
                        succeeded = True
                    else:
                        current_batch = max(1, current_batch // 2)
                        logger.warn(
                            f"  [{cam_label}] OOM → batchsize={current_batch}, retrying")
                        gc.collect()
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                else:
                    logger.error(f"  [{cam_label}] RuntimeError: {oom_exc}")
                    succeeded = True
            except Exception:
                logger.error(
                    f"  [{cam_label}] Inference error on {vname}:\n{traceback.format_exc()}")
                succeeded = True

        # Post-inference: filter H5
        h5_new = [p for p in dest_folder.glob("*.h5")
                  if p not in h5_before
                  and not p.name.startswith("BSOID_")
                  and not p.stem.endswith("_filtered")]
        if h5_new:
            final_h5       = h5_new[0]
            clean_filtered = dest_folder / f"{base_noext}_filtered.h5"
            if filter_types:
                try:
                    filter_dlc_h5(final_h5, filter_types, log_fn=logger,
                                  out_path=clean_filtered, fps=fps,
                                  likelihood_thresh=pcutoff,
                                  **dlc_filter_extra_kwargs(_adv))
                except Exception as fe:
                    logger.warn(f"  filter_dlc_h5 failed: {fe} — copying raw")
                    shutil.copy2(str(final_h5), str(clean_filtered))
            else:
                shutil.copy2(str(final_h5), str(clean_filtered))
            try:
                final_h5.rename(dest_folder / f"{base_noext}.h5")
            except Exception:
                pass

        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Per-camera bbox_threshold_adapt auto-detection
#  (see D:\CUBE\3D_AUTO_BBOX_THRESHOLD_PLAN.md)
# ─────────────────────────────────────────────────────────────────────────────

def _dry_pass_camera_yield(video: Path, threshold: float, sa_name: str,
                            model_name: str, detector_name: str, adv: dict,
                            logger, clip_seconds: float, work_dir: Path) -> float:
    """
    Trim a short throwaway clip from *video*, run plain base-Zoo inference
    (no adaptation, no training) at *threshold*, and return the fraction of
    frames with at least one detection whose confidence clears *threshold*.

    The clip and its results directory are deleted before returning -- this
    is diagnostic-only data, never a pipeline output. Raises on any failure
    (ffmpeg trim, inference, or H5 read) so the caller's own try/except can
    decide the fallback -- this function never silently returns a fabricated
    yield value.
    """
    import deeplabcut as dlc
    import pandas as pd

    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio_ffmpeg as _iio_ffmpeg
        ffmpeg_exe = _iio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"

    tag        = f"{video.stem}_dryclip_thr{int(round(threshold * 1000))}"
    clip_path  = work_dir / f"{tag}.mp4"
    dest_folder = work_dir / f"{tag}_results"

    try:
        cmd = [ffmpeg_exe, "-y", "-i", str(video), "-t", str(clip_seconds),
               "-c", "copy", str(clip_path)]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or not clip_path.exists() or clip_path.stat().st_size == 0:
            raise RuntimeError(
                f"ffmpeg failed to trim dry-pass clip for {video.name}: "
                f"{proc.stderr.decode(errors='replace')[-400:]}")

        dest_folder.mkdir(parents=True, exist_ok=True)
        sa_kw = dict(
            superanimal_name     = sa_name,
            model_name            = model_name,
            detector_name          = detector_name,
            scale_list             = _compute_scale_list(clip_path, adv),
            pcutoff                = 0.0,
            bbox_threshold          = threshold,
            max_individuals         = int(adv.get("dlc_max_individuals", 1)),
            batch_size              = _gpu_batch_size(),
            detector_batch_size     = _gpu_batch_size(),
            create_labeled_video    = False,
            video_adapt             = False,
            dest_folder             = str(dest_folder),
            device                  = "auto",
        )
        try:
            import inspect as _insp
            sig = _insp.signature(dlc.video_inference_superanimal).parameters
            if "superanimal_transfer_learning" in sig:
                sa_kw["superanimal_transfer_learning"] = bool(adv.get("dlc_transfer", True))
        except Exception:
            pass
        dlc.video_inference_superanimal([str(clip_path)], **sa_kw)

        h5_cands = [p for p in dest_folder.glob("*.h5")
                    if not p.name.startswith("BSOID_")]
        if not h5_cands:
            raise RuntimeError(
                f"Dry-pass inference produced no H5 for {video.name} at threshold {threshold}")

        df = pd.read_hdf(str(h5_cands[0]))
        scorer  = df.columns.get_level_values(0)[0]
        ll_cols = [c for c in df[scorer].columns if c[-1] == "likelihood"]
        if not ll_cols:
            return 0.0
        ll = df[scorer][ll_cols].to_numpy(dtype=float)
        per_frame_max = np.nanmax(ll, axis=1)
        return float(np.mean(per_frame_max >= threshold))
    finally:
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            if clip_path.exists():
                clip_path.unlink()
        except Exception:
            pass
        try:
            if dest_folder.exists():
                shutil.rmtree(dest_folder, ignore_errors=True)
        except Exception:
            pass


def _resolve_stage1_bbox_thresholds(sessions: list[dict], cam_labels: list[str],
                                     bbox_thr_adapt: float, overrides: dict,
                                     adv: dict, logger,
                                     *, autodetect: bool = True,
                                     clip_seconds: float = 20.0,
                                     outlier_ratio: float = 0.6,
                                     threshold_step: float = 0.1,
                                     threshold_min: float = 0.3,
                                     sa_name: str = "superanimal_quadruped",
                                     model_name: str = "hrnet_w32",
                                     detector_name: str = "fasterrcnn_mobilenet_v3_large_fpn",
                                     work_dir: "Path | None" = None,
                                     ) -> tuple[dict, dict]:
    """
    Stage 1: pre-adaptation dry-pass baseline + auto-threshold decision.

    Runs a cheap base-Zoo dry-pass inference per camera (skipping any camera
    with a manual override in *overrides*), computes the group's median
    yield, flags cameras below outlier_ratio * median, and steps a flagged
    camera's threshold down until it clears the floor or hits threshold_min.

    Returns (resolved: {cam_label: float}, report: {..., "per_camera": {...}})
    matching the {session_id}_bbox_threshold_report.json schema in the plan.
    Never raises -- any per-camera failure falls back to bbox_thr_adapt for
    that camera and is logged as a WARN; this is a best-effort optimization,
    never a hard requirement for the pipeline to proceed.
    """
    resolved: dict = {}
    report: dict = {"baseline_yield_median": None, "per_camera": {}}

    if not autodetect:
        return {cam: float(overrides.get(cam, bbox_thr_adapt)) for cam in cam_labels}, report

    if work_dir is None:
        work_dir = Path(".") / "stage1_dry_pass"
    dry_pass_dir = Path(work_dir) / "stage1_dry_pass"

    # Cameras eligible for Stage 1 (no manual override) and their representative video
    eligible: dict = {}
    for cam in cam_labels:
        entry = {"manual_override": overrides.get(cam), "stage1_yield": None,
                  "stage1_flagged": False, "final_threshold_used":
                      float(overrides.get(cam, bbox_thr_adapt))}
        report["per_camera"][cam] = entry
        if cam in overrides:
            resolved[cam] = float(overrides[cam])
            continue
        rep_video = select_representative(sessions, cam)
        if rep_video is None:
            resolved[cam] = bbox_thr_adapt
            continue
        eligible[cam] = rep_video

    if len(eligible) < 2:
        # No meaningful median/baseline possible with <2 cameras to compare --
        # skip entirely and use the shared threshold for every eligible camera.
        logger.info(
            "  [Stage 1] Fewer than 2 cameras without a manual override -- "
            "skipping auto-detection, using the shared threshold.")
        for cam in eligible:
            resolved[cam] = bbox_thr_adapt
            report["per_camera"][cam]["final_threshold_used"] = bbox_thr_adapt
        return resolved, report

    logger.step("  [Stage 1] Dry-pass baseline check across cameras...")
    yields: dict = {}
    for cam, rep_video in eligible.items():
        try:
            y = _dry_pass_camera_yield(
                rep_video, bbox_thr_adapt, sa_name, model_name, detector_name,
                adv, logger, clip_seconds, dry_pass_dir)
            yields[cam] = y
            report["per_camera"][cam]["stage1_yield"] = y
        except Exception as e:
            logger.warn(f"  [Stage 1] {cam} dry-pass failed, using shared "
                        f"threshold unchanged: {e}")
            resolved[cam] = bbox_thr_adapt
            report["per_camera"][cam]["final_threshold_used"] = bbox_thr_adapt

    if len(yields) < 2:
        for cam in eligible:
            if cam not in resolved:
                resolved[cam] = bbox_thr_adapt
                report["per_camera"][cam]["final_threshold_used"] = bbox_thr_adapt
        return resolved, report

    baseline_median = float(np.median(list(yields.values())))
    report["baseline_yield_median"] = baseline_median
    floor = baseline_median * outlier_ratio

    for cam, y in yields.items():
        flagged = y < floor
        report["per_camera"][cam]["stage1_flagged"] = flagged
        if not flagged:
            logger.info(f"  [Stage 1] {cam} dry-pass yield: {y:.2f} "
                        f"(baseline median: {baseline_median:.2f}) — OK")
            resolved[cam] = bbox_thr_adapt
            report["per_camera"][cam]["final_threshold_used"] = bbox_thr_adapt
            continue

        logger.warn(f"  [Stage 1] {cam} dry-pass yield: {y:.2f} "
                     f"(baseline median: {baseline_median:.2f}, floor: {floor:.3f}) "
                     f"— FLAGGED, stepping down")
        steps = [{"threshold": bbox_thr_adapt, "yield": y}]
        cur_thr = bbox_thr_adapt
        cur_yield = y
        rep_video = eligible[cam]
        while cur_yield < floor and cur_thr - threshold_step >= threshold_min - 1e-9:
            new_thr = round(cur_thr - threshold_step, 6)
            try:
                new_yield = _dry_pass_camera_yield(
                    rep_video, new_thr, sa_name, model_name, detector_name,
                    adv, logger, clip_seconds, dry_pass_dir)
            except Exception as e:
                logger.warn(f"  [Stage 1] {cam} step to {new_thr} failed: {e} "
                            f"— keeping {cur_thr}")
                break
            steps.append({"threshold": new_thr, "yield": new_yield})
            if new_yield >= floor:
                logger.success(f"  [Stage 1] {cam} threshold {cur_thr:.2f} → "
                                f"{new_thr:.2f}: yield {new_yield:.2f} — "
                                f"clears floor, using {new_thr:.2f}")
            else:
                logger.info(f"  [Stage 1] {cam} threshold {cur_thr:.2f} → "
                            f"{new_thr:.2f}: yield {new_yield:.2f} — still "
                            f"below floor, stepping down")
            cur_thr, cur_yield = new_thr, new_yield

        if cur_yield < floor:
            logger.warn(f"  [Stage 1] {cam} never cleared the floor by "
                        f"threshold_min={threshold_min} — using lowest value "
                        f"tried ({cur_thr:.2f}); this camera may have a real "
                        f"problem a threshold change can't fix (bad crop, "
                        f"occlusion, etc.)")

        resolved[cam] = cur_thr
        report["per_camera"][cam]["stage1_steps"] = steps
        report["per_camera"][cam]["final_threshold_used"] = cur_thr

    # Any camera not yet resolved (e.g. no rep video) falls back to shared default
    for cam in cam_labels:
        if cam not in resolved:
            resolved[cam] = float(overrides.get(cam, bbox_thr_adapt))

    return resolved, report


def _count_pseudo_label_images(adapt_dir: "Path | None") -> int:
    """
    Count pseudo-labeled training images under a camera's adaptation
    workspace. video_adapt=True writes labeled frames under a directory
    whose name contains "pseudo" -- counts image files there, or (if no such
    directory exists, e.g. the create_video_adaptation_project() code path)
    falls back to counting images under a "labeled-data" tree, which is
    where DLC's project-based adaptation stores its own extracted frames.
    """
    if adapt_dir is None or not Path(adapt_dir).exists():
        return 0
    adapt_dir = Path(adapt_dir)
    img_exts = {".png", ".jpg", ".jpeg"}

    pseudo_dirs = [p for p in adapt_dir.rglob("*") if p.is_dir() and "pseudo" in p.name.lower()]
    if pseudo_dirs:
        count = 0
        for d in pseudo_dirs:
            count += sum(1 for f in d.rglob("*") if f.is_file() and f.suffix.lower() in img_exts)
        return count

    labeled_dirs = [p for p in adapt_dir.rglob("labeled-data") if p.is_dir()]
    if labeled_dirs:
        count = 0
        for d in labeled_dirs:
            count += sum(1 for f in d.rglob("*") if f.is_file() and f.suffix.lower() in img_exts)
        return count

    return 0


def _run_camera_adaptation_with_retry(
        rep_video: Path, cam_label: str, adapt_work: Path, logger,
        detector_epochs: int, pose_epochs: int, sa_name: str, model_name: str,
        detector_name: str, pcutoff: float, bbox_thr_adapt: float, max_ind: int,
        inf_batch: int, det_batch: int, pseudo_thr: float, transfer: bool,
        _adv: dict, *, autodetect: bool, has_manual_override: bool,
        baseline_counts: dict, outlier_ratio: float, threshold_step: float,
        max_retries: int, report_entry: dict):
    """
    Stage 2: wraps _adapt_zoo_for_camera in a post-hoc pseudo-label-count
    floor check + retry loop, mirroring the OOM-retry loop's shape but keyed
    on pseudo-label count instead of a CUDA OOM exception.

    Returns the same (pose_ckpt, det_ckpt, snap_base, adapt_dir) tuple
    _adapt_zoo_for_camera returns, so the Phase 2 call site is unaffected.
    Never fires for a camera with a manual override, or when this is the
    first camera processed this run (no baseline_counts yet to compare
    against) -- both cases fall straight through to a single attempt.
    """
    attempt = 0
    cur_thr = bbox_thr_adapt
    steps: list = []
    result = _adapt_zoo_for_camera(
        rep_video, cam_label, adapt_work, logger, detector_epochs, pose_epochs,
        sa_name, model_name, detector_name, pcutoff, cur_thr, max_ind, inf_batch,
        det_batch, pseudo_thr, transfer, _adv)
    pose_ckpt, det_ckpt, snap_base, adapt_dir = result
    count = _count_pseudo_label_images(adapt_dir)
    steps.append({"threshold": cur_thr, "count": count})

    if not autodetect or has_manual_override or not baseline_counts:
        report_entry["stage2_retries"] = 0
        report_entry["final_pseudo_label_count"] = count
        return result

    baseline_median = float(np.median(list(baseline_counts.values())))
    floor = baseline_median * outlier_ratio

    while count < floor and attempt < max_retries:
        attempt += 1
        new_thr = round(max(0.0, cur_thr - threshold_step), 6)
        logger.warn(f"  [Stage 2] {cam_label} pseudo-label count: {count} "
                     f"(baseline median: {baseline_median:.0f}, floor: {floor:.0f}) "
                     f"— FLAGGED, retrying adaptation (attempt {attempt}/{max_retries}) "
                     f"at threshold {new_thr:.2f}")
        result = _adapt_zoo_for_camera(
            rep_video, cam_label, adapt_work, logger, detector_epochs, pose_epochs,
            sa_name, model_name, detector_name, pcutoff, new_thr, max_ind, inf_batch,
            det_batch, pseudo_thr, transfer, _adv)
        pose_ckpt, det_ckpt, snap_base, adapt_dir = result
        new_count = _count_pseudo_label_images(adapt_dir)
        steps.append({"threshold": new_thr, "count": new_count})
        if new_count > count:
            # keep the better of the two attempts -- Stage 2 never discards
            # adaptation work in favor of a worse retry
            count, cur_thr = new_count, new_thr
        if new_count >= floor:
            logger.success(f"  [Stage 2] {cam_label} retry {attempt}/{max_retries} "
                            f"pseudo-label count: {new_count} — clears floor")
            count = new_count
            break
        logger.info(f"  [Stage 2] {cam_label} retry {attempt}/{max_retries} "
                    f"pseudo-label count: {new_count} — still below floor")

    if count < floor:
        logger.warn(f"  [Stage 2] {cam_label} still below floor after "
                    f"{attempt} retr{'y' if attempt == 1 else 'ies'} — keeping "
                    f"last attempt's result (final count: {count}) and continuing")

    report_entry["stage2_steps"] = steps
    report_entry["stage2_retries"] = attempt
    report_entry["final_pseudo_label_count"] = count
    report_entry["final_threshold_used"] = cur_thr
    return result


def _resolve_calib_toml(session: dict, source_folders: list[Path], logger) -> Path:
    """
    Resolve the calibration.toml to use for this pipeline run.

    Auto-detects a calibration recording under source_folders (any folder
    whose name contains "calibration" holding >=2 camN-prefixed videos —
    matches Step 0's own _calibration/{session_id}/ convention, and any other
    such folder a user places under a source folder) and auto-generates
    calibration.toml from it via calibrate_cameras.auto_calibrate(). This is
    the primary path. The manually-set dlc_3d_calib_folder session key is
    only a backup, used when auto-calibration is off, finds nothing, or
    fails outright. Resume-safe: reuses an existing
    "{candidate}/calibration_auto/calibration.toml" instead of recalibrating.

    Resolution priority (highest first):
      A0. "{candidate}/calibration_auto/calibration.toml" for the
          auto-detected calibration recording -- CUBE's own validated
          anchor-only/auto-K/robust-filter pipeline output. Always wins,
          regardless of what else is lying around under source_folders.
      A.  Any OTHER pre-existing calibration.toml found under source_folders
          (newest by mtime) -- e.g. a manually-run calibrate_cameras.py CLI
          output placed somewhere non-standard. Only reached when A0 doesn't
          exist yet.
      B.  Fresh auto_calibrate() run on the detected candidate recording.
      C.  The manually-set dlc_3d_calib_folder session key.
    """
    calib_toml = None
    autocalib_enabled = bool(session.get("dlc_3d_autocalib_enabled", True))
    if autocalib_enabled:
        # Step A0 (highest priority): the canonical auto-calibration cache
        # for the auto-detected calibration recording -- "{candidate}/
        # calibration_auto/calibration.toml". This is the file CUBE's own
        # validated anchor-only/auto-K/robust-filter pipeline produced, and
        # is always preferred over anything the broad mtime-based scan below
        # might find, regardless of which file is newer on disk.
        #
        # Before this check existed, Step A (below) picked "whichever
        # calibration.toml anywhere under source_folders has the newest
        # mtime" -- a real, hit bug: an unrelated stray/manual/test
        # calibration.toml (e.g. dropped in a scratch subfolder while
        # experimenting with calibration options) silently outranked and
        # shadowed the real, accurate, auto-generated calibration purely by
        # having a later mtime, with no warning that a "better" file existed
        # right next to it. See D:\3D DLC\CLAUDE.md's "Calibration Accuracy
        # — Anchor-Only Pipeline" section for the real-data incident this
        # traces back to.
        try:
            from calibrate_cameras import (find_calibration_candidates,
                                            pick_best_calibration_candidate)
            _candidates = find_calibration_candidates(source_folders)
            _best = pick_best_calibration_candidate(_candidates)
            if _best is not None:
                _auto_toml = _best / "calibration_auto" / "calibration.toml"
                if _auto_toml.exists():
                    logger.info(
                        f"  [auto-calib] Using canonical auto-calibration cache "
                        f"(highest priority, ahead of any other calibration.toml "
                        f"found under the source folders): {_auto_toml}")
                    return _auto_toml
        except Exception as e:
            logger.warn(f"  [auto-calib] Canonical-cache pre-check failed "
                        f"(falling back to normal resolution): {e}")

        # Step A: Scan source_folders (and parent directories) for pre-existing calibration.toml files
        for sf in source_folders:
            sf_path = Path(sf)
            if not sf_path.is_dir():
                continue
            cands = list(sf_path.rglob("calibration.toml"))
            for p in [sf_path.parent, sf_path.parent.parent]:
                if p.is_dir() and str(p) != str(sf_path.anchor) and p != p.parent:
                    cands.extend(list(p.glob("calibration*/calibration.toml")))
                    cands.extend(list(p.glob("_calibration/*/calibration.toml")))
                    cands.extend(list(p.glob("calibration.toml")))
            if cands:
                unique_cands = []
                seen_paths = set()
                for c in cands:
                    try:
                        resolved = c.resolve()
                        if resolved not in seen_paths and c.exists():
                            seen_paths.add(resolved)
                            unique_cands.append(c)
                    except Exception:
                        pass
                if unique_cands:
                    unique_cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    calib_toml = unique_cands[0]
                    logger.info(f"  [auto-calib] Found existing calibration.toml under source folder: {calib_toml}")
                    return calib_toml

        # Step B: Auto-calibration on raw video recordings under source_folders
        try:
            from calibrate_cameras import (find_calibration_candidates,
                                            pick_best_calibration_candidate,
                                            auto_calibrate)
            candidates = find_calibration_candidates(source_folders)
            best = pick_best_calibration_candidate(candidates)
            if best is not None:
                auto_out = best / "calibration_auto"
                auto_toml = auto_out / "calibration.toml"
                if auto_toml.exists():
                    calib_toml = auto_toml
                    logger.info(f"  [auto-calib] Using cached auto-calibration: {calib_toml}")
                elif (best / "calibration.toml").exists():
                    calib_toml = best / "calibration.toml"
                    logger.info(f"  [auto-calib] Using existing calibration in candidate folder: {calib_toml}")
                else:
                    logger.info(f"  [auto-calib] Detected calibration recording: {best}")
                    calib_toml = auto_calibrate(
                        best, auto_out,
                        squares_x=int(session.get("dlc_3d_calib_board_squares_x", 8)),
                        squares_y=int(session.get("dlc_3d_calib_board_squares_y", 6)),
                        square_length=float(session.get(
                            "dlc_3d_calib_board_square_length_mm", 14.88)),
                        marker_length=float(session.get(
                            "dlc_3d_calib_board_marker_length_mm", 11.16)),
                        marker_bits=int(session.get("dlc_3d_calib_board_marker_bits", 4)),
                        dict_size=int(session.get("dlc_3d_calib_board_dict_size", 50)),
                        pooled=bool(session.get("dlc_3d_autocalib_pooled", False)),
                        shared_intrinsics=bool(session.get("dlc_3d_shared_intrinsics", True)),
                        shared_intrinsics_reference=(
                            str(session["dlc_3d_shared_intrinsics_reference"])
                            if session.get("dlc_3d_shared_intrinsics_reference") else None),
                        log_fn=logger.info)
                    logger.success(f"  [auto-calib] Wrote {calib_toml}")
            else:
                logger.info(
                    "  [auto-calib] No calibration recording detected under the source "
                    "folders (looked for a folder with \"calibration\" in its name "
                    "containing >=2 camN-prefixed videos) — falling back to the "
                    "manually-set Calibration folder.")
        except Exception as e:
            logger.warn(f"  [auto-calib] Auto-calibration failed, falling back to the "
                        f"manually-set Calibration folder: {e}")
            calib_toml = None

    # Step C: Fallback to manually-set Calibration folder (dlc_3d_calib_folder)
    if calib_toml is None:
        calib_folder_str = str(session.get("dlc_3d_calib_folder", "")).strip()
        if calib_folder_str:
            calib_folder = Path(calib_folder_str)
            if calib_folder.is_file() and calib_folder.name == "calibration.toml":
                calib_toml = calib_folder
            elif calib_folder.is_dir():
                manual_toml = calib_folder / "calibration.toml"
                sub_manual = calib_folder / "calibration" / "calibration.toml"
                if manual_toml.exists():
                    calib_toml = manual_toml
                elif sub_manual.exists():
                    calib_toml = sub_manual
                else:
                    rglob_tomls = list(calib_folder.rglob("calibration.toml"))
                    if rglob_tomls:
                        rglob_tomls.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        calib_toml = rglob_tomls[0]

                # If no calibration.toml found, check if manual folder holds raw camera videos and auto-calibrate
                if calib_toml is None:
                    try:
                        from calibrate_cameras import discover_camera_videos, auto_calibrate
                        discover_camera_videos(calib_folder)
                        auto_out = calib_folder / "calibration_auto"
                        logger.info(f"  Running auto-calibration on manually-set Calibration folder: {calib_folder}")
                        calib_toml = auto_calibrate(
                            calib_folder, auto_out,
                            squares_x=int(session.get("dlc_3d_calib_board_squares_x", 8)),
                            squares_y=int(session.get("dlc_3d_calib_board_squares_y", 6)),
                            square_length=float(session.get(
                                "dlc_3d_calib_board_square_length_mm", 14.88)),
                            marker_length=float(session.get(
                                "dlc_3d_calib_board_marker_length_mm", 11.16)),
                            marker_bits=int(session.get("dlc_3d_calib_board_marker_bits", 4)),
                            dict_size=int(session.get("dlc_3d_calib_board_dict_size", 50)),
                            pooled=bool(session.get("dlc_3d_autocalib_pooled", False)),
                            shared_intrinsics=bool(session.get("dlc_3d_shared_intrinsics", True)),
                            shared_intrinsics_reference=(
                                str(session["dlc_3d_shared_intrinsics_reference"])
                                if session.get("dlc_3d_shared_intrinsics_reference") else None),
                            log_fn=logger.info)
                    except Exception as e:
                        logger.warn(f"  Manual calibration folder auto-calibration failed: {e}")

            if calib_toml is not None:
                logger.info(f"  Using manually-set calibration folder (backup): {calib_toml}")

    if calib_toml is None:
        raise FileNotFoundError(
            "No calibration available: auto-calibration found no calibration "
            "recording under the source folders, and no calibration.toml exists "
            "in the manually-set Calibration folder "
            f"({session.get('dlc_3d_calib_folder', '') or '(not set)'}).\n"
            "Either add a calibration recording (a folder with \"calibration\" in "
            "its name, containing camN-prefixed videos) under a source folder, or "
            "set a manual Calibration folder in the 3D DLC tab as a backup.")

    return calib_toml


def _find_camera_h5(mp4: Path, prep_root: Path | None = None) -> Path | None:
    """
    Locate the *_filtered.h5 written for a corrected MP4 by
    _infer_camera_videos(). Checks prep_root/{stem}_results/ first (this
    run's "CUBE 1b - 3D DLC/_prep/" folder, where _infer_camera_videos now
    always writes when given a dest_root) if prep_root is given, then falls
    back to mp4.parent/{stem}_results/ (the legacy alongside-the-source-
    video location, either for a pre-refactor H5 or when prep_root wasn't
    given).
    """
    stem = mp4.stem  # e.g. cam0_30fps2026-06-23T16_46_24_corrected
    base = stem

    search_dirs = []
    if prep_root is not None:
        search_dirs.append(Path(prep_root) / f"{stem}_results")
    search_dirs.append(mp4.parent / f"{stem}_results")

    for results_dir in search_dirs:
        filtered = results_dir / f"{base}_filtered.h5"
        if filtered.exists():
            return filtered
        # Fallback: any *_filtered.h5 in the results dir
        cands = list(results_dir.glob("*_filtered.h5"))
        if cands:
            return cands[0]
    return None


def _offset_h5_for_calibration(h5_path: Path, x_off: float, y_off: float) -> Path:
    """
    Return a coordinate-shifted copy of h5_path with (x_off, y_off) added to
    every bodypart's x/y column, cached alongside the original (skip-if-
    exists; named so it never matches _find_camera_h5's "*_filtered.h5" glob).

    When a camera's video was cropped before DLC inference, its H5 is in
    cropped-frame coordinates; calibration refinement (Phase 2.5) needs full,
    uncropped-frame coordinates to match calibration.toml's own space, same
    as triangulation does in-memory in _triangulate_session().
    """
    if x_off == 0.0 and y_off == 0.0:
        return h5_path
    out_path = h5_path.parent / f"{h5_path.stem}_full_frame.h5"
    if out_path.exists():
        return out_path

    import pandas as pd
    df = pd.read_hdf(str(h5_path))
    scorer = df.columns.get_level_values(0)[0]
    bodyparts = df[scorer].columns.get_level_values(0).unique().tolist()
    for bp in bodyparts:
        df[(scorer, bp, "x")] = df[(scorer, bp, "x")] + x_off
        df[(scorer, bp, "y")] = df[(scorer, bp, "y")] + y_off

    tmp = out_path.with_suffix(".tmp.h5")
    df.to_hdf(str(tmp), key="df_with_missing", mode="w", format="fixed")
    tmp.replace(out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  Triangulation (per session)
# ─────────────────────────────────────────────────────────────────────────────

def _default_smooth_stat(pts3d: "np.ndarray"):
    """
    Reproduce aniposelib's own `default_smooth` statistic (1 / mean absolute
    frame-to-frame displacement, after NaN-interpolation + a size-7 median
    filter) so it can be recomputed on a cleaned copy of the data. Returns
    None if there is no usable frame-to-frame signal at all.
    """
    from aniposelib.cameras import interpolate_data, medfilt_data
    p3ds_intp = np.apply_along_axis(interpolate_data, 0, pts3d)
    p3ds_med = np.apply_along_axis(medfilt_data, 0, p3ds_intp, size=7)
    diffs = np.abs(np.diff(p3ds_med, axis=0))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    return 1.0 / diffs.mean()


def _degeneracy_corrected_scale_smooth(pts3d: "np.ndarray",
                                        degenerate_mask: "np.ndarray",
                                        scale_smooth_multiplier: float = 4.0,
                                        logger=None) -> float:
    """
    Correct CameraGroup.optim_points()'s smoothing-strength calibration for
    contamination by exact zero-fill artifacts.

    optim_points() sizes its temporal-smoothness penalty from THIS session's
    own frame-to-frame jitter (`default_smooth = 1/mean(|diff|)`), which is a
    sound design in general — tying the penalty to the data's own motion
    scale keeps it balanced against the (also data-scale-dependent)
    reprojection-error term, and that jitter statistic is already implicitly
    frame-rate-aware: the same real velocity covers less distance per frame
    at a higher fps, so `default_smooth` shrinks automatically at higher fps
    and grows at lower fps with no explicit fps term needed.

    (An earlier version of this function instead anchored to a fixed
    biological velocity ceiling in mm/s and derived scale_smooth directly
    from fps — a synthetic test showed that collapses the smoothing weight
    to ~1/200th of the library's own tested default, because it discards the
    relative-scale link to the reprojection-error term entirely. Reverted.)

    The one real flaw: optim_points() has no notion of `degenerate_mask`
    (this session's exact-zero triangulation-failure frames, <2 valid camera
    views — 45.8% of slots measured on real CUBE data), so those zeros are
    fed into its own jitter statistic as if they were genuine positions,
    diluting `default_smooth` and under-smoothing exactly where the data is
    worst. This recomputes the same statistic with degenerate slots masked
    to NaN (bridged by the same interpolation optim_points already applies to
    real NaN gaps) and rescales the multiplier by the resulting ratio, so a
    session with heavy degeneracy gets correspondingly stronger smoothing
    instead of weaker.
    """
    raw = _default_smooth_stat(pts3d)
    if raw is None or not degenerate_mask.any():
        return scale_smooth_multiplier

    pts3d_clean = pts3d.copy()
    pts3d_clean[degenerate_mask] = np.nan
    clean = _default_smooth_stat(pts3d_clean)
    if clean is None:
        return scale_smooth_multiplier

    scale_smooth = scale_smooth_multiplier * (clean / raw)

    if logger:
        logger.info(f"    degeneracy-corrected temporal smoothing: "
                    f"{100*degenerate_mask.mean():.1f}% degenerate slots, "
                    f"scale_smooth {scale_smooth_multiplier:.3g} -> {scale_smooth:.3g}")
    return float(np.clip(scale_smooth, 0.5, 50.0))


def _two_link_expected_position(A: "np.ndarray", C: "np.ndarray",
                                 raw_M: "np.ndarray", L1: float, L2: float,
                                 d_max: "float | None" = None
                                 ) -> "tuple[np.ndarray, np.ndarray]":
    """
    Given two well-tracked endpoints A, C of a two-rod chain A-M-C with known
    (session-learned) rod lengths L1=|A-M|, L2=|M-C|, compute the point on
    the constraint circle (the intersection of a sphere of radius L1 around A
    and a sphere of radius L2 around C — the standard two-link/"elbow"
    geometry) closest to `raw_M`'s side of the A-C baseline.

    `raw_M` only disambiguates WHICH side of the A-C line the point sits on
    (its direction from the baseline) — its distance/magnitude is not
    trusted, since that's exactly the noisy quantity this function exists to
    correct. This means a real, if noisy, raw detection still contributes
    useful information (the general arch direction is far more robust to
    triangulation error than the exact position), while the actual position
    on the circle comes entirely from the two trustworthy endpoints.

    d_max: an additional, session-data-derived hard ceiling on the A-C
    distance (typically a generous multiple of the 99th percentile observed
    among genuinely well-tracked frames) — needed because the formulaic
    `L1+L2`-based slack below isn't tight enough to catch the case where A or
    C *itself* has a rare, wild triangulation outlier (the same single-
    camera "confidently wrong" failure pattern documented for `nose`):
    confirmed on real data (2026-08-27) that even well-tracked anchor joints
    occasionally triangulate to absurd distances apart (up to ~1800mm on a
    ~100mm mouse) for isolated frames, and without this check those frames
    fall through the coarser slack test as "valid" and get assigned a wildly
    wrong circle position instead of being correctly left untouched.

    A, C, raw_M: (n_frames, 3). Returns (expected_M, valid) where `valid` is
    False for frames where A/C aren't both finite, are coincident, exceed
    `d_max` (if given), or the triangle inequality with L1/L2 is too badly
    violated to trust (more than 50% beyond the feasible A-C distance range)
    — those frames are left for the caller to handle (typically: fall back
    to raw_M unchanged).
    """
    d_vec = C - A
    d = np.linalg.norm(d_vec, axis=-1)
    finite = np.all(np.isfinite(A), axis=-1) & np.all(np.isfinite(C), axis=-1)
    lo, hi = abs(L1 - L2), L1 + L2
    slack = 0.5 * hi
    valid = finite & (d > 1e-6) & (d > lo - slack) & (d < hi + slack)
    if d_max is not None:
        valid = valid & (d <= d_max)

    d_safe = np.where(d > 1e-6, d, 1.0)
    d_clipped = np.clip(d_safe, max(lo, 1e-6), max(hi, lo + 1e-6))
    u = d_vec / d_safe[..., None]

    a = (d_clipped ** 2 + L1 ** 2 - L2 ** 2) / (2 * d_clipped)
    h = np.sqrt(np.maximum(L1 ** 2 - a ** 2, 0.0))
    center = A + a[..., None] * u

    raw_dir = raw_M - center
    raw_dir_perp = raw_dir - np.sum(raw_dir * u, axis=-1, keepdims=True) * u
    perp_norm = np.linalg.norm(raw_dir_perp, axis=-1)

    # Carry the last well-defined direction forward across frames where
    # raw_M gives no usable side information (e.g. it's exactly zero-filled
    # at the origin) — cheaper and safer than an arbitrary fixed axis, since
    # the animal's own arch direction changes slowly frame-to-frame.
    perp_unit = np.zeros_like(raw_dir_perp)
    last_dir = np.array([0.0, 0.0, 1.0])
    for i in range(len(perp_norm)):
        if perp_norm[i] > 1e-6:
            last_dir = raw_dir_perp[i] / perp_norm[i]
        perp_unit[i] = last_dir

    expected_M = center + h[..., None] * perp_unit
    return expected_M, valid


# Bodyparts this prior must never be pointed at as `mid_bp`, regardless of
# what SPINE_GEOMETRIC_PRIOR_TARGETS says — a hard rule, not just a
# documentation note, so a future addition to that list can't silently
# reintroduce this mistake. The whole technique rests on two assumptions
# that free limb end-effectors (paws, and the equivalent on any other
# quadruped schema) violate: (1) roughly fixed rod lengths to BOTH
# neighbours — a paw's own position is one more degree of freedom past the
# knee/ankle, not pinned between two rigid segments the way an interior
# joint is; (2) a slowly-changing bend/arch direction, which the carry-
# forward disambiguation in _two_link_expected_position relies on — a paw
# swings through its full range of motion in a handful of frames during
# normal locomotion, far faster than a spine's posture changes, so a stale
# carried-forward direction would be wrong far more often here than for any
# of the validated targets. In practice this is moot for the current chain
# topology (a paw is always a chain LEAF, never a two-neighbour "mid" node,
# so it could never be selected anyway) but is kept as an explicit,
# keyword-based rejection so it stays true if the chain topology changes.
TWO_LINK_PRIOR_FORBIDDEN_MID_KEYWORDS = (
    "paw", "hand", "foot", "digit", "toe", "finger", "claw",
)


def _apply_two_link_spine_prior(pts3d: "np.ndarray", combined_ll: "np.ndarray",
                                 degenerate_mask: "np.ndarray",
                                 bp_index: dict, mid_bp: str,
                                 end_a_bp: str, end_b_bp: str,
                                 ll_gate: float, logger=None,
                                 min_learn_frames: int = 20,
                                 min_overall_tracked_frac: float = 0.05,
                                 max_jitter_ratio: float = 3.0,
                                 min_reference_jitter_pairs: int = 20) -> "np.ndarray":
    """
    Confidence-weighted geometric prior for a single "pass-through" spine
    point (`mid_bp`, e.g. back_middle) whose own 2D detections are
    chronically weak, using its two skeleton-chain neighbours (`end_a_bp`,
    `end_b_bp`, e.g. neck_base/back_end) when THEY are well-tracked.

    Unlike `dlc_3d_bone_constraints` (aniposelib's optim_points, which only
    ties each segment's LENGTH to be consistent across time), this treats
    A-mid_bp-end_b as a two-rod chain and computes where mid_bp can
    physically be *this frame* given the two rods' learned lengths and the
    endpoints' current positions — a per-frame position constraint, not a
    temporal one. The two mechanisms are complementary and both run when
    both are enabled.

    Blend: `final = w*raw + (1-w)*expected`, where `w` is mid_bp's own
    per-frame combined likelihood (already gate/degenerate-aware — see
    _triangulate_session). A frame degenerate for mid_bp (w=0, no real 2D
    evidence at all) gets the full geometric estimate instead of the
    "collapses to world origin" artifact zero-fill otherwise produces;
    combined_ll itself is NOT modified here — a geometrically-inferred
    position is never reported as a confident detection (matches the
    existing degenerate-likelihood-zeroing convention).

    Rod lengths L1/L2 are learned as the median |A-mid|/|mid-C| distance
    over frames where mid_bp, end_a_bp and end_b_bp are ALL simultaneously
    at or above `ll_gate` and non-degenerate — i.e. only from mid_bp's own
    genuinely reliable frames. If fewer than `min_learn_frames` such frames
    exist, the whole prior is skipped for this session (logged) rather than
    learning unreliable rod lengths from too little data.

    Two automatic guardrails, generalized from real mistakes made and
    caught by hand during this technique's own validation (2026-08-27, see
    CLAUDE.md), so a future candidate doesn't have to rediscover them:

    1. `mid_bp` is refused outright (logged, `pts3d` unchanged) if it
       matches `TWO_LINK_PRIOR_FORBIDDEN_MID_KEYWORDS` — free limb
       end-effectors (paws etc.) violate this technique's core assumptions
       (see that constant's docstring) regardless of how the caller's
       target list is configured.
    2. `mid_bp` is refused if its own OVERALL non-degenerate rate (not just
       the rate co-occurring with well-tracked A/C, which `min_learn_frames`
       already checks) falls below `min_overall_tracked_frac`. This is the
       generalized version of the back_middle mistake: back_middle passed a
       naive "some real tracking beats none" bar but was actually only
       genuinely tracked 1.44% of the time, versus 14-56% for the 6 targets
       that were validated to work well — with real detections that sparse,
       the carry-forward direction goes stale across enormous gaps (up to
       1041 consecutive frames observed) before it can ever be refreshed.
       The default (5%) sits well above back_middle's rate and well below
       every validated target's.

    A third check runs AFTER computing the correction, not before: there is
    no ground truth to verify the filled-in positions against, but there is
    a self-consistent proxy that needs no ground truth — mid_bp's OWN
    observed frame-to-frame motion on the frames where it genuinely has real
    detections. If the newly-filled frames move dramatically more erratically
    than that (`max_jitter_ratio`x the median real-frame jitter), the
    correction is producing implausible motion for THIS specific bodypart
    and is rejected wholesale (pts3d returned unchanged, logged) rather than
    trusting a plausible-looking mechanism that happens not to work for this
    particular target. This is the same "measure it, reject if it doesn't
    actually help" pattern as calibration_refine_poses.py's Stage B gate —
    applied here per-bodypart instead of per-calibration-file. Skipped (not
    rejected) if fewer than `min_reference_jitter_pairs` real consecutive
    frame pairs exist to measure a meaningful reference from.
    """
    if mid_bp not in bp_index or end_a_bp not in bp_index or end_b_bp not in bp_index:
        return pts3d

    mid_bp_lower = mid_bp.lower()
    if any(kw in mid_bp_lower for kw in TWO_LINK_PRIOR_FORBIDDEN_MID_KEYWORDS):
        if logger:
            logger.warn(f"    two-link spine prior: refusing to target {mid_bp} — "
                        f"matches a forbidden free-end-effector keyword "
                        f"(paws/hands/feet move too fast and freely for this "
                        f"technique's fixed-rod/slow-bend assumptions)")
        return pts3d

    mi, ai, ci = bp_index[mid_bp], bp_index[end_a_bp], bp_index[end_b_bp]
    M, A, C = pts3d[:, mi, :], pts3d[:, ai, :], pts3d[:, ci, :]
    ll_m, ll_a, ll_c = combined_ll[:, mi], combined_ll[:, ai], combined_ll[:, ci]

    overall_tracked_frac = float((~degenerate_mask[:, mi]).mean())
    if overall_tracked_frac < min_overall_tracked_frac:
        if logger:
            logger.info(f"    two-link spine prior for {mid_bp}: only "
                        f"{100*overall_tracked_frac:.2f}% of frames have any "
                        f"real evidence at all (<{100*min_overall_tracked_frac:.0f}%) "
                        f"— too sparse to keep the carry-forward direction "
                        f"fresh across the resulting gaps, skipping")
        return pts3d

    well_tracked = ((ll_m >= ll_gate) & (ll_a >= ll_gate) & (ll_c >= ll_gate)
                    & ~degenerate_mask[:, mi] & ~degenerate_mask[:, ai]
                    & ~degenerate_mask[:, ci])
    n_learn = int(well_tracked.sum())
    if n_learn < min_learn_frames:
        if logger:
            logger.info(f"    two-link spine prior for {mid_bp}: only "
                        f"{n_learn} frames with {mid_bp}/{end_a_bp}/{end_b_bp} "
                        f"all well-tracked (<{min_learn_frames}) — skipping")
        return pts3d

    L1 = float(np.median(np.linalg.norm(M[well_tracked] - A[well_tracked], axis=-1)))
    L2 = float(np.median(np.linalg.norm(M[well_tracked] - C[well_tracked], axis=-1)))

    # Session-data-derived ceiling on the A-C distance, well above what any
    # genuine well-tracked frame shows (99th percentile x1.5) — see
    # _two_link_expected_position's `d_max` docstring for why the L1/L2-based
    # slack test alone isn't tight enough to catch a rare wild outlier in A
    # or C themselves.
    d_well_tracked = np.linalg.norm((C - A)[well_tracked], axis=-1)
    d_max = float(np.percentile(d_well_tracked, 99) * 1.5)

    # A degenerate mid_bp is exactly (0, 0, 0) (the zero-fill artifact), NOT
    # "no information" in the geometric sense _two_link_expected_position
    # relies on for its near-zero-perpendicular-component carry-forward
    # fallback: the vector from a MOVING constraint-circle center to a FIXED
    # world origin swings arbitrarily frame-to-frame as the animal moves,
    # which the disambiguation step would otherwise treat as real "which
    # side of the spine" signal. Confirmed on real data (2026-08-27,
    # back_middle, session 439158_3_48C_0000): with the origin fed straight
    # in, frame-to-frame jitter got ~3x WORSE (median 0.95mm -> 3.08mm, p95
    # 8.8mm -> 73.5mm) instead of better, entirely from this spurious
    # per-frame direction swinging around. Feeding NaN instead correctly
    # routes degenerate frames through the carry-forward path.
    M_for_side = M.copy()
    M_for_side[degenerate_mask[:, mi]] = np.nan

    expected_M, geo_valid = _two_link_expected_position(A, C, M_for_side, L1, L2, d_max=d_max)

    w_raw = np.clip(ll_m, 0.0, 1.0)
    blended = w_raw[..., None] * M + (1.0 - w_raw[..., None]) * expected_M
    out = pts3d.copy()
    out[geo_valid, mi, :] = blended[geo_valid]

    # A degenerate mid_bp frame where the geometry is ALSO momentarily
    # invalid (A or C's own triangulation strayed out of a plausible range
    # for that specific frame, e.g. d_max above) has nothing good to fall
    # back on: raw M is the (0, 0, 0) zero-fill artifact, not a real
    # detection. Confirmed on real data (2026-08-27, right_earbase and other
    # candidates): leaving such frames at that raw artifact — even though
    # they're a small minority within an otherwise-long degenerate run —
    # produces sharp "snap to origin, snap back" jumps between them and
    # their (correctly-computed, on-circle) neighbours, which dominated the
    # mean/p95 jitter despite a solid median (e.g. right_earbase: median
    # 1.3mm, but mean 19.7mm / p95 239mm from these isolated snaps). Carrying
    # the last successfully-computed position forward instead (real
    # detection or prior geometric estimate, whichever is more recent) holds
    # continuity through these isolated bad-geometry frames rather than
    # reintroducing the origin artifact this whole function exists to avoid.
    need_fill = degenerate_mask[:, mi] & ~geo_valid
    if need_fill.any():
        last_valid = None
        for i in range(len(out)):
            if not need_fill[i]:
                last_valid = out[i, mi, :]
            elif last_valid is not None:
                out[i, mi, :] = last_valid

    # Verify the correction actually helped, using the one ground-truth-free
    # signal available: mid_bp's OWN observed frame-to-frame motion on
    # frames where it has genuine detections. If the newly-filled frames
    # move far more erratically than that, this specific target isn't a
    # good fit for the technique's assumptions (even though it passed every
    # precondition above) and the whole correction is rejected rather than
    # trusted on faith — same "measure it, reject if it doesn't help"
    # pattern as calibration_refine_poses.py's Stage B gate.
    real_mask = ~degenerate_mask[:, mi]
    real_pairs = real_mask[:-1] & real_mask[1:]
    ref_jitter = np.linalg.norm(np.diff(M, axis=0), axis=-1)[real_pairs]
    if len(ref_jitter) >= min_reference_jitter_pairs:
        filled_pairs = ~real_pairs
        filled_jitter = np.linalg.norm(np.diff(out[:, mi, :], axis=0), axis=-1)[filled_pairs]
        if len(filled_jitter) > 0:
            ref_median = max(float(np.median(ref_jitter)), 1e-6)
            filled_median = float(np.median(filled_jitter))
            ratio = filled_median / ref_median
            if ratio > max_jitter_ratio:
                if logger:
                    logger.warn(
                        f"    two-link spine prior for {mid_bp}: REJECTED — filled-frame "
                        f"jitter ({filled_median:.3f}) is {ratio:.1f}x this bodypart's own "
                        f"real-detection jitter ({ref_median:.3f}), exceeding the "
                        f"{max_jitter_ratio:.1f}x threshold; pts3d left unchanged")
                return pts3d

    if logger:
        n_applied = int(geo_valid.sum())
        n_carried = int(need_fill.sum())
        logger.info(f"    two-link spine prior for {mid_bp} (rods "
                    f"{end_a_bp}={L1:.1f}, {end_b_bp}={L2:.1f}, d_max={d_max:.1f}, "
                    f"learned from {n_learn} frames): applied to "
                    f"{n_applied}/{len(geo_valid)} frames, carried-forward "
                    f"{n_carried} degenerate-and-geometry-invalid frames")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Head-orientation geometric gate
# ─────────────────────────────────────────────────────────────────────────────

# Reference bodyparts used to estimate the animal's head heading in 3D.
# neck_base and upper_jaw are, by a wide margin, the two best-tracked points
# in the whole SuperAnimal-Quadruped skeleton on this rig (22px / 34-38px
# median reprojection error vs. nose's 130-167px — see NOSE_TRACKING_ISSUE_
# REPORT.md and the backbone-redesign entry in CLAUDE.md's Crash Protection
# table), so they're used as a robust proxy for "which way is the face
# pointing" rather than trying to derive heading from the ambiguous point
# this gate exists to protect. Not GUI-exposed — a schema-specific
# implementation detail, like SPINE_GEOMETRIC_PRIOR_TARGETS' own anchors.
HEAD_ORIENT_GATE_REF_BODYPARTS = ("neck_base", "upper_jaw")


def _estimate_head_heading(pts2d: "np.ndarray", scores: "np.ndarray",
                            bp_index: dict, calib_toml: Path,
                            ll_gate: float, logger=None
                            ) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | None":
    """
    Estimate the animal's head-heading unit vector per frame from
    HEAD_ORIENT_GATE_REF_BODYPARTS = (neck_base, upper_jaw), by triangulating
    just those two bodyparts (cheap — 2 of ~25, not the full skeleton) through
    the same triangulate_cameras() every other bodypart uses, so this can't
    silently diverge from the rest of the pipeline's calibration handling.

    Returns (heading, head_pos, valid): heading/head_pos are (n_frames, 3)
    (heading = normalized upper_jaw-minus-neck_base direction; head_pos =
    neck_base position, used as the bearing-vector origin), valid is
    (n_frames,) bool. Returns None if either reference bodypart is missing
    from this session's schema — feature no-ops for the whole session,
    matching detect_turned_away_bins' "nose not in bodyparts" convention.
    """
    from cube_core import triangulate_cameras
    ref_a, ref_b = HEAD_ORIENT_GATE_REF_BODYPARTS
    if ref_a not in bp_index or ref_b not in bp_index:
        if logger:
            logger.warn(f"    head-orientation gate: '{ref_a}'/'{ref_b}' not both "
                        f"present in this session's bodyparts — skipping")
        return None

    ref_idx = [bp_index[ref_a], bp_index[ref_b]]
    pts2d_ref = pts2d[:, :, ref_idx, :]
    scores_ref = scores[:, :, ref_idx]
    pts3d_ref = triangulate_cameras(pts2d_ref, scores_ref, calib_toml, ll_gate=ll_gate,
                                     log_fn=(logger.info if logger else print))
    # pts3d_ref: (n_frames, 2, 3) -> [neck_base, upper_jaw]
    neck = pts3d_ref[:, 0, :]
    jaw = pts3d_ref[:, 1, :]

    degenerate = (np.all(np.abs(pts3d_ref) < 1e-6, axis=-1)
                  | np.any(np.isnan(pts3d_ref), axis=-1))  # (n_frames, 2)
    both_valid = ~degenerate[:, 0] & ~degenerate[:, 1]

    d_vec = jaw - neck
    d = np.linalg.norm(d_vec, axis=-1)
    valid = both_valid & (d > 1e-6)
    d_safe = np.where(d > 1e-6, d, 1.0)
    heading = d_vec / d_safe[..., None]
    return heading, neck, valid


def _resolve_camera_disagreement_via_orientation(
        pts2d: "np.ndarray", scores: "np.ndarray", pts3d: "np.ndarray",
        bp_index: dict, present_cams: list, calib_toml: Path,
        target_bodyparts: list, ll_gate: float, disagreement_px: float,
        min_turned_away_deg: float, min_angle_margin_deg: float,
        sid: str, logger=None) -> "np.ndarray":
    r"""
    Real-data-driven redesign (2026-08-27) of what was originally an
    unconditional per-frame angle gate — see the "Default changed to OFF"
    comment on dlc_3d_head_orient_gate in cube.py for why that version was
    reverted: a face is visible across a very wide range of viewing angles
    (profile views included), so "large heading-vs-camera angle" alone does
    NOT mean "this camera can't see the face" — applying that test to every
    frame regardless of whether the cameras even disagreed made nose
    measurably WORSE (degenerate rate ~7-9% -> 40-68%, jitter roughly
    doubled-to-tripled), confirmed at every threshold from 80-120 degrees on
    both real sessions in F:\3D CAM TEST\TRY 3.

    This version only ever acts as a TIE-BREAKER, not a blanket filter:
    1. Only looks at frames where the cameras that cleared ll_gate actually
       DISAGREE on target_bodyparts's position (per-camera reprojection error
       against the already-triangulated point exceeds disagreement_px) — a
       frame where every camera agrees is left completely untouched,
       regardless of viewing angle.
    2. Among only the disagreeing cameras for that frame, uses body
       orientation (heading = neck_base->upper_jaw vs. camera bearing) as a
       RELATIVE comparison, not an absolute cutoff: the camera with the
       largest angle is excluded only if it clearly stands out from the rest
       (> min_turned_away_deg AND at least min_angle_margin_deg ahead of the
       next-most-turned-away valid camera) — i.e. "this specific camera looks
       like the one seeing the back of the head, and it isn't a close call."
       If no camera stands out this way, the frame is left unresolved rather
       than guessed at.
    3. Re-triangulates target_bodyparts using only the surviving cameras for
       exactly the frames that got a camera excluded, and splices the result
       back into pts3d — everything else (bone constraints, combined-
       likelihood aggregation, degenerate-mask logic) runs on the corrected
       pts3d afterward exactly as if this had been the original triangulation.

    Requires >=3 cameras clearing ll_gate on a frame before it's even
    considered (can't drop to <2 and still triangulate; 2 cameras that
    disagree with no third to break the tie has no basis for picking one).

    Returns pts3d (same array, corrected in place for the resolved frames;
    also returned for call-site clarity). Never raises — any missing
    reference bodypart or all-frames-clean session is a logged no-op.
    """
    from cube_core import triangulate_cameras, camera_world_positions

    heading_est = _estimate_head_heading(pts2d, scores, bp_index, calib_toml,
                                          ll_gate, logger=logger)
    if heading_est is None:
        return pts3d
    heading, head_pos, heading_valid = heading_est
    if not heading_valid.any():
        if logger:
            logger.warn(f"  [{sid}] head-orientation disagreement resolution: "
                        f"no frames with a valid head heading — skipping")
        return pts3d

    target_idx = [bp_index[bp] for bp in target_bodyparts if bp in bp_index]
    if not target_idx:
        if logger:
            logger.warn(f"  [{sid}] head-orientation disagreement resolution: "
                        f"none of {target_bodyparts} present — skipping")
        return pts3d

    from aniposelib.cameras import CameraGroup
    cgroup = CameraGroup.load(str(calib_toml))
    cam_pos = camera_world_positions(cgroup)
    calib_names = cgroup.get_names()
    cams_by_name = dict(zip(calib_names, cgroup.cameras))
    n_cams, n_frames = pts2d.shape[0], pts2d.shape[1]
    eps = 1e-6

    for bi, bp_name in zip(target_idx, [b for b in target_bodyparts if b in bp_index]):
        p3d_bp = pts3d[:, bi, :]                          # (n_frames, 3)
        valid_gate = scores[:, :, bi] >= ll_gate           # (n_cams, n_frames)

        reproj_err = np.full((n_cams, n_frames), np.nan)
        for ci, cam_name in enumerate(present_cams):
            cam = cams_by_name.get(cam_name)
            if cam is None:
                continue
            err_vec = cam.reprojection_error(p3d_bp, pts2d[ci, :, bi, :])
            err = np.linalg.norm(err_vec.reshape(n_frames, -1), axis=-1)
            reproj_err[ci] = np.where(valid_gate[ci], err, np.nan)

        n_valid = np.sum(~np.isnan(reproj_err), axis=0)
        with np.errstate(invalid="ignore"):
            max_err = np.nanmax(np.where(n_valid[None, :] > 0, reproj_err, np.nan), axis=0)
        candidate_frames = np.where((n_valid >= 3) & (max_err > disagreement_px)
                                     & heading_valid)[0]
        if candidate_frames.size == 0:
            continue

        exclude_mask = np.zeros((n_cams, n_frames), dtype=bool)
        for f in candidate_frames:
            angles = []
            for ci, cam_name in enumerate(present_cams):
                if np.isnan(reproj_err[ci, f]):
                    continue
                c_world = cam_pos[calib_names.index(cam_name)]
                bearing = c_world - head_pos[f]
                bn = np.linalg.norm(bearing)
                if bn < eps:
                    continue
                cosang = np.clip(np.dot(heading[f], bearing / bn), -1.0, 1.0)
                angles.append((ci, np.degrees(np.arccos(cosang))))
            if len(angles) < 3:
                continue
            angles.sort(key=lambda t: t[1])
            max_ci, max_ang = angles[-1]
            second_ang = angles[-2][1]
            if max_ang > min_turned_away_deg and (max_ang - second_ang) > min_angle_margin_deg:
                exclude_mask[max_ci, f] = True

        if not exclude_mask.any():
            continue

        pts2d_bp = pts2d[:, :, [bi], :].copy()
        scores_bp = scores[:, :, [bi]].copy()
        scores_bp[exclude_mask, 0] = 0.0
        pts2d_bp[exclude_mask, 0, :] = np.nan

        pts3d_bp_new = triangulate_cameras(pts2d_bp, scores_bp, calib_toml,
                                            ll_gate=ll_gate,
                                            log_fn=(logger.info if logger else print))[:, 0, :]
        affected = np.where(exclude_mask.any(axis=0))[0]
        pts3d[affected, bi, :] = pts3d_bp_new[affected]

        if logger:
            logger.info(f"  [{sid}] Camera-disagreement resolution for {bp_name}: "
                        f"{len(candidate_frames)} disagreeing frames checked, "
                        f"{len(affected)} had a clear turned-away outlier camera "
                        f"excluded and re-triangulated")

    return pts3d


# ─────────────────────────────────────────────────────────────────────────────
#  Hard anatomical bone-length gate (2026-09-09, TMZ7 double-board calibration
#  investigation follow-up — see D:\3D DLC\CLAUDE.md's TMZ7 triangulation
#  handoff)
# ─────────────────────────────────────────────────────────────────────────────

# True physical maxima (mm) — unlike SPINE_GEOMETRIC_PRIOR_TARGETS' d_max (a
# session-learned 99th-percentile guess), these hold regardless of how much
# good tracking data a given session has: an ear-end genuinely cannot be more
# than ~12mm from its own ear-base, and a thigh-to-knee or knee-to-paw segment
# cannot exceed ~50mm on this animal (direct physical measurement, 2026-09-09).
# Keyed by an ordered bodypart pair; both bodyparts must be present in a
# session's schema for a pair to apply (silent no-op otherwise, matching every
# other schema-conditional feature in this module).
HARD_BONE_LENGTH_MAX_MM: dict = {
    ("left_earbase", "left_earend"): 12.0,
    ("right_earbase", "right_earend"): 12.0,
    ("front_left_thai", "front_left_knee"): 50.0,
    ("front_left_knee", "front_left_paw"): 50.0,
    ("front_right_thai", "front_right_knee"): 50.0,
    ("front_right_knee", "front_right_paw"): 50.0,
    ("back_left_thai", "back_left_knee"): 50.0,
    ("back_left_knee", "back_left_paw"): 50.0,
    ("back_right_thai", "back_right_knee"): 50.0,
    ("back_right_knee", "back_right_paw"): 50.0,
}


def _apply_hard_bone_length_gate(pts3d: "np.ndarray", raw_conf: "np.ndarray",
                                  bp_index: dict, sid: str = "", logger=None
                                  ) -> "np.ndarray":
    """
    NaN out whichever endpoint of a HARD_BONE_LENGTH_MAX_MM pair is less
    confident that frame, on any frame where the pair's triangulated distance
    exceeds a true anatomical maximum. Runs BEFORE dlc_3d_bone_constraints'
    optim_points() call, so a wild single-frame triangulation error is
    excluded from the constraint solve entirely rather than relying on
    optim_points()'s own soft length-consistency term to catch it after the
    fact — confirmed insufficient on real TMZ7 data (an 18,838mm nose-eye
    violation survived bone_constraints intact).

    Returns a NEW pts3d array (input not mutated).
    """
    out = pts3d.copy()
    n_frames = out.shape[0]
    for (a, b), max_mm in HARD_BONE_LENGTH_MAX_MM.items():
        if a not in bp_index or b not in bp_index:
            continue
        ia, ib = bp_index[a], bp_index[b]
        A, B = out[:, ia, :], out[:, ib, :]
        finite = np.all(np.isfinite(A), axis=-1) & np.all(np.isfinite(B), axis=-1)
        dist = np.full(n_frames, np.nan)
        dist[finite] = np.linalg.norm(A[finite] - B[finite], axis=-1)
        bad = finite & (dist > max_mm)
        if not bad.any():
            continue
        conf_a, conf_b = raw_conf[:, ia], raw_conf[:, ib]
        drop_a = bad & (conf_a <= conf_b)
        drop_b = bad & ~drop_a
        out[drop_a, ia, :] = np.nan
        out[drop_b, ib, :] = np.nan
        if logger:
            logger.info(f"  [{sid}] Hard bone-length gate {a}-{b} "
                        f"(max {max_mm:.0f}mm): {int(bad.sum())} frame(s) exceeded "
                        f"ceiling, weaker endpoint NaN'd")
    return out


# Session-learned torso-hub bone-length gate.
#
# HARD_BONE_LENGTH_MAX_MM above only covers the ear and distal-limb segments
# (thai-knee, knee-paw) -- it deliberately omits the torso "hub" segments
# (neck_base-back_end, back_end-thai, neck_base-thai) because a fixed
# physical-mm ceiling isn't appropriate there: unlike a rigid limb bone, the
# torso genuinely flexes with posture (arching/hunching/stretching), so its
# own length varies session-to-session and even frame-to-frame in ways the
# animal's own anatomy doesn't bound the way it bounds an ear or forearm.
#
# Found 2026-09-13, real-data investigation on MOUSE 1 48C_0000 (F:\TMZ7,
# manual frame-by-frame video review requested by the user): neck_base-
# back_end drifted smoothly from a stable ~50mm to ~113mm over roughly a
# dozen frames (frames ~753-780 of this session), dragging the entire
# dependent hindlimb chain along with it via dlc_3d_bone_constraints (every
# hindlimb bodypart's own per-frame displacement-clamp check stayed under
# threshold, since the drift's own per-frame delta was gradual -- this bug
# is invisible to a single-frame check and only shows up as an anatomically
# impossible CUMULATIVE segment length). Not an isolated one-off: the same
# signature recurred at least 3 more times in this one ~2-minute recording
# (frame ~218, ~2134-2135, and a second sustained ~40-frame episode at
# ~2722-2763), confirmed via direct neck_base-to-back_end distance
# inspection across the whole session.
#
# FIRST FIX ATTEMPT (2026-09-13, superseded same day): a static per-session
# ceiling (95th-percentile x 1.35) was tried first and found insufficient --
# it only catches the PEAK of a smooth multi-frame drift, not the ramp
# itself. Frames in the middle of a ~753-780 real episode (e.g. 50-81mm)
# sat under the static ceiling individually and survived ungated, and
# dlc_3d_bone_constraints' optim_points() then interpolated the handful of
# NaN'd peak frames using those still-corrupted neighbours as anchors,
# largely reconstructing the same bad trend the gate was meant to remove
# (confirmed directly: post-gate segment length at frames 758-764 was still
# 88-99mm, barely different from the pre-gate 84-113mm). Replaced below with
# a rolling-baseline version -- the same architectural fix already proven
# for _apply_displacement_clamp's own "must compare against real frames on
# BOTH sides, not a single absolute threshold" lesson -- applied to segment
# LENGTH instead of raw position.
#
# Window is wide enough that even the longest real corrupted episode found
# on this data (~42 consecutive frames, ~1.4s at 30fps) stays a clear
# minority of the window (TORSO_SEGMENT_ROLLING_WINDOW_S=2.5s each side ->
# ~150 frames total at 30fps, so a 42-frame episode is <30% of the window --
# comfortably under the ~50% contamination a median can tolerate). Delta
# ceiling (25mm) is a flat physically-reasoned floor, not session-learned:
# unlike the old percentile-based ceiling, the LOCAL median baseline this
# compares against already tracks the segment's own natural genuine
# variation (posture, animal size), so the delta only needs to be wide
# enough to not clip ordinary local flex, not to characterize the whole
# session's range.
TORSO_SEGMENT_LEARNED_GATE_PAIRS: list = [
    ("neck_base", "back_end"),
]
TORSO_SEGMENT_ROLLING_WINDOW_S = 2.5
TORSO_SEGMENT_ROLLING_DELTA_MM = 25.0
TORSO_SEGMENT_ROLLING_MIN_WINDOW_VALID = 15


def _apply_learned_bone_length_gate(pts3d: "np.ndarray", raw_conf: "np.ndarray",
                                     bp_index: dict, fps: float = 30.0,
                                     sid: str = "", logger=None
                                     ) -> "np.ndarray":
    """
    Like _apply_hard_bone_length_gate, but for TORSO_SEGMENT_LEARNED_GATE_PAIRS
    -- segments whose real length genuinely varies with posture, so a fixed
    physical-mm ceiling isn't appropriate. Rejects (NaNs the weaker-confidence
    endpoint of) any frame whose segment length deviates from a symmetric
    rolling-median baseline (built from real, finite frames on BOTH sides,
    window = +/-TORSO_SEGMENT_ROLLING_WINDOW_S seconds) by more than
    TORSO_SEGMENT_ROLLING_DELTA_MM -- see the comment above this dict for why
    a rolling baseline replaced an earlier static-ceiling attempt.

    Returns a NEW pts3d array (input not mutated).
    """
    out = pts3d.copy()
    window = max(3, int(round(TORSO_SEGMENT_ROLLING_WINDOW_S * max(fps, 1e-6))))
    for a, b in TORSO_SEGMENT_LEARNED_GATE_PAIRS:
        if a not in bp_index or b not in bp_index:
            continue
        ia, ib = bp_index[a], bp_index[b]
        A, B = out[:, ia, :], out[:, ib, :]
        finite = np.all(np.isfinite(A), axis=-1) & np.all(np.isfinite(B), axis=-1)
        conf_a, conf_b = raw_conf[:, ia], raw_conf[:, ib]
        idx_valid = np.where(finite)[0]
        n_valid = len(idx_valid)
        if n_valid < TORSO_SEGMENT_ROLLING_MIN_WINDOW_VALID:
            continue
        dist_valid = np.linalg.norm(A[idx_valid] - B[idx_valid], axis=-1)

        bad_frames = []
        for k in range(n_valid):
            f = idx_valid[k]
            lo = np.searchsorted(idx_valid, f - window, side="left")
            hi = np.searchsorted(idx_valid, f + window, side="right")
            if hi - lo < TORSO_SEGMENT_ROLLING_MIN_WINDOW_VALID:
                continue
            local = np.delete(dist_valid[lo:hi], k - lo)
            baseline = np.median(local)
            if abs(dist_valid[k] - baseline) > TORSO_SEGMENT_ROLLING_DELTA_MM:
                bad_frames.append(f)
        if not bad_frames:
            continue
        bad_frames = np.asarray(bad_frames)
        weaker_a = conf_a[bad_frames] <= conf_b[bad_frames]
        drop_a = bad_frames[weaker_a]
        drop_b = bad_frames[~weaker_a]
        out[drop_a, ia, :] = np.nan
        out[drop_b, ib, :] = np.nan
        if logger:
            logger.info(f"  [{sid}] Learned torso-segment gate {a}-{b} "
                        f"(rolling +/-{TORSO_SEGMENT_ROLLING_WINDOW_S:.1f}s baseline, "
                        f"delta>{TORSO_SEGMENT_ROLLING_DELTA_MM:.0f}mm): "
                        f"{len(bad_frames)} frame(s) exceeded local baseline, "
                        f"weaker endpoint NaN'd")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Generic frame-to-frame displacement clamp
# ─────────────────────────────────────────────────────────────────────────────

def _apply_displacement_clamp(pts3d: "np.ndarray", degenerate_mask: "np.ndarray",
                               fps: float, window: int = 5,
                               sid: str = "", logger=None
                               ) -> "tuple[np.ndarray, np.ndarray]":
    """
    Reject (NaN) any (frame, bodypart) whose position deviates from a short,
    symmetric rolling-median baseline (built from real, non-degenerate
    neighbouring frames on BOTH sides — past and future, not just the
    previous frame) by more than a physically-motivated ceiling:

        d_max(fps) = 20mm + 2400/fps

    Derived from two real reference points (10cm max jump at 30fps, 6cm at
    60fps): decomposes exactly into a fixed 20mm positional-noise floor
    (calibration/detection jitter, independent of frame rate) plus a hard
    ~2.4 m/s maximum real velocity term (2400mm/s / fps = max real distance
    covered in one frame). A rolling-median baseline (not a plain adjacent-
    frame diff) is used specifically so a multi-frame decaying bad-triangulation
    run — confirmed on real TMZ7 data to decay gradually across several
    frames, not spike-and-recover in one — gets caught in full rather than
    only at its two endpoints.

    Only ever tightens the data (NaNs a violating point); a genuinely fast
    real movement that stays within d_max of its own local window median is
    left untouched.

    Returns (new_pts3d, clamped_mask) — clamped_mask marks every (frame,
    bodypart) this function itself NaN'd.
    """
    out = pts3d.copy()
    n_frames, n_bp, _ = out.shape
    d_max = 20.0 + 2400.0 / max(fps, 1e-6)
    clamped = np.zeros((n_frames, n_bp), dtype=bool)
    valid = ~degenerate_mask

    for bi in range(n_bp):
        v = valid[:, bi]
        idx_valid = np.where(v)[0]
        n_valid = len(idx_valid)
        if n_valid < 3:
            continue
        pos_valid = out[idx_valid, bi, :]
        for k in range(n_valid):
            lo = max(0, k - window)
            hi = min(n_valid, k + window + 1)
            if hi - lo < 3:
                continue
            local = np.delete(pos_valid[lo:hi], k - lo, axis=0)
            baseline = np.median(local, axis=0)
            d = np.linalg.norm(pos_valid[k] - baseline)
            if d > d_max:
                f = idx_valid[k]
                out[f, bi, :] = np.nan
                clamped[f, bi] = True

    if logger and clamped.any():
        logger.info(f"  [{sid}] Displacement clamp (d_max={d_max:.1f}mm @ fps): "
                    f"{int(clamped.sum())} frame-bodypart slot(s) exceeded local "
                    f"rolling baseline, NaN'd")
    return out, clamped


# ─────────────────────────────────────────────────────────────────────────────
#  Nose multi-anchor position prior
# ─────────────────────────────────────────────────────────────────────────────

# `lower_jaw` was dropped 2026-09-09 after real TMZ7 data (all 6 sessions,
# post-calibration-fix) showed it tracked at only 69.1% overall -- WORSE than
# `nose` itself (90.9%), defeating the whole premise of using it as a
# rod-length anchor (an anchor should be a MORE trustworthy reference than
# the point it's helping recover, not less). left_eye/right_eye (95.9-96.2%)
# and upper_jaw (85.0%) all comfortably clear that bar. The generic
# min_overall_tracked_frac floor in _apply_nose_multipoint_prior() (5%) was
# designed to catch a catastrophically bad candidate (the historical
# back_middle, at 1.44%) -- not one that's merely worse than its peers -- so
# this needed a direct real-data comparison, not just that guardrail.
NOSE_MULTIPOINT_ANCHOR_CANDIDATES = ("left_eye", "right_eye", "upper_jaw")

# Generalizes the nose-only rigid-head recovery above to the rest of the
# face's RIGID cluster (nose/eyes/upper_jaw/earbases sit on a skull that can
# only translate/rotate as a whole -- see _rigid_head_seed_for_nose's
# docstring). lower_jaw is deliberately never a target OR a candidate here:
# it hinges relative to the skull, and that hinge is exactly the degree of
# freedom licking/gape needs to show up in the data -- rigidifying or
# bone-length-constraining it to the rest of the head would smooth away the
# behavioural signal itself, not just noise (see D:\3D DLC\CLAUDE.md's
# "Bodypart Informativeness By Behavior" note). Defaults on
# (dlc_3d_face_rigid_cluster_enabled, 2026-09-18) -- each entry reuses the
# identical, already-tested _flag_anatomically_inconsistent_nose/
# _apply_nose_multipoint_prior functions with nose_bp/candidates
# parametrized differently, not new code, so upper_jaw inherits the same
# validated behavior as nose's own mechanism rather than an unproven one.
FACE_RIGID_CLUSTER_EXTRA_TARGETS = (
    ("upper_jaw", ("nose", "left_eye", "right_eye")),
)

# The empirically weak, non-RIGID tier (TMZ7 real-data audit, 2026-09-17:
# baseline-vs-48C tracked-fraction drop of 16-25% relative, consistent across
# all 3 animals) -- articulated joints (thigh/knee) that genuinely bend, plus
# lower_jaw, which hinges relative to the skull and so can never join
# FACE_RIGID_CLUSTER_EXTRA_TARGETS's rigid-recovery mechanism above. No rigid
# shortcut exists for any of these (unlike the face cluster), so the only
# available lever is a second, more permissive ray+temporal-fallback pass
# scoped to just this tier -- see dlc_3d_weak_tier_extra_recovery_enabled.
# earbase/earend are deliberately excluded: earbase already has its own
# rigid-cluster/bone-length handling, and neither is diagnostic for any
# behaviour in the pain/OFT ethogram reference (see D:\3D DLC\CLAUDE.md's
# "Bodypart Informativeness By Behavior" note) -- not worth the extra
# recovery risk on a bodypart nothing downstream actually uses.
WEAK_TIER_NON_RIGID_BODYPARTS = (
    "lower_jaw",
    "front_left_thai", "front_right_thai", "back_left_thai", "back_right_thai",
    "front_left_knee", "front_right_knee", "back_left_knee", "back_right_knee",
)


def _multipoint_expected_position(P0: "np.ndarray", anchor_pos_list: list,
                                   anchor_valid_list: list, rod_lengths: list,
                                   n_iters: int = 12, damping: float = 0.01
                                   ) -> "tuple[np.ndarray, np.ndarray]":
    """
    Vectorized Gauss-Newton multilateration: solves for P minimizing
    sum_i (|P - A_i| - L_i)^2 over only the anchors valid that frame, across
    ALL frames simultaneously in a handful of fixed iterations (not a
    per-frame scipy.optimize call — this runs on thousands of frames and
    needs to stay fast). P0 seeds the solve and disambiguates a poorly-
    constrained (<3 valid anchors) frame, mirroring
    _two_link_expected_position's use of raw_M for the same purpose.

    A small fixed damping term keeps the per-frame 3x3 normal-equations
    matrix invertible even when fewer than 3 non-collinear anchors are
    valid (frames with 0-1 valid anchors barely move from P0; the caller
    is expected to reject those via the returned n_valid_anchors count).

    Returns (P, n_valid_anchors).
    """
    P = P0.copy()
    anchor_pos = np.stack(anchor_pos_list, axis=0)        # (n_anchor, n_frames, 3)
    anchor_valid = np.stack(anchor_valid_list, axis=0)    # (n_anchor, n_frames)
    rods = np.asarray(rod_lengths, dtype=float)           # (n_anchor,)
    n_valid_anchors = anchor_valid.sum(axis=0)            # (n_frames,)
    w = anchor_valid.astype(float)                        # (n_anchor, n_frames)
    n_frames = P0.shape[0]
    eye3 = np.eye(3)

    for _ in range(n_iters):
        diff = P[None, :, :] - anchor_pos                 # (n_anchor, n_frames, 3)
        dist = np.linalg.norm(diff, axis=-1)               # (n_anchor, n_frames)
        dist_safe = np.where(dist > 1e-6, dist, 1e-6)
        unit = diff / dist_safe[..., None]                  # (n_anchor, n_frames, 3)
        resid = (dist - rods[:, None]) * w                  # (n_anchor, n_frames)
        JTJ = np.einsum('af,afi,afj->fij', w, unit, unit) + damping * eye3
        JTr = np.einsum('af,afi,af->fi', w, unit, resid)
        # Explicit trailing singleton dim on the RHS -- NumPy 2.x's
        # linalg.solve batching no longer auto-detects a 2D b as a stack of
        # vectors against a 3D batched a; giving it (n_frames, 3, 1) forces
        # the unambiguous (m,m),(m,n)->(m,n) batched-matrix path instead.
        delta = np.linalg.solve(JTJ, (-JTr)[..., None])[..., 0]  # (n_frames, 3)
        P = P + delta
    return P, n_valid_anchors


def _linear_multilateration_seed(anchor_pos_list: list, anchor_valid_list: list,
                                  rod_lengths: list) -> "np.ndarray":
    """
    Closed-form (non-iterative) least-squares seed for the Gauss-Newton
    multilateration above. Needed because GN is a LOCAL method: point-to-
    sphere multilateration with exactly 3 non-collinear anchors has a
    genuine 2-fold reflection ambiguity (both roots satisfy every distance
    constraint exactly), and starting GN from a noisy or zero-filled raw
    triangulation (the only thing available on a degenerate frame — the
    actual case this whole prior exists for) can converge to the wrong
    root. Confirmed directly: GN started exactly at the true position stays
    there, but a modest (std=5mm) perturbation was enough to converge
    several mm off in a synthetic 3-anchor test.

    Derivation: subtracting each valid anchor's own sphere equation
    (|P-A_i|^2 = L_i^2) from their common MEAN eliminates the quadratic
    |P|^2 term, leaving one linear equation in P per valid anchor:
        2(A_i - mean(A)) . P = (|A_i|^2 - L_i^2) - mean(|A|^2 - L^2)
    Solved by weighted least squares across however many anchors are valid
    that frame (fully determined at 4 anchors, a well-conditioned
    approximate solution at 3 — still far better than an arbitrary seed).
    Vectorized across all frames at once via batched np.linalg.solve.
    """
    anchor_pos = np.stack(anchor_pos_list, axis=0)        # (n_anchor, n_frames, 3)
    anchor_valid = np.stack(anchor_valid_list, axis=0)    # (n_anchor, n_frames)
    rods = np.asarray(rod_lengths, dtype=float)           # (n_anchor,)
    w = anchor_valid.astype(float)                        # (n_anchor, n_frames)
    n_valid = w.sum(axis=0)
    safe_n = np.where(n_valid > 0, n_valid, 1.0)

    mean_A = np.einsum('af,afi->fi', w, anchor_pos) / safe_n[:, None]        # (n_frames, 3)
    sq = np.sum(anchor_pos ** 2, axis=-1) - rods[:, None] ** 2               # (n_anchor, n_frames)
    mean_sq = np.einsum('af,af->f', w, sq) / safe_n                          # (n_frames,)

    A_mat = 2.0 * (anchor_pos - mean_A[None, :, :])       # (n_anchor, n_frames, 3)
    b_vec = sq - mean_sq[None, :]                          # (n_anchor, n_frames)

    JTJ = np.einsum('af,afi,afj->fij', w, A_mat, A_mat) + 1e-6 * np.eye(3)[None, :, :]
    JTb = np.einsum('af,afi,af->fi', w, A_mat, b_vec)
    seed = np.linalg.solve(JTJ, JTb[..., None])[..., 0]
    return seed


def _canonical_head_shape_for_nose(pts3d: "np.ndarray", degenerate_mask: "np.ndarray",
                                    bp_index: dict, nose_bp: str,
                                    candidates: tuple, min_co_visible: int = 20,
                                    min_handedness_ref_frames: int = 10,
                                    sid: str = "", logger=None):
    """
    Shared canonical-shape builder for the two rigid-head-geometry passes
    below (`_rigid_head_seed_for_nose`, `_flag_anatomically_inconsistent_
    nose`). Reuses `_mds_canonical_shape()` (the same technique already
    used for the skeleton VIDEO's rigid head rendering, `_rigidify_head_
    points`) to build a robust, session-wide canonical shape from `nose_bp`
    + `candidates`, then centers and handedness-corrects it for per-frame
    Kabsch fitting.

    Two real bugs found and fixed via a noiseless synthetic test before
    this was ever run on real data:
    1. Centering: the canonical shape must be centered on the ANCHORS' OWN
       centroid, not the full (nose+anchors) MDS shape's centroid — the
       per-frame Kabsch fit only ever has the anchors' OBSERVED positions
       to center against (nose is exactly what's missing), so using a
       different reference point for the canonical side introduces a
       spurious translation bias equal to nose's own offset from the
       anchor centroid. Left a systematic ~9.5mm error when centered on
       all 4 points instead of the 3 anchors alone.
    2. Handedness: classical MDS reconstructs a shape matching all
       pairwise distances, but distances are invariant to REFLECTION as
       well as rotation — MDS can hand back a mirror image of the true
       rigid shape, which would put nose on the wrong side of the anchor
       plane for every subsequent per-frame prediction (a real, separate
       ~10mm systematic error, distinct from bug 1, confirmed the same
       way). Fixed by checking nose's signed position relative to the
       anchor-plane normal against real co-visible reference frames
       (majority-sign vote) and mirroring nose's canonical position across
       that plane if they disagree — an exact reflection that preserves
       every canonical anchor-nose distance the MDS shape was built to
       match, so it costs nothing if handedness was already correct.

    Returns (nose_canon, anchor_canon, anchor_idx) or None if there isn't
    enough co-visible data to build a canonical shape at all.
    """
    cand_bps = [b for b in candidates if b in bp_index]
    if nose_bp not in bp_index or len(cand_bps) < len(candidates) or len(cand_bps) < 3:
        return None

    all_bps = [nose_bp] + cand_bps
    cand_idx = np.array([bp_index[b] for b in all_bps])
    visible = ~degenerate_mask

    mds_shape = _mds_canonical_shape(cand_idx, pts3d, visible, min_co_visible=min_co_visible)
    if mds_shape is None:
        if logger:
            logger.info(f"  [{sid}] nose rigid-head geometry: not enough co-visible "
                        f"frames across nose+anchors to build a canonical shape — skipping")
        return None

    anchor_canon_raw = mds_shape[1:]
    anchor_centroid = anchor_canon_raw.mean(axis=0)
    anchor_canon = anchor_canon_raw - anchor_centroid
    nose_canon = mds_shape[0] - anchor_centroid
    anchor_idx = cand_idx[1:]
    ni = bp_index[nose_bp]

    both_vis = visible[:, ni] & np.all(visible[:, anchor_idx], axis=1)
    n_ref = int(both_vis.sum())
    if n_ref >= min_handedness_ref_frames and len(anchor_canon) >= 3:
        e1 = anchor_canon[1] - anchor_canon[0]
        e2 = anchor_canon[2] - anchor_canon[0]
        normal = np.cross(e1, e2)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        canon_sign = np.sign(np.dot(nose_canon - anchor_canon[0], normal))

        ref_idx = np.where(both_vis)[0]
        real_anchors = pts3d[ref_idx][:, anchor_idx, :]   # (n_ref, n_anchor, 3)
        real_nose = pts3d[ref_idx, ni, :]                  # (n_ref, 3)
        e1r = real_anchors[:, 1, :] - real_anchors[:, 0, :]
        e2r = real_anchors[:, 2, :] - real_anchors[:, 0, :]
        normals_r = np.cross(e1r, e2r)
        normals_r = normals_r / (np.linalg.norm(normals_r, axis=1, keepdims=True) + 1e-12)
        signs_r = np.sign(np.einsum('ij,ij->i', real_nose - real_anchors[:, 0, :], normals_r))
        majority_sign = np.sign(np.sum(signs_r))

        if canon_sign != 0 and majority_sign != 0 and canon_sign != majority_sign:
            d_signed = np.dot(nose_canon - anchor_canon[0], normal)
            nose_canon = nose_canon - 2 * d_signed * normal
            if logger:
                logger.info(f"  [{sid}] nose rigid-head geometry: corrected a mirrored "
                            f"MDS canonical shape (handedness fixed using {n_ref} real "
                            f"co-visible reference frames)")

    return nose_canon, anchor_canon, anchor_idx


def _rigid_head_seed_for_nose(pts3d: "np.ndarray", degenerate_mask: "np.ndarray",
                               bp_index: dict, nose_bp: str = "nose",
                               candidates: tuple = NOSE_MULTIPOINT_ANCHOR_CANDIDATES,
                               min_co_visible: int = 20, sid: str = "", logger=None
                               ) -> "tuple[np.ndarray, np.ndarray] | tuple[None, None]":
    """
    Recovers nose's position on a degenerate frame using a full rigid-body
    (Kabsch) fit, not just per-anchor distances — exploiting that, unlike a
    limb joint, a skull cannot bend, only translate/rotate as a whole, so
    the relative ANGULAR arrangement of nose's neighbours (not just their
    individual distances to nose) is real, exploitable information that
    pure multilateration discards entirely. See
    `_canonical_head_shape_for_nose()` for how the canonical shape is built
    and corrected.

    Deliberately NOT used to override an already-tracked nose position —
    that would repeat exactly the reasoning this codebase already rejected
    for applying a hard rigid-body transform to real analysis H5 data (a
    single reference frame's — or even a robust MDS shape's — geometry
    overriding a genuine detection is too aggressive for feature data; see
    CLAUDE.md's "Rigid head-shape constraint also added to real
    triangulation data" entry). Since this only ever fills a frame where
    nose has NO real detection to begin with, no genuine detection is ever
    overridden — strictly an upgrade over the zero-fill/distance-only
    alternative for exactly those frames, and inherently free of the
    reflection ambiguity `_multipoint_expected_position` has to guard
    against (a proper Kabsch fit with a determinant-sign correction always
    yields a true rotation, never a mirror).

    Requires nose_bp + ALL of `candidates` present in the schema and enough
    mutual co-visibility to build a canonical shape (else (None, None)).
    Per-frame, requires all of `candidates` visible simultaneously (the
    current 3-anchor set — after dropping the poorly-tracked lower_jaw — is
    exactly the Kabsch minimum for a unique 3D rotation, so there is no
    partial-subset fallback within this function; the caller falls back to
    distance-only multilateration for frames this can't solve).

    Returns (nose_est, solved_mask): nose_est is (n_frames, 3) (only valid
    where solved_mask is True), solved_mask is (n_frames,) bool.
    """
    shape = _canonical_head_shape_for_nose(
        pts3d, degenerate_mask, bp_index, nose_bp, candidates,
        min_co_visible=min_co_visible, sid=sid, logger=logger)
    if shape is None:
        return None, None
    nose_canon, anchor_canon, anchor_idx = shape

    n_frames = pts3d.shape[0]
    visible = ~degenerate_mask
    anchor_vis = visible[:, anchor_idx]     # (n_frames, n_anchor)
    n_anchor = len(anchor_idx)

    nose_est = np.full((n_frames, 3), np.nan)
    solved = np.zeros(n_frames, dtype=bool)
    for f in range(n_frames):
        if anchor_vis[f].sum() < n_anchor:   # require ALL candidates visible (== Kabsch minimum here)
            continue
        obs = pts3d[f, anchor_idx, :]
        obs_center = obs.mean(axis=0)
        obs_centered = obs - obs_center
        H = anchor_canon.T @ obs_centered
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        nose_est[f] = R @ nose_canon + obs_center
        solved[f] = True

    if logger:
        logger.info(f"    nose rigid-head prior: solvable on {int(solved.sum())}/"
                    f"{n_frames} frames (all {n_anchor} anchors simultaneously visible)")
    return nose_est, solved


def _flag_anatomically_inconsistent_nose(pts3d: "np.ndarray", degenerate_mask: "np.ndarray",
                                          bp_index: dict, nose_bp: str = "nose",
                                          candidates: tuple = NOSE_MULTIPOINT_ANCHOR_CANDIDATES,
                                          min_co_visible: int = 20,
                                          tolerance_mult: float = 4.0,
                                          tolerance_floor_mm: float = 5.0,
                                          min_reference_frames: int = 50,
                                          sid: str = "", logger=None
                                          ) -> "tuple[np.ndarray, np.ndarray]":
    """
    Detects a "confidently wrong" nose detection — a frame where nose has a
    real (non-degenerate, likelihood-clearing) triangulated position that is
    nonetheless anatomically inconsistent with the rest of the head. This is
    a documented, repeat real-data failure mode for nose specifically (a
    single camera's confidently-wrong detection, distinct from — and not
    caught by — a frame where nose simply has no data at all, which
    _rigid_head_seed_for_nose/_apply_nose_multipoint_prior already handle).

    Reuses the same rigid-body reasoning: since a skull can only rotate/
    translate as a whole, not bend, whenever ALL of `candidates` are
    simultaneously visible their observed positions alone predict where
    nose SHOULD be (via the session's own robust MDS canonical head shape —
    `_mds_canonical_shape`, median-based, so a minority of contaminated
    frames can't meaningfully distort it). A genuine detection should sit
    close to that prediction, give or take real small measurement noise;
    the "natural give" tolerance is LEARNED from this session's own
    well-tracked frames (their own actual-vs-predicted discrepancy)
    rather than assumed, then any frame exceeding tolerance_mult times that
    learned floor is flagged.

    Flagged frames have nose set to NaN (reclassified as degenerate) rather
    than corrected in place here — they flow through the exact same
    downstream recovery this pipeline already has for a genuinely missing
    nose (the caller recomputes degenerate_mask afterward; bone_constraints
    interpolates it, and _apply_nose_multipoint_prior/_rigid_head_seed_for_
    nose re-fill it from the anchors), rather than building a second,
    parallel "how do we blend a correction" mechanism.

    Returns (new_pts3d, flagged_mask).
    """
    n_frames = pts3d.shape[0]
    shape = _canonical_head_shape_for_nose(
        pts3d, degenerate_mask, bp_index, nose_bp, candidates,
        min_co_visible=min_co_visible, sid=sid, logger=logger)
    if shape is None:
        return pts3d, np.zeros(n_frames, dtype=bool)
    nose_canon, anchor_canon, anchor_idx = shape

    visible = ~degenerate_mask
    anchor_vis = visible[:, anchor_idx]
    ni = bp_index[nose_bp]
    nose_vis = visible[:, ni]
    n_anchor = len(anchor_idx)

    predicted = np.full((n_frames, 3), np.nan)
    solvable = np.zeros(n_frames, dtype=bool)
    for f in range(n_frames):
        if anchor_vis[f].sum() < n_anchor:
            continue
        obs = pts3d[f, anchor_idx, :]
        obs_center = obs.mean(axis=0)
        obs_centered = obs - obs_center
        H = anchor_canon.T @ obs_centered
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        predicted[f] = R @ nose_canon + obs_center
        solvable[f] = True

    check_mask = solvable & nose_vis
    if int(check_mask.sum()) < min_reference_frames:
        return pts3d, np.zeros(n_frames, dtype=bool)

    discrepancy = np.linalg.norm(pts3d[:, ni, :] - predicted, axis=-1)
    tolerance = max(float(np.percentile(discrepancy[check_mask], 90)) * tolerance_mult,
                     tolerance_floor_mm)
    flagged = check_mask & (discrepancy > tolerance)

    out = pts3d
    if flagged.any():
        out = pts3d.copy()
        out[flagged, ni, :] = np.nan
        if logger:
            logger.info(f"  [{sid}] Nose anatomical-consistency check: "
                        f"{int(flagged.sum())}/{int(check_mask.sum())} real-detection "
                        f"frames flagged as confidently-wrong (discrepancy > "
                        f"{tolerance:.1f}mm, learned from this session's own "
                        f"well-tracked frames) and marked degenerate for "
                        f"downstream recovery")
    return out, flagged


def _apply_nose_multipoint_prior(pts3d: "np.ndarray", degenerate_mask: "np.ndarray",
                                  bp_index: dict, nose_bp: str = "nose",
                                  candidates: tuple = NOSE_MULTIPOINT_ANCHOR_CANDIDATES,
                                  min_overall_tracked_frac: float = 0.05,
                                  min_learn_frames: int = 200,
                                  max_jitter_ratio: float = 3.0,
                                  min_reference_jitter_pairs: int = 20,
                                  d_max_mult: float = 1.5,
                                  sid: str = "", logger=None) -> "np.ndarray":
    """
    Multi-anchor position prior for `nose`, generalizing
    _apply_two_link_spine_prior's two-anchor rod-length constraint to the (up
    to) 4 candidate neighbours nose actually has on this schema: left_eye/
    right_eye/upper_jaw/lower_jaw. A genuine two-anchor prior can't be built
    for nose (SPINE_GEOMETRIC_PRIOR_TARGETS requires exactly 2 fixed
    neighbours forming a simple chain), but nose's neighbours form a
    non-collinear cluster around the face — exactly the geometry
    multilateration needs. More anchors than the minimum also make the
    result robust to any ONE anchor also being degenerate that frame, unlike
    the two-link prior (which has no fallback if either endpoint is bad).

    Learns each candidate's rod length (median nose-to-anchor distance among
    frames where nose ITSELF is well-tracked) and only uses candidates
    clearing min_overall_tracked_frac of overall frames — same "check real
    per-camera visibility before trusting a candidate" lesson as the
    back_middle exclusion (see CLAUDE.md's spine-geometric-prior history).
    Requires >=2 valid anchors per frame to attempt a fix.

    Same after-the-fact jitter-ratio guardrail as the two-link prior: rejects
    the WHOLE correction (returns pts3d unchanged) if the filled frames'
    jitter is more than max_jitter_ratio times nose's own observed
    real-detection jitter.
    """
    if nose_bp not in bp_index:
        return pts3d
    candidates = [a for a in candidates if a in bp_index and a != nose_bp]
    if len(candidates) < 2:
        return pts3d

    ni = bp_index[nose_bp]
    n_frames = pts3d.shape[0]
    nose_real = ~degenerate_mask[:, ni]
    if nose_real.sum() < min_learn_frames:
        if logger:
            logger.info(f"    nose multipoint prior: skipped — only "
                        f"{int(nose_real.sum())} well-tracked nose frames "
                        f"(< {min_learn_frames} minimum) to learn rod lengths from")
        return pts3d

    kept_anchors, rod_lengths, anchor_valid_arrays, anchor_pos_arrays = [], [], [], []
    for a in candidates:
        ai = bp_index[a]
        a_valid = ~degenerate_mask[:, ai]
        overall_frac = float(a_valid.mean())
        if overall_frac < min_overall_tracked_frac:
            if logger:
                logger.info(f"    nose multipoint prior: excluding candidate "
                            f"'{a}' — only {100*overall_frac:.1f}% overall tracked "
                            f"(< {100*min_overall_tracked_frac:.0f}% floor)")
            continue
        co_valid = nose_real & a_valid
        if co_valid.sum() < min_learn_frames:
            continue
        d = np.linalg.norm(pts3d[co_valid, ni, :] - pts3d[co_valid, ai, :], axis=-1)
        kept_anchors.append(a)
        rod_lengths.append(float(np.median(d)))
        anchor_valid_arrays.append(a_valid)
        anchor_pos_arrays.append(pts3d[:, ai, :])

    if len(kept_anchors) < 2:
        if logger:
            logger.info(f"    nose multipoint prior: skipped — fewer than 2 usable "
                        f"anchors survived (kept: {kept_anchors})")
        return pts3d

    # Seed GN with the closed-form linear multilateration estimate, not the
    # raw (possibly zero-filled/garbage) triangulation — see
    # _linear_multilateration_seed's docstring for why a bad seed can
    # converge GN to the wrong trilateration root on exactly the degenerate
    # frames this prior exists to fix.
    P0 = _linear_multilateration_seed(anchor_pos_arrays, anchor_valid_arrays, rod_lengths)
    P_fit, n_valid_anchors = _multipoint_expected_position(
        P0, anchor_pos_arrays, anchor_valid_arrays, rod_lengths)

    d_max_caps = []
    for a, rod, avalid in zip(kept_anchors, rod_lengths, anchor_valid_arrays):
        ai = bp_index[a]
        co_valid = nose_real & avalid
        if co_valid.sum() >= min_learn_frames:
            d = np.linalg.norm(pts3d[co_valid, ni, :] - pts3d[co_valid, ai, :], axis=-1)
            d_max_caps.append(np.percentile(d, 99) * d_max_mult)
    d_max_cap = min(d_max_caps) if d_max_caps else np.inf

    # Reject a frame's fit if it drifts too far from ANY anchor it claims to
    # satisfy (a poorly-conditioned GN solve, e.g. near-collinear anchors,
    # could otherwise converge to a spurious point).
    fit_ok = n_valid_anchors >= 2
    for rod, avalid, apos in zip(rod_lengths, anchor_valid_arrays, anchor_pos_arrays):
        d_fit = np.linalg.norm(P_fit - apos, axis=-1)
        bad = avalid & (np.abs(d_fit - rod) > d_max_cap)
        fit_ok = fit_ok & ~bad

    # Prefer the rigid-body (Kabsch) estimate wherever it's solvable — see
    # _rigid_head_seed_for_nose's docstring: a skull can only rotate/
    # translate as a whole (unlike a limb joint, which genuinely bends), so
    # its full angular arrangement is real, exploitable information the
    # distance-only GN fit above discards. Falls back to the GN fit only
    # for frames the rigid method can't solve (fewer than all `kept_anchors`
    # simultaneously visible).
    rigid_est, rigid_solved = _rigid_head_seed_for_nose(
        pts3d, degenerate_mask, bp_index, nose_bp=nose_bp,
        candidates=tuple(kept_anchors), logger=logger)

    P_final = P_fit
    if rigid_est is not None:
        use_rigid = rigid_solved & degenerate_mask[:, ni]
        P_final = np.where(use_rigid[:, None], rigid_est, P_fit)
        fit_ok = fit_ok | use_rigid

    need_fill = degenerate_mask[:, ni] & fit_ok
    out = pts3d.copy()
    out[need_fill, ni, :] = P_final[need_fill]

    if nose_real.sum() >= min_reference_jitter_pairs + 1 and need_fill.any():
        ref_jitter = np.linalg.norm(np.diff(pts3d[nose_real, ni, :], axis=0), axis=-1)
        fill_pairs = need_fill[1:] | need_fill[:-1]
        filled_jitter = np.linalg.norm(np.diff(out[:, ni, :], axis=0), axis=-1)[fill_pairs]
        if len(filled_jitter) > 0 and len(ref_jitter) >= min_reference_jitter_pairs:
            ref_median = max(float(np.median(ref_jitter)), 1e-6)
            ratio = float(np.median(filled_jitter)) / ref_median
            if ratio > max_jitter_ratio:
                if logger:
                    logger.warn(f"    nose multipoint prior: REJECTED — filled-frame "
                                f"jitter is {ratio:.1f}x nose's own real-detection "
                                f"jitter (> {max_jitter_ratio:.1f}x threshold); "
                                f"pts3d left unchanged")
                return pts3d

    if logger:
        logger.info(f"    nose multipoint prior (anchors {kept_anchors}, rods "
                    f"{[f'{r:.1f}' for r in rod_lengths]}): applied to "
                    f"{int(need_fill.sum())}/{n_frames} degenerate frames")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Ray + temporal fallback for 1-camera-only frames
# ─────────────────────────────────────────────────────────────────────────────

def _ray_temporal_fallback(pts2d: "np.ndarray", scores: "np.ndarray",
                            pts3d: "np.ndarray", degenerate_mask: "np.ndarray",
                            calib_toml: Path, present_cams: list, ll_gate: float,
                            max_gap_frames: int = 45,
                            max_ray_disagreement_mm: float = 60.0,
                            bp_subset_idx: "list | None" = None,
                            sid: str = "", logger=None
                            ) -> "tuple[np.ndarray, np.ndarray]":
    """
    Recovers a (frame, bodypart) that triangulate_cameras() zero-filled for
    having fewer than 2 valid camera views, specifically the common EXACTLY
    ONE camera clearing ll_gate case. Camera extrinsics are fixed within a
    session (the rig doesn't move), so that one camera's 2D detection always
    back-projects to a well-defined 3D ray, even though a single camera
    alone can't resolve depth along it. Depth is instead resolved from a
    short-window temporal estimate (linear interpolation between the
    nearest real, non-degenerate frames before and after): the point taken
    is whichever point on the ray sits closest to that temporal estimate.

    Deliberately conservative — this is a recovery path for otherwise
    fully-lost data, not a substitute for genuine multi-camera
    triangulation, and recovered points are NOT exempted from any later
    geometric check (hard bone-length gate, displacement clamp,
    bone_constraints all still run on them normally, exactly like any other
    point) since a ray/temporal guess can itself be wrong:
    - only attempts recovery when a real bracketing frame exists on BOTH
      sides within max_gap_frames frames (never extrapolates past the edge
      of a genuinely-tracked run)
    - rejects (leaves degenerate) if the ray and the temporal estimate
      disagree by more than max_ray_disagreement_mm — a sign the single
      confident camera's own detection is itself wrong that frame

    Recovered positions ALWAYS get likelihood=0 downstream regardless of
    whether they pass this function's own checks or any later one (see
    caller) — "a temporally-interpolated position is never reported as a
    confident detection" is an existing, deliberate convention in this
    pipeline (see dlc_3d_bone_constraints' own docs).

    bp_subset_idx: when given, only these bodypart column indices are
    considered for recovery (every other bodypart is left exactly as it
    was) -- lets a second, more permissive call target only the empirically
    weak, non-rigid tier (lower_jaw/thigh/knee) with wider gap/disagreement
    thresholds, without loosening recovery for bodyparts that don't need it.

    Returns (new_pts3d, recovered_mask).
    """
    import cv2
    from aniposelib.cameras import CameraGroup

    out = pts3d.copy()
    n_cams, n_frames, n_bp, _ = pts2d.shape
    recovered = np.zeros((n_frames, n_bp), dtype=bool)

    cgroup = CameraGroup.load(str(calib_toml))
    calib_names = cgroup.get_names()
    cams_by_name = dict(zip(calib_names, cgroup.cameras))

    cam_info = []
    for cam_name in present_cams:
        cam = cams_by_name.get(cam_name)
        if cam is None:
            cam_info.append(None)
            continue
        rvec = np.array(cam.get_rotation(), dtype=float).reshape(3)
        t = np.array(cam.get_translation(), dtype=float).reshape(3)
        R, _ = cv2.Rodrigues(rvec)
        C = -R.T @ t
        cam_info.append((R, C, cam.get_camera_matrix(), cam.get_distortions()))

    valid_gate = scores >= ll_gate                    # (n_cams, n_frames, n_bp)
    n_valid_cams = valid_gate.sum(axis=0)              # (n_frames, n_bp)
    one_cam_mask = degenerate_mask & (n_valid_cams == 1)

    _bp_range = range(n_bp) if bp_subset_idx is None else bp_subset_idx
    for bi in _bp_range:
        real = ~degenerate_mask[:, bi]
        real_idx = np.where(real)[0]
        if len(real_idx) < 2:
            continue
        candidates = np.where(one_cam_mask[:, bi])[0]
        if candidates.size == 0:
            continue
        for f in candidates:
            prev_pos = np.searchsorted(real_idx, f) - 1
            next_pos = np.searchsorted(real_idx, f)
            if prev_pos < 0 or next_pos >= len(real_idx):
                continue
            f_prev, f_next = int(real_idx[prev_pos]), int(real_idx[next_pos])
            if (f - f_prev) > max_gap_frames or (f_next - f) > max_gap_frames:
                continue
            if f_next == f_prev:
                continue
            w = (f - f_prev) / float(f_next - f_prev)
            temporal_est = (1 - w) * pts3d[f_prev, bi, :] + w * pts3d[f_next, bi, :]

            ci_arr = np.where(valid_gate[:, f, bi])[0]
            if ci_arr.size != 1:
                continue
            ci = int(ci_arr[0])
            info = cam_info[ci]
            if info is None:
                continue
            R, C, K, dist = info
            uv = pts2d[ci, f, bi, :].reshape(1, 1, 2).astype(np.float64)
            try:
                norm_pt = cv2.undistortPoints(uv, K, dist).reshape(2)
            except Exception:
                continue
            ray_dir = R.T @ np.array([norm_pt[0], norm_pt[1], 1.0])
            n = np.linalg.norm(ray_dir)
            if n < 1e-12:
                continue
            ray_dir = ray_dir / n

            t_star = np.dot(temporal_est - C, ray_dir)
            closest_on_ray = C + t_star * ray_dir
            disagreement = np.linalg.norm(closest_on_ray - temporal_est)
            if disagreement > max_ray_disagreement_mm:
                continue

            out[f, bi, :] = closest_on_ray
            recovered[f, bi] = True

    if logger and recovered.any():
        logger.info(f"  [{sid}] Ray + temporal fallback: recovered "
                    f"{int(recovered.sum())} otherwise-degenerate 1-camera "
                    f"frame-bodypart slot(s)")
    return out, recovered


def _triangulate_session(sess: dict, cam_labels: list[str],
                          calib_toml: Path, out_folder: Path,
                          ll_agg: str, logger,
                          use_ransac: bool = False,
                          ransac_threshold: float = 0.5,
                          confidence_weighted: bool = False,
                          gdop_weighted: bool = False,
                          gdop_weighted_bodyparts: "list | None" = None,
                          ll_threshold: float = 0.0,
                          ll_gate: float = 0.6,
                          median_window: int = 3,
                          bone_constraints: bool = True,
                          spine_geometric_prior: bool = False,
                          head_orient_gate: bool = False,
                          head_orient_gate_angle_deg: float = 135.0,
                          head_orient_gate_margin_deg: float = 20.0,
                          head_orient_gate_disagreement_px: float = 50.0,
                          head_orient_gate_bodyparts: "list | None" = None,
                          crop_rects: "dict | None" = None,
                          hard_bone_length_gate: bool = True,
                          torso_segment_gate: bool = True,
                          displacement_clamp: bool = True,
                          nose_multipoint_prior: bool = True,
                          face_rigid_cluster_enabled: bool = True,
                          ray_temporal_fallback: bool = True,
                          ray_temporal_max_gap_s: float = 1.5,
                          ray_temporal_max_ray_disagreement_mm: float = 60.0,
                          weak_tier_extra_recovery: bool = False,
                          weak_tier_bodyparts: tuple = WEAK_TIER_NON_RIGID_BODYPARTS,
                          weak_tier_ray_temporal_max_gap_s: float = 3.0,
                          weak_tier_ray_temporal_max_ray_disagreement_mm: float = 90.0,
                          hindpaw_floor_pin: bool = True,
                          hindpaw_floor_pin_max_lift_s: float = 10.0,
                          hindpaw_floor_pin_max_dual_lift_s: float = 0.2,
                          turned_away_percam_gate: bool = True) -> Path:
    """
    Triangulate all cameras for one session.

    Writes {out_folder}/{session_id}_3d_filtered.h5 (atomic).
    Returns the output H5 path.

    crop_rects: {cam_label: {"x","y","w","h"}} — when a camera was cropped
    before DLC inference (see _crop_camera_video), its H5 coordinates are
    relative to the cropped frame; the (x, y) offset is added back here so
    pts2d lines up with calibration.toml's full, uncropped pixel space.

    torso_segment_gate: session-learned analogue of hard_bone_length_gate for
    TORSO_SEGMENT_LEARNED_GATE_PAIRS (currently neck_base-back_end) -- segments
    whose real length varies with posture, so no fixed physical-mm ceiling is
    appropriate; see _apply_learned_bone_length_gate's own comment. Runs
    immediately after hard_bone_length_gate, before bone_constraints. Added
    2026-09-13.

    hard_bone_length_gate / displacement_clamp / nose_multipoint_prior /
    ray_temporal_fallback: four data-quality passes added 2026-09-09
    following the TMZ7 double-sided-calibration-board investigation (see
    HARD_BONE_LENGTH_MAX_MM, _apply_displacement_clamp,
    _apply_nose_multipoint_prior, _ray_temporal_fallback above). Applied in
    this order: ray+temporal fallback recovers otherwise fully-degenerate
    1-camera frames FIRST (so the recovered — and possibly messier — points
    are subject to every check below exactly like real data, never
    exempted); then the hard bone-length gate and displacement clamp both
    run before dlc_3d_bone_constraints' optim_points() call, so a violating
    point is excluded from that solve rather than relying on its soft
    length/smoothness regularization alone (confirmed insufficient on real
    TMZ7 data). All four default to on.

    hindpaw_floor_pin / hindpaw_floor_pin_max_lift_s (2026-09-09): a hind-
    paw-specific plausibility backstop, not a general data-quality pass —
    see pin_hindpaw_to_floor() in cube_core.py. A mouse's hind paws are
    essentially always ground-planted; a hind-paw Z trajectory that stays
    elevated for longer than hindpaw_floor_pin_max_lift_s seconds is
    treated as triangulation drift (not a real sustained lift) and pulled
    toward the session's own hind-paw-derived floor height. Runs LAST,
    after displacement_clamp, on whatever pts3d that pass leaves behind.
    Defaults on; a no-op whenever no hind-paw bodypart is present in this
    session's schema.

    head_orient_gate*: see _resolve_camera_disagreement_via_orientation — a
    tie-breaker (not a blanket filter) that only acts on frames where the
    cameras disagree on head_orient_gate_bodyparts (default ["nose"]) by more
    than head_orient_gate_disagreement_px, and even then only excludes a
    disagreeing camera if it clearly (by head_orient_gate_margin_deg) stands
    out as the most likely to be seeing the back of the animal's head
    (heading-vs-camera angle > head_orient_gate_angle_deg).

    turned_away_percam_gate (2026-09-18): a FIRST PASS, run per-camera on
    each camera's own raw 2D likelihood before triangulation even starts —
    reuses cube_core.detect_turned_away_bins, the same confidence-based
    algorithm now standard everywhere in the 2D pipeline. Deliberately NOT
    the same as running turned-away detection on the POST-triangulation
    combined likelihood: that was tried and rejected (see BSoidEngine.run(),
    the _is_3d branch of its own turned-away call — a combined-likelihood
    dip means "not enough cameras resolved this point" far more often than
    "turned away from every camera simultaneously", confirmed on a real
    4-camera TMZ7 run flagging an implausible 24.7% of bins). Per-camera,
    pre-triangulation, the signal is exactly what the algorithm was
    validated against: does THIS camera's own view show a genuinely-
    occluded/turned-away face. When a camera is flagged turned-away on a
    frame, that camera's FACE-region columns only (TURNED_AWAY_FACE_POINTS
    plus earbases, if present) are zeroed in `scores` for that frame before
    triangulate_cameras() runs, so a hallucinated/wrongly-placed face
    keypoint from a camera that can't actually see the face never gets to
    contribute — triangulation naturally falls back to whichever other
    camera(s) still have a real view, with the SAME downstream recovery
    machinery (nose_multipoint_prior, face_rigid_cluster, ray_temporal_
    fallback) already built to handle an excluded/degenerate point. Non-
    face bodyparts (limbs, torso, tail) are untouched — a camera that can't
    see the face because the animal is turned away often still tracks the
    body fine, so blanking the whole camera would discard good data for no
    benefit. A no-op per-camera when that camera's own H5 lacks a 'nose'
    bodypart (detect_turned_away_bins' own guard).
    """
    import pandas as pd
    from cube_core import (triangulate_cameras, _h5_has_z,
                            hindpaw_bp_indices, pin_hindpaw_to_floor)

    sid = sess["session_id"]
    session_dir = out_folder / sid
    session_dir.mkdir(parents=True, exist_ok=True)

    out_h5 = session_dir / f"{sid}_3d_filtered.h5"
    if out_h5.exists():
        if _h5_has_z(out_h5):
            logger.info(f"  [{sid}] Skipping (already exists): {out_h5.name}")
            return out_h5
        logger.info(
            f"  [{sid}] Existing {out_h5.name} has no z column — it's a prior "
            f"homography-fusion (2D) output, not a triangulated one. "
            f"Re-triangulating now that a calibration is available.")

    # Collect H5s per camera (only cameras that have a valid H5)
    _prep_root = out_folder / "_prep"
    cam_h5s: dict[str, Path] = {}
    for cam in cam_labels:
        mp4 = sess["cameras"].get(cam)
        if mp4 is None:
            continue
        h5 = _find_camera_h5(mp4, _prep_root)
        if h5 and h5.exists():
            cam_h5s[cam] = h5
        else:
            logger.warn(f"  [{sid}] No H5 for {cam} — will be skipped in triangulation")

    if len(cam_h5s) < 2:
        raise RuntimeError(
            f"[{sid}] Need at least 2 cameras with H5 files; found {len(cam_h5s)}")

    present_cams = sorted(cam_h5s.keys())
    logger.info(f"  [{sid}] Triangulating cameras: {present_cams}")

    # Load all DLC DataFrames; align bodyparts
    dfs: list = []
    bps_per_cam: list[list[str]] = []
    for cam in present_cams:
        df = pd.read_hdf(str(cam_h5s[cam]))
        scorer = df.columns.get_level_values(0)[0]
        bps = df[scorer].columns.get_level_values(0).unique().tolist()
        bps_per_cam.append(bps)
        dfs.append((scorer, df))

    # Intersect bodyparts across cameras for a consistent output
    common_bps = list(bps_per_cam[0])
    for bps_k in bps_per_cam[1:]:
        common_bps = [b for b in common_bps if b in bps_k]

    # Drop bodyparts confirmed chronically unreliable on both the CUBE 3D rig
    # and an independent 2D single-camera rig (schema/anatomy mismatch, not
    # fixable by camera count or triangulation method) — never triangulated,
    # never written to the 3D H5, so overlays/skeleton/legend/BSoid all
    # inherit the exclusion automatically since they derive their bodypart
    # list from this H5.
    _excluded_present = [b for b in common_bps if b in SUPERANIMAL_QUADRUPED_EXCLUDED_BODYPARTS]
    if _excluded_present:
        logger.info(f"  [{sid}] Excluding chronically-unreliable bodyparts: {_excluded_present}")
        common_bps = [b for b in common_bps if b not in SUPERANIMAL_QUADRUPED_EXCLUDED_BODYPARTS]

    if not common_bps:
        raise RuntimeError(f"[{sid}] No common bodyparts across cameras")

    # Determine frame count (guaranteed identical by Step 0; take minimum as safety)
    n_frames = min(len(df) for _, df in dfs)
    n_bp     = len(common_bps)
    n_cams   = len(present_cams)

    pts2d  = np.zeros((n_cams, n_frames, n_bp, 2), dtype=np.float64)
    scores = np.zeros((n_cams, n_frames, n_bp),    dtype=np.float64)

    for ci, (cam, (scorer, df)) in enumerate(zip(present_cams, dfs)):
        cam_rect = (crop_rects or {}).get(cam)
        x_off = float(cam_rect["x"]) if cam_rect else 0.0
        y_off = float(cam_rect["y"]) if cam_rect else 0.0
        for bi, bp in enumerate(common_bps):
            try:
                x  = df[scorer, bp, "x"].values[:n_frames].astype(float)
                y  = df[scorer, bp, "y"].values[:n_frames].astype(float)
                ll = df[scorer, bp, "likelihood"].values[:n_frames].astype(float)
            except Exception:
                x = y = ll = np.zeros(n_frames)
            pts2d[ci, :, bi, 0] = x + x_off
            pts2d[ci, :, bi, 1] = y + y_off
            scores[ci, :, bi]   = ll

    # Per-camera turned-away pre-triangulation gate -- see this function's
    # own docstring for why this runs HERE (per-camera, pre-triangulation)
    # rather than on the combined post-triangulation likelihood. Zeros only
    # the face-region columns of `scores` for a camera on frames where THAT
    # camera's own raw likelihood shows a genuine turned-away/occluded face,
    # so a hallucinated face keypoint from a camera that can't see the face
    # never enters triangulation -- limbs/torso/tail from that camera are
    # untouched.
    if turned_away_percam_gate and "nose" in [b.lower() for b in common_bps]:
        from cube_core import detect_turned_away_bins, TURNED_AWAY_FACE_POINTS
        _fps_gate = float(sess.get("fps", 30.0))
        _bp_lower_common = [b.lower() for b in common_bps]
        _face_col_idx = [i for i, b in enumerate(_bp_lower_common)
                          if b in TURNED_AWAY_FACE_POINTS or "earbase" in b]
        if _face_col_idx:
            for ci, cam in enumerate(present_cams):
                try:
                    _ta_bin_mask = detect_turned_away_bins(
                        scores[ci], common_bps, _fps_gate,
                        likelihood_thresh=max(ll_threshold, 0.3),
                        adaptive_pct=30.0, face_majority_frac=0.5,
                        min_window_s=1.0, merge_gap_s=0.5)
                except Exception as e:
                    logger.warn(f"  [{sid}] Per-camera turned-away gate failed for "
                                f"{cam}, continuing without it: {e}")
                    continue
                if _ta_bin_mask.size == 0 or not _ta_bin_mask.any():
                    continue
                _win_gate = max(1, int(round(_fps_gate / 10)))
                _ta_frame_mask = np.repeat(_ta_bin_mask, _win_gate)
                if _ta_frame_mask.shape[0] < n_frames:
                    _ta_frame_mask = np.concatenate([
                        _ta_frame_mask,
                        np.zeros(n_frames - _ta_frame_mask.shape[0], dtype=bool)])
                else:
                    _ta_frame_mask = _ta_frame_mask[:n_frames]
                if _ta_frame_mask.any():
                    scores[np.ix_([ci], _ta_frame_mask, _face_col_idx)] = 0.0
                    logger.info(f"  [{sid}] Per-camera turned-away gate: {cam} "
                                f"face columns zeroed on {int(_ta_frame_mask.sum())}/"
                                f"{n_frames} frames ({100*_ta_frame_mask.mean():.1f}%).")

    # Run triangulation
    gdop_weighted_bp_idx = None
    if gdop_weighted and gdop_weighted_bodyparts:
        bp_index_all = {bp: i for i, bp in enumerate(common_bps)}
        gdop_weighted_bp_idx = {bp_index_all[b] for b in gdop_weighted_bodyparts if b in bp_index_all}

    pts3d = triangulate_cameras(pts2d, scores, calib_toml,
                                use_ransac=use_ransac,
                                ransac_threshold=ransac_threshold,
                                confidence_weighted=confidence_weighted,
                                gdop_weighted=gdop_weighted,
                                gdop_weighted_bp_idx=gdop_weighted_bp_idx,
                                ll_gate=ll_gate,
                                median_window=median_window,
                                log_fn=logger.info)
    # pts3d: (n_frames, n_bp, 3)

    # Baseline (plain-method) triangulation, used ONLY to compute the
    # temporal-smoothness statistic below (_degeneracy_corrected_scale_smooth)
    # — never for the actual triangulated geometry, which stays whatever
    # method was configured above. Root cause this exists to close off: that
    # statistic pools frame-to-frame displacement across EVERY bodypart into
    # a single scalar (_default_smooth_stat), applied as ONE shared
    # optim_points(scale_smooth=...) weight for the WHOLE skeleton in one
    # joint solve. Confirmed on real data (D:\3D DLC\CLAUDE.md, 2026-08-27
    # "bone-constraint smoothing-statistic contamination" entry): scoping an
    # experimental method (gdop_weighted) to ONLY nose/upper_jaw still
    # measurably changed the final refined position of bodyparts that
    # method never touched at all (back_left_thai, front_right_knee, etc.),
    # because it shifted this one pooled statistic, not because of bone-
    # length-constraint graph connectivity. Freezing the statistic to a
    # canonical plain-triangulate() baseline means swapping ANY bodypart's
    # triangulation method — gdop_weighted or any future one — can no longer
    # silently reshape how strongly every OTHER bodypart gets smoothed.
    # Skipped (no extra triangulation cost) when no experimental method is
    # active (pts3d already IS the plain baseline then) or bone_constraints
    # is off (nothing downstream would use this statistic at all).
    pts3d_for_smoothing_stat = pts3d
    if bone_constraints and (use_ransac or confidence_weighted or gdop_weighted):
        try:
            pts3d_for_smoothing_stat = triangulate_cameras(
                pts2d, scores, calib_toml, ll_gate=ll_gate,
                median_window=median_window, log_fn=logger.info)
        except Exception as e:
            logger.warn(f"  [{sid}] Baseline triangulation for smoothing-statistic "
                        f"isolation failed, falling back to the active method's own "
                        f"points (smoothing strength may be affected by that method): {e}")
            pts3d_for_smoothing_stat = pts3d

    # Camera-disagreement resolution via body orientation: only for frames
    # where the cameras that cleared ll_gate actually DISAGREE on the
    # configured target bodyparts (default ["nose"]) does this look at body
    # orientation at all, and even then only to break the tie by excluding
    # whichever disagreeing camera clearly looks like it's seeing the back of
    # the animal — see _resolve_camera_disagreement_via_orientation's
    # docstring for why this replaced an earlier unconditional per-frame
    # angle gate that real-data validation showed made nose measurably worse.
    if head_orient_gate:
        _bp_index_for_gate = {bp: i for i, bp in enumerate(common_bps)}
        _gate_bps = list(head_orient_gate_bodyparts or ["nose"])
        try:
            pts3d = _resolve_camera_disagreement_via_orientation(
                pts2d, scores, pts3d, _bp_index_for_gate, present_cams, calib_toml,
                _gate_bps, ll_gate, head_orient_gate_disagreement_px,
                head_orient_gate_angle_deg, head_orient_gate_margin_deg,
                sid, logger=logger)
        except Exception as e:
            logger.warn(f"  [{sid}] Camera-disagreement resolution failed, "
                        f"continuing without it: {e}")

    # Frames/bodyparts where triangulation had fewer than 2 valid camera views
    # are zero-filled by triangulate_cameras() (documented zero-fill-on-failure
    # convention) — recorded here BEFORE bone-constraint refinement, since that
    # step's temporal-smoothness/bone-length terms can plausibly interpolate a
    # position for a frame with zero real 2D evidence, which is a reasonable
    # thing to draw but not a real detection to report as confident.
    #
    # aniposelib's triangulate_ransac() can also return actual NaN (not zero)
    # when it can't find enough agreeing cameras for a frame — confirmed on
    # real data (2026-08-27, once a board-dimension calibration bug was fixed
    # and RANSAC started correctly rejecting unresolvable points instead of
    # fitting them to bad-calibration geometry). np.abs(nan) < 1e-6 is always
    # False, so this must be checked separately or NaN silently reaches the
    # bone-constraint solver and the final H5 with a nonzero likelihood —
    # worse than the documented zero-fill convention, and it crashed
    # make_skeleton_video's axis-limit computation outright.
    degenerate_mask = (np.all(np.abs(pts3d) < 1e-6, axis=-1)
                        | np.any(np.isnan(pts3d), axis=-1))   # (n_frames, n_bp)

    # ---- 2026-09-09 data-quality passes (TMZ7 investigation follow-up) ----
    # Order matters: ray+temporal recovery runs FIRST so its (possibly
    # messier) recovered points are still subject to the hard bone-length
    # gate and displacement clamp below exactly like real data — never
    # exempted. `recovered_mask` is tracked separately so combined_ll can
    # still zero it later regardless of whether a recovered point survives
    # every check (a recovered position is never reported as confident).
    bp_index_early = {bp: i for i, bp in enumerate(common_bps)}
    recovered_mask = np.zeros_like(degenerate_mask)
    _fps_early = float(sess.get("fps", 30.0))
    if ray_temporal_fallback:
        try:
            # Time-based (not frame-count-based) so the real bracketing
            # window this was validated against (1.5s, from the original
            # 45-frame/30fps A/B test) is preserved identically at any fps
            # -- 45 frames at 30fps is only 0.75s at 60fps or 0.375s at
            # 120fps if left as a raw frame count, silently shrinking the
            # window the guardrail was actually tuned against.
            _ray_gap_frames = max(1, int(round(ray_temporal_max_gap_s * _fps_early)))
            pts3d, recovered_mask = _ray_temporal_fallback(
                pts2d, scores, pts3d, degenerate_mask, calib_toml, present_cams,
                ll_gate, max_gap_frames=_ray_gap_frames,
                max_ray_disagreement_mm=ray_temporal_max_ray_disagreement_mm,
                sid=sid, logger=logger)
        except Exception as e:
            logger.warn(f"  [{sid}] Ray + temporal fallback failed, "
                        f"continuing without it: {e}")

    # Second, more permissive recovery pass scoped to the empirically weak,
    # non-rigid tier only (WEAK_TIER_NON_RIGID_BODYPARTS) -- these bodyparts
    # have no rigid-cluster shortcut available (unlike nose/upper_jaw/eyes/
    # earbase above), so a wider gap/disagreement window is the only lever
    # left, and it's kept OFF the well-tracked bodyparts that don't need it.
    # Runs on whatever the first pass left degenerate, same ordering
    # rationale (still fully subject to every later geometric check below).
    if weak_tier_extra_recovery:
        try:
            _weak_idx = [bp_index_early[b] for b in weak_tier_bodyparts if b in bp_index_early]
            if _weak_idx:
                _weak_gap_frames = max(1, int(round(weak_tier_ray_temporal_max_gap_s * _fps_early)))
                degenerate_mask_for_weak = (np.all(np.abs(pts3d) < 1e-6, axis=-1)
                                            | np.any(np.isnan(pts3d), axis=-1))
                pts3d, _weak_recovered = _ray_temporal_fallback(
                    pts2d, scores, pts3d, degenerate_mask_for_weak, calib_toml, present_cams,
                    ll_gate, max_gap_frames=_weak_gap_frames,
                    max_ray_disagreement_mm=weak_tier_ray_temporal_max_ray_disagreement_mm,
                    bp_subset_idx=_weak_idx, sid=sid, logger=logger)
                recovered_mask = recovered_mask | _weak_recovered
        except Exception as e:
            logger.warn(f"  [{sid}] Weak-tier extra recovery pass failed, "
                        f"continuing without it: {e}")

    if hard_bone_length_gate:
        try:
            if ll_gate > 0.0:
                _visible_scores_early = np.where(scores >= ll_gate, scores, np.nan)
            else:
                _visible_scores_early = scores
            with np.errstate(invalid="ignore"):
                if ll_agg == "mean":
                    raw_conf = np.nanmean(_visible_scores_early, axis=0)
                elif ll_agg == "max":
                    raw_conf = np.nanmax(_visible_scores_early, axis=0)
                else:
                    raw_conf = np.nanmin(_visible_scores_early, axis=0)
            raw_conf = np.nan_to_num(raw_conf, nan=0.0)
            pts3d = _apply_hard_bone_length_gate(
                pts3d, raw_conf, bp_index_early, sid=sid, logger=logger)
        except Exception as e:
            logger.warn(f"  [{sid}] Hard bone-length gate failed, "
                        f"continuing without it: {e}")

    if torso_segment_gate:
        try:
            _fps_for_torso_gate = float(sess.get("fps", 30.0))
            if ll_gate > 0.0:
                _visible_scores_torso = np.where(scores >= ll_gate, scores, np.nan)
            else:
                _visible_scores_torso = scores
            with np.errstate(invalid="ignore"):
                if ll_agg == "mean":
                    raw_conf_torso = np.nanmean(_visible_scores_torso, axis=0)
                elif ll_agg == "max":
                    raw_conf_torso = np.nanmax(_visible_scores_torso, axis=0)
                else:
                    raw_conf_torso = np.nanmin(_visible_scores_torso, axis=0)
            raw_conf_torso = np.nan_to_num(raw_conf_torso, nan=0.0)
            pts3d = _apply_learned_bone_length_gate(
                pts3d, raw_conf_torso, bp_index_early, fps=_fps_for_torso_gate,
                sid=sid, logger=logger)
        except Exception as e:
            logger.warn(f"  [{sid}] Learned torso-segment gate failed, "
                        f"continuing without it: {e}")

    # A "confidently wrong" nose (a real, non-degenerate detection that's
    # nonetheless anatomically inconsistent with the rest of the head) is a
    # documented, repeat failure mode this codebase has hit before — the
    # hard bone-length gate above only checks limb/ear PAIRS, never nose,
    # and every other new pass here only ever fills a nose slot that's
    # already degenerate. Flag-and-NaN here (before the degenerate_mask
    # recompute below) so a flagged frame flows through the exact same
    # recovery this pipeline already has for a genuinely missing nose.
    if nose_multipoint_prior:
        _face_targets = [("nose", NOSE_MULTIPOINT_ANCHOR_CANDIDATES)]
        if face_rigid_cluster_enabled:
            _face_targets += list(FACE_RIGID_CLUSTER_EXTRA_TARGETS)
        for _target_bp, _target_anchors in _face_targets:
            try:
                _degenerate_for_face_check = (np.all(np.abs(pts3d) < 1e-6, axis=-1)
                                               | np.any(np.isnan(pts3d), axis=-1))
                pts3d, _wrong_face_mask = _flag_anatomically_inconsistent_nose(
                    pts3d, _degenerate_for_face_check, bp_index_early,
                    nose_bp=_target_bp, candidates=_target_anchors, sid=sid, logger=logger)
            except Exception as e:
                logger.warn(f"  [{sid}] Face anatomical-consistency check "
                            f"({_target_bp}) failed, continuing without it: {e}")

    fps = float(sess.get("fps", 30.0))
    if displacement_clamp:
        try:
            _degenerate_for_clamp = (np.all(np.abs(pts3d) < 1e-6, axis=-1)
                                      | np.any(np.isnan(pts3d), axis=-1))
            pts3d, _clamped_mask = _apply_displacement_clamp(
                pts3d, _degenerate_for_clamp, fps, sid=sid, logger=logger)
        except Exception as e:
            logger.warn(f"  [{sid}] Displacement clamp failed, "
                        f"continuing without it: {e}")

    if hindpaw_floor_pin:
        try:
            _hindpaw_idx = hindpaw_bp_indices(common_bps)
            if _hindpaw_idx:
                _valid_for_pin = ~(np.all(np.abs(pts3d) < 1e-6, axis=-1)
                                    | np.any(np.isnan(pts3d), axis=-1))
                pts3d, _floor_z = pin_hindpaw_to_floor(
                    pts3d, scores, _hindpaw_idx, fps,
                    valid_mask=_valid_for_pin, ll_gate=ll_gate,
                    max_lift_s=hindpaw_floor_pin_max_lift_s,
                    max_dual_lift_s=hindpaw_floor_pin_max_dual_lift_s,
                    sid=sid, log_fn=logger.info)
        except Exception as e:
            logger.warn(f"  [{sid}] Hind-paw floor pin failed, "
                        f"continuing without it: {e}")

    # Recompute the mask now that the fallback/gate/clamp above may have
    # filled some slots in and NaN'd others out.
    degenerate_mask = (np.all(np.abs(pts3d) < 1e-6, axis=-1)
                        | np.any(np.isnan(pts3d), axis=-1))   # (n_frames, n_bp)
    # ---- end 2026-09-09 data-quality passes ----

    # NaN (not zero) for every degenerate slot, kept alongside the zero-filled
    # `pts3d` below rather than replacing it. This matters because
    # optim_points() — and _default_smooth_stat()'s own replica of its
    # preprocessing — both start by calling aniposelib's interpolate_data()
    # on whatever p3ds array they're given, which does a sensible linear fill
    # across NaN gaps. Handing them the ALREADY-zero-filled array instead
    # defeats that: interpolate_data() has nothing to interpolate (the "gap"
    # already looks like real data sitting at the origin), so a literal
    # jump-to-(0,0,0)-and-back becomes part of the actual position data the
    # temporal-smoothness term has to reconcile with neighbouring frames.
    #
    # Confirmed directly on real data (2026-08-27): a well-tracked, camera-
    # detection-backed `neck_base` run (5 consecutive valid frames, DLC
    # likelihood 0.63-0.71) sat essentially stationary under the unconstrained
    # triangulation, but drifted 20+mm toward a nearby degenerate gap once
    # bone_constraints ran — the optimizer was pulling a genuinely good frame
    # toward the gap's zero-fill artifact, not smoothing real jitter. With
    # ~39% of all slots degenerate on real CUBE sessions, this touches a large
    # fraction of the data and was measured to make bone_constraints net
    # NEGATIVE (2-4x worse reprojection error, 2-3x worse jitter, both
    # sessions) — the opposite of its documented ~10% jitter improvement.
    # Passing NaN lets aniposelib's own interpolation produce a sensible
    # in-between estimate for the gap instead, so neighbouring valid frames
    # are no longer pulled toward a physically meaningless point.
    pts3d_nanpreserved = np.where(degenerate_mask[..., None], np.nan, pts3d)
    pts3d = np.nan_to_num(pts3d, nan=0.0)

    # Same NaN-preserving convention applied to the baseline points, so the
    # smoothing-statistic's degeneracy correction (below) also can't be
    # shifted by which frames the ACTIVE method happened to zero-fill
    # differently than plain triangulation would have. (Recomputed even when
    # the active method IS plain triangulation, i.e. pts3d_for_smoothing_stat
    # is just pts3d again — redundant in that case, but simpler and cheap
    # relative to the triangulation work already done above.)
    degenerate_mask_for_smoothing_stat = (
        np.all(np.abs(pts3d_for_smoothing_stat) < 1e-6, axis=-1)
        | np.any(np.isnan(pts3d_for_smoothing_stat), axis=-1))
    pts3d_for_smoothing_stat = np.where(
        degenerate_mask_for_smoothing_stat[..., None], np.nan, pts3d_for_smoothing_stat)

    # Bone-length-constrained + temporal-smoothness refinement (aniposelib
    # CameraGroup.optim_points): ties each anatomical segment's length across
    # frames using the same skeleton chain the visualisations use, which both
    # smooths jitter and lets well-tracked neighbours help stabilise a
    # momentarily-noisy point. optim_points() also carries an always-on
    # velocity-smoothness term (n_deriv_smooth=1, i.e. penalises frame-to-
    # frame jumps, not accelerations) — first-derivative smoothing is kept
    # rather than jerk/acceleration smoothing (n_deriv_smooth=2) because
    # rodent locomotion is genuinely ballistic (startle jumps, paw strikes,
    # rapid head snaps): a jerk-minimisation prior is the right model for
    # smooth reaching movements in primates, but would fight exactly the
    # sharp accelerations that are real mouse behaviour, not noise.
    #
    # Its *strength* (scale_smooth) is already implicitly frame-rate-aware:
    # it's a multiplier of a statistic computed from THIS session's own raw
    # triangulated frame-to-frame displacement, which shrinks automatically
    # at higher fps (same real velocity covers less distance per frame) and
    # grows at lower fps — no explicit fps term is needed on top of that.
    # What that auto-computed statistic doesn't account for is contamination
    # by exact zero-fill artifacts (`degenerate_mask` below, <2 valid camera
    # views — 45.8% of slots measured on real CUBE data): those get averaged
    # in as if they were genuine positions, diluting the statistic and
    # under-smoothing exactly where the data is worst.
    # _degeneracy_corrected_scale_smooth() corrects for that specifically
    # (see its docstring for why a fixed biological-velocity-ceiling version
    # was tried and reverted instead).
    if bone_constraints:
        try:
            from aniposelib.cameras import CameraGroup
            bp_index = {bp: i for i, bp in enumerate(common_bps)}
            skeleton_pairs = _infer_skeleton(common_bps, logger=logger)
            # Supplemental rigid-head constraint pairs, not present in
            # SUPERANIMAL_QUADRUPED_CHAINS: the chain only links nose-eye
            # and eye-earbase, which alone don't fully constrain the head
            # triangle's shape (fixing 2 of a triangle's 3 edge lengths
            # still leaves the third free to vary) -- adding eye-eye and
            # earbase-earbase closes that gap, so bone_constraints'
            # existing length-preservation actually holds the whole head
            # rigid instead of partially. Confirmed as a real, visible
            # artifact in a rendered skeleton video (2026-08-28) before
            # this fix -- "the shape of the head is impossible to discern
            # ... the head cannot alter shape." Length-only constraints
            # (aniposelib's optim_points(), same mechanism as every other
            # bone_constraints pair) rather than a hard rigid-body
            # transform: allows natural small measurement give instead of
            # forcing an exact pose, appropriate for real analysis data
            # feeding BSoid's pairwise-distance features (unlike the
            # separate, more aggressive Kabsch-based rigid fit used only
            # for the skeleton VIDEO's display, which never touches this
            # H5). Silently no-ops per pair if either bodypart isn't in
            # common_bps.
            skeleton_pairs = list(skeleton_pairs) + [
                p for p in (("left_eye", "right_eye"), ("left_earbase", "right_earbase"))
                if p[0] in bp_index and p[1] in bp_index
            ]
            constraints = [[bp_index[a], bp_index[b]] for a, b in skeleton_pairs
                           if a in bp_index and b in bp_index]
            if constraints:
                cgroup = CameraGroup.load(str(calib_toml))
                base_multiplier = _degeneracy_corrected_scale_smooth(
                    pts3d_for_smoothing_stat, degenerate_mask_for_smoothing_stat, logger=logger)

                # optim_points() ALSO independently recomputes this same
                # statistic internally (aniposelib's own
                # `default_smooth = 1/mean(|diff(medfilt(interp(p3ds)))|)`,
                # multiplied against whatever scale_smooth= we pass it) from
                # the p3ds ARGUMENT we're about to hand it — i.e. from
                # whichever triangulation method is actually active, not from
                # the frozen baseline base_multiplier was just computed from.
                # Passing base_multiplier straight through would still leave
                # that second channel open: aniposelib's own recomputation
                # would still be method-dependent even though our own
                # wrapper-level statistic no longer is. Canceling it out
                # (computing the exact scale_smooth value that makes
                # aniposelib's internal recomputation multiply back out to
                # the same baseline-derived target regardless of what it
                # recomputes) closes that second channel too, so the FINAL
                # smoothing strength applied to the whole skeleton is fully
                # pinned to the session's baseline motion scale — not to
                # whichever bodyparts an experimental method (gdop_weighted
                # or any future one) happened to touch. See
                # D:\3D DLC\CLAUDE.md, 2026-08-27 "bone-constraint smoothing-
                # statistic contamination" entry (this is Part 2 of that fix
                # — Part 1 alone was confirmed on real data to only partially
                # close the leak).
                baseline_raw = _default_smooth_stat(pts3d_for_smoothing_stat)
                active_raw = _default_smooth_stat(pts3d_nanpreserved)
                if baseline_raw and active_raw and active_raw > 0:
                    desired_full = base_multiplier * baseline_raw
                    scale_smooth = desired_full / active_raw
                else:
                    scale_smooth = base_multiplier

                # NaN-preserved input (see pts3d_nanpreserved above) — lets
                # aniposelib's own interpolate_data() fill degenerate gaps
                # sensibly instead of smoothing around a literal (0,0,0)
                # artifact. optim_points() always returns a fully-filled
                # array (interpolate_data has no NaN left to hand back), so
                # `pts3d` remains real-valued after this either way.
                pts3d = cgroup.optim_points(
                    pts2d, pts3d_nanpreserved, constraints=constraints, scores=scores,
                    scale_smooth=scale_smooth)
                logger.info(f"  [{sid}] Bone-length + degeneracy-corrected "
                            f"temporal-smoothness refinement applied "
                            f"({len(constraints)} segments, scale_smooth={scale_smooth:.3g}, "
                            f"baseline-pinned multiplier={base_multiplier:.3g})")
        except Exception as e:
            logger.warn(f"  [{sid}] Bone-constraint refinement failed, "
                        f"using unconstrained triangulation: {e}")

    # Combine likelihood across cameras — only over cameras that actually
    # cleared ll_gate (i.e. contributed to triangulation), not blindly across
    # all n_cams raw scores. On an orthogonal multi-camera rig, a lateral
    # bodypart (an ear, the back) is only ever visible to one camera pair at
    # a time BY DESIGN — the opposite pair correctly reports near-zero
    # likelihood because it structurally cannot see that side, not because
    # tracking failed. Aggregating with min/mean across all cameras (the
    # previous behavior) mistook that expected occlusion for low confidence:
    # confirmed on real data (2026-08-27) that left_earbase/right_earbase/
    # back_base are seen well (>0.6) by 0-1 of 4 cameras on most frames,
    # while more centrally-visible points (nose, paws) hit 3-4 — "min" agg
    # was reporting near-zero combined confidence for perfectly good
    # 2-camera detections of lateral bodyparts, not flagging bad tracking.
    # Excluding gated-out cameras from the aggregate (rather than including
    # their near-zero scores) makes the combined value reflect how well the
    # cameras that could actually see the point agreed, which is what
    # downstream consumers (dlc_3d_ll_threshold, BSoid bodypart conservation,
    # QC plots) intend to measure.
    if ll_gate > 0.0:
        visible_scores = np.where(scores >= ll_gate, scores, np.nan)
    else:
        visible_scores = scores
    with np.errstate(invalid="ignore"):
        if ll_agg == "mean":
            combined_ll = np.nanmean(visible_scores, axis=0)   # (n_frames, n_bp)
        elif ll_agg == "max":
            combined_ll = np.nanmax(visible_scores, axis=0)
        else:
            combined_ll = np.nanmin(visible_scores, axis=0)    # default "min" — most conservative among cameras that saw it
    # A (frame, bodypart) with zero cameras clearing the gate produces NaN
    # here (nanmin/nanmean/nanmax of an all-NaN slice) — that's exactly the
    # degenerate_mask case handled below, which zeroes it regardless.
    combined_ll = np.nan_to_num(combined_ll, nan=0.0)

    # Zero the likelihood for frames the original triangulation could not
    # solve at all, regardless of what the (optional) constraint refinement
    # did with them — otherwise a point with no real 2D evidence can still
    # carry a nonzero raw-camera-average likelihood and get drawn/trusted
    # downstream as if it were an actual detection.
    combined_ll = combined_ll.copy()
    combined_ll[degenerate_mask] = 0.0
    # Ray+temporal-recovered slots are never reported as confident, even if
    # they filled in a real position AND survived every later geometric
    # check (hard bone-length gate, displacement clamp, bone_constraints) —
    # matching the existing "a temporally-interpolated position is never a
    # confident detection" convention this pipeline already applies to
    # bone_constraints' own interpolated gaps.
    combined_ll[recovered_mask] = 0.0

    # Two-link geometric position prior: for each (mid_bp, end_a_bp, end_b_bp)
    # in SPINE_GEOMETRIC_PRIOR_TARGETS, corrects mid_bp's per-frame position
    # using its two well-tracked chain neighbours — a complementary, per-
    # FRAME position constraint that dlc_3d_bone_constraints' pure length-
    # across-TIME constraint can't provide. See _apply_two_link_spine_prior's
    # docstring. combined_ll is deliberately left untouched by this — a
    # geometrically-inferred position is never reported as a confident
    # detection. SPINE_GEOMETRIC_PRIOR_TARGETS is empty by default (its
    # original target, back_middle, was excluded above rather than kept as a
    # vehicle for this feature — see the exclusion note) — this is currently
    # inert infrastructure for testing a future, better-tracked candidate.
    if spine_geometric_prior and SPINE_GEOMETRIC_PRIOR_TARGETS:
        bp_index_full = {bp: i for i, bp in enumerate(common_bps)}
        for mid_bp, end_a_bp, end_b_bp in SPINE_GEOMETRIC_PRIOR_TARGETS:
            pts3d = _apply_two_link_spine_prior(
                pts3d, combined_ll, degenerate_mask, bp_index_full,
                mid_bp=mid_bp, end_a_bp=end_a_bp, end_b_bp=end_b_bp,
                ll_gate=ll_gate, logger=logger)

    # Nose multi-anchor position prior — see _apply_nose_multipoint_prior's
    # docstring. Like the two-link prior above, combined_ll is deliberately
    # left untouched by this (a geometrically-inferred position is never
    # reported as a confident detection).
    if nose_multipoint_prior:
        bp_index_full = {bp: i for i, bp in enumerate(common_bps)}
        _face_prior_targets = [("nose", NOSE_MULTIPOINT_ANCHOR_CANDIDATES)]
        if face_rigid_cluster_enabled:
            _face_prior_targets += list(FACE_RIGID_CLUSTER_EXTRA_TARGETS)
        for _target_bp, _target_anchors in _face_prior_targets:
            try:
                pts3d = _apply_nose_multipoint_prior(
                    pts3d, degenerate_mask, bp_index_full,
                    nose_bp=_target_bp, candidates=_target_anchors, sid=sid, logger=logger)
            except Exception as e:
                logger.warn(f"  [{sid}] Face multipoint prior ({_target_bp}) failed, "
                            f"continuing without it: {e}")

    # Per-point confidence gate: zero likelihood below threshold so BSoid prep
    # excludes these bodypart slots during bodypart conservation scoring.
    if ll_threshold > 0.0:
        combined_ll[combined_ll < ll_threshold] = 0.0

    # Build 4-coord H5: (scorer, bodyparts, ['x','y','z','likelihood'])
    scorer_name = "cube_3d"
    arrays = [
        [scorer_name] * (4 * n_bp),
        np.repeat(common_bps, 4).tolist(),
        ["x", "y", "z", "likelihood"] * n_bp,
    ]
    midx = pd.MultiIndex.from_arrays(arrays, names=["scorer", "bodyparts", "coords"])

    data = np.zeros((n_frames, 4 * n_bp), dtype=np.float64)
    for bi in range(n_bp):
        col = bi * 4
        data[:, col]     = pts3d[:, bi, 0]
        data[:, col + 1] = pts3d[:, bi, 1]
        data[:, col + 2] = pts3d[:, bi, 2]
        data[:, col + 3] = combined_ll[:, bi]

    df_out = pd.DataFrame(data, columns=midx)

    tmp = out_h5.with_suffix(".tmp.h5")
    df_out.to_hdf(str(tmp), key="df_with_missing", mode="w", format="fixed")
    tmp.replace(out_h5)
    logger.info(f"  [{sid}] 3D H5 written: {out_h5.name}")

    return out_h5


def _fuse_session_homography(sess: dict, cam_labels: list[str],
                              out_folder: Path, step1_prep: Path, logger) -> "Path | None":
    """
    Fuse one session's per-camera 2D H5s into a single 2D H5 via floor-
    homography fusion — the homography-fallback counterpart of
    _triangulate_session(), used instead of it when
    _run_3d_dlc_step()'s homography_fallback_mode is on (<=3 cameras, no
    calibration.toml resolvable). See C:\\Users\\param\\.claude\\plans\\
    woolly-cuddling-goose.md for the full rationale.

    Writes {out_folder}/{session_id}/{session_id}_3d_filtered.h5 — the
    SAME filename/location convention _triangulate_session() uses, so
    Phase 4's video_folders handoff and BSOID prep work identically
    regardless of which branch produced it. The output has no 'z' column
    (_h5_has_z() returns False), which is how a later re-run tells a
    fused output apart from a real triangulated one (see
    _triangulate_session()'s skip-if-exists check).

    Reuses fuse_doric_group_and_write_h5() (cube_core.py) completely
    unmodified, fed from this pipeline's own Phase 2 per-camera H5
    outputs (via _find_camera_h5()) rather than letting its directory-glob
    discovery re-find them independently — the assumed calling-blocking
    homography-pair calibration gate (ensure_camera_pairs_calibrated) is
    expected to have already run once for the whole batch before this is
    called per-session.
    """
    from cube_core import fuse_doric_group_and_write_h5, _h5_has_z

    sid = sess["session_id"]
    session_dir = out_folder / sid
    session_dir.mkdir(parents=True, exist_ok=True)

    out_h5 = session_dir / f"{sid}_3d_filtered.h5"
    if out_h5.exists():
        if _h5_has_z(out_h5):
            logger.warn(f"  [{sid}] Existing {out_h5.name} is already a real "
                        f"triangulated (3D) output — not overwriting it with a "
                        f"homography-fusion result.")
            return out_h5
        try:
            import pandas as _pd
            _pd.read_hdf(str(out_h5))
            _readable = True
        except Exception:
            _readable = False
        if _readable:
            logger.info(f"  [{sid}] Skipping (already exists): {out_h5.name}")
            return out_h5
        logger.warn(f"  [{sid}] Existing {out_h5.name} appears corrupt/unreadable "
                    f"— regenerating.")

    cam_entries = []
    for cam in cam_labels:
        mp4 = sess["cameras"].get(cam)
        if mp4 is None:
            continue
        h5 = _find_camera_h5(mp4, step1_prep)
        if h5 is not None and h5.exists():
            cam_entries.append({"cam_label": cam, "results_dir": h5.parent})
        else:
            logger.warn(f"  [{sid}] No H5 for {cam} — will be skipped in fusion")

    if len(cam_entries) < 2:
        raise RuntimeError(
            f"[{sid}] Need at least 2 cameras with H5 files; found {len(cam_entries)}")

    fused_path = fuse_doric_group_and_write_h5(cam_entries, out_h5, log_fn=logger.info)
    if fused_path is None:
        raise RuntimeError(f"[{sid}] Homography fusion failed — see log above")
    return fused_path


# ─────────────────────────────────────────────────────────────────────────────
#  Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def _run_3d_dlc_step(session: dict, settings, logger, pb, after_fn: Callable):
    """
    4-phase 3D DLC + Anipose pipeline.

    Phase 1: Validate inputs, discover Step 0 sessions.
    Phase 2: Per-camera Zoo adaptation + inference + fill masking.
    Phase 3: Parallel triangulation → 3D H5 + quad composite video.
    Phase 4: BSOID prep (inline, if auto_bsoid disabled or dlc_run_prep=True).

    ntfy notifications are sent at each major milestone.
    """
    # Import send_push_notification lazily to avoid circular import
    # (cube.py imports cube_3d_dlc at step launch time, not at module load)
    import importlib
    _cube_mod = importlib.import_module("cube")
    _notify = getattr(_cube_mod, "send_push_notification", lambda *a, **k: None)

    os.environ["DL_LIGHT"] = "True"

    try:
        import deeplabcut as dlc
    except ImportError:
        raise ImportError(
            "DeepLabCut is not installed. "
            "Activate your DLC conda environment and relaunch.")

    try:
        from aniposelib.cameras import CameraGroup  # noqa: F401 — import test
    except ImportError:
        raise ImportError(
            "aniposelib is not installed.  "
            "pip install aniposelib  (inside your cube conda environment)")

    # =========================================================================
    #  Phase 1 — Validate & Discover
    # =========================================================================
    logger.step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                "3D DLC Phase 1/4: Validation & Session Discovery")
    pb.step_indeterminate("3D DLC: Discovering sessions …")

    # On re-runs video_folders contains session subdirs, not source folders.
    # Use dlc_3d_source_folders when populated (set at end of first run), else video_folders.
    _raw_src = (session.get("dlc_3d_source_folders") or
                session.get("video_folders", []))
    source_folders = [Path(f) for f in _raw_src if Path(f).is_dir()]
    if not source_folders:
        raise FileNotFoundError(
            "No video source folders found.  "
            "Add Step 0 output folders in the main panel before running 3D DLC.")

    if bool(session.get("dlc_3d_autorun_step0", True)):
        _needs_step0 = _folders_needing_step0(source_folders, logger)
        if _needs_step0:
            logger.step(f"  Raw Step 0 recordings detected in {len(_needs_step0)} "
                        f"folder(s) — launching Step 0...")
            _autorun_step0(_needs_step0, logger)
            _still_needs = _folders_needing_step0(source_folders, logger)
            if _still_needs:
                raise RuntimeError(
                    f"Step 0 correction still incomplete for: "
                    f"{[str(f) for f in _still_needs]} — finish Step 0 (select "
                    f"folder, click Run, wait for completion) before re-running "
                    f"3D DLC, or disable dlc_3d_autorun_step0.")
            logger.success("  Step 0 correction complete for all folders")

    # Discover Step 0 sessions across all source folders; first occurrence wins.
    # Done BEFORE calibration resolution (moved up from its original position
    # below cam_labels/cube_run_root setup) because the homography-fusion
    # fallback decision below needs per-session camera counts, and this scan
    # depends only on source_folders (pure filename globbing, no calibration
    # or cam_labels needed) — see C:\Users\param\.claude\plans\
    # woolly-cuddling-goose.md.
    sessions = []
    session_src: dict[str, Path] = {}   # session_id → source folder Path
    for vf in source_folders:
        for sess_dict in find_step0_sessions(vf):
            sid = sess_dict["session_id"]
            if sid not in session_src:
                session_src[sid] = vf
                sessions.append(sess_dict)

    if not sessions:
        raise RuntimeError(
            f"No complete Step 0 sessions found in: "
            f"{[str(f) for f in source_folders]}")

    logger.success(
        f"  Found {len(sessions)} session(s) across {len(source_folders)} folder(s)")

    # Homography-fusion fallback: if no calibration.toml can be resolved,
    # fall back to 2D floor-homography fusion instead of aborting, but only
    # when every discovered session has <=3 cameras (a 4+-camera rig always
    # needs a real calibration, no fallback) and the user hasn't disabled
    # the fallback checkbox. See CLAUDE.md "Two/Three-Camera Homography
    # Fusion" and woolly-cuddling-goose.md for the full rationale.
    homography_fallback_mode = False
    try:
        calib_toml = _resolve_calib_toml(session, source_folders, logger)
    except FileNotFoundError:
        if should_use_homography_fallback(
                sessions, bool(session.get("dlc_3d_homography_fallback_enabled", True))):
            homography_fallback_mode = True
            calib_toml = None
            _max_cams = max((len(s["cameras"]) for s in sessions), default=0)
            logger.step(
                f"  No calibration.toml resolvable — falling back to "
                f"2/3-camera homography fusion ({_max_cams} camera(s) detected, "
                f"'Fall back to homography fusion' is enabled).")
        else:
            raise

    video_groups: dict = dict(session.get("video_groups", {}))

    # Unified per-run output root ("CUBE results <date>_<time>", see
    # C:\Users\param\.claude\plans\modular-purring-raven.md §1/§2). Reset
    # detection mirrors cube.py's _maybe_reset_cube_run_root(): if the
    # source folders changed since the last run recorded in
    # dlc_3d_source_folders (set at the end of Phase 4 below), clear any
    # stale cube_run_root so resolve_cube_run_root() re-checks for an
    # existing on-disk root (e.g. one Step 0 already created) instead of
    # continuing to write into a prior run's folder.
    from cube_core import resolve_cube_run_root
    _prior_src = session.get("dlc_3d_source_folders") or []
    # Compare through Path() on both sides, not raw strings -- dlc_3d_source_
    # folders can be stored with forward slashes (e.g. as typed/added via the
    # GUI) while `source_folders` here is always built via Path(f), which
    # str()-ifies to backslashes on Windows. Comparing the raw strings
    # directly made this "did the source folders change?" check a false
    # positive on every single run for a session using forward-slash storage
    # (identical folders, differently-spelled paths), which silently cleared
    # cube_run_root every time and forced resolve_cube_run_root() to
    # re-resolve from scratch -- confirmed as a real contributor to a 2026-
    # 09-09 incident where a fresh Step 1b run redid DLC Zoo adaptation for a
    # camera whose cache should have been reused untouched.
    if sorted(str(Path(f)) for f in source_folders) != sorted(str(Path(x)) for x in _prior_src):
        session["cube_run_root"] = ""
    # When pooling from >1 distinct source folder, the shared pooled
    # workspace (adapt_work / per-camera H5 cache below) must NOT be nested
    # inside whichever one sibling folder happens to already have an
    # existing "CUBE results *" root (find_existing_cube_run_root would
    # otherwise silently pick one of e.g. PBS/CLZ/DCZ and bury the shared
    # workspace inside just that one) -- it belongs one level up, at the
    # folders' own common parent (e.g. "F:\DAM EXP3 GQ"), so it reads as
    # pooled-across-siblings rather than living inside one sibling. Single-
    # source-folder runs are unaffected (unambiguous, current behavior).
    _distinct_src = sorted(set(str(f) for f in source_folders))
    _pooled_fallback_base = None
    _pooled_source_folders = source_folders
    if len(_distinct_src) > 1:
        try:
            _common = Path(os.path.commonpath(_distinct_src))
            if _common.is_dir():
                _pooled_fallback_base = _common
                _pooled_source_folders = [_common]
        except ValueError:
            pass  # e.g. folders span different drives -- fall back to default
    _cube_run_root = resolve_cube_run_root(
        existing_run_root=(session.get("cube_run_root") or "").strip(),
        source_folders=_pooled_source_folders,
        # Deliberately NOT session["output_root"] here -- that's the
        # general app workspace/log root (_resolve_work_dir(), often just
        # "<drive>:\CUBE_Pipeline"), not a 3D-DLC-specific placement choice.
        # dlc_3d_output_folder is the documented explicit override for
        # where 3D DLC writes; when it's blank, the common-parent fallback
        # above (or the single-folder default) decides instead.
        output_root=str(session.get("dlc_3d_output_folder") or "").strip(),
        fallback_base=_pooled_fallback_base,
        log_fn=logger,
    )
    session["cube_run_root"] = str(_cube_run_root)
    step1_root = _cube_run_root / "CUBE 1b - 3D DLC"
    step1_prep = step1_root / "_prep"
    step1_root.mkdir(parents=True, exist_ok=True)
    step1_prep.mkdir(parents=True, exist_ok=True)

    cam_labels_raw = session.get("dlc_3d_cam_labels", ["cam0", "cam1", "cam2", "cam3"])
    if isinstance(cam_labels_raw, str):
        cam_labels = [c.strip() for c in cam_labels_raw.split(",") if c.strip()]
    else:
        cam_labels = list(cam_labels_raw)

    # Per-session final-output root. step1_root/step1_prep above stay a
    # single POOLED workspace (Zoo adaptation models + per-camera inference
    # H5 cache, correctly shared across every session regardless of which
    # source folder it came from -- see Phase 2's representative-video
    # adaptation). But each session's own DELIVERABLES (3D H5, quad video,
    # overlays, likelihood QC, etc.) should land inside that session's own
    # source folder's own "CUBE results */CUBE 1b - 3D DLC", matching where
    # its Step 0 output already lives -- not pooled into whichever folder
    # happened to already have the newest pre-existing "CUBE results" root.
    # Found 2026-09-07: a run across 3 already-Step0'd folders (each with
    # its own "CUBE results <ts>" root) silently wrote every session's 3D
    # DLC output into only the newest of the 3 roots (resolve_cube_run_root
    # -> find_existing_cube_run_root picks one, discards the others), which
    # from the outside looked like "2 of 3 folders were never processed".
    # Skipped only when dlc_3d_output_folder is explicitly set -- that key
    # is documented as an explicit pooling override ("blank = each
    # session's source folder"), so honoring it here matches spec.
    _dlc3d_out_override = str(session.get("dlc_3d_output_folder") or "").strip()
    if _dlc3d_out_override:
        _folder_out_roots: dict[str, Path] | None = None
    else:
        _folder_out_roots = {}
        for _vf in dict.fromkeys(str(v) for v in session_src.values()):
            _folder_root = resolve_cube_run_root(
                existing_run_root="",
                source_folders=[Path(_vf)],
                output_root="",
                log_fn=logger,
            )
            _folder_step1 = _folder_root / "CUBE 1b - 3D DLC"
            _folder_step1.mkdir(parents=True, exist_ok=True)
            _folder_out_roots[_vf] = _folder_step1

    def _out_root(sid: str) -> Path:
        """Return the per-session deliverables output root -- that
        session's own source folder's own "CUBE 1b - 3D DLC" (see above),
        or step1_root when dlc_3d_output_folder pools everything explicitly.
        Every call site appends /sid itself for the per-session subfolder."""
        if _folder_out_roots is None:
            return step1_root
        return _folder_out_roots.get(str(session_src.get(sid, "")), step1_root)

    _adv           = session.get("dlc_advanced_cfg", {})
    sa_name        = str(_adv.get("dlc_superanimal_name", "superanimal_quadruped"))
    model_name     = str(_adv.get("dlc_architecture", "hrnet_w32"))
    detector_name  = str(_adv.get("dlc_detector",
                                   "fasterrcnn_mobilenet_v3_large_fpn"))
    pcutoff        = float(_adv.get("dlc_pcutoff",        0.3))
    # Production per-video inference bbox gate (used by _infer_camera_videos)
    # -- distinct from bbox_thr_adapt below, which only gates which detector
    # boxes are trusted as pseudo-label training data during adaptation.
    bbox_thr       = float(_adv.get("dlc_bbox_threshold", 0.6))
    bbox_thr_adapt = float(_adv.get("dlc_bbox_threshold_adapt", 0.75))
    bbox_thr_adapt_overrides = session.get(
        "dlc_3d_bbox_threshold_adapt_overrides", {})
    baseline_autodetect  = bool(session.get("dlc_3d_baseline_autodetect", False))
    baseline_clip_seconds = float(session.get("dlc_3d_baseline_clip_seconds", 20.0))
    baseline_outlier_ratio = float(session.get("dlc_3d_baseline_outlier_ratio", 0.6))
    baseline_threshold_step = float(session.get("dlc_3d_baseline_threshold_step", 0.1))
    baseline_threshold_min = float(session.get("dlc_3d_baseline_threshold_min", 0.3))
    baseline_max_retries = int(session.get("dlc_3d_baseline_max_retries", 2))
    max_ind        = int(_adv.get("dlc_max_individuals",  1))
    # dlc_det_epochs/dlc_pose_epochs are the same Advanced-DLC-panel keys the
    # 2D pipeline already reads (cube.py) -- reused here rather than adding
    # parallel 3D-only keys. Detector saturates (100% mAP@50/75) well before
    # 15 epochs in practice; pose head still improving at 15 -- the fallback
    # defaults below (10/25) apply only when the key is unset, so the 2D
    # pipeline's own fallback (n_epochs, unchanged) is unaffected.
    detector_epochs = int(_adv.get("dlc_det_epochs", 10))
    pose_epochs     = int(_adv.get("dlc_pose_epochs", 30))
    pseudo_thr      = float(settings.get("dlc_pseudo_thr", 0.4))
    _transfer       = bool(_adv.get("dlc_transfer", True))

    filter_key   = settings.get("dlc_filter", "Sequential  Median -> Gaussian")
    FILTER_OPTIONS = {
        "None (raw output)":                    [],
        "Median (spike removal)":               ["median"],
        "Gaussian smooth":                      ["gaussian"],
        "Butterworth low-pass":                 ["butterworth"],
        "Savitzky-Golay":                       ["savgol"],
        "Kalman smoother":                      ["kalman"],
        "Sequential  Median -> Gaussian":       ["median", "gaussian"],
        "Sequential  Median -> Butterworth":    ["median", "butterworth"],
        "Sequential  Median -> Savitzky-Golay": ["median", "savgol"],
    }
    filter_types = FILTER_OPTIONS.get(filter_key, ["median", "gaussian"])

    fps              = float(sessions[0]["fps"]) if sessions else 30.0
    ll_agg           = str(session.get("dlc_3d_ll_agg", "min"))
    use_ransac       = bool(session.get("dlc_3d_use_ransac", False))
    ransac_threshold = float(session.get("dlc_3d_ransac_threshold", 0.5))
    confidence_weighted = bool(session.get("dlc_3d_confidence_weighted", False))
    gdop_weighted    = bool(session.get("dlc_3d_gdop_weighted", False))  # flipped after real-data validation below
    _gdop_bps_raw    = str(session.get("dlc_3d_gdop_weighted_bodyparts", "nose,upper_jaw"))
    gdop_weighted_bps = [b.strip() for b in _gdop_bps_raw.split(",") if b.strip()]
    ll_threshold     = float(session.get("dlc_3d_ll_threshold", 0.0))
    ll_gate          = float(session.get("dlc_3d_ll_gate", 0.4))
    median_window    = int(session.get("dlc_3d_median_window", 3))
    bone_constraints = bool(session.get("dlc_3d_bone_constraints", True))
    spine_geo_prior  = bool(session.get("dlc_3d_spine_geometric_prior", True))
    head_orient_gate = bool(session.get("dlc_3d_head_orient_gate", False))
    head_orient_gate_angle = float(session.get("dlc_3d_head_orient_gate_angle_deg", 135.0))
    head_orient_gate_margin = float(session.get("dlc_3d_head_orient_gate_margin_deg", 20.0))
    head_orient_gate_disagreement = float(
        session.get("dlc_3d_head_orient_gate_disagreement_px", 50.0))
    _head_orient_bps_raw = str(session.get("dlc_3d_head_orient_gate_bodyparts", "nose"))
    head_orient_gate_bps = [b.strip() for b in _head_orient_bps_raw.split(",") if b.strip()]
    # 2026-09-09 data-quality passes (TMZ7 double-sided-calibration-board
    # investigation follow-up) — all default on; see _triangulate_session's
    # docstring and HARD_BONE_LENGTH_MAX_MM/_apply_displacement_clamp/
    # _apply_nose_multipoint_prior/_ray_temporal_fallback above.
    hard_bone_gate   = bool(session.get("dlc_3d_hard_bone_length_gate", True))
    torso_seg_gate   = bool(session.get("dlc_3d_torso_segment_gate", True))
    disp_clamp       = bool(session.get("dlc_3d_displacement_clamp", True))
    nose_multipoint  = bool(session.get("dlc_3d_nose_multipoint_prior", True))
    ray_temporal_fb  = bool(session.get("dlc_3d_ray_temporal_fallback", True))
    # Time-based (seconds), not a raw frame count, so the actual real-world
    # bracketing window stays identical regardless of session fps -- see
    # _triangulate_session's own comment at its call site.
    ray_temporal_max_gap_s = float(session.get("dlc_3d_ray_temporal_max_gap_s", 1.5))
    ray_temporal_max_disagreement = float(
        session.get("dlc_3d_ray_temporal_max_ray_disagreement_mm", 60.0))
    # Face rigid-cluster recovery generalization (nose already had this;
    # extends it to upper_jaw, never lower_jaw -- see FACE_RIGID_CLUSTER_
    # EXTRA_TARGETS's own comment) and the weak-tier (non-rigid) second-pass
    # recovery (lower_jaw/thigh/knee) -- both off by default until validated
    # the same way nose's own mechanism was.
    face_rigid_cluster = bool(session.get("dlc_3d_face_rigid_cluster_enabled", True))
    weak_tier_recovery = bool(session.get("dlc_3d_weak_tier_extra_recovery_enabled", False))
    weak_tier_gap_s = float(session.get("dlc_3d_weak_tier_ray_temporal_max_gap_s", 3.0))
    weak_tier_disagreement_mm = float(
        session.get("dlc_3d_weak_tier_ray_temporal_max_ray_disagreement_mm", 90.0))
    # Per-camera turned-away pre-triangulation gate (2026-09-18): reuses
    # cube_core.detect_turned_away_bins -- the SAME confidence-based
    # algorithm/defaults now standard across the 2D pipeline -- as a FIRST
    # PASS on each camera's own raw 2D likelihood, before triangulation.
    # See _triangulate_session's own comment at its call site for why this
    # runs pre-triangulation rather than on the combined post-triangulation
    # likelihood (which detect_turned_away_bins is deliberately never run
    # against -- see BSoidEngine.run()'s _is_3d branch comment).
    turned_away_percam_gate = bool(session.get("dlc_3d_turned_away_percam_gate_enabled", True))
    hindpaw_floor_pin = bool(session.get("dlc_3d_hindpaw_floor_pin", True))
    hindpaw_floor_pin_max_lift_s = float(session.get("dlc_3d_hindpaw_floor_pin_max_lift_s", 10.0))
    hindpaw_floor_pin_max_dual_lift_s = float(session.get("dlc_3d_hindpaw_floor_pin_max_dual_lift_s", 0.2))
    export_skel      = bool(session.get("dlc_3d_export_skeleton_video", True))
    del_orig_vid     = bool(session.get("dlc_3d_delete_orig_videos",  False))
    del_cam_h5s      = bool(session.get("dlc_3d_delete_cam_h5s",      False))
    overlay_2d       = bool(session.get("dlc_3d_overlay_2d",             True))
    overlay_dot_r    = int(session.get("dlc_3d_overlay_dot_radius",      3))
    overlay_skeleton = bool(session.get("dlc_3d_overlay_skeleton",       True))
    export_legend    = bool(session.get("dlc_3d_export_legend",          True))
    export_cam_qc    = bool(session.get("dlc_3d_export_cam_likelihood_qc", True))
    # 3D->2D reprojection QC (D:\CUBE\reproject_3d_to_2d_overlay.py, made a
    # standard output 2026-09-17): for each camera, draws its own raw 2D
    # DLC detection (solid dot) next to the triangulated 3D point
    # reprojected back into that camera's pixel space (hollow ring), for a
    # small "main" bodypart set -- the gap between the two IS that camera's
    # reprojection error, and is what first surfaced the cam3/TMZ7
    # calibration-quality gap documented in CLAUDE.md. Bordered 2x2
    # composite (make_quad_reprojection_video) mirrors make_quad_video's
    # own layout convention.
    export_reproj_qc = bool(session.get("dlc_3d_export_reprojection_qc", True))
    reproj_qc_ll_gate = float(session.get("dlc_3d_reprojection_qc_ll_gate", 0.4))
    plot_theme       = str(session.get("engine_cfg", {}).get("plot_theme", "light"))

    inf_batch = _gpu_batch_size()
    det_batch = int(_adv.get("dlc_det_batch", 0)) or inf_batch

    # Adaptation workspace: now a subfolder of this run's shared CUBE 1b - 3D DLC/
    # folder instead of the explicit-output/first-source-folder split used
    # previously. Literal dirname unchanged ("3d_adapt_workspace") so the two
    # exclusion sites that match on it by name (cube.py's
    # VIDEO_SCAN_EXCLUDE_DIRNAMES, cube_core.py's BSoid-prep nested-layout
    # scan) keep working unmodified regardless of where its parent moved.
    adapt_work = step1_root / "3d_adapt_workspace"
    adapt_work.mkdir(parents=True, exist_ok=True)

    # Per-camera crop (each camera has its own view angle, so this is a dict
    # of independent rectangles, not one shared crop like the 2D pipeline).
    # Applied before adaptation/inference AND to the final overlay/quad/
    # skeleton outputs — every downstream step reads sess["cameras"][cam] and
    # is unaware a substitution happened; _triangulate_session() is the only
    # place that needs to know, to add the offset back before triangulation.
    crop_rects = {cam: r for cam, r in session.get("dlc_3d_crop_rects", {}).items()
                  if int(r.get("w", 0)) > 0 and int(r.get("h", 0)) > 0}
    if not crop_rects:
        crop_rects = _recover_crop_rects_from_prep(step1_prep, logger)
        if crop_rects:
            session["dlc_3d_crop_rects"] = crop_rects
            try:
                session.save()
            except Exception:
                pass
    if crop_rects and homography_fallback_mode:
        logger.warn(
            "  [multi-cam] Per-camera crop rects are set while running in "
            "homography-fusion fallback mode — the floor-homography "
            "calibrator window operates on the corrected (uncropped) video, "
            "but this camera's DLC coordinates will be in CROPPED-frame "
            "space, which the fusion path (unlike _triangulate_session) does "
            "not offset-correct. Click reference points on the same crop "
            "region, or clear this camera's crop rect, to avoid a "
            "coordinate mismatch.")
    if crop_rects:
        try:
            import imageio_ffmpeg as _iio_ffmpeg
            _crop_ffmpeg = _iio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            _crop_ffmpeg = "ffmpeg"

        _crop_jobs = []
        for sess in sessions:
            for cam, rect in crop_rects.items():
                mp4 = sess["cameras"].get(cam)
                if mp4 is None:
                    continue
                _crop_jobs.append((sess, cam, mp4, rect))

        # Bounded thread pool -- each job is a plain FFmpeg subprocess call
        # (subprocess.run), so a crash inside FFmpeg itself can't take down
        # this process the way the skeleton-video renderer's in-process
        # matplotlib crashes could (see that function's own crash-history
        # comments). Cap kept low and explicit anyway to bound concurrent
        # GPU-encoder (h264_nvenc) sessions -- exceeding the driver's
        # concurrent-session limit just fails that one job, already caught
        # below exactly as it was in the sequential version.
        if _crop_jobs:
            with ThreadPoolExecutor(max_workers=min(3, len(_crop_jobs))) as _crop_pool:
                _crop_futures = {
                    _crop_pool.submit(_crop_camera_video, mp4, rect, _crop_ffmpeg, logger,
                                   step1_prep):
                        (sess, cam)
                    for sess, cam, mp4, rect in _crop_jobs
                }
                for fut in as_completed(_crop_futures):
                    sess, cam = _crop_futures[fut]
                    try:
                        sess["cameras"][cam] = fut.result()
                    except Exception as ce:
                        logger.warn(f"  [{sess['session_id']}/{cam}] Crop failed, "
                                    f"using uncropped video: {ce}")

    models: dict = dict(session.get("dlc_3d_models", {}))

    # Auto-detect already-completed adaptation from disk for any camera not
    # already cached in-memory. Runs before the Stage-1 dry-pass below so a
    # disk-recovered camera also skips that (it's gated on the same
    # uncached-camera list) — without this, a fresh session (in-memory
    # dlc_3d_models empty) redid the entire Zoo adaptation for every camera
    # even when adapt_work/adapt_{cam_label}/ already held real checkpoints
    # from a prior run of this same project.
    for _cam in cam_labels:
        if _cam in models:
            continue
        _cached = _autodetect_cached_adaptation(
            adapt_work, _cam, model_name, detector_name, logger)
        if _cached is not None:
            models[_cam] = _cached
            logger.info(f"  [{_cam}] Auto-detected cached adaptation from disk "
                        f"({Path(_cached['adapt_dir']).name})")
    if models != session.get("dlc_3d_models", {}):
        session["dlc_3d_models"] = models
        try:
            session.save()
        except Exception:
            pass

    # =========================================================================
    #  Stage 1 — pre-adaptation dry-pass baseline + auto-threshold decision
    #  (see D:\CUBE\3D_AUTO_BBOX_THRESHOLD_PLAN.md). Only runs for cameras not
    #  already cached in dlc_3d_models -- a cached model means adaptation
    #  already happened with whatever threshold was used then, so re-running
    #  the dry-pass would be wasted GPU time for no effect on this run.
    # =========================================================================
    bbox_report: dict = dict(session.get("dlc_3d_bbox_threshold_report", {}))
    uncached_cam_labels = [c for c in cam_labels if c not in models]
    if uncached_cam_labels:
        resolved_bbox_thr_adapt, stage1_report = _resolve_stage1_bbox_thresholds(
            sessions, uncached_cam_labels, bbox_thr_adapt, bbox_thr_adapt_overrides,
            _adv, logger,
            autodetect=baseline_autodetect,
            clip_seconds=baseline_clip_seconds,
            outlier_ratio=baseline_outlier_ratio,
            threshold_step=baseline_threshold_step,
            threshold_min=baseline_threshold_min,
            sa_name=sa_name, model_name=model_name, detector_name=detector_name,
            work_dir=adapt_work)
        bbox_report.setdefault("per_camera", {}).update(stage1_report.get("per_camera", {}))
        if stage1_report.get("baseline_yield_median") is not None:
            bbox_report["baseline_yield_median"] = stage1_report["baseline_yield_median"]
    else:
        resolved_bbox_thr_adapt = {}

    # =========================================================================
    #  Phase 2 — Per-camera Zoo adaptation + inference + fill masking
    # =========================================================================
    logger.step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                "3D DLC Phase 2/4: Per-camera DLC inference")

    # Import monkeypatch helper from cube.py if available
    try:
        _cube_mod_ref = importlib.import_module("cube")
        _monkeypatch = getattr(_cube_mod_ref, "_apply_dlc_monkeypatch",
                                lambda lg: None)
        _monkeypatch(logger)
    except Exception:
        pass

    # Stage 2's baseline: pseudo-label counts of cameras already adapted
    # earlier in *this* run (not cached from a prior run) -- populated as
    # the loop below processes each camera.
    stage2_baseline_counts: dict = {}
    stage1_adjusted: list = []

    for cam_label in cam_labels:
        logger.step(f"  Camera: {cam_label}")
        pb.step_start(f"3D DLC — {cam_label}: adaptation", 1)

        if cam_label not in models:
            rep_video = select_representative(sessions, cam_label)
            if rep_video is None:
                logger.warn(f"  No video found for {cam_label} — skipping")
                after_fn(lambda: pb.step_done())
                continue

            cam_bbox_thr_adapt = float(resolved_bbox_thr_adapt.get(
                cam_label, bbox_thr_adapt_overrides.get(cam_label, bbox_thr_adapt)))
            has_override = cam_label in bbox_thr_adapt_overrides
            report_entry = bbox_report.setdefault("per_camera", {}).setdefault(
                cam_label, {"manual_override": bbox_thr_adapt_overrides.get(cam_label),
                            "stage1_yield": None, "stage1_flagged": False,
                            "final_threshold_used": cam_bbox_thr_adapt})

            pose_ckpt, det_ckpt, snap_base, adapt_dir = _run_camera_adaptation_with_retry(
                rep_video, cam_label, adapt_work, logger,
                detector_epochs, pose_epochs, sa_name, model_name, detector_name,
                pcutoff, cam_bbox_thr_adapt, max_ind, inf_batch, det_batch,
                pseudo_thr, _transfer, _adv,
                autodetect=baseline_autodetect, has_manual_override=has_override,
                baseline_counts=dict(stage2_baseline_counts),
                outlier_ratio=baseline_outlier_ratio,
                threshold_step=baseline_threshold_step,
                max_retries=baseline_max_retries,
                report_entry=report_entry)

            final_thr = report_entry.get("final_threshold_used", cam_bbox_thr_adapt)
            final_count = report_entry.get("final_pseudo_label_count")
            if final_count is not None:
                stage2_baseline_counts[cam_label] = final_count
            if not has_override and (report_entry.get("stage1_flagged") or
                                      report_entry.get("stage2_retries")):
                stage1_adjusted.append(
                    f"{cam_label} auto-adjusted "
                    f"({bbox_thr_adapt:.2f}→{final_thr:.2f}"
                    f"{', ' + str(report_entry['stage2_retries']) + ' Stage-2 retry' if report_entry.get('stage2_retries') else ''})")

            models[cam_label] = {
                "pose_ckpt": pose_ckpt,
                "det_ckpt":  det_ckpt,
                "snap_base": snap_base,
                "adapt_dir": str(adapt_dir) if adapt_dir else None,
            }
            session["dlc_3d_models"] = models
            session["dlc_3d_bbox_threshold_report"] = bbox_report
            try:
                session.save()
            except Exception:
                pass

            _notify(session,
                    f"Camera {cam_label} adaptation complete on: {rep_video.name}",
                    title=f"CUBE 3D — {cam_label} Adapted", logger=logger)
        else:
            logger.info(f"  [{cam_label}] Using cached model")

        after_fn(lambda: pb.step_done())

        # ── Batch inference ──────────────────────────────────────────────────
        cam_info = models[cam_label]
        pose_ckpt    = cam_info.get("pose_ckpt")
        det_ckpt     = cam_info.get("det_ckpt")
        snap_base    = cam_info.get("snap_base")
        adapt_dir    = Path(cam_info["adapt_dir"]) if cam_info.get("adapt_dir") else None
        inf_cfg      = None
        if adapt_dir and adapt_dir.exists():
            cfg_cands = list(adapt_dir.rglob("config.yaml"))
            if cfg_cands:
                inf_cfg = str(cfg_cands[0])

        all_cam_videos = [s["cameras"][cam_label]
                          for s in sessions if cam_label in s["cameras"]]
        pb.step_start(f"3D DLC — {cam_label}: inference", len(all_cam_videos))

        _infer_camera_videos(
            dlc, all_cam_videos, cam_label,
            pose_ckpt, det_ckpt, snap_base, inf_cfg,
            sa_name, model_name, detector_name,
            pcutoff, bbox_thr, max_ind, filter_types, fps,
            logger, inf_batch, det_batch, _transfer, _adv,
            dest_root=step1_prep)

        after_fn(lambda: pb.step_done())

        # ── Fill-frame masking ───────────────────────────────────────────────
        for sess in sessions:
            mp4 = sess["cameras"].get(cam_label)
            ts_csv = sess["timestamps"].get(cam_label)
            if mp4 is None or ts_csv is None:
                continue
            h5 = _find_camera_h5(mp4, step1_prep)
            if h5 and h5.exists():
                try:
                    mask_fill_frames_in_h5(h5, ts_csv, h5, log_fn=logger.info)
                except Exception as fe:
                    logger.warn(f"  Fill-mask failed for {h5.name}: {fe}")

                percam_sil_gate = bool(session.get("dlc_3d_percam_silhouette_gate", False))
                percam_bone_gate = bool(session.get("dlc_3d_percam_bone_length_gate", False))
                if percam_sil_gate or percam_bone_gate:
                    try:
                        percam_pose_cleaning_gate(
                            h5, mp4, h5,
                            ll_gate=float(session.get("dlc_3d_ll_gate", 0.4)),
                            silhouette_gate=percam_sil_gate,
                            bone_length_gate=percam_bone_gate,
                            bone_len_factor=float(session.get("dlc_3d_percam_bone_len_factor", 3.0)),
                            sil_margin_dilate=int(session.get("dlc_3d_percam_sil_margin_px", 12)),
                            max_interp_gap_s=float(session.get("dlc_3d_percam_max_interp_gap_s", 5.0 / 30.0)),
                            fps=float(sess.get("fps", 30.0)),
                            jitter_reject_factor=float(session.get("dlc_3d_percam_jitter_reject_factor", 3.0)),
                            log_fn=logger.info)
                    except Exception as pe:
                        logger.warn(f"  Per-camera pose cleaning failed for {h5.name}: {pe}")

                if export_cam_qc:
                    sid = sess["session_id"]
                    qc_path = _out_root(sid) / sid / f"{sid}_{cam_label}_likelihood_qc.png"
                    if not qc_path.exists():
                        try:
                            fill_mask = np.genfromtxt(
                                str(ts_csv), delimiter=",", skip_header=1)
                            if fill_mask.ndim == 1:
                                fill_mask = fill_mask.reshape(1, -1)
                            fill_mask = fill_mask[:, 3].astype(bool)
                            qc_path.parent.mkdir(parents=True, exist_ok=True)
                            from cube_core import plot_likelihood_qc
                            plot_likelihood_qc([h5], qc_path, exclude_mask=fill_mask)
                        except Exception as qe:
                            logger.warn(f"  Per-camera likelihood QC failed for {h5.name}: {qe}")

        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ── Stage 1/2 summary log + per-session bbox_threshold_report.json ────────
    if baseline_autodetect:
        _unchanged = [c for c in cam_labels if c not in
                      [a.split(" ", 1)[0] for a in stage1_adjusted]]
        _summary_bits = list(stage1_adjusted)
        if _unchanged:
            _summary_bits.append(f"{'/'.join(_unchanged)} unchanged")
        logger.step(f"  3D DLC Phase 2 summary: {', '.join(_summary_bits)}")

    for sess in sessions:
        sid = sess["session_id"]
        try:
            report_path = (_out_root(sid) / sid /
                            f"{sid}_bbox_threshold_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(bbox_report, f, indent=2)
        except Exception as re_:
            logger.warn(f"  Failed to write bbox_threshold_report.json for {sid}: {re_}")

    session["dlc_3d_bbox_threshold_report"] = bbox_report
    try:
        session.save()
    except Exception:
        pass

    _notify(session,
            f"DLC inference complete for {len(cam_labels)} camera(s), "
            f"{len(sessions)} session(s).",
            title="CUBE 3D — Inference Complete", logger=logger)

    # =========================================================================
    #  Phase 2.5 — Pose-based calibration refinement (opt-in)
    # =========================================================================
    # Inherently calibration-based (refines calib_toml itself) — skipped
    # entirely in homography-fallback mode, where there is no calib_toml.
    if not homography_fallback_mode and bool(session.get("dlc_3d_calib_refine", True)):
        _refined_str = session.get("dlc_3d_calib_refined_path", "")
        if _refined_str and Path(_refined_str).exists():
            calib_toml = Path(_refined_str)
            logger.info(f"  [calib-refine] Using cached refined calibration: {calib_toml}")
        else:
            logger.step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                        "3D DLC Phase 2.5: Pose-based calibration refinement")
            try:
                from calibration_refine_poses import refine_calibration_with_poses

                cam_h5s_refine: dict = {c: [] for c in cam_labels}
                for sess in sessions:
                    for cam_label in cam_labels:
                        mp4 = sess["cameras"].get(cam_label)
                        h5 = _find_camera_h5(mp4, step1_prep) if mp4 is not None else None
                        if h5 is not None:
                            cam_rect = crop_rects.get(cam_label)
                            if cam_rect:
                                try:
                                    h5 = _offset_h5_for_calibration(
                                        h5, float(cam_rect["x"]), float(cam_rect["y"]))
                                except Exception as oe:
                                    logger.warn(f"  [calib-refine] Could not offset-correct "
                                                f"cropped H5 {h5.name}: {oe}")
                        cam_h5s_refine[cam_label].append(h5)

                orthogonal = bool(session.get("dlc_3d_calib_refine_orthogonal", True))
                expected_angle_deg = float(session.get(
                    "dlc_3d_calib_refine_expected_angle_deg", 90.0))
                prior_strength = float(session.get(
                    "dlc_3d_calib_refine_prior_strength", 0.3))
                tolerance_deg = float(session.get(
                    "dlc_3d_calib_refine_tolerance_deg", 15.0))
                min_likelihood = float(session.get(
                    "dlc_3d_calib_refine_min_likelihood", 0.6))

                refined_path, refine_report = refine_calibration_with_poses(
                    calib_toml, cam_h5s_refine, cam_labels,
                    orthogonal=orthogonal,
                    expected_angle_deg=expected_angle_deg,
                    prior_strength=prior_strength,
                    tolerance_deg=tolerance_deg,
                    min_likelihood=min_likelihood,
                    log_fn=logger.info)

                if refined_path is not None:
                    session["dlc_3d_calib_refined_path"] = str(refined_path)
                    calib_toml = refined_path
                    logger.success(
                        f"  [calib-refine] average_error {refine_report['error_before']:.3f}px "
                        f"-> {refine_report['error_after']:.3f}px — using refined calibration")
                else:
                    logger.info(
                        f"  [calib-refine] Skipped (status={refine_report.get('status')}) — "
                        "using original calibration.toml")
            except Exception as e:
                logger.warn(f"  [calib-refine] Refinement failed, continuing with "
                            f"original calibration.toml: {e}")

    # =========================================================================
    #  Phase 3 — Triangulation + quad composite video
    # =========================================================================
    logger.step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                "3D DLC Phase 3/4: Triangulation & composite video")

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"

    completed_sessions: list[dict] = []
    failed_sessions:    list[str]  = []

    if homography_fallback_mode:
        # Block once (not per-session) until every pairwise camera-label
        # combination has a cached floor homography — the same "block and
        # prompt" flow the Two/Three-Camera Homography Fusion feature has
        # always used, now reachable only from here.
        #
        # Scoped to cameras ACTUALLY present across the discovered sessions,
        # not the full (often-default) cam_labels config list — cam_labels
        # defaults to ["cam0","cam1","cam2","cam3"] regardless of how many
        # cameras a given rig really has (correct for the calibrated 4-
        # camera path, where every configured label is expected to exist).
        # Passing that unfiltered default here for a real 2-camera rig would
        # generate calibration-window prompts for phantom pairs (cam0-cam2,
        # cam2-cam3, ...) that have no video at all.
        present_cam_labels = present_camera_labels(sessions)

        # Per-camera DLC H5 lists for automatic pose-based homography
        # estimation (C:\Users\param\.claude\plans\woolly-cuddling-goose.md)
        # -- same _find_camera_h5() lookup _fuse_session_homography() and
        # Phase 2.5's calib-refine block already use, index-aligned across
        # cameras by iterating `sessions` in one fixed order. Phase 2 has
        # already completed for the whole batch by this point, so every
        # session's per-camera H5 needed for pooling already exists.
        cam_h5s_for_calib: dict[str, list] = {c: [] for c in present_cam_labels}
        for sess in sessions:
            for c in present_cam_labels:
                mp4 = sess["cameras"].get(c)
                cam_h5s_for_calib[c].append(
                    _find_camera_h5(mp4, step1_prep) if mp4 is not None else None)

        _cube_mod_gate = importlib.import_module("cube")
        _cube_mod_gate.ensure_camera_pairs_calibrated(
            present_cam_labels, after_fn, logger, cam_h5s=cam_h5s_for_calib,
            pose_refine_enabled=bool(session.get("dlc_3d_homography_pose_refine_enabled", True)),
            min_likelihood=float(session.get("dlc_3d_homography_pose_refine_min_likelihood", 0.8)),
            max_error_px=float(session.get("dlc_3d_homography_pose_refine_max_error_px", 10.0)))

        def _triang_worker(sess):
            return _fuse_session_homography(
                sess, present_cam_labels, _out_root(sess["session_id"]), step1_prep, logger)
    else:
        def _triang_worker(sess):
            return _triangulate_session(
                sess, cam_labels, calib_toml, _out_root(sess["session_id"]), ll_agg, logger,
                use_ransac=use_ransac,
                ransac_threshold=ransac_threshold,
                confidence_weighted=confidence_weighted,
                gdop_weighted=gdop_weighted,
                gdop_weighted_bodyparts=gdop_weighted_bps,
                ll_threshold=ll_threshold,
                ll_gate=ll_gate,
                median_window=median_window,
                bone_constraints=bone_constraints,
                spine_geometric_prior=spine_geo_prior,
                head_orient_gate=head_orient_gate,
                head_orient_gate_angle_deg=head_orient_gate_angle,
                head_orient_gate_margin_deg=head_orient_gate_margin,
                head_orient_gate_disagreement_px=head_orient_gate_disagreement,
                head_orient_gate_bodyparts=head_orient_gate_bps,
                crop_rects=crop_rects,
                hard_bone_length_gate=hard_bone_gate,
                torso_segment_gate=torso_seg_gate,
                displacement_clamp=disp_clamp,
                nose_multipoint_prior=nose_multipoint,
                face_rigid_cluster_enabled=face_rigid_cluster,
                ray_temporal_fallback=ray_temporal_fb,
                ray_temporal_max_gap_s=ray_temporal_max_gap_s,
                ray_temporal_max_ray_disagreement_mm=ray_temporal_max_disagreement,
                weak_tier_extra_recovery=weak_tier_recovery,
                weak_tier_ray_temporal_max_gap_s=weak_tier_gap_s,
                weak_tier_ray_temporal_max_ray_disagreement_mm=weak_tier_disagreement_mm,
                hindpaw_floor_pin=hindpaw_floor_pin,
                hindpaw_floor_pin_max_lift_s=hindpaw_floor_pin_max_lift_s,
                hindpaw_floor_pin_max_dual_lift_s=hindpaw_floor_pin_max_dual_lift_s,
                turned_away_percam_gate=turned_away_percam_gate)

    pb.step_start("3D DLC: Triangulation", len(sessions))

    with ThreadPoolExecutor(max_workers=min(4, len(sessions))) as pool:
        future_to_sid = {pool.submit(_triang_worker, s): s["session_id"]
                         for s in sessions}
        for fut in as_completed(future_to_sid):
            sid = future_to_sid[fut]
            try:
                out_h5 = fut.result()
                completed_sessions.append(
                    {"session_id": sid, "h5": out_h5,
                     "session": next(s for s in sessions if s["session_id"] == sid)})
                logger.success(f"  {'Fused' if homography_fallback_mode else 'Triangulated'}: {sid}")
            except Exception as exc:
                logger.error(f"  {'Fusion' if homography_fallback_mode else 'Triangulation'} FAILED for {sid}: {exc}")
                failed_sessions.append(sid)
            after_fn(lambda: pb.step_tick(1, len(sessions)))

    pb.step_done()

    _notify(session,
            f"{'Homography fusion' if homography_fallback_mode else 'Triangulation'} "
            f"complete: {len(completed_sessions)} OK, {len(failed_sessions)} failed.",
            title="CUBE 3D — Triangulation Complete", logger=logger)

    # ── Quad composite video per session ─────────────────────────────────────
    pb.step_start("3D DLC: Quad videos", len(completed_sessions))

    for item in completed_sessions:
        sid  = item["session_id"]
        sess = item["session"]
        session_dir = _out_root(sid) / sid
        quad_path   = session_dir / f"{sid}_quad.mp4"

        raw_cam_mp4s = {cam: sess["cameras"][cam]
                        for cam in cam_labels if cam in sess["cameras"]}
        cam_mp4s = dict(raw_cam_mp4s)

        # Single read of the (small) 3D H5 for this session, shared by the
        # bp_colors/skeleton_chain derivation below and the outline-skeleton
        # derivation further down (export_skel block) -- previously each
        # read it independently via its own pd.read_hdf() call.
        df3d = None
        scorer3d = None
        bps3d = None
        if overlay_2d or export_legend or export_skel:
            try:
                import pandas as pd
                df3d = pd.read_hdf(str(item["h5"]))
                scorer3d = df3d.columns.get_level_values(0)[0]
                bps3d = df3d[scorer3d].columns.get_level_values(0).unique().tolist()
            except Exception as be:
                logger.warn(f"  [{sid}] Could not read 3D H5 for downstream exports: {be}")

        bp_colors = None
        skeleton_chain = None
        if (overlay_2d or export_legend) and df3d is not None:
            try:
                from cube_core import bodypart_region_colors
                bp_colors = bodypart_region_colors(bps3d)
                if overlay_skeleton:
                    skeleton_chain = _infer_skeleton(bps3d, logger=logger)
            except Exception as be:
                logger.warn(f"  [{sid}] Could not derive bodypart colors: {be}")

        if export_legend and bp_colors is not None:
            legend_path = session_dir / f"{sid}_bodypart_legend.png"
            if not legend_path.exists():
                try:
                    plot_bodypart_legend(list(bp_colors.keys()), legend_path, theme=plot_theme)
                except Exception as le:
                    logger.warn(f"  [{sid}] Legend generation failed: {le}")

        overlay_generated: list = []   # temp per-camera overlay files -- only needed to
                                        # build the quad composite, deleted once it exists
                                        # (Video Explorer's example clips use *_quad.mp4,
                                        # not the per-camera overlay, per CLAUDE.md)
        if overlay_2d and bp_colors is not None and not quad_path.exists():
            overlaid: dict = {}
            _overlay_jobs = []
            for cam, mp4 in cam_mp4s.items():
                h5 = _find_camera_h5(mp4, step1_prep)
                if h5 is None or not h5.exists():
                    logger.warn(f"  [{sid}/{cam}] No H5 for overlay — using raw video")
                    overlaid[cam] = mp4
                    continue
                overlay_path = session_dir / f"{cam}_overlay.mp4"
                cam_fps = float(sess.get("fps", 30.0))
                ta_mask = _compute_turned_away_frame_mask(h5, cam_fps, logger)
                _overlay_jobs.append((cam, mp4, h5, overlay_path, ta_mask))

            # Bounded thread pool, capped lower than crop's (2 vs 3): each job
            # also does in-process cv2.VideoCapture decode + per-frame drawing
            # on top of the subprocess FFmpeg encode, so more workers means
            # more concurrent CPU decode load, not just more FFmpeg processes.
            # The actual encode still happens in a separate FFmpeg subprocess
            # (Popen), so a crash there can't take down this process the way
            # the skeleton-video renderer's in-process matplotlib work could
            # (that render loop keeps its own isolated-subprocess + retry
            # mitigation, untouched here).
            if _overlay_jobs:
                with ThreadPoolExecutor(max_workers=min(2, len(_overlay_jobs))) as _ov_pool:
                    _ov_futures = {
                        _ov_pool.submit(
                            overlay_keypoints_on_video, mp4, h5, overlay_path, bp_colors,
                            logger=logger, ffmpeg_exe=ffmpeg_exe, dot_radius=overlay_dot_r,
                            skeleton=skeleton_chain if overlay_skeleton else None,
                            turned_away_mask=ta_mask):
                            (cam, mp4, overlay_path)
                        for cam, mp4, h5, overlay_path, ta_mask in _overlay_jobs
                    }
                    for fut in as_completed(_ov_futures):
                        cam, mp4, overlay_path = _ov_futures[fut]
                        try:
                            fut.result()
                            overlaid[cam] = overlay_path
                            overlay_generated.append(overlay_path)
                        except Exception as oe:
                            logger.warn(f"  [{sid}/{cam}] Overlay failed, using raw video: {oe}")
                            overlaid[cam] = mp4
            cam_mp4s = overlaid

        if len(cam_mp4s) >= 2 and not quad_path.exists():
            try:
                make_quad_video(cam_mp4s, quad_path, ffmpeg_exe, logger)
                if quad_path.exists() and quad_path.stat().st_size > 0:
                    for ov in overlay_generated:
                        try:
                            ov.unlink()
                        except Exception:
                            pass
                    if del_orig_vid:
                        for mp4 in raw_cam_mp4s.values():
                            try:
                                mp4.unlink()
                                logger.info(f"  Deleted: {mp4.name}")
                            except Exception as de:
                                logger.warn(f"  Could not delete {mp4.name}: {de}")
            except Exception as qe:
                logger.error(f"  Quad video failed for {sid}: {qe}")
        else:
            if quad_path.exists():
                logger.info(f"  [{sid}] Quad video already exists")
            elif len(cam_mp4s) < 2:
                logger.warn(f"  [{sid}] Not enough camera videos for quad composite")

        if del_cam_h5s:
            out_h5 = item["h5"]
            if out_h5.exists():
                for cam in cam_labels:
                    mp4 = sess["cameras"].get(cam)
                    if mp4 is None:
                        continue
                    h5 = _find_camera_h5(mp4, step1_prep)
                    if not (h5 and h5.exists()):
                        continue
                    # Guard against the documented data-loss bug (CLAUDE.md,
                    # "dlc_3d_delete_cam_h5s=True + zero crop offset..."):
                    # this h5 is the camera's ONLY surviving per-camera pose
                    # record unless a distinct full-frame-coordinate copy
                    # exists next to it (_offset_h5_for_calibration's output,
                    # made during Phase 2.5 -- only for a nonzero crop offset,
                    # and only if calib-refine ran and succeeded for this
                    # camera). A zero-offset camera never gets such a copy at
                    # all (the cropped and full-frame coordinate spaces are
                    # identical, so _offset_h5_for_calibration returns the
                    # same path unchanged) -- deleting it would destroy the
                    # only copy of that camera's real pose data with no
                    # fallback. Skip deletion whenever no distinct survivor
                    # exists, rather than deleting unconditionally.
                    cam_rect = crop_rects.get(cam)
                    x_off = float(cam_rect["x"]) if cam_rect else 0.0
                    y_off = float(cam_rect["y"]) if cam_rect else 0.0
                    if x_off == 0.0 and y_off == 0.0:
                        logger.warn(f"  Kept cam H5 {h5.name}: zero crop offset means "
                                    f"no distinct full-frame copy exists to survive "
                                    f"deletion (dlc_3d_delete_cam_h5s skip-guard)")
                        continue
                    full_frame_copy = h5.parent / f"{h5.stem}_full_frame.h5"
                    if not full_frame_copy.exists():
                        logger.warn(f"  Kept cam H5 {h5.name}: no full-frame copy found "
                                    f"at {full_frame_copy.name} to survive deletion "
                                    f"(dlc_3d_delete_cam_h5s skip-guard)")
                        continue
                    try:
                        h5.unlink()
                        logger.info(f"  Deleted cam H5: {h5.name}")
                    except Exception:
                        pass

        if export_skel and not homography_fallback_mode:
            # Skeleton video needs z-coordinates — the fusion output has
            # none (2D floor-homography fusion, not triangulation).
            skel_path = session_dir / f"{sid}_skeleton3d.mp4"
            if not skel_path.exists():
                try:
                    # Reduced "body outline" style (validated on real data,
                    # 2026-08-28): a mandatory minimum point set (3D "cage"
                    # head — nose + both eyes + both ears, not a single
                    # flat triangle — plus a 3-hub spine and 4 two-point
                    # limbs), extended with any other bodypart this
                    # session tracks confidently, connected head-to-body
                    # via a pair of priority-fallback synthetic anchor
                    # points (see MOUSE_OUTLINE_HEAD_EDGES) so the head
                    # never visually detaches, with bodypart-specific
                    # median-filter smoothing purely for display. Falls
                    # back to the full unrestricted skeleton if this
                    # derivation fails for any reason (e.g. an H5 whose
                    # bodypart schema doesn't match SUPERANIMAL_QUADRUPED_
                    # CHAINS/KNOWN_CHAINS at all) rather than skipping the
                    # video entirely.
                    _outline_kwargs = {}
                    if df3d is not None:
                        try:
                            # df3d/bps3d reused from the single per-session H5
                            # read above instead of re-parsing the H5 from disk
                            # a second time; select_confident_outline_bodyparts
                            # also takes df= to avoid a third read.
                            _reduced_bps = select_confident_outline_bodyparts(
                                item["h5"], ll_gate=ll_threshold or 0.6, df=df3d)
                            _edges = build_outline_skeleton_edges(
                                _reduced_bps, all_bps=bps3d, logger=logger)
                            _outline_kwargs = dict(
                                render_bodyparts=_reduced_bps, render_edges=_edges,
                                render_smooth_frames=default_outline_smooth_map(_reduced_bps))
                        except Exception as oe:
                            logger.warn(f"  [{sid}] Outline skeleton derivation failed, "
                                        f"falling back to full skeleton: {oe}")
                    # Isolated in a subprocess, not called directly: this
                    # render loop has been observed crashing natively and
                    # unpredictably (see D:\3D DLC\CLAUDE.md's BLAS-thread-
                    # contention Crash Protection entry) — an ordinary
                    # try/except here does nothing against a native crash,
                    # which would otherwise take down this whole session's
                    # pipeline run, not just skip the skeleton video.
                    make_skeleton_video_isolated(
                        item["h5"], skel_path, fps, ffmpeg_exe, logger,
                        ll_gate=ll_threshold, theme=plot_theme, **_outline_kwargs)
                except Exception as se:
                    logger.error(f"  Skeleton video failed for {sid}: {se}")

        if export_reproj_qc and not homography_fallback_mode and df3d is not None and scorer3d is not None:
            # Homography fallback has no z-coordinate / calibration.toml to
            # reproject through -- same gate export_skel already uses.
            reproj_dir = session_dir / "reprojection_qc"
            quad_reproj_path = reproj_dir / f"{sid}_reprojection_qc_quad.mp4"
            try:
                from reproject_3d_to_2d_overlay import (
                    load_2d_h5, render_camera_overlay, make_quad_reprojection_video,
                    compute_turned_away_frame_mask)
                from aniposelib.cameras import CameraGroup

                X3D = df3d[scorer3d].xs("x", level="coords", axis=1).values
                Y3D = df3d[scorer3d].xs("y", level="coords", axis=1).values
                Z3D = df3d[scorer3d].xs("z", level="coords", axis=1).values
                LL3D = df3d[scorer3d].xs("likelihood", level="coords", axis=1).values
                XYZ3D = np.stack([X3D, Y3D, Z3D], axis=-1)

                cgroup_reproj = CameraGroup.load(str(calib_toml))
                cams_by_name = {c.get_name(): c for c in cgroup_reproj.cameras}

                if not quad_reproj_path.exists():
                    reproj_dir.mkdir(parents=True, exist_ok=True)
                    rendered_reproj: dict = {}
                    for cam in cam_labels:
                        cam_obj = cams_by_name.get(cam)
                        mp4 = raw_cam_mp4s.get(cam)
                        if cam_obj is None or mp4 is None:
                            continue
                        h5_2d = _find_camera_h5(mp4, step1_prep)
                        if h5_2d is None or not h5_2d.exists():
                            continue
                        out_path = reproj_dir / f"{cam}_{sid}_reprojection_qc.mp4"
                        try:
                            bps2d, X2D, Y2D, LL2D = load_2d_h5(h5_2d)
                            rect = crop_rects.get(cam, {"x": 0, "y": 0})
                            cam_fps = float(sess.get("fps", 30.0))
                            ta_mask = compute_turned_away_frame_mask(LL2D, bps2d, cam_fps)
                            render_camera_overlay(
                                cam, sid, mp4,
                                bps3d, XYZ3D, LL3D,
                                bps2d, X2D, Y2D, LL2D,
                                cam_obj, float(rect.get("x", 0)), float(rect.get("y", 0)),
                                out_path, reproj_qc_ll_gate,
                                turned_away_mask=ta_mask)
                            rendered_reproj[cam] = out_path
                        except Exception as re_:
                            logger.warn(f"  [{sid}/{cam}] Reprojection QC render failed: {re_}")

                    if len(rendered_reproj) >= 2:
                        make_quad_reprojection_video(rendered_reproj, quad_reproj_path, ffmpeg_exe)
                        for p in rendered_reproj.values():
                            try:
                                p.unlink()
                            except Exception:
                                pass
                    elif rendered_reproj:
                        logger.warn(f"  [{sid}] Only {len(rendered_reproj)} camera(s) rendered "
                                    f"for reprojection QC -- skipping quad composite")
            except Exception as rqe:
                logger.warn(f"  [{sid}] Reprojection QC skipped: {rqe}")

        after_fn(lambda: pb.step_tick(1, len(completed_sessions)))

    pb.step_done()

    _notify(session,
            f"Quad composite videos created for {len(completed_sessions)} session(s).",
            title="CUBE 3D — Quad Videos Created", logger=logger)

    # Persist source folders so a re-run can rediscover sessions correctly
    session["dlc_3d_source_folders"] = [str(vf) for vf in source_folders]

    # ── Replace video_folders with session output dirs; carry group labels ────
    new_session_dirs: list[str] = []
    new_video_groups: dict = dict(video_groups)  # preserve any pre-existing mappings
    for item in completed_sessions:
        sid  = item["session_id"]
        sdir = str(_out_root(sid) / sid)
        new_session_dirs.append(sdir)
        src = session_src.get(sid)
        if src:
            grp = video_groups.get(str(src), "")
            if grp:
                new_video_groups[sdir] = grp
    session["video_folders"] = new_session_dirs
    session["video_groups"]  = new_video_groups
    try:
        session.save()
    except Exception:
        pass

    # =========================================================================
    #  Phase 4 — BSOID Prep (inline when auto_bsoid is off or dlc_run_prep=True)
    # =========================================================================
    auto_bsoid = bool(session.get("auto_bsoid", True))
    run_prep   = bool(settings.get("dlc_run_prep", True))

    if run_prep and not auto_bsoid:
        logger.step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                    "3D DLC Phase 4/4: BSOID Pre-processing")
        from cube_core import run_bsoid_prep
        bsoid_roots = []
        for item in completed_sessions:
            sid = item["session_id"]
            sdir = _out_root(sid) / sid
            try:
                root = run_bsoid_prep(
                    str(sdir), log_fn=logger,
                    min_confidence=float(session.get("bsoid_min_conf", 0.30)),
                    conf_metric=str(session.get("bsoid_conf_metric", "median")),
                    min_session_frac=float(session.get("bsoid_min_sess_frac", 0.6)),
                    min_keep=int(session.get("bsoid_min_keep", 6)))
                if root:
                    bsoid_roots.append(str(root))
                    logger.success(f"  BSOID prep done: {sid}")
            except Exception as be:
                logger.error(f"  BSOID prep failed for {sid}: {be}")
        session["bsoid_ready_dirs"] = bsoid_roots
        try:
            session.save()
        except Exception:
            pass

        _notify(session,
                f"BSOID pre-processing done for {len(bsoid_roots)} session(s).",
                title="CUBE 3D — Pre-processing Complete", logger=logger)

    if failed_sessions:
        logger.warn(
            f"3D DLC finished with {len(failed_sessions)} triangulation failure(s): "
            f"{failed_sessions}")
    else:
        logger.success(
            f"3D DLC complete: {len(completed_sessions)} session(s) processed  "
            f"[{datetime.now().strftime('%H:%M:%S')}]")

    _notify(session,
            f"3D DLC complete: {len(completed_sessions)} OK"
            + (f", {len(failed_sessions)} failed" if failed_sessions else "") + ".",
            title="CUBE 3D — Complete", logger=logger)
