"""
CUBE — Multi-camera calibration.toml generator

Builds the calibration.toml consumed by cube_3d_dlc.py's Step 1b (3D DLC +
Anipose) from a folder of per-camera Charuco-board recordings, using
aniposelib directly.

Usage
-----
"C:\\Users\\param\\anaconda3\\envs\\cube\\python.exe" "D:\\CUBE\\calibrate_cameras.py" ^
    --videos-dir "F:\\...\\_calibration\\female1cage3_0004" ^
    --squares-x 7 --squares-y 10 --square-length 25 --marker-length 18.75 ^
    --out-dir "F:\\...\\calibration"

--videos-dir must contain one video per camera, named "<camN>_....<ext>"
(exactly what Step 0's copy_calibration_sessions() produces in
"{out_dir}/_calibration/{session_id}/" for any session whose name contains
"calibration"). Cameras are ordered alphabetically by camN (cam0, cam1, ...),
matching the same order cube_3d_dlc.py uses when it triangulates
(present_cams = sorted(cam_h5s.keys())) — this order MUST match or
triangulation will silently pair the wrong camera's calibration with the
wrong camera's 2D points.

--out-dir is written with calibration.toml; point CUBE's 3D DLC tab
"Calibration folder" at that same directory.
"""
from __future__ import annotations
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from aniposelib.boards import CharucoBoard, merge_rows, extract_points, extract_rtvecs
from aniposelib.cameras import CameraGroup, Camera
from aniposelib.utils import get_connections, get_initial_extrinsics


def make_camera_group(cam_names, extra_dist=False):
    """
    CameraGroup.from_names() doesn't expose extra_dist, but it matters: with
    the default extra_dist=False, aniposelib's bundle adjustment only ever
    refines a single distortion coefficient (k1) per camera, zeroing out
    everything else regardless of how much the lens actually needs. Build
    cameras directly so a second coefficient (k2) can be requested when a
    single-coefficient fit leaves a persistent, data-volume-independent
    reprojection-error floor (the signature of model mismatch, not noise).
    """
    return CameraGroup([Camera(name=name, extra_dist=extra_dist) for name in cam_names])


class LowFloorCharucoBoard(CharucoBoard):
    """
    CharucoBoard whose per-frame pose-estimation corner floor is lowered
    from aniposelib's hardcoded 7 down to a caller-supplied value (as low as
    4, the mathematical minimum for cv2.solvePnP). Use this only when the
    physical board never presents enough corners per frame to clear the
    library's default floors (verified, not assumed) — each individual
    frame's pose is noisier as a result; calibrate_pooled() below relies on
    pooling many such frames so joint bundle adjustment can average out
    that per-frame noise instead of trusting any single frame's pose.
    """
    def __init__(self, *args, min_corners_pose=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_corners_pose = min_corners_pose

    def estimate_pose_points(self, camera, corners, ids):
        if corners is None or ids is None or len(corners) < self.min_corners_pose:
            return None, None

        n_corners = corners.size // 2
        corners = np.reshape(corners, (n_corners, 1, 2))

        K = camera.get_camera_matrix()
        D = camera.get_distortions()
        all_obj_points = self.board.getChessboardCorners()
        if len(ids) > 0 and all_obj_points is not None:
            detected_obj_points = []
            detected_img_points = []
            for i, corner_id in enumerate(ids.flatten()):
                if corner_id < len(all_obj_points):
                    detected_obj_points.append(all_obj_points[corner_id])
                    detected_img_points.append(corners[i].reshape(2))
            if len(detected_obj_points) >= self.min_corners_pose:
                obj_points_array = np.array(detected_obj_points,
                                            dtype=np.float32).reshape(-1, 3)
                img_points_array = np.array(detected_img_points,
                                            dtype=np.float32).reshape(-1, 2)
                # cv2.solvePnP's default SOLVEPNP_ITERATIVE needs >= 6 points
                # (confirmed via a real crash, 2026-08-27: "DLT algorithm
                # needs at least 6 points" on a 5-corner frame) — it is NOT
                # actually usable down to 4 despite this class's docstring.
                # SOLVEPNP_EPNP supports the full >= 4 range this class is
                # meant for; keep ITERATIVE (generally more accurate with
                # enough points) whenever there are enough points for it.
                pnp_flags = (cv2.SOLVEPNP_ITERATIVE if len(detected_obj_points) >= 6
                             else cv2.SOLVEPNP_EPNP)
                ret, rvec, tvec = cv2.solvePnP(obj_points_array,
                                               img_points_array, K, D,
                                               flags=pnp_flags)
                if ret:
                    return rvec, tvec
        return None, None


def apply_fixed_intrinsics(cgroup, toml_path: Path):
    """
    Load a known-good calibration.toml (e.g. from a properly-recorded
    session with the same physical cameras/lenses) and copy each camera's
    matrix + distortions onto the matching camera (by name) in *cgroup*.
    Intrinsics belong to the lens/sensor, not the rig's position, so this
    is valid across sessions where the cameras were only re-aimed/moved,
    not re-focused/re-zoomed. Extrinsics (rotation/translation) are left
    untouched here — calibrate_pooled(..., fixed_intrinsics=True) then
    solves only those from the target session's own data.
    """
    source = CameraGroup.load(str(toml_path))
    by_name = {cam.get_name(): cam for cam in source.cameras}
    missing = [cam.get_name() for cam in cgroup.cameras if cam.get_name() not in by_name]
    if missing:
        raise ValueError(
            f"{toml_path} has no intrinsics for camera(s) {missing} — "
            "cannot apply as a fixed-intrinsics source.")
    for cam in cgroup.cameras:
        src = by_name[cam.get_name()]
        cam.set_camera_matrix(src.get_camera_matrix().copy())
        cam.set_distortions(src.get_distortions().copy())
        cam.set_size(src.get_size())


def calibrate_shared_intrinsics(cgroup, board, videos, min_points=9,
                                 reference_cam=None, extra_dist=False,
                                 max_frames=30, log_fn=print):
    """
    For rigs where every camera is the same physical hardware: letting each
    camera's intrinsics be estimated independently (aniposelib's default in
    both calibrate_videos() and calibrate_pooled()) can diverge to
    physically nonsensical values for a camera whose calibration footage
    didn't present enough depth/tilt variation to constrain its own solve —
    focal length and depth are coupled in single-camera calibration (the
    classic bas-relief ambiguity), so a perfectly good lens can still solve
    to an absurd focal length if the board never moved through varied
    depths in front of that specific camera. Confirmed on a real 4-camera
    CUBE rig (2026-08-27): 3 of 4 cameras solved to 9,000-28,000px focal
    lengths (vs. ~1,000px for the 4th) despite all 4 being identical
    hardware, and the cameras themselves tested fine (opened cleanly at
    matching resolution/format).

    This sidesteps that: ONE camera (whichever has the richest calibration
    data, or reference_cam if given) gets a real cv2.calibrateCamera()
    intrinsic solve, and that same matrix + distortion is copied onto every
    camera in the group. Only extrinsics (rotation/translation) are then
    solved per-camera from that camera's own footage — see
    calibrate_pooled(fixed_intrinsics=True), which this feeds into.

    Returns (all_rows, reference_cam) — all_rows is passed on to
    calibrate_pooled() so the board-detection pass isn't repeated.
    """
    all_rows = cgroup.get_rows_videos(videos, board, verbose=False)
    cgroup.set_camera_sizes_videos(videos)
    cam_names = cgroup.get_names()

    # cv2.calibrateCamera's internal homography-based initial guess
    # (initIntrinsicParams2D) can hit a degenerate/singular homography on
    # frames with only a handful of near-collinear corners — confirmed on
    # real data (2026-08-27) as a hard OpenCV assertion failure at the
    # caller-supplied min_points=4 floor (fine for the extrinsics-only
    # pooled solve below, too permissive for this step). Use a stricter
    # floor here regardless of the caller's min_points, since a rich
    # reference camera has no shortage of well-conditioned frames.
    intrinsic_min_points = max(min_points, 10)

    per_cam = {}
    for cam_name, cam, rows in zip(cam_names, cgroup.cameras, all_rows):
        objp, imgp = board.get_all_calibration_points(rows, min_points=intrinsic_min_points)
        per_cam[cam_name] = (objp, imgp, cam.get_size())
        log_fn(f"[shared-intrinsics] {cam_name}: {len(objp)} usable frames "
               f"(>= {intrinsic_min_points} corners)")

    if reference_cam is None:
        reference_cam = max(per_cam, key=lambda c: len(per_cam[c][0]))
    elif reference_cam not in per_cam:
        raise ValueError(f"reference_cam {reference_cam!r} not among cameras {cam_names}")

    objp, imgp, size = per_cam[reference_cam]
    if len(objp) < 5:
        raise RuntimeError(
            f"[shared-intrinsics] Reference camera {reference_cam} has only "
            f"{len(objp)} usable frames (>= {intrinsic_min_points} corners) — too few "
            "for a reliable intrinsic calibration. Pick a richer "
            "reference_cam or lower min_points.")

    mismatched = {c: s for c, (_, _, s) in per_cam.items() if s != size}
    if mismatched:
        raise ValueError(
            f"[shared-intrinsics] Camera sizes differ ({reference_cam}={size} "
            f"vs {mismatched}) — shared intrinsics requires identical sensor "
            "resolution across all cameras.")

    # cv2.calibrateCamera jointly refines a per-view pose for every single
    # frame it's given, and a rich reference camera easily has 1000+ usable
    # frames. Measured directly on real data (2026-08-27): wall time scales
    # as roughly O(n^3) in view count (20->0.5s, 50->7.9s, 100->57.7s,
    # 150->199.8s) — this is what turned an unbounded ~1600-2000-view call
    # into a multi-hour run. Accuracy plateaus far below that: a 3-trial
    # random-subset sweep at n=8/10/15/20/30 found n=10 could occasionally
    # land a badly-conditioned subset (one trial solved to a 663px focal
    # length vs. an ~870-985px consensus everywhere else), while n>=15
    # consistently gave low-variance, low-error results (n=30: 0.73-0.80px
    # error across trials, ~910-930px focal, ~1.4s). This also matches the
    # general OpenCV/community guidance of ~20-40 well-distributed views for
    # calibrateCamera, with marginal gains beyond that. 30 was chosen as the
    # default to sit safely inside that literature range while staying clear
    # of the n=10 outlier risk, and it's still trivially fast. Subsample
    # evenly across the recording rather than truncating (or randomly, as
    # the sweep above did only to test robustness) so the kept frames span
    # the full range of board poses/depths instead of clustering in whatever
    # segment happened to be detected first.
    if len(objp) > max_frames:
        idx = np.linspace(0, len(objp) - 1, max_frames).astype(int)
        objp = [objp[i] for i in idx]
        imgp = [imgp[i] for i in idx]
        log_fn(f"[shared-intrinsics] Subsampled {reference_cam} to {max_frames} "
               f"frames (evenly spaced) for the intrinsic solve")

    flags = 0 if extra_dist else (cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
    reproj_err, matrix, dist, _, _ = cv2.calibrateCamera(
        objp, imgp, tuple(size), None, None, flags=flags)
    log_fn(f"[shared-intrinsics] Reference camera: {reference_cam} "
           f"({len(objp)} frames) — single-camera reprojection error: "
           f"{reproj_err:.4f} px, focal length: {matrix[0, 0]:.1f}px")

    focal_ratio = matrix[0, 0] / size[0]
    if focal_ratio > 5.0 or focal_ratio < 0.1:
        log_fn(f"[shared-intrinsics] WARNING: solved focal length "
               f"({matrix[0, 0]:.1f}px) is {focal_ratio:.1f}x the image "
               f"width ({size[0]}px) — implausible for typical optics. The "
               "reference camera's own footage may still lack enough depth/"
               "tilt variation; try a different reference camera or "
               "re-record with more depth variety in front of it.")

    for cam in cgroup.cameras:
        cam.set_camera_matrix(matrix.copy())
        cam.set_distortions(dist.copy())
        cam.set_size(size)

    log_fn(f"[shared-intrinsics] Applied {reference_cam}'s intrinsics to "
           f"all cameras: {cam_names}")
    return all_rows, reference_cam


def robust_filter_rows(cam, rows, board, mad_k=8.0, max_frame_rms=5.0,
                        min_corners_after=6, log_fn=print):
    """
    Per-camera, per-frame outlier rejection on already-pose-estimated rows
    (row['rvec']/row['tvec'] must be set — call after estimate_pose_rows()),
    applied BEFORE frames are pooled into the cross-camera bundle
    adjustment. Catches motion-blur/marker-misdetection corruption at the
    source, rather than downstream where a single bad corner can silently
    corrupt an entire joint solve.

    Real-data motivation (2026-09-03, F:\\TMZ7 CALIBRATION_0002): a per-
    camera-pair mm-accuracy check (triangulating detected corners and
    comparing same-frame pairwise distances against known board geometry)
    found some pairs (cam1-cam2, cam0-cam1) with a stable ~10-17mm MEDIAN
    error but a catastrophic tail -- p95 in the thousands of mm, max up to
    46 million mm -- while a 5-timepoint visual spot-check of the same
    recording showed no obviously mislabeled corner IDs. That combination
    (clean-looking detections, but occasional wildly-wrong 3D results) is
    the signature of a small number of individual frames with a corrupted
    corner position (most likely motion blur while the board was being
    actively moved/rotated, e.g. during a transition between camera-pair
    phases) rather than a systematic per-camera problem -- exactly what a
    per-frame reprojection-residual outlier check is designed to catch.

    Two-stage, per frame:
      1. Corner-level: compute each detected corner's reprojection residual
         against the frame's own already-solved pose. Drop any corner whose
         residual exceeds median + mad_k * MAD (a robust, scale-free
         outlier bound -- MAD is unaffected by the very outliers it's
         being used to detect, unlike a mean+stdev bound).
      2. Frame-level: re-solve the pose from the surviving inlier corners
         only (a single bad corner can drag solvePnP's fitted pose off for
         every other corner in that frame too), then reject the WHOLE frame
         if too few corners survived stage 1, or if the re-solved pose's
         own RMS reprojection error still exceeds max_frame_rms pixels.

    Returns (kept_rows, stats) where stats is a dict of counts for logging.
    """
    K = cam.get_camera_matrix()
    D = cam.get_distortions()
    all_obj_points = board.board.getChessboardCorners()

    kept_rows = []
    n_frames_in = 0
    n_frames_dropped_corners = 0
    n_frames_dropped_rms = 0
    n_corners_dropped = 0

    for row in rows:
        if row.get('ids') is None or row.get('rvec') is None or row.get('tvec') is None:
            continue
        n_frames_in += 1

        ids = row['ids'].flatten()
        corners = row['corners'].reshape(-1, 2)
        if len(ids) < min_corners_after:
            n_frames_dropped_corners += 1
            n_corners_dropped += len(ids)
            continue

        obj = np.array([all_obj_points[i] for i in ids], dtype=np.float64)
        proj, _ = cv2.projectPoints(obj, row['rvec'], row['tvec'], K, D)
        proj = proj.reshape(-1, 2)
        resid = np.linalg.norm(corners - proj, axis=1)

        med = np.median(resid)
        mad = np.median(np.abs(resid - med)) + 1e-9
        keep_mask = resid <= med + mad_k * mad

        if keep_mask.sum() < min_corners_after:
            n_frames_dropped_corners += 1
            n_corners_dropped += len(ids)
            continue

        n_dropped_here = int((~keep_mask).sum())
        if n_dropped_here > 0:
            ids = ids[keep_mask]
            corners = corners[keep_mask]
            obj = obj[keep_mask]
            pnp_flags = cv2.SOLVEPNP_ITERATIVE if len(ids) >= 6 else cv2.SOLVEPNP_EPNP
            ok, rvec2, tvec2 = cv2.solvePnP(
                obj.astype(np.float32), corners.astype(np.float32), K, D, flags=pnp_flags)
            if not ok:
                n_frames_dropped_corners += 1
                n_corners_dropped += len(ids)
                continue
            proj2, _ = cv2.projectPoints(obj, rvec2, tvec2, K, D)
            rms = float(np.sqrt(np.mean(np.sum((corners - proj2.reshape(-1, 2)) ** 2, axis=1))))
            n_corners_dropped += n_dropped_here
        else:
            rvec2, tvec2 = row['rvec'], row['tvec']
            rms = float(np.sqrt(np.mean(resid ** 2)))

        if rms > max_frame_rms:
            n_frames_dropped_rms += 1
            n_corners_dropped += len(ids)
            continue

        new_row = dict(row)
        new_row['ids'] = ids.reshape(-1, 1).astype(row['ids'].dtype)
        new_row['corners'] = corners.reshape(-1, 1, 2).astype(row['corners'].dtype)
        new_row['rvec'] = rvec2
        new_row['tvec'] = tvec2
        new_row['filled'] = board.fill_points(new_row['corners'], new_row['ids'])
        kept_rows.append(new_row)

    log_fn(f"[robust-filter] {cam.get_name()}: {n_frames_in} frames -> "
           f"{len(kept_rows)} kept "
           f"({n_frames_dropped_corners} dropped for too-few-inlier-corners, "
           f"{n_frames_dropped_rms} dropped for high residual RMS after "
           f"re-solve, {n_corners_dropped} individual corners removed total)")
    return kept_rows


def geometric_filter_correspondences(cgroup, imgp, extra, max_rel_error=0.5,
                                      min_abs_tol_mm=5.0, min_frac_inlier_pairs=0.5,
                                      min_points_per_camera=300, log_fn=print):
    """
    Cross-camera geometric-consistency outlier rejection on pooled
    correspondences, run AFTER a rough initial extrinsics guess (from
    get_initial_extrinsics) but BEFORE the final bundle adjustment.

    Motivation (real-data finding, 2026-09-03, F:\\TMZ7 CALIBRATION_0002):
    robust_filter_rows() alone (per-camera, per-frame reprojection-residual
    outlier rejection) only moved this rig's average reprojection error
    43.5px -> 41.8px, a ~4% improvement that dropped under 4% of corners --
    far too small to explain the pairwise mm-accuracy checks that had
    found errors up to tens of millions of mm even among points seen by
    ALL 4 cameras. The reason: a per-camera residual filter can only catch
    a corner that disagrees with the REST OF ITS OWN FRAME's fitted
    single-camera pose. It structurally cannot catch a classic planar-pose
    "flip" ambiguity, where a Charuco board viewed near-fronto-parallel to
    one camera has two near-equally-good solvePnP solutions -- EVERY
    corner in that frame is perfectly self-consistent with the (wrong)
    pose fit to them, so no per-camera residual ever looks bad, yet that
    whole frame's pose disagrees catastrophically with what every OTHER
    camera simultaneously saw. Only a genuinely cross-camera check (does
    this frame's board reconstruct into a physically sane rigid shape
    relative to every other camera's own view of it, not just internally
    with itself?) can catch it.

    For each original detection ("frame", i.e. one board pose instant --
    extra['ids'] groups points by which detection they came from), roughly
    triangulates that frame's points with the CURRENT (still rough) camera
    extrinsics and compares triangulated pairwise distances against the
    board's own known rigid geometry (extra['objp'], exact by construction
    since square_length_mm fixes the real-world scale). A frame whose
    reconstructed shape disagrees badly with the known board -- regardless
    of how internally consistent any single camera's corners looked -- is
    dropped ENTIRELY (all cameras, not just the offending one), since a
    flip-ambiguity failure corrupts that whole frame's pose, not one
    camera's corner subset.

    A frame with < 2 points, or where every pairwise ground-truth distance
    is too small to give a meaningful relative-error denominator, has no
    signal to check against and is kept by default (nothing to reject on).
    """
    pts3d = cgroup.triangulate(imgp, undistort=True)
    objp = extra['objp']
    frame_ids = extra['ids']

    by_frame = defaultdict(list)
    for idx, fid in enumerate(frame_ids):
        if not np.any(np.isnan(pts3d[idx])):
            by_frame[fid].append(idx)

    bad_frame_ids = set()
    n_checked = 0
    for fid, idxs in by_frame.items():
        if len(idxs) < 2:
            continue
        idxs = np.array(idxs)
        p3 = pts3d[idxs]
        po = objp[idxs]
        d3 = np.linalg.norm(p3[:, None, :] - p3[None, :, :], axis=-1)
        do = np.linalg.norm(po[:, None, :] - po[None, :, :], axis=-1)
        iu = np.triu_indices(len(idxs), k=1)
        do_pairs, d3_pairs = do[iu], d3[iu]
        mask = do_pairs > min_abs_tol_mm
        if mask.sum() == 0:
            continue
        n_checked += 1
        abs_err = np.abs(d3_pairs[mask] - do_pairs[mask])
        rel_err = abs_err / do_pairs[mask]
        inlier = (rel_err <= max_rel_error) | (abs_err <= min_abs_tol_mm)
        if inlier.mean() < min_frac_inlier_pairs:
            bad_frame_ids.add(fid)

    if not bad_frame_ids:
        log_fn(f"[geometric-filter] {n_checked} multi-camera frames checked, "
               f"0 rejected — no cross-camera geometric inconsistency found.")
        return imgp, extra, 0

    keep_mask = np.array([fid not in bad_frame_ids for fid in frame_ids])
    n_points_dropped = int((~keep_mask).sum())
    imgp2 = imgp[:, keep_mask]
    extra2 = {k: (v[keep_mask] if isinstance(v, np.ndarray) and v.shape[0] == len(keep_mask) else v)
              for k, v in extra.items()}

    # Safety net: never let filtering starve a camera of correspondences
    # below what aniposelib's own internal per-iteration subsampling
    # (bundle_adjust_iter's n_samp_iter=200 draw) needs to reliably include
    # that camera. Real crash, 2026-09-03 (F:\TMZ7 CALIBRATION_0002,
    # max_rel_error=0.5): filtering out 39.4% of frames left one camera so
    # thin that a random subsample drew zero of its points, producing a NaN
    # in that camera's bundle-adjustment parameter vector and crashing
    # scipy's least_squares with "Initial guess is outside of provided
    # bounds" (a NaN compares False against any bound, including the
    # default (-inf, inf), which is what actually raises this). Matches
    # this codebase's standing convention of never letting an experimental/
    # aggressive method crash the pipeline outright -- fall back to the
    # unfiltered correspondences and warn instead.
    n_seen_per_cam = np.sum(~np.isnan(imgp2[:, :, 0]), axis=1)
    if np.any(n_seen_per_cam < min_points_per_camera):
        starved = {cgroup.get_names()[i]: int(n) for i, n in enumerate(n_seen_per_cam)
                   if n < min_points_per_camera}
        log_fn(f"[geometric-filter] WARNING: rejecting {len(bad_frame_ids)} frames would "
               f"leave camera(s) {starved} below {min_points_per_camera} correspondence "
               f"points -- too aggressive for this data (would starve the bundle "
               f"adjustment's per-camera parameters and risk a crash). Reverting to the "
               f"UNFILTERED correspondence set for this run. Consider a looser "
               f"--geometric-max-rel-error or investigate why so many frames disagree "
               f"(e.g. --geometric-max-rel-error 1.0, or check that specific camera's "
               f"footage for a systematic issue).")
        return imgp, extra, 0
    log_fn(f"[geometric-filter] {n_checked} multi-camera frames checked, "
           f"{len(bad_frame_ids)} rejected as geometrically inconsistent "
           f"(likely planar-pose flip ambiguity or ID misassociation) — "
           f"{n_points_dropped} correspondence points removed.")
    return imgp2, extra2, len(bad_frame_ids)


def select_best_frames(merged, cam_names, top_k=150, min_cameras=3, log_fn=print):
    """
    Since the rig's extrinsics are fixed for the whole calibration
    recording (the cameras don't move between the first and last frame),
    a joint bundle adjustment doesn't need thousands of frames of wildly
    variable quality pooled together -- it needs a modest number of GOOD,
    well-distributed board poses. Pooling everything and then filtering out
    the bad fraction (see geometric_filter_correspondences) still leaves
    the initial extrinsics guess (get_initial_extrinsics) built from the
    FULL unfiltered correspondence graph, while the actual optimization
    then runs on a smaller, differently-shaped filtered set -- confirmed on
    real data (2026-09-03, F:\\TMZ7 CALIBRATION_0002) to crash aniposelib's
    bundle adjustment ("Initial guess is outside of provided bounds") even
    in a single-pass call with no internal resampling, i.e. the mismatch
    itself is the problem, not which bundle-adjust variant is used.

    This selects the best `top_k` frames FIRST, before anything else is
    derived from the data -- extract_points, extract_rtvecs, and the
    initial extrinsics guess are then all built from the SAME small,
    curated set, so the initial guess and the optimization data are always
    mutually consistent by construction.

    A frame is scored by (a) how many cameras saw it with a usable pose
    (more camera coverage directly improves joint-solve conditioning) and
    (b) total corner count summed across those cameras (more corners means
    a better-constrained per-frame pose) -- camera coverage dominates the
    score (weighted first) since a frame all 4 cameras saw is far more
    valuable to a joint extrinsics solve than a frame with more corners
    but only 2 cameras. Frames seen by fewer than `min_cameras` cameras
    contribute little to inter-camera constraints and are excluded from
    ranking entirely, not merely deprioritized.

    `min_cameras` is silently clamped to `len(cam_names)` -- this became
    the default production behavior for ANY rig (see auto_calibrate's
    best_frames_top_k=100 default), including a possible future 2-camera
    rig, where a literal min_cameras=3 would find zero eligible frames and
    select nothing (min_cameras=3 was only ever validated/intended for the
    4-camera rig this was tuned on).
    """
    min_cameras = min(min_cameras, len(cam_names))
    scored = []
    for i, d in enumerate(merged):
        cams_present = [c for c in cam_names
                         if c in d and d[c].get('ids') is not None]
        if len(cams_present) < min_cameras:
            continue
        total_corners = sum(int(d[c]['ids'].size) for c in cams_present)
        score = (len(cams_present), total_corners)
        scored.append((score, i))
    scored.sort(reverse=True)
    keep_idx = {i for _, i in scored[:top_k]}
    selected = [d for i, d in enumerate(merged) if i in keep_idx]
    log_fn(f"[best-frames] Selected {len(selected)} of {len(merged)} multi-camera "
           f"frames (top {top_k} by camera-coverage then corner-count, requiring "
           f">= {min_cameras} cameras per frame) for a curated calibration solve.")
    return selected


def _held_out_median_mm_error(cgroup_fitted, board, cam_names, merged_all,
                               min_points, eval_min_cameras):
    """
    Real, held-out accuracy check: triangulate every correspondence in
    `merged_all` (the FULL detected pool, not whatever subset a candidate
    calibration was fit on) with the already-fitted `cgroup_fitted`,
    restrict to points seen by every one of `eval_min_cameras` cameras
    (the best-conditioned, most trustworthy geometry), and compare
    triangulated pairwise distances against the board's own known
    geometry (exact by construction from square_length_mm). Returns
    (median_mm, n_comparisons) -- median_mm is None if too few points
    qualify to give a meaningful number.

    This is the SAME methodology used to hand-validate best_frames_top_k
    =100 (2026-09-03, F:\\TMZ7): fit-set reprojection error alone is
    misleading for choosing K, since a smaller K's own reprojection error
    tends to look artificially better simply because there's less/less-
    diverse data to disagree with, not necessarily because the resulting
    calibration is more accurate on data it WASN'T fit on.
    """
    imgp_all, extra_all = extract_points(merged_all, board, cam_names=cam_names,
                                          min_cameras=2, min_points=min_points)
    n_seen = np.sum(~np.isnan(imgp_all[:, :, 0]), axis=0)
    keep = n_seen >= eval_min_cameras
    if keep.sum() < 20:
        return None, 0

    pts3d = cgroup_fitted.triangulate(imgp_all[:, keep], undistort=True)
    objp = extra_all['objp'][keep]
    fids = extra_all['ids'][keep]

    by_frame = defaultdict(list)
    for idx, fid in enumerate(fids):
        if not np.any(np.isnan(pts3d[idx])):
            by_frame[fid].append(idx)

    errs = []
    for fid, idxs in by_frame.items():
        if len(idxs) < 2:
            continue
        idxs = np.array(idxs)
        p3, po = pts3d[idxs], objp[idxs]
        d3 = np.linalg.norm(p3[:, None, :] - p3[None, :, :], axis=-1)
        do = np.linalg.norm(po[:, None, :] - po[None, :, :], axis=-1)
        iu = np.triu_indices(len(idxs), k=1)
        diff = np.abs(d3[iu] - do[iu])
        m = do[iu] > 1.0
        errs.append(diff[m])
    if not errs:
        return None, 0
    errs = np.concatenate(errs)
    return float(np.median(errs)), int(errs.size)


def auto_select_best_frames_top_k(cgroup, board, cam_names, merged_all,
                                   candidate_ks=(50, 75, 100, 150, 250, 400),
                                   min_points=9, min_cameras=3,
                                   eval_min_cameras=None, log_fn=print):
    """
    Automatically choose best_frames_top_k instead of a hardcoded value,
    by testing a small, BOUNDED grid of candidates and picking the one
    with the best HELD-OUT accuracy (see _held_out_median_mm_error) --
    not the fit set's own reprojection error, which is a misleading
    selection criterion (see that function's docstring).

    Kept fast deliberately: corner DETECTION (the genuinely slow ~2-3 min
    step) is done exactly once by the caller and never repeated here --
    `merged_all` and `cgroup` (for its already-resolved intrinsics) are
    passed in already built. Each candidate's fit + held-out evaluation
    only takes a few seconds once corners are detected (confirmed on real
    data, 2026-09-03: an 8-point manual sweep completed in well under a
    minute once detection was cached), so testing ~6 candidates adds
    negligible time on top of the one detection pass -- the grid is
    intentionally small and geometrically-ish spaced (not a fine linear
    sweep) rather than exhaustive, trading a little precision in the
    chosen K for a bounded, predictable runtime.

    `eval_min_cameras` defaults to the rig's full camera count (the
    strictest, most trustworthy check) but is silently clamped so a
    2-camera rig doesn't require an unreachable 4-camera check.
    """
    n_cams = len(cam_names)
    if eval_min_cameras is None:
        eval_min_cameras = n_cams
    eval_min_cameras = min(eval_min_cameras, n_cams)

    results = []
    for k in candidate_ks:
        cg = make_camera_group(cam_names, extra_dist=False)
        for cam_new, cam_ref in zip(cg.cameras, cgroup.cameras):
            cam_new.set_camera_matrix(cam_ref.get_camera_matrix().copy())
            cam_new.set_distortions(cam_ref.get_distortions().copy())
            cam_new.set_size(cam_ref.get_size())

        merged_k = select_best_frames(merged_all, cam_names, top_k=k,
                                       min_cameras=min_cameras, log_fn=lambda *a: None)
        if len(merged_k) < 10:
            log_fn(f"[auto-best-frames] k={k}: only {len(merged_k)} frames available "
                   f"(< 10 minimum) — skipping this candidate.")
            continue

        imgp_k, extra_k = extract_points(merged_k, board, min_cameras=2, min_points=min_points)
        rtvecs = extract_rtvecs(merged_k, cam_names=cam_names, min_cameras=2)
        rvecs, tvecs = get_initial_extrinsics(rtvecs, cam_names)
        cg.set_rotations(rvecs)
        cg.set_translations(tvecs)
        try:
            cg.bundle_adjust_iter(imgp_k, extra_k, verbose=False, only_extrinsics=True)
        except Exception as e:
            log_fn(f"[auto-best-frames] k={k}: fit failed ({e}) — skipping this candidate.")
            continue

        median_mm, n_cmp = _held_out_median_mm_error(
            cg, board, cam_names, merged_all, min_points, eval_min_cameras)
        if median_mm is None:
            log_fn(f"[auto-best-frames] k={k}: too few >= {eval_min_cameras}-camera "
                   f"points to evaluate — skipping this candidate.")
            continue

        results.append((k, median_mm, len(merged_k)))
        log_fn(f"[auto-best-frames] k={k}: {len(merged_k)} frames used, held-out "
               f"median error ({n_cmp} comparisons, >= {eval_min_cameras}-cam points) "
               f"= {median_mm:.3f}mm")

    if not results:
        log_fn(f"[auto-best-frames] No candidate succeeded — falling back to "
               f"k={candidate_ks[len(candidate_ks) // 2]} (grid midpoint).")
        return candidate_ks[len(candidate_ks) // 2]

    best_k, best_mm, best_n = min(results, key=lambda r: r[1])
    log_fn(f"[auto-best-frames] Selected k={best_k} (held-out median error "
           f"{best_mm:.3f}mm) from candidates: "
           f"{[(k, round(mm, 3)) for k, mm, _ in results]}")
    return best_k


def select_depth_augmentation_frames(merged, cam_names, per_pair_cap=40,
                                      min_corners=15, n_depth_buckets=5,
                                      log_fn=print):
    """
    select_best_frames() deliberately excludes any frame seen by fewer than
    3 cameras -- which throws away exactly the "move the board toward/away
    from ONE camera pair, out of view of the other two" depth-sweep phases
    a real capture protocol often includes (see user description,
    2026-09-03). That data is thrown away for good reason where it's used
    for GLOBAL rig linking (a 2-camera-only frame can't connect that pair
    to the rest of the rig) -- but it's genuinely valuable for refining
    just THAT pair's own depth/translation-scale: varying distance is the
    textbook-correct way to condition a stereo pair's baseline, which a
    single fixed distance leaves poorly constrained.

    This recovers that data as a SEPARATE, explicitly-scored pool, one
    camera pair at a time: candidate frames are exactly the merged
    detections where the visibility pattern is EXACTLY {camA, camB} (both
    of a pair, nobody else) -- matching the literal "disappears from the
    other two" capture pattern -- with at least `min_corners` on each
    camera. To make sure real depth VARIATION is captured (not just many
    near-duplicate poses at whatever single distance happened to detect
    best), candidates are bucketed by an apparent-board-scale proxy (mean
    pairwise pixel distance between camA's own detected corners -- smaller
    means farther away, larger means closer) into `n_depth_buckets` even
    quantile bins, then the best (by total corner count) candidate is
    taken from each bucket in round-robin order until `per_pair_cap` is
    reached -- so near AND far poses are both represented, not just
    whichever distance this recording happened to linger at longest.

    Returned frames are meant to be combined with select_best_frames()'s
    anchor frames into ONE joint bundle adjustment (see
    refine_with_depth_augmentation), not solved in isolation and grafted
    back onto the rig afterward -- an independently-solved pair pose would
    only be self-consistent with itself, not with the globally-solved
    other cameras.
    """
    import itertools as _it
    all_added = []
    for cam_a, cam_b in _it.combinations(cam_names, 2):
        pair_set = {cam_a, cam_b}
        candidates = []
        for i, d in enumerate(merged):
            cams_present = frozenset(c for c in cam_names
                                      if c in d and d[c].get('ids') is not None)
            if cams_present != pair_set:
                continue
            na = int(d[cam_a]['ids'].size)
            nb = int(d[cam_b]['ids'].size)
            if na < min_corners or nb < min_corners:
                continue
            corners_a = d[cam_a]['corners'].reshape(-1, 2)
            if len(corners_a) >= 2:
                scale = float(np.mean(np.linalg.norm(
                    corners_a[:, None, :] - corners_a[None, :, :], axis=-1)))
            else:
                scale = 0.0
            candidates.append((i, scale, na + nb))

        if not candidates:
            continue

        scales = np.array([c[1] for c in candidates])
        bucket_edges = np.quantile(scales, np.linspace(0, 1, n_depth_buckets + 1))
        buckets = [[] for _ in range(n_depth_buckets)]
        for cand in candidates:
            b = min(np.searchsorted(bucket_edges, cand[1], side='right') - 1,
                    n_depth_buckets - 1)
            buckets[max(b, 0)].append(cand)
        for b in buckets:
            b.sort(key=lambda c: -c[2])  # best (most corners) first within its bucket

        picked = []
        bi = 0
        while len(picked) < per_pair_cap and any(buckets):
            if buckets[bi % n_depth_buckets]:
                picked.append(buckets[bi % n_depth_buckets].pop(0))
            bi += 1
            if bi > n_depth_buckets * (per_pair_cap + 1):
                break

        log_fn(f"[depth-augment] {cam_a}-{cam_b}: {len(candidates)} exact-pair-only "
               f"depth-sweep frames found, selected {len(picked)} spanning "
               f"{n_depth_buckets} depth buckets (scale range "
               f"{scales.min():.0f}-{scales.max():.0f}px).")
        all_added.extend(merged[i] for i, _, _ in picked)

    return all_added


def refine_with_depth_augmentation(cgroup, board, cam_names, merged_anchor,
                                    merged_all, min_points=9,
                                    depth_per_pair_cap=40, depth_min_corners=15,
                                    depth_geometric_max_rel_error=0.5,
                                    depth_geometric_min_abs_tol_mm=5.0,
                                    log_fn=print) -> float:
    """
    Stage 2 of a two-stage calibration: `cgroup` must already hold a
    trustworthy solved extrinsics (e.g. from select_best_frames() + a
    successful bundle adjustment -- the "anchor" solve). This adds
    per-camera-pair depth-sweep frames (select_depth_augmentation_frames)
    to refine each pair's depth/translation-scale specifically, screened
    against the ALREADY-SOLVED anchor calibration (a real, trustworthy
    reference -- not the rough graph-based initial guess used elsewhere,
    which is what made an earlier attempt at this unreliable).

    Critically, this does NOT rebuild the initial extrinsics guess from
    scratch (which is what caused the earlier "Initial guess is outside of
    provided bounds" crash when combined with post-hoc filtering) --
    cgroup's cameras are left exactly as the anchor solve set them, and a
    single bundle_adjust() call is made directly, so its internal x0 is
    built from that already-converged, self-consistent state. Only kept
    if it doesn't regress the anchor-only reprojection error.
    """
    depth_frames = select_depth_augmentation_frames(
        merged_all, cam_names, per_pair_cap=depth_per_pair_cap,
        min_corners=depth_min_corners, log_fn=log_fn)
    if not depth_frames:
        log_fn("[depth-augment] No exact-pair-only depth-sweep frames found — skipping.")
        return None

    imgp_anchor, extra_anchor = extract_points(merged_anchor, board, min_cameras=2,
                                                min_points=min_points)
    anchor_error = cgroup.average_error(imgp_anchor, median=True)
    log_fn(f"[depth-augment] Anchor-only baseline (median reprojection error): "
           f"{anchor_error:.4f} px")

    # Screen depth candidates against the TRUSTED anchor calibration (not a
    # rough initial guess) before adding them to the combined pool.
    imgp_depth, extra_depth = extract_points(depth_frames, board, min_cameras=2,
                                              min_points=min_points)
    imgp_depth, extra_depth, n_rejected = geometric_filter_correspondences(
        cgroup, imgp_depth, extra_depth,
        max_rel_error=depth_geometric_max_rel_error,
        min_abs_tol_mm=depth_geometric_min_abs_tol_mm, log_fn=log_fn)

    merged_combined = merged_anchor + depth_frames
    imgp_combined, extra_combined = extract_points(merged_combined, board,
                                                     min_cameras=2, min_points=min_points)

    try:
        refined_error = cgroup.bundle_adjust(imgp_combined, extra_combined,
                                              loss='soft_l1', ftol=1e-4,
                                              max_nfev=1000, only_extrinsics=True,
                                              verbose=False)
    except Exception as e:
        log_fn(f"[depth-augment] WARNING: refinement bundle adjustment failed ({e}) "
               f"— keeping the anchor-only calibration untouched.")
        return None

    refined_median_error = cgroup.average_error(imgp_anchor, median=True)
    log_fn(f"[depth-augment] After depth-augmented refinement: {refined_error:.4f} px "
           f"(combined-pool mean), {refined_median_error:.4f} px (anchor-frames-only "
           f"median, vs {anchor_error:.4f} px baseline)")

    if refined_median_error > anchor_error * 1.02:
        log_fn(f"[depth-augment] WARNING: refinement regressed the anchor-frame "
               f"reprojection error ({anchor_error:.4f} -> {refined_median_error:.4f} px) "
               f"— this is a real risk (grafting depth constraints CAN pull the "
               f"solve away from the global-linking optimum if the depth data's own "
               f"scale is noisy). Caller should compare and decide whether to keep "
               f"this result or fall back to the anchor-only calibration.")

    return refined_error


def _pairwise_codetection_stats(all_rows, cam_names):
    """
    For every camera pair, the fraction of frames where BOTH cameras
    detected *any* board markers, relative to the rarer of the two
    cameras' own total detection count. Computed from raw per-frame
    detections (not the pose-floor-filtered rows), so it reflects genuine
    scene co-visibility rather than an artifact of min_points. Independent
    of any calibration solve — usable purely from board.get_rows_videos()
    output.
    """
    frame_ids = {}
    for cam, rows in zip(cam_names, all_rows):
        s = {}
        for r in rows:
            ids = r.get('ids')
            if ids is not None and len(ids) > 0:
                s[r['framenum']] = len(ids)
        frame_ids[cam] = s
    stats = {}
    for i in range(len(cam_names)):
        for j in range(i + 1, len(cam_names)):
            a, b = cam_names[i], cam_names[j]
            common = set(frame_ids[a]) & set(frame_ids[b])
            denom = min(len(frame_ids[a]), len(frame_ids[b])) or 1
            stats[(a, b)] = len(common) / denom
    return stats


def detect_ambiguous_camera_pairs(cgroup, all_rows, cam_names,
                                   angle_threshold_deg=40.0,
                                   codetect_ratio_factor=1.15,
                                   log_fn=print):
    """
    Auto-detect camera pairs whose correspondence data is contaminated by
    a double-sided calibration target (two Charuco boards glued back-to-
    back, sharing an identical marker-ID layout) — see D:\\3D DLC\\CLAUDE.md's
    TMZ7 triangulation investigation (2026-09-09) for the full diagnosis.

    Root cause this guards against: aniposelib's per-frame board-pose
    solve (CharucoBoard.estimate_pose_points -> cv2.solvePnP) has no
    notion of "this detection might be the mirrored back face" — it
    always fits ONE canonical board template. When two cameras sit
    roughly opposite each other on a rig (e.g. a "rough cube" with one
    camera per face) and the calibration target is double-sided, each of
    those cameras consistently sees a DIFFERENT physical print (front vs
    back) but decodes the SAME marker IDs, so aniposelib's cross-camera
    relative-pose composition (get_initial_extrinsics/get_connections)
    silently treats them as observing one shared rigid board — producing
    a self-consistent-looking but wrong relative pose. Worse: because the
    target is genuinely visible from both sides whenever it's held up,
    this INFLATES exactly those pairs' co-detection count, which is
    precisely what aniposelib's max-spanning-tree graph builder
    (get_calibration_graph) prioritizes — so the corrupted pairs don't
    just add noise, they get selected as the backbone of the initial
    extrinsics guess (confirmed directly on F:\\TMZ7 CALIBRATION_0002: the
    resulting spanning tree picked the two contaminated pairs first).

    A genuine single-sided target produces the OPPOSITE pattern: two
    cameras facing away from each other rarely see the same one-sided
    sheet at once, so co-detection rate should correlate with SMALL
    viewing-angle separation, not large. This function flags a pair only
    when both anomalies coincide: an unusually HIGH co-detection rate
    (>= codetect_ratio_factor x the rig's own median pairwise ratio) AND
    a small forward-axis angle (< angle_threshold_deg) computed from
    whatever calibration was just solved — the inverted-correlation
    fingerprint a genuinely near-duplicate-viewpoint mounting would not
    itself produce (nothing scene-dependent should drive co-detection
    rate specifically for one pair on a rig with a normal single-sided
    target, regardless of how close together two cameras happen to be).

    Only meaningful for a rig this technique was designed for — an even
    number of cameras > 3, built (or suspected to be built) as opposite-
    facing pairs. Returns an empty set immediately for any other camera
    count, since "opposite pair" isn't well defined there and this
    heuristic should never fire spuriously on e.g. a genuinely-linear or
    odd-camera-count rig.
    """
    n = len(cam_names)
    if n <= 3 or n % 2 != 0:
        return set()

    ratios = _pairwise_codetection_stats(all_rows, cam_names)
    if not ratios:
        return set()
    med = float(np.median(list(ratios.values())))

    centers, forwards = {}, {}
    for i, cam in enumerate(cam_names):
        cam_obj = cgroup.cameras[i]
        rvec = np.array(cam_obj.get_rotation(), dtype=float).reshape(3)
        t = np.array(cam_obj.get_translation(), dtype=float).reshape(3)
        R, _ = cv2.Rodrigues(rvec)
        centers[cam] = -R.T @ t
        forwards[cam] = R.T @ np.array([0.0, 0.0, 1.0])

    flagged = set()
    for (a, b), ratio in ratios.items():
        denom = (np.linalg.norm(forwards[a]) * np.linalg.norm(forwards[b])) or 1e-12
        cosang = np.dot(forwards[a], forwards[b]) / denom
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        log_fn(f"[ambiguous-pair] {a}-{b}: co-detect ratio={ratio:.2f} "
               f"(rig median={med:.2f}), computed angle={ang:.1f} deg")
        if ratio >= med * codetect_ratio_factor and ang < angle_threshold_deg:
            flagged.add((a, b))
            log_fn(f"[ambiguous-pair]   -> FLAGGED: high co-detection + small angle "
                   f"is the double-sided-target contamination fingerprint")
    return flagged


def _drop_conflicting_pair_rows(new_rows, cam_names, ambiguous_pairs, log_fn=print):
    """
    For every frame where BOTH cameras of a flagged pair have a posed
    detection, drop the weaker one (fewer corners) from ITS OWN per-camera
    row list. This breaks only the specific cross-pair link that
    get_calibration_graph()/extract_points() would otherwise use to (wrongly)
    tie those two cameras together via a shared "board pose" — every other
    camera's correspondence with either one, in the same or any other
    frame, is left completely untouched, so no real data is discarded
    beyond what's strictly needed to remove the contaminated linkage.
    """
    idx = {cam: i for i, cam in enumerate(cam_names)}
    by_frame = [{r['framenum']: r for r in rows} for rows in new_rows]
    drop_framenums = [set() for _ in cam_names]

    for a, b in ambiguous_pairs:
        ia, ib = idx[a], idx[b]
        common = set(by_frame[ia].keys()) & set(by_frame[ib].keys())
        for fn in common:
            na = len(by_frame[ia][fn]['ids'])
            nb = len(by_frame[ib][fn]['ids'])
            if na >= nb:
                drop_framenums[ib].add(fn)
            else:
                drop_framenums[ia].add(fn)

    n_dropped = sum(len(s) for s in drop_framenums)
    filtered = [
        [r for r in rows if r['framenum'] not in drop_framenums[i]]
        for i, rows in enumerate(new_rows)
    ]
    log_fn(f"[ambiguous-pair] Dropped {n_dropped} conflicting per-frame detections "
           f"across {len(ambiguous_pairs)} flagged pair(s) to break the false "
           f"shared-board linkage between them.")
    return filtered


def calibrate_pooled(cgroup, board, videos, min_points=4, verbose=True,
                      fixed_intrinsics=False, all_rows=None,
                      robust_filter=True, outlier_mad_k=8.0, max_frame_rms=5.0,
                      geometric_filter=False, geometric_max_rel_error=0.5,
                      geometric_min_abs_tol_mm=5.0,
                      best_frames_top_k="auto", best_frames_min_cameras=3,
                      best_frames_min_selected=10,
                      auto_exclude_ambiguous_pairs=True,
                      ambiguous_pair_angle_threshold_deg=40.0,
                      ambiguous_pair_codetect_ratio_factor=1.15,
                      log_fn=print):
    """
    Reimplementation of CameraGroup.calibrate_rows() that pools every
    detected frame (instead of discarding frames below aniposelib's
    hardcoded per-frame floors of 5/7/8 corners) into one joint bundle
    adjustment. Intended for boards that are physically too small to ever
    clear those floors in a single frame — relies on having many frames
    (hundreds+) per camera so the aggregate solve can average out each
    individual frame's weak/noisy pose estimate. See LowFloorCharucoBoard.

    fixed_intrinsics: when True, cgroup's cameras already carry trusted
    intrinsics (via apply_fixed_intrinsics or calibrate_shared_intrinsics)
    — skip re-estimating them from this session's weak per-frame data
    (which is what produced physically nonsensical focal lengths before)
    and solve only extrinsics.

    all_rows: reuse a board-detection pass already done by the caller
    (e.g. calibrate_shared_intrinsics) instead of re-decoding every video.

    auto_exclude_ambiguous_pairs: run the solve once, then check the
    result for the double-sided-target contamination fingerprint (see
    detect_ambiguous_camera_pairs) and, if found, re-solve with those
    pairs' cross-camera correspondences excluded. A true no-op (single
    solve, byte-identical to the pre-existing behavior) for any rig that
    doesn't show the fingerprint — including any rig with an odd or <=3
    camera count — so this defaults on without risk to already-working
    calibrations (e.g. F:\\3D CAM TEST\\TRY 3).
    """
    if all_rows is None:
        all_rows = cgroup.get_rows_videos(videos, board, verbose=verbose)

    cam_names = cgroup.get_names()

    def _solve_once(rows_in, log_fn=log_fn):
        if fixed_intrinsics:
            log_fn("Using fixed intrinsics — solving extrinsics only.")
        else:
            cgroup.set_camera_sizes_videos(videos)
            for rows, camera in zip(rows_in, cgroup.cameras):
                size = camera.get_size()
                objp, imgp = board.get_all_calibration_points(rows, min_points=min_points)
                mixed = [(o, i) for (o, i) in zip(objp, imgp) if len(o) >= min_points]
                if not mixed:
                    raise RuntimeError(
                        f"No frames with >= {min_points} corners for camera "
                        f"{camera.get_name()} — cannot estimate intrinsics.")
                objp2, imgp2 = zip(*mixed)
                matrix = cv2.initCameraMatrix2D(objp2, imgp2, tuple(size))
                camera.set_camera_matrix(matrix.copy())
                camera.zero_distortions()

        log_fn(str(cgroup.get_dicts()))

        rows_posed = list(rows_in)
        for i, (rows, cam) in enumerate(zip(rows_posed, cgroup.cameras)):
            rows_posed[i] = board.estimate_pose_rows(cam, rows)

        new_rows = [[r for r in rows if r['ids'] is not None and r['ids'].size >= min_points]
                    for rows in rows_posed]
        n_posed = [len(r) for r in new_rows]
        log_fn(f"Frames with a usable per-frame pose (>= {min_points} corners): {n_posed}")

        if robust_filter:
            new_rows = [
                robust_filter_rows(cam, rows, board, mad_k=outlier_mad_k,
                                    max_frame_rms=max_frame_rms,
                                    min_corners_after=min_points, log_fn=log_fn)
                for cam, rows in zip(cgroup.cameras, new_rows)
            ]

        return rows_posed, new_rows

    def _solve_from_new_rows(new_rows):
        merged = merge_rows(new_rows, cam_names=cam_names)

        top_k = best_frames_top_k
        if top_k == "auto":
            top_k = auto_select_best_frames_top_k(
                cgroup, board, cam_names, merged, min_points=min_points,
                min_cameras=best_frames_min_cameras, log_fn=log_fn)

        merged_use = merged
        if top_k and top_k > 0:
            merged_all = merged
            merged_use = select_best_frames(merged_all, cam_names, top_k=top_k,
                                             min_cameras=best_frames_min_cameras, log_fn=log_fn)
            if len(merged_use) < best_frames_min_selected:
                log_fn(f"[best-frames] WARNING: only {len(merged_use)} frames qualified "
                       f"(< {best_frames_min_selected} minimum) — too few for a reliable "
                       f"curated solve on this recording (short capture, or few frames "
                       f"with >= {best_frames_min_cameras} cameras visible at once). "
                       f"Falling back to the full unfiltered/robust-filtered pool instead.")
                merged_use = merged_all

        imgp, extra = extract_points(merged_use, board, min_cameras=2, min_points=min_points)
        log_fn(f"Pooled cross-camera point correspondences going into bundle adjustment: {imgp.shape[1]}")

        rtvecs = extract_rtvecs(merged_use, cam_names=cam_names, min_cameras=2)
        log_fn(str(get_connections(rtvecs, cam_names)))
        rvecs, tvecs = get_initial_extrinsics(rtvecs, cam_names)
        cgroup.set_rotations(rvecs)
        cgroup.set_translations(tvecs)

        imgp_unfiltered, extra_unfiltered = imgp, extra

        if not geometric_filter:
            error = cgroup.bundle_adjust_iter(imgp_unfiltered, extra_unfiltered,
                                               verbose=verbose, only_extrinsics=fixed_intrinsics)
            return error

        imgp_f, extra_f, _ = geometric_filter_correspondences(
            cgroup, imgp_unfiltered, extra_unfiltered,
            max_rel_error=geometric_max_rel_error,
            min_abs_tol_mm=geometric_min_abs_tol_mm, log_fn=log_fn)

        # bundle_adjust_iter() does its OWN internal iterative reweighting +
        # random per-iteration resampling (n_samp_iter=200, evolving error
        # threshold mu) -- redundant, and confirmed actively UNSTABLE, once
        # we've already cross-camera filtered: on real data (2026-09-03,
        # F:\TMZ7 CALIBRATION_0002), bundle_adjust_iter crashed with "Initial
        # guess is outside of provided bounds" on the filtered correspondences
        # regardless of how loose the geometric filter's own threshold was made
        # (tried 0.5/1.0/2.0 relative error, all three crashed identically) --
        # the instability is in aniposelib's internal random resampling
        # interacting with an already non-uniformly-filtered point set, not in
        # the filter's aggressiveness, so retrying with a looser threshold
        # cannot fix it. A single-pass cgroup.bundle_adjust() call (soft_l1
        # robust loss, no internal resampling loop) sidesteps the unstable code
        # path entirely -- the natural choice once outlier rejection has
        # already been done externally, since bundle_adjust_iter's whole
        # purpose (iteratively downweighting outliers via repeated resampling)
        # is now redundant with what the geometric filter already did.
        try:
            return cgroup.bundle_adjust(imgp_f, extra_f, loss='soft_l1',
                                         ftol=1e-4, max_nfev=1000,
                                         only_extrinsics=fixed_intrinsics,
                                         verbose=verbose)
        except Exception as e:
            log_fn(f"[geometric-filter] WARNING: single-pass bundle adjustment on the "
                   f"filtered correspondences failed ({e}) -- falling back to "
                   f"bundle_adjust_iter() on the UNFILTERED correspondences.")
            return cgroup.bundle_adjust_iter(imgp_unfiltered, extra_unfiltered,
                                              verbose=verbose, only_extrinsics=fixed_intrinsics)

    all_rows, new_rows = _solve_once(all_rows)
    error = _solve_from_new_rows(new_rows)

    if auto_exclude_ambiguous_pairs:
        # Iterate: excluding one round of flagged pairs can shift the
        # solved geometry enough to reveal further contamination that
        # wasn't visible against the original (badly corrupted) angle
        # estimate -- confirmed on real TMZ7 data, where a single pass
        # left one previously-"clean" adjacent pair sitting at an
        # implausible ~31 deg instead of ~90 deg. Capped at a small
        # number of rounds since each round can only ever shrink the
        # remaining correspondence graph, and total exclusion (all_flagged)
        # is monotonically non-decreasing, so this always terminates.
        all_flagged = set()
        for round_num in range(1, 4):
            ambiguous = detect_ambiguous_camera_pairs(
                cgroup, all_rows, cam_names,
                angle_threshold_deg=ambiguous_pair_angle_threshold_deg,
                codetect_ratio_factor=ambiguous_pair_codetect_ratio_factor,
                log_fn=log_fn)
            new_flags = ambiguous - all_flagged
            if not new_flags:
                if round_num > 1:
                    log_fn(f"[ambiguous-pair] round {round_num}: no new pairs "
                           f"flagged — stable, stopping.")
                break
            all_flagged |= new_flags
            log_fn(f"[ambiguous-pair] round {round_num}: {len(new_flags)} new "
                   f"pair(s) flagged ({sorted(new_flags)}); total excluded so far: "
                   f"{sorted(all_flagged)}")
            error_before = error
            new_rows = _drop_conflicting_pair_rows(
                new_rows, cam_names, new_flags, log_fn=log_fn)
            error = _solve_from_new_rows(new_rows)
            log_fn(f"[ambiguous-pair] round {round_num}: average_error "
                   f"before={error_before:.3f}, after={error:.3f}")
        else:
            log_fn(f"[ambiguous-pair] WARNING: still flagging new pairs after "
                   f"3 rounds — stopping to avoid over-excluding; final excluded "
                   f"set: {sorted(all_flagged)}")

    return error


def discover_camera_videos(videos_dir: Path) -> dict[str, Path]:
    """
    Find one video per camera in *videos_dir*, keyed by camN label.
    Expects "camN_..." filenames (Step 0's copy_calibration_sessions()
    naming convention). Raises if a camN has more than one matching file
    (ambiguous — trim the folder to exactly one video per camera).
    """
    by_cam: dict[str, list[Path]] = {}
    for p in sorted(videos_dir.iterdir()):
        if not p.is_file():
            continue
        m = re.match(r"^(cam\d+)_", p.name)
        if not m:
            continue
        by_cam.setdefault(m.group(1), []).append(p)

    ambiguous = {cam: files for cam, files in by_cam.items() if len(files) > 1}
    if ambiguous:
        raise ValueError(
            "More than one video per camera in "
            f"{videos_dir}: {ambiguous}\n"
            "Keep exactly one calibration recording per camera in this folder.")

    if len(by_cam) < 2:
        raise ValueError(
            f"Found only {len(by_cam)} camera video(s) in {videos_dir}; "
            "need at least 2 for calibration. Expected files named "
            "'camN_...' (e.g. cam0_Cam1BCF-28C4-01_Series0001.mp4).")

    return {cam: files[0] for cam, files in by_cam.items()}


def find_calibration_candidates(source_folders) -> list[Path]:
    """
    Scan *source_folders* (recursively) for calibration-recording directories:
    any directory whose own name contains "calibration" (case-insensitive)
    that directly holds >= 2 camN-prefixed video files.

    This matches Step 0's own calibration-session convention exactly
    (bonsai_video_corrector.py's split_calibration_sessions() flags any
    session_id containing "calibration", case-insensitive, and
    copy_calibration_sessions() copies its camN-prefixed videos into
    "{out_dir}/_calibration/{session_id}/" — a directory name that itself
    contains "calibration"), so Step 0 output is picked up automatically.
    It also picks up any other folder a user names/places with "calibration"
    in it, containing camN-prefixed videos directly.
    """
    candidates: list[Path] = []
    seen: set = set()
    for sf in source_folders:
        sf = Path(sf)
        if not sf.is_dir():
            continue
        for d in sf.rglob("*"):
            if not d.is_dir() or "calibration" not in d.name.lower():
                continue
            if d in seen:
                continue
            seen.add(d)
            try:
                cam_videos = discover_camera_videos(d)
            except ValueError:
                continue
            if len(cam_videos) >= 2:
                candidates.append(d)
    return candidates


def pick_best_calibration_candidate(candidates: list[Path]):
    """
    Most recent candidate by directory name. Step 0 session IDs embed an
    ISO-8601 timestamp, so lexicographic order on the name matches
    chronological order (same convention used elsewhere in this codebase,
    e.g. cube_3d_dlc.py's session sort) — the most recently recorded
    calibration is the one most likely to reflect the rig's current setup.
    """
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def auto_calibrate(videos_dir: Path, out_dir: Path, *,
                    squares_x: int, squares_y: int,
                    square_length: float, marker_length: float,
                    marker_bits: int = 4, dict_size: int = 50,
                    min_corners_intrinsic: int = 9,
                    pooled: bool = False, min_points: int = 4,
                    extra_dist: bool = False,
                    fixed_intrinsics_from=None,
                    shared_intrinsics: bool = False,
                    shared_intrinsics_reference: "str | None" = None,
                    manually_verify: bool = False,
                    robust_filter: bool = True,
                    outlier_mad_k: float = 8.0,
                    max_frame_rms: float = 5.0,
                    geometric_filter: bool = False,
                    geometric_max_rel_error: float = 0.5,
                    geometric_min_abs_tol_mm: float = 5.0,
                    best_frames_top_k: "int | str" = "auto",
                    best_frames_min_cameras: int = 3,
                    best_frames_min_selected: int = 10,
                    log_fn=print) -> Path:
    """
    Programmatic entry point behind this module's CLI (main(), below) —
    discovers camN-prefixed calibration videos in *videos_dir*, runs
    aniposelib Charuco calibration, and writes out_dir/calibration.toml.
    Used both by the CLI and by cube_3d_dlc.py's Phase 1 auto-calibration.

    Default calibration strategy as of 2026-09-03 ("anchor-only" pipeline,
    validated on real data: F:\\TMZ7 CALIBRATION_0002, 4-camera rig):
    robust_filter (per-camera, per-frame corner-residual outlier rejection)
    + best_frames_top_k=100 (curate a small, well-distributed set of the
    highest-camera-coverage/highest-corner-count frames BEFORE building the
    initial extrinsics guess, rather than pooling every detected frame —
    see select_best_frames()'s own docstring for why pooling-then-filtering
    doesn't work reliably). This combination cut real reprojection error
    43.5px -> 31.85px and, more importantly, cut real triangulated-vs-known-
    board-geometry mm error in the well-conditioned (4-camera-visible) case
    from a 9.87mm median / 148,378mm max (catastrophic outliers) down to a
    1.58mm median / 27.5mm max — essentially eliminating the outlier tail
    for the geometry that matters most. A --best-frames sweep (50/100/150/
    200/300/500/800/1200) found 100 as the clear sweet spot; more frames
    dilutes the curation and regresses back toward the unfiltered baseline
    (300 frames on this dataset spiked to 49.5px, worse than no curation at
    all — a bad cluster entering the pool at that size). geometric_filter
    (a separate, cross-camera consistency check) is validated as a
    diagnostic but NOT adopted as a default: on the same real data it
    repeatedly crashed aniposelib's bundle adjustment regardless of
    threshold looseness, and a two-stage depth-augmentation experiment
    built on top of it (see refine_with_depth_augmentation) showed no
    measurable improvement over best-frames alone. See CLAUDE.md's
    "Calibration Accuracy — Anchor-Only Pipeline" section for the full
    investigation and numbers.

    Physical board dimensions (squares_x/y, square_length, marker_length)
    cannot be inferred from video alone — no absolute scale reference exists
    in an image — so they must always be supplied by the caller.

    shared_intrinsics: for identical-hardware rigs, estimate intrinsics from
    one camera only (calibrate_shared_intrinsics) and copy them onto every
    camera, instead of letting each solve independently and risk diverging
    to nonsensical per-camera focal lengths. Mutually exclusive with
    fixed_intrinsics_from (that sources intrinsics from an external file
    instead of a camera in this same run).
    """
    if shared_intrinsics and fixed_intrinsics_from is not None:
        raise ValueError(
            "shared_intrinsics and fixed_intrinsics_from are mutually "
            "exclusive — pick one intrinsics source.")

    cam_videos = discover_camera_videos(videos_dir)
    cam_names = sorted(cam_videos.keys())
    log_fn(f"[auto-calib] Cameras (in calibration order): {cam_names}")
    for cam in cam_names:
        log_fn(f"[auto-calib]   {cam}: {cam_videos[cam].name}")

    use_pooled = pooled or fixed_intrinsics_from is not None or shared_intrinsics
    board_cls = LowFloorCharucoBoard if use_pooled else CharucoBoard
    board_kwargs = dict(
        squaresX=squares_x, squaresY=squares_y,
        square_length=square_length, marker_length=marker_length,
        marker_bits=marker_bits, dict_size=dict_size,
        manually_verify=manually_verify,
    )
    # LowFloorCharucoBoard's whole point is supporting boards that never
    # clear normal per-frame corner floors (down to min_points=4, the CLI's
    # general-purpose default) — that's a real need for --pooled and
    # --fixed-intrinsics-from, where the physical board genuinely may be too
    # small/far to do better. shared_intrinsics targets the opposite case
    # (identical, well-covered cameras where independent per-camera solves
    # only diverge because ONE camera's own footage lacked depth variation,
    # not because the board is hard to see) — reusing min_points=4 there
    # every frame no longer means what it used to and just admits noisier
    # poses for no reason. Confirmed on real data (2026-08-27): this rig
    # detects 1400-2000+ frames per camera at >=9 corners, so there's no
    # shortage of well-conditioned frames to require instead.
    effective_min_points = max(min_points, 9) if shared_intrinsics else min_points
    if use_pooled:
        board_kwargs["min_corners_pose"] = effective_min_points
    board = board_cls(**board_kwargs)

    cgroup = make_camera_group(cam_names, extra_dist=extra_dist)
    videos = [[str(cam_videos[cam])] for cam in cam_names]

    if fixed_intrinsics_from is not None:
        log_fn(f"[auto-calib] Loading fixed intrinsics from {fixed_intrinsics_from}")
        apply_fixed_intrinsics(cgroup, Path(fixed_intrinsics_from))

    log_fn("[auto-calib] Detecting board corners and calibrating "
           "— this can take a while for long recordings...")
    if shared_intrinsics:
        all_rows, ref_cam = calibrate_shared_intrinsics(
            cgroup, board, videos, min_points=effective_min_points,
            reference_cam=shared_intrinsics_reference,
            extra_dist=extra_dist, log_fn=log_fn)
        error = calibrate_pooled(cgroup, board, videos, min_points=effective_min_points,
                                  fixed_intrinsics=True, all_rows=all_rows,
                                  verbose=False, robust_filter=robust_filter,
                                  outlier_mad_k=outlier_mad_k, max_frame_rms=max_frame_rms,
                                  geometric_filter=geometric_filter,
                                  geometric_max_rel_error=geometric_max_rel_error,
                                  geometric_min_abs_tol_mm=geometric_min_abs_tol_mm,
                                  best_frames_top_k=best_frames_top_k,
                                  best_frames_min_cameras=best_frames_min_cameras,
                                  best_frames_min_selected=best_frames_min_selected,
                                  log_fn=log_fn)
    elif use_pooled:
        log_fn(f"[auto-calib] Pooled mode: per-frame floor lowered to "
               f"{min_points} corners.")
        error = calibrate_pooled(cgroup, board, videos, min_points=min_points,
                                  fixed_intrinsics=fixed_intrinsics_from is not None,
                                  verbose=False, robust_filter=robust_filter,
                                  outlier_mad_k=outlier_mad_k, max_frame_rms=max_frame_rms,
                                  geometric_filter=geometric_filter,
                                  geometric_max_rel_error=geometric_max_rel_error,
                                  geometric_min_abs_tol_mm=geometric_min_abs_tol_mm,
                                  best_frames_top_k=best_frames_top_k,
                                  best_frames_min_cameras=best_frames_min_cameras,
                                  best_frames_min_selected=best_frames_min_selected,
                                  log_fn=log_fn)
    else:
        try:
            error, _ = cgroup.calibrate_videos(videos, board,
                                                min_corners_intrinsic=min_corners_intrinsic,
                                                verbose=False)
        except Exception as e:
            log_fn(f"[auto-calib] Standard calibration failed ({e}). "
                   f"Retrying with pooled calibration (lowering per-frame floor to {min_points} corners)...")
            pooled_board = LowFloorCharucoBoard(
                squaresX=squares_x, squaresY=squares_y,
                square_length=square_length, marker_length=marker_length,
                marker_bits=marker_bits, dict_size=dict_size,
                manually_verify=manually_verify,
                min_corners_pose=min_points
            )
            try:
                error = calibrate_pooled(cgroup, pooled_board, videos, min_points=min_points,
                                          fixed_intrinsics=fixed_intrinsics_from is not None,
                                          verbose=False, robust_filter=robust_filter,
                                          outlier_mad_k=outlier_mad_k, max_frame_rms=max_frame_rms,
                                          geometric_filter=geometric_filter,
                                          geometric_max_rel_error=geometric_max_rel_error,
                                          geometric_min_abs_tol_mm=geometric_min_abs_tol_mm,
                                          best_frames_top_k=best_frames_top_k,
                                          best_frames_min_cameras=best_frames_min_cameras,
                                          best_frames_min_selected=best_frames_min_selected,
                                          log_fn=log_fn)
            except Exception as pooled_err:
                raise RuntimeError(
                    f"Calibration failed (standard: {e}, pooled: {pooled_err}). "
                    "No camera video frames contained enough Charuco corners for calibration. "
                    "Please verify board parameters: squares_x, squares_y, square_length, "
                    "marker_length, marker_bits, dict_size, and ensure the board is visible."
                ) from pooled_err

    log_fn(f"[auto-calib] Average reprojection error: {error:.4f} px")
    if error > 1.0:
        log_fn("[auto-calib] WARNING: reprojection error is high (>1px) — "
               "consider re-recording the calibration board with more/better "
               "coverage, or set a manual Calibration folder as a backup.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_toml = out_dir / "calibration.toml"
    # Atomic: dump to a .tmp sibling and rename onto out_toml only once the
    # write fully succeeds -- callers (_resolve_calib_toml) skip-if-exists
    # on out_toml, so a crash mid-dump would otherwise leave a truncated
    # TOML that's silently treated as a valid cached calibration forever.
    tmp_toml = out_toml.with_name(out_toml.name + ".tmp")
    cgroup.dump(str(tmp_toml))
    tmp_toml.replace(out_toml)
    log_fn(f"[auto-calib] Wrote {out_toml}")
    return out_toml


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", required=True, type=Path,
                     help="Folder containing one 'camN_...' calibration video per camera "
                          "(e.g. Step 0's _calibration/{session_id}/ output).")
    ap.add_argument("--out-dir", required=True, type=Path,
                     help="Where to write calibration.toml — point CUBE's 3D DLC tab "
                          "'Calibration folder' setting at this same directory.")
    ap.add_argument("--squares-x", required=True, type=int,
                     help="Number of Charuco squares across the board's WIDTH.")
    ap.add_argument("--squares-y", required=True, type=int,
                     help="Number of Charuco squares across the board's HEIGHT.")
    ap.add_argument("--square-length", required=True, type=float,
                     help="Physical side length of one checkerboard square, in mm "
                          "(sets the real-world scale of the whole calibration — "
                          "measure this on the actual printed board with calipers).")
    ap.add_argument("--marker-length", required=True, type=float,
                     help="Physical side length of one ArUco marker inside the squares, "
                          "in mm (must be smaller than --square-length).")
    ap.add_argument("--marker-bits", type=int, default=4,
                     help="ArUco marker bit size (default 4, i.e. DICT_4X4_*).")
    ap.add_argument("--dict-size", type=int, default=50,
                     help="Number of markers in the ArUco dictionary (default 50).")
    ap.add_argument("--manually-verify", action="store_true",
                     help="Open an interactive OpenCV window per auto-detected frame "
                          "so you can accept ('a') or reject ('d') it before it's used "
                          "in the solve. Requires a real display/terminal — cannot run "
                          "headless. Use together with --min-corners-intrinsic when the "
                          "board only ever partially fills the frame (few corners/frame).")
    ap.add_argument("--min-corners-intrinsic", type=int, default=9,
                     help="Minimum Charuco corners a frame must have to be used for "
                          "per-camera intrinsics estimation (aniposelib default: 9). "
                          "Lower this (e.g. 4) if the board never fills the whole frame "
                          "and no frame reaches the default threshold — calibration "
                          "quality will be lower with fewer corners per frame.")
    ap.add_argument("--pooled", action="store_true",
                     help="Use calibrate_pooled() instead of aniposelib's standard "
                          "calibrate_videos(). Bypasses aniposelib's hardcoded per-frame "
                          "floors (5 corners for intrinsics, 7 for per-frame pose, 8 for "
                          "extrinsics linking) and pools ALL detected frames — however few "
                          "corners each has — into one joint bundle adjustment. Only use "
                          "this when you've confirmed (not assumed) the physical board "
                          "never clears those floors in a single frame; relies on having "
                          "hundreds+ of frames per camera so the aggregate solve can "
                          "average out each frame's individually weak/noisy pose.")
    ap.add_argument("--min-points", type=int, default=4,
                     help="With --pooled: minimum corners a frame must have to contribute "
                          "at all (default 4, the mathematical floor for solvePnP). Used "
                          "consistently for intrinsics, per-frame pose, and extrinsics "
                          "linking in the pooled path.")
    ap.add_argument("--fixed-intrinsics-from", type=Path, default=None,
                     help="Path to a known-good calibration.toml (matched by camera name) "
                          "whose per-camera matrix/distortions/size are reused as-is for "
                          "this session, instead of re-estimating intrinsics from this "
                          "session's own (possibly weak) data. Only valid if the SAME "
                          "physical cameras/lenses were used, with unchanged zoom/focus — "
                          "intrinsics belong to the lens, not the rig's position. Only "
                          "extrinsics (rotation/translation) are solved from this session's "
                          "data. Implies --pooled.")
    ap.add_argument("--shared-intrinsics", action="store_true",
                     help="For rigs where every camera is the SAME physical hardware: "
                          "estimate intrinsics from just one camera (whichever has the "
                          "most usable calibration frames, or --shared-intrinsics-reference) "
                          "via a real cv2.calibrateCamera() solve, and copy that matrix/"
                          "distortion onto every camera instead of letting each solve "
                          "independently. Use this when per-camera calibration produces "
                          "wildly different focal lengths across cameras you know are "
                          "identical — that's usually the calibration board not covering "
                          "enough depth/tilt variation for every camera individually, not "
                          "a real hardware difference. Only extrinsics (rotation/"
                          "translation) are then solved per-camera. Mutually exclusive "
                          "with --fixed-intrinsics-from. Implies --pooled.")
    ap.add_argument("--shared-intrinsics-reference", type=str, default=None,
                     help="With --shared-intrinsics: camN label of the camera to source "
                          "intrinsics from (e.g. 'cam2'). Default: whichever camera has "
                          "the most usable (>= --min-points corners) calibration frames.")
    ap.add_argument("--extra-dist", action="store_true",
                     help="Let bundle adjustment refine a second distortion coefficient "
                          "(k2), not just k1 (aniposelib's default). If reprojection error "
                          "stays at a stable, high floor regardless of how much data is "
                          "used — the signature of the camera model itself being unable to "
                          "represent the lens, not a data-scarcity problem — this is worth "
                          "trying.")
    ap.add_argument("--no-robust-filter", action="store_true",
                     help="Disable the per-frame reprojection-residual outlier filter "
                          "(on by default with --pooled/--shared-intrinsics/"
                          "--fixed-intrinsics-from). The filter drops individual Charuco "
                          "corners (and, if too few survive, whole frames) whose "
                          "reprojection error against that frame's own solved board pose "
                          "is a statistical outlier (median + --outlier-mad-k * MAD) — "
                          "catches motion-blur-corrupted corners (e.g. from a board being "
                          "actively moved/rotated) before they can corrupt the joint "
                          "cross-camera bundle adjustment. Only disable this to reproduce "
                          "old (pre-filter) calibration output exactly.")
    ap.add_argument("--outlier-mad-k", type=float, default=8.0,
                     help="Robust-filter corner rejection threshold, in MADs (median "
                          "absolute deviations) above the frame's own median reprojection "
                          "residual (default 8.0 — generous, only rejects clearly-corrupted "
                          "corners). Lower it (e.g. 4-5) for stricter rejection if outliers "
                          "persist; raise it if legitimate high-angle corners are being "
                          "dropped.")
    ap.add_argument("--max-frame-rms", type=float, default=5.0,
                     help="Robust-filter whole-frame rejection threshold, in pixels — a "
                          "frame is dropped if its re-solved pose's own RMS reprojection "
                          "error (after corner-level outlier removal) still exceeds this "
                          "(default 5.0px).")
    ap.add_argument("--geometric-filter", action="store_true",
                     help="Enable the cross-camera geometric-consistency outlier filter "
                          "(off by default — see --best-frames, the validated default "
                          "strategy as of 2026-09-03). Unlike robust_filter (which only "
                          "checks a corner against the rest of ITS OWN camera's frame), "
                          "this triangulates each multi-camera detection with a rough "
                          "initial extrinsics guess and rejects the WHOLE frame (all "
                          "cameras) if the reconstructed shape disagrees with the board's "
                          "known geometry — catches planar-pose 'flip' ambiguity, where "
                          "every corner in a frame looks perfectly self-consistent to its "
                          "own camera but that camera's whole pose is wrong relative to "
                          "the others. Real diagnostic value (flagged 39%% of frames on a "
                          "real recording) but NOT adopted as a default: it repeatedly "
                          "crashed aniposelib's bundle adjustment on real data regardless "
                          "of threshold looseness, and combined with --best-frames showed "
                          "no measurable accuracy improvement in testing. Available for "
                          "diagnostic use or further experimentation, not general use.")
    ap.add_argument("--geometric-max-rel-error", type=float, default=0.5,
                     help="Geometric-filter rejection threshold: a frame's reconstructed "
                          "pairwise distances may deviate from the board's known geometry "
                          "by at most this fraction (default 0.5 = 50%%) before the frame "
                          "is flagged as an inlier pair. Combined with --geometric-min-abs-"
                          "tol-mm via OR (either condition passing keeps the pair).")
    ap.add_argument("--geometric-min-abs-tol-mm", type=float, default=5.0,
                     help="Geometric-filter absolute tolerance in mm — a pairwise distance "
                          "comparison within this many mm of the true board geometry is "
                          "always treated as an inlier regardless of relative error "
                          "(guards against tiny known-distances producing a huge spurious "
                          "relative error). Default 5.0mm.")
    ap.add_argument("--best-frames", type=str, default="auto",
                     help="Instead of pooling every detected frame, select only the best "
                          "N multi-camera frames (ranked by camera-coverage then corner "
                          "count) BEFORE building the initial extrinsics guess or running "
                          "bundle adjustment, so the guess and the optimization data are "
                          "always mutually consistent. Sound because a calibration "
                          "recording's extrinsics are fixed for the whole capture (the "
                          "rig doesn't move) — a modest number of GOOD, well-distributed "
                          "frames suffices; you don't need every frame. Default 'auto': "
                          "automatically picks N by testing a small, bounded grid of "
                          "candidates (50/75/100/150/250/400) and keeping whichever gives "
                          "the best HELD-OUT real-mm accuracy (not just the fit set's own "
                          "reprojection error, which is a misleading selection criterion — "
                          "see auto_select_best_frames_top_k()'s docstring). Detection is "
                          "done once regardless — 'auto' only adds a few seconds per "
                          "candidate on top of that, not a repeated multi-minute scan. "
                          "Validated on real data (2026-09-03, F:\\TMZ7): a hand-picked "
                          "N=100 already cut reprojection error 43.5px->31.85px and cut "
                          "real triangulated-vs-known-geometry mm error (4-camera-visible "
                          "case) from a 9.87mm median/148,378mm max down to 1.58mm "
                          "median/27.5mm max — 'auto' then found N=400 on the SAME "
                          "recording beats that further (0.28mm median, held-out), since "
                          "it optimizes real accuracy directly instead of a fixed guess. "
                          "Pass a specific integer to pin N manually (e.g. 100), or 0 to "
                          "pool every detected frame instead (old pre-fix behavior).")
    ap.add_argument("--best-frames-min-cameras", type=int, default=3,
                     help="With --best-frames: a frame must have been seen by at least "
                          "this many cameras to be eligible for selection at all (default "
                          "3 — a frame only 1-2 cameras saw contributes little to a joint "
                          "4-camera extrinsics solve). Silently clamped to the rig's own "
                          "camera count, so this doesn't break a future <=3-camera rig.")
    args = ap.parse_args()

    if not args.videos_dir.is_dir():
        sys.exit(f"--videos-dir not found: {args.videos_dir}")

    if args.manually_verify:
        print("Manual verification enabled: an OpenCV window will pop up for "
              "each auto-detected frame across all cameras. Press 'a' to accept, "
              "'d' to reject. This requires a real display — do not run headless.")

    out_toml = auto_calibrate(
        args.videos_dir, args.out_dir,
        squares_x=args.squares_x, squares_y=args.squares_y,
        square_length=args.square_length, marker_length=args.marker_length,
        marker_bits=args.marker_bits, dict_size=args.dict_size,
        min_corners_intrinsic=args.min_corners_intrinsic,
        pooled=args.pooled, min_points=args.min_points,
        extra_dist=args.extra_dist,
        fixed_intrinsics_from=args.fixed_intrinsics_from,
        shared_intrinsics=args.shared_intrinsics,
        shared_intrinsics_reference=args.shared_intrinsics_reference,
        manually_verify=args.manually_verify,
        robust_filter=not args.no_robust_filter,
        outlier_mad_k=args.outlier_mad_k,
        max_frame_rms=args.max_frame_rms,
        geometric_filter=args.geometric_filter,
        geometric_max_rel_error=args.geometric_max_rel_error,
        geometric_min_abs_tol_mm=args.geometric_min_abs_tol_mm,
        best_frames_top_k=(args.best_frames if args.best_frames.strip().lower() == "auto"
                           else int(args.best_frames)),
        best_frames_min_cameras=args.best_frames_min_cameras,
        log_fn=print)

    print(f"Set CUBE's 3D DLC tab 'Calibration folder' to: {out_toml.parent}")


if __name__ == "__main__":
    main()
