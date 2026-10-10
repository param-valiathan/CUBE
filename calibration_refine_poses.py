"""
Pose-based calibration refinement for CUBE's 3D pipeline.

Refines an existing board-derived `calibration.toml` using real DLC keypoints
(already labeled and matched across cameras) as correspondence data, instead of
re-running board calibration. Never overwrites `calibration.toml` — writes
`calibration_refined.toml` alongside it, and only when refinement measurably
lowers `CameraGroup.average_error(...)` on the pooled correspondence points.

Two stages, both operating on extrinsics only (intrinsics are left untouched —
board calibration already resolves focal length/distortion reasonably well;
the ambiguity this module targets is specifically extrinsic pose):

  Stage A — unconstrained bundle adjustment (`cgroup.bundle_adjust_iter`,
            already installed in aniposelib, outlier-robust).
  Stage B — orthogonality-prior joint refinement (`_refine_extrinsics_with_prior`),
            a small locally-implemented `scipy.optimize.least_squares` wrapper,
            architecturally identical in spirit to `_triangulate_weighted_dlt`/LOST
            already shipped in `cube_core.py`: it reuses aniposelib's `Camera`/
            `CameraGroup` objects and methods without modifying the installed
            aniposelib package (which has no concept of a geometric prior).
            Only runs when `orthogonal=True`.

See D:\\3D DLC\\POSE_CALIBRATION_REFINEMENT.md for the full design and literature
review this module implements.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Correspondence collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_pose_correspondences(cam_h5s: dict, cam_labels: list,
                                  min_likelihood: float = 0.8,
                                  max_points: int = 2000,
                                  min_points: int = 20,
                                  seed: int = 0,
                                  log_fn=print):
    """
    Pool high-confidence DLC keypoints common to all cameras, across frames,
    bodyparts, and sessions, into a single correspondence point set.

    Parameters
    ----------
    cam_h5s : dict[str, list[Path]] — camera label -> list of per-camera DLC H5
              paths, one per session, aligned by session index across cameras
              (i.e. cam_h5s[cam_a][i] and cam_h5s[cam_b][i] are the same session).
    cam_labels : preferred camera order (only cameras with a non-empty list in
              cam_h5s are used).
    min_likelihood : DLC likelihood threshold for a keypoint to be pooled.
    max_points : subsample cap for tractable bundle-adjustment runtime.
    min_points : minimum qualifying points required, else returns None.
    seed : RNG seed for the subsampling.

    Returns
    -------
    (p2ds, cam_order) where p2ds is (N_cams, N_points, 2) and cam_order is the
    list of camera labels matching the first axis, in the order actually used —
    or None (logged) if fewer than 2 cameras or too few qualifying points.
    """
    present = [c for c in cam_labels if cam_h5s.get(c)]
    if len(present) < 2:
        log_fn(f"[calib-refine] Need >=2 cameras with DLC H5s to pool "
               f"correspondences; found {len(present)}")
        return None

    n_sessions = min(len(cam_h5s[c]) for c in present)
    if n_sessions == 0:
        log_fn("[calib-refine] No sessions with H5s for the required cameras")
        return None

    all_pts = []  # list of (n_cams, k, 2) chunks, one per (session, bodypart)
    for si in range(n_sessions):
        dfs = []
        bps_per_cam = []
        ok = True
        for c in present:
            h5_path = cam_h5s[c][si] if si < len(cam_h5s[c]) else None
            if h5_path is None or not Path(h5_path).exists():
                ok = False
                break
            try:
                df = pd.read_hdf(str(h5_path))
                scorer = df.columns.get_level_values(0)[0]
                bps = df[scorer].columns.get_level_values(0).unique().tolist()
            except Exception as e:
                log_fn(f"[calib-refine] Failed to read {h5_path}: {e}")
                ok = False
                break
            dfs.append((scorer, df))
            bps_per_cam.append(bps)
        if not ok:
            continue

        common_bps = list(bps_per_cam[0])
        for bps_k in bps_per_cam[1:]:
            common_bps = [b for b in common_bps if b in bps_k]
        if not common_bps:
            continue

        n_frames = min(len(df) for _, df in dfs)
        for bp in common_bps:
            xs, ys, lls = [], [], []
            for (scorer, df) in dfs:
                try:
                    x = df[scorer, bp, "x"].values[:n_frames].astype(float)
                    y = df[scorer, bp, "y"].values[:n_frames].astype(float)
                    ll = df[scorer, bp, "likelihood"].values[:n_frames].astype(float)
                except Exception:
                    x = y = ll = np.full(n_frames, np.nan)
                xs.append(x); ys.append(y); lls.append(ll)
            xs = np.stack(xs)   # (n_cams, n_frames)
            ys = np.stack(ys)
            lls = np.stack(lls)
            good = np.all(lls >= min_likelihood, axis=0)
            if not np.any(good):
                continue
            pts = np.stack([xs[:, good], ys[:, good]], axis=-1)  # (n_cams, k, 2)
            all_pts.append(pts)

    if not all_pts:
        log_fn(f"[calib-refine] No qualifying correspondence points found "
               f"(min_likelihood={min_likelihood})")
        return None

    p2ds = np.concatenate(all_pts, axis=1)  # (n_cams, N_points, 2)
    n_points = p2ds.shape[1]
    if n_points < min_points:
        log_fn(f"[calib-refine] Only {n_points} qualifying correspondence "
               f"points (< {min_points} minimum) — skipping refinement")
        return None

    if n_points > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_points, size=max_points, replace=False)
        p2ds = p2ds[:, idx, :]

    log_fn(f"[calib-refine] Pooled {p2ds.shape[1]} correspondence points "
           f"across {len(present)} cameras ({n_sessions} session(s) scanned)")
    return p2ds, present


# ─────────────────────────────────────────────────────────────────────────────
#  Orthogonality-prior helpers
# ─────────────────────────────────────────────────────────────────────────────

def _consecutive_pairs(cam_labels: list) -> list:
    return [(cam_labels[i], cam_labels[i + 1]) for i in range(len(cam_labels) - 1)]


def _cam_angle_deg(cgroup, name_a: str, name_b: str) -> float:
    """Geodesic relative-rotation angle (degrees) between two named cameras."""
    names = cgroup.get_names()
    ca = cgroup.cameras[names.index(name_a)]
    cb = cgroup.cameras[names.index(name_b)]
    Ra, _ = cv2.Rodrigues(ca.get_rotation())
    Rb, _ = cv2.Rodrigues(cb.get_rotation())
    return _rel_angle_deg(Ra, Rb)


def _rel_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R_rel = Rb @ Ra.T
    c = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _camera_centers(cgroup) -> dict:
    """World-space camera centers, C = -R^T @ t, keyed by camera name."""
    centers = {}
    for cam in cgroup.cameras:
        R, _ = cv2.Rodrigues(np.asarray(cam.get_rotation(), dtype=np.float64))
        t = np.asarray(cam.get_translation(), dtype=np.float64).reshape(3)
        centers[cam.get_name()] = -R.T @ t
    return centers


def _camera_spread(centers: dict) -> float:
    """Mean pairwise distance (mm) between camera centers -- a single scalar
    summarizing how spread out the rig's cameras are in space. A real
    multi-camera rig has cameras genuinely apart from one another; a bundle
    adjustment that degenerates toward every camera occupying nearly the same
    point (see the geometry-collapse guardrail below) collapses this toward
    zero regardless of what `average_error` on the (possibly impoverished)
    correspondence set it was fit against reports."""
    names = list(centers.keys())
    if len(names) < 2:
        return 0.0
    dists = [np.linalg.norm(centers[a] - centers[b])
             for i, a in enumerate(names) for b in names[i + 1:]]
    return float(np.mean(dists))


def _refine_extrinsics_with_prior(cgroup, p2ds: np.ndarray, cam_labels: list,
                                   expected_angle_deg: float, tolerance_deg: float,
                                   prior_strength: float, ftol: float = 1e-6,
                                   max_nfev: int = 200, log_fn=print):
    """
    Stage B: joint reprojection + orthogonality-prior refinement.

    Mirrors aniposelib's own reprojection residual (`CameraGroup.reprojection_error`
    via `Camera.project`, the same math as its internal `_error_fun_bundle`) but is
    implemented locally because the prior residual — a geodesic-angle penalty
    pulling each auto-derived consecutive camera pair's relative rotation toward
    `expected_angle_deg` — has no equivalent in the installed aniposelib package.
    Jointly minimized via `scipy.optimize.least_squares`, not a post-hoc blend.

    Cameras are parameterized by extrinsics only (`Camera.get_params(only_extrinsics=True)`
    -> 6 numbers: rvec + tvec), matching Stage A's `only_extrinsics=True` bundle
    adjustment so both stages agree on what is/isn't being optimized.

    Guardrail: a pair is only given a prior residual if its *current* (Stage-A)
    angle already falls within `tolerance_deg` of `expected_angle_deg` — otherwise
    it is skipped (logged) as likely signalling a real rig issue rather than a
    fittable calibration error, and returned unmodified by this stage.

    Returns the same `cgroup` object, refined in place (also returned for
    convenience/chaining).
    """
    from scipy.optimize import least_squares

    names = cgroup.get_names()
    name_idx = {n: i for i, n in enumerate(names)}
    pairs = _consecutive_pairs(cam_labels)

    active_pairs = []
    for (a, b) in pairs:
        if a not in name_idx or b not in name_idx:
            continue
        angle = _cam_angle_deg(cgroup, a, b)
        if abs(angle - expected_angle_deg) > tolerance_deg:
            log_fn(f"[calib-refine] Pair ({a}, {b}): Stage-A angle {angle:.1f} deg "
                   f"deviates from expected {expected_angle_deg:.1f} deg by more "
                   f"than the {tolerance_deg:.1f} deg tolerance — skipping the "
                   f"orthogonality prior for this pair")
            continue
        active_pairs.append((name_idx[a], name_idx[b]))

    if not active_pairs:
        log_fn("[calib-refine] No camera pairs pass the tolerance guardrail — "
               "Stage B prior skipped entirely (cgroup left as Stage A produced it)")
        return cgroup

    w = float(np.sqrt(max(prior_strength, 0.0)))
    expected_rad = np.radians(expected_angle_deg)
    x0 = np.concatenate([cam.get_params(only_extrinsics=True) for cam in cgroup.cameras])

    def residual(x):
        for i, cam in enumerate(cgroup.cameras):
            cam.set_params(x[i * 6:(i + 1) * 6], only_extrinsics=True)
        p3ds = cgroup.triangulate(p2ds)
        errs = cgroup.reprojection_error(p3ds, p2ds, mean=False)  # (n_cams, n_pts, 2)
        reproj_res = np.nan_to_num(errs, nan=0.0).ravel()

        prior_res = np.empty(len(active_pairs), dtype=np.float64)
        for k, (ia, ib) in enumerate(active_pairs):
            Ra, _ = cv2.Rodrigues(x[ia * 6:ia * 6 + 3])
            Rb, _ = cv2.Rodrigues(x[ib * 6:ib * 6 + 3])
            theta = np.radians(_rel_angle_deg(Ra, Rb))
            prior_res[k] = w * (theta - expected_rad)

        return np.concatenate([reproj_res, prior_res])

    result = least_squares(residual, x0, ftol=ftol, max_nfev=max_nfev)

    for i, cam in enumerate(cgroup.cameras):
        cam.set_params(result.x[i * 6:(i + 1) * 6], only_extrinsics=True)

    return cgroup


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def refine_calibration_with_poses(calib_toml, cam_h5s: dict, cam_labels: list,
                                   orthogonal: bool, expected_angle_deg: float,
                                   prior_strength: float, tolerance_deg: float,
                                   min_likelihood: float, log_fn=print):
    """
    Orchestrates pose-based calibration refinement.

    Loads `calib_toml` -> pools DLC-keypoint correspondences -> Stage A
    (unconstrained bundle adjustment on extrinsics) -> Stage B (orthogonality
    prior, only if `orthogonal=True`) -> compares `average_error` before/after
    -> writes `{calib_toml.parent}/calibration_refined.toml` only if improved.
    Never modifies or overwrites `calib_toml` itself.

    Returns (refined_path_or_None, report_dict). `report_dict` always contains
    enough detail for logging/QC even when no file is written.
    """
    from aniposelib.cameras import CameraGroup

    calib_toml = Path(calib_toml)

    result = collect_pose_correspondences(
        cam_h5s, cam_labels, min_likelihood=min_likelihood, log_fn=log_fn)
    if result is None:
        return None, {"status": "insufficient_data"}
    p2ds, present = result

    full_cgroup = CameraGroup.load(str(calib_toml))
    try:
        cgroup_orig = full_cgroup.subset_cameras_names(present)
        cgroup = full_cgroup.subset_cameras_names(present)
    except IndexError as e:
        log_fn(f"[calib-refine] {e}")
        return None, {"status": "camera_name_mismatch"}

    error_before = float(cgroup_orig.average_error(p2ds, median=True))
    log_fn(f"[calib-refine] average_error before refinement: {error_before:.3f}px "
           f"({p2ds.shape[1]} points, cameras={present})")

    angles_before = {f"{a}-{b}": _cam_angle_deg(cgroup, a, b)
                      for a, b in _consecutive_pairs(present)}

    # Stage A — unconstrained bundle adjustment (extrinsics only)
    cgroup.bundle_adjust_iter(p2ds, only_extrinsics=True, verbose=False)
    # bundle_adjust_iter()'s own return value is computed on an internally
    # resampled subset (aniposelib resamples to n_samp_full=1000 points
    # before its final average_error call), not on the full `p2ds` array —
    # with ~2000 real correspondence points that's a different, randomly
    # resampled point set than error_before/error_after below. Recomputing
    # here on the exact same `p2ds` and method keeps all three error_* values
    # (before/stage_a/after) directly comparable, which matters once
    # error_stage_a is used as a decision threshold (see Stage B gate below),
    # not just a log line.
    error_stage_a = float(cgroup.average_error(p2ds, median=True))
    log_fn(f"[calib-refine] Stage A (unconstrained bundle adjustment) "
           f"average_error: {error_stage_a:.3f}px")

    angles_stage_a = {f"{a}-{b}": _cam_angle_deg(cgroup, a, b)
                       for a, b in _consecutive_pairs(present)}

    # Geometry-collapse guardrail: an unconstrained bundle adjustment over a
    # correspondence set that's too geometrically impoverished (e.g. pooled
    # almost entirely from head/torso keypoints clustered in a narrow depth
    # band, because most limb-joint bodyparts are chronically low-confidence
    # on this rig -- see D:\3D DLC\CLAUDE.md's hind-limb/thai-knee entries)
    # can degenerate toward every camera center converging on nearly the same
    # point in space. `average_error` on that SAME impoverished point set can
    # still look numerically fine for such a solution (it fits those specific
    # points), while the resulting calibration is physically nonsensical and
    # triangulates every OTHER point to a single frozen location -- confirmed
    # on real data (F:\TMZ7, 2026-09-09): all 4 cameras collapsed to within
    # 0.1mm of each other's distance-from-origin on two separate real runs
    # (183.6mm and 195.7mm respectively), and every triangulated bodypart's
    # XYZ was frozen bit-for-bit identical across every frame as a result.
    # Reject Stage A's result outright (never proceed to Stage B or the
    # error-based accept gate) if the camera rig's own spatial spread
    # collapsed relative to the pre-refinement calibration -- this is checked
    # independently of average_error precisely because that metric cannot
    # detect this failure mode on the data that produces it.
    centers_orig = _camera_centers(cgroup_orig)
    centers_stage_a = _camera_centers(cgroup)
    spread_orig = _camera_spread(centers_orig)
    spread_stage_a = _camera_spread(centers_stage_a)
    collapse_ratio = (spread_stage_a / spread_orig) if spread_orig > 1e-9 else 1.0
    log_fn(f"[calib-refine] Camera-rig spread (mean pairwise camera distance): "
           f"{spread_orig:.1f}mm before -> {spread_stage_a:.1f}mm after Stage A "
           f"(ratio={collapse_ratio:.3f})")
    if collapse_ratio < 0.3:
        log_fn(f"[calib-refine] REJECTED: Stage A collapsed the camera rig's "
               f"spatial spread to {collapse_ratio*100:.1f}% of the original "
               f"({spread_orig:.1f}mm -> {spread_stage_a:.1f}mm) -- a degenerate "
               f"solution regardless of its average_error. calibration_refined.toml "
               f"NOT written; the original calibration is left untouched.")
        return None, {
            "status": "geometry_collapsed",
            "error_before": error_before,
            "error_stage_a": error_stage_a,
            "spread_before_mm": spread_orig,
            "spread_stage_a_mm": spread_stage_a,
            "collapse_ratio": collapse_ratio,
        }

    # Stage B — orthogonality-prior joint refinement (opt-in), gated against
    # Stage A's own result: Stage B is kept only if it doesn't regress vs.
    # Stage A alone, otherwise reverted, so enabling it can only help or be a
    # no-op relative to Stage A — never make the final calibration worse than
    # Stage A already achieved (the previous gate only compared against the
    # pre-Stage-A original, which let a Stage-B regression slip through as
    # long as it still beat doing no refinement at all).
    stage_b_reverted = False
    if orthogonal:
        stage_a_params = {cam.get_name(): cam.get_params(only_extrinsics=True)
                           for cam in cgroup.cameras}
        cgroup = _refine_extrinsics_with_prior(
            cgroup, p2ds, present, expected_angle_deg, tolerance_deg,
            prior_strength, log_fn=log_fn)
        error_stage_b = float(cgroup.average_error(p2ds, median=True))
        if error_stage_b > error_stage_a:
            log_fn(f"[calib-refine] Stage B average_error ({error_stage_b:.3f}px) "
                   f"regressed vs. Stage A alone ({error_stage_a:.3f}px) — "
                   f"reverting to the Stage A result")
            for cam in cgroup.cameras:
                cam.set_params(stage_a_params[cam.get_name()], only_extrinsics=True)
            stage_b_reverted = True

    error_after = float(cgroup.average_error(p2ds, median=True))
    angles_after = {f"{a}-{b}": _cam_angle_deg(cgroup, a, b)
                     for a, b in _consecutive_pairs(present)}
    log_fn(f"[calib-refine] average_error after refinement: {error_after:.3f}px "
           f"(before: {error_before:.3f}px)")

    report = {
        "cameras": present,
        "n_points": int(p2ds.shape[1]),
        "orthogonal": bool(orthogonal),
        "stage_b_reverted": stage_b_reverted,
        "error_before": error_before,
        "error_stage_a": error_stage_a,
        "error_after": error_after,
        "angles_before_deg": angles_before,
        "angles_stage_a_deg": angles_stage_a,
        "angles_after_deg": angles_after,
    }

    if not (error_after < error_before):
        log_fn("[calib-refine] Refined average_error did not improve on the "
               "original — calibration_refined.toml NOT written")
        report["status"] = "not_improved"
        return None, report

    for cam in cgroup.cameras:
        idx = full_cgroup.get_names().index(cam.get_name())
        full_cgroup.cameras[idx].set_params(
            cam.get_params(only_extrinsics=True), only_extrinsics=True)

    refined_path = calib_toml.parent / "calibration_refined.toml"
    # Atomic: dump to a .tmp sibling and rename onto refined_path only once
    # the write fully succeeds -- callers cache/skip-if-exists on this path
    # (dlc_3d_calib_refined_path), so a crash mid-dump would otherwise leave
    # a truncated TOML that's silently treated as a valid refined
    # calibration forever.
    tmp_refined_path = refined_path.with_name(refined_path.name + ".tmp")
    full_cgroup.dump(str(tmp_refined_path))
    tmp_refined_path.replace(refined_path)
    log_fn(f"[calib-refine] Wrote {refined_path}")
    report["status"] = "written"
    return refined_path, report
