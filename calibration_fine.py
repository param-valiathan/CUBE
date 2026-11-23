"""FINE calibration for CUBE Step 1b (and the CUBE Capture "Run calibration" button): floor alignment, rigid-board bundle adjustment with PER-CAMERA lens,
held-out validation, consistency checks and accuracy figures in mm.   (Param Valiathan, laboratory of Patrik Ernfors, Karolinska Institutet)

THE STANDARD ANALYSIS since 2026-11-23 - calibrate_cameras.auto_calibrate() runs it by default (rigid_refine=True):

  finalize(out_dir, solver_toml, videos, floor_videos, ...)
    1. floor take (board flat and still, optional): z = 0 on the floor plane, camera frame = the floor frame (x along the board rows, z up);
    2. consistency check of the floor take (leave-one-camera-out: a camera that MOVED since the calibration stands out; normal 1-3 px);
    3. rigid-board bundle adjustment (calibration_rigid_ba.py): per camera pose + f + k1 + k2 + principal point, ONE rigid board pose per frame; judged on HELD-OUT moments
       (fit on even, test on odd) and the floor take, adopted only when not worse (median / p95 length error within +2 %, worst floor-take camera not worse);
    4. fitted on all moments, the floor re-aligned for the refined cameras;
    5. accuracy FIGURES IN MM (calibration_accuracy_<stamp>.png/.json) in every report folder;
    6. out_dir\\calibration.toml is written LAST (atomic) together with analysis.json (ANALYSIS_VERSION + everything above); the plain solver result is only ever
       out_dir\\calibration_solver_only.toml, so an interrupted run can never leave an unrefined calibration under the name Step 1b trusts.
Everything is tolerant: a failing step keeps the previous result and says why (analysis.json "notes").  All detection is single-threaded and light (cpu-friendly).
"""
import json
import os
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ANALYSIS_VERSION = "rigid-ba-2026-11-23"
GOOD_MM, OK_MM = 0.5, 1.0
PRIORS = dict(sig_f=8.0, sig_k=0.01, sig_c=4.0)


# ---------------------------------------------------------------------------------------------------------------- detector / board
def make_detector(squares_x=6, squares_y=5, square_mm=12.22, marker_mm=9.165, bits=4, dict_size=50):
    """Charuco detector with the relaxed small-marker profile (same as calibrate_cameras.apply_small_marker_detector_params; marker-corner refinement mode was tested
    on 2026-11-23: no measurable difference).  Board sizes in mm; object points are returned in mm by board_corners_mm()."""
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, f"DICT_{bits}X{bits}_{dict_size}"))
    board = cv2.aruco.CharucoBoard((int(squares_x), int(squares_y)), float(square_mm) / 1000, float(marker_mm) / 1000, d)
    dp = cv2.aruco.DetectorParameters()
    dp.adaptiveThreshWinSizeMin, dp.adaptiveThreshWinSizeMax, dp.adaptiveThreshWinSizeStep = 3, 53, 4
    dp.adaptiveThreshConstant, dp.minMarkerPerimeterRate, dp.polygonalApproxAccuracyRate, dp.errorCorrectionRate = 7, 0.01, 0.05, 0.8
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dp.perspectiveRemovePixelPerCell, dp.perspectiveRemoveIgnoredMarginPerCell, dp.minOtsuStdDev = 8, 0.2, 2.0
    return _Detector(cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), dp), int(squares_x) - 1, (int(squares_x) - 1) * (int(squares_y) - 1))


class _Detector:
    """the OpenCV CharucoDetector (attributes cannot be added to it) + the board size; every other attribute is the detector's own"""

    def __init__(self, det, n_cols, n_corners):
        self._det, self.n_cols, self.n_corners = det, n_cols, n_corners

    def __getattr__(self, name):
        return getattr(self._det, name)


def board_corners_mm(det):
    return det.getBoard().getChessboardCorners().astype(np.float64) * 1000.0


def _cams(cg):
    import cv2
    out = []
    for cam in cg.cameras:
        R = cv2.Rodrigues(np.asarray(cam.get_rotation(), dtype=np.float64))[0]
        out.append(dict(K=np.asarray(cam.get_camera_matrix(), dtype=np.float64), D=np.asarray(cam.get_distortions(), dtype=np.float64).ravel(), R=R,
                        t=np.asarray(cam.get_translation(), dtype=np.float64).ravel()))
    return out


# ---------------------------------------------------------------------------------------------------------------- detections over a take
def detect_moments(names, videos, step_s, det, log=print):
    """Board detections at common wall-clock moments every step_s seconds.  Frames of the six free-running cameras are matched by the CSV time stamps next to the videos
    (Step 0 copies them); without CSVs frame index / fps is used.  Returns (grid, {cam: [(ids, px) or None per moment]})."""
    import cv2
    from calibrate_cameras import read_sidecar_timestamps
    ts, caps = {}, {}
    for n in names:
        cap = cv2.VideoCapture(str(videos[n]))
        nf, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30.0
        t = read_sidecar_timestamps(Path(videos[n]))
        ts[n] = t if t is not None and len(t) >= 30 else np.arange(nf) / fps
        caps[n] = cap
    grid = np.arange(max(v[0] for v in ts.values()) + 0.2, min(v[-1] for v in ts.values()) - 0.2, step_s)
    dets = {}
    for n in names:
        t, cap = ts[n], caps[n]
        i = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        i = np.where(np.abs(t[i - 1] - grid) < np.abs(t[i] - grid), i - 1, i)
        found, cur = {}, -1
        for fi in sorted(set(int(x) for x in i)):
            while cur < fi - 1:
                if not cap.grab():
                    break
                cur += 1
            ok, fr = cap.read()
            cur += 1
            if not ok:
                continue
            try:
                cc_, ci_, _, _ = det.detectBoard(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
            except cv2.error:
                continue
            if ci_ is not None and len(ci_) >= 8:
                found[fi] = (ci_.ravel().astype(int), cc_.reshape(-1, 2).astype(np.float64))
        cap.release()
        dets[n] = [found.get(int(x)) for x in i]
        log(f"  detection {n}: board found in {sum(v is not None for v in dets[n])} of {len(grid)} sampled moments")
    return grid, dets


def _static_corners(names, floor_videos, det, n_frames=14, min_obs=4):
    """median pixel position of every board corner per camera over frames of a STATIC floor take -> (nc, ncorners, 2) with NaN"""
    import cv2
    nco = det.n_corners
    p2d = np.full((len(names), nco, 2), np.nan)
    for ci, n in enumerate(names):
        cap = cv2.VideoCapture(str(floor_videos[n]))
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        obs = {}
        for k in np.linspace(min(40, nf // 4), max(nf - 40, nf * 3 // 4), n_frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, fr = cap.read()
            if not ok:
                continue
            try:
                cc_, ci_, _, _ = det.detectBoard(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
            except cv2.error:
                continue
            if ci_ is not None:
                for c, i in zip(cc_.reshape(-1, 2), ci_.ravel()):
                    obs.setdefault(int(i), []).append(c)
        cap.release()
        for i, v in obs.items():
            if len(v) >= min_obs:
                p2d[ci, i] = np.median(np.array(v), axis=0)
    return p2d


# ---------------------------------------------------------------------------------------------------------------- floor
def fit_floor(toml_in, floor_videos, det, out_dir, log=print):
    """z = 0 on the plane of the floor take (board flat, static): port of the 2026-11-18 floor_plane.py (which depended on the old calibrate_arena.py - removed).
    Writes out_dir\\calibration.toml (floor frame: x along the board rows, z up), floor_plane.json, floor_report.txt.  Returns dict(toml, plane_rms_mm, scale_ratio, seen)."""
    import cv2
    from aniposelib.cameras import CameraGroup
    cg = CameraGroup.load(str(toml_in))
    names = cg.get_names()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    obj = board_corners_mm(det)
    p2d = _static_corners(names, floor_videos, det, n_frames=25, min_obs=5)
    seen = (~np.isnan(p2d[:, :, 0])).sum(axis=1)
    pts = cg.triangulate(p2d, progress=False)
    ok = np.isfinite(pts).all(axis=1) & ((~np.isnan(p2d[:, :, 0])).sum(axis=0) >= 2)
    if ok.sum() < 8:
        raise RuntimeError(f"only {int(ok.sum())} floor corners could be triangulated - the board must be flat, still and seen by most cameras")
    P = pts[ok]
    c0 = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c0)
    normal = Vt[2]
    centres = np.array([-cv2.Rodrigues(np.asarray(c.get_rotation(), dtype=np.float64))[0].T @ np.asarray(c.get_translation(), dtype=np.float64) for c in cg.cameras])
    if (centres.mean(axis=0) - c0) @ normal < 0:
        normal = -normal                                                                    # z up = towards the cameras
    resid = (P - c0) @ normal
    plane_rms = float(np.sqrt(np.mean(resid ** 2)))
    nc_ = det.n_cols
    sq = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1) if (i + 1) % nc_ != 0 and ok[i] and ok[i + 1]]
    sq_nom = float(np.linalg.norm(obj[1] - obj[0]))
    ratio = float(np.median(sq) / sq_nom) if sq else float("nan")
    xdir = pts[nc_ - 1] - pts[0] if ok[nc_ - 1] and ok[0] else P[-1] - P[0]
    xdir = xdir - (xdir @ normal) * normal
    xdir /= np.linalg.norm(xdir)
    ydir = np.cross(normal, xdir)
    A = np.vstack([xdir, ydir, normal])
    for cam in cg.cameras:
        R = cv2.Rodrigues(np.asarray(cam.get_rotation(), dtype=np.float64))[0]
        t = np.asarray(cam.get_translation(), dtype=np.float64)
        cam.set_rotation(cv2.Rodrigues(R @ A.T)[0].ravel())
        cam.set_translation(R @ c0 + t)
    pts2 = cg.triangulate(p2d, progress=False)
    lines = [f"floor take: corners seen per camera {dict(zip(names, seen.tolist()))}", f"floor corners triangulated: {int(ok.sum())} of {len(pts)}",
             f"plane fit: RMS distance of corners to plane = {plane_rms:.3f} mm (max {np.abs(resid).max():.3f} mm)",
             f"square size from the floor take: median {np.median(sq):.3f} mm (nominal {sq_nom:.3f} mm, ratio {ratio:.4f})" if sq else "square size: n/a",
             f"after re-expression: floor corners z = {np.nanmean(pts2[ok][:, 2]):.4f} mm (should be 0), spread {np.nanstd(pts2[ok][:, 2]):.3f} mm"]
    for n, cam in zip(names, cg.cameras):
        R = cv2.Rodrigues(np.asarray(cam.get_rotation(), dtype=np.float64))[0]
        c = -R.T @ np.asarray(cam.get_translation(), dtype=np.float64)
        lines.append(f"  {n}: x={c[0]:7.0f} y={c[1]:7.0f} height={c[2]:7.0f} mm")
    toml_out = out_dir / "calibration.toml"
    tmp = toml_out.with_name("calibration.toml.tmp")
    cg.dump(str(tmp)); tmp.replace(toml_out)
    json.dump({"normal_in_calibration_frame": normal.tolist(), "origin_in_calibration_frame": c0.tolist(), "A_floor_from_calibration": A.tolist(), "plane_rms_mm": plane_rms,
               "note": "z=0 is the TOP surface of the board lying on the floor (board thickness ~ a few mm above the true floor)"}, open(out_dir / "floor_plane.json", "w"), indent=1)
    (out_dir / "floor_report.txt").write_text("\n".join(lines), encoding="utf-8")
    for s in lines:
        log("  " + s)
    return dict(toml=toml_out, plane_rms_mm=plane_rms, scale_ratio=ratio, seen=seen.tolist())


def floor_loo(cg, floor_videos, det):
    """Leave-one-camera-out on a floor take: triangulate the corners from five cameras, project into the sixth.  A camera that MOVED since the calibration stands out
    (a moved BOARD does not).  Returns {cam: median px}; normal for a good calibration 1-3 px."""
    names = cg.get_names()
    p2d = _static_corners(names, floor_videos, det, n_frames=12, min_obs=4)
    out = {}
    for ho in range(len(names)):
        q = p2d.copy(); q[ho] = np.nan
        P = cg.triangulate(q, progress=False)
        ok = np.isfinite(P).all(axis=1) & ((~np.isnan(q[:, :, 0])).sum(axis=0) >= 3) & ~np.isnan(p2d[ho, :, 0])
        if ok.sum() >= 5:
            out[names[ho]] = float(np.median(np.linalg.norm(cg.project(P)[ho] - p2d[ho], axis=1)[ok]))
    return out


# ---------------------------------------------------------------------------------------------------------------- measurements in mm
def _same_face_groups(cams, dets, names, k, obj, same_face_dot=0.3, min_corners=8):
    import cv2
    seen, nrm = [], {}
    for ci, n in enumerate(names):
        d = dets[n][k]
        if d is None or len(d[0]) < min_corners:
            continue
        ids, pts = d
        ok, rv, _ = cv2.solvePnP(obj[ids].astype(np.float32), pts.astype(np.float32), cams[ci]["K"], cams[ci]["D"], flags=cv2.SOLVEPNP_IPPE if len(ids) < 8 else cv2.SOLVEPNP_ITERATIVE)
        if ok:
            seen.append(ci); nrm[ci] = cams[ci]["R"].T @ cv2.Rodrigues(rv)[0][:, 2]
    if len(seen) < 3:
        return []
    best = max(([m for m in seen if float(nrm[s] @ nrm[m]) > same_face_dot] for s in seen), key=len)
    return best if len(best) >= 3 else []


def measure_move(cg, pre, obj_mm, log=print):
    """known board lengths measured by triangulation (largest SAME-FACE camera group per moment) vs the true board geometry, in mm.  pre = (grid, dets)."""
    grid, dets = pre
    cams = _cams(cg)
    names = cg.get_names()
    nco = len(obj_mm)
    DN, ER, ZZ, DEPTH_PX = [], [], [], {n: [] for n in names}
    used = 0
    for k in range(len(grid)):
        best = _same_face_groups(cams, dets, names, k, obj_mm)
        if not best:
            continue
        p2d = np.full((len(names), nco, 2), np.nan)
        for ci in best:
            ids, pts = dets[names[ci]][k]
            p2d[ci, ids] = pts
        cnt = (~np.isnan(p2d[:, :, 0])).sum(axis=0)
        P = cg.triangulate(p2d, progress=False)
        ok = np.isfinite(P).all(axis=1) & (cnt >= 3)
        if ok.sum() < 8:
            continue
        used += 1
        idx = np.where(ok)[0]
        a, b = np.triu_indices(len(idx), 1)
        ia, ib = idx[a], idx[b]
        dm = np.linalg.norm(P[ia] - P[ib], axis=1)
        dn = np.linalg.norm(obj_mm[ia] - obj_mm[ib], axis=1)
        DN.append(dn); ER.append(dm - dn); ZZ.append((P[ia, 2] + P[ib, 2]) / 2)
        pr = cg.project(P)
        for ci in best:
            e = np.linalg.norm(pr[ci] - p2d[ci], axis=1)
            good = ok & np.isfinite(e)
            if good.any():
                zc = (cams[ci]["R"] @ P[good].T).T[:, 2] + cams[ci]["t"][2]
                DEPTH_PX[names[ci]].append((float(np.median(e[good])), float(np.median(zc)) / cams[ci]["K"][0, 0]))
    log(f"  moments used for the length test: {used} of {len(grid)}")
    if not DN:
        raise RuntimeError("no moment with >= 3 cameras on the same board face")
    return dict(dn=np.concatenate(DN), err=np.concatenate(ER), z=np.concatenate(ZZ), n_moments=used, n_sampled=len(grid),
                cam_px_mm={n: (float(np.median([a for a, _ in v])), float(np.median([b for _, b in v]))) for n, v in DEPTH_PX.items() if v})


def measure_floor(cg, floor_videos, det, log=print):
    """floor take (static board, NOT used in the solve): flatness = distance of the corners to their best-fit plane, leftover px per camera and in mm at that camera's distance"""
    cams = _cams(cg)
    names = cg.get_names()
    p2d = _static_corners(names, floor_videos, det, n_frames=14)
    P = cg.triangulate(p2d, progress=False)
    ok = np.isfinite(P).all(axis=1) & ((~np.isnan(p2d[:, :, 0])).sum(axis=0) >= 3)
    pr = cg.project(P)
    out = {}
    for ci, n in enumerate(names):
        e = np.linalg.norm(pr[ci] - p2d[ci], axis=1)
        g = ok & np.isfinite(e)
        if g.any():
            zc = (cams[ci]["R"] @ P[g].T).T[:, 2] + cams[ci]["t"][2]
            out[n] = (float(np.median(e[g])), float(np.median(zc)) / cams[ci]["K"][0, 0])
    log(f"  floor take: {int(ok.sum())} of {len(P)} corners triangulated")
    plane_rms, zmap = float("nan"), np.full(len(P), np.nan)
    if ok.sum() >= 6:
        Q = P[ok]; c0 = Q.mean(axis=0); _, _, Vt = np.linalg.svd(Q - c0)
        dist = (P - c0) @ Vt[2]
        plane_rms = float(np.sqrt(np.mean(dist[ok] ** 2)))
        zmap = np.where(ok, dist, np.nan)
    return dict(z=zmap, plane_rms=plane_rms, cam_px_mm=out)


# ---------------------------------------------------------------------------------------------------------------- refinement
def refine_calibration(toml_in, dets_pack, floor_videos, work_dir, det, priors=None, log=print):
    """held-out test + rigid-board BA with per-camera lens; returns dict(adopted, final_toml (floor-aligned when a floor take exists), reason, ...).  Never raises."""
    from aniposelib.cameras import CameraGroup
    import calibration_rigid_ba as rba
    work = Path(work_dir); work.mkdir(parents=True, exist_ok=True)
    pri = dict(PRIORS, **(priors or {}))
    out = dict(adopted=False, final_toml=str(toml_in), reason="", priors=pri)
    try:
        cg0 = CameraGroup.load(str(toml_in))
        names = cg0.get_names()
        obj = board_corners_mm(det)
        grid, dets = dets_pack
        nm = len(grid)
        train, test = list(range(0, nm, 2)), list(range(1, nm, 2))
        pre_test = (grid[test], {n: [dets[n][k] for k in test] for n in names})

        def evaluate(cg):
            mv = measure_move(cg, pre_test, obj, log=lambda s: None)
            ae = np.abs(mv["err"])
            r = dict(median=float(np.median(ae)), p95=float(np.percentile(ae, 95)), moments=mv["n_moments"], floor_px=[], floor_px_max=0.0, plane_rms=float("nan"))
            if floor_videos:
                fl = measure_floor(cg, floor_videos, det, log=lambda s: None)
                r.update(floor_px=[round(v[0], 2) for v in fl["cam_px_mm"].values()], floor_px_max=float(max(v[0] for v in fl["cam_px_mm"].values())), plane_rms=fl["plane_rms"])
            return r
        log("[refine] held-out test: fit on half of the moments, judge on the other half" + (" + the floor take" if floor_videos else ""))
        base = evaluate(cg0)
        log(f"[refine]   starting calibration: median {base['median']:.3f} mm, 95 % {base['p95']:.3f} mm" + (f", floor take worst camera {base['floor_px_max']:.2f} px" if floor_videos else ""))
        r_tr = rba.refine(toml_in, dets, grid, obj, names, moments=train, log=lambda s: log("  " + s), **pri)
        ref = evaluate(CameraGroup.load(str(rba.write_toml(toml_in, r_tr, work / "refined_train.toml"))))
        log(f"[refine]   refined (held-out):   median {ref['median']:.3f} mm, 95 % {ref['p95']:.3f} mm" + (f", floor take worst camera {ref['floor_px_max']:.2f} px" if floor_videos else ""))
        out.update(base=base, refined_heldout=ref)
        ok = ref["median"] <= base["median"] * 1.02 and ref["p95"] <= base["p95"] * 1.02 and (not floor_videos or ref["floor_px_max"] <= base["floor_px_max"] * 1.05)
        if not ok:
            out["reason"] = "the rigid-board refinement was not better on held-out data - the starting calibration is kept"
            log("[refine] " + out["reason"])
            return out
        log("[refine] adopted: fitting on ALL moments ...")
        r_all = rba.refine(toml_in, dets, grid, obj, names, log=lambda s: log("  " + s), **pri)
        t_all = rba.write_toml(toml_in, r_all, work / "refined_all.toml")
        final = t_all
        if floor_videos:
            ff = fit_floor(t_all, floor_videos, det, work / "refined_floor", log=lambda s: None)
            final = ff["toml"]
            out["plane_rms_align_mm"] = ff["plane_rms_mm"]
        fin = evaluate(CameraGroup.load(str(final)))
        log(f"[refine] final: median {fin['median']:.3f} mm, 95 % {fin['p95']:.3f} mm" + (f", floor take worst camera {fin['floor_px_max']:.2f} px, floor flat to {fin['plane_rms']:.3f} mm" if floor_videos else ""))
        out.update(adopted=True, final_toml=str(final), final_check=fin, lens=r_all["per_cam"], frames=r_all["frames"], rms_px=r_all["rms1"],
                   reason="better on held-out moments" + (" and on the floor take" if floor_videos else ""))
    except Exception as e:
        out["reason"] = f"refinement failed ({type(e).__name__}: {e}) - the starting calibration is kept"
        log("[refine] " + out["reason"])
    finally:
        (work / "refine_summary.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


def centres(toml):
    import cv2
    from aniposelib.cameras import CameraGroup
    cg = CameraGroup.load(str(toml))
    return {n: -cv2.Rodrigues(np.asarray(c.get_rotation(), dtype=np.float64))[0].T @ np.asarray(c.get_translation(), dtype=np.float64) for n, c in zip(cg.get_names(), cg.cameras)}


# ---------------------------------------------------------------------------------------------------------------- the figure
def make_report(toml, pre, floor_videos, det, report_dirs, stamp=None, prev_toml=None, verdict="", heldout=None, log=print):
    """one page, plain language, every error in mm -> calibration_accuracy_<stamp>.png/.json in every report dir; returns the numbers.  heldout = {'median','p95'} (headline)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aniposelib.cameras import CameraGroup
    stamp = stamp or time.strftime("%Y-%m-%d_%H-%M-%S")
    cg = CameraGroup.load(str(toml))
    names = cg.get_names()
    obj = board_corners_mm(det)
    sq_mm = float(np.linalg.norm(obj[1] - obj[0]))
    log("=== accuracy in mm (figures) ===")
    mv = measure_move(cg, pre, obj, log=log)
    fl = measure_floor(cg, floor_videos, det, log=log) if floor_videos else None
    ae = np.abs(mv["err"])
    adj = np.abs(mv["dn"] - sq_mm) < 0.5
    s = float(np.median((mv["dn"] + mv["err"]) / mv["dn"]))
    num = dict(stamp=stamp, verdict=verdict, moments=mv["n_moments"], pairs=int(len(ae)), median_abs_err_mm=float(np.median(ae)), p95_abs_err_mm=float(np.percentile(ae, 95)),
               adjacent_median_abs_err_mm=float(np.median(ae[adj])) if adj.any() else None, adjacent_p95_abs_err_mm=float(np.percentile(ae[adj], 95)) if adj.any() else None,
               scale_measured_over_nominal=s, per_camera_move_px_mm={n: dict(px=v[0], mm=v[0] * v[1]) for n, v in mv["cam_px_mm"].items()})
    if fl is not None:
        num.update(floor_height_rms_mm=float(np.sqrt(np.nanmean(fl["z"] ** 2))), floor_height_max_abs_mm=float(np.nanmax(np.abs(fl["z"]))), floor_plane_rms_mm=fl["plane_rms"],
                   per_camera_floor_px_mm={n: dict(px=v[0], mm=v[0] * v[1]) for n, v in fl["cam_px_mm"].items()})
    else:
        num.update(floor_height_rms_mm=None, floor_height_max_abs_mm=None, floor_plane_rms_mm=None, per_camera_floor_px_mm={})
    hm, hp = (heldout["median"], heldout["p95"]) if heldout else (num["median_abs_err_mm"], num["p95_abs_err_mm"])
    num["headline_median_mm"], num["headline_p95_mm"], num["headline_is_heldout"] = float(hm), float(hp), bool(heldout)

    C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", grey="#6b7280", ink="#111827")
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 13, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(19, 11.5), facecolor="white")
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.955, top=0.80, bottom=0.07, wspace=0.30, hspace=0.42)
    col = C["green"] if hp <= GOOD_MM else (C["orange"] if hp <= OK_MM else C["red"])
    fig.text(0.055, 0.945, f"Calibration accuracy   -   {stamp}" + (f"   -   {verdict}" if verdict else ""), fontsize=22, fontweight="bold", color=C["ink"])
    fig.text(0.055, 0.895, f"A length measured anywhere in the arena is typically off by {hm:.2f} mm", fontsize=19, color=col, fontweight="bold")
    how = ("tested on moments the calibration never saw" if heldout else f"tested on {num['pairs']:,} corner pairs from {num['moments']} moments of the move take")
    fig.text(0.055, 0.855, f"95 % of all measured lengths are within {hp:.2f} mm   (a mouse is ~ 90 mm long;  {how})", fontsize=14, color=C["grey"])
    rng = np.random.default_rng(0)
    sel = rng.choice(len(ae), size=min(6000, len(ae)), replace=False)
    ylim = max(1.5, 1.1 * np.percentile(ae, 99))

    ax = fig.add_subplot(gs[0, 0])
    ax.axhspan(-GOOD_MM, GOOD_MM, color=C["green"], alpha=0.12, lw=0)
    ax.scatter(mv["dn"][sel], mv["err"][sel], s=5, alpha=0.18, color=C["blue"], lw=0)
    c, md, lo, hi = _binned(mv["dn"], mv["err"], np.linspace(mv["dn"].min(), mv["dn"].max(), 9))
    ax.plot(c, md, color=C["ink"], lw=2.2); ax.fill_between(c, lo, hi, color=C["ink"], alpha=0.12, lw=0); ax.axhline(0, color=C["grey"], lw=1)
    ax.set_title("Does it measure a known length correctly?"); ax.set_xlabel("true length on the board (mm)"); ax.set_ylabel("measured - true (mm)"); ax.set_ylim(-ylim, ylim)
    ax.text(0.03, 0.04, "line = typical,  grey band = 90 % of pairs,  green = within 0.5 mm", transform=ax.transAxes, fontsize=11, color=C["grey"])

    ax = fig.add_subplot(gs[0, 1])
    ax.axhspan(-GOOD_MM, GOOD_MM, color=C["green"], alpha=0.12, lw=0)
    ax.scatter(mv["z"][sel], mv["err"][sel], s=5, alpha=0.18, color=C["orange"], lw=0)
    c, md, lo, hi = _binned(mv["z"], mv["err"], np.linspace(max(0, mv["z"].min()), mv["z"].max(), 9))
    ax.plot(c, md, color=C["ink"], lw=2.2); ax.fill_between(c, lo, hi, color=C["ink"], alpha=0.12, lw=0); ax.axhline(0, color=C["grey"], lw=1)
    ax.axvspan(0, 60, color=C["blue"], alpha=0.06, lw=0); ax.text(30, ylim * 0.55, "mouse\nheight", ha="center", va="top", fontsize=11, color=C["blue"])
    ax.set_title("Is it equally accurate at every height?"); ax.set_xlabel("height above the floor (mm)" if fl is not None else "z in the calibration frame (mm)"); ax.set_ylabel("measured - true length (mm)"); ax.set_ylim(-ylim, ylim)

    ax = fig.add_subplot(gs[0, 2])
    if fl is not None:
        zf = fl["z"].reshape(det.n_corners // det.n_cols, det.n_cols)
        lim = max(0.5, float(np.nanmax(np.abs(zf))))
        im = ax.imshow(zf, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
        for r in range(zf.shape[0]):
            for q in range(zf.shape[1]):
                if np.isfinite(zf[r, q]):
                    ax.text(q, r, f"{zf[r, q]:+.2f}", ha="center", va="center", fontsize=13, fontweight="bold", color="white" if abs(zf[r, q]) > 0.6 * lim else C["ink"])
        ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
        ax.set_title("Is the floor flat? (test not used in the solve)")
        ax.set_xlabel(f"distance of the board corners from their best-fit plane (mm)\nflat to {num['floor_height_rms_mm']:.2f} mm (RMS), worst {num['floor_height_max_abs_mm']:.2f} mm")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="mm")
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "no floor take: the floor test and z = 0 are not available", ha="center", va="center", fontsize=14, color=C["grey"])

    ax = fig.add_subplot(gs[1, 0])
    xs = np.arange(len(names)); w = 0.38
    mvv = [num["per_camera_move_px_mm"].get(n, dict(mm=np.nan, px=np.nan)) for n in names]
    flv = [num["per_camera_floor_px_mm"].get(n, dict(mm=np.nan, px=np.nan)) for n in names]
    b1 = ax.bar(xs - w / 2, [v["mm"] for v in mvv], w, color=C["blue"], label="move take")
    groups = [(b1, mvv)]
    if fl is not None:
        groups.append((ax.bar(xs + w / 2, [v["mm"] for v in flv], w, color=C["orange"], label="floor take (independent)"), flv))
    for bars, vals in groups:
        for bb, v in zip(bars, vals):
            if np.isfinite(v["mm"]):
                ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.005, f"{v['px']:.1f} px", ha="center", va="bottom", fontsize=9.5, color=C["grey"])
    ax.axhline(GOOD_MM, color=C["green"], lw=1.5, ls="--"); ax.text(len(names) - 0.5, GOOD_MM, " 0.5 mm", va="bottom", ha="right", fontsize=10, color=C["green"])
    ax.set_xticks(xs); ax.set_xticklabels(names); ax.set_ylabel("leftover error at the board (mm)")
    ax.set_title("How big is each camera's leftover error?"); ax.legend(frameon=False, fontsize=11, loc="upper left")

    ax = fig.add_subplot(gs[1, 1])
    chg = None
    if prev_toml and Path(prev_toml).is_file():
        try:
            new, old = centres(toml), centres(prev_toml)
            nm_ = sorted(set(new) & set(old))
            chg = sorted([(abs(np.linalg.norm(new[p] - new[q]) - np.linalg.norm(old[p] - old[q])), f"{p}-{q}") for i, p in enumerate(nm_) for q in nm_[i + 1:]], reverse=True)[:10]
        except Exception as e:
            log(f"  (previous calibration comparison skipped: {e})")
    if chg:
        v = [a for a, _ in chg][::-1]; lab = [b for _, b in chg][::-1]
        ax.barh(lab, v, color=[C["green"] if a < 2 else (C["orange"] if a < 6 else C["red"]) for a in v])
        for i, a in enumerate(v):
            ax.text(a + 0.15, i, f"{a:.1f}", va="center", fontsize=11, color=C["ink"])
        ax.set_xlabel("change in camera-to-camera distance (mm)")
        num["camera_pair_distance_change_max_mm"] = float(max(v))
        ax.set_title("Did the cameras move since the last calibration?")
        ax.text(0.98, 0.04, "green < 2 mm: did not move\nred > 6 mm: moved", transform=ax.transAxes, ha="right", fontsize=11, color=C["grey"])
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "no earlier calibration to compare with", ha="center", va="center", fontsize=14, color=C["grey"])

    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    pm = num["per_camera_move_px_mm"]
    lo_mm, hi_mm = np.nanmin([v["mm"] for v in pm.values()]), np.nanmax([v["mm"] for v in pm.values()])
    lines = [("In plain words", True),
             (f"Known lengths come out within {hm:.2f} mm", False), (f"typically, and within {hp:.2f} mm for 95 % of them.", False),
             (f"One {sq_mm:.2f} mm square is measured {sq_mm * s:.3f} mm", False), (f"(scale {100 * (s - 1):+.2f} %).", False),
             (f"The floor is flat to {num['floor_height_rms_mm']:.2f} mm." if fl is not None else "No floor take was available.", False),
             (f"Leftover error per camera: {lo_mm:.2f} - {hi_mm:.2f} mm.", False), ("", False),
             ("Green = within 0.5 mm, amber = within 1 mm,", False), ("red = more.", False),
             (f"The board's printed size ({sq_mm:.2f} mm) is", False), ("assumed: check it once with calipers.", False)]
    y = 0.97
    for t, bold in lines:
        ax.text(0.0, y, t, fontsize=14 if bold else 12.5, fontweight="bold" if bold else "normal", color=C["ink"] if bold else "#1f2937", va="top", clip_on=False)
        y -= 0.075
    names_out = []
    for d in report_dirs:
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        png = d / f"calibration_accuracy_{stamp}.png"
        fig.savefig(png, dpi=140, facecolor="white")
        (d / f"calibration_accuracy_{stamp}.json").write_text(json.dumps(num, indent=1), encoding="utf-8")
        names_out.append(str(png))
    plt.close(fig)
    for p in names_out:
        log(f"FIGURE: {p}")
    return num


def _binned(x, y, edges):
    c, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= 30:
            c.append((a + b) / 2); med.append(np.median(y[m])); lo.append(np.percentile(y[m], 5)); hi.append(np.percentile(y[m], 95))
    return np.array(c), np.array(med), np.array(lo), np.array(hi)


# ---------------------------------------------------------------------------------------------------------------- the whole standard analysis
def finalize(out_dir, solver_toml, videos, floor_videos=None, board=None, report_dirs=(), stamp=None, prev_toml=None, step_s=0.2, cache_dir=None, refine=True, log=print):
    """the standard analysis on top of a solver result.  Writes out_dir\\calibration.toml LAST (+ analysis.json, calibration_solver_only.toml).  Returns the analysis dict;
    ['final_toml'] is out_dir\\calibration.toml.  Never raises: on any failure the solver result becomes the calibration and analysis.json says why."""
    from aniposelib.cameras import CameraGroup
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    board = board or {}
    det = make_detector(board.get("squares_x", 6), board.get("squares_y", 5), board.get("square_length", 12.22), board.get("marker_length", 9.165),
                        board.get("marker_bits", 4), board.get("dict_size", 50))
    stamp = stamp or time.strftime("%Y-%m-%d_%H-%M-%S")
    work = Path(cache_dir) if cache_dir else out_dir / "_fine"
    work.mkdir(parents=True, exist_ok=True)
    notes, ana = [], dict(analysis_version=ANALYSIS_VERSION, stamp=stamp, floor_take=bool(floor_videos), solver_toml=str(solver_toml))
    current = Path(solver_toml)
    try:
        # 1. floor alignment of the solver result
        if floor_videos:
            log("[fine] floor take: z = 0 on the floor plane")
            ff = fit_floor(current, floor_videos, det, work / "solver_floor", log=log)
            current = ff["toml"]
            ana.update(floor_corners_seen=ff["seen"], plane_rms_solver_mm=ff["plane_rms_mm"], scale_ratio_floor_fit=ff["scale_ratio"])
            try:
                ana["floor_consistency_px"] = floor_loo(CameraGroup.load(str(current)), floor_videos, det)
                log("[fine] floor take consistency (held-out camera error, px; normal 1-3): " + ", ".join(f"{k} {v:.1f}" for k, v in ana["floor_consistency_px"].items()))
            except Exception as e:
                notes.append(f"consistency check skipped: {e}")
        # 2. detections of the move take (cached: a crash costs nothing)
        names = CameraGroup.load(str(current)).get_names()
        pk = work / "dets.pkl"
        if pk.exists():
            pre = pickle.load(open(pk, "rb")); log("[fine] detections from the cache")
        else:
            log("[fine] detecting the board over the move take ...")
            pre = detect_moments(names, videos, step_s, det, log=lambda s: log("  " + s))
            pickle.dump(pre, open(pk, "wb"))
        # 3. rigid-board refinement
        if refine:
            ref = refine_calibration(current, pre, floor_videos, work / "refine", det, log=log)
        else:
            ref = dict(adopted=False, final_toml=str(current), reason="refinement switched off")
        ana["refinement"] = {k: v for k, v in ref.items() if k not in ("lens",)}
        ana["lens"] = ref.get("lens")
        final = Path(ref["final_toml"]) if ref.get("adopted") else current
        heldout = dict(median=ref["refined_heldout"]["median"], p95=ref["refined_heldout"]["p95"]) if ref.get("adopted") else (
            dict(median=ref["base"]["median"], p95=ref["base"]["p95"]) if ref.get("base") else None)
        # 4. figures
        try:
            num = make_report(final, pre, floor_videos, det, list(report_dirs), stamp, prev_toml, "", heldout=heldout, log=log)
            ana["accuracy"] = num
        except Exception as e:
            notes.append(f"figure failed: {type(e).__name__}: {e}")
            log(f"[fine] (figure skipped: {type(e).__name__}: {e})")
        # 5. publish into out_dir - the final calibration LAST
        for f in ("floor_plane.json", "floor_report.txt"):
            src = (final.parent / f)
            if src.is_file():
                shutil.copy2(src, out_dir / f)
        tmp = out_dir / "calibration.toml.tmp"
        shutil.copy2(final, tmp)
        tmp.replace(out_dir / "calibration.toml")
        ana.update(adopted=bool(ref.get("adopted")), final_toml=str(out_dir / "calibration.toml"), heldout=heldout)
    except Exception as e:
        notes.append(f"fine analysis failed ({type(e).__name__}: {e}) - the solver result is the calibration")
        log("[fine] " + notes[-1])
        tmp = out_dir / "calibration.toml.tmp"
        shutil.copy2(solver_toml, tmp)
        tmp.replace(out_dir / "calibration.toml")
        ana.update(adopted=False, final_toml=str(out_dir / "calibration.toml"))
    ana["notes"] = notes
    (out_dir / "analysis.json").write_text(json.dumps(ana, indent=1, default=str), encoding="utf-8")
    return ana


def cache_is_current(auto_dir):
    """True when out_dir\\analysis.json carries the current ANALYSIS_VERSION (a calibration_auto cache of an OLDER analysis is not trusted by Step 1b)"""
    try:
        return json.loads((Path(auto_dir) / "analysis.json").read_text(encoding="utf-8")).get("analysis_version") == ANALYSIS_VERSION
    except (OSError, ValueError):
        return False
