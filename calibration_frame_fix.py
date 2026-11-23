"""
Restore the metric scale and world frame of a pose-refined calibration.

Why this exists (found 2026-11-19, 6-camera arena rig, `D:\\6CAM RECORDINGS`):
`calibration_refine_poses.py` refines the camera EXTRINSICS from animal
keypoints only.  Keypoints carry no metric information, so the bundle
adjustment is free to change the world's scale and to rotate/translate the
world frame (a 7-parameter similarity "gauge").  On the real rig the refined
calibration came out 1.344x larger than the arena calibration (cameras
spread 340 -> 457 mm), with the z=0 floor plane gone (paws near z = -285 mm)
and the frame tilted -- so every triangulated millimetre value (speeds,
distances, heights) was inflated by ~1.34 and referenced to a lost floor.
The geometry itself (reprojection) was better; only the gauge was wrong.

Fix: the ORIGINAL calibration (board-based, metric, floor-referenced) fixes
the gauge.  The camera centres of the refined and original calibrations are
the same physical points, so a similarity (Umeyama 1991) between the two
sets of centres gives the exact map refined-world -> original-world.  It is
applied to the refined cameras' extrinsics (projection is unchanged, only the
world coordinates become metric again) and, for data already triangulated
with a drifted calibration, to the 3D points themselves.

Convention used throughout:  X_ref = s * R @ X_new + t
(`new` = the refined/drifted frame, `ref` = the original, metric frame).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:  # cv2 is part of the cube env; imported lazily-safe for unit tests
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# ─────────────────────────────────────────────────────────────────────────────
#  Similarity estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_similarity(src: np.ndarray, dst: np.ndarray, min_spread_ratio: float = 0.02):
    """
    Umeyama similarity  dst ~= s * R @ src + t  (proper rotation, R in SO(3)).

    src, dst: (N, 3) corresponding points, N >= 3, not collinear.
    Returns dict(s, R, t, residual (N,), rms) or None when ill-posed
    (fewer than 3 points, or all points (nearly) on one line).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        return None
    mu_s, mu_d = src.mean(0), dst.mean(0)
    A, B = src - mu_s, dst - mu_d
    sv = np.linalg.svd(A, compute_uv=False)
    if sv[0] < 1e-9 or sv[1] < min_spread_ratio * sv[0]:
        return None                      # collinear: rotation about the line is undetermined
    n = len(src)
    cov = B.T @ A / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (A ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - s * R @ mu_s
    res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    return {"s": s, "R": R, "t": t, "residual": res, "rms": float(np.sqrt((res ** 2).mean())),
            "partial": False}


def estimate_similarity_two_points(src: np.ndarray, dst: np.ndarray):
    """Two-camera fallback: scale from the baseline, translation from camera 0,
    MINIMAL rotation taking the src baseline direction onto the dst one.  The
    roll about the baseline is unconstrained by two points and is left at the
    minimal-rotation value (flagged ``partial``)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != (2, 3) or dst.shape != (2, 3):
        return None
    vs, vd = src[1] - src[0], dst[1] - dst[0]
    ls, ld = np.linalg.norm(vs), np.linalg.norm(vd)
    if ls < 1e-9 or ld < 1e-9:
        return None
    a, b = vs / ls, vd / ld
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        R = np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(a, a)  # 180 deg about any perpendicular axis
        if c <= 0:
            perp = np.cross(a, [1.0, 0, 0])
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(a, [0, 1.0, 0])
            perp /= np.linalg.norm(perp)
            R = 2 * np.outer(perp, perp) - np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    s = ld / ls
    t = dst[0] - s * R @ src[0]
    res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    return {"s": float(s), "R": R, "t": t, "residual": res,
            "rms": float(np.sqrt((res ** 2).mean())), "partial": True}


def similarity_from_camera_centers(centers_ref: dict, centers_new: dict):
    """Similarity new->ref from the cameras present in both dicts (name -> (3,))."""
    names = [n for n in centers_ref if n in centers_new]
    if len(names) < 2:
        return None, names
    dst = np.array([centers_ref[n] for n in names], dtype=np.float64)
    src = np.array([centers_new[n] for n in names], dtype=np.float64)
    sim = estimate_similarity(src, dst) if len(names) >= 3 else None
    if sim is None:
        sim = estimate_similarity_two_points(src[:2], dst[:2]) if len(names) == 2 else None
        if sim is None and len(names) >= 3:     # collinear rig: use the two farthest cameras
            d = np.linalg.norm(src[:, None] - src[None], axis=-1)
            i, j = np.unravel_index(np.argmax(d), d.shape)
            sim = estimate_similarity_two_points(src[[i, j]], dst[[i, j]])
            if sim is not None:
                res = np.linalg.norm((sim["s"] * (sim["R"] @ src.T)).T + sim["t"] - dst, axis=1)
                sim["residual"], sim["rms"] = res, float(np.sqrt((res ** 2).mean()))
    return sim, names


# ─────────────────────────────────────────────────────────────────────────────
#  Applying a similarity
# ─────────────────────────────────────────────────────────────────────────────

def camera_centers(cgroup) -> dict:
    out = {}
    for cam in cgroup.cameras:
        R, _ = cv2.Rodrigues(np.asarray(cam.get_rotation(), dtype=np.float64))
        t = np.asarray(cam.get_translation(), dtype=np.float64).reshape(3)
        out[cam.get_name()] = -R.T @ t
    return out


def apply_similarity_to_camera(cam, s: float, R: np.ndarray, t: np.ndarray) -> None:
    """
    Re-express one camera's extrinsics for the world map  X_ref = s R X_new + t.

    With  x_cam = Rc X_new + tc  and  X_new = R^T (X_ref - t) / s :
        s * x_cam = (Rc R^T) X_ref + (s tc - Rc R^T t)
    so  R' = Rc R^T,  t' = s tc - R' t.  x_cam only changes by the positive
    factor s, which the pinhole projection (and the distortion model, which
    acts on x/z) is invariant to: image coordinates are unchanged.
    """
    Rc, _ = cv2.Rodrigues(np.asarray(cam.get_rotation(), dtype=np.float64))
    tc = np.asarray(cam.get_translation(), dtype=np.float64).reshape(3)
    Rn = Rc @ R.T
    tn = s * tc - Rn @ t
    cam.set_rotation(cv2.Rodrigues(Rn)[0].reshape(3))
    cam.set_translation(tn.reshape(3))


def apply_similarity_to_points(X: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """X (..., 3) -> s * R @ X + t.  NaN rows stay NaN."""
    X = np.asarray(X, dtype=np.float64)
    return s * (X @ R.T) + t


def realign_cgroup(ref_cgroup, new_cgroup, cameras: list | None = None, warn_frac: float = 0.10):
    """
    Put `new_cgroup`'s cameras into `ref_cgroup`'s world frame/scale, IN PLACE.
    The similarity is estimated from the camera centres of `cameras` (default:
    every camera present in both) and applied to those cameras.

    Returns an info dict (never raises for an ill-posed fit; ``applied`` False
    and a ``reason`` are returned instead, leaving `new_cgroup` untouched).
    """
    c_ref = camera_centers(ref_cgroup)
    c_new = camera_centers(new_cgroup)
    use = [n for n in (cameras or list(c_new)) if n in c_ref and n in c_new]
    sim, names = similarity_from_camera_centers({n: c_ref[n] for n in use}, {n: c_new[n] for n in use})
    if sim is None:
        return {"applied": False, "reason": f"ill-posed ({len(names)} usable camera(s))"}
    spread = float(np.mean([np.linalg.norm(c_ref[a] - c_ref[b])
                            for i, a in enumerate(names) for b in names[i + 1:]])) if len(names) > 1 else 0.0
    for cam in new_cgroup.cameras:
        if cam.get_name() in names:
            apply_similarity_to_camera(cam, sim["s"], sim["R"], sim["t"])
    info = {"applied": True, "scale": sim["s"], "scale_inverse": 1.0 / sim["s"],
            "R": sim["R"].tolist(), "t": sim["t"].tolist(), "rms_mm": sim["rms"],
            "residual_mm": dict(zip(names, [float(x) for x in sim["residual"]])),
            "cameras": names, "partial": bool(sim.get("partial", False)),
            "rig_spread_mm": spread,
            "poorly_fit": bool(spread > 0 and sim["rms"] > warn_frac * spread)}
    return info


# ─────────────────────────────────────────────────────────────────────────────
#  Repairing already-triangulated 3D H5 files
# ─────────────────────────────────────────────────────────────────────────────

def transform_3d_h5(h5_in: Path, h5_out: Path, s: float, R: np.ndarray, t: np.ndarray) -> dict:
    """
    Apply X -> s R X + t to every bodypart of a CUBE `*_3d_filtered.h5`.
    Zero-filled (degenerate) and NaN points are left exactly as they are
    (they are placeholders, not positions); likelihood columns are untouched.
    Atomic write (.tmp.h5 then rename); h5_in may equal h5_out.
    """
    import pandas as pd
    df = pd.read_hdf(str(h5_in))
    scorer = df.columns.get_level_values(0)[0]
    bps = df[scorer].columns.get_level_values(0).unique().tolist()
    n_moved = n_kept = 0
    for bp in bps:
        X = np.stack([df[(scorer, bp, k)].values for k in ("x", "y", "z")], axis=-1).astype(np.float64)
        real = np.isfinite(X).all(-1) & ~np.all(np.abs(X) < 1e-6, axis=-1)
        Xn = X.copy()
        Xn[real] = apply_similarity_to_points(X[real], s, R, t)
        for i, k in enumerate(("x", "y", "z")):
            df[(scorer, bp, k)] = Xn[:, i]
        n_moved += int(real.sum()); n_kept += int((~real).sum())
    h5_out = Path(h5_out)
    tmp = h5_out.with_suffix(".tmp.h5")
    df.to_hdf(str(tmp), key="df_with_missing", mode="w", format="fixed")
    tmp.replace(h5_out)
    return {"points_moved": n_moved, "placeholders_kept": n_kept}


def floor_check(h5: Path) -> dict:
    """Paw-height summary of a 3D H5: in a floor-referenced frame the stance
    paws cluster near z = 0.  Reported, never used to change anything."""
    import pandas as pd
    df = pd.read_hdf(str(h5))
    sc = df.columns.get_level_values(0)[0]
    g = df[sc]
    paws = [b for b in g.columns.get_level_values(0).unique() if "paw" in b]
    z = []
    for b in paws:
        ok = g[(b, "likelihood")].values > 0.4
        z.append(g[(b, "z")].values[ok])
    z = np.concatenate(z) if z else np.array([np.nan])
    return {"paw_z_median": float(np.median(z)), "paw_z_p5": float(np.percentile(z, 5)),
            "paw_z_p95": float(np.percentile(z, 95))}


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("original_toml", help="metric, floor-referenced calibration.toml (the gauge reference)")
    ap.add_argument("refined_toml", help="calibration_refined.toml with the drifted scale/frame")
    ap.add_argument("--out-toml", default=None,
                    help="write the realigned calibration here (default: overwrite refined_toml after "
                         "copying it to <name>_UNSCALED_backup.toml)")
    ap.add_argument("--h5", nargs="*", default=[], help="3D H5 files triangulated with the refined calibration "
                    "(transformed in place; originals copied to --backup-dir)")
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    from aniposelib.cameras import CameraGroup
    import shutil
    ref = CameraGroup.load(a.original_toml)
    new = CameraGroup.load(a.refined_toml)
    info = realign_cgroup(ref, new)
    print(json.dumps({k: v for k, v in info.items() if k not in ("R", "t")}, indent=2))
    if not info["applied"]:
        return 1
    if a.dry_run:
        return 0
    refined = Path(a.refined_toml)
    out = Path(a.out_toml) if a.out_toml else refined
    if out == refined:
        shutil.copy2(refined, refined.with_name(refined.stem + "_UNSCALED_backup" + refined.suffix))
    tmp = out.with_name(out.name + ".tmp")
    new.dump(str(tmp)); tmp.replace(out)
    print("wrote", out)
    R, t, s = np.array(info["R"]), np.array(info["t"]), info["scale"]
    if a.h5:
        bdir = Path(a.backup_dir) if a.backup_dir else Path(a.h5[0]).parent.parent / "_pre_scale_fix_backup"
        for h in map(Path, a.h5):
            dest = bdir / h.parent.name
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / h.name).exists():
                # a backup exists => this H5 was already transformed; a second pass would
                # apply the similarity twice (the toml step is idempotent, the H5 step is not)
                print(h.name, "SKIPPED: already transformed (backup exists in", dest, ")")
                continue
            shutil.copy2(h, dest / h.name)
            before = floor_check(h)
            r = transform_3d_h5(h, h, s, R, t)
            print(h.name, r, "paw z before", {k: round(v, 1) for k, v in before.items()},
                  "after", {k: round(v, 1) for k, v in floor_check(h).items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
