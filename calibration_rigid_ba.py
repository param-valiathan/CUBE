"""Rigid-board bundle adjustment with PER-CAMERA lens parameters  (CUBE; Param Valiathan, laboratory of Patrik Ernfors, Karolinska Institutet).

Why: aniposelib's bundle adjustment treats every triangulated corner as a FREE 3D point, so the strongest knowledge we have - the Charuco board is a RIGID ruler
of known size - is not used, and the lens model is shared by all six cameras (a pooled model fitted from ~100 views).  Here the unknowns are
    per camera : pose (6), focal length f (fx = fy), radial distortion k1, k2, principal point (cx, cy)          = 11
    per frame  : ONE rigid board pose (6)  - the 20 corners follow from the known board geometry (12.22 mm squares, z = 0 in the board frame)
and the residuals are the detected corner positions (pixels).  Weakly identified lens parameters are held by soft priors (Tikhonov) towards the starting
(pooled) model, so a camera that covers only part of the image cannot run away (the failure that made independent per-camera calibration diverge:
9262 / 838 / 3091 / 2798 px), and camera 0's pose is pinned (gauge: the world frame stays the floor frame of the starting calibration).
Robust loss (soft_l1) against bad detections; analytic sparsity (every residual touches one camera and one frame) -> fast, one thread.

  refine(toml_in, dets, grid, obj_mm, names, size=(1280, 720), ...) -> dict(result)    then  write_toml(toml_in, result, toml_out)
dets[name][k] = (corner ids, pixel positions) or None for sampled moment k (calibration_report._detections_move).  Only moments seen by >= 3 cameras that agree on
the board's FACE (double-sided board) are used.
"""
import time

import numpy as np


def _rodrigues(rv):
    """(N,3) rotation vectors -> (N,3,3)"""
    th = np.linalg.norm(rv, axis=1)
    th = np.maximum(th, 1e-12)
    k = rv / th[:, None]
    K = np.zeros((len(rv), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 2], k[:, 1], k[:, 2], -k[:, 0], -k[:, 1], k[:, 0]
    c, s = np.cos(th)[:, None, None], np.sin(th)[:, None, None]
    return np.eye(3)[None] * c + s * K + (1 - c) * (k[:, :, None] * k[:, None, :])


def _project(Xc, f, k1, k2, cx, cy):
    z = Xc[:, 2]
    x, y = Xc[:, 0] / z, Xc[:, 1] / z
    r2 = x * x + y * y
    d = 1 + k1 * r2 + k2 * r2 * r2
    return np.stack([f * x * d + cx, f * y * d + cy], axis=1)


def _cam_rt(cam):
    import cv2
    return cv2.Rodrigues(np.asarray(cam.get_rotation(), dtype=np.float64))[0], np.asarray(cam.get_translation(), dtype=np.float64).ravel()


def refine(toml_in, dets, grid, obj_mm, names, size=(1280, 720), moments=None, sig_f=20.0, sig_k=0.02, sig_c=10.0, max_nfev=60, loss_scale=1.5,
           same_face_dot=0.3, min_corners=8, agree_px=12.0, log=print):
    import cv2
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from aniposelib.cameras import CameraGroup
    cg = CameraGroup.load(str(toml_in))
    nc = len(names)
    assert cg.get_names() == list(names), "camera order of the calibration and of the detections differ"
    R0, t0, K0, D0 = [], [], [], []
    for cam in cg.cameras:
        R, t = _cam_rt(cam); R0.append(R); t0.append(t)
        K0.append(np.asarray(cam.get_camera_matrix(), dtype=np.float64)); D0.append(np.asarray(cam.get_distortions(), dtype=np.float64).ravel())
    obj = np.asarray(obj_mm, dtype=np.float64)

    # ---- frames: >= 3 cameras on the SAME board face, initial board pose (world) from the camera with most corners, kept only if the others agree
    obs_cam, obs_frame, obs_id, obs_px = [], [], [], []
    pose0 = []
    rej = dict(few_cameras=0, no_same_face_group=0, disagree=0)
    idx_moments = range(len(grid)) if moments is None else moments
    for k in idx_moments:
        seen, nrm, pn = [], {}, {}
        for ci, n in enumerate(names):
            d = dets[n][k]
            if d is None or len(d[0]) < min_corners:
                continue
            ids, pts = d
            ok, rv, tv = cv2.solvePnP(obj[ids].astype(np.float32), pts.astype(np.float32), K0[ci], D0[ci],
                                      flags=cv2.SOLVEPNP_IPPE if len(ids) < 8 else cv2.SOLVEPNP_ITERATIVE)
            if ok:
                Rb = cv2.Rodrigues(rv)[0]
                seen.append(ci); nrm[ci] = R0[ci].T @ Rb[:, 2]; pn[ci] = (Rb, tv.ravel())
        if len(seen) < 3:
            rej["few_cameras"] += 1
            continue
        best = max(([m for m in seen if float(nrm[s] @ nrm[m]) > same_face_dot] for s in seen), key=len)
        if len(best) < 3:
            rej["no_same_face_group"] += 1
            continue
        # initial board pose: rigid (Kabsch) fit of the board to the TRIANGULATED corners of the group - consistent across cameras even when a camera's
        # focal length is off (a single camera's solvePnP would be wrong in depth by the focal error, ~8 mm for 3 %, and the others would disagree)
        p2d = np.full((nc, 20, 2), np.nan)
        for ci in best:
            ids, pts = dets[names[ci]][k]
            p2d[ci, ids] = pts
        P3 = cg.triangulate(p2d, progress=False)
        okc = np.isfinite(P3).all(axis=1) & ((~np.isnan(p2d[:, :, 0])).sum(axis=0) >= 2)
        if okc.sum() < 6:
            rej["disagree"] += 1
            continue
        oc_, pc_ = obj[okc].mean(axis=0), P3[okc].mean(axis=0)
        U, _, Vt = np.linalg.svd((obj[okc] - oc_).T @ (P3[okc] - pc_))
        Rw = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        tw = pc_ - Rw @ oc_                                                          # board -> world
        # agreement check: reproject the initial board pose into the other group cameras
        bad = False
        for ci in best:
            ids, pts = dets[names[ci]][k]
            Xc = (R0[ci] @ (Rw @ obj[ids].T + tw[:, None])).T + t0[ci]
            pr, _ = cv2.projectPoints(Xc.astype(np.float64), np.zeros(3), np.zeros(3), K0[ci], D0[ci])
            if np.median(np.linalg.norm(pr.reshape(-1, 2) - pts, axis=1)) > agree_px:
                bad = True
        if bad:
            rej["disagree"] += 1
            continue
        fi = len(pose0)
        pose0.append(np.concatenate([cv2.Rodrigues(Rw)[0].ravel(), tw]))
        for ci in best:
            ids, pts = dets[names[ci]][k]
            for i_, p_ in zip(ids, pts):
                obs_cam.append(ci); obs_frame.append(fi); obs_id.append(int(i_)); obs_px.append(p_)
    nf = len(pose0)
    log(f"[rigid-ba] moments: {nf} usable; rejected: {rej}")
    if nf < 20:
        raise RuntimeError(f"only {nf} usable frames (>= 3 cameras on one board face) - record more of the move take ({rej})")
    oc, of, oid, opx = np.array(obs_cam), np.array(obs_frame), np.array(obs_id), np.array(obs_px, dtype=np.float64)
    nobs = len(oc)
    log(f"[rigid-ba] {nf} frames, {nobs} corner observations from {nc} cameras")

    # ---- parameters
    PC = 11
    x0 = np.zeros(nc * PC + nf * 6)
    for ci in range(nc):
        rv = cv2.Rodrigues(R0[ci])[0].ravel()
        x0[ci * PC:(ci + 1) * PC] = [*rv, *t0[ci], K0[ci][0, 0], D0[ci][0], D0[ci][1], K0[ci][0, 2], K0[ci][1, 2]]
    x0[nc * PC:] = np.concatenate(pose0)
    prior_idx = np.array([ci * PC + j for ci in range(nc) for j in (6, 7, 8, 9, 10)])
    prior_sig = np.array([s for ci in range(nc) for s in (sig_f, sig_k, sig_k, sig_c, sig_c)])
    prior_x0 = x0[prior_idx].copy()
    pin_idx = np.arange(6)                                                         # camera 0's pose is the gauge
    pin_sig = np.array([1e-4] * 3 + [1e-2] * 3)
    pin_x0 = x0[pin_idx].copy()
    objb = obj[oid]

    def unpack(x):
        C = x[:nc * PC].reshape(nc, PC)
        F = x[nc * PC:].reshape(nf, 6)
        return C, F

    def fun(x):
        C, F = unpack(x)
        Rb = _rodrigues(F[:, :3])[of]                                              # (nobs,3,3) board -> world
        Xw = np.einsum("nij,nj->ni", Rb, objb) + F[of, 3:]
        Rc = _rodrigues(C[:, :3])[oc]
        Xc = np.einsum("nij,nj->ni", Rc, Xw) + C[oc, 3:6]
        pr = _project(Xc, C[oc, 6], C[oc, 7], C[oc, 8], C[oc, 9], C[oc, 10])
        res = (pr - opx).ravel()
        return np.concatenate([res, (x[prior_idx] - prior_x0) / prior_sig, (x[pin_idx] - pin_x0) / pin_sig])

    nres = 2 * nobs + len(prior_idx) + len(pin_idx)
    A = lil_matrix((nres, len(x0)), dtype=np.int8)
    rows = np.arange(nobs)
    for j in range(PC):
        A[2 * rows, oc * PC + j] = 1; A[2 * rows + 1, oc * PC + j] = 1
    for j in range(6):
        A[2 * rows, nc * PC + of * 6 + j] = 1; A[2 * rows + 1, nc * PC + of * 6 + j] = 1
    for q, i in enumerate(prior_idx):
        A[2 * nobs + q, i] = 1
    for q, i in enumerate(pin_idx):
        A[2 * nobs + len(prior_idx) + q, i] = 1
    lo = np.full(len(x0), -np.inf); hi = np.full(len(x0), np.inf)
    for ci in range(nc):
        b = ci * PC
        lo[b + 6], hi[b + 6] = 0.7 * K0[ci][0, 0], 1.3 * K0[ci][0, 0]
        lo[b + 7], hi[b + 7], lo[b + 8], hi[b + 8] = -0.5, 0.3, -0.5, 0.5
        lo[b + 9], hi[b + 9], lo[b + 10], hi[b + 10] = size[0] / 2 - 120, size[0] / 2 + 120, size[1] / 2 - 90, size[1] / 2 + 90
    r0 = fun(x0)
    rms0 = float(np.sqrt(np.mean(r0[:2 * nobs] ** 2)))
    t0_ = time.time()
    sol = least_squares(fun, x0, jac_sparsity=A.tocsr(), bounds=(lo, hi), loss="soft_l1", f_scale=loss_scale, x_scale="jac", method="trf", max_nfev=max_nfev, verbose=0)
    r1 = fun(sol.x)
    rms1 = float(np.sqrt(np.mean(r1[:2 * nobs] ** 2)))
    med1 = float(np.median(np.linalg.norm(r1[:2 * nobs].reshape(-1, 2), axis=1)))
    C, F = unpack(sol.x)
    log(f"[rigid-ba] reprojection RMS {rms0:.3f} -> {rms1:.3f} px (median {med1:.3f} px) in {time.time() - t0_:.0f} s, {sol.nfev} evaluations, status {sol.status}")
    per_cam = {}
    for ci, n in enumerate(names):
        m = oc == ci
        e = np.linalg.norm(r1[:2 * nobs].reshape(-1, 2)[m], axis=1)
        per_cam[n] = dict(f=float(C[ci, 6]), f0=float(K0[ci][0, 0]), k1=float(C[ci, 7]), k2=float(C[ci, 8]), cx=float(C[ci, 9]), cy=float(C[ci, 10]),
                          median_px=float(np.median(e)), n=int(m.sum()))
        log(f"[rigid-ba]   {n}: f {K0[ci][0, 0]:.1f} -> {C[ci, 6]:.1f}  k1 {D0[ci][0]:+.3f} -> {C[ci, 7]:+.3f}  k2 {C[ci, 8]:+.3f}  centre ({C[ci, 9]:.0f},{C[ci, 10]:.0f})  median {per_cam[n]['median_px']:.2f} px")
    return dict(C=C, frames=nf, obs=nobs, rms0=rms0, rms1=rms1, median_px=med1, per_cam=per_cam, names=list(names))


def write_toml(toml_in, result, toml_out):
    import cv2
    from aniposelib.cameras import CameraGroup
    cg = CameraGroup.load(str(toml_in))
    for ci, cam in enumerate(cg.cameras):
        c = result["C"][ci]
        cam.set_camera_matrix(np.array([[c[6], 0, c[9]], [0, c[6], c[10]], [0, 0, 1.0]]))
        cam.set_distortions(np.array([c[7], c[8], 0.0, 0.0, 0.0]))
        cam.set_rotation(c[:3].copy())
        cam.set_translation(c[3:6].copy())
    cg.dump(str(toml_out))
    return toml_out
