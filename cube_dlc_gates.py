"""
Per-body-part plausibility gates for DeepLabCut output (Smart Adapt v4 validation, 2026-10-21).

Shared by two consumers so they cannot drift apart:
  * cube_core.filter_dlc_h5      -- cleans every DLC H5 CUBE produces (2D and 3D per-camera),
  * cube_adapt_pseudo.a1_flags   -- decides which base-model labels may be used as adaptation pseudo-labels.

Pure numpy/pandas; no DeepLabCut, no cube_core import (cube_core imports these names back, so existing call sites are unchanged).

Two gate levels are kept apart on purpose:
  gate        -- confidence at which a point is used to LEARN a body part's limits (default 0.4),
  apply_gate  -- confidence at which a point is CHECKED against those limits (scalar or per-body-part array; default = gate).
Points that are trusted downstream (filter: likelihood_thresh, pseudo-labels: the per-body-part threshold) but sit below `gate`
were previously never checked at all; apply_gate closes that gap without letting noisy points set the limits.
"""
from __future__ import annotations
import json as _json
from pathlib import Path
import numpy as np

# Head points: if two or more are flagged together in one frame, the remaining head points that also sit far from their own
# recent position are flagged too (a whole head detected on the wrong object, e.g. a person's face behind the cage).
_HEAD_GATE_NAMES = ("nose", "upper_jaw", "lower_jaw", "left_eye", "right_eye")


def _gate_at(gate, k):
    return float(gate[k]) if np.ndim(gate) else float(gate)


def _body_scale_px(all_x, all_y, all_ll, bps, gate) -> float:
    """Per-video body scale in pixels: median neck_base-back_end length when those exist, else 0.45 x the median diagonal of the
    confident points' bounding box (the two agree to within a few percent on SuperAnimal-Quadruped mice)."""
    try:
        if "neck_base" in bps and "back_end" in bps:
            a, b = bps.index("neck_base"), bps.index("back_end")
            ok = (all_ll[:, a] >= gate) & (all_ll[:, b] >= gate)
            if ok.sum() > 50:
                return float(np.nanmedian(np.hypot(all_x[ok, a] - all_x[ok, b], all_y[ok, a] - all_y[ok, b])))
        conf = all_ll >= gate
        xs = np.where(conf, all_x, np.nan); ys = np.where(conf, all_y, np.nan)
        with np.errstate(all="ignore"):
            diag = np.hypot(np.nanmax(xs, 1) - np.nanmin(xs, 1), np.nanmax(ys, 1) - np.nanmin(ys, 1))
        return 0.45 * float(np.nanmedian(diag))
    except Exception:
        return float("nan")


def _temporal_gate_flags(all_x, all_y, all_ll, bps, torso_px, gate=0.4, half_window=8, iters=3,
                         pct=98.0, mult=1.3, floor_frac=0.15, apply_gate=None, fps=None):
    """
    Per-body-part temporal plausibility gate. A point is flagged when it lies further than its OWN learned limit from that body
    part's median position over the surrounding +-half_window frames (points above `gate` only form the reference; the reference is
    recomputed `iters` times without the flagged points so a bad stretch cannot drag its own reference).
    limit = max(floor_frac x body scale, mult x the pct-th percentile of that body part's deviations in this video), learned from
    points >= `gate`; points >= `apply_gate` are checked against it. Catches the detector flipping a point between the animal and
    something else. `half_window` is defined at 30 fps; with `fps` it is rescaled so the window keeps the same duration.
    Returns (n, nbp) bool.
    """
    import pandas as _pd
    if apply_gate is None:
        apply_gate = gate
    if fps and np.isfinite(fps) and fps > 0:
        half_window = max(3, int(round(half_window * float(fps) / 30.0)))
    n, nbp = all_x.shape
    flag = np.zeros((n, nbp), bool)
    lim = np.full(nbp, np.nan)
    dev_all = np.full((n, nbp), np.nan)
    w = 2 * half_window + 1

    def _ref(v, ok):
        return _pd.Series(np.where(ok, v, np.nan)).rolling(w, center=True, min_periods=5).median().values

    for k in range(nbp):
        conf = all_ll[:, k] >= _gate_at(gate, k)
        if conf.sum() < 200:
            continue
        ok = conf.copy()
        d = None
        for _ in range(iters):
            d = np.hypot(all_x[:, k] - _ref(all_x[:, k], ok), all_y[:, k] - _ref(all_y[:, k], ok)) / torso_px
            c = conf & np.isfinite(d)
            if c.sum() < 100 or (c & ok).sum() <= 50:
                break
            lim[k] = max(floor_frac, mult * float(np.percentile(d[c & ok], pct)))
            ok = conf & ~(c & (d > lim[k]))
        if d is None or not np.isfinite(lim[k]):
            continue
        dev_all[:, k] = d
        flag[:, k] = (all_ll[:, k] >= _gate_at(apply_gate, k)) & np.isfinite(d) & (d > lim[k])
    hk = [bps.index(h) for h in _HEAD_GATE_NAMES if h in bps]
    if len(hk) >= 2:
        two = flag[:, hk].sum(1) >= 2
        for k in hk:
            if np.isfinite(lim[k]):
                flag[:, k] |= two & (all_ll[:, k] >= _gate_at(apply_gate, k)) & np.isfinite(dev_all[:, k]) & (dev_all[:, k] > 0.5 * lim[k])
    return flag


def _find_bbox_json(h5_path):
    """DLC's per-frame detector boxes saved beside the pose H5 (video_inference_superanimal writes <stem>_before_adapt.json).
    Note this is the pre-adaptation detector pass; after detector adaptation the boxes can differ slightly, which the learned margin absorbs."""
    try:
        h5_path = Path(h5_path)
        cand = h5_path.with_name(h5_path.stem + "_before_adapt.json")
        if cand.is_file():
            return cand
        for p in sorted(h5_path.parent.glob(h5_path.stem + "*.json")):
            return p
    except Exception:
        pass
    return None


def boxes_from_json(J, n):
    """(n, 4) array of [x, y, w, h] detector boxes (the best-scoring one per frame; NaN where a frame has none)."""
    B = np.full((n, 4), np.nan)
    for i in range(min(n, len(J))):
        bb, sc = J[i].get("bboxes", []), J[i].get("bbox_scores", [])
        if len(bb):
            B[i] = bb[int(np.argmax(sc)) if len(sc) else 0][:4]
    return B


def _bbox_gate_flags(h5_path, all_x, all_y, all_ll, torso_px, gate=0.4, pct=99.0, mult=1.3, floor_frac=0.03,
                     apply_gate=None, boxes=None):
    """
    Detector-box gate: a point lying outside DLC's own bounding box of the animal (by more than that body part's learned margin:
    mult x the pct-th percentile of its outside-distance over points >= `gate`, floor floor_frac x body scale) is flagged if it is
    >= `apply_gate`. Independent of every other keypoint, so it still works when a whole group of points (e.g. the head) has gone to
    the wrong object together. Boxes come from `boxes` ((n,4) x,y,w,h) or from the detector json beside the H5; without either
    nothing is flagged.
    """
    if apply_gate is None:
        apply_gate = gate
    n, nbp = all_x.shape
    flag = np.zeros((n, nbp), bool)
    if boxes is None:
        js = _find_bbox_json(h5_path)
        if js is None:
            return flag
        try:
            with open(js) as fh:
                boxes = boxes_from_json(_json.load(fh), n)
        except Exception:
            return flag
    B = np.asarray(boxes, float)
    l, t, r, b_ = B[:, 0], B[:, 1], B[:, 0] + B[:, 2], B[:, 1] + B[:, 3]
    for k in range(nbp):
        e = np.maximum.reduce([l - all_x[:, k], all_x[:, k] - r, t - all_y[:, k], all_y[:, k] - b_, np.zeros(n)])
        c = (all_ll[:, k] >= _gate_at(gate, k)) & np.isfinite(e)
        if c.sum() < 200:
            continue
        lim = max(floor_frac * torso_px, mult * float(np.percentile(e[c], pct)))
        flag[:, k] = (all_ll[:, k] >= _gate_at(apply_gate, k)) & np.isfinite(e) & (e > lim)
    return flag


# ----------------------------------------------------------------------------- confidence of bridged (gap-filled) points
def bridge_fill_variance(x, y, ll, good, filled, local_win=10, min_pairs=3):
    """
    Per-axis error variance (px^2) of linearly bridged points. A gap of g frames between two confident anchors is modelled as a Brownian
    bridge; the position error at step k is
        V_k = s^2 * k (g+1-k) / (g+1)   +   w0^2 * a_L^2   +   w1^2 * a_R^2,      w0 = (g+1-k)/(g+1), w1 = k/(g+1)
    s^2 = per-axis variance of the point's one-frame step, from the confident frames around the gap (local, floored at a quarter of the
    whole-video value so a locally still point is not over-trusted), winsorised at the 99th percentile.
    a^2 = per-axis measurement noise of an anchor, learned from the video: residual of confident points from their 5-frame median, in
    three likelihood bands (a low-likelihood anchor is a noisier anchor).
    Mouse motion is ballistic, so a random-walk bridge is used rather than a smooth-motion model.
    Returns an (n,) array: V on `filled` frames, NaN elsewhere (all NaN when there is too little data to estimate the model).
    """
    import pandas as _pd
    x = np.asarray(x, float); y = np.asarray(y, float); ll = np.asarray(ll, float)
    good = np.asarray(good, bool); filled = np.asarray(filled, bool)
    n = len(x); out = np.full(n, np.nan)
    if n < 20 or not filled.any():
        return out
    pair = good[1:] & good[:-1] & np.isfinite(x[1:]) & np.isfinite(x[:-1]) & np.isfinite(y[1:]) & np.isfinite(y[:-1])
    d2 = np.full(n - 1, np.nan)
    d2[pair] = np.diff(x)[pair] ** 2 + np.diff(y)[pair] ** 2
    if np.isfinite(d2).sum() < 10:
        return out
    d2w = np.minimum(d2, np.nanpercentile(d2, 99.0))                     # NaN stays NaN
    s2_glob = float(np.nanmean(d2w)) / 2.0

    xs = _pd.Series(np.where(good, x, np.nan)); ys = _pd.Series(np.where(good, y, np.nan))
    e2 = (xs - xs.rolling(5, center=True, min_periods=3).median()) ** 2 + (ys - ys.rolling(5, center=True, min_periods=3).median()) ** 2
    e2 = e2.to_numpy(); ok = good & np.isfinite(e2)
    if ok.sum() < 30:
        return out
    chi2_2_median = 2.0 * np.log(2.0)                                    # median of a chi-square(2) variable
    a2_all = max(float(np.median(e2[ok])) / chi2_2_median, 0.25)
    edges = (0.5, 0.75)
    band = np.digitize(np.where(np.isfinite(ll), ll, 0.0), edges)
    a2_band = []
    for b in range(len(edges) + 1):
        m = ok & (band == b)
        a2_band.append(max(float(np.median(e2[m])) / chi2_2_median, 0.25) if m.sum() >= 30 else a2_all)

    idx = np.flatnonzero(filled)
    starts = idx[np.r_[True, np.diff(idx) > 1]]
    ends = idx[np.r_[np.diff(idx) > 1, True]] + 1                         # exclusive
    for i, j in zip(starts, ends):
        if i < 1 or j >= n:
            continue
        g = j - i
        lo, hi = max(0, i - 1 - local_win), min(n - 1, j + local_win)
        seg = d2w[lo:hi]; seg = seg[np.isfinite(seg)]
        s2 = max(float(seg.mean()) / 2.0, 0.25 * s2_glob) if len(seg) >= min_pairs else s2_glob
        aL = a2_band[int(band[i - 1])]; aR = a2_band[int(band[j])]
        k = np.arange(1, g + 1, dtype=float)
        w0 = (g + 1 - k) / (g + 1); w1 = k / (g + 1)
        out[i:j] = np.maximum(s2 * k * (g + 1 - k) / (g + 1) + w0 ** 2 * aL + w1 ** 2 * aR, 1e-6)
    return out


# Heavy-tailed error law of a bridged point. With a Gaussian error the radial error r satisfies r^2/(2V) ~ Exp(1), but real bridge errors
# are mostly tiny with occasional large ones (a scale mixture of Gaussians), so P(r^2/(2V) > u) = (1 + u/THETA)^-ALPHA (Lomax). The two
# constants were fitted on TMZ7 cam0 (two videos) and checked on a held-out video; see ADAPT_LABEL_TESTS/testbed/fill_confidence_validation.py.
BRIDGE_LOMAX_ALPHA = 1.0739
BRIDGE_LOMAX_THETA = 0.2857


def bridge_confidence_from_variance(V, torso_px, tol_frac=0.05, alpha=None, theta=None):
    """P(radial error < tol_frac x body scale) for per-axis variance(s) V under the heavy-tailed law above (NaN stays NaN)."""
    alpha = BRIDGE_LOMAX_ALPHA if alpha is None else alpha
    theta = BRIDGE_LOMAX_THETA if theta is None else theta
    V = np.asarray(V, float); torso = np.asarray(torso_px, float)
    with np.errstate(all="ignore"):
        u = (float(tol_frac) * torso) ** 2 / (2.0 * V)
        c = 1.0 - (1.0 + u / theta) ** (-alpha)
    return np.where(np.isfinite(torso) & (torso > 0), c, np.nan)


def bridge_fill_confidence(x, y, ll, good, filled, torso_px, tol_frac=0.05, local_win=10, min_pairs=3):
    """Probability that each bridged point lies within `tol_frac` x body scale of where the animal's point really was (NaN off `filled`)."""
    V = bridge_fill_variance(x, y, ll, good, filled, local_win=local_win, min_pairs=min_pairs)
    return bridge_confidence_from_variance(V, torso_px, tol_frac)
