"""
cube_skeleton_render.py -- CUBE's standard 3D mouse skeleton video (GPU drawn).

Replaces the matplotlib/mplot3d renderer for the default skeleton video.  Design goals
(see CLAUDE.md "Standard skeleton video"):

* publication style: ONLY tracked keypoints (plus clearly marked schematic parts), no tail,
  light grid floor + walls so the 3D scene reads, light floor shadow, region colours.
* never crashes the machine / never slow:  no matplotlib in the frame loop, no mplot3d, no BLAS
  heavy lifting.  Every frame is rasterised on the GPU with torch (analytic anti-aliased
  line / circle / polygon coverage, 2x supersampled) and piped as raw RGB straight into the
  NVENC encoder (libx264 fallback).  `render_skeleton_video_isolated()` runs it in its own
  process with a timeout and capped CPU threads, so a native failure cannot take the pipeline
  down; the caller falls back to the legacy renderer.
* theme aware: "light" / "dark" (CUBE's engine_cfg.plot_theme) change background, arena, text and
  ink colours; every style choice is a session setting (dlc_3d_skel_*).
* layout "obl_top_side": large oblique view + top view + side view.  The numbered floor corners
  (1-4) are the same in every view; the side view says which edge it is looking from and the top
  view shows the side-view camera.  The arena box is built from everything that is drawn, so
  nothing is ever cut off.
* gaps up to `fill_s` seconds (default 0.5 s) are bridged by interpolation FOR THIS VIDEO ONLY
  (the H5 on disk is untouched); bridged points are drawn hollow / faded.

Standalone:  python cube_skeleton_render.py <3d_filtered.h5> <out.mp4|out.png> [--theme dark]
             [--fps 30] [--frames 100:160] [--still 450] [--height 720]
"""
from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

RENDERER_VERSION = "gpu-1"    # bump when the picture changes: videos made by an older version are re-rendered
GATE = 0.4                    # = cube_core.DISPLAY_LL_THRESHOLD (fills are capped below this)
SS = 2                        # supersampling factor (anti-aliasing)
PT = 100.0 / 72.0             # points -> pixels at 1080p; scaled by height/1080

USED = ["nose", "upper_jaw", "lower_jaw", "left_eye", "right_eye", "left_earend", "right_earend",
        "neck_base", "back_end",
        "front_left_thai", "front_right_thai", "front_left_paw", "front_right_paw",
        "back_left_thai", "back_right_thai", "back_left_knee", "back_right_knee",
        "back_left_paw", "back_right_paw"]
LIMBS = [("neck_base", "front_left_thai"), ("front_left_thai", "front_left_paw"),
         ("neck_base", "front_right_thai"), ("front_right_thai", "front_right_paw"),
         ("back_end", "back_left_thai"), ("back_left_thai", "back_left_knee"), ("back_left_knee", "back_left_paw"),
         ("back_end", "back_right_thai"), ("back_right_thai", "back_right_knee"), ("back_right_knee", "back_right_paw")]
HEAD_NAMES = ("nose", "lower_jaw", "upper_jaw", "left_eye", "right_eye", "left_earend", "right_earend")

DEFAULT_STYLE = dict(
    scheme="region",          # region | region_cb | sides | ramp | teal | mono
    line="medium",            # thin | medium | bold
    spine="curved",           # curved | tapered | straight
    shadow=True,              # light floor shadow of the skeleton (height cue)
    far_limbs="light",        # light | dashed | off  (limbs on the far side of the body, per view)
    head="contrast",          # contrast | filled | outline | tracked
    fill_s=0.5,               # bridge dropouts up to this many seconds (this video only)
    layout="obl_top_side",    # obl_top_side | oblique | two_oblique
    height=1080,              # output height in pixels (width is 16:9)
)
LINE_PT = {"thin": (1.3, 28), "medium": (2.2, 40), "bold": (3.4, 58)}      # (line pt, marker pt^2)

THEMES = {
    "light": dict(bg="#f5f6fa", text="#1a1a2e", tick="#444466", floor="#f2f4f7", wall="#fbfcfd",
                  edge="#c3cad4", grid="#dde2e9", muted="#777777", ink="#5A6472", paper="#ffffff",
                  shadow=0.11),
    "dark": dict(bg="#0d0d1a", text="#ffffff", tick="#aaaacc", floor="#1d252e", wall="#171d24",
                 edge="#46515e", grid="#2a343f", muted="#9a9a9a", ink="#c4cad2", paper="#0d0d1a",
                 shadow=0.45),
}


def style_from_session(session) -> dict:
    """Read the dlc_3d_skel_* session settings (missing keys -> defaults)."""
    s = dict(DEFAULT_STYLE)
    for k in DEFAULT_STYLE:
        v = session.get("dlc_3d_skel_" + k, None) if hasattr(session, "get") else None
        if v is not None:
            s[k] = v
    return s


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _mix(h, t, to):
    c = _rgb(h)
    return tuple(v * (1 - t) + (to / 255.0) * t for v in c)


def _side(b):
    return "L" if b.startswith(("left", "front_left", "back_left")) else ("R" if "right" in b else "M")


def _region(b):
    if b in HEAD_NAMES:
        return "head"
    if b in ("neck_base", "back_end"):
        return "spine"
    return "fore" if b.startswith("front") else "hind"


def colour_fn(scheme, dark):
    def lift(c):
        return tuple(v + (1 - v) * 0.3 for v in c) if dark else c
    if scheme in ("region", "region_cb"):
        m = ({"head": "#7B3294", "spine": "#5B6B7C", "fore": "#1B9E77", "hind": "#E08214"} if scheme == "region"
             else {"head": "#CC79A7", "spine": "#6E6E6E", "fore": "#0072B2", "hind": "#D55E00"})

        def f(b):
            c = _rgb(m[_region(b)])
            if "paw" in b:
                c = tuple(v * 0.62 for v in c)
            if _side(b) == "R":
                c = tuple(v * 0.8 + 0.2 for v in c)
            return lift(c)
        return f
    if scheme == "sides":
        warm, cool, mid = "#F08C2B", "#2B87D1", "#5B6B7C"
        def f(b):
            c = _rgb({"L": warm, "R": cool, "M": mid}[_side(b)])
            if "paw" in b:
                c = tuple(v * 0.7 for v in c)
            return lift(c)
        return f
    if scheme == "ramp":
        import matplotlib
        cmap = matplotlib.colormaps["viridis"]
        pos = {"nose": 0.0, "lower_jaw": 0.0, "left_eye": .07, "right_eye": .07, "left_earend": .12, "right_earend": .12,
               "neck_base": .30, "front_left_thai": .38, "front_right_thai": .38, "front_left_paw": .5,
               "front_right_paw": .5, "back_left_thai": .78, "back_right_thai": .78, "back_left_knee": .85,
               "back_right_knee": .85, "back_left_paw": .93, "back_right_paw": .93, "back_end": 1.0}
        return lambda b: tuple(cmap(1 - pos.get(b, .5))[:3])
    if scheme == "teal":
        sh = {"head": .55, "spine": .75, "fore": .9, "hind": .9}
        def f(b):
            k = sh[_region(b)] * (0.65 if "paw" in b else 1.0)
            c = np.array([0.0, 0.52, 0.55]) * k
            return tuple(np.clip(c + (0.25 * (1 - k) if dark else 0), 0, 1))
        return f
    g = (0.88,) * 3 if dark else (0.11,) * 3                      # mono
    return lambda b: g


# --------------------------------------------------------------------------- data
class SkelData:
    """3D H5 -> display arrays (gate, short-gap interpolation, display smoothing, floor at z = 0)."""

    def __init__(self, h5, fps, gate=GATE, fill_s=0.5):
        import pandas as pd
        from scipy.ndimage import median_filter, gaussian_filter1d
        df = pd.read_hdf(h5)
        sc = df.columns.get_level_values(0)[0]
        self.bps = df[sc].columns.get_level_values(0).unique().tolist()
        g = lambda c: df[sc].xs(c, level="coords", axis=1)[self.bps].values
        for need in ("nose", "neck_base", "back_end"):
            if need not in self.bps:
                raise ValueError(f"skeleton renderer needs body part '{need}' (mouse SuperAnimal schema)")
        XYZ = np.stack([g("x"), g("y"), g("z")], -1).astype(float)
        self.LL = np.nan_to_num(g("likelihood").astype(float))
        self.N, self.fps = len(df), float(fps)
        self.ix = {b: i for i, b in enumerate(self.bps)}
        max_gap = max(1, int(round(fill_s * fps)))
        self.max_gap = max_gap
        valid = (self.LL >= gate) & np.isfinite(XYZ).all(-1)
        self.tracked_frac = valid.mean(0)
        P = XYZ.copy()
        vis = valid.copy()
        idx = np.arange(self.N)
        for j in range(P.shape[1]):
            v = np.flatnonzero(valid[:, j])
            if v.size < 2:
                vis[:, j] = False
                continue
            for c in range(3):
                P[:, j, c] = np.interp(idx, v, XYZ[v, j, c])
            prev = np.maximum.accumulate(np.where(valid[:, j], idx, -10 ** 9))
            nxt = np.minimum.accumulate(np.where(valid[:, j], idx, 10 ** 9)[::-1])[::-1]
            ok = valid[:, j] | ((nxt - prev) <= max_gap + 1)
            vis[:, j] = ok & (prev > -10 ** 8) & (nxt < 10 ** 8)
        self.filled = vis & ~valid                       # bridged points (drawn hollow / faded)
        for b in USED:
            if b not in self.ix:
                continue
            j = self.ix[b]
            half = 2 if "paw" in b else (4 if any(k in b for k in ("eye", "ear", "nose", "jaw")) else 7)
            P[:, j] = median_filter(P[:, j], size=(2 * half + 1, 1), mode="nearest")
        paws = [b for b in ("back_left_paw", "back_right_paw") if b in self.ix] or \
               [b for b in USED if "paw" in b and b in self.ix]
        pz = np.concatenate([XYZ[valid[:, self.ix[b]], self.ix[b], 2] for b in paws]) if paws else np.array([0.0])
        if pz.size < 20:
            pz = XYZ[valid[:, self.ix["back_end"]], self.ix["back_end"], 2]
        lo, hi = np.percentile(pz, 0.5), np.percentile(pz, 60)
        cnt, ed = np.histogram(pz, bins=np.arange(lo, max(hi, lo + 2.0), 1.0))
        self.floor = float((ed[np.argmax(cnt)] + ed[np.argmax(cnt) + 1]) / 2) if cnt.size else float(lo)
        P[:, :, 2] -= self.floor
        self.P, self.vis = P, vis
        self._frames(median_filter, gaussian_filter1d)
        self.tip = "lower_jaw" if ("lower_jaw" in self.ix and
                                   self.tracked_frac[self.ix["lower_jaw"]] > self.tracked_frac[self.ix["nose"]]) else "nose"

    def pt(self, f, b):
        j = self.ix.get(b)
        return None if (j is None or not self.vis[f, j]) else self.P[f, j]

    def is_filled(self, f, b):
        j = self.ix.get(b)
        return bool(j is not None and self.filled[f, j])

    def _frames(self, median_filter, gaussian_filter1d):
        N = self.N
        rump, neck = self.P[:, self.ix["back_end"]], self.P[:, self.ix["neck_base"]]
        sp = neck - rump
        sp = np.stack([gaussian_filter1d(sp[:, c], 2.0) for c in range(3)], 1)
        fwd = sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-9)
        fb_left = np.cross(np.array([0, 0, 1.0]), fwd)
        pairs = [("left_eye", "right_eye"), ("left_earend", "right_earend"), ("front_left_thai", "front_right_thai"),
                 ("back_left_thai", "back_right_thai"), ("back_left_paw", "back_right_paw")]
        lat = np.full((N, 3), np.nan)
        for f in range(N):
            acc = []
            for a, b in pairs:
                pa, pb = self.pt(f, a), self.pt(f, b)
                if pa is not None and pb is not None:
                    v = pa - pb
                    v = v - fwd[f] * v.dot(fwd[f])
                    n = np.linalg.norm(v)
                    if n > 3:
                        acc.append(v / n)
            if acc:
                lat[f] = np.mean(acc, 0)
        ok = ~np.isnan(lat[:, 0])
        agree = float((np.sum(lat[ok] * fb_left[ok], 1) > 0).mean()) if ok.any() else 1.0
        fb_left = fb_left * (-1.0 if agree < 0.5 else 1.0)
        for f in range(N):
            if not ok[f] or np.dot(lat[f], fb_left[f]) < 0.2:
                lat[f] = fb_left[f]
        lat = np.stack([median_filter(lat[:, c], size=9, mode="nearest") for c in range(3)], 1)
        lat -= fwd * np.sum(lat * fwd, 1, keepdims=True)
        lat /= np.linalg.norm(lat, axis=1, keepdims=True) + 1e-9
        self.fwd, self.lat = fwd, lat
        c = (rump + neck) / 2
        self.cent = np.stack([gaussian_filter1d(c[:, k], 12) for k in range(3)], 1)
        self.L = float(np.median(np.linalg.norm(sp, axis=1)))


# --------------------------------------------------------------------------- geometry
def _kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def head_model(D):
    """ONE fixed head shape (median rigid pose of the tracked head points, plus a smooth outline and
    ear leaves in that frame).  None if the head points are not tracked together often enough."""
    if hasattr(D, "_hm"):
        return D._hm
    D._hm = None
    names = [D.tip, "left_eye", "right_eye", "left_earend", "right_earend"]
    if any(n not in D.ix for n in names):
        return None
    sets = []
    for fr in range(0, D.N, 2):
        pp = [D.pt(fr, n) for n in names]
        if all(p is not None for p in pp):
            sets.append(np.array(pp))
    if len(sets) < 12:
        return None
    from scipy.interpolate import splprep, splev
    T = sets[0] - sets[0].mean(0)
    for _ in range(4):
        al = []
        for S in sets:
            R, t = _kabsch(S, T)
            al.append(S @ R.T + t)
        T = np.median(al, 0)
        T = T - T.mean(0)
    tip, le, re, ael, aer = T
    e0 = (le + re) / 2
    lat = re - le
    hw = np.linalg.norm(lat) / 2
    lat = lat / (2 * hw)
    fwd = tip - e0
    dn = np.linalg.norm(fwd)
    fwd = fwd - lat * (fwd @ lat)
    fwd /= np.linalg.norm(fwd)
    up = np.cross(fwd, lat)
    up /= np.linalg.norm(up)
    at = lambda u, v, w=0.0: e0 + fwd * u * dn + lat * v * hw + up * w
    half = [(1.00, 0.0), (0.82, 0.38), (0.52, 0.75), (0.15, 1.30), (-0.30, 1.75), (-0.80, 1.90), (-1.15, 1.50), (-1.32, 0.75)]
    ring = [(u, v) for u, v in half] + [(-1.38, 0.0)] + [(u, -v) for u, v in reversed(half[1:])]
    ring = np.array([at(u, v) for u, v in ring])
    ring = np.vstack([ring, ring[:1]])
    tck, _ = splprep(ring.T, s=0, per=True)
    outline = np.array(splev(np.linspace(0, 1, 60), tck)).T
    back = at(-1.38, 0.0)
    ears = []
    for tipE, s in ((ael, -1), (aer, 1)):
        base = at(-0.75, s * 1.75)
        a = tipE - base
        L = np.linalg.norm(a)
        a /= L
        p = fwd - a * (fwd @ a)
        p = p / np.linalg.norm(p) if np.linalg.norm(p) > 1e-6 else up
        tt = np.linspace(0, 1, 14)
        w = 0.40 * L * np.sin(np.pi * tt) ** 0.8
        s1 = base + a * L * tt[:, None] + p * w[:, None]
        s2 = base + a * L * tt[:, None] - p * w[:, None]
        ears.append(np.vstack([s1, s2[::-1]]))
    D._hm = dict(names=names, T=T, outline=outline, back=back, ears=ears)
    return D._hm


def head_fit(D, f):
    M = head_model(D)
    if M is None:
        return None
    pts = [(i, D.pt(f, n)) for i, n in enumerate(M["names"])]
    pts = [(i, p) for i, p in pts if p is not None]
    if len(pts) < 3:
        return None
    idx = [i for i, _ in pts]
    R, t = _kabsch(M["T"][idx], np.array([p for _, p in pts]))
    tr = lambda X: np.asarray(X) @ R.T + t
    return dict(tip=tr(M["T"][0]), le=tr(M["T"][1]), re=tr(M["T"][2]), outline=tr(M["outline"]),
                back=tr(M["back"]), ears=[tr(e) for e in M["ears"]])


def curved_spine(D, f, n=22):
    rump, neck = D.pt(f, "back_end"), D.pt(f, "neck_base")
    fwd = D.fwd[f]
    ln = np.linalg.norm(neck - rump)
    tip = D.pt(f, D.tip)
    hd = (tip - neck) if tip is not None else fwd * ln
    hd = hd / (np.linalg.norm(hd) + 1e-9)
    m_r = fwd + (fwd - hd) * 0.30
    m_n = fwd + (hd - fwd) * 0.45
    m_r = m_r / (np.linalg.norm(m_r) + 1e-9) * ln * 0.55
    m_n = m_n / (np.linalg.norm(m_n) + 1e-9) * ln * 0.55
    ad = np.array([0, 0, 1.0]) - fwd * fwd[2]
    ad = ad / (np.linalg.norm(ad) + 1e-9)
    amp = 0.06 * ln * (1 - abs(fwd[2]))
    t = np.linspace(0, 1, n)[:, None]
    t2, t3 = t * t, t ** 3
    p = ((2 * t3 - 3 * t2 + 1) * rump + (t3 - 2 * t2 + t) * m_r + (-2 * t3 + 3 * t2) * neck + (t3 - t2) * m_n)
    return p + ad * amp * np.sin(np.pi * t)


def toe_ends(D, f, p, back):
    L = 5.5 if back else 4.5
    fwd, lat = D.fwd[f], D.lat[f]
    return [p + (fwd * math.cos(math.radians(a)) + lat * math.sin(math.radians(a))) * L for a in (-38, 0, 38)]


def arena_box(D):
    """Floor rectangle + height from EVERYTHING drawn (joints, head outline, ears, toes) over the clip."""
    pts = []
    for b in USED:
        j = D.ix.get(b)
        if j is not None:
            pts.append(D.P[D.vis[:, j], j])
    pts = np.concatenate(pts)
    for f in range(0, D.N, 3):
        h = head_fit(D, f)
        if h is not None:
            pts = np.vstack([pts, h["outline"], h["ears"][0], h["ears"][1]])
    for b in ("front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw"):
        j = D.ix.get(b)
        if j is not None:
            v = D.P[D.vis[:, j], j]
            pts = np.vstack([pts, v + [6, 6, 0], v - [6, 6, 0]])
    x0, x1 = np.percentile(pts[:, 0], [0.02, 99.98])
    y0, y1 = np.percentile(pts[:, 1], [0.02, 99.98])
    z1 = float(np.percentile(pts[:, 2], 99.98))
    pad = 8.0
    zmin = min(-4.0, float(np.percentile(pts[:, 2], 0.02)) - 2.0)
    return dict(x0=float(x0 - pad), x1=float(x1 + pad), y0=float(y0 - pad), y1=float(y1 + pad), H=max(45.0, z1 + 8.0),
                zmin=zmin)


# --------------------------------------------------------------------------- GPU rasteriser
class Raster:
    """Analytic anti-aliased 2D drawing on the GPU (torch).  Coordinates are OUTPUT pixels; the canvas is
    supersampled SSx and averaged down."""

    def __init__(self, W, H, device, ss=SS):
        import torch
        self.t = torch
        self.W, self.H, self.ss, self.dev = W, H, ss, device
        self.CW, self.CH = W * ss, H * ss
        self.X = (torch.arange(self.CW, device=device, dtype=torch.float32) + 0.5).view(1, -1)
        self.Y = (torch.arange(self.CH, device=device, dtype=torch.float32) + 0.5).view(-1, 1)
        self.canvas = torch.zeros(self.CH, self.CW, 3, device=device, dtype=torch.float32)
        self._col = {}
        self.clip = (0, 0, self.CW, self.CH)

    def col(self, c):
        k = tuple(float(v) for v in c)
        if k not in self._col:
            self._col[k] = self.t.tensor(k, device=self.dev, dtype=self.t.float32).view(1, 1, 3)
        return self._col[k]

    def set_clip(self, rect):                       # rect in output px (x0, y0, x1, y1)
        s = self.ss
        self.clip = (max(0, int(rect[0] * s)), max(0, int(rect[1] * s)), min(self.CW, int(rect[2] * s)), min(self.CH, int(rect[3] * s)))

    def clear_clip(self):
        self.clip = (0, 0, self.CW, self.CH)

    def _roi(self, lo, hi):
        x0 = max(self.clip[0], int(math.floor(lo[0])))
        y0 = max(self.clip[1], int(math.floor(lo[1])))
        x1 = min(self.clip[2], int(math.ceil(hi[0])))
        y1 = min(self.clip[3], int(math.ceil(hi[1])))
        return (x0, y0, x1, y1) if (x1 > x0 and y1 > y0) else None

    def _blend(self, roi, cov, col, alpha):
        x0, y0, x1, y1 = roi
        a = (cov * alpha).unsqueeze(-1)
        self.canvas[y0:y1, x0:x1].lerp_(self.col(col), a)

    def segs(self, P0, P1, hw, col, alpha=1.0, dash=None):
        """Round-capped segments P0->P1 (E,2), half widths hw (E,) or scalar; one colour, coverage = max (no
        double darkening where they overlap)."""
        t = self.t
        P0 = np.atleast_2d(np.asarray(P0, float)) * self.ss
        P1 = np.atleast_2d(np.asarray(P1, float)) * self.ss
        E = len(P0)
        hw = np.broadcast_to(np.asarray(hw, float) * self.ss, (E,)).copy()
        m = float(hw.max()) + 1.5
        lo = np.minimum(P0, P1).min(0) - m
        hi = np.maximum(P0, P1).max(0) + m
        roi = self._roi(lo, hi)
        if roi is None:
            return
        x0, y0, x1, y1 = roi
        h, w = y1 - y0, x1 - x0
        Xs, Ys = self.X[:, x0:x1].view(1, 1, w), self.Y[y0:y1, :].view(1, h, 1)
        cov_all = None
        step = max(1, int(4_000_000 // max(1, h * w)))
        for s in range(0, E, step):
            a = t.as_tensor(P0[s:s + step], device=self.dev, dtype=t.float32)
            b = t.as_tensor(P1[s:s + step], device=self.dev, dtype=t.float32)
            hwt = t.as_tensor(hw[s:s + step], device=self.dev, dtype=t.float32).view(-1, 1, 1)
            ax_, ay_ = a[:, 0].view(-1, 1, 1), a[:, 1].view(-1, 1, 1)
            dx, dy = (b[:, 0] - a[:, 0]).view(-1, 1, 1), (b[:, 1] - a[:, 1]).view(-1, 1, 1)
            L2 = (dx * dx + dy * dy).clamp_min(1e-6)
            tt = ((Xs - ax_) * dx + (Ys - ay_) * dy) / L2
            tc = tt.clamp(0, 1)
            d = t.hypot(Xs - (ax_ + tc * dx), Ys - (ay_ + tc * dy))
            cov = (hwt + 0.5 - d).clamp(0, 1)
            if dash is not None and E == 1:
                on, off = dash[0] * self.ss, dash[1] * self.ss
                s_along = tt * L2.sqrt()
                cov = cov * ((s_along % (on + off)) < on).float()
            cov = cov.amax(0)
            cov_all = cov if cov_all is None else t.maximum(cov_all, cov)
        self._blend(roi, cov_all, col, alpha)

    def circ(self, c, r, col, alpha=1.0, ring=0.0):
        t = self.t
        cx, cy, r, ring = c[0] * self.ss, c[1] * self.ss, r * self.ss, ring * self.ss
        m = r + ring + 1.5
        roi = self._roi((cx - m, cy - m), (cx + m, cy + m))
        if roi is None:
            return
        x0, y0, x1, y1 = roi
        d = t.hypot(self.X[:, x0:x1] - cx, self.Y[y0:y1, :] - cy)
        cov = ((r + 0.5 - d).clamp(0, 1) if ring <= 0 else (ring / 2 + 0.5 - (d - r).abs()).clamp(0, 1))
        self._blend(roi, cov, col, alpha)

    def poly(self, pts, col, alpha=1.0):
        t = self.t
        P = np.asarray(pts, float) * self.ss
        roi = self._roi(P.min(0), P.max(0) + 1)
        if roi is None:
            return
        x0, y0, x1, y1 = roi
        h, w = y1 - y0, x1 - x0
        Xs, Ys = self.X[:, x0:x1].view(1, 1, w), self.Y[y0:y1, :].view(1, h, 1)
        A = t.as_tensor(P, device=self.dev, dtype=t.float32)
        B = t.roll(A, -1, 0)
        step = max(1, int(6_000_000 // max(1, h * w)))
        inside = None
        for s in range(0, len(A), step):
            xa, ya = A[s:s + step, 0].view(-1, 1, 1), A[s:s + step, 1].view(-1, 1, 1)
            xb, yb = B[s:s + step, 0].view(-1, 1, 1), B[s:s + step, 1].view(-1, 1, 1)
            cond = (ya <= Ys) != (yb <= Ys)
            xi = xa + (Ys - ya) * (xb - xa) / (yb - ya + 1e-9)
            par = ((cond & (Xs < xi)).sum(0) % 2) == 1
            inside = par if inside is None else (inside ^ par)
        self._blend(roi, inside.float(), col, alpha)

    def _over(self, roi, a, S_cols):
        """canvas <- canvas * prod(1-a_e) + sum_e c_e a_e prod_{k>e}(1-a_k)   (ordered 'over', a: (E,h,w))."""
        t = self.t
        x0, y0, x1, y1 = roi
        om = 1.0 - a
        rc = t.cumprod(om.flip(0), 0).flip(0)                      # inclusive suffix products
        S = t.cat([rc[1:], t.ones_like(rc[:1])], 0)                # exclusive
        contrib = t.einsum("ehw,ec->hwc", a * S, S_cols)
        self.canvas[y0:y1, x0:x1].mul_(rc[0].unsqueeze(-1)).add_(contrib)

    def segs_multi(self, rows):
        """Many round-capped segments, each with its own width / colour / alpha / dash, composited in list order.
        rows: [x0, y0, x1, y1, half_width, r, g, b, alpha, dash_on, dash_off] in output px."""
        if len(rows) == 0:
            return
        t = self.t
        A = np.asarray(rows, np.float64)
        A[:, 0:5] *= self.ss
        A[:, 9:11] *= self.ss
        P0, P1 = A[:, 0:2], A[:, 2:4]
        m = float(A[:, 4].max()) + 1.5
        roi = self._roi(np.minimum(P0, P1).min(0) - m, np.maximum(P0, P1).max(0) + m)
        if roi is None:
            return
        x0, y0, x1, y1 = roi
        h, w = y1 - y0, x1 - x0
        Xs, Ys = self.X[:, x0:x1].view(1, 1, w), self.Y[y0:y1, :].view(1, h, 1)
        G = t.as_tensor(A.astype(np.float32), device=self.dev)
        dashed = bool((A[:, 9] < 1e8).any())
        step = max(1, int(24_000_000 // max(1, h * w)))
        for s in range(0, len(A), step):
            g = G[s:s + step]
            ax, ay, bx, by, hw = (g[:, i].view(-1, 1, 1) for i in range(5))
            dx, dy = bx - ax, by - ay
            L2 = (dx * dx + dy * dy).clamp_min(1e-6)
            tt = ((Xs - ax) * dx + (Ys - ay) * dy) / L2
            tc = tt.clamp(0, 1)
            d = t.hypot(Xs - (ax + tc * dx), Ys - (ay + tc * dy))
            cov = (hw + 0.5 - d).clamp(0, 1)
            if dashed:
                on, off = g[:, 9].view(-1, 1, 1), g[:, 10].view(-1, 1, 1)
                cov = cov * (((tt * L2.sqrt()) % (on + off).clamp_min(1e-3)) < on).float()
            self._over(roi, cov * g[:, 8].view(-1, 1, 1), g[:, 5:8])

    def circ_multi(self, rows):
        """Filled discs / rings, composited in list order.  rows: [cx, cy, r, ring_width, R, G, B, alpha]."""
        if len(rows) == 0:
            return
        t = self.t
        A = np.asarray(rows, np.float64)
        A[:, 0:4] *= self.ss
        m = float((A[:, 2] + A[:, 3]).max()) + 1.5
        roi = self._roi(np.array([(A[:, 0] - m).min(), (A[:, 1] - m).min()]),
                        np.array([(A[:, 0] + m).max(), (A[:, 1] + m).max()]))
        if roi is None:
            return
        x0, y0, x1, y1 = roi
        h, w = y1 - y0, x1 - x0
        Xs, Ys = self.X[:, x0:x1].view(1, 1, w), self.Y[y0:y1, :].view(1, h, 1)
        G = t.as_tensor(A.astype(np.float32), device=self.dev)
        step = max(1, int(24_000_000 // max(1, h * w)))
        for s in range(0, len(A), step):
            g = G[s:s + step]
            cx, cy, r, rw = (g[:, i].view(-1, 1, 1) for i in range(4))
            d = t.hypot(Xs - cx, Ys - cy)
            fill = (r + 0.5 - d).clamp(0, 1)
            ring = (rw / 2 + 0.5 - (d - r).abs()).clamp(0, 1)
            cov = t.where(rw > 0, ring, fill)
            self._over(roi, cov * g[:, 7].view(-1, 1, 1), g[:, 4:7])

    def snapshot(self):
        return self.canvas.clone()

    def restore(self, snap):
        self.canvas.copy_(snap)

    def downsample(self):
        t = self.t
        img = self.canvas.permute(2, 0, 1).unsqueeze(0)
        return t.nn.functional.avg_pool2d(img, self.ss).squeeze(0).permute(1, 2, 0)      # (H, W, 3) float


# --------------------------------------------------------------------------- views
class View:
    def __init__(self, kind, rect, box, az=None, el=24.0, extra_pts=None):
        self.kind, self.rect, self.box = kind, rect, box
        self.az = math.radians(az) if az is not None else None
        self.el = math.radians(el)
        pts = self.corners()
        if extra_pts is not None:
            pts = np.vstack([pts, extra_pts])
        q = self.uv(pts)
        lo, hi = q.min(0), q.max(0)
        x0, y0, x1, y1 = rect
        self.s = 0.965 * min((x1 - x0) / (hi[0] - lo[0]), (y1 - y0) / (hi[1] - lo[1]))
        self.c = ((lo + hi) / 2)
        self.px0 = ((x0 + x1) / 2, (y0 + y1) / 2)

    def corners(self):
        b = self.box
        return np.array([(x, y, z) for x in (b["x0"], b["x1"]) for y in (b["y0"], b["y1"]) for z in (b.get("zmin", -4.0), b["H"])])

    def uv(self, P):
        P = np.asarray(P, float)
        if self.kind == "top":
            return np.stack([P[..., 0], P[..., 1]], -1)
        if self.kind == "side":
            return np.stack([P[..., 0], P[..., 2]], -1)
        xr = math.cos(self.az) * P[..., 0] + math.sin(self.az) * P[..., 1]
        yr = -math.sin(self.az) * P[..., 0] + math.cos(self.az) * P[..., 1]
        return np.stack([xr, P[..., 2] * math.cos(self.el) + yr * math.sin(self.el)], -1)

    def px(self, P):
        q = self.uv(P)
        return np.stack([self.px0[0] + (q[..., 0] - self.c[0]) * self.s, self.px0[1] - (q[..., 1] - self.c[1]) * self.s], -1)

    def depth(self, P):
        P = np.asarray(P, float)
        yr = -math.sin(self.az) * P[..., 0] + math.cos(self.az) * P[..., 1]
        return yr * math.cos(self.el) - P[..., 2] * math.sin(self.el)


def draw_arena(R, V, th, u):
    """floor + (two far) walls + light 20 mm grid, once per video."""
    b = V.box
    x0, x1, y0, y1, H = b["x0"], b["x1"], b["y0"], b["y1"], b["H"]
    F = np.array([(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)], float)
    T = F + [0, 0, H]
    floor, wall, edge, grid = (_rgb(th[k]) for k in ("floor", "wall", "edge", "grid"))
    STEP = 20.0
    gw = 0.55 * u
    ew = 0.9 * u

    def quad_grid(a, b_, c, d):
        nu = max(1, int(round(np.linalg.norm(b_ - a) / STEP)))
        nv = max(1, int(round(np.linalg.norm(d - a) / STEP)))
        P0, P1 = [], []
        for i in range(1, nu):
            t = i / nu
            P0.append(a + (b_ - a) * t)
            P1.append(d + (c - d) * t)
        for j in range(1, nv):
            t = j / nv
            P0.append(a + (d - a) * t)
            P1.append(b_ + (c - b_) * t)
        if P0:
            R.segs(V.px(np.array(P0)), V.px(np.array(P1)), gw, grid)

    def outline(poly3):
        q = V.px(poly3)
        R.segs(q, np.roll(q, -1, 0), ew, edge)

    if V.kind == "top":
        R.poly(V.px(F), floor)
        quad_grid(*F)
        outline(F)
    elif V.kind == "side":
        W_ = np.array([(x0, 0, 0), (x1, 0, 0), (x1, 0, H), (x0, 0, H)], float)
        R.poly(V.px(W_), wall)
        zm = b.get("zmin", -4.0)
        R.poly(V.px(np.array([(x0, 0, zm), (x1, 0, zm), (x1, 0, 0), (x0, 0, 0)], float)), floor)
        P0 = [(gx, 0, 0) for gx in np.arange(x0 + STEP, x1 - 1, STEP)] + [(x0, 0, gz) for gz in np.arange(STEP, H - 1, STEP)]
        P1 = [(gx, 0, H) for gx in np.arange(x0 + STEP, x1 - 1, STEP)] + [(x1, 0, gz) for gz in np.arange(STEP, H - 1, STEP)]
        R.segs(V.px(np.array(P0)), V.px(np.array(P1)), gw, grid)
        R.segs(V.px(np.array([(x0, 0, 0)])), V.px(np.array([(x1, 0, 0)])), ew * 1.2, edge)
        outline(np.array([(x0, 0, 0), (x0, 0, H), (x1, 0, H), (x1, 0, 0)], float))
    else:
        mids = [(F[i] + F[(i + 1) % 4]) / 2 for i in range(4)]
        for i in np.argsort([-float(V.depth(m)) for m in mids])[:2]:
            a, c = F[i], F[(i + 1) % 4]
            quad = np.array([a, c, T[(i + 1) % 4], T[i]])
            R.poly(V.px(quad), wall)
            quad_grid(a, c, T[(i + 1) % 4], T[i])
            outline(quad)
        R.poly(V.px(F), floor)
        quad_grid(*F)
        outline(F)


# --------------------------------------------------------------------------- per-frame drawing
class Painter:
    def __init__(self, D, style, theme, W, H, device):
        self.D, self.st = D, style
        self.dark = theme == "dark"
        self.th = THEMES["dark" if self.dark else "light"]
        self.u = (H / 1080.0) * PT                        # points -> pixels
        lw_pt, ms_pt = LINE_PT.get(style["line"], LINE_PT["medium"])
        self.lw = lw_pt * self.u
        self.ms = math.sqrt(ms_pt) * 0.5 * self.u * 1.0   # marker radius px
        self.colour = colour_fn(style["scheme"], self.dark)
        self.mid = self.colour("neck_base") if style["scheme"].startswith("region") else \
            ((0.78,) * 3 if self.dark else (0.29,) * 3)
        self.ink = _rgb(self.th["ink"])
        self.paper = _rgb(self.th["paper"])
        self.R = Raster(W, H, device)

    # -- helpers
    def _seg(self, V, a, b, col, w, alpha=1.0, dash=None):
        qa, qb = V.px(a), V.px(b)
        self.R.segs(qa[None], qb[None], w / 2, col, alpha, dash)

    def frame(self, f, V):
        """One view of one frame.  Primitives are collected per layer and drawn with a handful of batched GPU calls
        (exact ordered 'over' compositing), not one call per bone."""
        D, st, R, th = self.D, self.st, self.R, self.th
        lw, ms, colour = self.lw, self.ms, self.colour
        tip = D.tip
        P = lambda b: D.pt(f, b)
        oblique = V.kind not in ("top", "side")
        far_mode = st["far_limbs"] if (oblique and st["far_limbs"] in ("light", "dashed")) else None
        c_depth = float(V.depth(D.cent[f])) if far_mode else 0.0
        far = lambda p: far_mode is not None and (float(V.depth(p)) - c_depth) > 3.0
        mono = st["scheme"] == "mono"
        dash_far = (2.6 * self.u, 1.8 * self.u)
        FAINT = 0.55
        nb, rp = P("neck_base"), P("back_end")
        px = V.px
        SOLID = (1e9, 0.0)

        body, headl, spine, circ = [], [], [], []

        def add(rows, a, b, w, col, alpha=1.0, dash=None):
            qa, qb = px(a), px(b)
            d = dash or SOLID
            rows.append([qa[0], qa[1], qb[0], qb[1], w / 2, col[0], col[1], col[2], alpha, d[0], d[1]])

        def addq(rows, qa, qb, w, col, alpha=1.0):
            rows.append([qa[0], qa[1], qb[0], qb[1], w / 2, col[0], col[1], col[2], alpha, 1e9, 0.0])

        def addc(q, r, col, alpha=1.0, ring=0.0):
            circ.append([q[0], q[1], r, ring, col[0], col[1], col[2], alpha])

        ring_w = max(1.0, 0.9 * self.u)

        # floor shadow
        if st["shadow"]:
            sh = []
            hs0 = head_fit(D, f) if st["head"] != "tracked" else None
            for a, b in LIMBS:
                if P(a) is not None and P(b) is not None:
                    qa = px(np.array([P(a)[0], P(a)[1], 0.0]))
                    qb = px(np.array([P(b)[0], P(b)[1], 0.0]))
                    sh.append([qa[0], qa[1], qb[0], qb[1], lw * 0.75 / 2, 0, 0, 0, th["shadow"], 1e9, 0.0])
            if hs0 is not None:
                o = px(np.c_[hs0["outline"][:, :2], np.zeros(len(hs0["outline"]))])
                for i in range(len(o)):
                    j = (i + 1) % len(o)
                    sh.append([o[i][0], o[i][1], o[j][0], o[j][1], lw * 0.75 / 2, 0, 0, 0, th["shadow"], 1e9, 0.0])
            if nb is not None and rp is not None:
                cs = curved_spine(D, f)
                q = px(np.c_[cs[:, :2], np.zeros(len(cs))])
                for i in range(len(q) - 1):
                    sh.append([q[i][0], q[i][1], q[i + 1][0], q[i + 1][1], lw / 2, 0, 0, 0, th["shadow"], 1e9, 0.0])
            if sh:       # one colour/alpha: max-coverage (overlaps do not darken), single call
                A = np.array(sh)
                R.segs(A[:, 0:2], A[:, 2:4], A[:, 4], (0, 0, 0), th["shadow"])

        # limbs (far-side ones lighter / dashed)
        for a, b in LIMBS:
            pa, pb = P(a), P(b)
            if pa is None or pb is None:
                continue
            w = lw * (1.3 if b.endswith("thai") else 0.85 if b.endswith("paw") else 1.0)
            al, dash = 1.0, None
            if mono and _side(b) == "R":
                dash = (3 * self.u, 1.6 * self.u)
            pw = b.rsplit("_", 1)[0] + "_paw"
            if P(pw) is None or D.is_filled(f, pw) or D.is_filled(f, b):
                al = 0.45
            if far((pa + pb) / 2):
                if far_mode == "light":
                    al = min(al, FAINT)
                else:
                    dash, al = dash_far, max(al, 0.9)
            add(body, pa, pb, w, colour(b), al, dash)
        # toes (schematic fan at every paw)
        for b in USED:
            if "paw" not in b or P(b) is None:
                continue
            p = P(b)
            ta = 0.45 if D.is_filled(f, b) else 1.0
            if far_mode == "light" and far(p):
                ta = min(ta, FAINT)
            for e in toe_ends(D, f, p, b.startswith("back")):
                add(body, p, e, lw * 0.6, colour(b), ta)
                addc(px(e), ms * 0.28 * 1.1, colour(b), ta)

        # head
        hs = head_fit(D, f) if st["head"] != "tracked" else None
        hc = colour(tip)
        polys = []
        if hs is not None:
            mode = st["head"]
            fa = {"outline": 0.08, "filled": 0.26, "contrast": 0.16}[mode]
            ea = {"outline": 0.10, "filled": 0.42, "contrast": 0.35}[mode]
            qo = px(hs["outline"])
            polys.append((qo, hc, fa))
            for i in range(len(qo)):
                addq(headl, qo[i], qo[(i + 1) % len(qo)], lw * 0.95, hc)
            for e, nm in zip(hs["ears"], ("left_earend", "right_earend")):
                qe = px(e)
                polys.append((qe, colour(nm), ea))
                for i in range(len(qe)):
                    addq(headl, qe[i], qe[(i + 1) % len(qe)], lw * 0.8, colour(nm))
            for pe, nm in ((hs["le"], "left_eye"), (hs["re"], "right_eye")):
                q = px(pe)
                if mode == "contrast":
                    addc(q, ms * 0.95, self.paper)
                    addc(q, ms * 0.78, self.ink)
                elif mode == "filled":
                    addc(q, ms * 1.1, self.paper)
                    addc(q, ms * 0.95, colour(nm))
                else:
                    addc(q, ms * 0.8, colour(nm))
            q = px(hs["tip"])
            if mode == "contrast":
                addc(q, ms * 1.25, self.paper)
                addc(q, ms * 1.08, self.ink)
            else:
                addc(q, ms * (1.3 if mode == "filled" else 1.1), colour(tip))
            if nb is not None:
                add(spine, hs["back"], nb, lw, self.mid)
        else:
            hp = [P(n) for n in (tip, "left_eye", "left_earend", "right_earend", "right_eye")]
            hp = [p for p in hp if p is not None]
            if len(hp) >= 3:
                polys.append((px(np.array(hp)), hc, 0.13))
            for p in hp:
                addc(px(p), ms, hc)

        # spine (curved / tapered / straight)
        if nb is not None and rp is not None:
            if st["spine"] == "straight":
                add(spine, rp, nb, lw * 1.15, self.mid)
            else:
                q = px(curved_spine(D, f))
                if st["spine"] == "tapered":
                    tt = np.linspace(0, 1, len(q) - 1)
                    ws = lw * (0.8 + 1.3 * np.sin(np.pi * (0.1 + 0.8 * tt)))
                else:
                    ws = np.full(len(q) - 1, lw * 1.15)
                for i in range(len(q) - 1):
                    addq(spine, q[i], q[i + 1], ws[i], self.mid)

        # joints
        for b in USED:
            if _region(b) == "head" or b not in D.ix or "paw" in b:
                continue
            p = P(b)
            if p is None:
                continue
            ll = float(D.LL[f, D.ix[b]])
            q = px(p)
            if D.is_filled(f, b) or ll < GATE:
                addc(q, ms, colour(b), 0.85, ring_w)
            else:
                a = float(np.clip(0.6 + (ll - GATE) / (1 - GATE) * 0.4, 0.6, 1))
                if far_mode == "light" and far(p) and _region(b) in ("fore", "hind"):
                    a = FAINT
                addc(q, ms, colour(b), a)
        for b in USED:                               # bridged paws: hollow marker
            if "paw" in b and P(b) is not None and D.is_filled(f, b):
                addc(px(P(b)), ms * 0.9, colour(b), 0.7, ring_w)

        # ---- draw, back to front, in a few batched calls
        R.segs_multi(body)
        for q, c, a in polys:
            R.poly(q, c, a)
        R.segs_multi(headl)
        R.segs_multi(spine)
        R.circ_multi(circ)


# --------------------------------------------------------------------------- layout, labels, overlay
def layout_panels(layout, W, H):
    fx = lambda a, b, c, d: (a * W, b * H, c * W, d * H)
    if layout == "oblique":
        return {"obl": fx(.010, .075, .990, .925)}
    if layout == "two_oblique":
        return {"obl": fx(.005, .075, .495, .925), "obl2": fx(.505, .075, .995, .925)}
    return {"obl": fx(.008, .070, .612, .925), "top": fx(.620, .070, .995, .505), "side": fx(.620, .520, .995, .925)}


def build_views(box, panels):
    views = {}
    cx, y0 = (box["x0"] + box["x1"]) / 2, box["y0"]
    for name, rect in panels.items():
        if name == "obl":
            views[name] = View("obl", rect, box, az=38)
        elif name == "obl2":
            views[name] = View("obl", rect, box, az=128)
        elif name == "top":      # room below edge 1-2 for the side-view camera marker
            views[name] = View("top", rect, box, extra_pts=np.array([[cx, y0 - 26, 0.0]]))
        else:
            views[name] = View("side", rect, box)
    return views


def _font_path():
    for p in (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if os.path.exists(p):
            return p
    try:
        import matplotlib
        return os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
    except Exception:
        return None


def build_overlay(D, views, st, th, W, H, u):
    """Static text / badges / triads / key as one RGBA layer (PIL, drawn once, 3x then averaged down)."""
    from PIL import Image, ImageDraw, ImageFont
    S = 3
    img = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fp = _font_path()
    F = lambda pt: ImageFont.truetype(fp, max(8, int(round(pt * u * S)))) if fp else ImageFont.load_default()
    txt = tuple(int(v * 255) for v in _rgb(th["text"])) + (255,)
    mut = tuple(int(v * 255) for v in _rgb(th["muted"])) + (255,)
    bgc = tuple(int(v * 255) for v in _rgb(th["bg"])) + (255,)
    box = next(iter(views.values())).box
    cxm, cym = (box["x0"] + box["x1"]) / 2, (box["y0"] + box["y1"]) / 2
    sp = lambda p: (p[0] * S, p[1] * S)

    def text(xy, s, font, fill, anchor="la"):
        d.text(sp(xy), s, font=font, fill=fill, anchor=anchor)

    def badge(xy, s):
        r = 8.5 * u * S
        x, y = sp(xy)
        d.ellipse([x - r, y - r, x + r, y + r], fill=bgc, outline=mut, width=max(1, int(0.8 * u * S)))
        d.text((x, y), s, font=F(7.5), fill=txt, anchor="mm")

    def arrow(p0, p1):
        (x0, y0), (x1, y1) = sp(p0), sp(p1)
        d.line([x0, y0, x1, y1], fill=mut, width=max(1, int(1.1 * u * S)))
        a = math.atan2(y1 - y0, x1 - x0)
        h = 5.5 * u * S
        d.polygon([(x1, y1), (x1 - h * math.cos(a - .45), y1 - h * math.sin(a - .45)),
                   (x1 - h * math.cos(a + .45), y1 - h * math.sin(a + .45))], fill=mut)

    for name, V in views.items():
        x0, y0, x1, y1 = V.rect
        cap = {"obl": "oblique view" if st["layout"] != "two_oblique" else "view 1",
               "obl2": "view 2 (90 deg round)", "top": "top view (looking down)",
               "side": "side view \u2014 seen from edge 1\u20132"}[name]
        text((x0 + 6, y0 - 2), cap, F(9), mut, "ld")
        # numbered floor corners (same corner = same number in every view)
        if V.kind == "side":
            for xw, lab, al in ((box["x0"], "1\u00b74", "rt"), (box["x1"], "2\u00b73", "lt")):
                q = V.px(np.array([xw, 0.0, 0.0]))
                badge((q[0] + (-14 if al == "rt" else 14) * u, q[1] + 11 * u), lab.split("\u00b7")[0] + "\u00b7" + lab.split("\u00b7")[1])
        else:
            for i, (x, y) in enumerate([(box["x0"], box["y0"]), (box["x1"], box["y0"]), (box["x1"], box["y1"]), (box["x0"], box["y1"])]):
                off = 7.0
                q = V.px(np.array([x + math.copysign(off, x - cxm), y + math.copysign(off, y - cym), 0.0]))
                badge(q, str(i + 1))
        # axis triad
        org = (x0 + 26 * u, y1 - 26 * u)
        for nm, vec in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
            if V.kind == "side" and nm == "y":
                continue
            if V.kind == "top" and nm == "z":
                continue
            v = (V.px(np.array(vec) * 20.0) - V.px(np.zeros(3)))
            n = np.linalg.norm(v)
            if n < 1e-6:
                continue
            v = v / n * 26 * u if V.kind != "obl" else v / n * 26 * u
            arrow(org, (org[0] + v[0], org[1] + v[1]))
            text((org[0] + v[0] * 1.3, org[1] + v[1] * 1.3), nm, F(8), mut, "mm")
        # scale bar (top / side)
        if V.kind in ("top", "side"):
            a = V.px(np.array([box["x1"] - 20.0, box["y0"] if V.kind == "top" else 0.0, 0.0]))
            b = V.px(np.array([box["x1"], box["y0"] if V.kind == "top" else 0.0, 0.0]))
            yy = y1 - 12 * u
            xx1 = x1 - 10 * u
            xx0 = xx1 - abs(b[0] - a[0])
            d.line([xx0 * S, yy * S, xx1 * S, yy * S], fill=mut, width=max(1, int(1.6 * u * S)))
            text(((xx0 + xx1) / 2, yy - 3 * u), "20 mm", F(7.5), mut, "mb")
        # side-view camera marker in the top view
        if V.kind == "top":
            xc = (box["x0"] + box["x1"]) / 2
            q0 = V.px(np.array([xc, box["y0"] - 21.0, 0.0]))
            q1 = V.px(np.array([xc, box["y0"] - 7.0, 0.0]))
            r = 5.5 * u
            d.polygon([sp((q0[0] - r, q0[1] + r * .55)), sp((q0[0] + r, q0[1] + r * .55)),
                       sp((q0[0] + r, q0[1] - r * .55)), sp((q0[0] - r, q0[1] - r * .55))], fill=mut)
            arrow((q0[0], q0[1] - r * .55), (q1[0], q1[1]))
            text((q0[0] + r * 1.6, q0[1] + 1 * u), "side-view camera", F(7.5), mut, "lm")
    # colour key + footer
    cf = colour_fn(st["scheme"], th["bg"] == "#0d0d1a")
    tip = D.tip
    sc = st["scheme"]
    if sc in ("region", "region_cb"):
        items = [("head", cf(tip)), ("spine", cf("neck_base")), ("forelimbs", cf("front_left_thai")),
                 ("hindlimbs", cf("back_left_thai")), ("paws = darker shade", cf("front_left_paw"))]
    elif sc == "sides":
        items = [("left", cf("front_left_paw")), ("right", cf("front_right_paw")), ("midline", cf("neck_base"))]
    elif sc == "ramp":
        items = [("head", cf(tip)), ("shoulders", cf("front_left_thai")), ("hips", cf("back_left_thai")), ("rump", cf("back_end"))]
    elif sc == "teal":
        items = [("head", cf(tip)), ("body", cf("neck_base")), ("limbs", cf("front_left_thai")), ("paws", cf("front_left_paw"))]
    else:
        items = [("left (solid)", cf("front_left_paw")), ("right (dashed)", cf("front_right_paw"))]
    x, y = 14 * u, H - 40 * u
    f9 = F(9)
    for lab, c in items:
        col = tuple(int(v * 255) for v in c) + (255,)
        d.ellipse([(x) * S, (y - 4) * S, (x + 8 * u) * S, (y + 4 * u) * S], fill=col)
        text((x + 12 * u, y), lab, f9, col, "lm")
        x += 12 * u + d.textlength(lab, font=f9) / S + 18 * u
    note = ("tracked keypoints \u00b7 head tip = %s \u00b7 toes schematic \u00b7 faded = bridged (\u2264 %.1f s) or far-side limb"
            " \u00b7 head = fixed shape fitted to tracked head points \u00b7 floor shadow shows height") % (tip.replace("_", " "), st["fill_s"])
    text((W - 14 * u, H - 14 * u), note, F(7.5), mut, "rd")
    ov = img.resize((W, H), Image.LANCZOS)
    return np.asarray(ov)


class Clock:
    """GPU glyph blit for the running 't = 12.34 s' label."""

    def __init__(self, W, H, u, th, device):
        import torch
        from PIL import Image, ImageDraw, ImageFont
        self.t, self.dev = torch, device
        fp = _font_path()
        font = ImageFont.truetype(fp, max(8, int(round(10.5 * u)))) if fp else ImageFont.load_default()
        self.tiles = {}
        self.h = 0
        for ch in "0123456789.:st= ":
            w = int(math.ceil(font.getlength(ch))) + 1
            im = Image.new("L", (max(w, 3), int(16 * u) + 4), 0)
            ImageDraw.Draw(im).text((0, 0), ch, font=font, fill=255)
            a = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).to(device)
            self.tiles[ch] = a
            self.h = a.shape[0]
        self.xy = (int(14 * u), int(10 * u))
        self.col = torch.tensor(_rgb(th["text"]), device=device, dtype=torch.float32)

    def blit(self, img, s):
        x, y = self.xy
        for ch in s:
            a = self.tiles.get(ch)
            if a is None:
                continue
            h, w = a.shape
            roi = img[y:y + h, x:x + w]
            roi.lerp_(self.col.view(1, 1, 3), a[:roi.shape[0], :roi.shape[1]].unsqueeze(-1))
            x += w - 1


# --------------------------------------------------------------------------- encoders
_NVENC = {}


def _nvenc_ok(ffmpeg_exe):
    if ffmpeg_exe not in _NVENC:
        try:
            r = subprocess.run([ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
                                "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, timeout=30)
            _NVENC[ffmpeg_exe] = r.returncode == 0
        except Exception:
            _NVENC[ffmpeg_exe] = False
    return _NVENC[ffmpeg_exe]


def _encoders(ffmpeg_exe, force_cpu=False):
    out = []
    if not force_cpu and _nvenc_ok(ffmpeg_exe):
        out.append(("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "21", "-b:v", "0"]))
    out.append(("libx264", ["-c:v", "libx264", "-threads", "2", "-crf", "20", "-preset", "veryfast"]))
    return out


# --------------------------------------------------------------------------- main render
def bounds_check(D, views, st):
    """Count drawn world points that fall outside their panel (should be 0: nothing cut off)."""
    bad = {k: 0 for k in views}
    for f in range(D.N):
        pts = [D.pt(f, b) for b in USED]
        pts = [p for p in pts if p is not None]
        hs = head_fit(D, f)
        if hs is not None:
            pts += list(hs["outline"]) + list(hs["ears"][0]) + list(hs["ears"][1])
        for b in USED:
            if "paw" in b and D.pt(f, b) is not None:
                pts += toe_ends(D, f, D.pt(f, b), b.startswith("back"))
        if not pts:
            continue
        pts = np.array(pts)
        for k, V in views.items():
            q = V.px(pts)
            x0, y0, x1, y1 = V.rect
            bad[k] += int(((q[:, 0] < x0) | (q[:, 0] > x1) | (q[:, 1] < y0) | (q[:, 1] > y1)).sum())
    return bad


def render_skeleton_video(h5_path, out_path, fps=30.0, ffmpeg_exe=None, logger=None, theme="light", style=None,
                          frame_range=None, still=None, progress=None, force_cpu_encode=False, gate=GATE):
    """Render the standard skeleton video (or one still PNG when `still` is a frame number)."""
    import torch
    log = (lambda m: logger.info(m)) if logger is not None else (lambda m: print(m, flush=True))
    st = dict(DEFAULT_STYLE)
    st.update({k: v for k, v in (style or {}).items() if v is not None})
    theme = "dark" if str(theme).lower() == "dark" else "light"
    th = THEMES[theme]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    t0 = time.time()
    D = SkelData(h5_path, fps, gate=float(gate), fill_s=float(st["fill_s"]))
    box = arena_box(D)
    H = int(st["height"]) - int(st["height"]) % 2
    W = int(round(H * 16 / 9)); W -= W % 2
    panels = layout_panels(st["layout"], W, H)
    views = build_views(box, panels)
    painter = Painter(D, st, theme, W, H, dev)
    R, u = painter.R, painter.u
    R.canvas[:] = R.col(_rgb(th["bg"]))
    for V in views.values():
        R.set_clip((V.rect[0] - 4, V.rect[1] - 4, V.rect[2] + 4, V.rect[3] + 4))
        draw_arena(R, V, th, u)
    R.clear_clip()
    snap = R.snapshot()
    ov = torch.from_numpy(build_overlay(D, views, st, th, W, H, u).astype(np.float32) / 255.0).to(dev)
    ov_rgb, ov_a = ov[..., :3], ov[..., 3:4]
    clock = Clock(W, H, u, th, dev)

    def one(f):
        R.restore(snap)
        for V in views.values():
            R.set_clip(V.rect)
            painter.frame(f, V)
        R.clear_clip()
        img = R.downsample()
        img = img * (1 - ov_a) + ov_rgb * ov_a
        clock.blit(img, "t = %.2f s" % (f / D.fps))
        return (img.clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()

    if still is not None:
        import cv2
        arr = one(int(still))
        cv2.imwrite(str(out_path), arr[..., ::-1])
        log(f"still written: {out_path} ({time.time() - t0:.1f}s)")
        return Path(out_path)

    a, b = (0, D.N) if frame_range is None else (max(0, int(frame_range[0])), min(D.N, int(frame_range[1])))
    out_path = Path(out_path)
    tmp = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    ffmpeg_exe = ffmpeg_exe or "ffmpeg"
    last_err = None
    for enc_name, enc_args in _encoders(ffmpeg_exe, force_cpu_encode):
        cmd = [ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "-", "-an"] + enc_args + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
        log(f"  skeleton video: {W}x{H} {b - a} frames, device={dev}, encoder={enc_name}")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        q = queue.Queue(maxsize=6)
        err = []

        def writer():
            try:
                while True:
                    item = q.get()
                    if item is None:
                        break
                    proc.stdin.write(item)
            except Exception as e:                      # encoder died
                err.append(e)
                while q.get() is not None:
                    pass
        th_w = threading.Thread(target=writer, daemon=True)
        th_w.start()
        try:
            for i, f in enumerate(range(a, b)):
                q.put(one(f).tobytes())
                if err:
                    break
                if (i + 1) % 50 == 0 or i + 1 == b - a:
                    msg = f"PROGRESS {i + 1}/{b - a} ({(i + 1) / max(1e-6, time.time() - t0):.1f} fps)"
                    log(msg)
                    if progress:
                        progress(i + 1, b - a)
        finally:
            q.put(None)
            th_w.join(timeout=60)
            try:
                proc.stdin.close()
            except Exception:
                pass
            rc = proc.wait(timeout=120)
            stderr = proc.stderr.read().decode(errors="replace")[-600:]
        if rc == 0 and not err:
            tmp.replace(out_path)
            log(f"  skeleton video written: {out_path.name} ({time.time() - t0:.1f}s)")
            return out_path
        last_err = f"{enc_name}: rc={rc} {stderr}"
        tmp.unlink(missing_ok=True)
        log(f"  encoder {enc_name} failed, trying next: {stderr[-200:]}")
    raise RuntimeError(f"skeleton video encode failed: {last_err}")


# --------------------------------------------------------------------------- isolated run
def render_skeleton_video_isolated(h5_path, out_path, fps, ffmpeg_exe, logger, theme="light", style=None,
                                   frame_range=None, stall_timeout_s=240, gate=GATE):
    """Run the renderer in its own low-priority process with capped CPU threads and a stall watchdog.
    Raises on failure so the caller can fall back to the legacy renderer."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--video", str(h5_path), str(out_path),
           "--fps", f"{fps:g}", "--theme", theme, "--ffmpeg", str(ffmpeg_exe or "ffmpeg"),
           "--style-json", json.dumps(style or {}), "--gate", f"{float(gate):g}"]
    if frame_range is not None:
        cmd += ["--frames", f"{int(frame_range[0])}:{int(frame_range[1])}"]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2",
               PYTHONUNBUFFERED="1")
    flags = 0x00004000 if os.name == "nt" else 0         # BELOW_NORMAL_PRIORITY_CLASS: keep the PC responsive
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                            creationflags=flags)
    last = [time.time()]
    lines = []

    def pump():
        for line in proc.stdout:
            line = line.rstrip()
            last[0] = time.time()
            lines.append(line)
            if line.startswith(("PROGRESS", "  skeleton video", "  encoder")):
                logger.info(line)
    t = threading.Thread(target=pump, daemon=True)
    t.start()
    while proc.poll() is None:
        time.sleep(1.0)
        if time.time() - last[0] > stall_timeout_s:
            proc.kill()
            raise RuntimeError(f"skeleton renderer stalled for {stall_timeout_s}s and was stopped")
    t.join(timeout=10)
    if proc.returncode != 0 or not Path(out_path).exists():
        raise RuntimeError("skeleton renderer failed (rc=%s): %s" % (proc.returncode, " | ".join(lines[-6:])))
    return Path(out_path)


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("out")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--theme", default="light")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--still", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--style-json", default="{}")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gate", type=float, default=GATE)
    a = ap.parse_args(argv)
    style = json.loads(a.style_json)
    if a.height:
        style["height"] = a.height
    fr = tuple(int(x) for x in a.frames.split(":")) if a.frames else None
    if a.check:
        st = dict(DEFAULT_STYLE); st.update(style)
        D = SkelData(a.h5, a.fps, fill_s=st["fill_s"])
        H = st["height"]; W = int(round(H * 16 / 9))
        views = build_views(arena_box(D), layout_panels(st["layout"], W, H))
        print("points outside their panel:", bounds_check(D, views, st))
        return
    ff = a.ffmpeg
    if ff is None:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ff = "ffmpeg"
    render_skeleton_video(a.h5, a.out, a.fps, ff, None, theme=a.theme, style=style, frame_range=fr, still=a.still,
                          gate=a.gate)


if __name__ == "__main__":
    _main(sys.argv[1:])
