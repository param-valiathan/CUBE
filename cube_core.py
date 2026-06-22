# -*- coding: utf-8 -*-
"""
cube_core.py  — Core analysis engine for CUBE: Comprehensive Unsupervised Behavioral Explorer
==============================================================================================
Contains ALL analysis logic.  No GUI code here - import into the main app.

Handles:
    DLC H5/CSV loading  (3-level AND 4-level MultiIndex - SuperAnimal safe)
    Feature extraction V2  (multi-scale 50/100/250 ms, body-normalised, angular)
    UMAP + HDBSCAN + MLP pipeline  (B-SOiD published methodology)
    Validation layer  (silhouette, CV accuracy, trustworthiness, DLC quality gates)
    Bout / epoch export in exact B-SOiD GUI format
    Video clip creation
    Plot generation
    DLC pre-processing (bodypart conservation, confidence filtering)
    Full audit-trail JSON summary + validation_report.json
"""

#   stdlib  
import json, pickle, re, shutil, time, traceback, warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

#   science (always available in a DLC/BSOID conda env)  
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # never open a display window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker

warnings.filterwarnings("ignore")

#  
#  CONSTANTS
#  

VERSION = "2.0"

# Analysis-behaviour version.  Bumped when a change alters the numeric output of
# a fresh run (new defaults, fixed sampling, etc.).  Stamped into the saved
# model pkl, feature_config.json and validation_report.json so any output can be
# traced to the behaviour that produced it.  cfg["compat_mode"] == "legacy_v2"
# restores the pre-2.1 numeric defaults/branches for exact reproduction of old
# runs (see BSoidEngine.DEFAULTS and _apply_compat_mode).
ANALYSIS_VERSION = "2.1"

PALETTE = [
    "#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
    "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC",
    "#00BCD4","#FF5722","#8BC34A","#9C27B0","#FFC107",
    "#3F51B5","#009688","#FF4081","#CDDC39","#795548",
]

# Mutable theme globals — updated by _apply_plot_theme() at the start of each run
_BG       = "#0d0d1a"
_PANEL    = "#1a1a2e"
_TEXT_COL = "white"       # primary text / axis title
_TICK_COL = "#aaaacc"     # tick labels, axis labels, colourbar labels

_THEME_COLORS = {
    "dark":  dict(bg="#0d0d1a", panel="#1a1a2e",
                  text="white",    tick="#aaaacc",  mpl_style="dark_background"),
    "light": dict(bg="#f5f6fa",  panel="#ffffff",
                  text="#1a1a2e", tick="#444466",   mpl_style="seaborn-v0_8-whitegrid"),
}


def _apply_plot_theme(theme: str = "dark") -> None:
    """Update module-level colour globals and matplotlib style for every plot."""
    global _BG, _PANEL, _TEXT_COL, _TICK_COL
    c = _THEME_COLORS.get(theme, _THEME_COLORS["dark"])
    _BG, _PANEL, _TEXT_COL, _TICK_COL = c["bg"], c["panel"], c["text"], c["tick"]
    try:
        plt.style.use(c["mpl_style"])
    except Exception:
        pass

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
DLC_EXTS   = {".csv", ".h5", ".hdf5"}

_TS_RE = re.compile(r"(\d{8}_\d{6})")   # YYYYMMDD_HHMMSS

#  
#  LOGGING HELPER  (thread-safe, writes to file and a queue for the GUI)
#  

import queue as _queue
import threading as _threading

class PipelineLogger:
    """Thread-safe logger - writes to file + exposes a queue for the GUI."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"pipeline_{ts}.log"
        self._q: _queue.Queue = _queue.Queue()
        self._lock = _threading.Lock()
        self._fh = open(self.log_path, "w", buffering=1, encoding="utf-8")

    # public API
    def info   (self, m): self._w("INFO",    m)
    def step   (self, m): self._w("STEP",    m)
    def warn   (self, m): self._w("WARN",    m)
    def error  (self, m): self._w("ERROR",   m)
    def success(self, m): self._w("SUCCESS", m)
    def __call__(self, m): self._w("INFO", str(m))   # drop-in for print()

    def _w(self, level, msg):
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{level:7s}]  {msg}"
        with self._lock:
            try:
                self._fh.write(line + "\n")
                self._fh.flush()
            except Exception:
                pass
        self._q.put((level, msg, ts))

    def close(self):
        try: self._fh.close()
        except Exception: pass


#  
#  COLOUR HELPERS
#  

def _cmap(i: int) -> str:
    return PALETTE[int(i) % len(PALETTE)]

def _hex_to_bgr(h: str) -> tuple:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


#  
#  DLC FILE LOADING
#  

def _normalise_dlc_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise any DLC MultiIndex to exactly 3 levels:
    (scorer, bodyparts, coords).
    Handles:
        4-level SuperAnimal output  (scorer, individuals, bodyparts, coords)
        2-level CSV with no scorer  (bodyparts, coords)
        3-level standard DLC        (pass-through)
    """
    nlevels = df.columns.nlevels

    if nlevels == 4:
        names  = list(df.columns.names)
        # find the 'individuals' level by name, fall back to position 1
        ind_lv = next((i for i,n in enumerate(names)
                       if n in ("individuals","individual")), 1)
        inds   = df.columns.get_level_values(ind_lv).unique()
        if len(inds) == 1:
            # single individual - just drop that level
            df = df.xs(inds[0], level=ind_lv, axis=1)
        else:
            # multi-animal - merge individual into bodypart label
            sc_lv = next((i for i,n in enumerate(names)
                          if n in ("scorer",)), 0)
            bp_lv = next((i for i,n in enumerate(names)
                          if n in ("bodyparts","bodypart")), 2)
            co_lv = next((i for i,n in enumerate(names)
                          if n in ("coords","coord")), 3)
            new_t = [(t[sc_lv], f"{t[ind_lv]}_{t[bp_lv]}", t[co_lv])
                     for t in df.columns]
            df.columns = pd.MultiIndex.from_tuples(
                new_t, names=["scorer","bodyparts","coords"])

    elif nlevels == 2:
        # CSV without scorer header row
        new_t = [("DLC_scorer", bp, co) for bp, co in df.columns]
        df.columns = pd.MultiIndex.from_tuples(
            new_t, names=["scorer","bodyparts","coords"])

    # At this point nlevels should be 3
    # Rename level labels to canonical names if they differ
    df.columns.names = ["scorer","bodyparts","coords"]
    return df


def load_dlc_file(path, likelihood_thresh: float = 0.3,
                  max_interp_gap_frames: int = None, log_fn=None,
                  return_quality: bool = False):
    """
    Load a DLC CSV or H5 file.

    Parameters
    ----------
    likelihood_thresh     : frames below this confidence are interpolated.
    max_interp_gap_frames : if set (>0), runs of consecutive low-confidence
        frames LONGER than this are NOT linearly interpolated across — the
        nearest good value is held flat instead (zero velocity).  This prevents
        long occlusions from being filled with a smooth straight-line trajectory
        that the feature engine would read as real low-velocity behavior.
        None / 0 = legacy behavior (interpolate across any gap).
    log_fn                : optional callable(str) for a per-file interpolation
        summary (worst bodyparts, frames held over long gaps).
    return_quality        : if True, return a 4th element — dict mapping each
        bodypart name to the fraction of frames below likelihood_thresh.  Used
        by BSoidEngine to identify chronically occluded keypoints and exclude
        them from feature extraction.

    Returns
    -------
    xy          : np.ndarray  (N_frames, n_bodyparts * 2)
    bodyparts   : list[str]
    fps_hint    : float | None   - extracted from filename if present
    ll_fracs    : dict[str, float]  (only when return_quality=True)
    """
    path = str(path)
    ext  = Path(path).suffix.lower()

    if ext in (".h5", ".hdf5"):
        df = pd.read_hdf(path)
    elif ext == ".csv":
        # peek at the first few lines to count header rows
        with open(path, encoding="utf-8", errors="replace") as fh:
            head = [fh.readline() for _ in range(5)]
        n_levels = sum(
            1 for l in head[:4]
            if l.strip() and not l.strip()[0].isdigit()
        )
        n_levels = max(n_levels, 2)
        df = pd.read_csv(path, header=list(range(n_levels)), index_col=0)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    df = _normalise_dlc_df(df)

    bodyparts = df.columns.get_level_values("bodyparts").unique().tolist()
    n_frames  = len(df)
    n_pts     = len(bodyparts)

    xy = np.full((n_frames, n_pts * 2), np.nan, dtype=float)
    ll = np.full((n_frames, n_pts),     np.nan, dtype=float)

    for i, bp in enumerate(bodyparts):
        try:
            sub = df.xs(bp, level="bodyparts", axis=1)
            # drop extra scorer level if present
            if sub.columns.nlevels > 1:
                sub.columns = sub.columns.get_level_values(-1)
            xy[:, 2*i]   = pd.to_numeric(
                sub.get("x",   pd.Series(np.nan, index=df.index)),
                errors="coerce").values
            xy[:, 2*i+1] = pd.to_numeric(
                sub.get("y",   pd.Series(np.nan, index=df.index)),
                errors="coerce").values
            ll[:, i]     = pd.to_numeric(
                sub.get("likelihood", pd.Series(1.0, index=df.index)),
                errors="coerce").values
        except Exception:
            pass   # leave as NaN; interpolation handles below

    # Interpolate low-likelihood frames
    frames = np.arange(n_frames)
    _cap = int(max_interp_gap_frames) if max_interp_gap_frames else 0
    _long_gap_frames = 0   # counted once per bodypart (on the x column)
    flat_held_frame_mask = np.zeros(n_frames, dtype=bool)
    # Per-bodypart flat-held masks so the pipeline can exclude only the
    # bodyparts that survive feature_bad_bp_thresh filtering.
    _flat_held_per_bp: list = [np.zeros(n_frames, dtype=bool) for _ in range(n_pts)]
    for col in range(xy.shape[1]):
        bp_i = col // 2
        lk   = ll[:, bp_i]
        bad  = (lk < likelihood_thresh) | np.isnan(xy[:, col])
        good = ~bad
        if good.sum() > 1:
            xy[bad, col] = np.interp(frames[bad], frames[good], xy[good, col])
            if _cap > 0:
                # Replace the linear ramp across over-long gaps with a flat hold
                # of the nearest good value (split at the gap midpoint).
                i = 0
                while i < n_frames:
                    if bad[i]:
                        j = i
                        while j < n_frames and bad[j]:
                            j += 1
                        if (j - i) > _cap:
                            left  = i - 1
                            right = j if j < n_frames else -1
                            if left >= 0 and right >= 0:
                                mid = (i + j) // 2
                                xy[i:mid, col] = xy[left, col]
                                xy[mid:j, col] = xy[right, col]
                            elif left >= 0:
                                xy[i:j, col] = xy[left, col]
                            elif right >= 0:
                                xy[i:j, col] = xy[right, col]
                            if col % 2 == 0:
                                _long_gap_frames += (j - i)
                                flat_held_frame_mask[i:j] = True
                                _flat_held_per_bp[bp_i][i:j] = True
                        i = j
                    else:
                        i += 1
        elif good.sum() == 1:
            xy[bad, col] = xy[good, col][0]
        else:
            xy[:, col] = 0.0

    # Per-file interpolation summary (worst-tracked bodyparts + long-gap holds)
    if log_fn is not None and n_pts > 0:
        fracs = [(bodyparts[i], float((ll[:, i] < likelihood_thresh).mean()))
                 for i in range(n_pts)]
        worst = [f"{b}:{f*100:.0f}%" for b, f in
                 sorted(fracs, key=lambda x: x[1], reverse=True)[:3] if f > 0.05]
        if worst or _long_gap_frames:
            msg = "    interp: "
            if worst:
                msg += "worst bodyparts below thresh — " + ", ".join(worst)
            if _long_gap_frames:
                msg += (f"; {_long_gap_frames} frame(s) across > "
                        f"{_cap}-frame gaps held flat (not ramped)")
            log_fn(msg)

    # Guess FPS from filename  (e.g. "60Hz" or "30fps")
    fps_hint = None
    m = re.search(r"(\d+)\s*(?:[Hh][Zz]|[Ff][Pp][Ss])", Path(path).stem)
    if m:
        fps_hint = float(m.group(1))

    if return_quality:
        ll_fracs = {bodyparts[i]: float((ll[:, i] < likelihood_thresh).mean())
                    for i in range(n_pts)}
        return xy, bodyparts, fps_hint, ll_fracs, _flat_held_per_bp
    return xy, bodyparts, fps_hint


#  
#  SMOOTHING
#  

def smooth_boxcar(xy: np.ndarray, fps: float, win_sec: float) -> np.ndarray:
    """Centred boxcar (moving average) smoothing per column."""
    win = max(1, int(round(fps * win_sec)))
    if win <= 1:
        return xy.copy()
    k = np.ones(win) / win
    return np.column_stack(
        [np.convolve(xy[:, c], k, mode="same") for c in range(xy.shape[1])]
    )


#
#  V2 FEATURE EXTRACTION HELPERS
#

def _find_spine_indices(bodyparts: list):
    """
    Return (head_idx, tail_idx) for spine-length normalisation.
    Matched by keyword priority; returns (None, None) if fewer than 2 found.
    """
    if not bodyparts:
        return None, None
    bp_lower = [b.lower() for b in bodyparts]
    head_kw  = ["nose", "snout", "head", "neck", "rostral"]
    tail_kw  = ["tailbase", "tail_base"]   # strict: only true tailbase landmarks
    head_idx = next(
        (i for kw in head_kw for i, b in enumerate(bp_lower) if kw in b),
        None)
    tail_idx = next(
        (len(bp_lower) - 1 - i
         for kw in tail_kw
         for i, b in enumerate(reversed(bp_lower)) if kw in b),
        None)
    if head_idx is not None and tail_idx is not None and head_idx != tail_idx:
        return head_idx, tail_idx
    return None, None


def _spine_norm_factor(xs: np.ndarray, ys: np.ndarray,
                        head_idx: int, tail_idx: int) -> np.ndarray:
    """
    Per-bin nose-to-tail distance (pixels).
    Floored at 10 px to prevent division-by-near-zero for occluded animals.
    Returns (n_bins,).
    """
    dx = xs[:, head_idx] - xs[:, tail_idx]
    dy = ys[:, head_idx] - ys[:, tail_idx]
    return np.maximum(np.sqrt(dx ** 2 + dy ** 2), 10.0)


def _angular_features(xs: np.ndarray, ys: np.ndarray,
                       bodyparts: list,
                       allow_fallback: bool = True) -> "np.ndarray | None":
    """
    Angles (radians) at the vertex B for consecutive body-axis triples A-B-C.
    Spine bodyparts detected by keyword.
    Returns (n_bins, n_angles) or None when fewer than 3 spine parts are found.

    allow_fallback : when True (pre-2.1 behaviour) and keyword matching finds
        < 3 spine parts, fall back to evenly-spaced bodypart indices.  Those
        triples need not lie on the body axis, so the resulting angles can be
        biologically meaningless.  When False (v2.1 default) the angular block is
        skipped entirely if no spine landmarks match by keyword.
    """
    if bodyparts is None or len(bodyparts) < 3:
        return None
    bp_lower  = [b.lower() for b in bodyparts]
    spine_kw  = ["nose", "snout", "neck", "spine", "back",
                 "body", "hip", "sacrum", "pelvis", "tailbase", "tail"]
    seen: set = set()
    spine_ids: list = []
    for kw in spine_kw:
        for i, b in enumerate(bp_lower):
            if kw in b and i not in seen:
                seen.add(i)
                spine_ids.append(i)
    if len(spine_ids) < 3:
        if not allow_fallback:
            return None
        n = len(bodyparts)
        spine_ids = list(range(0, n, max(1, n // 5)))[:6]
    if len(spine_ids) < 3:
        return None
    angles = []
    for k in range(len(spine_ids) - 2):
        ia, ib, ic = spine_ids[k], spine_ids[k + 1], spine_ids[k + 2]
        v1x = xs[:, ia] - xs[:, ib];  v1y = ys[:, ia] - ys[:, ib]
        v2x = xs[:, ic] - xs[:, ib];  v2y = ys[:, ic] - ys[:, ib]
        n1  = np.sqrt(v1x ** 2 + v1y ** 2) + 1e-8
        n2  = np.sqrt(v2x ** 2 + v2y ** 2) + 1e-8
        cos = np.clip((v1x * v2x + v1y * v2y) / (n1 * n2), -1.0, 1.0)
        angles.append(np.arccos(cos))
    return np.column_stack(angles) if angles else None


#
#  FEATURE EXTRACTION  (V1 — kept for backward compatibility)
#

def extract_features(xy: np.ndarray, fps: float) -> np.ndarray:
    """
    V1 feature extraction (original B-SOiD): 100-ms bins, pairwise distances
    + frame-to-frame displacement.  Retained for backwards compatibility with
    saved models.  New pipelines should call extract_features_v2().

    Returns (n_features, n_bins).
    """
    win    = max(1, int(round(fps / 10)))
    n_f, n_xy = xy.shape
    n_pts  = n_xy // 2
    n_bins = n_f  // win

    if n_bins < 2:
        raise ValueError(
            f"Recording too short: {n_f} frames at {fps} fps -> "
            f"{n_bins} 100-ms bins.  Need   {2*win} frames.")

    trimmed = xy[:n_bins * win]
    binned  = trimmed.reshape(n_bins, win, n_xy).mean(axis=1)
    xs      = binned[:, 0::2]
    ys      = binned[:, 1::2]

    pairs = []
    for i, j in combinations(range(n_pts), 2):
        d = np.sqrt((xs[:, i] - xs[:, j])**2 + (ys[:, i] - ys[:, j])**2)
        pairs.append(d)

    dx   = np.diff(xs, axis=0, prepend=xs[:1])
    dy   = np.diff(ys, axis=0, prepend=ys[:1])
    disp = np.sqrt(dx**2 + dy**2)

    feats = np.hstack([np.column_stack(pairs), disp]) if pairs else disp
    return feats.T   # (n_features, n_bins)


#
#  FEATURE EXTRACTION  (V2 — multi-scale, body-normalised, angular)
#

def extract_features_v2(xy: np.ndarray, fps: float,
                         bodyparts: list = None,
                         body_normalise: bool = True,
                         angular_fallback: bool = True) -> np.ndarray:
    """
    V2 multi-scale feature extraction (CUBE — Version 2 Framework).

    Improvements over V1
    --------------------
    Body-size normalisation : nose-to-tailbase distances (optional, user toggle)
    FPS-adaptive scales     : 100ms + 200ms at standard fps; 50ms added at 60fps+
    Kinematics              : velocity + smoothed acceleration (3-bin boxcar)
    Angular features        : angles between consecutive body-axis triples

    Parameters
    ----------
    xy             : (n_frames, n_pts*2)  smoothed xy coordinates
    fps            : recording frame rate
    bodyparts      : bodypart name list (enables normalisation + angular features)
    body_normalise : divide spatial features by nose-to-tailbase length each bin

    Returns
    -------
    (n_features, n_bins)  at 100-ms temporal resolution — same convention as V1
    """
    win100 = max(1, int(round(fps / 10)))    # 100 ms — reference resolution
    # 50ms = 1.5 frames at 30fps: captures DLC jitter, not real behavior.
    # Only add the fine scale when fps >= 60 where it resolves genuine fast movements.
    use_fine_scale = fps >= 60
    win_fine   = max(1, int(round(fps / 20)))  # ~50 ms at 60fps+
    win_coarse = max(1, int(round(fps / 5)))   # 200 ms — slow postural context

    n_f, n_xy = xy.shape
    n_pts     = n_xy // 2
    n_bins    = n_f  // win100

    if n_bins < 2:
        raise ValueError(
            f"Recording too short: {n_f} frames @ {fps} fps "
            f"-> {n_bins} 100-ms bins (need >= {2 * win100} frames).")

    # ── Body-part indices for spine normalisation ─────────────────────────────
    if body_normalise:
        head_idx, tail_idx = _find_spine_indices(bodyparts or [])
    else:
        head_idx, tail_idx = None, None

    # ── 100 ms bins (reference) ───────────────────────────────────────────────
    _b100_raw = xy[:n_bins * win100].reshape(n_bins, win100, n_xy)
    b100  = _b100_raw.mean(axis=1)
    xs100 = b100[:, 0::2]
    ys100 = b100[:, 1::2]

    # Within-bin positional variance — captures rapid oscillatory motion that
    # the .mean() binning erases.  Pain-related tremor (shaking, flinching,
    # writhing) at 10-15 Hz produces HIGH spread within the win100-frame window
    # even though consecutive bin means look similar.  Stronger at higher fps
    # (more raw frames per bin → better variance estimate).
    # win100 == 1 (fps < 15): variance is identically 0, no info added.
    _b100_var = _b100_raw.var(axis=1)
    xs100_var = _b100_var[:, 0::2]   # (n_bins, n_pts)
    ys100_var = _b100_var[:, 1::2]

    if head_idx is not None and tail_idx is not None:
        spine = _spine_norm_factor(xs100, ys100, head_idx, tail_idx)
        def _norm(v): return v / spine[:, None]
    else:
        def _norm(v): return v

    xs100n = _norm(xs100)
    ys100n = _norm(ys100)

    # Normalise within-bin variance by spine_length^2 (variance has units length^2)
    if head_idx is not None and tail_idx is not None:
        _spine_sq = np.maximum(spine, 10.0) ** 2
        f_withinbin = np.hstack([xs100_var / _spine_sq[:, None],
                                 ys100_var / _spine_sq[:, None]])
    else:
        f_withinbin = np.hstack([xs100_var, ys100_var])   # (n_bins, 2*n_pts)

    # ── Fine scale (50 ms) — only at 60fps+ ──────────────────────────────────
    if use_fine_scale:
        n_bins_fine = n_f // win_fine
        if n_bins_fine >= 2 and win_fine < win100:
            b_fine_raw = (xy[:n_bins_fine * win_fine]
                          .reshape(n_bins_fine, win_fine, n_xy).mean(axis=1))
            n_use = min(n_bins, n_bins_fine // 2)
            b_fine = b_fine_raw[:n_use * 2].reshape(n_use, 2, n_xy).mean(axis=1)
            if n_use < n_bins:
                b_fine = np.vstack(
                    [b_fine, np.tile(b_fine[-1:], (n_bins - n_use, 1))])
        else:
            b_fine = b100
        xs_fine = _norm(b_fine[:n_bins, 0::2])
        ys_fine = _norm(b_fine[:n_bins, 1::2])

    # ── Coarse scale (200 ms) → upsample back to 100 ms ──────────────────────
    n_bins_c = max(1, n_f // win_coarse)
    b_c_raw  = xy[:n_bins_c * win_coarse].reshape(n_bins_c, win_coarse, n_xy).mean(axis=1)
    ratio    = max(1, win_coarse // win100)
    b_c_up   = np.repeat(b_c_raw, ratio, axis=0)[:n_bins]
    if len(b_c_up) < n_bins:
        b_c_up = np.vstack([b_c_up,
                            np.tile(b_c_up[-1:], (n_bins - len(b_c_up), 1))])
    xs_coarse = _norm(b_c_up[:, 0::2])
    ys_coarse = _norm(b_c_up[:, 1::2])

    # ── Feature block: pairwise distances + velocity (+ smoothed acceleration) ─
    def _block(xs, ys, with_accel: bool = False):
        dists = [
            np.sqrt((xs[:, i] - xs[:, j]) ** 2 + (ys[:, i] - ys[:, j]) ** 2)
            for i, j in combinations(range(n_pts), 2)
        ]
        dx   = np.diff(xs, axis=0, prepend=xs[:1])
        dy   = np.diff(ys, axis=0, prepend=ys[:1])
        disp = np.sqrt(dx ** 2 + dy ** 2)
        parts = ([np.column_stack(dists)] if dists else []) + [disp]
        if with_accel:
            # 3-bin centred boxcar on velocity before differencing — reduces
            # DLC tracking jitter propagating into the acceleration signal
            k = np.ones(3) / 3.0
            if disp.shape[0] > 3:
                disp_sm = np.column_stack([
                    np.convolve(disp[:, c], k, mode="same")
                    for c in range(disp.shape[1])
                ])
            else:
                disp_sm = disp
            parts.append(np.abs(np.diff(disp_sm, axis=0, prepend=disp_sm[:1])))
        return np.hstack(parts)   # (n_bins, n_feats_per_block)

    f100    = _block(xs100n,   ys100n,   with_accel=True)
    f_coarse = _block(xs_coarse, ys_coarse, with_accel=False)

    # ── Temporal lag drift (state persistence) ────────────────────────────────
    # Normalised L2 distance between current 100ms feature vector and the
    # vector 5 bins (0.5 s) and 10 bins (1.0 s) ago.
    # LOW  = sustained state (rearing held, guarding, grooming, freeze)
    # HIGH = state just changed (onset/offset, rapid transitions)
    # Together with within-bin variance this cleanly tags pain behaviors:
    #   Shaking:       HIGH within-bin var + LOW  lag drift (stable oscillation)
    #   Flinch onset:  HIGH within-bin var + HIGH lag drift (sudden displacement)
    #   Rearing onset: LOW  within-bin var + HIGH lag drift (smooth rise)
    #   Rearing held:  LOW  within-bin var + LOW  lag drift (held posture)
    _f_norm    = f100 / (np.linalg.norm(f100, axis=1, keepdims=True) + 1e-9)
    _lag_parts = []
    for _lag in (5, 10):
        _lagged = np.vstack([_f_norm[:_lag], _f_norm[:-_lag]])
        _lag_parts.append(np.linalg.norm(_f_norm - _lagged, axis=1, keepdims=True))
    f_persist = np.hstack(_lag_parts)   # (n_bins, 2)

    # ── Angular features (body-axis curvature) ────────────────────────────────
    ang = _angular_features(xs100n, ys100n, bodyparts,
                            allow_fallback=angular_fallback)

    # ── Concatenate all blocks ────────────────────────────────────────────────
    # Within-bin variance carries no information when win100 <= 1 (fps < 15):
    # each bin holds a single frame so the variance is identically 0.  Drop the
    # all-zero block in that regime instead of feeding dead dimensions to UMAP.
    blocks = [f100]
    if win100 > 1:
        blocks.append(f_withinbin)
    blocks.append(f_persist)
    if use_fine_scale:
        f_fine = _block(xs_fine, ys_fine, with_accel=False)
        blocks.append(f_fine)
    blocks.append(f_coarse)
    if ang is not None:
        blocks.append(ang)

    feats = np.hstack(blocks)   # (n_bins, n_total_features)
    return feats.T              # (n_features, n_bins)


#  
#  UMAP
#  

def run_umap(feats_sc_T: np.ndarray, cfg: dict):
    """
    Fit UMAP on standardised features.

    Parameters
    ----------
    feats_sc_T : (n_samples, n_features)  - transposed & standardised
    cfg        : dict with umap_* keys

    Returns
    -------
    reducer   : fitted UMAP object
    embedding : (n_samples, n_components)
    """
    try:
        import umap as _umap
    except ImportError:
        raise ImportError("umap-learn is required.  pip install umap-learn")

    _umap_kwargs = dict(
        n_neighbors  = int(cfg.get("umap_n_neighbors",  60)),
        n_components = int(cfg.get("umap_n_components",  2)),
        min_dist     = float(cfg.get("umap_min_dist",  0.1)),
        random_state = int(cfg.get("umap_random_state", 42)),
        verbose      = False,
    )
    # Force single-threaded for a reproducible embedding: with random_state set
    # but n_jobs>1, umap-learn's NN-descent is still non-deterministic.  Older
    # umap-learn versions don't accept n_jobs, so fall back gracefully.
    if int(cfg.get("umap_n_jobs", 1)) == 1:
        try:
            reducer = _umap.UMAP(n_jobs=1, **_umap_kwargs)
        except TypeError:
            reducer = _umap.UMAP(**_umap_kwargs)
    else:
        reducer = _umap.UMAP(n_jobs=int(cfg.get("umap_n_jobs", 1)), **_umap_kwargs)
    return reducer, reducer.fit_transform(feats_sc_T)


#  
#  HDBSCAN  (auto-sweep min_cluster_size - B-SOiD default strategy)
#  

def run_hdbscan(embedding: np.ndarray, cfg: dict, n_total: int = None,
                log_fn=None):
    """
    Sweep min_cluster_size across a wide adaptive range and select the best
    solution using a two-mode strategy:

    target_n_clusters > 0 (user-specified target):
        Among all candidates with DBCV ≥ 75 % of the best DBCV, pick the
        solution whose cluster count is closest to the target.

    target_n_clusters == 0 (auto mode, default):
        Prefer solutions whose cluster count falls inside
        [preferred_clusters_lo, preferred_clusters_hi] (default 8–30),
        picking the highest DBCV within that range.  Falls back to the
        solution closest to the preferred range boundary when no in-range
        candidate exists.

    Both 'eom' and 'leaf' cluster selection methods are tried across the full
    sweep; DBCV (relative_validity_) measures internal cohesion and separation.

    Parameters
    ----------
    embedding : (n_samples, n_components)  UMAP embedding
    cfg       : pipeline config dict
    n_total   : full bin count before any subsampling; anchors mcs proportions.
                Falls back to embedding.shape[0].

    Returns
    -------
    best_clf    : fitted HDBSCAN object  (has .prediction_data_)
    best_labels : (n_samples,) int array - -1 = noise
    best_score  : DBCV score of the selected solution
    """
    try:
        import hdbscan as _hdb
    except ImportError:
        raise ImportError(
            "hdbscan is required.  conda install -c conda-forge hdbscan")

    # min_cluster_size is sized as a fraction of ref_n.  When UMAP runs on a
    # subsample, anchoring to the full bin count (n_total) makes the effective
    # mcs ~1/train_frac too large for the points actually being clustered, so
    # cluster granularity silently depends on the umap_full_thresh boundary.
    # "embedding" (default, v2.1) anchors to the clustered point count so the
    # proportion is honest; "full" reproduces the pre-2.1 behaviour.
    _anchor = str(cfg.get("hdbscan_mcs_anchor", "embedding")).lower()
    if _anchor == "full" and n_total is not None:
        ref_n = n_total
    else:
        ref_n = embedding.shape[0]

    # ── User preferences ──────────────────────────────────────────────────────
    target_n = int(cfg.get("target_n_clusters", 0))       # 0 = no specific target
    pref_lo  = int(cfg.get("preferred_clusters_lo", 8))   # auto-mode lower bound
    pref_hi  = int(cfg.get("preferred_clusters_hi", 30))  # auto-mode upper bound

    # ── Sweep bounds ──────────────────────────────────────────────────────────
    # pct values are in units of 0.1 % of ref_n.
    # pct=5  → mcs ≈ 0.5 % of ref_n   (finer clusters, higher counts)
    # pct=80 → mcs ≈ 8.0 % of ref_n   (coarser clusters, lower counts)
    # Default floor: 0.2% of bins (min 2), allowing brief-event clusters of ~3 bins
    # at 1200 bins (2-min, 30fps recording).  User can override via hdbscan_pct_lo.
    # hdbscan_pct_lo = 0 → auto; >0 → fixed override (units: 0.1%-of-bins steps).
    _pct_lo_auto = max(2, int(np.ceil(200.0 / ref_n)))
    pct_lo = int(cfg.get("hdbscan_pct_lo", 0)) or _pct_lo_auto
    pct_hi = int(cfg.get("hdbscan_pct_hi", 50))
    n_steps = 40

    # ── Extend sweep to finer mcs when user targets more clusters ─────────────
    # The default pct_lo (calibrated to ref_n = total bins) can be too coarse
    # when the UMAP embedding is a small subsample of the data.  If the user has
    # requested more clusters than the current lower bound can produce, push
    # pct_lo down so smaller min_cluster_size values are explored.
    _needed = target_n if target_n > 0 else pref_hi
    if _needed > 0:
        # Smallest mcs that still makes _needed clusters geometrically possible
        # from the embedding: assume ≤70% noise, each cluster needs ~6x mcs.
        _min_mcs = max(2, embedding.shape[0] // (_needed * 6))
        _ext_pct = max(1, int(np.ceil(_min_mcs * 1000.0 / ref_n)))
        if _ext_pct < pct_lo:
            pct_lo = _ext_pct
            n_steps = max(n_steps, 35)  # keep resolution across wider range
    pcts = sorted(set(
        max(pct_lo, int(round(pct_lo + (pct_hi - pct_lo) * i / (n_steps - 1))))
        for i in range(n_steps)
    ))

    _method_choice = str(cfg.get("hdbscan_method", "both")).lower().strip()
    if _method_choice in ("eom", "leaf"):
        methods = [_method_choice]
    else:
        methods = [m.strip() for m in
                   str(cfg.get("hdbscan_methods_to_try", "eom,leaf")).split(",")
                   if m.strip()]

    # ── Break exact coordinate ties before HDBSCAN ───────────────────────────
    # Flat interpolation over long tracking gaps produces identical feature
    # vectors that collapse to the same UMAP coordinates.  Exact duplicates set
    # mutual-reachability distances to zero → DBCV divides by zero → NaN for
    # every candidate.  Jitter at 1e-6 × per-axis std is imperceptible to
    # cluster geometry but eliminates the degeneracy.
    _emb_std = embedding.std(axis=0)
    _emb_std[_emb_std == 0] = 1.0          # guard against zero-variance axes
    embedding = embedding + np.random.default_rng(42).normal(
        0, 1e-4 * _emb_std, embedding.shape
    )

    # ── Sweep: collect every viable candidate ─────────────────────────────────
    # tuple: (score, n_clusters, labels, clf)
    candidates = []

    for method in methods:
        for pct in pcts:
            mcs = max(2, int(round(0.001 * pct * ref_n)))
            clf = _hdb.HDBSCAN(
                prediction_data          = True,
                min_cluster_size         = mcs,
                min_samples              = max(5, mcs // 5),
                metric                   = cfg.get("hdbscan_metric", "euclidean"),
                cluster_selection_method = method,
            ).fit(embedding)

            n_cl = len(set(clf.labels_)) - (1 if -1 in clf.labels_ else 0)
            if n_cl < 2:
                continue

            score = getattr(clf, "relative_validity_", -np.inf)
            candidates.append((score, n_cl, clf.labels_.copy(), clf))

    # ── Fallback: sweep produced nothing with ≥ 2 clusters ────────────────────
    if not candidates:
        mcs = max(2, int(round(0.001 * pct_lo * ref_n)))
        clf = _hdb.HDBSCAN(
            prediction_data          = True,
            min_cluster_size         = mcs,
            min_samples              = max(5, mcs // 5),
            metric                   = cfg.get("hdbscan_metric", "euclidean"),
            cluster_selection_method = methods[0],
        ).fit(embedding)
        return clf, clf.labels_.copy(), getattr(clf, "relative_validity_", float("nan")), "DBCV"

    best_dbcv = max(s for s, *_ in candidates)
    _score_label = "DBCV"

    # ── Degenerate-DBCV fallback ──────────────────────────────────────────────
    # relative_validity_ (DBCV) is non-finite for every candidate when the
    # mutual-reachability graph is degenerate — e.g. min_dist=0 packing,
    # duplicate embedding points, or an impoverished feature space (too few
    # bodyparts).  DBCV then cannot rank solutions and the selection below would
    # collapse to an arbitrary tie-break.  Re-score every candidate by
    # silhouette on the embedding so selection stays meaningful, and flag it.
    dbcv_degenerate = not np.isfinite(best_dbcv)
    if dbcv_degenerate:
        _score_label = "silhouette (DBCV fallback)"
        if log_fn:
            log_fn("  [VALID-WARN] DBCV is non-finite for all HDBSCAN candidates "
                   "(degenerate density graph — often too few bodyparts or "
                   "min_dist=0). Falling back to silhouette-ranked selection; "
                   "treat cluster quality for this run with caution.")
        try:
            from sklearn.metrics import silhouette_score
            _rng_sil = np.random.default_rng(42)
            _rescored = []
            for (_s, _ncl, _lbls, _clf) in candidates:
                _m = _lbls >= 0
                if _m.sum() < 2 or len(set(_lbls[_m])) < 2:
                    _rescored.append((-1.0, _ncl, _lbls, _clf))
                    continue
                _idx = np.flatnonzero(_m)
                if _idx.size > 5000:
                    _idx = _rng_sil.choice(_idx, 5000, replace=False)
                try:
                    _sil = float(silhouette_score(embedding[_idx], _lbls[_idx]))
                except Exception:
                    _sil = -1.0
                _rescored.append((_sil, _ncl, _lbls, _clf))
            candidates = _rescored
            best_dbcv = max(s for s, *_ in candidates)
        except Exception:
            pass  # sklearn unavailable; keep -inf scores, selection by diversity

    # Coefficient of variation of cluster sizes.  Solutions where clusters have
    # heterogeneous temporal footprints (brief events + sustained behaviors) are
    # biologically more realistic than uniformly-sized clusters.  A small bonus
    # prevents the DBCV-only criterion from always discarding small brief-event
    # clusters in favour of solutions where every cluster has the same density.
    def _cluster_cv(labels):
        sizes = np.array([(labels == c).sum() for c in set(labels) if c >= 0],
                         dtype=float)
        if len(sizes) < 2:
            return 0.0
        return np.std(sizes) / (np.mean(sizes) + 1e-9)

    _div_bonus   = float(cfg.get("hdbscan_diversity_bonus", 0.10))
    _dbcv_thresh = float(cfg.get("hdbscan_dbcv_thresh",    0.65))

    # ── Selection strategy ────────────────────────────────────────────────────
    if target_n > 0:
        # User-guided: pick closest to target with DBCV ≥ dbcv_thresh of best.
        thresh    = best_dbcv * _dbcv_thresh if best_dbcv > 0 else best_dbcv - 0.1
        qualified = [c for c in candidates if c[0] >= thresh] or candidates
        qualified.sort(key=lambda c: (abs(c[1] - target_n), -c[0]))
        chosen = qualified[0]
    else:
        # Auto mode: prefer solutions in [pref_lo, pref_hi].
        # Tiebreak with a small cluster-size CV bonus so solutions containing
        # both brief and sustained clusters are not unfairly penalised.
        in_range = [c for c in candidates if pref_lo <= c[1] <= pref_hi]
        if in_range:
            in_range.sort(key=lambda c: -(c[0] + _div_bonus * _cluster_cv(c[2])))
            chosen = in_range[0]
        else:
            # No candidate in preferred range — pick closest to range boundary
            # among solutions with DBCV ≥ dbcv_thresh of best.
            thresh = best_dbcv * _dbcv_thresh if best_dbcv > 0 else best_dbcv - 0.1
            boundary = sorted(
                [c for c in candidates if c[0] >= thresh],
                key=lambda c: (min(abs(c[1] - pref_lo), abs(c[1] - pref_hi)),
                               -(c[0] + _div_bonus * _cluster_cv(c[2])))
            )
            chosen = boundary[0] if boundary else \
                     sorted(candidates, key=lambda c: -c[0])[0]

    best_score, _, best_labels, best_clf = chosen
    return best_clf, best_labels, best_score, _score_label


#  
#  MLP CLASSIFIER
#  

def train_mlp(feats_sc: np.ndarray, labels: np.ndarray, cfg: dict):
    """
    Train MLP on HDBSCAN-labelled feature vectors (noise=-1 excluded).

    Returns
    -------
    clf      : fitted MLPClassifier  or  None if < 2 classes
    cv_scores: np.ndarray of CV accuracy scores
    """
    from sklearn.neural_network  import MLPClassifier
    from sklearn.model_selection import cross_val_score

    mask = labels >= 0
    X, y = feats_sc[:, mask].T, labels[mask]
    n_cl = len(np.unique(y))

    if n_cl < 2:
        return None, np.array([0.0])

    hidden = tuple(int(x) for x in
                   str(cfg.get("mlp_hidden", "100,50")).split(","))
    clf = MLPClassifier(
        hidden_layer_sizes = hidden,
        max_iter           = int(cfg.get("mlp_max_iter", 1000)),
        random_state       = int(cfg.get("umap_random_state", 42)),
    )
    clf.fit(X, y)
    k      = min(int(cfg.get("cv_folds", 5)), n_cl)
    scores = cross_val_score(clf, X, y, cv=k)
    return clf, scores


#  
#  PREDICTION (apply trained model to a new file)
#  

def predict_labels(xy_smooth: np.ndarray, _umap_model, mlp_model,
                   scaler, fps: float,
                   bodyparts: list = None,
                   body_normalise: bool = True,
                   pca_model=None,
                   min_confidence: float = 0.0,
                   angular_fallback: bool = True) -> np.ndarray:
    """
    Return per-frame integer labels for one session using the V2 feature set.
    _umap_model is kept for API / pkl compatibility; the MLP classifier
    operates directly in feature space (no UMAP transform at inference).
    pca_model, if provided, is applied after the StandardScaler and must match
    the one fitted during training.

    min_confidence : if > 0, bins where the MLP's top class probability is below
        this threshold are labelled -1 (unclassified) instead of being forced
        into the nearest cluster.  Important because HDBSCAN noise (often a large
        fraction of bins) is excluded from training but would otherwise be
        force-classified at inference.  0 = legacy behavior (always assign).
    """
    feats  = extract_features_v2(xy_smooth, fps, bodyparts,
                                  body_normalise=body_normalise,
                                  angular_fallback=angular_fallback)   # (n_feat, n_bins)
    scaled = scaler.transform(feats.T)                        # (n_bins, n_feat)
    if pca_model is not None:
        scaled = pca_model.transform(scaled)                  # (n_bins, n_pca)
    labels = mlp_model.predict(scaled)                        # (n_bins,)
    if min_confidence and min_confidence > 0 and hasattr(mlp_model, "predict_proba"):
        try:
            proba = mlp_model.predict_proba(scaled)
            labels = np.where(proba.max(axis=1) < float(min_confidence),
                              -1, labels)
        except Exception:
            pass
    win    = max(1, int(round(fps / 10)))
    fl     = np.repeat(labels, win)
    n_orig = xy_smooth.shape[0]
    if len(fl) < n_orig:
        fl = np.pad(fl, (0, n_orig - len(fl)), mode="edge")
    return fl[:n_orig].astype(int)


# ──────────────────────────────────────────────────────────────────────────────
#  HMM SMOOTHING  (post-hoc Multinomial HMM wrapper for B-SOiD predictions)
# ──────────────────────────────────────────────────────────────────────────────


def train_hmm(label_sequences: list, n_clusters: int,
              n_states: int = None, n_iter: int = 100, log_fn=None):
    """Fit a Multinomial (Categorical) HMM to B-SOiD MLP label sequences.

    Uses Baum-Welch EM.  n_states defaults to n_clusters (smoothing-only mode).

    Emission initialisation strategy
    ---------------------------------
    When n_states == n_clusters (smoothing-only mode) the emission matrix is
    seeded as a near-diagonal (identity-like) matrix with a small off-diagonal
    probability eps=0.05.  This anchors Baum-Welch so that state i learns to
    represent cluster i rather than converging to a degenerate permutation.
    After fitting, states are realigned to clusters via the Hungarian algorithm
    (scipy.optimize.linear_sum_assignment on the emission matrix) so the
    returned state IDs exactly match the original B-SOiD cluster IDs — keeping
    the analyser cluster→behaviour mapping valid.

    When n_states < n_clusters (macro-state discovery) a uniform Dirichlet
    initialisation is used; state IDs are arbitrary macro-state indices and
    the original cluster mapping no longer applies directly.

    Returns a fitted hmmlearn.hmm.CategoricalHMM.
    """
    try:
        from hmmlearn.hmm import CategoricalHMM
    except ImportError:
        raise ImportError(
            "hmmlearn is required for HMM smoothing.  "
            "Install it with:  pip install hmmlearn>=0.3.2")
    if n_states is None:
        n_states = n_clusters

    smoothing_mode = (n_states == n_clusters)

    # Build emission matrix BEFORE model construction so we can pass it in.
    # init_params excludes 'e' to prevent hmmlearn from overwriting our matrix.
    if smoothing_mode:
        # Near-diagonal: P(obs=j | state=i) ≈ 0.95 if i==j, 0.05/(k-1) else
        eps = 0.05
        emis = np.full((n_states, n_clusters), eps / max(1, n_clusters - 1))
        np.fill_diagonal(emis, 1.0 - eps)
        emis /= emis.sum(axis=1, keepdims=True)  # normalise (already sums to 1)
        _ip = "st"   # randomly init start + transition; we supply emission
    else:
        # Macro-state mode: uniform Dirichlet, all states/clusters equally likely
        rng  = np.random.default_rng(42)
        emis = rng.dirichlet(np.ones(n_clusters), size=n_states)
        _ip  = "st"

    model = CategoricalHMM(
        n_components=n_states,
        n_iter=n_iter,
        tol=1e-4,
        init_params=_ip,   # 's' + 't' randomised; 'e' we set manually
        params="ste",      # all params updated during EM
    )
    model.emissionprob_ = emis   # set BEFORE fit so hmmlearn validates shape

    X       = np.concatenate([s.reshape(-1, 1).astype(int) for s in label_sequences])
    lengths = [len(s) for s in label_sequences]
    model.fit(X, lengths)

    # ── State alignment (smoothing-only mode) ────────────────────────────────
    # After Baum-Welch the emission matrix may have permuted rows.  Use the
    # Hungarian algorithm to find the bijective assignment of states → clusters
    # that maximises total emission probability on the diagonal, then permute
    # all model parameters so state i ↔ cluster i.
    # Default alignment flags (pickled with the model so downstream consumers
    # and the analyser can tell whether state IDs == cluster IDs).
    model.cube_smoothing_mode = bool(smoothing_mode)
    model.cube_aligned        = False
    model.cube_emission_diag  = float("nan")
    if smoothing_mode:
        try:
            from scipy.optimize import linear_sum_assignment
            # cost[i,j] = −P(obs=j | state=i); minimise → maximise probability
            _, col_ind = linear_sum_assignment(-model.emissionprob_)
            # col_ind[old_state_i] = cluster that best matches old_state_i
            # perm[new_state_j]  = old_state whose best cluster is j
            perm = np.argsort(col_ind)
            model.startprob_   = model.startprob_[perm]
            model.transmat_    = model.transmat_[np.ix_(perm, perm)]
            model.emissionprob_ = model.emissionprob_[perm]
            model.cube_aligned = True
            # Alignment quality: mean diagonal emission after permutation.  A low
            # value means states do not map cleanly onto clusters (the smoothing
            # assumption is weak for this data), which downstream cluster→behaviour
            # mappings rely on — surface it rather than assuming perfect alignment.
            _diag = float(np.mean(np.diag(model.emissionprob_)))
            model.cube_emission_diag = _diag
            if log_fn and _diag < 0.5:
                log_fn(f"  [VALID-WARN] HMM state↔cluster alignment is weak "
                       f"(mean diagonal emission {_diag:.2f} < 0.5). Smoothed "
                       f"state IDs may not correspond cleanly to cluster IDs.")
        except ImportError:
            # scipy is normally present (hdbscan depends on it).  If it is not,
            # state IDs are NOT aligned to cluster IDs and the analyser's
            # cluster→behaviour mapping would silently break — warn loudly.
            if log_fn:
                log_fn("  [VALID-WARN] scipy unavailable — HMM states were NOT "
                       "aligned to cluster IDs (Hungarian assignment skipped). "
                       "Smoothed _hmm labels may not match cluster IDs; install "
                       "scipy for deterministic alignment.")

    return model


def decode_hmm(hmm_model, frame_labels: np.ndarray) -> np.ndarray:
    """Viterbi decode: returns (n_frames,) int array of HMM state IDs."""
    _, state_seq = hmm_model.decode(
        frame_labels.reshape(-1, 1).astype(int), algorithm="viterbi")
    return state_seq.astype(int)


def plot_duration_comparison(raw_labels: np.ndarray, hmm_labels: np.ndarray,
                              fps: float, out_path: Path):
    """Bout duration distributions — raw B-SOiD vs HMM-smoothed.

    Three panels: the two separate log-log histograms (kept for continuity) and
    an overlaid panel so the disappearance of the single-frame spike after
    smoothing is directly visible, with per-condition median markers.
    """
    def _durations(labels):
        bouts = labels_to_bouts(labels)
        return bouts["Run lengths"].values / fps

    raw_dur = _durations(raw_labels)
    hmm_dur = _durations(hmm_labels)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), facecolor=_BG)
    for ax in axes:
        _dark_ax(ax)

    all_dur = np.concatenate([raw_dur, hmm_dur])
    lo   = max(1e-3, float(all_dur.min()))
    hi   = float(all_dur.max()) + 0.1
    bins = np.logspace(np.log10(lo), np.log10(hi), 40)
    one_frame = 1.0 / fps
    _RAW_C, _HMM_C = "#F28E2B", "#4E79A7"

    for ax, durs, title, col in zip(
            axes[:2],
            [raw_dur, hmm_dur],
            ["Raw B-SOiD  (MLP output)", "HMM-smoothed  (Viterbi)"],
            [_RAW_C, _HMM_C]):
        ax.hist(durs, bins=bins, color=col, edgecolor=_BG, alpha=0.85)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Bout duration (s)")
        ax.set_ylabel("Count (log)")
        ax.set_title(title)
        ax.axvline(one_frame, color="#ff4081", linestyle="--",
                   linewidth=1.2, label=f"1 frame ({one_frame:.3f} s)")
        if len(durs):
            ax.axvline(float(np.median(durs)), color=col, linestyle=":",
                       linewidth=1.6, label=f"median {np.median(durs):.2f}s")
        ax.legend(fontsize=7, facecolor=_PANEL, labelcolor=_TEXT_COL)

    # Overlaid panel — the headline comparison.
    axo = axes[2]
    axo.hist(raw_dur, bins=bins, color=_RAW_C, alpha=0.5,
             label=f"Raw (median {np.median(raw_dur):.2f}s)" if len(raw_dur) else "Raw")
    axo.hist(hmm_dur, bins=bins, color=_HMM_C, alpha=0.5,
             label=f"HMM (median {np.median(hmm_dur):.2f}s)" if len(hmm_dur) else "HMM")
    axo.set_xscale("log"); axo.set_yscale("log")
    axo.set_xlabel("Bout duration (s)"); axo.set_ylabel("Count (log)")
    axo.set_title("Overlay  (raw vs HMM)")
    axo.axvline(one_frame, color="#ff4081", linestyle="--", linewidth=1.2,
                label=f"1 frame ({one_frame:.3f} s)")
    axo.legend(fontsize=7, facecolor=_PANEL, labelcolor=_TEXT_COL)

    fig.suptitle("Behavioral bout duration  —  before vs. after HMM smoothing",
                 color=_TEXT_COL, fontsize=12)
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_hmm_transition_matrix(hmm_model, out_path: Path,
                                state_names: list = None):
    """Heatmaps of the HMM learned transition matrix (transmat_).

    Left  : full matrix.  The diagonal (self-persistence) is partly imposed by
            the near-diagonal emission prior + Baum-Welch on the same labels, so
            it should not be read as a purely data-driven quantity.
    Right : off-diagonal only (diagonal zeroed, rows renormalised) so the
            behavioural 'grammar' — which state tends to follow which — is
            readable without the dominant diagonal saturating the colourmap.

    The chance line (1/(n-1)) is a visualisation reference, NOT a significance
    test.
    """
    A = hmm_model.transmat_
    n = A.shape[0]
    names = state_names or [f"S{i}" for i in range(n)]
    chance_floor = 1.0 / max(1, n - 1)
    cell_fs = max(5, 9 - n // 4)

    cmap_hmm = plt.cm.Blues.copy()
    cmap_hmm.set_bad(color=_PANEL)

    sz = max(6, n * 0.6 + 2)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(2 * sz, sz), facecolor=_BG)

    # ── Left: full matrix, below-chance off-diagonal masked ───────────────────
    _dark_ax(ax)
    display = A.copy().astype(float)
    for i in range(n):
        for j in range(n):
            if i != j and display[i, j] <= chance_floor:
                display[i, j] = np.nan
    im = ax.imshow(display, cmap=cmap_hmm, aspect="auto",
                   vmin=chance_floor, vmax=1.0)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=_TICK_COL)
    cb.set_label("Transition probability  (above-chance range)", color=_TICK_COL)
    for _a in (ax, ax2):
        _a.set_xticks(range(n))
        _a.set_xticklabels(names, rotation=45, ha="right",
                           color=_TICK_COL, fontsize=8)
        _a.set_yticks(range(n))
        _a.set_yticklabels(names, color=_TICK_COL, fontsize=8)
        _a.set_xlabel("State at t+1", color=_TICK_COL)
        _a.set_ylabel("State at t", color=_TICK_COL)
    ax.set_title(f"Full A[i→j]  (diagonal = self-persistence, partly prior-driven)\n"
                 f"off-diagonal ≤ chance ({chance_floor:.3f}) masked",
                 color=_TEXT_COL, fontsize=9)
    for i in range(n):
        for j in range(n):
            val = A[i, j]
            if i == j or val > chance_floor:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=cell_fs,
                        color="white" if val > 0.55 else _TICK_COL)

    # ── Right: off-diagonal grammar (diagonal removed, rows renormalised) ──────
    _dark_ax(ax2)
    off = A.copy().astype(float)
    np.fill_diagonal(off, 0.0)
    rs  = off.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    off_norm = off / rs
    vmax2 = float(np.nanmax(off_norm)) if np.isfinite(off_norm).any() else 1.0
    im2 = ax2.imshow(off_norm, cmap=cmap_hmm, aspect="auto",
                     vmin=0.0, vmax=max(vmax2, 1e-3))
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(colors=_TICK_COL)
    cb2.set_label("P(next state | a transition occurs)", color=_TICK_COL)
    ax2.set_title("Off-diagonal grammar  (self-transitions removed,\n"
                  "rows renormalised to next-state probability)",
                  color=_TEXT_COL, fontsize=9)
    for i in range(n):
        for j in range(n):
            if i != j and off_norm[i, j] > 0.10:
                ax2.text(j, i, f"{off_norm[i, j]:.2f}", ha="center", va="center",
                         fontsize=cell_fs,
                         color="white" if off_norm[i, j] > 0.55 * max(vmax2, 1e-3)
                         else _TICK_COL)

    plt.tight_layout()
    _savefig(fig, out_path)


def plot_dual_ethogram(raw_labels: np.ndarray, hmm_labels: np.ndarray,
                        fps: float, out_path: Path, tag: str,
                        cluster_names: dict = None):
    """Two-row ethogram: row 1 = raw B-SOiD MLP, row 2 = HMM Viterbi.

    cluster_names : optional {cluster_id: name} for the raw row's y-labels;
        falls back to C<id> when missing (safe to pass or omit).
    """
    uniq_raw = np.unique(raw_labels)
    uniq_hmm = np.unique(hmm_labels)
    t = np.arange(len(raw_labels)) / fps

    n_raw = len(uniq_raw)
    n_hmm = len(uniq_hmm)

    def _rawlabel(l):
        if cluster_names and int(l) in cluster_names and cluster_names[int(l)]:
            return f"C{l} {cluster_names[int(l)]}"
        return f"C{l}"

    fig, (ax_raw, ax_hmm) = plt.subplots(
        2, 1,
        figsize=(14, max(4, (n_raw + n_hmm) * 0.35 + 2)),
        facecolor=_BG, sharex=True)
    _dark_ax(ax_raw)
    _dark_ax(ax_hmm)

    for idx_u, lbl in enumerate(uniq_raw):
        sel = np.where(raw_labels == lbl)[0]
        ax_raw.scatter(t[sel], np.full(len(sel), idx_u),
                       c=_cmap(int(lbl)), s=8, marker="|", linewidths=3.5)
    ax_raw.set_yticks(range(n_raw))
    ax_raw.set_yticklabels([_rawlabel(l) for l in uniq_raw], color=_TEXT_COL, fontsize=7)
    ax_raw.set_title(f"Raw B-SOiD  |  {tag}", color=_TEXT_COL, fontsize=9)
    ax_raw.set_ylabel("Cluster", color=_TICK_COL, fontsize=8)

    for idx_u, lbl in enumerate(uniq_hmm):
        sel = np.where(hmm_labels == lbl)[0]
        ax_hmm.scatter(t[sel], np.full(len(sel), idx_u),
                       c=_cmap(int(lbl)), s=8, marker="|", linewidths=3.5)
    ax_hmm.set_yticks(range(n_hmm))
    ax_hmm.set_yticklabels([f"S{l}" for l in uniq_hmm], color=_TEXT_COL, fontsize=7)
    ax_hmm.set_title("HMM Viterbi (state-aligned)", color=_TEXT_COL, fontsize=9)
    ax_hmm.set_xlabel("Time (s)", color=_TICK_COL, fontsize=8)
    ax_hmm.set_ylabel("State", color=_TICK_COL, fontsize=8)

    plt.tight_layout()
    _savefig(fig, out_path)


def plot_syntax_network(hmm_model, out_path: Path,
                         state_names: list = None, min_prob: float = 0.05):
    """
    Publication-quality directed behavioral syntax graph.
    Left panel: spring-layout directed network with circular arc arrows.
    Right panel: chord (circular) diagram showing directional transition flow.
    Node size ∝ stationary probability · Edge/ribbon width ∝ transition probability.
    """
    try:
        import networkx as nx
    except ImportError:
        return

    A  = hmm_model.transmat_
    n  = A.shape[0]
    names = state_names or [f"S{i}" for i in range(n)]

    # Only show edges strictly above the chance level for this many states
    chance_floor  = 1.0 / max(1, n - 1)
    effective_min = max(min_prob, chance_floor)

    # Stationary distribution via power iteration
    pi = np.ones(n) / n
    for _ in range(500):
        pi = pi @ A
    pi /= pi.sum()

    # ── Build directed graph ──────────────────────────────────────────────────
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(names[i], weight=float(pi[i]))
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] > effective_min:
                G.add_edge(names[i], names[j], weight=float(A[i, j]))

    if G.number_of_edges() == 0:
        return

    node_colors = [_cmap(i) for i in range(n)]
    node_sizes  = [max(600, float(pi[i]) * 12000) for i in range(n)]

    # ── Figure: two panels ───────────────────────────────────────────────────
    fw = max(16, n * 0.9)
    fh = max(8,  n * 0.7)
    fig, (ax_net, ax_chord) = plt.subplots(
        1, 2, figsize=(fw, fh), facecolor=_BG)
    _dark_ax(ax_net)
    _dark_ax(ax_chord)
    ax_net.set_facecolor(_PANEL)
    ax_chord.set_facecolor(_PANEL)

    # ── Left: directed network ────────────────────────────────────────────────
    pos = nx.circular_layout(G)

    # Draw edges with width & alpha ∝ probability; dark contrasting color
    edges     = list(G.edges(data=True))
    max_wt    = max((d["weight"] for _, _, d in edges), default=1.0)
    # Normalise over the above-chance range so the weakest shown edge (just
    # above effective_min) maps to 0 and the strongest maps to 1.
    wt_range  = max(max_wt - effective_min, 1e-9)
    edge_list = [(u, v) for u, v, _ in edges]
    e_widths  = [max(1.2, (d["weight"] - effective_min) / wt_range * 12)
                 for _, _, d in edges]
    e_alphas  = [0.55 + (d["weight"] - effective_min) / wt_range * 0.45
                 for _, _, d in edges]
    e_colors  = [_cmap(list(G.nodes).index(u)) for u, _, _ in edges]

    # Draw each edge individually so alpha can vary
    for (u, v), ew, ea, ec in zip(edge_list, e_widths, e_alphas, e_colors):
        nx.draw_networkx_edges(
            G, pos, ax=ax_net,
            edgelist=[(u, v)],
            width=ew,
            edge_color=[ec],
            alpha=ea,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=max(14, ew * 2),
            connectionstyle="arc3,rad=0.18",
            min_source_margin=22,
            min_target_margin=22,
        )

    nx.draw_networkx_nodes(
        G, pos, ax=ax_net,
        node_size=node_sizes,
        node_color=node_colors,
        linewidths=1.8,
        edgecolors=_BG,
        alpha=0.95,
    )
    font_sz = max(6, 11 - n // 5)
    nx.draw_networkx_labels(G, pos, ax=ax_net,
                             font_color=_TEXT_COL,
                             font_size=font_sz,
                             font_weight="bold")
    ax_net.set_title(
        f"Behavioral Syntax Network  (p > {effective_min:.3f} = above chance)\n"
        "Node size ∝ stationary probability  ·  Arrow width ∝ transition probability",
        color=_TEXT_COL, fontsize=10, fontweight="bold", pad=10)
    ax_net.axis("off")

    # ── Right: chord (circular) diagram ──────────────────────────────────────
    theta  = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    cx     = np.cos(theta)
    cy     = np.sin(theta)
    margin = 1.55

    # Draw arcs as annotate arrows between node positions
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = float(A[i, j])
            if p <= effective_min:
                continue
            p_norm = (p - effective_min) / wt_range
            lw     = max(0.6, p_norm * 10)
            alpha  = 0.45 + p_norm * 0.55
            rad    = 0.20 if abs(i - j) > n // 3 else 0.12
            ax_chord.annotate(
                "", xy=(cx[j], cy[j]), xytext=(cx[i], cy[i]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=_cmap(i),
                    alpha=alpha, lw=lw,
                    connectionstyle=f"arc3,rad={rad}",
                    mutation_scale=max(10, lw * 2.5),
                ),
                zorder=2 + int(p_norm * 5),
            )

    # Node dots on chord ring
    nsize = 120 + 280 * (pi / pi.max())
    ax_chord.scatter(cx, cy, s=nsize, c=node_colors,
                     zorder=9, edgecolors=_BG, linewidths=1.4, alpha=0.96)
    for i, nm in enumerate(names):
        ox = cx[i] * margin
        oy = cy[i] * margin
        ha = "left" if cx[i] > 0.05 else ("right" if cx[i] < -0.05 else "center")
        ax_chord.text(ox, oy, nm, ha=ha, va="center",
                      fontsize=font_sz, color=_TEXT_COL,
                      fontweight="bold")

    ax_chord.set_xlim(-2.0, 2.0)
    ax_chord.set_ylim(-2.0, 2.0)
    ax_chord.set_aspect("equal")
    ax_chord.axis("off")
    ax_chord.set_title(
        "Chord Diagram  (directional transitions)\n"
        "Arc colour = source state  ·  Arc width ∝ transition probability",
        color=_TEXT_COL, fontsize=10, fontweight="bold", pad=10)

    plt.tight_layout(pad=2.0)
    _savefig(fig, out_path)


def plot_dwell_violin(epochs: "pd.DataFrame", out_path: Path, tag: str = ""):
    """
    Violin + strip plots of dwell-time distributions per behavioral state.
    Shows the full distribution shape rather than just mean ± SD.
    Each violin is colored by cluster ID.
    """
    if epochs is None or epochs.empty:
        return
    uniq = sorted(epochs["label"].unique())
    n    = len(uniq)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9 + 2), 5), facecolor=_BG)
    _dark_ax(ax)

    data   = [epochs.loc[epochs["label"] == lbl, "duration_sec"].values
              for lbl in uniq]
    colors = [_cmap(int(lbl)) for lbl in uniq]

    parts = ax.violinplot(data, positions=range(n),
                          showmedians=True, showextrema=True)
    parts["cmedians"].set_color("#ffd60a")
    parts["cmins"].set_color(_TICK_COL)
    parts["cmaxes"].set_color(_TICK_COL)
    parts["cbars"].set_color(_TICK_COL)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor(_BG)
        pc.set_alpha(0.72)

    # Overlay raw data as a strip (jittered dots)
    rng = np.random.default_rng(42)
    for i, d in enumerate(data):
        if len(d) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(d))
        ax.scatter(i + jitter, d,
                   color=colors[i], alpha=0.35, s=6, linewidths=0,
                   zorder=3)

    ax.set_xticks(range(n))
    ax.set_xticklabels([f"S{int(l)}" for l in uniq],
                       color=_TEXT_COL, fontsize=max(6, 10 - n // 6))
    ax.set_ylabel("Dwell time (s)", color=_TICK_COL, fontsize=10)
    title = f"Dwell-time distributions per state  –  {tag}" if tag else \
            "Dwell-time distributions per state"
    ax.set_title(title, color=_TEXT_COL, fontsize=11, fontweight="bold")

    plt.tight_layout()
    _savefig(fig, out_path)


def plot_sankey_sequences(all_frame_labels: list, out_path: Path,
                           n_steps: int = 5):
    """
    Sankey (alluvial) diagram: state occupancy at consecutive sequence positions.
    Each column shows the proportion of frames in each state at step k.
    Bezier ribbons connect same-state or transitioning populations between steps,
    illustrating how animals flow from state to state across a bout sequence.
    """
    import matplotlib.patches as mpatches_local
    from matplotlib.path import Path as MPath

    if not all_frame_labels:
        return

    # Build transition sequences: for each session extract label at every step
    # We work at bout level: take the nth bout label for each session
    all_bout_seqs = []
    for fl in all_frame_labels:
        if len(fl) == 0:
            continue
        bouts_obj = labels_to_bouts(np.asarray(fl))
        seq = bouts_obj["B-SOiD labels"].values
        if len(seq) >= 2:
            all_bout_seqs.append(seq)

    if not all_bout_seqs:
        return

    uniq_states = sorted({int(l) for seq in all_bout_seqs for l in seq})
    ns = len(uniq_states)
    si = {s: i for i, s in enumerate(uniq_states)}
    n_steps_use = min(n_steps, min(len(s) for s in all_bout_seqs))
    if n_steps_use < 2:
        return

    # Count state occupancy at each step position
    counts = np.zeros((n_steps_use, ns), dtype=float)
    for seq in all_bout_seqs:
        for k in range(n_steps_use):
            counts[k, si[int(seq[k])]] += 1
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    props  = counts / totals    # (n_steps, ns)  — row-stochastic fractions

    # Transition counts between adjacent steps
    trans = np.zeros((n_steps_use - 1, ns, ns), dtype=float)
    for seq in all_bout_seqs:
        for k in range(n_steps_use - 1):
            a, b = si[int(seq[k])], si[int(seq[k + 1])]
            trans[k, a, b] += 1
    for k in range(n_steps_use - 1):
        row_s = trans[k].sum(axis=1, keepdims=True)
        row_s[row_s == 0] = 1.0
        trans[k] /= row_s

    # ── Draw ─────────────────────────────────────────────────────────────────
    col_x  = np.linspace(0, 1, n_steps_use)
    col_w  = 0.05
    bar_h  = 0.88   # total height used by state bars in each column
    gap    = 0.006  # inter-state gap

    fig, ax = plt.subplots(figsize=(max(10, n_steps_use * 2.5), 7), facecolor=_BG)
    _dark_ax(ax)

    # Compute y-positions for each (step, state) bar
    y_bot = np.zeros((n_steps_use, ns), dtype=float)
    y_top = np.zeros((n_steps_use, ns), dtype=float)
    for k in range(n_steps_use):
        cum = 0.06   # start slightly above bottom
        for si_j in range(ns):
            h = props[k, si_j] * bar_h
            y_bot[k, si_j] = cum
            y_top[k, si_j] = cum + h
            cum += h + gap

    colors = [_cmap(s) for s in uniq_states]

    # Draw bezier ribbons first (behind bars)
    for k in range(n_steps_use - 1):
        x0 = col_x[k]  + col_w
        x1 = col_x[k + 1]
        cx0 = x0 + (x1 - x0) * 0.40
        cx1 = x0 + (x1 - x0) * 0.60

        # Offsets within source bar for each destination
        src_offsets  = np.zeros(ns, dtype=float)
        dst_offsets  = np.zeros(ns, dtype=float)

        for a in range(ns):
            src_h = y_top[k, a] - y_bot[k, a]
            for b in range(ns):
                p = float(trans[k, a, b])
                if p < 0.005:
                    continue
                ribbon_h_src = src_h * p
                ribbon_h_dst = (y_top[k + 1, b] - y_bot[k + 1, b]) * p

                ys0 = y_bot[k, a]  + src_offsets[a]
                ye0 = ys0 + ribbon_h_src
                ys1 = y_bot[k + 1, b] + dst_offsets[b]
                ye1 = ys1 + ribbon_h_dst

                src_offsets[a] += ribbon_h_src
                dst_offsets[b] += ribbon_h_dst

                verts = [
                    (x0, ys0), (cx0, ys0), (cx1, ys1), (x1, ys1),
                    (x1, ye1), (cx1, ye1), (cx0, ye0), (x0, ye0),
                    (x0, ys0),
                ]
                codes = (
                    [MPath.MOVETO] +
                    [MPath.CURVE4] * 3 +
                    [MPath.LINETO] +
                    [MPath.CURVE4] * 3 +
                    [MPath.CLOSEPOLY]
                )
                path   = MPath(verts, codes)
                patch  = mpatches_local.PathPatch(
                    path, facecolor=colors[a],
                    edgecolor="none", alpha=0.38, zorder=1)
                ax.add_patch(patch)

    # Draw state bars on top of ribbons
    for k in range(n_steps_use):
        for si_j in range(ns):
            h = y_top[k, si_j] - y_bot[k, si_j]
            if h < 1e-4:
                continue
            rect = mpatches_local.FancyBboxPatch(
                (col_x[k], y_bot[k, si_j]), col_w, h,
                boxstyle="square,pad=0",
                facecolor=colors[si_j], edgecolor=_BG,
                linewidth=0.5, zorder=3)
            ax.add_patch(rect)
            if h > 0.04:
                ax.text(col_x[k] + col_w / 2,
                        y_bot[k, si_j] + h / 2,
                        f"S{uniq_states[si_j]}",
                        ha="center", va="center",
                        fontsize=max(5, 9 - ns // 6),
                        color=_TEXT_COL, fontweight="bold", zorder=4)

    # Column labels
    for k, x in enumerate(col_x):
        ax.text(x + col_w / 2, 0.02, f"Step {k + 1}",
                ha="center", va="bottom", fontsize=9,
                color=_TICK_COL, fontweight="bold")

    # Legend patches
    legend_handles = [
        mpatches_local.Patch(color=colors[i], label=f"S{uniq_states[i]}")
        for i in range(ns)
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=8, facecolor=_PANEL,
              labelcolor=_TEXT_COL, title="State",
              title_fontsize=8, framealpha=0.7)

    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(0, 1.02)
    ax.set_title(
        f"Behavioral sequence Sankey diagram  (first {n_steps_use} bout positions)\n"
        "Bar height = state occupancy  ·  Ribbon width = transition flow",
        color=_TEXT_COL, fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_state_space_trajectory(embedding: np.ndarray,
                                  frame_labels: np.ndarray,
                                  fps: float,
                                  out_path: Path,
                                  tag: str = "",
                                  max_traj_frames: int = 3000):
    """
    2-D UMAP (or other embedding) scatter coloured by behavioral state with a
    temporal trajectory overlay.  The trajectory is sub-sampled to
    *max_traj_frames* for readability.
    Left panel: density-coloured scatter.
    Right panel: trajectory path through state space (time colour-mapped).
    """
    if embedding is None or len(embedding) == 0:
        return
    emb = np.asarray(embedding)
    lbl = np.asarray(frame_labels)
    if emb.shape[0] != len(lbl) or emb.shape[1] < 2:
        return

    uniq   = sorted(int(u) for u in np.unique(lbl) if u >= 0)
    colors = {u: _cmap(u) for u in uniq}

    fig, (ax_sc, ax_tr) = plt.subplots(1, 2, figsize=(16, 7), facecolor=_BG)
    for ax in (ax_sc, ax_tr):
        _dark_ax(ax)
        ax.set_facecolor(_PANEL)

    # ── Left: scatter coloured by state ──────────────────────────────────────
    valid = lbl >= 0
    if valid.any():
        for u in uniq:
            mask = valid & (lbl == u)
            if mask.any():
                ax_sc.scatter(emb[mask, 0], emb[mask, 1],
                              s=2, alpha=0.45, color=colors[u],
                              linewidths=0, label=f"S{u}")
        # Noise in gray
        noise = ~valid
        if noise.any():
            ax_sc.scatter(emb[noise, 0], emb[noise, 1],
                          s=1, alpha=0.15, color="#555566", linewidths=0)

    handles = [mpatches.Patch(color=colors[u], label=f"S{u}") for u in uniq[:24]]
    ax_sc.legend(handles=handles, fontsize=6, ncol=3,
                 facecolor=_PANEL, labelcolor=_TEXT_COL,
                 loc="upper right", framealpha=0.7)
    ax_sc.set_title(
        f"State-space scatter  –  {tag}" if tag else "State-space scatter",
        color=_TEXT_COL, fontsize=10, fontweight="bold")
    ax_sc.set_xlabel("Dim 1", color=_TICK_COL, fontsize=9)
    ax_sc.set_ylabel("Dim 2", color=_TICK_COL, fontsize=9)

    # ── Right: temporal trajectory ────────────────────────────────────────────
    n_frames = len(emb)
    step     = max(1, n_frames // max_traj_frames)
    idx      = np.arange(0, n_frames, step)
    t_norm   = idx / max(1, n_frames - 1)   # 0→1 time normalised

    # Draw trajectory line with colour mapped to time (purple→yellow)
    from matplotlib.collections import LineCollection as _LC
    cmap_traj = plt.cm.plasma
    pts   = emb[idx, :2].reshape(-1, 1, 2)
    segs  = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc    = _LC(segs, cmap=cmap_traj, linewidth=0.7,
                alpha=0.55, capstyle="round")
    lc.set_array(t_norm[:-1])
    lc.set_clim(0, 1)
    ax_tr.add_collection(lc)
    ax_tr.autoscale_view()

    # Scatter state identity (sub-sampled) on top
    for u in uniq:
        mask = lbl[idx] == u
        if mask.any():
            ax_tr.scatter(emb[idx[mask], 0], emb[idx[mask], 1],
                          s=4, alpha=0.65, color=colors[u],
                          linewidths=0, zorder=2)

    # Colorbar for time
    sm = plt.cm.ScalarMappable(cmap=cmap_traj,
                                norm=plt.Normalize(vmin=0, vmax=n_frames / fps))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_tr, shrink=0.75, pad=0.02)
    cb.ax.tick_params(colors=_TICK_COL, labelsize=7)
    cb.set_label("Time (s)", color=_TICK_COL, fontsize=8)
    cb.outline.set_edgecolor(_PANEL)

    ax_tr.set_title("Temporal trajectory through state space\n(colour = time)",
                    color=_TEXT_COL, fontsize=10, fontweight="bold")
    ax_tr.set_xlabel("Dim 1", color=_TICK_COL, fontsize=9)
    ax_tr.set_ylabel("Dim 2", color=_TICK_COL, fontsize=9)

    title = f"Continuous State-Space Projection  –  {tag}" if tag else \
            "Continuous State-Space Projection"
    fig.suptitle(title, color=_TEXT_COL, fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_path)


#
#  BOUT / EPOCH CONVERSION
#  



#
#  VALIDATION LAYER  (B-SOiD framework — Version 2)
#
#  Gates reference: Hsu & Bhatt et al., 2021 + CUBE V2 Framework spec.
#  All functions return a dict with keys: stage, status, warnings, <metric>.
#  status values: "pass" | "warn" | "block"
#  "block" means the result is flagged as INVALID — pipeline continues but
#  the validation_report.json records the failure for the user to review.
#

def validate_dlc_quality(dlc_paths: list,
                          likelihood_thresh: float = 0.3) -> dict:
    """
    Stage: DLC output quality.
    Warns if any bodypart has median likelihood < threshold AND the fraction
    of frames below threshold exceeds 20 % (per B-SOiD V2 spec).
    """
    report: dict = {"stage": "dlc_quality", "status": "pass",
                    "warnings": [], "sessions": {}}
    for fp in dlc_paths:
        try:
            ext = Path(str(fp)).suffix.lower()
            if ext in (".h5", ".hdf5"):
                df = pd.read_hdf(str(fp))
            else:
                with open(fp, encoding="utf-8", errors="replace") as fh:
                    head = [fh.readline() for _ in range(5)]
                n_lv = max(sum(1 for l in head[:4]
                               if l.strip() and not l.strip()[0].isdigit()), 2)
                df = pd.read_csv(fp, header=list(range(n_lv)), index_col=0)
            df  = _normalise_dlc_df(df)
            bad = []
            for bp in df.columns.get_level_values("bodyparts").unique():
                sub = df.xs(bp, level="bodyparts", axis=1)
                if sub.columns.nlevels > 1:
                    sub.columns = sub.columns.get_level_values(-1)
                ll   = pd.to_numeric(
                    sub.get("likelihood",
                            pd.Series(np.nan, index=df.index)),
                    errors="coerce")
                med  = float(ll.median())
                frac = float((ll < likelihood_thresh).mean())
                if med < likelihood_thresh and frac > 0.20:
                    bad.append({"bodypart": bp,
                                "median_ll": round(med, 3),
                                "frac_below": round(frac, 3)})
            if bad:
                report["warnings"].append(
                    f"{Path(fp).name}: {len(bad)} bodypart(s) below "
                    f"likelihood threshold (median < {likelihood_thresh}, "
                    f"> 20 % of frames)")
                if report["status"] == "pass":
                    report["status"] = "warn"
            report["sessions"][str(fp)] = {"bad_bodyparts": bad}
        except Exception as e:
            report["warnings"].append(f"Could not validate {fp}: {e}")
    return report


def validate_feature_consistency(all_feats: list, names: list) -> dict:
    """
    Stage: feature extraction consistency.
    Warns if cosine similarity between any pair of session mean-feature
    vectors is < 0.5 (indicates data quality mismatch across recordings).
    """
    from sklearn.metrics.pairwise import cosine_similarity
    report: dict = {"stage": "feature_consistency", "status": "pass",
                    "warnings": [], "min_similarity": None,
                    "sessions": list(names)}
    if len(all_feats) < 2:
        return report
    means = np.array([f.mean(axis=1) for f in all_feats])   # (n_sess, n_feat)
    sim   = cosine_similarity(means)
    mask  = ~np.eye(sim.shape[0], dtype=bool)
    if mask.any():
        mn  = float(sim[mask].min())
        # find the pair with lowest similarity for the warning message
        idx = np.unravel_index(
            np.where(mask, sim, 1.0).argmin(), sim.shape)
        pair = (names[idx[0]] if idx[0] < len(names) else str(idx[0]),
                names[idx[1]] if idx[1] < len(names) else str(idx[1]))
        report["min_similarity"] = round(mn, 3)
        report["worst_pair"] = list(pair)
        if mn < 0.5:
            report["warnings"].append(
                f"Low inter-session feature similarity (min={mn:.3f} < 0.5) "
                f"between '{pair[0]}' and '{pair[1]}'. "
                "Verify bodypart consistency across sessions.")
            report["status"] = "warn"
    return report


def validate_umap_trustworthiness(features: np.ndarray,
                                   embedding: np.ndarray,
                                   n_neighbors: int = 10) -> dict:
    """
    Stage: UMAP embedding quality.
    Warns if trustworthiness score < 0.8 (local neighbourhood not preserved).
    Subsamples to 5000 points for speed.
    """
    report: dict = {"stage": "umap_trustworthiness", "status": "pass",
                    "warnings": [], "trustworthiness": None}
    try:
        from sklearn.manifold import trustworthiness as _tw
        n = features.shape[0]
        if n > 5000:
            idx = np.random.default_rng(0).choice(n, 5000, replace=False)
            tw  = _tw(features[idx], embedding[idx],
                      n_neighbors=n_neighbors)
        else:
            tw  = _tw(features, embedding, n_neighbors=n_neighbors)
        report["trustworthiness"] = round(float(tw), 4)
        if tw < 0.8:
            report["warnings"].append(
                f"Low UMAP trustworthiness ({tw:.3f} < 0.8). "
                "Embedding may not faithfully preserve local structure. "
                "Consider increasing umap_n_neighbors.")
            report["status"] = "warn"
    except Exception as e:
        report["warnings"].append(f"Trustworthiness computation failed: {e}")
    return report


def validate_clustering(embedding: np.ndarray,
                         labels: np.ndarray) -> dict:
    """
    Stage: HDBSCAN clustering quality.
    Warns  if mean silhouette < 0.2 (potentially unreliable clusters).
    Blocks if mean silhouette < 0.0 (clusters are worse than random).
    Subsamples to 10 000 labelled points for speed.
    """
    report: dict = {"stage": "clustering", "status": "pass",
                    "warnings": [], "silhouette_score": None,
                    "blocked": False}
    from sklearn.metrics import silhouette_score
    mask = labels >= 0
    n_cl = len(np.unique(labels[mask]))
    if mask.sum() < 2 or n_cl < 2:
        report["warnings"].append(
            "Too few labelled samples/clusters for silhouette score.")
        report["status"] = "warn"
        return report
    try:
        idx = mask.nonzero()[0]
        if len(idx) > 10_000:
            idx = np.random.default_rng(42).choice(idx, 10_000, replace=False)
        ss  = float(silhouette_score(embedding[idx], labels[idx]))
        report["silhouette_score"] = round(ss, 4)
        if ss < 0.0:
            report["warnings"].append(
                f"NEGATIVE silhouette score ({ss:.3f}): HDBSCAN clusters are "
                "INVALID (worse than random assignment). Adjust UMAP / HDBSCAN "
                "settings or improve data quality.")
            report["status"] = "block"
            report["blocked"] = True
        elif ss < 0.2:
            report["warnings"].append(
                f"Low silhouette score ({ss:.3f} < 0.2): clustering may be "
                "unreliable. Inspect the UMAP plot before proceeding.")
            report["status"] = "warn"
    except Exception as e:
        report["warnings"].append(f"Silhouette computation failed: {e}")
    return report


def validate_mlp_accuracy(cv_scores: np.ndarray) -> dict:
    """
    Stage: MLP classifier accuracy.
    Warns  if mean CV accuracy < 0.7 (classifier may be unreliable).
    Blocks if mean CV accuracy < 0.5 (at-chance performance).
    """
    report: dict = {"stage": "mlp_accuracy", "status": "pass",
                    "warnings": [], "cv_mean": None, "blocked": False}
    if cv_scores is None or len(cv_scores) == 0:
        return report
    mean_acc = float(cv_scores.mean())
    report["cv_mean"] = round(mean_acc, 4)
    if mean_acc < 0.5:
        report["warnings"].append(
            f"CV accuracy {mean_acc:.3f} < 0.5: classifier is at chance level. "
            "This run is FLAGGED — check data quality and cluster count.")
        report["status"] = "block"
        report["blocked"] = True
    elif mean_acc < 0.7:
        report["warnings"].append(
            f"CV accuracy {mean_acc:.3f} < 0.7: classifier may be unreliable. "
            "Consider collecting more data or reducing cluster count.")
        report["status"] = "warn"
    return report


def labels_to_bouts(frame_labels: np.ndarray) -> pd.DataFrame:
    """
    Convert per-frame label array to run-length encoded bout table.

    Output columns match the B-SOiD GUI format exactly:
        "B-SOiD labels", "Start time (frames)", "Run lengths"
    """
    rows = []
    n    = len(frame_labels)
    i    = 0
    while i < n:
        lbl = int(frame_labels[i])
        j   = i + 1
        while j < n and int(frame_labels[j]) == lbl:
            j += 1
        rows.append({
            "B-SOiD labels":       lbl,
            "Start time (frames)": i,
            "Run lengths":         j - i,
        })
        i = j
    return pd.DataFrame(rows)


def bouts_to_epochs(bout_df: pd.DataFrame, fps: float,
                    min_dur: float = 0.0,
                    max_dur: float = 1e9) -> pd.DataFrame:
    """
    Expand bout table to one row per epoch with timing in seconds.
    Filters by [min_dur, max_dur].
    """
    rows = []
    for _, r in bout_df.iterrows():
        sf  = int(r["Start time (frames)"])
        rl  = int(r["Run lengths"])
        dur = rl / fps
        if min_dur <= dur <= max_dur:
            rows.append(dict(
                start_frame  = sf,
                end_frame    = sf + rl - 1,
                start_sec    = sf / fps,
                end_sec      = (sf + rl - 1) / fps,
                duration_sec = dur,
                label        = int(r["B-SOiD labels"]),
            ))
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["start_frame","end_frame","start_sec",
                 "end_sec","duration_sec","label"])


def epoch_stats(epochs: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster summary statistics of epoch durations."""
    if epochs.empty:
        return pd.DataFrame()
    return (
        epochs.groupby("label")["duration_sec"]
        .agg(count="count", mean="mean", std="std",
             median="median", min="min", max="max")
        .reset_index()
    )


def compute_cluster_kinematics(all_xy: list, all_frame_labels: list,
                                all_fps: list, bodyparts: list,
                                out_path: Path) -> "pd.DataFrame":
    """Interpretable per-cluster kinematic signatures from pose.

    Clusters are otherwise named only via example clips; this gives each one a
    quantitative fingerprint (mean centroid speed, body elongation, angular
    velocity of the body axis) useful for naming and cross-study comparison.
    Aggregates per-frame descriptors by cluster id across all sessions and
    writes cluster_kinematics.csv.  Spine-dependent metrics are NaN when no
    head/tail landmarks are present.
    """
    head_idx, tail_idx = _find_spine_indices(bodyparts or [])
    agg: dict = {}
    for xy, fl, fps in zip(all_xy, all_frame_labels, all_fps):
        n = int(min(len(fl), xy.shape[0]))
        if n < 2:
            continue
        xs = xy[:n, 0::2]; ys = xy[:n, 1::2]
        cx = xs.mean(axis=1); cy = ys.mean(axis=1)
        speed = np.hypot(np.diff(cx, prepend=cx[:1]),
                         np.diff(cy, prepend=cy[:1])) * fps
        if head_idx is not None and tail_idx is not None:
            elong = np.hypot(xs[:, head_idx] - xs[:, tail_idx],
                             ys[:, head_idx] - ys[:, tail_idx])
            ax_ang = np.arctan2(ys[:, head_idx] - ys[:, tail_idx],
                                xs[:, head_idx] - xs[:, tail_idx])
            angvel = np.abs(np.diff(np.unwrap(ax_ang),
                                    prepend=ax_ang[:1])) * fps
        else:
            elong = np.full(n, np.nan); angvel = np.full(n, np.nan)
        fl2 = np.asarray(fl[:n], dtype=int)
        for cid in np.unique(fl2[fl2 >= 0]):
            m = fl2 == cid
            d = agg.setdefault(int(cid),
                               {"speed": [], "elong": [], "angvel": [], "n": 0})
            d["speed"].append(speed[m]); d["elong"].append(elong[m])
            d["angvel"].append(angvel[m]); d["n"] += int(m.sum())
    rows = []
    for cid in sorted(agg):
        d = agg[cid]
        sp = np.concatenate(d["speed"]) if d["speed"] else np.array([np.nan])
        el = np.concatenate(d["elong"]) if d["elong"] else np.array([np.nan])
        av = np.concatenate(d["angvel"]) if d["angvel"] else np.array([np.nan])
        rows.append({
            "cluster_id":                 cid,
            "n_frames":                   d["n"],
            "mean_speed_px_s":            round(float(np.nanmean(sp)), 3),
            "mean_body_elongation_px":    round(float(np.nanmean(el)), 3),
            "mean_angular_velocity_rad_s": round(float(np.nanmean(av)), 3),
        })
    df = pd.DataFrame(rows)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(out_path), index=False)
    except Exception:
        pass
    return df


#  
#  PLOTS  (theme-aware; colours set by _apply_plot_theme() at run start)
#

def _savefig(fig: plt.Figure, path: Path, dpi: int = 150):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    except Exception as e:
        print(f"Warning: Failed to save plot {path.name}: {e}")
    finally:
        plt.close(fig)


def _dark_ax(ax):
    """Style a matplotlib Axes to match the current plot theme."""
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_TICK_COL)
    for sp in ax.spines.values():
        sp.set_edgecolor(_PANEL)
    ax.xaxis.label.set_color(_TICK_COL)
    ax.yaxis.label.set_color(_TICK_COL)
    ax.title.set_color(_TEXT_COL)


def plot_umap(embedding: np.ndarray, labels: np.ndarray,
              out_path: Path, tag: str = "train"):
    if embedding.shape[1] < 2:
        return
    valid = labels >= 0
    uniq  = np.unique(labels[valid])
    n_cl  = len(uniq)
    n_dim = embedding.shape[1]
    # Per-cluster point counts so the legend exposes tiny clusters honestly.
    counts = {int(u): int((valid & (labels == u)).sum()) for u in uniq}

    # When UMAP is 3-D, a single 2-D scatter (axes 1-2) hides any structure that
    # separates clusters along axis 3 — the static PNG would misrepresent the
    # embedding.  Render all three pairwise projections (1-2, 1-3, 2-3) instead.
    # The interactive umap_3d.html still carries the full 3-D view.
    pairs = [(0, 1), (0, 2), (1, 2)] if n_dim >= 3 else [(0, 1)]

    legend_rows = int(np.ceil(n_cl / 4))
    extra_h     = max(0, (legend_rows - 5) * 0.22)
    if len(pairs) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(9, 8 + extra_h), facecolor=_BG)
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6 + extra_h), facecolor=_BG)
        axes = list(np.ravel(axes))

    for ax, (i, j) in zip(axes, pairs):
        _dark_ax(ax)
        for u in uniq:
            m = valid & (labels == u)
            ax.scatter(embedding[m, i], embedding[m, j],
                       s=2, alpha=0.5, color=_cmap(u), label=f"C{u}")
        ax.set_xlabel(f"UMAP {i + 1}"); ax.set_ylabel(f"UMAP {j + 1}")

    _proj = "3-D embedding — pairwise projections" if n_dim >= 3 else "2-D embedding"
    fig.suptitle(f"UMAP {_proj}  [{tag}]  —  {n_cl} clusters",
                 color=_TEXT_COL, fontsize=13)

    # Legend with point counts; placed on the last axes / below for many clusters.
    handles = [mpatches.Patch(color=_cmap(u), label=f"C{u} (n={counts[int(u)]})")
               for u in uniq]
    _leg_ax = axes[-1]
    if n_cl <= 20:
        _leg_ax.legend(handles=handles, fontsize=7, ncol=2,
                       facecolor=_PANEL, edgecolor=_PANEL,
                       labelcolor=_TEXT_COL, loc="upper right")
    else:
        ncol = min(8, max(4, int(np.ceil(n_cl / 5))))
        _leg_ax.legend(handles=handles, fontsize=6, ncol=ncol,
                       facecolor=_PANEL, edgecolor=_PANEL,
                       labelcolor=_TEXT_COL,
                       loc="upper center",
                       bbox_to_anchor=(0.5, -0.12),
                       borderaxespad=0)
    _savefig(fig, out_path)


def seed_sweep_stability(feats_sc_T: np.ndarray, cfg: dict, n_seeds: int,
                          log_fn=None) -> dict:
    """Re-run UMAP+HDBSCAN over n_seeds random seeds to gauge partition stability.

    Internal quality gates (silhouette, DBCV, trustworthiness) measure how tight
    each cluster is, not whether the PARTITION is reproducible.  This sweep
    answers the latter: it reports the cluster-count distribution and the
    pairwise Adjusted Rand Index (ARI) between seeds.  High mean ARI (→1) means
    the clustering is stable; low ARI means cluster identities depend on the
    seed and should be treated with caution.

    Returns {seeds, counts, ari, mean_ari}.  Empty dict if n_seeds < 2 or the
    required libraries are missing.
    """
    if n_seeds is None or int(n_seeds) < 2:
        return {}
    try:
        from sklearn.metrics import adjusted_rand_score
    except Exception:
        return {}
    base_seed = int(cfg.get("umap_random_state", 42))
    seeds = [base_seed + i for i in range(int(n_seeds))]
    all_labels, counts = [], []
    for s in seeds:
        try:
            c2 = dict(cfg); c2["umap_random_state"] = s
            _, emb = run_umap(feats_sc_T, c2)
            _, lbls, _, _ = run_hdbscan(emb, c2, n_total=feats_sc_T.shape[0])
            all_labels.append(np.asarray(lbls))
            counts.append(len(set(int(x) for x in lbls if x >= 0)))
            if log_fn:
                log_fn(f"  [seed-sweep] seed {s}: {counts[-1]} clusters")
        except Exception:
            if log_fn:
                log_fn(f"  [seed-sweep] seed {s} failed; skipped")
    m = len(all_labels)
    if m < 2:
        return {}
    ari = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            try:
                a = float(adjusted_rand_score(all_labels[i], all_labels[j]))
            except Exception:
                a = np.nan
            ari[i, j] = ari[j, i] = a
    triu = ari[np.triu_indices(m, 1)]
    return dict(seeds=seeds, counts=counts, ari=ari,
                mean_ari=float(np.nanmean(triu)) if triu.size else 1.0)


def plot_cluster_stability(sweep: dict, out_path: Path):
    """Cluster-count distribution + pairwise ARI heatmap from seed_sweep_stability."""
    if not sweep or "ari" not in sweep:
        return
    counts = sweep.get("counts", [])
    ari    = np.asarray(sweep["ari"], dtype=float)
    seeds  = sweep.get("seeds", list(range(len(counts))))
    mean_ari = sweep.get("mean_ari", float("nan"))

    fig, (axc, axa) = plt.subplots(1, 2, figsize=(13, 5), facecolor=_BG)
    _dark_ax(axc); _dark_ax(axa)

    # Left: cluster-count distribution across seeds.
    if counts:
        _vals, _cnts = np.unique(counts, return_counts=True)
        axc.bar([str(v) for v in _vals], _cnts, color="#4E79A7", alpha=0.85)
    axc.set_xlabel("Cluster count"); axc.set_ylabel("Seeds")
    axc.set_title(f"Cluster-count stability across {len(seeds)} seeds\n"
                  f"(range {min(counts) if counts else 0}–{max(counts) if counts else 0})",
                  color=_TEXT_COL, fontsize=10)

    # Right: pairwise ARI heatmap.
    im = axa.imshow(ari, cmap=plt.cm.viridis, vmin=0.0, vmax=1.0, aspect="auto")
    cb = plt.colorbar(im, ax=axa, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=_TICK_COL)
    cb.set_label("Adjusted Rand Index", color=_TICK_COL)
    axa.set_xticks(range(len(seeds)))
    axa.set_xticklabels([str(s) for s in seeds], rotation=45, ha="right",
                        color=_TICK_COL, fontsize=7)
    axa.set_yticks(range(len(seeds)))
    axa.set_yticklabels([str(s) for s in seeds], color=_TICK_COL, fontsize=7)
    axa.set_title(f"Pairwise partition agreement (ARI)\n"
                  f"mean ARI = {mean_ari:.3f}  "
                  f"({'stable' if mean_ari >= 0.7 else 'unstable — interpret with caution'})",
                  color=_TEXT_COL, fontsize=10)
    plt.tight_layout()
    _savefig(fig, out_path)


def _tmat_from_labels(all_frame_labels: list):
    """
    Build a row-stochastic transition matrix from frame-label sequences.
    Returns (tmat, cluster_ids) — both None if no valid transitions exist.
    """
    from collections import Counter
    counts: Counter = Counter()
    all_ids: set = set()
    for fl in all_frame_labels:
        arr = np.asarray(fl, dtype=int)
        for a, b in zip(arr[:-1], arr[1:]):
            if int(a) >= 0 and int(b) >= 0 and a != b:
                counts[(int(a), int(b))] += 1
                all_ids.update([int(a), int(b)])
    if not all_ids:
        return None, None
    ids = sorted(all_ids)
    n   = len(ids)
    idx = {l: i for i, l in enumerate(ids)}
    T   = np.zeros((n, n), dtype=float)
    for (a, b), cnt in counts.items():
        T[idx[a], idx[b]] = cnt
    rs = T.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    T /= rs
    return T, ids


def plot_umap_3d_transitions(
    embedding: np.ndarray,
    labels: np.ndarray,
    tmat=None,
    cluster_ids=None,
    out_path=None,
    max_edges: int = 10,
    min_prob: float = 0.05,
    tag: str = "",
) -> None:
    """
    Render a 3D UMAP scatter with directional transition arrows.

    Only transitions strictly above chance (1 / (n_clusters − 1)) are drawn;
    from those, the top max_edges by probability are displayed.
    Outputs a static 3-viewpoint PNG (matplotlib) and an interactive HTML
    (plotly, if installed).  Both files share the stem of *out_path*.

    Parameters
    ----------
    embedding    : (n_samples, >=3) UMAP embedding
    labels       : (n_samples,) cluster labels  (-1 = noise)
    tmat         : (n_clusters, n_clusters) row-stochastic transition matrix
    cluster_ids  : list of cluster IDs matching tmat rows/cols
    out_path     : destination for the .html file; .png written alongside
    max_edges    : top-N above-chance transitions to draw (0 = no cap)
    min_prob     : minimum transition probability threshold
    tag          : title tag string
    """
    if embedding.shape[1] < 3:
        return
    valid = labels >= 0
    uniq  = sorted(set(labels[valid]))
    if not uniq:
        return

    e3 = embedding[:, :3]
    centroids = {u: e3[valid & (labels == u)].mean(axis=0) for u in uniq}

    # ── Global-ranked edge selection ──────────────────────────────────────────
    # Transitions at or below chance (1 / (n_clusters − 1)) carry no information
    # above a uniform random walk and are suppressed regardless of min_prob.
    chance_floor = 1.0 / max(1, len(uniq) - 1)
    effective_min = max(min_prob, chance_floor)

    edges = []   # (src, tgt, prob)
    if tmat is not None and cluster_ids is not None:
        idx_map   = {c: i for i, c in enumerate(cluster_ids)}
        all_cands = []
        for src in uniq:
            if src not in idx_map:
                continue
            si = idx_map[src]
            for tgt in uniq:
                if tgt == src or tgt not in idx_map:
                    continue
                prob = float(tmat[si, idx_map[tgt]])
                if prob > effective_min:
                    all_cands.append((prob, src, tgt))
        all_cands.sort(reverse=True)
        _cap = max_edges if max_edges > 0 else len(all_cands)
        edges = [(src, tgt, prob) for prob, src, tgt in all_cands[:_cap]]

    max_prob  = max((p for _, _, p in edges), default=1.0)
    # Normalise thickness/opacity over the above-chance range so the weakest
    # shown edge (just above chance_floor) maps to 0 and the strongest to 1.
    prob_range = max(max_prob - effective_min, 1e-9)
    n_labelled = min(5, len(edges))   # prob labels only on top-5 arrows

    title = (f"3D UMAP — {len(uniq)} clusters"
             + (f"  [{tag}]" if tag else "")
             + (f"  ·  top {len(edges)} transitions" if edges else ""))

    # Embedding extent drives proportional sizing
    extent = float(np.max(e3.max(axis=0) - e3.min(axis=0))) if len(uniq) > 1 else 1.0

    # ── Static PNG — matplotlib Axes3D, 3 fixed viewpoints ───────────────────
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import matplotlib.patches as _mpatches

        views = [(20, 45), (20, 200), (60, 100)]
        fig_s = plt.figure(figsize=(18, 7), facecolor=_BG)
        for vi, (elev, azim) in enumerate(views):
            ax = fig_s.add_subplot(1, 3, vi + 1, projection="3d")
            ax.set_facecolor(_PANEL)
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor("#333344")

            # Point cloud (sub-sampled for large datasets) — larger & more opaque
            for u in uniq:
                m   = valid & (labels == u)
                pts = e3[m]
                step = max(1, len(pts) // 4000)
                ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                           s=4, alpha=0.45, color=_cmap(u), depthshade=False)

            # Centroid markers — prominent, outlined for contrast
            for u in uniq:
                c = centroids[u]
                ax.scatter(*c, s=60, color=_cmap(u),
                           edgecolors=_TEXT_COL, linewidths=0.8,
                           zorder=5, depthshade=False)
                ax.text(c[0], c[1], c[2], f"  C{u}", fontsize=6,
                        color=_TEXT_COL, fontweight="bold", zorder=7)

            # Transition arrows: thick plot() shaft + quiver arrowhead
            for ei, (src, tgt, prob) in enumerate(edges):
                s_c  = centroids[src]
                t_c  = centroids[tgt]
                d    = t_c - s_c
                # Relative weight over the above-chance range: 0 = just above
                # chance, 1 = strongest transition.  Anchoring at effective_min
                # (not 0) ensures full thickness contrast among shown arrows.
                rel  = (prob - effective_min) / prob_range
                lw   = float(0.8 + rel * 2.2)        # 0.8–3.0 pt (was 1.5–9.0)
                alph = float(0.30 + rel * 0.35)       # 0.30–0.65 (was 0.45–0.95)
                col  = _cmap(src)

                # Halo: draw slightly thicker contrasting line behind for visibility
                ax.plot([s_c[0], t_c[0]], [s_c[1], t_c[1]], [s_c[2], t_c[2]],
                        color=_TEXT_COL, lw=lw + 0.8, alpha=alph * 0.30,
                        zorder=3, solid_capstyle="round")
                # Coloured shaft (80% of length, arrowhead fills the rest)
                mid = s_c + 0.80 * d
                ax.plot([s_c[0], mid[0]], [s_c[1], mid[1]], [s_c[2], mid[2]],
                        color=col, lw=lw, alpha=alph,
                        zorder=4, solid_capstyle="round")
                # Quiver arrowhead on the final 20%
                dx, dy, dz = 0.20 * d
                ax.quiver(mid[0], mid[1], mid[2], dx, dy, dz,
                          arrow_length_ratio=0.60,
                          color=col, alpha=alph, linewidth=lw * 0.5,
                          normalize=False, zorder=5)
                # Probability label on top-N arrows (first view only)
                if vi == 0 and ei < n_labelled:
                    lp = s_c + 0.50 * d
                    ax.text(lp[0], lp[1], lp[2], f"{prob:.2f}",
                            fontsize=5.5, color=_TEXT_COL, fontweight="bold",
                            ha="center", va="center", zorder=8)

            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("UMAP 1", fontsize=7, color=_TICK_COL, labelpad=1)
            ax.set_ylabel("UMAP 2", fontsize=7, color=_TICK_COL, labelpad=1)
            ax.set_zlabel("UMAP 3", fontsize=7, color=_TICK_COL, labelpad=1)
            ax.tick_params(colors=_TICK_COL, labelsize=5, pad=1)
            view_labels = ["Front-Left", "Back-Right", "Top-Down"]
            ax.set_title(view_labels[vi], color=_TICK_COL, fontsize=8, pad=4)

        # Transition-strength legend (line thickness key)
        if edges:
            leg_handles = [
                _mpatches.FancyArrow(0, 0, 1, 0, width=0.3,
                                     color=_cmap(edges[0][0]),
                                     label=f"strongest ({edges[0][2]:.2f})"),
                _mpatches.FancyArrow(0, 0, 1, 0, width=0.15,
                                     color=_cmap(edges[-1][0]),
                                     label=f"weakest shown ({edges[-1][2]:.2f})"),
            ]
            fig_s.legend(handles=leg_handles, loc="lower center",
                         ncol=2, fontsize=7, facecolor=_PANEL,
                         labelcolor=_TEXT_COL, edgecolor="#333355",
                         framealpha=0.8)

        fig_s.suptitle(title, color=_TEXT_COL, fontsize=12, y=1.01)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        if out_path is not None:
            _savefig(fig_s, Path(out_path).with_suffix(".png"))
        plt.close(fig_s)
    except Exception:
        pass

    # ── Interactive HTML — plotly ─────────────────────────────────────────────
    if out_path is None:
        return
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    fig_p = go.Figure()

    # Translucent point cloud — one trace per cluster for legend
    for u in uniq:
        m   = valid & (labels == u)
        pts = e3[m]
        step = max(1, len(pts) // 5000)
        fig_p.add_trace(go.Scatter3d(
            x=pts[::step, 0], y=pts[::step, 1], z=pts[::step, 2],
            mode="markers",
            marker=dict(size=3, color=_cmap(u), opacity=0.45),
            name=f"C{u}",
            legendgroup="clusters",
            legendgrouptitle=dict(text="Clusters") if u == uniq[0] else {},
            showlegend=True,
            hovertemplate=f"C{u}<extra></extra>",
        ))

    # Centroid nodes with labels — larger, always-on-top
    cxyz = np.array([centroids[u] for u in uniq])
    fig_p.add_trace(go.Scatter3d(
        x=cxyz[:, 0], y=cxyz[:, 1], z=cxyz[:, 2],
        mode="markers+text",
        marker=dict(size=9, color=[_cmap(u) for u in uniq],
                    line=dict(color=_TEXT_COL, width=2), opacity=1.0),
        text=[f"C{u}" for u in uniq],
        textposition="top center",
        textfont=dict(size=11, color=_TEXT_COL, family="Arial Black"),
        name="Centroids",
        legendgroup="centroids",
        legendgrouptitle=dict(text="Centroids"),
        showlegend=True,
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))

    # Transition arrows: thick shaft + cone arrowhead, width ∝ probability
    cone_size = extent * 0.06
    for ei, (src, tgt, prob) in enumerate(edges):
        s_c   = centroids[src]
        t_c   = centroids[tgt]
        d     = t_c - s_c
        norm  = float(np.linalg.norm(d))
        if norm < 1e-8:
            continue
        d_hat = d / norm
        rel   = (prob - effective_min) / prob_range
        # Width spans 2–8 px; opacity spans 0.35–0.70
        lw_px = max(2, int(2 + rel * 6))
        alph  = float(0.35 + rel * 0.35)
        color = _cmap(src)
        label = f"C{src}→C{tgt}: {prob:.3f}"

        # Shaft stops 18% short so the cone is fully visible
        shaft_end = s_c + 0.82 * d
        fig_p.add_trace(go.Scatter3d(
            x=[s_c[0], shaft_end[0], None],
            y=[s_c[1], shaft_end[1], None],
            z=[s_c[2], shaft_end[2], None],
            mode="lines",
            line=dict(color=color, width=lw_px),
            opacity=alph,
            showlegend=(ei == 0),
            legendgroup="transitions",
            legendgrouptitle=dict(text="Transitions") if ei == 0 else {},
            name="Transitions" if ei == 0 else "",
            hovertemplate=f"{label}<extra></extra>",
        ))
        # Cone arrowhead — size ∝ above-chance relative weight
        fig_p.add_trace(go.Cone(
            x=[t_c[0]], y=[t_c[1]], z=[t_c[2]],
            u=[d_hat[0]], v=[d_hat[1]], w=[d_hat[2]],
            sizemode="absolute",
            sizeref=cone_size * (0.4 + rel * 1.2),
            anchor="tip",
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=alph,
            hovertemplate=f"{label}<extra></extra>",
            showlegend=False,
        ))
        # Floating probability label near arrow midpoint
        mid = s_c + 0.50 * d
        fig_p.add_trace(go.Scatter3d(
            x=[mid[0]], y=[mid[1]], z=[mid[2]],
            mode="text",
            text=[f"{prob:.2f}"],
            textfont=dict(size=10, color=_TEXT_COL, family="Arial Black"),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig_p.update_layout(
        title=dict(text=title, font=dict(color=_TEXT_COL, size=14)),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        scene=dict(
            xaxis=dict(title="UMAP 1", color=_TICK_COL,
                       backgroundcolor=_BG, gridcolor="#2a2a44",
                       showbackground=True),
            yaxis=dict(title="UMAP 2", color=_TICK_COL,
                       backgroundcolor=_BG, gridcolor="#2a2a44",
                       showbackground=True),
            zaxis=dict(title="UMAP 3", color=_TICK_COL,
                       backgroundcolor=_BG, gridcolor="#2a2a44",
                       showbackground=True),
            bgcolor=_BG,
        ),
        legend=dict(font=dict(color=_TEXT_COL, size=9),
                    bgcolor=_PANEL, bordercolor="#333355",
                    groupclick="toggleitem"),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    html_path = Path(out_path).with_suffix(".html")
    try:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig_p.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    except Exception:
        pass


def plot_confusion(mlp_model, feats_sc: np.ndarray,
                   labels: np.ndarray, out_path: Path):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics         import ConfusionMatrixDisplay, confusion_matrix
    mask = labels >= 0
    X, y = feats_sc[:, mask].T, labels[mask]
    if len(np.unique(y)) < 2:
        return
    _, Xt, _, yt = train_test_split(X, y, test_size=0.2, random_state=42,
                                    stratify=y)
    yp  = mlp_model.predict(Xt)
    cm  = confusion_matrix(yt, yp, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8), facecolor=_BG)
    ax.set_facecolor(_PANEL)
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion matrix (normalised, 20 % hold-out)",
                 color=_TEXT_COL, fontsize=11)
    ax.tick_params(colors=_TICK_COL)
    ax.xaxis.label.set_color(_TICK_COL)
    ax.yaxis.label.set_color(_TICK_COL)
    _savefig(fig, out_path)


def plot_ethogram(frame_labels: np.ndarray, fps: float,
                  out_path: Path, tag: str, cluster_names: dict = None):
    """Per-session behavioural raster.

    cluster_names : optional {cluster_id: "behaviour name"} (e.g. from the Video
        Explorer annotation) used for the y-axis labels.  Falls back to C<id>
        when a name is missing, so passing it is always safe.
    """
    uniq = np.unique(frame_labels)
    t    = np.arange(len(frame_labels)) / fps

    def _ylabel(l):
        if cluster_names and int(l) in cluster_names and cluster_names[int(l)]:
            return f"C{l} {cluster_names[int(l)]}"
        return f"C{l}"

    fig, ax = plt.subplots(
        figsize=(14, max(3, len(uniq) * 0.55)), facecolor=_BG)
    _dark_ax(ax)
    for idx_u, lbl in enumerate(uniq):
        sel = np.where(frame_labels == lbl)[0]
        ax.scatter(t[sel], np.full(len(sel), idx_u),
                   c=_cmap(lbl), s=8, marker="|", linewidths=3.5)
    ax.set_yticks(range(len(uniq)))
    ax.set_yticklabels([_ylabel(l) for l in uniq], color=_TEXT_COL, fontsize=8)
    ax.set_xlabel("Time (s)", color=_TICK_COL)
    ax.set_title(f"Ethogram  –  {tag}", color=_TEXT_COL, fontsize=11)
    _savefig(fig, out_path)


def plot_cluster_durations(epochs: pd.DataFrame,
                            out_path: Path, tag: str):
    if epochs.empty:
        return
    uniq  = sorted(epochs.label.unique())
    ncols = min(4, len(uniq))
    nrows = int(np.ceil(len(uniq) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3),
                             facecolor=_BG, squeeze=False)
    for ax in axes.flat:
        _dark_ax(ax)
    for idx_g, grp in enumerate(uniq):
        r, c = divmod(idx_g, ncols)
        ax   = axes[r][c]
        durs = epochs.loc[epochs.label == grp, "duration_sec"]
        bins = min(20, max(5, len(durs) // 2))
        ax.hist(durs, bins=bins, color=_cmap(grp),
                edgecolor=_BG, alpha=0.85)
        ax.set_title(f"C{grp}  (n={len(durs)})", fontsize=9)
        ax.set_xlabel("Duration (s)", fontsize=8)
    for idx_g in range(len(uniq), nrows * ncols):
        r, c = divmod(idx_g, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle(f"Epoch duration distributions  –  {tag}",
                 color=_TEXT_COL, fontsize=11)
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_cluster_stats(epochs: pd.DataFrame, out_path: Path):
    if epochs.empty:
        return
    stats = epoch_stats(epochs)
    if stats.empty:
        return
    grps   = stats["label"].tolist()
    colors = [_cmap(g) for g in grps]
    xpos   = np.arange(len(grps))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor=_BG)
    for ax in axes:
        _dark_ax(ax)
    panels = [
        (axes[0], stats["mean"],               "Mean bout (s)"),
        (axes[1], stats["count"],              "Frequency (# bouts)"),
        (axes[2], stats["mean"] * stats["count"], "Total duration (s)"),
    ]
    for ax, vals, ylabel in panels:
        ax.bar(xpos, vals, color=colors, edgecolor="none", alpha=0.9)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"C{g}" for g in grps],
                           rotation=60, color="#aaaacc", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel)
    fig.suptitle("Cluster Statistics", color=_TEXT_COL, fontsize=12)
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_feature_quality(feats_list: list, names: list, out_path: Path):
    means = [float(np.mean(np.abs(f))) for f in feats_list]
    fig, ax = plt.subplots(
        figsize=(max(6, len(names) * 0.9 + 2), 4), facecolor=_BG)
    _dark_ax(ax)
    ax.bar(range(len(means)), means,
           color=[_cmap(i) for i in range(len(means))],
           edgecolor="none", alpha=0.9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right",
                       color="#aaaacc", fontsize=8)
    ax.set_ylabel("Mean |feature|")
    ax.set_title("Feature magnitude per session (data quality)")
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_likelihood_qc(dlc_paths: list, out_path: Path):
    """Per-bodypart likelihood violin plot across all sessions (quality diagnostic)."""
    from collections import defaultdict
    bp_ll: dict = defaultdict(list)
    for fp in dlc_paths:
        try:
            ext = Path(str(fp)).suffix.lower()
            if ext in (".h5", ".hdf5"):
                df = pd.read_hdf(str(fp))
            else:
                with open(fp, encoding="utf-8", errors="replace") as fh:
                    head = [fh.readline() for _ in range(5)]
                n_lv = max(sum(1 for l in head[:4]
                               if l.strip() and not l.strip()[0].isdigit()), 2)
                df = pd.read_csv(fp, header=list(range(n_lv)), index_col=0)
            df = _normalise_dlc_df(df)
            for bp in df.columns.get_level_values("bodyparts").unique():
                sub = df.xs(bp, level="bodyparts", axis=1)
                if sub.columns.nlevels > 1:
                    sub.columns = sub.columns.get_level_values(-1)
                ll = pd.to_numeric(
                    sub.get("likelihood", pd.Series(float("nan"), index=df.index)),
                    errors="coerce").dropna()
                bp_ll[bp].extend(ll.tolist())
        except Exception:
            pass
    if not bp_ll:
        return
    bps  = sorted(bp_ll.keys())
    data = [bp_ll[bp] for bp in bps]
    fig, ax = plt.subplots(figsize=(max(8, len(bps) * 0.9 + 2), 5), facecolor=_BG)
    _dark_ax(ax)
    parts = ax.violinplot(data, positions=range(len(bps)), showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(_cmap(i))
        pc.set_alpha(0.65)
    parts["cmedians"].set_color("#ffd60a")
    ax.axhline(0.3, color="#ff9800", linestyle="--", linewidth=1.2,
               label="0.3 likelihood threshold")
    ax.set_xticks(range(len(bps)))
    ax.set_xticklabels(bps, rotation=45, ha="right", color="#aaaacc", fontsize=8)
    ax.set_ylabel("Likelihood")
    ax.set_ylim(0, 1.05)
    ax.set_title("Bodypart detection likelihood  (quality diagnostic)")
    ax.legend(fontsize=8, facecolor=_PANEL, labelcolor=_TEXT_COL, loc="lower right")
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_transition_matrix(all_frame_labels: list, out_path: Path):
    """Behavioral state transition probability matrix  P(next | current).
    Diagonal (self-transitions) and off-diagonal entries ≤ chance are shown
    as background.  The colormap is anchored at chance_floor so the weakest
    visible arrow maps to the lightest colour and the strongest to the darkest.
    """
    from collections import Counter
    counts: Counter = Counter()
    all_labels: set = set()
    for fl in all_frame_labels:
        for a, b in zip(fl[:-1], fl[1:]):
            a, b = int(a), int(b)
            counts[(a, b)] += 1
            all_labels.update((a, b))
    if not all_labels:
        return
    labs = sorted(all_labels)
    n    = len(labs)
    idx  = {l: i for i, l in enumerate(labs)}
    T    = np.zeros((n, n), dtype=float)
    for (a, b), cnt in counts.items():
        T[idx[a], idx[b]] = cnt
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T /= row_sums
    chance_floor = 1.0 / max(1, n - 1)
    # NaN-mask self-transitions and below-chance entries so set_bad() shows
    # them as the panel background, leaving the colormap scale to span only
    # the above-chance range [chance_floor … max_observed_probability].
    display = T.copy().astype(float)
    np.fill_diagonal(display, np.nan)
    display[display <= chance_floor] = np.nan
    vmax_val = float(np.nanmax(display)) if not np.all(np.isnan(display)) else 1.0
    cmap_t = plt.cm.YlOrRd.copy()
    cmap_t.set_bad(color=_PANEL)
    sz = max(6, n * 0.55 + 2)
    fig, ax = plt.subplots(figsize=(sz, sz), facecolor=_BG)
    _dark_ax(ax)
    im = ax.imshow(display, cmap=cmap_t, aspect="auto",
                   vmin=chance_floor, vmax=vmax_val)
    cb = plt.colorbar(im, ax=ax)
    cb.ax.tick_params(colors="#aaaacc")
    cb.set_label("Transition probability  (above-chance range)", color="#aaaacc")
    tick_lbls = [f"C{l}" for l in labs]
    ax.set_xticks(range(n)); ax.set_xticklabels(tick_lbls, rotation=45, ha="right",
                                                  color="#aaaacc", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(tick_lbls, color="#aaaacc", fontsize=8)
    ax.set_xlabel("Next cluster"); ax.set_ylabel("Current cluster")
    ax.set_title(f"Behavioral state transition probabilities  P(next | current)\n"
                 f"(above-chance only, p > {chance_floor:.3f}  |  "
                 f"colourmap: {chance_floor:.3f} → {vmax_val:.3f})")
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_validation_summary(validation: dict, out_path: Path):
    """
    Combined validation dashboard — one panel per stage.
    Shows a pass/warn/block status badge and the key metric for each gate.
    """
    STAGES  = ["dlc_quality", "feature_consistency",
                "umap_trustworthiness", "clustering", "mlp_accuracy"]
    LABELS  = ["DLC quality", "Feature consistency",
                "UMAP trust.", "Clustering", "MLP accuracy"]
    SC_MAP  = {"pass": "#4caf50", "warn": "#ffd60a", "block": "#f44336"}

    present = [(s, l) for s, l in zip(STAGES, LABELS) if s in validation]
    if not present:
        return
    n   = len(present)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.5 + 1, 4.5),
                             facecolor=_BG, squeeze=False)
    for ax in axes.flat:
        _dark_ax(ax)

    for col, (stage, label) in enumerate(present):
        ax  = axes[0][col]
        rep = validation[stage]
        sc  = SC_MAP.get(rep.get("status", "pass"), "#aaaacc")

        if stage == "dlc_quality":
            # Bar: number of bad bodyparts per session
            sess   = rep.get("sessions", {})
            n_bad  = [len(v.get("bad_bodyparts", [])) for v in sess.values()]
            if n_bad:
                colors = [SC_MAP["block"] if b > 0 else SC_MAP["pass"]
                          for b in n_bad]
                ax.bar(range(len(n_bad)), n_bad, color=colors, edgecolor="none")
                ax.set_xticks([])
                ax.set_ylabel("# bad bodyparts", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="#aaaacc")

        elif stage == "feature_consistency":
            mn = rep.get("min_similarity")
            if mn is not None:
                ax.barh(["min sim"], [mn], color=sc)
                ax.axvline(0.5, color="#ffd60a", linestyle="--", linewidth=1.2,
                           label="thresh 0.5")
                ax.set_xlim(0, 1.05)
                ax.text(mn + 0.02, 0, f"{mn:.3f}", va="center",
                        color="#eaeaea", fontsize=8)

        elif stage == "umap_trustworthiness":
            tw = rep.get("trustworthiness")
            if tw is not None:
                ax.barh(["trust."], [tw], color=sc)
                ax.axvline(0.8, color="#ffd60a", linestyle="--", linewidth=1.2,
                           label="thresh 0.8")
                ax.set_xlim(0, 1.05)
                ax.text(tw + 0.01, 0, f"{tw:.3f}", va="center",
                        color="#eaeaea", fontsize=8)

        elif stage == "clustering":
            ss = rep.get("silhouette_score")
            if ss is not None:
                val = max(ss, 0)
                ax.barh(["silhouette"], [val], color=sc)
                ax.axvline(0.2, color="#ffd60a", linestyle="--", linewidth=1.2,
                           label="thresh 0.2")
                ax.set_xlim(-0.1 if ss < 0 else 0, 1.05)
                ax.text(val + 0.01, 0, f"{ss:.3f}", va="center",
                        color="#eaeaea", fontsize=8)

        elif stage == "mlp_accuracy":
            cv = rep.get("cv_mean")
            if cv is not None:
                ax.barh(["CV acc"], [cv], color=sc)
                ax.axvline(0.7, color="#ffd60a", linestyle="--", linewidth=1.2,
                           label="thresh 0.7")
                ax.set_xlim(0, 1.05)
                ax.text(cv + 0.01, 0, f"{cv:.3f}", va="center",
                        color="#eaeaea", fontsize=8)

        ax.set_title(label, color=_TEXT_COL, fontsize=9)
        ax.text(0.98, 0.97, rep.get("status", "?").upper(),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=sc, fontweight="bold")

    fig.suptitle("CUBE Validation Dashboard", color=_TEXT_COL, fontsize=12)
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_cv_scores(cv_scores: np.ndarray, out_path: Path):
    """MLP cross-validation accuracy — per-fold bars + summary box."""
    k = len(cv_scores)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor=_BG)
    for ax in (ax1, ax2):
        _dark_ax(ax)
    colors = [_cmap(i) for i in range(k)]
    ax1.bar(range(k), cv_scores, color=colors, edgecolor="none", alpha=0.9)
    ax1.axhline(cv_scores.mean(), color="#ffd60a", linestyle="--",
                linewidth=1.5, label=f"Mean = {cv_scores.mean():.3f}")
    ax1.set_xticks(range(k))
    ax1.set_xticklabels([f"Fold {i+1}" for i in range(k)], color="#aaaacc")
    ax1.set_ylabel("Accuracy"); ax1.set_ylim(0, 1.05)
    ax1.set_title("MLP CV accuracy per fold")
    ax1.legend(fontsize=8, facecolor=_PANEL, labelcolor=_TEXT_COL)
    ax2.boxplot(cv_scores, patch_artist=True,
                boxprops=dict(facecolor=_cmap(0), alpha=0.7),
                medianprops=dict(color="#ffd60a", linewidth=2),
                whiskerprops=dict(color="#aaaacc"),
                capprops=dict(color="#aaaacc"),
                flierprops=dict(color="#aaaacc", marker="o"))
    ax2.set_ylabel("Accuracy"); ax2.set_ylim(0, 1.05)
    ax2.set_title(
        f"CV summary  (mean={cv_scores.mean():.3f} ± {cv_scores.std():.3f})")
    plt.tight_layout()
    _savefig(fig, out_path)


#
#  DLC H5 POST-PROCESSING FILTER
#

def filter_dlc_h5(h5_path: Path, filter_types: list, log_fn=print,
                  out_path: "Path | None" = None,
                  fps: float = 30.0) -> "Path | None":
    """
    Apply filter pipeline to a DLC H5 pose file; save as <stem>_filtered.h5.
    Supports: median, gaussian, butterworth, savgol, kalman.
    Sequential filters are applied in list order.
    Pass out_path to override the default output location.
    Returns the filtered path, or None on failure.
    """
    if h5_path.stem.endswith("_filtered") and out_path is None:
        return h5_path
    out_path = Path(out_path) if out_path is not None else h5_path.with_name(h5_path.stem + "_filtered.h5")
    try:
        df     = pd.read_hdf(str(h5_path))
        df     = _normalise_dlc_df(df)
        result = df.copy()
        scorer = df.columns.get_level_values("scorer").unique()[0]
        for bp in df.columns.get_level_values("bodyparts").unique():
            for coord in ("x", "y"):
                try:
                    col    = (scorer, bp, coord)
                    series = result[col].astype(float).copy()
                    if "median" in filter_types:
                        from scipy.ndimage import median_filter as _mf
                        series = pd.Series(
                            _mf(series.values, size=7, mode="reflect"),
                            index=series.index)
                    if "gaussian" in filter_types:
                        from scipy.ndimage import gaussian_filter1d
                        series = pd.Series(
                            gaussian_filter1d(series.values, sigma=3.0),
                            index=series.index)
                    if "butterworth" in filter_types:
                        from scipy.signal import butter, filtfilt
                        cutoff = 5.0   # Hz — matches base script
                        nyq    = max(fps, 1.0) / 2.0
                        b, a   = butter(4, cutoff / nyq, btype="low")
                        series = pd.Series(
                            filtfilt(b, a, series.values),
                            index=series.index)
                    if "savgol" in filter_types:
                        from scipy.signal import savgol_filter
                        wl = min(15, len(series) - 1)
                        if wl % 2 == 0:
                            wl -= 1
                        if wl >= 3:
                            series = pd.Series(
                                savgol_filter(series.values.astype(float), wl, 3),
                                index=series.index)
                    if "kalman" in filter_types:
                        import numpy as _np
                        c = series.values.astype(float)
                        n = len(c)
                        x_k, p_k = c[0], 1.0
                        Q, R = 0.01, 1.0
                        fwd = _np.zeros(n); cov_fwd = _np.zeros(n)
                        for _i in range(n):
                            p_k += Q
                            K    = p_k / (p_k + R)
                            x_k  = x_k + K * (c[_i] - x_k)
                            p_k *= (1 - K)
                            fwd[_i] = x_k; cov_fwd[_i] = p_k
                        out_arr = fwd.copy()
                        for _i in range(n - 2, -1, -1):
                            G = cov_fwd[_i] / (cov_fwd[_i] + Q)
                            out_arr[_i] = out_arr[_i] + G * (out_arr[_i + 1] - fwd[_i])
                        series = pd.Series(out_arr, index=series.index)
                    result[col] = series.values
                except Exception:
                    pass
        result.to_hdf(str(out_path), key="df_with_missing", mode="w", format="fixed")
        log_fn(f"  Filtered H5: {out_path.name}")
        return out_path
    except Exception as e:
        log_fn(f"  [WARN] filter_dlc_h5 failed for {h5_path.name}: {e}")
        return None


#
#  VIDEO CREATION
#

def _open_writer(path: Path, fps: float, w: int, h: int):
    """Try several codecs in sequence; return (writer, actual_path)."""
    try:
        import cv2
    except ImportError:
        return None, path
    path.parent.mkdir(parents=True, exist_ok=True)
    for fourcc_str, ext in [("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi")]:
        p  = path.with_suffix(ext)
        vw = cv2.VideoWriter(str(p),
                             cv2.VideoWriter_fourcc(*fourcc_str),
                             fps, (w, h))
        if vw.isOpened():
            return vw, p
        vw.release()
    return None, path


def create_example_clips(video_path, epochs: pd.DataFrame,
                          out_dir: Path, source_fps: float,
                          output_fps: int = 15,
                          max_clips: int = 3,
                          animal_id: str = "",
                          max_clip_dur_sec: float = 8.0,
                          max_total_clips: int = 120,
                          clips_per_cluster: "dict | None" = None,
                          max_per_call: "int | None" = None):
    """Write up to max_clips short example videos per cluster.

    Files are saved to out_dir/example_clips/cluster_NN/cluster_NN_<animal>_example_MM.mp4
    so the video explorer can auto-load them by cluster.

    clips_per_cluster is an optional shared dict {cluster_label: n_written_so_far}
    that is updated in-place after each call.  When provided, a cluster is skipped
    as soon as it already has max_clips entries, so successive calls across multiple
    animals naturally produce a cross-animal mix with a per-cluster ceiling.
    Pass the same dict to every call in a session to enable this behaviour;
    omit it (or pass None) for single-video / standalone use.

    max_clip_dur_sec caps how long each exported clip can be, preventing runaway
    writes when epochs are very long.  max_total_clips is a hard per-call safety
    ceiling (regardless of clips_per_cluster) to prevent GDI exhaustion when
    there are very many clusters.

    Reads frames sequentially within each clip (one seek per clip start) to
    avoid the keyframe-decode penalty that random per-frame seeks incur on
    compressed video.  Clips are processed in start-frame order so the cap
    position moves forward through the file; small inter-clip gaps are bridged
    by reading-and-discarding rather than seeking.
    """
    try:
        import cv2
    except ImportError:
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    try:
        total          = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w              = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h              = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Write every source frame encoded at output_fps — this produces slow-motion
        # playback so short behaviours remain watchable.  Do NOT subsample here:
        # subsampling would preserve real-time speed while discarding frames, making
        # clips indistinguishable from the original and defeating the purpose.
        max_out_frames = max(1, int(max_clip_dur_sec * output_fps))
        max_src_frames = max_out_frames   # one source frame per output frame
        # For gaps narrower than this many source frames, read-and-discard is
        # cheaper than a keyframe seek (tune: ~2 s at source fps).
        skip_threshold = int(source_fps * 2)
        clips_root     = out_dir / "example_clips"
        animal_part    = f"_{animal_id}" if animal_id else ""

        # ── Build the full clip list before touching any video frames ─────────
        # pending: (sf, ef, grp, col_bgr, label_text, out_p)
        pending: list = []
        for grp in sorted(epochs.label.unique()):
            if len(pending) >= max_total_clips:
                break
            # How many more clips does this cluster still need?
            already = (clips_per_cluster.get(int(grp), 0)
                       if clips_per_cluster is not None else 0)
            slots = max_clips - already
            if max_per_call is not None:
                slots = min(slots, max_per_call)
            if slots <= 0:
                continue
            grp_epochs = epochs[epochs.label == grp].copy()
            if grp_epochs.empty:
                continue
            cluster_dir = clips_root / f"cluster_{int(grp):02d}"
            cluster_dir.mkdir(parents=True, exist_ok=True)
            col_bgr     = _hex_to_bgr(_cmap(int(grp)))
            median_dur  = grp_epochs["duration_sec"].median()
            grp_epochs["_dist"] = (grp_epochs["duration_sec"] - median_dur).abs()
            # Take at most `slots` clips (not max_clips) so we don't overshoot
            # the per-cluster ceiling when clips_per_cluster is in use
            subset = grp_epochs.nsmallest(slots, "_dist")
            for clip_i, (_, row) in enumerate(subset.iterrows()):
                if len(pending) >= max_total_clips:
                    break
                sf = max(0, int(row.start_frame))
                ef = (min(total - 1, int(row.end_frame)) if total > 0
                      else int(row.end_frame))
                ef = min(ef, sf + max_src_frames - 1)
                if ef < sf:
                    continue   # start frame beyond video end (truncated file)
                label_text = (f"t={row.start_sec:.1f}s  "
                              f"dur={row.duration_sec:.2f}s")
                out_p = (cluster_dir /
                         f"cluster_{int(grp):02d}{animal_part}"
                         f"_example_{clip_i+1:02d}.mp4")
                pending.append((sf, ef, int(grp), col_bgr, label_text, out_p))

        # Sort by start frame so the cap moves forward through the video
        pending.sort(key=lambda x: x[0])

        # ── Write clips with sequential reads ─────────────────────────────────
        cap_pos        = -1   # estimated current decode position (-1 = unknown)
        written_per_grp: dict = {}
        for sf, ef, grp, col_bgr, label_text, out_p in pending:
            # Decide: seek to sf, or bridge the gap with forward reads?
            gap = sf - cap_pos
            if cap_pos < 0 or gap < 0 or gap > skip_threshold:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
                cap_pos = sf
            elif gap > 0:
                # Bridge small forward gap by discarding unneeded frames
                for _ in range(gap):
                    if not cap.read()[0]:
                        cap_pos = -1
                        break
                    cap_pos += 1
                if cap_pos < 0:
                    continue

            writer, out_p = _open_writer(out_p, output_fps, w, h)
            if writer is None:
                cap_pos = -1   # position uncertain after failed open
                continue
            try:
                out_frames = 0
                while cap_pos <= ef and out_frames < max_out_frames:
                    ret, frame = cap.read()
                    if not ret:
                        cap_pos = -1
                        break
                    cap_pos += 1
                    cv2.rectangle(frame, (0, 0), (230, 46), (0, 0, 0), -1)
                    cv2.putText(frame, f"Cluster {grp}",
                                (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, col_bgr, 2, cv2.LINE_AA)
                    cv2.putText(frame, label_text,
                                (8, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (200, 200, 200), 1)
                    writer.write(frame)
                    out_frames += 1
            finally:
                writer.release()
            if out_frames > 0:
                written_per_grp[grp] = written_per_grp.get(grp, 0) + 1

        # Update the shared state so the next animal's call knows what's done
        if clips_per_cluster is not None:
            for grp, n in written_per_grp.items():
                clips_per_cluster[grp] = clips_per_cluster.get(grp, 0) + n
    finally:
        cap.release()


def create_labeled_video(video_path, frame_labels: np.ndarray,
                          out_dir: Path, source_fps: float,
                          output_fps: int = 15):
    """Write the full session video with per-frame cluster label overlay."""
    try:
        import cv2
    except ImportError:
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step  = max(1, int(round(source_fps / output_fps)))
        stem  = Path(str(video_path)).stem + "_bsoid_labeled"
        writer, out_p = _open_writer(
            out_dir / "labeled_videos" / stem, output_fps, w, h)
        if writer is None:
            return
        try:
            # Read all frames sequentially — no per-frame seek.
            # For step > 1, non-output frames are decoded but not annotated
            # or written; this is still far cheaper than keyframe-seeking to
            # every step-th frame in a compressed stream.
            src_fi = 0
            while src_fi < total:
                ret, frame = cap.read()
                if not ret:
                    break
                if src_fi % step == 0:
                    lbl     = int(frame_labels[min(src_fi, len(frame_labels) - 1)])
                    col_bgr = _hex_to_bgr(_cmap(lbl))
                    cv2.rectangle(frame, (0, 0), (180, 36), (0, 0, 0), -1)
                    cv2.putText(frame, f"C{lbl}", (8, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                col_bgr, 2, cv2.LINE_AA)
                    writer.write(frame)
                src_fi += 1
        finally:
            writer.release()
    finally:
        cap.release()


def create_umap_evolution_video(
    video_path,
    embedding: "np.ndarray",
    umap_labels: "np.ndarray",
    frame_labels: "np.ndarray",
    source_fps: float,
    out_path: "Path",
    output_fps: float = 15.0,
    umap_panel_width: int = 640,
    elev: float = 20.0,
    azim: float = -60.0,
    palette: "list | None" = None,
    progress_cb=None,
) -> "Path | None":
    """Side-by-side video: original recording left, 3-D UMAP buildup right.

    The UMAP panel grows from 0 points to the full session cloud as the video
    plays.  Cumulative centroid-to-centroid arrows thicken with each observed
    transition (only above-chance transitions are ever drawn).

    Parameters
    ----------
    video_path        : source video file
    embedding         : (n_session_bins, 3) UMAP coords pre-sliced to this session
    umap_labels       : (n_session_bins,) cluster IDs
    frame_labels      : (n_video_frames,) per-frame cluster IDs
    source_fps        : frames-per-second of the source video
    out_path          : destination path for the output video (stem is kept)
    output_fps        : frames per second of the exported video
    umap_panel_width  : pixel width of the UMAP panel
    elev / azim       : initial 3-D viewpoint
    palette           : optional colour list (falls back to module PALETTE)
    progress_cb       : optional callable(phase_str, pct_float) for progress

    Returns the resolved output path, or None on failure.
    """
    try:
        import cv2
    except ImportError:
        return None
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        return None

    _pal = palette or PALETTE

    def _col(c: int) -> str:
        return _pal[int(c) % len(_pal)]

    def _hex_bgr(h: str) -> tuple:
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)

    embedding  = np.asarray(embedding, dtype=float)
    umap_labels = np.asarray(umap_labels, dtype=int)
    frame_labels = np.asarray(frame_labels, dtype=int)
    n_bins     = embedding.shape[0]
    if n_bins == 0:
        return None

    # 100 ms bin stride
    bin_stride = max(1, int(round(source_fps / 10.0)))

    valid = umap_labels >= 0
    uniq  = sorted(set(umap_labels[valid]))
    if not uniq:
        return None

    centroids = {u: embedding[valid & (umap_labels == u)].mean(axis=0)
                 for u in uniq}
    n_clusters = len(uniq)

    # Fixed axis limits (5 % padding)
    lo = embedding.min(axis=0)
    hi = embedding.max(axis=0)
    pad = (hi - lo) * 0.05 + 1e-6
    ax_lo, ax_hi = lo - pad, hi + pad

    # Chance floor for transition arrows
    chance_floor = 1.0 / max(1, n_clusters - 1)

    # Build sorted transition event list: (bin_idx, from_c, to_c)
    trans_events: list = []
    for fi in range(1, len(frame_labels)):
        fc, pc = int(frame_labels[fi]), int(frame_labels[fi - 1])
        if fc != pc and fc >= 0 and pc >= 0:
            trans_events.append((fi // bin_stride, pc, fc))

    # Cumulative transition counts array indexed by (from_c, to_c) as dict
    T_cum: dict = {}   # (from_c, to_c) → count

    # ── Phase 1: pre-render UMAP panel frames ────────────────────────────────
    if progress_cb:
        progress_cb("Pre-rendering UMAP frames", 0.0)

    fig_w_in = umap_panel_width / 100.0
    fig_h_in = fig_w_in          # square figure fills the panel better
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), facecolor=_BG, dpi=100)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.04, top=0.94)
    canvas = FigureCanvasAgg(fig)

    umap_frames: dict = {}   # bin_idx → numpy (H, W, 3)
    trans_cursor = 0         # index into trans_events

    for b in range(n_bins):
        # Advance cumulative transition counts up to bin b
        while trans_cursor < len(trans_events) and trans_events[trans_cursor][0] <= b:
            _, fc, tc = trans_events[trans_cursor]
            T_cum[(fc, tc)] = T_cum.get((fc, tc), 0) + 1
            trans_cursor += 1

        fig.clf()
        fig.patch.set_facecolor(_BG)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(_BG)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#333344")

        # Scatter points up to current bin — primary visual: opaque cloud.
        pts_so_far = embedding[:b + 1]
        lbl_so_far = umap_labels[:b + 1]
        for u in uniq:
            m = (lbl_so_far == u)
            if m.any():
                step = max(1, m.sum() // 8000)
                px = pts_so_far[m][::step]
                ax.scatter(px[:, 0], px[:, 1], px[:, 2],
                           s=14, alpha=0.82, color=_col(u),
                           depthshade=False, zorder=4)

        # Highlight current bin's point
        cur_lbl = int(umap_labels[b])
        if cur_lbl >= 0:
            cp = embedding[b]
            ax.scatter(*cp, s=70, color=_col(cur_lbl),
                       edgecolors=_TEXT_COL, linewidths=1.4,
                       zorder=8, depthshade=False)

        # Cumulative transition arrows (only above-chance).
        # Normalise thickness and opacity over the above-chance range so the
        # weakest visible arrow (just above chance_floor) maps to minimum
        # weight and the strongest to maximum — consistent with all other
        # transition plots in the pipeline.
        if T_cum:
            # Pre-compute per-source row sums for probability estimation
            src_totals = {}
            for (s, _t), v in T_cum.items():
                src_totals[s] = src_totals.get(s, 0) + v
            # Gather all above-chance probabilities to anchor normalisation
            above_probs = {}
            for (src, tgt), cnt in T_cum.items():
                if src not in centroids or tgt not in centroids:
                    continue
                p = cnt / max(1, src_totals.get(src, 1))
                if p > chance_floor:
                    above_probs[(src, tgt)] = p
            if above_probs:
                max_prob_ev   = max(above_probs.values())
                prob_range_ev = max(max_prob_ev - chance_floor, 1e-9)
                # Show only the top-10 above-chance transitions so arrows
                # stay secondary and don't crowd the cluster cloud.
                top_edges = sorted(above_probs.items(),
                                   key=lambda kv: kv[1], reverse=True)[:10]
                for (src, tgt), prob_approx in top_edges:
                    rel  = (prob_approx - chance_floor) / prob_range_ev
                    lw   = float(0.6 + rel * 1.4)     # 0.6–2.0 px
                    alph = float(0.18 + rel * 0.22)   # 0.18–0.40
                    s_c  = centroids[src]
                    t_c  = centroids[tgt]
                    d    = t_c - s_c
                    mid  = s_c + 0.80 * d
                    ax.plot([s_c[0], mid[0]], [s_c[1], mid[1]], [s_c[2], mid[2]],
                            color=_col(src), lw=lw, alpha=alph,
                            zorder=2, solid_capstyle="round")
                    dx, dy, dz = 0.20 * d
                    ax.quiver(mid[0], mid[1], mid[2], dx, dy, dz,
                              arrow_length_ratio=0.6, color=_col(src),
                              alpha=alph, linewidth=lw * 0.4,
                              normalize=False, zorder=3)

        ax.set_xlim(ax_lo[0], ax_hi[0])
        ax.set_ylim(ax_lo[1], ax_hi[1])
        ax.set_zlim(ax_lo[2], ax_hi[2])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("UMAP 1", fontsize=6, color=_TICK_COL, labelpad=1)
        ax.set_ylabel("UMAP 2", fontsize=6, color=_TICK_COL, labelpad=1)
        ax.set_zlabel("UMAP 3", fontsize=6, color=_TICK_COL, labelpad=1)
        ax.tick_params(colors=_TICK_COL, labelsize=5, pad=0)
        ax.set_title(f"UMAP  bin {b + 1}/{n_bins}", color=_TICK_COL,
                     fontsize=7, pad=2)

        canvas.draw()
        # buffer_rgba() works across matplotlib versions; tostring_rgb() was
        # deprecated in 3.8 and removed in 3.10 (would raise AttributeError here).
        buf = np.asarray(canvas.buffer_rgba())   # (H, W, 4)
        umap_frames[b] = buf[..., :3].copy()     # drop alpha → RGB

        if progress_cb and b % max(1, n_bins // 50) == 0:
            progress_cb("Pre-rendering UMAP frames", b / n_bins)

    plt.close(fig)
    if progress_cb:
        progress_cb("Pre-rendering UMAP frames", 1.0)

    # ── Phase 2: assemble output video ───────────────────────────────────────
    if progress_cb:
        progress_cb("Assembling video", 0.0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Cap source height at 720 px to keep the output manageable
        target_h = min(src_h, 720)
        scale    = target_h / src_h
        vid_w    = int(src_w * scale)

        out_w = vid_w + umap_panel_width
        out_h = target_h

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer, resolved = _open_writer(out_path, output_fps, out_w, out_h)
        if writer is None:
            return None

        step = max(1, int(round(source_fps / output_fps)))

        try:
            src_fi = 0
            while src_fi < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if src_fi % step == 0:
                    bin_idx = min(src_fi // bin_stride, n_bins - 1)

                    # Left panel: original video (resized)
                    if scale != 1.0:
                        vid_panel = cv2.resize(frame, (vid_w, target_h))
                    else:
                        vid_panel = frame

                    # Right panel: pre-rendered UMAP frame (resized to panel)
                    umap_img = umap_frames[bin_idx]
                    umap_bgr = cv2.cvtColor(umap_img, cv2.COLOR_RGB2BGR)
                    umap_panel = cv2.resize(umap_bgr, (umap_panel_width, target_h))

                    combined = np.concatenate([vid_panel, umap_panel], axis=1)

                    # Overlay: cluster label and timestamp
                    t_sec   = src_fi / source_fps
                    mins    = int(t_sec // 60)
                    secs    = int(t_sec % 60)
                    lbl     = int(frame_labels[min(src_fi, len(frame_labels) - 1)])
                    col_bgr = _hex_bgr(_col(lbl))

                    # ── Large cluster label centred at the top of the left panel ──
                    label_text = f"Cluster {lbl}"
                    _font      = cv2.FONT_HERSHEY_DUPLEX
                    _fscale    = max(1.2, target_h / 500.0)
                    _thick     = max(2, int(_fscale * 2))
                    (tw, th_t), _bl = cv2.getTextSize(
                        label_text, _font, _fscale, _thick)
                    tx = max(0, (vid_w - tw) // 2)
                    ty = th_t + 16
                    # Dark semi-transparent backing strip
                    overlay = combined.copy()
                    cv2.rectangle(overlay,
                                  (0, 0), (vid_w, ty + _bl + 12),
                                  (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.55, combined, 0.45, 0, combined)
                    # White outline for contrast on any background
                    cv2.putText(combined, label_text, (tx, ty),
                                _font, _fscale, (30, 30, 30), _thick + 3,
                                cv2.LINE_AA)
                    # Coloured fill
                    cv2.putText(combined, label_text, (tx, ty),
                                _font, _fscale, col_bgr, _thick, cv2.LINE_AA)

                    # ── Small timestamp at bottom-left ──
                    cv2.rectangle(combined, (0, target_h - 28),
                                  (200, target_h), (0, 0, 0), -1)
                    cv2.putText(combined,
                                f"t={mins:02d}:{secs:02d}",
                                (8, target_h - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (200, 200, 200), 1, cv2.LINE_AA)
                    writer.write(combined)

                    if progress_cb and src_fi % max(1, total_frames // 50) == 0:
                        progress_cb("Assembling video", src_fi / total_frames)
                src_fi += 1
        finally:
            writer.release()
    finally:
        cap.release()

    if progress_cb:
        progress_cb("Assembling video", 1.0)
    return resolved


#
#  FILE DISCOVERY UTILITIES
#

def _is_bsoid_ready_h5(p: Path) -> bool:
    """True for *_filtered.h5 DLC outputs that haven't been BSOID-processed."""
    stem = p.stem
    if p.name.startswith("BSOID_"):
        return False
    if not stem.endswith("_filtered"):
        return False
    if "UN_filtered" in stem:
        return False
    return True


def _is_any_dlc_h5(p: Path) -> bool:
    """Fallback: any DLC H5 that is not a BSOID output (unfiltered included)."""
    return (not p.name.startswith("BSOID_")
            and "UN_filtered" not in p.stem
            and "bout_lengths" not in p.name.lower())


def find_dlc_files(folder) -> list:
    """
    Recursively find BSOID-ready DLC files (CSV / H5).
    Preference order: *_filtered.h5 > *_filtered.csv > any DLC file.
    """
    folder = Path(folder)
    filtered_h5  = sorted(p for p in folder.rglob("*.h5")
                          if _is_bsoid_ready_h5(p))
    filtered_csv = sorted(p for p in folder.rglob("*.csv")
                          if _is_bsoid_ready_h5(p))
    if filtered_h5:
        return filtered_h5
    if filtered_csv:
        return filtered_csv
    # fallback: any CSV/H5 that isn't an engine output
    return sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in DLC_EXTS
        and not p.name.startswith("BSOID_")
        and "UN_filtered" not in p.stem
        and "bout_lengths" not in p.name.lower()
    )


def find_videos(folder) -> dict:
    """Return {stem: Path} for all video files in the folder."""
    result = {}
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in VIDEO_EXTS:
            result[p.stem] = p
    return result


def pair_files(dlc_files: list, video_dict: dict) -> list:
    """
    Match DLC files to video files by stem / timestamp / prefix.
    Returns [(dlc_path, video_path_or_None), ...]
    """
    pairs = []
    for dp in dlc_files:
        stem  = dp.stem
        # 1. exact match
        vp = video_dict.get(stem)
        # 2. prefix match
        if not vp:
            for vstem, v in video_dict.items():
                if stem.startswith(vstem) or vstem.startswith(stem):
                    vp = v
                    break
        # 3. shared YYYYMMDD_HHMMSS timestamp
        if not vp:
            m = _TS_RE.search(stem)
            if m:
                ts = m.group(1)
                for vstem, v in video_dict.items():
                    if ts in vstem:
                        vp = v
                        break
        pairs.append((dp, vp))
    return pairs


#  
#  DLC PRE-PROCESSING  (from General_DLC_2_BSOID_DAMIEN_v5)
#  

BSOID_OUTPUT_ROOT   = "BSOID_Project_Ready"
BSOID_H5_SUBDIR     = "h5"
BSOID_CSV_SUBDIR    = "csv"
BSOID_VIDEO_SUBDIR  = "videos"
BSOID_ANALYSIS_SUBDIR = "output"
BSOID_SCORER_NAME   = "DLC_SuperAnimal"
MIN_BODYPART_CONFIDENCE = 0.35


def analyze_session_confidence(h5_path: Path) -> dict | None:
    """Returns {bodypart: {mean, median}} or None on failure."""
    try:
        df = pd.read_hdf(str(h5_path))
        df = _normalise_dlc_df(df)
        bparts = df.columns.get_level_values("bodyparts").unique()
        stats  = {}
        for bp in bparts:
            sub = df.xs(bp, level="bodyparts", axis=1)
            if sub.columns.nlevels > 1:
                sub.columns = sub.columns.get_level_values(-1)
            llh = pd.to_numeric(
                sub.get("likelihood", pd.Series(np.nan, index=df.index)),
                errors="coerce")
            stats[bp] = {"mean": float(llh.mean()), "median": float(llh.median())}
        return stats
    except Exception as e:
        print(f"  [WARN] Could not analyze {h5_path.name}: {e}")
        return None


def save_bsoid_h5_csv(df: pd.DataFrame, h5_dst: Path,
                       csv_dst: Path, bodyparts_to_keep: list):
    """Filter to conserved bodyparts and save H5 + CSV in B-SOiD format."""
    df = df.loc[:, (slice(None), bodyparts_to_keep, slice(None))]
    new_cols = pd.MultiIndex.from_product(
        [[BSOID_SCORER_NAME], bodyparts_to_keep, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df = df.copy()
    df.columns = new_cols
    df.index   = range(len(df))
    h5_dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(str(h5_dst), key="df_with_missing", mode="w", format="fixed")
    csv_dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(csv_dst))


def bsoid_short_name(h5_path: Path, folder_name: str,
                      max_folder_len: int = 28) -> str:
    """
    Build a short, filesystem-safe session name.
    Caps folder_name at max_folder_len characters to prevent Windows MAX_PATH
    (260-char) errors caused by long cage/condition folder names.
    Format: <truncated_folder>_<YYYYMMDD_HHMMSS>
    """
    m  = _TS_RE.search(h5_path.stem)
    ts = m.group(1) if m else ""
    short_folder = folder_name[:max_folder_len].rstrip("_- ")
    if ts:
        return f"{short_folder}_{ts}"
    # No YYYYMMDD_HHMMSS found — use the session stem so every file gets a unique name
    stem = h5_path.stem
    if stem.endswith("_filtered"):
        stem = stem[:-len("_filtered")]
    return stem[:80] if stem else short_folder


def cleanup_video_byproducts(results_folder: Path, log_fn=print):
    """
    Incremental cleanup: delete pseudo_* folders and DLC-generated .json files
    from a single video's _results folder immediately after inference completes.
    Call once per video inside the DLC loop to keep disk usage low and avoid
    Windows MAX_PATH errors from long pseudo_resized_* directory names.
    """
    rf = Path(results_folder)
    if not rf.exists():
        return
    pseudo_n = json_n = 0
    for entry in sorted(rf.iterdir()):
        if entry.is_dir() and entry.name.startswith("pseudo_"):
            try:
                shutil.rmtree(str(entry))
                log_fn(f"  [cleanup] Removed: {entry.name}")
                pseudo_n += 1
            except Exception as e:
                log_fn(f"  [cleanup] Could not remove {entry.name}: {e}")
    for jf in sorted(rf.glob("*.json")):
        try:
            jf.unlink()
            json_n += 1
        except Exception:
            pass
    if pseudo_n or json_n:
        log_fn(f"  [cleanup] {pseudo_n} pseudo dir(s), {json_n} JSON(s) removed "
               f"from {rf.name}.")


def cleanup_dlc_byproducts(root_path: Path, log_fn=print):
    """Global cleanup: delete pseudo_* folders and .json files left by DLC."""
    pseudo_count = json_count = 0
    search_dirs  = {root_path.resolve()}
    for item in root_path.rglob("*"):
        if item.is_dir() and item.name.endswith("_results"):
            search_dirs.add(item.resolve())
    for folder in sorted(search_dirs):
        if not folder.exists():
            continue
        for entry in sorted(folder.iterdir()):
            if entry.is_dir() and entry.name.startswith("pseudo_"):
                try:
                    shutil.rmtree(str(entry))
                    log_fn(f"  [cleanup] Deleted pseudo folder: {entry.name}")
                    pseudo_count += 1
                except Exception as e:
                    log_fn(f"  [cleanup] Could not delete {entry.name}: {e}")
        for jf in sorted(folder.glob("*.json")):
            try:
                jf.unlink()
                json_count += 1
            except Exception:
                pass
    log_fn(f"  [cleanup] {pseudo_count} pseudo folder(s), {json_count} JSON file(s) removed.")


def run_bsoid_prep(source_folder, log_fn=print,
                   min_confidence: float = MIN_BODYPART_CONFIDENCE,
                   conf_metric: str = "median",
                   min_session_frac: float = 0.6,
                   min_keep: int = 6) -> Path | None:
    """
    Scan source_folder for *_filtered.h5 files, filter to conserved bodyparts,
    export BSOID_Project_Ready/ structure.

    Bodypart conservation policy (single-view-camera friendly)
    ----------------------------------------------------------
    A bodypart is conserved if its per-session confidence (``conf_metric``,
    "median" or "mean") is >= ``min_confidence`` in at least ``min_session_frac``
    of the folder's sessions — NOT in every session.  This prevents one
    poorly-tracked recording from deleting a keypoint for the whole group, which
    is common with single-view setups where occlusion transiently tanks the mean.
    If fewer than ``min_keep`` bodyparts pass, the top-N keypoints by confidence
    are kept instead (with a warning) so the feature space never silently
    collapses to a handful of points.

    Returns path to BSOID_Project_Ready or None on failure.
    """
    root_path   = Path(source_folder)
    folder_name = root_path.name
    bsoid_root  = root_path / BSOID_OUTPUT_ROOT

    for sd in [BSOID_H5_SUBDIR, BSOID_CSV_SUBDIR,
               BSOID_VIDEO_SUBDIR, BSOID_ANALYSIS_SUBDIR]:
        (bsoid_root / sd).mkdir(parents=True, exist_ok=True)

    # Collect H5 sessions — prefer *_filtered.h5, fall back to any DLC H5
    sessions = []
    # Layout A: flat files in root
    flat = [p for p in sorted(root_path.glob("*.h5"))
            if _is_bsoid_ready_h5(p)]
    if not flat:
        flat = [p for p in sorted(root_path.glob("*.h5"))
                if _is_any_dlc_h5(p)]
    if flat:
        sessions = [(p.stem, p) for p in flat]
        log_fn(f"  Layout: FLAT - {len(sessions)} H5 file(s)")
    else:
        # Layout B: nested *_results/
        for sub in sorted(f for f in root_path.iterdir()
                          if f.is_dir() and f.name != BSOID_OUTPUT_ROOT):
            cands = [h for h in sub.rglob("*.h5") if _is_bsoid_ready_h5(h)]
            if not cands:
                cands = [h for h in sub.rglob("*.h5") if _is_any_dlc_h5(h)]
            if cands:
                cands.sort(key=lambda p: (1 if p.stem.endswith("_filtered") else 0),
                           reverse=True)
                sessions.append((sub.name, cands[0]))
        if sessions:
            log_fn(f"  Layout: NESTED - {len(sessions)} H5 file(s)")

    if not sessions:
        log_fn("  ERROR: No *_filtered.h5 files found.")
        return None

    # Analyze confidence
    all_stats = {}
    for sname, h5p in sessions:
        s = analyze_session_confidence(h5p)
        if s:
            all_stats[sname] = (h5p, s)
            log_fn(f"  Analyzed: {h5p.name}")

    if not all_stats:
        log_fn("  ERROR: All H5 files failed to parse.")
        return None

    # ── Determine conserved bodyparts ─────────────────────────────────────────
    # Universe of bodyparts = union across sessions, preserving first-session
    # order then appending any extras, so a stable column order is kept.
    all_bps_seen: list = []
    for _, ss in all_stats.values():
        for bp in ss.keys():
            if bp not in all_bps_seen:
                all_bps_seen.append(bp)
    n_sessions = len(all_stats)
    _metric = "median" if str(conf_metric).lower().startswith("med") else "mean"

    def _pass_frac(bp: str) -> float:
        n_pass = sum(1 for _, ss in all_stats.values()
                     if bp in ss and ss[bp][_metric] >= min_confidence)
        return n_pass / max(1, n_sessions)

    conserved = [bp for bp in all_bps_seen
                 if _pass_frac(bp) >= float(min_session_frac)]

    # Floor: never collapse below min_keep usable keypoints.  If the threshold
    # rule leaves too few, fall back to the top-N bodyparts ranked by mean
    # per-session confidence so the analysis keeps a workable feature set.
    if len(conserved) < int(min_keep) and all_bps_seen:
        def _mean_conf(bp: str) -> float:
            vals = [ss[bp][_metric] for _, ss in all_stats.values() if bp in ss]
            return float(np.mean(vals)) if vals else 0.0
        ranked = sorted(all_bps_seen, key=_mean_conf, reverse=True)
        target_n = max(int(min_keep), len(conserved))
        conserved = ranked[:target_n]
        log_fn(f"  [VALID-WARN] Only {sum(1 for bp in all_bps_seen if _pass_frac(bp) >= float(min_session_frac))} "
               f"bodypart(s) passed {_metric} >= {min_confidence} in "
               f">= {float(min_session_frac):.0%} of sessions; falling back to "
               f"top-{len(conserved)} by confidence to keep a usable feature set.")

    log_fn(f"  Conserved bodyparts ({len(conserved)}/{len(all_bps_seen)}; "
           f"metric={_metric}, thresh={min_confidence}, "
           f"min_sess_frac={float(min_session_frac):.0%}): {conserved}")

    if not conserved:
        log_fn(f"  ERROR: No bodyparts passed threshold {min_confidence}.")
        return None

    # Export
    for sname, (h5_src, _) in all_stats.items():
        short = bsoid_short_name(h5_src, folder_name)
        h5_dst  = bsoid_root / BSOID_H5_SUBDIR  / f"{short}.h5"
        csv_dst = bsoid_root / BSOID_CSV_SUBDIR / f"{short}.csv"
        try:
            df = pd.read_hdf(str(h5_src))
            df = _normalise_dlc_df(df)
            save_bsoid_h5_csv(df, h5_dst, csv_dst, conserved)
            log_fn(f"  Exported: {short}")
        except Exception as e:
            log_fn(f"  ERROR exporting {h5_src.name}: {e}")
            continue

        # Copy video for example-clip generation.
        # Priority: labeled/after_adapt → inference copy in _results/ → source video.
        m = _TS_RE.search(h5_src.stem)
        video_src = None
        if m:
            ts = m.group(1)
            for pattern in [f"*{ts}*after_adapt*.mp4",
                            f"*{ts}*labeled*.mp4"]:
                video_src = next(h5_src.parent.glob(pattern), None)
                if video_src:
                    break
        if not video_src:
            video_src = (next(h5_src.parent.glob("*after_adapt*.mp4"), None) or
                         next(h5_src.parent.glob("*_labeled.mp4"), None))
        if not video_src:
            # Inference copy: resized_<stem>.mp4 or <stem>.mp4 inside _results/
            base_stem = h5_src.stem.replace("_filtered", "")
            for pat in (f"resized_{base_stem}.mp4", f"{base_stem}.mp4"):
                p = h5_src.parent / pat
                if p.is_file():
                    video_src = p
                    break
        if not video_src:
            # Any MP4 in the _results folder (skip pseudo/before_adapt artifacts)
            for p in sorted(h5_src.parent.glob("*.mp4")):
                if not any(x in p.name for x in ("before_adapt", "pseudo")):
                    video_src = p
                    break
        if not video_src:
            # Original source video in the parent of the _results folder
            base_stem = h5_src.stem.replace("_filtered", "")
            parent_dir = h5_src.parent.parent
            for ext in VIDEO_EXTS:
                p = parent_dir / (base_stem + ext)
                if p.is_file():
                    video_src = p
                    break
        if not video_src:
            # Widest fallback: any video in the source folder root
            parent_dir = h5_src.parent.parent
            for p in sorted(parent_dir.glob("*")):
                if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("resized_"):
                    video_src = p
                    break
        if video_src:
            dst = bsoid_root / BSOID_VIDEO_SUBDIR / f"{short}.mp4"
            if not dst.exists():
                try:
                    shutil.copy2(str(video_src), str(dst))
                    log_fn(f"  Video copied: {dst.name}  (source: {video_src.name})")
                except Exception as _e:
                    log_fn(f"  [WARN] Could not copy video {video_src.name}: {_e}")
            else:
                log_fn(f"  Video already present: {dst.name}")
        else:
            log_fn(f"  [WARN] No video found for {short} — example clips will be skipped")

    # Confidence report
    rows = []
    for sname, (h5_src, sess_stats) in all_stats.items():
        short = bsoid_short_name(h5_src, folder_name)
        row   = {"Session": short}
        for bp in conserved:
            row[f"{bp}_mean"]   = sess_stats[bp]["mean"]
            row[f"{bp}_median"] = sess_stats[bp]["median"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        str(bsoid_root / "bodypart_confidence_report.csv"), index=False)

    cleanup_dlc_byproducts(root_path, log_fn)
    log_fn(f"  B-SOiD prep complete -> {bsoid_root}")
    return bsoid_root


#  
#  MAIN ENGINE CLASS
#  

class BSoidEngine:
    """
    Programmatic B-SOiD pipeline - no GUI required.

    Parameters
    ----------
    csv_folder   : folder with BSOID-ready CSV or H5 files
    video_folder : folder with source videos (matched by stem/timestamp)
    output_dir   : all outputs written here
    fps          : recording FPS (if None: guessed from filename or 30)
    logger       : PipelineLogger or any callable(str)
    progress_cb  : callable(current: int, total: int) - optional
    cfg          : dict of hyperparameter overrides
    """

    # ── Defaults ───────────────────────────────────────────────────────────────
    # Where CUBE follows B-SOiD (Hsu A.I. & Yttri E.A., 2021, Nat. Commun.
    # 12:5188) the reference value is used (likelihood_thresh, umap_n_neighbors,
    # umap_n_components, mlp_hidden, train_frac).  CUBE-specific parameters
    # (V2 multi-scale/angular features, umap_min_dist, the HDBSCAN sweep, the
    # post-hoc HMM) are intentional extensions — see the run() [AUDIT] block,
    # which reports them separately rather than claiming exact faithfulness.
    # Only min_epoch_dur_s / max_epoch_dur_s are intended as primary user inputs.
    # This is the canonical default set; the GUI (cube.py) derives from it.
    DEFAULTS = dict(
        likelihood_thresh     = 0.3,    # DLC confidence threshold (pub. default)
        max_interp_gap_sec    = 0.5,    # occlusions longer than this are held
                                        # flat, not ramped (0 = legacy interp)
        feature_bad_bp_thresh = 0.40,   # drop bodyparts whose mean bad-frame
                                        # fraction (< likelihood_thresh) across
                                        # all sessions meets or exceeds this
                                        # value.  0 = disabled.  Single-camera
                                        # rigs typically occlude back limbs 40-
                                        # 70% of the time; excluding those keeps
                                        # the feature space clean and prevents
                                        # HDBSCAN noise inflation.
        feature_dedup_jitter  = 1e-4,   # tiny Gaussian noise added to the UMAP
                                        # training subsample (raw feature units)
                                        # BEFORE standardisation.  Breaks any
                                        # exact-duplicate feature vectors so
                                        # HDBSCAN's mutual-reachability graph
                                        # stays non-degenerate.
                                        # 1e-4 raw units → ~2e-6 post-scaling
                                        # for typical distance features — fully
                                        # imperceptible to cluster geometry.
        flat_held_bp_frac_thresh = 0.5, # fraction of kept bodyparts that must be
                                        # simultaneously flat-held in a 100 ms bin
                                        # before that bin is excluded from UMAP /
                                        # HDBSCAN training.  0.5 = majority occlusion
                                        # (whole-animal dropout).  Set to 0.0 to
                                        # disable exclusion entirely.  Single-camera
                                        # recordings with natural individual-bodypart
                                        # occlusion rarely exceed 0.15–0.25, so the
                                        # 0.5 default leaves those bins in training.
                                        # 0 = disabled.
        boxcar_win_sec        = 0.07,   # 70 ms boxcar smoothing (pub. default)
        train_frac            = 0.3,    # fraction of bins for UMAP when N > threshold
        umap_full_thresh      = 10_000, # use full data for UMAP when N <= this
        umap_n_neighbors      = 0,      # 0 = auto (scales with recording length); >0 = fixed
        umap_n_components     = 3,      # 3-D UMAP embedding (pub. default)
        umap_min_dist         = 0.1,    # UMAP/B-SOiD convention. (<0.05 packs points so
                                        # tightly that HDBSCAN's DBCV becomes non-finite —
                                        # see the degenerate-DBCV fallback in run_hdbscan.
                                        # compat_mode="legacy_v2" restores the old 0.0.)
        umap_random_state     = 42,     # reproducibility seed
        hdbscan_metric        = "euclidean",  # (pub. default)
        hdbscan_method        = "eom",        # excess-of-mass (pub. default)
        mlp_hidden            = "100,50",     # 2-layer MLP (pub. default)
        mlp_max_iter          = 1000,
        mlp_confidence_thresh = 0.0,    # 0 = always assign; >0 = low-confidence
                                        # bins become -1 (unclassified) at inference
        cv_folds              = 5,
        # ── HDBSCAN options ───────────────────────────────────────────────────
        hdbscan_methods_to_try = "eom,leaf",  # both tried; selection logic picks best
        # ── Cluster count guidance ────────────────────────────────────────────
        target_n_clusters     = 0,    # 0 = auto; >0 = user-requested cluster count
        preferred_clusters_lo = 12,   # auto-mode: prefer cluster count ≥ this
        preferred_clusters_hi = 30,   # auto-mode: prefer cluster count ≤ this
        # ── Rare-cluster pruning ──────────────────────────────────────────────
        # Clusters whose share of total analysis time is below this threshold
        # are merged into noise before MLP training.  Prevents fragment clusters
        # driven by a handful of frames from polluting the behaviour space.
        # Expressed as a percentage of total bins (0.2 = 0.2 %).  Set to 0 to disable.
        min_cluster_freq      = 0.2,
        # ── HDBSCAN sweep tuning ──────────────────────────────────────────────
        # hdbscan_pct_lo : lower bound for the min_cluster_size sweep, in units
        #   of 0.1%-of-bins.  0 = auto (≥ 0.2% of bins).  Increase to 5 to restore
        #   the original 0.5%-of-bins floor if too many noise-clusters appear.
        # hdbscan_pct_hi : upper bound (5.0% of bins).  Rarely needs adjustment.
        # hdbscan_dbcv_thresh : candidates must have DBCV ≥ this fraction of the
        #   best observed DBCV.  Lower = more cluster-diversity tolerated (0.65),
        #   higher = stricter quality gate (0.75 was the original hardcoded value).
        # hdbscan_diversity_bonus : weight added to DBCV when ranking solutions that
        #   have heterogeneous cluster sizes.  0.10 gives a 10% bonus per unit of
        #   size CV, rewarding solutions that include both brief and sustained clusters.
        hdbscan_pct_lo          = 0,    # 0 = auto; >0 overrides (0.1%-of-bins units)
        hdbscan_pct_hi          = 50,   # 5.0% of bins
        hdbscan_dbcv_thresh     = 0.65, # DBCV fraction — was hardcoded 0.75
        hdbscan_diversity_bonus = 0.10, # cluster-size CV reward weight
        # ── Feature options ───────────────────────────────────────────────────
        body_normalise        = False,  # normalise by nose-to-tailbase length
        pca_pre_reduce        = "auto", # auto/on/off — reduce dims before UMAP
        # ── Primary user inputs (bout duration filter) ────────────────────────
        min_epoch_dur_s       = 0.0,    # minimum cluster bout duration (seconds)
        max_epoch_dur_s       = 1e9,    # maximum cluster bout duration (seconds)
        # ── Output options ────────────────────────────────────────────────────
        output_fps            = 15,
        max_clips_per_cluster = 3,
        umap_evolution_n      = 1,      # side-by-side UMAP-evolution videos to
                                        # auto-export after clustering (0 = off)
        save_plots            = True,
        save_videos           = True,
        save_example_clips    = True,
        save_labeled_video    = True,
        delete_labeled_videos = True,   # delete labeled_videos/ folder after run
        # ── Plot appearance ───────────────────────────────────────────────────
        plot_theme            = "dark",   # "dark" or "light"
        # ── HMM post-hoc smoothing ────────────────────────────────────────────
        hmm_enabled           = True,    # wrap MLP output with Multinomial HMM
        hmm_n_states          = 0,       # 0/None → n_clusters (smoothing-only mode)
        hmm_n_iter            = 100,     # Baum-Welch EM iterations
        hmm_min_prob          = 0.05,    # min edge probability in syntax network plot
        # ── Reproducibility / methodology (v2.1) ──────────────────────────────
        # compat_mode: "current" uses the v2.1 corrected behaviour; "legacy_v2"
        #   restores pre-2.1 numeric defaults (see _LEGACY_V2_DEFAULTS) so an old
        #   run can be reproduced exactly.  Only keys the user did NOT pass
        #   explicitly are reverted.
        compat_mode           = "current",
        # hdbscan_mcs_anchor: "embedding" sizes min_cluster_size against the
        #   points actually clustered (correct when UMAP runs on a subsample);
        #   "full" anchors against the full bin count (pre-2.1 behaviour).
        hdbscan_mcs_anchor    = "embedding",
        # angular_fallback: when no spine landmarks match by keyword, True uses
        #   evenly-spaced bodypart indices (pre-2.1; can yield meaningless angles),
        #   False skips the angular block entirely (v2.1 default).
        angular_fallback      = False,
        # seed_sweep_n: if >0, re-run UMAP+HDBSCAN over this many random seeds to
        #   assess cluster-count / partition stability (plots cluster_stability.png).
        #   0 = off (no extra runtime).
        seed_sweep_n          = 0,
    )

    # Pre-2.1 numeric defaults, restored when cfg["compat_mode"] == "legacy_v2"
    # for keys the caller did not set explicitly.  Keep in sync with the
    # corrected DEFAULTS above.
    _LEGACY_V2_DEFAULTS = dict(
        umap_min_dist         = 0.0,
        hdbscan_mcs_anchor    = "full",
        angular_fallback      = True,
    )

    def __init__(self, csv_folder, video_folder=None,
                 output_dir="bsoid_output", fps=None,
                 logger=None, progress_cb=None, stage_cb=None, cfg=None):
        # Accept csv_folder as a single path OR a list of paths (combined analysis)
        if isinstance(csv_folder, (str, Path)):
            self._csv_folders = [Path(csv_folder)]
        else:
            self._csv_folders = [Path(f) for f in csv_folder]
        self.csv_folder = self._csv_folders[0]   # backward-compat alias

        # Accept video_folder as a single path OR a list of paths
        if video_folder is None:
            self._vid_folders = []
            self.video_folder = None
        elif isinstance(video_folder, (str, Path)):
            self._vid_folders = [Path(video_folder)]
            self.video_folder = self._vid_folders[0]
        else:
            self._vid_folders = [Path(f) for f in video_folder if f is not None]
            self.video_folder = self._vid_folders[0] if self._vid_folders else None

        self.output_dir   = Path(output_dir)
        self._fps_arg     = fps
        self._log         = logger or print
        self._prog        = progress_cb or (lambda c, t: None)
        self._stage       = stage_cb   or (lambda s, d="": None)
        self._cfg         = {**self.DEFAULTS, **(cfg or {})}
        # Reproducibility: in legacy mode, revert v2.1 numeric changes for any
        # key the caller did not pass explicitly (explicit overrides win).
        if self._cfg.get("compat_mode") == "legacy_v2":
            _user_keys = set((cfg or {}).keys())
            for _k, _v in self._LEGACY_V2_DEFAULTS.items():
                if _k not in _user_keys:
                    self._cfg[_k] = _v

        # sub-dirs
        self._out_bouts  = self.output_dir / "bout_lengths"
        self._out_model  = self.output_dir / "model"
        self._out_plots  = self.output_dir / "plots"
        self._out_videos = self.output_dir / "videos"
        for d in (self._out_bouts, self._out_model,
                  self._out_plots, self._out_videos):
            d.mkdir(parents=True, exist_ok=True)

    #   main entry point  

    def run(self) -> dict:
        """Run the full V2 pipeline. Returns a results dict."""
        # Apply plot theme before any figure is drawn (updates _BG/_PANEL/_TEXT_COL/_TICK_COL)
        _apply_plot_theme(self._cfg.get("plot_theme", "dark"))

        self._log("=" * 64)
        self._log(f"  CUBE Engine  v{VERSION}")
        self._log("=" * 64)

        # ── Faithfulness audit: VERIFY against published reference values ──────
        # Each entry: cfg_key -> (published_value, comparator).  The audit
        # actively compares the live cfg to the reference and flags any
        # deviation, instead of printing a hardcoded "(pub: …)" string that
        # could assert faithfulness that isn't there.
        self._log("\n[AUDIT] B-SOiD reference parameter verification"
                  " (Hsu A.I. & Yttri E.A., 2021, Nat. Commun. 12:5188)")
        # B-SOiD reference values for the parameters CUBE keeps faithful to the
        # original pipeline.  CUBE-specific parameters (V2 multi-scale features,
        # adaptive n_neighbors, umap_min_dist, the HDBSCAN sweep strategy, the
        # post-hoc HMM) are intentional extensions and are NOT audited here —
        # they are reported separately below so faithfulness is never overstated.
        _PUB_REF = {
            "likelihood_thresh": 0.3,
            "umap_n_neighbors":  60,
            "umap_n_components": 3,
            "mlp_hidden":        "100,50",
            "train_frac":        0.3,
        }
        _n_mismatch = 0
        for _k, _pub in _PUB_REF.items():
            _val = self._cfg.get(_k)
            # n_neighbors == 0 means "auto" (resolved later from recording length)
            if _k == "umap_n_neighbors" and int(_val or 0) <= 0:
                self._log(f"  {_k:18s}: auto  (pub: {_pub}; resolved from "
                          f"recording length below)")
                continue
            try:
                _match = (abs(float(_val) - float(_pub)) < 1e-9
                          if isinstance(_pub, (int, float))
                          else str(_val) == str(_pub))
            except (TypeError, ValueError):
                _match = str(_val) == str(_pub)
            _flag = "OK " if _match else "!! "
            if not _match:
                _n_mismatch += 1
            self._log(f"  {_flag}{_k:18s}: {_val}  (pub: {_pub})")
        if _n_mismatch:
            self._log(f"  [AUDIT] {_n_mismatch} reference parameter(s) DEVIATE "
                      f"(marked !! above) — intentional overrides are fine but "
                      f"should be reported in methods.")
        else:
            self._log("  [AUDIT] All audited B-SOiD reference parameters match.")
        # CUBE-specific parameters — reported, not audited against B-SOiD.
        self._log("  [AUDIT] CUBE-specific (not in the B-SOiD reference): "
                  "V2 multi-scale features, angular features, post-hoc HMM.")
        self._log(f"    umap_min_dist     : {self._cfg.get('umap_min_dist')}  "
                  f"(CUBE-tuned; <0.05 can make HDBSCAN DBCV non-finite)")
        self._log(f"    hdbscan_mcs_anchor: {self._cfg.get('hdbscan_mcs_anchor')}"
                  f"  |  sweep {self._cfg.get('hdbscan_pct_lo') or 'auto'}–"
                  f"{self._cfg.get('hdbscan_pct_hi')} (0.1%-of-N units), "
                  f"min_samples=max(5, mcs//5)")
        self._log(f"    analysis_version  : {ANALYSIS_VERSION}  "
                  f"(compat_mode={self._cfg.get('compat_mode', 'current')})")
        _bn  = self._cfg.get("body_normalise", False)
        self._log(f"  Feature engine    : V2  (fps-adaptive scales, "
                  f"body_normalise={_bn}, angular)")

        _validation: dict = {}

        # 1. Discover & pair files  (all csv/video folders combined)
        self._log("\n[1/7]  Discovering files...")
        self._stage("1/7 — Discovering files")

        # Collect DLC files from every csv folder (combined multi-group analysis)
        dlc_files: list = []
        for _csv_dir in self._csv_folders:
            dlc_files.extend(find_dlc_files(_csv_dir))
        # Deduplicate by resolved path (in case folders overlap)
        _seen_dlc: set = set()
        _uniq_dlc: list = []
        for _f in dlc_files:
            _key = str(_f.resolve())
            if _key not in _seen_dlc:
                _seen_dlc.add(_key)
                _uniq_dlc.append(_f)
        dlc_files = _uniq_dlc

        # Collect videos from every video folder
        vid_dict: dict = {}
        for _vid_dir in self._vid_folders:
            vid_dict.update(find_videos(_vid_dir))

        pairs    = pair_files(dlc_files, vid_dict)
        n_paired = sum(1 for _, v in pairs if v)
        self._log(f"  {len(self._csv_folders)} csv folder(s), "
                  f"{len(dlc_files)} DLC file(s), {len(vid_dict)} video(s), "
                  f"{n_paired} paired")
        if not dlc_files:
            raise FileNotFoundError(
                f"No BSOID-ready files found in: "
                f"{', '.join(str(d) for d in self._csv_folders)}")

        # Validation gate 1: DLC quality
        try:
            _validation["dlc_quality"] = validate_dlc_quality(
                dlc_files, self._cfg["likelihood_thresh"])
            for w in _validation["dlc_quality"]["warnings"]:
                self._log(f"  [VALID-WARN] {w}")
        except Exception as e:
            self._log(f"  [VALID] DLC quality check failed: {e}")

        # 2. Load & smooth
        self._log("\n[2/7]  Loading & smoothing...")
        self._stage("2/7 — Loading DLC files", f"0/{len(pairs)}")
        all_xy, all_names, all_fps_list, all_bps, all_vpaths = [], [], [], [], []
        all_groups: list = []   # input folder each session came from (for export coverage)
        all_bp_bad_fracs: dict = {}  # bodypart -> list of per-session bad-frame fracs
        all_flat_held: list = []     # per-session per-bodypart flat-held masks (list[list[np.ndarray]])

        def _group_key(_fp: Path) -> str:
            # Which uploaded csv folder does this DLC file belong to?  Used so the
            # UMAP-evolution export can guarantee at least one video per folder.
            try:
                _fpr = _fp.resolve()
                for _cd in self._csv_folders:
                    _cdr = Path(_cd).resolve()
                    if _cdr == _fpr or _cdr in _fpr.parents:
                        return str(_cdr)
            except Exception:
                pass
            return str(_fp.parent)

        for i, (fp, _vp) in enumerate(pairs):
            self._log(f"  [{i+1}/{len(pairs)}]  {fp.name}")
            self._stage("2/7 — Loading DLC files", f"{i+1}/{len(pairs)}: {fp.name}")
            try:
                _gap_fps = float(self._fps_arg or 30.0)
                _max_gap = int(round(_gap_fps * float(
                    self._cfg.get("max_interp_gap_sec", 0.5))))
                xy, bps, fps_hint, ll_fracs, _flat_held = load_dlc_file(
                    fp, self._cfg["likelihood_thresh"],
                    max_interp_gap_frames=_max_gap, log_fn=self._log,
                    return_quality=True)
                for _bp, _frac in ll_fracs.items():
                    all_bp_bad_fracs.setdefault(_bp, []).append(_frac)
                fps = fps_hint or self._fps_arg or 30.0
                xy  = smooth_boxcar(xy, fps, self._cfg["boxcar_win_sec"])
                all_xy.append(xy)
                all_names.append(fp.stem)
                all_fps_list.append(float(fps))
                all_bps.append(bps)
                all_vpaths.append(str(_vp) if _vp else None)
                all_groups.append(_group_key(fp))
                all_flat_held.append(_flat_held)
            except Exception:
                self._log(f"  [WARN] Skipping {fp.name}:\n"
                          f"  {traceback.format_exc()}")
            self._prog(i + 1, len(pairs))

        if not all_xy:
            raise RuntimeError("No files could be loaded.")

        # Resolve a common bodypart set — intersection of all files, preserving
        # the order from the first file.  Needed when groups have different
        # tracked keypoints (e.g. female 22 bp vs male 28 bp).
        bps_sets   = [set(b) for b in all_bps]
        common_set = bps_sets[0].intersection(*bps_sets[1:]) if len(bps_sets) > 1 else bps_sets[0]
        bps_ref    = [bp for bp in all_bps[0] if bp in common_set]  # stable order
        n_dropped  = len(all_bps[0]) - len(bps_ref)
        if n_dropped:
            self._log(f"  [INFO] Bodypart intersection: {len(bps_ref)} common "
                      f"bodyparts across all files ({n_dropped} dropped from "
                      f"reference set that were absent in some files).")
            # Identify which file(s) caused the largest drop so the user knows
            # which group's tracking limited the shared feature space.
            _worst_n = -1
            _worst_nm = None
            for _nm_k, _bps_k in zip(all_names, all_bps):
                _lost = sum(1 for bp in all_bps[0] if bp not in set(_bps_k))
                if _lost > _worst_n:
                    _worst_n, _worst_nm = _lost, _nm_k
            if _worst_nm is not None and _worst_n > 0:
                self._log(f"  [INFO] Smallest-keypoint session: '{_worst_nm}' "
                          f"(missing {_worst_n} of the reference bodyparts).")
        # Quality gate: a tiny shared keypoint set produces an impoverished
        # feature space (few pairwise distances) and unreliable clustering.
        _min_keep = int(self._cfg.get("bodypart_min_keep", 6))
        if len(bps_ref) < _min_keep:
            self._log(f"  [VALID-WARN] Only {len(bps_ref)} bodyparts shared "
                      f"across all sessions (< {_min_keep}). Feature space is "
                      f"impoverished; clustering/DBCV may be unreliable. Improve "
                      f"DLC tracking or relax the conservation policy (median "
                      f"metric, lower min-session-fraction) in DLC & Prep "
                      f"settings.")
            self._stage("VALIDATION WARN",
                        f"only {len(bps_ref)} shared bodyparts — "
                        f"impoverished feature space")
        # Filter every xy array to the common bodypart columns
        for k, (bps_k, xy_k) in enumerate(zip(all_bps, all_xy)):
            if bps_k != bps_ref:
                col_idx = []
                for bp in bps_ref:
                    if bp in bps_k:
                        j = bps_k.index(bp)
                        col_idx.extend([2 * j, 2 * j + 1])
                all_xy[k] = xy_k[:, col_idx]

        # Drop chronically occluded bodyparts from the feature space.
        # At this point all_xy[k] columns are ordered by bps_ref, so we can
        # index directly.  Bodyparts whose mean bad-frame fraction across
        # sessions meets or exceeds feature_bad_bp_thresh are removed; they
        # contribute flat-interpolated artefacts (identical feature vectors)
        # that inflate HDBSCAN noise and cause DBCV to become non-finite.
        _feat_thresh = float(self._cfg.get("feature_bad_bp_thresh", 0.40))
        if _feat_thresh > 0 and all_bp_bad_fracs:
            _agg = {bp: float(np.mean(all_bp_bad_fracs.get(bp, [0.0])))
                    for bp in bps_ref}
            _drop = [bp for bp in bps_ref if _agg.get(bp, 0.0) >= _feat_thresh]
            if _drop:
                self._log(
                    f"  [FEAT-DROP] {len(_drop)} bodypart(s) excluded from "
                    f"feature extraction (mean bad-frame frac >= "
                    f"{_feat_thresh*100:.0f}% across sessions): {_drop}")
                _keep_pos = [i for i, bp in enumerate(bps_ref)
                             if bp not in set(_drop)]
                bps_ref = [bps_ref[i] for i in _keep_pos]
                _col_idx = []
                for i in _keep_pos:
                    _col_idx.extend([2 * i, 2 * i + 1])
                for k in range(len(all_xy)):
                    all_xy[k] = all_xy[k][:, _col_idx]

        fps = float(pd.Series(all_fps_list).mode()[0])
        self._log(f"  FPS = {fps}  |  bodyparts = {len(bps_ref)}")

        # all_flat_held stays as per-bp lists (one bool array per bodypart per
        # session) — the bin-mask block below uses them directly with the
        # fraction threshold so dropped bodyparts cannot inflate exclusion counts.

        if self._cfg["save_plots"]:
            try:
                plot_likelihood_qc(dlc_files,
                                   self._out_plots / "likelihood_qc.png")
            except Exception:
                self._log(f"  [WARN] likelihood_qc plot: "
                          f"{traceback.format_exc()}")

        # 3. Features (V2 — multi-scale, body-normalised, angular)
        _body_norm = bool(self._cfg.get("body_normalise", False))
        # v2.1: skip the angular block when no spine landmarks match by keyword
        # (legacy mode keeps the evenly-spaced fallback).  Used for every feature
        # call in this run so training and inference stay dimensionally aligned.
        _ang_fb = bool(self._cfg.get("angular_fallback", True))
        # If body normalisation is requested but no nose/tail spine landmarks are
        # present, extract_features_v2 silently skips it — warn so the user knows
        # spatial features stay in raw pixels (scale-variant across sessions).
        if _body_norm and _find_spine_indices(bps_ref) == (None, None):
            self._log("  [VALID-WARN] body_normalise is ON but no head/tail spine "
                      "landmarks were found among the shared bodyparts — "
                      "normalisation will be skipped and spatial features remain "
                      "in raw pixels (sensitive to camera distance / body size).")
        _scale_desc = ("50/100/200 ms" if fps >= 60 else "100/200 ms")
        self._log(f"\n[3/7]  Extracting V2 features  "
                  f"({_scale_desc} + angular, body_normalise={_body_norm})...")
        self._stage("3/7 — Extracting V2 features", f"scale={_scale_desc}")
        all_feats = []
        for i, (xy, name) in enumerate(zip(all_xy, all_names)):
            self._stage("3/7 — Extracting V2 features",
                        f"{i+1}/{len(all_xy)}: {name}")
            f = extract_features_v2(xy, fps, bps_ref, body_normalise=_body_norm,
                                    angular_fallback=_ang_fb)
            all_feats.append(f)
            self._log(f"  {name}: {f.shape[0]} features x {f.shape[1]} bins")
            self._prog(i + 1, len(all_xy))

        if self._cfg["save_plots"]:
            try:
                plot_feature_quality(all_feats, all_names,
                                     self._out_plots / "feature_quality.png")
            except Exception:
                pass

        # Validation gate 2: feature consistency across sessions
        try:
            _validation["feature_consistency"] = validate_feature_consistency(
                all_feats, all_names)
            for w in _validation["feature_consistency"]["warnings"]:
                self._log(f"  [VALID-WARN] {w}")
        except Exception as e:
            self._log(f"  [VALID] Feature consistency check failed: {e}")

        feats_cat = np.hstack(all_feats)   # (n_feat, total_bins)
        n_bins    = feats_cat.shape[1]
        rng       = np.random.default_rng(int(self._cfg["umap_random_state"]))

        # Build a per-bin boolean mask for bins where a MAJORITY of kept bodyparts
        # are simultaneously flat-held — indicating whole-animal dropout rather than
        # normal single-camera occlusion of individual limbs.  Only those bins are
        # excluded from UMAP/HDBSCAN training; the MLP still infers labels for them
        # at Step 7.  Natural single-camera occlusion (1–4 of 21 bodyparts, ~15–20%
        # fraction) falls well below the 0.5 default threshold and is left in training.
        _win100 = max(1, int(round(fps / 10)))
        _flat_held_bp_frac = float(self._cfg.get("flat_held_bp_frac_thresh", 0.5))
        _n_bps_ref = max(1, len(bps_ref))
        _flat_bin_masks: list = []
        for _k, _per_bp in enumerate(all_flat_held):
            _bps_k = all_bps[_k]
            _nb_k  = all_feats[_k].shape[1]
            if _nb_k == 0:
                _flat_bin_masks.append(np.array([], dtype=bool))
                continue
            _need = _nb_k * _win100
            _flat_count = np.zeros(_nb_k, dtype=float)
            for _bp in bps_ref:
                try:
                    _bp_idx = _bps_k.index(_bp)
                    if _bp_idx < len(_per_bp):
                        _bp_mask = _per_bp[_bp_idx]
                        _bp_arr  = (_bp_mask[:_need] if len(_bp_mask) >= _need
                                    else np.pad(_bp_mask, (0, _need - len(_bp_mask))))
                        _flat_count += _bp_arr.reshape(_nb_k, _win100).any(axis=1).astype(float)
                except ValueError:
                    pass
            _flat_bin_masks.append((_flat_count / _n_bps_ref) >= _flat_held_bp_frac)
        flat_held_bin_mask = (np.concatenate(_flat_bin_masks)
                              if _flat_bin_masks else np.zeros(n_bins, dtype=bool))
        n_flat = int(flat_held_bin_mask.sum())
        n_good = n_bins - n_flat
        if n_flat > 0 and n_good < 10:
            self._log(f"  [WARN] Only {n_good} good bins after flat-held exclusion "
                      f"— disabling exclusion for this run.")
            flat_held_bin_mask = np.zeros(n_bins, dtype=bool)
            n_flat, n_good = 0, n_bins
        feats_good = feats_cat[:, ~flat_held_bin_mask] if n_flat > 0 else feats_cat
        if n_flat > 0:
            self._log(f"  [OCCLUSION] {n_flat}/{n_bins} bins "
                      f"({100 * n_flat / n_bins:.1f}%) with ≥{100*_flat_held_bp_frac:.0f}% "
                      f"bodyparts simultaneously flat-held — excluded from UMAP/"
                      f"HDBSCAN training; MLP infers labels for these at Step 7.")

        # Record which contiguous slice of the full feature matrix belongs to
        # each session so the UMAP evolution video export can slice the saved
        # umap_embedding.npy to get session-specific 3-D coordinates.
        # Format: { "session_name": [start, end, "/path/to/video_or_null"], ... }
        try:
            _sbr: dict = {}
            _off = 0
            for _name, _feat, _vpath in zip(all_names, all_feats, all_vpaths):
                _nb = _feat.shape[1]
                _sbr[_name] = [_off, _off + _nb, _vpath]
                _off += _nb
            _sbr["_total_bins"] = n_bins
            (self._out_model / "session_bin_ranges.json").write_text(
                json.dumps(_sbr, indent=2))
        except Exception:
            pass

        # Use the full dataset when it is small enough (avoids UMAP over-smoothing
        # caused by a large n_neighbors/N_sample ratio); subsample only for large
        # recordings where UMAP runtime becomes a bottleneck.
        umap_full_thresh = int(self._cfg.get("umap_full_thresh", 10_000))
        if n_good <= umap_full_thresh:
            n_samp    = n_good
            feats_sub = feats_good
        else:
            n_samp    = max(1000, int(n_good * float(self._cfg["train_frac"])))
            idx       = rng.choice(n_good, n_samp, replace=False)
            feats_sub = feats_good[:, idx]

        self._log(f"  Total bins: {n_bins}  -> UMAP sample: {n_samp} "
                  f"({100 * n_samp / n_bins:.0f} %)")

        # Tiny per-sample jitter before standardisation breaks any remaining
        # exact-duplicate feature vectors among the good (non-flat-held) bins.
        _feat_jitter = float(self._cfg.get("feature_dedup_jitter", 1e-4))
        if _feat_jitter > 0:
            feats_sub = feats_sub + rng.normal(0, _feat_jitter, feats_sub.shape)

        from sklearn.preprocessing import StandardScaler
        scaler   = StandardScaler()
        feats_sc = scaler.fit_transform(feats_sub.T).T   # (n_feat, n_samp)
        # Keep the pre-PCA standardised features so UMAP trustworthiness is
        # measured against the actual feature space, not the PCA-reduced one.
        feats_sc_prepca = feats_sc

        # Optional PCA pre-reduction.  Auto-triggers when n_features >= n_samples/5
        # to keep the nearest-neighbour graph reliable for UMAP.
        pca_model = None
        pca_mode  = str(self._cfg.get("pca_pre_reduce", "auto")).lower()
        n_feat_orig, n_samp_umap = feats_sc.shape
        ratio = n_samp_umap / max(1, n_feat_orig)
        _do_pca = (pca_mode == "on") or (pca_mode == "auto" and ratio < 5.0)
        if _do_pca:
            from sklearn.decomposition import PCA
            n_pca = min(n_feat_orig - 1,
                        max(50, int(n_samp_umap ** 0.75)))
            pca_model = PCA(n_components=n_pca,
                            random_state=int(self._cfg["umap_random_state"]))
            feats_sc  = pca_model.fit_transform(feats_sc.T).T
            var_kept  = pca_model.explained_variance_ratio_.sum() * 100
            self._log(f"  PCA pre-reduction: {n_feat_orig} → {n_pca} dims "
                      f"({var_kept:.1f} % variance, sample/feature ratio "
                      f"was {ratio:.1f})")
            self._stage("3/7 — PCA done",
                        f"{n_feat_orig}→{n_pca} dims · {var_kept:.1f}% variance")
            if ratio < 2.0:
                self._log("  [VALID-WARN] sample/feature ratio < 2 even after "
                          "PCA — recording may be too short for reliable analysis")
                self._stage("VALIDATION WARN",
                            "sample/feature ratio < 2 — recording may be too short")

        # 4. UMAP
        # Adaptive n_neighbors: publication default 60 is calibrated to ~1200-bin
        # (2-min) sessions. n_bins is fps-independent (both 30fps and 60fps 2-min
        # videos give ~1200 bins because win100 = fps/10).
        # Formula: clip(avg_session_bins / 20, 15, 90)
        #   ~600 bins/sess  (1 min) → 30   fine-grained local structure
        #   ~1200 bins/sess (2 min) → 60   matches publication default
        #   ~1800 bins/sess (3 min) → 90   smoother manifold for longer recordings
        #   >1800 bins/sess         → 90   capped; higher values give diminishing returns
        # Using per-session average rather than total bins means the value stays
        # calibrated to recording length regardless of how many sessions are pooled.
        # User can override by setting umap_n_neighbors > 0 in cfg.
        _nn_cfg = int(self._cfg.get("umap_n_neighbors", 0))
        if _nn_cfg <= 0:
            _n_sessions  = max(1, len(all_feats))
            _avg_bins    = n_samp // _n_sessions   # use training sample, not total bins
            _nn_adaptive = max(15, min(90, _avg_bins // 20))
            self._cfg["umap_n_neighbors"] = _nn_adaptive
            self._log(f"  [UMAP] n_neighbors auto-set to {_nn_adaptive} "
                      f"(avg_training_bins={_avg_bins}, {_n_sessions} session(s), "
                      f"formula=clip(avg_bins/20, 15, 90); "
                      f"set umap_n_neighbors>0 to override)")
        self._log("\n[4/7]  Running UMAP  "
                  f"(n_components={self._cfg['umap_n_components']}, "
                  f"n_neighbors={self._cfg['umap_n_neighbors']})...")
        self._stage("4/7 — UMAP embedding",
                    f"n_components={self._cfg['umap_n_components']} · "
                    f"n_neighbors={self._cfg['umap_n_neighbors']} · "
                    f"N={n_samp} pts — may take several minutes…")
        umap_model, embedding = run_umap(feats_sc.T, self._cfg)
        self._log(f"  Embedding: {embedding.shape}")
        self._stage("4/7 — UMAP done", f"embedding shape {embedding.shape}")

        # Build the full-size embedding (n_bins rows) for umap_embedding.npy so
        # that session_bin_ranges.json slice indices remain valid.  Flat-held bins
        # are embedded via umap_model.transform() — they were excluded from UMAP
        # training but can still be projected for visualisation.
        try:
            if n_flat > 0:
                _feats_flat_sc = scaler.transform(feats_cat[:, flat_held_bin_mask].T)
                if pca_model is not None:
                    _feats_flat_sc = pca_model.transform(_feats_flat_sc)
                _emb_flat = umap_model.transform(_feats_flat_sc)
                embedding_save = np.empty((n_bins, embedding.shape[1]), dtype=float)
                if n_samp == n_good:
                    embedding_save[~flat_held_bin_mask] = embedding
                else:
                    _feats_good_sc = scaler.transform(feats_good.T)
                    if pca_model is not None:
                        _feats_good_sc = pca_model.transform(_feats_good_sc)
                    embedding_save[~flat_held_bin_mask] = umap_model.transform(
                        _feats_good_sc)
                embedding_save[flat_held_bin_mask] = _emb_flat
            else:
                embedding_save = embedding
        except Exception:
            self._log(f"  [WARN] Could not build full embedding for save: "
                      f"{traceback.format_exc()}")
            embedding_save = embedding

        # Validation gate 3: UMAP trustworthiness
        try:
            _validation["umap_trustworthiness"] = validate_umap_trustworthiness(
                feats_sc_prepca.T, embedding)
            for w in _validation["umap_trustworthiness"]["warnings"]:
                self._log(f"  [VALID-WARN] {w}")
        except Exception as e:
            self._log(f"  [VALID] Trustworthiness check failed: {e}")

        # 5. HDBSCAN
        self._log("\n[5/7]  HDBSCAN clustering  "
                  "(adaptive sweep, DBCV criterion)...")
        self._stage("5/7 — HDBSCAN clustering",
                    f"sweeping min_cluster_size over {n_samp} bins…")
        hdb_clf, hdb_labels, hdb_score, hdb_score_label = run_hdbscan(
            embedding, self._cfg, n_total=n_samp, log_fn=self._log)
        n_cl      = len(set(hdb_labels[hdb_labels >= 0]))
        noise     = (hdb_labels < 0).sum()
        noise_pct = 100 * noise / max(1, len(hdb_labels))
        self._log(f"  {n_cl} clusters, {noise} noise points "
                  f"({noise_pct:.1f} %), "
                  f"{hdb_score_label}={hdb_score:.3f}")
        self._stage("5/7 — HDBSCAN done",
                    f"{n_cl} clusters · {noise_pct:.0f}% noise · {hdb_score_label}={hdb_score:.3f}")

        # ── Rare-cluster pruning ──────────────────────────────────────────────
        # Drop any cluster whose fraction of total bins is below min_cluster_freq.
        # Such clusters represent behaviours so infrequent relative to the whole
        # recording session that they are likely noise fragments, not true states.
        # Pruned labels are reassigned to noise (-1) and the remaining cluster IDs
        # are renumbered contiguously before MLP training.
        # min_cluster_freq is stored as a percentage (e.g. 0.5 means 0.5 %)
        _min_freq_pct = float(self._cfg.get("min_cluster_freq", 0.5))
        _min_freq     = _min_freq_pct / 100.0
        # _hdb_remap: maps original HDBSCAN cluster IDs → renumbered IDs used in
        # hdb_labels (and therefore in the trained MLP).  Needed to keep the
        # fallback approximate_predict path in sync when pruning has occurred.
        _hdb_remap: dict = {}   # {orig_id: new_id}
        if _min_freq > 0 and n_cl >= 2:
            _pruned_ids = []
            _unique_ids = sorted(set(hdb_labels[hdb_labels >= 0]))
            for _cid in _unique_ids:
                _frac = (hdb_labels == _cid).sum() / max(1, n_samp)
                if _frac < _min_freq:
                    _pruned_ids.append(_cid)
            if _pruned_ids:
                for _cid in _pruned_ids:
                    hdb_labels[hdb_labels == _cid] = -1
                # Renumber remaining clusters 0, 1, 2, …
                _remaining = sorted(set(hdb_labels[hdb_labels >= 0]))
                _remap = {old: new for new, old in enumerate(_remaining)}
                _new_labels = hdb_labels.copy()
                for old, new in _remap.items():
                    _new_labels[hdb_labels == old] = new
                hdb_labels = _new_labels
                n_cl = len(_remaining)
                noise = (hdb_labels < 0).sum()
                noise_pct = 100 * noise / max(1, len(hdb_labels))
                self._log(
                    f"  [rare-cluster prune] Removed {len(_pruned_ids)} cluster(s) "
                    f"below {_min_freq_pct:.2f}% of total bins "
                    f"({', '.join(f'#{i}' for i in _pruned_ids)}) → "
                    f"{n_cl} clusters remain, {noise_pct:.1f}% noise"
                )
                self._stage("5/7 — HDBSCAN done",
                            f"{n_cl} clusters (after rare-cluster prune) · "
                            f"{noise_pct:.0f}% noise")
                _hdb_remap = dict(_remap)

        # Validation gate 4: clustering quality (silhouette)
        try:
            _validation["clustering"] = validate_clustering(embedding,
                                                            hdb_labels)
            for w in _validation["clustering"]["warnings"]:
                lvl = "[VALID-BLOCK]" if _validation["clustering"]["blocked"] \
                      else "[VALID-WARN]"
                self._log(f"  {lvl} {w}")
            if _validation["clustering"].get("blocked"):
                self._stage("VALIDATION BLOCK",
                            f"clustering quality: {_validation['clustering']['warnings'][0]}"
                            if _validation["clustering"]["warnings"] else "silhouette < 0")
            elif _validation["clustering"]["warnings"]:
                self._stage("VALIDATION WARN",
                            f"clustering: {_validation['clustering']['warnings'][0]}")
        except Exception as e:
            self._log(f"  [VALID] Clustering validation failed: {e}")

        # ── Cluster-stability seed sweep (optional) ───────────────────────────
        # Re-run UMAP+HDBSCAN over several seeds to measure how reproducible the
        # PARTITION is (the internal gates only measure cluster tightness).
        _n_sweep = int(self._cfg.get("seed_sweep_n", 0) or 0)
        if _n_sweep >= 2:
            try:
                self._log(f"\n[STABILITY]  Seed sweep ({_n_sweep} seeds) — "
                          f"assessing cluster-count / partition stability...")
                self._stage("Cluster-stability seed sweep",
                            f"{_n_sweep} seeds — re-running UMAP+HDBSCAN")
                _sweep = seed_sweep_stability(
                    feats_sc.T, self._cfg, _n_sweep, log_fn=self._log)
                if _sweep:
                    _validation["cluster_stability"] = {
                        "stage": "cluster_stability", "status":
                            "pass" if _sweep["mean_ari"] >= 0.7 else "warn",
                        "mean_ari": round(_sweep["mean_ari"], 4),
                        "cluster_counts": _sweep["counts"],
                        "warnings": ([] if _sweep["mean_ari"] >= 0.7 else
                                     [f"Mean ARI {_sweep['mean_ari']:.3f} < 0.7: "
                                      "cluster partition is seed-sensitive."]),
                    }
                    self._log(f"  Mean pairwise ARI = {_sweep['mean_ari']:.3f} "
                              f"(cluster counts {min(_sweep['counts'])}–"
                              f"{max(_sweep['counts'])})")
                    if self._cfg["save_plots"]:
                        plot_cluster_stability(
                            _sweep, self._out_plots / "cluster_stability.png")
            except Exception:
                self._log(f"  [WARN] seed sweep: {traceback.format_exc()}")

        # 2D and 3D UMAP plots are generated after inference (below) so that
        # they can be filtered to show only clusters actually predicted by the
        # MLP on these sessions — keeping the cluster count consistent across
        # umap_embedding.png, umap_3d.*, and dwell_time_distributions.png.

        # Save UMAP embedding + cluster labels as numpy arrays so cube_analyser
        # can display before/after UMAP views when the user recombines clusters.
        # embedding_save is full-size (n_bins rows); flat-held bins projected via
        # transform() so session_bin_ranges.json slice indices remain valid.
        # hdb_labels_all expands hdb_labels to n_bins with -1 for flat-held bins.
        if n_flat > 0:
            hdb_labels_all = np.full(n_bins, -1, dtype=int)
            if n_samp == n_good:
                # All good bins were used for UMAP/HDBSCAN training.
                hdb_labels_all[~flat_held_bin_mask] = hdb_labels
            else:
                # Subsampling occurred: map each label back to its original bin position
                # so umap_labels.npy stays the same length as umap_embedding.npy (n_bins).
                _good_positions = np.where(~flat_held_bin_mask)[0]
                hdb_labels_all[_good_positions[idx]] = hdb_labels
        else:
            hdb_labels_all = hdb_labels
        try:
            np.save(str(self._out_model / "umap_embedding.npy"), embedding_save)
            np.save(str(self._out_model / "umap_labels.npy"),    hdb_labels_all)
        except Exception:
            pass

        # 6. MLP
        self._log("\n[6/7]  Training MLP classifier  "
                  f"(hidden={self._cfg['mlp_hidden']})...")
        self._stage("6/7 — Training MLP",
                    f"hidden={self._cfg['mlp_hidden']} · {n_cl} classes")
        mlp_clf, cv_scores = train_mlp(feats_sc, hdb_labels, self._cfg)
        if mlp_clf is not None:
            self._log(f"  CV accuracy: {cv_scores.mean():.3f} "
                      f"+/- {cv_scores.std():.3f}")
            self._log("  [NOTE] CV accuracy measures how separable the HDBSCAN "
                      "clusters are in feature space (classifier self-consistency), "
                      "NOT behavioral validity. It is computed on non-noise bins "
                      "only; it does not validate the noise fraction or the "
                      "biological meaning of clusters.")
            self._stage("6/7 — MLP done",
                        f"CV accuracy {cv_scores.mean():.3f} "
                        f"± {cv_scores.std():.3f}")
            # Validation gate 5: classifier accuracy
            try:
                _validation["mlp_accuracy"] = validate_mlp_accuracy(cv_scores)
                for w in _validation["mlp_accuracy"]["warnings"]:
                    lvl = "[VALID-BLOCK]" if _validation["mlp_accuracy"][
                        "blocked"] else "[VALID-WARN]"
                    self._log(f"  {lvl} {w}")
                if _validation["mlp_accuracy"].get("blocked"):
                    self._stage("VALIDATION BLOCK",
                                f"MLP accuracy {cv_scores.mean():.3f} — at-chance performance")
                elif _validation["mlp_accuracy"]["warnings"]:
                    self._stage("VALIDATION WARN",
                                f"MLP accuracy {cv_scores.mean():.3f} — marginal classifier")
            except Exception as e:
                self._log(f"  [VALID] MLP accuracy check failed: {e}")
            if self._cfg["save_plots"]:
                try:
                    plot_confusion(mlp_clf, feats_sc, hdb_labels,
                                   self._out_plots / "confusion_matrix.png")
                    plot_cv_scores(cv_scores,
                                   self._out_plots / "cv_accuracy.png")
                except Exception:
                    pass
        else:
            self._log("  [WARN] < 2 clusters — MLP not trained.")
            self._stage("6/7 — MLP skipped", "< 2 clusters found")

        # Save model (includes bodyparts for V2 prediction consistency)
        model_path = self._out_model / "bsoid_model.pkl"
        with open(str(model_path), "wb") as fh:
            pickle.dump(dict(
                umap_model  = umap_model,
                hdb_clf     = hdb_clf,
                mlp_clf     = mlp_clf,
                scaler      = scaler,
                pca_model   = pca_model,
                cv_scores   = cv_scores.tolist(),
                fps         = fps,
                cfg         = self._cfg,
                bodyparts   = bps_ref,
                n_clusters  = int(n_cl),
                feature_ver = "v2",
                analysis_version = ANALYSIS_VERSION,
                compat_mode = self._cfg.get("compat_mode", "current"),
                created     = datetime.now().isoformat(),
            ), fh)
        self._log(f"  Model saved -> {model_path}")

        (self._out_model / "feature_config.json").write_text(
            json.dumps(dict(fps=fps, bodyparts=bps_ref,
                            boxcar_win_sec=self._cfg["boxcar_win_sec"],
                            n_features=int(feats_cat.shape[0]),
                            feature_version="v2",
                            analysis_version=ANALYSIS_VERSION,
                            compat_mode=self._cfg.get("compat_mode", "current"),
                            pca_n_components=(int(pca_model.n_components_)
                                              if pca_model is not None else None)),
                       indent=2))

        # 7. Predict & export
        self._log("\n[7/7]  Predicting & exporting...")
        self._stage("7/7 — Predicting & exporting", f"0/{len(pairs)} sessions")
        bout_paths, frame_paths, all_epochs, all_frame_labels = [], [], [], []
        # Collect example-clip tasks to write after the main loop so they can
        # be shuffled for an even cross-animal mix per cluster.
        _clip_tasks: list = []   # (vp, epochs_df, file_fps, animal_id)

        for i, ((fp, vp), (xy, name, file_fps)) in enumerate(
                zip(pairs, zip(all_xy, all_names, all_fps_list))):

            self._log(f"  [{i+1}/{len(pairs)}]  {name}")
            self._stage("7/7 — Predicting & exporting",
                        f"{i+1}/{len(pairs)}: {name}")

            if mlp_clf is not None:
                frame_labels = predict_labels(
                    xy, umap_model, mlp_clf, scaler, file_fps,
                    bodyparts=bps_ref, body_normalise=_body_norm,
                    pca_model=pca_model,
                    min_confidence=float(self._cfg.get("mlp_confidence_thresh", 0.0)),
                    angular_fallback=_ang_fb)
            else:
                # Fallback: no MLP → use HDBSCAN approximate_predict on V2 feats
                try:
                    import hdbscan as _hdb
                    f    = extract_features_v2(xy, file_fps, bps_ref,
                                              body_normalise=_body_norm,
                                              angular_fallback=_ang_fb)
                    sc   = scaler.transform(f.T)
                    if pca_model is not None:
                        sc = pca_model.transform(sc)
                    emb  = umap_model.transform(sc)
                    soft, _ = _hdb.approximate_predict(hdb_clf, emb)
                    # approximate_predict returns the original HDBSCAN cluster IDs
                    # (before rare-cluster pruning and renumbering).  Apply the
                    # same remap that was applied to hdb_labels so that the
                    # fallback cluster IDs match those used in the plots.
                    if _hdb_remap:
                        _remapped = np.full_like(soft, -1, dtype=int)
                        for _orig, _new in _hdb_remap.items():
                            _remapped[soft == _orig] = _new
                        soft = _remapped
                    win  = max(1, int(round(file_fps / 10)))
                    fl   = np.repeat(soft.astype(int), win)
                    n_f  = xy.shape[0]
                    if len(fl) < n_f:
                        fl = np.pad(fl, (0, n_f - len(fl)), mode="edge")
                    frame_labels = fl[:n_f]
                except Exception:
                    frame_labels = np.zeros(xy.shape[0], dtype=int)

            all_frame_labels.append(frame_labels)

            # Bout CSV (exact B-SOiD GUI format)
            bout_df = labels_to_bouts(frame_labels)
            bout_p  = self._out_bouts / f"{name}_bout_lengths.csv"
            bout_df.to_csv(str(bout_p), index=False)
            bout_paths.append(bout_p)

            # Frame-label CSV
            frame_df = pd.DataFrame({
                "frame":  np.arange(len(frame_labels)),
                "time_s": np.arange(len(frame_labels)) / file_fps,
                "label":  frame_labels,
            })
            frame_p = self._out_bouts / f"{name}_frame_labels.csv"
            frame_df.to_csv(str(frame_p), index=False)
            frame_paths.append(frame_p)

            # Epoch CSV (filtered by user-set min/max bout duration)
            epochs = bouts_to_epochs(
                bout_df, file_fps,
                min_dur=float(self._cfg["min_epoch_dur_s"]),
                max_dur=float(self._cfg["max_epoch_dur_s"]))
            epochs.to_csv(
                str(self._out_bouts / f"{name}_epochs.csv"), index=False)
            epoch_stats(epochs).to_csv(
                str(self._out_bouts / f"{name}_epoch_stats.csv"), index=False)
            all_epochs.append((epochs, name))

            if self._cfg["save_plots"] and not epochs.empty:
                try:
                    plot_ethogram(frame_labels, file_fps,
                                  self._out_plots / f"ethogram_{name}.png",
                                  name)
                    plot_cluster_durations(
                        epochs,
                        self._out_plots / f"cluster_durations_{name}.png",
                        name)
                    plot_cluster_stats(
                        epochs,
                        self._out_plots / f"cluster_stats_{name}.png")
                except Exception:
                    pass

            if vp and self._cfg["save_videos"]:
                if self._cfg["save_example_clips"] and not epochs.empty:
                    # Defer clip writing; process all animals together after the
                    # loop so clips can be shuffled for a cross-animal mix.
                    _clip_tasks.append(
                        (vp, epochs.copy(), file_fps, Path(vp).stem))
                if self._cfg["save_labeled_video"]:
                    try:
                        create_labeled_video(
                            vp, frame_labels, self._out_videos, file_fps,
                            output_fps=int(self._cfg["output_fps"]))
                    except Exception:
                        self._log(f"  [WARN] Labeled video: "
                                  f"{traceback.format_exc()}")

            self._prog(i + 1, len(pairs))

        # ── HMM smoothing pass (post-hoc Multinomial HMM wrapper) ────────────
        all_hmm_labels: list = []
        hmm_model = None
        if self._cfg.get("hmm_enabled", True) and all_frame_labels:
            try:
                _t0 = time.perf_counter()
                self._log("\n[HMM]  Training Multinomial HMM on MLP label sequences...")
                _hmm_n_states = self._cfg.get("hmm_n_states") or None
                if _hmm_n_states is not None:
                    _hmm_n_states = int(_hmm_n_states)
                hmm_model = train_hmm(
                    all_frame_labels,
                    n_clusters=int(n_cl),
                    n_states=_hmm_n_states,
                    n_iter=int(self._cfg.get("hmm_n_iter", 100)),
                    log_fn=self._log,
                )
                self._log(f"  HMM trained in {time.perf_counter() - _t0:.2f} s  "
                          f"({hmm_model.n_components} states, Baum-Welch)")
                for _raw, _name, _file_fps in zip(
                        all_frame_labels, all_names, all_fps_list):
                    _hmm_labels = decode_hmm(hmm_model, _raw)
                    all_hmm_labels.append(_hmm_labels)
                    _bout_hmm = labels_to_bouts(_hmm_labels)
                    _bout_hmm.to_csv(
                        str(self._out_bouts / f"{_name}_bout_lengths_hmm.csv"),
                        index=False)
                    pd.DataFrame({
                        "frame":  np.arange(len(_hmm_labels)),
                        "time_s": np.arange(len(_hmm_labels)) / _file_fps,
                        "label":  _hmm_labels,
                    }).to_csv(
                        str(self._out_bouts / f"{_name}_frame_labels_hmm.csv"),
                        index=False)
                    _ep_hmm = bouts_to_epochs(
                        _bout_hmm, _file_fps,
                        min_dur=float(self._cfg["min_epoch_dur_s"]),
                        max_dur=float(self._cfg["max_epoch_dur_s"]))
                    _ep_hmm.to_csv(
                        str(self._out_bouts / f"{_name}_epochs_hmm.csv"),
                        index=False)
                    epoch_stats(_ep_hmm).to_csv(
                        str(self._out_bouts / f"{_name}_epoch_stats_hmm.csv"),
                        index=False)
                _hmm_path = self._out_model / "hmm_model.pkl"
                with open(str(_hmm_path), "wb") as _fh:
                    pickle.dump(hmm_model, _fh)
                self._log(f"  HMM model saved -> {_hmm_path}")
            except Exception:
                self._log(f"  [WARN] HMM smoothing failed:\n"
                          f"{traceback.format_exc()}")

        # Example clips — written in shuffled animal order so each cluster's
        # quota is filled from a random mix of animals rather than exhausted
        # by whichever animal happens to appear first.
        if _clip_tasks:
            import random as _random
            _random.shuffle(_clip_tasks)
            _clips_per_cluster: dict = {}
            _max_clips = int(self._cfg["max_clips_per_cluster"])
            # Limit each animal to ceil(max_clips / n_animals) clips per cluster
            # so the quota is spread across animals rather than filled by whichever
            # animal happens to appear first in the shuffled order.
            _max_per_call = -(_max_clips // -len(_clip_tasks))  # ceiling division
            for _vp, _ep, _fps, _aid in _clip_tasks:
                try:
                    create_example_clips(
                        _vp, _ep, self._out_videos, _fps,
                        output_fps=int(self._cfg["output_fps"]),
                        max_clips=_max_clips,
                        animal_id=_aid,
                        clips_per_cluster=_clips_per_cluster,
                        max_per_call=_max_per_call)
                except Exception:
                    self._log(f"  [WARN] Clips: {traceback.format_exc()}")

        # Combined epochs across all sessions
        if all_epochs:
            combined = pd.concat(
                [ep.assign(session=n) for ep, n in all_epochs if not ep.empty],
                ignore_index=True)
            combined.to_csv(
                str(self.output_dir / "all_epochs_combined.csv"), index=False)

        # Per-cluster kinematic signatures (interpretable descriptors for naming).
        if all_frame_labels and all_xy:
            try:
                compute_cluster_kinematics(
                    all_xy, all_frame_labels, all_fps_list, bps_ref,
                    self.output_dir / "cluster_kinematics.csv")
                self._log("  [PLOT] cluster_kinematics.csv saved")
            except Exception:
                self._log(f"  [WARN] cluster_kinematics: "
                          f"{traceback.format_exc()}")

        # ── Consistent cluster set for all downstream plots ───────────────────
        # clusters_seen: cluster IDs actually predicted by the MLP across all
        # sessions that also have at least one epoch surviving the duration
        # filter.  Using this shared set for the 2-D UMAP, 3-D UMAP, and
        # dwell-time violin ensures all three plots show the same number of
        # clusters.
        clusters_seen = sorted(set(
            int(l)
            for ep, _ in all_epochs if not ep.empty
            for l in ep.label.unique()))
        # hdb_labels filtered to clusters_seen — bins whose cluster was not
        # observed in inference (or was entirely filtered by bout duration) are
        # treated as noise so the UMAP reflects the same active cluster set as
        # the dwell-time plot.  Fall back to all valid clusters when
        # clusters_seen is empty (e.g. duration filter rejects every bout).
        if clusters_seen:
            _active_set = set(clusters_seen)
            _hdb_labels_active = hdb_labels.copy()
            _hdb_labels_active[~np.isin(_hdb_labels_active,
                                        sorted(_active_set))] = -1
        else:
            _hdb_labels_active = hdb_labels.copy()

        if self._cfg["save_plots"] and all_frame_labels:
            try:
                plot_transition_matrix(
                    all_frame_labels,
                    self._out_plots / "transition_matrix.png")
            except Exception:
                self._log(f"  [WARN] transition_matrix: "
                          f"{traceback.format_exc()}")
            try:
                plot_umap(embedding, _hdb_labels_active,
                          self._out_plots / "umap_embedding.png")
            except Exception:
                self._log(f"  [WARN] umap_embedding: "
                          f"{traceback.format_exc()}")
            try:
                _tmat, _cids = _tmat_from_labels(all_frame_labels)
                plot_umap_3d_transitions(
                    embedding, _hdb_labels_active,
                    tmat=_tmat, cluster_ids=_cids,
                    out_path=self._out_plots / "umap_3d.html",
                    tag="clustering",
                )
            except Exception:
                self._log(f"  [WARN] umap_3d: {traceback.format_exc()}")

        # ── HMM diagnostic plots ──────────────────────────────────────────────
        if self._cfg["save_plots"] and all_hmm_labels:
            try:
                plot_duration_comparison(
                    np.concatenate(all_frame_labels),
                    np.concatenate(all_hmm_labels),
                    fps,
                    self._out_plots / "hmm_duration_comparison.png")
            except Exception:
                self._log(f"  [WARN] hmm_duration_comparison: "
                          f"{traceback.format_exc()}")
            try:
                plot_hmm_transition_matrix(
                    hmm_model,
                    self._out_plots / "hmm_transition_matrix.png")
            except Exception:
                self._log(f"  [WARN] hmm_transition_matrix: "
                          f"{traceback.format_exc()}")
            try:
                for _raw, _hmm_l, _name in zip(
                        all_frame_labels, all_hmm_labels, all_names):
                    plot_dual_ethogram(
                        _raw, _hmm_l, fps,
                        self._out_plots / f"hmm_ethogram_{_name}.png",
                        _name)
            except Exception:
                self._log(f"  [WARN] hmm_ethogram: "
                          f"{traceback.format_exc()}")
            try:
                plot_syntax_network(
                    hmm_model,
                    self._out_plots / "hmm_syntax_network.png",
                    min_prob=float(self._cfg.get("hmm_min_prob", 0.05)))
            except Exception:
                self._log(f"  [WARN] hmm_syntax_network: "
                          f"{traceback.format_exc()}")

        # ── Section 2: post-analysis publication plots ────────────────────────
        if self._cfg["save_plots"] and all_frame_labels:
            _export_labels = (
                list(all_hmm_labels) if all_hmm_labels else list(all_frame_labels)
            )

            # Dwell-time violin plots
            if all_epochs:
                try:
                    _comb_ep = pd.concat(
                        [ep for ep, _ in all_epochs if not ep.empty],
                        ignore_index=True)
                    if not _comb_ep.empty:
                        plot_dwell_violin(
                            _comb_ep,
                            self._out_plots / "dwell_time_distributions.png")
                        self._log("  [PLOT] dwell_time_distributions.png saved")
                except Exception:
                    self._log(f"  [WARN] dwell_violin: {traceback.format_exc()}")

            # Sankey behavioral-sequence flow diagram
            try:
                plot_sankey_sequences(
                    _export_labels,
                    self._out_plots / "sankey_sequences.png")
                self._log("  [PLOT] sankey_sequences.png saved")
            except Exception:
                self._log(f"  [WARN] sankey_sequences: {traceback.format_exc()}")

            # Continuous state-space projection (UMAP embedding + trajectory)
            try:
                _tag0 = all_names[0] if all_names else ""
                plot_state_space_trajectory(
                    embedding,
                    _hdb_labels_active,
                    fps,
                    self._out_plots / "state_space_projection.png",
                    _tag0)
                self._log("  [PLOT] state_space_projection.png saved")
            except Exception:
                self._log(f"  [WARN] state_space_projection: "
                          f"{traceback.format_exc()}")

        # ── Auto-generate UMAP evolution videos ──────────────────────────────
        _n_ev = int(self._cfg.get("umap_evolution_n", 1))
        if _n_ev > 0:
            try:
                import random as _rnd_ev
                import pandas as _pd_ev
                # Group candidates by their input folder so EVERY uploaded folder
                # gets at least one evolution video (umap_evolution_n is applied
                # per folder, floored at 1).  Without this, a single random draw
                # could leave some folders with no video.
                _cand_by_group: dict = {}
                for _ei, (_nm, _vp) in enumerate(zip(all_names, all_vpaths)):
                    if not (_vp and Path(_vp).is_file()):
                        continue
                    if _ei >= len(frame_paths):
                        continue
                    _sbr_e = _sbr.get(_nm)
                    if not _sbr_e:
                        continue
                    _sb, _se = int(_sbr_e[0]), int(_sbr_e[1])
                    if _se > len(embedding):
                        continue
                    _grp = (all_groups[_ei] if _ei < len(all_groups) else "all")
                    _cand_by_group.setdefault(_grp, []).append(
                        (_nm, _ei, _sb, _se, Path(_vp)))
                if _cand_by_group:
                    _per_group = max(1, _n_ev)
                    _ev_chosen = []
                    for _grp, _cands in _cand_by_group.items():
                        _ev_chosen.extend(
                            _rnd_ev.sample(_cands, min(_per_group, len(_cands))))
                    self._log(
                        f"  [UMAP-EV] Exporting {len(_ev_chosen)} evolution "
                        f"video(s) across {len(_cand_by_group)} folder(s) "
                        f"({_per_group} per folder)...")
                    _ev_out = self.output_dir / "videos" / "umap_evolution"
                    _ev_out.mkdir(parents=True, exist_ok=True)
                    for _ev_nm, _ev_idx, _ev_sb, _ev_se, _ev_vp in _ev_chosen:
                        try:
                            # Prefer HMM-smoothed frame labels when available
                            _hmm_fl_p = (self._out_bouts /
                                         f"{_ev_nm}_frame_labels_hmm.csv")
                            _raw_fl_p = frame_paths[_ev_idx]
                            _fl_p = _hmm_fl_p if _hmm_fl_p.is_file() else _raw_fl_p
                            # These CSVs have a header (frame,time_s,label); read
                            # the 'label' column, not iloc[:,0] (the frame index,
                            # whose header row 'frame' breaks int conversion).
                            _fl_df  = _pd_ev.read_csv(str(_fl_p))
                            _fl_col = ("label" if "label" in _fl_df.columns
                                       else _fl_df.columns[-1])
                            _fl = (_pd_ev.to_numeric(_fl_df[_fl_col],
                                                     errors="coerce")
                                   .dropna().to_numpy(dtype=int))
                            self._log(
                                f"  [UMAP-EV] Rendering '{_ev_nm}' "
                                f"(side-by-side video — this can take 1-2 min)...")
                            _ev_result = create_umap_evolution_video(
                                video_path=_ev_vp,
                                embedding=embedding[_ev_sb:_ev_se],
                                umap_labels=hdb_labels[_ev_sb:_ev_se],
                                frame_labels=_fl,
                                source_fps=fps,
                                out_path=_ev_out / f"{_ev_nm}_umap_evolution.mp4",
                                output_fps=float(self._cfg.get("output_fps", 15)),
                            )
                            if _ev_result:
                                self._log(
                                    f"  [UMAP-EV] Saved -> {_ev_result}")
                            else:
                                self._log(
                                    f"  [WARN] UMAP evolution video failed: "
                                    f"'{_ev_nm}'")
                        except Exception:
                            self._log(
                                f"  [WARN] UMAP evolution ({_ev_nm}): "
                                f"{traceback.format_exc()}")
                else:
                    self._log(
                        "  [UMAP-EV] Skipped: no sessions have associated "
                        "video files.")
            except Exception:
                self._log(
                    f"  [WARN] UMAP evolution video block: "
                    f"{traceback.format_exc()}")

        # Auto-groups (clusters_seen already computed above)
        groups = {f"C{c}": {"labels": [c], "color": _cmap(c)}
                  for c in clusters_seen}

        # ── Validation report ─────────────────────────────────────────────────
        all_warnings = [w for r in _validation.values()
                        for w in r.get("warnings", [])]
        any_block    = any(r.get("blocked", False)
                          for r in _validation.values())
        val_report   = dict(
            cube_version   = VERSION,
            analysis_version = ANALYSIS_VERSION,
            compat_mode    = self._cfg.get("compat_mode", "current"),
            created        = datetime.now().isoformat(),
            overall_status = "block" if any_block else
                             ("warn" if all_warnings else "pass"),
            stages         = _validation,
            all_warnings   = all_warnings,
        )
        (self.output_dir / "validation_report.json").write_text(
            json.dumps(val_report, indent=2))
        if self._cfg["save_plots"]:
            try:
                plot_validation_summary(
                    _validation,
                    self._out_plots / "validation_dashboard.png")
            except Exception:
                self._log(f"  [WARN] validation_dashboard: "
                          f"{traceback.format_exc()}")
        if any_block:
            self._log("\n  [!] VALIDATION BLOCKS DETECTED — see "
                      "validation_report.json")
        elif all_warnings:
            self._log(f"\n  [!] {len(all_warnings)} validation warning(s) — "
                      "see validation_report.json")
        else:
            self._log("\n  [✓] All validation gates passed.")

        # ── Summary JSON ──────────────────────────────────────────────────────
        summary = dict(
            cube_version  = VERSION,
            pipeline      = "CUBE: Comprehensive Unsupervised Behavioral Explorer",
            created       = datetime.now().isoformat(),
            fps           = float(fps),
            n_sessions    = len(all_names),
            sessions      = all_names,
            n_clusters    = int(n_cl),
            clusters      = clusters_seen,
            cv_accuracy   = float(cv_scores.mean()) if mlp_clf else None,
            cfg           = self._cfg,
            output_dir    = str(self.output_dir),
            bout_lengths  = [str(p) for p in bout_paths],
            frame_labels  = [str(p) for p in frame_paths],
            model         = str(model_path),
            feature_version = "v2",
            validation    = val_report["overall_status"],
        )
        (self.output_dir / "bsoid_run_summary.json").write_text(
            json.dumps(summary, indent=2))

        # Delete labeled_videos/ folder — these full-session labeled videos are
        # large and not needed after example clips have been created.
        if bool(self._cfg.get("delete_labeled_videos", True)):
            _lv_dir = self._out_videos / "labeled_videos"
            if _lv_dir.exists():
                try:
                    shutil.rmtree(str(_lv_dir))
                    self._log(f"  [cleanup] Deleted labeled_videos/ folder")
                except Exception as _lv_e:
                    self._log(f"  [cleanup] Could not delete labeled_videos/: {_lv_e}")

        self._log("\n" + "=" * 64)
        self._log(f"  Done!  {n_cl} clusters  |  {len(all_names)} session(s)")
        self._log(f"  Output -> {self.output_dir}")
        self._log("=" * 64 + "\n")
        self._stage("Done",
                    f"{n_cl} clusters · {len(all_names)} session(s) · "
                    f"CV={cv_scores.mean():.3f} · "
                    f"validation={val_report.get('overall_status','?')}")

        return dict(
            bout_lengths_paths = bout_paths,
            frame_label_paths  = frame_paths,
            groups             = groups,
            model_path         = model_path,
            output_dir         = self.output_dir,
            n_clusters         = int(n_cl),
            summary            = summary,
            validation         = val_report,
        )

    #   re-use saved model  

    @classmethod
    def predict_from_saved_model(cls,
                                  model_path,
                                  csv_folder,
                                  video_folder=None,
                                  output_dir="bsoid_predict",
                                  logger=None) -> dict:
        """Load a pkl and predict on new DLC files without retraining."""
        log = logger or print
        log(f"Loading model: {model_path}")
        with open(str(model_path), "rb") as fh:
            m = pickle.load(fh)
        engine = cls(csv_folder=csv_folder, video_folder=video_folder,
                     output_dir=output_dir, fps=m["fps"],
                     logger=logger, cfg=m["cfg"])
        dlc_files = find_dlc_files(csv_folder)
        vid_dict  = find_videos(video_folder) if video_folder else {}
        pairs     = pair_files(dlc_files, vid_dict)
        umap_m, mlp_m, scaler = m["umap_model"], m["mlp_clf"], m["scaler"]
        fps = float(m["fps"])
        bout_paths, frame_paths = [], []
        for fp, vp in pairs:
            try:
                xy, _, _ = load_dlc_file(fp, m["cfg"]["likelihood_thresh"])
                xy = smooth_boxcar(xy, fps, m["cfg"]["boxcar_win_sec"])
            except Exception:
                log(f"  [WARN] Skip {fp.name}: {traceback.format_exc()}")
                continue
            # Pass the saved feature-construction settings so inference features
            # match those the model was trained on.  Older pkls may lack some
            # keys; .get() defaults reproduce their original behaviour.
            _mcfg = m.get("cfg", {}) or {}
            fl    = predict_labels(
                xy, umap_m, mlp_m, scaler, fps,
                bodyparts=m.get("bodyparts"),
                body_normalise=bool(_mcfg.get("body_normalise", False)),
                pca_model=m.get("pca_model"),
                min_confidence=float(_mcfg.get("mlp_confidence_thresh", 0.0)),
                angular_fallback=bool(_mcfg.get("angular_fallback", True)))
            bd    = labels_to_bouts(fl)
            bp    = engine._out_bouts / f"{fp.stem}_bout_lengths.csv"
            bd.to_csv(str(bp), index=False)
            bout_paths.append(bp)
            fd    = pd.DataFrame({"frame": np.arange(len(fl)),
                                  "time_s": np.arange(len(fl)) / fps,
                                  "label": fl})
            fp2   = engine._out_bouts / f"{fp.stem}_frame_labels.csv"
            fd.to_csv(str(fp2), index=False)
            frame_paths.append(fp2)
        return dict(bout_lengths_paths=bout_paths,
                    frame_label_paths=frame_paths,
                    output_dir=engine.output_dir)
