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


def load_dlc_file(path, likelihood_thresh: float = 0.3):
    """
    Load a DLC CSV or H5 file.

    Returns
    -------
    xy          : np.ndarray  (N_frames, n_bodyparts * 2)
    bodyparts   : list[str]
    fps_hint    : float | None   - extracted from filename if present
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
    for col in range(xy.shape[1]):
        bp_i = col // 2
        lk   = ll[:, bp_i]
        bad  = (lk < likelihood_thresh) | np.isnan(xy[:, col])
        good = ~bad
        if good.sum() > 1:
            xy[bad, col] = np.interp(frames[bad], frames[good], xy[good, col])
        elif good.sum() == 1:
            xy[bad, col] = xy[good, col][0]
        else:
            xy[:, col] = 0.0

    # Guess FPS from filename  (e.g. "60Hz" or "30fps")
    fps_hint = None
    m = re.search(r"(\d+)\s*(?:[Hh][Zz]|[Ff][Pp][Ss])", Path(path).stem)
    if m:
        fps_hint = float(m.group(1))

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
                       bodyparts: list) -> "np.ndarray | None":
    """
    Angles (radians) at the vertex B for consecutive body-axis triples A-B-C.
    Spine bodyparts detected by keyword; falls back to evenly-spaced indices.
    Returns (n_bins, n_angles) or None when fewer than 3 spine parts are found.
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
                         body_normalise: bool = True) -> np.ndarray:
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
    b100  = xy[:n_bins * win100].reshape(n_bins, win100, n_xy).mean(axis=1)
    xs100 = b100[:, 0::2]
    ys100 = b100[:, 1::2]

    if head_idx is not None and tail_idx is not None:
        spine = _spine_norm_factor(xs100, ys100, head_idx, tail_idx)
        def _norm(v): return v / spine[:, None]
    else:
        def _norm(v): return v

    xs100n = _norm(xs100)
    ys100n = _norm(ys100)

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

    # ── Angular features (body-axis curvature) ────────────────────────────────
    ang = _angular_features(xs100n, ys100n, bodyparts)

    # ── Concatenate all blocks ────────────────────────────────────────────────
    blocks = [f100]
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

    reducer = _umap.UMAP(
        n_neighbors  = int(cfg.get("umap_n_neighbors",  60)),
        n_components = int(cfg.get("umap_n_components",  2)),
        min_dist     = float(cfg.get("umap_min_dist",  0.1)),
        random_state = int(cfg.get("umap_random_state", 42)),
        verbose      = False,
    )
    return reducer, reducer.fit_transform(feats_sc_T)


#  
#  HDBSCAN  (auto-sweep min_cluster_size - B-SOiD default strategy)
#  

def run_hdbscan(embedding: np.ndarray, cfg: dict, n_total: int = None):
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

    ref_n = n_total if n_total is not None else embedding.shape[0]

    # ── User preferences ──────────────────────────────────────────────────────
    target_n = int(cfg.get("target_n_clusters", 0))       # 0 = no specific target
    pref_lo  = int(cfg.get("preferred_clusters_lo", 8))   # auto-mode lower bound
    pref_hi  = int(cfg.get("preferred_clusters_hi", 30))  # auto-mode upper bound

    # ── Sweep bounds ──────────────────────────────────────────────────────────
    # pct values are in units of 0.1 % of ref_n.
    # pct=5  → mcs ≈ 0.5 % of ref_n   (finer clusters, higher counts)
    # pct=80 → mcs ≈ 8.0 % of ref_n   (coarser clusters, lower counts)
    pct_lo = max(5, int(np.ceil(500.0 / ref_n)))   # ≥ 0.5 % of bins, min 5
    pct_hi = 80                                     # 8.0 %
    n_steps = 25

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

    methods = [m.strip() for m in
               str(cfg.get("hdbscan_methods_to_try", "eom,leaf")).split(",")
               if m.strip()]

    # ── Sweep: collect every viable candidate ─────────────────────────────────
    # tuple: (score, n_clusters, labels, clf)
    candidates = []

    for method in methods:
        for pct in pcts:
            mcs = max(2, int(round(0.001 * pct * ref_n)))
            clf = _hdb.HDBSCAN(
                prediction_data          = True,
                min_cluster_size         = mcs,
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
            metric                   = cfg.get("hdbscan_metric", "euclidean"),
            cluster_selection_method = methods[0],
        ).fit(embedding)
        return clf, clf.labels_.copy(), getattr(clf, "relative_validity_", float("nan"))

    best_dbcv = max(s for s, *_ in candidates)

    # ── Selection strategy ────────────────────────────────────────────────────
    if target_n > 0:
        # User-guided: pick closest to target with DBCV ≥ 75 % of best.
        thresh    = best_dbcv * 0.75 if best_dbcv > 0 else best_dbcv - 0.1
        qualified = [c for c in candidates if c[0] >= thresh] or candidates
        qualified.sort(key=lambda c: (abs(c[1] - target_n), -c[0]))
        chosen = qualified[0]
    else:
        # Auto mode: prefer solutions in [pref_lo, pref_hi].
        in_range = [c for c in candidates if pref_lo <= c[1] <= pref_hi]
        if in_range:
            in_range.sort(key=lambda c: -c[0])
            chosen = in_range[0]
        else:
            # No candidate in preferred range — pick closest to range boundary
            # among solutions with DBCV ≥ 75 % of best.
            thresh = best_dbcv * 0.75 if best_dbcv > 0 else best_dbcv - 0.1
            boundary = sorted(
                [c for c in candidates if c[0] >= thresh],
                key=lambda c: min(abs(c[1] - pref_lo), abs(c[1] - pref_hi))
            )
            chosen = boundary[0] if boundary else \
                     sorted(candidates, key=lambda c: -c[0])[0]

    best_score, _, best_labels, best_clf = chosen
    return best_clf, best_labels, best_score


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
                   pca_model=None) -> np.ndarray:
    """
    Return per-frame integer labels for one session using the V2 feature set.
    _umap_model is kept for API / pkl compatibility; the MLP classifier
    operates directly in feature space (no UMAP transform at inference).
    pca_model, if provided, is applied after the StandardScaler and must match
    the one fitted during training.
    """
    feats  = extract_features_v2(xy_smooth, fps, bodyparts,
                                  body_normalise=body_normalise)   # (n_feat, n_bins)
    scaled = scaler.transform(feats.T)                        # (n_bins, n_feat)
    if pca_model is not None:
        scaled = pca_model.transform(scaled)                  # (n_bins, n_pca)
    labels = mlp_model.predict(scaled)                        # (n_bins,)
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
              n_states: int = None, n_iter: int = 100):
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
        except ImportError:
            pass  # scipy unavailable; states may not align perfectly with clusters

    return model


def decode_hmm(hmm_model, frame_labels: np.ndarray) -> np.ndarray:
    """Viterbi decode: returns (n_frames,) int array of HMM state IDs."""
    _, state_seq = hmm_model.decode(
        frame_labels.reshape(-1, 1).astype(int), algorithm="viterbi")
    return state_seq.astype(int)


def plot_duration_comparison(raw_labels: np.ndarray, hmm_labels: np.ndarray,
                              fps: float, out_path: Path):
    """Log-scale bout duration histograms — raw B-SOiD vs HMM-smoothed."""
    def _durations(labels):
        bouts = labels_to_bouts(labels)
        return bouts["Run lengths"].values / fps

    raw_dur = _durations(raw_labels)
    hmm_dur = _durations(hmm_labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=_BG)
    for ax in axes:
        _dark_ax(ax)

    all_dur = np.concatenate([raw_dur, hmm_dur])
    lo   = max(1e-3, float(all_dur.min()))
    hi   = float(all_dur.max()) + 0.1
    bins = np.logspace(np.log10(lo), np.log10(hi), 40)

    for ax, durs, title, col in zip(
            axes,
            [raw_dur, hmm_dur],
            ["Raw B-SOiD  (MLP output)", "HMM-smoothed  (Viterbi)"],
            ["#F28E2B", "#4E79A7"]):
        ax.hist(durs, bins=bins, color=col, edgecolor=_BG, alpha=0.85)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Bout duration (s)")
        ax.set_ylabel("Count (log)")
        ax.set_title(title)
        one_frame = 1.0 / fps
        ax.axvline(one_frame, color="#ff4081", linestyle="--",
                   linewidth=1.2, label=f"1 frame ({one_frame:.3f} s)")
        ax.legend(fontsize=7, facecolor=_PANEL, labelcolor=_TEXT_COL)

    fig.suptitle("Behavioral bout duration  —  before vs. after HMM smoothing",
                 color=_TEXT_COL, fontsize=12)
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_hmm_transition_matrix(hmm_model, out_path: Path,
                                state_names: list = None):
    """Heatmap of the HMM learned transition matrix (transmat_)."""
    A = hmm_model.transmat_
    n = A.shape[0]
    names = state_names or [f"S{i}" for i in range(n)]

    sz = max(6, n * 0.6 + 2)
    fig, ax = plt.subplots(figsize=(sz, sz), facecolor=_BG)
    _dark_ax(ax)
    im = ax.imshow(A, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    cb = plt.colorbar(im, ax=ax)
    cb.ax.tick_params(colors=_TICK_COL)
    cb.set_label("Transition probability", color=_TICK_COL)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right",
                        color=_TICK_COL, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, color=_TICK_COL, fontsize=8)
    ax.set_xlabel("State at t+1", color=_TICK_COL)
    ax.set_ylabel("State at t", color=_TICK_COL)
    ax.set_title("HMM learned transition matrix  A[i→j]  (diagonal = self-persistence)",
                 color=_TEXT_COL)
    cell_fs = max(5, 9 - n // 4)
    for i in range(n):
        for j in range(n):
            # Use white text on dark cells (high probability), black on light cells
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                    fontsize=cell_fs,
                    color="white" if A[i, j] > 0.4 else "black")
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_dual_ethogram(raw_labels: np.ndarray, hmm_labels: np.ndarray,
                        fps: float, out_path: Path, tag: str):
    """Two-row ethogram: row 1 = raw B-SOiD MLP, row 2 = HMM Viterbi."""
    uniq_raw = np.unique(raw_labels)
    uniq_hmm = np.unique(hmm_labels)
    t = np.arange(len(raw_labels)) / fps

    n_raw = len(uniq_raw)
    n_hmm = len(uniq_hmm)

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
    ax_raw.set_yticklabels([f"C{l}" for l in uniq_raw], color=_TEXT_COL, fontsize=7)
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
            if i != j and A[i, j] >= min_prob:
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
    edge_list = [(u, v) for u, v, _ in edges]
    e_widths  = [max(1.2, d["weight"] / max_wt * 12) for _, _, d in edges]
    e_alphas  = [0.55 + d["weight"] / max_wt * 0.45 for _, _, d in edges]
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
        f"Behavioral Syntax Network  (p > {min_prob:.2f})\n"
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
            if p < min_prob:
                continue
            p_norm = p / max_wt
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
    ax.set_yscale("log")
    title = f"Dwell-time distributions per state  –  {tag}" if tag else \
            "Dwell-time distributions per state"
    ax.set_title(title, color=_TEXT_COL, fontsize=11, fontweight="bold")
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

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
    cmap_traj = plt.cm.plasma
    for k in range(len(idx) - 1):
        ax_tr.plot(emb[idx[k]:idx[k + 2], 0],
                   emb[idx[k]:idx[k + 2], 1],
                   color=cmap_traj(t_norm[k]),
                   alpha=0.55, linewidth=0.7, solid_capstyle="round")

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
    fig, ax = plt.subplots(figsize=(9, 8), facecolor=_BG)
    _dark_ax(ax)
    for u in uniq:
        m = valid & (labels == u)
        ax.scatter(embedding[m, 0], embedding[m, 1],
                   s=2, alpha=0.5, color=_cmap(u), label=f"C{u}")
    ax.set_title(f"UMAP embedding  [{tag}]", color=_TEXT_COL, fontsize=13)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    handles = [mpatches.Patch(color=_cmap(u), label=f"C{u}")
               for u in uniq[:20]]
    ax.legend(handles=handles, fontsize=7, ncol=4,
              facecolor=_PANEL, edgecolor=_PANEL,
              labelcolor=_TEXT_COL, loc="upper right")
    _savefig(fig, out_path)


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
                  out_path: Path, tag: str):
    uniq = np.unique(frame_labels)
    t    = np.arange(len(frame_labels)) / fps
    fig, ax = plt.subplots(
        figsize=(14, max(3, len(uniq) * 0.55)), facecolor=_BG)
    _dark_ax(ax)
    for idx_u, lbl in enumerate(uniq):
        sel = np.where(frame_labels == lbl)[0]
        ax.scatter(t[sel], np.full(len(sel), idx_u),
                   c=_cmap(lbl), s=8, marker="|", linewidths=3.5)
    ax.set_yticks(range(len(uniq)))
    ax.set_yticklabels([f"C{l}" for l in uniq], color=_TEXT_COL, fontsize=8)
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
        ax.bar(xpos, vals, color=colors, edgecolor=_BG, alpha=0.9)
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
           edgecolor=_BG, alpha=0.9)
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
    """Behavioral state transition probability matrix  P(next | current)."""
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
    sz = max(6, n * 0.55 + 2)
    fig, ax = plt.subplots(figsize=(sz, sz), facecolor=_BG)
    _dark_ax(ax)
    im = ax.imshow(T, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    cb = plt.colorbar(im, ax=ax)
    cb.ax.tick_params(colors="#aaaacc")
    cb.set_label("Transition probability", color="#aaaacc")
    tick_lbls = [f"C{l}" for l in labs]
    ax.set_xticks(range(n)); ax.set_xticklabels(tick_lbls, rotation=45, ha="right",
                                                  color="#aaaacc", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(tick_lbls, color="#aaaacc", fontsize=8)
    ax.set_xlabel("Next cluster"); ax.set_ylabel("Current cluster")
    ax.set_title("Behavioral state transition probabilities  P(next | current)")
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
                ax.bar(range(len(n_bad)), n_bad, color=colors, edgecolor=_BG)
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
    ax1.bar(range(k), cv_scores, color=colors, edgecolor=_BG, alpha=0.9)
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
                          clips_per_cluster: "dict | None" = None):
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
        step           = max(1, int(round(source_fps / output_fps)))
        max_out_frames = max(1, int(max_clip_dur_sec * output_fps))
        max_src_frames = max_out_frames * step
        # For gaps narrower than this many source frames, read-and-discard is
        # cheaper than a keyframe seek (tune: ~2 s at source fps).
        skip_threshold = max(step, int(source_fps * 2))
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
                ef = min(total - 1, int(row.end_frame))
                ef = min(ef, sf + max_src_frames - 1)
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
                src_idx    = 0   # frames read since clip start
                while cap_pos <= ef and out_frames < max_out_frames:
                    ret, frame = cap.read()
                    if not ret:
                        cap_pos = -1
                        break
                    cap_pos += 1
                    if src_idx % step == 0:
                        cv2.rectangle(frame, (0, 0), (230, 46), (0, 0, 0), -1)
                        cv2.putText(frame, f"Cluster {grp}",
                                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, col_bgr, 2, cv2.LINE_AA)
                        cv2.putText(frame, label_text,
                                    (8, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (200, 200, 200), 1)
                        writer.write(frame)
                        out_frames += 1
                    src_idx += 1
            finally:
                writer.release()
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
                   min_confidence: float = MIN_BODYPART_CONFIDENCE) -> Path | None:
    """
    Scan source_folder for *_filtered.h5 files, filter to conserved bodyparts,
    export BSOID_Project_Ready/ structure.

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

    # Determine conserved bodyparts
    first_bps = list(list(all_stats.values())[0][1].keys())
    conserved = [
        bp for bp in first_bps
        if all(bp in ss and ss[bp]["mean"] >= min_confidence
               for _, ss in all_stats.values())
    ]
    log_fn(f"  Conserved bodyparts ({len(conserved)}/{len(first_bps)}): {conserved}")

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

    # ── Publication-faithful defaults ─────────────────────────────────────────
    # Parameters match Hsu & Bhatt et al. (2021) PLOS Comput. Biol. exactly.
    # Only min_epoch_dur_s / max_epoch_dur_s are intended as user inputs.
    DEFAULTS = dict(
        likelihood_thresh     = 0.3,    # DLC confidence threshold (pub. default)
        boxcar_win_sec        = 0.07,   # 70 ms boxcar smoothing (pub. default)
        train_frac            = 0.3,    # fraction of bins for UMAP when N > threshold
        umap_full_thresh      = 10_000, # use full data for UMAP when N <= this
        umap_n_neighbors      = 60,     # UMAP k-nearest neighbours (pub. default)
        umap_n_components     = 3,      # 3-D UMAP embedding (pub. default)
        umap_min_dist         = 0.1,    # UMAP min_dist (pub. default)
        umap_random_state     = 42,     # reproducibility seed
        hdbscan_metric        = "euclidean",  # (pub. default)
        hdbscan_method        = "eom",        # excess-of-mass (pub. default)
        mlp_hidden            = "100,50",     # 2-layer MLP (pub. default)
        mlp_max_iter          = 1000,
        cv_folds              = 5,
        # ── HDBSCAN options ───────────────────────────────────────────────────
        hdbscan_methods_to_try = "eom,leaf",  # both tried; selection logic picks best
        # ── Cluster count guidance ────────────────────────────────────────────
        target_n_clusters     = 0,    # 0 = auto; >0 = user-requested cluster count
        preferred_clusters_lo = 8,    # auto-mode: prefer cluster count ≥ this
        preferred_clusters_hi = 30,   # auto-mode: prefer cluster count ≤ this
        # ── Feature options ───────────────────────────────────────────────────
        body_normalise        = True,   # normalise by nose-to-tailbase length
        pca_pre_reduce        = "auto", # auto/on/off — reduce dims before UMAP
        # ── Primary user inputs (bout duration filter) ────────────────────────
        min_epoch_dur_s       = 0.0,    # minimum cluster bout duration (seconds)
        max_epoch_dur_s       = 1e9,    # maximum cluster bout duration (seconds)
        # ── Output options ────────────────────────────────────────────────────
        output_fps            = 15,
        max_clips_per_cluster = 3,
        save_plots            = True,
        save_videos           = True,
        save_example_clips    = True,
        save_labeled_video    = True,
        delete_labeled_videos = True,   # delete labeled_videos/ folder after run
        # ── Plot appearance ───────────────────────────────────────────────────
        plot_theme            = "dark",   # "dark" or "light"
        # ── HMM post-hoc smoothing ────────────────────────────────────────────
        hmm_enabled           = True,    # wrap MLP output with Multinomial HMM
        hmm_n_states          = None,    # None → n_clusters (smoothing-only mode)
        hmm_n_iter            = 100,     # Baum-Welch EM iterations
        hmm_min_prob          = 0.05,    # min edge probability in syntax network plot
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

        # ── Faithfulness audit: log publication-matched parameters ────────────
        self._log("\n[AUDIT] Publication parameter verification"
                  " (Hsu & Bhatt et al., 2021 PLOS Comput. Biol.)")
        self._log(f"  likelihood_thresh : {self._cfg['likelihood_thresh']}  "
                  "(pub: 0.3)")
        self._log(f"  boxcar_win_sec    : {self._cfg['boxcar_win_sec']}  "
                  "(pub: 0.07 s)")
        self._log(f"  umap_n_neighbors  : {self._cfg['umap_n_neighbors']}  "
                  "(pub: 60)")
        self._log(f"  umap_n_components : {self._cfg['umap_n_components']}  "
                  "(pub: 3)")
        self._log(f"  umap_min_dist     : {self._cfg['umap_min_dist']}  "
                  "(pub: 0.1)")
        self._log(f"  hdbscan_sweep     : 0.6 %–2.0 % of N  (pub: sweep)")
        self._log(f"  mlp_hidden        : {self._cfg['mlp_hidden']}  "
                  "(pub: 100,50)")
        self._log(f"  train_frac        : {self._cfg['train_frac']}  "
                  "(pub: 0.3)")
        _bn  = self._cfg.get("body_normalise", True)
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
        all_xy, all_names, all_fps_list, all_bps = [], [], [], []
        for i, (fp, _) in enumerate(pairs):
            self._log(f"  [{i+1}/{len(pairs)}]  {fp.name}")
            self._stage("2/7 — Loading DLC files", f"{i+1}/{len(pairs)}: {fp.name}")
            try:
                xy, bps, fps_hint = load_dlc_file(
                    fp, self._cfg["likelihood_thresh"])
                fps = fps_hint or self._fps_arg or 30.0
                xy  = smooth_boxcar(xy, fps, self._cfg["boxcar_win_sec"])
                all_xy.append(xy)
                all_names.append(fp.stem)
                all_fps_list.append(float(fps))
                all_bps.append(bps)
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
        # Filter every xy array to the common bodypart columns
        for k, (bps_k, xy_k) in enumerate(zip(all_bps, all_xy)):
            if bps_k != bps_ref:
                col_idx = []
                for bp in bps_ref:
                    if bp in bps_k:
                        j = bps_k.index(bp)
                        col_idx.extend([2 * j, 2 * j + 1])
                all_xy[k] = xy_k[:, col_idx]

        fps = float(pd.Series(all_fps_list).mode()[0])
        self._log(f"  FPS = {fps}  |  bodyparts = {len(bps_ref)}")

        if self._cfg["save_plots"]:
            try:
                plot_likelihood_qc(dlc_files,
                                   self._out_plots / "likelihood_qc.png")
            except Exception:
                self._log(f"  [WARN] likelihood_qc plot: "
                          f"{traceback.format_exc()}")

        # 3. Features (V2 — multi-scale, body-normalised, angular)
        _body_norm = bool(self._cfg.get("body_normalise", True))
        _scale_desc = ("50/100/200 ms" if fps >= 60 else "100/200 ms")
        self._log(f"\n[3/7]  Extracting V2 features  "
                  f"({_scale_desc} + angular, body_normalise={_body_norm})...")
        self._stage("3/7 — Extracting V2 features", f"scale={_scale_desc}")
        all_feats = []
        for i, (xy, name) in enumerate(zip(all_xy, all_names)):
            self._stage("3/7 — Extracting V2 features",
                        f"{i+1}/{len(all_xy)}: {name}")
            f = extract_features_v2(xy, fps, bps_ref, body_normalise=_body_norm)
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

        # Use the full dataset when it is small enough (avoids UMAP over-smoothing
        # caused by a large n_neighbors/N_sample ratio); subsample only for large
        # recordings where UMAP runtime becomes a bottleneck.
        umap_full_thresh = int(self._cfg.get("umap_full_thresh", 10_000))
        if n_bins <= umap_full_thresh:
            n_samp    = n_bins
            feats_sub = feats_cat
        else:
            n_samp    = max(1000, int(n_bins * float(self._cfg["train_frac"])))
            idx       = rng.choice(n_bins, n_samp, replace=False)
            feats_sub = feats_cat[:, idx]

        self._log(f"  Total bins: {n_bins}  -> UMAP sample: {n_samp} "
                  f"({100 * n_samp / n_bins:.0f} %)")

        from sklearn.preprocessing import StandardScaler
        scaler   = StandardScaler()
        feats_sc = scaler.fit_transform(feats_sub.T).T   # (n_feat, n_samp)

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

        # Validation gate 3: UMAP trustworthiness
        try:
            _validation["umap_trustworthiness"] = validate_umap_trustworthiness(
                feats_sc.T, embedding)
            for w in _validation["umap_trustworthiness"]["warnings"]:
                self._log(f"  [VALID-WARN] {w}")
        except Exception as e:
            self._log(f"  [VALID] Trustworthiness check failed: {e}")

        # 5. HDBSCAN
        self._log("\n[5/7]  HDBSCAN clustering  "
                  "(adaptive sweep, DBCV criterion)...")
        self._stage("5/7 — HDBSCAN clustering",
                    f"sweeping min_cluster_size over {n_bins} bins…")
        hdb_clf, hdb_labels, hdb_score = run_hdbscan(
            embedding, self._cfg, n_total=n_bins)
        n_cl      = len(set(hdb_labels[hdb_labels >= 0]))
        noise     = (hdb_labels < 0).sum()
        noise_pct = 100 * noise / max(1, len(hdb_labels))
        self._log(f"  {n_cl} clusters, {noise} noise points "
                  f"({noise_pct:.1f} %), "
                  f"DBCV={hdb_score:.3f}")
        self._stage("5/7 — HDBSCAN done",
                    f"{n_cl} clusters · {noise_pct:.0f}% noise · DBCV={hdb_score:.3f}")

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

        if self._cfg["save_plots"]:
            try:
                plot_umap(embedding, hdb_labels,
                          self._out_plots / "umap_embedding.png")
            except Exception:
                pass

        # Save UMAP embedding + cluster labels as numpy arrays so cube_analyser
        # can display before/after UMAP views when the user recombines clusters.
        try:
            np.save(str(self._out_model / "umap_embedding.npy"), embedding)
            np.save(str(self._out_model / "umap_labels.npy"),    hdb_labels)
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
                created     = datetime.now().isoformat(),
            ), fh)
        self._log(f"  Model saved -> {model_path}")

        (self._out_model / "feature_config.json").write_text(
            json.dumps(dict(fps=fps, bodyparts=bps_ref,
                            boxcar_win_sec=self._cfg["boxcar_win_sec"],
                            n_features=int(feats_cat.shape[0]),
                            feature_version="v2",
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
                    pca_model=pca_model)
            else:
                # Fallback: no MLP → use HDBSCAN approximate_predict on V2 feats
                try:
                    import hdbscan as _hdb
                    f    = extract_features_v2(xy, file_fps, bps_ref,
                                              body_normalise=_body_norm)
                    sc   = scaler.transform(f.T)
                    if pca_model is not None:
                        sc = pca_model.transform(sc)
                    emb  = umap_model.transform(sc)
                    soft, _ = _hdb.approximate_predict(hdb_clf, emb)
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
            for _vp, _ep, _fps, _aid in _clip_tasks:
                try:
                    create_example_clips(
                        _vp, _ep, self._out_videos, _fps,
                        output_fps=int(self._cfg["output_fps"]),
                        max_clips=int(self._cfg["max_clips_per_cluster"]),
                        animal_id=_aid,
                        clips_per_cluster=_clips_per_cluster)
                except Exception:
                    self._log(f"  [WARN] Clips: {traceback.format_exc()}")

        # Combined epochs across all sessions
        if all_epochs:
            combined = pd.concat(
                [ep.assign(session=n) for ep, n in all_epochs if not ep.empty],
                ignore_index=True)
            combined.to_csv(
                str(self.output_dir / "all_epochs_combined.csv"), index=False)

        if self._cfg["save_plots"] and all_frame_labels:
            try:
                plot_transition_matrix(
                    all_frame_labels,
                    self._out_plots / "transition_matrix.png")
            except Exception:
                self._log(f"  [WARN] transition_matrix: "
                          f"{traceback.format_exc()}")

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
                    hdb_labels,
                    fps,
                    self._out_plots / "state_space_projection.png",
                    _tag0)
                self._log("  [PLOT] state_space_projection.png saved")
            except Exception:
                self._log(f"  [WARN] state_space_projection: "
                          f"{traceback.format_exc()}")

        # Auto-groups
        clusters_seen = sorted(set(
            int(l)
            for ep, _ in all_epochs if not ep.empty
            for l in ep.label.unique()))
        groups = {f"C{c}": {"labels": [c], "color": _cmap(c)}
                  for c in clusters_seen}

        # ── Validation report ─────────────────────────────────────────────────
        all_warnings = [w for r in _validation.values()
                        for w in r.get("warnings", [])]
        any_block    = any(r.get("blocked", False)
                          for r in _validation.values())
        val_report   = dict(
            cube_version   = VERSION,
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
            fl    = predict_labels(xy, umap_m, mlp_m, scaler, fps)
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
