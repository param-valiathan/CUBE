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
import gc, json, os, pickle, re, shutil, time, traceback, warnings
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
    """
    Colour for cluster id i.  Ids within the base PALETTE get their exact
    literal colour (unchanged from before -- every existing run with <20
    clusters looks identical).  Beyond that, colours are drawn from a
    continuous colormap with golden-ratio hue spacing instead of wrapping
    PALETTE modulo its length -- wrapping made e.g. an 84-cluster split/merge
    plot visually indistinguishable from a ~20-cluster one, since ids 20
    positions apart got the exact same colour.
    """
    i = int(i)
    if i < len(PALETTE):
        return PALETTE[i]
    _extra = matplotlib.colormaps["gist_rainbow"]
    _frac = ((i - len(PALETTE)) * 0.61803398875) % 1.0   # golden-ratio spacing
    r, g, b, _a = _extra(_frac)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

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


def _h5_has_z(path) -> bool:
    """Return True if a DLC file (H5 or CSV) contains a 'z' coord column (3D output).

    BSoidEngine discovers BSOID-ready files from the csv/ subdirectory, so
    this function must handle CSV files as well as H5/HDF5 files.  It reads
    only the header rows for CSVs to keep the check fast.
    """
    try:
        _ext = Path(str(path)).suffix.lower()
        if _ext in (".h5", ".hdf5"):
            df = pd.read_hdf(str(path))
        elif _ext == ".csv":
            with open(str(path), encoding="utf-8", errors="replace") as _fh:
                _head = [_fh.readline() for _ in range(5)]
            _n_lev = max(2, sum(1 for _l in _head[:4]
                                if _l.strip() and not _l.strip()[0].isdigit()))
            df = pd.read_csv(str(path), nrows=1,
                             header=list(range(_n_lev)), index_col=0)
        else:
            return False
        # Use level index -1 (last level) which is always coords for both
        # normalised H5 and BSOID-ready CSV MultiIndex headers.
        return "z" in df.columns.get_level_values(-1).unique()
    except Exception:
        return False


def load_dlc_file(path, likelihood_thresh: float = 0.3,
                  max_interp_gap_frames: int = None, log_fn=None,
                  return_quality: bool = False, include_z: bool = False):
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
    ll          : np.ndarray  (N_frames, n_bodyparts) raw per-frame likelihood,
        BEFORE interpolation (only when return_quality=True) — used by issue
        2's adaptive visibility/occlusion feature block.
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

    # Detect 3D coord columns (written by cube_3d_dlc triangulation)
    _all_coords = df.columns.get_level_values("coords").unique().tolist()
    _has_z  = include_z and ("z" in _all_coords)
    n_coord = 3 if _has_z else 2   # columns per bodypart in the output array

    xy = np.full((n_frames, n_pts * n_coord), np.nan, dtype=float)
    ll = np.full((n_frames, n_pts),           np.nan, dtype=float)

    for i, bp in enumerate(bodyparts):
        try:
            sub = df.xs(bp, level="bodyparts", axis=1)
            # drop extra scorer level if present
            if sub.columns.nlevels > 1:
                sub.columns = sub.columns.get_level_values(-1)
            xy[:, n_coord*i]   = pd.to_numeric(
                sub.get("x",   pd.Series(np.nan, index=df.index)),
                errors="coerce").values
            xy[:, n_coord*i+1] = pd.to_numeric(
                sub.get("y",   pd.Series(np.nan, index=df.index)),
                errors="coerce").values
            if _has_z:
                xy[:, n_coord*i+2] = pd.to_numeric(
                    sub.get("z", pd.Series(0.0, index=df.index)),
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
        bp_i = col // n_coord
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
                            if col % n_coord == 0:
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
        return xy, bodyparts, fps_hint, ll_fracs, _flat_held_per_bp, ll
    return xy, bodyparts, fps_hint


#  
#  SMOOTHING
#  

def smooth_boxcar(xy: np.ndarray, fps: float, win_sec: float) -> np.ndarray:
    """Centred boxcar (moving average) smoothing per column."""
    n = xy.shape[0]
    if n == 0:
        return xy.copy()
    win = max(1, int(round(fps * win_sec)))
    # Clamp to the sequence length: np.convolve(..., mode="same") returns
    # max(len(a), len(kernel)) rather than len(a) whenever the kernel is
    # longer than the signal, which desyncs frame counts downstream.
    win = min(win, n)
    if win <= 1:
        return xy.copy()
    k = np.ones(win) / win
    return np.column_stack(
        [np.convolve(xy[:, c], k, mode="same") for c in range(xy.shape[1])]
    )


#
#  V2 FEATURE EXTRACTION HELPERS
#

#
#  BODY-REGION GROUPING (shared by feature weighting and visibility features)
#

# Ordered MOST-SPECIFIC-FIRST: real DLC bodypart lists (e.g. the SuperAnimal-
# Quadruped keypoint set: nose, upper_jaw, lower_jaw, right_eye, right_earbase,
# right_earend, right_antler_base, left_eye, left_earbase, left_earend,
# left_antler_base, neck_base, neck_end, throat_base, throat_end, back_base,
# back_end, back_middle, tail_base, front_left_thai, front_left_knee,
# front_left_paw, front_right_thai, front_right_knee, front_right_paw,
# back_left_paw, back_right_paw) contain an ambiguity trap: "back" appears in
# BOTH trunk names (back_base, back_end, back_middle) AND hindlimb names
# (back_left_paw, back_right_paw).  A flat/unordered "back" substring match
# would misclassify hindlimb paws as Trunk.  Hindlimb/forelimb COMPOUND
# tokens ("back_left", "front_right", ...) are therefore checked before the
# generic, greedy Trunk/Back keywords ("back", "spine", ...) — order matters.
_REGION_KEYWORDS = [
    ("Hindlimbs",    ["back_left", "back_right", "backleft", "backright",
                       "hindlimb", "hind_limb", "hindleg", "hind_leg",
                       "hindpaw", "hind_paw", "hind", "rear_left", "rear_right",
                       "rear", "ankle", "heel", "hock", "backpaw", "back_paw"]),
    ("Forelimbs",    ["front_left", "front_right", "frontleft", "frontright",
                       "forelimb", "fore_limb", "foreleg", "fore_leg",
                       "forepaw", "fore_paw", "fore", "thai", "thigh",
                       "elbow", "wrist", "forearm"]),
    ("Neck",         ["neck", "throat", "gorge", "nape"]),
    ("Tail",         ["tailbase", "tail_base", "tail"]),
    ("Head / Mouth", ["nose", "snout", "jaw", "mouth", "lip", "chin", "head",
                       "rostral", "eye", "ear", "antler", "cheek", "face",
                       "whisker", "muzzle"]),
    ("Trunk / Back", ["spine", "back", "body", "trunk", "hip", "sacrum",
                       "pelvis", "shoulder", "withers", "flank", "chest",
                       "belly", "abdomen", "girdle"]),
]


def group_bodyparts_by_region(bodyparts: list) -> dict:
    """
    Keyword-classify bodyparts into coarse anatomical regions for feature
    weighting (issue 1b) and visibility diagnostics (issue 2).  Bodyparts are
    arbitrary per-DLC-project strings with no fixed schema, so this uses the
    same keyword-priority idiom as _find_spine_indices/_angular_features —
    EXCEPT region matching is ORDER-SENSITIVE (most-specific-compound-token
    first: Hindlimbs -> Forelimbs -> Neck -> Tail -> Head/Mouth -> Trunk/Back
    -> Other) to avoid the "back" ambiguity trap: back_left_paw/back_right_paw
    (Hindlimbs) vs. back_base/back_end/back_middle (Trunk/Back) — see
    _REGION_KEYWORDS' comment.  Within each region the first matching keyword
    wins; first MATCHING REGION in priority order wins overall.

    Returns {region_name: [bodypart, ...]}.  ALL region names are always
    present as keys (possibly with an empty list) so downstream consumers
    (compute_visibility_features) get a fixed, dataset-independent column
    count regardless of which regions actually have matched bodyparts.
    Unmatched bodyparts go to 'Other'.
    """
    regions = {name: [] for name, _ in _REGION_KEYWORDS}
    regions["Other"] = []
    for bp in (bodyparts or []):
        bp_l = str(bp).lower()
        matched = False
        for name, kws in _REGION_KEYWORDS:
            if any(kw in bp_l for kw in kws):
                regions[name].append(bp)
                matched = True
                break
        if not matched:
            regions["Other"].append(bp)
    return regions


def _build_bodypart_weight_vector(bodyparts: list, bodypart_weights: "dict | None",
                                   n_pts: int) -> np.ndarray:
    """
    w[i] = bodypart_weights.get(bodyparts[i], 1.0).  Falls back to an all-ones
    vector (the no-op default) when bodypart_weights is None/empty or
    bodyparts is missing/length-mismatched — this is what guarantees
    extract_features_v2/_3d reproduce bit-identical output when no weighting
    is requested (multiplying by an exact 1.0 introduces no floating-point
    error).
    """
    if not bodypart_weights or not bodyparts or len(bodyparts) != n_pts:
        return np.ones(n_pts, dtype=float)
    return np.array([float(bodypart_weights.get(bp, 1.0)) for bp in bodyparts],
                    dtype=float)


def peek_dlc_bodyparts(path) -> list:
    """
    Read only the header of a DLC CSV/H5 file to discover its bodypart list,
    WITHOUT loading/interpolating the full pose data (contrast with
    load_dlc_file's full load/interpolate path).  Lets the GUI discover
    bodyparts before running the pipeline (e.g. for a body-region weighting
    picker).  Returns [] on any failure.
    """
    path = str(path)
    ext  = Path(path).suffix.lower()
    try:
        if ext in (".h5", ".hdf5"):
            df = pd.read_hdf(path, stop=1)
        elif ext == ".csv":
            with open(path, encoding="utf-8", errors="replace") as fh:
                head = [fh.readline() for _ in range(5)]
            n_levels = sum(
                1 for l in head[:4]
                if l.strip() and not l.strip()[0].isdigit()
            )
            n_levels = max(n_levels, 2)
            df = pd.read_csv(path, nrows=0, header=list(range(n_levels)),
                             index_col=0)
        else:
            return []
        df = _normalise_dlc_df(df)
        return df.columns.get_level_values("bodyparts").unique().tolist()
    except Exception:
        return []


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
#  ADAPTIVE VISIBILITY / OCCLUSION FEATURES  (issue 2 — "turned away" isolation)
#

def compute_adaptive_visibility_threshold(ll: np.ndarray, likelihood_thresh: float,
                                           adaptive_pct: float) -> np.ndarray:
    """
    Per-bodypart, per-session low-confidence threshold:
        max(likelihood_thresh, percentile(ll[:, i], adaptive_pct))
    Floored at the existing global constant so this only ever flags
    ADDITIONAL, session-relative degradation on top of what load_dlc_file's
    own interpolation already treats as bad — it never lowers the bar.
    ll : (n_frames, n_pts) raw per-frame likelihood array (as already computed
         inside load_dlc_file, before interpolation).
    Returns (n_pts,).
    """
    if ll is None or ll.size == 0:
        return np.array([], dtype=float)
    n_pts = ll.shape[1]
    out = np.empty(n_pts, dtype=float)
    for i in range(n_pts):
        col = ll[:, i]
        col = col[~np.isnan(col)]
        pct = float(np.percentile(col, adaptive_pct)) if col.size else likelihood_thresh
        out[i] = max(float(likelihood_thresh), pct)
    return out


def compute_visibility_features(ll: np.ndarray, bodyparts: list, win: int,
                                 adaptive_thresh: np.ndarray) -> np.ndarray:
    """
    Per-100ms-bin occlusion/visibility feature block (bin-aligned with
    win100 binning used everywhere else in the V2/3D feature extractors):
        col 0        : mean likelihood across bodyparts in the bin
        col 1        : fraction of bodyparts below their adaptive per-bodypart
                        threshold in the bin
        col 2..N     : fraction of THAT region's bodyparts below threshold in
                        the bin, one column per region from
                        group_bodyparts_by_region (fixed region count/order,
                        sorted by region name, regardless of which bodyparts
                        are actually present — keeps column count constant
                        across sessions for np.hstack).

    Returns (n_bins, 2 + n_regions).  n_bins = n_frames // win (same truncation
    convention as extract_features_v2's win100 binning).
    """
    regions    = group_bodyparts_by_region(bodyparts or [])
    region_names = sorted(regions)
    n_regions  = len(region_names)

    if ll is None or ll.size == 0:
        return np.zeros((0, 2 + n_regions), dtype=float)

    n_f, n_pts = ll.shape
    n_bins = n_f // max(1, win)
    if n_bins < 1 or n_pts == 0:
        return np.zeros((0, 2 + n_regions), dtype=float)

    ll_trim = ll[:n_bins * win]
    bin_ll  = np.nanmean(ll_trim.reshape(n_bins, win, n_pts), axis=1)  # (n_bins, n_pts)
    bin_ll  = np.nan_to_num(bin_ll, nan=0.0)

    thresh = np.asarray(adaptive_thresh, dtype=float)
    if thresh.shape[0] != n_pts:
        _fallback = float(np.nanmean(thresh)) if thresh.size else 0.3
        thresh = np.full(n_pts, _fallback)
    bad = bin_ll < thresh[None, :]                       # (n_bins, n_pts) bool

    mean_likelihood = bin_ll.mean(axis=1)                 # col 0
    frac_low_conf   = bad.mean(axis=1)                    # col 1

    region_cols = []
    for name in region_names:
        idx = [bodyparts.index(bp) for bp in regions[name] if bp in bodyparts]
        region_cols.append(bad[:, idx].mean(axis=1) if idx
                           else np.zeros(n_bins, dtype=float))

    return np.column_stack([mean_likelihood, frac_low_conf] + region_cols)


def visibility_feature_names(bodyparts: list) -> list:
    """Column names matching compute_visibility_features' output order —
    used by compute_cluster_confidence_profile / any CSV export."""
    region_names = sorted(group_bodyparts_by_region(bodyparts or []))
    return (["mean_visibility", "frac_low_conf"]
            + [f"frac_low_conf_{n.replace(' / ', '_').replace(' ', '_')}"
               for n in region_names])


def compute_session_visibility_block(ll: "np.ndarray | None", bodyparts: list,
                                      fps: float, likelihood_thresh: float = 0.3,
                                      adaptive_pct: float = 10) -> "np.ndarray | None":
    """
    One-call wrapper: adaptive threshold + compute_visibility_features for a
    single session.  Returns None when ll is unusable (None/empty/no
    bodyparts) so callers can skip cleanly — this is how
    visibility_features_enabled=False (or a legacy caller with no likelihood
    array available) reproduces the pre-2.2 feature layout exactly.
    """
    if ll is None or not bodyparts or getattr(ll, "size", 0) == 0:
        return None
    win = max(1, int(round(fps / 10)))
    thresh = compute_adaptive_visibility_threshold(ll, likelihood_thresh, adaptive_pct)
    return compute_visibility_features(ll, bodyparts, win, thresh)


def _append_visibility_block(f: np.ndarray, vis: "np.ndarray | None") -> np.ndarray:
    """
    Vstack a (n_bins, n_vis_cols) visibility block onto an (n_features, n_bins)
    feature matrix, defensively aligning bin counts (should already match
    exactly since both are derived from the same win-based binning of the
    same-length arrays).  No-op if vis is None/empty.  MUST be called
    identically at training-time per-session extraction and inference-time
    re-extraction — see BSoidEngine.run()'s two call sites.
    """
    if vis is None or vis.size == 0:
        return f
    n = min(vis.shape[0], f.shape[1])
    if n <= 0:
        return f
    return np.vstack([f[:, :n], vis[:n].T])


def detect_turned_away_bins(ll: np.ndarray, bodyparts: list, fps: float,
                             likelihood_thresh: float, adaptive_pct: float,
                             head_frac_on: float, min_window_s: float,
                             merge_gap_s: float) -> np.ndarray:
    """
    Per-bin boolean mask, True where the animal is judged turned away from
    camera.  Validated algorithm (v3, this session's corroboration passes
    against real DLC data / human video review — see
    CUBE_ANALYSIS_METHODOLOGY.md): a bin counts as turned-away when BOTH
    (a) the Head/Mouth region's frac_low_conf is >= head_frac_on AND (b) the
    nose keypoint's own bin-mean likelihood is below its adaptive
    per-bodypart/per-session threshold — the nose alone distinguishes "not
    facing the camera" from a one-sided head turn or motion blur that
    degrades several head keypoints without the nose itself losing
    confidence.  Sustained-window debouncing (contiguous-bin merge, gap
    merge, min_window_s duration floor) then drops single-bin jitter before
    windows are converted back to a per-bin mask.

    Reuses compute_adaptive_visibility_threshold / compute_session_visibility_
    block / visibility_feature_names for the Head/Mouth fraction and adaptive
    thresholds rather than recomputing independently — this MUST stay
    algorithmically identical to those functions' binning (win = round(fps/10))
    so the mask lines up bin-for-bin with every other 100ms-bin feature/label
    array in the pipeline.

    Guard: returns an all-False mask (matching the (n_bins,) shape implied by
    ll/win) if 'nose' isn't in bodyparts or ll is unusable — the caller should
    log a one-line warning rather than this function raising.
    """
    if ll is None or getattr(ll, "size", 0) == 0 or not bodyparts:
        return np.zeros(0, dtype=bool)
    if "nose" not in [str(b).lower() for b in bodyparts]:
        win = max(1, int(round(fps / 10)))
        n_bins = ll.shape[0] // win
        return np.zeros(max(0, n_bins), dtype=bool)

    vis = compute_session_visibility_block(ll, bodyparts, fps, likelihood_thresh, adaptive_pct)
    if vis is None or vis.size == 0:
        return np.zeros(0, dtype=bool)
    col_names = visibility_feature_names(bodyparts)
    head_col = col_names.index("frac_low_conf_Head_Mouth")

    win = max(1, int(round(fps / 10)))
    n_bins = vis.shape[0]

    bp_lower = [str(b).lower() for b in bodyparts]
    nose_idx = bp_lower.index("nose")
    adaptive_thresh = compute_adaptive_visibility_threshold(ll, likelihood_thresh, adaptive_pct)
    nose_thresh = float(adaptive_thresh[nose_idx])
    n_usable = n_bins * win
    nose_bin_mean = ll[:n_usable, nose_idx].reshape(n_bins, win).mean(axis=1)
    nose_low = nose_bin_mean < nose_thresh

    flagged = (vis[:, head_col] >= head_frac_on) & nose_low

    windows = []
    i = 0
    while i < n_bins:
        if flagged[i]:
            j = i
            while j + 1 < n_bins and flagged[j + 1]:
                j += 1
            windows.append([i * win / fps, (j + 1) * win / fps])
            i = j + 1
        else:
            i += 1

    merged = []
    for w in windows:
        if merged and w[0] - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = w[1]
        else:
            merged.append(w)
    merged = [w for w in merged if (w[1] - w[0]) >= min_window_s]

    mask = np.zeros(n_bins, dtype=bool)
    for w0, w1 in merged:
        b0 = int(round(w0 * fps / win))
        b1 = int(round(w1 * fps / win))
        b0 = max(0, min(n_bins, b0))
        b1 = max(0, min(n_bins, b1))
        if b1 > b0:
            mask[b0:b1] = True
    return mask


def _expand_bin_mask_to_frames(bin_mask: np.ndarray, fps: float,
                                n_frames: int) -> np.ndarray:
    """
    Expand a (n_bins,) boolean/int per-100ms-bin array to (n_frames,), using
    the EXACT SAME arithmetic predict_labels() uses to expand MLP bin labels
    to per-frame labels (win = round(fps/10); np.repeat; edge-pad if short;
    truncate if long) — kept as a single shared helper so the turned-away
    dedicated-label override and video-overlay mask can never drift out of
    alignment with how frame_labels itself is built.
    """
    win = max(1, int(round(fps / 10)))
    fl  = np.repeat(np.asarray(bin_mask), win)
    if len(fl) < n_frames:
        fl = np.pad(fl, (0, n_frames - len(fl)), mode="edge")
    return fl[:n_frames]


def _mask_out_segments(seq: np.ndarray, exclude_mask: np.ndarray) -> list:
    """
    Split seq into contiguous runs with exclude_mask==True dropped -- never
    returns a run that stitches together elements separated by an excluded
    stretch. Used to keep turned-away-from-camera bins/frames out of
    transition-counting and HMM training (plot_transition_matrix,
    _tmat_from_labels, train_hmm/train_hmm_soft) without introducing a fake
    transition across the gap. Works for both 1D label arrays and 2D
    per-bin probability matrices (fancy-indexing on axis 0). Defensively
    truncates to the shorter of the two inputs on a length mismatch.
    """
    n = min(len(seq), len(exclude_mask))
    seq, exclude_mask = seq[:n], np.asarray(exclude_mask[:n], dtype=bool)
    idx = np.flatnonzero(~exclude_mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    return [seq[run] for run in np.split(idx, breaks) if run.size]


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
                         angular_fallback: bool = True,
                         long_lag_drift: bool = False,
                         bodypart_weights: "dict | None" = None) -> np.ndarray:
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
    bodypart_weights : optional {bodypart_name: multiplier} (default 1.0 for any
        bodypart absent from the dict).  Pairwise-distance columns are scaled by
        sqrt(w_i * w_j); per-bodypart velocity/acceleration/within-bin-variance
        columns are scaled by w_i directly; the angular block is left unweighted
        (no natural sqrt(wi*wj) analogue, scoping decision not an oversight).
        None/{} (default) -> every multiplier is exactly 1.0 -> bit-identical
        output to pre-weighting behaviour.

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

    # ── Optional per-body-region feature weighting (issue 1b) ─────────────────
    # w=1 everywhere (default / bodypart_weights=None) ⇒ every multiplier below
    # is exactly 1.0 ⇒ bit-identical output to pre-weighting behaviour.
    _w = _build_bodypart_weight_vector(bodyparts, bodypart_weights, n_pts)
    _pair_idx = list(combinations(range(n_pts), 2))
    _pair_w = (np.sqrt(_w[[i for i, j in _pair_idx]] * _w[[j for i, j in _pair_idx]])
              if _pair_idx else np.array([]))

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
    # Per-bodypart weighting: column i (x) and column n_pts+i (y) both belong
    # to bodypart i, so tile w twice to match f_withinbin's column order.
    f_withinbin = f_withinbin * np.concatenate([_w, _w])[None, :]

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
    # Weighting (issue 1b) is applied HERE — the single place that builds
    # pairwise distances + velocity + acceleration for all three temporal
    # scales (100/200/50ms) — so it stays consistent across scales with one
    # implementation.  Pairwise distance column (i, j) is scaled by
    # sqrt(w_i * w_j) (geometric mean); per-bodypart velocity column i is
    # scaled by w_i directly.  Acceleration is computed FROM the already-
    # weighted velocity, so it inherits w_i's scaling automatically (diff of a
    # linearly-scaled signal is scaled the same way) — no separate case needed.
    def _block(xs, ys, with_accel: bool = False):
        dists = [
            np.sqrt((xs[:, i] - xs[:, j]) ** 2 + (ys[:, i] - ys[:, j]) ** 2)
            for i, j in combinations(range(n_pts), 2)
        ]
        dist_arr = np.column_stack(dists) if dists else None
        if dist_arr is not None and _pair_w.size:
            dist_arr = dist_arr * _pair_w[None, :]
        dx   = np.diff(xs, axis=0, prepend=xs[:1])
        dy   = np.diff(ys, axis=0, prepend=ys[:1])
        disp = np.sqrt(dx ** 2 + dy ** 2) * _w[None, :]
        parts = ([dist_arr] if dist_arr is not None else []) + [disp]
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
    _persist_lags = [5, 10] + ([20, 30] if long_lag_drift else [])
    _lag_parts = []
    for _lag in _persist_lags:
        if n_bins > _lag:
            _lagged = np.vstack([_f_norm[:_lag], _f_norm[:-_lag]])
            _lag_parts.append(np.linalg.norm(_f_norm - _lagged, axis=1, keepdims=True))
        else:
            _lag_parts.append(np.zeros((n_bins, 1)))
    f_persist = np.hstack(_lag_parts)   # (n_bins, 2 or 4)

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


def extract_features_3d(xyz: np.ndarray, fps: float,
                         bodyparts: list = None,
                         long_lag_drift: bool = False,
                         long_scale_bins: bool = False,
                         bodypart_weights: "dict | None" = None) -> np.ndarray:
    """
    3D feature extraction for multi-camera triangulated pose data.

    Parameters
    ----------
    xyz            : (N_frames, N_bp * 3)  — columns ordered [x0,y0,z0, x1,y1,z1, …]
    fps            : recording frame rate
    bodyparts      : bodypart name list (not used for normalisation; retained for
                     API parity with extract_features_v2)
    long_lag_drift : add 20-bin (2 s) and 30-bin (3 s) lag offsets on top of the
                     default 5- and 10-bin offsets; primary signal for sustained
                     states such as freezing and guarding (default off)
    long_scale_bins: add 500-ms and 1000-ms coarse temporal bins; tightens UMAP
                     clusters for slow sustained behaviours (default off)

    Returns
    -------
    (N_features, N_bins) at 100-ms temporal resolution — same shape convention
    as extract_features_v2 so BSoidEngine is unchanged.

    Feature blocks
    --------------
    100-ms   : N_bp*(N_bp-1)/2 pairwise 3D distances + N_bp 3D velocities + acceleration
    persist  : temporal lag drift at 5- and 10-bin offsets (+ 20- and 30-bin if long_lag_drift)
    50-ms    : same pairwise distances + velocities at finer scale (fps >= 60 only)
    200-ms   : same pairwise distances + velocities at coarser scale
    500-ms   : ultra-coarse scale (long_scale_bins only; gated on recording length)
    1000-ms  : ultra-coarse scale (long_scale_bins only; gated on recording length)
    """
    from itertools import combinations as _comb
    _ = bodyparts   # reserved for future 3D angular features; kept for API parity

    win100   = max(1, int(round(fps / 10)))
    win_c    = max(1, int(round(fps / 5)))
    # 50ms = 1.5 frames at 30fps: same gate as the 2D path — only meaningful at 60fps+
    # where it resolves genuine fast motion (flinches, escapes, rapid reorientations).
    use_fine_scale = fps >= 60
    win_fine = max(1, int(round(fps / 20)))  # ~50 ms at 60fps+
    win_500  = max(1, int(round(fps / 2)))   # 500 ms
    win_1000 = max(1, int(round(fps)))       # 1000 ms

    n_f, n_xyz = xyz.shape
    n_pts = n_xyz // 3
    n_bins = n_f // win100

    if n_bins < 2:
        raise ValueError(
            f"Recording too short: {n_f} frames @ {fps} fps "
            f"-> {n_bins} 100-ms bins (need >= {2 * win100} frames).")

    # ── Optional per-body-region feature weighting (issue 1b) ─────────────────
    # Same sqrt(wi*wj) / wi convention as extract_features_v2; w=1 everywhere
    # (default) is bit-identical to pre-weighting output.
    _w3d = _build_bodypart_weight_vector(bodyparts, bodypart_weights, n_pts)
    _pair_idx3d = list(_comb(range(n_pts), 2))
    _pair_w3d = (np.sqrt(_w3d[[i for i, j in _pair_idx3d]] * _w3d[[j for i, j in _pair_idx3d]])
                if _pair_idx3d else np.array([]))

    def _xyz_block(arr: np.ndarray, n: int, with_accel: bool = False):
        xs = arr[:, 0::3]; ys = arr[:, 1::3]; zs = arr[:, 2::3]
        dists = [
            np.sqrt((xs[:,i]-xs[:,j])**2 + (ys[:,i]-ys[:,j])**2 + (zs[:,i]-zs[:,j])**2)
            for i, j in _comb(range(n), 2)
        ]
        dist_arr = np.column_stack(dists) if dists else None
        if dist_arr is not None and n == n_pts and _pair_w3d.size:
            dist_arr = dist_arr * _pair_w3d[None, :]
        dx = np.diff(xs, axis=0, prepend=xs[:1])
        dy = np.diff(ys, axis=0, prepend=ys[:1])
        dz = np.diff(zs, axis=0, prepend=zs[:1])
        disp = np.sqrt(dx**2 + dy**2 + dz**2)
        if n == n_pts:
            disp = disp * _w3d[None, :]
        parts = ([dist_arr] if dist_arr is not None else []) + [disp]
        if with_accel:
            k = np.ones(3) / 3.0
            disp_sm = (np.column_stack([
                np.convolve(disp[:, c], k, mode="same")
                for c in range(disp.shape[1])
            ]) if disp.shape[0] > 3 else disp)
            parts.append(np.abs(np.diff(disp_sm, axis=0, prepend=disp_sm[:1])))
        return np.hstack(parts)

    # ── 100-ms bins ──────────────────────────────────────────────────────────
    b100 = xyz[:n_bins * win100].reshape(n_bins, win100, n_xyz).mean(axis=1)
    f100 = _xyz_block(b100, n_pts, with_accel=True)

    # ── Fine scale (50 ms) — only at 60fps+ ──────────────────────────────────
    if use_fine_scale:
        n_bins_fine = n_f // win_fine
        if n_bins_fine >= 2 and win_fine < win100:
            b_fine_raw = (xyz[:n_bins_fine * win_fine]
                          .reshape(n_bins_fine, win_fine, n_xyz).mean(axis=1))
            n_use = min(n_bins, n_bins_fine // 2)
            b_fine = b_fine_raw[:n_use * 2].reshape(n_use, 2, n_xyz).mean(axis=1)
            if n_use < n_bins:
                b_fine = np.vstack([b_fine, np.tile(b_fine[-1:], (n_bins - n_use, 1))])
        else:
            b_fine = b100

    # ── 200-ms coarse scale ───────────────────────────────────────────────────
    n_bc = max(1, n_f // win_c)
    b_c  = xyz[:n_bc * win_c].reshape(n_bc, win_c, n_xyz).mean(axis=1)
    ratio = max(1, win_c // win100)
    b_c_up = np.repeat(b_c, ratio, axis=0)[:n_bins]
    if len(b_c_up) < n_bins:
        b_c_up = np.vstack([b_c_up, np.tile(b_c_up[-1:], (n_bins - len(b_c_up), 1))])
    f_coarse = _xyz_block(b_c_up, n_pts, with_accel=False)

    # ── Temporal lag drift ────────────────────────────────────────────────────
    # Always emit a fixed number of columns so that all sessions produce the
    # same feature width regardless of recording length — required for np.hstack
    # across sessions.  Lags that exceed n_bins get a zero column (no drift
    # measurable; the session is shorter than the lag window).
    _fn = f100 / (np.linalg.norm(f100, axis=1, keepdims=True) + 1e-9)
    persist_lags = [5, 10] + ([20, 30] if long_lag_drift else [])
    _persist_parts = []
    for _lag in persist_lags:
        if n_bins > _lag:
            _lagged = np.vstack([_fn[:_lag], _fn[:-_lag]])
            _persist_parts.append(np.linalg.norm(_fn - _lagged, axis=1, keepdims=True))
        else:
            _persist_parts.append(np.zeros((n_bins, 1)))
    f_persist = np.hstack(_persist_parts)

    # ── 500-ms / 1000-ms ultra-coarse scales (optional) ──────────────────────
    # Always emit both blocks when long_scale_bins=True so that every session
    # produces the same feature width regardless of recording length.
    # Recordings too short for a given window get a zero block — uninformative
    # but dimensionally consistent for np.hstack across sessions.
    _n_coarse = f_coarse.shape[1]   # = n_pts*(n_pts-1)//2 + n_pts
    if long_scale_bins:
        if win_500 >= 2 and (n_f // win_500) >= 2:
            n_b500 = n_f // win_500
            b_500 = xyz[:n_b500 * win_500].reshape(n_b500, win_500, n_xyz).mean(axis=1)
            r500 = max(1, win_500 // win100)
            b_500_up = np.repeat(b_500, r500, axis=0)[:n_bins]
            if len(b_500_up) < n_bins:
                b_500_up = np.vstack([b_500_up, np.tile(b_500_up[-1:], (n_bins - len(b_500_up), 1))])
            f_500 = _xyz_block(b_500_up, n_pts, with_accel=False)
        else:
            f_500 = np.zeros((n_bins, _n_coarse))

        if win_1000 >= 2 and (n_f // win_1000) >= 2:
            n_b1000 = n_f // win_1000
            b_1000 = xyz[:n_b1000 * win_1000].reshape(n_b1000, win_1000, n_xyz).mean(axis=1)
            r1000 = max(1, win_1000 // win100)
            b_1000_up = np.repeat(b_1000, r1000, axis=0)[:n_bins]
            if len(b_1000_up) < n_bins:
                b_1000_up = np.vstack([b_1000_up, np.tile(b_1000_up[-1:], (n_bins - len(b_1000_up), 1))])
            f_1000 = _xyz_block(b_1000_up, n_pts, with_accel=False)
        else:
            f_1000 = np.zeros((n_bins, _n_coarse))

    # ── Concatenate all blocks ────────────────────────────────────────────────
    blocks = [f100, f_persist]
    if use_fine_scale:
        blocks.append(_xyz_block(b_fine[:n_bins], n_pts, with_accel=False))
    blocks.append(f_coarse)
    if long_scale_bins:
        blocks.append(f_500)
        blocks.append(f_1000)
    feats = np.hstack(blocks)
    return feats.T   # (N_features, N_bins)


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

    # Optional PCA pre-reduction: auto-triggers when n_samples/n_features < 5
    # and n_features > 50, preventing UMAP nearest-neighbour graph degradation
    # in high-density feature spaces (curse of dimensionality).
    pca_mode = str(cfg.get("pca_n_components", "auto")).lower().strip()
    n_samp, n_feat = feats_sc_T.shape
    _ratio = n_samp / max(1, n_feat)
    _do_pca = (pca_mode not in ("off", "-1", "0", "")) and (
        pca_mode == "on"
        or (pca_mode == "auto" and _ratio < 5.0 and n_feat > 50)
        or (pca_mode.isdigit() and int(pca_mode) > 1)
    )
    if _do_pca:
        from sklearn.decomposition import PCA as _PCA
        _n_pca = (int(pca_mode) if pca_mode.isdigit() and int(pca_mode) > 1
                  else min(n_feat - 1, max(50, int(n_samp ** 0.75))))
        _n_pca = max(1, min(_n_pca, n_feat - 1, n_samp))
        _pca = _PCA(n_components=_n_pca,
                    random_state=int(cfg.get("umap_random_state", 42)))
        feats_sc_T = _pca.fit_transform(feats_sc_T)
        _var = _pca.explained_variance_ratio_.sum() * 100
        print(f"  [PCA pre-UMAP] {n_feat} → {_n_pca} dims "
              f"({_var:.1f}% variance kept, sample/feature ratio was {_ratio:.1f})")

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
#  ADAPTIVE RESOURCE MANAGEMENT  (cores/RAM-aware parallelism budget for the
#  HDBSCAN-side stages: primary sweep, split_impure_clusters, seed_sweep_stability.
#  UMAP stays forced single-threaded elsewhere for embedding reproducibility --
#  not covered here.)
#

def detect_system_resources() -> dict:
    """Cores + RAM snapshot used to size adaptive parallelism."""
    import psutil
    vm = psutil.virtual_memory()
    return {
        "cpu_count":        os.cpu_count() or 1,
        "total_ram_gb":     vm.total / 1e9,
        "available_ram_gb": vm.available / 1e9,
        "ram_used_pct":     vm.percent,
    }


def compute_adaptive_n_jobs(cfg: dict, log_fn=None) -> int:
    """Core budget for HDBSCAN-side parallel stages.

    Targets `system_resource_target_pct` of logical cores (default 0.65, i.e.
    the 60-70% "ideal sustained" band), hard-capped at `system_resource_cap_pct`
    (default 0.80) regardless of core count.  Shrunk further if RAM is already
    under pressure at call time -- this is the crash-avoidance mechanism: the
    budget only ever shrinks from the target, it never expands past the cap.
    """
    res = detect_system_resources()
    target_pct = float(cfg.get("system_resource_target_pct", 0.65))
    cap_pct    = float(cfg.get("system_resource_cap_pct",    0.80))
    n_jobs = max(1, int(res["cpu_count"] * target_pct))
    n_jobs = min(n_jobs, max(1, int(res["cpu_count"] * cap_pct)))
    # Memory-pressure guard: >75% RAM already used, or <1.5 GB free per
    # planned worker, halves the budget rather than risking OOM under
    # concurrent HDBSCAN fits (each fit builds its own mutual-reachability /
    # condensed-tree working set).
    if res["ram_used_pct"] > 75 or res["available_ram_gb"] < 1.5 * n_jobs:
        n_jobs = max(1, n_jobs // 2)
        if log_fn:
            log_fn(f"  [SYSTEM] RAM pressure detected ({res['ram_used_pct']:.0f}% "
                   f"used, {res['available_ram_gb']:.1f} GB free) — "
                   f"reducing parallel budget to {n_jobs} worker(s).")
    return n_jobs


def resolve_n_jobs(cfg: dict, cfg_key: str, log_fn=None) -> int:
    """Resolve one of the HDBSCAN-side `*_n_jobs` cfg keys to an actual worker
    count.  `-1` (the shared default/"auto" sentinel across `hdbscan_sweep_n_jobs`,
    `hdbscan_split_n_jobs`, `seed_sweep_n_jobs`, `consensus_n_jobs`) means "auto-managed" when
    `auto_resource_management` is on -- resolved via compute_adaptive_n_jobs().
    Any other explicit value (1 = sequential, or a user-pinned positive
    integer) is honoured exactly as given, unchanged from prior behaviour.
    """
    raw = int(cfg.get(cfg_key, -1) or -1)
    if raw == -1 and bool(cfg.get("auto_resource_management", True)):
        try:
            return compute_adaptive_n_jobs(cfg, log_fn=log_fn)
        except Exception:
            # psutil missing, or any other resource-detection failure --
            # fall back to the pre-auto-management behaviour (-1 = literal
            # "all cores" via joblib) rather than crashing the pipeline.
            if log_fn:
                log_fn(f"  [SYSTEM] resource detection failed for {cfg_key} "
                       f"(psutil missing?) — falling back to all-cores.")
            return -1
    return raw if raw != -1 else -1


import contextlib

@contextlib.contextmanager
def _numba_single_thread():
    """Thread-local cap of numba's own parallel-region thread pool to 1.

    Confirmed thread-local (not process-global) via numba.set_num_threads —
    each thread's mask is independent, so this is safe to call concurrently
    from multiple joblib worker threads without racing each other. Restores
    this thread's previous count on exit.

    Why this matters: every outer `*_n_jobs` sweep in this module
    (seed_sweep_stability, split_impure_clusters, consensus_cluster) already
    forces the INNER run_hdbscan()/run_umap() call's own `*_sweep_n_jobs` cfg
    key to 1 so joblib doesn't nest worker pools -- but that only controls
    joblib-level dispatch. UMAP's neighbour search (pynndescent) is
    numba-jitted and, independently of any joblib setting, spins up its own
    thread pool sized to the full logical core count by default (confirmed
    via numba.get_num_threads() == 32 on a 16-core/32-thread box). With N
    outer joblib worker threads each then trying to claim ~all cores for its
    own UMAP fit, the result is N x cores contending threads -- severe
    oversubscription that thrashes instead of parallelising, which is why a
    "parallel" sweep can still run as slow as (or slower than) sequential.
    Scoping numba to 1 thread per worker here makes actual core usage match
    resolve_n_jobs()'s intended budget (n_jobs workers x 1 core each).
    No-op if numba isn't importable.
    """
    prev = None
    try:
        import numba
        prev = numba.get_num_threads()
        numba.set_num_threads(1)
    except Exception:
        pass
    try:
        yield
    finally:
        if prev is not None:
            try:
                numba.set_num_threads(prev)
            except Exception:
                pass


@contextlib.contextmanager
def _blas_single_thread_for_dispatch():
    """Cap BLAS (OpenBLAS/MKL, via threadpoolctl) to 1 thread for the
    duration of an ENTIRE joblib.Parallel(...) dispatch call -- wrap the
    Parallel(...) call itself in this, not each individual worker.

    threadpoolctl's limiter is documented as process-global with "no thread
    level isolation" (unlike numba's per-thread mask, see
    _numba_single_thread above) -- setting/restoring it independently inside
    each of N concurrently-running worker threads would race: one worker
    finishing and restoring the original (full) thread count mid-sweep would
    silently un-cap BLAS for every other still-running worker. A single
    enter/exit around the whole dispatch avoids that race entirely while
    still preventing oversubscription, since BLAS calls inside numpy/sklearn
    steps (PCA, scaling, HDBSCAN's numpy ops) would otherwise each try to
    claim all cores per worker on top of the numba contention above.
    No-op if threadpoolctl isn't importable.
    """
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            yield
    except ImportError:
        yield


def _patch_pynndescent_thread_safety():
    """Force pynndescent's internal leaf-array step to run single-threaded,
    process-wide, unconditionally -- same "safety over marginal speed"
    philosophy as the BLAS/MKL env vars forced at the top of cube.py and
    cube_analyser.py, applied here because this one has no env-var or
    config-flag equivalent to control it.

    Root cause (confirmed via a real crash + faulthandler stack trace, then
    verified by reading pynndescent 0.5.x's own source): UMAP's nearest-
    neighbour search (pynndescent) builds its random-projection forest with
    UMAP's own n_jobs respected (via NNDescent(n_jobs=...) -> make_forest())
    -- but pynndescent.rp_trees.rptree_leaf_array_parallel(), a SEPARATE
    step that runs unconditionally after forest-building on every single
    UMAP fit, HARDCODES joblib.Parallel(n_jobs=-1, ...) with no parameter to
    override it. CUBE's own UMAP(n_jobs=1) has zero effect on this one step
    -- confirmed by reading the call chain (NNDescent.__init__ ->
    rptree_leaf_array -> rptree_leaf_array_parallel, the last of which never
    receives or forwards n_jobs at all).

    This is harmless when a UMAP fit runs standalone (the primary run_umap()
    call in BSoidEngine.run()) -- the burst of up-to-all-core threads is
    short-lived and nothing else is contending for those cores. It becomes
    a real crash when run_umap() is called from an ALREADY-parallel outer
    joblib worker -- which split_impure_clusters()/seed_sweep_stability()/
    consensus_cluster() all do, each dispatching several such workers at
    once via their own *_n_jobs Parallel(...). Every one of those outer
    workers then independently bursts its own full-width (all-CPU-core)
    thread pool for this one pynndescent step AT THE SAME TIME -- unlike
    numba (capped per-worker via _numba_single_thread()) and BLAS (capped
    per-dispatch via _blas_single_thread_for_dispatch()), nothing in this
    module touched pynndescent's own joblib usage, because it isn't reached
    through either of those mechanisms. Confirmed as the actual crash cause
    via a real run's faulthandler dump: the full stack trace terminated in
    rptree_leaf_array_parallel(), called via run_umap() from inside
    _seed_sweep_one_seed(), itself one of seed_sweep_stability()'s parallel
    workers -- dozens of simultaneous per-worker bursts, severe enough to
    trigger a fatal SIGABRT.

    joblib.parallel_config(n_jobs=1) does NOT fix this: it only supplies a
    default for Parallel(...) calls that omit n_jobs, and pynndescent's call
    passes n_jobs=-1 explicitly -- confirmed empirically (an outer
    parallel_config(n_jobs=1) context measurably had no effect on a nested
    Parallel(n_jobs=-1) call's actual thread count). Patching the function
    itself to always dispatch with n_jobs=1 is safe: results are bit-for-
    bit identical, just serialized -- extracting each tree's leaf-index
    array from an already-built tree is cheap, and this is confirmed by
    benchmark to cost no measurable wall-clock time even on a full-size fit
    (this step was never the bottleneck; NN-descent search is). No-op if
    pynndescent isn't installed under this name/layout (future version
    changed the function's location), so a version bump degrades back to
    the pre-patch (nesting-unsafe) behaviour rather than raising.
    """
    try:
        import pynndescent.rp_trees as _rpt
        import joblib as _joblib

        def _patched_leaf_array_parallel(rp_forest):
            _max_leaf_size = np.max([t.leaf_size for t in rp_forest])
            return _joblib.Parallel(n_jobs=1, require="sharedmem")(
                _joblib.delayed(_rpt.get_leaves_from_tree)(t, _max_leaf_size)
                for t in rp_forest)

        _rpt.rptree_leaf_array_parallel = _patched_leaf_array_parallel
    except Exception:
        pass


_patch_pynndescent_thread_safety()


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

    target_n_clusters == 0 (auto mode):
        `hdbscan_selection_mode="floor_soft_cap"` (the default since Aug
        2026): one continuous ranking pass -- a hard floor at
        preferred_clusters_lo (never undershoot if avoidable) plus a soft
        linear penalty (`hdbscan_overshoot_penalty`) above
        preferred_clusters_hi.  Promoted after real 3-group seed-sweep
        testing showed the old rule (below) collapsing 5/8 seeds to a
        catastrophic 3-cluster outcome, vs. 0/8 under this rule (mean ARI
        0.345→0.589).  `hdbscan_selection_mode="legacy"` restores the old
        two-branch rule: prefer solutions whose cluster count falls inside
        [preferred_clusters_lo, preferred_clusters_hi], picking the highest
        DBCV within that range, falling back to the solution closest to the
        preferred range boundary when no in-range candidate exists -- this
        fallback is what produced the discontinuous collapse.

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
    if _anchor == "full":
        import warnings
        warnings.warn(
            "hdbscan_mcs_anchor='full' reproduces pre-v2.1 behaviour where "
            "min_cluster_size is anchored to the full bin count rather than the "
            "UMAP embedding size. When UMAP subsamples this produces systematically "
            "coarser clusters without warning. Use 'embedding' (the default) for "
            "correctly-proportioned cluster granularity.",
            DeprecationWarning, stacklevel=2)
    if _anchor == "full" and n_total is not None:
        ref_n = n_total
    else:
        ref_n = embedding.shape[0]

    # ── User preferences ──────────────────────────────────────────────────────
    target_n = int(cfg.get("target_n_clusters", 0))       # 0 = no specific target
    pref_lo  = int(cfg.get("preferred_clusters_lo", 8))   # auto-mode lower bound
    pref_hi  = int(cfg.get("preferred_clusters_hi", 20))  # auto-mode upper bound

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
    # hdbscan_sweep_n_steps: 40 (default) for the primary/whole-session sweep.
    # split_impure_clusters() overrides this to a coarser value for its local
    # re-clustering sweeps (Aug 2026 perf fix) -- a ~few-hundred-point local
    # subset only needs to distinguish "did this split into 2-3 clean sub-
    # clusters", not fine mcs resolution across the full dynamic range.
    n_steps = max(2, int(cfg.get("hdbscan_sweep_n_steps", 40) or 40))

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

    # ── Diagnostic: near-duplicate embedding points (one log line, primary
    # fit only -- log_fn is None for seed-sweep/split-pass recursive calls,
    # so this never repeats per-seed or per-split). A high fraction here is
    # the leading indicator of DBCV going degenerate / HDBSCAN noise blowing
    # up, and is usually caused by chronically low-confidence bodyparts
    # producing flat-interpolated, near-identical feature vectors across
    # many bins -- see auto_bodypart_weighting (DEFAULTS) for the automatic
    # mitigation this is meant to help diagnose.
    if log_fn:
        try:
            _n_uniq = len(np.unique(np.round(embedding, 6), axis=0))
            _dup_frac = 1.0 - _n_uniq / max(1, embedding.shape[0])
            if _dup_frac > 0.01:
                log_fn(f"  [DIAG] {embedding.shape[0] - _n_uniq}/{embedding.shape[0]} "
                      f"({_dup_frac * 100:.1f}%) near-duplicate embedding points "
                      f"before HDBSCAN — likely flat-interpolated bins from "
                      f"chronically low-confidence bodyparts; watch for DBCV "
                      f"going degenerate / high noise below.")
        except Exception:
            pass

    # ── Break exact coordinate ties before HDBSCAN ───────────────────────────
    # Flat interpolation over long tracking gaps produces identical feature
    # vectors that collapse to the same UMAP coordinates.  Exact duplicates set
    # mutual-reachability distances to zero → DBCV divides by zero → NaN for
    # every candidate.  Jitter at 1e-4 × per-axis std is imperceptible to
    # cluster geometry but eliminates the degeneracy.
    _emb_std = embedding.std(axis=0)
    _emb_std[_emb_std == 0] = 1.0          # guard against zero-variance axes
    embedding = embedding + np.random.default_rng(42).normal(
        0, 1e-4 * _emb_std, embedding.shape
    )

    # ── Sweep: collect every viable candidate ─────────────────────────────────
    # tuple: (score, n_clusters, labels, clf, method)
    # Dispatched via a thread pool sized by resolve_n_jobs() (System Resources,
    # DEFAULTS): threads, not processes -- run_umap()/hdbscan JIT-compile via
    # numba, and process-based (loky) workers compiling that JIT concurrently
    # for the first time hit a real Windows access-violation crash in testing
    # (numba's on-disk JIT cache isn't safe under concurrent cross-process
    # first-compilation). HDBSCAN's Cython core and numba's nopython-mode
    # functions both release the GIL during their heavy numeric work, so a
    # thread pool still parallelizes the actual computation -- same choice,
    # for the same reason, as split_impure_clusters() below. Result selection
    # afterward is order-independent (best-score pick over an unordered list),
    # so parallel completion order cannot change which candidate wins.
    _sweep_n_jobs = resolve_n_jobs(cfg, "hdbscan_sweep_n_jobs", log_fn=log_fn)
    # Oversubscription guard: once the outer sweep itself is running N fits
    # concurrently, each individual fit must not ALSO spawn hdbscan's own
    # internal core_dist_n_jobs threads (default 4) on top of that -- total
    # concurrency must stay bounded by the resolved budget alone.
    _core_dist_n_jobs = 1 if _sweep_n_jobs != 1 else None

    def _fit_one(method, pct):
        mcs = max(2, int(round(0.001 * pct * ref_n)))
        _kwargs = dict(
            prediction_data          = True,
            min_cluster_size         = mcs,
            min_samples              = max(5, mcs // 5),
            metric                   = cfg.get("hdbscan_metric", "euclidean"),
            cluster_selection_method = method,
            # relative_validity_ (DBCV) raises AttributeError without this
            # -- silently swallowed by the getattr() default below, which
            # made DBCV permanently unavailable (every candidate falling
            # through to the "degenerate" branch, unconditionally) rather
            # than only when the density graph is genuinely degenerate.
            gen_min_span_tree        = True,
        )
        if _core_dist_n_jobs is not None:
            _kwargs["core_dist_n_jobs"] = _core_dist_n_jobs
        # Oversubscription guard (thread-count half): mirrors split_impure_
        # clusters()/seed_sweep_stability()/consensus_cluster(), which all cap
        # numba to 1 thread per outer worker for the same reason -- HDBSCAN's
        # numba-jitted core-distance/boruvka code spins up its OWN thread pool
        # (sized to all CPU cores by default) independently of joblib's
        # n_jobs. Without this, each of the _sweep_n_jobs outer worker
        # threads above ALSO launches a full-width numba thread pool,
        # multiplying concurrency to roughly _sweep_n_jobs x core_count
        # threads -- confirmed on a real run: dozens of live threads and a
        # fatal "Aborted" (SIGABRT) crash with no Python traceback, the
        # thread-count analogue of the core_dist_n_jobs guard just above.
        with _numba_single_thread():
            clf = _hdb.HDBSCAN(**_kwargs).fit(embedding)

        n_cl = len(set(clf.labels_)) - (1 if -1 in clf.labels_ else 0)
        if n_cl < 2:
            return None

        score = getattr(clf, "relative_validity_", -np.inf)
        return (score, n_cl, clf.labels_.copy(), clf, method)

    _tasks = [(method, pct) for method in methods for pct in pcts]
    if _sweep_n_jobs != 1 and len(_tasks) > 1:
        try:
            from joblib import Parallel, delayed
            # Oversubscription guard (BLAS half): mirrors split_impure_
            # clusters()/seed_sweep_stability()/consensus_cluster(), which
            # all wrap their own Parallel(...) dispatch the same way -- see
            # _blas_single_thread_for_dispatch()'s docstring for why this has
            # to wrap the WHOLE dispatch call rather than each worker
            # individually (threadpoolctl's limiter is process-global, so
            # per-worker enter/exit would race). Without this, BLAS calls
            # inside each worker's own numpy/sklearn steps (PCA, distance
            # computation) would each try to claim all cores on top of the
            # numba contention _fit_one already guards against above.
            with _blas_single_thread_for_dispatch():
                _raw = Parallel(n_jobs=_sweep_n_jobs, prefer="threads")(
                    delayed(_fit_one)(method, pct) for method, pct in _tasks)
        except Exception:
            _raw = [_fit_one(method, pct) for method, pct in _tasks]
    else:
        _raw = [_fit_one(method, pct) for method, pct in _tasks]
    candidates = [r for r in _raw if r is not None]

    # ── Fallback: sweep produced nothing with ≥ 2 clusters ────────────────────
    if not candidates:
        mcs = max(2, int(round(0.001 * pct_lo * ref_n)))
        clf = _hdb.HDBSCAN(
            prediction_data          = True,
            min_cluster_size         = mcs,
            min_samples              = max(5, mcs // 5),
            metric                   = cfg.get("hdbscan_metric", "euclidean"),
            cluster_selection_method = methods[0],
            gen_min_span_tree        = True,
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
            for (_s, _ncl, _lbls, _clf, _method) in candidates:
                _m = _lbls >= 0
                if _m.sum() < 2 or len(set(_lbls[_m])) < 2:
                    _rescored.append((-1.0, _ncl, _lbls, _clf, _method))
                    continue
                _idx = np.flatnonzero(_m)
                if _idx.size > 5000:
                    _idx = _rng_sil.choice(_idx, 5000, replace=False)
                try:
                    _sil = float(silhouette_score(embedding[_idx], _lbls[_idx]))
                except Exception:
                    _sil = -1.0
                _rescored.append((_sil, _ncl, _lbls, _clf, _method))
            candidates = _rescored
            best_dbcv = max(s for s, *_ in candidates)
        except Exception:
            pass  # sklearn unavailable; keep -inf scores, selection by diversity

    # ── eom/leaf tie-breaking nudge (issue 4b) ────────────────────────────────
    # Scoring chain (documented per the plan): sweep min_cluster_size x
    # {eom, leaf} -> score by DBCV + size-diversity bonus (+ leaf bonus below,
    # ONLY once the condensed-tree merge pass is enabled) -> pick best
    # candidate -> optional split/merge iterative refinement -> rare-cluster
    # pruning.  Leaf-method HDBSCAN tends to fragment a single behaviour into
    # more, smaller, locally-purer sub-clusters than eom; that fragmentation is
    # exactly what merge_similar_clusters is built to safely undo (it mostly
    # occurs at low condensed-tree split-persistence, the merge pass's trigger
    # condition), so once merging is active leaf's extra-split downside is
    # self-correcting and its tighter-local-homogeneity upside stops being
    # penalised.  When hdbscan_merge_thresh == 0 (merge pass off — the
    # default is 0.08, on), the bonus is never applied and eom/leaf
    # selection is BYTE-FOR-BYTE unchanged from pre-issue-4 behaviour.
    _merge_thresh_cfg = float(cfg.get("hdbscan_merge_thresh", 0.0) or 0.0)
    _leaf_bonus = float(cfg.get("hdbscan_leaf_bonus", 0.03))
    if _merge_thresh_cfg > 0 and _leaf_bonus:
        candidates = [
            (s + _leaf_bonus if m == "leaf" else s, n, l, c, m)
            for (s, n, l, c, m) in candidates
        ]
        best_dbcv = max(s for s, *_ in candidates)

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

    # hdbscan_fine_bias: only active once the merge pass is enabled, since
    # merge can safely undo over-fragmentation afterward.  Nudges auto-mode
    # selection toward the finer end of [pref_lo, pref_hi] instead of always
    # settling on the coarsest DBCV peak in range — biasing toward "enough
    # clusters to separate distinct behaviours, let merge consolidate
    # near-duplicates" rather than "fewest clusters that still score well".
    # 0.0 (default when merge is off) is a hard no-op — in_range selection is
    # then byte-for-byte unchanged from pre-issue-4 behaviour.
    #
    # ALSO disabled when DBCV itself is degenerate (dbcv_degenerate, silhouette
    # fallback active): fine_bias's whole premise is "trust the score enough
    # to deliberately push toward a less-favoured-by-score-alone candidate,
    # because merge will safely clean up the extra fragmentation" -- that
    # requires the score to be a meaningful ranking signal in the first
    # place.  When it isn't (degenerate density graph), biasing selection
    # doesn't reliably pick a better candidate, it just adds noise to an
    # already-unreliable ranking.  Confirmed on real data: with fine_bias
    # active under a degenerate-DBCV run, seed-sweep stability got WORSE
    # (mean ARI 0.55 vs 0.71 on a comparable run without this gate, one seed
    # spiking to 28 clusters vs a stable 12-14 range) and noise increased
    # (55.2% vs 50.7%) rather than improving.
    _fine_bias = (float(cfg.get("hdbscan_fine_bias", 0.05))
                  if _merge_thresh_cfg > 0 and not dbcv_degenerate else 0.0)

    def _sel_score(c):
        _score, _n, _labels = c[0], c[1], c[2]
        bonus = _div_bonus * _cluster_cv(_labels)
        if _fine_bias and pref_hi > pref_lo:
            _n_clamped = max(pref_lo, min(pref_hi, _n))
            bonus += _fine_bias * (_n_clamped - pref_lo) / (pref_hi - pref_lo)
        return _score + bonus

    # ── Selection strategy ────────────────────────────────────────────────────
    if target_n > 0:
        # User-guided: pick closest to target with DBCV ≥ dbcv_thresh of best.
        thresh    = best_dbcv * _dbcv_thresh if best_dbcv > 0 else best_dbcv - 0.1
        qualified = [c for c in candidates if c[0] >= thresh] or candidates
        qualified.sort(key=lambda c: (abs(c[1] - target_n), -c[0]))
        chosen = qualified[0]
    elif str(cfg.get("hdbscan_selection_mode", "legacy")).lower().strip() == "floor_soft_cap":
        # Unified, continuous ranking: hard floor at pref_lo, soft linear
        # penalty above pref_hi.  Replaces the legacy in-range/boundary-
        # fallback branch split below, which discontinuously jumps to a
        # structurally different rule when no swept candidate falls in-range.
        floor_ok = [c for c in candidates if c[1] >= pref_lo]
        if not floor_ok:
            chosen = max(candidates, key=lambda c: c[1])
            if log_fn:
                log_fn(f"  [WARN] no sweep candidate reached "
                       f"preferred_clusters_lo={pref_lo}; dataset may lack "
                       f"enough structure/data. Selecting the closest "
                       f"available ({chosen[1]} clusters) — inspect this "
                       f"session's output with extra care.")
        else:
            overshoot_w = float(cfg.get("hdbscan_overshoot_penalty", 0.01))
            def _unified_score(c):
                return _sel_score(c) - overshoot_w * max(0, c[1] - pref_hi)
            chosen = max(floor_ok, key=_unified_score)
        if chosen[0] < _dbcv_thresh * best_dbcv:
            if log_fn:
                log_fn(f"  [WARN] selected solution's {_score_label}="
                       f"{chosen[0]:.3f} is below {_dbcv_thresh:.0%} of the "
                       f"sweep's best ({best_dbcv:.3f}); cluster quality may "
                       f"be weak despite satisfying the count floor.")
    else:
        # Legacy (default): prefer solutions in [pref_lo, pref_hi].
        # Tiebreak with a small cluster-size CV bonus (+ fine_bias above) so
        # solutions containing both brief and sustained clusters, and finer
        # partitions that merge can safely consolidate, are not unfairly
        # penalised.
        in_range = [c for c in candidates if pref_lo <= c[1] <= pref_hi]
        if in_range:
            in_range.sort(key=lambda c: -_sel_score(c))
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

    best_score, _, best_labels, best_clf, _ = chosen
    # Non-selected candidates' clf objects (each retains a condensed tree /
    # minimum spanning tree; up to ~80 coexist in `candidates` by construction)
    # need no explicit release here: they're plain refcounted objects with no
    # reference cycles, so they're freed the instant this function returns and
    # `candidates` goes out of scope -- an explicit gc.collect() would only
    # add a full generation-2 scan on every one of run_hdbscan()'s many nested
    # calls (once per seed-sweep seed, once per split-pass candidate) for no
    # memory benefit.

    return best_clf, best_labels, best_score, _score_label


#
#  CLUSTER CENTROIDS  (shared by issue 1a's clip selection and issue 4's merge pass)
#

def compute_cluster_centroids(embedding: np.ndarray, labels: np.ndarray) -> dict:
    """Mean embedding coordinate per cluster id (excludes noise, label < 0)."""
    centroids: dict = {}
    labels = np.asarray(labels)
    for cid in sorted(set(int(l) for l in labels if l >= 0)):
        centroids[cid] = embedding[labels == cid].mean(axis=0)
    return centroids


def attach_centroid_distance(epochs: "pd.DataFrame", embedding: np.ndarray,
                              labels: np.ndarray, centroids: dict,
                              bin_offset: int, win: int) -> "pd.DataFrame":
    """
    Adds `_centroid_dist` = mean L2 distance (embedding space) from each
    epoch's bins to its own cluster's centroid.  NaN where mapping/centroid is
    unavailable (epoch's cluster has no centroid, or maps outside the
    embedding range).  bin_offset/win follow the same session_bin_ranges.json
    (_sbr) / win100 conventions used everywhere else in BSoidEngine.run().
    """
    epochs = epochs.copy()
    if epochs.empty:
        epochs["_centroid_dist"] = pd.Series(dtype=float)
        return epochs
    n_bins_total = embedding.shape[0]
    win = max(1, int(win))
    dists = []
    for _, row in epochs.iterrows():
        cid = int(row["label"])
        centroid = centroids.get(cid)
        if centroid is None:
            dists.append(np.nan)
            continue
        b0 = bin_offset + int(row["start_frame"]) // win
        b1 = bin_offset + int(row["end_frame"])   // win
        b0 = max(0, min(b0, n_bins_total - 1))
        b1 = max(0, min(b1, n_bins_total - 1))
        if b1 < b0:
            b0, b1 = b1, b0
        seg = embedding[b0:b1 + 1]
        if seg.shape[0] == 0:
            dists.append(np.nan)
            continue
        dists.append(float(np.linalg.norm(seg - centroid[None, :], axis=1).mean()))
    epochs["_centroid_dist"] = dists
    return epochs


def enrich_bouts_from_bin_source(bout_df: "pd.DataFrame", per_bin_source,
                                  bin_offset: int, win: int, agg_fns,
                                  out_col_names=None) -> "pd.DataFrame":
    """
    Generalizes attach_centroid_distance's bout-frame -> per-bin-array slice
    -> aggregate -> append pattern into a reusable join utility (v6 K2 Step
    2), shared by this plan (per-bout kinematic directedness) and
    Environmental_Context_v6_Implementation_Plan.md's Phase 3 (per-bout
    region/object membership). attach_centroid_distance itself is untouched;
    this is a separate, more general sibling, not a refactor of it.

    bout_df: raw *_bout_lengths[_hmm].csv-shaped DataFrame for ONE session --
      columns "B-SOiD labels", "Start time (frames)", "Run lengths" (exact
      B-SOiD GUI schema). end_frame = start_frame + run_len - 1 is
      reconstructed here since bout CSVs don't carry it natively (unlike
      *_epochs.csv, which is a filtered subset -- this utility works directly
      off the unfiltered bout table).
    per_bin_source: either a single np.ndarray of shape (n_bins_total, ...)
      indexed by GLOBAL bin id (bin_offset + local bin, matching
      session_bin_ranges.json's convention), or a dict {name: np.ndarray} for
      the multi-column case (e.g. multiple region-membership channels).
    bin_offset, win: this session's slice into the global per-bin array(s),
      following the same session_bin_ranges.json (_sbr) / win100 convention
      used everywhere else in BSoidEngine.run() -- mirrors
      attach_centroid_distance's own b0/b1 lookup and clamping exactly.
    agg_fns: a single callable(segment: np.ndarray) -> scalar (paired with a
      single per_bin_source array), or a dict {name: callable} matching
      per_bin_source's keys when per_bin_source is a dict. Aggregator choice
      (mean for continuous values, mode for categorical/region-membership,
      circular mean for angles) is the caller's responsibility.
    out_col_names: optional single str or {name: str} renaming the appended
      column(s); defaults to per_bin_source's dict key(s), or "value" for the
      single-array case.

    Returns a copy of bout_df with the new column(s) appended, one row per
    input bout row in the same order. NaN where a bout's bin range maps
    outside per_bin_source, the per-bin slice is empty, or the aggregator
    raises on that slice.
    """
    out = bout_df.copy()

    multi   = isinstance(per_bin_source, dict)
    sources = per_bin_source if multi else {"value": per_bin_source}
    aggs    = agg_fns if multi else {"value": agg_fns}
    if out_col_names is None:
        names = {k: k for k in sources}
    elif isinstance(out_col_names, dict):
        names = out_col_names
    else:
        names = {"value": out_col_names}

    if out.empty:
        for key in sources:
            out[names.get(key, key)] = pd.Series(dtype=float)
        return out

    win = max(1, int(win))
    start_frames = out["Start time (frames)"].astype(int).to_numpy()
    run_lens     = out["Run lengths"].astype(int).to_numpy()
    end_frames   = start_frames + run_lens - 1

    for key, arr in sources.items():
        agg_fn = aggs[key]
        n_bins_total = arr.shape[0]
        col_vals = []
        for sf, ef in zip(start_frames, end_frames):
            b0 = bin_offset + int(sf) // win
            b1 = bin_offset + int(ef) // win
            b0 = max(0, min(b0, n_bins_total - 1))
            b1 = max(0, min(b1, n_bins_total - 1))
            if b1 < b0:
                b0, b1 = b1, b0
            seg = arr[b0:b1 + 1]
            if seg.shape[0] == 0:
                col_vals.append(np.nan)
                continue
            try:
                col_vals.append(agg_fn(seg))
            except Exception:
                col_vals.append(np.nan)
        out[names.get(key, key)] = col_vals

    return out


#
#  HIERARCHICAL / CONSENSUS REFINEMENT  (issue 4 — bidirectional split + merge)
#

def merge_similar_clusters(hdb_clf, labels: np.ndarray, embedding: np.ndarray,
                            merge_thresh: float = 0.0, log_fn=None) -> np.ndarray:
    """
    Lightweight post-pass that merges sibling HDBSCAN clusters whose
    condensed-tree split persistence is below `merge_thresh` (a FRACTION of
    the tree's max lambda_val, e.g. 0.05 = only barely separated), confirmed
    by centroid distance in embedding space so tree-adjacent-but-genuinely-
    distinct behaviours are not merged just because of tree topology.

    merge_thresh <= 0 (default) is a hard no-op — returns `labels` unchanged.

    Since every point in a given cluster shares an identical condensed-tree
    ancestor chain up to that cluster's own branch point, the split lambda
    between two clusters can be found EXACTLY from a single representative
    point per cluster (no sampling/approximation needed): walk each
    representative's ancestor chain to the root and find the lowest common
    ancestor; the lambda_val of the child edge just below that ancestor (on
    the branch with the higher lambda, i.e. the most recent shared split) is
    the two clusters' split persistence.
    """
    labels = np.asarray(labels).copy()
    if merge_thresh is None or merge_thresh <= 0:
        return labels
    try:
        df = hdb_clf.condensed_tree_.to_pandas()
    except Exception:
        return labels
    if df.empty:
        return labels

    child_parent = {int(r.child): int(r.parent) for r in df.itertuples(index=False)}
    child_lambda = {int(r.child): float(r.lambda_val) for r in df.itertuples(index=False)}
    max_lambda = float(df["lambda_val"].max()) or 1.0

    def ancestor_chain(node):
        chain = [node]
        cur = node
        seen = {node}
        while cur in child_parent:
            cur = child_parent[cur]
            if cur in seen:
                break
            seen.add(cur)
            chain.append(cur)
        return chain

    def split_lambda(p1, p2):
        c1 = ancestor_chain(p1)
        c2 = ancestor_chain(p2)
        s2 = set(c2)
        for i1, node in enumerate(c1):
            if node in s2:
                lam1 = child_lambda.get(c1[max(0, i1 - 1)], max_lambda)
                i2 = c2.index(node)
                lam2 = child_lambda.get(c2[max(0, i2 - 1)], max_lambda)
                return max(lam1, lam2)
        return 0.0

    cluster_ids = sorted(int(c) for c in set(labels) if c >= 0)
    if len(cluster_ids) < 2:
        return labels

    centroids = compute_cluster_centroids(embedding, labels)
    import itertools
    pair_dists = {(a, b): float(np.linalg.norm(centroids[a] - centroids[b]))
                  for a, b in itertools.combinations(cluster_ids, 2)}
    # Data-driven confirmation threshold: below-median inter-cluster centroid
    # distance (mirrors the "adapted to the data" spirit already used
    # elsewhere, e.g. hdbscan_dbcv_thresh) rather than a fixed absolute value.
    dist_confirm_thresh = float(np.median(list(pair_dists.values()))) if pair_dists else np.inf

    reps = {c: int(np.flatnonzero(labels == c)[0]) for c in cluster_ids}

    parent = {c: c for c in cluster_ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    merges_logged = []
    for a, b in itertools.combinations(cluster_ids, 2):
        lam = split_lambda(reps[a], reps[b])
        frac = lam / max_lambda if max_lambda > 0 else 0.0
        if frac <= merge_thresh and pair_dists[(a, b)] <= dist_confirm_thresh:
            union(a, b)
            merges_logged.append((a, b, frac, pair_dists[(a, b)]))

    groups: dict = {}
    for c in cluster_ids:
        groups.setdefault(find(c), []).append(c)
    real_groups = [g for g in groups.values() if len(g) > 1]
    if not real_groups:
        return labels

    new_labels = labels.copy()
    remap = {}
    for group in real_groups:
        target = min(group)
        for c in group:
            remap[c] = target
    for old, new in remap.items():
        new_labels[labels == old] = new

    # Renumber remaining clusters contiguously (same convention as
    # rare-cluster pruning) so downstream consumers keyed on a compact 0..N-1
    # id range stay consistent.
    remaining = sorted(set(int(l) for l in new_labels if l >= 0))
    renumber = {old: i for i, old in enumerate(remaining)}
    final_labels = new_labels.copy()
    for old, new in renumber.items():
        final_labels[new_labels == old] = new

    if log_fn and merges_logged:
        for a, b, frac, dist in merges_logged:
            log_fn(f"  [merge] cluster #{a} + #{b} — split persistence "
                   f"{frac:.3f} <= {merge_thresh:.3f} of max, centroid dist "
                   f"{dist:.3f} <= median {dist_confirm_thresh:.3f} -> merged")
        log_fn(f"  [merge] {len(real_groups)} merge group(s), "
               f"{len(cluster_ids)} -> {len(remaining)} clusters")

    return final_labels


def _mean_silhouette_per_cluster(X: np.ndarray, labels: np.ndarray,
                                  subsample: int = 10_000):
    """
    Shared helper (used by plot_cluster_validity AND split_impure_clusters so
    the silhouette computation is implemented once): per-bin silhouette score
    (sklearn.metrics.silhouette_samples, subsampled the same way
    validate_clustering already does) plus the per-cluster mean.

    X : (n_samples, n_dims) — the SAME space validate_clustering already uses
        (the UMAP embedding), not the raw/scaled feature matrix, so this
        diagnostic's numbers are directly comparable to the existing
        validate_clustering silhouette gate.

    Returns (sil_full, cluster_means):
        sil_full      : (n_samples,) float array, NaN for noise/unsampled bins
        cluster_means : {cluster_id: mean_silhouette}
    """
    from sklearn.metrics import silhouette_samples
    n = X.shape[0]
    sil_full = np.full(n, np.nan)
    labels = np.asarray(labels)
    mask = labels >= 0
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return sil_full, {}
    idx = np.flatnonzero(mask)
    if idx.size > subsample:
        idx = np.random.default_rng(42).choice(idx, subsample, replace=False)
    try:
        vals = silhouette_samples(X[idx], labels[idx])
    except Exception:
        return sil_full, {}
    sil_full[idx] = vals
    means = {}
    for c in sorted(set(labels[idx])):
        m = labels[idx] == c
        if m.any():
            means[int(c)] = float(np.nanmean(vals[m]))
    return sil_full, means


def split_impure_clusters(feats_sc: np.ndarray, embedding: np.ndarray,
                           labels: np.ndarray, split_silhouette_thresh,
                           cfg: dict, log_fn=None) -> np.ndarray:
    """
    Locally re-clusters any cluster whose MEAN per-bin silhouette falls below
    split_silhouette_thresh — the "impure/heterogeneous cluster" side of issue
    4's bidirectional refinement.  For each candidate cluster, re-runs
    run_umap + run_hdbscan restricted to just that cluster's rows of feats_sc.
    The split is accepted ONLY when a real, stable multi-cluster local result
    is found (finite DBCV > 0, >= 2 sub-clusters) — never forced.

    split_silhouette_thresh in (None, 0) is a hard no-op — returns `labels`
    unchanged (matches the "off by default" DEFAULTS key).

    Perf (Aug 2026): each candidate's local run_hdbscan() used to run the
    full production 40-step x 2-method sweep -- fine for the primary
    whole-session fit, but combinatorially expensive here since it repeats
    per candidate, per refinement iteration, per seed in any seed-sweep/
    consensus loop. Confirmed on a real 21-session dataset: one seed with an
    unusually fragmented raw partition took 25+ minutes in this function
    alone (vs 1-2 min for every other seed). Three independent mitigations,
    all opt-out via cfg for exact reproducibility of pre-fix behavior:
      1. Candidates are filtered to a stricter silhouette cutoff first
         (hdbscan_split_candidate_cutoff, default split_silhouette_thresh/2)
         -- only the worst-of-the-worst clusters attempt a split by default.
         If that cutoff leaves none (too strict for this partition), or still
         leaves more than hdbscan_split_max_candidates (default 10), falls
         back to the worst-N candidates by silhouette -- a hard ceiling on
         candidate count regardless of how fragmented a given partition is.
      2. Each candidate's local sweep uses hdbscan_split_sweep_n_steps
         (default 12, vs 40 for the primary fit) and a single method (eom
         only) -- a local few-hundred-point subset only needs to distinguish
         "did this split into hdbscan_split_max_subclusters clean pieces",
         not fine mcs resolution across the full dynamic range.
      3. Candidates are independent (disjoint point sets) and are attempted
         in parallel via joblib (hdbscan_split_n_jobs, default -1 = all
         cores). Sub-cluster id allocation still happens sequentially AFTER
         all results are collected, in the original candidate order, so
         results are deterministic regardless of worker scheduling.
    """
    labels = np.asarray(labels).copy()
    if not split_silhouette_thresh:
        return labels
    _, means = _mean_silhouette_per_cluster(embedding, labels)
    candidates = [c for c, m in means.items() if m < split_silhouette_thresh]
    if not candidates:
        return labels

    # ── Bound worst-case candidate count (mitigation 1) ────────────────────
    _max_candidates = int(cfg.get("hdbscan_split_max_candidates", 10) or 10)
    _cutoff_cfg = float(cfg.get("hdbscan_split_candidate_cutoff", 0) or 0)
    _strict_cutoff = _cutoff_cfg if _cutoff_cfg > 0 else split_silhouette_thresh * 0.5
    _strict = [c for c in candidates if means[c] < _strict_cutoff]
    if _strict:
        candidates = _strict
    if len(candidates) > _max_candidates:
        candidates = sorted(candidates, key=lambda c: means[c])[:_max_candidates]

    new_labels = labels.copy()
    next_id = (int(labels.max()) + 1) if labels.size and labels.max() >= 0 else 0
    # Minimum local sample size for a trustworthy split candidate.  The old
    # flat floor of 20 points was calibrated for a much lower-dimensional
    # feature space; it does not scale with how many features the pipeline
    # is actually extracting (589-900+ dims is routine once body-region/
    # angular features and more bodyparts are involved).  run_umap's own
    # auto-PCA trigger treats a sample/feature ratio below 5 as too thin to
    # trust (curse of dimensionality), but PCA has a floor of 50 components
    # -- so even WITH PCA, a local subset well under ~5x that floor (250
    # points) is still in the same degraded regime.  Observed on real data:
    # local splits attempted on 90-300 point subsets of a 915-feature space
    # (ratio 0.1-1.1 even after PCA), producing untrustworthy sub-clusters.
    _min_local_pts = int(cfg.get("hdbscan_split_min_points", 250))
    _base_nn  = int(cfg.get("umap_n_neighbors", 15) or 15)
    _max_sub  = int(cfg.get("hdbscan_split_max_subclusters", 3))
    _merge_thresh    = float(cfg.get("hdbscan_merge_thresh", 0.0) or 0.0)
    _split_n_steps   = int(cfg.get("hdbscan_split_sweep_n_steps", 12) or 12)

    # ── Precompute each candidate's local inputs (cheap, sequential) ───────
    # Only the small per-candidate slice (not the full feats_sc/labels) is
    # shipped to worker processes below -- keeps parallel-dispatch
    # serialization cost proportional to candidate size, not dataset size.
    tasks = []  # (cid, idx, sub_feats_T, local_cfg)
    for cid in candidates:
        idx = np.flatnonzero(labels == cid)
        if idx.size < _min_local_pts:   # too few points for a meaningful local re-embedding
            continue
        sub_feats_T = feats_sc[:, idx].T   # (n_sub, n_feat)
        local_cfg = dict(cfg)
        local_cfg["umap_n_neighbors"] = max(5, min(_base_nn, idx.size // 3))
        # preferred_clusters_lo/hi (default 8-30) and hdbscan_fine_bias are
        # calibrated for selecting a cluster count across the WHOLE session
        # -- both are wrong for locally re-clustering a single impure
        # cluster's idx.size points.  Inheriting them from cfg unmodified
        # made fine_bias (active whenever merge_thresh>0, the default)
        # systematically push local selection toward the TOP of the *global*
        # preferred range regardless of local scale, producing wild
        # over-fragmentation in practice (a single cluster split into 29-30
        # tiny fragments in one pass on a real run). A split should resolve
        # a handful of distinct sub-behaviours, not fragment extensively --
        # cap the local target range and disable fine_bias/leaf_bonus.
        local_cfg["preferred_clusters_lo"] = 2
        local_cfg["preferred_clusters_hi"] = _max_sub
        local_cfg["hdbscan_fine_bias"]     = 0.0
        local_cfg["hdbscan_leaf_bonus"]    = 0.0
        local_cfg["hdbscan_method"]        = "eom"
        local_cfg["hdbscan_sweep_n_steps"] = _split_n_steps
        # Force the LOCAL sweep sequential: hdbscan_split_n_jobs below already
        # parallelizes across candidates, so letting each candidate's own
        # run_hdbscan() also spawn a thread pool would nest parallelism and
        # oversubscribe past the resolved budget.
        local_cfg["hdbscan_sweep_n_jobs"] = 1
        tasks.append((cid, idx, sub_feats_T, local_cfg))

    if not tasks:
        return new_labels

    def _attempt_split(sub_feats_T, local_cfg, n_total):
        try:
            with _numba_single_thread():
                _, sub_embedding = run_umap(sub_feats_T, local_cfg)
                sub_clf, sub_labels, sub_score, _ = run_hdbscan(
                    sub_embedding, local_cfg, n_total=n_total)
        except Exception:
            return None
        n_sub_cl = len(set(sub_labels[sub_labels >= 0]))
        if n_sub_cl < 2 or n_sub_cl > _max_sub or not np.isfinite(sub_score) or sub_score <= 0:
            return None   # no stable local split found — leave this cluster untouched

        # Local self-merge, using sub_clf's OWN condensed tree (not the
        # outer/global hdb_clf). The iterative refinement loop's global
        # merge_similar_clusters pass can only evaluate persistence for
        # cluster ids present in the ORIGINAL global hdb_clf's condensed
        # tree -- the brand-new sub-cluster ids created here don't exist
        # there, so the global merge pass has no valid signal for them and
        # can never consolidate them, no matter how weakly separated they
        # are. sub_clf's tree DOES meaningfully describe these new ids since
        # they came directly from this fit, so self-consolidate here, before
        # they are ever written into the global label array.
        if _merge_thresh > 0:
            sub_labels = merge_similar_clusters(
                sub_clf, sub_labels, sub_embedding,
                merge_thresh=_merge_thresh, log_fn=None)
            n_sub_cl = len(set(sub_labels[sub_labels >= 0]))
            if n_sub_cl < 2:
                return None   # self-merge collapsed the split back to one cluster
        return sub_labels, sub_score, n_sub_cl

    _n_jobs = resolve_n_jobs(cfg, "hdbscan_split_n_jobs", log_fn=log_fn)
    if _n_jobs != 1 and len(tasks) > 1:
        try:
            from joblib import Parallel, delayed
            # Threads, not processes: run_umap() JIT-compiles via numba, and
            # process-based (loky) workers compiling that JIT concurrently
            # for the first time hit a real Windows access-violation crash in
            # testing (numba's on-disk JIT cache isn't safe under concurrent
            # cross-process first-compilation). HDBSCAN's Cython core and
            # numba's nopython-mode functions both release the GIL during
            # their heavy numeric work, so a thread pool still parallelizes
            # the actual computation -- and avoids process-pickling overhead,
            # the numba cache race, and Windows' multiprocessing requirement
            # that every CALLER script guard its entry point with
            # `if __name__ == "__main__":` (not guaranteed for every context
            # this function is called from).
            with _blas_single_thread_for_dispatch():
                raw_results = Parallel(n_jobs=_n_jobs, prefer="threads")(
                    delayed(_attempt_split)(t[2], t[3], t[1].size) for t in tasks)
        except Exception:
            raw_results = [_attempt_split(t[2], t[3], t[1].size) for t in tasks]
    else:
        raw_results = [_attempt_split(t[2], t[3], t[1].size) for t in tasks]

    # ── Sequential id assignment (deterministic regardless of worker order) ─
    for (cid, idx, _, _), res in zip(tasks, raw_results):
        if res is None:
            continue
        sub_labels, sub_score, n_sub_cl = res
        sub_ids_sorted = sorted(set(sub_labels[sub_labels >= 0]))
        first_sub = sub_ids_sorted[0]
        for sub_c in sub_ids_sorted:
            m = sub_labels == sub_c
            if sub_c == first_sub:
                new_labels[idx[m]] = cid          # first sub-cluster keeps the original id
            else:
                new_labels[idx[m]] = next_id
                next_id += 1
        noise_m = sub_labels < 0                  # local noise stays noise — never forced
        new_labels[idx[noise_m]] = -1

        if log_fn:
            log_fn(f"  [split] cluster #{cid} (mean silhouette "
                   f"{means[cid]:.3f} < {split_silhouette_thresh}) -> "
                   f"{n_sub_cl} sub-cluster(s) (local DBCV={sub_score:.3f})")

    return new_labels


def refine_clusters_iterative(feats_sc: np.ndarray, embedding: np.ndarray,
                               labels: np.ndarray, clf, cfg: dict,
                               log_fn=None) -> np.ndarray:
    """
    Iterative split -> merge refinement loop (issue 4, bidirectional):
    split_impure_clusters then merge_similar_clusters, repeated up to
    cfg['recluster_max_iterations'] times, stopping early once an iteration
    makes no changes.

    Hard no-op (returns `labels` unchanged, no logging) when
    hdbscan_split_silhouette_thresh is falsy AND hdbscan_merge_thresh <= 0,
    or recluster_max_iterations <= 0 -- matches the "off" DEFAULTS exactly,
    same gate as before this was extracted into a shared helper.

    Shared by BSoidEngine.run() (the primary partition, normally called with
    log_fn=self._log) and seed_sweep_stability() (per-seed, normally called
    with log_fn=None to avoid per-seed log spam) so per-seed cluster-count /
    ARI stability reflects the SAME refined partition users actually get on
    the primary seed, not just the pre-refinement HDBSCAN candidate.
    """
    labels = np.asarray(labels).copy()
    split_thresh = cfg.get("hdbscan_split_silhouette_thresh")
    merge_thresh = float(cfg.get("hdbscan_merge_thresh", 0.0) or 0.0)
    max_iter     = int(cfg.get("recluster_max_iterations", 2) or 0)
    if not ((split_thresh or merge_thresh > 0) and max_iter > 0):
        return labels

    if log_fn:
        log_fn(f"\n[5b/7]  Iterative split/merge refinement "
               f"(up to {max_iter} iteration(s))...")
    for _it in range(max_iter):
        before = labels.copy()
        labels = split_impure_clusters(feats_sc, embedding, labels,
                                        split_thresh, cfg, log_fn=log_fn)
        labels = merge_similar_clusters(clf, labels, embedding,
                                         merge_thresh=merge_thresh, log_fn=log_fn)
        if np.array_equal(before, labels):
            if log_fn:
                log_fn(f"  [refine] iteration {_it+1}: no changes — converged")
            break
        if log_fn:
            log_fn(f"  [refine] iteration {_it+1}: "
                   f"{len(set(before[before >= 0]))} -> "
                   f"{len(set(labels[labels >= 0]))} clusters")
    return labels


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
    _min_class_n = int(np.min(np.bincount(y)))
    k = min(int(cfg.get("cv_folds", 5)), n_cl, _min_class_n)
    if k < 2:
        scores = np.array([clf.score(X, y)])
    else:
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
                   angular_fallback: bool = True,
                   is_3d: bool = False,
                   long_lag_drift: bool = False,
                   long_scale_bins: bool = False,
                   bodypart_weights: "dict | None" = None,
                   ll: "np.ndarray | None" = None,
                   visibility_features_enabled: bool = True,
                   visibility_adaptive_pct: float = 10,
                   likelihood_thresh: float = 0.3,
                   return_proba: bool = False):
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
    is_3d          : when True, xy_smooth has shape (N_frames, N_bp*3) and
        extract_features_3d is used instead of extract_features_v2.
    bodypart_weights, ll, visibility_features_enabled, visibility_adaptive_pct,
    likelihood_thresh : MUST be passed identically to what was used at
        training-time feature extraction (BSoidEngine.run()) — a mismatch here
        silently desyncs the MLP's expected feature layout from what inference
        produces, corrupting every prediction.  ll=None (default) skips the
        visibility block entirely, matching visibility_features_enabled=False.
    return_proba : False (default) — return type and behavior are BYTE-IDENTICAL
        to before this parameter existed: a single (n_frames,) frame-expanded
        hard-label int array. This is the primary backward-compatibility
        guarantee for every existing caller (including
        BSoidEngine.predict_from_saved_model, which never passes this arg).
        True (B.1/B.2, Aug 2026) — additionally, ALWAYS calls
        mlp_model.predict_proba() (not gated behind min_confidence > 0, unlike
        the min_confidence block above) and returns a 3-tuple
        (frame_labels, bin_labels, bin_proba) instead: frame_labels is the
        same array as the False case; bin_labels is the pre-expansion
        per-bin hard-label array (length n_bins); bin_proba is the per-bin,
        per-class probability matrix, shape (n_bins, n_classes), each row
        summing to 1.0.
    """
    if is_3d:
        feats = extract_features_3d(xy_smooth, fps, bodyparts,
                                    long_lag_drift=long_lag_drift,
                                    long_scale_bins=long_scale_bins,
                                    bodypart_weights=bodypart_weights)
    else:
        feats  = extract_features_v2(xy_smooth, fps, bodyparts,
                                      body_normalise=body_normalise,
                                      angular_fallback=angular_fallback,
                                      long_lag_drift=long_lag_drift,
                                      bodypart_weights=bodypart_weights)   # (n_feat, n_bins)
    if visibility_features_enabled and ll is not None:
        _vis = compute_session_visibility_block(
            ll, bodyparts, fps, likelihood_thresh, visibility_adaptive_pct)
        feats = _append_visibility_block(feats, _vis)
    scaled = scaler.transform(feats.T)                        # (n_bins, n_feat)
    if pca_model is not None:
        scaled = pca_model.transform(scaled)                  # (n_bins, n_pca)
    labels = mlp_model.predict(scaled)                        # (n_bins,)
    _bin_proba = None
    if return_proba and hasattr(mlp_model, "predict_proba"):
        _bin_proba = mlp_model.predict_proba(scaled)
    if min_confidence and min_confidence > 0 and hasattr(mlp_model, "predict_proba"):
        try:
            proba = _bin_proba if _bin_proba is not None else mlp_model.predict_proba(scaled)
            labels = np.where(proba.max(axis=1) < float(min_confidence),
                              -1, labels)
        except Exception:
            pass
    _bin_labels = labels.copy()
    win    = max(1, int(round(fps / 10)))
    fl     = np.repeat(labels, win)
    n_orig = xy_smooth.shape[0]
    if len(fl) < n_orig:
        fl = np.pad(fl, (0, n_orig - len(fl)), mode="edge")
    frame_labels = fl[:n_orig].astype(int)
    if not return_proba:
        return frame_labels
    return frame_labels, _bin_labels.astype(int), _bin_proba


# ──────────────────────────────────────────────────────────────────────────────
#  HMM SMOOTHING  (post-hoc Multinomial HMM wrapper for B-SOiD predictions)
# ──────────────────────────────────────────────────────────────────────────────


def _compute_cluster_self_trans(label_sequences: list, n_clusters: int) -> dict:
    """Per-cluster self-transition probability derived from each cluster's own
    mean observed bout length within label_sequences: p_self = 1 - 1/mean_len,
    clamped to [0.5, 0.99]. Falls back to the flat-prior default (0.9) for any
    cluster with no observed bouts in this sequence set. Used by train_hmm()'s
    hmm_transition_prior="per_cluster" mode (B.3, Aug 2026) so a
    fast-flickering cluster's transition-matrix prior doesn't start from the
    same 90% self-transition assumption as a naturally long-bout one.
    """
    bout_lens: dict = {c: [] for c in range(n_clusters)}
    for seq in label_sequences:
        arr = np.asarray(seq).astype(int)
        if arr.size == 0:
            continue
        change = np.flatnonzero(np.diff(arr) != 0)
        starts = np.concatenate(([0], change + 1))
        ends   = np.concatenate((change, [arr.size - 1]))
        for s, e in zip(starts, ends):
            lbl = int(arr[s])
            if 0 <= lbl < n_clusters:
                bout_lens[lbl].append(int(e - s + 1))
    p_self = {}
    for c, lens in bout_lens.items():
        if lens:
            mean_len = float(np.mean(lens))
            p = 1.0 - 1.0 / max(mean_len, 1.0001)
            p_self[c] = float(np.clip(p, 0.5, 0.99))
        else:
            p_self[c] = 0.9
    return p_self


def _sanitize_labels_for_hmm(seq: np.ndarray, n_clusters: int) -> np.ndarray:
    """Replace any label outside [0, n_clusters) (e.g. -1 "unclassified" bins
    from predict_labels(..., min_confidence>0), or a leaked turned-away id)
    with the nearest valid neighbour's label via forward-fill then
    backward-fill. CategoricalHMM.fit()/.decode() require every observed
    symbol to be in [0, n_clusters) -- an out-of-range value raises inside
    hmmlearn (caught upstream by a broad except that silently disables HMM
    smoothing) or, if unvalidated, corrupts decoding via negative fancy
    indexing. Filling with the temporal neighbour matches the HMM's own
    self-persistence prior rather than injecting an arbitrary class.
    """
    arr = np.asarray(seq).astype(int)
    valid = (arr >= 0) & (arr < n_clusters)
    if valid.all():
        return arr
    if not valid.any():
        return np.zeros_like(arr)  # entire sequence invalid: arbitrary but harmless
    out = arr.copy()
    valid_idx = np.flatnonzero(valid)
    # forward-fill: each invalid position takes the last valid value at or before it
    fwd = np.searchsorted(valid_idx, np.arange(len(out)), side="right") - 1
    fwd_filled = np.where(fwd >= 0, out[valid_idx[np.clip(fwd, 0, None)]], -1)
    # backward-fill any still-invalid leading positions (before the first valid value)
    bwd = np.searchsorted(valid_idx, np.arange(len(out)), side="left")
    bwd_filled = out[valid_idx[np.clip(bwd, 0, len(valid_idx) - 1)]]
    out[~valid] = np.where(fwd_filled[~valid] >= 0, fwd_filled[~valid], bwd_filled[~valid])
    return out


def train_hmm(label_sequences: list, n_clusters: int,
              n_states: int = None, n_iter: int = 100, log_fn=None,
              transition_prior: str = "global", random_state: int = 42):
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

    transition_prior : "global" (default, current behavior) uses one flat
        90%-self/10%-spread transition-matrix prior for every state, same as
        always. "per_cluster" (B.3, Aug 2026) derives each cluster's own
        self-transition prior from its mean observed bout length in
        label_sequences instead (see _compute_cluster_self_trans) — only
        applies in smoothing-only mode (n_states == n_clusters), where state
        IDs are known to correspond to cluster IDs; silently falls back to
        "global" behavior in macro-state mode, where that correspondence
        doesn't hold.

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

    label_sequences = [_sanitize_labels_for_hmm(s, n_clusters) for s in label_sequences]

    # Build emission matrix BEFORE model construction so we can pass it in.
    # init_params excludes 'e' to prevent hmmlearn from overwriting our matrix.
    if smoothing_mode:
        # Near-diagonal: P(obs=j | state=i) ≈ 0.95 if i==j, 0.05/(k-1) else
        eps = 0.05
        emis = np.full((n_states, n_clusters), eps / max(1, n_clusters - 1))
        np.fill_diagonal(emis, 1.0 - eps)
        emis /= emis.sum(axis=1, keepdims=True)  # normalise (already sums to 1)
        _ip = "s"    # startprob random; transmat set manually below; emission supplied
    else:
        # Macro-state mode: uniform Dirichlet, all states/clusters equally likely
        rng  = np.random.default_rng(42)
        emis = rng.dirichlet(np.ones(n_clusters), size=n_states)
        _ip  = "s"   # startprob random; transmat set manually below

    model = CategoricalHMM(
        n_components=n_states,
        n_iter=n_iter,
        tol=1e-4,
        init_params=_ip,   # 's' only; 'e' and 't' we set manually
        params="ste",      # all params updated during EM
        random_state=random_state,
    )
    # Diagonal Dirichlet prior on the transition matrix: 90% self-transition,
    # 10% spread uniformly.  This anchors Baum-Welch away from degenerate
    # rapid-switching solutions on short recordings (state-flickering).
    # transition_prior="per_cluster" (B.3, Aug 2026) replaces the flat 0.9
    # with each cluster's own bout-duration-derived self-transition prior
    # instead -- smoothing-only mode only, see _compute_cluster_self_trans.
    if smoothing_mode and transition_prior == "per_cluster":
        _p_self = _compute_cluster_self_trans(label_sequences, n_clusters)
        _transmat_init = np.zeros((n_states, n_states))
        for _i in range(n_states):
            _p = _p_self.get(_i, 0.9)
            _transmat_init[_i, :] = (1.0 - _p) / max(1, n_states - 1)
            _transmat_init[_i, _i] = _p
        if log_fn:
            _pvals = list(_p_self.values())
            log_fn(f"  [per-cluster transition prior] self-transition prior "
                   f"range {min(_pvals):.3f}-{max(_pvals):.3f} "
                   f"(mean {np.mean(_pvals):.3f}, flat-prior default was 0.9)")
    else:
        _transmat_init = np.eye(n_states) * 0.9 + 0.1 / n_states
        _transmat_init /= _transmat_init.sum(axis=1, keepdims=True)
    model.transmat_ = _transmat_init
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
    _clean = _sanitize_labels_for_hmm(
        np.asarray(frame_labels), hmm_model.emissionprob_.shape[1])
    _, state_seq = hmm_model.decode(
        _clean.reshape(-1, 1).astype(int), algorithm="viterbi")
    return state_seq.astype(int)


def train_hmm_soft(bin_proba_sequences: list, n_clusters: int,
                    n_states: int = None, n_iter: int = 100, log_fn=None,
                    transition_prior: str = "global",
                    bin_label_sequences: list = None,
                    random_state: int = 42):
    """B.1 (Aug 2026): fit a GaussianHMM on per-bin MLP class-probability
    vectors (predict_labels(..., return_proba=True)'s bin_proba) instead of
    a CategoricalHMM on hard argmax labels (train_hmm's approach). Each
    probability vector lives on the (n_clusters-1)-simplex; GaussianHMM
    models it with a continuous per-state Gaussian emission, so a frame the
    MLP was genuinely uncertain about (a near-uniform proba row) contributes
    less confidently to the learned transition structure than a frame it was
    near-100% sure of -- CategoricalHMM cannot represent this distinction
    since it only ever sees the collapsed argmax label.

    n_states defaults to n_clusters (smoothing-only mode, mirrors train_hmm).
    covariance_type="diag" is used: probability-vector dimensions are
    correlated (they sum to 1) but a full covariance matrix over n_clusters
    dimensions is unnecessary complexity for the cluster counts this
    pipeline typically sees; revisit if this proves too coarse in practice.

    transition_prior : "global" (default, current behavior) uses the same
        flat 90%-self/10%-spread transition-matrix prior train_hmm() uses.
        "per_cluster" (B.3, wired in here Aug 2026 so B.1 and B.3 compose)
        derives each cluster's own self-transition prior the same way
        train_hmm() does (see _compute_cluster_self_trans), from
        bin_label_sequences -- the hard per-bin argmax labels
        predict_labels(..., return_proba=True) already returns alongside
        bin_proba, so no extra inference call is needed to get them. Falls
        back to "global" behavior in macro-state mode or when
        bin_label_sequences isn't provided, same as train_hmm().

    State/cluster alignment: GaussianHMM has no emissionprob_ equivalent to
    align on (train_hmm's Hungarian-assignment step is written specifically
    against a discrete emission matrix). Each state's Gaussian mean
    (model.means_[i], shape (n_clusters,)) is used as the analogous "which
    cluster does this state represent" signal instead -- it should be
    highest at the state's home cluster's coordinate, since that's what a
    confidently-classified bin_proba row for that cluster looks like near a
    one-hot corner of the simplex. Hungarian assignment on means_ (same role
    as train_hmm's emissionprob_-based alignment) then permutes state IDs to
    match cluster IDs.

    Returns a fitted hmmlearn.hmm.GaussianHMM with the same cube_* diagnostic
    attributes train_hmm() attaches (cube_smoothing_mode, cube_aligned,
    cube_emission_diag -- here the mean diagonal mass of the aligned means_
    matrix, the Gaussian analogue of the categorical emission diagnostic).
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise ImportError(
            "hmmlearn is required for HMM smoothing.  "
            "Install it with:  pip install hmmlearn>=0.3.2")
    if n_states is None:
        n_states = n_clusters

    smoothing_mode = (n_states == n_clusters)

    if smoothing_mode:
        # Near-one-hot means: state i's mean vector is mostly mass on
        # cluster i, small epsilon elsewhere -- same anchoring role as
        # train_hmm's near-diagonal emission-matrix initialisation.
        eps = 0.05
        means_init = np.full((n_states, n_clusters), eps / max(1, n_clusters - 1))
        np.fill_diagonal(means_init, 1.0 - eps)
    else:
        rng = np.random.default_rng(42)
        means_init = rng.dirichlet(np.ones(n_clusters), size=n_states)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        tol=1e-4,
        init_params="sc",  # startprob + covars auto-init; transmat/means set manually below
        params="stmc",     # all params updated during EM
        random_state=random_state,
    )
    # Same diagonal Dirichlet transition-matrix prior as train_hmm's default
    # (90% self-transition, 10% spread), UNLESS transition_prior="per_cluster"
    # (B.3) and bin-level hard labels were supplied -- mirrors train_hmm's own
    # per_cluster branch exactly (same helper, same clamping), just wired
    # into the Gaussian-emission path too so B.1 and B.3 compose.
    if (smoothing_mode and transition_prior == "per_cluster"
            and bin_label_sequences):
        _p_self = _compute_cluster_self_trans(bin_label_sequences, n_clusters)
        _transmat_init = np.zeros((n_states, n_states))
        for _i in range(n_states):
            _p = _p_self.get(_i, 0.9)
            _transmat_init[_i, :] = (1.0 - _p) / max(1, n_states - 1)
            _transmat_init[_i, _i] = _p
        if log_fn:
            _pvals = list(_p_self.values())
            log_fn(f"  [per-cluster transition prior] self-transition prior "
                   f"range {min(_pvals):.3f}-{max(_pvals):.3f} "
                   f"(mean {np.mean(_pvals):.3f}, flat-prior default was 0.9)")
    else:
        _transmat_init = np.eye(n_states) * 0.9 + 0.1 / n_states
        _transmat_init /= _transmat_init.sum(axis=1, keepdims=True)
    model.transmat_ = _transmat_init
    model.means_    = means_init

    X       = np.concatenate([np.asarray(p, dtype=float) for p in bin_proba_sequences], axis=0)
    lengths = [len(p) for p in bin_proba_sequences]
    model.fit(X, lengths)

    model.cube_smoothing_mode = bool(smoothing_mode)
    model.cube_aligned        = False
    model.cube_emission_diag  = float("nan")
    if smoothing_mode:
        try:
            from scipy.optimize import linear_sum_assignment
            _, col_ind = linear_sum_assignment(-model.means_)
            perm = np.argsort(col_ind)
            model.startprob_ = model.startprob_[perm]
            model.transmat_  = model.transmat_[np.ix_(perm, perm)]
            model.means_     = model.means_[perm]
            # covars_ (the public property) returns full (n_states, n_dim,
            # n_dim) matrices for convenience even under covariance_type=
            # "diag", but its setter validates strictly against the diag
            # shape (n_states, n_dim) -- permute the actual diag-shaped
            # backing attribute (_covars_) directly instead.
            model._covars_   = model._covars_[perm]
            model.cube_aligned = True
            _diag = float(np.mean(np.diag(model.means_)))
            model.cube_emission_diag = _diag
            if log_fn and _diag < 0.5:
                log_fn(f"  [VALID-WARN] Soft-emission HMM state<->cluster "
                       f"alignment is weak (mean diagonal mean-vector mass "
                       f"{_diag:.2f} < 0.5). Smoothed state IDs may not "
                       f"correspond cleanly to cluster IDs.")
        except ImportError:
            if log_fn:
                log_fn("  [VALID-WARN] scipy unavailable — soft-emission HMM "
                       "states were NOT aligned to cluster IDs (Hungarian "
                       "assignment skipped). Smoothed _hmm labels may not "
                       "match cluster IDs; install scipy for deterministic "
                       "alignment.")

    return model


def decode_hmm_soft(hmm_model, bin_proba: np.ndarray) -> np.ndarray:
    """Viterbi decode a GaussianHMM (B.1) on one session's per-bin
    probability-vector sequence. Mirrors decode_hmm()'s role for the
    categorical path -- returns (n_bins,) int array of HMM state IDs."""
    _, state_seq = hmm_model.decode(
        np.asarray(bin_proba, dtype=float), algorithm="viterbi")
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
                                   n_neighbors: int = 15) -> dict:
    """
    Stage: UMAP embedding quality.
    Warns if trustworthiness score < 0.8 (local neighbourhood not preserved).
    Subsamples to 5000 points for speed.
    n_neighbors=15 matches the publication benchmark specification (Section 3.2).
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


def compute_bout_directedness(bout_df: "pd.DataFrame", xy: np.ndarray, fps: float,
                               min_bout_frames: int = 5) -> "pd.DataFrame":
    """
    Per-bout kinematic directedness metrics for ONE session (v6 K2 Step 3).

    Unlike Step 2's enrich_bouts_from_bin_source (bin-scoped join), this
    computes directly over each bout's own frame range against the
    already-loaded per-frame centroid trajectory -- no bin lookup needed,
    since the computation is naturally bout-scoped.

    bout_df: raw *_bout_lengths[_hmm].csv-shaped bout table for this session
      ("B-SOiD labels", "Start time (frames)", "Run lengths").
    xy: (n_frames, 2) per-frame centroid trajectory (x, y) for this same
      session, derived the same way compute_cluster_kinematics derives its
      centroid (mean of bodypart x / y columns each frame).
    fps: this session's frame rate.
    min_bout_frames: bouts spanning fewer frames than this skip
      straightness_ratio and heading_consistency (NaN) -- both are noisy or
      undefined on a 2-3 frame bout (flagged as a risk in the original design
      doc's verification section). net_displacement_px, path_length_px, and
      mean_speed_px_s remain well-defined at any bout length and are always
      computed.

    Returns a copy of bout_df with five new columns appended:
      net_displacement_px, path_length_px, straightness_ratio,
      mean_speed_px_s, heading_consistency.
    straightness_ratio = net_displacement / path_length (1.0 = perfectly
      straight line, ~0 = meandering/returns near start). heading_consistency
      is the mean resultant length of the bout's unit step-heading vectors
      (1.0 = every step points the same direction, ~0 = directionally
      random) -- both NaN when path_length is 0 (no net motion) or the bout
      is shorter than min_bout_frames.
    net_rotation_index is deferred to v7+ alongside Morris Water Maze (see
    Kinematic_Transition_v6_Implementation_Plan.md Step 3).
    """
    out = bout_df.copy()
    if out.empty:
        for col in ("net_displacement_px", "path_length_px", "straightness_ratio",
                    "mean_speed_px_s", "heading_consistency"):
            out[col] = pd.Series(dtype=float)
        return out

    n_frames_total = xy.shape[0]
    start_frames = out["Start time (frames)"].astype(int).to_numpy()
    run_lens     = out["Run lengths"].astype(int).to_numpy()

    net_disp, path_len, straight, mean_spd, head_con = [], [], [], [], []

    for sf, rl in zip(start_frames, run_lens):
        sf = max(0, min(int(sf), n_frames_total - 1))
        ef = max(0, min(sf + int(rl) - 1, n_frames_total - 1))
        if ef < sf:
            sf, ef = ef, sf
        seg = xy[sf:ef + 1]
        n_pts = seg.shape[0]
        if n_pts < 2:
            net_disp.append(0.0)
            path_len.append(0.0)
            mean_spd.append(0.0)
            straight.append(np.nan)
            head_con.append(np.nan)
            continue

        deltas    = np.diff(seg, axis=0)
        step_dist = np.hypot(deltas[:, 0], deltas[:, 1])
        pl        = float(step_dist.sum())
        nd        = float(np.hypot(seg[-1, 0] - seg[0, 0], seg[-1, 1] - seg[0, 1]))
        dur_s     = n_pts / fps

        net_disp.append(nd)
        path_len.append(pl)
        mean_spd.append(pl / dur_s if dur_s > 0 else 0.0)

        if (ef - sf + 1) < min_bout_frames or pl == 0:
            straight.append(np.nan)
            head_con.append(np.nan)
            continue

        straight.append(nd / pl)
        nz = step_dist > 0
        if nz.any():
            unit     = deltas[nz] / step_dist[nz, None]
            mean_vec = unit.mean(axis=0)
            head_con.append(float(np.hypot(mean_vec[0], mean_vec[1])))
        else:
            head_con.append(np.nan)

    out["net_displacement_px"] = net_disp
    out["path_length_px"]      = path_len
    out["straightness_ratio"]  = straight
    out["mean_speed_px_s"]     = mean_spd
    out["heading_consistency"] = head_con
    return out


def compute_cluster_confidence_profile(vis_feats: np.ndarray, vis_col_names: list,
                                        labels: np.ndarray, out_path: Path,
                                        low_conf_floor: float = 0.40) -> "pd.DataFrame":
    """
    Aggregate the per-bin visibility/occlusion feature block (issue 2,
    compute_visibility_features) by cluster id — modeled on
    compute_cluster_kinematics' aggregation pattern, but operating on the
    already bin-aligned vis_feats/labels (both indexed the same way as
    hdb_labels_all / embedding_save) rather than re-deriving frame-level
    descriptors, since visibility features are natively per-bin.

    Writes cluster_confidence.csv with mean_visibility, mean_frac_low_conf,
    per-region fractions, and a boolean low_confidence_flag — so a cluster
    that is mostly "animal turned away / occluded" is NOT presented to the
    user as if it were a real behaviour.

    vis_feats     : (n_bins, n_vis_cols) — see visibility_feature_names() for
                    column order.
    vis_col_names : column names matching vis_feats' columns.
    labels        : (n_bins,) cluster id per bin (hdb_labels_all convention).
    low_conf_floor: absolute floor mirroring the existing
                    feature_bad_bp_thresh=0.40 convention.
    """
    labels = np.asarray(labels)
    n = min(vis_feats.shape[0], labels.shape[0])
    vis_feats = vis_feats[:n]
    labels    = labels[:n]
    if vis_feats.size == 0 or n == 0:
        df = pd.DataFrame()
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(str(out_path), index=False)
        except Exception:
            pass
        return df

    mean_frac_col = 1 if len(vis_col_names) > 1 else 0
    overall_mean_frac = float(np.nanmean(vis_feats[:, mean_frac_col])) \
        if vis_feats.shape[0] else 0.0
    flag_thresh = max(low_conf_floor, 2.0 * overall_mean_frac)

    rows = []
    for cid in sorted(set(int(l) for l in labels if l >= 0)):
        m = labels == cid
        if not m.any():
            continue
        means = np.nanmean(vis_feats[m], axis=0)
        row = {"cluster_id": cid, "n_bins": int(m.sum())}
        for name, val in zip(vis_col_names, means):
            row[name] = round(float(val), 4)
        _frac = row.get(vis_col_names[mean_frac_col], 0.0) if vis_col_names else 0.0
        row["low_confidence_flag"] = bool(_frac > flag_thresh)
        rows.append(row)

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
    """
    Writes fig to path.  Save failures are NOT swallowed here — they
    propagate after cleanup, so the caller's own try/except (every
    plot_XXX() call site wraps its call individually, exactly so one broken
    plot can't take down an otherwise-successful run) gets the real
    exception and can log an accurate [WARN] with a traceback.

    Previously this caught the exception and only did a bare print() with no
    re-raise, which meant a savefig failure was invisible to every caller:
    the calling code would carry on and log "<name> saved" even though the
    file was never written (observed for cluster_validity.png — the run log
    said "saved" but the file didn't exist on disk).  plt.close(fig) still
    always runs via finally, so a save failure never leaks the figure.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
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


def plot_split_merge_refinement(embedding: np.ndarray, labels_before: np.ndarray,
                                 labels_after: np.ndarray, out_path: Path,
                                 tag: str = "train") -> None:
    """
    Before/after diagnostic for the iterative split/merge refinement pass
    (issue 4).  Left/middle panels: UMAP scatter coloured by cluster id
    before vs. after refinement.  Right panel: a before-cluster x
    after-cluster contingency heatmap, row-normalised so each cell is the
    fraction of a before-cluster's bins that ended up with each
    after-cluster id — several rows collapsing onto one column is a merge
    (over-split near-duplicates consolidated); one row spreading across
    several columns is a split (an impure cluster resolved into its
    constituent behaviours).  Only called when refinement actually changed
    the labels (see call site), so this always shows a real before/after.
    """
    if embedding.shape[1] < 2:
        return
    labels_before = np.asarray(labels_before)
    labels_after  = np.asarray(labels_after)
    uniq_before = np.unique(labels_before[labels_before >= 0])
    uniq_after  = np.unique(labels_after[labels_after >= 0])
    if len(uniq_before) == 0 or len(uniq_after) == 0:
        return

    fig, (ax_b, ax_a, ax_c) = plt.subplots(1, 3, figsize=(20, 7), facecolor=_BG)
    for ax in (ax_b, ax_a, ax_c):
        _dark_ax(ax)

    for u in uniq_before:
        m = labels_before == u
        ax_b.scatter(embedding[m, 0], embedding[m, 1],
                     s=2, alpha=0.5, color=_cmap(int(u)))
    ax_b.set_title(f"Before — {len(uniq_before)} clusters (raw HDBSCAN)",
                   fontsize=11)
    ax_b.set_xlabel("UMAP 1"); ax_b.set_ylabel("UMAP 2")

    for u in uniq_after:
        m = labels_after == u
        ax_a.scatter(embedding[m, 0], embedding[m, 1],
                     s=2, alpha=0.5, color=_cmap(int(u)))
    ax_a.set_title(f"After — {len(uniq_after)} clusters (refined)",
                   fontsize=11)
    ax_a.set_xlabel("UMAP 1"); ax_a.set_ylabel("UMAP 2")

    mat = np.zeros((len(uniq_before), len(uniq_after)))
    for bi, ub in enumerate(uniq_before):
        row_mask = labels_before == ub
        row_n = int(row_mask.sum())
        if row_n == 0:
            continue
        for ai, ua in enumerate(uniq_after):
            mat[bi, ai] = float(np.logical_and(row_mask, labels_after == ua).sum()) / row_n
    im = ax_c.imshow(mat, cmap=plt.cm.viridis, vmin=0, vmax=1, aspect="auto")
    cb = plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=_TICK_COL)
    cb.set_label("Fraction of before-cluster's bins", color=_TICK_COL)
    ax_c.set_xticks(range(len(uniq_after)))
    ax_c.set_xticklabels([f"C{int(u)}" for u in uniq_after], rotation=90, fontsize=7)
    ax_c.set_yticks(range(len(uniq_before)))
    ax_c.set_yticklabels([f"C{int(u)}" for u in uniq_before], fontsize=7)
    ax_c.set_xlabel("After-refinement cluster")
    ax_c.set_ylabel("Before-refinement cluster")
    ax_c.set_title("Cluster remapping", fontsize=11)

    fig.suptitle(f"Split/Merge Refinement Effect  [{tag}]  —  did refinement "
                 "fix fragmentation or impurity?",
                 color=_TEXT_COL, fontsize=13, y=0.98)
    fig.text(0.5, 0.90,
             "Panels 1-2: colours should form tighter, more separated blobs "
             "after refinement   |   Panel 3: rows sharing one column = "
             "merge; one row spread across columns = split",
             ha="center", va="top", color=_TICK_COL, fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.86))
    _savefig(fig, out_path)


def _seed_sweep_one_seed(s: int, feats_sc_T: np.ndarray, cfg: dict, n_total: int,
                          progress_cb=None):
    """Single-seed UMAP+HDBSCAN(+refinement) fit for seed_sweep_stability()'s
    per-seed loop.

    Module-level (not a closure) so it can be dispatched via joblib's
    thread-based pool (T1.P, ARI-stability plan Aug 2026) -- kept as plain
    functions rather than requiring pickling since threads share the parent
    process's memory. Catches its own exceptions and returns None on failure --
    the caller logs the skip, exactly mirroring the sequential loop's existing
    try/except behaviour, so a single seed's failure never propagates and
    takes down the rest of the sweep.

    cfg["_debug_fail_seeds"], if present, is a test-only hook (an iterable of
    seed values to deliberately fail on) used by the crash-isolation test --
    it is never set by production code and has zero effect unless a caller
    explicitly injects it.

    progress_cb(s, result), if given, is called exactly once right before
    returning (result is None on failure) -- lets the parallel dispatch path
    report per-seed completion as it happens instead of going silent for the
    whole sweep's duration (each seed's own UMAP+HDBSCAN(+refinement) fit can
    run for minutes, and with no output in between a stalled seed and a busy
    one look identical from the log). PipelineLogger is lock-protected, so
    calling it from worker threads here is safe.
    """
    try:
        if s in (cfg.get("_debug_fail_seeds") or ()):
            raise RuntimeError(f"[debug] injected failure for seed {s}")
        c2 = dict(cfg); c2["umap_random_state"] = s
        # Force per-seed HDBSCAN work sequential: seed_sweep_n_jobs already
        # parallelizes across seeds, so letting each seed's own run_hdbscan()
        # (below) OR refine_clusters_iterative()'s split pass (which calls
        # split_impure_clusters(), itself parallel via hdbscan_split_n_jobs)
        # also spawn their own thread pools would nest parallelism -- outer
        # seeds x inner workers -- and oversubscribe well past the resolved
        # budget.
        c2["hdbscan_sweep_n_jobs"] = 1
        c2["hdbscan_split_n_jobs"] = 1
        with _numba_single_thread():
            _, emb = run_umap(feats_sc_T, c2)
            clf, lbls, dbcv_score, _ = run_hdbscan(emb, c2, n_total=n_total)
            lbls = refine_clusters_iterative(
                feats_sc_T.T, emb, lbls, clf, c2, log_fn=None)
        lbls = np.asarray(lbls)
        count = len(set(int(x) for x in lbls if x >= 0))
        result = dict(seed=s, labels=lbls, count=count, dbcv=float(dbcv_score))
        if progress_cb is not None:
            try:
                progress_cb(s, result)
            except Exception:
                pass
        return result
    except Exception:
        if progress_cb is not None:
            try:
                progress_cb(s, None)
            except Exception:
                pass
        return None


def seed_sweep_stability(feats_sc_T: np.ndarray, cfg: dict, n_seeds: int,
                          log_fn=None) -> dict:
    """Re-run UMAP+HDBSCAN(+split/merge refinement) over n_seeds random seeds
    to gauge partition stability.

    Internal quality gates (silhouette, DBCV, trustworthiness) measure how tight
    each cluster is, not whether the PARTITION is reproducible.  This sweep
    answers the latter: it reports the cluster-count distribution and the
    pairwise Adjusted Rand Index (ARI) between seeds.  High mean ARI (→1) means
    the clustering is stable; low ARI means cluster identities depend on the
    seed and should be treated with caution.

    Each seed's labels are passed through refine_clusters_iterative() (the
    same split/merge pass BSoidEngine.run() applies to the primary seed) so
    the reported stability reflects the refined partition users actually get,
    not just the pre-refinement HDBSCAN candidate.  A hard no-op when
    refinement is disabled in cfg, so this changes nothing for runs with the
    split/merge pass off.  Refinement's own per-cluster decisions are not
    logged here (log_fn=None passed through) to avoid per-seed log spam —
    only the resulting cluster count is reported, same as before.

    cfg["seed_sweep_n_jobs"] (default -1 = auto-managed, see System Resources;
    T1.P ARI-stability plan Aug 2026): when != 1, dispatches each seed's
    independent UMAP+HDBSCAN(+refinement) fit via joblib.Parallel with a
    thread-based pool (prefer="threads") -- same choice as
    hdbscan_split_n_jobs above, made for the same reason: process-based
    (loky) workers hit a real Windows access-violation crash from concurrent
    numba first-time JIT compilation, and HDBSCAN/numba both release the GIL
    during their heavy numeric work so threads still parallelise the actual
    computation. Per-seed results are collected back in original seed order
    regardless of completion order, so output content is identical to the
    sequential path -- only wall-clock time changes.

    Returns {seeds, counts, ari, mean_ari, dbcv}.  `dbcv` is each seed's DBCV
    (relative_validity_) from its own run_hdbscan() selection, aligned with
    `counts` (successful seeds only, same order). Empty dict if n_seeds < 2
    or the required libraries are missing.
    """
    if n_seeds is None or int(n_seeds) < 2:
        return {}
    try:
        from sklearn.metrics import adjusted_rand_score
    except Exception:
        return {}
    base_seed = int(cfg.get("umap_random_state", 42))
    seeds = [base_seed + i for i in range(int(n_seeds))]
    n_jobs = resolve_n_jobs(cfg, "seed_sweep_n_jobs", log_fn=log_fn)
    all_labels, counts, dbcv_scores = [], [], []
    if n_jobs == 1:
        results = [_seed_sweep_one_seed(s, feats_sc_T, cfg, feats_sc_T.shape[0])
                   for s in seeds]
    else:
        try:
            from joblib import Parallel, delayed
            if log_fn:
                log_fn(f"  [seed-sweep] dispatching {len(seeds)} seeds across "
                       f"n_jobs={n_jobs} worker threads...")
            # Report as each seed's worker thread finishes -- without this the
            # log goes silent for the sweep's entire wall-clock (each seed can
            # take minutes), indistinguishable from a hang. PipelineLogger is
            # lock-protected so calling log_fn from these worker threads is safe.
            _t0 = time.time()
            _progress_lock = _threading.Lock()
            _progress_done = [0]
            def _on_seed_done(s, r):
                if not log_fn:
                    return
                with _progress_lock:
                    _progress_done[0] += 1
                    n_done = _progress_done[0]
                status = f"{r['count']} clusters, DBCV={r['dbcv']:.3f}" if r else "failed"
                log_fn(f"  [seed-sweep] {n_done}/{len(seeds)} done — seed {s}: "
                       f"{status}  ({time.time() - _t0:.0f}s elapsed)")
            # Heartbeat: all seeds are dispatched at once and do similar work,
            # so their completions tend to arrive in a burst near the end
            # rather than spread across the sweep -- _on_seed_done alone can
            # still leave a long silent stretch early on. A periodic "still
            # running" line makes that stretch distinguishable from a hang.
            _hb_stop = _threading.Event()
            def _heartbeat():
                while not _hb_stop.wait(30):
                    with _progress_lock:
                        n_done = _progress_done[0]
                    if log_fn:
                        log_fn(f"  [seed-sweep] still running — "
                               f"{n_done}/{len(seeds)} seeds done, "
                               f"{time.time() - _t0:.0f}s elapsed...")
            _threading.Thread(target=_heartbeat, daemon=True).start()
            try:
                with _blas_single_thread_for_dispatch():
                    results = Parallel(n_jobs=n_jobs, prefer="threads")(
                        delayed(_seed_sweep_one_seed)(
                            s, feats_sc_T, cfg, feats_sc_T.shape[0],
                            progress_cb=_on_seed_done)
                        for s in seeds
                    )
            finally:
                _hb_stop.set()
        except Exception:
            results = [_seed_sweep_one_seed(s, feats_sc_T, cfg, feats_sc_T.shape[0])
                       for s in seeds]
    for s, r in zip(seeds, results):
        if r is None:
            if log_fn:
                log_fn(f"  [seed-sweep] seed {s} failed; skipped")
        else:
            all_labels.append(r["labels"])
            counts.append(r["count"])
            dbcv_scores.append(r.get("dbcv", float("nan")))
            if log_fn:
                log_fn(f"  [seed-sweep] seed {s}: {r['count']} clusters, "
                       f"DBCV={r.get('dbcv', float('nan')):.3f}")
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
    # mean_ari excludes any pair touching a degenerate seed (fewer than
    # seed_sweep_min_valid_clusters real clusters, default 6 -- i.e. that
    # seed's fit collapsed/under-fit rather than finding a trustworthy
    # partition). Comparing a degenerate partition against a real one is
    # guaranteed near-zero ARI regardless of whether the genuine structure is
    # actually seed-stable, so leaving them in systematically drags the mean
    # down and can spuriously trip consensus_auto_threshold. `ari` (the full
    # matrix, degenerate seeds included) and `counts` are returned unfiltered
    # so callers/plots can still see and label which seeds were degenerate.
    _min_valid = int(cfg.get("seed_sweep_min_valid_clusters", 6) or 6)
    _stable = [i for i in range(m) if counts[i] >= _min_valid]
    if len(_stable) >= 2:
        _pairs = [ari[i, j] for a_, i in enumerate(_stable)
                  for j in _stable[a_ + 1:]]
        mean_ari = float(np.nanmean(_pairs)) if _pairs else 1.0
        stable_counts = [counts[i] for i in _stable]
        if len(_stable) < m and log_fn:
            _degenerate_seeds = [seeds[i] for i in range(m) if i not in _stable]
            log_fn(f"  [seed-sweep] excluding degenerate seed(s) "
                   f"{_degenerate_seeds} (< {_min_valid} clusters found) from "
                   f"the mean ARI -- comparing them against real partitions "
                   f"is guaranteed near-zero regardless of true stability.")
    else:
        triu = ari[np.triu_indices(m, 1)]
        mean_ari = float(np.nanmean(triu)) if triu.size else 1.0
        stable_counts = counts
    return dict(seeds=seeds, counts=counts, ari=ari, labels=all_labels,
                mean_ari=mean_ari, stable_counts=stable_counts,
                dbcv=dbcv_scores)


def seed_sweep_stability_bootstrap(feats_good: np.ndarray, cfg: dict, n_seeds: int,
                                    train_frac: float = None, vary_seed: bool = True,
                                    log_fn=None) -> dict:
    """Bootstrap/subsample cluster-stability diagnostic (T1.1, ARI-stability
    plan Aug 2026).  Ben-Hur, Elisseeff & Guyon 2002, "A stability based
    method for discovering structure in clustered data" (subsampling-based
    stability): cluster independent subsamples of the data and measure
    agreement only over the points common to each pair of subsamples (HDBSCAN
    labels are only defined for points actually included in a given fit).

    Unlike seed_sweep_stability() (which fixes the training subsample and
    only varies umap_random_state), this function draws a *fresh*
    train_frac-sized subsample of bin indices per seed from the pre-subsample
    feature pool, so it captures sampling variance as well as (optionally)
    UMAP optimisation variance.  Used to separate the two sources:

    - vary_seed=True  (default): subsample AND umap_random_state both vary
      per seed -> combined variance (the realistic end-to-end picture).
    - vary_seed=False: umap_random_state is held fixed at cfg's base seed;
      only the subsample varies -> isolates sampling-noise variance alone.
    - seed_sweep_stability() is the complementary ablation (subsample fixed,
      only umap_random_state varies) -- it is not duplicated here, call it
      separately for that leg of the comparison.

    This is a standalone offline diagnostic, not wired into BSoidEngine.run()
    -- it introduces no cfg keys and has zero effect on the production
    pipeline by construction.

    Parameters
    ----------
    feats_good : (n_features, n_good_bins)  pre-train_frac-subsample feature
                 pool (i.e. before run()'s own rng.choice subsampling step).
    cfg        : pipeline config dict (used for umap_*/hdbscan_* settings and
                 as the source of the base random_state / default train_frac).
    n_seeds    : number of independent subsample+cluster repeats.
    train_frac : fraction of n_good_bins to draw per seed. None -> cfg's own
                 "train_frac" (default 0.3).
    vary_seed  : see above.
    log_fn     : optional logger callable.

    Returns
    -------
    dict(seeds, counts, ari, mean_ari, pair_overlap_sizes) -- pair_overlap_sizes
    is a {(seed_i, seed_j): overlap_n} dict, new relative to
    seed_sweep_stability()'s return shape.  Empty dict if n_seeds < 2, the
    required libraries are missing, or feats_good is too small.
    """
    if n_seeds is None or int(n_seeds) < 2:
        return {}
    try:
        from sklearn.metrics import adjusted_rand_score
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return {}
    feats_good = np.asarray(feats_good)
    n_good = feats_good.shape[1]
    if n_good < 60:   # need at least ~2x the 30-point overlap floor to be meaningful
        return {}
    base_seed = int(cfg.get("umap_random_state", 42))
    _train_frac = float(train_frac) if train_frac is not None else float(cfg.get("train_frac", 0.3))
    n_samp = max(2, int(round(n_good * _train_frac)))
    n_samp = min(n_samp, n_good)
    seeds = [base_seed + i for i in range(int(n_seeds))]
    all_idx, all_labels, counts = [], [], []
    used_seeds = []
    for s in seeds:
        try:
            rng_i = np.random.default_rng(s)
            idx = rng_i.choice(n_good, n_samp, replace=False)
            feats_sub = feats_good[:, idx]
            c2 = dict(cfg)
            c2["umap_random_state"] = s if vary_seed else base_seed
            scaler = StandardScaler()
            feats_sc = scaler.fit_transform(feats_sub.T)   # (n_samp, n_feat)
            _, emb = run_umap(feats_sc, c2)
            clf, lbls, _, _ = run_hdbscan(emb, c2, n_total=n_good)
            lbls = refine_clusters_iterative(feats_sc, emb, lbls, clf, c2, log_fn=None)
            all_idx.append(idx)
            all_labels.append(np.asarray(lbls))
            counts.append(len(set(int(x) for x in lbls if x >= 0)))
            used_seeds.append(s)
            if log_fn:
                log_fn(f"  [bootstrap-sweep] seed {s}: {counts[-1]} clusters "
                       f"(n_samp={n_samp}, vary_seed={vary_seed})")
        except Exception:
            if log_fn:
                log_fn(f"  [bootstrap-sweep] seed {s} failed; skipped")
    m = len(all_labels)
    if m < 2:
        return {}
    ari = np.full((m, m), np.nan)
    np.fill_diagonal(ari, 1.0)
    pair_overlap_sizes = {}
    for i in range(m):
        for j in range(i + 1, m):
            inter, ii, jj = np.intersect1d(all_idx[i], all_idx[j], return_indices=True)
            pair_overlap_sizes[(int(used_seeds[i]), int(used_seeds[j]))] = int(len(inter))
            if len(inter) < 30:
                continue
            try:
                a = float(adjusted_rand_score(all_labels[i][ii], all_labels[j][jj]))
            except Exception:
                a = np.nan
            ari[i, j] = ari[j, i] = a
    triu = ari[np.triu_indices(m, 1)]
    triu = triu[~np.isnan(triu)]
    return dict(seeds=used_seeds, counts=counts, ari=ari,
                mean_ari=float(np.mean(triu)) if triu.size else float("nan"),
                pair_overlap_sizes=pair_overlap_sizes)


def _consensus_one_seed(s: int, feats_sc_T: np.ndarray, cfg: dict, n_samp: int,
                         progress_cb=None):
    """Single-seed UMAP+HDBSCAN fit for consensus_cluster()'s per-seed loop.

    Module-level (not a closure), same reasoning as _seed_sweep_one_seed:
    dispatched via joblib's thread-based pool, so it needs to be picklable-
    shaped even though threads share the parent process's memory. No
    refine_clusters_iterative() here -- consensus_cluster()'s own
    co-association/Ward-linkage step resolves fragmentation across seeds
    instead, so per-seed refinement would be redundant cost (see the
    docstring note above about one seed's split pass costing 25+ minutes
    on a real dataset).

    progress_cb(s, result), if given, is called exactly once right before
    returning (result is None on failure) -- see _seed_sweep_one_seed's
    docstring for why: without it the log goes silent for the whole
    dispatch's wall-clock, indistinguishable from a hang.
    """
    try:
        c2 = dict(cfg); c2["umap_random_state"] = s
        # Force per-seed HDBSCAN sweep sequential: consensus_n_jobs already
        # parallelizes across seeds, so letting each seed's own run_hdbscan()
        # also spawn a thread pool would nest parallelism and oversubscribe
        # past the resolved budget (same reasoning as _seed_sweep_one_seed).
        c2["hdbscan_sweep_n_jobs"] = 1
        with _numba_single_thread():
            _, emb = run_umap(feats_sc_T, c2)
            clf, lbls, _, _ = run_hdbscan(emb, c2, n_total=n_samp, log_fn=None)
        lbls = np.asarray(lbls)
        n_cl = len(set(int(x) for x in lbls if x >= 0))
        result = dict(seed=s, labels=lbls, count=n_cl)
        if progress_cb is not None:
            try:
                progress_cb(s, result)
            except Exception:
                pass
        return result
    except Exception:
        if progress_cb is not None:
            try:
                progress_cb(s, None)
            except Exception:
                pass
        return None


def consensus_cluster(feats_sc_T: np.ndarray, cfg: dict, n_seeds: int,
                       log_fn=None, embedding: "np.ndarray | None" = None):
    """
    Opt-in alternative to trusting a single seed's HDBSCAN partition: run
    UMAP+HDBSCAN(+refinement) across n_seeds random seeds (same per-seed
    procedure as seed_sweep_stability), build a co-association matrix (the
    fraction of seeds that placed each pair of bins in the same non-noise
    cluster), then cluster THAT matrix once with Ward-linkage hierarchical
    clustering to get a single seed-independent final partition.

    Rationale (Aug 2026 investigation): on a real short-session dataset,
    UMAP's embedding topology itself varied enormously by random seed (6-seed
    sweep: cluster counts 1-25, mean pairwise ARI 0.31) -- no HDBSCAN sweep-
    spacing or selection-logic change affected this, because the instability
    originates upstream, in the embedding, not in how min_cluster_size is
    swept. This is a widely-reported general property of UMAP+HDBSCAN
    pipelines (seed-dependent local optima), not specific to that dataset.
    Consensus/co-association clustering is the standard general mitigation
    for exactly this failure mode (evidence-accumulation clustering; see e.g.
    Monti et al. 2003 consensus clustering, and Gaia eDR3 stellar-substructure
    clustering literature (arXiv:2208.01056) which keeps only clusters
    "consistently not associated with noise" across repeated runs) --
    it doesn't need to know WHY seeds disagree, only resolves the
    disagreement by construction, so it should generalize to other unstable
    datasets even though the underlying cause may differ.

    Validated on that dataset: Ward linkage produced well-separated clusters
    (3.1-3.7x higher mean co-association within clusters than between) with
    balanced sizes; average-linkage and complete-linkage were also tried and
    both collapsed to a single giant cluster (they chain through the noisy
    co-association matrix) -- Ward is not a swappable default here.

    Cost: ~n_seeds x the UMAP+HDBSCAN runtime of a single fit, plus an
    O(n_samples^2) co-association matrix (~200MB at n_samples=7000, float32;
    scales quadratically -- see consensus_max_memory_gb, a size guard checked
    before the expensive per-seed loop runs at all).
    Off by default; enable with consensus_clustering_enabled=True.

    Post-hoc refinement (Aug 2026, opt-in via consensus_refine_enabled):
    since this partition doesn't come from a single HDBSCAN fit, the
    primary path's merge_similar_clusters() (needs a condensed_tree_) can't
    be reused -- refine_consensus_clusters() below applies split_impure_
    clusters() unchanged plus a new co-association-based merge instead. Also
    always computes feature-space DBCV/silhouette (quality["dbcv_feature_
    space"]/["silhouette_feature_space"]) so this partition's quality is
    numerically comparable to the primary path's, unlike separation_ratio.

    embedding : the PRIMARY single-seed UMAP embedding, optional. Used only
        as a proxy space for refine_consensus_clusters()'s split-impurity
        screening when consensus_refine_enabled=True -- consensus labels
        weren't derived from it (same caveat already documented for
        skipping the silhouette validation gate on consensus mode
        elsewhere). None skips the split half of refinement.

    Returns (labels, quality_dict) with quality_dict = {n_seeds_used,
    per_seed_counts, n_target, separation_ratio, dbcv_feature_space,
    silhouette_feature_space}, or None if fewer than 2 seeds produced a
    usable partition (caller should fall back to the primary single-seed
    result).
    """
    if n_seeds is None or int(n_seeds) < 2:
        return None
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
    except Exception:
        return None

    base_seed = int(cfg.get("umap_random_state", 42))
    seeds = [base_seed + i for i in range(int(n_seeds))]
    n_samp = feats_sc_T.shape[0]

    # ── Memory guard (Aug 2026) ─────────────────────────────────────────────
    # The co-association matrix is O(n_samp^2) float32 -- ~200MB at 7,000
    # training bins, but that scales quadratically: ~1.6GB at 20,000 bins,
    # ~10GB at 50,000. Checked BEFORE the expensive per-seed UMAP+HDBSCAN
    # loop runs (not after), so a dataset that's too large aborts immediately
    # rather than burning n_seeds x a full clustering pass and only then
    # failing at the allocation. consensus_max_memory_gb=0 disables the
    # guard entirely (unbounded, pre-Aug-2026 behavior) for callers who have
    # already sized their own hardware for it.
    _mem_limit_gb = float(cfg.get("consensus_max_memory_gb", 4.0) or 0)
    _est_gb = (n_samp ** 2 * 4) / (1024 ** 3)   # float32 co-association matrix
    if _mem_limit_gb > 0 and _est_gb > _mem_limit_gb:
        if log_fn:
            log_fn(f"  [WARN] consensus clustering skipped: co-association "
                   f"matrix would need ~{_est_gb:.1f} GB for {n_samp} training "
                   f"bins (limit {_mem_limit_gb:.1f} GB, set via "
                   f"consensus_max_memory_gb). Falling back to the primary "
                   f"single-seed HDBSCAN result. Raise consensus_max_memory_gb "
                   f"explicitly if this machine has the RAM to spare, or lower "
                   f"train_frac to reduce the training-bin count.")
        return None

    # NOT running refine_clusters_iterative() per seed here (unlike
    # seed_sweep_stability's diagnostic loop): its split pass runs a full
    # 40-step x 2-method HDBSCAN sweep PER impure cluster candidate, per
    # refinement iteration. On a real 21-session dataset this compounded
    # catastrophically for one seed (25+ min for a single seed's split pass
    # alone, vs 1-2 min for every other seed observed) when that seed's raw
    # partition happened to have many impure clusters. The co-association/
    # Ward-linkage step below already resolves fragmentation via cross-seed
    # agreement, so per-seed refinement's main benefit is largely redundant
    # here for a large, unpredictable worst-case cost. See _consensus_one_seed.
    #
    # consensus_n_jobs (default -1 = auto-managed, same resolve_n_jobs /
    # compute_adaptive_n_jobs budget as seed_sweep_n_jobs / hdbscan_split_
    # n_jobs) dispatches each seed's independent UMAP+HDBSCAN fit via joblib
    # thread pool. This loop used to be strictly sequential regardless of
    # cfg -- unlike seed_sweep_stability and split_impure_clusters, which
    # both already had a working `*_n_jobs` dispatch, this one had none,
    # so the 8-seed (default consensus_n_seeds) co-association pass ran
    # one full UMAP+HDBSCAN fit after another with most cores idle.
    n_jobs = resolve_n_jobs(cfg, "consensus_n_jobs", log_fn=log_fn)
    if n_jobs == 1:
        results = [_consensus_one_seed(s, feats_sc_T, cfg, n_samp) for s in seeds]
    else:
        try:
            from joblib import Parallel, delayed
            if log_fn:
                log_fn(f"  [consensus] dispatching {len(seeds)} seeds across "
                       f"n_jobs={n_jobs} worker threads...")
            # Report as each seed's worker thread finishes -- same reasoning
            # as seed_sweep_stability's dispatch: each seed's UMAP+HDBSCAN fit
            # can take minutes, and with no output in between a stalled seed
            # and a busy one look identical from the log.
            _t0 = time.time()
            _progress_lock = _threading.Lock()
            _progress_done = [0]
            def _on_seed_done(s, r):
                if not log_fn:
                    return
                with _progress_lock:
                    _progress_done[0] += 1
                    n_done = _progress_done[0]
                status = f"{r['count']} clusters" if r else "failed"
                log_fn(f"  [consensus] {n_done}/{len(seeds)} done — seed {s}: "
                       f"{status}  ({time.time() - _t0:.0f}s elapsed)")
            # Heartbeat: all seeds are dispatched at once and do similar work,
            # so their completions tend to arrive in a burst near the end
            # rather than spread across the run -- see seed_sweep_stability's
            # matching heartbeat for the same reasoning.
            _hb_stop = _threading.Event()
            def _heartbeat():
                while not _hb_stop.wait(30):
                    with _progress_lock:
                        n_done = _progress_done[0]
                    if log_fn:
                        log_fn(f"  [consensus] still running — "
                               f"{n_done}/{len(seeds)} seeds done, "
                               f"{time.time() - _t0:.0f}s elapsed...")
            _threading.Thread(target=_heartbeat, daemon=True).start()
            try:
                with _blas_single_thread_for_dispatch():
                    results = Parallel(n_jobs=n_jobs, prefer="threads")(
                        delayed(_consensus_one_seed)(
                            s, feats_sc_T, cfg, n_samp, progress_cb=_on_seed_done)
                        for s in seeds
                    )
            finally:
                _hb_stop.set()
        except Exception:
            results = [_consensus_one_seed(s, feats_sc_T, cfg, n_samp) for s in seeds]

    per_seed_labels, per_seed_counts, _ok_seeds = [], [], []
    for s, r in zip(seeds, results):
        if r is None:
            if log_fn:
                log_fn(f"  [consensus] seed {s} failed; skipped")
        else:
            per_seed_labels.append(np.asarray(r["labels"]))
            per_seed_counts.append(r["count"])
            _ok_seeds.append(s)
            if log_fn:
                log_fn(f"  [consensus] seed {s}: {r['count']} clusters")

    if len(per_seed_labels) < 2:
        return None

    # ── Derived seed-stability stats (Aug 2026) ─────────────────────────────
    # Pairwise ARI across this function's OWN per-seed partitions, computed
    # for free from data already in memory (no extra UMAP/HDBSCAN calls).
    # Uses the exact same keys seed_sweep_stability() returns (seeds, counts,
    # ari, mean_ari) so callers/plot_cluster_stability() can consume either
    # dict interchangeably. CAVEAT: unlike seed_sweep_stability, the per-seed
    # partitions here are NOT passed through refine_clusters_iterative() (see
    # the per-seed loop comment above for why) -- so this mean_ari is measured
    # on unrefined partitions and is not numerically comparable to historical
    # sweep-based ARI values, including the ones consensus_auto_threshold=0.6
    # was calibrated against. Harmless when consensus is force-enabled (no
    # threshold decision is being made from it) -- must NOT be used as an
    # input to the auto-trigger decision itself.
    try:
        from sklearn.metrics import adjusted_rand_score
        _m = len(per_seed_labels)
        _ari = np.eye(_m)
        for _i in range(_m):
            for _j in range(_i + 1, _m):
                try:
                    _a = float(adjusted_rand_score(per_seed_labels[_i],
                                                    per_seed_labels[_j]))
                except Exception:
                    _a = np.nan
                _ari[_i, _j] = _ari[_j, _i] = _a
        # Same degenerate-seed exclusion as seed_sweep_stability() -- a seed
        # with fewer than seed_sweep_min_valid_clusters real clusters
        # compared against a real partition is guaranteed near-zero ARI
        # regardless of true stability.
        _min_valid = int(cfg.get("seed_sweep_min_valid_clusters", 6) or 6)
        _stable = [i for i in range(_m) if per_seed_counts[i] >= _min_valid]
        if len(_stable) >= 2:
            _pairs = [_ari[i, j] for a_, i in enumerate(_stable)
                      for j in _stable[a_ + 1:]]
            _mean_ari = float(np.nanmean(_pairs)) if _pairs else 1.0
            _stable_counts = [per_seed_counts[i] for i in _stable]
        else:
            _triu = _ari[np.triu_indices(_m, 1)]
            _mean_ari = float(np.nanmean(_triu)) if _triu.size else 1.0
            _stable_counts = per_seed_counts
        _stability = dict(seeds=_ok_seeds, counts=per_seed_counts, ari=_ari,
                           mean_ari=_mean_ari, stable_counts=_stable_counts)
    except Exception:
        _stability = {}

    # Co-association matrix: fraction of seeds placing bin i and bin j in the
    # same non-noise cluster. Built incrementally per-cluster (not as a full
    # n_samp x n_samp x n_seeds tensor) to keep memory bounded.
    M = len(per_seed_labels)
    co_assoc = np.zeros((n_samp, n_samp), dtype=np.float32)
    for lbls in per_seed_labels:
        for cid in set(int(x) for x in lbls if x >= 0):
            idx = np.flatnonzero(lbls == cid)
            co_assoc[np.ix_(idx, idx)] += 1.0
    co_assoc /= M
    np.fill_diagonal(co_assoc, 1.0)

    dist = np.clip(1.0 - co_assoc, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    linkage_method = str(cfg.get("consensus_linkage", "ward")).lower().strip()
    Z = linkage(condensed, method=linkage_method)

    # Pick the cut (n_target clusters) within the preferred range that best
    # separates within- from between-cluster co-association, rather than a
    # fixed count -- mirrors the same [pref_lo, pref_hi] the primary HDBSCAN
    # selection uses, so consensus mode doesn't need its own separate tuning.
    target_n = int(cfg.get("target_n_clusters", 0))
    pref_lo  = int(cfg.get("preferred_clusters_lo", 12))
    pref_hi  = int(cfg.get("preferred_clusters_hi", 20))
    candidates = [target_n] if target_n > 0 else list(range(pref_lo, pref_hi + 1))

    rng = np.random.default_rng(base_seed)
    sample_n = min(20000, n_samp * n_samp)
    idx_i = rng.integers(0, n_samp, sample_n)
    idx_j = rng.integers(0, n_samp, sample_n)

    best = None  # (separation_ratio, n_target, labels)
    for n_target in candidates:
        lbls_c = fcluster(Z, t=n_target, criterion="maxclust") - 1
        same = lbls_c[idx_i] == lbls_c[idx_j]
        if same.sum() < 10 or (~same).sum() < 10:
            continue
        intra = co_assoc[idx_i[same], idx_j[same]].mean()
        inter = co_assoc[idx_i[~same], idx_j[~same]].mean()
        ratio = float(intra / max(1e-6, inter))
        if best is None or ratio > best[0]:
            best = (ratio, n_target, lbls_c)

    if best is None:
        return None
    separation_ratio, n_target, labels = best

    # Post-hoc split + merge refinement (Aug 2026, opt-in) -- see
    # refine_consensus_clusters()'s docstring for why merge_similar_
    # clusters() (condensed-tree based) can't be reused here.
    if bool(cfg.get("consensus_refine_enabled", False)):
        labels = refine_consensus_clusters(
            feats_sc_T, labels, co_assoc, embedding, cfg, log_fn=log_fn)

    # Rare-cluster pruning to -1 (noise), same convention/threshold semantics
    # as the primary path's min_cluster_freq pass further down in run() --
    # applied here too so a maxclust cut that happens to isolate a tiny sliver
    # doesn't masquerade as a real cluster.
    _min_freq_pct = float(cfg.get("min_cluster_freq", 0.5))
    _min_freq = _min_freq_pct / 100.0
    for cid in sorted(set(labels[labels >= 0])):
        if (labels == cid).sum() / max(1, n_samp) < _min_freq:
            labels[labels == cid] = -1
    _remaining = sorted(set(labels[labels >= 0]))
    _remap = {old: new for new, old in enumerate(_remaining)}
    labels = np.array([_remap.get(int(x), -1) for x in labels], dtype=int)

    quality = dict(n_seeds_used=M, per_seed_counts=per_seed_counts,
                   n_target=n_target, separation_ratio=separation_ratio)
    quality.update(_stability)
    # Feature-space DBCV/silhouette (Aug 2026): computed on the FINAL
    # (post-refinement, post-pruning) labels -- by explicit choice, matched
    # on the primary-path side too (see run(), computed after
    # refine_clusters_iterative()/rare-cluster pruning there as well) so the
    # two numbers describe the actual DELIVERED partition in both cases, not
    # an intermediate candidate. Always computed (cheap given the subsample
    # cap, purely additive).
    quality["dbcv_feature_space"] = _dbcv_feature_space(
        feats_sc_T, labels, log_fn=log_fn)
    quality["silhouette_feature_space"] = validate_clustering(
        feats_sc_T, labels).get("silhouette_score")
    return labels, quality


def merge_by_coassociation(labels: np.ndarray, co_assoc: np.ndarray,
                            thresh: float, log_fn=None) -> np.ndarray:
    """
    Greedily merge the cluster pair with the highest mean cross-cluster
    co-association, repeating while that max stays >= thresh. This is
    consensus_cluster()'s native analogue of merge_similar_clusters()'s
    condensed-tree-based criterion -- unusable for consensus partitions
    since they don't come from a single HDBSCAN fit (no condensed_tree_ to
    read). Using mean co-association instead is arguably more principled
    here: it directly asks "how often did the 8 seeds agree these two
    clusters' bins belonged together," the same signal consensus_cluster()
    itself is built from.

    thresh <= 0 (default) is a hard no-op -- returns `labels` unchanged.
    """
    labels = np.asarray(labels).copy()
    if thresh is None or thresh <= 0:
        return labels
    while True:
        ids = sorted(set(int(x) for x in labels if x >= 0))
        if len(ids) < 2:
            break
        idx_by_id = {c: np.flatnonzero(labels == c) for c in ids}
        best = None  # (mean_coassoc, ci, cj)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                ci, cj = ids[a], ids[b]
                m = float(co_assoc[np.ix_(idx_by_id[ci], idx_by_id[cj])].mean())
                if best is None or m > best[0]:
                    best = (m, ci, cj)
        if best is None or best[0] < thresh:
            break
        _, ci, cj = best
        labels[labels == cj] = ci
        if log_fn:
            log_fn(f"  [consensus-merge] cluster #{ci} + #{cj} — mean "
                   f"cross-cluster co-association {best[0]:.3f} >= {thresh} "
                   f"-> merged")
    return labels


def refine_consensus_clusters(feats_sc_T: np.ndarray, labels: np.ndarray,
                               co_assoc: np.ndarray,
                               embedding: "np.ndarray | None",
                               cfg: dict, log_fn=None) -> np.ndarray:
    """
    Post-hoc split + merge refinement for consensus_cluster() output
    (Aug 2026, opt-in via consensus_refine_enabled). Split reuses
    split_impure_clusters() completely unchanged -- it doesn't need the
    original per-seed HDBSCAN fits, only feats_sc + an embedding to screen
    impure clusters (re-clustering itself is a fresh local UMAP+HDBSCAN on
    feats_sc subsets). Merge uses merge_by_coassociation() instead of
    merge_similar_clusters() (condensed-tree based, unusable here -- see
    that function's docstring).

    `embedding` is the PRIMARY single-seed embedding, supplied only as a
    proxy space for split's impurity screening -- consensus labels weren't
    derived from it (same caveat this codebase already documents for
    skipping the silhouette validation gate on consensus mode). None skips
    the split half and runs merge-only.
    """
    labels = np.asarray(labels).copy()
    split_thresh = cfg.get("hdbscan_split_silhouette_thresh")
    merge_thresh = float(cfg.get("consensus_merge_coassoc_thresh", 0.0) or 0.0)
    max_iter = int(cfg.get("recluster_max_iterations", 2) or 0)
    if not ((split_thresh or merge_thresh > 0) and max_iter > 0):
        return labels
    if log_fn:
        log_fn(f"\n[consensus-refine]  split/merge refinement "
               f"(up to {max_iter} iteration(s))...")
    feats_sc = feats_sc_T.T
    for _it in range(max_iter):
        before = labels.copy()
        if split_thresh and embedding is not None:
            labels = split_impure_clusters(feats_sc, embedding, labels,
                                            split_thresh, cfg, log_fn=log_fn)
        labels = merge_by_coassociation(labels, co_assoc, merge_thresh,
                                         log_fn=log_fn)
        if np.array_equal(before, labels):
            if log_fn:
                log_fn(f"  [consensus-refine] iteration {_it+1}: "
                       f"no changes — converged")
            break
    return labels


def _dbcv_feature_space(X: np.ndarray, labels: np.ndarray, max_n: int = 5000,
                         log_fn=None) -> float:
    """
    Standalone DBCV (Density-Based Clustering Validation) for an arbitrary
    (X, labels) pair. Unlike run_hdbscan()'s relative_validity_ (only
    available on a freshly-fit HDBSCAN clusterer), hdbscan.validity.
    validity_index() scores any partition directly, so this works on
    consensus_cluster() output (which never has its own HDBSCAN fit) as
    well as any other labels. Scoring in the standardized FEATURE space
    (rather than a particular seed's UMAP embedding) makes the number
    comparable across partitions that don't share one embedding -- e.g.
    consensus vs. the primary single-seed fit. Subsamples to max_n points
    (DBCV's MST step is roughly O(n^2)).
    """
    try:
        from hdbscan.validity import validity_index
    except ImportError:
        if log_fn:
            log_fn("  [WARN] feature-space DBCV skipped: hdbscan.validity unavailable")
        return float("nan")
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    if X.shape[0] > max_n:
        idx = np.random.default_rng(42).choice(X.shape[0], max_n, replace=False)
        X, labels = X[idx], labels[idx]
    if len(set(int(l) for l in labels if l >= 0)) < 2:
        return float("nan")
    try:
        # mst_raw_dist=True: skip hdbscan.validity's all-points-core-distance
        # step, which raises (1/distance)**d with d defaulted to X.shape[1]
        # (589 for CUBE's V2 feature space). At that exponent, any within-
        # cluster pair closer than ~0.3 in this standardized space overflows
        # float64 to inf, which then propagates to NaN in the final validity
        # formula (confirmed: 21/34 clusters NaN on a real run). Raw MST
        # distances (no core-distance density estimate) stay finite by
        # construction, at the cost of no longer weighting by local density —
        # acceptable here since the score only needs to be a stable, relative
        # comparison across configs, not a literal density-based validity
        # value. hdbscan.validity's own docstring notes this flag is meant
        # for exactly this kind of instability ("elongated clusters ... in
        # close proximity").
        return float(validity_index(X, labels.astype(np.intp), metric="euclidean",
                                     mst_raw_dist=True))
    except Exception as e:
        if log_fn:
            log_fn(f"  [WARN] feature-space DBCV failed: {e}")
        return float("nan")


def plot_cluster_stability(sweep: dict, out_path: Path):
    """Cluster-count distribution + pairwise ARI heatmap from seed_sweep_stability."""
    if not sweep or "ari" not in sweep:
        return
    counts = sweep.get("counts", [])
    ari    = np.asarray(sweep["ari"], dtype=float)
    seeds  = sweep.get("seeds", list(range(len(counts))))
    mean_ari = sweep.get("mean_ari", float("nan"))
    dbcv   = [d for d in sweep.get("dbcv", []) if d == d]  # drop NaN

    fig, (axc, axa) = plt.subplots(1, 2, figsize=(13, 5), facecolor=_BG)
    _dark_ax(axc); _dark_ax(axa)

    # Left: cluster-count distribution across seeds.
    if counts:
        _vals, _cnts = np.unique(counts, return_counts=True)
        axc.bar([str(v) for v in _vals], _cnts, color="#4E79A7", alpha=0.85)
    axc.set_xlabel("Cluster count"); axc.set_ylabel("Seeds")
    _dbcv_note = (f"  |  DBCV mean {np.mean(dbcv):.3f} "
                  f"(range {min(dbcv):.3f}–{max(dbcv):.3f})" if dbcv else "")
    axc.set_title(f"Cluster-count stability across {len(seeds)} seeds\n"
                  f"(range {min(counts) if counts else 0}–{max(counts) if counts else 0})"
                  f"{_dbcv_note}",
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


def plot_cluster_volatility(sweep: dict, out_path: Path):
    """Per-cluster volatility across seed_sweep_stability() seeds.

    Aggregate ARI (plot_cluster_stability) only reports whole-partition
    agreement -- it cannot say whether a low ARI is one unstable cluster
    splitting differently or the whole partition shifting a little.  This
    matches seed 0's clusters to every other seed's clusters via Hungarian
    assignment on 1-Jaccard bin-index overlap, and reports each reference
    cluster's mean best-match Jaccard across seeds as a volatility score.
    """
    if not sweep or "labels" not in sweep or not sweep["labels"]:
        return
    all_labels = [np.asarray(l, dtype=int) for l in sweep["labels"]]
    if len(all_labels) < 2:
        return

    from scipy.optimize import linear_sum_assignment

    ref = all_labels[0]
    ref_ids = sorted(c for c in set(ref) if c >= 0)
    if not ref_ids:
        return
    n_bins = ref.shape[0]
    ref_sets  = {c: set(np.flatnonzero(ref == c).tolist()) for c in ref_ids}
    ref_sizes = {c: len(ref_sets[c]) for c in ref_ids}

    best_jaccard = {c: [] for c in ref_ids}
    for other in all_labels[1:]:
        other_ids = sorted(c for c in set(other) if c >= 0)
        if not other_ids:
            for c in ref_ids:
                best_jaccard[c].append(0.0)
            continue
        other_sets = {c: set(np.flatnonzero(other == c).tolist()) for c in other_ids}
        cost = np.ones((len(ref_ids), len(other_ids)))
        for i, rc in enumerate(ref_ids):
            rs = ref_sets[rc]
            for j, oc in enumerate(other_ids):
                osc = other_sets[oc]
                union = len(rs | osc)
                cost[i, j] = 1.0 - (len(rs & osc) / union if union else 0.0)
        row_ind, col_ind = linear_sum_assignment(cost)
        matched = {ref_ids[i]: 1.0 - cost[i, j] for i, j in zip(row_ind, col_ind)}
        for c in ref_ids:
            best_jaccard[c].append(matched.get(c, 0.0))

    volatility = {c: 1.0 - float(np.mean(best_jaccard[c])) if best_jaccard[c] else 1.0
                  for c in ref_ids}

    order = sorted(ref_ids, key=lambda c: -volatility[c])
    vols  = [volatility[c] for c in order]
    sizes = [ref_sizes[c] for c in order]
    pct   = [100.0 * s / n_bins if n_bins else 0.0 for s in sizes]
    ylabels = [f"cluster {c}  (n={sizes[i]}, {pct[i]:.1f}%)" for i, c in enumerate(order)]

    fig_h = max(3.0, 0.32 * len(order) + 1.2)
    fig, ax = plt.subplots(figsize=(9, fig_h), facecolor=_BG)
    _dark_ax(ax)
    y = np.arange(len(order))
    ax.barh(y, vols, color="#E15759", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, color=_TICK_COL, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Volatility  (1 − mean best-match Jaccard across seeds)",
                  color=_TEXT_COL)
    _seed0 = sweep.get("seeds", [0])[0] if sweep.get("seeds") else 0
    ax.set_title(f"Per-cluster volatility across {len(all_labels)} seeds\n"
                 f"(reference = seed {_seed0}; most-changing clusters at top)",
                 color=_TEXT_COL, fontsize=10)
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_cluster_hierarchy(feats_sc, labels, out_path,
                            bodypart_names=None, linkage_method="ward",
                            centroid_out_path=None):
    """
    Dendrogram of the production clustering result's cluster centroids in
    standardized feature space.  Deliberately uses feature-space distance
    rather than the UMAP embedding: UMAP inter-cluster distances are only
    locally meaningful, while feature-space centroid distance is directly
    interpretable as "how different is the underlying movement/pose pattern"
    -- the question the user is actually asking when manually merging
    clusters after inspection.

    Branch color is deliberately a single neutral tone throughout (just
    thicker than a default dendrogram) -- tree topology itself, not branch
    color, is what shows which clusters are more similar to each other.
    Each LEAF gets a colored dot + label in that cluster's permanent CUBE
    identity color (the same _cmap() mapping every other cluster plot in
    this module uses), so a specific cluster can be spotted here and then
    cross-referenced against an ethogram/UMAP/etc. at a glance -- that is
    the only thing color is meant to encode on this plot.
    """
    if feats_sc is None or labels is None:
        return
    labels = np.asarray(labels)
    uniq = sorted(c for c in set(labels.tolist()) if c >= 0)
    if len(uniq) < 2:
        return

    from scipy.cluster.hierarchy import linkage, dendrogram

    X = np.asarray(feats_sc)
    # Accept either (n_feat, n_samp) or (n_samp, n_feat); orient to (n_samp, n_feat).
    if X.shape[0] == labels.shape[0]:
        pass
    elif X.shape[1] == labels.shape[0]:
        X = X.T
    else:
        return

    sizes = {c: int((labels == c).sum()) for c in uniq}
    centroids = np.vstack([X[labels == c].mean(axis=0) for c in uniq])

    if centroid_out_path is not None:
        try:
            np.savez(centroid_out_path, centroids=centroids,
                     cluster_ids=np.array(uniq),
                     linkage_method=np.array(linkage_method))
        except Exception:
            pass

    Z = linkage(centroids, method=linkage_method)
    leaf_labels = [f"C{c}" for c in uniq]

    fig_w = max(8.0, 0.55 * len(uniq) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5), facecolor=_BG)
    _dark_ax(ax)
    dn = dendrogram(Z, labels=leaf_labels, ax=ax,
                     link_color_func=lambda k: _TICK_COL)
    for line in ax.get_lines():
        line.set_linewidth(2.2)
        line.set_alpha(0.95)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.tick_params(axis="y", colors=_TICK_COL)

    # Leaf identity: colored dot + "C{n} (n=...)" label stacked below the
    # axis, using scipy's own fixed leaf-spacing formula (5, 15, 25, ...)
    # directly -- NOT ax.get_xmajorticklabels(), which comes back EMPTY
    # once labelbottom=False is set above (matplotlib drops those Text
    # objects from that accessor entirely, not just their visibility).
    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    n_leaves = len(dn["leaves"])
    leaf_x = 5.0 + 10.0 * np.arange(n_leaves)
    for x, leaf_idx in zip(leaf_x, dn["leaves"]):
        c = uniq[leaf_idx]
        col = _cmap(c)
        ax.scatter([x], [y0 - 0.045 * yr], s=100, color=col,
                   edgecolor=_BG, linewidth=1.0, zorder=6, clip_on=False)
        ax.text(x, y0 - 0.09 * yr, f"C{c}\n(n={sizes[c]})", color=col,
                fontsize=8, fontweight="bold", ha="center", va="top",
                linespacing=1.3, clip_on=False)
    ax.set_ylim(y0 - 0.24 * yr, y1)

    ax.set_ylabel(f"{linkage_method.title()} linkage distance (feature space)",
                  color=_TEXT_COL)
    ax.set_title(f"Cluster hierarchy ({len(uniq)} clusters, feature-space centroids)",
                 color=_TEXT_COL, fontsize=10)
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_cluster_validity(embedding: np.ndarray, labels: np.ndarray,
                           feats_sc: np.ndarray, out_path: Path,
                           hdb_clf=None) -> None:
    """
    Cluster validity/consistency diagnostic for the PRIMARY HDBSCAN
    clustering (embedding + raw features) — parallels plot_cluster_stability's
    two-panel layout and theme-global convention (module-level _BG/_PANEL/
    _TEXT_COL/_TICK_COL, set by _apply_plot_theme(); never hardcode colours).

    Left  : per-cluster silhouette diagram (sklearn.metrics.silhouette_samples,
            subsampled the same way validate_clustering already does) —
            horizontal bars per cluster, sorted, coloured by cluster id, with
            the overall mean silhouette line.
    Right : HDBSCAN condensed-tree plot (hdb_clf.condensed_tree_.plot) showing
            cluster persistence/stability directly from the fitted classifier
            — visual grounding for issue 4's condensed-tree merge pass.  When
            hdb_clf is None (e.g. cube_analyser.py's on-the-fly figure, which
            only has the saved .npy arrays, not the fitted classifier), the
            right panel is a placeholder note instead.

    feats_sc : (n_samples, n_features) — i.e. feats_sc.T from BSoidEngine.run().
        Accepted for API symmetry with the run() call site / potential future
        feature-space diagnostics, but the silhouette diagram itself is
        computed on `embedding` (matching validate_clustering's existing
        embedding-based silhouette gate, so the two numbers are comparable).
    """
    _ = feats_sc   # reserved; silhouette uses `embedding` (see docstring)
    sil_full, cluster_means = _mean_silhouette_per_cluster(embedding, labels)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 7), facecolor=_BG)
    _dark_ax(axl)

    if cluster_means:
        overall_mean = float(np.nanmean(list(cluster_means.values())))
        y0 = 0
        yticks, yticklabels = [], []
        for cid in sorted(cluster_means):
            vals = np.sort(sil_full[(labels == cid) & ~np.isnan(sil_full)])
            if vals.size == 0:
                continue
            y1 = y0 + vals.size
            axl.fill_betweenx(np.arange(y0, y1), 0, vals,
                              facecolor=_cmap(cid), edgecolor=_cmap(cid), alpha=0.85)
            yticks.append((y0 + y1) / 2)
            yticklabels.append(f"#{cid}")
            y0 = y1 + max(5, int(0.02 * len(sil_full)))
        axl.axvline(overall_mean, color=_TEXT_COL, linestyle="--", linewidth=1,
                    label=f"mean = {overall_mean:.3f}")
        axl.set_yticks(yticks)
        axl.set_yticklabels(yticklabels, color=_TICK_COL, fontsize=8)
        axl.legend(loc="lower right", facecolor=_PANEL, edgecolor="none",
                   labelcolor=_TEXT_COL, fontsize=8)
    else:
        axl.text(0.5, 0.5, "Not enough labelled clusters\nfor a silhouette diagram",
                 ha="center", va="center", color=_TEXT_COL, transform=axl.transAxes)
    axl.set_xlabel("Silhouette coefficient", color=_TEXT_COL)
    axl.set_title("Per-cluster silhouette diagram", color=_TEXT_COL, fontsize=11)

    axr.set_facecolor(_PANEL)
    if hdb_clf is not None and hasattr(hdb_clf, "condensed_tree_"):
        # select_clusters=True draws boundaries around hdb_clf's OWN
        # originally-selected clusters, which is only meaningful when
        # `labels` (used for the left panel) still matches that original
        # selection.  After split/merge refinement and/or rare-cluster
        # pruning, `labels` has typically been renumbered/reshaped and no
        # longer corresponds 1:1 -- so the boundaries would visualise a
        # stale, pre-refinement partition.  Only ask for them when the
        # two are still known to agree (i.e. this run's cluster count
        # matches hdb_clf's own selected-cluster count); otherwise fall
        # back to the unselected tree, which is still informative.
        _orig_n_cl = int(getattr(hdb_clf, "labels_", np.array([])).max() + 1) \
            if getattr(hdb_clf, "labels_", None) is not None else -1
        _cur_n_cl = len(set(int(l) for l in labels if l >= 0))
        _select = _orig_n_cl == _cur_n_cl
        _orig_fig_axes = set(fig.axes)   # to strip any leaked colorbar axis on retry

        def _try_plot(select_clusters: bool) -> bool:
            # condensed_tree_.plot(colorbar=True) appends a NEW colorbar axis
            # to the figure via plt.colorbar() -- axr.clear() only clears axr
            # itself, so a failed first attempt (select_clusters=True) leaves
            # its colorbar axis orphaned on the figure; without removing it
            # here, a successful retry ends up with two overlapping colorbars.
            for _a in list(fig.axes):
                if _a not in _orig_fig_axes:
                    fig.delaxes(_a)
            axr.clear()
            axr.set_facecolor(_PANEL)
            hdb_clf.condensed_tree_.plot(select_clusters=select_clusters,
                                         axis=axr, colorbar=True)
            # condensed_tree_.plot() defers some rendering (e.g. colorbar
            # tick/label layout, or -- confirmed on real data -- an upstream
            # hdbscan bug where a cluster with near-infinite lambda values
            # falls back to a 1-element numpy array instead of a scalar for
            # the selection-ellipse height, which only raises once
            # matplotlib actually lays out that Ellipse) until the figure is
            # actually drawn, so a malformed value there can raise only at
            # savefig() time -- too late for a try/except around plot()
            # alone to catch it gracefully. Force that rendering NOW, inside
            # the try, so a failure here still gets a fallback instead of
            # only surfacing later as a generic [WARN] on the whole figure.
            axr.figure.canvas.draw()
            return True

        _ok = False
        try:
            _ok = _try_plot(_select)
        except Exception:
            # The select_clusters=True path is the one that hits the
            # upstream Ellipse-height bug (only triggered when at least one
            # selected cluster has a non-finite persistence-lambda range,
            # i.e. an extremely dense cluster) -- retry once without cluster
            # boundaries before giving up entirely, so users still get the
            # tree structure itself instead of nothing.
            if _select:
                try:
                    _ok = _try_plot(False)
                    _select = False
                except Exception:
                    _ok = False
            else:
                _ok = False

        if _ok:
            axr.set_title("HDBSCAN condensed tree (cluster persistence)"
                          + ("" if _select else "\n(unselected — refined "
                             "since original fit)"),
                          color=_TEXT_COL, fontsize=11)
            axr.tick_params(colors=_TICK_COL)
            for spine in axr.spines.values():
                spine.set_color(_TICK_COL)
        else:
            for _a in list(fig.axes):
                if _a not in _orig_fig_axes:
                    fig.delaxes(_a)
            axr.clear()
            _dark_ax(axr)
            axr.text(0.5, 0.5, "Condensed-tree plot unavailable\n"
                     "(hdbscan condensed_tree_.plot() failed)",
                     ha="center", va="center", color=_TEXT_COL,
                     transform=axr.transAxes)
            axr.set_xticks([]); axr.set_yticks([])
    else:
        _dark_ax(axr)
        axr.text(0.5, 0.5, "No fitted HDBSCAN classifier available\n"
                 "(condensed-tree view requires the pipeline-exported plot)",
                 ha="center", va="center", color=_TEXT_COL, transform=axr.transAxes)
        axr.set_xticks([]); axr.set_yticks([])

    fig.suptitle("Cluster Validity — is each cluster tight and well-separated?",
                 color=_TEXT_COL, fontsize=13, y=0.98)
    fig.text(0.5, 0.925,
             "Left: bars below 0 or well left of the mean are split candidates "
             "(hdbscan_split_silhouette_thresh)   |   Right: short/thin "
             "branches are weak splits worth merging (hdbscan_merge_thresh)",
             ha="center", va="top", color=_TICK_COL, fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.88))
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


def plot_publication_metrics(metrics: dict, out_path: Path):
    """
    Publication-ready figure showing all five Section 3.2 benchmark metrics
    with target thresholds from Section 3.4.  Saved to publication_metrics.png.

    metrics keys expected:
        silhouette      float   clustering quality (target ≥ 0.71)
        trustworthiness float   manifold fidelity (target ≥ 0.92)
        mean_ari        float   cluster stability ARI (target ≥ 0.70)
        runtime_min     float   min of processing per min of video (target ≤ 1.2)
        peak_memory_gb  float   peak RAM in GB (target ≤ 3.8)
    """
    ITEMS = [
        ("silhouette",      "Silhouette\nScore",         0.71, 1.0,  True,  "≥ 0.71\n(target)"),
        ("trustworthiness",  "UMAP\nTrustworthiness",    0.92, 1.0,  True,  "≥ 0.92\n(target)"),
        ("mean_ari",         "Cluster\nStability (ARI)", 0.70, 1.0,  True,  "≥ 0.70\n(target)"),
        ("runtime_min",      "Runtime\n(min/min video)", 1.2,  None, False, "≤ 1.2\n(target)"),
        ("peak_memory_gb",   "Peak Memory\n(GB)",        3.8,  None, False, "≤ 3.8\n(target)"),
    ]
    PASS_COL  = "#4caf50"
    WARN_COL  = "#ffd60a"
    TARGET_COL= "#ff7043"

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), facecolor=_BG)
    fig.suptitle("CUBE Publication Benchmark Metrics  (Section 3.2)",
                 color=_TEXT_COL, fontsize=12, fontweight="bold")

    for ax, (key, label, target, xlim_max, higher_is_better, target_label) in zip(axes, ITEMS):
        _dark_ax(ax)
        val = metrics.get(key)
        if val is None:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", color="#888888", fontsize=11)
            ax.set_title(label, color=_TEXT_COL, fontsize=9)
            continue

        if higher_is_better:
            passed = val >= target
        else:
            passed = val <= target
        color = PASS_COL if passed else WARN_COL

        if xlim_max is not None:
            ax.barh([label], [val], color=color, edgecolor="none", height=0.5)
            ax.axvline(target, color=TARGET_COL, linestyle="--", linewidth=1.5)
            ax.set_xlim(0, xlim_max * 1.05)
            ax.text(min(val + xlim_max * 0.02, xlim_max * 0.95), 0,
                    f"{val:.3f}", va="center", color="#eaeaea", fontsize=10)
            ax.text(target + xlim_max * 0.01, 0.42, target_label,
                    va="bottom", color=TARGET_COL, fontsize=7)
        else:
            # Runtime / memory: bar without fixed xlim
            bar_max = max(val, target) * 1.3
            ax.barh([label], [val], color=color, edgecolor="none", height=0.5)
            ax.axvline(target, color=TARGET_COL, linestyle="--", linewidth=1.5)
            ax.set_xlim(0, bar_max)
            ax.text(val + bar_max * 0.02, 0, f"{val:.2f}", va="center",
                    color="#eaeaea", fontsize=10)
            ax.text(target + bar_max * 0.01, 0.42, target_label,
                    va="bottom", color=TARGET_COL, fontsize=7)

        badge = "PASS" if passed else "WARN"
        badge_col = PASS_COL if passed else WARN_COL
        ax.text(0.97, 0.95, badge, transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=badge_col, fontweight="bold")
        ax.set_title(label, color=_TEXT_COL, fontsize=9)
        ax.set_yticks([])

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
            # Prefer embedding-space proximity to the cluster's own centroid
            # (issue 1a) — two clips of "typical duration" can sit at opposite
            # ends of a cluster in embedding space, which is the actual root
            # cause of "inconsistent" example clips.  Falls back to the
            # original duration-based selection when _centroid_dist isn't
            # available (legacy/standalone callers with a hand-built epochs
            # DataFrame) — no signature break, no new required parameter.
            if "_centroid_dist" in grp_epochs.columns and grp_epochs["_centroid_dist"].notna().any():
                grp_epochs["_dist"] = grp_epochs["_centroid_dist"]
            else:
                median_dur = grp_epochs["duration_sec"].median()
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
                          output_fps: int = 15,
                          turned_away_frame_mask: "np.ndarray | None" = None):
    """Write the full session video with per-frame cluster label overlay.

    turned_away_frame_mask : optional (n_video_frames,) boolean array (v3
        turned-away detection, see detect_turned_away_bins /
        _expand_bin_mask_to_frames). Where True, an additional amber
        "TURNED AWAY" banner is burned in — the exact drawing style validated
        in this session's scratch corroboration (overlay_turned_away.py):
        solid amber (BGR (0,165,255)) rectangle + black text. Drawn as a
        full-width strip at the BOTTOM of the frame so it never collides with
        the existing top-left "C{lbl}" cluster-label box. Independent of the
        cluster-label box: always drawn when the mask is True for a frame,
        regardless of what `lbl` says (this is a diagnostic overlay of
        detection, not of the dedicated-label override applied to the
        exported CSVs).
    """
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
        _ta_mask = (np.asarray(turned_away_frame_mask, dtype=bool)
                    if turned_away_frame_mask is not None
                    and len(turned_away_frame_mask) > 0 else None)
        _banner_h = max(40, h // 12)
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
                    if (_ta_mask is not None
                            and _ta_mask[min(src_fi, len(_ta_mask) - 1)]):
                        cv2.rectangle(frame, (0, h - _banner_h), (w, h),
                                     (0, 165, 255), -1)
                        cv2.putText(frame, "TURNED AWAY",
                                    (10, h - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 0, 0), 2, cv2.LINE_AA)
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
    """Filter to conserved bodyparts and save H5 + CSV in B-SOiD format.

    Preserves the z coord column when the input is a 3D triangulated H5
    (coords = [x, y, z, likelihood]) so that BSoidEngine._h5_has_z() can
    detect 3D format downstream.  2D H5s (coords = [x, y, likelihood]) are
    written identically to before.
    """
    df = df.loc[:, (slice(None), bodyparts_to_keep, slice(None))]
    # Detect 3D format: 4 coords per bodypart including 'z'
    all_coords = df.columns.get_level_values("coords").unique().tolist()
    _has_z     = "z" in all_coords
    coords_out = ["x", "y", "z", "likelihood"] if _has_z else ["x", "y", "likelihood"]
    new_cols = pd.MultiIndex.from_product(
        [[BSOID_SCORER_NAME], bodyparts_to_keep, coords_out],
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


def triangulate_cameras(pts2d: np.ndarray,
                        scores: np.ndarray,
                        calib_toml,
                        use_ransac: bool = False,
                        ransac_threshold: float = 0.5,
                        ll_gate: float = 0.6,
                        median_window: int = 3,
                        log_fn=print) -> np.ndarray:
    """
    Triangulate 2D keypoints from N cameras into 3D world coordinates.

    Parameters
    ----------
    pts2d            : (N_cams, N_frames, N_bp, 2)  — pixel x/y per camera
    scores           : (N_cams, N_frames, N_bp)     — DLC likelihood per keypoint
    calib_toml       : path to aniposelib calibration.toml
    use_ransac       : use RANSAC-based triangulation (robust for occluded keypoints)
    ransac_threshold : reprojection error threshold in pixels for RANSAC inlier test
    ll_gate          : pre-triangulation drop-gate; any camera view with DLC likelihood
                       below this value has its score zeroed before aniposelib so that
                       view is excluded from triangulation (0.0 = off)
    median_window    : kernel size for post-triangulation temporal median filter applied
                       per bodypart per axis to remove single-frame 3D jitter; must be
                       odd; 0 or 1 = off
    log_fn           : callable for log messages

    Returns
    -------
    (N_frames, N_bp, 3)  — X, Y, Z in calibration world space
    """
    try:
        from aniposelib.cameras import CameraGroup
    except ImportError:
        raise ImportError(
            "aniposelib is required for 3D triangulation.\n"
            "Install with: pip install aniposelib")

    cgroup = CameraGroup.load(str(calib_toml))

    N_cams, N_frames, N_bp, _ = pts2d.shape
    if scores.shape != (N_cams, N_frames, N_bp):
        raise ValueError(
            f"scores shape {scores.shape} does not match "
            f"pts2d shape {pts2d.shape[:3]}")

    pts3d = np.zeros((N_frames, N_bp, 3), dtype=np.float64)
    for bp_i in range(N_bp):
        p2d = pts2d[:, :, bp_i, :]   # (N_cams, N_frames, 2)
        sc  = scores[:, :, bp_i]      # (N_cams, N_frames)

        # Pre-triangulation drop-gate: zero out scores for any camera view whose
        # DLC likelihood falls below ll_gate so that view contributes nothing to
        # the weighted triangulation rather than just being down-weighted.
        if ll_gate > 0.0:
            sc = sc.copy()
            sc[sc < ll_gate] = 0.0

        p3  = None

        if use_ransac:
            try:
                p3 = cgroup.triangulate_ransac(
                    p2d, scores=sc,
                    threshold=ransac_threshold,
                    progress=False)
            except (AttributeError, TypeError):
                pass          # fall through to standard triangulate below
            except Exception as e:
                log_fn(f"  [WARN] triangulate_ransac bp {bp_i}: {e}")
                p3 = np.zeros((N_frames, 3), dtype=np.float64)

        if p3 is None:
            try:
                p3 = cgroup.triangulate(p2d, scores=sc, progress=False)
            except TypeError:
                # older aniposelib may not accept scores kwarg
                try:
                    p3 = cgroup.triangulate(p2d, progress=False)
                except Exception as e:
                    log_fn(f"  [WARN] triangulate bp {bp_i}: {e}")
                    p3 = np.zeros((N_frames, 3), dtype=np.float64)
            except Exception as e:
                log_fn(f"  [WARN] triangulate bp {bp_i}: {e}")
                p3 = np.zeros((N_frames, 3), dtype=np.float64)

        if p3 is None or p3.shape != (N_frames, 3):
            if p3 is not None:
                log_fn(f"  [WARN] triangulate bp {bp_i}: unexpected shape {p3.shape}, "
                       f"expected ({N_frames}, 3) — zeroing bodypart")
            p3 = np.zeros((N_frames, 3), dtype=np.float64)
        pts3d[:, bp_i, :] = p3

    # Post-triangulation temporal median filter: removes single-frame 3D jitter
    # caused by momentary tracking artifacts that survive RANSAC spatial filtering.
    _mw = int(median_window)
    if _mw > 1:
        if _mw % 2 == 0:
            _mw += 1  # enforce odd kernel
        from scipy.signal import medfilt
        for bp_i in range(N_bp):
            for ax in range(3):
                pts3d[:, bp_i, ax] = medfilt(pts3d[:, bp_i, ax], kernel_size=_mw)

    return pts3d


def _bsoid_prep_discover_and_analyze(source_folder, log_fn=print):
    """
    Discover *_filtered.h5 sessions in source_folder and compute per-bodypart
    confidence stats for each (factored out of run_bsoid_prep so the batch
    variant below can pool stats across several folders before deciding
    which bodyparts to keep).

    Returns (folder_name, bsoid_root, all_stats) where
    all_stats = {session_name: (h5_path, {bodypart: {mean, median}})},
    or None if no valid sessions were found (already logged).
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

    return folder_name, bsoid_root, all_stats


def _bsoid_prep_determine_conserved(all_stats: dict, min_confidence: float,
                                     conf_metric: str, min_session_frac: float,
                                     min_keep: int, log_fn=print) -> list:
    """
    Shared conserved-bodyparts determination (factored out of run_bsoid_prep).
    Works identically on a single folder's all_stats or on a dict pooled
    across multiple folders — the caller decides the scope; this function
    just applies the same "confidence >= min_confidence in >= min_session_frac
    of sessions" rule to whatever sessions it's given.
    """
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
    return conserved


def _bsoid_prep_export(root_path: Path, folder_name: str, bsoid_root: Path,
                        all_stats: dict, conserved: list, log_fn=print) -> Path:
    """
    Export conserved-bodypart H5/CSV + copy videos + write the confidence
    report for one folder, given an already-determined `conserved` bodypart
    list (factored out of run_bsoid_prep so the same export logic works
    whether `conserved` was computed on this folder alone or pooled across
    several folders for cross-group comparability).
    """
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
            # 3D mode: prefer the quad composite video explicitly
            quad_cands = sorted(h5_src.parent.glob("*_quad.mp4"))
            if quad_cands:
                video_src = quad_cands[0]
        if not video_src:
            # Any MP4 in the session folder (skip pseudo/before_adapt artifacts)
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


def run_bsoid_prep(source_folder, log_fn=print,
                   min_confidence: float = MIN_BODYPART_CONFIDENCE,
                   conf_metric: str = "median",
                   min_session_frac: float = 0.85,
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

    NOTE: when pre-processing several folders that will later be combined into
    one cross-group analysis (e.g. multiple experimental groups/conditions),
    prefer run_bsoid_prep_batch() instead — it makes this same decision ONCE,
    pooled across every folder's sessions, rather than each folder choosing its
    own conserved set independently and relying on a later intersection to
    reconcile them (which compounds three separately-conservative cuts into one
    overly aggressive one). This function is kept as-is for single-folder use.

    Returns path to BSOID_Project_Ready or None on failure.
    """
    discovered = _bsoid_prep_discover_and_analyze(source_folder, log_fn)
    if discovered is None:
        return None
    folder_name, bsoid_root, all_stats = discovered

    conserved = _bsoid_prep_determine_conserved(
        all_stats, min_confidence, conf_metric, min_session_frac, min_keep, log_fn)
    if not conserved:
        return None

    return _bsoid_prep_export(Path(source_folder), folder_name, bsoid_root,
                               all_stats, conserved, log_fn)


def run_bsoid_prep_batch(source_folders: list, log_fn=print,
                         min_confidence: float = MIN_BODYPART_CONFIDENCE,
                         conf_metric: str = "median",
                         min_session_frac: float = 0.85,
                         min_keep: int = 6) -> list:
    """
    Pre-process multiple source folders (e.g. one per experimental
    group/condition that will be combined into a single cross-group CUBE
    analysis) with ONE shared conserved-bodyparts list, computed once from
    every session across every folder pooled together.

    Why (Aug 2026): the per-folder run_bsoid_prep() + a later cross-group
    bodypart intersection (in BSoidEngine.run(), Step 2) both individually
    make sense, but composed they compound. On a real 3-group dataset (7
    sessions/group), each folder's independent 60%-of-its-own-sessions cut
    kept 27/27/24 bodyparts; intersecting those three already-conservative
    sets left only 22 shared. Pooling the identical rule across all 21
    sessions in one pass avoids that compounding -- but at the original 60%
    threshold it then pooled in 6 bodyparts (up to 71% bad frames in some
    sessions) that individual folders had correctly excluded, and which
    measurably hurt clustering (DBCV 0.370->0.164, mean seed-stability ARI
    0.620->0.427 in real testing). Raising min_session_frac's default to
    0.85 keeps the pooling fix (still ONE shared list, comparability
    unaffected either way) while excluding those specific bodyparts again,
    recovering DBCV/ARI. Tune per-dataset via the GUI's "BP keep if passes
    >=" (Advanced Settings, DLC & Prep) if your data's tracking quality
    differs.

    Returns [bsoid_root, ...], one per folder that had valid sessions
    (in source_folders order; folders that failed discovery are omitted,
    matching run_bsoid_prep's None-on-failure convention for a single
    folder).
    """
    discovered = []
    for folder in source_folders:
        log_fn(f"  Scanning: {Path(folder).name}")
        d = _bsoid_prep_discover_and_analyze(folder, log_fn)
        if d is not None:
            discovered.append((folder, *d))
    if not discovered:
        return []

    # Pool every session's stats across all folders for a single, shared
    # conserved-bodyparts determination. Keyed by (folder_name, session) so
    # identically-named sessions in different groups can't collide.
    pooled_stats: dict = {}
    for folder, folder_name, bsoid_root, all_stats in discovered:
        for sname, entry in all_stats.items():
            pooled_stats[f"{folder_name}::{sname}"] = entry

    log_fn(f"\n  Pooling {len(pooled_stats)} session(s) across "
           f"{len(discovered)} folder(s) for a single shared "
           f"conserved-bodyparts determination (guarantees cross-group "
           f"comparability without compounding independent per-folder cuts):")
    conserved = _bsoid_prep_determine_conserved(
        pooled_stats, min_confidence, conf_metric, min_session_frac, min_keep,
        log_fn)
    if not conserved:
        return []

    roots = []
    for folder, folder_name, bsoid_root, all_stats in discovered:
        log_fn(f"\n  Exporting: {folder_name}")
        r = _bsoid_prep_export(Path(folder), folder_name, bsoid_root,
                               all_stats, conserved, log_fn)
        if r:
            roots.append(r)
    return roots


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
                                        # rigs occlude/degrade a bodypart's
                                        # tracking for up to ~20% of a session
                                        # in practice (measured via real
                                        # bodypart_confidence_report data and
                                        # the visibility-feature turned-away
                                        # detector, Aug 2026 -- not the 40-70%
                                        # figure this comment previously cited,
                                        # which overstated typical occlusion
                                        # frequency); excluding chronically
                                        # occluded bodyparts keeps the feature
                                        # space clean and prevents HDBSCAN
                                        # noise inflation.
        feature_bad_bp_thresh_with_visibility = 0.70,
                                        # Governs bodypart dropping INSTEAD of
                                        # feature_bad_bp_thresh above, whenever
                                        # visibility_features_enabled=True (the
                                        # default). Real confidence-report data
                                        # (bodypart_confidence_report, Aug 2026)
                                        # showed limb-joint keypoints (thigh/
                                        # knee) and neck/throat/back-middle
                                        # regularly average 0.27-0.41 confidence
                                        # per session -- close to or under the
                                        # plain 0.40 threshold even in normal
                                        # tracking, while head/mouth keypoints
                                        # stayed >=0.45 even in the worst
                                        # session. Dropping a limb joint
                                        # permanently removes real behavioural
                                        # signal from every well-tracked frame
                                        # too, directly undermining bodypart_
                                        # weights up-weighting of that same
                                        # region. With visibility features on,
                                        # low-confidence bins (occlusion, animal
                                        # turned away) are instead isolated into
                                        # their own cluster (see compute_
                                        # visibility_features) rather than
                                        # requiring the bodypart's removal, so
                                        # only near-unusable bodyparts (>=70%
                                        # bad) need dropping here. Set equal to
                                        # feature_bad_bp_thresh to restore the
                                        # stricter legacy drop behaviour even
                                        # with visibility features on.
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
        hdbscan_method        = "both",       # tries eom + leaf, DBCV picks best
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
        preferred_clusters_hi = 20,   # auto-mode: prefer cluster count ≤ this
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
        # hdbscan_selection_mode : "legacy" keeps the old two-branch
        #   in-range/boundary-fallback rule, which can discontinuously jump to
        #   a structurally different rule when no swept candidate falls inside
        #   [preferred_clusters_lo, preferred_clusters_hi].  "floor_soft_cap"
        #   (default since Aug 2026) replaces it with one continuous ranking
        #   pass: a hard floor at preferred_clusters_lo plus a soft linear
        #   penalty (hdbscan_overshoot_penalty) above preferred_clusters_hi.
        # hdbscan_overshoot_penalty : only active under "floor_soft_cap". Score
        #   penalty per cluster above preferred_clusters_hi (0 = no ceiling).
        hdbscan_selection_mode    = "floor_soft_cap",  # "legacy" | "floor_soft_cap"
                                        # Promoted Aug 2026 (real 3-group,
                                        # 8-seed sweep: legacy collapsed 5/8
                                        # seeds to 3 clusters, floor_soft_cap
                                        # collapsed 0/8; mean ARI 0.345→0.589;
                                        # zero quality regression on the
                                        # deterministic primary run).  See
                                        # CLUSTER_SELECTION_SIMPLIFICATION_PLAN_2026-08.md
                                        # changelog.
        hdbscan_overshoot_penalty = 0.01,
        # cluster_hierarchy_enabled : dendrogram of the production clustering
        #   result's cluster centroids in feature space -- supports manual
        #   post-hoc cluster merging. Additive-only output, safe to default on.
        cluster_hierarchy_enabled = True,
        cluster_hierarchy_linkage = "ward",   # "ward" | "average" | "complete"
        # ── Feature options ───────────────────────────────────────────────────
        body_normalise        = False,  # normalise by nose-to-tailbase length
        long_lag_drift        = False,  # add 2-s and 3-s lag drift offsets (freeze/guarding detection)
        long_scale_bins       = False,  # add 500-ms and 1000-ms coarse temporal bins (off by default)
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
        plot_theme            = "light",  # "dark" or "light"
        # ── HMM post-hoc smoothing ────────────────────────────────────────────
        hmm_enabled           = True,    # wrap MLP output with Multinomial HMM
        hmm_n_states          = 0,       # 0/None → n_clusters (smoothing-only mode)
        hmm_n_iter            = 100,     # Baum-Welch EM iterations
        hmm_random_state      = 42,      # seeds CategoricalHMM/GaussianHMM startprob_
                                        #   init so train_hmm/train_hmm_soft are
                                        #   reproducible across identical-input calls
        hmm_min_prob          = 0.05,    # min edge probability in syntax network plot
        # hmm_transition_prior: "per_cluster" (default since Aug 2026, was
        #   "global" pre-Aug-2026) derives each cluster's own self-transition
        #   prior from its mean observed bout duration (see
        #   _compute_cluster_self_trans / train_hmm docstring) instead of a
        #   flat 90%/10% self/spread prior for every state -- validated on
        #   real data (A.4/B.6 test suite, Aug 2026) to avoid over-smoothing
        #   naturally brief behaviours. Set to "global" to restore the old
        #   flat-prior behavior exactly.
        hmm_transition_prior  = "per_cluster",
        # hmm_smoothing_level: "bin" (default since Aug 2026, was "frame"
        #   pre-Aug-2026) trains/decodes the HMM on the underlying per-bin
        #   sequence (1/win the length of the frame-repeated sequence), then
        #   expands the decoded bin-level states back to frame resolution
        #   the same way predict_labels() expands raw bin labels -- avoids
        #   diluting the learned transition matrix with trivial within-bin
        #   self-transitions, and is faster (fewer sequence elements for
        #   Baum-Welch/Viterbi). Only takes effect when hmm_emission_mode
        #   is "categorical" -- "soft" emissions already operate at bin
        #   resolution unconditionally (see train_hmm_soft). Set to "frame"
        #   to restore the old frame-repeated behavior exactly.
        hmm_smoothing_level   = "bin",
        # hmm_emission_mode: "soft" (default since Aug 2026, was
        #   "categorical" pre-Aug-2026) fits a GaussianHMM on the MLP's
        #   per-bin class-probability vectors instead of a CategoricalHMM on
        #   hard argmax labels (see train_hmm_soft docstring) -- lets frames
        #   the MLP was uncertain about get less-confident smoothing than
        #   frames it was near-100% sure of. Set to "categorical" to restore
        #   the old hard-label HMM behavior exactly.
        hmm_emission_mode     = "soft",
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
        #   0 = off. Default 6 -- adds runtime but stability is worth checking
        #   on every run rather than only when the user remembers to enable it.
        seed_sweep_n          = 6,
        # seed_sweep_n_jobs: dispatch of seed_sweep_stability()'s per-seed
        #   UMAP+HDBSCAN(+refinement) fits via joblib (thread-based pool,
        #   same choice as hdbscan_split_n_jobs). Default 1 (sequential)
        #   since Aug 2026, after a real run hit a Windows heap-corruption
        #   fault (0xc0000374, inside ntdll.dll's allocator -- not
        #   reproducible on demand, so a safety margin rather than a
        #   pinned-down fix). Measured cost of going sequential: none --
        #   the per-worker numba/BLAS single-threading already in effect
        #   (see _numba_single_thread/_blas_single_thread_for_dispatch)
        #   left little real parallel throughput to lose; a real 6-seed
        #   sweep timed marginally FASTER sequential (490s) than parallel
        #   (503s). -1 = auto-managed (System Resources block below);
        #   >1 = pin an exact worker count.
        seed_sweep_n_jobs      = 1,
        # seed_sweep_min_valid_clusters: a per-seed partition with fewer than
        #   this many real clusters is treated as degenerate (a collapsed/
        #   under-fit fit, not a meaningful partition to compare against
        #   others) and excluded from the mean pairwise ARI in BOTH
        #   seed_sweep_stability() and consensus_cluster()'s own per-seed
        #   stability stats. Comparing a degenerate partition (e.g. 1
        #   cluster) against a well-formed one is guaranteed near-zero ARI
        #   regardless of whether the genuine structure is actually stable,
        #   so leaving them in systematically drags the mean down and can
        #   spuriously trip consensus_auto_threshold. Default 6 (a partition
        #   with fewer than 6 clusters is not a trustworthy comparison point
        #   for datasets in this pipeline's typical 8-30 preferred-cluster
        #   range). Degenerate seeds are still logged and counted, just
        #   excluded from the ARI mean itself.
        seed_sweep_min_valid_clusters = 6,
        # ── Consensus/co-association clustering (opt-in, Aug 2026) ────────────
        # consensus_clustering_enabled: False (default) = current behavior,
        #   the primary run_hdbscan() fit on umap_random_state's single seed
        #   is used unchanged. True replaces it with a partition built from
        #   agreement across consensus_n_seeds independent seeds (see
        #   consensus_cluster()'s docstring) -- the general fix for datasets
        #   where seed_sweep_stability reports low mean_ari (UMAP embedding
        #   topology itself is seed-unstable, not just HDBSCAN's selection).
        #   Costs roughly consensus_n_seeds x the primary UMAP+HDBSCAN
        #   runtime, so it's opt-in rather than a new default for everyone.
        # consensus_n_seeds: how many seeds to build the co-association
        #   matrix from. 8 balances a sturdier consensus against runtime;
        #   the validated test used 12.
        # consensus_n_jobs: dispatch of consensus_cluster()'s per-seed
        #   UMAP+HDBSCAN fits via joblib (thread-based pool, same choice as
        #   seed_sweep_n_jobs/hdbscan_split_n_jobs). Default 1 (sequential)
        #   alongside seed_sweep_n_jobs -- same Aug 2026 safety margin
        #   (see seed_sweep_n_jobs above for the full rationale). Also
        #   measured at no real cost: an 8-seed consensus run timed 668s
        #   sequential vs 692s parallel. -1 = auto-managed (System
        #   Resources block below); >1 = pin an exact worker count.
        # consensus_linkage: "ward" (default; validated -- gave well-
        #   separated, balanced clusters). "average" and "complete" were
        #   both tested and collapsed to a single giant cluster on real data
        #   (they chain through the noisy co-association matrix) -- avoid.
        consensus_clustering_enabled = False,
        consensus_n_seeds            = 8,
        consensus_n_jobs             = 1,
        consensus_linkage            = "ward",
        # consensus_max_memory_gb: safety guard on the O(n_training_bins^2)
        #   co-association matrix (float32) -- ~200MB at 7,000 bins, ~1.6GB at
        #   20,000, ~10GB at 50,000. Checked BEFORE the n_seeds x UMAP+HDBSCAN
        #   loop runs, so an oversized dataset aborts immediately (falls back
        #   to the primary single-seed result) instead of burning that
        #   runtime and only then failing at allocation. 0 = no limit.
        consensus_max_memory_gb      = 4.0,
        # consensus_refine_enabled: ON by default (Aug 2026) -- when
        #   consensus clustering runs, the PRIMARY path's split/merge
        #   refinement pass is unconditionally skipped for its output (its
        #   split step assumes hdb_clf/embedding come from the same fit as
        #   the labels being refined, which isn't true for consensus; see
        #   the _consensus_used branch in run()). Without this on, consensus
        #   labels get NO impurity screening at all -- confirmed on a real
        #   run to leave a large (3,142-bin), badly-separated catch-all
        #   cluster completely unsplit. When True, consensus_cluster()'s
        #   output instead goes through refine_consensus_clusters() -- split
        #   reuses split_impure_clusters() unchanged
        #   (hdbscan_split_silhouette_thresh/recluster_max_iterations, same
        #   keys as the primary path); merge uses merge_by_coassociation()
        #   (consensus_merge_coassoc_thresh) instead of
        #   merge_similar_clusters() (needs a condensed_tree_, which
        #   consensus partitions don't have). See consensus_cluster()'s
        #   docstring (Aug 2026). Costs an extra split/merge pass on top of
        #   consensus's own n_seeds x UMAP+HDBSCAN runtime; set False to
        #   restore the old "consensus output taken as-is" behavior.
        consensus_refine_enabled     = True,
        # consensus_merge_coassoc_thresh: mean cross-cluster co-association
        #   (fraction of the n_seeds that grouped two clusters' bins together)
        #   above which merge_by_coassociation() merges them. Only active
        #   when consensus_refine_enabled=True. 0.5 is a reasoned starting
        #   value ("more than half the seeds agreed"), not yet empirically
        #   calibrated -- same status hdbscan_overshoot_penalty had at
        #   ship time. <= 0 disables the merge half specifically (split can
        #   still run on its own).
        consensus_merge_coassoc_thresh = 0.5,
        # consensus_auto_threshold: if seed_sweep_n>=2 (on by default) and the
        #   resulting mean_ari falls below this, consensus clustering is
        #   auto-enabled even though consensus_clustering_enabled defaults to
        #   False -- UNLESS the caller explicitly set
        #   consensus_clustering_enabled=False (always respected as a
        #   deliberate opt-out). Set to 0 to disable auto-triggering entirely
        #   (matches "warn only, never auto-switch" behavior); consensus
        #   clustering remains available via consensus_clustering_enabled=True.
        #   Re-enabled at 0.55 (Aug 2026): briefly disabled after
        #   hdbscan_selection_mode="floor_soft_cap" (see
        #   CLUSTER_SELECTION_SIMPLIFICATION_PLAN) fixed catastrophic
        #   undershoot, on the theory that seed-sweep ARI would now be
        #   consistently stable enough not to need it. A real 4-config
        #   A/B test on a 21-session combined dataset (baseline / down-
        #   weighted bodyparts / umap_n_neighbors=60 / both) showed mean ARI
        #   still landing in 0.18-0.29 across every config -- nowhere close
        #   to stable. 0.55 sits just under the old 0.6 bar (itself
        #   calibrated against a real mean_ari=0.512 case that needed to
        #   trigger) with a small margin so runs already close to reliable
        #   don't pay consensus's ~n_seeds x runtime cost unnecessarily.
        #   Consensus itself is now also less risky to auto-fire than when
        #   0.6 was chosen: it gained its own opt-in post-hoc split/merge
        #   refinement (consensus_refine_enabled) and always reports
        #   feature-space DBCV/silhouette directly comparable to the primary
        #   path's, instead of being a refinement/validation-free black box
        #   scored only by separation_ratio.
        consensus_auto_threshold     = 0.55,
        # kinematic_directedness_enabled (v6 K2, Kinematic_Transition_v6_
        #   Implementation_Plan.md): off by default, opt-in only. When True,
        #   writes an additional per-session sidecar CSV,
        #   <stem>_bout_lengths_hmm_enriched.csv, alongside the canonical
        #   *_bout_lengths_hmm.csv -- the canonical 3-column B-SOiD-format
        #   file plus five new per-bout directedness columns from
        #   compute_bout_directedness() (net_displacement_px, path_length_px,
        #   straightness_ratio, mean_speed_px_s, heading_consistency). The
        #   canonical bout CSV itself is never touched, in either state of
        #   this flag -- when False (default), no sidecar is written at all
        #   (not written empty), and every existing output file is
        #   byte-identical to pre-v6 output.
        kinematic_directedness_enabled = False,
        # ── Body-region feature weighting (issue 1b) ──────────────────────────
        # bodypart_weights: {bodypart_name: multiplier}.  {} (default) = every
        #   multiplier is exactly 1.0 -> bit-identical output to unweighted
        #   behaviour.  Off by default: unlike issue 2's visibility features,
        #   this requires domain judgment about which regions matter and must
        #   not silently change existing runs.
        bodypart_weights       = {},
        # auto_bodypart_weighting: data-driven, automatic COMPLEMENT to the
        #   manual bodypart_weights above.  ON by default (unlike manual
        #   weighting) -- this doesn't require domain judgment, it purely
        #   reads each bodypart's own aggregate bad-frame fraction across
        #   sessions and tapers its weight down continuously for bodyparts
        #   that are chronically unreliable but not bad enough to hard-drop
        #   -- those otherwise enter feature extraction at full weight,
        #   contributing flat-interpolated near-duplicate vectors that
        #   inflate HDBSCAN noise and push DBCV toward degenerate.  An
        #   explicit entry in bodypart_weights for a given bodypart always
        #   wins over the auto-computed one. False reverts to pre-existing
        #   behaviour (manual weights only, or uniform if none set).
        auto_bodypart_weighting = True,
        # auto_bp_weight_session_thresh: per-session bad-frame-fraction
        #   threshold above which that session counts as "affected" for a
        #   bodypart (reuses the same 0.3 convention as the DLC-quality
        #   warning gate). auto_bp_weight_lo/_hi (below) then taper on the
        #   FRACTION OF SESSIONS affected, not a magnitude statistic --
        #   robust to both failure modes seen on real data: a mean-based
        #   stat dilutes a bodypart bad (42-67%) in only 5 of 21 sessions
        #   down to mean=0.157 (missed entirely), while a magnitude
        #   percentile/max-based stat over-corrects, since with ~20 sessions
        #   almost every bodypart has at least one session with a bad-frame
        #   spike (confirmed: 16 of 22 bodyparts flagged, mostly jaws/paws
        #   driven by a single high-turned-away session). Counting sessions
        #   affected catches the former (5/21 = 24% of sessions, well above
        #   auto_bp_weight_lo) without over-triggering on the latter (a
        #   single fluke session stays under auto_bp_weight_lo). The
        #   turned-away-driven confound itself is separately handled by
        #   turned_away_exclude_from_bad_frac below.
        auto_bp_weight_session_thresh = 0.3,
        # auto_bp_weight_lo/_hi: linear taper bounds on the FRACTION OF
        #   SESSIONS affected (see auto_bp_weight_session_thresh) above.
        #   <= lo -> untouched (weight 1.0); >= hi -> weight hits
        #   auto_bp_weight_floor; linear in between. lo=0.10 (~2/21
        #   sessions) filters one-off flukes; hi=0.35 (~7/21 sessions) is
        #   comfortably below "bad in most sessions" (which feature_bad_bp_
        #   thresh[_with_visibility]'s hard-drop gate, above, already
        #   handles).
        auto_bp_weight_lo      = 0.10,
        auto_bp_weight_hi      = 0.35,
        # auto_bp_weight_floor: minimum auto-computed weight (never fully
        #   zeroes a bodypart out via this path -- full removal is
        #   feature_bad_bp_thresh[_with_visibility]'s job, above).
        auto_bp_weight_floor   = 0.35,
        # turned_away_exclude_from_bad_frac: when True (default), each
        #   bodypart's per-session bad-frame fraction (used by both the
        #   FEAT-DROP hard-drop gate and auto-weighting above) is recomputed
        #   excluding frames already flagged turned-away-from-camera, before
        #   either gate sees it. Raw per-frame likelihood naturally drops
        #   for face/forepaw bodyparts during turned-away moments -- they're
        #   genuinely out of camera view -- but those exact frames are
        #   already excluded from UMAP/HDBSCAN training separately (Step 3),
        #   so counting them as "bad tracking" here double-penalises the
        #   bodypart for something that isn't chronic mistracking. False
        #   reverts to the raw (uncorrected) fraction -- also the
        #   compat_mode="legacy_v2" default, since this changes what
        #   FEAT-DROP drops (pre-existing behaviour, not new this session).
        turned_away_exclude_from_bad_frac = True,
        # ── Adaptive visibility / occlusion features (issue 2) ────────────────
        # visibility_features_enabled: ON by default (unlike bodypart_weights)
        #   — this is the direct fix for turned-away/occluded frames polluting
        #   real clusters, of the same "always-on diagnostic" kind as the
        #   existing f_withinbin/f_persist blocks.  False reproduces the
        #   pre-2.2 feature layout exactly (legacy reproducibility escape
        #   hatch, same pattern as compat_mode/angular_fallback).
        visibility_features_enabled = True,
        # visibility_adaptive_pct: per-bodypart/per-session percentile floor
        #   (see compute_adaptive_visibility_threshold) layered on top of the
        #   existing global likelihood_thresh constant.  Always active
        #   whenever the visibility feature block runs (no separate toggle —
        #   it is a pure function of each session's own likelihood
        #   distribution, so gating it independently would add a no-benefit
        #   toggle).
        visibility_adaptive_pct = 10,
        # ── Turned-away-from-camera detection/exclusion (v3, validated against
        #    real DLC data + human video review this session) ─────────────────
        # exclude_turned_away: ON by default — bins where the animal is judged
        #   turned away from the camera (see detect_turned_away_bins) are
        #   excluded from UMAP/HDBSCAN training and given a dedicated "Turned
        #   Away" label instead of being force-classified into a real
        #   behaviour cluster.  False disables the exclusion/dedicated-label
        #   behaviour (falls back to the pre-existing visibility-feature-only
        #   handling — HDBSCAN may or may not naturally separate these bins on
        #   its own, no forced intervention); this is the GUI-exposed escape
        #   hatch and, at False, reproduces pre-existing clustering output
        #   exactly (true no-op path, same pattern as bodypart_weights={}).
        exclude_turned_away     = True,
        # turned_away_conf_thresh: the Head/Mouth region frac_low_conf cutoff
        #   (head_frac_on) above which a bin becomes a turned-away candidate,
        #   subject to the nose-corroboration AND-gate and window debouncing
        #   below.  0.30 is the validated "sensitive/amber" threshold from
        #   this session's three-pass corroboration against real video —
        #   GUI-editable.  Reuses the existing likelihood_thresh and
        #   visibility_adaptive_pct cfg keys for the underlying adaptive-
        #   threshold machinery (no duplicate threshold logic).
        turned_away_conf_thresh = 0.30,
        # turned_away_min_window_s: sustained-window duration floor (seconds)
        #   below which a flagged run of bins is treated as jitter and
        #   dropped, rather than a genuine turn-away.  Not GUI-exposed
        #   (validated value); editable via engine_cfg for advanced use.
        turned_away_min_window_s = 0.4,
        # turned_away_merge_gap_s: two flagged windows separated by a gap of
        #   at most this many seconds are merged into one window before the
        #   min-window-s floor is applied — bridges brief drop-outs within a
        #   single sustained turn-away.  Not GUI-exposed (validated value);
        #   editable via engine_cfg for advanced use.
        turned_away_merge_gap_s = 0.5,
        # auto_flag_impure_clusters: OFF by default (Aug 2026) -- deliberately
        #   opt-in, unlike most refinement passes in this file. After split/
        #   merge refinement + rare-cluster pruning, when enabled, any real
        #   cluster whose mean per-bin silhouette (in the primary UMAP
        #   embedding) is STILL below auto_flag_impure_silhouette_thresh is
        #   folded into the same reserved "Turned Away" display id as
        #   exclude_turned_away's confidence-based detector, instead of being
        #   presented as a real behaviour. Catches the case
        #   split_impure_clusters can't: a heterogeneous catch-all cluster
        #   that (a) split was never attempted on -- e.g.
        #   consensus_clustering_enabled=True with consensus_refine_enabled=
        #   False skips the whole split/merge pass for consensus labels (see
        #   the _consensus_used branch above run()) -- or (b) split WAS
        #   attempted but no stable local sub-partition was found, so the
        #   impure cluster was left untouched by design (split never forces a
        #   split).
        #   Why opt-in: this is a purely GEOMETRIC check (cluster shape in
        #   the embedding) and cannot distinguish a genuine artifact cluster
        #   (what it's meant to catch -- e.g. animal turned away but
        #   paws/tail stay confidently tracked, so the confidence-based
        #   turned-away detector's Head/Mouth-region likelihood never drops
        #   enough to trip turned_away_conf_thresh) from a real but
        #   naturally diverse behaviour that legitimately has a mediocre
        #   silhouette. Auto-relabeling the latter as "Turned Away" would
        #   silently discard real data. GUI-exposed under Advanced Settings;
        #   inspect a flagged cluster's example clips before trusting the
        #   label. Bins are excluded from the UMAP/transition plots and
        #   dwell-time stats the same way confidence-based turned-away bins
        #   are, but are NOT excluded from MLP training (unlike confidence-
        #   based turned-away bins) -- the cluster stays a real, consistently
        #   -predicted class internally; only the DISPLAY id is remapped, per
        #   session, at export time. No-op (and skips the whole pass, with a
        #   log line) when exclude_turned_away=False, since there is no
        #   reserved id to route into.
        auto_flag_impure_clusters = False,
        # auto_flag_impure_silhouette_thresh: mean-silhouette cutoff for the
        #   post-hoc flagging pass above. Falls back to
        #   hdbscan_split_silhouette_thresh (the same quality bar split
        #   candidacy uses) when left at its default 0 -- set explicitly to
        #   decouple the two thresholds.
        auto_flag_impure_silhouette_thresh = 0.0,
        # ── Hierarchical/consensus refinement (issue 4 — bidirectional) ───────
        # hdbscan_merge_thresh: condensed-tree sibling-merge persistence-
        #   fraction cutoff.  0.08 (default, on) merges sibling clusters that
        #   split at <=8% of the tree's max lambda_val (i.e. barely-separated
        #   siblings — the code's own docstring calls 0.05 "barely
        #   separated"; 0.08 gives a little more headroom given the
        #   over-fragmentation seen in practice, e.g. a single "licking"
        #   behaviour split across 3 clusters, "sniffing" across 7, in run
        #   cube_results_20260618_015149 which had this pass disabled).  Set
        #   to 0.0 to fully disable (hard no-op), or pass compat_mode=
        #   "legacy_v2" to reproduce pre-refinement runs exactly.
        hdbscan_merge_thresh    = 0.08,
        # hdbscan_leaf_bonus: added to every leaf-method candidate's score
        #   ONLY once hdbscan_merge_thresh > 0 (leaf's extra fragmentation is
        #   then self-corrected by the merge pass).  Never applied when the
        #   merge pass is off — eom/leaf selection is then byte-for-byte
        #   unchanged from pre-issue-4 behaviour.
        hdbscan_leaf_bonus      = 0.03,
        # hdbscan_fine_bias: only active once hdbscan_merge_thresh > 0 (same
        #   gate as hdbscan_leaf_bonus). Nudges auto-mode candidate selection
        #   toward the finer end of [preferred_clusters_lo, preferred_
        #   clusters_hi] instead of always settling on the coarsest DBCV
        #   peak in range — biases toward "enough clusters to separate
        #   distinct behaviours, let the merge pass consolidate near-
        #   duplicates afterward" rather than "fewest clusters that still
        #   score well". Never applied when the merge pass is off.
        hdbscan_fine_bias       = 0.05,
        # hdbscan_split_silhouette_thresh: mean per-cluster silhouette below
        #   which a cluster is a candidate for local re-clustering (impurity
        #   fix).  0.2 (default, on) only catches clusters well below a
        #   healthy silhouette (the same run above reported an *aggregate*
        #   silhouette of ~0.49 with ARI 0.59 seed-instability, so 0.2 is
        #   conservative — it targets clearly-impure clusters, not ones
        #   merely below average).  None fully disables the split pass
        #   (hard no-op).
        hdbscan_split_silhouette_thresh = 0.2,
        # hdbscan_split_max_subclusters: hard cap on how many sub-clusters a
        #   single split_impure_clusters() candidate may accept in one pass
        #   (also used as the local re-clustering's preferred_clusters_hi,
        #   see split_impure_clusters -- the whole-session preferred range
        #   and fine_bias are deliberately NOT inherited for this local
        #   call). Kept deliberately low (3): a split is meant to resolve
        #   the handful (typically 2-3) of genuinely distinct sub-behaviours
        #   mixed into one impure cluster, not fragment it extensively --
        #   without this cap, a single cluster could split into 20-30+ tiny
        #   fragments in one pass (seen in practice on real data once
        #   fine_bias was pushing local selection toward a whole-session-
        #   scale cluster count).
        hdbscan_split_max_subclusters = 3,
        # hdbscan_split_min_points: minimum points a candidate cluster must
        #   have before a local split re-embedding is even attempted.  Not
        #   a flat legacy value (20) -- sized at 5x the PCA floor (50
        #   components, see run_umap) that a high-dimensional local subset
        #   would be reduced to, since a sample/feature ratio below ~5 is
        #   the same curse-of-dimensionality regime run_umap's own auto-PCA
        #   trigger is built to avoid. Feature counts routinely exceed
        #   500-900 dims (more bodyparts -> quadratically more pairwise-
        #   distance features), so a flat 20-point floor let splits run on
        #   90-300 point subsets in practice -- far too few to trust the
        #   resulting sub-clusters or local DBCV score.
        hdbscan_split_min_points = 250,
        # ── Split-pass performance bounds (Aug 2026) ──────────────────────────
        # split_impure_clusters() used to run a full 40-step x 2-method
        # HDBSCAN sweep per impure-cluster candidate -- fine for one candidate,
        # combinatorially expensive when a partition has many (confirmed on
        # real data: one seed took 25+ min vs 1-2 min for others). These three
        # keys bound that cost; see split_impure_clusters()'s docstring.
        # hdbscan_split_max_candidates: hard ceiling on candidates attempted
        #   per refinement iteration (worst-silhouette-first).
        # hdbscan_split_candidate_cutoff: 0/unset = auto (split_silhouette_
        #   thresh / 2) -- a stricter silhouette bar than the base split gate,
        #   applied before the max_candidates ceiling. Only clusters at or
        #   below this silhouette attempt a split by default.
        # hdbscan_split_sweep_n_steps: mcs steps for each candidate's LOCAL
        #   sweep (default 12, vs 40 for the primary whole-session sweep) --
        #   a few-hundred-point local subset doesn't need fine resolution to
        #   find 2-3 sub-clusters. hdbscan_method is also forced to "eom"
        #   only for these local sweeps (skips the "leaf" pass).
        # hdbscan_split_n_jobs: parallel workers for the (independent,
        #   disjoint-point) candidate loop. -1 = auto-managed (see System
        #   Resources block below; falls back to literal "all cores" when
        #   auto_resource_management is off); 1 = sequential (exact
        #   pre-Aug-2026 candidate-processing order, minus the other two
        #   bounds above).
        hdbscan_split_max_candidates    = 10,
        hdbscan_split_candidate_cutoff  = 0,
        hdbscan_split_sweep_n_steps     = 12,
        hdbscan_split_n_jobs            = -1,
        # recluster_max_iterations: cap on the split -> merge -> repeat
        #   refinement loop (both passes are individually gated above).
        #   2 = one split pass to resolve impurity, one merge pass to
        #   consolidate resulting fragmentation / pre-existing near-
        #   duplicates.
        recluster_max_iterations = 2,

        # ── System Resources (Aug 2026 perf fix) ───────────────────────────────
        # The primary HDBSCAN min_cluster_size sweep (run_hdbscan, ~40 steps x
        # up to 2 methods) used to run as a plain sequential loop -- most cores
        # sat idle for the majority of a run's wall-clock time. These keys make
        # it (and hdbscan_split_n_jobs / seed_sweep_n_jobs above/below) adaptive:
        # -1 on any of those three now means "auto-managed" rather than literal
        # "all cores" -- resolved once per run via compute_adaptive_n_jobs(),
        # which targets system_resource_target_pct of logical cores, hard-capped
        # at system_resource_cap_pct, and shrinks further if RAM is already
        # under pressure. Explicit non--1 values on those three keys (1 =
        # sequential, or a user-pinned count) are unaffected -- this only
        # changes the *default*/auto behaviour.
        auto_resource_management        = True,
        system_resource_target_pct      = 0.65,   # ideal sustained (60-70% band)
        system_resource_cap_pct         = 0.80,   # hard ceiling, never exceeded
        # hdbscan_sweep_n_jobs: default changed -1 -> 1 (sequential) Aug 2026
        # after a real crash_diagnostics.log capture of this exact call site
        # (run_hdbscan's primary sweep, not seed_sweep/consensus) faulting with
        # "Windows fatal exception: code 0xc0000374" (heap corruption) despite
        # the _numba_single_thread/_blas_single_thread_for_dispatch
        # oversubscription guards around _fit_one. seed_sweep_n_jobs and
        # consensus_n_jobs were already made sequential for the identical
        # crash signature -- this brings the primary sweep in line with those
        # two rather than leaving it as the one remaining parallel call site
        # with the same underlying hazard (joblib threading backend + numba
        # JIT + HDBSCAN's Cython core all contending under threads on
        # Windows). Not reliably reproducible on demand, so this is a safety
        # margin rather than a pinned-down fix, matching the rationale
        # already applied to seed_sweep_n_jobs/consensus_n_jobs.
        hdbscan_sweep_n_jobs            = 1,
    )

    # Pre-2.1 numeric defaults, restored when cfg["compat_mode"] == "legacy_v2"
    # for keys the caller did not set explicitly.  Keep in sync with the
    # corrected DEFAULTS above.
    _LEGACY_V2_DEFAULTS = dict(
        umap_min_dist         = 0.0,
        hdbscan_mcs_anchor    = "full",
        angular_fallback      = True,
        # Split/merge refinement was added after v2.1 and defaulted to off;
        # legacy_v2 reproduces pre-refinement runs exactly unless the caller
        # explicitly overrides these.
        hdbscan_merge_thresh             = 0.0,
        hdbscan_split_silhouette_thresh  = None,
        # Automatic confidence-based bodypart weighting was added after v2.1
        # and defaulted to off in earlier runs; same reproducibility escape
        # hatch as above.
        auto_bodypart_weighting          = False,
        # Turned-away-frame exclusion from bad-frame-fraction stats changes
        # what the pre-existing FEAT-DROP hard-drop gate drops; keep it off
        # so legacy_v2 reproduces older runs' drop decisions exactly.
        turned_away_exclude_from_bad_frac = False,
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
        # Which keys the caller explicitly passed (vs. inherited from
        # DEFAULTS) -- used by compat_mode="legacy_v2" below, and by the
        # consensus-clustering auto-trigger (Aug 2026) to distinguish a
        # deliberate consensus_clustering_enabled=False opt-out from just
        # never having set it.
        self._explicit_cfg_keys = set((cfg or {}).keys())
        # Reproducibility: in legacy mode, revert v2.1 numeric changes for any
        # key the caller did not pass explicitly (explicit overrides win).
        if self._cfg.get("compat_mode") == "legacy_v2":
            for _k, _v in self._LEGACY_V2_DEFAULTS.items():
                if _k not in self._explicit_cfg_keys:
                    self._cfg[_k] = _v

        # sub-dirs
        self._out_bouts  = self.output_dir / "bout_lengths"
        self._out_model  = self.output_dir / "model"
        self._out_plots  = self.output_dir / "plots"
        self._out_videos = self.output_dir / "videos"
        for d in (self._out_bouts, self._out_model,
                  self._out_plots, self._out_videos):
            d.mkdir(parents=True, exist_ok=True)

    def _mem_checkpoint(self, label: str):
        """gc.collect() + psutil RSS sample at a pipeline stage boundary.
        Tracks a running peak in self._peak_rss_gb (psutil's RSS captures real
        process memory including numpy/BLAS/native temporaries, unlike the
        tracemalloc-based peak_memory_gb metric in run(), which only sees the
        Python heap). Purely diagnostic + cleanup -- never affects results."""
        gc.collect()
        try:
            import psutil
            rss_gb = psutil.Process().memory_info().rss / 1e9
            self._peak_rss_gb = max(getattr(self, "_peak_rss_gb", 0.0), rss_gb)
            self._log(f"  [MEMORY] {label}: RSS={rss_gb:.2f} GB "
                      f"(peak {self._peak_rss_gb:.2f} GB)")
        except Exception:
            pass

    #   main entry point

    def run(self) -> dict:
        """Run the full V2 pipeline. Returns a results dict."""
        # Track total wall-clock runtime and peak memory for the publication
        # benchmark metrics (Section 3.2: Runtime Efficiency + Peak Memory).
        import tracemalloc as _tm
        _tm.start()
        _run_t0 = time.perf_counter()

        # Apply plot theme before any figure is drawn (updates _BG/_PANEL/_TEXT_COL/_TICK_COL)
        _apply_plot_theme(self._cfg.get("plot_theme", "light"))

        self._log("=" * 64)
        self._log(f"  CUBE Engine  v{VERSION}")
        self._log("=" * 64)

        # ── System resources: detect once, report the resolved parallelism
        # budget for the HDBSCAN-side stages (sweep/split/seed-sweep). Actual
        # resolution happens per-call via resolve_n_jobs()/compute_adaptive_
        # n_jobs() -- this is a preview so the auto-managed behaviour is
        # visible rather than silent, and each heavy stage re-checks RAM
        # pressure independently right before it dispatches.
        self._peak_rss_gb = 0.0
        try:
            _res = detect_system_resources()
            self._log(f"\n[SYSTEM] {_res['cpu_count']} logical cores  |  "
                      f"RAM {_res['available_ram_gb']:.1f}/{_res['total_ram_gb']:.1f} GB "
                      f"available ({_res['ram_used_pct']:.0f}% used)")
            if bool(self._cfg.get("auto_resource_management", True)):
                _budget = compute_adaptive_n_jobs(self._cfg, log_fn=self._log)
                self._log(f"  [SYSTEM] auto-managed parallel budget: {_budget} "
                          f"worker(s)  (target "
                          f"{float(self._cfg.get('system_resource_target_pct', 0.65)):.0%}"
                          f" of cores, cap "
                          f"{float(self._cfg.get('system_resource_cap_pct', 0.80)):.0%})"
                          f"  — applies to the HDBSCAN sweep/split/seed-sweep "
                          f"stages; UMAP embedding stays single-threaded for "
                          f"reproducibility.")
            else:
                self._log("  [SYSTEM] auto_resource_management is OFF — "
                          "using explicit *_n_jobs cfg values as-is.")
        except Exception:
            self._log("  [SYSTEM] resource detection unavailable (psutil "
                      "missing?) — falling back to explicit *_n_jobs cfg values.")

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

        # Detect 3D mode from the first available DLC file
        _is_3d = _h5_has_z(pairs[0][0]) if pairs else False
        if _is_3d:
            self._log("  [3D] Detected 3D H5 files — using 3D feature extraction "
                      "(true Euclidean distances + velocities in world space)")
        _nc = 3 if _is_3d else 2   # columns per bodypart: 3 for xyz, 2 for xy

        # 2. Load & smooth
        self._log("\n[2/7]  Loading & smoothing...")
        self._stage("2/7 — Loading DLC files", f"0/{len(pairs)}")
        all_xy, all_names, all_fps_list, all_bps, all_vpaths = [], [], [], [], []
        all_groups: list = []   # input folder each session came from (for export coverage)
        all_bp_bad_fracs: dict = {}  # bodypart -> list of per-session bad-frame fracs
        all_flat_held: list = []     # per-session per-bodypart flat-held masks (list[list[np.ndarray]])
        all_ll: list = []            # per-session raw per-frame likelihood arrays (issue 2 visibility features)

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
                xy, bps, fps_hint, ll_fracs, _flat_held, _ll = load_dlc_file(
                    fp, self._cfg["likelihood_thresh"],
                    max_interp_gap_frames=_max_gap, log_fn=self._log,
                    return_quality=True, include_z=_is_3d)
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
                all_ll.append(_ll)
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
                        col_idx.extend([_nc * j + c for c in range(_nc)])
                all_xy[k] = xy_k[:, col_idx]
                # Mirror the same filtering on the raw likelihood arrays (1 col
                # per bodypart, not _nc-per-bodypart) so all_ll stays column-
                # aligned with bps_ref for the visibility feature block (issue 2).
                if k < len(all_ll) and all_ll[k] is not None and all_ll[k].size:
                    _ll_idx = [bps_k.index(bp) for bp in bps_ref if bp in bps_k]
                    all_ll[k] = all_ll[k][:, _ll_idx]

        # Exclude turned-away-from-camera frames from all_bp_bad_fracs before
        # it feeds either the FEAT-DROP gate (_agg, below) or the auto-weight
        # block further down. all_bp_bad_fracs is raw per-frame-likelihood
        # based (built in Step 2, load_dlc_file) and knows nothing about
        # turned-away frames -- detect_turned_away_bins only runs later, in
        # Step 3, purely for exclusion-from-training/labelling. A bodypart
        # that's only invisible DURING turned-away moments (jaws, forepaws)
        # isn't actually chronically mistracked -- those exact frames are
        # already excluded from UMAP/HDBSCAN training separately, so counting
        # them here just double-penalises the bodypart, and with a single
        # high-turned-away session can make an otherwise well-tracked
        # bodypart look bad enough to trigger both FEAT-DROP and
        # auto-weighting (confirmed on real data: jaw/forepaw bodyparts
        # flagged by auto-weighting turned out to be driven by one session
        # with 34.7% turned-away time, not chronic mistracking). Recompute
        # each bodypart's per-session bad-frame fraction using only
        # non-turned-away frames. Gated so compat_mode="legacy_v2"
        # reproduces older runs exactly (FEAT-DROP's drop set is pre-existing
        # behaviour, not new this session).
        if (bool(self._cfg.get("turned_away_exclude_from_bad_frac", True))
                and all_bp_bad_fracs and "nose" in [str(b).lower() for b in bps_ref]):
            _ta_vis_pct = float(self._cfg.get("visibility_adaptive_pct", 10))
            _ta_conf_thresh = float(self._cfg.get("turned_away_conf_thresh", 0.30))
            _ta_min_window_s = float(self._cfg.get("turned_away_min_window_s", 0.4))
            _ta_merge_gap_s = float(self._cfg.get("turned_away_merge_gap_s", 0.5))
            _ll_thresh = float(self._cfg["likelihood_thresh"])
            _n_corrected = 0
            for k in range(len(all_ll)):
                _llk = all_ll[k]
                if _llk is None or not getattr(_llk, "size", 0):
                    continue
                _fps_k = float(all_fps_list[k]) if k < len(all_fps_list) else 30.0
                try:
                    _ta_bin_mask = detect_turned_away_bins(
                        _llk, bps_ref, _fps_k, _ll_thresh, _ta_vis_pct,
                        _ta_conf_thresh, _ta_min_window_s, _ta_merge_gap_s)
                except Exception:
                    continue
                if _ta_bin_mask.size == 0 or not _ta_bin_mask.any():
                    continue
                _win = max(1, int(round(_fps_k / 10)))
                _ta_frame_mask = np.repeat(_ta_bin_mask, _win)
                # Pad (trailing partial-bin remainder frames, treated as not
                # turned-away) or truncate to exactly match _llk's frame
                # count -- np.repeat's length is n_bins*_win, which can be
                # shorter OR longer than _llk.shape[0] depending on rounding.
                if _ta_frame_mask.shape[0] < _llk.shape[0]:
                    _ta_frame_mask = np.concatenate([
                        _ta_frame_mask,
                        np.zeros(_llk.shape[0] - _ta_frame_mask.shape[0], dtype=bool)])
                else:
                    _ta_frame_mask = _ta_frame_mask[:_llk.shape[0]]
                _keep = ~_ta_frame_mask
                if not _keep.any():
                    continue
                for j, bp in enumerate(bps_ref):
                    if bp not in all_bp_bad_fracs or k >= len(all_bp_bad_fracs[bp]):
                        continue
                    all_bp_bad_fracs[bp][k] = float(np.mean(_llk[_keep, j] < _ll_thresh))
                    _n_corrected += 1
            if _n_corrected:
                self._log(
                    f"  [TURNED-AWAY-CORRECT] adjusted bad-frame fraction for "
                    f"{_n_corrected} bodypart-session pair(s) to exclude "
                    f"turned-away frames (prevents turned-away spikes from "
                    f"masquerading as chronic mistracking).")

        # Drop chronically occluded bodyparts from the feature space.
        # At this point all_xy[k] columns are ordered by bps_ref, so we can
        # index directly.  Bodyparts whose mean bad-frame fraction across
        # sessions meets or exceeds feature_bad_bp_thresh are removed; they
        # contribute flat-interpolated artefacts (identical feature vectors)
        # that inflate HDBSCAN noise and cause DBCV to become non-finite.
        # When visibility_features_enabled=True (default), the much higher
        # feature_bad_bp_thresh_with_visibility bar applies instead -- with
        # visibility features on, transient/periodic occlusion (e.g. limb
        # joints during normal movement, animal turned away from camera) is
        # isolated via the visibility feature block rather than requiring
        # wholesale bodypart removal, so only near-unusable bodyparts need
        # dropping here. See DEFAULTS for the calibration rationale.
        _vis_enabled_for_drop = bool(self._cfg.get("visibility_features_enabled", True))
        _feat_thresh = float(self._cfg.get(
            "feature_bad_bp_thresh_with_visibility" if _vis_enabled_for_drop
            else "feature_bad_bp_thresh",
            0.70 if _vis_enabled_for_drop else 0.40))
        if all_bp_bad_fracs:
            _agg = {bp: float(np.mean(all_bp_bad_fracs.get(bp, [0.0])))
                    for bp in bps_ref}
        else:
            _agg = {}
        if _feat_thresh > 0 and _agg:
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
                    _col_idx.extend([_nc * i + c for c in range(_nc)])
                for k in range(len(all_xy)):
                    all_xy[k] = all_xy[k][:, _col_idx]
                for k in range(len(all_ll)):
                    if all_ll[k] is not None and all_ll[k].size:
                        all_ll[k] = all_ll[k][:, _keep_pos]
                _agg = {bp: _agg[bp] for bp in bps_ref}

        # Automatic confidence-based bodypart down-weighting (data-driven,
        # complements the hard FEAT-DROP gate above). Bodyparts that survive
        # the drop but are still chronically unreliable still contribute
        # flat-interpolated, near-duplicate feature vectors at full weight,
        # which is exactly what pushes HDBSCAN noise up and DBCV toward
        # degenerate -- FEAT-DROP alone only catches the most extreme cases
        # (default threshold 0.70 with visibility features on). This tapers
        # weight continuously instead of a second hard cutoff, so
        # moderately-bad bodyparts are down-weighted rather than either
        # fully trusted or fully discarded.
        #
        # Uses the fraction of SESSIONS a bodypart is genuinely bad in
        # ("chronic" = affects a recurring subset of sessions), NOT the mean
        # used by the FEAT-DROP gate above and NOT a percentile-of-magnitude
        # statistic. A bodypart that is severely bad (e.g. 50-67%) in a
        # handful of sessions but fine in most others has its MEAN diluted
        # below any reasonable threshold -- observed in practice on real
        # data: the exact bodyparts named as the worst-tracked in individual
        # sessions (back_right_knee, back_middle, throat_base/end, etc.)
        # never appeared in the [AUTO-WEIGHT] line at all, because their
        # session-mean landed under auto_bp_weight_lo even though specific
        # sessions were badly affected. A percentile-of-magnitude statistic
        # (e.g. max) "fixes" that but over-corrects: with ~20 sessions,
        # almost every bodypart has at least one session with a bad-frame
        # spike (often driven by turned-away moments, though that specific
        # confound is now handled upstream via
        # turned_away_exclude_from_bad_frac), so a max-based stat flags most
        # bodyparts rather than the genuinely chronic ones (confirmed on
        # real data: 16 of 22 bodyparts flagged, mostly jaws/paws driven by
        # a single high-turned-away session). Counting SESSIONS AFFECTED
        # (bad_frac > auto_bp_weight_session_thresh in that session) instead
        # of a magnitude statistic is robust to both problems: a bodypart
        # bad in only 1-2 sessions (a fluke) stays under auto_bp_weight_lo
        # untouched, while one bad in a recurring subset of sessions still
        # crosses it regardless of how many other sessions are clean.
        #
        # An explicit user-set entry in bodypart_weights (e.g. via the GUI's
        # Body-Region Weights panel) always wins for that bodypart -- this
        # only fills in bodyparts the user has not already customised.
        _auto_on = bool(self._cfg.get("auto_bodypart_weighting", True))
        if _auto_on and all_bp_bad_fracs:
            _sess_thresh = float(self._cfg.get("auto_bp_weight_session_thresh", 0.3))
            _agg_auto = {bp: float(np.mean([1.0 if f > _sess_thresh else 0.0
                                             for f in all_bp_bad_fracs.get(bp, [0.0])]))
                         for bp in bps_ref}
            _lo    = float(self._cfg.get("auto_bp_weight_lo", 0.10))
            _hi    = float(self._cfg.get("auto_bp_weight_hi", 0.35))
            _floor = float(self._cfg.get("auto_bp_weight_floor", 0.35))
            _user_bpw = dict(self._cfg.get("bodypart_weights") or {})
            _auto_bpw = {}
            if _hi > _lo:
                for bp, frac in _agg_auto.items():
                    if bp in _user_bpw or frac <= _lo:
                        continue
                    _t = min(1.0, (frac - _lo) / (_hi - _lo))
                    _auto_bpw[bp] = round(1.0 - _t * (1.0 - _floor), 3)
            if _auto_bpw:
                _preview = ", ".join(f"{bp}={w:.2f}" for bp, w in
                                     sorted(_auto_bpw.items(), key=lambda kv: kv[1])[:6])
                _more = f", +{len(_auto_bpw) - 6} more" if len(_auto_bpw) > 6 else ""
                self._log(
                    f"  [AUTO-WEIGHT] {len(_auto_bpw)} bodypart(s) down-weighted "
                    f"(affected in {_lo:.0%}-{_hi:.0%} of sessions -> "
                    f"weight 1.00-{_floor:.2f}): {_preview}{_more}")
                self._cfg["bodypart_weights"] = {**_auto_bpw, **_user_bpw}

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

        # 3. Features — dispatch to 3D or V2 extractor based on input format
        _body_norm    = bool(self._cfg.get("body_normalise",  False))
        _long_lag     = bool(self._cfg.get("long_lag_drift",  False))
        _long_scales  = bool(self._cfg.get("long_scale_bins", False))
        # v2.1: skip the angular block when no spine landmarks match by keyword
        # (legacy mode keeps the evenly-spaced fallback).  Used for every feature
        # call in this run so training and inference stay dimensionally aligned.
        _ang_fb = bool(self._cfg.get("angular_fallback", True))
        if _is_3d:
            # 3D world-coordinate data: body normalisation (pixel-space nose-to-tail)
            # is not meaningful; skip the 2D-only checks.
            if _body_norm:
                self._log("  [INFO] 3D mode: body_normalise ignored "
                          "(world-coordinate distances are already scale-invariant).")
            _3d_scales = (("50/" if fps >= 60 else "") + "100/200"
                          + ("/500/1000" if _long_scales else "") + " ms")
            _3d_lag = "5/10" + ("/20/30-bin" if _long_lag else "-bin") + " lag drift"
            self._log(f"\n[3/7]  Extracting 3D features  "
                      f"({_3d_scales} · true Euclidean distances + velocities · {_3d_lag})...")
            self._stage("3/7 — Extracting 3D features", f"scale={_3d_scales}")
        else:
            # If body normalisation is requested but no nose/tail spine landmarks are
            # present, extract_features_v2 silently skips it — warn so the user knows
            # spatial features stay in raw pixels (scale-variant across sessions).
            if _body_norm and _find_spine_indices(bps_ref) == (None, None):
                self._log("  [VALID-WARN] body_normalise is ON but no head/tail spine "
                          "landmarks were found among the shared bodyparts — "
                          "normalisation will be skipped and spatial features remain "
                          "in raw pixels (sensitive to camera distance / body size).")
            _scale_desc = ("50/100/200 ms" if fps >= 60 else "100/200 ms")
            _lag_desc = "long-lag" if _long_lag else "std-lag"
            self._log(f"\n[3/7]  Extracting V2 features  "
                      f"({_scale_desc} + angular, body_normalise={_body_norm}, {_lag_desc})...")
            self._stage("3/7 — Extracting V2 features", f"scale={_scale_desc}")
        # Issue 1b (body-region weighting) / issue 2 (visibility features):
        # MUST be applied identically here (training) and at inference-time
        # re-extraction (predict_labels / the no-MLP fallback below) — a
        # mismatch silently desyncs the MLP's expected feature layout.
        _bp_weights = self._cfg.get("bodypart_weights") or None
        _vis_enabled = bool(self._cfg.get("visibility_features_enabled", True))
        _vis_pct     = float(self._cfg.get("visibility_adaptive_pct", 10))
        # Turned-away-from-camera detection (v3, validated) — always computed
        # regardless of visibility_features_enabled, so the overlay video/log
        # can show detected windows even if the visibility feature BLOCK
        # itself is disabled.  Uses the same likelihood_thresh/adaptive_pct
        # machinery as the visibility features, plus its own GUI-editable
        # confidence threshold and (non-GUI) debounce windows.
        _ta_conf_thresh = float(self._cfg.get("turned_away_conf_thresh", 0.30))
        _ta_min_window_s = float(self._cfg.get("turned_away_min_window_s", 0.4))
        _ta_merge_gap_s  = float(self._cfg.get("turned_away_merge_gap_s", 0.5))
        all_feats = []
        all_vis: list = []   # per-session (n_bins, n_vis_cols) — for cluster_confidence.csv
        all_turned_away: list = []   # per-session (n_bins,) bool mask
        for i, (xy, name) in enumerate(zip(all_xy, all_names)):
            if _is_3d:
                self._stage("3/7 — Extracting 3D features",
                            f"{i+1}/{len(all_xy)}: {name}")
                f = extract_features_3d(xy, fps, bps_ref,
                                        long_lag_drift=_long_lag,
                                        long_scale_bins=_long_scales,
                                        bodypart_weights=_bp_weights)
            else:
                self._stage("3/7 — Extracting V2 features",
                            f"{i+1}/{len(all_xy)}: {name}")
                f = extract_features_v2(xy, fps, bps_ref, body_normalise=_body_norm,
                                        angular_fallback=_ang_fb,
                                        long_lag_drift=_long_lag,
                                        bodypart_weights=_bp_weights)
            _vis = None
            if _vis_enabled and i < len(all_ll):
                _vis = compute_session_visibility_block(
                    all_ll[i], bps_ref, fps,
                    float(self._cfg["likelihood_thresh"]), _vis_pct)
            if _vis is not None:
                all_vis.append(_vis)
                f = _append_visibility_block(f, _vis)
            _n_bins_i = f.shape[1]
            if i < len(all_ll) and all_ll[i] is not None and getattr(all_ll[i], "size", 0):
                if "nose" not in [str(b).lower() for b in bps_ref]:
                    self._log(f"  [WARN] {name}: turned-away detection skipped "
                              f"— no 'nose' bodypart in the shared bodypart set.")
                _ta_mask = detect_turned_away_bins(
                    all_ll[i], bps_ref, fps,
                    float(self._cfg["likelihood_thresh"]), _vis_pct,
                    _ta_conf_thresh, _ta_min_window_s, _ta_merge_gap_s)
            else:
                _ta_mask = np.zeros(0, dtype=bool)
            # Align to this session's actual bin count (defensive — should
            # already match since both derive from the same fps/win binning).
            if _ta_mask.shape[0] != _n_bins_i:
                _tmp = np.zeros(_n_bins_i, dtype=bool)
                _n_copy = min(_ta_mask.shape[0], _n_bins_i)
                _tmp[:_n_copy] = _ta_mask[:_n_copy]
                _ta_mask = _tmp
            all_turned_away.append(_ta_mask)
            _n_ta = int(_ta_mask.sum())
            if _n_ta:
                self._log(f"  [TURNED-AWAY] {name}: {_n_ta}/{_n_bins_i} bins "
                          f"({100 * _n_ta / max(1, _n_bins_i):.1f}%) flagged "
                          f"turned-away from camera.")
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

        # vis_cat: bin-aligned with feats_cat's columns (same session-concat
        # order) — kept separate from feats_cat (which already has the vis
        # block folded in) purely so compute_cluster_confidence_profile can
        # slice it directly later without needing to know feats_cat's exact
        # column layout.
        _vis_col_names = visibility_feature_names(bps_ref) if _vis_enabled else []
        vis_cat = (np.vstack(all_vis)
                  if (_vis_enabled and all_vis and len(all_vis) == len(all_feats))
                  else None)

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

        # Turned-away-from-camera bins (v3, computed per-session above,
        # always — independent of visibility_features_enabled). Combined with
        # flat_held_bin_mask into a single training-exclusion mask ONLY when
        # exclude_turned_away=True. At False, turned_away_bin_mask is still
        # available below (dedicated label / video overlay) but is NOT folded
        # into the training exclusion mask — exclude_mask then reduces to
        # exactly flat_held_bin_mask, which is the byte-identical, true no-op
        # path back to pre-existing (pre-turned-away-feature) clustering
        # output — the GUI escape hatch.
        turned_away_bin_mask = (np.concatenate(all_turned_away)
                                if all_turned_away else np.zeros(n_bins, dtype=bool))
        if turned_away_bin_mask.shape[0] != n_bins:
            # Defensive re-align — should already match exactly since both are
            # derived from the same per-session bin counts as feats_cat.
            _tmp_ta = np.zeros(n_bins, dtype=bool)
            _n_copy_ta = min(turned_away_bin_mask.shape[0], n_bins)
            _tmp_ta[:_n_copy_ta] = turned_away_bin_mask[:_n_copy_ta]
            turned_away_bin_mask = _tmp_ta
        n_turned_away = int(turned_away_bin_mask.sum())
        _exclude_turned_away = bool(self._cfg.get("exclude_turned_away", True))

        exclude_mask = ((flat_held_bin_mask | turned_away_bin_mask)
                        if _exclude_turned_away else flat_held_bin_mask)
        n_excl = int(exclude_mask.sum())
        n_good = n_bins - n_excl
        if n_excl > 0 and n_good < 10:
            self._log(f"  [WARN] Only {n_good} good bins after flat-held/"
                      f"turned-away exclusion — disabling exclusion for this run.")
            exclude_mask = np.zeros(n_bins, dtype=bool)
            flat_held_bin_mask = np.zeros(n_bins, dtype=bool)
            n_flat, n_excl, n_good = 0, 0, n_bins
        feats_good = feats_cat[:, ~exclude_mask] if n_excl > 0 else feats_cat
        if n_flat > 0:
            self._log(f"  [OCCLUSION] {n_flat}/{n_bins} bins "
                      f"({100 * n_flat / n_bins:.1f}%) with ≥{100*_flat_held_bp_frac:.0f}% "
                      f"bodyparts simultaneously flat-held — excluded from UMAP/"
                      f"HDBSCAN training; MLP infers labels for these at Step 7.")
        if n_turned_away > 0:
            self._log(
                f"  [TURNED-AWAY] {n_turned_away}/{n_bins} bins "
                f"({100 * n_turned_away / n_bins:.1f}%) flagged turned-away from "
                f"camera across all sessions"
                + (" — excluded from UMAP/HDBSCAN training; will receive a "
                   "dedicated 'Turned Away' label."
                   if _exclude_turned_away else
                   " — NOT excluded from training (exclude_turned_away=False); "
                   "HDBSCAN may or may not separate them naturally."))

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
        # Formula: clip(avg_session_bins / 20, 30, 90)
        #   ~600 bins/sess  (1 min) → 30   floor -- see floor rationale below
        #   ~1200 bins/sess (2 min) → 60   matches publication default
        #   ~1800 bins/sess (3 min) → 90   smoother manifold for longer recordings
        #   >1800 bins/sess         → 90   capped; higher values give diminishing returns
        # Using per-session average rather than total bins means the value stays
        # calibrated to recording length regardless of how many sessions are pooled.
        # User can override by setting umap_n_neighbors > 0 in cfg.
        #
        # Floor raised 15 -> 30 (Aug 2026): a short-session real dataset (avg
        # ~333 training bins/session, well under the 600-bin reference point)
        # auto-set to 16 under the old floor and showed severe UMAP embedding
        # seed-instability -- a 6-seed sweep gave cluster counts ranging 1-25
        # (mean pairwise ARI 0.31, one seed collapsing to a single cluster).
        # Small neighborhoods make UMAP's stochastic optimization far more
        # sensitive to its random seed's local minimum; this is a general
        # property of UMAP, not specific to that dataset. Forcing the SAME
        # feature matrix through n_neighbors=30 instead of the auto 16 raised
        # mean pairwise ARI to 0.57 (n_neighbors=40 was worse, 0.43 -- 30 is
        # not just "higher is better", it was the tested sweet spot for that
        # dataset). Only validated against one dataset so far; if a future
        # short-session dataset still shows low seed_sweep_stability ARI even
        # at this floor, prefer raising umap_n_neighbors explicitly over
        # raising this floor again without new evidence.
        _nn_cfg = int(self._cfg.get("umap_n_neighbors", 0))
        if _nn_cfg <= 0:
            _n_sessions  = max(1, len(all_feats))
            _avg_bins    = n_samp // _n_sessions   # use training sample, not total bins
            _nn_adaptive = max(30, min(90, _avg_bins // 20))
            self._cfg["umap_n_neighbors"] = _nn_adaptive
            self._log(f"  [UMAP] n_neighbors auto-set to {_nn_adaptive} "
                      f"(avg_training_bins={_avg_bins}, {_n_sessions} session(s), "
                      f"formula=clip(avg_bins/20, 30, 90); "
                      f"set umap_n_neighbors>0 to override)")
        self._mem_checkpoint("3/7 feature extraction done")
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
        self._mem_checkpoint("4/7 UMAP embedding done")

        # Build the full-size embedding (n_bins rows) for umap_embedding.npy so
        # that session_bin_ranges.json slice indices remain valid.  Bins that
        # were not part of the UMAP training sample — flat-held bins, and any
        # good bins left out by train_frac subsampling — are projected via
        # umap_model.transform() for visualisation.  Bins that WERE part of
        # the training sample reuse the exact fit `embedding` at their
        # original positions rather than being re-projected: UMAP's
        # out-of-sample transform() is measurably less tight than the fit it
        # approximates, and previously this branch re-transformed every good
        # bin (including the trained ones) whenever subsampling and flat-held
        # exclusion were both active, silently degrading the saved embedding
        # used by cube_analyser's UMAP/Energy-Landscape views.
        try:
            embedding_save = np.empty((n_bins, embedding.shape[1]), dtype=float)
            _good_positions = (np.where(~exclude_mask)[0] if n_excl > 0
                                else np.arange(n_bins))
            if n_samp == n_good:
                embedding_save[_good_positions] = embedding
            else:
                embedding_save[_good_positions[idx]] = embedding
                _held_out = np.ones(n_good, dtype=bool)
                _held_out[idx] = False
                if _held_out.any():
                    _feats_ho_sc = scaler.transform(feats_good[:, _held_out].T)
                    if pca_model is not None:
                        _feats_ho_sc = pca_model.transform(_feats_ho_sc)
                    embedding_save[_good_positions[_held_out]] = \
                        umap_model.transform(_feats_ho_sc)
            if n_excl > 0:
                _feats_excl_sc = scaler.transform(feats_cat[:, exclude_mask].T)
                if pca_model is not None:
                    _feats_excl_sc = pca_model.transform(_feats_excl_sc)
                embedding_save[exclude_mask] = umap_model.transform(_feats_excl_sc)
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
        # Original HDBSCAN output labels, kept so approximate_predict's
        # fallback inference path (which re-queries hdb_clf directly and gets
        # back ORIGINAL ids) can still be mapped to the final id space after
        # split/merge refinement + rare-cluster pruning (see _hdb_remap below).
        _orig_hdb_labels = hdb_labels.copy()
        n_cl      = len(set(hdb_labels[hdb_labels >= 0]))
        noise     = (hdb_labels < 0).sum()
        noise_pct = 100 * noise / max(1, len(hdb_labels))
        self._log(f"  {n_cl} clusters, {noise} noise points "
                  f"({noise_pct:.1f} %), "
                  f"{hdb_score_label}={hdb_score:.3f}")
        self._stage("5/7 — HDBSCAN done",
                    f"{n_cl} clusters · {noise_pct:.0f}% noise · {hdb_score_label}={hdb_score:.3f}")

        # ── Cluster-stability seed sweep ────────────────────────────────────────
        # Moved here (Aug 2026, was after refinement/pruning below) so its
        # mean_ari can inform the consensus-clustering auto-trigger just
        # below, instead of only being logged as an after-the-fact warning
        # once the primary partition was already exported. Re-run UMAP+
        # HDBSCAN over several seeds to measure how reproducible the
        # PARTITION is (the internal quality gates only measure cluster
        # tightness, not reproducibility).
        # _consensus_forced: consensus is being run unconditionally (not via
        # the auto-trigger below, which itself needs _sweep to decide) --
        # in that case the standalone sweep's only purpose (deciding WHETHER
        # to trigger consensus) is already moot, so skip it and let
        # consensus_cluster() report its own per-seed stability stats for
        # free instead (see the block right after the consensus call below).
        _consensus_forced = bool(self._cfg.get("consensus_clustering_enabled", False))
        _n_sweep = int(self._cfg.get("seed_sweep_n", 0) or 0)
        _sweep = None
        if _n_sweep >= 2 and not _consensus_forced:
            try:
                self._log(f"\n[STABILITY]  Seed sweep ({_n_sweep} seeds) — "
                          f"assessing cluster-count / partition stability...")
                self._stage("Cluster-stability seed sweep",
                            f"{_n_sweep} seeds — re-running UMAP+HDBSCAN")
                _sweep = seed_sweep_stability(
                    feats_sc.T, self._cfg, _n_sweep, log_fn=self._log)
                if _sweep:
                    _sc = _sweep.get('stable_counts', _sweep['counts'])
                    self._log(f"  Mean pairwise ARI = {_sweep['mean_ari']:.3f} "
                              f"(cluster counts {min(_sc)}–{max(_sc)} across "
                              f"seeds used in the mean; full sweep range "
                              f"{min(_sweep['counts'])}–{max(_sweep['counts'])})")
                    _dbcv = [d for d in _sweep.get('dbcv', []) if d == d]  # drop NaN
                    if _dbcv:
                        self._log(f"  Per-seed DBCV: "
                                  f"{', '.join(f'{d:.3f}' for d in _dbcv)} "
                                  f"(mean {np.mean(_dbcv):.3f}, "
                                  f"range {min(_dbcv):.3f}–{max(_dbcv):.3f})")
            except Exception:
                self._log(f"  [WARN] seed sweep: {traceback.format_exc()}")
        elif _consensus_forced:
            self._log("\n[STABILITY]  Seed sweep skipped: "
                      "consensus_clustering_enabled=True already forces "
                      "consensus clustering on, so the standalone sweep's "
                      "only purpose (deciding whether to trigger it) is "
                      "moot -- stability stats will be reported from "
                      "consensus clustering's own per-seed partitions instead.")

        # ── Consensus/co-association clustering ─────────────────────────────
        # Replaces the single primary seed's HDBSCAN partition with one built
        # from agreement across several independent seeds -- see
        # consensus_cluster()'s docstring for the Aug 2026 investigation that
        # motivated this (UMAP embedding topology itself was seed-unstable on
        # a real dataset; no HDBSCAN sweep/selection tuning fixed it, but
        # consensus clustering did). Costs roughly consensus_n_seeds x the
        # primary UMAP+HDBSCAN runtime, so it only runs when explicitly
        # requested (consensus_clustering_enabled=True) OR auto-triggered:
        # the seed sweep above came back below consensus_auto_threshold
        # (default 0.5 -- clearly unstable, distinct from the existing 0.7
        # "pass" bar used elsewhere for cluster_stability's warn-only gate)
        # AND the caller did not explicitly set consensus_clustering_enabled
        # =False (a deliberate opt-out is always respected -- auto-trigger
        # only fills in when the caller never expressed a preference).
        _explicit_off = ("consensus_clustering_enabled" in self._explicit_cfg_keys
                          and not self._cfg.get("consensus_clustering_enabled"))
        _auto_threshold = float(self._cfg.get("consensus_auto_threshold", 0.5) or 0)
        _auto_triggered = (not self._cfg.get("consensus_clustering_enabled", False)
                            and not _explicit_off
                            and _auto_threshold > 0
                            and _sweep and _sweep.get("mean_ari", 1.0) < _auto_threshold)
        if _auto_triggered:
            self._log(f"  [CONSENSUS-AUTO] seed sweep mean ARI "
                      f"{_sweep['mean_ari']:.3f} < {_auto_threshold} — "
                      f"partition is seed-unstable, auto-enabling consensus "
                      f"clustering (set consensus_clustering_enabled=False "
                      f"explicitly to opt out of this auto-trigger).")
        _consensus_used = False
        if self._cfg.get("consensus_clustering_enabled", False) or _auto_triggered:
            _cn_seeds = int(self._cfg.get("consensus_n_seeds", 8) or 8)
            self._log(f"\n[CONSENSUS]  Building {_cn_seeds}-seed co-association "
                      f"consensus partition...")
            self._stage("5/7 — Consensus clustering",
                        f"{_cn_seeds} seeds — co-association + Ward linkage")
            try:
                _cons = consensus_cluster(feats_sc.T, self._cfg, _cn_seeds,
                                          log_fn=self._log, embedding=embedding)
            except Exception:
                _cons = None
                self._log(f"  [WARN] consensus clustering failed: "
                          f"{traceback.format_exc()}")
            if _cons is not None:
                hdb_labels, _cons_quality = _cons
                _orig_hdb_labels = hdb_labels.copy()
                hdb_score = _cons_quality["separation_ratio"]
                hdb_score_label = "consensus separation ratio"
                n_cl      = len(set(hdb_labels[hdb_labels >= 0]))
                noise     = (hdb_labels < 0).sum()
                noise_pct = 100 * noise / max(1, len(hdb_labels))
                _consensus_used = True
                self._log(f"  Consensus: {n_cl} clusters, {noise} noise points "
                          f"({noise_pct:.1f} %), separation_ratio="
                          f"{hdb_score:.2f}x (within- vs between-cluster mean "
                          f"co-association across {_cons_quality['n_seeds_used']} "
                          f"seeds; per-seed counts={_cons_quality['per_seed_counts']})")
                self._log(f"  [feature-space] DBCV="
                          f"{_cons_quality.get('dbcv_feature_space')}  "
                          f"silhouette={_cons_quality.get('silhouette_feature_space')} "
                          f"(comparable across configs, unlike separation_ratio)")
                self._stage("5/7 — Consensus done",
                            f"{n_cl} clusters · {noise_pct:.0f}% noise · "
                            f"separation={hdb_score:.2f}x")
                # Standalone sweep was skipped above (_consensus_forced) --
                # consensus_cluster() computed the same seeds/counts/ari/
                # mean_ari stats for free from its own per-seed partitions
                # (see consensus_cluster()'s "Derived seed-stability stats"
                # block); wire them into _sweep here so cluster_stability.png
                # and _validation["cluster_stability"] below work completely
                # unmodified -- they only ever read _sweep's keys, not which
                # code path produced it.
                if _consensus_forced and _sweep is None and "mean_ari" in _cons_quality:
                    _sweep = {k: _cons_quality[k] for k in
                              ("seeds", "counts", "ari", "mean_ari", "stable_counts")
                              if k in _cons_quality}
                    _sc = _sweep.get('stable_counts', _sweep['counts'])
                    self._log(f"  Mean pairwise ARI = {_sweep['mean_ari']:.3f} "
                              f"(cluster counts {min(_sc)}–{max(_sc)} across "
                              f"seeds used in the mean; full sweep range "
                              f"{min(_sweep['counts'])}–{max(_sweep['counts'])}) "
                              f"[derived from consensus clustering's own "
                              f"per-seed partitions, unrefined -- not directly "
                              f"comparable to standalone seed-sweep values]")
            else:
                self._log("  [WARN] consensus clustering produced no usable "
                          "result (too few seeds succeeded) — falling back to "
                          "the primary single-seed HDBSCAN result.")

        # ── Iterative split + merge refinement (issue 4, bidirectional) ───────
        # On by default (hdbscan_split_silhouette_thresh=0.2,
        # hdbscan_merge_thresh=0.08) to correct the over-fragmentation /
        # impurity seen in practice (e.g. a single "licking" behaviour split
        # across 3 clusters). Set either threshold to its no-op value
        # (None / 0.0) to disable that pass, or both to fully skip the loop
        # — e.g. via compat_mode="legacy_v2" to reproduce pre-refinement
        # runs exactly. Logic lives in refine_clusters_iterative() (shared
        # with seed_sweep_stability(), so per-seed stability reflects this
        # SAME refined partition, not just the pre-refinement candidate).
        # Snapshot for the before/after diagnostic plot below — isolates the
        # refinement pass's own effect from the separate rare-cluster-pruning
        # renumbering that happens afterward (see split_merge_refinement.png
        # call site further down).
        _pre_refine_labels = hdb_labels.copy()
        if _consensus_used:
            # This primary-path pass is always skipped for consensus labels:
            # its split step's local re-embedding assumes hdb_clf/embedding
            # come from the SAME fit as the labels being refined -- not true
            # here (hdb_clf is still the primary single-seed fit) -- and its
            # merge step needs hdb_clf.condensed_tree_, which consensus
            # partitions don't have. If consensus_refine_enabled=True,
            # consensus_cluster() already ran its OWN split (reused
            # split_impure_clusters) + merge (merge_by_coassociation) pass
            # internally before returning -- see the [consensus-refine] log
            # lines above -- so this is not a second, redundant skip; it
            # only prevents a genuinely incompatible second attempt here.
            self._log("  [refine] this pass skipped for consensus labels "
                      "(condensed-tree merge criterion doesn't apply; see "
                      "consensus_refine_enabled for consensus's own native "
                      "split/merge pass)")
        else:
            hdb_labels = refine_clusters_iterative(
                feats_sc, embedding, hdb_labels, hdb_clf, self._cfg,
                log_fn=self._log)
        n_cl      = len(set(hdb_labels[hdb_labels >= 0]))
        noise     = (hdb_labels < 0).sum()
        noise_pct = 100 * noise / max(1, len(hdb_labels))
        _post_refine_labels = hdb_labels.copy()

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

        # Compose a comprehensive ORIGINAL-hdbscan-id -> final-id remap when
        # split/merge refinement and/or rare-cluster pruning changed anything,
        # so the approximate_predict fallback path (which only ever sees
        # hdb_clf's ORIGINAL output ids) stays correctly mapped end-to-end.
        # NOTE: split_impure_clusters can in principle send different points
        # of one original cluster to different final ids; this 1:1 remap uses
        # one representative point per original id (exact for merge/prune,
        # which always move a whole cluster together — the overwhelmingly
        # common case since the fallback path only runs when mlp_clf is None,
        # i.e. < 2 clusters were found, leaving little room for split/merge to
        # act in the first place).
        if not np.array_equal(_orig_hdb_labels, hdb_labels) or _hdb_remap:
            _comprehensive_remap = {}
            for _oid in sorted(set(int(x) for x in _orig_hdb_labels if x >= 0)):
                _pos = np.flatnonzero(_orig_hdb_labels == _oid)
                if _pos.size:
                    _comprehensive_remap[_oid] = int(hdb_labels[_pos[0]])
            _hdb_remap = _comprehensive_remap

        # ── Auto-flag unresolved impure clusters ────────────────────────────
        # Runs on the FINAL (post-refinement, post-pruning) hdb_labels, in the
        # PRIMARY embedding -- the same embedding/labels split_impure_clusters
        # itself scores candidates in, so this uses an identical yardstick.
        # Deliberately does NOT touch hdb_labels/n_cl/_hdb_remap (the cluster
        # stays real for MLP training) -- see auto_flag_impure_clusters'
        # DEFAULTS docstring for why this is a display-layer remap, applied
        # per session below, not a training-time exclusion.
        _auto_impure_ids: set = set()
        if bool(self._cfg.get("auto_flag_impure_clusters", True)) and n_cl >= 2:
            if not _exclude_turned_away:
                self._log("  [IMPURE] auto_flag_impure_clusters=True but "
                          "exclude_turned_away=False -- no reserved display id "
                          "to route into, skipping.")
            else:
                _impure_thresh = float(
                    self._cfg.get("auto_flag_impure_silhouette_thresh", 0.0) or 0.0)
                if _impure_thresh <= 0:
                    _impure_thresh = float(
                        self._cfg.get("hdbscan_split_silhouette_thresh", 0.0) or 0.0)
                if _impure_thresh > 0:
                    _, _impure_sil_means = _mean_silhouette_per_cluster(
                        embedding, hdb_labels)
                    # Worst-first: sorted by silhouette ascending, so a
                    # forced cap below (the <2-real-clusters safety margin)
                    # keeps the WORST offenders and spares the least-bad
                    # ones, rather than an arbitrary id-order truncation.
                    _candidates = sorted(
                        (c for c, m in _impure_sil_means.items() if m < _impure_thresh),
                        key=lambda c: _impure_sil_means[c])
                    # Never flag away every real cluster (or down to fewer
                    # than 2) -- leaves the MLP with < 2 classes to train on,
                    # a hard failure elsewhere in run(). Same safety margin
                    # as the "< 2 clusters" guards already used around
                    # HDBSCAN/consensus. Cap to the worst offenders instead
                    # of skipping the whole pass when ALL candidates would
                    # breach the floor -- e.g. 3 clusters, 2 flagged as
                    # impure: flag just the worse of the two rather than
                    # flagging neither.
                    _max_flaggable = max(0, n_cl - 2)
                    if _candidates and _max_flaggable == 0:
                        self._log(
                            f"  [WARN] {len(_candidates)} cluster(s) below the "
                            f"impure-cluster threshold, but this run only has "
                            f"{n_cl} real cluster(s) total -- flagging any "
                            f"would leave < 2 real clusters, skipping "
                            f"auto-flag entirely for this run.")
                    elif _candidates:
                        _capped = len(_candidates) > _max_flaggable
                        if _capped:
                            self._log(
                                f"  [WARN] {len(_candidates)} cluster(s) below "
                                f"the impure-cluster threshold, but flagging "
                                f"all of them would leave < 2 real clusters -- "
                                f"flagging only the worst {_max_flaggable} "
                                f"(by silhouette) and leaving the rest as real "
                                f"clusters.")
                        _flag_ids = _candidates[:_max_flaggable]
                        _auto_impure_ids = set(_flag_ids)
                        self._log(
                            f"  [IMPURE] cluster(s) "
                            f"{', '.join(f'#{c} (silhouette={_impure_sil_means[c]:.3f})' for c in _flag_ids)} "
                            f"remain below the impure-cluster threshold "
                            f"({_impure_thresh}) after refinement -- auto-"
                            f"flagging as low-quality and folding into the "
                            f"'Turned Away' display bucket "
                            f"(auto_flag_impure_clusters=True). These stay "
                            f"real classes in the trained MLP; only the "
                            f"per-session CSV/plot labels are remapped.")

        # Feature-space DBCV/silhouette (Aug 2026): computed on the FINAL
        # (post-refinement, post-pruning) hdb_labels -- matches
        # consensus_cluster()'s own feature-space scoring, which (as of this
        # same change) is also computed on ITS final post-refinement/pruning
        # labels, not an intermediate candidate -- so the two numbers
        # describe the actual DELIVERED partition in both cases and are
        # directly comparable, unlike separation_ratio vs. embedding-space
        # DBCV. Skipped here for consensus mode: already computed and logged
        # right after consensus_cluster() returned (same final labels --
        # nothing between there and here changes them, the rare-cluster
        # pruning above is a no-op re-application for consensus, which
        # already pruned with the same min_cluster_freq/denominator).
        if _consensus_used:
            self._log("  [feature-space] DBCV/silhouette already reported "
                      "above (consensus_cluster()'s own final-partition score)")
        else:
            try:
                _dbcv_fs = _dbcv_feature_space(feats_sc.T, hdb_labels, log_fn=self._log)
                _sil_fs  = validate_clustering(feats_sc.T, hdb_labels).get("silhouette_score")
                self._log(f"  [feature-space] DBCV={_dbcv_fs}  silhouette={_sil_fs} "
                          f"(comparable across configs, incl. consensus mode)")
            except Exception:
                self._log(f"  [WARN] feature-space DBCV/silhouette: "
                          f"{traceback.format_exc()}")

        # Validation gate 4: clustering quality (silhouette)
        # Skipped for consensus mode: silhouette here is computed against the
        # PRIMARY single-seed embedding, which consensus labels are not
        # derived from (they come from co-association across many seeds'
        # independent embeddings) -- that mismatch produces a misleading low
        # score even when the consensus partition's own separation_ratio
        # (already logged above) is strong. Not a meaningful check for this
        # path, so skip rather than report a number that doesn't mean what it
        # normally means.
        if _consensus_used:
            self._log("  [VALID] clustering-quality gate skipped for "
                      "consensus mode (see separation_ratio above instead)")
        else:
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

        # ── Cluster validity diagnostic plot (issue 3) ─────────────────────────
        if self._cfg["save_plots"]:
            try:
                plot_cluster_validity(embedding, hdb_labels, feats_sc.T,
                                      self._out_plots / "cluster_validity.png",
                                      hdb_clf=hdb_clf)
                self._log("  [PLOT] cluster_validity.png saved")
            except Exception:
                self._log(f"  [WARN] cluster_validity plot: "
                          f"{traceback.format_exc()}")

            if self._cfg.get("cluster_hierarchy_enabled", True):
                # Always save BOTH theme variants regardless of this run's
                # active plot_theme -- unlike every other plot in the suite
                # (which only ever needs to match the GUI's current theme),
                # this one is meant to be pulled into external docs/talks in
                # whichever theme fits the destination, so it saves both up
                # front rather than requiring a full re-run to get the other.
                _hierarchy_theme_before = str(self._cfg.get("plot_theme", "dark"))
                try:
                    for _theme_name in ("dark", "light"):
                        _apply_plot_theme(_theme_name)
                        plot_cluster_hierarchy(
                            feats_sc.T, hdb_labels,
                            self._out_plots / f"cluster_hierarchy_{_theme_name}.png",
                            linkage_method=str(self._cfg.get("cluster_hierarchy_linkage", "ward")),
                            centroid_out_path=(
                                self._out_model / "cluster_feature_centroids.npz"
                                if _theme_name == "dark" else None))
                    self._log("  [PLOT] cluster_hierarchy_dark.png + "
                              "cluster_hierarchy_light.png saved")
                    self._log("  [PLOT] cluster_feature_centroids.npz saved")
                except Exception:
                    self._log(f"  [WARN] cluster_hierarchy plot: "
                              f"{traceback.format_exc()}")
                finally:
                    _apply_plot_theme(_hierarchy_theme_before)

            # Split/merge before/after diagnostic — only when refinement
            # actually changed something (skip a no-op plot when the pass
            # was off, or ran but converged with zero changes). Uses the
            # pre-/post-refinement snapshots, NOT the final (post-pruning)
            # hdb_labels, so this isolates the refinement pass's own effect
            # from rare-cluster-pruning's separate renumbering.
            if not np.array_equal(_pre_refine_labels, _post_refine_labels):
                try:
                    plot_split_merge_refinement(
                        embedding, _pre_refine_labels, _post_refine_labels,
                        self._out_plots / "split_merge_refinement.png")
                    self._log("  [PLOT] split_merge_refinement.png saved")
                except Exception:
                    self._log(f"  [WARN] split_merge_refinement plot: "
                              f"{traceback.format_exc()}")

        # ── Cluster-stability validation + plot ─────────────────────────────
        # _sweep was already computed earlier (before the consensus-clustering
        # decision, which needs mean_ari) -- just report/plot it here now that
        # self._out_plots exists. See the seed-sweep block above for the
        # actual UMAP+HDBSCAN re-runs (not duplicated here).
        if _sweep:
            try:
                _validation["cluster_stability"] = {
                    "stage": "cluster_stability", "status":
                        "pass" if _sweep["mean_ari"] >= 0.7 else "warn",
                    "mean_ari": round(_sweep["mean_ari"], 4),
                    "cluster_counts": _sweep["counts"],
                    "dbcv_scores": [round(d, 4) if d == d else None
                                    for d in _sweep.get("dbcv", [])],
                    "warnings": ([] if _sweep["mean_ari"] >= 0.7 else
                                 [f"Mean ARI {_sweep['mean_ari']:.3f} < 0.7: "
                                  "cluster partition is seed-sensitive."
                                  + (" Consensus clustering was auto-enabled "
                                     "for this run to compensate."
                                     if _auto_triggered else "")]),
                }
                if self._cfg["save_plots"]:
                    plot_cluster_stability(
                        _sweep, self._out_plots / "cluster_stability.png")
                    plot_cluster_volatility(
                        _sweep, self._out_plots / "cluster_volatility.png")
            except Exception:
                self._log(f"  [WARN] cluster-stability plot/validation: "
                          f"{traceback.format_exc()}")

        # 2D and 3D UMAP plots are generated after inference (below) so that
        # they can be filtered to show only clusters actually predicted by the
        # MLP on these sessions — keeping the cluster count consistent across
        # umap_embedding.png, umap_3d.*, and dwell_time_distributions.png.

        # Save UMAP embedding + cluster labels as numpy arrays so cube_analyser
        # can display before/after UMAP views when the user recombines clusters.
        # embedding_save is full-size (n_bins rows); excluded bins (flat-held
        # and/or turned-away) projected via transform() so
        # session_bin_ranges.json slice indices remain valid.
        # hdb_labels_all expands hdb_labels to n_bins with -1 for excluded bins.
        if n_samp == n_good and n_excl == 0:
            # All bins were used for UMAP/HDBSCAN training — lengths already match.
            hdb_labels_all = hdb_labels
        else:
            # Either flat-held/turned-away bins were excluded, or train_frac
            # subsampling occurred (or both) — map each label back to its
            # original bin position so umap_labels.npy stays the same length
            # as umap_embedding.npy (n_bins). Untrained positions (held-out
            # good bins, excluded bins) get -1 (noise) since HDBSCAN never
            # clustered them.
            hdb_labels_all = np.full(n_bins, -1, dtype=int)
            _good_positions = (np.where(~exclude_mask)[0] if n_excl > 0
                                else np.arange(n_bins))
            if n_samp == n_good:
                hdb_labels_all[_good_positions] = hdb_labels
            else:
                hdb_labels_all[_good_positions[idx]] = hdb_labels
        try:
            np.save(str(self._out_model / "umap_embedding.npy"), embedding_save)
            np.save(str(self._out_model / "umap_labels.npy"),    hdb_labels_all)
        except Exception:
            pass

        # Cluster centroids (issue 1a) — mean embedding coordinate per cluster,
        # used to pick example clips by embedding proximity instead of nearest-
        # to-median-duration (see attach_centroid_distance / create_example_clips
        # below), and reused by the merge pass above.
        _cluster_centroids = compute_cluster_centroids(embedding_save, hdb_labels_all)
        self._mem_checkpoint("5/7 HDBSCAN clustering done")

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
                            feature_version="v3d" if _is_3d else "v2",
                            analysis_version=ANALYSIS_VERSION,
                            compat_mode=self._cfg.get("compat_mode", "current"),
                            pca_n_components=(int(pca_model.n_components_)
                                              if pca_model is not None else None)),
                       indent=2))

        self._mem_checkpoint("6/7 MLP training + model save done")

        # 7. Predict & export
        self._log("\n[7/7]  Predicting & exporting...")
        self._stage("7/7 — Predicting & exporting", f"0/{len(pairs)} sessions")
        bout_paths, frame_paths, all_epochs, all_frame_labels = [], [], [], []
        # _need_bin_detail: True when B.2 (bin-level HMM smoothing) or B.1
        # (soft-probability HMM emissions) needs predict_labels()'s new
        # per-bin detail (see predict_labels' return_proba docstring). False
        # (both new modes off, the default) means predict_labels() is called
        # exactly as before -- same call, same single-array return type.
        _hmm_smoothing_level = str(self._cfg.get("hmm_smoothing_level", "frame"))
        _hmm_emission_mode   = str(self._cfg.get("hmm_emission_mode", "categorical"))
        _need_bin_detail = (_hmm_smoothing_level == "bin"
                            or _hmm_emission_mode == "soft")
        all_bin_labels: list = []   # per-bin (pre-expansion) label sequences, one per session
        all_bin_proba:  list = []   # per-bin per-class probability matrices, one per session
        all_bin_win:    list = []   # win (frames-per-bin) used for each session's expansion
        # Collect example-clip tasks to write after the main loop so they can
        # be shuffled for an even cross-animal mix per cluster.
        _clip_tasks: list = []   # (vp, epochs_df, file_fps, animal_id)

        for i, ((fp, vp), (xy, name, file_fps)) in enumerate(
                zip(pairs, zip(all_xy, all_names, all_fps_list))):

            self._log(f"  [{i+1}/{len(pairs)}]  {name}")
            self._stage("7/7 — Predicting & exporting",
                        f"{i+1}/{len(pairs)}: {name}")

            # Session bin width — needed both by the no-MLP fallback below and
            # by attach_centroid_distance (issue 1a) after epochs are built.
            win = max(1, int(round(file_fps / 10)))
            _ll_session = all_ll[i] if i < len(all_ll) else None

            if mlp_clf is not None:
                if _need_bin_detail:
                    frame_labels, _bl, _bp = predict_labels(
                        xy, umap_model, mlp_clf, scaler, file_fps,
                        bodyparts=bps_ref, body_normalise=_body_norm,
                        pca_model=pca_model,
                        min_confidence=float(self._cfg.get("mlp_confidence_thresh", 0.0)),
                        angular_fallback=_ang_fb, is_3d=_is_3d,
                        long_lag_drift=_long_lag, long_scale_bins=_long_scales,
                        bodypart_weights=_bp_weights,
                        ll=_ll_session,
                        visibility_features_enabled=_vis_enabled,
                        visibility_adaptive_pct=_vis_pct,
                        likelihood_thresh=float(self._cfg["likelihood_thresh"]),
                        return_proba=True)
                    all_bin_labels.append(_bl)
                    all_bin_proba.append(_bp)
                    all_bin_win.append(win)
                else:
                    frame_labels = predict_labels(
                        xy, umap_model, mlp_clf, scaler, file_fps,
                        bodyparts=bps_ref, body_normalise=_body_norm,
                        pca_model=pca_model,
                        min_confidence=float(self._cfg.get("mlp_confidence_thresh", 0.0)),
                        angular_fallback=_ang_fb, is_3d=_is_3d,
                        long_lag_drift=_long_lag, long_scale_bins=_long_scales,
                        bodypart_weights=_bp_weights,
                        ll=_ll_session,
                        visibility_features_enabled=_vis_enabled,
                        visibility_adaptive_pct=_vis_pct,
                        likelihood_thresh=float(self._cfg["likelihood_thresh"]))
            else:
                # Fallback: no MLP → use HDBSCAN approximate_predict on V2/3D feats
                try:
                    import hdbscan as _hdb
                    if _is_3d:
                        f = extract_features_3d(xy, file_fps, bps_ref,
                                                long_lag_drift=_long_lag,
                                                long_scale_bins=_long_scales,
                                                bodypart_weights=_bp_weights)
                    else:
                        f = extract_features_v2(xy, file_fps, bps_ref,
                                                body_normalise=_body_norm,
                                                angular_fallback=_ang_fb,
                                                long_lag_drift=_long_lag,
                                                bodypart_weights=_bp_weights)
                    if _vis_enabled and _ll_session is not None:
                        _vis_f = compute_session_visibility_block(
                            _ll_session, bps_ref, file_fps,
                            float(self._cfg["likelihood_thresh"]), _vis_pct)
                        f = _append_visibility_block(f, _vis_f)
                    sc   = scaler.transform(f.T)
                    if pca_model is not None:
                        sc = pca_model.transform(sc)
                    emb  = umap_model.transform(sc)
                    soft, _ = _hdb.approximate_predict(hdb_clf, emb)
                    # approximate_predict returns the original HDBSCAN cluster IDs
                    # (before split/merge refinement, rare-cluster pruning, and
                    # renumbering).  Apply the same remap applied to hdb_labels
                    # so the fallback cluster IDs match those used in the plots.
                    if _hdb_remap:
                        _remapped = np.full_like(soft, -1, dtype=int)
                        for _orig, _new in _hdb_remap.items():
                            _remapped[soft == _orig] = _new
                        soft = _remapped
                    fl   = np.repeat(soft.astype(int), win)
                    n_f  = xy.shape[0]
                    if len(fl) < n_f:
                        fl = np.pad(fl, (0, n_f - len(fl)), mode="edge")
                    frame_labels = fl[:n_f]
                except Exception:
                    frame_labels = np.zeros(xy.shape[0], dtype=int)

            all_frame_labels.append(frame_labels)

            # Turned-away dedicated label (v3): a reserved cluster id one
            # above the highest real HDBSCAN id (0..n_cl-1), so it never
            # collides with a real cluster.  Only overrides the DISPLAY copy
            # used for CSV/plot export — the raw `frame_labels` appended
            # above stays untouched so HMM training (below) never sees an
            # out-of-range observation id (hmmlearn's CategoricalHMM is fit
            # with n_clusters categories; feeding it n_cl would raise/corrupt
            # the emission matrix).  Only applied when exclude_turned_away is
            # True — at False this is a pure no-op (_display_labels ==
            # frame_labels exactly), matching the GUI escape hatch.
            _ta_id = int(n_cl)
            _ta_bin_mask = (all_turned_away[i] if i < len(all_turned_away)
                            else np.zeros(0, dtype=bool))
            _ta_frame_mask = _expand_bin_mask_to_frames(
                _ta_bin_mask, file_fps, xy.shape[0])
            _display_labels = frame_labels.copy()
            # turned_away_reason: per-frame provenance for WHY a frame was
            # relabeled into the reserved Turned-Away id, so it's still
            # possible to tell "genuinely occluded/turned away" apart from
            # "auto-flagged impure cluster" after the fact -- both currently
            # share one reserved display id, but the underlying cause
            # matters for anyone auditing an auto-flag decision later.
            _reason = np.full(len(_display_labels), "", dtype=object)
            if _exclude_turned_away and _ta_frame_mask.any():
                _display_labels[_ta_frame_mask] = _ta_id
                _reason[_ta_frame_mask] = "confidence"
            # Auto-flagged impure clusters (see auto_flag_impure_clusters
            # above) fold into the same reserved id -- frame_labels (the raw
            # MLP prediction, untouched) still carries the real cluster id,
            # so this only ever affects the DISPLAY copy, same guarantee as
            # the confidence-based override just above.
            if _auto_impure_ids:
                _impure_mask = np.isin(frame_labels, sorted(_auto_impure_ids))
                if _impure_mask.any():
                    _display_labels[_impure_mask] = _ta_id
                    _reason[_impure_mask] = "auto_impure_cluster"

            # Bout CSV (exact B-SOiD GUI format -- fixed 3-column schema,
            # no raw-label/reason columns here since one bout can span
            # frames whose ORIGINAL cluster/reason differ once merged into
            # the same Turned-Away run; see the frame-label CSV instead for
            # per-frame provenance).
            bout_df = labels_to_bouts(_display_labels)
            bout_p  = self._out_bouts / f"{name}_bout_lengths.csv"
            bout_df.to_csv(str(bout_p), index=False)
            bout_paths.append(bout_p)

            # Frame-label CSV. raw_label preserves the real, un-relabeled
            # predicted cluster id for EVERY frame (identical to `label`
            # except where a Turned-Away override fired) -- so cube_analyser.py
            # or any future stage can still recover which real cluster a
            # given Turned-Away frame originally came from, filter it back
            # in, or audit an auto-flag decision, without needing to rerun
            # the pipeline. turned_away_reason is "" for frames never
            # overridden, "confidence" for the existing likelihood-based
            # detector, "auto_impure_cluster" for auto_flag_impure_clusters.
            frame_df = pd.DataFrame({
                "frame":               np.arange(len(_display_labels)),
                "time_s":              np.arange(len(_display_labels)) / file_fps,
                "label":               _display_labels,
                "raw_label":           frame_labels,
                "turned_away_reason":  _reason,
            })
            frame_p = self._out_bouts / f"{name}_frame_labels.csv"
            frame_df.to_csv(str(frame_p), index=False)
            frame_paths.append(frame_p)

            # Epoch CSV (filtered by user-set min/max bout duration)
            epochs = bouts_to_epochs(
                bout_df, file_fps,
                min_dur=float(self._cfg["min_epoch_dur_s"]),
                max_dur=float(self._cfg["max_epoch_dur_s"]))
            # Attach embedding-space distance-to-centroid (issue 1a) so
            # create_example_clips can select clips by embedding proximity
            # instead of nearest-to-median-duration.
            try:
                _bin_offset = int(_sbr.get(name, [0])[0])
                epochs = attach_centroid_distance(
                    epochs, embedding_save, hdb_labels_all,
                    _cluster_centroids, bin_offset=_bin_offset, win=win)
            except Exception:
                pass
            epochs.to_csv(
                str(self._out_bouts / f"{name}_epochs.csv"), index=False)
            epoch_stats(epochs).to_csv(
                str(self._out_bouts / f"{name}_epoch_stats.csv"), index=False)
            all_epochs.append((epochs, name))

            # Clip-generation epochs use a SEPARATE label array that skips
            # the auto-impure-cluster override (confidence-based turned-away
            # still applies) -- example clips are for VISUAL review, and
            # folding an auto-flagged cluster's clips into the same
            # cluster_{n_cl} "Turned Away" folder as genuine turned-away
            # clips in the video explorer would make it impossible to
            # eyeball whether the auto-flag call was actually correct. Real
            # cluster ids stay real here; the bout/frame CSVs, ethogram, and
            # all quantitative exports above/below are unaffected (still use
            # `epochs`/`_display_labels`).
            if _auto_impure_ids:
                _clip_labels = frame_labels.copy()
                if _exclude_turned_away and _ta_frame_mask.any():
                    _clip_labels[_ta_frame_mask] = _ta_id
                _clip_epochs = bouts_to_epochs(
                    labels_to_bouts(_clip_labels), file_fps,
                    min_dur=float(self._cfg["min_epoch_dur_s"]),
                    max_dur=float(self._cfg["max_epoch_dur_s"]))
                try:
                    _bin_offset = int(_sbr.get(name, [0])[0])
                    _clip_epochs = attach_centroid_distance(
                        _clip_epochs, embedding_save, hdb_labels_all,
                        _cluster_centroids, bin_offset=_bin_offset, win=win)
                except Exception:
                    pass
            else:
                _clip_epochs = epochs

            _ta_names = {_ta_id: "Turned Away"} if _exclude_turned_away else None
            if self._cfg["save_plots"] and not epochs.empty:
                try:
                    plot_ethogram(_display_labels, file_fps,
                                  self._out_plots / f"ethogram_{name}.png",
                                  name, cluster_names=_ta_names)
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
                if self._cfg["save_example_clips"] and not _clip_epochs.empty:
                    # Defer clip writing; process all animals together after the
                    # loop so clips can be shuffled for a cross-animal mix.
                    # Uses _clip_epochs (real cluster ids for auto-flagged
                    # clusters), NOT `epochs` -- see its construction above.
                    _clip_tasks.append(
                        (vp, _clip_epochs.copy(), file_fps, Path(vp).stem))
                if self._cfg["save_labeled_video"]:
                    try:
                        # Fold auto-impure frames into the same banner mask
                        # as confidence-based turned-away, so the labeled
                        # video's overlay stays consistent with the CSV
                        # export: the "C{lbl}" box still shows the TRUE raw
                        # cluster id either way (create_labeled_video always
                        # labels off `frame_labels`, never `_display_labels`)
                        # -- this only affects whether the amber banner also
                        # appears.
                        _banner_mask = _ta_frame_mask
                        if _auto_impure_ids:
                            _banner_mask = _banner_mask | np.isin(
                                frame_labels, sorted(_auto_impure_ids))
                        create_labeled_video(
                            vp, frame_labels, self._out_videos, file_fps,
                            output_fps=int(self._cfg["output_fps"]),
                            turned_away_frame_mask=_banner_mask)
                    except Exception:
                        self._log(f"  [WARN] Labeled video: "
                                  f"{traceback.format_exc()}")

            self._prog(i + 1, len(pairs))

        # ── HMM smoothing pass (post-hoc Multinomial HMM wrapper) ────────────
        all_hmm_labels: list = []
        hmm_model = None
        # _soft_ok / _bin_level_ok: the requested new mode was possible AND
        # every session actually produced the detail it needs (the no-MLP
        # approximate_predict fallback path above never populates
        # all_bin_labels/all_bin_proba -- if even one session took that
        # path, the bin-level and frame-level sequence lists would desync).
        # Falls back to the unchanged frame-level categorical path with a
        # warning rather than silently mismatching sessions. Soft (B.1)
        # takes priority over bin-level-categorical (B.2) when both happen
        # to be requested together, since soft emissions already operate at
        # bin resolution.
        _soft_ok = (_hmm_emission_mode == "soft"
                   and len(all_bin_proba) == len(all_frame_labels)
                   and len(all_frame_labels) > 0)
        _bin_level_ok = (not _soft_ok and _hmm_smoothing_level == "bin"
                         and len(all_bin_labels) == len(all_frame_labels)
                         and len(all_frame_labels) > 0)
        if _hmm_emission_mode == "soft" and not _soft_ok and all_frame_labels:
            self._log("  [WARN] hmm_emission_mode='soft' requested but not "
                      "every session produced bin-probability detail (e.g. "
                      "the no-MLP approximate_predict fallback was used for "
                      "one or more sessions) -- falling back to categorical "
                      "HMM smoothing for this run.")
        if (_hmm_smoothing_level == "bin" and not _soft_ok and not _bin_level_ok
                and all_frame_labels):
            self._log("  [WARN] hmm_smoothing_level='bin' requested but not "
                      "every session produced bin-level detail (e.g. the "
                      "no-MLP approximate_predict fallback was used for one "
                      "or more sessions) -- falling back to frame-level HMM "
                      "smoothing for this run.")
        # Turned-away-excluded training sequences (transition analysis fix,
        # Aug 2026): HMM training must never learn a transition into/out of a
        # frame the animal was turned away from camera during -- that frame's
        # MLP-predicted "real" cluster is just a best-effort guess on data
        # excluded from UMAP/HDBSCAN training in the first place. Splitting
        # each session at turned-away boundaries (rather than dropping those
        # bins from a single concatenated sequence) is essential -- train_hmm/
        # train_hmm_soft already treat each LIST ELEMENT as an independent
        # sequence (hmmlearn's lengths= mechanism), so more/shorter segments
        # per session is a safe, drop-in substitution: no transition is ever
        # counted across the dropped gap. A pure no-op (identical to the
        # original per-session lists) when exclude_turned_away=False, matching
        # this codebase's existing escape-hatch convention for that flag.
        def _ta_bin_mask_for(hi: int, n_bins: int) -> np.ndarray:
            # Fallback is a FULL-LENGTH all-False mask (nothing excluded),
            # not a zero-length one -- an empty mask would make
            # _mask_out_segments' min(len(seq), len(mask)) truncate the
            # whole session to nothing, silently discarding all of its data
            # from training instead of just skipping the (inapplicable)
            # exclusion. Should only ever be hit defensively; all_turned_away
            # is appended once per session unconditionally in Step 3.
            if hi < len(all_turned_away) and len(all_turned_away[hi]) == n_bins:
                return np.asarray(all_turned_away[hi], dtype=bool)
            return np.zeros(n_bins, dtype=bool)

        def _ta_frame_mask_for(hi: int, fps: float, n_frames: int) -> np.ndarray:
            # Frame-level counterpart of _ta_bin_mask_for. Guards
            # _expand_bin_mask_to_frames against being handed a zero-length
            # bin mask -- np.pad(..., mode="edge") raises ValueError on an
            # empty array, so an all-False, FULL-length frame mask (nothing
            # excluded) is built directly instead of routing through expand
            # in the defensive/missing-entry case.
            if hi < len(all_turned_away) and len(all_turned_away[hi]) > 0:
                return _expand_bin_mask_to_frames(all_turned_away[hi], fps, n_frames)
            return np.zeros(n_frames, dtype=bool)

        _hmm_bin_proba_train = all_bin_proba
        _hmm_bin_labels_train = all_bin_labels
        _hmm_frame_labels_train = all_frame_labels
        if _exclude_turned_away and all_turned_away:
            if _soft_ok:
                _seg_proba, _seg_bin_lbl = [], []
                for _hi in range(len(all_bin_proba)):
                    _ta_m = _ta_bin_mask_for(_hi, len(all_bin_proba[_hi]))
                    _seg_proba.extend(_mask_out_segments(all_bin_proba[_hi], _ta_m))
                    _seg_bin_lbl.extend(_mask_out_segments(all_bin_labels[_hi], _ta_m))
                if _seg_proba:
                    _hmm_bin_proba_train, _hmm_bin_labels_train = _seg_proba, _seg_bin_lbl
                else:
                    self._log("  [WARN] turned-away exclusion left no bins for "
                              "HMM training (entire dataset flagged turned-away?) "
                              "-- falling back to the unsegmented sequences.")
            elif _bin_level_ok:
                _seg_bin_lbl = []
                for _hi in range(len(all_bin_labels)):
                    _ta_m = _ta_bin_mask_for(_hi, len(all_bin_labels[_hi]))
                    _seg_bin_lbl.extend(_mask_out_segments(all_bin_labels[_hi], _ta_m))
                if _seg_bin_lbl:
                    _hmm_bin_labels_train = _seg_bin_lbl
                else:
                    self._log("  [WARN] turned-away exclusion left no bins for "
                              "HMM training (entire dataset flagged turned-away?) "
                              "-- falling back to the unsegmented sequences.")
            else:
                _seg_frame_lbl = []
                for _hi in range(len(all_frame_labels)):
                    _ta_frame_m = _ta_frame_mask_for(
                        _hi, all_fps_list[_hi], len(all_frame_labels[_hi]))
                    _seg_frame_lbl.extend(
                        _mask_out_segments(all_frame_labels[_hi], _ta_frame_m))
                if _seg_frame_lbl:
                    _hmm_frame_labels_train = _seg_frame_lbl
                else:
                    self._log("  [WARN] turned-away exclusion left no frames for "
                              "HMM training (entire dataset flagged turned-away?) "
                              "-- falling back to the unsegmented sequences.")

        if self._cfg.get("hmm_enabled", True) and all_frame_labels:
            try:
                _t0 = time.perf_counter()
                _hmm_n_states = self._cfg.get("hmm_n_states") or None
                if _hmm_n_states is not None:
                    _hmm_n_states = int(_hmm_n_states)
                if _soft_ok:
                    self._log("\n[HMM]  Training soft-emission GaussianHMM "
                              "on MLP per-bin probability vectors "
                              "(hmm_emission_mode='soft')...")
                    hmm_model = train_hmm_soft(
                        _hmm_bin_proba_train,
                        n_clusters=int(n_cl),
                        n_states=_hmm_n_states,
                        n_iter=int(self._cfg.get("hmm_n_iter", 100)),
                        log_fn=self._log,
                        transition_prior=str(self._cfg.get("hmm_transition_prior", "global")),
                        bin_label_sequences=_hmm_bin_labels_train,
                        random_state=int(self._cfg.get("hmm_random_state", 42)),
                    )
                elif _bin_level_ok:
                    self._log("\n[HMM]  Training Multinomial HMM on MLP "
                              "label sequences (bin-level resolution, "
                              "hmm_smoothing_level='bin')...")
                    hmm_model = train_hmm(
                        _hmm_bin_labels_train,
                        n_clusters=int(n_cl),
                        n_states=_hmm_n_states,
                        n_iter=int(self._cfg.get("hmm_n_iter", 100)),
                        log_fn=self._log,
                        transition_prior=str(self._cfg.get("hmm_transition_prior", "global")),
                        random_state=int(self._cfg.get("hmm_random_state", 42)),
                    )
                else:
                    self._log("\n[HMM]  Training Multinomial HMM on MLP label sequences...")
                    hmm_model = train_hmm(
                        _hmm_frame_labels_train,
                        n_clusters=int(n_cl),
                        n_states=_hmm_n_states,
                        n_iter=int(self._cfg.get("hmm_n_iter", 100)),
                        log_fn=self._log,
                        transition_prior=str(self._cfg.get("hmm_transition_prior", "global")),
                        random_state=int(self._cfg.get("hmm_random_state", 42)),
                    )
                self._log(f"  HMM trained in {time.perf_counter() - _t0:.2f} s  "
                          f"({hmm_model.n_components} states, Baum-Welch)")
                for _hi, (_raw, _name, _file_fps) in enumerate(zip(
                        all_frame_labels, all_names, all_fps_list)):
                    # _hmm_labels is decoded from the RAW (unoverridden) MLP
                    # label sequence — hmmlearn's CategoricalHMM was fit with
                    # exactly n_cl categories, so decode_hmm's input/output
                    # here must never contain the reserved turned-away id.
                    # The dedicated-label override is applied afterward, only
                    # to the CSV-export copy, exactly mirroring the raw-output
                    # handling above.
                    if _soft_ok:
                        _decoded_bins = decode_hmm_soft(hmm_model, all_bin_proba[_hi])
                        _win = all_bin_win[_hi]
                        _hmm_labels = np.repeat(_decoded_bins, _win)
                        _n_orig = len(_raw)
                        if len(_hmm_labels) < _n_orig:
                            _hmm_labels = np.pad(
                                _hmm_labels, (0, _n_orig - len(_hmm_labels)),
                                mode="edge")
                        _hmm_labels = _hmm_labels[:_n_orig].astype(int)
                    elif _bin_level_ok:
                        # Decode at bin resolution, then expand back to frame
                        # resolution the SAME way predict_labels() expands
                        # raw bin labels (np.repeat(.., win), edge-pad to the
                        # original frame count) -- so every downstream
                        # consumer (CSV export, labeled video, epochs/bouts)
                        # sees a frame-length array exactly as it does today.
                        _decoded_bins = decode_hmm(hmm_model, all_bin_labels[_hi])
                        _win = all_bin_win[_hi]
                        _hmm_labels = np.repeat(_decoded_bins, _win)
                        _n_orig = len(_raw)
                        if len(_hmm_labels) < _n_orig:
                            _hmm_labels = np.pad(
                                _hmm_labels, (0, _n_orig - len(_hmm_labels)),
                                mode="edge")
                        _hmm_labels = _hmm_labels[:_n_orig].astype(int)
                    else:
                        _hmm_labels = decode_hmm(hmm_model, _raw)
                    all_hmm_labels.append(_hmm_labels)
                    _hmm_ta_bin_mask = (all_turned_away[_hi]
                                        if _hi < len(all_turned_away)
                                        else np.zeros(0, dtype=bool))
                    _hmm_ta_frame_mask = _expand_bin_mask_to_frames(
                        _hmm_ta_bin_mask, _file_fps, len(_hmm_labels))
                    _hmm_display = _hmm_labels.copy()
                    _hmm_reason = np.full(len(_hmm_display), "", dtype=object)
                    if _exclude_turned_away and _hmm_ta_frame_mask.any():
                        _hmm_display[_hmm_ta_frame_mask] = int(n_cl)
                        _hmm_reason[_hmm_ta_frame_mask] = "confidence"
                    # Auto-flagged impure clusters (see auto_flag_impure_clusters
                    # above) must be folded in here too -- cube_analyser.py's
                    # _prefer_hmm() loads these *_hmm files over the raw ones
                    # whenever HMM smoothing is on (the default), so without
                    # this an auto-flagged cluster would still show up as a
                    # real behaviour in exactly the files the analyser
                    # actually reads, defeating the whole point of flagging it.
                    if _auto_impure_ids:
                        _hmm_impure_mask = np.isin(_hmm_labels, sorted(_auto_impure_ids))
                        if _hmm_impure_mask.any():
                            _hmm_display[_hmm_impure_mask] = int(n_cl)
                            _hmm_reason[_hmm_impure_mask] = "auto_impure_cluster"
                    _bout_hmm = labels_to_bouts(_hmm_display)
                    _bout_hmm.to_csv(
                        str(self._out_bouts / f"{_name}_bout_lengths_hmm.csv"),
                        index=False)
                    # v6 K2 sidecar (opt-in, off by default): per-bout
                    # kinematic directedness metrics, written as a SEPARATE
                    # file so the canonical 3-column bout CSV above is never
                    # touched. See kinematic_directedness_enabled in DEFAULTS.
                    if bool(self._cfg.get("kinematic_directedness_enabled", False)):
                        try:
                            _xy_k = all_xy[_hi]
                            _centroid_xy = np.column_stack([
                                _xy_k[:, 0::2].mean(axis=1),
                                _xy_k[:, 1::2].mean(axis=1)])
                            _bout_hmm_enriched = compute_bout_directedness(
                                _bout_hmm, _centroid_xy, _file_fps)
                            _bout_hmm_enriched.to_csv(
                                str(self._out_bouts /
                                    f"{_name}_bout_lengths_hmm_enriched.csv"),
                                index=False)
                        except Exception:
                            self._log(
                                f"  [WARN] kinematic directedness sidecar "
                                f"({_name}): {traceback.format_exc()}")
                    pd.DataFrame({
                        "frame":               np.arange(len(_hmm_display)),
                        "time_s":              np.arange(len(_hmm_display)) / _file_fps,
                        "label":               _hmm_display,
                        "raw_label":           _hmm_labels,
                        "turned_away_reason":  _hmm_reason,
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

        self._mem_checkpoint("7/7 per-session prediction + export done")

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
        # Uses the turned-away DISPLAY labels (reserved id override applied),
        # built fresh here rather than reusing all_frame_labels (which stays
        # raw/unoverridden throughout — it already fed HMM training above,
        # and hmmlearn's CategoricalHMM would reject/corrupt on an id outside
        # its fitted n_clusters categories) so "Turned Away" gets its own row
        # in cluster_kinematics.csv rather than being folded into whatever
        # real cluster the MLP guessed for those bins.
        if all_frame_labels and all_xy:
            try:
                if _exclude_turned_away:
                    _kin_labels = []
                    for _ki, (_fl, _xy_k, _fps_k) in enumerate(
                            zip(all_frame_labels, all_xy, all_fps_list)):
                        _kin_ta_mask = (all_turned_away[_ki]
                                        if _ki < len(all_turned_away)
                                        else np.zeros(0, dtype=bool))
                        _kin_ta_frame_mask = _expand_bin_mask_to_frames(
                            _kin_ta_mask, _fps_k, _xy_k.shape[0])
                        _disp = _fl.copy()
                        if _kin_ta_frame_mask.any():
                            _n_copy_k = min(len(_disp), len(_kin_ta_frame_mask))
                            _disp[:_n_copy_k][_kin_ta_frame_mask[:_n_copy_k]] = int(n_cl)
                        _kin_labels.append(_disp)
                else:
                    _kin_labels = all_frame_labels
                compute_cluster_kinematics(
                    all_xy, _kin_labels, all_fps_list, bps_ref,
                    self.output_dir / "cluster_kinematics.csv")
                self._log("  [PLOT] cluster_kinematics.csv saved")
            except Exception:
                self._log(f"  [WARN] cluster_kinematics: "
                          f"{traceback.format_exc()}")

        # Per-cluster confidence/visibility profile (issue 2) — flags clusters
        # that are mostly "animal turned away / occluded" rather than a real
        # behaviour, so they are not presented to the user as one.
        if vis_cat is not None:
            try:
                compute_cluster_confidence_profile(
                    vis_cat, _vis_col_names, hdb_labels_all,
                    self.output_dir / "cluster_confidence.csv")
                self._log("  [PLOT] cluster_confidence.csv saved")
            except Exception:
                self._log(f"  [WARN] cluster_confidence: "
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

        # Turned-away-excluded sequences for transition-counting plots (same
        # rationale/mechanism as the HMM training segmentation above) --
        # transition_matrix.png and the umap_3d.html transition overlay must
        # not count a transition into/out of a turned-away frame. No-op when
        # exclude_turned_away=False.
        _transition_frame_labels = all_frame_labels
        if _exclude_turned_away and (all_turned_away or _auto_impure_ids) and all_frame_labels:
            _seg_trans = []
            for _hi in range(len(all_frame_labels)):
                _ta_frame_m = _ta_frame_mask_for(
                    _hi, all_fps_list[_hi], len(all_frame_labels[_hi]))
                # Auto-flagged impure clusters (see auto_flag_impure_clusters)
                # are just as behaviourally meaningless for transition
                # counting as confidence-based turned-away bins -- fold their
                # frames into the same exclusion mask so neither shows up as
                # a phantom source/target in transition_matrix.png or the
                # umap_3d.html transition overlay.
                if _auto_impure_ids:
                    _ta_frame_m = _ta_frame_m | np.isin(
                        all_frame_labels[_hi], sorted(_auto_impure_ids))
                _seg_trans.extend(
                    _mask_out_segments(all_frame_labels[_hi], _ta_frame_m))
            if _seg_trans:
                _transition_frame_labels = _seg_trans
            else:
                self._log("  [WARN] turned-away/impure-cluster exclusion left "
                          "no frames for transition-matrix/UMAP-transition "
                          "plots (entire dataset flagged?) -- falling back to "
                          "the unsegmented sequences.")

        if self._cfg["save_plots"] and all_frame_labels:
            try:
                plot_transition_matrix(
                    _transition_frame_labels,
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
                _tmat, _cids = _tmat_from_labels(_transition_frame_labels)
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

        # ── Publication benchmark metrics (Section 3.2) ───────────────────────
        _run_elapsed   = time.perf_counter() - _run_t0
        _peak_mem_gb   = _tm.get_traced_memory()[1] / 1e9
        _tm.stop()

        # Total video duration in minutes: sum of bins × 100 ms per bin.
        # all_frame_labels contains per-session HMM/MLP label arrays (1 per bin).
        _total_vid_min = None
        try:
            _total_bins    = sum(len(lbl) for lbl in all_frame_labels)
            _bin_sec       = 0.1   # 100-ms bins throughout CUBE
            _total_vid_min = (_total_bins * _bin_sec) / 60.0
        except Exception:
            pass
        _runtime_min = (_run_elapsed / 60.0) / max(1.0, _total_vid_min or 1.0)

        self._mem_checkpoint("run complete")
        _pub_metrics = dict(
            silhouette      = _validation.get("clustering",  {}).get("silhouette_score"),
            trustworthiness = _validation.get("umap_trustworthiness", {}).get("trustworthiness"),
            mean_ari        = _validation.get("cluster_stability", {}).get("mean_ari"),
            runtime_min     = round(_runtime_min, 4),
            # peak_memory_gb: tracemalloc-based, Python-heap allocations only.
            # peak_rss_gb: psutil-based process RSS, also covers numpy/BLAS/
            # native temporaries (the majority of real usage in this pipeline)
            # -- see _mem_checkpoint(). Both reported; neither replaces the other.
            peak_memory_gb  = round(_peak_mem_gb, 3),
            peak_rss_gb     = round(getattr(self, "_peak_rss_gb", 0.0), 3),
            total_runtime_s = round(_run_elapsed, 1),
            total_video_min = round(_total_vid_min, 2) if _total_vid_min else None,
        )
        self._log(f"\n  [Benchmark] Silhouette={_pub_metrics['silhouette']}  "
                  f"Trust={_pub_metrics['trustworthiness']}  "
                  f"ARI={_pub_metrics['mean_ari']}  "
                  f"Runtime={_pub_metrics['runtime_min']:.3f} min/min  "
                  f"RAM={_pub_metrics['peak_memory_gb']:.2f} GB "
                  f"(peak RSS {_pub_metrics['peak_rss_gb']:.2f} GB)")

        (self.output_dir / "publication_metrics.json").write_text(
            json.dumps(_pub_metrics, indent=2))

        if self._cfg["save_plots"]:
            try:
                plot_publication_metrics(
                    _pub_metrics,
                    self._out_plots / "publication_metrics.png")
            except Exception:
                self._log(f"  [WARN] publication_metrics plot: "
                          f"{traceback.format_exc()}")

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
            # turned_away_cluster_id: the reserved display id (cube_core.py's
            # own "Turned Away" pseudo-cluster, see exclude_turned_away) used
            # in the exported bout/frame CSVs -- lets cube_analyser.py filter
            # it out of dwell-time/transition/reclustering analysis at load
            # time. None when exclude_turned_away=False (no reserved id used).
            turned_away_cluster_id = int(n_cl) if _exclude_turned_away else None,
            # auto_flagged_impure_cluster_ids: real HDBSCAN/consensus cluster
            # ids (in the pre-Turned-Away-remap id space) that
            # auto_flag_impure_clusters folded into turned_away_cluster_id
            # this run -- see the [IMPURE] log line for each one's silhouette.
            # Empty list when the pass didn't fire (disabled, no candidates,
            # or the "don't leave <2 real clusters" guard tripped). Recorded
            # here (not just logged) since the decision isn't reproducible
            # from the saved model alone -- predict_from_saved_model() does
            # NOT re-derive or re-apply it for later inference-only runs.
            auto_flagged_impure_cluster_ids = sorted(int(c) for c in _auto_impure_ids),
            cv_accuracy   = float(cv_scores.mean()) if mlp_clf else None,
            cfg           = self._cfg,
            output_dir    = str(self.output_dir),
            bout_lengths  = [str(p) for p in bout_paths],
            frame_labels  = [str(p) for p in frame_paths],
            model         = str(model_path),
            feature_version = "v2",
            validation    = val_report["overall_status"],
            benchmark     = _pub_metrics,
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
                # return_quality=True so the raw per-frame likelihood array
                # (ll) is available for the visibility feature block below —
                # REQUIRED for train/inference symmetry since
                # visibility_features_enabled defaults to True, so most saved
                # models were trained with the visibility block folded into
                # their expected feature layout.
                xy, _, _, _, _, _ll = load_dlc_file(
                    fp, m["cfg"]["likelihood_thresh"], return_quality=True)
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
                angular_fallback=bool(_mcfg.get("angular_fallback", True)),
                long_lag_drift=bool(_mcfg.get("long_lag_drift", False)),
                long_scale_bins=bool(_mcfg.get("long_scale_bins", False)),
                bodypart_weights=_mcfg.get("bodypart_weights") or None,
                ll=_ll,
                visibility_features_enabled=bool(
                    _mcfg.get("visibility_features_enabled", True)),
                visibility_adaptive_pct=float(
                    _mcfg.get("visibility_adaptive_pct", 10)),
                likelihood_thresh=float(m["cfg"]["likelihood_thresh"]))
            bd    = labels_to_bouts(fl)
            bp    = engine._out_bouts / f"{fp.stem}_bout_lengths.csv"
            bd.to_csv(str(bp), index=False)
            bout_paths.append(bp)
            # v6 K2 sidecar consistency: this path has no _hmm bout variant
            # at all (predict_from_saved_model skips HMM smoothing), so the
            # enriched sidecar pairs with the raw bout CSV above --
            # "<stem>_bout_lengths_enriched.csv", not "..._hmm_enriched.csv".
            if bool(_mcfg.get("kinematic_directedness_enabled", False)):
                try:
                    _centroid_xy = np.column_stack([
                        xy[:, 0::2].mean(axis=1), xy[:, 1::2].mean(axis=1)])
                    _bd_enriched = compute_bout_directedness(bd, _centroid_xy, fps)
                    _bd_enriched.to_csv(
                        str(engine._out_bouts / f"{fp.stem}_bout_lengths_enriched.csv"),
                        index=False)
                except Exception:
                    log(f"  [WARN] kinematic directedness sidecar "
                        f"({fp.stem}): {traceback.format_exc()}")
            fd    = pd.DataFrame({"frame": np.arange(len(fl)),
                                  "time_s": np.arange(len(fl)) / fps,
                                  "label": fl})
            fp2   = engine._out_bouts / f"{fp.stem}_frame_labels.csv"
            fd.to_csv(str(fp2), index=False)
            frame_paths.append(fp2)
        return dict(bout_lengths_paths=bout_paths,
                    frame_label_paths=frame_paths,
                    output_dir=engine.output_dir)
