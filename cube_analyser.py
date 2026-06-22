# -*- coding: utf-8 -*-
"""
Created on Sat May 16 01:16:59 2026

@author: param


B-SOiD Behavioral Analysis Suite
========================================

New in v6:
    FIX: Colour changes in the Group Editor and EG Color picker are now
         correctly reflected in ALL graphs (combined, top-N, recombination).
    Reclustering now incorporates TRANSITION DYNAMICS:
        Average inter-cluster transition probability matrices are computed per
        animal and averaged across the cohort.
        Cosine similarity of outgoing transition profiles (which clusters does
        C_i most commonly precede?) is blended with the change-pattern
        correlation distance (70% change-pattern + 30% transition proximity).
        Clusters that co-occur in time are pulled together, making merged groups
        more biologically coherent.
    NEW PLOTS in Unbiased Analytics:
        "Dist Matrix"   : 3-panel correlation / transition / blended distance
                          matrices, ordered by dendrogram.
        "Transitions"   : Average transition heatmap + outgoing-profile cosine
                          similarity matrix.
        "Cluster Stats" : Per-cluster mean bout, total duration, frequency with
                          filtered clusters shown as hollow bars.
    Separate y-scale per behaviour group in combined bar charts so small
    differences in quiet behaviours are visible alongside large ones.
    Stars on ALL bar graphs denoting Kruskal-Wallis significance.
    Biological relevance filter: set minimum mean bout duration (ms),
    minimum total duration (s), or minimum frequency before reclustering
    to discard clusters that are too brief or too rare to be meaningful.

New in v4 (retained):
    TSV import, colour picker, drag-and-drop reordering, Unbiased Analytics tab.
"""

#   stdlib
import os as _os_early
import sys as _sys_early

# Force BLAS/MKL to single-threaded mode unconditionally.  setdefault() is NOT
# used because conda already sets these variables; without overriding them, loky
# workers (spawned fresh on Windows) can inherit a multi-threaded config and
# race on MKL pool initialisation, causing segfaults.  With numpy 2.x the
# threading model changed and the race window is wider.
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    _os_early.environ[_k] = "1"

# When run from a desktop shortcut without `conda activate`, the conda env's
# DLL directory is absent from PATH.  Loky workers are spawned fresh on Windows
# (spawn, not fork) and inherit this truncated PATH, so `import numpy` inside
# each worker can't find mkl_intel_thread.dll and raises a segfault.
# Prepend the correct dirs now — workers will inherit this corrected PATH.
if _sys_early.platform == "win32":
    _py_dir = _os_early.path.dirname(_os_early.path.abspath(_sys_early.executable))
    _path_lower = _os_early.environ.get("PATH", "").lower()
    _extra = _os_early.pathsep.join(
        d for d in (
            _os_early.path.join(_py_dir, "Library", "bin"),
            _os_early.path.join(_py_dir, "Library", "mingw-w64", "bin"),
        )
        if _os_early.path.isdir(d) and d.lower() not in _path_lower
    )
    if _extra:
        _os_early.environ["PATH"] = _extra + _os_early.pathsep + _os_early.environ.get("PATH", "")

del _os_early, _sys_early

import json
import pathlib
import re
import sys
import traceback
from datetime import datetime

#   third-party  
try:
    import customtkinter as ctk
except ImportError:
    sys.exit("customtkinter not found.  pip install customtkinter")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox

try:
    from scipy import stats as sp_stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import squareform
    SCIPY_CLUSTER_OK = True
except ImportError:
    SCIPY_CLUSTER_OK = False
    squareform = None  # type: ignore

try:
    from sklearn.metrics import silhouette_score
    SK_OK = True
except ImportError:
    SK_OK = False

#  
# CONSTANTS
#  
DEFAULT_FPS    = 30
APP_TITLE      = "CUBE Behavioral Analysis Suite"
BSOID_SUBDIR   = "BSOID"
RESULTS_SUBDIR = "Analysis_Results"

# Minimum number of animals required before PCA is inserted into the
# GroupPredictor pipeline.  Below this threshold each LOO training set has
# fewer than (_MIN_PCA_ANIMALS - 1) samples; PCA components derived from so
# few points are noise-dominated and statistically unreliable.  The Elastic
# Net's own L1 regularisation handles high-dimensional inputs (e.g. the 272
# Transition features) better than PCA would for small cohorts.
_MIN_PCA_ANIMALS = 20

PALETTE = [
    "#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
    "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC",
    "#00BCD4","#FF5722","#8BC34A","#9C27B0","#FFC107",
    "#3F51B5","#009688","#FF4081","#CDDC39","#795548",
]

THEMES = {
    "dark": {
        "bg":          "#0d0d1a",
        "panel":       "#12121f",
        "card":        "#1a1a2e",
        "card2":       "#16162a",
        "text":        "white",
        "subtext":     "#aaaacc",
        "muted":       "#555566",
        "border":      "#333355",
        "fig_bg":      "#0d0d1a",
        "ax_bg":       "#1a1a2e",
        "track_bg":    "#2d2d44",
        "spine":       "#444466",
        "tick":        "white",
        "btn_del":     "#5a1e1e",
        "btn_del_h":   "#8b2020",
        "btn_add":     "#1a3a1a",
        "btn_folder":  "#1a3a5a",
        "btn_save":    "#1a3a1e",
        "btn_load":    "#2a3a5a",
        "btn_unbiased":"#2a1a4a",
        "hdr_bg":      "#1a1a35",
        "hdr_text":    "#aaaaff",
        "row_even":    "#1e1e32",
        "row_odd":     "#16162a",
        "ctk_mode":    "dark",
        "mpl_style":   "dark_background",
        "drag_highlight": "#3a3a66",
    },
    "light": {
        "bg":          "#f0f2f8",
        "panel":       "#ffffff",
        "card":        "#e8ecf4",
        "card2":       "#dde2ef",
        "text":        "#1a1a2e",
        "subtext":     "#444466",
        "muted":       "#999bb0",
        "border":      "#c0c4d8",
        "fig_bg":      "#f5f6fa",
        "ax_bg":       "#ffffff",
        "track_bg":    "#dde2ef",
        "spine":       "#c0c4d8",
        "tick":        "#1a1a2e",
        "btn_del":     "#c0504d",
        "btn_del_h":   "#a83532",
        "btn_add":     "#3d8b3d",
        "btn_folder":  "#2a7ab8",
        "btn_save":    "#2d8a60",
        "btn_load":    "#4a6eb0",
        "btn_unbiased":"#6c3a99",
        "hdr_bg":      "#e0e4f4",
        "hdr_text":    "#3344aa",
        "row_even":    "#eef0fa",
        "row_odd":     "#f8f9fd",
        "ctk_mode":    "light",
        "mpl_style":   "seaborn-v0_8-whitegrid",
        "drag_highlight": "#d0d8f0",
    },
}

_THEME_KEY = "dark"

def T() -> dict:
    return THEMES[_THEME_KEY]


#  
# DATA LAYER - CSV + TSV loading
#  

def extract_fps(path: pathlib.Path) -> int:
    m = re.search(r"(\d+)Hz", path.name, re.IGNORECASE)
    return int(m.group(1)) if m else DEFAULT_FPS


def _is_hmm_file(p: pathlib.Path) -> bool:
    return "_hmm" in p.stem.lower()


def _prefer_hmm(files: list) -> list:
    """
    Given a mixed list of bout_lengths CSV paths, return HMM-smoothed versions
    when available, otherwise fall back to raw versions.

    Rule: if EVERY raw file has a corresponding *_hmm counterpart, return the
    HMM set (so the analyser uses the temporally-smoothed labels).  If only
    some sessions have HMM files, still return the full HMM set (those sessions
    were processed with HMM; the others weren't).  Never return a mix.
    """
    hmm_files = [p for p in files if _is_hmm_file(p)]
    raw_files  = [p for p in files if not _is_hmm_file(p)]
    if hmm_files:
        return sorted(hmm_files)
    return sorted(raw_files)


def find_bsoid_files(root: pathlib.Path) -> list:
    """
    Return sorted list of bout_lengths CSV/TSV files.

    Priority: HMM-smoothed (*_bout_lengths_hmm.csv) files are preferred over
    raw (*_bout_lengths.csv) files when both are present in the same folder.
    This ensures the temporally-consistent HMM labels produced by CUBE's
    post-hoc HMM pass are used for analysis rather than the noisier raw MLP
    output.  To force raw labels, delete or move the _hmm files.

    Search order: root/BSOID/, root/, any bout_lengths/ subfolder,
    then any *bout_lengths*.csv anywhere under root.
    """
    # 1. Standard locations
    for d in (root / BSOID_SUBDIR, root):
        if d.is_dir():
            hits = sorted(
                p for p in list(d.glob("*.csv")) + list(d.glob("*.tsv"))
                if "bout_lengths" in p.name.lower()
            )
            if hits:
                return _prefer_hmm(hits)

    # 2. Any subdirectory named "bout_lengths" (e.g. cube_results/bout_lengths/)
    for candidate in sorted(root.rglob("bout_lengths")):
        if candidate.is_dir():
            hits = sorted(
                p for p in list(candidate.glob("*.csv")) + list(candidate.glob("*.tsv"))
                if "bout_lengths" in p.name.lower()
            )
            if hits:
                return _prefer_hmm(hits)

    # 3. Full recursive scan for any *bout_lengths*.csv / .tsv file
    hits = sorted(p for p in root.rglob("*bout_lengths*.csv"))
    if not hits:
        hits = sorted(p for p in root.rglob("*bout_lengths*.tsv"))
    return _prefer_hmm(hits)


def find_cluster_mapping(root: pathlib.Path) -> pathlib.Path | None:
    """
    Search for a cluster_behaviour_mapping.tsv produced by Video Explorer.
    Checks root recursively first, then walks up to the grandparent directory
    so the file is found even when Phase 4 exported to a sibling folder.
    """
    for p in sorted(root.rglob("cluster_behaviour_mapping.tsv")):
        return p
    # Walk up to grandparent (2 levels) to catch exports next to cube_results/
    for ancestor in [root.parent, root.parent.parent]:
        if ancestor == root or not ancestor.is_dir():
            continue
        for p in sorted(ancestor.rglob("cluster_behaviour_mapping.tsv")):
            return p
    return None


def find_phase4_session_json(root: pathlib.Path) -> pathlib.Path | None:
    """
    Search for a Phase 4 (Video Explorer) session JSON file.
    These contain a 'behaviour_groups' key and are saved to a user-chosen
    location.  Searches root recursively then parent/grandparent dirs.
    Returns the first match, or None.
    """
    search_roots = [root, root.parent, root.parent.parent]
    for sr in search_roots:
        if not sr.is_dir():
            continue
        for p in sorted(sr.rglob("*.json")):
            if p.stat().st_size > 50_000_000:
                continue  # skip huge files
            try:
                import json as _json
                data = _json.loads(p.read_text(encoding="utf-8", errors="ignore")[:8192])
                if isinstance(data, dict) and "behaviour_groups" in data:
                    return p
            except Exception:
                continue
    return None


def load_groups_from_phase4_session(path: pathlib.Path) -> dict:
    """
    Convert a Phase 4 (Video Explorer) session JSON into the analyser's
    groups format: {group_name: {"labels": [int, ...], "color": str}}.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    behaviour_groups = data.get("behaviour_groups", {})
    if not behaviour_groups:
        raise ValueError("No behaviour_groups found in session JSON.")
    groups: dict = {}
    pal_idx = 0
    for _uid, bgd in behaviour_groups.items():
        name = str(bgd.get("name", f"Group {_uid}")).strip()
        if not name:
            continue
        colour = bgd.get("colour") or bgd.get("color") or PALETTE[pal_idx % len(PALETTE)]
        cluster_ids = [int(c) for c in bgd.get("cluster_ids", [])]
        if name not in groups:
            groups[name] = {"labels": cluster_ids, "color": colour}
            pal_idx += 1
        else:
            groups[name]["labels"].extend(cluster_ids)
    return groups


def groups_from_all_clusters(all_labels: list) -> dict:
    """
    Build a one-group-per-cluster mapping when no behaviour groups have been
    defined.  Each cluster gets its own group named "Cluster N".
    """
    groups: dict = {}
    for i, lbl in enumerate(sorted(all_labels)):
        groups[f"Cluster {lbl}"] = {
            "labels": [int(lbl)],
            "color":  PALETTE[i % len(PALETTE)],
        }
    return groups


def find_umap_data(root: pathlib.Path):
    """
    Search for umap_embedding.npy and umap_labels.npy saved by cube_core.

    Strategy (most-specific first):
    1. Recurse downward from root (covers the case where the user selects the
       cube_results_* folder or any ancestor of it).
    2. Walk up to 4 parent levels and recurse from each — catches the common
       case where the analyser root is set to the bout_lengths/ subdirectory
       while model/ is a sibling directory at the cube_results_* level.

    Returns the pair whose embedding file is deepest in the most-recent
    cube_results_* tree, or (None, None) if nothing is found.
    """
    def _search(directory: pathlib.Path):
        """Return (emb, lbl) pair from the most-recently-modified model dir."""
        emb_candidates = sorted(directory.rglob("umap_embedding.npy"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
        for emb_p in emb_candidates:
            lbl_p = emb_p.with_name("umap_labels.npy")
            if lbl_p.exists():
                return emb_p, lbl_p
        return None, None

    # Try the given root first
    emb, lbl = _search(root)
    if emb and lbl:
        return emb, lbl

    # Walk up parent directories (up to 4 levels) and re-search from each
    candidate = root.resolve()
    for _ in range(4):
        parent = candidate.parent
        if parent == candidate:
            break           # filesystem root
        candidate = parent
        emb, lbl = _search(candidate)
        if emb and lbl:
            return emb, lbl

    return None, None


def load_csv(path: pathlib.Path) -> pd.DataFrame:
    """Load a B-SOiD bout_lengths CSV (original format). Raises ValueError on error."""
    required = {"B-SOiD labels", "Start time (frames)", "Run lengths"}
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    _FAST_COLS = ["B-SOiD labels", "Start time (frames)", "Run lengths"]
    try:
        try:
            df = pd.read_csv(path, sep=sep, usecols=_FAST_COLS)
        except Exception:
            df = pd.read_csv(path, sep=sep)
    except Exception as e:
        raise ValueError(f"Cannot read {path.name}: {e}") from e
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")
    df = df.rename(columns={
        "B-SOiD labels":       "label",
        "Start time (frames)": "start_frame",
        "Run lengths":         "run_len",
    })
    for col in ("label", "start_frame", "run_len"):
        df[col] = df[col].astype(int)
    return df.sort_values("start_frame").reset_index(drop=True)


def load_mapping_tsv(path: pathlib.Path) -> dict:
    """
    Load a cluster_behaviour_mapping.tsv produced by Video Explorer v2+.
    Returns: {behaviour_group_name: {"labels": [int,...], "color": str}}
    TSV columns: bsoid_group_id, behaviour_group, status, example_count, gif_path, mp4_paths
    """
    sep = "\t"
    try:
        df = pd.read_csv(path, sep=sep)
    except Exception as e:
        raise ValueError(f"Cannot read TSV {path.name}: {e}") from e

    required = {"bsoid_group_id", "behaviour_group", "status"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"TSV missing columns: {missing}")

    groups: dict = {}
    for _, row in df.iterrows():
        if str(row.get("status", "")).strip() != "assigned":
            continue
        bg_name = str(row["behaviour_group"]).strip()
        if not bg_name:
            continue
        gid = int(row["bsoid_group_id"])
        if bg_name not in groups:
            groups[bg_name] = {"labels": [], "color": PALETTE[len(groups) % len(PALETTE)]}
        groups[bg_name]["labels"].append(gid)

    return groups


#  
# PER-CLUSTER METRIC EXTRACTION  (used by Unbiased Analytics)
#  

def compute_per_cluster_metrics(df: pd.DataFrame, fps: int) -> pd.DataFrame:
    """
    Compute per-cluster (individual B-SOiD label) metrics for one animal.
    Returns DataFrame: cluster_id, total_duration, frequency, mean_bout
    """
    rows = []
    for label in df["label"].unique():
        sub   = df[df["label"] == label]
        durs  = sub["run_len"].values / fps
        rows.append({
            "cluster_id":     int(label),
            "total_duration": float(durs.sum()),
            "frequency":      len(durs),
            "mean_bout":      float(durs.mean()),
        })
    return pd.DataFrame(rows).set_index("cluster_id")


def compute_per_pair_transition_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Per directed pair (Ci→Cj) row-normalised transition probability for one animal.

    Returns a DataFrame indexed by 'Ci→Cj' strings with column 'transition_prob'.
    Only pairs with observed probability > 0 are included; structural zeros
    (pairs never taken by this animal) are absent and will be filled as 0.0
    by run_cluster_statistics, consistent with how cluster metrics are handled.
    Uses compute_transition_matrix which already row-normalises so each row sums
    to 1.0 across genuine (off-diagonal) transitions.
    """
    mat, cids = compute_transition_matrix(df)
    rows = []
    for ri, ci in enumerate(cids):
        for rj, cj in enumerate(cids):
            if ci != cj and mat[ri, rj] > 0:
                rows.append({"cluster_id": f"C{ci}→C{cj}",
                             "transition_prob": float(mat[ri, rj])})
    if not rows:
        return pd.DataFrame(columns=["transition_prob"]).rename_axis("cluster_id")
    return pd.DataFrame(rows).set_index("cluster_id")


#
# STANDARD BOUT METRICS
#  

def merge_bouts(df: pd.DataFrame, labels: set) -> list:
    sub = df[df["label"].isin(labels)][["start_frame", "run_len"]].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("start_frame")
    starts = sub["start_frame"].to_numpy()
    ends   = (sub["start_frame"] + sub["run_len"]).to_numpy()
    merged_starts, merged_ends = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s <= merged_ends[-1] + 1:
            merged_ends[-1] = max(merged_ends[-1], e)
        else:
            merged_starts.append(s)
            merged_ends.append(e)
    return [{"start": int(s), "end": int(e)}
            for s, e in zip(merged_starts, merged_ends)]


def compute_metrics(df: pd.DataFrame, groups: dict, fps: int) -> dict:
    out = {}
    for gname, ginfo in groups.items():
        evs = merge_bouts(df, set(ginfo["labels"]))
        if not evs:
            out[gname] = dict(total_duration=0.0, frequency=0,
                              latency=None, mean_bout=0.0, events=[])
            continue
        durs = [(e["end"] - e["start"]) / fps for e in evs]
        for e, d in zip(evs, durs):
            e.update(start_s=e["start"] / fps, end_s=e["end"] / fps, dur_s=d)
        out[gname] = dict(
            total_duration=round(sum(durs), 3),
            frequency=len(evs),
            latency=round(evs[0]["start"] / fps, 3),
            mean_bout=round(float(np.mean(durs)), 3),
            events=evs,
        )
    return out


def session_duration(df: pd.DataFrame, fps: int) -> float:
    r = df.loc[df["start_frame"].idxmax()]
    return (int(r["start_frame"]) + int(r["run_len"])) / fps


def compute_combined(animal_data: list, groups: dict) -> dict:
    """
    Compute per-animal metrics and aggregate by experimental group.
    Returns:
      records, uid_to_idx, exp_groups, grand
    """
    if not animal_data:
        raise ValueError("compute_combined: animal_data is empty.")
    required_keys = {"uid", "name", "df", "fps", "exp_group"}
    for i, ani in enumerate(animal_data):
        missing = required_keys - set(ani.keys())
        if missing:
            raise ValueError(f"Animal [{i}] missing keys: {missing}")
        if not isinstance(ani["df"], pd.DataFrame) or ani["df"].empty:
            raise ValueError(f"Animal '{ani['name']}' has empty DataFrame.")

    uid_to_idx: dict = {}
    records: list    = []
    for ani in animal_data:
        uid = ani["uid"]
        if uid in uid_to_idx:
            raise ValueError(f"Duplicate uid={uid}.")
        rec = {
            "uid":       uid,
            "name":      ani["name"],
            "exp_group": ani["exp_group"],
            "fps":       ani["fps"],
            "df":        ani["df"],
            "metrics":   compute_metrics(ani["df"], groups, ani["fps"]),
            "session_s": session_duration(ani["df"], ani["fps"]),
        }
        uid_to_idx[uid] = len(records)
        records.append(rec)

    exp_groups: dict = {}
    for rec in records:
        exp_groups.setdefault(rec["exp_group"], [])
        exp_groups[rec["exp_group"]].append(rec["uid"])

    grand: dict = {}
    for eg, uids in exp_groups.items():
        grand[eg] = {}
        eg_recs = [records[uid_to_idx[u]] for u in uids]
        for beh_group in groups:
            vals = {k: [] for k in ("total_duration","frequency","latency","mean_bout")}
            for rec in eg_recs:
                m = rec["metrics"].get(beh_group, {})
                vals["total_duration"].append(m.get("total_duration", 0.0))
                vals["frequency"].append(float(m.get("frequency", 0)))
                lat = m.get("latency")
                if lat is not None:
                    vals["latency"].append(lat)
                vals["mean_bout"].append(m.get("mean_bout", 0.0))
            grand[eg][beh_group] = {
                k: {
                    "mean": float(np.mean(v)) if v else 0.0,
                    "sem":  float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                    "n":    len(v),
                    "vals": v,
                }
                for k, v in vals.items()
            }

    return {
        "records":    records,
        "uid_to_idx": uid_to_idx,
        "exp_groups": exp_groups,
        "grand":      grand,
    }


#
# UNBIASED ANALYTICS - STATISTICAL ENGINE
#

def benjamini_hochberg(pvals) -> "np.ndarray":
    """Benjamini-Hochberg FDR-adjusted q-values for an array of p-values.

    Returns q-values in the same order as the input.  Implemented directly (no
    statsmodels dependency): sort ascending, scale each by n/rank, enforce
    monotonicity from the largest p downward, and clip to [0, 1].
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order  = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]   # monotone non-decreasing in p
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def fdr_sig_column(stats_df) -> tuple:
    """Return (column_name, label) for significance calls: the FDR q-value when
    present (fresh stats), otherwise the raw p-value (older cached frames)."""
    cols = getattr(stats_df, "columns", [])
    if "qval_kw" in cols:
        return "qval_kw", "q"
    return "pval_kw", "p"


def _prevalence_test(group_present: dict):
    """Test whether the presence/absence of a behaviour differs across groups.

    group_present : {exp_group: [1/0, ...]}  (1 = animal expressed the cluster).
    Returns (statistic, p_value).  Fisher exact for a 2x2 (two groups), else a
    chi-square test on the groups x {present, absent} contingency table.  Returns
    (nan, 1.0) when the table is degenerate (all-present or all-absent).
    """
    if not SCIPY_OK:
        return float("nan"), 1.0
    egs = [g for g in group_present if len(group_present[g]) > 0]
    if len(egs) < 2:
        return float("nan"), 1.0
    table = []
    for g in egs:
        pres = int(sum(group_present[g]))
        absent = int(len(group_present[g]) - pres)
        table.append([pres, absent])
    table = np.asarray(table, dtype=int)
    # Degenerate: a column is all-zero (every animal present, or every absent).
    if table[:, 0].sum() == 0 or table[:, 1].sum() == 0:
        return float("nan"), 1.0
    try:
        if table.shape == (2, 2):
            stat, p = sp_stats.fisher_exact(table)
            return float(stat), float(p)
        stat, p, _, _ = sp_stats.chi2_contingency(table)
        return float(stat), float(p)
    except Exception:
        return float("nan"), 1.0


def _present_only_kw(group_vals: dict, group_present: dict):
    """Kruskal-Wallis on magnitudes from animals that EXPRESSED the cluster.

    Separates 'how much' from 'how often' so a behaviour that is merely present
    in one group and absent in another is not reported as a magnitude effect.
    Returns (statistic, p_value, n_used).  (nan, 1.0, n) when too few present.
    """
    if not SCIPY_OK:
        return float("nan"), 1.0, 0
    present_groups = []
    n_used = 0
    for g in group_vals:
        vs = [v for v, pr in zip(group_vals[g], group_present[g]) if pr]
        if vs:
            present_groups.append(np.asarray(vs, dtype=float))
            n_used += len(vs)
    if len(present_groups) < 2 or n_used < 3:
        return float("nan"), 1.0, n_used
    # KW needs some variation; identical values across all groups → not testable.
    if len({round(float(v), 12) for grp in present_groups for v in grp}) < 2:
        return float("nan"), 1.0, n_used
    try:
        stat, p = sp_stats.kruskal(*present_groups)
        return float(stat), float(p), n_used
    except Exception:
        return float("nan"), 1.0, n_used


def pairwise_posthoc(group_vals: dict) -> list:
    """Pairwise post-hoc comparisons across experimental groups for one cluster.

    Localises an omnibus Kruskal-Wallis effect to specific group pairs.  Uses
    Dunn's test (scikit-posthocs) when available, else pairwise Mann-Whitney U,
    in both cases with a Benjamini-Hochberg adjustment over the pair family.
    Returns a list of {group_a, group_b, p, q, direction}.  Empty for <2 groups
    or when scipy is unavailable.
    """
    if not SCIPY_OK:
        return []
    import itertools as _it
    names = [g for g in group_vals if len(group_vals[g]) > 0]
    if len(names) < 2:
        return []
    pairs = list(_it.combinations(names, 2))

    # Optional Dunn's test (rank-based, matches the KW omnibus).
    dunn = None
    try:
        import scikit_posthocs as _sph
        data, labels = [], []
        for g in names:
            data.extend(list(group_vals[g]))
            labels.extend([g] * len(group_vals[g]))
        dunn = _sph.posthoc_dunn(
            pd.DataFrame({"v": data, "g": labels}),
            val_col="v", group_col="g", p_adjust=None)
    except Exception:
        dunn = None

    recs, raw_p = [], []
    for a, b in pairs:
        va = np.asarray(group_vals[a], dtype=float)
        vb = np.asarray(group_vals[b], dtype=float)
        if dunn is not None:
            try:
                p = float(dunn.loc[a, b])
            except Exception:
                p = 1.0
        else:
            try:
                _, p = sp_stats.mannwhitneyu(va, vb, alternative="two-sided")
            except Exception:
                p = 1.0
        direction = ">" if np.mean(va) >= np.mean(vb) else "<"
        recs.append({"group_a": a, "group_b": b,
                     "p": float(p), "direction": direction})
        raw_p.append(p)
    q = benjamini_hochberg(np.asarray(raw_p, dtype=float))
    for r, qq in zip(recs, q):
        r["q"] = float(qq)
    return recs


def draw_sig_brackets(ax, group_names: list, all_pts: list, y_start: float,
                      y_step: float, tick_color="#aaaacc",
                      sig_color="#FF4081") -> float:
    """Draw pairwise post-hoc significance brackets onto a bar axis.

    group_names : x-axis group order (index i sits at x=i).
    all_pts     : list of per-group value lists (same order as group_names).
    y_start     : y of the first (lowest) bracket; subsequent stacked above by
                  y_step.  Returns the y above the topmost bracket so the caller
                  can extend ylim.  Only pairs with q < 0.05 are annotated, so a
                  panel with no significant pairs adds nothing.
    """
    gv = {g: list(v) for g, v in zip(group_names, all_pts)}
    recs = pairwise_posthoc(gv)
    sig = [r for r in recs if r.get("q", 1.0) < 0.05]
    if not sig:
        return y_start
    # Order brackets by group separation so short ones stack below long ones.
    idx = {g: i for i, g in enumerate(group_names)}
    sig.sort(key=lambda r: abs(idx[r["group_a"]] - idx[r["group_b"]]))
    y = y_start
    for r in sig:
        xa, xb = idx[r["group_a"]], idx[r["group_b"]]
        x0, x1 = min(xa, xb), max(xa, xb)
        ax.plot([x0, x0, x1, x1], [y, y + y_step * 0.15, y + y_step * 0.15, y],
                lw=0.6, color=tick_color)
        q = r["q"]
        star = ("***" if q < 0.001 else "**" if q < 0.01 else "*")
        ax.text((x0 + x1) / 2, y + y_step * 0.17, star, ha="center",
                fontsize=10, color=sig_color, fontweight="bold")
        y += y_step
    return y


def family_wide_fdr(stats_by_metric: dict) -> pd.DataFrame:
    """Pool p-values across ALL (metric x cluster) tests and apply one BH pass.

    stats_by_metric : {metric_name: stats_df_from_run_cluster_statistics}.
    Per-metric FDR under-corrects when many metrics are reported; this returns a
    tidy frame (metric, cluster_id, pval_kw, qval_family) with the family-wide
    q-value so significance can be judged across the whole reported family.
    """
    rows = []
    for _metric, _df in stats_by_metric.items():
        if _df is None or getattr(_df, "empty", True):
            continue
        for _, r in _df.iterrows():
            rows.append({"metric": _metric,
                         "cluster_id": r.get("cluster_id"),
                         "pval_kw": float(r.get("pval_kw", 1.0))})
    if not rows:
        return pd.DataFrame(columns=["metric", "cluster_id",
                                     "pval_kw", "qval_family"])
    out = pd.DataFrame(rows)
    out["qval_family"] = benjamini_hochberg(out["pval_kw"].to_numpy())
    return out.sort_values("qval_family").reset_index(drop=True)


# Columns produced by the parametric (ANOVA / eta-squared) path.  Hidden from
# user-facing tables unless show_parametric is requested — they are unstable at
# the small n typical of behavioural cohorts and epsilon-squared is the headline.
PARAMETRIC_COLUMNS = ("stat_anova", "pval_anova", "effect_size_eta2")


def drop_parametric_columns(df: "pd.DataFrame", show_parametric: bool = False):
    """Return df without the parametric columns unless show_parametric is True."""
    if show_parametric or df is None or getattr(df, "empty", True):
        return df
    cols = [c for c in PARAMETRIC_COLUMNS if c in df.columns]
    return df.drop(columns=cols) if cols else df


def run_cluster_statistics(animal_data: list, metric: str = "total_duration",
                           with_posthoc: bool = True,
                           _compute_fn=None) -> pd.DataFrame:
    """
    Run Kruskal-Wallis (non-parametric, robust for small n) and ANOVA across
    experimental groups for every cluster present in the data.

    Returns DataFrame sorted by p-value:
      cluster_id, stat_kw, pval_kw, stat_anova, pval_anova,
      effect_size_eta2, direction, group_means...

    _compute_fn: optional callable(ani) -> DataFrame indexed by cluster_id.
      Defaults to compute_per_cluster_metrics. Pass compute_per_pair_transition_probs
      (wrapped in a lambda) to test transition pairs instead of clusters.
    """
    if not SCIPY_OK:
        raise RuntimeError("scipy not available. pip install scipy")

    if _compute_fn is None:
        _compute_fn = lambda ani: compute_per_cluster_metrics(ani["df"], ani["fps"])

    # Build per-cluster, per-group value lists
    eg_map: dict = {}      # exp_group -> list of per_cluster DataFrames
    for ani in animal_data:
        eg  = ani["exp_group"]
        pcm = _compute_fn(ani)
        eg_map.setdefault(eg, []).append(pcm)

    all_clusters = set()
    for frames in eg_map.values():
        for pcm in frames:
            all_clusters.update(pcm.index.tolist())
    all_clusters = sorted(all_clusters)
    eg_names     = list(eg_map.keys())

    rows = []
    for cid in all_clusters:
        group_vals = {}
        group_present = {}   # eg -> list of 1/0 (cluster expressed by that animal)
        for eg in eg_names:
            vs, pres = [], []
            for pcm in eg_map[eg]:
                if cid in pcm.index:
                    vs.append(float(pcm.loc[cid, metric]))
                    pres.append(1)
                else:
                    # Structural zero: the animal never expressed this behavior.
                    # Kept as 0.0 for the KW test, but tracked separately as a
                    # prevalence so "never performed" is not conflated with a
                    # measured low value.
                    vs.append(0.0)
                    pres.append(0)
            group_vals[eg] = vs
            group_present[eg] = pres

        all_vals = [v for vs in group_vals.values() for v in vs]
        groups_for_test = [np.array(vs) for vs in group_vals.values()]

        # Need at least 2 groups and 2 values total
        if len(groups_for_test) < 2 or len(all_vals) < 3:
            continue

        # Kruskal-Wallis
        try:
            stat_kw, pval_kw = sp_stats.kruskal(*groups_for_test)
        except Exception:
            stat_kw, pval_kw = np.nan, 1.0

        # One-way ANOVA
        try:
            stat_an, pval_an = sp_stats.f_oneway(*groups_for_test)
        except Exception:
            stat_an, pval_an = np.nan, 1.0

        # Eta-squared effect size from ANOVA (kept for backward compatibility;
        # parametric, unstable at n=3-4 with skewed durations — not the headline).
        try:
            grand_mean = np.mean(all_vals)
            ss_between = sum(len(vs) * (np.mean(vs) - grand_mean)**2 for vs in groups_for_test)
            ss_total   = sum((v - grand_mean)**2 for v in all_vals)
            eta2       = ss_between / ss_total if ss_total > 0 else 0.0
        except Exception:
            eta2 = 0.0

        # Epsilon-squared: the effect size that matches the Kruskal-Wallis test
        # (Tomczak & Tomczak 2014).  epsilon^2 = H / (n - 1), range [0, 1].
        # Consistent with the non-parametric p-value used as the headline.
        try:
            n_tot   = len(all_vals)
            epsilon2 = (float(stat_kw) / (n_tot - 1)
                        if (np.isfinite(stat_kw) and n_tot > 1) else 0.0)
            epsilon2 = float(min(max(epsilon2, 0.0), 1.0))
        except Exception:
            epsilon2 = 0.0

        # ── Two-part decomposition (structural zeros) ─────────────────────────
        # The zero-padded KW above conflates "absent" with "low".  Separate the
        # two: a prevalence test (presence/absence across groups) and a KW on the
        # magnitudes of only the animals that expressed the cluster.
        stat_prev, pval_prev = _prevalence_test(group_present)
        stat_kwp, pval_kwp, n_present_tot = _present_only_kw(
            group_vals, group_present)

        row = {
            "cluster_id":         cid,
            "stat_kw":            stat_kw,
            "pval_kw":            pval_kw,
            "stat_anova":         stat_an,
            "pval_anova":         pval_an,
            "effect_size_eta2":   eta2,
            "effect_size_epsilon2": epsilon2,
            "neg_log10_p":        -np.log10(max(pval_kw, 1e-300)),
            # Two-part (raw p; q-values added after the loop via BH)
            "stat_prevalence":    stat_prev,
            "pval_prevalence":    pval_prev,
            "stat_kw_present":    stat_kwp,
            "pval_kw_present":    pval_kwp,
            "n_present_total":    int(n_present_tot),
        }
        for eg in eg_names:
            row[f"mean_{eg}"] = float(np.mean(group_vals[eg]))
            # Prevalence: fraction of animals in this group that expressed the
            # cluster at all (exposes differences driven by presence/absence
            # rather than magnitude — e.g. a behavior only one group performs).
            _pres = group_present[eg]
            row[f"prevalence_{eg}"] = (float(np.mean(_pres)) if _pres else 0.0)
            # Per-group sample sizes — small-n KW has coarse achievable p-values;
            # surfacing n keeps significance calls interpretable.
            row[f"n_{eg}"]         = int(len(_pres))
            row[f"n_present_{eg}"] = int(sum(_pres))
        # Pairwise post-hoc localisation (only stored when >2 groups & requested).
        if with_posthoc and len(eg_names) > 2:
            try:
                _ph = pairwise_posthoc(group_vals)
                row["posthoc_pairs"] = "; ".join(
                    f"{r['group_a']}{r['direction']}{r['group_b']} "
                    f"q={r['q']:.3g}" for r in _ph)
                _sig = [r for r in _ph if r["q"] < 0.05]
                row["posthoc_min_q"] = (min(r["q"] for r in _ph)
                                        if _ph else float("nan"))
                row["posthoc_n_sig_pairs"] = len(_sig)
            except Exception:
                row["posthoc_pairs"] = ""
                row["posthoc_min_q"] = float("nan")
                row["posthoc_n_sig_pairs"] = 0
        rows.append(row)

    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        # Benjamini-Hochberg FDR correction across all clusters tested.  With
        # ~30 clusters x several metrics, uncorrected p-values are not
        # publishable — qval_kw is the value that should drive significance
        # calls (volcano thresholds, heatmap stars).
        df_res["qval_kw"] = benjamini_hochberg(df_res["pval_kw"].to_numpy())
        df_res["neg_log10_q"] = -np.log10(
            np.clip(df_res["qval_kw"].to_numpy(), 1e-300, None))
        # FDR for the two-part tests (same cluster family).
        if "pval_prevalence" in df_res.columns:
            df_res["qval_prevalence"] = benjamini_hochberg(
                df_res["pval_prevalence"].to_numpy())
        if "pval_kw_present" in df_res.columns:
            df_res["qval_kw_present"] = benjamini_hochberg(
                df_res["pval_kw_present"].to_numpy())
        # sig_driver: what makes a cluster significant at q<0.05 —
        # the magnitude (present-only KW), the prevalence (presence/absence),
        # both, or neither.  Lets figures annotate WHY a cluster differs and
        # prevents a presence/absence effect being read as a magnitude change.
        _SIG = 0.05

        def _driver(r):
            mag  = float(r.get("qval_kw_present", 1.0)) < _SIG
            prev = float(r.get("qval_prevalence", 1.0)) < _SIG
            if mag and prev:
                return "both"
            if mag:
                return "magnitude"
            if prev:
                return "prevalence"
            return "none"
        df_res["sig_driver"] = df_res.apply(_driver, axis=1)
        df_res = df_res.sort_values("qval_kw").reset_index(drop=True)
    return df_res


# ──────────────────────────────────────────────────────────────────────────────
#  ADVANCED ANALYSES  (Tier 4 — additive; operate on exported bout labels)
#  Each function is standalone and writes its own artifacts to out_dir.  None of
#  them alter or depend on the existing analysis flow.
# ──────────────────────────────────────────────────────────────────────────────

def _frame_label_array(df: pd.DataFrame) -> np.ndarray:
    """Reconstruct the per-frame cluster-id array from a bout table
    (columns: label, start_frame, run_len).  Gaps are -1 (unlabelled)."""
    if df.empty:
        return np.zeros(0, dtype=int)
    end = int((df["start_frame"] + df["run_len"]).max())
    arr = np.full(end, -1, dtype=int)
    for _, r in df.iterrows():
        s = int(r["start_frame"]); e = s + int(r["run_len"])
        arr[s:e] = int(r["label"])
    return arr


def behavioral_fingerprint_classification(animal_data: list,
                                          out_dir: "pathlib.Path",
                                          metric: str = "total_duration") -> dict:
    """Test whether the behavioural repertoire AS A WHOLE separates groups.

    Builds a per-animal usage vector (one feature per cluster) and trains a
    cross-validated classifier to predict the experimental group, with a
    permutation test for significance and per-behaviour importances.  This turns
    'individual clusters differ' into the stronger 'the repertoire discriminates
    groups' claim (cf. Wiltschko et al. 2020).  Writes fingerprint_classification
    .json + .png.  Returns the result dict (or {'error': ...}).
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import (cross_val_score, permutation_test_score,
                                              StratifiedKFold)
    except Exception as e:
        return {"error": f"scikit-learn required: {e}"}
    # Per-animal usage vector over the union of clusters.
    all_clusters = sorted({int(c) for ani in animal_data
                           for c in compute_per_cluster_metrics(
                               ani["df"], ani["fps"]).index})
    if not all_clusters:
        return {"error": "no clusters"}
    X, y = [], []
    for ani in animal_data:
        pcm = compute_per_cluster_metrics(ani["df"], ani["fps"])
        vec = [float(pcm.loc[c, metric]) if c in pcm.index else 0.0
               for c in all_clusters]
        tot = sum(vec)
        X.append([v / tot for v in vec] if tot > 0 else vec)  # normalise to fractions
        y.append(ani["exp_group"])
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    groups = sorted(set(y))
    # Need ≥2 groups and ≥2 animals per group for a meaningful CV.
    counts = {g: int((y == g).sum()) for g in groups}
    if len(groups) < 2 or min(counts.values()) < 2:
        return {"error": f"need ≥2 groups with ≥2 animals each; have {counts}"}
    k = max(2, min(5, min(counts.values())))
    # Linear classifier (standardised) — appropriate for the small-n, high-dim
    # usage-vector regime, and n_jobs=1 avoids native-thread instability.
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    try:
        acc = cross_val_score(clf, X, y, cv=cv, n_jobs=1).mean()
        score, _, pval = permutation_test_score(
            clf, X, y, cv=cv, n_permutations=199, random_state=42, n_jobs=1)
    except Exception as e:
        return {"error": f"classification failed: {e}"}
    clf.fit(X, y)
    # Importance = mean |coefficient| across one-vs-rest classes (per cluster).
    _coef = np.abs(clf.named_steps["logisticregression"].coef_)
    importance = _coef.mean(axis=0) if _coef.ndim == 2 else np.ravel(_coef)
    chance = max(counts.values()) / len(y)
    result = {
        "n_animals": int(len(y)), "groups": counts,
        "cv_accuracy": round(float(acc), 4),
        "chance_accuracy": round(float(chance), 4),
        "permutation_p": round(float(pval), 5),
        "n_features": len(all_clusters),
        "cluster_importance": {int(c): round(float(im), 5)
                               for c, im in zip(all_clusters, importance)},
        "metric": metric,
    }
    try:
        out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "fingerprint_classification.json").write_text(
            json.dumps(result, indent=2))
        order = np.argsort(importance)[::-1][:15]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([f"C{all_clusters[i]}" for i in order],
               [importance[i] for i in order], color="#4E79A7")
        ax.set_ylabel("RandomForest importance")
        ax.set_title(f"Behavioural fingerprint — CV acc {acc:.2f} "
                     f"(chance {chance:.2f}, perm p={pval:.3f})")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(str(out_dir / "fingerprint_classification.png"), dpi=150)
        plt.close(fig)
    except Exception:
        pass
    return result


def sequence_statistics(animal_data: list,
                        out_dir: "pathlib.Path",
                        n_shuffles: int = 200) -> "pd.DataFrame":
    """Per-animal behavioural-sequence complexity + bigram enrichment.

    For each animal computes the normalised transition entropy (a single
    'behavioural complexity' index in [0,1]) from the bout-label sequence, and
    tests bigram (A→B) over-representation against order-shuffled nulls.  Writes
    sequence_stats.csv + sequence_bigram_enrichment.csv.  Returns the per-animal
    summary frame.
    """
    rng = np.random.default_rng(42)
    rows = []
    bigram_rows = []
    for ani in animal_data:
        seq = ani["df"].sort_values("start_frame")["label"].astype(int).values
        # Collapse consecutive duplicates → bout sequence.
        seq = seq[np.insert(np.diff(seq) != 0, 0, True)]
        if len(seq) < 3:
            rows.append({"animal": ani.get("name", ani.get("uid")),
                         "exp_group": ani["exp_group"], "n_bouts": int(len(seq)),
                         "transition_entropy": np.nan, "complexity_index": np.nan})
            continue
        states = sorted(set(seq))
        idx = {s: i for i, s in enumerate(states)}
        n = len(states)
        T = np.zeros((n, n))
        for a, b in zip(seq[:-1], seq[1:]):
            T[idx[a], idx[b]] += 1
        # Normalised transition entropy: mean row entropy / log(n).
        ent = []
        for i in range(n):
            r = T[i]; tot = r.sum()
            if tot > 0:
                p = r[r > 0] / tot
                ent.append(-(p * np.log(p)).sum())
        mean_ent = float(np.mean(ent)) if ent else 0.0
        complexity = mean_ent / np.log(n) if n > 1 else 0.0
        rows.append({"animal": ani.get("name", ani.get("uid")),
                     "exp_group": ani["exp_group"], "n_bouts": int(len(seq)),
                     "transition_entropy": round(mean_ent, 4),
                     "complexity_index": round(float(complexity), 4)})
        # Bigram enrichment vs shuffled order.
        obs = {}
        for a, b in zip(seq[:-1], seq[1:]):
            obs[(a, b)] = obs.get((a, b), 0) + 1
        null = {k: [] for k in obs}
        for _ in range(n_shuffles):
            sh = rng.permutation(seq)
            cnt = {}
            for a, b in zip(sh[:-1], sh[1:]):
                if (a, b) in obs:
                    cnt[(a, b)] = cnt.get((a, b), 0) + 1
            for k in obs:
                null[k].append(cnt.get(k, 0))
        for (a, b), o in obs.items():
            nd = np.asarray(null[(a, b)], dtype=float)
            mu, sd = float(nd.mean()), float(nd.std())
            z = (o - mu) / sd if sd > 0 else 0.0
            bigram_rows.append({"animal": ani.get("name", ani.get("uid")),
                                "exp_group": ani["exp_group"],
                                "from": int(a), "to": int(b), "observed": int(o),
                                "null_mean": round(mu, 2), "z_score": round(z, 3)})
    df = pd.DataFrame(rows)
    try:
        out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "sequence_stats.csv", index=False)
        pd.DataFrame(bigram_rows).to_csv(
            out_dir / "sequence_bigram_enrichment.csv", index=False)
    except Exception:
        pass
    return df


def time_resolved_usage(animal_data: list,
                        out_dir: "pathlib.Path",
                        window_s: float = 300.0) -> "pd.DataFrame":
    """Per-animal cluster-usage trajectories over time (binned by window_s).

    Session-total metrics hide onset/habituation/time-course effects (drug
    kinetics, pain time-course).  This bins each session into window_s segments
    and reports the time-fraction in each cluster per window.  Writes
    usage_over_time.csv + usage_over_time.png.  Returns the tidy frame.
    """
    rows = []
    for ani in animal_data:
        fps = float(ani["fps"]); arr = _frame_label_array(ani["df"])
        if arr.size == 0:
            continue
        win = max(1, int(round(window_s * fps)))
        n_win = int(np.ceil(arr.size / win))
        for w in range(n_win):
            seg = arr[w * win:(w + 1) * win]
            seg = seg[seg >= 0]
            if seg.size == 0:
                continue
            vals, cnts = np.unique(seg, return_counts=True)
            for v, c in zip(vals, cnts):
                rows.append({"animal": ani.get("name", ani.get("uid")),
                             "exp_group": ani["exp_group"],
                             "window_idx": w,
                             "window_start_s": round(w * window_s, 1),
                             "cluster_id": int(v),
                             "time_fraction": round(float(c) / seg.size, 4)})
    df = pd.DataFrame(rows)
    try:
        out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "usage_over_time.csv", index=False)
        if not df.empty:
            piv = (df.groupby(["window_start_s", "cluster_id"])["time_fraction"]
                     .mean().unstack(fill_value=0.0))
            fig, ax = plt.subplots(figsize=(10, 5))
            for cid in piv.columns:
                ax.plot(piv.index, piv[cid], marker="o", label=f"C{cid}")
            ax.set_xlabel(f"Time (s, {int(window_s)}s windows)")
            ax.set_ylabel("Mean time fraction")
            ax.set_title("Cluster usage over time (mean across animals)")
            ax.legend(fontsize=7, ncol=4)
            fig.tight_layout()
            fig.savefig(str(out_dir / "usage_over_time.png"), dpi=150)
            plt.close(fig)
    except Exception:
        pass
    return df


def external_label_validation(animal_data: list,
                              annotations: "pd.DataFrame",
                              out_dir: "pathlib.Path") -> dict:
    """Validate unsupervised clusters against human annotations.

    annotations : DataFrame with columns [animal, start_frame, end_frame, label]
        (label = human behaviour name).  Matched to each animal's per-frame
        cluster array; computes homogeneity, completeness, V-measure and Cohen's
        kappa (against the majority cluster→behaviour mapping).  Writes
        external_validation.json.  This is the only EXTERNAL anchor of cluster
        validity — everything else in the pipeline is internal geometry.
    """
    try:
        from sklearn.metrics import (homogeneity_completeness_v_measure,
                                      cohen_kappa_score)
    except Exception as e:
        return {"error": f"scikit-learn required: {e}"}
    name_by = {ani.get("name", ani.get("uid")): ani for ani in animal_data}
    clu, hum = [], []
    for aname, sub in annotations.groupby("animal"):
        ani = name_by.get(aname)
        if ani is None:
            continue
        arr = _frame_label_array(ani["df"])
        for _, r in sub.iterrows():
            s = int(r["start_frame"]); e = int(r["end_frame"])
            s = max(0, s); e = min(arr.size, e)
            for f in range(s, e):
                if arr[f] >= 0:
                    clu.append(int(arr[f])); hum.append(str(r["label"]))
    if len(clu) < 2 or len(set(hum)) < 2:
        return {"error": "insufficient overlapping annotated frames"}
    clu = np.asarray(clu); hum = np.asarray(hum)
    homo, comp, vmeas = homogeneity_completeness_v_measure(hum, clu)
    # Majority cluster→behaviour mapping for a confusion-style kappa.
    mapping = {}
    for c in set(clu):
        labs, cnts = np.unique(hum[clu == c], return_counts=True)
        mapping[int(c)] = labs[int(np.argmax(cnts))]
    pred = np.array([mapping[int(c)] for c in clu])
    kappa = float(cohen_kappa_score(hum, pred))
    result = {"n_frames": int(len(clu)),
              "homogeneity": round(float(homo), 4),
              "completeness": round(float(comp), 4),
              "v_measure": round(float(vmeas), 4),
              "cohen_kappa": round(kappa, 4),
              "cluster_to_behaviour": mapping}
    try:
        out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "external_validation.json").write_text(json.dumps(result, indent=2))
    except Exception:
        pass
    return result


def run_advanced_analyses(animal_data: list, out_dir: "pathlib.Path",
                          annotations: "pd.DataFrame" = None,
                          metric: str = "total_duration") -> dict:
    """Run all additive Tier-4 analyses, writing artifacts to out_dir.

    External validation only runs when an annotations frame is supplied.
    Returns a dict of {analysis_name: result-or-error} — each analysis is
    independent, so one failing does not stop the others.
    """
    out_dir = pathlib.Path(out_dir)
    results = {}
    for name, fn in (
            ("fingerprint", lambda: behavioral_fingerprint_classification(
                animal_data, out_dir, metric)),
            ("sequence",    lambda: sequence_statistics(animal_data, out_dir)),
            ("time_resolved", lambda: time_resolved_usage(animal_data, out_dir))):
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = {"error": str(e)}
    if annotations is not None:
        try:
            results["external_validation"] = external_label_validation(
                animal_data, annotations, out_dir)
        except Exception as e:
            results["external_validation"] = {"error": str(e)}
    return results


def compute_transition_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Compute normalised row-stochastic transition matrix between B-SOiD cluster IDs.
    Transitions are counted between *consecutive* bouts (ignoring run-length).
    Returns: (matrix, cluster_ids_list)
    """
    labels_seq = df.sort_values("start_frame")["label"].values
    cids = sorted(set(labels_seq))
    idx  = {c: i for i, c in enumerate(cids)}
    n    = len(cids)
    mat  = np.zeros((n, n), dtype=float)
    for a, b in zip(labels_seq[:-1], labels_seq[1:]):
        if a != b:          # only genuine transitions
            mat[idx[a], idx[b]] += 1.0
    # row-normalise
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return mat / row_sums, cids


def build_transition_similarity(animal_data: list) -> dict:
    """
    Average per-animal transition matrices across all animals and
    return a clusterxcluster co-occurrence / transition similarity matrix.
    Two clusters are *similar* when they frequently transition to/from the
    same set of targets (i.e. they live in the same temporal neighbourhood).

    Returns: {
      "tmat_avg":    (n_clusters x n_clusters) averaged transition probabilities,
      "sim_matrix":  (n_clusters x n_clusters) cosine-similarity of outgoing
                     transition profiles,
      "cluster_ids": list of cluster IDs
    }
    """
    all_cids = sorted({
        int(lab) for ani in animal_data
        for lab in ani["df"]["label"].unique()
    })
    n = len(all_cids)
    idx = {c: i for i, c in enumerate(all_cids)}
    tmat_sum = np.zeros((n, n), dtype=float)
    count = 0
    for ani in animal_data:
        mat, ani_cids = compute_transition_matrix(ani["df"])
        for ri, ci in enumerate(ani_cids):
            for cj_i, cj in enumerate(ani_cids):
                if ci in idx and cj in idx:
                    tmat_sum[idx[ci], idx[cj]] += mat[ri, cj_i]
        count += 1
    tmat_avg = tmat_sum / max(count, 1)

    # cosine similarity of outgoing profiles
    norms = np.linalg.norm(tmat_avg, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tmat_norm = tmat_avg / norms
    sim_matrix = tmat_norm @ tmat_norm.T
    np.fill_diagonal(sim_matrix, 1.0)

    return {
        "tmat_avg":    tmat_avg,
        "sim_matrix":  sim_matrix,
        "cluster_ids": all_cids,
    }


def _compute_per_animal_transition_vectors(animal_data: list) -> tuple:
    """
    Returns (X, pair_names) where
      X          — (n_animals, n_pairs) float, union cluster set,
                   diagonal pairs omitted, missing entries = 0.
      pair_names — ["Ci->Cj", ...] matching column order of X.
    Pure function; no GUI coupling.
    """
    all_cids = sorted({
        int(lab) for a in animal_data
        for lab in a["df"]["label"].unique()
    })
    n = len(all_cids)
    cl_idx = {c: i for i, c in enumerate(all_cids)}
    pair_names = [f"C{i}→C{j}" for i in all_cids for j in all_cids if i != j]

    rows = []
    for ani in animal_data:
        mat, ani_cids = compute_transition_matrix(ani["df"])
        full_mat = np.zeros((n, n), dtype=float)
        for ri, ci in enumerate(ani_cids):
            for cj_i, cj in enumerate(ani_cids):
                if ci in cl_idx and cj in cl_idx:
                    full_mat[cl_idx[ci], cl_idx[cj]] += mat[ri, cj_i]
        flat = [full_mat[cl_idx[i], cl_idx[j]]
                for i in all_cids for j in all_cids if i != j]
        rows.append(flat)

    X = np.array(rows, dtype=float) if rows else np.zeros((0, len(pair_names)))
    return X, pair_names


def build_raw_recluster_result(animal_data: list) -> dict:
    """
    Build a minimal recluster-compatible result dict from raw (unreclustered)
    animal data so transition/stats plots can be displayed before reclustering runs.
    The linkage is derived from transition-profile similarity alone.
    """
    if not SCIPY_CLUSTER_OK:
        raise RuntimeError("scipy required for preview plotting")

    trans_result = build_transition_similarity(animal_data)
    all_cids     = trans_result["cluster_ids"]
    n            = len(all_cids)

    sim_mat    = trans_result["sim_matrix"]
    trans_dist = 1.0 - sim_mat
    np.fill_diagonal(trans_dist, 0)
    np.clip(trans_dist, 0, None, out=trans_dist)

    if n >= 2:
        cond_dist = squareform(trans_dist, checks=False)
        cond_dist = np.clip(cond_dist, 0, None)
        lnk = linkage(cond_dist, method="ward")
    else:
        lnk = np.zeros((0, 4))

    return {
        "cluster_ids":          all_cids,
        "all_raw_cluster_ids":  all_cids,
        "filtered_cluster_ids": set(all_cids),  # all kept in raw view
        "linkage_matrix":       lnk,
        "dist_matrix":          trans_dist,
        "corr_dist_matrix":     trans_dist,     # approximation for raw view
        "trans_dist_matrix":    trans_dist,
        "transition_result":    trans_result,
        "silhouette_scores":    {},
        "inertia_scores":       {},
        "eg_names":             [],
        "feature_matrix":       np.zeros((n, 1)),
        "_is_raw_preview":      True,
    }


def _draw_circle_network(ax, net_tmat: np.ndarray, nx_pos, ny_pos,
                          net_labels: list, node_colors: list, t: dict,
                          threshold_pct: int = 90):
    """
    Draw a directed circular transition network on *ax*.
    Only above-chance edges are ever drawn (chance = 1/(nn-1) for a uniform
    random walk among nn nodes).  Among those, the top (100-threshold_pct) %
    by probability are shown; edges are sorted weakest-first so the strongest
    render on top.  Arrow thickness and opacity are scaled relative to the
    chance floor so visual weight represents strength above chance, not just
    absolute probability.
    """
    nn   = len(net_labels)
    flat = net_tmat[net_tmat > 0]

    def _draw_nodes():
        out_strength = net_tmat.sum(axis=1)
        max_s  = out_strength.max() if out_strength.max() > 0 else 1.0
        node_sz = 100 + 250 * (out_strength / max_s)
        ax.scatter(nx_pos, ny_pos, s=node_sz, c=node_colors,
                   zorder=8, edgecolors=t["border"], linewidths=0.8, alpha=0.92)
        for i, lbl in enumerate(net_labels):
            off = 1.18
            ax.text(nx_pos[i] * off, ny_pos[i] * off, lbl,
                    ha="center", va="center",
                    fontsize=max(4, 8 - nn // 15),
                    color=t["tick"], fontweight="bold")
        ax.set_xlim(-1.55, 1.55)
        ax.set_ylim(-1.55, 1.55)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor(t["ax_bg"])

    if len(flat) == 0:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center",
                color=t["tick"], transform=ax.transAxes, fontsize=9)
        _draw_nodes()
        return

    # Strict chance floor: uniform random walk probability among nn nodes
    chance_floor = 1.0 / max(1, nn - 1)

    # Filter to above-chance transitions only before any percentile computation
    above_chance = flat[flat > chance_floor]
    if len(above_chance) == 0:
        ax.text(0.5, 0.98,
                f"No above-chance transitions  (chance = {chance_floor:.3f})",
                ha="center", va="top", color=t["muted"],
                transform=ax.transAxes, fontsize=8)
        _draw_nodes()
        return

    flat_max    = float(above_chance.max())
    # Percentile applied within the above-chance distribution only
    edge_thresh = (float(np.percentile(above_chance, threshold_pct))
                   if threshold_pct > 0 else chance_floor)
    # Always keep threshold strictly above chance
    edge_thresh = max(edge_thresh, chance_floor)
    # Guard: if threshold equals or exceeds max, show all above-chance edges
    if edge_thresh >= flat_max:
        edge_thresh = chance_floor

    # Collect above-threshold edges; sort weakest first so strongest draws on top
    edge_list = []
    for i in range(nn):
        for j in range(nn):
            if i == j:
                continue
            p = float(net_tmat[i, j])
            if p > edge_thresh:
                edge_list.append((p, i, j))
    edge_list.sort()

    # Normalise over the above-chance range: 0 = just above chance, 1 = strongest
    p_range = max(flat_max - chance_floor, 1e-9)
    for p, i, j in edge_list:
        p_norm = (p - chance_floor) / p_range
        lw     = 1.5 + p_norm ** 0.5 * 9.5     # 1.5 – 11.0 px
        alpha  = 0.60 + p_norm * 0.40           # 0.60 – 1.00
        mscale = 12 + p_norm * 14               # arrowhead 12 – 26
        rad    = 0.22 if abs(i - j) > nn // 3 else 0.13
        ax.annotate(
            "", xy=(nx_pos[j], ny_pos[j]),
            xytext=(nx_pos[i], ny_pos[i]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=node_colors[i],
                alpha=alpha, lw=lw,
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=mscale,
            ),
            zorder=2 + int(p_norm * 5),
        )

    _draw_nodes()


def compute_bio_exclusions(animal_data: list, dur_filter: dict) -> set:
    """Return set of cluster IDs that fail the biological relevance filter.

    Averages each metric across only the animals that express the cluster,
    matching the logic used inside compute_reclustering_suggestions.
    """
    if not dur_filter:
        return set()
    min_mean_ms = dur_filter.get("min_mean_duration_ms", None)
    min_total_s = dur_filter.get("min_total_duration_s", None)
    min_freq    = dur_filter.get("min_frequency",        None)
    if min_mean_ms is None and min_total_s is None and min_freq is None:
        return set()

    cluster_metrics: dict = {}  # cid -> list of per-animal metric Series
    for ani in animal_data:
        pcm = compute_per_cluster_metrics(ani["df"], ani["fps"])
        for cid in pcm.index:
            cluster_metrics.setdefault(cid, []).append(pcm.loc[cid])

    excluded: set = set()
    for cid, rows in cluster_metrics.items():
        if min_mean_ms is not None:
            if float(np.mean([r["mean_bout"] for r in rows])) * 1000 < min_mean_ms:
                excluded.add(cid)
                continue
        if min_total_s is not None:
            if float(np.mean([r["total_duration"] for r in rows])) < min_total_s:
                excluded.add(cid)
                continue
        if min_freq is not None:
            if float(np.mean([r["frequency"] for r in rows])) < min_freq:
                excluded.add(cid)
                continue
    return excluded


def compute_reclustering_suggestions(animal_data: list, max_k: int = 10,
                                      metric: str = "total_duration",
                                      duration_filter: dict = None) -> dict:
    """
    Agglomerative reclustering of B-SOiD clusters based on their *pattern of
    change* across experimental groups AND their *temporal co-occurrence* as
    captured by transition dynamics.

    KEY DESIGN CHOICES
    ------------------
    1. Feature vector = mean metric value per exp-group (clusters x groups).
    2. Each cluster profile is L2-row-normalised (direction, not magnitude).
    3. Distance metric = blend of:
         a) Correlation distance (1 - Pearson r) on change-pattern profiles
            (same as v5, gold-standard from gene-expression literature).
         b) 1 - cosine_similarity of outgoing transition profiles.
       Combined: dist =   * corr_dist + (1- ) * trans_dist
         = transition_weight parameter (default 0.3, so transitions add 30%).
       Behavioural clusters that frequently co-occur in time are pulled closer.
    4. Ward linkage on the blended distance matrix.
    5. Silhouette evaluated on the blended distance matrix.
    6. Optional duration_filter removes biologically irrelevant clusters before
       clustering.  Filter dict keys (all optional):
         min_mean_duration_ms    drop cluster if mean bout < N ms (per animal avg)
         min_total_duration_s    drop cluster if total duration < N s  (per animal avg)
         min_frequency           drop cluster if mean frequency < N bouts

    Returns dict with keys:
      feature_matrix, X_normed, dist_matrix, linkage_matrix,
      silhouette_scores, inertia_scores, eg_names, cluster_ids,
      transition_result,      new: transition similarity data
      filtered_cluster_ids    new: ids that survived the duration filter
    """
    if not (SCIPY_OK and SK_OK and SCIPY_CLUSTER_OK):
        raise RuntimeError("scipy + scikit-learn required for reclustering. "
                           "pip install scipy scikit-learn")
    from scipy.spatial.distance import cdist, squareform

    #   Duration filter  
    duration_filter = duration_filter or {}
    min_mean_ms   = duration_filter.get("min_mean_duration_ms",   None)
    min_total_s   = duration_filter.get("min_total_duration_s",   None)
    min_freq      = duration_filter.get("min_frequency",          None)

    eg_map: dict = {}
    for ani in animal_data:
        eg = ani["exp_group"]
        pcm = compute_per_cluster_metrics(ani["df"], ani["fps"])
        eg_map.setdefault(eg, []).append((pcm, ani["fps"]))

    # build per-cluster averages for filtering
    all_raw_clusters = sorted(set(
        cid
        for frames in eg_map.values()
        for pcm, _ in frames
        for cid in pcm.index
    ))
    eg_names_all = list(eg_map.keys())

    # Compute per-cluster mean metrics across all animals for filtering
    def _avg_metric(cid, mkey):
        vals = []
        for frames in eg_map.values():
            for pcm, fps in frames:
                if cid in pcm.index:
                    vals.append(float(pcm.loc[cid, mkey]))
        return float(np.mean(vals)) if vals else 0.0

    filtered_cluster_ids = []
    for cid in all_raw_clusters:
        if min_mean_ms is not None:
            if _avg_metric(cid, "mean_bout") * 1000 < min_mean_ms:
                continue
        if min_total_s is not None:
            if _avg_metric(cid, "total_duration") < min_total_s:
                continue
        if min_freq is not None:
            if _avg_metric(cid, "frequency") < min_freq:
                continue
        filtered_cluster_ids.append(cid)

    all_clusters = filtered_cluster_ids
    if len(all_clusters) < 2:
        raise ValueError(
            f"Only {len(all_clusters)} cluster(s) passed the duration filter.  "
            "Relax the minimum duration / frequency thresholds.")

    eg_names = eg_names_all

    #   Build raw feature matrix: mean per (cluster x exp_group)  
    feat_rows = {}
    for cid in all_clusters:
        row = {}
        for eg in eg_names:
            vals = [float(pcm.loc[cid, metric]) if cid in pcm.index else 0.0
                    for pcm, _ in eg_map[eg]]
            row[eg] = float(np.mean(vals))
        feat_rows[cid] = row

    feat_df = pd.DataFrame(feat_rows).T
    feat_df.index.name = "cluster_id"

    #   L2-normalise rows  
    norms = np.linalg.norm(feat_df.values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_normed = feat_df.values / norms

    #   Correlation-distance on change patterns  
    if X_normed.shape[0] >= 2 and X_normed.shape[1] >= 2:
        try:
            corr_dist = cdist(X_normed, X_normed, metric="correlation")
            corr_dist = np.nan_to_num(corr_dist, nan=2.0)
            np.fill_diagonal(corr_dist, 0.0)
        except Exception:
            corr_dist = cdist(X_normed, X_normed, metric="euclidean")
    else:
        corr_dist = cdist(X_normed, X_normed, metric="euclidean")

    #   Transition dynamics  
    # Pass the original animal_data (unfiltered fps, but we only use cids in all_clusters)
    trans_result = build_transition_similarity(animal_data)
    trans_cids   = trans_result["cluster_ids"]
    trans_idx    = {c: i for i, c in enumerate(trans_cids)}
    n_c          = len(all_clusters)
    trans_dist   = np.ones((n_c, n_c), dtype=float)   # default = max distance
    for ai, ca in enumerate(all_clusters):
        for bi, cb in enumerate(all_clusters):
            if ca in trans_idx and cb in trans_idx:
                sim = trans_result["sim_matrix"][trans_idx[ca], trans_idx[cb]]
                trans_dist[ai, bi] = 1.0 - float(sim)
    np.fill_diagonal(trans_dist, 0.0)
    # Normalise trans_dist to [0, 2] to match correlation distance range
    td_max = trans_dist.max()
    if td_max > 0:
        trans_dist = trans_dist / td_max * 2.0

    #   Blended distance matrix (70% change pattern + 30% transition)  
    alpha    = 0.70   # weight for corr_dist; 1-alpha for trans_dist
    dist_mat = alpha * corr_dist + (1.0 - alpha) * trans_dist
    np.fill_diagonal(dist_mat, 0.0)
    dist_condensed = squareform(dist_mat, checks=False)

    try:
        lnk = linkage(dist_condensed, method="average")
    except Exception:
        lnk = linkage(X_normed, method="ward")

    #   Silhouette and inertia  
    sil_scores     = {}
    inertia_scores = {}
    max_k_actual   = min(max_k, len(all_clusters) - 1)

    for k in range(2, max_k_actual + 1):
        labels = fcluster(lnk, t=k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        try:
            sil = silhouette_score(dist_mat, labels, metric="precomputed")
        except Exception:
            sil = np.nan
        sil_scores[k] = sil
        inertia = 0.0
        for ci in set(labels):
            pts = X_normed[labels == ci]
            inertia += float(np.sum((pts - pts.mean(axis=0))**2))
        inertia_scores[k] = inertia

    return {
        "feature_matrix":       feat_df,
        "X_normed":             X_normed,
        "dist_matrix":          dist_mat,
        "corr_dist_matrix":     corr_dist,
        "trans_dist_matrix":    trans_dist,
        "transition_result":    trans_result,
        "linkage_matrix":       lnk,
        "silhouette_scores":    sil_scores,
        "inertia_scores":       inertia_scores,
        "eg_names":             eg_names,
        "cluster_ids":          all_clusters,
        "filtered_cluster_ids": filtered_cluster_ids,
        "all_raw_cluster_ids":  all_raw_clusters,
    }


#  
# UNBIASED ANALYTICS - PLOTTING
#  

def build_volcano_figure(stats_df: pd.DataFrame, top_n: int, p_thresh: float,
                          t: dict = None,
                          groups: dict = None,
                          metric: str = None) -> plt.Figure:
    """
    Volcano plot: effect size (epsilon², matched to Kruskal-Wallis) vs
    -log10(FDR q-value).  Significance is judged on the Benjamini-Hochberg
    corrected q-value, not the raw p-value, because ~30 clusters are tested.
    Pass groups to colour-code each cluster by user-defined behaviour group.
    """
    if t is None:
        t = T()

    # Use FDR-corrected q-values and the non-parametric effect size when present
    # (fresh stats_df); fall back to raw p / eta² for older cached frames.
    _has_q   = "qval_kw" in stats_df.columns
    _has_eps = "effect_size_epsilon2" in stats_df.columns
    _sig_col = "qval_kw" if _has_q else "pval_kw"
    _y_col   = "neg_log10_q" if "neg_log10_q" in stats_df.columns else "neg_log10_p"
    _x_col   = "effect_size_epsilon2" if _has_eps else "effect_size_eta2"
    _sig_word = "q" if _has_q else "p"

    # Map cluster ID → (group_name, colour) from user-defined groups
    cid_to_group: dict = {}
    if groups:
        for gname, ginfo in groups.items():
            for cid in ginfo.get("labels", []):
                cid_to_group[cid] = (gname, ginfo["color"])

    with plt.style.context(t["mpl_style"]):
        fig, ax = plt.subplots(figsize=(9, 6), facecolor=t["fig_bg"])
        ax.set_facecolor(t["ax_bg"])

        neg_log_thresh = -np.log10(p_thresh)
        top_ids = set(stats_df.head(top_n)["cluster_id"].tolist())

        _ann: list = []   # (x, y, label_text) — filled below, placed after loop

        for _, row in stats_df.iterrows():
            _raw = row["cluster_id"]
            try:
                cid = int(_raw)
            except (ValueError, TypeError):
                cid = _raw          # string pair ID e.g. "C2→C5"
            x    = row[_x_col]
            y    = row[_y_col]
            sig  = (row[_sig_col] < p_thresh)
            top  = cid in top_ids
            in_grp = cid in cid_to_group

            if top:
                color  = "#FF4081"
                marker = "*"
                size   = 140
                zorder = 5
                edge   = "none"
            elif in_grp and sig:
                color  = cid_to_group[cid][1]
                marker = "D"
                size   = 70
                zorder = 4
                edge   = t["border"]
            elif in_grp:
                color  = cid_to_group[cid][1]
                marker = "D"
                size   = 45
                zorder = 3
                edge   = "none"
            elif sig:
                color  = "#FFC107"
                marker = "o"
                size   = 60
                zorder = 4
                edge   = "none"
            else:
                color  = t["muted"]
                marker = "o"
                size   = 30
                zorder = 2
                edge   = "none"

            ax.scatter(x, y, c=color, s=size, marker=marker,
                       alpha=0.85, edgecolors=edge, linewidths=0.6, zorder=zorder)
            if top:   # only label the top-N units; group colours/markers carry group info
                _lbl = f"C{cid}" if isinstance(cid, (int, np.integer)) else str(cid)
                _ann.append((float(x), float(y), _lbl))

        # ── Overlap-aware annotation pass ──────────────────────────────────
        if _ann:
            _xr = max(float(stats_df[_x_col].max() - stats_df[_x_col].min()), 1e-6)
            _yr = max(float(stats_df[_y_col].max() - stats_df[_y_col].min()), 1e-6)
            # Scale data coords to approximate display points (9×6 fig, ~80% axes)
            _xs, _ys = 518.0 / _xr, 346.0 / _yr
            # Larger offsets + leader lines keep labels clear of dense clusters
            _CANDS = [(32, 18), (-32, 18), (32, -18), (-32, -18),
                      (48, 4), (-48, 4), (4, 38), (4, -38),
                      (42, 30), (-42, 30), (42, -30), (-42, -30)]
            _placed: list = []   # (disp_x, disp_y) of placed label anchors
            for (px, py, lbl) in sorted(_ann, key=lambda v: -v[1]):
                dx, dy = px * _xs, py * _ys
                chosen = _CANDS[0]
                best_d = -1.0
                for (ox, oy) in _CANDS:
                    min_d = min(
                        ((dx + ox - lx) ** 2 + (dy + oy - ly) ** 2) ** 0.5
                        for lx, ly in _placed
                    ) if _placed else float("inf")
                    if min_d > best_d:
                        best_d, chosen = min_d, (ox, oy)
                _placed.append((dx + chosen[0], dy + chosen[1]))
                ax.annotate(lbl, (px, py), fontsize=7, color=t["tick"],
                            textcoords="offset points", xytext=chosen,
                            arrowprops=dict(arrowstyle="-", color=t["tick"],
                                            lw=0.6, alpha=0.5))

        ax.axhline(neg_log_thresh, color="#888888", lw=1, ls="--", alpha=0.7,
                   label=f"{_sig_word} = {p_thresh}")
        _x_label = ("Effect Size (ε², Kruskal-Wallis)" if _has_eps
                    else "Effect Size (η²)")
        _y_label = ("−log₁₀(FDR q-value)" if _has_q else "−log₁₀(p-value)")
        ax.set_xlabel(_x_label, color=t["tick"])
        ax.set_ylabel(_y_label, color=t["tick"])
        _is_trans = (metric == "transition_prob")
        _unit_word = "Transition Pair" if _is_trans else "Cluster"
        ax.set_title(
            f"Volcano Plot — {_unit_word} Significance (FDR-corrected)"
            if _has_q else f"Volcano Plot — {_unit_word} Significance",
            color=t["tick"], fontweight="bold")

        legend_handles = [
            mpatches.Patch(color="#FF4081",
                           label=f"Top {top_n} {'transition pairs' if _is_trans else 'clusters'}"),
            mpatches.Patch(color="#FFC107",
                           label=f"Significant ({_sig_word} < {p_thresh})"),
        ]
        ax.legend(handles=legend_handles, fontsize=8, framealpha=0.8,
                  facecolor=t["ax_bg"], labelcolor=t["tick"],
                  bbox_to_anchor=(1.02, 1.0), loc="upper left",
                  borderaxespad=0)
        _style_ax(ax, t)
        fig.tight_layout(rect=[0, 0, 0.80, 1])
    return fig


def build_heatmap_figure(stats_df: pd.DataFrame, animal_data: list,
                          top_n: int, p_thresh: float,
                          metric: str = "total_duration",
                          eg_colors_override: dict = None,
                          groups: dict = None,
                          t: dict = None) -> plt.Figure:
    """
    Ordered hierarchical heatmap with dendrogram.
    Rows = top-N significant clusters, cols = animals (grouped by exp_group).
    Colour strip above columns uses user-selected EG colours.
    When groups is supplied a right-side strip annotates each row with its
    user-defined behaviour group.
    """
    if t is None:
        t = T()
    if not SCIPY_CLUSTER_OK:
        raise RuntimeError("scipy.cluster required.")

    # Filter to significant top-N (FDR q-value when available)
    _sig_col, _ = fdr_sig_column(stats_df)
    sig_df  = stats_df[stats_df[_sig_col] < p_thresh].head(top_n)
    top_ids = sig_df["cluster_id"].tolist()
    if not top_ids:
        top_ids = stats_df.head(min(top_n, len(stats_df)))["cluster_id"].tolist()

    # Build matrix: rows=clusters, cols=animals
    animals_sorted = sorted(animal_data, key=lambda a: a["exp_group"])
    n_animals = len(animals_sorted)
    # Precompute per-animal metric DataFrames once (not once per cluster).
    if metric == "transition_prob":
        _pcm_cache = [compute_per_pair_transition_probs(a["df"]) for a in animals_sorted]
    else:
        _pcm_cache = [compute_per_cluster_metrics(a["df"], a["fps"]) for a in animals_sorted]
    mat = np.zeros((len(top_ids), n_animals))
    for ci, cid in enumerate(top_ids):
        for ai in range(n_animals):
            pcm = _pcm_cache[ai]
            if cid in pcm.index:
                mat[ci, ai] = float(pcm.loc[cid, metric])

    # Row-normalise (z-score across animals)
    mat_z = np.zeros_like(mat)
    for i in range(len(top_ids)):
        row = mat[i]
        std = row.std()
        mat_z[i] = (row - row.mean()) / std if std > 0 else row - row.mean()

    # Hierarchical clustering of rows
    if len(top_ids) >= 2:
        lnk  = linkage(mat_z, method="ward")
        dend = dendrogram(lnk, no_plot=True)
        row_order = dend["leaves"]
    else:
        row_order = list(range(len(top_ids)))

    mat_ordered = mat_z[row_order, :]
    ids_ordered = [top_ids[i] for i in row_order]

    eg_names = list(dict.fromkeys(a["exp_group"] for a in animals_sorted))
    eg_colors_map: dict = {}
    for i, eg in enumerate(eg_names):
        if eg_colors_override and eg in eg_colors_override:
            eg_colors_map[eg] = eg_colors_override[eg]
        else:
            eg_colors_map[eg] = PALETTE[i % len(PALETTE)]
    col_colors = [eg_colors_map[a["exp_group"]] for a in animals_sorted]

    # Figure: wider columns for readability (1.2 in per animal, min 10 in)
    col_w  = max(1.2, 10.0 / max(n_animals, 1))
    fig_w  = max(10, n_animals * col_w + 4.5)   # extra for dend + cbar
    fig_h  = max(6,  len(ids_ordered) * 0.38 + 3.0)

    cid_to_grp = _cluster_group_map(groups)
    has_groups = bool(cid_to_grp)

    with plt.style.context(t["mpl_style"]):
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=t["fig_bg"])
        # GridSpec: dendrogram | heatmap | [group strip] | colorbar
        dend_frac  = 1
        heat_frac  = max(n_animals * 2, 10)
        grp_frac   = 1 if has_groups else 0
        cbar_frac  = 1
        n_cols     = 4 if has_groups else 3
        width_ratios = ([dend_frac, heat_frac, grp_frac, cbar_frac]
                        if has_groups else [dend_frac, heat_frac, cbar_frac])
        gs = GridSpec(2, n_cols, figure=fig,
                      height_ratios=[1, 9],
                      width_ratios=width_ratios,
                      hspace=0.0, wspace=0.04,
                      left=0.06, right=0.97, top=0.91, bottom=0.14)

        # Dendrogram
        ax_dend = fig.add_subplot(gs[1, 0])
        ax_dend.set_facecolor(t["ax_bg"])
        if len(top_ids) >= 2:
            dendrogram(lnk, orientation="left", ax=ax_dend,
                       color_threshold=0,
                       above_threshold_color=t["subtext"],
                       link_color_func=lambda k: t["subtext"])
        ax_dend.axis("off")

        # Heatmap
        ax_heat = fig.add_subplot(gs[1, 1])
        im = ax_heat.imshow(mat_ordered, aspect="auto",
                            cmap="RdBu_r", interpolation="nearest",
                            vmin=-2.5, vmax=2.5)
        ax_heat.set_yticks(range(len(ids_ordered)))
        ax_heat.set_yticklabels(
            [f"C{cid}" if isinstance(cid, (int, np.integer)) else str(cid)
             for cid in ids_ordered],
            fontsize=8, color=t["tick"])
        ax_heat.set_xticks(range(n_animals))
        ax_heat.set_xticklabels([a["name"][:12] for a in animals_sorted],
                                 rotation=40, ha="right",
                                 fontsize=max(6, 8 - n_animals // 8),
                                 color=t["tick"])
        # Explicit xlim so ax_top can share it exactly
        ax_heat.set_xlim(-0.5, n_animals - 0.5)
        _style_ax(ax_heat, t)

        # Colour-strip: shares x-axis with heatmap → always aligned
        ax_top = fig.add_subplot(gs[0, 1], sharex=ax_heat)
        ax_top.set_facecolor(t["fig_bg"])
        for ai, clr in enumerate(col_colors):
            ax_top.add_patch(mpatches.Rectangle(
                (ai - 0.5, 0), 1.0, 1.0,
                facecolor=clr, edgecolor="none", alpha=0.88))
        ax_top.set_ylim(0, 1)
        ax_top.set_yticks([])
        plt.setp(ax_top.get_xticklabels(), visible=False)
        for spine in ax_top.spines.values():
            spine.set_visible(False)

        # Group labels centred over each exp-group block
        grp_start: dict = {}
        for ai, ani in enumerate(animals_sorted):
            eg = ani["exp_group"]
            if eg not in grp_start:
                grp_start[eg] = ai
        for eg, start in grp_start.items():
            end_idx = start
            for ai, ani in enumerate(animals_sorted):
                if ani["exp_group"] == eg:
                    end_idx = ai
            mid = (start + end_idx) / 2.0
            ax_top.text(mid, 0.5, eg,
                        ha="center", va="center",
                        fontsize=max(6, 8 - len(eg_names) // 3),
                        color=t["text"], fontweight="bold")

        # Legend in top-left of colour strip
        patches_leg = [mpatches.Patch(color=eg_colors_map[eg], label=eg)
                       for eg in eg_names]
        ax_top.legend(handles=patches_leg, loc="upper left", fontsize=7,
                      framealpha=0.3, facecolor=t["ax_bg"],
                      labelcolor=t["tick"], ncol=len(eg_names),
                      bbox_to_anchor=(0, 1.02, 1, 0.1),
                      mode="expand", borderaxespad=0)

        # Right-side behaviour-group strip (one cell per row)
        if has_groups:
            ax_grp = fig.add_subplot(gs[1, 2])
            ax_grp.set_facecolor(t["fig_bg"])
            seen_gnames: set = set()
            for ri, cid in enumerate(ids_ordered):
                cid_int = int(cid) if isinstance(cid, (int, np.integer)) else cid
                gname, gcolor = cid_to_grp.get(cid_int, ("", t["muted"]))
                ax_grp.add_patch(mpatches.Rectangle(
                    (0, ri - 0.5), 1.0, 1.0,
                    facecolor=gcolor, edgecolor="none", alpha=0.85))
                if gname and gname not in seen_gnames:
                    ax_grp.text(0.5, ri, gname,
                                ha="center", va="center",
                                fontsize=max(5, 7 - len(ids_ordered) // 15),
                                color=t["text"], clip_on=True)
                    seen_gnames.add(gname)
            ax_grp.set_xlim(0, 1)
            ax_grp.set_ylim(-0.5, len(ids_ordered) - 0.5)
            ax_grp.set_yticks([])
            ax_grp.set_xticks([])
            ax_grp.set_title("Group", color=t["tick"], fontsize=7)
            for spine in ax_grp.spines.values():
                spine.set_visible(False)
            # Colorbar in column 3
            ax_cbar = fig.add_subplot(gs[1, 3])
        else:
            # Colorbar in column 2
            ax_cbar = fig.add_subplot(gs[1, 2])

        cbar = fig.colorbar(im, cax=ax_cbar)
        cbar.set_label("Z-score", color=t["tick"], fontsize=8)
        cbar.ax.tick_params(colors=t["tick"], labelsize=7)
        cbar.outline.set_edgecolor(t["spine"])

        _unit_ht = "Transition Pairs" if metric == "transition_prob" else "Clusters"
        fig.suptitle(f"Hierarchical Heatmap — Top {len(ids_ordered)} Significant {_unit_ht}  "
                     f"({fdr_sig_column(stats_df)[1]} < {p_thresh}, {metric})",
                     color=t["tick"], fontweight="bold", fontsize=11)
    return fig


def build_reclustering_figure(recluster_result: dict, t: dict = None) -> plt.Figure:
    """Elbow + Silhouette plots for reclustering validation."""
    if t is None:
        t = T()

    sil = recluster_result["silhouette_scores"]
    ine = recluster_result["inertia_scores"]

    if not sil:
        with plt.style.context(t["mpl_style"]):
            fig, ax = plt.subplots(figsize=(7, 3), facecolor=t["fig_bg"])
            ax.text(0.5, 0.5, "Not enough clusters to evaluate (need   3)",
                    ha="center", va="center", color=t["tick"],
                    transform=ax.transAxes)
            _style_ax(ax, t)
        return fig

    ks_sil = sorted(sil.keys())
    ks_ine = sorted(ine.keys())

    with plt.style.context(t["mpl_style"]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                        facecolor=t["fig_bg"])
        # Elbow
        ax1.plot(ks_ine, [ine[k] for k in ks_ine],
                 color="#4E79A7", lw=2, marker="o", ms=6)
        ax1.set_xlabel("Number of merged groups (k)", color=t["tick"])
        ax1.set_ylabel("Within-cluster inertia", color=t["tick"])
        ax1.set_title("Elbow Plot", color=t["tick"], fontweight="bold")
        _style_ax(ax1, t)

        # Silhouette
        sv = [sil[k] for k in ks_sil]
        best_k = ks_sil[int(np.argmax(sv))]
        colors = ["#FF4081" if k == best_k else "#59A14F" for k in ks_sil]
        bars = ax2.bar(ks_sil, sv, color=colors, edgecolor="none", width=0.7)
        ax2.axhline(0, color=t["spine"], lw=0.8, ls="--")
        ax2.set_xlabel("Number of merged groups (k)", color=t["tick"])
        ax2.set_ylabel("Silhouette Score", color=t["tick"])
        ax2.set_title(f"Silhouette Scores  (best k = {best_k})",
                      color=t["tick"], fontweight="bold")
        for bar, k in zip(bars, ks_sil):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01, str(k),
                     ha="center", fontsize=8, color=t["tick"])
        _style_ax(ax2, t)

        fig.suptitle("Reclustering Validation - Choose Optimal k",
                     color=t["tick"], fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def build_recombination_comparison_figure(recluster_result: dict,
                                           k_list: list,
                                           animal_data: list,
                                           metric: str = "total_duration",
                                           eg_colors_override: dict = None,
                                           t: dict = None) -> plt.Figure:
    """
    Comparative direction-of-change visualisation for selected k values.

    Layout:  one column of subplots per k value.
    Within each subplot:
        X-axis  = merged groups (RG-1, RG-2, ..., RG-k) - each is a "grouped column"
        Nested bars within each merged group = experimental groups
        Y-axis  = NORMALISED metric  (each merged group's mean in the FIRST /
                  reference exp-group = 1.0, so all others show fold-change)
        Individual animal dots are overlaid
        Error bars = SEM across animals within that exp-group for that merged group

    This makes the *direction* of change the primary visual signal, regardless
    of whether one merged group is 10x larger than another in absolute terms.
    The reference group (first exp-group alphabetically / by insertion order) is
    always plotted as a dashed 1.0 baseline.
    """
    if t is None:
        t = T()

    lnk      = recluster_result["linkage_matrix"]
    eg_names = recluster_result["eg_names"]
    cids     = recluster_result["cluster_ids"]

    # Build per-animal per-cluster metric lookup once
    # animal_pcm_cache: {uid: DataFrame(index=cluster_id)}
    animal_pcm_cache = {
        ani["uid"]: compute_per_cluster_metrics(ani["df"], ani["fps"])
        for ani in animal_data
    }

    eg_colors: dict = {}
    for i, eg in enumerate(eg_names):
        if eg_colors_override and eg in eg_colors_override:
            eg_colors[eg] = eg_colors_override[eg]
        else:
            eg_colors[eg] = PALETTE[i % len(PALETTE)]
    ref_eg    = eg_names[0]   # reference exp-group for normalisation
    n_k       = len(k_list)
    n_eg      = len(eg_names)

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(
            n_k, 1,
            figsize=(max(10, len(eg_names) * 2.2 + 3), 5.5 * n_k),
            facecolor=t["fig_bg"],
            squeeze=False,
        )

        for row_i, k in enumerate(k_list):
            ax = axes[row_i, 0]
            labels = fcluster(lnk, t=k, criterion="maxclust")

            #   build merged-group definitions  
            # merged_groups: {merged_id: [cluster_id, ...]}
            merged_groups: dict = {}
            for ci, cid in zip(labels, cids):
                merged_groups.setdefault(int(ci), []).append(cid)
            mg_ids = sorted(merged_groups.keys())
            n_mg   = len(mg_ids)

            bar_w   = 0.7 / max(n_eg, 1)
            x_base  = np.arange(n_mg)
            rng     = np.random.default_rng(99 + k)

            #   compute per-(merged_group, exp_group) stats  
            # For each merged group we want:
            #   raw_mean[eg], raw_sem[eg], per_animal_vals[eg] = list
            # Then normalise so ref_eg = 1.0
            mg_stats = {}
            for mg in mg_ids:
                cluster_ids_in = merged_groups[mg]
                raw: dict = {eg: [] for eg in eg_names}
                for ani in animal_data:
                    eg  = ani["exp_group"]
                    pcm = animal_pcm_cache[ani["uid"]]
                    # sum (or mean) metric across all clusters in this merged group
                    total = 0.0
                    for cid in cluster_ids_in:
                        if cid in pcm.index:
                            total += float(pcm.loc[cid, metric])
                    raw[eg].append(total)
                means = {eg: float(np.mean(v)) if v else 0.0 for eg, v in raw.items()}
                sems  = {
                    eg: float(np.std(v, ddof=1) / np.sqrt(len(v)))
                    if len(v) > 1 else 0.0
                    for eg, v in raw.items()
                }
                ref_val = means[ref_eg] if means[ref_eg] != 0 else 1.0
                mg_stats[mg] = {
                    "means":   means,
                    "sems":    sems,
                    "raw":     raw,
                    "ref_val": ref_val,
                    "n_clusters": len(cluster_ids_in),
                }

            #   plot grouped bars  
            for ei, eg in enumerate(eg_names):
                offset  = (ei - (n_eg - 1) / 2) * bar_w
                norm_means, norm_sems, pts_x_all, pts_y_all = [], [], [], []

                for xi, mg in enumerate(mg_ids):
                    st      = mg_stats[mg]
                    ref_val = st["ref_val"]
                    nm      = st["means"][eg] / ref_val
                    ns      = st["sems"][eg]  / ref_val
                    norm_means.append(nm)
                    norm_sems.append(ns)

                    # per-animal dots
                    for v in st["raw"][eg]:
                        pts_x_all.append(
                            x_base[xi] + offset + rng.uniform(-bar_w*0.22, bar_w*0.22)
                        )
                        pts_y_all.append(v / ref_val)

                color = eg_colors[eg]
                # reference group: hatch to distinguish
                hatch = "" if eg != ref_eg else "///"
                ax.bar(x_base + offset, norm_means, width=bar_w * 0.88,
                       color=color, alpha=0.82, edgecolor="none",
                       linewidth=0.6, label=eg, hatch=hatch, zorder=2)
                ax.errorbar(x_base + offset, norm_means, yerr=norm_sems,
                            fmt="none", color=t["tick"], lw=1.2, capsize=3, zorder=3)
                ax.scatter(pts_x_all, pts_y_all,
                           color=t["ax_bg"], edgecolors=color,
                           s=22, linewidths=0.7, alpha=0.85, zorder=4)

            # reference line at 1.0
            ax.axhline(1.0, color=t["spine"], lw=1.0, ls="--", alpha=0.7,
                       label=f"ref ({ref_eg})")

            #   significance stars per merged group  
            if SCIPY_OK and n_eg >= 2:
                for xi, mg in enumerate(mg_ids):
                    st = mg_stats[mg]
                    groups_for_test = [np.array(st["raw"][eg]) for eg in eg_names]
                    try:
                        _, pv = sp_stats.kruskal(*groups_for_test)
                    except Exception:
                        pv = 1.0
                    stars = ("***" if pv < 0.001 else
                             "**"  if pv < 0.01  else
                             "*"   if pv < 0.05  else "")
                    if stars:
                        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 2.0
                        ax.text(x_base[xi], ymax * 0.95, stars,
                                ha="center", fontsize=11, color="#FF4081",
                                fontweight="bold")

            # x-axis labels: show RG-N and how many raw clusters it contains
            ax.set_xticks(x_base)
            ax.set_xticklabels(
                [f"RG-{mg}\n(n={mg_stats[mg]['n_clusters']} clust.)"
                 for mg in mg_ids],
                fontsize=9, color=t["tick"],
            )
            ax.set_ylabel(f"Normalised {metric.replace('_',' ')}\n(ref = {ref_eg})",
                          color=t["tick"])
            ax.set_title(f"k = {k} merged groups  -  direction of change vs {ref_eg}",
                         color=t["tick"], fontweight="bold", fontsize=11)
            ax.legend(fontsize=8, framealpha=0.3, facecolor=t["ax_bg"],
                      labelcolor=t["tick"], loc="upper right")
            _style_ax(ax, t)

        fig.suptitle(
            "Recombination Comparison  -  Normalised change across experimental groups\n"
            "(clusters are merged by similarity of change pattern, not absolute magnitude)",
            color=t["tick"], fontsize=11, fontweight="bold"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def build_top_n_barplot(stats_df: pd.DataFrame, animal_data: list,
                         top_n: int, p_thresh: float,
                         metric: str = "total_duration",
                         eg_colors_override: dict = None,
                         t: dict = None,
                         groups: dict = None,
                         combined: dict = None) -> plt.Figure:
    """
    Bar chart of top-N clusters: one subplot per cluster so each can have
    its own y-scale.  Stars always shown (*, **, ***, ns).
    Pass groups+combined to append user-defined behaviour-group panels.
    eg_colors_override: {eg_name: hex}
    """
    if t is None:
        t = T()

    _sig_col, _ = fdr_sig_column(stats_df)
    sig_df  = stats_df[stats_df[_sig_col] < p_thresh].head(top_n)
    top_ids = sig_df["cluster_id"].tolist()
    if not top_ids:
        top_ids = stats_df.head(min(top_n, len(stats_df)))["cluster_id"].tolist()
    if not top_ids:
        fig, ax = plt.subplots(figsize=(7, 3), facecolor=t["fig_bg"] if t else "#0d0d1a")
        ax.text(0.5, 0.5, "No clusters to display.", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    _pcm_fn = (
        (lambda a: compute_per_pair_transition_probs(a["df"]))
        if metric == "transition_prob"
        else
        (lambda a: compute_per_cluster_metrics(a["df"], a["fps"]))
    )
    eg_map: dict = {}
    for ani in animal_data:
        eg = ani["exp_group"]
        eg_map.setdefault(eg, []).append(_pcm_fn(ani))
    eg_names = list(eg_map.keys())
    eg_colors: dict = {}
    for i, eg in enumerate(eg_names):
        if eg_colors_override and eg in eg_colors_override:
            eg_colors[eg] = eg_colors_override[eg]
        else:
            eg_colors[eg] = PALETTE[i % len(PALETTE)]

    # Significance stars use the FDR q-value when available (else raw p).
    pvals_map = stats_df.set_index("cluster_id")[_sig_col].to_dict()

    n_c  = len(top_ids)
    n_eg = len(eg_names)
    rng  = np.random.default_rng(7)

    #   layout: separate subplot per cluster for independent y-scales  
    ncols = min(n_c, 5)
    nrows = int(np.ceil(n_c / ncols))
    fig_w = max(12, ncols * 2.4)
    fig_h = max(4,  nrows * 3.2) + 0.6
    with plt.style.context(t["mpl_style"]):
        fig, axes_flat = plt.subplots(
            nrows, ncols,
            figsize=(fig_w, fig_h),
            facecolor=t["fig_bg"],
            squeeze=False,
        )
        axes_flat = axes_flat.flatten()

        for xi, cid in enumerate(top_ids):
            ax   = axes_flat[xi]
            pv   = pvals_map.get(cid, 1.0)

            x_pos  = np.arange(n_eg)
            means, sems, all_pts = [], [], []
            for eg in eg_names:
                vs = [float(pcm.loc[cid, metric]) if cid in pcm.index else 0.0
                      for pcm in eg_map[eg]]
                means.append(float(np.mean(vs)))
                sems.append(float(np.std(vs, ddof=1) / np.sqrt(len(vs))) if len(vs) > 1 else 0.0)
                all_pts.append(vs)

            for ei, eg in enumerate(eg_names):
                ec = eg_colors[eg]
                ax.bar(x_pos[ei], means[ei], width=0.6,
                       color=ec, alpha=0.82, edgecolor="none",
                       linewidth=0.6, zorder=2)
                ax.errorbar(x_pos[ei], means[ei], yerr=sems[ei], fmt="none",
                            color=t["tick"], lw=1.4, capsize=3, zorder=3)
                jit = rng.uniform(-0.12, 0.12, len(all_pts[ei]))
                ax.scatter(x_pos[ei] + jit, all_pts[ei],
                           color=t["ax_bg"], edgecolors=ec,
                           s=22, linewidths=0.7, alpha=0.88, zorder=4)

            # significance bracket + stars — always shown (inc. ns)
            stars = ("***" if pv < 0.001 else "**" if pv < 0.01
                     else "*" if pv < 0.05 else "ns")
            data_top = max((m + s for m, s in zip(means, sems)), default=0)
            max_pt   = max((max(pts) for pts in all_pts if pts), default=0)
            data_ceil = max(data_top, max_pt)
            head_top = max(data_ceil * 1.65, 0.001)
            ax.set_ylim(bottom=0, top=head_top)
            if n_eg >= 2:
                bh = head_top * 0.88
                ax.plot([0, 0, n_eg - 1, n_eg - 1],
                        [bh, bh + head_top * 0.025, bh + head_top * 0.025, bh],
                        lw=0.8, color=t["tick"])
                star_color = "#FF4081" if stars != "ns" else t["muted"]
                ax.text((n_eg - 1) / 2, bh + head_top * 0.033, stars,
                        ha="center", fontsize=10, color=star_color,
                        fontweight="bold" if stars != "ns" else "normal")
                # With >2 groups, localise the omnibus effect with pairwise
                # post-hoc brackets (FDR-adjusted) below the omnibus bracket so
                # the single star is no longer ambiguous about WHICH pair differs.
                if n_eg > 2 and stars != "ns":
                    try:
                        _ytop = draw_sig_brackets(
                            ax, eg_names, all_pts,
                            y_start=data_ceil * 1.06, y_step=head_top * 0.09,
                            tick_color=t["tick"])
                        if _ytop > bh * 0.92:
                            ax.set_ylim(top=max(head_top, _ytop * 1.12))
                    except Exception:
                        pass

            ax.set_xticks(x_pos)
            ax.set_xticklabels(eg_names, rotation=22, ha="right",
                               color=t["tick"], fontsize=7)
            _cid_label = f"C{cid}" if isinstance(cid, (int, np.integer)) else str(cid)
            ax.set_title(_cid_label, color=t["tick"],
                         fontsize=9, fontweight="bold")
            ax.set_ylabel(metric.replace("_", " ").title(),
                          color=t["tick"], fontsize=7)
            _style_ax(ax, t)

        # hide empty subplots
        for xi in range(len(top_ids), len(axes_flat)):
            axes_flat[xi].set_visible(False)

        handles = [mpatches.Patch(color=eg_colors[eg], label=eg) for eg in eg_names]
        fig.legend(handles=handles, loc="upper right", fontsize=8,
                   framealpha=0.3, facecolor=t["ax_bg"], labelcolor=t["tick"])
        _sw = fdr_sig_column(stats_df)[1]
        _unit_pl = "Transition Pairs" if metric == "transition_prob" else "Clusters"
        fig.suptitle(
            f"Top {len(top_ids)} Significant {_unit_pl}  ({_sw} < {p_thresh})  -  "
            f"Each panel has its own y-scale  -  "
            f"* {_sw}<0.05  ** {_sw}<0.01  *** {_sw}<0.001",
            color=t["tick"], fontweight="bold", fontsize=10, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

    # ── User-defined behaviour-group panels (appended as a second figure) ──────
    if groups and combined and SCIPY_OK:
        records_c  = combined["records"]
        uid2idx_c  = combined["uid_to_idx"]
        expg_c     = combined["exp_groups"]
        eg_names_c = list(expg_c.keys())
        bg_names   = list(groups.keys())
        n_bg       = len(bg_names)
        pvals_bg   = compute_combined_pvals(combined, groups)
        rng2       = np.random.default_rng(13)

        ncols_bg = min(n_bg, 5)
        nrows_bg = int(np.ceil(n_bg / ncols_bg))
        fig_bg_w = max(12, ncols_bg * 2.6)
        fig_bg_h = max(4,  nrows_bg * 3.4) + 0.8

        with plt.style.context(t["mpl_style"]):
            fig2, ax2s = plt.subplots(
                nrows_bg, ncols_bg,
                figsize=(fig_bg_w, fig_bg_h),
                facecolor=t["fig_bg"], squeeze=False,
            )
            ax2s_flat = ax2s.flatten()

            for bi, bg in enumerate(bg_names):
                ax2 = ax2s_flat[bi]
                pv2_dict = pvals_bg.get((metric, bg)) or {}
                qv2 = pv2_dict.get("qval")
                bg_color = groups[bg]["color"]
                x2 = np.arange(len(eg_names_c))
                means2, sems2, pts2 = [], [], []
                for eg in eg_names_c:
                    vs = [records_c[uid2idx_c[uid]]["metrics"]
                          .get(bg, {}).get(metric) or 0.0
                          for uid in expg_c[eg]]
                    means2.append(float(np.mean(vs)))
                    sems2.append(float(np.std(vs, ddof=1) / np.sqrt(len(vs)))
                                 if len(vs) > 1 else 0.0)
                    pts2.append(vs)

                for ei, eg in enumerate(eg_names_c):
                    ec2 = eg_colors.get(eg, PALETTE[ei % len(PALETTE)])
                    ax2.bar(x2[ei], means2[ei], width=0.6,
                            color=ec2, edgecolor="none",
                            linewidth=0, alpha=0.82, zorder=2)
                    ax2.errorbar(x2[ei], means2[ei], yerr=sems2[ei],
                                 fmt="none", color=t["tick"], lw=1.2, capsize=3, zorder=3)
                    jit2 = rng2.uniform(-0.12, 0.12, len(pts2[ei]))
                    ax2.scatter(x2[ei] + jit2, pts2[ei], color=t["ax_bg"],
                                edgecolors=ec2, s=22, linewidths=0.7,
                                alpha=0.88, zorder=4)

                stars2 = ("***" if qv2 is not None and qv2 < 0.001 else
                          "**"  if qv2 is not None and qv2 < 0.01  else
                          "*"   if qv2 is not None and qv2 < 0.05  else "ns")
                dt2  = max((m + s for m, s in zip(means2, sems2)), default=0)
                mp2  = max((max(p) for p in pts2 if p), default=0)
                ht2  = max(dt2 * 1.40, mp2 * 1.40, 0.001)
                ax2.set_ylim(bottom=0, top=ht2)
                if len(eg_names_c) >= 2:
                    bh2 = ht2 * 0.78
                    ax2.plot([0, 0, len(eg_names_c) - 1, len(eg_names_c) - 1],
                             [bh2, bh2 + ht2 * 0.04, bh2 + ht2 * 0.04, bh2],
                             lw=0.9, color=t["tick"])
                    sc2 = "#FF4081" if stars2 != "ns" else t["muted"]
                    ax2.text((len(eg_names_c) - 1) / 2, bh2 + ht2 * 0.055,
                             stars2, ha="center", fontsize=10, color=sc2,
                             fontweight="bold" if stars2 != "ns" else "normal")

                ax2.set_xticks(x2)
                ax2.set_xticklabels(eg_names_c, rotation=22, ha="right",
                                    color=t["tick"], fontsize=7)
                ax2.set_title(bg, color=bg_color, fontsize=9, fontweight="bold")
                ax2.set_ylabel(metric.replace("_", " ").title(),
                               color=t["tick"], fontsize=7)
                _style_ax(ax2, t)

            for bi in range(n_bg, len(ax2s_flat)):
                ax2s_flat[bi].set_visible(False)

            handles2 = [mpatches.Patch(color=eg_colors.get(eg, PALETTE[i % len(PALETTE)]),
                                       label=eg)
                        for i, eg in enumerate(eg_names_c)]
            fig2.legend(handles=handles2, loc="upper right", fontsize=8,
                        framealpha=0.3, facecolor=t["ax_bg"], labelcolor=t["tick"])
            fig2.suptitle(
                f"User-Defined Behaviour Groups  —  {metric.replace('_',' ').title()}"
                f"  —  * FDR q<0.05  ** q<0.01  *** q<0.001",
                color=t["tick"], fontweight="bold", fontsize=10, y=0.99)
            fig2.tight_layout(rect=[0, 0, 1, 0.97])
        return fig, fig2

    return fig


def build_distance_matrix_figure(recluster_result: dict, groups: dict = None,
                                  t: dict = None) -> plt.Figure:
    """
    3-panel figure:
      Left:  Blended distance matrix (main clustering input)
      Middle: Correlation-distance matrix (change-pattern similarity)
      Right:  Transition-distance matrix (temporal co-occurrence)
    Rows/cols ordered by the linkage dendrogram so structure is visible.
    When groups is supplied, tick labels are coloured by behaviour group.
    """
    if t is None:
        t = T()
    cids       = recluster_result["cluster_ids"]
    lnk        = recluster_result["linkage_matrix"]
    dist_blend = recluster_result["dist_matrix"]
    corr_dist  = recluster_result.get("corr_dist_matrix", dist_blend)
    trans_dist = recluster_result.get("trans_dist_matrix",
                                      np.ones_like(dist_blend))

    # Order clusters by correlation-distance linkage so clusters with similar
    # change patterns between experimental groups appear adjacent in all panels.
    if len(cids) >= 2 and SCIPY_CLUSTER_OK:
        try:
            corr_condensed = squareform(corr_dist, checks=False)
            corr_condensed = np.clip(corr_condensed, 0, None)
            corr_lnk = linkage(corr_condensed, method="ward")
            dend  = dendrogram(corr_lnk, no_plot=True)
            order = dend["leaves"]
        except Exception:
            dend  = dendrogram(lnk, no_plot=True)
            order = dend["leaves"]
    elif len(cids) >= 2:
        dend  = dendrogram(lnk, no_plot=True)
        order = dend["leaves"]
    else:
        order = list(range(len(cids)))

    cids_ord   = [cids[i] for i in order]
    labels_ord = [f"C{c}" for c in cids_ord]
    cid_to_grp = _cluster_group_map(groups)

    def _reorder(m):
        return m[np.ix_(order, order)]

    def _tick_colors(tick_pos_list):
        out = []
        for i in tick_pos_list:
            cid_int = int(cids_ord[i])
            out.append(cid_to_grp[cid_int][1] if cid_int in cid_to_grp else t["tick"])
        return out

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(
            1, 3,
            figsize=(max(18, len(cids) * 0.28 * 3 + 6), max(5, len(cids) * 0.28 + 2.5)),
            facecolor=t["fig_bg"],
        )
        # Correlation distance first (primary input), blended second (includes
        # correlation), transition distance last (secondary input).
        titles = [
            "Correlation Distance\n(change-pattern across experimental groups)",
            "Blended Distance\n(70 % correlation + 30 % transition  —  clustering input)",
            "Transition Distance\n(temporal co-occurrence)",
        ]
        mats  = [_reorder(corr_dist), _reorder(dist_blend), _reorder(trans_dist)]
        cmaps = ["RdBu_r", "viridis", "YlOrRd"]

        for ax, mat, title, cmap in zip(axes, mats, titles, cmaps):
            im = ax.imshow(mat, cmap=cmap, aspect="auto", interpolation="nearest")
            n = len(labels_ord)
            tick_step   = max(1, n // 20)
            tick_pos    = list(range(0, n, tick_step))
            tick_labels = [labels_ord[i] for i in tick_pos]
            fsize = max(4, 7 - n // 20)
            ax.set_xticks(tick_pos)
            for lbl_obj, clr in zip(
                    ax.set_xticklabels(tick_labels, rotation=70, fontsize=fsize),
                    _tick_colors(tick_pos)):
                lbl_obj.set_color(clr)
            ax.set_yticks(tick_pos)
            for lbl_obj, clr in zip(
                    ax.set_yticklabels(tick_labels, fontsize=fsize),
                    _tick_colors(tick_pos)):
                lbl_obj.set_color(clr)
            ax.set_title(title, color=t["tick"], fontsize=9, fontweight="bold")
            cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
            cb.ax.tick_params(colors=t["tick"], labelsize=7)
            cb.outline.set_edgecolor(t["spine"])
            _style_ax(ax, t)

        fig.suptitle(
            "Cluster Distance Matrices  (ordered by change-pattern similarity)\n"
            "Clusters that increase/decrease together appear adjacent  ·  "
            "Blended = 70 % correlation + 30 % transition",
            color=t["tick"], fontweight="bold", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def build_transition_figure(recluster_result: dict, groups: dict = None,
                             t: dict = None) -> plt.Figure:
    """
    Visualise transition dynamics — three panels:
      Top-left:  Average transition probability heatmap (row-stochastic)
      Top-right: Transition-profile similarity (cosine similarity)
      Bottom:    Circular transition network — directed arrows sized by probability
    When groups is supplied, heatmap tick labels and network nodes are coloured
    by behaviour group.
    """
    if t is None:
        t = T()
    trans = recluster_result.get("transition_result")
    if trans is None:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No transition data available.\nRun Reclustering first.",
                ha="center", va="center", color=t["tick"], transform=ax.transAxes)
        _style_ax(ax, t); return fig

    cids_all = recluster_result["cluster_ids"]
    lnk      = recluster_result["linkage_matrix"]
    tmat_avg = trans["tmat_avg"]
    sim_mat  = trans["sim_matrix"]
    trans_cids = trans["cluster_ids"]

    # Restrict to filtered clusters, reindex into trans_cids
    t_idx  = {c: i for i, c in enumerate(trans_cids)}
    subset = [t_idx[c] for c in cids_all if c in t_idx]
    tmat_sub = tmat_avg[np.ix_(subset, subset)]
    sim_sub  = sim_mat[np.ix_(subset, subset)]

    if len(cids_all) >= 2:
        dend  = dendrogram(lnk, no_plot=True)
        order = dend["leaves"]
    else:
        order = list(range(len(cids_all)))

    cids_ord   = [cids_all[i] for i in order]
    labels_ord = [f"C{c}" for c in cids_ord]
    tmat_ord   = tmat_sub[np.ix_(order, order)]
    sim_ord    = sim_sub[np.ix_(order, order)]
    n          = len(labels_ord)

    cid_to_grp = _cluster_group_map(groups)

    def _label_color(cid_int):
        return cid_to_grp[cid_int][1] if cid_int in cid_to_grp else t["tick"]

    row_h  = max(5, n * 0.32 + 2)
    net_h  = max(6, min(n * 0.4 + 2, 12))
    fig_w  = max(16, n * 0.32 * 2 + 5)

    with plt.style.context(t["mpl_style"]):
        fig = plt.figure(figsize=(fig_w, row_h + net_h), facecolor=t["fig_bg"])
        gs  = fig.add_gridspec(2, 2, height_ratios=[row_h, net_h],
                               hspace=0.35, wspace=0.25,
                               left=0.06, right=0.96, top=0.93, bottom=0.04)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        tick_step = max(1, n // 20)
        tick_pos  = list(range(0, n, tick_step))
        tick_lbls = [labels_ord[i] for i in tick_pos]
        tick_clrs = [_label_color(int(cids_ord[i])) for i in tick_pos]
        fsize     = max(4, 7 - n // 20)

        # ── ax1: transition probability heatmap ──────────────────────────────
        # NaN-mask diagonal and below-chance entries so the colormap spans only
        # the above-chance range [chance_floor … max_observed_prob].
        chance_floor_t = 1.0 / max(1, n - 1)
        tmat_display = tmat_ord.copy().astype(float)
        np.fill_diagonal(tmat_display, np.nan)
        tmat_display[tmat_display <= chance_floor_t] = np.nan
        vmax_tmat = float(np.nanmax(tmat_display)) \
            if not np.all(np.isnan(tmat_display)) else 1.0
        cmap_tmat = plt.cm.magma.copy()
        cmap_tmat.set_bad(color=t["ax_bg"])

        im1 = ax1.imshow(tmat_display, cmap=cmap_tmat, aspect="auto",
                         interpolation="nearest",
                         vmin=chance_floor_t, vmax=vmax_tmat)
        ax1.set_xticks(tick_pos)
        for lbl, clr in zip(ax1.set_xticklabels(tick_lbls, rotation=70, fontsize=fsize),
                             tick_clrs):
            lbl.set_color(clr)
        ax1.set_yticks(tick_pos)
        for lbl, clr in zip(ax1.set_yticklabels(tick_lbls, fontsize=fsize), tick_clrs):
            lbl.set_color(clr)
        ax1.set_title(
            f"Average Transition Probabilities\n"
            f"(diagonal excluded, above-chance only  p > {chance_floor_t:.3f}  |  "
            f"colourmap: {chance_floor_t:.3f} → {vmax_tmat:.3f})",
            color=t["tick"], fontsize=9, fontweight="bold")
        cb1 = fig.colorbar(im1, ax=ax1, shrink=0.75, pad=0.02)
        cb1.ax.tick_params(colors=t["tick"], labelsize=7)
        cb1.outline.set_edgecolor(t["spine"])
        _style_ax(ax1, t)

        # ── ax2: transition-profile similarity (cosine) ───────────────────────
        # Cosine similarity is not a probability, so no chance-floor rescaling.
        im2 = ax2.imshow(sim_ord, cmap="viridis", aspect="auto",
                         interpolation="nearest", vmin=0, vmax=1)
        ax2.set_xticks(tick_pos)
        for lbl, clr in zip(ax2.set_xticklabels(tick_lbls, rotation=70, fontsize=fsize),
                             tick_clrs):
            lbl.set_color(clr)
        ax2.set_yticks(tick_pos)
        for lbl, clr in zip(ax2.set_yticklabels(tick_lbls, fontsize=fsize), tick_clrs):
            lbl.set_color(clr)
        ax2.set_title("Transition-Profile Similarity\n(cosine similarity of outgoing profiles)",
                      color=t["tick"], fontsize=9, fontweight="bold")
        cb2 = fig.colorbar(im2, ax=ax2, shrink=0.75, pad=0.02)
        cb2.ax.tick_params(colors=t["tick"], labelsize=7)
        cb2.outline.set_edgecolor(t["spine"])
        _style_ax(ax2, t)

        # ── Circular transition network ───────────────────────────────────────
        max_net    = min(n, 30)
        net_order  = list(range(max_net))
        net_labels = [labels_ord[i] for i in net_order]
        net_tmat   = tmat_ord[np.ix_(net_order, net_order)]
        np.fill_diagonal(net_tmat, 0)
        # Row-normalise: tmat_avg is an animal-average so rows may not sum to 1;
        # subsetting to net_order makes this worse.  Normalising is required for
        # the 1/(nn-1) chance floor in _draw_circle_network to be meaningful.
        _bt_rs = net_tmat.sum(axis=1, keepdims=True)
        _bt_rs[_bt_rs == 0] = 1.0
        net_tmat = net_tmat / _bt_rs

        nn     = len(net_labels)
        theta  = np.linspace(0, 2 * np.pi, nn, endpoint=False) - np.pi / 2
        nx_pos = np.cos(theta)
        ny_pos = np.sin(theta)

        node_colors = [
            cid_to_grp[int(cids_ord[i])][1]
            if int(cids_ord[i]) in cid_to_grp
            else PALETTE[int(cids_ord[i]) % len(PALETTE)]
            for i in net_order
        ]
        _draw_circle_network(ax3, net_tmat, nx_pos, ny_pos,
                             net_labels, node_colors, t)
        ax3.set_title(
            f"Transition Network  (above-chance edges only, n={nn} clusters)\n"
            "Node size = total outgoing strength · Arrow width = conditional transition probability",
            color=t["tick"], fontsize=9, fontweight="bold")

        # Group legend on network panel
        if cid_to_grp:
            seen: dict = {}
            for gname, gcolor in cid_to_grp.values():
                seen[gname] = gcolor
            leg_patches = [mpatches.Patch(color=c, label=g) for g, c in seen.items()]
            ax3.legend(handles=leg_patches, fontsize=7, framealpha=0.3,
                       facecolor=t["ax_bg"], labelcolor=t["tick"],
                       loc="upper right")

        fig.suptitle(
            "Transition Dynamics  (heatmaps ordered by dendrogram)\n"
            "Clusters that frequently transition to the same targets are pulled together in reclustering",
            color=t["tick"], fontweight="bold", fontsize=11)
    return fig


def build_group_transition_comparison_figure(
        animal_data: list,
        eg_colors: dict,
        groups: dict = None,
        top_n: int = 10,
        t: dict = None,
) -> plt.Figure:
    """
    4-panel EG comparison figure for transition probabilities.
    Panel A (top row): Per-EG average transition heatmaps, shared colourbar.
    Panel B (bottom-left): Differential heatmap (2-EG) or std across EG means (>2-EG).
    Panel C (bottom-right): Transition volcano (2-EG only) or placeholder.
    Panel D (bottom-centre): Per-animal transition PCA scatter coloured by EG.
    """
    if t is None:
        t = T()

    eg_names_all = list(dict.fromkeys(a["exp_group"] for a in animal_data))
    eg_names = eg_names_all[:4]          # cap display at 4
    truncated = len(eg_names_all) > 4
    n_eg = len(eg_names)

    if n_eg < 2:
        fig, ax = plt.subplots(figsize=(7, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "Need at least 2 experimental groups.",
                ha="center", va="center", color=t["tick"], transform=ax.transAxes)
        _style_ax(ax, t)
        return fig

    # ── Union cluster set ─────────────────────────────────────────────────────
    all_cids = sorted({
        int(lab) for a in animal_data for lab in a["df"]["label"].unique()
    })
    n_cl = len(all_cids)
    cl_idx = {c: i for i, c in enumerate(all_cids)}
    cid_to_grp = _cluster_group_map(groups)

    # ── Per-EG average T matrices ─────────────────────────────────────────────
    eg_tmats: dict = {}
    for eg in eg_names:
        eg_anis = [a for a in animal_data if a["exp_group"] == eg]
        tmat_sum = np.zeros((n_cl, n_cl), dtype=float)
        for ani in eg_anis:
            mat, ani_cids = compute_transition_matrix(ani["df"])
            for ri, ci in enumerate(ani_cids):
                for cj_i, cj in enumerate(ani_cids):
                    if ci in cl_idx and cj in cl_idx:
                        tmat_sum[cl_idx[ci], cl_idx[cj]] += mat[ri, cj_i]
        eg_tmats[eg] = tmat_sum / max(len(eg_anis), 1)

    # ── Per-animal transition vectors (for Panel D PCA) ───────────────────────
    X_pca, pair_names_pca = _compute_per_animal_transition_vectors(animal_data)
    eg_labels_pca = [a["exp_group"] for a in animal_data]

    # ── EG colour map (fall back to PALETTE) ─────────────────────────────────
    eg_col: dict = {}
    for i, eg in enumerate(eg_names):
        eg_col[eg] = eg_colors.get(eg, PALETTE[i % len(PALETTE)])

    # ── Tick label colours ────────────────────────────────────────────────────
    tick_labels = [f"C{c}" for c in all_cids]
    tick_clrs   = [cid_to_grp[c][1] if c in cid_to_grp else t["tick"]
                   for c in all_cids]
    fsize = max(4, 7 - n_cl // 20)
    tick_step = max(1, n_cl // 20)
    tick_pos  = list(range(0, n_cl, tick_step))

    # ── Shared vmin/vmax for Panel A ─────────────────────────────────────────
    chance_floor = 1.0 / max(1, n_cl - 1)
    all_vals = []
    for eg in eg_names:
        m = eg_tmats[eg].copy().astype(float)
        np.fill_diagonal(m, np.nan)
        m[m <= chance_floor] = np.nan
        all_vals.append(m)
    stacked = np.array(all_vals)
    vmax_a = float(np.nanmax(stacked)) if not np.all(np.isnan(stacked)) else 1.0

    # ── Figure layout ─────────────────────────────────────────────────────────
    n_top_cols = max(n_eg, 3)
    cell_size  = max(0.28, 5.0 / max(n_cl, 1))
    hmap_size  = n_cl * cell_size + 2.0
    fig_w      = max(16, n_top_cols * (hmap_size + 0.5) + 2)
    fig_h      = max(10, hmap_size * 2 + 3)

    with plt.style.context(t["mpl_style"]):
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=t["fig_bg"])
        fig.patch.set_facecolor(t["fig_bg"])

        outer_gs = GridSpec(2, 1, figure=fig,
                            height_ratios=[1, 1],
                            hspace=0.45,
                            left=0.05, right=0.97, top=0.91, bottom=0.06)

        top_gs = GridSpecFromSubplotSpec(
            1, n_eg, subplot_spec=outer_gs[0], wspace=0.1)
        bot_gs = GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer_gs[1],
            wspace=0.35, width_ratios=[1, 1.4, 1])

        # ── Panel A — per-EG heatmaps ─────────────────────────────────────────
        ax_egs = [fig.add_subplot(top_gs[0, i]) for i in range(n_eg)]
        cmap_a = plt.cm.magma.copy()
        cmap_a.set_bad(color=t["ax_bg"])
        _last_im = None
        for idx_eg, eg in enumerate(eg_names):
            ax = ax_egs[idx_eg]
            m  = eg_tmats[eg].copy().astype(float)
            np.fill_diagonal(m, np.nan)
            m[m <= chance_floor] = np.nan
            _last_im = ax.imshow(m, cmap=cmap_a, aspect="auto",
                                 interpolation="nearest",
                                 vmin=chance_floor, vmax=vmax_a)
            ax.set_xticks([tick_pos[k] for k in range(len(tick_pos))])
            for lbl, clr in zip(
                    ax.set_xticklabels(
                        [tick_labels[p] for p in tick_pos],
                        rotation=70, fontsize=fsize),
                    [tick_clrs[p] for p in tick_pos]):
                lbl.set_color(clr)
            ax.set_yticks([tick_pos[k] for k in range(len(tick_pos))])
            for lbl, clr in zip(
                    ax.set_yticklabels(
                        [tick_labels[p] for p in tick_pos],
                        fontsize=fsize),
                    [tick_clrs[p] for p in tick_pos]):
                lbl.set_color(clr)
            _n_eg_anis = sum(1 for a in animal_data if a["exp_group"] == eg)
            ax.set_title(
                f"{eg}  (n={_n_eg_anis})",
                color=eg_col.get(eg, t["tick"]), fontsize=9, fontweight="bold")
            # EG colour strip above each heatmap
            try:
                from matplotlib.patches import Rectangle as _Rect
                _strip_h = 0.04
                ax.add_patch(_Rect((0, 1.01), 1, _strip_h,
                                   transform=ax.transAxes, clip_on=False,
                                   facecolor=eg_col.get(eg, t["tick"]),
                                   linewidth=0))
            except Exception:
                pass
            _style_ax(ax, t)

        # Shared colorbar to the right of Panel A
        if _last_im is not None:
            cb = fig.colorbar(_last_im, ax=ax_egs, shrink=0.65, pad=0.02)
            cb.ax.tick_params(colors=t["tick"], labelsize=7)
            cb.outline.set_edgecolor(t["spine"])

        if truncated:
            fig.text(0.5, 0.95,
                     f"⚠ Showing first 4 of {len(eg_names_all)} experimental groups",
                     ha="center", color=t["muted"], fontsize=8)

        # ── Panel B — differential heatmap (bottom-left) ──────────────────────
        ax_b = fig.add_subplot(bot_gs[0, 0])
        _style_ax(ax_b, t)
        cmap_b = plt.cm.RdBu_r.copy()
        cmap_b.set_bad(color=t["ax_bg"])

        if n_eg == 2:
            eg_a, eg_b = eg_names[0], eg_names[1]
            delta = eg_tmats[eg_a].copy() - eg_tmats[eg_b].copy()
            np.fill_diagonal(delta, np.nan)
            off_diag = delta[~np.isnan(delta)]
            clip = float(np.percentile(np.abs(off_diag), 95)) if off_diag.size else 1.0
            clip = max(clip, 1e-6)
            im_b = ax_b.imshow(delta, cmap=cmap_b, aspect="auto",
                               interpolation="nearest", vmin=-clip, vmax=clip)
            ax_b.set_xticks([tick_pos[k] for k in range(len(tick_pos))])
            for lbl, clr in zip(
                    ax_b.set_xticklabels(
                        [tick_labels[p] for p in tick_pos],
                        rotation=70, fontsize=fsize),
                    [tick_clrs[p] for p in tick_pos]):
                lbl.set_color(clr)
            ax_b.set_yticks([tick_pos[k] for k in range(len(tick_pos))])
            for lbl, clr in zip(
                    ax_b.set_yticklabels(
                        [tick_labels[p] for p in tick_pos],
                        fontsize=fsize),
                    [tick_clrs[p] for p in tick_pos]):
                lbl.set_color(clr)
            ax_b.set_title(
                f"ΔT  ({eg_a} − {eg_b})\n"
                f"red = more in {eg_a}  ·  blue = more in {eg_b}",
                color=t["tick"], fontsize=8, fontweight="bold")
            cb_b = fig.colorbar(im_b, ax=ax_b, shrink=0.65, pad=0.02)
            cb_b.ax.tick_params(colors=t["tick"], labelsize=7)
            cb_b.set_label("ΔT  (A − B)", color=t["tick"], fontsize=7)
            cb_b.outline.set_edgecolor(t["spine"])
        else:
            # >2 EGs: cell-wise std across per-EG means
            stack_eg = np.stack([eg_tmats[eg] for eg in eg_names], axis=0)
            std_mat  = np.std(stack_eg, axis=0)
            np.fill_diagonal(std_mat, np.nan)
            cmap_std = plt.cm.magma.copy()
            cmap_std.set_bad(color=t["ax_bg"])
            im_b = ax_b.imshow(std_mat, cmap=cmap_std, aspect="auto",
                               interpolation="nearest")
            ax_b.set_xticks([tick_pos[k] for k in range(len(tick_pos))])
            for lbl, clr in zip(
                    ax_b.set_xticklabels(
                        [tick_labels[p] for p in tick_pos],
                        rotation=70, fontsize=fsize),
                    [tick_clrs[p] for p in tick_pos]):
                lbl.set_color(clr)
            ax_b.set_yticks([tick_pos[k] for k in range(len(tick_pos))])
            for lbl, clr in zip(
                    ax_b.set_yticklabels(
                        [tick_labels[p] for p in tick_pos],
                        fontsize=fsize),
                    [tick_clrs[p] for p in tick_pos]):
                lbl.set_color(clr)
            ax_b.set_title(
                "Std(T) across EG means\n"
                "(switch to 2 EGs for a differential view)",
                color=t["tick"], fontsize=8, fontweight="bold")
            cb_b = fig.colorbar(im_b, ax=ax_b, shrink=0.65, pad=0.02)
            cb_b.ax.tick_params(colors=t["tick"], labelsize=7)
            cb_b.outline.set_edgecolor(t["spine"])

        # ── Panel D — per-animal transition PCA (bottom-centre) ───────────────
        ax_d = fig.add_subplot(bot_gs[0, 1])
        _style_ax(ax_d, t)
        _markers = ["o", "s", "^", "D"]
        if X_pca.shape[0] >= 2 and X_pca.shape[1] >= 2:
            try:
                from sklearn.decomposition import PCA as _PCA
                _pca = _PCA(n_components=2)
                _X2  = _pca.fit_transform(X_pca)
                for eg_i, eg in enumerate(eg_names):
                    _mask = [i for i, eg_l in enumerate(eg_labels_pca)
                             if eg_l == eg]
                    if not _mask:
                        continue
                    _pts = _X2[_mask]
                    ax_d.scatter(
                        _pts[:, 0], _pts[:, 1],
                        color=eg_col.get(eg, PALETTE[eg_i % len(PALETTE)]),
                        marker=_markers[eg_i % len(_markers)],
                        alpha=0.82,
                        edgecolors=t["spine"], linewidths=0.6,
                        label=eg, s=55, zorder=3)
                    # Convex hull outline if ≥3 animals in this EG
                    if len(_mask) >= 3 and SCIPY_OK:
                        try:
                            from scipy.spatial import ConvexHull as _CH
                            _hull = _CH(_pts)
                            _hv   = np.append(_hull.vertices, _hull.vertices[0])
                            ax_d.plot(_pts[_hv, 0], _pts[_hv, 1],
                                      color=eg_col.get(eg, PALETTE[eg_i % len(PALETTE)]),
                                      alpha=0.3, lw=1)
                        except Exception:
                            pass
                var1 = _pca.explained_variance_ratio_[0] * 100
                var2 = _pca.explained_variance_ratio_[1] * 100
                ax_d.set_xlabel(f"PC1 ({var1:.1f}%)", color=t["tick"], fontsize=8)
                ax_d.set_ylabel(f"PC2 ({var2:.1f}%)", color=t["tick"], fontsize=8)
                ax_d.set_title("Per-animal Transition PCA",
                               color=t["tick"], fontsize=9, fontweight="bold")
                _leg_patches = [
                    mpatches.Patch(color=eg_col.get(eg, PALETTE[i % len(PALETTE)]),
                                   label=eg)
                    for i, eg in enumerate(eg_names)]
                ax_d.legend(handles=_leg_patches, fontsize=7, framealpha=0.3,
                            facecolor=t["ax_bg"], labelcolor=t["tick"])
            except ImportError:
                ax_d.text(0.5, 0.5, "sklearn not available\n(install scikit-learn)",
                          ha="center", va="center",
                          color=t["muted"], transform=ax_d.transAxes, fontsize=9)
                ax_d.set_title("Per-animal Transition PCA",
                               color=t["tick"], fontsize=9, fontweight="bold")
        else:
            ax_d.text(0.5, 0.5,
                      "Insufficient data for PCA\n(need ≥2 animals and ≥2 clusters)",
                      ha="center", va="center",
                      color=t["muted"], transform=ax_d.transAxes, fontsize=9)
            ax_d.set_title("Per-animal Transition PCA",
                           color=t["tick"], fontsize=9, fontweight="bold")

        # ── Panel C — transition volcano (bottom-right, 2-EG only) ────────────
        ax_c = fig.add_subplot(bot_gs[0, 2])
        _style_ax(ax_c, t)
        if n_eg == 2 and X_pca.shape[0] >= 2:
            eg_a, eg_b = eg_names[0], eg_names[1]
            _mask_a = [i for i, eg_l in enumerate(eg_labels_pca) if eg_l == eg_a]
            _mask_b = [i for i, eg_l in enumerate(eg_labels_pca) if eg_l == eg_b]
            _ok_volcano = (len(_mask_a) >= 2 and len(_mask_b) >= 2 and SCIPY_OK)

            if _ok_volcano:
                try:
                    from scipy.stats import ttest_ind as _ttest
                    _ann_v: list = []
                    # Collect per-pair stats
                    _pair_stats: list = []
                    for col_i, pname in enumerate(pair_names_pca):
                        _va = X_pca[_mask_a, col_i]
                        _vb = X_pca[_mask_b, col_i]
                        _delta_v = float(np.mean(_va) - np.mean(_vb))
                        try:
                            _, _pv = _ttest(_va, _vb, equal_var=False)
                        except Exception:
                            _pv = 1.0
                        _pv = max(float(_pv), 1e-300)
                        _pair_stats.append((pname, _delta_v, _pv))

                    _all_deltas = [abs(s[1]) for s in _pair_stats]
                    _top_thresh = (sorted(_all_deltas, reverse=True)[top_n - 1]
                                   if len(_all_deltas) >= top_n else 0.0)

                    ax_c.axhline(-np.log10(0.05), ls="--", color=t["muted"],
                                 lw=0.8, alpha=0.7)
                    ax_c.axvline(0, ls="--", color=t["muted"], lw=0.8, alpha=0.7)

                    for pname, dv, pv in _pair_stats:
                        _y_v  = -np.log10(pv)
                        _sig  = pv < 0.05
                        _top  = abs(dv) >= _top_thresh and _sig
                        # Determine source cluster pair from "Ci→Cj" string
                        _parts = pname.split("→")
                        try:
                            _ci_int = int(_parts[0][1:])
                        except Exception:
                            _ci_int = -1
                        _in_grp = _ci_int in cid_to_grp

                        if _top:
                            _col_v = "#FF4081"; _mk_v = "o"; _sz_v = 80; _zv = 5
                        elif _in_grp and _sig:
                            _col_v = cid_to_grp[_ci_int][1]; _mk_v = "D"
                            _sz_v = 55; _zv = 4
                        elif _in_grp:
                            _col_v = cid_to_grp[_ci_int][1]; _mk_v = "D"
                            _sz_v = 35; _zv = 3
                        elif _sig:
                            _col_v = "#FFC107"; _mk_v = "o"; _sz_v = 45; _zv = 4
                        else:
                            _col_v = t["muted"]; _mk_v = "o"; _sz_v = 20; _zv = 2

                        ax_c.scatter(dv, _y_v, color=_col_v, marker=_mk_v,
                                     s=_sz_v, zorder=_zv,
                                     edgecolors="none" if not _sig else t["spine"],
                                     linewidths=0.4)
                        if _top:
                            _ann_v.append((dv, _y_v, pname))

                    for _ax_v, _ay_v, _alb in _ann_v:
                        ax_c.annotate(
                            _alb, (_ax_v, _ay_v),
                            xytext=(6, 6), textcoords="offset points",
                            fontsize=6, color=t["tick"],
                            arrowprops=dict(arrowstyle="-", color=t["muted"],
                                            lw=0.5))

                    ax_c.set_xlabel(f"ΔT  ({eg_a} − {eg_b})",
                                    color=t["tick"], fontsize=8)
                    ax_c.set_ylabel("−log₁₀(p)", color=t["tick"], fontsize=8)
                    ax_c.set_title(
                        "Transition Volcano\n(Welch t-test per pair)",
                        color=t["tick"], fontsize=9, fontweight="bold")
                except Exception as _exc_v:
                    ax_c.text(0.5, 0.5, f"Volcano error:\n{_exc_v}",
                              ha="center", va="center",
                              color=t["muted"], transform=ax_c.transAxes, fontsize=8)
            else:
                ax_c.text(0.5, 0.5,
                          "Need ≥2 animals per group\nand scipy for Volcano.",
                          ha="center", va="center",
                          color=t["muted"], transform=ax_c.transAxes, fontsize=9)
                ax_c.set_title("Transition Volcano",
                               color=t["tick"], fontsize=9, fontweight="bold")
        else:
            ax_c.text(0.5, 0.5,
                      "Volcano only available\nfor exactly 2 experimental groups.",
                      ha="center", va="center",
                      color=t["muted"], transform=ax_c.transAxes, fontsize=9)
            ax_c.set_title("Transition Volcano",
                           color=t["tick"], fontsize=9, fontweight="bold")

        # Behaviour-group legend (shared, attached to Panel B)
        if cid_to_grp:
            _seen_grp: dict = {}
            for gname, gclr in cid_to_grp.values():
                _seen_grp[gname] = gclr
            _grp_patches = [mpatches.Patch(color=c, label=g)
                            for g, c in _seen_grp.items()]
            ax_b.legend(handles=_grp_patches, fontsize=7, framealpha=0.3,
                        facecolor=t["ax_bg"], labelcolor=t["tick"],
                        title="Behaviour groups", title_fontsize=6,
                        loc="upper right")

        fig.suptitle(
            "Group Transition Comparison  —  average T(i→j) per experimental group",
            color=t["tick"], fontweight="bold", fontsize=11)

    return fig


def build_cluster_stats_figure(recluster_result: dict, animal_data: list,
                                groups: dict = None,
                                t: dict = None) -> plt.Figure:
    """
    Per-cluster statistics overview: one row per metric.
    Each cluster is a point; colour = cluster ID (cycle through PALETTE), or
    behaviour-group colour when groups is supplied.
    Panels:
      - Mean bout duration (ms)
      - Total duration (s) per animal mean
      - Frequency (bouts) per animal mean
    Filtered clusters shown filled; filtered-out clusters shown as hollow.
    """
    if t is None:
        t = T()

    filtered_cids = set(recluster_result.get("filtered_cluster_ids", []))
    all_raw       = recluster_result.get("all_raw_cluster_ids",
                                         recluster_result["cluster_ids"])

    # aggregate per-cluster across all animals
    agg: dict = {cid: {"mean_bouts": [], "totals": [], "freqs": []}
                 for cid in all_raw}
    for ani in animal_data:
        pcm = compute_per_cluster_metrics(ani["df"], ani["fps"])
        for cid in all_raw:
            if cid in pcm.index:
                agg[cid]["mean_bouts"].append(float(pcm.loc[cid, "mean_bout"]) * 1000)
                agg[cid]["totals"].append(float(pcm.loc[cid, "total_duration"]))
                agg[cid]["freqs"].append(float(pcm.loc[cid, "frequency"]))

    cids_sorted = sorted(all_raw)
    labels_plot = [f"C{c}" for c in cids_sorted]
    cid_to_grp  = _cluster_group_map(groups)

    def _bar_color(cid):
        if cid in cid_to_grp:
            return cid_to_grp[cid][1]
        return PALETTE[cid % len(PALETTE)]

    metrics_plot = [
        ("Mean Bout Duration (ms)", "mean_bouts"),
        ("Total Duration (s)", "totals"),
        ("Frequency (# bouts)", "freqs"),
    ]

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(
            len(metrics_plot), 1,
            figsize=(max(14, len(all_raw) * 0.35 + 2), 4.5 * len(metrics_plot)),
            facecolor=t["fig_bg"],
            sharex=True,
        )
        x_pos = np.arange(len(cids_sorted))
        bar_w = 0.75

        for ai, (ylabel, mkey) in enumerate(metrics_plot):
            ax = axes[ai]
            vals  = [np.mean(agg[c][mkey]) if agg[c][mkey] else 0.0
                     for c in cids_sorted]
            sems  = [np.std(agg[c][mkey], ddof=1) / np.sqrt(len(agg[c][mkey]))
                     if len(agg[c][mkey]) > 1 else 0.0
                     for c in cids_sorted]
            alphas= [0.9 if c in filtered_cids else 0.25 for c in cids_sorted]
            edgec = [_bar_color(c) for c in cids_sorted]
            fills = [ec if c in filtered_cids else "none" for c, ec in
                     zip(cids_sorted, edgec)]

            for xi, (v, s, fc, ec, al) in enumerate(zip(vals, sems, fills, edgec, alphas)):
                ax.bar(xi, v, width=bar_w,
                       color=fc if fc != "none" else "none",
                       edgecolor=ec, linewidth=1.2, alpha=al, zorder=2)
                ax.errorbar(xi, v, yerr=s, fmt="none",
                            color=t["tick"], lw=1.0, capsize=2, zorder=3)

            ax.set_ylabel(ylabel, color=t["tick"], fontsize=9)
            _style_ax(ax, t)

        axes[-1].set_xticks(x_pos)
        tick_lbl_objs = axes[-1].set_xticklabels(
            labels_plot, rotation=60, ha="right",
            fontsize=max(5, 8 - len(all_raw) // 15))
        for lbl, cid in zip(tick_lbl_objs, cids_sorted):
            lbl.set_color(_bar_color(cid))

        # Build legend: filter markers + group colours (when available)
        legend_handles = [
            mpatches.Patch(color=PALETTE[0], alpha=0.9, label="Included in reclustering"),
            mpatches.Patch(color=PALETTE[0], alpha=0.25, label="Filtered out (below threshold)",
                           fill=False, edgecolor=PALETTE[0], linewidth=1.2),
        ]
        if cid_to_grp:
            seen_grp: dict = {}
            for gname, gcolor in cid_to_grp.values():
                seen_grp[gname] = gcolor
            legend_handles += [mpatches.Patch(color=c, label=g)
                               for g, c in seen_grp.items()]
        axes[0].legend(handles=legend_handles, fontsize=8, framealpha=0.3,
                       facecolor=t["ax_bg"], labelcolor=t["tick"])
        fig.suptitle(
            "Cluster Statistics Overview\n"
            "Hollow bars = clusters removed by duration/frequency filter",
            color=t["tick"], fontweight="bold", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def extract_reclustered_groups(recluster_result: dict, k: int) -> dict:
    """
    Convert a chosen k into a behaviour-groups dict compatible with the
    Group Editor and compute_combined / Combined Analysis.

    Returns:
        { "RG-1": {"labels": [cid, ...], "color": hex},
          "RG-2": {...}, ... }

    Cluster IDs in each merged group correspond directly to B-SOiD label
    integers and can be used as-is in compute_metrics / compute_combined.
    """
    lnk  = recluster_result["linkage_matrix"]
    cids = recluster_result["cluster_ids"]
    labels = fcluster(lnk, t=k, criterion="maxclust")
    groups = {}
    for ci, cid in zip(labels, cids):
        key = f"RG-{int(ci)}"
        if key not in groups:
            groups[key] = {"labels": [], "color": PALETTE[(int(ci) - 1) % len(PALETTE)]}
        groups[key]["labels"].append(int(cid))
    return dict(sorted(groups.items()))   # stable sort RG-1, RG-2, ...


#
# STANDARD PLOTTING

#


def _cluster_group_map(groups: dict) -> dict:
    """Return {cluster_id (int): (group_name, group_color)} from user-defined groups."""
    out: dict = {}
    for gname, gdata in (groups or {}).items():
        color = gdata.get("color", "#888888")
        for cid in gdata.get("labels", []):
            out[int(cid)] = (gname, color)
    return out


def _style_ax(ax, t: dict):
    ax.set_facecolor(t["ax_bg"])
    ax.tick_params(colors=t["tick"])
    for spine in ax.spines.values():
        spine.set_color(t["spine"])
    ax.xaxis.label.set_color(t["tick"])
    ax.yaxis.label.set_color(t["tick"])
    ax.title.set_color(t["tick"])


def plot_ethogram(ax, groups, metrics, session_s, t):
    gnames = list(groups.keys())
    ypos   = range(len(gnames) - 1, -1, -1)
    for y, gn in zip(ypos, gnames):
        color = groups[gn]["color"]
        ax.barh(y, session_s, left=0, height=0.55,
                color=t["track_bg"], alpha=0.6, zorder=1)
        for ev in metrics.get(gn, {}).get("events", []):
            ax.barh(y, ev["dur_s"], left=ev["start_s"],
                    height=0.55, color=color, alpha=0.92, zorder=2)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(gnames, color=t["tick"])
    ax.set_xlabel("Time (s)", color=t["tick"])
    ax.set_title("Ethogram", color=t["tick"], fontweight="bold")
    ax.set_xlim(0, session_s)
    _style_ax(ax, t)


def plot_time_bar(ax, groups, metrics, t):
    gnames = [g for g in groups if metrics.get(g, {}).get("total_duration", 0) > 0]
    durs   = [metrics[g]["total_duration"] for g in gnames]
    colors = [groups[g]["color"] for g in gnames]
    bars   = ax.barh(gnames, durs, color=colors, edgecolor="none")
    total  = sum(durs) or 1
    for bar, dur in zip(bars, durs):
        ax.text(bar.get_width() + total*0.01,
                bar.get_y() + bar.get_height()/2,
                f"{100*dur/total:.1f}%", va="center",
                color=t["tick"], fontsize=9)
    ax.set_xlabel("Total Time (s)", color=t["tick"])
    ax.set_title("Time per Behaviour", color=t["tick"], fontweight="bold")
    _style_ax(ax, t)


def plot_latency(ax, groups, metrics, t):
    gnames, lats, colors = [], [], []
    for gn, gi in groups.items():
        lat = metrics.get(gn, {}).get("latency")
        if lat is not None:
            gnames.append(gn); lats.append(lat); colors.append(gi["color"])
    if not gnames:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                color=t["tick"], transform=ax.transAxes)
        _style_ax(ax, t)
        return
    bars = ax.bar(gnames, lats, color=colors, edgecolor="none")
    for bar, lat in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{lat:.1f}s", ha="center", color=t["tick"], fontsize=9)
    ax.set_ylabel("Latency (s)", color=t["tick"])
    ax.set_title("Time to First Occurrence", color=t["tick"], fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    _style_ax(ax, t)


def plot_frequency(ax, groups, metrics, t):
    gnames = list(groups.keys())
    freqs  = [metrics.get(g, {}).get("frequency", 0) for g in gnames]
    colors = [groups[g]["color"] for g in gnames]
    bars   = ax.bar(gnames, freqs, color=colors, edgecolor="none")
    for bar, f in zip(bars, freqs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.2,
                str(f), ha="center", color=t["tick"], fontsize=9)
    ax.set_ylabel("Frequency (# bouts)", color=t["tick"])
    ax.set_title("Bout Frequency", color=t["tick"], fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    _style_ax(ax, t)


def plot_mean_bout(ax, groups, metrics, t):
    gnames = list(groups.keys())
    means  = [metrics.get(g, {}).get("mean_bout", 0) for g in gnames]
    colors = [groups[g]["color"] for g in gnames]
    ax.bar(gnames, means, color=colors, edgecolor="none")
    ax.set_ylabel("Mean Bout Duration (s)", color=t["tick"])
    ax.set_title("Mean Bout Duration", color=t["tick"], fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    _style_ax(ax, t)


def build_single_figure(groups, metrics, session_s, t=None) -> plt.Figure:
    if t is None:
        t = T()
    with plt.style.context(t["mpl_style"]):
        fig = plt.figure(figsize=(18, 11), facecolor=t["fig_bg"])
        gs  = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.4,
                               left=0.08, right=0.97, top=0.93, bottom=0.08)
        ax_eth  = fig.add_subplot(gs[0, :])
        ax_time = fig.add_subplot(gs[1, 0])
        ax_lat  = fig.add_subplot(gs[1, 1])
        ax_freq = fig.add_subplot(gs[2, 0])
        ax_mean = fig.add_subplot(gs[2, 1])
        plot_ethogram( ax_eth,  groups, metrics, session_s, t)
        plot_time_bar( ax_time, groups, metrics, t)
        plot_latency(  ax_lat,  groups, metrics, t)
        plot_frequency(ax_freq, groups, metrics, t)
        plot_mean_bout(ax_mean, groups, metrics, t)
        fig.suptitle("CUBE Behavioral Analysis",
                     color=t["tick"], fontsize=16, fontweight="bold", y=0.98)
    return fig


def build_preview_figure(groups, metrics, session_s, t=None) -> plt.Figure:
    if t is None:
        t = T()
    h = max(2.0, len(groups) * 0.6 + 0.8)
    with plt.style.context(t["mpl_style"]):
        fig, ax = plt.subplots(figsize=(8, h), facecolor=t["fig_bg"])
        plot_ethogram(ax, groups, metrics, session_s, t)
        fig.tight_layout(pad=0.5)
    return fig


def compute_combined_pvals(combined: dict, groups: dict) -> dict:
    """
    Compute KW p-values for every (metric_key, beh_group) pair, then apply
    Benjamini-Hochberg FDR correction across all tests simultaneously.
    Returns {(metric_key, beh_group): {"pval": float|None, "qval": float|None}}.
    """
    if not SCIPY_OK:
        return {}
    records    = combined["records"]
    uid_to_idx = combined["uid_to_idx"]
    exp_groups = combined["exp_groups"]
    eg_names   = list(exp_groups.keys())
    raw_pvals: dict = {}
    for metric_key in ("total_duration", "frequency", "latency", "mean_bout"):
        for bg in groups:
            if len(eg_names) < 2:
                raw_pvals[(metric_key, bg)] = None
                continue
            group_vals = [
                np.array([records[uid_to_idx[uid]]["metrics"]
                          .get(bg, {}).get(metric_key) or 0.0
                          for uid in exp_groups[eg]], dtype=float)
                for eg in eg_names
            ]
            try:
                _, pv = sp_stats.kruskal(*group_vals)
            except Exception:
                try:
                    _, pv = sp_stats.f_oneway(*group_vals)
                except Exception:
                    pv = None
            raw_pvals[(metric_key, bg)] = pv

    # BH FDR correction across the full family of tests
    keys  = list(raw_pvals.keys())
    p_arr = np.array([raw_pvals[k] if raw_pvals[k] is not None else 1.0 for k in keys])
    q_arr = benjamini_hochberg(p_arr)
    return {k: {"pval": raw_pvals[k],
                "qval": float(q_arr[i]) if raw_pvals[k] is not None else None}
            for i, k in enumerate(keys)}


def build_per_group_transition_figure(
        recluster_result: dict, animal_data: list, t: dict = None) -> plt.Figure:
    """
    One transition probability heatmap per experimental group so users can
    directly compare how behaviour-cluster sequences differ between conditions.
    """
    if t is None:
        t = T()
    if recluster_result is None:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No animals loaded.", ha="center", va="center",
                color=t["tick"], transform=ax.transAxes)
        _style_ax(ax, t)
        return fig

    cids_all = recluster_result["cluster_ids"]
    lnk      = recluster_result.get("linkage_matrix")

    if lnk is not None and len(cids_all) >= 2:
        dend  = dendrogram(lnk, no_plot=True)
        order = dend["leaves"]
    else:
        order = list(range(len(cids_all)))

    cids_ordered = [cids_all[i] for i in order]
    labels_ord   = [f"C{c}" for c in cids_ordered]
    n            = len(cids_ordered)

    groups_eg: dict = {}
    for ani in animal_data:
        groups_eg.setdefault(ani["exp_group"], []).append(ani)
    eg_names = list(groups_eg.keys())
    n_eg     = len(eg_names)

    if n_eg == 0:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No animals loaded.", ha="center", va="center",
                color=t["tick"], transform=ax.transAxes)
        _style_ax(ax, t)
        return fig

    ncols     = min(n_eg, 3)
    nrows     = int(np.ceil(n_eg / ncols))
    cell_size = max(3.5, n * 0.25 + 1.5)
    is_raw    = recluster_result.get("_is_raw_preview", False)

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * cell_size + 0.5, nrows * cell_size + 1.2),
            facecolor=t["fig_bg"], squeeze=False,
        )
        axes_flat = axes.flatten()
        tick_step = max(1, n // 15)
        tick_pos  = list(range(0, n, tick_step))
        tick_lbls = [labels_ord[i] for i in tick_pos]
        fsize     = max(4, 7 - n // 20)

        for ei, eg in enumerate(eg_names):
            ax_eg    = axes_flat[ei]
            ani_list = groups_eg[eg]

            # Build per-group averaged transition matrix aligned to all trans_cids
            all_eg_cids = sorted({int(l) for a in ani_list
                                  for l in a["df"]["label"].unique()})
            idx_eg  = {c: i for i, c in enumerate(all_eg_cids)}
            ng      = len(all_eg_cids)
            tsum    = np.zeros((ng, ng), dtype=float)
            cnt     = 0
            for ani in ani_list:
                mat, ani_cids = compute_transition_matrix(ani["df"])
                for ri, ci in enumerate(ani_cids):
                    for cj_i, cj in enumerate(ani_cids):
                        if ci in idx_eg and cj in idx_eg:
                            tsum[idx_eg[ci], idx_eg[cj]] += mat[ri, cj_i]
                cnt += 1
            tmat_eg = tsum / max(cnt, 1)

            # Remap onto shared cids_ordered ordering
            disp = np.zeros((n, n), dtype=float)
            for ri, cid_r in enumerate(cids_ordered):
                for ci, cid_c in enumerate(cids_ordered):
                    if cid_r in idx_eg and cid_c in idx_eg:
                        disp[ri, ci] = tmat_eg[idx_eg[cid_r], idx_eg[cid_c]]
            np.fill_diagonal(disp, 0)
            # Row-normalise after remap (averaging + subsetting break row-sum=1)
            _d_rs = disp.sum(axis=1, keepdims=True); _d_rs[_d_rs == 0] = 1.0
            disp = disp / _d_rs
            # Mask below-chance entries so the colourscale spans only signal
            _cf = 1.0 / max(1, n - 1)
            disp_d = disp.astype(float)
            disp_d[disp_d <= _cf] = np.nan
            _cmap_m = plt.cm.magma.copy(); _cmap_m.set_bad(color=t["ax_bg"])

            eg_color = PALETTE[ei % len(PALETTE)]
            vmax     = float(np.nanmax(disp_d)) if not np.all(np.isnan(disp_d)) else 1.0
            im = ax_eg.imshow(disp_d, cmap=_cmap_m, aspect="auto",
                              interpolation="nearest", vmin=_cf, vmax=vmax)
            ax_eg.set_xticks(tick_pos)
            ax_eg.set_xticklabels(tick_lbls, rotation=70, fontsize=fsize,
                                   color=t["tick"])
            ax_eg.set_yticks(tick_pos)
            ax_eg.set_yticklabels(tick_lbls, fontsize=fsize, color=t["tick"])
            ax_eg.set_title(f"{eg}  (n = {len(ani_list)})",
                            color=eg_color, fontsize=9, fontweight="bold")
            cb = fig.colorbar(im, ax=ax_eg, shrink=0.75, pad=0.02)
            cb.ax.tick_params(colors=t["tick"], labelsize=7)
            cb.outline.set_edgecolor(t["spine"])
            _style_ax(ax_eg, t)

        for ei in range(n_eg, len(axes_flat)):
            axes_flat[ei].set_visible(False)

        note = "  [Original clusters — run Reclustering to update]" if is_raw else ""
        fig.suptitle(
            f"Transition Probabilities by Experimental Group{note}\n"
            "(above-chance only · diagonal = 0 · same cluster order as Transitions view)",
            color=t["tick"], fontweight="bold", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def build_per_group_network_figure(
        recluster_result: dict, animal_data: list, t: dict = None) -> plt.Figure:
    """
    One circular transition network per experimental group on a shared cluster
    layout so the groups can be compared directly.
    """
    if t is None:
        t = T()
    if recluster_result is None:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No animals loaded.", ha="center", va="center",
                color=t["tick"], transform=ax.transAxes)
        _style_ax(ax, t); return fig

    cids_all = recluster_result["cluster_ids"]
    lnk      = recluster_result.get("linkage_matrix")
    if lnk is not None and len(cids_all) >= 2:
        dend  = dendrogram(lnk, no_plot=True)
        order = dend["leaves"]
    else:
        order = list(range(len(cids_all)))

    cids_ordered = [cids_all[i] for i in order]
    max_net  = min(len(cids_ordered), 25)
    net_cids = cids_ordered[:max_net]
    nn       = len(net_cids)

    theta  = np.linspace(0, 2 * np.pi, nn, endpoint=False) - np.pi / 2
    nx_pos = np.cos(theta)
    ny_pos = np.sin(theta)
    node_colors = [PALETTE[c % len(PALETTE)] for c in net_cids]
    net_labels  = [f"C{c}" for c in net_cids]

    groups_eg: dict = {}
    for ani in animal_data:
        groups_eg.setdefault(ani["exp_group"], []).append(ani)
    eg_names = list(groups_eg.keys())
    n_eg     = len(eg_names)

    if n_eg == 0:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No animals loaded.", ha="center", va="center",
                color=t["tick"], transform=ax.transAxes)
        _style_ax(ax, t); return fig

    ncols = min(n_eg, 3)
    nrows = int(np.ceil(n_eg / ncols))
    cell  = max(5, nn * 0.28 + 3)

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * cell + 0.3, nrows * cell + 1.2),
            facecolor=t["fig_bg"], squeeze=False,
        )
        axes_flat = axes.flatten()

        for ei, eg in enumerate(eg_names):
            ax_eg    = axes_flat[ei]
            ani_list = groups_eg[eg]

            all_eg_cids = sorted({int(l) for a in ani_list
                                   for l in a["df"]["label"].unique()})
            idx_eg = {c: i for i, c in enumerate(all_eg_cids)}
            ng     = len(all_eg_cids)
            tsum   = np.zeros((ng, ng), dtype=float)
            cnt    = 0
            for ani in ani_list:
                mat, ani_cids = compute_transition_matrix(ani["df"])
                for ri, ci in enumerate(ani_cids):
                    for cj_i, cj in enumerate(ani_cids):
                        if ci in idx_eg and cj in idx_eg:
                            tsum[idx_eg[ci], idx_eg[cj]] += mat[ri, cj_i]
                cnt += 1
            tmat_eg = tsum / max(cnt, 1)

            # Remap to shared net_cids layout
            net_tmat = np.zeros((nn, nn), dtype=float)
            for ri, cid_r in enumerate(net_cids):
                for ci, cid_c in enumerate(net_cids):
                    if cid_r in idx_eg and cid_c in idx_eg:
                        net_tmat[ri, ci] = tmat_eg[idx_eg[cid_r], idx_eg[cid_c]]
            np.fill_diagonal(net_tmat, 0)
            # Row-normalise after remap: averaging and subsetting to net_cids means
            # rows may no longer sum to 1, invalidating the 1/(nn-1) chance floor.
            _net_rs = net_tmat.sum(axis=1, keepdims=True)
            _net_rs[_net_rs == 0] = 1.0
            net_tmat = net_tmat / _net_rs

            _draw_circle_network(ax_eg, net_tmat, nx_pos, ny_pos,
                                 net_labels, node_colors, t)
            eg_color = PALETTE[ei % len(PALETTE)]
            ax_eg.set_title(f"{eg}  (n = {len(ani_list)})",
                            color=eg_color, fontsize=9, fontweight="bold")

        for ei in range(n_eg, len(axes_flat)):
            axes_flat[ei].set_visible(False)

        is_raw = recluster_result.get("_is_raw_preview", False)
        note   = "  [Original clusters — run Reclustering to update]" if is_raw else ""
        fig.suptitle(
            f"Transition Networks by Experimental Group{note}\n"
            "Shared cluster layout · Arrow width = conditional transition probability  "
            "(above-chance edges only)",
            color=t["tick"], fontweight="bold", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def build_group_aggregate_network_figure(
        animal_data: list, groups: dict, t: dict = None) -> plt.Figure:
    """
    Circular network where each NODE is a user-defined behaviour group.
    Edge weight = mean transition probability from clusters in group_i to
    clusters in group_j, shown separately for each experimental group.
    """
    if t is None:
        t = T()
    if not groups:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No behaviour groups defined.\n"
                "Create groups in the Group Editor first.",
                ha="center", va="center", color=t["tick"],
                transform=ax.transAxes, fontsize=9)
        _style_ax(ax, t); return fig

    group_names  = list(groups.keys())
    group_colors = [groups[g].get("color", PALETTE[i % len(PALETTE)])
                    for i, g in enumerate(group_names)]
    ng = len(group_names)

    # cluster-id → group index
    cid_to_gi: dict = {}
    for gi, gname in enumerate(group_names):
        for cid in groups[gname].get("labels", []):
            cid_to_gi[int(cid)] = gi

    def _build_group_tmat(ani_list: list) -> np.ndarray:
        gmat = np.zeros((ng, ng), dtype=float)
        gcnt = np.zeros((ng, ng), dtype=float)
        for ani in ani_list:
            mat, ani_cids = compute_transition_matrix(ani["df"])
            for ri, ci in enumerate(ani_cids):
                if int(ci) not in cid_to_gi:
                    continue
                gi_r = cid_to_gi[int(ci)]
                for cj_i, cj in enumerate(ani_cids):
                    if int(cj) not in cid_to_gi:
                        continue
                    gi_c = cid_to_gi[int(cj)]
                    if gi_r != gi_c:   # exclude self-transitions between same group
                        gmat[gi_r, gi_c] += mat[ri, cj_i]
                        gcnt[gi_r, gi_c] += 1.0
        safe = np.where(gcnt > 0, gcnt, 1.0)
        return gmat / safe

    groups_eg: dict = {}
    for ani in animal_data:
        groups_eg.setdefault(ani["exp_group"], []).append(ani)
    eg_names = list(groups_eg.keys())
    n_eg     = len(eg_names)

    panels = ([(eg, groups_eg[eg]) for eg in eg_names]
              if n_eg > 1 else [("All animals", animal_data)])

    theta  = np.linspace(0, 2 * np.pi, ng, endpoint=False) - np.pi / 2
    nx_pos = np.cos(theta)
    ny_pos = np.sin(theta)

    ncols = min(len(panels), 3)
    nrows = int(np.ceil(len(panels) / ncols))
    cell  = max(5, ng * 0.6 + 3)

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * cell + 0.3, nrows * cell + 1.4),
            facecolor=t["fig_bg"], squeeze=False,
        )
        axes_flat = axes.flatten()

        for ei, (label, ani_list) in enumerate(panels):
            if ei >= len(axes_flat):
                break
            ax_eg = axes_flat[ei]
            gmat  = _build_group_tmat(ani_list)
            # Row-normalise: values from _build_group_tmat are averaged cluster-pair
            # probabilities whose rows don't sum to 1.  Normalising makes them
            # "conditional probabilities given leaving the group", which is the correct
            # interpretation for chance_floor = 1/(ng-1) inside _draw_circle_network.
            _rs = gmat.sum(axis=1, keepdims=True)
            _rs[_rs == 0] = 1.0
            gmat = gmat / _rs
            _draw_circle_network(ax_eg, gmat, nx_pos, ny_pos,
                                 group_names, group_colors, t)
            eg_color = PALETTE[ei % len(PALETTE)]
            ax_eg.set_title(label, color=eg_color, fontsize=10, fontweight="bold")

        for ei in range(len(panels), len(axes_flat)):
            axes_flat[ei].set_visible(False)

        fig.suptitle(
            "Behaviour Group Transition Network\n"
            "Each node = one user-defined behaviour group  ·  "
            "Arrow width = conditional transition probability  (above-chance edges only)",
            color=t["tick"], fontweight="bold", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.91])
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  BEHAVIORAL EXPLORER  —  Group-comparison figure builders
# ──────────────────────────────────────────────────────────────────────────────

def build_diff_heatmap_figure(
        recluster_result: dict, animal_data: list,
        ctrl_group: str = None, t: dict = None) -> plt.Figure:
    """
    Difference heatmaps for all experimental-group pairs.
    For each (A, B) pair: Δ = P_B − P_A transition matrix.
    Warm colours = B transitions more; cool = A transitions more.
    Ordered by the dendrogram from reclustering for consistent axes.
    """
    if t is None:
        t = T()

    # Gather experimental groups
    groups_eg: dict = {}
    for ani in animal_data:
        groups_eg.setdefault(ani["exp_group"], []).append(ani)
    eg_names = list(groups_eg.keys())
    n_eg     = len(eg_names)

    if n_eg < 2:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5,
                "Need ≥ 2 experimental groups for difference heatmaps.",
                ha="center", va="center", color=t["tick"],
                transform=ax.transAxes)
        _style_ax(ax, t)
        return fig

    if recluster_result is None:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No animals loaded.", ha="center", va="center",
                color=t["tick"], transform=ax.transAxes)
        _style_ax(ax, t); return fig

    cids_all = recluster_result["cluster_ids"]
    lnk      = recluster_result.get("linkage_matrix")
    if lnk is not None and len(cids_all) >= 2:
        from scipy.cluster.hierarchy import dendrogram as _dend
        order = _dend(lnk, no_plot=True)["leaves"]
    else:
        order = list(range(len(cids_all)))
    cids_ord  = [cids_all[i] for i in order]
    labels    = [f"C{c}" for c in cids_ord]
    n         = len(cids_ord)

    def _group_tmat(ani_list):
        """Row-normalised conditional transition matrix averaged over *ani_list*."""
        idx_eg = {c: i for i, c in enumerate(cids_ord)}
        tsum   = np.zeros((n, n), dtype=float)
        cnt    = 0
        for ani in ani_list:
            mat, ani_cids = compute_transition_matrix(ani["df"])
            for ri, ci in enumerate(ani_cids):
                for cj_i, cj in enumerate(ani_cids):
                    if ci in idx_eg and cj in idx_eg:
                        tsum[idx_eg[ci], idx_eg[cj]] += mat[ri, cj_i]
            cnt += 1
        tmat = tsum / max(cnt, 1)
        np.fill_diagonal(tmat, 0)
        _rs = tmat.sum(axis=1, keepdims=True); _rs[_rs == 0] = 1.0
        return tmat / _rs

    tmats = {eg: _group_tmat(groups_eg[eg]) for eg in eg_names}

    # Determine reference group (ctrl_group or first)
    ref_eg = ctrl_group if (ctrl_group and ctrl_group in tmats) else eg_names[0]

    # Build pair list: (ctrl, other) for every other group
    pairs   = [(ref_eg, eg) for eg in eg_names if eg != ref_eg]
    n_pairs = len(pairs)

    # Layout: n_pairs rows × 3 columns (ctrl | exp | diff)
    cell    = max(3.5, n * 0.22 + 1.5)
    fsize   = max(4, 7 - n // 20)
    tick_step = max(1, n // 15)
    tick_pos  = list(range(0, n, tick_step))
    tick_lbl  = [labels[i] for i in tick_pos]

    _cf_d  = 1.0 / max(1, n - 1)   # chance floor for absolute panels
    _cmap_d = plt.cm.magma.copy(); _cmap_d.set_bad(color=t["ax_bg"])

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(
            n_pairs, 3,
            figsize=(cell * 3 + 1.0, cell * n_pairs + 1.2),
            facecolor=t["fig_bg"], squeeze=False)

        for row, (eg_a, eg_b) in enumerate(pairs):
            col_a = PALETTE[eg_names.index(eg_a) % len(PALETTE)]
            col_b = PALETTE[eg_names.index(eg_b) % len(PALETTE)]

            for col_i, (mat, title, cmap_name, vmin, vmax, cblabel) in enumerate([
                (tmats[eg_a], f"{eg_a}  (reference)",
                 "magma", _cf_d, None, "P(transition | leaving cluster)"),
                (tmats[eg_b], f"{eg_b}",
                 "magma", _cf_d, None, "P(transition | leaving cluster)"),
                (tmats[eg_b] - tmats[eg_a], f"Δ = {eg_b} − {eg_a}",
                 "RdBu_r", None, None, "ΔP"),
            ]):
                ax = axes[row, col_i]
                if col_i < 2:
                    # Absolute panels: mask below-chance, anchor colourscale at chance
                    mat_d = mat.astype(float)
                    mat_d[mat_d <= _cf_d] = np.nan
                    vmax_use = float(np.nanmax(mat_d)) if not np.all(np.isnan(mat_d)) else 1.0
                    im = ax.imshow(mat_d, cmap=_cmap_d, aspect="auto",
                                   interpolation="nearest",
                                   vmin=_cf_d, vmax=vmax_use)
                else:
                    # Difference panel: symmetric around 0, no chance masking
                    vmax_use = float(np.abs(mat).max()) if np.abs(mat).max() > 0 else 1.0
                    im = ax.imshow(mat, cmap=cmap_name, aspect="auto",
                                   interpolation="nearest",
                                   vmin=-vmax_use, vmax=vmax_use)
                ax.set_xticks(tick_pos)
                ax.set_xticklabels(tick_lbl, rotation=70,
                                   fontsize=fsize, color=t["tick"])
                ax.set_yticks(tick_pos)
                ax.set_yticklabels(tick_lbl, fontsize=fsize, color=t["tick"])
                tc = col_a if col_i == 0 else (col_b if col_i == 1 else t["tick"])
                ax.set_title(title, color=tc, fontsize=9, fontweight="bold")
                cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
                cb.ax.tick_params(colors=t["tick"], labelsize=7)
                cb.set_label(cblabel, color=t["tick"], fontsize=7)
                cb.outline.set_edgecolor(t["spine"])
                _style_ax(ax, t)

    fig.suptitle(
        "Transition Difference Heatmaps  (reference group subtracted)\n"
        "Red = experimental group transitions MORE  ·  Blue = transitions LESS",
        color=t["tick"], fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def build_dwell_violin_figure(
        animal_data: list, t: dict = None) -> plt.Figure:
    """
    Per-group dwell-time violin plots: one panel per experimental group,
    one violin per behavioral state.  Dwell time = bout duration in seconds.
    """
    if t is None:
        t = T()

    groups_eg: dict = {}
    for ani in animal_data:
        groups_eg.setdefault(ani["exp_group"], []).append(ani)
    eg_names = list(groups_eg.keys())
    n_eg     = len(eg_names)

    if n_eg == 0:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No animals loaded.", ha="center", va="center",
                color=t["tick"], transform=ax.transAxes)
        _style_ax(ax, t); return fig

    # Collect all cluster IDs across all animals
    all_cids = sorted({int(l) for ani in animal_data
                       for l in ani["df"]["label"].unique()})
    n_cids   = len(all_cids)
    if n_cids == 0:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No labels found.", ha="center", va="center",
                color=t["tick"], transform=ax.transAxes)
        _style_ax(ax, t); return fig

    ncols = min(n_eg, 3)
    nrows = int(np.ceil(n_eg / ncols))
    cell_w = max(4, n_cids * 0.6 + 2)

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * cell_w, nrows * 5.5 + 1.0),
            facecolor=t["fig_bg"], squeeze=False)

        for ei, eg in enumerate(eg_names):
            r, c    = divmod(ei, ncols)
            ax      = axes[r][c]
            _style_ax(ax, t)
            ani_list = groups_eg[eg]
            eg_color = PALETTE[ei % len(PALETTE)]

            data   = []
            labels = []
            colors = []
            for cid in all_cids:
                bouts_all = []
                for ani in ani_list:
                    fps_a = float(ani.get("fps", 30))
                    sub   = ani["df"][ani["df"]["label"] == cid]
                    # Run lengths → seconds (column is "run_len" after load_csv rename)
                    if "run_len" in sub.columns:
                        bouts_all.extend((sub["run_len"].values / fps_a).tolist())
                    elif "Run lengths" in sub.columns:
                        bouts_all.extend((sub["Run lengths"].values / fps_a).tolist())
                    elif "duration_sec" in sub.columns:
                        bouts_all.extend(sub["duration_sec"].values.tolist())
                data.append(np.array(bouts_all, dtype=float))
                labels.append(f"C{cid}")
                colors.append(PALETTE[cid % len(PALETTE)])

            non_empty = [i for i, d in enumerate(data) if len(d) > 0]
            if not non_empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        color=t["tick"], transform=ax.transAxes)
                ax.set_title(eg, color=eg_color, fontsize=9, fontweight="bold")
                continue

            data_ne   = [data[i]   for i in non_empty]
            labels_ne = [labels[i] for i in non_empty]
            colors_ne = [colors[i] for i in non_empty]

            parts = ax.violinplot(data_ne, positions=range(len(non_empty)),
                                  showmedians=True, showextrema=True)
            parts["cmedians"].set_color("#ffd60a")
            for sp in ("cmins", "cmaxes", "cbars"):
                parts[sp].set_color(t["spine"])
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors_ne[i])
                pc.set_edgecolor(t["ax_bg"])
                pc.set_alpha(0.72)

            rng = np.random.default_rng(42)
            for i, d in enumerate(data_ne):
                jit = rng.uniform(-0.12, 0.12, size=len(d))
                ax.scatter(i + jit, d,
                           color=colors_ne[i], alpha=0.3, s=5, linewidths=0,
                           zorder=3)

            ax.set_xticks(range(len(non_empty)))
            ax.set_xticklabels(labels_ne, color=t["tick"],
                               fontsize=max(5, 9 - n_cids // 8), rotation=45)
            ax.set_ylabel("Dwell time (s)", color=t["tick"], fontsize=9)
            ax.set_title(f"{eg}  (n = {len(ani_list)})",
                         color=eg_color, fontsize=9, fontweight="bold")

        for ei in range(n_eg, nrows * ncols):
            r, c = divmod(ei, ncols)
            axes[r][c].set_visible(False)

    fig.suptitle(
        "Dwell-time distributions by experimental group\n"
        "Each violin = distribution of bout durations for one state · "
        "Dots = individual bouts",
        color=t["tick"], fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def build_sankey_figure(
        animal_data: list, t: dict = None,
        n_steps: int = 5,
        anchor_cluster: int = None) -> plt.Figure:
    """
    Sankey (alluvial) flow diagram per experimental group.
    Each column is a consecutive bout position; ribbons show how the
    population flows from state to state across the sequence.

    anchor_cluster: cluster ID to re-zero each animal's sequence at its
    first occurrence of that cluster.  None = auto-detect as the source
    cluster of the strongest normalised transition across all data.
    Animals that never exhibit the anchor cluster are excluded per group.
    """
    if t is None:
        t = T()

    from matplotlib.path import Path as MPath

    groups_eg: dict = {}
    for ani in animal_data:
        groups_eg.setdefault(ani["exp_group"], []).append(ani)
    eg_names = list(groups_eg.keys())
    n_eg     = len(eg_names)

    if n_eg == 0:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No animals loaded.", ha="center", va="center",
                color=t["tick"], transform=ax.transAxes)
        _style_ax(ax, t); return fig

    all_cids = sorted({int(l) for ani in animal_data
                       for l in ani["df"]["label"].unique()})
    ns = len(all_cids)
    si = {c: i for i, c in enumerate(all_cids)}

    def _raw_seqs(ani_list):
        seqs = []
        for ani in ani_list:
            seq = ani["df"].sort_values("start_frame")["label"].values
            if len(seq) >= 2:
                seqs.append([int(x) for x in seq])
        return seqs

    def _auto_anchor(raw_seqs):
        trans = np.zeros((ns, ns), dtype=float)
        for seq in raw_seqs:
            for k in range(len(seq) - 1):
                if seq[k] in si and seq[k + 1] in si:
                    trans[si[seq[k]], si[seq[k + 1]]] += 1
        row_s = trans.sum(axis=1, keepdims=True)
        row_s[row_s == 0] = 1.0
        norm = trans / row_s
        np.fill_diagonal(norm, 0.0)
        a_idx = int(np.unravel_index(norm.argmax(), norm.shape)[0])
        return all_cids[a_idx]

    def _apply_anchor(seqs, anchor):
        out = []
        for seq in seqs:
            idx = next((i for i, lbl in enumerate(seq) if lbl == anchor), None)
            if idx is not None and len(seq) - idx >= 2:
                out.append(seq[idx:])
        return out

    def _build_flow(seqs, n_steps_use):
        counts = np.zeros((n_steps_use, ns), dtype=float)
        trans  = np.zeros((n_steps_use - 1, ns, ns), dtype=float)
        for seq in seqs:
            for k in range(min(n_steps_use, len(seq))):
                counts[k, si[seq[k]]] += 1
            for k in range(min(n_steps_use - 1, len(seq) - 1)):
                trans[k, si[seq[k]], si[seq[k + 1]]] += 1
        tot = counts.sum(axis=1, keepdims=True)
        tot[tot == 0] = 1.0
        props = counts / tot
        for k in range(n_steps_use - 1):
            rs = trans[k].sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            trans[k] /= rs
        return props, trans

    # Collect raw (unanchored) sequences for every group, then resolve anchor once
    raw_by_eg = {eg: _raw_seqs(groups_eg[eg]) for eg in eg_names}
    all_raw   = [s for seqs in raw_by_eg.values() for s in seqs]
    resolved_anchor = anchor_cluster if anchor_cluster is not None else _auto_anchor(all_raw)
    anchor_label    = f"C{resolved_anchor}"
    anchor_note     = "user-defined" if anchor_cluster is not None else "auto — strongest transition"

    ncols = min(n_eg, 2)
    nrows = int(np.ceil(n_eg / ncols))
    fw    = max(10, n_steps * 2.2) * ncols
    fh    = 7.5 * nrows

    colors_state = [PALETTE[c % len(PALETTE)] for c in all_cids]
    col_w  = 0.04
    bar_h  = 0.88
    gap    = 0.005

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fw, fh),
            facecolor=t["fig_bg"], squeeze=False)

        for ei, eg in enumerate(eg_names):
            r, c     = divmod(ei, ncols)
            ax       = axes[r][c]
            ax.set_facecolor(t["ax_bg"])
            ax.axis("off")
            eg_color = PALETTE[ei % len(PALETTE)]

            seqs       = _apply_anchor(raw_by_eg[eg], resolved_anchor)
            n_excluded = len(raw_by_eg[eg]) - len(seqs)
            if not seqs:
                ax.text(0.5, 0.5,
                        f"No data\n(anchor {anchor_label} never observed)",
                        ha="center", va="center",
                        color=t["tick"], transform=ax.transAxes)
                ax.set_title(f"{eg}", color=eg_color, fontsize=10,
                             fontweight="bold")
                continue

            n_use = min(n_steps, max(len(s) for s in seqs))
            if n_use < 2:
                ax.text(0.5, 0.5, "Sequences too short",
                        ha="center", va="center", color=t["tick"],
                        transform=ax.transAxes)
                continue

            props, trans = _build_flow(seqs, n_use)
            col_x_use    = np.linspace(0, 1, n_use)

            y_bot = np.zeros((n_use, ns), dtype=float)
            y_top = np.zeros((n_use, ns), dtype=float)
            for k in range(n_use):
                cum = 0.06
                for si_j in range(ns):
                    h = props[k, si_j] * bar_h
                    y_bot[k, si_j] = cum
                    y_top[k, si_j] = cum + h
                    cum += h + gap

            # Ribbons
            for k in range(n_use - 1):
                x0 = col_x_use[k] + col_w
                x1 = col_x_use[k + 1]
                cx0 = x0 + (x1 - x0) * 0.40
                cx1 = x0 + (x1 - x0) * 0.60
                src_off = np.zeros(ns, dtype=float)
                dst_off = np.zeros(ns, dtype=float)
                _sk_cf = 1.0 / max(1, ns - 1)
                for a in range(ns):
                    sh = y_top[k, a] - y_bot[k, a]
                    for b in range(ns):
                        p = float(trans[k, a, b])
                        if p <= _sk_cf:
                            continue
                        rhs = sh * p
                        rhd = (y_top[k+1, b] - y_bot[k+1, b]) * p
                        ys0 = y_bot[k, a]   + src_off[a]
                        ye0 = ys0 + rhs
                        ys1 = y_bot[k+1, b] + dst_off[b]
                        ye1 = ys1 + rhd
                        src_off[a] += rhs
                        dst_off[b] += rhd
                        verts = [
                            (x0, ys0), (cx0, ys0), (cx1, ys1), (x1, ys1),
                            (x1, ye1), (cx1, ye1), (cx0, ye0), (x0, ye0),
                            (x0, ys0),
                        ]
                        codes = ([MPath.MOVETO] + [MPath.CURVE4]*3 +
                                 [MPath.LINETO] + [MPath.CURVE4]*3 +
                                 [MPath.CLOSEPOLY])
                        patch = mpatches.PathPatch(
                            MPath(verts, codes),
                            facecolor=colors_state[a],
                            edgecolor="none", alpha=0.35, zorder=1)
                        ax.add_patch(patch)

            # Bars
            for k in range(n_use):
                for si_j in range(ns):
                    h = y_top[k, si_j] - y_bot[k, si_j]
                    if h < 1e-4:
                        continue
                    rect = mpatches.FancyBboxPatch(
                        (col_x_use[k], y_bot[k, si_j]), col_w, h,
                        boxstyle="square,pad=0",
                        facecolor=colors_state[si_j],
                        edgecolor=t["ax_bg"], linewidth=0.5, zorder=3,
                        picker=True,
                        gid=f"cluster:{all_cids[si_j]}")
                    ax.add_patch(rect)
                    if h > 0.05:
                        ax.text(col_x_use[k] + col_w/2,
                                y_bot[k, si_j] + h/2,
                                f"C{all_cids[si_j]}",
                                ha="center", va="center",
                                fontsize=max(4, 8 - ns // 6),
                                color=t["text"], fontweight="bold", zorder=4)

            for k, x in enumerate(col_x_use):
                ax.text(x + col_w/2, 0.01, f"Step {k+1}",
                        ha="center", va="bottom", fontsize=8,
                        color=t["tick"], fontweight="bold")

            excl_note = (f"  ({n_excluded} excl. — no {anchor_label})"
                         if n_excluded else "")
            ax.set_xlim(-0.04, 1.06)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{eg}  (n = {len(seqs)}{excl_note})",
                         color=eg_color, fontsize=10, fontweight="bold",
                         pad=4)

        for ei in range(n_eg, nrows * ncols):
            r, c = divmod(ei, ncols)
            axes[r][c].set_visible(False)

    fig.suptitle(
        f"Behavioral Sequence Sankey  (first {n_steps} bout positions)\n"
        f"Anchor: {anchor_label}  [{anchor_note}]  ·  "
        "Bar height = state occupancy  ·  Ribbon width = transition flow  ·  "
        "Click a bar to re-anchor",
        color=t["tick"], fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  BEHAVIORAL EXPLORER  —  Behaviour-group-level figure builders
#  (Parallel to the cluster-level builders above but nodes = user groups)
# ──────────────────────────────────────────────────────────────────────────────

def build_dwell_violin_beh_groups_figure(
        animal_data: list, groups: dict, t: dict = None) -> plt.Figure:
    """
    Dwell-time violin plots using user-defined behavioural groups.
    One panel per experimental group; one violin per behavioural group
    aggregating bouts from all constituent clusters.
    """
    if t is None:
        t = T()
    if not groups:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No behaviour groups defined.\n"
                "Create groups in the Group Editor first.",
                ha="center", va="center", color=t["tick"],
                transform=ax.transAxes, fontsize=9)
        _style_ax(ax, t); return fig

    groups_eg: dict = {}
    for ani in animal_data:
        groups_eg.setdefault(ani["exp_group"], []).append(ani)
    eg_names = list(groups_eg.keys())
    n_eg     = len(eg_names)

    if n_eg == 0:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No animals loaded.", ha="center", va="center",
                color=t["tick"], transform=ax.transAxes)
        _style_ax(ax, t); return fig

    group_names  = list(groups.keys())
    group_colors = [groups[g].get("color", PALETTE[i % len(PALETTE)])
                    for i, g in enumerate(group_names)]
    ng           = len(group_names)

    ncols  = min(n_eg, 3)
    nrows  = int(np.ceil(n_eg / ncols))
    cell_w = max(4, ng * 0.7 + 2)

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * cell_w, nrows * 5.5 + 1.0),
            facecolor=t["fig_bg"], squeeze=False)

        for ei, eg in enumerate(eg_names):
            r, c     = divmod(ei, ncols)
            ax       = axes[r][c]
            _style_ax(ax, t)
            ani_list = groups_eg[eg]
            eg_color = PALETTE[ei % len(PALETTE)]

            data = []
            for gi, gname in enumerate(group_names):
                cids      = {int(c) for c in groups[gname].get("labels", [])}
                bouts_all = []
                for ani in ani_list:
                    fps_a = float(ani.get("fps", 30))
                    sub   = ani["df"][ani["df"]["label"].isin(cids)]
                    if "run_len" in sub.columns and len(sub):
                        bouts_all.extend(
                            (sub["run_len"].values / fps_a).tolist())
                    elif "Run lengths" in sub.columns and len(sub):
                        bouts_all.extend(
                            (sub["Run lengths"].values / fps_a).tolist())
                data.append(np.array(bouts_all, dtype=float))

            non_empty = [i for i, d in enumerate(data) if len(d) > 0]
            if not non_empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        color=t["tick"], transform=ax.transAxes)
                ax.set_title(f"{eg}  (n = {len(ani_list)})",
                             color=eg_color, fontsize=9, fontweight="bold")
                continue

            data_ne   = [data[i]        for i in non_empty]
            labels_ne = [group_names[i] for i in non_empty]
            colors_ne = [group_colors[i] for i in non_empty]

            parts = ax.violinplot(data_ne, positions=range(len(non_empty)),
                                  showmedians=True, showextrema=True)
            parts["cmedians"].set_color("#ffd60a")
            for sp in ("cmins", "cmaxes", "cbars"):
                parts[sp].set_color(t["spine"])
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors_ne[i])
                pc.set_edgecolor(t["ax_bg"])
                pc.set_alpha(0.72)

            rng = np.random.default_rng(42)
            for i, d in enumerate(data_ne):
                jit = rng.uniform(-0.12, 0.12, size=len(d))
                ax.scatter(i + jit, d, color=colors_ne[i],
                           alpha=0.3, s=5, linewidths=0, zorder=3)

            ax.set_xticks(range(len(non_empty)))
            ax.set_xticklabels(labels_ne, color=t["tick"],
                               fontsize=max(5, 9 - ng // 6), rotation=45,
                               ha="right")
            ax.set_ylabel("Dwell time (s)", color=t["tick"], fontsize=9)
            ax.set_title(f"{eg}  (n = {len(ani_list)})",
                         color=eg_color, fontsize=9, fontweight="bold")

        for ei in range(n_eg, nrows * ncols):
            r, c = divmod(ei, ncols)
            axes[r][c].set_visible(False)

    fig.suptitle(
        "Dwell-time distributions by experimental group  [Behaviour Groups]\n"
        "Each violin = bout durations for one behaviour group  ·  Dots = individual bouts",
        color=t["tick"], fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def build_sankey_beh_groups_figure(
        animal_data: list, groups: dict, t: dict = None,
        n_steps: int = 5,
        anchor_group: str = None) -> plt.Figure:
    """
    Sankey (alluvial) flow diagram per experimental group using user-defined
    behavioural groups as states.  Cluster labels are mapped to groups first.

    anchor_group: name of the behavioural group to re-zero each animal's
    group-level sequence at its first occurrence.  None = auto-detect as
    source of the strongest normalised group-level transition.
    Animals whose group sequence never contains the anchor group are excluded.
    """
    if t is None:
        t = T()

    from matplotlib.path import Path as MPath

    if not groups:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No behaviour groups defined.\n"
                "Create groups in the Group Editor first.",
                ha="center", va="center", color=t["tick"],
                transform=ax.transAxes, fontsize=9)
        _style_ax(ax, t); return fig

    group_names  = list(groups.keys())
    group_colors = [groups[g].get("color", PALETTE[i % len(PALETTE)])
                    for i, g in enumerate(group_names)]
    ng = len(group_names)

    cid_to_gi: dict = {}
    for gi, gname in enumerate(group_names):
        for cid in groups[gname].get("labels", []):
            cid_to_gi[int(cid)] = gi

    groups_eg: dict = {}
    for ani in animal_data:
        groups_eg.setdefault(ani["exp_group"], []).append(ani)
    eg_names = list(groups_eg.keys())
    n_eg     = len(eg_names)

    if n_eg == 0:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No animals loaded.", ha="center", va="center",
                color=t["tick"], transform=ax.transAxes)
        _style_ax(ax, t); return fig

    def _raw_seqs_grp(ani_list):
        seqs = []
        for ani in ani_list:
            seq  = ani["df"].sort_values("start_frame")["label"].values
            gseq = [cid_to_gi[int(x)] for x in seq if int(x) in cid_to_gi]
            if len(gseq) >= 2:
                seqs.append(gseq)
        return seqs

    def _auto_anchor_grp(raw_gseqs):
        trans = np.zeros((ng, ng), dtype=float)
        for seq in raw_gseqs:
            for k in range(len(seq) - 1):
                trans[seq[k], seq[k + 1]] += 1
        row_s = trans.sum(axis=1, keepdims=True)
        row_s[row_s == 0] = 1.0
        norm = trans / row_s
        np.fill_diagonal(norm, 0.0)
        a_idx = int(np.unravel_index(norm.argmax(), norm.shape)[0])
        return group_names[a_idx]

    def _apply_anchor_grp(gseqs, anchor_gi):
        out = []
        for seq in gseqs:
            idx = next((i for i, gi in enumerate(seq) if gi == anchor_gi), None)
            if idx is not None and len(seq) - idx >= 2:
                out.append(seq[idx:])
        return out

    def _build_flow(seqs, n_steps_use):
        counts = np.zeros((n_steps_use, ng), dtype=float)
        trans  = np.zeros((n_steps_use - 1, ng, ng), dtype=float)
        for seq in seqs:
            for k in range(min(n_steps_use, len(seq))):
                counts[k, seq[k]] += 1
            for k in range(min(n_steps_use - 1, len(seq) - 1)):
                trans[k, seq[k], seq[k + 1]] += 1
        tot = counts.sum(axis=1, keepdims=True)
        tot[tot == 0] = 1.0
        props = counts / tot
        for k in range(n_steps_use - 1):
            rs = trans[k].sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            trans[k] /= rs
        return props, trans

    # Collect raw group sequences, then resolve anchor once across all data
    raw_grp_by_eg = {eg: _raw_seqs_grp(groups_eg[eg]) for eg in eg_names}
    all_raw_grp   = [s for seqs in raw_grp_by_eg.values() for s in seqs]
    if anchor_group is not None and anchor_group in group_names:
        resolved_group = anchor_group
        anchor_note    = "user-defined"
    else:
        resolved_group = _auto_anchor_grp(all_raw_grp)
        anchor_note    = "auto — strongest transition"
    resolved_gi = group_names.index(resolved_group)

    ncols  = min(n_eg, 2)
    nrows  = int(np.ceil(n_eg / ncols))
    fw     = max(10, n_steps * 2.2) * ncols
    fh     = 7.5 * nrows
    col_w  = 0.04
    bar_h  = 0.88
    gap    = 0.005

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fw, fh),
            facecolor=t["fig_bg"], squeeze=False)

        for ei, eg in enumerate(eg_names):
            r, c     = divmod(ei, ncols)
            ax       = axes[r][c]
            ax.set_facecolor(t["ax_bg"])
            ax.axis("off")
            eg_color = PALETTE[ei % len(PALETTE)]

            seqs       = _apply_anchor_grp(raw_grp_by_eg[eg], resolved_gi)
            n_excluded = len(raw_grp_by_eg[eg]) - len(seqs)
            if not seqs:
                ax.text(0.5, 0.5,
                        f"No data\n(anchor '{resolved_group}' never observed)",
                        ha="center", va="center", color=t["tick"],
                        transform=ax.transAxes)
                ax.set_title(f"{eg}", color=eg_color,
                             fontsize=10, fontweight="bold")
                continue

            n_use = min(n_steps, max(len(s) for s in seqs))
            if n_use < 2:
                ax.text(0.5, 0.5, "Sequences too short",
                        ha="center", va="center",
                        color=t["tick"], transform=ax.transAxes)
                continue

            props, trans = _build_flow(seqs, n_use)
            col_x_use    = np.linspace(0, 1, n_use)

            y_bot = np.zeros((n_use, ng), dtype=float)
            y_top = np.zeros((n_use, ng), dtype=float)
            for k in range(n_use):
                cum = 0.06
                for gi in range(ng):
                    h = props[k, gi] * bar_h
                    y_bot[k, gi] = cum
                    y_top[k, gi] = cum + h
                    cum += h + gap

            for k in range(n_use - 1):
                x0  = col_x_use[k] + col_w
                x1  = col_x_use[k + 1]
                cx0 = x0 + (x1 - x0) * 0.40
                cx1 = x0 + (x1 - x0) * 0.60
                src_off = np.zeros(ng, dtype=float)
                dst_off = np.zeros(ng, dtype=float)
                _sk_cf_g = 1.0 / max(1, ng - 1)
                for a in range(ng):
                    sh = y_top[k, a] - y_bot[k, a]
                    for b in range(ng):
                        p = float(trans[k, a, b])
                        if p <= _sk_cf_g:
                            continue
                        rhs = sh * p
                        rhd = (y_top[k+1, b] - y_bot[k+1, b]) * p
                        ys0 = y_bot[k, a]   + src_off[a]
                        ye0 = ys0 + rhs
                        ys1 = y_bot[k+1, b] + dst_off[b]
                        ye1 = ys1 + rhd
                        src_off[a] += rhs
                        dst_off[b] += rhd
                        verts = [
                            (x0, ys0), (cx0, ys0), (cx1, ys1), (x1, ys1),
                            (x1, ye1), (cx1, ye1), (cx0, ye0), (x0, ye0),
                            (x0, ys0),
                        ]
                        codes = ([MPath.MOVETO] + [MPath.CURVE4]*3 +
                                 [MPath.LINETO] + [MPath.CURVE4]*3 +
                                 [MPath.CLOSEPOLY])
                        ax.add_patch(mpatches.PathPatch(
                            MPath(verts, codes),
                            facecolor=group_colors[a],
                            edgecolor="none", alpha=0.35, zorder=1))

            for k in range(n_use):
                for gi in range(ng):
                    h = y_top[k, gi] - y_bot[k, gi]
                    if h < 1e-4:
                        continue
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (col_x_use[k], y_bot[k, gi]), col_w, h,
                        boxstyle="square,pad=0",
                        facecolor=group_colors[gi],
                        edgecolor=t["ax_bg"], linewidth=0.5, zorder=3))
                    if h > 0.04:
                        ax.text(col_x_use[k] + col_w/2,
                                y_bot[k, gi] + h/2,
                                group_names[gi],
                                ha="center", va="center",
                                fontsize=max(4, 8 - ng // 4),
                                color=t["text"], fontweight="bold", zorder=4)

            for k, x in enumerate(col_x_use):
                ax.text(x + col_w/2, 0.01, f"Step {k+1}",
                        ha="center", va="bottom", fontsize=8,
                        color=t["tick"], fontweight="bold")
            excl_note = (f"  ({n_excluded} excl. — no '{resolved_group}')"
                         if n_excluded else "")
            ax.set_xlim(-0.04, 1.06)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{eg}  (n = {len(seqs)}{excl_note})",
                         color=eg_color, fontsize=10, fontweight="bold", pad=4)

        for ei in range(n_eg, nrows * ncols):
            r, c = divmod(ei, ncols)
            axes[r][c].set_visible(False)

    fig.suptitle(
        f"Behaviour Group Sequence Sankey  (first {n_steps} bout positions)  "
        "[User-Defined Groups]\n"
        f"Anchor: '{resolved_group}'  [{anchor_note}]  ·  "
        "Bar height = group occupancy  ·  Ribbon width = transition flow",
        color=t["tick"], fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def build_group_aggregate_transition_figure(
        animal_data: list, groups: dict, t: dict = None) -> plt.Figure:
    """
    Per-experimental-group transition probability heatmaps at the
    behavioural-group level.  Each cell = mean probability of transitioning
    from one user-defined group to another.
    """
    if t is None:
        t = T()
    if not groups:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No behaviour groups defined.\n"
                "Create groups in the Group Editor first.",
                ha="center", va="center", color=t["tick"],
                transform=ax.transAxes, fontsize=9)
        _style_ax(ax, t); return fig

    group_names  = list(groups.keys())
    group_colors = [groups[g].get("color", PALETTE[i % len(PALETTE)])
                    for i, g in enumerate(group_names)]
    ng = len(group_names)

    cid_to_gi: dict = {}
    for gi, gname in enumerate(group_names):
        for cid in groups[gname].get("labels", []):
            cid_to_gi[int(cid)] = gi

    groups_eg: dict = {}
    for ani in animal_data:
        groups_eg.setdefault(ani["exp_group"], []).append(ani)
    eg_names = list(groups_eg.keys())
    n_eg     = len(eg_names)

    if n_eg == 0:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No animals loaded.", ha="center", va="center",
                color=t["tick"], transform=ax.transAxes)
        _style_ax(ax, t); return fig

    def _build_gmat(ani_list: list) -> np.ndarray:
        gmat = np.zeros((ng, ng), dtype=float)
        gcnt = np.zeros((ng, ng), dtype=float)
        for ani in ani_list:
            mat, ani_cids = compute_transition_matrix(ani["df"])
            for ri, ci in enumerate(ani_cids):
                if int(ci) not in cid_to_gi:
                    continue
                gi_r = cid_to_gi[int(ci)]
                for cj_i, cj in enumerate(ani_cids):
                    if int(cj) not in cid_to_gi:
                        continue
                    gi_c = cid_to_gi[int(cj)]
                    gmat[gi_r, gi_c] += mat[ri, cj_i]
                    gcnt[gi_r, gi_c] += 1.0
        safe = np.where(gcnt > 0, gcnt, 1.0)
        tmat = gmat / safe
        np.fill_diagonal(tmat, 0)
        return tmat

    ncols     = min(n_eg, 3)
    nrows     = int(np.ceil(n_eg / ncols))
    cell_size = max(3.5, ng * 0.45 + 2)
    fsize     = max(5, 9 - ng // 5)

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * cell_size + 0.5, nrows * cell_size + 1.2),
            facecolor=t["fig_bg"], squeeze=False,
        )
        axes_flat = axes.flatten()

        _cf_g = 1.0 / max(1, ng - 1)
        _cmap_mg = plt.cm.magma.copy(); _cmap_mg.set_bad(color=t["ax_bg"])

        for ei, eg in enumerate(eg_names):
            ax_eg    = axes_flat[ei]
            gmat     = _build_gmat(groups_eg[eg])
            # Row-normalise (diagonal already 0) → conditional exit probabilities
            _g_rs = gmat.sum(axis=1, keepdims=True); _g_rs[_g_rs == 0] = 1.0
            gmat  = gmat / _g_rs
            # Mask below-chance entries
            gmat_d = gmat.astype(float)
            gmat_d[gmat_d <= _cf_g] = np.nan
            eg_color = PALETTE[ei % len(PALETTE)]
            vmax     = float(np.nanmax(gmat_d)) if not np.all(np.isnan(gmat_d)) else 1.0

            im = ax_eg.imshow(gmat_d, cmap=_cmap_mg, aspect="auto",
                              interpolation="nearest", vmin=_cf_g, vmax=vmax)
            ax_eg.set_xticks(range(ng))
            ax_eg.set_xticklabels(group_names, rotation=45, ha="right",
                                   fontsize=fsize, color=t["tick"])
            ax_eg.set_yticks(range(ng))
            ax_eg.set_yticklabels(group_names, fontsize=fsize, color=t["tick"])
            for tick, col in zip(ax_eg.get_xticklabels(), group_colors):
                tick.set_color(col)
            for tick, col in zip(ax_eg.get_yticklabels(), group_colors):
                tick.set_color(col)
            ax_eg.set_title(f"{eg}  (n = {len(groups_eg[eg])})",
                            color=eg_color, fontsize=9, fontweight="bold")
            cb = fig.colorbar(im, ax=ax_eg, shrink=0.75, pad=0.02)
            cb.ax.tick_params(colors=t["tick"], labelsize=7)
            cb.outline.set_edgecolor(t["spine"])
            cb.set_label("P(transition | leaving group)", color=t["tick"], fontsize=7)
            _style_ax(ax_eg, t)

        for ei in range(n_eg, len(axes_flat)):
            axes_flat[ei].set_visible(False)

        fig.suptitle(
            "Behaviour Group Transition Probabilities by Experimental Group  "
            "[User-Defined Groups]\n"
            "(above-chance only · diagonal = 0 · conditional on leaving group)",
            color=t["tick"], fontweight="bold", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def build_diff_heatmap_beh_groups_figure(
        animal_data: list, groups: dict,
        ctrl_group: str = None, t: dict = None) -> plt.Figure:
    """
    Behavioural-group-level difference heatmaps.  Same layout as
    build_diff_heatmap_figure but nodes are user-defined groups.
    """
    if t is None:
        t = T()
    if not groups:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No behaviour groups defined.\n"
                "Create groups in the Group Editor first.",
                ha="center", va="center", color=t["tick"],
                transform=ax.transAxes, fontsize=9)
        _style_ax(ax, t); return fig

    group_names = list(groups.keys())
    ng          = len(group_names)

    cid_to_gi: dict = {}
    for gi, gname in enumerate(group_names):
        for cid in groups[gname].get("labels", []):
            cid_to_gi[int(cid)] = gi

    groups_eg: dict = {}
    for ani in animal_data:
        groups_eg.setdefault(ani["exp_group"], []).append(ani)
    eg_names = list(groups_eg.keys())
    n_eg     = len(eg_names)

    if n_eg < 2:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5,
                "Need ≥ 2 experimental groups for difference heatmaps.",
                ha="center", va="center", color=t["tick"],
                transform=ax.transAxes)
        _style_ax(ax, t); return fig

    def _group_tmat(ani_list: list) -> np.ndarray:
        gmat = np.zeros((ng, ng), dtype=float)
        gcnt = np.zeros((ng, ng), dtype=float)
        for ani in ani_list:
            mat, ani_cids = compute_transition_matrix(ani["df"])
            for ri, ci in enumerate(ani_cids):
                if int(ci) not in cid_to_gi:
                    continue
                gi_r = cid_to_gi[int(ci)]
                for cj_i, cj in enumerate(ani_cids):
                    if int(cj) not in cid_to_gi:
                        continue
                    gi_c = cid_to_gi[int(cj)]
                    gmat[gi_r, gi_c] += mat[ri, cj_i]
                    gcnt[gi_r, gi_c] += 1.0
        safe = np.where(gcnt > 0, gcnt, 1.0)
        tmat = gmat / safe
        np.fill_diagonal(tmat, 0)
        _rs = tmat.sum(axis=1, keepdims=True); _rs[_rs == 0] = 1.0
        return tmat / _rs

    tmats  = {eg: _group_tmat(groups_eg[eg]) for eg in eg_names}
    ref_eg = ctrl_group if (ctrl_group and ctrl_group in tmats) else eg_names[0]
    pairs  = [(ref_eg, eg) for eg in eg_names if eg != ref_eg]
    n_pairs = len(pairs)

    cell  = max(3.5, ng * 0.45 + 2)
    fsize = max(5, 9 - ng // 5)

    _cf_bg  = 1.0 / max(1, ng - 1)
    _cmap_bg = plt.cm.magma.copy(); _cmap_bg.set_bad(color=t["ax_bg"])

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(
            n_pairs, 3,
            figsize=(cell * 3 + 1.0, cell * n_pairs + 1.2),
            facecolor=t["fig_bg"], squeeze=False)

        for row, (eg_a, eg_b) in enumerate(pairs):
            col_a = PALETTE[eg_names.index(eg_a) % len(PALETTE)]
            col_b = PALETTE[eg_names.index(eg_b) % len(PALETTE)]

            for col_i, (mat, title, cmap_name, cblabel) in enumerate([
                (tmats[eg_a], f"{eg_a}  (reference)", "magma",
                 "P(transition | leaving group)"),
                (tmats[eg_b], f"{eg_b}", "magma",
                 "P(transition | leaving group)"),
                (tmats[eg_b] - tmats[eg_a], f"Δ = {eg_b} − {eg_a}", "RdBu_r", "ΔP"),
            ]):
                ax = axes[row, col_i]
                if col_i < 2:
                    mat_d = mat.astype(float)
                    mat_d[mat_d <= _cf_bg] = np.nan
                    vmax_use = float(np.nanmax(mat_d)) if not np.all(np.isnan(mat_d)) else 1.0
                    im = ax.imshow(mat_d, cmap=_cmap_bg, aspect="auto",
                                   interpolation="nearest",
                                   vmin=_cf_bg, vmax=vmax_use)
                else:
                    vmax_use = float(np.abs(mat).max()) if np.abs(mat).max() > 0 else 1.0
                    im = ax.imshow(mat, cmap=cmap_name, aspect="auto",
                                   interpolation="nearest",
                                   vmin=-vmax_use, vmax=vmax_use)
                ax.set_xticks(range(ng))
                ax.set_xticklabels(group_names, rotation=45, ha="right",
                                   fontsize=fsize, color=t["tick"])
                ax.set_yticks(range(ng))
                ax.set_yticklabels(group_names, fontsize=fsize, color=t["tick"])
                tc = col_a if col_i == 0 else (col_b if col_i == 1 else t["tick"])
                ax.set_title(title, color=tc, fontsize=9, fontweight="bold")
                cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
                cb.ax.tick_params(colors=t["tick"], labelsize=7)
                cb.set_label(cblabel, color=t["tick"], fontsize=7)
                cb.outline.set_edgecolor(t["spine"])
                _style_ax(ax, t)

    fig.suptitle(
        "Behaviour Group Transition Difference Heatmaps  [User-Defined Groups]\n"
        "Red = experimental group transitions MORE  ·  Blue = transitions LESS",
        color=t["tick"], fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def build_combined_figure(groups, combined, eg_colors_override=None, t=None) -> plt.Figure:
    """
    Combined figure with one subplot per behavioural group per metric so that
    each group can use its own y-scale.  Stars denote statistical significance.
    eg_colors_override: {eg_name: hex}    if supplied, overrides PALETTE for exp groups.
    """
    if t is None:
        t = T()
    grand      = combined["grand"]
    records    = combined["records"]
    uid_to_idx = combined["uid_to_idx"]
    exp_groups = combined["exp_groups"]
    beh_gnames = list(groups.keys())
    eg_names   = list(exp_groups.keys())

    #   use caller-supplied colours or fall back to PALETTE  
    eg_colors: dict = {}
    for i, eg in enumerate(eg_names):
        if eg_colors_override and eg in eg_colors_override:
            eg_colors[eg] = eg_colors_override[eg]
        else:
            eg_colors[eg] = PALETTE[i % len(PALETTE)]

    METRICS = [
        ("total_duration", "Total Duration (s)"),
        ("frequency",      "Frequency (# bouts)"),
        ("latency",        "Latency (s)"),
        ("mean_bout",      "Mean Bout (s)"),
    ]
    n_metrics  = len(METRICS)
    n_beh      = len(beh_gnames)
    n_eg       = len(eg_names)
    rng        = np.random.default_rng(42)

    #   layout: rows = metrics, cols = behaviour groups  
    fig_w = max(14, n_beh * 2.8 + 1)
    fig_h = n_metrics * 3.6 + 0.8
    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(
            n_metrics, n_beh,
            figsize=(fig_w, fig_h),
            facecolor=t["fig_bg"],
            squeeze=False,
        )
        fig.subplots_adjust(hspace=0.62, wspace=0.48,
                            left=0.07, right=0.97, top=0.93, bottom=0.08)

        for mi, (metric_key, ylabel) in enumerate(METRICS):
            for bi, bg in enumerate(beh_gnames):
                ax = axes[mi, bi]
                beh_color = groups[bg]["color"]     # behavioural group colour
                x_base = np.arange(n_eg)

                #   compute p-value if scipy available  
                pval = None
                if SCIPY_OK and n_eg >= 2:
                    group_vals = []
                    for eg in eg_names:
                        vs = [records[uid_to_idx[uid]]["metrics"]
                              .get(bg, {}).get(metric_key) or 0.0
                              for uid in exp_groups[eg]]
                        group_vals.append(np.array(vs, dtype=float))
                    try:
                        _, pval = sp_stats.kruskal(*group_vals)
                    except Exception:
                        try:
                            _, pval = sp_stats.f_oneway(*group_vals)
                        except Exception:
                            pval = None

                for ei, eg in enumerate(eg_names):
                    ec   = eg_colors[eg]
                    mean = grand[eg][bg][metric_key]["mean"]
                    sem  = grand[eg][bg][metric_key]["sem"]
                    n    = grand[eg][bg][metric_key]["n"]

                    # bar with behaviour-group colour border, eg fill
                    ax.bar(x_base[ei], mean, width=0.6,
                           color=ec, edgecolor=beh_color,
                           linewidth=1.4, alpha=0.82, zorder=2)
                    ax.errorbar(x_base[ei], mean, yerr=sem, fmt="none",
                                color=t["tick"], linewidth=1.4, capsize=4, zorder=3)

                    # individual dots
                    pts = [records[uid_to_idx[uid]]["metrics"]
                           .get(bg, {}).get(metric_key) or 0.0
                           for uid in exp_groups[eg]]
                    jitter = rng.uniform(-0.12, 0.12, len(pts))
                    ax.scatter(x_base[ei] + jitter, pts,
                               color=t["ax_bg"], edgecolors=ec,
                               s=32, linewidths=0.9, alpha=0.88, zorder=4)

                    y_max = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else (mean * 1.3 or 1)
                    ax.text(x_base[ei], y_max * 0.02, f"n={n}",
                            ha="center", va="bottom", color=t["tick"], fontsize=7)

                #   significance stars — always shown (inc. ns)
                if pval is not None and n_eg >= 2:
                    stars = ("***" if pval < 0.001 else "**" if pval < 0.01
                             else "*" if pval < 0.05 else "ns")
                    all_pts_flat = [records[uid_to_idx[uid]]["metrics"]
                                    .get(bg, {}).get(metric_key) or 0.0
                                    for eg in eg_names for uid in exp_groups[eg]]
                    data_top = max(
                        (grand[eg][bg][metric_key]["mean"] +
                         grand[eg][bg][metric_key]["sem"]
                         for eg in eg_names), default=0)
                    head_top = max(data_top * 1.40,
                                   max(all_pts_flat or [0]) * 1.40, 0.001)
                    ax.set_ylim(bottom=0, top=head_top)
                    x0, x1 = 0, n_eg - 1
                    bh = head_top * 0.80
                    ax.plot([x0, x0, x1, x1],
                            [bh, bh + head_top * 0.04, bh + head_top * 0.04, bh],
                            lw=1.0, color=t["tick"])
                    star_color = "#FF4081" if stars != "ns" else t["muted"]
                    ax.text((x0 + x1) / 2, bh + head_top * 0.055, stars,
                            ha="center", fontsize=10, color=star_color,
                            fontweight="bold" if stars != "ns" else "normal")

                ax.set_xticks(x_base)
                ax.set_xticklabels(eg_names, rotation=22, ha="right",
                                   color=t["tick"], fontsize=8)
                # column header on top row only
                if mi == 0:
                    ax.set_title(bg, color=beh_color,
                                 fontweight="bold", fontsize=9)
                # row label on left column only
                if bi == 0:
                    ax.set_ylabel(ylabel, color=t["tick"], fontsize=8)
                _style_ax(ax, t)

        #   row metric labels (right side)  
        for mi, (_, ylabel) in enumerate(METRICS):
            axes[mi, -1].annotate(
                ylabel, xy=(1.02, 0.5), xycoords="axes fraction",
                rotation=-90, va="center", ha="left",
                fontsize=8, color=t["subtext"],
            )

        # legend for exp groups
        handles = [mpatches.Patch(color=eg_colors[eg], label=eg) for eg in eg_names]
        fig.legend(handles=handles, loc="upper right", fontsize=8,
                   framealpha=0.3, facecolor=t["ax_bg"], labelcolor=t["tick"],
                   title="Exp Group", title_fontsize=8)
        fig.suptitle(
            "Combined Multi-Animal Analysis  (Mean   SEM, individual animals shown)\n"
            "Columns = behaviour groups (own y-scale)  -  * p<0.05  ** p<0.01  *** p<0.001",
            color=t["tick"], fontsize=11, fontweight="bold", y=0.98)
    return fig


def build_combined_group_figure(bg_name: str, groups: dict, combined: dict,
                                eg_colors_override: dict = None, t=None) -> plt.Figure:
    """
    2x2 metric figure for a single behaviour group, comparing all exp groups.
    One subplot per metric: total_duration, frequency, latency, mean_bout.
    """
    if t is None:
        t = T()
    grand      = combined["grand"]
    records    = combined["records"]
    uid_to_idx = combined["uid_to_idx"]
    exp_groups = combined["exp_groups"]
    eg_names   = list(exp_groups.keys())
    bg_color   = groups[bg_name]["color"]
    n_eg       = len(eg_names)
    rng        = np.random.default_rng(42)

    eg_colors: dict = {}
    for i, eg in enumerate(eg_names):
        if eg_colors_override and eg in eg_colors_override:
            eg_colors[eg] = eg_colors_override[eg]
        else:
            eg_colors[eg] = PALETTE[i % len(PALETTE)]

    METRICS = [
        ("total_duration", "Total Duration (s)"),
        ("frequency",      "Frequency (# bouts)"),
        ("latency",        "Latency (s)"),
        ("mean_bout",      "Mean Bout (s)"),
    ]

    # Pass 1: collect raw KW p-values for all 4 metrics, then apply BH correction
    metric_keys = [m for m, _ in METRICS]
    raw_pvals_m: dict = {}
    if SCIPY_OK and n_eg >= 2:
        for metric_key in metric_keys:
            group_vals = [
                np.array([records[uid_to_idx[uid]]["metrics"]
                           .get(bg_name, {}).get(metric_key) or 0.0
                           for uid in exp_groups[eg]], dtype=float)
                for eg in eg_names
            ]
            try:
                _, pv = sp_stats.kruskal(*group_vals)
            except Exception:
                try:
                    _, pv = sp_stats.f_oneway(*group_vals)
                except Exception:
                    pv = None
            raw_pvals_m[metric_key] = pv
        p_arr_m  = np.array([raw_pvals_m[k] if raw_pvals_m[k] is not None else 1.0
                              for k in metric_keys])
        q_arr_m  = benjamini_hochberg(p_arr_m)
        qvals_m  = {metric_keys[i]: float(q_arr_m[i]) if raw_pvals_m[metric_keys[i]] is not None else None
                    for i in range(len(metric_keys))}
    else:
        qvals_m = {k: None for k in metric_keys}

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.5),
                                  facecolor=t["fig_bg"], squeeze=False)
        fig.subplots_adjust(hspace=0.52, wspace=0.42,
                            left=0.10, right=0.85, top=0.87, bottom=0.13)
        fig.suptitle(f"Behaviour Group: {bg_name}  —  Mean ± SEM by Exp Group"
                     f"\n* FDR q<0.05  ** q<0.01  *** q<0.001  (BH correction across 4 metrics)",
                     color=bg_color, fontsize=11, fontweight="bold", y=0.99)

        for idx, (metric_key, ylabel) in enumerate(METRICS):
            ax    = axes[idx // 2, idx % 2]
            x_base = np.arange(n_eg)
            qval  = qvals_m.get(metric_key)

            for ei, eg in enumerate(eg_names):
                ec   = eg_colors[eg]
                mean = grand[eg][bg_name][metric_key]["mean"]
                sem  = grand[eg][bg_name][metric_key]["sem"]
                n    = grand[eg][bg_name][metric_key]["n"]
                ax.bar(x_base[ei], mean, width=0.6,
                       color=ec, edgecolor="none",
                       linewidth=0, alpha=0.82, zorder=2)
                ax.errorbar(x_base[ei], mean, yerr=sem, fmt="none",
                            color=t["tick"], linewidth=1.4, capsize=4, zorder=3)
                pts = [records[uid_to_idx[uid]]["metrics"]
                       .get(bg_name, {}).get(metric_key) or 0.0
                       for uid in exp_groups[eg]]
                jitter = rng.uniform(-0.12, 0.12, len(pts))
                ax.scatter(x_base[ei] + jitter, pts,
                           color=t["ax_bg"], edgecolors=ec,
                           s=34, linewidths=0.9, alpha=0.88, zorder=4)
                y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else (mean * 1.3 or 1)
                ax.text(x_base[ei], y_top * 0.02, f"n={n}",
                        ha="center", va="bottom", color=t["tick"], fontsize=7)

            if qval is not None and n_eg >= 2:
                stars = ("***" if qval < 0.001 else "**" if qval < 0.01
                         else "*" if qval < 0.05 else "ns")
                all_pts_flat = [records[uid_to_idx[uid]]["metrics"]
                                .get(bg_name, {}).get(metric_key) or 0.0
                                for eg in eg_names for uid in exp_groups[eg]]
                data_top = max(
                    (grand[eg][bg_name][metric_key]["mean"] +
                     grand[eg][bg_name][metric_key]["sem"]
                     for eg in eg_names), default=0)
                head_top = max(data_top * 1.40,
                               max(all_pts_flat or [0]) * 1.40, 0.001)
                ax.set_ylim(bottom=0, top=head_top)
                x0, x1 = 0, n_eg - 1
                bh = head_top * 0.80
                ax.plot([x0, x0, x1, x1],
                        [bh, bh + head_top * 0.04, bh + head_top * 0.04, bh],
                        lw=1.0, color=t["tick"])
                star_color = "#FF4081" if stars != "ns" else t["muted"]
                ax.text((x0 + x1) / 2, bh + head_top * 0.055, stars,
                        ha="center", fontsize=10, color=star_color,
                        fontweight="bold" if stars != "ns" else "normal")

            ax.set_xticks(x_base)
            ax.set_xticklabels(eg_names, rotation=25, ha="right",
                               color=t["tick"], fontsize=9)
            ax.set_ylabel(ylabel, color=t["tick"], fontsize=9)
            ax.set_title(ylabel.split(" (")[0], color=t["tick"],
                         fontsize=10, fontweight="bold")
            _style_ax(ax, t)

        handles = [mpatches.Patch(color=eg_colors[eg], label=eg) for eg in eg_names]
        fig.legend(handles=handles, bbox_to_anchor=(0.87, 0.88),
                   loc="upper left", fontsize=9,
                   framealpha=0.3, facecolor=t["ax_bg"], labelcolor=t["tick"],
                   title="Exp Group", title_fontsize=8)
    return fig


def build_combined_ethogram_figure(groups, combined, t=None) -> plt.Figure:
    if t is None:
        t = T()
    records    = combined["records"]
    uid_to_idx = combined["uid_to_idx"]
    exp_groups = combined["exp_groups"]
    ordered    = [(eg, uid) for eg, uids in exp_groups.items() for uid in uids]
    n_ani = len(ordered)
    if n_ani == 0:
        fig, ax = plt.subplots(figsize=(14, 2), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, "No animals loaded", ha="center", va="center",
                color=t["tick"], transform=ax.transAxes)
        _style_ax(ax, t)
        return fig
    eg_names   = list(exp_groups.keys())
    eg_palette = {eg: PALETTE[i % len(PALETTE)] for i, eg in enumerate(eg_names)}
    fig_h = max(4, n_ani * 1.3 + 1.5)
    with plt.style.context(t["mpl_style"]):
        fig, ax = plt.subplots(figsize=(16, fig_h), facecolor=t["fig_bg"])
        fig.subplots_adjust(left=0.22, right=0.97, top=0.92, bottom=0.08)
        yticks, ylabels, ycolors = [], [], []
        for row_i, (eg, uid) in enumerate(ordered):
            rec   = records[uid_to_idx[uid]]
            y     = n_ani - 1 - row_i
            am    = rec["metrics"]
            dur_s = rec["session_s"]
            ec    = eg_palette[eg]
            band_alpha = 0.10 if row_i % 2 == 0 else 0.04
            ax.barh(y, dur_s, left=0, height=0.82,
                    color=ec, alpha=band_alpha, zorder=0)
            ax.barh(y, dur_s, left=0, height=0.55,
                    color=t["track_bg"], alpha=0.5, zorder=1)
            for beh_g, ginfo in groups.items():
                evs = am.get(beh_g, {}).get("events", [])
                if evs:
                    # broken_barh sends ONE draw call regardless of event count,
                    # avoiding thousands of GDI objects that can crash win32k.sys.
                    ax.broken_barh(
                        [(ev["start_s"], ev["dur_s"]) for ev in evs],
                        (y - 0.275, 0.55),
                        facecolors=ginfo["color"], alpha=0.9, zorder=2)
            yticks.append(y)
            ylabels.append(f"[{eg}] #{uid}  {rec['name']}")
            ycolors.append(ec)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8.5)
        for tick_lbl, col in zip(ax.get_yticklabels(), ycolors):
            tick_lbl.set_color(col)
        ax.set_xlabel("Time (s)", color=t["tick"])
        ax.set_xlim(0)
        ax.set_ylim(-0.6, n_ani - 0.4)
        beh_patches = [mpatches.Patch(color=ginfo["color"], label=gn)
                       for gn, ginfo in groups.items()]
        eg_patches  = [mpatches.Patch(color=eg_palette[eg], label=eg)
                       for eg in eg_names]
        legend1 = ax.legend(handles=beh_patches, loc="upper right", fontsize=8,
                            title="Behaviour", title_fontsize=8,
                            framealpha=0.35, facecolor=t["ax_bg"],
                            labelcolor=t["tick"])
        ax.add_artist(legend1)
        ax.legend(handles=eg_patches, loc="lower right", fontsize=8,
                  title="Exp Group", title_fontsize=8,
                  framealpha=0.35, facecolor=t["ax_bg"],
                  labelcolor=t["tick"])
        ax.set_title(
            f"Ethograms - {n_ani} animals across {len(eg_names)} experimental group(s)",
            color=t["tick"], fontweight="bold", fontsize=12)
        _style_ax(ax, t)
    return fig


def build_combined_transition_figure(
        bg_name: str,
        groups: dict,
        combined: dict,
        eg_colors_override: dict = None,
        t: dict = None):
    """Top-10 transition pairs within a behaviour group, ranked by between-group
    KW significance (not prevalence). Returns (fig, stats_df) or (None, None).
    """
    if t is None:
        t = T()
    if not SCIPY_OK:
        return None, None

    records    = combined["records"]
    exp_groups = combined["exp_groups"]
    eg_names   = list(exp_groups.keys())
    if len(eg_names) < 2:
        return None, None

    cluster_ids = set(groups[bg_name]["labels"])
    bg_color    = groups[bg_name]["color"]

    # ── Per-animal transition probabilities (within this behaviour group) ─────
    animal_trans: dict = {}   # uid -> {(ci, cj): prob}
    all_pairs:    set  = set()
    for rec in records:
        mat, cids = compute_transition_matrix(rec["df"])
        trans: dict = {}
        for ri, ci in enumerate(cids):
            if ci not in cluster_ids:
                continue
            for rj, cj in enumerate(cids):
                if cj not in cluster_ids or ci == cj:
                    continue
                trans[(ci, cj)] = float(mat[ri, rj])
                all_pairs.add((ci, cj))
        animal_trans[rec["uid"]] = trans

    if not all_pairs:
        return None, None

    # ── Exp group colours (shared with all other combined figures) ────────────
    eg_colors: dict = {}
    for i, eg in enumerate(eg_names):
        if eg_colors_override and eg in eg_colors_override:
            eg_colors[eg] = eg_colors_override[eg]
        else:
            eg_colors[eg] = PALETTE[i % len(PALETTE)]

    # ── KW test per pair — sorted by group DIFFERENCE, not prevalence ─────────
    all_pairs_list = sorted(all_pairs)
    raw_pvals: dict = {}
    for pair in all_pairs_list:
        group_vals = [
            np.array([animal_trans.get(uid, {}).get(pair, 0.0)
                      for uid in exp_groups[eg]], dtype=float)
            for eg in eg_names
        ]
        try:
            _, pv = sp_stats.kruskal(*group_vals)
        except Exception:
            try:
                _, pv = sp_stats.f_oneway(*group_vals)
            except Exception:
                pv = None
        raw_pvals[pair] = pv

    valid_pairs = [p for p in all_pairs_list if raw_pvals[p] is not None]
    if not valid_pairs:
        return None, None

    p_arr = np.array([raw_pvals[p] for p in valid_pairs])
    q_arr = benjamini_hochberg(p_arr)
    qvals = {valid_pairs[i]: float(q_arr[i]) for i in range(len(valid_pairs))}

    # Top-10 by ascending q: most significantly different between groups first
    top10  = sorted(valid_pairs, key=lambda p: qvals[p])[:10]
    n_top  = len(top10)
    n_eg   = len(eg_names)
    n_cols = min(n_top, 5)
    n_rows = (n_top + n_cols - 1) // n_cols

    STAR_THRESH = [(0.001, "***"), (0.01, "**"), (0.05, "*")]

    def _stars(qv):
        if qv is None:
            return "N/A"
        for thresh, sym in STAR_THRESH:
            if qv < thresh:
                return sym
        return "ns"

    rng    = np.random.default_rng(42)
    x_base = np.arange(n_eg)

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.5 * n_cols, 3.2 * n_rows + 0.9),
            facecolor=t["fig_bg"],
            squeeze=False)
        fig.subplots_adjust(
            hspace=0.72, wspace=0.45,
            left=0.10, right=0.87,
            top=1.0 - 0.55 / (3.2 * n_rows + 0.9),
            bottom=0.14)
        fig.suptitle(
            f"Behaviour Group: {bg_name}  —  Top Transitions by Group Difference"
            f"\n* FDR q<0.05  ** q<0.01  *** q<0.001  (BH correction, KW test)",
            color=bg_color, fontsize=11, fontweight="bold", y=0.99)

        for idx, pair in enumerate(top10):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]
            ci, cj = pair
            qval = qvals[pair]

            means, sems, pts_list = [], [], []
            for eg in eg_names:
                vals = np.array([animal_trans.get(uid, {}).get(pair, 0.0)
                                 for uid in exp_groups[eg]], dtype=float)
                means.append(float(np.mean(vals)))
                sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                             if len(vals) > 1 else 0.0)
                pts_list.append(vals)

            for ei, eg in enumerate(eg_names):
                ec = eg_colors[eg]
                ax.bar(x_base[ei], means[ei], width=0.6,
                       color=ec, edgecolor="none", alpha=0.82, zorder=2)
                ax.errorbar(x_base[ei], means[ei], yerr=sems[ei], fmt="none",
                            color=t["tick"], linewidth=1.4, capsize=4, zorder=3)
                jitter = rng.uniform(-0.12, 0.12, len(pts_list[ei]))
                ax.scatter(x_base[ei] + jitter, pts_list[ei],
                           color=t["ax_bg"], edgecolors=ec,
                           s=34, linewidths=0.9, alpha=0.88, zorder=4)

            # Significance bracket spanning all groups
            stars = _stars(qval)
            if n_eg >= 2:
                all_pts_flat = [v for pts in pts_list for v in pts]
                data_top  = max((m + s for m, s in zip(means, sems)), default=0)
                head_top  = max(data_top * 1.40,
                                max(all_pts_flat or [0]) * 1.40, 0.001)
                ax.set_ylim(bottom=0, top=head_top)
                x0, x1   = 0, n_eg - 1
                bh        = head_top * 0.80
                ax.plot([x0, x0, x1, x1],
                        [bh, bh + head_top * 0.04,
                         bh + head_top * 0.04, bh],
                        lw=1.0, color=t["tick"])
                star_color = "#FF4081" if stars != "ns" else t["muted"]
                ax.text((x0 + x1) / 2, bh + head_top * 0.055, stars,
                        ha="center", fontsize=10, color=star_color,
                        fontweight="bold" if stars != "ns" else "normal")

            ax.set_xticks(x_base)
            ax.set_xticklabels(eg_names, rotation=25, ha="right",
                               color=t["tick"], fontsize=8)
            ax.set_ylabel("Transition prob.", color=t["tick"], fontsize=8)
            ax.set_title(f"C{ci}→C{cj}  (q={qval:.3f})",
                         color=t["tick"], fontsize=9, fontweight="bold")
            _style_ax(ax, t)

        # Hide unused subplot cells
        for idx in range(n_top, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)

        handles = [mpatches.Patch(color=eg_colors[eg], label=eg)
                   for eg in eg_names]
        fig.legend(handles=handles,
                   bbox_to_anchor=(0.88, 0.88), loc="upper left",
                   fontsize=9, framealpha=0.3, facecolor=t["ax_bg"],
                   labelcolor=t["tick"],
                   title="Exp Group", title_fontsize=8)

    # ── Stats DataFrame for CSV export ────────────────────────────────────────
    stats_rows = []
    for pair in top10:
        ci, cj   = pair
        qval     = qvals[pair]
        pval     = raw_pvals[pair]
        stars    = _stars(qval)
        for eg in eg_names:
            vals = np.array([animal_trans.get(uid, {}).get(pair, 0.0)
                             for uid in exp_groups[eg]], dtype=float)
            n = len(vals)
            stats_rows.append({
                "beh_group":    bg_name,
                "transition":   f"C{ci}→C{cj}",
                "exp_group":    eg,
                "mean":         round(float(np.mean(vals)), 6),
                "sem":          round(float(np.std(vals, ddof=1) / np.sqrt(n))
                                      if n > 1 else 0.0, 6),
                "n":            n,
                "p_value_KW":   round(pval, 6) if pval is not None else "N/A",
                "q_value_FDR":  round(qval, 6),
                "significance": stars,
            })

    return fig, pd.DataFrame(stats_rows)


#
# EXPORT
#

def export_results(root, groups, metrics, figure, csv_path, fps,
                   out_override=None):
    out = pathlib.Path(out_override) if out_override else (root / RESULTS_SUBDIR)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for ext in ("png", "pdf"):
        figure.savefig(out / f"analysis_{ts}.{ext}",
                       dpi=300 if ext == "png" else None,
                       bbox_inches="tight",
                       facecolor=figure.get_facecolor())
    rows = []
    for gname, m in metrics.items():
        rows.append({
            "Group":              gname,
            "Labels":             ",".join(str(l) for l in groups[gname]["labels"]),
            "Color":              groups[gname]["color"],
            "Total Duration (s)": m["total_duration"],
            "Frequency":          m["frequency"],
            "Latency (s)":        m["latency"] if m["latency"] is not None else "N/A",
            "Mean Bout (s)":      m["mean_bout"],
        })
    pd.DataFrame(rows).to_csv(out / f"metrics_{ts}.csv", index=False)
    log = {
        "created":    ts,
        "source_csv": str(csv_path),
        "fps":        fps,
        "groups": {
            gn: {"labels": gi["labels"], "color": gi["color"]}
            for gn, gi in groups.items()
        },
    }
    with open(out / "Mapping_Log.json", "w") as fh:
        json.dump(log, fh, indent=2)
    return out


#  
# CANVAS PANEL
#  

class CanvasPanel(ctk.CTkFrame):
    def __init__(self, parent, **kw):
        super().__init__(parent, fg_color=T()["panel"], **kw)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._mpl_canvas = None   # named to avoid colliding with CTkFrame._canvas
        self._figure = None

    def show_figure(self, fig: plt.Figure):
        if self._figure and self._figure is not fig:
            if self._mpl_canvas is not None:
                try:
                    self._mpl_canvas.get_tk_widget().unbind("<Destroy>")
                except Exception:
                    pass
            plt.close(self._figure)
        for w in self.winfo_children():
            w.destroy()
        self._figure = fig
        self._mpl_canvas = FigureCanvasTkAgg(fig, master=self)
        self._mpl_canvas.draw()
        self._mpl_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        tb_fr = ctk.CTkFrame(self, fg_color=T()["card"], height=36)
        tb_fr.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self._mpl_canvas, tb_fr)

    def get_figure(self):
        return self._figure

    def clear(self, message=""):
        if self._figure:
            if self._mpl_canvas is not None:
                try:
                    self._mpl_canvas.get_tk_widget().unbind("<Destroy>")
                except Exception:
                    pass
            plt.close(self._figure)
        self._figure = None
        self._mpl_canvas = None
        for w in self.winfo_children():
            w.destroy()
        if message:
            ctk.CTkLabel(self, text=message,
                         text_color=T()["muted"],
                         font=ctk.CTkFont(size=14),
                         justify="center"
                         ).grid(row=0, column=0)


#  
# GROUP ROW WIDGET  (with drag-reorder handles)
#  

class GroupRow(ctk.CTkFrame):
    COLS = 8

    def __init__(self, parent, group_name, color, all_labels,
                 selected, on_change, on_delete, on_move_up, on_move_down, **kw):
        super().__init__(parent, fg_color=T()["card"],
                         corner_radius=8, **kw)
        self.on_change   = on_change
        self.on_delete   = on_delete
        self.on_move_up  = on_move_up
        self.on_move_down = on_move_down
        self._color_hex  = color

        self.columnconfigure(4, weight=1)

        #   drag handle (  order buttons)  
        drag_fr = ctk.CTkFrame(self, fg_color="transparent", width=26)
        drag_fr.grid(row=0, column=0, rowspan=2, padx=(6, 2), pady=6)
        ctk.CTkButton(drag_fr, text=" ", width=22, height=18,
                      fg_color=T()["card2"], hover_color=T()["drag_highlight"],
                      command=lambda: self.on_move_up(self),
                      font=ctk.CTkFont(size=10),
                      ).pack(pady=(0, 1))
        ctk.CTkButton(drag_fr, text=" ", width=22, height=18,
                      fg_color=T()["card2"], hover_color=T()["drag_highlight"],
                      command=lambda: self.on_move_down(self),
                      font=ctk.CTkFont(size=10),
                      ).pack()

        #   colour swatch  
        self.swatch = ctk.CTkButton(
            self, text="", width=36, height=36,
            fg_color=color, hover_color=color,
            command=self._pick_color, corner_radius=6,
        )
        self.swatch.grid(row=0, column=1, padx=(2, 4), pady=10, rowspan=2)

        #   group name  
        self.name_var = ctk.StringVar(value=group_name)
        ctk.CTkEntry(self, textvariable=self.name_var, width=140,
                     placeholder_text="Group name",
                     ).grid(row=0, column=2, padx=4, pady=(10, 2))
        self.name_var.trace_add("write", lambda *_: self.on_change())

        self._badge_var = ctk.StringVar(value="0 labels")
        ctk.CTkLabel(self, textvariable=self._badge_var,
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11),
                     ).grid(row=1, column=2, padx=4, pady=(0, 8), sticky="w")

        #   all / none shortcuts  
        btn_fr = ctk.CTkFrame(self, fg_color="transparent")
        btn_fr.grid(row=0, column=3, rowspan=2, padx=4)
        ctk.CTkButton(btn_fr, text="All",  width=52, height=26,
                      command=self._select_all).pack(pady=2)
        ctk.CTkButton(btn_fr, text="None", width=52, height=26,
                      command=self._select_none).pack(pady=2)

        #   label checkbox grid  
        self._label_scroll = ctk.CTkScrollableFrame(
            self, fg_color=T()["card2"], corner_radius=6, height=96)
        self._label_scroll.grid(row=0, column=4, rowspan=2,
                                padx=(8, 4), pady=8, sticky="nsew")
        self._check_vars: dict = {}
        self._build_label_grid(all_labels, selected)

        #   delete button  
        ctk.CTkButton(
            self, text=" ", width=30, height=30,
            fg_color=T()["btn_del"], hover_color=T()["btn_del_h"],
            command=lambda: self.on_delete(self),
        ).grid(row=0, column=5, padx=(4, 10), pady=10, rowspan=2)

    def _build_label_grid(self, all_labels: list, selected: list):
        for w in self._label_scroll.winfo_children():
            w.destroy()
        self._check_vars.clear()
        for i, lbl in enumerate(sorted(all_labels)):
            var = tk.BooleanVar(value=lbl in selected)
            self._check_vars[lbl] = var
            cb = ctk.CTkCheckBox(
                self._label_scroll, text=str(lbl), variable=var,
                width=68, command=self._on_cb,
                checkbox_width=16, checkbox_height=16,
            )
            cb.grid(row=i // self.COLS, column=i % self.COLS,
                    padx=2, pady=1, sticky="w")
        self._update_badge()

    def rebuild_label_grid(self, all_labels: list):
        prev = set(self.get_labels())
        self._build_label_grid(all_labels, prev)

    def _on_cb(self):
        self._update_badge()
        self.on_change()

    def _update_badge(self):
        n = sum(1 for v in self._check_vars.values() if v.get())
        self._badge_var.set(f"{n} label{'s' if n != 1 else ''} selected")

    def _select_all(self):
        for v in self._check_vars.values():
            v.set(True)
        self._on_cb()

    def _select_none(self):
        for v in self._check_vars.values():
            v.set(False)
        self._on_cb()

    def _pick_color(self):
        result = colorchooser.askcolor(color=self._color_hex, title="Pick colour")
        if result and result[1]:
            self._color_hex = result[1]
            self.swatch.configure(fg_color=self._color_hex,
                                  hover_color=self._color_hex)
            self.on_change()

    def get_name(self)   -> str:  return self.name_var.get().strip() or "Unnamed"
    def get_color(self)  -> str:  return self._color_hex
    def get_labels(self) -> list: return [l for l, v in self._check_vars.items() if v.get()]
    def set_labels(self, labels):
        for l, v in self._check_vars.items():
            v.set(l in labels)
        self._update_badge()


#  
# GROUP EDITOR WINDOW  (with reordering + TSV import)
#  

class GroupEditorWindow(ctk.CTkToplevel):

    def __init__(self, parent_app):
        super().__init__()
        self.app = parent_app
        self.title("Behaviour Group Editor")
        self.geometry("1050x680")
        self.minsize(780, 440)
        self.resizable(True, True)

        self._group_rows: list = []
        self._all_labels: list = []

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        #   toolbar  
        tb = ctk.CTkFrame(self, fg_color=T()["card"], corner_radius=8)
        tb.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))

        ctk.CTkLabel(tb, text="Behaviour Groups",
                     font=ctk.CTkFont(size=15, weight="bold"),
                     ).pack(side="left", padx=14, pady=8)

        for label, cmd, key in [
            ("v Apply & Analyse", self._apply,       "btn_save"),
            ("Save: Save Preset",    self._save_preset,  "btn_folder"),
            ("Open: Load Preset",    self._load_preset,  "btn_load"),
            ("  Import TSV",     self._import_tsv,   "btn_unbiased"),
            ("+ Add Group",       self._add_row,      "btn_add"),
        ]:
            ctk.CTkButton(tb, text=label, command=cmd, width=138,
                          fg_color=T()[key]).pack(side="right", padx=4, pady=6)

        #   scrollable rows  
        self._scroll = ctk.CTkScrollableFrame(
            self, fg_color=T()["panel"], corner_radius=8)
        self._scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self._scroll.columnconfigure(0, weight=1)

        self.protocol("WM_DELETE_WINDOW", self.withdraw)

    #   public API  

    def refresh_labels(self, all_labels: list):
        self._all_labels = all_labels
        for row in self._group_rows:
            row.rebuild_label_grid(all_labels)

    def load_groups_from_dict(self, groups: dict, all_labels: list):
        self._all_labels = all_labels
        for row in list(self._group_rows):
            row.destroy()
        self._group_rows.clear()
        for gname, ginfo in groups.items():
            self._add_row(name=gname, color=ginfo.get("color"),
                          selected=ginfo.get("labels", []))
        # Append individual groups for any cluster not covered by the provided groups
        covered = {int(lbl) for ginfo in groups.values() for lbl in ginfo.get("labels", [])}
        for i, lbl in enumerate(sorted(all_labels)):
            if int(lbl) not in covered:
                self._add_row(
                    name=f"Cluster {lbl}",
                    color=PALETTE[(len(groups) + i) % len(PALETTE)],
                    selected=[int(lbl)],
                )

    def merge_groups_from_dict(self, groups: dict):
        """Append new groups without removing existing ones."""
        for gname, ginfo in groups.items():
            self._add_row(name=gname, color=ginfo.get("color"),
                          selected=ginfo.get("labels", []))

    def get_groups(self) -> dict:
        out  = {}
        seen = set()
        for row in self._group_rows:
            name   = row.get_name()
            labels = row.get_labels()
            if not labels:
                continue
            key = name if name not in seen else f"{name}_{len(seen)}"
            seen.add(key)
            out[key] = {"labels": labels, "color": row.get_color()}
        return out

    #   reorder helpers  

    def _move_row(self, row: "GroupRow", direction: int):
        """Move row up (-1) or down (+1)."""
        idx = self._group_rows.index(row)
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(self._group_rows):
            return
        self._group_rows[idx], self._group_rows[new_idx] = (
            self._group_rows[new_idx], self._group_rows[idx]
        )
        self._repack_rows()
        self._on_change()

    def _repack_rows(self):
        for i, r in enumerate(self._group_rows):
            r.grid(row=i, column=0, sticky="ew", padx=4, pady=3)

    #   internals  

    def _add_row(self, name=None, color=None, selected=None):
        idx   = len(self._group_rows)
        color = color or PALETTE[idx % len(PALETTE)]
        name  = name  or f"Group {idx + 1}"
        row   = GroupRow(
            self._scroll,
            group_name=name,
            color=color,
            all_labels=self._all_labels,
            selected=selected or [],
            on_change=self._on_change,
            on_delete=self._delete_row,
            on_move_up=lambda r: self._move_row(r, -1),
            on_move_down=lambda r: self._move_row(r, +1),
        )
        row.grid(row=idx, column=0, sticky="ew", padx=4, pady=3)
        self._group_rows.append(row)
        self._on_change()

    def _delete_row(self, row: GroupRow):
        row.destroy()
        self._group_rows.remove(row)
        self._repack_rows()
        self._on_change()

    def _on_change(self):
        self.app.on_groups_changed()

    def _apply(self):
        self.app.on_groups_changed()
        self.app._analyse()
        self.withdraw()

    def _import_tsv(self):
        """Import a cluster_behaviour_mapping.tsv from Video Explorer."""
        path = filedialog.askopenfilename(
            parent=self, title="Import Video Explorer TSV",
            filetypes=[("TSV files", "*.tsv"), ("All files", "*")],
        )
        if not path:
            return
        try:
            groups = load_mapping_tsv(pathlib.Path(path))
        except Exception as e:
            messagebox.showerror("TSV Import Error", str(e), parent=self)
            return
        if not groups:
            messagebox.showwarning("Empty TSV",
                "No assigned clusters found in the TSV.\n"
                "Make sure clusters are annotated in Video Explorer first.",
                parent=self)
            return
        self.load_groups_from_dict(groups, self._all_labels)
        self._on_change()
        messagebox.showinfo("Imported",
            f"Imported {len(groups)} behaviour groups from TSV.", parent=self)

    def _save_preset(self):
        groups = self.get_groups()
        if not groups:
            messagebox.showwarning("Empty", "No groups to save.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Save Group Preset",
            defaultextension=".json",
            filetypes=[("JSON preset", "*.json"), ("All files", "*")],
        )
        if not path:
            return
        data = {"groups": {
            gn: {"labels": gi["labels"], "color": gi["color"]}
            for gn, gi in groups.items()
        }}
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        messagebox.showinfo("Saved", f"Preset saved:\n{path}", parent=self)

    def _load_preset(self):
        path = filedialog.askopenfilename(
            parent=self, title="Load Group Preset",
            filetypes=[("JSON preset", "*.json"), ("All files", "*")],
        )
        if not path:
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)
            return
        groups = data.get("groups", {})
        if not groups:
            messagebox.showerror("Error", "No groups found in file.", parent=self)
            return
        self.load_groups_from_dict(groups, self._all_labels)
        self._on_change()


#  
# ANIMAL LIST PANEL  (supports CSV + TSV data files + per-animal color picking)
#  

class AnimalListPanel(ctk.CTkFrame):

    def __init__(self, parent, on_change, **kw):
        super().__init__(parent, fg_color=T()["panel"], corner_radius=8, **kw)
        self.on_change   = on_change
        self._animals: list = []
        self._uid_counter: int = 0
        # per-exp-group color overrides { eg_name: hex }
        self._eg_colors: dict = {}

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        #   toolbar  
        tb = ctk.CTkFrame(self, fg_color=T()["card"])
        tb.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))
        ctk.CTkLabel(tb, text="Animals & Exp Groups",
                     font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10, pady=6)
        ctk.CTkButton(tb, text="  EG Colors", width=100, fg_color=T()["btn_load"],
                      command=self._edit_eg_colors).pack(side="right", padx=4, pady=4)
        ctk.CTkButton(tb, text="+ Add CSV/TSV", width=110, fg_color=T()["btn_add"],
                      command=self._add).pack(side="right", padx=4, pady=4)
        ctk.CTkButton(tb, text="  Remove Selected", width=130, fg_color=T()["btn_del"],
                      command=self._remove_selected).pack(side="right", padx=4, pady=4)
        ctk.CTkButton(tb, text="Fill Selected", width=100, fg_color=T()["btn_load"],
                      command=self._fill_selected).pack(side="right", padx=4, pady=4)

        #   column header  
        hdr = ctk.CTkFrame(self, fg_color=T()["hdr_bg"])
        hdr.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 0))
        for col_txt, col_w in [("", 24), ("#", 28), ("Animal", 130),
                                ("Label 1", 90), ("Label 2", 80), ("Label 3", 80), ("FPS", 46)]:
            ctk.CTkLabel(hdr, text=col_txt, width=col_w, anchor="w",
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=T()["hdr_text"],
                         ).pack(side="left", padx=3, pady=4)

        #   scrollable rows  
        self._rows_frame = ctk.CTkScrollableFrame(
            self, fg_color=T()["panel"], corner_radius=0)
        self._rows_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=(0, 2))
        self._rows_frame.columnconfigure(0, weight=1)

        #   info label  
        self._info_lbl = ctk.CTkLabel(
            self, text="No animals loaded.",
            text_color=T()["subtext"], font=ctk.CTkFont(size=11))
        self._info_lbl.grid(row=3, column=0, padx=8, pady=4, sticky="w")

    def _edit_eg_colors(self):
        """Open a small dialog to assign custom colors to each experimental group."""
        egs = list({self._make_compound(a) for a in self._animals})
        if not egs:
            messagebox.showinfo("No groups", "Load animals and set Exp Groups first.")
            return
        win = ctk.CTkToplevel(self)
        win.title("Experimental Group Colors")
        win.geometry("340x" + str(60 + len(egs) * 52))
        win.resizable(False, False)
        win.grab_set()
        swatches = {}
        for i, eg in enumerate(sorted(egs)):
            cur_color = self._eg_colors.get(eg, PALETTE[i % len(PALETTE)])
            row_fr = ctk.CTkFrame(win, fg_color=T()["card"], corner_radius=6)
            row_fr.pack(fill="x", padx=12, pady=4)
            ctk.CTkLabel(row_fr, text=eg, width=160, anchor="w"
                         ).pack(side="left", padx=10, pady=8)
            sw = ctk.CTkButton(row_fr, text="", width=36, height=28,
                               fg_color=cur_color, hover_color=cur_color,
                               corner_radius=5)
            sw.pack(side="right", padx=10)
            swatches[eg] = {"btn": sw, "color": tk.StringVar(value=cur_color)}

            def _pick(eg=eg):
                res = colorchooser.askcolor(
                    color=swatches[eg]["color"].get(),
                    title=f"Colour for {eg}")
                if res and res[1]:
                    swatches[eg]["color"].set(res[1])
                    swatches[eg]["btn"].configure(fg_color=res[1],
                                                   hover_color=res[1])
            sw.configure(command=_pick)

        def _apply():
            for eg, d in swatches.items():
                self._eg_colors[eg] = d["color"].get()
            win.destroy()

        ctk.CTkButton(win, text="Apply", command=_apply,
                      fg_color=T()["btn_save"]).pack(pady=8)

    def get_eg_colors(self) -> dict:
        return dict(self._eg_colors)

    #   load / remove  

    def _add(self):
        paths = filedialog.askopenfilenames(
            title="Select CUBE CSV or TSV files",
            filetypes=[("CSV/TSV files", "*.csv *.tsv"), ("CSV files", "*.csv"),
                       ("TSV files", "*.tsv"), ("All files", "*")],
        )
        added = 0
        for p in paths:
            path = pathlib.Path(p)
            try:
                df  = load_csv(path)
                fps = extract_fps(path)
                uid = self._uid_counter
                self._uid_counter += 1
                entry = {
                    "uid":       uid,
                    "name":      path.stem,
                    "path":      path,
                    "df":        df,
                    "fps":       fps,
                    "exp_group": tk.StringVar(master=self, value="Control"),
                    "label2":    tk.StringVar(master=self, value=""),
                    "label3":    tk.StringVar(master=self, value=""),
                    "_selected": tk.BooleanVar(master=self, value=False),
                    "_row_frame": None,
                }
                self._animals.append(entry)
                self._build_row(entry, len(self._animals) - 1)
                added += 1
            except Exception as e:
                messagebox.showerror("Load Error", f"{path.name}:\n{e}")
        if added:
            self._update_info()
            self.on_change()

    def _build_row(self, entry: dict, idx: int):
        fg = T()["row_even"] if idx % 2 == 0 else T()["row_odd"]
        rf = ctk.CTkFrame(self._rows_frame, fg_color=fg, corner_radius=4)
        rf.pack(fill="x", pady=1, padx=2)
        entry["_row_frame"] = rf
        ctk.CTkCheckBox(rf, text="", variable=entry["_selected"],
                        width=24, checkbox_width=16, checkbox_height=16,
                        ).pack(side="left", padx=(6, 2), pady=6)
        ctk.CTkLabel(rf, text=f"{entry['uid']}", width=28, anchor="center",
                     text_color=T()["muted"], font=ctk.CTkFont(size=10),
                     ).pack(side="left", padx=2)
        ctk.CTkLabel(rf, text=entry["name"][:22], width=130, anchor="w",
                     text_color=T()["text"], font=ctk.CTkFont(size=11),
                     ).pack(side="left", padx=4)
        eg_entry = ctk.CTkEntry(rf, textvariable=entry["exp_group"],
                               width=90, placeholder_text="e.g. Control")
        eg_entry.pack(side="left", padx=4, pady=4)
        entry["_eg_entry"] = eg_entry
        l2_entry = ctk.CTkEntry(rf, textvariable=entry["label2"],
                                width=80, placeholder_text="optional")
        l2_entry.pack(side="left", padx=3, pady=4)
        entry["_label2_entry"] = l2_entry
        l3_entry = ctk.CTkEntry(rf, textvariable=entry["label3"],
                                width=80, placeholder_text="optional")
        l3_entry.pack(side="left", padx=3, pady=4)
        entry["_label3_entry"] = l3_entry

        _NAV_COLS = ["_eg_entry", "_label2_entry", "_label3_entry"]

        def _bind_nav(widget, col_key, _e=entry):
            def _nav(event, _entry=_e, _col=col_key):
                sym = event.keysym
                idx = next((j for j, a in enumerate(self._animals)
                            if a is _entry), -1)
                if idx < 0:
                    return
                c = _NAV_COLS.index(_col)

                def _focus(animal, col):
                    w = animal.get(col)
                    if w and w.winfo_exists():
                        w.focus_set()
                        w.select_range(0, "end")

                if sym in ("Down", "Return"):
                    if idx + 1 < len(self._animals):
                        _focus(self._animals[idx + 1], _col)
                    return "break"
                elif sym == "Up":
                    if idx > 0:
                        _focus(self._animals[idx - 1], _col)
                    return "break"
                elif sym == "Right" and c < len(_NAV_COLS) - 1:
                    try:
                        _tk_entry = getattr(widget, "_entry", widget)
                        at_end = (_tk_entry.index("insert") >= len(widget.get()))
                    except Exception:
                        at_end = True
                    if at_end:
                        _focus(_entry, _NAV_COLS[c + 1])
                        return "break"
                elif sym == "Left" and c > 0:
                    try:
                        _tk_entry = getattr(widget, "_entry", widget)
                        at_start = (_tk_entry.index("insert") == 0)
                    except Exception:
                        at_start = True
                    if at_start:
                        _focus(_entry, _NAV_COLS[c - 1])
                        return "break"

            widget.bind("<Down>",   _nav)
            widget.bind("<Up>",     _nav)
            widget.bind("<Return>", _nav)
            widget.bind("<Right>",  _nav)
            widget.bind("<Left>",   _nav)

        _bind_nav(eg_entry,  "_eg_entry")
        _bind_nav(l2_entry,  "_label2_entry")
        _bind_nav(l3_entry,  "_label3_entry")
        ctk.CTkLabel(rf, text=f"{entry['fps']}Hz", width=46, anchor="center",
                     text_color=T()["subtext"], font=ctk.CTkFont(size=10),
                     ).pack(side="left", padx=2)
        ctk.CTkButton(rf, text=" ", width=28, height=24,
                      fg_color=T()["btn_del"], hover_color=T()["btn_del_h"],
                      command=lambda e=entry: self._remove_entry(e),
                      ).pack(side="right", padx=(2, 6), pady=4)

    def _remove_selected(self):
        to_remove = [a for a in self._animals if a["_selected"].get()]
        if not to_remove:
            messagebox.showinfo("Remove",
                "Tick the checkbox next to animals to remove them.", parent=self)
            return
        for entry in to_remove:
            self._remove_entry(entry)

    def _fill_selected(self):
        """Bulk-assign Label 1 / Label 2 / Label 3 to all selected animals.
        Any field left blank in the dialog is skipped (not overwritten)."""
        targets = [a for a in self._animals if a["_selected"].get()]
        if not targets:
            messagebox.showinfo("Fill Selected",
                "Tick the checkbox next to animals you want to fill.", parent=self)
            return

        dlg = ctk.CTkToplevel(self)
        dlg.title("Fill Selected Animals")
        dlg.resizable(False, False)
        dlg.grab_set()

        v1 = tk.StringVar()
        v2 = tk.StringVar()
        v3 = tk.StringVar()

        for row_i, (lbl, var, ph) in enumerate([
            ("Label 1", v1, "leave blank to keep current"),
            ("Label 2", v2, "leave blank to keep current"),
            ("Label 3", v3, "leave blank to keep current"),
        ]):
            ctk.CTkLabel(dlg, text=lbl, anchor="w").grid(
                row=row_i, column=0, padx=(12, 4), pady=6, sticky="w")
            ctk.CTkEntry(dlg, textvariable=var, width=200, placeholder_text=ph).grid(
                row=row_i, column=1, padx=(4, 12), pady=6)

        def _apply():
            l1 = v1.get().strip()
            l2 = v2.get().strip()
            l3 = v3.get().strip()
            for a in targets:
                if l1:
                    a["exp_group"].set(l1)
                if l2:
                    a["label2"].set(l2)
                if l3:
                    a["label3"].set(l3)
            self._update_info()
            dlg.destroy()

        ctk.CTkButton(dlg, text="Apply", command=_apply,
                      fg_color=T()["btn_add"]).grid(
            row=3, column=0, columnspan=2, pady=(4, 12))

    def _remove_entry(self, entry: dict):
        if entry in self._animals:
            self._animals.remove(entry)
            if entry["_row_frame"] and entry["_row_frame"].winfo_exists():
                entry["_row_frame"].destroy()
            self._update_info()
            self.on_change()

    def _update_info(self):
        n   = len(self._animals)
        egs = {self._make_compound(a) for a in self._animals}
        if n == 0:
            self._info_lbl.configure(text="No animals loaded.")
        else:
            eg_txt = f"  -  {len(egs)} exp group{'s' if len(egs) != 1 else ''}"
            self._info_lbl.configure(
                text=f"{n} animal{'s' if n != 1 else ''} loaded{eg_txt}")

    def _read_eg(self, a: dict) -> str:
        """Read Label 1 (exp_group) from the entry widget."""
        eg_widget = a.get("_eg_entry")
        if eg_widget and eg_widget.winfo_exists():
            return eg_widget.get().strip() or "Default"
        return a["exp_group"].get().strip() or "Default"

    def _read_extra_label(self, a: dict, key: str) -> str:
        """Read an extra label (label2 or label3) from its entry widget."""
        w = a.get(f"_{key}_entry")
        if w and w.winfo_exists():
            return w.get().strip()
        return a[key].get().strip()

    def _make_compound(self, a: dict) -> str:
        """Combine Label 1, 2, and 3 (if set) into a single ' | '-separated string."""
        parts = [self._read_eg(a)]
        for key in ("label2", "label3"):
            val = self._read_extra_label(a, key)
            if val:
                parts.append(val)
        return " | ".join(parts)

    @staticmethod
    def _label_key_to_str(label_key: str) -> str:
        """Convert dropdown display label to internal group_by key."""
        return {"Label 1": "label1", "Label 2": "label2",
                "Label 3": "label3", "All Labels": "all"}.get(label_key, "label1")

    def get_animals(self, group_by: str = "label1") -> list:
        """group_by controls which label column becomes exp_group:
        'label1' = Label 1 (default), 'label2' = Label 2, 'label3' = Label 3,
        'all' = all three joined with ' | '."""
        out = []
        for a in self._animals:
            if group_by == "all":
                eg = self._make_compound(a)
            elif group_by == "label2":
                eg = self._read_extra_label(a, "label2") or self._read_eg(a)
            elif group_by == "label3":
                eg = self._read_extra_label(a, "label3") or self._read_eg(a)
            else:
                eg = self._read_eg(a)
            out.append({
                "uid":       a["uid"],
                "name":      a["name"],
                "path":      a["path"],
                "df":        a["df"],
                "fps":       a["fps"],
                "exp_group": eg,
            })
        return out

    def animal_count(self) -> int:
        return len(self._animals)

    def clear_all(self):
        """Remove all animals (used when loading a new folder)."""
        for entry in list(self._animals):
            if entry["_row_frame"] and entry["_row_frame"].winfo_exists():
                entry["_row_frame"].destroy()
        self._animals.clear()
        self._update_info()

    def add_files_from_paths(self, paths: list):
        """Programmatically add files without showing a dialog."""
        if not paths:
            return
        from concurrent.futures import ThreadPoolExecutor

        def _read(idx_path):
            idx, path = idx_path
            return idx, path, load_csv(path), extract_fps(path)

        # Phase 1: parallel I/O — executor shuts down (wait=True) on exit,
        # joining all threads before we touch any tk widgets.
        read_results = {}
        with ThreadPoolExecutor(max_workers=min(8, len(paths))) as ex:
            futures = {ex.submit(_read, (i, p)): i for i, p in enumerate(paths)}
            for fut in futures:
                try:
                    idx, path, df, fps = fut.result()
                    read_results[idx] = (path, df, fps)
                except Exception:
                    pass
            futures.clear()
        # all worker threads are done here

        # Phase 2: UI construction on the main thread, in original path order
        added = 0
        for idx in sorted(read_results):
            path, df, fps = read_results[idx]
            uid = self._uid_counter
            self._uid_counter += 1
            entry = {
                "uid":        uid,
                "name":       path.stem,
                "path":       path,
                "df":         df,
                "fps":        fps,
                "exp_group":  tk.StringVar(master=self, value="Control"),
                "label2":     tk.StringVar(master=self, value=""),
                "label3":     tk.StringVar(master=self, value=""),
                "_selected":  tk.BooleanVar(master=self, value=False),
                "_row_frame": None,
            }
            self._animals.append(entry)
            self._build_row(entry, len(self._animals) - 1)
            added += 1
        read_results.clear()

        if added:
            self._update_info()
            self.on_change()


#  
# UNBIASED ANALYTICS TAB PANEL
#  

class UnbiasedAnalyticsPanel(ctk.CTkFrame):
    """
    Standalone tab panel for automated, data-driven comparison.
    Operates independently of user-defined manual groups.

    After reclustering, the user can save any k as a group preset and
    immediately use it in Combined Analysis via load_groups_to_editor_fn.
    """

    # Maximum number of C(n,k) combinations before falling back to greedy.
    # C(24,4)=10,626 and C(24,3)=2,024 are well under this limit.
    EXHAUSTIVE_COMBO_LIMIT = 15_000

    def __init__(self, parent, get_animals_fn, load_groups_to_editor_fn=None,
                 get_groups_fn=None, get_combined_fn=None, **kw):
        super().__init__(parent, fg_color=T()["panel"], **kw)
        self._get_animals           = get_animals_fn
        self._load_groups_to_editor = load_groups_to_editor_fn
        self._get_groups_fn         = get_groups_fn   # () -> {group_name: {labels, color}}
        self._get_combined_fn       = get_combined_fn  # () -> combined dict or None
        self._stats_df:    pd.DataFrame | None = None
        self._recluster:   dict | None          = None
        self._raw_preview: dict | None          = None  # pre-recluster transition preview
        self._last_animals: list | None         = None  # inputs of the last stats run
        self._last_metric:  str | None          = None
        self._excluded_clusters: set            = set() # clusters removed by bio filter
        # Stats display/export options (S3/S4): parametric ANOVA/eta-squared are
        # hidden by default; family-wide FDR table is written alongside exports.
        self._show_parametric = False

        self.columnconfigure(0, weight=0, minsize=296)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_controls()
        self._build_plot_area()

    #   controls (left panel)  

    def _build_controls(self):
        ctrl = ctk.CTkScrollableFrame(self, fg_color=T()["bg"],
                                       corner_radius=0, width=294)
        ctrl.grid(row=0, column=0, sticky="nsew", padx=(6, 2), pady=6)
        ctrl.columnconfigure(0, weight=1)

        def section(parent, title):
            fr = ctk.CTkFrame(parent, fg_color=T()["card"], corner_radius=8)
            fr.pack(fill="x", padx=4, pady=4)
            ctk.CTkLabel(fr, text=title,
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=T()["hdr_text"],
                         ).pack(anchor="w", padx=10, pady=(8, 2))
            return fr

        #   Statistical settings
        sf = section(ctrl, "Statistical Settings")
        ctk.CTkLabel(sf, text="Group by:", text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._group_by_var = ctk.StringVar(value="Label 1")
        ctk.CTkOptionMenu(sf, variable=self._group_by_var,
                          values=["Label 1", "Label 2", "Label 3", "All Labels"],
                          width=240).pack(padx=12, pady=(2, 4))
        ctk.CTkLabel(sf, text="Metric:", text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._metric_var = ctk.StringVar(value="total_duration")
        ctk.CTkOptionMenu(sf, variable=self._metric_var,
                          values=["total_duration", "frequency", "mean_bout",
                                  "transition_prob"],
                          width=240).pack(padx=12, pady=(2, 4))
        ctk.CTkLabel(sf, text="p-value threshold:", text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._pthresh_var = ctk.StringVar(value="0.05")
        ctk.CTkEntry(sf, textvariable=self._pthresh_var, width=100
                     ).pack(anchor="w", padx=12, pady=(2, 8))

        #   Top-N  
        nf = section(ctrl, "Top-N Clusters")
        ctk.CTkLabel(nf, text="N =", text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._topn_var    = ctk.StringVar(value="10")
        self._topn_slider = ctk.CTkSlider(nf, from_=1, to=30, number_of_steps=29,
                                           command=self._slider_topn, width=240)
        self._topn_slider.set(10)
        self._topn_slider.pack(padx=12, pady=(2, 0))
        self._topn_lbl = ctk.CTkLabel(nf, text="Top 10 clusters",
                                       text_color=T()["subtext"],
                                       font=ctk.CTkFont(size=10))
        self._topn_lbl.pack(anchor="w", padx=12, pady=(0, 8))

        #   Biological Relevance Filter
        df_sec = section(ctrl, "Biological Relevance Filter")
        ctk.CTkLabel(df_sec,
                     text="Remove clusters that are too brief or\n"
                          "too rare to be biologically meaningful.\n"
                          "Applied by both Run buttons below.\n"
                          "Leave blank to disable each filter.",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=10),
                     justify="left").pack(anchor="w", padx=12, pady=(2, 6))

        ctk.CTkLabel(df_sec, text="Min mean bout duration (ms):",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._min_mean_ms_var = ctk.StringVar(value="")
        ctk.CTkEntry(df_sec, textvariable=self._min_mean_ms_var,
                     width=120, placeholder_text="e.g. 200").pack(anchor="w", padx=12, pady=(2, 4))

        ctk.CTkLabel(df_sec, text="Min total duration (s, per animal):",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._min_total_s_var = ctk.StringVar(value="")
        ctk.CTkEntry(df_sec, textvariable=self._min_total_s_var,
                     width=120, placeholder_text="e.g. 2").pack(anchor="w", padx=12, pady=(2, 4))

        ctk.CTkLabel(df_sec, text="Min frequency (bouts per animal):",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._min_freq_var = ctk.StringVar(value="")
        ctk.CTkEntry(df_sec, textvariable=self._min_freq_var,
                     width=120, placeholder_text="e.g. 5").pack(anchor="w", padx=12, pady=(2, 8))

        ctk.CTkButton(ctrl, text="   Run Statistical Analysis",
                      command=self._run_stats,
                      fg_color=T()["btn_unbiased"],
                      height=36, font=ctk.CTkFont(size=12, weight="bold"),
                      ).pack(fill="x", padx=8, pady=6)

        #   Reclustering
        rf_sec = section(ctrl, "Automated Reclustering")

        ctk.CTkLabel(rf_sec,
                     text="Clusters are merged by pattern of change\nacross conditions "
                          "(correlation distance).\nAbsolute magnitude is ignored.",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=10),
                     justify="left").pack(anchor="w", padx=12, pady=(2, 6))

        ctk.CTkLabel(rf_sec, text="Max k (merged groups):",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._maxk_var = ctk.StringVar(value="10")
        ctk.CTkEntry(rf_sec, textvariable=self._maxk_var, width=80
                     ).pack(anchor="w", padx=12, pady=(2, 4))

        ctk.CTkLabel(rf_sec, text="Compare k values (comma-sep):",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._compare_k_var = ctk.StringVar(value="3,5,8")
        ctk.CTkEntry(rf_sec, textvariable=self._compare_k_var, width=200
                     ).pack(anchor="w", padx=12, pady=(2, 8))

        ctk.CTkButton(ctrl, text="   Run Reclustering",
                      command=self._run_reclustering,
                      fg_color=T()["btn_folder"], height=34,
                      ).pack(fill="x", padx=8, pady=4)

        #   Save reclustered groups
        sg_sec = section(ctrl, "Save Reclustered Groups")
        ctk.CTkLabel(sg_sec,
                     text="Choose k, then save as a Group Preset\n"
                          "or push directly to the Group Editor\nfor Combined Analysis.",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=10),
                     justify="left").pack(anchor="w", padx=12, pady=(2, 6))

        ctk.CTkLabel(sg_sec, text="k to save:", text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(2, 0))
        self._save_k_var = ctk.StringVar(value="5")
        ctk.CTkEntry(sg_sec, textvariable=self._save_k_var, width=80
                     ).pack(anchor="w", padx=12, pady=(2, 8))

        ctk.CTkButton(ctrl, text="Save:  Save Groups as Preset (.json)",
                      command=self._save_groups_preset,
                      fg_color=T()["btn_save"],
                      ).pack(fill="x", padx=8, pady=(4, 2))

        ctk.CTkButton(ctrl, text="->  Push Groups to Group Editor",
                      command=self._push_groups_to_editor,
                      fg_color=T()["btn_folder"],
                      ).pack(fill="x", padx=8, pady=(2, 2))

        ctk.CTkButton(ctrl, text="Save:  Save All Graphs to Folder",
                      command=self._save_all_graphs,
                      fg_color=T()["btn_save"],
                      ).pack(fill="x", padx=8, pady=(2, 4))

        #   status
        self._status_lbl = ctk.CTkLabel(ctrl, text="",
                                         text_color=T()["subtext"],
                                         wraplength=268, justify="left",
                                         font=ctk.CTkFont(size=10))
        self._status_lbl.pack(padx=8, pady=4, anchor="w")

        #   export stats
        ctk.CTkButton(ctrl, text="Save: Export Stats CSV",
                      command=self._export_stats,
                      fg_color=T()["btn_save"],
                      ).pack(fill="x", padx=8, pady=(4, 2))

    #   plot area

    def _build_plot_area(self):
        right = ctk.CTkFrame(self, fg_color=T()["panel"])
        right.grid(row=0, column=1, sticky="nsew", padx=(2, 6), pady=6)
        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=0)
        right.rowconfigure(1, weight=1)

        sel_fr = ctk.CTkFrame(right, fg_color=T()["card"], corner_radius=8)
        sel_fr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 2))
        ctk.CTkLabel(sel_fr, text="View:",
                     font=ctk.CTkFont(size=11)).pack(side="left", padx=8, pady=6)
        self._plot_mode = ctk.StringVar(value="Top-N Bar")
        ctk.CTkSegmentedButton(
            sel_fr,
            values=["Top-N Bar", "Volcano", "Heatmap",
                    "Elbow/Silhouette", "Recombination",
                    "Dist Matrix", "Transitions", "Grp Transitions",
                    "Cluster Stats"],
            variable=self._plot_mode,
            command=self._switch_plot,
        ).pack(side="left", padx=4, pady=6)
        ctk.CTkButton(sel_fr, text="Save Graph",
                      command=self._save_current_graph,
                      fg_color=T()["btn_save"], width=100,
                      ).pack(side="right", padx=6, pady=6)

        # Scrollable plot area — supports tall multi-panel figures
        self._plot_scroll_canvas = tk.Canvas(
            right, bg=T()["fig_bg"], highlightthickness=0)
        self._plot_scroll_ysb = ctk.CTkScrollbar(
            right, command=self._plot_scroll_canvas.yview)
        self._plot_scroll_canvas.configure(
            yscrollcommand=self._plot_scroll_ysb.set)
        self._plot_scroll_canvas.grid(
            row=1, column=0, sticky="nsew", padx=(4, 0), pady=(2, 4))
        self._plot_scroll_ysb.grid(
            row=1, column=1, sticky="ns", pady=(2, 4))

        self._plot_inner_frame = ctk.CTkFrame(
            self._plot_scroll_canvas, fg_color=T()["panel"])
        self._plot_scroll_window = self._plot_scroll_canvas.create_window(
            (0, 0), window=self._plot_inner_frame, anchor="nw")

        def _on_plot_frame_configure(e):
            self._plot_scroll_canvas.configure(
                scrollregion=self._plot_scroll_canvas.bbox("all"))
        self._plot_inner_frame.bind("<Configure>", _on_plot_frame_configure)

        def _on_plot_canvas_configure(e):
            self._plot_scroll_canvas.itemconfig(
                self._plot_scroll_window, width=e.width)
        self._plot_scroll_canvas.bind("<Configure>", _on_plot_canvas_configure)

        def _on_mousewheel(e):
            self._plot_scroll_canvas.yview_scroll(
                int(-1 * (e.delta / 120)), "units")
        self._plot_scroll_canvas.bind("<MouseWheel>", _on_mousewheel)

        self._current_figure = None
        self._current_mpl_canvas = None
        self._show_placeholder(
            "Load animals in the Combined Analysis tab,\n"
            "then click   Run Statistical Analysis.")

    #   plot display helpers

    def _show_placeholder(self, msg: str):
        for w in self._plot_inner_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(self._plot_inner_frame, text=msg,
                     text_color=T()["muted"],
                     font=ctk.CTkFont(size=14),
                     justify="center").pack(pady=40, padx=20)

    def _show_figure(self, fig):
        if self._current_figure and self._current_figure is not fig:
            plt.close(self._current_figure)
        self._current_figure = fig
        for w in self._plot_inner_frame.winfo_children():
            w.destroy()
        self._current_mpl_canvas = FigureCanvasTkAgg(
            fig, master=self._plot_inner_frame)
        self._current_mpl_canvas.draw()
        widget = self._current_mpl_canvas.get_tk_widget()
        widget.pack(fill="x", expand=True, padx=2, pady=(0, 2))
        tb_fr = ctk.CTkFrame(self._plot_inner_frame,
                             fg_color=T()["card2"], height=30)
        tb_fr.pack(fill="x")
        NavigationToolbar2Tk(self._current_mpl_canvas, tb_fr)

        def _bind_wheel(w):
            w.bind("<MouseWheel>",
                   lambda e: self._plot_scroll_canvas.yview_scroll(
                       int(-1 * (e.delta / 120)), "units"))
            for child in w.winfo_children():
                _bind_wheel(child)
        _bind_wheel(widget)
        self._plot_scroll_canvas.yview_moveto(0)

    def _save_current_graph(self):
        if self._current_figure is None:
            messagebox.showwarning("No graph", "Run an analysis first.")
            return
        mode = self._plot_mode.get().replace("/", "_").replace(" ", "_").lower()
        path = filedialog.asksaveasfilename(
            title="Save Current Graph",
            initialfile=f"unbiased_{mode}.png",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"),
                       ("SVG", "*.svg"), ("All", "*")],
        )
        if not path:
            return
        try:
            self._current_figure.savefig(
                path, dpi=200, bbox_inches="tight",
                facecolor=self._current_figure.get_facecolor())
            messagebox.showinfo("Saved", f"Graph saved:\n{path}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    def _save_all_graphs(self, out_dir: "pathlib.Path | None" = None):
        """Generate and save all graph modes to a directory.

        If *out_dir* is supplied the file dialog is skipped and the graphs are
        written directly there (used by the top-level Save All Results command).
        When called interactively (out_dir is None) the standard folder dialog is
        shown and a warning is raised if no data is available.
        """
        if out_dir is None:
            if self._stats_df is None and self._recluster is None:
                messagebox.showwarning(
                    "No data", "Run Statistical Analysis or Reclustering first.")
                return
            directory = filedialog.askdirectory(title="Choose folder to save all graphs")
            if not directory:
                return
            out = pathlib.Path(directory)
        else:
            out = pathlib.Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
        animals  = self._apply_bio_filter(
            self._get_animals(group_by=self._group_by_key()))
        p_thresh, top_n, metric = self._get_params()
        eg_colors = {}
        try:
            parent = self.master
            while parent is not None:
                if hasattr(parent, "_animal_panel"):
                    eg_colors = parent._animal_panel.get_eg_colors()
                    break
                parent = getattr(parent, "master", None)
        except Exception:
            pass

        user_groups   = self._get_user_groups()
        user_combined = self._get_user_combined()

        # Build raw preview if recluster not available (for raw-ok modes)
        recluster_data = self._recluster
        if recluster_data is None and len(animals) > 0:
            try:
                recluster_data = (self._raw_preview
                                  or build_raw_recluster_result(animals))
            except Exception:
                recluster_data = None

        saved, skipped = [], []
        # (fname, mode_label, needs_stats, needs_full_recluster, raw_ok)
        modes = {
            "top_n_bar":          ("Top-N Bar",          True,  False, False),
            "volcano":            ("Volcano",             True,  False, False),
            "heatmap":            ("Heatmap",             True,  False, False),
            "elbow_silhouette":   ("Elbow/Silhouette",    False, True,  False),
            "recombination":      ("Recombination",       False, True,  False),
            "dist_matrix":        ("Dist Matrix",         False, False, True),
            "transitions":        ("Transitions",         False, False, True),
            "cluster_stats":      ("Cluster Stats",       False, False, True),
        }
        for fname, (mode, needs_stats, needs_full_rc, raw_ok) in modes.items():
            if needs_stats and self._stats_df is None:
                skipped.append(mode); continue
            if needs_full_rc and self._recluster is None:
                skipped.append(mode); continue
            if raw_ok and recluster_data is None:
                skipped.append(mode); continue
            try:
                figs_to_save = []
                if mode == "Top-N Bar":
                    result = build_top_n_barplot(
                        self._stats_df, animals, top_n, p_thresh, metric,
                        eg_colors or None,
                        groups=None if metric == "transition_prob" else user_groups or None,
                        combined=user_combined)
                    figs_to_save = list(result) if isinstance(result, tuple) else [result]
                elif mode == "Volcano":
                    figs_to_save = [build_volcano_figure(
                        self._stats_df, top_n, p_thresh,
                        groups=user_groups or None,
                        metric=metric)]
                elif mode == "Heatmap":
                    figs_to_save = [build_heatmap_figure(
                        self._stats_df, animals, top_n, p_thresh, metric,
                        eg_colors or None,
                        groups=user_groups or None)]
                elif mode == "Elbow/Silhouette":
                    figs_to_save = [build_reclustering_figure(self._recluster)]
                elif mode == "Recombination":
                    raw = self._compare_k_var.get()
                    k_list = [int(x.strip()) for x in raw.split(",")
                              if x.strip().isdigit()] or [3, 5]
                    max_avail = max(self._recluster["silhouette_scores"].keys(), default=2)
                    k_list = [k for k in k_list if 2 <= k <= max_avail] or [2]
                    figs_to_save = [build_recombination_comparison_figure(
                        self._recluster, k_list, animals, metric, eg_colors or None)]
                elif mode == "Dist Matrix":
                    figs_to_save = [build_distance_matrix_figure(
                        recluster_data, groups=user_groups or None)]
                elif mode == "Transitions":
                    figs_to_save = [build_transition_figure(
                        recluster_data, groups=user_groups or None)]
                    try:
                        from cube_core import plot_umap_3d_transitions as _plot_umap_3d
                        emb_p, lbl_p = find_umap_data(out)
                        if emb_p is not None:
                            _emb   = np.load(str(emb_p))
                            _lbl   = np.load(str(lbl_p))
                            _trans = (recluster_data or {}).get("transition_result")
                            _tmat  = _trans.get("tmat_avg")    if _trans else None
                            _cids  = _trans.get("cluster_ids") if _trans else None
                            _3d_path = out / "umap_3d_transitions.html"
                            _plot_umap_3d(
                                _emb, _lbl,
                                tmat=_tmat, cluster_ids=_cids,
                                out_path=_3d_path,
                                tag="transitions",
                            )
                            saved.append("umap_3d_transitions.html")
                            saved.append("umap_3d_transitions.png")
                    except Exception:
                        pass
                elif mode == "Cluster Stats":
                    figs_to_save = [build_cluster_stats_figure(
                        recluster_data, animals, groups=user_groups or None)]
                else:
                    continue
                for fi, fig in enumerate(figs_to_save):
                    suffix = f"_{fi}" if len(figs_to_save) > 1 else ""
                    fpath  = out / f"unbiased_{fname}{suffix}.png"
                    fig.savefig(str(fpath), dpi=200, bbox_inches="tight",
                                facecolor=fig.get_facecolor())
                    plt.close(fig)
                saved.append(fname)
            except Exception as exc:
                skipped.append(f"{mode}: {exc}")

        msg = f"Saved {len(saved)} graphs to:\n{out}"
        if skipped:
            msg += f"\n\nSkipped:\n" + "\n".join(skipped)
        messagebox.showinfo("Saved", msg)

    #   callbacks

    def _slider_topn(self, val):
        n = int(val)
        self._topn_var.set(str(n))
        self._topn_lbl.configure(text=f"Top {n} clusters")

    def _status(self, msg, color=""):
        self._status_lbl.configure(text=msg, text_color=color or T()["subtext"])
        self.update_idletasks()

    def _get_params(self):
        try:
            p_thresh = float(self._pthresh_var.get())
        except ValueError:
            p_thresh = 0.05
        try:
            top_n = int(float(self._topn_var.get()))
        except ValueError:
            top_n = 10
        return p_thresh, top_n, self._metric_var.get()

    def _group_by_key(self) -> str:
        return AnimalListPanel._label_key_to_str(self._group_by_var.get())

    def _get_save_k(self) -> int:
        try:
            return max(2, int(self._save_k_var.get()))
        except ValueError:
            return 5

    def _get_dur_filter(self) -> dict:
        """Read the three Biological Relevance Filter fields into a dict."""
        dur_filter: dict = {}
        try:
            v = self._min_mean_ms_var.get().strip()
            if v:
                dur_filter["min_mean_duration_ms"] = float(v)
        except ValueError:
            pass
        try:
            v = self._min_total_s_var.get().strip()
            if v:
                dur_filter["min_total_duration_s"] = float(v)
        except ValueError:
            pass
        try:
            v = self._min_freq_var.get().strip()
            if v:
                dur_filter["min_frequency"] = float(v)
        except ValueError:
            pass
        return dur_filter

    def _apply_bio_filter(self, animals: list) -> list:
        """Return a copy of animals with excluded clusters removed from each df."""
        if not self._excluded_clusters:
            return animals
        out = []
        for ani in animals:
            df = ani["df"]
            df_filtered = df[~df["label"].isin(self._excluded_clusters)].copy()
            out.append({**ani, "df": df_filtered})
        return out

    def _run_stats(self):
        animals_raw = self._get_animals(group_by=self._group_by_key())
        # Compute exclusions from the filter fields BEFORE applying them
        self._excluded_clusters = compute_bio_exclusions(animals_raw, self._get_dur_filter())
        animals = self._apply_bio_filter(animals_raw)
        if len(animals) < 2:
            messagebox.showwarning("Need animals",
                "Add at least 2 animals (in Combined Analysis tab) first.")
            return
        egs = {a["exp_group"] for a in animals}
        if len(egs) < 2:
            messagebox.showwarning("Need groups",
                "At least 2 experimental groups required.")
            return
        if not SCIPY_OK:
            messagebox.showerror("Missing library",
                "scipy is required:\n  pip install scipy")
            return
        self._raw_preview = None  # invalidate cached raw preview
        p_thresh, top_n, metric = self._get_params()
        self._status("Running statistical tests...")
        try:
            if metric == "transition_prob":
                self._stats_df = run_cluster_statistics(
                    animals, metric="transition_prob",
                    _compute_fn=lambda ani: compute_per_pair_transition_probs(ani["df"]))
            else:
                self._stats_df = run_cluster_statistics(animals, metric)
            # Retain inputs so the export can build the family-wide FDR table
            # across all metrics without re-selecting groups.
            self._last_animals = animals
            self._last_metric  = metric
        except Exception:
            messagebox.showerror("Stats Error", traceback.format_exc())
            self._status("Error.", "#cc4444")
            return
        _sig_col, _sw = fdr_sig_column(self._stats_df)
        n_sig = int((self._stats_df[_sig_col] < p_thresh).sum())
        _unit = "transition pairs" if metric == "transition_prob" else "clusters"
        self._status(
            f"Done. {len(self._stats_df)} {_unit} tested.\n"
            f"{n_sig} significant at {_sw} < {p_thresh} (FDR-corrected).", "#88cc88")
        self._switch_plot(self._plot_mode.get())

    def _run_reclustering(self):
        animals = self._get_animals(group_by=self._group_by_key())
        if len(animals) < 2:
            messagebox.showwarning("Need animals",
                "Add at least 2 animals in Combined Analysis tab first.")
            return
        if not (SCIPY_OK and SK_OK and SCIPY_CLUSTER_OK):
            messagebox.showerror("Missing libraries",
                "scipy + scikit-learn required:\n"
                "  pip install scipy scikit-learn")
            return
        try:
            max_k = int(self._maxk_var.get())
        except ValueError:
            max_k = 10
        _, _, metric = self._get_params()

        dur_filter = self._get_dur_filter()

        filter_desc = (
            f"  min_mean_bout={dur_filter.get('min_mean_duration_ms', '-')} ms  "
            f"  min_total={dur_filter.get('min_total_duration_s', '-')} s  "
            f"  min_freq={dur_filter.get('min_frequency', '-')}"
        )
        self._raw_preview = None  # invalidate cached raw preview
        self._status(f"Running reclustering...\n(transition-aware, biologically filtered)\n{filter_desc}")
        try:
            self._recluster = compute_reclustering_suggestions(
                animals, max_k=max_k, metric=metric,
                duration_filter=dur_filter)
        except Exception as exc:
            messagebox.showerror("Reclustering Error", traceback.format_exc())
            self._status(str(exc), "#cc4444")
            return

        raw_ids      = set(self._recluster.get("all_raw_cluster_ids", []))
        kept_ids     = set(self._recluster.get("filtered_cluster_ids", self._recluster["cluster_ids"]))
        self._excluded_clusters = raw_ids - kept_ids

        n_all      = len(raw_ids)
        n_filtered = len(kept_ids)
        best_k = max(self._recluster["silhouette_scores"],
                     key=self._recluster["silhouette_scores"].get,
                     default=None)
        self._status(
            f"Reclustering complete.\n"
            f"{n_filtered}/{n_all} clusters kept after filter.\n"
            f"Best k = {best_k} (silhouette).  Set k in 'Save' to export.", "#88cc88")
        if best_k is not None:
            self._save_k_var.set(str(best_k))
            # Auto-populate compare_k field with values around best_k so the
            # Recombination view uses the user's actual data range, not hardcoded 3,5,8
            max_avail = max(self._recluster["silhouette_scores"].keys(), default=2)
            k_opts = sorted(set(
                k for k in [
                    max(2, best_k - 2),
                    max(2, best_k - 1),
                    best_k,
                    min(max_avail, best_k + 1),
                    min(max_avail, best_k + 2),
                ] if 2 <= k <= max_avail
            ))
            self._compare_k_var.set(",".join(str(k) for k in k_opts))
        self._switch_plot(self._plot_mode.get())

    def _get_user_groups(self) -> dict:
        """Return the user-defined behaviour groups, or {} if unavailable."""
        try:
            if self._get_groups_fn:
                return self._get_groups_fn() or {}
        except Exception:
            pass
        return {}

    def _get_user_combined(self):
        """Return the combined analysis result, or None if unavailable."""
        try:
            if self._get_combined_fn:
                return self._get_combined_fn()
        except Exception:
            pass
        return None

    def _switch_plot(self, mode: str):
        animals  = self._apply_bio_filter(
            self._get_animals(group_by=self._group_by_key()))
        p_thresh, top_n, metric = self._get_params()

        # Modes that require stats results
        stats_needed = mode in ("Top-N Bar", "Volcano", "Heatmap")
        # Modes that need full reclustering (no raw preview available)
        full_recluster_needed = mode in ("Elbow/Silhouette", "Recombination")
        # Modes that can run on raw animal data OR reclustered data
        raw_ok_modes = ("Dist Matrix", "Transitions", "Cluster Stats")

        if stats_needed and self._stats_df is None:
            self._show_placeholder("Run Statistical Analysis first.")
            return
        if full_recluster_needed and self._recluster is None:
            self._show_placeholder("Run Reclustering first.")
            return

        # Determine which recluster dict to pass for raw-ok modes
        recluster_data = None
        if mode in raw_ok_modes:
            if self._recluster is not None:
                recluster_data = self._recluster
            elif len(animals) > 0:
                if self._raw_preview is None:
                    try:
                        self._raw_preview = build_raw_recluster_result(animals)
                    except Exception:
                        self._show_placeholder(
                            "Could not compute raw preview.\n"
                            "Run Reclustering for transition plots.")
                        return
                recluster_data = self._raw_preview
            else:
                self._show_placeholder(
                    "Add animals in Combined Analysis tab first.")
                return

        # Retrieve eg_colors from the animal panel (if accessible)
        eg_colors = {}
        try:
            eg_colors = self._get_animals.__self__._animal_panel.get_eg_colors()  # type: ignore
        except Exception:
            pass
        if not eg_colors:
            try:
                parent = self.master
                while parent is not None:
                    if hasattr(parent, "_animal_panel"):
                        eg_colors = parent._animal_panel.get_eg_colors()
                        break
                    parent = getattr(parent, "master", None)
            except Exception:
                pass

        user_groups   = self._get_user_groups()
        user_combined = self._get_user_combined()

        try:
            if mode == "Top-N Bar":
                result = build_top_n_barplot(
                    self._stats_df, animals, top_n, p_thresh, metric,
                    eg_colors_override=eg_colors or None,
                    groups=None if metric == "transition_prob" else user_groups or None,
                    combined=user_combined)
                if isinstance(result, tuple):
                    fig_clusters, fig_groups = result
                    self._show_figure(fig_clusters)
                    _win = tk.Toplevel(self)
                    _win.title("Top-N: User-Defined Behaviour Groups")
                    _win.configure(bg=T()["bg"])
                    _c = FigureCanvasTkAgg(fig_groups, master=_win)
                    _c.draw()
                    _c.get_tk_widget().pack(fill="both", expand=True)
                    NavigationToolbar2Tk(_c, _win)
                    def _on_popup_close(_fig=fig_groups, _canvas=_c, _w=_win):
                        import matplotlib.pyplot as _plt2
                        import gc as _gc
                        # Break canvas ↔ toolbar cycle explicitly so
                        # tk.StringVar / PhotoImage __del__ runs here on the
                        # main thread rather than from a GC pass in a background
                        # thread (which would crash the Tcl interpreter).
                        try:
                            _tb = getattr(_canvas, 'toolbar', None)
                            if _tb is not None:
                                _canvas.toolbar = None
                                try:
                                    _tb.canvas = None
                                except Exception:
                                    pass
                            for _a in ('_tkphoto', 'photo'):
                                if getattr(_canvas, _a, None) is not None:
                                    try:
                                        setattr(_canvas, _a, None)
                                    except Exception:
                                        pass
                            try:
                                _canvas.figure = None
                            except Exception:
                                pass
                        except Exception:
                            pass
                        try:
                            _plt2.close(_fig)
                        except Exception:
                            pass
                        try:
                            _fig.canvas = None
                        except Exception:
                            pass
                        try:
                            _w.destroy()
                        except Exception:
                            pass
                        _gc.collect()
                    _win.protocol("WM_DELETE_WINDOW", _on_popup_close)
                else:
                    self._show_figure(result)
            elif mode == "Volcano":
                fig = build_volcano_figure(self._stats_df, top_n, p_thresh,
                                           groups=user_groups or None,
                                           metric=metric)
                self._show_figure(fig)
            elif mode == "Heatmap":
                fig = build_heatmap_figure(self._stats_df, animals,
                                            top_n, p_thresh, metric,
                                            eg_colors_override=eg_colors or None,
                                            groups=user_groups or None)
                self._show_figure(fig)
            elif mode == "Elbow/Silhouette":
                self._show_figure(build_reclustering_figure(self._recluster))
            elif mode == "Recombination":
                raw = self._compare_k_var.get()
                try:
                    k_list = [int(x.strip()) for x in raw.split(",")
                              if x.strip().isdigit()]
                except Exception:
                    k_list = [3, 5, 8]
                if not k_list:
                    k_list = [3, 5]
                max_avail = max(self._recluster["silhouette_scores"].keys(), default=2)
                k_list    = [k for k in k_list if 2 <= k <= max_avail]
                if not k_list:
                    k_list = [2]
                self._show_figure(build_recombination_comparison_figure(
                    self._recluster, k_list, animals, metric,
                    eg_colors_override=eg_colors or None))
            elif mode == "Dist Matrix":
                self._show_figure(build_distance_matrix_figure(
                    recluster_data, groups=user_groups or None))
            elif mode == "Transitions":
                self._show_figure(build_transition_figure(
                    recluster_data, groups=user_groups or None))
            elif mode == "Grp Transitions":
                if not animals:
                    self._show_placeholder(
                        "Add animals in Combined Analysis tab first.")
                    return
                _egs = {a["exp_group"] for a in animals}
                if len(_egs) < 2:
                    self._show_placeholder(
                        "Assign animals to at least 2 experimental groups\n"
                        "to use the Grp Transitions view.")
                    return
                _, top_n, _ = self._get_params()
                self._show_figure(
                    build_group_transition_comparison_figure(
                        animals,
                        eg_colors=eg_colors or {},
                        groups=user_groups or None,
                        top_n=top_n))
            elif mode == "Cluster Stats":
                self._show_figure(build_cluster_stats_figure(
                    recluster_data, animals, groups=user_groups or None))
            else:
                return
            # Show advisory status when displaying pre-recluster raw preview
            if recluster_data is not None and recluster_data.get("_is_raw_preview"):
                self._status(
                    "Showing original clusters (pre-reclustering).\n"
                    "Run Reclustering to see transition-aware merged view.", "#b8860b")
        except Exception:
            messagebox.showerror("Plot Error", traceback.format_exc())

    #   save reclustered groups  

    def _get_groups_for_k(self, k: int) -> dict | None:
        if self._recluster is None:
            messagebox.showwarning("No reclustering",
                "Run Reclustering first.")
            return None
        max_avail = max(self._recluster["silhouette_scores"].keys(), default=2)
        if not (2 <= k <= max_avail):
            messagebox.showwarning("k out of range",
                f"k must be between 2 and {max_avail}.")
            return None
        return extract_reclustered_groups(self._recluster, k)

    def _save_groups_preset(self):
        k = self._get_save_k()
        groups = self._get_groups_for_k(k)
        if groups is None:
            return
        path = filedialog.asksaveasfilename(
            title=f"Save k={k} reclustered groups as preset",
            defaultextension=".json",
            filetypes=[("JSON preset", "*.json"), ("All files", "*")],
        )
        if not path:
            return
        data = {"groups": {
            gn: {"labels": gi["labels"], "color": gi["color"]}
            for gn, gi in groups.items()
        }}
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        messagebox.showinfo("Saved",
            f"k={k} preset saved with {len(groups)} merged groups:\n{path}\n\n"
            f"Load it in the Group Editor -> Load Preset to use in Combined Analysis.")

    def _push_groups_to_editor(self):
        """Push the chosen k reclustered groups directly to the Group Editor."""
        k = self._get_save_k()
        groups = self._get_groups_for_k(k)
        if groups is None:
            return
        if self._load_groups_to_editor is None:
            messagebox.showwarning("Not available",
                "Group Editor integration not wired. Save as preset instead.")
            return
        self._load_groups_to_editor(groups)
        self._status(
            f"v k={k} groups pushed to Group Editor.\n"
            f"Switch to Combined Analysis and run it.", "#88cc88")
        messagebox.showinfo("Pushed to Editor",
            f"{len(groups)} reclustered groups (k={k}) loaded into the Group Editor.\n\n"
            f"They appear in the editor and the Combined Analysis tab will use them "
            f"when you click Run Combined Analysis.")

    def export_csv_data(self, out_dir: pathlib.Path) -> int:
        """Write stats CSV + family FDR + advanced analyses to out_dir.
        Returns number of files/folders written. Silent on partial failures."""
        if self._stats_df is None:
            return 0
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0

        out_df = drop_parametric_columns(
            self._stats_df, getattr(self, "_show_parametric", False))
        out_df.to_csv(out_dir / "stats.csv", index=False)
        n += 1

        _last_metric_sv = getattr(self, "_last_metric", None)
        try:
            if getattr(self, "_last_animals", None) and _last_metric_sv != "transition_prob":
                _by_metric = {}
                for _m in ("total_duration", "frequency", "mean_bout"):
                    try:
                        _by_metric[_m] = run_cluster_statistics(
                            self._last_animals, _m, with_posthoc=False)
                    except Exception:
                        pass
                _fam = family_wide_fdr(_by_metric)
                if not _fam.empty:
                    _fam.to_csv(out_dir / "stats_family_fdr.csv", index=False)
                    n += 1
        except Exception:
            pass

        try:
            if getattr(self, "_last_animals", None) and _last_metric_sv != "transition_prob":
                _adv_dir = out_dir / "advanced_analyses"
                _res = run_advanced_analyses(
                    self._last_animals, _adv_dir,
                    metric=getattr(self, "_last_metric", "total_duration"))
                _ok = [k for k, v in _res.items()
                       if not (isinstance(v, dict) and v.get("error"))]
                n += len(_ok)
        except Exception:
            pass

        return n

    def _export_stats(self):
        if self._stats_df is None:
            messagebox.showwarning("No data", "Run Statistical Analysis first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Stats CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*")],
        )
        if not path:
            return
        # Hide the unstable parametric columns unless explicitly requested (S4).
        out_df = drop_parametric_columns(
            self._stats_df, getattr(self, "_show_parametric", False))
        out_df.to_csv(path, index=False)
        # Family-wide FDR companion (S3): pool p-values across all metrics so
        # significance is judged over the whole reported family, not per-metric.
        # Skipped for transition_prob — pair-level and cluster-level p-values
        # should not be pooled in the same FDR family.
        _extra = ""
        _last_metric_val = getattr(self, "_last_metric", None)
        try:
            if getattr(self, "_last_animals", None) and _last_metric_val != "transition_prob":
                _metrics = ("total_duration", "frequency", "mean_bout")
                _by_metric = {}
                for _m in _metrics:
                    try:
                        _by_metric[_m] = run_cluster_statistics(
                            self._last_animals, _m, with_posthoc=False)
                    except Exception:
                        pass
                _fam = family_wide_fdr(_by_metric)
                if not _fam.empty:
                    _fam_path = pathlib.Path(path).with_name(
                        pathlib.Path(path).stem + "_family_fdr.csv")
                    _fam.to_csv(_fam_path, index=False)
                    _extra = f"\nFamily-wide FDR table:\n{_fam_path}"
        except Exception:
            pass
        # Additive advanced analyses (fingerprint, sequence, time-resolved) into
        # an 'advanced_analyses' subfolder beside the stats CSV.
        try:
            if getattr(self, "_last_animals", None) and _last_metric_val != "transition_prob":
                _adv_dir = pathlib.Path(path).parent / "advanced_analyses"
                _res = run_advanced_analyses(
                    self._last_animals, _adv_dir,
                    metric=getattr(self, "_last_metric", "total_duration"))
                _ok = [k for k, v in _res.items()
                       if not (isinstance(v, dict) and v.get("error"))]
                if _ok:
                    _extra += (f"\nAdvanced analyses ({', '.join(_ok)}):\n{_adv_dir}")
        except Exception:
            pass
        messagebox.showinfo("Saved", f"Statistics saved:\n{path}{_extra}")


def build_umap_comparison_figure(embedding, labels, new_groups: dict,
                                  t: dict = None) -> "plt.Figure":
    """
    Side-by-side UMAP scatter:
      Left  — original cluster IDs (as produced by HDBSCAN)
      Right — new combined behaviour groups after recombination

    Parameters
    ----------
    embedding   : np.ndarray (n_samples, ≥2)  – UMAP coordinates
    labels      : np.ndarray (n_samples,)      – original cluster IDs (-1 = noise)
    new_groups  : {group_name: {"labels": [int,...], "color": str}}
    """
    if t is None:
        t = T()

    import numpy as _np

    x = _np.asarray(embedding[:, 0], dtype=float)
    y = _np.asarray(embedding[:, 1], dtype=float)
    labels = _np.asarray(labels, dtype=int)

    # Guard: stale model files can produce mismatched lengths.
    if len(x) != len(labels):
        _n = min(len(x), len(labels))
        x      = x[:_n]
        y      = y[:_n]
        labels = labels[:_n]

    # Map original cluster IDs → new group index
    new_label_arr = _np.full(len(labels), -1, dtype=int)
    group_names   = list(new_groups.keys())
    for gi, gname in enumerate(group_names):
        for cid in new_groups[gname].get("labels", []):
            new_label_arr[labels == cid] = gi

    unique_orig = sorted(set(labels.tolist()))
    n_orig      = max(1, len([c for c in unique_orig if c >= 0]))
    try:
        cmap_orig = plt.cm.get_cmap("tab20", n_orig)
    except Exception:
        cmap_orig = plt.cm.get_cmap("hsv", n_orig)

    with plt.style.context(t["mpl_style"]):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6), facecolor=t["fig_bg"])

        #   Left: original HDBSCAN clusters
        orig_idx = 0
        for cl in unique_orig:
            mask = labels == cl
            if cl < 0:
                ax1.scatter(x[mask], y[mask], c="#555566", s=3,
                            alpha=0.25, linewidths=0, label="noise")
            else:
                color = cmap_orig(orig_idx / max(n_orig - 1, 1))
                ax1.scatter(x[mask], y[mask], c=[color], s=5,
                            alpha=0.65, linewidths=0, label=f"C{cl}")
                orig_idx += 1

        ax1.set_title("Original Clusters  (HDBSCAN)",
                      color=t["tick"], fontweight="bold", fontsize=12, pad=8)
        ax1.set_xlabel("UMAP-1", color=t["tick"], fontsize=10)
        ax1.set_ylabel("UMAP-2", color=t["tick"], fontsize=10)
        # Compact legend: only show if ≤ 20 clusters
        if n_orig <= 20:
            ax1.legend(fontsize=7, loc="upper right", ncol=2,
                       facecolor=t["ax_bg"], edgecolor=t["border"],
                       labelcolor=t["tick"], markerscale=2)
        _style_ax(ax1, t)

        #   Right: recombined behaviour groups
        for gi, gname in enumerate(group_names):
            mask = new_label_arr == gi
            if not mask.any():
                continue
            color = new_groups[gname].get("color", PALETTE[gi % len(PALETTE)])
            ax2.scatter(x[mask], y[mask], c=[color], s=5,
                        alpha=0.75, linewidths=0, label=gname)
        unassigned = new_label_arr == -1
        if unassigned.any():
            ax2.scatter(x[unassigned], y[unassigned], c="#555566", s=3,
                        alpha=0.2, linewidths=0, label="unassigned")

        ax2.set_title("Combined Behaviour Groups  (after recombination)",
                      color=t["tick"], fontweight="bold", fontsize=12, pad=8)
        ax2.set_xlabel("UMAP-1", color=t["tick"], fontsize=10)
        ax2.set_ylabel("UMAP-2", color=t["tick"], fontsize=10)
        ax2.legend(fontsize=8, loc="upper right", ncol=1,
                   facecolor=t["ax_bg"], edgecolor=t["border"],
                   labelcolor=t["tick"], markerscale=2)
        _style_ax(ax2, t)

        fig.suptitle("UMAP — Before vs. After Recombination",
                     color=t["tick"], fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _label_text_color(hex_color: str) -> str:
    """Return '#111111' or 'white' depending on background luminance."""
    try:
        r, g, b, _ = mcolors.to_rgba(hex_color)
        return "#111111" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "white"
    except Exception:
        return "white"


def build_energy_landscape_figure(
        animal_data: list, umap_embedding, umap_labels,
        t: dict = None, groups: dict = None) -> "plt.Figure":
    """
    Behavioral energy landscape over UMAP space — 3-column layout per group.
      col 0 – UMAP scatter (cluster colours) + translucent energy contour overlay
      col 1 – 2D filled energy heatmap (inferno_r) + top-5 valley labels (★)
      col 2 – clean 3D surface (coolwarm, single viewpoint from above)
    Z = −ln(weighted KDE density); valleys = common behaviours (low energy).
    """
    if t is None:
        t = T()

    import numpy as _np

    def _placeholder(msg):
        fig, ax = plt.subplots(figsize=(7, 3), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, msg, ha="center", va="center", color=t["tick"],
                transform=ax.transAxes, fontsize=9, multialignment="center")
        _style_ax(ax, t)
        return fig

    if umap_embedding is None or umap_labels is None:
        return _placeholder(
            "UMAP data not available.\n"
            "Run cube_core to generate umap_embedding.npy / umap_labels.npy.")

    if not SCIPY_OK:
        return _placeholder(
            "scipy is required for the Energy Landscape.\npip install scipy")

    try:
        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  # type: ignore
        from scipy.ndimage import gaussian_filter as _gfilt
        from scipy.stats  import gaussian_kde   as _gkde
    except ImportError:
        return _placeholder(
            "scipy / mpl_toolkits not available for Energy Landscape.")

    umap_x = _np.asarray(umap_embedding[:, 0], dtype=float)
    umap_y = _np.asarray(umap_embedding[:, 1], dtype=float)
    umap_l = _np.asarray(umap_labels, dtype=int)

    # Guard: older model files saved before a bug-fix could produce mismatched
    # array lengths (embedding n_bins vs labels n_samp).  Truncate to the
    # shorter so every downstream boolean mask is consistent.
    if len(umap_x) != len(umap_l):
        _n = min(len(umap_x), len(umap_l))
        umap_x = umap_x[:_n]
        umap_y = umap_y[:_n]
        umap_l = umap_l[:_n]

    # Noise exclusion: HDBSCAN labels -1 as noise; skip those points entirely.
    _noise_mask = umap_l < 0

    # cluster-id → group name mapping (from user-defined groups, if any)
    _cid_to_gname: dict = {}
    if groups:
        for _gname, _gdata in (groups or {}).items():
            for _gcid in _gdata.get("labels", []):
                _cid_to_gname[int(_gcid)] = _gname

    groups_eg: dict = {}
    for ani in animal_data:
        groups_eg.setdefault(ani["exp_group"], []).append(ani)
    eg_names = list(groups_eg.keys())
    n_eg = len(eg_names)
    if n_eg == 0:
        return _placeholder("No animals loaded.")

    GRID_N  = 200
    N_VALS  = 5      # top-N valleys to label
    ncols   = 3
    nrows   = n_eg
    fig_w   = max(22.0, ncols * 7.5)
    fig_h   = max(6.0,  nrows * 4.8 + 1.8)

    # Gentle blue→red colormap: blue = low energy/common, red = high energy/rare
    from matplotlib.colors import LinearSegmentedColormap as _LSC
    _ECMAP = _LSC.from_list("gentle_br", [
        "#3B6FA0",   # muted steel blue  (low energy, common states)
        "#89BBD9",   # sky blue
        "#D8CFC4",   # warm off-white    (mid energy)
        "#D4897A",   # muted salmon
        "#A83228",   # muted brick red   (high energy, rare states)
    ], N=256)

    x_rng = umap_x.max() - umap_x.min()
    y_rng = umap_y.max() - umap_y.min()
    # Pad 20 % beyond data extent so KDE drops to near-zero at grid edges.
    x_pad = x_rng * 0.20
    y_pad = y_rng * 0.20
    xi = _np.linspace(umap_x.min() - x_pad, umap_x.max() + x_pad, GRID_N)
    yi = _np.linspace(umap_y.min() - y_pad, umap_y.max() + y_pad, GRID_N)
    XX, YY = _np.meshgrid(xi, yi)

    # ── Per-cluster UMAP scatter colours (shared) ──────────────────────────
    scatter_colors = _np.array(
        [mcolors.to_rgba(PALETTE[int(l) % len(PALETTE)], alpha=0.65)
         for l in umap_l])

    # ── Data-support mask (shared across groups) ────────────────────────────
    try:
        _kde_support    = _gkde(_np.vstack([umap_x, umap_y]))
        _ZZ_support     = _kde_support(
            _np.vstack([XX.ravel(), YY.ravel()])).reshape(GRID_N, GRID_N)
        _support_thresh = _np.percentile(_ZZ_support, 20)
        _support_mask   = _ZZ_support < _support_thresh   # True = empty space
    except Exception:
        _support_mask = _np.zeros((GRID_N, GRID_N), dtype=bool)

    with plt.style.context(t["mpl_style"]):
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=t["fig_bg"], dpi=150)
        gs  = fig.add_gridspec(nrows, ncols, width_ratios=[1, 1.15, 1],
                               left=0.06, right=0.97, top=0.88, bottom=0.06,
                               wspace=0.30, hspace=0.55)

        for ei, eg in enumerate(eg_names):
            ani_list = groups_eg[eg]
            eg_color = PALETTE[ei % len(PALETTE)]

            # ── Per-cluster occupancy probability (noise excluded) ─────────
            cl_counts: dict = {}
            for ani in ani_list:
                for lbl in ani["df"]["label"].values:
                    k = int(lbl)
                    if k < 0:
                        continue   # skip noise frames
                    cl_counts[k] = cl_counts.get(k, 0) + 1
            total = max(sum(cl_counts.values()), 1)
            P_cl  = {k: v / total for k, v in cl_counts.items()}

            # Weight UMAP points by cluster occupancy probability; noise → 0
            weights = _np.array(
                [0.0 if int(l) < 0 else P_cl.get(int(l), 1e-9)
                 for l in umap_l], dtype=float)
            wsum = weights.sum()
            weights = weights / wsum if wsum > 0 else \
                      _np.ones_like(weights) / len(weights)

            # ── Weighted KDE → energy surface ──────────────────────────────
            try:
                kde     = _gkde(_np.vstack([umap_x, umap_y]),
                                weights=weights, bw_method="scott")
                ZZ_dens = kde(_np.vstack([XX.ravel(), YY.ravel()])
                              ).reshape(GRID_N, GRID_N)
            except Exception:
                ZZ_dens = _np.full((GRID_N, GRID_N), 1e-9)
                for cid, p in P_cl.items():
                    mask = umap_l == cid
                    if not mask.any():
                        continue
                    cx, cy = float(umap_x[mask].mean()), float(umap_y[mask].mean())
                    dist2  = (XX - cx) ** 2 + (YY - cy) ** 2
                    sigma  = max(x_rng, y_rng) * 0.08
                    ZZ_dens += p * _np.exp(-dist2 / (2 * sigma ** 2))

            ZZ_dens   = _np.maximum(ZZ_dens, 1e-12)
            ZZ_smooth = _gfilt(ZZ_dens, sigma=0.7)
            ZZ_energy = -_np.log(_np.maximum(ZZ_smooth, 1e-12))

            supported  = ~_support_mask
            E_lo       = float(_np.nanmin(ZZ_energy[supported]))
            ZZ_shifted = _np.maximum(ZZ_energy - E_lo, 0.0)
            E_hi       = float(_np.nanmax(ZZ_shifted[supported]))
            ZZ_norm    = ZZ_shifted / max(E_hi, 1e-9)
            ZZ_norm[_support_mask] = 0.0

            # NaN mask for 2D contourf so unsupported cells render as background
            ZZ_plot = ZZ_norm.copy().astype(float)
            ZZ_plot[_support_mask] = _np.nan
            # NaN mask for 3D surface so the flat padded skirt is not rendered
            ZZ_surface = ZZ_norm.copy().astype(float)
            ZZ_surface[_support_mask] = _np.nan

            # ── Top-N valley clusters (highest occupancy = lowest energy) ──
            top_n = sorted(P_cl.items(), key=lambda kv: kv[1], reverse=True)[:N_VALS]

            # ── Scatter point order (sort by cluster; noise excluded) ──────
            sort_idx = _np.argsort(umap_l)
            _valid   = umap_l[sort_idx] >= 0
            sx = umap_x[sort_idx][_valid]
            sy = umap_y[sort_idx][_valid]
            sc = scatter_colors[sort_idx][_valid]

            # ══════════════════════════════════════════════════════════════════
            # Col 0 — UMAP scatter + energy contour overlay
            # ══════════════════════════════════════════════════════════════════
            ax0 = fig.add_subplot(gs[ei, 0])
            ax0.set_facecolor(t["ax_bg"])
            # Translucent energy contour behind the scatter (subtle, no lines)
            ax0.contourf(xi, yi, ZZ_plot, levels=8, cmap=_ECMAP, alpha=0.22)
            ax0.scatter(sx, sy, c=sc, s=14, linewidths=0, alpha=0.80)

            # ── Cluster / group centroid labels (overlap-suppressed) ────────
            _vx = umap_x[~_noise_mask]
            _vy = umap_y[~_noise_mask]
            _vl = umap_l[~_noise_mask]
            _total_v = max(len(_vl), 1)
            _prox = min(x_rng, y_rng) * 0.10   # min gap between label centres
            _placed: list = []                   # (cx, cy) of placed labels

            if _cid_to_gname:
                # Group mode: one badge per group at the group's point centroid
                _done_groups: set = set()
                for _gcid in sorted(_np.unique(_vl),
                                    key=lambda c: -(_vl == c).sum()):
                    _gn = _cid_to_gname.get(int(_gcid))
                    if _gn is None or _gn in _done_groups:
                        continue
                    _gpts = _np.isin(_vl,
                        [c for c, n in _cid_to_gname.items() if n == _gn])
                    if not _gpts.any():
                        continue
                    _done_groups.add(_gn)
                    _gcx = float(_vx[_gpts].mean())
                    _gcy = float(_vy[_gpts].mean())
                    if any(_np.hypot(_gcx - px, _gcy - py) < _prox
                           for px, py in _placed):
                        continue
                    _placed.append((_gcx, _gcy))
                    _gcol = groups[_gn].get("color",
                                            PALETTE[len(_placed) % len(PALETTE)])
                    ax0.text(_gcx, _gcy, _gn,
                             color=_label_text_color(_gcol),
                             fontsize=6.5, fontweight="bold",
                             ha="center", va="center",
                             bbox=dict(boxstyle="round,pad=0.2",
                                       facecolor=_gcol, edgecolor="none",
                                       alpha=0.85),
                             zorder=10)
            else:
                # Cluster mode: badge per non-noise cluster, skip tiny/crowded
                for _gcid in sorted(_np.unique(_vl),
                                    key=lambda c: -(_vl == c).sum()):
                    _gmask = _vl == _gcid
                    if _gmask.sum() / _total_v < 0.005:
                        continue
                    _gcx = float(_vx[_gmask].mean())
                    _gcy = float(_vy[_gmask].mean())
                    if any(_np.hypot(_gcx - px, _gcy - py) < _prox
                           for px, py in _placed):
                        continue
                    _placed.append((_gcx, _gcy))
                    _gcol = PALETTE[int(_gcid) % len(PALETTE)]
                    ax0.text(_gcx, _gcy, f"C{_gcid}",
                             color=_label_text_color(_gcol),
                             fontsize=6.5, fontweight="bold",
                             ha="center", va="center",
                             bbox=dict(boxstyle="round,pad=0.2",
                                       facecolor=_gcol, edgecolor="none",
                                       alpha=0.85),
                             zorder=10)

            ax0.set_title(eg, color=eg_color,
                          fontweight="bold", fontsize=12, pad=5)
            ax0.set_xlabel("UMAP-1", color=t["tick"], fontsize=10)
            ax0.set_ylabel("UMAP-2", color=t["tick"], fontsize=10)
            ax0.tick_params(colors=t["tick"], labelsize=8)
            for sp in ("top", "right"):
                ax0.spines[sp].set_visible(False)
            for sp in ("bottom", "left"):
                ax0.spines[sp].set_edgecolor(t["spine"])
            ax0.set_xlim(umap_x.min() - x_rng * 0.02, umap_x.max() + x_rng * 0.02)
            ax0.set_ylim(umap_y.min() - y_rng * 0.02, umap_y.max() + y_rng * 0.02)

            # ══════════════════════════════════════════════════════════════════
            # Col 1 — 2D filled energy heatmap + top-N valley labels
            # ══════════════════════════════════════════════════════════════════
            ax1 = fig.add_subplot(gs[ei, 1])
            ax1.set_facecolor(t["ax_bg"])
            cf = ax1.contourf(xi, yi, ZZ_plot, levels=12, cmap=_ECMAP)
            ax1.contour(xi, yi, ZZ_plot, levels=5,
                        colors=t["tick"], linewidths=0.7, alpha=0.50)
            cbar = fig.colorbar(cf, ax=ax1, fraction=0.038, pad=0.02)
            cbar.set_label("−ln P  (energy)", color=t["tick"], fontsize=9)
            cbar.ax.tick_params(colors=t["tick"], labelsize=8)
            cbar.outline.set_edgecolor(t["spine"])

            for _rank, (cid, _p) in enumerate(top_n):
                mask = umap_l == cid
                if not mask.any():
                    continue
                cx = float(umap_x[mask].mean())
                cy = float(umap_y[mask].mean())
                ax1.scatter(cx, cy, marker="*", s=350,
                            color="#FFD700", zorder=6,
                            edgecolors=t["tick"], linewidths=1.0)
                # Annotate with a short line connecting label to the star
                ax1.annotate(
                    f"C{cid}",
                    xy=(cx, cy),
                    xytext=(cx + x_rng * 0.04, cy + y_rng * 0.06),
                    color=t["tick"], fontsize=9.5, fontweight="bold",
                    ha="center", va="bottom", zorder=8,
                    arrowprops=dict(arrowstyle="-", color=t["tick"],
                                    lw=0.8, alpha=0.7),
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc=t["ax_bg"], alpha=0.75, ec=t["spine"]),
                )

            ax1.set_title(f"Energy Heatmap  (★ = top {N_VALS} valleys)",
                          color=t["tick"], fontsize=10, pad=5)
            ax1.set_xlabel("UMAP-1", color=t["tick"], fontsize=10)
            ax1.set_ylabel("UMAP-2", color=t["tick"], fontsize=10)
            ax1.tick_params(colors=t["tick"], labelsize=8)
            for sp in ("top", "right"):
                ax1.spines[sp].set_visible(False)
            for sp in ("bottom", "left"):
                ax1.spines[sp].set_edgecolor(t["spine"])
            ax1.set_xlim(umap_x.min() - x_rng * 0.02, umap_x.max() + x_rng * 0.02)
            ax1.set_ylim(umap_y.min() - y_rng * 0.02, umap_y.max() + y_rng * 0.02)

            # ══════════════════════════════════════════════════════════════════
            # Col 2 — Clean 3D surface (single viewpoint from above)
            # ══════════════════════════════════════════════════════════════════
            ax3d = fig.add_subplot(gs[ei, 2], projection="3d")
            ax3d.set_facecolor(t["ax_bg"])

            ax3d.plot_surface(XX, YY, ZZ_surface, cmap=_ECMAP,
                              alpha=0.92, linewidth=0, antialiased=True,
                              rstride=1, cstride=1)

            # ── Two-pass valley labelling ───────────────────────────────────
            # Pass 1: find the actual local energy minimum for each top-N cluster.
            _ZZ_search = ZZ_norm.copy()
            _nan_fill  = float(_np.nanmax(ZZ_norm)) + 1.0
            _ZZ_search[_np.isnan(_ZZ_search)] = _nan_fill
            _vpts = []   # (vx, vy, vz, cid)
            for _rank, (cid, _p) in enumerate(top_n):
                mask = umap_l == cid
                if not mask.any():
                    continue
                _ix0 = max(0, int(_np.searchsorted(xi, float(umap_x[mask].min()))) - 1)
                _ix1 = min(GRID_N, int(_np.searchsorted(xi, float(umap_x[mask].max()))) + 2)
                _iy0 = max(0, int(_np.searchsorted(yi, float(umap_y[mask].min()))) - 1)
                _iy1 = min(GRID_N, int(_np.searchsorted(yi, float(umap_y[mask].max()))) + 2)
                if _ix1 <= _ix0 or _iy1 <= _iy0:
                    _cx = float(umap_x[mask].mean())
                    _cy = float(umap_y[mask].mean())
                    _ix0 = max(0, int(_np.argmin(_np.abs(xi - _cx))) - 2)
                    _ix1 = _ix0 + 5
                    _iy0 = max(0, int(_np.argmin(_np.abs(yi - _cy))) - 2)
                    _iy1 = _iy0 + 5
                _sub = _ZZ_search[_iy0:_iy1, _ix0:_ix1]
                _li  = _np.unravel_index(_np.argmin(_sub), _sub.shape)
                vx   = float(xi[_ix0 + _li[1]])
                vy   = float(yi[_iy0 + _li[0]])
                vz   = float(ZZ_norm[_iy0 + _li[0], _ix0 + _li[1]])
                _vpts.append((vx, vy, 0.0 if _np.isnan(vz) else vz, cid))

            # Pass 2: assign staggered label Z-heights for nearby valleys.
            # Proximity threshold = 15 % of the shorter UMAP axis span.
            _prox = min(x_rng, y_rng) * 0.18
            _lz   = [vz for (_, _, vz, _) in _vpts]   # base label heights
            for _i in range(len(_vpts)):
                for _j in range(_i):
                    _d = _np.hypot(_vpts[_i][0] - _vpts[_j][0],
                                   _vpts[_i][1] - _vpts[_j][1])
                    if _d < _prox:
                        _lz[_i] = max(_lz[_i], _lz[_j] + 0.16)

            # Draw dots, stems, and non-overlapping labels.
            for _i, (vx, vy, vz, cid) in enumerate(_vpts):
                stem_top = _lz[_i] + 0.07
                ax3d.scatter([vx], [vy], [vz], color="#FFD700",
                             s=40, zorder=9, depthshade=False)
                ax3d.plot([vx, vx], [vy, vy], [vz, stem_top],
                          color=t["tick"], lw=0.9, alpha=0.75, zorder=8)
                ax3d.text(vx, vy, stem_top + 0.01, f"C{cid}",
                          color=t["tick"], fontsize=9, fontweight="bold",
                          ha="center", va="bottom", zorder=10)

            ax3d.view_init(elev=38, azim=215)
            ax3d.set_xlabel("UMAP-1", color=t["tick"], fontsize=10, labelpad=4)
            ax3d.set_ylabel("UMAP-2", color=t["tick"], fontsize=10, labelpad=4)
            ax3d.set_zlabel("−ln P",  color=t["tick"], fontsize=10, labelpad=4)
            ax3d.set_title("3D Energy Surface",
                           color=t["tick"], fontsize=10, pad=5)
            ax3d.tick_params(colors=t["tick"], labelsize=7)
            for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor(t["border"])
            ax3d.grid(True, alpha=0.12, color=t["border"])

        fig.suptitle(
            "Behavioral Energy Landscape  ·  blue = common states  ·  red = rare states  ·  ★ = top valleys",
            color=t["tick"], fontsize=11, fontweight="bold", y=0.95)
    return fig


def build_umap_groups_figure(
        umap_embedding, umap_labels, groups: dict,
        t: dict = None) -> "plt.Figure":
    """
    Side-by-side 2-D UMAP scatter.
    Left panel: original clusters (coloured by PALETTE, labelled).
    Right panel: user-defined behaviour groups overlay.
    """
    if t is None:
        t = T()

    import numpy as _np

    def _placeholder(msg):
        fig, ax = plt.subplots(figsize=(7, 5), facecolor=t["fig_bg"])
        ax.text(0.5, 0.5, msg, ha="center", va="center", color=t["tick"],
                transform=ax.transAxes, fontsize=9, multialignment="center")
        _style_ax(ax, t)
        return fig

    if umap_embedding is None or umap_labels is None:
        return _placeholder(
            "UMAP data not available.\n"
            "Run cube_core to generate umap_embedding.npy / umap_labels.npy.")

    if not groups:
        return _placeholder(
            "No behaviour groups defined.\nCreate groups in the Group Editor first.")

    umap_x = _np.asarray(umap_embedding[:, 0], dtype=float)
    umap_y = _np.asarray(umap_embedding[:, 1], dtype=float)
    umap_l = _np.asarray(umap_labels, dtype=int)

    # Guard: stale model files can produce mismatched lengths (embedding n_bins
    # vs labels n_samp).  Truncate to the shorter so masks stay consistent.
    if len(umap_x) != len(umap_l):
        _n = min(len(umap_x), len(umap_l))
        umap_x = umap_x[:_n]
        umap_y = umap_y[:_n]
        umap_l = umap_l[:_n]

    group_names = list(groups.keys())
    cid_to_gi: dict = {}
    for gi, gname in enumerate(group_names):
        for cid in groups[gname].get("labels", []):
            cid_to_gi[int(cid)] = gi

    point_gi = _np.array([cid_to_gi.get(int(l), -1) for l in umap_l], dtype=int)

    with plt.style.context(t["mpl_style"]):
        fig, (ax_orig, ax_grp) = plt.subplots(
            1, 2, figsize=(16, 6.5), facecolor=t["fig_bg"])

        # ── Left panel: original clusters ──────────────────────────────────
        ax_orig.set_facecolor(t["ax_bg"])
        for cid in sorted(_np.unique(umap_l)):
            mask = umap_l == cid
            color = PALETTE[int(cid) % len(PALETTE)]
            ax_orig.scatter(umap_x[mask], umap_y[mask],
                            c=[color], s=5, alpha=0.65,
                            linewidths=0, rasterized=True, zorder=2)
            cx = float(umap_x[mask].mean())
            cy = float(umap_y[mask].mean())
            ax_orig.text(cx, cy, f"C{cid}",
                         color=_label_text_color(color),
                         fontsize=6.5, fontweight="bold",
                         ha="center", va="center",
                         bbox=dict(boxstyle="round,pad=0.2",
                                   facecolor=color, edgecolor="none", alpha=0.82),
                         zorder=10)
        ax_orig.set_xlabel("UMAP-1", color=t["tick"], fontsize=11)
        ax_orig.set_ylabel("UMAP-2", color=t["tick"], fontsize=11)
        ax_orig.set_title("UMAP — Original Clusters",
                          color=t["tick"], fontweight="bold", fontsize=13, pad=10)
        _style_ax(ax_orig, t)

        # ── Right panel: behaviour groups ───────────────────────────────────
        ax_grp.set_facecolor(t["ax_bg"])
        unassigned = point_gi < 0
        if unassigned.any():
            ax_grp.scatter(umap_x[unassigned], umap_y[unassigned],
                           c=t.get("muted", "#555566"), s=3, alpha=0.12,
                           linewidths=0, rasterized=True, zorder=1)
        for gi, gname in enumerate(group_names):
            mask = point_gi == gi
            if not mask.any():
                continue
            color = groups[gname].get("color", PALETTE[gi % len(PALETTE)])
            ax_grp.scatter(umap_x[mask], umap_y[mask],
                           c=[color], s=9, alpha=0.78,
                           linewidths=0, rasterized=True, zorder=2 + gi)
            cx = float(umap_x[mask].mean())
            cy = float(umap_y[mask].mean())
            ax_grp.annotate(
                gname, xy=(cx, cy),
                color="white", fontsize=8.5, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=color, edgecolor="none", alpha=0.82),
                zorder=12,
            )
        ax_grp.set_xlabel("UMAP-1", color=t["tick"], fontsize=11)
        ax_grp.set_ylabel("UMAP-2", color=t["tick"], fontsize=11)
        ax_grp.set_title("UMAP — Behavioural Groups",
                         color=t["tick"], fontweight="bold", fontsize=13, pad=10)
        _style_ax(ax_grp, t)

        fig.tight_layout()
    return fig


def save_umap_groups_3d(
        embedding, labels, groups: dict,
        out_path,
        tag: str = "") -> None:
    """
    Write a 3-D UMAP scatter coloured by user-defined behavioural groups.

    Point cloud is translucent; group centroids are large labelled nodes.
    Outputs <out_path>.html (interactive plotly) and <out_path>.png
    (3-viewpoint matplotlib static).  Fails silently.

    Parameters
    ----------
    embedding : (n, >=3) UMAP embedding
    labels    : (n,) cluster labels (-1 = noise)
    groups    : {group_name: {"labels": [cluster_ids], "color": hex}}
    out_path  : output path (extension replaced)
    tag       : title tag
    """
    try:
        import numpy as _np
        emb = _np.asarray(embedding)
        lbl = _np.asarray(labels, dtype=int)
        if emb.shape[1] < 3:
            return
        e3 = emb[:, :3]

        # Build point → group-index mapping
        group_names  = list(groups.keys())
        group_colors = [groups[g].get("color", PALETTE[i % len(PALETTE)])
                        for i, g in enumerate(group_names)]
        cid_to_gi: dict = {}
        for gi, gname in enumerate(group_names):
            for cid in groups[gname].get("labels", []):
                cid_to_gi[int(cid)] = gi

        point_gi = _np.array([cid_to_gi.get(int(l), -1) for l in lbl], dtype=int)

        # Group centroids in 3D
        g_centroids = {}
        for gi, gname in enumerate(group_names):
            mask = point_gi == gi
            if mask.any():
                g_centroids[gi] = e3[mask].mean(axis=0)

        title = f"3D UMAP — Behavioural Groups  [{tag}]" if tag else \
                "3D UMAP — Behavioural Groups"

        # ── Static PNG ────────────────────────────────────────────────────────
        try:
            import matplotlib.pyplot as _plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            t_cur = T()
            bg, panel, tc, tkc = (t_cur["fig_bg"], t_cur["ax_bg"],
                                  t_cur["tick"], t_cur["tick"])
            views   = [(20, 45), (20, 200), (60, 100)]
            fig_s   = _plt.figure(figsize=(18, 6), facecolor=bg)
            for vi, (elev, azim) in enumerate(views):
                ax = fig_s.add_subplot(1, 3, vi + 1, projection="3d")
                ax.set_facecolor(panel)
                unassigned = point_gi < 0
                if unassigned.any():
                    ax.scatter(e3[unassigned, 0], e3[unassigned, 1],
                               e3[unassigned, 2],
                               s=1, alpha=0.08, color="#555566",
                               depthshade=False)
                for gi, gname in enumerate(group_names):
                    mask = point_gi == gi
                    if not mask.any():
                        continue
                    col = group_colors[gi]
                    ax.scatter(e3[mask, 0], e3[mask, 1], e3[mask, 2],
                               s=1, alpha=0.30, color=col, depthshade=False)
                    if gi in g_centroids:
                        c = g_centroids[gi]
                        ax.scatter(*c, s=55, color=col,
                                   edgecolors="white", linewidths=0.5,
                                   zorder=5, depthshade=False)
                        ax.text(c[0], c[1], c[2], f" {gname}",
                                fontsize=5, color=col, zorder=6)
                ax.view_init(elev=elev, azim=azim)
                ax.set_xlabel("UMAP 1", fontsize=7, color=tkc)
                ax.set_ylabel("UMAP 2", fontsize=7, color=tkc)
                ax.set_zlabel("UMAP 3", fontsize=7, color=tkc)
                ax.tick_params(colors=tkc, labelsize=6)
                ax.set_title(f"view {vi + 1}", color=tc, fontsize=9)
            fig_s.suptitle(title, color=tc, fontsize=12)
            png_path = pathlib.Path(out_path).with_suffix(".png")
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig_s.savefig(str(png_path), dpi=150, bbox_inches="tight",
                          facecolor=fig_s.get_facecolor())
            _plt.close(fig_s)
        except Exception:
            pass

        # ── Interactive HTML ──────────────────────────────────────────────────
        try:
            import plotly.graph_objects as go
        except ImportError:
            return
        t_cur = T()
        bg_col = t_cur["fig_bg"]
        panel_col = t_cur["ax_bg"]

        fig_p = go.Figure()
        unassigned = point_gi < 0
        if unassigned.any():
            fig_p.add_trace(go.Scatter3d(
                x=e3[unassigned, 0], y=e3[unassigned, 1], z=e3[unassigned, 2],
                mode="markers",
                marker=dict(size=2, color="#555566", opacity=0.10),
                name="Unassigned", showlegend=True,
                hovertemplate="Unassigned<extra></extra>",
            ))
        for gi, gname in enumerate(group_names):
            mask = point_gi == gi
            if not mask.any():
                continue
            col = group_colors[gi]
            fig_p.add_trace(go.Scatter3d(
                x=e3[mask, 0], y=e3[mask, 1], z=e3[mask, 2],
                mode="markers",
                marker=dict(size=2, color=col, opacity=0.25),
                name=gname, legendgroup=gname, showlegend=True,
                hovertemplate=f"{gname}<extra></extra>",
            ))
            if gi in g_centroids:
                c = g_centroids[gi]
                fig_p.add_trace(go.Scatter3d(
                    x=[c[0]], y=[c[1]], z=[c[2]],
                    mode="markers+text",
                    marker=dict(size=8, color=col,
                                line=dict(color=t_cur["tick"], width=1),
                                opacity=1.0),
                    text=[gname], textposition="top center",
                    textfont=dict(size=10, color=t_cur["tick"]),
                    name=f"{gname} centroid",
                    legendgroup=gname, showlegend=False,
                    hovertemplate=f"<b>{gname}</b><extra></extra>",
                ))
        fig_p.update_layout(
            title=dict(text=title,
                       font=dict(color=t_cur["tick"], size=14)),
            paper_bgcolor=bg_col, plot_bgcolor=bg_col,
            scene=dict(
                xaxis=dict(title="UMAP 1", color=t_cur["tick"],
                           backgroundcolor=bg_col, gridcolor="#333355"),
                yaxis=dict(title="UMAP 2", color=t_cur["tick"],
                           backgroundcolor=bg_col, gridcolor="#333355"),
                zaxis=dict(title="UMAP 3", color=t_cur["tick"],
                           backgroundcolor=bg_col, gridcolor="#333355"),
                bgcolor=bg_col,
            ),
            legend=dict(font=dict(color=t_cur["tick"], size=9),
                        bgcolor=panel_col, bordercolor="#333355"),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        html_path = pathlib.Path(out_path).with_suffix(".html")
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig_p.write_html(str(html_path), include_plotlyjs="cdn",
                         full_html=True)
    except Exception:
        pass


def save_group_transitions_3d(
        embedding, labels, groups: dict,
        animals: list,
        out_path,
        tag: str = "",
        max_edges_per_eg: int = 8,
        min_prob: float = 0.08) -> None:
    """
    Write a 3D UMAP with group-level transition arrows coloured by
    experimental group.

    Each experimental group gets its own set of arrows (coloured uniquely),
    so you can compare whether ctrl vs. treated animals differ in which
    behavioural-group transitions are dominant.  Only the top-ranked
    transitions per experimental group are shown to keep the plot readable.

    Parameters
    ----------
    embedding         : (n, >=3) UMAP embedding
    labels            : (n,) cluster labels
    groups            : {group_name: {"labels": [cids], "color": hex}}
    animals           : list of animal dicts with "df" and "exp_group" keys
    out_path          : output path (extension replaced)
    tag               : title tag
    max_edges_per_eg  : top-N transitions shown per experimental group
    min_prob          : minimum group-transition probability to draw
    """
    try:
        import numpy as _np
        emb = _np.asarray(embedding)
        lbl = _np.asarray(labels, dtype=int)
        if emb.shape[1] < 3:
            return
        e3 = emb[:, :3]

        group_names  = list(groups.keys())
        group_colors = [groups[g].get("color", PALETTE[i % len(PALETTE)])
                        for i, g in enumerate(group_names)]
        n_g = len(group_names)
        if n_g < 2:
            return

        cid_to_gi: dict = {}
        for gi, gname in enumerate(group_names):
            for cid in groups[gname].get("labels", []):
                cid_to_gi[int(cid)] = gi

        # Group centroids in 3D UMAP space
        point_gi = _np.array([cid_to_gi.get(int(l), -1) for l in lbl], dtype=int)
        g_centroids = {}
        for gi in range(n_g):
            mask = point_gi == gi
            if mask.any():
                g_centroids[gi] = e3[mask].mean(axis=0)

        # Compute group-level transition counts per experimental group
        # (sum raw counts across all animals in that exp_group, then normalise)
        eg_counts: dict = {}   # {eg_name: np.array(n_g, n_g)}
        for ani in animals:
            eg  = str(ani.get("exp_group", "Unknown"))
            df  = ani.get("df")
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            if eg not in eg_counts:
                eg_counts[eg] = _np.zeros((n_g, n_g), dtype=float)
            seq = df.sort_values("start_frame")["label"].values
            gi_seq = [cid_to_gi.get(int(lab), -1) for lab in seq]
            for a, b in zip(gi_seq[:-1], gi_seq[1:]):
                if a >= 0 and b >= 0 and a != b:
                    eg_counts[eg][a, b] += 1.0

        # Row-normalise each exp-group matrix
        eg_tmats: dict = {}
        for eg, mat in eg_counts.items():
            rs = mat.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            eg_tmats[eg] = mat / rs

        eg_names = sorted(eg_tmats.keys())
        if not eg_names:
            return

        # Colour palette for experimental groups (distinct from behavioural groups)
        EG_PALETTE = [
            "#e63946", "#2196F3", "#4CAF50", "#FF9800",
            "#9C27B0", "#00BCD4", "#FF5722", "#8BC34A",
        ]
        eg_colors = {eg: EG_PALETTE[i % len(EG_PALETTE)]
                     for i, eg in enumerate(eg_names)}

        # Never show group-level transitions at or below chance
        chance_floor  = 1.0 / max(1, n_g - 1)
        effective_min = max(min_prob, chance_floor)

        # Select top edges per experimental group (global rank within each)
        eg_edges: dict = {}   # {eg: [(gi, gj, prob)]}
        for eg in eg_names:
            mat = eg_tmats[eg]
            cands = []
            for gi in range(n_g):
                if gi not in g_centroids:
                    continue
                for gj in range(n_g):
                    if gj == gi or gj not in g_centroids:
                        continue
                    prob = float(mat[gi, gj])
                    if prob > effective_min:
                        cands.append((prob, gi, gj))
            cands.sort(reverse=True)
            eg_edges[eg] = [(gi, gj, prob)
                            for prob, gi, gj in cands[:max_edges_per_eg]]

        # Global max probability across all exp-groups and normalise over the
        # above-chance range so every arrow's thickness is comparable.
        _all_probs  = [prob for _eg_e in eg_edges.values() for _, _, prob in _eg_e]
        _max_p_all  = max(_all_probs, default=1.0)
        _p_range    = max(_max_p_all - effective_min, 1e-9)

        extent = float(_np.max(e3.max(axis=0) - e3.min(axis=0))) if len(g_centroids) > 1 else 1.0
        title  = (f"3D UMAP — Group Transitions by Experimental Group  [{tag}]"
                  if tag else
                  "3D UMAP — Group Transitions by Experimental Group")

        # ── Static PNG ────────────────────────────────────────────────────────
        try:
            import matplotlib.pyplot as _plt
            import matplotlib.patches as _mpatches
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            t_cur  = T()
            bg     = t_cur["fig_bg"]
            panel  = t_cur["ax_bg"]
            tc     = t_cur["tick"]
            views  = [(20, 45), (20, 200), (60, 100)]
            fig_s  = _plt.figure(figsize=(18, 6), facecolor=bg)
            for vi, (elev, azim) in enumerate(views):
                ax = fig_s.add_subplot(1, 3, vi + 1, projection="3d")
                ax.set_facecolor(panel)
                # Translucent point cloud coloured by behavioural group
                for gi, gname in enumerate(group_names):
                    mask = point_gi == gi
                    if mask.any():
                        ax.scatter(e3[mask, 0], e3[mask, 1], e3[mask, 2],
                                   s=1, alpha=0.20, color=group_colors[gi],
                                   depthshade=False)
                # Group centroid nodes
                for gi, gname in enumerate(group_names):
                    if gi not in g_centroids:
                        continue
                    c   = g_centroids[gi]
                    col = group_colors[gi]
                    ax.scatter(*c, s=55, color=col,
                               edgecolors="white", linewidths=0.5,
                               zorder=5, depthshade=False)
                    ax.text(c[0], c[1], c[2], f" {gname}",
                            fontsize=5, color=col, zorder=6)
                # Arrows per experimental group — thickness ∝ above-chance range
                for eg in eg_names:
                    eg_col = eg_colors[eg]
                    for gi, gj, prob in eg_edges[eg]:
                        s_c = g_centroids[gi]
                        t_c = g_centroids[gj]
                        dx, dy, dz = t_c - s_c
                        rel   = (prob - effective_min) / _p_range
                        alpha = float(0.40 + rel * 0.55)
                        lw    = float(0.8 + rel * 4.2)
                        ax.quiver(s_c[0], s_c[1], s_c[2], dx, dy, dz,
                                  arrow_length_ratio=0.28,
                                  color=eg_col, alpha=alpha, linewidth=lw,
                                  normalize=False)
                ax.view_init(elev=elev, azim=azim)
                ax.set_xlabel("UMAP 1", fontsize=7, color=tc)
                ax.set_ylabel("UMAP 2", fontsize=7, color=tc)
                ax.set_zlabel("UMAP 3", fontsize=7, color=tc)
                ax.tick_params(colors=tc, labelsize=6)
                ax.set_title(f"view {vi + 1}", color=tc, fontsize=9)
            # Legend for exp groups
            handles = [_mpatches.Patch(color=eg_colors[eg], label=eg)
                       for eg in eg_names]
            fig_s.legend(handles=handles, loc="lower center",
                         ncol=min(len(eg_names), 6),
                         fontsize=8, facecolor=bg,
                         labelcolor=tc, edgecolor="#333355")
            fig_s.suptitle(title, color=tc, fontsize=12, y=1.02)
            png_path = pathlib.Path(out_path).with_suffix(".png")
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig_s.savefig(str(png_path), dpi=150, bbox_inches="tight",
                          facecolor=fig_s.get_facecolor())
            _plt.close(fig_s)
        except Exception:
            pass

        # ── Interactive HTML ──────────────────────────────────────────────────
        try:
            import plotly.graph_objects as go
        except ImportError:
            return
        t_cur     = T()
        bg_col    = t_cur["fig_bg"]
        panel_col = t_cur["ax_bg"]
        cone_size = extent * 0.055

        fig_p = go.Figure()
        # Point cloud coloured by behavioural group
        for gi, gname in enumerate(group_names):
            mask = point_gi == gi
            if not mask.any():
                continue
            fig_p.add_trace(go.Scatter3d(
                x=e3[mask, 0], y=e3[mask, 1], z=e3[mask, 2],
                mode="markers",
                marker=dict(size=2, color=group_colors[gi], opacity=0.20),
                name=gname, legendgroup=gname, showlegend=True,
                legendgrouptitle=dict(text="Beh. groups") if gi == 0 else {},
                hovertemplate=f"{gname}<extra></extra>",
            ))
        # Group centroid nodes
        for gi, gname in enumerate(group_names):
            if gi not in g_centroids:
                continue
            c   = g_centroids[gi]
            col = group_colors[gi]
            fig_p.add_trace(go.Scatter3d(
                x=[c[0]], y=[c[1]], z=[c[2]],
                mode="markers+text",
                marker=dict(size=10, color=col,
                            line=dict(color=t_cur["tick"], width=1), opacity=1.0),
                text=[gname], textposition="top center",
                textfont=dict(size=11, color=t_cur["tick"]),
                name=f"{gname} centroid",
                legendgroup=gname, showlegend=False,
                hovertemplate=f"<b>{gname}</b><extra></extra>",
            ))
        # Arrows per experimental group (shaft + cone)
        for eg in eg_names:
            eg_col = eg_colors[eg]
            first  = True
            for gi, gj, prob in eg_edges[eg]:
                s_c  = _np.array(g_centroids[gi])
                t_c  = _np.array(g_centroids[gj])
                d    = t_c - s_c
                norm = float(_np.linalg.norm(d))
                if norm < 1e-8:
                    continue
                d_hat = d / norm
                rel   = (prob - effective_min) / _p_range
                alpha = float(0.45 + rel * 0.50)
                lw_px = max(5, int(5 + rel * 15))
                shaft_end = s_c + 0.82 * d
                gn  = f"eg_{eg}"
                fig_p.add_trace(go.Scatter3d(
                    x=[s_c[0], shaft_end[0], None],
                    y=[s_c[1], shaft_end[1], None],
                    z=[s_c[2], shaft_end[2], None],
                    mode="lines",
                    line=dict(color=eg_col, width=lw_px),
                    opacity=alpha,
                    name=eg, legendgroup=gn,
                    showlegend=first,
                    legendgrouptitle=dict(text="Exp. groups") if first else {},
                    hovertemplate=(
                        f"<b>{eg}</b>: {group_names[gi]}→"
                        f"{group_names[gj]}: {prob:.3f}<extra></extra>"),
                ))
                fig_p.add_trace(go.Cone(
                    x=[t_c[0]], y=[t_c[1]], z=[t_c[2]],
                    u=[d_hat[0]], v=[d_hat[1]], w=[d_hat[2]],
                    sizemode="absolute",
                    sizeref=cone_size * (0.4 + rel * 1.2),
                    anchor="tip",
                    colorscale=[[0, eg_col], [1, eg_col]],
                    showscale=False,
                    opacity=alpha,
                    hovertemplate=(
                        f"<b>{eg}</b>: {group_names[gi]}→"
                        f"{group_names[gj]}: {prob:.3f}<extra></extra>"),
                    showlegend=False,
                ))
                first = False

        fig_p.update_layout(
            title=dict(text=title,
                       font=dict(color=t_cur["tick"], size=13)),
            paper_bgcolor=bg_col, plot_bgcolor=bg_col,
            scene=dict(
                xaxis=dict(title="UMAP 1", color=t_cur["tick"],
                           backgroundcolor=bg_col, gridcolor="#333355"),
                yaxis=dict(title="UMAP 2", color=t_cur["tick"],
                           backgroundcolor=bg_col, gridcolor="#333355"),
                zaxis=dict(title="UMAP 3", color=t_cur["tick"],
                           backgroundcolor=bg_col, gridcolor="#333355"),
                bgcolor=bg_col,
            ),
            legend=dict(font=dict(color=t_cur["tick"], size=9),
                        bgcolor=panel_col, bordercolor="#333355",
                        groupclick="toggleitem"),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        html_path = pathlib.Path(out_path).with_suffix(".html")
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig_p.write_html(str(html_path), include_plotlyjs="cdn",
                         full_html=True)
    except Exception:
        pass


#
# GROUP PREDICTOR TAB PANEL
#

def _gp_n_jobs(mode: str = "thread") -> int:
    """Safe joblib n_jobs for GroupPredictorPanel.

    thread mode (default): Parallel(prefer="threads") calls. Threads share
    memory so there is no per-worker RAM overhead; leave a small fraction of
    cores free for the GUI/OS and cap at 32.

    loky mode: Parallel(backend="loky") calls — true multiprocessing, each
    worker has its own GIL. Each process is a full Python interpreter
    (~100–150 MB), so leave ≥25% of cores free and cap at 8.

    process mode: reserved for the optional nested permutation test where
    each worker also runs greedy selection (~400 MB); very tight cap.
    """
    import os as _os
    cores = _os.cpu_count() or 4
    if mode == "process":
        return max(1, min(cores // 2, 4))
    if mode == "loky":
        free_l = max(2, cores // 4)
        return max(1, min(cores - free_l, 8))
    # thread mode: leave at least 2 cores free (or ~12% on large machines).
    free = max(2, cores // 8)
    return max(1, min(cores - free, 32))


class GroupPredictorPanel(ctk.CTkFrame):
    """
    'Group Predictor' tab — asks whether a mouse's behavioral profile can
    predict its experimental group.

    Runs three parallel models (Frequency / Total Duration / Transition
    Probability), each evaluated with Leave-One-Out CV and a permutation test.
    Results are shown in a comparison table; clicking a row drills into the
    confusion matrix and feature-importance chart for that model.

    Requires
    --------
    get_animals_fn() → list of animal dicts  {name, df, fps, exp_group, ...}
    get_groups_fn()  → {group_name: {"labels": [int], "color": str}}
    """

    _ALGO_OPTS = ["Elastic Net Logistic Reg.", "SVM (linear)"]
    EXHAUSTIVE_COMBO_LIMIT = 15_000

    def __init__(self, parent, get_animals_fn, get_groups_fn=None, **kw):
        super().__init__(parent, fg_color=T()["panel"], **kw)
        self._get_animals   = get_animals_fn
        self._get_groups_fn = get_groups_fn or (lambda: {})

        # Last run state
        self._results: list = []          # [{name, X, y, le, feat_names, scores, pval, kappa, cm, coef}]
        self._selected_model: int = 0     # index into _results
        self._open_figs: list = []        # matplotlib figures to close on next redraw
        self._overview_fig  = None        # Figure 0 (overview bar + kappa chart)
        self._null_comp_fig = None        # Figure 1 (null distribution comparison)
        self._perm_display_mode = "cond"  # "cond" or "nested" — which null to show
        self._last_chance = 0.5           # stored in _finish for re-draw after toggle
        self._nested_running = False      # True while background nested perm is running
        self._nested_btn = None           # reference to Run Nested Test button (left panel)
        self._perm_toggle_frame = None    # Conditional/Nested toggle frame (left panel)

        # Multi-factor prediction target
        self._factor_var = tk.StringVar(value="Combined")

        # Threading state
        import queue as _queue
        import threading as _threading
        self._progress_q = _queue.Queue()
        self._running = False
        self._run_btn = None              # set in _build_controls
        self._cancel_btn = None
        self._cancel_event = _threading.Event()
        self._loading_progress_bar = None
        self._loading_status_lbl   = None

        self.columnconfigure(0, weight=0, minsize=256)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_controls()
        self._build_results_area()

    # ── Controls (left sidebar) ───────────────────────────────────────────────

    def _build_controls(self):
        ctrl = ctk.CTkScrollableFrame(
            self, fg_color=T()["bg"], corner_radius=0, width=254)
        ctrl.grid(row=0, column=0, sticky="nsew", padx=(6, 2), pady=6)
        ctrl.columnconfigure(0, weight=1)

        def section(title):
            fr = ctk.CTkFrame(ctrl, fg_color=T()["card"], corner_radius=8)
            fr.pack(fill="x", padx=4, pady=4)
            ctk.CTkLabel(fr, text=title,
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=T()["hdr_text"],
                         ).pack(anchor="w", padx=10, pady=(8, 2))
            return fr

        # Feature source
        sf = section("Feature Source")
        self._source_var = ctk.StringVar(value="clusters")
        for label, val in [("Individual clusters",           "clusters"),
                           ("Behavior groups",               "groups"),
                           ("Mix (both)",                    "mix"),
                           ("Custom (select clusters + groups)", "custom")]:
            ctk.CTkRadioButton(sf, text=label, variable=self._source_var,
                               value=val,
                               command=self._on_source_changed,
                               text_color=T()["subtext"],
                               ).pack(anchor="w", padx=12, pady=2)
        sf.pack_configure(pady=(4, 0))

        # Cluster checklist (shown only when source == "custom")
        self._clusters_section = section("Clusters to Include")
        self._clusters_inner = ctk.CTkScrollableFrame(
            self._clusters_section, fg_color=T()["card2"],
            corner_radius=6, height=120)
        self._clusters_inner.pack(fill="x", padx=8, pady=(2, 8))
        self._cluster_vars: dict = {}   # cluster_id -> BooleanVar
        ctk.CTkLabel(self._clusters_section,
                     text="(load animals first)",
                     text_color=T()["muted"],
                     font=ctk.CTkFont(size=10),
                     ).pack(anchor="w", padx=12, pady=(0, 6))
        self._clusters_section.pack_forget()   # hidden until source == custom

        # Behavior groups checklist (shown when source is "groups", "mix", or "custom")
        self._groups_section = section("Behavior Groups to Include")
        self._groups_inner = ctk.CTkScrollableFrame(
            self._groups_section, fg_color=T()["card2"],
            corner_radius=6, height=100)
        self._groups_inner.pack(fill="x", padx=8, pady=(2, 8))
        self._group_vars: dict = {}   # group_name -> BooleanVar
        ctk.CTkLabel(self._groups_section,
                     text="(load animals first)",
                     text_color=T()["muted"],
                     font=ctk.CTkFont(size=10),
                     ).pack(anchor="w", padx=12, pady=(0, 6))
        self._groups_section.pack_forget()   # hidden until source != clusters

        # Algorithm
        af = section("Algorithm")
        self._algo_var = ctk.StringVar(value=self._ALGO_OPTS[0])
        ctk.CTkOptionMenu(af, variable=self._algo_var,
                          values=self._ALGO_OPTS,
                          width=220).pack(padx=12, pady=(4, 10))

        # Predict by (multi-factor)
        pb_f = section("Predict by")
        ctk.CTkLabel(pb_f,
                     text="When animals have multiple labels (e.g. Ctrl | Male),\n"
                          "choose which factor to classify.",
                     text_color=T()["muted"],
                     font=ctk.CTkFont(size=10),
                     justify="left").pack(anchor="w", padx=12, pady=(2, 4))
        for lbl, val in [("Combined (all labels)", "Combined"),
                         ("Factor 1 only",         "Factor1"),
                         ("Factor 2 only",         "Factor2"),
                         ("Factor 3 only",         "Factor3")]:
            ctk.CTkRadioButton(pb_f, text=lbl, variable=self._factor_var,
                               value=val).pack(anchor="w", padx=16, pady=2)
        ctk.CTkFrame(pb_f, height=6, fg_color="transparent").pack()   # spacer

        # Permutation count
        pf = section("Permutation Test")
        self._nperm_var = ctk.StringVar(value="199")
        ctk.CTkOptionMenu(pf, variable=self._nperm_var,
                          values=["99", "199", "499", "999"],
                          width=220).pack(padx=12, pady=(4, 4))
        ctk.CTkLabel(pf,
                     text="Full pipeline (greedy+LOO) re-run with shuffled\n"
                          "group labels. Higher = more precise p-values,\n"
                          "slower. p(n)=nested test; p(c)=conditional test.",
                     text_color=T()["muted"],
                     font=ctk.CTkFont(size=10),
                     justify="left").pack(anchor="w", padx=12, pady=(0, 4))
        self._nested_btn = ctk.CTkButton(
            pf,
            text="Run Nested Permutation Test",
            command=self._trigger_nested_test,
            width=220, height=28,
            state="disabled",
        )
        self._nested_btn.pack(padx=12, pady=(0, 4))
        # Toggle frame — hidden until nested test completes; populated by _refresh_perm_toggle.
        # Created here so its position in pf is fixed (after the button, before bottom padding).
        self._perm_toggle_frame = ctk.CTkFrame(pf, fg_color="transparent")
        self._perm_toggle_frame.pack(padx=12, pady=(0, 4))
        self._perm_toggle_frame.pack_forget()  # hide immediately; shown by _refresh_perm_toggle

        # Max contributors
        mc_f = section("Max Contributors")
        ctk.CTkLabel(mc_f,
                     text="Max clusters / groups used as features\n"
                          "(top N by variance; 'All' = no limit)",
                     text_color=T()["muted"],
                     font=ctk.CTkFont(size=10),
                     justify="left").pack(anchor="w", padx=12, pady=(2, 0))
        self._max_contrib_var = ctk.StringVar(value="5")
        ctk.CTkOptionMenu(mc_f, variable=self._max_contrib_var,
                          values=["All", "1", "2", "3", "4", "5", "6", "8",
                                  "10", "15", "20", "30"],
                          width=220).pack(padx=12, pady=(4, 10))

        # Model name (mix / custom only)
        self._model_name_section = section("Model Name")
        ctk.CTkLabel(self._model_name_section,
                     text="Optional label for this run configuration",
                     text_color=T()["muted"],
                     font=ctk.CTkFont(size=10)).pack(anchor="w", padx=12, pady=(2, 0))
        self._model_name_var = ctk.StringVar(value="")
        ctk.CTkEntry(self._model_name_section,
                     textvariable=self._model_name_var,
                     placeholder_text="e.g. Mix-Locomotion",
                     width=220).pack(padx=12, pady=(4, 10))
        self._model_name_section.pack_forget()   # hidden until mix / custom

        self._view_model_var = tk.IntVar(value=0)
        self._model_radio_btns: list = []

        # Run
        self._run_btn = ctk.CTkButton(
            ctrl, text="   Run Models",
            command=self._run,
            fg_color=T()["btn_unbiased"],
            height=38,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self._run_btn.pack(fill="x", padx=8, pady=8)

        self._cancel_btn = ctk.CTkButton(
            ctrl, text="   Cancel",
            command=self._cancel,
            fg_color=T()["btn_del"],
            hover_color=T()["btn_del_h"],
            height=32,
            state="disabled",
        )
        self._cancel_btn.pack(fill="x", padx=8, pady=(0, 4))

        # Export
        ctk.CTkButton(
            ctrl, text="Save:  Export Results",
            command=self._export_csv,
            fg_color=T()["btn_save"],
        ).pack(fill="x", padx=8, pady=(0, 4))

        # Status label
        self._status_lbl = ctk.CTkLabel(
            ctrl, text="",
            text_color=T()["subtext"],
            wraplength=230, justify="left",
            font=ctk.CTkFont(size=11))
        self._status_lbl.pack(anchor="w", padx=12, pady=4)

    def _on_source_changed(self):
        src = self._source_var.get()
        if src == "clusters":
            self._clusters_section.pack_forget()
            self._groups_section.pack_forget()
            self._model_name_section.pack_forget()
        elif src == "custom":
            self._clusters_section.pack(fill="x", padx=4, pady=4)
            self._refresh_cluster_checklist()
            self._groups_section.pack(fill="x", padx=4, pady=4)
            self._refresh_group_checklist()
            self._model_name_section.pack(fill="x", padx=4, pady=4)
        elif src == "mix":
            self._clusters_section.pack_forget()
            self._groups_section.pack(fill="x", padx=4, pady=4)
            self._refresh_group_checklist()
            self._model_name_section.pack(fill="x", padx=4, pady=4)
        else:   # groups
            self._clusters_section.pack_forget()
            self._groups_section.pack(fill="x", padx=4, pady=4)
            self._refresh_group_checklist()
            self._model_name_section.pack_forget()

    def _refresh_cluster_checklist(self):
        """Rebuild the scrollable cluster checklist from the currently loaded animals."""
        for w in self._clusters_inner.winfo_children():
            w.destroy()
        self._cluster_vars.clear()
        animals = self._get_animals()
        all_clusters: set = set()
        for a in animals:
            if "df" in a and "label" in a["df"].columns:
                all_clusters.update(a["df"]["label"].unique())
        if not all_clusters:
            ctk.CTkLabel(self._clusters_inner,
                         text="No clusters found — load animals first.",
                         text_color=T()["muted"],
                         font=ctk.CTkFont(size=10)).pack(anchor="w", padx=6)
            return
        for c in sorted(all_clusters):
            var = tk.BooleanVar(value=True)
            self._cluster_vars[c] = var
            ctk.CTkCheckBox(self._clusters_inner, text=f"Cluster {c}",
                            variable=var,
                            text_color=T()["subtext"],
                            font=ctk.CTkFont(size=11),
                            ).pack(anchor="w", padx=4, pady=1)

    def _refresh_group_checklist(self):
        """Rebuild the scrollable group checklist from the current groups dict."""
        for w in self._groups_inner.winfo_children():
            w.destroy()
        self._group_vars.clear()
        groups = self._get_groups_fn() or {}
        if not groups:
            ctk.CTkLabel(self._groups_inner, text="No behavior groups defined.",
                         text_color=T()["muted"],
                         font=ctk.CTkFont(size=10)).pack(anchor="w", padx=6)
            return
        for gname in groups:
            var = tk.BooleanVar(value=True)
            self._group_vars[gname] = var
            ctk.CTkCheckBox(self._groups_inner, text=gname,
                            variable=var,
                            text_color=T()["subtext"],
                            font=ctk.CTkFont(size=11),
                            ).pack(anchor="w", padx=4, pady=1)

    # ── Results area (right panel) ────────────────────────────────────────────

    def _build_results_area(self):
        right = ctk.CTkFrame(self, fg_color=T()["panel"])
        right.grid(row=0, column=1, sticky="nsew", padx=(2, 6), pady=6)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=0)   # model overview figure (Figure 0)
        right.rowconfigure(1, weight=0)   # caveat bar
        right.rowconfigure(2, weight=1)   # detail plots

        # Internal state only — no display widgets needed
        self._tbl_rows: list = []
        self._run_config_lbl = None   # unused; config shown via status label

        # ── Model overview figure (Figure 0 — shown after Run) ──
        self._overview_fr = ctk.CTkFrame(right, fg_color=T()["panel"])
        self._overview_fr.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        self._overview_fr.columnconfigure(0, weight=1)

        # ── Caveat bar ──
        self._caveat_fr = ctk.CTkFrame(right, fg_color="#7a6000", corner_radius=6)
        self._caveat_fr.grid(row=1, column=0, sticky="ew", padx=4, pady=2)
        self._caveat_lbl = ctk.CTkLabel(
            self._caveat_fr, text="",
            text_color="#ffe87c",
            wraplength=700, justify="left",
            font=ctk.CTkFont(size=11))
        self._caveat_lbl.pack(anchor="w", padx=10, pady=5)
        self._caveat_fr.grid_remove()   # hidden until caveats exist

        # ── Detail plots (scrollable canvas) ──
        detail_outer = ctk.CTkFrame(right, fg_color=T()["panel"])
        detail_outer.grid(row=2, column=0, sticky="nsew", padx=4, pady=(2, 4))
        detail_outer.columnconfigure(0, weight=1)
        detail_outer.columnconfigure(1, weight=0)
        detail_outer.rowconfigure(0, weight=1)

        self._detail_canvas = tk.Canvas(
            detail_outer, bg=T()["fig_bg"], highlightthickness=0)
        self._detail_ysb = ctk.CTkScrollbar(
            detail_outer, command=self._detail_canvas.yview)
        self._detail_canvas.configure(yscrollcommand=self._detail_ysb.set)
        self._detail_canvas.grid(row=0, column=0, sticky="nsew")
        self._detail_ysb.grid(row=0, column=1, sticky="ns")

        self._detail_inner = ctk.CTkFrame(
            self._detail_canvas, fg_color=T()["panel"])
        self._detail_win = self._detail_canvas.create_window(
            (0, 0), window=self._detail_inner, anchor="nw")

        self._detail_inner.bind(
            "<Configure>",
            lambda e: self._detail_canvas.configure(
                scrollregion=self._detail_canvas.bbox("all")))
        self._detail_canvas.bind(
            "<Configure>",
            lambda e: self._detail_canvas.itemconfig(
                self._detail_win, width=e.width))
        self._detail_canvas.bind(
            "<MouseWheel>",
            lambda e: self._detail_canvas.yview_scroll(
                int(-1 * (e.delta / 120)), "units"))

        self._show_detail_placeholder(
            "Load animals in the Combined Analysis tab, assign\n"
            "experimental groups, then click   Run Models.")

    # ── Figure cleanup helper ─────────────────────────────────────────────────

    @staticmethod
    def _close_fig(fig):
        """
        Close a matplotlib figure and explicitly break every tkinter reference
        cycle so that FigureCanvasTkAgg, NavigationToolbar2Tk, tk.StringVar,
        and tk.PhotoImage are freed by Python's reference-counting on the
        calling (main) thread — not deferred to the cyclic GC.

        Why this matters: NavigationToolbar2Tk is a plain Python object, not a
        tk widget subclass.  Calling .destroy() on its parent frame removes the
        tk-side widgets but leaves the Python toolbar object alive, reachable
        only via canvas.toolbar / toolbar.canvas.  Setting fig.canvas = None
        breaks fig → canvas but leaves the canvas ↔ toolbar cycle intact.
        Python's cyclic GC may then collect that cycle from a joblib worker
        thread, causing tk.StringVar.__del__ / tk.PhotoImage.__del__ to call
        into the Tcl interpreter off the main thread:
            RuntimeError: main thread is not in main loop
            Tcl_AsyncDelete: async handler deleted by the wrong thread  ← crash

        Explicitly nulling canvas.toolbar and toolbar.canvas converts the
        deallocation to a refcount operation that runs immediately here on the
        main thread, eliminating the race entirely.
        """
        import matplotlib.pyplot as _plt
        canvas = getattr(fig, 'canvas', None)
        if canvas is not None:
            # Break toolbar ↔ canvas cycle: both sides must be cleared so the
            # toolbar (and its tk.StringVar) reach refcount 0 right here.
            toolbar = getattr(canvas, 'toolbar', None)
            if toolbar is not None:
                try:
                    canvas.toolbar = None
                except Exception:
                    pass
                try:
                    toolbar.canvas = None
                except Exception:
                    pass
            # Null the PhotoImage so its __del__ runs here via refcount,
            # not deferred to the GC.  Attribute name varies by mpl version.
            for _attr in ('_tkphoto', 'photo'):
                if getattr(canvas, _attr, None) is not None:
                    try:
                        setattr(canvas, _attr, None)
                    except Exception:
                        pass
            # Unbind matplotlib's filter_destroy before nulling canvas.figure.
            # If the Tk widget is still alive when the parent window is later
            # closed manually, the <Destroy> event fires and filter_destroy tries
            # to access canvas.figure._canvas_callbacks — which crashes when
            # figure is already None.  Removing the binding prevents that.
            try:
                _tkw = canvas.get_tk_widget()
                _tkw.unbind("<Destroy>")
            except Exception:
                pass
            # Break canvas → figure back-reference before fig.canvas = None.
            try:
                canvas.figure = None
            except Exception:
                pass
        # Deregister from matplotlib's Gcf.
        try:
            _plt.close(fig)
        except Exception:
            pass
        # Break fig → canvas; canvas refcount reaches 0 and is freed here.
        try:
            fig.canvas = None
        except Exception:
            pass

    # ── Placeholder helpers ───────────────────────────────────────────────────

    def _show_detail_placeholder(self, msg: str):
        import matplotlib.pyplot as _plt
        # Destroy canvas widgets BEFORE closing figures — same order as
        # _draw_detail.  plt.close() on a live embedded figure tries to destroy
        # the canvas widget; if the widget is already gone it raises a TclError
        # that crashes the process on the next redraw.
        for w in self._detail_inner.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        for fig in self._open_figs:
            self._close_fig(fig)
        self._open_figs = []
        import gc as _gc
        _gc.collect()
        ctk.CTkLabel(self._detail_inner, text=msg,
                     text_color=T()["muted"],
                     font=ctk.CTkFont(size=13),
                     justify="center").pack(pady=50, padx=20)

    def _show_loading_overlay(self):
        """Show a loading screen in the results area while models are running."""
        import matplotlib.pyplot as _plt
        # Close any open figures before destroying their canvas widgets.
        for w in self._overview_fr.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        for w in self._detail_inner.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        for fig in self._open_figs:
            self._close_fig(fig)
        self._open_figs = []
        # Overview and null-comparison figures are not in _open_figs but also
        # hold FigureCanvasTkAgg objects.  _close_fig breaks their canvas /
        # toolbar / PhotoImage / StringVar cycles explicitly so nothing is left
        # for the cyclic GC to collect from a background thread.
        for _attr in ("_overview_fig", "_null_comp_fig"):
            _fig = getattr(self, _attr, None)
            if _fig is not None:
                self._close_fig(_fig)
                setattr(self, _attr, None)
        import gc as _gc
        _gc.collect()

        # Clear stale results from the previous run so no prior-run data leaks
        # through if the new run fails before _finish() can replace them.
        self._results = []

        # Hide the Conditional/Nested toggle frame left by a prior nested test
        # so it doesn't float above the loading animation during the new run.
        if self._perm_toggle_frame:
            for _w in self._perm_toggle_frame.winfo_children():
                try:
                    _w.destroy()
                except Exception:
                    pass
            self._perm_toggle_frame.pack_forget()

        outer = ctk.CTkFrame(self._detail_inner, fg_color="transparent")
        outer.pack(expand=True, fill="both", pady=80, padx=40)

        _logo_lbl = None
        try:
            from PIL import Image as _PILImage, ImageTk as _ImageTk
            _logo_file = ("CUBE_logo dark theme.png"
                          if _THEME_KEY == "dark" else "CUBE_logo.png")
            _logo_path = pathlib.Path(__file__).parent / _logo_file
            if _logo_path.is_file():
                _raw = _PILImage.open(str(_logo_path))
                _raw.load()
                _raw.thumbnail((400, 180), _PILImage.LANCZOS)
                _pil_img = _raw.copy()
                _raw.close()
                # Use ImageTk.PhotoImage with an explicit master (the toplevel
                # that owns this widget) to avoid a multi-Tk-interpreter clash:
                # ctk.CTkImage creates its PhotoImage in whatever interpreter
                # happens to be the global default (cube.py's tk.Tk root when
                # launched from the main pipeline), making the image invisible
                # to widgets in the analyser's ctk.CTk interpreter.
                _tk_root = outer.winfo_toplevel()
                _tk_photo = _ImageTk.PhotoImage(_pil_img, master=_tk_root)
                _logo_lbl = tk.Label(
                    outer, image=_tk_photo,
                    bg=T()["bg"], bd=0, highlightthickness=0)
                _logo_lbl._photo = _tk_photo  # prevent GC
                _logo_lbl.pack(pady=(0, 32))
        except Exception as _logo_err:
            print(f"[CUBE] Loading overlay logo failed: {_logo_err}")
        if _logo_lbl is None:
            ctk.CTkLabel(outer, text="CUBE",
                         font=ctk.CTkFont(size=56, weight="bold"),
                         text_color=T()["subtext"]).pack(pady=(0, 32))
        ctk.CTkLabel(outer, text="Analysis underway — please wait",
                     font=ctk.CTkFont(size=13),
                     text_color=T()["subtext"]).pack(pady=(0, 20))

        self._loading_progress_bar = ctk.CTkProgressBar(
            outer, mode="determinate", width=420)
        self._loading_progress_bar.set(0)
        self._loading_progress_bar.pack(pady=(0, 10))

        self._loading_status_lbl = ctk.CTkLabel(
            outer, text="",
            font=ctk.CTkFont(size=11),
            text_color=T()["muted"],
            wraplength=420, justify="center")
        self._loading_status_lbl.pack()

    def _status(self, msg: str, color: str = None):
        self._status_lbl.configure(
            text=msg, text_color=color or T()["subtext"])

    # ── Feature matrix builders ───────────────────────────────────────────────

    @staticmethod
    def _compute_sample_weights(animal_data: list) -> "np.ndarray":
        """Per-animal weight proportional to sqrt(n_bouts - 1).

        Used in transition-feature mode to down-weight animals with short
        recordings whose conditional probability estimates are noisy.
        Median-normalised (median animal = weight 1.0), clipped to [0.1, 10.0].
        """
        counts = np.array([max(len(a["df"]) - 1, 1) for a in animal_data],
                          dtype=float)
        w = np.sqrt(counts)
        med = np.median(w)
        if med > 0:
            w = w / med
        return np.clip(w, 0.1, 10.0)

    @staticmethod
    def _build_freq_or_dur(animal_data: list, metric: str,
                           cluster_ids=None) -> tuple:
        """
        Build (X, y, feature_names) from per-cluster frequency or total_duration.
        metric: "frequency" | "total_duration"
        cluster_ids: optional set/list restricting which clusters become columns.
        """
        all_clusters: set = set()
        for a in animal_data:
            all_clusters.update(a["df"]["label"].unique())
        all_clusters_sorted = sorted(
            c for c in all_clusters
            if cluster_ids is None or c in cluster_ids)

        rows, labels = [], []
        for a in animal_data:
            pcm = compute_per_cluster_metrics(a["df"], a["fps"])
            vec = []
            for c in all_clusters_sorted:
                if c in pcm.index:
                    vec.append(float(pcm.loc[c, metric]))
                else:
                    vec.append(0.0)
            rows.append(vec)
            labels.append(a["exp_group"])
        X = np.array(rows, dtype=float)
        feat_names = [f"Cluster {c}" for c in all_clusters_sorted]
        return X, labels, feat_names

    @staticmethod
    def _build_group_feat(animal_data: list, groups: dict, metric: str) -> tuple:
        """
        Build (X, y, feature_names) from per-behavior-group frequency or duration.
        """
        group_names = list(groups.keys())
        rows, labels = [], []
        for a in animal_data:
            m = compute_metrics(a["df"], groups, a["fps"])
            vec = [float(m.get(gn, {}).get(metric, 0.0)) for gn in group_names]
            rows.append(vec)
            labels.append(a["exp_group"])
        X = np.array(rows, dtype=float)
        return X, labels, group_names

    @staticmethod
    def _build_transition(animal_data: list, cluster_ids=None) -> tuple:
        """
        Build (X, y, feature_names) from empirical cluster-transition probabilities.
        Consecutive bouts define transitions; the self-transition diagonal is excluded
        (off-diagonal switching grammar only).
        cluster_ids: optional set/list restricting which clusters form the feature matrix.
        When restricted, the full matrix is built first then the sub-matrix is extracted
        and row-renormalised within the subset.
        """
        all_clusters: set = set()
        for a in animal_data:
            all_clusters.update(a["df"]["label"].unique())
        all_cl_sorted = sorted(all_clusters)
        n_cl_all = len(all_cl_sorted)
        cl_arr = np.array(all_cl_sorted)

        # Subset exposed as features
        cl_sorted = sorted(c for c in all_clusters
                           if cluster_ids is None or c in cluster_ids)
        n_cl = len(cl_sorted)
        sub_idxs = [all_cl_sorted.index(c) for c in cl_sorted]

        rows, labels = [], []
        for a in animal_data:
            df = a["df"].reset_index(drop=True)
            mat = np.zeros((n_cl_all, n_cl_all), dtype=float)
            lbls = df["label"].values
            if len(lbls) > 1:
                valid = np.isin(lbls, cl_arr)
                both = valid[:-1] & valid[1:]
                if both.any():
                    idxs = np.searchsorted(cl_arr, lbls)
                    np.add.at(mat, (idxs[:-1][both], idxs[1:][both]), 1.0)
            row_sums = mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            mat /= row_sums
            if cluster_ids is not None and n_cl < n_cl_all:
                # Extract sub-matrix for the subset and renormalise rows
                sub = mat[np.ix_(sub_idxs, sub_idxs)].copy()
                rsums = sub.sum(axis=1, keepdims=True)
                rsums[rsums == 0] = 1.0
                sub /= rsums
                mask = ~np.eye(n_cl, dtype=bool)
                flat = sub[mask]
            else:
                mask = ~np.eye(n_cl_all, dtype=bool)
                flat = mat[mask]
            rows.append(flat)
            labels.append(a["exp_group"])

        feat_names = [f"T_{cl_sorted[i]}→{cl_sorted[j]}"
                      for i in range(n_cl) for j in range(n_cl) if i != j]
        X = np.array(rows, dtype=float)
        return X, labels, feat_names

    @staticmethod
    def _build_transition_pairs(animal_data: list, pairs: list) -> tuple:
        """
        Build (X, y, feature_names) for an explicit ordered list of directed
        transition pairs [(i,j), …].  Each column is the row-normalised
        probability of transitioning from cluster i to cluster j for one animal.
        """
        if not pairs:
            return (np.zeros((len(animal_data), 0)),
                    [a["exp_group"] for a in animal_data], [])
        all_clusters = sorted({c for p in pairs for c in p})
        cl_idx = {c: k for k, c in enumerate(all_clusters)}
        cl_arr = np.array(all_clusters)
        n_all = len(all_clusters)
        rows, labels = [], []
        for a in animal_data:
            df = a["df"].reset_index(drop=True)
            mat = np.zeros((n_all, n_all), dtype=float)
            lbls = df["label"].values
            if len(lbls) > 1:
                valid = np.isin(lbls, cl_arr)
                both = valid[:-1] & valid[1:]
                if both.any():
                    src = np.searchsorted(cl_arr, lbls[:-1][both])
                    dst = np.searchsorted(cl_arr, lbls[1:][both])
                    np.add.at(mat, (src, dst), 1.0)
            row_sums = mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            mat /= row_sums
            rows.append([mat[cl_idx[i], cl_idx[j]] for i, j in pairs])
            labels.append(a["exp_group"])
        feat_names = [f"T_{i}→{j}" for i, j in pairs]
        return np.array(rows, dtype=float), labels, feat_names

    @staticmethod
    def _build_group_transitions(animal_data: list, groups: dict,
                                  include_cluster_pairs: bool = False) -> tuple:
        """
        Build (X, y, feature_names) from directed group-to-group transition
        probabilities.

        groups: {group_name: {"labels": [cluster_ids], ...}, ...}
        include_cluster_pairs: if True, also append raw cluster pair columns
                               for clusters not fully covered by a group ("mix" mode).

        Feature names: "G_{src}->G_{dst}" for group pairs,
                       "T_{i}->{j}"       for raw pairs (mix mode only).

        P(G_dst | G_src) = sum_{i in G_src, j in G_dst, i!=j} count(i->j)
                           / sum_{i in G_src, j not in G_src} count(i->j)
        """
        group_names = sorted(groups.keys())
        if not group_names:
            X_raw, y_raw, fn_raw = GroupPredictorPanel._build_transition(
                animal_data)
            return X_raw, y_raw, fn_raw

        # Map cluster_id → group_name for each cluster that belongs to a group
        cid_to_gname: dict = {}
        for gname in group_names:
            for cid in groups[gname].get("labels", []):
                cid_to_gname[int(cid)] = gname

        # Union of all cluster IDs across all animals
        all_cids_set: set = set()
        for a in animal_data:
            all_cids_set.update(int(v) for v in a["df"]["label"].unique())
        all_cids = sorted(all_cids_set)
        n_all = len(all_cids)
        cl_arr = np.array(all_cids)
        cl_idx = {c: k for k, c in enumerate(all_cids)}

        # Precompute raw count matrices per animal
        raw_mats: list = []
        y: list = []
        for a in animal_data:
            lbls = a["df"]["label"].values.astype(int)
            mat  = np.zeros((n_all, n_all), dtype=float)
            if len(lbls) > 1:
                valid = np.isin(lbls, cl_arr)
                both  = valid[:-1] & valid[1:]
                if both.any():
                    src = np.searchsorted(cl_arr, lbls[:-1][both])
                    dst = np.searchsorted(cl_arr, lbls[1:][both])
                    np.add.at(mat, (src, dst), 1.0)
            raw_mats.append(mat)
            y.append(a["exp_group"])

        # Group-pair feature columns
        g_pair_names = [f"G_{s}→G_{d}"
                        for s in group_names for d in group_names if s != d]

        rows = []
        for mat in raw_mats:
            row_g = []
            for g_src in group_names:
                src_cids = [cl_idx[int(c)] for c in groups[g_src].get("labels", [])
                            if int(c) in cl_idx]
                # Denominator: all transitions leaving G_src to clusters NOT in G_src
                g_src_set = {int(c) for c in groups[g_src].get("labels", [])}
                non_src_idxs = [k for c, k in cl_idx.items() if c not in g_src_set]
                denom = (mat[np.ix_(src_cids, non_src_idxs)].sum()
                         if src_cids and non_src_idxs else 0.0)
                denom = denom if denom > 0 else 1.0
                for g_dst in group_names:
                    if g_src == g_dst:
                        continue
                    dst_cids = [cl_idx[int(c)] for c in groups[g_dst].get("labels", [])
                                if int(c) in cl_idx]
                    numer = (mat[np.ix_(src_cids, dst_cids)].sum()
                             if src_cids and dst_cids else 0.0)
                    row_g.append(numer / denom)
            rows.append(row_g)

        X = np.array(rows, dtype=float)
        feat_names = list(g_pair_names)

        if include_cluster_pairs:
            # Append raw T_i→j for clusters NOT already fully covered by a group
            covered = set(cid_to_gname.keys())
            uncovered = [c for c in all_cids if c not in covered]
            if uncovered:
                raw_rows = []
                for mat in raw_mats:
                    sub_idx = [cl_idx[c] for c in uncovered]
                    sub = mat[np.ix_(sub_idx, sub_idx)].copy()
                    rs  = sub.sum(axis=1, keepdims=True)
                    rs[rs == 0] = 1.0
                    sub /= rs
                    mask = ~np.eye(len(uncovered), dtype=bool)
                    raw_rows.append(sub[mask])
                X_raw = np.array(raw_rows, dtype=float)
                raw_names = [f"T_{uncovered[i]}→{uncovered[j]}"
                             for i in range(len(uncovered))
                             for j in range(len(uncovered)) if i != j]
                X = np.hstack([X, X_raw]) if X.shape[1] else X_raw
                feat_names = feat_names + raw_names

        return X, y, feat_names

    def _build_mix(self, animal_data: list, groups: dict, metric: str) -> tuple:
        """Horizontally concatenate cluster + group feature matrices."""
        Xc, yc, fc = self._build_freq_or_dur(animal_data, metric)
        Xg, _,  fg = self._build_group_feat(animal_data, groups, metric)
        # Deduplicate: if a group consists of a single cluster that's already
        # in the cluster matrix, drop it from the group side.
        keep = []
        cluster_labels_in_fc = set(fc)   # "Cluster N"
        for gi, gname in enumerate(fg):
            # Single-cluster group?  Check if "Cluster X" is already present.
            g_labels = (groups or {}).get(gname, {}).get("labels", [])
            if (len(g_labels) == 1
                    and f"Cluster {g_labels[0]}" in cluster_labels_in_fc):
                continue
            keep.append(gi)
        if keep:
            Xg_kept = Xg[:, keep]
            fg_kept  = [fg[k] for k in keep]
            X = np.hstack([Xc, Xg_kept])
            feat_names = fc + fg_kept
        else:
            X = Xc
            feat_names = fc
        return X, yc, feat_names

    @staticmethod
    def _build_custom(animal_data: list, sel_clusters: list,
                      sel_groups: dict, metric: str) -> tuple:
        """Feature matrix from user-selected clusters + user-selected groups."""
        Xc, yc, fc = GroupPredictorPanel._build_freq_or_dur(
            animal_data, metric, cluster_ids=set(sel_clusters))
        if sel_groups:
            Xg, _, fg = GroupPredictorPanel._build_group_feat(
                animal_data, sel_groups, metric)
            X = np.hstack([Xc, Xg])
            feat_names = list(fc) + list(fg)
        else:
            X, feat_names = Xc, list(fc)
        return X, yc, feat_names

    # ── Classification core ───────────────────────────────────────────────────

    def _make_pipeline(self, min_class_count: int, n_samples: int = 50,
                       n_features: int = None, feature_type: str = "freq_dur",
                       k: int = None, greedy_mode: bool = False,
                       algo: str = None):
        """Build a sklearn Pipeline for classification.

        greedy_mode=True  — used by _greedy_cluster_selection, _greedy_transition_selection,
                            and the _v() Shapley closures.  No inner CV: a fixed-C classifier
                            is used so hyperparameter search cannot inflate LOO scores on the
                            ~3-sample inner folds that arise with n≈12 animals.
                            Confirmed fix: LogisticRegressionCV inflated C4 (KW p=0.57) to
                            91.7% LOO and ranked it first; fixed LR C=1.0 drops it to 66.7%
                            and correctly selects C1 (KW p=0.055) first.  SVM greedy mode
                            drops GridSearchCV and probability=True to avoid degenerate Platt
                            scaling on LOO inner folds.
        greedy_mode=False — used by _run_loo (Path A/B): full CV flexibility is appropriate
                            because _run_loo already runs its own permutation test.
        algo              — optional pre-captured algorithm string; if None, reads
                            self._algo_var.get() (safe only on the main thread).
        """
        algo = algo if algo is not None else self._algo_var.get()
        try:
            from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
            from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
            from sklearn.preprocessing import RobustScaler, PolynomialFeatures
            from sklearn.decomposition import PCA
            from sklearn.pipeline import make_pipeline as _mkpipe

            # Step 1 — remove near-zero-variance features
            steps = [VarianceThreshold(threshold=1e-10)]

            # Step 2 — supervised feature selection (inside LOO, so no label leakage)
            # n_eff: number of features entering PolynomialFeatures / PCA sizing
            if k is not None and n_features is not None and k < n_features:
                effective_k = max(1, k)
                steps.append(SelectKBest(f_classif, k=effective_k))
                n_eff = effective_k
            else:
                n_eff = n_features

            # Step 3 — scale
            steps.append(RobustScaler())

            # Step 4 — pairwise interaction terms for freq/duration features only
            # Use n_eff (post-selection count) so Poly expansion is correctly sized.
            if feature_type == "freq_dur" and n_eff is not None and n_eff >= 2:
                steps.append(PolynomialFeatures(
                    degree=2, interaction_only=True, include_bias=False))
                # features after expansion: orig + orig*(orig-1)/2
                expanded = n_eff + n_eff * (n_eff - 1) // 2
            else:
                expanded = n_eff if n_eff is not None else n_samples

            # Step 5 — adaptive PCA: compress when features outnumber samples,
            # but only when the cohort is large enough for stable components.
            # Below _MIN_PCA_ANIMALS the LOO training sets are too small for
            # PCA to produce reliable components; L1 regularisation in the
            # classifier handles the high-dimensional regime instead.
            if (n_eff is not None
                    and expanded > max(5, n_samples)
                    and n_samples >= _MIN_PCA_ANIMALS):
                # Cap at n_samples-2 so LOO training (n_samples-1 rows, rank
                # n_samples-2) can always fill all components without a
                # whiten division-by-zero.
                n_comp = max(1, min(n_samples - 2, 15))
                steps.append(PCA(n_components=n_comp, whiten=True, random_state=42))

            if "Elastic" in algo:
                if greedy_mode:
                    # Fixed-C logistic regression — no inner CV, no hyperparameter search.
                    # Inner CV on ~3 samples per fold inflates LOO for clusters that happen
                    # to align with the fold order (not biological group separation).
                    # max_iter=500 / tol=1e-3 matches the exhaustive-search template and
                    # is sufficient for the (n-1) × k tiny problems in greedy/Shapley LOO.
                    clf = LogisticRegression(
                        C=1.0, solver="lbfgs",
                        class_weight="balanced", max_iter=500, tol=1e-3,
                        random_state=42)
                else:
                    # Full elastic-net CV for the main LOO path: _run_loo runs its own
                    # permutation test so the inner CV flexibility is appropriate here.
                    inner_cv = max(2, min(3, min_class_count))
                    cs  = [0.001, 0.01, 0.1, 1, 10] if n_samples < 25 \
                          else [0.0001, 0.001, 0.01, 0.1, 1, 10]
                    l1r = [0.3, 0.5, 0.7]            if n_samples < 25 \
                          else [0.1, 0.3, 0.5, 0.7, 0.9]
                    clf = LogisticRegressionCV(
                        Cs=cs, solver="saga",
                        l1_ratios=l1r, cv=inner_cv, max_iter=5000, tol=1e-3,
                        class_weight="balanced", random_state=42)
                steps.append(clf)
            else:   # SVM
                from sklearn.svm import SVC
                if greedy_mode:
                    # Fixed-C linear SVC — no GridSearchCV inner CV.
                    # probability=True is kept so Path A can draw the ROC panel and
                    # _extract_importances can read coef_.  The crash came from Platt-CV
                    # nested inside GridSearchCV (5-fold on ~7 samples); removing
                    # GridSearchCV fixes that — bare SVC Platt-CV on n_train≈11 is safe.
                    clf = SVC(C=1.0, kernel="linear", probability=True,
                              class_weight="balanced", max_iter=20000, random_state=42)
                else:
                    # Full grid search for the main LOO path; probability=True is needed
                    # for predict_proba() in _run_loo and coef_ in _extract_importances.
                    # error_score=0.0 prevents NaN propagation if an inner fold fails.
                    from sklearn.model_selection import GridSearchCV
                    inner_cv = max(2, min(3, min_class_count))
                    svc = SVC(kernel="linear", probability=True,
                              class_weight="balanced", max_iter=20000, random_state=42)
                    clf = GridSearchCV(
                        svc,
                        {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
                        cv=inner_cv,
                        scoring="balanced_accuracy",
                        error_score=0.0,
                        n_jobs=1,
                    )
                steps.append(clf)

            return _mkpipe(*steps)
        except ImportError as exc:
            raise RuntimeError(
                f"scikit-learn is required for Group Predictor: {exc}") from exc

    @staticmethod
    def _run_loo(pipeline, X: np.ndarray, y_enc: np.ndarray,
                 n_permutations: int = 199, progress_fn=None,
                 sample_weight=None) -> tuple:
        """
        Returns (loo_acc, pval, perm_scores, pred_labels, pred_proba, bal_acc).
        progress_fn(msg) is called periodically so callers can update UI.

        Actual LOO folds and permutation-test folds are all dispatched as a
        single flat job list so every available thread stays busy throughout.
        LeaveOneOut is deterministic: fold fi always tests sample fi, so
        flat_results[pi*n_folds + fi] is unambiguously the prediction for
        sample fi under permutation pi.
        """
        from sklearn.model_selection import LeaveOneOut
        from sklearn.base import clone as _clone
        from sklearn.metrics import balanced_accuracy_score
        from joblib import Parallel, delayed

        cv = LeaveOneOut()
        n = len(y_enc)
        n_classes = len(np.unique(y_enc))
        fold_splits = list(cv.split(X))   # fold fi: test_idx = [fi]
        n_folds = len(fold_splits)
        _clf_name = pipeline.steps[-1][0]  # e.g. "logisticregression", "svc"

        # ── LOO pass: actual predictions (parallel across folds) ──────────────
        # Send a regex-compatible message so _run_worker's progress bar advances.
        if progress_fn:
            progress_fn(f"LOO fold 1/{n}")

        def _eval_actual_fold(train_idx, test_idx):
            p = _clone(pipeline)
            try:
                _fit_kw = ({f"{_clf_name}__sample_weight": sample_weight[train_idx]}
                           if sample_weight is not None else {})
                p.fit(X[train_idx], y_enc[train_idx], **_fit_kw)
                lbl = int(p.predict(X[test_idx])[0])
                proba = (p.predict_proba(X[test_idx])[0]
                         if hasattr(p, "predict_proba") else np.zeros(n_classes))
            except Exception:
                vals, cnts = np.unique(y_enc[train_idx], return_counts=True)
                lbl = int(vals[np.argmax(cnts)])
                proba = np.zeros(n_classes)
            return test_idx[0], lbl, proba

        pred_labels = np.empty(n, dtype=int)
        pred_proba  = np.zeros((n, n_classes), dtype=float)
        for idx, lbl, proba in Parallel(n_jobs=_gp_n_jobs(), prefer="threads")(
                delayed(_eval_actual_fold)(tr, te) for tr, te in fold_splits):
            pred_labels[idx] = lbl
            try:
                pred_proba[idx] = proba   # shape guard: zeros if proba is wrong length
            except (ValueError, TypeError):
                pass

        loo_acc = (pred_labels == y_enc).mean()
        bal_acc = balanced_accuracy_score(y_enc, pred_labels)

        # Signal LOO completion so the progress bar advances to 65% of model slice.
        if progress_fn:
            progress_fn(f"LOO fold {n}/{n}")

        # ── Permutation test: flat (n_perm × n_folds) job list ────────────────
        # Pre-generate all shuffles so results are reproducible regardless of
        # thread scheduling order (each worker receives a fixed y_perm array).
        rng = np.random.default_rng(42)
        all_y_perms = [rng.permutation(y_enc) for _ in range(n_permutations)]

        if progress_fn:
            progress_fn(
                f"Permutation test ({n_permutations} perms × {n_folds} folds)")

        # Build a lighter pipeline for permutation folds.  The actual LOO pass
        # above uses LogisticRegressionCV (inner CV over 3 l1_ratios × 2-3 Cs),
        # which is appropriate for the real score.  For null folds, inner CV is
        # unnecessary: with random labels any C gives near-chance performance, so
        # the null distribution is essentially unchanged by fixing C=1.0.  This
        # reduces per-fold work from ~6 SAGA fits (CV) to 1, cutting perm-test
        # wall time by ~6× while keeping p-values conservative (real model uses a
        # more-optimised classifier than the null baseline).
        try:
            from sklearn.linear_model import (LogisticRegressionCV as _LRCV,
                                              LogisticRegression as _LRfast)
            from sklearn.pipeline import Pipeline as _Pipe
            from sklearn.model_selection import GridSearchCV as _GSCV
            _last_name, _last_est = pipeline.steps[-1]
            if isinstance(_last_est, _LRCV):
                _perm_clf = _LRfast(
                    C=1.0, solver="saga", l1_ratio=0.5,
                    class_weight="balanced", max_iter=500, tol=1e-3,
                    random_state=42)
                _perm_pipe = _Pipe(pipeline.steps[:-1] + [(_last_name, _perm_clf)])
            elif isinstance(_last_est, _GSCV):
                from sklearn.svm import SVC as _SVC
                _perm_clf = _SVC(C=1.0, kernel="linear", probability=True,
                                 class_weight="balanced", max_iter=20000,
                                 random_state=42)
                _perm_pipe = _Pipe(pipeline.steps[:-1] + [(_last_name, _perm_clf)])
            else:
                _perm_pipe = pipeline
        except Exception:
            _perm_pipe = pipeline

        def _eval_perm_fold(y_perm, train_idx, test_idx):
            # One fit for one (permutation, fold) pair.  Uses _perm_pipe (fixed-C
            # classifier) rather than the full CV pipeline for ~6× lower cost.
            # sample_weight is NOT permuted: it reflects recording-length quality,
            # not group membership, so the same weights apply to every null shuffle.
            # Use _clf_name (always defined) not _last_name (try-block scope).
            p = _clone(_perm_pipe)
            try:
                _fit_kw = ({f"{_clf_name}__sample_weight": sample_weight[train_idx]}
                           if sample_weight is not None else {})
                p.fit(X[train_idx], y_perm[train_idx], **_fit_kw)
                return int(p.predict(X[test_idx])[0])
            except Exception:
                vals, cnts = np.unique(y_perm[train_idx], return_counts=True)
                return int(vals[np.argmax(cnts)])

        # Batch into ~10 groups so progress_fn("Perm N/M") can fire between
        # batches and advance the bar from 65% to 100% of the model slice.
        # The loky process pool is reused across batches (get_reusable_executor)
        # so there is no per-batch re-spawn cost.
        # batch_size=n_folds groups one permutation's folds into a single IPC
        # transaction (n_folds jobs sent/received together), cutting round-trips
        # by n_folds while preserving per-fold task granularity for load balance.
        _BATCH = max(1, n_permutations // 10)
        _ipc_batch = max(1, n_folds)   # jobs per IPC transaction
        flat_results: list = []
        _perm_loky_ok = True
        for _b_start in range(0, n_permutations, _BATCH):
            _b_end = min(_b_start + _BATCH, n_permutations)
            # Pre-build task list so both the loky and fallback paths use
            # the same data without consuming a generator twice.
            _perm_tasks = [
                (all_y_perms[_pi], tr, te)
                for _pi in range(_b_start, _b_end)
                for tr, te in fold_splits]
            if _perm_loky_ok:
                try:
                    _batch = Parallel(n_jobs=_gp_n_jobs("loky"), backend="loky",
                                      max_nbytes=None, batch_size=_ipc_batch)(
                        delayed(_eval_perm_fold)(y_p, tr, te)
                        for y_p, tr, te in _perm_tasks)
                except Exception:
                    _perm_loky_ok = False
                    _batch = [_eval_perm_fold(y_p, tr, te)
                              for y_p, tr, te in _perm_tasks]
            else:
                _batch = [_eval_perm_fold(y_p, tr, te)
                          for y_p, tr, te in _perm_tasks]
            flat_results.extend(_batch)
            if progress_fn:
                progress_fn(f"Perm {_b_end}/{n_permutations}")

        # flat_results[pi*n_folds : (pi+1)*n_folds] = predictions for all n
        # samples under permutation pi (in sample order, since LOO is ordered).
        perm_scores = np.array([
            balanced_accuracy_score(
                all_y_perms[pi],
                flat_results[pi * n_folds : (pi + 1) * n_folds])
            for pi in range(n_permutations)])

        # +1 correction avoids p=0 (Phipson & Smyth 2010).
        # Compare against bal_acc: permuted models are also scored by balanced
        # accuracy, so the null distribution and test statistic match.
        pval = (np.sum(perm_scores >= bal_acc) + 1) / (n_permutations + 1)
        return loo_acc, pval, perm_scores, pred_labels, pred_proba, bal_acc

    @staticmethod
    def _run_one_null_perm(y_perm_groups, animal_data, greedy_metric,
                           greedy_cids, max_steps, min_class_count,
                           n_classes, cancel_event,
                           X_pre=None, y_pre=None, col_map_pre=None,
                           raw_mats_pre=None, cl_idx_pre=None,
                           sel_groups=None):
        """
        Single nested null permutation: re-runs greedy forward selection + LOO
        for one shuffled group-label array.  Returns float balanced_accuracy,
        or 1.0/n_classes on any failure.

        Fast path (loky-compatible): pass precomputed X_pre / raw_mats_pre so
        the inner greedy loop slices numpy arrays instead of calling
        _build_freq_or_dur (which calls compute_per_cluster_metrics for every
        animal on every candidate — the dominant cost at ~3 600 calls per perm).
        With numpy array arguments the caller should use backend="loky" for true
        process parallelism; cancel_event must be None (threading.Event is not
        pickleable across processes).

        Fallback: pass animal_data + cancel_event (original thread-based path,
        used automatically when precomputed arrays are not available).
        """
        from sklearn.preprocessing import (RobustScaler, PolynomialFeatures,
                                           LabelEncoder)
        from sklearn.linear_model import LogisticRegression as _LR
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline
        from sklearn.base import clone as _clone
        from sklearn.metrics import balanced_accuracy_score
        import numpy as _np

        chance = 1.0 / max(n_classes, 2)

        def _cancelled():
            return cancel_event is not None and cancel_event.is_set()

        # ── helper: build minimal greedy-mode pipeline ────────────────────────
        def _build_pipe(n_animals, n_feats):
            steps = [VarianceThreshold(threshold=1e-10), RobustScaler()]
            if n_feats >= 2:
                steps.append(PolynomialFeatures(
                    degree=2, interaction_only=True, include_bias=False))
                expanded = n_feats + n_feats * (n_feats - 1) // 2
            else:
                expanded = n_feats
            if expanded > max(5, n_animals):
                steps.append(PCA(n_components=max(1, min(n_animals - 2, 15)),
                                 whiten=True, random_state=42))
            steps.append(_LR(C=1.0, solver="lbfgs", class_weight="balanced",
                             max_iter=500, tol=1e-3, random_state=42))
            return make_pipeline(*steps)

        # ── helper: manual LOO on string-label arrays ─────────────────────────
        def _manual_loo_str(X, y_str):
            n = len(y_str)
            pipe = _build_pipe(n, X.shape[1])
            preds = _np.empty(n, dtype=y_str.dtype)
            for ti in range(n):
                mask = _np.ones(n, dtype=bool)
                mask[ti] = False
                p = _clone(pipe)
                try:
                    p.fit(X[mask], y_str[mask])
                    preds[ti] = p.predict(X[[ti]])[0]
                except Exception:
                    vals, cnts = _np.unique(y_str[mask], return_counts=True)
                    preds[ti] = vals[_np.argmax(cnts)]
            return preds

        # ── helper: get (X_c, y_c) for a candidate set ───────────────────────
        # Fast path: slice precomputed arrays (no DataFrame I/O).
        # Fallback: call the original builders (requires animal_data_perm).
        is_transition = (greedy_metric == "transition")
        y_perm_arr = _np.array(y_perm_groups)

        def _get_Xy(candidate, anim_perm):
            # freq/dur fast path
            if not is_transition and X_pre is not None and col_map_pre is not None:
                cs = [col_map_pre[c] for c in candidate if c in col_map_pre]
                if len(cs) == len(candidate):
                    return X_pre[:, cs], y_perm_arr
            # transition fast path
            if is_transition and raw_mats_pre is not None and cl_idx_pre is not None:
                try:
                    _ucls = sorted({c for p in candidate for c in p})
                    _sub = [cl_idx_pre[c] for c in _ucls]
                    _loc = {c: k for k, c in enumerate(_ucls)}
                    rows = []
                    for _raw in raw_mats_pre:
                        _sm = _raw[_np.ix_(_sub, _sub)]
                        _rs = _sm.sum(axis=1, keepdims=True)
                        _rs[_rs == 0] = 1.0
                        _sm = _sm / _rs
                        rows.append([_sm[_loc[pi], _loc[pj]]
                                     for pi, pj in candidate])
                    return _np.array(rows, dtype=float), y_perm_arr
                except Exception:
                    pass
            # fallback: rebuild from DataFrames
            if anim_perm is None:
                return None, None
            try:
                if is_transition:
                    X_c, y_c_list, _ = GroupPredictorPanel._build_transition_pairs(
                        anim_perm, candidate)
                else:
                    _c_ints = [c for c in candidate if isinstance(c, int)]
                    _c_gdict = ({n: sel_groups[n] for n in candidate
                                 if isinstance(n, str) and n in sel_groups}
                                if sel_groups else {})
                    if _c_ints and _c_gdict:
                        X_c, y_c_list, _ = GroupPredictorPanel._build_custom(
                            anim_perm, _c_ints, _c_gdict, greedy_metric)
                    elif _c_ints:
                        X_c, y_c_list, _ = GroupPredictorPanel._build_freq_or_dur(
                            anim_perm, greedy_metric, cluster_ids=set(_c_ints))
                    elif _c_gdict:
                        X_c, y_c_list, _ = GroupPredictorPanel._build_group_feat(
                            anim_perm, _c_gdict, greedy_metric)
                    else:
                        return None, None
                return X_c, _np.array(y_c_list)
            except Exception:
                return None, None

        try:
            # Shallow-copy only needed for the DataFrame fallback path
            anim_perm = (None if animal_data is None else
                         [dict(a, exp_group=g)
                          for a, g in zip(animal_data, y_perm_groups)])

            if is_transition:
                cids = sorted(greedy_cids)
                all_pairs = [(ci, cj) for ci in cids for cj in cids if ci != cj]
                remaining = list(all_pairs)
            else:
                remaining = list(greedy_cids)

            selected = []
            prev_acc = -1.0

            # Sequential greedy forward search
            for _step in range(max_steps):
                if _cancelled():
                    return chance

                best_acc_step, best_item = -1.0, None

                for item in remaining:
                    candidate = selected + [item]
                    X_c, y_c = _get_Xy(candidate, anim_perm)
                    if X_c is None or X_c.shape[1] == 0 or len(_np.unique(y_c)) < 2:
                        continue
                    preds_c = _manual_loo_str(X_c, y_c)
                    _uq_nc = _np.unique(y_c)
                    acc_c = float(_np.mean([(preds_c[y_c == _g] == _g).mean()
                                            for _g in _uq_nc]))
                    if acc_c > best_acc_step:
                        best_acc_step, best_item = acc_c, item

                if best_item is None:
                    break
                if selected and best_acc_step < prev_acc - 1e-9:
                    break
                selected.append(best_item)
                remaining.remove(best_item)
                prev_acc = best_acc_step

            if not selected:
                return chance

            X_f, y_f = _get_Xy(selected, anim_perm)
            if X_f is None or X_f.shape[1] == 0 or len(_np.unique(y_f)) < 2:
                return chance

            # Groups are already candidate feature units in the greedy search
            # (passed via greedy_cids) so no post-hoc append is needed here.

            preds_f = _manual_loo_str(X_f, y_f)
            le_f = LabelEncoder().fit(y_f)
            return float(balanced_accuracy_score(le_f.transform(y_f),
                                                 le_f.transform(preds_f)))

        except Exception:
            return chance

    @staticmethod
    def _extract_importances(pipeline, n_features: int) -> tuple:
        """Return (importances_1d, coef_matrix_2d) where matrix is (n_classes, n_features).
        Handles GridSearchCV(SVC) wrapper and back-projects through PCA if present."""
        try:
            clf = pipeline[-1]

            # GridSearchCV(SVC) — unwrap best estimator
            if hasattr(clf, "best_estimator_"):
                inner = clf.best_estimator_
                coef = inner.coef_
                coef_matrix = coef if coef.ndim == 2 else coef.reshape(1, -1)
            else:
                coef = clf.coef_
                coef_matrix = coef if coef.ndim == 2 else coef.reshape(1, -1)

            # Back-project through PCA if it appears in the pipeline steps
            pca = next((step for _, step in pipeline.steps[:-1]
                        if hasattr(step, "components_")), None)
            if pca is not None:
                # PCA.transform with whiten=True divides each component activation
                # by sqrt(explained_variance_[i]), so the effective projection matrix
                # is components_ / sqrt(ev)[:, np.newaxis], not components_ alone.
                # Without this correction the back-projection over-weights high-variance
                # PCs and under-weights the subtle PCs that carry group discrimination.
                if getattr(pca, "whiten", False):
                    eff = pca.components_ / (pca.explained_variance_[:, np.newaxis] ** 0.5)
                else:
                    eff = pca.components_
                coef_matrix = coef_matrix @ eff

            importances = np.abs(coef_matrix).mean(axis=0)
            return importances, coef_matrix
        except (AttributeError, ValueError):
            return np.zeros(n_features), None

    @staticmethod
    def _exhaustive_eval_combo(combo, X_full, y_c, col_map, n_animals,
                                base_pipe_template):
        """Pickleable per-combo LOO worker for _best_k_subsets (loky backend).
        Returns (balanced_accuracy, combo) or None if column mapping fails.
        All sklearn imports are local so each loky process initialises cleanly."""
        from sklearn.base import clone as _clone_sk
        from sklearn.metrics import balanced_accuracy_score
        import numpy as _np
        cs = [col_map[c] for c in combo if c in col_map]
        if len(cs) != len(combo):
            return None
        X_sub = X_full[:, cs]
        pred = _np.empty(n_animals, dtype=y_c.dtype)
        for test_i in range(n_animals):
            mask = _np.ones(n_animals, dtype=bool)
            mask[test_i] = False
            p = _clone_sk(base_pipe_template)
            try:
                p.fit(X_sub[mask], y_c[mask])
                pred[test_i] = p.predict(X_sub[[test_i]])[0]
            except Exception:
                vals, cnts = _np.unique(y_c[mask], return_counts=True)
                pred[test_i] = vals[_np.argmax(cnts)]
        return (float(balanced_accuracy_score(y_c, pred)), combo)

    @staticmethod
    def _exhaustive_eval_batch(combos, X_full, y_c, col_map, n_animals,
                               base_pipe_template):
        """Process a batch of combos in a single loky worker job.
        Packing multiple combos per job cuts per-job pickle/dispatch overhead
        for the large C(n,k) exhaustive search.
        Returns list parallel to `combos`: (balanced_accuracy, combo) or None."""
        from sklearn.base import clone as _clone_sk
        from sklearn.metrics import balanced_accuracy_score
        import numpy as _np
        out = []
        for combo in combos:
            cs = [col_map[c] for c in combo if c in col_map]
            if len(cs) != len(combo):
                out.append(None)
                continue
            X_sub = X_full[:, cs]
            pred = _np.empty(n_animals, dtype=y_c.dtype)
            for test_i in range(n_animals):
                mask = _np.ones(n_animals, dtype=bool)
                mask[test_i] = False
                p = _clone_sk(base_pipe_template)
                try:
                    p.fit(X_sub[mask], y_c[mask])
                    pred[test_i] = p.predict(X_sub[[test_i]])[0]
                except Exception:
                    vals, cnts = _np.unique(y_c[mask], return_counts=True)
                    pred[test_i] = vals[_np.argmax(cnts)]
            out.append((float(balanced_accuracy_score(y_c, pred)), combo))
        return out

    @staticmethod
    def _exhaustive_eval_pair_batch(combos, X_full, y_c, col_map, n_animals,
                                    base_pipe_template):
        """Batch loky worker for _best_k_pair_subsets.
        combos: list of k-tuples of pair-tuples, e.g. [((0,1),(2,3)), ...].
        X_full uses global row-normalised probabilities (from _build_transition).
        Returns list parallel to combos: (balanced_accuracy, combo) or None."""
        from sklearn.base import clone as _clone_sk
        from sklearn.metrics import balanced_accuracy_score
        import numpy as _np
        out = []
        for combo in combos:
            cs = [col_map[p] for p in combo if p in col_map]
            if len(cs) != len(combo):
                out.append(None)
                continue
            X_sub = X_full[:, cs]
            pred = _np.empty(n_animals, dtype=y_c.dtype)
            for test_i in range(n_animals):
                mask = _np.ones(n_animals, dtype=bool)
                mask[test_i] = False
                p = _clone_sk(base_pipe_template)
                try:
                    p.fit(X_sub[mask], y_c[mask])
                    pred[test_i] = p.predict(X_sub[[test_i]])[0]
                except Exception:
                    vals, cnts = _np.unique(y_c[mask], return_counts=True)
                    pred[test_i] = vals[_np.argmax(cnts)]
            out.append((float(balanced_accuracy_score(y_c, pred)), combo))
        return out

    def _best_k_subsets(self, animal_data: list, cluster_ids: list,
                        k: int,
                        top_n: int = 5, progress_fn=None,
                        metric: str = "frequency") -> tuple:
        """
        Exhaustive LOO search over all C(n, k) cluster subsets using
        joblib CPU parallelism (no GPU — fits are too small to amortise transfer).

        Returns (trace, top_combos):
          trace      — [(cid, best_bal_acc), ...] for the best combo, asc by cid.
                       Each entry carries the same bal_acc (whole-combo score).
                       Format is Path-A-compatible: [cid for cid, _ in trace].
          top_combos — [(combo_tuple, bal_acc), ...] top-top_n sorted descending.
        Returns ([], []) on total failure so callers fall back to greedy.
        """
        import math
        from itertools import combinations as _combos
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.preprocessing import RobustScaler, PolynomialFeatures
        from sklearn.linear_model import LogisticRegression as _LR
        from joblib import Parallel, delayed

        cids = sorted(cluster_ids)
        n    = len(cids)
        if k < 1 or k > n:
            return [], []

        # ── Precompute full feature matrix once ───────────────────────────────
        try:
            X_full, y_list, feat_names = self._build_freq_or_dur(
                animal_data, metric, cluster_ids=set(cids))
        except Exception:
            return [], []
        y_c = np.array(y_list)
        if X_full.shape[1] == 0 or len(np.unique(y_c)) < 2:
            return [], []

        # Map cluster_id → column index in X_full
        col_map = {}
        for ci, fname in enumerate(feat_names):
            try:
                col_map[int(fname.split()[-1])] = ci
            except (ValueError, IndexError):
                pass

        # ── Per-fold pipeline matching Path A exactly ─────────────────────────
        # Scale, poly-expand and (optionally) PCA are all fitted inside each LOO
        # fold so the held-out animal never leaks into the preprocessing step.
        # Previously the scaler was fitted on all n_animals before the loop,
        # which inflated scores vs the confirmed LOO shown in the overview —
        # most visibly for weak-signal models where true LOO ≈ chance.
        from sklearn.pipeline import make_pipeline as _mkpipe
        from sklearn.decomposition import PCA as _PCA
        from sklearn.feature_selection import VarianceThreshold as _VT

        _n_animals = len(y_c)
        total      = math.comb(n, k)
        all_combos = list(_combos(cids, k))

        # Build the pipeline template ONCE — the structure is identical for every
        # combo because k (number of columns) is constant across the exhaustive
        # search.  max_iter=500 / tol=1e-3 is sufficient for the (n-1) × k small
        # problems here; the old limit of 3000 was borrowed from the full LOO path.
        # VarianceThreshold matches _make_pipeline step 1 and prevents zero-variance
        # columns from inflating the PolynomialFeatures expansion count.
        _steps_tpl = [_VT(threshold=1e-10), RobustScaler()]
        if k >= 2:
            _steps_tpl.append(PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False))
            _expanded_tpl = k + k * (k - 1) // 2
        else:
            _expanded_tpl = k
        if _expanded_tpl > max(5, _n_animals):
            _steps_tpl.append(_PCA(n_components=max(1, min(_n_animals - 2, 15)),
                                   whiten=True, random_state=42))
        _steps_tpl.append(_LR(C=1.0, solver="lbfgs",
                               class_weight="balanced", max_iter=500,
                               tol=1e-3, random_state=42))
        _base_pipe_template = _mkpipe(*_steps_tpl)

        if progress_fn:
            progress_fn(f"Exhaustive: 0/{total} combos (k={k})")
        # Chunk combos so each loky job handles ~total/(n_workers*20) combos.
        # This cuts per-job dispatch/pickle overhead 20-50x vs one-job-per-combo
        # while still giving ≥ n_workers jobs per Parallel call (all workers busy)
        # and ≈20 progress bar ticks across the full search.
        _n_workers_ex = _gp_n_jobs("loky")
        _chunk_sz = max(1, total // (_n_workers_ex * 20))
        _chunks = [all_combos[_i:_i + _chunk_sz]
                   for _i in range(0, total, _chunk_sz)]
        _n_chunks = len(_chunks)
        # Each progress batch dispatches exactly n_workers chunks so all
        # workers stay busy; number of bar ticks ≈ n_chunks / n_workers.
        _PROG_EX = max(1, _n_workers_ex)
        _raw: list = []
        _loky_ok = True
        for _b_start in range(0, _n_chunks, _PROG_EX):
            _b_end = min(_b_start + _PROG_EX, _n_chunks)
            if _loky_ok:
                try:
                    _batch_res = Parallel(n_jobs=_n_workers_ex, backend="loky",
                                          max_nbytes=None)(
                        delayed(GroupPredictorPanel._exhaustive_eval_batch)(
                            _chunks[_ci], X_full, y_c, col_map, _n_animals,
                            _base_pipe_template)
                        for _ci in range(_b_start, _b_end))
                except Exception:
                    _loky_ok = False
                    _batch_res = [
                        GroupPredictorPanel._exhaustive_eval_batch(
                            _chunks[_ci], X_full, y_c, col_map, _n_animals,
                            _base_pipe_template)
                        for _ci in range(_b_start, _b_end)]
            else:
                _batch_res = [
                    GroupPredictorPanel._exhaustive_eval_batch(
                        _chunks[_ci], X_full, y_c, col_map, _n_animals,
                        _base_pipe_template)
                    for _ci in range(_b_start, _b_end)]
            for _chunk_out in _batch_res:
                _raw.extend(_chunk_out)
            if progress_fn:
                progress_fn(
                    f"Exhaustive: {min(_b_end * _chunk_sz, total)}/{total}"
                    f" combos (k={k})")
        results = [r for r in _raw if r is not None]
        if not results:
            return [], []

        results.sort(key=lambda x: x[0], reverse=True)
        top_combos = [(combo, bal_acc) for bal_acc, combo in results[:top_n]]
        best_bal_acc, best_combo = results[0]
        trace = [(cid, best_bal_acc) for cid in sorted(best_combo)]
        return trace, top_combos

    def _best_k_pair_subsets(self, animal_data: list, all_pairs: list,
                              k: int, top_n: int = 5,
                              progress_fn=None) -> tuple:
        """
        Exhaustive LOO search over all C(n_pairs, k) directed-pair subsets.

        Uses global row-normalised transition probabilities (same normalisation
        as _build_transition) so the full matrix can be precomputed once and
        pair combos evaluated by column slicing — identical performance pattern
        to _best_k_subsets for clusters.

        Returns (trace, top_combos):
          trace      — [((i,j), best_bal_acc), ...] for the best combo, sorted
                       by pair.  Format is Path-A-compatible.
          top_combos — [(combo_tuple_of_pairs, bal_acc), ...] top-top_n desc.
        Returns ([], []) on failure so callers fall back to greedy.
        """
        import math
        import re as _re_kp
        from itertools import combinations as _combos
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.linear_model import LogisticRegression as _LR
        from sklearn.preprocessing import RobustScaler
        from sklearn.feature_selection import VarianceThreshold as _VT
        from sklearn.pipeline import make_pipeline as _mkpipe
        from joblib import Parallel, delayed

        pairs = sorted(all_pairs)
        n_pairs = len(pairs)
        if k < 1 or k > n_pairs:
            return [], []

        # Precompute the full global-normalised transition matrix once.
        # Row normalisation is over all clusters (not just the selected subset),
        # making each feature independent of which other pairs are in the combo
        # and allowing column slicing rather than per-combo matrix reconstruction.
        try:
            X_full, y_list, feat_names = self._build_transition(animal_data)
        except Exception:
            return [], []
        y_c = np.array(y_list)
        if X_full.shape[1] == 0 or len(np.unique(y_c)) < 2:
            return [], []

        # Map each (i, j) pair to its column index in X_full
        col_map: dict = {}
        for ci, fname in enumerate(feat_names):
            m = _re_kp.match(r"T_(\d+)→(\d+)", fname)
            if m:
                col_map[(int(m.group(1)), int(m.group(2)))] = ci

        valid_pairs = [p for p in pairs if p in col_map]
        if len(valid_pairs) < k:
            return [], []

        # Pipeline: VT → RobustScaler → fixed-C LR (no PolynomialFeatures —
        # transition pairs already encode two-cluster relations).
        _n_animals = len(y_c)
        _steps_tpl = [_VT(threshold=1e-10), RobustScaler(),
                      _LR(C=1.0, solver="lbfgs", class_weight="balanced",
                          max_iter=500, tol=1e-3, random_state=42)]
        _base_pipe = _mkpipe(*_steps_tpl)

        total     = math.comb(len(valid_pairs), k)
        all_combos = list(_combos(valid_pairs, k))

        if progress_fn:
            progress_fn(f"Exhaustive: 0/{total} pair-combos (k={k})")

        _n_workers_ex = _gp_n_jobs("loky")
        _chunk_sz  = max(1, total // (_n_workers_ex * 20))
        _chunks    = [all_combos[_i:_i + _chunk_sz]
                      for _i in range(0, total, _chunk_sz)]
        _n_chunks  = len(_chunks)
        _PROG_EX   = max(1, _n_workers_ex)
        _raw: list = []
        _loky_ok   = True

        for _b_start in range(0, _n_chunks, _PROG_EX):
            _b_end = min(_b_start + _PROG_EX, _n_chunks)
            if _loky_ok:
                try:
                    _batch_res = Parallel(
                        n_jobs=_n_workers_ex, backend="loky",
                        max_nbytes=None)(
                        delayed(GroupPredictorPanel._exhaustive_eval_pair_batch)(
                            _chunks[_ci], X_full, y_c, col_map,
                            _n_animals, _base_pipe)
                        for _ci in range(_b_start, _b_end))
                except Exception:
                    _loky_ok = False
                    _batch_res = [
                        GroupPredictorPanel._exhaustive_eval_pair_batch(
                            _chunks[_ci], X_full, y_c, col_map,
                            _n_animals, _base_pipe)
                        for _ci in range(_b_start, _b_end)]
            else:
                _batch_res = [
                    GroupPredictorPanel._exhaustive_eval_pair_batch(
                        _chunks[_ci], X_full, y_c, col_map,
                        _n_animals, _base_pipe)
                    for _ci in range(_b_start, _b_end)]
            for _chunk_out in _batch_res:
                _raw.extend(_chunk_out)
            if progress_fn:
                progress_fn(
                    f"Exhaustive: {min(_b_end * _chunk_sz, total)}/{total}"
                    f" pair-combos (k={k})")

        results = [r for r in _raw if r is not None]
        if not results:
            return [], []

        results.sort(key=lambda x: x[0], reverse=True)
        top_combos = [(combo, bal_acc) for bal_acc, combo in results[:top_n]]
        best_bal_acc, best_combo = results[0]
        trace = [(pair, best_bal_acc) for pair in sorted(best_combo)]
        return trace, top_combos

    def _greedy_cluster_selection(self, animal_data: list, cluster_ids: list,
                                   min_class_count: int,
                                   max_steps: int = 12,
                                   progress_fn=None,
                                   metric: str = "frequency",
                                   algo=None) -> tuple:
        """
        Greedy forward cluster selection (LOO, no permutation test, for speed).
        Returns (trace, ties_by_step) where:
          trace — [(cluster_id, cumulative_balanced_accuracy), ...] in selection order.
          ties_by_step — {step_index: [(bal_acc, cluster_id), ...]} of all candidates
                         that tied for the best score at each step (including winner).
        Each step adds the cluster that most improves balanced LOO accuracy.
        max_steps: stop after this many clusters are added (driven by Max Contributors).
        metric: "frequency" | "total_duration" | "transition"
        Candidate clusters within each step are evaluated in parallel (prefer threads).
        """
        from sklearn.model_selection import LeaveOneOut
        from sklearn.base import clone as _clone
        from joblib import Parallel, delayed

        remaining = sorted(cluster_ids)
        selected: list = []
        trace: list = []
        _ties_by_step: dict = {}
        loo = LeaveOneOut()
        _pipe_ft = "transition" if metric == "transition" else "freq_dur"

        # Precompute the full feature matrix for freq/dur metrics so _eval_cluster
        # can slice columns instead of calling _build_freq_or_dur (which calls
        # compute_per_cluster_metrics for every animal) on every candidate at
        # every greedy step. For transition the row-renormalisation per subset
        # means we cannot slice a precomputed matrix, so that path stays as-is.
        _X_pre = _y_pre = _col_map_pre = None
        if metric != "transition":
            try:
                _X_pre, _y_pre_list, _fn_pre = self._build_freq_or_dur(
                    animal_data, metric, cluster_ids=set(remaining))
                _y_pre = np.array(_y_pre_list)
                _col_map_pre = {}
                for _ci, _fn in enumerate(_fn_pre):
                    try:
                        _col_map_pre[int(_fn.split()[-1])] = _ci
                    except (ValueError, IndexError):
                        pass
            except Exception:
                _X_pre = None

        _greedy_loky_ok = True   # flipped False on first loky failure; stays sequential
        while remaining and len(trace) < max_steps:
            if self._cancel_event.is_set():
                break
            if progress_fn:
                progress_fn(
                    f"Greedy step {len(trace) + 1}/{min(max_steps, len(cluster_ids))}"
                    f" — testing {len(remaining)} candidate(s)…")

            _selected_snap = list(selected)

            # ── Phase 1: build X_c for each candidate once ────────────────────
            candidate_Xs: dict = {}
            for c in remaining:
                _cand = _selected_snap + [c]
                try:
                    if metric == "transition":
                        _Xc, _yc_list, _ = self._build_transition(
                            animal_data, cluster_ids=set(_cand))
                        _yc = np.array(_yc_list)
                    elif (_X_pre is not None
                          and all(cc in _col_map_pre for cc in _cand)):
                        _Xc = _X_pre[:, [_col_map_pre[cc] for cc in _cand]]
                        _yc = _y_pre
                    else:
                        _Xc, _yc_list, _ = self._build_freq_or_dur(
                            animal_data, metric, cluster_ids=set(_cand))
                        _yc = np.array(_yc_list)
                    if _Xc.shape[1] > 0 and len(np.unique(_yc)) >= 2:
                        candidate_Xs[c] = (_Xc, _yc)
                except Exception:
                    pass

            if not candidate_Xs:
                break

            _n_anim_g = next(iter(candidate_Xs.values()))[1].shape[0]
            _fold_splits_g = list(loo.split(range(_n_anim_g)))
            _n_folds_g = len(_fold_splits_g)
            try:
                _pipe_tpl_g = self._make_pipeline(
                    min_class_count, _n_anim_g,
                    len(_selected_snap) + 1, _pipe_ft, greedy_mode=True,
                    algo=algo)
            except Exception:
                break

            # ── Phase 2: flat candidate × fold task list ───────────────────────
            # Pin dicts/objects as defaults so each iteration's closure is
            # independent of later rebindings of the loop-level names.
            def _eval_cand_fold(c, tr, te,
                                _cxs=candidate_Xs, _ptpl=_pipe_tpl_g):
                _Xc, _yc = _cxs[c]
                p = _clone(_ptpl)
                try:
                    p.fit(_Xc[tr], _yc[tr])
                    return int(p.predict(_Xc[te])[0]), int(_yc[te[0]])
                except Exception:
                    vals, cnts = np.unique(_yc[tr], return_counts=True)
                    return int(vals[np.argmax(cnts)]), int(_yc[te[0]])

            _valid_cands = [c for c in remaining if c in candidate_Xs]
            _flat_tasks  = [(c, tr, te)
                            for c in _valid_cands
                            for tr, te in _fold_splits_g]
            if _greedy_loky_ok:
                try:
                    _flat_res = Parallel(n_jobs=_gp_n_jobs("loky"), backend="loky",
                                         max_nbytes=None)(
                        delayed(_eval_cand_fold)(c, tr, te) for c, tr, te in _flat_tasks)
                except Exception:
                    _greedy_loky_ok = False
                    _flat_res = [_eval_cand_fold(c, tr, te) for c, tr, te in _flat_tasks]
            else:
                _flat_res = [_eval_cand_fold(c, tr, te) for c, tr, te in _flat_tasks]

            valid = []
            for _ci, c in enumerate(_valid_cands):
                _chunk = _flat_res[_ci * _n_folds_g : (_ci + 1) * _n_folds_g]
                _preds = np.array([r[0] for r in _chunk])
                _trues = np.array([r[1] for r in _chunk])
                _uq = np.unique(_trues)
                if len(_uq) < 2:
                    continue
                _bal = float(np.mean([(_preds[_trues == g] == g).mean()
                                      for g in _uq]))
                valid.append((_bal, c))
            if not valid:
                break
            _best_step_acc = max(v[0] for v in valid)
            _tied_step = [(acc, c) for acc, c in valid
                          if abs(acc - _best_step_acc) < 1e-9]
            _ties_by_step[len(trace)] = _tied_step
            best_acc, best_c = _tied_step[0][0], _tied_step[0][1]
            # Early stopping: don't add a cluster that decreases accuracy
            # from the previous step (plateau is allowed; decline is not).
            if trace and best_acc < trace[-1][1] - 1e-9:
                break
            selected.append(best_c)
            remaining.remove(best_c)
            trace.append((best_c, best_acc))
        return trace, _ties_by_step

    def _greedy_mixed_selection(
        self,
        animal_data: list,
        feature_units: list,
        groups_dict: dict,
        min_class_count: int,
        max_steps: int = 12,
        progress_fn=None,
        metric: str = "frequency",
        algo=None
    ) -> tuple:
        """
        Greedy forward selection over a mixed pool of cluster IDs (int) and
        group names (str).  Evaluates clusters and user-defined behaviour groups
        as co-equal candidate feature units at each step.

        Returns (trace, ties_by_step) — identical shape to
        _greedy_cluster_selection but trace entries may be int or str.
        """
        from sklearn.model_selection import LeaveOneOut
        from sklearn.base import clone as _clone
        from joblib import Parallel, delayed

        remaining = list(feature_units)
        selected: list = []
        trace: list = []
        _ties_by_step: dict = {}
        loo = LeaveOneOut()

        def _build_for_candidate(candidate):
            _cids = [c for c in candidate if isinstance(c, int)]
            _gnames = {n: groups_dict[n] for n in candidate
                       if isinstance(n, str) and n in groups_dict}
            try:
                if _cids and _gnames:
                    return GroupPredictorPanel._build_custom(
                        animal_data, _cids, _gnames, metric)
                elif _cids:
                    return GroupPredictorPanel._build_freq_or_dur(
                        animal_data, metric, cluster_ids=set(_cids))
                elif _gnames:
                    return GroupPredictorPanel._build_group_feat(
                        animal_data, _gnames, metric)
            except Exception:
                pass
            return None, None, None

        _mixed_loky_ok = True   # flipped False on first loky failure; stays sequential
        while remaining and len(trace) < max_steps:
            if self._cancel_event.is_set():
                break
            if progress_fn:
                progress_fn(
                    f"Greedy step {len(trace) + 1}/{min(max_steps, len(feature_units))}"
                    f" — testing {len(remaining)} candidate(s) (mixed mode)…")

            _selected_snap = list(selected)

            # ── Phase 1: build X_c for each candidate unit once ───────────────
            _unit_Xs: dict = {}
            for u in remaining:
                _cand_m = _selected_snap + [u]
                try:
                    _Xc_m, _yc_ml, _ = _build_for_candidate(_cand_m)
                    if _Xc_m is None or _Xc_m.shape[1] == 0:
                        continue
                    _yc_m = np.array(_yc_ml)
                    if len(np.unique(_yc_m)) >= 2:
                        _unit_Xs[u] = (_Xc_m, _yc_m)
                except Exception:
                    pass

            if not _unit_Xs:
                break

            _n_anim_m = next(iter(_unit_Xs.values()))[1].shape[0]
            _fold_splits_m = list(loo.split(range(_n_anim_m)))
            _n_folds_m = len(_fold_splits_m)
            try:
                _pipe_tpl_m = self._make_pipeline(
                    min_class_count, _n_anim_m,
                    len(_selected_snap) + 1, "freq_dur", greedy_mode=True,
                    algo=algo)
            except Exception:
                break

            # ── Phase 2: flat unit × fold task list ────────────────────────────
            def _eval_unit_fold(u, tr, te,
                                _uxs=_unit_Xs, _ptpl=_pipe_tpl_m):
                _Xc_m, _yc_m = _uxs[u]
                p = _clone(_ptpl)
                try:
                    p.fit(_Xc_m[tr], _yc_m[tr])
                    return int(p.predict(_Xc_m[te])[0]), int(_yc_m[te[0]])
                except Exception:
                    vals, cnts = np.unique(_yc_m[tr], return_counts=True)
                    return int(vals[np.argmax(cnts)]), int(_yc_m[te[0]])

            _valid_units = [u for u in remaining if u in _unit_Xs]
            _flat_tasks_m = [(u, tr, te)
                             for u in _valid_units
                             for tr, te in _fold_splits_m]
            if _mixed_loky_ok:
                try:
                    _flat_res_m = Parallel(n_jobs=_gp_n_jobs("loky"), backend="loky",
                                           max_nbytes=None)(
                        delayed(_eval_unit_fold)(u, tr, te)
                        for u, tr, te in _flat_tasks_m)
                except Exception:
                    _mixed_loky_ok = False
                    _flat_res_m = [_eval_unit_fold(u, tr, te) for u, tr, te in _flat_tasks_m]
            else:
                _flat_res_m = [_eval_unit_fold(u, tr, te) for u, tr, te in _flat_tasks_m]

            valid = []
            for _ci, u in enumerate(_valid_units):
                _chunk_m = _flat_res_m[_ci * _n_folds_m : (_ci + 1) * _n_folds_m]
                _preds_m = np.array([r[0] for r in _chunk_m])
                _trues_m = np.array([r[1] for r in _chunk_m])
                _uq_m = np.unique(_trues_m)
                if len(_uq_m) < 2:
                    continue
                _bal_m = float(np.mean([(_preds_m[_trues_m == g] == g).mean()
                                        for g in _uq_m]))
                valid.append((_bal_m, u))
            if not valid:
                break
            _best_step_acc = max(v[0] for v in valid)
            _tied_step = [(acc, u) for acc, u in valid
                          if abs(acc - _best_step_acc) < 1e-9]
            _ties_by_step[len(trace)] = _tied_step
            best_acc, best_u = _tied_step[0][0], _tied_step[0][1]
            if trace and best_acc < trace[-1][1] - 1e-9:
                break
            selected.append(best_u)
            remaining.remove(best_u)
            trace.append((best_u, best_acc))
        return trace, _ties_by_step

    def _greedy_transition_selection(self, animal_data: list, cluster_ids,
                                     min_class_count: int,
                                     max_steps: int = 12,
                                     progress_fn=None,
                                     algo=None,
                                     allowed_pairs=None) -> tuple:
        """
        Greedy forward selection at the directed-edge (T_i→j) level.
        Returns (trace, ties_by_step) where:
          trace — [((i,j), cumulative_balanced_accuracy), …] in selection order.
          ties_by_step — {step_index: [(bal_acc, (i,j)), ...]} of all candidates
                         that tied for the best score at each step (including winner).
        Each step adds the directed transition that most improves balanced LOO accuracy.
        Candidate pairs within each step are evaluated in parallel (prefer threads).
        allowed_pairs: optional pre-filtered list of (i,j) pairs to consider instead
                       of the full cross-product of cluster_ids (used to restrict to
                       above-chance transitions).
        """
        from sklearn.model_selection import LeaveOneOut
        from sklearn.base import clone as _clone
        from joblib import Parallel, delayed

        cl = sorted(cluster_ids)
        all_pairs = (list(allowed_pairs) if allowed_pairs is not None
                     else [(ci, cj) for ci in cl for cj in cl if ci != cj])
        remaining = list(all_pairs)
        selected: list = []
        trace: list = []
        _ties_by_step: dict = {}
        loo = LeaveOneOut()

        # Precompute raw (un-normalised) per-animal transition count matrices for
        # all clusters once.  _eval_pair then extracts the subset, normalises and
        # picks the pair columns — identical to _build_transition_pairs but without
        # iterating over all frames on every candidate evaluation.
        _cl_arr_pre = np.array(cl)
        _cl_idx_pre = {c: k for k, c in enumerate(cl)}
        _n_pre = len(cl)
        _y_pairs_pre = [a["exp_group"] for a in animal_data]
        _raw_mats_pre: list = []
        for _a in animal_data:
            _lbls = _a["df"]["label"].values
            _mat = np.zeros((_n_pre, _n_pre), dtype=float)
            if len(_lbls) > 1:
                _valid = np.isin(_lbls, _cl_arr_pre)
                _idxs = np.searchsorted(_cl_arr_pre, _lbls)
                _vi = np.where(_valid)[0]
                if len(_vi) > 1:
                    _consec = np.where(np.diff(_vi) == 1)[0]
                    np.add.at(_mat,
                              (_idxs[_vi[_consec]], _idxs[_vi[_consec + 1]]),
                              1.0)
            _raw_mats_pre.append(_mat)

        _trans_loky_ok = True   # flipped False on first loky failure; stays sequential
        while remaining and len(trace) < max_steps:
            if self._cancel_event.is_set():
                break
            if progress_fn:
                progress_fn(
                    f"Greedy step {len(trace) + 1}/{min(max_steps, len(all_pairs))}"
                    f" — testing {len(remaining)} transitions…")

            _selected_snap = list(selected)

            # ── Phase 1: build X_c for each candidate pair once ───────────────
            _pair_Xs: dict = {}
            for pair in remaining:
                _cand_t = _selected_snap + [pair]
                try:
                    _ucls_t = sorted({c for _pp in _cand_t for c in _pp})
                    _sub_idxs_t = [_cl_idx_pre[c] for c in _ucls_t]
                    _ci_loc_t = {c: k for k, c in enumerate(_ucls_t)}
                    _rows_t = []
                    for _raw in _raw_mats_pre:
                        _sub_t = _raw[np.ix_(_sub_idxs_t, _sub_idxs_t)]
                        _rs_t = _sub_t.sum(axis=1, keepdims=True)
                        _rs_t[_rs_t == 0] = 1.0
                        _sub_t = _sub_t / _rs_t
                        _rows_t.append([_sub_t[_ci_loc_t[pi], _ci_loc_t[pj]]
                                        for pi, pj in _cand_t])
                    _Xc_t = np.array(_rows_t, dtype=float)
                    _yc_t = np.array(_y_pairs_pre)
                except Exception:
                    try:
                        _Xc_t, _yc_tl, _ = self._build_transition_pairs(
                            animal_data, _cand_t)
                        _yc_t = np.array(_yc_tl)
                    except Exception:
                        continue
                if _Xc_t.shape[1] > 0 and len(np.unique(_yc_t)) >= 2:
                    _pair_Xs[pair] = (_Xc_t, _yc_t)

            if not _pair_Xs:
                break

            _n_anim_t = next(iter(_pair_Xs.values()))[1].shape[0]
            _fold_splits_t = list(loo.split(range(_n_anim_t)))
            _n_folds_t = len(_fold_splits_t)
            try:
                _pipe_tpl_t = self._make_pipeline(
                    min_class_count, _n_anim_t,
                    len(_selected_snap) + 1, "transition", greedy_mode=True,
                    algo=algo)
            except Exception:
                break

            # ── Phase 2: flat pair × fold task list ────────────────────────────
            def _eval_pair_fold(pair, tr, te,
                                _pxs=_pair_Xs, _ptpl=_pipe_tpl_t):
                _Xc_t, _yc_t = _pxs[pair]
                p = _clone(_ptpl)
                try:
                    p.fit(_Xc_t[tr], _yc_t[tr])
                    return int(p.predict(_Xc_t[te])[0]), int(_yc_t[te[0]])
                except Exception:
                    vals, cnts = np.unique(_yc_t[tr], return_counts=True)
                    return int(vals[np.argmax(cnts)]), int(_yc_t[te[0]])

            _valid_pairs = [pair for pair in remaining if pair in _pair_Xs]
            _flat_tasks_t = [(pair, tr, te)
                             for pair in _valid_pairs
                             for tr, te in _fold_splits_t]
            if _trans_loky_ok:
                try:
                    _flat_res_t = Parallel(n_jobs=_gp_n_jobs("loky"), backend="loky",
                                           max_nbytes=None)(
                        delayed(_eval_pair_fold)(pair, tr, te)
                        for pair, tr, te in _flat_tasks_t)
                except Exception:
                    _trans_loky_ok = False
                    _flat_res_t = [_eval_pair_fold(pair, tr, te)
                                   for pair, tr, te in _flat_tasks_t]
            else:
                _flat_res_t = [_eval_pair_fold(pair, tr, te)
                               for pair, tr, te in _flat_tasks_t]

            valid = []
            for _ci, pair in enumerate(_valid_pairs):
                _chunk_t = _flat_res_t[_ci * _n_folds_t : (_ci + 1) * _n_folds_t]
                _preds_t = np.array([r[0] for r in _chunk_t])
                _trues_t = np.array([r[1] for r in _chunk_t])
                _uq_t = np.unique(_trues_t)
                if len(_uq_t) < 2:
                    continue
                _bal_t = float(np.mean([(_preds_t[_trues_t == g] == g).mean()
                                        for g in _uq_t]))
                valid.append((_bal_t, pair))
            if not valid:
                break
            _best_step_acc = max(v[0] for v in valid)
            _tied_step = [(acc, p) for acc, p in valid
                          if abs(acc - _best_step_acc) < 1e-9]
            _ties_by_step[len(trace)] = _tied_step
            best_acc, best_pair = _tied_step[0][0], _tied_step[0][1]
            if trace and best_acc < trace[-1][1] - 1e-9:
                break
            selected.append(best_pair)
            remaining.remove(best_pair)
            trace.append((best_pair, best_acc))
        return trace, _ties_by_step

    def _compute_shapley(self, animal_data: list, selected_cids: list,
                         min_class_count: int, n_classes: int,
                         progress_fn=None, metric: str = "frequency",
                         sel_groups: dict = None, algo=None):
        """
        Shapley-based cluster importance.
        Returns (baseline_acc, [(cluster_id, phi), ...]) sorted desc by phi,
        or None on failure.
        Exact (all 2^N coalitions) for N <= 8; Monte-Carlo (150 permutations)
        for larger N.  phi values sum to baseline_acc - chance_level.
        metric: "frequency" | "total_duration" | "transition"
        """
        import math
        from itertools import combinations
        import random as _random
        from sklearn.model_selection import LeaveOneOut
        from sklearn.base import clone as _clone

        # Use the pre-captured algo string (passed from main thread via _run_worker).
        # Fallback to self._algo_var.get() only when called directly from the main thread
        # (e.g. standalone tests), since StringVar.get() is not safe off the main thread.
        _algo_snap = algo if algo is not None else self._algo_var.get()

        chance = 1.0 / max(n_classes, 2)
        loo = LeaveOneOut()
        v_cache = {}

        def _v(key: frozenset) -> float:
            if key in v_cache:
                return v_cache[key]
            if not key:
                v_cache[key] = chance
                return chance
            try:
                if metric == "transition":
                    X, y_list, _ = self._build_transition(
                        animal_data, cluster_ids=key)
                elif sel_groups:
                    _key_cids = [c for c in key if isinstance(c, int)]
                    _key_gdict = {n: sel_groups[n] for n in key
                                  if isinstance(n, str) and n in sel_groups}
                    if _key_cids and _key_gdict:
                        X, y_list, _ = GroupPredictorPanel._build_custom(
                            animal_data, _key_cids, _key_gdict, metric)
                    elif _key_cids:
                        X, y_list, _ = self._build_freq_or_dur(
                            animal_data, metric,
                            cluster_ids=frozenset(_key_cids))
                    elif _key_gdict:
                        X, y_list, _ = GroupPredictorPanel._build_group_feat(
                            animal_data, _key_gdict, metric)
                    else:
                        v_cache[key] = chance
                        return chance
                else:
                    X, y_list, _ = self._build_freq_or_dur(
                        animal_data, metric, cluster_ids=key)
                y = np.array(y_list)
                if X.shape[1] == 0 or len(np.unique(y)) < 2:
                    v_cache[key] = chance
                    return chance
                _pipe_ft = "transition" if metric == "transition" else "freq_dur"
                pipe = self._make_pipeline(
                    min_class_count, len(y), X.shape[1], _pipe_ft,
                    greedy_mode=True, algo=_algo_snap)
                pred = np.empty(len(y), dtype=y.dtype)
                for tr, te in loo.split(X):
                    p = _clone(pipe)
                    try:
                        p.fit(X[tr], y[tr])
                        pred[te[0]] = p.predict(X[te])[0]
                    except Exception:
                        vals, cnts = np.unique(y[tr], return_counts=True)
                        pred[te[0]] = vals[np.argmax(cnts)]
                _uq_v = np.unique(y)
                acc = float(np.mean([(pred[y == _g] == _g).mean()
                                     for _g in _uq_v]))
            except Exception:
                acc = chance
            v_cache[key] = acc
            return acc

        N = len(selected_cids)
        phi = {c: 0.0 for c in selected_cids}
        # raw_marginals[c][s] accumulates v(S∪{c})−v(S) for all |S|=s
        raw_marginals = {c: {s: [] for s in range(N)} for c in selected_cids}

        try:
            if N <= 8:
                fac_N = math.factorial(N)
                for c in selected_cids:
                    others = [x for x in selected_cids if x != c]
                    for k in range(N):
                        w = (math.factorial(k) * math.factorial(N - k - 1)
                             / fac_N)
                        for S in combinations(others, k):
                            fs = frozenset(S)
                            marginal = _v(fs | {c}) - _v(fs)
                            phi[c] += w * marginal
                            raw_marginals[c][k].append(marginal)
            else:
                # Monte-Carlo Shapley: pre-generate all permutations, batch-compute
                # every distinct prefix coalition in parallel, then sum phi values
                # from cache (O(1) per marginal contribution).
                n_mc = 150
                _mc_rng = _random.Random(42)
                _all_perms = [_mc_rng.sample(selected_cids, N)
                              for _ in range(n_mc)]

                # Collect all distinct prefix coalitions across all permutations
                _to_compute = [fs for fs in {
                    frozenset(_p[:_i])
                    for _p in _all_perms
                    for _i in range(N + 1)
                } if fs and fs not in v_cache]

                if _to_compute:
                    if progress_fn:
                        progress_fn(
                            f"Shapley: evaluating {len(_to_compute)} "
                            f"coalitions in parallel…")
                    from joblib import Parallel as _Par, delayed as _del
                    _pipe_ft_s = ("transition" if metric == "transition"
                                  else "freq_dur")

                    def _eval_coalition(key):
                        from sklearn.model_selection import LeaveOneOut as _LOO
                        from sklearn.base import clone as _cl
                        import numpy as _np
                        _loo = _LOO()
                        try:
                            if metric == "transition":
                                _X, _yl, _ = self._build_transition(
                                    animal_data, cluster_ids=key)
                            elif sel_groups:
                                _ec_cids = [c for c in key if isinstance(c, int)]
                                _ec_gdict = {n: sel_groups[n] for n in key
                                             if isinstance(n, str) and n in sel_groups}
                                if _ec_cids and _ec_gdict:
                                    _X, _yl, _ = GroupPredictorPanel._build_custom(
                                        animal_data, _ec_cids, _ec_gdict, metric)
                                elif _ec_cids:
                                    _X, _yl, _ = GroupPredictorPanel._build_freq_or_dur(
                                        animal_data, metric,
                                        cluster_ids=frozenset(_ec_cids))
                                elif _ec_gdict:
                                    _X, _yl, _ = GroupPredictorPanel._build_group_feat(
                                        animal_data, _ec_gdict, metric)
                                else:
                                    return key, chance
                            else:
                                _X, _yl, _ = self._build_freq_or_dur(
                                    animal_data, metric, cluster_ids=key)
                            _y = _np.array(_yl)
                            if _X.shape[1] == 0 or len(_np.unique(_y)) < 2:
                                return key, chance
                            _pipe = self._make_pipeline(
                                min_class_count, len(_y), _X.shape[1],
                                _pipe_ft_s, greedy_mode=True, algo=_algo_snap)
                            _pred = _np.empty(len(_y), dtype=_y.dtype)
                            for _tr, _te in _loo.split(_X):
                                _p = _cl(_pipe)
                                try:
                                    _p.fit(_X[_tr], _y[_tr])
                                    _pred[_te[0]] = _p.predict(_X[_te])[0]
                                except Exception:
                                    _vs, _cs = _np.unique(
                                        _y[_tr], return_counts=True)
                                    _pred[_te[0]] = _vs[_np.argmax(_cs)]
                            _uq_ec = _np.unique(_y)
                            return key, float(_np.mean(
                                [(_pred[_y == _g] == _g).mean()
                                 for _g in _uq_ec]))
                        except Exception:
                            return key, chance

                    v_cache.update(_Par(n_jobs=_gp_n_jobs(), prefer="threads")(
                        _del(_eval_coalition)(fs) for fs in _to_compute))

                # Phi summation is now pure cache lookups
                for _mc_i, _perm in enumerate(_all_perms):
                    if progress_fn and (_mc_i % 15 == 0 or _mc_i == n_mc - 1):
                        progress_fn(
                            f"Shapley MC summation {_mc_i + 1}/{n_mc}…")
                    for _i, _c in enumerate(_perm):
                        _marginal = (_v(frozenset(_perm[:_i + 1]))
                                     - _v(frozenset(_perm[:_i])))
                        phi[_c] += _marginal
                        raw_marginals[_c][_i].append(_marginal)
                for c in selected_cids:
                    phi[c] /= n_mc

            # Average marginal contribution per coalition size
            phi_by_size = {
                c: {s: (sum(v) / len(v)) if v else 0.0
                    for s, v in raw_marginals[c].items()}
                for c in selected_cids
            }
            baseline = _v(frozenset(selected_cids))
            ranked = sorted([(c, phi[c]) for c in selected_cids],
                            key=lambda x: x[1], reverse=True)
            return baseline, ranked, phi_by_size, dict(v_cache)
        except Exception:
            return None

    def _compute_shapley_transitions(self, animal_data: list,
                                     selected_pairs: list,
                                     min_class_count: int, n_classes: int,
                                     progress_fn=None, algo=None):
        """
        Shapley values for directed transition pairs.
        Returns (baseline_acc, [((i,j), phi), …]) sorted desc by phi, or None.
        Exact (all 2^N subsets) for N ≤ 8; Monte Carlo (150 perms) otherwise.
        """
        import math
        from itertools import combinations
        import random as _random
        from sklearn.model_selection import LeaveOneOut
        from sklearn.base import clone as _clone

        _algo_snap = algo if algo is not None else self._algo_var.get()

        chance = 1.0 / max(n_classes, 2)
        loo = LeaveOneOut()
        v_cache: dict = {}

        def _v(key: frozenset) -> float:
            if key in v_cache:
                return v_cache[key]
            if not key:
                v_cache[key] = chance
                return chance
            try:
                pairs_list = sorted(key)
                X, y_list, _ = self._build_transition_pairs(
                    animal_data, pairs_list)
                y = np.array(y_list)
                if X.shape[1] == 0 or len(np.unique(y)) < 2:
                    v_cache[key] = chance
                    return chance
                pipe = self._make_pipeline(
                    min_class_count, len(y), X.shape[1], "transition",
                    greedy_mode=True, algo=_algo_snap)
                pred = np.empty(len(y), dtype=y.dtype)
                for tr, te in loo.split(X):
                    p = _clone(pipe)
                    try:
                        p.fit(X[tr], y[tr])
                        pred[te[0]] = p.predict(X[te])[0]
                    except Exception:
                        vals, cnts = np.unique(y[tr], return_counts=True)
                        pred[te[0]] = vals[np.argmax(cnts)]
                _uq_vt = np.unique(y)
                acc = float(np.mean([(pred[y == _g] == _g).mean()
                                     for _g in _uq_vt]))
            except Exception:
                acc = chance
            v_cache[key] = acc
            return acc

        N = len(selected_pairs)
        phi = {p: 0.0 for p in selected_pairs}
        raw_marginals = {p: {s: [] for s in range(N)} for p in selected_pairs}

        try:
            if N <= 8:
                fac_N = math.factorial(N)
                for pair in selected_pairs:
                    others = [x for x in selected_pairs if x != pair]
                    for k in range(N):
                        w = (math.factorial(k) * math.factorial(N - k - 1)
                             / fac_N)
                        for S in combinations(others, k):
                            fs = frozenset(S)
                            marginal = _v(fs | {pair}) - _v(fs)
                            phi[pair] += w * marginal
                            raw_marginals[pair][k].append(marginal)
            else:
                # MC Shapley: batch all distinct prefix coalitions, evaluate
                # in parallel, then sum phi values from cache lookups.
                n_mc = 150
                _mc_rng = _random.Random(42)
                _all_perms = [_mc_rng.sample(selected_pairs, N)
                              for _ in range(n_mc)]

                _to_compute = [fs for fs in {
                    frozenset(_p[:_i])
                    for _p in _all_perms
                    for _i in range(N + 1)
                } if fs and fs not in v_cache]

                if _to_compute:
                    if progress_fn:
                        progress_fn(
                            f"Shapley transitions: evaluating "
                            f"{len(_to_compute)} coalitions in parallel…")
                    from joblib import Parallel as _Par, delayed as _del

                    def _eval_pair_coalition(key):
                        from sklearn.model_selection import LeaveOneOut as _LOO
                        from sklearn.base import clone as _cl
                        import numpy as _np
                        _loo = _LOO()
                        try:
                            _pairs = sorted(key)
                            _X, _yl, _ = self._build_transition_pairs(
                                animal_data, _pairs)
                            _y = _np.array(_yl)
                            if _X.shape[1] == 0 or len(_np.unique(_y)) < 2:
                                return key, chance
                            _pipe = self._make_pipeline(
                                min_class_count, len(_y), _X.shape[1],
                                "transition", greedy_mode=True,
                                algo=_algo_snap)
                            _pred = _np.empty(len(_y), dtype=_y.dtype)
                            for _tr, _te in _loo.split(_X):
                                _p = _cl(_pipe)
                                try:
                                    _p.fit(_X[_tr], _y[_tr])
                                    _pred[_te[0]] = _p.predict(_X[_te])[0]
                                except Exception:
                                    _vs, _cs = _np.unique(
                                        _y[_tr], return_counts=True)
                                    _pred[_te[0]] = _vs[_np.argmax(_cs)]
                            _uq_pc = _np.unique(_y)
                            return key, float(_np.mean(
                                [(_pred[_y == _g] == _g).mean()
                                 for _g in _uq_pc]))
                        except Exception:
                            return key, chance

                    v_cache.update(_Par(n_jobs=_gp_n_jobs(), prefer="threads")(
                        _del(_eval_pair_coalition)(fs) for fs in _to_compute))

                for _mc_i, _perm in enumerate(_all_perms):
                    if progress_fn and (_mc_i % 15 == 0 or _mc_i == n_mc - 1):
                        progress_fn(
                            f"Shapley MC summation {_mc_i + 1}/{n_mc}…")
                    for _i, _pair in enumerate(_perm):
                        _marginal = (_v(frozenset(_perm[:_i + 1]))
                                     - _v(frozenset(_perm[:_i])))
                        phi[_pair] += _marginal
                        raw_marginals[_pair][_i].append(_marginal)
                for pair in selected_pairs:
                    phi[pair] /= n_mc

            phi_by_size = {
                p: {s: (sum(v) / len(v)) if v else 0.0
                    for s, v in raw_marginals[p].items()}
                for p in selected_pairs
            }
            baseline = _v(frozenset(selected_pairs))
            ranked = sorted([(p, phi[p]) for p in selected_pairs],
                            key=lambda x: x[1], reverse=True)
            return baseline, ranked, phi_by_size, dict(v_cache)
        except Exception:
            return None

    def _greedy_group_transition_selection(self, animal_data: list, groups: dict,
                                            min_class_count: int,
                                            max_steps: int = 20,
                                            progress_fn=None,
                                            algo=None) -> tuple:
        """
        Forward greedy selection at the group-pair level.
        Builds the full group-pair probability matrix once, then column-slices
        for each candidate at each step — same pattern as _greedy_transition_selection.
        Returns (trace, ties_by_step) where trace items are (feature_name, bal_acc).
        """
        from sklearn.model_selection import LeaveOneOut
        from sklearn.base import clone as _clone

        X_full, y_list, feat_names = self._build_group_transitions(
            animal_data, groups)
        y = np.array(y_list)
        if X_full.shape[1] == 0 or len(np.unique(y)) < 2:
            return [], {}

        col_map   = {name: i for i, name in enumerate(feat_names)}
        remaining = list(feat_names)
        sel_cols: list = []
        trace: list   = []
        _ties_by_step: dict = {}
        loo  = LeaveOneOut()
        step = 0

        while remaining and step < max_steps:
            if progress_fn:
                progress_fn(
                    f"Grp-transition greedy step {step + 1} "
                    f"({len(remaining)} candidates left)")
            valid: list = []
            for cand_name in remaining:
                _cols = sel_cols + [col_map[cand_name]]
                X_c   = X_full[:, _cols]
                if len(np.unique(y)) < 2:
                    continue
                try:
                    _pipe = self._make_pipeline(
                        min_class_count, len(y), X_c.shape[1],
                        "transition", greedy_mode=True, algo=algo)
                    _preds = np.empty(len(y), dtype=y.dtype)
                    for _tr, _te in loo.split(X_c):
                        _p = _clone(_pipe)
                        try:
                            _p.fit(X_c[_tr], y[_tr])
                            _preds[_te[0]] = _p.predict(X_c[_te])[0]
                        except Exception:
                            _vals, _cnts = np.unique(y[_tr], return_counts=True)
                            _preds[_te[0]] = _vals[np.argmax(_cnts)]
                    _uq  = np.unique(y)
                    _bal = float(np.mean(
                        [(_preds[y == _g] == _g).mean() for _g in _uq]))
                    valid.append((_bal, cand_name))
                except Exception:
                    continue

            if not valid:
                break
            _best_acc = max(v[0] for v in valid)
            _tied = [(acc, n) for acc, n in valid
                     if abs(acc - _best_acc) < 1e-9]
            _ties_by_step[step] = _tied
            best_name = _tied[0][1]
            if trace and _best_acc < trace[-1][1] - 1e-9:
                break
            sel_cols.append(col_map[best_name])
            remaining.remove(best_name)
            trace.append((best_name, _best_acc))
            step += 1

        return trace, _ties_by_step

    def _best_k_group_pair_subsets(self, animal_data: list, groups: dict,
                                    min_class_count: int,
                                    k: int, top_n: int = 5,
                                    progress_fn=None) -> tuple:
        """
        Exhaustive LOO search over all C(n_group_pairs, k) group-pair subsets.
        Returns (trace, top_combos); falls back to ([], []) on failure.
        Typically feasible because n_groups is small (2–8).
        """
        import math
        from itertools import combinations
        from sklearn.model_selection import LeaveOneOut
        from sklearn.base import clone as _clone

        X_full, y_list, feat_names = self._build_group_transitions(
            animal_data, groups)
        y = np.array(y_list)
        n_cols = X_full.shape[1]
        if n_cols == 0 or len(np.unique(y)) < 2 or k > n_cols:
            return [], []

        n_combos  = math.comb(n_cols, k)
        loo        = LeaveOneOut()
        fold_splits = list(loo.split(X_full))
        results: list = []

        for combo_i, combo in enumerate(combinations(range(n_cols), k)):
            if progress_fn and (combo_i % max(1, n_combos // 20) == 0):
                progress_fn(f"Exhaustive: {combo_i}/{n_combos}")
            X_c = X_full[:, list(combo)]
            try:
                _pipe = self._make_pipeline(
                    min_class_count, len(y), X_c.shape[1],
                    "transition", greedy_mode=True)
                _preds = np.empty(len(y), dtype=y.dtype)
                for _tr, _te in fold_splits:
                    _p = _clone(_pipe)
                    try:
                        _p.fit(X_c[_tr], y[_tr])
                        _preds[_te[0]] = _p.predict(X_c[_te])[0]
                    except Exception:
                        _vals, _cnts = np.unique(y[_tr], return_counts=True)
                        _preds[_te[0]] = _vals[np.argmax(_cnts)]
                _uq  = np.unique(y)
                _bal = float(np.mean(
                    [(_preds[y == _g] == _g).mean() for _g in _uq]))
                _names = tuple(feat_names[c] for c in combo)
                results.append((_bal, _names))
            except Exception:
                continue

        if not results:
            return [], []
        results.sort(key=lambda x: x[0], reverse=True)
        top_combos = [(_names, _bal) for _bal, _names in results[:top_n]]
        best_bal, best_combo = results[0]
        trace = [(_name, best_bal) for _name in sorted(best_combo)]
        return trace, top_combos

    def _compute_shapley_group_transitions(self, animal_data: list, groups: dict,
                                            selected_pair_names: list,
                                            min_class_count: int, n_classes: int,
                                            progress_fn=None, algo=None):
        """
        Shapley importance values for group-pair features.
        Mirrors _compute_shapley_transitions: exact for N≤8, Monte Carlo otherwise.
        Returns (baseline_acc, [(name, phi), ...], phi_by_size, v_cache) or None.
        """
        import math
        from itertools import combinations
        import random as _random
        from sklearn.model_selection import LeaveOneOut
        from sklearn.base import clone as _clone

        X_full, y_list, feat_names = self._build_group_transitions(
            animal_data, groups)
        y = np.array(y_list)
        col_map = {name: i for i, name in enumerate(feat_names)}
        N = len(selected_pair_names)
        if N == 0 or X_full.shape[1] == 0 or len(np.unique(y)) < 2:
            return None

        _algo_snap = algo if algo is not None else self._algo_var.get()
        chance = 1.0 / max(n_classes, 2)
        loo     = LeaveOneOut()
        v_cache: dict = {}

        def _v(key: frozenset) -> float:
            if key in v_cache:
                return v_cache[key]
            if not key:
                v_cache[key] = chance
                return chance
            try:
                _cols = [col_map[n] for n in key if n in col_map]
                if not _cols:
                    v_cache[key] = chance
                    return chance
                X_c = X_full[:, _cols]
                if X_c.shape[1] == 0 or len(np.unique(y)) < 2:
                    v_cache[key] = chance
                    return chance
                _pipe  = self._make_pipeline(
                    min_class_count, len(y), X_c.shape[1],
                    "transition", greedy_mode=True, algo=_algo_snap)
                _preds = np.empty(len(y), dtype=y.dtype)
                for _tr, _te in loo.split(X_c):
                    _p = _clone(_pipe)
                    try:
                        _p.fit(X_c[_tr], y[_tr])
                        _preds[_te[0]] = _p.predict(X_c[_te])[0]
                    except Exception:
                        _vals, _cnts = np.unique(y[_tr], return_counts=True)
                        _preds[_te[0]] = _vals[np.argmax(_cnts)]
                _uq  = np.unique(y)
                _acc = float(np.mean(
                    [(_preds[y == _g] == _g).mean() for _g in _uq]))
            except Exception:
                _acc = chance
            v_cache[key] = _acc
            return _acc

        try:
            phi: dict          = {n: 0.0 for n in selected_pair_names}
            raw_marginals: dict = {n: {} for n in selected_pair_names}

            if N <= 8:
                for _size in range(N + 1):
                    for _sub in combinations(selected_pair_names, _size):
                        _v(frozenset(_sub))
                for _name in selected_pair_names:
                    _others = [n for n in selected_pair_names if n != _name]
                    for _size in range(N):
                        for _pre in combinations(_others, _size):
                            _pre_s  = frozenset(_pre)
                            _with_s = _pre_s | {_name}
                            _marg   = _v(_with_s) - _v(_pre_s)
                            _w = (math.factorial(_size)
                                  * math.factorial(N - _size - 1)
                                  / math.factorial(N))
                            phi[_name] += _w * _marg
                            raw_marginals[_name].setdefault(_size, []).append(_marg)
            else:
                n_mc  = 150
                _rng  = _random.Random(42)
                _perms = [_rng.sample(selected_pair_names, N) for _ in range(n_mc)]
                for _perm in _perms:
                    for _i in range(N):
                        _pre_s  = frozenset(_perm[:_i])
                        _with_s = _pre_s | {_perm[_i]}
                        _marg   = _v(_with_s) - _v(_pre_s)
                        phi[_perm[_i]] += _marg
                        raw_marginals[_perm[_i]].setdefault(_i, []).append(_marg)
                for _name in selected_pair_names:
                    phi[_name] /= n_mc

            phi_by_size = {
                n: {s: (sum(v) / len(v)) if v else 0.0
                    for s, v in raw_marginals[n].items()}
                for n in selected_pair_names
            }
            baseline = _v(frozenset(selected_pair_names))
            ranked   = sorted([(n, phi[n]) for n in selected_pair_names],
                               key=lambda x: x[1], reverse=True)
            return baseline, ranked, phi_by_size, dict(v_cache)
        except Exception:
            return None

    @staticmethod
    def _style_axes(ax):
        """Apply theme colours to a matplotlib Axes in one call."""
        ax.set_facecolor(T()["ax_bg"])
        ax.tick_params(colors=T()["tick"])
        for spine in ax.spines.values():
            spine.set_edgecolor(T()["spine"])

    def save_all_figures(self, out_dir: "pathlib.Path", ts: str) -> int:
        """Save overview + detail figures for ALL models. Returns count saved."""
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0

        if self._overview_fig is not None:
            for ext in ("png", "pdf"):
                self._overview_fig.savefig(
                    str(out_dir / f"predictor_overview_{ts}.{ext}"),
                    dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor=self._overview_fig.get_facecolor())
            n += 1

        if self._null_comp_fig is not None:
            for ext in ("png", "pdf"):
                self._null_comp_fig.savefig(
                    str(out_dir / f"predictor_null_comparison_{ts}.{ext}"),
                    dpi=300 if ext == "png" else None,
                    bbox_inches="tight",
                    facecolor=self._null_comp_fig.get_facecolor())
            n += 1

        # Save detail figures for every model, not just the currently selected one.
        # _draw_detail re-renders the detail panel and repopulates self._open_figs
        # for each model in turn; we save after each call before the next call
        # closes the previous model's figures.
        orig_selected = self._selected_model
        for res in self._results:
            if res.get("error"):
                continue
            self._draw_detail(res)
            model_slug = res["name"].replace(" ", "_").replace(".", "").lower()
            for fi, fig in enumerate(self._open_figs):
                if fig is None:
                    continue
                for ext in ("png", "pdf"):
                    fig.savefig(
                        str(out_dir / f"predictor_{model_slug}_detail{fi + 1}_{ts}.{ext}"),
                        dpi=300 if ext == "png" else None,
                        bbox_inches="tight", facecolor=fig.get_facecolor())
                n += 1

        # Restore the originally selected model
        if self._results:
            self._draw_detail(
                self._results[min(orig_selected, len(self._results) - 1)])

        return n

    # ── Run entry point ───────────────────────────────────────────────────────

    def _cancel(self):
        """Signal the background worker to stop after the current step."""
        self._cancel_event.set()
        self._status("Cancelling after current step…")
        if self._cancel_btn:
            self._cancel_btn.configure(state="disabled")

    def _run(self):
        if self._running:
            return
        if getattr(self, "_nested_running", False):
            self._status(
                "Nested permutation test is running — wait for it to finish first.",
                color=T()["error"])
            return

        from sklearn.preprocessing import LabelEncoder

        # Map "Predict by" radio to the group_by key expected by get_animals().
        # "Combined" uses all label columns joined with " | "; individual factors
        # use their label column directly — no compound-splitting needed.
        factor = self._factor_var.get()
        _gb_key = {"Combined": "all", "Factor1": "label1",
                   "Factor2": "label2", "Factor3": "label3"}.get(factor, "label1")
        animals = self._get_animals(group_by=_gb_key)
        if not animals:
            self._status("No animals loaded. Use Combined Analysis tab first.",
                         color=T()["error"])
            return

        # Drop animals without a valid experimental group assignment
        animals = [a for a in animals
                   if a.get("exp_group") and str(a["exp_group"]).strip()]
        if not animals:
            self._status("No animals have an experimental group assigned.",
                         color=T()["error"])
            return

        # Validate: need ≥2 distinct experimental groups
        exp_groups = sorted({a["exp_group"] for a in animals})
        if len(exp_groups) < 2:
            self._status(
                "Assign at least 2 different experimental groups to run classification.",
                color=T()["error"])
            self._show_caveats(
                ["All animals share the same exp_group — cannot run classification."])
            return

        # Encode labels (LabelEncoder sorts internally; explicit sort above keeps
        # exp_groups consistent with what le.classes_ will be)
        le = LabelEncoder()
        le.fit(exp_groups)

        # Get selected behavior groups
        src = self._source_var.get()
        all_groups = self._get_groups_fn() or {}
        if src in ("groups", "mix", "custom"):
            selected_gnames = [gn for gn, var in self._group_vars.items()
                               if var.get()]
            if not selected_gnames and src == "groups":
                self._status("Select at least one behavior group.", color=T()["error"])
                return
            sel_groups = {k: v for k, v in all_groups.items()
                          if k in selected_gnames}
        else:
            sel_groups = all_groups

        # Build three feature matrices (fast — on main thread)
        self._status("Building feature matrices…")
        self.update_idletasks()
        _custom_cls = None   # set only for src == "custom"; drives greedy selection
        try:
            if src == "clusters":
                Xf, yf, ff = self._build_freq_or_dur(animals, "frequency")
                Xd, yd, fd = self._build_freq_or_dur(animals, "total_duration")
            elif src == "groups":
                Xf, yf, ff = self._build_group_feat(animals, sel_groups, "frequency")
                Xd, yd, fd = self._build_group_feat(animals, sel_groups, "total_duration")
            elif src == "custom":
                sel_clusters = [c for c, var in self._cluster_vars.items()
                                if var.get()]
                if not sel_clusters:
                    self._status("Select at least one cluster.", color=T()["error"])
                    return
                _custom_cls = sel_clusters
                Xf, yf, ff = self._build_custom(animals, sel_clusters, sel_groups, "frequency")
                Xd, yd, fd = self._build_custom(animals, sel_clusters, sel_groups, "total_duration")
            else:   # mix
                Xf, yf, ff = self._build_mix(animals, sel_groups, "frequency")
                Xd, yd, fd = self._build_mix(animals, sel_groups, "total_duration")
            if src in ("groups", "mix"):
                Xt, yt, ft = self._build_group_transitions(
                    animals, sel_groups,
                    include_cluster_pairs=(src == "mix"))
            else:
                Xt, yt, ft = self._build_transition(animals)
        except Exception as exc:
            self._status(f"Feature build error: {exc}", color=T()["error"])
            return

        # Max-contributors cap — passed into the pipeline (SelectKBest inside LOO)
        max_contrib_str = self._max_contrib_var.get()
        _k = int(max_contrib_str) if max_contrib_str != "All" else None

        # Snapshot run config for display in _finish
        self._last_source       = src
        self._last_max_contrib  = max_contrib_str
        self._last_model_name   = (self._model_name_var.get().strip()
                                   if src in ("mix", "custom") else "")

        # Encode y
        yf_enc = le.transform(yf).astype(int)
        yd_enc = le.transform(yd).astype(int)
        yt_enc = le.transform(yt).astype(int)

        # Count per-class samples for inner CV sizing
        from collections import Counter
        min_class_count = min(Counter(yf_enc).values())
        n_total = len(animals)
        n_feats_max = max(Xf.shape[1], Xd.shape[1], Xt.shape[1])

        # Collect caveats
        caveats = self._collect_caveats(
            animals, exp_groups, min_class_count, n_total, n_feats_max)

        n_perm = int(self._nperm_var.get())
        # Snapshot animal names now so export stays aligned even if list changes later
        animal_names = [a["name"] for a in animals]
        # Each model gets its own n_features + feature_type so _make_pipeline
        # can size PCA and interaction expansion correctly per model.
        # 7th element: cluster_ids for greedy forward selection (Custom mode only)
        # 8th element: k for SelectKBest inside the pipeline (None = keep all)
        # 9th element: sel_groups dict for custom/mix so Path A can include groups in LOO
        _model_sel_groups = sel_groups if src in ("custom", "mix") else {}
        # Transition model uses group-level features when source is groups/mix
        _t_feature_type = ("group_transition" if src in ("groups", "mix")
                           else "transition")
        _t_sel_groups   = sel_groups if src in ("groups", "mix") else {}
        model_defs = [
            ("Frequency",        Xf, yf_enc, ff, Xf.shape[1], "freq_dur",      _custom_cls, _k, _model_sel_groups),
            ("Total Duration",   Xd, yd_enc, fd, Xd.shape[1], "freq_dur",      _custom_cls, _k, _model_sel_groups),
            ("Transition Prob.", Xt, yt_enc, ft, Xt.shape[1], _t_feature_type, None,        _k, _t_sel_groups),
        ]
        chance = 1.0 / len(exp_groups)

        # Capture algo string on main thread before spawning the worker.
        # ctk.StringVar.get() calls into the Tcl interpreter; doing that from
        # a background thread on Windows is unsafe and can terminate the process.
        _algo = self._algo_var.get()

        # Disable Run + nested-test button, enable Cancel, clear any previous cancel signal
        self._cancel_event.clear()
        self._running = True
        if self._run_btn:
            self._run_btn.configure(state="disabled", text="Running…")
        if self._cancel_btn:
            self._cancel_btn.configure(state="normal")
        if self._nested_btn:
            self._nested_btn.configure(state="disabled")
        self._status("Starting… (GUI stays responsive)")
        self._show_loading_overlay()
        # _show_loading_overlay calls gc.collect() internally, but that runs
        # while its own local variables (_fig, fig, w) still hold references
        # to the figures being cleaned up.  Those refs drop when the function
        # returns, making the canvas / toolbar / PhotoImage / StringVar cycles
        # newly unreachable.  A second collect here, after the locals are gone,
        # ensures tk finalizers run on the main thread rather than inside a
        # joblib worker that triggers automatic GC later.
        import gc as _gc_pre
        _gc_pre.collect()

        # Clear queue from any previous run
        while not self._progress_q.empty():
            try:
                self._progress_q.get_nowait()
            except Exception:
                break

        import threading
        t = threading.Thread(
            target=self._run_worker,
            args=(min_class_count, model_defs, le, exp_groups, n_total,
                  caveats, chance, n_perm, animal_names, animals, _algo),
            daemon=True,
        )
        self._bg_thread = t
        t.start()
        self.after(150, self._poll_progress)

    def _run_worker(self, min_class_count, model_defs, le, exp_groups,
                    n_total, caveats, chance, n_perm, animal_names,
                    animal_data=None, algo=None):
        """Background thread: fits all three models and pushes results to queue."""
        # Kill any loky pool left over from the previous run before spawning new
        # workers.  The end-of-run shutdown kills workers but leaves the loky
        # resource-tracker process alive; if the user re-runs immediately that
        # orphan process + new workers all import MKL simultaneously, which
        # triggers a Windows kernel crash.  Shutting down here guarantees the
        # old resource tracker is gone before the first new Parallel() call.
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True, kill_workers=True)
        except Exception:
            pass
        # OPENBLAS_NUM_THREADS / OMP_NUM_THREADS are set at module level before
        # numpy is imported, so they are always effective.
        from sklearn.metrics import cohen_kappa_score, confusion_matrix
        from sklearn.base import clone as _clone

        results = []
        for i, (mname, X, y_enc, feat_names, n_features, feature_type,
                custom_cls, k_contrib, model_sel_groups) in enumerate(model_defs):
            if self._cancel_event.is_set():
                self._progress_q.put(("status", "Run cancelled by user."))
                break
            _n_models   = len(model_defs)
            _base_p     = i / _n_models
            _model_slice = 1.0 / _n_models
            self._progress_q.put(("progress", _base_p))
            self._progress_q.put(("status",
                f"Model {i+1}/3 — {mname}: starting LOO ({len(y_enc)} folds)…"))

            def progress_fn(msg, _name=mname, _bp=_base_p, _sl=_model_slice):
                self._progress_q.put(("status", f"{_name}: {msg}"))
                _lm = re.match(r"LOO fold (\d+)/(\d+)", msg)
                if _lm:
                    _p = _bp + (0.15 + int(_lm.group(1)) / int(_lm.group(2)) * 0.50) * _sl
                    self._progress_q.put(("progress", _p))
                    return
                _pm = re.match(r"Perm (\d+)/(\d+)", msg)
                if _pm:
                    # 65% → 90% of the model slice as permutation batches complete.
                    # Callers set bar to 0.90 after _run_loo returns; stopping here
                    # prevents a visible step-back when that line executes.
                    _p = _bp + (0.65 + int(_pm.group(1)) / int(_pm.group(2)) * 0.25) * _sl
                    self._progress_q.put(("progress", _p))

            try:
                from sklearn.feature_selection import VarianceThreshold, SelectKBest
                import re as _re_g

                # ── Step 1: Greedy cluster selection ──────────────────────────
                # VT depends only on X, so we can determine surviving candidates
                # with a cheap standalone fit — no full LOO pass needed yet.
                _vt_pre = VarianceThreshold(threshold=1e-10).fit(X)
                _pregreedy_names = [fn for fn, keep
                                    in zip(feat_names, _vt_pre.get_support()) if keep]

                trace          = None
                top_combos     = []
                _greedy_ties   = {}
                _combo_perm_results      = []
                _best_combo_perm_scores  = np.array([], dtype=float)
                _search_method = "greedy"
                _trace_is_custom = (custom_cls is not None)
                _greedy_metric = ("transition"
                                  if feature_type in ("transition", "group_transition")
                                  else ("total_duration" if mname == "Total Duration"
                                        else "frequency"))
                if animal_data is not None:
                    _greedy_cids  = []   # integer cluster IDs surviving VT
                    _greedy_gnames = []  # group names surviving VT
                    if feature_type == "group_transition":
                        # Greedy/exhaustive handled via dedicated methods below;
                        # set a sentinel so the _greedy_units block is entered.
                        _greedy_units = ["__group_transition__"]
                    elif feature_type == "transition":
                        # Transition features are "T_i→j" — collect the union of
                        # all cluster IDs that appear as source or target.
                        _seen_cids: set = set()
                        for _fn in _pregreedy_names:
                            _m = _re_g.match(r"T_(\d+)→(\d+)", _fn)
                            if _m:
                                _seen_cids.add(int(_m.group(1)))
                                _seen_cids.add(int(_m.group(2)))
                        _greedy_cids  = sorted(_seen_cids)
                        _greedy_units = _greedy_cids  # no groups for transitions
                    else:
                        _group_feat_name_set = (set(model_sel_groups.keys())
                                                if model_sel_groups else set())
                        for _fn in _pregreedy_names:
                            _m = _re_g.match(r"Cluster (\d+)$", _fn)
                            if _m:
                                _greedy_cids.append(int(_m.group(1)))
                            elif _fn in _group_feat_name_set:
                                _greedy_gnames.append(_fn)
                        _greedy_units = _greedy_cids + _greedy_gnames

                    if _greedy_units:
                        def _greedy_prog(msg, _mn=mname, _bp=_base_p, _sl=_model_slice):
                            self._progress_q.put(("status", f"{_mn}: {msg}"))
                            _em = re.match(r"Exhaustive: (\d+)/(\d+)", msg)
                            if _em:
                                # 0% → 15% of model slice while exhaustive combos run.
                                _p = _bp + (int(_em.group(1)) / int(_em.group(2)) * 0.15) * _sl
                                self._progress_q.put(("progress", _p))

                        if feature_type == "group_transition":
                            # ── Group-transition: exhaustive or greedy over G_src→G_dst ──
                            import math as _math_gt
                            _n_gp = len([n for n in feat_names
                                         if n.startswith("G_") and "→" in n])
                            _use_exhaustive_gt = False
                            if k_contrib is not None and _n_gp > 0:
                                _n_combos_gt = (_math_gt.comb(_n_gp, k_contrib)
                                                if k_contrib <= _n_gp else 0)
                                _use_exhaustive_gt = (
                                    0 < _n_combos_gt <= self.EXHAUSTIVE_COMBO_LIMIT)
                                if _use_exhaustive_gt:
                                    self._progress_q.put(("status",
                                        f"{mname}: exhaustive group-pair search "
                                        f"C({_n_gp},{k_contrib})={_n_combos_gt} "
                                        f"group-pair combos…"))
                                    try:
                                        trace, top_combos = \
                                            self._best_k_group_pair_subsets(
                                                animal_data, model_sel_groups,
                                                min_class_count,
                                                k=k_contrib, top_n=5,
                                                progress_fn=_greedy_prog)
                                        if trace:
                                            _search_method = "exhaustive"
                                        else:
                                            _use_exhaustive_gt = False
                                    except Exception as _exc:
                                        self._progress_q.put(("status",
                                            f"{mname}: warning — exhaustive group-pair "
                                            f"search failed ({_exc}); falling back."))
                                        _use_exhaustive_gt = False

                            if not _use_exhaustive_gt:
                                _gt_steps = (k_contrib if k_contrib is not None
                                             else max(_n_gp, 1))
                                self._progress_q.put(("status",
                                    f"{mname}: greedy group-pair trace "
                                    f"({_n_gp} group pairs, "
                                    f"max {_gt_steps} steps)…"))
                                try:
                                    trace, _greedy_ties = \
                                        self._greedy_group_transition_selection(
                                            animal_data, model_sel_groups,
                                            min_class_count,
                                            max_steps=_gt_steps,
                                            progress_fn=_greedy_prog,
                                            algo=algo)
                                except Exception as _exc:
                                    self._progress_q.put(("status",
                                        f"{mname}: warning — greedy group-pair "
                                        f"selection failed ({_exc}); skipping trace."))
                                    trace = None
                                    _greedy_ties = {}

                        elif feature_type == "transition":
                            # ── Transition: exhaustive when C(n_pairs,k) ≤ limit ──
                            import math as _math_t
                            _all_t_pairs = [(ci, cj)
                                            for ci in _greedy_cids
                                            for cj in _greedy_cids if ci != cj]

                            # Prune to above-chance pairs only.
                            # Uniform random-walk chance floor: 1 / (n_clusters - 1),
                            # identical to the formula used in all transition plots.
                            # Use the cross-animal column mean as the per-pair summary;
                            # keep only pairs strictly above the theoretical floor.
                            _col_mn = X.mean(axis=0)
                            _pair_col: dict = {}
                            for _ci2, _fn2 in enumerate(feat_names):
                                _pts = _fn2[2:].split("→")
                                if len(_pts) == 2:
                                    try:
                                        _pair_col[(int(_pts[0]), int(_pts[1]))] = _ci2
                                    except ValueError:
                                        pass
                            _chance_floor_greedy = 1.0 / max(1, len(_greedy_cids) - 1)
                            _n_before = len(_all_t_pairs)
                            _all_t_pairs = [
                                _p for _p in _all_t_pairs
                                if (_p in _pair_col
                                    and _col_mn[_pair_col[_p]]
                                    > _chance_floor_greedy)
                            ]
                            _n_pruned = _n_before - len(_all_t_pairs)
                            if _n_pruned:
                                self._progress_q.put(("status",
                                    f"{mname}: pruned {_n_pruned}/{_n_before} "
                                    f"below-chance transitions (floor={_chance_floor_greedy:.4f}) → "
                                    f"{len(_all_t_pairs)} remain."))

                            _n_pairs = len(_all_t_pairs)
                            _use_exhaustive_t = False

                            if k_contrib is not None:
                                _n_combos_t = (_math_t.comb(_n_pairs, k_contrib)
                                               if k_contrib <= _n_pairs else 0)
                                _use_exhaustive_t = (
                                    0 < _n_combos_t <= self.EXHAUSTIVE_COMBO_LIMIT)
                                if _use_exhaustive_t:
                                    self._progress_q.put(("status",
                                        f"{mname}: exhaustive pair search "
                                        f"C({_n_pairs},{k_contrib})={_n_combos_t} "
                                        f"pair-combos…"))
                                    try:
                                        trace, top_combos = \
                                            self._best_k_pair_subsets(
                                                animal_data, _all_t_pairs,
                                                k=k_contrib, top_n=5,
                                                progress_fn=_greedy_prog)
                                        if trace:
                                            _search_method = "exhaustive"
                                        else:
                                            _use_exhaustive_t = False
                                    except Exception as _exc:
                                        self._progress_q.put(("status",
                                            f"{mname}: warning — exhaustive pair "
                                            f"search failed ({_exc}); falling back "
                                            f"to greedy."))
                                        _use_exhaustive_t = False

                            if not _use_exhaustive_t:
                                # Greedy fallback (k not set, or combo count > limit)
                                _greedy_steps = (k_contrib if k_contrib is not None
                                                 else _n_pairs)
                                self._progress_q.put(("status",
                                    f"{mname}: greedy transition trace "
                                    f"({len(_greedy_cids)} clusters → {_n_pairs} "
                                    f"directed pairs, max {_greedy_steps} steps)…"))
                                try:
                                    trace, _greedy_ties = \
                                        self._greedy_transition_selection(
                                            animal_data, set(_greedy_cids),
                                            min_class_count,
                                            max_steps=_greedy_steps,
                                            progress_fn=_greedy_prog,
                                            algo=algo,
                                            allowed_pairs=_all_t_pairs)
                                except Exception as _exc:
                                    self._progress_q.put(("status",
                                        f"{mname}: warning — greedy transition "
                                        f"selection failed ({_exc}); skipping trace."))
                                    trace = None
                                    _greedy_ties = {}

                        elif _greedy_gnames:
                            # ── Mixed mode: clusters + groups as co-equal units ───
                            # Exhaustive is not attempted when groups are present.
                            _greedy_steps = (k_contrib if k_contrib is not None
                                             else len(_greedy_units))
                            self._progress_q.put(("status",
                                f"{mname}: greedy mixed selection "
                                f"({len(_greedy_cids)} cluster(s) + "
                                f"{len(_greedy_gnames)} group(s), "
                                f"max {_greedy_steps} steps)…"))
                            try:
                                trace, _greedy_ties = self._greedy_mixed_selection(
                                    animal_data, _greedy_units, model_sel_groups,
                                    min_class_count,
                                    max_steps=_greedy_steps,
                                    progress_fn=_greedy_prog,
                                    metric=_greedy_metric,
                                    algo=algo)
                            except Exception as _exc:
                                self._progress_q.put(("status",
                                    f"{mname}: warning — greedy mixed "
                                    f"selection failed ({_exc}); skipping trace."))
                                trace = None
                                _greedy_ties = {}

                        else:
                            # ── Freq/dur (cluster-only): exhaustive when C(n,k) ≤ limit
                            import math as _math
                            _n_cids = len(_greedy_cids)
                            _use_exhaustive = False

                            if k_contrib is not None:
                                _n_combos = (_math.comb(_n_cids, k_contrib)
                                             if k_contrib <= _n_cids else 0)
                                _use_exhaustive = (0 < _n_combos
                                                   <= self.EXHAUSTIVE_COMBO_LIMIT)
                                if _use_exhaustive:
                                    self._progress_q.put(("status",
                                        f"{mname}: exhaustive search "
                                        f"C({_n_cids},{k_contrib})={_n_combos} "
                                        f"combos…"))
                                    try:
                                        trace, top_combos = self._best_k_subsets(
                                            animal_data, _greedy_cids,
                                            k=k_contrib, top_n=5,
                                            progress_fn=_greedy_prog,
                                            metric=_greedy_metric)
                                        if trace:
                                            _search_method = "exhaustive"
                                        else:
                                            _use_exhaustive = False
                                    except Exception as _exc:
                                        self._progress_q.put(("status",
                                            f"{mname}: warning — exhaustive search "
                                            f"failed ({_exc}); falling back to greedy."))
                                        _use_exhaustive = False

                            if not _use_exhaustive:
                                # Greedy path (k too large, or exhaustive failed)
                                _greedy_steps = (k_contrib if k_contrib is not None
                                                 else _n_cids)
                                self._progress_q.put(("status",
                                    f"{mname}: greedy cluster selection "
                                    f"({_n_cids} clusters, "
                                    f"max {_greedy_steps} steps)…"))
                                try:
                                    trace, _greedy_ties = self._greedy_cluster_selection(
                                        animal_data, _greedy_cids, min_class_count,
                                        max_steps=_greedy_steps,
                                        progress_fn=_greedy_prog,
                                        metric=_greedy_metric,
                                        algo=algo)
                                except Exception as _exc:
                                    self._progress_q.put(("status",
                                        f"{mname}: warning — greedy cluster "
                                        f"selection failed ({_exc}); skipping trace."))
                                    trace = None
                                    _greedy_ties = {}

                                # "All" mode upgrade: if greedy peak_k is within
                                # the exhaustive limit, refine with exact search.
                                if trace and k_contrib is None:
                                    _peak_acc_g = max(a for _, a in trace)
                                    _cut_g = next(
                                        i for i, (_, a) in enumerate(trace)
                                        if a >= _peak_acc_g - 1e-9)
                                    _peak_k = _cut_g + 1
                                    _n_at_peak = (_math.comb(_n_cids, _peak_k)
                                                  if _peak_k <= _n_cids else 0)
                                    if 0 < _n_at_peak <= self.EXHAUSTIVE_COMBO_LIMIT:
                                        self._progress_q.put(("status",
                                            f"{mname}: greedy peak k={_peak_k}, "
                                            f"C({_n_cids},{_peak_k})={_n_at_peak} "
                                            f"→ upgrading to exhaustive…"))
                                        try:
                                            _ex_tr, _ex_top = self._best_k_subsets(
                                                animal_data, _greedy_cids,
                                                k=_peak_k, top_n=5,
                                                progress_fn=_greedy_prog,
                                                metric=_greedy_metric)
                                            if _ex_tr:
                                                trace          = _ex_tr
                                                top_combos     = _ex_top
                                                _search_method = "exhaustive"
                                                _greedy_ties   = {}  # exhaustive supersedes greedy ties
                                        except Exception as _exc:
                                            self._progress_q.put(("status",
                                                f"{mname}: warning — exhaustive "
                                                f"upgrade failed ({_exc}); "
                                                f"keeping greedy result."))
                                            # keep greedy result

                # ── Step 1b: Exhaustive — permutation-test all tied top combos ───
                # When exhaustive search produced multiple combos with identical LOO
                # accuracy, re-rank them by conditional permutation p-value so the
                # most group-specific combination rises to #1 and becomes the primary
                # trace used by Path A and the overview plots.
                if (_search_method == "exhaustive" and top_combos
                        and animal_data is not None and n_perm > 0
                        and not self._cancel_event.is_set()):
                    _combo_perm_results    = []
                    _combo_perm_scores_map = {}   # frozenset(combo) → perm_scores
                    for _ci_ex, (_combo_ex, _combo_acc_ex) in enumerate(top_combos):
                        if self._cancel_event.is_set():
                            break
                        # Label units as T_i→j for pairs or C{n} for clusters
                        _unit_lbl = (lambda c: f"T_{c[0]}→{c[1]}"
                                     if isinstance(c, tuple) else f"C{c}")
                        self._progress_q.put(("status",
                            f"{mname}: perm-testing combo "
                            f"{_ci_ex + 1}/{len(top_combos)} "
                            f"({{{','.join(_unit_lbl(c) for c in sorted(_combo_ex, key=str))}}})…"))
                        try:
                            if _greedy_metric == "transition":
                                _X_ce, _y_ce_raw, _ = self._build_transition_pairs(
                                    animal_data, list(_combo_ex))
                            else:
                                _X_ce, _y_ce_raw, _ = self._build_freq_or_dur(
                                    animal_data, _greedy_metric,
                                    cluster_ids=set(_combo_ex))
                            _y_ce_enc = le.transform(_y_ce_raw).astype(int)
                            _pipe_ce = self._make_pipeline(
                                min_class_count, len(_y_ce_enc),
                                _X_ce.shape[1], feature_type, greedy_mode=True,
                                algo=algo)
                            # Single call: dispatches all (n_perm × n_folds) folds
                            # as one flat loky batch instead of n_perm serial calls.
                            # perm_scores stored so Path A can reuse them directly.
                            _, _cpval, _cp_scores, _, _, _cb_acc = self._run_loo(
                                _pipe_ce, _X_ce, _y_ce_enc, n_perm,
                                lambda m: None)
                            _combo_perm_results.append(
                                (_combo_ex, _cb_acc, _cpval))
                            _combo_perm_scores_map[frozenset(_combo_ex)] = _cp_scores
                        except Exception:
                            _combo_perm_results.append(
                                (_combo_ex, _combo_acc_ex, 1.0))
                    if _combo_perm_results and not self._cancel_event.is_set():
                        # Re-rank: primary = p-value ascending, secondary = accuracy desc
                        _combo_perm_results.sort(key=lambda r: (r[2], -r[1]))
                        _best_cpr = _combo_perm_results[0]
                        trace = [(_cid, _best_cpr[1])
                                 for _cid in sorted(_best_cpr[0], key=str)]
                        top_combos = [(_c, _a) for _c, _a, _ in _combo_perm_results]
                        # Retrieve the best combo's null distribution so Path A can
                        # reuse it without a second (n_perm × n_folds) loky batch.
                        _best_combo_perm_scores = _combo_perm_scores_map.get(
                            frozenset(_best_cpr[0]), np.array([], dtype=float))
                        _unit_lbl2 = (lambda c: f"T_{c[0]}→{c[1]}"
                                      if isinstance(c, tuple) else f"C{c}")
                        self._progress_q.put(("status",
                            f"{mname}: best combo by specificity: "
                            f"{{{','.join(_unit_lbl2(c) for c in sorted(_best_cpr[0], key=str))}}} "
                            f"(p={_best_cpr[2]:.3f})"))

                # ── Step 2: LOO — two mutually exclusive paths ────────────────
                # Path A (greedy succeeded): run LOO on the greedy-identified
                # cluster set, no SelectKBest inside folds.  Always use Path A
                # when a greedy trace exists — including when k_contrib is None
                # ("All"), because feeding every cluster into the pipeline bloats
                # the transition feature space as O(n²) and collapses accuracy.
                # The trace is pruned to the shortest prefix that achieves its
                # peak accuracy, so plateau clusters added after the best step
                # are excluded ("ignore further clusters once accuracy stops rising").
                # Path B (no greedy trace / no animal_data): original pipeline.
                _fixed_ids         = []    # cluster IDs used in all folds (Path A only)
                _nested_params     = None  # params needed for on-demand nested perm
                _tied_perm_results = []    # greedy tied-peak candidates ranked by pval
                # Down-weight short-recording animals in transition mode: their
                # row-normalised probabilities are unbiased but high-variance.
                _sw = (GroupPredictorPanel._compute_sample_weights(animal_data)
                       if _greedy_metric == "transition" and animal_data else None)
                if trace and animal_data is not None:
                    # Prune greedy traces to the prefix that first reaches max
                    # accuracy — drops zero-gain plateau clusters.
                    # Exhaustive traces must NOT be pruned: every entry carries
                    # the same best_bal_acc (whole-combo score), so _cut=0 would
                    # collapse the combo to a single cluster.
                    if _search_method != "exhaustive":
                        _peak_acc = max(a for _, a in trace)
                        _cut = next(i for i, (_, a) in enumerate(trace)
                                    if a >= _peak_acc - 1e-9)
                        trace = trace[:_cut + 1]

                    # ── Path A: greedy/exhaustive-informed LOO ────────────────
                    if _greedy_metric == "transition":
                        _sel_pairs = [pair for pair, _ in trace]
                        X_loo, y_loo_raw, fn_loo = self._build_transition_pairs(
                            animal_data, _sel_pairs)
                        _sel_method = ("exhaustive" if _search_method == "exhaustive"
                                       else "greedy")
                        self._progress_q.put(("status",
                            f"{mname}: LOO on {len(_sel_pairs)} {_sel_method}-selected "
                            f"transition(s) ({len(y_loo_raw) if y_loo_raw else '?'} folds)…"))
                    else:
                        _sel_units = [u for u, _ in trace]
                        _sel_cids  = [u for u in _sel_units if isinstance(u, int)]
                        _sel_gdict = {u: model_sel_groups[u]
                                      for u in _sel_units
                                      if isinstance(u, str)
                                      and model_sel_groups
                                      and u in model_sel_groups}
                        if _sel_gdict and _sel_cids:
                            X_loo, y_loo_raw, fn_loo = self._build_custom(
                                animal_data, _sel_cids, _sel_gdict,
                                _greedy_metric)
                            _feat_sfx = (f"{len(_sel_cids)} cluster(s)"
                                         f" + {len(_sel_gdict)} group(s)")
                        elif _sel_gdict:
                            # Only groups selected by greedy (no clusters)
                            X_loo, y_loo_raw, fn_loo = \
                                GroupPredictorPanel._build_group_feat(
                                    animal_data, _sel_gdict, _greedy_metric)
                            _feat_sfx = f"{len(_sel_gdict)} group(s)"
                        else:
                            X_loo, y_loo_raw, fn_loo = self._build_freq_or_dur(
                                animal_data, _greedy_metric,
                                cluster_ids=set(_sel_cids))
                            _feat_sfx = f"{len(_sel_cids)} cluster(s)"
                        self._progress_q.put(("status",
                            f"{mname}: LOO on {_feat_sfx} "
                            f"({len(y_loo_raw) if y_loo_raw else '?'} folds)…"))
                    y_loo = le.transform(y_loo_raw).astype(int)
                    # No k= → no SelectKBest; greedy already did feature selection.
                    # greedy_mode=True: same fixed-C classifier used in the greedy trace
                    # so the LOO accuracy here is directly comparable to the bar chart.
                    pipeline = self._make_pipeline(
                        min_class_count, len(y_loo), X_loo.shape[1], feature_type,
                        greedy_mode=True, algo=algo)
                    if _search_method == "exhaustive" and _combo_perm_results:
                        # Step 1b already ran the perm test for this exact combo;
                        # run LOO-only here (n_perm=0) and reuse Step 1b's null
                        # distribution — avoids a redundant (n_perm × n_folds)
                        # loky batch on the same data.
                        loo_acc, _, _, pred_labels, pred_proba, bal_acc = \
                            self._run_loo(pipeline, X_loo, y_loo, 0, progress_fn,
                                          sample_weight=_sw)
                        perm_scores = _best_combo_perm_scores
                        pval        = _combo_perm_results[0][2]
                    else:
                        # Greedy / no Step 1b: dispatch all (n_perm × n_folds)
                        # permutation folds as one flat loky batch.
                        loo_acc, pval, perm_scores, pred_labels, pred_proba, bal_acc = \
                            self._run_loo(pipeline, X_loo, y_loo, n_perm, progress_fn,
                                          sample_weight=_sw)
                    _pval_type = "cond"

                    # ── Greedy: perm-test tied candidates at the peak step ────
                    # Only for greedy (not exhaustive, which already tested all
                    # combos in Step 1b).  At the step where accuracy first peaked
                    # (_cut), multiple candidates may have tied — test each one
                    # with a conditional permutation test to find which is most
                    # specifically informative about experimental groups.
                    if (_search_method != "exhaustive" and _greedy_ties
                            and n_perm > 0 and not self._cancel_event.is_set()):
                        _peak_step_ties = _greedy_ties.get(_cut, [])
                        _winner_id  = trace[-1][0]
                        _prefix_ids = [item for item, _ in trace[:-1]]
                        _alternatives = [c for _, c in _peak_step_ties
                                         if c != _winner_id][:5]
                        if _alternatives:
                            progress_fn(
                                f"{mname}: testing {len(_alternatives)} "
                                f"tied peak candidate(s)…")
                        for _alt in _alternatives:
                            if self._cancel_event.is_set():
                                break
                            try:
                                _alt_set = _prefix_ids + [_alt]
                                if _greedy_metric == "transition":
                                    _X_alt, _y_alt_raw, _ = \
                                        self._build_transition_pairs(
                                            animal_data, _alt_set)
                                else:
                                    _alt_cids = [c for c in _alt_set
                                                 if isinstance(c, int)]
                                    _alt_gdict = {n: model_sel_groups[n]
                                                  for n in _alt_set
                                                  if isinstance(n, str)
                                                  and model_sel_groups
                                                  and n in model_sel_groups}
                                    if _alt_gdict:
                                        _X_alt, _y_alt_raw, _ = \
                                            self._build_custom(
                                                animal_data, _alt_cids,
                                                _alt_gdict, _greedy_metric)
                                    else:
                                        _X_alt, _y_alt_raw, _ = \
                                            self._build_freq_or_dur(
                                                animal_data, _greedy_metric,
                                                cluster_ids=set(_alt_cids))
                                _y_alt_enc = le.transform(
                                    _y_alt_raw).astype(int)
                                if (_X_alt.shape[1] == 0
                                        or len(np.unique(_y_alt_enc)) < 2):
                                    continue
                                # Single call: flat loky batch for all folds.
                                _, _alt_pval, _, _, _, _alt_bal = \
                                    self._run_loo(pipeline, _X_alt,
                                                  _y_alt_enc, n_perm,
                                                  lambda m: None)
                                _alt_label = (
                                    f"T_{_alt[0]}→{_alt[1]}"
                                    if isinstance(_alt, tuple)
                                    else (f"C{_alt}" if isinstance(_alt, int)
                                          else str(_alt)))
                                _tied_perm_results.append({
                                    "candidate": _alt,
                                    "label":     _alt_label,
                                    "bal_acc":   _alt_bal,
                                    "pval":      _alt_pval,
                                })
                            except Exception:
                                continue
                        if not self._cancel_event.is_set():
                            if _tied_perm_results:
                                _tied_perm_results.sort(key=lambda r: r["pval"])
                            # Prepend the winner so the full ranked list is self-contained
                            _winner_label = (
                                f"T_{_winner_id[0]}→{_winner_id[1]}"
                                if isinstance(_winner_id, tuple)
                                else (f"C{_winner_id}"
                                      if isinstance(_winner_id, int)
                                      else str(_winner_id)))
                            _tied_perm_results.insert(0, {
                                "candidate": _winner_id,
                                "label":     _winner_label,
                                "bal_acc":   bal_acc,
                                "pval":      pval,
                            })

                    # ── Fixed feature IDs (used in every LOO fold) ────────────
                    if _greedy_metric == "transition":
                        _fixed_ids = [f"{pair[0]}→{pair[1]}"
                                      for pair, _ in trace]
                    else:
                        # Mixed greedy: trace units can be int (cluster) or str (group)
                        _fixed_ids = [u for u, _ in trace]

                    # ── Params needed for on-demand nested permutation ────────
                    _nested_params = {
                        "greedy_metric":   _greedy_metric,
                        "greedy_cids":     list(_greedy_units),
                        "trace_len":       len(trace),
                        "min_class_count": min_class_count,
                        "n_groups":        len(exp_groups),
                        "y_loo_raw":       list(y_loo_raw),
                        "animal_data":     animal_data,
                        "n_samples":       len(y_loo),
                        "feature_type":    feature_type,
                        "sel_groups":      model_sel_groups,
                    }
                    self._progress_q.put(("progress", _base_p + 0.90 * _model_slice))

                    kappa = float(cohen_kappa_score(y_loo, pred_labels))
                    cm = confusion_matrix(y_loo, pred_labels)
                    p2 = _clone(pipeline)
                    _p2_fit_kw = ({f"{p2.steps[-1][0]}__sample_weight": _sw}
                                  if _sw is not None else {})
                    p2.fit(X_loo, y_loo, **_p2_fit_kw)
                    importances, coef_matrix = \
                        self._extract_importances(p2, X_loo.shape[1])
                    _vt2 = next((s for _, s in p2.steps
                                 if isinstance(s, VarianceThreshold)), None)
                    _poly2 = next((s for _, s in p2.steps
                                   if hasattr(s, "powers_")), None)
                    display_feat_names = list(fn_loo)
                    if _vt2 is not None:
                        _sup2 = _vt2.get_support()
                        if len(_sup2) == len(display_feat_names):
                            display_feat_names = [fn for fn, keep
                                                  in zip(display_feat_names, _sup2)
                                                  if keep]
                    if _poly2 is not None:
                        _n_o2 = len(display_feat_names)
                        display_feat_names = display_feat_names + [
                            f"{display_feat_names[_ii]} × {display_feat_names[_jj]}"
                            for _ii in range(_n_o2)
                            for _jj in range(_ii + 1, _n_o2)
                        ]
                    feat_names = list(fn_loo)
                    X = X_loo
                    y_enc = y_loo

                else:
                    # ── Path B: original pipeline (SelectKBest inside LOO) ────
                    pipeline = self._make_pipeline(
                        min_class_count, len(y_enc), n_features, feature_type,
                        k=k_contrib, algo=algo)
                    # _run_loo clones internally for every fold; no pre-clone needed
                    loo_acc, pval, perm_scores, pred_labels, pred_proba, bal_acc = \
                        self._run_loo(pipeline, X, y_enc, n_perm, progress_fn,
                                      sample_weight=_sw)
                    self._progress_q.put(("progress", _base_p + 0.90 * _model_slice))
                    _pval_type = "cond"
                    kappa = float(cohen_kappa_score(y_enc, pred_labels))
                    cm = confusion_matrix(y_enc, pred_labels)
                    # Final fit on all data for importances / coef_matrix
                    p2 = _clone(pipeline)
                    _p2_fit_kw = ({f"{p2.steps[-1][0]}__sample_weight": _sw}
                                  if _sw is not None else {})
                    p2.fit(X, y_enc, **_p2_fit_kw)
                    importances, coef_matrix = self._extract_importances(p2, X.shape[1])

                    # Build display_feat_names: VT mask → SKB mask → Poly expansion
                    display_feat_names = list(feat_names)
                    vt_step  = next((s for _, s in p2.steps
                                     if isinstance(s, VarianceThreshold)), None)
                    skb_step = next((s for _, s in p2.steps
                                     if isinstance(s, SelectKBest)), None)
                    poly_step = next((s for _, s in p2.steps
                                      if hasattr(s, "powers_")), None)
                    if vt_step is not None:
                        sup = vt_step.get_support()
                        if len(sup) == len(display_feat_names):
                            display_feat_names = [fn for fn, keep
                                                  in zip(display_feat_names, sup) if keep]
                    if skb_step is not None:
                        sup = skb_step.get_support()
                        if len(sup) == len(display_feat_names):
                            display_feat_names = [fn for fn, keep
                                                  in zip(display_feat_names, sup) if keep]
                    if poly_step is not None:
                        n_o = len(display_feat_names)
                        display_feat_names = display_feat_names + [
                            f"{display_feat_names[ii]} × {display_feat_names[jj]}"
                            for ii in range(n_o) for jj in range(ii + 1, n_o)
                        ]

                # ── Step 3: Shapley importance ────────────────────────────────
                shapley_result      = None
                shapley_phi_by_size = None
                shapley_vcache      = {}
                if trace and len(trace) >= 2 and animal_data is not None:
                    _n_cls = len(np.unique(y_enc))
                    _exact = len(trace) <= 8
                    _unit = ("transitions"
                             if feature_type in ("transition", "group_transition")
                             else "clusters")
                    self._progress_q.put(("status",
                        f"{mname}: computing Shapley values "
                        f"({len(trace)} {_unit}, "
                        f"{'exact' if _exact else '≈ Monte Carlo'})…"))
                    def _shap_prog(msg, _mn=mname):
                        self._progress_q.put(("status", f"{_mn} Shapley: {msg}"))
                    try:
                        if feature_type == "group_transition":
                            _shap_full = self._compute_shapley_group_transitions(
                                animal_data, model_sel_groups,
                                [t[0] for t in trace],
                                min_class_count, _n_cls,
                                progress_fn=_shap_prog,
                                algo=algo)
                        elif feature_type == "transition":
                            _shap_full = self._compute_shapley_transitions(
                                animal_data, [t[0] for t in trace],
                                min_class_count, _n_cls,
                                progress_fn=_shap_prog,
                                algo=algo)
                        else:
                            _shap_full = self._compute_shapley(
                                animal_data, [t[0] for t in trace],
                                min_class_count, _n_cls,
                                progress_fn=_shap_prog,
                                metric=_greedy_metric,
                                sel_groups=model_sel_groups if model_sel_groups else None,
                                algo=algo)
                        if _shap_full is not None and len(_shap_full) == 4:
                            _sb, _sr, _spbs, _svc = _shap_full
                            shapley_result      = (_sb, _sr)
                            shapley_phi_by_size = _spbs
                            shapley_vcache      = _svc
                    except Exception as _exc:
                        self._progress_q.put(("status",
                            f"{mname}: warning — Shapley computation "
                            f"failed ({_exc}); skipping importance chart."))

                results.append({
                    "name":                  mname,
                    "X":                     X,
                    "y_enc":                 y_enc,
                    "le":                    le,
                    "feat_names":            feat_names,
                    "display_feat_names":    display_feat_names,
                    "animal_names":          animal_names,
                    "acc":                   loo_acc,
                    "bal_acc":               bal_acc,
                    "pval":                  pval,
                    "pval_type":             _pval_type,
                    "perm_scores":           perm_scores,
                    "nested_perm_scores":    None,
                    "nested_perm_p":         None,
                    "fixed_cluster_ids":     _fixed_ids,
                    "_nested_params":        _nested_params,
                    "kappa":                 kappa,
                    "cm":                    cm,
                    "importances":           importances,
                    "coef_matrix":           coef_matrix,
                    "pred_labels":           pred_labels,
                    "pred_proba":            pred_proba,
                    "cluster_selection_trace": trace,
                    "trace_is_custom":       _trace_is_custom,
                    "shapley_result":        shapley_result,
                    "shapley_phi_by_size":   shapley_phi_by_size,
                    "shapley_vcache":        shapley_vcache,
                    "feature_type":          feature_type,
                    "search_method":         _search_method,
                    "top_combos":            top_combos,
                    "tied_perm_results":     _tied_perm_results,
                    "combo_perm_results":    _combo_perm_results,
                })
                self._progress_q.put(("progress", (i + 1) / _n_models))
            except Exception as exc:
                results.append({
                    "name": mname, "error": str(exc),
                    "acc": None, "bal_acc": None, "pval": None,
                    "pval_type": "cond",
                    "perm_scores": None, "kappa": None,
                    "cm": None, "importances": None,
                    "coef_matrix": None, "pred_proba": None,
                    "display_feat_names": None,
                    "cluster_selection_trace": None,
                    "trace_is_custom": False,
                    "shapley_result": None,
                    "shapley_phi_by_size": None,
                    "shapley_vcache": {},
                    "animal_names": animal_names,
                    "search_method": "greedy",
                    "top_combos": [],
                    "tied_perm_results":  [],
                    "combo_perm_results": [],
                })
                self._progress_q.put(("progress", (i + 1) / _n_models))

        # Shut down loky worker processes and WAIT for them to terminate.
        # wait=False (the former value) leaves workers alive while the daemon
        # thread exits; if the user re-runs before they finish dying, a second
        # pool spawns on top of the first — up to 16 Python processes loading
        # MKL simultaneously, which triggers the win32k kernel crash on the
        # June 2026 Windows update.  wait=True is safe here because this is a
        # daemon thread, not the GUI thread, so blocking it doesn't freeze the UI.
        # kill_workers=True force-terminates the OS processes synchronously so
        # they are gone before the next run's pool spawns; without it, wait=True
        # only waits for pending jobs to finish, not for the processes to exit.
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True, kill_workers=True)
        except Exception:
            pass
        finally:
            # "done" is guaranteed even if the loky shutdown raises unexpectedly,
            # so _poll_progress can always unlock the Run button.
            self._progress_q.put(("done", (results, chance, caveats, n_total, exp_groups)))

    def _poll_progress(self):
        """Called every 150 ms on the main thread to drain the progress queue."""
        import queue as _queue
        try:
            while True:
                kind, data = self._progress_q.get_nowait()
                if kind == "status":
                    self._status(data)
                    if (self._loading_status_lbl is not None
                            and self._loading_status_lbl.winfo_exists()):
                        self._loading_status_lbl.configure(text=data)
                elif kind == "progress":
                    if (self._loading_progress_bar is not None
                            and self._loading_progress_bar.winfo_exists()):
                        self._loading_progress_bar.set(float(data))
                elif kind == "done":
                    self._loading_progress_bar = None
                    self._loading_status_lbl   = None
                    try:
                        self._finish(data)
                    except Exception as _exc:
                        self._status(f"Display error: {_exc}", color=T()["error"])
                    finally:
                        self._running = False
                        if self._run_btn:
                            self._run_btn.configure(state="normal",
                                                    text="   Run Models")
                        if self._cancel_btn:
                            self._cancel_btn.configure(state="disabled")
                    return
                elif kind == "error":
                    self._loading_progress_bar = None
                    self._loading_status_lbl   = None
                    self._status(data, color=T()["error"])
                    self._running = False
                    if self._run_btn:
                        self._run_btn.configure(state="normal",
                                                text="   Run Models")
                    if self._cancel_btn:
                        self._cancel_btn.configure(state="disabled")
                    return
        except _queue.Empty:
            pass
        # Dead-thread guard: if the worker thread exited without sending "done"
        # (e.g. an unhandled BaseException like MemoryError that escaped both the
        # per-model except and the cleanup finally), unlock the GUI so the user
        # can try again rather than waiting forever.
        if self._running and hasattr(self, "_bg_thread") and not self._bg_thread.is_alive():
            self._loading_progress_bar = None
            self._loading_status_lbl   = None
            self._status("Analysis thread exited unexpectedly — please re-run.",
                         color=T()["error"])
            self._running = False
            if self._run_btn:
                self._run_btn.configure(state="normal", text="   Run Models")
            if self._cancel_btn:
                self._cancel_btn.configure(state="disabled")
            return
        # Reschedule until done
        self.after(150, self._poll_progress)

    def _finish(self, payload):
        """Called on the main thread when the worker completes successfully."""
        results, chance, caveats, n_total, exp_groups = payload
        self._results = results
        self._running = False
        self._last_chance = chance
        self._perm_display_mode = "cond"   # reset to conditional on each new run
        self._nested_running = False
        self._loading_progress_bar = None
        self._loading_status_lbl   = None
        if self._run_btn:
            self._run_btn.configure(state="normal", text="   Run Models")

        self._draw_overview(chance)
        self._draw_null_comparison()
        self._show_caveats(caveats)
        self._selected_model = 0
        self._view_model_var.set(0)
        # Enable nested test button when at least one model supports it
        _has_nested = any(
            r.get("_nested_params") is not None
            for r in results if not r.get("error"))
        if self._nested_btn:
            self._nested_btn.configure(
                state="normal" if _has_nested else "disabled",
                text="Run Nested Permutation Test")
        # Reset the Conditional/Nested toggle (hidden until nested test runs)
        if self._perm_toggle_frame:
            for _w in self._perm_toggle_frame.winfo_children():
                try:
                    _w.destroy()
                except Exception:
                    pass
            self._perm_toggle_frame.pack_forget()
        self._select_model(0)
        n_done = len([r for r in results if not r.get("error")])
        n_total_models = 3
        if len(results) < n_total_models:
            self._status(
                f"Cancelled — {n_done}/{n_total_models} models completed. "
                f"({len(exp_groups)} groups, n={n_total} animals)")
        else:
            self._status(f"Done. {len(exp_groups)} groups, n={n_total} animals.")

    def _collect_caveats(self, animals, exp_groups, min_class_count,
                          n_total, n_feats_max) -> list:
        caveats = []
        if min_class_count < 10:
            caveats.append(
                f"n < 10 in at least one group (min={min_class_count}) — "
                "LOO accuracy estimates have high variance; interpret cautiously.")
        for eg in exp_groups:
            n_eg = sum(1 for a in animals if a["exp_group"] == eg)
            if n_eg <= 2:
                caveats.append(
                    f"Only {n_eg} animal(s) in group '{eg}' — "
                    "leave-one-out is unreliable for that class.")
        if n_feats_max > n_total:
            caveats.append(
                f"n_features ({n_feats_max}) > n_animals ({n_total}) — "
                "model may overfit even with regularization.")
        # Transition-specific
        for a in animals:
            if len(a["df"]) < 10:
                caveats.append(
                    f"Animal '{a['name']}' has < 10 bouts — "
                    "transition probability estimates are unreliable.")
                break
        caveats.append(
            "Transition model uses the sequential bout order only — "
            "session recording order matters.")
        return caveats

    # ── Table update ──────────────────────────────────────────────────────────

    def _update_table(self, chance: float):
        # Metrics are displayed in the overview figure; no table widgets to update.
        pass

    # ── Caveat bar ────────────────────────────────────────────────────────────

    def _show_caveats(self, _caveats: list):
        self._caveat_fr.grid_remove()

    # ── Figure 0: model overview (shown immediately after Run) ───────────────

    def _draw_overview(self, chance: float):
        import matplotlib.pyplot as _plt
        for w in self._overview_fr.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        if self._overview_fig is not None:
            self._close_fig(self._overview_fig)
            self._overview_fig = None
        if self._null_comp_fig is not None:
            self._close_fig(self._null_comp_fig)
            self._null_comp_fig = None
        results = [r for r in self._results if r.get("acc") is not None]
        if not results:
            return

        try:
            import matplotlib.pyplot as _plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _FCA
            from matplotlib.patches import Patch as _Patch
            names     = [r["name"]                  for r in results]
            bal_accs  = [r.get("bal_acc", r["acc"]) for r in results]
            pvals     = [r["pval"]                  for r in results]
            kappas    = [r["kappa"]                 for r in results]
            pvaltypes = [r.get("pval_type", "cond") for r in results]

            # BH FDR correction across models
            _finite_idx = [i for i, p in enumerate(pvals) if p is not None]
            if len(_finite_idx) > 1:
                _bh_arr = benjamini_hochberg(np.array([pvals[i] for i in _finite_idx]))
                qvals = list(pvals)
                for _i, _q in zip(_finite_idx, _bh_arr):
                    qvals[_i] = float(_q)
            else:
                qvals = list(pvals)

            def _pcol(p):
                if p is None:  return "#888888"
                if p <= 0.05:  return "#4CAF50"
                if p <= 0.10:  return "#FF9800"
                return "#F44336"

            bar_cols = [_pcol(q) for q in qvals]
            x = np.arange(len(names))

            fig, (ax_a, ax_k) = _plt.subplots(1, 2, figsize=(12, 2.5),
                                               facecolor=T()["fig_bg"])
            self._overview_fig = fig

            # ── Accuracy bars (balanced accuracy only) ──
            self._style_axes(ax_a)
            w = 0.5
            bars_bal = ax_a.bar(x, bal_accs, w, color=bar_cols,
                                edgecolor=T()["spine"], linewidth=1.2,
                                label="Balanced Accuracy", zorder=3)
            ax_a.axhline(chance, color=T()["subtext"], linestyle="--",
                         linewidth=1.2, label=f"Chance ({chance:.0%})", zorder=2)
            for bar, ba, qv, pt in zip(bars_bal, bal_accs, qvals, pvaltypes):
                # q(n) = BH-corrected nested; q(c) = BH-corrected conditional
                _psuf = "(n)" if pt == "nested" else "(c)"
                plab = f"\nq{_psuf} = {qv:.3f}" if qv is not None else ""
                ax_a.text(bar.get_x() + bar.get_width() / 2, ba + 0.03,
                          f"{ba:.0%}{plab}",
                          ha="center", va="bottom", fontsize=8,
                          linespacing=1.4, color=T()["text"])
            ax_a.set_xticks(x)
            ax_a.set_xticklabels(names, color=T()["tick"], fontsize=10)
            ax_a.set_ylabel("Balanced Accuracy", color=T()["text"], fontsize=10)
            ax_a.set_ylim(0, 1.55)
            ax_a.set_title("Model Accuracy (LOO-CV)", color=T()["text"],
                           fontsize=11, pad=6)
            ax_a.yaxis.grid(True, color=T()["spine"], alpha=0.35, zorder=0)

            # ── κ dot chart ──
            self._style_axes(ax_k)
            k_min = min(kappas)
            k_max = max(kappas)
            y_lo  = min(-0.05, k_min - 0.20)
            y_hi  = max( 1.30, k_max + 0.35)
            for ref in [0.2, 0.4, 0.6]:
                if y_lo <= ref <= y_hi:
                    ax_k.axhline(ref, color=T()["spine"], linestyle=":", linewidth=0.8)
            ax_k.scatter(x, kappas, s=95, color=bar_cols,
                         edgecolors=T()["spine"], linewidths=1.2, zorder=3)
            for xi, kap in enumerate(kappas):
                y_above = kap + 0.04
                if y_above > 1.0:
                    # Push labels for near-perfect kappa above 1.2 to clear the data point
                    y_text, va_kap = max(y_above, 1.22), "bottom"
                elif y_above < y_hi - 0.04:
                    y_text, va_kap = y_above, "bottom"
                else:
                    y_text, va_kap = kap - 0.10, "top"
                ax_k.text(xi, y_text, f"κ={kap:.2f}",
                          ha="center", va=va_kap, fontsize=8, color=T()["text"])
            ax_k.set_xticks(x)
            ax_k.set_xticklabels(names, color=T()["tick"], fontsize=10)
            ax_k.set_ylabel("Cohen's κ", color=T()["text"], fontsize=10)
            ax_k.set_ylim(y_lo, y_hi)
            ax_k.set_title("Chance-Corrected Agreement", color=T()["text"],
                           fontsize=11, pad=6)
            ax_k.yaxis.grid(True, color=T()["spine"], alpha=0.35, zorder=0)
            legend_els = [_Patch(facecolor="#4CAF50", label="q ≤ 0.05"),
                          _Patch(facecolor="#FF9800", label="q ≤ 0.10"),
                          _Patch(facecolor="#F44336", label="q > 0.10")]
            ax_k.legend(handles=legend_els, bbox_to_anchor=(1.02, 1.0),
                        loc="upper left", borderaxespad=0,
                        fontsize=8, facecolor=T()["ax_bg"],
                        edgecolor=T()["spine"], labelcolor=T()["text"])

            fig.tight_layout(pad=2.0, w_pad=4.0, rect=[0, 0.04, 0.88, 1])
            fig.text(0.01, 0.01, "q = BH FDR-corrected p(c)/p(n) across models",
                     fontsize=6.5, color=T()["subtext"], style="italic", va="bottom")
            canvas = _FCA(fig, master=self._overview_fr)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="x", expand=True, padx=4, pady=2)
        except Exception as exc:
            ctk.CTkLabel(self._overview_fr,
                         text=f"Overview plot error: {exc}",
                         text_color=T()["muted"],
                         font=ctk.CTkFont(size=10)).pack(anchor="w", padx=8)

    def _draw_null_comparison(self):
        """
        Publication-quality null-distribution comparison: one panel per model
        showing the nested permutation null distribution (KDE + histogram fill)
        with the observed balanced accuracy marked.

        Design principles:
        - Scipy KDE gives a smooth publication-quality density curve.
        - Colorblind-safe palette (Okabe-Ito) used for significance.
        - Only bottom and left spines visible (minimal, journal-ready).
        - All colours drawn from T() so the figure matches dark/light theme.
        - Saved at 300 DPI to PNG+PDF via save_all_figures().
        """
        import matplotlib.pyplot as _plt
        import numpy as _np

        results = [r for r in self._results
                   if (r.get("perm_scores") is not None
                       and len(r.get("perm_scores", [])) > 0
                       and r.get("bal_acc") is not None)]
        if not results:
            return

        # BH FDR correction across models
        _pv_list = [r.get("pval") for r in results]
        _fn_idx  = [i for i, p in enumerate(_pv_list) if p is not None]
        if len(_fn_idx) > 1:
            _bh_q   = benjamini_hochberg(np.array([_pv_list[i] for i in _fn_idx]))
            _qv_list = list(_pv_list)
            for _i, _q in zip(_fn_idx, _bh_q):
                _qv_list[_i] = float(_q)
        else:
            _qv_list = list(_pv_list)

        # ── Okabe-Ito palette (colorblind-safe) ──────────────────────────────
        # Null distribution fill: muted slate-blue (visible on dark + light bg)
        _NULL_FILL  = "#7BA4C8"
        _NULL_LINE  = "#4D7EA8"
        # Significance: green / amber / vermilion  (Okabe-Ito)
        _COL_SIG    = "#009E73"   # p ≤ 0.05
        _COL_MARG   = "#E69F00"   # p ≤ 0.10
        _COL_NS     = "#D55E00"   # p > 0.10
        _COL_CHANCE = T()["subtext"]

        def _sig_col(p):
            if p is None:    return "#888888"
            if p <= 0.05:    return _COL_SIG
            if p <= 0.10:    return _COL_MARG
            return _COL_NS

        def _stars(p):
            if p is None:    return ""
            if p <= 0.001:   return "***"
            if p <= 0.01:    return "**"
            if p <= 0.05:    return "*"
            if p <= 0.10:    return "†"
            return "n.s."

        n_mod = len(results)
        fig_w = min(12.0, 4.2 * n_mod)   # match overview bar chart width cap
        fig_h = 3.2

        try:
            fig, axes = _plt.subplots(
                1, n_mod, figsize=(fig_w, fig_h),
                facecolor=T()["fig_bg"],
                sharey=False)
            if n_mod == 1:
                axes = [axes]
            self._null_comp_fig = fig

            for _idx, (ax, res) in enumerate(zip(axes, results)):
                null_sc = _np.asarray(res["perm_scores"], dtype=float)
                obs     = float(res["bal_acc"])
                qv      = _qv_list[_idx]
                pt      = res.get("pval_type", "cond")
                n_perm  = len(null_sc)
                n_cls   = (len(res["le"].classes_)
                           if res.get("le") is not None else 2)
                chance  = 1.0 / max(n_cls, 2)
                mname   = res["name"]
                oc      = _sig_col(qv)

                # ── Axes cosmetics: minimal spines ────────────────────────────
                ax.set_facecolor(T()["ax_bg"])
                for side, sp in ax.spines.items():
                    sp.set_visible(side in ("bottom", "left"))
                    if side in ("bottom", "left"):
                        sp.set_color(T()["spine"])
                        sp.set_linewidth(0.8)
                ax.tick_params(axis="both", colors=T()["tick"],
                               length=3, width=0.7)
                ax.yaxis.grid(True, color=T()["spine"],
                              alpha=0.20, lw=0.5, zorder=0)
                ax.set_axisbelow(True)

                # ── X-axis range ─────────────────────────────────────────────
                _lo = max(0.0, min(null_sc.min(), chance, obs) - 0.06)
                _hi = min(1.0, max(null_sc.max(), obs) + 0.06)
                x_grid = _np.linspace(_lo, _hi, 500)

                # ── Null distribution: KDE with histogram fill ────────────────
                _y_max = 1.0
                try:
                    from scipy.stats import gaussian_kde as _gkde
                    # Guard against degenerate (zero-variance) distributions
                    if null_sc.std() > 1e-6:
                        kde = _gkde(null_sc)
                        density = kde(x_grid)
                        _y_max = float(density.max())
                        # Histogram fill for texture (transparent)
                        ax.hist(null_sc, bins=min(25, max(8, n_perm // 8)),
                                density=True,
                                color=_NULL_FILL, alpha=0.20,
                                edgecolor="none", zorder=2)
                        # KDE fill
                        ax.fill_between(x_grid, density,
                                        color=_NULL_FILL, alpha=0.40, zorder=3)
                        # KDE outline
                        ax.plot(x_grid, density,
                                color=_NULL_LINE, linewidth=1.5,
                                zorder=4, label=f"Null (N={n_perm})")
                        # Tail shading (null ≥ observed)
                        _tm = x_grid >= obs
                        if _tm.any():
                            ax.fill_between(x_grid[_tm], density[_tm],
                                            color=oc, alpha=0.50, zorder=5)
                    else:
                        raise ValueError("zero-variance null")
                except Exception:
                    # Fallback: plain histogram when scipy unavailable or
                    # null distribution has no spread
                    counts, edges = _np.histogram(null_sc, bins=15, density=True)
                    _y_max = float(counts.max()) if len(counts) else 1.0
                    ax.bar(edges[:-1], counts, width=_np.diff(edges),
                           align="edge", color=_NULL_FILL, alpha=0.55,
                           edgecolor=_NULL_LINE, linewidth=0.6, zorder=2,
                           label=f"Null (N={n_perm})")
                    _tm = edges[:-1] >= obs
                    if _tm.any():
                        ax.bar(edges[:-1][_tm], counts[_tm],
                               width=_np.diff(edges)[_tm],
                               align="edge", color=oc, alpha=0.65,
                               edgecolor=oc, linewidth=0.6, zorder=3)

                # ── Chance level ──────────────────────────────────────────────
                ax.axvline(chance, color=_COL_CHANCE,
                           linewidth=0.9, linestyle=":",
                           zorder=6, alpha=0.75,
                           label=f"Chance ({chance:.0%})")

                # ── Observed balanced accuracy ─────────────────────────────────
                ax.axvline(obs, color=oc,
                           linewidth=2.0, linestyle="--",
                           zorder=7, label=f"Observed ({obs:.1%})")

                # ── Statistical annotation box ────────────────────────────────
                null_mean = float(null_sc.mean())
                null_std  = float(null_sc.std())
                z_score   = ((obs - null_mean) / null_std
                             if null_std > 1e-8 else 0.0)
                _ptag = "q(n)" if pt == "nested" else "q(c)"
                _pstr = f"{qv:.3f}" if qv is not None else "—"
                _star = _stars(qv)
                ann = (f"Bal. Acc. = {obs:.1%}\n"
                       f"{_ptag} = {_pstr}  {_star}\n"
                       f"Z = {z_score:+.2f}   Nᵖ = {n_perm}")
                ax.text(0.97, 0.97, ann,
                        transform=ax.transAxes,
                        ha="right", va="top",
                        fontsize=8.0, linespacing=1.55,
                        color=T()["text"],
                        bbox=dict(boxstyle="round,pad=0.40",
                                  facecolor=T()["fig_bg"],
                                  edgecolor=T()["spine"],
                                  linewidth=0.7,
                                  alpha=0.88),
                        zorder=11)

                # ── Axis decoration ───────────────────────────────────────────
                ax.set_xlabel("Balanced Accuracy",
                              color=T()["text"], fontsize=9.5, labelpad=5)
                ax.set_xlim(_lo, _hi)
                ax.set_ylim(bottom=0)
                ax.set_title(mname,
                             color=T()["text"], fontsize=11,
                             fontweight="semibold", pad=7)
                ax.xaxis.set_tick_params(labelsize=8)
                ax.yaxis.set_tick_params(labelsize=7.5)

                # Y-axis label on leftmost panel only
                if res is results[0]:
                    ax.set_ylabel("Density",
                                  color=T()["text"], fontsize=9.5, labelpad=5)
                    ax.legend(fontsize=7.5,
                              facecolor=T()["ax_bg"],
                              edgecolor=T()["spine"],
                              labelcolor=T()["tick"],
                              framealpha=0.85,
                              loc="upper left",
                              handlelength=1.4)
                else:
                    ax.set_ylabel("")

            # ── Figure-level title and footnote ───────────────────────────────
            _has_nested_display = any(r.get("pval_type") == "nested"
                                      for r in results)
            if _has_nested_display:
                _fig_title = "Group Discriminability — Nested Permutation Null Test"
                _note = (
                    "p(n): full greedy+LOO pipeline re-run per permutation — "
                    "accounts for feature selection bias.   "
                    "Shaded tail = null permutations ≥ observed accuracy.   "
                    "Z = (obs − nullμ) / nullσ.   q(n) = BH FDR-corrected across models.")
            else:
                _fig_title = "Group Discriminability — Conditional Permutation Null Test"
                _note = (
                    "p(c): classifier-only permutation on the fixed greedy-selected "
                    "cluster set — answers 'are these specific clusters predictive?'   "
                    "Shaded tail = null permutations ≥ observed accuracy.   "
                    "Z = (obs − nullμ) / nullσ.   q(c) = BH FDR-corrected across models.")
            fig.suptitle(_fig_title,
                         color=T()["text"], fontsize=11,
                         fontweight="bold", y=0.97)
            fig.text(0.5, 0.02, _note,
                     ha="center", va="bottom",
                     fontsize=6.5, color=T()["subtext"],
                     style="italic")

            fig.tight_layout(rect=[0, 0.08, 1, 0.92], pad=1.2, w_pad=2.0)
            # Figure stored; embedded in _draw_detail (scrollable section)

        except Exception as exc:
            self._null_comp_fig = None
            import traceback as _tb
            print(f"Null comparison build error: {exc}\n{_tb.format_exc()}")

    # ── Model selection & detail view ─────────────────────────────────────────

    def _select_model(self, idx: int):
        if not self._results or idx >= len(self._results):
            return
        self._selected_model = idx
        res = self._results[idx]
        if res.get("error"):
            self._show_detail_placeholder(
                f"Model '{res['name']}' failed:\n{res['error']}")
            return
        self._draw_detail(res)

    # ── Nested permutation test (on-demand) ───────────────────────────────────

    def _trigger_nested_test(self):
        """Start the nested permutation test in a background thread."""
        if getattr(self, "_nested_running", False):
            return
        self._nested_running = True
        if self._nested_btn:
            self._nested_btn.configure(state="disabled",
                                       text="Running nested test…")
        self._status("Running nested permutation test…")
        import threading as _thr
        n_perm = int(self._nperm_var.get())  # read on main thread — tkinter vars are not thread-safe
        _thr.Thread(target=self._nested_worker, args=(n_perm,), daemon=True).start()

    def _nested_worker(self, n_perm):
        """Worker: re-runs greedy+LOO under shuffled labels for all Path A models."""
        try:
            from joblib import Parallel as _Parallel, delayed as _delayed
            import os as _os
            _rng_null = np.random.default_rng(42)

            for res in self._results:
                if res.get("error") or res.get("_nested_params") is None:
                    continue
                p               = res["_nested_params"]
                _greedy_metric  = p["greedy_metric"]
                _greedy_cids    = p["greedy_cids"]
                _trace_len      = p["trace_len"]
                _min_cc         = p["min_class_count"]
                _n_groups       = p["n_groups"]
                _y_loo_raw      = p["y_loo_raw"]
                _animal_data    = p["animal_data"]
                _null_sel_groups = p.get("sel_groups", {})
                le              = res["le"]
                bal_acc         = res.get("bal_acc", res["acc"])

                _y_base = list(_y_loo_raw)
                _all_perms = [list(_rng_null.permutation(_y_base))
                              for _ in range(n_perm)]

                # Precompute feature arrays (mirrors Path A nested block)
                _null_X_pre = _null_y_pre = _null_col_map = None
                _null_raw_mats = _null_cl_idx = None
                try:
                    if _greedy_metric != "transition":
                        # _greedy_cids may contain group names (str) in mixed mode;
                        # the precomputed cluster matrix only covers integer IDs.
                        _null_int_cids = [c for c in _greedy_cids
                                          if isinstance(c, int)]
                        if _null_int_cids:
                            (_null_X_pre, _null_y_raw,
                             _null_fn) = self._build_freq_or_dur(
                                _animal_data, _greedy_metric,
                                cluster_ids=set(_null_int_cids))
                            _null_y_pre = np.array(_null_y_raw)
                            _null_col_map = {}
                            for _ci_i, _fn in enumerate(_null_fn):
                                try:
                                    _null_col_map[int(_fn.split()[-1])] = _ci_i
                                except (ValueError, IndexError):
                                    pass
                    else:
                        _all_ncids  = sorted(_greedy_cids)
                        _null_cl_arr = np.array(_all_ncids)
                        _null_cl_idx = {c: k for k, c in enumerate(_all_ncids)}
                        _n_nc = len(_all_ncids)
                        _null_raw_mats = []
                        for _a in _animal_data:
                            _lbls = _a["df"]["label"].values
                            _mat  = np.zeros((_n_nc, _n_nc), dtype=float)
                            if len(_lbls) > 1:
                                _ok = np.isin(_lbls, _null_cl_arr)
                                _ii = np.searchsorted(_null_cl_arr, _lbls)
                                _vi = np.where(_ok)[0]
                                if len(_vi) > 1:
                                    _cc = np.where(np.diff(_vi) == 1)[0]
                                    np.add.at(
                                        _mat,
                                        (_ii[_vi[_cc]], _ii[_vi[_cc + 1]]),
                                        1.0)
                            _null_raw_mats.append(_mat)
                except Exception:
                    pass

                # When groups are present we must use the thread path so that
                # animal_data is available to compute group features per permutation.
                _use_loky = (((_null_X_pre is not None)
                              or (_null_raw_mats is not None))
                             and not _null_sel_groups)
                _n_jobs = max(1, min((_os.cpu_count() or 4) // 2, 4))

                if _use_loky:
                    try:
                        nested_scores = np.array(
                            _Parallel(n_jobs=_n_jobs, backend="loky",
                                      max_nbytes=None)(
                                _delayed(
                                    GroupPredictorPanel._run_one_null_perm
                                )(
                                    list(yp), None, _greedy_metric,
                                    list(_greedy_cids), _trace_len,
                                    _min_cc, _n_groups, None,
                                    _null_X_pre, _null_y_pre,
                                    _null_col_map, _null_raw_mats, _null_cl_idx,
                                    _null_sel_groups)
                                for yp in _all_perms))
                    except Exception:
                        # loky failed — retry with threads; precomputed arrays are
                        # still accessible in-process so the fast path still fires.
                        nested_scores = np.array(
                            _Parallel(n_jobs=_n_jobs, prefer="threads")(
                                _delayed(
                                    GroupPredictorPanel._run_one_null_perm
                                )(
                                    list(yp), None, _greedy_metric,
                                    list(_greedy_cids), _trace_len,
                                    _min_cc, _n_groups, None,
                                    _null_X_pre, _null_y_pre,
                                    _null_col_map, _null_raw_mats, _null_cl_idx,
                                    _null_sel_groups)
                                for yp in _all_perms))
                    # wait=True: block this daemon thread until the workers are
                    # fully dead.  kill_workers=True force-terminates OS processes
                    # so they are gone before the next run's pool spawns — without
                    # it, wait=True only waits for pending jobs, not process exit.
                    try:
                        from joblib.externals.loky import get_reusable_executor
                        get_reusable_executor().shutdown(wait=True, kill_workers=True)
                    except Exception:
                        pass
                else:
                    _sg_cap = _null_sel_groups
                    def _eval_n(y_pg, _sg=_sg_cap):
                        return GroupPredictorPanel._run_one_null_perm(
                            y_pg, _animal_data, _greedy_metric,
                            list(_greedy_cids), _trace_len,
                            _min_cc, _n_groups, None,
                            sel_groups=_sg)
                    nested_scores = np.array(
                        _Parallel(n_jobs=_n_jobs, prefer="threads")(
                            _delayed(_eval_n)(yp) for yp in _all_perms))

                nested_p = ((float(np.sum(nested_scores >= bal_acc)) + 1)
                            / (len(nested_scores) + 1))
                res["nested_perm_scores"] = nested_scores
                res["nested_perm_p"]      = nested_p
        except Exception as _ne:
            _msg = str(_ne)
            self.after(0, lambda m=_msg: self._status(
                f"Nested permutation failed: {m}", color=T()["error"]))
        finally:
            self._nested_running = False
            self.after(0, self._on_nested_test_done)

    def _refresh_perm_toggle(self):
        """Rebuild the Conditional / Nested toggle in the left panel permutation section."""
        if self._perm_toggle_frame is None:
            return
        _any_nested = any(r.get("nested_perm_scores") is not None
                          for r in self._results if not r.get("error"))
        if not _any_nested:
            self._perm_toggle_frame.pack_forget()
            return
        for _w in self._perm_toggle_frame.winfo_children():
            try:
                _w.destroy()
            except Exception:
                pass
        _mode = getattr(self, "_perm_display_mode", "cond")
        ctk.CTkLabel(self._perm_toggle_frame,
                     text="Null shown:",
                     text_color=T()["muted"],
                     font=ctk.CTkFont(size=9)).pack(side="left", padx=(0, 4))
        ctk.CTkButton(self._perm_toggle_frame, text="Conditional",
                      width=90, height=24,
                      fg_color=(T()["border"] if _mode == "cond" else T()["card"]),
                      command=lambda: self._switch_perm_mode("cond")
                      ).pack(side="left", padx=(0, 2))
        ctk.CTkButton(self._perm_toggle_frame, text="Nested",
                      width=70, height=24,
                      fg_color=(T()["border"] if _mode == "nested" else T()["card"]),
                      command=lambda: self._switch_perm_mode("nested")
                      ).pack(side="left", padx=(0, 4))
        self._perm_toggle_frame.pack(padx=12, pady=(2, 6))

    def _on_nested_test_done(self):
        """Called on main thread when nested worker completes."""
        if self._nested_btn:
            self._nested_btn.configure(state="normal",
                                       text="Run Nested Permutation Test")
        self._refresh_perm_toggle()
        # Only show "complete" if at least one model actually produced nested scores;
        # on failure the worker already scheduled an error status via self.after(0, …)
        # which runs before this callback, so we must not overwrite it here.
        _any_done = any(r.get("nested_perm_scores") is not None
                        for r in self._results if not r.get("error"))
        if _any_done:
            self._status("Nested permutation test complete.")
        # _draw_overview must run first — it closes _null_comp_fig.
        # _draw_null_comparison must follow so the new figure exists for _draw_detail.
        self._draw_overview(self._last_chance)
        self._draw_null_comparison()
        if self._results and self._selected_model < len(self._results):
            self._draw_detail(self._results[self._selected_model])

    def _switch_perm_mode(self, mode: str):
        """Toggle histogram display between conditional and nested null."""
        self._perm_display_mode = mode
        for res in self._results:
            if res.get("error"):
                continue
            if mode == "nested" and res.get("nested_perm_scores") is not None:
                if "_cond_perm_scores" not in res:
                    res["_cond_perm_scores"] = res.get("perm_scores")
                    res["_cond_pval"]        = res.get("pval")
                    res["_cond_pval_type"]   = res.get("pval_type")
                res["perm_scores"] = res["nested_perm_scores"]
                res["pval"]        = res["nested_perm_p"]
                res["pval_type"]   = "nested"
            elif mode == "cond" and "_cond_perm_scores" in res:
                res["perm_scores"] = res["_cond_perm_scores"]
                res["pval"]        = res["_cond_pval"]
                res["pval_type"]   = res["_cond_pval_type"]
        self._refresh_perm_toggle()
        # _draw_overview must run first — it closes _null_comp_fig.
        # _draw_null_comparison must follow so the new figure exists for _draw_detail.
        self._draw_overview(self._last_chance)
        self._draw_null_comparison()
        if self._results and self._selected_model < len(self._results):
            self._draw_detail(self._results[self._selected_model])

    def _draw_detail(self, res: dict):
        import matplotlib.pyplot as _plt
        # Destroy Tk widgets BEFORE closing figures.  If plt.close() runs first
        # it tries to destroy the embedded canvas widget; the subsequent widget
        # loop then hits an already-dead widget and raises a TclError that
        # crashes the whole process on replot.
        for w in self._detail_inner.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        # Break the _null_comp_fig ↔ old-canvas reference cycle immediately after
        # the widget is destroyed and before any background thread can run.
        # Without this, the orphaned FigureCanvasTkAgg (holding a dead Tk widget
        # reference) may be deferred to Python's cyclic GC and collected from a
        # background LOO thread, causing a TclError that terminates the process.
        if self._null_comp_fig is not None:
            try:
                self._null_comp_fig.canvas = None
            except Exception:
                pass
        for fig in self._open_figs:
            self._close_fig(fig)
        self._open_figs = []
        import gc as _gc
        _gc.collect()

        le          = res["le"]
        cm          = res["cm"]
        classes     = le.classes_
        n_classes   = len(classes)
        # display_feat_names includes interaction terms (e.g. "Cluster 3 × Cluster 7")
        # and is sized to match importances after PolynomialFeatures expansion.
        feat_names  = res.get("display_feat_names") or res["feat_names"]
        importances = res["importances"]
        coef_matrix = res.get("coef_matrix")
        pred_proba  = res.get("pred_proba")
        perm_scores = res.get("perm_scores")
        pval_type   = res.get("pval_type", "cond")
        y_enc       = res["y_enc"]
        pred_labels = res["pred_labels"]
        acc         = res["acc"]
        bal_acc     = res.get("bal_acc", acc)
        kappa       = res["kappa"]
        pval        = res["pval"]

        try:
            import re as _re
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg as _FCA,
                NavigationToolbar2Tk as _NTK,
            )
            from matplotlib.patches import Patch as _Patch

            _GROUP_PALETTE = [
                "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
                "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
            ]
            grp_colors = {cls: _GROUP_PALETTE[i % len(_GROUP_PALETTE)]
                          for i, cls in enumerate(classes)}

            def _embed(fig, pady=(4, 2)):
                self._open_figs.append(fig)
                canvas = _FCA(fig, master=self._detail_inner)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="x", expand=True,
                                            padx=4, pady=pady)
                tb_fr = ctk.CTkFrame(self._detail_inner,
                                     fg_color=T()["card2"], height=28)
                tb_fr.pack(fill="x", padx=4, pady=(0, 6))
                tb_fr.pack_propagate(False)
                _NTK(canvas, tb_fr)

            def _embed_error(label):
                ctk.CTkLabel(
                    self._detail_inner,
                    text=f"⚠ Plot error — {label}",
                    text_color=T()["muted"],
                    font=ctk.CTkFont(size=10),
                    justify="left").pack(anchor="w", padx=12, pady=4)

            # ════════════════════════════════════════════════════════════════
            # Null comparison (all models) — shown at top of scrollable area
            # ════════════════════════════════════════════════════════════════
            if self._null_comp_fig is not None:
                try:
                    nc_canvas = _FCA(self._null_comp_fig,
                                     master=self._detail_inner)
                    nc_canvas.draw()
                    nc_canvas.get_tk_widget().pack(fill="x", expand=True,
                                                   padx=4, pady=(4, 2))
                except Exception:
                    pass

            # ── Cluster info bar + View Model selector (always shown) ────────
            _fids = res.get("fixed_cluster_ids", [])
            _ctrl_fr = ctk.CTkFrame(self._detail_inner,
                                    fg_color=T()["card2"],
                                    corner_radius=6)
            _ctrl_fr.pack(fill="x", padx=4, pady=(0, 6))

            # Left: features locked in across all folds (bold)
            if _fids:
                def _fmt_fid(f):
                    if isinstance(f, int):
                        return f"C{f}"
                    return str(f)   # group name or "i→j" string — display as-is
                _fid_str = "  ".join(_fmt_fid(f) for f in _fids)
                _fid_prefix = ("Features" if any(isinstance(f, str) for f in _fids)
                               else "Clusters")
                ctk.CTkLabel(
                    _ctrl_fr,
                    text=f"{_fid_prefix} (all folds):   {_fid_str}",
                    text_color=T()["text"],
                    font=ctk.CTkFont(size=11, weight="bold"),
                    justify="left"
                ).pack(side="left", padx=(12, 6), pady=7)

            # Greedy tied-peak: show the most specifically informative candidate
            _tpr = res.get("tied_perm_results") or []
            if len(_tpr) > 1:
                _best_tied = _tpr[0]  # lowest p-value = most specific
                ctk.CTkLabel(
                    _ctrl_fr,
                    text=f"Most specific (perm):  {_best_tied['label']}"
                         f"   p={_best_tied['pval']:.3f}",
                    text_color=T()["hdr_text"],
                    font=ctk.CTkFont(size=10, weight="bold"),
                    justify="left"
                ).pack(side="left", padx=(4, 6), pady=7)

            # Right: View Model radio buttons
            _vm_fr = ctk.CTkFrame(_ctrl_fr, fg_color="transparent")
            _vm_fr.pack(side="right", padx=10, pady=4)
            ctk.CTkLabel(_vm_fr,
                         text="View Model:",
                         text_color=T()["subtext"],
                         font=ctk.CTkFont(size=10, weight="bold"),
                         ).pack(side="left", padx=(0, 8))
            self._model_radio_btns = []
            for _mi, _mname in enumerate(["Frequency", "Total Duration", "Transition Prob."]):
                _rb = ctk.CTkRadioButton(
                    _vm_fr, text=_mname,
                    variable=self._view_model_var, value=_mi,
                    command=lambda i=_mi: self._select_model(i),
                    text_color=T()["subtext"],
                )
                _rb.pack(side="left", padx=(0, 10))
                self._model_radio_btns.append(_rb)

            # ════════════════════════════════════════════════════════════════
            # Figure 1 — Confusion matrix + ROC curve
            # ════════════════════════════════════════════════════════════════
            has_proba = (pred_proba is not None
                         and pred_proba.sum() > 0)
            n_cols = 2 if has_proba else 1
            fig1_w = max(8, n_classes * 1.5) + (5.5 if has_proba else 0)
            fig1_h = max(5.5, n_classes * 1.4)
            fig1, axes1 = _plt.subplots(1, n_cols,
                                        figsize=(fig1_w, fig1_h),
                                        facecolor=T()["fig_bg"],
                                        constrained_layout=True)
            if n_cols == 1:
                axes1 = [axes1]
            ax_cm = axes1[0]

            # Confusion matrix
            self._style_axes(ax_cm)
            row_n = cm.sum(axis=1, keepdims=True)
            cm_norm = np.where(row_n > 0, cm / row_n, 0.0)
            im = ax_cm.imshow(cm_norm, cmap="Blues",
                              vmin=0, vmax=1, aspect="auto")
            _plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04,
                          label="Recall")
            im.colorbar.ax.yaxis.set_tick_params(color=T()["tick"])
            im.colorbar.ax.tick_params(labelcolor=T()["tick"])
            ax_cm.set_xticks(range(n_classes))
            ax_cm.set_yticks(range(n_classes))
            ax_cm.set_xticklabels(classes, rotation=30, ha="right",
                                   color=T()["tick"], fontsize=9)
            ax_cm.set_yticklabels(classes, color=T()["tick"], fontsize=9)
            ax_cm.set_xlabel("Predicted", color=T()["text"], fontsize=10)
            ax_cm.set_ylabel("True", color=T()["text"], fontsize=10)
            ax_cm.set_title(f"{res['name']} — Confusion Matrix",
                             color=T()["text"], fontsize=11, pad=8)
            # Annotate: count + row%
            for i in range(n_classes):
                for j in range(n_classes):
                    val  = cm[i, j]
                    pct  = cm_norm[i, j]
                    cell_col = "white" if pct > 0.55 else T()["text"]
                    ax_cm.text(j, i, f"{val}\n{pct:.0%}",
                               ha="center", va="center", fontsize=8,
                               color=cell_col, linespacing=1.4)
            # Bold outline on diagonal
            for k in range(n_classes):
                ax_cm.add_patch(_plt.Rectangle(
                    (k - 0.5, k - 0.5), 1, 1,
                    fill=False, edgecolor=T()["text"],
                    linewidth=2.2, clip_on=True))
            # Recall + precision annotations
            col_n = cm.sum(axis=0)
            for i in range(n_classes):
                rec = cm_norm[i, i]
                ax_cm.text(n_classes - 0.42, i,
                            f"  R={rec:.0%}",
                            va="center", fontsize=7.5,
                            color=T()["subtext"])
            for j in range(n_classes):
                prec = (cm[j, j] / col_n[j]) if col_n[j] > 0 else 0.0
                ax_cm.text(j, n_classes - 0.42,
                            f"P={prec:.0%}",
                            ha="center", va="top", fontsize=7.5,
                            color=T()["subtext"])
            ax_cm.set_xlim(-0.5, n_classes + 0.9)
            ax_cm.set_ylim(n_classes - 0.2, -0.5)

            # ROC curve
            if has_proba:
                from sklearn.metrics import roc_curve, auc as _auc
                ax_roc = axes1[1]
                self._style_axes(ax_roc)
                ax_roc.plot([0, 1], [0, 1], linestyle="--",
                             color=T()["subtext"], linewidth=1,
                             label="Chance")
                if n_classes == 2:
                    fpr, tpr, _ = roc_curve(y_enc, pred_proba[:, 1])
                    roc_auc = _auc(fpr, tpr)
                    ax_roc.plot(fpr, tpr, color=_GROUP_PALETTE[0],
                                linewidth=2,
                                label=f"AUC = {roc_auc:.2f}")
                    ax_roc.fill_between(fpr, tpr, alpha=0.15,
                                        color=_GROUP_PALETTE[0])
                else:
                    macro_tpr = np.zeros(100)
                    fpr_grid  = np.linspace(0, 1, 100)
                    class_aucs = []
                    for ci, cls in enumerate(classes):
                        y_bin = (y_enc == ci).astype(int)
                        if y_bin.sum() == 0:
                            continue
                        fpr_c, tpr_c, _ = roc_curve(y_bin, pred_proba[:, ci])
                        roc_c = _auc(fpr_c, tpr_c)
                        class_aucs.append(roc_c)
                        col = _GROUP_PALETTE[ci % len(_GROUP_PALETTE)]
                        ax_roc.plot(fpr_c, tpr_c, color=col,
                                    linewidth=1.5, alpha=0.8,
                                    label=f"{cls} (AUC={roc_c:.2f})")
                        macro_tpr += np.interp(fpr_grid, fpr_c, tpr_c)
                    if class_aucs:
                        macro_tpr /= len(class_aucs)
                        macro_auc = _auc(fpr_grid, macro_tpr)
                        ax_roc.plot(fpr_grid, macro_tpr,
                                    color=T()["text"], linewidth=2.2,
                                    linestyle="--",
                                    label=f"Macro AUC={macro_auc:.2f}")
                        ax_roc.fill_between(fpr_grid, macro_tpr,
                                            alpha=0.10, color=T()["text"])
                ax_roc.set_xlabel("False Positive Rate",
                                   color=T()["text"], fontsize=10)
                ax_roc.set_ylabel("True Positive Rate",
                                   color=T()["text"], fontsize=10)
                ax_roc.set_title("ROC Curve (LOO-CV)",
                                  color=T()["text"], fontsize=11, pad=8)
                ax_roc.legend(fontsize=8, facecolor=T()["ax_bg"],
                               edgecolor=T()["spine"],
                               labelcolor=T()["text"])
                n_a = len(y_enc)
                ax_roc.text(0.98, 0.04,
                             f"n = {n_a} animals",
                             ha="right", va="bottom",
                             transform=ax_roc.transAxes,
                             fontsize=8, color=T()["subtext"])

            _embed(fig1, pady=(4, 2))

            # ════════════════════════════════════════════════════════════════
            # Figure 2 — Permutation null distribution
            # ════════════════════════════════════════════════════════════════
            if perm_scores is not None and len(perm_scores) > 0:
                # pval_type "nested": greedy+LOO re-run per permutation (Path A).
                # pval_type "cond":   classifier re-run on fixed X per perm (Path B).
                _is_nested = (pval_type == "nested")
                _ptype_title = (
                    "Nested Null (greedy+LOO re-run per perm)"
                    if _is_nested else
                    "Conditional Null (fixed features, labels shuffled)")
                # Null scores are balanced accuracy; compare against bal_acc.
                _ref_acc = bal_acc
                fig3, ax3 = _plt.subplots(figsize=(8, 4.5),
                                           facecolor=T()["fig_bg"])
                self._style_axes(ax3)
                ax3.hist(perm_scores, bins=30, color="#4E79A7",
                         alpha=0.72, edgecolor=T()["spine"],
                         linewidth=0.5, zorder=3,
                         label=f"Null ({len(perm_scores)} permutations)")
                ax3.axvline(_ref_acc, color="#F28E2B", linewidth=2.2,
                            linestyle="--",
                            label=f"Observed: {_ref_acc:.0%}")
                # Shade the tail (null scores ≥ observed)
                tail_vals = perm_scores[perm_scores >= _ref_acc]
                if len(tail_vals):
                    bins_edge = np.linspace(perm_scores.min(),
                                            perm_scores.max(), 31)
                    ax3.hist(tail_vals, bins=bins_edge, color="#F44336",
                             alpha=0.45, zorder=4)
                _psuf = "(n)" if _is_nested else "(c)"
                ax3.text(0.97, 0.92,
                         f"p{_psuf} = {pval:.3f} (raw)",
                         transform=ax3.transAxes, ha="right", va="top",
                         fontsize=9, color="#F28E2B",
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor=T()["fig_bg"],
                                   edgecolor=T()["spine"], alpha=0.85))
                ax3.set_xlabel("Balanced Accuracy (null)",
                               color=T()["text"], fontsize=10)
                ax3.set_ylabel("Count", color=T()["text"], fontsize=10)
                ax3.set_title(
                    f"Permutation Test — {res['name']}\n"
                    f"({_ptype_title})",
                    color=T()["text"], fontsize=10, pad=8)
                ax3.legend(fontsize=8, facecolor=T()["ax_bg"],
                           edgecolor=T()["spine"], labelcolor=T()["text"])
                ax3.yaxis.grid(True, color=T()["spine"], alpha=0.35, zorder=0)
                fig3.tight_layout(pad=2.5)
                _embed(fig3, pady=(2, 2))

            # ════════════════════════════════════════════════════════════════
            # Figure 3 — Feature contribution (signed bar / heatmap)
            # ════════════════════════════════════════════════════════════════
            if coef_matrix is not None and importances is not None \
                    and len(importances) > 0:
                top_n   = min(20, len(feat_names))
                top_idx = np.argsort(importances)[::-1][:top_n]
                top_feat_names = [feat_names[i] for i in top_idx]

                # Truncate long names (interaction terms) to prevent axis overflow
                _MAX_FN = 28
                top_feat_names = [fn if len(fn) <= _MAX_FN
                                  else fn[:_MAX_FN - 1] + "…"
                                  for fn in top_feat_names]

                # Detect mix-mode: cluster features are named "Cluster N",
                # group features are named by the user (no digit-only suffix).
                _is_cluster = [bool(_re.match(r"Cluster \d+$", fn))
                                for fn in top_feat_names]

                _max_imp = float(np.abs(importances).max()) if len(importances) else 0.0
                if _max_imp < 1e-10:
                    # All coefficients zeroed by regularisation — show a note
                    ctk.CTkLabel(
                        self._detail_inner,
                        text="ℹ  All feature contributions are near-zero.\n"
                             "The regularised model found no discriminative"
                             " features in this source.\n"
                             "Suggestions: switch to Total Duration, use Custom"
                             " mode with fewer clusters, or add more animals.",
                        text_color=T()["subtext"],
                        font=ctk.CTkFont(size=9),
                        wraplength=620,
                        justify="left",
                    ).pack(anchor="w", padx=8, pady=(4, 10))
                elif n_classes == 2:
                    # Signed bar (binary)
                    if coef_matrix.shape[0] == 1:
                        signed = coef_matrix[0][top_idx]
                    else:
                        signed = (coef_matrix[1] - coef_matrix[0])[top_idx]

                    fig4_h = max(5, top_n * 0.42)
                    fig4, ax4 = _plt.subplots(figsize=(9, fig4_h),
                                               facecolor=T()["fig_bg"])
                    self._style_axes(ax4)
                    bar_cols4 = ["#E07B54" if v >= 0 else "#4E79A7"
                                 for v in signed]
                    ax4.barh(range(top_n)[::-1], signed, color=bar_cols4,
                             edgecolor=T()["spine"], linewidth=0.6, zorder=3)
                    ax4.axvline(0, color=T()["text"], linewidth=0.9,
                                linestyle="--", zorder=2)
                    # Y-tick labels with bold for group features
                    ax4.set_yticks(list(range(top_n)[::-1]))
                    ax4.set_yticklabels(top_feat_names, fontsize=9,
                                        color=T()["tick"])
                    for lbl, is_cl in zip(ax4.get_yticklabels(), _is_cluster[::-1]):
                        lbl.set_fontweight("normal" if is_cl else "bold")
                    ax4.set_xlabel("Coefficient  (+ → group B, − → group A)",
                                   color=T()["text"], fontsize=10)
                    ax4.set_title(
                        f"{res['name']} — Feature Contributions\n"
                        f"({classes[0]} ← 0 → {classes[1]})",
                        color=T()["text"], fontsize=11, pad=8)
                    ax4.yaxis.grid(False)
                    ax4.xaxis.grid(True, color=T()["spine"],
                                   alpha=0.35, zorder=0)
                    legend_b = [_Patch(facecolor="#E07B54",
                                       label=f"+ → {classes[1]}"),
                                _Patch(facecolor="#4E79A7",
                                       label=f"− → {classes[0]}")]
                    if any(not ic for ic in _is_cluster):
                        legend_b.append(_Patch(facecolor=T()["ax_bg"],
                                               edgecolor=T()["spine"],
                                               label="Bold = behavior group",
                                               linewidth=1))
                    ax4.legend(handles=legend_b, fontsize=8,
                               facecolor=T()["ax_bg"], edgecolor=T()["spine"],
                               labelcolor=T()["text"])
                    fig4.tight_layout(pad=2.5)
                    _embed(fig4, pady=(2, 4))

                else:
                    # Heatmap (multiclass)
                    top_coef = coef_matrix[:, top_idx]   # (n_classes, top_n)
                    fig4_w   = max(9, top_n * 0.60 + 2.5)
                    fig4_h   = max(5, n_classes * 1.2 + 2.5)
                    fig4, ax4 = _plt.subplots(figsize=(fig4_w, fig4_h),
                                               facecolor=T()["fig_bg"])
                    self._style_axes(ax4)
                    vmax = np.abs(top_coef).max() or 1.0
                    im4  = ax4.imshow(top_coef, cmap="RdBu_r",
                                      vmin=-vmax, vmax=vmax, aspect="auto")
                    cb4 = _plt.colorbar(im4, ax=ax4, fraction=0.025, pad=0.02)
                    cb4.set_label("Coefficient", color=T()["text"], fontsize=9)
                    cb4.ax.yaxis.set_tick_params(color=T()["tick"])
                    cb4.ax.tick_params(labelcolor=T()["tick"])
                    # Annotate cells if small enough
                    if top_n <= 15 and n_classes <= 7:
                        for ri in range(n_classes):
                            for ci in range(top_n):
                                val_c = top_coef[ri, ci]
                                ax4.text(ci, ri,
                                         f"{val_c:.2f}",
                                         ha="center", va="center",
                                         fontsize=7,
                                         color="white" if abs(val_c) > vmax * 0.6
                                         else T()["text"])
                    ax4.set_xticks(range(top_n))
                    ax4.set_xticklabels(top_feat_names, rotation=40,
                                        ha="right", fontsize=8,
                                        color=T()["tick"])
                    # Bold group feature labels
                    for lbl, is_cl in zip(ax4.get_xticklabels(), _is_cluster):
                        lbl.set_fontweight("normal" if is_cl else "bold")
                    ax4.set_yticks(range(n_classes))
                    ax4.set_yticklabels(classes, fontsize=9, color=T()["tick"])
                    ax4.set_title(
                        f"{res['name']} — Coefficient Matrix\n"
                        "(rows = groups, columns = top features; "
                        "bold = behavior group)",
                        color=T()["text"], fontsize=11, pad=8)
                    fig4.tight_layout(pad=2.5)
                    fig4.subplots_adjust(
                        bottom=max(0.22, 0.06 * min(top_n, 15) / 15 + 0.12))
                    _embed(fig4, pady=(2, 4))

                # Note for transition model: only shown when source is "custom",
                # since groups/mix now use group-level transition features.
                source_val = self._source_var.get()
                if (res["name"] == "Transition Prob."
                        and source_val == "custom"):
                    ctk.CTkLabel(
                        self._detail_inner,
                        text="ℹ  Transition features use raw cluster"
                             " labels in custom mode.",
                        text_color=T()["subtext"],
                        font=ctk.CTkFont(size=9),
                    ).pack(anchor="w", padx=8, pady=(0, 6))

            # ════════════════════════════════════════════════════════════════
            # Figure 4 — Greedy cluster selection trace (all models)
            # ════════════════════════════════════════════════════════════════
            try:
                trace               = res.get("cluster_selection_trace")
                shapley_result      = res.get("shapley_result")
                shapley_phi_by_size = res.get("shapley_phi_by_size")
                shapley_vcache      = res.get("shapley_vcache") or {}
                top_combos          = res.get("top_combos") or []
                _search_method = res.get("search_method", "greedy")
                _is_exhaustive = (_search_method == "exhaustive"
                                  and len(top_combos) > 0)
                if trace and len(trace) >= 1:
                    n_steps = len(trace)

                    def _cid_label(c):
                        """Return a display label for a trace key (int, str, or tuple)."""
                        if isinstance(c, tuple):
                            return f"T_{c[0]}→{c[1]}"
                        if isinstance(c, int):
                            return f"C{c}"
                        return str(c)   # group name — display as-is

                    # Selection-order data
                    sel_cids   = [t[0] for t in trace]
                    cum_accs   = [t[1] for t in trace]
                    incremental = [cum_accs[0]] + [
                        max(0.0, cum_accs[si] - cum_accs[si - 1])
                        for si in range(1, n_steps)
                    ]
                    chance_level = 1.0 / n_classes

                    # Colour keyed by cluster id / pair (consistent across all panels)
                    cid_to_col = {cid: _GROUP_PALETTE[i % len(_GROUP_PALETTE)]
                                  for i, cid in enumerate(sel_cids)}
                    # Step badge: selection order label for each cid/pair
                    step_badge = {cid: f"#{i + 1}" for i, cid in enumerate(sel_cids)}

                    # Sorted order for left panel in greedy mode (desc incremental gain)
                    sort_idx = sorted(range(n_steps),
                                      key=lambda i: incremental[i], reverse=True)
                    sorted_cids  = [sel_cids[i]      for i in sort_idx]
                    sorted_gains = [incremental[i]   for i in sort_idx]

                    # Figure layout: up to 4 panels when Shapley is available.
                    # Exhaustive mode always gets the Essentiality middle panel even
                    # when n_steps == 1 (greedy peaked at step 1, k=1 exhaustive upgrade).
                    has_shapley     = (shapley_result is not None
                                       and len(shapley_result) == 2)
                    has_phi_by_size = bool(shapley_phi_by_size)

                    # Best combo used by both Figure D (panel 3) and Figure A (panel 4)
                    _best_combo = (frozenset(top_combos[0][0])
                                   if (_is_exhaustive and top_combos)
                                   else frozenset(sel_cids))
                    _best_k = len(_best_combo)

                    _n_left = len(top_combos) if _is_exhaustive else n_steps
                    if _is_exhaustive or n_steps >= 2:
                        if has_shapley and has_phi_by_size:
                            n_panels = 4
                        elif has_shapley:
                            n_panels = 3
                        else:
                            n_panels = 2
                    else:
                        n_panels = 1
                    fig5_w = min(34, max(16, _n_left * 2.5 + 8.0 * n_panels))
                    # Give the left panel (Top-N Combinations) extra width so
                    # its bar labels don't overlap the adjacent panels.
                    _wr = ([1.5] + [1.0] * (n_panels - 1)) if (_is_exhaustive and n_panels > 1) else None
                    fig5, _axes5 = _plt.subplots(
                        1, n_panels, figsize=(fig5_w, 8.0),
                        gridspec_kw=({"width_ratios": _wr} if _wr else {}),
                        facecolor=T()["fig_bg"])
                    if n_panels == 1:
                        _axes5 = [_axes5]
                    ax_sorted = _axes5[0]
                    ax_cum    = _axes5[1] if n_panels >= 2 else None
                    ax_ctx    = _axes5[2] if n_panels >= 3 else None   # Figure D
                    ax_coal   = _axes5[3] if n_panels == 4 else None   # Figure A
                    for _ax in _axes5:
                        self._style_axes(_ax)

                    # ── Left panel ───────────────────────────────────────────────
                    if _is_exhaustive:
                        # Exhaustive: horizontal ranked bars (top-N combos) ──────
                        # When combo_perm_results is present, bars are ranked by
                        # conditional permutation p-value (most specific first) and
                        # each bar carries both accuracy and p-value labels.
                        _cpr = res.get("combo_perm_results") or []
                        _has_pvals = bool(_cpr)
                        if _has_pvals:
                            _combo_accs  = [acc  for _, acc, _   in _cpr]
                            _combo_pvals = [pv   for _, _,   pv  in _cpr]
                            _combo_lbls  = [
                                "{" + ",".join(_cid_label(c)
                                               for c in sorted(combo, key=str)) + "}"
                                for combo, _, _ in _cpr]
                        else:
                            _combo_accs  = [bal  for _, bal in top_combos]
                            _combo_pvals = [None] * len(top_combos)
                            _combo_lbls  = [
                                "{" + ",".join(_cid_label(c)
                                               for c in sorted(combo, key=str)) + "}"
                                for combo, _ in top_combos]
                        # BH FDR correction across combinations
                        _fn_c = [i for i, p in enumerate(_combo_pvals) if p is not None]
                        if len(_fn_c) > 1:
                            _bh_c = benjamini_hochberg(
                                np.array([_combo_pvals[i] for i in _fn_c]))
                            _combo_qvals = list(_combo_pvals)
                            for _i, _q in zip(_fn_c, _bh_c):
                                _combo_qvals[_i] = float(_q)
                        else:
                            _combo_qvals = list(_combo_pvals)
                        _n_shown     = len(_combo_lbls)
                        # y positions: best combo at top (highest y index)
                        _ypos = list(range(_n_shown - 1, -1, -1))
                        # Use max accuracy for xlim so re-ranking by p-value never clips bars
                        _best_acc_ex = max(_combo_accs) if _combo_accs else 0.0
                        for _ri, (_lbl, _acc, _qv, _yp) in enumerate(
                                zip(_combo_lbls, _combo_accs,
                                    _combo_qvals, _ypos)):
                            _col   = _GROUP_PALETTE[_ri % len(_GROUP_PALETTE)]
                            _alpha = 1.0 if _ri == 0 else max(0.45, 1.0 - _ri * 0.12)
                            ax_sorted.barh(_yp, _acc, color=_col, alpha=_alpha,
                                           edgecolor=T()["spine"], linewidth=0.7,
                                           zorder=3)
                            # Accuracy + q-value label beyond bar end
                            # q-value on second line keeps label narrow so it
                            # does not overlap the adjacent panels.
                            _pv_str = (f"\nq(c)={_qv:.3f}" if _qv is not None else "")
                            ax_sorted.text(_acc + 0.005, _yp,
                                           f"{_acc:.1%}{_pv_str}",
                                           va="center", ha="left", fontsize=7.5,
                                           color=T()["text"], fontweight="bold",
                                           zorder=5)
                            # Rank badge inside bar
                            ax_sorted.text(0.005, _yp, f"#{_ri + 1}",
                                           va="center", ha="left", fontsize=7,
                                           color="white", fontweight="bold",
                                           zorder=6)
                        ax_sorted.axvline(chance_level, color=T()["subtext"],
                                          linestyle="--", linewidth=1.2,
                                          label=f"Chance ({chance_level:.0%})",
                                          zorder=2)
                        ax_sorted.set_yticks(_ypos)
                        ax_sorted.set_yticklabels(
                            _combo_lbls, color=T()["tick"], fontsize=8)
                        ax_sorted.set_xlabel("Balanced LOO Accuracy",
                                             color=T()["text"], fontsize=10)
                        _k_shown = (len(_cpr[0][0]) if _has_pvals and _cpr
                                    else (len(top_combos[0][0]) if top_combos else 0))
                        _title_suffix = (" — ranked by specificity"
                                         if _has_pvals else "")
                        ax_sorted.set_title(
                            f"Top-{_n_shown} Combinations  (k={_k_shown})"
                            f"{_title_suffix}",
                            color=T()["text"], fontsize=10, pad=8)
                        ax_sorted.set_xlim(0, min(1.0, _best_acc_ex + 0.22))
                        ax_sorted.xaxis.set_major_formatter(
                            _plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
                        ax_sorted.legend(fontsize=8, facecolor=T()["ax_bg"],
                                         edgecolor=T()["spine"],
                                         labelcolor=T()["text"], loc="lower right")
                        ax_sorted.xaxis.grid(True, color=T()["spine"],
                                             alpha=0.35, zorder=0)
                    else:
                        # Greedy: sorted contribution bars ────────────────────────
                        if n_panels == 1:
                            # Only one contributor after pruning — text message only
                            fig5.set_size_inches(10, 3)
                            ax_sorted.axis("off")
                            _s_lbl = _cid_label(sorted_cids[0]) if sorted_cids else "?"
                            _s_acc = f"{cum_accs[0]:.1%}" if cum_accs else "?"
                            _uw = "pair" if res.get("feature_type") == "transition" else "cluster"
                            ax_sorted.text(0.5, 0.65,
                                           f"Single {_uw} identified: {_s_lbl}",
                                           ha="center", va="center", fontsize=14,
                                           color=T()["text"], fontweight="bold",
                                           transform=ax_sorted.transAxes)
                            ax_sorted.text(0.5, 0.46,
                                           f"LOO balanced accuracy: {_s_acc}",
                                           ha="center", va="center", fontsize=11,
                                           color=T()["text"],
                                           transform=ax_sorted.transAxes)
                            ax_sorted.text(0.5, 0.26,
                                           "Essentiality, Shapley φ and Synergy panels"
                                           " require ≥ 2 contributors after pruning.",
                                           ha="center", va="center", fontsize=9,
                                           color=T()["subtext"],
                                           transform=ax_sorted.transAxes)
                        else:
                            # n_steps >= 2: sorted contribution bars
                            _inc_top = max(sorted_gains) if sorted_gains else 1
                            _inc_top = max(_inc_top, 0.01)
                            for di, (cid, gain) in enumerate(
                                    zip(sorted_cids, sorted_gains)):
                                col   = cid_to_col[cid]
                                alpha = 1.0 if gain > 0.001 else 0.38
                                ax_sorted.bar(di, gain, color=col, alpha=alpha,
                                              edgecolor=T()["spine"], linewidth=0.7,
                                              zorder=3)
                                if gain > 0.001:
                                    ax_sorted.text(
                                        di, gain + _inc_top * 0.06, f"+{gain:.0%}",
                                        ha="center", fontsize=8.5,
                                        color=T()["text"], fontweight="bold", zorder=5)
                                if gain > _inc_top * 0.25:
                                    badge_y  = gain / 2
                                    badge_col = "white"
                                    badge_va  = "center"
                                else:
                                    badge_y  = gain + _inc_top * 0.24
                                    badge_col = T()["subtext"]
                                    badge_va  = "bottom"
                                ax_sorted.text(
                                    di, badge_y, step_badge[cid],
                                    ha="center", va=badge_va, fontsize=7,
                                    color=badge_col, fontweight="bold", zorder=6)
                            ax_sorted.axhline(chance_level, color=T()["subtext"],
                                              linestyle="--", linewidth=1.2,
                                              label=f"Chance ({chance_level:.0%})",
                                              zorder=2)
                            ax_sorted.set_xticks(range(n_steps))
                            ax_sorted.set_xticklabels(
                                [f"{_cid_label(c)}\n{cum_accs[sel_cids.index(c)]:.0%}"
                                 for c in sorted_cids],
                                color=T()["tick"], fontsize=8, rotation=0, ha="center")
                            ax_sorted.set_ylabel("LOO Accuracy Gain (sorted)",
                                                 color=T()["text"], fontsize=10)
                            ax_sorted.set_title(
                                "Greedy Contribution  (sorted by gain)",
                                color=T()["text"], fontsize=10, pad=8)
                            ax_sorted.set_ylim(bottom=0, top=_inc_top * 1.85)
                            ax_sorted.yaxis.set_major_formatter(
                                _plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
                            ax_sorted.legend(fontsize=8, facecolor=T()["ax_bg"],
                                             edgecolor=T()["spine"],
                                             labelcolor=T()["text"], loc="upper right")
                            ax_sorted.yaxis.grid(True, color=T()["spine"],
                                                 alpha=0.35, zorder=0)

                    # ── Middle panel ──────────────────────────────────────────
                    if ax_cum is not None:
                        if _is_exhaustive:
                            # Exhaustive: cluster essentiality bars ──────────────
                            import collections as _collections
                            _counter = _collections.Counter(
                                cid for combo, _ in top_combos for cid in combo)
                            _best_set = set(top_combos[0][0]) if top_combos else set()
                            _ess_cids = sorted(_counter, key=lambda c: -_counter[c])
                            _ess_fracs = [_counter[c] / max(len(top_combos), 1)
                                          for c in _ess_cids]
                            for _ei, (_ec, _ef) in enumerate(
                                    zip(_ess_cids, _ess_fracs)):
                                _col = (cid_to_col.get(_ec, T()["subtext"])
                                        if _ec in _best_set else T()["subtext"])
                                _alpha = 1.0 if _ec in _best_set else 0.55
                                ax_cum.bar(_ei, _ef, color=_col, alpha=_alpha,
                                           edgecolor=T()["spine"], linewidth=0.7,
                                           zorder=3)
                                if _ec in _best_set:
                                    ax_cum.text(
                                        _ei, _ef + 0.03, "★",
                                        ha="center", fontsize=11,
                                        color=T()["text"], zorder=5)
                            ax_cum.axhline(1.0, color=T()["subtext"],
                                           linestyle="--", linewidth=1.2,
                                           label="In all top combos", zorder=2)
                            ax_cum.set_xticks(range(len(_ess_cids)))
                            ax_cum.set_xticklabels(
                                [_cid_label(c) for c in _ess_cids],
                                color=T()["tick"], fontsize=9)
                            ax_cum.set_ylabel("Fraction of Top Combos",
                                              color=T()["text"], fontsize=10)
                            _ess_unit = ("Pair" if res.get("feature_type") == "transition"
                                         else "Cluster")
                            ax_cum.set_title(
                                f"{_ess_unit} Essentiality  (★ = in best combo)",
                                color=T()["text"], fontsize=10, pad=8)
                            ax_cum.set_ylim(0, 1.28)
                            ax_cum.yaxis.set_major_formatter(
                                _plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
                            ax_cum.legend(fontsize=8, facecolor=T()["ax_bg"],
                                          edgecolor=T()["spine"],
                                          labelcolor=T()["text"])
                            ax_cum.yaxis.grid(True, color=T()["spine"],
                                              alpha=0.35, zorder=0)
                        else:
                            # Greedy: cumulative accuracy line (existing) ─────────
                            x_pts = list(range(n_steps))
                            ax_cum.plot(x_pts, cum_accs, marker="o",
                                        linewidth=2.5, color="#4E79A7",
                                        markersize=9, zorder=4)
                            ax_cum.fill_between(x_pts, chance_level, cum_accs,
                                                alpha=0.12, color="#4E79A7", zorder=2)
                            ax_cum.axhline(chance_level, color=T()["subtext"],
                                           linestyle="--", linewidth=1.2,
                                           label=f"Chance ({chance_level:.0%})",
                                           zorder=2)
                            _y_ceil = max(max(cum_accs) + 0.22, 1.10)
                            for si, (cid, acc) in enumerate(
                                    zip(sel_cids, cum_accs)):
                                _above = (_y_ceil - acc) >= 0.16
                                _dy    = 18 if _above else -30
                                _va    = "bottom" if _above else "top"
                                ax_cum.annotate(
                                    f"{_cid_label(cid)}\n{acc:.0%}",
                                    (si, acc),
                                    textcoords="offset points",
                                    xytext=(0, _dy),
                                    ha="center", va=_va, fontsize=8,
                                    color=cid_to_col[cid], fontweight="bold",
                                    arrowprops=dict(
                                        arrowstyle="-",
                                        color=cid_to_col[cid],
                                        lw=0.7, shrinkA=0, shrinkB=4))
                            ax_cum.set_xticks(x_pts)
                            ax_cum.set_xticklabels(
                                [_cid_label(c) for c in sel_cids],
                                color=T()["tick"], fontsize=11)
                            ax_cum.set_ylabel("Cumulative Balanced Accuracy",
                                              color=T()["text"], fontsize=10)
                            _cum_xlabel = (
                                "Transition added (greedy selection order)"
                                if res.get("feature_type") == "transition"
                                else ("Feature added (greedy selection order)"
                                      if any(isinstance(c, str)
                                             for c in sel_cids)
                                      else "Cluster added (greedy selection order)"))
                            ax_cum.set_xlabel(_cum_xlabel,
                                              color=T()["text"], fontsize=10)
                            ax_cum.set_title(
                                "Cumulative Balanced Accuracy — Greedy Steps",
                                color=T()["text"], fontsize=10, pad=8)
                            ax_cum.set_ylim(0, _y_ceil)
                            ax_cum.yaxis.set_major_formatter(
                                _plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
                            ax_cum.legend(fontsize=8, facecolor=T()["ax_bg"],
                                          edgecolor=T()["spine"],
                                          labelcolor=T()["text"])
                            ax_cum.yaxis.grid(True, color=T()["spine"],
                                              alpha=0.35, zorder=0)

                    # ── Panel 3: Figure D — Overall φ vs in-best-k (dual bar) ──
                    if ax_ctx is not None and has_shapley:
                        try:
                            shap_baseline, shap_ranked = shapley_result
                            _n_exact  = len(trace) <= 8
                            _meth_tag = "exact" if _n_exact else "≈ MC"
                            _phi_map  = dict(shap_ranked)
                            _best_v   = shapley_vcache.get(_best_combo)
                            _ctx_map  = {}
                            if _best_v is not None:
                                for _cid in _best_combo:
                                    _wv = shapley_vcache.get(_best_combo - {_cid})
                                    if _wv is not None:
                                        _ctx_map[_cid] = _best_v - _wv
                            _d_cids   = [cid for cid, _ in shap_ranked]
                            _x_d      = np.arange(len(_d_cids))
                            _w_d      = 0.35
                            _phi_vals = [_phi_map.get(c, 0.0) for c in _d_cids]
                            _ctx_vals = [_ctx_map.get(c, np.nan) for c in _d_cids]
                            _cols_d   = [cid_to_col.get(c, _GROUP_PALETTE[0])
                                         for c in _d_cids]
                            ax_ctx.bar(_x_d - _w_d / 2, _phi_vals, _w_d,
                                       color=_cols_d, alpha=0.72,
                                       label="Overall Shapley φ", zorder=3)
                            ax_ctx.bar(_x_d + _w_d / 2, _ctx_vals, _w_d,
                                       color=_cols_d, alpha=1.0,
                                       edgecolor=T()["text"], linewidth=1.2,
                                       label=f"In-best-k  (k={_best_k})", zorder=3)
                            ax_ctx.axhline(0, color=T()["subtext"],
                                           linestyle="--", linewidth=1.0, zorder=2)
                            ax_ctx.set_xticks(_x_d)
                            ax_ctx.set_xticklabels(
                                [_cid_label(c) for c in _d_cids],
                                color=T()["tick"], fontsize=9)
                            ax_ctx.yaxis.set_major_formatter(
                                _plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
                            ax_ctx.set_ylabel("Contribution",
                                              color=T()["text"], fontsize=10)
                            ax_ctx.yaxis.grid(True, color=T()["spine"],
                                              alpha=0.35, zorder=0)
                            ax_ctx.legend(fontsize=8, facecolor=T()["ax_bg"],
                                          edgecolor=T()["spine"],
                                          labelcolor=T()["text"])
                            ax_ctx.set_title(
                                f"Shapley φ  [{_meth_tag}]  ·  overall (left) vs in-best-k (right)",
                                color=T()["text"], fontsize=10, pad=8)
                            _shap_sum = sum(_phi_vals)
                            _ok = ("✓" if abs(_shap_sum - (shap_baseline
                                                            - chance_level)) < 0.03
                                   else "≈")
                            ax_ctx.text(
                                0.5, -0.17,
                                f"Σφ = {_shap_sum:.1%}  |  "
                                f"baseline = {shap_baseline:.0%}  |  "
                                f"chance = {chance_level:.0%}  |  "
                                f"Σφ ≈ baseline − chance: {_ok}",
                                ha="center", transform=ax_ctx.transAxes,
                                fontsize=7, color=T()["subtext"])
                        except Exception as _exc_ctx:
                            ax_ctx.text(0.5, 0.5,
                                        f"Context panel unavailable\n({_exc_ctx})",
                                        ha="center", va="center",
                                        transform=ax_ctx.transAxes,
                                        color=T()["subtext"], fontsize=9)

                    # ── Panel 4: Figure A — Synergy Profile (top 5 by |φ|) ────
                    if ax_coal is not None and has_phi_by_size:
                        try:
                            _N_coal  = n_steps
                            _sizes   = list(range(_N_coal))
                            _km1     = _best_k - 1
                            _phi_map2 = dict(shapley_result[1])
                            _top5 = sorted(
                                shapley_phi_by_size.keys(),
                                key=lambda c: abs(_phi_map2.get(c, 0.0)),
                                reverse=True)[:5]
                            for cid in _top5:
                                _col = cid_to_col.get(cid, _GROUP_PALETTE[0])
                                _ys  = [shapley_phi_by_size[cid].get(s, 0.0)
                                        for s in _sizes]
                                ax_coal.plot(_sizes, _ys, marker="o",
                                             markersize=5, linewidth=1.8,
                                             color=_col, zorder=3)
                                ax_coal.text(
                                    _sizes[-1] + 0.1, _ys[-1],
                                    _cid_label(cid),
                                    va="center", ha="left", fontsize=8,
                                    color=_col, fontweight="bold")
                                for _si in range(1, len(_ys)):
                                    if _ys[_si - 1] < -0.005 and _ys[_si] > 0.005:
                                        ax_coal.annotate(
                                            "synergistic\nspecialist",
                                            xy=((_si - 1 + _si) / 2.0, 0.0),
                                            xytext=(6, 14),
                                            textcoords="offset points",
                                            fontsize=6.5, fontstyle="italic",
                                            color=_col, alpha=0.9,
                                            arrowprops=dict(arrowstyle="-",
                                                            color=_col, lw=0.7))
                                        break
                            ax_coal.axhline(0, color=T()["subtext"],
                                            linestyle="--", linewidth=1.2,
                                            zorder=2)
                            if 0 <= _km1 < _N_coal:
                                ax_coal.axvline(
                                    _km1, color=T()["subtext"],
                                    linestyle=":", linewidth=1.5, zorder=2,
                                    label=f"best-k context  (k={_best_k})")
                                ax_coal.text(
                                    _km1 + 0.06, 0.96, f"k={_best_k}",
                                    color=T()["subtext"], fontsize=7,
                                    va="top", ha="left", rotation=90,
                                    transform=ax_coal.get_xaxis_transform())
                            ax_coal.set_xticks(_sizes)
                            ax_coal.set_xticklabels(
                                [str(s) for s in _sizes],
                                color=T()["tick"], fontsize=9)
                            ax_coal.set_xlabel(
                                "Coalition size when joining (s)",
                                color=T()["text"], fontsize=10)
                            ax_coal.set_ylabel(
                                "Avg marginal contribution",
                                color=T()["text"], fontsize=10)
                            ax_coal.yaxis.set_major_formatter(
                                _plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
                            ax_coal.yaxis.grid(True, color=T()["spine"],
                                               alpha=0.35, zorder=0)
                            ax_coal.legend(fontsize=7, facecolor=T()["ax_bg"],
                                           edgecolor=T()["spine"],
                                           labelcolor=T()["text"],
                                           loc="upper left")
                            _meth_s = "exact" if _N_coal <= 8 else "≈ MC"
                            ax_coal.set_title(
                                f"Synergy Profile  [{_meth_s}]",
                                color=T()["text"], fontsize=10, pad=8)
                            ax_coal.text(
                                0.5, -0.14,
                                "avg marginal contribution by coalition size"
                                "  (top 5 by |φ|)",
                                ha="center", transform=ax_coal.transAxes,
                                fontsize=7, color=T()["subtext"])
                            ax_coal.set_xlim(
                                -0.3,
                                _N_coal - 0.5 + max(1.5, _N_coal * 0.18))
                        except Exception as _exc_coal:
                            ax_coal.text(
                                0.5, 0.5,
                                f"Synergy Profile unavailable\n({_exc_coal})",
                                ha="center", va="center",
                                transform=ax_coal.transAxes,
                                color=T()["subtext"], fontsize=9)

                    _trace_custom = res.get("trace_is_custom", False)
                    _trace_title = (
                        "Custom Mode — Per-Cluster Contribution (selected clusters)"
                        if _trace_custom else
                        "Per-Cluster Discrimination Power (all clusters)"
                    )
                    _method_tag = (
                        "[Exhaustive]" if _search_method == "exhaustive"
                        else "[Greedy]")
                    fig5.suptitle(
                        f"{_trace_title}  —  {res['name']}  {_method_tag}",
                        color=T()["text"], fontsize=10, y=0.995)
                    _bot5 = 0.22 if (ax_ctx is not None or ax_coal is not None) else 0.11
                    fig5.subplots_adjust(bottom=_bot5, top=0.87,
                                         wspace=0.70, left=0.06, right=0.96)
                    _embed(fig5, pady=(2, 4))
            except Exception as _exc5:
                _embed_error(f"Cluster selection trace: {_exc5}")

            # ════════════════════════════════════════════════════════════════
            # Figure 5 — Per-animal LOO probability strips  (last: granular detail)
            # ════════════════════════════════════════════════════════════════
            if has_proba:
                n_animals = len(y_enc)
                # Sort: by true group, then by descending P(true group)
                p_true = pred_proba[np.arange(n_animals), y_enc]
                order  = np.lexsort((-p_true, y_enc))
                fig2_h = max(4.5, n_animals * 0.60 + 1.8)
                fig2, ax2 = _plt.subplots(figsize=(12, fig2_h),
                                           facecolor=T()["fig_bg"],
                                           constrained_layout=True)
                self._style_axes(ax2)
                for rank, ai in enumerate(order):
                    left  = 0.0
                    probs = pred_proba[ai]
                    tg    = y_enc[ai]
                    pg    = pred_labels[ai]
                    correct = (tg == pg)
                    border_col = "#4CAF50" if correct else "#F44336"
                    for ci, cls in enumerate(classes):
                        p_seg = probs[ci]
                        col   = grp_colors[cls]
                        ax2.barh(rank, p_seg, left=left,
                                 color=col, edgecolor="none",
                                 height=0.75, zorder=3)
                        left += p_seg
                    # Draw border rect
                    ax2.barh(rank, 1.0, left=0,
                             color="none", edgecolor=border_col,
                             linewidth=2.2, height=0.75, zorder=4)
                    # Labels
                    true_name  = classes[tg]
                    pred_name  = classes[pg]
                    glyph      = "✓" if correct else "✗"
                    ax2.text(-0.02, rank, true_name,
                             ha="right", va="center", fontsize=8,
                             color=T()["text"])
                    ax2.text(1.02, rank, f"{pred_name} {glyph}",
                             ha="left",  va="center", fontsize=8,
                             color="#4CAF50" if correct else "#F44336")

                ax2.set_xlim(-0.28, 1.40)
                ax2.set_ylim(-0.6, n_animals - 0.4)
                ax2.set_yticks([])
                ax2.set_xlabel("Predicted Probability", color=T()["text"],
                               fontsize=10)
                ax2.set_title(f"{res['name']} — Per-Animal LOO Predictions",
                              color=T()["text"], fontsize=11, pad=8)
                ax2.text(0.5, 1.01,
                         f"Balanced Acc: {bal_acc:.0%}  ·  κ = {kappa:.2f}",
                         ha="center", va="bottom",
                         transform=ax2.transAxes,
                         fontsize=9, color=T()["subtext"])
                # Legend
                legend_els = [_Patch(facecolor=grp_colors[cls], label=cls)
                              for cls in classes]
                legend_els += [
                    _Patch(facecolor="none", edgecolor="#4CAF50",
                           linewidth=2, label="Correct border"),
                    _Patch(facecolor="none", edgecolor="#F44336",
                           linewidth=2, label="Wrong border"),
                ]
                ax2.legend(handles=legend_els, loc="lower right", fontsize=8,
                           facecolor=T()["ax_bg"], edgecolor=T()["spine"],
                           labelcolor=T()["text"], ncol=min(4, len(legend_els)))
                _embed(fig2, pady=(2, 4))

        except Exception as exc:
            # Append error rather than replacing the whole view — earlier
            # figures that already rendered should remain visible.
            try:
                ctk.CTkLabel(
                    self._detail_inner,
                    text=f"⚠ Plot error: {exc}",
                    text_color=T()["muted"],
                    font=ctk.CTkFont(size=10),
                    justify="left").pack(anchor="w", padx=12, pady=4)
            except Exception:
                pass

    # ── Export CSV ────────────────────────────────────────────────────────────

    def export_csv_data(self, out_dir: pathlib.Path) -> int:
        """Write all predictor result CSVs to out_dir. Returns number of files written."""
        valid = [r for r in self._results
                 if not r.get("error") and r.get("acc") is not None]
        if not valid:
            return 0
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0

        # 1. Model summary
        # BH FDR correction across models (mirrors display in graphs)
        _sum_pvals  = [r["pval"] for r in valid]
        _sum_fn_idx = [i for i, p in enumerate(_sum_pvals) if p is not None]
        if len(_sum_fn_idx) > 1:
            _sum_bh = benjamini_hochberg(np.array([_sum_pvals[i] for i in _sum_fn_idx]))
            _sum_qvals = list(_sum_pvals)
            for _i, _q in zip(_sum_fn_idx, _sum_bh):
                _sum_qvals[_i] = float(_q)
        else:
            _sum_qvals = list(_sum_pvals)

        summary_rows = []
        for res, qv in zip(valid, _sum_qvals):
            le        = res["le"]
            n_classes = len(le.classes_)
            n_animals = len(res["y_enc"])
            summary_rows.append({
                "model":             res["name"],
                "loo_accuracy":      round(float(res["acc"]), 4),
                "balanced_accuracy": round(float(res.get("bal_acc", res["acc"])), 4),
                "cohen_kappa":       round(float(res["kappa"]), 4)
                                     if res["kappa"] is not None else None,
                "pval":              round(float(res["pval"]), 4)
                                     if res["pval"] is not None else None,
                "qval_bh":           round(float(qv), 4)
                                     if qv is not None else None,
                "pval_type":         res.get("pval_type", "cond"),
                "n_animals":         n_animals,
                "n_classes":         n_classes,
                "chance_level":      round(1.0 / n_classes, 4),
            })
        pd.DataFrame(summary_rows).to_csv(out_dir / "model_summary.csv", index=False)
        n += 1

        # 2. Per-animal predictions with class probabilities
        pred_rows = []
        for res in valid:
            le           = res["le"]
            classes      = le.classes_
            y_enc        = res["y_enc"]
            pred_labels  = res["pred_labels"]
            pred_proba   = res.get("pred_proba")
            animal_names = res.get("animal_names") or [
                f"Animal {i}" for i in range(len(y_enc))]
            for i, name in enumerate(animal_names):
                row = {
                    "model":           res["name"],
                    "animal":          name,
                    "true_group":      le.inverse_transform([y_enc[i]])[0],
                    "predicted_group": le.inverse_transform([pred_labels[i]])[0],
                    "correct":         int(y_enc[i] == pred_labels[i]),
                }
                if pred_proba is not None:
                    for ci, cls in enumerate(classes):
                        row[f"P({cls})"] = round(float(pred_proba[i, ci]), 4)
                pred_rows.append(row)
        pd.DataFrame(pred_rows).to_csv(out_dir / "predictions.csv", index=False)
        n += 1

        # 3. Confusion matrix (long format)
        cm_rows = []
        for res in valid:
            le      = res["le"]
            classes = le.classes_
            cm      = res.get("cm")
            if cm is None:
                continue
            for i, true_cls in enumerate(classes):
                row_total = int(cm[i].sum())
                for j, pred_cls in enumerate(classes):
                    count = int(cm[i, j])
                    cm_rows.append({
                        "model":           res["name"],
                        "true_group":      true_cls,
                        "predicted_group": pred_cls,
                        "count":           count,
                        "recall":          round(count / row_total, 4)
                                           if row_total > 0 else 0.0,
                    })
        if cm_rows:
            pd.DataFrame(cm_rows).to_csv(out_dir / "confusion_matrix.csv", index=False)
            n += 1

        # 4. Feature importances + coefficients
        imp_rows = []
        for res in valid:
            feat_names  = (res.get("display_feat_names") or res.get("feat_names") or [])
            importances = res.get("importances")
            coef_matrix = res.get("coef_matrix")
            le          = res["le"]
            classes     = le.classes_
            if importances is None or len(importances) == 0:
                continue
            n_coef_rows = coef_matrix.shape[0] if coef_matrix is not None else 0
            for fi, fname in enumerate(feat_names):
                if fi >= len(importances):
                    break
                row = {
                    "model":      res["name"],
                    "feature":    fname,
                    "importance": round(float(importances[fi]), 6),
                }
                if coef_matrix is not None and fi < coef_matrix.shape[1]:
                    for ci, cls in enumerate(classes):
                        if ci < n_coef_rows:
                            row[f"coef_{cls}"] = round(float(coef_matrix[ci, fi]), 6)
                imp_rows.append(row)
        if imp_rows:
            (pd.DataFrame(imp_rows)
               .sort_values(["model", "importance"], ascending=[True, False])
               .to_csv(out_dir / "feature_importances.csv", index=False))
            n += 1

        # 5. Permutation null distribution
        perm_rows = []
        for res in valid:
            perm_scores = res.get("perm_scores")
            if perm_scores is None or len(perm_scores) == 0:
                continue
            for score in perm_scores:
                perm_rows.append({
                    "model":             res["name"],
                    "null_balanced_acc": round(float(score), 6),
                })
        if perm_rows:
            pd.DataFrame(perm_rows).to_csv(out_dir / "perm_null_scores.csv", index=False)
            n += 1

        return n

    def _export_csv(self):
        if not self._results:
            messagebox.showinfo("Nothing to export",
                                "Run models first.", parent=self)
            return
        valid = [r for r in self._results
                 if not r.get("error") and r.get("acc") is not None]
        if not valid:
            messagebox.showwarning("Export",
                                   "No completed models to export.",
                                   parent=self)
            return
        folder = filedialog.askdirectory(
            parent=self,
            title="Export Group Predictor Results — Choose Output Folder",
        )
        if not folder:
            return
        try:
            n = self.export_csv_data(pathlib.Path(folder))
            messagebox.showinfo(
                "Exported",
                f"Saved {n} file(s) to:\n{folder}",
                parent=self,
            )
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc), parent=self)

    # ── Public reset (called when animals are reloaded) ───────────────────────

    def reset(self):
        import matplotlib.pyplot as _plt
        for fig in self._open_figs:
            self._close_fig(fig)
        self._open_figs = []
        if self._overview_fig is not None:
            self._close_fig(self._overview_fig)
            self._overview_fig = None
        if self._null_comp_fig is not None:
            self._close_fig(self._null_comp_fig)
            self._null_comp_fig = None
        for w in self._overview_fr.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        self._results = []
        self._model_radio_btns = []   # widgets will be destroyed by _show_detail_placeholder
        if self._nested_btn:
            self._nested_btn.configure(state="disabled",
                                       text="Run Nested Permutation Test")
        if self._perm_toggle_frame:
            for _w in self._perm_toggle_frame.winfo_children():
                try:
                    _w.destroy()
                except Exception:
                    pass
            self._perm_toggle_frame.pack_forget()
        self._show_detail_placeholder(
            "Animals reloaded — click   Run Models to recompute.")
        self._caveat_fr.grid_remove()
        self._status("")


#
# BEHAVIORAL EXPLORER TAB PANEL
#

class BehavioralExplorerPanel(ctk.CTkFrame):
    """
    Scrollable 'Behavioral Explorer' tab for group-level comparisons.

    Plot views
    ----------
    Diff Heatmap     — Δ transition matrices (exp − ctrl)
    Dwell Violin     — Dwell-time distribution per state × group
    Sankey           — Alluvial sequence flow per group
    Group Transitions — Per-group transition probability heatmaps
    Group Networks   — Per-group circular transition networks

    Requires
    --------
    get_animals_fn()  → list  of animal dicts (same format as CombinedAnalysis)
    get_recluster_fn() → dict | None   recluster_result from UnbiasedAnalytics
    """

    _VIEWS = [
        "Diff Heatmap",
        "Dwell Violin",
        "Sankey",
        "Group Transitions",
        "Group Networks",
        "Energy Landscape",
    ]

    def __init__(self, parent,
                 get_animals_fn,
                 get_recluster_fn=None,
                 get_groups_fn=None,
                 get_umap_fn=None,
                 **kw):
        super().__init__(parent, fg_color=T()["panel"], **kw)
        self._get_animals   = get_animals_fn
        self._get_recluster = get_recluster_fn or (lambda: None)
        self._get_groups_fn = get_groups_fn
        self._get_umap_fn   = get_umap_fn
        self._current_fig   = None
        self._current_figs  = []
        self._current_mpl   = None

        self.columnconfigure(0, weight=0, minsize=256)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_controls()
        self._build_plot_area()

    # ── Controls (left panel) ─────────────────────────────────────────────────

    def _build_controls(self):
        ctrl = ctk.CTkScrollableFrame(
            self, fg_color=T()["bg"], corner_radius=0, width=254)
        ctrl.grid(row=0, column=0, sticky="nsew", padx=(6, 2), pady=6)
        ctrl.columnconfigure(0, weight=1)

        def section(title):
            fr = ctk.CTkFrame(ctrl, fg_color=T()["card"], corner_radius=8)
            fr.pack(fill="x", padx=4, pady=4)
            ctk.CTkLabel(fr, text=title,
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=T()["hdr_text"],
                         ).pack(anchor="w", padx=10, pady=(8, 2))
            return fr

        # Group-by label selector
        gb = section("Group By Label")
        ctk.CTkLabel(gb, text="Treat this label as experimental group:",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._group_by_var = ctk.StringVar(value="Label 1")
        ctk.CTkOptionMenu(gb, variable=self._group_by_var,
                          values=["Label 1", "Label 2", "Label 3", "All Labels"],
                          width=220).pack(padx=12, pady=(2, 8))

        # Diff heatmap settings
        dh = section("Difference Heatmap")
        ctk.CTkLabel(dh, text="Reference / Control group:",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._ctrl_var = ctk.StringVar(value="(auto — first group)")
        self._ctrl_entry = ctk.CTkEntry(dh, textvariable=self._ctrl_var, width=220)
        self._ctrl_entry.pack(padx=12, pady=(2, 8))

        ctk.CTkLabel(dh,
                     text="Leave blank to use the first EG as control.\n"
                          "Type the exact group name to pin a specific group.",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=10),
                     justify="left").pack(anchor="w", padx=12, pady=(0, 8))

        # Sankey settings
        sk = section("Sankey  (Sequence Flow)")
        ctk.CTkLabel(sk, text="Bout positions to show:",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._nsteps_var = ctk.StringVar(value="5")
        self._nsteps_slider = ctk.CTkSlider(
            sk, from_=2, to=10, number_of_steps=8,
            command=lambda v: self._nsteps_var.set(str(int(round(v)))),
            width=220)
        self._nsteps_slider.set(5)
        self._nsteps_slider.pack(padx=12, pady=(2, 0))
        self._nsteps_lbl = ctk.CTkLabel(
            sk, textvariable=self._nsteps_var,
            text_color=T()["subtext"], font=ctk.CTkFont(size=10))
        self._nsteps_lbl.pack(anchor="w", padx=12, pady=(0, 6))

        ctk.CTkLabel(sk, text="Anchor cluster  (blank = auto):",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._anchor_cluster_entry = ctk.CTkEntry(
            sk, width=120, placeholder_text="e.g. C6 or 6")
        self._anchor_cluster_entry.pack(anchor="w", padx=12, pady=(2, 4))
        self._anchor_cluster_entry.bind("<Return>",   self._generate)
        self._anchor_cluster_entry.bind("<FocusOut>", self._generate)
        ctk.CTkLabel(sk,
                     text="Leave blank: auto-selects the source cluster of\n"
                          "the strongest normalised transition in your data.",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=10),
                     justify="left").pack(anchor="w", padx=12, pady=(0, 4))

        ctk.CTkLabel(sk, text="Anchor beh. group  (blank = auto):",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._anchor_group_entry = ctk.CTkEntry(
            sk, width=180, placeholder_text="e.g. Locomotion")
        self._anchor_group_entry.pack(anchor="w", padx=12, pady=(2, 4))
        self._anchor_group_entry.bind("<Return>",   self._generate)
        self._anchor_group_entry.bind("<FocusOut>", self._generate)
        ctk.CTkLabel(sk,
                     text="Exact group name for the Beh-Group Sankey.\n"
                          "Leave blank for the same auto-detect logic.",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=10),
                     justify="left").pack(anchor="w", padx=12, pady=(0, 8))

        # Biological Relevance Filter
        bf = section("Biological Relevance Filter")
        ctk.CTkLabel(bf,
                     text="Remove clusters too brief or rare to be\n"
                          "biologically meaningful. Applied universally\n"
                          "to all plots. Leave blank to disable.",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=10),
                     justify="left").pack(anchor="w", padx=12, pady=(2, 6))

        ctk.CTkLabel(bf, text="Min mean bout duration (ms):",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._brf_min_mean_ms_var = ctk.StringVar(value="")
        ctk.CTkEntry(bf, textvariable=self._brf_min_mean_ms_var,
                     width=120, placeholder_text="e.g. 200"
                     ).pack(anchor="w", padx=12, pady=(2, 4))

        ctk.CTkLabel(bf, text="Min total duration (s, per animal):",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._brf_min_total_s_var = ctk.StringVar(value="")
        ctk.CTkEntry(bf, textvariable=self._brf_min_total_s_var,
                     width=120, placeholder_text="e.g. 2"
                     ).pack(anchor="w", padx=12, pady=(2, 4))

        ctk.CTkLabel(bf, text="Min frequency (bouts per animal):",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._brf_min_freq_var = ctk.StringVar(value="")
        ctk.CTkEntry(bf, textvariable=self._brf_min_freq_var,
                     width=120, placeholder_text="e.g. 5"
                     ).pack(anchor="w", padx=12, pady=(2, 8))

        # Run button
        ctk.CTkButton(
            ctrl, text="   Generate Plot",
            command=self._generate,
            fg_color=T()["btn_unbiased"],
            height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(fill="x", padx=8, pady=6)

        # Save buttons
        ctk.CTkButton(
            ctrl, text="Save: Save Graph",
            command=self._save_graph,
            fg_color=T()["btn_save"],
        ).pack(fill="x", padx=8, pady=(2, 2))

        ctk.CTkButton(
            ctrl, text="Save: Export CSV",
            command=self._export_csv,
            fg_color=T()["btn_save"],
        ).pack(fill="x", padx=8, pady=(0, 6))

        # Status label
        self._status_lbl = ctk.CTkLabel(
            ctrl, text="",
            text_color=T()["subtext"],
            wraplength=240, justify="left",
            font=ctk.CTkFont(size=11))
        self._status_lbl.pack(anchor="w", padx=12, pady=4)

    def _group_by_key(self) -> str:
        return AnimalListPanel._label_key_to_str(self._group_by_var.get())

    # ── Plot area (right panel) ───────────────────────────────────────────────

    def _build_plot_area(self):
        right = ctk.CTkFrame(self, fg_color=T()["panel"])
        right.grid(row=0, column=1, sticky="nsew", padx=(2, 6), pady=6)
        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=0)
        right.rowconfigure(1, weight=1)

        # View selector
        sel_fr = ctk.CTkFrame(right, fg_color=T()["card"], corner_radius=8)
        sel_fr.grid(row=0, column=0, columnspan=2, sticky="ew",
                    padx=4, pady=(4, 2))
        ctk.CTkLabel(sel_fr, text="View:",
                     font=ctk.CTkFont(size=11)).pack(side="left", padx=8, pady=6)
        self._view_var = ctk.StringVar(value=self._VIEWS[0])
        ctk.CTkSegmentedButton(
            sel_fr,
            values=self._VIEWS,
            variable=self._view_var,
            command=lambda v: self._generate(),
        ).pack(side="left", padx=4, pady=6)

        # Scrollable canvas
        self._canvas = tk.Canvas(
            right, bg=T()["fig_bg"], highlightthickness=0)
        self._ysb = ctk.CTkScrollbar(right, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._ysb.set)
        self._canvas.grid(row=1, column=0, sticky="nsew",
                          padx=(4, 0), pady=(2, 4))
        self._ysb.grid(row=1, column=1, sticky="ns", pady=(2, 4))

        self._inner = ctk.CTkFrame(self._canvas, fg_color=T()["panel"])
        self._win_id = self._canvas.create_window(
            (0, 0), window=self._inner, anchor="nw")

        self._inner.bind(
            "<Configure>",
            lambda e: self._canvas.configure(
                scrollregion=self._canvas.bbox("all")))
        self._canvas.bind(
            "<Configure>",
            lambda e: self._canvas.itemconfig(self._win_id, width=e.width))
        self._canvas.bind(
            "<MouseWheel>",
            lambda e: self._canvas.yview_scroll(
                int(-1 * (e.delta / 120)), "units"))

        self._show_placeholder(
            "Select a view and click   Generate Plot.\n\n"
            "Load animals in the Combined Analysis tab first.\n"
            "For Diff Heatmap & Sankey, run Reclustering in\n"
            "Unbiased Analytics first (optional but recommended).")

    # ── Display helpers ───────────────────────────────────────────────────────

    def _show_placeholder(self, msg: str):
        for w in self._inner.winfo_children():
            w.destroy()
        ctk.CTkLabel(self._inner, text=msg,
                     text_color=T()["muted"],
                     font=ctk.CTkFont(size=13),
                     justify="center").pack(pady=50, padx=20)

    def _show_figures(self, figs: list):
        """Display one or more figures stacked in the scrollable canvas."""
        # Destroy canvas widgets first, then break Python reference cycles.
        for w in self._inner.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        # Explicitly break canvas ↔ toolbar ↔ StringVar/PhotoImage cycles so
        # tk __del__ runs here via refcount, not from a GC pass inside a
        # background thread (which crashes the Tcl interpreter).
        for old in self._current_figs:
            _oc = getattr(old, 'canvas', None)
            if _oc is not None:
                _tb = getattr(_oc, 'toolbar', None)
                if _tb is not None:
                    try:
                        _oc.toolbar = None
                    except Exception:
                        pass
                    try:
                        _tb.canvas = None
                    except Exception:
                        pass
                for _a in ('_tkphoto', 'photo'):
                    if getattr(_oc, _a, None) is not None:
                        try:
                            setattr(_oc, _a, None)
                        except Exception:
                            pass
                try:
                    _oc.figure = None
                except Exception:
                    pass
            try:
                plt.close(old)
            except Exception:
                pass
            try:
                old.canvas = None
            except Exception:
                pass
        self._current_mpl  = None   # release old canvas ref before gc
        self._current_figs = list(figs)
        self._current_fig  = figs[0] if figs else None

        import gc as _gc
        _gc.collect()

        def _bind_wheel(w):
            w.bind("<MouseWheel>",
                   lambda e: self._canvas.yview_scroll(
                       int(-1 * (e.delta / 120)), "units"))
            for child in w.winfo_children():
                _bind_wheel(child)

        is_sankey = self._view_var.get() == "Sankey"
        last_canvas = None
        for fi, fig in enumerate(figs):
            mpl_c = FigureCanvasTkAgg(fig, master=self._inner)
            mpl_c.draw()
            widget = mpl_c.get_tk_widget()
            widget.pack(fill="x", expand=True, padx=2, pady=(0, 2))
            _bind_wheel(widget)
            if is_sankey and fi == 0:
                mpl_c.mpl_connect("pick_event", self._on_sankey_pick)
            last_canvas = mpl_c

        # Single toolbar on the last figure
        if last_canvas is not None:
            self._current_mpl = last_canvas
            tb_fr = ctk.CTkFrame(self._inner, fg_color=T()["card2"], height=30)
            tb_fr.pack(fill="x")
            NavigationToolbar2Tk(last_canvas, tb_fr)

    def _status(self, msg: str, color: str = None):
        self._status_lbl.configure(
            text=msg, text_color=color or T()["subtext"])

    # ── Export CSV ────────────────────────────────────────────────────────────

    def export_csv_data(self, out_dir: pathlib.Path) -> int:
        """Compute and write Behavioral Explorer data CSVs. Returns number of files written."""
        animals = self._apply_brf(self._get_animals(group_by=self._group_by_key()))
        if not animals:
            return 0
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0

        all_cids = sorted({
            int(lab) for ani in animals
            for lab in ani["df"]["label"].unique()
        })
        if not all_cids:
            return 0

        idx    = {c: i for i, c in enumerate(all_cids)}
        n_cids = len(all_cids)

        groups_eg: dict = {}
        for ani in animals:
            groups_eg.setdefault(ani["exp_group"], []).append(ani)

        # ── 1. Per-group cluster-level transition matrices ────────────────
        tmat_by_eg: dict = {}
        tmat_rows: list  = []
        for eg, ani_list in groups_eg.items():
            tsum = np.zeros((n_cids, n_cids), dtype=float)
            cnt  = 0
            for ani in ani_list:
                mat, ani_cids = compute_transition_matrix(ani["df"])
                for ri, ci in enumerate(ani_cids):
                    for cj_i, cj in enumerate(ani_cids):
                        if ci in idx and cj in idx:
                            tsum[idx[ci], idx[cj]] += mat[ri, cj_i]
                cnt += 1
            tmat = tsum / max(cnt, 1)
            np.fill_diagonal(tmat, 0)
            rs = tmat.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            tmat = tmat / rs
            tmat_by_eg[eg] = tmat
            for i, ci in enumerate(all_cids):
                for j, cj in enumerate(all_cids):
                    if i != j:
                        tmat_rows.append({
                            "exp_group":       eg,
                            "from_cluster":    f"C{ci}",
                            "to_cluster":      f"C{cj}",
                            "transition_prob": round(float(tmat[i, j]), 6),
                        })
        if tmat_rows:
            pd.DataFrame(tmat_rows).to_csv(
                out_dir / "transition_matrices.csv", index=False)
            n += 1

        # ── 2. Pairwise delta transition matrices ─────────────────────────
        eg_names = list(tmat_by_eg.keys())
        if len(eg_names) >= 2:
            ref_eg    = eg_names[0]
            diff_rows: list = []
            for eg_b in eg_names[1:]:
                delta = tmat_by_eg[eg_b] - tmat_by_eg[ref_eg]
                for i, ci in enumerate(all_cids):
                    for j, cj in enumerate(all_cids):
                        if i != j:
                            diff_rows.append({
                                "reference_group":  ref_eg,
                                "comparison_group": eg_b,
                                "from_cluster":     f"C{ci}",
                                "to_cluster":       f"C{cj}",
                                "delta_prob":       round(float(delta[i, j]), 6),
                            })
            if diff_rows:
                pd.DataFrame(diff_rows).to_csv(
                    out_dir / "diff_transition_matrices.csv", index=False)
                n += 1

        # ── 3. Per-animal dwell-time statistics ───────────────────────────
        dwell_rows: list = []
        for ani in animals:
            fps_a = float(ani.get("fps", 30))
            name  = ani.get("name", "?")
            eg    = ani.get("exp_group", "")
            for cid in all_cids:
                sub = ani["df"][ani["df"]["label"] == cid]
                if "run_len" in sub.columns:
                    bouts = sub["run_len"].values / fps_a
                elif "Run lengths" in sub.columns:
                    bouts = sub["Run lengths"].values / fps_a
                elif "duration_sec" in sub.columns:
                    bouts = sub["duration_sec"].values
                else:
                    bouts = np.array([], dtype=float)
                bouts = bouts.astype(float)
                dwell_rows.append({
                    "animal":         name,
                    "exp_group":      eg,
                    "cluster":        f"C{cid}",
                    "n_bouts":        len(bouts),
                    "mean_dwell_s":   round(float(bouts.mean()), 4)   if len(bouts) else 0.0,
                    "median_dwell_s": round(float(np.median(bouts)), 4) if len(bouts) else 0.0,
                    "std_dwell_s":    round(float(bouts.std()), 4)    if len(bouts) else 0.0,
                    "total_dwell_s":  round(float(bouts.sum()), 4),
                })
        if dwell_rows:
            pd.DataFrame(dwell_rows).to_csv(out_dir / "dwell_stats.csv", index=False)
            n += 1

        return n

    def _export_csv(self):
        animals = self._apply_brf(self._get_animals(group_by=self._group_by_key()))
        if not animals:
            messagebox.showinfo("Nothing to export",
                                "Load animals in the Combined Analysis tab first.",
                                parent=self)
            return
        folder = filedialog.askdirectory(
            parent=self,
            title="Export Behavioral Explorer Data — Choose Output Folder",
        )
        if not folder:
            return
        try:
            n = self.export_csv_data(pathlib.Path(folder))
            messagebox.showinfo(
                "Exported",
                f"Saved {n} file(s) to:\n{folder}",
                parent=self,
            )
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc), parent=self)

    def _on_sankey_pick(self, event):
        """Re-anchor the Sankey when the user clicks a behaviour bar."""
        gid = getattr(event.artist, "get_gid", lambda: None)()
        if not gid or not gid.startswith("cluster:"):
            return
        cid = gid.split(":", 1)[1]
        self._anchor_cluster_entry.delete(0, "end")
        self._anchor_cluster_entry.insert(0, cid)
        self._status(f"Re-anchoring Sankey on C{cid}…")
        self.after(30, self._generate)

    def _get_user_groups(self) -> dict:
        """Return the user-defined behaviour groups, or {} if unavailable."""
        try:
            if self._get_groups_fn:
                return self._get_groups_fn() or {}
        except Exception:
            pass
        return {}

    def _get_umap(self):
        """Return (embedding, labels) tuple, or (None, None) if unavailable."""
        try:
            if self._get_umap_fn:
                result = self._get_umap_fn()
                if result and len(result) == 2:
                    return result[0], result[1]
        except Exception:
            pass
        return None, None

    # ── Biological relevance filter ───────────────────────────────────────────

    def _get_brf_params(self):
        """Return (min_mean_ms, min_total_s, min_freq) with None for blanks."""
        out = []
        for var in (self._brf_min_mean_ms_var,
                    self._brf_min_total_s_var,
                    self._brf_min_freq_var):
            try:
                v = var.get().strip()
                out.append(float(v) if v else None)
            except (ValueError, AttributeError):
                out.append(None)
        return tuple(out)

    def _apply_brf(self, animals: list) -> list:
        """
        Remove rows whose cluster label belongs to a cluster that fails the
        biological relevance filter.  Returns a new list of animal dicts with
        filtered dataframes; the original dicts are not mutated.
        When no thresholds are set returns the original list unchanged.
        """
        min_mean_ms, min_total_s, min_freq = self._get_brf_params()
        if min_mean_ms is None and min_total_s is None and min_freq is None:
            return animals

        import numpy as _np

        # Collect per-cluster, per-animal stats (mean bout s, total dur s, freq)
        cl_stats: dict = {}   # cid -> {"mb": [], "td": [], "fr": []}
        for ani in animals:
            df  = ani["df"]
            fps = float(ani.get("fps", 30) or 30)
            if "duration_sec" in df.columns:
                dur = df["duration_sec"]
            elif "run_len" in df.columns:
                dur = df["run_len"] / fps
            else:
                continue
            for cid in df["label"].unique():
                key = int(cid)
                mask = df["label"] == cid
                bouts = dur[mask]
                if key not in cl_stats:
                    cl_stats[key] = {"mb": [], "td": [], "fr": []}
                cl_stats[key]["mb"].append(float(bouts.mean()) if len(bouts) else 0.0)
                cl_stats[key]["td"].append(float(bouts.sum()))
                cl_stats[key]["fr"].append(int(mask.sum()))

        excluded: set = set()
        for cid, s in cl_stats.items():
            if min_mean_ms is not None:
                if _np.mean(s["mb"]) * 1000 < min_mean_ms:
                    excluded.add(cid); continue
            if min_total_s is not None:
                if _np.mean(s["td"]) < min_total_s:
                    excluded.add(cid); continue
            if min_freq is not None:
                if _np.mean(s["fr"]) < min_freq:
                    excluded.add(cid)

        if not excluded:
            return animals

        filtered = []
        for ani in animals:
            df_f = ani["df"][~ani["df"]["label"].isin(excluded)].reset_index(drop=True)
            filtered.append({**ani, "df": df_f})

        n_removed = len(excluded)
        self._status(
            f"Bio filter active: {n_removed} cluster(s) removed "
            f"({', '.join(f'C{c}' for c in sorted(excluded))}).",
            T().get("warn", "#e8a838"))
        return filtered

    # ── Generation ────────────────────────────────────────────────────────────

    def _generate(self, _=None):
        view      = self._view_var.get()
        animals   = self._apply_brf(self._get_animals(group_by=self._group_by_key()))
        recluster = self._get_recluster()
        groups    = self._get_user_groups()

        if not animals:
            self._show_placeholder(
                "No animals loaded.\nAdd animals in the Combined Analysis tab.")
            self._status("No animals loaded.", T()["muted"])
            return

        # If reclustering hasn't been run yet, build a raw preview from the
        # existing cluster assignments so all plots can render immediately.
        # After reclustering is performed the caller's getter returns real data
        # and the plots are regenerated to show the impact.
        is_raw_preview = False
        if recluster is None:
            try:
                recluster = build_raw_recluster_result(animals)
                is_raw_preview = True
            except Exception:
                recluster = None   # some views don't need it; let them handle None

        self._status(f"Building {view}…")
        self.update_idletasks()
        t = T()

        try:
            figs = []

            if view == "Diff Heatmap":
                ctrl_raw = self._ctrl_var.get().strip()
                ctrl     = ctrl_raw if ctrl_raw and ctrl_raw != "(auto — first group)" \
                           else None
                figs.append(build_diff_heatmap_figure(recluster, animals, ctrl, t))
                if groups:
                    figs.append(
                        build_diff_heatmap_beh_groups_figure(animals, groups, ctrl, t))

            elif view == "Dwell Violin":
                figs.append(build_dwell_violin_figure(animals, t))
                if groups:
                    figs.append(
                        build_dwell_violin_beh_groups_figure(animals, groups, t))

            elif view == "Sankey":
                n_steps = max(2, min(10, int(
                    round(float(self._nsteps_var.get())))))
                _ak = self._anchor_cluster_entry.get().strip().lstrip("Cc")
                anchor_cluster = int(_ak) if _ak.lstrip("-").isdigit() else None
                anchor_group   = self._anchor_group_entry.get().strip() or None
                anchor_note = (f"C{anchor_cluster}  [user-defined]"
                               if anchor_cluster is not None else "auto")
                self._status(f"Building Sankey  (anchor: {anchor_note})…")
                self.update_idletasks()
                figs.append(build_sankey_figure(
                    animals, t, n_steps=n_steps,
                    anchor_cluster=anchor_cluster))
                if groups:
                    figs.append(
                        build_sankey_beh_groups_figure(
                            animals, groups, t, n_steps=n_steps,
                            anchor_group=anchor_group))

            elif view == "Group Transitions":
                figs.append(
                    build_per_group_transition_figure(recluster, animals, t))
                if groups:
                    figs.append(
                        build_group_aggregate_transition_figure(
                            animals, groups, t))

            elif view == "Group Networks":
                figs.append(
                    build_per_group_network_figure(recluster, animals, t))
                if groups:
                    figs.append(
                        build_group_aggregate_network_figure(animals, groups, t))

            elif view == "Energy Landscape":
                emb, lbl = self._get_umap()
                figs.append(
                    build_energy_landscape_figure(animals, emb, lbl, t, groups))
                if groups:
                    figs.append(
                        build_umap_groups_figure(emb, lbl, groups, t))

            else:
                self._status("Unknown view.", T()["muted"])
                return

            self._show_figures(figs)
            suffix = "  +  behaviour groups" if groups else ""
            if is_raw_preview:
                self._status(
                    f"{view} ready{suffix}  "
                    "[original clusters — run Reclustering to update]",
                    T()["warn"] if "warn" in T() else "#e8a838")
            else:
                self._status(f"{view} ready{suffix}.", T()["subtext"])

        except Exception as exc:
            import traceback as _tb
            self._show_placeholder(
                f"Error generating {view}:\n{exc}\n\n"
                f"{_tb.format_exc()[-400:]}")
            self._status(f"Error: {exc}", "#ff6666")

    # ── Save ─────────────────────────────────────────────────────────────────

    def save_all_figures(self, out_dir: "pathlib.Path", ts: str) -> int:
        """Generate and save every explorer view to out_dir. Returns count saved."""
        animals = self._apply_brf(self._get_animals(group_by=self._group_by_key()))
        if not animals:
            return 0
        out_dir.mkdir(parents=True, exist_ok=True)

        recluster = self._get_recluster()
        if recluster is None:
            try:
                recluster = build_raw_recluster_result(animals)
            except Exception:
                recluster = None

        groups = self._get_user_groups()
        t      = T()
        sfx_map = ["_clusters", "_beh_groups"]
        n = 0

        # Read UI settings once so each view uses consistent parameters
        try:
            _n_steps = max(2, min(10, int(round(float(self._nsteps_var.get())))))
        except (ValueError, AttributeError):
            _n_steps = 5
        _ak = self._anchor_cluster_entry.get().strip().lstrip("Cc")
        _anchor_cluster = int(_ak) if _ak.lstrip("-").isdigit() else None
        _anchor_group   = self._anchor_group_entry.get().strip() or None
        ctrl_raw = self._ctrl_var.get().strip()
        ctrl     = ctrl_raw if ctrl_raw and ctrl_raw != "(auto — first group)" else None

        for view in self._VIEWS:
            view_slug = view.replace(" ", "_").lower()
            try:
                figs = []
                if view == "Diff Heatmap":
                    figs.append(
                        build_diff_heatmap_figure(recluster, animals, ctrl, t))
                    if groups:
                        figs.append(
                            build_diff_heatmap_beh_groups_figure(
                                animals, groups, ctrl, t))
                elif view == "Dwell Violin":
                    figs.append(build_dwell_violin_figure(animals, t))
                    if groups:
                        figs.append(
                            build_dwell_violin_beh_groups_figure(animals, groups, t))
                elif view == "Sankey":
                    figs.append(build_sankey_figure(
                        animals, t,
                        n_steps=_n_steps, anchor_cluster=_anchor_cluster))
                    if groups:
                        figs.append(build_sankey_beh_groups_figure(
                            animals, groups, t,
                            n_steps=_n_steps, anchor_group=_anchor_group))
                elif view == "Group Transitions":
                    figs.append(
                        build_per_group_transition_figure(recluster, animals, t))
                    if groups:
                        figs.append(
                            build_group_aggregate_transition_figure(
                                animals, groups, t))
                elif view == "Group Networks":
                    figs.append(
                        build_per_group_network_figure(recluster, animals, t))
                    if groups:
                        figs.append(
                            build_group_aggregate_network_figure(
                                animals, groups, t))
                elif view == "Energy Landscape":
                    emb, lbl = self._get_umap()
                    figs.append(
                        build_energy_landscape_figure(animals, emb, lbl, t, groups))
                    if groups:
                        figs.append(
                            build_umap_groups_figure(emb, lbl, groups, t))
                        try:
                            _tag = f"explorer_{view_slug}_{ts}"
                            save_umap_groups_3d(
                                emb, lbl, groups,
                                out_path=str(
                                    out_dir /
                                    f"explorer_{view_slug}_3d_groups_{ts}.html"),
                                tag=_tag)
                            save_group_transitions_3d(
                                emb, lbl, groups, animals,
                                out_path=str(
                                    out_dir /
                                    f"explorer_{view_slug}_3d_exp_transitions_{ts}.html"),
                                tag=_tag)
                        except Exception:
                            pass
            except Exception:
                continue

            for i, fig in enumerate(figs):
                sfx = sfx_map[i] if i < len(sfx_map) else f"_{i}"
                for ext in ("png", "pdf"):
                    fig.savefig(
                        str(out_dir / f"explorer_{view_slug}{sfx}_{ts}.{ext}"),
                        dpi=300 if ext == "png" else None,
                        bbox_inches="tight", facecolor=fig.get_facecolor())
                plt.close(fig)
                n += 1

        return n

    def _save_graph(self):
        if not self._current_figs:
            self._status("Nothing to save yet.", T()["muted"])
            return
        p = filedialog.asksaveasfilename(
            title="Save explorer graph",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"),
                       ("All", "*")],
        )
        if not p:
            return
        try:
            base  = pathlib.Path(p)
            saved = []
            if len(self._current_figs) == 1:
                self._current_figs[0].savefig(
                    str(base), dpi=200, bbox_inches="tight",
                    facecolor=self._current_figs[0].get_facecolor())
                saved.append(base.name)
            else:
                suffixes = ["_clusters", "_beh_groups"]
                for i, fig in enumerate(self._current_figs):
                    sfx   = suffixes[i] if i < len(suffixes) else f"_{i}"
                    fpath = base.parent / (base.stem + sfx + base.suffix)
                    fig.savefig(str(fpath), dpi=200, bbox_inches="tight",
                                facecolor=fig.get_facecolor())
                    saved.append(fpath.name)

            # ── 3D companion figures for the Energy Landscape view ────────────
            # When the user saves the Energy Landscape (which includes the
            # behavioural-groups UMAP), also write:
            #   <stem>_3d_groups.html/.png   — 3D scatter coloured by beh. group
            #   <stem>_3d_exp_transitions.html/.png — group transitions per exp-group
            if self._view_var.get() == "Energy Landscape":
                emb, lbl = self._get_umap()
                groups   = self._get_user_groups()
                if emb is not None and lbl is not None and groups:
                    animals = self._apply_brf(self._get_animals(group_by=self._group_by_key()))
                    tag     = base.stem

                    path_3d_grp = base.parent / (base.stem + "_3d_groups.html")
                    save_umap_groups_3d(emb, lbl, groups,
                                        out_path=str(path_3d_grp), tag=tag)
                    if path_3d_grp.exists():
                        saved.append(path_3d_grp.name)
                    if path_3d_grp.with_suffix(".png").exists():
                        saved.append(path_3d_grp.with_suffix(".png").name)

                    path_3d_trans = base.parent / (
                        base.stem + "_3d_exp_transitions.html")
                    save_group_transitions_3d(
                        emb, lbl, groups, animals,
                        out_path=str(path_3d_trans), tag=tag)
                    if path_3d_trans.exists():
                        saved.append(path_3d_trans.name)
                    if path_3d_trans.with_suffix(".png").exists():
                        saved.append(path_3d_trans.with_suffix(".png").name)

            self._status(f"Saved → {', '.join(saved)}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))


#
# MAIN APPLICATION
#

class BSOiDApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1440x880")
        self.minsize(1050, 660)
        self.resizable(True, True)
        _ico = pathlib.Path(__file__).parent / "CUBE.ico"
        if _ico.is_file():
            try:
                self.wm_iconbitmap(str(_ico))
            except Exception:
                pass

        self._root_dir:       pathlib.Path | None = None
        self._csv_paths:      list               = []
        self._active_csv:     pathlib.Path | None = None
        self._df:             pd.DataFrame | None = None
        self._fps:            int                 = DEFAULT_FPS
        self._all_labels:     list                = []
        self._metrics:        dict                = {}
        self._combined:       dict | None         = None
        self._editor:         GroupEditorWindow | None = None
        self._umap_embedding: object              = None   # np.ndarray or None
        self._umap_labels:    object              = None   # np.ndarray or None
        self._ignore_groups_var: ctk.BooleanVar | None = None  # set in _build_ui

        ctk.set_appearance_mode("dark")
        self._build_ui()

    #   UI construction  

    def _build_ui(self):
        self.columnconfigure(0, weight=0, minsize=290)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        #   LEFT PANEL  
        left = ctk.CTkFrame(self, fg_color=T()["bg"], corner_radius=0)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)

        # title + theme
        hdr = ctk.CTkFrame(left, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=12, pady=(16, 6))
        hdr.columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text="   CUBE Suite",
                     font=ctk.CTkFont(size=17, weight="bold"),
                     ).grid(row=0, column=0, sticky="w")
        self._theme_var = ctk.StringVar(value="Dark")
        ctk.CTkSegmentedButton(
            hdr, values=["Dark", "Light"],
            variable=self._theme_var,
            command=self._toggle_theme, width=130,
        ).grid(row=0, column=1, sticky="e")

        # folder card
        fc = ctk.CTkFrame(left, fg_color=T()["card"], corner_radius=8)
        fc.grid(row=1, column=0, padx=12, pady=4, sticky="ew")
        fc.columnconfigure(0, weight=1)
        self._folder_lbl = ctk.CTkLabel(
            fc, text="No folder selected",
            text_color=T()["subtext"], wraplength=240,
            justify="left", font=ctk.CTkFont(size=11),
        )
        self._folder_lbl.grid(row=0, column=0, padx=10, pady=(8, 2), sticky="w")
        bf = ctk.CTkFrame(fc, fg_color="transparent")
        bf.grid(row=1, column=0, padx=8, pady=(2, 8), sticky="ew")
        ctk.CTkButton(bf, text="Open: Folder", command=self._select_folder,
                      width=110, fg_color=T()["btn_folder"]
                      ).pack(side="left", padx=(0, 4))
        ctk.CTkButton(bf, text="  Load Mapping", command=self._load_mapping,
                      width=120, fg_color=T()["btn_load"]
                      ).pack(side="left")

        # Ignore-groups toggle
        self._ignore_groups_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            fc, text="Ignore imported groups — use clusters directly",
            variable=self._ignore_groups_var,
            font=ctk.CTkFont(size=11),
            command=self._on_ignore_groups_changed,
        ).grid(row=2, column=0, padx=10, pady=(0, 8), sticky="w")

        # csv / fps card
        mc = ctk.CTkFrame(left, fg_color=T()["card"], corner_radius=8)
        mc.grid(row=2, column=0, padx=12, pady=4, sticky="ew")
        mc.columnconfigure(1, weight=1)
        ctk.CTkLabel(mc, text="File:").grid(row=0, column=0, padx=(10, 4), pady=8)
        self._csv_combo = ctk.CTkComboBox(
            mc, values=[], state="readonly",
            command=self._csv_changed, width=190,
        )
        self._csv_combo.grid(row=0, column=1, padx=(0, 10), pady=8)
        ctk.CTkLabel(mc, text="FPS:").grid(row=1, column=0, padx=(10, 4))
        self._fps_var = ctk.StringVar(value=str(DEFAULT_FPS))
        ctk.CTkEntry(mc, textvariable=self._fps_var, width=70,
                     ).grid(row=1, column=1, padx=(0, 10), pady=(0, 8), sticky="w")
        self._fps_var.trace_add("write", lambda *_: self._refresh_preview())

        # action buttons
        ac = ctk.CTkFrame(left, fg_color=T()["card"], corner_radius=8)
        ac.grid(row=3, column=0, padx=12, pady=4, sticky="ew")
        ac.columnconfigure(0, weight=1)
        _btns = [
            ("   Group Editor",            self._open_editor,     "btn_folder"),
            ("   Analyse & Preview",       self._analyse,         "btn_add"),
            ("Save:  Save Results",        self._save,            "btn_save"),
            ("Save All:  Save All Results",self._save_all_results,"btn_save"),
        ]
        for row_i, (text, cmd, key) in enumerate(_btns):
            ctk.CTkButton(ac, text=text, command=cmd,
                          fg_color=T()[key],
                          ).grid(row=row_i, column=0, padx=10,
                                 pady=(8 if row_i == 0 else 4,
                                       8 if row_i == len(_btns) - 1 else 2),
                                 sticky="ew")

        # status
        self._status_lbl = ctk.CTkLabel(
            left, text="", text_color=T()["subtext"],
            wraplength=260, justify="left", font=ctk.CTkFont(size=11),
        )
        self._status_lbl.grid(row=4, column=0, padx=14, pady=(4, 14), sticky="w")

        #   RIGHT PANEL (tabs)  
        right = ctk.CTkFrame(self, fg_color=T()["bg"], corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self._tabs = ctk.CTkTabview(right, fg_color=T()["panel"], anchor="nw")
        self._tabs.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self._tab_preview   = self._tabs.add("Live Preview")
        self._tab_analysis  = self._tabs.add("Full Analysis")
        self._tab_metrics   = self._tabs.add("Metrics Table")
        self._tab_combined  = self._tabs.add("Combined Analysis")
        self._tab_unbiased  = self._tabs.add("Unbiased Analytics")
        self._tab_explorer  = self._tabs.add("Behavioral Explorer")
        self._tab_predictor = self._tabs.add("Group Predictor")

        for tab in (self._tab_preview, self._tab_analysis,
                    self._tab_metrics, self._tab_combined,
                    self._tab_unbiased, self._tab_explorer,
                    self._tab_predictor):
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)

        self._preview_panel = CanvasPanel(self._tab_preview)
        self._preview_panel.grid(row=0, column=0, sticky="nsew")
        self._preview_panel.clear(
            "Select a folder and open the Group Editor\n"
            "to see the live ethogram preview."
        )

        self._analysis_panel = CanvasPanel(self._tab_analysis)
        self._analysis_panel.grid(row=0, column=0, sticky="nsew")
        self._analysis_panel.clear("Click 'Analyse & Preview' to generate plots.")

        self._metrics_frame = ctk.CTkFrame(self._tab_metrics, fg_color=T()["panel"])
        self._metrics_frame.grid(row=0, column=0, sticky="nsew")
        self._metrics_frame.columnconfigure(0, weight=1)
        self._metrics_frame.rowconfigure(0, weight=1)
        ctk.CTkLabel(
            self._metrics_frame, text="Metrics will appear here after analysis.",
            text_color=T()["muted"], font=ctk.CTkFont(size=14),
        ).grid(row=0, column=0)

        self._build_combined_tab()

        # Unbiased Analytics panel - wired to push groups back into the editor
        self._unbiased_panel = UnbiasedAnalyticsPanel(
            self._tab_unbiased,
            get_animals_fn=self._get_animals_for_unbiased,
            load_groups_to_editor_fn=self._load_reclustered_groups,
            get_groups_fn=self._get_groups,
            get_combined_fn=lambda: getattr(self, "_combined", None),
        )
        self._unbiased_panel.grid(row=0, column=0, sticky="nsew")

        # Behavioral Explorer — group-comparison panel
        self._explorer_panel = BehavioralExplorerPanel(
            self._tab_explorer,
            get_animals_fn=self._get_animals_for_unbiased,
            get_recluster_fn=lambda: getattr(
                self._unbiased_panel, "_recluster", None),
            get_groups_fn=self._get_groups,
            get_umap_fn=lambda: (
                getattr(self, "_umap_embedding", None),
                getattr(self, "_umap_labels",    None),
            ),
        )
        self._explorer_panel.grid(row=0, column=0, sticky="nsew")

        # Group Predictor — experimental group classification from behavioral profiles
        self._predictor_panel = GroupPredictorPanel(
            self._tab_predictor,
            get_animals_fn=self._get_animals_bio_filtered,
            get_groups_fn=self._get_groups,
        )
        self._predictor_panel.grid(row=0, column=0, sticky="nsew")

    def _get_animals_for_unbiased(self, group_by: str = "label1") -> list:
        if hasattr(self, "_animal_panel"):
            return self._animal_panel.get_animals(group_by=group_by)
        return []

    def _get_animals_bio_filtered(self, group_by: str = "label1") -> list:
        """Like _get_animals_for_unbiased but respects the Unbiased Analytics bio filter."""
        animals = self._get_animals_for_unbiased(group_by=group_by)
        if hasattr(self, "_unbiased_panel"):
            return self._unbiased_panel._apply_bio_filter(animals)
        return animals

    def _load_reclustered_groups(self, groups: dict):
        """
        Push a reclustered groups dict into the Group Editor.
        Asks the user whether to REPLACE existing groups or KEEP both.
        """
        self._open_editor()
        existing = self._editor.get_groups()
        if existing:
            replace = messagebox.askyesno(
                "Push Reclustered Groups",
                f"The Group Editor already has {len(existing)} group(s).\n\n"
                "YES — Replace all groups with the reclustered groups\n"
                "NO  — Keep existing groups AND add reclustered groups",
                icon="question",
            )
        else:
            replace = True

        if replace:
            self._editor.load_groups_from_dict(groups, self._all_labels)
        else:
            self._editor.merge_groups_from_dict(groups)

        self._editor.deiconify()
        self._editor.lift()
        self.on_groups_changed()
        # Switch to Combined Analysis tab so the user can immediately run it
        self._tabs.set("Combined Analysis")

        # Show UMAP comparison if cube_core saved the embedding data
        if (self._umap_embedding is not None
                and self._umap_labels is not None
                and len(groups) > 0):
            self._show_umap_comparison(groups)

    def _show_umap_comparison(self, new_groups: dict):
        """Open a Toplevel window showing old vs. new UMAP side by side."""
        try:
            fig = build_umap_comparison_figure(
                self._umap_embedding, self._umap_labels, new_groups)
        except Exception:
            return   # best-effort; don't block the main workflow

        win = ctk.CTkToplevel(self)
        win.title("UMAP — Before vs. After Recombination")
        win.geometry("1300x620")
        win.resizable(True, True)
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)
        win.rowconfigure(1, weight=0)

        panel = CanvasPanel(win)
        panel.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        panel.show_figure(fig)

        def _save():
            p = filedialog.asksaveasfilename(
                parent=win,
                title="Save UMAP comparison",
                defaultextension=".png",
                filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("All", "*")],
            )
            if not p:
                return
            try:
                fig.savefig(p, dpi=200, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
            except Exception as exc:
                messagebox.showerror("Save Error", str(exc), parent=win)

        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        ctk.CTkButton(btn_row, text="Save:  Save Image", command=_save,
                      fg_color=T()["btn_save"], width=160).pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Close", command=win.destroy,
                      width=100).pack(side="right", padx=4)

        win.lift()
        win.focus_force()

    def _build_combined_tab(self):
        first_build = not hasattr(self, "_animal_panel")

        if first_build:
            for w in self._tab_combined.winfo_children():
                w.destroy()
        else:
            if hasattr(self, "_combined_rp") and self._combined_rp.winfo_exists():
                self._combined_rp.destroy()

        self._tab_combined.columnconfigure(0, weight=0, minsize=320)
        self._tab_combined.columnconfigure(1, weight=1)
        self._tab_combined.rowconfigure(0, weight=1)   # content fills all vertical space

        if first_build:
            self._animal_panel = AnimalListPanel(
                self._tab_combined, on_change=lambda: None,
            )
            self._animal_panel.grid(row=0, column=0, sticky="nsew",
                                    padx=(8, 4), pady=4)

        # right-hand area — no separate title row, button bar is the only header
        rp = ctk.CTkFrame(self._tab_combined, fg_color=T()["panel"])
        rp.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=4)
        rp.columnconfigure(0, weight=1)
        rp.columnconfigure(1, weight=0)
        rp.rowconfigure(1, weight=1)
        self._combined_rp = rp

        btns = ctk.CTkFrame(rp, fg_color=T()["card"])
        btns.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 2))
        ctk.CTkButton(btns, text="   Run Combined Analysis",
                      command=self._run_combined,
                      fg_color=T()["btn_folder"]).pack(side="left", padx=6, pady=6)
        ctk.CTkButton(btns, text="Save:  Export Combined",
                      command=self._export_combined,
                      fg_color=T()["btn_save"]).pack(side="left", padx=4, pady=6)
        ctk.CTkLabel(btns, text="Group by:",
                     text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(side="left", padx=(16, 2), pady=6)
        if not hasattr(self, "_combined_group_by_var"):
            self._combined_group_by_var = ctk.StringVar(value="Label 1")
        ctk.CTkOptionMenu(btns, variable=self._combined_group_by_var,
                          values=["Label 1", "Label 2", "Label 3", "All Labels"],
                          width=120).pack(side="left", padx=2, pady=6)

        # Scrollable plot area: tk.Canvas + vertical scrollbar
        self._scroll_canvas = tk.Canvas(
            rp, bg=T()["fig_bg"], highlightthickness=0)
        self._scroll_ysb = ctk.CTkScrollbar(
            rp, command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=self._scroll_ysb.set)
        self._scroll_canvas.grid(row=1, column=0, sticky="nsew", padx=(4, 0), pady=4)
        self._scroll_ysb.grid(row=1, column=1, sticky="ns", pady=4)

        self._combined_plot_frame = ctk.CTkFrame(
            self._scroll_canvas, fg_color=T()["panel"])
        self._scroll_window = self._scroll_canvas.create_window(
            (0, 0), window=self._combined_plot_frame, anchor="nw")

        def _on_frame_configure(e):
            self._scroll_canvas.configure(
                scrollregion=self._scroll_canvas.bbox("all"))
        self._combined_plot_frame.bind("<Configure>", _on_frame_configure)

        def _on_canvas_configure(e):
            self._scroll_canvas.itemconfig(self._scroll_window, width=e.width)
        self._scroll_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(e):
            self._scroll_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        self._scroll_canvas.bind("<MouseWheel>", _on_mousewheel)

        self._combined_figs: list = []

        # Placeholder message
        self._show_combined_placeholder(
            "Add animals (CSV or TSV), set Exp Group for each,\n"
            "then click Run Combined Analysis."
        )

    def _show_combined_placeholder(self, message: str):
        for w in self._combined_plot_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(self._combined_plot_frame, text=message,
                     text_color=T()["muted"],
                     font=ctk.CTkFont(size=13),
                     justify="center").pack(pady=60)

    def _clear_combined_plots(self):
        for fig in self._combined_figs:
            try:
                plt.close(fig)
            except Exception:
                pass
        self._combined_figs.clear()
        for w in self._combined_plot_frame.winfo_children():
            w.destroy()

    def _add_combined_figure(self, fig, title: str = ""):
        self._combined_figs.append(fig)
        card = ctk.CTkFrame(self._combined_plot_frame,
                            fg_color=T()["card"], corner_radius=6)
        card.pack(fill="x", padx=4, pady=(0, 8), expand=True)
        if title:
            ctk.CTkLabel(card, text=title,
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=T()["subtext"]).pack(anchor="w", padx=10, pady=(6, 2))
        fc = FigureCanvasTkAgg(fig, master=card)
        fc.draw()
        widget = fc.get_tk_widget()
        widget.pack(fill="x", expand=True, padx=2, pady=(0, 4))
        tb_fr = ctk.CTkFrame(card, fg_color=T()["card2"], height=30)
        tb_fr.pack(fill="x")
        NavigationToolbar2Tk(fc, tb_fr)

        def _bind_wheel(w):
            w.bind("<MouseWheel>",
                   lambda e: self._scroll_canvas.yview_scroll(
                       int(-1 * (e.delta / 120)), "units"))
            for child in w.winfo_children():
                _bind_wheel(child)
        _bind_wheel(widget)

    #   editor  

    def _open_editor(self):
        if self._editor is None or not self._editor.winfo_exists():
            self._editor = GroupEditorWindow(self)
            if self._all_labels:
                self._editor.refresh_labels(self._all_labels)
        self._editor.deiconify()
        self._editor.lift()
        self._editor.focus_force()

    def on_groups_changed(self):
        self._refresh_preview()
        if hasattr(self, "_predictor_panel"):
            self._predictor_panel._refresh_group_checklist()

    #   folder / CSV  

    def _select_folder(self):
        d = filedialog.askdirectory(title="Select experiment root folder")
        if not d:
            return
        self._root_dir = pathlib.Path(d)
        self._folder_lbl.configure(text=str(self._root_dir),
                                   text_color=T()["text"])

        # Reset cached UMAP data
        self._umap_embedding = None
        self._umap_labels    = None

        csv_ok = False
        try:
            files = find_bsoid_files(self._root_dir)
            if not files:
                raise FileNotFoundError(
                    "No bout_lengths CSV/TSV files found.\n"
                    "Searched recursively from:\n" + str(self._root_dir) + "\n\n"
                    "Run Step 3 (CUBE Clustering) to generate them first.")
            self._csv_paths = files
            self._csv_combo.configure(values=[p.name for p in files])
            self._csv_combo.set(files[0].name)
            self._load_csv(files[0])
            csv_ok = True
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

        if csv_ok:
            # Auto-populate Combined Analysis animal panel with all found files
            self._animal_panel.clear_all()
            self._animal_panel.add_files_from_paths(files)
            n = self._animal_panel.animal_count()
            if n:
                self._status(
                    f"Loaded {n} animal file{'s' if n != 1 else ''} from "
                    f"{self._root_dir.name}",
                    T()["subtext"],
                )
                try:
                    self._tabs.set("Combined Analysis")
                except Exception:
                    pass

            # Reset predictor panel so stale results are cleared
            if hasattr(self, "_predictor_panel"):
                self._predictor_panel.reset()

            # Auto-load cluster→behaviour mapping (Phase 4 output or fallback)
            self._auto_load_groups()

            # Load UMAP embedding/labels if cube_core saved them
            emb_p, lbl_p = find_umap_data(self._root_dir)
            if emb_p and lbl_p:
                try:
                    import numpy as _np
                    self._umap_embedding = _np.load(str(emb_p))
                    self._umap_labels    = _np.load(str(lbl_p))
                except Exception:
                    pass   # UMAP data is optional

    def _csv_changed(self, name):
        for p in self._csv_paths:
            if p.name == name:
                self._load_csv(p)
                break

    def _load_csv(self, path: pathlib.Path):
        try:
            df = load_csv(path)
        except ValueError as e:
            messagebox.showerror("File Error", str(e))
            return
        self._df         = df
        self._active_csv = path
        self._fps        = extract_fps(path)
        self._fps_var.set(str(self._fps))
        self._all_labels = sorted(df["label"].unique().tolist())
        if self._editor and self._editor.winfo_exists():
            self._editor.refresh_labels(self._all_labels)
        dur = session_duration(df, self._fps)
        self._status(
            f"{path.name}\n"
            f"{len(df)} bouts - {len(self._all_labels)} labels - {dur:.1f}s"
        )
        self._refresh_preview()

    def _on_ignore_groups_changed(self):
        """Re-run group loading whenever the ignore-groups checkbox is toggled."""
        if self._root_dir and self._all_labels:
            self._auto_load_groups()

    def _auto_load_groups(self):
        """
        Try to auto-detect behaviour groups for the current root directory:
          1. cluster_behaviour_mapping.tsv exported by Phase 4 Video Explorer
          2. Phase 4 session JSON  (contains 'behaviour_groups' key)
          3. Fallback: one group per cluster so analysis can proceed immediately

        When 'Ignore imported groups' is checked, steps 1 and 2 are skipped
        and all clusters are loaded as individual groups directly.

        In all cases, any cluster not covered by an imported group is appended
        as an individual 'Cluster N' group so no data is ever silently dropped.
        """
        if not self._root_dir or not self._all_labels:
            return
        groups_loaded = False
        ignore = self._ignore_groups_var is not None and self._ignore_groups_var.get()

        if not ignore:
            # 1. TSV exported by Phase 4
            mapping_path = find_cluster_mapping(self._root_dir)
            if mapping_path:
                try:
                    groups = load_mapping_tsv(mapping_path)
                    if groups:
                        self._open_editor()
                        self._editor.load_groups_from_dict(groups, self._all_labels)
                        self._status(
                            f"Auto-loaded mapping ({len(groups)} groups) from:\n"
                            f"{mapping_path.name}",
                            T()["subtext"],
                        )
                        groups_loaded = True
                except Exception:
                    pass

            # 2. Phase 4 session JSON
            if not groups_loaded:
                session_path = find_phase4_session_json(self._root_dir)
                if session_path:
                    try:
                        groups = load_groups_from_phase4_session(session_path)
                        if groups:
                            self._open_editor()
                            self._editor.load_groups_from_dict(groups, self._all_labels)
                            self._status(
                                f"Auto-loaded {len(groups)} groups from Phase 4 session:\n"
                                f"{session_path.name}",
                                T()["subtext"],
                            )
                            groups_loaded = True
                    except Exception:
                        pass

        # 3. Fallback (or forced when ignore-groups is active): one group per cluster
        if not groups_loaded:
            auto_groups = groups_from_all_clusters(self._all_labels)
            self._open_editor()
            self._editor.load_groups_from_dict(auto_groups, self._all_labels)
            if ignore:
                self._status(
                    f"Groups ignored. Loaded {len(auto_groups)} clusters directly.",
                    T()["subtext"],
                )
            else:
                self._status(
                    f"No behaviour mapping found. Created {len(auto_groups)} groups "
                    f"(one per cluster). Use the Group Editor to merge/rename them.",
                    T()["subtext"],
                )

    def _load_mapping(self):
        path = filedialog.askopenfilename(
            title="Load Mapping / Preset  (JSON preset, Phase 4 session JSON, or TSV)",
            filetypes=[
                ("All supported", "*.json *.tsv"),
                ("JSON preset / Phase 4 session", "*.json"),
                ("Video Explorer TSV", "*.tsv"),
                ("All", "*"),
            ],
        )
        if not path:
            return
        p = pathlib.Path(path)
        groups: dict = {}
        fps = DEFAULT_FPS

        if p.suffix.lower() == ".tsv":
            # TSV exported by Phase 4 Video Explorer
            try:
                groups = load_mapping_tsv(p)
            except Exception as e:
                messagebox.showerror("TSV Load Error", str(e))
                return
        else:
            # JSON — could be analyser preset OR Phase 4 session JSON
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                return
            if "behaviour_groups" in data:
                # Phase 4 (Video Explorer) session JSON
                try:
                    groups = load_groups_from_phase4_session(p)
                except Exception as e:
                    messagebox.showerror("Phase 4 Session Load Error", str(e))
                    return
            else:
                # Analyser preset JSON  {"groups": {...}, "fps": ...}
                groups = data.get("groups", {})
                fps    = data.get("fps", DEFAULT_FPS)

        if not groups:
            messagebox.showerror("Error", "No groups found in file.")
            return
        self._fps_var.set(str(fps))
        self._open_editor()
        self._editor.load_groups_from_dict(groups, self._all_labels)
        self._status(f"Mapping loaded: {len(groups)} groups")
        self._refresh_preview()

    #   preview  

    def _refresh_preview(self, *_):
        if self._df is None:
            return
        groups = self._get_groups()
        if not groups:
            return
        fps     = self._get_fps()
        metrics = compute_metrics(self._df, groups, fps)
        dur     = session_duration(self._df, fps)
        fig     = build_preview_figure(groups, metrics, dur)
        self._preview_panel.show_figure(fig)

    #   analyse  

    def _analyse(self):
        if self._df is None:
            messagebox.showwarning("No data", "Select a folder first.")
            return
        groups = self._get_groups()
        if not groups:
            messagebox.showwarning(
                "No groups",
                "Open the Group Editor and define at least one group with labels."
            )
            return
        fps     = self._get_fps()
        metrics = compute_metrics(self._df, groups, fps)
        dur     = session_duration(self._df, fps)
        self._metrics = metrics
        fig = build_single_figure(groups, metrics, dur)
        self._analysis_panel.show_figure(fig)
        self._build_metrics_table(groups, metrics)
        self._tabs.set("Full Analysis")
        self._status("Analysis complete v", "#88cc88")

    #   combined analysis  

    def _run_combined(self):
        self._combined_trans_figs_added = []  # bg_names that got a transitions fig
        self._combined_trans_stats      = []  # DataFrames, one per such bg_name
        groups  = self._get_groups()
        _gb_key = AnimalListPanel._label_key_to_str(
            getattr(self, "_combined_group_by_var", ctk.StringVar(value="Label 1")).get())
        animals = self._animal_panel.get_animals(group_by=_gb_key)
        if not groups:
            messagebox.showwarning("No Groups", "Define groups in the Group Editor first.")
            return
        if len(animals) < 2:
            messagebox.showwarning("Need animals",
                "Add at least 2 CSV/TSV files to run combined analysis.")
            return
        uids = [a["uid"] for a in animals]
        if len(uids) != len(set(uids)):
            messagebox.showerror("Internal Error", "Duplicate uid detected.")
            return
        egs = {a["exp_group"] for a in animals}
        if len(egs) < 1:
            messagebox.showwarning("No Exp Groups",
                "Assign experimental groups to animals first.")
            return
        try:
            self._combined = compute_combined(animals, groups)
        except Exception:
            messagebox.showerror("Compute Error", traceback.format_exc())
            return
        n_computed = len(self._combined["records"])
        if n_computed != len(animals):
            messagebox.showerror("Data Integrity Error",
                f"Expected {len(animals)} animals but got {n_computed}.")
            return
        try:
            eg_colors = self._animal_panel.get_eg_colors()
            self._clear_combined_plots()

            # Flush any pending GDI/Tk messages from the clear step before
            # we start creating new canvases — prevents object accumulation
            # that can overflow win32k.sys resources on Windows.
            try:
                self._combined_plot_frame.winfo_toplevel().update_idletasks()
            except Exception:
                pass

            # Ethogram: all animals coloured by experimental group
            fig_eth = build_combined_ethogram_figure(groups, self._combined)
            self._add_combined_figure(
                fig_eth,
                "Ethograms by Animal  (coloured by Experimental Group)")
            try:
                self._combined_plot_frame.winfo_toplevel().update_idletasks()
            except Exception:
                pass

            # One 2×2 metric figure + one transitions figure per behaviour group
            for bg_name in groups:
                fig_grp = build_combined_group_figure(
                    bg_name, groups, self._combined,
                    eg_colors_override=eg_colors)
                self._add_combined_figure(fig_grp)
                # Drain the Tk/GDI event queue between figures to release
                # win32k resources before the next canvas is drawn.
                try:
                    self._combined_plot_frame.winfo_toplevel().update_idletasks()
                except Exception:
                    pass

                fig_trans, trans_stats = build_combined_transition_figure(
                    bg_name, groups, self._combined,
                    eg_colors_override=eg_colors)
                if fig_trans is not None:
                    self._add_combined_figure(
                        fig_trans,
                        f"Transitions within '{bg_name}'"
                        f"  (top differences, BH-FDR corrected)")
                    self._combined_trans_figs_added.append(bg_name)
                    self._combined_trans_stats.append(trans_stats)
                try:
                    self._combined_plot_frame.winfo_toplevel().update_idletasks()
                except Exception:
                    pass

            # Scroll back to the top
            self._scroll_canvas.yview_moveto(0)
        except Exception:
            messagebox.showerror("Plot Error", traceback.format_exc())
            return
        eg_names = list(self._combined["exp_groups"].keys())
        n_ani    = len(self._combined["records"])
        self._status(
            f"Combined: {n_ani} animals  -  {len(eg_names)} exp group(s)  -  "
            f"{len(groups)} behaviour group(s)  ✓", "#88cc88")

    def _export_combined(self):
        if not self._combined:
            messagebox.showwarning("No data", "Run combined analysis first.")
            return
        if not self._root_dir:
            messagebox.showwarning("No folder", "Select a root folder first.")
            return
        out = self._root_dir / RESULTS_SUBDIR
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save every figure that was generated (ethogram + per-group metrics +
        # per-group transition figures interleaved in the same order as _run_combined)
        groups        = self._get_groups()
        _trans_added  = getattr(self, "_combined_trans_figs_added", [])
        bg_names      = ["ethogram"]
        for _bn in groups:
            bg_names.append(_bn)
            if _bn in _trans_added:
                bg_names.append(f"{_bn}_transitions")
        for i, fig in enumerate(getattr(self, "_combined_figs", [])):
            label = bg_names[i] if i < len(bg_names) else f"plot_{i}"
            safe  = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
            for ext in ("png", "pdf"):
                fig.savefig(out / f"combined_{safe}_{ts}.{ext}",
                            dpi=300 if ext == "png" else None,
                            bbox_inches="tight",
                            facecolor=fig.get_facecolor())
        grand  = self._combined["grand"]
        # Pre-compute p-values (BH FDR corrected) for all metric × beh_group combos
        pvals_all = compute_combined_pvals(self._combined, groups)
        STAR_THRESH = [(0.001, "***"), (0.01, "**"), (0.05, "*")]

        def _stars(qv):
            if qv is None:
                return "N/A"
            for thresh, sym in STAR_THRESH:
                if qv < thresh:
                    return sym
            return "ns"

        rows = []
        for eg_name, beh_dict in grand.items():
            for beh_group in groups:
                for metric in ("total_duration", "frequency", "latency", "mean_bout"):
                    d       = beh_dict[beh_group][metric]
                    pv_dict = pvals_all.get((metric, beh_group)) or {}
                    pv      = pv_dict.get("pval")
                    qv      = pv_dict.get("qval")
                    rows.append({
                        "Exp_Group":        eg_name,
                        "Beh_Group":        beh_group,
                        "Metric":           metric,
                        "Mean":             round(d["mean"], 4),
                        "SEM":              round(d["sem"],  4),
                        "N":                d["n"],
                        "p_value_KW":       round(pv, 6) if pv is not None else "N/A",
                        "q_value_FDR":      round(qv, 6) if qv is not None else "N/A",
                        "Significance_FDR": _stars(qv),
                    })
        pd.DataFrame(rows).to_csv(out / f"combined_summary_{ts}.csv", index=False)
        _trans_stats_list = getattr(self, "_combined_trans_stats", [])
        if _trans_stats_list:
            pd.concat(_trans_stats_list, ignore_index=True).to_csv(
                out / f"combined_top10_transitions_{ts}.csv", index=False)
        n_saved = len(getattr(self, "_combined_figs", []))
        messagebox.showinfo("Exported",
            f"Saved {n_saved} figure(s) + CSV summary to:\n{out}")

    #   save all results (all tabs)

    def _save_all_results(self):
        """Save every generated graph and data table from all tabs to one folder."""
        directory = filedialog.askdirectory(
            title="Save All Results — Choose Output Folder")
        if not directory:
            return
        out_root = pathlib.Path(directory)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved   = []
        skipped = []

        # ── Single-animal figures ─────────────────────────────────────────
        out_single   = out_root / "single_animal"
        preview_fig  = self._preview_panel.get_figure()
        analysis_fig = self._analysis_panel.get_figure()
        if preview_fig or analysis_fig or self._metrics:
            out_single.mkdir(parents=True, exist_ok=True)
            n_figs = 0
            if preview_fig:
                preview_fig.savefig(
                    str(out_single / f"preview_{ts}.png"),
                    dpi=200, bbox_inches="tight",
                    facecolor=preview_fig.get_facecolor())
                n_figs += 1
            if analysis_fig:
                for ext in ("png", "pdf"):
                    analysis_fig.savefig(
                        str(out_single / f"analysis_{ts}.{ext}"),
                        dpi=300 if ext == "png" else None,
                        bbox_inches="tight",
                        facecolor=analysis_fig.get_facecolor())
                n_figs += 1
            if self._metrics:
                groups = self._get_groups()
                if groups:
                    rows = [
                        {"Group":              gname,
                         "Labels":             ",".join(str(l) for l in groups[gname]["labels"]),
                         "Color":              groups[gname]["color"],
                         "Total Duration (s)": m["total_duration"],
                         "Frequency":          m["frequency"],
                         "Latency (s)":        m["latency"] if m["latency"] is not None else "N/A",
                         "Mean Bout (s)":      m["mean_bout"]}
                        for gname, m in self._metrics.items()
                    ]
                    pd.DataFrame(rows).to_csv(
                        out_single / f"metrics_{ts}.csv", index=False)
            saved.append(f"Single Animal ({n_figs} figure(s))")
        else:
            skipped.append("Single Animal — run 'Analyse & Preview' first")

        # ── Combined Analysis figures ─────────────────────────────────────
        out_combined  = out_root / "combined_analysis"
        combined_figs = getattr(self, "_combined_figs", [])
        if combined_figs and self._combined:
            out_combined.mkdir(parents=True, exist_ok=True)
            groups       = self._get_groups()
            _trans_added = getattr(self, "_combined_trans_figs_added", [])
            bg_names     = ["ethogram"]
            for _bn in groups:
                bg_names.append(_bn)
                if _bn in _trans_added:
                    bg_names.append(f"{_bn}_transitions")
            for i, fig in enumerate(combined_figs):
                label = bg_names[i] if i < len(bg_names) else f"plot_{i}"
                safe  = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in label)
                for ext in ("png", "pdf"):
                    fig.savefig(
                        str(out_combined / f"combined_{safe}.{ext}"),
                        dpi=300 if ext == "png" else None,
                        bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            # p-value CSV (BH FDR corrected across all metric × beh_group tests)
            pvals_all   = compute_combined_pvals(self._combined, groups)
            STAR_THRESH = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
            def _stars(qv):
                if qv is None:
                    return "N/A"
                for thresh, sym in STAR_THRESH:
                    if qv < thresh:
                        return sym
                return "ns"
            grand = self._combined["grand"]
            rows  = []
            for eg_name, beh_dict in grand.items():
                for beh_group in groups:
                    for metric in ("total_duration", "frequency", "latency", "mean_bout"):
                        d       = beh_dict[beh_group][metric]
                        pv_dict = pvals_all.get((metric, beh_group)) or {}
                        pv      = pv_dict.get("pval")
                        qv      = pv_dict.get("qval")
                        rows.append({
                            "Exp_Group":        eg_name,
                            "Beh_Group":        beh_group,
                            "Metric":           metric,
                            "Mean":             round(d["mean"], 4),
                            "SEM":              round(d["sem"],  4),
                            "N":                d["n"],
                            "p_value_KW":       round(pv, 6) if pv is not None else "N/A",
                            "q_value_FDR":      round(qv, 6) if qv is not None else "N/A",
                            "Significance_FDR": _stars(qv),
                        })
            pd.DataFrame(rows).to_csv(
                out_combined / f"combined_summary_{ts}.csv", index=False)
            _trans_stats_list = getattr(self, "_combined_trans_stats", [])
            if _trans_stats_list:
                pd.concat(_trans_stats_list, ignore_index=True).to_csv(
                    out_combined / f"combined_top10_transitions_{ts}.csv", index=False)
            saved.append(f"Combined Analysis ({len(combined_figs)} figure(s) + CSV)")
        else:
            skipped.append("Combined Analysis — run 'Run Combined Analysis' first")

        # ── Unbiased Analytics ────────────────────────────────────────────
        out_unbiased = out_root / "unbiased_analytics"
        try:
            self._unbiased_panel._save_all_graphs(out_dir=out_unbiased)
            n_csv = self._unbiased_panel.export_csv_data(out_unbiased)
            saved.append(f"Unbiased Analytics ({n_csv} CSV(s))")
        except Exception as exc:
            skipped.append(f"Unbiased Analytics — {exc}")

        # ── Group Predictor ───────────────────────────────────────────────
        out_predictor = out_root / "group_predictor"
        try:
            n_figs = self._predictor_panel.save_all_figures(out_predictor, ts)
            n_csv  = self._predictor_panel.export_csv_data(out_predictor)
            if n_figs or n_csv:
                saved.append(
                    f"Group Predictor ({n_figs} figure(s), {n_csv} CSV(s))")
            else:
                skipped.append("Group Predictor — run the predictor first")
        except Exception as exc:
            skipped.append(f"Group Predictor — {exc}")

        # ── Behavioral Explorer ───────────────────────────────────────────
        out_explorer = out_root / "behavioral_explorer"
        try:
            n_figs = self._explorer_panel.save_all_figures(out_explorer, ts)
            n_csv  = self._explorer_panel.export_csv_data(out_explorer)
            if n_figs or n_csv:
                saved.append(
                    f"Behavioral Explorer ({n_figs} figure(s), {n_csv} CSV(s))")
            else:
                skipped.append("Behavioral Explorer — generate a plot first")
        except Exception as exc:
            skipped.append(f"Behavioral Explorer — {exc}")

        # ── Summary message ───────────────────────────────────────────────
        msg = f"Results saved to:\n{out_root}\n"
        if saved:
            msg += "\nSaved:\n" + "".join(f"  ✓  {s}\n" for s in saved)
        if skipped:
            msg += "\nSkipped (no data yet):\n" + "".join(f"  -  {s}\n" for s in skipped)
        messagebox.showinfo("Save All Results", msg)
        self._status(f"All results saved → {out_root}", "#88cc88")

    #   save (single animal)

    def _save(self):
        if not self._root_dir:
            messagebox.showwarning("No folder", "Select a root folder first.")
            return
        if not self._metrics:
            messagebox.showwarning("No analysis", "Run 'Analyse & Preview' first.")
            return
        groups = self._get_groups()
        if not groups:
            return
        fig = self._analysis_panel.get_figure()
        if fig is None:
            messagebox.showwarning("No figure", "No analysis figure to save.")
            return
        try:
            out = export_results(
                self._root_dir, groups, self._metrics,
                fig, self._active_csv, self._get_fps(),
                out_override=getattr(self, "_comparison_plot_dir", None),
            )
            messagebox.showinfo("Saved", f"Results saved to:\n{out}")
            self._status(f"Saved -> {out}", "#88cc88")
        except Exception:
            messagebox.showerror("Save Error", traceback.format_exc())

    #   metrics table  

    def _build_metrics_table(self, groups, metrics):
        for w in self._metrics_frame.winfo_children():
            w.destroy()
        self._metrics_frame.columnconfigure(0, weight=1)
        self._metrics_frame.rowconfigure(1, weight=1)

        cols       = ["Group", "Labels", "Total (s)", "Frequency",
                      "Latency (s)", "Mean Bout (s)"]
        col_widths = [120, 210, 90, 80, 95, 105]

        hdr = ctk.CTkFrame(self._metrics_frame, fg_color=T()["hdr_bg"])
        hdr.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        for j, (col, w) in enumerate(zip(cols, col_widths)):
            ctk.CTkLabel(hdr, text=col, width=w,
                         font=ctk.CTkFont(weight="bold"),
                         text_color=T()["hdr_text"],
                         ).grid(row=0, column=j, padx=4, pady=6)

        scroll = ctk.CTkScrollableFrame(self._metrics_frame, fg_color=T()["panel"])
        scroll.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        for i, (gname, m) in enumerate(metrics.items()):
            fg = T()["row_even"] if i % 2 == 0 else T()["row_odd"]
            rf = ctk.CTkFrame(scroll, fg_color=fg, corner_radius=4)
            rf.pack(fill="x", pady=1, padx=2)
            color  = groups[gname]["color"]
            labels = ",".join(str(l) for l in groups[gname]["labels"])
            vals   = [
                gname, labels, str(m["total_duration"]), str(m["frequency"]),
                str(m["latency"]) if m["latency"] is not None else "-",
                str(m["mean_bout"]),
            ]
            for j, (val, w) in enumerate(zip(vals, col_widths)):
                ctk.CTkLabel(
                    rf, text=val, width=w, anchor="w",
                    text_color=color if j == 0 else T()["text"],
                ).grid(row=0, column=j, padx=4, pady=5, sticky="w")

    #   theme toggle  

    def _toggle_theme(self, value: str):
        global _THEME_KEY
        _THEME_KEY = value.lower()
        ctk.set_appearance_mode(T()["ctk_mode"])
        try:
            plt.style.use(T()["mpl_style"])
        except Exception:
            pass
        self._status("Theme switched.", T()["subtext"])
        self._build_combined_tab()
        if self._metrics and self._df is not None:
            self._analyse()
        self._refresh_preview()

    #   helpers  

    def _get_fps(self) -> int:
        try:
            v = int(self._fps_var.get())
            return v if v > 0 else DEFAULT_FPS
        except ValueError:
            return DEFAULT_FPS

    def _get_groups(self) -> dict:
        if self._editor and self._editor.winfo_exists():
            return self._editor.get_groups()
        return {}

    def _status(self, msg: str, color: str = ""):
        self._status_lbl.configure(text=msg,
                                   text_color=color or T()["subtext"])
        self.update_idletasks()


#  
# ENTRY POINT
#  

def main():
    app = BSOiDApp()
    app.mainloop()


if __name__ == "__main__":
    main()