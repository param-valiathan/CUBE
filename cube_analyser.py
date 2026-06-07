# -*- coding: utf-8 -*-
"""
Created on Sat May 16 01:16:59 2026

@author: param


B-SOiD Behavioral Analysis Suite  v6.0
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
from matplotlib.gridspec import GridSpec
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
APP_TITLE      = "CUBE Behavioral Analysis Suite  v6"
BSOID_SUBDIR   = "BSOID"
RESULTS_SUBDIR = "Analysis_Results"

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
        "btn_del":     "#ffcccc",
        "btn_del_h":   "#ffaaaa",
        "btn_add":     "#ccf0cc",
        "btn_folder":  "#cce0f8",
        "btn_save":    "#ccf0d8",
        "btn_load":    "#dde8f8",
        "btn_unbiased":"#e8d8f8",
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
    Search recursively for umap_embedding.npy and umap_labels.npy saved by cube_core.
    Returns (embedding_path, labels_path) or (None, None).
    """
    emb_path = lbl_path = None
    for p in sorted(root.rglob("umap_embedding.npy")):
        emb_path = p
        break
    for p in sorted(root.rglob("umap_labels.npy")):
        lbl_path = p
        break
    if emb_path and lbl_path:
        return emb_path, lbl_path
    return None, None


def load_csv(path: pathlib.Path) -> pd.DataFrame:
    """Load a B-SOiD bout_lengths CSV (original format). Raises ValueError on error."""
    required = {"B-SOiD labels", "Start time (frames)", "Run lengths"}
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
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


#  
# STANDARD BOUT METRICS
#  

def merge_bouts(df: pd.DataFrame, labels: set) -> list:
    events, cur = [], None
    for _, row in df.iterrows():
        if row["label"] in labels:
            end = row["start_frame"] + row["run_len"]
            if cur is None:
                cur = {"start": int(row["start_frame"]), "end": end}
            elif row["start_frame"] <= cur["end"] + 1:
                cur["end"] = max(cur["end"], end)
            else:
                events.append(cur)
                cur = {"start": int(row["start_frame"]), "end": end}
    if cur:
        events.append(cur)
    return events


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

def run_cluster_statistics(animal_data: list, metric: str = "total_duration") -> pd.DataFrame:
    """
    Run Kruskal-Wallis (non-parametric, robust for small n) and ANOVA across
    experimental groups for every cluster present in the data.

    Returns DataFrame sorted by p-value:
      cluster_id, stat_kw, pval_kw, stat_anova, pval_anova,
      effect_size_eta2, direction, group_means...
    """
    if not SCIPY_OK:
        raise RuntimeError("scipy not available. pip install scipy")

    # Build per-cluster, per-group value lists
    eg_map: dict = {}      # exp_group -> list of per_cluster DataFrames
    for ani in animal_data:
        eg  = ani["exp_group"]
        pcm = compute_per_cluster_metrics(ani["df"], ani["fps"])
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
        for eg in eg_names:
            vs = []
            for pcm in eg_map[eg]:
                if cid in pcm.index:
                    vs.append(float(pcm.loc[cid, metric]))
                else:
                    vs.append(0.0)
            group_vals[eg] = vs

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

        # Eta-squared effect size from ANOVA
        try:
            grand_mean = np.mean(all_vals)
            ss_between = sum(len(vs) * (np.mean(vs) - grand_mean)**2 for vs in groups_for_test)
            ss_total   = sum((v - grand_mean)**2 for v in all_vals)
            eta2       = ss_between / ss_total if ss_total > 0 else 0.0
        except Exception:
            eta2 = 0.0

        row = {
            "cluster_id":       cid,
            "stat_kw":          stat_kw,
            "pval_kw":          pval_kw,
            "stat_anova":       stat_an,
            "pval_anova":       pval_an,
            "effect_size_eta2": eta2,
            "neg_log10_p":      -np.log10(max(pval_kw, 1e-300)),
        }
        for eg in eg_names:
            row[f"mean_{eg}"] = float(np.mean(group_vals[eg]))
        rows.append(row)

    df_res = pd.DataFrame(rows).sort_values("pval_kw").reset_index(drop=True)
    return df_res


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
    Edges above *threshold_pct* percentile are shown (default top 10 %).
    Use a lower threshold_pct (e.g. 20) for graphs with few nodes so more
    edges are visible.  Edges are sorted weakest-first so strongest render on top.
    """
    nn   = len(net_labels)
    flat = net_tmat[net_tmat > 0]
    if len(flat) == 0:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center",
                color=t["tick"], transform=ax.transAxes, fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_facecolor(t["ax_bg"])
        return

    edge_thresh = float(np.percentile(flat, threshold_pct)) if threshold_pct > 0 else 0.0
    flat_max    = float(flat.max())
    if flat_max <= edge_thresh:
        edge_thresh = float(flat.min())

    # Collect edges above threshold and sort weakest first (strongest drawn on top)
    edge_list = []
    for i in range(nn):
        for j in range(nn):
            if i == j:
                continue
            p = float(net_tmat[i, j])
            if p >= edge_thresh:
                edge_list.append((p, i, j))
    edge_list.sort()

    for p, i, j in edge_list:
        p_norm = (p - edge_thresh) / max(flat_max - edge_thresh, 1e-9)
        lw     = 0.8 + p_norm ** 0.45 * 11        # ~0.8 – 11.8 px
        alpha  = 0.55 + p_norm * 0.45             # ~0.55 – 1.0  (no fuzzy dim arrows)
        mscale = 10 + p_norm * 14                 # arrowhead size
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
                          groups: dict = None) -> plt.Figure:
    """
    Volcano plot: effect size (eta²) vs -log10(p-value).
    Pass groups to colour-code each cluster by user-defined behaviour group.
    """
    if t is None:
        t = T()

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

        for _, row in stats_df.iterrows():
            cid  = int(row["cluster_id"])
            x    = row["effect_size_eta2"]
            y    = row["neg_log10_p"]
            sig  = (row["pval_kw"] < p_thresh)
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
            if top or in_grp:
                label = f"C{cid}"
                if in_grp:
                    label += f"\n({cid_to_group[cid][0]})"
                ax.annotate(label, (x, y), fontsize=6,
                            color=t["tick"], textcoords="offset points",
                            xytext=(4, 4))

        ax.axhline(neg_log_thresh, color="#888888", lw=1, ls="--", alpha=0.7,
                   label=f"p = {p_thresh}")
        ax.set_xlabel("Effect Size (η²)", color=t["tick"])
        ax.set_ylabel("−log₁₀(p-value)", color=t["tick"])
        ax.set_title("Volcano Plot — Cluster Significance", color=t["tick"],
                     fontweight="bold")

        legend_handles = [
            mpatches.Patch(color="#FF4081", label=f"Top {top_n} clusters"),
            mpatches.Patch(color="#FFC107", label=f"Significant (p < {p_thresh})"),
            mpatches.Patch(color=t["muted"],  label="Not significant"),
        ]
        if groups:
            for gname, ginfo in groups.items():
                legend_handles.append(
                    mpatches.Patch(color=ginfo["color"], label=f"Group: {gname}",
                                   hatch="//"))
        ax.legend(handles=legend_handles, fontsize=8, framealpha=0.3,
                  facecolor=t["ax_bg"], labelcolor=t["tick"])
        _style_ax(ax, t)
        fig.tight_layout()
    return fig


def build_heatmap_figure(stats_df: pd.DataFrame, animal_data: list,
                          top_n: int, p_thresh: float,
                          metric: str = "total_duration",
                          eg_colors_override: dict = None,
                          t: dict = None) -> plt.Figure:
    """
    Ordered hierarchical heatmap with dendrogram.
    Rows = top-N significant clusters, cols = animals (grouped by exp_group).
    Colour strip above columns uses user-selected EG colours.
    """
    if t is None:
        t = T()
    if not SCIPY_CLUSTER_OK:
        raise RuntimeError("scipy.cluster required.")

    # Filter to significant top-N
    sig_df  = stats_df[stats_df["pval_kw"] < p_thresh].head(top_n)
    top_ids = sig_df["cluster_id"].tolist()
    if not top_ids:
        top_ids = stats_df.head(min(top_n, len(stats_df)))["cluster_id"].tolist()

    # Build matrix: rows=clusters, cols=animals
    animals_sorted = sorted(animal_data, key=lambda a: a["exp_group"])
    n_animals = len(animals_sorted)
    mat = np.zeros((len(top_ids), n_animals))
    for ci, cid in enumerate(top_ids):
        for ai, ani in enumerate(animals_sorted):
            pcm = compute_per_cluster_metrics(ani["df"], ani["fps"])
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

    with plt.style.context(t["mpl_style"]):
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=t["fig_bg"])
        # 3-column GridSpec: dendrogram | heatmap | colorbar
        # Colorbar gets its own narrow column so it never steals space from ax_heat
        dend_frac  = 1
        heat_frac  = max(n_animals * 2, 10)
        cbar_frac  = 1
        gs = GridSpec(2, 3, figure=fig,
                      height_ratios=[1, 9],
                      width_ratios=[dend_frac, heat_frac, cbar_frac],
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
        ax_heat.set_yticklabels([f"C{cid}" for cid in ids_ordered],
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

        # Colorbar in its own column — does NOT touch ax_heat width
        ax_cbar = fig.add_subplot(gs[1, 2])
        cbar = fig.colorbar(im, cax=ax_cbar)
        cbar.set_label("Z-score", color=t["tick"], fontsize=8)
        cbar.ax.tick_params(colors=t["tick"], labelsize=7)
        cbar.outline.set_edgecolor(t["spine"])

        fig.suptitle(f"Hierarchical Heatmap — Top {len(ids_ordered)} Significant Clusters  "
                     f"(p < {p_thresh}, {metric})",
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
        bars = ax2.bar(ks_sil, sv, color=colors, edgecolor=t["border"], width=0.7)
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
                       color=color, alpha=0.82, edgecolor=t["border"],
                       linewidth=0.6, label=eg, hatch=hatch, zorder=2)
                ax.errorbar(x_base + offset, norm_means, yerr=norm_sems,
                            fmt="none", color=t["tick"], lw=1.2, capsize=3, zorder=3)
                ax.scatter(pts_x_all, pts_y_all,
                           color="white", edgecolors=color,
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

    sig_df  = stats_df[stats_df["pval_kw"] < p_thresh].head(top_n)
    top_ids = sig_df["cluster_id"].tolist()
    if not top_ids:
        top_ids = stats_df.head(min(top_n, len(stats_df)))["cluster_id"].tolist()
    if not top_ids:
        fig, ax = plt.subplots(figsize=(7, 3), facecolor=t["fig_bg"] if t else "#0d0d1a")
        ax.text(0.5, 0.5, "No clusters to display.", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    eg_map: dict = {}
    for ani in animal_data:
        eg = ani["exp_group"]
        pcm = compute_per_cluster_metrics(ani["df"], ani["fps"])
        eg_map.setdefault(eg, []).append(pcm)
    eg_names = list(eg_map.keys())
    eg_colors: dict = {}
    for i, eg in enumerate(eg_names):
        if eg_colors_override and eg in eg_colors_override:
            eg_colors[eg] = eg_colors_override[eg]
        else:
            eg_colors[eg] = PALETTE[i % len(PALETTE)]

    pvals_map = stats_df.set_index("cluster_id")["pval_kw"].to_dict()

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
                       color=ec, alpha=0.82, edgecolor=t["border"],
                       linewidth=0.6, zorder=2)
                ax.errorbar(x_pos[ei], means[ei], yerr=sems[ei], fmt="none",
                            color=t["tick"], lw=1.4, capsize=3, zorder=3)
                jit = rng.uniform(-0.12, 0.12, len(all_pts[ei]))
                ax.scatter(x_pos[ei] + jit, all_pts[ei],
                           color="white", edgecolors=ec,
                           s=22, linewidths=0.7, alpha=0.88, zorder=4)

            # significance bracket + stars — always shown (inc. ns)
            stars = ("***" if pv < 0.001 else "**" if pv < 0.01
                     else "*" if pv < 0.05 else "ns")
            data_top = max((m + s for m, s in zip(means, sems)), default=0)
            max_pt   = max((max(pts) for pts in all_pts if pts), default=0)
            head_top = max(data_top * 1.40, max_pt * 1.40, 0.001)
            ax.set_ylim(bottom=0, top=head_top)
            if n_eg >= 2:
                bh = head_top * 0.78
                ax.plot([0, 0, n_eg - 1, n_eg - 1],
                        [bh, bh + head_top * 0.04, bh + head_top * 0.04, bh],
                        lw=0.9, color=t["tick"])
                star_color = "#FF4081" if stars != "ns" else t["muted"]
                ax.text((n_eg - 1) / 2, bh + head_top * 0.055, stars,
                        ha="center", fontsize=10, color=star_color,
                        fontweight="bold" if stars != "ns" else "normal")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(eg_names, rotation=22, ha="right",
                               color=t["tick"], fontsize=7)
            ax.set_title(f"C{cid}", color=t["tick"],
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
        fig.suptitle(
            f"Top {len(top_ids)} Significant Clusters  (p < {p_thresh})  -  "
            f"Each panel has its own y-scale  -  * p<0.05  ** p<0.01  *** p<0.001",
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
                pv2 = pvals_bg.get((metric, bg), 1.0)
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
                            color=ec2, edgecolor=bg_color,
                            linewidth=1.2, alpha=0.82, zorder=2)
                    ax2.errorbar(x2[ei], means2[ei], yerr=sems2[ei],
                                 fmt="none", color=t["tick"], lw=1.2, capsize=3, zorder=3)
                    jit2 = rng2.uniform(-0.12, 0.12, len(pts2[ei]))
                    ax2.scatter(x2[ei] + jit2, pts2[ei], color="white",
                                edgecolors=ec2, s=22, linewidths=0.7,
                                alpha=0.88, zorder=4)

                stars2 = ("***" if pv2 < 0.001 else "**" if pv2 < 0.01
                          else "*" if pv2 < 0.05 else "ns")
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
                f"  —  * p<0.05  ** p<0.01  *** p<0.001",
                color=t["tick"], fontweight="bold", fontsize=10, y=0.99)
            fig2.tight_layout(rect=[0, 0, 1, 0.97])
        return fig, fig2

    return fig


def build_distance_matrix_figure(recluster_result: dict, t: dict = None) -> plt.Figure:
    """
    3-panel figure:
      Left:  Blended distance matrix (main clustering input)
      Middle: Correlation-distance matrix (change-pattern similarity)
      Right:  Transition-distance matrix (temporal co-occurrence)
    Rows/cols ordered by the linkage dendrogram so structure is visible.
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

    labels_ord = [f"C{c}" for c in [cids[i] for i in order]]

    def _reorder(m):
        return m[np.ix_(order, order)]

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
            ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels,
                rotation=70, fontsize=max(4, 7 - n//20), color=t["tick"])
            ax.set_yticks(tick_pos); ax.set_yticklabels(tick_labels,
                fontsize=max(4, 7 - n//20), color=t["tick"])
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


def build_transition_figure(recluster_result: dict, t: dict = None) -> plt.Figure:
    """
    Visualise transition dynamics — three panels:
      Top-left:  Average transition probability heatmap (row-stochastic)
      Top-right: Transition-profile similarity (cosine similarity)
      Bottom:    Circular transition network — directed arrows sized by probability
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

    labels_ord = [f"C{cids_all[i]}" for i in order]
    tmat_ord   = tmat_sub[np.ix_(order, order)]
    sim_ord    = sim_sub[np.ix_(order, order)]
    n          = len(labels_ord)

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
        fsize     = max(4, 7 - n // 20)

        for ax, mat, title, cmap in [
            (ax1, tmat_ord, "Average Transition Probabilities\n(row-stochastic, diagonal excluded)", "magma"),
            (ax2, sim_ord,  "Transition-Profile Similarity\n(cosine similarity of outgoing profiles)", "viridis"),
        ]:
            display = mat.copy()
            if ax is ax1:
                np.fill_diagonal(display, 0)
            im = ax.imshow(display, cmap=cmap, aspect="auto",
                           interpolation="nearest", vmin=0)
            ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lbls,
                rotation=70, fontsize=fsize, color=t["tick"])
            ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lbls,
                fontsize=fsize, color=t["tick"])
            ax.set_title(title, color=t["tick"], fontsize=9, fontweight="bold")
            cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
            cb.ax.tick_params(colors=t["tick"], labelsize=7)
            cb.outline.set_edgecolor(t["spine"])
            _style_ax(ax, t)

        # ── Circular transition network ───────────────────────────────────────
        max_net    = min(n, 30)
        net_order  = list(range(max_net))
        net_labels = [labels_ord[i] for i in net_order]
        net_tmat   = tmat_ord[np.ix_(net_order, net_order)]
        np.fill_diagonal(net_tmat, 0)

        nn     = len(net_labels)
        theta  = np.linspace(0, 2 * np.pi, nn, endpoint=False) - np.pi / 2
        nx_pos = np.cos(theta)
        ny_pos = np.sin(theta)

        node_colors = [PALETTE[i % len(PALETTE)] for i in range(nn)]
        _draw_circle_network(ax3, net_tmat, nx_pos, ny_pos,
                             net_labels, node_colors, t)
        ax3.set_title(
            f"Transition Network  (top 10 % of edges shown, n={nn} clusters)\n"
            "Node size = total outgoing strength · Arrow width = transition probability",
            color=t["tick"], fontsize=9, fontweight="bold")

        fig.suptitle(
            "Transition Dynamics  (heatmaps ordered by dendrogram)\n"
            "Clusters that frequently transition to the same targets are pulled together in reclustering",
            color=t["tick"], fontweight="bold", fontsize=11)
    return fig


def build_cluster_stats_figure(recluster_result: dict, animal_data: list,
                                t: dict = None) -> plt.Figure:
    """
    Per-cluster statistics overview: one row per metric.
    Each cluster is a point; colour = cluster ID (cycle through PALETTE).
    Panels:
      - Mean bout duration (ms)
      - Total duration (s) per animal mean
      - Frequency (bouts) per animal mean
      - Mean transition entropy (how diverse are outgoing transitions?)
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
            edgec = [PALETTE[i % len(PALETTE)] for i in range(len(cids_sorted))]
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
        axes[-1].set_xticklabels(labels_plot, rotation=60, ha="right",
                                  fontsize=max(5, 8 - len(all_raw) // 15),
                                  color=t["tick"])

        legend_handles = [
            mpatches.Patch(color=PALETTE[0], alpha=0.9, label="Included in reclustering"),
            mpatches.Patch(color=PALETTE[0], alpha=0.25, label="Filtered out (below threshold)",
                           fill=False, edgecolor=PALETTE[0], linewidth=1.2),
        ]
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
    bars   = ax.barh(gnames, durs, color=colors, edgecolor=t["border"], linewidth=0.6)
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
    bars = ax.bar(gnames, lats, color=colors, edgecolor=t["border"], linewidth=0.6)
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
    bars   = ax.bar(gnames, freqs, color=colors, edgecolor=t["border"], linewidth=0.6)
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
    ax.bar(gnames, means, color=colors, edgecolor=t["border"], linewidth=0.6)
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
    Compute Kruskal-Wallis p-values for every (metric_key, beh_group) pair.
    Returns {(metric_key, beh_group): p_value_or_None}.
    """
    if not SCIPY_OK:
        return {}
    records    = combined["records"]
    uid_to_idx = combined["uid_to_idx"]
    exp_groups = combined["exp_groups"]
    eg_names   = list(exp_groups.keys())
    pvals: dict = {}
    for metric_key in ("total_duration", "frequency", "latency", "mean_bout"):
        for bg in groups:
            if len(eg_names) < 2:
                pvals[(metric_key, bg)] = None
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
            pvals[(metric_key, bg)] = pv
    return pvals


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

            eg_color = PALETTE[ei % len(PALETTE)]
            vmax     = float(disp.max()) if disp.max() > 0 else 1.0
            im = ax_eg.imshow(disp, cmap="magma", aspect="auto",
                              interpolation="nearest", vmin=0, vmax=vmax)
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
            "(row-stochastic, diagonal = 0 — same cluster order as Transitions view)",
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
    node_colors = [PALETTE[i % len(PALETTE)] for i in range(nn)]
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
            "Shared cluster layout · Arrow width = transition probability  "
            "(top 10 % of edges shown)",
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
            _draw_circle_network(ax_eg, gmat, nx_pos, ny_pos,
                                 group_names, group_colors, t,
                                 threshold_pct=20)   # low threshold — few nodes, show most edges
            eg_color = PALETTE[ei % len(PALETTE)]
            ax_eg.set_title(label, color=eg_color, fontsize=10, fontweight="bold")

        for ei in range(len(panels), len(axes_flat)):
            axes_flat[ei].set_visible(False)

        fig.suptitle(
            "Behaviour Group Transition Network\n"
            "Each node = one user-defined behaviour group  ·  "
            "Arrow width = aggregate transition probability  (top 80 % of edges shown)",
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
        """Average row-stochastic transition matrix over *ani_list*."""
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
        return tmat

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
                 "magma", 0, None, "P(transition)"),
                (tmats[eg_b], f"{eg_b}",
                 "magma", 0, None, "P(transition)"),
                (tmats[eg_b] - tmats[eg_a], f"Δ = {eg_b} − {eg_a}",
                 "RdBu_r", None, None, "ΔP"),
            ]):
                ax = axes[row, col_i]
                vmax_use = float(np.abs(mat).max()) if vmax is None else vmax
                vmin_use = -vmax_use if col_i == 2 else 0.0
                im = ax.imshow(mat, cmap=cmap_name, aspect="auto",
                               interpolation="nearest",
                               vmin=vmin_use, vmax=vmax_use)
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
            ax.set_yscale("log")
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
        n_steps: int = 5) -> plt.Figure:
    """
    Sankey (alluvial) flow diagram per experimental group.
    Each column is a consecutive bout position; ribbons show how the
    population flows from state to state across the sequence.
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

    def _bout_seqs(ani_list):
        seqs = []
        for ani in ani_list:
            seq = ani["df"].sort_values("start_frame")["label"].values
            if len(seq) >= 2:
                seqs.append([int(x) for x in seq])
        return seqs

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

            seqs   = _bout_seqs(groups_eg[eg])
            if not seqs:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        color=t["tick"], transform=ax.transAxes)
                ax.set_title(f"{eg}", color=eg_color, fontsize=10,
                             fontweight="bold")
                continue

            n_use = min(n_steps, min(len(s) for s in seqs))
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
                for a in range(ns):
                    sh = y_top[k, a] - y_bot[k, a]
                    for b in range(ns):
                        p = float(trans[k, a, b])
                        if p < 0.01:
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
                        edgecolor=t["ax_bg"], linewidth=0.5, zorder=3)
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

            ax.set_xlim(-0.04, 1.06)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{eg}  (n = {len(groups_eg[eg])})",
                         color=eg_color, fontsize=10, fontweight="bold",
                         pad=4)

        for ei in range(n_eg, nrows * ncols):
            r, c = divmod(ei, ncols)
            axes[r][c].set_visible(False)

    fig.suptitle(
        f"Behavioral Sequence Sankey  (first {n_steps} bout positions)\n"
        "Bar height = state occupancy  ·  Ribbon width = transition flow",
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
            ax.set_yscale("log")
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
        n_steps: int = 5) -> plt.Figure:
    """
    Sankey (alluvial) flow diagram per experimental group using user-defined
    behavioural groups as states.  Cluster labels are mapped to groups first.
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

    def _bout_seqs_grp(ani_list):
        seqs = []
        for ani in ani_list:
            seq  = ani["df"].sort_values("start_frame")["label"].values
            gseq = [cid_to_gi[int(x)] for x in seq if int(x) in cid_to_gi]
            if len(gseq) >= 2:
                seqs.append(gseq)
        return seqs

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

            seqs = _bout_seqs_grp(groups_eg[eg])
            if not seqs:
                ax.text(0.5, 0.5, "No data (no clusters assigned to groups?)",
                        ha="center", va="center", color=t["tick"],
                        transform=ax.transAxes)
                ax.set_title(f"{eg}", color=eg_color,
                             fontsize=10, fontweight="bold")
                continue

            n_use = min(n_steps, min(len(s) for s in seqs))
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
                for a in range(ng):
                    sh = y_top[k, a] - y_bot[k, a]
                    for b in range(ng):
                        p = float(trans[k, a, b])
                        if p < 0.01:
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
            ax.set_xlim(-0.04, 1.06)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{eg}  (n = {len(groups_eg[eg])})",
                         color=eg_color, fontsize=10, fontweight="bold", pad=4)

        for ei in range(n_eg, nrows * ncols):
            r, c = divmod(ei, ncols)
            axes[r][c].set_visible(False)

    fig.suptitle(
        f"Behaviour Group Sequence Sankey  (first {n_steps} bout positions)  "
        "[User-Defined Groups]\n"
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

        for ei, eg in enumerate(eg_names):
            ax_eg    = axes_flat[ei]
            gmat     = _build_gmat(groups_eg[eg])
            eg_color = PALETTE[ei % len(PALETTE)]
            vmax     = float(gmat.max()) if gmat.max() > 0 else 1.0

            im = ax_eg.imshow(gmat, cmap="magma", aspect="auto",
                              interpolation="nearest", vmin=0, vmax=vmax)
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
            cb.set_label("P(transition)", color=t["tick"], fontsize=7)
            _style_ax(ax_eg, t)

        for ei in range(n_eg, len(axes_flat)):
            axes_flat[ei].set_visible(False)

        fig.suptitle(
            "Behaviour Group Transition Probabilities by Experimental Group  "
            "[User-Defined Groups]\n"
            "(row-stochastic, diagonal = 0  ·  nodes are user-defined groups)",
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
        return tmat

    tmats  = {eg: _group_tmat(groups_eg[eg]) for eg in eg_names}
    ref_eg = ctrl_group if (ctrl_group and ctrl_group in tmats) else eg_names[0]
    pairs  = [(ref_eg, eg) for eg in eg_names if eg != ref_eg]
    n_pairs = len(pairs)

    cell  = max(3.5, ng * 0.45 + 2)
    fsize = max(5, 9 - ng // 5)

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(
            n_pairs, 3,
            figsize=(cell * 3 + 1.0, cell * n_pairs + 1.2),
            facecolor=t["fig_bg"], squeeze=False)

        for row, (eg_a, eg_b) in enumerate(pairs):
            col_a = PALETTE[eg_names.index(eg_a) % len(PALETTE)]
            col_b = PALETTE[eg_names.index(eg_b) % len(PALETTE)]

            for col_i, (mat, title, cmap_name, cblabel) in enumerate([
                (tmats[eg_a], f"{eg_a}  (reference)", "magma", "P(transition)"),
                (tmats[eg_b], f"{eg_b}",              "magma", "P(transition)"),
                (tmats[eg_b] - tmats[eg_a],
                 f"Δ = {eg_b} − {eg_a}", "RdBu_r", "ΔP"),
            ]):
                ax = axes[row, col_i]
                vmax_use = float(np.abs(mat).max()) if np.abs(mat).max() > 0 else 1.0
                vmin_use = -vmax_use if col_i == 2 else 0.0
                im = ax.imshow(mat, cmap=cmap_name, aspect="auto",
                               interpolation="nearest",
                               vmin=vmin_use, vmax=vmax_use)
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
                               color="white", edgecolors=ec,
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

    with plt.style.context(t["mpl_style"]):
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.5),
                                  facecolor=t["fig_bg"], squeeze=False)
        fig.subplots_adjust(hspace=0.52, wspace=0.42,
                            left=0.10, right=0.85, top=0.87, bottom=0.13)
        fig.suptitle(f"Behaviour Group: {bg_name}  —  Mean ± SEM by Exp Group",
                     color=bg_color, fontsize=12, fontweight="bold", y=0.97)

        for idx, (metric_key, ylabel) in enumerate(METRICS):
            ax = axes[idx // 2, idx % 2]
            x_base = np.arange(n_eg)

            pval = None
            if SCIPY_OK and n_eg >= 2:
                group_vals = [
                    np.array([records[uid_to_idx[uid]]["metrics"]
                               .get(bg_name, {}).get(metric_key) or 0.0
                               for uid in exp_groups[eg]], dtype=float)
                    for eg in eg_names
                ]
                try:
                    _, pval = sp_stats.kruskal(*group_vals)
                except Exception:
                    try:
                        _, pval = sp_stats.f_oneway(*group_vals)
                    except Exception:
                        pval = None

            for ei, eg in enumerate(eg_names):
                ec   = eg_colors[eg]
                mean = grand[eg][bg_name][metric_key]["mean"]
                sem  = grand[eg][bg_name][metric_key]["sem"]
                n    = grand[eg][bg_name][metric_key]["n"]
                ax.bar(x_base[ei], mean, width=0.6,
                       color=ec, edgecolor=bg_color,
                       linewidth=1.4, alpha=0.82, zorder=2)
                ax.errorbar(x_base[ei], mean, yerr=sem, fmt="none",
                            color=t["tick"], linewidth=1.4, capsize=4, zorder=3)
                pts = [records[uid_to_idx[uid]]["metrics"]
                       .get(bg_name, {}).get(metric_key) or 0.0
                       for uid in exp_groups[eg]]
                jitter = rng.uniform(-0.12, 0.12, len(pts))
                ax.scatter(x_base[ei] + jitter, pts,
                           color="white", edgecolors=ec,
                           s=34, linewidths=0.9, alpha=0.88, zorder=4)
                y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else (mean * 1.3 or 1)
                ax.text(x_base[ei], y_top * 0.02, f"n={n}",
                        ha="center", va="bottom", color=t["tick"], fontsize=7)

            if pval is not None and n_eg >= 2:
                stars = ("***" if pval < 0.001 else "**" if pval < 0.01
                         else "*" if pval < 0.05 else "ns")
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
                for ev in am.get(beh_g, {}).get("events", []):
                    ax.barh(y, ev["dur_s"], left=ev["start_s"],
                            height=0.55, color=ginfo["color"], alpha=0.9, zorder=2)
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

        #   column header  
        hdr = ctk.CTkFrame(self, fg_color=T()["hdr_bg"])
        hdr.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 0))
        for col_txt, col_w in [("", 24), ("#", 28), ("Animal", 165),
                                ("Exp Group", 110), ("FPS", 46)]:
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
        egs = list({self._read_eg(a) for a in self._animals})
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
        ctk.CTkLabel(rf, text=entry["name"][:26], width=165, anchor="w",
                     text_color=T()["text"], font=ctk.CTkFont(size=11),
                     ).pack(side="left", padx=4)
        eg_entry = ctk.CTkEntry(rf, textvariable=entry["exp_group"],
                               width=110, placeholder_text="e.g. Control")
        eg_entry.pack(side="left", padx=6, pady=4)
        entry["_eg_entry"] = eg_entry
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

    def _remove_entry(self, entry: dict):
        if entry in self._animals:
            self._animals.remove(entry)
            if entry["_row_frame"] and entry["_row_frame"].winfo_exists():
                entry["_row_frame"].destroy()
            self._update_info()
            self.on_change()

    def _update_info(self):
        n   = len(self._animals)
        egs = {self._read_eg(a) for a in self._animals}
        if n == 0:
            self._info_lbl.configure(text="No animals loaded.")
        else:
            eg_txt = f"  -  {len(egs)} exp group{'s' if len(egs) != 1 else ''}"
            self._info_lbl.configure(
                text=f"{n} animal{'s' if n != 1 else ''} loaded{eg_txt}")

    def _read_eg(self, a: dict) -> str:
        """Read the current experimental group name from the entry widget."""
        eg_widget = a.get("_eg_entry")
        if eg_widget and eg_widget.winfo_exists():
            return eg_widget.get().strip() or "Default"
        return a["exp_group"].get().strip() or "Default"

    def get_animals(self) -> list:
        out = []
        for a in self._animals:
            out.append({
                "uid":       a["uid"],
                "name":      a["name"],
                "path":      a["path"],
                "df":        a["df"],
                "fps":       a["fps"],
                "exp_group": self._read_eg(a),
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
        added = 0
        for path in paths:
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
                    "_selected": tk.BooleanVar(master=self, value=False),
                    "_row_frame": None,
                }
                self._animals.append(entry)
                self._build_row(entry, len(self._animals) - 1)
                added += 1
            except Exception:
                pass
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
        ctk.CTkLabel(sf, text="Metric:", text_color=T()["subtext"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(4, 0))
        self._metric_var = ctk.StringVar(value="total_duration")
        ctk.CTkOptionMenu(sf, variable=self._metric_var,
                          values=["total_duration", "frequency", "mean_bout"],
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

        #   Biological Relevance Filter  (shown BEFORE run button so user sets it first)
        df_sec = section(ctrl, "Biological Relevance Filter")
        ctk.CTkLabel(df_sec,
                     text="Remove clusters that are too brief or\n"
                          "too rare to be biologically meaningful.\n"
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

        ctk.CTkButton(ctrl, text="   Run Reclustering  (applies filter above)",
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
                    "Dist Matrix", "Transitions",
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
        animals  = self._get_animals()
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
                        groups=user_groups or None,
                        combined=user_combined)
                    figs_to_save = list(result) if isinstance(result, tuple) else [result]
                elif mode == "Volcano":
                    figs_to_save = [build_volcano_figure(
                        self._stats_df, top_n, p_thresh,
                        groups=user_groups or None)]
                elif mode == "Heatmap":
                    figs_to_save = [build_heatmap_figure(
                        self._stats_df, animals, top_n, p_thresh, metric,
                        eg_colors or None)]
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
                    figs_to_save = [build_distance_matrix_figure(recluster_data)]
                elif mode == "Transitions":
                    figs_to_save = [build_transition_figure(recluster_data)]
                elif mode == "Cluster Stats":
                    figs_to_save = [build_cluster_stats_figure(recluster_data, animals)]
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

    def _get_save_k(self) -> int:
        try:
            return max(2, int(self._save_k_var.get()))
        except ValueError:
            return 5

    def _run_stats(self):
        animals = self._get_animals()
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
            self._stats_df = run_cluster_statistics(animals, metric)
        except Exception:
            messagebox.showerror("Stats Error", traceback.format_exc())
            self._status("Error.", "#cc4444")
            return
        n_sig = int((self._stats_df["pval_kw"] < p_thresh).sum())
        self._status(
            f"Done. {len(self._stats_df)} clusters tested.\n"
            f"{n_sig} significant at p < {p_thresh}.", "#88cc88")
        self._switch_plot(self._plot_mode.get())

    def _run_reclustering(self):
        animals = self._get_animals()
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

        # Build duration filter from UI
        dur_filter = {}
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

        n_all      = len(self._recluster.get("all_raw_cluster_ids", []))
        n_filtered = len(self._recluster["cluster_ids"])
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
        animals  = self._get_animals()
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
                    groups=user_groups or None,
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
                else:
                    self._show_figure(result)
            elif mode == "Volcano":
                fig = build_volcano_figure(self._stats_df, top_n, p_thresh,
                                           groups=user_groups or None)
                self._show_figure(fig)
            elif mode == "Heatmap":
                fig = build_heatmap_figure(self._stats_df, animals,
                                            top_n, p_thresh, metric,
                                            eg_colors_override=eg_colors or None)
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
                self._show_figure(build_distance_matrix_figure(recluster_data))
            elif mode == "Transitions":
                self._show_figure(build_transition_figure(recluster_data))
            elif mode == "Cluster Stats":
                self._show_figure(build_cluster_stats_figure(recluster_data, animals))
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
        self._stats_df.to_csv(path, index=False)
        messagebox.showinfo("Saved", f"Statistics saved:\n{path}")


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

    x = embedding[:, 0]
    y = embedding[:, 1]

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


def build_energy_landscape_figure(
        animal_data: list, umap_embedding, umap_labels,
        t: dict = None) -> "plt.Figure":
    """
    3D behavioral energy landscape over UMAP space.
    Three views per group (groups are rows):
      col 0 – cluster-coloured, from above  (elev=25, azim=45)
      col 1 – cluster-coloured, from below  (elev=-20, azim=215)
      col 2 – coolwarm energy map, from below with mesh backbone
               blue = low energy (common), red = high energy (rare)
    Z = −ln(weighted KDE density); no text labels on the surface.
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
            "scipy / mpl_toolkits not available for 3D plotting.")

    umap_x = _np.asarray(umap_embedding[:, 0], dtype=float)
    umap_y = _np.asarray(umap_embedding[:, 1], dtype=float)
    umap_l = _np.asarray(umap_labels, dtype=int)

    groups_eg: dict = {}
    for ani in animal_data:
        groups_eg.setdefault(ani["exp_group"], []).append(ani)
    eg_names = list(groups_eg.keys())
    n_eg = len(eg_names)
    if n_eg == 0:
        return _placeholder("No animals loaded.")

    GRID_N = 60
    # col 0: cluster-coloured above | col 1: cluster-coloured below | col 2: coolwarm below
    VIEW_ANGLES = [(25, 45), (-20, 215), (-28, 45)]
    ncols = 3
    nrows = n_eg
    fig_w = max(16.5, ncols * 5.5)
    fig_h = max(5.0,  nrows * 5.2 + 0.8)

    x_rng = umap_x.max() - umap_x.min()
    y_rng = umap_y.max() - umap_y.min()
    xi = _np.linspace(umap_x.min() - x_rng * 0.05,
                      umap_x.max() + x_rng * 0.05, GRID_N)
    yi = _np.linspace(umap_y.min() - y_rng * 0.05,
                      umap_y.max() + y_rng * 0.05, GRID_N)
    XX, YY = _np.meshgrid(xi, yi)

    # ── Nearest-cluster assignment for every grid point (shared across groups) ──
    cid_list = sorted(int(c) for c in _np.unique(umap_l))
    cent_xy  = _np.array(
        [(float(umap_x[umap_l == c].mean()), float(umap_y[umap_l == c].mean()))
         for c in cid_list])                                       # (n_cl, 2)
    pts_flat = _np.column_stack([XX.ravel(), YY.ravel()])          # (GRID_N², 2)
    dists2   = ((pts_flat[:, None, :] - cent_xy[None, :, :]) ** 2
                ).sum(axis=2)                                      # (GRID_N², n_cl)
    nearest_ci       = dists2.argmin(axis=1)
    nearest_cid_grid = _np.array(
        [cid_list[i] for i in nearest_ci]).reshape(GRID_N, GRID_N)

    # RGBA face-colour array: (GRID_N-1, GRID_N-1, 4) — one colour per quad
    face_base = _np.zeros((GRID_N - 1, GRID_N - 1, 4))
    for cid in cid_list:
        rgba = _np.array(mcolors.to_rgba(PALETTE[cid % len(PALETTE)], alpha=0.85))
        face_base[nearest_cid_grid[:-1, :-1] == cid] = rgba

    # Wireframe stride for the mesh backbone (col 2)
    wstep = max(1, GRID_N // 12)

    # ── Data-support mask (shared across groups) ──────────────────────────────
    # Empty UMAP corridors between clusters have near-zero KDE density and
    # produce −ln(≈0) spikes that dominate normalization and bury real clusters.
    # An unweighted KDE of the raw point positions identifies where data actually
    # lives; grid cells below the 10th-percentile support are masked to NaN so
    # they are skipped by plot_surface and excluded from normalisation.
    try:
        _kde_support   = _gkde(_np.vstack([umap_x, umap_y]))
        _ZZ_support    = _kde_support(
            _np.vstack([XX.ravel(), YY.ravel()])).reshape(GRID_N, GRID_N)
        _support_thresh = _np.percentile(_ZZ_support, 10)
        _support_mask   = _ZZ_support < _support_thresh   # True = empty space
    except Exception:
        _support_mask = _np.zeros((GRID_N, GRID_N), dtype=bool)

    with plt.style.context(t["mpl_style"]):
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=t["fig_bg"])

        for ei, eg in enumerate(eg_names):
            ani_list = groups_eg[eg]
            eg_color = PALETTE[ei % len(PALETTE)]

            # Per-cluster occupancy probability for this experimental group
            cl_counts: dict = {}
            for ani in ani_list:
                for lbl in ani["df"]["label"].values:
                    k = int(lbl)
                    cl_counts[k] = cl_counts.get(k, 0) + 1
            total = max(sum(cl_counts.values()), 1)
            P_cl  = {k: v / total for k, v in cl_counts.items()}

            # Weight each UMAP point by its cluster's occupancy probability
            # High P → high density → low −ln(P) → valley (common, low energy)
            # Low P  → low density  → high −ln(P) → peak  (rare,  high energy)
            weights = _np.array(
                [P_cl.get(int(l), 1e-9) for l in umap_l], dtype=float)
            wsum = weights.sum()
            weights = weights / wsum if wsum > 0 else \
                      _np.ones_like(weights) / len(weights)

            # Weighted 2-D KDE → density surface
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
            ZZ_smooth = _gfilt(ZZ_dens, sigma=1.5)
            ZZ_energy = -_np.log(_np.maximum(ZZ_smooth, 1e-12))

            # Mask empty UMAP corridors so their −ln(≈0) spikes don't
            # dominate normalisation and crush the real cluster landscape.
            ZZ_energy[_support_mask] = _np.nan

            # Normalise 0–1 over supported region: 0=common, 1=rare
            E_lo = float(_np.nanmin(ZZ_energy))
            E_hi = float(_np.nanmax(ZZ_energy))
            ZZ_norm = (ZZ_energy - E_lo) / max(E_hi - E_lo, 1e-9)

            # ── Top-3 common (blue) and top-3 rare (red) clusters for labels ──
            sorted_by_p = sorted(P_cl.items(), key=lambda kv: kv[1], reverse=True)
            n_lbl = min(3, len(sorted_by_p))
            label_clusters = (
                [(cid, "#5599ff") for cid, _ in sorted_by_p[:n_lbl]] +   # common
                [(cid, "#ff4444") for cid, _ in sorted_by_p[-n_lbl:]]    # rare
            )

            for vi, (elev, azim) in enumerate(VIEW_ANGLES):
                ax3d = fig.add_subplot(
                    nrows, ncols, ei * ncols + vi + 1, projection="3d")
                ax3d.set_facecolor(t["ax_bg"])

                if vi < 2:
                    # ── Cluster-coloured surface (views 0 & 1) ──────────────
                    ax3d.plot_surface(XX, YY, ZZ_norm,
                                      facecolors=face_base,
                                      linewidth=0, antialiased=True)
                else:
                    # ── coolwarm energy surface (view 2) ────────────────────
                    # blue = low energy = common states
                    # red  = high energy = rare states
                    ax3d.plot_surface(XX, YY, ZZ_norm,
                                      cmap="coolwarm", alpha=0.88,
                                      linewidth=0, antialiased=True)
                    # Sparse wireframe backbone to show mesh undulation
                    ax3d.plot_wireframe(
                        XX[::wstep, ::wstep],
                        YY[::wstep, ::wstep],
                        ZZ_norm[::wstep, ::wstep],
                        color="#dddddd", alpha=0.22, linewidth=0.5)

                # ── Cluster labels: top-3 common (blue) / top-3 rare (red) ──
                # Float labels slightly off the surface; direction depends on
                # whether the viewer is above or below the landscape.
                z_offset = -0.10 if elev < 0 else 0.08
                for cid, lcolor in label_clusters:
                    mask = umap_l == cid
                    if not mask.any():
                        continue
                    cx = float(umap_x[mask].mean())
                    cy = float(umap_y[mask].mean())
                    # Z at the nearest grid node to this centroid
                    ix = int(_np.argmin(_np.abs(xi - cx)))
                    iy = int(_np.argmin(_np.abs(yi - cy)))
                    cz = float(ZZ_norm[iy, ix])
                    ax3d.text(cx, cy, cz + z_offset,
                              f"C{cid}",
                              color=lcolor, fontsize=7, fontweight="bold",
                              ha="center", va="bottom", zorder=8)

                ax3d.view_init(elev=elev, azim=azim)
                ax3d.set_xlabel("UMAP-1", color=t["tick"], fontsize=7, labelpad=2)
                ax3d.set_ylabel("UMAP-2", color=t["tick"], fontsize=7, labelpad=2)
                ax3d.set_zlabel("−ln P",  color=t["tick"], fontsize=7, labelpad=2)
                VIEW_TITLES = [eg, f"{eg}  ↓", f"{eg}  energy"]
                ax3d.set_title(VIEW_TITLES[vi], color=eg_color,
                               fontweight="bold", fontsize=10, pad=4)
                ax3d.tick_params(colors=t["tick"], labelsize=5)
                for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
                    pane.fill = False
                    pane.set_edgecolor(t["border"])
                ax3d.grid(True, alpha=0.2, color=t["border"])

        fig.suptitle(
            "3-D Behavioral Energy Landscape  "
            "(−ln P: blue = common / low energy, red = rare / high energy)",
            color=t["tick"], fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
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
                         color="white", fontsize=6.5, fontweight="bold",
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
        handles = []
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
            handles.append(mpatches.Patch(color=color, label=gname))
        ax_grp.set_xlabel("UMAP-1", color=t["tick"], fontsize=11)
        ax_grp.set_ylabel("UMAP-2", color=t["tick"], fontsize=11)
        ax_grp.set_title("UMAP — Behavioural Groups",
                         color=t["tick"], fontweight="bold", fontsize=13, pad=10)
        if handles:
            ax_grp.legend(handles=handles, fontsize=9, loc="upper right",
                          facecolor=t["ax_bg"], edgecolor=t["border"],
                          labelcolor=t["tick"], framealpha=0.88)
        _style_ax(ax_grp, t)

        fig.tight_layout()
    return fig


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
        self._nsteps_lbl.pack(anchor="w", padx=12, pady=(0, 8))

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

        # Save button
        ctk.CTkButton(
            ctrl, text="Save: Save Graph",
            command=self._save_graph,
            fg_color=T()["btn_save"],
        ).pack(fill="x", padx=8, pady=(2, 6))

        # Status label
        self._status_lbl = ctk.CTkLabel(
            ctrl, text="",
            text_color=T()["subtext"],
            wraplength=240, justify="left",
            font=ctk.CTkFont(size=11))
        self._status_lbl.pack(anchor="w", padx=12, pady=4)

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
        # Close previous figures
        for old in self._current_figs:
            try:
                plt.close(old)
            except Exception:
                pass
        self._current_figs = list(figs)
        self._current_fig  = figs[0] if figs else None

        for w in self._inner.winfo_children():
            w.destroy()

        def _bind_wheel(w):
            w.bind("<MouseWheel>",
                   lambda e: self._canvas.yview_scroll(
                       int(-1 * (e.delta / 120)), "units"))
            for child in w.winfo_children():
                _bind_wheel(child)

        last_canvas = None
        for fi, fig in enumerate(figs):
            mpl_c = FigureCanvasTkAgg(fig, master=self._inner)
            mpl_c.draw()
            widget = mpl_c.get_tk_widget()
            widget.pack(fill="x", expand=True, padx=2, pady=(0, 2))
            _bind_wheel(widget)
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
        animals   = self._apply_brf(self._get_animals())
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
                figs.append(build_sankey_figure(animals, t, n_steps=n_steps))
                if groups:
                    figs.append(
                        build_sankey_beh_groups_figure(
                            animals, groups, t, n_steps=n_steps))

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
                    build_energy_landscape_figure(animals, emb, lbl, t))
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
        ctk.CTkLabel(hdr, text="   CUBE Suite  v6",
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

        for tab in (self._tab_preview, self._tab_analysis,
                    self._tab_metrics, self._tab_combined,
                    self._tab_unbiased, self._tab_explorer):
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

    def _get_animals_for_unbiased(self) -> list:
        if hasattr(self, "_animal_panel"):
            return self._animal_panel.get_animals()
        return []

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

    def _auto_load_groups(self):
        """
        Try to auto-detect behaviour groups for the current root directory:
          1. cluster_behaviour_mapping.tsv exported by Phase 4 Video Explorer
          2. Phase 4 session JSON  (contains 'behaviour_groups' key)
          3. Fallback: one group per cluster so analysis can proceed immediately
        Safe to call multiple times; silently skips if _root_dir or _all_labels
        are not yet set.
        """
        if not self._root_dir or not self._all_labels:
            return
        groups_loaded = False

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

        # 3. Fallback: one group per cluster
        if not groups_loaded:
            auto_groups = groups_from_all_clusters(self._all_labels)
            self._open_editor()
            self._editor.load_groups_from_dict(auto_groups, self._all_labels)
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
        groups  = self._get_groups()
        animals = self._animal_panel.get_animals()
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

            # Ethogram: all animals coloured by experimental group
            fig_eth = build_combined_ethogram_figure(groups, self._combined)
            self._add_combined_figure(
                fig_eth,
                "Ethograms by Animal  (coloured by Experimental Group)")

            # One 2×2 metric figure per behaviour group
            for bg_name in groups:
                fig_grp = build_combined_group_figure(
                    bg_name, groups, self._combined,
                    eg_colors_override=eg_colors)
                self._add_combined_figure(fig_grp)

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
        # Save every figure that was generated (ethogram + per-group metrics)
        groups   = self._get_groups()
        bg_names = ["ethogram"] + list(groups.keys())
        for i, fig in enumerate(getattr(self, "_combined_figs", [])):
            label = bg_names[i] if i < len(bg_names) else f"plot_{i}"
            safe  = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
            for ext in ("png", "pdf"):
                fig.savefig(out / f"combined_{safe}_{ts}.{ext}",
                            dpi=300 if ext == "png" else None,
                            bbox_inches="tight",
                            facecolor=fig.get_facecolor())
        grand  = self._combined["grand"]
        # Pre-compute p-values for all metric × behaviour_group combos
        pvals_all = compute_combined_pvals(self._combined, groups)
        STAR_THRESH = [(0.001, "***"), (0.01, "**"), (0.05, "*")]

        def _stars(pv):
            if pv is None:
                return "N/A"
            for thresh, sym in STAR_THRESH:
                if pv < thresh:
                    return sym
            return "ns"

        rows = []
        for eg_name, beh_dict in grand.items():
            for beh_group in groups:
                for metric in ("total_duration", "frequency", "latency", "mean_bout"):
                    d  = beh_dict[beh_group][metric]
                    pv = pvals_all.get((metric, beh_group))
                    rows.append({
                        "Exp_Group":        eg_name,
                        "Beh_Group":        beh_group,
                        "Metric":           metric,
                        "Mean":             round(d["mean"], 4),
                        "SEM":              round(d["sem"],  4),
                        "N":                d["n"],
                        "p_value_KW":       round(pv, 6) if pv is not None else "N/A",
                        "Significance":     _stars(pv),
                    })
        pd.DataFrame(rows).to_csv(out / f"combined_summary_{ts}.csv", index=False)
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
            groups   = self._get_groups()
            bg_names = ["ethogram"] + list(groups.keys())
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
            # p-value CSV
            pvals_all   = compute_combined_pvals(self._combined, groups)
            STAR_THRESH = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
            def _stars(pv):
                if pv is None:
                    return "N/A"
                for thresh, sym in STAR_THRESH:
                    if pv < thresh:
                        return sym
                return "ns"
            grand = self._combined["grand"]
            rows  = []
            for eg_name, beh_dict in grand.items():
                for beh_group in groups:
                    for metric in ("total_duration", "frequency", "latency", "mean_bout"):
                        d  = beh_dict[beh_group][metric]
                        pv = pvals_all.get((metric, beh_group))
                        rows.append({
                            "Exp_Group":    eg_name,
                            "Beh_Group":    beh_group,
                            "Metric":       metric,
                            "Mean":         round(d["mean"], 4),
                            "SEM":          round(d["sem"],  4),
                            "N":            d["n"],
                            "p_value_KW":   round(pv, 6) if pv is not None else "N/A",
                            "Significance": _stars(pv),
                        })
            pd.DataFrame(rows).to_csv(
                out_combined / f"combined_summary_{ts}.csv", index=False)
            saved.append(f"Combined Analysis ({len(combined_figs)} figure(s) + CSV)")
        else:
            skipped.append("Combined Analysis — run 'Run Combined Analysis' first")

        # ── Unbiased Analytics ────────────────────────────────────────────
        out_unbiased = out_root / "unbiased_analytics"
        try:
            self._unbiased_panel._save_all_graphs(out_dir=out_unbiased)
            saved.append("Unbiased Analytics")
        except Exception as exc:
            skipped.append(f"Unbiased Analytics — {exc}")

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