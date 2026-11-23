# -*- coding: utf-8 -*-
"""
Blind Human Cluster-Validation GUI
===================================
Standalone tool to sanity-check programmatic behavioural clusters (from any
CUBE run, the B-SOiD-vs-CUBE comparison, the medoid-vs-consensus comparison,
the arena-split comparison, etc.) against real human judgement.

Point it at any folder. It recursively scans for "*.mp4" underneath, infers
each clip's cluster id and "condition" (method/run) from its own folder
names, and also looks for "most_changed_clusters_index.csv" (preferred) or
"cluster_stability.csv" files (written by
script05_medoid_vs_consensus_clusters.py / script06_arena_split_before_after.py)
to build a QUEUE of only the "relevant" comparisons -- rather than letting a
rater freely combine arbitrary, unrelated clusters. Within each scope (e.g.
one temporal setting, or one method), the most-changed clusters on each side
are ranked by their own individual stability score and matched one-to-one
by rank (1st-most-changed with 1st-most-changed, etc.), giving exactly
min(n_side_a, n_side_b) real, video-backed comparisons per scope -- not a
full cross-product and not strict reciprocal-best-match pairing (confirmed
on real data to often leave zero video-backed reciprocal pairs). Scopes are
walked as complete blocks (all of one scope's comparisons before the next),
most-changed pair first within each block. Submitting a rating automatically
advances to the next pair. (If no index/stability CSV exists anywhere under
the chosen folder, it falls back to a manual "pick any two clusters" mode so
the tool still works against a plain folder of cluster videos.)

Each comparison is shown side by side -- looping, slow-motion (0.33x),
zoomed (1.5x) video, blind (labeled only "Group A" / "Group B", with which
real cluster lands left/right randomized per comparison). Ratings (0-10
sliders: difference, per-side distinctiveness, per-side internal
consistency) are collected per named rater and autosaved continuously --
both as an in-progress draft (so a window close mid-rating loses nothing)
and as a permanent append-only log on submit. A separate Results tab
summarizes and plots everything collected so far, and can export the
analysis alongside the always-current raw log.

Run (use the CUBE environment's Python, never the system Python; its path is saved in
cube_env_path.txt by install_cube.bat):
    python   (with the CUBE environment active) cube_cluster_validation_gui.py

Reuses cube_video_explorer.py's VideoTile/MultiPlayer widgets for playback
(that module has no dependency on cube_core.py, so importing it standalone
here is safe) rather than reimplementing video decoding/looping/zoom/speed.
"""

import os
import re
import sys
import json
import random
import platform
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cube_video_explorer import MultiPlayer, C, _raise_window  # noqa: E402

import pandas as pd  # noqa: E402
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
import matplotlib  # noqa: E402
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402

# On Windows, an app that isn't DPI-aware can get clicks delivered to the
# wrong widget on non-100%-scaled/mixed-DPI setups -- same fix cube.py and
# cube_video_explorer.py apply for themselves.
if platform.system() == "Windows":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

SPEED = 0.33               # slow-motion playback factor
ZOOM = 1.5                 # tile zoom factor
MAX_CLIPS_PER_SIDE = 10

CLUSTER_RE = re.compile(r"^cluster[_-]?(\d+)$", re.IGNORECASE)
GENERIC_SEGMENTS = {"videos", "example_clips", "examples", "clips"}
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")

SCORE_FIELDS = [
    ("difference",
     "How different are these two behaviours from each other?  "
     "(0 = identical / very similar, 10 = totally different)"),
    ("distinct_relative",
     "Which group looks like a more distinct, well-defined behaviour?  "
     "(0 = Group A much more distinct, 5 = about equal, 10 = Group B much more distinct)"),
    ("consistency_relative",
     "Which group is more internally consistent (same behaviour repeated across its clips)?  "
     "(0 = Group A much more consistent, 5 = about equal, 10 = Group B much more consistent)"),
]

# A single discrete choice question, separate from the 0-10 sliders above:
# a forced pick rather than a graded score, for the one question that's
# naturally a choice ("which one IS the better behaviour") rather than a
# matter of degree.
CHOICE_FIELD = "better_behavior"
CHOICE_PROMPT = "Overall, which is the better-defined behaviour -- Group A or Group B?"
CHOICE_OPTIONS = [("A", "Group A"), ("tie", "Tie / can't tell"), ("B", "Group B")]


def _safe(s) -> str:
    return _UNSAFE_RE.sub("_", str(s))[:150]


def _bind_wheel_to_canvas(bind_root, canvas: tk.Canvas):
    """Make mouse-wheel scrolling work over a scrollable canvas even when the
    cursor is over one of the many child widgets packed inside it (a
    VideoTile, a Label, etc.) rather than the canvas's own bare background.

    Tk's <MouseWheel> only fires on the exact widget under the cursor, and
    does not bubble to parents -- so canvas.bind("<MouseWheel>", ...) alone
    only works in the (rare, here effectively nonexistent) gaps of empty
    canvas background, which is why scrolling appeared completely broken
    once the canvas was full of video tiles. Instead, bind globally
    (bind_all) and, on every wheel event, walk UP from whatever widget is
    literally under the pointer (found via winfo_containing) looking for
    this canvas among its ancestors -- if found, scroll it.
    """
    def _find_target(event) -> bool:
        try:
            w = bind_root.winfo_containing(event.x_root, event.y_root)
        except Exception:
            return False
        while w is not None:
            if w is canvas:
                return True
            w = w.master
        return False

    def _on_wheel(event):
        if _find_target(event):
            canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def _on_button4(event):
        if _find_target(event):
            canvas.yview_scroll(-1, "units")

    def _on_button5(event):
        if _find_target(event):
            canvas.yview_scroll(1, "units")

    bind_root.bind_all("<MouseWheel>", _on_wheel, add="+")
    bind_root.bind_all("<Button-4>", _on_button4, add="+")
    bind_root.bind_all("<Button-5>", _on_button5, add="+")


# ── video discovery ──────────────────────────────────────────────────────

def scan_videos(root: Path) -> dict:
    """Recursively find *.mp4 under root, grouped as
    {condition_label: {cluster_id: [Path, ...]}}.

    cluster_id comes from the first path segment matching cluster_(\\d+).
    condition_label is built from the 1-2 path segments immediately above
    the cluster folder, skipping generic names ("videos", "example_clips")
    -- so it resolves to e.g. "medoid" / "consensus" for the method-
    comparison layout, or a run-tag for a plain per-condition
    videos\\example_clips\\cluster_NN\\ layout, without assuming one fixed
    folder structure.
    """
    tree = defaultdict(lambda: defaultdict(list))
    for p in root.rglob("*.mp4"):
        try:
            parts = p.relative_to(root).parts[:-1]
        except ValueError:
            continue
        cluster_id = None
        cluster_idx = None
        for i, seg in enumerate(parts):
            m = CLUSTER_RE.match(seg)
            if m:
                cluster_id = int(m.group(1))
                cluster_idx = i
        if cluster_id is None:
            continue
        above = parts[:cluster_idx]
        meaningful = [s for s in above if s.lower() not in GENERIC_SEGMENTS]
        condition = "/".join(meaningful[-2:]) if meaningful else "(root)"
        tree[condition][cluster_id].append(p)
    return {c: dict(clusters) for c, clusters in tree.items()}


def _find_cluster_dir(base: Path, cluster_id: int):
    """Find the cluster_NN subdirectory of base matching cluster_id, without
    assuming a fixed zero-padding width (folders are seen as both
    "cluster_00" and "cluster_9" across different CUBE scripts)."""
    if not base.is_dir():
        return None
    for d in base.iterdir():
        if d.is_dir():
            m = CLUSTER_RE.match(d.name)
            if m and int(m.group(1)) == cluster_id:
                return d
    return None


def _scope_label(scope_dir: Path, root: Path) -> str:
    try:
        label = str(scope_dir.relative_to(root))
    except ValueError:
        label = scope_dir.name
    return scope_dir.name if label in (".", "") else label


def _load_match_matrix(scope_dir: Path):
    """Loads the real bin-level overlap evidence for a scope, if present:
    "data/contingency_counts.csv" (raw co-occurrence counts, rows = one
    side's cluster ids, columns = the other's -- written by script05/06/07)
    averaged with whatever "data/frac_of_*.csv" files share its exact shape
    (row- and column-normalized share matrices, same row/col layout, just
    different normalization -- see those scripts' own "How to read" docs).

    This is real ground truth (how many/what fraction of raw data bins two
    clusters actually share), not a lossy 2D visualization proxy -- used in
    preference to guessing at similarity from each cluster's own isolated
    stability_score.

    Returns (matrix_df, row_side_hint) where row_side_hint is the row
    column's own header with a trailing "_cluster" stripped (e.g. "medoid",
    "temporal_off") -- used by _side_is_rows() to work out which of two
    known side names the rows actually correspond to, without assuming an
    exact naming match (the row header can carry an extra prefix, e.g.
    "temporal_off" for a side this code otherwise calls "off"). Returns
    (None, None) if no contingency_counts.csv exists for this scope.
    """
    data_dir = scope_dir / "data"
    cont_path = data_dir / "contingency_counts.csv"
    if not cont_path.exists():
        return None, None
    try:
        cont = pd.read_csv(cont_path)
    except Exception:
        return None, None
    row_col = cont.columns[0]
    row_hint = row_col[:-len("_cluster")] if row_col.endswith("_cluster") else row_col
    try:
        cont = cont.set_index(row_col)
        cont.index = cont.index.astype(int)
        cont.columns = cont.columns.astype(int)
    except (ValueError, TypeError):
        return None, None

    frac_frames = []
    for fp in sorted(data_dir.glob("frac_of_*.csv")):
        try:
            f = pd.read_csv(fp)
            f = f.set_index(f.columns[0])
            f.index = f.index.astype(int)
            f.columns = f.columns.astype(int)
            if f.shape == cont.shape:
                frac_frames.append(f)
        except Exception:
            continue

    if frac_frames:
        matrix = sum(frac_frames) / len(frac_frames)
    else:
        total = cont.to_numpy().sum()
        matrix = cont / total if total > 0 else cont
    return matrix, row_hint


def _side_is_rows(row_hint: str, side_a: str, side_b: str):
    """Which of side_a/side_b the match matrix's rows correspond to, via
    fuzzy (substring, case-insensitive) name matching -- handles a row
    header like "temporal_off_cluster" against a side value of "off"
    ("off" in "temporal_off"). Returns True/False, or None if ambiguous."""
    h, a, b = row_hint.lower(), side_a.lower(), side_b.lower()
    a_match, b_match = (a in h or h in a), (b in h or h in b)
    if a_match and not b_match:
        return True
    if b_match and not a_match:
        return False
    return None


def _pairs_from_rows(df: pd.DataFrame, scope_label: str, scope_dir: Path, videos_dir: Path) -> list:
    """Shared row-to-pair logic for both source formats (both carry the same
    side/cluster_id/best_match_side/best_match_id/stability_score columns).

    Builds ONE-TO-ONE matched pairs between this scope's two sides -- not a
    full cross-product (which would give 5x5=25 near-duplicate comparisons
    per scope for a typical top-5-per-side export) and not each cluster's
    own isolated best_match_id either (confirmed on real data: a
    most-changed cluster's own recorded best match is usually a cluster
    that was itself never independently flagged as most-changed on the
    other side, so it has no exported video -- temporal_OFF's own
    5-most-changed-per-side clusters have ZERO reciprocal matches with
    video on both sides).

    Preferred method: an optimal one-to-one assignment (Hungarian
    algorithm, scipy.optimize.linear_sum_assignment) over the REAL bin-
    overlap evidence in data/contingency_counts.csv + frac_of_*.csv
    (_load_match_matrix), restricted to just this scope's exported
    clusters -- maximizes total real overlap across all pairs jointly,
    rather than each cluster unilaterally picking its own single favorite
    (which frequently isn't reciprocated within this restricted subset).
    match_score is that pair's overlap share -- LOW match_score is a real,
    honestly-reported signal that even the best available partner for that
    cluster within this small exported subset wasn't a good one (which can
    happen when only ~5 clusters per side were exported), not a bug.

    Falls back to rank-by-own-stability-score zipping (match_score=None,
    "quality unknown") only if no usable contingency/frac data exists for
    this scope, or scipy isn't installed.
    """
    own_score: dict = {}
    sides_seen: list = []
    for _, r in df.iterrows():
        try:
            side, cid, score = str(r["side"]), int(r["cluster_id"]), float(r["stability_score"])
        except (KeyError, ValueError, TypeError):
            continue
        own_score[(side, cid)] = score
        if side not in sides_seen:
            sides_seen.append(side)

    matrix, row_hint = _load_match_matrix(scope_dir) if scope_dir is not None else (None, None)

    pairs = []
    seen = set()
    for i, side_a in enumerate(sides_seen):
        for side_b in sides_seen[i + 1:]:
            ids_a = sorted(cid for (s, cid) in own_score if s == side_a)
            ids_b = sorted(cid for (s, cid) in own_score if s == side_b)

            pairing = None  # list of (id_a, id_b, match_score_or_None)
            if SCIPY_OK and matrix is not None and row_hint is not None and ids_a and ids_b:
                a_is_rows = _side_is_rows(row_hint, side_a, side_b)
                if a_is_rows is not None:
                    sub = (matrix.reindex(index=ids_a, columns=ids_b) if a_is_rows
                           else matrix.reindex(index=ids_b, columns=ids_a).T).fillna(0).to_numpy()
                    if sub.shape[0] and sub.shape[1]:
                        row_ind, col_ind = linear_sum_assignment(-sub)
                        pairing = [(ids_a[r], ids_b[c], float(sub[r, c])) for r, c in zip(row_ind, col_ind)]

            if pairing is None:
                ranked_a = sorted(ids_a, key=lambda cid: own_score[(side_a, cid)])
                ranked_b = sorted(ids_b, key=lambda cid: own_score[(side_b, cid)])
                pairing = [(a, b, None) for a, b in zip(ranked_a, ranked_b)]

            for id_a, id_b, match_score in pairing:
                key = tuple(sorted([(side_a, id_a), (side_b, id_b)]))
                if key in seen:
                    continue
                dir_a = _find_cluster_dir(videos_dir / side_a / "example_clips", id_a)
                dir_b = _find_cluster_dir(videos_dir / side_b / "example_clips", id_b)
                clips_a = sorted(dir_a.glob("*.mp4")) if dir_a else []
                clips_b = sorted(dir_b.glob("*.mp4")) if dir_b else []
                if not clips_a or not clips_b:
                    continue
                seen.add(key)
                own_min = min(own_score[(side_a, id_a)], own_score[(side_b, id_b)])
                pairs.append(dict(
                    scope=scope_label,
                    comparison_key=f"{scope_label}::{side_a}c{id_a}__vs__{side_b}c{id_b}",
                    condition_a=f"{scope_label}/{side_a}", cluster_a=id_a, paths_a=clips_a,
                    condition_b=f"{scope_label}/{side_b}", cluster_b=id_b, paths_b=clips_b,
                    stability_score=own_min,
                    match_score=match_score,
                ))
    return pairs


def _nearest_ancestor_named(p: Path, name: str):
    """The closest ancestor of p whose folder name matches `name`
    (case-insensitive), or None. Used to locate a file's "scope" (the run
    folder holding both videos\\ and data\\) without assuming any fixed
    depth -- a future export layout that nests the index file one level
    deeper/shallower than today's still resolves correctly."""
    for anc in p.parents:
        if anc.name.lower() == name:
            return anc
    return None


def find_relevant_pairs(root: Path) -> list:
    """Recursively find the "relevant" (matched, both-sides-have-video)
    cluster comparisons under root, preferring the authoritative
    "most_changed_clusters_index.csv" files (one per side, normally at
    "<scope>/videos/<side>/most_changed_clusters_index.csv" -- these list
    EXACTLY which clusters were actually exported for that side, each with
    its best match) over the older shared "<scope>/data/cluster_stability.csv"
    (which lists every cluster's best match whether or not video was ever
    exported for it, requiring this function to re-derive "does video exist
    on both sides" itself). Both files share the same side/cluster_id/
    best_match_side/best_match_id/stability_score columns.

    Every "*.csv" of either name anywhere under root is found via a plain
    recursive rglob (not a fixed-depth glob under an assumed folder) --
    each file's own "scope" (the run folder holding its videos\\ and data\\
    siblings) is then found by walking UP from the file to the nearest
    "videos" ancestor (for an index file) or "data" ancestor (for a
    stability file), so this keeps working even if a future export nests
    these files at a different depth than today's convention.

    Per scope: if any most_changed_clusters_index.csv files resolve to it,
    their rows (combined across all sides found) are used exclusively;
    cluster_stability.csv is only consulted as a fallback for a scope where
    no index file exists at all -- so a scope that has been re-exported
    with the newer per-side index automatically switches over without any
    configuration.

    Returns a list of dicts, one per relevant pair:
    {scope, condition_a, cluster_a, paths_a, condition_b, cluster_b, paths_b,
     stability_score, match_score}, sorted by scope first (alphabetically --
     so one scope's comparisons are walked through as a complete block
     before moving to the next) and then by match_score descending within
     each scope (the best-evidenced, highest-confidence pairs first; pairs
     with no match_score -- the rank-zip fallback -- sort last).
    """
    index_files_by_scope = defaultdict(list)
    for p in root.rglob("most_changed_clusters_index.csv"):
        videos_dir = _nearest_ancestor_named(p, "videos")
        if videos_dir is not None:
            index_files_by_scope[videos_dir.parent].append(p)

    stability_files_by_scope = {}
    for p in root.rglob("cluster_stability.csv"):
        data_dir = _nearest_ancestor_named(p, "data")
        scope_dir = data_dir.parent if data_dir is not None else p.parent.parent
        stability_files_by_scope.setdefault(scope_dir, p)

    pairs = []
    for scope_dir in set(index_files_by_scope) | set(stability_files_by_scope):
        videos_dir = scope_dir / "videos"
        if not videos_dir.is_dir():
            continue
        scope_label = _scope_label(scope_dir, root)

        if scope_dir in index_files_by_scope:
            frames = []
            for idx_csv in index_files_by_scope[scope_dir]:
                try:
                    frames.append(pd.read_csv(idx_csv))
                except Exception:
                    continue
            if not frames:
                continue
            df = pd.concat(frames, ignore_index=True)
        else:
            try:
                df = pd.read_csv(stability_files_by_scope[scope_dir])
            except Exception:
                continue

        pairs.extend(_pairs_from_rows(df, scope_label, scope_dir, videos_dir))

    # Within each scope block, show the best-EVIDENCED pairs first (highest
    # real bin-overlap match_score) rather than each cluster's own isolated
    # "how much did I individually change" score -- a rater should see the
    # comparisons this tool is most confident are a genuine match before
    # the weaker, more-forced ones. Pairs with no match_score (contingency
    # data unavailable, rank-zip fallback used) sort after all scored ones.
    pairs.sort(key=lambda p: (p["scope"], p["match_score"] is None, -(p["match_score"] or 0)))
    return pairs


# ── atomic persistence helpers ───────────────────────────────────────────

def _atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _rewrite_csv_from_jsonl(jsonl_path: Path, csv_path: Path):
    records = _load_jsonl(jsonl_path)
    if not records:
        return
    df = pd.DataFrame(records)
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)


# ── plotting (self-styled to match cube_video_explorer's light palette,
#    kept independent of cube_core.py so this lightweight rating tool
#    doesn't drag in cube_core's heavy analysis-stack imports) ───────────

def _style_fig(fig, ax):
    fig.patch.set_facecolor(C["panel"])
    ax.set_facecolor(C["panel2"])
    ax.tick_params(colors=C["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(C["border"])
    ax.title.set_color(C["text"])
    ax.xaxis.label.set_color(C["text"])
    ax.yaxis.label.set_color(C["text"])


def plot_summary_overview(summary_df: pd.DataFrame):
    fig = plt.Figure(figsize=(8, 4), dpi=100)
    ax = fig.add_subplot(111)
    _style_fig(fig, ax)
    if summary_df.empty:
        ax.text(0.5, 0.5, "No ratings yet", ha="center", va="center", color=C["text_dim"])
        return fig
    x = list(range(len(summary_df)))
    zeros = pd.Series(0.0, index=summary_df.index)
    ax.bar([i - 0.25 for i in x], summary_df.get("difference_mean", zeros), width=0.25,
           label="Difference (A vs B)", color=C["accent"])
    ax.bar(x, summary_df.get("distinct_relative_mean", zeros), width=0.25,
           label="More distinct (0=A, 10=B)", color=C["accent3"])
    ax.bar([i + 0.25 for i in x], summary_df.get("consistency_relative_mean", zeros), width=0.25,
           label="More consistent (0=A, 10=B)", color=C["accent2"])
    # The two relative sliders are centered on 5 ("about equal"), not on 0
    # like Difference -- a dashed reference line at 5 makes that centering
    # visible at a glance instead of misreading a bar near 5 as "middling".
    ax.axhline(5, color=C["text_dim"], linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["comparison_key"], rotation=40, ha="right", fontsize=7)
    ax.set_ylim(0, 10)
    ax.legend(fontsize=7)
    ax.set_title("Mean scores by comparison  (dashed line = \"about equal\" for the relative sliders)")
    fig.tight_layout()
    return fig


def plot_one_comparison(key: str, sub: pd.DataFrame):
    fig = plt.Figure(figsize=(7, 4), dpi=100)
    ax = fig.add_subplot(111)
    _style_fig(fig, ax)
    cols_present = [f for f, _ in SCORE_FIELDS if f in sub.columns]
    data = [sub[c].dropna().values for c in cols_present]
    colors = [C["accent"], C["accent2"], C["accent3"], C["accent4"], C["accent5"], C["btn_hover"]]
    bp = ax.boxplot(data, labels=[c.replace("_", " ") for c in cols_present], patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylim(0, 10)
    ax.set_title(f"{key}  (n={len(sub)} ratings)", fontsize=10)
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    fig.tight_layout()
    return fig


def plot_votes_overview(summary_df: pd.DataFrame):
    """Stacked bar chart of the discrete "which is better" choice votes,
    one stack per comparison -- the analysis counterpart to the choice
    question, which is categorical and so can't go in the 0-10 slider
    chart above."""
    fig = plt.Figure(figsize=(8, 3.5), dpi=100)
    ax = fig.add_subplot(111)
    _style_fig(fig, ax)
    vote_cols = [f"vote_{v}" for v, _ in CHOICE_OPTIONS]
    if summary_df.empty or not all(c in summary_df.columns for c in vote_cols):
        ax.text(0.5, 0.5, "No choice votes yet", ha="center", va="center", color=C["text_dim"])
        return fig
    x = list(range(len(summary_df)))
    colors = [C["accent"], C["btn_hover"], C["accent3"]]
    bottoms = [0.0] * len(summary_df)
    for (value, label), color in zip(CHOICE_OPTIONS, colors):
        vals = summary_df[f"vote_{value}"].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottoms, label=label, color=color, width=0.6)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["comparison_key"], rotation=40, ha="right", fontsize=7)
    ax.legend(fontsize=7)
    ax.set_title("\"Which is the better behaviour?\" votes by comparison")
    fig.tight_layout()
    return fig


# ── rating dialog (separate window, per the requirement) ────────────────

class RatingDialog(tk.Toplevel):
    def __init__(self, app: "App", comparison: dict):
        super().__init__(app.root)
        self.app = app
        self.comparison = comparison
        self.title("Rate this comparison")
        self.configure(bg=C["bg"])
        self.geometry("600x650")
        self.minsize(480, 380)
        _raise_window(self)

        draft = app.load_draft(comparison["rater"], comparison["comparison_key"])

        # Footer (Submit/Close/Next) is packed FIRST with side="bottom" so
        # it's always visible and never scrolls out of view, regardless of
        # how tall the scrollable content above it is or how small the
        # window gets resized.
        if app.queue_mode:
            # Same big, unmissable navigation available from the main
            # window, duplicated here so a rater never has to close this
            # dialog first just to move on to the next comparison.
            nav_row = tk.Frame(self, bg=C["bg"])
            nav_row.pack(side="bottom", fill="x", padx=14, pady=(0, 4))
            tk.Button(nav_row, text="◀ Prev", command=self._nav_prev,
                      bg=C["btn"], fg=C["text"], relief="flat", padx=12, pady=10,
                      font=("Segoe UI", 10, "bold"), cursor="hand2").pack(side="left")
            tk.Button(nav_row, text="▶   NEXT COMPARISON (skip, don't submit)   ▶",
                      command=self._nav_next,
                      bg=C["accent"], fg="white", relief="flat", padx=16, pady=10,
                      font=("Segoe UI", 11, "bold"), cursor="hand2").pack(
                side="left", fill="x", expand=True, padx=(10, 0))

        btn_row = tk.Frame(self, bg=C["bg"])
        btn_row.pack(side="bottom", fill="x", padx=14, pady=10)
        tk.Button(btn_row, text="Close (keep draft)", command=self._close_keep_draft,
                  bg=C["btn"], fg=C["text"], relief="flat", padx=14, pady=10,
                  font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side="left")
        tk.Button(btn_row, text="✓  SUBMIT  →  NEXT COMPARISON", command=self._submit,
                  bg=C["accent2"], fg="white", font=("Segoe UI", 12, "bold"),
                  relief="flat", padx=18, pady=10, cursor="hand2").pack(
            side="right", fill="x", expand=True, padx=(10, 0))

        # Scrollable body -- this is the fix for "can't scroll down": the
        # previous version packed everything directly into the Toplevel
        # with a fixed height, so on a small screen (or a long notes entry)
        # the sliders/notes/buttons could run off the bottom of the window
        # with no way to reach them.
        body_border = tk.Frame(self, bg=C["border"], bd=1)
        body_border.pack(side="top", fill="both", expand=True, padx=8, pady=(8, 0))
        canvas = tk.Canvas(body_border, bg=C["bg"], highlightthickness=0)
        sb = tk.Scrollbar(body_border, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        _bind_wheel_to_canvas(self, canvas)

        body = tk.Frame(canvas, bg=C["bg"])
        body_win = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(body_win, width=e.width))

        tk.Label(body, text=f"Rater: {comparison['rater']}",
                 font=("Segoe UI", 10, "bold"), bg=C["bg"], fg=C["text"]
                 ).pack(anchor="w", padx=14, pady=(12, 2))
        tk.Label(body, text="Sliders autosave as you move them, even if you close this window.",
                 font=("Segoe UI", 8), bg=C["bg"], fg=C["text_dim"]
                 ).pack(anchor="w", padx=14, pady=(0, 8))

        self.vars = {}
        for field, prompt in SCORE_FIELDS:
            frm = tk.Frame(body, bg=C["bg"])
            frm.pack(fill="x", padx=14, pady=5)
            tk.Label(frm, text=prompt, font=("Segoe UI", 9), bg=C["bg"], fg=C["text"],
                     wraplength=520, justify="left").pack(anchor="w")
            init_val = float(draft.get(field, 0.0)) if draft else 0.0
            var = tk.DoubleVar(master=self, value=init_val)
            self.vars[field] = var
            row = tk.Frame(frm, bg=C["bg"])
            row.pack(fill="x")
            val_lbl = tk.Label(row, text=f"{init_val:.0f}", font=("Segoe UI", 9, "bold"),
                                bg=C["bg"], fg=C["accent"], width=3)
            val_lbl.pack(side="right")

            def _on_change(v, val_lbl=val_lbl):
                val_lbl.configure(text=f"{float(v):.0f}")
                self._autosave()

            scale = tk.Scale(row, from_=0, to=10, orient="horizontal", variable=var,
                              command=_on_change, bg=C["bg"], highlightthickness=0,
                              troughcolor=C["panel2"], length=420, resolution=1)
            scale.pack(side="left", fill="x", expand=True)

        # The one discrete-choice question, separate from the sliders above.
        choice_frm = tk.Frame(body, bg=C["bg"])
        choice_frm.pack(fill="x", padx=14, pady=(4, 5))
        tk.Label(choice_frm, text=CHOICE_PROMPT, font=("Segoe UI", 9), bg=C["bg"], fg=C["text"],
                 wraplength=520, justify="left").pack(anchor="w")
        init_choice = draft.get(CHOICE_FIELD, "tie") if draft else "tie"
        self.choice_var = tk.StringVar(master=self, value=init_choice)
        choice_row = tk.Frame(choice_frm, bg=C["bg"])
        choice_row.pack(fill="x", pady=(2, 0))
        for value, label in CHOICE_OPTIONS:
            tk.Radiobutton(choice_row, text=label, value=value, variable=self.choice_var,
                            bg=C["bg"], fg=C["text"], selectcolor=C["panel2"],
                            activebackground=C["bg"], font=("Segoe UI", 9, "bold"),
                            command=self._autosave, cursor="hand2").pack(side="left", padx=(0, 18))

        tk.Label(body, text="Notes (optional):", font=("Segoe UI", 9, "bold"),
                 bg=C["bg"], fg=C["text"]).pack(anchor="w", padx=14, pady=(8, 0))
        self.notes = tk.Text(body, height=4, width=50)
        self.notes.pack(padx=14, pady=(2, 12), fill="x")
        if draft and draft.get("notes"):
            self.notes.insert("1.0", draft["notes"])
        self.notes.bind("<KeyRelease>", lambda e: self._autosave())

        self.protocol("WM_DELETE_WINDOW", self._close_keep_draft)
        self._autosave()

    def _current_state(self) -> dict:
        state = {f: self.vars[f].get() for f, _ in SCORE_FIELDS}
        state[CHOICE_FIELD] = self.choice_var.get()
        state["notes"] = self.notes.get("1.0", "end").strip()
        return state

    def _autosave(self):
        self.app.save_draft(self.comparison["rater"], self.comparison["comparison_key"],
                             self._current_state())

    def _close_keep_draft(self):
        self._autosave()
        self.destroy()

    def _nav_next(self):
        self._autosave()
        self.destroy()
        self.app.next_queue_pair()

    def _nav_prev(self):
        self._autosave()
        self.destroy()
        self.app.prev_queue_pair()

    def _submit(self):
        state = self._current_state()
        record = dict(self.comparison)
        record.update(state)
        record["timestamp_iso"] = datetime.now(timezone.utc).isoformat()
        self.app.submit_response(record)
        self.app.clear_draft(self.comparison["rater"], self.comparison["comparison_key"])
        self.destroy()
        self.app.on_comparison_submitted()


# ── new-comparison picker dialog ─────────────────────────────────────────

class NewComparisonDialog(tk.Toplevel):
    def __init__(self, app: "App"):
        super().__init__(app.root)
        self.app = app
        self.title("New Comparison")
        self.configure(bg=C["bg"])
        _raise_window(self)

        conditions = sorted(app.tree.keys())

        tk.Label(self, text="Rater name:", bg=C["bg"], fg=C["text"],
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=2,
                                                       sticky="w", padx=14, pady=(12, 0))
        tk.Entry(self, textvariable=app.rater_var, width=42).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=14, pady=(2, 10))

        self.cond_a = tk.StringVar(master=self)
        self.cond_b = tk.StringVar(master=self)
        self.clus_a = tk.StringVar(master=self)
        self.clus_b = tk.StringVar(master=self)

        def _build_side(label, cond_var, clus_var, col):
            frm = tk.LabelFrame(self, text=label, bg=C["bg"], fg=C["text"],
                                 font=("Segoe UI", 10, "bold"))
            frm.grid(row=2, column=col, padx=10, pady=6, sticky="n")
            tk.Label(frm, text="Condition:", bg=C["bg"], fg=C["text"]).pack(anchor="w", padx=8, pady=(6, 0))
            cond_cb = ttk.Combobox(frm, textvariable=cond_var, values=conditions,
                                    state="readonly", width=30)
            cond_cb.pack(padx=8, pady=4)
            tk.Label(frm, text="Cluster:", bg=C["bg"], fg=C["text"]).pack(anchor="w", padx=8)
            clus_cb = ttk.Combobox(frm, textvariable=clus_var, values=[], state="readonly", width=30)
            clus_cb.pack(padx=8, pady=(4, 8))

            def _on_cond(*_):
                ids = sorted(app.tree.get(cond_var.get(), {}).keys())
                clus_cb["values"] = [str(i) for i in ids]
                if ids:
                    clus_var.set(str(ids[0]))

            cond_cb.bind("<<ComboboxSelected>>", _on_cond)
            return _on_cond

        init_a = _build_side("Cluster A", self.cond_a, self.clus_a, 0)
        init_b = _build_side("Cluster B", self.cond_b, self.clus_b, 1)

        if conditions:
            self.cond_a.set(conditions[0])
            init_a()
            self.cond_b.set(conditions[1] if len(conditions) > 1 else conditions[0])
            init_b()
            if len(conditions) == 1:
                ids = sorted(app.tree[conditions[0]].keys())
                if len(ids) > 1:
                    self.clus_b.set(str(ids[1]))

        tk.Button(self, text="Start Comparison", command=self._start,
                  bg=C["accent"], fg="white", font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=16, pady=8, cursor="hand2").grid(
            row=3, column=0, columnspan=2, pady=(6, 14))

    def _start(self):
        if not self.app.rater_var.get().strip():
            messagebox.showwarning("Rater name required", "Please enter your name before starting.")
            return
        try:
            cond_a, clus_a = self.cond_a.get(), int(self.clus_a.get())
            cond_b, clus_b = self.cond_b.get(), int(self.clus_b.get())
        except (ValueError, tk.TclError):
            messagebox.showwarning("Pick clusters", "Please select a condition and cluster on both sides.")
            return
        self.destroy()
        self.app.start_comparison(cond_a, clus_a, cond_b, clus_b)


# ── main application ─────────────────────────────────────────────────────

class App:
    def __init__(self, root: tk.Tk, video_root: Path, tree: dict, pairs: list):
        self.root = root
        self.video_root = video_root
        self.tree = tree
        self.pairs = pairs                     # [] -> freeform mode; else -> queue mode
        self.queue_mode = bool(pairs)
        self.queue_index = 0
        self.queue_finished = False
        self.current = None

        self.out_dir = video_root / "_validation_results"
        self.drafts_dir = self.out_dir / "_drafts"
        self.analysis_dir = self.out_dir / "analysis"
        self.responses_jsonl = self.out_dir / "responses.jsonl"
        self.responses_csv = self.out_dir / "responses.csv"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.rater_var = tk.StringVar(master=root, value="")

        root.title("CUBE — Blind Cluster Validation")
        root.geometry("1440x920")
        root.configure(bg=C["bg"])

        n_conditions = len(tree)
        n_clusters = sum(len(v) for v in tree.values())
        n_clips = sum(len(paths) for cl in tree.values() for paths in cl.values())
        header = tk.Frame(root, bg=C["header"])
        header.pack(fill="x")
        tk.Label(header, text=f"Root: {video_root}", bg=C["header"], fg=C["header_text"],
                 font=("Segoe UI", 9)).pack(side="left", padx=12, pady=6)
        if self.queue_mode:
            n_scopes = len({p["scope"] for p in pairs})
            mode_text = f"{len(pairs)} matched comparison(s) queued across {n_scopes} scope(s)"
        else:
            mode_text = (f"{n_conditions} condition(s), {n_clusters} cluster(s), {n_clips} clip(s) found "
                         f"— no most_changed_clusters_index.csv/cluster_stability.csv found, manual pairing mode")
        tk.Label(header, text=mode_text, bg=C["header"], fg=C["header_text"],
                 font=("Segoe UI", 9)).pack(side="right", padx=12, pady=6)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)
        rate_tab = tk.Frame(self.notebook, bg=C["bg"])
        results_tab = tk.Frame(self.notebook, bg=C["bg"])
        self.notebook.add(rate_tab, text="Rate Comparisons")
        self.notebook.add(results_tab, text="Results")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._build_rate_tab(rate_tab)
        self._build_results_tab(results_tab)

        if self.queue_mode:
            self.load_queue_pair(0)

    # ── Rate Comparisons tab ────────────────────────────────────────────

    def _build_rate_tab(self, parent):
        top = tk.Frame(parent, bg=C["bg"])
        top.pack(fill="x", padx=10, pady=8)
        tk.Label(top, text="Rater name:", bg=C["bg"], fg=C["text"],
                 font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Entry(top, textvariable=self.rater_var, width=26).pack(side="left", padx=(4, 16))

        if not self.queue_mode:
            tk.Button(top, text="New Comparison", command=self.open_new_comparison_dialog,
                      bg=C["accent"], fg="white", relief="flat", padx=10, pady=6,
                      font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side="left")

        self.rate_btn = tk.Button(top, text="Open Rating Form", command=self.open_rating_dialog,
                                   bg=C["accent2"], fg="white", relief="flat", padx=10, pady=6,
                                   font=("Segoe UI", 9, "bold"), cursor="hand2", state="disabled")
        self.rate_btn.pack(side="left", padx=8)
        self.status_lbl = tk.Label(top, text="No comparison loaded.",
                                    bg=C["bg"], fg=C["text_dim"], font=("Segoe UI", 10, "bold"))
        self.status_lbl.pack(side="left", padx=16)

        panes = tk.Frame(parent, bg=C["bg"])
        panes.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        left_box = tk.Frame(panes, bg=C["panel"], highlightbackground=C["border"], highlightthickness=1)
        left_box.pack(side="left", fill="both", expand=True, padx=(0, 5))
        right_box = tk.Frame(panes, bg=C["panel"], highlightbackground=C["border"], highlightthickness=1)
        right_box.pack(side="left", fill="both", expand=True, padx=(5, 0))

        tk.Label(left_box, text="Group A", font=("Segoe UI", 13, "bold"),
                 bg=C["panel"], fg=C["text"]).pack(pady=6)
        tk.Label(right_box, text="Group B", font=("Segoe UI", 13, "bold"),
                 bg=C["panel"], fg=C["text"]).pack(pady=6)

        self.left_canvas, self.left_player = self._make_player_area(left_box)
        self.right_canvas, self.right_player = self._make_player_area(right_box)

        # Big, unmissable navigation bar -- separate from the (easy to
        # overlook) small buttons in the top bar, so "how do I move on to
        # the next cluster comparison" always has one obvious answer,
        # whether or not a rating was actually submitted for this pair.
        if self.queue_mode:
            nav = tk.Frame(parent, bg=C["bg"])
            nav.pack(fill="x", padx=10, pady=(0, 12))
            tk.Button(nav, text="◀ Prev", command=self.prev_queue_pair,
                      bg=C["btn"], fg=C["text"], relief="flat", padx=16, pady=14,
                      font=("Segoe UI", 11, "bold"), cursor="hand2").pack(side="left")
            self.next_btn = tk.Button(nav, text="▶   NEXT COMPARISON   ▶",
                                       command=self.next_queue_pair,
                                       bg=C["accent"], fg="white", relief="flat",
                                       padx=20, pady=14, font=("Segoe UI", 14, "bold"),
                                       cursor="hand2")
            self.next_btn.pack(side="left", fill="x", expand=True, padx=(10, 0))

    def _make_player_area(self, parent):
        border = tk.Frame(parent, bg=C["border"], bd=1)
        border.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        canvas = tk.Canvas(border, bg=C["panel2"], highlightthickness=0)
        sb = tk.Scrollbar(border, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        _bind_wheel_to_canvas(self.root, canvas)

        player = MultiPlayer(canvas)
        win = canvas.create_window((0, 0), window=player, anchor="nw")
        state = {"width": 480}

        player.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        def _on_canvas_configure(e):
            state["width"] = e.width
            canvas.itemconfig(win, width=max(e.width, player.winfo_reqwidth()))

        canvas.bind("<Configure>", _on_canvas_configure)
        canvas._avail_width_state = state
        return canvas, player

    def open_new_comparison_dialog(self):
        NewComparisonDialog(self)

    def _load_comparison(self, comparison_key, cond_a, clus_a, paths_a, cond_b, clus_b, paths_b,
                          status_prefix="", match_score=None):
        """Shared loader for both freeform (start_comparison) and queue
        (load_queue_pair) modes: samples up to MAX_CLIPS_PER_SIDE clips per
        side, randomizes which real side renders left vs. right (blinding),
        and loads both MultiPlayers."""
        sample_a = random.sample(paths_a, min(len(paths_a), MAX_CLIPS_PER_SIDE))
        sample_b = random.sample(paths_b, min(len(paths_b), MAX_CLIPS_PER_SIDE))

        left_is_a = random.random() < 0.5
        left_paths, right_paths = (sample_a, sample_b) if left_is_a else (sample_b, sample_a)

        self.current = dict(
            comparison_key=comparison_key,
            condition_a=cond_a, cluster_a=clus_a,
            condition_b=cond_b, cluster_b=clus_b,
            left_is=("A" if left_is_a else "B"),
            n_clips_left=len(left_paths), n_clips_right=len(right_paths),
            match_score=match_score,
        )

        self.left_player.load(left_paths, speed=SPEED, zoom=ZOOM,
                               avail_width=self.left_canvas._avail_width_state["width"])
        self.right_player.load(right_paths, speed=SPEED, zoom=ZOOM,
                                avail_width=self.right_canvas._avail_width_state["width"])

        # match_score = that pair's real bin-overlap share (from
        # data/contingency_counts.csv + frac_of_*.csv) -- shown so a rater
        # can see when this tool itself has low confidence in the pairing
        # (can happen with only ~5 exported clusters per side: even the
        # best available partner within that small subset may be weak).
        if match_score is None:
            match_note = ""
        elif match_score >= 0.15:
            match_note = f"  (match confidence: high, {match_score:.0%} bin overlap)"
        elif match_score >= 0.05:
            match_note = f"  (match confidence: moderate, {match_score:.0%} bin overlap)"
        else:
            match_note = f"  (match confidence: LOW, only {match_score:.0%} bin overlap -- weak/forced pairing)"
        self.status_lbl.configure(
            text=f"{status_prefix}Group A ({len(left_paths)} clip(s))  vs  "
                 f"Group B ({len(right_paths)} clip(s)){match_note}")
        self.rate_btn.configure(state="normal")

    def start_comparison(self, cond_a, clus_a, cond_b, clus_b):
        self._load_comparison(
            comparison_key=f"{cond_a}__c{clus_a}__vs__{cond_b}__c{clus_b}",
            cond_a=cond_a, clus_a=clus_a, paths_a=self.tree[cond_a][clus_a],
            cond_b=cond_b, clus_b=clus_b, paths_b=self.tree[cond_b][clus_b],
            status_prefix="Comparing: ")

    # ── queue mode (relevant-pairs-only, auto-advance) ──────────────────

    def load_queue_pair(self, index: int):
        if not self.pairs:
            return
        index = max(0, min(index, len(self.pairs) - 1))
        self.queue_index = index
        self.queue_finished = False
        # Undo _finish_queue's "done" restyle in case we're navigating back
        # (Prev) out of the finished state -- otherwise the button would be
        # stuck disabled/grey even though there's a real pair loaded again.
        self.next_btn.configure(text="▶   NEXT COMPARISON   ▶", bg=C["accent"], state="normal")
        p = self.pairs[index]
        self._load_comparison(
            comparison_key=p["comparison_key"],
            cond_a=p["condition_a"], clus_a=p["cluster_a"], paths_a=p["paths_a"],
            cond_b=p["condition_b"], clus_b=p["cluster_b"], paths_b=p["paths_b"],
            status_prefix=f"Comparison {index + 1} of {len(self.pairs)}:  ",
            match_score=p.get("match_score"))

    def next_queue_pair(self):
        if not self.pairs:
            return
        if self.queue_index >= len(self.pairs) - 1:
            self._finish_queue()
        else:
            self.load_queue_pair(self.queue_index + 1)

    def prev_queue_pair(self):
        if self.pairs:
            self.load_queue_pair(self.queue_index - 1)

    def _finish_queue(self):
        """Reached the end of the relevant-comparison set -- stop here
        rather than silently looping back to comparison 1 (which could
        make a rater re-rate the whole set without realizing it). Prev
        still works to go back and review/re-rate an earlier pair.

        Made deliberately loud (a real popup, not just a status-label
        sentence easy to miss next to a big blue button that otherwise
        looks like it should still do something) plus a restyled/disabled
        Next button, so reaching the end is never mistaken for the button
        having silently stopped working.
        """
        already_finished = self.queue_finished
        self.queue_finished = True
        self.current = None
        self.left_player.load([])
        self.right_player.load([])
        self.rate_btn.configure(state="disabled")
        self.next_btn.configure(text="✔   ALL COMPARISONS RATED", bg=C["btn"], state="disabled")
        self.status_lbl.configure(
            text=f"All {len(self.pairs)} relevant comparisons rated. "
                 f"Use ◀ Prev to review or re-rate an earlier one.")
        if not already_finished:
            messagebox.showinfo(
                "Queue complete",
                f"You've rated (or skipped through) all {len(self.pairs)} relevant comparisons.\n\n"
                "Use ◀ Prev to go back and review or re-rate an earlier one.")

    def on_comparison_submitted(self):
        """Called after a rating is submitted -- advance automatically in
        queue mode; reopen the free picker in freeform mode."""
        if self.queue_mode:
            self.next_queue_pair()
        else:
            self.open_new_comparison_dialog()

    def open_rating_dialog(self):
        if self.current is None:
            return
        rater = self.rater_var.get().strip()
        if not rater:
            messagebox.showwarning("Rater name required", "Please enter your name first.")
            return
        comparison = dict(self.current)
        comparison["rater"] = rater
        RatingDialog(self, comparison)

    # ── draft / response persistence ────────────────────────────────────

    def _draft_path(self, rater, key) -> Path:
        return self.drafts_dir / f"{_safe(rater)}__{_safe(key)}.json"

    def save_draft(self, rater, key, state: dict):
        _atomic_write_text(self._draft_path(rater, key), json.dumps(state, indent=2))

    def load_draft(self, rater, key):
        p = self._draft_path(rater, key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def clear_draft(self, rater, key):
        p = self._draft_path(rater, key)
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    def submit_response(self, record: dict):
        _atomic_append_jsonl(self.responses_jsonl, record)
        _rewrite_csv_from_jsonl(self.responses_jsonl, self.responses_csv)

    # ── Results tab ──────────────────────────────────────────────────────

    def _build_results_tab(self, parent):
        top = tk.Frame(parent, bg=C["bg"])
        top.pack(fill="x", padx=10, pady=8)
        tk.Button(top, text="Refresh", command=self.refresh_results,
                  bg=C["accent"], fg="white", relief="flat", padx=10, pady=6,
                  font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side="left")
        tk.Button(top, text="Export Analysis", command=self.export_analysis,
                  bg=C["accent2"], fg="white", relief="flat", padx=10, pady=6,
                  font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side="left", padx=8)
        self.results_info = tk.Label(top, text="", bg=C["bg"], fg=C["text_dim"])
        self.results_info.pack(side="left", padx=16)

        body = tk.Frame(parent, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        tk.Label(body, text="Summary by comparison:", bg=C["bg"], fg=C["text"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        table_frame = tk.Frame(body, bg=C["bg"])
        table_frame.pack(fill="x", pady=(2, 0))
        cols = (["comparison_key", "n_ratings", "n_raters"]
                + [f"{c}_mean" for c, _ in SCORE_FIELDS]
                + [f"vote_{v}" for v, _ in CHOICE_OPTIONS])
        vote_headings = {f"vote_{v}": f"votes: {lbl}" for v, lbl in CHOICE_OPTIONS}
        self.summary_tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=8)
        for c in cols:
            self.summary_tree.heading(c, text=vote_headings.get(c, c))
            self.summary_tree.column(c, width=95, anchor="center")
        self.summary_tree.pack(side="left", fill="x", expand=True)
        tsb = tk.Scrollbar(table_frame, orient="vertical", command=self.summary_tree.yview)
        self.summary_tree.configure(yscrollcommand=tsb.set)
        tsb.pack(side="right", fill="y")

        tk.Label(body, text="Repeat ratings (same rater, same comparison — test-retest):",
                 bg=C["bg"], fg=C["text"], font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(10, 0))
        rep_frame = tk.Frame(body, bg=C["bg"])
        rep_frame.pack(fill="x", pady=(2, 0))
        self.repeat_tree = ttk.Treeview(rep_frame, columns=["rater", "comparison_key", "n_submissions"],
                                         show="headings", height=4)
        for c in ["rater", "comparison_key", "n_submissions"]:
            self.repeat_tree.heading(c, text=c)
            self.repeat_tree.column(c, width=160, anchor="center")
        self.repeat_tree.pack(side="left", fill="x", expand=True)

        self.chart_frame = tk.Frame(body, bg=C["panel"])
        self.chart_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.chart_canvas = None

        self.votes_chart_frame = tk.Frame(body, bg=C["panel"])
        self.votes_chart_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.votes_chart_canvas = None

    def _on_tab_changed(self, event):
        if self.notebook.tab(self.notebook.select(), "text") == "Results":
            self.refresh_results()

    def load_responses(self) -> pd.DataFrame:
        records = _load_jsonl(self.responses_jsonl)
        return pd.DataFrame(records) if records else pd.DataFrame()

    def summarize(self, df: pd.DataFrame):
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        score_cols = [f for f, _ in SCORE_FIELDS if f in df.columns]
        agg = {"n_ratings": ("rater", "count"), "n_raters": ("rater", "nunique")}
        for c in score_cols:
            agg[f"{c}_mean"] = (c, "mean")
            agg[f"{c}_sd"] = (c, "std")
        summary = df.groupby("comparison_key").agg(**agg).reset_index()

        # Vote counts for the discrete choice question (older rows collected
        # before this question existed simply have no CHOICE_FIELD column,
        # and contribute 0 votes rather than erroring).
        vote_cols = [f"vote_{value}" for value, _ in CHOICE_OPTIONS]
        if CHOICE_FIELD in df.columns:
            votes = df.groupby(["comparison_key", CHOICE_FIELD]).size().unstack(fill_value=0)
            for value, _label in CHOICE_OPTIONS:
                if value not in votes.columns:
                    votes[value] = 0
            votes = votes[[v for v, _ in CHOICE_OPTIONS]]
            votes.columns = vote_cols
            summary = summary.merge(votes.reset_index(), on="comparison_key", how="left")
        for col in vote_cols:
            if col not in summary.columns:
                summary[col] = 0
            summary[col] = summary[col].fillna(0).astype(int)

        rep = df.groupby(["rater", "comparison_key"]).size().reset_index(name="n_submissions")
        repeats = rep[rep["n_submissions"] > 1]
        return summary, repeats

    def refresh_results(self):
        df = self.load_responses()
        summary, repeats = self.summarize(df)

        for iid in self.summary_tree.get_children():
            self.summary_tree.delete(iid)
        for _, row in summary.iterrows():
            vals = [row["comparison_key"], int(row["n_ratings"]), int(row["n_raters"])]
            for c, _ in SCORE_FIELDS:
                col = f"{c}_mean"
                vals.append(f"{row[col]:.1f}" if col in summary.columns and pd.notna(row.get(col)) else "-")
            for v, _ in CHOICE_OPTIONS:
                col = f"vote_{v}"
                vals.append(int(row[col]) if col in summary.columns else 0)
            self.summary_tree.insert("", "end", values=vals)

        for iid in self.repeat_tree.get_children():
            self.repeat_tree.delete(iid)
        for _, row in repeats.iterrows():
            self.repeat_tree.insert("", "end",
                                     values=[row["rater"], row["comparison_key"], int(row["n_submissions"])])

        self.results_info.configure(
            text=f"{len(df)} total rating(s) across {summary.shape[0]} comparison(s)"
            if not df.empty else "No ratings collected yet.")

        if self.chart_canvas is not None:
            self.chart_canvas.get_tk_widget().destroy()
        fig = plot_summary_overview(summary)
        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

        if self.votes_chart_canvas is not None:
            self.votes_chart_canvas.get_tk_widget().destroy()
        votes_fig = plot_votes_overview(summary)
        self.votes_chart_canvas = FigureCanvasTkAgg(votes_fig, master=self.votes_chart_frame)
        self.votes_chart_canvas.draw()
        self.votes_chart_canvas.get_tk_widget().pack(fill="both", expand=True)

    def export_analysis(self):
        df = self.load_responses()
        summary, repeats = self.summarize(df)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        summary_path = self.analysis_dir / "summary_by_comparison.csv"
        repeats_path = self.analysis_dir / "repeat_ratings.csv"
        tmp = summary_path.with_name(summary_path.name + ".tmp")
        summary.to_csv(tmp, index=False)
        os.replace(tmp, summary_path)
        tmp = repeats_path.with_name(repeats_path.name + ".tmp")
        repeats.to_csv(tmp, index=False)
        os.replace(tmp, repeats_path)

        if not summary.empty:
            fig = plot_summary_overview(summary)
            fig.savefig(self.analysis_dir / "rater_agreement.png", dpi=150,
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            votes_fig = plot_votes_overview(summary)
            votes_fig.savefig(self.analysis_dir / "vote_breakdown.png", dpi=150,
                               facecolor=votes_fig.get_facecolor())
            plt.close(votes_fig)
            for key in summary["comparison_key"]:
                sub = df[df["comparison_key"] == key]
                fig2 = plot_one_comparison(key, sub)
                fig2.savefig(self.analysis_dir / f"{_safe(key)}_scores.png", dpi=150,
                             facecolor=fig2.get_facecolor())
                plt.close(fig2)

        messagebox.showinfo("Export complete", f"Analysis exported to:\n{self.analysis_dir}")
        self.refresh_results()


# ── entry point ───────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Select a folder to search (recursively) for cluster_NN example-clip videos")
    if not folder:
        root.destroy()
        return
    video_root = Path(folder)
    tree = scan_videos(video_root)
    if not tree:
        messagebox.showerror(
            "No videos found",
            f"No cluster_NN\\*.mp4 files were found recursively under:\n{video_root}")
        root.destroy()
        return
    pairs = find_relevant_pairs(video_root)
    if not pairs:
        messagebox.showinfo(
            "No matched comparisons found",
            "No most_changed_clusters_index.csv or cluster_stability.csv was found "
            "under this folder (or none of their clusters have matching video on "
            "both sides), so relevant cluster pairs can't be determined "
            "automatically.\n\n"
            "Falling back to manual pairing mode -- you'll pick which two "
            "clusters to compare yourself.")
    root.deiconify()
    App(root, video_root, tree, pairs)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            messagebox.showerror("Fatal error", traceback.format_exc())
        except Exception:
            pass
        raise
