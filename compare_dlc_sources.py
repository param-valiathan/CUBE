"""
compare_dlc_sources.py

Stage 1 of the CUBE validation pipeline: compares raw DLC pose-tracking
OUTPUT ONLY (no B-SOiD, no CUBE clustering yet -- those are separate,
later scripts) across public reference datasets downloaded for testing.

Every source is loaded through cube_core.load_dlc_file(), CUBE's own DLC
normaliser/interpolator, so every metric reflects exactly what CUBE itself
would see as input.

Design goal: once you run CUBE's own DLC step on the paradigm-matched
videos in F:\\CUBE_test_data\\<paradigm>\\*\\videos\\, point CUBE_RUN_DIRS
(below) at the resulting DLC output folders. Re-running this script then
adds your CUBE tracking as a new source *within its matching paradigm*
and compares it against that paradigm's published reference tracking.
Until then, every source here is "reference/published" data and is drawn
in the peach/magenta palette -- CUBE blue (#3B79A4) is reserved and stays
unused until a CUBE_RUN_DIRS entry is filled in, per project convention
(CUBE blue == only ever CUBE's own analysis output).
"""

from __future__ import annotations

import re
import sys
import warnings
import itertools
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sstats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from cube_core import load_dlc_file  # noqa: E402

# ------------------------------------------------------------------
#  CONFIG
# ------------------------------------------------------------------

DATA_ROOT   = Path("F:/CUBE_test_data")
OUTPUT_ROOT = Path(__file__).parent / "dlc_output_comparison"
LIKELIHOOD_THRESH = 0.3

# Fill these in once you've run CUBE's own DLC step on the matching
# paradigm's videos folder. None = not yet available (default).
CUBE_RUN_DIRS = {
    "open_field":         None,
    "elevated_plus_maze": None,
    "forced_swim_test":   None,
}

SOURCE_MANIFEST = [
    dict(name="bsoid_demo", paradigm="open_field", fps_assumed=30,
         path=DATA_ROOT / "open_field/bsoid_demo/yttri-bottomup_dlc-model/examples/dlc_tracking_and_labels"),
    dict(name="sturman_oft", paradigm="open_field", fps_assumed=25,
         path=DATA_ROOT / "open_field/sturman_oft/dlc_tracking_and_labels/Output_DLC"),
    dict(name="patterns2025_batch1", paradigm="novel_object", fps_assumed=30,
         path=DATA_ROOT / "novel_object/zenodo_unsupervised_eval/dlc_tracking_and_labels/extracted_batch1"),
    dict(name="patterns2025_batch2", paradigm="novel_object", fps_assumed=30,
         path=DATA_ROOT / "novel_object/zenodo_unsupervised_eval/dlc_tracking_and_labels/extracted_batch2"),
    dict(name="sturman_epm", paradigm="elevated_plus_maze", fps_assumed=25,
         path=DATA_ROOT / "elevated_plus_maze/sturman_epm/dlc_tracking_and_labels/Output_DLC"),
    dict(name="sturman_fst", paradigm="forced_swim_test", fps_assumed=25,
         path=DATA_ROOT / "forced_swim_test/sturman_fst/dlc_tracking_and_labels/Output_DLC"),
]

for _paradigm, _dir in CUBE_RUN_DIRS.items():
    if _dir is not None:
        SOURCE_MANIFEST.append(dict(
            name=f"cube_{_paradigm}", paradigm=_paradigm, fps_assumed=None,
            path=Path(_dir), is_cube=True,
        ))

for _s in SOURCE_MANIFEST:
    _s.setdefault("is_cube", False)

# Expandable two-hue reference palette (peach / magenta families).
# Add more hex values to either list to support additional reference
# sources without touching plotting code.
CUBE_BLUE = "#3B79A4"
PALETTE = {
    "peach":   ["#F2A679", "#D98A5A", "#F7C39D", "#C9723C"],
    "magenta": ["#E8A0BF", "#C97A9E", "#F0BFD6", "#B85C85"],
    "cube":    CUBE_BLUE,
}

# canonical landmark aliasing for best-effort cross-model comparison
_CANON_PATTERNS = [
    (re.compile(r"nose|snout", re.I),                       "nose"),
    (re.compile(r"^head(?!.*base)|headcent", re.I),          "head"),
    (re.compile(r"neck", re.I),                              "neck"),
    (re.compile(r"body.?cent|centroid|^center$|^body$", re.I), "body_center"),
    (re.compile(r"tail.?base|tailbase", re.I),               "tail_base"),
]


def canonical_landmark(bp_name: str):
    for pat, canon in _CANON_PATTERNS:
        if pat.search(bp_name):
            return canon
    return None


# ------------------------------------------------------------------
#  COLOR ASSIGNMENT
# ------------------------------------------------------------------

def build_color_map(sources: list[dict]) -> dict:
    """source name -> hex color. Non-CUBE sources alternate peach/magenta
    shades in manifest order; CUBE sources always get CUBE_BLUE."""
    colors = {}
    hue_cycle = itertools.cycle(["peach", "magenta"])
    shade_idx = {"peach": 0, "magenta": 0}
    for s in sources:
        if s["is_cube"]:
            colors[s["name"]] = CUBE_BLUE
            continue
        hue = next(hue_cycle)
        shades = PALETTE[hue]
        colors[s["name"]] = shades[shade_idx[hue] % len(shades)]
        shade_idx[hue] += 1
    return colors


# ------------------------------------------------------------------
#  LOADING
# ------------------------------------------------------------------

@dataclass
class SessionData:
    source: str
    paradigm: str
    is_cube: bool
    session_id: str
    path: Path
    xy: np.ndarray
    bodyparts: list
    fps: float
    ll_fracs: dict
    ll_raw: np.ndarray


def discover_dlc_files(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    candidates = [p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in (".csv", ".h5", ".hdf5")]

    def stem_key(p: Path) -> str:
        return re.sub(r"\.(h5|hdf5|csv)$", "", p.name, flags=re.I)

    by_stem: dict[str, Path] = {}
    for p in sorted(candidates):
        key = stem_key(p)
        cur = by_stem.get(key)
        if cur is None or (cur.suffix.lower() == ".csv" and p.suffix.lower() in (".h5", ".hdf5")):
            by_stem[key] = p
    return sorted(by_stem.values())


def load_source(cfg: dict) -> list[SessionData]:
    files = discover_dlc_files(cfg["path"])
    records = []
    for f in files:
        try:
            xy, bodyparts, fps_hint, ll_fracs, _flat_held, ll_raw = load_dlc_file(
                f, likelihood_thresh=LIKELIHOOD_THRESH, return_quality=True)
        except Exception as e:
            warnings.warn(f"[{cfg['name']}] failed to load {f.name}: {e}")
            continue
        if xy.shape[0] < 10:
            continue
        fps = fps_hint or cfg["fps_assumed"] or 30.0
        records.append(SessionData(
            source=cfg["name"], paradigm=cfg["paradigm"], is_cube=cfg["is_cube"],
            session_id=f.stem, path=f, xy=xy, bodyparts=bodyparts, fps=fps,
            ll_fracs=ll_fracs, ll_raw=ll_raw,
        ))
    return records


# ------------------------------------------------------------------
#  METRICS
# ------------------------------------------------------------------

def compute_bodypart_metrics(sess: SessionData) -> pd.DataFrame:
    rows = []
    n_frames = sess.xy.shape[0]
    for i, bp in enumerate(sess.bodyparts):
        x = sess.xy[:, 2 * i]
        y = sess.xy[:, 2 * i + 1]
        vx, vy = np.diff(x), np.diff(y)
        vel_px_frame = np.sqrt(vx ** 2 + vy ** 2)
        vel_px_s = vel_px_frame * sess.fps
        jitter = np.abs(np.diff(vel_px_frame))  # |acceleration| proxy for tracking noise

        ll = sess.ll_raw[:, i]
        ll = ll[~np.isnan(ll)]
        mean_ll = float(np.mean(ll)) if len(ll) else np.nan

        rows.append(dict(
            source=sess.source, paradigm=sess.paradigm, is_cube=sess.is_cube,
            session_id=sess.session_id, bodypart=bp,
            canonical=canonical_landmark(bp),
            n_frames=n_frames, fps=sess.fps,
            mean_likelihood=mean_ll,
            interp_rate=sess.ll_fracs.get(bp, np.nan),
            mean_velocity_px_s=float(np.nanmean(vel_px_s)) if len(vel_px_s) else np.nan,
            std_velocity_px_s=float(np.nanstd(vel_px_s)) if len(vel_px_s) else np.nan,
            cv_velocity=(float(np.nanstd(vel_px_s) / np.nanmean(vel_px_s))
                         if len(vel_px_s) and np.nanmean(vel_px_s) not in (0, np.nan) else np.nan),
            mean_jitter=float(np.nanmean(jitter)) if len(jitter) else np.nan,
            pos_std_x=float(np.nanstd(x)),
            pos_std_y=float(np.nanstd(y)),
            bbox_area=float((np.nanmax(x) - np.nanmin(x)) * (np.nanmax(y) - np.nanmin(y))),
        ))
    return pd.DataFrame(rows)


def build_metrics_table(all_sessions: list[SessionData]) -> pd.DataFrame:
    parts = [compute_bodypart_metrics(s) for s in all_sessions]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def session_level_summary(bp_metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per session: mean of each metric across that session's bodyparts."""
    agg_cols = ["mean_likelihood", "interp_rate", "mean_velocity_px_s",
                "std_velocity_px_s", "cv_velocity", "mean_jitter",
                "pos_std_x", "pos_std_y", "bbox_area"]
    grp = bp_metrics.groupby(["source", "paradigm", "is_cube", "session_id"], as_index=False)
    return grp[agg_cols].mean()


# ------------------------------------------------------------------
#  STATISTICS
# ------------------------------------------------------------------

def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_var = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    pooled_sd = np.sqrt(pooled_var)
    return float((np.mean(a) - np.mean(b)) / pooled_sd) if pooled_sd > 0 else np.nan


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    gt = int(np.sum(a[:, None] > b[None, :]))
    lt = int(np.sum(a[:, None] < b[None, :]))
    return (gt - lt) / (len(a) * len(b)) if len(a) and len(b) else np.nan


def dunn_test(groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Post-hoc pairwise test following a significant Kruskal-Wallis, using
    pooled rank-based z-statistics with tie correction (Dunn 1964)."""
    names = list(groups.keys())
    all_vals = np.concatenate([groups[n] for n in names])
    N = len(all_vals)
    ranks = sstats.rankdata(all_vals)
    idx = 0
    grank = {}
    for n in names:
        L = len(groups[n])
        grank[n] = ranks[idx:idx + L]
        idx += L
    _, counts = np.unique(all_vals, return_counts=True)
    tie_term = np.sum(counts ** 3 - counts)
    tie_correction = 1 - tie_term / (N ** 3 - N) if N > 1 else 1.0

    rows = []
    for i, j in itertools.combinations(range(len(names)), 2):
        ni, nj = len(groups[names[i]]), len(groups[names[j]])
        Ri, Rj = grank[names[i]].mean(), grank[names[j]].mean()
        se = np.sqrt(max(tie_correction, 1e-12) * (N * (N + 1) / 12.0) * (1.0 / ni + 1.0 / nj))
        z = (Ri - Rj) / se if se > 0 else 0.0
        p = 2 * (1 - sstats.norm.cdf(abs(z)))
        rows.append(dict(group_a=names[i], group_b=names[j], z=z, dunn_p=p))
    return pd.DataFrame(rows)


def compare_metric_across_sources(df: pd.DataFrame, metric: str, group_col="source") -> dict:
    all_groups = {k: v[metric].dropna().values for k, v in df.groupby(group_col)}
    groups = {k: v for k, v in all_groups.items() if len(v) >= 3}
    dropped = {k: len(v) for k, v in all_groups.items() if len(v) < 3}
    if len(groups) < 2:
        return dict(metric=metric, groups=groups, dropped=dropped, ok=False)

    normality_p = {k: (float(sstats.shapiro(v)[1]) if 3 <= len(v) <= 5000 else np.nan)
                    for k, v in groups.items()}
    lev_stat, lev_p = sstats.levene(*groups.values())
    cv = {k: (float(np.std(v) / np.mean(v)) if np.mean(v) != 0 else np.nan) for k, v in groups.items()}

    if len(groups) > 2:
        omni_stat, omni_p = sstats.kruskal(*groups.values())
        omni_name = "kruskal_wallis"
        posthoc = dunn_test(groups)
        posthoc["dunn_p_fdr"] = fdr_bh(posthoc["dunn_p"].values)
    else:
        (na, a), (nb, b) = list(groups.items())
        omni_stat, omni_p = sstats.mannwhitneyu(a, b, alternative="two-sided")
        omni_name = "mann_whitney_u"
        posthoc = pd.DataFrame()

    pairwise_rows = []
    names = list(groups.keys())
    for i, j in itertools.combinations(range(len(names)), 2):
        a, b = groups[names[i]], groups[names[j]]
        u_stat, u_p = sstats.mannwhitneyu(a, b, alternative="two-sided")
        ks_stat, ks_p = sstats.ks_2samp(a, b)
        t_stat, t_p = sstats.ttest_ind(a, b, equal_var=False)
        pairwise_rows.append(dict(
            group_a=names[i], group_b=names[j], n_a=len(a), n_b=len(b),
            mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
            mannwhitney_p=u_p, ks_p=ks_p, welch_t_p=t_p,
            cliffs_delta=cliffs_delta(a, b), cohens_d=cohens_d(a, b),
        ))
    pairwise = pd.DataFrame(pairwise_rows)
    if len(pairwise):
        pairwise["mannwhitney_p_fdr"] = fdr_bh(pairwise["mannwhitney_p"].values)
        pairwise["ks_p_fdr"] = fdr_bh(pairwise["ks_p"].values)

    return dict(metric=metric, ok=True, groups=groups, dropped=dropped, normality_p=normality_p,
                levene_stat=float(lev_stat), levene_p=float(lev_p), cv=cv,
                omnibus_name=omni_name, omnibus_stat=float(omni_stat), omnibus_p=float(omni_p),
                pairwise=pairwise, posthoc_dunn=posthoc)


# ------------------------------------------------------------------
#  PLOTTING (dual light/dark theme)
# ------------------------------------------------------------------

THEMES = {
    "light": dict(bg="#FFFFFF", panel="#F4F4F4", text="#1A1A1A", tick="#4A4A4A", grid="#DADADA"),
    "dark":  dict(bg="#1E1E1E", panel="#262626", text="#EAEAEA", tick="#BFBFBF", grid="#3A3A3A"),
}


def _styled_fig(theme: str, figsize=(10, 6)):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=figsize, facecolor=t["bg"])
    ax.set_facecolor(t["panel"])
    ax.tick_params(colors=t["tick"])
    for spine in ax.spines.values():
        spine.set_color(t["tick"])
    ax.title.set_color(t["text"])
    ax.xaxis.label.set_color(t["text"])
    ax.yaxis.label.set_color(t["text"])
    ax.grid(True, axis="y", color=t["grid"], alpha=0.5, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax, t


def _save_dual(fig_dark, fig_light, stem: str):
    for theme, fig in (("light", fig_light), ("dark", fig_dark)):
        out_dir = OUTPUT_ROOT / "plots" / theme
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}_{theme}.png", dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)


def _sig_stars(p: float) -> str:
    if np.isnan(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def plot_distribution(df: pd.DataFrame, metric: str, ylabel: str, title: str,
                       stem: str, colors: dict, log_y: bool = False):
    sources = sorted(df["source"].unique(), key=lambda s: (not df.loc[df.source == s, "is_cube"].iloc[0], s))
    figs = {}
    for theme in ("light", "dark"):
        fig, ax, t = _styled_fig(theme, figsize=(max(7, 1.3 * len(sources)), 6))
        data = [df.loc[df.source == s, metric].dropna().values for s in sources]
        parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.8)
        for pc, s in zip(parts["bodies"], sources):
            pc.set_facecolor(colors[s])
            pc.set_edgecolor(t["text"])
            pc.set_alpha(0.75)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color(t["text"])
                parts[key].set_linewidth(1.0)
        for i, s in enumerate(sources, start=1):
            ax.scatter(np.random.normal(i, 0.04, size=len(data[i - 1])), data[i - 1],
                       s=8, color=colors[s], alpha=0.35, edgecolors="none", zorder=3)
        ax.set_xticks(range(1, len(sources) + 1))
        ax.set_xticklabels(sources, rotation=25, ha="right", color=t["text"])
        ax.set_ylabel(ylabel, color=t["text"])
        ax.set_title(title, color=t["text"], fontsize=13, fontweight="bold")
        if log_y:
            ax.set_yscale("log")
        figs[theme] = fig
    _save_dual(figs["dark"], figs["light"], stem)


def plot_bar_with_ci(df: pd.DataFrame, metric: str, ylabel: str, title: str,
                      stem: str, colors: dict):
    sources = sorted(df["source"].unique(), key=lambda s: (not df.loc[df.source == s, "is_cube"].iloc[0], s))
    figs = {}
    for theme in ("light", "dark"):
        fig, ax, t = _styled_fig(theme, figsize=(max(7, 1.3 * len(sources)), 6))
        means, cis = [], []
        for s in sources:
            vals = df.loc[df.source == s, metric].dropna().values
            means.append(np.mean(vals) if len(vals) else np.nan)
            if len(vals) > 1:
                se = np.std(vals, ddof=1) / np.sqrt(len(vals))
                cis.append(1.96 * se)
            else:
                cis.append(0.0)
        bar_colors = [colors[s] for s in sources]
        ax.bar(range(len(sources)), means, yerr=cis, color=bar_colors,
               edgecolor=t["text"], linewidth=0.8, capsize=4, zorder=3)
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels(sources, rotation=25, ha="right", color=t["text"])
        ax.set_ylabel(ylabel, color=t["text"])
        ax.set_title(title, color=t["text"], fontsize=13, fontweight="bold")
        figs[theme] = fig
    _save_dual(figs["dark"], figs["light"], stem)


def plot_pvalue_heatmap(pairwise: pd.DataFrame, sources: list, pcol: str,
                         title: str, stem: str):
    n = len(sources)
    mat = np.full((n, n), np.nan)
    for _, row in pairwise.iterrows():
        i, j = sources.index(row.group_a), sources.index(row.group_b)
        mat[i, j] = row[pcol]
        mat[j, i] = row[pcol]
    figs = {}
    for theme in ("light", "dark"):
        fig, ax, t = _styled_fig(theme, figsize=(max(6, n * 1.1), max(5, n * 1.0)))
        cmap = matplotlib.cm.get_cmap("RdYlGn" if theme == "light" else "RdYlGn")
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=0.1, aspect="equal")
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            color="black" if theme == "light" else "black", fontsize=9)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(sources, rotation=35, ha="right", color=t["text"])
        ax.set_yticklabels(sources, color=t["text"])
        ax.set_title(title, color=t["text"], fontsize=13, fontweight="bold")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=t["tick"])
        plt.setp(plt.getp(cbar.ax, "yticklabels"), color=t["text"])
        cbar.set_label("p-value", color=t["text"])
        figs[theme] = fig
    _save_dual(figs["dark"], figs["light"], stem)


def plot_trajectories(sessions_by_source: dict[str, list[SessionData]], paradigm: str,
                       colors: dict, stem: str, max_per_source: int = 2):
    sources = list(sessions_by_source.keys())
    n = len(sources)
    figs = {}
    for theme in ("light", "dark"):
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), facecolor=THEMES[theme]["bg"])
        if n == 1:
            axes = [axes]
        t = THEMES[theme]
        for ax, s in zip(axes, sources):
            ax.set_facecolor(t["panel"])
            ax.tick_params(colors=t["tick"])
            for spine in ax.spines.values():
                spine.set_color(t["tick"])
            for sess in sessions_by_source[s][:max_per_source]:
                canon_idx = [i for i, bp in enumerate(sess.bodyparts) if canonical_landmark(bp) == "body_center"]
                i = canon_idx[0] if canon_idx else 0
                x, y = sess.xy[:, 2 * i], sess.xy[:, 2 * i + 1]
                ax.plot(x, y, color=colors[s], alpha=0.7, linewidth=0.8)
            ax.set_title(s, color=t["text"], fontsize=10)
            ax.set_aspect("equal", adjustable="datalim")
            ax.invert_yaxis()
        fig.suptitle(f"Example trajectories \u2014 {paradigm}", color=t["text"],
                      fontsize=13, fontweight="bold")
        figs[theme] = fig
    _save_dual(figs["dark"], figs["light"], stem)


def plot_dashboard(session_df: pd.DataFrame, colors: dict, paradigm: str, stem: str):
    sources = sorted(session_df["source"].unique(), key=lambda s: (not session_df.loc[session_df.source == s, "is_cube"].iloc[0], s))
    metrics = [
        ("mean_likelihood", "mean likelihood"),
        ("interp_rate", "interpolated frame fraction"),
        ("mean_velocity_px_s", "mean velocity (px/s)"),
        ("mean_jitter", "mean jitter (|\u0394v|, px/frame)"),
        ("bbox_area", "explored bbox area (px\u00b2)"),
        ("cv_velocity", "CV of velocity"),
    ]
    figs = {}
    for theme in ("light", "dark"):
        t = THEMES[theme]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=t["bg"])
        for ax, (metric, label) in zip(axes.flat, metrics):
            ax.set_facecolor(t["panel"])
            ax.tick_params(colors=t["tick"], labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(t["tick"])
            data = [session_df.loc[session_df.source == s, metric].dropna().values for s in sources]
            bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                             medianprops=dict(color=t["text"]),
                             whiskerprops=dict(color=t["tick"]),
                             capprops=dict(color=t["tick"]),
                             flierprops=dict(markeredgecolor=t["tick"], markersize=3))
            for patch, s in zip(bp["boxes"], sources):
                patch.set_facecolor(colors[s])
                patch.set_alpha(0.8)
            ax.set_xticks(range(1, len(sources) + 1))
            ax.set_xticklabels(sources, rotation=30, ha="right", fontsize=7, color=t["text"])
            ax.set_title(label, color=t["text"], fontsize=10)
            ax.grid(True, axis="y", color=t["grid"], alpha=0.4, linewidth=0.5)
            ax.set_axisbelow(True)
        fig.suptitle(f"CUBE DLC-output QC dashboard \u2014 {paradigm}", color=t["text"],
                      fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        figs[theme] = fig
    _save_dual(figs["dark"], figs["light"], stem)


# ------------------------------------------------------------------
#  REPORT
# ------------------------------------------------------------------

def write_report(all_stats: dict, out_path: Path):
    lines = ["# CUBE DLC-output comparison \u2014 summary report", ""]
    lines.append("Stage 1 (DLC tracking output only). CUBE-blue series indicate CUBE's own "
                  "DLC run; all other series are published/reference tracking data.")
    lines.append("")
    for paradigm, metric_results in all_stats.items():
        lines.append(f"## {paradigm}")
        any_ok = any(res.get("ok") for res in metric_results.values())
        if not any_ok:
            dropped_note = {}
            for res in metric_results.values():
                dropped_note.update(res.get("dropped", {}))
            if dropped_note:
                skipped = ", ".join(f"{k} (n={v})" for k, v in dropped_note.items())
                lines.append(f"- Cross-source statistics skipped: too few sessions "
                              f"(n<3 required per group) in: {skipped}. "
                              f"Distribution plots for this paradigm were still generated.")
            else:
                lines.append("- Only one source available for this paradigm — "
                              "distribution plots generated, no cross-source statistics yet.")
            lines.append("")
            continue
        for metric, res in metric_results.items():
            if not res.get("ok"):
                dropped = res.get("dropped", {})
                if dropped:
                    skipped = ", ".join(f"{k} (n={v})" for k, v in dropped.items())
                    lines.append(f"### {metric}")
                    lines.append(f"- Skipped: too few sessions in {skipped} (n<3 required).")
                    lines.append("")
                continue
            lines.append(f"### {metric}")
            lines.append(f"- Omnibus ({res['omnibus_name']}): stat={res['omnibus_stat']:.3f}, "
                          f"p={res['omnibus_p']:.4g}")
            lines.append(f"- Levene's test (variance homogeneity): stat={res['levene_stat']:.3f}, "
                          f"p={res['levene_p']:.4g} "
                          f"({'variances differ' if res['levene_p'] < 0.05 else 'variances similar'})")
            cv_str = ", ".join(f"{k}={v:.3f}" for k, v in res["cv"].items())
            lines.append(f"- Coefficient of variation per source: {cv_str}")
            pw = res["pairwise"]
            if len(pw):
                sig = pw[pw["mannwhitney_p_fdr"] < 0.05].sort_values("cliffs_delta", key=abs, ascending=False)
                if len(sig):
                    lines.append("- Significant pairwise differences (FDR<0.05), largest effect first:")
                    for _, r in sig.head(5).iterrows():
                        lines.append(f"  - {r.group_a} vs {r.group_b}: "
                                      f"Mann-Whitney p_fdr={r.mannwhitney_p_fdr:.4g}, "
                                      f"Cliff's delta={r.cliffs_delta:.3f}, "
                                      f"Cohen's d={r.cohens_d:.3f}")
                else:
                    lines.append("- No pairwise differences survived FDR correction.")
            lines.append("")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------
#  3D STAGE (AcinoSet multi-camera triangulation validation)
#
#  Tests cube_core.triangulate_cameras() against AcinoSet's published FTE
#  ground-truth 3D reconstructions. AcinoSet's own 2D DLC keypoints were
#  deliberately not kept (see F:\CUBE_test_data\3d_multiview\acinoset\...) --
#  this stage expects CUBE's own 2D DLC output on the same videos, so it
#  isolates triangulation correctness from 2D-model accuracy only once you
#  wire up CUBE_3D_RUN_DIRS below; until then it just prepares the
#  calibration.toml files so they're ready to drop into CUBE's 3D DLC tab.
# ------------------------------------------------------------------

ACINOSET_ROOT = DATA_ROOT / "3d_multiview/acinoset/2017_08_29"

ACINOSET_SESSIONS = [
    dict(rig="bottom_rig", session="phantom_flick2"),
    dict(rig="bottom_rig", session="zorro_flick2"),
    dict(rig="top_rig", session="jules_run1_1"),
    dict(rig="top_rig", session="jules_run1_2"),
    dict(rig="top_rig", session="phantom_flick1_1"),
    dict(rig="top_rig", session="phantom_run1_1"),
    dict(rig="top_rig", session="zorro_flick1_1"),
    dict(rig="top_rig", session="zorro_flick1_2"),
]

# Point each session at CUBE's own 3D-triangulated H5 output once you've run
# the 3D DLC step (per 3D_DLC_Integration_Plan_v6.md) on that session's
# videos/ + the converted calibration.toml. None = not yet available.
CUBE_3D_RUN_DIRS: dict[str, "str | None"] = {
    s["session"]: None for s in ACINOSET_SESSIONS
}


def acinoset_calib_to_toml(scene_json_path: Path, out_toml_path: Path, fisheye: bool = True) -> Path:
    """Convert AcinoSet's camera_scene_sba.json (OpenCV-style k/d/r/t per
    camera) into an aniposelib calibration.toml that cube_core.triangulate_
    cameras() can load directly via CameraGroup.load()."""
    import json
    import cv2
    from aniposelib.cameras import Camera, FisheyeCamera, CameraGroup

    with open(scene_json_path) as f:
        scene = json.load(f)

    size = scene["camera_resolution"]
    cams = []
    CamCls = FisheyeCamera if fisheye else Camera
    for i, c in enumerate(scene["cameras"]):
        k = np.array(c["k"], dtype=float)
        d = np.array(c["d"], dtype=float).flatten()
        r_mat = np.array(c["r"], dtype=float)
        t = np.array(c["t"], dtype=float).flatten()
        rvec, _ = cv2.Rodrigues(r_mat)
        cam = CamCls(name=f"cam{i + 1}", size=size)
        cam.set_camera_matrix(k)
        cam.set_distortions(d)
        cam.set_rotation(rvec.flatten())
        cam.set_translation(t)
        cams.append(cam)

    cgroup = CameraGroup(cams)
    out_toml_path.parent.mkdir(parents=True, exist_ok=True)
    cgroup.dump(str(out_toml_path))
    return out_toml_path


def load_fte_ground_truth(fte_pickle_path: Path) -> dict:
    """AcinoSet's fte.pickle -> {'positions': (N_frames, N_bp, 3), 'start_frame': int}."""
    import pickle
    with open(fte_pickle_path, "rb") as f:
        d = pickle.load(f)
    return dict(positions=np.asarray(d["positions"], dtype=float),
                start_frame=int(d.get("start_frame", 0)))


def load_cube_3d_h5(h5_path: Path) -> np.ndarray:
    """Load a CUBE 3D-triangulated H5 (coords = x,y,z,likelihood) -> (N_frames, N_bp, 3)."""
    df = pd.read_hdf(h5_path)
    scorer = df.columns.get_level_values(0)[0]
    sub = df[scorer]
    bps = sub.columns.get_level_values(0).unique().tolist()
    x = sub.xs("x", level=-1, axis=1)[bps].values
    y = sub.xs("y", level=-1, axis=1)[bps].values
    z = sub.xs("z", level=-1, axis=1)[bps].values
    return np.stack([x, y, z], axis=-1)


def procrustes_align(a: np.ndarray, b: np.ndarray):
    """Best-fit rigid+scale alignment of a onto b (Umeyama). a, b: (N, 3).
    Returns a_aligned, (R, s, t)."""
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    a0, b0 = a - mu_a, b - mu_b
    U, S, Vt = np.linalg.svd(a0.T @ b0)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    s = S.sum() / max((a0 ** 2).sum(), 1e-12)
    a_aligned = s * (a0 @ R) + mu_b
    return a_aligned, (R, s, mu_b - s * (mu_a @ R))


def compare_3d_trajectory(cube_xyz: np.ndarray, gt_xyz: np.ndarray) -> dict:
    """cube_xyz, gt_xyz: (N_frames, N_bp, 3), same frame range/bodypart order.
    Returns raw + Procrustes-aligned per-frame-per-bodypart Euclidean error."""
    n_f = min(cube_xyz.shape[0], gt_xyz.shape[0])
    n_bp = min(cube_xyz.shape[1], gt_xyz.shape[1])
    c, g = cube_xyz[:n_f, :n_bp], gt_xyz[:n_f, :n_bp]

    raw_err = np.linalg.norm(c - g, axis=-1)  # (n_f, n_bp)

    c_flat, g_flat = c.reshape(-1, 3), g.reshape(-1, 3)
    valid = ~(np.isnan(c_flat).any(axis=1) | np.isnan(g_flat).any(axis=1))
    if valid.sum() >= 4:
        c_aligned_flat, _ = procrustes_align(c_flat[valid], g_flat[valid])
        aligned_err_flat = np.linalg.norm(c_aligned_flat - g_flat[valid], axis=-1)
        aligned_err = np.full(valid.shape, np.nan)
        aligned_err[valid] = aligned_err_flat
        aligned_err = aligned_err.reshape(n_f, n_bp)
    else:
        aligned_err = np.full((n_f, n_bp), np.nan)

    return dict(raw_error=raw_err, aligned_error=aligned_err, n_frames=n_f, n_bodyparts=n_bp)


def plot_3d_error(err_dict: dict, session: str, stem: str):
    figs = {}
    for theme in ("light", "dark"):
        t = THEMES[theme]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=t["bg"])
        for ax, (key, label) in zip(axes, [("raw_error", "raw"), ("aligned_error", "Procrustes-aligned")]):
            ax.set_facecolor(t["panel"])
            ax.tick_params(colors=t["tick"])
            for spine in ax.spines.values():
                spine.set_color(t["tick"])
            err = err_dict[key]
            per_bp = [err[:, i][~np.isnan(err[:, i])] for i in range(err.shape[1])]
            bp = ax.boxplot(per_bp, patch_artist=True, widths=0.6,
                             medianprops=dict(color=t["text"]),
                             whiskerprops=dict(color=t["tick"]),
                             capprops=dict(color=t["tick"]),
                             flierprops=dict(markeredgecolor=t["tick"], markersize=3))
            for patch in bp["boxes"]:
                patch.set_facecolor(CUBE_BLUE)
                patch.set_alpha(0.8)
            ax.set_xlabel("bodypart index", color=t["text"])
            ax.set_ylabel("3D error (world units)", color=t["text"])
            ax.set_title(label, color=t["text"], fontsize=11)
            ax.grid(True, axis="y", color=t["grid"], alpha=0.4, linewidth=0.5)
            ax.set_axisbelow(True)
        fig.suptitle(f"CUBE 3D triangulation vs AcinoSet ground truth \u2014 {session}",
                      color=t["text"], fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        figs[theme] = fig
    _save_dual(figs["dark"], figs["light"], stem)


def run_3d_stage():
    print("\n--- 3D stage (AcinoSet triangulation validation) ---")
    if not ACINOSET_ROOT.is_dir():
        print(f"  AcinoSet data not found at {ACINOSET_ROOT} — skipping 3D stage.")
        return

    out_dir = OUTPUT_ROOT / "3d_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rigs_converted = set()
    for s in ACINOSET_SESSIONS:
        rig, session = s["rig"], s["session"]
        rig_dir = ACINOSET_ROOT / rig
        session_dir = rig_dir / "sessions" / session

        if rig not in rigs_converted:
            scene_json = rig_dir / "calibration" / "camera_scene_sba.json"
            toml_out = rig_dir / "calibration" / "calibration.toml"
            if scene_json.is_file():
                try:
                    acinoset_calib_to_toml(scene_json, toml_out)
                    print(f"  [{rig}] calibration.toml written -> {toml_out}")
                except ImportError as e:
                    print(f"  [{rig}] calibration conversion skipped: {e}")
                except Exception as e:
                    print(f"  [{rig}] calibration conversion FAILED: {e}")
            rigs_converted.add(rig)

        gt_pickle = session_dir / "ground_truth_3d" / "fte.pickle"
        if not gt_pickle.is_file():
            print(f"  [{session}] no fte.pickle found — skipping")
            continue
        gt = load_fte_ground_truth(gt_pickle)

        cube_dir = CUBE_3D_RUN_DIRS.get(session)
        if cube_dir is None:
            print(f"  [{session}] ground truth ready ({gt['positions'].shape[0]} frames, "
                  f"{gt['positions'].shape[1]} bodyparts) — awaiting CUBE 3D DLC run "
                  f"(set CUBE_3D_RUN_DIRS['{session}'])")
            continue

        cube_h5 = Path(cube_dir)
        if not cube_h5.is_file():
            print(f"  [{session}] CUBE_3D_RUN_DIRS path not found: {cube_h5}")
            continue
        cube_xyz = load_cube_3d_h5(cube_h5)
        gt_xyz = gt["positions"]
        err = compare_3d_trajectory(cube_xyz, gt_xyz)
        raw_valid = err["raw_error"][~np.isnan(err["raw_error"])]
        aligned_valid = err["aligned_error"][~np.isnan(err["aligned_error"])]
        print(f"  [{session}] median raw error={np.median(raw_valid):.3f}, "
              f"median Procrustes-aligned error={np.median(aligned_valid):.3f} "
              f"(n={err['n_frames']} frames x {err['n_bodyparts']} bodyparts)")

        summary_rows = [dict(bodypart_idx=i,
                              median_raw_error=float(np.nanmedian(err["raw_error"][:, i])),
                              median_aligned_error=float(np.nanmedian(err["aligned_error"][:, i])))
                         for i in range(err["n_bodyparts"])]
        pd.DataFrame(summary_rows).to_csv(out_dir / f"{session}__3d_error_by_bodypart.csv", index=False)
        plot_3d_error(err, session, f"{session}__3d_triangulation_error")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "metrics").mkdir(exist_ok=True)

    colors = build_color_map(SOURCE_MANIFEST)
    print("Color assignment:")
    for s in SOURCE_MANIFEST:
        print(f"  {s['name']:24s} ({s['paradigm']:20s}) -> {colors[s['name']]}"
              f"{'  [CUBE]' if s['is_cube'] else ''}")

    all_sessions: list[SessionData] = []
    for cfg in SOURCE_MANIFEST:
        recs = load_source(cfg)
        print(f"[{cfg['name']}] loaded {len(recs)} session(s) from {cfg['path']}")
        all_sessions.extend(recs)

    if not all_sessions:
        print("No sessions loaded — nothing to compare. Check SOURCE_MANIFEST paths.")
        return

    bp_metrics = build_metrics_table(all_sessions)
    bp_metrics.to_csv(OUTPUT_ROOT / "metrics" / "session_bodypart_metrics.csv", index=False)

    session_df = session_level_summary(bp_metrics)
    session_df.to_csv(OUTPUT_ROOT / "metrics" / "session_summary_metrics.csv", index=False)

    metric_specs = [
        ("mean_likelihood", "mean likelihood", "likelihood_distribution", False),
        ("interp_rate", "fraction of frames interpolated", "interpolation_rate", False),
        ("mean_velocity_px_s", "mean velocity (px/s)", "velocity_distribution", True),
        ("mean_jitter", "mean jitter (px/frame)", "jitter_distribution", True),
        ("bbox_area", "explored bbox area (px\u00b2)", "bbox_area_distribution", True),
    ]

    all_stats: dict[str, dict] = {}
    for paradigm in sorted(session_df["paradigm"].unique()):
        pdf = session_df[session_df.paradigm == paradigm]
        pdf_bp = bp_metrics[bp_metrics.paradigm == paradigm]
        sources_here = sorted(pdf["source"].unique())
        if len(sources_here) < 2:
            print(f"[{paradigm}] only one source present ({sources_here}) — "
                  f"distribution plots only, no cross-source stats yet.")

        all_stats[paradigm] = {}
        for metric, ylabel, stem, logy in metric_specs:
            title = f"{ylabel} \u2014 {paradigm}"
            plot_distribution(pdf, metric, ylabel, title, f"{paradigm}__{stem}", colors, log_y=logy)
            if len(sources_here) >= 2:
                res = compare_metric_across_sources(pdf, metric)
                all_stats[paradigm][metric] = res
                if res.get("ok") and len(res["pairwise"]):
                    res["pairwise"].to_csv(
                        OUTPUT_ROOT / "metrics" / f"{paradigm}__{metric}__pairwise_tests.csv", index=False)
                    plot_pvalue_heatmap(res["pairwise"], sorted(res["groups"].keys()),
                                        "mannwhitney_p_fdr",
                                        f"Pairwise Mann-Whitney p (FDR) \u2014 {ylabel} \u2014 {paradigm}",
                                        f"{paradigm}__{stem}__pvalue_heatmap")
                    plot_pvalue_heatmap(res["pairwise"], sorted(res["groups"].keys()),
                                        "ks_p",
                                        f"Pairwise KS-test p \u2014 {ylabel} \u2014 {paradigm}",
                                        f"{paradigm}__{stem}__ks_heatmap")

        if len(sources_here) >= 2:
            plot_dashboard(pdf, colors, paradigm, f"{paradigm}__dashboard")

        sessions_by_source = {}
        for s in sources_here:
            sessions_by_source[s] = [sess for sess in all_sessions
                                      if sess.source == s and sess.paradigm == paradigm]
        plot_trajectories(sessions_by_source, paradigm, colors, f"{paradigm}__example_trajectories")

    write_report(all_stats, OUTPUT_ROOT / "report_summary.md")
    print(f"\nDone. Outputs written to: {OUTPUT_ROOT}")

    run_3d_stage()


if __name__ == "__main__":
    main()
