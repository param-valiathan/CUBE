"""Guided labelling: behaviour groups, the analyser mapping, and the before/after report.

What the user's labels change in a CUBE run, made visible and carried to the next stage:

  * ``behaviour_groups`` / ``cluster_names``  - which final clusters the labels prove to be one behaviour
    (a behaviour made of several clusters is ONE group), and a readable name for every such cluster.
  * ``write_mapping_tsv`` / ``write_behaviour_groups_json`` - the improved clusters + behaviour groups in the
    format the Behaviour Analyser already reads (cluster_behaviour_mapping.tsv: several clusters sharing a
    behaviour_group are one behaviour), so the next step uses the improved result, not the blind one.
  * ``snapshot_before`` - at the start of a labelled run, keeps the PREVIOUS (blind) result's per-frame cluster
    labels, so a before/after comparison survives CUBE re-using the same results folder.
  * ``produce_plots`` - the "Produce plots" button: before/after UMAPs, Coverage/Leakage/Ratio before vs after
    (full per-behaviour tables), a labelled hierarchy, all written into ``plots/guided/`` of the run.

Pure library: no Tk. Everything is best-effort and never raises into the engine.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import cube_guided as cg

SNAP_DIR = "guided/before_run"
GUIDED_PLOTS = "guided"
_GUIDED_COL = "guided_source"


# ------------------------------------------------------------------ groups & names
def behaviour_groups(ctx, labels) -> dict:
    """{behaviour (user spelling): [cluster ids]} for clusters the labels prove pure (>=1 behaviour each)."""
    labels = np.asarray(labels)
    out: dict = {}
    for cid, b in ctx.dominant(labels).items():
        if ctx.is_user_pure(labels, cid):
            out.setdefault(ctx._display.get(b, b), []).append(int(cid))
    return {k: sorted(v) for k, v in sorted(out.items())}


def cluster_names(groups: dict) -> dict:
    """{cluster id: 'C12 · Rearing (2/3)'}; the (k/n) part only when a behaviour has several clusters."""
    names = {}
    for beh, cids in groups.items():
        for i, c in enumerate(cids, 1):
            names[int(c)] = f"C{c} · {beh}" + (f" ({i}/{len(cids)})" if len(cids) > 1 else "")
    return names


def write_mapping_tsv(path, labels, groups: dict) -> str:
    """cluster_behaviour_mapping.tsv in Video Explorer's format (+ guided columns). A file already at ``path``
    that was NOT written by this module (e.g. the user's own Video Explorer export) is kept as ``.bak``."""
    path = Path(path)
    labels = np.asarray(labels)
    of = {c: (b, i + 1, len(cl)) for b, cl in groups.items() for i, c in enumerate(cl)}
    rows = []
    for c in sorted(int(x) for x in np.unique(labels[labels >= 0])):
        b, part, n_parts = of.get(c, ("", 0, 0))
        rows.append({"bsoid_group_id": c, "behaviour_group": b, "status": "assigned" if b else "pending",
                     "example_count": 0, "gif_path": "", "mp4_paths": "",
                     "n_bins": int((labels == c).sum()), "clusters_in_behaviour": n_parts, "part": part,
                     _GUIDED_COL: "user_labels"})
    if path.exists():
        try:
            if _GUIDED_COL not in pd.read_csv(path, sep="\t", nrows=1).columns:
                shutil.copy2(path, str(path) + ".bak")
        except Exception:
            shutil.copy2(path, str(path) + ".bak")
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return str(path)


def write_behaviour_groups_json(path, groups: dict, extra: dict | None = None) -> str:
    payload = {"behaviour_groups": {b: {"clusters": cl, "n_clusters": len(cl)} for b, cl in groups.items()},
               "multi_cluster_behaviours": sorted(b for b, cl in groups.items() if len(cl) > 1),
               "written": time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {})}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------- before-snapshot
def snapshot_before(output_dir, log=None) -> bool:
    """Keep the previous blind result's per-frame cluster labels (small CSVs) under guided/before_run/ so a
    before/after comparison survives CUBE reusing this folder. Only when the existing result is a blind run and
    no snapshot exists yet. Never raises. Returns True if a snapshot was taken."""
    log = log or (lambda m: None)
    try:
        out = Path(output_dir)
        dest = out / SNAP_DIR
        if dest.exists():
            return False
        bout, sbr, summ = out / "bout_lengths", out / "model" / "session_bin_ranges.json", out / "bsoid_run_summary.json"
        files = sorted(bout.glob("*_frame_labels_hmm.csv")) or sorted(bout.glob("*_frame_labels.csv"))
        if not files or not sbr.is_file():
            return False
        if summ.is_file() and "guided_labels" in summ.read_text(encoding="utf-8", errors="ignore"):
            return False                                  # the existing result is already a labelled run
        (dest / "bout_lengths").mkdir(parents=True, exist_ok=True)
        (dest / "model").mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / "bout_lengths" / f.name)
        shutil.copy2(sbr, dest / "model" / "session_bin_ranges.json")
        (dest / "snapshot.json").write_text(json.dumps(
            {"taken": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_sessions": len(files),
             "note": "per-frame cluster labels of the blind run that preceded the first labelled run"}, indent=2))
        log(f"  [GUIDED] kept the previous (blind) cluster labels for the before/after report: {dest}")
        return True
    except Exception as e:                                # noqa: BLE001
        log(f"  [GUIDED] WARN: could not snapshot the previous result ({e}).")
        return False


# ----------------------------------------------------------------- loading runs
class RunLabels:
    """Per-frame / per-bin cluster labels of one run (a CUBE 3 folder or a before_run snapshot)."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        sbr = self.dir / "model" / "session_bin_ranges.json"
        if not sbr.is_file():
            raise FileNotFoundError(f"{sbr} not found")
        raw = json.loads(sbr.read_text())
        self.ranges = cg.clean_bin_ranges(raw)
        self.n_bins = int(raw.get("_total_bins", max((e for _, e in self.ranges.values()), default=0)))
        self.frame_labels: dict = {}
        self.fps: dict = {}
        bout = self.dir / "bout_lengths"
        from cube_guided_session import session_stems
        for key in self.ranges:
            for suffix, stem in ((sf, st) for sf in ("_frame_labels_hmm.csv", "_frame_labels.csv")
                                 for st in session_stems(key)):
                p = bout / f"{stem}{suffix}"
                if p.is_file():
                    df = pd.read_csv(p)
                    col = "label" if "label" in df else df.columns[2]
                    self.frame_labels[key] = df[col].to_numpy(int)
                    t = df["time_s"].to_numpy(float) if "time_s" in df else None
                    self.fps[key] = (1.0 / float(t[1] - t[0])) if (t is not None and len(t) > 2 and t[1] > t[0]) else 30.0
                    break

    def per_bin(self, ranges=None, fps_map=None) -> np.ndarray:
        """Cluster per bin in the bin space of ``ranges`` (default: this run's own); -1 when unknown."""
        ranges = ranges or self.ranges
        n = max((e for _, e in ranges.values()), default=0)
        out = np.full(n, -1, dtype=int)
        for key, (s0, s1) in ranges.items():
            fl = self.frame_labels.get(key)
            if fl is None or not len(fl):
                continue
            fps = float((fps_map or {}).get(key) or self.fps.get(key, 30.0))
            win = max(1, int(round(fps / 10.0)))
            f = (np.arange(s1 - s0) * win).astype(int)
            ok = f < len(fl)
            out[s0:s1][ok] = fl[f[ok]]
        return out


def pools_by_purity(cluster_per_bin, truth, min_bins=5, min_purity=0.7) -> dict:
    """{behaviour: [cluster ids]} - purity-gated, each cluster used by one behaviour (the repo convention)."""
    import cube_guided_eval as ge
    keep = np.array([t is not None and str(t) != "" for t in truth])
    if not keep.any():
        return {}
    tab = ge.behaviour_table(np.asarray(cluster_per_bin)[keep], np.asarray(truth, dtype=object)[keep],
                             min_labeled=min_bins, purity_floor=min_purity)
    return {r.Behaviour: [int(x) for x in str(r.Pool).split(",") if x != ""] for r in tab.itertuples()}


def metric_tables(before_bins, after_bins, truth, min_bins=5, min_purity=0.7):
    """Full per-behaviour Coverage / Leakage / Ratio, before and after, on the LABELLED bins only
    (unlabelled bins have no truth and would count as leakage). Returns (before_df, after_df, merged_df)."""
    import cube_guided_eval as ge
    t = np.asarray(truth, dtype=object)
    keep = np.array([x is not None and str(x) != "" for x in t])
    tb = ge.behaviour_table(np.asarray(before_bins)[keep], t[keep], min_labeled=min_bins, purity_floor=min_purity)
    ta = ge.behaviour_table(np.asarray(after_bins)[keep], t[keep], min_labeled=min_bins, purity_floor=min_purity)
    for df in (tb, ta):
        # the repo's purity-gated table reports Ratio 0 when Leakage is exactly 0; here a pool with coverage and NO
        # leakage is the best possible outcome, so it is shown as inf (not as a misleading 0)
        free = (df["Leakage"] == 0) & (df["Coverage"] > 0)
        df.loc[free, "Ratio"] = np.inf
    m = tb.merge(ta, on="Behaviour", suffixes=("_before", "_after"))
    m["Coverage_change"] = m["Coverage_after"] - m["Coverage_before"]
    m["Leakage_change"] = m["Leakage_after"] - m["Leakage_before"]
    m["Ratio_change"] = m["Ratio_after"] - m["Ratio_before"]
    m["n_clusters_before"] = [len([p for p in str(x).split(",") if p]) for x in m["Pool_before"]]
    m["n_clusters_after"] = [len([p for p in str(x).split(",") if p]) for x in m["Pool_after"]]
    return tb, ta, m


# ------------------------------------------------------------------------ plots
def _palette_for(behaviours: list, ls) -> dict:
    cols = {}
    for b in getattr(ls, "behaviours", []) or []:
        if isinstance(b, dict) and b.get("name"):
            cols[str(b["name"])] = b.get("color") or None
    fallback = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
                "#008080", "#9a6324", "#800000", "#808000"]
    out = {}
    for i, b in enumerate(behaviours):
        out[b] = cols.get(b) or fallback[i % len(fallback)]
    return out


def _view_limits(emb, i, j):
    lo = np.nanpercentile(emb[:, [i, j]], 1, axis=0)
    hi = np.nanpercentile(emb[:, [i, j]], 99, axis=0)
    pad = 0.08 * (hi - lo)
    return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])


def plot_before_after_umap(emb, before_lab, after_lab, beh_after, truth, ls, out_path, theme="light",
                           cap=500, seed=0):
    """Rows: before clusters / after clusters / after behaviours / your labels - ONE embedding (the after
    run's), so a point never moves between rows, only its colour changes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cube_core as cc
    cc._apply_plot_theme(theme)
    rng = np.random.default_rng(seed)
    emb = np.asarray(emb, dtype=float)
    dims = [(0, 1), (0, 2), (1, 2)] if emb.shape[1] >= 3 else [(0, 1)]
    behs = sorted({str(b) for b in beh_after if b} | {str(t) for t in truth if t and str(t) not in ("Background", "__neither__")})
    bcol = _palette_for(behs, ls)

    def sub(mask):
        idx = np.flatnonzero(mask)
        return idx if idx.size <= cap else rng.choice(idx, cap, replace=False)

    rows = [("Before: CUBE clusters", before_lab, "cluster"), ("After: CUBE clusters", after_lab, "cluster"),
            ("After: behaviours (clusters of one behaviour share a colour)", beh_after, "beh"),
            ("Your labels (ground truth)", truth, "truth")]
    fig, axes = plt.subplots(len(rows), len(dims), figsize=(4.6 * len(dims) + 3.2, 3.9 * len(rows)),
                             facecolor=cc._BG, squeeze=False)
    for r, (title, lab, kind) in enumerate(rows):
        lab = np.asarray(lab, dtype=object)
        if kind == "cluster":
            num = np.array([-1 if v is None else int(v) for v in lab])
            keys = sorted({int(v) for v in num if v >= 0})
            colour = {k: cc._cmap(k) for k in keys}
            members = {k: np.flatnonzero(num == k) for k in keys}
        else:
            keys = [k for k in behs if np.any(lab == k)]
            colour = {k: bcol[k] for k in keys}
            members = {k: np.flatnonzero(lab == k) for k in keys}
        for c, (i, j) in enumerate(dims):
            ax = axes[r, c]
            ax.set_facecolor(cc._BG)
            ax.scatter(emb[:, i], emb[:, j], s=1, color="#9e9e9e", alpha=0.10, linewidths=0, rasterized=True)
            for k in keys:
                idx = members[k]
                if idx.size > cap:
                    idx = rng.choice(idx, cap, replace=False)
                big = kind == "truth"
                ax.scatter(emb[idx, i], emb[idx, j], s=14 if big else 3, color=colour[k],
                           alpha=0.9 if big else 0.55, linewidths=0, rasterized=True)
            xl, yl = _view_limits(emb, i, j)
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(cc._TICK_COL)
            ax.set_xlabel(f"UMAP {i + 1}", color=cc._TICK_COL, fontsize=8)
            ax.set_ylabel(f"UMAP {j + 1}", color=cc._TICK_COL, fontsize=8)
            if c == 0:
                ax.set_title(title, color=cc._TEXT_COL, fontsize=10, loc="left", fontweight="bold")
        shown = keys if kind != "cluster" else sorted(keys, key=lambda k: -len(members[k]))[:18]
        handles = [plt.Line2D([], [], marker="o", ls="", color=colour[k], markersize=6,
                              label=(f"C{k}" if kind == "cluster" else str(k))) for k in shown]
        if handles:
            axes[r, -1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7,
                               frameon=False, labelcolor=cc._TEXT_COL, ncol=1 if len(handles) <= 12 else 2)
    fig.suptitle("Before vs after guided labelling - one shared UMAP (the after run's); only the colours change",
                 color=cc._TEXT_COL, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cc._savefig(fig, Path(out_path))


def plot_behaviour_panels(emb, before_bins, after_bins, truth, before_pools, after_pools, merged, ls, out_path,
                          theme="light", max_behaviours=8):
    """One column per behaviour: top = the clusters that made up that behaviour BEFORE, bottom = AFTER,
    your labelled bins of that behaviour outlined. Titles carry Coverage / Leakage / Ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cube_core as cc
    cc._apply_plot_theme(theme)
    emb = np.asarray(emb, dtype=float)
    behs = [b for b in merged.sort_values("n_bins_after", ascending=False)["Behaviour"].tolist()][:max_behaviours]
    if not behs:
        return
    col = _palette_for(behs, ls)
    fig, axes = plt.subplots(2, len(behs), figsize=(3.6 * len(behs), 7.4), facecolor=cc._BG, squeeze=False)
    xl, yl = _view_limits(emb, 0, 1)
    t = np.asarray(truth, dtype=object)
    bb, ab = np.asarray(before_bins), np.asarray(after_bins)
    for c, b in enumerate(behs):
        row = merged[merged["Behaviour"] == b].iloc[0]
        for r, (bins, pools, tag) in enumerate(((bb, before_pools, "before"), (ab, after_pools, "after"))):
            ax = axes[r, c]
            ax.set_facecolor(cc._BG)
            ax.scatter(emb[:, 0], emb[:, 1], s=1, color="#9e9e9e", alpha=0.10, linewidths=0, rasterized=True)
            pool = pools.get(b, [])
            inp = np.isin(bins, pool) if pool else np.zeros(len(bins), bool)
            ax.scatter(emb[inp, 0], emb[inp, 1], s=6, color=col[b], alpha=0.75, linewidths=0, rasterized=True)
            lab = t == b
            ax.scatter(emb[lab, 0], emb[lab, 1], s=18, facecolors="none", edgecolors=cc._TEXT_COL,
                       linewidths=0.5, alpha=0.7, rasterized=True)
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
            ax.set_xticks([])
            ax.set_yticks([])
            cov, leak, rat = (row[f"Coverage_{tag}"], row[f"Leakage_{tag}"], row[f"Ratio_{tag}"])
            if not pool:
                ttl = f"{b} - {tag}\nno cluster is pure enough for this behaviour"
            else:
                ttl = (f"{b} - {tag}\n{len(pool)} cluster(s)  cov {cov:.0f}%  leak {leak:.0f}%  ratio "
                       f"{'inf' if np.isinf(rat) else f'{rat:.2f}'}")
            ax.set_title(ttl, color=cc._TEXT_COL, fontsize=8)
            for sp in ax.spines.values():
                sp.set_color(cc._TICK_COL)
    fig.suptitle("Each behaviour's cluster pool before vs after (coloured = pool, rings = your labelled bins)",
                 color=cc._TEXT_COL, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cc._savefig(fig, Path(out_path))


def plot_metrics_before_after(merged, out_path, theme="light", note=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cube_core as cc
    cc._apply_plot_theme(theme)
    behs = merged["Behaviour"].tolist()
    x = np.arange(len(behs))
    fig, axes = plt.subplots(1, 3, figsize=(max(11, 1.1 * len(behs) * 3), 4.8), facecolor=cc._BG)
    for ax, (name, unit) in zip(axes, (("Coverage", "%"), ("Leakage", "%"), ("Ratio", ""))):
        b = merged[f"{name}_before"].fillna(0).to_numpy(float)
        a = merged[f"{name}_after"].fillna(0).to_numpy(float)
        if name == "Ratio":                                   # inf = coverage with no leakage at all: draw capped
            fin = np.concatenate([b[np.isfinite(b)], a[np.isfinite(a)], [1.0]])
            cap = 1.25 * fin.max()
            infa, infb = ~np.isfinite(a), ~np.isfinite(b)
            a, b = np.where(infa, cap, a), np.where(infb, cap, b)
            for xi in x[infa]:
                ax.text(xi + 0.2, cap, "\u221e", ha="center", va="bottom", color=cc._TEXT_COL, fontsize=11)
            for xi in x[infb]:
                ax.text(xi - 0.2, cap, "\u221e", ha="center", va="bottom", color=cc._TEXT_COL, fontsize=11)
        ax.set_facecolor(cc._BG)
        ax.bar(x - 0.2, b, 0.38, color="#9e9e9e", label="before")
        ax.bar(x + 0.2, a, 0.38, color="#2e7d32" if name != "Leakage" else "#c62828", label="after")
        ax.set_xticks(x)
        ax.set_xticklabels(behs, rotation=35, ha="right", color=cc._TICK_COL, fontsize=8)
        ax.tick_params(axis="y", colors=cc._TICK_COL)
        top = max(1.0, float(np.nanmax(np.concatenate([a, b]))) * 1.15)
        ax.set_ylim(0, top)
        for tag_, vals, off in (("before", merged[f"Pool_before"], -0.2), ("after", merged[f"Pool_after"], 0.2)):
            for xi, pool_ in zip(x, vals):
                if not str(pool_).strip():                   # no pure cluster -> say so instead of an empty bar
                    ax.text(xi + off, top * 0.02, "no pure\ncluster", ha="center", va="bottom",
                            color=cc._TICK_COL, fontsize=6.5, rotation=90)
        better = "higher is better" if name != "Leakage" else "lower is better"
        ax.set_title(f"{name} {('(' + unit + ')') if unit else ''}  -  {better}", color=cc._TEXT_COL, fontsize=10)
        for sp in ax.spines.values():
            sp.set_color(cc._TICK_COL)
        ax.legend(frameon=False, labelcolor=cc._TEXT_COL, fontsize=8)
    if note:
        fig.text(0.5, 0.005, note, ha="center", color=cc._TICK_COL, fontsize=8)
    fig.suptitle("Coverage / Leakage / Ratio per behaviour - before vs after your labels",
                 color=cc._TEXT_COL, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cc._savefig(fig, Path(out_path))


def plot_labelled_hierarchy(feats, per_bin_clusters, groups, out_dir, themes=("light", "dark"), linkage="ward"):
    """cluster_hierarchy with the user's behaviour names under every leaf and a bracket joining the clusters
    of one behaviour. Rebuilt from the run's saved features + final per-bin clusters."""
    import matplotlib
    matplotlib.use("Agg")
    import cube_core as cc
    f = np.asarray(feats, dtype=float)
    lab = np.asarray(per_bin_clusters).astype(int).copy()
    ok = np.isfinite(f).all(axis=1) & (lab >= 0)
    if ok.sum() < 20:
        return []
    names = cluster_names(groups)
    outs = []
    for th in themes:
        cc._apply_plot_theme(th)
        p = Path(out_dir) / f"cluster_hierarchy_user_labelled_{th}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        cc.plot_cluster_hierarchy(f[ok], lab[ok], p, linkage_method=linkage, cluster_names=names,
                                  behaviour_groups=groups)
        outs.append(str(p))
    return outs


# ----------------------------------------------------------------- the button
def produce_plots(after_dir, labels_path, before_dir=None, out_dir=None, theme="light", min_bins=5,
                  min_purity=0.7, log=None, progress=None) -> dict:
    """Write the before/after report into ``<after_dir>/plots/guided/``. ``after_dir`` = the CUBE 3 - Clusters
    folder of the (re)clustered run; ``before_dir`` = the blind run's folder (or its before_run snapshot);
    defaults to the snapshot the labelled run kept. Returns {name: path or message}."""
    import matplotlib
    matplotlib.use("Agg")      # runs in a worker thread inside a Tk app: never let matplotlib open its own Tk window
    log = log or (lambda m: None)
    prog = progress or (lambda f, t=None: None)
    after_dir = Path(after_dir)
    res: dict = {}
    prog(0.02, "Reading the run and your labels ...")
    out = Path(out_dir) if out_dir else after_dir / "plots" / GUIDED_PLOTS
    d_umap, d_met, d_hier, d_tab = (out / "01_umap_before_after", out / "02_coverage_leakage_ratio",
                                    out / "03_hierarchy", out / "04_tables")
    for d in (d_umap, d_met, d_hier, d_tab):
        d.mkdir(parents=True, exist_ok=True)
    from cube_guided_session import GuidedData
    ls = cg.load_labels(labels_path)
    gd = GuidedData(after_dir / "model", after_dir / "bout_lengths")
    after_bins = RunLabels(after_dir).per_bin(gd.ranges, gd.fps_map)
    if len(after_bins) != gd.n_bins:
        after_bins = np.pad(after_bins, (0, max(0, gd.n_bins - len(after_bins))), constant_values=-1)[:gd.n_bins]
    y, _w = cg.labels_to_bins(ls, gd.ranges, gd.fps_map, gd.n_bins)
    truth = np.array([None if v in ("", "__neither__") else str(v) for v in y], dtype=object)
    n_lab = int(sum(t is not None and t != "Background" for t in truth))
    res["n_labelled_bins"] = n_lab

    prog(0.2, "Working out which clusters make up each of your behaviours ...")
    # --- improved clusters / behaviour groups for the analyser (always, even without a before run)
    pools_after = pools_by_purity(after_bins, truth, min_bins, min_purity)
    groups = {b: sorted(set(c)) for b, c in pools_after.items() if c}
    res["mapping_tsv"] = write_mapping_tsv(after_dir / "cluster_behaviour_mapping.tsv", after_bins, groups)
    res["behaviour_groups_json"] = write_behaviour_groups_json(
        after_dir / "behaviour_groups.json", groups,
        {"source": "user labels", "labels_sha1": ls.sha1, "min_purity": min_purity, "min_labelled_bins": min_bins})
    res["behaviour_groups"] = groups

    prog(0.3, "Saving the behaviour groups for the Behaviour Analyser ...")
    # --- hierarchy with user names
    prog(0.35, "Drawing the cluster hierarchy with your behaviour names ...")
    try:
        if gd.feats is not None:
            res["hierarchy"] = plot_labelled_hierarchy(gd.feats, after_bins, groups, d_hier)
    except Exception as e:                                  # noqa: BLE001
        res["hierarchy_error"] = str(e)
        log(f"[WARN] labelled hierarchy skipped: {e}")

    # --- before
    bdir = Path(before_dir) if before_dir else after_dir / SNAP_DIR
    if not (bdir / "model" / "session_bin_ranges.json").is_file():
        res["before"] = ("no 'before' run available - pick the blind run's folder, or run CUBE once without "
                         "labels first (a labelled run keeps that earlier result automatically)")
        log(f"[WARN] {res['before']}")
        return res
    prog(0.5, "Reading the earlier (before) clustering ...")
    before_bins = RunLabels(bdir).per_bin(gd.ranges, gd.fps_map)
    before_bins = np.pad(before_bins, (0, max(0, gd.n_bins - len(before_bins))), constant_values=-1)[:gd.n_bins]

    prog(0.55, "Computing Coverage / Leakage / Ratio before vs after ...")
    tb, ta, merged = metric_tables(before_bins, after_bins, truth, min_bins, min_purity)
    merged.to_csv(d_tab / "coverage_leakage_ratio_before_after.csv", index=False)
    tb.to_csv(d_tab / "coverage_leakage_ratio_before.csv", index=False)
    ta.to_csv(d_tab / "coverage_leakage_ratio_after.csv", index=False)
    import cube_guided_eval as ge
    note = ("In-sample: the labels shown here were also given to CUBE. Treat as a diagnostic of how well the "
            "clusters now match YOUR labels, not as held-out accuracy. Evaluated on labelled bins only.")
    md = [f"# Coverage / Leakage / Ratio - before vs after your labels\n", note + "\n",
          "## Before (blind run)\n", ge.format_table_md(tb), "\n\n## After (with your labels)\n",
          ge.format_table_md(ta), "\n\n## Change\n",
          ge.format_table_md(merged[["Behaviour", "Coverage_before", "Coverage_after", "Leakage_before",
                                     "Leakage_after", "Ratio_before", "Ratio_after", "n_clusters_before",
                                     "n_clusters_after"]]), "\n"]
    (d_tab / "coverage_leakage_ratio_before_after.md").write_text("\n".join(md), encoding="utf-8")
    plot_metrics_before_after(merged, d_met / "coverage_leakage_ratio_before_after.png", theme, note)
    res["metrics"] = merged.to_dict("records")

    prog(0.7, "Drawing the before/after UMAPs ...")
    pools_before = pools_by_purity(before_bins, truth, min_bins, min_purity)
    if gd.embedding is not None:
        beh_after = np.array([""] * len(after_bins), dtype=object)
        for b, cl in pools_after.items():
            beh_after[np.isin(after_bins, cl)] = b
        beh_after = np.where(beh_after == "", None, beh_after)
        truth_plot = np.array([None if (t is None or t == "Background") else t for t in truth], dtype=object)
        plot_before_after_umap(gd.embedding, np.where(before_bins >= 0, before_bins, None).astype(object),
                               np.where(after_bins >= 0, after_bins, None).astype(object),
                               beh_after, truth_plot, ls, d_umap / "umap_before_after_shared.png", theme)
        plot_behaviour_panels(gd.embedding, before_bins, after_bins, truth, pools_before, pools_after, merged, ls,
                              d_umap / "umap_behaviour_pools_before_after.png", theme)
        prog(0.92, "Drawing the cluster-flow diagram ...")
        try:
            import cube_core as cc
            cc.plot_condition_comparison_sankey(before_bins[(before_bins >= 0) & (after_bins >= 0)],
                                                after_bins[(before_bins >= 0) & (after_bins >= 0)],
                                                d_umap / "cluster_flow_before_to_after.html",
                                                "Before", "After (with your labels)", theme=theme)
        except Exception as e:                              # noqa: BLE001
            log(f"[WARN] sankey skipped: {e}")
    else:
        res["umap"] = "no UMAP embedding saved for the after run - UMAP panels skipped"
    prog(1.0, "Done")
    res["out_dir"] = str(out)
    return res
