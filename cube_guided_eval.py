"""Guided-labelling evaluation helpers (pure numpy/pandas; no Tk, no cube_core).

Conventions (taken from the validated tables in
BSOID_IMPROVEMENT_BASELINE_AND_EXPERIMENT_PLAN.md / BSOID_STANDALONE_IMPROVEMENT_PLAN.md,
which CROSS_CLUSTER_RECLAIM_RUNS\\epm_manual_crosswalk.py feeds via its
time-weighted contingency of (cluster, manual_label) -> seconds):

  For a truth behaviour B and a "pred group" G (one pred label, or a pool of
  pred labels when ``combine=True``), over bins that carry a truth label:
    Coverage = weight(bins with truth==B and pred in G) / weight(bins with truth==B)
    Leakage  = weight(bins with pred in G and truth!=B) / weight(bins with pred in G)
    Ratio    = Coverage / Leakage   (0 if Coverage==0, inf if Leakage==0 and
               Coverage>0).  NOTE: this is Coverage/Leakage, which reproduces the
               documented numbers (e.g. 5.24/66.59 = 0.0787); it is NOT
               Coverage/(Coverage+Leakage) as the contract text suggests.
  Coverage and Leakage are reported in percent, as in the reference tables.

DEFAULT METHOD (user standing rule): "purity_gated" -- only high-purity clusters (purity
among labelled bins >= purity_floor, default 0.7, >= min_labeled bins), each cluster
assigned to at most one behaviour (no reuse across behaviours); see _purity_gated_table.
The F1 matching below is the legacy method="f1".

BOUT-LEVEL METRICS (``bout_metrics``, added alongside the frame-level table, not replacing
it): segmental F1@IoU (0.10/0.25/0.50), predicted/true bout-count ratio + median bout
duration (fragmentation), and normalised segment-sequence edit score, following the
action-segmentation convention (Lea et al. 2017; MS-TCN; also reported by DLC2Action,
Kozlova et al. 2025). Frame-level purity cannot see a behaviour shattered into many
short bouts; these can. ``map_clusters_to_behaviours`` applies the same purity-gated
assignment so cluster output and direct-classifier output are scored identically.

Matching a pred group to a truth behaviour (method="f1"):
  * Single (combine=False): the single pred label with the highest F1 between
    recall (Coverage) and precision (1-Leakage).
  * Combined (combine=True): start from that best single label, then greedily
    add further pred labels one at a time while the pool's F1 strictly
    improves (many-to-one: one behaviour split across several clusters).
  The pool is reported in the ``Pool`` column.  Pred labels that are
  unlabelled ("" / NaN / None) never join a pool but their bins still count in
  the truth total (so they lower Coverage).

Dropping rules: truth that is unlabelled ("" / NaN / None) or in ``background``
is dropped entirely (those bins are not evaluated, and are not rows).  Labels
may be strings or ints; they are compared as strings.  Optional per-bin
``weights`` (e.g. seconds per bin) are honoured everywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MISSING = object()


def _norm(arr) -> np.ndarray:
    """Normalise labels to an object array of str, with None for unlabelled."""
    a = np.asarray(arr, dtype=object).ravel()
    out = np.empty(len(a), dtype=object)
    for i, v in enumerate(a):
        if v is None:
            out[i] = None
            continue
        if isinstance(v, float):
            if np.isnan(v):
                out[i] = None
                continue
            if v == int(v):
                v = int(v)
        if isinstance(v, (np.floating,)):
            if np.isnan(v):
                out[i] = None
                continue
            if v == int(v):
                v = int(v)
        s = str(v).strip()
        out[i] = s if s != "" and s.lower() not in ("nan", "none") else None
    return out


def _f1(cov: float, leak: float) -> float:
    prec = 1.0 - leak
    return 0.0 if cov + prec <= 0 else 2 * cov * prec / (cov + prec)


def _stats(in_group: np.ndarray, is_b: np.ndarray, w: np.ndarray):
    tot_b = w[is_b].sum()
    tot_g = w[in_group].sum()
    hit = w[in_group & is_b].sum()
    cov = hit / tot_b if tot_b > 0 else 0.0
    leak = (tot_g - hit) / tot_g if tot_g > 0 else float("nan")
    return cov, leak, tot_g


def _ratio(cov: float, leak: float) -> float:
    if cov <= 0 or leak != leak:
        return 0.0
    return float("inf") if leak == 0 else cov / leak


def _purity_gated_assign(p, t, w, bg, purity_floor, min_labeled):
    """Cluster -> behaviour assignment shared by the Coverage/Leakage table and the
    bout-level metrics. Returns (behaviours, clusters, b_tot, assign)."""
    labelled = np.array([x is not None and x not in bg for x in t])
    behaviours = sorted({x for x in t[labelled]})
    clusters = sorted({x for x in p if x is not None})
    b_tot = {b: w[t == b].sum() for b in behaviours}
    assign = {}
    for c in clusters:
        inc = p == c
        ov = {b: w[inc & (t == b)].sum() for b in behaviours}
        n_lab = sum(ov.values())
        if not behaviours or n_lab <= 0:
            continue
        best = max(behaviours, key=lambda b: ov[b] / b_tot[b] if b_tot[b] else 0.0)
        if ov[best] <= 0:
            continue
        if n_lab >= min_labeled and ov[best] / n_lab >= purity_floor:
            assign[c] = best
    return behaviours, clusters, b_tot, assign


def _purity_gated_table(p, t, w, bg, purity_floor, min_labeled) -> pd.DataFrame:
    """Repo convention (exp26_bsoid_purity_gated.py): each pred cluster is pooled
    into AT MOST ONE behaviour -- the one with the largest share of that behaviour's
    total -- and only if its purity among labelled bins (non-background truth) is
    >= purity_floor and it holds >= min_labeled labelled bins.  No reuse of a
    cluster by two behaviours.  Leakage = share of the pool's ALL bins (incl.
    background/unlabelled truth) that are not the behaviour.  Ratio = Cov/Leak,
    0 when the pool is empty or Leakage is 0/NaN (as in the validated script)."""
    cols = ["Behaviour", "n_bins", "Weight", "Coverage", "Leakage", "Ratio", "Pool"]
    behaviours, clusters, b_tot, assign = _purity_gated_assign(p, t, w, bg, purity_floor, min_labeled)
    rows = []
    for b in behaviours:
        pool = [c for c in clusters if assign.get(c) == b]
        inpool = np.isin(p, pool) if pool else np.zeros(len(p), bool)
        tot, hit = w[inpool].sum(), w[inpool & (t == b)].sum()
        cov = hit / b_tot[b] if (pool and b_tot[b]) else 0.0
        leak = (tot - hit) / tot if (pool and tot) else float("nan")
        ratio = cov / leak if (pool and leak == leak and leak > 0) else 0.0
        rows.append([b, int((t == b).sum()), float(b_tot[b]), 100 * cov,
                     100 * leak if leak == leak else float("nan"), ratio,
                     ",".join(pool)])
    return pd.DataFrame(rows, columns=cols)


def behaviour_table(pred, truth, weights=None, background=("Background", "", "__neither__"),
                    combine: bool = False, method: str = "purity_gated",
                    purity_floor: float = 0.7, min_labeled: int = 5) -> pd.DataFrame:
    """One row per truth behaviour: full Coverage/Leakage/Ratio table.

    method="purity_gated" (default, the repo's validated convention): high-purity
    clusters only, each cluster used by at most one behaviour (no reuse).
    method="f1": legacy best-F1 matching (clusters may be reused; ``combine``
    applies only here)."""
    p = _norm(pred)
    t = _norm(truth)
    if len(p) != len(t):
        raise ValueError("pred and truth must have the same length")
    w = np.ones(len(t)) if weights is None else np.asarray(weights, float).ravel()
    if len(w) != len(t):
        raise ValueError("weights must match length of truth")
    bg = {str(b) for b in background}
    if method == "purity_gated":
        return _purity_gated_table(p, t, w, bg, purity_floor, min_labeled)
    if method != "f1":
        raise ValueError("method must be 'purity_gated' or 'f1'")
    keep = np.array([x is not None and x not in bg for x in t])
    p, t, w = p[keep], t[keep], w[keep]
    cols = ["Behaviour", "n_bins", "Weight", "Coverage", "Leakage", "Ratio", "Pool"]
    behaviours = sorted(set(t))
    pred_labels = sorted({x for x in p if x is not None})
    rows = []
    for b in behaviours:
        is_b = t == b
        best, best_f = None, -1.0
        for g in pred_labels:
            cov, leak, tg = _stats(p == g, is_b, w)
            if tg <= 0:
                continue
            f = _f1(cov, leak)
            if f > best_f + 1e-12:
                best, best_f = g, f
        if best is None:
            rows.append([b, int(is_b.sum()), float(w[is_b].sum()), 0.0, float("nan"), 0.0, ""])
            continue
        pool = [best]
        if combine:
            while True:
                cur = np.isin(p, pool)
                cov, leak, _ = _stats(cur, is_b, w)
                cur_f = _f1(cov, leak)
                cand, cand_f = None, cur_f
                for g in pred_labels:
                    if g in pool:
                        continue
                    c2, l2, _ = _stats(np.isin(p, pool + [g]), is_b, w)
                    f2 = _f1(c2, l2)
                    if f2 > cand_f + 1e-12:
                        cand, cand_f = g, f2
                if cand is None:
                    break
                pool.append(cand)
        cov, leak, _ = _stats(np.isin(p, pool), is_b, w)
        rows.append([b, int(is_b.sum()), float(w[is_b].sum()), 100 * cov,
                     100 * leak if leak == leak else float("nan"),
                     _ratio(cov, leak), ",".join(pool)])
    return pd.DataFrame(rows, columns=cols)


def per_session_f1(pred, truth, session_of_bin, background=("Background", "", "__neither__")) -> pd.DataFrame:
    """Macro F1 per session over non-background truth behaviours present in
    that session. Bins with unlabelled truth are dropped; background-truth
    bins are kept (a predicted behaviour there is a false positive)."""
    p = _norm(pred)
    t = _norm(truth)
    s = np.asarray(session_of_bin, dtype=object).ravel()
    bg = {str(b) for b in background}
    rows = []
    for sess in list(dict.fromkeys(s.tolist())):
        m = (s == sess) & np.array([x is not None for x in t])
        if not m.any():
            continue
        ps, ts = p[m], t[m]
        classes = sorted({x for x in ts if x not in bg})
        f1s = []
        for c in classes:
            tp = np.sum((ps == c) & (ts == c))
            fp = np.sum((ps == c) & (ts != c))
            fn = np.sum((ps != c) & (ts == c))
            d = 2 * tp + fp + fn
            f1s.append(2 * tp / d if d > 0 else 0.0)
        rows.append([sess, float(np.mean(f1s)) if f1s else float("nan"),
                     len(classes), int(m.sum())])
    return pd.DataFrame(rows, columns=["session", "macro_f1", "n_behaviours", "n_bins"])


def purity(cluster_labels, truth) -> pd.DataFrame:
    """Per-cluster dominant truth behaviour and purity among labelled bins."""
    c = _norm(cluster_labels)
    t = _norm(truth)
    if len(c) != len(t):
        raise ValueError("cluster_labels and truth must have the same length")
    rows = []
    for g in sorted({x for x in c if x is not None}):
        m = c == g
        n = int(m.sum())
        lab = [x for x in t[m] if x is not None]
        if lab:
            vc = pd.Series(lab).value_counts()
            dom, top = str(vc.index[0]), int(vc.iloc[0])
            rows.append([g, n, len(lab), dom, top / len(lab)])
        else:
            rows.append([g, n, 0, "", float("nan")])
    return pd.DataFrame(rows, columns=["cluster", "n_bins", "n_labelled",
                                       "dominant_behaviour", "purity"])


def map_clusters_to_behaviours(pred, truth, weights=None,
                               background=("Background", "", "__neither__"),
                               purity_floor: float = 0.7, min_labeled: int = 5,
                               background_label: str = "Background") -> np.ndarray:
    """Relabel a cluster sequence into behaviour space using the same purity-gated,
    no-reuse assignment as ``behaviour_table``. Clusters not assigned to any behaviour
    (and unlabelled preds) become ``background_label``. Output aligns bin-for-bin with
    ``pred`` so it can be fed straight to ``bout_metrics``."""
    p, t = _norm(pred), _norm(truth)
    if len(p) != len(t):
        raise ValueError("pred and truth must have the same length")
    w = np.ones(len(t)) if weights is None else np.asarray(weights, float).ravel()
    bg = {str(b) for b in background}
    _, _, _, assign = _purity_gated_assign(p, t, w, bg, purity_floor, min_labeled)
    out = np.empty(len(p), dtype=object)
    for i, c in enumerate(p):
        out[i] = assign.get(c, background_label) if c is not None else background_label
    return out


def _segments(seq) -> list:
    """Run-length encode an object array into (label, start, end_exclusive); None
    bins break runs and are not emitted as segments."""
    segs, i, n = [], 0, len(seq)
    while i < n:
        lab, j = seq[i], i + 1
        while j < n and seq[j] == lab:
            j += 1
        if lab is not None:
            segs.append((lab, i, j))
        i = j
    return segs


def _segment_f1_counts(pred_segs, truth_segs, tau: float):
    """(TP, FP, FN) at IoU threshold ``tau`` for ONE class: each pred segment is
    matched to its best-IoU truth segment, which can be claimed only once
    (MS-TCN / Lea et al. segmental-F1 convention)."""
    if not truth_segs:
        return 0, len(pred_segs), 0
    ts = np.array([[s, e] for _, s, e in truth_segs], float)
    hit = np.zeros(len(ts), bool)
    tp = fp = 0
    for _, ps, pe in pred_segs:
        inter = np.clip(np.minimum(pe, ts[:, 1]) - np.maximum(ps, ts[:, 0]), 0, None)
        union = (pe - ps) + (ts[:, 1] - ts[:, 0]) - inter
        iou = inter / union
        k = int(np.argmax(iou))
        if iou[k] >= tau and not hit[k]:
            hit[k] = True
            tp += 1
        else:
            fp += 1
    return tp, fp, len(ts) - int(hit.sum())


def _levenshtein(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def bout_metrics(pred_behaviour, truth, session_of_bin=None, bin_s: float | None = None,
                 background=("Background", "", "__neither__"),
                 iou_thresholds=(0.1, 0.25, 0.5),
                 background_label: str = "Background"):
    """Bout-level (segment) metrics, complementing the frame-level Coverage/Leakage/Ratio.

    ``pred_behaviour`` must already be in BEHAVIOUR space (a direct classifier's output,
    or clusters passed through ``map_clusters_to_behaviours``). Bins are assumed to be in
    time order within each session; runs never span a session boundary. Truth bins that
    are unlabelled are dropped from both sequences (they also break runs); truth in
    ``background`` and pred that is None are both treated as ``background_label``.

    Returns (per_behaviour, per_session):
      per_behaviour: one row per non-background truth behaviour --
        n_truth_bouts, n_pred_bouts, FragRatio (pred/truth bouts: >1 = fragmented,
        <1 = merged), median_truth_dur / median_pred_dur (seconds if ``bin_s`` given,
        else bins), and segmental F1@tau (%) for each tau, TP/FP/FN pooled over sessions.
      per_session: Edit score (%) = (1 - Levenshtein / max(len)) over the ordered
        sequences of segment labels (penalises over-segmentation, missed bouts and
        order errors); last row 'ALL (mean)'.
    """
    p, t = _norm(pred_behaviour), _norm(truth)
    if len(p) != len(t):
        raise ValueError("pred_behaviour and truth must have the same length")
    s = (np.zeros(len(t), dtype=object) if session_of_bin is None
         else np.asarray(session_of_bin, dtype=object).ravel())
    if len(s) != len(t):
        raise ValueError("session_of_bin must match length of truth")
    bg = {str(b) for b in background}
    scale = 1.0 if bin_s is None else float(bin_s)

    labelled = np.array([x is not None for x in t])
    t = np.array([None if x is None else (background_label if x in bg else x) for x in t], dtype=object)
    p = np.array([background_label if (x is None and lab) else (x if lab else None)
                  for x, lab in zip(p, labelled)], dtype=object)
    p = np.array([background_label if (x is not None and x in bg) else x for x in p], dtype=object)

    behaviours = sorted({x for x in t if x is not None and x != background_label})
    pooled = {b: {tau: [0, 0, 0] for tau in iou_thresholds} for b in behaviours}
    tdur = {b: [] for b in behaviours}
    pdur = {b: [] for b in behaviours}
    sess_rows = []
    for sess in list(dict.fromkeys(s.tolist())):
        m = s == sess
        tseg, pseg = _segments(t[m]), _segments(p[m])
        if not tseg:
            continue
        for b in behaviours:
            ts_b = [x for x in tseg if x[0] == b]
            ps_b = [x for x in pseg if x[0] == b]
            tdur[b] += [(e - st) * scale for _, st, e in ts_b]
            pdur[b] += [(e - st) * scale for _, st, e in ps_b]
            for tau in iou_thresholds:
                tp, fp, fn = _segment_f1_counts(ps_b, ts_b, tau)
                acc = pooled[b][tau]
                acc[0] += tp
                acc[1] += fp
                acc[2] += fn
        a, b_ = [x[0] for x in pseg], [x[0] for x in tseg]
        edit = 100.0 * (1.0 - _levenshtein(a, b_) / max(len(a), len(b_), 1))
        sess_rows.append([sess, edit, len(tseg), len(pseg)])

    cols = (["Behaviour", "n_truth_bouts", "n_pred_bouts", "FragRatio",
             "median_truth_dur", "median_pred_dur"]
            + [f"F1@{int(round(100 * tau))}" for tau in iou_thresholds])
    rows = []
    for b in behaviours:
        nt, npd = len(tdur[b]), len(pdur[b])
        f1s = []
        for tau in iou_thresholds:
            tp, fp, fn = pooled[b][tau]
            d = 2 * tp + fp + fn
            f1s.append(100.0 * 2 * tp / d if d > 0 else float("nan"))
        rows.append([b, nt, npd, npd / nt if nt else float("nan"),
                     float(np.median(tdur[b])) if nt else float("nan"),
                     float(np.median(pdur[b])) if npd else float("nan")] + f1s)
    per_beh = pd.DataFrame(rows, columns=cols)
    per_sess = pd.DataFrame(sess_rows, columns=["session", "edit_score", "n_truth_segments", "n_pred_segments"])
    if len(per_sess):
        per_sess.loc[len(per_sess)] = ["ALL (mean)", per_sess.edit_score.mean(),
                                       int(per_sess.n_truth_segments.sum()),
                                       int(per_sess.n_pred_segments.sum())]
    return per_beh, per_sess


def format_table_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (floats to 4 sig. figs)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if v != v:
                return "n/a"
            if np.isinf(v):
                return "inf"
            return f"{v:.4g}"
        return str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)

