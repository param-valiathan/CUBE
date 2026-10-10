"""What the user's labels say about HOW LONG behaviours last and WHAT FOLLOWS WHAT (guided labelling).

User labels are ground truth for more than "which bins are behaviour X": the length of a labelled bout and the
order of adjacent labelled bouts are real observations of the behaviour's dynamics. This module turns them into

  * per-behaviour bout-duration statistics (only trusted at >= ``min_bouts`` bouts, default 5),
  * observed behaviour->behaviour transitions and the boundaries where they happen,
  * an HMM transition-matrix PRIOR (``guided_transmat``): self-transition pulled toward the labelled median bout
    length, observed transitions made more likely, nothing unobserved made less likely than the base prior,
  * an audit comparing the final bout lengths / boundaries with the labelled ones (``bout_length_audit``,
    ``boundary_agreement``) -- written to model/guided_dynamics.json so a mismatch is visible, not silent.

Pure functions, no engine imports. The engine calls them only when a labels file is attached and
``guided_dynamics_enabled`` is on; with the switch off (or no labels) nothing here runs.

A behaviour may consist of several clusters. A cluster bout can therefore be SHORTER than the behaviour's labelled
bout, never longer: a behaviour mapped to one cluster gets its self-transition blended toward the labelled value,
a behaviour spread over several clusters only has its clusters' self-transition CAPPED at that value.
"""
from __future__ import annotations

import numpy as np

NEITHER_MARK = "__neither__"
_SKIP = ("", "background", "neither", NEITHER_MARK)


def _norm(v) -> str:
    return str(v).strip().casefold()


def _is_behaviour(lab: dict) -> bool:
    name = _norm(lab.get("behaviour", ""))
    ans = lab.get("answer")
    if name in _SKIP or (isinstance(ans, str) and _norm(ans) == "neither"):
        return False
    return True


def derive_dynamics(ls, fps_map, min_bouts: int = 5, merge_gap_s: float = 0.25, trans_gap_s: float = 1.0) -> dict:
    """Bout statistics and transitions from a LabelSet (time-based labels).

    Consecutive same-behaviour segments closer than ``merge_gap_s`` are one bout. Two different behaviours
    count as a transition only when the second starts within ``trans_gap_s`` of the first's end (anything in
    between, e.g. Background, breaks adjacency). Returns
      behaviours: {norm_name: {name, n_bouts, median_s, p10_s, p90_s, reliable}}
      transitions: {(a, b): count}, boundaries: [(session, t_s, a, b)]
    """
    by_sess: dict = {}
    for lab in ls.labels:
        if not _is_behaviour(lab) or lab.get("source") == "query":     # query clips are fixed ~1 s windows, not bouts
            continue
        try:
            by_sess.setdefault(lab["session"], []).append(
                (float(lab["start_s"]), float(lab["end_s"]), _norm(lab["behaviour"]), str(lab["behaviour"]).strip()))
        except (KeyError, TypeError, ValueError):
            continue
    names: dict = {}
    bouts: dict = {}                     # norm -> [duration_s]
    trans: dict = {}
    boundaries: list = []
    for sess, segs in by_sess.items():
        segs.sort()
        merged: list = []
        for s0, s1, nb, nm in segs:
            names.setdefault(nb, nm)
            if merged and merged[-1][2] == nb and s0 - merged[-1][1] <= merge_gap_s:
                merged[-1][1] = max(merged[-1][1], s1)
            else:
                merged.append([s0, s1, nb])
        for i, (s0, s1, nb) in enumerate(merged):
            bouts.setdefault(nb, []).append(s1 - s0)
            if i + 1 < len(merged):
                n0, n1, nn = merged[i + 1]
                if nn != nb and 0.0 <= n0 - s1 <= trans_gap_s:
                    trans[(nb, nn)] = trans.get((nb, nn), 0) + 1
                    boundaries.append((sess, 0.5 * (s1 + n0), nb, nn))
    behaviours = {}
    for nb, durs in bouts.items():
        d = np.asarray(durs, dtype=float)
        behaviours[nb] = {"name": names.get(nb, nb), "n_bouts": int(d.size), "median_s": float(np.median(d)),
                          "p10_s": float(np.percentile(d, 10)), "p90_s": float(np.percentile(d, 90)),
                          "reliable": bool(d.size >= int(min_bouts))}
    return {"behaviours": behaviours, "transitions": trans, "boundaries": boundaries, "min_bouts": int(min_bouts)}


def _p_self_from_bins(median_bins: float) -> float:
    return float(np.clip(1.0 - 1.0 / max(float(median_bins), 1.0001), 0.5, 0.99))


def guided_transmat(dyn: dict, cluster_behaviour: dict, n_clusters: int, bin_s: float, steps_per_bin: float = 1.0,
                    base: np.ndarray | None = None, blend: float = 0.5, trans_boost: float = 1.0,
                    min_trans: int = 3):
    """HMM transition-matrix prior informed by the labels. ``cluster_behaviour`` = {cluster id: normalised
    behaviour name} for clusters with labelled evidence only. ``bin_s`` = seconds per classified bin,
    ``steps_per_bin`` = HMM time steps per bin (1 for bin-level HMMs, frames-per-bin for frame-level).
    Returns (matrix, info). Returns (base, info) unchanged when the labels support nothing."""
    n = int(n_clusters)
    if base is None:
        base = np.eye(n) * 0.9 + 0.1 / n
        base = base / base.sum(axis=1, keepdims=True)
    T = np.array(base, dtype=float, copy=True)
    info = {"self_adjusted": {}, "transitions_boosted": 0, "skipped": []}
    beh = dyn.get("behaviours", {})
    clusters_of: dict = {}
    for c, b in cluster_behaviour.items():
        if 0 <= int(c) < n:
            clusters_of.setdefault(b, []).append(int(c))
    steps_per_s = float(steps_per_bin) / max(float(bin_s), 1e-9)
    p_new = np.diag(T).copy()
    for b, cl in clusters_of.items():
        st = beh.get(b)
        if st is None or not st["reliable"]:
            info["skipped"].append({"behaviour": b, "why": "fewer than min_bouts labelled bouts"})
            continue
        p_lab = _p_self_from_bins(st["median_s"] * steps_per_s)
        for c in cl:
            p0 = p_new[c]
            p1 = (1.0 - blend) * p0 + blend * p_lab if len(cl) == 1 else min(p0, p_lab)
            p_new[c] = float(np.clip(p1, 0.5, 0.99))
            info["self_adjusted"][int(c)] = {"behaviour": st["name"], "from": round(float(p0), 4),
                                            "to": round(float(p_new[c]), 4), "labelled_median_s": round(st["median_s"], 3)}
    # behaviour-level persistence: a behaviour made of several clusters lasts as long as the user's labelled bouts,
    # however its clusters alternate. So the probability of LEAVING the behaviour (to any cluster outside it) is
    # pulled toward 1 - p_label (by `blend`), and the mass freed stays inside the behaviour (own cluster + siblings,
    # in proportion to what they already had). Switching between a behaviour's own clusters stays free.
    for i in range(n):                          # first apply the per-cluster self-transition adjustments above
        off_ = T[i].copy()
        off_[i] = 0.0
        s_ = off_.sum()
        if s_ > 0:
            T[i] = off_ * ((1.0 - p_new[i]) / s_)
        T[i, i] = p_new[i]
    P = np.array(T, dtype=float)
    for b, cl in clusters_of.items():
        st = beh.get(b)
        if st is None or not st["reliable"] or len(cl) < 2:
            continue
        q_lab = 1.0 - _p_self_from_bins(st["median_s"] * steps_per_s)
        inside = np.zeros(n, dtype=bool)
        inside[cl] = True
        for i in cl:
            leave = float(P[i, ~inside].sum())
            target = (1.0 - blend) * leave + blend * q_lab
            if leave > target > 0:
                freed = leave - target
                P[i, ~inside] *= target / leave
                P[i, inside] += freed * P[i, inside] / max(P[i, inside].sum(), 1e-12)
                info.setdefault("leave_adjusted", {})[int(i)] = {"behaviour": st["name"], "leave_from": round(leave, 4),
                                                                  "leave_to": round(target, 4)}
    T = P
    p_new = np.diag(T).copy()
    # observed transitions: raise the off-diagonal weight between the clusters of A and of B
    tr = {k: v for k, v in dyn.get("transitions", {}).items() if v >= min_trans}
    tmax = max(tr.values()) if tr else 0
    W = np.ones((n, n))
    for (a, b), cnt in tr.items():
        for i in clusters_of.get(a, []):
            for j in clusters_of.get(b, []):
                if i != j:
                    W[i, j] += trans_boost * cnt / tmax
                    info["transitions_boosted"] += 1
    for i in range(n):
        off = T[i].copy()
        off[i] = 0.0
        off = off * W[i]
        s = off.sum()
        if s > 0:
            T[i] = off * ((1.0 - p_new[i]) / s)
        T[i, i] = p_new[i]
        T[i] /= T[i].sum()
    return T, info


def _bouts(arr: np.ndarray):
    arr = np.asarray(arr)
    if arr.size == 0:
        return []
    ch = np.flatnonzero(np.diff(arr) != 0)
    st = np.concatenate(([0], ch + 1))
    en = np.concatenate((ch, [arr.size - 1]))
    return [(int(arr[s]), int(e - s + 1)) for s, e in zip(st, en)]


def bout_length_audit(dyn: dict, cluster_behaviour: dict, label_sequences: list, step_s: float) -> dict:
    """Median final bout length per behaviour (clusters of one behaviour fused) vs the labelled median.
    ``label_sequences`` = per-session final label arrays at ``step_s`` seconds per element."""
    cmap = {int(c): b for c, b in cluster_behaviour.items()}
    got: dict = {}
    for seq in label_sequences:
        seq = np.asarray(seq).astype(int)
        mapped = np.array([hash(cmap[v]) if v in cmap else -1 for v in seq], dtype=np.int64)
        rev = {hash(b): b for b in cmap.values()}
        for code, ln in _bouts(mapped):
            if code in rev:
                got.setdefault(rev[code], []).append(ln * step_s)
    out = {}
    for b, st in dyn.get("behaviours", {}).items():
        d = got.get(b)
        if not d:
            continue
        out[st["name"]] = {"labelled_median_s": round(st["median_s"], 3), "labelled_bouts": st["n_bouts"],
                           "predicted_median_s": round(float(np.median(d)), 3), "predicted_bouts": len(d),
                           "ratio_pred_over_labelled": round(float(np.median(d)) / max(st["median_s"], 1e-9), 2),
                           "reliable": st["reliable"]}
    return out


def boundary_agreement(boundaries_bins: list, label_sequences_by_session: dict, feats_speed_by_session: dict,
                       tol_bins: int = 2) -> dict:
    """How well final labels and the feature-space 'cliffs' line up with the user's labelled behaviour changes.
    boundaries_bins = [(session, bin_in_session)]; label_sequences_by_session = {session: per-bin final labels};
    feats_speed_by_session = {session: per-bin feature-space speed}. A boundary 'has a label change' when the
    final label differs between bins b-tol and b+tol; its 'cliff ratio' is the max speed within +/-tol over the
    session's median speed."""
    n = hit = cliff_n = 0
    ratios: list = []
    for sess, b in boundaries_bins:
        lab = label_sequences_by_session.get(sess)
        sp = feats_speed_by_session.get(sess)
        if lab is None or sp is None:
            continue
        lo, hi = max(0, b - tol_bins), min(len(lab) - 1, b + tol_bins)
        if hi <= lo:
            continue
        n += 1
        hit += int(lab[lo] != lab[hi])
        med = float(np.nanmedian(sp)) if np.isfinite(sp).any() else 0.0
        if med > 0:
            r = float(np.nanmax(sp[lo:hi + 1]) / med)
            ratios.append(r)
            cliff_n += int(r >= 1.5)
    return {"n_boundaries": n, "label_change_at_boundary": hit,
            "label_change_frac": round(hit / n, 3) if n else None,
            "cliff_ratio_median": round(float(np.median(ratios)), 2) if ratios else None,
            "cliff_frac_ge_1p5": round(cliff_n / len(ratios), 3) if ratios else None}


def feature_speed(feats: np.ndarray) -> np.ndarray:
    """Per-bin feature-space speed |x[t+1]-x[t-1]| of a (n_feat, n_bins) matrix, z-scored per feature first."""
    X = np.asarray(feats, dtype=float)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Z = (X - np.nanmean(X, axis=1, keepdims=True)) / sd
    Z = np.nan_to_num(Z, nan=0.0)
    sp = np.zeros(Z.shape[1])
    if Z.shape[1] > 2:
        sp[1:-1] = np.linalg.norm(Z[:, 2:] - Z[:, :-2], axis=0)
        sp[0], sp[-1] = sp[1], sp[-2]
    return sp


# ----------------------------------------------------------------------------- engine-facing helpers
def win100_for(fps) -> int:
    return max(1, int(round(float(fps) / 10.0)))


def engine_prior(ctx, ls, hdb_labels, n_clusters: int, fps_list, bin_level: bool, cfg: dict):
    """Everything the engine needs for the HMM hook in one call. Returns (matrix_or_None, report_dict).
    Never raises; None means 'use the ordinary prior'."""
    try:
        dyn = derive_dynamics(ls, None, min_bouts=int(cfg.get("guided_min_bouts", 5)))
        cl_beh = ctx.dominant(np.asarray(hdb_labels))
        wins = [win100_for(f) for f in fps_list]
        bin_s = float(np.mean([w / float(f) for w, f in zip(wins, fps_list)]))
        steps = 1.0 if bin_level else float(np.mean(wins))
        blend = float(cfg.get("guided_dynamics_blend", 0.5))
        boost = float(cfg.get("guided_transition_boost", 1.0))
        T, info = guided_transmat(dyn, cl_beh, int(n_clusters), bin_s, steps, blend=blend, trans_boost=boost)
        rep = {"dynamics": {k: v for k, v in dyn.items() if k != "boundaries"},
               "prior": info, "cluster_behaviour": {int(c): b for c, b in cl_beh.items()}}
        rep["dynamics"]["transitions"] = {f"{a}->{b}": n for (a, b), n in dyn["transitions"].items()}
        used = (bool(info["self_adjusted"]) or info["transitions_boosted"] > 0 or bool(info.get("leave_adjusted"))
                or any(len(c) > 1 for c in {b: [k for k, v in cl_beh.items() if v == b] for b in set(cl_beh.values())}.values()))
        applied: dict = {}

        def adjust(base):
            # layered on whatever base prior the HMM built (per-cluster or flat), not on a fresh flat one
            M, inf2 = guided_transmat(dyn, cl_beh, int(n_clusters), bin_s, steps, base=np.asarray(base, float),
                                      blend=blend, trans_boost=boost)
            applied.update(inf2)
            rep["prior"] = inf2
            return M
        return (adjust if used else None), {"dyn": dyn, "report": rep, "cluster_behaviour": cl_beh}
    except Exception as e:                                             # noqa: BLE001
        return None, {"error": str(e)}


def engine_audit(pack: dict, hmm_frame_labels: list, fps_list: list, names: list, ranges: dict,
                 feats_cat: np.ndarray) -> dict:
    """Post-run audit (bout lengths + boundary / cliff agreement) appended to the prior report."""
    rep = dict(pack.get("report", {}))
    dyn, cl_beh = pack["dyn"], pack["cluster_behaviour"]
    wins = [win100_for(f) for f in fps_list]
    seqs = [np.asarray(l)[::w] for l, w in zip(hmm_frame_labels, wins)]
    step_s = float(np.mean([w / float(f) for w, f in zip(wins, fps_list)]))
    rep["bout_length_audit"] = bout_length_audit(dyn, cl_beh, seqs, step_s)
    bins, labs, spd = [], {}, {}
    fps_of = dict(zip(names, fps_list))
    win_of = dict(zip(names, wins))
    for nm, seq in zip(names, seqs):
        labs[nm] = seq
        r = ranges.get(nm)
        if r is not None:
            spd[nm] = feature_speed(feats_cat[:, int(r[0]):int(r[1])])
    for sess, t, _a, _b in dyn["boundaries"]:
        if sess in fps_of:
            bins.append((sess, int(t * fps_of[sess] / win_of[sess])))
    rep["boundary_agreement"] = boundary_agreement(bins, labs, spd)
    return rep
